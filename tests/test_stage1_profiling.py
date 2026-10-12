"""Profiling on cached codes retains row order, missing values and privacy hashes."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from datanonymous.config import Settings
from datanonymous.detect import dates, numeric, text
from datanonymous.detect.ids import id_text, normalize_id
from datanonymous.generate import ids
from datanonymous.model import ColumnProfile, ColumnType, RawColumn, RawTable
from datanonymous.privacy import matches
from datanonymous.profile import entity, stats, structure
from datanonymous.profile.codes import codes_of


def _arr(values):
    out = np.asarray(values, dtype=object)
    out.flags.writeable = False
    return out


def _legacy_entities(values):
    named, blanks = {}, []
    out = np.empty(len(values), dtype=np.int64)
    for row, value in enumerate(values):
        key = normalize_id(value) if isinstance(value, str) and id_text(value) else None
        if key is None:
            blanks.append(row)
        else:
            out[row] = named.setdefault(key, len(named))
    for offset, row in enumerate(blanks, len(named)):
        out[row] = offset
    return out, len(named), len(blanks)


@pytest.mark.parametrize("seed", range(15))
def test_entity_codes_retain_normalization_first_appearance_and_blank_singletons(seed):
    pool = _arr([None, "", "  ", "-", "?", "00123", "123", "123.0", "A-12", "a12", "007", "7"])
    values = _arr(np.random.default_rng(seed).choice(pool, size=500))
    actual, n_named, n_blank = entity.entity_index(values)
    expected, expected_named, expected_blank = _legacy_entities(values)
    np.testing.assert_array_equal(actual, expected)
    assert (n_named, n_blank) == (expected_named, expected_blank)
    assert entity.entity_keys(values) == [normalize_id(v) if isinstance(v, str) and id_text(v) else None
                                          for v in values]


@pytest.mark.parametrize("changing", [0, 1, 2, 100])
def test_entity_constancy_retains_exact_99_percent_boundary(changing):
    raw_id = [str(7000 + i) for i in range(100) for _ in range(2)] + ["", None, "-"]
    raw_value = [None if i % 3 == 0 else f"level{i % 5}" for i in range(100) for _ in range(2)]
    for i in range(changing):
        raw_value[2 * i + 1] = "changed"
    raw_value += ["blank row", "native null row", "placeholder row"]
    table = RawTable(None, ["student_id", "label"],
                     [RawColumn("student_id", _arr(raw_id)), RawColumn("label", _arr(raw_value))], len(raw_id))
    profiles = [ColumnProfile(index=0, name="student_id", type=ColumnType.IDENTIFIER, role="id"),
                ColumnProfile(index=1, name="label", type=ColumnType.FACTOR, role="row")]
    public, private = entity.profile_entity(table, profiles, Settings(mode="fidelity"))
    assert public.n_entities == 103 and public.blank_id_rows == 3
    assert public.multi_row_ids == 100 and public.rows_per_id == {2: 100}
    assert public.constant_columns == ([1] if changing <= 1 else [])
    if changing <= 1:
        assert public.constant_shares[1] == (100 - changing) / 100
        np.testing.assert_array_equal(private["entity_values"][1], np.asarray(raw_value, dtype=object)[::2][:100]
                                      .tolist() + raw_value[-3:])


@pytest.mark.parametrize("seed", range(10))
def test_per_entity_modal_values_keep_first_tie_and_native_null(seed):
    rng = np.random.default_rng(seed)
    values = _arr(rng.choice(_arr(["A", "B", "", None, float("nan"), "C"]), size=200))
    index = rng.integers(0, 12, size=len(values))
    codes, uniques = pd.factorize(values, use_na_sentinel=False)
    expected = []
    for person in range(12):
        chosen = Counter(codes[index == person].tolist()).most_common(1)[0][0]
        value = uniques[chosen]
        expected.append(value if isinstance(value, str) else None)
    actual = entity.per_entity_values(index, values, 12)
    assert actual.tolist() == expected
    assert not actual.flags.writeable
    assert codes_of(actual)[0] is codes_of(actual)[0]


@pytest.mark.parametrize("columns", [
    [], [[]], [[None, None, None]], [["", "a", None, "a", "é"], ["001", "1", "2", "1", "3"]],
    [["text", None, "nan", "None", ""], [None, "", "text", "None", "nan"]],
    [["a", "a", "b"], ["x"]], [[1, 2, 1, None], ["1", "2", "1", None]],
    [[None, float("nan"), "", "null"], ["x", "x", "x", "x"]],
])
def test_privacy_row_hashes_are_exactly_pandas_object_hashes(columns):
    arrays = [_arr(values) for values in columns]
    frame = pd.DataFrame({str(i): pd.Series(values, dtype=object) for i, values in enumerate(arrays)})
    expected = (pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
                if columns else np.zeros(0, dtype=np.uint64))
    np.testing.assert_array_equal(matches.row_hashes(arrays), expected)


def test_duplicate_rows_count_native_nulls_and_respect_excluded_columns():
    arrays = [_arr(["a", None, "a", None, "b"]), _arr(["x", "y", "x", "y", "z"]), _arr(list("12345"))]
    table = RawTable(None, ["one", "two", "skip"],
                     [RawColumn(name, value) for name, value in zip(["one", "two", "skip"], arrays)], 5)
    profiles = [ColumnProfile(index=i, name=name, type=ColumnType.FACTOR,
                              role="excluded" if i == 2 else "row")
                for i, name in enumerate(table.header)]
    result = structure._duplicate_rows(table, profiles, Settings(mode="fidelity"))
    assert result == {"rows": 2, "rate": 0.4, "status": "reproduced"}
    assert structure._factorize(arrays[0])[0] is codes_of(arrays[0])[0]


def test_duplicate_rows_retains_pandas_mixed_null_fallback():
    values = _arr([None, float("nan"), None, float("nan"), "", ""])
    table = RawTable(None, ["value"], [RawColumn("value", values)], len(values))
    profiles = [ColumnProfile(index=0, name="value", type=ColumnType.FACTOR, role="row")]
    expected = int(pd.DataFrame({"value": pd.Series(values, dtype=object)}).duplicated().sum())
    assert structure._duplicate_rows(table, profiles, Settings(mode="fidelity"))["rows"] == expected


@pytest.mark.parametrize("kind, hook, name", [
    (ColumnType.FREE_TEXT, text, "text_stats_counts"),
    (ColumnType.DATE, dates, "date_stats_counts"),
    (ColumnType.DECIMAL, numeric, "numeric_stats_extended_counts"),
])
def test_column_stats_passes_ordered_stripped_counts_without_expanding(kind, hook, name, monkeypatch):
    seen = []

    def forbid_expansion(counts):
        pytest.fail("the default statistics hook must consume counts without expanding rows")

    def receive(counts, profile):
        seen.append(list(counts.items()))
        return {"counted_rows": sum(counts.values())}

    monkeypatch.setattr(hook, name, receive)
    monkeypatch.setattr(stats, "_expand", forbid_expansion)
    profile = ColumnProfile(index=0, name="value", type=kind, missing_policy="typed", na_tokens=frozenset({"NA"}))
    result = stats.column_stats(None, profile, 10, counts={" beta ": 7, "alpha": 2, "beta": 3, None: 1, "NA": 1}, n=14)
    assert seen == [[("beta", 10), ("alpha", 2)]]
    assert result["counted_rows"] == 12 and result["missing"] == 2


@pytest.mark.parametrize("kind, hook, name", [
    (ColumnType.FREE_TEXT, text, "text_stats"),
    (ColumnType.DATE, dates, "date_stats"),
    (ColumnType.DECIMAL, numeric, "numeric_stats_extended"),
])
@pytest.mark.parametrize("empty", [False, True])
def test_column_stats_preserves_legacy_list_hook_overrides(kind, hook, name, empty, monkeypatch):
    seen = []
    profile = ColumnProfile(index=0, name="value", type=kind, missing_policy="typed", na_tokens=frozenset({"NA"}))

    def receive(values, received_profile):
        assert isinstance(values, list)
        assert received_profile is profile
        seen.append(values)
        return {"legacy_rows": len(values)}

    def forbid_counts(counts, profile):
        pytest.fail("a replaced public list hook must take precedence over the default counts hook")

    monkeypatch.setattr(hook, name, receive)
    monkeypatch.setattr(hook, name + "_counts", forbid_counts)
    counts = {None: 1, "NA": 1} if empty else {" beta ": 7, "alpha": 2, "beta": 3, None: 1, "NA": 1}
    result = stats.column_stats(None, profile, 10, counts=counts, n=sum(counts.values()))
    expected = [] if empty else ["beta"] * 10 + ["alpha"] * 2
    assert seen == [expected]
    assert result["legacy_rows"] == len(expected) and result["missing"] == 2


def test_state_stats_counts_merge_case_and_weight_without_row_expansion():
    profile = ColumnProfile(index=0, name="state", type=ColumnType.STATE, params={"_min_cell": 4})
    assert text.text_stats_counts({"il": 3, "IL": 2, "CA": 4, "NY": 1}, profile) == {
        "n_states": 3, "states_below_min_cell": 1, "state_count_profile": [5, 4]}


def test_id_rendering_normalizes_once_per_distinct_nonblank_form(monkeypatch):
    raw = _arr([" 0012345 ", "12345", "12345.0", "0098765", "", None, "-"] * 30)
    registry = ids.IdRegistry()
    registry.register("studentid", "f.csv", None, 0, "student_id", raw)
    registry.build(53)
    normalize = ids.normalize_id
    calls = []

    def record(value):
        calls.append(value)
        return normalize(value)

    monkeypatch.setattr(ids, "normalize_id", record)
    output = registry.render("studentid", raw)
    assert calls == ["0012345", "12345", "12345.0", "0098765"]
    for offset in range(0, len(raw), 7):
        assert output[offset:offset + 7].tolist() == output[:7].tolist()
    assert output[4:7].tolist() == ["", None, "-"]
    assert output[0].startswith(" ") and output[0].endswith(" ")


@pytest.mark.parametrize("columns", [[["", "\0", ""]], [["x\0a", "x\0b"]],
                                    [["x\0a", "x\0b"], ["a", "a"]],
                                    [["x\0a", "x\0b", None], ["a", "a", "a"]]])
def test_nul_duplicate_rows_and_privacy_hashes_match_object_frame(columns):
    arrays = [_arr(values) for values in columns]
    frame = pd.DataFrame({str(i): pd.Series(values, dtype=object) for i, values in enumerate(arrays)})
    table = RawTable(None, list(frame), [RawColumn(name, values) for name, values in zip(frame, arrays)],
                     len(frame))
    profiles = [ColumnProfile(i, name, ColumnType.FACTOR, role="row") for i, name in enumerate(frame)]
    assert structure._duplicate_rows(table, profiles, Settings(mode="fidelity"))["rows"] == int(frame.duplicated().sum())
    np.testing.assert_array_equal(matches.row_hashes(arrays),
                                  pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64))


def test_nul_id_rendering_keeps_exact_raw_forms_and_legacy_missing_mapping_error():
    raw = _arr(["x\0a", "x\0b", "x\0a"])
    registry = ids.IdRegistry()
    registry.register("id", "f.json", None, 0, "id", raw)
    registry.namespaces["id"].mapping = {"xa": "AA"}
    with pytest.raises(KeyError, match="xb"):
        registry.render("id", raw)
    registry.namespaces["id"].mapping["xb"] = "BB"
    assert registry.render("id", raw).tolist() == ["a\0a", "b\0b", "a\0a"]


@pytest.mark.parametrize("values", [["2020-01-01", "2020-01-01\0x", "2020-01-02"],
                                    ["2020-01-01\0x", "2020-01-01", "2020-01-02"],
                                    ["2020-01-01\0", "2020-01-01\0x"]])
def test_nul_date_counts_preserve_legacy_factorization_weights(values):
    profile = ColumnProfile(0, "date", ColumnType.DATE,
                            params={"kind": "date", "formats": [{"format": "%Y-%m-%d"}]})
    values = values * 3
    codes, uniques = pd.factorize(np.asarray(values, dtype=object))
    expected = dates._stats(dates.view(profile.params, uniques.tolist()),
                            np.bincount(codes, minlength=len(uniques)), profile.params)
    assert dates.date_stats(values, profile) == expected
    assert dates.date_stats_counts(Counter(values), profile) == expected
