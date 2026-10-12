"""First-appearance codes must preserve the existing profiling semantics."""

from collections import Counter
import gc
import json
import weakref

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from datanonymous.config import ColumnOptions, Settings
from datanonymous.detect.context import DetectContext
from datanonymous.io import arrowio, jsonio
from datanonymous.io.registry import ReadRequest
from datanonymous.model import RawColumn
from datanonymous.profile import codes as module
from datanonymous.profile.codes import codes_of, counts_of, has_nul, register_arrow, subset_codes


def immutable(items):
    values = np.asarray(items, dtype=object)
    values.flags.writeable = False
    return values


def assert_equivalent(values, result):
    codes, uniques = result
    expected, labels = pd.factorize(values, use_na_sentinel=False)
    labels = [label if isinstance(label, str) else None for label in labels]
    assert codes.dtype == np.int64
    assert not codes.flags.writeable
    assert not uniques.flags.writeable
    np.testing.assert_array_equal(codes, expected)
    assert uniques.tolist() == labels


@given(st.lists(st.one_of(st.text(), st.none()), max_size=150))
def test_codes_and_counts_match_existing_primitives(items):
    values = immutable(items)
    assert_equivalent(values, codes_of(values))
    expected = Counter(values.tolist())
    assert counts_of(values) == expected
    assert list(counts_of(values)) == list(expected)
    assert codes_of(values)[0] is codes_of(values)[0]


@given(st.lists(st.one_of(st.text(), st.none()), max_size=100), st.data())
def test_subsets_have_their_own_first_appearance(items, data):
    values = immutable(items)
    rows = np.asarray(data.draw(st.lists(st.integers(0, len(items) - 1), max_size=100))
                      if items else [], dtype=np.int64)
    assert_equivalent(values[rows], subset_codes(values, rows))


@pytest.mark.parametrize("items", [[], [None], [None, "a", None, "b"],
                                   ["a", "b", None, "a", "c", None],
                                   ["a", None, "b", None, "c"], ["", "NA", " a", "a"]])
@pytest.mark.parametrize("chunked", [False, True])
def test_arrow_codes_preserve_null_position_and_chunk_order(items, chunked):
    values = immutable(items)
    array = (pa.chunked_array([pa.array(items[:2], type=pa.string()),
                              pa.array(items[2:], type=pa.string())])
             if chunked else pa.array(items, type=pa.string()))
    register_arrow(values, array)
    assert_equivalent(values, codes_of(values))
    assert list(counts_of(values).items()) == list(Counter(items).items())


def test_cache_cannot_be_changed_through_returned_codes_or_counts():
    values = immutable(["b", None, "a", "b"])
    codes, uniques = codes_of(values)
    with pytest.raises(ValueError):
        codes[0] = 100
    with pytest.raises(ValueError):
        codes.setflags(write=True)
    with pytest.raises(ValueError):
        uniques[0] = "changed"
    counts = counts_of(values)
    counts["b"] = 999
    assert counts_of(values)["b"] == 2
    assert_equivalent(values, codes_of(values))


def test_source_lifetime_evicts_cache_including_arrow():
    values = immutable(["a", "b"])
    key = id(values)
    ref = weakref.ref(values)
    register_arrow(values, pa.array(["a", "b"]))
    result = codes_of(values)
    assert key in module._CACHE
    del values
    gc.collect()
    assert ref() is None
    assert key not in module._CACHE
    assert result[1].tolist() == ["a", "b"]


def test_id_collision_is_checked_and_stale_callback_preserves_replacement():
    old = immutable(["old"])
    codes_of(old)
    old_entry = module._CACHE[id(old)]
    current = immutable(["current"])
    # Deterministically emulate an id-key collision; real allocator reuse is
    # platform-dependent and usually evicted by the weakref callback first.
    module._CACHE[id(current)] = old_entry
    assert codes_of(current)[1].tolist() == ["current"]
    replacement = module._CACHE[id(current)]
    module._CACHE[id(old)] = replacement
    old_ref = weakref.ref(old)
    old_key = id(old)
    del old
    gc.collect()
    assert old_ref() is None
    assert module._CACHE[old_key] is replacement
    module._CACHE.pop(old_key)


def test_mutable_array_is_recomputed_and_never_cached():
    values = np.asarray(["a", "b", "a"], dtype=object)
    assert codes_of(values)[1].tolist() == ["a", "b"]
    assert id(values) not in module._CACHE
    values[1] = "a"
    assert codes_of(values)[1].tolist() == ["a"]
    assert counts_of(values) == {"a": 3}
    register_arrow(values, pa.array(["a", "a", "a"]))
    assert id(values) not in module._CACHE


def test_readonly_view_of_mutable_base_is_not_cached():
    parent = np.asarray(["a", "b"], dtype=object)
    values = parent.view()
    values.flags.writeable = False
    assert codes_of(values)[1].tolist() == ["a", "b"]
    assert id(values) not in module._CACHE
    parent[0] = "b"
    assert counts_of(values) == {"b": 2}


def test_mixed_numbers_and_native_nulls_keep_exact_counter_semantics():
    nan1, nan2 = float("nan"), float("nan")
    values = immutable([1, 1.0, True, None, nan1, nan2, nan1, "1", pd.NA])
    expected = Counter(values.tolist())
    assert list(counts_of(values).items()) == list(expected.items())
    expected_codes, expected_uniques = pd.factorize(values, use_na_sentinel=False)
    actual_codes, actual_uniques = codes_of(values)
    np.testing.assert_array_equal(actual_codes, expected_codes)
    for actual, expected_value in zip(actual_uniques, expected_uniques, strict=True):
        if isinstance(expected_value, float) and np.isnan(expected_value):
            assert isinstance(actual, float) and np.isnan(actual)
        else:
            assert actual == expected_value
    assert id(values) not in module._CACHE
    # A subset's representative must remain True rather than inherit 1.
    result = subset_codes(values, np.asarray([2]))
    assert result[1].tolist() == [True]
    assert result[1][0] is True


def test_string_and_nan_is_not_cached_or_counted_as_none():
    nan = float("nan")
    values = immutable(["a", None, nan, nan])
    assert list(counts_of(values).items()) == list(Counter(values.tolist()).items())
    codes_of(values)
    assert id(values) not in module._CACHE


def test_raw_column_boundary_rejects_source_mutation():
    column = RawColumn("source", np.asarray(["a", None], dtype=object))
    with pytest.raises(ValueError):
        column.values[0] = "changed"
    assert column.values.tolist() == ["a", None]


def test_arrow_registration_rejects_invalid_contract():
    values = immutable(["a"])
    with pytest.raises(ValueError, match="equal lengths"):
        register_arrow(values, pa.array(["a", "b"]))
    with pytest.raises(TypeError, match="strings"):
        register_arrow(values, pa.array([1]))


@pytest.mark.parametrize("items", [["", "\0"], ["x\0a", "x\0b", "x"],
                                   ["x\0a", None, "x\0b", "x"], ["\0", "", "\0\0"]])
@pytest.mark.parametrize("chunked", [False, True])
def test_nul_retains_pandas_codes_but_exact_counter_counts(items, chunked):
    values = immutable(items)
    array = (pa.chunked_array([pa.array(items[:1]), pa.array(items[1:])])
             if chunked else pa.array(items))
    register_arrow(values, array)
    assert has_nul(values)
    assert module._CACHE[id(values)].arrow is None
    assert_equivalent(values, codes_of(values))
    assert list(counts_of(values).items()) == list(Counter(items).items())
    for rows in [np.asarray([1]), np.arange(len(values) - 1, -1, -1),
                 np.asarray([i for i, value in enumerate(items) if value is not None])]:
        assert_equivalent(values[rows], subset_codes(values, rows))


@given(st.lists(st.one_of(st.none(), st.text(alphabet="ab01\0\x01\x1f\n\ré😀", max_size=10)),
                max_size=80), st.data())
def test_control_characters_keep_codes_counts_and_selected_representatives(items, data):
    values = immutable(items)
    register_arrow(values, pa.array(items, type=pa.string()))
    assert_equivalent(values, codes_of(values))
    assert list(counts_of(values).items()) == list(Counter(items).items())
    rows = np.asarray(data.draw(st.lists(st.integers(0, len(items) - 1), max_size=80))
                      if items else [], dtype=np.int64)
    assert_equivalent(values[rows], subset_codes(values, rows))


def test_nul_eligibility_is_evicted_and_rechecked_for_mutable_bases():
    values = immutable(["", "\0"])
    counts_of(values)
    key, ref = id(values), weakref.ref(values)
    del values
    gc.collect()
    assert ref() is None and key not in module._CACHE
    base = np.asarray(["a", "b"], dtype=object)
    view = base.view()
    view.flags.writeable = False
    assert not has_nul(view)
    base[1] = "\0"
    assert has_nul(view)
    assert counts_of(view) == {"a": 1, "\0": 1}
    assert id(view) not in module._CACHE


@pytest.mark.parametrize("format", ["json", "parquet", "feather"])
def test_supported_readers_detect_nul_distinct_from_blank(tmp_path, format):
    import pyarrow.feather as feather
    import pyarrow.parquet as parquet

    path = tmp_path / ("values." + format)
    items = ["", "\0", "\0"]
    if format == "json":
        path.write_text(json.dumps([{"category": value} for value in items]), encoding="utf-8")
        read = jsonio.read_json
    else:
        table = pa.table({"category": items})
        (parquet.write_table if format == "parquet" else feather.write_feather)(table, path)
        read = arrowio.read_parquet if format == "parquet" else arrowio.read_feather
    settings = Settings(mode="fidelity")
    column = read(ReadRequest(path, path.name, format, None, settings)).tables[0].columns[0]
    ctx = DetectContext(index=0, name=column.name, values=column.values, file_rel=path.name,
                        table=None, format=format, options=ColumnOptions(), settings=settings,
                        native=column.native)
    assert column.values.tolist() == items
    assert list(ctx.raw_counts.items()) == [("", 1), ("\0", 2)]
    assert ctx.n_blank == 1
    assert ctx.text_values == ctx.typed_values == {"\0": 2}
