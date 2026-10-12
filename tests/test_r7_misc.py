"""R7 misc: delimiter sniffing among consistent candidates (R7-05), exact rows of D35 pseudonyms in within-ID columns
(R7-06), and linear-time D35/D34 bookkeeping (R7-07).

* R7-05: a semicolon (tab, pipe) file whose header and every row hold a ``Last, First`` name splits consistently at
  the comma too. Every consistent candidate is weighed: quoting first (a split that cuts quoted values apart loses),
  then punctuation (a character followed by a space is text), then text-likeness (comma, then ``;``, ``|``, tab).
  A guess the quoting did not decide is a ``delimiter_ambiguous`` warning with the ``--delimiter`` fix, and the
  report's layout self-check (which re-reads with the same delimiter and so cannot disagree) is WARN.
* R7-06: a pseudonym writes no real value, so entity balancing deals it like any level: it keeps its source level's
  rows however few IDs hold it (D35: exact counts). A rare REAL value (an allowlisted vocabulary word) is still
  never tied to block sizes (that would describe the few real IDs holding it). Thousands of one-ID pseudonyms are
  made exact in linear time before the squared-error descent.
* R7-07: the pseudonym inverse map is built once per map (functional-dependency rebuilds were O(K^2)); the per-unit
  duplicate-row rule runs in one pass over (row, unit) pairs; the D34 screen's derived indexes, folder listings and
  file counts are built once, not per flagged name.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import EXAMPLE_FILES, EXAMPLES, read_dicts, run_cli
from datanonymous.detect import text as text_detect
from datanonymous.detect.text import PSEUDO_MAP_KEY
from datanonymous.generate import pseudonyms
from datanonymous.io import compress, csvio, sniff
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType, RawColumn, RawTable
from datanonymous.privacy import headers as d34
from datanonymous.profile import entity as entity_mod
from datanonymous.profile import structure as st

REPO = Path(__file__).resolve().parent.parent


# =========================================================================== R7-05 delimiter sniffing
def _lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


SEMI_ROSTER = _lines("student_id;Last, First", "S1001;Smith, Jane", "S1001;Jones, Jane", "S1002;Brown, Mark",
                     "S1002;Brown, Mark")


@pytest.mark.parametrize("delim", [";", "\t", "|"])
def test_text_commas_never_win_over_the_real_delimiter(delim: str) -> None:
    text = _lines(*(delim.join(r) for r in (["student_id", "Last, First", "grade"], ["S1001", "Smith, Jane", "9"],
                                            ["S1002", "Brown, Mark", "10"], ["S1003", "Lee, Ann", "11"])))
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.method, info.n_fields, info.alternatives) == (delim, "punctuation", 3, {",": 2})
    assert info.ambiguous


def test_the_codex_roster_reads_with_the_semicolon() -> None:
    info = sniff.sniff_delimiter(SEMI_ROSTER)
    assert (info.delimiter, info.method, info.alternatives) == (";", "punctuation", {",": 2})


def test_punctuation_beats_more_columns() -> None:
    """The comma would give more columns (``Name, Given, Middle``), but it is text punctuation."""
    text = _lines("id;Name, Given, Middle", "1;Smith, Jane, Q", "2;Brown, Mark, R")
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.n_fields, info.alternatives) == (";", 2, {",": 3})


def test_a_comma_file_with_spaced_semicolons_in_a_column_stays_comma() -> None:
    text = _lines("id,Reading; Math,score", "1,a; b,5", "2,c; d,6", "3,e; f,7")
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.method, info.n_fields) == (",", "punctuation", 3)


def test_without_spaces_the_least_text_like_candidate_splits() -> None:
    text = _lines("student_id;Last,First", "S1001;Smith,Jane", "S1002;Brown,Mark", "S1003;Lee,Ann")
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.method, info.alternatives) == (";", "text-likeness", {",": 2})
    assert info.ambiguous


def test_quoting_decides_and_is_no_ambiguity() -> None:
    text = _lines('"student_id";"Last, First"', '"S1001";"Smith, Jane"', '"S1002";"Brown, Mark"')
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.method) == (";", "quoting") and not info.ambiguous
    result = csvio.parse_text_bytes(text.encode())
    assert result.tables[0].header == ["student_id", "Last, First"]
    assert not result.layout.text.delimiter_alternatives
    assert not [m for m in result.messages if m.code == "delimiter_ambiguous"]


@pytest.mark.parametrize("text, delim", [
    ("a;b;c\n1,5;x;3\n2,5;y;4\n", ";"),  # decimal commas: the header holds none
    ("a,b\n1,2\n3,4\n", ","),
    ("id\tname\n1\tSmith, J\n2\tLee\n", "\t"),
    ('sid,name,score\n"S01","Smith, J",85\n"S02","Lee",90\n', ","),
])
def test_one_consistent_candidate_is_no_ambiguity(text: str, delim: str) -> None:
    info = sniff.sniff_delimiter(text)
    assert (info.delimiter, info.method, info.alternatives) == (delim, "consistency", {})


def _delimited_inputs() -> list[Path]:
    paths = [EXAMPLES / n for n in EXAMPLE_FILES]
    for p in sorted((REPO / "tests" / "fixtures").rglob("*")):
        name = p.name.lower()
        stem = name.rsplit(".", 1)[0] if name.endswith((".gz", ".bz2", ".xz", ".zip")) else name
        if p.is_file() and stem.endswith((".csv", ".tsv", ".txt")):
            paths.append(p)
    return paths


def test_no_example_or_fixture_file_is_ambiguous() -> None:
    """No false warning: every delimited example and fixture file has exactly one consistent candidate."""
    seen = 0
    for p in _delimited_inputs():
        name = p.name.lower()
        comp = next((c for c in ("gz", "bz2", "xz", "zip") if name.endswith("." + c)), None)
        try:
            data = compress.read_bytes(p, comp) if comp else p.read_bytes()
            text = sniff.decode(data, sniff.sniff_encoding(data))
            if not text.strip() or text.lstrip().startswith("<"):
                continue
            info = sniff.sniff_delimiter(text)
        except Exception:  # noqa: BLE001 - fixtures of failing files (binary, ragged rows ...) fail as before
            continue
        seen += 1
        assert info.alternatives == {}, p.relative_to(REPO).as_posix()
    assert seen >= len(EXAMPLE_FILES)


def _run(tmp: Path, name: str, data: bytes, *flags: str, seed: str = "31", tag: str = "run"):
    inp = tmp / f"{tag}_in"
    inp.mkdir(parents=True, exist_ok=True)
    (inp / name).write_bytes(data)
    out, rep = tmp / f"{tag}_out", tmp / f"{tag}_rep"
    res = run_cli(inp / name, "--out-dir", out, "--report-dir", rep, "--seed", seed, "--force", *flags)
    entry = json.loads((rep / "files" / (name + ".json")).read_text(encoding="utf-8"))
    return res, out, rep, entry


def test_semicolon_roster_keeps_its_id_blocks_and_warns(tmp_path: Path) -> None:
    res, out, rep, entry = _run(tmp_path, "input.csv", SEMI_ROSTER.encode())
    assert res.exit_code == 0, res.output
    lines = (out / "input.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "student_id;Last, First"
    rows = [ln.split(";") for ln in lines[1:]]
    assert all(len(r) == 2 for r in rows)
    ids = [r[0] for r in rows]
    assert sorted(Counter(ids).values()) == [2, 2]  # two IDs, two rows each (the source blocks)
    assert not {"S1001", "S1002"} & set(ids) and all(len(i) == 5 and i.startswith("S") for i in ids)
    warn = [m for m in entry["messages"] if m["code"] == "delimiter_ambiguous"]
    assert len(warn) == 1 and warn[0]["level"] == "warning"
    assert warn[0]["fix_commands"] == ["--delimiter semicolon", "--delimiter comma"]
    for secret in ("S1001", "Smith", "Jane", "Brown"):
        assert secret not in json.dumps(warn)
    assert entry["checks"]["layout_selfcheck"]["result"] == "WARN"
    assert "semicolon used; comma also consistent" in entry["checks"]["layout_selfcheck"]["detail"]
    assert entry["layout"]["text"]["delimiter_alternatives"] == {",": 2}
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert report["files"][0]["status"] == "WARN"


def test_confirming_the_delimiter_clears_the_warning(tmp_path: Path) -> None:
    res, out, _, entry = _run(tmp_path, "input.csv", SEMI_ROSTER.encode(), "--delimiter", "semicolon")
    assert res.exit_code == 0, res.output
    assert not [m for m in entry["messages"] if m["code"] == "delimiter_ambiguous"]
    assert entry["checks"]["layout_selfcheck"]["result"] == "PASS"
    assert "delimiter_alternatives" not in entry["layout"]["text"]


def test_inspect_shows_the_ambiguity(tmp_path: Path) -> None:
    src = tmp_path / "input.csv"
    src.write_bytes(SEMI_ROSTER.encode())
    res = run_cli("inspect", src)
    assert res.exit_code == 0, res.output
    assert "ambiguous delimiter" in res.output and "Smith" not in res.output


# =========================================================================== R7-06 exact rows of pseudonyms
def _rare_blocks() -> bytes:
    rows = [["student_id", "Activity", "Visit"]]
    for i in range(6):
        for j in range(1 if i < 3 else 20):
            rows.append([f"S{1000 + i}", "Quenby" if i < 3 else "Zelwick", str(j)])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue().encode()


@pytest.mark.parametrize("seed", ["31", "1", "2", "3"])
def test_pseudonyms_keep_their_level_rows_on_unequal_blocks(tmp_path: Path, seed: str) -> None:
    res, out, _, entry = _run(tmp_path, "input.csv", _rare_blocks(), seed=seed)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "input.csv")
    per_id: dict[str, set[str]] = {}
    for r in syn:
        per_id.setdefault(r["student_id"], set()).add(r["Activity"])
    assert all(len(v) == 1 for v in per_id.values())  # still constant within each ID
    assert sorted(Counter(r["student_id"] for r in syn).values()) == [1, 1, 1, 20, 20, 20]
    counts = Counter(r["Activity"] for r in syn)
    assert sorted(counts.values()) == [3, 60]  # each pseudonym keeps its source level's rows
    assert not {"Quenby", "Zelwick"} & set(counts)
    assert "fidelity_drift" not in [m["code"] for m in entry["messages"]]


def test_a_pseudonymised_token_among_numbers_keeps_its_rows(tmp_path: Path) -> None:
    """A within-ID numbers column with a rare token (D35 pseudonymised): the value bins and the token are dealt to
    ID blocks with their source rows, the token under its pseudonym (it got none before: 4-5 rows of 40)."""
    rows = [["student_id", "Lexile", "visit"]]
    for i in range(60):
        score = "Qx-7" if i < 2 else str(100 + (i * 37) % 400)
        rows += [[f"S{1000 + i}", score, str(j)] for j in range(20 if i < 2 else 1 + i % 3)]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    for seed in ("1", "2"):
        res, out, _, _ = _run(tmp_path, "data.csv", buf.getvalue().encode(), seed=seed, tag=seed)
        assert res.exit_code == 0, res.output
        toks = Counter(r["Lexile"] for r in read_dicts(out / "data.csv") if not r["Lexile"].isdigit())
        assert list(toks.values()) == [40] and "Qx-7" not in toks


def test_a_rare_real_label_is_still_never_tied_to_block_sizes() -> None:
    """Label 0 is a real value held by one ID (an allowlisted word): it stays where it was dealt. Labels 1 and 2
    are pseudonyms held by three IDs each: their rows come out exact."""
    sizes = np.asarray([50, 1, 1, 1, 20, 20, 20] + [3] * 20)
    real_labels = np.asarray([0, 1, 1, 1, 2, 2, 2] + [3] * 20)
    target = np.bincount(real_labels, weights=sizes, minlength=4)
    deal = np.asarray([3, 2, 1, 2, 1, 2, 1, 0] + [3] * 19)
    out = st.balance_entity_rows(deal, sizes, target, np.random.default_rng(4), 10, [True, False, False, True])
    assert out[7] == 0 and np.array_equal(np.bincount(out), np.bincount(deal))
    rows = np.bincount(out, weights=sizes, minlength=4)
    assert rows[1] == 3 and rows[2] == 60


def test_balance_entity_values_targets_pseudonyms() -> None:
    sizes = np.asarray([1, 1, 1, 20, 20, 20])
    values = np.asarray(["Duis", "Esse", "Duis", "Esse", "Duis", "Esse"], dtype=object)
    row_counts = {"Duis": 3, "Esse": 60}
    kept = st.balance_entity_values(values, sizes, row_counts, (), True, np.random.default_rng(0), 10)
    moved = st.balance_entity_values(values, sizes, row_counts, (), True, np.random.default_rng(0), 10,
                                     frozenset({"Duis", "Esse"}))
    assert list(kept) == list(values)  # real values held by 3 IDs: never matched to block sizes
    assert {v: int(sizes[moved == v].sum()) for v in ("Duis", "Esse")} == {"Duis": 3, "Esse": 60}


def test_thousands_of_one_id_pseudonyms_are_exact_and_quick() -> None:
    rs = np.random.default_rng(3)
    n = 20_000
    sizes = np.minimum(rs.geometric(0.05, n), 400)
    labels = rs.integers(0, 10, n)
    rare = rs.random(n) < 0.15
    labels[rare] = 10 + rs.integers(0, 4000, int(rare.sum()))
    _, labels = np.unique(labels, return_inverse=True)
    k = int(labels.max()) + 1
    held = np.bincount(labels, minlength=k)
    target = np.bincount(labels, weights=sizes, minlength=k)
    start = time.perf_counter()
    out = st.balance_entity_rows(rs.permutation(labels), sizes, target, np.random.default_rng(1), 10, held >= 10)
    assert time.perf_counter() - start < 10  # the descent alone took ~40 s here
    assert np.array_equal(np.bincount(out, minlength=k), held)
    assert np.array_equal(np.bincount(out, weights=sizes, minlength=k), target)


def test_balancing_with_pseudonyms_is_deterministic() -> None:
    rs = np.random.default_rng(5)
    sizes = rs.integers(1, 40, 400)
    labels = rs.integers(0, 60, 400)
    held = np.bincount(labels, minlength=60)
    target = np.bincount(labels, weights=sizes, minlength=60)
    deal = rs.permutation(labels)
    a = st.balance_entity_rows(deal, sizes, target, np.random.default_rng(9), 10, held >= 10)
    b = st.balance_entity_rows(deal, sizes, target, np.random.default_rng(9), 10, held >= 10)
    assert np.array_equal(a, b)


# =========================================================================== R7-07 linear bookkeeping
def _profile(mapping: dict[str, str], t: ColumnType = ColumnType.FACTOR, params: dict | None = None) -> ColumnProfile:
    p = ColumnProfile(index=0, name="Activity", type=t, params=params or {})
    p.model = ColumnModel(nonmissing_counts={k: 1 for k in mapping}, missing_counts={})
    p.model.extra[PSEUDO_MAP_KEY] = mapping
    return p


def test_the_inverse_map_is_built_once_per_map() -> None:
    p = _profile({"Quenby": "Duis", "Zelwick": "Esse"})
    inv = pseudonyms.inverse_map(p)
    assert inv == {"Duis": "Quenby", "Esse": "Zelwick"} and pseudonyms.inverse_map(p) is inv
    assert pseudonyms.real_value(p, "Esse") == "Zelwick" and pseudonyms.real_value(p, "Band") == "Band"
    p.model.extra[PSEUDO_MAP_KEY] = {"Quenby": "Odio"}  # a new map: a new inverse
    assert pseudonyms.real_value(p, "Odio") == "Quenby" and pseudonyms.real_value(p, "Duis") == "Duis"


def test_inverse_lookups_scale_linearly() -> None:
    k = 40_000
    p = _profile({f"Rare activity {i:05d}": f"Duis tempor {i:05d}" for i in range(k)})
    start = time.perf_counter()
    got = [pseudonyms.real_value(p, f"Duis tempor {i:05d}") for i in range(k)]
    assert time.perf_counter() - start < 5  # rebuilding the inverse per call: ~K^2 = 1.6e9 dictionary steps
    assert got[123] == "Rare activity 00123"


def test_multi_select_items_map_back_through_the_cached_inverse() -> None:
    p = _profile({"Quenby": "Duis"}, ColumnType.MULTI_SELECT, {"delimiter": ";"})
    assert pseudonyms.real_value(p, "Duis;Band") == "Quenby;Band"


def test_case_forms_get_distinct_pseudonyms() -> None:
    forms = [f"Rose Garden {i}" for i in range(300)] + [f"ROSE GARDEN {i}" for i in range(300)]
    p = ColumnProfile(index=0, name="Club", type=ColumnType.FACTOR,
                      params={text_detect.PSEUDO_KEY: [[f, f"[pseudonymized {i}]"] for i, f in enumerate(forms)]})
    p.model = ColumnModel(nonmissing_counts={f: 1 for f in forms}, missing_counts={})
    a = pseudonyms.draw_map([p], np.random.default_rng(3), frozenset())
    b = pseudonyms.draw_map([p], np.random.default_rng(3), frozenset())
    assert a == b and len(set(a.values())) == len(forms)
    assert all(a[f"ROSE GARDEN {i}"] == a[f"Rose Garden {i}"].upper() for i in range(300))


def _reference_duplicates(keys: entity_mod.HolderKeys, own_values: np.ndarray, masks: dict[str, np.ndarray]):
    """The rule as it was written per unit (a full row mask each): the vectorised one must agree exactly."""
    rows = keys._combined()
    found = {}
    for k, mask in masks.items():
        r = int(mask.sum())
        if r >= 2:
            same = np.unique(rows[mask], return_counts=True)[1]
            if len(same) < r:
                found[k] = (mask, same)
    if not found:
        return {}
    own_codes = np.asarray(pd.factorize(np.asarray(own_values, dtype=object), use_na_sentinel=False)[0])
    n_groups = int(own_codes.max()) + 1
    sizes = np.bincount(own_codes, minlength=n_groups).astype(float)
    pairs = sizes * (sizes - 1)
    rest = keys._combined(0)
    width = int(rest.max()) + 1
    joint, counts = np.unique(own_codes * width + rest, return_counts=True)
    alike = np.zeros(n_groups)
    np.add.at(alike, joint // width, counts.astype(float) * (counts - 1))
    out = {}
    for k, (mask, same) in found.items():
        r = int(mask.sum())
        mine = np.unique(own_codes[mask])
        base = float(pairs.sum() - pairs[mine].sum())
        if base > 0:
            share = float(alike.sum() - alike[mine].sum()) / base
        else:
            outside = np.bincount(rest[~mask], minlength=width) / max(1, len(rest) - r)
            share = float((outside * outside).sum())
        if share > entity_mod.DUPLICATE_SHARE:
            continue
        observed = float((same * (same - 1) / 2).sum())
        expected = r * (r - 1) / 2 * share
        if observed > expected + 3 * expected ** 0.5 + 1:
            out[k] = int(len(same))
    return out


def _masks(values: np.ndarray, p, wanted) -> dict[str, np.ndarray]:
    rows, unit, keys = text_detect.unit_rows(values, p, wanted)
    out = {}
    for u, k in enumerate(keys):
        m = np.zeros(len(values), dtype=bool)
        m[rows[unit == u]] = True
        out[k] = m
    return out


@pytest.mark.parametrize("multi", [False, True])
def test_duplicate_rule_is_unchanged_by_the_one_pass_form(multi: bool) -> None:
    rs = np.random.default_rng(11 if multi else 7)
    found = 0
    for _ in range(80):
        n = int(rs.integers(20, 300))
        if multi:
            items = [f"it{j}" for j in range(int(rs.integers(2, 12)))]
            own = [";".join(sorted(set(rs.choice(items, int(rs.integers(1, 3)))))) for _ in range(n)]
        else:
            own = [f"lv{int(x) % 30}" for x in rs.zipf(1.3, n)]
        a = rs.integers(0, int(rs.choice([2, 50, 1000])), n).astype(str).tolist()
        b = rs.integers(0, 1000, n).astype(str).tolist()
        dup = rs.random(n) < float(rs.choice([0.0, 0.2, 0.5]))
        for i in range(1, n):
            if dup[i]:
                own[i], a[i], b[i] = own[i - 1], a[i - 1], b[i - 1]
        raw = RawTable(None, ["own", "a", "b"], [RawColumn(h, np.asarray(v, dtype=object))
                                                for h, v in (("own", own), ("a", a), ("b", b))], n)
        p = SimpleNamespace(type=ColumnType.MULTI_SELECT if multi else ColumnType.FACTOR, params={"delimiter": ";"},
                            na_tokens=frozenset(), missing_policy="text")
        values = np.asarray(own, dtype=object)
        wanted = list(text_detect.unit_holders(values, p))
        keys = entity_mod.HolderKeys([], False, "rows", table=raw)
        got = keys.duplicates(0, values, text_detect.unit_rows(values, p, wanted))
        ref = _reference_duplicates(entity_mod.HolderKeys([], False, "rows", table=raw), values,
                                    _masks(values, p, wanted))
        assert got == ref
        found += bool(got)
    assert found >= 1  # the rule fired (and agreed) on some tables


def test_unit_rows_pairs_each_row_with_its_units() -> None:
    p = SimpleNamespace(type=ColumnType.MULTI_SELECT, params={"delimiter": ";"}, na_tokens=frozenset(),
                        missing_policy="text")
    values = np.asarray(["Band;Choir", "Choir", "", "band; Art", None], dtype=object)
    rows, unit, keys = text_detect.unit_rows(values, p, ["band", "choir", "art"])
    pairs = sorted((keys[u], int(r)) for r, u in zip(rows, unit))
    assert pairs == [("art", 3), ("band", 0), ("band", 3), ("choir", 0), ("choir", 1)]
    assert text_detect.unit_rows(values, p, ["drama"])[2] == []


def test_name_screen_indexes_follow_every_new_flag() -> None:
    screen = d34.NameScreen()
    screen.add("Garcia_Ana", "file", frozenset({"name"}), "file 1 of 1", written=True)
    assert screen.standalone_texts() == {"Garcia_Ana"} and screen.kinds_of("Garcia_Ana") == {"name"}
    assert screen.texts_in("scores Garcia_Ana 2024", standalone=True) == {"Garcia_Ana"}
    screen.add("Lopez_Maria", "column", frozenset({"name", "id"}), "column 2 of 3", written=True, group_only=True)
    screen.add("Lopez_Maria", "sheet", frozenset({"email"}), "sheet 1 of 1", written=True)
    assert screen.standalone_texts() == {"Garcia_Ana", "Lopez_Maria"}
    assert screen.kinds_of("Lopez_Maria") == {"name", "id", "email"}
    assert screen.texts_in("x Lopez_Maria y") == {"Lopez_Maria"}
    assert screen.kinds_of("Other") == frozenset()


def test_many_flagged_files_in_one_folder_are_numbered_by_one_listing(tmp_path: Path, monkeypatch) -> None:
    """Per-student files named by real IDs: each flagged file's place in its folder comes from one listing of the
    folder (it was listed and sorted again for every flagged file)."""
    src = tmp_path / "in"
    ids = [f"{482913 + 7919 * i}" for i in range(25)]
    (src / "iep").mkdir(parents=True)
    roster = "student_id,Score\n" + "".join(f"{s},{50 + i}\n" for i, s in enumerate(ids))
    (src / "roster.csv").write_text(roster, encoding="utf-8")
    for s in ids[:12]:
        (src / "iep" / f"IEP_{s}.csv").write_text("Goal,Minutes\n" + "".join(f"{g},{g * 5}\n" for g in range(12)),
                                                  encoding="utf-8")
    calls = Counter()
    real_listdir = d34._listdir

    def counting(folder: Path):
        calls[folder.name] += 1
        return real_listdir(folder)
    monkeypatch.setattr(d34, "_listdir", counting)
    res = run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--seed", "7")
    assert res.exit_code == 3, res.output
    report = json.loads((tmp_path / "rep" / "report.json").read_text(encoding="utf-8"))
    failed = sorted(f["reasons"][0] for f in report["files"] if f["status"] == "FAIL")
    assert len(failed) == 12
    places = sorted(int(m.group(1)) for r in failed for m in [re.search(r"file (\d+) of 12 in folder iep/ ", r)] if m)
    assert places == list(range(1, 13))
    assert calls["iep"] == 1
    for s in ids[:12]:
        assert s not in res.output and s not in json.dumps(report)
