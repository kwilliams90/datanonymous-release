"""Stage 1 verification remains bound to disk bytes and ordered values."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datanonymous import pipeline
from datanonymous.io import csvio, registry
from datanonymous.model import (ColumnModel, ColumnProfile, ColumnType, EntityProfile, FileLayout,
                               FileResult, RawColumn, RawTable, ReadResult, RunResult, SyntheticColumn,
                               SyntheticTable, TableProfile)
from datanonymous.privacy import leaks
from datanonymous.profile.stats import is_missing, numeric_view


def read_result(items=("a", "b")):
    column = RawColumn("x", np.asarray(items, dtype=object))
    return ReadResult(FileLayout("csv"), [RawTable(None, ["x"], [column], len(items))])


def file_result(name="data.csv"):
    return FileResult(name, name, "csv", written=True, layout=FileLayout("csv"))


def scanner():
    return SimpleNamespace(index=SimpleNamespace(flagged_in=lambda *args: set(), texts=None),
                           distinctive=lambda *args, **kwargs: [])


def test_reread_reuses_verified_bytes_after_replace_and_releases(tmp_path, monkeypatch):
    source = tmp_path / "temporary.csv"
    source.write_bytes(b"x\na\nb\n")
    key = leaks.reread_key(source)
    dest = tmp_path / "data.csv"
    os.replace(source, dest)
    fr, rr = file_result(), read_result()
    pipeline._keep_reread(fr, rr, dest, [fr], key)
    assert fr._reread is rr
    assert fr._reread_key == leaks.reread_key(dest)
    monkeypatch.setattr(registry, "output_handler", lambda *args: SimpleNamespace(
        key="csv", read=lambda request: pytest.fail("verified bytes were parsed twice")))
    assert leaks._scan_data_file(fr, tmp_path, scanner()) == []
    assert fr._reread is None and fr._reread_key is None and fr._reread_cells == 0


def test_disk_change_invalidates_verified_parse(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_bytes(b"x\na\nb\n")
    fr = file_result()
    pipeline._keep_reread(fr, read_result(), path, [fr], leaks.reread_key(path))
    old = path.stat()
    path.write_bytes(b"x\na\nc\n")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
    reads = []
    fresh = read_result(("a", "c"))
    monkeypatch.setattr(registry, "output_handler", lambda *args: SimpleNamespace(
        key="csv", read=lambda request: reads.append(request.path) or fresh))
    assert leaks._scan_data_file(fr, tmp_path, scanner()) == []
    assert reads == [path] and fr._reread is None


def test_change_between_verification_and_retention_is_not_cached(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"x\na\nb\n")
    key = leaks.reread_key(path)
    path.write_bytes(b"x\nchanged\n")
    fr = file_result()
    pipeline._keep_reread(fr, read_result(), path, [fr], key)
    assert fr._reread is None


def test_total_reread_cell_budget_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_REREAD_CELL_BUDGET", 3)
    files = [file_result("first.csv"), file_result("second.csv")]
    for fr in files:
        path = tmp_path / fr.out_rel_path
        path.write_bytes(b"x\na\nb\n")
        pipeline._keep_reread(fr, read_result(), path, files, leaks.reread_key(path))
    assert files[0]._reread_cells == 2 and files[1]._reread is None
    files[0].clear_reread()
    path = tmp_path / files[1].out_rel_path
    pipeline._keep_reread(files[1], read_result(), path, files, leaks.reread_key(path))
    assert files[1]._reread_cells == 2


@pytest.mark.parametrize("missing", [True, False])
def test_per_file_cleanup_runs_on_missing_file_or_scanner_exception(tmp_path, monkeypatch, missing):
    path = tmp_path / "data.csv"
    path.write_bytes(b"x\na\nb\n")
    fr = file_result()
    pipeline._keep_reread(fr, read_result(), path, [fr], leaks.reread_key(path))
    if missing:
        path.unlink()
        assert leaks._scan_data_file(fr, tmp_path, scanner()) == []
    else:
        def broken(*args):
            raise RuntimeError("scanner failure")
        data = scanner()
        data.index.flagged_in = broken
        monkeypatch.setattr(registry, "output_handler", lambda *args: SimpleNamespace(key="csv"))
        with pytest.raises(RuntimeError, match="scanner failure"):
            leaks._scan_data_file(fr, tmp_path, data)
    assert fr._reread is None


@pytest.mark.parametrize("broken", [False, True])
def test_scan_cleanup_even_when_index_empty_or_scan_fails(monkeypatch, broken):
    fr = file_result()
    fr._reread, fr._reread_cells = read_result(), 2
    run = SimpleNamespace(files=[fr])
    def scan(*args):
        if broken:
            raise RuntimeError("scan failure")
        return []
    monkeypatch.setattr(leaks, "_scan_outputs", scan)
    monkeypatch.setattr(leaks, "_scan_incomplete", lambda *args: [])
    assert leaks.scan_outputs(run, {}) == []
    assert fr._reread is None and fr._reread_cells == 0


@pytest.mark.parametrize("indices,n", [([2, 0, 2, 1, 0], 3), ([1, 1], 3),
                                      ([-1, 1, -2, 0], 2), ([], 2)])
def test_first_rows_preserves_old_index_semantics(indices, n):
    index = np.asarray(indices, dtype=np.int64)
    expected = np.full(n, -1, dtype=np.int64)
    for row in range(len(index) - 1, -1, -1):
        expected[index[row]] = row
    np.testing.assert_array_equal(pipeline._first_rows(index, n), expected)
    assert pipeline._first_rows(None, n) is None
    assert pipeline._first_rows(index, 0) is None


@pytest.mark.parametrize("items", [[" 1.5 ", None, "NA", "", "2.5", "1.5", "inf"],
                                   [None, "", "NA"], []])
def test_distinct_numeric_view_preserves_finite_row_order(items):
    values = np.asarray(items, dtype=object)
    profile = ColumnProfile(0, "score", ColumnType.FLOAT, na_tokens=frozenset({"NA"}))
    stripped = [value.strip() for value in values
                if not is_missing(value, profile.missing_policy, profile.na_tokens)]
    np.testing.assert_array_equal(pipeline._synthetic_numbers(values, profile), numeric_view(profile, stripped))


@pytest.mark.parametrize("items", [["1", "1\0x"], ["1\0x", "1", "2"], ["1\0x", None, "1"]])
def test_nul_numeric_view_uses_actual_rows(items):
    values = np.asarray(items, dtype=object)
    values.flags.writeable = False
    profile = ColumnProfile(0, "score", ColumnType.FLOAT)
    stripped = [value.strip() for value in values
                if not is_missing(value, profile.missing_policy, profile.na_tokens)]
    np.testing.assert_array_equal(pipeline._synthetic_numbers(values, profile), numeric_view(profile, stripped))


def test_nul_exact_extreme_check_uses_actual_rows_even_with_supplied_codes():
    values = np.asarray(["1234567890123456\0x", "1234567890123456"], dtype=object)
    values.flags.writeable = False
    profile = ColumnProfile(0, "score", ColumnType.INTEGER, role="row", model=SimpleNamespace(
        nonmissing_counts={"1234567890123456": 1}, sorted_numeric=np.array([1234567890123456.])))
    assert not pipeline._extremes_missing(values, profile, uniques=pipeline.codes_of(values)[1])


def test_entity_fidelity_receives_exact_nul_counts(tmp_path, monkeypatch):
    values = np.asarray(["", "", "\0", "\0"], dtype=object)
    profile = ColumnProfile(0, "x", ColumnType.FACTOR, role="entity")
    tp = TableProfile(None, ["x"], 4, [profile], entity=EntityProfile(n_entities=2))
    tp.private.entity_index = np.asarray([0, 0, 1, 1])
    tp.private.entity_models[0] = ColumnModel()
    tp.private.extra["entity_values"] = {0: values[[0, 2]]}
    st = SyntheticTable(None, ["x"], 4, [SyntheticColumn(0, "x", values)], tp)
    fr = FileResult("data.json", "data.json", "json", layout=FileLayout("json"), tables=[tp])
    run = RunResult("run", tmp_path, tmp_path, None, 7, False, files=[fr])
    handler = SimpleNamespace(key="json", write=lambda req: req.path.write_bytes(b"[]"),
                              read=lambda req: ReadResult(fr.layout, []), layout_diff=lambda *args: [],
                              inferred_dtypes=lambda *args: None)
    monkeypatch.setattr(registry, "get", lambda *args: handler)
    monkeypatch.setattr(registry, "output_handler", lambda *args: handler)
    monkeypatch.setattr(pipeline, "generate_table", lambda *args: st)
    monkeypatch.setattr(pipeline.matches, "enforce", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline.structure_mod, "finalize_rows", lambda *args: [])
    monkeypatch.setattr(leaks, "scan_table", lambda *args: [])
    monkeypatch.setattr(pipeline.selfcheck, "compare", lambda *args: [])
    seen = []

    def compare(p, real, synthetic, *args, **kwargs):
        if kwargs.get("basis") == "entities":
            seen.append(synthetic)
        return {"verdict": "pass", "checks": []}

    monkeypatch.setattr(pipeline.fidelity, "compare_column", compare)
    monkeypatch.setattr(pipeline.fidelity, "with_row_check", lambda entity, rows: entity)
    assert pipeline.generate_file(fr, pipeline.Settings(mode="fidelity"), run, pipeline.IdRegistry(), {}) is not None, run.debug
    expected = pipeline.column_stats(values[[0, 2]], profile, 10, counts={"": 1, "\0": 1}, n=2)
    assert seen == [expected]
    assert seen[0]["missing"] == 1


def test_readonly_final_column_keeps_single_column_trailing_blank_writer_behavior():
    rr = csvio.parse_text_bytes(b"score\n1\n\n2")
    values = np.asarray(["1", "2", ""], dtype=object)
    expected = csvio.serialize(rr.layout, ["score"], [values], rr.layout.text.column_quoting,
                               rr.layout.text.header_quoted)
    values.flags.writeable = False
    actual = csvio.serialize(rr.layout, ["score"], [values], rr.layout.text.column_quoting,
                             rr.layout.text.header_quoted)
    assert actual == expected
    assert values.tolist() == ["1", "2", ""]
    assert csvio.parse_text_bytes(actual).tables[0].n_rows == 3


@pytest.fixture
def literal_package(tmp_path, monkeypatch):
    root = tmp_path / "package"
    root.mkdir()
    (root / "a.py").write_text('"""Private docstring."""\nMESSAGE = "alpha beta"\nN = 42\n', encoding="utf-8")
    nested = root / "sub"
    nested.mkdir()
    (nested / "b.py").write_text('class Thing:\n    MESSAGE = "gamma delta"\n', encoding="utf-8")
    monkeypatch.setattr(leaks, "_tool_source_root", lambda: root)
    leaks._tool_literals.cache_clear()
    yield root
    leaks._tool_literals.cache_clear()


def test_persistent_literals_equal_fresh_extraction_and_avoid_ast_on_cache_hit(literal_package, monkeypatch):
    expected = tuple(text for path in sorted(literal_package.rglob("*.py"))
                     for text in leaks.literal_texts(path.read_text(encoding="utf-8")) + [path.stem])
    assert leaks._tool_literals() == expected
    leaks._tool_literals.cache_clear()
    monkeypatch.setattr(leaks, "literal_texts", lambda *args: pytest.fail("warm cache parsed source"))
    assert leaks._tool_literals() == expected


def test_literal_cache_invalidates_on_source_bytes_even_with_same_stat(literal_package):
    first = leaks._tool_literals()
    path = literal_package / "a.py"
    old = path.stat()
    path.write_bytes(path.read_bytes().replace(b"alpha", b"omega"))
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    leaks._tool_literals.cache_clear()
    second = leaks._tool_literals()
    assert "alpha beta" in first and "alpha beta" not in second and "omega beta" in second
    assert len(list((literal_package / "__pycache__").glob("tool_literals-*.json"))) == 2


@pytest.mark.parametrize("contents", ["not JSON", "[]", '{"literals": 1}', "tamper"])
def test_malformed_or_corrupted_literal_cache_recomputes_exactly(literal_package, contents):
    expected = leaks._tool_literals()
    path = next((literal_package / "__pycache__").glob("tool_literals-*.json"))
    if contents == "tamper":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["literals"].append("Injected exemption")
        contents = json.dumps(payload)
    path.write_text(contents, encoding="utf-8")
    leaks._tool_literals.cache_clear()
    assert leaks._tool_literals() == expected


def test_readonly_literal_cache_location_falls_back_to_fresh_computation(literal_package, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("read only")
    monkeypatch.setattr(leaks.tempfile, "mkstemp", denied)
    expected = leaks._tool_literals()
    assert "alpha beta" in expected
    assert not list((literal_package / "__pycache__").glob("tool_literals-*.json"))
    leaks._tool_literals.cache_clear()
    assert leaks._tool_literals() == expected


def test_literal_extraction_walks_ast_once_and_preserves_collection_exclusions(monkeypatch):
    walk = leaks.ast.walk
    calls = []
    def counted(tree):
        calls.append(tree)
        return walk(tree)
    monkeypatch.setattr(leaks.ast, "walk", counted)
    names = ", ".join(f'"Name{i}"' for i in range(60))
    source = f'"""Doc words."""\nNAMES = ({names},)\nSMALL = ("Alpha", "Beta")\n'
    texts = leaks.literal_texts(source)
    assert calls and len(calls) == 1
    assert texts == ["Alpha", "Beta"]
