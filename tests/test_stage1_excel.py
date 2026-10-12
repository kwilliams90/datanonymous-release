"""The XLSX dtype shortcut agrees with the actual pandas workbook reader."""

from __future__ import annotations

import datetime as dt
import io
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from datanonymous.config import Settings
from datanonymous.errors import ReadError
from datanonymous.io import excel_read, excelio
from datanonymous.io.registry import ReadRequest


def _read(path, *, header=1, ranges=None):
    return excelio.read_xlsx(ReadRequest(path, path.name, "xlsx", None,
                                        Settings(mode="fidelity", header_row=header, range=ranges or [])))


def _reference(path, info):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = pd.read_excel(path, sheet_name=info["name"], header=info["header_row"] - 1,
                                  nrows=info["n_rows"], usecols=list(range(info["first_col"] - 1, info["last_col"])))
        return list(map(str, frame.dtypes))
    except Exception:
        return None


def _check(path, *, header=1, ranges=None, require_fast=False):
    result = _read(path, header=header, ranges=ranges)
    if require_fast:
        assert result.layout.extra.get("_xlsx_dtypes")
    assert "_xlsx_dtypes" not in result.layout.to_public().get("extra", {})
    for info in result.layout.extra["sheets"]:
        if info.get("notes_only") or not info.get("header_row"):
            continue
        assert excelio.inferred_dtypes(path, result.layout, info["name"]) == _reference(path, info)
    return result


def _write_values(path, values):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["row_id", "value"])
    for i, value in enumerate(values, 1):
        ws.append([i, value])
    wb.save(path)
    wb.close()


NATIVE_VALUES = [None, "", True, False, 0, 1, -1, 1.5, 1e20, 1e-12, "0", "1", "TRUE", "False",
                 "NA", "NULL", "nan", "#N/A", "#DIV/0!", "x", "2024-01-01", "1e20", "18446744073709551616",
                 dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 2, 3, 4, 5, 123000), dt.time(3, 4, 5),
                 dt.timedelta(hours=27, minutes=5)]
VALUE = st.one_of(st.sampled_from(NATIVE_VALUES), st.integers(-(10**19), 10**19),
                  st.floats(min_value=-1e25, max_value=1e25, allow_nan=False, allow_infinity=False),
                  st.text(alphabet="abc XYZ0123\t\n\ré😀", max_size=12))


@settings(max_examples=100, deadline=None)
@given(st.lists(VALUE, min_size=1, max_size=25))
def test_mixed_workbooks_match_read_excel(values):
    # In-memory workbook construction still exercises the actual openpyxl XML
    # round trip: Excel serializes large integers and date precision differently.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        path = Path(folder) / "mixed.xlsx"
        _write_values(path, values)
        _check(path, require_fast=True)


@pytest.mark.parametrize("value", NATIVE_VALUES, ids=repr)
def test_native_storage_mixtures(value, tmp_path):
    path = tmp_path / "mixed.xlsx"
    for ordinary in [[1, 0, 1], [True, False, True], [1.5, 0.5, 1.5], ["1", "0", "1"],
                     [dt.datetime(2024, 1, 1)] * 3]:
        _write_values(path, ordinary + [value] + ordinary)
        _check(path, require_fast=True)


def test_distinct_native_values_keep_type_and_order():
    buf = excel_read.ColBuf()
    for i, (value, kind) in enumerate([(True, "b"), (1, "n"), ("1", "s"), (1.0, "n"),
                                       (False, "b"), (0, "n"), ("#N/A", "e"), ("#DIV/0!", "e"), (None, "")]):
        buf.put(i, excel_read.XCell(value, kind, "General"))
    distinct = excel_read._native_dtype_values(buf, len(buf.values))
    assert [(type(v), v) for v in distinct[:6]] == [(bool, True), (int, 1), (str, "1"), (bool, False),
                                                   (int, 0), (float, np.nan)]
    assert distinct[-1] == ""
    assert len(distinct) == 7


@pytest.mark.parametrize("case", ["title", "footer", "blank_row", "range", "merged", "side_notes", "numeric_header"])
def test_nonplain_windows_use_full_reader(case, tmp_path, monkeypatch):
    path = tmp_path / "layout.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    rows = [["row_id", "value"], [1, 20], [2, 30], [3, 40]]
    header = 1
    ranges = []
    if case == "title":
        rows.insert(0, ["District report"])
        header = 2
    elif case == "footer":
        rows.extend([[], ["Total", 90]])
    elif case == "blank_row":
        rows.insert(2, [])
    elif case == "range":
        ranges = ["A1:B3"]
    elif case == "numeric_header":
        rows[0] = [2024, 2025]
    for row in rows:
        ws.append(row)
    if case == "merged":
        ws.merge_cells("A1:B1")
    elif case == "side_notes":
        ws["D2"] = "Note about an unrelated group"
    wb.save(path)
    wb.close()
    result = _check(path, header=header, ranges=ranges)
    assert not result.layout.extra.get("_xlsx_dtypes")
    called = []
    original = pd.read_excel

    def read_excel(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", read_excel)
    assert excelio.inferred_dtypes(path, result.layout, "Sheet") is not None
    assert called


def test_plain_sheet_needs_no_second_workbook_parse(tmp_path, monkeypatch):
    path = tmp_path / "plain.xlsx"
    _write_values(path, [1, 2, 3])
    result = _check(path, require_fast=True)
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: pytest.fail("duplicate workbook parse"))
    assert excelio.inferred_dtypes(path, result.layout, "Sheet") == ["int64", "int64"]


def test_dtype_cache_rejects_changed_file_or_window(tmp_path):
    path = tmp_path / "plain.xlsx"
    _write_values(path, [1, 2, 3])
    result = _check(path, require_fast=True)
    _write_values(path, ["one", "two", "three"])
    assert excelio.inferred_dtypes(path, result.layout, "Sheet") == ["int64", "str"]
    _write_values(path, [1, 2, "three"])
    result = _check(path, require_fast=True)
    result.layout.extra["sheets"][0]["n_rows"] = 2
    assert excelio.inferred_dtypes(path, result.layout, "Sheet") == ["int64", "int64"]


def test_dtype_cache_respects_pandas_string_option(tmp_path):
    path = tmp_path / "plain.xlsx"
    _write_values(path, ["one", "two"])
    result = _check(path, require_fast=True)
    with pd.option_context("future.infer_string", False):
        assert excelio.inferred_dtypes(path, result.layout, "Sheet") == ["int64", "object"]


def test_dtype_cache_follows_atomic_rename(tmp_path, monkeypatch):
    source = tmp_path / "temporary.xlsx"
    dest = tmp_path / "output.xlsx"
    _write_values(source, [1, 2, 3])
    result = _check(source, require_fast=True)
    os.replace(source, dest)
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: pytest.fail("rename should preserve verified dtypes"))
    assert excelio.inferred_dtypes(dest, result.layout, "Sheet") == ["int64", "int64"]


def test_dtype_cache_does_not_follow_unrelated_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.xlsx"
    copy = tmp_path / "copy.xlsx"
    _write_values(source, [1, 2, 3])
    result = _check(source, require_fast=True)
    copy.write_bytes(source.read_bytes())
    stat = source.stat()
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    calls = []
    original = pd.read_excel

    def read_excel(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", read_excel)
    assert excelio.inferred_dtypes(copy, result.layout, "Sheet") == ["int64", "int64"]
    assert calls  # identical contents/timestamps are insufficient without the same inode


def test_cli_reuses_source_and_output_dtypes(tmp_path, monkeypatch):
    from conftest import run_cli

    source = tmp_path / "input" / "input.xlsx"
    source.parent.mkdir()
    _write_values(source, list(range(20, 60)))
    calls = []
    original = pd.read_excel

    def read_excel(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", read_excel)
    result = run_cli(source, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep",
                     "--header-row", "1", "--seed", "1")
    assert result.exit_code == 0, result.output
    assert not calls


def test_truncated_dimensions_do_not_hide_rows_or_columns(tmp_path):
    path = tmp_path / "plain.xlsx"
    _write_values(path, [1, 2, 3])
    original = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(path, "w") as dest:
        for info in source.infolist():
            body = source.read(info)
            if info.filename == "xl/worksheets/sheet1.xml":
                body = body.replace(b'<dimension ref="A1:B4"', b'<dimension ref="A1:A1"')
            dest.writestr(info, body)
    result = _check(path, require_fast=True)
    assert result.tables[0].n_rows == 3
    assert len(result.tables[0].columns) == 2


ROOT = Path(__file__).resolve().parents[1]
WORKBOOKS = sorted((ROOT / "tests/fixtures").rglob("*.xlsx"))


@pytest.mark.parametrize("path", WORKBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_fixture_workbooks_match_read_excel(path):
    try:
        _check(path, header=None)
    except ReadError:
        # The optimization does not change which malformed fixtures are read.
        return
