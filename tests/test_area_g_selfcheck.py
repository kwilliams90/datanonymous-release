"""AREA G self-check: every CSV layout aspect is compared, the format hook is called, facts are recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from datanonymous.config import Settings
from datanonymous.io import csvio, registry
from datanonymous.io.registry import ReadRequest
from datanonymous.model import SyntheticColumn, SyntheticTable
from datanonymous.profile.table import profile_table
from datanonymous.report import selfcheck

SOURCE = (b'id,name,score,"note"\n1,a,10,x\n2,b,NA,x\n3,"c, d",30,y\n4,e,,y\n')


def _read(path: Path):
    return registry.get("csv").read(ReadRequest(path, path.name, "csv", None, Settings(mode="fidelity"), purpose="selfcheck"))


def _synth(src: Path) -> tuple:
    rr = _read(src)
    tp = profile_table(rr.tables[0], src.name, "csv", Settings(mode="fidelity"))
    cols = [SyntheticColumn(p.index, p.name, rr.tables[0].columns[p.index].values) for p in tp.columns]
    return rr, SyntheticTable(tp.name, list(tp.header), tp.n_rows, cols, tp)


def _compare(tmp_path: Path, out_bytes: bytes) -> tuple[list[str], dict]:
    src = tmp_path / "src.csv"
    src.write_bytes(SOURCE)
    rr, st = _synth(src)
    out = tmp_path / "out.csv"
    out.write_bytes(out_bytes)
    ro = _read(out)
    issues = selfcheck.compare(rr.layout, ro.layout, [st], ro.tables, csvio.layout_diff)
    return issues, selfcheck.facts(st.profile)


def test_identical_output_has_no_issue_and_records_facts(tmp_path: Path) -> None:
    issues, facts = _compare(tmp_path, SOURCE)
    assert issues == []
    assert facts == {"rows_out": 4, "columns_out": 4, "header_matches_written": True, "header_matches_source": True}


@pytest.mark.parametrize("variant,needle", [
    (b"\xef\xbb\xbf" + SOURCE, "byte-order mark"),
    (SOURCE.replace(b"\n", b"\r\n"), "line_ending"),
    (SOURCE.replace(b",", b";").replace(b'"c; d"', b'"c, d"'), "delimiter"),
    (SOURCE + b"5,f,50,z\n", "row count"),
    (SOURCE.replace(b'id,name', b'ID,name', 1), "header differs"),
    (SOURCE.replace(b'"note"', b"note"), "header quoting"),
    (SOURCE.replace(b",NA,", b",N/A,"), "missing tokens differ"),
    (SOURCE[:-1], "trailing_newline"),
    (SOURCE + b"\n", "trailing_blank_lines"),
    (SOURCE.replace(b"1,a,10", b'1,"a",10').replace(b"2,b,", b'2,"b",').replace(b"4,e,", b'4,"e",'), "quoting"),
])
def test_each_csv_layout_aspect_is_compared(tmp_path: Path, variant: bytes, needle: str) -> None:
    issues, _ = _compare(tmp_path, variant)
    assert any(needle in i for i in issues), issues


def test_coverage_lists_every_csv_aspect() -> None:
    cov = selfcheck.coverage("csv")
    for aspect in ("encoding", "byte-order mark", "delimiter", "quote character", "line ending", "trailing newline",
                   "trailing blank lines", "header text and order", "header quoting", "per-column quoting",
                   "missing-value tokens per column", "row count"):
        assert aspect in cov
    assert "xlsx layout (format hook)" in selfcheck.coverage("xlsx")


def test_the_output_formats_layout_hook_is_called(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_bytes(SOURCE)
    rr, st = _synth(src)
    calls = []

    def hook(source, output, source_tables, output_tables):  # noqa: ANN001
        calls.append((source, output, source_tables, output_tables))
        return ["sheet order differs"]

    issues = selfcheck.compare(rr.layout, rr.layout, [st], rr.tables, hook)
    assert "sheet order differs" in issues and calls[0][3] is rr.tables


def test_pandas_dtype_drift_is_a_warning_text() -> None:
    assert selfcheck.dtype_messages(["int64", "str"], ["int64", "float64"], ["a", "b"]) == [
        "column 'b': pandas reads str from the source but float64 from the output"]
    assert selfcheck.dtype_messages(["int64"], ["int64"], ["a"]) == []
