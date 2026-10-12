"""R6 csv-layout: CSV/TSV/TXT outputs whose layout is derived from the source (R6-06).

``--exclude`` can leave a delimited file with one column (or fewer columns than the sniffer saw in the source),
and ``--null`` can leave a column with nothing but blanks. The written file is then checked against what the
written table can carry, never against facts it cannot show:

* a one-column file shows no delimiter: values holding the source's delimiter are quoted for it, and the
  self-check (like the leak scan) re-reads the output with the source's delimiter instead of sniffing one;
* every line after the header of a one-column file is a value: trailing blank lines (which would read back as
  blank values) are not written, a file whose values are all blank ends with a line break (otherwise the last
  blank value would be lost), and a blank header is quoted (a blank first line is no header row);
* a quoting policy counts as different only when it writes the column's values differently (a column whose
  values all need quotes cannot show whether the source quoted everything);
* the pandas dtypes of the kept columns are still compared with their source columns.

Real defects (a wrong delimiter, extra rows, a changed header, an observable quoting change) still fail.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import lzma
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import run_cli
from datanonymous.config import Settings
from datanonymous.io import csvio, registry
from datanonymous.io.registry import OutTable, ReadRequest, WriteRequest
from datanonymous.model import ColumnQuoting, SyntheticColumn, SyntheticTable
from datanonymous.profile.table import profile_table
from datanonymous.report import selfcheck

N = 12
IDS = [f"S{4710 + 7 * i}" for i in range(N)]
SCORES = [str(10 + 3 * i) for i in range(N)]
GRADES = ["A", "B", "C", "D"] * 3


def _table(delim: str, header: list[str], rows: list[list[str]], le: str = "\n", final: bool = True) -> str:
    text = le.join([delim.join(header)] + [delim.join(r) for r in rows])
    return text + (le if final else "")


def _scores(delim: str = ",", le: str = "\n", final: bool = True) -> str:
    return _table(delim, ["student_id", "Score"], [[a, b] for a, b in zip(IDS, SCORES)], le, final)


def _pack(name: str, data: bytes, comp: str | None) -> tuple[str, bytes]:
    if comp == "gz":
        return name + ".gz", gzip.compress(data, mtime=0)
    if comp == "bz2":
        return name + ".bz2", bz2.compress(data)
    if comp == "xz":
        return name + ".xz", lzma.compress(data)
    if comp == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(name, data)
        return name + ".zip", buf.getvalue()
    return name, data


def _unpack(path: Path, comp: str | None) -> bytes:
    raw = path.read_bytes()
    if comp == "gz":
        return gzip.decompress(raw)
    if comp == "bz2":
        return bz2.decompress(raw)
    if comp == "xz":
        return lzma.decompress(raw)
    if comp == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return z.read(z.namelist()[0])
    return raw


def _run(tmp: Path, name: str, data: bytes, *flags: str, comp: str | None = None, seed: str = "5",
         tag: str = "run") -> tuple[object, bytes, dict]:
    inp = tmp / (tag + "_in")
    inp.mkdir(parents=True, exist_ok=True)
    fname, payload = _pack(name, data, comp)
    (inp / fname).write_bytes(payload)
    out, rep = tmp / (tag + "_out"), tmp / (tag + "_rep")
    res = run_cli(inp / fname, "--out-dir", out, "--report-dir", rep, "--seed", seed, "--force", *flags)
    entry = json.loads((rep / "files" / (fname + ".json")).read_text(encoding="utf-8"))
    written = _unpack(out / fname, comp) if (out / fname).exists() else b""
    return res, written, entry


def _reread(data: bytes, delimiter: str | None, encoding: str | None = None):
    rr = csvio.parse_text_bytes(data, encoding, delimiter, check_header=False)
    return rr.tables[0], rr.layout.text


def _ok(res, entry: dict) -> None:
    assert res.exit_code == 0, res.output
    assert entry["written"] and entry["status"] != "FAIL" and entry["selfcheck"] == [], entry["reasons"]
    assert not [m for m in entry["messages"] if m.get("code") == "layout_mismatch"]


# --------------------------------------------------------------------------- R6-06: one column left by --exclude
@pytest.mark.parametrize("name,delim,comp,le,enc", [
    ("scores.csv", ",", None, "\n", "utf-8"),
    ("scores.tsv", "\t", None, "\n", "utf-8"),
    ("scores.txt", "|", None, "\n", "utf-8"),
    ("scores.csv", ";", None, "\n", "utf-8"),
    ("scores.csv", ",", None, "\r\n", "utf-8"),
    ("scores.csv", ",", "gz", "\n", "utf-8"),
    ("scores.csv", ",", "zip", "\n", "utf-8"),
    ("scores.tsv", "\t", "bz2", "\n", "utf-8"),
    ("scores.txt", "|", "xz", "\r\n", "utf-8"),
    ("scores.tsv", "\t", None, "\n", "utf-16-le"),
])
def test_one_column_projection_passes_the_self_check(tmp_path: Path, name: str, delim: str, comp: str | None,
                                                     le: str, enc: str) -> None:
    bom = b"\xff\xfe" if enc == "utf-16-le" else b""
    res, written, entry = _run(tmp_path, name, bom + _scores(delim, le).encode(enc), "--exclude", "student_id",
                               comp=comp)
    _ok(res, entry)
    assert written.startswith(bom)
    table, text = _reread(written, delim, "utf-16" if bom else None)
    assert table.header == ["Score"] and table.n_rows == N
    assert text.line_ending == le and text.trailing_newline
    assert not any(i.encode(enc) in written for i in IDS)
    # the report and docs give the delimiter to read it with: the source's
    assert entry["output_layout"]["text"]["delimiter"] == delim


@pytest.mark.parametrize("final", [True, False])
def test_one_column_projection_keeps_the_trailing_newline_choice(tmp_path: Path, final: bool) -> None:
    res, written, entry = _run(tmp_path, "s.csv", _scores(final=final).encode(), "--exclude", "student_id")
    _ok(res, entry)
    assert written.endswith(b"\n") == final
    assert _reread(written, ",")[0].n_rows == N


def test_trailing_blank_lines_are_not_written_as_blank_values(tmp_path: Path) -> None:
    res, written, entry = _run(tmp_path, "s.csv", (_scores() + "\n\n").encode(), "--exclude", "student_id")
    _ok(res, entry)
    table, text = _reread(written, ",")
    assert table.n_rows == N and "" not in table.columns[0].values.tolist()  # N in = N out, no extra blanks
    assert not written.endswith(b"\n\n") and text.trailing_blank_lines == 0
    # the same source without --exclude keeps its trailing blank lines
    res2, written2, entry2 = _run(tmp_path, "s.csv", (_scores() + "\n\n").encode(), tag="full")
    _ok(res2, entry2)
    assert written2.endswith(b"\n\n\n")


def test_blank_last_value_without_final_line_break(tmp_path: Path) -> None:
    vals = [s if i % 3 else "" for i, s in enumerate(SCORES)]
    vals[-1] = ""
    src = _table(",", ["student_id", "Score"], [[a, b] for a, b in zip(IDS, vals)], final=False)
    res, written, entry = _run(tmp_path, "s.csv", src.encode(), "--exclude", "student_id")
    _ok(res, entry)
    table, _ = _reread(written, ",")
    assert table.n_rows == N and table.columns[0].values.tolist().count("") == vals.count("")
    assert not written.endswith(b"\n")


@pytest.mark.parametrize("final", [True, False])
@pytest.mark.parametrize("how", ["exclude", "exclude_and_null", "null_single_column"])
def test_all_blank_one_column_keeps_every_row(tmp_path: Path, final: bool, how: str) -> None:
    if how == "exclude":
        src = _table(",", ["student_id", "Note"], [[a, ""] for a in IDS], final=final)
        flags = ["--exclude", "student_id"]
    elif how == "exclude_and_null":
        src = _table(",", ["student_id", "Score", "Grade"], [[a, b, g] for a, b, g in zip(IDS, SCORES, GRADES)],
                     final=final)
        flags = ["--exclude", "student_id", "--exclude", "Grade", "--null", "Score"]
    else:
        src = _table(",", ["Score"], [[b] for b in SCORES], final=final)
        flags = ["--null", "Score"]
    res, written, entry = _run(tmp_path, "s.csv", src.encode(), *flags)
    _ok(res, entry)
    table, _ = _reread(written, "," if how != "null_single_column" else None)
    assert len(table.header) == 1 and table.n_rows == N
    assert set(table.columns[0].values.tolist()) == {""}
    # N blank values need N line breaks after the header: the file ends with one even if the source did not
    assert written == (table.header[0] + "\n" * (N + 1)).encode()


def test_blank_header_of_the_one_column_left_is_quoted(tmp_path: Path) -> None:
    src = _table(",", ["", "student_id"], [[str(i), a] for i, a in enumerate(IDS)])
    res, written, entry = _run(tmp_path, "s.csv", src.encode(), "--exclude", "student_id")
    _ok(res, entry)
    assert written.startswith(b'""\n')
    table, _ = _reread(written, ",")
    assert table.header == [""] and table.n_rows == N


def test_values_holding_the_source_delimiter_are_quoted_for_it(tmp_path: Path) -> None:
    src = _table(",", ["student_id", "Grade"], [[a, f'"{g}, x"'] for a, g in zip(IDS, GRADES)])
    res, written, entry = _run(tmp_path, "s.csv", src.encode(), "--exclude", "student_id")
    _ok(res, entry)
    rows = list(csv.reader(io.StringIO(written.decode(), newline="")))
    assert all(len(r) == 1 for r in rows) and len(rows) == N + 1
    assert all(", " in r[0] for r in rows[1:])


#: A ';' file with a column of irregular commas (the sniffer finds ';') and a 'Band, Level' column: what is left
#: after excluding the irregular column has one ',' per line, which a sniffer prefers to ';'.
BAND = "Band, Level"
NOTES = ["a, b", "c", "d, e, f", "g"] * 3


def _bands() -> bytes:
    return _table(";", ["student_id", "Notes", BAND], [[a, n, f"{g}, x"] for a, n, g in zip(IDS, NOTES, GRADES)]
                  ).encode()


@pytest.mark.parametrize("excluded,left", [
    (("student_id", "Notes"), [BAND]),                  # one column: no delimiter to sniff
    (("Notes",), ["student_id", BAND]),                 # two columns that sniff as ','
])
def test_output_is_read_with_the_source_delimiter_not_resniffed(tmp_path: Path, excluded: tuple, left: list) -> None:
    flags = [f for col in excluded for f in ("--exclude", col)]
    res, written, entry = _run(tmp_path, "s.csv", _bands(), *flags, "--force-factor", BAND)
    _ok(res, entry)
    table, _ = _reread(written, ";")
    assert table.header == left and table.n_rows == N
    assert entry["output_layout"]["text"]["delimiter"] == ";"


@pytest.mark.parametrize("header", ['student_id,Grade', '"student_id","Grade"'])
def test_quoting_that_every_value_needs_is_not_a_difference(tmp_path: Path, header: str) -> None:
    """Every value quoted in the source: the one column left needs its quotes anyway (',' inside)."""
    src = header + "\n" + "\n".join(f'"{a}","{g}, x"' for a, g in zip(IDS, GRADES)) + "\n"
    res, written, entry = _run(tmp_path, "s.csv", src.encode(), "--exclude", "student_id")
    _ok(res, entry)
    assert written.splitlines()[0] == header.split(",")[1].encode()
    assert all(line.startswith(b'"') for line in written.splitlines()[1:])


def test_quoted_empty_values_survive_the_projection(tmp_path: Path) -> None:
    rows = [[a, b if i % 3 else '""'] for i, (a, b) in enumerate(zip(IDS, SCORES))]
    res, written, entry = _run(tmp_path, "s.csv", _table(",", ["student_id", "Score"], rows).encode(),
                               "--exclude", "student_id")
    _ok(res, entry)
    assert written.count(b'""') == 4 and _reread(written, ",")[0].n_rows == N


@pytest.mark.parametrize("final", [True, False])
def test_header_only_projection(tmp_path: Path, final: bool) -> None:
    res, written, entry = _run(tmp_path, "s.csv", ("student_id,Score" + ("\n" if final else "")).encode(),
                               "--exclude", "student_id")
    _ok(res, entry)
    assert written == b"Score" + (b"\n" if final else b"")


def test_one_column_projection_is_deterministic(tmp_path: Path) -> None:
    src = (_scores(final=False) + "").encode()
    a = _run(tmp_path, "s.csv", src, "--exclude", "student_id", tag="a")[1]
    b = _run(tmp_path, "s.csv", src, "--exclude", "student_id", tag="b")[1]
    assert a == b and a


def test_dtype_drift_of_kept_columns_is_still_compared(tmp_path: Path) -> None:
    """The kept columns' pandas dtypes are paired with their source columns, also in a one-column output."""
    src = _table(",", ["student_id", "Score", "Grade"], [[a, b, g] for a, b, g in zip(IDS, SCORES, GRADES)])
    for tag, flags in (("kept", ()), ("projected", ("--exclude", "Grade")), ("one", ("--exclude", "Grade",
                                                                                     "--exclude", "student_id"))):
        res, _, entry = _run(tmp_path, "s.csv", src.encode(), "--null", "Score", *flags, tag=tag)
        _ok(res, entry)
        check = entry["checks"]["dtypes"]
        assert check["result"] == "WARN" and "Score" in check["detail"], (tag, check)


# --------------------------------------------------------------------------- the self-check still catches defects
SRC = _scores().encode()


def _source(tmp_path: Path, data: bytes = SRC):
    p = tmp_path / "src.csv"
    p.write_bytes(data)
    rr = registry.get("csv").read(ReadRequest(p, p.name, "csv", None, Settings(mode="fidelity")))
    tp = profile_table(rr.tables[0], p.name, "csv", Settings(mode="fidelity"))
    return rr, tp


def _one_column(rr, tp, keep: int = 1, values: list | None = None) -> SyntheticTable:
    p = tp.columns[keep]
    vals = rr.tables[0].columns[keep].values if values is None else np.array(values, dtype=object)
    return SyntheticTable(tp.name, [p.name], tp.n_rows, [SyntheticColumn(p.index, p.name, vals)], tp)


def _write(tmp_path: Path, rr, st: SyntheticTable) -> Path:
    out = tmp_path / "out.csv"
    ot = OutTable(st.name, st.header, [c.values for c in st.columns], [c.index for c in st.columns], st.n_rows)
    csvio.write(WriteRequest(out, rr.layout, [ot], Settings(mode="fidelity")))
    return out


def _check(rr, st: SyntheticTable, out: Path) -> list[str]:
    # re-read as the pipeline does: with the source's encoding and delimiter
    s = Settings(mode="fidelity", encoding=rr.layout.encoding, delimiter=rr.layout.text.delimiter)
    ro = registry.get("csv").read(ReadRequest(out, out.name, "csv", None, s, purpose="selfcheck"))
    return selfcheck.compare(rr.layout, ro.layout, [st], ro.tables, csvio.layout_diff)


def test_the_written_one_column_file_passes(tmp_path: Path) -> None:
    rr, tp = _source(tmp_path, (_scores() + "\n\n").encode())
    st = _one_column(rr, tp)
    assert _check(rr, st, _write(tmp_path, rr, st)) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda b: b + b"\n\n", "row count"),                      # blank lines written after a one-column table
    (lambda b: b.replace(b"Score", b"Scores", 1), "header differs"),
    (lambda b: b.replace(b"\n", b"\r\n"), "line_ending"),
    (lambda b: b[:-1], "trailing_newline"),
    (lambda b: b"\xef\xbb\xbf" + b, "byte-order mark"),
    (lambda b: b.replace(b"\n1", b'\n"1'), "row count"),       # an unbalanced quote swallows the next lines
])
def test_defects_of_a_one_column_file_still_fail(tmp_path: Path, mutate, needle: str) -> None:
    rr, tp = _source(tmp_path)
    st = _one_column(rr, tp)
    out = _write(tmp_path, rr, st)
    out.write_bytes(mutate(out.read_bytes()))
    issues = _check(rr, st, out)
    assert any(needle in i for i in issues), issues


def test_a_wrong_delimiter_in_a_wider_output_still_fails(tmp_path: Path) -> None:
    rr, tp = _source(tmp_path)
    cols = [SyntheticColumn(p.index, p.name, rr.tables[0].columns[p.index].values) for p in tp.columns]
    st = SyntheticTable(tp.name, list(tp.header), tp.n_rows, cols, tp)
    out = tmp_path / "out.csv"
    out.write_bytes(SRC.replace(b",", b";"))
    assert any("header differs" in i for i in _check(rr, st, out))


def test_an_observable_quoting_change_still_fails(tmp_path: Path) -> None:
    """Source quotes every value; a one-column output with values that need no quotes must keep them."""
    rr, tp = _source(tmp_path, ('"student_id","Score"\n' + "\n".join(f'"{a}","{b}"' for a, b in zip(IDS, SCORES))
                                + "\n").encode())
    st = _one_column(rr, tp)
    out = _write(tmp_path, rr, st)
    assert _check(rr, st, out) == [] and out.read_bytes().count(b'"') == 2 * (N + 1)
    out.write_bytes(out.read_bytes().replace(b'"', b"").replace(b"Score", b'"Score"'))
    assert any("quoting always -> minimal" in i for i in _check(rr, st, out))


def test_the_writer_adjusts_only_what_one_column_cannot_carry() -> None:
    from datanonymous.model import TextDialect

    t = TextDialect(delimiter=",", trailing_newline=False, trailing_blank_lines=2, header_quoted=[False, False],
                    column_quoting=[ColumnQuoting(), ColumnQuoting()])
    blank = np.array([None, "", None], dtype=object)
    one = csvio.written_dialect(t, [""], [blank], [1])
    assert (one.delimiter, one.trailing_blank_lines, one.trailing_newline, one.header_quoted) == (
        ",", 0, True, [False, True])
    assert t.trailing_blank_lines == 2 and t.header_quoted == [False, False]  # the source dialect is untouched
    some = csvio.written_dialect(t, ["x"], [np.array(["1", None], dtype=object)], [1])
    assert (some.trailing_newline, some.header_quoted) == (False, [False, False])
    quoted_empty = TextDialect(delimiter=",", trailing_newline=False, column_quoting=[ColumnQuoting("minimal", True)])
    assert csvio.written_dialect(quoted_empty, ["x"], [blank], [0]).trailing_newline is False  # '""' ends it
    assert csvio.written_dialect(t, ["a", "b"], [blank, blank], [0, 1]) is t  # two columns: the source's
