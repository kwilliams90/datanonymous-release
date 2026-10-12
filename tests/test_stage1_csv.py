"""Stage 1 CSV optimizations agree with the authoritative stdlib/pandas paths."""

from __future__ import annotations

import csv
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from datanonymous.errors import ReadError
from datanonymous.io import csvio, sniff
from datanonymous.profile import codes


EDGE_TEXTS = [
    "a,b\n1,2\n3,4\n", "a,b\r\n1,2\r\n3,4\r\n", "a,b\r1,2\r3,4\r",
    "a,b\n1,2\n3,4", 'a,b\n"x, y",2\n', 'a,b\n"he said ""hi""",2\n',
    'a,b\n"line1\nline2",2\n3,4\n', 'a,b\r\n"line1\r\nline2",2\r\n3,4\r\n',
    "a,b\n5'10\",x\nab\"c,y\n", 'a,b\n"abc"def,1\n"x"  ,2\n', 'a,b\n "abc",1\n',
    'a,b\n"",1\n,2\n', "a,b\n1,2\n\n\n", "a,b\n1,2\n\n3,4\n",
    "a,b\n1,2,3\n4,5\n", "a,b\n1\n4,5\n", 'a,b\n"abc,1\n2,3\n',
    "a,b\n 1 , 2 \n\t,\n", "név,b\nÁrvíztűrő,2\n😀,3\n", "a,b,\n1,2,\n",
    "a,a,\n1,2,3\n", '"a","b"\n1,2\n', 'a,b\n"x\ry",2\n', 'a,b\n",1\n',
    "a\tb\n1\t2\n", "a;b\n1,5;2\n", 'a|b\n"x|y"|2\n', "a,b\nNA,NULL\n,null\n",
    "a,b\n1,2\n \n3,4\n", 'a,b\n"p1\n\np2",1\n', "a,b\r\n1,2\r3,4\r\n",
    'a,b\n"x""",1\n', "a,b\n" + "x" * 3_000_000 + ",1\n",
    "a,b\n1,2\nReport generated on Monday\n", "a\n1\n\n2\n", "a,b\n", "\n1,2\n",
    'a,b\n,"\r', 'a,b\n,"\n', "a,b\n\r,",
]


def _result(data: bytes, delimiter: str):
    try:
        rr = csvio.parse_text_bytes(data, delimiter=delimiter, check_header=False)
    except ReadError as exc:
        return type(exc), str(exc), exc.fix
    return (rr.layout.to_public(), rr.layout.text, rr.messages,
            [(t.header, [c.values.tolist() for c in t.columns], t.n_rows, t.layout) for t in rr.tables])


@pytest.mark.parametrize("text", EDGE_TEXTS, ids=lambda text: repr(text[:40]))
def test_arrow_parse_matches_stdlib_path(text, monkeypatch):
    delimiter = "\t" if text.startswith("a\tb") else ";" if text.startswith("a;b") else "|" if text.startswith("a|b") else ","
    data = text.encode("utf-8")
    fast = _result(data, delimiter)
    monkeypatch.setattr(csvio, "_arrow_records", lambda *a, **kw: None)
    assert fast == _result(data, delimiter)


CELL = st.text(alphabet="abc XYZ0123,;|\t\r\n\"é😀", max_size=35)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(CELL, CELL, CELL), min_size=1, max_size=15), st.sampled_from([",", ";", "|", "\t"]),
       st.sampled_from(["\r\n", "\n", "\r"]))
def test_arrow_random_quoted_records(rows, delimiter, ending):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator=ending, quoting=csv.QUOTE_ALL)
    writer.writerow(["a", "b", "c"])
    writer.writerows(rows)
    text = stream.getvalue()
    reference = csvio._parse_records(text, delimiter)
    arrow = csvio._arrow_records(text, delimiter, reference[0])
    if arrow is not None:
        assert [c.to_pylist() for c in arrow.columns] == list(map(list, zip(*reference)))
    parsed = csvio.parse_text_bytes(text.encode(), delimiter=delimiter, check_header=False).tables[0]
    assert [c.values.tolist() for c in parsed.columns] == list(map(list, zip(*rows)))


@pytest.mark.parametrize("text", ["a,b\n1,2\n\n3,4\n", "a,b\n1,2,3\n", "a,b\nx\x00,2\n",
                                   "a,b\n1,2\n \n", 'a,b\n"abc,1\n2,3\n'])
def test_irregular_arrow_parse_declines(text):
    assert csvio._arrow_records(text, ",", ["a", "b"]) is None


@settings(max_examples=1000, deadline=None)
@given(st.text(alphabet='ab12 ,"\r\n', min_size=1, max_size=100))
def test_arrow_permissive_quoting_matches_stdlib(body):
    text = "a,b\n" + body
    arrow = csvio._arrow_records(text, ",", ["a", "b"])
    if arrow is None:
        return
    rows = csvio._parse_records(text, ",")
    while rows and rows[-1] == []:
        rows.pop()
    assert all(len(row) == 2 for row in rows)
    assert [c.to_pylist() for c in arrow.columns] == list(map(list, zip(*rows)))


def test_reader_registers_immutable_arrow_values():
    table = csvio.parse_text_bytes(b"id,score\n1,20\n2,30\n1,20\n").tables[0]
    col = table.columns[0].values
    assert not col.flags.writeable
    assert codes._CACHE[id(col)].arrow is not None
    assert codes.codes_of(col)[1].tolist() == ["1", "2"]
    assert codes.counts_of(col) == {"1": 2, "2": 1}


def _reference_records(text, delimiter, limit, min_records):
    stream = io.StringIO(text, newline="")
    rows = []
    try:
        for row in csv.reader(stream, delimiter=delimiter, quotechar='"', strict=False):
            rows.append(row)
            if len(rows) >= min_records and stream.tell() >= limit:
                break
    except csv.Error:
        return []
    return rows


@pytest.mark.parametrize("ending", ["\r\n", "\n", "\r"])
@pytest.mark.parametrize("limit", [0, 17, 4095, 4096, 8192, 65536])
def test_sniff_slice_handles_multiline_and_unterminated_records(ending, limit):
    for text in [f'a,b{ending}"' + ("x" * 4000 + ending) * 5 + f'",1{ending}2,3{ending}' * 4,
                 f'a,b{ending}"' + ("x" * 4000 + ending) * 5,
                 (f'a,b{ending}1,2{ending}' * 10000), 'a,b\n"' + "x" * 100_000 + '",2\n']:
        for delimiter in [",", "|"]:
            assert sniff._records(text, delimiter, limit, 3) == _reference_records(text, delimiter, limit, 3)


def test_header_arrays_keep_repeated_header_exception():
    for rows in [[("a", "x"), ("y", "b")], [("a", "b"), ("x", "y")]]:
        lists = list(map(list, zip(*rows)))
        arrays = [np.array(c, dtype=object) for c in lists]
        assert sniff.header_problems(["a", "b"], arrays) == sniff.header_problems(["a", "b"], lists)


DTYPE_SPECIAL = ["", "NA", "N/A", "NULL", "None", "nan", "NaN", "true", "false", "True", "FALSE",
                 "+5", "-0", "1.", ".5", "1e2", "1e309", "1e-999", "inf", "-inf", "Infinity",
                 "9223372036854775807", "9223372036854775808", "18446744073709551615",
                 "18446744073709551616", "-9223372036854775809", "000000000000000000001",
                 "1_000", "1,000", "0x10", " 5 ", "1\n", "2\r", "3\v", "4\f", "é", "hello",
                 "1.2.3", "--1", "+", "1e", "nanosecond", "TRUE ", " NA "]
DTYPE_VALUE = st.one_of(st.sampled_from(DTYPE_SPECIAL), st.integers(-(10**20), 10**20).map(str),
                       st.floats(allow_nan=False, allow_infinity=False).map(str), CELL)


def _pandas_dtype(values):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(["v", "x"])
    writer.writerows((v, "0") for v in values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return str(pd.read_csv(io.StringIO(stream.getvalue()), low_memory=False).dtypes.iloc[0])


@settings(max_examples=400, deadline=None)
@given(st.lists(DTYPE_VALUE, min_size=1, max_size=60))
def test_dtype_witness_matches_pandas(values):
    assert csvio._dtype_from_distinct(list(dict.fromkeys(values))) == _pandas_dtype(values)


@pytest.mark.parametrize("special", DTYPE_SPECIAL)
def test_dtype_witness_numeric_mixtures(special):
    for ordinary in [[str(i) for i in range(-20, 50)], [str(i / 10) for i in range(-20, 50)]]:
        assert len(csvio._dtype_witnesses(ordinary)) < 8
        for values in [ordinary + [special], [special] + ordinary, ordinary[:30] + [special] + ordinary[30:]]:
            assert csvio._dtype_from_distinct(values) == _pandas_dtype(values)


ROOT = Path(__file__).resolve().parents[1]
CSV_FIXTURES = sorted([p for folder in (ROOT / "tests/fixtures", ROOT / "example-data")
                       for p in folder.rglob("*") if p.suffix.lower() in {".csv", ".tsv", ".txt"}])


@pytest.mark.parametrize("path", CSV_FIXTURES, ids=lambda path: str(path.relative_to(ROOT)))
def test_fixture_dtypes_match_full_pandas_read(path):
    try:
        rr = csvio.parse_text_bytes(path.read_bytes(), check_header=False)
    except ReadError:
        return
    actual = csvio.inferred_dtypes(path, rr.layout)
    rr.layout.extra.pop("_csv_arrow", None)
    assert actual == csvio.inferred_dtypes(path, rr.layout)


@pytest.mark.parametrize("data", [b"id,score\n1,2\nReport generated on Monday\n", b"id,score\n1,2\n\n3,4\n",
                                  b"id,score\r\n1,2\n", b"id,score\n1,5'10\"\n", b"score\n1\n2\n"])
def test_irregular_dtype_layout_falls_back(data, tmp_path, monkeypatch):
    path = tmp_path / "in.csv"
    path.write_bytes(data)
    rr = csvio.parse_text_bytes(data, delimiter=",", check_header=False)
    monkeypatch.setattr(csvio, "_inferred_dtypes_arrow", lambda *a: pytest.fail("irregular dtype fast path"))
    csvio.inferred_dtypes(path, rr.layout)


def test_dtype_hook_reads_current_file_bytes(tmp_path):
    path = tmp_path / "in.csv"
    path.write_bytes(b"id,score\n1,20\n2,30\n")
    layout = csvio.parse_text_bytes(path.read_bytes()).layout
    assert csvio.inferred_dtypes(path, layout) == ["int64", "int64"]
    path.write_bytes(b"id,score\n1,text\n2,more\n")
    assert csvio.inferred_dtypes(path, layout) == ["int64", "str"]
