"""CSV reading/writing: byte-identical round trips and sniffers (acceptance criterion 2)."""

from __future__ import annotations

import pytest

from datanonymous.errors import ReadError
from datanonymous.io import csvio, sniff

from conftest import EXAMPLE_FILES, EXAMPLES


def roundtrip(data: bytes, **kw) -> tuple[bytes, object]:
    r = csvio.parse_text_bytes(data, **kw)
    t = r.tables[0]
    lay = r.layout
    out = csvio.serialize(lay, t.header, [c.values for c in t.columns], lay.text.column_quoting,
                          lay.text.header_quoted)
    return out, r


@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_examples_roundtrip_byte_identical(name: str) -> None:
    data = (EXAMPLES / name).read_bytes()
    out, _ = roundtrip(data)
    assert out == data


def test_example_line_endings_and_whitespace() -> None:
    _, r = roundtrip((EXAMPLES / "StudentsPerformance.csv").read_bytes())
    assert r.layout.text.line_ending == "\r\n"
    for name in ("student-scores.csv", "students_data.csv"):
        _, r = roundtrip((EXAMPLES / name).read_bytes())
        assert r.layout.text.line_ending == "\n"
        assert r.layout.encoding == "utf-8" and r.layout.bom == b""
    _, r = roundtrip((EXAMPLES / "students_data.csv").read_bytes())
    sit = r.tables[0].columns[r.tables[0].header.index("SitKCSE")].values.tolist()
    assert "Central " in sit and "Eastern " in sit


CASES = {
    "bom_utf8": "﻿Student ID,Name\r\n001,José\r\n002,Zoë\r\n".encode("utf-8"),
    "utf16_tab": b"\xff\xfe" + "id\tname\r\n1\tAé\r\n2\tB\r\n".encode("utf-16-le"),
    "cp1252": "name,note\nJosé,“quoted” café\nZoë,x\n".encode("cp1252"),
    "pipe": b"a|b|c\n1|x, y|3\n4|z|6\n",
    "semicolon": b"a;b;c\n1,5;x;3\n2,5;y;4\n",
    "single_column_blanks": b"student_id\n000123\n\n004567\n",
    "quote_all": b'"sid","name","score"\r\n"000123","Smith, J","85"\r\n"000124","Lee","90"\r\n',
    "strings_only": b'sid,name,score\n"S01","Smith, J",85\n"S02","Lee",90\n',
    "embedded": b'id,text\n1,"line one\nline two"\n2,"say ""hi"", ok"\n3,plain\n',
    "no_trailing_newline": b"a,b\n1,2\n3,4",
    "trailing_blank_line": b"a,b\n1,2\n\n",
    "dup_blank_headers": b"id,score,score,,Unnamed\n1,2,3,4,5\n",
    "header_only": b"a,b,c\n",
    "na_tokens": b"x,y\nNA,none\n,None\nnull,NULL\n",
    "empty": b"",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_roundtrip_cases(name: str) -> None:
    data = CASES[name]
    out, _ = roundtrip(data)
    assert out == data


def test_encoding_sniff() -> None:
    assert sniff.sniff_encoding(CASES["cp1252"]).encoding == "cp1252"
    info = sniff.sniff_encoding(CASES["bom_utf8"])
    assert info.encoding == "utf-8" and info.bom == b"\xef\xbb\xbf"
    assert sniff.sniff_encoding(CASES["utf16_tab"]).encoding == "utf-16-le"
    low = sniff.sniff_encoding(b"a\n\x81\x8d\n")
    assert low.encoding == "latin-1" and low.messages and low.messages[0].level == "warning"


def test_delimiters_and_quoting_detected() -> None:
    assert csvio.parse_text_bytes(CASES["pipe"]).layout.text.delimiter == "|"
    assert csvio.parse_text_bytes(CASES["semicolon"]).layout.text.delimiter == ";"
    assert csvio.parse_text_bytes(CASES["utf16_tab"]).layout.text.delimiter == "\t"
    assert csvio.parse_text_bytes(CASES["single_column_blanks"]).layout.text.delimiter is None
    assert csvio.parse_text_bytes(CASES["quote_all"]).layout.text.quoting == "all"
    assert csvio.parse_text_bytes(CASES["strings_only"]).layout.text.quoting == "strings-only"


def test_values_are_raw_strings() -> None:
    r = csvio.parse_text_bytes(CASES["na_tokens"])
    t = r.tables[0]
    assert t.columns[0].values.tolist() == ["NA", "", "null"]
    assert t.columns[1].values.tolist() == ["none", "None", "NULL"]
    r = csvio.parse_text_bytes(CASES["dup_blank_headers"])
    assert r.tables[0].header == ["id", "score", "score", "", "Unnamed"]
    r = csvio.parse_text_bytes(CASES["single_column_blanks"])
    assert r.tables[0].columns[0].values.tolist() == ["000123", "", "004567"]


def test_ragged_rows_fail() -> None:
    with pytest.raises(ReadError):
        csvio.parse_text_bytes(b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n")


def test_nul_bytes_fail() -> None:
    with pytest.raises(ReadError):
        csvio.parse_text_bytes(b"a\x00,b\x00\n1\x00,2\x00\n")


def test_ambiguous_full_quoting_resolved_from_file() -> None:
    # every quoted value needs quotes; other columns are minimal -> minimal
    data = b'id,comment\n1,"late, often"\n2,\n'
    r = csvio.parse_text_bytes(data)
    assert r.layout.text.column_quoting[1].policy == "minimal"


# --------------------------------------------------------------------------- property: writer/reader agree
import numpy as np  # noqa: E402
from hypothesis import assume, given, settings, strategies as st  # noqa: E402

from datanonymous.model import ColumnQuoting, FileLayout, TextDialect  # noqa: E402

_cell = st.text(alphabet=st.sampled_from(list('ab ,;"\n\r\t0123456789.-xyzAZ')), min_size=1, max_size=8)


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=2, max_value=4).flatmap(
    lambda k: st.tuples(st.just(k), st.lists(st.lists(_cell, min_size=k, max_size=k), min_size=1, max_size=6))),
    st.sampled_from(["\n", "\r\n"]), st.sampled_from(["minimal", "always", "nonnumeric"]))
def test_serialize_then_parse_recovers_values(table, le, policy) -> None:
    k, rows = table
    header = [f"h{i}" for i in range(k)]
    layout = FileLayout("csv", encoding="utf-8", text=TextDialect(delimiter=",", line_ending=le))
    cols = []
    for i in range(k):
        arr = np.empty(len(rows), dtype=object)
        arr[:] = [r[i] for r in rows]
        cols.append(arr)
    data = csvio.serialize(layout, header, cols, [ColumnQuoting(policy)] * k, [False] * k)
    back = csvio.parse_text_bytes(data, delimiter=",").tables[0]
    # K-17 deliberately interprets note-like final rows as layout. Random
    # minimal quoting can produce such a note (e.g. "0, A" after "0,0").
    # Its documented interpretation has a deterministic regression below.
    assume(not back.layout.get("_footer_rows"))
    assert back.header == header
    assert [c.values.tolist() for c in back.columns] == [c.tolist() for c in cols]


def test_note_like_last_record_is_footer_layout() -> None:
    result = csvio.parse_text_bytes(b"h0,h1\n0,0\n0, A\n", delimiter=",")
    table = result.tables[0]
    assert table.n_rows == 1
    assert [c.values.tolist() for c in table.columns] == [["0"], ["0"]]
    assert table.layout["_footer_rows"] == [(1, {1: "0, A"})]


# --------------------------------------------------------------------------- single-column quoting, bare quotes
def test_single_column_pandas_commas_roundtrip_and_quoting() -> None:
    import io

    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"course": ["Algebra, Part 1", "Biology", "Art, Studio", "Biology"] * 5})
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    out, r = roundtrip(data)
    assert out == data
    assert r.layout.text.delimiter is None and r.layout.text.extra_quote_chars == ","
    # generated values containing the delimiter are quoted as well: default pd.read_csv still parses the file
    vals = np.array(["X, Y", "Z", "W, V"], dtype=object)
    out2 = csvio.serialize(r.layout, ["course"], [vals], r.layout.text.column_quoting, r.layout.text.header_quoted)
    assert pd.read_csv(io.BytesIO(out2)).shape == (3, 1)


def test_single_column_unquoted_commas_stay_unquoted() -> None:
    data = b"name\nSmith, J\nLee\n"
    out, r = roundtrip(data)
    assert out == data and r.layout.text.extra_quote_chars == ""


@pytest.mark.parametrize("data", [b"id,height\n1,5'10\"\n2,6'1\"\n", b'a,b\n1, "x"\n2,y\n'])
def test_bare_quote_characters_roundtrip(data: bytes) -> None:
    out, r = roundtrip(data)
    assert out == data


def test_single_column_without_final_newline_never_ends_with_a_blank() -> None:
    import numpy as np

    r = csvio.parse_text_bytes(b"score\n1\n\n2")
    assert not r.layout.text.trailing_newline
    vals = np.array(["1", "2", ""], dtype=object)  # a blank landed in the last row
    out = csvio.serialize(r.layout, ["score"], [vals], r.layout.text.column_quoting, r.layout.text.header_quoted)
    back = csvio.parse_text_bytes(out)
    assert not back.layout.text.trailing_newline
    assert back.tables[0].n_rows == 3
    assert sorted(back.tables[0].columns[0].values.tolist()) == ["", "1", "2"]


# --------------------------------------------------------------------------- header plausibility
@pytest.mark.parametrize("data", [
    b"0414003,Philippa,Quackenbush,philippa.q@realschool.k12.us,766-16-2186\n"
    b"0123456,Ann,Lee,al@realschool.k12.us,123-45-6789\n",
    b"F,White,Quackenbush\nM,Black,Lee\nF,White,Diaz\nM,Hispanic,Ng\n",
    b"Contact Zebulon Abernathy at zebulon@realschool.k12.us for access.",
    b"This file lists every export made by the district last year\n1\n2\n",
])
def test_first_line_that_looks_like_data_fails_without_quoting_it(data: bytes) -> None:
    with pytest.raises(csvio.HeaderLooksLikeData) as ei:
        csvio.parse_text_bytes(data)
    msg = ei.value.message
    for word in ("Philippa", "Quackenbush", "realschool", "766", "Zebulon", "White", "district"):
        assert word not in msg
    assert ei.value.fix == csvio.HEADER_FIX


@pytest.mark.parametrize("data", [
    b"School,2019,2020\nA,10,12\nB,11,13\n",
    b"id,name\n1,a\nid,name\n2,b\n",
    b"q1,q2\n1,2\n",
    b"What is your favourite subject at school this year?,Grade\nMath,5\nArt,6\n",
    b"Other\nOther\n\nOther\n",
    b"student names\nAnn Lee\n",
])
def test_plausible_headers_are_accepted(data: bytes) -> None:
    csvio.parse_text_bytes(data)
