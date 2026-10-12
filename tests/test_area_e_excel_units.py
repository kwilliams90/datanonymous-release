"""AREA E unit tests: canonical values, header/footer heuristics, typing, placeholders, footer formulas."""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pytest

from datanonymous.io import excel_read as xr
from datanonymous.io import excel_write as xw
from datanonymous.io.excel_read import ColBuf, SheetScan, XCell


def s(v: str) -> XCell:
    return XCell(v, "s", "General")


def n(v: float, fmt: str = "General") -> XCell:
    return XCell(v, "n", fmt)


# --------------------------------------------------------------------------- canonical values
@pytest.mark.parametrize("v, want", [
    (5, "5"), (-12, "-12"), (3.0, "3"), (0.1, "0.1"), (2.75, "2.75"), (1.5e-07, "0.00000015"), (-0.0, "0"),
    (653396000000.0, "653396000000"), (True, "TRUE"), (False, "FALSE"),
])
def test_canon_number(v, want) -> None:
    assert xr.canon_number(v) == want


def test_canon_dates_per_kind() -> None:
    d = dt.datetime(2024, 3, 1)
    assert xr.canon_date(d, "date") == "2024-03-01"
    assert xr.canon_date(dt.datetime(2024, 3, 1, 8, 30, 15, 600000), "datetime") == "2024-03-01 08:30:16"
    assert xr.canon_date(dt.time(7, 5), "time") == "07:05:00"
    assert xr.canon_date(dt.timedelta(hours=25, minutes=3), "duration") == "25:03:00"
    assert xr.canon_date(-dt.timedelta(minutes=90), "duration") == "-1:30:00"


def test_date_kind_decided_per_column() -> None:
    buf = ColBuf()
    for i, v in enumerate([dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)]):
        buf.put(i, XCell(v, "d", "mm/dd/yyyy"))
    vals, nat = xr.finalize_column(buf, 3)
    assert vals.tolist() == ["2024-01-02", "2024-01-03", ""]
    assert nat["date_kind"] == "date" and nat["main"] == "d" and nat["formats"]["d"] == "mm/dd/yyyy"
    buf2 = ColBuf()
    buf2.put(0, XCell(dt.datetime(2024, 1, 2), "d", "m/d/yyyy h:mm"))
    vals2, nat2 = xr.finalize_column(buf2, 1)
    assert nat2["date_kind"] == "datetime" and vals2.tolist() == ["2024-01-02 00:00:00"]


def test_finalize_native_mixed_column_and_errors() -> None:
    buf = ColBuf()
    cells = [s("KG"), n(1), n(2), s("KG"), XCell("#N/A", "e", "General"), n(12)]
    for i, c in enumerate(cells):
        buf.put(i, c)
    vals, nat = xr.finalize_column(buf, len(cells), formulas=2)
    assert vals.tolist() == ["KG", "1", "2", "KG", "#N/A", "12"]
    assert nat["types"] == {"s": 2, "n": 3, "e": 1} and nat["main"] == "n"
    assert nat["errors"] == ["#N/A"] and nat["formulas"] == 2
    assert nat["value_types"] == {"KG": "s", "1": "n", "2": "n", "12": "n"}


# --------------------------------------------------------------------------- header heuristic
def rows_of(*rows: list) -> list[tuple[int, dict[int, XCell]]]:
    out = []
    for r, row in enumerate(rows, 1):
        out.append((r, {c: (s(v) if isinstance(v, str) else n(v)) for c, v in enumerate(row, 1) if v is not None}))
    return out


def test_header_after_title_rows() -> None:
    rows = rows_of(["District Report"], ["Printed 10/01/2024"], [], ["id", "name", "grade", "score"],
                   ["001", "A", "KG", 5], ["002", "B", "01", 7])
    assert xr.detect_header_row(rows) == 4


def test_header_needs_text_unique_and_wide_row_after() -> None:
    # a numeric row is never the header; duplicate labels are not a header
    rows = rows_of([1, 2, 3, 4], ["a", "a", "b", "c"], ["id", "x", "y", "z"], ["1", 2, 3, 4])
    assert xr.detect_header_row(rows) == 3
    # no wide row after: the fallback still takes the first header-like row (header-only table)
    assert xr.detect_header_row(rows_of(["Title"], [], ["a", "b", "c"])) == 3
    assert xr.detect_header_row(rows_of([], [])) is None


def scan(rows: list[list], formulas: dict[int, dict[int, str]] | None = None, **kw):
    sc = SheetScan("S", **kw)
    for r, cells in rows_of(*rows):
        sc.feed(r, cells)
    sx = xr.SheetXml()
    for r, cols in (formulas or {}).items():
        sx.formula_tail.append((r, cols))
    return sc.finish(sx)


def test_footer_total_row_with_formula_and_note() -> None:
    rows = [["id", "score", "credits"]] + [[f"00{i}", 50 + i, 1.5] for i in range(5)]
    rows += [[None, None, 7.5], [], ["Source: nightly export"]]
    out = scan(rows, formulas={7: {3: "=SUM(C2:C6)"}})
    assert out.header_row == 1 and out.n_rows == 5 and out.data_end == 6
    assert [r for r, _ in out.footer_rows] == [7, 9]


def test_footer_label_row_and_no_false_footers() -> None:
    rows = [["id", "score"]] + [[f"00{i}", i] for i in range(4)] + [["Grand Total", 6]]
    out = scan(rows)
    assert out.n_rows == 4 and [r for r, _ in out.footer_rows] == [6]
    # a last data row with one text cell and no blank row before it is data, not a note
    rows2 = [["id", "score"]] + [[f"00{i}", i] for i in range(4)] + [["009"]]
    out2 = scan(rows2)
    assert out2.n_rows == 5 and not out2.footer_rows
    # single-column tables never get note rows
    rows3 = [["name"], ["a"], ["b"], [], ["c"]]
    assert scan(rows3).n_rows == 4


def test_trailing_blank_rows_are_not_data() -> None:
    rows = [["a", "b"], [1, 2], [None, 3], [], [], []]
    out = scan(rows)
    assert out.n_rows == 2 and out.first_col == 1 and out.last_col == 2


def test_region_and_header_override() -> None:
    rows = [["x"], ["a", "b", "c", "zzz"], [1, 2, 3, 9], [4, 5, 6, 9], ["below"]]
    out = scan(rows, region=xr.Region(1, 2, 3, 4))
    assert out.header_row == 2 and out.n_rows == 2 and out.last_col == 3
    assert out.dropped_below == 1 and out.dropped_side == 3
    out2 = scan([["junk", "row"], ["a", "b"], [1, 2]], header_override=2)
    assert out2.header_row == 2 and out2.n_rows == 1 and len(out2.title_rows) == 1


def test_parse_ranges() -> None:
    r = xr.parse_ranges(["'My Sheet'!A5:H200", "B2:C9"])
    assert r["My Sheet"] == xr.Region(1, 5, 8, 200) and r[None] == xr.Region(2, 2, 3, 9)
    from datanonymous.errors import UsageError

    with pytest.raises(UsageError):
        xr.parse_ranges(["Sheet1!nonsense"])


# --------------------------------------------------------------------------- writer typing
NAT_TEXT = {"kind": "excel", "types": {"s": 3}, "main": "s", "formats": {"s": "@"}, "date_kind": None,
            "errors": [], "formulas": 0}
NAT_NUM = {"kind": "excel", "types": {"n": 5, "e": 1}, "main": "n", "formats": {"n": "0.00"}, "date_kind": None,
           "errors": ["#N/A"], "formulas": 0}


def test_typed_value_keeps_text_text_and_numbers_numbers() -> None:
    assert xw.typed_value("000123", NAT_TEXT) == ("000123", "s")
    assert xw.typed_value("=HYPERLINK(1)", NAT_TEXT) == ("=HYPERLINK(1)", "s")
    assert xw.typed_value("12", NAT_NUM) == (12, "n")
    assert xw.typed_value("3.25", NAT_NUM) == (3.25, "n")
    assert xw.typed_value("#N/A", NAT_NUM) == ("#N/A", "e")
    assert xw.typed_value("*", NAT_NUM) == ("*", "s")  # cannot be a number: text cell
    assert xw.typed_value("", NAT_NUM) == (None, "")
    assert xw.typed_value(None, NAT_NUM) == (None, "")
    assert xw.typed_value("x", None) == ("x", "s")


def test_typed_value_dates_booleans_and_mixed() -> None:
    nat_d = {"types": {"d": 2}, "main": "d", "formats": {"d": "mm/dd/yyyy"}, "date_kind": "date", "errors": []}
    assert xw.typed_value("2024-03-01", nat_d) == (dt.datetime(2024, 3, 1), "d")
    nat_t = dict(nat_d, date_kind="time")
    assert xw.typed_value("08:30:00", nat_t) == (dt.time(8, 30), "d")
    nat_u = dict(nat_d, date_kind="duration")
    assert xw.typed_value("25:03:00", nat_u) == (dt.timedelta(hours=25, minutes=3), "d")
    nat_b = {"types": {"b": 2}, "main": "b", "formats": {}, "errors": []}
    assert xw.typed_value("TRUE", nat_b) == (True, "b") and xw.typed_value("FALSE", nat_b) == (False, "b")
    nat_m = {"types": {"s": 1, "n": 3}, "main": "n", "formats": {}, "errors": [],
             "value_types": {"KG": "s", "1": "n", "05": "s"}}
    assert xw.typed_value("KG", nat_m) == ("KG", "s")
    assert xw.typed_value("05", nat_m) == ("05", "s")  # the source stored this level as text
    assert xw.typed_value("7", nat_m) == (7, "n")  # unseen numeric value: number


# --------------------------------------------------------------------------- placeholders
def shape(text: str) -> str:
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", text)))


def test_placeholder_same_shape_and_synthetic_dates() -> None:
    ph = xw.Placeholders(np.random.default_rng(1), keep=False)
    src = "Printed by jsmith on 10/01/2024 08:15 AM (Room B-12)"
    out = ph.text(src)
    assert out != src and len(out) == len(src) and shape(out) == shape(src)
    assert "jsmith" not in out
    m = re.search(r"(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d) ([AP]M)", out)
    assert m and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 12
    assert ph.text(src) == out  # same text, same placeholder within a workbook
    iso = ph.text("As of 2024-09-30")
    assert re.fullmatch(r"[A-Za-z]{2} [a-z]{2} \d{4}-\d{2}-\d{2}", iso)
    dt.date.fromisoformat(iso[-10:])
    month = ph.text("Fall Report, September 3, 2024")
    assert re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December) "
                     r"\d{1,2}, \d{4}$", month)


def test_placeholder_values_and_keep() -> None:
    ph = xw.Placeholders(np.random.default_rng(2), keep=False)
    v = ph.value(n(4521))
    assert isinstance(v, int) and 1000 <= v <= 9999
    f = ph.value(n(12.75))
    assert isinstance(f, float) and len(xr.canon_number(f).split(".")[0]) == 2
    d = ph.value(XCell(dt.datetime(2024, 10, 1, 8, 15), "d", "mm/dd/yyyy h:mm"))
    assert isinstance(d, dt.datetime) and d != dt.datetime(2024, 10, 1, 8, 15)
    keep = xw.Placeholders(np.random.default_rng(2), keep=True)
    assert keep.text("Printed by jsmith") == "Printed by jsmith" and keep.value(n(4521)) == 4521


def test_placeholders_are_seeded() -> None:
    a = xw.Placeholders(np.random.default_rng(7), False).text("Report for Lincoln ES 2024")
    b = xw.Placeholders(np.random.default_rng(7), False).text("Report for Lincoln ES 2024")
    assert a == b


# --------------------------------------------------------------------------- footer formulas
def test_parse_aggregate_forms() -> None:
    assert xw.parse_aggregate("=SUM(E5:E44)") == ("SUM", [(5, 5, 5, 44)])
    assert xw.parse_aggregate("=sum($E$5:$E$44, F5:F44)") == ("SUM", [(5, 5, 5, 44), (6, 5, 6, 44)])
    assert xw.parse_aggregate("=SUBTOTAL(9,C2:C9)") == ("SUM", [(3, 2, 3, 9)])
    assert xw.parse_aggregate("=SUBTOTAL(101,C2:C9)") == ("AVERAGE", [(3, 2, 3, 9)])
    assert xw.parse_aggregate("=COUNTA(B:B)") == ("COUNTA", [(2, 1, 2, 0)])
    for f in ("=SUM(Sheet2!A1:A9)", "=SUM(IF(A1:A9>0,1))", "=MAX(A1:A9)", "=SUBTOTAL(4,A1:A9)", "=A1+A2", None):
        assert xw.parse_aggregate(f) is None


def test_evaluate_aggregate_semantics() -> None:
    vals = [1, 2.5, None, "x", True, dt.datetime(2024, 1, 1)]
    assert xw.evaluate_aggregate("COUNTA", vals) == 5
    assert xw.evaluate_aggregate("COUNT", vals) == 3  # numbers and dates
    assert xw.evaluate_aggregate("SUM", [1, 2.5, None, "x"]) == 3.5
    assert xw.evaluate_aggregate("AVERAGE", [1, 2, None]) == 1.5
    assert xw.evaluate_aggregate("AVERAGE", [None, "x"]) == "#DIV/0!"
    assert xw.evaluate_aggregate("SUM", [1, "#N/A"]) == "#N/A"  # errors propagate as in Excel
    assert xw.evaluate_aggregate("SUM", [dt.datetime(2024, 1, 1)]) is xw._NO_VALUE


def test_parse_date_forms() -> None:
    assert xw.parse_date("2024-03-01 08:30:00", "datetime") == dt.datetime(2024, 3, 1, 8, 30)
    assert xw.parse_date("2024-03-01T08:30:00+05:00", "datetime") == dt.datetime(2024, 3, 1, 8, 30)
    assert xw.parse_date("not a date", "date") is None
    assert xw.parse_date("-1:30:00", "duration") == -dt.timedelta(minutes=90)
