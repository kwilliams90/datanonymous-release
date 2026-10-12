"""AREA A hardening: adversarial inputs and regression tests for issues found in review.

Each test names the defect it pins down. Privacy tests check that values which could identify someone
(IDs, phone numbers, e-mail addresses, free text) left over in a date column are never copied to the
output or the report, while harmless placeholders keep the shape code relies on.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from datanonymous.detect import dates as D
from datanonymous.detect import dates_formats as F
from datanonymous.generate import dates as G
from datanonymous.model import ColumnType as T
from datanonymous.profile.stats import numeric_view

from conftest import run_cli
from test_area_a_support import detect, parse_all, profile_of, run_table

BASE = dt.date(2023, 1, 2)
RNG = np.random.default_rng(11)
DAYS = [BASE + dt.timedelta(days=int(x)) for x in RNG.integers(0, 400, 400)]


def synth(values: list[str], name: str = "col", seed: int = 1, **cli):
    tp, cols = run_table({name: values}, seed=seed, **cli)
    return profile_of(tp, name), list(cols[name])


def iso_dates(n: int = 400) -> list[str]:
    """``n`` distinct ISO dates (so a handful of leftovers stay under 1% of the distinct values)."""
    return [(BASE + dt.timedelta(days=i)).isoformat() for i in range(n)]


# --------------------------------------------------------------------------- privacy: unparsed leftovers
@pytest.mark.parametrize("secret", ["104023", "S104023", "20230512", "555-867-5309", "jdoe@example.org", "123-45-6789",
                                    "١٢٣٤٥"])
def test_identifier_like_leftovers_are_never_copied(secret: str) -> None:
    """Regression: any short leftover (an ID, a phone number, an e-mail) used to be written verbatim."""
    vals = iso_dates() + [secret]
    det = detect(vals, name="visit_date")
    assert det.type is T.DATE
    assert secret not in (det.params.get("_other_tokens") or [])
    assert det.params.get("unparsed_replaced") == 1
    p, out = synth(vals, name="visit_date")
    assert secret not in out and len(out) == len(vals)
    v, kept = parse_all(p, out)
    assert np.all(v.cls >= 0)  # the row got a generated date instead


@pytest.mark.parametrize("token,kind", [("TBD", "word"), ("Unknown", "word"), ("not enrolled", "word"),
                                        ("0", "word"), ("n/a", "word"), ("2023-02-30", "date"),
                                        ("9999-99-99", "date"), ("15MAR2024", "date")])
def test_placeholder_and_out_of_range_leftovers_stay_tokens(token: str, kind: str) -> None:
    """Placeholders, out-of-range dates (section 4) and rare valid layouts are kept verbatim at their rates."""
    assert D.leftover_kinds([token], ["%Y-%m-%d"]) == [kind]
    vals = iso_dates() + [token] * 2
    p, out = synth(vals, name="d")
    assert p.type is T.DATE and p.params["other_tokens"] == 2
    assert out.count(token) == 2


def test_leftover_kinds_rules() -> None:
    keys = ["%m/%d/%Y"]
    got = D.leftover_kinds(["99/99/9999", "02/30/2024", "104023", "Jane Q. Public", "TBD", "x@y.z", "000",
                            "a note that is much too long to be a placeholder", "Pending review", "-1"], keys)
    assert got == ["date", "date", None, None, "word", None, "word", None, "word", None]
    assert D.leftover_kinds([], keys) == []
    # loose special shapes (ISO durations) never make an ID-like code a token; valid values still are
    assert D.leftover_kinds(["A1234567B", "P3D", "20230512"], ["@dur_iso"]) == [None, "date", None]


def test_leftovers_never_reach_out_dir_or_report(tmp_path: Path) -> None:
    """End to end: identifier-like leftovers in a date column appear nowhere; placeholders keep their counts."""
    rng = np.random.default_rng(5)
    rows = [[f"S{100000 + i}", (dt.date(2021, 1, 1) + dt.timedelta(days=int(rng.integers(0, 1500)))).strftime(
        "%m/%d/%Y")] for i in range(900)]  # ~650 distinct dates: 5 leftovers stay under 1%
    secrets = {5: "S100099", 6: "555-867-5309", 7: "parent.name@example.com"}
    for i, s in secrets.items():
        rows[i][1] = s
    rows[8][1] = "TBD"
    rows[9][1] = "02/30/2023"
    src = tmp_path / "in" / "v.csv"
    src.parent.mkdir()
    lines = ["student_id,visit_date"] + [",".join(r) for r in rows]
    src.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(src.parent, "--out-dir", out, "--report-dir", rep, "--seed", "4")
    assert res.exit_code == 0, res.output
    data = (out / "v.csv").read_bytes()
    assert data.count(b"\r\n") == 901  # CRLF kept
    syn = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    assert len(syn) == 901
    col = [r[1] for r in syn[1:]]
    assert col.count("TBD") == 1 and col.count("02/30/2023") == 1
    for path in [p for p in out.rglob("*") if p.is_file()] + [p for p in rep.rglob("*") if p.is_file()]:
        text = path.read_text(encoding="utf-8")
        for s in list(secrets.values()) + ["S100099"]:
            assert s not in text, (path.name, s)
    report = json.loads((rep / "files" / "v.csv.json").read_text(encoding="utf-8"))
    params = {c["name"]: c["params"] for c in report["tables"][0]["columns"]}["visit_date"]
    assert params["other_tokens"] == 2 and params["unparsed_replaced"] == 3 and "_other_tokens" not in params
    assert "TBD" not in json.dumps(report)  # leftover token values stay private


# --------------------------------------------------------------------------- detection: h:mm[:ss]
@pytest.mark.parametrize("values", [
    [f"{a}:{b}" for a in range(10, 24) for b in range(1, 4)] * 3,  # student:teacher ratios 15:1
    [f"{a}:{b}" for a in range(0, 8) for b in range(0, 8)] * 3,  # game scores 3:2
])
def test_ratios_and_scores_are_not_times(values: list[str]) -> None:
    """Regression: one-digit minutes were accepted, so ratio/score columns became TIME and were regenerated."""
    assert detect(values, name="ratio").type not in D.DATE_TYPES


def test_minutes_and_seconds_need_two_digits() -> None:
    assert F.get_format("%H:%M").parse(["8:05", "8:5", "08:05"]).ok.tolist() == [True, False, True]
    assert F.get_format("%Y-%m-%d %H:%M:%S").parse(["2024-01-01 8:05:09", "2024-01-01 8:5:9"]).ok.tolist() == [
        True, False]
    assert F.get_format("%I:%M %p").parse(["7:45 AM", "7:4 AM"]).ok.tolist() == [True, False]
    # the hour stays flexible (8:30 and 08:30), the render spec keeps minutes padded
    det = detect([f"{h}:{m:02d}" for h in range(1, 12) for m in range(0, 60, 7)], name="t")
    assert det.type is T.TIME and det.params["formats"][0]["render"]["pad"] == {"H": False, "M": True}


def test_three_way_ambiguity_names_every_reading() -> None:
    """%y/%m/%d values with every part <= 12 read three ways; the warning names all of them."""
    vals = [f"{y:02d}/{m:02d}/{d:02d}" for y in range(10, 13) for m in range(1, 13) for d in range(1, 13)]
    det = detect(vals, name="d")
    assert det.ambiguous and "%d/%m/%y" in det.ambiguity and "%y/%m/%d" in det.ambiguity
    warn = next(m for m in det.messages if m.code == "ambiguous_date_order")
    assert warn.fix and "--dayfirst d" in warn.fix and "--date-format d=FMT" in warn.fix
    assert "10/" not in warn.text  # messages name formats, never values


# --------------------------------------------------------------------------- detection: TIME vs DURATION layouts
def test_mixed_clock_layouts_stay_time() -> None:
    """Regression: h:mm mixed with h:mm:ss (every hour < 24) was claimed by the any-layout duration parser."""
    vals = [f"{h:02d}:{m:02d}" for h in range(7, 18) for m in range(0, 60, 5)]
    vals += [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(range(7, 18), range(3, 60, 5), range(11, 60, 4))] * 5
    det = detect(vals, name="arrival")
    assert det.type is T.TIME
    assert [f["format"] for f in det.params["formats"]] == ["%H:%M", "%H:%M:%S"]
    p, out = synth(vals, name="arrival")
    assert sum(len(v) == 5 for v in out) == sum(len(v) == 5 for v in vals)  # layout shares kept exactly
    real, syn = numeric_view(p, vals), numeric_view(p, out)
    assert (syn.min(), syn.max()) == (real.min(), real.max())  # D14 in a layout that can write them


def test_mixed_duration_layouts_keep_shares_and_extremes() -> None:
    rng = np.random.default_rng(8)
    vals = []
    for i in range(300):
        secs = int(rng.integers(0, 50 * 3600))
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        vals.append(f"{h}:{m:02d}" if i % 3 else f"{h}:{m:02d}:{s:02d}")
    vals.append("49:59:59")  # the exact max needs seconds
    det = detect(vals, name="x")
    assert det.type is T.DURATION
    assert {f["format"] for f in det.params["formats"]} == {"@dur_hm", "@dur_hms"}
    p, out = synth(vals, name="x")
    assert Counter(v.count(":") for v in out) == Counter(v.count(":") for v in vals)
    real, syn = numeric_view(p, vals), numeric_view(p, out)
    assert (syn.min(), syn.max()) == (real.min(), real.max())


def test_clock_duration_keys() -> None:
    assert F.get_format("@dur_hm").parse(["25:00", "25:00:00"]).ok.tolist() == [True, False]
    assert F.get_format("@dur_hms").parse(["25:00", "25:00:00", "25:00:00.5"]).ok.tolist() == [False, True, False]
    assert F.get_format("@dur_hmsf").parse(["25:00:00", "25:00:00.5"]).ok.tolist() == [False, True]
    assert F.get_format("@dur_clock").parse(["25:00", "25:00:00", "25:00:00.5"]).ok.all()  # --date-format
    assert detect(["1:30", "26:15", "0:45"] * 5, name="x").params["formats"][0]["format"] == "@dur_hm"


# --------------------------------------------------------------------------- generation: mixed resolutions
def test_mixed_date_and_datetime_rows_stay_in_bounds_with_exact_extremes() -> None:
    """Regression: a date-only row rendered on the minimum day fell below the real minimum instant, and the
    exact minimum (with a time) could be placed in a date-only row and lost (D14)."""
    rng = np.random.default_rng(45)
    vals = []
    for i in range(12):
        t = dt.datetime(2020, 1, 1) + dt.timedelta(days=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 86400)))
        vals.append(t.strftime("%d.%m.%Y") if i % 2 else t.strftime("%Y-%m-%d %H:%M:%S"))
    vals.append("2019-12-31 12:10:00")  # the minimum has a time of day
    for seed in range(15):
        p, out = synth(vals, name="stamp", seed=seed)
        real, syn = numeric_view(p, vals), numeric_view(p, out)
        assert syn.min() == real.min() and syn.max() == real.max(), seed
        assert "2019-12-31 12:10:00" in out


def test_fit_to_formats_and_constrained_extremes() -> None:
    day = F.US_DAY
    lo, hi = 10 * day + 3600 * F.US, 20 * day
    y = np.array([10 * day + 7200 * F.US, 15 * day + 5, 10 * day + 3700 * F.US], dtype=np.int64)
    res = np.array([day, 60 * F.US, F.US], dtype=np.int64)
    fit = G.fit_to_formats(y, res, lo, hi)
    assert fit[0] == 11 * day and fit[1] == 15 * day and fit[2] == 10 * day + 3700 * F.US
    assert np.all((fit >= lo) & (fit <= hi))
    # the minimum (10 days + 1 h) cannot be written by the date-only row: it goes to a row with a time
    assert G.ensure_extremes(fit, lo, hi, res)
    assert fit.tolist() == [11 * day, hi, lo]
    # durations truncate toward zero
    d = G.fit_to_formats(np.array([-1830 * F.US, 1830 * F.US]), np.array([60 * F.US] * 2), -1860 * F.US,
                         1860 * F.US, toward_zero=True)
    assert d.tolist() == [-1800 * F.US, 1800 * F.US]


# --------------------------------------------------------------------------- detection: org codes are not serials
@pytest.mark.parametrize("name", ["EntryCode", "Exit_Code", "entry_code"])
def test_org_code_integers_are_not_excel_serials(name: str) -> None:
    """Regression: 5-digit entry/exit codes (D10 org codes) were read as Excel day serials."""
    vals = [str(v) for v in (12001, 12002, 13050, 21000, 30500)] * 20
    assert detect(vals, name=name).type is T.ORG_CODE
    assert detect(vals, name=name, date_cols=[name]).subtype == "excel_serial"  # the explicit flag still works
    assert detect(vals, name="entry_date").subtype == "excel_serial"


# --------------------------------------------------------------------------- docs: synthetic only (D20)
def test_doc_fields_use_synthetic_statistics() -> None:
    vals = [d.isoformat() if i % 20 < 11 else d.strftime("%m/%d/%Y") for i, d in enumerate(DAYS)] + ["0/0/0"] * 3
    p, out = synth(vals, name="d")
    fake_syn = {"format_shares": {"%Y-%m-%d": 0.25, "%m/%d/%Y": 0.75}, "sentinels": {"0/0/0": 3}}
    fields = D.doc_fields(p, fake_syn)
    assert fields["formats"] == {"%Y-%m-%d": 0.25, "%m/%d/%Y": 0.75}
    assert fields["sentinel_tokens"] == ["0/0/0"]
    assert "sentinel_tokens" not in D.doc_fields(p, {})  # nothing from the real params


def test_epoch_integers_are_posixct_in_r() -> None:
    p, _ = synth([str(1_700_000_000 + 3600 * i) for i in range(100)], name="updated")
    assert p.subtype == "epoch_s" and D.doc_fields(p, p.stats)["r_class"] == "POSIXct"
    p2, _ = synth([d.strftime("%Y%m%d") for d in DAYS], name="enroll")
    assert "r_class" not in D.doc_fields(p2, p2.stats)  # YYYYMMDD integers keep the type's Date class


def test_zone_name_note_for_r() -> None:
    p, _ = synth([f"2024-03-{d:02d} 13:45:00 {'EST' if d % 2 else 'CST'}" for d in range(1, 29)], name="ts")
    fields = D.doc_fields(p, p.stats)
    assert fields["readr"] == "c" and "%Z" in fields["note"] and "tz =" in fields["note"]


# --------------------------------------------------------------------------- native contract: Excel cells
@pytest.mark.parametrize("number_format,kind", [
    ("mm/dd/yyyy", "date"), ("mm/dd/yyyy h:mm", "datetime"), ("h:mm AM/PM", "time"), ("hh:mm:ss", "time"),
    ("[h]:mm:ss", "duration"), ("[$-409]mmmm d, yyyy", "date"), ("mmm-yy", "date"), ("General", "date"),
    ('yyyy-mm-dd"T"hh:mm', "datetime"),
])
def test_excel_number_format_kinds(number_format: str, kind: str) -> None:
    assert D.native_temporal({"cell_type": "d", "number_format": number_format})["kind"] == kind


def test_excel_time_only_cells_are_detected_as_time() -> None:
    native = {"cell_type": "d", "number_format": "h:mm AM/PM"}
    det = detect([f"{h:02d}:{m:02d}:00" for h in range(7, 16) for m in (0, 15, 30, 45)], name="start", native=native)
    assert det.type is T.TIME and det.source == "native"


# --------------------------------------------------------------------------- edge-case columns
@pytest.mark.parametrize("values,expected", [
    ([], T.EMPTY),
    (["", "", "NA"], T.EMPTY),
    (["2024-01-01"], T.CONSTANT),
    (["2024-01-01", ""], T.FLAG_SINGLE_TOKEN),
    (["2024-01-01", "2024-01-02"], T.DATE),
])
def test_tiny_columns(values: list[str], expected: T) -> None:
    det = detect(values, name="d")
    assert det.type is expected
    if values:
        p, out = synth(values, name="d")
        assert len(out) == len(values)
        assert Counter(v for v in out if v in ("", "NA")) == Counter(v for v in values if v in ("", "NA"))


def test_one_date_row_among_missing_and_sentinels() -> None:
    vals = ["2024-03-15", "1900-01-01", "1900-01-01", "", "NA", "0000-00-00"]
    p, out = synth(vals, name="d")
    assert p.type is T.DATE and len(out) == len(vals)
    assert Counter(out) == Counter(vals)  # every sentinel, blank and NA at its exact count; the date is the only one


def test_odd_and_huge_values() -> None:
    far = iso_dates() + ["9999-01-01"]  # one far value under 1% of rows: a real extreme, not a token
    p, out = synth(far, name="d")
    assert "9999-01-01" in out and max(out) == "9999-01-01"
    assert all(len(v) == 10 for v in out)
    big = [f"{h}:{m:02d}:{s:02d}" for h, m, s in zip(range(999_990, 1_000_000), range(10, 20), range(30, 40))]
    p2, out2 = synth(big * 3, name="elapsed")
    assert p2.type is T.DURATION and set(out2) <= set(big) | set(out2)
    assert max(numeric_view(p2, out2)) == max(numeric_view(p2, big))
    neg = ["-1:30:00", "25:00:00", "2:00:00", "-0:15:30"] * 4
    p3, out3 = synth(neg, name="delta")
    assert p3.type is T.DURATION and min(numeric_view(p3, out3)) == -5400.0


def test_unicode_and_whitespace_variants() -> None:
    vals = [d.isoformat() + " " for d in DAYS[:40]] + ["　" + d.isoformat() for d in DAYS[40:60]] + [
        d.isoformat() for d in DAYS[60:]]
    p, out = synth(vals, name="d")
    assert p.type is T.DATE
    assert sum(v.endswith(" ") for v in out) == 40 and sum(v.startswith("　") for v in out) == 20
    v, _ = parse_all(p, out)
    assert np.all(v.cls >= 0)
    # full-width digits are not ASCII dates: the column is not a date
    assert detect(["２０２４-01-0" + str(i) for i in range(1, 10)], name="d").type not in D.DATE_TYPES


def test_determinism_and_row_count_with_every_token_kind() -> None:
    vals = ([d.strftime("%m/%d/%Y") for d in DAYS] + ["1/1/1900"] * 5 + ["0/0/0"] * 4 + ["TBD"] * 3
            + ["02/30/2024"] + [""] * 30 + ["NULL"] * 6 + [" " + DAYS[0].strftime("%m/%d/%Y")] * 9)
    a = synth(vals, name="exit_date", seed=21)[1]
    b = synth(vals, name="exit_date", seed=21)[1]
    c = synth(vals, name="exit_date", seed=22)[1]
    assert a == b and a != c and len(a) == len(vals)
    for tok in ("1/1/1900", "0/0/0", "TBD", "02/30/2024", "", "NULL"):
        assert a.count(tok) == vals.count(tok), tok
    assert sum(v.startswith(" ") for v in a) == 9
