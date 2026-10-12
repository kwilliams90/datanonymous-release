"""AREA A detection: one table-driven test per rule of design 3 #12, #19-24, section 4 and section 10."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from datanonymous.detect import dates as D
from datanonymous.detect import dates_formats as F
from datanonymous.errors import UnsupportedColumnError, UsageError
from datanonymous.model import ColumnType as T

from test_area_a_support import ctx_for, days, detect

RNG = np.random.default_rng(42)
BASE = dt.date(2023, 8, 14)
DAYS = days(BASE, 400, RNG)


def fmt_of(det) -> list[str]:
    return [f["format"] for f in det.params["formats"]]


def render(det, key: str = "pad") -> dict:
    return det.params["formats"][0]["render"].get(key, {})


# --------------------------------------------------------------------------- the format catalog
def test_candidate_catalog_is_explicit_and_large() -> None:
    assert len(F.CANDIDATES) >= 38
    assert len(set(F.CANDIDATES)) == len(F.CANDIDATES)
    for key in F.CANDIDATES:
        f = F.get_format(key)
        assert f.kind in ("date", "datetime", "year_month", "time", "duration"), key
    # preference order: ISO first, month-first before day-first
    assert F.CANDIDATES.index("%Y-%m-%d") == 0
    assert F.CANDIDATES.index("%m/%d/%Y") < F.CANDIDATES.index("%d/%m/%Y")


@pytest.mark.parametrize("bad", ["%Q", "%Y-%Y", "%I:%M", "%d/%m", "%H:%M %z", "50%"])
def test_invalid_formats_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        F.get_format(bad)


def test_flexible_and_compact_widths() -> None:
    f = F.get_format("%m/%d/%Y")
    p = f.parse(["3/5/2024", "03/05/2024", "12/31/2024", "13/01/2024", "2/30/2024", "2/29/2024", "2/29/2023"])
    assert p.ok.tolist() == [True, True, True, False, False, True, False]
    g = F.get_format("%Y%m%d")  # compact: exactly 8 digits (pandas' %Y%m%d accepts the 7-digit 2024315)
    assert g.parse(["20240315", "2024315", "202403150"]).ok.tolist() == [True, False, False]


def test_parse_components_and_calendar_math() -> None:
    p = F.get_format("%d-%b-%y %I.%M.%S.%f %p").parse(["14-AUG-24 01.45.07.123000 PM", "01-jan-70 12.00.00.5 AM"])
    assert p.ok.all()
    assert (p.year.tolist(), p.month.tolist(), p.day.tolist()) == ([2024, 1970], [8, 1], [14, 1])
    assert (p.hour.tolist(), p.minute.tolist(), p.second.tolist()) == ([13, 0], [45, 0], [7, 0])
    assert p.micro.tolist() == [123000, 500000]
    ords = np.arange(-800000, 3000000, 997)
    y, m, d = F.civil_from_days(ords)
    assert np.array_equal(F.days_from_civil(y, m, d), ords)
    assert F.iso_day(F.ordinal(2024, 2, 29)) == "2024-02-29"
    assert F.weekday(np.array([F.ordinal(2024, 3, 18)]))[0] == 0  # a Monday


# --------------------------------------------------------------------------- DATE
def test_iso_and_padding_render_spec() -> None:
    det = detect([d.isoformat() for d in DAYS])
    assert det.type is T.DATE and fmt_of(det) == ["%Y-%m-%d"]
    assert render(det) == {"d": True, "m": True}
    unpadded = detect([f"{d.month}/{d.day}/{d.year}" for d in DAYS])
    assert fmt_of(unpadded) == ["%m/%d/%Y"] and render(unpadded) == {"d": False, "m": False}
    padded = detect([d.strftime("%m/%d/%Y") for d in DAYS])
    assert render(padded) == {"d": True, "m": True}
    assert not padded.ambiguous


def test_day_first_detected_when_a_day_exceeds_12() -> None:
    det = detect([d.strftime("%d/%m/%Y") for d in DAYS])
    assert det.type is T.DATE and fmt_of(det) == ["%d/%m/%Y"] and not det.ambiguous
    assert detect([d.strftime("%d.%m.%Y") for d in DAYS]).params["formats"][0]["format"] == "%d.%m.%Y"


def test_ambiguous_month_day_defaults_to_month_first_with_warning() -> None:
    vals = [f"{int(m):02d}/{int(d):02d}/2024" for m, d in zip(RNG.integers(1, 13, 200), RNG.integers(1, 13, 200))]
    det = detect(vals, name="visit")
    assert det.ambiguous and "month/day" in det.ambiguity
    assert fmt_of(det) == ["%m/%d/%Y"]
    warn = [m for m in det.messages if m.code == "ambiguous_date_order"]
    assert warn and warn[0].level == "warning" and "--dayfirst visit" in warn[0].fix
    first = detect(vals, name="visit", dayfirst=["visit"])
    assert fmt_of(first) == ["%d/%m/%Y"] and not first.ambiguous and first.params.get("dayfirst")
    # --dayfirst on an unambiguous column has no effect (info message)
    eu = detect([d.strftime("%m/%d/%Y") for d in DAYS], name="x", dayfirst=["x"])
    assert fmt_of(eu) == ["%m/%d/%Y"] and any(m.code == "dayfirst_unused" for m in eu.messages)


def test_mixed_formats_greedy_cover() -> None:
    vals = [d.isoformat() if i % 20 < 11 else d.strftime("%m/%d/%Y") for i, d in enumerate(DAYS)]
    det = detect(vals)
    assert det.type is T.DATE_MIXED_FORMATS
    shares = {f["format"]: f["share"] for f in det.params["formats"]}
    assert set(shares) == {"%Y-%m-%d", "%m/%d/%Y"}
    assert shares["%Y-%m-%d"] == pytest.approx(0.55, abs=0.01)
    assert any(m.code == "date_mixed_formats" for m in det.messages)


def test_mixed_format_needs_one_percent_and_at_most_three_formats() -> None:
    # 4 layouts of 25% each: no cover of <= 3 formats reaches 99%
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d%b%Y", "%b %d, %Y"]
    vals = [d.strftime(fmts[i % 4]) for i, d in enumerate(DAYS * 2)]
    assert detect(vals).type not in D.DATE_TYPES
    # a 0.5% minority layout is not a format: it stays a token (<= 1% of distinct values)
    base = [(BASE + dt.timedelta(days=i)).isoformat() for i in range(400)]
    det = detect(base + ["15MAR2024", "16MAR2024"])
    assert det.type is T.DATE and fmt_of(det) == ["%Y-%m-%d"] and det.params["other_tokens"] == 2


def test_month_names_any_case() -> None:
    sas = detect([d.strftime("%d%b%Y").upper() for d in DAYS])
    assert fmt_of(sas) == ["%d%b%Y"] and render(sas, "case") == {"b": "upper"}
    oracle = detect([f"{d.day:02d}-{d.strftime('%b').upper()}-{d.year % 100:02d}" for d in DAYS])
    assert fmt_of(oracle) == ["%d-%b-%y"] and oracle.params["two_digit_year"]["pivot"].startswith("69")
    text = detect([f"{d.strftime('%b')} {d.day}, {d.year}" for d in DAYS])
    assert fmt_of(text) == ["%b %d, %Y"] and render(text, "case") == {"b": "title"} and render(text) == {"d": False}
    low = detect([f"{d.day} {d.strftime('%B').lower()} {d.year}" for d in DAYS])
    assert fmt_of(low) == ["%d %B %Y"] and render(low, "case") == {"B": "lower"}


def test_two_digit_years_pivot_and_shift() -> None:
    rng = np.random.default_rng(1)
    years = rng.integers(1950, 2012, 300)
    vals = [f"{int(m):02d}/{int(d):02d}/{int(y) % 100:02d}" for m, d, y in
            zip(rng.integers(1, 13, 300), rng.integers(1, 29, 300), years)]
    dob = detect(vals, name="birth_date")
    tdy = dob.params["two_digit_year"]
    assert tdy["shift"] == "future" and "DOB" in tdy["reason"]
    assert dob.params["person_level"] and not dob.params["calendar"]
    v = D.view(dob.params, vals)
    y, _, _ = F.civil_from_days(v.day)
    assert y.min() >= 1950 and y.max() <= 2011  # 50-68 read as 1950-1968, not 2050-2068
    # a near-future schedule column keeps its future dates (the shift would widen the range)
    today = dt.date.today()
    sched = [(today + dt.timedelta(days=int(k))).strftime("%m/%d/%y") for k in range(-200, 200, 2)]
    det = detect(sched, name="next_review")
    assert det.params["two_digit_year"]["shift"] == "none"
    # pivot itself: 69 -> 1969, 68 -> 2068
    p = F.get_format("%m/%d/%y").parse(["01/01/69", "01/01/68"])
    assert p.year.tolist() == [1969, 2068]


# --------------------------------------------------------------------------- DATETIME / DATETIME_TZ
def test_ampm_datetimes_and_tokens() -> None:
    vals = [f"{d.month:02d}/{d.day:02d}/{d.year} {int(h):02d}:{int(m):02d} {'pm' if i % 2 else 'am'}"
            for i, (d, h, m) in enumerate(zip(DAYS, RNG.integers(1, 13, 400), RNG.integers(0, 60, 400)))]
    det = detect(vals)
    assert det.type is T.DATETIME and fmt_of(det) == ["%m/%d/%Y %I:%M %p"]
    assert det.params["formats"][0]["render"]["ampm"] == {"am": "am", "pm": "pm"}
    assert det.params["resolution"] == "minute"


def test_mixed_offsets_parsed_in_utc() -> None:
    a = detect(["2024-03-15T13:45:00-05:00", "2024-03-15T12:45:00-06:00", "2024-01-15T08:00:00-06:00",
                "2024-07-04T23:30:00-05:00"] * 5, name="ts")
    assert a.type is T.DATETIME_TZ and a.params["tz_tokens"] == {"-05:00": 10, "-06:00": 10}
    v = D.view(a.params, ["2024-03-15T13:45:00-05:00", "2024-03-15T12:45:00-06:00"])
    nums = v.numbers()
    assert nums[0] == nums[1]  # the same instant
    assert F.iso_instant(int(nums[0] * F.US), True) == "2024-03-15T18:45:00Z"
    z = detect([f"2024-03-{d:02d}T10:15:30.{d * 7:03d}Z" for d in range(1, 29)])
    assert z.type is T.DATETIME_TZ and fmt_of(z) == ["%Y-%m-%dT%H:%M:%S.%f%z"]
    assert z.params["formats"][0]["render"]["frac_digits"] == 3


def test_zone_names_and_ampm_are_not_zones() -> None:
    det = detect([f"2024-03-{d:02d} 13:45:00 {'EST' if d % 2 else 'CST'}" for d in range(1, 29)])
    assert det.type is T.DATETIME_TZ and fmt_of(det) == ["%Y-%m-%d %H:%M:%S %Z"]
    v = D.view(det.params, ["2024-03-01 13:45:00 EST", "2024-03-01 12:45:00 CST"])
    assert v.numbers()[0] == v.numbers()[1]
    ampm = detect([f"03/{d:02d}/2024 01:{d:02d} PM" for d in range(1, 29)])
    assert fmt_of(ampm) == ["%m/%d/%Y %I:%M %p"]
    assert F.is_zone_token("America/Chicago") and not F.is_zone_token("PM")


def test_sas_oracle_and_ctime_datetimes() -> None:
    assert fmt_of(detect([f"{d:02d}MAR2024:13:45:{d:02d}" for d in range(1, 29)])) == ["%d%b%Y:%H:%M:%S"]
    assert fmt_of(detect([f"{d:02d}-AUG-24 01.45.00.000000 PM" for d in range(1, 29)])) == [
        "%d-%b-%y %I.%M.%S.%f %p"]


# --------------------------------------------------------------------------- TIME / DURATION
def test_time_versus_duration() -> None:
    hhmm = [f"{int(h):02d}:{int(m) * 5:02d}" for h, m in zip(RNG.integers(7, 18, 200), RNG.integers(0, 12, 200))]
    t = detect(hhmm, name="start")
    assert t.type is T.TIME and t.params["granularity_seconds"] == 300.0
    dur = detect(hhmm, name="duration_hrs")  # a duration header hint
    assert dur.type is T.DURATION and fmt_of(dur) == ["@dur_hm"]
    over = detect([f"{int(h)}:{int(m):02d}:00" for h, m in zip(RNG.integers(0, 40, 200), RNG.integers(0, 60, 200))])
    assert over.type is T.DURATION
    # one value >= 24 h among hundreds still makes the column a duration (section 3 #24)
    almost = [f"{h % 24}:{m:02d}" for h in range(24) for m in range(0, 60, 5)] + ["25:00"]
    assert detect(almost, name="x").type is T.DURATION
    ampm = detect([f"{int(h)}:{int(m):02d} AM" for h, m in zip(RNG.integers(1, 12, 200), RNG.integers(0, 60, 200))])
    assert ampm.type is T.TIME and render(ampm) == {"I": False, "M": True}


def test_iso_and_pandas_durations() -> None:
    iso = detect([f"PT{h}H{m}M" if h else f"PT{m}M" for h, m in zip(range(30), range(1, 31))])
    assert iso.type is T.DURATION and fmt_of(iso) == ["@dur_iso"]
    assert F.get_format("@dur_iso").parse(["P", "PT", "P1D", "PT1.5S", "P1Y"]).ok.tolist() == [
        False, False, True, True, False]
    pd_style = detect([f"{d} days {h:02d}:30:00" for d, h in zip(range(20), range(20))])
    assert pd_style.type is T.DURATION and fmt_of(pd_style) == ["@dur_days"]
    assert pd_style.params["formats"][0]["render"]["style"] == "pandas"


# --------------------------------------------------------------------------- YEAR_MONTH
def test_year_month() -> None:
    det = detect([f"{d.year}-{d.month:02d}" for d in DAYS])
    assert det.type is T.YEAR_MONTH and fmt_of(det) == ["%Y-%m"]
    assert detect([d.strftime("%b %Y") for d in DAYS]).type is T.YEAR_MONTH
    assert fmt_of(detect([f"{d.strftime('%b')}-{d.year % 100:02d}" for d in DAYS])) == ["%b-%y"]


# --------------------------------------------------------------------------- DATE_AS_INT
def test_yyyymmdd_integers() -> None:
    vals = [d.strftime("%Y%m%d") for d in DAYS]
    det = detect(vals, name="enroll")
    assert det.type is T.DATE_AS_INT and det.subtype == "yyyymmdd"
    assert detect(vals + ["2024315"], name="x").type is not T.DATE_AS_INT  # \d{8} guard
    assert detect(vals + ["20241332"], name="x").type is not T.DATE_AS_INT  # 100% valid required
    with_zero = detect(vals + ["0"] * 5, name="x")
    assert with_zero.type is T.DATE_AS_INT and with_zero.params["sentinels"] == {"0": 5}
    assert detect([str(1990 + i % 30) for i in range(100)], name="yr").type is T.INTEGER_YEAR


def test_excel_serials_need_a_hint_or_date_cols() -> None:
    serials = [str((d - dt.date(1899, 12, 30)).days) for d in DAYS]
    assert detect(serials, name="amount").type is T.INTEGER
    hinted = detect(serials, name="entry_date")
    assert hinted.type is T.DATE_AS_INT and hinted.subtype == "excel_serial"
    flagged = detect(serials, name="amount", date_cols=["amount"])
    assert flagged.subtype == "excel_serial"
    assert D.view(hinted.params, ["45366"]).numbers()[0] == F.ordinal(2024, 3, 15)


def test_epoch_needs_a_hint() -> None:
    secs = [str(1_700_000_000 + 3600 * i) for i in range(100)]
    assert detect(secs, name="amount").type is T.INTEGER
    assert detect(secs, name="updated").subtype == "epoch_s"
    ms = [str(1_700_000_000_000 + 3_600_123 * i) for i in range(100)]
    det = detect(ms, name="created")
    assert det.subtype == "epoch_ms" and det.params["resolution"] == "millisecond"


# --------------------------------------------------------------------------- what dates never claim
@pytest.mark.parametrize("values,expected", [
    (["2019-2020", "2020-2021", "2021-2022"], T.SCHOOL_YEAR),
    (["2324", "2425", "2223"], T.SCHOOL_YEAR),
    (["SY2324", "SY2425", "SY2223"], T.SCHOOL_YEAR),
    (["0101", "0215", "0331"], T.DIGIT_CODE),
    (["1.5", "2.25", "3.75", "4.0"], T.FLOAT),
])
def test_dates_never_claim_other_types(values: list[str], expected: T) -> None:
    assert detect(values * 10, name="v").type is expected


def test_non_dates_fall_through_quickly() -> None:
    text = [f"Student needs help with unit {i}" for i in range(500)]
    assert D.detect_dates(ctx_for(text, "notes")) is None
    codes = [f"AB-{i:04d}" for i in range(500)]
    assert D.detect_dates(ctx_for(codes, "code")) is None


# --------------------------------------------------------------------------- sentinels and tokens
def test_sentinels() -> None:
    vals = [d.strftime("%m/%d/%Y") for d in DAYS[:300]] + ["01/01/1900"] * 3 + ["12/31/9999"] * 2 + ["0/0/0"] * 4
    det = detect(vals, name="exit_date")
    assert det.params["sentinels"] == {"0/0/0": 4, "01/01/1900": 3, "12/31/9999": 2}
    v = D.view(det.params, ["01/01/1900", "0/0/0", DAYS[0].strftime("%m/%d/%Y")])
    assert v.cls.tolist() == [D.SENTINEL, D.SENTINEL, 0]
    assert np.isnan(v.numbers()[:2]).all()
    # >= 1% of rows, >= --min-cell (10) rows and >= 50 years from the median (K-07: 4 rows are real dates)
    far = [d.isoformat() for d in DAYS[:300]] + ["1850-06-01"] * 10
    assert detect(far).params["sentinels"] == {"1850-06-01": 10}
    assert "sentinels" not in detect([d.isoformat() for d in DAYS[:300]] + ["1850-06-01"] * 4).params
    # a far value under 1% of rows is a real (extreme) value, not a token
    rare = [d.isoformat() for d in DAYS[:300]] + ["1850-06-01"]
    assert "sentinels" not in detect(rare).params


def test_unparsed_leftovers() -> None:
    base = [(BASE + dt.timedelta(days=i)).isoformat() for i in range(300)]
    det = detect(base + ["TBD", "a note that is too long to be a token"], name="d")
    assert det.type is T.DATE
    assert det.params["other_tokens"] == 1 and det.params["_other_tokens"] == ["TBD"]
    assert det.params["unparsed_replaced"] == 1
    assert "TBD" not in str(det.params.get("sentinels")) and "_other_tokens" not in str(
        {k: v for k, v in det.params.items() if not k.startswith("_")})


# --------------------------------------------------------------------------- flags and overrides
def test_date_format_flag() -> None:
    vals = [d.strftime("%d/%m/%Y") for d in DAYS]
    det = detect(vals, name="d", date_format=["d=%-d/%-m/%Y"])
    assert fmt_of(det) == ["%d/%m/%Y"] and "--date-format" in det.reason
    with pytest.raises(UnsupportedColumnError):
        detect(vals, name="d", date_format=["d=%Y-%m-%d"])
    with pytest.raises(UsageError):
        detect(vals, name="d", date_format=["d=%Q"])
    assert detect([d.strftime("%Y%m%d") for d in DAYS], name="d", date_format=["d=%Y%m%d"]).type is T.DATE_AS_INT


def test_force_type() -> None:
    hhmm = [f"{h:02d}:{m:02d}" for h in range(8, 12) for m in range(0, 60, 10)]
    assert detect(hhmm, name="x", force_type=["x=DURATION"]).type is T.DURATION
    assert detect([d.isoformat() for d in DAYS], name="x", force_type=["x=DATE_MIXED_FORMATS"]).type is \
        T.DATE_MIXED_FORMATS
    mixed = [d.isoformat() if i % 2 else d.strftime("%m/%d/%Y") for i, d in enumerate(DAYS)]
    forced_date = detect(mixed, name="x", force_type=["x=DATE"])
    assert forced_date.type is T.DATE and len(forced_date.params["formats"]) == 2
    with pytest.raises(UnsupportedColumnError):
        detect(["25:00", "26:30"] * 5, name="x", force_type=["x=TIME"])
    with pytest.raises(UnsupportedColumnError):
        detect([d.isoformat() for d in DAYS], name="x", force_type=["x=DATETIME"])
    serials = [str((d - dt.date(1899, 12, 30)).days) for d in DAYS]
    assert detect(serials, name="amount", force_type=["amount=DATE_AS_INT"]).subtype == "excel_serial"


def test_calendar_detection() -> None:
    rng = np.random.default_rng(3)
    school = [BASE + dt.timedelta(days=int(i)) for i in range(0, 120) if (BASE + dt.timedelta(days=int(i))).weekday() < 5]
    att = [school[int(i)].isoformat() for i in rng.integers(0, 30, 600)]
    det = detect(att, name="attendance_date")
    assert det.params["calendar"] and det.params["rows_per_distinct_date"] >= 10
    assert det.params["weekdays_observed"] == [0, 1, 2, 3, 4]
    sparse = detect([d.isoformat() for d in DAYS], name="withdraw_date")
    assert not sparse.params["calendar"]
    assert detect([d.isoformat() for d in DAYS], name="w", calendar=["w"]).params["calendar_reason"] == "--calendar"
    dob = detect(att, name="dob", calendar=["dob"])
    assert not dob.params["calendar"] and any(m.code == "calendar_ignored" for m in dob.messages)


def test_whitespace_variants_recorded() -> None:
    vals = [d.isoformat() for d in DAYS[:100]] + [" " + d.isoformat() for d in DAYS[100:120]]
    det = detect(vals)
    assert det.type is T.DATE
    ws = {(a, b): c for a, b, c in det.params["whitespace"]}
    assert ws[(" ", "")] == 20 and ws[("", "")] == 100


# --------------------------------------------------------------------------- native typed storage
def test_native_contract() -> None:
    native = {"temporal": {"kind": "date", "tz": None, "unit": "D"}}
    det = detect([d.isoformat() for d in DAYS], name="x", native=native)
    assert det.type is T.DATE and det.source == "native" and det.params["native"]["kind"] == "date"
    midnight = detect([d.isoformat() + " 00:00:00" for d in DAYS], name="x", native=native)
    assert midnight.type is T.DATE and fmt_of(midnight) == ["%Y-%m-%d 00:00:00"]
    ts = {"arrow_type": "timestamp[us, tz=America/Chicago]"}
    assert D.native_temporal(ts) == {"kind": "datetime", "tz": "America/Chicago", "unit": None}
    aware = detect([f"2024-03-{d:02d} 08:30:00-05:00" for d in range(1, 29)], name="x", native=ts)
    assert aware.type is T.DATETIME_TZ
    naive = detect([f"2024-03-{d:02d} 08:30:{d:02d}" for d in range(1, 29)], name="x",
                   native={"arrow_type": "timestamp[us]"})
    assert naive.type is T.DATETIME
    tm = detect([f"{h:02d}:30:00" for h in range(24)], name="duration", native={"arrow_type": "time64[us]"})
    assert tm.type is T.TIME  # native time wins over a duration header hint
    du = detect(["1:30:00", "0:45:00", "2:00:00"] * 3, name="x", native={"arrow_type": "duration[s]"})
    assert du.type is T.DURATION
    assert D.native_temporal({"stata_format": "%td"})["kind"] == "date"
    assert D.native_temporal({"stata_format": "%tc"})["kind"] == "datetime"
    assert D.native_temporal({"cell_type": "d", "number_format": "mm/dd/yyyy h:mm"})["kind"] == "datetime"
    assert D.native_temporal({"cell_type": "d", "number_format": "mm/dd/yyyy"})["kind"] == "date"
    for other in (None, {}, {"arrow_type": "string"}, {"stata_format": "%9.0g"}, {"cell_types": {"n": 5}}):
        assert D.native_temporal(other) is None
    assert D.native_detection(ctx_for(["2024-01-01", "2024-01-02"], "x")) is None


def test_large_columns_are_sampled_then_validated() -> None:
    n = 6000
    iso = [(dt.date(2000, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(n)]
    det = detect(iso)
    assert det.type is T.DATE and det.params["formats"][0]["share"] == 1.0
    # 2% of the distinct values in another layout: the sample decides, every distinct value is checked
    us = [(dt.date(2020, 1, 1) + dt.timedelta(days=i)).strftime("%m/%d/%Y") for i in range(120)]
    det2 = detect(iso + us)
    assert det2.type is T.DATE_MIXED_FORMATS
