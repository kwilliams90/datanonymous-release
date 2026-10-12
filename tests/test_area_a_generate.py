"""AREA A generation, rendering, statistics, numeric view, dictionary fields and the native contract."""

from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from datanonymous.detect import dates as D
from datanonymous.detect import dates_formats as F
from datanonymous.generate import dates as G
from datanonymous.model import ColumnModel, ColumnProfile
from datanonymous.model import ColumnType as T
from datanonymous.profile.stats import column_stats, numeric_view
from datanonymous.report import docs

from test_area_a_support import days, parse_all, profile_of, run_table

RNG = np.random.default_rng(7)
BASE = dt.date(2023, 8, 14)
DAYS = days(BASE, 400, RNG)


def synth(values: list[str], name: str = "col", seed: int = 1, **cli):
    tp, cols = run_table({name: values}, seed=seed, **cli)
    return profile_of(tp, name), list(cols[name])


def conforms(p, out: list[str]) -> bool:
    """Every synthetic non-missing value is a token or re-parses with one of the detected formats."""
    v, vals = parse_all(p, out)
    return bool(np.all(v.cls != D.UNPARSED)) and len(vals) > 0


# --------------------------------------------------------------------------- rendering fidelity
@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d%b%Y", "%b %d, %Y", "%d-%b-%y",
                                 "%A, %B %d, %Y", "%Y%m%d", "%m-%d-%Y"])
def test_round_trip_every_value_reparses(fmt: str) -> None:
    vals = [d.strftime(fmt) for d in DAYS]
    if "%b" in fmt and "%y" in fmt:
        vals = [v.upper() for v in vals]
    p, out = synth(vals)
    assert p.type in D.DATE_TYPES
    assert conforms(p, out)
    assert sorted({len(v) for v in out}) == sorted({len(v) for v in vals}) or "%B" in fmt or "%b %d" in fmt


def test_unpadded_and_case_preserved() -> None:
    vals = [f"{d.month}/{d.day}/{d.year}" for d in DAYS]
    p, out = synth(vals)
    assert any(len(v.split("/")[0]) == 1 for v in out) and not any(v.startswith("0") for v in out)
    assert not any("%" in v or "#" in v for v in out)  # never strftime('%-m') leftovers
    sas = [d.strftime("%d%b%Y").upper() for d in DAYS]
    _, out2 = synth(sas)
    assert all(v[2:5].isupper() for v in out2)
    low = [d.strftime("%d-%b-%Y").lower() for d in DAYS]
    _, out3 = synth(low)
    assert all(v[3:6].islower() for v in out3)


def test_ampm_and_fraction_digits_rendered() -> None:
    vals = [f"{d.month:02d}/{d.day:02d}/{d.year} {h:02d}:{m:02d} {'PM' if i % 2 else 'AM'}"
            for i, (d, h, m) in enumerate(zip(DAYS, RNG.integers(1, 13, 400), RNG.integers(0, 60, 400)))]
    p, out = synth(vals)
    assert conforms(p, out) and {v[-2:] for v in out} == {"AM", "PM"}
    ms = [f"2024-03-{d:02d}T10:15:{s:02d}.{s * 13 % 1000:03d}Z" for d in range(1, 29) for s in range(0, 60, 7)]
    p2, out2 = synth(ms)
    assert conforms(p2, out2) and all(len(v.split(".")[1]) == 4 for v in out2)  # 3 digits + 'Z'


def test_whitespace_variants_rendered_at_their_rates() -> None:
    vals = [d.isoformat() for d in DAYS[:300]] + [d.isoformat() + "  " for d in DAYS[300:400]]
    p, out = synth(vals)
    assert sum(v.endswith("  ") for v in out) == 100 and conforms(p, out)


# --------------------------------------------------------------------------- distributions
def test_person_level_dates_interpolate_and_snap_weekdays() -> None:
    sch = [BASE + dt.timedelta(days=i) for i in range(280) if (BASE + dt.timedelta(days=i)).weekday() < 5]
    rng = np.random.default_rng(5)
    vals = [sch[int(i)].strftime("%m/%d/%Y") for i in rng.integers(0, len(sch), 300)]
    p, out = synth(vals, name="withdraw_date")
    assert not p.params["calendar"] and p.params["weekdays_observed"] == [0, 1, 2, 3, 4]
    parsed = [dt.datetime.strptime(v, "%m/%d/%Y").date() for v in out]
    assert all(d.weekday() < 5 for d in parsed)  # weekends never observed -> never generated
    real = {dt.datetime.strptime(v, "%m/%d/%Y").date() for v in vals}
    assert min(parsed) == min(real) and max(parsed) == max(real)  # D14 exact extremes
    assert any(d not in real for d in parsed)  # interpolated, not resampled


def test_calendar_columns_use_observed_dates_only() -> None:
    rng = np.random.default_rng(9)
    obs = [BASE + dt.timedelta(days=int(i)) for i in rng.choice(200, 25, replace=False)]
    vals = [obs[int(i)].isoformat() for i in rng.integers(0, 25, 500)]
    p, out = synth(vals, name="attendance_date")
    assert p.params["calendar"]
    assert set(out) <= set(vals)
    assert Counter(out) == Counter(vals)  # exact quota at the same N
    # the same values in a DOB-hinted column are interpolated (never restricted to observed dates)
    p2, out2 = synth(vals, name="birth_date")
    assert not p2.params["calendar"] and not set(out2) <= set(vals)


def test_no_exact_extremes_flag() -> None:
    vals = [d.isoformat() for d in DAYS]
    p, out = synth(vals, name="d", seed=3)
    assert min(out) == min(vals) and max(out) == max(vals)
    p2, _ = synth(vals, name="d", seed=3, no_exact_extremes=["d"])
    assert p2.params.get("_no_exact_extremes") and not p2.flags.get("extreme_value_adjusted")


def test_mixed_format_shares_are_exact() -> None:
    vals = [d.isoformat() if i % 20 < 11 else d.strftime("%m/%d/%Y") for i, d in enumerate(DAYS)]
    p, out = synth(vals)
    assert sum("-" in v for v in out) == sum("-" in v for v in vals)
    assert conforms(p, out)


def test_tz_offsets_and_local_times() -> None:
    import zoneinfo

    chi = zoneinfo.ZoneInfo("America/Chicago")
    vals = []
    for d in DAYS:
        loc = dt.datetime.combine(d, dt.time(8)).replace(tzinfo=chi) + dt.timedelta(minutes=int(RNG.integers(0, 600)))
        vals.append(loc.isoformat())
    p, out = synth(vals)
    assert p.type is T.DATETIME_TZ
    assert Counter(v[-6:] for v in out) == Counter(v[-6:] for v in vals)  # offset pmf reapplied exactly
    assert conforms(p, out)
    for v in out:  # every rendered value is a valid instant
        dt.datetime.fromisoformat(v)
    real = np.sort(numeric_view(p, vals))
    syn = np.sort(numeric_view(p, out))
    assert real[0] == syn[0] and real[-1] == syn[-1]  # exact UTC extremes


def test_iana_zone_names_render_with_the_right_offset() -> None:
    vals = [f"2024-{m:02d}-{d:02d} 10:00:00 America/Chicago" for m in (1, 7) for d in range(1, 29)]
    p, out = synth(vals)
    assert p.type is T.DATETIME_TZ
    v = D.view(p.params, out)
    assert np.all(v.cls >= 0)
    for text, inst in zip(out, v.numbers()):
        local = dt.datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=__import__("zoneinfo").ZoneInfo(
            "America/Chicago"))
        assert local.timestamp() == pytest.approx(inst)


def test_time_granularity_and_range() -> None:
    vals = [f"{h:02d}:{m:02d}" for h, m in zip(RNG.integers(7, 10, 300), RNG.integers(0, 4, 300) * 15)]
    p, out = synth(vals, name="start_time")
    assert p.type is T.TIME and p.params["granularity_seconds"] == 900.0
    assert all(int(v[3:]) % 15 == 0 for v in out)
    assert min(out) == min(vals) and max(out) == max(vals)


@pytest.mark.parametrize("vals", [
    [f"{h}:{m:02d}:{s:02d}" for h, m, s in zip(range(0, 50), range(10, 60), range(5, 55))],
    [f"PT{h}H{m}M" if h else f"PT{m}M" for h, m in zip(range(25), range(5, 30))],
    [f"{d} days {h % 24:02d}:{m:02d}:00" for d, h, m in zip(range(25), range(25), range(30, 55))],
])
def test_durations_round_trip(vals: list[str]) -> None:
    p, out = synth(vals * 4, name="elapsed")
    assert p.type is T.DURATION and conforms(p, out)
    real = numeric_view(p, vals)
    syn = numeric_view(p, out)
    assert syn.min() == real.min() and syn.max() == real.max()


def test_sentinels_written_verbatim_at_exact_rates() -> None:
    vals = ([d.strftime("%m/%d/%Y") for d in DAYS[:300]] + ["1/1/1900"] * 7 + ["12/31/9999"] * 3
            + ["0/0/0"] * 5 + [""] * 20 + ["NA"] * 4)
    p, out = synth(vals, name="exit_date")
    c, r = Counter(out), Counter(vals)
    for tok in ("1/1/1900", "12/31/9999", "0/0/0", "", "NA"):
        assert c[tok] == r[tok], tok
    assert conforms(p, out)


def test_short_tokens_kept_long_text_never_copied() -> None:
    base = [(BASE + dt.timedelta(days=i)).isoformat() for i in range(300)]
    secret = "call Jane Q. Public before releasing"
    p, out = synth(base + ["TBD"] * 2 + [secret])
    assert out.count("TBD") == 2 and secret not in out and conforms(p, out)


def test_two_digit_years_stay_two_digits() -> None:
    rng = np.random.default_rng(2)
    vals = [f"{int(m):02d}/{int(d):02d}/{int(y) % 100:02d}" for m, d, y in
            zip(rng.integers(1, 13, 300), rng.integers(1, 29, 300), rng.integers(1950, 2012, 300))]
    p, out = synth(vals, name="dob")
    assert all(len(v) == 8 for v in out) and conforms(p, out)
    st_syn = column_stats(np.array(out, dtype=object), p, 10)
    assert st_syn["min"] >= "1950-01-01" and st_syn["max"] <= "2011-12-31"
    years = {int(v[-2:]) for v in out}
    assert not any(12 <= y <= 49 for y in years)  # no wrap-around through the pivot


def test_date_as_int_forms() -> None:
    ymd = [d.strftime("%Y%m%d") for d in DAYS]
    p, out = synth(ymd, name="enroll")
    assert p.type is T.DATE_AS_INT and all(len(v) == 8 and v.isdigit() for v in out)
    ser = [str((d - dt.date(1899, 12, 30)).days) for d in DAYS]
    p2, out2 = synth(ser, name="entry_date")
    assert p2.subtype == "excel_serial" and all(len(v) == 5 for v in out2) and min(out2) == min(ser)
    ep = [str(1_700_000_000 + int(x)) for x in RNG.integers(0, 10_000_000, 400)]
    p3, out3 = synth(ep, name="updated")
    assert p3.subtype == "epoch_s" and all(len(v) == 10 for v in out3)
    assert min(map(int, out3)) == min(map(int, ep)) and max(map(int, out3)) == max(map(int, ep))


def test_year_month_generation() -> None:
    vals = [f"{d.year}-{d.month:02d}" for d in DAYS]
    p, out = synth(vals)
    assert p.type is T.YEAR_MONTH and set(out) <= set(vals) and conforms(p, out)


def test_within_id_dob_is_constant_and_person_level() -> None:
    rng = np.random.default_rng(4)
    ids, dobs = [], []
    for k in range(120):
        dob = (dt.date(2008, 1, 1) + dt.timedelta(days=int(rng.integers(0, 3000)))).strftime("%m/%d/%Y")
        for _ in range(int(rng.integers(1, 5))):
            ids.append(f"{100000 + k}")
            dobs.append(dob)
    tp, cols = run_table({"student_id": ids, "dob": dobs, "visit": [d.isoformat() for d in DAYS[:len(ids)]]})
    p = profile_of(tp, "dob")
    assert p.role == "entity" and p.type is T.DATE
    per: dict[str, set[str]] = {}
    for sid, d in zip(cols["student_id"], cols["dob"]):
        per.setdefault(sid, set()).add(d)
    assert all(len(v) == 1 for v in per.values())
    assert not p.params["calendar"]


def test_deterministic_for_a_seed() -> None:
    vals = [d.isoformat() for d in DAYS]
    assert synth(vals, seed=11)[1] == synth(vals, seed=11)[1]
    assert synth(vals, seed=11)[1] != synth(vals, seed=12)[1]


# --------------------------------------------------------------------------- statistics, numbers, docs
def test_date_stats_are_aggregates() -> None:
    vals = [d.isoformat() for d in DAYS] + ["1900-01-01"] * 3
    p, _ = synth(vals, name="d")
    st = p.stats
    for key in ("min", "max", "mean", "median", "quantiles", "year_pmf", "month_pmf", "weekday_pmf",
                "weekend_share", "rows_per_distinct_date", "sentinels"):
        assert key in st, key
    assert "level_shares" not in st and "pmf" not in st  # never mistaken for a level column
    assert st["min"] == min(d.isoformat() for d in DAYS) and st["sentinels"] == {"1900-01-01": 3}
    dtp, _ = synth([f"2024-03-{d:02d} {h:02d}:00:00" for d in range(1, 29) for h in (0, 9, 13)])
    for key in ("time_of_day_quantiles", "midnight_share", "resolution"):
        assert key in dtp.stats
    assert dtp.stats["midnight_share"] == pytest.approx(1 / 3, abs=1e-6)


def test_date_numbers_units() -> None:
    p = ColumnProfile(index=0, name="d", type=T.DATE, params=D.detect_from_inputs(D.DateInputs(
        name="d", typed={"1970-03-01": 1, "1970-01-02": 1}, raw_counts={}, hints=__import__(
            "datanonymous.hints", fromlist=["header_hints"]).header_hints("d"))).params)
    assert D.date_numbers(["1970-01-02", "1970-03-01", "1970-01-02"], p).tolist() == [1.0, 59.0, 1.0]
    assert D.date_numbers([], p).size == 0
    assert D.date_numbers(["x"], ColumnProfile(index=0, name="d", type=T.DATE)) is None


@pytest.mark.parametrize("vals,readr", [
    ([d.isoformat() for d in DAYS], "D"),
    ([d.strftime("%m/%d/%Y") for d in DAYS], "c"),
    ([f"2024-03-{d:02d}T10:00:{d:02d}Z" for d in range(1, 29)], "T"),
    ([f"{h:02d}:{m:02d}" for h in range(8, 12) for m in range(0, 60, 10)], "t"),
    ([d.strftime("%Y%m%d") for d in DAYS], "i"),
])
def test_doc_fields(vals: list[str], readr: str) -> None:
    p, _ = synth(vals, name="d")
    fields = D.doc_fields(p, p.stats)
    assert fields["readr"] == readr
    assert fields.get("format")
    if readr == "c":
        assert "parse_date_time" in fields["note"] and "%m/%d/%Y" in fields["note"]
    entry = docs.column_entry(p, p.stats, None, 10)
    assert entry["readr"] == readr and entry["r_class"] in ("Date", "POSIXct", "hms")


# --------------------------------------------------------------------------- native contract helpers
def test_to_native_conversions() -> None:
    p, out = synth([d.isoformat() for d in DAYS], name="d")
    nat = G.to_native(p, out + [None, "", "NA"])
    assert all(isinstance(x, dt.date) for x in nat[:-3]) and nat[-3:] == [None, None, None]
    assert nat[0].isoformat() == out[0]
    p2, out2 = synth([f"2024-03-{d:02d}T10:15:00-05:00" for d in range(1, 29)], name="ts")
    aware = G.to_native(p2, out2)
    assert all(x.tzinfo is not None and x.utcoffset() == dt.timedelta(hours=-5) for x in aware)
    p3, out3 = synth([f"{h}:{m:02d}:00" for h, m in zip(range(30), range(30))], name="elapsed")
    assert all(isinstance(x, dt.timedelta) for x in G.to_native(p3, out3))
    p4, out4 = synth([f"{h:02d}:{m:02d}" for h in range(7, 12) for m in range(0, 60, 15)], name="start")
    assert all(isinstance(x, dt.time) for x in G.to_native(p4, out4))


def test_native_column_round_trip_through_generation() -> None:
    """A reader's canonical strings come back in exactly the same canonical form."""
    native = {"temporal": {"kind": "datetime", "tz": "America/Chicago", "unit": "us"}}
    from test_area_a_support import ctx_for
    from datanonymous.detect.types import detect_column
    from datanonymous.profile.table import build_model
    from datanonymous.generate.table import generate_values

    vals = [f"2024-0{m}-{d:02d} 08:30:00{'-05:00' if m > 3 else '-06:00'}" for m in (1, 7) for d in range(1, 29)]
    ctx = ctx_for(vals, "ts", native=native)
    det = detect_column(ctx)
    assert det.type is T.DATETIME_TZ and det.source == "native"
    p = ColumnProfile(index=0, name="ts", type=det.type, params=det.params, na_tokens=frozenset(), native=native)
    p.model = build_model(ctx.values, p, ctx.raw_counts)
    out, _ = generate_values(p, p.model, len(vals), np.random.default_rng(0))
    assert all(len(v) == 25 and v[10] == " " for v in out)
    zone = G.to_native(p, list(out))
    assert all(str(x.tzinfo) == "America/Chicago" for x in zone)


def test_foreign_detection_without_params_is_regenerated() -> None:
    """A date type from another hook (no AREA A params) is re-read from the model."""
    vals = [d.strftime("%m/%d/%Y") for d in DAYS]
    model = ColumnModel(nonmissing_counts=dict(Counter(vals)))
    p = ColumnProfile(index=0, name="d", type=T.DATE)
    out, flags = G.generate_dates(p, model, 50, np.random.default_rng(1))
    text = G.render_dates(p, out)
    assert p.params["formats"][0]["format"] == "%m/%d/%Y"
    assert all(dt.datetime.strptime(t, "%m/%d/%Y") for t in text)


def test_snap_and_extremes_helpers() -> None:
    sat = F.ordinal(2024, 3, 16)
    y = np.array([sat, sat + 1, sat + 2])
    assert G.snap_weekdays(y.copy(), [0, 1, 2, 3, 4], sat - 10, sat + 10).tolist() == [sat - 1, sat + 2, sat + 2]
    z = np.array([5, 5, 6])
    assert G.ensure_extremes(z, 1, 9) and sorted(z.tolist()) == [1, 5, 9]
    assert not G.ensure_extremes(np.array([3]), 3, 3)


# --------------------------------------------------------------------------- property test
_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y", "%b %d, %Y", "%m/%d/%Y %I:%M %p", "%Y-%m-%dT%H:%M:%S",
            "%d%b%Y:%H:%M:%S", "%Y-%m"]


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(fmt=st.sampled_from(_FORMATS), start=st.integers(0, 40000), span=st.integers(30, 4000),
       n=st.integers(40, 160), seed=st.integers(0, 10_000))
def test_property_generated_values_conform_and_stay_in_range(fmt: str, start: int, span: int, n: int,
                                                             seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = dt.datetime(1950, 1, 1) + dt.timedelta(days=start)
    vals = [(base + dt.timedelta(seconds=int(s) * 60)).strftime(fmt) for s in rng.integers(0, span * 1440, n)]
    p, out = synth(vals, name="v", seed=seed)
    if p.type not in D.DATE_TYPES:  # e.g. a 2-value column became a FACTOR
        return
    assert conforms(p, out)
    real, syn = numeric_view(p, vals), numeric_view(p, out)
    assert syn.min() >= real.min() and syn.max() <= real.max()


@pytest.mark.slow
def test_performance_one_million_rows() -> None:
    import time

    n = 1_000_000
    secs = np.random.default_rng(0).integers(0, 9 * 365 * 86400, n)
    vals = (np.datetime64("2015-01-01T00:00:00") + secs.astype("timedelta64[s]")).astype(str).tolist()
    t = time.perf_counter()
    synth(vals[:n], name="ts")
    assert time.perf_counter() - t < 60
