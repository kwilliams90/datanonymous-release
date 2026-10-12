"""Round-2 adversarial review, dates/text cluster (R2-13, R2-14).

* R2-13: time values were generated on the nearest *fixed* step (30/15/5/1 min, 1 s, 1 ms ...), so an
  hourly column gained :30 values, a 10-minute column :05 values, a ':15 every hour' column :00 values,
  whole-day durations half hours. Values now stay on the observed grid ``phase + k * step`` (any step,
  with its phase; a step that does not divide a day restarts at midnight). The coarser siblings are
  covered too: dates with an observed calendar grid (month precision on the 1st, the 15th, month ends,
  biweekly, quarterly, yearly) and quarterly YEAR_MONTH values.
* R2-14: raw case forms of one value that share a broad case style (``aLiCe``, ``alICe``, ``AliCE``)
  were rendered identically, collapsing distinct values and frequencies. Every per-distinct text
  generator (filler text, names, e-mails, cities, addresses, URLs, invented labels, code masks, forced
  STATE masks, the run-wide name redraw) and MULTI_SELECT items now keep each raw form apart.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import zoneinfo
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import read_dicts, run_cli
from datanonymous.detect import dates as D
from datanonymous.detect.text import case_style
from datanonymous.generate import dates as G
from datanonymous.generate import text as textgen
from datanonymous.model import ColumnType as T
from datanonymous.profile.stats import numeric_view

from test_area_a_support import profile_of, run_table

US = 1_000_000
DAY = 86400 * US


def synth(values: list[str], name: str = "col", seed: int = 1, **cli: Any):
    tp, cols = run_table({name: values}, seed=seed, **cli)
    return profile_of(tp, name), list(cols[name])


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def wall(v: str) -> dt.datetime:
    """Local wall time of an ISO datetime text (offset dropped)."""
    return dt.datetime.fromisoformat(v).replace(tzinfo=None)


def tod_minutes(v: str) -> int:
    """Minutes since midnight of an h:mm[:ss] [AM|PM] time text."""
    parts = v.split()
    h, m = (int(x) for x in parts[0].split(":")[:2])
    if len(parts) > 1:
        h = h % 12 + (12 if parts[1] == "PM" else 0)
    return h * 60 + m


def dur_minutes(v: str) -> int:
    h, m = (int(x) for x in v.split(":")[:2])
    return h * 60 + m


# =========================================================================== R2-13 time grid
def test_reviewer_case_hourly_offsets_keep_the_hourly_local_grid(tmp_path: Path) -> None:
    """The reviewer's fixture: hourly local times written with +05:30, +05:45 and -03:30."""
    rows = []
    for i in range(120):
        off = ["+05:30", "+05:45", "-03:30"][i % 3]
        rows.append([f"2024-03-{i % 25 + 1:02d}T{8 + i % 10:02d}:00:00{off}", i])
    src = write_csv(tmp_path / "in" / "data.csv", ["event_timestamp", "score"], rows)
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(src.parent, "--out-dir", out, "--report-dir", rep, "--seed", "123")
    assert res.exit_code == 0, res.output
    syn = [r["event_timestamp"] for r in read_dicts(out / "data.csv")]
    assert len(syn) == 120
    assert all(v[14:19] == "00:00" for v in syn), [v for v in syn if v[14:19] != "00:00"]
    assert {wall(v).hour for v in syn} <= set(range(8, 18))
    assert Counter(v[-6:] for v in syn) == Counter(r[0][-6:] for r in rows)  # offsets by exact quota
    real = sorted(dt.datetime.fromisoformat(r[0]) for r in rows)
    got = sorted(dt.datetime.fromisoformat(v) for v in syn)
    assert got[0] == real[0] and got[-1] == real[-1]  # D14 exact UTC extremes
    # same seed, same bytes
    res = run_cli(src.parent, "--out-dir", tmp_path / "out2", "--report-dir", tmp_path / "rep2", "--seed", "123")
    assert res.exit_code == 0 and hashes(tmp_path / "out2") == hashes(out)


_H = [f"2024-03-{d:02d}" for d in range(1, 29)]

TIME_GRIDS: dict[str, tuple[str, list[str], float, Callable[[str], bool]]] = {
    "hourly datetime": ("event_ts", [f"{d} {h:02d}:00:00" for d in _H for h in range(8, 17)], 3600,
                        lambda v: v[14:19] == "00:00"),
    "2-hourly datetime": ("event_ts", [f"{d} {h:02d}:00" for d in _H for h in range(8, 19, 2)], 7200,
                          lambda v: int(v[11:13]) % 2 == 0 and v[14:16] == "00"),
    "hourly ISO Z": ("event_ts", [f"{d}T{h:02d}:00:00Z" for d in _H for h in range(6, 20)], 3600,
                     lambda v: v[14:19] == "00:00"),
    "10-minute time": ("start_time", [f"{h:02d}:{m:02d}" for h in range(7, 12) for m in range(0, 60, 10)] * 3, 600,
                       lambda v: int(v[3:5]) % 10 == 0),
    "20-minute time": ("start_time", [f"{h:02d}:{m:02d}" for h in range(7, 12) for m in range(0, 60, 20)] * 3,
                       1200, lambda v: int(v[3:5]) % 20 == 0),
    "every hour at :15": ("start_time", [f"{h:02d}:15" for h in range(7, 18)] * 5, 3600, lambda v: v[3:5] == "15"),
    ":15 and :45": ("start_time", [f"{h:02d}:{m}" for h in range(7, 18) for m in ("15", "45")] * 3, 1800,
                    lambda v: v[3:5] in ("15", "45")),
    "10-second time": ("start_time", [f"08:{m:02d}:{s:02d}" for m in range(30) for s in range(0, 60, 10)], 10,
                       lambda v: int(v[6:8]) % 10 == 0),
    "100 ms datetime": ("event_ts", [f"2024-03-01 08:00:{s:02d}.{f}00" for s in range(40) for f in range(10)], 0.1,
                        lambda v: v.endswith("00")),
    "hourly AM/PM": ("appt_time", [f"{h}:00 {ap}" for h in range(1, 12) for ap in ("AM", "PM")] * 3, 3600,
                     lambda v: tod_minutes(v) % 60 == 0),
    "hourly durations": ("duration_hrs", [f"{h}:00:00" for h in range(1, 40)] * 2, 3600,
                         lambda v: v.endswith(":00:00")),
    "whole-day durations": ("elapsed", [f"{24 * d}:00:00" for d in range(1, 20)] * 3, 86400,
                            lambda v: dur_minutes(v) % 1440 == 0),
    "90-minute durations": ("duration", [f"{90 * k // 60}:{90 * k % 60:02d}" for k in range(1, 30)] * 2, 5400,
                            lambda v: dur_minutes(v) % 90 == 0),
    "7-hourly with an offset": ("shift_ts", [f"{d}T{h:02d}:00:00+05:45" for d in _H for h in (0, 7, 14, 21)], 25200,
                                lambda v: wall(v).hour in (0, 7, 14, 21) and v[14:19] == "00:00"),
    "hourly epoch seconds": ("event_time", [str(1709251200 + 3600 * (i % 200)) for i in range(400)], 3600,
                             lambda v: int(v) % 3600 == 0),
    "date-only and 4-hourly rows": ("event_ts", _H * 2 + [f"{d} {h:02d}:00" for d in _H for h in (8, 12, 16)], 14400,
                                    lambda v: len(v) == 10 or v[11:16] in ("08:00", "12:00", "16:00")),
}


@pytest.mark.parametrize("case", sorted(TIME_GRIDS))
def test_time_values_stay_on_the_observed_grid(case: str) -> None:
    col, vals, step, on_grid = TIME_GRIDS[case]
    for seed in (1, 2, 3):
        p, out = synth(vals, col, seed=seed)
        assert p.params["granularity_seconds"] == pytest.approx(step), case
        off = [v for v in out if not on_grid(v)]
        assert not off, (case, seed, off[:5])
        assert len(out) == len(vals)
        real, syn = np.sort(numeric_view(p, vals)), np.sort(numeric_view(p, out))
        assert real[0] == syn[0] and real[-1] == syn[-1], case  # D14 exact extremes stay
    assert synth(vals, col, seed=7)[1] == synth(vals, col, seed=7)[1]  # seed determinism


def test_mixed_offsets_hourly_grid_holds_at_the_utc_bounds() -> None:
    """Rows clipped to a UTC bound written with another offset move inward onto the hourly local grid."""
    rng = np.random.default_rng(4)
    vals = ["2024-03-01T08:00:00+05:30", "2024-03-30T17:00:00-03:30"]
    for _ in range(400):
        d = dt.date(2024, 3, 1) + dt.timedelta(days=int(rng.integers(0, 30)))
        off = ["+05:30", "+05:45", "-03:30", "+00:00"][int(rng.integers(0, 4))]
        vals.append(f"{d.isoformat()}T{int(rng.integers(8, 18)):02d}:00:00{off}")
    for seed in range(1, 6):
        p, out = synth(vals, "event_time", seed=seed)
        assert p.params["granularity_seconds"] == 3600.0
        assert {wall(v).minute for v in out} == {0} and {wall(v).second for v in out} == {0}
        real, syn = np.sort(numeric_view(p, vals)), np.sort(numeric_view(p, out))
        assert real[0] == syn[0] and real[-1] == syn[-1]


def test_iana_zone_hourly_grid_across_dst() -> None:
    chi = zoneinfo.ZoneInfo("America/Chicago")
    vals = []
    for i in range(240):
        d = dt.date(2024, 2, 20) + dt.timedelta(days=i % 60)  # spans the March DST change
        t = dt.datetime.combine(d, dt.time(7 + i % 10)).replace(tzinfo=chi)
        vals.append(t.strftime("%Y-%m-%d %H:%M:%S ") + "America/Chicago")
    for seed in (1, 2):
        p, out = synth(vals, "class_start", seed=seed)
        loc = [dt.datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S") for v in out]
        assert {x.minute for x in loc} == {0} and {x.hour for x in loc} <= set(range(7, 17))


def test_native_parquet_hourly_timestamps_times_and_durations(tmp_path: Path) -> None:
    base = dt.datetime(2024, 3, 1, 8)
    ts = [base + dt.timedelta(days=i % 20, hours=i % 9) for i in range(90)]
    table = pa.table({
        "event_time": pa.array(ts, type=pa.timestamp("us")),
        "start_time": pa.array([dt.time(7 + i % 11) for i in range(90)], type=pa.time64("us")),
        "duration": pa.array([dt.timedelta(hours=1 + i % 30) for i in range(90)], type=pa.duration("s")),
    })
    src = tmp_path / "in" / "hourly.parquet"
    src.parent.mkdir()
    pq.write_table(table, src)
    out = tmp_path / "out"
    res = run_cli(src, "--out-dir", out, "--report-dir", tmp_path / "rep", "--seed", "5")
    assert res.exit_code == 0, res.output
    syn = pq.read_table(out / "hourly.parquet")
    assert syn.schema.equals(table.schema)
    assert all(v.minute == 0 and v.second == 0 for v in syn["event_time"].to_pylist())
    assert all(v.minute == 0 and v.second == 0 for v in syn["start_time"].to_pylist())
    assert all(v.total_seconds() % 3600 == 0 for v in syn["duration"].to_pylist())


def test_time_grid_helper() -> None:
    hourly = np.array([8, 9, 13, 17]) * 3600 * US
    assert D.time_grid(hourly) == (3600 * US, 0)
    quarter_past = hourly + 15 * 60 * US
    assert D.time_grid(quarter_past) == (3600 * US, 15 * 60 * US)
    assert D.time_grid(np.array([90, 180, 270]) * 60 * US) == (90 * 60 * US, 0)
    assert D.time_grid(np.array([-3600, 7200]) * US) == (3 * 3600 * US, 2 * 3600 * US)  # signed durations
    # one distinct value: the written unit (nothing to generate in between)
    assert D.time_grid(np.array([8 * 3600 * US])) == (D.granularity(np.array([8 * 3600 * US])), 0)
    assert D.time_grid(np.array([], dtype=np.int64))[1] == 0
    # the resolution label stays the written unit
    p, _ = synth([f"2024-01-0{d} {h:02d}:00:00" for d in range(1, 8) for h in range(8, 12)], "ts")
    assert p.params["resolution"] == "minute" and p.params["granularity_seconds"] == 3600.0
    assert "grid_phase_seconds" not in p.params
    p, _ = synth([f"{h:02d}:15" for h in range(7, 18)] * 3, "start_time")
    assert p.params["grid_phase_seconds"] == 900.0


def test_fit_to_grid_restarts_each_day_for_steps_that_do_not_divide_it() -> None:
    step, period = 7 * 3600 * US, DAY  # 00, 07, 14, 21 every day
    day5 = 5 * DAY
    y = np.array([day5 + 3 * 3600 * US, day5 + 22 * 3600 * US, day5 - 3600 * US, day5 + 14 * 3600 * US])
    got = G.fit_to_grid(y, step, 0, 0, 10 * DAY, period)
    hours = ((got % DAY) // (3600 * US)).tolist()
    assert hours == [0, 21, 21, 14]
    assert got.tolist()[2] == day5 - 3 * 3600 * US  # 23:00 the day before -> 21:00 that day
    # bounds: a value below lo moves to the next grid point, above hi to the previous one
    lo, hi = day5 + 7 * 3600 * US, day5 + 21 * 3600 * US
    got = G.fit_to_grid(np.array([day5 + 1, day5 + 23 * 3600 * US]), step, 0, lo, hi, period)
    assert got.tolist() == [lo, hi]
    # a phase-shifted global grid (every hour at :15)
    got = G.fit_to_grid(np.array([10 * 60 * US, 80 * 60 * US]), 3600 * US, 15 * 60 * US, -DAY, DAY)
    assert got.tolist() == [-45 * 60 * US, 75 * 60 * US]


# =========================================================================== R2-13 coarser: calendar grids
_RNG = np.random.default_rng(5)


def _month_end(y: int, m: int) -> dt.date:
    return dt.date(y, m, calendar.monthrange(y, m)[1])


_START = dt.date(2020, 1, 3)
DATE_GRIDS: dict[str, tuple[str, list[str], Callable[[dt.date], bool]]] = {
    "month-precision DOB": ("birth_date", [dt.date(int(y), int(m), 1).isoformat() for y, m in
                                           zip(_RNG.integers(2008, 2014, 300), _RNG.integers(1, 13, 300))],
                            lambda d: d.day == 1),
    "monthly 1st": ("snapshot_date", [dt.date(2000 + k // 12, k % 12 + 1, 1).isoformat() for k in range(240)],
                    lambda d: d.day == 1),
    "month ends": ("period_end", [_month_end(2000 + k // 12, k % 12 + 1).isoformat() for k in range(240)],
                   lambda d: d == _month_end(d.year, d.month)),
    "the 15th": ("pay_date", [dt.date(2000 + k // 12, k % 12 + 1, 15).isoformat() for k in range(240)],
                 lambda d: d.day == 15),
    "biweekly": ("pay_date", [(_START + dt.timedelta(days=14 * k)).isoformat() for k in range(150)],
                 lambda d: (d - _START).days % 14 == 0),
    "quarterly 1st": ("quarter_start", [dt.date(2000 + k // 4, 3 * (k % 4) + 1, 1).isoformat() for k in range(80)],
                      lambda d: d.day == 1 and d.month in (1, 4, 7, 10)),
    "yearly July 1": ("fiscal_start", [dt.date(2000 + k, 7, 1).isoformat() for k in range(25)] * 3,
                      lambda d: (d.month, d.day) == (7, 1)),
    "monthly US m/d/Y": ("snapshot_date", [f"{k % 12 + 1}/1/{2000 + k // 12}" for k in range(240)],
                         lambda d: d.day == 1),
    "monthly YYYYMMDD": ("snapshot_date", [f"{2000 + k // 12}{k % 12 + 1:02d}01" for k in range(240)],
                         lambda d: d.day == 1),
}


def _parse_day(v: str) -> dt.date:
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(v, f).date()
        except ValueError:
            pass
    raise ValueError(v)


@pytest.mark.parametrize("case", sorted(DATE_GRIDS))
def test_dates_stay_on_the_observed_calendar_grid(case: str) -> None:
    col, vals, on_grid = DATE_GRIDS[case]
    assert all(on_grid(_parse_day(v)) for v in vals)
    for seed in (1, 2, 3):
        p, out = synth(vals, col, seed=seed)
        assert not p.params.get("calendar"), case  # generated by interpolation, not observed dates only
        off = [v for v in out if not on_grid(_parse_day(v))]
        assert not off, (case, seed, off[:5])
        assert min(map(_parse_day, out)) == min(map(_parse_day, vals))  # D14 exact extremes
        assert max(map(_parse_day, out)) == max(map(_parse_day, vals))
    assert synth(vals, col, seed=9)[1] == synth(vals, col, seed=9)[1]


def test_month_precision_dob_keeps_exact_extremes_and_spreads_over_months() -> None:
    col, vals, _ = DATE_GRIDS["month-precision DOB"]
    p, out = synth(vals, col)
    assert p.params["person_level"] and p.params["day_of_month"] == 1
    assert len({v[:7] for v in out}) >= 0.8 * len({v[:7] for v in vals})  # still interpolated, not collapsed


def test_monthly_datetimes_and_quarterly_year_months() -> None:
    vals = [dt.datetime(2000 + k // 12, k % 12 + 1, 1).isoformat(sep=" ") for k in range(240)]
    p, out = synth(vals, "snapshot_ts")
    assert p.type is T.DATETIME and all(_parse_day(v).day == 1 and v.endswith("00:00:00") for v in out)
    q = [f"{2000 + k // 4}-{3 * (k % 4) + 1:02d}" for k in range(80)]
    p, out = synth(q, "report_month")
    assert p.type is T.YEAR_MONTH and p.params["month_grid"] == [3, 0]
    assert all(int(v[5:]) in (1, 4, 7, 10) for v in out) and min(out) == min(q) and max(out) == max(q)


def test_date_grid_detection() -> None:
    days = D.days_from_civil(np.array([2024, 2024, 2024]), np.array([1, 2, 3]), np.array([1, 1, 1]))
    assert D.date_grid(days) == {"day_of_month": 1}
    ends = D.days_from_civil(np.array([2024, 2024, 2024]), np.array([1, 2, 3]), np.array([31, 29, 31]))
    assert D.date_grid(ends) == {"month_end": True}
    two = D.days_from_civil(np.array([2024, 2024]), np.array([1, 2]), np.array([31, 29]))
    assert D.date_grid(two)["day_grid"][0] == 29  # two dates: every grid point is an observed date
    q = D.days_from_civil(np.array([2024, 2024, 2025]), np.array([1, 4, 1]), np.array([1, 1, 1]))
    assert D.date_grid(q)["month_grid"] == [3, (2024 - 1970) * 12 % 3]
    biweekly = np.array([0, 14, 42]) + 19000
    assert D.date_grid(biweekly) == {"day_grid": [14, 19000 % 14]}
    assert D.date_grid(np.array([19000])) == {}
    rng = np.random.default_rng(1)
    assert D.date_grid(rng.integers(18000, 19000, 200)) == {}  # ordinary event dates: no grid
    assert D.month_length(np.array([2023, 2024, 1900, 2000]), np.array([2, 2, 2, 2])).tolist() == [28, 29, 28, 29]


def test_grid_days_respects_observed_weekdays() -> None:
    lo = int(D.days_from_civil(np.array([2024]), np.array([1]), np.array([1]))[0])
    hi = int(D.days_from_civil(np.array([2024]), np.array([12]), np.array([1]))[0])
    grid = G.grid_days({"day_of_month": 1}, lo, hi, [0, 1, 2, 3, 4])
    y, m, d = D.civil_from_days(grid)
    assert set(d.tolist()) == {1} and set(D.weekday(grid).tolist()) <= {0, 1, 2, 3, 4}
    assert G.grid_days({}, lo, hi, [0]) is None
    assert G.snap_to(np.array([lo - 5, lo + 3, hi + 9]), np.array([lo, lo + 6, hi])).tolist() == [lo, lo, hi]
    assert G.snap_to(np.array([lo + 3]), np.array([lo, lo + 6])).tolist() == [lo]  # tie: the earlier day


# =========================================================================== R2-14 case forms
def _forms(word: str) -> list[str]:
    return [word.lower(), word.upper(), word.title(), word.swapcase(), word[:2].upper() + word[2:].lower()]


def _check_kept(real: list[str], syn: list[str]) -> None:
    assert len(syn) == len(real)
    assert len(set(syn)) == len(set(real)), (Counter(syn), Counter(real))
    assert sorted(Counter(syn).values()) == sorted(Counter(real).values())
    real_cf = {v.strip().casefold() for v in real}
    assert not real_cf & {v.strip().casefold() for v in syn}  # never a real value


def test_reviewer_case_mixed_case_variants_stay_distinct(tmp_path: Path) -> None:
    for col in ("comment", "first_name"):
        rows = [[v, i] for i in range(20) for v in ["aLiCe", "alICe", "AliCE"]]
        src = write_csv(tmp_path / col / "in" / "data.csv", [col, "score"], rows)
        out = tmp_path / col / "out"
        res = run_cli(src.parent, "--out-dir", out, "--report-dir", tmp_path / col / "rep", "--seed", "123")
        assert res.exit_code == 0, res.output
        syn = [r[col] for r in read_dicts(out / "data.csv")]
        _check_kept([r[0] for r in rows], syn)
        assert sorted(Counter(syn).values()) == [20, 20, 20]
        assert all(case_style(v) == "mixed" for v in syn)  # each form keeps its own (mixed) case pattern
        res = run_cli(src.parent, "--out-dir", tmp_path / col / "o2", "--report-dir", tmp_path / col / "r2",
                      "--seed", "123")
        assert hashes(tmp_path / col / "o2") == hashes(out)


TEXT_CASES: dict[str, tuple[str, list[str], dict[str, Any]]] = {
    "free text": ("notes", [x for _ in range(15) for x in ["Needs Glasses Daily", "needs GLASSES daily",
                                                            "NEEDS glasses DAILY", "Needs glasses daily"]], {}),
    "comment with digits": ("comment", [x for _ in range(20) for x in ["Ab12", "aB12", "AB12", "ab12"]], {}),
    "comment with whitespace": ("comment", [x for _ in range(20) for x in ["aLiCe", "alICe", " aLiCe", "aLiCe "]],
                                {}),
    "first name": ("first_name", [x for _ in range(20) for x in ["aLiCe", "alICe", "AliCE", "ALICE"]], {}),
    "last name": ("last_name", [x for _ in range(20) for x in ["McDonald", "MCDONALD", "Mcdonald", "mcdonald",
                                                                "McDONALD"]], {}),
    "apostrophe name": ("last_name", [x for _ in range(20) for x in ["O'Brien", "O'brien", "o'brien"]], {}),
    "full name": ("student_name", [x for _ in range(20) for x in ["Alice Smith", "ALICE smith", "alice SMITH",
                                                                  "Alice SMITH"]], {}),
    "email": ("email", [x for _ in range(20) for x in ["John.Smith@Gmail.com", "john.smith@gmail.com",
                                                       "JOHN.SMITH@GMAIL.COM", "john.SMITH@gmail.com"]], {}),
    "city": ("city", [x for _ in range(20) for x in ["New York", "NEW YORK", "new york", "New YORK", "nEW yORK"]],
             {}),
    "address": ("address", [x for _ in range(20) for x in ["12 Main St", "12 MAIN ST", "12 main st", "12 Main ST"]],
                {}),
    "url": ("website", [x for _ in range(20) for x in ["http://Example.COM/Path", "http://example.com/path",
                                                       "HTTP://EXAMPLE.COM/PATH", "http://example.com/PATH"]], {}),
    "invented labels": ("career_goal", [f for i in range(25) for f in _forms(f"Career Path {chr(65 + i)}x")] * 2,
                        {}),
    "code masks": ("locker_code", [x for j in range(30) for x in [f"Ab-{j:04d}", f"aB-{j:04d}", f"AB-{j:04d}"]] * 2,
                   {}),
    "forced state masks": ("region", [x for _ in range(20) for x in ["aLpHa", "AlPhA", "ALPHA", "alpha"]],
                           {"pii_col": ["region=state"]}),
}


@pytest.mark.parametrize("case", sorted(TEXT_CASES))
def test_every_text_generator_keeps_case_forms_apart(case: str) -> None:
    col, vals, cli = TEXT_CASES[case]
    for seed in (1, 2):
        tp, cols = run_table({col: vals, "score": [str(i) for i in range(len(vals))]}, seed=seed, **cli)
        p = profile_of(tp, col)
        assert p.type in (T.FREE_TEXT, T.PERSON_NAME, T.EMAIL, T.CITY, T.STREET_ADDRESS, T.URL,
                          T.HIGH_CARD_CATEGORICAL, T.STATE), (case, p.type)
        syn = list(cols[col])
        _check_kept(vals, syn)
    again = run_table({col: vals, "score": [str(i) for i in range(len(vals))]}, seed=2, **cli)[1][col]
    assert list(again) == syn  # seed determinism


def test_invented_labels_keep_each_forms_case_style() -> None:
    col, vals, _ = TEXT_CASES["invented labels"]
    tp, cols = run_table({col: vals}, seed=3)
    assert profile_of(tp, col).type is T.HIGH_CARD_CATEGORICAL
    assert Counter(map(case_style, cols[col])) == Counter(map(case_style, vals))


def test_single_form_values_are_rendered_as_before() -> None:
    """A value with one raw form keeps the generator's own rendering (no change for ordinary columns)."""
    groups = [textgen.Group("alice", "Alice", [("Alice", 3)], 3), textgen.Group("bob", "BOB", [(" BOB", 2)], 2)]
    labels, counts = textgen.render_groups(groups, ["Nobis", "QUAM"])
    assert labels == ["Nobis", " QUAM"] and counts == [3, 2]


def test_style_variants_guarantees_distinct_renderings() -> None:
    word = "abc"
    forms = sorted({"".join(c.upper() if b >> i & 1 else c for i, c in enumerate(word)) for b in range(8)})
    assert len(forms) == 8
    out = textgen.style_variants("xyz", forms, forms[0], None)
    assert len(set(out.values())) == 8 and {v.casefold() for v in out.values()} == {"xyz"}
    for f in forms:  # upper / lower / title forms keep their style exactly
        if case_style(f) in ("upper", "lower", "title"):
            assert case_style(out[f]) == case_style(f)
    # a base with too few letters for every form: fresh synthetic values fill in, all distinct
    fresh_values = iter(["f1q", "f2r", "f3s", "f4t", "f5u", "f6v", "f7w", "f8x"])
    out = textgen.style_variants("x1", forms, forms[0], lambda: next(fresh_values))
    assert len(set(out.values())) == 8 and len({v.casefold() for v in out.values()}) == 7
    with pytest.raises(Exception):
        textgen.style_variants("12", ["ab", "AB"], "ab", None)


def test_label_without_letters_falls_back_to_fresh_labels() -> None:
    """A digits-only header gives invented labels with no letters to re-case: other labels keep forms apart."""
    vals = [f for i in range(25) for f in [f"Alpha {chr(65 + i)}ix", f"ALPHA {chr(65 + i)}IX",
                                           f"alpha {chr(97 + i)}ix"]] * 2
    tp, cols = run_table({"2024": vals}, seed=1)
    assert profile_of(tp, "2024").type is T.HIGH_CARD_CATEGORICAL
    _check_kept(vals, list(cols["2024"]))


def test_multi_select_items_keep_their_case_forms() -> None:
    rng = np.random.default_rng(3)
    items = [f"Item{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXY"]
    rows = []
    for _ in range(400):
        chosen = list(rng.choice(items, size=int(rng.integers(1, 4)), replace=False))
        rows.append(";".join(c.upper() if rng.random() < 0.3 else c for c in chosen))
    tp, cols = run_table({"programs": rows}, seed=1, force_type=["programs=MULTI_SELECT"])
    assert profile_of(tp, "programs").type is T.MULTI_SELECT
    real = Counter(it for v in rows for it in v.split(";"))
    syn = Counter(it for v in cols["programs"] for it in v.split(";"))
    assert syn == real  # every raw item form at its exact count (none folded into its most common form)


def test_name_redraw_keeps_case_forms_apart(tmp_path: Path) -> None:
    """The run-wide real-full-name redraw (R1-04) re-renders every case form of a redrawn value apart."""
    d = tmp_path / "in"
    forms = ["Zelmira Quenby", "ZELMIRA QUENBY", "zelmira QUENBY"]
    write_csv(d / "people.csv", ["teacher_name", "score"], [[f, i] for i in range(20) for f in forms])
    res = run_cli(d, "--out-dir", tmp_path / "o1", "--report-dir", tmp_path / "r1", "--seed", "123")
    assert res.exit_code == 0, res.output
    first = [r["teacher_name"] for r in read_dicts(tmp_path / "o1" / "people.csv")]
    words = next(v for v in first if case_style(v) == "title").split(" ", 1)
    # another file now holds the synthetic name as a real person: the redraw must pick another name
    write_csv(d / "students.csv", ["first_name", "last_name"], [words])
    res = run_cli(d, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "123")
    assert res.exit_code == 0, res.output
    second = [r["teacher_name"] for r in read_dicts(tmp_path / "o2" / "people.csv")]
    assert " ".join(words).casefold() not in {v.casefold() for v in second}
    assert sorted(Counter(second).values()) == [20, 20, 20]
    assert len({v.casefold() for v in second}) == 1
    assert Counter(map(case_style, second)) == Counter(map(case_style, first))
