"""Review round 1, dates cluster: nanosecond precision (R1-11), fraction-digit variants and the exact
extremes they carry (R1-12), and the local wall-time grid, weekdays and calendar of offset datetimes (R1-13)."""

from __future__ import annotations

import datetime as dt
import zoneinfo
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import run_cli
from datanonymous.detect import dates as D
from datanonymous.detect import dates_formats as F
from datanonymous.generate import dates as G
from datanonymous.model import ColumnType as T
from datanonymous.profile.stats import numeric_view

from test_area_a_support import profile_of, run_table


def synth(values: list[str], name: str = "col", seed: int = 1, **cli):
    tp, cols = run_table({name: values}, seed=seed, **cli)
    return profile_of(tp, name), list(cols[name])


def frac_digits(text: str) -> int:
    """Fraction digits of the seconds in a datetime/time/duration text (0 when none)."""
    core = text.rstrip("S").split("+")[0].rstrip("Z")
    return len(core.split(".")[1]) if "." in core else 0


# --------------------------------------------------------------------------- R1-11 nanoseconds
def _ns_table() -> pa.Table:
    return pa.table({
        "event_time": pa.array([1704110400000000000 + i for i in range(30)], type=pa.int64()).cast(pa.timestamp("ns")),
        "clock": pa.array([43200000000000 + i for i in range(30)], type=pa.int64()).cast(pa.time64("ns")),
        "duration": pa.array([10000000000000 + i for i in range(30)], type=pa.int64()).cast(pa.duration("ns")),
    })


def test_parquet_nanoseconds_keep_their_precision_and_exact_extremes(tmp_path: Path) -> None:
    src = tmp_path / "in" / "ns.parquet"
    src.parent.mkdir()
    table = _ns_table()
    pq.write_table(table, src)
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 0, res.output
    syn = pq.read_table(out / "ns.parquet")
    assert syn.schema.equals(table.schema)
    for name in table.column_names:
        real = table[name].cast(pa.int64()).to_pylist()
        got = syn[name].cast(pa.int64()).to_pylist()
        assert len(got) == len(real)
        assert min(got) == min(real) and max(got) == max(real), name  # D14: the exact ns extremes
        assert len(set(got)) > 10, name  # not collapsed onto one microsecond
        assert all(min(real) <= g <= max(real) for g in got)


def test_text_nanosecond_fractions_are_modelled_in_nanoseconds() -> None:
    vals = [f"2024-01-01T12:00:00.{i * 7919:09d}" for i in range(1, 60)]
    p, out = synth(vals, "event_time")
    assert p.type is T.DATETIME and p.params["time_unit"] == "ns"
    # the written unit is the nanosecond; the values lie on an observed 7919 ns grid, which they keep (R2-13)
    assert p.params["resolution"] == "nanosecond" and p.params["granularity_seconds"] == 7.919e-06
    assert all(int(v.rsplit(".", 1)[1]) % 7919 == 0 for v in out)
    assert all(frac_digits(v) == 9 for v in out)
    assert min(out) == min(vals) and max(out) == max(vals)
    assert len({v[-3:] for v in out}) > 5  # the sub-microsecond digits vary, as in the source
    assert p.stats["max"] == "2024-01-01T12:00:00.000467221"  # report statistics keep the 9 digits


def test_nanosecond_times_and_negative_durations() -> None:
    times = [f"08:00:00.{i:09d}" for i in range(1, 40)]
    p, out = synth(times, "start_time")
    assert p.type is T.TIME and p.params["time_unit"] == "ns"
    assert min(out) == min(times) and max(out) == max(times)
    durs = [f"-0:00:00.000000{i:03d}" for i in range(100, 160)] + [f"0:00:01.{i:09d}" for i in range(50)]
    p2, out2 = synth(durs, "elapsed")
    assert p2.type is T.DURATION and p2.params["time_unit"] == "ns"
    real = sorted(D.view(p2.params, durs).dur.tolist())
    got = sorted(D.view(p2.params, out2).dur.tolist())
    assert got[0] == real[0] == -159 and got[-1] == real[-1] == 1_000_000_049
    assert any(v.startswith("-0:00:00.000000") for v in out2)  # sub-microsecond negatives keep their sign


def test_nanoseconds_outside_the_int64_range_fall_back_to_microseconds() -> None:
    vals = ([f"2024-01-{d:02d}T12:00:00.{i:09d}" for d in range(1, 28) for i in range(1, 6)]
            + [f"2500-01-{d:02d}T12:00:00.{i:09d}" for d in range(1, 28) for i in range(1, 6)])
    p, out = synth(vals, "event_time")
    assert p.type is T.DATETIME and "time_unit" not in p.params
    assert any(m.code == "date_subus_truncated" for m in p.messages)
    v = D.view(p.params, out)
    assert np.all(v.cls >= 0)


def test_nanosecond_parse_and_render_helpers() -> None:
    f = F.get_format("%H:%M:%S.%f")
    p = f.parse(["12:00:00.123456789", "12:00:00.5", "12:00:00.000000001"])
    assert p.micro.tolist() == [123456, 500000, 0] and p.nano.tolist() == [789, 0, 1]
    c = F.Components.from_local(np.zeros(2, dtype=np.int64), np.array([43_200_123_456_789_000 // 1000, 1]), tpu=1000)
    assert f.render(c, {"frac_digits": 9}).tolist() == ["12:00:00.123456789", "00:00:00.000000001"]
    assert F.frac_value(123456, 789, 7) == 1234567 and F.frac_value(123456, 789, 3) == 123
    assert F.hms(1, tpu=1000) == "00:00:00.000000001" and F.hms(1) == "00:00:00.000001"


# --------------------------------------------------------------------------- R1-12 fraction-digit variants
def test_minority_fraction_digits_and_the_exact_maximum_are_written() -> None:
    vals = [f"2024-01-01T12:00:{i:02d}.{i % 10}" for i in range(30)] + ["2024-01-01T12:00:59.123456"]
    p, out = synth(vals, "event_time")
    render = p.params["formats"][0]["render"]
    assert render["frac_digits"] == 1 and render["frac_digits_pmf"] == {"1": 30, "6": 1}
    assert max(out) == "2024-01-01T12:00:59.123456" and min(out) == min(vals)  # D14
    assert Counter(frac_digits(v) for v in out) == Counter({1: 30, 6: 1})  # exact quota of the layouts
    assert p.flags["extreme_value_adjusted"]


def test_iso_duration_fraction_variants_and_exact_extremes() -> None:
    vals = [f"PT{i}.1S" for i in range(1, 30)] + ["PT59.123456S"]
    p, out = synth(vals, "duration")
    assert p.type is T.DURATION
    real, syn = numeric_view(p, vals), numeric_view(p, out)
    assert syn.max() == real.max() == 59.123456 and syn.min() == real.min()
    assert "PT59.123456S" in out
    assert Counter(frac_digits(v) for v in out) == Counter({1: 29, 6: 1})


def test_bare_seconds_keep_their_share_next_to_fractions() -> None:
    """ISO durations: 'PT5S' (no fraction) and 'PT5.25S' are two layouts of one format, at their shares."""
    vals = [f"PT{i % 50}S" for i in range(1, 91)] + [f"PT{i}.25S" for i in range(1, 11)]
    p, out = synth(vals, "duration")
    assert p.params["formats"][0]["render"]["frac_digits_pmf"] == {"0": 90, "2": 10}
    assert Counter(frac_digits(v) for v in out) == Counter({0: 90, 2: 10})


def test_single_digit_count_columns_are_unchanged() -> None:
    vals = [f"2024-03-{d:02d}T10:15:{s:02d}.{s * 13 % 1000:03d}Z" for d in range(1, 29) for s in range(0, 60, 7)]
    p, out = synth(vals)
    assert "frac_digits_pmf" not in p.params["formats"][0]["render"] and "time_unit" not in p.params
    assert all(frac_digits(v) == 3 for v in out)


def test_extreme_layout_without_a_row_is_given_to_the_nearest_row() -> None:
    """Within-ID constant columns are generated per entity: a precision held by one real row may get no
    generated row, so the row nearest to the extreme takes the layout the source wrote it in."""
    ids, vals = [], []
    for k in range(80):
        v = "2024-01-01T12:00:59.123456" if k == 79 else f"2024-01-01T12:00:{k % 59:02d}.{k % 10}"
        for _ in range(3):
            ids.append(f"{100000 + k}")
            vals.append(v)
    tp, cols = run_table({"student_id": ids, "enroll_time": vals})
    p = profile_of(tp, "enroll_time")
    out = list(cols["enroll_time"])
    assert p.role == "entity"
    assert max(out) == "2024-01-01T12:00:59.123456" and min(out) == min(vals)


def test_writable_extremes_helper_converts_a_row_layout() -> None:
    params = {"formats": [{"format": "%H:%M:%S.%f", "render": {"frac_digits": 1}}]}
    y = np.array([0, 100_000, 200_000, 300_000], dtype=np.int64)
    fmt = np.zeros(4, dtype=np.int16)
    fd = np.full(4, -1, dtype=np.int8)
    tz = np.full(4, "", dtype=object)
    res = G.row_resolution(params, fmt, fd)
    assert res.tolist() == [100_000] * 4
    hi = 312_345
    layouts = {0: (0, 1, ""), hi: (0, 6, "")}
    assert G._writable_extremes(y, 0, hi, res, layouts, params, fmt, fd, tz, 1)
    assert hi in y.tolist() and 0 in y.tolist()
    i = y.tolist().index(hi)
    assert fd[i] == 6 and res[i] == 1


# --------------------------------------------------------------------------- R1-13 offsets: local grid, weekdays
def _local(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text).replace(tzinfo=None)


def test_offset_not_a_multiple_of_the_local_grid_keeps_the_local_grid() -> None:
    """+05:45 local times every 30 minutes are a 15-minute grid in UTC: the written times stay on :00/:30."""
    vals = [f"2024-03-{d:02d}T{h:02d}:{mi:02d}:00+05:45"
            for d in range(4, 30) for h in (8, 9, 10, 11) for mi in (0, 30)]
    p, out = synth(vals, "event_time")
    assert p.type is T.DATETIME_TZ and p.params["granularity_seconds"] == 1800.0
    assert {_local(v).minute for v in out} == {0, 30}
    assert all(v.endswith("+05:45") for v in out)
    real, syn = np.sort(numeric_view(p, vals)), np.sort(numeric_view(p, out))
    assert real[0] == syn[0] and real[-1] == syn[-1]  # exact UTC extremes


def test_offset_weekdays_are_the_local_weekdays() -> None:
    """School hours Mon-Fri at +09:00 start on the previous UTC day: no local weekend may appear."""
    rows = []
    for k in range(56):
        d = dt.date(2024, 3, 4) + dt.timedelta(days=k)
        if d.weekday() < 5:
            rows += [f"{d.isoformat()}T{h:02d}:00:00+09:00" for h in range(8, 16)]
    for seed in (1, 2, 3):
        p, out = synth(rows, "event_time", seed=seed)
        assert p.params["weekdays_observed"] == [0, 1, 2, 3, 4]
        assert {_local(v).weekday() for v in out} <= {0, 1, 2, 3, 4}
        assert {_local(v).hour for v in out} <= set(range(8, 16))


def test_offset_calendar_columns_use_observed_local_dates() -> None:
    days = [dt.date(2024, 3, 4) + dt.timedelta(days=k) for k in range(0, 40, 3)]
    rows = [f"{d.isoformat()}T{h:02d}:{mi:02d}:00-05:00" for d in days for h in (8, 12, 19, 21) for mi in (0, 15, 30)]
    p, out = synth(rows, "attendance_time")
    assert p.params["calendar"]
    assert {_local(v).date() for v in out} <= set(days)


def test_mixed_offsets_keep_the_local_grid_at_the_bounds() -> None:
    """Rows clipped to a UTC bound written with another offset move inward onto their local grid."""
    rng = np.random.default_rng(11)
    vals = ["2024-03-01T08:00:00+05:30", "2024-03-30T11:30:00+05:30"]
    for _ in range(400):
        d = dt.date(2024, 3, 1) + dt.timedelta(days=int(rng.integers(0, 30)))
        h, mi = int(rng.integers(8, 12)), int(rng.integers(0, 2)) * 30
        vals.append(f"{d.isoformat()}T{h:02d}:{mi:02d}:00{'+05:30' if rng.random() < 0.5 else '+05:45'}")
    for seed in range(1, 6):
        p, out = synth(vals, "event_time", seed=seed)
        assert {_local(v).minute for v in out} <= {0, 30}
        real, syn = np.sort(numeric_view(p, vals)), np.sort(numeric_view(p, out))
        assert real[0] == syn[0] and real[-1] == syn[-1]
        assert syn[0] >= real[0] and syn[-1] <= real[-1]
    assert Counter(v[-6:] for v in out) == Counter(v[-6:] for v in vals)  # offset pmf reapplied exactly


def test_iana_zone_names_keep_local_hours_weekdays_and_grid() -> None:
    chi = zoneinfo.ZoneInfo("America/Chicago")
    vals = []
    for i in range(300):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=int(i * 1.2))
        if d.weekday() >= 5:
            continue
        t = dt.datetime.combine(d, dt.time(8 + i % 9, (i % 2) * 30)).replace(tzinfo=chi)
        vals.append(t.strftime("%Y-%m-%d %H:%M:%S ") + "America/Chicago")
    p, out = synth(vals, "class_start")
    loc = [dt.datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S") for v in out]
    assert {x.weekday() for x in loc} <= {0, 1, 2, 3, 4}
    assert {x.minute for x in loc} <= {0, 30} and {x.hour for x in loc} <= set(range(8, 17))
    real, syn = np.sort(numeric_view(p, vals)), np.sort(numeric_view(p, out))
    assert real[0] == syn[0] and real[-1] == syn[-1]


def test_utc_only_columns_are_generated_as_before() -> None:
    """With offset 0 (Z) the local and UTC models coincide; seeded output is stable across runs."""
    vals = [f"2024-03-{d:02d}T{h:02d}:{m:02d}:00Z" for d in range(1, 29) for h in (8, 13) for m in (0, 20, 40)]
    a = synth(vals, "ts", seed=4)[1]
    assert a == synth(vals, "ts", seed=4)[1]
    p, _ = synth(vals, "ts", seed=4)
    assert p.params["weekdays_observed"] == sorted({dt.date(2024, 3, d).weekday() for d in range(1, 29)})
