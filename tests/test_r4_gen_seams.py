"""Round-4 adversarial review, gen-seams cluster (R4-06, R4-07).

R4-06 - bug class: a generated value's UTC offset was drawn by exact quota over the observed offsets,
independently of its date, so the (local time, offset) pairs no longer followed the source's time zone. A
Parquet ``timestamp[us, tz=America/Chicago]`` column of daily 09:00 events was written back as 08:00, 09:00 and
10:00 local times (the writer converts each instant into the column's zone), while the report computed its
statistics from the generated text and said PASS. Variants: native named zones (Parquet and Feather; US,
European and southern-hemisphere DST; microsecond and nanosecond units; ``CST6CDT``-style names), an hourly
grid across DST changes, within-ID constant timestamps (D4), declared rules (re-rendered values), and text
columns whose ISO offsets (``-06:00``/``-05:00``) or abbreviations (``CST``/``CDT``) follow one zone's DST
rules. Excel and Stata have no zoned temporal types, so they have no equivalent. The structural fix:

* ``detect.dates.zone_rule`` names the zone whose rules give a column's offsets (the native zone, or one
  inferred from text offsets that change with DST and match a zone exactly; mixed-zone columns get none);
* the generator places each local wall time in that zone (offsets derived, never drawn; a wall time a DST
  change skips moves onto the nearest existing grid time), and ``render_dates`` writes every value with the
  zone's offset at its instant, in the observed token style - so every path that renders dates is covered;
* the pipeline computes a native date/time column's statistics and fidelity from the re-read output and
  warns (``written_values_differ``) when the writer stored a value differently from the generated text.

R4-07 - bug class: free text rebuilt e-mail addresses letter run by letter run (``magnam.amet@mollitia.sit``),
bypassing D24's reserved domains; the same happened to URLs and web domains, and random digits could form a
real-looking phone number, SSN or IP address. Variants: FREE_TEXT notes (e-mail incl. Unicode local parts, URL,
bare domain, NANP / bare / international phone, SSN, IPv4, street address), multi-select items, code masks of
copy-gated values, numeric tokens, digit-only phone numbers under a contact header, D33 prose, a name with an
e-mail inside, and the Drawer's fallback masks. The structural fix (``generate.embedded``): spans are found by
one function and synthesised by their PII generator's rules (one synthetic span per real span of a column),
``mask_fill``/``mimic_words``/``digit_mask`` never rebuild a span character by character, and every candidate
the ``Drawer`` accepts passes ``embedded.guard``.

False-refusal checks: ordinary notes (scores, rooms, times, abbreviations, file names, versions) hold no span;
single-offset and mixed-zone text keep their behaviour; example-data output is unchanged.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
import pytest

from conftest import run_cli
from datanonymous.detect import dates as D
from datanonymous.detect.dates_formats import US, days_from_civil, zone_local_offsets, zone_offsets_for_local
from datanonymous.detect.text import is_reserved_domain
from datanonymous.generate import embedded, pseudonyms, text
from datanonymous.generate import dates as G
from datanonymous.generate.pii import generate_pii
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType

from test_area_a_support import detect, profile_of, run_table

T = ColumnType


# =========================================================================== helpers
def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "42"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def file_report(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    for t in file_report(rep, name)["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def message_codes(rep: Path, name: str) -> list[str]:
    fr = file_report(rep, name)
    codes = [m["code"] for m in fr.get("messages") or []]
    for t in fr["tables"]:
        codes += [m["code"] for m in t.get("messages") or []]
        for c in t["columns"]:
            codes += [m["code"] for m in c.get("messages") or []]
    return codes


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def zoned(zone: str, local: list[dt.datetime], unit: str = "us") -> pa.Array:
    idx = pd.DatetimeIndex(local).tz_localize(zone)
    return pa.Array.from_pandas(idx).cast(pa.timestamp(unit, tz=zone))


def daily(start: dt.datetime, n: int, weekdays_only: bool = False) -> list[dt.datetime]:
    out: list[dt.datetime] = []
    d = start
    while len(out) < n:
        if not weekdays_only or d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def ids(n: int) -> list[str]:
    return [f"S{1000 + i}" for i in range(n)]


# =========================================================================== R4-06: named zones
ZONE_CASES = [
    ("America/Chicago", "us", "parquet"),  # the review's fixture
    ("America/Chicago", "us", "feather"),
    ("Europe/London", "ms", "parquet"),  # DST on other dates, offset +00:00 in winter
    ("Australia/Sydney", "us", "parquet"),  # southern hemisphere: DST ends in April
    ("America/New_York", "ns", "parquet"),  # nanosecond unit
    ("CST6CDT", "us", "parquet"),  # a rule zone named without a slash
]


@pytest.mark.parametrize("zone, unit, fmt", ZONE_CASES)
def test_native_named_zone_keeps_the_local_grid_across_dst(tmp_path: Path, zone: str, unit: str, fmt: str) -> None:
    local = daily(dt.datetime(2025, 1, 1, 9, 0), 200, weekdays_only=True)
    tab = pa.table({"student_id": ids(200), "event_time": zoned(zone, local, unit)})
    src = tmp_path / "in" / f"events.{fmt}"
    src.parent.mkdir()
    (pq.write_table if fmt == "parquet" else pf.write_feather)(tab, src)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    got = (pq.read_table if fmt == "parquet" else pf.read_table)(out / src.name)
    assert got.schema.field("event_time").type == tab.schema.field("event_time").type
    real = tab["event_time"].to_pylist()
    syn = got["event_time"].to_pylist()
    assert len(syn) == len(real)  # N in = N out
    assert Counter((d.hour, d.minute) for d in syn) == {(9, 0): 200}  # was 08:00 / 09:00 / 10:00
    assert {d.weekday() for d in syn} <= {0, 1, 2, 3, 4}
    assert min(syn) == min(real) and max(syn) == max(real)  # D14 exact extremes
    z = ZoneInfo(zone)
    assert all(d.utcoffset() == d.astimezone(z).utcoffset() for d in syn)
    assert "written_values_differ" not in message_codes(rep, src.name)
    col = column_report(rep, src.name, "event_time")
    assert set(col["synthetic"]["tz_pmf"]) <= set(col["real"]["tz_pmf"])
    assert col["fidelity"]["verdict"] in ("ok", "adjusted")


def test_hourly_grid_across_both_dst_changes(tmp_path: Path) -> None:
    zone = "America/Chicago"
    local = [dt.datetime(2025, 2, 3) + dt.timedelta(days=d, hours=h)
             for d in range(0, 300) for h in range(8, 16) if (dt.date(2025, 2, 3) + dt.timedelta(days=d)).weekday() < 5]
    tab = pa.table({"student_id": ids(len(local)), "scan_time": zoned(zone, local)})
    src = tmp_path / "in" / "scans.parquet"
    src.parent.mkdir()
    pq.write_table(tab, src)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    syn = pq.read_table(out / "scans.parquet")["scan_time"].to_pylist()
    assert {d.minute for d in syn} == {0}
    assert {d.hour for d in syn} <= set(range(8, 16))  # never 07:00 or 16:00
    assert {d.weekday() for d in syn} <= {0, 1, 2, 3, 4}


def test_within_id_constant_timestamp_in_a_named_zone(tmp_path: Path) -> None:
    zone = "America/Denver"
    n_ids = 120
    enrolled = daily(dt.datetime(2024, 8, 20, 7, 30), n_ids)
    rows_id, rows_t = [], []
    for i in range(n_ids):
        for _ in range(3):
            rows_id.append(f"S{1000 + i}")
            rows_t.append(enrolled[i] + dt.timedelta(days=0))
    tab = pa.table({"student_id": rows_id, "enrolled_at": zoned(zone, rows_t),
                    "score": pa.array([float(i % 17) for i in range(len(rows_id))])})
    src = tmp_path / "in" / "enrol.parquet"
    src.parent.mkdir()
    pq.write_table(tab, src)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert column_report(rep, "enrol.parquet", "enrolled_at")["role"] == "entity"
    syn = pq.read_table(out / "enrol.parquet")["enrolled_at"].to_pylist()
    assert Counter((d.hour, d.minute) for d in syn) == {(7, 30): len(rows_t)}


def test_text_offsets_following_a_zone_are_derived_from_it(tmp_path: Path) -> None:
    ny = ZoneInfo("America/New_York")
    vals = [(dt.datetime(2025, 1, 2, 10, 30) + dt.timedelta(days=i)).replace(tzinfo=ny).isoformat() for i in range(250)]
    src = write_csv(tmp_path / "in" / "log.csv", {"student_id": ids(250), "logged_at": vals})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert column_report(rep, "log.csv", "logged_at")["params"]["zone_rule"] == "America/New_York"
    syn = [r["logged_at"] for r in read_csv(out / "log.csv")]
    for s in syn:
        t = dt.datetime.fromisoformat(s)
        assert (t.hour, t.minute) == (10, 30) and t.utcoffset() == t.astimezone(ny).utcoffset(), s
    assert {s[-6:] for s in syn} == {"-05:00", "-04:00"}


def test_zone_abbreviations_follow_the_zone() -> None:
    chicago = ZoneInfo("America/Chicago")
    vals = []
    for i in range(240):
        t = (dt.datetime(2025, 1, 6, 9, 0) + dt.timedelta(days=i)).replace(tzinfo=chicago)
        vals.append(t.strftime("%Y-%m-%d %H:%M ") + t.tzname())
    tp, cols = run_table({"student_id": ids(240), "event_time": vals})
    p = profile_of(tp, "event_time")
    assert p.params["zone_rule"] == "America/Chicago"
    for s in cols["event_time"].tolist():
        day = dt.datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=chicago)
        assert s.endswith(" " + day.tzname()) and s[11:16] == "09:00", s


def test_mixed_zones_keep_their_token_shares() -> None:
    r = random.Random(3)
    vals = [(dt.datetime(2025, 1, 1, 10, 30) + dt.timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S")
            + r.choice(["-05:00", "-08:00"]) for i in range(200)]
    tp, cols = run_table({"student_id": ids(200), "logged_at": vals})
    assert "zone_rule" not in profile_of(tp, "logged_at").params  # no single zone explains the offsets
    syn_off, real_off = Counter(s[-6:] for s in cols["logged_at"].tolist()), Counter(s[-6:] for s in vals)
    assert set(syn_off) == set(real_off) and all(abs(syn_off[k] - real_off[k]) <= 1 for k in real_off)
    assert {s[11:16] for s in cols["logged_at"].tolist()} == {"10:30"}


def test_zone_name_tokens_keep_their_local_times() -> None:
    """Text with IANA zone names already carries its rules per token (unchanged; vectorised offsets)."""
    vals = [(dt.datetime(2025, 1, 6, 9, 0) + dt.timedelta(days=i)).strftime("%Y-%m-%d %H:%M ") + "America/Chicago"
            for i in range(200)]
    tp, cols = run_table({"student_id": ids(200), "event_time": vals})
    assert "zone_rule" not in profile_of(tp, "event_time").params
    assert {s[11:] for s in cols["event_time"].tolist()} == {"09:00 America/Chicago"}


@pytest.mark.parametrize("tokens", [["Z"], ["+00:00"], ["-05:00"], ["+05:30"]])
def test_single_offset_columns_need_no_zone(tokens: list[str]) -> None:
    vals = [(dt.datetime(2025, 1, 1, 8, 15) + dt.timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S") + tokens[0]
            for i in range(120)]
    det = detect(vals, "logged_at")
    assert det.type is T.DATETIME_TZ and "zone_rule" not in det.params


@pytest.mark.parametrize("tz", ["UTC", "+05:30"])
def test_native_fixed_offset_zones_are_unchanged(tmp_path: Path, tz: str) -> None:
    local = daily(dt.datetime(2025, 1, 1, 9, 45), 120)
    off = dt.timedelta(hours=5, minutes=30) if tz == "+05:30" else dt.timedelta(0)
    inst = pa.array([d - off for d in local], type=pa.timestamp("us")).cast(pa.timestamp("us", tz=tz))
    src = tmp_path / "in" / "fixed.parquet"
    src.parent.mkdir()
    pq.write_table(pa.table({"student_id": ids(120), "t": inst}), src)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert D.zone_rule(column_report(rep, "fixed.parquet", "t")["params"]) is None
    syn = pq.read_table(out / "fixed.parquet")["t"].to_pylist()
    assert {(d.hour, d.minute) for d in syn} == {(9, 45)}


def test_seeded_runs_are_byte_identical_for_zoned_timestamps(tmp_path: Path) -> None:
    local = daily(dt.datetime(2025, 1, 1, 9, 0), 150)
    src = tmp_path / "in" / "events.parquet"
    src.parent.mkdir()
    pq.write_table(pa.table({"student_id": ids(150), "event_time": zoned("America/Chicago", local)}), src)
    a = run(src.parent, tmp_path, name="a", seed="9")
    b = run(src.parent, tmp_path, name="b", seed="9")
    assert a[0].exit_code == b[0].exit_code == 0
    assert hashes(a[1]) == hashes(b[1])


# --------------------------------------------------------------------------- fidelity on what is written
def test_statistics_describe_the_written_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the old seam (offsets drawn independently of the date): the writer then stores other local
    times than the generated text; the report must warn and describe what is stored."""
    local = daily(dt.datetime(2025, 1, 1, 9, 0), 200)
    src = tmp_path / "in" / "events.parquet"
    src.parent.mkdir()
    pq.write_table(pa.table({"student_id": ids(200), "event_time": zoned("America/Chicago", local)}), src)
    monkeypatch.setattr(D, "zone_rule", lambda params: None)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert "written_values_differ" in message_codes(rep, "events.parquet")
    written = pq.read_table(out / "events.parquet")["event_time"].to_pylist()
    assert {d.hour for d in written} != {9}  # the seam: other local hours stored than generated
    stored = Counter(d.isoformat()[-6:] for d in written)
    col = column_report(rep, "events.parquet", "event_time")
    assert Counter(col["synthetic"]["tz_pmf"]) == stored


# --------------------------------------------------------------------------- zone helpers
def test_vectorised_local_offsets_match_python_fold0() -> None:
    rng = np.random.default_rng(5)
    base = int(days_from_civil(np.array([2024]), np.array([1]), np.array([1]))[0]) * 86_400 * US
    local = base + rng.integers(0, 2 * 366 * 86_400, 3000) * US
    # the skipped and repeated hours of both 2024 changes
    for day, hour in ((dt.date(2024, 3, 10), 2), (dt.date(2024, 11, 3), 1)):
        d0 = int(days_from_civil(np.array([day.year]), np.array([day.month]), np.array([day.day]))[0])
        local = np.append(local, [d0 * 86_400 * US + (hour * 3600 + m * 60) * US for m in (0, 15, 30, 59)])
    for zone in ("America/Chicago", "Europe/London", "Australia/Lord_Howe"):
        assert np.array_equal(zone_local_offsets(zone, local), zone_offsets_for_local(zone, local)), zone


def test_a_skipped_wall_time_off_the_grid_moves_to_an_existing_grid_time() -> None:
    zone = "America/Chicago"
    day_t = 86_400 * US
    d0 = int(days_from_civil(np.array([2025]), np.array([3]), np.array([9]))[0])
    two_am = d0 * day_t + 2 * 3600 * US  # skipped: 02:00 does not exist on 2025-03-09
    off = int(zone_local_offsets(zone, np.array([two_am]))[0])
    inst = np.array([two_am - off * 60 * US], dtype=np.int64)  # written as 03:00 CDT: off a 2-hour grid
    lo, hi = inst[0] - 30 * day_t, inst[0] + 30 * day_t
    moved = G.zone_settle(inst, np.array([True]), zone, 2 * 3600 * US, 0, lo, hi, 1, np.array([False]))
    t = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=int(moved[0]))
    loc = t.astimezone(ZoneInfo(zone))
    assert loc.minute == 0 and loc.hour % 2 == 0 and loc.date() == dt.date(2025, 3, 9)


@pytest.mark.parametrize("tokens, expect", [
    ({"-06:00": 5, "-05:00": 5}, ("-06:00", "-05:00")),
    ({"-0600": 5, "-0500": 5}, ("-0600", "-0500")),
    ({"CST": 5, "CDT": 5}, ("CST", "CDT")),
    ({"-06:00": 9}, ("-06:00", "-05:00")),  # an offset the source never wrote takes its style
    ({"CST": 9}, ("CST", "CDT")),
])
def test_rendered_offsets_use_the_source_token_style(tokens: dict[str, int], expect: tuple[str, str]) -> None:
    fmt = "%Y-%m-%d %H:%M %Z" if any(t.isalpha() for t in tokens) else "%Y-%m-%dT%H:%M:%S%z"
    pad = {"pad": {"H": True, "M": True, "S": True, "d": True, "m": True}}
    params = {"kind": "datetime", "formats": [{"format": fmt, "render": pad}], "tz_tokens": tokens,
              "zone_rule": "America/Chicago"}
    p = ColumnProfile(index=0, name="t", type=T.DATETIME_TZ, params=params)
    vals = G._empty(2)
    vals["fmt"] = 0
    vals["text"] = None
    for i, (m, d) in enumerate(((1, 15), (7, 15))):  # 09:00 local = 15:00 / 14:00 UTC
        vals["day"][i] = int(days_from_civil(np.array([2025]), np.array([m]), np.array([d]))[0])
        vals["us"][i] = (15 - i) * 3600 * US
    out = G.render_dates(p, vals).tolist()
    assert all(s.endswith(e) for s, e in zip(out, expect)), out
    assert all("09:00" in s for s in out), out


def test_declared_rule_values_are_rendered_with_the_zone_offset() -> None:
    """Any re-rendered value (declared rules, derived rows) goes through render_dates, which derives the
    offset from the zone at the value's instant."""
    pad = {"pad": {"H": True, "M": True, "S": True, "d": True, "m": True}}
    params = {"kind": "datetime", "formats": [{"format": "%Y-%m-%dT%H:%M:%S%z", "render": pad}],
              "tz_tokens": {"-06:00": 3}, "zone_rule": "America/Chicago"}
    p = ColumnProfile(index=0, name="t", type=T.DATETIME_TZ, params=params)
    vals = G._empty(1)
    vals["fmt"] = 0
    vals["text"] = None
    vals["tz"] = "-06:00"  # a stale token from the row the value came from
    vals["day"][0] = int(days_from_civil(np.array([2025]), np.array([7]), np.array([1]))[0])
    vals["us"][0] = 14 * 3600 * US
    assert G.render_dates(p, vals).tolist() == ["2025-07-01T09:00:00-05:00"]


# =========================================================================== R4-07: shapes inside text
@pytest.mark.parametrize("value, kind, span", [
    ("Contact Althea Voss at althea.voss@district.edu before class", "email", "althea.voss@district.edu"),
    ("Write to José.Núñez@colegio.mx today", "email", "José.Núñez@colegio.mx"),
    ("See https://www.district.edu/staff, then call", "url", "https://www.district.edu"),
    ("See www.district.edu.", "url", "www.district.edu"),
    ("Portal springfield.k12.il.us has the forms", "domain", "springfield.k12.il.us"),
    ("Laptop IP 10.20.30.40 was blocked", "ip", "10.20.30.40"),
    ("SSN on file 123-45-6789", "ssn", "123-45-6789"),
    ("call (312) 555-0187 after 3pm", "phone", "(312) 555-0187"),
    ("Call mom 3125550199 tonight", "phone", "3125550199"),
    ("Intl +44 20 7946 0958 in London", "phone", "+44 20 7946 0958"),
    ("Lives at 1234 Oak Street with her aunt", "address", "1234 Oak Street"),
])
def test_spans_are_found(value: str, kind: str, span: str) -> None:
    spans = embedded.pii_spans(value)
    assert [(value[s:e], k) for s, e, k in spans] == [(span, kind)]


@pytest.mark.parametrize("value", [
    "Scored 3.5 on the quiz.", "Ended.It was fine", "Grade 10 math in Room 204", "Bus 12 at 7:45 a.m.",
    "e.g. extra time on tests", "Mr. Smith and Mrs. Jones", "U.S. history", "uploaded report.pdf and notes.txt",
    "2024-25 school year", "version 1.2.3", "Student ID 12345", "attended 2024-03-15 10:30", "GPA 3.85/4.0",
    "Score 98.6 percent",
])
def test_ordinary_notes_hold_no_span(value: str) -> None:
    assert embedded.pii_spans(value, digit_phone=True) == []


def unsafe_spans(value: str) -> list[tuple[str, str]]:
    """Personal-data shapes in a generated value that break the PII generators' rules."""
    bad = []
    for s, e, kind in embedded.pii_spans(value, digit_phone=True, address=False):
        span = value[s:e]
        if kind == "email" and not is_reserved_domain(span.rpartition("@")[2]):
            bad.append((kind, span))
        elif kind in ("url", "domain"):
            host = re.sub(r"(?i)^(?:https?|ftp)://", "", span)
            host = re.split(r"[:/]", host[4:] if host.lower().startswith("www.") else host)[0]
            if not is_reserved_domain(host):
                bad.append((kind, span))
        elif kind == "ssn" and span[0] != "9":
            bad.append((kind, span))
        elif kind == "ip" and not span.startswith(("192.0.2.", "198.51.100.", "203.0.113.", "198.18.", "198.19.")):
            bad.append((kind, span))
    return bad


NOTES = [
    ("Contact Althea Voss at althea.voss@district.edu before the next class", 20),
    ("Email ysabel.quispe@gmail.com or call (312) 555-0187 after 3pm", 15),
    ("See https://www.district.edu/staff/voss for the schedule.", 12),
    ("Portal springfield.k12.il.us has the forms", 12),
    ("SSN on file 123-45-6789 needs checking", 11),
    ("Lives at 1234 Oak Street with her aunt", 11),
    ("Call mom 3125550199 tonight", 10),
    ("Laptop IP 10.20.30.40 was blocked", 10),
    ("Intl contact +44 20 7946 0958 in London", 10),
    ("Write to José.Núñez@colegio.mx today", 10),
]
REAL_SPANS = ("althea.voss@district.edu", "ysabel.quispe@gmail.com", "district.edu", "gmail.com",
              "springfield.k12.il.us", "123-45-6789", "555-0187", "3125550199", "10.20.30.40", "7946 0958",
              "colegio.mx", "Oak Street", "1234 Oak")


def test_review_fixture_embedded_email_uses_a_reserved_domain(tmp_path: Path) -> None:
    note = "Contact Althea Voss at althea.voss@district.edu before the next class"
    src = write_csv(tmp_path / "in" / "embedded_email.csv",
                    {"student_id": ids(60), "Notes": [note] * 60, "score": list(range(60))})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    syn = [r["Notes"] for r in read_csv(out / "embedded_email.csv")]
    emails = [m.group() for v in syn for m in embedded.EMAIL_SPAN_RE.finditer(v)]
    assert len(emails) == 60 and all(is_reserved_domain(e.rpartition("@")[2]) for e in emails)
    assert not any("district" in v or "althea" in v.casefold() for v in syn)


def test_free_text_spans_follow_their_pii_rules(tmp_path: Path) -> None:
    rows = [t for t, c in NOTES for _ in range(c)]
    rows += [f"Note {i}: reach parent{i}@school{i % 3}.org or 847-555-{1000 + i}" for i in range(30)]
    src = write_csv(tmp_path / "in" / "notes.csv", {"student_id": ids(len(rows)), "Notes": rows})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert file_report(rep, "notes.csv")["checks"]["leak_scan"]["result"] == "PASS"
    syn = [r["Notes"] for r in read_csv(out / "notes.csv")]
    assert len(syn) == len(rows)
    bad = [b for v in syn for b in unsafe_spans(v)]
    assert not bad, bad[:5]
    blob = "\n".join(syn)
    assert not [s for s in REAL_SPANS if s in blob]
    # one synthetic span per real span: the 20 copies of the first note carry one synthetic address
    first = Counter(v for v in syn).most_common(1)[0]
    assert first[1] == 20 and "@example.org" in first[0]
    # six real e-mail domains (district.edu, gmail.com, colegio.mx, school0/1/2.org) -> six reserved ones
    doms = {m.group().rpartition("@")[2].casefold() for v in syn for m in embedded.EMAIL_SPAN_RE.finditer(v)}
    assert len(doms) == 6 and all(is_reserved_domain(d) for d in doms)


def test_free_text_with_spans_is_seed_deterministic(tmp_path: Path) -> None:
    rows = [t for t, c in NOTES for _ in range(c)]
    src = write_csv(tmp_path / "in" / "notes.csv", {"student_id": ids(len(rows)), "Notes": rows})
    a = run(src.parent, tmp_path, name="a", seed="5")
    b = run(src.parent, tmp_path, name="b", seed="5")
    assert a[0].exit_code == b[0].exit_code == 0
    assert hashes(a[1]) == hashes(b[1])


def test_same_real_span_maps_to_one_synthetic_span() -> None:
    model = ColumnModel(nonmissing_counts={"Ask althea.voss@district.edu first": 3,
                                           "Then email althea.voss@district.edu again": 2,
                                           "Or ysabel@other.org": 4})
    p = ColumnProfile(index=0, name="Notes", type=T.FREE_TEXT, model=model)
    vals = text.lorem(p, model, 9, np.random.default_rng(1))
    emails = Counter(m.group() for v in vals for m in embedded.EMAIL_SPAN_RE.finditer(v))
    assert len(emails) == 2 and all(is_reserved_domain(e.rpartition("@")[2]) for e in emails)
    assert sorted(emails.values()) == [4, 5]
    assert {e.rpartition("@")[2] for e in emails} == {"example.org", "example.com"}  # ranked by rows


def test_a_reserved_real_span_is_never_reused() -> None:
    model = ColumnModel(nonmissing_counts={"Portal example.org for forms": 5, "Portal district.org for forms": 5})
    p = ColumnProfile(index=0, name="Notes", type=T.FREE_TEXT, model=model)
    synth = embedded.SpanSynth.build(p, list(model.nonmissing_counts.items()), np.random.default_rng(2))
    hosts = set(synth.mapping["url"].values())
    assert "example.org" not in hosts and all(is_reserved_domain(h) for h in hosts) and len(hosts) == 2


def test_free_text_without_spans_draws_no_extra_randomness() -> None:
    model = ColumnModel(nonmissing_counts={"Needs extra time on tests": 4, "Sits near the front": 3})
    p = ColumnProfile(index=0, name="Notes", type=T.FREE_TEXT, model=model)
    assert embedded.SpanSynth.build(p, list(model.nonmissing_counts.items()), np.random.default_rng(0)) is None
    a = text.lorem(p, model, 7, np.random.default_rng(11)).tolist()
    pyr = text.py_random(np.random.default_rng(11))
    word = lambda n, run: text.lorem_word(n, pyr)  # noqa: E731
    assert text._mimic_plain("Needs extra time on tests", pyr, word) in a


# --------------------------------------------------------------------------- the other generator paths
@pytest.mark.parametrize("template", [
    "ana.lopez@district.edu", "call 312-555-0199", "SSN 123-45-6789", "host 10.20.30.40", "https://district.edu/x",
    "see springfield.k12.il.us", "(773) 201-4455 ext 12",
])
def test_masks_and_mimics_never_rebuild_a_span(template: str) -> None:
    pyr = random.Random(4)
    word = lambda n, run: text.lorem_word(n, pyr)  # noqa: E731
    for _ in range(20):
        for v in (text.mask_fill(template, pyr), text.mimic_words(template, pyr, word)):
            assert not unsafe_spans(v), v
            assert not any(s in v for s in ("district", "123-45-6789", "10.20.30.40", "springfield", "201-4455"))


def test_random_notes_never_carry_an_unsafe_shape() -> None:
    """Seeded fuzz over the generator paths: notes mixing words with every kind of span."""
    r = random.Random(7)
    pieces = ["ana{}@district.edu", "Jo.{}@mail.school.org", "https://site{}.org/p", "www.k{}.net",
              "portal{}.k12.il.us", "(312) 555-01{:02d}", "312.555.01{:02d}", "+1 312 555 01{:02d}",
              "call 77355501{:02d} now", "+44 20 7946 09{:02d}", "123-45-67{:02d}", "10.0.0.{}"]
    words = "the student needs extra time see call email after lunch at".split()
    notes = {}
    for i in range(150):
        k = [r.choice(words) for _ in range(r.randint(2, 6))]
        k.insert(r.randint(0, len(k)), r.choice(pieces).format(i % 90 + 10))
        notes[" ".join(k)] = r.randint(1, 3)
    model = ColumnModel(nonmissing_counts=notes)
    p = ColumnProfile(index=0, name="Notes", type=T.FREE_TEXT, model=model)
    out = text.lorem(p, model, sum(notes.values()), np.random.default_rng(2)).tolist()
    assert len(out) == sum(notes.values())
    pyr = random.Random(3)
    masks = [text.mask_fill(n, pyr) for n in notes]
    mimics = [text.mimic_words(n, pyr, lambda n_, run: text.lorem_word(n_, pyr)) for n in notes]
    for v in out + masks + mimics:
        assert not unsafe_spans(v), v
    real = {n[s:e].casefold() for n in notes for s, e, _ in embedded.pii_spans(n, True)}
    assert not {v[s:e].casefold() for v in out for s, e, _ in embedded.pii_spans(v, True)} & real


def test_digit_only_phone_numbers_follow_the_phone_rules() -> None:
    pyr = random.Random(1)
    for v in ("3125550187", "13125550187", "3125550187.0"):
        out = text.digit_mask(v, pyr)
        assert len(out) == len(v) and not unsafe_spans(out) and out != v
        assert re.sub(r"\D", "", out.split(".")[0])[-7] in "23456789"
    assert text.digit_mask("-12.50", pyr)[0] == "-"  # ordinary numbers stay numbers


def _pseudo_profile(name: str, t: T, counts: dict[str, int], units: list[str], **params: Any) -> ColumnProfile:
    """A profile whose ``units`` the D35 allowlist did not keep, with its pseudonym map built (D35)."""
    from datanonymous.detect.text import PSEUDO_KEY

    model = ColumnModel(nonmissing_counts=dict(counts))
    params[PSEUDO_KEY] = [[u, f"[pseudonymized {i + 1}]"] for i, u in enumerate(units)]
    p = ColumnProfile(index=0, name=name, type=t, model=model, params=params)

    class _T:
        columns = [p]

        class private:  # noqa: N801 - a stand-in for TablePrivate
            extra: dict[str, Any] = {}
    pseudonyms.build_maps(_T, lambda prof: np.random.default_rng(3))
    return p


def test_multi_select_items_with_embedded_shapes() -> None:
    # D35: items with personal-data shapes are never kept; their pseudonyms follow the PII rules
    counts = {"Mom: ana@x.edu;Dad (312) 555-0199": 20, "Bus;Walker": 20, "Grandma 847-555-0101;Bus": 20,
              "Aunt: aunt.k@mail.com;Walker": 20}
    p = _pseudo_profile("Pickup", T.MULTI_SELECT, counts,
                        ["Mom: ana@x.edu", "Dad (312) 555-0199", "Grandma 847-555-0101", "Aunt: aunt.k@mail.com"],
                        delimiter=";")
    m2 = pseudonyms.generation_model(p, p.model)
    vals = list(m2.nonmissing_counts)
    assert not [b for v in vals for b in unsafe_spans(v)]
    blob = "\n".join(vals)
    assert not any(s in blob for s in ("ana@x.edu", "aunt.k@mail.com", "(312) 555-0199", "847-555-0101"))
    assert sum(m2.nonmissing_counts.values()) == 80


def test_code_masks_of_copy_gated_values() -> None:
    model = ColumnModel(nonmissing_counts={"L-101": 20, "L-102": 20, "jo7@x.io": 20, "ab.1@x.io": 20})
    p = ColumnProfile(index=0, name="Locker", type=T.HIGH_CARD_CATEGORICAL, subtype="code", model=model)
    vals, _ = text.invented_labels_flags(p, model, 80, np.random.default_rng(1))
    emails = [v for v in vals.tolist() if "@" in v]
    assert len(emails) == 40 and all(is_reserved_domain(v.rpartition("@")[2]) for v in emails)
    assert {v for v in vals.tolist() if "@" not in v} and all(re.fullmatch(r"[A-Z]-\d{3}", v)
                                                               for v in vals.tolist() if "@" not in v)


def test_numeric_tokens_holding_a_phone() -> None:
    # D35: a token among numbers holding a phone number is never kept; its pseudonym follows the phone rules
    p = _pseudo_profile("Score", T.NUMERIC_WITH_TOKENS,
                        {"12": 5, "call 312-555-0199": 2, "*": 3, " call 312-555-0199": 1}, ["call 312-555-0199"],
                        tokens=["*", "call 312-555-0199"], format={"decimal_mark": "."})
    out = [pseudonyms.mapped_value(p, v) for v in ("12", "call 312-555-0199", "*", " call 312-555-0199")]
    assert out[0] == "12" and out[2] == "*"
    assert "312-555-0199" not in out[1] and not unsafe_spans(out[1]) and out[3] == " " + out[1]


def test_name_generator_with_an_email_inside_is_guarded() -> None:
    model = ColumnModel(nonmissing_counts={"Ana Lopez <ana.lopez@district.edu>": 10, "Luis Garcia": 10})
    p = ColumnProfile(index=0, name="Parent_Name", type=T.PERSON_NAME, subtype="full", model=model)
    vals, _ = generate_pii(p, model, 20, np.random.default_rng(1), lambda: None)
    assert not [b for v in vals.tolist() for b in unsafe_spans(v)]
    assert not any("district" in v for v in vals.tolist())


def test_drawer_guards_every_candidate_and_fallback() -> None:
    pyr = random.Random(0)
    d = text.Drawer({"x"})
    got = d.draw(lambda a: "reach lydia.cross@mollitia.sit or 847-293-1182", pyr)
    assert not unsafe_spans(got) and "@example.org" in got and "847-293-1182" in got
    # every make() candidate is taken: the fallback masks of the last candidate are guarded too
    real = {"a@example.org"}
    d2 = text.Drawer(real)
    got2 = d2.draw(lambda a: "a@example.org", pyr, attempts=3)
    assert got2.casefold() not in real and not unsafe_spans(got2) and got2.endswith("@example.org")
    # the lengthen fallback terminates and never extends a reserved domain into an unreserved one
    d3 = text.Drawer(set(), fallback=lambda base, r: base)
    d3.used.add("z@example.org")
    got3 = d3.draw(lambda a: "z@example.org", pyr, attempts=2)
    assert got3 != "z@example.org" and not unsafe_spans(got3)


def test_guard_leaves_safe_values_alone() -> None:
    pyr = random.Random(0)
    for v in ("lydia.cross@example.org", "https://www.example.com/a", "(312) 555-0101", "922-05-4488",
              "192.0.2.7", "Needs extra time", "Mary Smith", "+44 20 7946 0958", "AB-1234"):
        assert embedded.guard(v, pyr) is v


def test_d33_prose_with_an_email(tmp_path: Path) -> None:
    vals = ["Please send the forms to the office before the trip",
            "Email the nurse at nurse.jones@district.edu after the second class"] * 30
    src = write_csv(tmp_path / "in" / "reasons.csv", {"student_id": ids(60), "Reason": vals})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    syn = [r["Reason"] for r in read_csv(out / "reasons.csv")]
    assert not [b for v in syn for b in unsafe_spans(v)] and not any("district" in v for v in syn)
    assert sum("@example.org" in v for v in syn) == 30


def test_contact_numbers_under_a_contact_header(tmp_path: Path) -> None:
    vals = ["3125550187", "7735550163", "8475550182"] * 20
    src = write_csv(tmp_path / "in" / "contacts.csv", {"student_id": ids(60), "Emergency_Contact": vals})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    syn = [r["Emergency_Contact"] for r in read_csv(out / "contacts.csv")]
    assert not set(syn) & set(vals)
    assert all(len(v) == 10 and v[3] in "23456789" for v in syn)
