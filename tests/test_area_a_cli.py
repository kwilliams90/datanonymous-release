"""AREA A end to end: the CLI on tests/fixtures/dates/dates.csv (types, formats, fidelity, privacy, flags)."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datanonymous.detect import dates as D

from conftest import run_cli
from test_area_a_support import FIXTURES

SRC = FIXTURES / "dates.csv"
EXPECTED = {
    "student_id": "IDENTIFIER", "iso_date": "DATE", "us_padded": "DATE", "us_unpadded": "DATE", "eu_date": "DATE",
    "ambiguous_date": "DATE", "mixed_date": "DATE_MIXED_FORMATS", "dob": "DATE", "sas_date": "DATE",
    "oracle_date": "DATE", "month_name_date": "DATE", "ampm_ts": "DATETIME", "tz_ts": "DATETIME_TZ",
    "created_at": "DATETIME_TZ", "time_in": "TIME", "time_ampm": "TIME", "duration": "DURATION",
    "session_iso": "DURATION", "enroll_yyyymmdd": "DATE_AS_INT", "entry_date_serial": "DATE_AS_INT",
    "updated_epoch": "DATE_AS_INT", "year_month": "YEAR_MONTH", "attendance_date": "DATE", "exit_date": "DATE",
    "sy_long": "SCHOOL_YEAR", "sy_short": "SCHOOL_YEAR", "sy_yy": "SCHOOL_YEAR", "sy_code": "SCHOOL_YEAR",
    "sy_spaced": "SCHOOL_YEAR", "sy_concat": "SCHOOL_YEAR",
}
FORMATS = {
    "iso_date": "%Y-%m-%d", "us_padded": "%m/%d/%Y", "us_unpadded": "%m/%d/%Y", "eu_date": "%d/%m/%Y",
    "dob": "%m/%d/%y", "sas_date": "%d%b%Y", "oracle_date": "%d-%b-%y", "month_name_date": "%b %d, %Y",
    "ampm_ts": "%m/%d/%Y %I:%M %p", "tz_ts": "%Y-%m-%dT%H:%M:%S%z", "created_at": "%Y-%m-%dT%H:%M:%S.%f%z",
    "time_in": "%H:%M", "time_ampm": "%I:%M %p", "duration": "@dur_hms", "session_iso": "@dur_iso",
    "enroll_yyyymmdd": "%Y%m%d", "entry_date_serial": "@excel_serial", "updated_epoch": "@epoch_s",
    "year_month": "%Y-%m",
}


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def dates_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("area_a")
    out, rep = base / "out", base / "rep"
    res = run_cli(SRC, "--out-dir", out, "--report-dir", rep, "--seed", "5")
    assert res.exit_code == 0, res.output
    report = json.loads((rep / "files" / "dates.csv.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in report["tables"][0]["columns"]}
    return SimpleNamespace(out=out, rep=rep, report=report, cols=cols, real=rows(SRC), syn=rows(out / "dates.csv"))


def test_fixture_is_reproducible() -> None:
    spec = importlib.util.spec_from_file_location("area_a_make_fixtures", FIXTURES / "make_fixtures.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build().encode("utf-8") == SRC.read_bytes()
    assert SRC.stat().st_size < 300_000


def test_detected_types_and_formats(dates_run: SimpleNamespace) -> None:
    assert {k: c["type"] for k, c in dates_run.cols.items()} == EXPECTED
    for name, fmt in FORMATS.items():
        assert [f["format"] for f in dates_run.cols[name]["params"]["formats"]] == [fmt], name
    assert dates_run.cols["ambiguous_date"]["ambiguous"]
    assert dates_run.cols["dob"]["role"] == "entity"
    assert dates_run.cols["entry_date_serial"]["subtype"] == "excel_serial"
    assert dates_run.cols["attendance_date"]["params"]["calendar"] is True
    assert dates_run.cols["exit_date"]["params"]["sentinels"] == {"0/0/0": 13, "1/1/1900": 10, "12/31/9999": 8}


def test_same_shape_output(dates_run: SimpleNamespace) -> None:
    assert len(dates_run.syn) == len(dates_run.real)
    assert list(dates_run.syn[0]) == list(dates_run.real[0])
    assert SRC.read_bytes().split(b"\n", 1)[0] == (dates_run.out / "dates.csv").read_bytes().split(b"\n", 1)[0]


def test_every_synthetic_value_reparses_with_the_public_params(dates_run: SimpleNamespace) -> None:
    for name, col in dates_run.cols.items():
        if col["type"] not in {t.value for t in D.DATE_TYPES}:
            continue
        na = set(col["real"]["missing_tokens"])
        vals = [r[name].strip() for r in dates_run.syn if r[name] not in na and r[name].strip()]
        v = D.view(col["params"], vals)
        bad = [x for x, c in zip(vals, v.cls) if c == D.UNPARSED]
        assert not bad, (name, bad[:3])


def test_distributions(dates_run: SimpleNamespace) -> None:
    real, syn = dates_run.real, dates_run.syn
    for name, col in dates_run.cols.items():
        if col["type"] in {t.value for t in D.DATE_TYPES}:
            assert col["fidelity"]["verdict"] == "ok", (name, col["fidelity"])
            assert Counter(r[name] for r in real if r[name] in ("", "NA")) == Counter(
                r[name] for r in syn if r[name] in ("", "NA")), name
    # sentinels written verbatim at their exact rates
    for tok in ("1/1/1900", "12/31/9999", "0/0/0"):
        assert sum(r["exit_date"] == tok for r in syn) == sum(r["exit_date"] == tok for r in real)
    # calendar-like attendance: observed dates only; sparse exit dates: weekdays only (snapped)
    assert {r["attendance_date"] for r in syn} <= {r["attendance_date"] for r in real}
    import datetime as dt

    wd = {dt.datetime.strptime(r["exit_date"], "%m/%d/%Y").weekday() for r in syn
          if r["exit_date"] not in ("", "1/1/1900", "12/31/9999", "0/0/0")}
    assert wd <= {0, 1, 2, 3, 4}
    # DST offsets follow the source's zone (America/Chicago, inferred from the offsets; R4-06): every value's
    # offset is the zone's at its instant, so the offset shares follow the generated dates (close to the
    # source's); mixed formats at their shares
    from zoneinfo import ZoneInfo

    assert dates_run.cols["tz_ts"]["params"]["zone_rule"] == "America/Chicago"
    chicago = ZoneInfo("America/Chicago")
    for r in syn:
        t = dt.datetime.fromisoformat(r["tz_ts"])
        assert t.utcoffset() == t.astimezone(chicago).utcoffset(), r["tz_ts"]
    syn_off, real_off = Counter(r["tz_ts"][-6:] for r in syn), Counter(r["tz_ts"][-6:] for r in real)
    assert set(syn_off) == set(real_off)
    assert all(abs(syn_off[k] - real_off[k]) <= 0.02 * len(real) for k in real_off)
    assert sum("-" in r["mixed_date"] for r in syn) == sum("-" in r["mixed_date"] for r in real)
    # durations keep hours >= 24; two-digit DOB years never wrap through the pivot
    assert any(int(r["duration"].split(":")[0]) >= 24 for r in syn if r["duration"] not in ("", "NA"))
    assert not {int(r["dob"][-2:]) for r in syn} & set(range(13, 58))
    # D14: the exact extremes appear
    for name in ("iso_date", "enroll_yyyymmdd", "duration"):
        st_real, st_syn = dates_run.cols[name]["real"], dates_run.cols[name]["synthetic"]
        assert (st_real["min"], st_real["max"]) == (st_syn["min"], st_syn["max"]), name


def test_privacy_no_real_ids_or_raw_dates_in_outputs(dates_run: SimpleNamespace) -> None:
    real_ids = {r["student_id"] for r in dates_run.real}
    assert not real_ids & {r["student_id"] for r in dates_run.syn}
    raw_dobs = {r["dob"] for r in dates_run.real}
    docs = [p for p in dates_run.out.iterdir() if p.name != "dates.csv"]
    report_files = [p for p in dates_run.rep.rglob("*") if p.is_file()]
    assert docs and report_files
    for path in docs + report_files:
        text = path.read_text(encoding="utf-8")
        assert not any(i in text for i in real_ids), path.name
        assert not any(d in text for d in raw_dobs), path.name  # aggregates only, never raw values


def test_dictionary_fields(dates_run: SimpleNamespace) -> None:
    schema = json.loads((dates_run.out / "schema.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in schema["files"][0]["tables"][0]["columns"]}
    readr = {k: c["readr"] for k, c in cols.items()}
    assert readr["iso_date"] == "D" and readr["tz_ts"] == "T" and readr["created_at"] == "T"
    assert readr["time_in"] == "t" and readr["enroll_yyyymmdd"] == "i" and readr["entry_date_serial"] == "i"
    assert readr["updated_epoch"] == "d"
    for name in ("us_padded", "eu_date", "sas_date", "ampm_ts", "exit_date", "year_month", "duration"):
        assert readr[name] == "c", name
    # integration: schema.json is types and formats only (AREA G, D20/D21), so the R parsing note is published
    # as 'parse' (AREA B's key for the same instruction) and mixed formats without their row shares; the shares
    # (synthetic) are in the data dictionary
    assert "parse_date_time" in cols["sas_date"]["parse"] and cols["sas_date"]["format"] == "%d%b%Y"
    assert cols["mixed_date"]["formats"] == ["%Y-%m-%d", "%m/%d/%Y"]
    dd = (dates_run.out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    mixed = next(line for line in dd.splitlines() if "`mixed_date`" in line)
    assert re.search(r"formats `%Y-%m-%d` \(5[45]%\) / `%m/%d/%Y` \(4[45]%\)", mixed), mixed
    assert cols["iso_date"]["r_class"] == "Date" and cols["tz_ts"]["r_class"] == "POSIXct"
    assert cols["time_in"]["r_class"] == "hms" and cols["duration"]["r_class"] == "difftime"


def test_deterministic_output(dates_run: SimpleNamespace, tmp_path: Path) -> None:
    res = run_cli(SRC, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "5")
    assert res.exit_code == 0
    assert (tmp_path / "o" / "dates.csv").read_bytes() == (dates_run.out / "dates.csv").read_bytes()


# --------------------------------------------------------------------------- flags on a small file
def _write(path: Path, header: str, lines: list[str]) -> Path:
    """Write the input under <tmp>/in/ (the out dir may not sit inside the input's folder)."""
    path = path.parent / "in" / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


def test_flags_end_to_end(tmp_path: Path) -> None:
    import datetime as dt

    rng = np.random.default_rng(3)
    lines = []
    for i in range(500):  # ~290 distinct dates: two leftovers stay under 1% of the distinct values
        d = dt.date(2024, 1, 1) + dt.timedelta(days=int(rng.integers(0, 400)))
        amb = f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 13)):02d}/2024"
        note = "TBD" if i < 3 else ("sent home early after a long conversation with the family" if i == 3
                                    else d.strftime("%m/%d/%Y"))
        lines.append(f"{amb},{d.isoformat()},{(d - dt.date(1899, 12, 30)).days},{note}")
    src = _write(tmp_path / "flags.csv", "amb,visit,amount,withdraw_date", lines)
    out = tmp_path / "out"
    res = run_cli(src, "--out-dir", out, "--seed", "2", "--dayfirst", "amb", "--calendar", "visit",
                  "--date-cols", "amount", "--no-exact-extremes", "visit")
    assert res.exit_code == 0, res.output
    rep = json.loads((tmp_path / "out-report" / "files" / "flags.csv.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in rep["tables"][0]["columns"]}
    assert cols["amb"]["params"]["formats"][0]["format"] == "%d/%m/%Y" and not cols["amb"]["ambiguous"]
    assert cols["visit"]["params"]["calendar_reason"] == "--calendar"
    assert cols["amount"]["type"] == "DATE_AS_INT" and cols["amount"]["subtype"] == "excel_serial"
    assert cols["withdraw_date"]["params"]["other_tokens"] == 3
    syn = rows(out / "flags.csv")
    real = rows(src)
    assert {r["visit"] for r in syn} <= {r["visit"] for r in real}
    assert sum(r["withdraw_date"] == "TBD" for r in syn) == 3
    secret = "sent home early after a long conversation with the family"
    for path in list(out.rglob("*")) + list((tmp_path / "out-report").rglob("*")):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8"), path.name
    # inspect shows the date types
    res = run_cli("inspect", src, "--json", "--date-cols", "amount")
    assert res.exit_code == 0
    types = {c["name"]: c["type"] for c in json.loads(res.output)["files"][0]["tables"][0]["columns"]}
    assert types == {"amb": "DATE", "visit": "DATE", "amount": "DATE_AS_INT", "withdraw_date": "DATE"}


def test_date_format_mismatch_fails_the_file_with_a_fix(tmp_path: Path) -> None:
    src = _write(tmp_path / "f.csv", "d", [f"2024-01-{d:02d}" for d in range(1, 29)])
    res = run_cli(src, "--out-dir", tmp_path / "o", "--seed", "1", "--date-format", "d=%m/%d/%Y")
    assert res.exit_code == 3
    assert not (tmp_path / "o" / "f.csv").exists()
    data = json.loads((tmp_path / "o-report" / "report.json").read_text(encoding="utf-8"))
    assert data["failures"] and "--date-format" in json.dumps(data["failures"])
    ok = run_cli(src, "--out-dir", tmp_path / "o2", "--seed", "1", "--date-format", "d=%Y-%m-%d")
    assert ok.exit_code == 0
    bad = run_cli(src, "--out-dir", tmp_path / "o3", "--seed", "1", "--date-format", "d=%Q")
    assert bad.exit_code == 2
