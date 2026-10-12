"""Build tests/fixtures/dates/dates.csv deterministically (all values fabricated).

Run from anywhere: ``python tests/fixtures/dates/make_fixtures.py``. The output is
committed; the AREA A tests check that re-running this script reproduces it
byte for byte.

Columns (design section 8 ``dates.csv`` plus an attendance-like calendar column):
student_id (ID, 1-5 rows each), iso_date, us_padded, us_unpadded, eu_date
(day-first, some days > 12), ambiguous_date (every day <= 12), mixed_date (55%
ISO / 45% US), dob (``%m/%d/%y``, 1958-2012, constant per student), sas_date
(``15MAR2024``), oracle_date (``14-AUG-24``), month_name_date (``Mar 5, 2024``),
ampm_ts (``03/15/2024 01:45 PM``), tz_ts (ISO with America/Chicago DST offsets),
created_at (ISO, milliseconds, ``Z``), time_in (24 h, 5-minute grid),
time_ampm (``7:45 AM``), duration (``h:mm:ss`` with hours >= 24), session_iso
(``PT1H30M``), enroll_yyyymmdd (integers), entry_date_serial (Excel serials,
date header hint), updated_epoch (Unix seconds, header hint), year_month
(``2024-03``), attendance_date (school days, >= 10 rows per date), exit_date
(sparse school days with 1/1/1900, 12/31/9999 and 0/0/0 sentinels and blanks),
and the six school-year forms (FOUNDATION SCHOOL_YEAR, never dates).
"""

from __future__ import annotations

import datetime as dt
import zoneinfo
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 500
SEED = 20241004


def school_days(start: dt.date, end: dt.date, holidays: set[dt.date]) -> list[dt.date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def build() -> str:
    rng = np.random.default_rng(SEED)
    chicago = zoneinfo.ZoneInfo("America/Chicago")
    # students: 1-5 rows each, DOB constant per student
    ids: list[str] = []
    dobs: list[str] = []
    sid = 104000
    while len(ids) < N:
        sid += int(rng.integers(1, 30))
        k = int(rng.integers(1, 6))
        y = int(rng.integers(1958, 2013))
        dob = dt.date(y, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365)))
        for _ in range(k):
            ids.append(f"S{sid:06d}")
            dobs.append(f"{dob.month:02d}/{dob.day:02d}/{dob.year % 100:02d}")
    ids, dobs = ids[:N], dobs[:N]

    base = dt.date(2023, 8, 14)
    days = [base + dt.timedelta(days=int(x)) for x in rng.integers(0, 290, N)]
    hol = {dt.date(2023, 9, 4), dt.date(2023, 10, 9), dt.date(2023, 11, 23), dt.date(2023, 11, 24),
           dt.date(2024, 1, 15), dt.date(2024, 2, 19)}
    sch = school_days(dt.date(2023, 8, 14), dt.date(2024, 5, 24), hol)
    cal = [sch[int(i)] for i in rng.choice(len(sch), 40, replace=False)]
    cal_w = rng.random(40) + 0.3
    attendance = [cal[int(i)] for i in rng.choice(40, N, p=cal_w / cal_w.sum())]

    rows: dict[str, list[str]] = {"student_id": ids}
    rows["iso_date"] = [d.isoformat() for d in days]
    rows["us_padded"] = [d.strftime("%m/%d/%Y") for d in days]
    rows["us_unpadded"] = [f"{d.month}/{d.day}/{d.year}" for d in days]
    rows["eu_date"] = [f"{d.day:02d}/{d.month:02d}/{d.year}" for d in days]
    amb = [dt.date(2024, int(m), int(dd)) for m, dd in zip(rng.integers(1, 13, N), rng.integers(1, 13, N))]
    rows["ambiguous_date"] = [d.strftime("%m/%d/%Y") for d in amb]
    rows["mixed_date"] = [d.isoformat() if rng.random() < 0.55 else d.strftime("%m/%d/%Y") for d in days]
    rows["dob"] = dobs
    rows["sas_date"] = [d.strftime("%d%b%Y").upper() for d in days]
    rows["oracle_date"] = [f"{d.day:02d}-{d.strftime('%b').upper()}-{d.year % 100:02d}" for d in days]
    rows["month_name_date"] = [f"{d.strftime('%b')} {d.day}, {d.year}" for d in days]
    ampm = []
    tz = []
    created = []
    for d in days:
        sec = int(rng.integers(7 * 3600, 17 * 3600))
        t = dt.datetime.combine(d, dt.time()) + dt.timedelta(seconds=sec - sec % 60)
        h12 = t.hour % 12 or 12
        ampm.append(f"{t.month:02d}/{t.day:02d}/{t.year} {h12:02d}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}")
        loc = (dt.datetime.combine(d, dt.time()) + dt.timedelta(minutes=15 * int(rng.integers(24, 80)))
               ).replace(tzinfo=chicago)
        off = loc.utcoffset()
        assert off is not None
        mins = int(off.total_seconds() // 60)
        tz.append(loc.strftime("%Y-%m-%dT%H:%M:%S") + f"{'-' if mins < 0 else '+'}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}")
        c = dt.datetime.combine(d, dt.time()) + dt.timedelta(milliseconds=int(rng.integers(0, 86_400_000)))
        created.append(c.strftime("%Y-%m-%dT%H:%M:%S") + f".{c.microsecond // 1000:03d}Z")
    rows["ampm_ts"] = ampm
    rows["tz_ts"] = tz
    rows["created_at"] = created
    rows["time_in"] = [f"{int(h):02d}:{int(m) * 5:02d}" for h, m in zip(rng.integers(7, 10, N), rng.integers(0, 12, N))]
    rows["time_ampm"] = [f"{int(h)}:{int(m):02d} {'AM' if h < 12 else 'PM'}"
                         for h, m in zip(rng.choice([7, 8, 9, 10, 11], N), rng.integers(0, 60, N))]
    rows["duration"] = [f"{int(h)}:{int(m):02d}:{int(s):02d}" for h, m, s in
                        zip(rng.integers(0, 40, N), rng.integers(0, 60, N), rng.integers(0, 60, N))]
    rows["session_iso"] = [f"PT{int(h)}H{int(m)}M" if h else f"PT{int(m)}M" for h, m in
                           zip(rng.integers(0, 4, N), rng.integers(1, 60, N))]
    rows["enroll_yyyymmdd"] = [d.strftime("%Y%m%d") for d in days]
    rows["entry_date_serial"] = [str((d - dt.date(1899, 12, 30)).days) for d in days]
    rows["updated_epoch"] = [str(int(dt.datetime.combine(d, dt.time(), dt.timezone.utc).timestamp())
                                 + int(rng.integers(0, 86400))) for d in days]
    rows["year_month"] = [f"{d.year}-{d.month:02d}" for d in days]
    rows["attendance_date"] = [d.strftime("%m/%d/%Y") for d in attendance]
    exit_ = []
    sparse = [sch[int(i)] for i in rng.choice(len(sch), 120, replace=False)]
    for _ in range(N):
        u = rng.random()
        if u < 0.40:
            exit_.append("")
        elif u < 0.43:
            exit_.append("1/1/1900")
        elif u < 0.45:
            exit_.append("12/31/9999")
        elif u < 0.47:
            exit_.append("0/0/0")
        else:
            d = sparse[int(rng.integers(0, len(sparse)))]
            exit_.append(f"{d.month}/{d.day}/{d.year}")
    rows["exit_date"] = exit_
    sy = rng.integers(2019, 2025, N)
    rows["sy_long"] = [f"{y}-{y + 1}" for y in sy]
    rows["sy_short"] = [f"{y}-{(y + 1) % 100:02d}" for y in sy]
    rows["sy_yy"] = [f"{y % 100:02d}-{(y + 1) % 100:02d}" for y in sy]
    rows["sy_code"] = [f"SY{y % 100:02d}{(y + 1) % 100:02d}" for y in sy]
    rows["sy_spaced"] = [f"SY {y}-{(y + 1) % 100:02d}" for y in sy]
    rows["sy_concat"] = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in sy]
    # a few NA tokens / blanks in typed columns
    for col in ("iso_date", "us_padded", "time_in", "duration"):
        for i in rng.choice(N, 12, replace=False):
            rows[col][int(i)] = "NA" if rng.random() < 0.5 else ""
    header = list(rows)
    lines = [",".join(header)]
    for i in range(N):
        cells = []
        for h in header:
            v = rows[h][i]
            cells.append(f'"{v}"' if "," in v else v)
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def main() -> None:
    (HERE / "dates.csv").write_bytes(build().encode("utf-8"))


if __name__ == "__main__":
    main()
