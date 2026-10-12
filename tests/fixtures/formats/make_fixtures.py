"""Build the AREA F format fixtures deterministically (run from anywhere; writes next to this file).

    python tests/fixtures/formats/make_fixtures.py

Files (all small, committed):

* ``types.parquet``  Arrow-native types: decimal(9,2), ordered dictionary with an unused
  level, unordered dictionary, timestamp[us, America/Chicago] across DST, date32,
  time64[us], int32 with nulls (field metadata), bool with nulls, float64, strings
  (IDs and first names), schema metadata; zstd codec with one gzip column; 50-row
  row groups.
* ``nested.parquet`` list / struct / binary columns for the unsupported path, plus an
  all-null list column (readable) and a plain column.
* ``types.feather``  the types table as Feather V2 (zstd, 40-row record batches).
* ``labels.dta``     Stata 118: value labels (fully labelled ints), variable labels,
  data label, ``%td`` dates, strings, doubles and ints with missing values.
* ``records.json``   top-level array of flat records (indent=2, ensure_ascii): ints,
  floats, bools, nulls, missing keys, non-ASCII names.
* ``lines.jsonl``    JSON lines, compact separators, ensure_ascii=False.
* ``students.csv.gz`` / ``students.csv.zip``  the same small CSV, gzip (stored name,
  fixed mtime) and single-member zip (fixed date/time).
"""

from __future__ import annotations

import datetime as dt
import decimal
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
N = 120
FIRST = ["Ana", "Ben", "Chloe", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kofi", "Lena",
         "Mateo", "Nadia", "Oskar", "Priya", "Quinn", "Rosa", "Sven", "Tariq", "Uma", "Vera", "Wen", "Ximena",
         "Yusuf", "Zoe", "Amelie", "Bruno", "Celine", "Dario"]


def _rng() -> np.random.Generator:
    return np.random.default_rng(20261004)


def types_table() -> pa.Table:
    r = _rng()
    ids = [f"S{100000 + 7 * i:06d}" for i in range(N)]
    names = [FIRST[int(k)] for k in r.integers(0, len(FIRST), N)]
    grade_levels = ["PK", "KG", "01", "02", "03", "04", "05", "06"]  # "06" never used
    grade_idx = r.integers(0, 7, N).astype(np.int8)
    school_levels = ["Adams MS", "Lincoln ES", "Unused HS"]
    school_idx = r.integers(0, 2, N).astype(np.int8)
    bal = [None if k % 9 == 0 else decimal.Decimal(int(r.integers(-5000, 250000))).scaleb(-2) for k in range(N)]
    base = int(dt.datetime(2023, 8, 1, 12, tzinfo=dt.timezone.utc).timestamp())
    ts = [None if k % 11 == 0 else (base + int(r.integers(0, 200 * 86400))) * 10**6 + int(r.integers(0, 10**6))
          for k in range(N)]
    dob_base = (dt.date(2010, 1, 1) - dt.date(1970, 1, 1)).days
    dob = [None if k % 13 == 0 else dob_base + int(r.integers(0, 6 * 365)) for k in range(N)]
    start = [(7 * 3600 + int(r.integers(0, 4 * 3600))) * 10**6 for _ in range(N)]
    absences = [None if k % 7 == 0 else int(r.integers(0, 30)) for k in range(N)]
    ecodis = [None if k % 10 == 0 else bool(r.integers(0, 2)) for k in range(N)]
    gpa = np.round(r.uniform(1.0, 4.0, N), 2)
    fields = [
        pa.field("student_id", pa.string()),
        pa.field("first_name", pa.string()),
        pa.field("grade", pa.dictionary(pa.int8(), pa.string(), ordered=True)),
        pa.field("school", pa.dictionary(pa.int8(), pa.string())),
        pa.field("balance", pa.decimal128(9, 2)),
        pa.field("enrolled_at", pa.timestamp("us", tz="America/Chicago")),
        pa.field("dob", pa.date32()),
        pa.field("start_time", pa.time64("us")),
        pa.field("absences", pa.int32(), metadata={"description": "days absent this term"}),
        pa.field("ecodis", pa.bool_()),
        pa.field("gpa", pa.float64(), nullable=False),
    ]
    arrays = [
        pa.array(ids, pa.string()),
        pa.array(names, pa.string()),
        pa.DictionaryArray.from_arrays(pa.array(grade_idx, pa.int8()), pa.array(grade_levels), ordered=True),
        pa.DictionaryArray.from_arrays(pa.array(school_idx, pa.int8()), pa.array(school_levels)),
        pa.array(bal, pa.decimal128(9, 2)),
        pa.array(ts, pa.int64()).cast(pa.timestamp("us", tz="America/Chicago")),
        pa.array(dob, pa.int32()).cast(pa.date32()),
        pa.array(start, pa.int64()).cast(pa.time64("us")),
        pa.array(absences, pa.int32()),
        pa.array(ecodis, pa.bool_()),
        pa.array(gpa, pa.float64()),
    ]
    schema = pa.schema(fields, metadata={"source": "datanonymous area F fixture", "version": "1"})
    return pa.Table.from_arrays(arrays, schema=schema)


def nested_table() -> pa.Table:
    n = 12
    return pa.table({
        "record_id": pa.array([f"R{i:03d}" for i in range(n)]),
        "score": pa.array([float(i % 5) for i in range(n)]),
        "tags": pa.array([["a", "b"] if i % 3 else [] for i in range(n)], pa.list_(pa.string())),
        "address": pa.array([{"city": f"Town{i % 4}", "zip": f"7{i:04d}"} for i in range(n)],
                            pa.struct([("city", pa.string()), ("zip", pa.string())])),
        "photo": pa.array([bytes([i, i + 1]) for i in range(n)], pa.binary()),
        "unused_list": pa.nulls(n, pa.list_(pa.int32())),
    })


def labels_frame() -> tuple[pd.DataFrame, dict, dict]:
    r = _rng()
    n = 60
    df = pd.DataFrame({
        "stuid": [f"{10000 + 3 * i:06d}" for i in range(n)],
        "ecodis": r.integers(0, 2, n).astype(np.int8),
        "grade": r.choice(np.array([0, 1, 2, 3], dtype=np.int8), n),
        "dob": pd.to_datetime([dt.date(2012, 1, 1) + dt.timedelta(days=int(k)) for k in r.integers(0, 1500, n)]),
        "score": np.where(np.arange(n) % 8 == 0, np.nan, np.round(r.uniform(40, 100, n), 1)),
        "visits": pd.array([None if i % 6 == 0 else int(r.integers(0, 40)) for i in range(n)], dtype="Int16"),
        "town": [["Northfield", "Easton", "Westbury"][int(k)] for k in r.integers(0, 3, n)],
    })
    value_labels = {"ecodis": {0: "No", 1: "Yes"}, "grade": {0: "KG", 1: "First", 2: "Second", 3: "Third"}}
    variable_labels = {"stuid": "Student ID", "ecodis": "Economically disadvantaged", "grade": "Grade level",
                       "dob": "Date of birth", "score": "Reading score"}
    return df, value_labels, variable_labels


def records() -> list[dict]:
    r = _rng()
    out = []
    for i in range(40):
        rec: dict = {"id": 1000 + i, "name": FIRST[int(r.integers(0, len(FIRST)))] + ("é" if i % 7 == 0 else ""),
                     "score": round(float(r.uniform(50, 100)), 1), "active": bool(r.integers(0, 2)),
                     "level": ["low", "mid", "high"][int(r.integers(0, 3))]}
        if i % 5 == 0:
            rec["score"] = None
        if i % 4 != 0:
            rec["nickname"] = "kid" + str(int(r.integers(0, 9)))
        rec["visits"] = int(r.integers(0, 12))
        out.append(rec)
    return out


def students_csv() -> bytes:
    r = _rng()
    lines = ["student_id,first_name,grade,score"]
    for i in range(50):
        lines.append(f"{200000 + 11 * i},{FIRST[int(r.integers(0, len(FIRST)))]},"
                     f"{['KG', '01', '02', '03'][int(r.integers(0, 4))]},{int(r.integers(40, 100))}")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def main() -> None:
    t = types_table()
    codecs = {name: "zstd" for name in t.schema.names}
    codecs["gpa"] = "gzip"
    pq.write_table(t, HERE / "types.parquet", compression=codecs, row_group_size=50)
    pq.write_table(nested_table(), HERE / "nested.parquet")
    feather.write_feather(t, HERE / "types.feather", compression="zstd", chunksize=40)
    df, vl, varl = labels_frame()
    df.to_stata(HERE / "labels.dta", write_index=False, version=118, value_labels=vl, variable_labels=varl,
                data_label="Area F labelled fixture", convert_dates={"dob": "td"},
                time_stamp=dt.datetime(2026, 10, 4, 9, 30))
    recs = records()
    (HERE / "records.json").write_bytes((json.dumps(recs, indent=2) + "\n").encode("utf-8"))
    jl = "\n".join(json.dumps(x, separators=(",", ":"), ensure_ascii=False) for x in recs[:30]) + "\n"
    (HERE / "lines.jsonl").write_bytes(jl.encode("utf-8"))
    raw = students_csv()
    buf = io.BytesIO()
    with gzip.GzipFile(filename="students.csv", mode="wb", fileobj=buf, mtime=1759570000) as gz:
        gz.write(raw)
    (HERE / "students.csv.gz").write_bytes(buf.getvalue())
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("students.csv", date_time=(2026, 10, 4, 9, 30, 0)), raw,
                    compress_type=zipfile.ZIP_DEFLATED)
    (HERE / "students.csv.zip").write_bytes(zbuf.getvalue())


if __name__ == "__main__":
    main()
