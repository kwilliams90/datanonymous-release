"""AREA F: Parquet / Feather reading, native types, writing with zero schema drift, nested/binary, hive."""

from __future__ import annotations

import decimal
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from datanonymous.config import Settings
from datanonymous.detect.context import DetectContext
from datanonymous.detect.types import detect_column
from datanonymous.errors import ReadError, UnsupportedColumnError, WriteError
from datanonymous.io import arrowio
from datanonymous.model import ColumnType, FileLayout
from test_area_f_util import AB_COLUMNS, FIX, all_bytes, identity_write, read, report, run_dir, text_of

NULL_AB = [x for c in AB_COLUMNS for x in ("--null", c)]


def assert_structural_schema(src: pa.Schema, dst: pa.Schema) -> None:
    """Zero schema drift (types, nullability, order) and only structural metadata (R1-01): the fixture's
    free-form schema metadata ('source', 'version') and field description are never copied."""
    assert dst.equals(src, check_metadata=False)
    assert dst.metadata is None and all(f.metadata is None for f in dst)


def _write(tmp: Path, table: pa.Table, name: str = "t.parquet", **kw) -> Path:
    p = tmp / name
    if name.endswith(".feather"):
        feather.write_feather(table, p, **kw)
    else:
        pq.write_table(table, p, **kw)
    return p


def _ctx(values: list, native: dict, name: str = "c") -> DetectContext:
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    s = Settings(mode="fidelity")
    return DetectContext(index=0, name=name, values=arr, file_rel="t.parquet", table=None, format="parquet",
                         options=s.column_options("t.parquet", None, name), settings=s, native=native)


# --------------------------------------------------------------------------- reading / canonical text
def test_read_types_parquet_canonical_text() -> None:
    rr = read(FIX / "types.parquet", "parquet")
    t = rr.tables[0]
    assert t.header == ["student_id", "first_name", "grade", "school", "balance", "enrolled_at", "dob",
                        "start_time", "absences", "ecodis", "gpa"]
    assert t.n_rows == 120
    v = {c.name: c.values.tolist() for c in t.columns}
    assert v["balance"][0] is None and v["balance"][1] == "551.68"
    assert v["enrolled_at"][1] == "2023-11-16 19:23:46.721597-06:00"  # standard time
    assert v["enrolled_at"][3].endswith("-05:00")  # daylight time
    assert v["dob"][1] == "2011-03-31"
    assert v["start_time"][0] == "07:26:47"
    assert v["absences"][0] is None and v["absences"][1] == "17"
    assert v["ecodis"][0] is None and set(v["ecodis"][1:]) <= {"true", "false", None}
    assert v["gpa"][0] == "3.17"
    natives = {c.name: c.native for c in t.columns}
    assert natives["grade"]["kind"] == "dictionary"
    assert natives["grade"]["dictionary_values"] == ["PK", "KG", "01", "02", "03", "04", "05", "06"]
    assert natives["enrolled_at"]["frac_digits"] == 6 and natives["start_time"]["frac_digits"] == 0
    ex = rr.layout.extra
    assert ex["row_group_rows"] == 50 and ex["row_groups"] == 3 and ex["codec"] == "mixed"
    assert ex["_codecs"][-1] == "GZIP" and ex["_codecs"][0] == "ZSTD"
    assert ex["arrow_schema_stored"] is True and ex["parquet_version"] == "2.6"
    assert ex[arrowio.SCHEMA_KEY].metadata[b"source"] == b"datanonymous area F fixture"
    pub = rr.layout.to_public()["extra"]
    assert "_arrow_schema" not in pub and "_codecs" not in pub and pub["codec"] == "mixed"


def test_read_feather_details() -> None:
    rr = read(FIX / "types.feather", "feather")
    ex = rr.layout.extra
    assert ex["feather_version"] == 2 and ex["codec"] == "zstd"
    assert ex["batch_rows"] == 40 and ex["record_batches"] == 3
    assert [c.values[1] for c in rr.tables[0].columns][:3] == ["S100007", "Ben", "05"]


@pytest.mark.parametrize("codec", ["lz4", "zstd", "uncompressed"])
def test_feather_codec_detected(tmp_path: Path, codec: str) -> None:
    p = _write(tmp_path, pa.table({"a": [1, 2, 3]}), "x.feather", compression=codec)
    assert read(p, "feather").layout.extra["codec"] == codec


def test_float_and_special_values(tmp_path: Path) -> None:
    t = pa.table({"f64": pa.array([2.0, 0.1, float("nan"), float("inf"), -float("inf"), None, 1e-7]),
                  "f32": pa.array([0.1, 2.5, None, 3.0, 1.0, 1.0, 1.0], pa.float32())})
    rr = read(_write(tmp_path, t), "parquet")
    f64, f32 = (c.values.tolist() for c in rr.tables[0].columns)
    assert f64 == ["2.0", "0.1", "NaN", "inf", "-inf", None, "0.0000001"]
    assert f32[:4] == ["0.1", "2.5", None, "3.0"]  # float32 shortest text, not 0.10000000149...


def test_temporal_rendering_units(tmp_path: Path) -> None:
    t = pa.table({
        "ts_s": pa.array([0, 86399], pa.int64()).cast(pa.timestamp("s")),
        "ts_ms_utc": pa.array([1500, None], pa.int64()).cast(pa.timestamp("ms", tz="UTC")),
        "ts_off": pa.array([0, 3600 * 10**9], pa.int64()).cast(pa.timestamp("ns", tz="+05:30")),
        "d64": pa.array([86400000, None], pa.int64()).cast(pa.date64()),
        "t32": pa.array([3661, 0], pa.int32()).cast(pa.time32("s")),
        "dur": pa.array([-90 * 10**6, 100 * 3600 * 10**6 + 5], pa.int64()).cast(pa.duration("us")),
        "old": pa.array([-2208988800 - 86400 * 365 * 30, 0], pa.int64()).cast(pa.timestamp("s", tz="America/Chicago")),
    })
    v = {c.name: c.values.tolist() for c in read(_write(tmp_path, t), "parquet").tables[0].columns}
    assert v["ts_s"] == ["1970-01-01 00:00:00", "1970-01-01 23:59:59"]
    assert v["ts_ms_utc"] == ["1970-01-01 00:00:01.500+00:00", None]
    assert v["ts_off"] == ["1970-01-01 05:30:00+05:30", "1970-01-01 06:30:00+05:30"]  # no fractions: none shown
    assert v["d64"] == ["1970-01-02", None]
    assert v["t32"] == ["01:01:01", "00:00:00"]
    assert v["dur"] == ["-00:01:30.000000", "100:00:00.000005"]
    assert v["old"][0].endswith("-05:50:36")  # local mean time: an offset with seconds


@pytest.mark.parametrize("text,expect", [
    ("2024-03-10 01:59:59-06:00", (1710035999, 0, -21600, True)),  # wall-clock seconds
    ("2024-03-10T08:00:00Z", (1710057600, 0, 0, True)),
    ("2024-03-10 08:00:00.5+0530", (1710057600, 500_000_000, 19800, True)),
    ("1999-12-31", (946598400, 0, None, False)),
    ("2024-02-30", None),
    ("nope", None),
])
def test_parse_datetime(text: str, expect) -> None:
    assert arrowio.parse_datetime(text) == expect


def test_parse_clock() -> None:
    assert arrowio.parse_clock("01:02:03.25") == (3723 * 10**9 + 250_000_000)
    assert arrowio.parse_clock("-100:00:00") == -360000 * 10**9
    assert arrowio.parse_clock("1:61:00") is None


# --------------------------------------------------------------------------- native detection
def test_detect_native_dictionary_and_decimal() -> None:
    rr = read(FIX / "types.parquet", "parquet")
    cols = {c.name: c for c in rr.tables[0].columns}
    g = arrowio.detect_native(_ctx(cols["grade"].values.tolist(), cols["grade"].native, "grade"))
    assert g.type is ColumnType.ORDINAL and g.source == "native"
    assert g.params["levels"] == ["PK", "KG", "01", "02", "03", "04", "05", "06"]
    s = arrowio.detect_native(_ctx(cols["school"].values.tolist(), cols["school"].native, "school"))
    assert s.type is ColumnType.FACTOR and s.params["dictionary_size"] == 3 and s.params["n_levels"] == 2
    # numbers, decimals included, are typed by the value rules (header rules first; R4-05): DECIMAL there
    d = detect_column(_ctx(cols["balance"].values.tolist(), cols["balance"].native, "balance"))
    assert d.type is ColumnType.DECIMAL and d.source == "native"
    assert d.params["precision"] == 9 and d.params["scale"] == 2 and d.params["decimals"] == 2
    # plain ints/strings/bools/floats/decimals: no opinion (value rules decide)
    for name in ("absences", "student_id", "ecodis", "gpa", "balance"):
        assert arrowio.detect_native(_ctx(cols[name].values.tolist(), cols[name].native, name)) is None


def test_detect_native_temporal_uses_area_a_or_fails_with_fix(monkeypatch) -> None:
    from datanonymous.detect import context, dates

    rr = read(FIX / "types.parquet", "parquet")
    col = {c.name: c for c in rr.tables[0].columns}["dob"]
    seen = {}

    def fake(ctx):
        seen["forced"] = ctx.forced_type
        return context.Detection(ColumnType.DATE, params={"format": "%Y-%m-%d"})

    monkeypatch.setattr(dates, "detect_dates", fake)
    ctx = _ctx(col.values.tolist(), col.native, "dob")
    det = arrowio.detect_native(ctx)
    assert seen["forced"] is ColumnType.DATE and ctx.forced_type is None
    assert det.type is ColumnType.DATE and det.source == "native" and det.params["format"] == "%Y-%m-%d"
    monkeypatch.setattr(dates, "detect_dates", lambda ctx: None)
    with pytest.raises(UnsupportedColumnError) as ei:
        arrowio.detect_native(_ctx(col.values.tolist(), col.native, "dob"))
    assert ei.value.fix.startswith("--null dob") and "--exclude dob" in ei.value.fix
    # tz-aware timestamps ask for DATETIME_TZ
    ts = {c.name: c for c in rr.tables[0].columns}["enrolled_at"]
    monkeypatch.setattr(dates, "detect_dates", fake)
    arrowio.detect_native(_ctx(ts.values.tolist(), ts.native, "enrolled_at"))
    assert seen["forced"] is ColumnType.DATETIME_TZ


# --------------------------------------------------------------------------- identity round trips
@pytest.mark.parametrize("name,fmt", [("types.parquet", "parquet"), ("types.feather", "feather")])
def test_identity_roundtrip_zero_drift(tmp_path: Path, name: str, fmt: str) -> None:
    rr = read(FIX / name, fmt)
    out = identity_write(rr, fmt, tmp_path / name)
    reader = pq.read_table if fmt == "parquet" else feather.read_table
    src, dst = reader(FIX / name), reader(out)
    assert_structural_schema(src.schema, dst.schema)
    for a, b in zip(src.schema, dst.schema):
        assert a.nullable == b.nullable
    # every value and the dictionary order; an unused entry is written only when it is tool vocabulary (R2-03):
    # the grade level "06" no row holds stays, the unused school "Unused HS" (never checked) is left out
    assert dst.drop_columns(["school"]).equals(src.drop_columns(["school"]))
    assert dst.column("school").to_pylist() == src.column("school").to_pylist()
    assert dst.column("school").combine_chunks().dictionary.to_pylist() == ["Adams MS", "Lincoln ES"]
    rr2 = read(out, fmt, purpose="selfcheck")
    assert arrowio.layout_diff(rr.layout, rr2.layout, [{}], rr2.tables) == []
    if fmt == "parquet":
        md_s, md_o = pq.ParquetFile(FIX / name).metadata, pq.ParquetFile(out).metadata
        assert [md_o.row_group(i).num_rows for i in range(md_o.num_row_groups)] == [50, 50, 20]
        assert [md_o.row_group(0).column(j).compression for j in range(md_o.num_columns)] == \
               [md_s.row_group(0).column(j).compression for j in range(md_s.num_columns)]
    else:
        assert read(out, fmt).layout.extra["codec"] == "zstd"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_identity_roundtrip_feather_v1(tmp_path: Path) -> None:
    t = pa.table({"a": pa.array([1, None, 3], pa.int64()), "s": pa.array(["x", None, "z"])})
    p = _write(tmp_path, t, "v1.feather", version=1)
    rr = read(p, "feather")
    assert rr.layout.extra["feather_version"] == 1
    out = identity_write(rr, "feather", tmp_path / "out.feather")
    assert out.read_bytes()[:4] == b"FEA1"
    assert feather.read_table(out).equals(feather.read_table(p))


def test_parquet_writer_settings_mirrored(tmp_path: Path) -> None:
    t = pa.table({"a": pa.array(range(30), pa.int16()), "b": pa.array([str(i % 3) for i in range(30)])})
    p = tmp_path / "s.parquet"
    pq.write_table(t, p, version="1.0", compression="gzip", use_dictionary=["b"], write_statistics=False,
                   row_group_size=7, store_schema=False)
    rr = read(p, "parquet")
    out = identity_write(rr, "parquet", tmp_path / "o.parquet")
    ms, mo = pq.ParquetFile(p).metadata, pq.ParquetFile(out).metadata
    assert mo.format_version == ms.format_version == "1.0"
    assert mo.num_row_groups == ms.num_row_groups == 5
    assert b"ARROW:schema" not in (mo.metadata or {})
    for j in range(2):
        a, b = ms.row_group(0).column(j), mo.row_group(0).column(j)
        assert (a.compression, a.is_stats_set) == (b.compression, b.is_stats_set)
        assert ("RLE_DICTIONARY" in a.encodings or "PLAIN_DICTIONARY" in a.encodings) == (
            "RLE_DICTIONARY" in b.encodings or "PLAIN_DICTIONARY" in b.encodings)
    assert pq.read_table(out).equals(pq.read_table(p))


# --------------------------------------------------------------------------- writer conversions
def _field_write(values: list, field: pa.Field, native: dict | None = None):
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arrowio.values_to_arrow(arr, field, native or {"kind": arrowio.arrow_kind(field.type)})


def test_writer_parses_area_a_forms() -> None:
    ts = _field_write(["2024-03-10 01:59:59-06:00", "2024-03-10T08:00:00Z", None, ""],
                      pa.field("t", pa.timestamp("us", tz="America/Chicago")))
    assert ts.cast(pa.int64()).to_pylist() == [1710057599 * 10**6, 1710057600 * 10**6, None, None]
    # a zoned value without an offset is localised in the column's zone
    loc = _field_write(["2024-07-01 12:00:00", None, "2024-07-01 12:00:00"],
                       pa.field("t", pa.timestamp("s", tz="America/Chicago")))
    assert loc.cast(pa.int64()).to_pylist() == [1719853200, None, 1719853200]
    assert _field_write(["2024-01-02"], pa.field("d", pa.date32())).to_pylist()[0].isoformat() == "2024-01-02"
    assert _field_write(["01:00:00.5"], pa.field("t", pa.time64("ns"))).cast(pa.int64()).to_pylist() == [
        3600 * 10**9 + 5 * 10**8]
    assert _field_write(["-00:00:02"], pa.field("d", pa.duration("ms"))).cast(pa.int64()).to_pylist() == [-2000]
    dec = _field_write(["12.5", "-0.004", None], pa.field("m", pa.decimal128(9, 2)))
    assert dec.to_pylist() == [decimal.Decimal("12.50"), decimal.Decimal("-0.00"), None]
    assert _field_write(["true", "FALSE", None], pa.field("b", pa.bool_())).to_pylist() == [True, False, None]
    assert _field_write(["NaN", "1e3", ""], pa.field("f", pa.float32())).to_pylist()[1:] == [1000.0, None]


@pytest.mark.parametrize("values,t", [(["x"], pa.int32()), (["1.5"], pa.int8()), (["300"], pa.int8()),
                                      (["maybe"], pa.bool_()), (["2024-13-01"], pa.date32()),
                                      (["25:00:00"], pa.time64("us")), (["1e30"], pa.decimal128(9, 2))])
def test_writer_refuses_values_the_type_cannot_hold(values, t) -> None:
    with pytest.raises(WriteError) as ei:
        _field_write(values, pa.field("col", t))
    assert "col" in ei.value.message and values[0] not in ei.value.message  # the value itself is never quoted


def test_dictionary_rebuilt_for_non_label_types() -> None:
    from datanonymous.model import ColumnProfile

    src = pa.DictionaryArray.from_arrays(pa.array([0, 1], pa.int8()), pa.array(["Real A", "Real B"]))
    field = pa.field("who", src.type)
    native = {"kind": "dictionary", "dictionary": src.dictionary, "dictionary_values": ["Real A", "Real B"]}
    p = ColumnProfile(index=0, name="who", type=ColumnType.HIGH_CARD_CATEGORICAL)
    arr = np.array(["who_2", "who_1", None], dtype=object)
    out = arrowio.values_to_arrow(arr, field, native, p)
    assert out.type == src.type and out.dictionary.to_pylist() == ["who_1", "who_2"]  # no real label kept
    p2 = ColumnProfile(index=0, name="who", type=ColumnType.FACTOR)
    out2 = arrowio.values_to_arrow(np.array(["Real B", None], dtype=object), field, native, p2)
    # label-safe: the source entries the rows hold (an unused entry was never checked, R2-03)
    assert out2.dictionary.to_pylist() == ["Real B"] and out2.indices.to_pylist() == [0, None]


def test_layout_diff_reports_drift() -> None:
    s = pa.schema([pa.field("a", pa.int32(), metadata={"d": "x"}), pa.field("b", pa.string())],
                  metadata={"k": "v"})
    # free-form source metadata copied into the output is drift (R1-01: only structural metadata is written)
    o = pa.schema([pa.field("a", pa.int64(), metadata={"d": "x"}), pa.field("b", pa.string(), nullable=False)],
                  metadata={"k": "v"})
    src = FileLayout("parquet", extra={arrowio.SCHEMA_KEY: s, "parquet_version": "2.6", "_codecs": ["ZSTD", "ZSTD"],
                                       "row_group_rows": 10})
    out = FileLayout("parquet", extra={arrowio.SCHEMA_KEY: o, "parquet_version": "1.0", "_codecs": ["SNAPPY", "ZSTD"],
                                       "row_group_rows": 20})
    issues = arrowio.layout_diff(src, out, [], [])
    text = "\n".join(issues)
    assert "Arrow type int32 -> int64" in text and "nullability" in text and "field metadata" in text
    assert "schema (file) metadata" in text and "version 2.6 -> 1.0" in text and "codec ZSTD -> SNAPPY" in text
    assert "row-group size" in text
    # an excluded column is not drift; the source's free-form metadata is expected to be absent (R1-01)
    o2 = pa.schema([pa.field("b", pa.string())])
    assert arrowio.schema_diff(s, o2) == []
    o3 = pa.schema([pa.field("b", pa.string())], metadata={"k": "v"})  # copied free-form metadata is a difference
    assert any("schema (file) metadata" in i for i in arrowio.schema_diff(s, o3))


# --------------------------------------------------------------------------- nested / binary / hive / NOT NULL
def test_nested_columns_fail_before_writing_with_fix(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedColumnError) as ei:
        read(FIX / "nested.parquet", "parquet")
    msg = ei.value.message
    assert "'tags' (list<element: string>)" in msg and "'address' (struct" in msg and "'photo' (binary)" in msg
    assert "unused_list" not in msg  # all-null nested columns are readable
    for c in ("tags", "address", "photo"):
        assert f"--null {c} or --exclude {c}" in ei.value.fix
    res, out, rep = run_dir(FIX / "nested.parquet", tmp_path)
    assert res.exit_code == 3 and not (out / "nested.parquet").exists()
    fails = report(rep)["failures"]
    assert fails and "--null tags" in " ".join(fails[0]["fixes"])


def test_nested_null_and_exclude_keep_schema(tmp_path: Path) -> None:
    res, out, rep = run_dir(FIX / "nested.parquet", tmp_path, "--null", "tags", "--null", "address",
                            "--exclude", "photo")
    assert res.exit_code == 0, res.output
    src, dst = pq.read_table(FIX / "nested.parquet"), pq.read_table(out / "nested.parquet")
    assert dst.schema.names == ["record_id", "score", "tags", "address", "unused_list"]
    for name in ("tags", "address", "unused_list"):
        assert dst.schema.field(name).type == src.schema.field(name).type
        assert dst.column(name).null_count == dst.num_rows
    blob = all_bytes(out, rep)
    for real in (b"Town1", b"Town3", b"70005"):
        assert real not in blob  # nested values are never copied or stringified
    codes = {w["code"] for w in report(rep)["warnings"]}
    assert "column_excluded" in codes  # --exclude changes the schema: warned


def test_hive_partition_folder_fails(tmp_path: Path) -> None:
    root = tmp_path / "in"
    part = root / "year=2024"
    part.mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2, 3]}), part / "part-0.parquet")
    pq.write_table(pa.table({"a": [4, 5, 6]}), root / "plain.parquet")
    res, out, rep = run_dir(root, tmp_path)
    assert res.exit_code == 3
    fails = {f["source"]: f for f in report(rep)["failures"]}
    assert set(fails) == {"year=2024/part-0.parquet"}
    assert "hive-partitioned" in fails["year=2024/part-0.parquet"]["reasons"][0]
    assert (out / "plain.parquet").is_file() and not (out / "year=2024").exists()
    # a single file inside a partition folder is refused too
    with pytest.raises(ReadError):
        read(part / "part-0.parquet", "parquet")


def test_not_null_field_cannot_be_nulled(tmp_path: Path) -> None:
    res, out, rep = run_dir(FIX / "types.parquet", tmp_path, *NULL_AB, "--null", "gpa")
    assert res.exit_code == 3
    f = report(rep)["failures"][0]
    assert "NOT NULL" in f["reasons"][0] and f["fixes"] == ["--exclude gpa (a NOT NULL column cannot be written "
                                                            "all-missing)"]


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("name,fmt", [("types.parquet", "parquet"), ("types.feather", "feather")])
def test_cli_end_to_end_schema_and_privacy(tmp_path: Path, name: str, fmt: str) -> None:
    res, out, rep = run_dir(FIX / name, tmp_path, *NULL_AB)
    assert res.exit_code == 0, res.output
    reader = pq.read_table if fmt == "parquet" else feather.read_table
    src, dst = reader(FIX / name), reader(out / name)
    assert_structural_schema(src.schema, dst.schema)  # zero schema drift, free-form metadata dropped
    assert dst.num_rows == src.num_rows
    # dictionary columns keep their order and the levels the rows hold (plus the unused grade level "06",
    # tool vocabulary); the unused school "Unused HS" is never written (R2-03); only observed levels appear
    for c in ("grade", "school"):
        a, b = src.column(c).combine_chunks(), dst.column(c).combine_chunks()
        used = [v for v in a.dictionary.to_pylist() if v in set(a.to_pylist()) or (c == "grade" and v == "06")]
        assert b.dictionary.to_pylist() == used
        assert set(b.to_pylist()) - {None} <= set(a.to_pylist())
    assert sorted(x for x in dst.column("grade").to_pylist()) == sorted(src.column("grade").to_pylist())
    # missingness is exact, values stay in range, types are native
    for c in ("absences", "ecodis"):
        assert dst.column(c).null_count == src.column(c).null_count
    gpa = dst.column("gpa").to_numpy()
    assert gpa.min() >= 1.0 and gpa.max() <= 4.0
    # privacy: IDs and first names never real, anywhere in the out dir or the report
    real_ids = set(src.column("student_id").to_pylist())
    real_names = set(src.column("first_name").to_pylist())
    assert not real_ids & set(dst.column("student_id").to_pylist())
    assert not real_names & set(dst.column("first_name").to_pylist())
    blob = all_bytes(out, rep)
    for v in sorted(real_ids)[:40]:
        assert v.encode() not in blob
    txt = text_of(out, rep)
    for nm in real_names:
        assert f'"{nm}"' not in txt and f"| {nm} " not in txt
    data = report(rep)
    assert data["totals"]["PASS"] + data["totals"]["WARN"] == 1
    entry = data["files"][0] if "files" in data else None
    assert entry is None or entry.get("status") in ("PASS", "WARN")
    js = (rep / "files" / f"{name}.json").read_text(encoding="utf-8")
    assert "datanonymous area F fixture" not in js  # schema metadata never reaches the report
    assert "days absent" not in js


@pytest.mark.parametrize("name", ["types.parquet", "types.feather", "nested.parquet"])
def test_same_seed_same_bytes(tmp_path: Path, name: str) -> None:
    flags = NULL_AB if name.startswith("types") else ["--null", "tags", "--null", "address", "--null", "photo"]
    r1, o1, _ = run_dir(FIX / name, tmp_path / "a", *flags, seed="11")
    r2, o2, _ = run_dir(FIX / name, tmp_path / "b", *flags, seed="11")
    assert r1.exit_code == r2.exit_code == 0
    assert (o1 / name).read_bytes() == (o2 / name).read_bytes()


def test_inferred_dtypes_match_source(tmp_path: Path) -> None:
    res, out, rep = run_dir(FIX / "types.parquet", tmp_path, *NULL_AB)
    assert res.exit_code == 0
    lay = read(FIX / "types.parquet", "parquet").layout
    src = arrowio.inferred_dtypes(FIX / "types.parquet", lay, None)
    dst = arrowio.inferred_dtypes(out / "types.parquet", lay, None)
    assert src is not None and src == dst and src.count("category") == 2
    assert "dtype_mismatch" not in (rep / "files" / "types.parquet.md").read_text(encoding="utf-8")


def test_area_a_b_columns_generated_or_failed_clearly(tmp_path: Path) -> None:
    """Without --null the A/B columns are either generated with the exact Arrow type (areas A and B
    merged) or the file FAILs naming the column and a fix (foundation stubs) - never written wrongly."""
    res, out, rep = run_dir(FIX / "types.parquet", tmp_path)
    if res.exit_code == 0:
        src, dst = pq.read_table(FIX / "types.parquet"), pq.read_table(out / "types.parquet")
        assert_structural_schema(src.schema, dst.schema)
        for c in AB_COLUMNS:
            assert dst.column(c).null_count == src.column(c).null_count
        return
    assert res.exit_code == 3 and not (out / "types.parquet").exists()
    f = report(rep)["failures"][0]
    reason = f["reasons"][0]
    assert any(c in reason for c in AB_COLUMNS)
    assert "cannot be stored as Arrow" not in reason  # never a writer error: A/B either work or FAIL in detection
    assert f["fixes"]


def test_native_nulls_in_id_and_value_columns(tmp_path: Path) -> None:
    """Typed formats use None for nulls: ID planning and missingness patterns must treat it as missing."""
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({"student_id": ["S1", None, "S3", "S4", "S5"], "v": [1, 2, None, 4, 5]}),
                   src / "ids.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    dst = pq.read_table(out / "ids.parquet")
    assert dst.column("student_id").null_count == 1 and dst.column("v").null_count == 1
    assert not {"S1", "S3", "S4", "S5"} & set(dst.column("student_id").to_pylist())


def test_empty_tables(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({"a": pa.array([], pa.int32()), "g": pa.array([], pa.string()).dictionary_encode()}),
                   src / "e.parquet")
    feather.write_feather(pa.table({"a": pa.array([], pa.int64())}), src / "e.feather")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert pq.read_table(out / "e.parquet").schema.equals(pq.read_table(src / "e.parquet").schema)
    assert feather.read_table(out / "e.feather").num_rows == 0


def test_unknown_or_bad_parquet_fails(tmp_path: Path) -> None:
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"PAR1" + b"\x00" * 20)
    with pytest.raises(ReadError):
        read(p, "parquet")


def test_dictionary_unification_across_row_groups(tmp_path: Path) -> None:
    t1 = pa.table({"g": pa.array(["a", "b"]).dictionary_encode()})
    t2 = pa.table({"g": pa.array(["c", "a"]).dictionary_encode()})
    p = tmp_path / "d.parquet"
    with pq.ParquetWriter(p, t1.schema) as w:
        w.write_table(t1)
        w.write_table(t2)
    rr = read(p, "parquet")
    assert rr.tables[0].columns[0].values.tolist() == ["a", "b", "c", "a"]
    out = identity_write(rr, "parquet", tmp_path / "o.parquet")
    assert pq.read_table(out).column("g").to_pylist() == ["a", "b", "c", "a"]


def test_sas_like_extension_type_is_unsupported(tmp_path: Path) -> None:
    t = pa.table({"id": ["a", "b"], "u": pa.array([b"0123456789abcdef", None], pa.binary(16))})
    p = _write(tmp_path, t)
    with pytest.raises(UnsupportedColumnError):
        read(p, "parquet")
    shutil.copy(p, tmp_path / "copy.parquet")
    rr = read(tmp_path / "copy.parquet", "parquet", settings=_settings_null("u"))
    assert rr.tables[0].columns[1].values.tolist() == [None, None]


def _settings_null(col: str) -> Settings:
    from conftest import build_fidelity_settings as build_settings

    return build_settings({"null": [col]})


def _simulate_areas_a_b(monkeypatch) -> None:
    """Stand-ins for the AREA A/B hooks: detect the forced type, regenerate observed canonical texts."""
    from datanonymous.detect import context
    from datanonymous.detect import dates as ddates
    from datanonymous.generate import dates as gdates
    from datanonymous.generate import numeric as gnumeric
    from datanonymous.generate.quota import quota_sample

    def detect(ctx):
        return context.Detection(ctx.forced_type, params={"simulated": True}) if ctx.forced_type else None

    def generate(profile, model, n, rng):
        values, adjusted = quota_sample(list(model.nonmissing_counts), list(model.nonmissing_counts.values()), n,
                                        rng)
        return values, {"rare_category_adjusted": adjusted}

    monkeypatch.setattr(ddates, "detect_dates", detect)
    monkeypatch.setattr(gdates, "generate_dates", generate)
    monkeypatch.setattr(gdates, "render_dates", lambda p, v: v)
    monkeypatch.setattr(gnumeric, "generate_numeric_extended", generate)
    monkeypatch.setattr(gnumeric, "render_numeric_extended", lambda p, v: v)


@pytest.mark.parametrize("name,fmt", [("types.parquet", "parquet"), ("types.feather", "feather")])
def test_e2e_with_simulated_area_a_b(monkeypatch, tmp_path: Path, name: str, fmt: str) -> None:
    """The writer stores AREA A/B output (canonical texts) back in the exact Arrow types."""
    _simulate_areas_a_b(monkeypatch)
    res, out, rep = run_dir(FIX / name, tmp_path)
    assert res.exit_code == 0, res.output
    reader = pq.read_table if fmt == "parquet" else feather.read_table
    src, dst = reader(FIX / name), reader(out / name)
    assert_structural_schema(src.schema, dst.schema)
    for c in AB_COLUMNS:
        assert dst.column(c).null_count == src.column(c).null_count
        assert set(dst.column(c).to_pylist()) <= set(src.column(c).to_pylist())  # simulated: observed values
    types = {c["name"]: c["type"] for c in json.loads((rep / "files" / f"{name}.json").read_text(
        encoding="utf-8"))["tables"][0]["columns"]}
    assert types["balance"] == "DECIMAL" and types["dob"] == "DATE" and types["enrolled_at"] == "DATETIME_TZ"
    # a dictionary column under an org-code header is a D10 org code, as it is in a text file (R4-05)
    assert types["start_time"] == "TIME" and types["grade"] == "ORDINAL" and types["school"] == "ORG_CODE"
