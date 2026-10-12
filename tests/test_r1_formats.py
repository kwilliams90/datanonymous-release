"""Round-1 review fixes, formats cluster.

* R1-01: free-form format metadata (Arrow schema/field metadata, Stata data and
  variable labels, gzip/zip stored names) is never copied to the synthetic file;
  only structural metadata is kept, the self-check expects exactly that, and the
  file gets a counts-only warning.
* R1-14: Arrow decimals wider than float64 (more than 15 significant digits) are
  generated on exact scaled integers: valid files are written with the exact
  extremes instead of FAILing.
* R1-16: statistics of large finite numbers are finite, and report / --json
  output is strict JSON (no Infinity/NaN tokens).
"""

from __future__ import annotations

import gzip
import io
import json
import math
import statistics
import warnings
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from datanonymous.generate import numeric as G
from datanonymous.generate.table import generate_values
from datanonymous.io import arrowio, compress, statio
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType, FileLayout
from datanonymous.profile.stats import numeric_summary
from datanonymous.report.report import _json, finite_json
from test_area_f_util import all_bytes, read, report, run_dir

NAME = b"Zelmira"
SURNAME = b"Quenby"
PHONE = b"312-555-0199"
EMAIL = b"zelmira.quenby@realhospital.org"
MARKERS = (NAME, SURNAME, PHONE, EMAIL, b"realhospital", b"HIV")


def _codes(rep: Path, name: str) -> list[str]:
    return [w["code"] for w in report(rep)["warnings"] if w.get("file") == name]


def _strict(text: str) -> object:
    def bad(c: str) -> object:
        raise ValueError(f"non-standard JSON constant {c}")

    return json.loads(text, parse_constant=bad)


def _assert_no_markers(*dirs: Path) -> None:
    blob = all_bytes(*dirs)
    for m in MARKERS:
        assert m not in blob, m


# =========================================================================== R1-01 Arrow metadata
def _arrow_table(n: int = 30) -> pa.Table:
    schema = pa.schema(
        [pa.field("score", pa.int32(), metadata={b"comment": b"Patient Zelmira Quenby has HIV. Call 312-555-0199.",
                                                 b"PARQUET:field_id": b"7"}),
         pa.field("level", pa.string(), metadata={b"PARQUET:field_id": b"8"})],
        metadata={b"owner": EMAIL, b"org.apache.spark.sql.parquet.row.metadata": b'{"comment": "Zelmira"}'})
    return pa.Table.from_arrays([pa.array([i % 9 for i in range(n)], pa.int32()),
                                 pa.array([["lo", "mid", "hi"][i % 3] for i in range(n)])], schema=schema)


@pytest.mark.parametrize("name", ["meta.parquet", "meta.feather"])
def test_arrow_free_form_metadata_is_never_copied(tmp_path: Path, name: str) -> None:
    src = tmp_path / "src"
    src.mkdir()
    t = _arrow_table()
    (pq.write_table if name.endswith(".parquet") else feather.write_feather)(t, src / name)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output  # the self-check accepts the structural-only metadata
    dst = (pq.read_schema if name.endswith(".parquet") else lambda p: feather.read_table(p).schema)(out / name)
    assert dst.equals(t.schema, check_metadata=False)
    assert dst.metadata is None  # owner email, Spark schema JSON with a comment: dropped
    # the Parquet field id is structural and kept; the free-text comment is not
    assert dst.field("score").metadata == {b"PARQUET:field_id": b"7"}
    assert dst.field("level").metadata == {b"PARQUET:field_id": b"8"}
    _assert_no_markers(out, rep)
    assert "arrow_metadata_dropped" in _codes(rep, name)
    data = report(rep)
    assert data["files"][0]["status"] == "WARN" and data["files"][0]["checks"]["layout_selfcheck"] == "PASS"


def test_arrow_metadata_warning_counts_never_quote(tmp_path: Path) -> None:
    p = tmp_path / "m.parquet"
    pq.write_table(_arrow_table(), p)
    rr = read(p, "parquet")
    (msg,) = [m for m in rr.messages if m.code == "arrow_metadata_dropped"]
    assert "2 schema (file) metadata item(s)" in msg.text and "1 column (field) metadata item(s)" in msg.text
    for m in MARKERS:
        assert m.decode() not in msg.text
    # the self-check re-read of our own output says nothing
    assert not [m for m in read(p, "parquet", purpose="selfcheck").messages if m.code == "arrow_metadata_dropped"]


def test_pandas_metadata_keeps_structure_and_drops_free_text(tmp_path: Path) -> None:
    n = 40
    df = pd.DataFrame({"score": [i % 7 for i in range(n)],
                       "grade": pd.Categorical([["A", "B", "C"][i % 3] for i in range(n)], ordered=True),
                       "when": pd.date_range("2024-01-01", periods=n, freq="D").tz_localize("America/Chicago")})
    df = df[df.score > 1]  # a non-range index: stored as __index_level_0__
    df.attrs = {"owner": "Zelmira Quenby"}
    df.columns.name = "Quenby export"
    src = tmp_path / "src"
    src.mkdir()
    df.to_parquet(src / "pd.parquet")
    assert b"pandas" in pq.read_schema(src / "pd.parquet").metadata
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    a, b = pd.read_parquet(src / "pd.parquet"), pd.read_parquet(out / "pd.parquet")
    assert list(b.columns) == list(a.columns) and b.dtypes.astype(str).tolist() == a.dtypes.astype(str).tolist()
    assert b.index.name == a.index.name and len(b) == len(a)  # the stored index is restored as the index
    assert b.attrs == {} and b.columns.name is None  # DataFrame.attrs and the column-level name: dropped
    md = pq.read_schema(out / "pd.parquet").metadata
    assert set(md) == {b"pandas"}
    _assert_no_markers(out, rep)
    assert "arrow_metadata_dropped" in _codes(rep, "pd.parquet")
    assert "dtype_mismatch" not in _codes(rep, "pd.parquet")


def test_pandas_metadata_unit_rules() -> None:
    names = ["a", "b"]
    base = {"index_columns": [{"kind": "range", "name": None, "start": 0, "stop": 3, "step": 1}],
            "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object",
                                "metadata": {"encoding": "UTF-8"}}],
            "columns": [{"name": n, "field_name": n, "pandas_type": "int64", "numpy_type": "int64",
                         "metadata": None} for n in names],
            "creator": {"library": "pyarrow", "version": "25.0.1"}, "pandas_version": "3.0.6"}
    raw = json.dumps(base).encode()
    assert arrowio._pandas_metadata(raw, names) == (raw, 0)  # nothing free-form: the same bytes
    # an excluded column's entry goes; nothing is counted as dropped free text
    clean, lost = arrowio._pandas_metadata(raw, ["a"])
    assert lost == 0 and [c["field_name"] for c in json.loads(clean)["columns"]] == ["a"]
    # a free-text range-index name, attributes and an unknown entry-metadata key are dropped and counted
    bad = json.loads(raw)
    bad["index_columns"][0]["name"] = "Zelmira Quenby"
    bad["attributes"] = {"owner": "Zelmira"}
    bad["columns"][0]["metadata"] = {"description": "Zelmira"}
    clean, lost = arrowio._pandas_metadata(json.dumps(bad).encode(), names)
    assert lost == 3 and b"Zelmira" not in clean
    assert json.loads(clean)["index_columns"][0]["name"] is None
    # a dtype that is not a plain token, or an unknown entry key: the key is dropped whole
    for patch in ({"numpy_type": "Zelmira Quenby"}, {"comment": "x"}):
        worse = json.loads(raw)
        worse["columns"][1].update(patch)
        assert arrowio._pandas_metadata(json.dumps(worse).encode(), names)[0] is None
    assert arrowio._pandas_metadata(b"not json", names) == (None, 1)
    # field metadata: only an integer Parquet field id survives
    assert arrowio.safe_field_metadata({b"PARQUET:field_id": b"3", b"d": b"x"}) == ({b"PARQUET:field_id": b"3"}, 1)
    assert arrowio.safe_field_metadata({b"PARQUET:field_id": b"Zelmira"}) == (None, 1)


def test_arrow_self_check_flags_copied_metadata() -> None:
    s = pa.schema([pa.field("a", pa.int32(), metadata={"d": "note"})], metadata={"owner": "x"})
    assert arrowio.schema_diff(s, pa.schema([pa.field("a", pa.int32())])) == []
    copied = pa.schema([pa.field("a", pa.int32(), metadata={"d": "note"})], metadata={"owner": "x"})
    issues = arrowio.schema_diff(s, copied)
    assert any("field metadata" in i for i in issues) and any("schema (file) metadata" in i for i in issues)


# =========================================================================== R1-01 Stata labels
def test_stata_data_and_variable_labels_are_never_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame({"score": np.arange(25, dtype=np.int16) % 6,
                  "lvl": np.array([i % 3 for i in range(25)], dtype=np.int8)}).to_stata(
        src / "m.dta", write_index=False, version=118, data_label="Patient Zelmira Quenby has HIV.",
        variable_labels={"score": "Call patient at 312-555-0199.", "lvl": "Level"},
        value_labels={"lvl": {0: "low", 1: "mid", 2: "high"}})
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output  # the self-check expects blank labels
    with pd.io.stata.StataReader(out / "m.dta") as r:
        r.read()
        assert r.data_label == "" and set(r.variable_labels().values()) == {""}
        assert r.value_labels() == {"lvl": {0: "low", 1: "mid", 2: "high"}}  # FACTOR levels are kept (D11)
    _assert_no_markers(out, rep)
    assert "stata_labels_dropped" in _codes(rep, "m.dta")
    msg = [w["text"] for w in report(rep)["warnings"] if w["code"] == "stata_labels_dropped"][0]
    assert "the data label and 2 variable label(s)" in msg


def test_stata_self_check_flags_written_labels() -> None:
    cols = {"v": {"typ": "h", "fmt": "%8.0g", "labels": None, "var_label": ""}}
    src = FileLayout("dta", extra={"write_version": 118, "stata_version": 118, "byteorder": "<",
                                   "_data_label": "real", "_columns": {"v": dict(cols["v"], var_label="real")}})
    ok = FileLayout("dta", extra={"stata_version": 118, "byteorder": "<", "_data_label": "", "_columns": cols})
    assert statio.layout_diff(src, ok, [{}], []) == []
    bad = FileLayout("dta", extra={"stata_version": 118, "byteorder": "<", "_data_label": "real",
                                   "_columns": {"v": dict(cols["v"], var_label="real")}})
    issues = statio.layout_diff(src, bad, [{}], [])
    assert any("data label" in i for i in issues) and any("variable label" in i for i in issues)


# =========================================================================== R1-01 container names
PAYLOAD = b"first_name,score\r\n" + b"".join(f"Name{i % 4},{i}\r\n".encode() for i in range(30))


def _gz(path: Path, member: str, payload: bytes = PAYLOAD) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(filename=member, mode="wb", fileobj=buf, mtime=0) as g:
        g.write(payload)
    path.write_bytes(buf.getvalue())


def _zip(path: Path, member: str, payload: bytes = PAYLOAD) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, payload)


@pytest.mark.parametrize("name", ["c.csv.gz", "c.csv.zip", "j.jsonl.gz", "j.jsonl.zip"])
def test_container_names_are_never_copied(tmp_path: Path, name: str) -> None:
    src = tmp_path / "src"
    src.mkdir()
    payload = PAYLOAD if ".csv" in name else b"".join(
        (json.dumps({"score": i, "lvl": ["a", "b"][i % 2]}) + "\n").encode() for i in range(30))
    member = "exports/Private Zelmira Quenby Export.csv"
    (_gz if name.endswith(".gz") else _zip)(src / name, member, payload)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output  # the self-check expects the output's own name
    inner = name.rsplit(".", 1)[0]
    if name.endswith(".gz"):
        info = compress._gzip_header((out / name).read_bytes())
        assert info["fname"] == inner
    else:
        with zipfile.ZipFile(out / name) as zf:
            assert zf.namelist() == [inner]
    _assert_no_markers(out, rep)
    assert "container_name_replaced" in _codes(rep, name)
    # same seed, same bytes (D26)
    res2, out2, _ = run_dir(src, tmp_path / "again")
    assert res2.exit_code == 0 and (out2 / name).read_bytes() == (out / name).read_bytes()


def test_container_name_equal_to_the_output_name_has_no_warning(tmp_path: Path) -> None:
    p = tmp_path / "students.csv.zip"
    _zip(p, "students.csv")
    rr = read(p, "csv", compression="zip")
    assert "container_name_replaced" not in [m.code for m in rr.messages]
    g = tmp_path / "plain.csv.gz"
    g.write_bytes(gzip.compress(PAYLOAD, mtime=0))  # no stored name: none is written either
    assert compress.container_messages(read(g, "csv", compression="gz").layout.extra) == []
    out = tmp_path / "plain.csv.gz.datanonymous-tmp"
    compress.write_bytes(out, PAYLOAD, "gz", read(g, "csv", compression="gz").layout)
    assert compress._gzip_header(out.read_bytes())["fname"] is None


def test_container_self_check_flags_a_copied_name(tmp_path: Path) -> None:
    s, o = tmp_path / "s.csv.gz", tmp_path / "s.csv.gz.datanonymous-tmp"
    _gz(s, "Zelmira.csv")
    _gz(o, "Zelmira.csv")  # as if the writer had copied the source's stored name
    ls = compress.container_layout(s, "gz")
    lo = compress.container_layout(o, "gz")
    issues = compress.container_diff(FileLayout("csv", extra=ls), FileLayout("csv", extra=lo))
    assert any("stored file name differs" in i for i in issues)
    _gz(o, "s.csv")
    lo = compress.container_layout(o, "gz")
    assert compress.container_diff(FileLayout("csv", extra=ls), FileLayout("csv", extra=lo)) == []


# =========================================================================== R1-14 wide decimals
def test_wide_decimal_near_the_type_limit_is_written_with_exact_extremes(tmp_path: Path) -> None:
    """The review's reproduction: decimal128(38, 6) values at the limit FAILed (float rounding to 1e32)."""
    lo, hi = Decimal("99999999999999999999999999999999.999998"), Decimal("99999999999999999999999999999999.999999")
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(pa.table({"amount": pa.array([lo, hi] * 20, pa.decimal128(38, 6))}), src / "w.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    t = pq.read_table(out / "w.parquet")
    assert t.schema.field("amount").type == pa.decimal128(38, 6) and t.num_rows == 40
    assert Counter(t.column("amount").to_pylist()) == Counter({lo: 20, hi: 20})


@pytest.mark.parametrize("typ,scale", [(pa.decimal128(38, 18), 18), (pa.decimal256(76, 10), 10)])
def test_wide_decimal_spread_keeps_exact_bounds_and_fits(tmp_path: Path, typ: pa.DataType, scale: int) -> None:
    rng = np.random.default_rng(3)
    q = Decimal(1).scaleb(-scale)
    base = Decimal("123456789012345678.5") if scale == 10 else Decimal("123.45")
    vals = [(base + Decimal(int(x)) * q * 7919).quantize(q) for x in rng.integers(-10**6, 10**6, 300)]
    lo, hi = min(vals), max(vals)
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(pa.table({"amount": pa.array(vals, typ)}), src / "s.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    got = pq.read_table(out / "s.parquet").column("amount").to_pylist()
    assert len(got) == 300 and min(got) == lo and max(got) == hi  # exact D14 extremes, never outside
    assert all(lo <= v <= hi for v in got)
    assert len(set(got) - set(vals)) > 100  # interpolated, not resampled
    res2, out2, _ = run_dir(src, tmp_path / "again")
    assert (out2 / "s.parquet").read_bytes() == (out / "s.parquet").read_bytes()  # D26


def _dec_profile(values: list[str], scale: int) -> ColumnProfile:
    p = ColumnProfile(index=0, name="amount", type=ColumnType.DECIMAL,
                      params={"precision": 38, "scale": scale, "decimals": scale})
    p.model = ColumnModel(nonmissing_counts=dict(Counter(values)),
                          sorted_numeric=np.sort(np.array([float(v) for v in values])))
    return p


def test_wide_decimal_generation_unit() -> None:
    vals = [f"{12345678901234567 + 3 * i}.{i % 100:02d}" for i in range(50)]
    p = _dec_profile(vals, 2)
    syn, flags = generate_values(p, p.model, 50, np.random.default_rng(0))
    got = [Decimal(v) for v in syn.tolist()]
    assert min(got) == Decimal(min(vals, key=Decimal)) and max(got) == Decimal(max(vals, key=Decimal))
    assert all(len(v.split(".")[1]) == 2 for v in syn.tolist())
    again, _ = generate_values(p, p.model, 50, np.random.default_rng(0))
    assert again.tolist() == syn.tolist()
    # a narrow column keeps the float generator (unchanged behaviour)
    narrow = _dec_profile([f"{i}.25" for i in range(50)], 2)
    y, _ = G.generate_numeric_extended(narrow, narrow.model, 10, np.random.default_rng(0))
    assert all(isinstance(v, float) for v in y.tolist())


def test_wide_decimal_entity_model_floors_row_extremes() -> None:
    """D4: a within-ID-constant decimal keeps the row-level exact extremes lost by the per-entity model."""
    vals = [f"{10**20 + i}.5" for i in range(40)]
    row = _dec_profile(vals, 1)
    entity = ColumnModel(nonmissing_counts=dict(Counter(vals[5:30])))
    out, flags = G._generate_decimal_exact(row, entity, 12, np.random.default_rng(1), 1, True)
    assert Decimal(vals[0]) in out.tolist() and Decimal(vals[-1]) in out.tolist()
    assert flags["extreme_value_adjusted"]


def test_decimal_writer_quantizes_beyond_28_digits() -> None:
    field = pa.field("m", pa.decimal128(38, 6))
    arr = arrowio.values_to_arrow(np.array(["99999999999999999999999999999999.999999", None], dtype=object),
                                  field, {}, None)
    assert arr.to_pylist() == [Decimal("99999999999999999999999999999999.999999"), None]


# =========================================================================== R1-16 strict JSON statistics
def test_large_finite_numbers_have_finite_statistics() -> None:
    x = np.array([i * 1e180 for i in range(1, 61)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no overflow on the way
        st = numeric_summary(x)
    assert st["sd"] == pytest.approx(statistics.stdev([i * 1.0 for i in range(1, 61)]) * 1e180, rel=1e-9)
    assert st["mean"] == pytest.approx(30.5e180, rel=1e-12)
    wide = numeric_summary(np.array([-1.7e308, 1.7e308, 0.5, 1.25] * 10))
    assert all(isinstance(v, float) and math.isfinite(v) for v in (wide["mean"], wide["sd"], wide["median"]))
    assert all(math.isfinite(v) for v in wide["quantiles"].values())
    # unchanged for ordinary numbers
    small = np.array([1.5, 2.5, 4.0, 10.0])
    assert numeric_summary(small)["sd"] == round(float(small.std(ddof=1)), 6)
    # non-finite input never yields a non-finite statistic: 'unavailable' (None)
    st = numeric_summary(np.array([1.0, np.inf]))
    assert st["max"] is None and st["mean"] is None


def test_report_json_is_strict_for_huge_floats(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "huge.csv").write_text("amount\n" + "".join(f"{i * 1e180:.2e}\n" for i in range(1, 61)), encoding="utf-8")
    res, out, rep = run_dir(src, tmp_path, "--json")
    assert res.exit_code == 0, res.output
    for p in list(rep.rglob("*.json")) + list(out.rglob("*.json")):
        _strict(p.read_text(encoding="utf-8"))
    col = _strict((rep / "files" / "huge.csv.json").read_text(encoding="utf-8"))["tables"][0]["columns"][0]
    assert math.isfinite(col["real"]["sd"]) and math.isfinite(col["synthetic"]["sd"])
    stdout = getattr(res, "stdout", res.output)
    _strict(stdout[stdout.index("{"):])


def test_report_serialisation_never_emits_non_finite_tokens() -> None:
    obj = {"a": float("inf"), "b": [float("nan"), 1.5, (float("-inf"), 2)], "c": {"d": np.float64("nan")}}
    assert finite_json(obj) == {"a": None, "b": [None, 1.5, [None, 2]], "c": {"d": None}}
    assert _strict(_json(obj)) == {"a": None, "b": [None, 1.5, [None, 2]], "c": {"d": None}}
