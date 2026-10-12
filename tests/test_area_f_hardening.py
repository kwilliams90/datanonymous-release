"""AREA F hardening: adversarial inputs and regression tests for issues found after the build.

Each test names the defect it pins down. Privacy tests assert that no real value (names in
value-label tables or Arrow dictionaries) reaches the out dir or the report.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from datanonymous.config import Settings
from datanonymous.errors import ReadError, WriteError
from datanonymous.io import arrowio, csvio, jsonio, registry, statio
from datanonymous.io.registry import OutTable, WriteRequest
from datanonymous.model import ColumnProfile, ColumnType, FileLayout, TableProfile
from test_area_f_util import FIX, all_bytes, identity_write, read, report, run_dir

NAMES = ["Zebulon Quirk", "Ophelia Marchetti", "Thaddeus Brightwater", "Wilhelmina Oduya"]


def _src(tmp_path: Path) -> Path:
    d = tmp_path / "in"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _leaks(*dirs: Path, values=NAMES) -> list[str]:
    blob = all_bytes(*dirs)
    return [v for v in values if v.encode("utf-8") in blob or v.encode("latin-1") in blob]


def _stata_meta(path: Path) -> dict:
    with pd.io.stata.StataReader(path) as r:
        df = r.read(convert_categoricals=False, convert_dates=False)
        return {"vl": r.value_labels(), "typ": r._typlist, "lbl": r._lbllist, "df": df}


# --------------------------------------------------------------------------- --null never copies real labels
def _dict_names_table(n: int = 40) -> pa.Table:
    return pa.table({"rid": [f"r{i}" for i in range(n)],
                     "first_name": pa.array([NAMES[i % 4] for i in range(n)]).dictionary_encode(),
                     "score": [float(i % 9) for i in range(n)]})


@pytest.mark.parametrize("name", ["names.parquet", "names.feather"])
def test_null_dictionary_column_never_copies_its_dictionary(tmp_path: Path, name: str) -> None:
    """Regression: --null on an Arrow dictionary column wrote the SOURCE dictionary (every real
    label, e.g. names) into the output with all-null indices. It must keep only the type."""
    src = _src(tmp_path)
    t = _dict_names_table()
    if name.endswith(".parquet"):
        pq.write_table(t, src / name)
    else:
        feather.write_feather(t, src / name)
    res, out, rep = run_dir(src, tmp_path, "--null", "first_name")
    assert res.exit_code == 0, res.output
    dst = (pq.read_table if name.endswith(".parquet") else feather.read_table)(out / name)
    col = dst.column("first_name").combine_chunks()
    assert col.type == t.schema.field("first_name").type  # same Arrow type (dictionary<string, int32>)
    assert col.null_count == dst.num_rows and len(col.dictionary) == 0
    assert _leaks(out, rep) == []


def test_null_dictionary_unit_keeps_type_and_order_flag() -> None:
    src = pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], pa.int8()), pa.array(["Real A", "Real B"]),
                                         ordered=True)
    field = pa.field("g", src.type)
    p = ColumnProfile(index=0, name="g", type=ColumnType.EMPTY, role="null")
    native = {"kind": "dictionary", "dictionary": src.dictionary, "dictionary_values": ["Real A", "Real B"]}
    out = arrowio.values_to_arrow(np.array([None, None, None], dtype=object), field, native, p)
    assert out.type == src.type and out.type.ordered and out.null_count == 3 and len(out.dictionary) == 0


def _labelled_names_dta(path: Path, n: int = 40, version: int = 118, name: str = "teacher_name") -> None:
    df = pd.DataFrame({"rid": [f"r{i}" for i in range(n)],
                       name: np.array([i % 4 + 1 for i in range(n)], dtype=np.int8),
                       "score": np.arange(n, dtype=np.float64) % 7})
    df.to_stata(path, write_index=False, version=version,
                value_labels={name: {k + 1: v for k, v in enumerate(NAMES)}},
                variable_labels={name: "Homeroom teacher"})


def test_null_labelled_stata_column_drops_real_labels(tmp_path: Path) -> None:
    """Regression: --null on a labelled Stata column wrote its real value-label table (names)."""
    src = _src(tmp_path)
    _labelled_names_dta(src / "t.dta")
    res, out, rep = run_dir(src, tmp_path, "--null", "teacher_name")
    assert res.exit_code == 0, res.output
    m = _stata_meta(out / "t.dta")
    assert "teacher_name" not in m["vl"] and m["typ"][1] == "b"  # storage type kept, no label table
    assert m["df"]["teacher_name"].isna().all()
    assert _leaks(out, rep) == []
    assert not any(w["code"] == "layout_mismatch" for w in report(rep)["warnings"])


# --------------------------------------------------------------------------- labelled columns that are not label-safe
def test_labelled_pii_column_is_relabelled_not_failed(tmp_path: Path) -> None:
    """Regression: a labelled Stata column with a PII header (D12 -> invented labels) FAILed with a
    misleading 'remove the type override' fix. It is now written as a labelled int column whose
    value labels are the synthetic labels on new codes; no real label is written."""
    src = _src(tmp_path)
    _labelled_names_dta(src / "t.dta")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    m = _stata_meta(out / "t.dta")
    labels = m["vl"]["teacher_name"]
    assert m["typ"][1] == "b" and m["lbl"][1] == "teacher_name"
    assert sorted(labels) == [1, 2, 3, 4] and not set(labels.values()) & set(NAMES)
    codes = m["df"]["teacher_name"]
    assert sorted(codes.value_counts().tolist()) == [10, 10, 10, 10]  # distinct count and frequencies kept
    assert set(codes) <= set(labels)
    assert _leaks(out, rep) == []
    assert pd.read_stata(out / "t.dta")["teacher_name"].dtype.name == "category"  # still a labelled column


def test_labelled_column_force_text_relabelled_same_bytes(tmp_path: Path) -> None:
    r1, o1, rep1 = run_dir(FIX / "labels.dta", tmp_path / "a", "--null", "dob", "--force-text", "grade", seed="3")
    r2, o2, _ = run_dir(FIX / "labels.dta", tmp_path / "b", "--null", "dob", "--force-text", "grade", seed="3")
    assert r1.exit_code == r2.exit_code == 0, r1.output
    assert (o1 / "labels.dta").read_bytes() == (o2 / "labels.dta").read_bytes()
    m = _stata_meta(o1 / "labels.dta")
    assert not set(m["vl"]["grade"].values()) & {"KG", "First", "Second", "Third"}
    assert m["vl"]["ecodis"] == {0: "No", 1: "Yes"}  # label-safe columns keep their real labels


def _write_dta_unit(tmp_path: Path, values: list, profile_type: ColumnType, role: str = "row",
                    native_typ: str = "b") -> dict:
    """write_dta for one labelled column with a hand-built profile (no generation)."""
    src = tmp_path / "src.dta"
    pd.DataFrame({"g": np.array([1, 2, 1], dtype=np.int8)}).to_stata(
        src, write_index=False, version=118, value_labels={"g": {1: "Real One", 2: "Real Two"}})
    rr = read(src, "dta")
    col = rr.tables[0].columns[0]
    native = dict(col.native)
    native["typ"] = native_typ
    p = ColumnProfile(index=0, name="g", type=profile_type, role=role)
    p.native = native
    tp = TableProfile(None, ["g"], len(values), [p])
    ot = OutTable(None, ["g"], [np.array(values, dtype=object)], [0], len(values), {}, tp)
    out = tmp_path / "out.dta"
    registry.get("dta").write(WriteRequest(out, rr.layout, [ot], Settings(mode="fidelity"), seed=1))
    rr2 = read(out, "dta", purpose="selfcheck")
    return {"meta": _stata_meta(out), "diff": statio.layout_diff(rr.layout, rr2.layout, [tp.layout], rr2.tables),
            "policy": tp.layout.get(statio.LABEL_POLICY_KEY)}


def test_non_label_numbers_are_written_without_the_real_labels(tmp_path: Path) -> None:
    r = _write_dta_unit(tmp_path, ["7", "9", None], ColumnType.IDENTIFIER)
    assert r["meta"]["df"]["g"].tolist()[:2] == [7, 9] and "g" not in r["meta"]["vl"]
    assert r["policy"] == {"g": "dropped"} and r["diff"] == []


def test_relabelled_codes_extend_past_the_source_codes(tmp_path: Path) -> None:
    r = _write_dta_unit(tmp_path, ["g_3", "g_1", "g_2", None], ColumnType.HIGH_CARD_CATEGORICAL)
    assert r["meta"]["vl"]["g"] == {1: "g_1", 2: "g_2", 3: "g_3"}
    assert r["meta"]["df"]["g"].tolist()[:3] == [3, 1, 2] and r["diff"] == []


def test_layout_diff_rejects_real_labels_on_relabelled_columns(tmp_path: Path) -> None:
    r = _write_dta_unit(tmp_path, ["Real One", "x"], ColumnType.HIGH_CARD_CATEGORICAL)
    assert any("source value label" in i for i in r["diff"])  # the self-check catches a real label


def test_label_safe_columns_keep_source_labels(tmp_path: Path) -> None:
    r = _write_dta_unit(tmp_path, ["Real Two", "Real One", None], ColumnType.FACTOR)
    assert r["meta"]["vl"]["g"] == {1: "Real One", 2: "Real Two"} and r["policy"] == {} and r["diff"] == []


# --------------------------------------------------------------------------- storage-type overflow
def test_stata_byte_id_overflow_fails_cleanly(tmp_path: Path) -> None:
    """Regression: synthetic IDs too wide for a byte column raised an internal TypeError (pandas
    Int8). The file now FAILs with a clear reason and an --exclude/--null fix (D9)."""
    src = _src(tmp_path)
    rng = np.random.default_rng(5)
    pd.DataFrame({"stuid": np.array(rng.permutation(np.arange(1, 101))[:80], dtype=np.int8),
                  "v": np.arange(80, dtype=np.int8) % 3}).to_stata(src / "ids.dta", write_index=False, version=118)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 3
    f = report(rep)["failures"][0]
    assert "internal error" not in f["reasons"][0]
    assert "'stuid'" in f["reasons"][0] and "Stata storage type" in f["reasons"][0]
    assert f["fixes"][0].startswith("--exclude stuid or --null stuid")


@pytest.mark.parametrize("typ,bad", [("b", 101), ("b", -128), ("h", 32741), ("l", 2147483621)])
def test_stata_integer_ranges_exclude_missing_codes(typ: str, bad: int) -> None:
    """Values above Stata's non-missing maximum (byte 100, int 32740, long 2147483620) are its
    missing-value codes: pandas would silently widen the storage type."""
    lo, hi = statio.STATA_INT_RANGES[typ]
    assert statio._stored_numbers("c", [float(lo), float(hi)], typ, None).tolist() == [lo, hi]
    with pytest.raises(WriteError) as ei:
        statio._stored_numbers("c", [float(bad)], typ, None)
    assert str(bad) not in ei.value.message and "--null c" in ei.value.fix


def test_stata_non_finite_values_fail_cleanly() -> None:
    with pytest.raises(WriteError):
        statio._stored_numbers("c", [float("nan")], "l", None)


def test_arrow_id_overflow_names_the_real_cause(tmp_path: Path) -> None:
    src = _src(tmp_path)
    rng = np.random.default_rng(5)
    ids = rng.permutation(np.arange(1, 121))[:90]
    pq.write_table(pa.table({"student_id": pa.array(ids.tolist(), pa.int8()),
                             "v": [int(x) for x in rng.integers(0, 3, 90)]}), src / "ids.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 3 and not (out / "ids.parquet").exists()
    f = report(rep)["failures"][0]
    assert "IDENTIFIER" in f["reasons"][0] and "int8" in f["reasons"][0]
    assert "type override" not in f["fixes"][0] and "--exclude student_id" in f["fixes"][0]


def test_writer_fix_mentions_force_type_only_when_used() -> None:
    field = pa.field("c", pa.int32())
    forced = ColumnProfile(index=0, name="c", type=ColumnType.FREE_TEXT, source="force-type")
    plain = ColumnProfile(index=0, name="c", type=ColumnType.HIGH_CARD_CATEGORICAL)
    for p, expect in ((forced, "remove --force-type for 'c'"), (plain, "--null c or --exclude c")):
        with pytest.raises(WriteError) as ei:
            arrowio.values_to_arrow(np.array(["c_1"], dtype=object), field, {"kind": "int"}, p)
        assert ei.value.fix.startswith(expect) and p.type.value in ei.value.message


# --------------------------------------------------------------------------- legacy INT96 timestamps
def _int96_file(path: Path) -> pa.Table:
    t = pa.table({"t": pa.array([i * 86400 * 10**9 + 123_000 for i in range(20)], pa.int64()).cast(
        pa.timestamp("ns")), "k": list(range(20))})
    pq.write_table(t, path, use_deprecated_int96_timestamps=True, store_schema=False, version="1.0")
    return t


def test_int96_timestamps_are_mirrored(tmp_path: Path) -> None:
    """Regression: Spark/Hive-style INT96 timestamps were written as INT64 microseconds, so the
    Arrow type drifted timestamp[ns] -> timestamp[us] (self-check FAIL, or a writer error)."""
    src = tmp_path / "spark.parquet"
    t = _int96_file(src)
    rr = read(src, "parquet")
    assert rr.layout.extra["int96_timestamps"] is True
    out = identity_write(rr, "parquet", tmp_path / "out.parquet")
    assert pq.ParquetFile(out).schema.column(0).physical_type == "INT96"
    assert pq.read_table(out).equals(pq.read_table(src)) and pq.read_table(out).schema.field("t").type == t.schema[0].type
    rr2 = read(out, "parquet", purpose="selfcheck")
    assert arrowio.layout_diff(rr.layout, rr2.layout, [{}], rr2.tables) == []
    plain = dict(rr2.layout.extra, int96_timestamps=False)
    assert any("INT96" in i for i in arrowio.layout_diff(rr.layout, FileLayout("parquet", extra=plain), [], []))


def test_int96_with_simulated_dates_end_to_end(monkeypatch, tmp_path: Path) -> None:
    from test_area_f_arrow import _simulate_areas_a_b

    _simulate_areas_a_b(monkeypatch)
    src = _src(tmp_path)
    _int96_file(src / "spark.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert pq.ParquetFile(out / "spark.parquet").schema.column(0).physical_type == "INT96"


# --------------------------------------------------------------------------- compressed CSV self-check
def test_csv_layout_diff_compares_the_container(tmp_path: Path) -> None:
    """The CSV self-check now compares the container (zip member name, gzip header) too.

    R1-01: the member name must be the output file's own name (the source's is never copied)."""
    payload = b"a,b\r\n1,x\r\n"
    s, o, good = tmp_path / "s.csv.zip", tmp_path / "o.csv.zip", tmp_path / "g.csv.zip"
    for p, member in ((s, "export.csv"), (o, "other.csv"), (good, "g.csv")):
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(zipfile.ZipInfo(member, date_time=(2020, 1, 1, 0, 0, 0)), payload)
    ls = read(s, "csv", compression="zip").layout
    lo = read(o, "csv", compression="zip", purpose="selfcheck").layout
    assert any("name differs" in i for i in csvio.layout_diff(ls, lo))
    assert csvio.layout_diff(ls, read(good, "csv", compression="zip", purpose="selfcheck").layout) == []


# --------------------------------------------------------------------------- JSON lines
def test_jsonl_mixed_line_endings_read_with_warning(tmp_path: Path) -> None:
    """Regression: a JSONL file mixing CRLF and LF FAILed with 'line 2 holds more than one JSON value'."""
    p = tmp_path / "m.jsonl"
    p.write_bytes(b'{"a": 1}\r\n{"a": 2}\n{"a": 3}\r\n')
    rr = read(p, "jsonl")
    assert rr.tables[0].columns[0].values.tolist() == ["1", "2", "3"]
    assert [m.code for m in rr.messages] == ["line_endings_mixed", "json_format_approximate"]
    assert rr.layout.extra["line_ending"] == "CRLF"
    out = identity_write(rr, "jsonl", tmp_path / "o.jsonl")
    assert out.read_bytes() == b'{"a": 1}\r\n{"a": 2}\r\n{"a": 3}\r\n'


@pytest.mark.parametrize("text,expect", [("a\r\nb\nc\n", ("\n", True)), ("a\r\nb\r\n", ("\r\n", False)),
                                         ("a\nb\r\n", ("\n", True)), ("a\r\nb\n", ("\r\n", True)),
                                         ("one line", ("\n", False))])
def test_line_ending_majority(text: str, expect: tuple) -> None:
    assert jsonio.line_ending(text) == expect


def test_jsonl_record_spanning_lines_has_a_clear_error(tmp_path: Path) -> None:
    p = tmp_path / "pretty.jsonl"
    p.write_text('{"a": 1}\n{\n  "a": 2\n}\n', encoding="utf-8")
    with pytest.raises(ReadError) as ei:
        read(p, "jsonl")
    assert "spans several lines" in ei.value.message and "one record per line" in ei.value.fix


# --------------------------------------------------------------------------- Stata labels
def test_blank_value_label_keeps_codes(tmp_path: Path) -> None:
    """A blank label would read as a missing value (its code lost on writing): codes are shown."""
    src = tmp_path / "blank.dta"
    pd.DataFrame({"g": np.array([1, 2, 9, 1], dtype=np.int8)}).to_stata(
        src, write_index=False, version=118, value_labels={"g": {1: "Yes", 2: "No", 9: " "}})
    rr = read(src, "dta")
    col = rr.tables[0].columns[0]
    assert col.values.tolist() == ["1", "2", "9", "1"] and col.native["labels_rendered"] is False
    out = identity_write(rr, "dta", tmp_path / "o.dta")
    assert _stata_meta(out)["df"]["g"].tolist() == [1, 2, 9, 1]


# --------------------------------------------------------------------------- NDJSON saved as .json
def test_json_lines_content_in_a_json_file_is_mirrored(tmp_path: Path) -> None:
    """Regression: NDJSON exports named .json were skipped as 'not a table'. They are read and
    written back as JSON lines under the same name; a lone object stays 'not a table'."""
    src = _src(tmp_path)
    text = "\n".join(json.dumps({"id": 100 + i, "lvl": ["a", "b"][i % 2], "x": i * 1.5}) for i in range(30)) + "\n"
    (src / "nd.json").write_text(text, encoding="utf-8")
    (src / "one.json").write_text('{"a": 1}\n', encoding="utf-8")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    data = report(rep)
    assert [s["path"] for s in data["skipped"]] == ["one.json"]
    lines = (out / "nd.json").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30 and all(list(json.loads(x)) == ["id", "lvl", "x"] for x in lines)
    rr = read(src / "nd.json", "json")
    assert rr.layout.extra["json_layout"] == "lines" and rr.layout.format == "json"
    assert "json_lines_content" in [m.code for m in rr.messages]
    assert jsonio.inferred_dtypes(src / "nd.json", rr.layout, None) == ["int64", "str", "float64"]


# --------------------------------------------------------------------------- privacy sweep over every format
def test_metadata_and_labels_never_reach_the_report(tmp_path: Path) -> None:
    """Free-text metadata (Arrow schema/field metadata, Stata data/variable labels, gzip/zip member
    names) reaches neither the out dir (R1-01: it is never copied) nor the report dir; labels of
    non-label columns (PII-hinted dictionary / value-label columns) reach neither."""
    import gzip

    src = _src(tmp_path)
    n = 30
    t = pa.table({"rid": [f"r{i}" for i in range(n)],
                  "first_name": pa.array([f"MARKERDICT{i % 3}" for i in range(n)]).dictionary_encode(),
                  "lvl": pa.array([["lo", "hi"][i % 2] for i in range(n)]).dictionary_encode(),
                  "v": pa.array([i % 5 for i in range(n)], pa.int32())})
    t = t.replace_schema_metadata({"note": "MARKERSCHEMA"})
    t = t.cast(pa.schema([f.with_metadata({"d": "MARKERFIELD"}) if f.name == "v" else f for f in t.schema],
                         metadata=t.schema.metadata))
    pq.write_table(t, src / "a.parquet")
    feather.write_feather(t, src / "a.feather")
    pd.DataFrame({"rid": [f"r{i}" for i in range(n)], "counselor_name": np.array([i % 3 for i in range(n)],
                                                                                    dtype=np.int8),
                  "v": np.arange(n, dtype=np.int16) % 4}).to_stata(
        src / "a.dta", write_index=False, version=118, data_label="MARKERDATALABEL",
        variable_labels={"v": "MARKERVARLABEL"},
        value_labels={"counselor_name": {0: "MARKERVL0", 1: "MARKERVL1", 2: "MARKERVL2"}})
    payload = b"rid,v\r\n" + b"".join(f"r{i},{i % 4}\r\n".encode() for i in range(n))
    buf = io.BytesIO()
    with gzip.GzipFile(filename="MARKERGZNAME.csv", mode="wb", fileobj=buf, mtime=1) as g:
        g.write(payload)
    (src / "b.csv.gz").write_bytes(buf.getvalue())
    with zipfile.ZipFile(src / "c.csv.zip", "w") as zf:
        zf.writestr("MARKERZIPMEMBER.csv", payload)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    rep_blob = all_bytes(rep)
    out_blob = all_bytes(out)
    for marker in ("MARKERSCHEMA", "MARKERFIELD", "MARKERDATALABEL", "MARKERVARLABEL", "MARKERGZNAME",
                   "MARKERZIPMEMBER", "MARKERDICT", "MARKERVL"):
        assert marker.encode() not in rep_blob, marker
    for marker in ("MARKERDICT", "MARKERVL"):  # labels of PII-hinted columns are never copied
        assert marker.encode() not in out_blob, marker
    for marker in ("MARKERSCHEMA", "MARKERFIELD", "MARKERDATALABEL", "MARKERVARLABEL", "MARKERGZNAME",
                   "MARKERZIPMEMBER"):  # free-form format metadata is never copied either (R1-01)
        assert marker.encode() not in out_blob, marker


# --------------------------------------------------------------------------- JSON number columns never get strings
def test_text_in_a_json_number_column_fails_before_writing(tmp_path: Path) -> None:
    p = tmp_path / "n.json"
    p.write_text(json.dumps([{"n": 1.5, "b": True}, {"n": 2, "b": False}]), encoding="utf-8")
    rr = read(p, "json")
    for cols, bad in (([np.array(["n_1", "3"], dtype=object), np.array(["true", "false"], dtype=object)], "'n'"),
                      ([np.array(["1", "3"], dtype=object), np.array(["yes", None], dtype=object)], "'b'")):
        with pytest.raises(WriteError) as ei:
            identity_write(rr, "json", tmp_path / "o.json", columns=cols)
        assert bad in ei.value.message and "n_1" not in ei.value.message and "yes" not in ei.value.message
        assert "--null" in ei.value.fix


def test_exponent_numbers_generated_or_failed_clearly(tmp_path: Path) -> None:
    """json.dumps writes floats below 1e-4 as 9.43e-05. Without AREA B's scientific-notation floats
    the column is text to the value rules: the file must FAIL before writing with --null/--exclude,
    never write JSON strings into a number column (which the self-check then rejected with a
    misleading --no-strict fix). With AREA B merged the numbers are written as numbers."""
    src = _src(tmp_path)
    rng = np.random.default_rng(4)
    recs = [{"p": float(x), "q": int(rng.integers(0, 3))} for x in rng.uniform(0, 1e-4, 40)]
    (src / "sci.json").write_text(json.dumps(recs), encoding="utf-8")
    res, out, rep = run_dir(src, tmp_path)
    if res.exit_code == 0:
        fake = json.loads((out / "sci.json").read_text(encoding="utf-8"))
        assert all(isinstance(r["p"], float) for r in fake)
        return
    assert res.exit_code == 3 and not (out / "sci.json").exists()
    f = report(rep)["failures"][0]
    assert "cannot be written as JSON number values" in f["reasons"][0]
    assert f["fixes"] == ["--null p or --exclude p"]


# --------------------------------------------------------------------------- unused labels are reported (counts only)
def test_unused_dictionary_levels_are_reported(tmp_path: Path) -> None:
    from test_area_f_arrow import _ctx

    rr = read(FIX / "types.parquet", "parquet")
    cols = {c.name: c for c in rr.tables[0].columns}
    for name in ("grade", "school"):
        det = arrowio.detect_native(_ctx(cols[name].values.tolist(), cols[name].native, name))
        assert det.params["unused_levels"] == 1
        if name == "grade":  # the unused grade level "06" is tool vocabulary: kept, nothing to report (R2-03)
            assert det.messages == [] and det.params["levels"][-1] == "06"
            continue
        (msg,) = det.messages  # the unused school is never written (R2-03)
        assert msg.code == "dictionary_unused_levels" and msg.level == "info" and "Unused" not in msg.text
        assert "not written" in msg.text and msg.fix is None


def test_force_text_rebuilds_a_dictionary_with_unused_levels(tmp_path: Path) -> None:
    from test_area_f_arrow import NULL_AB

    res, out, rep = run_dir(FIX / "types.parquet", tmp_path, *NULL_AB, "--force-text", "school")
    assert res.exit_code == 0, res.output
    d = pq.read_table(out / "types.parquet").column("school").combine_chunks().dictionary.to_pylist()
    assert not {"Adams MS", "Lincoln ES", "Unused HS"} & set(d)
    assert b"Unused HS" not in all_bytes(out, rep)


def test_unused_stata_value_labels_are_reported(tmp_path: Path) -> None:
    src = tmp_path / "u.dta"
    pd.DataFrame({"g": np.array([1, 2, 1], dtype=np.int8)}).to_stata(
        src, write_index=False, version=118, value_labels={"g": {1: "A", 2: "B", 3: "Never used", 4: "Nor this"}})
    rr = read(src, "dta")
    col = rr.tables[0].columns[0]
    from datanonymous.detect.context import DetectContext

    s = Settings(mode="fidelity")
    ctx = DetectContext(index=0, name="g", values=col.values, file_rel="u.dta", table=None, format="dta",
                        options=s.column_options("u.dta", None, "g"), settings=s, native=col.native)
    det = statio.detect_native(ctx)
    assert det.params["unused_labels"] == 2
    assert [m.code for m in det.messages] == ["value_labels_unused"] and "Never" not in det.messages[0].text


def test_all_missing_labelled_stata_column_is_written(tmp_path: Path) -> None:
    """Regression: a labelled column with no values (EMPTY) wrote no label table, and the self-check
    then FAILed the file ('value labels differ'). Its labels are not known to be safe: no table."""
    src = _src(tmp_path)
    pd.DataFrame({"k": np.arange(20, dtype=np.int8) % 3, "g": pd.array([None] * 20, dtype="Int8")}).to_stata(
        src / "t.dta", write_index=False, version=118, value_labels={"g": {0: "No", 1: "Yes"}})
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    m = _stata_meta(out / "t.dta")
    assert "g" not in m["vl"] and m["df"]["g"].isna().all() and m["typ"][1] == "b"


def test_concatenated_pretty_objects_in_json_fail_clearly(tmp_path: Path) -> None:
    p = tmp_path / "concat.json"
    p.write_text('{\n  "a": 1\n}\n{\n  "a": 2\n}\n', encoding="utf-8")
    with pytest.raises(ReadError) as ei:
        read(p, "json")
    assert "spans several lines" in ei.value.message and "JSON array of records" in ei.value.fix


# --------------------------------------------------------------------------- SAS dates -> Parquet
def test_sas_dates_are_stored_in_a_parquet_unit(monkeypatch, tmp_path: Path) -> None:
    """Regression: pandas reads SAS dates as datetime64[s]; Parquet has no seconds unit, so the
    output came back as timestamp[ms] and the self-check saw schema drift. SAS input is cast to
    the Parquet-storable unit when read (stubbed reader: SAS files cannot be created here)."""
    from test_area_f_arrow import _simulate_areas_a_b

    _simulate_areas_a_b(monkeypatch)
    frame = pd.DataFrame({"VISIT": pd.Series(pd.to_datetime(["2020-01-01", "2021-05-06", None] * 10)).astype(
        "datetime64[s]"), "N": np.arange(30, dtype=np.float64) % 4})
    monkeypatch.setattr(statio, "read_sas_frame", lambda path, fmt, encoding: frame.copy())
    src = _src(tmp_path)
    (src / "visits.sas7bdat").write_bytes(b"stub")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    t = pq.read_table(out / "visits.parquet")
    assert t.schema.field("VISIT").type == pa.timestamp("ms") and t.column("VISIT").null_count == 10
    assert not any(w["code"] == "layout_mismatch" for w in report(rep)["warnings"])


def test_parquet_storable_keeps_other_types() -> None:
    t = pa.table({"a": pa.array([1], pa.int8()), "s": pa.array([0], pa.int64()).cast(pa.timestamp("s", tz="UTC")),
                  "t": pa.array([1], pa.int32()).cast(pa.time32("s"))})
    out = statio.parquet_storable(t)
    assert [str(f.type) for f in out.schema] == ["int8", "timestamp[ms, tz=UTC]", "time32[ms]"]
    assert out.column("s").cast(pa.int64()).to_pylist() == [0] and out.column("t").cast(pa.int32()).to_pylist() == [
        1000]


def test_ordered_dictionary_under_a_pii_header_is_not_copied(tmp_path: Path) -> None:
    """Regression: an ORDERED dictionary column of names (R ordered factor / pandas ordered
    categorical) became ORDINAL, which the D12 PII-header guard never rewrites: every real name
    was written to the output dictionary and listed as ORDINAL levels in the report."""
    src = _src(tmp_path)
    n = 40
    col = pa.DictionaryArray.from_arrays(pa.array([i % 4 for i in range(n)], pa.int8()), pa.array(NAMES),
                                         ordered=True)
    pq.write_table(pa.table({"rid": [f"r{i}" for i in range(n)], "first_name": col,
                             "band": pa.DictionaryArray.from_arrays(pa.array([i % 3 for i in range(n)], pa.int8()),
                                                                    pa.array(["low", "mid", "high"]), ordered=True)}),
                   src / "o.parquet")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    dst = pq.read_table(out / "o.parquet")
    assert dst.schema.equals(pq.read_table(src / "o.parquet").schema)
    assert not set(dst.column("first_name").combine_chunks().dictionary.to_pylist()) & set(NAMES)
    assert dst.column("band").combine_chunks().dictionary.to_pylist() == ["low", "mid", "high"]  # still ORDINAL
    assert _leaks(out, rep) == []
