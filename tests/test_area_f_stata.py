"""AREA F: Stata .dta reading/writing (labels, formats, versions) and SAS input (stubbed reader)."""

from __future__ import annotations

import datetime as dt
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from datanonymous.config import Settings
from datanonymous.detect.context import DetectContext
from datanonymous.errors import ReadError, UnsupportedColumnError
from datanonymous.io import statio
from datanonymous.model import ColumnType
from test_area_f_util import FIX, all_bytes, identity_write, read, report, run_dir, text_of


def _meta(path: Path) -> dict:
    with pd.io.stata.StataReader(path) as r:
        df = r.read(convert_categoricals=False, convert_dates=False, convert_missing=False)
        return {"version": r._format_version, "typ": r._typlist, "fmt": r._fmtlist, "lbl": r._lbllist,
                "vl": r.value_labels(), "varl": r.variable_labels(), "dl": r.data_label, "ts": r.time_stamp,
                "bo": r._byteorder, "df": df}


def _ctx(col, name: str) -> DetectContext:
    s = Settings(mode="fidelity")
    return DetectContext(index=0, name=name, values=col.values, file_rel="x.dta", table=None, format="dta",
                         options=s.column_options("x.dta", None, name), settings=s, native=col.native)


# --------------------------------------------------------------------------- reading
def test_read_labels_dta() -> None:
    rr = read(FIX / "labels.dta", "dta")
    cols = {c.name: c for c in rr.tables[0].columns}
    assert rr.tables[0].header == ["stuid", "ecodis", "grade", "dob", "score", "visits", "town"]
    assert set(cols["ecodis"].values.tolist()) == {"No", "Yes"}  # labelled ints shown by their labels
    assert set(cols["grade"].values.tolist()) <= {"KG", "First", "Second", "Third"}
    assert cols["dob"].values[0] == "2012-06-30" and cols["dob"].native["kind"] == "date"
    assert cols["score"].values[0] is None and cols["score"].values[1] == "86.0"
    assert cols["visits"].values[0] is None and cols["visits"].native["typ"] == "h"
    assert cols["stuid"].values[0] == "010000" and cols["stuid"].native["kind"] == "str"
    ex = rr.layout.extra
    assert ex["stata_version"] == 118 and ex["write_version"] == 118 and ex["byteorder"] == "<"
    assert ex["_data_label"] == "Area F labelled fixture" and ex["_time_stamp"] == "04 Oct 2026 09:30"
    pub = rr.layout.to_public()["extra"]
    assert not any(k.startswith("_") for k in pub) and pub["value_label_tables"] == 2
    assert cols["grade"].native["var_label"] == "Grade level"


def test_detect_native_stata() -> None:
    rr = read(FIX / "labels.dta", "dta")
    cols = {c.name: c for c in rr.tables[0].columns}
    det = statio.detect_native(_ctx(cols["grade"], "grade"))
    assert det.type is ColumnType.FACTOR and det.source == "native" and det.params["value_labels"] == 4
    assert statio.detect_native(_ctx(cols["score"], "score")) is None
    assert statio.detect_native(_ctx(cols["stuid"], "stuid")) is None


def test_detect_native_stata_dates(monkeypatch) -> None:
    from datanonymous.detect import context, dates

    rr = read(FIX / "labels.dta", "dta")
    dob = {c.name: c for c in rr.tables[0].columns}["dob"]
    monkeypatch.setattr(dates, "detect_dates", lambda ctx: None)
    with pytest.raises(UnsupportedColumnError) as ei:
        statio.detect_native(_ctx(dob, "dob"))
    assert "--null dob" in ei.value.fix
    monkeypatch.setattr(dates, "detect_dates", lambda ctx: context.Detection(ctx.forced_type))
    assert statio.detect_native(_ctx(dob, "dob")).type is ColumnType.DATE


# --------------------------------------------------------------------------- identity round trips
def test_identity_roundtrip_labels(tmp_path: Path) -> None:
    rr = read(FIX / "labels.dta", "dta")
    out = identity_write(rr, "dta", tmp_path / "labels.dta")
    a, b = _meta(FIX / "labels.dta"), _meta(out)
    for k in ("version", "typ", "fmt", "lbl", "vl", "ts", "bo"):
        assert a[k] == b[k], k
    # free-text labels are never copied (R1-01): the data label and every variable label are blank
    assert a["dl"] and b["dl"] == "" and any(a["varl"].values()) and not any(b["varl"].values())
    assert "stata_labels_dropped" in [m.code for m in rr.messages]
    pd.testing.assert_frame_equal(a["df"], b["df"])
    rr2 = read(out, "dta", purpose="selfcheck")
    assert statio.layout_diff(rr.layout, rr2.layout, [{}], rr2.tables) == []
    assert statio.inferred_dtypes(out, rr2.layout, None) == statio.inferred_dtypes(FIX / "labels.dta", rr.layout,
                                                                                    None)


@pytest.mark.parametrize("version", [114, 117, 118, 119])
def test_versions_formats_and_storage_types(tmp_path: Path, version: int) -> None:
    src = tmp_path / f"v{version}.dta"
    df = pd.DataFrame({
        "b": np.array([1, -5, 7], dtype=np.int8), "h": pd.array([300, None, -2], dtype="Int16"),
        "l": np.array([70000, 1, 2], dtype=np.int32), "f": np.array([1.25, np.nan, 2.5], dtype=np.float32),
        "d": np.array([0.1, 2.0, np.nan]), "s": ["alpha", "", "be ta"], "dd": np.array([0, 365, 22000],
                                                                                        dtype=np.int32),
        "tc": np.array([0.0, 1500.0, 86400000.0 * 365 + 1]),
    })
    df.to_stata(src, write_index=False, version=version, byteorder=">" if version == 117 else "<",
                time_stamp=dt.datetime(2020, 2, 3, 4, 5), data_label="labelled")
    statio.patch_formats(src, version, ["%9.0g", "%8.0g", "%-12.0gc", "%9.2f", "%10.3f", "%-12s", "%tdDD/NN/CCYY",
                                        "%tc"])
    rr = read(src, "dta")
    cols = {c.name: c for c in rr.tables[0].columns}
    assert cols["dd"].values.tolist() == ["1960-01-01", "1960-12-31", "2020-03-26"]  # 1960 is a leap year
    assert cols["tc"].values.tolist() == ["1960-01-01 00:00:00.000", "1960-01-01 00:00:01.500",
                                          "1960-12-31 00:00:00.001"]
    assert cols["f"].values.tolist() == ["1.25", None, "2.5"] and cols["h"].values.tolist() == ["300", None, "-2"]
    out = identity_write(rr, "dta", tmp_path / "out.dta")
    a, b = _meta(src), _meta(out)
    for k in ("version", "typ", "fmt", "ts", "bo"):
        assert a[k] == b[k], k
    assert a["dl"] == "labelled" and b["dl"] == ""  # free text: never copied (R1-01)
    pd.testing.assert_frame_equal(a["df"], b["df"])


def test_strl_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "strl.dta"
    pd.DataFrame({"note": ["x" * 3000, "short", ""], "k": [1, 2, 3]}).to_stata(
        src, write_index=False, version=118, convert_strl=["note"])
    rr = read(src, "dta")
    assert rr.tables[0].columns[0].native["kind"] == "strl"
    out = identity_write(rr, "dta", tmp_path / "out.dta")
    assert _meta(out)["typ"][0] == "Q"
    assert _meta(out)["df"]["note"].tolist() == ["x" * 3000, "short", ""]


def test_non_unique_labels_keep_codes(tmp_path: Path) -> None:
    src = tmp_path / "dup.dta"
    pd.DataFrame({"g": np.array([0, 1, 2, 1], dtype=np.int8)}).to_stata(
        src, write_index=False, version=118, value_labels={"g": {0: "Other", 1: "Other", 2: "X"}})
    rr = read(src, "dta")
    col = rr.tables[0].columns[0]
    assert col.values.tolist() == ["0", "1", "2", "1"] and col.native["labels_rendered"] is False
    assert statio.detect_native(_ctx(col, "g")).type is ColumnType.FACTOR
    out = identity_write(rr, "dta", tmp_path / "out.dta")
    m = _meta(out)
    # the codes are kept; the label texts were never checked (only the codes were shown), so the table is not
    # written (R2-03)
    assert m["df"]["g"].tolist() == [0, 1, 2, 1] and "g" not in m["vl"]


def test_label_text_equal_to_unlabelled_code_keeps_codes(tmp_path: Path) -> None:
    src = tmp_path / "clash.dta"
    pd.DataFrame({"g": np.array([1, 2, 2], dtype=np.int8)}).to_stata(
        src, write_index=False, version=118, value_labels={"g": {1: "2"}})
    col = read(src, "dta").tables[0].columns[0]
    assert col.values.tolist() == ["1", "2", "2"]  # '2' would be ambiguous as a label


def test_partially_labelled_column(tmp_path: Path) -> None:
    src = tmp_path / "part.dta"
    pd.DataFrame({"score": np.array([45, 99, 70, 99], dtype=np.int16)}).to_stata(
        src, write_index=False, version=118, value_labels={"score": {99: "refused"}})
    rr = read(src, "dta")
    assert rr.tables[0].columns[0].values.tolist() == ["45", "refused", "70", "refused"]
    out = identity_write(rr, "dta", tmp_path / "out.dta")
    assert _meta(out)["df"]["score"].tolist() == [45, 99, 70, 99]


def test_extended_missing_values_warned(tmp_path: Path) -> None:
    src = tmp_path / "ext.dta"
    pd.DataFrame({"x": np.array([1.5, 2.0, 3.0])}).to_stata(src, write_index=False, version=118)
    data = src.read_bytes()
    dot_a = struct.pack("<d", 8.990660123939097e+307)  # Stata's '.a' for doubles
    src.write_bytes(data.replace(struct.pack("<d", 2.0), dot_a, 1))
    rr = read(src, "dta")
    assert rr.tables[0].columns[0].values.tolist() == ["1.5", None, "3.0"]
    assert [m.code for m in rr.messages] == ["stata_extended_missing"]


def test_bad_dta_fails() -> None:
    with pytest.raises(ReadError):
        read(FIX / "records.json", "dta")


def test_version_sniff() -> None:
    assert statio._version_of(b"<stata_dta><header><release>118</release>") == 118
    assert statio._version_of(bytes([113, 2, 1, 0])) == 113


# --------------------------------------------------------------------------- end to end
def test_cli_labels_dta(tmp_path: Path) -> None:
    res, out, rep = run_dir(FIX / "labels.dta", tmp_path, "--null", "dob")
    assert res.exit_code == 0, res.output
    a, b = _meta(FIX / "labels.dta"), _meta(out / "labels.dta")
    assert b["dl"] == "" and not any(b["varl"].values())  # free-text labels: never copied (R1-01)
    for k in ("version", "typ", "fmt", "lbl", "vl", "bo"):
        if k == "typ":
            assert [isinstance(x, int) for x in a[k]] == [isinstance(x, int) for x in b[k]]
            assert [x for x in a[k] if not isinstance(x, int)] == [x for x in b[k] if not isinstance(x, int)]
        else:
            assert a[k] == b[k], k
    sdf, odf = a["df"], b["df"]
    assert len(odf) == len(sdf)
    assert set(odf["ecodis"]) <= {0, 1} and set(odf["grade"]) <= {0, 1, 2, 3}
    assert sorted(odf["grade"]) == sorted(sdf["grade"])  # exact quota of the labelled levels
    assert odf["dob"].isna().all() and odf["score"].isna().sum() == sdf["score"].isna().sum()
    assert not set(sdf["stuid"]) & set(odf["stuid"])  # synthetic IDs
    assert pd.read_stata(out / "labels.dta").dtypes.astype(str).tolist() == \
        pd.read_stata(FIX / "labels.dta").dtypes.astype(str).tolist()
    blob = all_bytes(out, rep)
    for sid in sdf["stuid"]:
        assert sid.encode() not in blob
    txt = text_of(rep)
    assert "Area F labelled fixture" not in txt  # the data label stays out of the report


def test_cli_labels_dta_with_simulated_dates(monkeypatch, tmp_path: Path) -> None:
    from test_area_f_arrow import _simulate_areas_a_b

    _simulate_areas_a_b(monkeypatch)
    res, out, rep = run_dir(FIX / "labels.dta", tmp_path)
    assert res.exit_code == 0, res.output
    b = _meta(out / "labels.dta")
    assert b["fmt"][3] == "%td" and b["typ"][3] == "d"
    assert set(pd.read_stata(out / "labels.dta")["dob"]) <= set(pd.read_stata(FIX / "labels.dta")["dob"])


def test_dta_same_seed_same_bytes(tmp_path: Path) -> None:
    r1, o1, _ = run_dir(FIX / "labels.dta", tmp_path / "a", "--null", "dob", seed="9")
    r2, o2, _ = run_dir(FIX / "labels.dta", tmp_path / "b", "--null", "dob", seed="9")
    assert r1.exit_code == r2.exit_code == 0
    assert (o1 / "labels.dta").read_bytes() == (o2 / "labels.dta").read_bytes()


# --------------------------------------------------------------------------- SAS (stubbed reader)
def _sas_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame({
        "STUDENT_ID": [f"{500000 + 13 * i}" for i in range(n)],
        "FIRST_NAME": [["Olga", "Pavel", "Rita", "Sami"][i % 4] + str(i) for i in range(n)],
        "SCORE": np.where(np.arange(n) % 9 == 0, np.nan, np.round(rng.uniform(10, 90, n), 1)),
        "GRADE": [float(g) for g in rng.integers(1, 6, n)],
        "SITE": [["North", "South"][i % 2] for i in range(n)],
    })


def _stub_sas(monkeypatch, frame: pd.DataFrame | Exception) -> list:
    calls = []

    def fake(path, fmt, encoding):
        calls.append((Path(path).name, fmt, encoding))
        if isinstance(frame, Exception):
            raise frame
        return frame.copy()

    monkeypatch.setattr(statio, "read_sas_frame", fake)
    return calls


@pytest.mark.parametrize("name,fmt", [("survey.sas7bdat", "sas7bdat"), ("survey.xpt", "xport")])
def test_sas_written_as_parquet(monkeypatch, tmp_path: Path, name: str, fmt: str) -> None:
    """SAS files cannot be created here: pd.read_sas is stubbed with a frame shaped like its output."""
    calls = _stub_sas(monkeypatch, _sas_frame())
    src = tmp_path / "in"
    src.mkdir()
    (src / name).write_bytes(b"not really SAS")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert calls[0] == (name, fmt, "infer" if fmt == "sas7bdat" else "latin-1")
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".parquet") == ["survey.parquet"]
    t = pq.read_table(out / "survey.parquet")
    real = _sas_frame()
    assert t.schema.names == list(real.columns) and t.num_rows == len(real)
    assert str(t.schema.field("SCORE").type) == "double" and t.column("SCORE").null_count == 5
    assert not set(real["STUDENT_ID"]) & set(t.column("STUDENT_ID").to_pylist())
    assert not set(real["FIRST_NAME"]) & set(t.column("FIRST_NAME").to_pylist())
    data = report(rep)
    assert data["files"][0]["output"] == "survey.parquet" and data["files"][0]["status"] == "WARN"
    assert any(w["code"] == "sas_as_parquet" for w in data["warnings"])
    blob = all_bytes(out, rep)
    for v in real["FIRST_NAME"]:
        assert v.encode() not in blob


def test_sas_read_error_fails_with_fix(monkeypatch, tmp_path: Path) -> None:
    _stub_sas(monkeypatch, ValueError("bad header"))
    src = tmp_path / "in"
    src.mkdir()
    (src / "broken.sas7bdat").write_bytes(b"xx")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 3
    f = report(rep)["failures"][0]
    assert f["source"] == "broken.sas7bdat" and "export the dataset" in f["fixes"][0]
    assert "bad header" not in json.dumps(report(rep))  # exception text never reaches the report


def test_sas_inferred_dtypes_use_read_sas(monkeypatch, tmp_path: Path) -> None:
    calls = _stub_sas(monkeypatch, _sas_frame())
    from datanonymous.model import FileLayout

    dts = statio.inferred_dtypes(tmp_path / "x.xpt", FileLayout("sas", extra={"sas_format": "xport"}), None)
    assert dts == [str(t) for t in _sas_frame().dtypes]
    assert calls[-1][1:] == ("xport", None)
