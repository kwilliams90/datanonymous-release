"""AREA F: flat JSON / JSON lines (typed values, missing keys, formatting mirror, nested values)."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datanonymous.errors import NotTabular, ReadError, UnsupportedColumnError
from datanonymous.io import jsonio
from datanonymous.model import FileLayout
from test_area_f_util import FIX, all_bytes, identity_write, read, report, run_dir, text_of

RECS = [{"id": 1, "name": "Zoë", "score": 9.50, "ok": True, "tag": None},
        {"id": 2, "name": "A/B", "ok": False, "tag": "x"},
        {"id": 3, "name": "C", "score": 1e-3, "ok": True, "tag": "y"}]


def _roundtrip(tmp_path: Path, text: str | bytes, name: str = "x.json") -> tuple[bytes, bytes, object]:
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    fmt = "jsonl" if name.endswith(".jsonl") else "json"
    rr = read(p, fmt)
    out = identity_write(rr, fmt, tmp_path / ("out_" + name))
    return p.read_bytes(), out.read_bytes(), rr


# --------------------------------------------------------------------------- reading
def test_read_records_json() -> None:
    rr = read(FIX / "records.json", "json")
    t = rr.tables[0]
    assert t.header == ["id", "name", "score", "active", "level", "nickname", "visits"]  # one consistent order
    cols = {c.name: c.values.tolist() for c in t.columns}
    assert cols["id"][0] == "1000" and cols["score"][1] == "54.4" and cols["active"][0] == "false"
    assert cols["score"][0] is None  # JSON null
    assert cols["nickname"][0] == jsonio.MISSING_KEY  # key absent from the record
    assert Counter(v == jsonio.MISSING_KEY for v in cols["nickname"])[True] == 10
    ex = rr.layout.extra
    assert ex["format_exact"] is True and ex["json_layout"] == "array" and ex["ensure_ascii"] is True
    assert ex[jsonio.KINDS_KEY] == {"id": ["number"], "name": ["string"], "score": ["number"], "active": ["bool"],
                                    "level": ["string"], "nickname": ["string"], "visits": ["number"]}
    assert rr.messages == []
    assert jsonio.MISSING_KEY.strip() == ""  # a blank token: counted as missing by every policy


def test_number_text_is_kept_exactly(tmp_path: Path) -> None:
    p = tmp_path / "n.json"
    p.write_text('[{"a": 1.50, "b": 10, "c": -0, "d": 1E5, "e": NaN, "f": 12345678901234567890}]',
                 encoding="utf-8")
    v = [c.values[0] for c in read(p, "json").tables[0].columns]
    assert v == ["1.50", "10", "-0", "1E5", "NaN", "12345678901234567890"]


# --------------------------------------------------------------------------- formatting mirror (identity)
@pytest.mark.parametrize("text", [
    json.dumps(RECS, indent=2) + "\n",
    json.dumps(RECS, indent=4).replace("\n", "\r\n"),
    json.dumps(RECS, indent="\t", ensure_ascii=False),
    json.dumps(RECS),
    json.dumps(RECS, separators=(",", ":")),
    "[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in RECS) + "\n]\n",
    "[\n  " + ",\n  ".join(json.dumps(r, separators=(", ", ": ")) for r in RECS) + "\n]",
    json.dumps(RECS, indent=2).replace("\\u00eb", "\\u00EB"),
    pd.DataFrame(RECS).to_json(orient="records"),
    "[]",
    "[{}, {}]\n",
])
def test_identity_formatting_is_byte_exact(tmp_path: Path, text: str) -> None:
    src, out, rr = _roundtrip(tmp_path, text)
    assert rr.layout.extra["format_exact"] is True
    assert out == src


@pytest.mark.parametrize("text", [
    "\n".join(json.dumps(r) for r in RECS) + "\n",
    "\r\n".join(json.dumps(r, separators=(",", ":")) for r in RECS),
    pd.DataFrame(RECS).to_json(orient="records", lines=True),
])
def test_identity_jsonl_is_byte_exact(tmp_path: Path, text: str) -> None:
    src, out, rr = _roundtrip(tmp_path, text, "x.jsonl")
    assert rr.layout.extra["format_exact"] is True and out == src


def test_bom_and_cp1252_encoding_mirrored(tmp_path: Path) -> None:
    src, out, rr = _roundtrip(tmp_path, b"\xef\xbb\xbf" + json.dumps(RECS, ensure_ascii=False).encode("utf-8"))
    assert out == src and rr.layout.bom == b"\xef\xbb\xbf"
    src2, out2, rr2 = _roundtrip(tmp_path, json.dumps(RECS, ensure_ascii=False).encode("cp1252"), "c.json")
    assert rr2.layout.encoding == "cp1252" and out2 == src2


def test_inconsistent_formatting_is_approximate(tmp_path: Path) -> None:
    text = '[{"a": 1, "b": 2},\n {"b": 3, "a": 4}]'
    src, out, rr = _roundtrip(tmp_path, text)
    codes = [m.code for m in rr.messages]
    assert "json_format_approximate" in codes and "json_key_order" in codes
    assert json.loads(out) == [{"a": 1, "b": 2}, {"a": 4, "b": 3}]  # values and types kept


def test_jsonl_blank_lines_dropped_with_warning(tmp_path: Path) -> None:
    src, out, rr = _roundtrip(tmp_path, '{"a": 1}\n\n{"a": 2}\n', "x.jsonl")
    assert [m.code for m in rr.messages][0] == "blank_lines_dropped"
    assert out == b'{"a": 1}\n{"a": 2}\n'


# --------------------------------------------------------------------------- typed values
def test_value_kinds_written_back(tmp_path: Path) -> None:
    text = json.dumps([{"zip": "02134", "n": 5, "s": "7", "b": True, "mix": 3},
                       {"zip": "10001", "n": 6, "s": "8", "b": False, "mix": "N/A"}])
    p = tmp_path / "k.json"
    p.write_text(text, encoding="utf-8")
    rr = read(p, "json")
    cols = [np.array(v, dtype=object) for v in (["00001", "99999"], ["7", "8"], ["9", "x"], ["false", "true"],
                                                ["N/A", "4"])]
    out = identity_write(rr, "json", tmp_path / "o.json", columns=cols)
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"zip": "00001", "n": 7, "s": "9", "b": False, "mix": "N/A"},
        {"zip": "99999", "n": 8, "s": "x", "b": True, "mix": 4}]


def test_missing_keys_and_nulls_written_back(tmp_path: Path) -> None:
    rr = read(FIX / "lines.jsonl", "jsonl")
    out = identity_write(rr, "jsonl", tmp_path / "o.jsonl")
    assert out.read_bytes() == (FIX / "lines.jsonl").read_bytes()


def test_ambiguous_string_number_column_fails(tmp_path: Path) -> None:
    p = tmp_path / "amb.json"
    p.write_text('[{"code": 12}, {"code": "12"}, {"code": "x"}]', encoding="utf-8")
    with pytest.raises(UnsupportedColumnError) as ei:
        read(p, "json")
    assert "--null code or --exclude code" in ei.value.fix


@pytest.mark.parametrize("text,exc", [
    ('{"a": 1}', NotTabular), ("[1, 2, 3]", NotTabular), ('"text"', NotTabular),
    ('[{"a": 1}, 2]', ReadError), ('[{"a": 1, "a": 2}]', ReadError), ('[{"a": 1}', ReadError),
    ('[{"a": 1}] trailing', ReadError), ("", ReadError),
])
def test_shapes(tmp_path: Path, text: str, exc: type) -> None:
    p = tmp_path / "s.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(exc):
        read(p, "json")


def test_non_record_json_is_skipped_not_failed(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    (src / "config.json").write_text('{"setting": true}', encoding="utf-8")
    (src / "rows.json").write_text(json.dumps(RECS), encoding="utf-8")
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    data = report(rep)
    assert [s["path"] for s in data["skipped"]] == ["config.json"]
    assert not (out / "config.json").exists() and (out / "rows.json").is_file()


# --------------------------------------------------------------------------- nested values
def _nested_file(tmp_path: Path) -> Path:
    src = tmp_path / "in"
    src.mkdir()
    recs = [{"id": f"P{i:03d}", "score": i % 7, "tags": ["secret-tag", str(i)] if i % 2 else None,
             "addr": {"city": "Realtown", "zip": "73301"}} for i in range(12)]
    (src / "nested.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    return src


def test_nested_values_fail_with_fix(tmp_path: Path) -> None:
    src = _nested_file(tmp_path)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 3 and not (out / "nested.json").exists()
    f = report(rep)["failures"][0]
    assert "'tags'" in f["reasons"][0] and "'addr'" in f["reasons"][0]
    assert "--null tags or --exclude tags" in f["fixes"][0] and "--null addr or --exclude addr" in f["fixes"][0]


def test_nested_null_and_exclude(tmp_path: Path) -> None:
    src = _nested_file(tmp_path)
    res, out, rep = run_dir(src, tmp_path, "--null", "tags", "--exclude", "addr")
    assert res.exit_code == 0, res.output
    recs = json.loads((out / "nested.json").read_text(encoding="utf-8"))
    assert len(recs) == 12 and all(list(r) == ["id", "score", "tags"] and r["tags"] is None for r in recs)
    blob = all_bytes(out, rep)
    assert b"secret-tag" not in blob and b"Realtown" not in blob and b"73301" not in blob


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("name,fmt", [("records.json", "json"), ("lines.jsonl", "jsonl")])
def test_cli_end_to_end(tmp_path: Path, name: str, fmt: str) -> None:
    res, out, rep = run_dir(FIX / name, tmp_path)
    assert res.exit_code == 0, res.output
    raw = (FIX / name).read_text(encoding="utf-8")
    syn = (out / name).read_text(encoding="utf-8")
    load = (lambda s: json.loads(s)) if fmt == "json" else (lambda s: [json.loads(x) for x in s.splitlines()])
    real, fake = load(raw), load(syn)
    assert len(fake) == len(real)
    order = ["id", "name", "score", "active", "level", "nickname", "visits"]
    for r in fake:
        assert [k for k in order if k in r] == list(r)  # same key order
    for key in order:
        assert sum(key in r for r in fake) == sum(key in r for r in real)  # missing-key rate exact
        assert sum(r.get(key, 0) is None for r in fake) == sum(r.get(key, 0) is None for r in real)
        kinds = {type(r[key]).__name__ for r in fake if r.get(key) is not None}
        assert kinds == {type(r[key]).__name__ for r in real if r.get(key) is not None}, key
    # formatting mirrored: same indentation/separators, so the first lines look alike
    assert syn.splitlines()[0][:2] == raw.splitlines()[0][:2] and syn.endswith("\n")
    # privacy: names never copied; ids synthetic
    real_names = {r["name"] for r in real}
    assert not real_names & {r["name"] for r in fake}
    txt = text_of(out, rep)
    for nm in real_names:
        for form in (json.dumps(nm), json.dumps(nm, ensure_ascii=False), f"| {nm} |", f"'{nm}'"):
            assert form not in txt
    rr_src = read(FIX / name, fmt)
    rr_out = read(out / name, fmt, purpose="selfcheck")
    assert jsonio.layout_diff(rr_src.layout, rr_out.layout, [{}], rr_out.tables) == []
    assert jsonio.inferred_dtypes(out / name, rr_out.layout, None) == jsonio.inferred_dtypes(FIX / name,
                                                                                            rr_src.layout, None)


def test_json_same_seed_same_bytes(tmp_path: Path) -> None:
    r1, o1, _ = run_dir(FIX / "records.json", tmp_path / "a", seed="5")
    r2, o2, _ = run_dir(FIX / "records.json", tmp_path / "b", seed="5")
    assert r1.exit_code == r2.exit_code == 0
    assert (o1 / "records.json").read_bytes() == (o2 / "records.json").read_bytes()


def test_jsonl_gz_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    raw = (FIX / "lines.jsonl").read_bytes()
    (src / "lines.jsonl.gz").write_bytes(gzip.compress(raw, mtime=1234567))
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    data = (out / "lines.jsonl.gz").read_bytes()
    assert data[4:8] == (1234567).to_bytes(4, "little")  # gzip MTIME mirrored
    lines = gzip.decompress(data).decode("utf-8").splitlines()
    assert len(lines) == 30 and all(json.loads(x) for x in lines)


def test_layout_diff_flags_new_value_kinds() -> None:
    s = FileLayout("json", extra={jsonio.KINDS_KEY: {"a": ["number"]}, jsonio.STYLE_KEY: {"key_sep": ": "},
                                  "json_layout": "array"})
    o = FileLayout("json", extra={jsonio.KINDS_KEY: {"a": ["number", "string"]}, jsonio.STYLE_KEY: {"key_sep": ":"},
                                  "json_layout": "array"})
    issues = jsonio.layout_diff(s, o, [], [])
    assert any("value type(s) string" in i for i in issues) and any("key sep" in i for i in issues)
