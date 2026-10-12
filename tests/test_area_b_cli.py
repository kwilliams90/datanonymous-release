"""AREA B end to end: the numeric fixtures through the CLI (formats, tokens, notations, privacy, determinism)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import numeric as N
from datanonymous.io import csvio
from datanonymous.model import ColumnType as T
from datanonymous.profile.table import profile_table

from conftest import REPO, read_dicts, run_cli

FIX = REPO / "tests" / "fixtures" / "numeric"
DATA = ("floats.csv", "suppressed.csv")
AREA_B = {"NUMERIC_FORMATTED", "NUMERIC_WITH_TOKENS", "DECIMAL"}


@dataclass
class Run:
    inp: Path
    out: Path
    rep: Path
    output: str


def _run(tmp: Path, files: tuple[str, ...], *args: str) -> Run:
    inp = tmp / "input"
    inp.mkdir()
    for f in files:
        shutil.copy(FIX / f, inp / f)
    out, rep = tmp / "out", tmp / "rep"
    res = run_cli(inp, "--out-dir", out, "--report-dir", rep, *args)
    assert res.exit_code == 0, res.output
    return Run(inp, out, rep, res.output)


@pytest.fixture(scope="module")
def b_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _run(tmp_path_factory.mktemp("area_b"), DATA, "--seed", "7")


@pytest.fixture(scope="module")
def comma_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _run(tmp_path_factory.mktemp("area_b_comma"), ("decimal_comma.csv",), "--seed", "7", "--decimal", ",")


def profiles(name: str, **cli):
    raw = csvio.parse_text_bytes((FIX / name).read_bytes())
    tp = profile_table(raw.tables[0], name, "csv", build_settings(dict(cli)))
    return {p.name: p for p in tp.columns}


def file_report(run: Run, name: str) -> dict:
    return json.loads((run.rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- layout and types
@pytest.mark.parametrize("name", DATA)
def test_layout_mirrored(b_run: Run, name: str) -> None:
    src, out = (FIX / name).read_bytes(), (b_run.out / name).read_bytes()
    assert src.split(b"\n", 1)[0] == out.split(b"\n", 1)[0]
    assert src.count(b"\n") == out.count(b"\n") and b"\r\n" not in out
    data = json.loads((b_run.rep / "report.json").read_text(encoding="utf-8"))
    assert data["totals"]["FAIL"] == 0


def test_detected_types() -> None:
    cols = profiles("floats.csv")
    want = {
        "student_id": T.IDENTIFIER, "gpa": T.FLOAT, "gpa_r": T.FLOAT, "attendance_rate": T.NUMERIC_FORMATTED,
        "fee_balance": T.NUMERIC_FORMATTED, "refund": T.NUMERIC_FORMATTED, "adjustment": T.NUMERIC_FORMATTED,
        "price_eur": T.NUMERIC_FORMATTED, "tuition_gbp": T.NUMERIC_FORMATTED, "proportion": T.FLOAT,
        "stata_prop": T.FLOAT, "change": T.FLOAT, "p_value": T.FLOAT, "sci_id": T.FLOAT, "income": T.FLOAT,
        "bimodal": T.FLOAT, "score_x10": T.INTEGER, "sentinel_hours": T.INTEGER, "big_count": T.NUMERIC_FORMATTED,
    }
    assert {k: v.type for k, v in cols.items()} == want
    assert cols["p_value"].subtype == "scientific" and cols["sci_id"].params["excel_mangled"]
    assert cols["sentinel_hours"].subtype == "discrete"  # -9/999 are ordinary pmf values
    assert cols["stata_prop"].params["leading_zero"] is False and cols["change"].params["plus_sign"]
    cols = profiles("suppressed.csv")
    assert {k: v.type for k, v in cols.items()} == {
        "school_id": T.ORG_CODE, "subgroup": T.FACTOR, "n_tested": T.NUMERIC_WITH_TOKENS,
        "pct_proficient": T.NUMERIC_WITH_TOKENS, "avg_score": T.NUMERIC_WITH_TOKENS, "grade_level": T.ORDINAL,
        "status": T.FACTOR, "days_enrolled": T.NUMERIC_WITH_TOKENS, "fee": T.NUMERIC_WITH_TOKENS}
    assert cols["avg_score"].model.missing_counts == {"--": cols["avg_score"].stats["missing"]}


# --------------------------------------------------------------------------- per-column fidelity of the text forms
def test_floats_text_forms(b_run: Run) -> None:
    cols = profiles("floats.csv")
    real, syn = read_dicts(FIX / "floats.csv"), read_dicts(b_run.out / "floats.csv")
    col = lambda rows, c: [r[c] for r in rows]  # noqa: E731
    for c, p in cols.items():
        if p.type is T.NUMERIC_FORMATTED:
            nums = [N.parse_with_format(v, p.params["format"]) for v in col(syn, c)]
            assert not any(math.isnan(x) for x in nums), c
            assert min(nums) == p.params["min"] and max(nums) == p.params["max"], c
    fee = col(syn, "fee_balance")
    assert any(v.startswith("($") for v in fee) and not any(v.startswith("-") for v in fee)
    assert any(v.startswith("-$") for v in col(syn, "refund"))
    assert any(v.startswith("$-") for v in col(syn, "adjustment"))
    assert all(re.fullmatch(r"[0-9]+\.[0-9]%", v) for v in col(syn, "attendance_rate"))
    assert all(N.EXCEL_SCI_RE.fullmatch(v) for v in col(syn, "sci_id"))
    pv = col(syn, "p_value")
    assert any("e-0" in v for v in pv) and any("e" not in v for v in pv)
    assert all(v.startswith(".") for v in col(syn, "stata_prop"))
    assert all(re.fullmatch(r"[+-][0-9]+\.[0-9]|0\.0", v) for v in col(syn, "change"))
    assert all(re.fullmatch(r"[0-9]\.[0-9]{2}|NA|", v) for v in col(syn, "gpa"))
    assert Counter(v for v in col(syn, "gpa") if v in ("", "NA")) == Counter(
        v for v in col(real, "gpa") if v in ("", "NA"))
    assert all(re.fullmatch(r"[0-9]+\.0", v) for v in col(syn, "score_x10"))
    assert Counter(col(syn, "sentinel_hours")) == Counter(col(real, "sentinel_hours"))


def test_suppressed_tokens_exact(b_run: Run) -> None:
    cols = profiles("suppressed.csv")
    real, syn = read_dicts(FIX / "suppressed.csv"), read_dicts(b_run.out / "suppressed.csv")
    for c, p in cols.items():
        if p.type is not T.NUMERIC_WITH_TOKENS:
            continue
        fmt = p.params["format"]
        r_tok = Counter(r[c] for r in real if math.isnan(N.parse_with_format(r[c], fmt)))
        s_tok = Counter(s[c] for s in syn if math.isnan(N.parse_with_format(s[c], fmt)))
        assert r_tok == s_tok, c  # every token (and NA token) at its exact count
        nums = [N.parse_with_format(s[c], fmt) for s in syn]
        nums = [x for x in nums if not math.isnan(x)]
        assert min(nums) == p.params["min"] and max(nums) == p.params["max"], c
    assert Counter(r["grade_level"] for r in real) == Counter(s["grade_level"] for s in syn)


def test_report_fidelity_and_token_stats(b_run: Run) -> None:
    for name in DATA:
        rep = file_report(b_run, name)
        for c in rep["tables"][0]["columns"]:
            if c["type"] in AREA_B or c["subtype"] == "scientific":
                assert c["fidelity"]["verdict"] == "ok", (name, c["name"], c["fidelity"])
                assert any(ch["check"] == "ks" for ch in c["fidelity"]["checks"]), c["name"]
            if c["type"] == "NUMERIC_WITH_TOKENS":
                assert c["real"]["token_counts"] == c["synthetic"]["token_counts"], c["name"]
            if c["type"] == "NUMERIC_FORMATTED":
                assert c["synthetic"]["format_conformance"] == 1.0
                assert c["real"]["template"] == c["synthetic"]["template"]


def test_out_dir_docs(b_run: Run) -> None:
    schema = json.loads((b_run.out / "schema.json").read_text(encoding="utf-8"))
    entries = {e["name"]: e for f in schema["files"] for t in f["tables"] for e in t["columns"]}
    assert entries["fee_balance"]["readr"] == "c" and entries["fee_balance"]["template"] == "$#,##0.00;($#,##0.00)"
    assert entries["fee_balance"]["r_class"] == "numeric" and "parse_number" in entries["fee_balance"]["parse"]
    assert entries["n_tested"]["readr"] == "c" and entries["n_tested"]["r_class"] == "character"
    assert entries["p_value"]["readr"] == "d"
    dd = (b_run.out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "NUMERIC_WITH_TOKENS" in dd and "NUMERIC_FORMATTED" in dd


# --------------------------------------------------------------------------- privacy
def _all_text(d: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in sorted(d.rglob("*")) if p.is_file())


def test_no_real_ids_or_private_tokens_leak(b_run: Run) -> None:
    real_ids = {r["student_id"] for r in read_dicts(FIX / "floats.csv")}
    syn_ids = {r["student_id"] for r in read_dicts(b_run.out / "floats.csv")}
    assert not real_ids & syn_ids
    out_text, rep_text = _all_text(b_run.out), _all_text(b_run.rep)
    for rid in real_ids:
        assert rid not in out_text and rid not in rep_text
    # tokens that are not suppression codes are written to the data only, never to the report or docs
    assert "Withdrawn" in (b_run.out / "suppressed.csv").read_text(encoding="utf-8")
    assert "Withdrawn" not in rep_text
    for doc in ("README.md", "DATA_DICTIONARY.md", "schema.json"):
        assert "Withdrawn" not in (b_run.out / doc).read_text(encoding="utf-8")


def test_no_full_row_copies(b_run: Run) -> None:
    for name in DATA:
        real = {tuple(r.values()) for r in read_dicts(FIX / name)}
        assert not any(tuple(s.values()) in real for s in read_dicts(b_run.out / name))


# --------------------------------------------------------------------------- decimal comma
def test_decimal_comma_with_flag(comma_run: Run) -> None:
    src, out = (FIX / "decimal_comma.csv").read_bytes(), (comma_run.out / "decimal_comma.csv").read_bytes()
    assert out.split(b"\r\n", 1)[0] == src.split(b"\r\n", 1)[0]
    assert out.count(b"\r\n") == src.count(b"\r\n")
    text = out.decode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        out.decode("utf-8")  # the euro sign stays a cp1252 byte
    rows = [line.split(";") for line in text.split("\r\n")[1:] if line]
    cols = profiles("decimal_comma.csv", decimal=",")
    header = text.split("\r\n", 1)[0].split(";")
    for c in ("betrag", "anteil", "preis", "faktor"):
        p = cols[c]
        assert p.type is T.NUMERIC_FORMATTED and p.params["format"]["decimal_mark"] == ","
        vals = [r[header.index(c)] for r in rows]
        nums = [N.parse_with_format(v, p.params["format"]) for v in vals]
        assert not any(math.isnan(x) for x in nums), c
        assert min(nums) == p.params["min"] and max(nums) == p.params["max"]
    assert all(re.fullmatch(r"-?[0-9]{1,3}(\.[0-9]{3})*,[0-9]{2}", r[header.index("betrag")]) for r in rows)
    assert all(r[header.index("preis")].endswith(" \u20ac") for r in rows)


def test_decimal_comma_without_flag_stays_text(tmp_path: Path) -> None:
    run = _run(tmp_path, ("decimal_comma.csv",), "--seed", "7")
    rep = file_report(run, "decimal_comma.csv")
    types = {c["name"]: c["type"] for c in rep["tables"][0]["columns"]}
    assert types["betrag"] == "HIGH_CARD_CATEGORICAL" and types["faktor"] == "HIGH_CARD_CATEGORICAL"
    real = (FIX / "decimal_comma.csv").read_bytes().decode("cp1252").split("\r\n")
    out = (run.out / "decimal_comma.csv").read_bytes().decode("cp1252").split("\r\n")
    i = real[0].split(";").index("betrag")
    real_vals = {line.split(";")[i] for line in real[1:] if line}
    assert not real_vals & {line.split(";")[i] for line in out[1:] if line}


# --------------------------------------------------------------------------- forced types
def test_forced_area_b_types_end_to_end(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    lines = ["amount,score"] + [f"{i * 1.25:.2f},{i if i % 9 else 'x'}" for i in range(1, 120)]
    (inp / "forced.csv").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(inp, "--out-dir", out, "--report-dir", rep, "--seed", "2",
                  "--force-type", "amount=DECIMAL", "--force-type", "score=NUMERIC_WITH_TOKENS")
    assert res.exit_code == 0, res.output
    cols = {c["name"]: c for c in json.loads((rep / "files" / "forced.csv.json").read_text(encoding="utf-8"))
            ["tables"][0]["columns"]}
    assert cols["amount"]["type"] == "DECIMAL" and cols["amount"]["params"]["scale"] == 2
    assert cols["score"]["type"] == "NUMERIC_WITH_TOKENS" and cols["score"]["source"] == "force-type"
    syn = read_dicts(out / "forced.csv")
    assert all(re.fullmatch(r"[0-9]+\.[0-9]{2}", r["amount"]) for r in syn)
    assert Counter(r["score"] for r in syn)["x"] == 13
    assert "'x'" not in (rep / "files" / "forced.csv.md").read_text(encoding="utf-8")


def test_forced_type_that_does_not_apply_fails_the_file(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "bad.csv").write_bytes(b"name\nann\nbob\ncy\n")
    res = run_cli(inp, "--out-dir", tmp_path / "out", "--seed", "1", "--force-type", "name=NUMERIC_FORMATTED")
    assert res.exit_code == 3
    assert not (tmp_path / "out" / "bad.csv").exists()


# --------------------------------------------------------------------------- determinism and fixtures
def test_same_seed_same_bytes(b_run: Run, tmp_path: Path) -> None:
    again = _run(tmp_path, DATA, "--seed", "7")
    for name in DATA:
        assert (again.out / name).read_bytes() == (b_run.out / name).read_bytes()


def test_inspect_is_ascii(tmp_path: Path) -> None:
    res = run_cli("inspect", FIX / "floats.csv")
    assert res.exit_code == 0 and res.output.isascii()
    assert "NUMERIC_FORMATTED" in res.output and "FLOAT(scientific)" in res.output


def test_fixtures_are_reproducible(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_numeric_fixtures", FIX / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(tmp_path)
    for name in (*DATA, "decimal_comma.csv"):
        assert (tmp_path / name).read_bytes() == (FIX / name).read_bytes(), name
