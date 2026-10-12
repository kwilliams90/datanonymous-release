"""AREA B hardening end to end: the adversarial long-format fixture through the CLI (D4 floors, forms, privacy)."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO, run_cli

FIX = REPO / "tests" / "fixtures" / "numeric"
NAME = "adversarial.csv"


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("make_numeric_fixtures_adv", FIX / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_fixture_module()


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _run(tmp: Path, *args: str) -> tuple[Path, Path, str]:
    inp = tmp / "input"
    inp.mkdir()
    shutil.copy(FIX / NAME, inp / NAME)
    out, rep = tmp / "out", tmp / "rep"
    res = run_cli(inp, "--out-dir", out, "--report-dir", rep, *args)
    assert res.exit_code == 0, res.output
    return out, rep, res.output


@pytest.fixture(scope="module")
def adv(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    return _run(tmp_path_factory.mktemp("area_b_adv"), "--seed", "11")


def _columns(rep: Path) -> dict[str, dict]:
    data = json.loads((rep / "files" / f"{NAME}.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in data["tables"][0]["columns"]}


def test_layout_bom_crlf_and_row_count(adv) -> None:
    out, rep, _ = adv
    src, dst = (FIX / NAME).read_bytes(), (out / NAME).read_bytes()
    assert dst.startswith(b"\xef\xbb\xbf") and src.split(b"\r\n", 1)[0] == dst.split(b"\r\n", 1)[0]
    assert dst.count(b"\r\n") == src.count(b"\r\n") and dst.count(b"\n") == src.count(b"\n")
    assert len(rows(out / NAME)) == len(rows(FIX / NAME))
    data = json.loads((rep / "files" / f"{NAME}.json").read_text(encoding="utf-8"))
    assert data["status"] == "WARN" and data["selfcheck"] == []


def test_within_id_constant_columns_keep_minority_token_and_extreme(adv) -> None:
    """D4 + D14: '<10' and the maximum balance only occur as a minority value inside one student."""
    out, rep, _ = adv
    real, syn = rows(FIX / NAME), rows(out / NAME)
    assert Counter(r["score_supp"] for r in real)["<10"] == 1
    assert Counter(r["balance"] for r in real)[MOD.MAX_BALANCE] == 1
    assert "<10" in {r["score_supp"] for r in syn} and "*" in {r["score_supp"] for r in syn}
    assert MOD.MAX_BALANCE in {r["balance"] for r in syn}
    by_id: dict[str, set] = {}
    for r in syn:  # still one value per synthetic student (D4)
        by_id.setdefault(r["student_id"], set()).add((r["score_supp"], r["balance"]))
    assert all(len(v) == 1 for v in by_id.values())
    cols = _columns(rep)
    assert cols["score_supp"]["role"] == "entity" and cols["balance"]["role"] == "entity"
    assert cols["score_supp"]["flags"]["rare_category_adjusted"]
    assert cols["balance"]["flags"]["extreme_value_adjusted"]


def test_trailing_point_numbers_keep_their_dtype(adv) -> None:
    out, rep, _ = adv
    syn = rows(out / NAME)
    assert all(re.fullmatch(r"[0-9]+\.", r["sas_amount"]) for r in syn)
    src = pd.read_csv(FIX / NAME, encoding="utf-8-sig")
    dst = pd.read_csv(out / NAME, encoding="utf-8-sig")
    assert src["sas_amount"].dtype == dst["sas_amount"].dtype == "float64"
    text = (rep / "files" / f"{NAME}.md").read_text(encoding="utf-8")
    assert "sas_amount': pandas reads" not in text


def test_wide_range_scientific_column(adv) -> None:
    out, _, _ = adv
    real, syn = [r["wide_sci"] for r in rows(FIX / NAME)], [r["wide_sci"] for r in rows(out / NAME)]
    assert all(re.fullmatch(r"[0-9]\.[0-9]{3}e[+-][0-9]{2,3}", v) for v in syn)
    assert Counter(syn).most_common(1)[0][1] <= 2  # no collapse onto the minimum
    nums = [float(v) for v in syn]
    assert min(nums) == min(map(float, real)) and max(nums) == max(map(float, real))
    assert sum(float(v) < 1e-100 for v in syn) > 0.1 * len(syn)


def test_excel_general_long_numbers(adv) -> None:
    out, rep, output = adv
    cols = _columns(rep)
    acct = cols["account_no"]
    assert acct["type"] == "FLOAT" and acct["subtype"] == "scientific"
    assert [m["code"] for m in acct["messages"]] == ["excel_scientific_ids"]
    assert "min" not in acct["params"] and "min" not in acct["real"] and "quantiles" not in acct["real"]
    syn = [r["account_no"] for r in rows(out / NAME)]
    assert any(re.fullmatch(r"[0-9]{11}", v) for v in syn) and any("E+11" in v for v in syn)
    real = {r["account_no"] for r in rows(FIX / NAME)}
    rep_text = "\n".join(p.read_text(encoding="utf-8") for p in rep.rglob("*") if p.is_file())
    assert not any(v in rep_text for v in real)


def test_fullwidth_digits_stay_levels(adv) -> None:
    out, rep, _ = adv
    assert _columns(rep)["fw_level"]["type"] == "FACTOR"
    real, syn = [r["fw_level"] for r in rows(FIX / NAME)], [r["fw_level"] for r in rows(out / NAME)]
    assert Counter(real) == Counter(syn) and not any(ch.isascii() for v in syn for ch in v)


def test_ambiguous_grouping_flagged(adv) -> None:
    _, rep, _ = adv
    c = _columns(rep)["count_amb"]
    assert c["type"] == "NUMERIC_FORMATTED" and c["ambiguous"] and "--decimal ," in c["ambiguity"]
    res = run_cli("inspect", FIX / NAME)
    assert res.exit_code == 0 and res.output.isascii() and "AMBIGUOUS" in res.output


def test_private_tokens_only_in_the_data(adv) -> None:
    out, rep, _ = adv
    syn = Counter(r["visits"] for r in rows(out / NAME))
    real = Counter(r["visits"] for r in rows(FIX / NAME))
    from datanonymous.detect.names import stray_name

    for tok in MOD.FAKE_NAME_TOKENS:
        assert real[tok] > 0
        if not stray_name(tok):  # two words on no name list ('Quillon Varga'): no evidence of a name (R3-04)
            assert syn[tok] == real[tok]  # verbatim at the exact count
            continue
        # R4 copy gate: a person's name typed into a number column ('Tess Okonkwo-Lind': known given name and
        # surnames) is a same-shape mask at the token's exact count, never the real name
        assert syn[tok] == 0
        shape = "".join("a" if ch.isalpha() else ch for ch in tok)
        masks = [v for v in syn if "".join("a" if ch.isalpha() else ch for ch in v) == shape]
        assert len(masks) == 1 and syn[masks[0]] == real[tok]
    assert any(stray_name(tok) for tok in MOD.FAKE_NAME_TOKENS)
    texts = [p.read_text(encoding="utf-8") for p in rep.rglob("*") if p.is_file()]
    texts += [(out / d).read_text(encoding="utf-8") for d in ("README.md", "DATA_DICTIONARY.md", "schema.json")]
    for tok in MOD.FAKE_NAME_TOKENS:
        for part in tok.replace("-", " ").split():
            assert not any(part in t for t in texts), part


def test_real_student_ids_never_copied(adv) -> None:
    out, rep, _ = adv
    real = {r["student_id"] for r in rows(FIX / NAME)}
    text = (out / NAME).read_text(encoding="utf-8-sig")
    text += "\n".join(p.read_text(encoding="utf-8") for p in rep.rglob("*") if p.is_file())
    assert not any(i in text for i in real)


def test_same_seed_same_bytes(adv, tmp_path: Path) -> None:
    out, _, _ = adv
    out2, _, _ = _run(tmp_path, "--seed", "11")
    assert (out / NAME).read_bytes() == (out2 / NAME).read_bytes()


def test_adversarial_fixture_is_reproducible(tmp_path: Path) -> None:
    MOD.main(tmp_path)
    for name in ("floats.csv", "suppressed.csv", "decimal_comma.csv", NAME):
        assert (tmp_path / name).read_bytes() == (FIX / name).read_bytes(), name
