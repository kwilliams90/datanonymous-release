"""AREA C end to end: CLI runs on tests/fixtures/text, privacy (nothing real leaks), shapes, determinism."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import run_cli

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "text"
FREE = FIXTURES / "free_text.csv"
NA = frozenset({"", "N/A", "NA", "NULL", "null", "None", "-", "--", "."})

#: column -> expected detected type (subtype)
EXPECTED = {
    "student_id": ("IDENTIFIER", None), "first_name": ("PERSON_NAME", "first"),
    "middle_name": ("PERSON_NAME", "middle"), "last_name": ("PERSON_NAME", "last"),
    "student_name": ("PERSON_NAME", "full"), "student_email": ("EMAIL", None), "parent_email": ("EMAIL", None),
    "parent_phone": ("PHONE", None), "ssn": ("SSN", None), "home_address": ("STREET_ADDRESS", None),
    "city": ("CITY", None), "state": ("STATE", None), "zip": ("ZIP", None), "programs": ("MULTI_SELECT", None),
    "activities": ("MULTI_SELECT", None), "home_lat": ("LAT", None), "home_lon": ("LON", None),
    "teacher_name": ("PERSON_NAME", "full"), "Guardian": ("PERSON_NAME", "full"), "comments": ("FREE_TEXT", None),
    "medical_alert": ("FREE_TEXT", None), "portfolio_url": ("URL", None), "last_login_ip": ("IP", None),
    "locker_code": ("HIGH_CARD_CATEGORICAL", "code"), "career_goal": ("HIGH_CARD_CATEGORICAL", None),
}
#: never-copy columns: no synthetic cell may equal a real cell (case-insensitive, trimmed)
NEVER_COPY = ["first_name", "middle_name", "last_name", "student_name", "student_email", "parent_email",
              "parent_phone", "ssn", "home_address", "city", "teacher_name", "Guardian", "comments", "medical_alert",
              "portfolio_url", "last_login_ip", "locker_code", "career_goal"]


def read(path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    with open(path, encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class FixtureRun:
    inp: Path
    out: Path
    rep: Path
    output: str


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> FixtureRun:
    base = tmp_path_factory.mktemp("area_c")
    inp = base / "in"
    inp.mkdir()
    for name in ("free_text.csv", "names_cp1252.csv"):
        shutil.copyfile(FIXTURES / name, inp / name)
    out, rep = base / "out", base / "rep"
    res = run_cli(inp, "--out-dir", out, "--report-dir", rep, "--seed", "11")
    assert res.exit_code == 0, res.output
    return FixtureRun(inp, out, rep, res.output)


def test_fixture_generator_is_deterministic(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_text_fixtures", FIXTURES / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import random

    rows = mod.make_free_text(random.Random(20261004))
    with open(FREE, encoding="utf-8", newline="") as fh:
        assert list(csv.reader(fh)) == rows


def test_inspect_detects_every_area_c_type() -> None:
    res = run_cli("inspect", FIXTURES, "--json")
    assert res.exit_code == 0, res.output
    files = {f["source"]: f for f in json.loads(res.output)["files"]}
    cols = {c["name"]: c for c in files["free_text.csv"]["tables"][0]["columns"]}
    for name, (t, sub) in EXPECTED.items():
        assert cols[name]["type"] == t, (name, cols[name]["type"], cols[name]["reason"])
        if sub is not None:
            assert cols[name]["subtype"] == sub, name
    cp = {c["name"]: c for c in files["names_cp1252.csv"]["tables"][0]["columns"]}
    assert cp["FirstName"]["type"] == "PERSON_NAME" and cp["LastName"]["type"] == "PERSON_NAME"


def test_same_shape_rows_and_header(fx: FixtureRun) -> None:
    real, syn = read(FREE), read(fx.out / "free_text.csv")
    assert len(syn) == len(real) and list(syn[0]) == list(real[0])
    src, out = FREE.read_bytes(), (fx.out / "free_text.csv").read_bytes()
    assert out.split(b"\n", 1)[0] == src.split(b"\n", 1)[0] and b"\r\n" not in out


def test_no_real_value_in_never_copy_columns(fx: FixtureRun) -> None:
    real, syn = read(FREE), read(fx.out / "free_text.csv")
    for c in NEVER_COPY:
        r = {row[c].strip().casefold() for row in real if row[c].strip() not in NA}
        s = {row[c].strip().casefold() for row in syn if row[c].strip() not in NA}
        assert not r & s, (c, sorted(r & s)[:3])


def test_no_real_first_last_pair_on_a_row(fx: FixtureRun) -> None:
    real, syn = read(FREE), read(fx.out / "free_text.csv")
    pairs = {(r["first_name"].casefold(), r["last_name"].casefold()) for r in real}
    assert not any((s["first_name"].casefold(), s["last_name"].casefold()) in pairs for s in syn)
    full = {r["student_name"].casefold() for r in real}
    assert not any(f"{s['last_name']}, {s['first_name']}".casefold() in full for s in syn)


def _real_sensitive_values() -> set[str]:
    real = read(FREE)
    vals: set[str] = set()
    for row in real:
        for c in NEVER_COPY:
            v = row[c].strip()
            if v not in NA and len(v) >= 4:
                vals.add(v)
    return vals


def test_nothing_real_in_the_out_dir_or_the_report(fx: FixtureRun) -> None:
    vals = _real_sensitive_values()
    pattern = {v: re.compile(r"(?<![\w])" + re.escape(v) + r"(?![\w])") for v in vals}
    docs = [p for p in fx.rep.rglob("*") if p.is_file()] + [p for p in fx.out.iterdir() if p.suffix in (".md", ".json")]
    assert docs
    for path in docs:
        text = path.read_text(encoding="utf-8")
        hits = [v for v, rx in pattern.items() if rx.search(text)]
        assert not hits, (path.name, hits[:5])
    for path in fx.out.glob("*.csv"):
        enc = "cp1252" if "cp1252" in path.name else "utf-8"
        with open(path, encoding=enc, newline="") as fh:
            cells = {c.strip() for row in list(csv.reader(fh))[1:] for c in row}
        assert not vals & cells, (path.name, sorted(vals & cells)[:5])


def test_report_has_aggregates_only_for_pii_columns(fx: FixtureRun) -> None:
    data = json.loads((fx.rep / "files" / "free_text.csv.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in data["tables"][0]["columns"]}
    for c in NEVER_COPY:
        for side in ("real", "synthetic"):
            st = cols[c][side]
            assert "levels" not in st and "mode" not in st and "level_shares" not in st, (c, side)
    em = cols["parent_email"]["real"]
    # D21: real domains are not reproduced in the output, so the report has their counts, never their names
    assert em["domains"] == {} and em["domains_below_min_cell"] >= 1
    assert em["domain_count_profile"] and all(v >= 10 for v in em["domain_count_profile"])
    syn = cols["parent_email"]["synthetic"]
    assert all(re.fullmatch(r"([a-z0-9]+\.)?example\.(org|com|net)", d) for d in syn["domains"])
    assert sorted(syn["domains"].values(), reverse=True) == em["domain_count_profile"]  # same shares by rank
    assert syn["n_domains"] == em["n_domains"]
    assert cols["portfolio_url"]["real"]["hosts"] == {} and cols["portfolio_url"]["real"]["n_hosts"] >= 3
    ph = cols["parent_phone"]
    assert set(ph["real"]["area_codes"]) == {"214", "972"} and 0 <= ph["synthetic"]["exchange_555_share"] < 1.0
    assert ph["real"]["masks"] == ph["synthetic"]["masks"]
    assert cols["ssn"]["synthetic"]["invalid_area_share"] == 1.0
    report_text = "".join(p.read_text(encoding="utf-8") for p in fx.rep.rglob("*") if p.is_file())
    for z in {r["zip"][:5] for r in read(FREE)}:
        assert z not in report_text, z  # ZIP codes are not label-safe (D21)


def test_distinct_counts_frequency_profiles_and_missingness(fx: FixtureRun) -> None:
    real, syn = read(FREE), read(fx.out / "free_text.csv")
    # integration: a column that AREA D rebuilds from other columns (D5 derived template, held in 100% of the
    # source rows: student_name = 'last_name, first_name') keeps the template instead of its own frequency
    # profile; its missingness is still exact
    data = json.loads((fx.rep / "files" / "free_text.csv.json").read_text(encoding="utf-8"))
    templates = {t["column"]: t for t in data["tables"][0]["structure"]["derived_templates"]
                 if t["status"] == "applied"}
    assert set(templates) == {"student_name"}
    for c in NEVER_COPY + ["state", "zip", "programs"]:
        rc = Counter(r[c] for r in real)
        sc = Counter(s[c] for s in syn)
        assert sum(v for k, v in rc.items() if k.strip() in NA) == sum(v for k, v in sc.items() if k.strip() in NA), c
        if c in templates:
            continue
        assert sorted(v for k, v in rc.items() if k.strip() not in NA) == sorted(
            v for k, v in sc.items() if k.strip() not in NA), c
    filled = [s for s in syn if s["student_name"].strip() not in NA]
    assert filled and all(s["student_name"] == f"{s['last_name']}, {s['first_name']}" for s in filled)


def test_teacher_name_three_fake_names_with_exact_counts(fx: FixtureRun) -> None:
    syn = read(fx.out / "free_text.csv")
    c = Counter(s["teacher_name"] for s in syn)
    assert sorted(c.values()) == [50, 90, 100]
    assert {re.match(r"(Mrs|Mr|Ms)\.", k).group(1) for k in c} == {"Mrs", "Mr", "Ms"}
    assert not {"Alvarez", "Whitcomb", "Thibodeaux", "Neil"} & {w for k in c for w in re.findall(r"[A-Za-z]+", k)}


def test_zip_state_latlon_and_multi_select_shapes(fx: FixtureRun) -> None:
    real, syn = read(FREE), read(fx.out / "free_text.csv")
    assert Counter(s["state"] for s in syn) == Counter(r["state"] for r in real)
    sz = Counter(s["zip"][:5] for s in syn)
    rz = Counter(r["zip"][:5] for r in real)
    for z, n in rz.items():
        if n >= 10:
            assert sz[z] == n, z  # kept exactly, incl. the leading zero of 07030
        else:
            assert sz[z] == 0, z
    assert sz["07030"] >= 10
    assert all(re.fullmatch(r"\d{5}(-\d{4})?", s["zip"]) for s in syn)
    assert sum("-" in s["zip"] for s in syn) == sum("-" in r["zip"] for r in real)
    assert Counter(s["programs"] for s in syn) == Counter(r["programs"] for r in real)
    lat = [float(s["home_lat"]) for s in syn]
    rlat = [float(r["home_lat"]) for r in real]
    assert min(rlat) <= min(lat) and max(lat) <= max(rlat)
    assert all(re.fullmatch(r"-?\d+\.\d{6}", s["home_lon"]) for s in syn)
    acts = [s["activities"] for s in syn if s["activities"]]
    items = {"Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate"}
    assert all(set(a.split("|")) <= items for a in acts)


def test_email_and_url_hosts_are_reserved(fx: FixtureRun) -> None:
    syn = read(fx.out / "free_text.csv")
    for s in syn:
        for c in ("student_email", "parent_email"):
            if s[c]:
                assert re.search(r"@([a-z0-9]+\.)?example\.(org|com|net)$", s[c]), s[c]
        if s["portfolio_url"]:
            assert re.match(r"https?://(www\.)?([a-z0-9]+\.)?example\.(org|com|net)/", s["portfolio_url"])


def test_cp1252_names_stay_encodable(fx: FixtureRun) -> None:
    src = (fx.inp / "names_cp1252.csv").read_bytes()
    out = (fx.out / "names_cp1252.csv").read_bytes()
    assert out.count(b"\r\n") == src.count(b"\r\n")
    syn = read(fx.out / "names_cp1252.csv", encoding="cp1252")
    real = read(fx.inp / "names_cp1252.csv", encoding="cp1252")
    assert not {r["FirstName"] for r in real} & {s["FirstName"] for s in syn}
    assert not {r["LastName"] for r in real} & {s["LastName"] for s in syn}
    assert any(not s["FirstName"].isascii() or not s["LastName"].isascii() for s in syn)
    data = json.loads((fx.rep / "report.json").read_text(encoding="utf-8"))
    assert data["totals"]["FAIL"] == 0


def test_same_seed_same_bytes(tmp_path: Path) -> None:
    outs = []
    for k in range(2):
        out = tmp_path / f"o{k}"
        res = run_cli(FREE, "--out-dir", out, "--report-dir", tmp_path / f"r{k}", "--seed", "3")
        assert res.exit_code == 0, res.output
        outs.append((out / "free_text.csv").read_bytes())
    assert outs[0] == outs[1]
    res = run_cli(FREE, "--out-dir", tmp_path / "o9", "--report-dir", tmp_path / "r9", "--seed", "4")
    assert (tmp_path / "o9" / "free_text.csv").read_bytes() != outs[0]


def test_same_bytes_across_processes_with_different_hash_seeds(tmp_path: Path) -> None:
    """D26: no generator depends on Python's per-process string hashing (set order)."""
    import os
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    outs = []
    for hs in ("1", "4242"):
        env = {**os.environ, "PYTHONPATH": str(src), "PYTHONHASHSEED": hs}
        out = tmp_path / f"o{hs}"
        r = subprocess.run([sys.executable, "-m", "datanonymous", "--mode", "fidelity", str(FREE), "--out-dir", str(out), "--report-dir",
                            str(tmp_path / f"r{hs}"), "--seed", "9"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        outs.append((out / "free_text.csv").read_bytes())
    assert outs[0] == outs[1]


def test_within_id_constant_pii_columns(tmp_path: Path) -> None:
    """D4 columns (one value per student) go through the per-entity model: still fake, still constant."""
    real = read(FREE)
    rows = ["student_id,first_name,parent_email,score"]
    for i, r in enumerate(real[:120]):
        for k in range(1 + i % 3):
            rows.append(f"{r['student_id']},{r['first_name']},{r['parent_email']},{(i * 7 + k) % 100}")
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "long.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    out = tmp_path / "o"
    res = run_cli(inp, "--out-dir", out, "--report-dir", tmp_path / "r", "--seed", "5")
    assert res.exit_code == 0, res.output
    syn = read(out / "long.csv")
    by_id: dict[str, set] = {}
    for s in syn:
        by_id.setdefault(s["student_id"], set()).add((s["first_name"], s["parent_email"]))
    assert all(len(v) == 1 for v in by_id.values())  # constant within each synthetic student
    real_first = {r["first_name"].casefold() for r in real}
    real_mail = {r["parent_email"].casefold() for r in real if r["parent_email"]}
    assert not real_first & {s["first_name"].casefold() for s in syn}
    assert not real_mail & {s["parent_email"].casefold() for s in syn}
    inspect = json.loads(run_cli("inspect", inp, "--json").output)
    cols = {c["name"]: c for c in inspect["files"][0]["tables"][0]["columns"]}
    assert cols["first_name"]["role"] == "entity" and cols["first_name"]["type"] == "PERSON_NAME"


def test_overrides_force_factor_and_pii_col(tmp_path: Path) -> None:
    out = tmp_path / "o"
    res = run_cli(FREE, "--out-dir", out, "--report-dir", tmp_path / "r", "--seed", "2",
                  "--force-factor", "state", "--force-factor", "teacher_name", "--pii-col", "career_goal=last_name")
    assert res.exit_code == 0, res.output
    real, syn = read(FREE), read(out / "free_text.csv")
    assert Counter(s["teacher_name"] for s in syn) == Counter(r["teacher_name"] for r in real)  # real labels kept
    careers = {r["career_goal"].casefold() for r in real}
    assert not careers & {s["career_goal"].casefold() for s in syn}
    assert all(re.fullmatch(r"[A-Z][a-z]+( [A-Z][a-z]+)?", s["career_goal"]) for s in syn)
