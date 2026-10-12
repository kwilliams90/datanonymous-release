"""AREA D end to end: the tests/fixtures/ids fixtures through the CLI (IDs, entities, structure, privacy)."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from conftest import REPO, read_dicts, read_rows, run_cli
from datanonymous.detect.ids import luhn_check_digit, luhn_valid, normalize_id

FIX = REPO / "tests" / "fixtures" / "ids"
SEED = "11"


@pytest.fixture(scope="session")
def ids_run(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("area_d")
    out, rep = base / "out", base / "rep"
    res = run_cli(FIX, "--out-dir", out, "--report-dir", rep, "--seed", SEED)
    assert res.exit_code == 0, res.output
    return out, rep


def _report(rep: Path, rel: str) -> dict:
    return json.loads((rep / "files" / (rel + ".json")).read_text(encoding="utf-8"))["tables"][0]


def _by(rows: list[dict], key: str, val: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r[key], set()).add(r[val])
    return out


# --------------------------------------------------------------------------- long format: aliases, nesting, FD, keys
def test_long_ids_structure_kept(ids_run) -> None:
    out, rep = ids_run
    real = read_dicts(FIX / "long_ids.csv")
    syn = read_dicts(out / "long_ids.csv")
    assert len(syn) == len(real)
    # rows per ID identical
    assert Counter(Counter(r["student_id"] for r in real).values()) == Counter(
        Counter(s["student_id"] for s in syn).values())
    # aliases stay 1:1
    for a, b in (("student_id", "state_id"), ("student_id", "student_number"), ("state_id", "student_number")):
        assert all(len(v) == 1 for v in _by(syn, a, b).values()) and all(len(v) == 1 for v in _by(syn, b, a).values())
    # nesting kept: one family per student, same students-per-family profile
    assert all(len(v) == 1 for v in _by(syn, "student_id", "family_ident").values())
    fam_real = Counter(len(v) for v in _by(real, "family_ident", "student_id").values())
    fam_syn = Counter(len(v) for v in _by(syn, "family_ident", "student_id").values())
    assert fam_real == fam_syn
    # within-ID constants (D4)
    for col in ("gender", "race", "dob"):
        assert all(len(v) == 1 for v in _by(syn, "student_id", col).values()), col
    # FD school_id -> school_name with the real pairs (org codes keep real values, D10)
    assert set(zip((s["school_id"] for s in syn), (s["school_name"] for s in syn))) <= set(
        zip((r["school_id"] for r in real), (r["school_name"] for r in real)))
    # composite key (student_id, event_date) unique
    pairs = [(s["student_id"], s["event_date"]) for s in syn]
    assert len(pairs) == len(set(pairs))
    # same shapes, never real
    for col, check in (("student_id", lambda v: len(v) == 7 and v.isdigit()),
                       ("state_id", lambda v: len(v) == 10 and v.isdigit()),
                       ("student_number", lambda v: len(v) == 7 and v[0] == "S" and v[1:].isdigit()),
                       ("family_ident", lambda v: len(v) == 6 and v.isdigit())):
        assert all(check(s[col]) for s in syn), col
        assert not {s[col] for s in syn} & {r[col] for r in real}, col
    t = _report(rep, "long_ids.csv")
    assert t["entity"]["aliases"] == [["student_id", "state_id", "student_number"]]
    assert t["structure"]["functional_dependencies"][0]["text"] == "school_id -> school_name"
    assert t["structure"]["composite_keys"][0]["columns"] == ["student_id", "event_date"]
    assert t["structure"]["panels"][0]["kind"] == "panel"


# --------------------------------------------------------------------------- multi-table: coverage, padding per raw form
def test_multi_table_keys_and_orphans_reproduced(ids_run) -> None:
    out, rep = ids_run
    rs, re_ = read_dicts(FIX / "multi_table" / "students.csv"), read_dicts(FIX / "multi_table" / "enrollments.csv")
    ss, se = read_dicts(out / "multi_table" / "students.csv"), read_dicts(out / "multi_table" / "enrollments.csv")
    # padding per raw form: '0012345' in students, '12345' in enrollments
    assert all(len(s["student_id"]) == 7 and s["student_id"].startswith("00") for s in ss)
    assert all(not e["student_id"].startswith("0") and len(e["student_id"]) == 5 for e in se
               if normalize_id(e["student_id"]) in {normalize_id(s["student_id"]) for s in ss})
    # join rate / orphan rows reproduced exactly (D5 cross-file key coverage)
    def orphans(students, enr, col_s, col_e):
        keys = {normalize_id(s[col_s]) for s in students}
        return sum(normalize_id(e[col_e]) not in keys for e in enr)

    assert orphans(ss, se, "student_id", "student_id") == orphans(rs, re_, "student_id", "student_id") == 15
    assert orphans(ss, se, "StudentNumber", "student_number") == orphans(rs, re_, "StudentNumber", "student_number")
    # the two ID systems stay aliases across both files
    alias = {normalize_id(s["student_id"]): s["StudentNumber"] for s in ss}
    assert all(alias[normalize_id(e["student_id"])] == e["student_number"] for e in se
               if normalize_id(e["student_id"]) in alias)
    # FD and composite key in enrollments
    assert all(len(v) == 1 for v in _by(se, "course_code", "course_title").values())
    pairs = [(e["student_id"], e["course_code"]) for e in se]
    assert len(pairs) == len(set(pairs))
    cov = _report(rep, "multi_table/enrollments.csv")["structure"]["cross_file_keys"]
    by_col = {c["column"]: c for c in cov}
    assert by_col["student_id"]["parent"] == "multi_table/students.csv::student_id"
    assert by_col["student_id"]["orphan_rows"] == 15 and not by_col["student_id"]["foreign_key"]
    # schools.csv: org code -> org code FD kept with real values
    sch = read_dicts(out / "multi_table" / "schools.csv")
    real_sch = {(r["school_id"], r["school_name"]) for r in read_dicts(FIX / "multi_table" / "schools.csv")}
    assert {(r["school_id"], r["school_name"]) for r in sch} == real_sch


def test_cross_file_keys_can_be_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("structure:\n  disable: [cross_file_keys]\n", encoding="utf-8")
    res = run_cli(FIX / "multi_table", "--out-dir", tmp_path / "o", "--seed", SEED, "--config", cfg)
    assert res.exit_code == 0, res.output
    ss, se = read_dicts(tmp_path / "o" / "students.csv"), read_dicts(tmp_path / "o" / "enrollments.csv")
    keys = {normalize_id(s["student_id"]) for s in ss}
    assert sum(normalize_id(e["student_id"]) in keys for e in se) < 10  # files no longer share IDs


# --------------------------------------------------------------------------- layout: sort order, blocks, duplicates
def test_sort_order_and_leading_missing_reproduced(ids_run) -> None:
    out, _ = ids_run
    syn = read_dicts(out / "sorted.csv")
    key = [(s["school_code"], s["exit_date"] != "", s["exit_date"], int(s["student_id"])) for s in syn]
    assert key == sorted(key)
    att = read_dicts(out / "attendance.csv")
    assert [(int(a["student_id"]), a["att_date"]) for a in att] == sorted((int(a["student_id"]), a["att_date"])
                                                                           for a in att)
    pairs = [(a["student_id"], a["att_date"]) for a in att]
    assert len(pairs) == len(set(pairs))
    lm = read_dicts(out / "leading_missing.csv")
    assert [r["new_field"] == "" for r in lm] == [True] * 60 + [False] * 140


def test_duplicate_rows_count_reproduced(ids_run) -> None:
    out, rep = ids_run
    for name, want in (("dups.csv", 15), ("leading_missing.csv", None)):
        real = [tuple(r) for r in read_rows(FIX / name)[1:]]
        syn = [tuple(r) for r in read_rows(out / name)[1:]]
        real_d = len(real) - len(set(real))
        assert want is None or real_d == want
        assert len(syn) - len(set(syn)) == real_d, name
    assert _report(rep, "dups.csv")["structure"]["duplicate_rows"]["rows"] == 15


def test_layout_switches_can_be_disabled(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    for name in ("sorted.csv", "dups.csv"):
        shutil.copy(FIX / name, d / name)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("structure:\n  disable: [sort_order, duplicate_rows]\n", encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--seed", SEED, "--config", cfg)
    assert res.exit_code == 0, res.output
    syn = read_dicts(tmp_path / "o" / "sorted.csv")
    key = [(s["school_code"], s["exit_date"] != "", s["exit_date"], int(s["student_id"])) for s in syn]
    assert key != sorted(key)
    data = json.loads((tmp_path / "o-report" / "files" / "sorted.csv.json").read_text(encoding="utf-8"))
    assert data["tables"][0]["structure"]["sort_order"]["status"].startswith("disabled by config")


# --------------------------------------------------------------------------- derived templates
def test_templates_rebuilt_from_synthetic_inputs(ids_run) -> None:
    out, rep = ids_run
    real = read_dicts(FIX / "templates.csv")
    syn = read_dicts(out / "templates.csv")
    for s in syn:
        f, la = s["first_name"], s["last_name"]
        low = lambda v: "".join(ch for ch in v if ch.isalnum()).lower()  # noqa: E731
        assert s["email"] == f"{low(f)}.{low(la)}.{s['id']}@example.org"
        assert s["username"] == f"{f[0].lower()}{low(la)}"
        assert s["full_name"] == f"{la}, {f}"
    for col in ("first_name", "last_name", "email", "username", "full_name"):
        assert not {s[col] for s in syn} & {r[col] for r in real}, col
    tm = {t["column"]: t for t in _report(rep, "templates.csv")["structure"]["derived_templates"]}
    assert set(tm) == {"email", "username", "full_name"} and all(t["status"] == "applied" for t in tm.values())


def test_example_email_template(example_run) -> None:
    syn = read_dicts(example_run.out / "student-scores.csv")
    low = lambda v: "".join(ch for ch in v if ch.isalnum()).lower()  # noqa: E731
    assert all(s["email"] == f"{low(s['first_name'])}.{low(s['last_name'])}.{s['id']}@example.org" for s in syn)


# --------------------------------------------------------------------------- ID candidates and declared IDs
def test_id_candidates_listed_and_redacted(ids_run) -> None:
    out, rep = ids_run
    t = _report(rep, "ids.csv")
    cols = {c["name"]: c for c in t["columns"]}
    for name in ("record_uuid", "md5", "sha1", "sha256", "row_num", "ticket", "badge"):
        assert cols[name]["params"].get("id_candidate"), name
        assert "redacted" in cols[name]["real"], name
    assert not cols["school_code"]["params"].get("id_candidate")
    text = (rep / "files" / "ids.csv.md").read_text(encoding="utf-8") + (rep / "files" / "ids.csv.json").read_text(
        encoding="utf-8")
    real = read_dicts(FIX / "ids.csv")
    for col in ("local_id", "student_number", "record_uuid", "md5", "ticket", "badge"):
        assert not any(r[col] in text for r in real), col


def test_declared_hex_and_patterned_ids(tmp_path: Path) -> None:
    res = run_cli(FIX / "ids.csv", "--out-dir", tmp_path / "o", "--seed", SEED, "--id-col", "record_uuid",
                  "--id-col", "md5", "--id-col", "ticket", "--id-col", "badge")
    assert res.exit_code == 0, res.output
    real, syn = read_dicts(FIX / "ids.csv"), read_dicts(tmp_path / "o" / "ids.csv")
    hexchars = set("0123456789abcdef")
    for col in ("record_uuid", "md5", "ticket", "badge", "local_id", "student_number"):
        assert not {s[col] for s in syn} & {r[col] for r in real}, col
    assert all(len(s["md5"]) == 32 and set(s["md5"]) <= hexchars for s in syn)
    assert all([p == "-" for p in s["record_uuid"]] == [p == "-" for p in real[0]["record_uuid"]] for s in syn)
    assert all(len(s["ticket"]) == 7 and s["ticket"][2] == "-" and s["ticket"][:2].isupper() for s in syn)
    assert all(len(s["badge"]) == 6 and s["badge"].isdigit() for s in syn)
    # aliases declared together stay 1:1
    assert len({(s["local_id"], s["record_uuid"]) for s in syn}) == len(syn) == len({s["record_uuid"] for s in syn})


def test_id_check_luhn(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    rows = ["card_id,points"]
    seen = set()
    k = 0
    while len(seen) < 150:
        body = str(4_000_000 + 7919 * k)
        k += 1
        full = body + str(luhn_check_digit(body))
        seen.add(full)
        rows.append(f"{full},{k % 9}")
    (d / "cards.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--seed", SEED, "--id-col", "card_id", "--id-check", "luhn")
    assert res.exit_code == 0, res.output
    syn = read_dicts(tmp_path / "o" / "cards.csv")
    assert all(luhn_valid(s["card_id"]) and len(s["card_id"]) == 8 for s in syn)
    assert not {s["card_id"] for s in syn} & seen
    res = run_cli("inspect", d, "--id-col", "card_id", "--id-check", "luhn", "-v")
    assert "Luhn" in res.output
    assert run_cli(d, "--out-dir", tmp_path / "o2", "--id-check", "mod11").exit_code == 2


# --------------------------------------------------------------------------- rules
def test_rules_suggested_then_applied_from_config(ids_run, tmp_path: Path) -> None:
    _, rep = ids_run
    sugg = {r["rule"] for r in _report(rep, "rules.csv")["structure"]["suggested_rules"]}
    assert sugg == {"start_year < grad_year", "credits_earned <= credits_attempted"}
    cfg = tmp_path / "c.yaml"
    cfg.write_text('rules:\n  - "start_year < grad_year"\n  - rule: "credits_attempted >= credits_earned"\n'
                   '    file: rules.csv\n', encoding="utf-8")
    res = run_cli(FIX / "rules.csv", "--out-dir", tmp_path / "o", "--seed", SEED, "--config", cfg)
    assert res.exit_code == 0, res.output
    real = read_dicts(FIX / "rules.csv")
    syn = read_dicts(tmp_path / "o" / "rules.csv")
    assert all(int(s["start_year"]) < int(s["grad_year"]) for s in syn)
    assert all(int(s["credits_earned"]) <= int(s["credits_attempted"]) for s in syn)
    gaps_real = {int(r["grad_year"]) - int(r["start_year"]) for r in real}
    assert {int(s["grad_year"]) - int(s["start_year"]) for s in syn} <= gaps_real
    # the low side keeps its exact distribution; the high side is derived
    assert Counter(s["start_year"] for s in syn) == Counter(r["start_year"] for r in real)
    data = json.loads((tmp_path / "o-report" / "files" / "rules.csv.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in data["tables"][0]["structure"]["declared_rules"]] == ["applied", "applied"]
    bad = tmp_path / "bad.yaml"
    bad.write_text('rules: ["start_year = grad_year"]\n', encoding="utf-8")
    assert run_cli(FIX / "rules.csv", "--out-dir", tmp_path / "o3", "--config", bad).exit_code == 2


# --------------------------------------------------------------------------- inspect, determinism, privacy
def test_inspect_lists_structure(tmp_path: Path) -> None:
    res = run_cli("inspect", FIX)
    assert res.exit_code == 0, res.output
    for text in ("ID aliases (1:1, kept 1:1): student_id = state_id = student_number",
                 "ID nesting: student_id within family_ident", "functional dependency school_id -> school_name",
                 "composite key (student_id, event_date) unique", "derived template email =",
                 "suggested rule start_year < grad_year", "sorted by school_code asc, exit_date asc (missing first)",
                 "duplicate rows: 15", "leading blocks of missing values: new_field (60 rows)",
                 "cross-file key student_id -> multi_table/students.csv::student_id", "aggregate-shaped table",
                 "multi-year snapshots", "panel"):
        assert text in res.output, text


def test_same_seed_same_bytes(ids_run, tmp_path: Path) -> None:
    out, _ = ids_run
    res = run_cli(FIX, "--out-dir", tmp_path / "o", "--seed", SEED, "-q")
    assert res.exit_code == 0
    for p in sorted(out.rglob("*.csv")):
        rel = p.relative_to(out)
        assert (tmp_path / "o" / rel).read_bytes() == p.read_bytes(), rel


def test_byte_identical_across_processes(tmp_path: Path) -> None:
    """Structure application never depends on Python's per-process string hashing (D26)."""
    import os
    import subprocess
    import sys

    d = tmp_path / "in"
    (d / "multi_table").mkdir(parents=True)
    for name in ("long_ids.csv", "templates.csv", "dups.csv", "attendance.csv", "multi_table/students.csv",
                 "multi_table/enrollments.csv"):
        shutil.copy(FIX / name, d / name)
    outs = []
    for hs in ("0", "4242"):
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONHASHSEED=hs)
        out = tmp_path / f"o{hs}"
        r = subprocess.run([sys.executable, "-m", "datanonymous", "--mode", "fidelity", str(d), "--out-dir", str(out), "--seed", "5", "-q"],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(out)
    for p in sorted(outs[0].rglob("*.csv")):
        assert p.read_bytes() == (outs[1] / p.relative_to(outs[0])).read_bytes(), p.name


_SENSITIVE = (("long_ids.csv", ("student_id", "state_id", "student_number", "family_ident", "dob")),
              ("ids.csv", ("local_id", "student_number", "record_uuid", "md5", "sha1", "sha256", "ticket", "badge")),
              ("multi_table/students.csv", ("student_id", "StudentNumber")),
              ("multi_table/enrollments.csv", ("student_id", "student_number")),
              ("attendance.csv", ("student_id",)), ("snapshots.csv", ("student_id",)), ("sorted.csv", ("student_id",)),
              ("templates.csv", ("first_name", "last_name", "email", "username", "full_name")))


def test_ids_never_equal_a_real_id_of_their_namespace(ids_run) -> None:
    """D9: per ID namespace (columns linked by name across files), no synthetic ID is a real one (D8 keys)."""
    out, _ = ids_run
    from datanonymous.hints import normalize

    real: dict[str, set[str]] = {}
    syn: dict[str, set[str]] = {}
    for name, cols in _SENSITIVE:
        for c in cols:
            if c in ("dob", "record_uuid", "md5", "sha1", "sha256", "ticket", "badge", "first_name", "last_name",
                     "email", "username", "full_name"):
                continue
            ns = normalize(c)
            real.setdefault(ns, set()).update(normalize_id(r[c]) for r in read_dicts(FIX / name))
            syn.setdefault(ns, set()).update(normalize_id(r[c]) for r in read_dicts(out / name))
    for ns in real:
        assert not real[ns] & syn[ns], ns


def test_no_real_sensitive_value_in_out_dir_or_report(ids_run) -> None:
    """No real ID/name/e-mail/hash/code value anywhere in the out dir or the report dir (D21).

    Integration (AREA A dates): ``dob`` is a date column, not an ID/name, so the synthetic DATA may hold real
    birth dates: its exact min and max by design (D14, every numeric/date column; README "a unique extreme
    value can equal one real person's value") and interpolated days that fall on a real one by chance
    (design 5.2.2, section 10: person-level dates are never drawn from the observed dates). Raw DOB text
    stays out of the docs and the report; ``test_dob_is_person_level_with_exact_extremes`` checks the rest.
    """
    out, rep = ids_run
    import re

    real: set[str] = set()
    real_dob: set[str] = set()
    for name, cols in _SENSITIVE:
        for r in read_dicts(FIX / name):
            real.update(r[c].strip() for c in cols if len(r[c].strip()) >= 5 and c != "dob")
            if "dob" in cols:
                real_dob.add(r["dob"].strip())
    token = re.compile(r"[0-9A-Za-z][0-9A-Za-z\-]*")
    data_files = {out / name for name, _ in _SENSITIVE} | set(out.rglob("*.csv"))
    for path in [p for p in out.rglob("*") if p.is_file()] + [p for p in rep.rglob("*") if p.is_file()]:
        text = path.read_text(encoding="utf-8")
        found = set(token.findall(text))
        if path.suffix == ".csv":
            for row in read_rows(path):
                found.update(v.strip() for v in row)
        check = real if path in data_files else real | real_dob
        hits = found & check
        hits |= {v for v in check if len(v) >= 7 and not v.isdigit() and v in text}  # dob, e-mails, 'Last, First'
        assert not hits, (path, sorted(hits)[:5])


def test_dob_is_person_level_with_exact_extremes(ids_run) -> None:
    """D4 + D14 + section 10: one DOB per synthetic ID, interpolated (never drawn from the real dates), with
    the exact real min and max present and flagged."""
    out, rep = ids_run
    col = next(c for c in _report(rep, "long_ids.csv")["columns"] if c["name"] == "dob")
    assert col["role"] == "entity" and col["params"]["person_level"] is True and col["params"]["calendar"] is False
    assert col["flags"]["extreme_value_adjusted"] is True
    real = read_dicts(FIX / "long_ids.csv")
    syn = read_dicts(out / "long_ids.csv")
    real_dob = {r["dob"] for r in real}
    by_id = _by(syn, "student_id", "dob")
    assert all(len(v) == 1 for v in by_id.values())
    syn_dob = [next(iter(v)) for v in by_id.values()]

    def key(d: str) -> tuple[str, str, str]:
        m, dd, y = d.split("/")
        return y, m, dd

    lo, hi = min(real_dob, key=key), max(real_dob, key=key)
    assert lo in syn_dob and hi in syn_dob
    assert all(key(lo) <= key(d) <= key(hi) for d in syn_dob)
    # interpolated, not resampled: most synthetic birth dates are days no real student was born on
    assert sum(d in real_dob for d in syn_dob) < len(syn_dob) / 2


def test_templates_and_composite_keys_can_be_disabled(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    for name in ("templates.csv", "attendance.csv"):
        shutil.copy(FIX / name, d / name)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("structure:\n  disable: [derived_templates, composite_keys]\n", encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--seed", SEED, "--config", cfg)
    assert res.exit_code == 0, res.output
    syn = read_dicts(tmp_path / "o" / "templates.csv")
    # integration: email is an EMAIL column (AREA C), so without the template it gets independent synthetic
    # addresses on reserved domains (D24) rather than the foundation's invented labels; the template is gone
    assert all(s["email"].rsplit("@", 1)[-1].endswith(("example.org", "example.com", "example.net")) for s in syn)
    built = [f"{s['first_name'].lower()}.{s['last_name'].lower()}.{s['id']}" for s in syn]
    assert sum(s["email"].split("@")[0] == b for s, b in zip(syn, built)) < len(syn) / 10
    assert sum(s["full_name"] == f"{s['last_name']}, {s['first_name']}" for s in syn) < len(syn) / 10
    att = read_dicts(tmp_path / "o" / "attendance.csv")
    pairs = [(a["student_id"], a["att_date"]) for a in att]
    assert len(pairs) != len(set(pairs))  # nothing keeps (student, date) unique any more


def test_report_json_structure_fields(ids_run) -> None:
    _, rep = ids_run
    t = _report(rep, "sorted.csv")["structure"]
    assert set(t) >= {"functional_dependencies", "derived_templates", "composite_keys", "suggested_rules", "panels",
                      "sort_order", "duplicate_rows", "declared_rules", "cross_file_keys"}
    assert [k["column"] for k in t["sort_order"]["keys"]] == ["school_code", "exit_date", "student_id"]
    md = (rep / "files" / "long_ids.csv.md").read_text(encoding="utf-8")
    assert "ID nesting: student_id within family_ident" in md and "composite key (student_id, event_date)" in md
