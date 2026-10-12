"""Round 5, structure and generation (R5-03, R5-04, R5-05).

R5-03: composite-key enforcement only moved whole rows between ID blocks. When independent generation made too
many copies of a key value - a (course, term) pair drawn 46 times for 40 students who each take every pair
once, or 17 distinct dates for one student's 20 visits - no move could help: duplicates stayed while the key's
status said 'enforced' and the run exited 0. Now a pair key is rebuilt block by block from both columns' values
(each value keeps its count; every block takes its share of each column and pairs them distinctly, Gale-Ryser),
rows and single key values swap between blocks (rows with a blank key included), and an interpolated date that
still repeats takes the nearest free date of its own kind (observed weekdays and grid, row layout). What cannot
be fixed is reported with its count: the status reads 'enforced, but violated in N synthetic row(s)' and the
file is warned about, never 'enforced'.

R5-04: a derived template (email = first.last@domain) was rebuilt from independently generated name parts, so the
e-mail lost its distinct count and frequency profile (two staff members sharing an e-mail, a repeated e-mail
splitting into several) while fidelity passed every invented-value column unconditionally. Now the template's
inputs are regrouped to the source's value groups when the rebuilt profile differs (each group's units share
inputs; each input keeps its synthetic values and counts; which values pair up is shuffled among groups of one
size, not copied from the source), and fidelity compares the distinct count, largest count and values held once
of every invented-value column (names, e-mails, labels, filler text, template targets), allowing for the
difference in filled rows and the rows a template kept independent.

R5-05: declared inequality rules accepted only plain numbers, so 'Paid <= Due' on '$1,234.50' columns failed the
whole file. Rules now read NUMERIC_FORMATTED and NUMERIC_WITH_TOKENS columns through their learned number format
(tokens such as '*' or '<10' are no numbers), move the values verbatim (format and tokens kept) and write any
built value through the column's format; suggestions and the final re-check use the same reading.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.generate import structure as gs
from datanonymous.generate.ids import IdRegistry
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.io import csvio
from datanonymous.model import ColumnProfile, ColumnType
from datanonymous.privacy import matches
from datanonymous.profile import structure as st
from datanonymous.profile.table import profile_table
from datanonymous.report import fidelity

T = ColumnType
FIRST = ["Alice", "Brenda", "Carla", "Denise", "Edgar", "Frank", "Gwen", "Hector", "Irene", "Jasper"]
LAST = ["Smith", "Jones", "Brown", "Davis", "Moore", "Clark", "Lewis", "Young", "Walker", "Hall"]
RESERVED = ("example.org", "example.com", "example.net")


# --------------------------------------------------------------------------- helpers
def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _cli(tmp_path: Path, header: list[str], rows: list[list], *flags: str, seed: int = 42,
         rules: list[str] | None = None, exit_code: int = 0) -> tuple[list[dict[str, str]], dict, str]:
    src = _write_csv(tmp_path / "in" / "data.csv", header, rows)
    args = [str(src), "--out-dir", str(tmp_path / "out"), "--report-dir", str(tmp_path / "rep"),
            "--seed", str(seed), "--force", *flags]
    if rules is not None:
        cfg = tmp_path / "rules.yaml"
        cfg.write_text("rules:\n" + "".join(f"  - {json.dumps(r)}\n" for r in rules), encoding="utf-8")
        args += ["--config", str(cfg)]
    res = CliRunner().invoke(main, ["--mode", "fidelity", *args], catch_exceptions=False)
    assert res.exit_code == exit_code, res.output
    out_path = tmp_path / "out" / "data.csv"
    out = []
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as fh:
            out = list(csv.DictReader(fh))
    report = json.loads((tmp_path / "rep" / "files" / "data.csv.json").read_text(encoding="utf-8"))
    return out, report, res.output


def _codes(report: dict) -> list[str]:
    msgs = list(report.get("messages", []))
    for t in report.get("tables", []):
        msgs += t.get("messages", [])
        for c in t["columns"]:
            msgs += c.get("messages", [])
    return [m.get("code") for m in msgs]


def _column(report: dict, name: str) -> dict:
    return next(c for c in report["tables"][0]["columns"] if c["name"] == name)


def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _profile(text: str, settings: Settings | None = None):
    r = csvio.parse_text_bytes(text.encode("utf-8"))
    return profile_table(r.tables[0], "t.csv", "csv", settings or Settings(mode="fidelity"))


def _gen(text: str, seed: int = 1, settings: Settings | None = None):
    """profile -> generate -> privacy -> finalize, as the pipeline does: (profile, synthetic table, columns)."""
    s = settings or Settings(mode="fidelity")
    tp = _profile(text, s)
    reg = IdRegistry()
    ns = {}
    for i in tp.entity.id_columns:
        reg.register(tp.columns[i].name, "t.csv", None, i, tp.columns[i].name, tp.private.id_raw[i])
        ns[i] = tp.columns[i].name
    reg.build(seed)
    synth = generate_table(tp, GenContext(seed, "t.csv", reg, ns, s))
    matches.enforce(tp, synth, np.random.default_rng(seed), None)
    synth.messages += st.finalize_rows(tp, synth, lambda purpose: np.random.default_rng(seed + 1), s)
    return tp, synth, {c.name: c.values.tolist() for c in synth.columns}


def _dups(cols: dict[str, list], names: list[str]) -> int:
    rows = [tuple(cols[n][i] for n in names) for i in range(len(cols[names[0]]))]
    rows = [r for r in rows if all(str(x).strip() for x in r)]
    return len(rows) - len(set(rows))


def _profile_of(values) -> list[int]:
    return sorted(Counter(v for v in values if str(v).strip()).values(), reverse=True)


# =========================================================================== R5-03: composite keys
def _full_panel(n_students: int = 40) -> list[list]:
    return [[f"S{1000 + s}", f"C{c:02d}", t] for s in range(n_students) for c in range(3) for t in range(1, 5)]


@pytest.mark.parametrize("seed", [42, 1, 2])
def test_codex_composite_pair_reproduction(tmp_path: Path, seed: int) -> None:
    """Codex composite_pair: 40 students x 3 courses x 4 terms (480 unique keys). Was: 42 duplicate keys, exit 0,
    'enforced'. Now no duplicate, both columns keep their exact counts, status 'enforced'."""
    rows = _full_panel()
    out, report, _ = _cli(tmp_path, ["student_id", "course", "term"], rows, seed=seed)
    keys = [(r["student_id"], r["course"], r["term"]) for r in out]
    assert len(keys) == 480 and len(set(keys)) == 480
    assert Counter(r["course"] for r in out) == Counter(r[1] for r in rows)
    assert Counter(r["term"] for r in out) == Counter(str(r[2]) for r in rows)
    ck = report["tables"][0]["structure"]["composite_keys"]
    assert [k["status"] for k in ck] == ["enforced"]
    assert "composite_key_duplicates" not in _codes(report)


def test_codex_one_student_visits_reproduction(tmp_path: Path) -> None:
    """Codex one_student_visits: one student, 20 visits on 20 days. Was: 17 distinct synthetic days (4 duplicate
    keys). Now 20 distinct days inside the source range, exact min and max kept (D14)."""
    rows = [["S1234", f"2024-03-{d:02d}"] for d in range(1, 21)]
    out, report, _ = _cli(tmp_path, ["student_id", "visit_date"], rows)
    days = [r["visit_date"] for r in out]
    assert len(set(days)) == 20
    assert min(days) == "2024-03-01" and max(days) == "2024-03-20"
    assert all(re.fullmatch(r"2024-03-\d\d", d) for d in days)
    assert [k["status"] for k in report["tables"][0]["structure"]["composite_keys"]] == ["enforced"]
    assert "composite_key_duplicates" not in _codes(report)


def _variant(kind: str) -> tuple[list[str], list[list], list[str]]:
    rng = np.random.default_rng(0)
    if kind == "partial_panel":
        rows = []
        for s in range(60):
            for c in rng.choice(12, size=int(rng.integers(2, 8)), replace=False):
                for t in rng.choice(["F", "S", "U"], size=int(rng.integers(1, 4)), replace=False):
                    rows.append([f"S{2000 + s}", f"C{c:02d}", t])
        return ["student_id", "course", "term"], rows, ["student_id", "course", "term"]
    if kind == "skewed_panel":
        rows = []
        for s in range(50):
            rows += [[f"S{3000 + s}", "MATH", t] for t in ["T1", "T2", "T3", "T4", "T5", "T6"]]
            for t in rng.choice(["T1", "T2", "T3", "T4", "T5", "T6"], size=2, replace=False):
                rows.append([f"S{3000 + s}", str(rng.choice(["ART", "PE"])), t])
        return ["student_id", "course", "term"], rows, ["student_id", "course", "term"]
    if kind == "date_period_panel":
        rows = [[f"S{5000 + s}", f"2024-09-{d:02d}", p] for s in range(30) for d in range(2, 7) for p in range(1, 5)]
        return ["student_id", "att_date", "period"], rows, ["student_id", "att_date", "period"]
    if kind == "few_students_dates":
        rows = [[f"S{100 + s}", f"2024-{m:02d}-{d:02d}"] for s in range(3) for m in (3, 4) for d in range(2, 28, 2)]
        return ["student_id", "visit_date"], rows, ["student_id", "visit_date"]
    if kind == "datetime_minutes":
        rows = [["S9001", f"2024-03-01 08:{m:02d}"] for m in range(40)]
        return ["student_id", "ts"], rows, ["student_id", "ts"]
    if kind == "levels_and_blank_rows":
        rows = []
        for s in range(30):
            rows += [[f"S{7000 + s}", t, str(j)] for j, t in enumerate(["F", "S", "U"])]
            rows.append([f"S{7000 + s}", "", "9"])
        return ["student_id", "term", "k"], rows, ["student_id", "term"]
    if kind == "constant_and_key":
        rows = []
        for s in range(50):
            g = str(rng.choice(["M", "F"]))
            for t in rng.choice(["F", "S", "U", "W"], size=int(rng.integers(1, 5)), replace=False):
                rows.append([f"S{6000 + s}", g, t, str(int(rng.integers(0, 100)))])
        return ["student_id", "gender", "term", "score"], rows, ["student_id", "term"]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["partial_panel", "skewed_panel", "date_period_panel", "few_students_dates",
                                  "datetime_minutes", "levels_and_blank_rows", "constant_and_key"])
def test_composite_key_variants_hold_with_exact_counts(kind: str) -> None:
    """Pair keys (full, partial, skewed, date x period panels), single date keys of one or few IDs, a datetime
    key whose dates are drawn from the observed ones (only its time moves), level keys with blank-key rows, and
    keys beside within-ID constants: no ID repeats a key, level columns keep their exact counts."""
    header, rows, key = _variant(kind)
    text = _csv_text(header, rows)
    real = {h: [str(r[i]) for r in rows] for i, h in enumerate(header)}
    for seed in (1, 2, 3, 4):
        tp, synth, out = _gen(text, seed)
        assert [tuple(k["columns"]) for k in tp.structure.composite_keys] == [tuple(key)], kind
        assert _dups(out, key) == 0, (kind, seed)
        assert [k["status"] for k in tp.structure.composite_keys] == ["enforced"]
        assert "composite_key_duplicates" not in {m.code for m in synth.messages}
        for c in key[1:]:
            if tp.columns[header.index(c)].type in (T.ORG_CODE, T.FACTOR, T.ORDINAL, T.INTEGER):
                assert Counter(out[c]) == Counter(real[c]), (kind, c, seed)  # values only rearranged


def test_pair_key_rebuild_keeps_fds_and_blank_patterns() -> None:
    """A key column that determines another (course -> course_name) moves with it; a column blank in some rows
    keeps its blank count; the key holds."""
    names = {"C00": "Algebra", "C01": "Biology", "C02": "Chemistry"}
    rng = np.random.default_rng(1)
    rows = [[f"S{1000 + s}", c, names[c], t, "" if rng.random() < 0.1 else str(rng.choice(["A", "B", "C"]))]
            for s in range(40) for c in names for t in range(1, 5)]
    text = _csv_text(["student_id", "course", "course_name", "term", "grade"], rows)
    for seed in (1, 2, 3):
        tp, _, out = _gen(text, seed)
        assert [f["text"] for f in tp.structure.functional_dependencies] == ["course -> course_name"]
        assert _dups(out, ["student_id", "course", "term"]) == 0
        assert all(names[c] == n for c, n in zip(out["course"], out["course_name"]))
        assert sum(1 for g in out["grade"] if not g) == sum(1 for r in rows if not r[4])


def test_redrawn_dates_keep_observed_weekdays_and_range() -> None:
    """Two students on school days only (Mon-Fri): a repeated date moves to a free school day in the range."""
    import datetime as dt

    days = [dt.date(2024, 9, 2) + dt.timedelta(days=i) for i in range(60)]
    wk = [d.isoformat() for d in days if d.weekday() < 5]
    rows = [["S7001", d] for d in wk[:30]] + [["S7002", d] for d in wk[5:35]]
    text = _csv_text(["student_id", "visit_date"], rows)
    for seed in (1, 2, 3):
        _, _, out = _gen(text, seed)
        assert _dups(out, ["student_id", "visit_date"]) == 0
        got = [dt.date.fromisoformat(d) for d in out["visit_date"]]
        assert all(d.weekday() < 5 for d in got)
        assert min(got).isoformat() == wk[0] and max(got).isoformat() == wk[34]


def test_composite_rebuild_is_deterministic(tmp_path: Path) -> None:
    rows = _full_panel(20)
    a, _, _ = _cli(tmp_path / "a", ["student_id", "course", "term"], rows, seed=7)
    b, _, _ = _cli(tmp_path / "b", ["student_id", "course", "term"], rows, seed=7)
    assert a == b


def test_unfixable_key_is_never_reported_enforced(tmp_path: Path, monkeypatch) -> None:
    """When the repeats cannot be removed (simulated: every repair disabled), the status gives their count, the
    file is warned about (exit 0, WARN), and the warning names the count - never a silent 'enforced'."""
    monkeypatch.setattr(gs, "_construct_pairs", lambda *a, **k: False)
    monkeypatch.setattr(gs, "_swap_repair", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_redraw_key_values", lambda *a, **k: None)
    out, report, _ = _cli(tmp_path, ["student_id", "course", "term"], _full_panel())
    dups = len(out) - len({(r["student_id"], r["course"], r["term"]) for r in out})
    assert dups > 0
    ck = report["tables"][0]["structure"]["composite_keys"][0]
    assert ck["status"] == f"enforced, but violated in {dups} synthetic row(s)"
    assert ck["synthetic_rows_violating"] == dups
    assert report["status"] == "WARN"
    texts = [m["text"] for m in report["tables"][0]["messages"] if m["code"] == "composite_key_duplicates"]
    texts += [m["text"] for c in report["tables"][0]["columns"] for m in c["messages"]
              if m["code"] == "composite_key_duplicates"]
    assert texts and any(t.startswith(f"{dups} row(s)") for t in texts)


def test_status_after_privacy_repair_reflects_the_final_rows() -> None:
    """finalize's repair sets the final status: 'enforced' when the repair succeeds, the count otherwise."""
    rows = [r + [str(1000 + 37 * i)] for i, r in enumerate(_full_panel(10))]  # a score: rows rarely equal real ones
    text = _csv_text(["student_id", "course", "term", "score"], rows)
    tp, synth, out = _gen(text, 3)
    assert tp.structure.composite_keys[0]["status"] == "enforced"
    columns = {c.index: c.values for c in synth.columns}
    # a privacy-style swap of one key column between two ID blocks repeats a pair in both: repaired
    blk = tp.private.entity_index
    r = int(np.flatnonzero(blk == blk[0])[0])
    s = next(int(x) for x in np.flatnonzero(blk != blk[0])
             if columns[1][x] != columns[1][r] and columns[2][x] != columns[2][r])
    columns[1][r], columns[1][s] = columns[1][s], columns[1][r]
    assert _dups({"i": blk.tolist(), "a": columns[1].tolist(), "b": columns[2].tolist()}, ["i", "a", "b"]) > 0
    msgs = gs.repair_composite_keys(tp, columns, np.random.default_rng(0), gs._RealRowChecker(tp, columns))
    assert tp.structure.composite_keys[0]["status"] == "enforced" and not msgs
    # a repeat no swap can remove (a value overwritten: one pair now has 11 copies for 10 students) is counted
    columns[1][r], columns[2][r] = columns[1][s], columns[2][s]
    msgs = gs.repair_composite_keys(tp, columns, np.random.default_rng(0), gs._RealRowChecker(tp, columns))
    assert tp.structure.composite_keys[0]["status"] == "enforced, but violated in 1 synthetic row(s)"
    assert [m.code for m in msgs] == ["composite_key_duplicates"] and "after the privacy swaps" in msgs[0].text


def test_gale_ryser_precheck() -> None:
    codes = np.array([0] * 5 + [1] * 3)
    assert gs._gale_ryser_fails(codes, np.array([0, 1, 2, 3, 0, 1, 2, 3]))  # value 0 five times, four blocks
    assert not gs._gale_ryser_fails(codes, np.array([0, 1, 2, 3, 4, 0, 1, 2]))


# =========================================================================== R5-04: derived templates
def _entity_email_rows() -> list[list]:
    """Codex derived_entity_emails: 8 staff, each with one first.last e-mail on 15 rows."""
    rows = []
    for i in range(8):
        f, ln = ["Alice", "Alice", "Brenda", "Brenda", "Carla", "Carla", "Denise", "Denise"][i], \
            ["Smith", "Jones", "Jones", "Brown", "Brown", "Davis", "Davis", "Smith"][i]
        rows += [[f"T{1000 + i}", f, ln, f"{f.lower()}.{ln.lower()}@district.edu", str(7 * k)] for k in range(15)]
    return rows


def _check_email(out: list[dict[str, str]], first: str = "first_name", last: str = "last_name") -> None:
    for r in out:
        local, _, dom = r["email"].partition("@")
        assert local == f"{r[first].lower()}.{r[last].lower()}" and dom in RESERVED


@pytest.mark.parametrize("seed", [2, 42, 1])
def test_codex_derived_entity_emails_reproduction(tmp_path: Path, seed: int) -> None:
    """Was (seed 2): 7 e-mails with counts [15 x 6, 30] - two staff members shared one. Now 8 e-mails of 15 rows,
    one per staff ID, the template holds and fidelity checks (and passes) the profile."""
    rows = _entity_email_rows()
    out, report, _ = _cli(tmp_path, ["staff_id", "first_name", "last_name", "email", "Score"], rows, seed=seed)
    assert _profile_of(r["email"] for r in out) == [15] * 8
    by_staff = {}
    for r in out:
        by_staff.setdefault(r["staff_id"], set()).add(r["email"])
    assert all(len(v) == 1 for v in by_staff.values()) and len({e for v in by_staff.values() for e in v}) == 8
    _check_email(out)
    for c in ("first_name", "last_name"):
        assert _profile_of(r[c] for r in out) == _profile_of(r[["first_name", "last_name"].index(c) + 1]
                                                             for r in rows)
    fid = _column(report, "email")["fidelity"]
    assert fid["verdict"] == "ok" and any(ch["check"] == "distinct" and "deviation" in ch for ch in fid["checks"])
    real_vals = {v for r in rows for v in r[1:4]}
    assert not real_vals & {v for r in out for v in (r["first_name"], r["last_name"], r["email"])}


@pytest.mark.parametrize("seed", [2, 42])
def test_codex_derived_email_repeats_reproduction(tmp_path: Path, seed: int) -> None:
    """No ID column: 8 e-mails repeated 15 times each. Was: 16 e-mails of 5-11 rows. Now 8 of 15, the name
    columns keep their exact profiles and the template holds on every row."""
    rows = []
    for i in range(8):
        f, ln = ["Alice", "Alice", "Brenda", "Brenda", "Carla", "Carla", "Denise", "Denise"][i], \
            ["Smith", "Jones", "Jones", "Brown", "Brown", "Davis", "Davis", "Smith"][i]
        rows += [[f, ln, f"{f.lower()}.{ln.lower()}@district.edu", str(8 * k + i)] for k in range(15)]
    out, report, _ = _cli(tmp_path, ["first_name", "last_name", "email", "Score"], rows, seed=seed)
    assert _profile_of(r["email"] for r in out) == [15] * 8
    assert _profile_of(r["first_name"] for r in out) == [30] * 4
    assert _profile_of(r["last_name"] for r in out) == [30] * 4
    _check_email(out)
    assert _column(report, "email")["fidelity"]["verdict"] == "ok"


def _tpl_variant(kind: str) -> tuple[list[str], list[list], str, list[str]]:
    rng = np.random.default_rng(5)
    if kind == "entity_unequal_blocks":
        rows: list[list] = []
        used: set = set()
        for i in range(30):
            while True:
                f, ln = FIRST[int(rng.integers(10))], LAST[int(rng.integers(10))]
                if (f, ln) not in used:
                    used.add((f, ln))
                    break
            rows += [[f"T{2000 + i}", f, ln, f"{f.lower()}.{ln.lower()}@district.edu", str(int(rng.integers(99)))]
                     for _ in range(int(rng.integers(1, 9)))]
        return ["staff_id", "first_name", "last_name", "email", "Score"], rows, "email", ["first_name", "last_name"]
    if kind == "digits_and_domains":
        rows = []
        for i in range(12):
            f, ln = FIRST[i % 6], LAST[i // 2 % 6]
            dig = str(int(rng.integers(1, 99))) if i % 3 else ""
            dom = "district.edu" if i % 4 else "school.org"
            rows += [[f, ln, f"{f[0].lower()}{ln.lower()}{dig}@{dom}", str(k)] for k in range(10)]
        return ["first_name", "last_name", "email", "Score"], rows, "email", ["first_name", "last_name"]
    if kind == "full_name":
        rows = []
        for i in range(10):
            f, ln = FIRST[i // 2], LAST[(i * 3) % 7]
            rows += [[f, ln, f"{f} {ln}", str(k)] for k in range(int(rng.integers(5, 15)))]
        return ["first_name", "last_name", "full_name", "Score"], rows, "full_name", ["first_name", "last_name"]
    if kind == "shared_inputs":
        rows = []
        for i in range(10):
            f, ln = FIRST[i // 2], LAST[(i * 3) % 7]
            rows += [[f, ln, f"{f} {ln}", f"{f.lower()}.{ln.lower()}@district.edu", str(k)] for k in range(12)]
        return (["first_name", "last_name", "full_name", "email", "Score"], rows, "email",
                ["first_name", "last_name"])
    if kind == "blank_inputs":
        rows = []
        for i in range(10):
            f, ln = FIRST[i // 2], LAST[(i * 3) % 7]
            for k in range(12):
                rows.append(["", ln, "N/A", str(k)] if (k == 0 and i % 3 == 0)
                            else [f, ln, f"{f.lower()}.{ln.lower()}@district.edu", str(k)])
        return ["first_name", "last_name", "email", "Score"], rows, "email", ["first_name", "last_name"]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["entity_unequal_blocks", "digits_and_domains", "full_name", "shared_inputs",
                                  "blank_inputs"])
def test_template_variants_keep_the_target_profile(kind: str) -> None:
    """Unequal ID blocks (e-mails per ID and per row kept), digit suffixes and several domains, a full-name
    target, two templates on the same inputs, blank inputs with an N/A placeholder: the target keeps its distinct
    count and frequency profile, row-level inputs their exact profiles, every template holds, no real value."""
    header, rows, target, inputs = _tpl_variant(kind)
    text = _csv_text(header, rows)
    real = {h: [str(r[i]) for r in rows] for i, h in enumerate(header)}
    for seed in range(1, 6):
        tp, synth, out = _gen(text, seed)
        assert all(t["status"] == "applied" for t in tp.structure.derived_templates)
        assert _profile_of(out[target]) == _profile_of(real[target]), (kind, seed)
        if "staff_id" in header:
            per_id = {}
            for sid, v in zip(out["staff_id"], out[target]):
                per_id.setdefault(sid, set()).add(v)
            assert all(len(v) == 1 for v in per_id.values())
        else:
            for c in inputs:
                assert _profile_of(out[c]) == _profile_of(real[c]), (kind, c, seed)
        for i in range(len(out[target])):
            f, ln, v = out["first_name"][i], out["last_name"][i], out[target][i]
            if v == "N/A":
                continue
            if target == "full_name" or (kind == "shared_inputs" and target == "full_name"):
                assert v == f"{f} {ln}"
            elif kind == "digits_and_domains":
                local, _, dom = v.partition("@")
                assert re.fullmatch(re.escape(f[0].lower() + ln.lower()) + r"\d*", local) and dom in RESERVED
            else:
                assert v.partition("@")[0] == f"{f.lower()}.{ln.lower()}" and v.partition("@")[2] in RESERVED
        if kind == "shared_inputs":
            assert all(out["full_name"][i] == f"{out['first_name'][i]} {out['last_name'][i]}"
                       for i in range(len(out["full_name"])))
            assert _profile_of(out["full_name"]) == _profile_of(real["full_name"])
        assert not set(out[target]) & (set(real[target]) - {"N/A"})


def test_digit_suffixes_follow_the_group(tmp_path: Path) -> None:
    """A repeated e-mail with digits keeps one suffix per group (the digits are drawn per group, not per row)."""
    header, rows, target, _ = _tpl_variant("digits_and_domains")
    _, _, out = _gen(_csv_text(header, rows), 3)
    by_name: dict = {}
    for f, ln, v in zip(out["first_name"], out["last_name"], out[target]):
        by_name.setdefault((f, ln), set()).add(v)
    assert len(by_name) == 12 and all(len(v) == 1 for v in by_name.values())
    real_domains = Counter(str(r[2]).partition("@")[2] for r in rows)
    syn_domains = Counter(v.partition("@")[2] for v in out[target])
    assert sorted(real_domains.values()) == sorted(syn_domains.values())


def test_template_with_an_id_input_is_not_regrouped(monkeypatch) -> None:
    """email = first.last.id (unique per row, as in example-data student-scores): the ID makes every value
    distinct, so the inputs are never regrouped."""
    calls = []
    orig = gs.regroup_template
    monkeypatch.setattr(gs, "regroup_template", lambda *a, **k: calls.append(1) or orig(*a, **k))
    rows = [[i + 1, FIRST[i % 10], LAST[(i * 7) % 10],
             f"{FIRST[i % 10].lower()}.{LAST[(i * 7) % 10].lower()}.{i + 1}@x.com", i % 5] for i in range(200)]
    tp, _, out = _gen(_csv_text(["id", "first_name", "last_name", "email", "k"], rows), 1)
    assert not calls and "groups" not in tp.structure.private["templates"][0]
    assert len(set(out["email"])) == 200


def test_template_groups_hold_ranks_not_values() -> None:
    """The private group data holds counts, lengths, domain ranks and input frequency ranks - never a value."""
    tp = _profile(_csv_text(["staff_id", "first_name", "last_name", "email", "Score"], _entity_email_rows()))
    g = tp.structure.private["templates"][0]["groups"]
    assert g["unit"] == "entity" and list(g["units"]) == [1] * 8 and list(g["unit_size"]) == [15] * 8
    assert all(np.issubdtype(np.asarray(v).dtype, np.integer) for k, v in g.items() if k not in ("unit", "inputs"))


def test_fidelity_detects_a_merged_or_split_template(tmp_path: Path, monkeypatch) -> None:
    """With the regroup turned off (the old behaviour), seed 2 merges two staff e-mails: fidelity now flags it
    (it used to pass every invented-value column unconditionally)."""
    monkeypatch.setattr(gs, "_profile_differs", lambda *a, **k: False)
    for seed in range(1, 30):  # the first seed whose independent name pairs merge two e-mails
        out, report, _ = _cli(tmp_path / str(seed), ["staff_id", "first_name", "last_name", "email", "Score"],
                              _entity_email_rows(), seed=seed)
        if _profile_of(r["email"] for r in out) != [15] * 8:
            break
    else:
        raise AssertionError("no seed merged two e-mails")
    fid = _column(report, "email")["fidelity"]
    assert fid["verdict"] == "drift" and "distinct" in fid["reason"]
    assert "fidelity_drift" in _codes(report)


def _never_copy(t: ColumnType = T.EMAIL, role: str = "row", **params) -> ColumnProfile:
    p = ColumnProfile(index=0, name="email", type=t)
    p.role = role
    p.params.update(params)
    return p


def _stats(n: int, missing: int, distinct: int, max_count: int, singletons: int) -> dict:
    return {"n": n, "missing": missing, "missing_rate": missing / n, "distinct": distinct,
            "frequency_profile": {"max_count": max_count, "singletons": singletons}}


def test_fidelity_distinct_check_units() -> None:
    real = _stats(120, 0, 8, 15, 0)
    ok = fidelity.compare_column(_never_copy(), real, _stats(120, 0, 8, 15, 0))
    assert ok["verdict"] == "ok"
    merged = fidelity.compare_column(_never_copy(), real, _stats(120, 0, 7, 30, 0))
    assert merged["verdict"] == "drift" and "distinct" in merged["reason"]
    split = fidelity.compare_column(_never_copy(T.PERSON_NAME), real, _stats(120, 0, 16, 11, 0))
    assert split["verdict"] == "drift"
    # two fewer filled rows (sampled missingness) may lose up to two values; rows a template kept independent too
    big = _stats(1000, 0, 8, 125, 0)
    assert fidelity.compare_column(_never_copy(), big, _stats(1000, 2, 7, 125, 0))["verdict"] == "ok"
    assert fidelity.compare_column(_never_copy(), big, _stats(1000, 2, 5, 125, 0))["verdict"] == "drift"
    assert fidelity.compare_column(_never_copy(_template_rows_kept=3), real,
                                   _stats(120, 0, 10, 15, 2))["verdict"] == "ok"
    # IDs, row-masked digit codes and multi-select items only report; a within-ID constant's rows are judged
    # per ID (its entity comparison), not per row
    for p in (_never_copy(T.IDENTIFIER), _never_copy(T.DIGIT_CODE), _never_copy(T.MULTI_SELECT)):
        assert fidelity.compare_column(p, real, _stats(120, 0, 50, 5, 20))["verdict"] == "ok"
    assert fidelity.compare_column(_never_copy(role="entity"), real, _stats(120, 0, 50, 5, 20),
                                   block_share=0.1, n_eff=8.0)["verdict"] == "ok"
    rows_fid = fidelity.compare_column(_never_copy(role="entity"), real, _stats(120, 0, 50, 5, 20), basis="rows")
    assert rows_fid["verdict"] == "ok"
    ent = fidelity.compare_column(_never_copy(role="entity"), real, _stats(120, 0, 7, 30, 0), basis="entities")
    assert ent["verdict"] == "drift"


def test_example_template_target_passes_the_new_check(example_run) -> None:
    """example-data student-scores: email = first.last.id@domain stays unchanged and passes the profile check."""
    rep = json.loads((example_run.rep / "files" / "student-scores.csv.json").read_text(encoding="utf-8"))
    for name in ("first_name", "last_name", "email"):
        fid = _column(rep, name)["fidelity"]
        assert fid["verdict"] == "ok", (name, fid["reason"])
        assert any(ch["check"] == "distinct" and "deviation" in ch for ch in fid["checks"])


# =========================================================================== R5-05: rules on formatted numbers
def _money(s: str, comma: bool = False) -> float | None:
    s = s.strip()
    if not re.search(r"\d", s) or s.startswith("<"):
        return None
    neg = s.startswith("(") or s.startswith("-")
    t = re.sub(r"[^\d,.]", "", s)
    t = t.replace(".", "").replace(",", ".") if comma else t.replace(",", "")
    v = float(t)
    return -v if neg else v


def _rule_holds(out: list[dict[str, str]], lo: str, hi: str, strict: bool = False, comma: bool = False) -> bool:
    for r in out:
        a, b = _money(r[lo], comma), _money(r[hi], comma)
        if a is None or b is None:
            continue
        if (b <= a) if strict else (b < a):
            return False
    return True


def test_codex_formatted_rule_reproduction(tmp_path: Path) -> None:
    """Codex formatted_rule: Paid $100.00-$457.00 <= Due $150.00-$626.00 in all 120 rows. Was: exit 3 ('both
    columns must be numbers'). Now the rule is applied, holds in every row, and both columns keep '$x.00'."""
    rows = [[f"S{1000 + i}", f"${100 + 3 * i:.2f}", f"${150 + 4 * i:.2f}"] for i in range(120)]
    out, report, _ = _cli(tmp_path, ["student_id", "Paid", "Due"], rows, rules=["Paid <= Due"])
    assert len(out) == 120 and _rule_holds(out, "Paid", "Due")
    assert all(re.fullmatch(r"\$\d+\.00", r[c]) for r in out for c in ("Paid", "Due"))
    assert [d["status"] for d in report["tables"][0]["structure"]["declared_rules"]] == ["applied"]


def _rule_variant(kind: str) -> tuple[list[str], list[list], str, str, bool, str | None, tuple[str, ...]]:
    rng = np.random.default_rng(3)
    if kind == "thousands_cents":
        rows = []
        for i in range(150):
            a = float(rng.uniform(500, 5000))
            rows.append([f"S{1000 + i}", f"${a:,.2f}", f"${a + float(rng.uniform(0, 3000)):,.2f}"])
        return ["student_id", "Paid", "Due"], rows, "Paid", "Due", False, r"\$\d{1,3}(,\d{3})*\.\d\d", ()
    if kind == "percent_strict":
        rows = []
        for i in range(150):
            a = float(rng.uniform(0, 80))
            rows.append([f"S{1000 + i}", f"{a:.1f}%", f"{a + float(rng.uniform(0.5, 20)):.1f}%"])
        return ["student_id", "pre", "post"], rows, "pre", "post", True, r"\d+\.\d%", ()
    if kind == "suppression_tokens":
        rows = []
        for i in range(200):
            a = int(rng.integers(10, 400))
            lo = "*" if i % 17 == 0 else ("<10" if i % 23 == 0 else f"{a:,}")
            hi = "*" if i % 19 == 0 else f"{a + int(rng.integers(0, 2000)):,}"
            rows.append([f"S{1000 + i}", lo, hi])
        return (["school", "proficient", "tested"], rows, "proficient", "tested", False, r"\*|<10|\d{1,3}(,\d{3})*",
                ())
    if kind == "formatted_vs_plain":
        rows = [[f"S{1000 + i}", f"${10 * i:,}", str(10 * i + int(rng.integers(0, 50)))] for i in range(120)]
        return ["student_id", "fee", "cap"], rows, "fee", "cap", False, None, ()
    if kind == "accounting_negatives":
        rows = []
        for i in range(150):
            a = float(rng.uniform(-500, 500))
            b = a + float(rng.uniform(0, 300))
            rows.append([f"S{1000 + i}"] + [f"(${-v:,.2f})" if v < 0 else f"${v:,.2f}" for v in (a, b)])
        return ["student_id", "low", "high"], rows, "low", "high", False, r"\(?\$\d{1,3}(,\d{3})*\.\d\d\)?", ()
    if kind == "within_id_constant":
        rows = []
        for s in range(60):
            a = float(rng.uniform(100, 900))
            b = a + float(rng.uniform(0, 400))
            rows += [[f"S{1000 + s}", f"${a:,.2f}", f"${b:,.2f}", str(int(rng.integers(0, 100)))]
                     for _ in range(int(rng.integers(1, 5)))]
        return ["student_id", "Paid", "Due", "score"], rows, "Paid", "Due", False, r"\$\d{1,3}(,\d{3})*\.\d\d", ()
    if kind == "decimal_comma":
        rows = []

        def eu(v: float) -> str:
            return f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

        for i in range(150):
            a = float(rng.uniform(100, 5000))
            rows.append([f"S{1000 + i}", eu(a), eu(a + float(rng.uniform(0, 3000)))])
        return ["student_id", "lo", "hi"], rows, "lo", "hi", False, None, ("--decimal", ",")
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["thousands_cents", "percent_strict", "suppression_tokens", "formatted_vs_plain",
                                  "accounting_negatives", "within_id_constant", "decimal_comma"])
def test_formatted_rule_variants(tmp_path: Path, kind: str) -> None:
    """Thousands and cents, a strict rule on percentages, suppression tokens in both columns ('*', '<10': not
    numbers, kept verbatim at their counts), a formatted low column against a plain integer high one, accounting
    negatives, a within-ID-constant pair and a decimal comma: the rule holds and every value keeps its format."""
    header, rows, lo, hi, strict, pat, flags = _rule_variant(kind)
    rule = f"{lo} {'<' if strict else '<='} {hi}"
    for seed in (1, 2):
        out, report, _ = _cli(tmp_path / str(seed), header, rows, *flags, seed=seed, rules=[rule])
        assert len(out) == len(rows)
        assert _rule_holds(out, lo, hi, strict, comma=kind == "decimal_comma"), (kind, seed)
        assert [d["status"] for d in report["tables"][0]["structure"]["declared_rules"]] == ["applied"]
        if pat:
            assert all(re.fullmatch(pat, r[c]) for r in out for c in (lo, hi) if r[c].strip()), kind
        if kind == "suppression_tokens":
            for c in (lo, hi):
                tok = lambda v: v in ("*", "<10")  # noqa: E731
                assert Counter(r[c] for r in out if tok(r[c])) == Counter(r[header.index(c)] for r in rows
                                                                           if tok(r[header.index(c)]))


def test_rule_view_reads_formatted_numbers_and_skips_tokens() -> None:
    tp = _profile(_csv_text(["k", "amt", "cnt"],
                            [[i, f"${1000 + 7 * i:,.2f}", "*" if i % 9 == 0 else f"{30 + i:,}"] for i in range(60)]))
    pa, pc = tp.columns[1], tp.columns[2]
    assert pa.type is T.NUMERIC_FORMATTED and pc.type is T.NUMERIC_WITH_TOKENS
    va = st.rule_view(np.array(["$1,234.50", " $12.00 ", "", None], dtype=object), pa)
    assert va is not None and va.unit == "number"
    assert va.vals[:2].tolist() == [1234.5, 12.0] and va.ok.tolist() == [True, True, False, False]
    vc = st.rule_view(np.array(["1,204", "*", "<10"], dtype=object), pc)
    assert vc is not None and vc.ok.tolist() == [True, False, False] and vc.vals[0] == 1204.0


def test_built_rule_values_use_the_column_format() -> None:
    """A value built past the high column's range is written through its number format (integer-valued
    '$x.00' rounds to whole dollars, a cents column to cents)."""
    tp = _profile(_csv_text(["k", "amt", "price"],
                            [[i, f"${100 + 3 * i:,.2f}", f"${1000 + 7.25 * i:,.2f}"] for i in range(60)]))
    whole, cents = tp.columns[1], tp.columns[2]
    assert gs._rule_decimals(whole) == 0 and gs._rule_decimals(cents) == 2
    hv = st.rule_view(np.array(["$100.00"], dtype=object), whole)
    out = gs._render_once(whole, hv, np.array([0]), np.array([1234.0]))
    assert out.tolist() == ["$1,234.00"] or out.tolist() == ["$1234.00"]
    hv2 = st.rule_view(np.array(["$1,000.00"], dtype=object), cents)
    assert st.rule_view(gs._render_once(cents, hv2, np.array([0]), np.array([2345.5])), cents).vals[0] == 2345.5


def test_rules_on_formatted_columns_are_suggested() -> None:
    rows = [[f"S{1000 + i}", f"${100 + 3 * i:.2f}", f"${150 + 4 * i:.2f}"] for i in range(120)]
    tp = _profile(_csv_text(["student_id", "Paid", "Due"], rows))
    sug = [r["rule"] for r in tp.structure.suggested_rules]
    assert "Paid < Due" in sug


def test_rows_sorted_by_a_formatted_amount_stay_sorted() -> None:
    """The same reading for the sort-order detection: a file sorted by '$1,234.50' amounts of varying widths was
    not seen as sorted (the amounts were no numbers to it); now the order is found and reproduced."""
    rng = np.random.default_rng(2)
    vals = sorted(float(v) for v in rng.uniform(50, 3000, size=120))
    ids = rng.permutation(120)
    rows = [[f"S{1000 + int(ids[i])}", f"${v:,.2f}", str(int(rng.integers(0, 100)))] for i, v in enumerate(vals)]
    tp, _, out = _gen(_csv_text(["student_id", "amount", "score"], rows), 1)
    keys = tp.structure.sort_order.get("keys") or []
    assert [(k["column"], k["order"]) for k in keys] == [("amount", "numeric")]
    nums = [float(x.replace("$", "").replace(",", "")) for x in out["amount"]]
    assert nums == sorted(nums)


def test_fixture_templates_lossless_kept_lossy_reported(tmp_path: Path) -> None:
    """tests/fixtures/ids/templates.csv: full_name = last + ', ' + first keeps the source's 245 values and their
    counts (was 252); username = initial(first) + last merges different first names into one value, which the
    synthetic names cannot repeat, so its distinct count is reported, not checked; every template still holds and
    the name columns keep their exact profiles."""
    fixture = Path(__file__).parent / "fixtures" / "ids" / "templates.csv"
    with fixture.open(newline="", encoding="utf-8") as fh:
        real = list(csv.DictReader(fh))
    header = list(real[0])
    out, report, _ = _cli(tmp_path, header, [[r[h] for h in header] for r in real], seed=1)
    assert _profile_of(r["full_name"] for r in out) == _profile_of(r["full_name"] for r in real)
    for c in ("first_name", "last_name"):
        assert _profile_of(r[c] for r in out) == _profile_of(r[c] for r in real)
    assert all(r["full_name"] == f"{r['last_name']}, {r['first_name']}" for r in out)
    assert all(r["username"] == r["first_name"][0].lower() + r["last_name"].lower() for r in out)
    assert _column(report, "full_name")["fidelity"]["verdict"] == "ok"
    user = _column(report, "username")["fidelity"]
    assert user["verdict"] == "ok" and "not compared" in user["reason"]
    assert "fidelity_drift" not in _codes(report)
