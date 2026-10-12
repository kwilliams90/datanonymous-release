"""Round-2 review fixes, structure-gen cluster (R2-09, R2-11, R2-16).

* R2-09: a functional-dependency rebuild no longer breaks the whole-row missingness patterns (D6): each root
  determinant's synthetic values are dealt to the rows (entities) whose sampled pattern wants the dependents'
  mapped blank/non-blank state, and composite keys move whole rows, so FDs, composite keys and patterns hold
  together. Variants: determinant blanks, several dependents, NA tokens, never-copy dependents, within-ID
  constants, two composite keys, pooled patterns (conflicts are warned, never silent).
* R2-11: every declared rule on one high column is solved together (B = max(lows) + an observed gap), rules and
  FDs run in column-dependency order, cycles and unbuildable rules FAIL the file with the fix, and every applied
  rule is re-checked on the final output.
* R2-16: rules between dates are built in the date model and written in each row's own layout (ISO, m/d/Y,
  datetimes with minutes or offsets, DATE_AS_INT, year-month, time of day, mixed formats, typed Parquet dates);
  only applicable rules are suggested.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.config import Settings
from conftest import build_fidelity_settings as build_settings
from datanonymous.errors import GenerationError
from datanonymous.generate import structure as gs
from datanonymous.generate.ids import IdRegistry
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.privacy import matches
from datanonymous.profile import structure as st
from test_area_d_structure import _FR, _csv, _profile


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _report(rep: Path, rel: str = "data.csv") -> dict:
    return json.loads((rep / "files" / (rel + ".json")).read_text(encoding="utf-8"))


def _codes(rep: Path, rel: str = "data.csv") -> list[str]:
    r = _report(rep, rel)
    msgs = list(r.get("messages", []))
    for t in r.get("tables", []):
        msgs += t.get("messages", [])
        for c in t["columns"]:
            msgs += c.get("messages", [])
    return [m.get("code") for m in msgs]


def _run(tmp_path: Path, case: str, header: list[str], rows: list[list[object]], rules: list | None = None,
         seed: int = 1, *extra: str):
    d = tmp_path / case
    _write(d / "in" / "data.csv", header, rows)
    args = [d / "in", "--out-dir", d / "out", "--report-dir", d / "rep", "--seed", str(seed), "--force", *extra]
    if rules is not None:
        cfg = d / "rules.yaml"
        cfg.write_text("rules:\n" + "".join(f"  - {json.dumps(r)}\n" for r in rules), encoding="utf-8")
        args += ["--config", cfg]
    res = run_cli(*args)
    out = d / "out" / "data.csv"
    return res, (read_dicts(out) if out.exists() else []), d / "rep"


def _gen(tp, settings: Settings | None = None, seed: int = 1):
    """generate -> privacy -> finalize, as the pipeline does; returns the synthetic table."""
    s = settings or Settings(mode="fidelity")
    reg = IdRegistry()
    ns = {}
    for i in tp.entity.id_columns:
        reg.register(tp.columns[i].name, "t.csv", None, i, tp.columns[i].name, tp.private.id_raw[i])
        ns[i] = tp.columns[i].name
    reg.build(seed)
    synth = generate_table(tp, GenContext(seed, "t.csv", reg, ns, s))
    matches.enforce(tp, synth, np.random.default_rng(seed), None)
    synth.messages += st.finalize_rows(tp, synth, lambda purpose: np.random.default_rng(seed + 1), s)
    return synth


def _cols(synth) -> dict[str, list]:
    return {c.name: c.values.tolist() for c in synth.columns}


def _blank(v) -> bool:
    return v is None or str(v).strip() in ("", "NA")


def _patterns(table: dict[str, list], names: list[str]) -> Counter:
    return Counter(tuple(_blank(table[n][r]) for n in names) for r in range(len(table[names[0]])))


def _fd_holds(table: dict[str, list], det: str, dep: str) -> bool:
    seen: dict = {}
    return all(seen.setdefault(a, b) == b for a, b in zip(table[det], table[dep]))


def _real(text: str) -> dict[str, list]:
    rows = list(csv.reader(text.splitlines()))
    return {h: [r[i] for r in rows[1:]] for i, h in enumerate(rows[0])}


# =========================================================================== R2-09: FDs vs whole-row missingness
def _codex_fd_rows() -> list[list[object]]:
    """Codex probe3 fd_missingness: school_code -> flag; flag and score blank together (A) or filled (B)."""
    return [[f"S{i + 100}", code, ("" if code == "A" else "Y"), ("" if code == "A" else str(i + 10))]
            for i in range(30) for code in ["A", "B"]]


@pytest.mark.parametrize("seed", [123, 1, 2, 3])
def test_fd_rebuild_keeps_whole_row_patterns_fd_and_composite_key(tmp_path: Path, seed: int) -> None:
    res, out, rep = _run(tmp_path, "fdm", ["student_id", "school_code", "flag", "score"], _codex_fd_rows(), seed=seed)
    assert res.exit_code == 0, res.output
    assert Counter((r["flag"] == "", r["score"] == "") for r in out) == {(True, True): 30, (False, False): 30}
    assert Counter((r["school_code"], r["flag"]) for r in out) == {("A", ""): 30, ("B", "Y"): 30}
    assert len({(r["student_id"], r["school_code"]) for r in out}) == 60  # composite key kept
    codes = _codes(rep)
    assert "fd_missingness_conflict" not in codes and "missingness_pattern_changed" not in codes
    assert "composite_key_duplicates" not in codes


def _variant_rows(kind: str, seed: int = 0) -> tuple[str, list[str], list[tuple[str, str]]]:
    """(csv text, pattern columns, FD pairs) for one FD x missingness variant."""
    rng = np.random.default_rng(seed)
    rows: list[list[object]] = []
    if kind == "no_composite":  # one row per person
        for i in range(150):
            code = str(rng.choice(["A", "B", "C"]))
            rows.append([2000 + i, code, {"A": "", "B": "Y", "C": "N"}[code],
                         "" if code == "A" else int(rng.integers(0, 100)), int(rng.integers(0, 9))])
        return _csv(["student_id", "school_code", "flag", "score", "k"], rows), ["flag", "score"], [
            ("school_code", "flag")]
    if kind == "determinant_blanks":  # a blank code maps to a filled label, code X to a blank one
        for i in range(160):
            code = str(rng.choice(["", "X", "Y", "Z"]))
            label = {"": "Unknown", "X": "", "Y": "Yes", "Z": "Zed"}[code]
            rows.append([3000 + i, code, label, "" if code == "X" else f"n{int(rng.integers(0, 5))}",
                         int(rng.integers(0, 9))])
        return _csv(["student_id", "exit_code", "exit_label", "exit_note", "k"], rows), \
            ["exit_code", "exit_label", "exit_note"], [("exit_code", "exit_label")]
    if kind == "two_dependents":  # school -> two flags blank for different schools, each co-missing
        for i in range(200):
            sc = str(rng.choice(["S1", "S2", "S3"]))
            f1 = "" if sc == "S1" else "Y"
            f2 = "" if sc == "S2" else "Z"
            rows.append([4000 + i, sc, f1, f2, "" if sc == "S1" else int(rng.integers(0, 50)),
                         "" if sc == "S2" else int(rng.integers(0, 50))])
        return _csv(["student_id", "school_code", "lunch_flag", "bus_flag", "lunch_n", "bus_n"], rows), \
            ["lunch_flag", "bus_flag", "lunch_n", "bus_n"], [("school_code", "lunch_flag"),
                                                             ("school_code", "bus_flag")]
    if kind == "na_token":  # a typed integer dependent whose missing token is NA
        for i in range(160):
            sc = str(rng.choice(["A", "B", "C", "D"]))
            cap = {"A": "NA", "B": "300", "C": "450", "D": "520"}[sc]
            rows.append([5000 + i, sc, cap, "NA" if sc == "A" else int(rng.integers(0, 90))])
        return _csv(["student_id", "school_code", "capacity", "score"], rows), ["capacity", "score"], [
            ("school_code", "capacity")]
    if kind == "never_copy_dependent":  # a person-name dependent: synthetic names, same blanks
        names = {"A": "", "B": "Maria Lopez", "C": "John Carter", "D": "Ann Smith"}
        for i in range(160):
            sc = str(rng.choice(list(names)))
            rows.append([6000 + i, sc, names[sc], "" if sc == "A" else int(rng.integers(0, 90))])
        return _csv(["student_id", "school_code", "principal_name", "score"], rows), ["principal_name", "score"], [
            ("school_code", "principal_name")]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["no_composite", "determinant_blanks", "two_dependents", "na_token",
                                  "never_copy_dependent"])
def test_fd_missingness_variants(kind: str) -> None:
    text, cols, fds = _variant_rows(kind)
    real = _real(text)
    tp = _profile(text)
    found = {(f["determinant"], f["dependent"]) for f in tp.structure.functional_dependencies}
    assert set(fds) <= found, (kind, found)
    for seed in (1, 2, 3):
        tp = _profile(text)
        synth = _gen(tp, seed=seed)
        out = _cols(synth)
        assert _patterns(out, cols) == _patterns(real, cols), (kind, seed)
        for det, dep in fds:
            assert _fd_holds(out, det, dep), (kind, seed, det, dep)
        assert not {m.code for m in synth.messages} & {"fd_missingness_conflict", "missingness_pattern_changed"}
        if kind == "never_copy_dependent":
            assert not set(out["principal_name"]) & {"Maria Lopez", "John Carter", "Ann Smith"}


def _entity_rows(seed: int = 0) -> str:
    """Within-ID constants: school -> flag (blank for A), and exit_date blank exactly when flag is."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(60):
        sc = "A" if s % 2 else "B"
        flag = "" if sc == "A" else "Y"
        ex = "" if sc == "A" else f"2024-05-{int(rng.integers(1, 28)):02d}"
        for _ in range(int(rng.integers(2, 5))):
            rows.append([7000 + 3 * s, sc, flag, ex, int(rng.integers(0, 100))])
    return _csv(["student_id", "school_code", "flag", "exit_date", "score"], rows)


def test_fd_missingness_within_id_constants() -> None:
    text = _entity_rows()
    tp = _profile(text)
    assert {tp.columns[i].name for i in tp.entity.constant_columns} >= {"school_code", "flag", "exit_date"}
    assert ("school_code", "flag") in {(f["determinant"], f["dependent"]) for f in tp.structure.functional_dependencies}
    real = _real(text)

    def per_entity(t: dict[str, list]) -> Counter:
        first: dict = {}
        for r, sid in enumerate(t["student_id"]):
            first.setdefault(sid, r)
        return Counter((_blank(t["flag"][r]), _blank(t["exit_date"][r])) for r in first.values())

    for seed in (1, 2, 3):
        tp = _profile(text)
        out = _cols(_gen(tp, seed=seed))
        assert per_entity(out) == per_entity(real), seed
        assert _fd_holds(out, "school_code", "flag")


def test_pooled_patterns_conflict_is_warned_not_silent() -> None:
    """Patterns pooled below --min-cell are random per column: an FD conflict there follows the FD, with a warning."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(120):
        sc = str(rng.choice(["S1", "S2", "S3"]))
        rows.append([8000 + i, sc, "" if sc == "S1" else "Y", "" if sc == "S2" else "Z"])
    text = _csv(["student_id", "school_code", "lunch_flag", "bus_flag"], rows)
    s = build_settings({"min_cell": 100}, {})
    real = _real(text)
    for seed in (1, 2, 3, 4):
        tp = _profile(text, s)
        synth = _gen(tp, s, seed=seed)
        out = _cols(synth)
        assert _fd_holds(out, "school_code", "lunch_flag") and _fd_holds(out, "school_code", "bus_flag")
        same = _patterns(out, ["lunch_flag", "bus_flag"]) == _patterns(real, ["lunch_flag", "bus_flag"])
        assert same or "fd_missingness_conflict" in {m.code for m in synth.messages}, seed


def _two_key_rows(seed: int = 0) -> str:
    """Two composite keys (student, visit_date) and (student, visit_no); reason/code blank together."""
    rng = np.random.default_rng(seed)
    days = [f"2024-09-{d:02d}" for d in range(1, 21)]
    rows = []
    for s in range(50):
        k = int(rng.integers(3, 9))
        for j, d in enumerate(sorted(rng.choice(days, k, replace=False).tolist())):
            absent = bool(rng.random() < 0.3)
            rows.append([9000 + 7 * s, d, f"V{j + 1:02d}", "" if not absent else str(rng.choice(["ILL", "TRIP"])),
                         "" if not absent else str(rng.choice(["E", "U"])), int(rng.integers(0, 9))])
    return _csv(["student_id", "visit_date", "visit_no", "reason", "reason_code", "k"], rows)


def test_two_composite_keys_and_patterns_hold_together() -> None:
    """Composite keys move whole rows; enforcing one key never breaks another (each keeps the other's columns)."""
    text = _two_key_rows()
    tp = _profile(text)
    keys = sorted(tuple(k["columns"]) for k in tp.structure.composite_keys)
    assert keys == [("student_id", "visit_date"), ("student_id", "visit_no")], keys
    real = _real(text)
    for seed in (1, 2, 3):
        tp = _profile(text)
        reg = IdRegistry()
        reg.register("student_id", "t.csv", None, 0, "student_id", tp.private.id_raw[0])
        reg.build(seed)
        synth = generate_table(tp, GenContext(seed, "t.csv", reg, {0: "student_id"}, Settings(mode="fidelity")))
        out = _cols(synth)  # before the privacy swaps (which may move a key value alone; repaired best-effort)
        n = len(out["student_id"])
        assert len(set(zip(out["student_id"], out["visit_date"]))) == n, seed
        assert len(set(zip(out["student_id"], out["visit_no"]))) == n, seed
        assert _patterns(out, ["reason", "reason_code"]) == _patterns(real, ["reason", "reason_code"]), seed
        assert Counter(out["visit_date"]) == Counter(real["visit_date"])  # column totals unchanged
        matches.enforce(tp, synth, np.random.default_rng(seed), None)
        msgs = st.finalize_rows(tp, synth, lambda purpose: np.random.default_rng(seed + 1), Settings(mode="fidelity"))
        out = _cols(synth)
        assert len(set(zip(out["student_id"], out["visit_date"]))) == n, seed
        left = n - len(set(zip(out["student_id"], out["visit_no"])))
        assert not left or "composite_key_duplicates" in {m.code for m in msgs}  # never silent
        assert _patterns(out, ["reason", "reason_code"]) == _patterns(real, ["reason", "reason_code"]), seed
        assert "missingness_pattern_changed" not in {m.code for m in msgs}


def test_greedy_assign_keeps_tuple_interface() -> None:
    """The tuple-writing helper still assigns whole tuples (kept for callers that move a few columns)."""
    rng = np.random.default_rng(0)
    vals = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    cols = {0: vals.copy()}
    blocks = np.array([0, 0, 1, 1, 2, 2])
    res = gs._greedy_assign(np.arange(6), [(v,) for v in vals.tolist()], vals.tolist(), blocks, cols, (0,), rng)
    assert res == 0 and all(len(set(cols[0][blocks == b].tolist())) == 2 for b in range(3))


# =========================================================================== R2-11: rules sharing a target
def _shared_target_rows(n: int = 200, seed: int = 987) -> list[list[int]]:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a, b, d = r.randrange(100, 1000), r.randrange(100, 1000), r.randrange(100, 1000)
        out.append([a, b, d, max(a, b, d) + r.randrange(1, 10)])
    return out


def _holds(rows: list[dict[str, str]], rule: str) -> bool:
    lo, op, hi = rule.split()
    if op in (">=", ">"):
        lo, hi, op = hi, lo, {">=": "<=", ">": "<"}[op]
    cmp = (lambda x, y: x < y) if op == "<" else (lambda x, y: x <= y)
    return all(cmp(float(r[lo]), float(r[hi])) for r in rows if r[lo] != "" and r[hi] != "")


@pytest.mark.parametrize("rules", [
    ["a <= c", "b <= c"],  # codex probe5
    ["a < c", "b <= c"],
    ["c >= a", "b < c", "d <= c"],
    ["a <= c", "a < c", "b <= c"],
])
def test_rules_sharing_a_target_hold_together(tmp_path: Path, rules: list[str]) -> None:
    rows = _shared_target_rows()
    for seed in (123, 2):
        res, out, rep = _run(tmp_path, f"same{seed}", ["a", "b", "d", "c"], rows, rules, seed)
        assert res.exit_code == 0, res.output
        for rule in rules:
            assert _holds(out, rule), (rule, seed)
        statuses = [d["status"] for d in _report(rep)["tables"][0]["structure"]["declared_rules"]]
        assert statuses == ["applied"] * len(rules)
        # R3-06: c keeps its own values (in its source range, exact max); the rules decide where they go. Its
        # min (160) needs a row whose lows are all below it: three independent lows rarely give one, so then
        # no arrangement can place it and the report says so (extremes_not_emitted), never a broken rule
        real_c, syn_c = [row[3] for row in rows], [int(r["c"]) for r in out]
        assert min(real_c) <= min(syn_c) and max(syn_c) == max(real_c), seed
        if min(syn_c) != min(real_c):
            assert "extremes_not_emitted" in _codes(rep)


def test_rule_chain_in_reverse_order_with_a_shared_target(tmp_path: Path) -> None:
    r = random.Random(123)
    rows = []
    for _ in range(200):
        a = r.randrange(100, 1000)
        b = a + r.randrange(10, 100)
        rows.append([a, b, b + r.randrange(10, 100)])
    rules = ["b <= c", "a <= b", "a <= c"]
    res, out, rep = _run(tmp_path, "chain", ["a", "b", "c"], rows, rules, 123)
    assert res.exit_code == 0, res.output
    assert all(_holds(out, rule) for rule in rules)


def test_rules_on_a_decimal_target_with_mixed_precision() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(150):
        a = round(float(rng.uniform(0, 10)), 2)
        b = round(float(rng.uniform(0, 10)), 3)
        c = np.ceil(max(a, b) * 10) / 10 + float(rng.choice([0.0, 0.5, 1.2]))
        rows.append([f"{a:.2f}", f"{b:.3f}", f"{c:.1f}"])
    text = _csv(["planned", "booked", "logged"], rows)
    s = build_settings({}, {"rules": ["planned <= logged", "booked <= logged"]})
    for seed in (1, 2, 3):
        tp = _profile(text, s)
        st.cross_file_structure([_FR("t.csv", tp)], s)
        out = _cols(_gen(tp, s, seed))
        assert all(len(v.split(".")[1]) == 1 for v in out["logged"])
        assert all(float(c) >= float(a) and float(c) >= float(b)
                   for a, b, c in zip(out["planned"], out["booked"], out["logged"])), seed


def test_rules_between_within_id_constants_share_a_target() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for s in range(90):
        a, b = int(rng.integers(2012, 2020)), int(rng.integers(2012, 2020))
        c = max(a, b) + int(rng.choice([0, 1, 2]))
        for _ in range(int(rng.integers(2, 4))):
            rows.append([1000 + 7 * s, a, b, c, int(rng.integers(0, 100))])
    text = _csv(["student_id", "KCSE", "entry_year", "Yr_JoinCampus", "score"], rows)
    s = build_settings({}, {"rules": ["KCSE <= Yr_JoinCampus", "entry_year <= Yr_JoinCampus"]})
    for seed in (1, 2):
        tp = _profile(text, s)
        assert {tp.columns[i].name for i in tp.entity.constant_columns} >= {"KCSE", "entry_year", "Yr_JoinCampus"}
        st.cross_file_structure([_FR("t.csv", tp)], s)
        assert tp.structure.private["rule_targets"][3].get("per_entity")
        out = _cols(_gen(tp, s, seed))
        per: dict = {}
        for sid, a, b, c in zip(out["student_id"], out["KCSE"], out["entry_year"], out["Yr_JoinCampus"]):
            per.setdefault(sid, set()).add(c)
            assert int(c) >= int(a) and int(c) >= int(b), seed
        assert all(len(v) == 1 for v in per.values())  # still constant within the ID (D4)


def test_rule_low_column_rebuilt_by_a_functional_dependency() -> None:
    """school_code -> base_year (an FD) and 'base_year <= grad_year': the FD runs before the rule reads it."""
    rng = np.random.default_rng(6)
    base = {"577": 2015, "530": 2016, "612": 2018, "701": 2019}
    rows = []
    for i in range(240):
        sc = str(rng.choice(list(base)))
        rows.append([1000 + i, sc, base[sc], base[sc] + int(rng.integers(0, 6)), int(rng.integers(0, 100))])
    text = _csv(["student_id", "school_code", "base_year", "grad_year", "score"], rows)
    s = build_settings({}, {"rules": ["base_year <= grad_year"]})
    tp = _profile(text, s)
    assert ("school_code", "base_year") in {(f["determinant"], f["dependent"])
                                            for f in tp.structure.functional_dependencies}
    for seed in (1, 2, 3):
        tp = _profile(text, s)
        st.cross_file_structure([_FR("t.csv", tp)], s)
        out = _cols(_gen(tp, s, seed))
        assert _fd_holds(out, "school_code", "base_year")
        assert all(int(g) >= int(b) for b, g in zip(out["base_year"], out["grad_year"])), seed


def test_rule_cycle_fails_the_file_with_the_fix(tmp_path: Path) -> None:
    rows = [[i, i] for i in range(50)]
    res, out, rep = _run(tmp_path, "cycle", ["a", "b"], rows, ["a <= b", "b <= a"])
    assert res.exit_code == 3 and not out
    r = _report(rep)
    assert r["status"] == "FAIL" and any("cycle" in x for x in r["reasons"])


def test_file_qualified_rules_build_only_this_files_target() -> None:
    rows = _shared_target_rows(120)
    text = _csv(["a", "b", "d", "c"], rows)
    s = build_settings({}, {"rules": [{"rule": "a <= c", "file": "t.csv"}, {"rule": "b <= c", "file": "o.csv"}]})
    tp = _profile(text, s)
    st.cross_file_structure([_FR("t.csv", tp)], s)
    assert [m[2] for m in tp.structure.private["rule_targets"][3]["members"]] == [("a", "c", False)]
    out = _cols(_gen(tp, s, 1))
    assert all(int(c) >= int(a) for a, c in zip(out["a"], out["c"]))


def test_check_rules_reports_a_violation_strict_and_no_strict() -> None:
    text = _csv(["a", "c"], [[i, i + 3] for i in range(60)])
    for no_strict in (False, True):
        s = build_settings({"no_strict": no_strict}, {"rules": ["a <= c"]})
        tp = _profile(text, s)
        st.cross_file_structure([_FR("t.csv", tp)], s)
        synth = _gen(tp, s, 1)
        cols = {c.index: c.values.copy() for c in synth.columns}
        cols[1][0] = str(int(cols[0][0]) - 1)  # break the rule in one row
        if not no_strict:
            with pytest.raises(GenerationError):
                gs.check_rules(tp, cols, s)
        else:
            msgs = gs.check_rules(tp, cols, s)
            assert [m.code for m in msgs] == ["rule_violated"]
            assert tp.structure.declared_rules[0]["synthetic_rows_violating"] == 1


# =========================================================================== R2-16: date rules
def test_codex_date_rule_is_enforced(tmp_path: Path) -> None:
    d0 = dt.date(2024, 1, 1)
    rows = [[(d0 + dt.timedelta(days=i)).isoformat(), (d0 + dt.timedelta(days=i + 5)).isoformat()] for i in range(40)]
    for seed in (123, 1, 2):
        res, out, rep = _run(tmp_path, f"dr{seed}", ["start_date", "end_date"], rows, ["start_date <= end_date"], seed)
        assert res.exit_code == 0, res.output
        gaps = Counter((dt.date.fromisoformat(r["end_date"]) - dt.date.fromisoformat(r["start_date"])).days
                       for r in out)
        # R3-06: end_date keeps its own (interpolated) dates in its source range, dealt so the rule holds and
        # the gaps stay near the observed 5 days (exactly 5 wherever the dates allow)
        assert min(gaps) >= 0 and gaps.most_common(1)[0][0] == 5 and gaps[5] >= 20, gaps
        ends = sorted(r["end_date"] for r in out)
        assert ends[0] == "2024-01-06" and ends[-1] == "2024-02-14"
        t = _report(rep)["tables"][0]["structure"]
        assert [d["status"] for d in t["declared_rules"]] == ["applied"]
        assert "rule_not_applied" not in _codes(rep)


def _date_case(kind: str, rng: random.Random) -> tuple[list[str], list[list[object]], str, object]:
    """(header, rows, rule, parser) for one date-rule variant."""
    d0 = dt.date(2023, 8, 1)
    rows: list[list[object]] = []
    if kind == "us_nonpadded_strict":
        for _ in range(80):
            a = d0 + dt.timedelta(days=rng.randrange(0, 300))
            b = a + dt.timedelta(days=rng.choice([1, 2, 7, 30]))
            rows.append([f"{a.month}/{a.day}/{a.year}", f"{b.month}/{b.day}/{b.year}", rng.randrange(100)])
        return ["entry_date", "exit_date", "score"], rows, "entry_date < exit_date", \
            lambda s: dt.datetime.strptime(s, "%m/%d/%Y")
    if kind == "datetime_minutes_strict":
        for _ in range(80):
            a = dt.datetime(2024, 1, 1, 8, 0) + dt.timedelta(minutes=rng.randrange(0, 60 * 24 * 30))
            b = a + dt.timedelta(minutes=rng.choice([15, 30, 45, 90]))
            rows.append([a.strftime("%Y-%m-%d %H:%M"), b.strftime("%Y-%m-%d %H:%M"), rng.randrange(100)])
        return ["login", "logout", "score"], rows, "login < logout", \
            lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
    if kind == "datetime_offsets":
        for _ in range(80):
            tz = dt.timezone(dt.timedelta(hours=rng.choice([-5, -6])))
            a = dt.datetime(2024, 3, 1, 9, 0, tzinfo=tz) + dt.timedelta(minutes=rng.randrange(0, 60 * 24 * 20))
            b = (a + dt.timedelta(minutes=rng.choice([5, 60, 600]))).astimezone(
                dt.timezone(dt.timedelta(hours=rng.choice([-5, -6]))))
            rows.append([a.isoformat(), b.isoformat(), rng.randrange(100)])
        return ["start_ts", "end_ts", "score"], rows, "start_ts <= end_ts", dt.datetime.fromisoformat
    if kind == "date_as_int":
        for _ in range(80):
            a = d0 + dt.timedelta(days=rng.randrange(0, 300))
            b = a + dt.timedelta(days=rng.choice([0, 3, 10]))
            rows.append([a.strftime("%Y%m%d"), b.strftime("%Y%m%d"), rng.randrange(100)])
        return ["start_date", "end_date", "score"], rows, "start_date <= end_date", \
            lambda s: dt.datetime.strptime(s, "%Y%m%d")
    if kind == "year_month":
        for _ in range(80):
            a = rng.randrange(0, 36)
            b = a + rng.choice([0, 1, 6])
            rows.append([f"{2020 + a // 12}-{a % 12 + 1:02d}", f"{2020 + b // 12}-{b % 12 + 1:02d}",
                         rng.randrange(100)])
        return ["first_month", "last_month", "score"], rows, "first_month <= last_month", \
            lambda s: dt.datetime.strptime(s, "%Y-%m")
    if kind == "time_of_day_strict":
        for _ in range(80):
            a = rng.randrange(7 * 60, 15 * 60)
            b = a + rng.choice([45, 50, 90])
            rows.append([f"{a // 60:02d}:{a % 60:02d}", f"{b // 60:02d}:{b % 60:02d}", rng.randrange(100)])
        return ["period_start", "period_end", "score"], rows, "period_start < period_end", \
            lambda s: dt.datetime.strptime(s, "%H:%M")
    if kind == "mixed_formats":
        for i in range(120):
            a = d0 + dt.timedelta(days=rng.randrange(0, 300))
            b = a + dt.timedelta(days=rng.choice([0, 2, 9]))
            fa, fb = ("%Y-%m-%d", "%m/%d/%Y") if i % 2 else ("%m/%d/%Y", "%Y-%m-%d")
            rows.append([a.strftime(fa), b.strftime(fb), rng.randrange(100)])

        def parse(s: str) -> dt.datetime:
            return dt.datetime.strptime(s, "%Y-%m-%d" if "-" in s else "%m/%d/%Y")
        return ["start_date", "end_date", "score"], rows, "start_date <= end_date", parse
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["us_nonpadded_strict", "datetime_minutes_strict", "datetime_offsets",
                                  "date_as_int", "year_month", "time_of_day_strict", "mixed_formats"])
def test_date_rule_variants(tmp_path: Path, kind: str) -> None:
    header, rows, rule, parse = _date_case(kind, random.Random(3))
    lo, op, hi = rule.split()
    real_shapes = Counter(len(str(r[1])) for r in rows)
    for seed in (5, 6):
        res, out, rep = _run(tmp_path, f"{kind}{seed}", header, rows, [rule], seed)
        assert res.exit_code == 0, res.output
        assert out and all(parse(r[lo]) < parse(r[hi]) if op == "<" else parse(r[lo]) <= parse(r[hi])
                           for r in out), (kind, seed)
        assert [d["status"] for d in _report(rep)["tables"][0]["structure"]["declared_rules"]] == ["applied"]
        assert set(Counter(len(r[hi]) for r in out)) <= set(real_shapes)  # written in the source layouts


def test_date_rule_keeps_sentinel_tokens(tmp_path: Path) -> None:
    rng = random.Random(8)
    d0 = dt.date(2023, 8, 1)
    rows = []
    for i in range(120):
        a = d0 + dt.timedelta(days=rng.randrange(0, 300))
        end = "9999-12-31" if i % 6 == 0 else (a + dt.timedelta(days=rng.choice([1, 20]))).isoformat()
        rows.append([a.isoformat(), end, rng.randrange(100)])
    res, out, rep = _run(tmp_path, "sent", ["enroll_date", "exit_date", "score"], rows, ["enroll_date <= exit_date"])
    assert res.exit_code == 0, res.output
    assert sum(r["exit_date"] == "9999-12-31" for r in out) == 20
    assert all(r["exit_date"] >= r["enroll_date"] for r in out)


def test_date_rule_in_typed_parquet(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = random.Random(9)
    d0 = dt.date(2022, 9, 1)
    a = [d0 + dt.timedelta(days=rng.randrange(0, 200)) for _ in range(80)]
    b = [x + dt.timedelta(days=rng.choice([0, 4, 15])) for x in a]
    d = tmp_path / "pq"
    (d / "in").mkdir(parents=True)
    pq.write_table(pa.table({"start_date": pa.array(a, pa.date32()), "end_date": pa.array(b, pa.date32()),
                             "score": list(range(80))}), d / "in" / "data.parquet")
    cfg = d / "rules.yaml"
    cfg.write_text('rules: ["start_date <= end_date"]\n', encoding="utf-8")
    res = run_cli(d / "in", "--out-dir", d / "out", "--report-dir", d / "rep", "--seed", "1", "--config", cfg)
    assert res.exit_code == 0, res.output
    t = pq.read_table(d / "out" / "data.parquet")
    assert t.schema.field("end_date").type == pa.date32()
    s, e = t.column("start_date").to_pylist(), t.column("end_date").to_pylist()
    assert all(y >= x for x, y in zip(s, e) if x is not None and y is not None)


def test_suggested_rules_apply_as_suggested(tmp_path: Path) -> None:
    """Every suggestion's config text, fed back, is applied and holds (no unusable configuration)."""
    rng = random.Random(10)
    d0 = dt.date(2023, 8, 1)
    rows = []
    for i in range(150):
        a = d0 + dt.timedelta(days=rng.randrange(0, 200))
        b = a + dt.timedelta(days=rng.choice([1, 3, 40]))
        y = rng.randrange(2014, 2021)
        rows.append([a.isoformat(), b.isoformat(), y, y + rng.choice([0, 1, 2, 5]), rng.randrange(100)])
    header = ["entry_date", "exit_date", "KCSE", "Yr_JoinCampus", "score"]
    res, _, rep = _run(tmp_path, "sugg", header, rows)
    assert res.exit_code == 0, res.output
    sugg = _report(rep)["tables"][0]["structure"]["suggested_rules"]
    assert {s["kind"] for s in sugg} == {"date", "number"}
    texts = [s["rule"] for s in sugg]
    res, out, rep = _run(tmp_path, "sugg2", header, rows, texts, 2)
    assert res.exit_code == 0, res.output
    statuses = [d["status"] for d in _report(rep)["tables"][0]["structure"]["declared_rules"]]
    assert statuses == ["applied"] * len(texts)
    for rule in texts:
        lo, op, hi = rule.split()
        if "date" in lo:
            assert all(r[lo] < r[hi] if op == "<" else r[lo] <= r[hi] for r in out), rule
        else:
            assert _holds(out, rule), rule


def test_unbuildable_rules_fail_and_are_never_suggested(tmp_path: Path) -> None:
    rng = random.Random(11)
    d0 = dt.date(2023, 8, 1)
    rows = []
    for i in range(80):
        a = d0 + dt.timedelta(days=rng.randrange(0, 300))
        rows.append([a.isoformat(), (dt.datetime.combine(a, dt.time(10, 0)) + dt.timedelta(days=rng.choice([0, 30])))
                     .strftime("%Y-%m-%d %H:%M"), f"note {rng.randrange(1000)} words here", rng.randrange(100)])
    header = ["d", "ts", "comment", "score"]
    res, _, rep = _run(tmp_path, "nosugg", header, rows)
    assert res.exit_code == 0 and not _report(rep)["tables"][0]["structure"]["suggested_rules"]
    for rule, why in (("d <= ts", "different kinds"), ("score <= comment", "numbers")):
        res, out, rep = _run(tmp_path, "bad" + rule.split()[0], header, rows, [rule])
        assert res.exit_code == 3 and not out, (rule, res.output)
        r = _report(rep)
        assert r["status"] == "FAIL" and any(why in x for x in r["reasons"]) and r.get("fixes"), r
    # a rule on a column the run does not generate has nothing to enforce: a warning, the file is written
    res, out, rep = _run(tmp_path, "excl", header, rows, ["score <= d"], 1, "--exclude", "score")
    assert res.exit_code == 0 and out
    assert "rule_not_applied" in _codes(rep)


def test_mixed_role_rule_fails_and_is_not_suggested(tmp_path: Path) -> None:
    rng = random.Random(12)
    rows = []
    for s in range(60):
        y = rng.randrange(2010, 2015)
        for _ in range(rng.randrange(2, 5)):
            rows.append([1000 + 3 * s, y, y + rng.randrange(1, 6), rng.randrange(100)])
    header = ["student_id", "cohort_year", "test_year", "score"]
    res, _, rep = _run(tmp_path, "roles", header, rows)
    assert res.exit_code == 0
    assert not _report(rep)["tables"][0]["structure"]["suggested_rules"]
    res, out, rep = _run(tmp_path, "roles2", header, rows, ["cohort_year <= test_year"])
    assert res.exit_code == 3 and not out
    assert any("constant within the ID" in x for x in _report(rep)["reasons"])
