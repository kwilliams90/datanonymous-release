"""AREA D: structural relationships (D5, D16, section 6) - detection and generation-time application."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from datanonymous.config import Settings
from conftest import build_fidelity_settings as build_settings
from datanonymous.errors import UsageError
from datanonymous.generate import structure as gs
from datanonymous.generate.ids import IdRegistry
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.io import csvio
from datanonymous.privacy import matches
from datanonymous.profile import structure as st
from datanonymous.profile import templates as tpl
from datanonymous.profile.table import profile_table


def _profile(text: str, settings: Settings | None = None, name: str = "t.csv"):
    r = csvio.parse_text_bytes(text.encode("utf-8"))
    return profile_table(r.tables[0], name, "csv", settings or Settings(mode="fidelity"))


def _generate(tp, settings: Settings | None = None, seed: int = 1, name: str = "t.csv"):
    s = settings or Settings(mode="fidelity")
    reg = IdRegistry()
    ns = {}
    for i in tp.entity.id_columns:
        reg.register(tp.columns[i].name, name, None, i, tp.columns[i].name, tp.private.id_raw[i])
        ns[i] = tp.columns[i].name
    reg.build(seed)
    synth = generate_table(tp, GenContext(seed, name, reg, ns, s))
    matches.enforce(tp, synth, np.random.default_rng(seed), None)
    st.finalize_rows(tp, synth, lambda purpose: np.random.default_rng(seed + 1), s)
    return {c.name: c.values.tolist() for c in synth.columns}


def _csv(header: list[str], rows: list[list[object]]) -> str:
    return ",".join(header) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows)


# --------------------------------------------------------------------------- functional dependencies
SCHOOLS = {"577": "Lincoln", "530": "Adams", "612": "Roosevelt", "701": "Jefferson"}


def _school_rows(n: int, seed: int = 0) -> list[list[object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sid = str(rng.choice(list(SCHOOLS)))
        rows.append([sid, SCHOOLS[sid], str(rng.choice(["F", "M"])), int(rng.integers(0, 100)),
                     f"Teacher{int(rng.integers(0, 40))}" if sid != "701" else f"Teacher{100 + int(sid) % 3}"])
    return rows


def test_fd_detected_and_applied() -> None:
    text = _csv(["school_id", "school_name", "gender", "score", "advisor"], _school_rows(400))
    tp = _profile(text)
    fds = tp.structure.functional_dependencies
    assert [(f["determinant"], f["dependent"], f["kind"]) for f in fds] == [("school_id", "school_name", "1:1")]
    assert fds[0]["status"] == "applied" and "Lincoln" not in str(fds)  # names only, never values
    out = _generate(tp)
    pairs = set(zip(out["school_id"], out["school_name"]))
    assert pairs == set(SCHOOLS.items())
    assert Counter(out["school_name"]) == Counter(r[1] for r in _school_rows(400))


def test_fd_needs_support_and_skips_unique_determinants() -> None:
    rows = [[f"C{i:03d}", f"Course {i}", i % 3] for i in range(60)]  # unique code -> title, not org codes
    tp = _profile(_csv(["code", "title", "grp"], rows))
    assert tp.structure.functional_dependencies == []
    # org code -> org code: kept even when every value is unique (D10 keeps both)
    rows = [[k, v] for k, v in SCHOOLS.items()]
    tp = _profile(_csv(["school_id", "school_name"], rows))
    assert [f["text"] for f in tp.structure.functional_dependencies] == ["school_id -> school_name"]


def test_fd_dependent_that_is_never_copied_maps_onto_synthetic_values() -> None:
    rng = np.random.default_rng(2)
    teachers = {"A": "Ms Avery", "B": "Mr Blake", "C": "Dr Carmen"}
    rows = []
    for _ in range(300):
        sec = str(rng.choice(list(teachers)))
        rows.append([sec, teachers[sec], int(rng.integers(0, 50))])
    tp = _profile(_csv(["section_code", "teacher_name", "points"], rows))
    fds = tp.structure.functional_dependencies
    assert fds and fds[0]["dependent"] == "teacher_name"
    out = _generate(tp)
    names = set(out["teacher_name"])
    assert not names & set(teachers.values())
    by_sec: dict[str, set[str]] = {}
    for s_, t_ in zip(out["section_code"], out["teacher_name"]):
        by_sec.setdefault(s_, set()).add(t_)
    assert all(len(v) == 1 for v in by_sec.values()) and len(names) == 3


def test_fd_disabled_by_config() -> None:
    s = build_settings({}, {"structure": {"disable": ["functional_dependencies"]}})
    tp = _profile(_csv(["school_id", "school_name", "gender", "score", "advisor"], _school_rows(400)), s)
    assert tp.structure.functional_dependencies[0]["status"].startswith("disabled by config")
    out = _generate(tp, s)
    assert len(set(zip(out["school_id"], out["school_name"]))) > 4  # independent again


# --------------------------------------------------------------------------- derived templates
def test_template_inference_parts() -> None:
    first = np.array(["Ann", "Bob", "Cy", "Dee", "Eve"] * 4, dtype=object)
    last = np.array(["Lee", "Moss", "Nash", "Oak", "Pike", "Ray", "Sun", "Tao", "Uhl", "Vo"] * 2, dtype=object)
    ids = np.array([str(i) for i in range(1, 21)], dtype=object)
    local = [f"{f.lower()}.{la.lower()}.{i}" for f, la, i in zip(first, last, ids)]
    parts = tpl.infer_template(local, {0: first, 1: last, 2: ids}, 10, True)
    assert parts == (("f", 0, "lower_alnum"), ("s", "."), ("f", 1, "lower_alnum"), ("s", "."), ("f", 2, "as_is"))
    user = [f"{f[0].lower()}{la.lower()}" for f, la in zip(first, last)]
    assert tpl.infer_template(user, {0: first, 1: last}, 10, False) == (
        ("f", 0, "initial_lower"), ("f", 1, "lower_alnum"))
    # free trailing digits (jsmith, jsmith2)
    sfx = [u + ("" if k % 3 else str(k)) for k, u in enumerate(user)]
    parts = tpl.infer_template(sfx, {0: first, 1: last}, 10, False)
    assert parts is not None and parts[-1] == ("d",)
    assert tpl.infer_template(["x" + u for u in user], {0: first, 1: last}, 10, False) is None
    assert tpl.infer_template(local[:5], {0: first[:5], 1: last[:5], 2: ids[:5]}, 10, True) is None  # < min rows


def test_reserved_domains_never_real() -> None:
    out = tpl.reserved_domains(Counter({"example.org": 5, "school.edu": 9, "a.com": 1, "b.com": 1, "c.com": 1}))
    doms = [d for d, _ in out]
    assert "example.org" not in doms and len(set(doms)) == 5
    assert doms[0] == "example.com" and [c for _, c in out] == [9, 5, 1, 1, 1]


def test_template_applied_and_collisions_fall_back() -> None:
    rng = np.random.default_rng(5)
    fn = ["Ann", "Bob", "Cy", "Dee", "Eve", "Fay", "Gus", "Hal"]
    ln = ["Lee", "Moss", "Nash", "Oak", "Pike", "Ray", "Sun", "Tao", "Uhl"]
    rows = []
    for i in range(1, 121):
        f, la = str(rng.choice(fn)), str(rng.choice(ln))
        rows.append([i, f, la, f"{f.lower()}.{la.lower()}{i}@school.example-isd.org", f"{la}, {f}"])
    text = "id,first_name,last_name,email,full_name\n" + "".join(
        f'{r[0]},{r[1]},{r[2]},{r[3]},"{r[4]}"\n' for r in rows)
    tp = _profile(text)
    t = {d["column"]: d for d in tp.structure.derived_templates}
    assert set(t) == {"email", "full_name"} and t["email"]["email_domains"] == 1
    assert "school" not in str(tp.structure.to_public([c.name for c in tp.columns]))
    out = _generate(tp)
    for i, f, la, e, full in zip(out["id"], out["first_name"], out["last_name"], out["email"], out["full_name"]):
        assert e == f"{tpl.TRANSFORMS['lower_alnum'](f)}.{tpl.TRANSFORMS['lower_alnum'](la)}{i}@example.org"
        assert full == f"{la}, {f}"
    assert not set(out["email"]) & {r[3] for r in rows}


# --------------------------------------------------------------------------- composite keys
def _visits(n_students: int = 40, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    days = [f"2024-09-{d:02d}" for d in range(1, 21)]
    rows = []
    for s in range(n_students):
        g = str(rng.choice(["F", "M"]))
        for d in sorted(rng.choice(days, int(rng.integers(5, 19)), replace=False).tolist()):
            rows.append([1000 + s, d, str(rng.choice(["P", "A"])), g])
    return _csv(["student_id", "visit_date", "code", "gender"], rows)


def test_composite_key_detected_and_enforced() -> None:
    tp = _profile(_visits())
    assert [k["columns"] for k in tp.structure.composite_keys] == [["student_id", "visit_date"]]
    real_sizes = Counter(Counter(r.split(",")[0] for r in _visits().splitlines()[1:]).values())
    for seed in (1, 2, 3):
        tp = _profile(_visits())
        out = _generate(tp, seed=seed)
        pairs = list(zip(out["student_id"], out["visit_date"]))
        assert len(pairs) == len(set(pairs))
        assert Counter(Counter(out["student_id"]).values()) == real_sizes  # rows per ID untouched


def test_composite_key_exact_column_totals() -> None:
    tp = _profile(_visits(60, 4))
    real = Counter(tp.columns[1].model.nonmissing_counts)
    out = _generate(tp, seed=7)
    syn = Counter(out["visit_date"])
    assert sorted(syn.values()) == sorted(real.values())  # invented labels: same frequency profile
    pairs = list(zip(out["student_id"], out["visit_date"]))
    assert len(pairs) == len(set(pairs))


def test_greedy_assignment_handles_tight_blocks() -> None:
    # every block needs all 6 values: only the greedy largest-capacity rule succeeds reliably
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(30), 6)
    vals = np.empty(180, dtype=object)
    vals[:] = [str(v) for v in rng.permutation(np.tile(np.arange(6), 30))]
    cols = {0: vals}
    slots = np.arange(180)
    residual = gs._greedy_assign(slots, [(v,) for v in vals.tolist()], vals.tolist(), blocks, cols, (0,), rng)
    assert residual == 0
    for b in range(30):
        assert sorted(cols[0][blocks == b].tolist()) == [str(v) for v in range(6)]


# --------------------------------------------------------------------------- inequality rules
def test_rule_parsing() -> None:
    rules = st.parse_rules(["a <= b", "c > d", {"rule": "'x y' < z", "file": "sub\\f.csv"}, "e >= f"])
    assert [(r["lo"], r["hi"], r["strict"], r["file"]) for r in rules] == [
        ("a", "b", False, None), ("d", "c", True, None), ("x y", "z", True, "sub/f.csv"), ("f", "e", False, None)]
    for bad in (["a = b"], [{"rules": "a<b"}], ["a <= a"], [3]):
        with pytest.raises(UsageError):
            st.parse_rules(bad)


def _years(n: int = 120, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        a = int(rng.integers(2014, 2021))
        rows.append([a, a + int(rng.choice([0, 1, 1, 2, 5])), int(rng.integers(0, 10)), int(rng.integers(50, 100))])
    return _csv(["KCSE", "Yr_JoinCampus", "hours", "score"], rows)


def test_rule_suggestions_skip_scale_coincidences() -> None:
    tp = _profile(_years())
    rules = [r["rule"] for r in tp.structure.suggested_rules]
    assert rules == ["KCSE <= Yr_JoinCampus"]  # hours < score holds only by scale
    assert tp.structure.suggested_rules[0]["config"] == 'rules: ["KCSE <= Yr_JoinCampus"]'


def test_declared_rule_applied_by_construction() -> None:
    s = build_settings({}, {"rules": ["Yr_JoinCampus >= KCSE"]})
    tp = _profile(_years(), s)
    msgs = st.cross_file_structure([_FR("t.csv", tp)], s)
    assert not [m for m in msgs if m.level == "warning"]
    assert tp.structure.declared_rules[0]["status"] == "applied"
    assert sorted(tp.structure.swap_groups) == [[0, 1]]
    out = _generate(tp, s)
    a = np.array([int(v) for v in out["KCSE"]])
    b = np.array([int(v) for v in out["Yr_JoinCampus"]])
    assert np.all(b >= a)
    assert set(np.unique(b - a).tolist()) <= {0, 1, 2, 5}


def test_unusable_rules_warn() -> None:
    s = build_settings({}, {"rules": ["KCSE <= nope", {"rule": "KCSE <= score", "file": "other.csv"}]})
    tp = _profile(_years(), s)
    msgs = st.cross_file_structure([_FR("t.csv", tp)], s)
    assert sum(m.code == "rule_unused" for m in msgs) == 2
    with pytest.raises(UsageError):
        st.cross_file_structure([_FR("t.csv", tp)], build_settings({}, {"rules": ["KCSE ~ x"]}))


class _FR:
    """Minimal FileResult stand-in for cross_file_structure."""

    def __init__(self, rel: str, *tables) -> None:
        self.rel_path = rel
        self.tables = list(tables)
        self.failed = False


# --------------------------------------------------------------------------- panels (D16)
def test_panels_and_multi_year_snapshots() -> None:
    tp = _profile(_visits())
    assert [p["kind"] for p in tp.structure.panels] == ["panel"]
    assert any(m.code == "panel_detected" for c in tp.columns for m in c.messages)
    rng = np.random.default_rng(1)
    rows = []
    for s in range(80):
        econ = rng.random() < 0.5
        for y in ("2022-23", "2023-24", "2024-25"):
            econ = econ if rng.random() > 0.2 else not econ
            rows.append([5000 + s, y, "Y" if econ else "N"])
    tp = _profile(_csv(["student_id", "school_year", "econ"], rows))
    p = tp.structure.panels[0]
    assert p["kind"] == "multi_year" and p["changing_attributes"] == ["econ"]


def test_aggregate_table_warning() -> None:
    rows = [[s, g, n, "*" if n < 10 else f"{50 + n % 40}.5"] for s in ("577", "530") for g, n in
            (("All", 400), ("Female", 210), ("Male", 190), ("EL", 7), ("SPED", 33))]
    tp = _profile(_csv(["school_code", "subgroup", "n_tested", "pct_proficient"], rows))
    assert [p["kind"] for p in tp.structure.panels] == ["aggregate"]
    assert "suppression" in tp.structure.panels[0]["text"]


# --------------------------------------------------------------------------- sort order, duplicates
def test_sort_keys_detection() -> None:
    rng = np.random.default_rng(3)
    rows = [[str(rng.choice(["530", "577"])), int(rng.integers(0, 50)), 100 + i, str(rng.choice(["x", "y"]))]
            for i in range(80)]
    rows.sort(key=lambda r: (r[0], -r[1], r[2]))
    tp = _profile(_csv(["school", "score", "sid", "tag"], rows))
    keys = [(k["column"], k["direction"]) for k in tp.structure.sort_order["keys"]]
    assert keys == [("school", "asc"), ("score", "desc"), ("sid", "asc")]
    out = _generate(tp)
    syn = list(zip(out["school"], (int(v) for v in out["score"]), (int(v) for v in out["sid"])))
    assert syn == sorted(syn, key=lambda r: (r[0], -r[1], r[2]))


def test_unsorted_table_has_no_sort_keys_but_leading_missing_block() -> None:
    rng = np.random.default_rng(4)
    rows = [[str(rng.choice(["a", "b", "c"])), int(rng.integers(0, 90)), "" if i < 25 else str(rng.choice(["u", "v"]))]
            for i in range(100)]
    tp = _profile(_csv(["grp", "pts", "late"], rows))
    so = tp.structure.sort_order
    assert "keys" not in so and so["leading_missing"] == [{"column": "late", "rows": 25}]
    out = _generate(tp)
    assert out["late"][:25] == [""] * 25 and "" not in out["late"][25:]


def test_duplicate_rows_reproduced() -> None:
    rng = np.random.default_rng(6)
    base = {(str(rng.choice(["F", "M"])), str(rng.choice(list("ABCDE"))), int(rng.integers(0, 500))) for _ in range(400)}
    base = sorted(base)[:200]
    rows = [list(r) for r in base] + [list(base[i]) for i in rng.choice(len(base), 12, replace=False)]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    tp = _profile(_csv(["g", "r", "v"], rows))
    assert tp.structure.duplicate_rows["rows"] == 12
    out = _generate(tp)
    syn = list(zip(out["g"], out["r"], out["v"]))
    assert len(syn) - len(set(syn)) == 12
    for col in ("g", "r"):
        assert Counter(out[col]) == Counter(r[("g", "r").index(col)] for r in rows)


def test_describe_lines_are_names_and_counts_only() -> None:
    tp = _profile(_csv(["school_id", "school_name", "gender", "score", "advisor"], _school_rows(400)))
    lines = st.describe_structure(tp)
    assert any("functional dependency school_id -> school_name" in ln for ln in lines)
    assert not any(v in " ".join(lines) for v in SCHOOLS.values())


def test_swap_groups_merge_overlaps() -> None:
    sp = st.StructureProfile()
    sp.private = {"fds": [{"det": 0, "dep": 1, "mapping": {}}, {"det": 0, "dep": 2, "mapping": {}}],
                  "templates": [{"target": 5, "parts": (("f", 2, "as_is"), ("s", "."), ("f", 4, "lower"))}]}
    assert st.swap_groups(sp, []) == [[0, 1, 2, 4, 5]]


def test_template_constant_within_id_stays_constant() -> None:
    rng = np.random.default_rng(8)
    fn = ["Ann", "Bob", "Cy", "Dee", "Eve", "Fay"]
    ln = ["Lee", "Moss", "Nash", "Oak", "Pike"]
    rows = []
    for s in range(60):
        f, la = str(rng.choice(fn)), str(rng.choice(ln))
        dom = "a.example-isd.org" if s % 3 else "b.example-isd.org"
        for _ in range(1 + s % 4):
            rows.append([7000 + s, f, la, f"{f.lower()}{la.lower()}{7000 + s}@{dom}", int(rng.integers(0, 9))])
    tp = _profile(_csv(["student_id", "first_name", "last_name", "email", "pts"], rows))
    assert tp.columns[3].role == "entity" and tp.structure.derived_templates[0]["status"] == "applied"
    out = _generate(tp)
    per: dict[str, set[str]] = {}
    for sid, e in zip(out["student_id"], out["email"]):
        per.setdefault(sid, set()).add(e)
    assert all(len(v) == 1 for v in per.values())
    doms = Counter(next(iter(v)).split("@")[1] for v in per.values())
    assert set(doms) <= {"example.org", "example.com"} and sorted(doms.values()) == [20, 40]


def test_sort_desc_with_missing_last() -> None:
    rng = np.random.default_rng(9)
    rows = [[int(rng.integers(0, 100)) if i < 70 else "", str(rng.choice(["a", "b"]))] for i in range(90)]
    rows[:70] = sorted(rows[:70], key=lambda r: -r[0])
    tp = _profile(_csv(["points", "tag"], rows))
    k = tp.structure.sort_order["keys"][0]
    assert (k["column"], k["direction"], k["missing"]) == ("points", "desc", "last")
    out = _generate(tp)
    vals = out["points"]
    assert vals[70:] == [""] * 20
    assert [int(v) for v in vals[:70]] == sorted((int(v) for v in vals[:70]), reverse=True)


def test_unlinked_key_column_warning() -> None:
    s = build_settings({"id_col": ["a.csv::buddy"]})
    ta = _profile(_csv(["buddy", "x"], [[100 + i, i % 3] for i in range(40)]), s, "a.csv")
    tb = _profile(_csv(["buddy", "y"], [[100 + i % 30, i % 2] for i in range(50)]), s, "b.csv")
    st.cross_file_structure([_FR("a.csv", ta), _FR("b.csv", tb)], s)
    msgs = [m for c in tb.columns for m in c.messages if m.code == "unlinked_key_column"]
    assert msgs and msgs[0].fix == "--id-col b.csv::buddy"
