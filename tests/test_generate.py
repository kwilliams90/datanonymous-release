"""Generation: example acceptance (criterion 4) and generator unit tests (design 8.3-8.5)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from datanonymous.detect import text as dtext
from datanonymous.errors import IdSpaceError
from datanonymous.generate import quota
from datanonymous.generate.ids import IdNamespace, IdRegistry, build_mapping
from datanonymous.generate.numeric import generate_continuous
from datanonymous.generate.table import is_level_generated
from datanonymous.model import ColumnModel, ColumnType as T, MissingnessModel
from datanonymous.profile.structure import profile_missingness, sample_missing_mask

from conftest import EXAMPLE_FILES, EXAMPLES, read_dicts, read_rows, run_cli
from test_detect import profile_example


# --------------------------------------------------------------------------- example acceptance (criterion 4)
@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_same_header_order_and_row_count(example_run, name: str) -> None:
    src = (EXAMPLES / name).read_bytes()
    out = (example_run.out / name).read_bytes()
    first = lambda b: b.split(b"\n", 1)[0]  # noqa: E731
    assert first(out) == first(src)
    assert out.count(b"\n") == src.count(b"\n")
    assert out.count(b"\r\n") == src.count(b"\r\n")
    assert len(read_rows(example_run.out / name)) == len(read_rows(EXAMPLES / name))


@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_level_columns_exact_counts_and_extremes(example_run, name: str) -> None:
    cols, _ = profile_example(name)
    real = read_dicts(EXAMPLES / name)
    syn = read_dicts(example_run.out / name)
    for c, p in cols.items():
        r = [row[c] for row in real]
        s = [row[c] for row in syn]
        if is_level_generated(p):
            # every factor/ordinal/boolean level and discrete integer with its exact source count; a level the D35
            # allowlist did not keep appears with its count under a pseudonym
            pseudo = dtext.pseudonymized(p)
            kept_r = Counter(v for v in r if v.strip() not in pseudo)
            assert kept_r == Counter(v for v in s if v in kept_r), c
            assert sorted(Counter(v for v in r if v.strip() in pseudo).values()) == sorted(
                Counter(v for v in s if v not in kept_r).values()), c
        if p.type in (T.INTEGER, T.INTEGER_YEAR):
            rv, sv = [float(x) for x in r], [float(x) for x in s]
            assert min(sv) == min(rv) and max(sv) == max(rv), c
        if p.type is T.INTEGER and p.subtype == "continuous":
            rv, sv = np.array([float(x) for x in r]), np.array([float(x) for x in s])
            assert sv.min() >= rv.min() and sv.max() <= rv.max()
            unseen = np.mean(~np.isin(sv, rv))
            assert unseen >= 0.8, (c, unseen)


def test_no_real_names_or_emails(example_run) -> None:
    real = read_dicts(EXAMPLES / "student-scores.csv")
    syn = read_dicts(example_run.out / "student-scores.csv")
    for c in ("first_name", "last_name", "email"):
        assert not ({r[c] for r in real} & {s[c] for s in syn}), c
    text = (example_run.out / "student-scores.csv").read_text(encoding="utf-8")
    # e-mails are rebuilt from the synthetic names on reserved domains only (D5 derived template, D24)
    assert "gslingacademy" not in text
    assert {s["email"].split("@")[1] for s in syn if "@" in s["email"]} <= {"example.org", "example.com", "example.net"}
    # ids: sequential 1..2000 stays 1..2000 in order
    assert [s["id"] for s in syn] == [str(i) for i in range(1, 2001)]


@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_no_synthetic_row_equals_a_real_row(example_run, name: str) -> None:
    real = read_rows(EXAMPLES / name)
    syn = read_rows(example_run.out / name)
    header = real[0]
    keep = [i for i, h in enumerate(header) if h != "id"]
    rset = {tuple(r[i] for i in keep) for r in real[1:]}
    assert not any(tuple(s[i] for i in keep) in rset for s in syn[1:])


def test_same_seed_identical_different_seed_differs(example_run, tmp_path: Path) -> None:
    out2 = tmp_path / "o2"
    res = run_cli(EXAMPLES, "--out-dir", out2, "--report-dir", tmp_path / "r2", "--seed", "1")
    assert res.exit_code == 0
    for f in example_run.out.iterdir():
        assert (out2 / f.name).read_bytes() == f.read_bytes(), f.name
    out3 = tmp_path / "o3"
    res = run_cli(EXAMPLES, "--out-dir", out3, "--report-dir", tmp_path / "r3", "--seed", "2")
    assert res.exit_code == 0
    for name in EXAMPLE_FILES:
        assert (out3 / name).read_bytes() != (example_run.out / name).read_bytes()


# --------------------------------------------------------------------------- quota and floors
def test_largest_remainder_exact() -> None:
    a = quota.largest_remainder([5, 3, 2], 10)
    assert a.tolist() == [5, 3, 2]
    a = quota.largest_remainder([1, 1, 1], 10)
    assert a.sum() == 10 and max(a) - min(a) <= 1


def test_floor_sets_adjusted_only_when_needed() -> None:
    alloc, adj = quota.allocate([90, 9, 1], 100)
    assert alloc.tolist() == [90, 9, 1] and not adj
    alloc, adj = quota.allocate([97, 2, 1], 10)
    assert (alloc >= 1).all() and alloc.sum() == 10 and adj


def test_quota_sample_shuffled_and_exact() -> None:
    rng = np.random.default_rng(0)
    vals, adj = quota.quota_sample(["a", "b"], [3, 7], 10, rng)
    assert Counter(vals.tolist()) == {"a": 3, "b": 7} and not adj


def test_continuous_extremes_and_bounds() -> None:
    xs = np.sort(np.random.default_rng(1).lognormal(10, 1, 200).round())
    model = ColumnModel(sorted_numeric=xs)
    y, adj = generate_continuous(model, 200, np.random.default_rng(2), 0)
    assert y.min() == xs.min() and y.max() == xs.max()
    y2, _ = generate_continuous(model, 200, np.random.default_rng(2), 0, exact_extremes=False)
    assert y2.min() >= xs.min() and y2.max() <= xs.max()


# --------------------------------------------------------------------------- IDs
def _ns(values: list[str]) -> IdNamespace:
    reg = IdRegistry()
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    reg.register("k", "f.csv", None, 0, "student_id", arr)
    return reg.namespaces["k"]


def test_ids_sequential_identity() -> None:
    ns = _ns([str(i) for i in range(1, 51)])
    build_mapping(ns, np.random.default_rng(0))
    assert all(ns.mapping[k] == int(k) for k in ns.mapping)


def test_ids_same_shape_never_real_order_preserving() -> None:
    real = [f"{i:07d}" for i in (12, 4567, 89012, 123456, 999)]
    ns = _ns(real)
    build_mapping(ns, np.random.default_rng(0))
    syn = [ns.mapping[str(int(r))] for r in real]
    assert not set(syn) & {int(r) for r in real}
    order = sorted(range(len(real)), key=lambda i: int(real[i]))
    assert [syn[i] for i in order] == sorted(syn)


def test_ids_patterned() -> None:
    real = [f"S{i:06d}" for i in range(1, 300, 7)]
    ns = _ns(real)
    build_mapping(ns, np.random.default_rng(0))
    for r in real:
        v = ns.mapping[r.casefold()]
        assert v[0] == "S" and len(v) == len(r) and v[1:].isdigit() and v != r


def test_id_space_exhausted() -> None:
    # 2-digit space nearly full: unpadded integers take three digits (D9; K-02 harden 2 A2-6), zero-padded ones FAIL
    ns = _ns([str(i) for i in range(10, 100) if i != 50] + ["5"])
    ns.raw_values.update({str(i) for i in range(10, 100)})
    build_mapping(ns, np.random.default_rng(0))
    syn = list(ns.mapping.values())
    real = {int(k) for k in ns.representatives}
    assert len(set(syn)) == len(syn) and not set(syn) & real and ns.notes
    assert max(len(str(v)) for v in syn) == 3
    ns = _ns([f"{i:03d}" for i in range(10, 1000) if i != 50] + ["005"])  # every padded 3-character ID but 050
    with pytest.raises(IdSpaceError, match="zero-padded IDs keep their width"):
        build_mapping(ns, np.random.default_rng(0))


# --------------------------------------------------------------------------- missingness
def test_missingness_patterns_exact_and_pooled() -> None:
    n = 200
    a = np.zeros(n, dtype=bool)
    a[:40] = True
    b = a.copy()  # co-missing with a
    c = np.zeros(n, dtype=bool)
    c[100:103] = True  # rare pattern -> pooled
    model = profile_missingness({0: a, 1: b, 2: c}, n, 10)
    assert len(model.pooled) == 1
    m = sample_missing_mask(model, n, np.random.default_rng(0))
    assert m[:, 0].sum() == 40 and m[:, 1].sum() == 40 and m[:, 2].sum() == 3
    assert (m[:, 0] == m[:, 1]).all()  # co-missing kept


def test_empty_missingness_model() -> None:
    m = sample_missing_mask(MissingnessModel(), 5, np.random.default_rng(0))
    assert m.shape == (5, 0)


# --------------------------------------------------------------------------- long format, cross-file IDs
def _long_fixture(d: Path) -> None:
    rng = np.random.default_rng(11)
    enr = ["student_id,gender,race,school_id,score,exit_date,exit_code"]
    stu = ["StudentID,lunch"]
    sid = 4000000
    for s in range(150):
        sid += int(rng.integers(1, 40))
        g, r = rng.choice(["F", "M"]), rng.choice(["W", "B", "H", "NA"])
        if s % 2 == 0:
            stu.append(f"{sid},{rng.choice(['free', 'paid'])}")
        for _ in range(int(rng.integers(1, 6))):
            sc = "NA" if rng.random() < 0.1 else str(int(rng.integers(200, 400)))
            ex = rng.random() < 0.2
            enr.append(f"{sid:08d},{g},{r},{rng.choice(['577', '530'])},{sc},{'2024-03-01' if ex else ''},"
                       f"{rng.choice(['W1', 'W2']) if ex else ''}")
    (d / "enrollment.csv").write_text("\n".join(enr) + "\n", encoding="utf-8", newline="")
    (d / "students.csv").write_text("\n".join(stu) + "\n", encoding="utf-8", newline="")


def test_long_format_ids_constants_missingness(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    _long_fixture(src)
    out = tmp_path / "out"
    res = run_cli(src, "--out-dir", out, "--report-dir", tmp_path / "rep", "--seed", "3")
    assert res.exit_code == 0, res.output
    real = read_dicts(src / "enrollment.csv")
    syn = read_dicts(out / "enrollment.csv")
    # rows-per-ID multiset identical, IDs same shape and never real
    rc = Counter(Counter(r["student_id"] for r in real).values())
    sc = Counter(Counter(s["student_id"] for s in syn).values())
    assert rc == sc
    assert not {r["student_id"] for r in real} & {s["student_id"] for s in syn}
    assert all(len(s["student_id"]) == 8 and s["student_id"].isdigit() for s in syn)
    # within-ID constants hold
    for col in ("gender", "race"):
        per = {}
        for s in syn:
            per.setdefault(s["student_id"], set()).add(s[col])
        assert all(len(v) == 1 for v in per.values()), col
    # co-missing exit_date/exit_code kept; NA tokens reproduced exactly
    assert all((s["exit_date"] == "") == (s["exit_code"] == "") for s in syn)
    assert Counter(r["score"] == "NA" for r in real) == Counter(s["score"] == "NA" for s in syn)
    assert Counter(r["race"] for r in real)["NA"] > 0 and Counter(s["race"] for s in syn)["NA"] > 0
    # cross-file consistency (D8): padded 8-digit and unpadded IDs map to the same synthetic entity
    syn_ids = {int(s["student_id"]) for s in syn}
    stu_syn = read_dicts(out / "students.csv")
    assert {int(s["StudentID"]) for s in stu_syn} <= syn_ids
    assert len(stu_syn) == len(read_dicts(src / "students.csv"))


# --------------------------------------------------------------------------- privacy swap-redraw (D23)
def _combo_file(path: Path, n: int, sizes: tuple[int, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = [",".join(f"c{i}" for i in range(len(sizes)))]
    for _ in range(n):
        lines.append(",".join(f"v{int(rng.integers(s))}" for s in sizes))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def test_swap_redraw_removes_matches_keeps_marginals(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    _combo_file(src / "combo.csv", 500, (10, 10, 10, 5), 1)
    res = run_cli(src, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0, res.output
    real = read_rows(src / "combo.csv")[1:]
    syn = read_rows(tmp_path / "o" / "combo.csv")[1:]
    assert not {tuple(r) for r in real} & {tuple(s) for s in syn}
    for i in range(4):
        assert Counter(r[i] for r in real) == Counter(s[i] for s in syn)
    import json
    priv = json.loads((tmp_path / "r" / "files" / "combo.csv.json").read_text(encoding="utf-8"))["tables"][0]["privacy"]
    assert priv["initial_matches"] > 0 and priv["remaining_matches"] == 0 and not priv["report_only"]


def test_report_only_and_gate(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    _combo_file(src / "tiny.csv", 100, (2, 2), 2)
    res = run_cli(src, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0  # above 20% expected matches: reported, not swapped
    assert "report" in (tmp_path / "r" / "REPORT.md").read_text(encoding="utf-8")
    res = run_cli(src, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "1",
                  "--max-row-matches", "5")
    assert res.exit_code == 3
    assert not (tmp_path / "o2" / "tiny.csv").exists()


def test_include_exclude_globs(tmp_path: Path, example_copy: Path) -> None:
    (example_copy / "sub").mkdir()
    (example_copy / "sub" / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    res = run_cli(example_copy, "--out-dir", tmp_path / "o", "--seed", "1", "--exclude-path", "sub/*",
                  "--include", "*.csv", "--exclude-path", "student-scores.csv")
    assert res.exit_code == 0, res.output
    written = sorted(p.name for p in (tmp_path / "o").iterdir() if p.suffix == ".csv")
    assert written == ["StudentsPerformance.csv", "students_data.csv"]


def test_swap_groups_move_together(tmp_path: Path) -> None:
    from datanonymous.config import Settings
    from datanonymous.generate.ids import IdRegistry
    from datanonymous.generate.table import GenContext, generate_table
    from datanonymous.io import csvio
    from datanonymous.privacy import matches
    from datanonymous.profile.table import profile_table

    _combo_file(tmp_path / "combo.csv", 500, (10, 10, 10, 5), 1)
    s = Settings(mode="fidelity")
    r = csvio.parse_text_bytes((tmp_path / "combo.csv").read_bytes())
    tp = profile_table(r.tables[0], "combo.csv", "csv", s)
    tp.structure.swap_groups = [[0, 1]]  # e.g. an FD pair found by AREA D
    st = generate_table(tp, GenContext(1, "combo.csv", IdRegistry(), {}, s))
    before = Counter(zip(st.columns[0].values.tolist(), st.columns[1].values.tolist()))
    priv = matches.enforce(tp, st, np.random.default_rng(0), None)
    after = Counter(zip(st.columns[0].values.tolist(), st.columns[1].values.tolist()))
    assert priv["swaps"] > 0 and priv["remaining_matches"] == 0
    assert before == after


# --------------------------------------------------------------------------- ID registry (D8/D9)
def _arr(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def test_dense_ids_with_a_gap_spill_to_other_widths() -> None:
    real = [str(i) for i in range(1, 2001) if i != 50]
    ns = _ns(real)
    build_mapping(ns, np.random.default_rng(0))
    syn = [ns.mapping[k] for k in sorted(ns.mapping, key=int)]
    assert not ns.identity and ns.notes
    assert len(set(syn)) == len(real)
    assert not set(syn) & {int(r) for r in real}
    assert max(len(str(s)) for s in syn) <= 4  # within the observed width range


def test_id_space_too_small_over_the_whole_width_range() -> None:
    # 1..9 repeated: person keys, only '0' is free in their width: unpadded integers take two digits (D9: noted, no
    # FAIL; K-02 harden 2 A2-6); zero-padded 01..09 cannot (D9 FAILs clearly)
    ns = _ns([str(i) for i in range(1, 10)] * 2)
    build_mapping(ns, np.random.default_rng(0))
    syn = list(ns.mapping.values())
    assert len(set(syn)) == 9 and not set(syn) & set(range(1, 10)) and ns.notes
    ns = _ns([f"{i:02d}" for i in range(1, 100)] * 2)
    with pytest.raises(IdSpaceError):
        build_mapping(ns, np.random.default_rng(0))


def test_repeated_1_to_n_person_keys_are_remapped() -> None:
    ns = _ns([str(i) for i in range(1, 201) for _ in range(2)])
    build_mapping(ns, np.random.default_rng(0))
    assert not ns.identity
    assert not set(ns.mapping.values()) & set(range(1, 201))


def test_identity_is_a_namespace_property_shared_with_child_files() -> None:
    reg = IdRegistry()
    reg.register("studentid", "students.csv", None, 0, "student_id", _arr([str(i) for i in range(1, 201)]))
    child = _arr([str(i) for i in range(5, 201, 3)] * 2)
    reg.register("studentid", "scores.csv", None, 0, "student_id", child)
    reg.build(1)
    ns = reg.namespaces["studentid"]
    assert ns.identity and ns.error is None
    assert reg.render("studentid", child).tolist() == child.tolist()


def test_digit_and_patterned_ids_share_a_namespace() -> None:
    reg = IdRegistry()
    digits = _arr([str(i) for i in range(1, 201) for _ in range(2)])
    pattern = _arr([f"S{i:04d}" for i in range(1, 60)])
    reg.register("studentid", "a.csv", None, 0, "student_id", digits)
    reg.register("studentid", "b.csv", None, 0, "student_id", pattern)
    reg.build(2)
    assert reg.namespaces["studentid"].error is None
    a, b = reg.render("studentid", digits), reg.render("studentid", pattern)
    assert not set(a.tolist()) & set(digits.tolist()) and not set(b.tolist()) & set(pattern.tolist())
    assert all(v.isdigit() for v in a.tolist()) and all(v[0] == "S" and len(v) == 5 for v in b.tolist())


def test_render_keeps_case_punctuation_and_whitespace_per_raw_form() -> None:
    reg = IdRegistry()
    a = _arr(["s79521", "S-79522", " S79523 ", "S79524"])
    b = _arr(["S79521", "s79522"])
    reg.register("studentid", "demo.csv", None, 0, "student_id", a)
    reg.register("studentid", "enroll.csv", None, 0, "student_id", b)
    reg.build(3)
    ra, rb = reg.render("studentid", a).tolist(), reg.render("studentid", b).tolist()
    from datanonymous.detect.ids import char_mask, normalize_id

    for raw, syn in zip(a.tolist() + b.tolist(), ra + rb):
        assert char_mask(syn) == char_mask(raw) and syn != raw  # same case, punctuation, whitespace, width
    assert normalize_id(ra[0]) == normalize_id(rb[0]) and normalize_id(ra[1]) == normalize_id(rb[1])


def test_id_space_error_names_columns_and_a_qualified_fix() -> None:
    reg = IdRegistry()
    # zero-padded IDs keep their width (D9): every 2-character ID 01-99 is real, so no synthetic one is left (an
    # unpadded 1..9 would take two digits instead, K-02 harden 2 A2-6)
    reg.register("studentid", "sub/a.csv", None, 0, "student_id", _arr([f"{i:02d}" for i in range(1, 100)] * 2))
    reg.build(1)
    err = reg.namespaces["studentid"].error
    assert err is not None and "'student_id'" in err.message and "sub/a.csv" in err.message
    assert "--null sub/a.csv::student_id" in err.fix and "--force-text" not in err.fix


# --------------------------------------------------------------------------- text generators
def test_free_text_keeps_distinct_count_frequency_profile_and_lengths() -> None:
    from datanonymous.generate import text
    from datanonymous.model import ColumnProfile

    counts = {"call home": 133, "met with parent": 133, "absent excuse": 130, "ok": 115}
    model = ColumnModel(nonmissing_counts=dict(counts), real_values=frozenset(counts))
    p = ColumnProfile(index=0, name="note", type=T.FREE_TEXT)
    vals = text.lorem(p, model, sum(counts.values()), np.random.default_rng(4))
    c = Counter(vals.tolist())
    assert sorted(c.values()) == sorted(counts.values())
    assert not set(c) & set(counts)
    assert sorted(len(s) for s in c) == sorted(len(s) for s in counts)


def test_generate_text_extended_returns_flags() -> None:
    from datanonymous.generate import text
    from datanonymous.model import ColumnProfile

    model = ColumnModel(nonmissing_counts={"ELL;SPED": 5, "ELL": 3}, real_values=frozenset({"ELL;SPED", "ELL"}))
    p = ColumnProfile(index=0, name="programs", type=T.MULTI_SELECT)
    values, flags = text.generate_text_extended(p, model, 8, np.random.default_rng(0))
    assert len(values) == 8 and flags == {"rare_category_adjusted": False}


def test_within_id_constant_keeps_a_minority_level(tmp_path: Path) -> None:
    rows = ["student_id,gender,score"]
    rng = np.random.default_rng(3)
    for s in range(1, 151):
        g = "F" if s % 2 else "M"
        for k in range(2):
            rows.append(f"{1000 + 3 * s},{'X' if (s == 7 and k == 1) else g},{int(rng.integers(0, 100))}")
    d = tmp_path / "in"
    d.mkdir()
    (d / "l.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    out, rep = tmp_path / "o", tmp_path / "r"
    assert run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1").exit_code == 0
    syn = read_dicts(out / "l.csv")
    assert {r["gender"] for r in syn} == {"F", "M", "X"}
    import json

    cols = json.loads((rep / "files" / "l.csv.json").read_text(encoding="utf-8"))["tables"][0]["columns"]
    gender = next(c for c in cols if c["name"] == "gender")
    assert gender["role"] == "entity" and gender["flags"]["rare_category_adjusted"]
