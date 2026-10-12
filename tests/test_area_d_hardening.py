"""AREA D hardening: adversarial inputs and regression tests for the IDs/entities/structure area."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as hst

from datanonymous.config import Settings
from conftest import build_fidelity_settings as build_settings
from datanonymous.detect.ids import normalize_id
from datanonymous.errors import IdSpaceError
from datanonymous.generate import structure as gs
from datanonymous.generate.ids import IdRegistry, allocate_digit_ids
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.privacy import matches
from datanonymous.profile import structure as st
from test_area_d_ids import _arr, _reason
from test_area_d_structure import _FR, _csv, _generate, _profile


# --------------------------------------------------------------------------- synthetic digit IDs (D8/D9)
def test_plain_key_inside_a_padded_column_keeps_its_own_width() -> None:
    """A plain '7' written in a zero-padded column is bounded to 1 digit: it must stay 1 digit wide."""
    reg = IdRegistry()
    other = _arr([f"{i:04d}" for i in range(2, 30)])  # keys 2..29, always padded: one free 1-digit value
    mixed = _arr(["7", "00020", "00033", "00046"])  # '7' plain inside a padded column
    reg.register("k", "a.csv", None, 0, "student_id", other)
    reg.register("k", "b.csv", None, 0, "student_id", mixed)
    reg.build(3)
    ns = reg.namespaces["k"]
    assert ns.error is None and not ns.notes  # no visible width change was needed
    out = reg.render("k", mixed).tolist()
    assert [len(v) for v in out] == [1, 5, 5, 5]
    assert all(len(v) == 4 for v in reg.render("k", other).tolist())


def test_bounds_respected_when_spilling_wider() -> None:
    """Keys move only to widths their padding hides; plain keys keep their number of digits."""
    keys = [str(i) for i in range(1, 10)] + [str(i) for i in range(10, 100)]
    nat = {k: len(k) for k in keys}
    bound = {k: 2 for k in keys if len(k) == 1}  # 1-digit keys written '01'..'09': may become 2 digits
    bound["5"] = 1  # but '5' is also written plainly as '5' in a padded column
    real = np.asarray(sorted(int(k) for k in keys) + [0], dtype=np.int64)  # 1-digit space: no free value
    real = real[real != 5]
    notes: list[str] = []
    # 2-digit space is full too: 90 keys + 8 movers > 90 values, and the padded 1-digit keys may not take a third
    # digit (an unpadded integer may: D9, K-02 harden 2 A2-6)
    with pytest.raises(IdSpaceError):
        allocate_digit_ids(keys, nat, bound, np.sort(real), np.random.default_rng(0), notes,
                           pad_limit={k: 2 for k in keys if len(k) == 1})
    keys2 = [str(i) for i in range(1, 10)] + ["10", "11"]
    nat2 = {k: len(k) for k in keys2}
    bound2 = {k: 2 for k in keys2 if len(k) == 1}
    bound2["5"] = 1
    real2 = np.asarray(sorted({int(k) for k in keys2 if k != "5"} | {0}), dtype=np.int64)
    out = allocate_digit_ids(keys2, nat2, bound2, real2, np.random.default_rng(0), notes)
    got = dict(zip(keys2, out.tolist()))
    assert got["5"] == 5 or len(str(got["5"])) == 1  # its only free 1-digit value
    assert all(len(str(got[k])) <= 2 for k in keys2) and not notes
    assert not set(got.values()) & {int(k) for k in keys2 if k != "5"} and len(set(got.values())) == len(keys2)


def test_no_all_zero_synthetic_id_unless_needed() -> None:
    """'0' / '0000' reads as a missing ID: it is used only when the other 1-digit values are all real."""
    for seed in range(20):
        reg = IdRegistry()
        col = _arr([f"{i:04d}" for i in (1, 3, 5)] + ["7", "00042", "00042"])
        reg.register("k", "f.csv", None, 0, "student_id", col)
        reg.build(seed)
        assert all(v.strip("0") for v in reg.render("k", col).tolist()), seed
    reg = IdRegistry()
    col = _arr([str(i) for i in range(1, 10)] + ["5" + "0" * 3])  # every non-zero digit is real
    reg.register("k", "f.csv", None, 0, "student_id", col)
    reg.build(1)
    assert reg.namespaces["k"].error is None and "0" in reg.render("k", col).tolist()


def test_digit_mapping_has_no_rank_linkage() -> None:
    """D8: the synthetic order is not the real order, so the mapping cannot be rebuilt by sorting both lists."""
    rng = np.random.default_rng(4)
    real = [f"{v:07d}" for v in sorted(rng.choice(np.arange(10_000, 999_999), 150, replace=False).tolist())]
    reg = IdRegistry()
    reg.register("k", "a.csv", None, 0, "student_id", _arr(real))
    reg.build(1)
    out = [int(v) for v in reg.render("k", _arr(real)).tolist()]
    ranks = np.argsort(np.argsort(out))
    assert abs(np.corrcoef(ranks, np.arange(len(out)))[0, 1]) < 0.4  # sorted input -> unsorted synthetic IDs
    near = sum(abs(o - int(r)) < 10_000 for o, r in zip(out, real))
    assert near < 15  # an order-preserving draw put ~all synthetic IDs next to their real ID


def test_sorted_file_stays_sorted_by_the_synthetic_ids(tmp_path) -> None:
    from conftest import read_dicts, run_cli

    rng = np.random.default_rng(2)
    ids = sorted(rng.choice(np.arange(100_000, 999_999), 200, replace=False).tolist())
    d = tmp_path / "in"
    d.mkdir()
    (d / "s.csv").write_text(_csv(["student_id", "g"], [[i, rng.choice(["a", "b"])] for i in ids]), encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "5")
    assert res.exit_code == 0, res.output
    syn = [int(r["student_id"]) for r in read_dicts(tmp_path / "o" / "s.csv")]
    assert syn == sorted(syn) and not set(syn) & set(ids)


@pytest.mark.parametrize("n_real", [45, 49])
def test_patterned_ids_in_a_nearly_full_mask_space(n_real: int) -> None:
    """T00-T99 holds 100 values: 49 real codes leave 51 free, enough for 49 synthetic ones (no spurious failure)."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        vals = [f"T{v:02d}" for v in rng.choice(100, n_real, replace=False)]
        reg = IdRegistry()
        reg.register("k", "f.csv", None, 0, "staff_id", _arr(vals))
        reg.build(seed)
        ns = reg.namespaces["k"]
        assert ns.error is None, seed
        out = reg.render("k", _arr(vals)).tolist()
        assert all(re.fullmatch(r"T\d\d", v) for v in out)
        assert not set(out) & set(vals) and len(set(out)) == n_real


def test_patterned_ids_fail_clearly_when_the_space_is_too_small() -> None:
    vals = [f"T{v:02d}" for v in range(60)]  # 60 real in a 100-value space: only 40 free
    reg = IdRegistry()
    reg.register("k", "f.csv", None, 0, "staff_id", _arr(vals))
    reg.build(1)
    err = reg.namespaces["k"].error
    assert err is not None and "T0" not in err.message and "T5" not in err.message  # no real value quoted


# --------------------------------------------------------------------------- ID candidates (D7, section 3 #13)
def test_uuid_and_hash_candidates_ignore_missing_like_placeholders() -> None:
    import hashlib
    import uuid

    rng = np.random.default_rng(1)
    uuids = [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(60)] + ["N/A", "NULL", "N/A"]
    assert _reason("record_ref", uuids) == "UUID values"
    hashes = [hashlib.sha1(rng.bytes(8)).hexdigest() for _ in range(60)] + ["NA", "NA"]
    assert _reason("fingerprint", hashes).startswith("40-character hex hash")


def test_unique_fixed_width_integers_are_candidates() -> None:
    """An unpadded unique 6-digit badge number is an INTEGER: its real values (or min/max) would be written."""
    rng = np.random.default_rng(3)
    badges = [str(v) for v in rng.choice(np.arange(100_000, 999_999), 80, replace=False)]
    assert _reason("badge", badges) == "unique fixed-width codes (6 characters)"
    # mixed-width unique amounts (Expense_Semester: 4-5 digits, 86% one width) stay plain numbers
    amounts = [str(v) for v in rng.choice(np.arange(5_000, 50_000), 62, replace=False)]
    amounts = [a for a in amounts if len(a) == 5][:53] + [str(v) for v in range(5_001, 5_010)]
    assert _reason("Expense_Semester", amounts) is None


# --------------------------------------------------------------------------- declared rules (D5)
def _entity_years(n: int = 80, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        a = int(rng.integers(2014, 2021))
        b = a + int(rng.choice([0, 1, 2, 5]))
        for _ in range(int(rng.integers(2, 5))):
            rows.append([1000 + 7 * s, a, b, int(rng.integers(0, 100))])
    return _csv(["student_id", "KCSE", "Yr_JoinCampus", "score"], rows)


def test_declared_rule_between_within_id_constants_is_drawn_once_per_entity() -> None:
    s = build_settings({}, {"rules": ["KCSE <= Yr_JoinCampus"]})
    tp = _profile(_entity_years(), s)
    assert tp.columns[1].role == tp.columns[2].role == "entity"
    st.cross_file_structure([_FR("t.csv", tp)], s)
    d = tp.structure.private["rule_data"][("KCSE", "Yr_JoinCampus", False)]
    assert d.get("per_entity") and len(d["gaps"]) == tp.entity.n_entities  # gaps counted per entity
    reg = IdRegistry()
    reg.register("student_id", "t.csv", None, 0, "student_id", tp.private.id_raw[0])
    reg.build(1)
    # before the privacy step (which rebuilds entity columns from first rows and would hide the bug)
    synth = generate_table(tp, GenContext(1, "t.csv", reg, {0: "student_id"}, s))
    cols = {c.name: c.values.tolist() for c in synth.columns}
    per_id: dict[str, set[str]] = {}
    for sid, b in zip(cols["student_id"], cols["Yr_JoinCampus"]):
        per_id.setdefault(sid, set()).add(b)
    assert all(len(v) == 1 for v in per_id.values())  # still constant within the ID (D4)
    assert all(int(b) >= int(a) for a, b in zip(cols["KCSE"], cols["Yr_JoinCampus"]))
    first = {}
    for sid, a, b in zip(cols["student_id"], cols["KCSE"], cols["Yr_JoinCampus"]):
        first.setdefault(sid, int(b) - int(a))
    real_gaps = Counter(int(g) for g in d["gaps"].tolist())
    # R3-06: Yr_JoinCampus keeps its own per-entity values, dealt to the entities so the per-entity gaps come
    # out as observed (the deal and its gap repair: at most a few entities off)
    assert sum((Counter(first.values()) - real_gaps).values()) <= 4, (Counter(first.values()), real_gaps)


# --------------------------------------------------------------------------- templates tied to an ID (D5)
def _keyed_rows(n: int = 400, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sid = 200_000 + 3 * i
        dept = str(rng.choice(["Math", "Arts", "Bio", "Chem", "Hist"]))
        rows.append([sid, dept, f"{dept.lower()}-{sid}", int(rng.integers(0, 101)), str(rng.choice(["F", "M"])),
                     str(rng.choice(["KG", "01", "02", "03"]))])
    return _csv(["student_id", "dept", "locker_tag", "score", "gender", "grade"], rows)


def test_template_with_an_id_input_survives_privacy_swaps() -> None:
    tp = _profile(_keyed_rows())
    tmpl = [t for t in tp.structure.derived_templates if t["column"] == "locker_tag"]
    assert tmpl and tmpl[0]["status"] == "applied" and "student_id" in tmpl[0]["inputs"]
    swaps = 0
    for seed in (1, 2, 3, 4):
        tp = _profile(_keyed_rows())
        reg = IdRegistry()
        reg.register("student_id", "t.csv", None, 0, "student_id", tp.private.id_raw[0])
        reg.build(seed)
        synth = generate_table(tp, GenContext(seed, "t.csv", reg, {0: "student_id"}, Settings(mode="fidelity")))
        swaps += matches.enforce(tp, synth, np.random.default_rng(seed), None)["swaps"]
        st.finalize_rows(tp, synth, lambda purpose: np.random.default_rng(seed + 1), Settings(mode="fidelity"))
        out = {c.name: c.values.tolist() for c in synth.columns}
        for sid, dept, tag in zip(out["student_id"], out["dept"], out["locker_tag"]):
            assert tag == f"{dept.lower()}-{sid}", (seed, sid, dept, tag)
    assert swaps  # the privacy step did move 'dept' (and the tag with it) between rows


def test_pinned_template_columns_never_move_for_duplicate_rows() -> None:
    tp = _profile(_keyed_rows())
    units = gs._row_units(tp, {p.index: None for p in tp.columns})
    moved = {c for u in units for c in u}
    assert not moved & {1, 2}  # dept and locker_tag stay with their student_id


# --------------------------------------------------------------------------- tiny and degenerate tables
@pytest.mark.parametrize("text", [
    "student_id,x\n",
    "student_id,x\n0012345,a\n",
    "student_id,x\n0012345,a\n0012345,b\n",
    "student_id,x,y\n,,\n,,\n,,\n",
    "student_id,x\n007,a\n007,a\n007,a\n",
    "student_id,x\n1,a\n",  # a lone '1' is not a 1..n row-number sequence: remapped (D9 keeps sequences only)
    "student_id,x\n001,a\n",
])
def test_degenerate_tables_generate_n_rows(text: str) -> None:
    tp = _profile(text)
    out = _generate(tp)
    n = text.count("\n") - 1
    assert all(len(v) == n for v in out.values())
    real = {normalize_id(r.split(",")[0]) for r in text.splitlines()[1:] if r.split(",")[0]}
    assert not {normalize_id(v) for v in out["student_id"] if v} & real


def test_settings_default_is_untouched_by_structure_detectors() -> None:
    s = Settings(mode="fidelity")
    _profile(_keyed_rows(60), s)
    assert s.rules is None or s.rules == []


def _decimal_rule_rows(n: int = 150, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        a = round(float(rng.uniform(0, 10)), 2)
        b = np.ceil(a * 10) / 10 + float(rng.choice([0.0, 0.0, 0.5, 1.2]))
        rows.append([f"{a:.2f}", f"{b:.1f}", int(rng.integers(0, 9))])
    return _csv(["hours_planned", "hours_logged", "k"], rows)


def test_declared_decimal_rule_survives_rounding_to_the_dependent_precision() -> None:
    s = build_settings({}, {"rules": ["hours_planned <= hours_logged"]})
    for seed in (1, 2, 3):
        tp = _profile(_decimal_rule_rows(seed=seed), s)
        assert tp.columns[0].type.value == "FLOAT" and tp.columns[1].type.value == "FLOAT"
        st.cross_file_structure([_FR("t.csv", tp)], s)
        out = _generate(tp, s, seed=seed)
        a = [float(v) for v in out["hours_planned"]]
        b = out["hours_logged"]
        assert all(len(v.split(".")[1]) == 1 for v in b)  # B keeps its one decimal
        assert all(float(y) >= x for x, y in zip(a, b)), seed


def test_declared_rule_keeps_the_dependent_exact_extremes() -> None:
    """D14: B's exact min and max are emitted at least once, by construction (an A value + an observed gap)."""
    rng = np.random.default_rng(8)
    rows = []
    for i in range(200):
        a = int(rng.integers(1, 51))
        rows.append([a, a + int(rng.choice([0, 1, 2])), int(rng.integers(0, 9))])
    rows[17] = [50, 80, 3]  # the only row reaching B's max: A = 50 with a gap of 30
    rows[42] = [1, 1, 3]  # B's min
    text = _csv(["credits_earned", "credits_attempted", "k"], rows)
    real_b = [r[1] for r in rows]
    s = build_settings({}, {"rules": ["credits_earned <= credits_attempted"]})
    for seed in range(1, 9):
        tp = _profile(text, s)
        st.cross_file_structure([_FR("t.csv", tp)], s)
        out = _generate(tp, s, seed=seed)
        a = [int(v) for v in out["credits_earned"]]
        b = [int(v) for v in out["credits_attempted"]]
        assert all(y >= x for x, y in zip(a, b))
        assert min(b) == min(real_b) and max(b) == max(real_b), seed
        assert tp.columns[1].flags["extreme_value_adjusted"] or max(b) == 80


def test_template_found_despite_missing_like_placeholders() -> None:
    rng = np.random.default_rng(0)
    first = ["Avery", "Blake", "Carmen", "Dante", "Elena", "Felix", "Greta", "Hugo", "Iris", "Jonas", "Kira", "Leon"]
    last = ["Ashby", "Brennan", "Calloway", "Dunmore", "Eastwick", "Fairley", "Garrow", "Hollis"]
    rows = []
    for i in range(1, 301):
        f, la = str(rng.choice(first)), str(rng.choice(last))
        rows.append([i, f, la, f"{f.lower()}.{la.lower()}.{i}@district.org" if rng.random() > 0.1 else "N/A"])
    tp = _profile(_csv(["id", "first_name", "last_name", "email"], rows))
    t = tp.structure.derived_templates
    assert t and t[0]["column"] == "email" and t[0]["status"] == "applied"
    assert t[0]["rows"] == sum(r[3] != "N/A" for r in rows)
    out = _generate(tp)
    for i, f, la, e in zip(out["id"], out["first_name"], out["last_name"], out["email"]):
        assert e == "N/A" or e.endswith(f".{i}@example.org")


def test_template_with_an_unpadded_id_in_a_padded_column() -> None:
    """jsmith.12345@... built from student_id '0012345': the ID is written without its zero padding."""
    rng = np.random.default_rng(6)
    first = ["Avery", "Blake", "Carmen", "Dante", "Elena", "Felix", "Greta", "Hugo"]
    rows = []
    for sid in rng.choice(np.arange(1_000, 999_999), 150, replace=False).tolist():
        f = str(rng.choice(first))
        for _ in range(int(rng.integers(1, 4))):
            rows.append([f"{sid:07d}", f, f"{f.lower()}.{sid}@school.example-isd.org", int(rng.integers(0, 100))])
    tp = _profile(_csv(["student_id", "first_name", "email", "score"], rows))
    t = tp.structure.derived_templates
    assert t and t[0]["status"] == "applied" and "no_leading_zeros(student_id)" in t[0]["template"]
    out = _generate(tp)
    for sid, f, e in zip(out["student_id"], out["first_name"], out["email"]):
        assert e.endswith(f".{int(sid)}@example.org") and len(sid) == 7
    per_id: dict[str, set[str]] = {}
    for sid, e in zip(out["student_id"], out["email"]):
        per_id.setdefault(sid, set()).add(e)
    assert all(len(v) == 1 for v in per_id.values())  # constant within the ID, like the source


# --------------------------------------------------------------------------- end to end: adversarial directory
_FIRST = ["Zorvath", "Quillane", "Éowynne", "O'Brannagh", "Mary-Kestrel", "Þóra", "Ximenaël", "Yusrielle"]
_LAST = ["Vantorre", "Kéllaway", "O'Dunphy", "Smythe-Orlov", "Ngwenyaz", "Østerbek", "Ibárruriz", "Quenneville"]
_TEACHERS = ["Mrs Alvara Teacherone", "Mr Brannoc Teachertwo", "Ms Cyrene Teacherthree", "Dr Dellwyn Teacherfour",
             "Mr Eskander Teacherfive", "Ms Fenwyck Teachersix"]


def _write_adversarial(d) -> dict[str, set[str]]:
    """Adversarial inputs; returns the real sensitive values per file."""
    import csv

    rng = np.random.default_rng(12)
    d.mkdir(parents=True, exist_ok=True)
    real: dict[str, set[str]] = {}
    rows = []
    for _ in range(90):
        sid = f"{int(rng.integers(1, 99_999)):06d}"
        f, la = str(rng.choice(_FIRST)), str(rng.choice(_LAST))
        email = f"{f.lower()}.{la.lower()}{int(rng.integers(1, 99))}@realschool.k12.us"
        start = int(rng.integers(2015, 2020))
        for _ in range(int(rng.integers(1, 4))):
            sec = int(rng.integers(0, len(_TEACHERS)))
            rows.append([sid, f, la, email, f"SEC{sec}", _TEACHERS[sec], start, start + int(rng.choice([3, 4, 5])),
                         int(rng.integers(0, 101))])
    real["a_crlf.csv"] = {v for r in rows for v in (r[0], r[1], r[2], r[3], r[5])}
    with open(d / "a_crlf.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\r\n")
        w.writerow(["student_id", "first_name", "last_name", "email", "section_code", "section_teacher",
                    "start_year", "grad_year", "score"])
        w.writerows(rows)
    rows = []
    for i in range(60):
        v = f"{i * 13 + 7:05d}" if i % 3 else str(i * 13 + 7)  # mixed padding in one column
        if i % 7 == 0:
            v = f" {v} "  # whitespace around the ID
        rows.append([v, str(10**19 + int(rng.integers(0, 10**9))), f"T{int(rng.choice(100)):02d}" if i < 40 else "",
                     str(int(rng.integers(0, 3)))])
    real["b_ids.csv"] = {v for r in rows for v in (r[0].strip(), r[1]) if v}
    with open(d / "b_ids.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows([["student_id", "state_id", "staff_id", "x"], *rows])
    (d / "c_header_only.csv").write_bytes(b"student_id,first_name\n")
    (d / "c_one_row.csv").write_bytes("student_id,first_name,email\n0012345,Zorvath,zorvath.q@realdomain.org\n"
                                      .encode("utf-8"))
    real["c_one_row.csv"] = {"0012345", "Zorvath", "zorvath.q@realdomain.org"}
    (d / "c_all_missing.csv").write_bytes(("student_id,a,b\n" + "".join(f"{i:04d},,NA\n" for i in range(1, 30)))
                                          .encode("utf-8"))
    return real


def _tokens(text: str) -> set[str]:
    return set(re.split(r"[\s,;|\"'()\[\]{}<>`=:]+", text)) | {line.strip() for line in text.splitlines()}


def test_adversarial_directory_end_to_end(tmp_path) -> None:
    from conftest import read_rows, run_cli

    src = tmp_path / "in"
    real = _write_adversarial(src)
    outs = []
    for k in (1, 2):
        out, rep = tmp_path / f"o{k}", tmp_path / f"r{k}"
        res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "21")
        assert res.exit_code == 0, res.output
        outs.append((out, rep))
    (o1, r1), (o2, r2) = outs
    for p in sorted(src.iterdir()):
        a, b = (o1 / p.name).read_bytes(), (o2 / p.name).read_bytes()
        assert a == b, p.name  # same seed -> same bytes (D26)
        assert len(read_rows(o1 / p.name)) == len(read_rows(p))  # N rows in = N rows out (D15)
        assert (b"\r\n" in a) == (b"\r\n" in p.read_bytes())  # line endings mirrored
    sensitive = {v for v in set().union(*real.values()) if len(v) >= 4}  # short numbers occur in scores
    for path in [q for q in o1.rglob("*") if q.is_file()] + [q for q in r1.rglob("*") if q.is_file()]:
        text = path.read_text(encoding="utf-8")
        hits = _tokens(text) & sensitive
        hits |= {v for v in sensitive if len(v) >= 8 and not v.isdigit() and v in text}
        assert not hits, (path.name, sorted(hits)[:5])
    by_section: dict[str, set[str]] = {}
    for r in read_rows(o1 / "a_crlf.csv")[1:]:
        by_section.setdefault(r[4], set()).add(r[5])
        assert r[3].endswith(("@example.org", "@example.com", "@example.net"))
    assert all(len(v) == 1 for v in by_section.values())  # section -> teacher kept, teacher names synthetic
    syn_ids = {normalize_id(r[0]) for name in ("a_crlf.csv", "b_ids.csv", "c_one_row.csv", "c_all_missing.csv")
               for r in read_rows(o1 / name)[1:]}
    real_ids = {normalize_id(r[0]) for name in ("a_crlf.csv", "b_ids.csv", "c_one_row.csv", "c_all_missing.csv")
                for r in read_rows(src / name)[1:]}
    assert not syn_ids & real_ids  # one student_id namespace across the run (D8), never a real ID (D9)
    ids = read_rows(o1 / "b_ids.csv")[1:]
    raw = read_rows(src / "b_ids.csv")[1:]
    # rows are re-sorted on the synthetic IDs (the source is sorted by student_id): compare shapes as multisets
    def shapes(rows: list[list[str]]) -> Counter:
        return Counter((len(r[0]), r[0][:1].isspace(), len(r[0].strip()) > 1 and r[0].strip().startswith("0"),
                        len(r[1]), len(r[2])) for r in rows)

    assert shapes(ids) == shapes(raw)  # each raw form keeps its width, padding and whitespace
    syn_order = [int(r[0]) for r in ids]
    assert syn_order == sorted(syn_order)  # sort order reproduced on the synthetic IDs


def test_native_nulls_from_typed_formats() -> None:
    """Parquet/Excel readers give None for a null cell; pd.factorize turns it into NaN (crashed .strip())."""
    from datanonymous.model import RawColumn, RawTable
    from datanonymous.profile.table import profile_table

    rng = np.random.default_rng(0)
    n = 60
    sid = [f"{1000 + i // 3}" if i % 13 else None for i in range(n)]
    score = sorted((str(int(rng.integers(0, 100))) for i in range(n) if i % 7), key=int)
    score += [None] * (n - len(score))
    school = [None if i % 11 == 0 else str(rng.choice(["577", "530"])) for i in range(n)]
    sname = [{"577": "Lincoln", "530": "Adams"}.get(s) for s in school]
    cols = [RawColumn("student_id", _arr(sid)), RawColumn("score", _arr(score)), RawColumn("school_id", _arr(school)),
            RawColumn("school_name", _arr(sname))]
    tp = profile_table(RawTable(None, [c.name for c in cols], cols, n), "t.parquet", "parquet", Settings(mode="fidelity"))
    assert [fd["text"] for fd in tp.structure.functional_dependencies] == ["school_id -> school_name"]
    assert tp.structure.sort_order["keys"][0] == {"column": "score", "direction": "asc", "missing": "last",
                                                  "order": "numeric"}
    out = _generate(tp)
    assert set(zip(out["school_id"], out["school_name"])) <= {("577", "Lincoln"), ("530", "Adams"), (None, None)}
    assert out["score"][-(n - sum(1 for i in range(n) if i % 7)):] == [None] * (n - sum(1 for i in range(n) if i % 7))
    assert sum(v is None for v in out["student_id"]) == sid.count(None)
    real = {normalize_id(v) for v in sid if v}
    assert not {normalize_id(v) for v in out["student_id"] if v} & real


def _grades(n_students: int = 120, seed: int = 0) -> str:
    """Grades by (course, term): a course can repeat in another term, a term has several courses."""
    rng = np.random.default_rng(seed)
    courses = ["MATH1", "ENG1", "SCI1", "HIST1", "ART1", "PE1"]
    terms = ["T1", "T2", "T3", "T4"]
    rows = []
    for s in range(n_students):
        pairs = {(str(rng.choice(courses)), str(rng.choice(terms))) for _ in range(int(rng.integers(3, 9)))}
        for c, t in sorted(pairs):
            rows.append([f"{50_000 + 17 * s}", c, t, str(rng.choice(["A", "B", "C", "D", "F"]))])
    return _csv(["student_id", "course_code", "term", "letter_grade"], rows)


def test_two_column_composite_key_detected_and_enforced() -> None:
    tp = _profile(_grades())
    assert [k["columns"] for k in tp.structure.composite_keys] == [["student_id", "course_code", "term"]]
    assert any("(student_id, course_code, term) unique" in line for line in st.describe_structure(tp))
    for seed in (1, 2, 3):
        tp = _profile(_grades())
        out = _generate(tp, seed=seed)
        triples = list(zip(out["student_id"], out["course_code"], out["term"]))
        assert len(triples) == len(set(triples)), seed
        real = Counter(r.split(",")[1] for r in _grades().splitlines()[1:])
        assert Counter(out["course_code"]) == real  # exact column totals kept


def test_declared_rule_wins_over_a_functional_dependency_on_its_column() -> None:
    """grade_level -> expected_grad (an FD) and 'start_year <= expected_grad' declared: the rule sets the column."""
    rng = np.random.default_rng(2)
    rows = []
    for i in range(240):
        g = int(rng.integers(9, 13))
        rows.append([1000 + i, g, 2026 + 12 - g - int(rng.integers(0, 3)), 2026 + 12 - g])
    text = _csv(["student_id", "grade_level", "start_year", "expected_grad"], rows)
    s = build_settings({}, {"rules": ["start_year <= expected_grad"]})
    tp = _profile(text, s)
    assert any(fd["dependent"] == "expected_grad" for fd in tp.structure.functional_dependencies)
    st.cross_file_structure([_FR("t.csv", tp)], s)
    fd = [f for f in tp.structure.functional_dependencies if f["dependent"] == "expected_grad"][0]
    assert fd["status"].startswith("not applied")
    for seed in (1, 2):
        out = _generate(tp, s, seed=seed)
        assert all(int(b) >= int(a) for a, b in zip(out["start_year"], out["expected_grad"]))


@hsettings(max_examples=150, deadline=None)
@given(ints=hst.lists(hst.integers(min_value=0, max_value=9_999), min_size=1, max_size=120, unique=True),
       extra_pad=hst.lists(hst.integers(min_value=0, max_value=3), min_size=120, max_size=120),
       padded=hst.lists(hst.booleans(), min_size=120, max_size=120), seed=hst.integers(0, 2**31))
def test_allocation_invariants(ints: list[int], extra_pad: list[int], padded: list[bool], seed: int) -> None:
    """Distinct, never real, no all-zero value unless needed; padded-only keys within their padding, plain keys
    at their own number of digits - unless a visible change was reported."""
    keys = sorted({str(v) for v in ints}, key=int)
    nat = {k: len(k) for k in keys}
    bound = {k: nat[k] + extra_pad[i] for i, k in enumerate(keys) if padded[i]}
    real = np.asarray(sorted(int(k) for k in keys), dtype=np.int64)
    notes: list[str] = []
    try:
        out = allocate_digit_ids(keys, nat, bound, real, np.random.default_rng(seed), notes).tolist()
    except IdSpaceError:
        return
    assert len(set(out)) == len(keys) and not set(out) & set(real.tolist())
    if not notes:
        for k, v in zip(keys, out):
            w = len(str(v))
            assert (w <= bound[k]) if k in bound else (w == nat[k]), (k, v)
    free_nonzero = 9 - sum(1 for v in real.tolist() if 1 <= v <= 9)
    ones = sum(1 for v in out if v < 10)
    if 0 in out:
        assert ones > free_nonzero  # zero only as a last resort


def test_differently_named_column_holding_real_ids_is_treated_as_an_id(tmp_path) -> None:
    """A foreign key under an unlisted name is an ID column (R2-01): synthetic IDs from the parent's mapping, never
    its real values, with a warning saying so."""
    import json

    from conftest import read_dicts, run_cli

    rng = np.random.default_rng(9)
    ids = rng.choice(np.arange(1_000_000, 9_999_999), 80, replace=False).tolist()
    d = tmp_path / "in"
    d.mkdir()
    (d / "students.csv").write_text(_csv(["student_id", "grade"], [[i, rng.choice(["01", "02"])] for i in ids]),
                                    encoding="utf-8")
    rows = [[int(rng.choice(ids)), int(rng.integers(40, 101)), int(rng.integers(1, 81))] for _ in range(300)]
    (d / "visits.csv").write_text(_csv(["visitor", "score", "room"], rows), encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "3")
    assert res.exit_code == 0, res.output
    rep = json.loads((tmp_path / "r" / "files" / "visits.csv.json").read_text(encoding="utf-8"))["tables"][0]
    warned = {c["name"] for c in rep["columns"] for m in c["messages"] if m["code"] == "id_key_ref"}
    assert warned == {"visitor"}  # not the 40-100 scores, not the 1-80 room numbers
    types = {c["name"]: c["type"] for c in rep["columns"]}
    assert types["visitor"] == "IDENTIFIER" and types["score"] != "IDENTIFIER" and types["room"] != "IDENTIFIER"
    syn = {r["visitor"] for r in read_dicts(tmp_path / "o" / "visits.csv")}
    assert not syn & {str(i) for i in ids}
    assert syn <= {r["student_id"] for r in read_dicts(tmp_path / "o" / "students.csv")}  # the join is kept
    docs = [tmp_path / "o" / "DATA_DICTIONARY.md", tmp_path / "o" / "schema.json"]
    for path in [q for q in (tmp_path / "r").rglob("*") if q.is_file()] + docs:
        text = path.read_text(encoding="utf-8")
        assert not any(str(i) in text for i in ids), path.name  # its real IDs (min/max) are not reported
