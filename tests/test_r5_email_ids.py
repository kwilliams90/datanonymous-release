"""Round-5 review fix, email-ids cluster (R5-02): e-mail addresses and usernames used as IDs.

* E-mail addresses under an ID header (``student_id``, ``--id-col Email``, ``--person-id student_email``)
  get e-mail addresses built like an EMAIL column's (D24): name-like local parts that are no real local part,
  real username or real ID of the run, on reserved domains (example.org/.com/.net, never a real domain),
  1:1 and consistent across files and case forms (D8), never real (D9).
* Patterned IDs keep a common letter prefix only when every ID shares it; two or more letters only when at
  least ``--min-cell`` IDs share them and they are neither name-like nor a real ID (a username such as
  ``jsmith`` is never kept), else the first letter.
* The leak scan checks ID columns with the e-mail and username rules too (same column, run-wide namespaces,
  report/docs), re-checking the prefix the generator kept.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as hst

from conftest import read_dicts, run_cli
from datanonymous.detect.ids import (
    checked_prefix,
    common_letter_prefix,
    is_email_id,
    kept_prefix,
    name_like,
    normalize_id,
    username_stem,
)
from datanonymous.generate import ids as ids_mod
from datanonymous.generate.ids import IdRegistry, build_mapping
from datanonymous.privacy import leaks
from datanonymous.privacy.leaks import Scanner, SensitiveIndex, build_allow

RESERVED = ("example.org", "example.com", "example.net")


def _arr(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _file_report(rep: Path, rel: str) -> dict:
    return json.loads((rep / "files" / (rel + ".json")).read_text(encoding="utf-8"))


def _render(values: list[str], seed: int = 1, key: str = "studentid", **build) -> list[str]:
    reg = IdRegistry()
    raw = _arr(values)
    reg.register(key, "f.csv", None, 0, "student_id", raw)
    reg.build(seed, **build)
    assert reg.namespaces[key].error is None
    return reg.render(key, raw).tolist()


def _reserved(domain: str) -> bool:
    d = domain.casefold()
    return any(d == r or d.endswith("." + r) for r in RESERVED)


def _locals(values) -> set[str]:
    return {v.strip().rpartition("@")[0].casefold() for v in values if "@" in v}


def _stems(values, prefix: str = "") -> set[str]:
    return {s for s in (username_stem(v, prefix) for v in values) if s}


PEOPLE = ["ana.lopez", "jsmith23", "mary.jones", "bob_k", "li.wei", "omar.f", "zoe.k", "sam.p", "kim.t", "lee.r",
          "pat.q", "max.w"]


# --------------------------------------------------------------------------- shared rules (detect.ids)
@pytest.mark.parametrize(("value", "expected"), [
    ("jsmith@district.edu", True), ("Ana.Lopez@District.EDU", True), (" s104019@k12.ca.us ", True),
    ("S12@2021", False), ("jsmith@district", False), ("a@b", False), ("S104019", False), ("j smith@x.org", False),
])
def test_is_email_id(value: str, expected: bool) -> None:
    assert is_email_id(value) is expected


@pytest.mark.parametrize(("value", "prefix", "stem"), [
    ("jsmith23", "", "jsmith"), ("ana.lopez", "", "analopez"), ("Ana.Lopez@District.EDU", "", "analopez"),
    ("jsmith23", "j", "smith"), ("S104019", "s", None), ("STUDENT0001", "student", None), ("104019", "", None),
    ("7c42a60b-6245-4094-baa0-4d30f538e710", "", None), ("ab12", "", None), ("s104019@k12.ca.us", "s", None),
])
def test_username_stem(value: str, prefix: str, stem: str | None) -> None:
    assert username_stem(value, prefix) == stem


def test_name_like_reads_names_and_usernames_not_codes() -> None:
    for w in ("jsmith", "johnsmith", "analopez", "mjones", "maria", "smith"):
        assert name_like(w), w
    for w in ("s", "st", "stu", "emp", "staff", "student", "person", "case", "test", "teacher", "acct", "tkt"):
        assert not name_like(w), w


@pytest.mark.parametrize(("reps", "banned", "expected"), [
    ([f"S{i:06d}" for i in range(3)], set(), "s"),  # one letter: always kept (S104019)
    ([f"STU{i:04d}" for i in range(30)], set(), "stu"),  # a scheme code shared by >= --min-cell IDs
    ([f"STU{i:04d}" for i in range(5)], set(), "s"),  # a few IDs: only the first letter
    ([f"msmith{i:02d}" for i in range(12)], set(), "m"),  # name-like: never kept whole
    (["jsmith", "jsmith2", "jsmithers"], set(), "j"),
    (["johnsmith"], set(), "j"),  # one person's whole username
    ([f"AB{i}" for i in range(20)], {"ab"}, "a"),  # the prefix is itself a real ID
    ([f"A{i}" for i in range(20)], {"a"}, ""),
    (["7c42a60b-6245-4094-baa0-4d30f538e710", "8d42a60b-6245-4094-baa0-4d30f538e710"], set(), ""),
])
def test_kept_prefix(reps: list[str], banned: set[str], expected: str) -> None:
    assert kept_prefix(reps, banned) == expected


def test_checked_prefix() -> None:
    assert checked_prefix("stu", 30) == "stu" and checked_prefix("stu", 3) == "s" and checked_prefix("stu") == "stu"
    assert checked_prefix("jsmith") == "j" and checked_prefix("s", 1) == "s" and checked_prefix("") == ""


# --------------------------------------------------------------------------- generator: e-mail addresses as IDs
def test_single_person_email_id_codex_probe() -> None:
    """Codex R5-02: 20 visits of one student whose student_id is an e-mail address."""
    real = ["johnsmith@district.edu"] * 20
    out = _render(real, seed=42)
    assert len(set(out)) == 1 and len(out) == 20  # 1:1, the ID block keeps its rows
    local, _, domain = out[0].rpartition("@")
    assert _reserved(domain) and domain.casefold() != "district.edu"
    assert local.casefold() != "johnsmith" and "johnsmith" not in out[0].casefold()
    assert username_stem(out[0]) != "johnsmith"


def test_email_ids_sharing_a_prefix_never_reuse_a_real_local_part() -> None:
    """Codex R5-02: student0..student9@district.edu kept 'student' + a digit, recreating real local parts."""
    real = [f"student{i % 10}@district.edu" for i in range(100)]
    out = _render(real, seed=42)
    assert len(set(out)) == 10
    assert Counter(Counter(out).values()) == Counter(Counter(real).values())
    assert not _locals(out) & _locals(real)
    assert not _stems(out) & _stems(real)  # 'student' itself is a real username stem
    assert all(_reserved(v.rpartition("@")[2]) for v in out)
    assert not {normalize_id(v) for v in out} & {normalize_id(v) for v in real}


def test_email_id_shapes_and_mapping() -> None:
    real = [f"{p}@school.org" for p in PEOPLE] * 3
    out = _render(real, seed=5)
    m = dict(zip(real, out))
    assert len(set(m.values())) == len(PEOPLE)  # 1:1
    for r, s in m.items():
        rl, sl = r.rpartition("@")[0], s.rpartition("@")[0]
        assert [ch for ch in rl if not ch.isalnum()] == [ch for ch in sl if not ch.isalnum()]  # separators kept
        assert s.rpartition("@")[2] == "example.org"
    assert not _locals(out) & _locals(real) and not _stems(out) & _stems(real)


def test_email_id_domains_ranked_by_rows_and_never_real() -> None:
    real = (["a1@district.edu", "b2@district.edu", "c3@district.edu"] * 10 + ["d4@example.org"] * 6
            + ["e5@k12.ca.us"] * 2)
    out = _render(real, seed=3)
    dom = {r.rpartition("@")[2]: s.rpartition("@")[2] for r, s in zip(real, out)}
    assert len(set(dom.values())) == 3  # the number of distinct domains is kept
    # the most rows first: example.org, but example.org is a real domain here, so never reused for another one
    assert dom["district.edu"] == "example.com" and dom["example.org"] == "example.net"
    assert dom["k12.ca.us"] not in {"district.edu", "example.org", "k12.ca.us"} and _reserved(dom["k12.ca.us"])


def test_email_ids_consistent_across_files_and_case_forms() -> None:
    reg = IdRegistry()
    a = _arr(["JSmith@District.EDU", "ana.lopez@district.edu", " JSmith@District.EDU "])
    b = _arr(["jsmith@district.edu", "ANA.LOPEZ@DISTRICT.EDU"])
    reg.register("studentid", "a.csv", None, 0, "student_id", a)
    reg.register("studentid", "b.csv", None, 0, "student_id", b)
    reg.build(9)
    ra, rb = reg.render("studentid", a).tolist(), reg.render("studentid", b).tolist()
    assert normalize_id(ra[0]) == normalize_id(rb[0]) and normalize_id(ra[1]) == normalize_id(rb[1])  # D8
    assert ra[2] == " " + ra[0] + " "  # surrounding whitespace kept
    assert rb[1] == rb[1].upper() and rb[1].endswith("@EXAMPLE.ORG")  # an upper-case form stays upper-case
    assert ra[1] == ra[1].lower() and rb[0] == rb[0].lower()
    assert ra[0].rpartition("@")[2] == "example.org" and ra[0][:2].isupper()  # 'JS' case pattern of the local part


def test_mixed_namespace_keeps_each_kind() -> None:
    real = ["S1001", "S1002", "104019", "104020", "jdoe@district.edu", "mary.k@district.edu"] * 2
    out = _render(real, seed=4)
    m = dict(zip(real, out))
    assert m["S1001"][0] == "S" and len(m["S1001"]) == 5 and m["104019"].isdigit()
    assert is_email_id(m["jdoe@district.edu"]) and _reserved(m["jdoe@district.edu"].rpartition("@")[2])
    assert not {normalize_id(v) for v in out} & {normalize_id(v) for v in real}
    assert len(set(out)) == len(set(real))


def test_email_id_mapping_is_seeded() -> None:
    real = [f"{p}@school.org" for p in PEOPLE]
    assert _render(real, seed=11) == _render(real, seed=11)
    assert _render(real, seed=11) != _render(real, seed=12)


def test_email_id_avoids_run_usernames_ids_and_email_columns(monkeypatch) -> None:
    """A name-like local part that is a real username/ID of another namespace, a real local part of an EMAIL
    column of the run, a real address or a real ID of a file that FAILed before planning is never taken (it
    falls back to a same-shape mask)."""
    blocked = ["jsmith23", "mary.jones", "carol.white", "dora.flint"]
    calls: list[str] = []

    def fake(local: str, attempt: int, pyr) -> str:  # noqa: ANN001
        v = blocked[len(calls) % len(blocked)]
        calls.append(v)
        return v

    monkeypatch.setattr(ids_mod, "_fake_local", fake)
    reg = IdRegistry()
    emails = _arr(["abcdefgh@district.edu", "zyxwvuts@district.edu"])
    reg.register("studentid", "a.csv", None, 0, "student_id", emails)
    reg.register("login", "b.csv", None, 0, "login", _arr(["jsmith23", "mary.jones"]))
    reg.build(2, avoid_emails=["carol.white@example.org"], avoid_ids=["dora.flint@district.edu"])  # a FAILed file
    out = reg.render("studentid", emails).tolist()
    assert calls  # the name-like candidates were tried
    for v in out:
        loc = v.rpartition("@")[0]
        assert loc not in blocked and username_stem(v) not in {"jsmith", "maryjones", "carolwhite", "doraflint"}
        assert _reserved(v.rpartition("@")[2])


# --------------------------------------------------------------------------- generator: usernames as IDs
def test_username_ids_never_keep_a_real_username() -> None:
    for real in (["jsmith23"] * 8, ["johnsmith"] * 5, ["jsmith", "jsmith2", "jsmithers"] * 3, PEOPLE * 2):
        out = _render(real, seed=8)  # 'johnsmith' alone used to FAIL (the whole value was the kept prefix)
        assert len(set(out)) == len(set(real))
        assert not {normalize_id(v) for v in out} & {normalize_id(v) for v in real}
        assert not any(v.casefold().startswith(("jsmith", "johnsmith")) for v in out)
        prefix = kept_prefix(sorted(set(real)), set())
        assert not _stems(out, prefix) & _stems(real, prefix)
        assert [len(v) for v in out] == [len(v) for v in real]  # same shape


_WORD = hst.text("abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=8)
_USER = hst.builds(lambda a, sep, b, d: a + sep + b + d, _WORD, hst.sampled_from(["", ".", "_"]),
                   hst.text("abcdefghijklmnopqrstuvwxyz", max_size=6), hst.text("0123456789", max_size=3))


@hsettings(max_examples=40, deadline=None)
@given(users=hst.lists(_USER, min_size=1, max_size=25, unique=True), as_email=hst.booleans(),
       domain=hst.sampled_from(["district.edu", "example.org", "K12.CA.US"]), seed=hst.integers(0, 2**16))
def test_property_email_and_username_ids(users: list[str], as_email: bool, domain: str, seed: int) -> None:
    """Any usernames or addresses: 1:1 with the row counts, never a real ID, local part or username (with the
    kept prefix), reserved domains for addresses, the source shape for usernames."""
    real = [(u + "@" + domain) if as_email else u for u in users for _ in range(1 + len(u) % 3)]
    out = _render(real, seed=seed)
    m: dict[str, str] = {}
    for r, s in zip(real, out):
        assert m.setdefault(normalize_id(r), normalize_id(s)) == normalize_id(s)  # D8: one synthetic per real ID
    assert len(set(m.values())) == len(m)  # 1:1
    assert not set(m.values()) & set(m)  # never a real ID
    if as_email:
        assert all(_reserved(s.rpartition("@")[2]) for s in out)
        assert not _locals(out) & _locals(real) and not _stems(out) & _stems(real)
    else:
        prefix = kept_prefix(list(dict.fromkeys(real)), set(m))
        assert not _stems(out, prefix) & _stems(real, prefix)
        assert [len(s) for s in out] == [len(r) for r in real]


def test_prefixed_codes_keep_their_scheme_prefix() -> None:
    real = [f"STU{i:04d}" for i in range(1, 41)]
    out = _render(real, seed=6)
    assert all(v.startswith("STU") and v[3:].isdigit() for v in out) and not set(out) & set(real)
    out = _render([f"S{i:06d}" for i in range(1, 300, 7)], seed=6)
    assert all(v[0] == "S" and v[1:].isdigit() for v in out)
    few = _render([f"STU{i:04d}" for i in range(1, 6)], seed=6)
    assert all(v[0] == "S" and v[1:3].isalpha() and v[1:3].isupper() for v in few)


def test_build_mapping_direct_call_applies_the_rules() -> None:
    reg = IdRegistry()
    reg.register("k", "f.csv", None, 0, "student_id", _arr(["johnsmith@district.edu", "jsmith23"]))
    ns = reg.namespaces["k"]
    build_mapping(ns, np.random.default_rng(0))
    assert ns.mapping[normalize_id("johnsmith@district.edu")].endswith("@example.org")
    assert ns.prefix == "j" and not ns.mapping[normalize_id("jsmith23")].casefold().startswith("jsmith")


# --------------------------------------------------------------------------- leak scan: e-mail and username rules
def test_same_column_rules_for_email_and_username_ids() -> None:
    real = leaks._RealSets.of(leaks.ID, ["johnsmith@district.edu", "jsmith23", "S104019"])
    hits = leaks._same_column_hits
    assert hits(leaks.ID, real, ["johnsmith@tycljcrh.gfd"]) >= 1  # the Codex output: a real local part
    assert hits(leaks.ID, real, ["jsmith57"]) == 1  # a real username
    assert hits(leaks.ID, real, ["priscilla@example.org", "kqwert57", "S583920"]) == 0
    with_prefix = leaks._RealSets.of(leaks.ID, ["STUDENT0001", "STUDENT0002"], "student")
    assert hits(leaks.ID, with_prefix, ["STUDENT4821"]) == 0  # a kept scheme prefix is no username


def test_namespace_users_rechecks_the_kept_prefix() -> None:
    codes = {f"STUDENT{i:04d}" for i in range(30)}
    assert leaks.namespace_users(codes, {"student"}) == ("student", set())
    # a "prefix" one person's username is: cut to its first letter, so the username is still read
    assert leaks.namespace_users({"zqxvbn23"}, {"zqxvbn"}) == ("z", {"qxvbn"})
    assert leaks.namespace_users({"jsmith23", "jsmith24"}, {"jsmith"})[0] == "j"


def test_real_id_hits_email_and_username_rules_run_wide() -> None:
    idx = SensitiveIndex()
    idx.namespaces["a"] = leaks._Namespace(exact={"johnsmith@district.edu"}, normalized={"johnsmithdistrictedu"},
                                           origins={0}, locals={"johnsmith"}, stems={"johnsmith"})
    idx.namespaces["b"] = leaks._Namespace(exact={"kwong23"}, normalized={"kwong23"}, origins={1}, stems={"kwong"})
    assert idx.real_id_hits({"johnsmith@example.org"}) == (1, {0})
    assert idx.real_id_hits({"kwong77"}) == (1, {1})
    assert idx.real_id_hits({"priscilla@example.org", "qzwx77"}) == (0, set())


def _old_email_mapping(ns, keys, real_keys, used, rng, foreign, users) -> None:  # noqa: ANN001
    """The pre-R5 behaviour: the real local part kept, the domain masked."""
    for k in keys:
        local = ns.representatives[k].rpartition("@")[0]
        ns.mapping[k] = local + "@" + "".join(chr(97 + int(rng.integers(26))) for _ in range(8)) + ".gfd"


def test_leak_scan_fails_an_email_id_with_a_real_local_part(tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "in"
    _write(d / "visits.csv", ["student_id", "visit_date"],
           [["johnsmith@district.edu", f"2024-03-{i + 1:02}"] for i in range(20)])
    monkeypatch.setattr(ids_mod, "_email_mapping", _old_email_mapping)
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "42")
    assert res.exit_code == 3, res.output
    assert not (tmp_path / "o" / "visits.csv").exists()
    assert _file_report(tmp_path / "r", "visits.csv")["status"] == "FAIL"
    for f in (tmp_path / "r").rglob("*"):
        if f.is_file():
            assert "johnsmith" not in f.read_text(encoding="utf-8", errors="replace").casefold()
    assert "johnsmith" not in res.output.casefold()


@pytest.mark.parametrize("username", ["jsmith23", "zqxvbn23"])
def test_leak_scan_fails_a_kept_username_prefix(tmp_path: Path, monkeypatch, username: str) -> None:
    """With the prefix rule undone (the whole common prefix kept), the scan FAILs the file: a name-like prefix
    before writing, a prefix of fewer than --min-cell IDs after writing (namespace-wide re-check)."""
    d = tmp_path / "in"
    _write(d / "logins.csv", ["student_id", "minutes"], [[username, 30 + i] for i in range(12)])
    monkeypatch.setattr(ids_mod, "kept_prefix",
                        lambda reps, banned=frozenset(), min_ids=10: common_letter_prefix(reps))
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 3, res.output
    assert not (tmp_path / "o" / "logins.csv").exists()
    report = (tmp_path / "r" / "REPORT.md").read_text(encoding="utf-8")
    assert username not in report and username[:-2] not in report


def test_report_scan_indexes_email_id_local_parts(tmp_path: Path) -> None:
    from datanonymous import pipeline
    from datanonymous.config import Settings

    d = tmp_path / "in"
    _write(d / "visits.csv", ["student_id", "score"], [[f"{p}@district.edu", i] for i, p in enumerate(PEOPLE)])
    run = pipeline.execute(Settings(mode="fidelity", seed=1, force=True), d, tmp_path / "o", tmp_path / "r")
    assert pipeline.exit_code(run) == 0
    scanner = Scanner(SensitiveIndex.build(run.files), build_allow(run))
    rules = {h.rule for h in scanner.hits_in_text("contact mary.jones about it")}
    assert "email local part" in rules
    assert {h.rule for h in scanner.hits_in_text("write to MARY.JONES@DISTRICT.EDU")} & {"email", "id"}


# --------------------------------------------------------------------------- end to end (Codex R5-02 shapes)
@pytest.mark.parametrize(("header", "values", "flags"), [
    (["student_id", "visit_date"], [["johnsmith@district.edu", f"2024-03-{i + 1:02}"] for i in range(20)], ()),
    (["student_id", "score"], [[f"student{i % 10}@district.edu", i % 51] for i in range(100)], ()),
    (["student_id", "Email"], [[f"S{i + 1000}", f"student10{i % 3 + 1}@district.edu"] for i in range(60)],
     ("--id-col", "Email")),
    (["student_email", "score"], [[f"{PEOPLE[i % 12]}@school.org", i] for i in range(48)],
     ("--person-id", "student_email")),
    (["login", "score"], [[PEOPLE[i % 12], i] for i in range(48)], ("--id-col", "login")),
])
def test_cli_email_and_username_ids(tmp_path: Path, header: list[str], values: list[list[object]],
                                    flags: tuple[str, ...]) -> None:
    d = tmp_path / "in"
    _write(d / "source.csv", header, values)
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "42", *flags)
    assert res.exit_code == 0, res.output
    rows = read_dicts(tmp_path / "o" / "source.csv")
    assert len(rows) == len(values)  # N in = N out
    col = flags[1] if flags else header[0]
    j = header.index(col)
    real = [str(r[j]) for r in values]
    syn = [r[col] for r in rows]
    assert sorted(Counter(syn).values()) == sorted(Counter(real).values())  # 1:1 with the row counts
    assert not {normalize_id(v) for v in syn} & {normalize_id(v) for v in real}
    assert not _locals(syn) & _locals(real)
    assert not _stems(syn, "") & _stems(real, "")
    if is_email_id(real[0]):
        assert all(_reserved(v.rpartition("@")[2]) for v in syn)
    assert _file_report(tmp_path / "r", "source.csv")["checks"]["leak_scan"]["result"] == "PASS"


@pytest.mark.parametrize("fmt", ["dta", "parquet", "xlsx"])
def test_email_ids_in_typed_formats(tmp_path: Path, fmt: str) -> None:
    """Synthetic addresses may be longer than the real ones (reserved domains): typed writers take them."""
    d = tmp_path / "in"
    d.mkdir()
    df = pd.DataFrame({"student_id": [f"{p}@d.io" for p in PEOPLE] * 3, "score": list(range(36))})
    path = d / f"s.{fmt}"
    if fmt == "dta":
        df.to_stata(path, write_index=False, version=118)
    elif fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_excel(path, index=False)
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "3")
    assert res.exit_code == 0, res.output
    out = tmp_path / "o" / f"s.{fmt}"
    o = pd.read_stata(out) if fmt == "dta" else pd.read_parquet(out) if fmt == "parquet" else pd.read_excel(out)
    syn = o["student_id"].astype(str).tolist()
    assert len(set(syn)) == len(PEOPLE) and all(v.endswith("@example.org") for v in syn)
    assert not _locals(syn) & _locals(df["student_id"])
