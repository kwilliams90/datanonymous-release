"""Round 6, cluster ids: the run's real IDs whatever a column's output role, and whose people D35 counts.

* R6-02: a column that would be an ID column but a flag made it something else (``--exclude``, ``--null``,
  ``--force-factor``, ``--force-text``, ``--pii-col``, ``--org-code``, ``--force-type``; or an excluded / nulled
  column an ID flag names too: ``profile.entity.shadow_id``) still holds real IDs of the run. They stay in every
  index of the run's real IDs (``profile.entity.shadow_namespaces``): the D34 header/path screen (a pivot with one
  column per student FAILs), the leak scanner's real-ID index, the D9 allocation (no synthetic ID equals one, digit
  IDs included) and the key discovery (a column holding them is an ID column, R2-01), beside D35's holder count.
* R6-05: beside an ID of the row's own person (``id_role`` ``'subject'``), the IDs of other people (a teacher,
  staff member or guardian, ``'other'``) never lower a count of holders: one teacher of a whole class is one
  teacher, not one student. Without a subject ID every key still counts (the safe side), and so do keys of unknown
  role (record IDs).

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.generate.ids import IdRegistry
from datanonymous.model import FileResult
from datanonymous.privacy import headers as d34
from datanonymous.privacy import leaks
from datanonymous.profile import entity
from test_area_a_support import profile_of, run_table

PIVOT_IDS = ["S47382", "S59731", "S68291"]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "42", *extra)
    return res, out, rep


def blob(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def file_report(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    for t in file_report(rep, name)["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


# =========================================================================== R6-02: D34 (headers and paths)
def _pivot(src: Path, id_header: str = "student_id") -> None:
    write_csv(src / "roster.csv", {id_header: PIVOT_IDS, "Score": ["10", "20", "30"], "Enabled": ["Yes", "No", "Yes"]})
    write_csv(src / "scores.csv", {sid: ["11", "12", "13"] for sid in PIVOT_IDS})


@pytest.mark.parametrize("flags", [
    [], ["--exclude", "student_id"], ["--null", "student_id"], ["--force-text", "student_id"],
    ["--force-factor", "student_id"], ["--pii-col", "student_id=zip"], ["--force-type", "student_id=FACTOR"],
    ["--exclude", "roster.csv::student_id"], ["--org-code", "student_id"]])
def test_pivot_on_flagged_id_columns_fails(tmp_path: Path, flags: list[str]) -> None:
    # R6-02: --exclude / --null (any flag) on the ID column took its IDs out of the D34 index, so a sibling file
    # with one column per student was written with the real IDs as its headers
    src = tmp_path / "in"
    _pivot(src)
    res, out, rep = run(src, tmp_path, *flags)
    assert res.exit_code == 3, res.output
    assert not (out / "scores.csv").exists() and (out / "roster.csv").exists()
    assert file_report(rep, "scores.csv")["status"] == "FAIL"
    assert "a real ID of this run" in " ".join(file_report(rep, "scores.csv")["reasons"])
    if not {"--force-factor", "--org-code"} & set(flags):  # these write the column's real values by choice
        assert not [s for s in PIVOT_IDS if s in blob(out, rep, console=res.output)]


def test_pivot_on_an_excluded_column_named_an_id_by_flag(tmp_path: Path) -> None:
    # --id-col names the column an ID; --exclude drops it: its IDs are still real IDs of the run
    src = tmp_path / "in"
    _pivot(src, "Learner")
    res, out, _ = run(src, tmp_path, "--id-col", "Learner", "--exclude", "Learner")
    assert res.exit_code == 3, res.output
    assert not (out / "scores.csv").exists() and (out / "roster.csv").exists()


@pytest.mark.parametrize("header,flag,role,kept", [("Ref_No", "person_id", "subject", False),
                                                    ("Ref_No", "id_col", "record", False),
                                                    ("Ref_No", "force_type", "record", False),
                                                    ("Ref_No", None, "other", False), ("Cohort", None, None, True)])
def test_an_excluded_column_named_an_id_by_flag_counts_its_people(header: str, flag: str | None, role: str | None,
                                                                  kept: bool) -> None:
    # D35: --person-id Ref_No --exclude Ref_No: the excluded IDs still show ONE person holds the value (an --id-col
    # numbers records: it still bounds the holders). K-02 harden 2 (A2-7): with no flag, an ID word over ID-like keys
    # ('Ref_No') would be an ID column (id-shape), so dropped it still is a shadow ID
    n = 360
    cols = {header: [str(41000 + 7 * (i // 12)) for i in range(n)],
            "Plan": ["Zyxwell Bracewalk Plan" if i < 12 else ("Extended Time", "Small Group")[(i // 12) % 2]
                     for i in range(n)],
            "Minutes": [str(i % 37) for i in range(n)]}
    flags: dict[str, list[str]] = {"exclude": [header]}
    if flag:
        flags[flag] = [f"{header}=IDENTIFIER"] if flag == "force_type" else [header]
    tp, out = run_table(cols, **flags)
    p = profile_of(tp, header)
    assert p.role == "excluded" and entity.shadow_id(p, None) is (role is not None)
    if role:
        assert entity.id_role(p) == role
    # without an ID flag or an ID header nothing says the excluded values are IDs: rows count (12 rows)
    assert ("Zyxwell Bracewalk Plan" in dtext.pseudonymized(profile_of(tp, "Plan"))) is not kept


@pytest.mark.parametrize("flag", ["--exclude", "--null"])
def test_per_student_file_names_with_flagged_id_columns_fail(tmp_path: Path, flag: str) -> None:
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": PIVOT_IDS, "Score": ["10", "20", "30"], "Grade": ["9", "10", "9"]})
    write_csv(src / "iep" / f"{PIVOT_IDS[0]}.csv", {"Goal": ["Reading", "Math"], "Minutes": ["30", "45"]})
    res, out, rep = run(src, tmp_path, flag, "student_id")
    assert res.exit_code == 3, res.output
    assert (out / "roster.csv").exists() and not (out / "iep" / f"{PIVOT_IDS[0]}.csv").exists()
    assert PIVOT_IDS[0] not in blob(out, rep, console=res.output)


def test_inspect_screens_flagged_id_columns(tmp_path: Path) -> None:
    src = tmp_path / "in"
    _pivot(src)
    res = run_cli("inspect", src, "--null", "student_id")
    assert res.exit_code == 3, res.output
    assert not [s for s in PIVOT_IDS if s in res.output]


def test_run_id_keys_and_local_keys_hold_shadow_ids() -> None:
    tp, _ = run_table({"student_id": PIVOT_IDS * 4, "Score": [str(i) for i in range(12)]}, exclude=["student_id"])
    p = profile_of(tp, "student_id")
    assert p.role == "excluded" and entity.shadow_id(p, None)
    assert set(tp.private.shadow_id_raw) == {p.index} and not tp.entity.id_columns
    fr = FileResult("f.csv", "f.csv", "csv", tables=[tp])
    keys = d34.run_id_keys([fr])
    assert keys and all(keys.match(s) for s in PIVOT_IDS)
    assert d34._table_id_values(tp) == set(PIVOT_IDS)


# =========================================================================== R6-02: the leak scanner
@pytest.mark.parametrize("flag", ["exclude", "null", "force_text", "force_factor"])
def test_leak_index_holds_shadow_ids(flag: str) -> None:
    tp, _ = run_table({"student_id": PIVOT_IDS * 4, "Score": [str(i) for i in range(12)]}, **{flag: ["student_id"]})
    idx = leaks.SensitiveIndex.build([FileResult("f.csv", "f.csv", "csv", tables=[tp])])
    n, origins = idx.real_id_hits({"S47382", "s-59731", "S10000"})
    assert n == 2 and {idx.origins[o].column for o in origins} == {"student_id"}
    # no output column is checked against them as its own (a forced column writes what its flag asked for)
    assert not idx.column_namespace
    scanner = leaks.Scanner(idx, leaks.Allow(tokens=set(), phrases=set()))
    assert scanner.hits_in_text("see S68291 for details")


def test_row_number_shadow_columns_are_no_real_ids() -> None:
    # D9: 1..n row numbers are kept as they are by an ID column, so an excluded one is no index entry either
    tp, _ = run_table({"id": [str(i) for i in range(1, 31)], "Score": [str(i % 7) for i in range(30)]},
                      exclude=["id"])
    assert tp.private.shadow_id_raw
    assert entity.shadow_namespaces([FileResult("f.csv", "f.csv", "csv", tables=[tp])]) == []


# =========================================================================== R6-02: D9 (never a real ID)
def test_unregistered_real_digit_ids_are_avoided() -> None:
    # avoid_ids (FAILed files, shadow IDs) used to bind patterned and e-mail IDs only: digit IDs could take one
    for seed in range(8):
        reg = IdRegistry()
        reg.register("staffid", "f.csv", None, 0, "staff_id", np.asarray(["3", "5"], dtype=object))
        reg.build(seed, avoid_ids=["0", "1", "2", "4", "6", "7"])
        assert set(reg.namespaces["staffid"].mapping.values()) == {8, 9}, seed


def test_synthetic_ids_never_equal_an_excluded_columns_real_ids(tmp_path: Path) -> None:
    src = tmp_path / "in"
    excluded = [str(v) for v in range(10, 96)]  # 86 of the 90 two-digit numbers
    write_csv(src / "a.csv", {"student_id": excluded, "Score": [str(i % 9) for i in range(len(excluded))],
                              "Grade": [str(9 + i % 4) for i in range(len(excluded))]})
    staff = ["96", "97", "98", "99", "150"] * 3  # unpadded integers: they may take 3 digits when 2 run out (D9)
    write_csv(src / "b.csv", {"staff_id": staff, "Minutes": [str(5 * i) for i in range(len(staff))]})
    res, out, rep = run(src, tmp_path, "--exclude", "a.csv::student_id")
    assert res.exit_code == 0, res.output
    syn = {r["staff_id"] for r in read_dicts(out / "b.csv")}
    assert len(syn) == 5 and not syn & set(excluded) and not syn & set(staff)
    assert file_report(rep, "b.csv")["checks"]["leak_scan"]["result"] == "PASS"


# =========================================================================== R6-02: key discovery (R2-01)
def _buddies(src: Path) -> list[str]:
    ids = [str(480000 + 37 * i) for i in range(40)]
    write_csv(src / "roster.csv", {"student_id": ids, "Buddy": [ids[(i * 7 + 3) % 40] for i in range(40)],
                                   "Score": [str(50 + i) for i in range(40)]})
    return ids


@pytest.mark.parametrize("flags", [[], ["--null", "student_id"], ["--exclude", "student_id"],
                                   ["--force-text", "student_id"]])
def test_a_column_holding_flagged_ids_is_an_id_column(tmp_path: Path, flags: list[str]) -> None:
    # with --null / --exclude student_id, Buddy (the real IDs of student_id) was written as an integer column
    src = tmp_path / "in"
    ids = _buddies(src)
    res, out, rep = run(src, tmp_path, *flags)
    assert res.exit_code == 0, res.output
    assert not {r["Buddy"] for r in read_dicts(out / "roster.csv")} & set(ids)
    buddy = column_report(rep, "roster.csv", "Buddy")
    assert buddy["type"] == "IDENTIFIER"
    text = " ".join(m["text"] for m in buddy["messages"] if m["code"] == "id_key_ref")
    assert ("no mapping to share" in text) is bool(flags) and ("joins between them are kept" in text) is not flags
    assert not [s for s in ids if s in blob(out, rep, console=res.output)]


def test_flagged_columns_are_judged_by_their_flags(tmp_path: Path) -> None:
    # two columns the user flagged are never 'ID columns' of each other: no warning, no new ID column
    src = tmp_path / "in"
    ids = [f"S{1000 + i}" for i in range(30)]
    write_csv(src / "a.csv", {"student_id": ids, "Score": [str(i) for i in range(30)]})
    write_csv(src / "b.csv", {"student_id": ids[::-1], "Minutes": [str(2 * i) for i in range(30)]})
    res, out, rep = run(src, tmp_path, "--force-text", "student_id")
    assert res.exit_code == 0, res.output
    for name in ("a.csv", "b.csv"):
        col = column_report(rep, name, "student_id")
        assert col["type"] != "IDENTIFIER"
        assert not any(m.get("code") == "unlinked_key_column" for m in col.get("messages", [])), name


# =========================================================================== R6-05: whose IDs count
def _activity(other: str | None, first: bool = False) -> dict[str, list[str]]:
    """30 students (one row each); 15 hold each activity; ``other`` (a teacher/staff/guardian ID) is one value."""
    cols: dict[str, list[str]] = {}
    if other and first:
        cols[other] = ["T1000"] * 30
    cols["student_id"] = [f"S{1000 + i}" for i in range(30)]
    cols["Activity"] = ["Chess Guild"] * 15 + ["Robotics Guild"] * 15
    if other and not first:
        cols[other] = ["T1000"] * 30
    return cols


@pytest.mark.parametrize("other,first", [("teacher_id", False), ("teacher_id", True), ("StaffID", False),
                                         ("Guardian_ID", False), ("EmpID", True), (None, False)])
def test_other_peoples_ids_never_lower_the_subjects_count(other: str | None, first: bool) -> None:
    tp, cols = run_table(_activity(other, first))
    p = profile_of(tp, "Activity")
    assert dtext.pseudonymized(p) == {}, (other, first)
    assert Counter(cols["Activity"].tolist()) == {"Chess Guild": 15, "Robotics Guild": 15}


@pytest.mark.parametrize("flags", [{"exclude": ["teacher_id"]}, {"null": ["teacher_id"]},
                                   {"force_text": ["teacher_id"]}])
def test_flagged_other_peoples_ids_never_lower_the_subjects_count(flags: dict[str, list[str]]) -> None:
    tp, _ = run_table(_activity("teacher_id"), **flags)
    assert dtext.pseudonymized(profile_of(tp, "Activity")) == {}, flags


def test_cli_teacher_shared_by_a_class(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "activity.csv", _activity("teacher_id"))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert Counter(r["Activity"] for r in read_dicts(out / "activity.csv")) == {"Chess Guild": 15,
                                                                                  "Robotics Guild": 15}
    assert "pseudonymized_levels" not in column_report(rep, "activity.csv", "Activity")["params"]


def test_without_a_subject_id_every_key_still_counts() -> None:
    # no ID of the row's own person: nothing shows who the rows' people are, so the teacher ID still bounds them
    cols = _activity("teacher_id")
    cols["dcid"] = [str(70000 + i) for i in range(30)]
    del cols["student_id"]
    tp, _ = run_table(cols)
    assert set(dtext.pseudonymized(profile_of(tp, "Activity"))) == {"Chess Guild", "Robotics Guild"}


def test_keys_of_unknown_role_still_count_beside_a_subject_id() -> None:
    # a record key (--id-col on a header that names no person) may number groups: it keeps the safe minimum
    cols = _activity(None)
    cols["Section_Key"] = ["K1"] * 30
    tp, _ = run_table(cols, id_col=["Section_Key"])
    assert entity.id_role(profile_of(tp, "Section_Key")) == "record"
    assert set(dtext.pseudonymized(profile_of(tp, "Activity"))) == {"Chess Guild", "Robotics Guild"}


def test_one_student_under_many_teachers_is_still_one_holder() -> None:
    # the round-3 protection stays: the subject ID decides, and it shows ONE student
    n = 360
    cols = {"teacher_id": [f"T{100 + i % 12}" for i in range(n)],
            "student_id": [f"S{5000 + i // 12}" for i in range(n)],
            "Plan": ["Zyxwell Bracewalk Plan" if i < 12 else ("Extended Time", "Small Group")[(i // 12) % 2]
                     for i in range(n)]}
    tp, out = run_table(cols)
    assert "Zyxwell Bracewalk Plan" in dtext.pseudonymized(profile_of(tp, "Plan"))
    assert "Zyxwell Bracewalk Plan" not in set(out["Plan"].tolist())


def _keys(columns: dict[str, list[str]]) -> entity.HolderKeys:
    from conftest import build_fidelity_settings as build_settings
    from datanonymous.model import RawColumn, RawTable
    from datanonymous.profile.table import profile_table

    n = len(next(iter(columns.values())))
    raw = RawTable(None, list(columns), [RawColumn(k, np.asarray(v, dtype=object)) for k, v in columns.items()], n)
    tp = profile_table(raw, "f.csv", "csv", build_settings({}))
    return entity.d35_keys(raw, tp.columns, tp.entity)


def test_teacher_ids_are_dropped_only_beside_a_subject_id() -> None:
    hk = _keys(_activity("teacher_id"))
    assert hk.people and len(hk.keys) == 1  # student_id only
    assert [int(np.unique(k).size) for k in hk.keys] == [30]
    cols = _activity("teacher_id")
    cols["dcid"] = [str(70000 + i) for i in range(30)]
    del cols["student_id"]
    hk = _keys(cols)
    assert not hk.people and len(hk.keys) == 2  # teacher_id and dcid: no subject, every key counts
