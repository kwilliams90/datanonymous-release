"""Known issues K-04, K-05, K-06, K-08, K-11 (cluster "holders"): who holds a value (D35).

Fail-safe principle: when the evidence about who the people are is weak or conflicting, the SMALLER holder count
wins (fewer holders -> pseudonymise).

* K-04: the report advises ``--id-col`` for person or record IDs and ``--person-id`` only for the row's own person;
  an ``--person-id`` claim (an incident number) never outvotes a table whose header proves its people.
* K-05: the same for an ``--person-id`` on a staff badge column.
* K-06: a bare ``SID`` proves people only when a name column of the row's own person agrees with it or its keys are
  the run's proven student IDs; otherwise it numbers records (sessions, visits), with a ``d35_rows_counted`` warning.
* K-08: the ZIP small-cell policy counts holders as D35 does, not rows.
* K-11: a multi-select written as combinations never rebuilds a rare whole value (``Okafor, Tuan``) from two items
  each held by many; the report never calls its real values invented.

Hardening (attack on the fix):

* H-1: a bare SID that nothing corroborates numbers records in its table, but over the run its counts rank as a
  claim (the fewest wins): a staff, incident or row-numbered sibling that repeats a value never outvotes the one SID
  that holds it (no flag, and with the report's own ``--id-col`` advice).
* H-6: the ``d35_rows_counted`` warning of such a table says what is unknown (whether SID is the student's ID) and
  how to confirm it (``--person-id SID``); a dense student-ID range still proves nothing by values alone (a session
  numbering inside it would match as well).
* H-7: a rare combination writes real items together only when, in its order, they are the real items of a whole
  value D35 keeps as it is (that text is written anyway), else its most widely held item; every other item becomes
  that item's ONE pseudonym for the run, so item prevalence holds under the pseudonyms. Two common halves of a name
  never stand together unless that whole name is itself common.

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.generate import pseudonyms
from datanonymous.model import ColumnType as T
from datanonymous.model import SyntheticColumn, SyntheticTable
from datanonymous.privacy import leaks
from datanonymous.profile import entity

from test_area_a_support import detect, profile_of, run_table

CANARY = "Kiwi anaphylaxis Zorbek"
COMMON = ("Extended time", "Preferential seating", "Read aloud", "Small group testing")


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "7"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def found(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> list[str]:
    blob = "\n".join(p.read_bytes().decode("utf-8", errors="replace")
                     for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file())
    return [w for w in words if w in blob + "\n" + console]


def file_report(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def warning_codes(rep: Path, name: str) -> list[str]:
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    return [w["code"] for w in data["warnings"] if w.get("file") == name]


def students(n: int = 60) -> dict[str, list[str]]:
    """One row per student (``Student_ID``, a header that proves the row's own person); the first student alone
    holds CANARY, the others one of COMMON (15 each)."""
    return {"Student_ID": [str(100100 + 7 * i) for i in range(n)],
            "Accommodation": [CANARY if i == 0 else COMMON[i % 4] for i in range(n)]}


def log(key: str, keys: list[str], canary_rows: int = 12, other_rows: int = 200) -> dict[str, list[str]]:
    """A log keyed by ``key`` (one value per row of ``keys``, cycled); the first ``canary_rows`` rows (one student's)
    hold CANARY, the rest COMMON."""
    n = canary_rows + other_rows
    return {key: [keys[i % len(keys)] for i in range(n)],
            "Accommodation": [CANARY if i < canary_rows else COMMON[i % 4] for i in range(n)]}


def cross(tmp: Path, name: str, table: dict[str, list[str]], *flags: str, run_name: str = "run"):
    src = tmp / f"{run_name}-in"
    write_csv(src / "students.csv", students())
    write_csv(src / name, table)
    res, out, rep = run(src, tmp, *flags, name=run_name)
    assert res.exit_code == 0, res.output
    return res, out, rep


# =========================================================================== K-04: the report's advice
def test_integer_id_advice_names_id_col_first() -> None:
    ids = [str(4810000 + s * 37) for s in range(60) for _ in range(1 + s % 5)]
    det = detect(ids, "Incident_ID")
    msg = next(m for m in det.messages if m.code == "possible_sparse_code")
    assert msg.fix.startswith("--id-col Incident_ID")
    assert "--person-id Incident_ID only when it numbers the row's own person" in msg.fix
    assert "never for incident or record numbers" in msg.text


def test_id_candidate_advice_names_id_col_first() -> None:
    n = 60
    tp, _ = run_table({"Ticket": [f"TK{900000 + i * 13}" for i in range(n)], "Grade": [str(9 + i % 4) for i in range(n)]})
    msg = next(m for m in tp.messages if m.code in ("no_id_column", "id_candidates") and m.fix)
    assert msg.fix.startswith("--id-col COL for each person or record ID column")
    assert "--person-id COL only for the one that numbers the row's own person" in msg.fix


# =========================================================================== K-04, K-05: claims never outvote proof
@pytest.mark.parametrize("key,keys,flag", [
    ("Incident_ID", [str(9000 + i) for i in range(212)], "--person-id"),  # K-04: each incident claimed a person
    ("Badge_No", [str(20001 + 37 * i) for i in range(12)], "--person-id"),  # K-05: 12 nurses claimed 12 people
])
def test_a_person_id_claim_never_outvotes_proven_students(tmp_path: Path, key: str, keys: list[str],
                                                          flag: str) -> None:
    res, out, rep = cross(tmp_path, "log.csv", log(key, keys), flag, key)
    assert found((CANARY,), out, rep, console=res.output) == []
    for name in ("students.csv", "log.csv"):
        assert file_report(rep, name)["checks"]["leak_scan"]["result"] == "PASS"
    # utility: the values the proven table shows held by >= 10 students stay real in both files
    for name in ("students.csv", "log.csv"):
        written = Counter(r["Accommodation"] for r in read_dicts(out / name))
        assert set(COMMON) <= set(written), name
    # N in = N out, and the pseudonym keeps the canary's exact counts in each file
    syn_students = Counter(r["Accommodation"] for r in read_dicts(out / "students.csv"))
    syn_log = Counter(r["Accommodation"] for r in read_dicts(out / "log.csv"))
    fake = set(syn_students) - set(COMMON)
    assert len(fake) == 1 and syn_students[next(iter(fake))] == 1 and syn_log[next(iter(fake))] == 12
    assert sum(syn_log.values()) == 212


def test_a_person_id_claim_alone_still_counts_its_people(tmp_path: Path) -> None:
    # the explicit claim is honoured where nothing proves fewer holders: 12 badges hold the value, no
    # d35_rows_counted warning (the user said the column numbers the row's own person)
    src = tmp_path / "in"
    write_csv(src / "log.csv", log("Badge_No", [str(20001 + 37 * i) for i in range(12)]))
    res, out, rep = run(src, tmp_path, "--person-id", "Badge_No")
    assert res.exit_code == 0, res.output
    assert Counter(r["Accommodation"] for r in read_dicts(out / "log.csv"))[CANARY] == 12
    assert "d35_rows_counted" not in warning_codes(rep, "log.csv")


def test_a_claim_on_the_runs_student_ids_is_proof(tmp_path: Path) -> None:
    # --person-id on a column holding the students' own IDs (another header): its counts are of proven people, so a
    # value 12 students hold there is kept in every file (the most people any proven table shows). The header names
    # no student ('Kid_Ref' is a K-02 student-ID header, a header proof by itself), so this is a claim the run's
    # student IDs corroborate
    ids = students()["Student_ID"]
    table = {"Ref_No": ids[:12] + ids[12:60] * 3,
             "Accommodation": [CANARY] * 12 + [COMMON[i % 4] for i in range(144)]}
    res, out, rep = cross(tmp_path, "log.csv", table, "--person-id", "Ref_No")
    assert Counter(r["Accommodation"] for r in read_dicts(out / "log.csv"))[CANARY] == 12
    assert Counter(r["Accommodation"] for r in read_dicts(out / "students.csv"))[CANARY] == 1
    assert "d35_rows_counted" not in warning_codes(rep, "log.csv")


def _col(field: str, level: Any, holders: dict[str, int]) -> Any:
    return SimpleNamespace(name=field, params={dtext.PENDING_KEY: True, dtext.PEOPLE_KEY: level,
                                               dtext.HOLDERS_KEY: holders})


def test_run_holders_ranks_proof_over_claims_over_rows() -> None:
    P, C = dtext.PEOPLE_PROVEN, dtext.PEOPLE_CLAIMED
    cols = [_col("Plan", P, {"a": 1, "b": 15}), _col("Plan", P, {"a": 3}), _col("Plan", C, {"a": 40, "c": 30}),
            _col("Plan", C, {"c": 11, "d": 50}), _col("Plan", 0, {"a": 90, "d": 70, "e": 60})]
    people: dict[str, set[str]] = {}
    out = dtext.run_holders(cols, people_units=people)["plan"]
    assert out["a"] == 3  # proven: the most people any proven table shows; claims and rows never outvote it
    assert out["b"] == 15
    assert out["c"] == 11 and out["d"] == 50  # claims only: the FEWEST any claim shows
    assert out["e"] == 60  # rows only where no table counts people
    assert people["plan"] == {"a", "b", "c", "d"}
    # an old boolean people flag reads as proof
    assert dtext.run_holders([_col("Plan", True, {"a": 1}), _col("Plan", C, {"a": 20})])["plan"]["a"] == 1


# =========================================================================== K-06: a bare SID
@pytest.mark.parametrize("name,flags,proof", [
    ("Student_ID", {}, entity.PROOF_HEADER), ("LASID", {}, entity.PROOF_HEADER), ("Perm ID", {}, entity.PROOF_HEADER),
    ("SSID", {}, entity.PROOF_HEADER), ("SID", {}, entity.PROOF_SID), ("SID", {"id_col": ["SID"]}, entity.PROOF_SID),
    ("Incident_ID", {"person_id": ["Incident_ID"]}, entity.PROOF_FLAG),
    ("SID", {"person_id": ["SID"]}, entity.PROOF_FLAG),
    ("Student_ID", {"person_id": ["Student_ID"]}, entity.PROOF_HEADER),
    ("TeacherID", {}, None), ("DCID", {}, None)])
def test_subject_proof(name: str, flags: dict[str, list[str]], proof: str | None) -> None:
    det = detect([f"{1000 + i // 3}" for i in range(30)], name, **flags)
    assert det.type is T.IDENTIFIER, name
    p = SimpleNamespace(name=name, source=det.source, type=det.type, params=det.params)
    assert entity.subject_proof(p) == proof, name


def test_a_bare_sid_of_sessions_never_clears_a_student_value(tmp_path: Path) -> None:
    sessions = log("SID", [str(5000 + i) for i in range(212)])
    sessions["Accommodation"][12:24] = ["Zarnow sensory kit"] * 12  # held only here, by 12 sessions
    res, out, rep = cross(tmp_path, "sessions.csv", sessions)
    assert found((CANARY,), out, rep, console=res.output) == []
    # the sessions are records: a value only they hold is kept for its rows, with the warning and its fix
    assert Counter(r["Accommodation"] for r in read_dicts(out / "sessions.csv"))["Zarnow sensory kit"] == 12
    assert "d35_rows_counted" in warning_codes(rep, "sessions.csv")
    assert "d35_rows_counted" not in warning_codes(rep, "students.csv")


def test_a_bare_sid_of_the_runs_students_proves_people(tmp_path: Path) -> None:
    # SID holds the students' own IDs: its people are proven (no warning), and its counts decide with the
    # students table's (a value 12 students hold in the log is kept)
    ids = students()["Student_ID"]
    table = {"SID": ids[:12] + ids[12:60] * 3, "Accommodation": [CANARY] * 12 + [COMMON[i % 4] for i in range(144)]}
    res, out, rep = cross(tmp_path, "log.csv", table)
    assert "d35_rows_counted" not in warning_codes(rep, "log.csv")
    assert Counter(r["Accommodation"] for r in read_dicts(out / "log.csv"))[CANARY] == 12


def test_dense_session_numbers_never_match_dense_student_ids_by_chance(tmp_path: Path) -> None:
    # sessions numbered 5000-5059 beside students numbered 5000-5059: equal numbers of a dense, row-number-like range
    # are chance (as R9-01 reads them), never proof that the sessions are the students
    src = tmp_path / "in"
    write_csv(src / "students.csv", {"Student_ID": [str(5000 + i) for i in range(60)],
                                     "Accommodation": [CANARY if i == 0 else COMMON[i % 4] for i in range(60)]})
    write_csv(src / "sessions.csv", log("SID", [str(5000 + i) for i in range(60)]))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found((CANARY,), out, rep, console=res.output) == []


def test_a_bare_sid_backed_by_names_proves_people() -> None:
    n = 120
    names = [f"{a}, {b}" for a in ("Adams", "Brook", "Cole", "Dunn", "Ellis", "Ford", "Gray", "Hale", "Ives", "Judd")
             for b in ("Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lia")]
    cols = {"SID": [str(5000 + i) for i in range(n)], "Name": names,
            "Club": ["Zebra Club" if i < 12 else "Chess" for i in range(n)]}
    tp, _ = run_table(cols)
    p = profile_of(tp, "Club")
    assert not any(m.code == "d35_rows_counted" for m in p.messages)
    assert dtext.pseudonymized(p) == {}  # 12 students (12 SIDs, 12 names)
    # the same SIDs numbering visits (one name, many SIDs) do not agree with the names: records, not proof
    cols["Name"] = [names[0]] * 12 + names[12:]
    tp, _ = run_table(cols)
    assert set(dtext.pseudonymized(profile_of(tp, "Club"))) == {"Zebra Club"}  # one name: one holder


def test_one_to_one() -> None:
    a = np.array([0, 1, 2, 3, -1])
    assert entity._one_to_one(a, np.array([5, 6, 7, 8, 9]))
    assert not entity._one_to_one(a, np.array([5, 5, 5, 5, 9]))  # one name, many IDs
    assert not entity._one_to_one(np.array([-1, -1]), np.array([1, 2]))


# =========================================================================== K-08: ZIP codes
def test_zip_small_cell_policy_counts_people_not_rows() -> None:
    n_students, per = 40, 3
    sid = [f"S{300000 + s}" for s in range(n_students) for _ in range(per)] + ["S399999"] * 12
    zips = [("94103", "94110", "94112")[(s + r) % 3] for s in range(n_students) for r in range(per)] + ["99577"] * 12
    tp, cols = run_table({"student_id": sid, "Zip": zips})
    assert profile_of(tp, "Zip").type is T.ZIP
    out = Counter(cols["Zip"].tolist())
    assert "99577" not in out  # one student's code on 12 rows: one holder
    assert {"94103", "94110", "94112"} <= set(out) and sum(out.values()) == len(zips)
    assert sorted(out.values()) == sorted(Counter(zips).values())  # exact counts (the rare code under another code)


def test_zip_held_by_many_people_is_kept_and_rows_count_without_ids() -> None:
    zips = ["99577"] * 12 + [("94103", "94110")[i % 2] for i in range(100)]
    tp, cols = run_table({"student_id": [f"S{1000 + i}" for i in range(112)], "Zip": zips})
    assert Counter(cols["Zip"].tolist())["99577"] == 12  # 12 students
    tp, cols = run_table({"Zip": zips})
    assert Counter(cols["Zip"].tolist())["99577"] == 12  # no ID: rows count (D35)


# =========================================================================== K-11: whole combinations
LASTS, FIRSTS = ("Okafor", "Garcia", "Nguyen"), ("Lin", "Maria", "Tuan")


def pickup(n_students: int = 120) -> list[str]:
    """``Last, First`` pairs (two common halves each) under a non-personal header, with single-word values that
    make the comma column a multi-select; ``Okafor, Tuan`` is held by one student."""
    pairs = [f"{a}, {b}" for a in LASTS for b in FIRSTS if (a, b) != ("Okafor", "Tuan")]
    vals = [pairs[i % len(pairs)] for i in range(n_students - 19)] + ["Bus"] * 9 + ["Walker"] * 9 + ["Okafor, Tuan"]
    return vals


@pytest.mark.parametrize("header", ["Pickup_List", "Carpool"])
def test_a_rare_pair_of_common_halves_is_never_rebuilt(header: str) -> None:
    vals = pickup()
    tp, cols = run_table({"student_id": [f"S{1000 + i}" for i in range(len(vals))], header: vals})
    p = profile_of(tp, header)
    assert p.type is T.MULTI_SELECT and dtext.multi_select_mode(p) == "combinations"
    out = cols[header].tolist()
    assert "Okafor, Tuan" not in out and "Okafor, Tuan" in dtext.pseudonymized(p)
    assert len(out) == len(vals) and sorted(Counter(out).values()) == sorted(Counter(vals).values())
    kept = [v for v in set(vals) if vals.count(v) >= 10 and v != "Bus"]
    assert set(kept) <= set(out)  # combinations held by >= 10 students stay real
    assert any(m.code == "pseudonymized_combinations" for m in p.messages)
    assert leaks.sensitive_kind(p) == leaks.LABEL


def test_two_guardian_rows_never_rebuild_a_one_student_name() -> None:
    vals = (["Garcia, Maria"] * 20 + ["Okafor, Maria"] * 20 + ["Garcia, Tuan"] * 20 + ["Nguyen, Maria"] * 20
            + ["Nguyen, Tuan"] * 15 + ["Garcia, Maria; Garcia, Tuan"] * 12 + ["Nguyen, Maria; Nguyen, Tuan"] * 12
            + ["Okafor, Tuan"])
    tp, cols = run_table({"student_id": [f"S{2000 + i}" for i in range(len(vals))], "Authorized_Pickup": vals})
    assert profile_of(tp, "Authorized_Pickup").type is T.MULTI_SELECT
    out = cols["Authorized_Pickup"].tolist()
    assert "Okafor, Tuan" not in out
    assert Counter(out)["Garcia, Maria; Garcia, Tuan"] == 12


def test_vocabulary_combinations_and_common_combinations_stay_real() -> None:
    # no regression: a rare combination of vocabulary words is kept (D35 keeps vocabulary whatever its count), a rare
    # item is pseudonymised item by item as before (the combination is not pseudonymised a second time)
    n = 120
    vals = (["ELL;SPED"] * 40 + ["ELL"] * 30 + ["Robotics Club;ELL"] * 20 + ["SPED"] * 22 + ["ELL;SPED;504"] * 2
            + ["Ana Lopez;SPED"] * 3 + ["Robotics Club;SPED"] * 3)
    assert len(vals) == n
    tp, cols = run_table({"student_id": [f"S{3000 + i}" for i in range(n)], "Services": vals})
    p = profile_of(tp, "Services")
    assert p.type is T.MULTI_SELECT
    assert set(dtext.pseudonymized(p)) == {"Ana Lopez", "Robotics Club;SPED"}
    out = Counter(cols["Services"].tolist())
    assert out["ELL;SPED;504"] == 2 and out["Robotics Club;ELL"] == 20 and out["ELL;SPED"] == 40
    fakes = [v for v in out if v not in vals]
    assert len(fakes) == 2
    # the pseudonym of the whole combination keeps its vocabulary item and its shape
    combo = next(v for v in fakes if len(v) == len("Robotics Club;SPED"))
    assert combo.endswith(";SPED") and not combo.startswith("Robotics") and out[combo] == 3


def test_combination_pseudonyms_are_deterministic() -> None:
    vals = pickup()
    sid = [f"S{1000 + i}" for i in range(len(vals))]
    _, a = run_table({"student_id": sid, "Pickup_List": vals}, seed=3)
    _, b = run_table({"student_id": sid, "Pickup_List": vals}, seed=3)
    assert a["Pickup_List"].tolist() == b["Pickup_List"].tolist()


def test_leak_scan_catches_a_rebuilt_combination() -> None:
    vals = pickup()
    tp, cols = run_table({"student_id": [f"S{1000 + i}" for i in range(len(vals))], "Pickup_List": vals})
    p = profile_of(tp, "Pickup_List")
    synth = cols["Pickup_List"].copy()
    synth[0] = "Okafor, Tuan"
    st = SyntheticTable(tp.name, tp.header, tp.n_rows, [SyntheticColumn(p.index, p.name, synth)], tp)
    assert any(m.code == "copied_value" and m.column == "Pickup_List" for m in leaks.scan_table(tp, st))
    st_ok = SyntheticTable(tp.name, tp.header, tp.n_rows, [SyntheticColumn(p.index, p.name, cols["Pickup_List"])], tp)
    assert not [m for m in leaks.scan_table(tp, st_ok) if m.code == "copied_value"]


def test_cli_pickup_list_passes_and_never_says_invented(tmp_path: Path) -> None:
    vals = pickup()
    src = tmp_path / "in"
    write_csv(src / "pickup.csv", {"student_id": [f"S{1000 + i}" for i in range(len(vals))], "Pickup_List": vals})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Okafor, Tuan",), out, rep, console=res.output) == []
    data = file_report(rep, "pickup.csv")
    assert data["checks"]["leak_scan"]["result"] == "PASS"
    md = (rep / "files" / "pickup.csv.md").read_text(encoding="utf-8")
    line = next(ln for ln in md.splitlines() if ln.startswith("- `Pickup_List`: **"))
    assert "invented" not in line and "whole combinations are written by exact quota" in line


# =========================================================================== H-1: a bare SID's counts are a claim
def sid_students(n: int = 60) -> dict[str, list[str]]:
    """:func:`students` keyed by a bare ``SID`` (no name column: nothing ties it to the students)."""
    t = students(n)
    return {"SID": t["Student_ID"], "Accommodation": t["Accommodation"]}


@pytest.mark.parametrize("name,table,flags", [
    # 12 staff members hold the value one student holds (Staff_ID numbers other people)
    ("staff.csv", {"Staff_ID": [f"E{5000 + i}" for i in range(80)],
                   "Accommodation": [CANARY if i < 12 else COMMON[i % 4] for i in range(80)]}, ()),
    # K-05 on top: --person-id on the staff column claims 12 people; the fewest claim (one SID) still wins
    ("staff.csv", {"Staff_ID": [f"E{5000 + i}" for i in range(80)],
                   "Accommodation": [CANARY if i < 12 else COMMON[i % 4] for i in range(80)]},
     ("--person-id", "Staff_ID")),
    # 11 incidents without an ID column, and the same with the report's own advice, --id-col Incident_ID
    ("incidents.csv", log("Incident_ID", [str(9000 + 7 * i) for i in range(211)], canary_rows=11), ()),
    ("incidents.csv", log("Incident_ID", [str(9000 + 7 * i) for i in range(211)], canary_rows=11),
     ("--id-col", "Incident_ID")),
])
def test_a_records_sibling_never_outvotes_a_bare_sid(tmp_path: Path, name: str, table: dict[str, list[str]],
                                                     flags: tuple[str, ...]) -> None:
    src = tmp_path / "in"
    write_csv(src / "students.csv", sid_students())
    write_csv(src / name, table)
    res, out, rep = run(src, tmp_path, *flags)
    assert res.exit_code == 0, res.output
    assert found((CANARY,), out, rep, console=res.output) == []
    for f in ("students.csv", name):
        assert file_report(rep, f)["checks"]["leak_scan"]["result"] == "PASS"
    # the pseudonym keeps the exact counts in both files, and the values the SIDs show held by >= 10 stay real
    syn = Counter(r["Accommodation"] for r in read_dicts(out / "students.csv"))
    fake = set(syn) - set(COMMON)
    assert len(fake) == 1 and syn[next(iter(fake))] == 1 and set(COMMON) <= set(syn)
    rows = sum(1 for v in table["Accommodation"] if v == CANARY)
    assert Counter(r["Accommodation"] for r in read_dicts(out / name))[next(iter(fake))] == rows


def test_run_holders_ranks_a_bare_sid_as_a_claim() -> None:
    P, C = dtext.PEOPLE_PROVEN, dtext.PEOPLE_CLAIMED
    sid = _col("Plan", C, {"a": 1, "b": 30, "c": 12})
    sid.params[dtext.SID_KEY] = ["SID"]
    people: dict[str, set[str]] = {}
    out = dtext.run_holders([sid, _col("Plan", 0, {"a": 12, "b": 40, "d": 50}), _col("Plan", C, {"c": 11})],
                            people_units=people)["plan"]
    assert out == {"a": 1, "b": 30, "c": 11, "d": 50}  # records never outvote it; the fewest claim wins
    assert people["plan"] == {"c"}  # the SID's own units are no asserted person's (d35_rows_counted)
    assert dtext.run_holders([sid, _col("Plan", P, {"a": 3, "b": 2})])["plan"] == {"a": 3, "b": 2, "c": 12}
    keys = entity.HolderKeys([], False, "SID values", sids=("SID",))
    assert dtext.people_level(keys) == C
    assert dtext.people_level(entity.HolderKeys([], False, "rows")) == 0


# =========================================================================== H-6: the warning of a bare SID table
def attendance(n: int = 120) -> dict[str, list[str]]:
    ids = [str(482913 + 37 * i) for i in range(n)]
    reasons = ("Illness", "Medical appointment", "Family emergency", "Zarnow clinic visit")
    return {"SID": [ids[i % n] for i in range(3 * n)], "Reason": [reasons[(i // 7) % 4] for i in range(3 * n)]}


def test_the_sid_warning_says_what_is_unknown_and_how_to_confirm(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "attendance.csv", attendance())
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    msg = next(w for w in data["warnings"] if w["code"] == "d35_rows_counted" and w.get("column") == "Reason")
    assert "distinct SID values hold them" in msg["text"] and "nothing shows that SID numbers" in msg["text"]
    assert "no ID or student-name column of this table" not in msg["text"]  # false for a student SID
    assert msg["fix"].startswith("--person-id SID if SID is the student's own ID")
    # the advice works: confirmed, the SIDs count as people and the warning goes
    res, out, rep = run(src, tmp_path, "--person-id", "SID", name="confirmed")
    assert res.exit_code == 0, res.output
    assert "d35_rows_counted" not in warning_codes(rep, "attendance.csv")


def test_a_dense_roster_does_not_prove_a_sid_log_by_its_values(tmp_path: Path) -> None:
    # H-6 (the attacker's subset rule refuted): every SID of the log is a roster ID, but the roster numbers its
    # students 1001-1300 densely, where a session numbering 1001-1220 is a subset as well (next test); the log keeps
    # its warning, and confirming SID with --person-id (the warning's own fix) removes it
    ids = [str(1001 + i) for i in range(300)]
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"Student_ID": ids, "Home_Language": ["English", "Spanish", "Somali"] * 100})
    visits = [ids[(7 * i) % 300] for i in range(220) for _ in range(1 + i % 3)]
    reasons = ("Headache", "Stomach ache", "Injury", "Medication")
    write_csv(src / "nurse.csv", {"SID": visits, "Reason": [reasons[i % 4] for i in range(len(visits))]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "d35_rows_counted" in warning_codes(rep, "nurse.csv")
    res, out, rep = run(src, tmp_path, "--person-id", "SID", name="confirmed")
    assert res.exit_code == 0, res.output
    assert "d35_rows_counted" not in warning_codes(rep, "nurse.csv")


def test_a_session_numbering_inside_a_dense_roster_never_proves_people(tmp_path: Path) -> None:
    # why a SID whose values are a SUBSET of a dense roster's IDs proves nothing: here SID numbers sessions 1001-1220
    # (one row each) and one student's 12 sessions hold CANARY, which the roster shows held by one student. Read as
    # proven people, the 12 session numbers would keep CANARY in both files (12 + 1 rows); as a claim, the fewest wins
    ids = [str(1001 + i) for i in range(300)]
    src = tmp_path / "in"
    write_csv(src / "students.csv", {"Student_ID": ids,
                                     "Accommodation": [CANARY if i == 0 else COMMON[i % 4] for i in range(300)]})
    write_csv(src / "sessions.csv", {"SID": ids[:220],
                                     "Accommodation": [CANARY if i < 12 else COMMON[i % 4] for i in range(220)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found((CANARY,), out, rep, console=res.output) == []
    fake = set(Counter(r["Accommodation"] for r in read_dicts(out / "students.csv"))) - set(COMMON)
    assert len(fake) == 1
    assert Counter(r["Accommodation"] for r in read_dicts(out / "sessions.csv"))[next(iter(fake))] == 12


# =========================================================================== H-7: rare combinations keep what is common
ACC = ("Extended time", "Small group", "Read aloud", "Preferential seating")


def accommodations() -> list[str]:
    """An ordinary multi-select (written as combinations): four common items; every pair but ``Small group;Read
    aloud`` is held by >= 10 students, every triple by fewer."""
    plan = [((0,), 44), ((1,), 37), ((2,), 54), ((3,), 51), ((0, 1), 16), ((0, 3), 15), ((0, 2), 15), ((2, 3), 15),
            ((1, 3), 14), ((1, 2), 4), ((0, 2, 3), 8), ((0, 1, 3), 6), ((0, 1, 2), 6), ((1, 2, 3), 5)]
    return [";".join(ACC[i] for i in combo) for combo, n in plan for _ in range(n)]


def items_of(values: list[str], d: str = ";") -> Counter:
    return Counter(x.strip() for v in values for x in v.split(d) if x.strip())


def written_forms(p: Any, d: str) -> dict[str, str]:
    """Each rare whole combination of column ``p`` -> its written form (``generate.pseudonyms``)."""
    return {real: syn for real, syn in pseudonyms.level_map(p).items() if d in real}


def item_pseudonyms(forms: dict[str, str], d: str) -> dict[str, str]:
    """Real item -> its pseudonym inside the written combinations; asserts one pseudonym per item."""
    fake_of: dict[str, str] = {}
    for real, syn in forms.items():
        for r, s in zip(real.split(d), syn.split(d)):
            if r.strip() != s.strip():
                assert fake_of.setdefault(r.strip(), s.strip()) == s.strip(), r  # consistent over the run
    return fake_of


def test_rare_combinations_keep_a_common_sub_combination_and_one_pseudonym_per_item() -> None:
    vals = accommodations()
    tp, cols = run_table({"student_id": [f"S{4000 + i}" for i in range(len(vals))], "Accommodations": vals})
    p = profile_of(tp, "Accommodations")
    assert p.type is T.MULTI_SELECT and dtext.multi_select_mode(p) == "combinations"
    out = cols["Accommodations"].tolist()
    assert sorted(Counter(out).values()) == sorted(Counter(vals).values())  # exact combination counts (D14)
    held = Counter(vals)
    rare = {v for v in held if held[v] < 10 and ";" in v}
    assert len(rare) == 5 and not rare & set(out)
    forms = written_forms(p, ";")
    assert set(forms) == rare
    real_items = items_of(vals)
    for real, syn in forms.items():
        kept = [x for x in syn.split(";") if x in real_items]
        if real.count(";") == 2:
            # a triple keeps the two items one of its common pairs holds, in its order (that text is written anyway)
            assert len(kept) == 2 and held[";".join(kept)] >= 10, real
        else:
            # the rare pair keeps one item: the most widely held (Read aloud, 107 rows, over Small group, 88)
            assert kept == ["Read aloud"], real
    # one pseudonym per replaced item (not one per combination), and item prevalence holds under it
    fake_of = item_pseudonyms(forms, ";")
    out_items = items_of(out)
    assert set(out_items) == set(real_items) | set(fake_of.values())
    for r, c in real_items.items():
        assert out_items[r] + out_items.get(fake_of.get(r, ""), 0) == c, r
    # five rare combinations replace one item each, under at most three pseudonyms
    assert sum(out_items[s] for s in fake_of.values()) == sum(held[v] for v in rare)
    assert len(fake_of) <= 3


def test_cli_rare_combinations_pass_the_leak_scan(tmp_path: Path) -> None:
    vals = accommodations()
    src = tmp_path / "in"
    write_csv(src / "iep.csv", {"Student_ID": [f"S{7000 + i}" for i in range(len(vals))], "Accommodations": vals})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert file_report(rep, "iep.csv")["checks"]["leak_scan"]["result"] == "PASS"
    written = [r["Accommodations"] for r in read_dicts(out / "iep.csv")]
    held = Counter(vals)
    assert not {v for v in held if held[v] < 10 and ";" in v} & set(written)
    assert sorted(Counter(written).values()) == sorted(held.values())
    # every item keeps at least 85% of its rows (the rest are its one pseudonym; 76-83% when every item of
    # every rare combination got a fresh pseudonym)
    real_items, out_items = items_of(vals), items_of(written)
    assert all(out_items[i] >= 0.85 * real_items[i] for i in ACC)


def test_combo_stay_of_columns_that_disagree() -> None:
    # one column may write "x;y" together inside the combination, the other "y;z": together, at most one real item
    a, b = (frozenset({"ell"}), ("x", "y")), (frozenset({"ell", "504"}), ("y", "z"))
    assert pseudonyms._combo_stay([a]) == {"ell", "x", "y"}
    assert pseudonyms._combo_stay([a, a]) == {"ell", "x", "y"}
    assert pseudonyms._combo_stay([a, b]) == {"ell", "y"}
    assert pseudonyms._combo_stay([a, (frozenset(), ("z", "w"))]) == set()
    assert dtext._subsequence(("a", "c"), ("a", "b", "c")) and not dtext._subsequence(("c", "a"), ("a", "b", "c"))


def two_guardians() -> list[str]:
    """``Last, First`` names joined by commas only (two guardians in one cell); ``Okafor, Tuan, Scott, Zelda`` holds
    one student's guardians: Okafor, Tuan and Scott are common halves, Zelda is held by one student."""
    pairs = ["Okafor, Lin", "Garcia, Tuan", "Scott, Maria", "Nguyen, Tuan", "Okafor, Maria", "Scott, Lin"]
    vals = [pairs[i % len(pairs)] for i in range(96)] + ["Bus"] * 12 + ["Walker"] * 11
    return vals + ["Okafor, Tuan, Scott, Zelda"]


def test_a_rare_item_never_lets_two_common_halves_stand_together() -> None:
    vals = two_guardians()
    tp, cols = run_table({"student_id": [f"S{5000 + i}" for i in range(len(vals))], "Pickup": vals})
    p = profile_of(tp, "Pickup")
    assert p.type is T.MULTI_SELECT and dtext.multi_select_mode(p) == "combinations"
    out = cols["Pickup"].tolist()
    assert sorted(Counter(out).values()) == sorted(Counter(vals).values())
    real_items = items_of(vals, ",")
    for v in set(out) - set(vals):
        parts = [x.strip() for x in v.split(",")]
        assert sum(1 for x in parts if x in real_items) <= 1, v  # never "Okafor, Tuan" beside a pseudonymised Zelda
    assert not any("Okafor, Tuan" in v or "Zelda" in v for v in out)


def test_a_common_whole_name_may_stand_inside_a_rare_combination() -> None:
    # "Tuan, Scott" is itself a whole value 12 students hold (written as it is anyway): inside the rare
    # "Okafor, Tuan, Scott, Zelda" those two items stay together; Okafor and Zelda become their pseudonyms, so the
    # one-student name "Okafor, Tuan" is still never rebuilt
    vals = two_guardians()[:-1] + ["Tuan, Scott"] * 12 + ["Okafor, Tuan, Scott, Zelda"]
    tp, cols = run_table({"student_id": [f"S{5000 + i}" for i in range(len(vals))], "Pickup": vals})
    p = profile_of(tp, "Pickup")
    assert p.type is T.MULTI_SELECT and dtext.multi_select_mode(p) == "combinations"
    forms = written_forms(p, ",")
    syn = forms["Okafor, Tuan, Scott, Zelda"]
    parts = [x.strip() for x in syn.split(",")]
    assert parts[1:3] == ["Tuan", "Scott"] and "Okafor" not in parts and "Zelda" not in parts
    out = cols["Pickup"].tolist()
    assert not any("Okafor, Tuan" in v or "Zelda" in v for v in out)
    assert Counter(out)[syn] == 1 and Counter(out)["Tuan, Scott"] == 12


def test_cli_a_kept_item_only_in_rare_combinations_passes_the_leak_scan(tmp_path: Path) -> None:
    # Okafor is held by 12 students, but only inside combinations each held by fewer than 10: it is the item those
    # combinations keep, and the leak scan excuses exactly it (no false FAIL); the rare whole values are never written
    common = [f"{a}, {b}" for a in ("Garcia", "Nguyen") for b in ("Lin", "Maria", "Tuan")]
    vals = ([common[i % 6] for i in range(72)] + ["Okafor, Ann"] * 6 + ["Okafor, Bea"] * 6 + ["Ann"] * 4
            + ["Bea"] * 4 + ["Bus"] * 9 + ["Walker"] * 9)
    src = tmp_path / "in"
    write_csv(src / "pickup.csv", {"student_id": [f"S{6000 + i}" for i in range(len(vals))], "Pickup_List": vals})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert file_report(rep, "pickup.csv")["checks"]["leak_scan"]["result"] == "PASS"
    written = Counter(r["Pickup_List"] for r in read_dicts(out / "pickup.csv"))
    assert not {"Okafor, Ann", "Okafor, Bea"} & set(written)
    assert sum(c for v, c in written.items() if v.startswith("Okafor, ")) == 12
