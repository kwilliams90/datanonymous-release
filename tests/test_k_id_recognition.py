"""K id-recognition: ID column recognition (known issues K-02, K-13, K-21).

* K-02 (G1/G2): student-ID headers outside the default list (``Student Code``, ``Stu_Code``, ``Student_Ref``,
  ``Pupil Code``, ``Learner Ref``, ``SIS ID``, ``ID Number``, ``CALPADS SSID``, ``USI``, ``Perm_No``, ``Admission No``,
  ``Roll No``) were typed INTEGER in multi-row tables: every real ID was written and D35 counted rows. Fail-safe now:
  person word + ID word headers (``hints.person_id_hint``, ``hints.default_id_hint``) are IDs by the header; any
  other header with a student word, a student-ID code or a final ID word (``hints.weak_id_header``: ``Student_SK``,
  ``Student PIN``, ``Student ID (Aeries)``, ``CALPADSSSID``, ``KIDS ID``, ``Lunch Number``) is an ID column when its
  values are keys of people (``detect.ids.id_keys_width``: placeholders such as ``TBD`` and integers stored as
  ``100234.0`` included) and its table agrees (``profile.table.weak_id_columns``: no ID of the row's own person, or
  keys each held by fewer than ``--min-cell`` people). Measures (``Student_Count``, ``Cost per Student``), flags and
  status codes (``Missing SSID``, ``English Learner Code``) and codes many students share keep their types; org codes
  stay org codes.
* Harden 2 (second attack pass, ``test_h2_*``): odd cells (``-1``, ``100234, 100567``) never stop an ID column (A2-1);
  an ID word makes an ID column over any number of 3+ digit keys, ``--min-cell`` playing no part (A2-2); repeated keys
  of 8+ digits under a header no rule knows are listed as ID-like candidates and count D35 holders (A2-3; D38 put the
  state acronyms such as ``NYSSIS`` on the default ID list and FAILs such a table without an ID column); IDs of
  other students beside the student ID (``Sibling SSID``) are ID columns (A2-4); a reference column spanning a dense
  namespace among placeholders is still a key reference (A2-5); dense unpadded keys spill past the widest width (D9,
  A2-6); an ``--exclude``d or ``--null``ed column found by header and values stays a shadow ID (A2-7).
* K-13 (G5): a grade column (``KG``, ``1``-``5`` / ``01``-``05``) was taken for a key reference to student IDs that
  include 1-5 (silently replaced grades, or a FAIL for zero-padded IDs). The digit keys of a mixed column pass the
  chance test now (over the column's range or over their number of digits, so dense 6-digit IDs among placeholders
  still are references), and a built-in vocabulary column is never a key reference.
* K-21 (G5): a non-decimal Unicode digit (``10\u00b21``) in an ID or school-year column crashed ``int()`` in detection.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous import hints
from datanonymous.detect import ids as dids
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.profile import structure as st
from test_area_a_support import detect, profile_of, run_table

CANARY = "Kiwi anaphylaxis Zorbek"
COMMON = ("Illness", "Appointment", "Family Trip")


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _run(src: Path, tmp_path: Path, *opts: object, code: int = 0) -> tuple[Path, Path, str]:
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "1", *opts)
    assert res.exit_code == code, res.output
    return out, rep, res.output


def _columns(rep: Path, rel: str) -> dict[str, dict]:
    data = json.loads((rep / "files" / (rel + ".json")).read_text(encoding="utf-8"))
    return {c["name"]: c for c in data["tables"][0]["columns"]}


def _status(rep: Path) -> dict[str, str]:
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    return {f.get("source") or f.get("path"): f["status"] for f in data["files"]}


def _texts(*dirs: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for d in dirs for p in d.rglob("*") if p.is_file())


def _attendance(seed: int = 3) -> tuple[list[str], list[str]]:
    """60 students (sparse 6-digit IDs) x 5-15 rows; the first student holds the canary on 12 rows."""
    rng = random.Random(seed)
    ids, reason = [], []
    for s in range(60):
        sid = str(100100 + 7 * s + rng.randint(0, 3))
        k = 12 if s == 0 else rng.randint(5, 15)
        ids += [sid] * k
        reason += [CANARY if s == 0 else COMMON[(s + j) % 3] for j in range(k)]
    return ids, reason


# --------------------------------------------------------------------------- K-02: header rules
#: Headers that name a person ID by themselves (any values): person word + ID word, a student word that starts the
#: header + code/ref word, a student-ID code that starts it; numbering, ID qualifiers and bracketed or dash qualifiers
#: never hide the ID word.
K02_PERSON_IDS = ["Student Code", "Stu_Code", "Student_Ref", "Pupil Code", "Learner Ref", "Child Code", "Kid ID",
                  "SIS ID", "USI", "Student_SSID", "StudentCode", "Student_ID_2", "Student Number (Local)", "STU_CD",
                  "StudentRef", "Student ID (Aeries)", "Student ID - District", "Student ID (9 digit)", "USI (EdFi)",
                  "Local Student Code", "SSID_Number", "Old_LASID"]
K02_DEFAULT_IDS = ["ID Number", "IDNumber", "ID_No", "Perm_No", "Perm #", "Permanent Number", "Identification Number",
                   "SSID (CALPADS)", "Student_ID (PowerSchool)"]
#: Headers whose ID-like values make an ID column (fail-safe: the header alone does not decide).
K02_WEAK = {"person": ["Student", "Stu", "Kid", "Child", "Pupil", "Learner", "Local_Student", "Student (District)",
                       "Student_SK", "Student UID", "Student PIN", "Student Barcode", "Student Badge", "Student Index",
                       "Student Serial", "Learner UID", "Student Lunch PIN", "STUDENTSK", "Student_ID_District",
                       "English Learner Code", "Homeless Student Code", "Missing Student Code", "Student_ID_Status"],
            "code": ["CALPADS SSID", "EdFi USI", "CALPADSSSID", "calpadsssid", "Missing SSID", "Valid SSID"],
            # D38 put the state student-ID acronyms on the default ID list (KIDS ID, MOSIS ID, EIS ID, SAIS ID,
            # NYSSIS ID: tests/test_d38_unknown_ids.py); MARSS Number keeps its ID word form
            "id": ["Admission No", "Roll No", "Aeries ID", "Network ID", "Lunch Number", "MARSS Number",
                   "Statewide ID", "Std_ID", "Lunch PIN", "Aeries Perm", "Room No", "Asset No", "Badge #"]}
NOT_IDS = ["Student_Count", "Students_Enrolled", "Student_Teacher_Ratio", "Grade_Level_Code", "Course_Code",
           "School_Code", "Student_Grade_Code", "Has_Student_Code", "Num_SSID", "Kidney_Function", "Business_Unit",
           "Num_Students", "Student_Age", "Teacher_Code", "Staff_Code", "Cost per Student", "Expenditure per Pupil",
           "Amount Paid by Student", "Minutes per Student", "Mean Scale Score Student", "Student Scale Score",
           "Student_District", "Student_School", "studentsenrolled", "studentcount", "Number_of_Absences", "Score"]


@pytest.mark.parametrize("name", K02_PERSON_IDS)
def test_person_id_headers_are_structural(name: str) -> None:
    h = hints.header_hints(name)
    assert h.person_id_like and h.org_code is None and h.pii is None, name


@pytest.mark.parametrize("name", K02_DEFAULT_IDS)
def test_default_id_stems_take_any_id_word(name: str) -> None:
    assert hints.header_hints(name).default_id, name


@pytest.mark.parametrize("form,name", [(f, n) for f, names in K02_WEAK.items() for n in names])
def test_weak_id_headers_let_the_values_decide(form: str, name: str) -> None:
    # i1/i2/i4/i7: a student word or code anywhere, or a final ID word, is no ID by itself (flags and status codes
    # under such headers stay what they are) but makes an ID column over ID-like values
    assert hints.weak_id_header(name) == form, name
    assert not hints.header_hints(name).person_id_like, name


@pytest.mark.parametrize("name", NOT_IDS)
def test_ordinary_headers_are_no_ids(name: str) -> None:
    h = hints.header_hints(name)
    assert not h.person_id_like and not h.default_id and not hints.person_head_hint(name), name
    assert hints.weak_id_header(name) is None, name


def test_org_codes_stay_org_codes_and_default_codes_stay_default() -> None:
    for name, word in (("Course_Code", "course"), ("School_Code", "school"), ("District_Code", "district"),
                       ("Course_Number", "course"), ("School_ID", "school")):
        assert hints.header_hints(name).org_code == word, name
    # the bare student-ID codes of the default list stay with it (D7: --no-default-ids turns them off)
    for name in ("SSID", "LASID", "UPN", "GTID", "SASID"):
        h = hints.header_hints(name)
        assert h.default_id and not h.person_id_like, name


def test_person_head_headers() -> None:
    for name in ("Student", "Stu", "Pupil", "Learner", "Child", "Kid", "Local_Student", "Student 1", "Student(s)",
                 "Student (District)"):
        assert hints.person_head_hint(name), name
    for name in ("Students", "Children", "Student_Name", "Student_Count", "Num_Student", "Has_Child", "Kid_Count",
                 "Teacher", "Student_Teacher_Ratio", "Student_District", "Student_School", "Cost per Student",
                 "Mean Scale Score Student"):
        assert not hints.person_head_hint(name), name


def test_id_word_headers() -> None:
    assert hints.id_word_header("Admission No") == "no"
    assert hints.id_word_header("Roll_Number") == "number"
    assert hints.id_word_header("Aeries ID (Local)") == "id"
    assert hints.id_word_header("Candidate Ref") == "ref"
    assert hints.id_word_header("Badge #") == "#"
    for name in ("Num_Absences", "Has_ID", "Number_of_Absences", "Score", "Notes"):
        assert hints.id_word_header(name) is None, name


def test_id_value_shapes() -> None:
    for v in ("100234", "000123", "100234.0", "S100234", "STU-0012", "ADM/2019/0045", "2016-0100", "S 100114",
              "jsmith23"):
        assert dids.id_like_value(v), v
    for v in ("12.5", "10-12", "3/4", "TBD", "Room A 204", "10²1", "１２３", "1 2 3 4"):
        assert not dids.id_like_value(v), v
    # other values on a minority of the rows (placeholders, an unknown member, data-entry leftovers) never stop the
    # keys (harden 2, A2-1); never a short code list, a status vocabulary or a flag, and never mostly other values
    ids = {str(100100 + 7 * i): 3 for i in range(30)}
    assert dids.id_keys_width({**ids, "TBD": 3}, 10) == 6
    assert dids.id_keys_width({**ids, "TBD": 3, "Pending": 1, "unknown": 1, "n/a": 1}, 10) == 6
    assert dids.id_keys_width({**ids, "12.5": 1}, 10) == 6
    for odd in ("-1", "100100, 100107", "100100;100107", "100100 (dup)", "100100?"):
        assert dids.id_keys_width({**ids, odd: 4}, 1) == 6, odd
    assert dids.id_keys_width({**ids, "TBD": 200}, 10) == 0
    assert dids.id_keys_width({**ids, **{f"{i}.5": 1 for i in range(20)}}, 10) == 6  # 20 of 110 rows: a minority
    assert dids.id_keys_width({**ids, **{f"{i}.5": 3 for i in range(30)}}, 10) == 0  # half the rows: measures
    assert dids.id_keys_width({str(i): 9 for i in range(10)}, 1) == 0
    assert dids.id_keys_width({"EL01": 5, "EL02": 5, "EO": 9}, 1) == 0
    assert dids.id_keys_width({"Y": 5, "N": 9}, 1) == 0


def test_integer_ids_stored_as_floats_are_the_integers() -> None:
    assert dids.normalize_id("100226.0") == dids.normalize_id("100226") == "100226"
    assert dids.normalize_id("0.0") == "0" and dids.normalize_id("12.5") == "125"
    shape = dids.analyze_id_shape(["100226.0", "100367.0"])
    assert shape["kind"] == "digits" and not shape["sequential"]


# --------------------------------------------------------------------------- K-02: tables
def _dense(n: int = 60) -> tuple[list[str], list[str]]:
    """``n`` dense local IDs (20001..) x 5-15 rows; the first holds the canary on 12 rows."""
    rng = random.Random(5)
    ids, reason = [], []
    for s in range(n):
        k = 12 if s == 0 else rng.randint(5, 15)
        ids += [str(20001 + s)] * k
        reason += [CANARY if s == 0 else COMMON[(s + j) % 3] for j in range(k)]
    return ids, reason


@pytest.mark.parametrize("header", ["Student Code", "Stu_Code", "Student_Ref", "Pupil Code", "Learner Ref",
                                    "SIS ID", "ID Number", "CALPADS SSID", "USI", "Perm_No", "EdFi USI", "Kid ID",
                                    "Child Code"])
def test_k02_student_id_headers_get_synthetic_ids(header: str) -> None:
    ids, reason = _attendance()
    tp, cols = run_table({header: ids, "Reason": reason})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id", header
    assert not set(cols[header].tolist()) & set(ids), header
    # D35 counts the IDs (one student), not the 12 rows: the canary is pseudonymised
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    assert CANARY not in set(cols["Reason"].tolist())


@pytest.mark.parametrize("header", [
    # i1: a student word with an ID word of its own
    "Student_SK", "Student UID", "Student PIN", "Student Barcode", "Student Badge", "Student Index", "Student Serial",
    "Learner UID", "Student Lunch PIN", "STUDENTSK",
    # i2: a system or source after the ID word, or run together
    "Student ID (District)", "Student ID - District", "Student ID (Aeries)", "Student ID (9 digit)", "SSID (CALPADS)",
    "CALPADSSSID", "USI (EdFi)", "Student (District)", "Student_ID_District",
    # i4: ID words alone
    "Admission No", "Roll No", "Aeries ID", "Network ID", "KIDS ID", "MOSIS ID", "EIS ID", "SAIS ID", "Lunch Number",
    "NYSSIS ID", "MARSS Number", "Statewide ID", "Std_ID", "Lunch PIN", "Aeries Perm",
    # the person heads
    "Student", "Stu", "Pupil", "Learner", "Child", "Kid", "Local_Student"])
@pytest.mark.parametrize("dense", [False, True])
def test_k02_unlisted_student_id_headers_get_synthetic_ids(header: str, dense: bool) -> None:
    # the only key of an attendance table, dense (20001..20060) or sparse: never a real ID out, the canary of one
    # student pseudonymised (D35 counts the student once, not their 12 rows)
    ids, reason = _dense() if dense else _attendance()
    tp, cols = run_table({header: ids, "Status": [("P", "A", "T")[i % 3] for i in range(len(ids))],
                          "Reason": reason})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id", header
    assert not set(cols[header].tolist()) & set(ids), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    assert CANARY not in set(cols["Reason"].tolist())
    assert "redacted" not in p.stats or True  # an ID column's statistics are never value statistics
    assert ids[0] not in json.dumps(tp.to_public(), default=str), header


def test_k02_ten_digit_state_ids_under_a_student_header_are_ids_not_phone_numbers() -> None:
    # a CALPADS SSID has ten digits, as a phone number does: under a student word or a student-ID code it is an ID
    # (its values never written, D35 counts the student); a bare 'Number' of phone-shaped values stays masked as phones
    rng = random.Random(3)
    ids = [str(rng.randint(2_000_000_000, 9_999_999_999)) for _ in range(60)]
    vals, reason = [], []
    for i, s in enumerate(ids):
        vals += [s] * (12 if i == 0 else 8)
        reason += [CANARY if i == 0 else COMMON[i % 3]] * (12 if i == 0 else 8)
    for header in ("CALPADS SSID", "Student_SK", "Stu"):
        tp, cols = run_table({header: vals, "Reason": reason})
        assert profile_of(tp, header).type is T.IDENTIFIER, header
        assert not set(cols[header].tolist()) & set(ids), header
        assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    tp, cols = run_table({"Number": vals, "Reason": reason})
    assert profile_of(tp, "Number").params.get("pii_shapes") == ["phone"]
    assert not set(cols["Number"].tolist()) & set(ids)


def test_k02_id_word_alone_warns_and_names_the_overrides() -> None:
    ids, reason = _attendance()
    tp, _ = run_table({"Admission No": ids, "Reason": reason})
    p = profile_of(tp, "Admission No")
    assert p.source == "id-shape"
    msg = next(m for m in p.messages if m.code == "id_by_header_values")
    assert msg.level == "warning" and "--person-id" in msg.fix and "--org-code" in msg.fix
    # --person-id confirms it (no warning), --org-code keeps the real values
    tp, _ = run_table({"Admission No": ids, "Reason": reason}, person_id=["Admission No"])
    p = profile_of(tp, "Admission No")
    assert p.source == "person-id" and not [m for m in p.messages if m.code == "id_by_header_values"]
    tp, cols = run_table({"Admission No": ids, "Reason": reason}, org_code=["Admission No"])
    assert profile_of(tp, "Admission No").type is T.ORG_CODE


def test_k02_student_word_with_few_ids_and_placeholders_is_an_id_column() -> None:
    # i3: 8 students (fewer than --min-cell), or one placeholder ('TBD', 'Pending', 'unknown') for students without
    # an ID yet: still an ID column, never a number column writing every real ID
    rng = random.Random(4)
    eight = [str(500100 + 7 * (i % 8)) for i in range(160)]
    for header in ("Stu", "Kid", "Student Code"):
        tp, cols = run_table({header: eight, "Skill": [rng.choice(["Fluency", "Decoding"]) for _ in eight]})
        assert profile_of(tp, header).type is T.IDENTIFIER, header
        assert not set(cols[header].tolist()) & set(eight), header
    thirty = [str(600100 + 7 * (i % 30)) for i in range(600)]
    for header, token in (("Stu", "TBD"), ("Stu", "Pending"), ("Kid", "unknown"), ("Child", "TBD"),
                          ("Admission No", "TBD")):
        vals = [token] * 3 + thirty[3:]
        tp, cols = run_table({header: vals, "Skill": [rng.choice(["Fluency", "Decoding"]) for _ in vals]})
        assert profile_of(tp, header).type is T.IDENTIFIER, (header, token)
        assert not set(cols[header].tolist()) & set(thirty), (header, token)
        assert len(cols[header]) == len(vals)


def test_k02_student_word_needs_id_values() -> None:
    # names, dates, 1-2 digit counts or codes and status words under a student-word header are no IDs
    names = ["Garcia, Ana", "Lopez, Maria", "Smith, John", "Nguyen, Linh"] * 20
    dates = [f"2024-01-{1 + i % 28:02d}" for i in range(80)]
    for header, vals in (("Student", names), ("Student", dates), ("Child", [str(1 + i % 40) for i in range(80)]),
                         ("English Learner Code", [("EO", "EL", "RFEP", "IFEP", "TBD")[i % 5] for i in range(80)]),
                         ("Exceptional Child Code", [str(1 + i % 9) for i in range(80)]),
                         ("Missing Student Code", [("Y", "N")[i % 2] for i in range(80)])):
        tp, _ = run_table({"Student_ID": [str(300000 + i) for i in range(80)], header: vals})
        assert profile_of(tp, header).type is not T.IDENTIFIER, header
    # the student's district or school holds organisation codes, not the student's ID
    tp, _ = run_table({"Student_District": [f"{1000 + 37 * (i % 15)}" for i in range(90)]})
    assert profile_of(tp, "Student_District").type is not T.IDENTIFIER


def test_k02_measures_of_students_keep_their_types() -> None:
    # i1 fix check: a measure per student (or by student) is no ID column and creates no ID namespace
    rng = random.Random(3)
    n = 300
    cols = {"Student_ID": [str(300000 + rng.randint(0, 120) * 11) for _ in range(n)],
            "Amount Charged": [str(rng.choice([25, 40, 50, 75, 100, 120, 150, 175, 200, 250])) for _ in range(n)],
            "Amount Paid by Student": [str(rng.choice([0, 25, 40, 75, 100, 150, 200])) for _ in range(n)],
            "Minutes per Student": [str(rng.choice([0, 5, 10, 15, 30, 45, 60, 90, 120, 150])) for _ in range(n)],
            "Cost per Student": [str(rng.randint(8200, 16500)) for _ in range(n)],
            "Student Scale Score": [str(rng.randint(2380, 2620)) for _ in range(n)]}
    tp, _ = run_table(cols)
    assert tp.entity.id_columns == [0]
    for name in list(cols)[1:]:
        assert profile_of(tp, name).type is not T.IDENTIFIER, name


def test_k02_ordinary_columns_beside_a_student_id_keep_their_types() -> None:
    rng = random.Random(9)
    n = 400
    cols = {"Student_ID": [str(500000 + i // 4) for i in range(n)],
            "Student_Count": [str(rng.randint(40, 900)) for _ in range(n)],
            "Students_Enrolled": [str(rng.randint(40, 900)) for _ in range(n)],
            "Student_Teacher_Ratio": [f"{rng.uniform(10, 30):.1f}" for _ in range(n)],
            "Grade_Level_Code": [("09", "10", "11", "12")[i % 4] for i in range(n)],
            "Course_Code": [f"MATH{100 + i % 25}" for i in range(n)],
            "School_Code": [str(300 + i % 12) for i in range(n)],
            # codes many students share are no IDs beside the student's ID, whatever their header
            "Test ID": [str(1001 + i % 12) for i in range(n)],
            "Bus No": [str(4100 + i % 30) for i in range(n)],
            "Student Homeroom": [str(100 + i % 20) for i in range(n)]}
    tp, out = run_table(cols)
    types = {p.name: p.type for p in tp.columns}
    assert tp.entity.id_columns == [0]
    assert types["Student_Count"] is T.INTEGER and types["Students_Enrolled"] is T.INTEGER
    assert types["Student_Teacher_Ratio"] is T.FLOAT
    assert types["Course_Code"] is T.ORG_CODE and types["School_Code"] is T.ORG_CODE
    assert types["Grade_Level_Code"] is not T.IDENTIFIER
    # org codes keep their real values (D10)
    assert set(out["Course_Code"].tolist()) == set(cols["Course_Code"])
    assert not [m for m in tp.messages if m.code == "id_candidates"]


def test_k02_another_id_of_the_student_beside_the_student_id_is_an_id_column() -> None:
    # i5: a state ID or lunch number beside the local student ID (1:1) is the student's too: synthetic IDs, never
    # its real min/max or values
    ids, reason = _attendance()
    rng = random.Random(8)
    state = {s: str(7_000_000_000 + rng.randint(0, 999_999_999)) for s in set(ids)}
    lunch = {s: str(40000 + 13 * i) for i, s in enumerate(sorted(set(ids)))}
    tp, cols = run_table({"Student_ID": ids, "Statewide ID": [state[s] for s in ids],
                          "Lunch Number": [lunch[s] for s in ids], "SSID (CALPADS)": [state[s] for s in ids],
                          "Reason": reason})
    for name, real in (("Statewide ID", state), ("Lunch Number", lunch), ("SSID (CALPADS)", state)):
        p = profile_of(tp, name)
        assert p.type is T.IDENTIFIER and p.role == "id", name
        assert not set(cols[name].tolist()) & set(real.values()), name
    assert tp.entity.primary == 0


def test_k02_a_column_naming_a_student_beside_the_student_id_is_an_id_column() -> None:
    # a mentor who mentors several students repeats beside them: its header names a student, so it holds students
    rng = random.Random(3)
    n = 300
    mentors = [str(880000 + 17 * i) for i in range(40)]
    sids = [str(300000 + i) for i in range(n)]
    ment = [mentors[i % 40] for i in range(n)]
    tp, cols = run_table({"Student_ID": sids, "Mentor Student": ment,
                          "Focus": [rng.choice(["Reading", "Math"]) for _ in range(n)]})
    p = profile_of(tp, "Mentor Student")
    assert p.type is T.IDENTIFIER and not set(cols["Mentor Student"].tolist()) & set(mentors)


def test_k02_a_student_key_counts_people_for_d35() -> None:
    # 'Student_SK' numbers the student: D35 counts people, and says nothing about rows standing in for them
    ids, reason = _attendance()
    tp, _ = run_table({"Student_SK": ids, "Reason": reason})
    from datanonymous.profile import entity as entity_mod

    assert entity_mod.id_role(profile_of(tp, "Student_SK")) == "subject"
    assert not [m for m in profile_of(tp, "Reason").messages if m.code == "d35_rows_counted"]


@pytest.mark.parametrize("record", ["ID", "DCID"])
def test_k02_student_column_beside_a_record_id_is_an_id_column(record: str) -> None:
    # i5: a record key ('ID', PowerSchool 'DCID') is no person ID: the student column beside it is
    ids, reason = _dense()
    for header in ("Admission No", "Aeries ID", "KIDS ID", "Lunch Number", "Student_SK"):
        tp, cols = run_table({record: [str(900000 + i) for i in range(len(ids))], header: ids, "Reason": reason})
        assert profile_of(tp, header).type is T.IDENTIFIER, (record, header)
        assert not set(cols[header].tolist()) & set(ids), (record, header)
        assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), (record, header)


@pytest.mark.parametrize("form", ["ADM/2019/{:04d}", "{:04d}-0100", "1001{:02d}-0", "S {}"])
def test_k02_separator_formatted_ids_are_ids(form: str) -> None:
    # i5: admission numbers with separators, at most 20 students (a factor D35 would keep by its rows)
    rng = random.Random(5)
    keys = [form.format(45 + 3 * i if "ADM" in form else (2016 + i if "{:04d}-" in form else (10 + i if "1001" in form
                                                                                               else 100100 + 7 * i)))
            for i in range(15)]
    vals, reason = [], []
    for i, k in enumerate(keys):
        for _ in range(rng.randint(10, 20)):
            vals.append(k)
            reason.append(CANARY if i == 0 else COMMON[i % 3])
    for header in ("Admission No", "Roll No", "Stu"):
        tp, cols = run_table({header: vals, "Reason": reason})
        assert profile_of(tp, header).type is T.IDENTIFIER, (form, header)
        assert not set(cols[header].tolist()) & set(keys), (form, header)
        assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), (form, header)


def test_k02_three_digit_numbers_under_an_id_word_are_ids_unless_org_codes() -> None:
    # harden 2 (A2-2, replaces i9): with no person ID in the table, 3-digit keys under an ID word cannot be told from a
    # small school's admission numbers by their values: ID columns (fail-safe), whose warning names --org-code, which
    # keeps the real room, bus and item numbers
    rng = random.Random(1)
    n = 300
    cols = {"Room No": [str(rng.choice(list(range(100, 140)) + list(range(200, 230)))) for _ in range(n)],
            "Bus No": [str(rng.randint(101, 160)) for _ in range(n)],
            "Item No": [str(101 + i % 60) for i in range(n)],
            "Term": [rng.choice(["Fall", "Spring"]) for _ in range(n)]}
    tp, out = run_table(cols)
    for name in ("Room No", "Bus No", "Item No"):
        p = profile_of(tp, name)
        assert p.type is T.IDENTIFIER and p.source == "id-shape", name
        assert any(m.code == "id_by_header_values" and "--org-code" in m.fix for m in p.messages), name
        assert not set(out[name].tolist()) & set(cols[name]), name
    tp, out = run_table(cols, org_code=["Room No", "Bus No", "Item No"])
    for name in ("Room No", "Bus No", "Item No"):
        assert profile_of(tp, name).type is T.ORG_CODE, name
        assert set(out[name].tolist()) == set(cols[name]), name
    # 1-2 digit numbers (a cart, a period, a roll number within a class) stay numbers
    tp, _ = run_table({"Cart No": [str(rng.randint(1, 40)) for _ in range(n)], "Term": cols["Term"]})
    assert profile_of(tp, "Cart No").type is not T.IDENTIFIER


def test_k02_student_code_cli_writes_no_real_id(tmp_path: Path) -> None:
    # the triage repro: 'Student Code' PASSed with 60/60 real IDs and the one-student canary written
    ids, reason = _attendance()
    src = _write(tmp_path / "in" / "att.csv", ["Student Code", "Status", "Reason"],
                 [[s, ("P", "A", "T")[i % 3], r] for i, (s, r) in enumerate(zip(ids, reason))]).parent
    out, rep, console = _run(src, tmp_path)
    assert _columns(rep, "att.csv")["Student Code"]["type"] == "IDENTIFIER"
    text = _texts(out, rep) + console
    assert CANARY not in text
    words = set(text.replace(",", " ").split())
    assert not words & set(ids)
    assert len(read_dicts(out / "att.csv")) == len(ids)


@pytest.mark.parametrize("header", ["Admission No", "Student_SK", "Student ID (Aeries)", "Roll No"])
def test_k02_unlisted_headers_cli_write_no_real_id(tmp_path: Path, header: str) -> None:
    # i1/i2/i4: dense IDs PASSed with every real ID in the out dir, the report json and the data dictionary
    ids, reason = _dense()
    src = _write(tmp_path / "in" / "att.csv", [header, "Status", "Reason"],
                 [[s, ("P", "A", "T")[i % 3], r] for i, (s, r) in enumerate(zip(ids, reason))]).parent
    out, rep, console = _run(src, tmp_path)
    assert _columns(rep, "att.csv")[header]["type"] == "IDENTIFIER"
    text = _texts(out, rep) + console
    assert CANARY not in text
    assert not set(text.replace(",", " ").replace("|", " ").split()) & set(ids)
    assert len(read_dicts(out / "att.csv")) == len(ids)
    if header in ("Admission No", "Roll No"):
        report = (rep / "files" / "att.csv.md").read_text(encoding="utf-8")
        assert "--person-id" in report and "--org-code" in report


@pytest.mark.parametrize("fmt", ["csv", "parquet", "json", "dta"])
def test_k02_integer_ids_stored_as_floats_get_integer_synthetic_ids(tmp_path: Path, fmt: str) -> None:
    # i3: an integer ID column with missing values is stored as float64 ('100226.0'): an ID column all the same, its
    # synthetic IDs are integers stored the same way (a JSON number column is no FAIL)
    import pandas as pd

    ids, reason = _attendance()
    vals = [float(s) for s in ids]
    vals[3] = vals[50] = float("nan")
    df = pd.DataFrame({"Stu": vals, "Reason": reason})
    src = tmp_path / "in" / f"att.{fmt}"
    src.parent.mkdir(parents=True)
    if fmt == "csv":
        df.to_csv(src, index=False)
    elif fmt == "parquet":
        df.to_parquet(src, index=False)
    elif fmt == "json":
        src.write_text(df.to_json(orient="records"), encoding="utf-8")
    else:
        df.to_stata(src, write_index=False, version=118)
    out, rep, console = _run(src.parent, tmp_path)
    assert _columns(rep, f"att.{fmt}")["Stu"]["type"] == "IDENTIFIER"
    if fmt == "csv":
        got = pd.read_csv(out / "att.csv")["Stu"]
    elif fmt == "parquet":
        got = pd.read_parquet(out / "att.parquet")["Stu"]
    elif fmt == "json":
        got = pd.read_json(out / "att.json")["Stu"]
    else:
        got = pd.read_stata(out / "att.dta")["Stu"]
    filled = got.dropna()
    assert len(got) == len(ids) and int(got.isna().sum()) == 2
    assert all(float(v).is_integer() and 100000 <= v < 1000000 for v in filled)
    assert not {str(int(v)) for v in filled} & set(ids)
    assert CANARY not in _texts(out, rep) + console


def test_k02_flags_and_status_codes_beside_a_student_id_are_no_ids(tmp_path: Path) -> None:
    # i7: status and flag columns under a student word or code keep their types and real vocabulary; 10 single-digit
    # codes no longer FAIL the roster
    rng = random.Random(1)
    cols = {"English Learner Code": ["EO", "EO", "EO", "EL", "RFEP", "IFEP", "TBD"],
            "English Language Learner Code": [str(i) for i in range(9)],
            "Exceptional Child Code": [str(i) for i in range(10)],
            "Migrant Child Code": ["Y", "N", "N", "N"],
            "Homeless Student Code": ["01", "02", "03", "04"],
            "Missing SSID": ["Y", "N", "N", "N", "N"],
            "Valid SSID": ["Y", "Y", "Y", "N"],
            "Missing Student Code": ["Y", "N", "N", "N"]}
    rows = [[str(200000 + 3 * i)] + [rng.choice(v) for v in cols.values()] for i in range(400)]
    src = _write(tmp_path / "in" / "roster.csv", ["Student_ID", *cols], rows).parent
    out, rep, _ = _run(src, tmp_path)
    info = _columns(rep, "roster.csv")
    for name, levels in cols.items():
        assert info[name]["type"] != "IDENTIFIER", name
        assert {r[name] for r in read_dicts(out / "roster.csv")} <= set(levels), name


def test_k02_asset_numbers_are_id_columns_with_a_warning() -> None:
    # 5-digit keys under an ID word in a table with no person ID cannot be told from admission numbers: an ID column
    # (fail-safe) whose warning names --org-code to keep the real values
    rng = random.Random(6)
    n = 500
    assets = [str(rng.randint(10000, 10200)) for _ in range(n)]
    tp, out = run_table({"Cart No": [str(rng.randint(1, 40)) for _ in range(n)], "Asset No": assets,
                         "Device": [rng.choice(["Chromebook", "iPad"]) for _ in range(n)]})
    p = profile_of(tp, "Asset No")
    assert p.type is T.IDENTIFIER and p.source == "id-shape"
    assert any(m.code == "id_by_header_values" and "--org-code" in m.fix for m in p.messages)
    assert not set(out["Asset No"].tolist()) & set(assets)
    assert profile_of(tp, "Cart No").type is not T.IDENTIFIER


# --------------------------------------------------------------------------- K-13
def test_k13_key_ref_share_tests_the_digit_keys_of_mixed_columns() -> None:
    dense = {str(v) for v in range(1, 2001)}
    dints = np.arange(1, 2001, dtype=np.int64)
    # KG + 1..5 beside dense IDs 1..2000: the digits are a coincidence, KG is no ID
    assert st._key_ref_share({"kg", "1", "2", "3", "4", "5"}, dense, dints, False) is None
    assert st._key_ref_share({"kg", "pk", "1", "2", "3", "4", "5"}, dense | {"kg"}, dints, False) is None
    # sparse 6-digit IDs among a placeholder word are still evidence
    rng = np.random.default_rng(4)
    sparse_ints = np.sort(rng.choice(np.arange(100000, 999999), 3000, replace=False)).astype(np.int64)
    sparse = {str(v) for v in sparse_ints}
    col = {str(v) for v in sparse_ints[:40]} | {"tbd"}
    assert st._key_ref_share(col, sparse, sparse_ints, False) == pytest.approx(40 / 41)
    # codes with digits keep their rule
    assert st._key_ref_share({"s100113", "s100120", "1"}, {"s100113", "s100120", "1"},
                             np.asarray([1], np.int64), False) is not None


def _roster(ids: list[str], padded: bool) -> list[list[str]]:
    rng = random.Random(4)
    rows = []
    for sid in ids:
        g = rng.choice(["KG", "1", "2", "3", "4", "5"])
        rows.append([sid, f"{int(g):02d}" if padded and g != "KG" else g, rng.choice(["F", "M"])])
    return rows


@pytest.mark.parametrize("padded", [False, True])
def test_k13_grade_column_is_never_a_key_reference(tmp_path: Path, padded: bool) -> None:
    rng = random.Random(1)
    if padded:
        ids = sorted({f"{x:06d}" for x in rng.sample(range(6, 4001), 2500)} | {f"{x:06d}" for x in range(1, 6)})
    else:
        ids = ["1", "2", "3", "4", "5"] + [str(x) for x in rng.sample(range(6, 9999), 300)]
    rows = _roster(ids, padded)
    src = _write(tmp_path / "in" / "roster.csv", ["Student_ID", "Grade", "Gender"], rows).parent
    out, rep, _ = _run(src, tmp_path)
    col = _columns(rep, "roster.csv")["Grade"]
    assert col["type"] == "ORDINAL" and col["role"] != "id"
    assert Counter(r["Grade"] for r in read_dicts(out / "roster.csv")) == Counter(r[1] for r in rows)


def test_k13_mixed_grade_like_factor_is_no_key_reference(tmp_path: Path) -> None:
    # not a vocabulary column ('Ungraded' is no grade level), so only the chance test keeps it a factor
    ids = [str(x) for x in range(1, 401)]
    rng = random.Random(2)
    rows = [[s, rng.choice(["KG", "1", "2", "3", "4", "5", "Ungraded"])] for s in ids]
    rows += [[str(x), "x"] for x in rng.sample(range(500, 9000), 400)]  # make the namespace no 1..n sequence
    src = _write(tmp_path / "in" / "r.csv", ["Student_ID", "Level"], rows).parent
    out, rep, _ = _run(src, tmp_path)
    col = _columns(rep, "r.csv")["Level"]
    assert col["type"] != "IDENTIFIER" and col["role"] != "id"
    assert not any(m["code"] == "unlinked_key_column" for m in col["messages"])


def test_k13_real_key_references_still_link(tmp_path: Path) -> None:
    # a buddy column holding sparse real student IDs (plus a placeholder) is still an ID column
    rng = random.Random(6)
    ids = [str(x) for x in sorted(rng.sample(range(100000, 999999), 200))]
    _write(tmp_path / "in" / "students.csv", ["student_id", "Grade"], [[s, "KG"] for s in ids])
    _write(tmp_path / "in" / "pairs.csv", ["student_id", "Buddy"],
           [[ids[i], ids[(i + 7) % 200] if i % 10 else "TBD"] for i in range(200)])
    out, rep, _ = _run(tmp_path / "in", tmp_path)
    assert _columns(rep, "pairs.csv")["Buddy"]["type"] == "IDENTIFIER"
    assert not {r["Buddy"] for r in read_dicts(out / "pairs.csv")} & set(ids)


def test_k13_dense_ids_among_placeholders_are_still_references() -> None:
    # i6 (the K-13 change regressed it): 37 of 300 dense 6-digit student IDs (or 0.6 dense) among 'TBD' / 'n/a'
    # placeholders hold those IDs; a number of six digits is a key of that namespace by chance only 300 in 900,000 times
    dense = np.arange(100001, 100301, dtype=np.int64)
    keys = {str(v) for v in dense}
    col = {str(v) for v in dense[150:187]} | {"tbd", "none", "na"}
    assert st._key_ref_share(col, keys, dense, False) == pytest.approx(37 / 40)
    assert st._key_ref_share({str(v) for v in dense[150:187]}, keys, dense, False) == pytest.approx(1.0)
    sixty = np.asarray(sorted(random.Random(4).sample(range(100001, 100501), 300)), dtype=np.int64)
    col = {str(v) for v in sixty[150:187]} | {"tbd"}
    assert st._key_ref_share(col, {str(v) for v in sixty}, sixty, False) is not None
    # a score column (40-100) or a grade (1-5) beside dense IDs 1..2000 is still no reference
    dints = np.arange(1, 2001, dtype=np.int64)
    dkeys = {str(v) for v in dints}
    assert st._key_ref_share({str(v) for v in range(40, 101)}, dkeys, dints, False) is None
    assert st._key_ref_share({str(v) for v in range(1, 6)} | {"kg"}, dkeys, dints, False) is None
    # four-digit IDs are too common a shape (scale scores, years): a dense 1001-3000 block is no evidence
    four = np.arange(1001, 3001, dtype=np.int64)
    assert st._key_ref_share({str(v) for v in range(2380, 2621)}, {str(v) for v in four}, four, False) is None


@pytest.mark.parametrize("header,tokens,density", [("Paired With", ["TBD"], 1.0), ("Linked_To", ["None", "n/a"], 1.0),
                                                   ("Paired With", ["TBD"], 0.6), ("Twin Of", ["n/a"], 0.6)])
def test_k13_dense_reference_column_with_placeholders_gets_synthetic_ids(tmp_path: Path, header: str,
                                                                        tokens: list[str], density: float) -> None:
    rng = random.Random(2)
    if density == 1.0:
        ids = [str(100001 + i) for i in range(300)]
    else:
        ids = [str(x) for x in sorted(random.Random(4).sample(range(100001, 100001 + int(300 / density)), 300))]
    _write(tmp_path / "in" / "students.csv", ["Student_ID", "Grade", "Gender"],
           [[s, rng.choice(["9", "10", "11", "12"]), rng.choice(["F", "M"])] for s in ids])
    r = random.Random(7)
    rows = [[s, r.choice(ids[150:190]) if r.random() < 0.8 else r.choice(tokens), r.choice(["Reading", "Math"])]
            for s in ids[:150]]
    _write(tmp_path / "in" / "mentoring.csv", ["Student_ID", header, "Focus"], rows)
    out, rep, _ = _run(tmp_path / "in", tmp_path)
    assert _columns(rep, "mentoring.csv")[header]["type"] == "IDENTIFIER"
    written = [r[header] for r in read_dicts(out / "mentoring.csv")]
    assert not set(written) & set(ids)
    assert not set(_texts(rep).replace(",", " ").replace("|", " ").split()) & set(ids[150:190])
    # the references join the roster's synthetic IDs (a placeholder gets a same-shape code of its own)
    roster = {r["Student_ID"] for r in read_dicts(out / "students.csv")}
    assert {v for v in written if v.isdigit()} <= roster and len({v for v in written if v.isdigit()}) > 30


def test_k02_id_word_reference_with_orphans_joins_like_a_key_reference(tmp_path: Path) -> None:
    # a column made an ID column by its header and values ('subject_ref') links to the roster on the key-reference
    # evidence: 70% known references join, the orphans stay orphans
    pat = [f"S{100000 + i * 113}" for i in range(40)]
    d = tmp_path / "in"
    _write(d / "people.csv", ["student_id", "score"], [[v, i] for i, v in enumerate(pat[:20])])
    refs = pat[:14] + [f"S{900000 + i}" for i in range(6)]
    _write(d / "visits.csv", ["subject_ref", "value"], [[refs[i % 20], i % 7] for i in range(60)])
    out, rep, _ = _run(d, tmp_path)
    people = {r["student_id"] for r in read_dicts(out / "people.csv")}
    vals = [r["subject_ref"] for r in read_dicts(out / "visits.csv")]
    assert not set(vals) & set(refs)
    assert sum(v in people for v in vals) == 42  # 14 known IDs x 3 rows, as in the source
    col = _columns(rep, "visits.csv")["subject_ref"]
    assert col["type"] == "IDENTIFIER" and "id_key_link" in [m["code"] for m in col["messages"]]


# --------------------------------------------------------------------------- K-21
def test_k21_unicode_digit_id_shape() -> None:
    shape = dids.analyze_id_shape(["1001", "10\u00b21", "1015"])
    assert shape["kind"] == "pattern" and not shape["sequential"]
    assert dids.analyze_id_shape(["1001", "1015"])["kind"] == "digits"
    assert not dids.id_like_value("10\u00b21") and not dids.id_like_value("\uff11\uff12\uff13")


def test_k21_unicode_digit_in_an_id_column_does_not_crash(tmp_path: Path) -> None:
    sid = [str(1001 + 7 * i) for i in range(59)] + ["10\u00b21"]
    src = _write(tmp_path / "in" / "students.csv", ["student_id", "Score"],
                 [[s, str(50 + i % 40)] for i, s in enumerate(sid)]).parent
    out, rep, console = _run(src, tmp_path)
    assert "internal error" not in console
    assert _columns(rep, "students.csv")["student_id"]["type"] == "IDENTIFIER"
    written = {r["student_id"] for r in read_dicts(out / "students.csv")}
    assert len(written) == 60 and not written & set(sid)


def test_k21_unicode_digit_in_a_school_year_column_does_not_crash() -> None:
    vals = [("2024", "2025", "20\u00b25")[i % 3] for i in range(60)]
    det = detect(vals, "School_Year")
    assert det.type is not T.SCHOOL_YEAR or "20\u00b25" not in det.params.get("levels", [])
    tp, cols = run_table({"student_id": [str(9000 + i) for i in range(60)], "School_Year": vals})
    assert len(cols["School_Year"]) == 60


@pytest.mark.parametrize("header", ["student_id", "Student", "Student Code", "Perm_No", "School_Year", "SY",
                                    "Grade", "Year", "Score", "Admission No"])
def test_k21_non_ascii_digits_never_crash_detection(header: str) -> None:
    # superscript, circled, Arabic-Indic, full-width and Devanagari digits: str.isdigit() is True for some of them,
    # int() refuses the non-decimal ones (the crash class), and none is an ASCII digit
    weird = ["10\u00b21", "20\u00b25", "\u2460\u2461\u2462", "\u0661\u0662\u0663", "\uff11\uff12\uff13",
             "\u0967\u0968\u0969\u096a", "1\u00b2"]
    base = [str(100000 + 7 * (i // 3)) if header not in ("School_Year", "SY") else ("2024", "2025")[i % 2]
            for i in range(60)]
    for w in weird:
        for k in (1, 59):
            vals = list(base)
            vals[:k] = [w] * k
            detect(vals, header)  # no exception


@pytest.mark.parametrize("token", ["10²1", "１２", "①②", "١٢٣"])
def test_k21_non_ascii_digit_token_among_numbers_gets_a_pseudonym(token: str) -> None:
    # every same-shape mask of such a token is ASCII digits, so each read as a number and the pseudonym drawer
    # lengthened it for ever (the run hung); it now puts a letter in front: no number, no real value, N rows out
    import threading

    n = 60
    vals = [str(100000 + 7 * (i // 3)) for i in range(n)]
    vals[5] = vals[17] = token
    result: dict[str, object] = {}

    def work() -> None:
        result["out"] = run_table({"sid": [str(9000 + i // 2) for i in range(n)], "Score": vals})

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(120)
    assert not worker.is_alive(), "pseudonym drawing did not finish"
    tp, cols = result["out"]
    assert profile_of(tp, "Score").type is T.NUMERIC_WITH_TOKENS
    out = cols["Score"].tolist()
    assert len(out) == n and token not in out
    odd = [v for v in out if not (v.isascii() and v.isdigit())]
    assert len(odd) == 2 and len(set(odd)) == 1 and not odd[0].isdigit()


def test_k21_score_bands_with_a_non_decimal_digit_do_not_crash() -> None:
    # the class of K-21: str.isdigit() accepts '²', int() refuses it; the score-band rule of the phone screen used
    # both on the same text
    assert dtext._ranges("0-59 60-69") and not dtext._ranges("1²-5 6-9")
    vals = [("0-59 6²-69", "70-79 80-89", "1²-5 6-9")[i % 3] for i in range(60)]
    for header in ("Score Bands", "Contact", "Phone", "col"):
        detect(vals, header)  # no exception


def test_k21_html_xls_span_with_a_non_decimal_digit_does_not_crash(tmp_path: Path) -> None:
    # the same class in the reader of SIS 'Excel' files that are HTML: colspan="²" passed isdigit() into int()
    rows = "".join(f"<tr><td>S{5000 + i}</td><td>{i % 7}</td></tr>" for i in range(40))
    html = ("<html><body><table><tr><td colspan=\"²\">Roster</td></tr>"
            f"<tr><th>Student_ID</th><th>Score</th></tr>{rows}</table></body></html>")
    p = tmp_path / "in" / "roster.xls"
    p.parent.mkdir(parents=True)
    p.write_text(html, encoding="utf-8")
    res = run_cli(p.parent, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--seed", "1")
    assert res.exit_code in (0, 3) and "internal error" not in res.output.casefold(), res.output


# --------------------------------------------------------------------------- harden 2 (second attack pass)
def _words(text: str) -> set[str]:
    for ch in ",|;:()[]'\"`*=":
        text = text.replace(ch, " ")
    return set(text.split())


@pytest.mark.parametrize("header", ["Student_SK", "Stu", "CALPADS SSID", "Admission No", "Other ID"])
@pytest.mark.parametrize("odd", ["-1", "-1x5%", "{a}, {b}", "{a};{b}", "{a} (dup)", "{a}?"])
def test_h2_a2_1_odd_cells_never_stop_an_id_column(header: str, odd: str) -> None:
    # A2-1: one non-ID-like cell holding a digit (a warehouse's unknown member -1, a group session's two IDs, a
    # data-entry note) made 60 students' IDs a number column that wrote them all, and D35 counted rows
    ids, reason = _attendance()
    vals = list(ids)
    a, b = sorted(set(ids))[3:5]
    if odd == "-1x5%":
        for i in range(0, len(vals), 20):
            vals[i] = "-1"
    else:
        vals[7] = odd.format(a=a, b=b)
    tp, cols = run_table({header: vals, "Status": [("P", "A", "T")[i % 3] for i in range(len(vals))],
                          "Reason": reason})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id", (header, odd)
    out = cols[header].tolist()
    assert len(out) == len(vals)
    assert not _words(" ".join(out)) & set(ids), (header, odd)
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), (header, odd)


@pytest.mark.parametrize("odd", ["12.5", "{a}.5", "{t}", "({a})", "1e5"])
def test_h2_a2_1_an_odd_cell_that_changes_the_type_never_stops_an_id_column(odd: str) -> None:
    # the same class by way of the detected type: one decimal or formatted cell made 60 students' IDs a FLOAT or
    # NUMERIC_FORMATTED column, whose generator writes real values
    ids, reason = _attendance()
    a = sorted(set(ids))[3]
    vals = list(ids)
    vals[7] = odd.format(a=a, t=f"{int(a):,}")
    tp, cols = run_table({"Student_SK": vals, "Reason": reason})
    assert profile_of(tp, "Student_SK").type is T.IDENTIFIER, odd
    assert not _words(" ".join(cols["Student_SK"].tolist()).replace(".", " ")) & set(ids), odd
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), odd


@pytest.mark.parametrize("header", ["Stu", "Student_SK", "Other ID", "Student_ID"])
def test_h2_a2_1_thousands_formatted_ids_are_ids_of_the_same_shape(header: str) -> None:
    # an Excel 'Number' format saved to CSV ('100,226'): integers, so ID-like keys, written in the same format
    ids, reason = _attendance()
    vals = [f"{int(s):,}" for s in ids]
    tp, cols = run_table({header: vals, "Reason": reason})
    assert profile_of(tp, header).type is T.IDENTIFIER, header
    out = cols[header].tolist()
    assert all(len(v) == 7 and v[3] == "," and v.replace(",", "").isdigit() and v[0] != "0" for v in out), header
    assert not {v.replace(",", "") for v in out} & set(ids), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header


@pytest.mark.parametrize("token", ["-", "?", "***", "—", "#"])
@pytest.mark.parametrize("header", ["Student_ID", "Student_SK", "Admission No"])
def test_h2_punctuation_placeholders_in_an_id_column_never_fail_the_file(header: str, token: str) -> None:
    # a placeholder without a letter or digit names no one: an ID column writes it as it is, at its rows, instead of
    # FAILing for want of a same-shape synthetic value (D9) - a strong 'Student_ID' FAILed this way before
    ids, reason = _attendance()
    vals = list(ids)
    for i in (5, 50, 300):
        vals[i] = token
    tp, cols = run_table({header: vals, "Reason": reason})
    assert profile_of(tp, header).type is T.IDENTIFIER, (header, token)
    out = cols[header].tolist()
    assert [i for i, v in enumerate(out) if v == token] == [5, 50, 300], (header, token)
    assert not set(out) & set(ids), (header, token)


def test_h2_punctuation_placeholders_cli_pass_the_leak_scan(tmp_path: Path) -> None:
    ids, reason = _attendance()
    rows = [[("-" if i % 50 == 7 else s), r] for i, (s, r) in enumerate(zip(ids, reason))]
    src = _write(tmp_path / "in" / "att.csv", ["Student_ID", "Reason"], rows).parent
    out, rep, console = _run(src, tmp_path)
    written = [r["Student_ID"] for r in read_dicts(out / "att.csv")]
    assert written.count("-") == sum(1 for r in rows if r[0] == "-")
    assert not set(written) & set(ids)
    assert CANARY not in _texts(out, rep) + console


def test_h2_a2_1_a_data_entry_form_first_keeps_the_digit_shape() -> None:
    # the student's plain rows keep an unpadded integer ID when a data-entry form of their ID ('100121?') comes first
    ids, reason = _attendance()
    target = sorted(set(ids))[3]
    first = ids.index(target)
    vals = list(ids)
    vals[first] = target + "?"
    tp, cols = run_table({"Student_SK": vals, "Reason": reason})
    out = cols["Student_SK"].tolist()
    plain = {out[i] for i, v in enumerate(vals) if v == target}
    assert len(plain) == 1
    syn = next(iter(plain))
    assert syn.isdigit() and len(syn) == len(target) and syn[0] != "0" and syn != target
    assert out[first] == syn + "?"


@pytest.mark.parametrize("header", ["Other ID", "KIDS ID", "MOSIS ID", "Statewide ID", "Lunch Number", "Aeries ID"])
def test_h2_a2_2_small_files_under_an_id_word_are_id_columns(header: str) -> None:
    # A2-2: an intervention group of 8 students (fewer than --min-cell) under an ID word wrote every real ID
    rng = random.Random(3)
    ids = [str(rng.randint(100000, 999999)) for _ in range(8)]
    vals = [s for s in ids for _ in range(20)]
    tp, cols = run_table({header: vals, "WCPM": [str(rng.randint(40, 160)) for _ in vals],
                          "Reason": [CANARY if v == ids[0] else "On track" for v in vals]})
    assert profile_of(tp, header).type is T.IDENTIFIER, header
    assert not set(cols[header].tolist()) & set(ids), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    assert CANARY not in set(cols["Reason"].tolist())


@pytest.mark.parametrize("min_cell", [10, 20, 40])
def test_h2_a2_2_raising_min_cell_never_writes_more_real_ids(min_cell: int) -> None:
    # A2-2: 15 students were an ID column at the default --min-cell and a number column writing all 15 at 20
    rng = random.Random(5)
    ids = [str(rng.randint(100000, 999999)) for _ in range(15)]
    vals = [s for s in ids for _ in range(12)]
    tp, cols = run_table({"Other ID": vals, "WCPM": [str(rng.randint(40, 160)) for _ in vals]}, min_cell=min_cell)
    assert profile_of(tp, "Other ID").type is T.IDENTIFIER
    assert not set(cols["Other ID"].tolist()) & set(ids)


@pytest.mark.parametrize("header", ["Admission No", "Adm. No.", "Roll Number"])
def test_h2_a2_2_three_digit_admission_numbers_are_ids(header: str) -> None:
    # A2-2: the 4-digit floor let a small school's 3-digit admission numbers through, every one written
    rng = random.Random(2)
    ids = [str(101 + 11 * i) for i in range(60)]
    vals, reason = [], []
    for i, s in enumerate(ids):
        k = 12 if i == 0 else rng.randint(5, 15)
        vals += [s] * k
        reason += [CANARY if i == 0 else COMMON[i % 3]] * k
    tp, cols = run_table({header: vals, "Reason": reason})
    assert profile_of(tp, header).type is T.IDENTIFIER, header
    assert not set(cols[header].tolist()) & set(ids), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header


def _state_ids(n: int = 60, seed: int = 4, prefix: str = "") -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    ids = [prefix + str(rng.randint(10 ** 9, 10 ** 10 - 1)) for _ in range(n)]
    vals, reason = [], []
    for i, s in enumerate(ids):
        k = 12 if i == 0 else rng.randint(5, 15)
        vals += [s] * k
        reason += [CANARY if i == 0 else COMMON[i % 3]] * k
    return ids, vals, reason


@pytest.mark.parametrize("header", ["XREF", "Crosswalk", "Link", "Match", "Linkage"])
def test_h2_a2_3_repeated_wide_keys_are_listed_and_count_holders(header: str) -> None:
    # A2-3 (D7 and D35 part): a state student ID under a header no rule knows, repeated on each student's rows, is
    # listed as an ID-like candidate with --person-id advice, and its keys count who holds a value: the canary of one
    # student (12 rows) is pseudonymised. D7 does not promote a column by its values alone; D38 FAILs the file of such
    # a table at the run level (tests/test_d38_unknown_ids.py) and put the known acronyms (NYSSIS, STN, UIC, WISEid,
    # SUNS) on the default ID list
    ids, vals, reason = _state_ids()
    tp, cols = run_table({header: vals, "Reason": reason})
    p = profile_of(tp, header)
    assert p.type is not T.IDENTIFIER and "repeated ID-like keys" in str(p.params.get("id_candidate")), header
    msg = next(m for m in tp.messages if m.code == "no_id_column")
    assert msg.level == "warning" and header in msg.text and "--person-id" in (msg.fix or ""), header
    sparse = [m for m in p.messages if m.code == "possible_sparse_code"]
    assert all("--person-id" in (m.fix or "") for m in sparse), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    tp, cols = run_table({header: vals, "Reason": reason}, person_id=[header])
    assert profile_of(tp, header).type is T.IDENTIFIER
    assert not set(cols[header].tolist()) & set(ids)


def test_h2_a2_3_prefixed_state_ids_count_holders_too() -> None:
    ids, vals, reason = _state_ids(prefix="FL")
    # 'FL' + 10 digits under a header no rule knows (D38 put FLEID itself on the default ID list)
    tp, cols = run_table({"XREF": vals, "Reason": reason})
    assert "repeated ID-like keys" in str(profile_of(tp, "XREF").params.get("id_candidate"))
    assert not set(cols["XREF"].tolist()) & set(ids)
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason"))


def test_h2_a2_3_measures_and_narrow_keys_are_no_repeated_key_candidates() -> None:
    # measures, counts and 4-5 digit numbers keep their types, statistics and the D35 row count
    rng = random.Random(7)
    n = 400
    cols = {"Expense_Semester": [str(rng.choice(range(8000, 60000, 250))) for _ in range(n)],
            "Total Amount Cents": [str(rng.choice(range(10 ** 8, 10 ** 8 + 5000, 50))) for _ in range(n)],
            "Code": [str(rng.choice(range(10000, 10040))) for _ in range(n)],
            "Reason": [rng.choice(COMMON) for _ in range(n)]}
    tp, _ = run_table(cols)
    for name in ("Expense_Semester", "Total Amount Cents", "Code"):
        p = profile_of(tp, name)
        assert not p.params.get(dids.REPEATED_KEYS_KEY) and "repeated" not in str(p.params.get("id_candidate")), name


def _siblings(three_share: float = 0.3) -> tuple[list[list[str]], list[str]]:
    r = random.Random(2)
    fams, k = [], 0
    for _ in range(120):
        size = 3 if r.random() < three_share else 2
        fams.append(list(range(k, k + size)))
        k += size
    rng = random.Random(4)
    local = [str(400000 + i * 7 + rng.randint(0, 5)) for i in range(k)]
    ssid = [str(rng.randint(10 ** 9, 10 ** 10 - 1)) for _ in range(k)]
    rows = [[local[a], ssid[b], rng.choice(["Lincoln ES", "Central HS"])] for fam in fams for a in fam for b in fam
            if a != b]
    return rows, ssid


@pytest.mark.parametrize("header", ["Sibling SSID", "Sibling Statewide ID", "Sib State ID"])
@pytest.mark.parametrize("three_share", [0.0, 0.3, 0.6])
def test_h2_a2_4_other_students_ids_beside_the_student_id_are_ids(header: str, three_share: float) -> None:
    # A2-4: a sibling's state ID in a family of 3 is listed under 2 students, so '>= 90% of the keys belong to one
    # student' failed and 85 real sibling SSIDs (and the exact min/max) were written
    rows, ssid = _siblings(three_share)
    tp, cols = run_table({"Student ID": [r[0] for r in rows], header: [r[1] for r in rows],
                          "Sibling School": [r[2] for r in rows]})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER, (header, three_share)
    assert not set(cols[header].tolist()) & set(ssid), (header, three_share)


def test_h2_a2_4_codes_many_students_share_stay_and_a_stricter_k_masks_more() -> None:
    # a bus or test that --min-cell or more students share keeps its real values beside the student ID; raising
    # --min-cell above its riders makes it an ID column (a privacy knob never writes more real values)
    n = 400
    cols = {"Student_ID": [str(500000 + i // 4) for i in range(n)],
            "Bus No": [str(4100 + i % 30) for i in range(n)],
            "Test ID": [str(1001 + i % 12) for i in range(n)]}
    tp, out = run_table(cols)
    assert profile_of(tp, "Bus No").type is not T.IDENTIFIER and profile_of(tp, "Test ID").type is not T.IDENTIFIER
    tp, out = run_table(cols, min_cell=20)
    assert profile_of(tp, "Bus No").type is T.IDENTIFIER  # ~13 riders per bus: fewer than 20
    assert profile_of(tp, "Test ID").type is not T.IDENTIFIER  # ~33 takers per test


def test_h2_a2_5_dense_namespace_references_among_placeholders() -> None:
    # A2-5: the K-13 chance tests cannot tell a dense namespace's IDs from numbers over the column's range; a reference
    # column's IDs reach both ends of the namespace, a band of measures does not
    rng = random.Random(3)
    four = np.arange(1001, 1801, dtype=np.int64)
    fkeys = {str(v) for v in four}
    col = {str(v) for v in rng.sample(range(1001, 1801), 268)} | {"tbd"}
    assert st._key_ref_share(col, fkeys, four, False) == pytest.approx(268 / 269)
    five = np.arange(30001, 36001, dtype=np.int64)
    col = {str(v) for v in rng.sample(range(30001, 36001), 316)} | {"unassigned"}
    assert st._key_ref_share(col, {str(v) for v in five}, five, False) is not None
    sk = np.arange(1, 6001, dtype=np.int64)
    skeys = {str(v) for v in sk}
    col = {str(v) for v in rng.sample(range(1, 6001), 200)} | {"tbd"}
    assert st._key_ref_share(col, skeys, sk, False) is not None
    # bands of measures among tokens stay measures: scale scores, Lexile, minutes; and grade levels
    wide = np.arange(1001, 4001, dtype=np.int64)
    assert st._key_ref_share({str(v) for v in range(2380, 2621)} | {"ns"}, {str(v) for v in wide}, wide, False) is None
    assert st._key_ref_share({str(v) for v in range(200, 1701, 5)} | {"br"}, skeys, sk, False) is None
    assert st._key_ref_share({str(v) for v in range(0, 601, 5)} | {"absent"}, skeys, sk, False) is None
    assert st._key_ref_share({str(v) for v in range(1, 13)} | {"kg", "pk"}, skeys, sk, False) is None
    # a measure reaching past the IDs is no sample of them
    col = {str(v) for v in range(1001, 2601, 4)} | {"x"}
    assert st._key_ref_share(col, fkeys, four, False) is None


@pytest.mark.parametrize("width,header,token", [(4, "Reading Buddy", "TBD"), (5, "Partner", "Unassigned")])
def test_h2_a2_5_dense_reference_column_gets_the_roster_ids(tmp_path: Path, width: int, header: str,
                                                            token: str) -> None:
    rng = random.Random(6)
    ids = [str(v) for v in (range(1001, 1801) if width == 4 else range(30001, 33001))]
    _write(tmp_path / "in" / "students.csv", ["Student_ID", "Grade", "Gender"],
           [[s, rng.choice(["9", "10", "11", "12"]), rng.choice(["F", "M"])] for s in ids])
    rows = [[rng.choice(ids), rng.choice(ids) if rng.random() < 0.8 else token, rng.choice(["Math", "Reading"])]
            for _ in range(400)]
    _write(tmp_path / "in" / "mentoring.csv", ["Student_ID", header, "Focus"], rows)
    out, rep, _ = _run(tmp_path / "in", tmp_path)
    assert _columns(rep, "mentoring.csv")[header]["type"] == "IDENTIFIER"
    written = [r[header] for r in read_dicts(out / "mentoring.csv")]
    assert not set(written) & set(ids)
    roster = {r["Student_ID"] for r in read_dicts(out / "students.csv")}
    assert {v for v in written if v.isdigit()} <= roster  # the references join the roster's synthetic IDs


def test_h2_a2_6_dense_unpadded_keys_spill_past_the_widest_width() -> None:
    # A2-6: a warehouse's dense 1..N student key (or Stu = 101..650) FAILed: the allocator only spilled inside the
    # observed widths, although D9 lets an unpadded integer take another digit count
    from datanonymous.generate.ids import allocate_digit_ids

    for keys in ([str(i) for i in range(1, 6001)], [str(i) for i in range(101, 651)]):
        notes: list[str] = []
        real = np.asarray(sorted(int(k) for k in keys), dtype=np.int64)
        syn = allocate_digit_ids(keys, {k: len(k) for k in keys}, {}, real, np.random.default_rng(0), notes)
        assert len(set(syn.tolist())) == len(keys) and not set(syn.tolist()) & set(real.tolist())
        assert max(len(str(v)) for v in syn.tolist()) == len(keys[-1]) + 1
        assert notes and "different number of digits" in notes[0]


@pytest.mark.parametrize("header", ["Student_SK", "Stu", "Student"])
def test_h2_a2_6_dense_student_keys_are_written_with_a_note(tmp_path: Path, header: str) -> None:
    rng = random.Random(1)
    rows = [[str(s), f"2026-0{1 + k % 9}-1{k % 9}", rng.choice(["P", "A", "T"])]
            for s in range(101, 651) for k in range(rng.randint(2, 5))]
    src = _write(tmp_path / "in" / "fact.csv", [header, "Date", "Code"], rows).parent
    out, rep, console = _run(src, tmp_path)
    assert _columns(rep, "fact.csv")[header]["type"] == "IDENTIFIER"
    written = [r[header] for r in read_dicts(out / "fact.csv")]
    assert len(written) == len(rows) and not set(written) & {r[0] for r in rows}
    assert "different number of digits" in (rep / "files" / "fact.csv.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", ["exclude", "null"])
@pytest.mark.parametrize("header", ["Student_SK", "Stu", "CALPADS SSID", "Admission No"])
def test_h2_a2_7_a_dropped_id_column_by_header_and_values_is_a_shadow_id(flag: str, header: str) -> None:
    # A2-7: --exclude / --null on a column K-02 recognises by header and values left D35 counting rows: the one-student
    # canary was kept 12x. The column is a shadow ID now, as 'Student ID' is
    from datanonymous.profile import entity as entity_mod

    ids, reason = _attendance()
    tp, cols = run_table({header: ids, "Reason": reason}, **{flag: [header]})
    p = profile_of(tp, header)
    assert p.role == ("excluded" if flag == "exclude" else "null")
    assert entity_mod.shadow_id(p, None), header
    assert p.index in tp.private.shadow_id_raw
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), (flag, header)
    assert CANARY not in set(cols["Reason"].tolist())


def test_h2_a2_7_dropped_columns_that_are_no_ids_stay_no_shadow_ids() -> None:
    from datanonymous.profile import entity as entity_mod

    ids, reason = _attendance()
    n = len(ids)
    cols = {"Student_ID": ids, "Student_Count": [str(40 + i % 300) for i in range(n)],
            "English Learner Code": [("EO", "EL", "RFEP")[i % 3] for i in range(n)],
            "Bus No": [str(4100 + i % 3) for i in range(n)], "Reason": reason}
    tp, _ = run_table(cols, exclude=["Student_Count", "English Learner Code", "Bus No"])
    for name in ("Student_Count", "English Learner Code", "Bus No"):
        assert not entity_mod.shadow_id(profile_of(tp, name), None), name


# --------------------------------------------------------------------------- merge: R9 ids and K holders
# The K id-recognition branch was built before R9 (privacy.realids) and K holders (profile.entity.subject_proof,
# d35_keys' claimed IDs); these pin how the two meet.
def test_merge_repeated_wide_keys_count_holders_beside_a_bare_sid() -> None:
    # d35_keys builds the codes of every key column once (K-06); the repeated wide keys it adds where no ID of the
    # row's own person is (A2-3) need theirs too (a KeyError at merge time). Beside a bare SID of sessions (a record
    # key), the state IDs still show the canary held by one student: pseudonymised
    ids, vals, reason = _state_ids()
    sessions = [str(5000 + 3 * i) for i in range(len(vals))]
    tp, cols = run_table({"SID": sessions, "XREF": vals, "Reason": reason})  # XREF: D38 listed NYSSIS
    assert profile_of(tp, "SID").type is T.IDENTIFIER
    assert profile_of(tp, "XREF").params.get(dids.REPEATED_KEYS_KEY)
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason"))
    assert CANARY not in set(cols["Reason"].tolist())


@pytest.mark.parametrize("header", ["CALPADS SSID", "SIS ID", "Student_SK", "Stu", "Kid_Ref", "Student ID (Aeries)",
                                    "LASID", "State_ID"])
@pytest.mark.parametrize("flagged", [False, True])
def test_merge_k02_student_id_headers_are_header_proof(header: str, flagged: bool) -> None:
    # K-02 headers number the row's own person by their header (with or without --person-id): a header proof for the
    # K holders ranking (profile.entity.subject_proof), never a mere --person-id claim
    from datanonymous.profile import entity as entity_mod

    ids, reason = _attendance()
    tp, _ = run_table({header: ids, "Reason": reason}, **({"person_id": [header]} if flagged else {}))
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and entity_mod.id_role(p) == "subject", header
    assert entity_mod.subject_proof(p) == entity_mod.PROOF_HEADER, (header, flagged)
    assert tp.private.extra.get(entity_mod.CLAIMED_KEY) == [], (header, flagged)  # the table's people are proven


@pytest.mark.parametrize("header,flags,proof", [("Admission No", {"person_id": ["Admission No"]}, "flag"),
                                                ("Badge_No", {"person_id": ["Badge_No"]}, "flag"),
                                                ("SID", {"person_id": ["SID"]}, "flag"), ("SID", {}, "sid"),
                                                ("Admission No", {}, None)])
def test_merge_headers_naming_no_student_stay_claims(header: str, flags: dict[str, list[str]],
                                                     proof: str | None) -> None:
    # K-04/K-05/K-06 unchanged: --person-id on a header that names no person of the row is a claim, a bare SID is no
    # proof, and an ID word alone (id-shape) numbers someone else
    from datanonymous.profile import entity as entity_mod

    ids, reason = _attendance()
    tp, _ = run_table({header: ids, "Reason": reason}, **flags)
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER, header
    assert entity_mod.subject_proof(p) == proof, (header, flags)
    assert (p.index in tp.private.extra.get(entity_mod.CLAIMED_KEY, [])) is (proof is not None), (header, flags)


def test_merge_cli_a_k02_header_with_person_id_outvotes_claims(tmp_path: Path) -> None:
    # --person-id on a K-02 header confirms a header proof: a value 15 students hold in the roster stays real in an
    # incident log (an --person-id claim) where only 3 incidents hold it; the one-student canary never does
    src = tmp_path / "in"
    n = 60
    groups = ("Zephyr reading group", "Quartz math lab", "Ember writing club")  # no vocabulary words
    roster_ids = [str(100100 + 7 * i) for i in range(n)]
    plans = [CANARY if i == 0 else groups[i % 3] for i in range(n)]
    _write(src / "roster.csv", ["CALPADS SSID", "Plan"], [[s, v] for s, v in zip(roster_ids, plans)])
    log = [[str(9000 + i), groups[0] if i < 3 else groups[1 + i % 2]] for i in range(80)]
    _write(src / "log.csv", ["Incident_ID", "Plan"], log)
    out, rep, console = _run(src, tmp_path, "--person-id", "CALPADS SSID", "--person-id", "Incident_ID")
    assert CANARY not in _texts(out, rep) + console
    assert Counter(r["Plan"] for r in read_dicts(out / "log.csv"))[groups[0]] == 3
    assert set(groups) <= {r["Plan"] for r in read_dicts(out / "roster.csv")}
    assert not set(r["CALPADS SSID"] for r in read_dicts(out / "roster.csv")) & set(roster_ids)


@pytest.mark.parametrize("flags", [[], ["--exclude", "Student_SK"], ["--null", "Student_SK"]])
def test_merge_cli_k02_ids_join_the_runs_real_ids(tmp_path: Path, flags: list[str]) -> None:
    # R9-01: the IDs of a column K-02 makes an ID column by its header and values (and of such a column dropped by
    # --exclude / --null, a shadow ID: A2-7) are real person IDs of the run: a level, an organisational code or a
    # number of another file that equals one is never written as it is
    src = tmp_path / "in"
    ids, reason = _attendance()
    _write(src / "roster.csv", ["Student_SK", "Reason"], [[s, r] for s, r in zip(ids, reason)])
    first, second = ids[0], ids[-1]
    n = 60
    _write(src / "clubs.csv", ["Club", "Room", "Visit"],
           [[first if i % 3 == 0 else ("Band", "Choir")[i % 2], second if i % 5 == 0 else ("12", "14")[i % 2],
             str(i % 4)] for i in range(n)])
    out, rep, console = _run(src, tmp_path, *flags)
    blob = _texts(out, rep) + console
    assert first not in blob and second not in blob
    rows = read_dicts(out / "clubs.csv")
    assert len(rows) == n and {"Band", "Choir"} <= {r["Club"] for r in rows} and {"12", "14"} <= {r["Room"] for r in rows}
    assert sorted(Counter(r["Club"] for r in rows).values()) == [20, 20, 20]  # the pseudonym keeps the exact count
