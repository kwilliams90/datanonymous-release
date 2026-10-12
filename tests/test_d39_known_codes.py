"""D39 (user, 2026-10-11): known non-person code headers are org codes (D10: kept real) and never trip D38.

* NCES school and district IDs (``NCES``, ``NCESSCH``, ``NCES_SCH_ID``, ``School_NCES_ID``, ``NCES School ID``,
  ``NCES_ID``, ``NCESID``, ``LEAID``, ``LEA_ID``, the Common Core of Data's ``SCHID``/``ST_SCHID``/``ST_LEAID``),
  California CDS codes (``CDS``, ``CDS Code``, ``CDS_CODE``, ``CDSCode``), Texas county-district-campus numbers
  (``CDC``, ``County-District-Campus Number``; ``Campus Number`` and ``District Number`` were D10 org codes already),
  general-ledger account codes (``GL_Account``, ``Account Code``), course reference numbers (``CRN``, ``Term_CRN``)
  and ISBNs (``ISBN``, ``ISBN13``, ``ISBN-10``) are org-code headers (``hints.org_code_hint``,
  ``hints.known_code_hint``), case-insensitively and with separators.
* The generic finance segment words ``Fund``, ``Function`` and ``Object`` (also ``Fund_Code``, ``Obj_Cd``) make an org
  code only over code values (``detect.types.finance_code_values``: digits of one width, or dash/space/dotted segment
  codes); free text, amounts and single digits are unchanged. A bare ``Account`` (a lunch, fee or library account
  number is a person's) is never one: it still FAILs under D38 in a table with no ID, and beside a student ID it is
  unaffected.
* ``NCES`` is no plural of the ``NCE`` measure any more (``hints.measure_header``): NCES headers are org codes by rule.
* Deliberately left out, so they still FAIL under D38: census tracts and ``GEOID`` (where a student lives), provider
  NPIs (a person's ID) and ``Barcode`` (it may be a student ID card).
* D34/D37: ``Rt`` and ``Rte`` (route abbreviations) are numbered-thing words like ``Route``: ``Rt 1201`` and
  ``Rte_1305`` headers are numbered routes, never real IDs as bare-number names.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from conftest import read_dicts
from datanonymous import hints
from datanonymous.detect.types import finance_code_values
from datanonymous.model import ColumnType as T
from datanonymous.privacy import headers as d34
from datanonymous.profile.table import unrecognised_key_columns
from test_area_a_support import detect, profile_of, run_table
from test_d38_unknown_ids import attendance, column_report, everything, keys, run, statuses, write_csv

# =========================================================================== header readings
KNOWN = ["NCES", "nces", "NCESSCH", "NCES_SCH_ID", "NCESSCH_ID", "School_NCES_ID", "NCES School ID", "NCES_ID",
         "NCESID", "NCES_LEAID", "NCES LEA ID", "NCES District ID", "District_NCES_ID", "LEAID", "LEA_ID", "LEA Code",
         "ST_LEAID", "SCHID", "SCH_ID", "ST_SCHID", "CDS", "CDS Code", "CDS_CODE", "CDSCode",
         "County_District_School_Code", "CDC", "CDC Number", "County-District-Campus Number", "County_District",
         "GL_Account", "GL Account", "GL Account Number", "GLAccount", "GL_Acct_No", "General Ledger Account",
         "Account Code", "Acct_Code", "Account_Cd", "AccountCode", "CRN", "Term_CRN", "TermCRN", "CRN_Term",
         "Course_CRN", "Course Reference Number", "ISBN", "isbn", "ISBN13", "ISBN-10", "ISBN_13", "ISBN Number",
         "Textbook_ISBN"]
#: D39 leaves these out (or they name no code): no org code, no finance segment
NOT_CODES = ["Account", "Account_Number", "Acct_No", "Account #", "Lunch_Account", "Student_Account",
             "Student_Account_Code", "LEA", "Lea", "Leano", "LEA_Name", "GL", "GL_Code", "Census_Tract", "GEOID",
             "Tract", "NPI", "Provider_NPI", "Barcode", "Item", "Batch", "Scan", "Cents", "Encumbrance", "Line", "XREF",
             "Household", "Student_CRN", "Student_NCES"]
FINANCE = ["Fund", "Function", "Object", "Fund_Code", "FUND_CD", "Object Code", "Obj_Cd", "Func",
           "Fund_Function_Object"]
NOT_FINANCE = ["Functional", "Objective", "Funding", "Fund_Balance", "Fund Amount", "Fund_ID", "Object_ID", "Num_Fund",
               "Executive_Function", "Funds", "Objects"]


@pytest.mark.parametrize("name", KNOWN)
def test_known_code_headers_are_org_codes(name: str) -> None:
    h = hints.header_hints(name)
    assert h.org_code and hints.known_code_hint(name), name
    assert not h.person_id_like and not h.default_id and h.pii is None, name


@pytest.mark.parametrize("name", ["Campus Number", "District Number", "Campus", "SchoolID", "Section_Key"])
def test_d10_org_codes_stay_d10(name: str) -> None:
    assert hints.org_code_hint(name) and not hints.known_code_hint(name), name


@pytest.mark.parametrize("name", NOT_CODES)
def test_left_out_and_generic_headers_are_no_codes(name: str) -> None:
    h = hints.header_hints(name)
    assert h.org_code is None and h.finance_segment is None and not hints.known_code_hint(name), name


def test_a_student_word_keeps_a_known_code_the_students() -> None:
    # fail-safe: after a student word a D39 word is no organisation (K-02 lets the values decide)
    for name in ("Student_CRN", "Student_NCES", "Student CDS"):
        assert hints.weak_id_header(name) == "person", name
    assert hints.weak_id_header("Student_School") is None  # D10's own words still are


@pytest.mark.parametrize("name", FINANCE)
def test_finance_segment_words_wait_for_their_values(name: str) -> None:
    h = hints.header_hints(name)
    assert h.finance_segment and h.org_code is None, name


@pytest.mark.parametrize("name", NOT_FINANCE)
def test_other_finance_like_headers_are_no_segments(name: str) -> None:
    assert hints.finance_segment_hint(name) is None, name


def test_nces_is_no_plural_of_nce() -> None:
    for name in ("NCES", "NCES_ID", "NCES School ID", "School_NCES_ID"):
        assert not hints.measure_header(name), name
    for name in ("NCE", "NCE_Score", "Scores", "Mins", "Amounts"):
        assert hints.measure_header(name), name  # every other measure and plural reading is unchanged


@pytest.mark.parametrize("values", [["199", "211", "240"], ["11", "23", "99"], ["6119"],
                                    ["101611900001", "199116119000"],
                                    ["199-11-6119-00", "211-11-6129-00"], ["10 1000 6110", "10 2000 6110"],
                                    ["10.1000.6110", "10.2000.6110"], ["0100", "0200"]])
def test_finance_code_values(values: list[str]) -> None:
    assert finance_code_values(dict.fromkeys(values, 3))


@pytest.mark.parametrize("values", [["1", "2", "3"], ["199", "1999"], ["6119.50", "6120.25"], ["-12", "-13"],
                                    ["1,200", "1,300"], ["Instruction", "Admin"], ["199-11-6119-00", "199-11-61190-00"],
                                    ["E-199-11", "E-211-11"], ["１９９", "２１１"], []])
def test_no_finance_code_values(values: list[str]) -> None:
    assert not finance_code_values(dict.fromkeys(values, 3))


# =========================================================================== detection
def cycle(values: list[str], n: int = 120) -> list[str]:
    return [values[i % len(values)] for i in range(n)]


def test_a_finance_segment_over_codes_is_an_org_code() -> None:
    for name, values in (("Fund", ["199", "211", "240"]), ("Function", ["11", "23", "36"]),
                         ("Object_Code", ["6119", "6219", "6399"]), ("Fund", ["199-11-6119-00", "211-11-6129-00"])):
        det = detect(cycle(values), name)
        assert det.type is T.ORG_CODE and "D39" in det.reason, (name, values)


@pytest.mark.parametrize("name, values, expected", [
    ("Fund", [f"{1000 + 37.25 * i:.2f}" for i in range(40)], T.FLOAT),  # amounts
    ("Fund", [str(5 + 977 * i) for i in range(40)], T.INTEGER),  # integers of varying width
    ("Function", ["Instruction", "Administration", "Transportation"], T.FACTOR),  # free text
    ("Object", ["1", "2", "3", "4", "5"], T.INTEGER),  # single digits
])
def test_a_finance_segment_over_other_values_is_unchanged(name: str, values: list[str], expected: T) -> None:
    det = detect(cycle(values), name)
    assert det.type is expected and det.type is not T.ORG_CODE, (name, det.type)


def test_a_known_code_says_d39() -> None:
    det = detect(cycle(keys(n=20, digits=12)), "NCESSCH")
    assert det.type is T.ORG_CODE and "known non-person code" in det.reason and "D39" in det.reason
    det = detect(cycle(["101", "102", "103"]), "School")
    assert det.type is T.ORG_CODE and "D39" not in det.reason  # D10's words keep their reason


# =========================================================================== D38: no FAIL for known codes
WRITTEN = ["NCESSCH", "NCES_SCH_ID", "School_NCES_ID", "NCES School ID", "NCES_ID", "LEAID", "LEA_ID", "SCHID",
           "CDS Code", "CDS_CODE", "CDSCode", "CDS", "CDC", "County-District-Campus Number", "GL_Account",
           "Account Code", "Fund", "Object", "Function", "Fund_Code", "CRN", "Term_CRN", "ISBN", "ISBN13"]


@pytest.mark.parametrize("header", WRITTEN)
def test_known_codes_are_org_codes_in_a_table_with_no_id(header: str) -> None:
    # D38's case (wide keys repeated across rows, no recognised ID column): kept real as an org code, no FAIL
    ids = keys(digits=12)
    tp, cols = run_table(attendance(header, ids))
    p = profile_of(tp, header)
    assert p.type is T.ORG_CODE and p.role == "row", header
    assert unrecognised_key_columns(tp) == [], header
    assert set(cols[header].tolist()) == set(ids), header  # real values kept (D10)


def npis(n: int = 60, seed: int = 4) -> list[str]:
    return keys(n=n, digits=9, seed=seed, prefix="1")  # ten digits starting with 1 (no North American phone)


@pytest.mark.parametrize("header, digits", [("Census_Tract", 11), ("GEOID", 11), ("Tract", 11), ("NPI", 10),
                                            ("Barcode", 13), ("Account", 12), ("Item", 13), ("Batch", 10),
                                            ("Scan", 12), ("Cents", 8), ("Encumbrance", 8), ("Line", 8),
                                            ("Household", 10)])
def test_left_out_codes_still_fail(header: str, digits: int) -> None:
    # D39 leaves out census tracts/GEOID (residence), provider NPIs (a person's ID) and Barcode (a student ID card);
    # the other headers name no known code: D38 still flags and skips their table
    ids = npis() if header == "NPI" else keys(digits=digits)
    tp, _ = run_table(attendance(header, ids))
    assert profile_of(tp, header).type is not T.ORG_CODE, header
    assert [p.name for p in unrecognised_key_columns(tp)] == [header], header


def test_known_codes_are_written_and_left_out_codes_fail_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "in"
    known = {"nces.csv": "NCESSCH", "cds.csv": "CDS Code", "cdc.csv": "CDC", "gl.csv": "GL_Account",
             "fund.csv": "Fund", "crn.csv": "Term_CRN", "isbn.csv": "ISBN13", "lea.csv": "LEA_ID"}
    left = {"tract.csv": "GEOID", "npi.csv": "NPI", "barcode.csv": "Barcode", "account.csv": "Account"}
    real: dict[str, list[str]] = {}
    for k, (file, header) in enumerate({**known, **left}.items()):
        real[file] = npis(30, seed=40 + k) if header == "NPI" else keys(n=30, digits=12, seed=40 + k)
        write_csv(src / file, attendance(header, real[file], seed=k))
    write_csv(src / "roster.csv", {"Student_ID": [str(500000 + 7 * i) for i in range(30)],
                                   "Grade": [str(9 + i % 4) for i in range(30)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    st = statuses(rep)
    assert {f for f, s in st.items() if s == "FAIL"} == set(left), st
    for file, header in known.items():
        assert column_report(rep, file, header)["type"] == "ORG_CODE", file
        written = {r[header] for r in read_dicts(out / file)}
        assert written == set(real[file]), file  # kept real (D10)
    blob = everything(out, rep, console=res.output)
    for file in left:
        assert not (out / file).exists(), file
        assert not {k for k in real[file] if k in blob}, file  # D38's FAIL never prints a value


def test_lea_id_is_a_district_code_not_a_synthetic_id() -> None:
    # before D39 a 7-digit LEA_ID in a table with no other ID became a weak ID column (synthetic IDs); D10 keeps
    # district codes real
    ids = keys(digits=7)
    tp, cols = run_table(attendance("LEA_ID", ids))
    p = profile_of(tp, "LEA_ID")
    assert p.type is T.ORG_CODE and p.role == "row"
    assert set(cols["LEA_ID"].tolist()) == set(ids)


def test_a_lunch_account_beside_a_student_id_is_unaffected() -> None:
    # a bare 'Account' may be a person's (lunch, fee, library): never an org code, so its real numbers are never kept
    # as levels; beside a student ID D38 does not apply, and the column stays what it was (an ID-like candidate)
    rng = random.Random(3)
    sids = [str(500000 + 7 * i) for i in range(60)]
    acct = {s: str(rng.randint(10_000_000, 99_999_999)) for s in sids}
    rows = [rng.choice(sids) for _ in range(300)]
    tp, cols = run_table({"Student_ID": rows, "Account": [acct[s] for s in rows],
                          "Amount": [f"{rng.uniform(1, 9):.2f}" for _ in rows]})
    p = profile_of(tp, "Account")
    assert p.type is T.INTEGER and p.role != "id"
    assert unrecognised_key_columns(tp) == []
    assert len(set(cols["Account"].tolist()) & set(acct.values())) <= 2  # interpolated numbers, as before D39


def test_a_finance_segment_beside_a_student_id_is_an_org_code() -> None:
    rng = random.Random(4)
    rows = [str(500000 + 7 * rng.randint(0, 59)) for _ in range(300)]
    funds = [rng.choice(["199", "211", "240", "461"]) for _ in rows]
    tp, cols = run_table({"Student_ID": rows, "Fund": funds})
    assert profile_of(tp, "Fund").type is T.ORG_CODE
    assert sorted(cols["Fund"].tolist()) == sorted(funds)  # exact counts (D14)


def test_org_code_on_a_known_code_changes_nothing() -> None:
    # the header decides already (as for 'School'): no d38_flagged_keys note, no withheld statistics
    ids = keys(digits=12)
    by_flag, _ = run_table(attendance("NCESSCH", ids), org_code=["NCESSCH"])
    by_header, _ = run_table(attendance("NCESSCH", ids))
    for tp in (by_flag, by_header):
        p = profile_of(tp, "NCESSCH")
        assert p.type is T.ORG_CODE and "d38_flagged_keys" not in [m.code for m in p.messages]
    assert profile_of(by_flag, "NCESSCH").source == "org-code" and profile_of(by_header, "NCESSCH").source == "value"


# =========================================================================== Rt / Rte numbered routes
CHARTER = [str(x) for x in range(1001, 4501)]
GROUP_B = ["1180", "1349", "1518", "1561"]


class NS:
    def __init__(self, values: list[str]) -> None:
        self.raw_values = set(values)
        self.representatives = set(values)


@pytest.mark.parametrize("make", ["Rt {}", "Rte_{}", "RT{}", "Rte {}", "Rt_{}", "Bus Rt {}", "Route {}"])
def test_rt_and_rte_are_numbered_routes(make: str) -> None:
    keys_ = d34.IdKeys([NS(CHARTER)])
    assert d34.id_group_flags(keys_, [make.format(x) for x in GROUP_B]) == {}, make
    assert set(d34.id_group_flags(keys_, list(GROUP_B))) == {0, 1, 2, 3}  # bare numbers are real IDs (D37)


def test_rt_headers_are_written_beside_dense_ids(tmp_path: Path) -> None:
    src = tmp_path / "in"
    rng = random.Random(4)
    write_csv(src / "roster.csv", {"student_id": CHARTER, "grade": [rng.choice(["6", "7", "8"]) for _ in CHARTER]})

    def riders(make: str) -> dict[str, list[Any]]:
        return {"Grade": [str(g) for g in range(6, 12)],
                **{make.format(x): [str(rng.randint(0, 40)) for _ in range(6)] for x in GROUP_B}}

    write_csv(src / "Bus_Riders.csv", riders("Rt {}"))
    write_csv(src / "Bus_Riders_PM.csv", riders("Rte_{}"))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "FAIL" not in set(statuses(rep).values()), statuses(rep)
    header = (out / "Bus_Riders.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "Grade,Rt 1180,Rt 1349,Rt 1518,Rt 1561"
