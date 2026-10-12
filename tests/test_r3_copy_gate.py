"""Round-3 adversarial review, copy-gate cluster (R3-01, R3-02, R3-03).

Bug class: real person data reached the output through a path that writes values *verbatim* without being a
text/PII type: a ``Notes`` column typed INTEGER copied phone numbers (R3-01), low-cardinality prose became a
FACTOR whose levels were copied (R3-02), MULTI_SELECT items (e-mail addresses, names) were copied (R3-03).
The fix is one structural gate (``detect.text.copy_gate``): every value a detection would write as it is -
CONSTANT/FLAG/FACTOR/ORG_CODE/ORDINAL levels, DIGIT_CODE levels, MULTI_SELECT items, discrete numbers and the
exact extremes, NUMERIC_WITH_TOKENS tokens - goes through it. Since D35 the gate is an allowlist: a text value
with a personal-data shape (e-mail, URL, phone, SSN, street address, also inside longer text) is never kept,
whatever its count, and gets a same-shape pseudonym that follows its PII generator's rules (the column keeps its
type and its other values); digit-only 10/11-digit phones under a person/contact/notes header make a number
column same-shape masks. The leak scanner indexes every value the gate did not keep. A header naming a
note/comment column wins over numeric typing (D12).

The variants each reproduced a copied real value before the fix; the false-refusal tests pin that ordinary
school labels (long category labels, course titles, Likert levels, program lists, codes) stay real levels.
Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import EXAMPLES, read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.privacy import leaks

from test_area_a_support import detect, profile_of, run_table

PHONES = ["3125550199", "7735550109", "6305550184"]
PROSE = ["Call Zelmira Quenby at 312-555-0199 before dismissal",
         "Return medication to Althea Voss after the second class",
         "Contact Floriana Drexel when the student needs transport"]
EMAIL_LISTS = ["zelmira.quenby@district.edu;althea.voss@district.edu",
               "althea.voss@district.edu;floriana.drexel@district.edu",
               "floriana.drexel@district.edu"]
REAL_WORDS = ("Zelmira", "Quenby", "Althea", "Voss", "Floriana", "Drexel", "zelmira.quenby", "althea.voss",
              "floriana.drexel", "312-555-0199", *PHONES)


def cycle(values: list[str], n: int = 120) -> list[str]:
    return [values[i % len(values)] for i in range(n)]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def ids(n: int = 120) -> list[str]:
    return [f"S{i:05d}" for i in range(n)]


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "7"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def texts(*dirs: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file())


def assert_absent(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> None:
    blob = texts(*dirs) + "\n" + console
    found = [w for w in words if w in blob]
    assert not found, f"{len(found)} real fixture value(s) reached the output or report"


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def file_report(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    for t in file_report(rep, name)["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


# --------------------------------------------------------------------------- the shape screen
@pytest.mark.parametrize("value, kind", [
    ("Call Zelmira Quenby at 312-555-0199 before dismissal", "phone"),
    ("(312) 555-0199", "phone"),
    ("312.555.0199", "phone"),
    ("+1 312 555 0199", "phone"),
    ("1-800-555-0199", "phone"),
    ("Call mom 3125550199 after 3", "phone"),
    ("cell: +44 20 7946 0958", "phone"),
    ("zelmira.quenby@district.edu", "email"),
    ("Front Office;zq@gmail.com", "email"),
    ("see https://social.example.net/zq", "url"),
    ("www.zelmira-quenby.net", "url"),
    ("SSN 900-12-3456", "ssn"),
    ("900 12 3456", "ssn"),
    ("1428 Elmwood Avenue", "address"),
    ("Lives at 1428 Elmwood Ave, Apt 2", "address"),
    ("Bus stop: 77 N Main St", "address"),
])
def test_shapes_found_inside_values(value: str, kind: str) -> None:
    assert kind in dtext.pii_shapes(value)


@pytest.mark.parametrize("value", [
    "some college", "associate's degree", "Free/Reduced Lunch", "free/reduced", "AP English Language and Composition",
    "Introduction to the History of the United States", "Neither agree nor disagree", "Strongly agree",
    "Standard Nearly Met", "ECONOMICS AND STATISTICS", "Rift Valley", "Grade 12 Food Drive", "5K Fun Run",
    "K-2 Literacy Path", "Room 101, 2nd Floor", "Period 3, 5 Day Plan", "Algebra 1 (2 credits)", "Score 85-90",
    "2023-10-05", "10/05/2023", "2015-2016", "SY 2023-24", "Version 1.2.3.4", "+10 points", "ELL;SPED;504",
    "3125550199", "100%", "$1,234.50", "Extended time on tests and quizzes", "Section 04 - Room 12",
    "NCES 170993000123", "Rate 3125550199.5", "12/31/2023 1234567890",
])
def test_ordinary_labels_hold_no_shape(value: str) -> None:
    assert dtext.pii_shapes(value) == frozenset()


def test_digit_only_phone_needs_a_person_contact_or_notes_header() -> None:
    assert dtext.pii_shapes("3125550199") == frozenset()
    assert dtext.pii_shapes("3125550199", digit_phone=True) == {"phone"}
    assert dtext.pii_shapes("13125550199", digit_phone=True) == {"phone"}
    assert dtext.pii_shapes("3125550199.0", digit_phone=True) == {"phone"}
    assert dtext.pii_shapes("123456789", digit_phone=True) == frozenset()  # 9 digits: not a phone


# --------------------------------------------------------------------------- R3-01: notes over numeric typing
@pytest.mark.parametrize("header", ["Notes", "Teacher_Notes", "Notes_2", "Comments", "MedicalAlert", "Narrative"])
@pytest.mark.parametrize("values", [cycle(PHONES), [p + ".0" for p in cycle(PHONES)],
                                    [str(3125550100 + 7 * i) for i in range(120)], cycle(["1", "2", "3"])])
def test_notes_header_wins_over_numbers(header: str, values: list[str]) -> None:
    det = detect(values, header)
    assert det.type is T.HIGH_CARD_CATEGORICAL and det.subtype == "code"
    assert det.params.get("_numeric_mask") is True
    assert det.params["numeric_storage"] in ("INTEGER", "FLOAT")


@pytest.mark.parametrize("header, expected", [("Note_Count", T.INTEGER), ("Alert_Level", T.INTEGER),
                                              ("Comment_Score", T.INTEGER), ("Num_Notes", T.INTEGER)])
def test_counts_and_levels_of_notes_stay_numbers(header: str, expected: T) -> None:
    det = detect(cycle(["1", "2", "3", "4"]), header)
    assert det.type is expected


@pytest.mark.parametrize("header", ["Guardian", "Emergency", "Emergency_Number", "Parent_Contact", "Note_Count"])
def test_digit_phones_under_person_contact_or_notes_header_are_masked(header: str) -> None:
    det = detect(cycle(PHONES), header)
    # R4-02: a contact header ('Emergency', 'Emergency_Number') holding phone numbers is a PHONE column (fake
    # numbers); person and notes headers mask them
    assert det.type is T.PHONE or (det.type is T.HIGH_CARD_CATEGORICAL and det.subtype == "code")


@pytest.mark.parametrize("header", ["district_code", "nces_id", "account", "amount"])
def test_unhinted_ten_digit_numbers_are_not_phones(header: str) -> None:
    det = detect(cycle(PHONES), header)
    assert det.type is not T.HIGH_CARD_CATEGORICAL
    assert "pii_values" not in [m.code for m in det.messages]


def test_phone_header_numbers_with_tokens_are_masked() -> None:
    # 86% phones (below the PHONE type's 90%) plus 'refused': numbers with tokens wrote the exact min/max phone
    values = [("refused" if i % 7 == 0 else str(3125550100 + i)) for i in range(120)]
    det = detect(values, "Home_Phone")
    assert det.type is T.HIGH_CARD_CATEGORICAL and det.subtype == "code"
    tp, cols = run_table({"Home_Phone": values})
    out = set(cols["Home_Phone"].tolist())
    assert not out & set(values)
    assert "refused" not in out  # a text token is masked too (never kept as a word under a personal header)


# --------------------------------------------------------------------------- R3-02: copied labels and constants
@pytest.mark.parametrize("values, header", [
    (cycle(PROSE), "accommodation"),
    ([PROSE[0]] * 60, "accommodation"),  # CONSTANT
    ([PROSE[0] if i % 2 else "" for i in range(60)], "pickup"),  # FLAG_SINGLE_TOKEN
    (cycle(["Phone", "Text", "zelmira.quenby@gmail.com"]), "contact_pref"),
    (cycle(["Birth certificate", "Passport", "SSN 900-12-3456"]), "verification"),
    (cycle(["Hidden", "Private", "https://social.example.net/zq"]), "profile"),
    (cycle(["Front office", "Gym", "1428 Elmwood Avenue"]), "pickup_location"),
    (cycle(["Lincoln Elementary", "Washington Middle", "zq.principal@lincoln-sd.org"]), "School"),  # ORG_CODE
])
def test_labels_with_personal_data_are_never_copied(values: list[str], header: str) -> None:
    # D35: a value with a personal-data shape is never kept, whatever its count; the column keeps its type and its
    # other values follow the allowlist (vocabulary words or held by >= --min-cell rows here)
    det = detect(values, header)
    assert det.type in (T.FACTOR, T.CONSTANT, T.FLAG_SINGLE_TOKEN, T.ORG_CODE) and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({header: values})
    p = profile_of(tp, header)
    shaped = {v for v in values if v.strip() and dtext.pii_shapes(v)}
    assert shaped and shaped <= set(dtext.pseudonymized(p))
    assert "pseudonymized_levels" in [m.code for m in p.messages]
    out = cols[header].tolist()
    assert not {v.strip().casefold() for v in out if v.strip()} & {v.casefold() for v in shaped}
    assert len(out) == len(values) and leaks.sensitive_kind(p) == leaks.LABEL


@pytest.mark.parametrize("values, header", [
    (cycle(["Lincoln Elementary", "Washington Middle", "1200 Oak Street Annex"]), "Building"),
    (cycle(["Lincoln Elementary (www.lincoln-sd.org)", "Washington Middle"]), "School_Name"),
])
def test_org_code_address_or_website_is_kept(values: list[str], header: str) -> None:
    # D10: a building's street address or a school's website is not a person's; the org code keeps its values
    det = detect(values, header)
    assert det.type is T.ORG_CODE and "pii_values" not in [m.code for m in det.messages]


def test_numbers_with_a_personal_token_mask_only_that_token() -> None:
    scores = [str(40 + (i * 7) % 60) for i in range(120)]
    values = [("absent, called 312-555-0199" if i % 20 == 0 else scores[i]) for i in range(120)]
    det = detect(values, "score")
    assert det.type is T.NUMERIC_WITH_TOKENS and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"score": values})
    assert set(dtext.pseudonymized(profile_of(tp, "score"))) == {"absent, called 312-555-0199"}
    out = cols["score"].tolist()
    assert "absent, called 312-555-0199" not in out
    tokens = [v for v in out if not v.isdigit()]
    assert len(tokens) == values.count("absent, called 312-555-0199")
    assert all(len(t) == len("absent, called 312-555-0199") and t.count(",") == 1 for t in tokens)
    nums = [int(v) for v in out if v.isdigit()]
    assert 40 <= min(nums) and max(nums) <= 99  # the numbers keep their range (not masked)
    assert leaks.sensitive_kind(profile_of(tp, "score")) == leaks.LABEL


# --------------------------------------------------------------------------- R3-03: multi-select items
def test_multi_select_email_items_are_replaced_item_by_item() -> None:
    values = cycle(EMAIL_LISTS)
    det = detect(values, "cc")
    assert det.type is T.MULTI_SELECT and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"cc": values})
    out = cols["cc"].tolist()
    real_items = {it for v in EMAIL_LISTS for it in v.split(";")}
    assert set(dtext.pseudonymized(profile_of(tp, "cc"))) == real_items  # D35: e-mail items are never kept
    syn_items = [it for v in out for it in v.split(";")]
    assert not set(syn_items) & real_items
    assert all(it.endswith(("@example.org", "@example.com", "@example.net")) for it in syn_items)
    # the structure is kept: items per value, number of distinct items and combinations, item totals
    assert sorted(v.count(";") for v in out) == sorted(v.count(";") for v in values)
    assert len(set(syn_items)) == len(real_items) and len(set(out)) == len(set(values))
    real_totals = sorted(sum(v.split(";").count(it) for v in values) for it in real_items)
    assert sorted(syn_items.count(it) for it in set(syn_items)) == real_totals
    assert leaks.sensitive_kind(profile_of(tp, "cc")) == leaks.LABEL


def test_multi_select_items_mode_with_phones() -> None:
    items = ["Front Office", "Nurse", "(312) 555-0199", "(773) 555-0109", "Counselor", "(630) 555-0184"]
    rng = np.random.default_rng(3)
    values = [";".join(sorted(set(rng.choice(items, size=int(rng.integers(1, 4))).tolist()))) for _ in range(300)]
    det = detect(values, "call_list")
    assert det.type is T.MULTI_SELECT and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"call_list": values})
    real_phones = {it for it in items if it.startswith("(")}
    assert set(dtext.pseudonymized(profile_of(tp, "call_list"))) == real_phones
    syn_items = {it for v in cols["call_list"].tolist() for it in v.split(";")}
    assert not syn_items & real_phones and {"Front Office", "Nurse", "Counselor"} <= syn_items  # labels kept
    phones = [it for it in syn_items if any(ch.isdigit() for ch in it)]
    assert phones and all(it[6] in "23456789" and it.startswith("(") for it in phones)


def test_multi_select_within_id_constant_items_are_replaced() -> None:
    # D4: one value per synthetic student, repeated on its rows; still no real item
    sid = [f"S{i // 3:04d}" for i in range(150)]
    cc = [EMAIL_LISTS[(i // 3) % 3] for i in range(150)]
    tp, cols = run_table({"student_id": sid, "cc": cc})
    assert profile_of(tp, "cc").role == "entity"
    real_items = {it for v in EMAIL_LISTS for it in v.split(";")}
    out = cols["cc"].tolist()
    assert not {it for v in out for it in v.split(";")} & real_items
    by_id: dict[str, set[str]] = {}
    for k, v in zip(cols["student_id"].tolist(), out):
        by_id.setdefault(k, set()).add(v)
    assert all(len(v) == 1 for v in by_id.values())


def test_plain_multi_select_is_unchanged() -> None:
    values = cycle(["ELL;SPED", "SPED;504", "ELL", "GT;ELL"])
    det = detect(values, "programs")
    assert det.type is T.MULTI_SELECT and "sensitive_items" not in det.params
    tp, cols = run_table({"programs": values})
    assert set(cols["programs"].tolist()) == set(values)
    assert leaks.sensitive_kind(profile_of(tp, "programs")) is None


@pytest.mark.parametrize("header", ["participants", "Recipients", "Attendees", "Award_Nominees"])
def test_lists_of_people_under_collective_headers_are_never_copied(header: str) -> None:
    values = cycle(["Zelmira Quenby;Althea Voss", "Althea Voss;Floriana Drexel", "Floriana Drexel;Zelmira Quenby"])
    det = detect(values, header)
    assert det.type in (T.HIGH_CARD_CATEGORICAL, T.PERSON_NAME)
    assert det.params.get("pii_hint") == "full_name"


def test_singular_participant_status_stays_a_factor() -> None:
    det = detect(cycle(["Yes", "No", "Withdrawn"]), "Participant")
    assert det.type is T.FACTOR


# --------------------------------------------------------------------------- leak scanner indexes the gate's columns
def test_leak_scan_catches_a_copied_item_of_a_sensitive_multi_select() -> None:
    values = cycle(EMAIL_LISTS)
    tp, cols = run_table({"cc": values})
    p = profile_of(tp, "cc")
    synth = cols["cc"].copy()
    synth[0] = "althea.voss@district.edu;" + synth[0].split(";")[0]  # one real item copied into a combination
    from datanonymous.model import SyntheticColumn, SyntheticTable

    st = SyntheticTable(tp.name, tp.header, tp.n_rows, [SyntheticColumn(p.index, p.name, synth)], tp)
    msgs = leaks.scan_table(tp, st)
    assert any(m.code == "copied_value" and m.column == "cc" for m in msgs)
    st_ok = SyntheticTable(tp.name, tp.header, tp.n_rows, [SyntheticColumn(p.index, p.name, cols["cc"])], tp)
    assert not [m for m in leaks.scan_table(tp, st_ok) if m.code == "copied_value"]


def test_embedded_personal_data_is_indexed_in_any_layout() -> None:
    terms = leaks._Terms()
    terms.add(leaks.FREE_TEXT, ["Call Zelmira Quenby at 312-555-0199 before dismissal", "mail zq.x@district.edu now",
                                "SSN 900-12-3456 on file"])
    assert "3125550199" in terms.phone_digits and "zq.x@district.edu" in terms.emails
    assert "900123456" in terms.ssn_digits
    idx = leaks.SensitiveIndex()
    idx.origins.append(leaks.Origin("f.csv", None, "notes", leaks.FREE_TEXT))
    idx._merge_terms(terms, 0)
    scanner = leaks.DataScanner(idx, leaks.Allow())
    rules = {r for _, r, _ in scanner.distinctive(["reach her on (312) 555 0199"])}
    assert "phone digits" in rules
    rules = {r for _, r, _ in scanner.distinctive(["cc ZQ.X@district.edu"])}
    assert "email" in rules


# --------------------------------------------------------------------------- end to end (the codex fixtures)
CASES = {
    "numeric_notes": ({"Notes": cycle(PHONES)}, "Notes"),
    # D35: rare prose (3 students each) is never copied; the phone in PROSE[0] is never kept whatever its count
    "prose_pass": ({"accommodation": [PROSE[i % 3] if i < 9 else "None" for i in range(120)]}, "accommodation"),
    "free_text_constant": ({"accommodation": [PROSE[0]] * 120}, "accommodation"),
    "multi_emails": ({"cc": cycle(EMAIL_LISTS)}, "cc"),
    "multi_emails_recipients": ({"recipients": cycle(EMAIL_LISTS)}, "recipients"),
    "multi_names": ({"participants": cycle(["Zelmira Quenby;Althea Voss", "Althea Voss;Floriana Drexel",
                                            "Floriana Drexel;Zelmira Quenby"])}, "participants"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_end_to_end_no_real_value_reaches_output_or_report(tmp_path: Path, case: str) -> None:
    cols, col = CASES[case]
    src = write_csv(tmp_path / "in" / "roster.csv", {"student_id": ids(), **cols, "grade": cycle(["5", "6", "7"])})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert_absent(REAL_WORDS, out, rep, console=res.output)
    rows = read_dicts(out / "roster.csv")
    assert len(rows) == 120 and list(rows[0]) == ["student_id", col, "grade"]
    assert file_report(rep, "roster.csv")["checks"]["leak_scan"]["result"] == "PASS"
    # determinism: the same seed gives byte-identical output
    res2, out2, _ = run(src.parent, tmp_path, name="again")
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)


def test_end_to_end_native_storage(tmp_path: Path) -> None:
    n = 60
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({
        "student_id": ids(n),
        "Notes": pa.array([int(p) for p in cycle(PHONES, n)], pa.int64()),
        "contact_pref": pa.array(cycle(["Phone", "Text", "zelmira.quenby@gmail.com"], n)).dictionary_encode(),
        "cc": pa.array(cycle(EMAIL_LISTS, n)),
    }), src / "a.parquet")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    t = pq.read_table(out / "a.parquet")
    assert t.schema.field("Notes").type == pa.int64()
    assert not {str(v) for v in t.column("Notes").to_pylist()} & set(PHONES)
    assert_absent(REAL_WORDS, out, rep, console=res.output)


# --------------------------------------------------------------------------- false refusals
def test_ordinary_school_labels_stay_real_levels(tmp_path: Path) -> None:
    n = 120
    cols = {
        "student_id": ids(n),
        "lunch": cycle(["Free/Reduced Lunch", "Paid", "Free"], n),
        "parental level of education": cycle(["some college", "associate's degree", "high school",
                                              "some high school", "bachelor's degree", "master's degree"], n),
        "Course": cycle(["AP English Language and Composition", "Algebra I Honors",
                         "Introduction to the History of the United States"], n),
        "programs": cycle(["ELL;SPED", "SPED;504", "ELL", "GT;ELL"], n),
        "survey": cycle(["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree"], n),
        "score_band": cycle(["85-90", "90-95", "95-100"], n),
        "activity": cycle(["Grade 12 Food Drive", "5K Fun Run", "K-2 Literacy Path"], n),
        "room": cycle(["Room 101, 2nd Floor", "Gym", "Library"], n),
        "district_code": cycle(PHONES, n),
        "Note_Count": cycle(["0", "1", "2", "3"], n),
        "Alert_Level": cycle(["1", "2", "3"], n),
        "accommodations": cycle(["Extended Time;Read Aloud", "Small Group;Extended Time", "Read Aloud"], n),
        "incident": cycle(["Fighting - Mutual Combat", "Disruptive Behavior", "ISS - 1 day", "Tardy (3+)"], n),
        "bus_route": cycle(["Route 12", "Route 7", "Walker"], n),
        "mark": cycle(["A-", "B+", "C", "P", "NG"], n),
    }
    src = write_csv(tmp_path / "in" / "school.csv", cols)
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    rows = read_dicts(out / "school.csv")
    for col, values in cols.items():
        if col == "student_id":
            continue
        if col == "district_code":
            # R6-01: a bare 10-digit North American number is a phone shape in any text unit, whatever the header,
            # so these phone-like codes get consistent same-shape pseudonyms (counts kept), never their real values
            assert not {r[col] for r in rows} & set(values), col
            assert sorted(Counter(r[col] for r in rows).values()) == sorted(Counter(values).values()), col
            continue
        assert {r[col] for r in rows} == set(values), col  # real levels kept
        codes = [m["code"] for m in column_report(rep, "school.csv", col)["messages"]]
        assert "pii_values" not in codes and "prose_levels" not in codes, col


def test_example_data_detection_unchanged() -> None:
    res = run_cli("inspect", EXAMPLES, "--json")
    assert res.exit_code == 0
    report = json.loads(res.output)
    types = {}
    for f in report["files"]:
        for t in f["tables"]:
            for c in t["columns"]:
                types[(f["source"], c["name"])] = c["type"]
                codes = [m["code"] for m in c["messages"]]
                assert "pii_values" not in codes and "prose_levels" not in codes
    assert not [k for k, v in types.items() if v in ("HIGH_CARD_CATEGORICAL", "FREE_TEXT")]
    assert types[("StudentsPerformance.csv", "lunch")] == "FACTOR"
    assert types[("students_data.csv", "Course")] == "ORG_CODE"
    assert types[("student-scores.csv", "career_aspiration")] == "FACTOR"


def test_pure_prose_levels_follow_d35() -> None:
    # D35 superseded D33 (tests/test_r3_prose_d33.py): PII-free prose stays a factor and follows the allowlist
    # (rare levels get pseudonyms, levels shared by >= --min-cell entities are kept); no content warning
    for values, header in ((cycle(PROSE[1:]), "accommodation"),
                           (cycle(["Advanced Placement Computer Science Principles", "International Baccalaureate "
                                   "Diploma", "Career and Technical Education Pathway"]), "program_track")):
        det = detect(values, header)
        assert det.type is T.FACTOR and det.params.get(dtext.PENDING_KEY)
        assert not {"prose_levels", "sentence_levels"} & {m.code for m in det.messages}


def test_explicit_force_factor_keeps_real_values() -> None:
    det = detect(cycle(["Phone", "Text", "zelmira.quenby@gmail.com"]), "contact_pref",
                 force_factor=["contact_pref"])
    assert det.type is T.FACTOR and "pii_values" not in [m.code for m in det.messages]
