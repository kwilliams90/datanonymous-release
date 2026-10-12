"""Round-4 adversarial review, copy-gate cluster (R4-01, R4-02, R4-03): one type-agnostic copy gate.

Bug class (rounds 3 and 4): a path that writes real source values verbatim was not covered by the privacy gate.
R3 found INTEGER notes, FACTOR prose and MULTI_SELECT e-mails; R4 found MULTI_SELECT staff names written
``Smith, J.`` (the comma made them multi-select items, and the name backstop never looked at multi-selects:
R4-01), phone numbers under an ``SMS`` header typed as discrete INTEGER levels (``SMS`` was no phone header:
R4-02), and an Arrow ordered dictionary of sentences naming people kept as ORDINAL levels (D33 only looked at
CONSTANT/FLAG/FACTOR: R4-03).

The fix is structural (``detect.text.copy_gate``, whose docstring lists every path that writes real values and
the rule it follows):

* every value-copying detection - levels of CONSTANT, FLAG_SINGLE_TOKEN, FACTOR and native ORDINAL
  dictionaries, MULTI_SELECT values and items, NUMERIC_WITH_TOKENS tokens, numbers - goes through the same
  gate. Since D35 the gate is an allowlist for text values: a value is kept only when it is a vocabulary word or
  held by >= ``--min-cell`` entities with no personal-data shape; every other value gets a same-shape
  pseudonym (the layout ``Surname, I.`` or ``Mr. Surname`` is kept, the name is not). The person-name and
  sentence screens of R4-01/R4-03 no longer judge values (a name shared by >= k entities under a header that
  hints no personal data is D35's accepted residual risk); the name layouts stay for header screens (D34);
* the phone/contact header vocabulary is one structural rule (``hints.phone_header_hint``): ``'phone'``
  headers (``Parent_Cell``, ``Tel_No``, ``WhatsApp``, ``SMS_Number``) are PII hints that win over numeric
  typing; ``'contact'`` headers (``SMS``, ``Text``, ``Home``, ``Primary``) make phone-number values a PHONE
  column and never copy digit-only phone numbers, while their ordinary labels stay;
* the leak scanner excuses a real level only for a column the gate cleared (``detect.text.copy_cleared``): the
  scanner's excuse list and the gate share one source of truth.

False-refusal tests pin that ordinary school data keeps its real values: example-data, every
``tests/fixtures`` folder, org codes (``Smith Hall`` under ``Building``), short codes, Likert, grade levels,
``Building A`` / ``Day A`` labels, phoneme scores, text levels. Assertion messages never print a real value
beyond the invented fixture data.
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
import pyarrow.feather as pf
import pyarrow.parquet as pq
import pytest

from conftest import EXAMPLES, REPO, read_dicts, run_cli
from datanonymous import hints, pipeline
from datanonymous.config import Settings
from datanonymous.detect import names
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.privacy import leaks

from test_area_a_support import detect, profile_of, run_table

STAFF = ["Voss, A.", "Quenby, Z.", "Drexel, F."]  # unknown surnames: only the layout says "person"
STAFF_KNOWN = ["Smith, J.", "Garcia, M.", "Jones, A."]
PHONES = ["3125550187", "7735550163", "8475550182"]
SENTENCES = ["Please return medication to Althea Voss after the last class",
             "Please send the forms to Ysabel Quispe before the next class"]
REAL_WORDS = ("Voss", "Quenby", "Drexel", "Althea", "Ysabel", "Quispe", "Smith, J.", "Garcia, M.", "Jones, A.",
              *PHONES)


def cycle(values: list[Any], n: int = 60) -> list[Any]:
    return [values[i % len(values)] for i in range(n)]


def ids(n: int = 60) -> list[str]:
    return [f"S{i + 1000}" for i in range(n)]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "42"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def texts(*dirs: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", errors="replace")
                     for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file())


def found(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> int:
    blob = texts(*dirs) + "\n" + console
    return sum(1 for w in words if w in blob)


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    data = json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
    for t in data["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def leak_result(rep: Path, name: str) -> str:
    data = json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
    return data["checks"]["leak_scan"]["result"]


def codes(det: Any) -> list[str]:
    return [m.code for m in det.messages]


# =========================================================================== R4-02: phone headers, structural
@pytest.mark.parametrize("header", [
    "Phone", "Phones", "Home_Phone", "HomePhone", "homephone", "phonenumber", "Phone_No", "Telephone", "Tel",
    "Tel_No", "TelNo", "Parent_Tel", "Cell", "Cell_No", "CellNumber", "Parent_Cell", "Guardian1_Cell",
    "Mom_Cell", "Cell_Phone", "Cellular", "Mobile", "Mobile_No", "mobileno", "Fax", "Pager", "WhatsApp",
    "Whatsapp_Number", "SMS_Number", "SMSNumber", "smsno", "Txt_No", "Home_Ph", "Student_Ph", "Ph_No",
    "Mob_No", "Contact_Number", "Emergency_Contact_Number", "Day_Phone", "dayphone", "Phone_Primary",
    "Phone_2", "Phone_Ext", "Guardian1_Phone", "Guardian_Contact_Number", "Parent_Contact_No", "Work_Phone",
    "Evening_Phone", "Telephone_Number", "Cell #",
])
def test_phone_headers(header: str) -> None:
    assert hints.phone_header_hint(header) == "phone", header
    assert hints.header_hints(header).pii == "phone", header


@pytest.mark.parametrize("header", [
    "SMS", "Sms", "SMS_Contact", "Parent_SMS", "Txt", "Text", "Parent_Text", "Text_Number", "Home", "Home_2",
    "Work", "Primary", "Alternate", "Business", "Evening", "Emergency", "Home_Number", "Work_No",
    "Emergency_Number", "Phone_Type", "Cell_Carrier", "SMS_Opt_In", "Cell_Biology", "Alt_Text",
])
def test_contact_headers(header: str) -> None:
    assert hints.phone_header_hint(header) == "contact", header
    assert hints.header_hints(header).pii != "phone", header  # labels under them stay labels


@pytest.mark.parametrize("header", [
    "Phoneme_Segmentation", "PSF_Phoneme", "Phonemic_Awareness", "Phonics", "Saxophone", "Microphone_Count",
    "Text_Level", "Informational_Text", "Textbook", "Context", "Day_Number", "Primary_Language",
    "Home_Language", "Home_School", "Homework", "Contact_Notes", "pH", "pH_Level", "Graph", "Hotel",
    "Mobility", "Has_Cell", "Num_Phones", "No_Phone", "Cells", "score", "Grade",
])
def test_other_headers_are_no_phone_headers(header: str) -> None:
    assert hints.phone_header_hint(header) is None, header
    assert hints.header_hints(header).pii != "phone", header


@pytest.mark.parametrize("header", ["SMS", "Text", "Txt", "Home", "Primary", "Parent_Cell", "Tel_No", "WhatsApp",
                                    "Pager", "Mobile_No", "Home_Ph", "Cell_Number", "Work_Number", "SMS_Number"])
@pytest.mark.parametrize("values", [cycle(PHONES), ["(312) 555-0187", "773-555-0163", "847.555.0182"] * 20])
def test_phone_numbers_under_phone_and_contact_headers_are_phones(header: str, values: list[str]) -> None:
    det = detect(values, header)
    assert det.type is T.PHONE, (header, det.type)


@pytest.mark.parametrize("header", ["SMS", "Text", "Home", "Parent_Cell", "Tel_No"])
def test_phone_numbers_mixed_with_labels_are_never_copied(header: str) -> None:
    values = cycle(PHONES, 50) + ["Opted out"] * 10
    det = detect(values, header)
    assert det.type in (T.HIGH_CARD_CATEGORICAL, T.FREE_TEXT, T.PHONE, T.FACTOR), (header, det.type)
    tp, cols = run_table({"student_id": ids(), header: values})
    out = cols[header].tolist()
    assert not set(out) & set(PHONES)
    if det.type is T.FACTOR:  # D35: the phone numbers get pseudonyms on exchange 555; the label is kept
        assert set(dtext.pseudonymized(profile_of(tp, header))) == set(PHONES) and out.count("Opted out") == 10
        assert all(v[3] in "23456789" for v in out if v != "Opted out")


@pytest.mark.parametrize("values", [[p + ".0" for p in cycle(PHONES)], cycle(["07700900123", "07700900456",
                                                                              "07700900789"])])
def test_float_storage_and_leading_zero_phones_under_sms(values: list[str]) -> None:
    tp, cols = run_table({"student_id": ids(), "SMS": values})
    p = profile_of(tp, "SMS")
    assert p.type in (T.PHONE, T.HIGH_CARD_CATEGORICAL), p.type
    assert not set(cols["SMS"].tolist()) & set(values)


@pytest.mark.parametrize("header,values,expected", [
    ("Phoneme_Segmentation_Fluency", [str(10 + i % 50) for i in range(60)], T.INTEGER),
    ("Text_Level", cycle(["J", "K", "L", "M", "N"]), T.FACTOR),
    ("Day_Number", [str(1 + i % 180) for i in range(60)], T.INTEGER),
    ("Primary_Language", cycle(["English", "Spanish", "Vietnamese"]), T.FACTOR),
    ("Work", cycle(["Full time", "Part time", "None"]), T.FACTOR),
    ("Home", cycle(["Own", "Rent", "Shelter"]), T.FACTOR),
    ("SMS", cycle(["Opted in", "Opted out", "Unknown"]), T.FACTOR),
    ("SMS_Opt_In", cycle(["Y", "N"]), T.BOOLEAN),
    ("Phone_Type", cycle(["Home", "Cell", "Work"]), T.FACTOR),
    ("Cell_Biology", [str(55 + i % 40) for i in range(60)], T.INTEGER),
    ("Text", cycle(["Literary", "Informational", "Poetry"]), T.FACTOR),
    ("Primary", cycle(["Yes", "No"]), T.BOOLEAN),
])
def test_contact_and_look_alike_headers_keep_ordinary_values(header: str, values: list[str], expected: T) -> None:
    det = detect(values, header)
    assert det.type is expected, (header, det.type)
    assert "pii_values" not in codes(det) and "pii_hint" not in codes(det)


@pytest.mark.parametrize("header", ["SSN", "Social_Security_Number", "Social_Sec_No", "Soc_Sec_Num", "SS_Number"])
def test_ssn_headers_win_over_numeric_typing(header: str) -> None:
    assert hints.header_hints(header).pii == "ssn", header
    values = [str(900120000 + 7919 * i) for i in range(60)]
    tp, cols = run_table({"student_id": ids(), header: values})
    assert profile_of(tp, header).type in (T.SSN, T.HIGH_CARD_CATEGORICAL)
    assert not set(cols[header].tolist()) & set(values)


def test_numbers_under_a_phone_header_win_over_numeric_typing() -> None:
    det = detect([str(1 + i % 5) for i in range(60)], "Parent_Cell")  # not phone numbers: still never copied
    assert det.type is T.HIGH_CARD_CATEGORICAL and det.subtype == "code"


def test_sms_end_to_end_csv_and_native_int64(tmp_path: Path) -> None:
    """The R4-02 reproduction: 60 rows, SMS numbers each held 20 times, PASS before the fix."""
    src = write_csv(tmp_path / "in" / "sms.csv", {"student_id": ids(), "SMS": cycle(PHONES),
                                                  "score": [f"{33.125 + i * 1.718:.3f}" for i in range(60)]})
    pq.write_table(pa.table({"student_id": ids(), "SMS": pa.array([int(p) for p in cycle(PHONES)], pa.int64()),
                             "Text": pa.array([int(p) for p in cycle(PHONES)], pa.int64())}),
                   src.parent / "native.parquet")
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(PHONES, out, rep, console=res.output) == 0
    assert column_report(rep, "sms.csv", "SMS")["type"] == "PHONE"
    t = pq.read_table(out / "native.parquet")
    for col in ("SMS", "Text"):
        assert t.schema.field(col).type == pa.int64()
        vals = t.column(col).to_pylist()
        assert len(vals) == 60 and not {str(v) for v in vals} & set(PHONES)
    assert leak_result(rep, "sms.csv") == "PASS" and leak_result(rep, "native.parquet") == "PASS"
    res2, out2, _ = run(src.parent, tmp_path, name="again")
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)


def test_contact_header_phones_are_indexed_by_the_leak_scanner(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "s.csv", {"student_id": ids(), "SMS": cycle(PHONES)})
    run_ = pipeline.execute(Settings(mode="fidelity", seed=3), src.parent, tmp_path / "out", tmp_path / "rep")
    fr = run_.files[0]
    assert leaks.sensitive_kind(profile_of(fr.tables[0], "SMS")) == leaks.PHONE
    data = leaks.DataScanner(leaks.SensitiveIndex.build(run_.files), leaks.build_allow(run_))
    assert data.distinctive([f"call {PHONES[0][:3]}-{PHONES[0][3:6]}-{PHONES[0][6:]}"])


# =========================================================================== R4-01: names written with initials
@pytest.mark.parametrize("value,kind", [
    ("Smith, J.", "initials"), ("Voss, A.", "initials"), ("Garcia Lopez, M. A.", "initials"),
    ("Smith, J", "initials"), ("J. Smith", "initials"), ("J.K. Quenby", "initials"), ("Quenby Z.", "initials"),
    ("Mr. Smith", "title"), ("Mrs Quenby", "title"), ("Dr. Drexel", "title"), ("Coach Okafor", "title"),
])
def test_name_layouts(value: str, kind: str) -> None:
    assert names.name_layout_kind(value) == kind
    assert names.classify_value(value) == names.DISTINCTIVE


@pytest.mark.parametrize("value", ["Building A", "Day A", "Plan B", "Group C", "Algebra, A", "Vitamin D",
                                   "Room 12", "Mr. 12", "Period 3, A.", "smith, j.", "SPED, ELL", "Team B"])
def test_label_layouts_are_not_names(value: str) -> None:
    assert names.name_layout_kind(value) is None


@pytest.mark.parametrize("values", [STAFF, STAFF_KNOWN, ["Z. Quenby", "A. Voss", "F. Drexel"],
                                    ["Mrs. Quenby", "Mr. Voss", "Ms. Drexel"], ["Quenby Z.", "Voss A.", "Drexel F."],
                                    ["Mr. Brown", "Mrs. White", "Ms. Green"]])
def test_initial_and_title_layouts_make_a_name_verdict(values: list[str]) -> None:
    assert names.name_value_verdict({v: 20 for v in values}) is not None


#: R4-01's value-name detection is retired by D35: names are never judged by their content. A person header
#: (D12) replaces every value; under any other header a name shared by >= --min-cell entities is kept (the
#: accepted residual risk) and a rare one gets a same-shape pseudonym (its layout kept).
NAME_VALUES = [STAFF, STAFF_KNOWN, ["Z. Quenby", "A. Voss", "F. Drexel"], ["Mrs. Quenby", "Mr. Voss", "Ms. Drexel"],
               ["Smith 101", "Garcia 104", "Nguyen 210"],
               ["Quenby, Z. - Rm 101", "Voss, A. - Rm 104", "Drexel, F. - Rm 210"],
               ["101 - Patel", "104 - Okafor", "210 - Kowalski"]]


def _rare(values: list[str], n: int = 60, other: str = "Library") -> list[str]:
    """Each of ``values`` held by 3 students, ``other`` (an ordinary frequent label) by the rest."""
    return [values[i % len(values)] if i < 3 * len(values) else other for i in range(n)]


@pytest.mark.parametrize("header", ["Homeroom", "Case_Manager", "Advisory", "Award_Recipient", "Assigned",
                                    "Team_Captain", "Club_President", "Scholarship_Winner"])
@pytest.mark.parametrize("values", [STAFF, STAFF_KNOWN])
def test_comma_names_follow_the_header_or_the_allowlist(header: str, values: list[str]) -> None:
    det = detect(cycle(values), header)
    assert "value_names" not in codes(det) and (det.type is not T.PERSON_NAME or det.params.get("pii_hint"))
    tp, cols = run_table({"student_id": ids(), header: _rare(values)})
    out = cols[header].tolist()
    assert not set(out) & set(values)
    if profile_of(tp, header).type is T.FACTOR:  # (a person header replaces every value, D12)
        assert out.count("Library") == 51


@pytest.mark.parametrize("values", NAME_VALUES)
def test_rare_names_keep_their_layout_under_pseudonyms(values: list[str]) -> None:
    tp, cols = run_table({"student_id": ids(), "Homeroom": _rare(values)})
    p = profile_of(tp, "Homeroom")
    assert p.type is T.FACTOR and set(dtext.pseudonymized(p)) == set(values)
    fakes = Counter(v for v in cols["Homeroom"].tolist() if v != "Library")
    assert sorted(fakes.values()) == [3, 3, 3]
    real_words = {w.strip(",.").casefold() for v in values for w in v.split() if w[0].isalpha() and len(w) > 2}
    assert not {w.strip(",.").casefold() for v in fakes for w in v.split()} & real_words
    def shape(v: str) -> str:
        return "".join("A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else c for c in v)

    assert sorted(map(shape, fakes)) == sorted(map(shape, values))


@pytest.mark.parametrize("value,blanks", [("Quenby, Z.", False), ("Mrs. Quenby", True), ("Smith, J.", False)])
def test_single_value_names_follow_the_allowlist(value: str, blanks: bool) -> None:
    # a constant (or flag) held by >= --min-cell students is kept (residual); held by fewer, pseudonymised
    for n, kept in ((60, True), (8, False)):
        values = [value] * (n - n // 6) + ([""] * (n // 6) if blanks else [value] * (n // 6))
        tp, cols = run_table({"student_id": ids(n), "Advisory": values})
        assert (value in cols["Advisory"].tolist()) is kept, n


PEOPLE = ["Ana Garcia", "Luis Smith", "Maria Lopez", "Zelmira Quenby", "Althea Voss", "Floriana Drexel",
          "Ysabel Quispe", "Tess Okonkwo", "Quillon Varga", "Juan Perez"]


@pytest.mark.parametrize("d,style", [("; ", "full"), ("|", "surname"), ("; ", "initials")])
def test_lists_of_rare_people_have_every_item_pseudonymised(d: str, style: str) -> None:
    def item(name: str) -> str:
        first, last = name.split()
        return {"full": name, "surname": last, "initials": f"{last}, {first[0]}."}[style]

    lists = [d.join([item(PEOPLE[i % 10]), item(PEOPLE[(i + 3) % 10])]) if i < 30 else "None" for i in range(60)]
    det = detect(lists, "Committee")
    assert det.type is T.MULTI_SELECT and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"student_id": ids(), "Committee": lists})
    p = profile_of(tp, "Committee")
    assert leaks.sensitive_kind(p) == leaks.LABEL
    sep = p.params["delimiter"]
    real_items = {it.strip().casefold() for v in lists for it in v.split(sep) if v != "None"}
    assert set(map(str.casefold, dtext.pseudonymized(p))) == real_items  # each person in 6 rows: rare
    syn = cols["Committee"].tolist()
    syn_items = {it.strip().casefold() for v in syn for it in v.split(sep) if v != "None"}
    assert len(syn) == 60 and not syn_items & real_items and syn.count("None") == 30
    assert Counter(v.count(sep) for v in syn) == Counter(v.count(sep) for v in lists)  # items per value kept


def test_native_ordered_dictionary_of_rare_names(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    idx = [i % 3 + 1 if i < 9 else 0 for i in range(60)]
    arr = pa.DictionaryArray.from_arrays(pa.array(idx, pa.int8()), pa.array(["Library", *STAFF]), ordered=True)
    pf.write_feather(pa.table({"student_id": ids(), "Homeroom": arr}), src / "h.feather")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    c = column_report(rep, "h.feather", "Homeroom")
    assert c["type"] == "ORDINAL" and c["params"]["pseudonymized_levels"] == 3
    back = pf.read_table(out / "h.feather")["Homeroom"]
    vals = back.to_pylist()
    assert len(vals) == 60 and not set(vals) & set(STAFF) and vals.count("Library") == 51 and back.type.ordered
    assert found(("Voss", "Quenby", "Drexel"), out, rep, console=res.output) == 0


@pytest.mark.parametrize("token", ["Quenby, Z.", "Mrs. Voss", "Maria Garcia"])
def test_name_tokens_of_a_number_column_are_masked(token: str) -> None:
    # D35: a rare token among numbers gets a same-shape pseudonym; a vocabulary token ('Absent') is kept
    values = [str(i) for i in range(40)] + [token] * 5 + ["Absent"] * 5
    det = detect(values, "visits")
    assert det.type is T.NUMERIC_WITH_TOKENS and det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"visits": values})
    assert set(dtext.pseudonymized(profile_of(tp, "visits"))) == {token}
    syn = Counter(cols["visits"].tolist())
    assert syn[token] == 0 and syn["Absent"] == 5  # the name masked at its count, the plain token kept


@pytest.mark.parametrize("header,values", [
    ("Homeroom", ["Room 101", "Room 104", "Gym"]), ("Lunch", ["Hall 1", "Hall 2", "Annex 3"]),
    ("activity", ["Grade 12 Food Drive", "5K Fun Run", "K-2 Literacy Path"]), ("bus_route", ["Route 12", "Route 7"]),
    ("Homeroom", ["101", "104", "210"]), ("period", ["Period 1", "Period 2", "Period 3"]),
])
def test_room_and_number_labels_stay_labels(header: str, values: list[str]) -> None:
    det = detect(cycle(values), header)
    assert det.type is not T.PERSON_NAME and "value_names" not in codes(det)


def test_stata_value_labels_of_names(tmp_path: Path) -> None:
    import pandas as pd

    src = tmp_path / "in"
    src.mkdir()
    df = pd.DataFrame({"student_id": ids(), "advisor": [i % 3 + 1 for i in range(60)]})
    df.to_stata(src / "a.dta", write_index=False, version=118,
                value_labels={"advisor": {1: STAFF[0], 2: STAFF[1], 3: STAFF[2]}})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert column_report(rep, "a.dta", "advisor")["type"] == "PERSON_NAME"
    assert found(("Voss", "Quenby", "Drexel"), out, rep, console=res.output) == 0


@pytest.mark.parametrize("token,name", [
    ("Quillon Garcia", True), ("Tess Okonkwo-Lind", True), ("Smith, J.", True), ("Maria Garcia", True),
    ("Absent Excused", False), ("Medical Exempt", False), ("Not Tested", False), ("Home School", False),
    ("Did Not Attend", False), ("Transfer Pending", False), ("Exempt", False), ("Garcia", False),
    ("Quillon Varga", False),  # two words on no name list: no evidence (as R3-04's rare names alone)
])
def test_names_among_numbers(token: str, name: bool) -> None:
    assert names.stray_name(token) is name
    assert ("name" in dtext.token_kinds(token)) is name


def test_r4_01_end_to_end(tmp_path: Path) -> None:
    """The R4-01 reproduction with names held by 3 students each (D35: rare values never copied; a name held by
    >= --min-cell students under a header that hints no personal data is the accepted residual risk)."""
    d = tmp_path / "in"
    for header in ("Homeroom", "Case_Manager", "Advisory", "Award_Recipient"):
        write_csv(d / f"{header}.csv", {"student_id": ids(), header: _rare(STAFF_KNOWN),
                                        "score": [str(i % 11 + 30) for i in range(60)]})
        write_csv(d / f"{header}_rare.csv", {"student_id": ids(), header: _rare(STAFF),
                                             "score": [str(i % 11 + 30) for i in range(60)]})
    res, out, rep = run(d, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(REAL_WORDS[:9], out, rep, console=res.output) == 0
    for f in sorted(d.iterdir()):
        rows = read_dicts(out / f.name)
        col = f.stem.removesuffix("_rare")
        assert len(rows) == 60 and all(", " in r[col] and r[col].endswith(".") for r in rows
                                       if r[col] != "Library")  # layout kept
        assert leak_result(rep, f.name) == "PASS"
    res2, out2, _ = run(d, tmp_path, name="again")
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)


# =========================================================================== R4-03: ordered dictionaries
@pytest.mark.parametrize("fmt", ["feather", "parquet"])
def test_ordered_dictionary_rare_sentences_are_pseudonymised(tmp_path: Path, fmt: str) -> None:
    """The R4-03 reproduction (an ordered Arrow dictionary of sentences naming people) with each sentence held
    by 3 students: D35 writes their pseudonyms in the entries' places; the column stays ORDINAL."""
    src = tmp_path / "in"
    src.mkdir()
    arr = pa.DictionaryArray.from_arrays(pa.array([i % 2 + 1 if i < 6 else 0 for i in range(60)], pa.int8()),
                                         pa.array(["Library", *SENTENCES]), ordered=True)
    table = pa.table({"student_id": ids(), "Reason": arr})
    path = src / f"notes.{fmt}"
    (pf.write_feather if fmt == "feather" else pq.write_table)(table, path)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    c = column_report(rep, path.name, "Reason")
    assert c["type"] == "ORDINAL" and "pseudonymized_levels" in [m["code"] for m in c["messages"]]
    back = (pf.read_table if fmt == "feather" else pq.read_table)(out / path.name)
    assert pa.types.is_dictionary(back.schema.field("Reason").type) and back.schema.field("Reason").type.ordered
    vals = back["Reason"].to_pylist()
    assert len(vals) == 60 and not set(vals) & set(SENTENCES)
    assert found(("Althea", "Voss", "Ysabel", "Quispe", "medication"), out, rep, console=res.output) == 0
    assert leak_result(rep, path.name) == "PASS"


@pytest.mark.parametrize("levels,warning", [
    (["Strongly disagree", "Disagree", "Agree", "Strongly agree"], None),
    (["Below Basic", "Basic", "Proficient", "Advanced"], None),
    (["Introduction to the History of the United States", "Advanced Placement United States History",
      "International Baccalaureate History of the Americas"], None),  # D35 retired the prose_levels warning
    (["Tier 1", "Tier 2", "Tier 3"], None),
    (["Lark", "Wren", "Finch"], None),
])
def test_ordered_dictionaries_of_labels_stay_ordinal(tmp_path: Path, levels: list[str], warning: str | None) -> None:
    arr = pa.DictionaryArray.from_arrays(pa.array([i % len(levels) for i in range(60)], pa.int8()),
                                         pa.array(levels), ordered=True)
    src = tmp_path / "in"
    src.mkdir()
    pf.write_feather(pa.table({"student_id": ids(), "level": arr}), src / "o.feather")
    run_ = pipeline.execute(Settings(mode="fidelity", seed=2), src, tmp_path / "out", tmp_path / "rep")
    assert pipeline.exit_code(run_) == 0
    p = profile_of(run_.files[0].tables[0], "level")
    assert p.type is T.ORDINAL and p.source == "native"
    msgs = [m.code for m in p.messages]
    if warning:
        assert warning in msgs
    assert "sentence_levels" not in msgs and "value_names" not in msgs
    assert p.params.get(dtext.COPY_GATE_KEY) == "allowlist" and dtext.copy_cleared(p)
    back = pf.read_table(tmp_path / "out" / "o.feather")["level"]
    assert set(back.to_pylist()) == set(levels) and back.type.ordered


def test_sentence_tokens_of_a_number_column_are_masked() -> None:
    sentence = "Moved to the other class after the first week"
    values = [str(i) for i in range(40)] + [sentence] * 3 + ["Exempt"] * 4
    det = detect(values, "score")
    assert det.params.get(dtext.PENDING_KEY)  # D35: rare tokens get pseudonyms ('Exempt' is vocabulary)
    tp, cols = run_table({"score": values})
    assert set(dtext.pseudonymized(profile_of(tp, "score"))) == {sentence}
    out = cols["score"].tolist()
    assert sentence not in out and out.count("Exempt") == 4


# =========================================================================== one gate for every copying path
def _every_kind_of_column(n: int = 120) -> dict[str, list[str]]:
    rng = np.random.default_rng(4)
    return {
        "student_id": [f"S{i:05d}" for i in range(n)],
        "gender": cycle(["F", "M", "X"], n),
        "school": cycle(["Lincoln Elementary", "Smith Hall Academy"], n),
        "Building": cycle(["Smith Hall", "Garcia Annex", "Main"], n),
        "flag": ["Y"] * (n - 20) + [""] * 20,
        "constant": ["2024"] * n,
        "passed": cycle(["true", "false"], n),
        "survey": cycle(["Agree", "Disagree", "Neutral"], n),
        "grade": cycle(["K", "1", "2", "3"], n),
        "sy": cycle(["2023-24", "2024-25"], n),
        "score": [str(int(x)) for x in rng.integers(0, 100, n)],
        "gpa": [f"{x:.2f}" for x in rng.uniform(1, 4, n)],
        "amount": [f"${x:,.2f}" for x in rng.uniform(1000, 9000, n)],
        "suppressed": [str(int(x)) for x in rng.integers(10, 90, n - 10)] + ["<10"] * 10,
        "enrolled": [f"2024-0{1 + i % 9}-1{i % 9}" for i in range(n)],
        "code": cycle(["001", "002", "017"], n),
        "zip": cycle(["60601", "60614", "02134"], n),
        "state": cycle(["IL", "MA", "NY"], n),
        "programs": cycle(["ELL;SPED", "SPED;504", "ELL"], n),
        "email": [f"kid{i}@example.org" for i in range(n)],
        "Homeroom": cycle(STAFF_KNOWN, n),
        "SMS": cycle(PHONES, n),
        "notes": cycle(["Call 312-555-0187 first", "Allergic to nuts"], n),
        "Reason": cycle(SENTENCES, n),
    }


def test_every_value_copying_column_passed_the_gate() -> None:
    tp, _ = run_table(_every_kind_of_column())
    copying = []
    for p in tp.columns:
        if dtext.copies_real_values(p):
            copying.append(p.name)
            assert dtext.copy_cleared(p), (p.name, p.type)
            assert p.params[dtext.COPY_GATE_KEY] in ("allowlist", "vocabulary", "numbers", "dates", "zip_policy")
        else:
            assert dtext.COPY_GATE_KEY not in p.params, p.name
    # D35: Homeroom and Reason keep their type; their values follow the allowlist (held by 40 / 60 rows: kept)
    assert {"gender", "school", "Building", "flag", "constant", "passed", "survey", "grade", "sy", "score", "gpa",
            "amount", "suppressed", "enrolled", "code", "zip", "state", "programs", "Homeroom",
            "Reason"} <= set(copying)
    for col in ("SMS", "notes", "email", "student_id"):
        assert col not in copying, col


def test_every_native_value_copying_column_passed_the_gate(tmp_path: Path) -> None:
    import datetime as dt
    from decimal import Decimal

    n = 60
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({
        "student_id": ids(n),
        "program": pa.array(cycle(["ELL", "SPED", "GT"], n)).dictionary_encode(),
        "level": pa.DictionaryArray.from_arrays(pa.array([i % 3 for i in range(n)], pa.int8()),
                                                pa.array(["Tier 1", "Tier 2", "Tier 3"]), ordered=True),
        "balance": pa.array([Decimal(f"{i + 3}.25") for i in range(n)], pa.decimal128(10, 2)),
        "enrolled": pa.array([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)], pa.date32()),
        "score": pa.array([i % 7 for i in range(n)], pa.int16()),
        "advisor": pa.array(cycle(STAFF, n)).dictionary_encode(),
        "Reason": pa.DictionaryArray.from_arrays(pa.array([i % 2 for i in range(n)], pa.int8()), pa.array(SENTENCES),
                                                 ordered=True),
    }), src / "n.parquet")
    run_ = pipeline.execute(Settings(mode="fidelity", seed=2), src, tmp_path / "out", tmp_path / "rep")
    assert pipeline.exit_code(run_) == 0
    tp = run_.files[0].tables[0]
    for p in tp.columns:
        assert not dtext.copies_real_values(p) or dtext.copy_cleared(p), (p.name, p.type)
    kinds = {p.name: (p.type, dtext.copies_real_values(p)) for p in tp.columns}
    assert kinds["advisor"] == (T.PERSON_NAME, False) and kinds["Reason"] == (T.ORDINAL, True)  # D35: allowlist
    for col in ("program", "level", "balance", "enrolled", "score"):
        assert kinds[col][1], col


def test_explicit_overrides_are_cleared_without_the_gate() -> None:
    det = detect(cycle(STAFF), "Homeroom", force_factor=["Homeroom"])
    assert det.type is T.FACTOR and det.source == "force-factor" and dtext.COPY_GATE_KEY not in det.params
    assert dtext.copy_cleared(det)
    tp, cols = run_table({"Homeroom": cycle(STAFF)}, force_factor=["Homeroom"])
    assert set(cols["Homeroom"].tolist()) == set(STAFF)


@pytest.mark.parametrize("header,values,expected", [
    ("Building", cycle(["Smith Hall", "Garcia Annex", "Jones Gym"]), T.ORG_CODE),  # D10: real values kept
    # D35 pass 1: a comma that splits every value into one pair of different parts is no list (pair_layout)
    ("Section", cycle(["Smith, J. - P1", "Garcia, M. - P2"]), T.ORG_CODE),  # org-code header: D10
    ("School", cycle(["Mrs. Quenby's Academy", "Dr. Voss Elementary"]), T.ORG_CODE),
    ("Lunch", cycle(["Day A", "Day B"]), T.FACTOR),
    ("Wing", cycle(["Building A", "Building B", "Building C"]), T.FACTOR),
    ("programs", cycle(["ELL;SPED", "SPED;504", "ELL", "GT;ELL"]), T.MULTI_SELECT),
    ("activities", cycle(["Chess, Robotics", "Drama, Band", "Chess"]), T.MULTI_SELECT),
    ("survey", cycle(["Strongly agree", "Agree", "Disagree"]), T.ORDINAL),
    ("grade_level", cycle(["PK", "KG", "01", "02"]), T.ORDINAL),
    ("course_section", cycle(["Algebra, A", "Biology, B", "Chemistry, C"]), T.ORG_CODE),
    ("Class", cycle(["Algebra, A", "Biology, B", "Chemistry, C"]), T.FACTOR),
    ("plan", cycle(["Plan A", "Plan B", "Plan C"]), T.FACTOR),
    ("race", cycle(["White", "Black", "Asian", "Hispanic"]), T.FACTOR),
    ("career_aspiration", cycle(["Lawyer", "Doctor", "Software Engineer", "Unknown"]), T.FACTOR),
    ("code", cycle(["AB-12", "CD-34", "EF-56"]), T.FACTOR),
])
def test_ordinary_school_values_keep_their_real_levels(header: str, values: list[str], expected: T) -> None:
    det = detect(values, header)
    assert det.type is expected, (header, det.type)
    assert not {"value_names", "pii_values", "sentence_levels"} & set(codes(det)), header
    assert "sensitive_items" not in det.params
    tp, cols = run_table({header: values})
    if header == "Section":  # D36: the teacher's name inside the code gets a fake name, the rest stays real
        written = set(cols[header].tolist())
        assert not written & set(values) and {w.rsplit(" - ", 1)[1] for w in written} == {"P1", "P2"}
        return
    assert set(cols[header].tolist()) == set(values), header


def test_example_data_output_and_detection_unchanged() -> None:
    res = run_cli("inspect", EXAMPLES, "--json")
    assert res.exit_code == 0
    for f in json.loads(res.output)["files"]:
        for t in f["tables"]:
            for c in t["columns"]:
                assert c["type"] not in ("PERSON_NAME", "PHONE") or c["name"] in ("first_name", "last_name"), (
                    f["source"], c["name"])
                assert not {"value_names", "pii_values", "sentence_levels"} & {m["code"] for m in c["messages"]}


@pytest.mark.parametrize("folder", sorted(p.name for p in (REPO / "tests" / "fixtures").iterdir() if p.is_dir()))
def test_fixtures_get_no_new_name_or_phone_conversions(folder: str) -> None:
    res = run_cli("inspect", REPO / "tests" / "fixtures" / folder, "--json")
    report = json.loads(res.stdout)
    hits = [(f.get("source"), c["name"]) for f in report.get("files", []) for t in f.get("tables") or []
            for c in t["columns"]
            if "contact_header" in c.get("params", {}) or {"sentence_levels"} & {m["code"] for m in c["messages"]}]
    assert not hits


# =========================================================================== the leak scanner shares the gate
@pytest.fixture(scope="module")
def staff_run(tmp_path_factory: pytest.TempPathFactory):
    """A real student name 'Zelmira Quenby' and an advisory group 'Quenby' (one unknown word: not judged a name,
    so the gate clears it and the report lists it)."""
    base = tmp_path_factory.mktemp("r4_gate_scan")
    # one name per student (D35 hardening pass 2: names one to one with student_id leave the count to the IDs)
    others = [f"{f} {s}" for f in ("Ana", "Luis", "Maria", "Juan", "Tess", "Ysabel") for s in
              ("Garcia", "Smith", "Lopez", "Perez", "Okonkwo", "Varga", "Drexel", "Voss", "Lark", "Moss")]
    src = write_csv(base / "in" / "roster.csv", {
        "student_id": ids(), "Student_Name": ["Zelmira Quenby", "Althea Voss", "Floriana Drexel"] + others[:57],
        "Advisory_Group": cycle(["Quenby", "Lark", "Wren"])})
    return base, src


def test_scanner_excuses_only_cleared_levels(staff_run) -> None:
    base, src = staff_run
    run_ = pipeline.execute(Settings(mode="fidelity", seed=5), src.parent, base / "out", base / "rep")
    assert pipeline.exit_code(run_) == 0
    tp = run_.files[0].tables[0]
    p = profile_of(tp, "Advisory_Group")
    assert p.type is T.FACTOR and dtext.copy_cleared(p)
    allow = leaks.build_allow(run_)
    assert "quenby" in allow.tokens  # a cleared real level: reproduced anyway (D21)
    assert leaks.cross_column_values(p, ["Quenby", "Lorem"]) == ["Lorem"]
    p.params.pop(dtext.COPY_GATE_KEY)  # a level that never passed the gate
    assert not dtext.copy_cleared(p)
    assert "quenby" not in leaks.build_allow(run_).tokens
    assert leaks.cross_column_values(p, ["Quenby", "Lorem"]) == ["Quenby", "Lorem"]


def test_a_path_that_skips_the_gate_fails_the_leak_scan(staff_run, monkeypatch: pytest.MonkeyPatch) -> None:
    base, src = staff_run
    res, out, rep = run(src.parent, base, name="gated")
    assert res.exit_code == 0, res.output
    assert leak_result(rep, "roster.csv") == "PASS"
    monkeypatch.setattr(dtext, "_clearance", lambda det: "")  # simulate a copying path the gate never stamped
    res, out, rep = run(src.parent, base, name="ungated")
    assert res.exit_code == 3, res.output
    assert not (out / "roster.csv").exists()  # D19: FAILed, not written
