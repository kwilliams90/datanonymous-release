"""Round-2 adversarial review, names/headers cluster (R2-04, R2-05, R2-06, R2-17).

Each round-1 fix covered the reproduced case only; these tests pin the whole bug class:

* R2-04: a person-role header with numbering or ranking (``Guardian1``, ``Contact #3``, ``Parent(s)``,
  ``Guardian_Primary``, ``Name_2``, ``Referred_By`` ...) was no name hint, so a small name column became a
  factor whose real levels were copied.
* R2-05: the headerless-name refusal judged the whole first line at once and only by letter case, so one
  upper-case cell (or one numeric, code or known-word cell) let a real person through as column names.
  Cells are now judged one by one; only a recognisable column name over different values vouches for the line.
* R2-06: the run-wide full-name index held word pairs only, so a single-word real full name (``Whitney``,
  ``Mr. Whitney``, a first name with a blank last name) could be written by another file.
* R2-17: a first record longer than the 64 KB sniffing sample (or a header that long) left no complete
  record to sniff, and a valid CSV failed as ragged.

Assertion messages never print a real value beyond the invented fixture names.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import openpyxl
import pytest

from conftest import read_dicts, run_cli
from datanonymous import hints
from datanonymous.generate import pii
from datanonymous.generate import text as gtext
from datanonymous.io import csvio, sniff

REAL = ("Zelmira", "Quenby", "Balthazar", "Vosswyn", "Mirelda", "Xaverton", "Lysander", "Wexforth")
NAME_KEYS = ("full_name", "first_name", "last_name", "middle_name")


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _texts(*dirs: Path) -> dict[str, str]:
    return {str(p): p.read_text(encoding="utf-8", errors="replace")
            for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()}


def _assert_absent(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> None:
    texts = _texts(*dirs)
    texts["console"] = console
    found = {name: [w for w in words if w in text] for name, text in texts.items()}
    found = {k: v for k, v in found.items() if v}
    assert not found, found


def _run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}_out", tmp / f"{name}_rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "123", *extra)
    return res, out, rep


def _hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def _columns(rep: Path, name: str) -> dict[str, dict]:
    data = json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in data["tables"][0]["columns"]}


# =========================================================================== R2-04 numbered person-role headers
@pytest.mark.parametrize("header", [
    "Guardian1", "Guardian_1", "Guardian 2", "guardian01", "GUARDIAN1", "Parent2", "Contact1", "Contact #3",
    "EmergencyContact2", "Teacher_2", "Counselor 3", "Parent 1st", "Parent(s)", "Guardian(s)", "Guardian_Primary",
    "Contact_Secondary", "Teacher_of_Record", "Parent_A", "Guardians", "Teachers", "Coaches", "Proctor2",
    "Name1", "Name_2", "Names", "MotherName2", "mother_name_02", "Guardian1Name", "Referred_By", "EnteredBy",
    "Entered By", "Tchr_Name", "Mom", "Pupil", "Teacher_Aide", "Witness_1", "Alias",
])
def test_numbered_and_ranked_role_headers_are_full_name_hints(header: str) -> None:
    assert hints.header_hints(header).pii == "full_name"


@pytest.mark.parametrize("header,key", [
    ("Name_First", "first_name"), ("NameLast", "last_name"), ("Name (Last, First)", "full_name"),
    ("Name_Last_First", "full_name"), ("Guardian_First_Primary", "first_name"), ("Guardian2First", "first_name"),
    ("Guardian_2_Last", "last_name"), ("First1", "first_name"), ("Last_2", "last_name"),
    ("Contact1_FirstName", "first_name"),
])
def test_name_part_headers_with_numbering(header: str, key: str) -> None:
    assert hints.header_hints(header).pii == key


@pytest.mark.parametrize("header", [
    "Guardian1_ID", "Guardian1_Relationship", "Num_Guardians", "Total_Students", "is_parent", "Has_Guardian2",
    "Guardian2_Flag", "Parent_Count", "Teacher_Code", "School_Name", "FileName1", "course_name_2", "Grade_A",
    "first_day", "last_login", "LastSchool", "Student_ID", "score1", "Q1", "Contact_Method", "Teacher_Type",
    "Sort_By", "Paid_By", "Student_New", "Primary", "Of_Record",
])
def test_other_headers_stay_unhinted(header: str) -> None:
    assert hints.header_hints(header).pii not in NAME_KEYS


def test_guardian1_email_and_id_keep_their_own_hints() -> None:
    assert hints.header_hints("Guardian1Email").pii == "email"
    assert hints.header_hints("Guardian1_Phone").pii == "phone"
    assert hints.header_hints("Guardian1_ID").person_id_like


@pytest.mark.parametrize("header", ["Guardian1", "Contact #3", "Parent(s)", "Teacher_of_Record", "Name_2",
                                    "Referred_By", "Guardian_Primary"])
def test_numbered_role_names_are_never_copied(tmp_path: Path, header: str) -> None:
    rows = [["Zelmira Quenby", 1], ["Balthazar Vosswyn", 2]] * 15
    src = _write(tmp_path / "in" / "data.csv", [header, "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "data.csv")
    assert len(syn) == 30 and len({r[header] for r in syn}) == 2  # N in = N out, distinct count kept
    _assert_absent(list(REAL[:4]), out, rep, console=res.output)
    assert _columns(rep, "data.csv")[header]["type"] == "PERSON_NAME"


# =========================================================================== R2-05 headerless name lists
def _cols(rows: list[list[str]]) -> list[list[str]]:
    return [list(c) for c in zip(*rows)]


NAMES = [["Balthazar", "Vosswyn"], ["Mirelda", "Xaverton"], ["Lysander", "Wexforth"]]


@pytest.mark.parametrize("header,rows", [
    (["Zelmira", "Quenby"], NAMES),
    (["ZELMIRA", "Quenby"], NAMES),  # the R2-05 reproduction: one cell in another letter case
    (["zelmira", "Quenby"], NAMES),
    (["ZELMIRA", "QUENBY"], NAMES),
    (["McDonald", "Quenby"], NAMES),
    (["Zelmira", "Quenby"], [["Balthazar", "123"], ["Mirelda", "456"]]),  # a numeric column does not vouch
    (["Zelmira", "Quenby", "12"], [["Balthazar", "Vosswyn", "15"], ["Mirelda", "Xaverton", "9"]]),
    (["Zelmira Quenby", "Teacher"], [["Balthazar Vosswyn", "Student"], ["Mirelda Xaverton", "Parent"]]),
    (["Mary Ann de la Cruz", "Quenby"], NAMES),  # more than three words
    (["Quenby, Zelmira"], [["Vosswyn, Balthazar"], ["Xaverton, Mirelda"]]),  # "Last, First"
    (["S1001", "Zelmira"], [["S1002", "Balthazar"], ["S1003", "Mirelda"]]),  # an ID column does not vouch
    (["S1001", "12"], [["S1002", "15"], ["S1003", "9"]]),  # a real ID as a column name
    (["Zelmira", "Price"], NAMES),  # a surname that is also a column-name word
    (["Price"], [["Vosswyn"], ["Xaverton"]]),
    (["Townsend", "Keyes"], NAMES),  # surnames that start like header hints (town, key)
    (["S1001", "Missed 3 days, called home twice"],
     [["S1002", "Late to class 4 times this week"], ["S1003", "Parent meeting on Friday at noon"]]),
    (["Zelmira", "Yes"], [["Balthazar", "No"], ["Mirelda", "Yes"]]),
    (["Zelmira", "NA"], NAMES),
    # a column-name word over a column of words or text can be a value: it never vouches
    (["Zelmira", "Quenby", "Math"], [["Balthazar", "Vosswyn", "Algebra II"], ["Mirelda", "Xaverton", "World History"]]),
    (["Zelmira", "Allergy"], [["Balthazar", "Severe peanut allergy, carries an epipen"],
                              ["Mirelda", "Asthma, inhaler in the nurse office"]]),
    (["Zelmira", "Notes"], [["Balthazar", ""], ["Mirelda", ""]]),
])
def test_headerless_first_records_are_refused(header: list[str], rows: list[list[str]]) -> None:
    prob = sniff.header_problems(header, _cols(rows))
    assert prob is not None and prob.record_like
    assert not any(h in prob.text() for h in header if h not in ("12", "NA"))  # positions only


@pytest.mark.parametrize("header,rows", [
    (["Region", "Program"], [["North", "Math"], ["South", "Reading"], ["East", "Math"]]),
    (["first_name", "last_name"], NAMES),
    (["FIRST", "LAST"], NAMES),
    (["Student", "Teacher"], [["Balthazar Vosswyn", "Mirelda Xaverton"], ["Lysander Wexforth", "Ana Ruiz"]]),
    (["Name", "Score"], [["Balthazar", "12"], ["Mirelda", "14"]]),
    (["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "math score"],
     [["female", "group B", "bachelor's degree", "standard", "none", "72"],
      ["male", "group C", "some college", "free/reduced", "completed", "69"]]),
    (["YearofStudy", "Course", "Age", "KCSE"],
     [["first year", "Bachelor of Science", "21", "B"], ["second year", "Bachelor of Arts", "22", "A"]]),
    (["Mascot", "Score"], [["Tigers", "12"], ["Eagles", "15"]]),  # vouched by a column name over numbers
    (["student_id", "Mascot"], [["S1001", "Tigers"], ["S1002", "Eagles"]]),  # ... or by an identifier shape
    (["StudentID", "Hometown"], [["1001", "Springfield"], ["1002", "Shelbyville"]]),
    (["Active"], [["TRUE"], ["FALSE"], ["TRUE"]]),
    (["Teacher"], [["Vosswyn"], ["Xaverton"]]),
    (["Timestamp", "What did you like about the course?"],
     [["2024-01-01 10:00", "The projects were fun and useful"], ["2024-01-02 11:00", "Group work"]]),
    (["Hair Color", "Age"], [["Brown", "12"], ["Light Brown", "13"]]),
    (["Home Language"], [["Spanish"], ["English"], ["Haitian Creole"]]),
    (["Q1", "Q2"], [["A", "B"], ["C", "D"]]),
])
def test_real_headers_are_still_headers(header: list[str], rows: list[list[str]]) -> None:
    assert sniff.header_problems(header, _cols(rows)) is None


@pytest.mark.parametrize("value,shape", [
    ("123 Main St", "street address"), ("60614-1234", "ZIP+4"), ("192.168.10.4", "IP address"),
    ("www.example.org", "web address"),
])
def test_personal_value_shapes_in_the_first_line(value: str, shape: str) -> None:
    prob = sniff.header_problems(["Zelmira", value], [["Balthazar"], ["x"]])
    assert prob is not None and shape in prob.text() and value not in prob.text()


@pytest.mark.parametrize("first", [["ZELMIRA", "Quenby"], ["Zelmira", "Quenby", "12"], ["Quenby, Zelmira", "B"]])
def test_headerless_csv_fails_without_exposing_a_person(tmp_path: Path, first: list[str]) -> None:
    rest = [["Balthazar", "Vosswyn", "15"], ["Mirelda", "Xaverton", "9"], ["Lysander", "Wexforth", "11"]]
    rows = [r[:len(first)] for r in rest]
    if first[0].startswith("Quenby"):
        rows = [["Vosswyn, Balthazar", "A"], ["Xaverton, Mirelda", "B"], ["Wexforth, Lysander", "A"]]
    src = _write(tmp_path / "in" / "names.csv", first, rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert not (out / "names.csv").exists()
    _assert_absent(list(REAL), out, rep, console=res.output)
    entry = next(f for f in json.loads((rep / "report.json").read_text(encoding="utf-8"))["files"]
                 if f["source"] == "names.csv")
    assert entry["status"] == "FAIL" and "--header-row 1" in json.dumps(entry)
    # the user can assert that the line is the header
    res, out, _ = _run(src, tmp_path, "--header-row", "1", name="asserted")
    assert res.exit_code == 0, res.output
    assert len(read_dicts(out / "names.csv")) == 3


def test_headerless_excel_name_list_fails(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in (["ZELMIRA", "Quenby"], ["Balthazar", "Vosswyn"], ["Mirelda", "Xaverton"], ["Lysander", "Wexforth"]):
        ws.append(row)
    (tmp_path / "in").mkdir()
    wb.save(tmp_path / "in" / "names.xlsx")
    res, out, rep = _run(tmp_path / "in", tmp_path)
    assert res.exit_code == 3, res.output
    assert not (out / "names.xlsx").exists()
    _assert_absent(list(REAL), out, rep, console=res.output)


# =========================================================================== R2-06 single-word full names
@pytest.mark.parametrize("value,key", [
    ("Whitney", "whitney"), ("Mr. Whitney", "whitney"), ("Whitney Jr", "whitney"), ("Smith Smith", "smith"),
    ("de la Cruz", "cruz"), ("J. K.", "#jk"), ("Mary Smith", None), ("Burton, Chelsea", None), ("Mrs.", None),
])
def test_whole_name_key(value: str, key: str | None) -> None:
    assert pii.whole_name_key(value) == key


def _two_runs(tmp_path: Path, people_header: list[str], people_rows: list[list[object]], other_header: list[str],
              make_other) -> tuple[list[dict], list[dict], Path, Path]:  # noqa: ANN001
    """Run people.csv alone, then again next to others.csv built from that first synthetic output."""
    d = tmp_path / "in"
    _write(d / "people.csv", people_header, people_rows)
    res, out1, _ = _run(d, tmp_path, name="alone")
    assert res.exit_code == 0, res.output
    first = read_dicts(out1 / "people.csv")
    _write(d / "others.csv", other_header, make_other(first))
    res, out2, rep2 = _run(d, tmp_path, name="together")
    assert res.exit_code == 0, res.output
    return first, read_dicts(out2 / "people.csv"), out2, rep2


@pytest.mark.parametrize("form", ["{}", "Mr. {}", "{} {}", "{} Jr", "{}, J."])
def test_single_word_full_name_of_another_file_is_never_written(tmp_path: Path, form: str) -> None:
    first, second, out, rep = _two_runs(
        tmp_path, ["full_name", "score"], [["Zelmira", i] for i in range(30)], ["full_name", "score"],
        lambda syn: [[form.format(syn[0]["full_name"], syn[0]["full_name"]), i] for i in range(30)])
    victim = first[0]["full_name"]
    if form == "{}, J.":  # two name words (one is an initial): not a single-word name, nothing to avoid
        return
    assert len(second) == 30 and victim.casefold() not in {r["full_name"].casefold() for r in second}
    if form != "{}":  # the exact value is never drawn at all (R7: the run's real texts, privacy.texts)
        assert '"name_pairs_redrawn"' in (rep / "files" / "people.csv.json").read_text(encoding="utf-8")
    res, again, _ = _run(tmp_path / "in", tmp_path, name="again")
    assert res.exit_code == 0 and _hashes(again) == _hashes(out)  # same seed, byte-identical


def test_lone_first_name_column_never_writes_a_single_word_full_name(tmp_path: Path) -> None:
    first, second, _, _ = _two_runs(
        tmp_path, ["first_name", "score"], [["Zelmira", i] for i in range(30)], ["full_name", "score"],
        lambda syn: [[syn[0]["first_name"], i] for i in range(30)])
    assert first[0]["first_name"] not in {r["first_name"] for r in second}


def test_first_name_with_blank_last_name_never_writes_a_single_word_full_name(tmp_path: Path) -> None:
    rows = [["Zelmira", "Quenby", i] for i in range(20)] + [["Zelmira", "", i] for i in range(10)]
    first, second, _, _ = _two_runs(
        tmp_path, ["first_name", "last_name", "score"], rows, ["full_name", "score"],
        lambda syn: [[syn[0]["first_name"], i] for i in range(30)])
    victim = first[0]["first_name"]
    assert sum(r["last_name"] == "" for r in second) == 10
    assert victim not in {r["first_name"] for r in second}  # any row could hold it without a last name


def test_real_first_name_with_blank_last_name_is_a_full_name(tmp_path: Path) -> None:
    first, second, _, _ = _two_runs(
        tmp_path, ["full_name", "score"], [["Zelmira", i] for i in range(30)], ["first_name", "last_name"],
        lambda syn: [[syn[0]["full_name"], ""]] * 30)
    assert first[0]["full_name"] not in {r["full_name"] for r in second}


def test_first_and_middle_name_never_spell_a_real_full_name(tmp_path: Path) -> None:
    rows = [["Zelmira", "Quenby", "Vosswyn", i] for i in range(30)]
    first, second, _, _ = _two_runs(
        tmp_path, ["first_name", "middle_name", "last_name", "score"], rows, ["full_name", "score"],
        lambda syn: [[f"{syn[0]['first_name']} {syn[0]['middle_name']}", i] for i in range(30)])
    real = (first[0]["first_name"], first[0]["middle_name"])
    assert real not in {(r["first_name"], r["middle_name"]) for r in second}


def test_final_check_fails_a_file_that_spells_a_real_full_name(tmp_path: Path, monkeypatch) -> None:
    """Defence in depth: with the generation-time redraw disabled, the pre-write check FAILs the file."""
    d = tmp_path / "in"
    _write(d / "people.csv", ["full_name", "score"], [["Zelmira", i] for i in range(30)])
    res, out1, _ = _run(d, tmp_path, name="alone")
    victim = read_dicts(out1 / "people.csv")[0]["full_name"]
    _write(d / "others.csv", ["full_name", "score"], [[victim, i] for i in range(30)])
    monkeypatch.setattr(pii, "avoid_real_full_names", lambda tp, cols, rng: [])
    monkeypatch.setattr(gtext, "real_text", lambda text: False)  # nor the run's real texts (R7, privacy.texts)
    res, out, rep = _run(d, tmp_path, name="unprotected")
    assert res.exit_code == 3, res.output
    assert not (out / "people.csv").exists()
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    entry = next(f for f in report["files"] if f["source"] == "people.csv")
    assert entry["status"] == "FAIL" and "full name" in json.dumps(entry)
    assert victim not in json.dumps(entry)


# =========================================================================== R2-17 long first records
_BIG = "word " * 60000  # 300 KB
_LONG_TEXTS = {
    "comma": (lambda: "comment,score\n" + "".join(f"{_BIG}{i},{i}\n" for i in range(3)), ",", 2),
    "tab": (lambda: "comment\tscore\n" + "".join(f"{_BIG}{i}\t{i}\n" for i in range(3)), "\t", 2),
    "semicolon_crlf": (lambda: "comment;score\r\n" + "".join(f"{_BIG}{i};{i}\r\n" for i in range(3)), ";", 2),
    "one_record": (lambda: "comment,score\n" + f"{_BIG},1\n", ",", 2),
    "quoted_line_breaks": (lambda: "comment,score\n" + "".join(f'"{_BIG}\nline, two {i}",{i}\n' for i in range(3)),
                           ",", 2),
    "over_1mb_cells": (lambda: "comment,score\n" + "".join(f'"{"x, " * 700000}{i}",{i}\n' for i in range(3)),
                       ",", 2),
    "wide_header": (lambda: ",".join(f"column_{i:05d}" for i in range(8000)) + "\n"
                    + "\n".join(",".join(str((i * j) % 7) for i in range(8000)) for j in range(3)) + "\n", ",", 8000),
}


@pytest.mark.parametrize("case", list(_LONG_TEXTS))
def test_long_records_do_not_break_delimiter_sniffing(case: str) -> None:
    make, delim, n_cols = _LONG_TEXTS[case]
    text = make()
    assert sniff.sniff_delimiter(text).delimiter == delim
    table = csvio.parse_text_bytes(text.encode("utf-8")).tables[0]
    assert len(table.header) == n_cols


def test_quoting_scan_sees_records_beyond_the_scan_budget() -> None:
    text = '"comment","score"\n' + "".join(f'"{"word " * 300000}{i}","{i}"\n' for i in range(3))
    q = sniff.scan_quoting(text, ",", 2)
    assert q.label == "all" and [c.policy for c in q.columns] == ["always", "always"]


def test_long_comment_csv_runs_without_delimiter(tmp_path: Path) -> None:
    rows = [["Zelmira private narrative " * 10000, i] for i in range(3)]
    src = _write(tmp_path / "in" / "data.csv", ["comment", "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "data.csv")
    assert len(syn) == 3
    _assert_absent(["Zelmira", "narrative"], out, rep, console=res.output)
