"""D34: column headers, sheet names, file names and folder names hold no personal data.

Scripts depend on the mirrored paths and headers, so the tool never renames them: a name that holds personal data
(an e-mail/phone/SSN shape, a person's name, a real ID of the run) FAILs its file (D19: not written, exit 3) with a
fix, and the flagged text is never printed - not in the out dir, the report (md/json/manifest), the console, the
``inspect`` output or the out-dir docs. The location stays identifiable (column number, sheet number, the file's or
folder's place in its folder). False refusals matter as much: generic headers, school and place names, month words
and single words that are on a name list pass. Every name here is invented fixture data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import EXAMPLES, run_cli
from datanonymous.privacy import headers as d34
from datanonymous.privacy import leaks

N = 40
STUDENTS = ["Ana Garcia", "Luis Hernandez", "Maria Lopez"]
IDS = ["482913", "593024", "604135", "715246"]


# --------------------------------------------------------------------------- helpers
def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def numbers(n_cols: int, n: int = N, base: int = 50) -> list[list[Any]]:
    return [[base + (r * 7 + c * 3) % 50 for c in range(n_cols)] for r in range(n)]


def gradebook_rows(n_students: int) -> list[list[Any]]:
    return [[f"HW{r + 1}"] + [60 + (r * 5 + c * 11) % 40 for c in range(n_students)] for r in range(N)]


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "7"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def everything(*dirs: Path, console: str = "") -> str:
    """Every path and every file's text under ``dirs`` plus the console, case-folded."""
    parts = [console]
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            parts.append(p.relative_to(d).as_posix())
            if p.is_file():
                data = p.read_bytes()
                parts.append(data.decode("utf-8", errors="replace"))
                parts.append(data.decode("latin-1"))
    return "\n".join(parts).casefold()


def assert_never_printed(secrets: list[str], *dirs: Path, console: str = "") -> None:
    blob = everything(*dirs, console=console)
    forms = set()
    for s in secrets:
        forms |= {s, s.replace(" ", "_"), s.replace(" ", "%20"), json.dumps(s)[1:-1]}
    found = sorted(f for f in forms if f.casefold() in blob)
    assert not found, f"{len(found)} flagged name form(s) reached the out dir, the report or the console"


def report(rep: Path) -> dict[str, Any]:
    return json.loads((rep / "report.json").read_text(encoding="utf-8"))


def entry(rep: Path, source: str) -> dict[str, Any]:
    return next(f for f in report(rep)["files"] if f["source"] == source)


def d34_reasons(rep: Path) -> list[str]:
    return [r for f in report(rep)["files"] for r in f["reasons"] if r.startswith(d34.D34_PREFIX)]


def written(out: Path) -> list[str]:
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- the screen itself
@pytest.mark.parametrize("name", [
    "Garcia_Ana", "Ana Garcia", "AnaGarcia", "garcia-ana", "GARCIA_ANA", "Smith_John_IEP", "Luis.Hernandez",
    "Smith, J.", "Garcia Lopez, M. A.", "J. Smith", "Smith J.", "Mr. Smith", "Mrs Garcia", "Dr. Lee",
    "Mr Smith Class", "Maria J Hernandez", "Student 1 - Ana Garcia", "Ana Garcia Reading Score",
    "Garcia_Ana_Course_Grades", "Olivia_Kim", "O'Brien_Patrick",
])
def test_person_names_are_flagged(name: str) -> None:
    assert d34.name_kinds(name) == {"name"}


@pytest.mark.parametrize("name", [
    # generic headers
    "First_Name", "Last Name", "Teacher", "Homeroom", "Guardian1", "Student Name", "SchoolName", "Teacher_Name",
    "Student_First", "Guardian_Last", "Counselor", "Case Manager", "Referred By", "Emergency Contact 1",
    "Max_Date", "Min_Date", "Max Score", "Math_Score", "ELA_Score", "Gross_Pay", "Fee_Paid", "Nick_Name",
    "Mark_Total", "Max_Price", "Will_Return", "Grant_Amount", "Long_Term_Goal", "Dean List", "Honor Roll",
    "Foster_Care", "Walker", "Lane Number", "Hall Pass", "Chase Bank",
    # school, building and place names that contain person names
    "Lincoln High", "Martin Luther King Jr Elementary", "Washington Middle School", "Smith Hall", "Garcia Hall",
    "Kennedy Center", "Jefferson Academy", "Jackson County", "Madison District", "Lincoln Campus",
    "Hoover Building", "Room 12", "Carnegie Library", "George Washington Carver Elementary",
    "Dr Martin Luther King Jr Elementary", "John Adams Middle School", "Jefferson Davis MS", "Lincoln Park",
    "Martin Luther King Jr Day", "Fort Worth", "St John", "San Jose", "Mount Vernon", "Santa Ana",
    # month and day words, single words on a name list
    "May", "June", "April", "August", "Grant", "Price", "Rose", "Hall", "Will", "Mark", "Chase", "Lane",
    "Reading", "Max", "Ana", "Garcia",
    # other labels
    "U.S. History", "Plan B.", "Total, N", "Prof_Level", "Rev_Date", "Coach Rating", "Q1", "Week_1", "2019",
    "2019-20", "201920", "20230915",
])
def test_generic_school_and_single_word_names_pass(name: str) -> None:
    assert d34.name_kinds(name) == frozenset()


@pytest.mark.parametrize("name, kind", [
    ("ana.garcia@school.org", "email"), ("Call 312-555-0199", "phone"), ("312-555-0199", "phone"),
    ("SSN 123-45-6789", "ssn"), ("Lives at 12 Oak Street", "address"),
])
def test_personal_data_shapes_are_flagged(name: str, kind: str) -> None:
    assert kind in d34.name_kinds(name)


def test_a_url_alone_is_no_personal_data() -> None:
    assert d34.name_kinds("www.district.org") == frozenset() and d34.name_kinds("https://example.com/x") == frozenset()


def test_real_id_check_uses_parts_and_d8_keys_but_not_years_or_short_parts() -> None:
    class NS:
        raw_values = {"482913", "S-0042"}
        representatives = {"482913", "s0042"}

    ids = d34.IdKeys([NS()])
    assert ids.match("482913") and ids.match("score_482913") and ids.match("0482913")  # D8 key: leading zeros
    assert ids.match("S0042") and ids.match("s-0042")  # D8 key: case and punctuation
    assert not ids.match("4829") and not ids.match("2019") and not ids.match("Week 12")

    class Six:  # 6-digit student IDs: a date header must never be read as one by its D8 key
        raw_values = {"932024", "202409"}
        representatives = {"932024", "202409"}

    six = d34.IdKeys([Six()])
    assert not six.match("9/3/2024") and not six.match("2024-09") and not six.match("2024-09-03 00:00:00")
    assert six.match("932024")

    class Four:  # 4-digit record IDs: a whole header equal to one is flagged, a number inside a label is not
        raw_values = {"1000", "1500"}
        representatives = {"1000", "1500"}

    four = d34.IdKeys([Four()])
    assert four.match("1500") and not four.match("Lexile 1000") and not four.match("1500 m")
    assert d34.date_like("2019") and d34.date_like("201920") and d34.date_like("20230915")
    assert not d34.date_like("482913")
    assert "482913" not in d34.id_candidates("482")  # too short to be an ID here


# --------------------------------------------------------------------------- end to end: FAIL, never printed
def test_wide_gradebook_with_student_name_headers_fails_csv_and_xlsx(tmp_path: Path) -> None:
    src = tmp_path / "in"
    header = ["Assignment", *STUDENTS, "Class Average"]
    write_csv(src / "gradebook.csv", header, [r + [75] for r in gradebook_rows(len(STUDENTS))])
    df = pd.DataFrame(gradebook_rows(len(STUDENTS)), columns=["Assignment", *STUDENTS])
    df.to_excel(src / "gradebook.xlsx", index=False, sheet_name="Grades")
    write_csv(src / "plain.csv", ["Week", "Score"], numbers(2))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert written(out) == ["DATA_DICTIONARY.md", "README.md", "plain.csv", "schema.json"]
    csv_entry, xlsx_entry = entry(rep, "gradebook.csv"), entry(rep, "gradebook.xlsx")
    assert csv_entry["status"] == "FAIL" and not csv_entry["written"]
    assert "the headers of 3 of 5 columns (columns 2-4; a person's name: 3)" in csv_entry["reasons"][0]
    assert "the headers of 3 of 4 columns of sheet 1 of 1 (columns 2-4" in xlsx_entry["reasons"][0]
    assert any("--exclude-path GLOB" in f and "rename" in f for f in csv_entry["fixes"])
    per_file = json.loads((rep / "files" / "gradebook.csv.json").read_text(encoding="utf-8"))
    assert per_file["details_withheld"] and per_file["tables"] == [] and per_file["layout"] is None
    assert_never_printed(STUDENTS, out, rep, console=res.output)


@pytest.mark.parametrize("prefix, extra", [("S", ()), ("", ("--header-row", "1"))])
def test_pivoted_table_with_real_student_ids_as_headers_fails(tmp_path: Path, prefix: str,
                                                              extra: tuple[str, ...]) -> None:
    """One column per student: the headers are real IDs of the roster's namespace (exactly, by their D8 key, or
    inside a longer header). Digit-only headers already fail the header-row check (a long number reads as data),
    so ``--header-row 1`` asserts the header for that variant."""
    src = tmp_path / "in"
    ids = [prefix + i for i in IDS]
    write_csv(src / "roster.csv", ["StudentID", "Grade", "Score"],
              [[ids[r] if r < len(ids) else f"{prefix}{800000 + r}", 9 + r % 4, 50 + r] for r in range(N)])
    third = f"{prefix.lower()}-{IDS[2]}" if prefix else "0" + IDS[2]  # same ID by its D8 key
    write_csv(src / "pivot.csv", ["Assignment", ids[0], f"score_{ids[1]}", third], gradebook_rows(3))
    res, out, rep = run(src, tmp_path, *extra)
    assert res.exit_code == 3, res.output
    assert "pivot.csv" not in written(out) and "roster.csv" in written(out)
    reason = entry(rep, "pivot.csv")["reasons"][0]
    assert "the headers of 3 of 4 columns (columns 2-4; a real ID of this run: 3)" in reason
    assert_never_printed(ids[:3] + IDS[:3], out, rep, console=res.output)


@pytest.mark.parametrize("extra", [(), ("--header-row", "1")])
def test_headers_with_email_phone_and_ssn_fail(tmp_path: Path, extra: tuple[str, ...]) -> None:
    """A header line holding an e-mail address, a phone number or an SSN already reads as a data record (the
    header-row check FAILs it); asserted as the header (``--header-row 1``), D34 FAILs it."""
    src = tmp_path / "in"
    secrets = ["zelda.quist@district.edu", "312-555-0187", "123-45-6789"]
    write_csv(src / "contacts.csv", ["Week", secrets[0], f"Call {secrets[1]}", f"SSN {secrets[2]}"], numbers(4))
    res, out, rep = run(src, tmp_path, *extra)
    assert res.exit_code == 3, res.output
    assert "contacts.csv" not in written(out)
    reason = entry(rep, "contacts.csv")["reasons"][0]
    if extra:
        assert reason.startswith(d34.D34_PREFIX) and "columns 2-4" in reason
        assert "an e-mail address: 1" in reason and "a phone number: 1" in reason and "an SSN: 1" in reason
    else:
        assert "does not look like a header row" in reason
    assert_never_printed(secrets + ["zelda.quist", "3125550187", "123456789"], out, rep, console=res.output)


def test_excel_sheet_named_after_a_student_fails(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    with pd.ExcelWriter(src / "book.xlsx") as xw:
        pd.DataFrame(numbers(2), columns=["Week", "Score"]).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(numbers(2), columns=["Week", "Score"]).to_excel(xw, sheet_name="Lopez, Maria", index=False)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert "book.xlsx" not in written(out)
    assert "the name of sheet 2 of 2 (a person's name)" in entry(rep, "book.xlsx")["reasons"][0]
    assert_never_printed(["Lopez, Maria", "Maria Lopez", "Lopez"], out, rep, console=res.output)


def test_file_named_after_a_student_fails_and_its_path_is_redacted(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "students" / "Smith_John_IEP.csv", ["Goal", "Minutes"], numbers(2))
    write_csv(src / "students" / "attendance.csv", ["Week", "Days"], numbers(2))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert written(out) == ["DATA_DICTIONARY.md", "README.md", "schema.json", "students/attendance.csv"]
    failed = [f for f in report(rep)["files"] if f["status"] == "FAIL"]
    assert [f["source"] for f in failed] == ["students/[redacted-1].csv"]
    reason = failed[0]["reasons"][0]
    size = (src / "students" / "Smith_John_IEP.csv").stat().st_size
    assert "file 2 of 2 in folder students/ (sorted by name, ignoring case); .csv" in reason
    assert f"{size:,} bytes" in reason
    assert (rep / "files" / "students" / "[redacted-1].csv.md").is_file()
    manifest = json.loads((rep / "manifest.json").read_text(encoding="utf-8"))
    assert "students/[redacted-1].csv" in [f["source"] for f in manifest["files"]]
    assert_never_printed(["Smith_John_IEP", "Smith_John"], out, rep, console=res.output)


def test_folder_named_after_a_student_fails_every_file_under_it(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "students" / "Garcia_Ana" / "scores.csv", ["Week", "Score"], numbers(2))
    write_csv(src / "students" / "Garcia_Ana" / "sub" / "reading.csv", ["Week", "Words"], numbers(2))
    (src / "students" / "Garcia_Ana" / "notes.docx").write_bytes(b"not a table")
    write_csv(src / "students" / "Group_B" / "scores.csv", ["Week", "Score"], numbers(2))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert written(out) == ["DATA_DICTIONARY.md", "README.md", "schema.json", "students/Group_B/scores.csv"]
    data = report(rep)
    failed = sorted(f["source"] for f in data["files"] if f["status"] == "FAIL")
    assert failed == ["students/[redacted-1]/scores.csv", "students/[redacted-1]/sub/reading.csv"]
    assert all("folder 1 of 2 in folder students/ (sorted by name, ignoring case), with 3 file(s) under it" in r
               for r in d34_reasons(rep))
    assert [s["path"] for s in data["skipped"]] == ["students/[redacted-1]/notes.docx"]
    as_json = run_cli(src, "--out-dir", tmp_path / "j-out", "--report-dir", tmp_path / "j-rep", "--seed", "7", "--json")
    assert as_json.exit_code == 3
    assert json.loads(as_json.output)["failures"][0]["source"].startswith("students/[redacted-1]/")
    assert_never_printed(["Garcia_Ana", "Garcia Ana"], out, rep, tmp_path / "j-out", tmp_path / "j-rep",
                         console=res.output + as_json.output)


def test_parquet_field_and_stata_variable_and_json_key_names_fail(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    df = pd.DataFrame({"Week": range(N), "Maria_Lopez": range(N), "Score": range(N)})
    df.to_parquet(src / "w.parquet")
    df.to_feather(src / "w.feather")
    pd.DataFrame({"week": range(N), "Lopez_Maria": range(N)}).to_stata(src / "w.dta", write_index=False)
    (src / "w.json").write_text(json.dumps([{"Week": i, "Maria Lopez": i} for i in range(N)]), encoding="utf-8")
    (src / "w.jsonl").write_text("\n".join(json.dumps({"Week": i, "maria.lopez@district.edu": i}) for i in range(N))
                                 + "\n", encoding="utf-8")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert [f for f in written(out) if f.startswith("w.")] == []
    for name, col in (("w.parquet", "column 2 of 3"), ("w.feather", "column 2 of 3"), ("w.dta", "column 2 of 2"),
                      ("w.json", "column 2 of 2"), ("w.jsonl", "column 2 of 2")):
        assert f"the header of {col}" in entry(rep, name)["reasons"][0], name
    assert_never_printed(["Maria_Lopez", "Lopez_Maria", "Maria Lopez", "maria.lopez@district.edu"], out, rep,
                         console=res.output)


def test_stata_variable_labels_are_never_written_so_they_pass_and_are_never_printed(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    pd.DataFrame({"week": range(N), "score": range(N)}).to_stata(
        src / "labelled.dta", write_index=False, variable_labels={"score": "Score of Maria Lopez"})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "labelled.dta" in written(out)
    assert_never_printed(["Maria Lopez"], out, rep, console=res.output)


# --------------------------------------------------------------------------- false refusals
def test_generic_headers_school_names_and_single_words_pass_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "in"
    header = ["First_Name", "Last Name", "Teacher", "Homeroom", "Guardian1", "Student Name", "SchoolName"]
    write_csv(src / "roster.csv", header,
              [[f"F{r}", f"L{r}", f"T{r % 3}", f"H{r % 4}", f"G{r}", f"S{r}", "North"] for r in range(N)])
    places = ["Lincoln High", "Martin Luther King Jr Elementary", "Washington Middle School", "Smith Hall",
              "Jefferson Academy", "Kennedy Center", "Jackson County", "Madison District", "Lincoln Campus",
              "Hoover Building", "Room 12", "Carnegie Library"]
    write_csv(src / "Lincoln High" / "by_school.csv", ["Week", *places], numbers(len(places) + 1))
    singles = ["May", "June", "Grant", "Price", "Rose", "Hall", "Will", "Mark", "Chase", "Lane", "Reading"]
    write_csv(src / "Washington Middle School" / "Mark.csv", ["Week", *singles], numbers(len(singles) + 1))
    with pd.ExcelWriter(src / "May.xlsx") as xw:
        pd.DataFrame(numbers(2), columns=["Week", "Score"]).to_excel(xw, sheet_name="Lincoln High", index=False)
        pd.DataFrame(numbers(2), columns=["Week", "Score"]).to_excel(xw, sheet_name="June", index=False)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert not d34_reasons(rep) and "D34" not in res.output
    assert "Lincoln High/by_school.csv" in written(out) and "Washington Middle School/Mark.csv" in written(out)
    assert "May.xlsx" in written(out)


def test_attendance_dates_as_headers_are_no_student_ids(tmp_path: Path) -> None:
    """Date headers of an attendance pivot read as 6-8 digits without their separators; with 6-digit student IDs
    (``932024``, ``202409``) that would equal real IDs by chance, so a date header is compared exactly only."""
    src = tmp_path / "in"
    ids = ["932024", "202409", "942024", "102024"] + [str(500000 + r) for r in range(N - 4)]
    write_csv(src / "roster.csv", ["StudentID", "Grade"], [[i, 9 + r % 4] for r, i in enumerate(ids)])
    dates = ["9/3/2024", "9/4/2024", "10/2/2024", "2024-09", "2024-25"]
    write_csv(src / "attendance.csv", ["Period", *dates], [[f"P{r % 8}"] + ["P"] * len(dates) for r in range(N)])
    res, out, rep = run(src, tmp_path, "--header-row", "1")
    assert res.exit_code == 0, res.output
    assert not d34_reasons(rep) and "attendance.csv" in written(out)


def test_example_data_and_fixtures_raise_no_flag(tmp_path: Path) -> None:
    res, out, rep = run(EXAMPLES, tmp_path, name="examples", seed="1")
    assert res.exit_code == 0 and not d34_reasons(rep)
    fixtures = Path(__file__).resolve().parent / "fixtures"
    res, out, rep = run(fixtures, tmp_path, name="fixtures", seed="1")
    assert not d34_reasons(rep) and "D34" not in res.output
    assert not any(m.get("code") == "personal_names_redacted" for m in report(rep)["notes"])


# --------------------------------------------------------------------------- overrides and other paths
def test_keep_name_keeps_a_flagged_name_that_is_not_personal_data(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "gradebook.csv", ["Assignment", *STUDENTS[:2]], gradebook_rows(2))
    res, _, rep = run(src, tmp_path, "--keep-name", STUDENTS[0], name="one")
    assert res.exit_code == 3
    assert "the header of column 3 of 3 (a person's name)" in entry(rep, "gradebook.csv")["reasons"][0]
    res, out, _ = run(src, tmp_path, "--keep-name", STUDENTS[0], "--keep-name", STUDENTS[1], name="both")
    assert res.exit_code == 0, res.output
    assert (out / "gradebook.csv").read_text(encoding="utf-8").splitlines()[0] == "Assignment,Ana Garcia,Luis Hernandez"


def test_keep_name_from_the_config_file(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "Garcia_Ana" / "scores.csv", ["Week", "Score"], numbers(2))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("keep_name: [Garcia_Ana]\n", encoding="utf-8")
    res, out, _ = run(src, tmp_path, "--config", cfg)
    assert res.exit_code == 0, res.output
    assert "Garcia_Ana/scores.csv" in written(out)


def test_excluded_column_and_excluded_file_are_redacted_but_not_failed(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "gradebook.csv", ["Assignment", "Score", STUDENTS[0]], gradebook_rows(2))
    write_csv(src / "Smith_John_IEP.csv", ["Goal", "Minutes"], numbers(2))
    res, out, rep = run(src, tmp_path, "--exclude", f"gradebook.csv::{STUDENTS[0]}",
                        "--exclude-path", "Smith_John_IEP.csv")
    assert res.exit_code == 0, res.output
    assert (out / "gradebook.csv").read_text(encoding="utf-8").splitlines()[0] == "Assignment,Score"
    data = report(rep)
    assert [s["path"] for s in data["skipped"]] == ["[redacted-1].csv"]
    assert any(n.get("code") == "personal_names_redacted" for n in data["notes"])
    assert_never_printed([STUDENTS[0], "Smith_John_IEP"], out, rep, console=res.output)


def test_inspect_fails_the_same_files_and_never_prints_the_names(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "Garcia_Ana" / "scores.csv", ["Week", "Score"], numbers(2))
    write_csv(src / "gradebook.csv", ["Assignment", *STUDENTS], gradebook_rows(3))
    write_csv(src / "kept.csv", ["Assignment", "Score", STUDENTS[2]], gradebook_rows(2))
    res = run_cli("inspect", src, "--exclude", f"kept.csv::{STUDENTS[2]}")
    assert res.exit_code == 3
    assert "== [redacted-1]/scores.csv  [csv]  FAIL" in res.output
    assert "the headers of 3 of 4 columns (columns 2-4" in res.output
    res_json = run_cli("inspect", src, "--json", "--exclude", f"kept.csv::{STUDENTS[2]}")
    assert res_json.exit_code == 3
    assert_never_printed(STUDENTS + ["Garcia_Ana"], console=res.output + res_json.output)


def test_stale_copy_under_a_flagged_folder_is_reported_redacted(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "Garcia_Ana" / "scores.csv", ["Week", "Score"], numbers(2))
    write_csv(src / "plain.csv", ["Week", "Score"], numbers(2))
    out, rep = tmp_path / "out", tmp_path / "rep"
    first = run_cli(src, "--out-dir", out, "--report-dir", tmp_path / "rep1", "--seed", "3",
                    "--keep-name", "Garcia_Ana")  # an earlier run that kept the name (or a tool version before D34)
    assert first.exit_code == 0 and (out / "Garcia_Ana" / "scores.csv").is_file()
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "3", "--force")
    assert res.exit_code == 3
    stale = [w for w in report(rep)["warnings"] if w["code"] == "stale_output"]
    assert len(stale) == 1 and stale[0]["text"].startswith("[redacted-1]/scores.csv: ")
    assert "delete it before the out dir is shared" in stale[0]["text"]
    assert "- `[redacted-1]/scores.csv`" in (out / "README.md").read_text(encoding="utf-8")
    texts = [p for p in out.rglob("*") if p.is_file() and p.suffix in (".md", ".json")]
    assert_never_printed(["Garcia_Ana"], rep, console=res.output + "".join(p.read_text(encoding="utf-8")
                                                                          for p in texts))


def test_leak_scan_shows_an_unknown_out_dir_path_redacted(tmp_path: Path) -> None:
    """A file in the out dir that this run did not write (left by an earlier run of other data) is scanned; when
    it holds a real value, the error names it by its path with every name that holds personal data redacted."""
    src = tmp_path / "in"
    phones = [f"312-555-{1000 + r:04d}" for r in range(N)]
    write_csv(src / "contacts.csv", ["StudentID", "Phone"], [[f"S{900000 + r}", p] for r, p in enumerate(phones)])
    out, rep = tmp_path / "out", tmp_path / "rep"
    write_csv(out / "Garcia_Ana" / "old.csv", ["Note"], [[f"call {phones[3]}"]])
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "5", "--force")
    assert res.exit_code == 3
    errors = [e for e in report(rep)["errors"] if e["code"] == "leak_found"]
    assert errors and "[redacted]/old.csv" in errors[0]["text"]
    assert_never_printed(["Garcia_Ana"], rep, console=res.output)


def test_same_seed_gives_identical_output_and_reports(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "gradebook.csv", ["Assignment", *STUDENTS], gradebook_rows(3))
    write_csv(src / "Garcia_Ana" / "scores.csv", ["Week", "Score"], numbers(2))
    write_csv(src / "plain.csv", ["Week", "Score"], numbers(2))
    runs = [run(src, tmp_path, name=f"r{k}", seed="11") for k in range(2)]
    (res_a, out_a, rep_a), (res_b, out_b, rep_b) = runs
    assert res_a.exit_code == res_b.exit_code == 3
    assert written(out_a) == written(out_b)
    for rel in written(out_a):
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel
    files_a = sorted(p.relative_to(rep_a).as_posix() for p in (rep_a / "files").rglob("*.json"))
    files_b = sorted(p.relative_to(rep_b).as_posix() for p in (rep_b / "files").rglob("*.json"))
    assert files_a == files_b
    for rel in files_a:
        assert (rep_a / rel).read_bytes() == (rep_b / rel).read_bytes(), rel
    a, b = report(rep_a), report(rep_b)
    assert a["files"] == b["files"] and a["failures"] == b["failures"] and a["notes"] == b["notes"]


# --------------------------------------------------------------------------- the leak scanner knows them
def test_leak_index_holds_flagged_names_and_the_allowlist_never_excuses_them() -> None:
    screen = d34.NameScreen()
    screen.add("Ana Garcia", "column", frozenset({"name"}), "column 2 of 4", written=True, number=2)
    screen.flags[0].file = "gradebook.csv"
    idx = leaks.SensitiveIndex.build([], screen)
    assert not idx.empty and idx.flagged_in("score of ana garcia") == {0}
    scanner = leaks.Scanner(idx, leaks.Allow(tokens={"ana", "garcia"}, phrases=set()))
    hits = scanner.hits_in_text("the column Ana Garcia was renamed")
    assert hits and all(h.origins == frozenset({0}) for h in hits)
    assert screen.redact_text("col 'Ana Garcia', ANA GARCIA, Ana%20Garcia") == (
        "col '[redacted-1]', [redacted-1], [redacted-1]")
    assert screen.display_path("Garcia_Ana/x.csv").startswith("[redacted]/")  # an unscreened path part
