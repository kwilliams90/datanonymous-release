"""ADR-0006: in a table with no person column, sharing mode counts each row as a different person, unless something
suggests that a person appears on several rows.

Every input here is invented, except ``example-data/StudentsPerformance.csv`` (the repository's public example).
Tests run ``pipeline.execute`` (or the CLI) and look only at what an operator or recipient sees: the output folder,
the report folder, the exit code and the findings that decide release.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

import pytest

from datanonymous.config import ColumnRef, Settings
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import read_records, student_id

EXAMPLE = Path(__file__).resolve().parents[1] / "example-data" / "StudentsPerformance.csv"
SEED = 573291
ASSUMPTION = "Each row was treated as a different person"
PERSON_ID_FIX = "--person-id"
N = 60
COMMON = ("Velmont Track", "Ostrava Track", "Quinlo Track")  # 19 rows each
RARE = "Zorbex Academy"  # 3 rows: fewer than --min-cell (10)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _program(i: int) -> str:
    return RARE if i < 3 else COMMON[i % 3]


def _rows(extra: dict | None = None) -> list[dict[str, str]]:
    """One row per (invented) person, no person column: a rare program held by 3 rows, common ones by 19."""
    return [{"Program": _program(i), "Score": f"{(5100 + (i * 571) % 3999) / 100:.2f}",
             **{k: v(i) for k, v in (extra or {}).items()}} for i in range(N)]


def _run(original: Path, base: Path, **settings):
    out, report = base / "synthetic", base / "local-report"
    run = execute(Settings(mode="sharing", seed=SEED, **settings), original, out, report)
    return run, out, report


def _attention(report: Path) -> str:
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    return md.split("## What needs attention", 1)[1].split("## Appendix", 1)[0]


def _cells(out: Path) -> set[str]:
    return {v for path in out.rglob("*.csv") for row in read_records(path) for v in row.values()}


# --------------------------------------------------------------------------- the example file


@pytest.fixture(scope="module")
def example_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("example")
    original = base / "original"
    original.mkdir()
    shutil.copy(EXAMPLE, original / EXAMPLE.name)
    run, out, report = _run(original, base)
    return run, out, report


def test_the_example_is_written_with_its_columns_protected_not_nulled(example_run):
    run, out, _ = example_run
    assert exit_code(run) == 0, [(m.code, m.text) for m in blocking_findings(run)]
    written = read_records(out / EXAMPLE.name)
    source = read_records(EXAMPLE)
    assert len(written) == len(source)
    assert list(written[0]) == list(source[0])
    nulled = {c for c in written[0] if all(row[c] == "" for row in written)}
    # only a column that describes another person (the parents) stays unprotectable
    assert nulled == {"parental level of education"}
    for column in ("gender", "race/ethnicity", "lunch", "test preparation course",
                   "math score", "reading score", "writing score"):
        assert len({row[column] for row in written}) > 1, column


def test_the_example_releases_no_value_held_by_fewer_than_min_cell_rows(example_run):
    _, out, _ = example_run
    written = read_records(out / EXAMPLE.name)
    source = read_records(EXAMPLE)
    for column in written[0]:
        held = Counter(row[column] for row in source)
        rare = {v for v, c in held.items() if c < 10}
        released = {row[column] for row in written} - {""}
        assert not released & rare, (column, sorted(released & rare))


def test_the_report_names_the_assumption_and_the_person_id_fix(example_run):
    _, _, report = example_run
    attention = _attention(report)
    line = next(line for line in attention.splitlines() if ASSUMPTION in line)
    assert EXAMPLE.name in line
    fix = attention.split(line, 1)[1].splitlines()[1]
    assert PERSON_ID_FIX in fix


def test_the_assumption_does_not_block_release(example_run):
    run, _, _ = example_run
    assert run.package["released"]
    assert not blocking_findings(run)
    assert all(f.written for f in run.files)


# --------------------------------------------------------------------------- invented tables


def test_a_value_held_by_fewer_than_min_cell_rows_is_suppressed(tmp_path):
    original = tmp_path / "original"
    _write(original / "programs.csv", _rows())
    run, out, report = _run(original, tmp_path)
    assert exit_code(run) == 0, [(m.code, m.text) for m in blocking_findings(run)]
    written = read_records(out / "programs.csv")
    programs = {row["Program"] for row in written}
    assert RARE not in _cells(out)
    assert programs - {""} and programs - {""} <= set(COMMON)
    assert ASSUMPTION in _attention(report)


def test_a_higher_min_cell_suppresses_more(tmp_path):
    original = tmp_path / "original"
    _write(original / "programs.csv", _rows())
    run, out, _ = _run(original, tmp_path, min_cell=20)
    # 19 rows hold each common program: none of them reaches 20, so no program value is released
    assert not set(COMMON) & _cells(out)
    assert RARE not in _cells(out)


def test_exact_duplicate_rows_still_count_once(tmp_path):
    original = tmp_path / "original"
    rows = _rows()
    # 4 distinct records of a fourth program, each exported 3 times: 12 rows, but 4 people
    dup = "Harrowgate Track"
    rows += [{"Program": dup, "Score": f"{97 + r / 10:.2f}"} for r in range(4) for _ in range(3)]
    _write(original / "programs.csv", rows)
    run, out, _ = _run(original, tmp_path)
    assert exit_code(run) == 0
    assert dup not in _cells(out)


# --------------------------------------------------------------------------- signs of repeats keep today's behaviour


def _names(i: int) -> str:
    """60 different invented full names."""
    surnames = ("Kettle", "Lorrin", "Mapple", "Norvik", "Oskel", "Pardell", "Quorry", "Rensik", "Sallow", "Tormy")
    return f"{('Alvo', 'Brisen', 'Corra', 'Dimo', 'Esta', 'Faro')[i % 6]} {surnames[(i // 6) % 10]}"


SIGNS = {
    # a person-name column that repeats a value (30 names, 2 rows each)
    "repeated name": ({"Name": lambda i: _names(i // 2)}, {}),
    # an e-mail column that repeats a value
    "repeated e-mail": ({"Email": lambda i: f"person{i // 2}@example.org"}, {}),
    # a column of repeated wide ID-like keys (nulled, so D38 does not fail its file)
    "repeated ID-like keys": ({"Ref_Key": lambda i: str(830012345 + (i // 2) * 7919)},
                              {"null": [ColumnRef("Ref_Key")]}),
    # ADR-0006 amendment: repeated names still block when no own-person key is unique and filled on every row
    "repeated name, repeated e-mail": ({"Name": lambda i: _names(i // 2),
                                        "Email": lambda i: f"person{i // 2}@example.org"}, {}),
    "repeated name, e-mail with blanks": ({"Name": lambda i: _names(i // 2),
                                           "Email": lambda i: "" if i % 10 == 0 else f"person{i}@example.org"}, {}),
    # a record number is different on every row even when a person has several rows: no evidence of namesakes
    "repeated name, record id": ({"id": lambda i: str(i + 1), "Name": lambda i: _names(i // 2)}, {}),
    # a bare SID numbers sessions or visits as often as students
    "repeated name, bare SID": ({"SID": lambda i: student_id(i), "Name": lambda i: _names(i // 2)}, {}),
    # a repeated ID-like column is a sign even beside a unique e-mail
    "repeated name, unique e-mail, repeated ID-like keys": (
        {"Name": lambda i: _names(i // 2), "Email": lambda i: f"person{i}@example.org",
         "Ref_Key": lambda i: str(830012345 + (i // 2) * 7919)}, {"null": [ColumnRef("Ref_Key")]}),
}
NO_SIGNS = {
    "distinct names": ({"Name": lambda i: _names(i)}, {}),
    "distinct e-mails": ({"Email": lambda i: f"person{i}@example.org"}, {}),
    # ADR-0006 amendment: an own-person e-mail filled and different on every row makes repeated names namesakes
    "repeated name, unique e-mail": ({"Name": lambda i: _names(i // 2),
                                      "Email": lambda i: f"person{i}@example.org"}, {}),
    "repeated first and last names, unique e-mail": (
        {"First_Name": lambda i: _names(i // 2).split()[0], "Last_Name": lambda i: _names(i // 2).split()[1],
         "Email": lambda i: f"person{i}@example.org"}, {}),
}


@pytest.mark.parametrize("sign", sorted(SIGNS))
def test_a_sign_of_repeats_keeps_todays_behaviour(tmp_path, sign):
    extra, settings = SIGNS[sign]
    original = tmp_path / "original"
    _write(original / "programs.csv", _rows(extra))
    run, out, report = _run(original, tmp_path, **settings)
    attention = _attention(report)
    assert ASSUMPTION not in attention
    if (out / "programs.csv").exists():  # today's behaviour: the column that needs counts of people is nulled
        assert all(row["Program"] == "" for row in read_records(out / "programs.csv"))
    assert not set(COMMON) & _cells(out)
    assert RARE not in _cells(out)
    assert "person column" in attention or "which person each row belongs to" in attention


@pytest.mark.parametrize("case", sorted(NO_SIGNS))
def test_a_name_or_e_mail_column_that_never_repeats_is_no_sign(tmp_path, case):
    extra, settings = NO_SIGNS[case]
    original = tmp_path / "original"
    _write(original / "programs.csv", _rows(extra))
    run, out, report = _run(original, tmp_path, **settings)
    written = read_records(out / "programs.csv")
    assert {row["Program"] for row in written} - {""} <= set(COMMON)
    assert {row["Program"] for row in written} & set(COMMON)
    assert RARE not in _cells(out)
    assert ASSUMPTION in _attention(report)


def test_repeated_names_beside_a_unique_own_person_id_do_not_block(tmp_path):
    """ADR-0006 amendment, the ID side: an ID of the row's own person, different on every row, counts the people
    itself, so the repeated names are namesakes and the table needs no assumption."""
    original = tmp_path / "original"
    _write(original / "programs.csv", _rows({"Student_ID": student_id, "Name": lambda i: _names(i // 2)}))
    run, out, report = _run(original, tmp_path)
    assert exit_code(run) == 0, [(m.code, m.text) for m in blocking_findings(run)]
    written = read_records(out / "programs.csv")
    assert {row["Program"] for row in written} - {""} <= set(COMMON)
    assert {row["Program"] for row in written} & set(COMMON)
    assert RARE not in _cells(out)
    assert ASSUMPTION not in _attention(report)


def test_student_scores_no_longer_fails_for_no_people(tmp_path):
    """``example-data/student-scores.csv``: 2,000 rows, repeated first+last names, ``id`` and ``email`` unique."""
    original = tmp_path / "original"
    original.mkdir()
    source = EXAMPLE.parent / "student-scores.csv"
    shutil.copy(source, original / source.name)
    run, _, report = _run(original, tmp_path)
    assert "sharing_no_people" not in {m.code for m in blocking_findings(run)}
    attention = _attention(report)
    assert "sharing_no_people" not in attention
    assert ASSUMPTION in attention


# --------------------------------------------------------------------------- the assumption never crosses files


def test_rows_of_one_file_never_raise_the_holders_of_another(tmp_path):
    original = tmp_path / "original"
    shared = "Quillon Track"
    # a file with student IDs: 2 students hold the shared program, the others common ones
    _write(original / "roster.csv", [{"Student_ID": student_id(i),
                                      "Program": shared if i < 2 else COMMON[i % 3],
                                      "Score": f"{(4000 + (i * 389) % 3000) / 100:.2f}"} for i in range(N)])
    # a file without a person column: 40 rows hold the shared program
    _write(original / "programs.csv", [{"Program": shared if i < 40 else COMMON[i % 3],
                                        "Score": f"{(5100 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)])
    run, out, report = _run(original, tmp_path)
    assert exit_code(run) == 0, [(m.code, m.text) for m in blocking_findings(run)]
    roster = read_records(out / "roster.csv")
    assert shared not in {row["Program"] for row in roster}
    assert {row["Program"] for row in roster} & set(COMMON)
    attention = _attention(report)
    assert ASSUMPTION in attention
    line = next(line for line in attention.splitlines() if ASSUMPTION in line)
    assert "programs.csv" in line and "roster.csv" not in line
