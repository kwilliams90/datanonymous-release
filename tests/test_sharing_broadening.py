"""ADR-0004: sharing mode writes a rare text shape or date format in a broader held class instead of refusing.

Every input here is invented. Tests run a whole sharing run (``pipeline.execute`` or the CLI) and look only at what an
operator or recipient sees: the output folder, the report folder, the leak scan's verdict and the exit code.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous import messages
from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import read_records, student_id

N = 60
RARE = range(56, 60)  # four students: too few to hold a class on their own (--min-cell 10)
CHANGE_CODE = "sharing_format_broadened"


def _locker(i: int) -> str:
    """Pairs of students share a locker code: ``AB-1234`` for 56 students, ``ABC-12345`` for four."""
    j = i // 2
    if i in RARE:
        return f"QR{'VW'[j % 2]}-{52000 + j * 311:05d}"
    return f"{chr(65 + j % 7)}{chr(75 + j % 9)}-{1000 + j * 317:04d}"


def _enrolled(i: int) -> str:
    return "03/15/2024" if i in RARE else "2024-03-15"


def _plan_note(i: int, extra: str = " with x{letter}q{i}@commx.test") -> str:
    """The same words for everyone; the rare notes have more words, by default an e-mail address."""
    note = f"Met to plan week {10 + i % 40}"
    return note + extra.format(letter=chr(97 + i % 26), i=i) if i in RARE else note


def _nickname(i: int) -> str:
    """Made-up names of one shape; the rare ones carry the surname prefix ``Mc`` (a class defined by a name)."""
    first = "Z" + "aeiou"[i % 5] + "r" + "aeiou"[i // 5 % 5] + "x"
    return f"{first} {'Mc' if i in RARE else 'Bo'}{chr(66 + i % 20)}{'aeiou'[i // 20]}n"


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roster(original: Path, **columns) -> list[dict[str, str]]:
    rows = [{"Student_ID": student_id(i), **{name: make(i) for name, make in columns.items()},
             "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def _run(base: Path, **columns):
    original, out, report = base / "original", base / "synthetic", base / "local-report"
    source = _roster(original, **columns)
    run = execute(Settings(mode="sharing", seed=573291), original, out, report)
    return source, run, out, report


def _released(run) -> None:
    assert run.package["released"], [(m.code, m.text) for m in blocking_findings(run)]
    assert not blocking_findings(run)  # the leak scan passed and nothing else blocks the file
    assert exit_code(run) == 0
    assert all(f.written for f in run.files)


def _report(report: Path) -> str:
    """Every text file of the report folder (REPORT.md and the per-file reports)."""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(report.rglob("*")) if p.is_file())


def test_a_rare_code_shape_is_written_in_a_held_shape(tmp_path):
    source, run, out, report = _run(tmp_path, Locker_Code=_locker)
    _released(run)
    written = [row["Locker_Code"] for row in read_records(out / "roster.csv")]
    assert len(written) == N and all(written)
    # every code is written in the held class; the rare ABC-12345 class never appears
    assert all(re.fullmatch(r"[A-Z]{2}-\d{4}", code) for code in written), sorted(set(written))
    assert not [code for code in written if re.fullmatch(r"[A-Z]{3}-\d{5}", code)]
    assert not {row["Locker_Code"] for row in source} & set(written)  # never a real code


def test_a_rare_date_format_is_written_in_the_held_format(tmp_path):
    source, run, out, report = _run(tmp_path, Enrolled=_enrolled)
    _released(run)
    written = [row["Enrolled"] for row in read_records(out / "roster.csv")]
    assert len(written) == N and all(written)
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) for date in written), sorted(set(written))
    assert not [date for date in written if "/" in date]
    # the shared schema and data dictionary name only the held format, and not that another one existed
    shared = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
    assert "%Y-%m-%d" in shared and "%m/%d/%Y" not in shared and "MIXED" not in shared


@pytest.mark.parametrize("column,make", [("Locker_Code", _locker), ("Enrolled", _enrolled)])
def test_the_broadening_is_reported_as_an_informational_change(tmp_path, column, make):
    _, run, _, report = _run(tmp_path, **{column: make})
    _released(run)
    entry = messages.lookup(CHANGE_CODE)
    assert entry is not None and entry.kind == messages.CHANGE and entry.sentence.strip()
    found = [m for f in run.files for m in f.all_messages() if m.code == CHANGE_CODE]
    assert [(m.level, m.column) for m in found] == [("info", column)]
    md = _report(report)
    assert CHANGE_CODE in md and column in md
    # the report says what changed without the rare layout itself
    line = next(line for line in md.splitlines() if CHANGE_CODE in line)
    assert "QR" not in line and "03/15/2024" not in line


@pytest.mark.parametrize("column,make", [("Plan_Note", _plan_note), ("Nickname", _nickname)])
def test_a_rare_class_defined_by_personal_data_is_not_broadened_and_the_column_is_nulled(tmp_path, column, make):
    source, run, out, report = _run(tmp_path, **{column: make})
    _released(run)
    written = read_records(out / "roster.csv")
    assert all(row[column] == "" for row in written)
    assert all(row["Score"] for row in written)
    md = _report(report)
    assert "nulled automatically" in md and column in md and "shared by too few people" in md
    assert not [m for f in run.files for m in f.all_messages() if m.code == CHANGE_CODE]


def test_the_same_rare_note_without_personal_data_is_broadened(tmp_path):
    source, run, out, report = _run(tmp_path, Plan_Note=lambda i: _plan_note(i, " with extra time"))
    _released(run)
    written = [row["Plan_Note"] for row in read_records(out / "roster.csv")]
    assert all(written)
    shapes = {re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "0", v))) for v in written}
    assert shapes == {"Aaa aa aaaa aaaa 00"}, shapes  # the rare five-word shape never appears
    assert [m.column for f in run.files for m in f.all_messages() if m.code == CHANGE_CODE] == ["Plan_Note"]


def test_a_rare_class_defined_by_an_embedded_e_mail_still_refuses_a_column_in_a_relationship(tmp_path):
    # Plan_Note is determined by Advisory (a dependency), so nulling it would break that relationship; it comes
    # first, so its own refusal is the one the run reports
    def note(i: int) -> str:
        group = i // 6
        return f"Met to plan week {10 + group}" + (f" with x{chr(97 + group)}q@commx.test" if group == 9 else "")

    _, run, out, report = _run(tmp_path, Plan_Note=note, Advisory=lambda i: f"A{101 + i // 6}")
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written and exit_code(run) == 3
    assert not (out / "roster.csv").exists()
    failures = (report / "sharing-failures.csv").read_text(encoding="utf-8")
    assert "roster.csv" in failures and "sharing_rare_class" in failures, failures


def test_refuse_unsupported_does_not_stop_broadening(tmp_path):
    original, base = tmp_path / "original", tmp_path
    _roster(original, Locker_Code=_locker, Enrolled=_enrolled)
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(base / "synthetic"), "--report-dir",
                                       str(base / "local-report"), "--seed", "573291", "--refuse-unsupported"],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    written = read_records(base / "synthetic" / "roster.csv")
    assert all(re.fullmatch(r"[A-Z]{2}-\d{4}", row["Locker_Code"]) for row in written)
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["Enrolled"]) for row in written)
