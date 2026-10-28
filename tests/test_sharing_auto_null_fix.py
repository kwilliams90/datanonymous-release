"""An automatically nulled column's fix belongs to the reason it was nulled (ADR-0001, issue 21).

Every input here is invented. Tests run the CLI and read what an operator sees: the report's *What needs attention*
items and its Appendix list of Nulled columns.
"""

from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner

from datanonymous.cli import main

from release_corpus import student_id

N = 60
EDUCATION = ("Qorvath Diploma", "Belmira Certificate", "Tandrel Degree")  # each held by 20 rows
GENERIC = "If it holds IDs, declare it with"


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scores(original: Path) -> None:
    _write(original / "scores.csv", [{"Student_ID": student_id(i), "Term": ("Fall", "Spring")[i % 2],
                                      "Points": f"{(6000 + (i * 389) % 3000) / 100:.2f}"} for i in range(N)])


def _report(original: Path, base: Path, *flags: str) -> str:
    out, report = base / "synthetic", base / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291", *flags], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return (report / "REPORT.md").read_text(encoding="utf-8")


def _attention(md: str) -> dict[str, str]:
    """column -> the item's text and its Fix line, from *What needs attention*."""
    section = md.split("## What needs attention", 1)[1].split("\n## ", 1)[0]
    items: dict[str, str] = {}
    lines = section.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- ") and "nulled automatically" in line:
            fix = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("- Fix:") else ""
            items[line.split("`")[3]] = line + "\n" + fix
    return items


def _appendix(md: str) -> dict[str, str]:
    section = md.split("### Nulled columns", 1)[1].split("\n#", 1)[0]
    return {line.split("`")[3]: line for line in section.splitlines() if line.startswith("- ")}


def test_a_table_without_a_person_column_suggests_person_id_not_id_col(tmp_path):
    original = tmp_path / "original"
    # ADR-0006: a repeated name is a sign that a person has several rows, so the rows cannot count as people
    _write(original / "survey.csv", [{"Name": f"{('Alvo', 'Brisen', 'Corra')[i % 3]} {('Kettle', 'Lorrin')[i // 30]}",
                                      "Group": ("North", "South", "East")[i % 3],
                                      "Parent_Education": EDUCATION[i // 20],
                                      "Score": f"{(5100 + i * 73) / 100:.2f}",
                                      "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)])
    md = _report(original, tmp_path)
    items = _attention(md)
    appendix = _appendix(md)
    assert set(items) >= {"Group", "Score"}, md
    for column in ("Group", "Score"):
        assert "--person-id" in items[column] and GENERIC not in items[column], items[column]
        assert "--person-id" in appendix[column], appendix[column]
    # another person's attribute: leave it out, never declare it as an ID
    for text in (items["Parent_Education"], appendix["Parent_Education"]):
        assert "--exclude" in text and "--id-col" not in text, text


def test_another_persons_attribute_in_a_table_with_people_suggests_leaving_it_out(tmp_path):
    original = tmp_path / "original"
    _write(original / "roster.csv", [{"Student_ID": student_id(i), "Parent_Education": EDUCATION[i // 20],
                                      "Score": f"{(5100 + i * 73) / 100:.2f}",
                                      "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)])
    _scores(original)
    md = _report(original, tmp_path)
    for text in (_attention(md)["Parent_Education"], _appendix(md)["Parent_Education"]):
        assert "--exclude" in text and "--id-col" not in text and "--person-id" not in text, text


def test_an_id_like_column_nulled_after_generation_still_suggests_id_col(tmp_path):
    original = tmp_path / "original"
    _write(original / "roster.csv", [{"Student_ID": student_id(i), "Locker": str(100000 + (i * 7919) % 900000),
                                      "Score": f"{(5100 + i * 73) / 100:.2f}",
                                      "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)])
    _scores(original)
    md = _report(original, tmp_path)
    assert "--id-col Locker" in _attention(md)["Locker"]
    assert "--id-col Locker" in _appendix(md)["Locker"]
