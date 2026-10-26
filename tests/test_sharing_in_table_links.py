"""ADR-0001 amendment: a column whose only relationship stays inside its own table (a dependency, a template) is
nulled like any other non-key column when sharing mode cannot protect it, and the report says which link is no
longer kept. Keys, cross-file joins, aliases, nesting and rules the operator declared still leave the file out.

Every input here is invented. Tests run the CLI (or ``pipeline.execute``) and look only at what an operator or
recipient sees: the output folder, the report folder and the exit code.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous import pipeline
from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.errors import ColumnRefusal
from datanonymous.pipeline import execute, exit_code

from release_corpus import read_records, student_id

N = 60
YEARS = ("first year", "second year", "third year", "fourth year")  # 15 students each
FIRST = ("Zorane", "Nelutu", "Pavira", "Quoste", "Belank", "Tirvan")  # 10 students each, one layout
LAST = ("Pelvavo", "Norrulo", "Quindre", "Ostravi", "Melunde")  # 12 students each, one layout


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scores(i: int) -> dict[str, str]:
    return {"Score": f"{(5100 + i * 73) / 100:.2f}", "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"}


def _cli(original: Path, base: Path, *flags: str):
    out, report = base / "synthetic", base / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291", *flags], catch_exceptions=False)
    return result, out, report


def _section(report: Path, title: str) -> str:
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert f"\n## {title}\n" in md, md
    return md.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def _lines(section: str, column: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith("- ") and f"`{column}`" in line]


def _output_text(out: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in out.rglob("*") if p.is_file())


def _failures(report: Path) -> str:
    return (report / "sharing-failures.csv").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- a dependency inside the table
def _dependency_roster(original: Path) -> list[dict[str, str]]:
    """Join_Year is fixed by Year_of_Study (as students_data.csv's Yr_JoinCampus by YearofStudy). A number in a
    dependency cannot have its extremes protected, so sharing mode refuses it."""
    rows = [{"Student_ID": student_id(i), "Year_of_Study": YEARS[i // 15], "Join_Year": str(2025 - i // 15),
             **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def test_a_column_whose_only_relationship_is_a_dependency_is_nulled_and_the_link_reported(tmp_path):
    original = tmp_path / "original"
    source = _dependency_roster(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N and list(written[0]) == list(source[0])
    assert all(row["Join_Year"] == "" for row in written)  # the dependent side is nulled
    assert {row["Year_of_Study"] for row in written} <= set(YEARS) and all(row["Year_of_Study"] for row in written)
    # the report names the lost link under What changed, and keeps the nulled column under What needs attention
    changed = _section(report, "What changed")
    assert any("the link between Year_of_Study and Join_Year is not kept" in line
               for line in _lines(changed, "Join_Year")), changed
    attention = _section(report, "What needs attention")
    assert any("nulled automatically" in line and "Year_of_Study" in line
               for line in _lines(attention, "Join_Year")), attention
    assert "link between" not in attention
    assert "roster.csv" not in _failures(report)
    # the shareable README says a link is not kept, without naming the source's linked columns
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "A link between columns of this file is not kept" in readme
    assert "the link between Year_of_Study and Join_Year" not in readme


def test_refuse_unsupported_still_leaves_a_dependency_column_out(tmp_path):
    original = tmp_path / "original"
    _dependency_roster(original)
    result, out, report = _cli(original, tmp_path, "--refuse-unsupported")
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists()
    assert "sharing_numeric_structure" in _failures(report)


def test_the_dependent_side_is_nulled_when_the_column_it_depends_on_is_refused(tmp_path):
    """Year_of_Study has one value too few students hold; in a dependency that refuses it. Nulling Campus_Block (its
    dependent) instead lets Year_of_Study be protected like any other column: its rare value is merged away."""
    original = tmp_path / "original"
    years = (*YEARS, "fifth year")
    rows = [{"Student_ID": student_id(i), "Year_of_Study": years[min(i // 14, 4)],
             "Campus_Block": "North" if i < 28 else "South", **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", rows)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Campus_Block"] == "" for row in written)
    kept = {row["Year_of_Study"] for row in written}
    assert kept and "" not in kept and kept <= set(YEARS) and "fifth year" not in kept
    changed = _section(report, "What changed")
    assert any("the link between Year_of_Study and Campus_Block is not kept" in line
               for line in _lines(changed, "Campus_Block")), changed
    assert not _lines(_section(report, "What needs attention"), "Year_of_Study")


def test_a_dependency_column_refused_while_generated_is_nulled_and_the_link_reported(tmp_path, monkeypatch):
    rounds = _refused_while_generated(monkeypatch, "Homeroom")
    original = tmp_path / "original"
    # Homeroom follows from Locker: a dependency inside the table
    source = [{"Student_ID": student_id(i), "Locker": ("North", "South", "West")[i % 3],
               "Homeroom": f"R{101 + i % 3}", **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", source)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Homeroom"] == "" and row["Locker"] for row in written)
    assert not {row["Homeroom"] for row in source} & {v for row in written for v in row.values()}
    changed = _section(report, "What changed")
    assert any("link between" in line and "Locker" in line for line in _lines(changed, "Homeroom")), changed
    assert rounds == ["Homeroom"]


# --------------------------------------------------------------------------- a template inside the table
def _template_roster(original: Path, **extra) -> list[dict[str, str]]:
    rows = [{"Student_ID": student_id(i), "First_Name": FIRST[i % 6], "Last_Name": LAST[i % 5],
             "Student_Email": f"{FIRST[i % 6].lower()}.{LAST[i % 5].lower()}.{student_id(i).lower()}@example.org",
             **{k: v(i) for k, v in extra.items()}, **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def test_a_template_column_is_nulled_and_the_link_reported(tmp_path):
    """Student_Email is built from First_Name, Last_Name and Student_ID. A template over person names keeps
    the source's grouping, so sharing mode refuses it; the column the template builds is nulled."""
    original = tmp_path / "original"
    source = _template_roster(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    assert all(row["Student_Email"] == "" for row in written)
    assert all(row["First_Name"] and row["Last_Name"] for row in written)
    changed = _section(report, "What changed")
    assert any("Student_Email is no longer built from First_Name" in line for line in changed.splitlines()), changed
    assert _lines(_section(report, "What needs attention"), "Student_Email")
    # no real e-mail address, nor any real local part, appears anywhere in the output folder
    text = _output_text(out).casefold()
    assert not [row["Student_Email"] for row in source if row["Student_Email"].split("@")[0] in text]


def test_a_template_input_nulled_for_its_own_reason_takes_its_link_and_leaks_nothing(tmp_path, monkeypatch):
    """First_Name is refused while it is generated (its own problem, not the template's): it is nulled, the e-mail
    address is no longer built from it, and no real first name reaches the output, inside an address or anywhere."""
    rounds = _refused_while_generated(monkeypatch, "First_Name")
    original = tmp_path / "original"
    source = _template_roster(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["First_Name"] == "" for row in written)
    text = _output_text(out).casefold()
    assert not [name for name in FIRST if name.casefold() in text]
    assert not [row["Student_Email"] for row in source if row["Student_Email"].casefold() in text]
    changed = _section(report, "What changed")
    assert any("no longer built from" in line and "First_Name" in line for line in changed.splitlines()), changed
    assert rounds and rounds[0] == "First_Name"


# --------------------------------------------------------------------------- relationships nulling would break
def _refused_while_generated(monkeypatch, column: str) -> list[str]:
    """Make generation refuse ``column`` (while it is not nulled) the way the sharing generators do; returns one entry
    per generation of a table holding it."""
    generate = pipeline.generate_table
    rounds: list[str] = []

    def refusing(profile, ctx):  # noqa: ANN001
        held = [p for p in profile.columns if p.name == column]
        if held and held[0].role not in ("null", "excluded"):
            rounds.append(column)
            raise ColumnRefusal(f"sharing mode: column {column!r} has no month-preserving date cover",
                                code="sharing_date_no_cover", column=column, index=held[0].index,
                                plain="its dates cannot be moved enough within their months to hide the earliest "
                                      "and latest real dates")
        return generate(profile, ctx)

    monkeypatch.setattr(pipeline, "generate_table", refusing)
    return rounds


def _advisors(original: Path) -> None:
    _write(original / "advisors.csv", [{"Advisor_ID": f"ADV{700 + a}", "Office": ("East", "West")[a % 2]}
                                       for a in range(12)])


#: column -> (rows of roster.csv, extra input files, CLI flags): each column is a key or in a relationship that reaches
#: past its own table, and also fixes another column of the table (Group, a dependency inside it)
KEYS = {
    "key": ("Student_ID", lambda i: {"Student_ID": student_id(i), "Group": f"G{i % 4}"}, None, ()),
    "join": ("Advisor_ID", lambda i: {"Student_ID": student_id(i), "Advisor_ID": f"ADV{700 + i % 12}",
                                      "Group": f"G{i % 3}"}, _advisors, ()),
    "alias": ("Student_Number", lambda i: {"Student_ID": student_id(i), "Student_Number": f"N{88000 + i * 7}",
                                           "Group": f"G{i % 4}"}, None, ("--id-col", "Student_Number")),
    "nesting": ("Class_ID", lambda i: {"Student_ID": student_id(i), "Class_ID": f"C{300 + i % 12}",
                                       "Group": f"G{i % 3}"}, None, ("--id-col", "Class_ID")),
}


@pytest.mark.parametrize("kind", sorted(KEYS))
def test_a_key_join_alias_or_nesting_column_still_leaves_the_file_out(tmp_path, monkeypatch, kind):
    column, row, more, flags = KEYS[kind]
    rounds = _refused_while_generated(monkeypatch, column)
    original = tmp_path / "original"
    _write(original / "roster.csv", [{**row(i), **_scores(i)} for i in range(N)])
    if more:
        more(original)
    result, out, report = _cli(original, tmp_path, *flags)
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists()
    assert "roster.csv" in _failures(report) and "sharing_date_no_cover" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")
    # each file holding it generated once and left out (advisors.csv holds the joined IDs too): no round of nulling
    assert rounds == [column] * (2 if more else 1)


def test_a_dependency_column_in_a_declared_rule_still_leaves_the_file_out(tmp_path):
    """Join_Year is in a dependency (nullable alone) and in a rule the operator declared: the rule wins."""
    original = tmp_path / "original"
    _write(original / "roster.csv", [{"Student_ID": student_id(i), "Year_of_Study": YEARS[i // 15],
                                      "Join_Year": str(2025 - i // 15), "Grad_Year": str(2029 - i // 15 + i % 2),
                                      **_scores(i)} for i in range(N)])
    run = execute(Settings(mode="sharing", seed=573291, rules=["Join_Year < Grad_Year"]), original,
                  tmp_path / "synthetic", tmp_path / "report")
    assert exit_code(run) == 3
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written
    assert not [p.name for tp in roster.tables for p in tp.columns if p.role == "null"]
