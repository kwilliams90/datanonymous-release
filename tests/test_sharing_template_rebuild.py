"""Sharing mode writes a column built by a template (an e-mail address built from a student's names and ID) by applying
the template to the row's fictional values, when that is safe, instead of nulling it (issue #30, ADR-0001 amendment).

Safe means: every input is fictional in the output (generated names, minted IDs), every literal part is one the
generator would write anyway (the domain is always the generator's fictional domain, never the real one), and the
template's shape is held by at least --min-cell people. Otherwise the column is nulled and the lost link reported.

Every input here is invented. Tests run the CLI (or ``pipeline.execute``) and look only at what an operator or
recipient sees: the output folder, the report folder and the exit code.
"""

from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.generate.pii import RESERVED_DOMAINS
from datanonymous.pipeline import execute, exit_code

from release_corpus import read_records, student_id

REPO = Path(__file__).resolve().parents[1]
N = 60
FIRST = ("Zorane", "Nelutu", "Pavira", "Quoste", "Belank", "Tirvan")  # 10 students each
LAST = ("Pelvavo", "Norrulo", "Quindre", "Ostravi", "Melunde")  # 12 students each
#: a real-looking school-district domain held by every student: it names an institution, so it is never released
DISTRICT = "riverbendisd.k12.tx.us"


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


def _roster(original: Path, email) -> list[dict[str, str]]:  # noqa: ANN001
    rows = [{"Student_ID": student_id(i), "First_Name": FIRST[i % 6], "Last_Name": LAST[i % 5],
             "Student_Email": email(i), **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def _built(i: int, domain: str) -> str:
    return f"{FIRST[i % 6].lower()}.{LAST[i % 5].lower()}.{student_id(i).lower()}@{domain}"


def _low(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum()).lower()


def _local(row: dict[str, str], *columns: str) -> str:
    return ".".join(_low(row[c]) for c in columns)


def _fictional_domain(address: str) -> bool:
    domain = address.rpartition("@")[2]
    return any(domain == d or domain.endswith("." + d) for d in RESERVED_DOMAINS)


def test_a_template_e_mail_is_rebuilt_from_each_rows_fictional_values(tmp_path):
    original = tmp_path / "original"
    source = _roster(original, lambda i: _built(i, "example.org"))
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    domains = set()
    for row in written:
        local, _, domain = row["Student_Email"].rpartition("@")
        assert local == _local(row, "First_Name", "Last_Name", "Student_ID"), row
        domains.add(domain)
    assert len(domains) == 1 and _fictional_domain(written[0]["Student_Email"]), domains
    # the inputs are fictional: no real name or ID, so no real address or local part
    real_ids = {row["Student_ID"] for row in source}
    assert not {row["Student_ID"] for row in written} & real_ids
    assert not {row["First_Name"] for row in written} & set(FIRST)
    assert not {row["Last_Name"] for row in written} & set(LAST)
    text = _output_text(out).casefold()
    assert not [row["Student_Email"] for row in source if row["Student_Email"].split("@")[0] in text]
    # What changed says the e-mail was rebuilt, not that its link was lost; nothing needs attention for it
    changed = _section(report, "What changed")
    assert any("Student_Email was rebuilt from the fictional First_Name, Last_Name and Student_ID" in line
               for line in _lines(changed, "Student_Email")), changed
    assert "no longer built from" not in changed
    assert not _lines(_section(report, "What needs attention"), "Student_Email")
    # the shareable README says it in general terms, without naming the source's linked columns
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "was rebuilt from their fictional values" in readme
    assert "rebuilt from the fictional First_Name" not in readme
    assert "A link between columns of this file is not kept" not in readme


def test_a_real_institution_domain_held_by_every_row_is_never_released(tmp_path):
    original = tmp_path / "original"
    source = _roster(original, lambda i: _built(i, DISTRICT))
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Student_Email"] for row in written)
    assert all(_fictional_domain(row["Student_Email"]) for row in written), written[:3]
    assert all(row["Student_Email"].rpartition("@")[0] == _local(row, "First_Name", "Last_Name", "Student_ID")
               for row in written)
    # neither the domain nor any part of it that names the district reaches the output folder
    text = _output_text(out).casefold()
    assert DISTRICT not in text and "riverbendisd" not in text
    assert not [row["Student_Email"] for row in source if row["Student_Email"].split("@")[0] in text]
    assert "leak" not in (report / "sharing-failures.csv").read_text(encoding="utf-8").casefold()


def test_a_template_with_free_digits_is_still_nulled_and_its_link_reported(tmp_path):
    """The free digits after the names (pelvavo07, pelvavo123) copy the source's digit-run lengths, a literal part
    the generator would not write by itself: the e-mail address is nulled as before."""
    original = tmp_path / "original"
    source = _roster(original, lambda i: f"{FIRST[i % 6].lower()}.{LAST[i % 5].lower()}{(i * 37) % 1000}@{DISTRICT}")
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Student_Email"] == "" for row in written)
    assert all(row["First_Name"] and row["Last_Name"] for row in written)
    changed = _section(report, "What changed")
    assert any("Student_Email is no longer built from First_Name and Last_Name" in line
               for line in changed.splitlines()), changed
    assert "rebuilt from the fictional" not in changed
    assert _lines(_section(report, "What needs attention"), "Student_Email")
    text = _output_text(out).casefold()
    assert DISTRICT not in text
    assert not [row["Student_Email"] for row in source if row["Student_Email"].split("@")[0] in text]


def test_a_template_with_an_input_that_is_not_fictional_is_still_nulled(tmp_path):
    """Campus is an ordinary category whose real values are written: an address built from it is not built from
    fictional values only, so it is nulled and the lost link reported."""
    original = tmp_path / "original"
    campus = ("north", "south", "east")
    rows = [{"Student_ID": student_id(i), "First_Name": FIRST[i % 6], "Last_Name": LAST[i % 5],
             "Campus": campus[i % 3],
             "Student_Email": f"{FIRST[i % 6].lower()}.{LAST[i % 5].lower()}.{campus[i % 3]}@example.org",
             **_scores(i)} for i in range(N)]
    _write(original / "roster.csv", rows)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Student_Email"] == "" for row in written)
    changed = _section(report, "What changed")
    assert any("Student_Email is no longer built from" in line for line in changed.splitlines()), changed


def test_student_scores_writes_each_e_mail_from_the_rows_fictional_values(tmp_path):
    """Acceptance: example-data's student-scores.csv builds email as first.last.id@domain."""
    source = read_records(REPO / "example-data" / "student-scores.csv")
    run = execute(Settings(mode="sharing", seed=1), REPO / "example-data" / "student-scores.csv",
                  tmp_path / "o", tmp_path / "r")
    assert exit_code(run) == 0
    written = read_records(tmp_path / "o" / "student-scores.csv")
    assert len(written) == len(source)
    for row in written:
        assert row["email"] == _local(row, "first_name", "last_name", "id") + "@example.org", row
    assert not {row["email"] for row in written} & {row["email"] for row in source}
    real_locals = {row["email"].rpartition("@")[0].casefold() for row in source}
    assert not {row["email"].rpartition("@")[0] for row in written} & real_locals
    md = (tmp_path / "r" / "REPORT.md").read_text(encoding="utf-8")
    assert "email was rebuilt from the fictional first_name, last_name and id" in md
    assert "email is no longer built from" not in md
