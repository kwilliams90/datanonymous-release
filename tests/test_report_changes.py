"""Issue 18: "What changed" lists the deliberate privacy changes, in the report with counts and in the README without.

Every input is invented. Tests run the CLI and read what an operator sees (REPORT.md, the console) and what a
recipient sees (the output folder's README.md).
"""

from __future__ import annotations

import csv
from pathlib import Path

from click.testing import CliRunner

from datanonymous import messages
from datanonymous.cli import main

from release_corpus import student_id

BANNER = "PRIVATE – DO NOT SHARE"
RARE = ("Qorvath", "Belmira", "Tandrel", "Ovwick")
EDUCATION = ("Qorvath Diploma", "Belmira Certificate", "Tandrel Degree")


def _roster(folder: Path, n: int = 120) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Student_ID", "Birth_Date", "Program", "Score", "Other_Score", "Parental_Education_Level",
                         "Notes"])
        for i in range(n):
            program = RARE[i] if i < len(RARE) else ("Alpine", "Boreal")[i % 2]
            # Score is continuous; the first row alone holds the extreme
            score = "9999.99" if i == 0 else f"{(5100 + i * 73) / 100:.2f}"
            writer.writerow([student_id(i), f"2008-{1 + i % 12:02d}-{1 + (i * 7) % 28:02d}", program, score,
                             f"{(4000 + (i * 571) % 3999) / 100:.2f}", EDUCATION[i % 3], f"note {i}"])


def _run(base: Path, *flags: str):
    original = base / "original"
    _roster(original)
    out, report = base / "synthetic", base / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291", *flags], catch_exceptions=False)
    return result, out, report


def _section(md: str, heading: str) -> str:
    return md.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _changed(md: str) -> str:
    return _section(md, "What changed")


def test_what_changed_sits_between_generated_and_attention_with_one_plain_line_per_change(tmp_path):
    result, out, report = _run(tmp_path, "--null", "Notes")
    assert result.exit_code == 0, result.output
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert md.index("\n## What was generated\n") < md.index("\n## What changed\n") < md.index("\n## What needs attention\n")
    changed = _changed(md)
    lines = [ln for ln in changed.splitlines() if ln.startswith("- ")]
    assert any("Birth_Date" in ln and "dates shifted; month patterns kept" in ln for ln in lines), changed
    assert any("Program" in ln and "4 rare categories merged into common ones" in ln for ln in lines), changed
    assert any("Score" in ln and "numeric extremes removed" in ln for ln in lines), changed
    assert any("Parental_Education_Level" in ln and "nulled automatically" in ln for ln in lines), changed
    assert any("Notes" in ln and "nulled by you" in ln for ln in lines), changed
    # one line per change: no column is listed twice for the same change
    assert len(lines) == len(set(lines))


def test_nulled_columns_are_distinguished_and_changes_are_not_problems(tmp_path):
    result, out, report = _run(tmp_path, "--null", "Notes")
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    changed, attention = _changed(md), _section(md, "What needs attention")
    auto = next(ln for ln in changed.splitlines() if "Parental_Education_Level" in ln)
    mine = next(ln for ln in changed.splitlines() if "Notes" in ln)
    assert "automatically" in auto and "by you" not in auto
    assert "by you" in mine and "automatically" not in mine
    # the auto-nulled column is still a problem to decide on; the other changes are not
    assert "Parental_Education_Level" in attention
    assert "dates shifted" not in attention and "rare categories" not in attention
    assert "numeric extremes" not in attention and "Notes" not in attention


def test_the_console_summary_lists_what_changed_between_the_other_headings(tmp_path):
    result, out, report = _run(tmp_path, "--null", "Notes")
    lines = result.output.splitlines()
    assert lines.index("What was generated") < lines.index("What changed") < lines.index("What needs attention")
    text = result.output.split("What changed", 1)[1].split("What needs attention", 1)[0]
    assert "dates shifted; month patterns kept" in text and "4 rare categories merged into common ones" in text
    assert text.isascii()


def test_a_run_with_no_changes_has_no_what_changed_section(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    with (original / "plain.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Student_ID", "Program"])
        for i in range(60):
            writer.writerow([student_id(i), ("Alpine", "Boreal")[i % 2]])
    out, report = tmp_path / "synthetic", tmp_path / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291"], catch_exceptions=False)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "\n## What changed\n" not in md and "What changed" not in result.output


def test_the_readme_lists_the_same_transformations_without_counts_or_reasons(tmp_path):
    result, out, report = _run(tmp_path, "--null", "Notes")
    readme = (out / "README.md").read_text(encoding="utf-8")
    changed = _changed(readme)
    assert "dates shifted; month patterns kept" in changed
    assert "rare categories merged into common ones" in changed
    assert "numeric extremes removed" in changed
    assert "Parental_Education_Level" in changed and "Notes" in changed
    assert "is empty" in changed
    assert not any(ch.isdigit() for ch in changed), changed
    for word in ("automatically", "by you", "because", "describes another person", "--null"):
        assert word not in changed
    assert "rows per person" not in changed or "number" not in changed  # no source facts about counts


def test_a_code_registered_as_a_change_appears_without_report_code(tmp_path, monkeypatch):
    from datanonymous import pipeline
    from datanonymous.config import Settings
    from datanonymous.model import Message
    from datanonymous.report import report as rep

    monkeypatch.setitem(messages.CATALOGUE, "test_only_change",
                        messages.Entry(messages.CHANGE, "Something was deliberately altered for the test."))
    original = tmp_path / "original"
    _roster(original)
    run = pipeline.execute(Settings(seed=573291), original, tmp_path / "out", tmp_path / "rep")
    run.files[0].messages.append(Message("info", "test_only_change", "internal detail 42"))
    md = rep.run_markdown(run, rep.run_json(run))
    assert "Something was deliberately altered for the test." in _changed(md)
    assert "internal detail 42" not in _changed(md)
    assert "Something was deliberately altered" not in _section(md, "What needs attention")
