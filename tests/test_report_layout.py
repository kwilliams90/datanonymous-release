"""Issue 13: the report leads with what was generated and what needs attention; the criteria and codes follow.

Every input is invented. Tests run the CLI and read only what an operator sees: REPORT.md, the console summary,
report.json, sharing-failures.csv and the output folder.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from click.testing import CliRunner

from datanonymous.cli import main

from release_corpus import student_id

BANNER = "PRIVATE – DO NOT SHARE"


def _roster(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Student_ID", "Score", "Other_Score"])
        for i in range(60):
            writer.writerow([student_id(i), f"{(5100 + i * 73) / 100:.2f}", f"{(4000 + (i * 571) % 3999) / 100:.2f}"])


def _messy_folder(base: Path) -> Path:
    original = base / "original"
    _roster(original)
    for name in ("a.pdf", "b.pdf", "docs/c.pdf", "scan.png"):
        (original / name).parent.mkdir(parents=True, exist_ok=True)
        (original / name).write_bytes(b"%PDF-1.4 not a table")
    (original / "broken.xlsx").write_bytes(b"this is not a workbook")
    return original


def _run(original: Path, base: Path, *flags: str):
    out, report = base / "synthetic", base / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291", *flags], catch_exceptions=False)
    return result, out, report


def _section(md: str, heading: str) -> str:
    return md.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _items(section: str) -> list[list[str]]:
    """Each top-level bullet of a section with its indented fix lines."""
    items: list[list[str]] = []
    for line in section.splitlines():
        if line.startswith("- "):
            items.append([line])
        elif line.startswith("  ") and items:
            items[-1].append(line)
    return items


def test_report_opens_with_the_banner_then_the_sections_in_order(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert md.splitlines()[0].lstrip("# ").startswith(BANNER)
    order = [md.index(f"\n## {h}\n") for h in ("What was generated", "What needs attention", "Appendix")]
    assert order == sorted(order)
    assert "\n## Verdict" not in md  # the verdict table is a subsection of the Appendix now
    appendix_at = md.index("\n## Appendix\n")
    assert md.index("| Check | Key | Result") > appendix_at
    assert md.index("Overall verdict:") > appendix_at
    assert md.index("### Per-file checks") > appendix_at


def test_what_was_generated_lists_files_rows_columns_and_the_output_folder(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    generated = _section(md, "What was generated")
    assert str(out).replace("\\", "/") in generated.replace("\\", "/")
    assert "1 file written" in generated
    row = next(line for line in generated.splitlines() if "roster.csv" in line)
    assert "60" in row and "3" in row


def test_non_table_files_are_one_routine_line_and_the_corrupt_workbook_needs_attention(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    generated = _section(md, "What was generated")
    attention = _section(md, "What needs attention")
    assert "4 non-table files skipped (3 .pdf, 1 .png)" in generated
    assert ".pdf" not in attention and "scan.png" not in attention
    assert "broken.xlsx" in attention
    # the full list is still in the appendix
    appendix = md.split("\n## Appendix\n", 1)[1]
    assert "scan.png" in appendix and "c.pdf" in appendix


def test_every_attention_item_has_a_plain_sentence_and_a_fix(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    attention = _section((report / "REPORT.md").read_text(encoding="utf-8"), "What needs attention")
    items = ["\n".join(chunk) for chunk in _items(attention)]
    assert items
    for item in items:
        assert re.search(r"\(`[a-z_0-9]+`\)", item), item  # the plain sentence, with its catalogue code
        assert "- Fix: " in item, item


def test_an_actionable_skipped_input_is_listed_with_a_fix(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    (original / "survey.sav").write_bytes(b"$FL2")
    (original / "notes.docx").write_text("hello", encoding="utf-8")
    result, out, report = _run(original, tmp_path)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    attention = _section(md, "What needs attention")
    assert "survey.sav" in attention and "Fix: " in attention and "(`skipped_unsupported_format`)" in attention
    assert "notes.docx" not in attention
    assert "1 non-table file skipped (1 .docx)" in _section(md, "What was generated")


def test_deselected_inputs_are_not_actionable(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    (original / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result, out, report = _run(original, tmp_path, "--exclude-path", "other.csv")
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "other.csv" not in _section(md, "What needs attention")
    assert "other.csv" not in _section(md, "What was generated")
    assert "other.csv" in md.split("\n## Appendix\n", 1)[1]


def test_a_clean_run_says_nothing_needs_attention(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    result, out, report = _run(original, tmp_path)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "Nothing needs attention." in _section(md, "What needs attention")


def test_an_auto_nulled_column_is_under_attention_with_its_reason_and_an_alternative(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    education = ("Qorvath Diploma", "Belmira Certificate", "Tandrel Degree")
    with (original / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        # two continuous scores, so no written row can equal a real one once the column is nulled
        writer.writerow(["Student_ID", "Parental_Education_Level", "Score", "Other_Score"])
        for i in range(60):
            writer.writerow([student_id(i), education[i // 20], f"{(5100 + i * 73) / 100:.2f}",
                             f"{(4000 + (i * 571) % 3999) / 100:.2f}"])
    result, out, report = _run(original, tmp_path)
    assert result.exit_code == 0, result.output
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    attention = _section(md, "What needs attention")
    item = "\n".join(next(c for c in _items(attention) if "Parental_Education_Level" in c[0]))
    assert "nulled automatically" in item and "describes another person" in item and "Fix: " in item


def test_the_console_uses_the_same_headings_and_counts(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    lines = result.output.splitlines()
    assert "What was generated" in lines and "What needs attention" in lines
    assert lines.index("What was generated") < lines.index("What needs attention")
    text = result.output
    assert "1 file written" in text and "4 non-table files skipped (3 .pdf, 1 .png)" in text
    attention = text.split("What needs attention", 1)[1]
    assert "broken.xlsx" in attention and "scan.png" not in attention
    assert text.isascii()


def test_report_json_manifest_and_failures_csv_keep_their_content(tmp_path):
    result, out, report = _run(_messy_folder(tmp_path), tmp_path)
    data = json.loads((report / "report.json").read_text(encoding="utf-8"))
    assert {"verdict", "totals", "files", "skipped", "failures", "errors", "warnings", "notes"} <= set(data)
    assert {s["path"] for s in data["skipped"]} >= {"a.pdf", "scan.png"}
    assert [f["source"] for f in data["failures"]] == ["broken.xlsx"]
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    assert {"files", "skipped", "settings", "seed"} <= set(manifest)
    with (report / "sharing-failures.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {r["input_path"] for r in rows} >= {"broken.xlsx", "a.pdf"}
    assert list(rows[0]) == ["input_path", "outcome", "reason"]
