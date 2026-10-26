"""ADR-0001, refusals raised while values are generated: a non-key column sharing mode refuses once generation has
started (after its profile was protected) is nulled and the run generated again, instead of leaving the whole file
out. A key or relationship column, or ``--refuse-unsupported``, still leaves the file out.

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
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import read_records, student_id

N = 60


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roster(original: Path) -> list[dict[str, str]]:
    """Start_Month is a person date written as year and month only: its profile protects, but the date generator
    has no calendar cover for that precision and refuses it while the file is generated (``sharing_date_no_cover``)."""
    rows = [{"Student_ID": student_id(i), "Start_Month": f"2023-{9 + i % 3:02d}",
             "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def _scores(original: Path) -> None:
    _write(original / "scores.csv", [{"Student_ID": student_id(i), "Term": ("Fall", "Spring")[i % 2],
                                      "Points": f"{(6000 + (i * 389) % 3000) / 100:.2f}"} for i in range(N)])


def _cli(original: Path, base: Path, *flags: str):
    out, report = base / "synthetic", base / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(out), "--report-dir", str(report),
                                       "--seed", "573291", *flags], catch_exceptions=False)
    return result, out, report


def _nulled_section(report: Path) -> dict[str, str]:
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "## Nulled columns" in md, md
    section = md.split("## Nulled columns", 1)[1].split("\n## ", 1)[0]
    return {line.split("`")[3]: line for line in section.splitlines() if line.startswith("- ")}


def _failures(report: Path) -> str:
    return (report / "sharing-failures.csv").read_text(encoding="utf-8")


def _text(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.rglob("*") if p.is_file())


def test_a_non_key_column_refused_during_generation_is_nulled_and_the_file_written(tmp_path):
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    assert list(written[0]) == list(source[0])  # header and position kept
    assert all(row["Start_Month"] == "" for row in written)  # every cell missing
    assert all(row["Score"] and row["Student_ID"] for row in written)  # the rest of the file is usable
    assert (out / "scores.csv").is_file()
    # no real value of the nulled column appears anywhere in the output folder
    real = {row["Start_Month"] for row in source}
    assert not real & {v for row in written for v in row.values()}
    assert not [v for v in real if v in _text(out)]
    # the report names the column, gives the plain reason and its fix, and the file is not listed as left out
    line = _nulled_section(report)["Start_Month"]
    assert "automatically" in line and "earliest and latest real dates" in line, line
    assert "--exclude" in line, line
    assert "Start_Month" in result.output and "nulled automatically" in result.output, result.output
    assert "roster.csv" not in _failures(report)


def test_the_leak_scan_passes_and_nothing_blocks_release(tmp_path):
    original, out, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _roster(original)
    run = execute(Settings(mode="sharing", seed=573291), original, out, report)
    assert run.package["released"], [(m.code, m.text) for m in blocking_findings(run)]
    assert not blocking_findings(run)
    assert exit_code(run) == 0
    assert all(f.written for f in run.files)


def test_an_only_auto_nulled_run_satisfies_all_or_nothing(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    result, out, _ = _cli(original, tmp_path, "--all-or-nothing")
    assert result.exit_code == 0, result.output
    assert {p.name for p in out.glob("*.csv")} == {"roster.csv", "scores.csv"}
    assert all(row["Start_Month"] == "" for row in read_records(out / "roster.csv"))


def test_refuse_unsupported_leaves_the_file_out_as_before(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    result, out, report = _cli(original, tmp_path, "--refuse-unsupported")
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    assert "sharing_date_no_cover" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")


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


def test_the_same_refusal_on_a_key_column_still_leaves_the_file_out(tmp_path, monkeypatch):
    """A dependency column refused the same way is nulled (ADR-0001 amendment; test_sharing_in_table_links.py)."""
    column = "Student_ID"
    rounds = _refused_while_generated(monkeypatch, column)
    original = tmp_path / "original"
    _write(original / "roster.csv", [{"Student_ID": student_id(i), "Locker": ("North", "South", "West")[i % 3],
                                      "Homeroom": f"R{101 + i % 3}", "Score": f"{(5100 + i * 73) / 100:.2f}"}
                                     for i in range(N)])
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists()
    assert "roster.csv" in _failures(report) and "sharing_date_no_cover" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")
    assert rounds == [column]  # generated once and left out: no round of nulling


def test_a_non_key_column_refused_by_any_generator_is_nulled(tmp_path, monkeypatch):
    rounds = _refused_while_generated(monkeypatch, "Other_Score")
    original = tmp_path / "original"
    source = [{"Student_ID": student_id(i), "Locker": ("North", "South", "West")[i % 3],
               "Score": f"{(5100 + i * 73) / 100:.2f}",
               "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)]
    _write(original / "roster.csv", source)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Other_Score"] == "" and row["Score"] for row in written)
    assert not {row["Other_Score"] for row in source} & {v for row in written for v in row.values()}
    assert "automatically" in _nulled_section(report)["Other_Score"]
    assert rounds == ["Other_Score"]  # refused once, then generated with the column nulled
