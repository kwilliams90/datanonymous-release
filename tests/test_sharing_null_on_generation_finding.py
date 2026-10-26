"""ADR-0001, findings made while values are generated: a ``verify_values`` error or a leak-scan hit about one non-key
column (or a generation error that names one) nulls that column and the run is generated again, instead of leaving
the whole file out. The same on a key or relationship column, a finding about whole rows or the whole file, or
``--refuse-unsupported`` still leaves the file out.

No natural input makes these checks fire (they guard against generator bugs), so each test injects one: it copies
the real values of a column into the generated table (a real leak) and wraps the check so it reports them, the way
the check would. Every input here is invented. Tests run the CLI and look only at what an operator or recipient
sees: the output folder, the report folder and the exit code.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from datanonymous import pipeline
from datanonymous.cli import main
from datanonymous.errors import GenerationError
from datanonymous.model import Message
from datanonymous.privacy import leaks
from datanonymous.profile import sharing as sharing_profile
from datanonymous.profile import structure as structure_mod

from release_corpus import read_records, student_id

N = 60


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roster(original: Path) -> list[dict[str, str]]:
    """Other_Score has three decimals, so no other column can hold one of its real values by chance. Homeroom
    follows from Locker (a dependency the copy keeps), so nulling either would break it."""
    rows = [{"Student_ID": student_id(i), "Locker": ("North", "South", "West")[i % 3],
             "Homeroom": f"R{101 + i % 3}", "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(40000 + (i * 5711) % 39990) / 1000:.3f}"} for i in range(N)]
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
    # the section ends at the next heading of any level: the lines of later sections (notes, warnings) are not its own
    section = re.split(r"\n#+ ", md.split("## Nulled columns", 1)[1], maxsplit=1)[0]
    return {line.split("`")[3]: line for line in section.splitlines() if line.startswith("- ")}


def _failures(report: Path) -> str:
    return (report / "sharing-failures.csv").read_text(encoding="utf-8")


def _text(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.rglob("*") if p.is_file())


def _leak_real_values(monkeypatch, source: list[dict[str, str]], column: str) -> set[str]:
    """Once rows are finalised, write ``column``'s real values into the roster's generated table while the column is
    not nulled: a real leak for the checks to find. Returns the real values."""
    real = [row[column] for row in source]
    finalize = structure_mod.finalize_rows

    def leaking(profile, synth, rng_for, settings):  # noqa: ANN001
        messages = finalize(profile, synth, rng_for, settings)
        held = [p for p in profile.columns if p.name == column and p.role not in ("null", "excluded")]
        if held and any(p.name == "Homeroom" for p in profile.columns):
            for c in synth.columns:
                if c.index == held[0].index:
                    c.values = np.array([real[i % len(real)] for i in range(synth.n_rows)], dtype=object)
        return messages

    monkeypatch.setattr(structure_mod, "finalize_rows", leaking)
    return set(real)


def _scan_finds(monkeypatch, real: set[str], column: str, *extra: Message) -> list[str]:
    """The table leak scan reports a copied real value of ``column`` (plus ``extra`` findings when it does); returns
    one entry per hit."""
    scan = leaks.scan_table
    hits: list[str] = []

    def scanning(tp, synth):  # noqa: ANN001
        out = scan(tp, synth)
        for c in synth.columns:
            if c.name == column and {str(v) for v in c.values} & real:
                hits.append(column)
                out.append(Message("error", "copied_value", f"{leaks.LEAK_PREFIX}synthetic value(s) equal a real value",
                                   column=column, fix=f"--null {column}"))
                out.extend(extra)
        return out

    monkeypatch.setattr(leaks, "scan_table", scanning)
    return hits


def _verify_finds(monkeypatch, real: set[str] | None, column: str, *, written: bool) -> list[str]:
    """``verify_values`` reports a restored person-date extreme for ``column`` when its values hold a real one (any
    value, ``real`` None): on the generated values, or (``written``) only on the values re-read from the written
    file."""
    verify = sharing_profile.verify_values
    calls: list[str] = []
    hits: list[str] = []

    def verifying(profile, values):  # noqa: ANN001
        out = verify(profile, values)
        if profile.name == column and profile.role not in ("null", "excluded"):
            calls.append(column)
            if (len(calls) % 2 == 0) == written and (real is None or {str(v) for v in values} & real):
                hits.append(column)
                out.append(Message("error", "sharing_date_extreme_restored",
                                   "generated value(s) restored an original normalized person-date extreme",
                                   column=column, fix="null or exclude this column"))
        return out

    monkeypatch.setattr(sharing_profile, "verify_values", verifying)
    return hits


def _assert_nulled_and_written(result, out: Path, report: Path, source: list[dict[str, str]], real: set[str]) -> None:
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    assert list(written[0]) == list(source[0])  # header and position kept
    assert all(row["Other_Score"] == "" for row in written)  # every cell missing
    assert all(row["Score"] and row["Student_ID"] for row in written)  # the rest of the file is usable
    assert (out / "scores.csv").is_file()
    # no real value of the nulled column appears anywhere in the output folder
    assert not real & {v for row in written for v in row.values()}
    assert not [v for v in real if v in _text(out)]
    line = _nulled_section(report)["Other_Score"]
    assert "automatically" in line, line
    assert "Other_Score" in result.output and "nulled automatically" in result.output, result.output
    assert not (report / "sharing-failures.csv").exists() or "roster.csv" not in _failures(report)


def test_a_leak_scan_hit_on_a_non_key_column_nulls_it_and_writes_the_file(tmp_path, monkeypatch):
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, "Other_Score")
    hits = _scan_finds(monkeypatch, real, "Other_Score")
    result, out, report = _cli(original, tmp_path)
    _assert_nulled_and_written(result, out, report, source, real)
    assert "privacy check" in _nulled_section(report)["Other_Score"]
    assert hits == ["Other_Score"]  # found once, then generated with the column nulled


def test_a_verify_values_error_on_a_non_key_column_nulls_it_and_writes_the_file(tmp_path, monkeypatch):
    """Only the leak is injected: the real ``verify_values`` check finds the restored numeric extremes."""
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, "Other_Score")
    result, out, report = _cli(original, tmp_path)
    _assert_nulled_and_written(result, out, report, source, real)
    assert "rare extreme number" in _nulled_section(report)["Other_Score"]


def test_a_verify_values_error_on_the_written_values_nulls_the_column(tmp_path, monkeypatch):
    """The values re-read from the written file fail ``verify_values`` (the generated ones passed)."""
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    hits = _verify_finds(monkeypatch, None, "Other_Score", written=True)
    result, out, report = _cli(original, tmp_path)
    _assert_nulled_and_written(result, out, report, source, {row["Other_Score"] for row in source})
    assert "rare earliest or latest date" in _nulled_section(report)["Other_Score"]
    assert hits == ["Other_Score"]


def test_a_generation_error_naming_a_non_key_column_nulls_it(tmp_path, monkeypatch):
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    generate = pipeline.generate_table
    rounds: list[str] = []

    def failing(profile, ctx):  # noqa: ANN001
        held = [p for p in profile.columns if p.name == "Other_Score" and p.role not in ("null", "excluded")]
        if held:
            rounds.append("Other_Score")
            raise GenerationError("sharing mode: a value exceeds the source representation",
                                  code="sharing_date_out_of_range", column="Other_Score")
        return generate(profile, ctx)

    monkeypatch.setattr(pipeline, "generate_table", failing)
    result, out, report = _cli(original, tmp_path)
    _assert_nulled_and_written(result, out, report, source, {row["Other_Score"] for row in source})
    assert "could not be generated" in _nulled_section(report)["Other_Score"]
    assert rounds == ["Other_Score"]


@pytest.mark.parametrize("check", ["leak", "verify"])
def test_the_same_on_a_dependency_column_nulls_it_and_reports_the_link(tmp_path, monkeypatch, check):
    """ADR-0001 amendment: Homeroom's only relationship (it follows from Locker) stays inside the table."""
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, "Homeroom")
    if check == "leak":
        _scan_finds(monkeypatch, real, "Homeroom")
    else:
        _verify_finds(monkeypatch, real, "Homeroom", written=False)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    assert all(row["Homeroom"] == "" and row["Locker"] for row in read_records(out / "roster.csv"))
    assert not [v for v in real if v in _text(out)]
    assert "automatically" in _nulled_section(report)["Homeroom"]
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "the link between Locker and Homeroom is not kept" in md.split("## What changed", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize("check", ["leak", "verify"])
def test_the_same_on_a_key_column_still_leaves_the_file_out(tmp_path, monkeypatch, check):
    column = "Student_ID"
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, column)
    hits = (_scan_finds(monkeypatch, real, column) if check == "leak"
            else _verify_finds(monkeypatch, real, column, written=False))
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    assert "roster.csv" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")
    assert not [v for v in real if v in _text(out)]
    assert hits == [column]  # generated once and left out: no round of nulling


def test_a_generation_error_naming_a_key_column_still_leaves_the_file_out(tmp_path, monkeypatch):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    generate = pipeline.generate_table

    def failing(profile, ctx):  # noqa: ANN001
        if any(p.name == "Homeroom" for p in profile.columns):
            raise GenerationError("cannot generate the key", column="Student_ID")
        return generate(profile, ctx)

    monkeypatch.setattr(pipeline, "generate_table", failing)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra", [
    Message("error", "real_full_name", f"{leaks.LEAK_PREFIX}a row spells a real person's full name"),
    Message("error", "row_matches_remaining", "rows equal real rows", column="Other_Score"),
    Message("error", "copied_value", f"{leaks.LEAK_PREFIX}a real value", column="Not_A_Column"),
], ids=["file", "rows", "unknown-column"])
def test_a_row_or_file_level_hit_still_leaves_the_file_out(tmp_path, monkeypatch, extra):
    """Alongside a hit on a non-key column: one finding the column cannot answer for is enough to leave it out."""
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, "Other_Score")
    hits = _scan_finds(monkeypatch, real, "Other_Score", extra)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    assert "roster.csv" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")
    assert not [v for v in real if v in _text(out)]
    assert hits == ["Other_Score"]


@pytest.mark.parametrize("check", ["leak", "verify"])
def test_refuse_unsupported_leaves_the_file_out_as_before(tmp_path, monkeypatch, check):
    original = tmp_path / "original"
    source = _roster(original)
    _scores(original)
    real = _leak_real_values(monkeypatch, source, "Other_Score")
    hits = (_scan_finds(monkeypatch, real, "Other_Score") if check == "leak"
            else _verify_finds(monkeypatch, real, "Other_Score", written=False))
    result, out, report = _cli(original, tmp_path, "--refuse-unsupported")
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    assert "roster.csv" in _failures(report)
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")
    assert not [v for v in real if v in _text(out)]
    assert hits == ["Other_Score"]
