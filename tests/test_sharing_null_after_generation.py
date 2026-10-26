"""ADR-0001, second trigger: a blocking finding about one non-key column, found once the file is generated, nulls that
column and generates the file again with the same seed, instead of leaving the whole file out.

Every input here is invented. Tests run the CLI (or ``pipeline.execute``) and look only at what an operator or
recipient sees: the output folder, the report folder and the exit code.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.model import Message
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings
from datanonymous.profile import structure as structure_mod

from release_corpus import corpus_rows, read_records, student_id, write_corpus

N = 60
SEED = "573291"


def _locker(i: int) -> str:
    """A unique six-digit number per student: it looks like an ID (``id_candidates``, ``possible_sparse_code``)."""
    return str(100000 + (i * 7919) % 900000)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roster(original: Path, locker=_locker) -> list[dict[str, str]]:
    rows = [{"Student_ID": student_id(i), "Locker": locker(i), "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(N)]
    _write(original / "roster.csv", rows)
    return rows


def _scores(original: Path) -> None:
    _write(original / "scores.csv", [{"Student_ID": student_id(i), "Term": ("Fall", "Spring")[i % 2],
                                      "Points": f"{(6000 + (i * 389) % 3000) / 100:.2f}"} for i in range(N)])


def _cli(original: Path, base: Path, *flags: str, seed: str | None = SEED):
    out, report = base / "synthetic", base / "local-report"
    args = [str(original), "--out-dir", str(out), "--report-dir", str(report), *flags]
    if seed is not None:
        args += ["--seed", seed]
    result = CliRunner().invoke(main, args, catch_exceptions=False)
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


@pytest.mark.parametrize("locker", [_locker, lambda i: str(i + 1)], ids=["sparse_unique", "sequential"])
def test_an_id_like_number_column_that_is_no_key_is_nulled_and_the_file_written(tmp_path, locker):
    original = tmp_path / "original"
    source = _roster(original, locker)
    _scores(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    assert list(written[0]) == list(source[0])  # header and position kept
    assert all(row["Locker"] == "" for row in written)  # every cell missing
    assert all(row["Score"] and row["Other_Score"] for row in written)  # the rest of the file is usable
    assert (out / "scores.csv").is_file()
    # no real locker number appears anywhere in the output folder
    real = {row["Locker"] for row in source}
    cells = {v for row in written for v in row.values()}
    assert not real & cells
    if locker is _locker:
        assert not [v for v in real if v in _text(out)]
    # the report names the column, the plain reason and the alternative of declaring it as an ID
    line = _nulled_section(report)["Locker"]
    assert "automatically" in line and "--id-col Locker" in line, line
    if locker is _locker:  # found after generation (the sequential one is refused while it is protected, ADR-0001)
        assert "looks like an ID" in line, line
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
    assert all(row["Locker"] == "" for row in read_records(out / "roster.csv"))


def test_the_regenerated_file_is_reproducible_from_the_seed_the_report_gives(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    first, out, report = _cli(original, tmp_path / "random", seed=None)
    assert first.exit_code == 0, first.output
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    seed = md.split("- Seed: ", 1)[1].split()[0]
    assert "(random; pass --seed to reproduce)" in md
    again, out_again, _ = _cli(original, tmp_path / "again", seed=seed)
    assert again.exit_code == 0, again.output
    for name in ("roster.csv", "scores.csv"):
        assert (out / name).read_bytes() == (out_again / name).read_bytes(), name
    # the rest of the file is what the operator gets by nulling the column themselves with the same seed
    by_hand, out_by_hand, _ = _cli(original, tmp_path / "by-hand", "--null", "Locker", seed=seed)
    assert by_hand.exit_code == 0, by_hand.output
    assert read_records(out / "roster.csv") == read_records(out_by_hand / "roster.csv")


def test_a_file_with_row_level_findings_is_still_left_out(tmp_path):
    original, out, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original, "categorical_only")
    run = execute(Settings(mode="sharing", seed=573291), original, out, report)
    refused = {f.rel_path for f in run.files
               if any(m.code == "row_matches_report_only" for m in f.all_messages())}
    assert refused
    assert all(not f.written for f in run.files if f.rel_path in refused)
    assert not {p.name for p in out.glob("*.csv")} & refused
    assert not any(p.role == "null" for f in run.files for tp in f.tables for p in tp.columns)
    assert exit_code(run) == 3
    failures = _failures(report)
    assert all(name in failures for name in refused)


def test_a_row_level_finding_next_to_an_id_like_column_still_leaves_the_file_out(tmp_path):
    original, out, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original, "categorical_only")
    run = execute(Settings(mode="sharing", seed=573291), original, out, report)
    refused = sorted(f.rel_path for f in run.files
                     if any(m.code == "row_matches_report_only" for m in f.all_messages()))
    assert refused
    name = refused[0]
    rows = corpus_rows("categorical_only")[name]
    _write(original / name, [{**row, "Locker": _locker(k)} for k, row in enumerate(rows)])
    base = tmp_path / "with-locker"
    result, out, report = _cli(original, base)
    assert result.exit_code == 3, result.output
    assert not (out / name).exists()
    assert name in _failures(report)


def test_refuse_unsupported_leaves_the_file_out_as_before(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    result, out, report = _cli(original, tmp_path, "--refuse-unsupported")
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    failures = _failures(report)
    assert "roster.csv" in failures and "withheld_by_findings" in failures
    assert "## Nulled columns" not in (report / "REPORT.md").read_text(encoding="utf-8")


def test_refuse_unsupported_with_all_or_nothing_writes_nothing(tmp_path):
    original = tmp_path / "original"
    _roster(original)
    _scores(original)
    result, out, _ = _cli(original, tmp_path, "--refuse-unsupported", "--all-or-nothing")
    assert result.exit_code == 3, result.output
    assert not out.exists() or not any(out.iterdir())


def _finding_on(monkeypatch, column: str, *, stubborn: bool) -> list[str]:
    """Give every generated table holding ``column`` an invented blocking finding about it (while it is not nulled,
    or, ``stubborn``, always); returns the list of the tables that got it, one entry per generation."""
    finalize = structure_mod.finalize_rows
    rounds: list[str] = []

    def with_finding(profile, synth, rng_for, settings):  # noqa: ANN001
        messages = finalize(profile, synth, rng_for, settings)
        held = [p for p in profile.columns if p.name == column]
        if held and any(p.name == "Locker" for p in profile.columns) and (stubborn or held[0].role != "null"):
            rounds.append(profile.name or "")
            messages.append(Message("warning", "composite_key_duplicates", "an invented blocking finding",
                                    column=column))
        return messages

    monkeypatch.setattr(structure_mod, "finalize_rows", with_finding)
    return rounds


def test_a_column_scoped_finding_on_another_column_nulls_it_with_a_plain_reason(tmp_path, monkeypatch):
    _finding_on(monkeypatch, "Other_Score", stubborn=False)
    original = tmp_path / "original"
    _roster(original, lambda i: ("North", "South", "West")[i % 3])  # an ordinary column: nothing else blocks it
    _scores(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert all(row["Other_Score"] == "" and row["Score"] for row in written)
    line = _nulled_section(report)["Other_Score"]
    assert "automatically" in line and "privacy check" in line, line


@pytest.mark.parametrize("column", ["Student_ID", "Homeroom"], ids=["key", "dependency"])
def test_a_column_scoped_finding_on_a_key_or_relationship_column_still_leaves_the_file_out(
        tmp_path, monkeypatch, column):
    _finding_on(monkeypatch, column, stubborn=False)
    original = tmp_path / "original"
    # Homeroom follows from Locker (a dependency the copy keeps), so nulling either would break it
    rows = [{"Student_ID": student_id(i), "Locker": ("North", "South", "West")[i % 3],
             "Homeroom": f"R{101 + i % 3}", "Score": f"{(5100 + i * 73) / 100:.2f}"} for i in range(N)]
    _write(original / "roster.csv", rows)
    _scores(original)
    run = execute(Settings(mode="sharing", seed=573291), original, tmp_path / "synthetic", tmp_path / "local-report")
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written
    assert any("composite_key_duplicates" in r for r in roster.reasons), roster.reasons
    assert exit_code(run) == 3
    assert (tmp_path / "synthetic" / "scores.csv").is_file()
    assert not (tmp_path / "synthetic" / "roster.csv").exists()


def test_a_finding_that_nulling_does_not_clear_leaves_the_file_out(tmp_path, monkeypatch):
    """A column-scoped finding that comes back once its column is nulled: the file is left out, no endless rounds."""
    rounds = _finding_on(monkeypatch, "Other_Score", stubborn=True)
    original = tmp_path / "original"
    _roster(original, lambda i: ("North", "South", "West")[i % 3])
    _scores(original)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 3, result.output
    assert (out / "scores.csv").is_file()
    assert not (out / "roster.csv").exists()
    assert "roster.csv" in _failures(report)
    assert "composite_key_duplicates" in (report / "REPORT.md").read_text(encoding="utf-8")
    assert rounds == ["", ""]  # generated once as it is, once with the column nulled, then left out
