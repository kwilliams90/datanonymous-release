"""ADR-0001: sharing mode nulls a non-key column it cannot protect instead of leaving the file out.

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
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import read_records, student_id

N = 60
EDUCATION = ("Qorvath Diploma", "Belmira Certificate", "Tandrel Degree")  # each held by 20 students
COLORS = ("Zerulean", "Vintaro", "Malbrisk", "Quellow", "Ostrine", "Pardune", "Ferrisk", "Lumbray")  # 7-8 each
FEES = {0: "3.15", 1: "7.85"}  # two values, five students each, everyone else blank

#: column -> (value of student i, words of the plain reason, words of the old jargon refusal, catalogue code)
CASES = {
    "Parental_Education_Level": (lambda i: EDUCATION[i // 20], "describes another person",
                                 "intrinsic attribute of another person", "sharing_other_person_attribute"),
    "Homeroom_Color": (lambda i: COLORS[i % 8], "too few people share",
                       "no category or missing value supported by enough distinct people", "sharing_no_support"),
    "Bus_Fee": (lambda i: FEES[i // 5] if i < 10 else "", "highest and lowest values",
                "no feasible numeric support after removing unsupported original extrema", "sharing_numeric_no_support"),
}


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _roster(original: Path, column: str, extra: dict[str, object] | None = None) -> list[dict[str, str]]:
    value = CASES[column][0]
    rows = [{"Student_ID": student_id(i), column: value(i),
             "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}",
             **{k: v(i) for k, v in (extra or {}).items()}} for i in range(N)]
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


def _cells(out: Path) -> set[str]:
    return {v for path in out.rglob("*.csv") for row in read_records(path) for v in row.values()}


def _text(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.rglob("*") if p.is_file())


@pytest.mark.parametrize("column", sorted(CASES))
def test_an_unprotectable_non_key_column_is_nulled_and_the_file_written(tmp_path, column):
    original = tmp_path / "original"
    source = _roster(original, column)
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N
    assert list(written[0]) == list(source[0])  # header and position kept
    assert all(row[column] == "" for row in written)  # every cell missing
    assert all(row["Score"] for row in written)  # the rest of the file is usable
    # no real value of the nulled column appears anywhere in the output folder
    real = {row[column] for row in source} - {""}
    assert not real & _cells(out)
    text = _text(out)
    assert not [v for v in real if not v.replace(".", "").isdigit() and v in text]
    # the report lists it as a warning naming the column and the plain reason
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "nulled automatically" in md and column in md and CASES[column][1] in md
    assert "roster.csv" not in (report / "sharing-failures.csv").read_text(encoding="utf-8")


def test_the_leak_scan_passes_and_nothing_blocks_release(tmp_path):
    original, out, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _roster(original, "Parental_Education_Level")
    run = execute(Settings(mode="sharing", seed=573291), original, out, report)
    assert run.package["released"], [(m.code, m.text) for m in blocking_findings(run)]
    assert not blocking_findings(run)
    assert exit_code(run) == 0
    assert all(f.written for f in run.files)


def test_an_only_auto_nulled_run_satisfies_all_or_nothing(tmp_path):
    original = tmp_path / "original"
    _roster(original, "Parental_Education_Level")
    _scores(original)
    result, out, _ = _cli(original, tmp_path, "--all-or-nothing")
    assert result.exit_code == 0, result.output
    assert {p.name for p in out.glob("*.csv")} == {"roster.csv", "scores.csv"}
    assert all(row["Parental_Education_Level"] == "" for row in read_records(out / "roster.csv"))


@pytest.mark.parametrize("column", sorted(CASES))
def test_refuse_unsupported_leaves_the_file_out_as_before(tmp_path, column):
    original = tmp_path / "original"
    _roster(original, column)
    _scores(original)
    result, out, report = _cli(original, tmp_path, "--refuse-unsupported")
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists() and (out / "scores.csv").is_file()
    failures = (report / "sharing-failures.csv").read_text(encoding="utf-8")
    assert "roster.csv" in failures and CASES[column][3] in failures and CASES[column][2] not in failures


def test_refuse_unsupported_with_all_or_nothing_writes_nothing(tmp_path):
    original = tmp_path / "original"
    _roster(original, "Parental_Education_Level")
    _scores(original)
    result, out, _ = _cli(original, tmp_path, "--refuse-unsupported", "--all-or-nothing")
    assert result.exit_code == 3, result.output
    assert not out.exists() or not any(out.iterdir())


def test_refuse_unsupported_is_accepted_as_a_config_key(tmp_path):
    original = tmp_path / "original"
    _roster(original, "Parental_Education_Level")
    config = tmp_path / "config.yaml"
    config.write_text("refuse_unsupported: true\n", encoding="utf-8")
    result, out, _ = _cli(original, tmp_path, "--config", str(config))
    assert result.exit_code == 3, result.output
    assert not (out / "roster.csv").exists()


def _reasons(base: Path, original: Path, **settings) -> list[str]:
    run = execute(Settings(mode="sharing", seed=573291, **settings), original, base / "synthetic", base / "report")
    assert exit_code(run) == 3
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written
    return roster.reasons


def test_a_column_in_a_declared_rule_still_leaves_the_file_out_with_todays_reason(tmp_path):
    original = tmp_path / "original"
    _write(original / "roster.csv", [{"Student_ID": student_id(i), "Parent_Income": f"{42000 + (i // 20) * 10000}.50",
                                      "Score": f"{(5100 + i * 73) / 100:.2f}"} for i in range(N)])
    rules = ["Parent_Income >= Score"]
    reasons = _reasons(tmp_path / "default", original, rules=rules)
    assert any("Parent_Income" in r and "intrinsic attribute of another person" in r for r in reasons), reasons
    assert reasons == _reasons(tmp_path / "refuse", original, rules=rules, refuse_unsupported=True)


def test_a_column_in_a_dependency_inside_its_table_is_nulled_and_the_link_reported(tmp_path):
    """ADR-0001 amendment: a dependency inside the table no longer stops nulling (keys, joins and declared rules do,
    test_sharing_in_table_links.py); the report says which link is not kept."""
    original = tmp_path / "original"
    source = _roster(original, "Homeroom_Color", {"Homeroom": lambda i: f"R{101 + i % 8}"})
    result, out, report = _cli(original, tmp_path)
    assert result.exit_code == 0, result.output
    written = read_records(out / "roster.csv")
    assert len(written) == N and all(row["Homeroom_Color"] == "" and row["Score"] for row in written)
    assert not ({row["Homeroom_Color"] for row in source} | {row["Homeroom"] for row in source}) & _cells(out)
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    changed = md.split("\n## What changed\n", 1)[1].split("\n## ", 1)[0]
    assert "link between" in changed and "Homeroom_Color" in changed and "is not kept" in changed, changed


def test_the_report_tells_operator_nulls_from_automatic_ones(tmp_path):
    original = tmp_path / "original"
    # an operator-nulled column of its own: Score and Other_Score stay, so no written row can equal a real one
    _roster(original, "Parental_Education_Level", {"Third_Score": lambda i: f"{(3000 + (i * 211) % 2999) / 100:.2f}"})
    result, out, report = _cli(original, tmp_path, "--null", "Third_Score")
    assert result.exit_code == 0, result.output
    assert all(row["Third_Score"] == "" and row["Parental_Education_Level"] == ""
               for row in read_records(out / "roster.csv"))
    md = (report / "REPORT.md").read_text(encoding="utf-8")
    attention = md.split("\n## What needs attention\n", 1)[1].split("\n## ", 1)[0]
    assert "Parental_Education_Level" in attention and "Other_Score" not in attention  # only the automatic one
    section = md.split("### Nulled columns", 1)[1].split("\n##", 1)[0]  # the Appendix lists both kinds
    by_column = {line.split("`")[3]: line for line in section.splitlines() if line.startswith("- ")}
    assert "by you" in by_column["Third_Score"] and "--null" in by_column["Third_Score"]
    auto = by_column["Parental_Education_Level"]
    assert "automatically" in auto and "describes another person" in auto


@pytest.mark.parametrize("also_null", [[], ["--null", "Other_Score"]], ids=["written", "row_matches"])
def test_an_automatic_null_gives_what_nulling_the_column_by_hand_gives(tmp_path, also_null):
    """The column takes the same path as --null: same output, and the same privacy checks of the rest of the file (a
    roster left with only Score has synthetic rows equal to real ones, so it is left out either way)."""
    original = tmp_path / "original"
    _roster(original, "Parental_Education_Level")
    auto, out_auto, report_auto = _cli(original, tmp_path / "auto", *also_null)
    by_hand, out_by_hand, report_by_hand = _cli(original, tmp_path / "by-hand", *also_null,
                                                "--null", "Parental_Education_Level")
    assert auto.exit_code == by_hand.exit_code == (3 if also_null else 0), (auto.output, by_hand.output)
    assert sorted(p.name for p in out_auto.glob("*.csv")) == sorted(p.name for p in out_by_hand.glob("*.csv"))
    for path in out_auto.glob("*.csv"):
        assert path.read_bytes() == (out_by_hand / path.name).read_bytes()
    if also_null:
        assert "roster.csv" in (report_auto / "sharing-failures.csv").read_text(encoding="utf-8")
