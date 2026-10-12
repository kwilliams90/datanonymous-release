"""AREA G report: machine-checkable verdict, stable report.json keys, fix rendering, per-file reasons."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from conftest import EXAMPLES, run_cli
from datanonymous.report import report as rep_mod
from datanonymous.report.report import CRITERIA, fix_commands, fix_markdown

CRITERIA_KEYS = ["written", "layout_selfcheck", "rows", "header", "dtypes", "fidelity", "leak_scan",
                 "chance_matches", "warnings", "adjusted_flags"]


# --------------------------------------------------------------------------- fix rendering
def test_course_fix_renders_command_in_code_and_prose_plain() -> None:
    fix = "--force-text Course replaces the labels with invented ones (if they could identify someone)"
    md = fix_markdown(fix, ["Course"])
    assert md == "`--force-text Course` replaces the labels with invented ones (if they could identify someone)"
    assert fix_commands(fix, ["Course"]) == ["--force-text Course"]


@pytest.mark.parametrize("fix,cands,expected_md,expected_cmds", [
    ("--force-factor Expense_Semester", ["Expense_Semester"], "`--force-factor Expense_Semester`",
     ["--force-factor Expense_Semester"]),
    ("--null COL or --exclude COL", [], "`--null COL` or `--exclude COL`", ["--null COL", "--exclude COL"]),
    ("--force-factor parental level of education (or --org-code parental level of education) keeps the real labels",
     ["parental level of education"],
     '`--force-factor "parental level of education"` (or `--org-code "parental level of education"`) keeps the '
     "real labels",
     ['--force-factor "parental level of education"', '--org-code "parental level of education"']),
    ("remove --force-type for 'Age', or use --force-factor Age / --force-text Age", ["Age"],
     "remove `--force-type` for 'Age', or use `--force-factor Age` / `--force-text Age`",
     ["--force-type", "--force-factor Age", "--force-text Age"]),
    ("use --force-type Age=FLOAT", ["Age"], "use `--force-type Age=FLOAT`", ["--force-type Age=FLOAT"]),
    ("pass --encoding utf-8 or report this as a bug", [], "pass `--encoding utf-8` or report this as a bug",
     ["--encoding utf-8"]),
    ("pass --encoding if the text looks wrong", [], "pass `--encoding` if the text looks wrong", ["--encoding"]),
    ("raise --max-row-matches, or add columns that vary", [], "raise `--max-row-matches`, or add columns that vary",
     ["--max-row-matches"]),
    ("--no-strict writes it anyway (layout differences become warnings)", [],
     "`--no-strict` writes it anyway (layout differences become warnings)", ["--no-strict"]),
    ("re-run with -v to see the traceback, and report it", [],
     "re-run with `-v` to see the traceback, and report it", ["-v"]),
    ("--id-col COL (or --person-id COL) for each person/record ID column", [],
     "`--id-col COL` (or `--person-id COL`) for each person/record ID column", ["--id-col COL", "--person-id COL"]),
    ("otherwise --null p.csv::student_id or --exclude p.csv::student_id", ["p.csv::student_id", "student_id"],
     "otherwise `--null p.csv::student_id` or `--exclude p.csv::student_id`",
     ["--null p.csv::student_id", "--exclude p.csv::student_id"]),
    ("add a header row (the column names) as the first line of the file", [],
     "add a header row (the column names) as the first line of the file", []),
])
def test_fix_rendering(fix: str, cands: list[str], expected_md: str, expected_cmds: list[str]) -> None:
    assert fix_markdown(fix, cands) == expected_md
    assert fix_commands(fix, cands) == expected_cmds
    # no prose ever ends up inside a code span
    for span in re.findall(r"`([^`]+)`", fix_markdown(fix, cands)):
        assert span.startswith("-"), span


def test_code_spans_survive_backticks_and_pipes() -> None:
    assert rep_mod.code_span("a`b") == "``a`b``"
    assert fix_markdown("--null a|b", ["a|b"]) == '`--null "a|b"`'
    assert "\\|" in fix_markdown("see a | b", [])


# --------------------------------------------------------------------------- the example run report
def test_report_starts_with_a_machine_checkable_verdict(example_run) -> None:
    md = (example_run.rep / "REPORT.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    assert lines[0] == "# datanonymous run report" and lines[2] == "## Verdict"
    m = re.search(r"^Overall verdict: (PASS|WARN|FAIL)$", md, re.M)
    data = json.loads((example_run.rep / "report.json").read_text(encoding="utf-8"))
    assert m and m.group(1) == data["verdict"]["overall"] == "WARN"
    assert md.index("## Verdict") < md.index("## Files") < md.index("## Warnings")
    for key in CRITERIA_KEYS + ["run_errors", "run_warnings"]:
        assert f"`{key}`" in md.split("## Run")[0]
    assert "### Per-file checks" in md
    assert "Self-check coverage" in md and "line ending" in md


def test_report_json_schema_and_stable_keys(example_run) -> None:
    data = json.loads((example_run.rep / "report.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert {"schema_version", "tool", "version", "mode", "seed", "seed_was_random", "verdict", "totals", "files",
            "skipped", "failures", "errors", "warnings", "notes", "docs"} <= set(data)
    v = data["verdict"]
    assert {"overall", "rule", "exit_code", "criteria", "files", "selfcheck_coverage"} <= set(v)
    assert [c["key"] for c in v["criteria"]] == CRITERIA_KEYS + ["run_errors", "run_warnings"]
    assert [k for k, _, _ in CRITERIA] == CRITERIA_KEYS
    for f in v["files"]:
        assert list(f["checks"]) == CRITERIA_KEYS
        assert all(c["result"] in ("PASS", "WARN", "FAIL", "N/A", "INFO") for c in f["checks"].values())
    files = {f["source"]: f for f in data["files"]}
    sd = files["students_data.csv"]
    assert sd["rows"] == sd["rows_out"] == 62 and sd["checks"]["rows"] == "PASS"
    assert sd["checks"]["warnings"] == "WARN" and sd["checks"]["adjusted_flags"] == "INFO"
    assert files["StudentsPerformance.csv"]["checks"]["layout_selfcheck"] == "PASS"
    assert v["exit_code"] == 0
    expense = next(w for w in data["warnings"] if w.get("column") == "Expense_Semester")
    assert expense["fix_commands"] == ["--force-factor Expense_Semester"]


def test_fix_with_trailing_prose_is_rendered_correctly(tmp_path: Path) -> None:
    # (the small-file Course warning this checked was retired by D35; a D35 pseudonym note has a fix with prose
    # after the flag; example-data's SitKCSE regions became vocabulary words in D35 hardening pass 2)
    src = tmp_path / "in"
    src.mkdir()
    rows = ["Region,Score"] + [f"{'Zeta Shore' if i < 3 else 'Central'},{i}" for i in range(40)]
    (src / "regions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--seed", "1")
    assert res.exit_code == 0, res.output
    per_file = (tmp_path / "rep" / "files" / "regions.csv.md").read_text(encoding="utf-8")
    assert "Fix: `--force-factor Region` keeps the real values" in per_file
    assert "`--force-factor Region keeps" not in per_file


def test_per_file_reports_give_column_verdict_reasons(example_run) -> None:
    pj = json.loads((example_run.rep / "files" / "students_data.csv.json").read_text(encoding="utf-8"))
    assert pj["schema_version"] == "1.0" and list(pj["checks"]) == CRITERIA_KEYS
    cols = {c["name"]: c for c in pj["tables"][0]["columns"]}
    fid = cols["Expense_Semester"]["fidelity"]
    assert fid["verdict"] in ("ok", "adjusted") and "KS" in fid["reason"]
    assert any(ch["check"] == "ks" and ch["limit"] == 0.05 for ch in fid["checks"])
    assert "level share" in cols["Course"]["fidelity"]["reason"]
    md = (example_run.rep / "files" / "students_data.csv.md").read_text(encoding="utf-8")
    assert "## Checks" in md and "Column fidelity (verdict: reason)" in md
    assert re.search(r"- `Expense_Semester`: \*\*(ok|adjusted)\*\* - .*KS", md)
    assert "## Privacy" in md and "extreme exposure" in md
    assert pj["extreme_exposure"] and all(e["held_by_fewer_than"] == 10 for e in pj["extreme_exposure"])


def test_extreme_exposure_is_information_only(example_run) -> None:
    data = json.loads((example_run.rep / "report.json").read_text(encoding="utf-8"))
    notes = [n for n in data["notes"] if n["code"] == "extreme_exposure"]
    assert notes and all(n["level"] == "info" for n in notes)
    assert {n["file"] for n in notes} >= {"students_data.csv"}
    files = {f["source"]: f for f in data["files"]}
    assert files["StudentsPerformance.csv"]["status"] == "PASS"  # info never changes the status
    md = (example_run.rep / "REPORT.md").read_text(encoding="utf-8")
    assert "extreme exposure" in md and "`--no-exact-extremes students_data.csv::Expense_Accommodation`" in md
    assert "`--no-exact-extremes COL`" not in md  # concrete commands, never a placeholder
    assert ".." not in md.replace("...", "").replace("1..n", "")  # one period per sentence
    by_col = {(n["file"], n["column"], n["extreme"]): n for n in notes}
    age = by_col[("students_data.csv", "Age", "max")]
    assert age["how"] == "kept" and "fix" not in age  # discrete: every observed value is reproduced
    exp = by_col[("students_data.csv", "Expense_Accommodation", "max")]
    assert exp["how"] == "forced" and exp["fix_commands"] == ["--no-exact-extremes students_data.csv::Expense_Accommodation"]


def test_console_summary_is_ascii_with_verdict(example_run) -> None:
    out = example_run.result.output
    assert out.isascii()
    assert re.search(r"^Verdict: WARN", out, re.M)


# --------------------------------------------------------------------------- failures, skipped, header/self-check
def test_failures_have_exact_fixes_and_na_checks(tmp_path: Path) -> None:
    res = run_cli(EXAMPLES, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1",
                  "--force-type", "students_data.csv::Age=DATE")
    assert res.exit_code == 3
    md = (tmp_path / "r" / "REPORT.md").read_text(encoding="utf-8")
    assert "Overall verdict: FAIL" in md
    fail_section = md.split("## Failures")[1].split("##")[0]
    assert "`students_data.csv`: not written." in fail_section and "- Fix: " in fail_section
    assert "`--force-type`" in fail_section or "`--force-factor" in fail_section
    data = json.loads((tmp_path / "r" / "report.json").read_text(encoding="utf-8"))
    entry = next(f for f in data["verdict"]["files"] if f["source"] == "students_data.csv")
    assert entry["checks"]["written"]["result"] == "FAIL"
    assert entry["checks"]["rows"]["result"] == "N/A" and entry["checks"]["leak_scan"]["result"] == "N/A"
    failure = data["failures"][0]
    assert failure["fix_commands"] and all(c.startswith("--") for c in failure["fix_commands"])


def test_skipped_files_listed_with_reasons(tmp_path: Path, example_copy: Path) -> None:
    (example_copy / "notes.pdf").write_bytes(b"%PDF-1.4")
    (example_copy / "data.sav").write_bytes(b"x")
    res = run_cli(example_copy, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0
    md = (tmp_path / "r" / "REPORT.md").read_text(encoding="utf-8")
    skipped = md.split("## Skipped (not copied)")[1].split("##")[0]
    assert "`notes.pdf`: not a supported tabular file type (.pdf)" in skipped
    assert "`data.sav`: unsupported tabular format: SPSS file" in skipped


def test_excluded_column_makes_the_header_check_warn(tmp_path: Path) -> None:
    res = run_cli(EXAMPLES / "students_data.csv", "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r",
                  "--seed", "1", "--exclude", "Course")
    assert res.exit_code == 0
    data = json.loads((tmp_path / "r" / "report.json").read_text(encoding="utf-8"))
    checks = data["verdict"]["files"][0]["checks"]
    assert checks["header"]["result"] == "WARN" and "--exclude" in checks["header"]["detail"]
    assert checks["rows"]["result"] == "PASS"


def test_layout_mismatch_fails_or_warns_with_no_strict(tmp_path: Path, monkeypatch) -> None:
    from datanonymous.io import registry

    h = registry.FORMATS["csv"]
    monkeypatch.setitem(registry.FORMATS, "csv",
                        dataclasses.replace(h, layout_diff=lambda *a: ["delimiter: source ',', output ';'"]))
    src = tmp_path / "in"
    src.mkdir()
    shutil.copy(EXAMPLES / "students_data.csv", src / "students_data.csv")
    res = run_cli(src, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 3
    data = json.loads((tmp_path / "r" / "report.json").read_text(encoding="utf-8"))
    c = data["verdict"]["files"][0]["checks"]["layout_selfcheck"]
    assert c["result"] == "FAIL" and "delimiter" in c["detail"]
    res = run_cli(src, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "1", "--no-strict")
    assert res.exit_code == 0
    data = json.loads((tmp_path / "r2" / "report.json").read_text(encoding="utf-8"))
    c = data["verdict"]["files"][0]["checks"]["layout_selfcheck"]
    assert c["result"] == "WARN" and "--no-strict" in c["detail"]
    assert data["verdict"]["overall"] == "WARN"


def test_overall_verdict_matches_exit_code_rule(tmp_path: Path) -> None:
    from datanonymous.model import FileResult, Message, RunResult, Status

    run = RunResult("run", Path("in"), Path("o"), Path("r"), 1, False)
    assert rep_mod.verdict(run)["overall"] == "PASS"
    run.messages.append(Message("warning", "x", "w"))
    assert rep_mod.verdict(run)["overall"] == "WARN"
    run.files.append(FileResult("a.csv", "a.csv", "csv", status=Status.FAIL))
    v = rep_mod.verdict(run)
    assert v["overall"] == "FAIL" and v["exit_code"] == 3


def test_inspect_json_checks_are_not_applicable() -> None:
    res = run_cli("inspect", EXAMPLES / "students_data.csv", "--json")
    assert res.exit_code == 0
    checks = json.loads(res.output)["files"][0]["checks"]
    assert checks["written"]["result"] == "N/A" and checks["leak_scan"]["result"] == "N/A"
    assert checks["rows"]["result"] == "N/A"
