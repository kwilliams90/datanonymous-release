"""The plain-language message catalogue: every blocking code is explained, and the report shows the explanation."""

from __future__ import annotations

import ast
import csv
import re
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

import datanonymous
from datanonymous import messages
from datanonymous.cli import main
from datanonymous.config import Settings
from datanonymous.model import Message
from datanonymous.pipeline import execute
from datanonymous.privacy.sharing_policy import REFUSED_SUGGESTIONS, finding_category, names_refused_option

from release_corpus import corpus_rows, student_id, write_corpus

SRC = Path(datanonymous.__file__).parent


# --------------------------------------------------------------------------- codes the source can raise
def _strings(node: ast.AST | None) -> list[str]:
    """Every string literal an expression can evaluate to (a conditional gives both branches)."""
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _strings(node.body) + _strings(node.orelse)
    return []


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _reachable_codes() -> tuple[set[str], set[str]]:
    """(blocking finding codes, failure codes) read from the source, so a new code cannot be forgotten here."""
    blocking: set[str] = set()
    failures: set[str] = {"fail"}  # FileResult.fail() without a more specific code
    for path in SRC.rglob("*.py"):
        for call in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Call)):
            name = _call_name(call)
            if name == "Message" and len(call.args) >= 2:
                for level in _strings(call.args[0]):
                    for code in _strings(call.args[1]):
                        if finding_category(Message(level, code, "")) == "blocking":
                            blocking.add(code)
            # a failure code is a ``code=`` keyword, or the third argument of fail() / _refuse()
            failures.update(c for kw in call.keywords if kw.arg == "code" for c in _strings(kw.value))
            method = isinstance(call.func, ast.Attribute)
            if (name == "fail" and method or name == "_refuse") and len(call.args) >= 3:
                failures.update(_strings(call.args[2]))
    return blocking, failures


def test_the_scan_finds_the_codes_it_is_meant_to_guard():
    blocking, failures = _reachable_codes()
    assert {"copied_value", "id_candidates", "sharing_rare_unit_restored", "unlinked_key_column"} <= blocking
    assert {"fail", "sharing_other_person_attribute", "withheld_by_findings"} <= failures
    assert "fidelity_drift" not in blocking  # a utility warning does not block


def test_every_blocking_code_and_failure_code_has_a_plain_sentence_and_a_plain_fix():
    blocking, failures = _reachable_codes()
    missing = sorted(c for c in blocking | failures if messages.lookup(c) is None)
    assert not missing, f"add these codes to datanonymous/messages.py: {missing}"
    for code in blocking | failures:
        entry = messages.lookup(code)
        assert entry.kind == messages.PROBLEM, code
        assert entry.sentence.strip() and (entry.fix or "").strip(), code


def test_a_plain_sentence_has_no_internal_jargon():
    jargon = re.compile(r"\bD\d{2}\b|\bFD\b|PII|\bNaN\b|\bparams?\b|\bartifact|_[a-z]+_|\bprofile\b")
    for code, entry in messages.CATALOGUE.items():
        assert not jargon.search(entry.sentence + " " + (entry.fix or "")), code


def test_no_catalogue_fix_names_an_option_sharing_mode_refuses():
    for code, entry in messages.CATALOGUE.items():
        assert not names_refused_option(entry.fix), (code, entry.fix)
        assert not names_refused_option(entry.sentence), (code, entry.sentence)
    assert names_refused_option("use --force-factor Age") and not names_refused_option("use --exclude Age")


def test_the_catalogue_has_a_change_kind_for_informational_codes(monkeypatch):
    monkeypatch.setattr(messages, "CATALOGUE", dict(messages.CATALOGUE))
    messages.change("dates_shifted", "Dates were shifted; month patterns were kept.")
    entry = messages.lookup("dates_shifted")
    assert entry.kind == messages.CHANGE and entry.fix is None
    assert messages.render("dates_shifted") == ("Dates were shifted; month patterns were kept.", None)
    assert messages.render("never_registered") is None


def test_a_code_the_catalogue_does_not_know_is_not_made_up(monkeypatch):
    assert messages.render("no_such_code", "text") is None


# --------------------------------------------------------------------------- what the operator reads
def _blocked_roster_run(tmp_path, column="Parent_Education", value_sets=("Graduate", "College", "Secondary")):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    rows = [{"Student_ID": student_id(i), column: value_sets[i // 20],
             "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(60)]
    with (original / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assert set(Counter(row[column] for row in rows).values()) == {20}
    return original, output, report


def test_a_refused_column_is_explained_in_plain_words_with_its_code_kept(tmp_path):
    original, output, report = _blocked_roster_run(tmp_path)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not run.package["released"]
    text = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "This column describes a parent or teacher, not the student, so it can't be shared safely" in text
    assert "(`sharing_other_person_attribute`)" in text
    assert "Leave it out with `--exclude`" in text
    assert "intrinsic attribute" not in text and "distinct-person" not in text
    detail = (report / "files" / "roster.csv.md").read_text(encoding="utf-8")
    assert "This column describes a parent or teacher" in detail and "sharing_other_person_attribute" in detail
    assert "[Parent_Education]" in detail or "Parent_Education" in text


def test_failure_log_and_console_use_the_plain_sentence(tmp_path):
    original, output, report = _blocked_roster_run(tmp_path)
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(output), "--report-dir", str(report),
                                       "--seed", "573291"], catch_exceptions=False)
    assert result.exit_code == 3, result.output
    assert "can't be shared safely" in result.output
    assert "intrinsic attribute" not in result.output
    with (report / "sharing-failures.csv").open(encoding="utf-8", newline="") as stream:
        reasons = [row["reason"] for row in csv.DictReader(stream)]
    assert reasons and all("can't be shared safely" in r and "sharing_other_person_attribute" in r for r in reasons)


def test_sharing_reports_never_suggest_an_option_sharing_refuses(tmp_path):
    original, output, report = _blocked_roster_run(tmp_path)
    execute(Settings(mode="sharing", seed=573291), original, output, report)
    shown = [p for p in report.rglob("*.md")] + [report / "sharing-failures.csv"]
    assert shown
    for path in shown:
        text = path.read_text(encoding="utf-8")
        if path.name == "REPORT.md":  # the criteria table above the findings describes checks; it suggests nothing
            text = text[text.index("## Failures"):]
        assert not names_refused_option(text), (path.name, [f for f in REFUSED_SUGGESTIONS if f in text])


def test_a_blocking_warning_is_shown_as_a_plain_sentence_and_fix_with_its_code(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    lines = (original / "iep.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    (original / "iep.csv").write_text("".join(lines[:5]) + "\n" + "".join(lines[5:]), encoding="utf-8")
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not (output / "iep.csv").exists() and (output / "roster.csv").exists()
    assert {f.rel_path for f in run.files if f.failed} == {"iep.csv"}
    text = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "Blank lines between records in the source were dropped (`blank_lines_dropped`)." in text
    assert "Remove the blank lines from the source and run again if they matter." in text
    assert "This file was left out because a privacy check found a problem in it (`withheld_by_findings`)." in text
    assert "blank line(s) between records were dropped" not in text  # the internal wording is gone
    assert set(corpus_rows()) >= {"iep.csv"}


def test_fidelity_mode_keeps_its_own_wording(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    lines = (original / "iep.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    (original / "iep.csv").write_text("".join(lines[:5]) + "\n" + "".join(lines[5:]), encoding="utf-8")
    execute(Settings(mode="fidelity", seed=573291), original, output, report)
    text = (report / "REPORT.md").read_text(encoding="utf-8")
    assert "blank line(s) between records were dropped" in text
    assert "blank_lines_dropped" not in text.replace("blank line(s)", "")
