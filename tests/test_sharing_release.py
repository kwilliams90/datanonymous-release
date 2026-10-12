"""Direct sharing-mode release tests over an independently invented school corpus."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import ColumnRef, Settings
from datanonymous.errors import UsageError
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy import package
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import (MEASURE_COLUMNS, PERSON_COUNT, RARE_VOCABULARY, corpus_rows, hash_inventory,
                            read_records, student_id, write_corpus)


def _execute(base: Path, variant: str = "canonical", seed: int = 573291):
    original, output, report = base / "original", base / "synthetic", base / "local-report"
    write_corpus(original, variant)
    run = execute(Settings(mode="sharing", seed=seed), original, output, report)
    return original, output, report, run


def _diagnostic(run):
    return [(m.level, m.code, m.text) for m in blocking_findings(run)]


@pytest.fixture(scope="module")
def canonical_release(tmp_path_factory):
    return _execute(tmp_path_factory.mktemp("canonical_release"))


def test_fixture_has_supported_local_donors_and_unique_original_membership():
    tables = corpus_rows()
    assert len(tables["roster.csv"]) == PERSON_COUNT
    for rows in tables.values():
        person_counts = Counter(row["Student_ID"] for row in rows)
        assert min(Counter(person_counts.values()).values()) >= 10
        for column in rows[0]:
            if column.endswith("_ID") or column == "DOB" or column in MEASURE_COLUMNS:
                continue
            holders = {}
            for row in rows:
                holders.setdefault(row[column], set()).add(row["Student_ID"])
            assert min(map(len, holders.values())) >= 10
    iep = {r["Student_ID"] for r in tables["iep.csv"]}
    discipline = {r["Student_ID"] for r in tables["discipline.csv"]}
    assert iep & discipline == {student_id(0)}
    assert {r["DOB"][:7] for r in tables["roster.csv"]} == {"2008-09"}


def test_complete_canonical_package_promotes_without_privacy_warnings(canonical_release):
    original, output, report, run = canonical_release
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 0
    assert not blocking_findings(run)
    assert run.data_mode == "sharing"
    assert output.is_dir() and report.is_dir()
    assert all(f.written for f in run.files)
    assert not list(report.glob(".sharing-staging-*"))
    assert run.package["inventory_sha256"] == hash_inventory(output)
    assert set(hash_inventory(output)) == {*corpus_rows(), "README.md", "DATA_DICTIONARY.md", "schema.json"}
    for name in corpus_rows():
        assert len(read_records(output / name)) == len(read_records(original / name))
    schema = json.loads((output / "schema.json").read_text(encoding="utf-8"))
    assert schema["data_mode"] == "sharing" and "seed" not in schema
    docs = "\n".join((output / name).read_text(encoding="utf-8")
                     for name in ("README.md", "DATA_DICTIONARY.md", "schema.json"))
    assert "573291" not in docs and "--seed" not in docs
    assert "N output equals N input" in docs and "valid joins" in docs


def test_release_has_fictional_aliases_families_memberships_and_valid_composites(canonical_release):
    original, output, _, run = canonical_release
    assert run.package["released"], _diagnostic(run)
    tables = {name: read_records(output / name) for name in corpus_rows()}
    source = {name: read_records(original / name) for name in corpus_rows()}
    roster = tables["roster.csv"]
    roster_ids = {row["Student_ID"] for row in roster}
    assert len(roster_ids) == PERSON_COUNT
    assert len({row["State_Student_ID"] for row in roster}) == PERSON_COUNT
    assert len({row["Family_ID"] for row in roster}) == 60
    relation = {r["Student_ID"]: (r["State_Student_ID"], r["Family_ID"]) for r in roster}
    for name, rows in tables.items():
        assert set(r["Student_ID"] for r in rows) <= roster_ids
        for row in rows:
            assert relation[row["Student_ID"]] == (row["State_Student_ID"], row["Family_ID"])
        for column in ("Student_ID", "State_Student_ID", "Family_ID"):
            assert not {r[column] for r in rows} & {r[column] for r in source["roster.csv"]}
        source_counts = Counter(r["Student_ID"] for r in source[name])
        output_counts = Counter(r["Student_ID"] for r in rows)
        assert Counter(source_counts.values()) == Counter(output_counts.values())
    for name, date in (("discipline.csv", "Incident_Date"), ("events.csv", "Event_Date")):
        rows = tables[name]
        assert len({(r["Student_ID"], r[date]) for r in rows}) == len(rows)
    iep = {r["Student_ID"] for r in tables["iep.csv"]}
    discipline = {r["Student_ID"] for r in tables["discipline.csv"]}
    # A stable source-person mapping preserves the unique source overlap exactly.
    assert len(iep & discipline) > 1
    dates = [r["DOB"] for r in roster]
    bounds = {min(r["DOB"] for r in source["roster.csv"]), max(r["DOB"] for r in source["roster.csv"])}
    assert not set(dates) & bounds
    assert all(date.startswith("2008-09-") for date in dates)


def test_default_cli_actually_uses_sharing_without_compatibility_helper(tmp_path):
    original = tmp_path / "original"
    write_corpus(original)
    output, report = tmp_path / "synthetic", tmp_path / "local-report"
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(output),
                                     "--report-dir", str(report), "--seed", "573291"],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    schema = json.loads((output / "schema.json").read_text(encoding="utf-8"))
    assert schema["data_mode"] == "sharing" and "seed" not in schema
    evidence = json.loads((report / "report.json").read_text(encoding="utf-8"))["sharing_package"]
    assert evidence["released"] and not evidence["blocking_codes"]


def test_source_relabeling_and_row_reversal_do_not_change_fictional_relationships(tmp_path, canonical_release):
    _, first_output, _, first = canonical_release
    assert first.package["released"], _diagnostic(first)
    original, output = tmp_path / "original", tmp_path / "synthetic"
    write_corpus(original)
    for path in original.glob("*.csv"):
        rows = read_records(path)
        for row in rows:
            for column in ("Student_ID", "State_Student_ID", "Family_ID"):
                value = row[column]
                row[column] = value[0] + str(int(value[1:]) + 100003)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(reversed(rows))
    second = execute(Settings(mode="sharing", seed=573291), original, output, tmp_path / "local-report")
    assert second.package["released"], _diagnostic(second)
    for name in corpus_rows():
        def identities(root):
            return [(r["Student_ID"], r["State_Student_ID"], r["Family_ID"])
                    for r in read_records(root / name)]
        assert identities(first_output) == identities(output)


def test_rare_iep_vocabulary_cannot_borrow_common_roster_holders(tmp_path):
    original, output, _, run = _execute(tmp_path, "rare_iep")
    roster_holders = {r["Student_ID"] for r in read_records(original / "roster.csv")
                      if r["Support_Program"] == RARE_VOCABULARY}
    fact_holders = {r["Student_ID"] for r in read_records(original / "iep.csv")
                    if r["Support_Program"] == RARE_VOCABULARY}
    assert len(roster_holders) == 60 and len(fact_holders) == 1
    if run.package["released"]:
        assert RARE_VOCABULARY not in {r["Support_Program"] for r in read_records(output / "iep.csv")}
    else:
        assert not output.exists()
        assert any("rare" in m.text or "distinct people" in m.text for m in blocking_findings(run)), _diagnostic(run)


@pytest.mark.parametrize("variant, diagnostic", [("thirteen_events", "count class"), ("rare_id", "format class")])
def test_rare_person_count_and_identifier_shape_refuse_the_whole_release(tmp_path, variant, diagnostic):
    _, output, report, run = _execute(tmp_path, variant)
    assert not run.package["released"]
    assert not output.exists() and report.is_dir()
    assert all(not f.written for f in run.files)
    assert any(diagnostic in m.text for m in blocking_findings(run)), _diagnostic(run)


@pytest.mark.parametrize("column,actors", [
    ("Parent_Name", ["Clara O'Baxen", "Daria D'Buxen", "Elena D'Bixen"]),
    ("Guardian_Email", ["baaan@first.test", "babcn@other.test", "bacdn@other.test"]),
    ("Teacher_Name", ["Clara O'Baxen", "Daria D'Buxen", "Elena D'Bixen"]),
    ("Staff_Name", ["Clara O'Baxen", "Daria D'Buxen", "Elena D'Bixen"]),
    ("Parent_Education", ["Graduate", "College", "Secondary"]),
    ("Guardian_Income", ["42000.50", "52000.75", "62000.25"]),
    ("Teacher_DOB", ["1978-01-02", "1981-01-15", "1982-01-24"]),
    ("Staff_Salary", ["42000.50", "52000.75", "62000.25"]),
    ("Teacher_Comments", ["Contact baaan@first.test", "Contact babcn@other.test", "Contact bacdn@other.test"]),
    ("Parent_Notes", ["Call 212-555-0101", "Call 312-555-0102", "Call 312-555-0103"]),
])
def test_intrinsic_actor_attributes_with_many_students_withhold_the_whole_release(tmp_path, column, actors):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    rows = [{"Student_ID": student_id(i), column: actors[i // 20],
             "Score": f"{(5100 + i * 73) / 100:.2f}",
             "Other_Score": f"{(4000 + (i * 571) % 3999) / 100:.2f}"} for i in range(60)]
    with (original / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Each source actor reaches twenty students, yet the O' name/domain or
    # attribute value can still have exactly one statistical donor (D49).
    assert set(Counter(row[column] for row in rows).values()) == {20}
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert all(not f.written for f in run.files)
    assert any("intrinsic attribute of another person" in m.text for m in blocking_findings(run)), _diagnostic(run)


def test_explicit_staff_person_id_does_not_claim_an_unimplemented_actor_ownership_contract(tmp_path):
    original, output = tmp_path / "original", tmp_path / "synthetic"
    original.mkdir()
    rows = [{"Staff_ID": f"T{410003 + i * 37:06d}",
             "Staff_Name": f"Ba{chr(97 + i // 26)}{chr(97 + i % 26)}n Wilson",
             "Score": f"{(5100 + i * 73) / 100:.2f}"} for i in range(60)]
    with (original / "staff.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    settings = Settings(mode="sharing", seed=573291, person_id=[ColumnRef("Staff_ID")])
    run = execute(settings, original, output, tmp_path / "local-report")
    assert not run.package["released"] and not output.exists()
    assert any("support for that actor" in m.text for m in blocking_findings(run)), _diagnostic(run)


def test_stale_destination_is_refused_without_touching_existing_evidence(tmp_path):
    original, output = tmp_path / "original", tmp_path / "synthetic"
    write_corpus(original)
    output.mkdir()
    stale = output / "stale.csv"
    stale.write_bytes(b"private-stale-evidence\n")
    before = hash_inventory(output)
    with pytest.raises(UsageError, match="does not yet exist|not empty"):
        execute(Settings(mode="sharing"), original, output, tmp_path / "local-report")
    assert hash_inventory(output) == before


def test_partial_input_failure_withholds_other_generated_tables(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert "sharing_incomplete" in run.package["blocking_codes"]
    assert all(not f.written for f in run.files)
    assert any(any(path.glob("*.csv")) for path in report.glob(".sharing-staging-*"))


def test_foreign_staging_artifact_blocks_promotion(tmp_path, monkeypatch):
    verify = package.inventory

    def contaminate(stage, expected):
        (stage / "stale.csv").write_bytes(b"invented unapproved artifact\n")
        return verify(stage, expected)

    monkeypatch.setattr(package, "inventory", contaminate)
    _, output, _, run = _execute(tmp_path)
    assert not run.package["released"] and not output.exists()
    assert "sharing_inventory" in run.package["blocking_codes"]
    assert run.package["inventory_sha256"] == {}


def test_saturated_categorical_only_package_is_still_refused(tmp_path):
    _, output, _, run = _execute(tmp_path, "categorical_only")
    assert not run.package["released"] and not output.exists()
    assert "row_matches_report_only" in run.package["blocking_codes"]
    assert all(not f.written for f in run.files)
