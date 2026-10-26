"""Direct sharing-mode release tests over an independently invented school corpus."""

from __future__ import annotations

import csv
import errno
import json
import os
import shutil
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


def _cli(*args):
    return CliRunner().invoke(main, [str(a) for a in args], catch_exceptions=False)


def test_cli_writes_the_other_tables_exits_3_and_points_to_the_failure_log(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")
    result = _cli(original, "--out-dir", output, "--report-dir", report, "--seed", "573291")
    assert result.exit_code == 3, result.output
    assert set(hash_inventory(output)) == {*corpus_rows(), *DOCS}
    assert "1 input file(s) were not written to the output folder" in result.output
    assert "sharing-failures.csv" in result.output


def test_cli_all_or_nothing_writes_nothing_when_one_table_fails(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")
    result = _cli(original, "--out-dir", output, "--report-dir", report, "--seed", "573291", "--all-or-nothing")
    assert result.exit_code == 3, result.output
    assert not output.exists()
    assert "nothing was written to the output folder" in result.output


def test_cli_help_says_the_out_dir_must_be_new_or_empty():
    text = " ".join(_cli("run", "--help").output.split())
    assert "--out-dir" in text and "new or empty folder" in text
    assert "--all-or-nothing" in text


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
    assert run.package["released"], _diagnostic(run)
    if (output / "iep.csv").exists():
        assert RARE_VOCABULARY not in {r["Support_Program"] for r in read_records(output / "iep.csv")}
    else:  # the fact table is left out on its own; the roster's common holders do not rescue it
        iep = next(f for f in run.files if f.rel_path == "iep.csv")
        assert any("rare" in r or "distinct people" in r for r in iep.reasons), iep.reasons
        assert (output / "roster.csv").exists()


@pytest.mark.parametrize("variant, diagnostic, refused", [
    ("thirteen_events", "count class", {"events.csv"}),
    # the e-mail ID's holder is in the roster and in discipline; IEP and events refer to the roster
    ("email_id", "e-mail", {"roster.csv", "discipline.csv"}),
])
def test_rare_person_count_and_identifier_shape_leave_out_only_the_offending_tables(tmp_path, variant, diagnostic,
                                                                                    refused):
    original, output, report, run = _execute(tmp_path, variant)
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 3
    assert {f.rel_path for f in run.files if not f.written} == refused
    assert set(hash_inventory(output)) == {*corpus_rows(variant), *DOCS} - refused
    failures = {r["input_path"]: r for r in _failures(report)}
    assert set(failures) == refused
    assert all(diagnostic in r["reason"] for r in failures.values()), failures
    # the remaining tables keep their rows, invent their keys, and join among themselves only
    source_ids = {r["Student_ID"] for r in read_records(original / "roster.csv")}
    written = {name: read_records(output / name) for name in corpus_rows(variant) if name not in refused}
    for name, rows in written.items():
        assert len(rows) == len(read_records(original / name))
        assert not {r["Student_ID"] for r in rows} & source_ids
    if "roster.csv" in written:
        relation = {r["Student_ID"]: (r["State_Student_ID"], r["Family_ID"]) for r in written["roster.csv"]}
        for rows in written.values():
            assert all(relation[r["Student_ID"]] == (r["State_Student_ID"], r["Family_ID"]) for r in rows)


def test_a_rare_id_layout_is_written_in_the_held_layout_in_every_table(tmp_path):
    # ADR-0004: the rare ID's holder (in the roster and in discipline) gets a fictional ID in the held layout
    original, output, _, run = _execute(tmp_path, "rare_id")
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 0
    assert set(hash_inventory(output)) == {*corpus_rows("rare_id"), *DOCS}
    source_ids = {r["Student_ID"] for r in read_records(original / "roster.csv")}
    written = {name: read_records(output / name) for name in corpus_rows("rare_id")}
    ids = [r["Student_ID"] for rows in written.values() for r in rows]
    assert {(len(i), i[0], i[1:].isdigit()) for i in ids} == {(7, "S", True)}
    assert not set(ids) & source_ids
    roster = {r["Student_ID"] for r in written["roster.csv"]}
    assert len(roster) == PERSON_COUNT
    assert all({r["Student_ID"] for r in rows} <= roster for rows in written.values())


def test_tables_whose_join_target_is_left_out_are_planned_without_that_join(tmp_path):
    # email_id: the roster (the e-mail ID's holder) is left out. IEP and events join to it, so each keeps its rows
    # and is planned over the IDs the remaining tables hold.
    _, output, _, run = _execute(tmp_path, "email_id")
    assert run.package["released"], _diagnostic(run)
    for name in ("iep.csv", "events.csv"):
        fr = next(f for f in run.files if f.rel_path == name)
        assert "roster.csv" in {link.get("parent", "").split("::")[0]
                                for t in fr.tables for link in t.structure.cross_file_keys}
        assert fr.written
    iep, events = (read_records(output / name) for name in ("iep.csv", "events.csv"))
    ids = [r["Student_ID"] for r in iep + events]
    assert len({len(i) for i in ids}) == 1 and not any("-" in i for i in ids)
    # the joins between the written tables still hold: IEP and events hold the same 120 fictional people
    assert {r["Student_ID"] for r in iep} == {r["Student_ID"] for r in events}
    assert len({r["Student_ID"] for r in iep}) == 120


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_row_numbers_left_without_their_roster_are_not_kept_as_row_numbers(tmp_path):
    # The roster's Student_ID is the row numbers 1..100 (kept as they are, D9); an e-mail State_Student_ID
    # leaves it out. The child's 31..50 are not row numbers on their own: they get fictional IDs that are no real ID
    # of the run, and the left-out roster's 1..100 range does not show through.
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    state = [f"T{720007 + i * 41:06d}" for i in range(100)]
    state[0] = "pat.quill@example.org"
    _write_csv(original / "roster.csv", [
        {"Student_ID": str(i + 1), "State_Student_ID": state[i], "Grade": str(6 + i % 4),
         "Score": f"{(5500 + (i * 71) % 3500) / 100:.2f}"} for i in range(100)])
    child_ids = [str(i + 1) for i in range(30, 50)]
    _write_csv(original / "services.csv", [
        {"Student_ID": child_ids[j], "Frequency": ("Weekly", "Monthly")[j % 2],
         "Minutes": f"{(1500 + (j * 137) % 9000) / 100:.2f}"} for j in range(20)])
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert {f.rel_path for f in run.files if not f.written} == {"roster.csv"}
    assert any("e-mail" in r for r in next(f for f in run.files if f.rel_path == "roster.csv").reasons)
    ids = [r["Student_ID"] for r in read_records(output / "services.csv")]
    assert len(set(ids)) == 20
    assert not set(ids) & set(child_ids)
    assert not set(ids) & {str(i) for i in range(1, 101)}
    # the leak scan does not skip the column as kept row numbers, whichever file it reads it from
    for fr in run.files:
        for tp in fr.tables:
            column = next(p for p in tp.columns if p.name == "Student_ID")
            assert not column.params.get("_identity_mapped"), fr.rel_path
            assert not str(column.params.get("id_mapping", "")).startswith("kept"), fr.rel_path


def _roster_and_services(original: Path) -> None:
    # invented: the roster's Student_ID is 1..100 (row numbers) and its Parent_Name is another person's attribute,
    # which fails the roster after population planning; the services table holds 31..50
    original.mkdir()
    _write_csv(original / "roster.csv", [{"Student_ID": str(i), "Parent_Name": "Clara Qunon",
                                          "Score": f"{i * 137 / 100:.2f}"} for i in range(1, 101)])
    _write_csv(original / "services.csv", [{"Student_ID": str(i), "Hours": f"{i * 137 / 100:.2f}",
                                            "Cost": f"{i * 243 / 100:.2f}"} for i in range(31, 51)])


def test_a_roster_failing_after_planning_does_not_keep_its_row_numbers(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _roster_and_services(original)
    # --refuse-unsupported: Parent_Name fails the roster instead of being nulled (ADR-0001)
    run = execute(Settings(mode="sharing", seed=573291, refuse_unsupported=True), original, output, report)
    assert run.package["released"], _diagnostic(run)
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written and any("Parent_Name" in r for r in roster.reasons)
    ids = [r["Student_ID"] for r in read_records(output / "services.csv")]
    assert len(set(ids)) == 20
    assert not set(ids) & {str(i) for i in range(1, 101)}
    for fr in run.files:
        for tp in fr.tables:
            column = next(p for p in tp.columns if p.name == "Student_ID")
            assert not column.params.get("_identity_mapped"), fr.rel_path
    assert not any(p.name.startswith(".sharing-staging-") for p in report.iterdir())


def test_include_selection_satisfies_all_or_nothing(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _roster_and_services(original)
    run = execute(Settings(mode="sharing", seed=573291, include=["services.csv"], all_or_nothing=True),
                  original, output, report)
    assert run.package["released"], (_diagnostic(run), run.package)
    assert (output / "services.csv").is_file() and not (output / "roster.csv").exists()


def test_all_or_nothing_still_refuses_an_unsupported_input(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _roster_and_services(original)
    (original / "notes.docx").write_bytes(b"not a table")
    run = execute(Settings(mode="sharing", seed=573291, include=["services.csv", "notes.docx"],
                           all_or_nothing=True), original, output, report)
    assert "sharing_skipped_inputs" in run.package["blocking_codes"]


def test_too_many_labels_is_written_with_no_option_sharing_refuses(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    _write_csv(original / "data.csv", [{"student_id": f"S{i:04d}", "category": f"label_{i}"} for i in range(30)])
    run = execute(Settings(mode="sharing", seed=1), original, output, report)
    assert run.package["released"], _diagnostic(run)
    labels = {r["category"] for r in read_records(output / "data.csv")}
    assert not labels & {f"label_{i}" for i in range(30)}
    fixes = " ".join(m.fix or "" for f in run.files for m in f.all_messages())
    assert "--force-factor" not in fixes and "--org-code" not in fixes


def test_currency_with_cents_is_written_in_its_format(tmp_path):
    # invented amounts; the extremes were read back through the "$" format and crashed (TypeError) before
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    amounts = [52000 + (i * 7919) % 30000 + (i % 100) / 100 for i in range(300)]
    _write_csv(original / "pay.csv", [{"Student_ID": f"S{i:05d}", "Grade": str(6 + i % 4), "Amount": f"${a:,.2f}"}
                                      for i, a in enumerate(amounts)])
    run = execute(Settings(mode="sharing", seed=1), original, output, report)
    assert run.package["released"], (_diagnostic(run), [f.reasons for f in run.files])
    written = [r["Amount"] for r in read_records(output / "pay.csv")]
    assert all(v.startswith("$") and "," in v and v[-3] == "." for v in written)


def test_all_or_nothing_withholds_every_table_when_the_population_refuses_one(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original, "thirteen_events")
    run = execute(Settings(mode="sharing", seed=573291, all_or_nothing=True), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert all(not f.written for f in run.files)
    assert exit_code(run) == 3
    assert any("count class" in m.text for m in blocking_findings(run)), _diagnostic(run)


def test_a_population_refusal_that_names_no_table_withholds_every_table(tmp_path, monkeypatch):
    from datanonymous.errors import UnsupportedColumnError
    from datanonymous.generate import population

    def refuse(*args, **kwargs):
        raise UnsupportedColumnError("sharing population: invented run-level refusal")

    monkeypatch.setattr(population, "_mint", refuse)
    _, output, report, run = _execute(tmp_path)
    assert not run.package["released"] and not output.exists()
    assert all(not f.written for f in run.files)
    assert sorted(r["input_path"] for r in _failures(report)) == sorted(corpus_rows())
    assert all("invented run-level refusal" in r["reason"] for r in _failures(report))


INTRINSIC_ACTOR_ATTRIBUTES = pytest.mark.parametrize("column,actors", [
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


def _actor_roster(original: Path, column: str, actors: list[str]) -> list[dict[str, str]]:
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
    return rows


@INTRINSIC_ACTOR_ATTRIBUTES
def test_intrinsic_actor_attributes_with_many_students_withhold_the_whole_release(tmp_path, column, actors):
    # --refuse-unsupported: the old whole-file refusal (ADR-0001)
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _actor_roster(original, column, actors)
    run = execute(Settings(mode="sharing", seed=573291, refuse_unsupported=True), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert all(not f.written for f in run.files)
    assert any("intrinsic attribute of another person" in m.text for m in blocking_findings(run)), _diagnostic(run)


@INTRINSIC_ACTOR_ATTRIBUTES
def test_intrinsic_actor_attributes_are_nulled_and_appear_nowhere_in_the_output(tmp_path, column, actors):
    # ADR-0001: by default the column is nulled and the file written
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    _actor_roster(original, column, actors)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"] and exit_code(run) == 0, _diagnostic(run)
    written = read_records(output / "roster.csv")
    assert len(written) == 60 and all(row[column] == "" for row in written)
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
    assert not [actor for actor in actors if actor in text]


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
    settings = Settings(mode="sharing", seed=573291, person_id=[ColumnRef("Staff_ID")], refuse_unsupported=True)
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
    with pytest.raises(UsageError, match="already contains files; .*new or empty folder"):
        execute(Settings(mode="sharing"), original, output, tmp_path / "local-report")
    assert hash_inventory(output) == before


def test_existing_empty_out_dir_receives_the_package(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    output.mkdir()
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 0
    assert set(hash_inventory(output)) == {*corpus_rows(), "README.md", "DATA_DICTIONARY.md", "schema.json"}
    assert not list(report.glob(".sharing-staging-*"))


def test_an_existing_empty_out_dir_is_kept_not_replaced(tmp_path):
    """A mount point, or a folder whose parent is read-only, cannot be removed; its permissions must stay."""
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    output.mkdir()
    before = os.stat(output)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert os.path.samestat(before, os.stat(output))
    assert set(hash_inventory(output)) == {*corpus_rows(), *DOCS}


def _cross_device_into(monkeypatch, output):
    """os.rename into ``output`` fails as it does across drives or mounts."""
    real = os.rename

    def rename(src, dst):
        if Path(dst).parent == output:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(src, dst)

    monkeypatch.setattr(os, "rename", rename)


def test_an_existing_out_dir_on_another_drive_receives_copies(tmp_path, monkeypatch):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    output.mkdir()
    _cross_device_into(monkeypatch, output)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert run.package["inventory_sha256"] == hash_inventory(output)
    assert not list(report.glob(".sharing-staging-*"))


def test_a_copy_that_fails_partway_leaves_the_out_dir_empty(tmp_path, monkeypatch):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    output.mkdir()
    _cross_device_into(monkeypatch, output)
    real, calls = shutil.copy2, []

    def copy2(src, dst, **kw):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(src, dst, **kw)

    monkeypatch.setattr(shutil, "copy2", copy2)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not run.package["released"]
    assert "sharing_promotion" in run.package["blocking_codes"]
    assert output.is_dir() and not any(output.iterdir())
    assert exit_code(run) == 3


def _failures(report: Path) -> list[dict[str, str]]:
    with (report / "sharing-failures.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


DOCS = {"README.md", "DATA_DICTIONARY.md", "schema.json"}


def test_a_broken_table_is_left_out_and_the_others_are_written(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 3
    assert set(hash_inventory(output)) == {*corpus_rows(), *DOCS}
    assert run.package["inventory_sha256"] == hash_inventory(output)
    assert not list(report.glob(".sharing-staging-*"))
    rows = _failures(report)
    assert [(r["input_path"], r["outcome"]) for r in rows] == [("broken.xlsx", "failed")]
    assert rows[0]["reason"]
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert f"1 of {len(corpus_rows()) + 1} input files were not written" in readme
    assert "joins hold only among the files in this folder" in readme
    assert "broken" not in readme


def test_skipped_inputs_are_logged_and_do_not_block(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "notes.docx").write_bytes(b"invented notes")
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert run.package["released"], _diagnostic(run)
    assert exit_code(run) == 0
    assert set(hash_inventory(output)) == {*corpus_rows(), *DOCS}
    assert [(r["input_path"], r["outcome"]) for r in _failures(report)] == [("notes.docx", "skipped")]
    assert f"1 of {len(corpus_rows()) + 1} input files were not written" in (output / "README.md").read_text(
        encoding="utf-8")


def test_a_complete_run_logs_no_failures_and_says_nothing_was_left_out(canonical_release):
    _, output, report, _ = canonical_release
    assert _failures(report) == []
    assert "were not written" not in (output / "README.md").read_text(encoding="utf-8")


def test_when_every_table_fails_nothing_is_written_and_an_empty_out_dir_stays_empty(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original, "thirteen_events")
    for name in set(corpus_rows()) - {"events.csv"}:
        (original / name).unlink()
    output.mkdir()
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not run.package["released"]
    assert exit_code(run) == 3
    assert output.is_dir() and not any(output.iterdir())
    assert [r["input_path"] for r in _failures(report)] == ["events.csv"]


def test_all_or_nothing_withholds_every_table_when_one_fails(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")
    run = execute(Settings(mode="sharing", seed=573291, all_or_nothing=True), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert "sharing_incomplete" in run.package["blocking_codes"]
    assert all(not f.written for f in run.files)
    assert exit_code(run) == 3
    assert "broken.xlsx" in {r["input_path"] for r in _failures(report)}


def test_all_or_nothing_withholds_every_table_when_an_input_is_skipped(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    write_corpus(original)
    (original / "notes.docx").write_bytes(b"invented notes")
    run = execute(Settings(mode="sharing", seed=573291, all_or_nothing=True), original, output, report)
    assert not run.package["released"] and not output.exists()
    assert "sharing_skipped_inputs" in run.package["blocking_codes"]


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


def test_saturated_categorical_only_tables_are_still_refused(tmp_path):
    _, output, report, run = _execute(tmp_path, "categorical_only")
    refused = {f.rel_path for f in run.files
               if any(m.code == "row_matches_report_only" for m in f.all_messages())}
    assert refused
    assert all(not f.written for f in run.files if f.rel_path in refused)
    assert run.package["released"], _diagnostic(run)
    assert not refused & set(hash_inventory(output))
    assert refused <= {r["input_path"] for r in _failures(report)}
    assert exit_code(run) == 3
