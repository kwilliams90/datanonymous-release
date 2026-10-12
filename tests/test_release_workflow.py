"""U06/U07 consumer evidence; inputs are wholly invented in release_corpus."""

import csv
import datetime as dt
from collections import Counter

import pytest

from datanonymous.config import Settings
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings
from release_corpus import PERSON_COUNT, RARE_VOCABULARY, read_records, write_corpus
from release_workflow import ContractError, SchoolCalendar, analyze, read_tables


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def release(base, variant="canonical"):
    source, copy = base / "original", base / "synthetic"
    write_corpus(source, variant)
    if variant == "rare_iep":
        # This independently specified frequency schedule lets all common
        # programs have both frequencies. The original rare_iep variant has a
        # program/frequency dependency, which sharing correctly refuses rather
        # than restoring a suppressed label through that dependency.
        rows = read_records(source / "iep.csv")
        for i, row in enumerate(rows):
            row["Service_Frequency"] = ("Weekly", "Monthly")[(i // 2) % 2]
        write_rows(source / "iep.csv", rows)
    run = execute(Settings(mode="sharing", seed=573291), source, copy, base / "local-report")
    assert run.package["released"], [(m.code, m.text) for m in blocking_findings(run)]
    assert exit_code(run) == 0
    return source, copy


@pytest.fixture(scope="module")
def workflow_release(tmp_path_factory):
    return release(tmp_path_factory.mktemp("workflow_release"))


def test_same_reader_joins_summaries_and_chart_contract_on_both_sides(workflow_release):
    source, copy = workflow_release
    expected_rows = {"roster.csv": 240, "iep.csv": 120, "discipline.csv": 300, "events.csv": 360}
    for directory in (source, copy):
        result = analyze(directory)
        assert result["row_counts"] == expected_rows
        assert result["people"] == PERSON_COUNT and result["families"] == 60
        aliases = analyze(directory, join_key="State_Student_ID")
        assert {k: v for k, v in aliases.items() if k != "join_key"} == {
            k: v for k, v in result.items() if k != "join_key"}
        assert sum(group["rows"] for group in result["program_summary"]) == 120
        assert sum(group["measured_rows"] for group in result["program_summary"]) == 120
        assert sum(point["value"] for point in result["chart_data"]) == 120
        assert len({point["label"] for point in result["chart_data"]}) == len(result["chart_data"])
        assert all(isinstance(point["value"], int) and point["value"] >= 0 for point in result["chart_data"])
        assert {r["academic_year"] for r in result["term_summary"]} == {"2024-2025"}
        assert {r["term"] for r in result["term_summary"]} == {"Fall"}
        assert {r["table"]: sum(x["rows"] for x in result["term_summary"] if x["table"] == r["table"])
                for r in result["term_summary"]} == {"discipline.csv": 300, "events.csv": 360}
        # Independently recompute the chart inputs from the reader's raw records,
        # rather than comparing two potentially identically broken workflows.
        raw = read_records(directory / "iep.csv")
        labels = {"Speech": "Communication", "Learning": "Academic", "Mobility": "Access"}
        expected = Counter(labels.get(row["Support_Program"], "Other") for row in raw)
        assert {point["label"]: point["value"] for point in result["chart_data"]} == expected
    # Schema compatibility is distinct from aggregate equality or plot appearance.
    for name in expected_rows:
        assert list(read_tables(source)[name][0]) == list(read_tables(copy)[name][0])


def test_invented_source_has_known_summary_answers(workflow_release):
    result = analyze(workflow_release[0])
    groups = {row["group"]: row for row in result["program_summary"]}
    assert set(groups) == {"Academic", "Access", "Communication"}
    for group, remainder in (("Communication", 0), ("Academic", 1), ("Access", 2)):
        total = sum(1500 + (i * 137) % 9000 for i in range(remainder, 120, 3)) / 100
        assert groups[group]["rows"] == groups[group]["people"] == 40
        assert groups[group]["total_minutes"] == pytest.approx(total)
        assert groups[group]["mean_minutes"] == pytest.approx(total / 40)


def test_category_suppressed_from_copy_is_still_accepted_in_production(tmp_path):
    source, copy = release(tmp_path, "rare_iep")
    assert RARE_VOCABULARY in {r["Support_Program"] for r in read_records(source / "iep.csv")}
    assert RARE_VOCABULARY not in {r["Support_Program"] for r in read_records(copy / "iep.csv")}
    source_result, copy_result = analyze(source), analyze(copy)
    assert next(row["rows"] for row in source_result["program_summary"] if row["group"] == "Communication") == 1
    assert "Communication" not in {row["group"] for row in copy_result["program_summary"]}
    assert sum(row["rows"] for row in source_result["program_summary"]) == 120
    assert sum(row["rows"] for row in copy_result["program_summary"]) == 120


def test_unknown_valid_category_and_missing_measure_do_not_drop_records(tmp_path):
    source = tmp_path / "original"
    write_corpus(source)
    rows = read_records(source / "iep.csv")
    rows[0]["Support_Program"] = "Dual Language"  # independently invented, absent from the main copy
    rows[0]["Service_Minutes"] = ""
    rows[1]["Support_Program"] = "Dual Language"
    rows[1]["Service_Minutes"] = "25.5"
    write_rows(source / "iep.csv", rows)
    result = analyze(source)
    other = next(row for row in result["program_summary"] if row["group"] == "Other")
    assert other == {"group": "Other", "rows": 2, "people": 2, "measured_rows": 1,
                     "total_minutes": 25.5, "mean_minutes": 25.5}
    assert sum(row["rows"] for row in result["program_summary"]) == 120
    assert sum(row["measured_rows"] for row in result["program_summary"]) == 119
    assert next(point["value"] for point in result["chart_data"] if point["label"] == "Other") == 2


def test_summer_fall_spring_and_school_year_joins_survive_sharing(tmp_path):
    source, copy = tmp_path / "original", tmp_path / "synthetic"
    write_corpus(source)
    rows = read_records(source / "events.csv")
    ids = list(dict.fromkeys(row["Student_ID"] for row in rows))
    months = {key: (3, 7, 10)[i % 3] for i, key in enumerate(ids)}
    for row in rows:
        row["Event_Date"] = f"2024-{months[row['Student_ID']]:02d}-{row['Event_Date'][-2:]}"
    write_rows(source / "events.csv", rows)
    run = execute(Settings(mode="sharing", seed=573291), source, copy, tmp_path / "local-report")
    assert run.package["released"], [(m.code, m.text) for m in blocking_findings(run)]
    for directory in (source, copy):
        summary = [row for row in analyze(directory)["term_summary"] if row["table"] == "events.csv"]
        assert {(row["academic_year"], row["term"]) for row in summary} == {
            ("2023-2024", "Spring"), ("2023-2024", "Summer"), ("2024-2025", "Fall")}
        assert {term: sum(row["rows"] for row in summary if row["term"] == term)
                for term in ("Spring", "Summer", "Fall")} == {"Spring": 120, "Summer": 120, "Fall": 120}


def test_calendar_boundaries_use_institution_policy_and_local_timezone():
    default = SchoolCalendar()
    assert default.period("2024-08-31") == ("2023-2024", "Summer")
    assert default.period("2024-09-01") == ("2024-2025", "Fall")
    assert default.period("2025-01-01") == ("2024-2025", "Spring")
    # Boundary instant is still August 31 in this independently chosen zone.
    local = SchoolCalendar(timezone=dt.timezone(dt.timedelta(hours=-5)))
    assert local.period("2024-09-01T02:00:00Z") == ("2023-2024", "Summer")
    august_start = SchoolCalendar(8, (("Spring", (1, 2, 3, 4, 5)), ("Summer", (6, 7)),
                                      ("Fall", (8, 9, 10, 11, 12))))
    assert august_start.period("2024-08-01") == ("2024-2025", "Fall")
    with pytest.raises(ContractError):
        default.date("2024-09-01T02:00:00")
    with pytest.raises(ContractError):
        SchoolCalendar(terms=(("Spring", (1, 2)),))


@pytest.mark.parametrize("damage", ["orphan", "alias", "duplicate_roster", "duplicate_event", "date", "measure", "schema"])
def test_consumer_refuses_broken_production_contracts(tmp_path, damage):
    source = tmp_path / "original"
    write_corpus(source)
    name = "roster.csv" if damage == "duplicate_roster" else "events.csv"
    rows = read_records(source / name)
    if damage == "orphan":
        rows[0]["Student_ID"] = "UNKNOWN"
    elif damage == "alias":
        rows[0]["State_Student_ID"] = rows[-1]["State_Student_ID"]
    elif damage in {"duplicate_roster", "duplicate_event"}:
        rows.append(dict(rows[0]))
    elif damage == "date":
        rows[0]["Event_Date"] = "2024-02-30"
    elif damage == "measure":
        rows[0]["Participation_Score"] = "NaN"
    else:
        for row in rows:
            row.pop("Participation_Score")
    write_rows(source / name, rows)
    with pytest.raises(ContractError):
        analyze(source)
