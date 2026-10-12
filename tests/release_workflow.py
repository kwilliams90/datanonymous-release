"""Runnable production-style CSV consumer for the independently invented corpus.

This codebook is an invented public specification, not a source vocabulary
export. Its fallback deliberately accepts additional nonempty categories.
Run: python tests/release_workflow.py PATH_TO_CORPUS_OR_SHARING_COPY
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROGRAM_GROUPS = {"Speech": "Communication", "Learning": "Academic",
                  "Mobility": "Access", "General": "General"}
IDENTITY = ("Student_ID", "State_Student_ID", "Family_ID")
SPECS = {
    "roster.csv": (*IDENTITY, "DOB", "Grade", "School", "Support_Program", "Assessment_Score"),
    "iep.csv": (*IDENTITY, "Support_Program", "Service_Frequency", "Service_Minutes"),
    "discipline.csv": (*IDENTITY, "Incident_Date", "Incident_Type", "Disposition", "Instructional_Minutes"),
    "events.csv": (*IDENTITY, "Event_Date", "Event_Type", "Participation_Score"),
}


class ContractError(ValueError):
    """Input cannot meet the consumer's schema, identity, or date contract."""


@dataclass(frozen=True)
class SchoolCalendar:
    """An independently specified school calendar; never inferred from a copy.

    Default terms are Spring Jan-May, Summer Jun-Aug, Fall Sep-Dec. A district
    can supply its own month partition and academic-year start month. Aware
    timestamps are converted to the institution timezone before grouping.
    """

    year_start_month: int = 9
    terms: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("Spring", (1, 2, 3, 4, 5)), ("Summer", (6, 7, 8)), ("Fall", (9, 10, 11, 12)))
    timezone: dt.tzinfo = dt.timezone.utc

    def __post_init__(self):
        months = [month for _, term_months in self.terms for month in term_months]
        if not 1 <= self.year_start_month <= 12 or sorted(months) != list(range(1, 13)):
            raise ContractError("calendar must partition months 1..12 and specify a valid year start")
        if len({name for name, _ in self.terms}) != len(self.terms):
            raise ContractError("calendar term names must be unique")

    def date(self, value: str) -> dt.date:
        try:
            if len(value) == 10:
                return dt.date.fromisoformat(value)
            instant = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                raise ContractError("timestamps must include a timezone")
            return instant.astimezone(self.timezone).date()
        except (ValueError, OverflowError) as exc:
            raise ContractError("invalid or timezone-free ISO date/time") from exc

    def period(self, value: str) -> tuple[str, str]:
        date = self.date(value)
        start = date.year - (date.month < self.year_start_month)
        term = next(name for name, months in self.terms if date.month in months)
        return f"{start}-{start + 1}", term


def program_group(value: str) -> str:
    """Unknown valid categories are included, rather than filtered or rejected."""
    if not value.strip():
        return "Missing"
    return PROGRAM_GROUPS.get(value, "Other")


def number(value: str) -> Decimal | None:
    if not value.strip():
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ContractError("invalid numerical measure") from exc
    if not result.is_finite():
        raise ContractError("measure must be finite or missing")
    return result


def read_tables(directory: Path) -> dict[str, list[dict[str, str]]]:
    tables = {}
    for name, required in SPECS.items():
        with (directory / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or not set(required) <= set(header):
                raise ContractError(f"{name}: missing or duplicate required columns")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ContractError(f"{name}: malformed row")
            tables[name] = rows
    return tables


def analyze(directory: Path, calendar: SchoolCalendar | None = None,
            join_key: str = "Student_ID") -> dict:
    """Read either side with the same code; no source-to-copy row comparison.

    Counts and measure totals use every fact row once. Means exclude missing
    measures and expose the denominator. Chart data contains logical aggregates
    suitable for ordinary bars; resemblance to a source plot is not validation.
    """
    calendar = calendar or SchoolCalendar()
    if join_key not in IDENTITY[:2]:
        raise ContractError("join key must be a student key or its state alias")
    tables = read_tables(directory)
    roster = tables["roster.csv"]
    for key in IDENTITY[:2]:
        values = [row[key] for row in roster]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ContractError("roster student keys and aliases must be nonempty and unique")
    if any(not row["Family_ID"] for row in roster):
        raise ContractError("family key must be nonempty")
    people = {row[join_key]: row for row in roster}
    for row in roster:
        calendar.date(row["DOB"])
        try:
            if not 0 <= int(row["Grade"]) <= 12:
                raise ValueError
        except ValueError as exc:
            raise ContractError("grade must be an integer from 0 through 12") from exc
        number(row["Assessment_Score"])

    joined = {}
    for name, rows in tables.items():
        joined[name] = []
        for row in rows:
            person = people.get(row[join_key])
            if person is None or any(row[key] != person[key] for key in IDENTITY):
                raise ContractError(f"{name}: orphan or inconsistent student alias/family")
            # Explicit many-to-one join avoids silently multiplying fact rows.
            joined[name].append((row, person))

    programs = defaultdict(lambda: {"rows": 0, "people": set(), "measured_rows": 0,
                                    "total_minutes": Decimal(0)})
    for row, _ in joined["iep.csv"]:
        summary = programs[program_group(row["Support_Program"])]
        summary["rows"] += 1
        summary["people"].add(row["Student_ID"])
        minutes = number(row["Service_Minutes"])
        if minutes is not None:
            summary["measured_rows"] += 1
            summary["total_minutes"] += minutes
    program_summary = []
    for group, summary in sorted(programs.items()):
        count = summary["measured_rows"]
        total = summary["total_minutes"]
        program_summary.append({"group": group, "rows": summary["rows"],
                                "people": len(summary["people"]), "measured_rows": count,
                                "total_minutes": float(total),
                                "mean_minutes": float(total / count) if count else None})

    periods = Counter()
    for name, date_column, measure in (
        ("discipline.csv", "Incident_Date", "Instructional_Minutes"),
        ("events.csv", "Event_Date", "Participation_Score"),
    ):
        composites = set()
        for row, person in joined[name]:
            day = calendar.date(row[date_column])
            composite = (row["Student_ID"], day)
            if composite in composites:
                raise ContractError(f"{name}: duplicate student/date composite")
            composites.add(composite)
            number(row[measure])
            year, term = calendar.period(row[date_column])
            periods[(name, person["School"], year, term)] += 1
    term_summary = [{"table": table, "school": school, "academic_year": year,
                     "term": term, "rows": count}
                    for (table, school, year, term), count in sorted(periods.items())]
    return {"row_counts": {name: len(rows) for name, rows in tables.items()},
            "join_key": join_key, "people": len(roster),
            "families": len({row["Family_ID"] for row in roster}),
            "program_summary": program_summary, "term_summary": term_summary,
            "chart_data": [{"label": row["group"], "value": row["rows"]}
                           for row in program_summary]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--year-start-month", type=int, default=9)
    parser.add_argument("--join-key", choices=IDENTITY[:2], default="Student_ID")
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, SchoolCalendar(args.year_start_month), args.join_key), indent=2))


if __name__ == "__main__":
    main()
