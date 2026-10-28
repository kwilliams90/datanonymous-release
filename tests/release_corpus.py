"""Independently invented, deterministic school corpus for public-release attacks.

No real records, source-derived fixture, or generator seed is used here. The
only unique person-level role is an intentional membership attack: student 0
alone occurs in both IEP and discipline. Ordinary local marginal units and
person row-count classes have at least ten distinct donors.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PERSON_COUNT = 240
PROVENANCE = "Independently invented arithmetic fixture; no real student data or release generator seed."
RARE_VOCABULARY = "Speech"
MEASURE_COLUMNS = frozenset({"Assessment_Score", "Service_Minutes", "Instructional_Minutes", "Participation_Score"})


def student_id(i: int) -> str:
    return f"S{410003 + i * 37:06d}"


def state_id(i: int) -> str:
    return f"T{720007 + i * 41:06d}"


def family_id(i: int) -> str:
    return f"F{830009 + i * 43:06d}"


def corpus_rows(variant: str = "canonical") -> dict[str, list[dict[str, str]]]:
    """Return fresh records; variants each change one independently stated risk."""
    if variant not in {"canonical", "categorical_only", "rare_iep", "thirteen_events", "rare_id", "email_id"}:
        raise ValueError(f"unknown fixture variant: {variant}")
    ids = [student_id(i) for i in range(PERSON_COUNT)]
    if variant == "rare_id":
        ids[-1] = "XX-901234"
    if variant == "email_id":  # sharing mode still refuses e-mail-address IDs (ADR-0004)
        ids[-1] = "pat.quill@example.org"

    def identity(i: int) -> dict[str, str]:
        return {"Student_ID": ids[i], "State_Student_ID": state_id(i),
                "Family_ID": family_id(i // 4)}

    roster = []
    for i in range(PERSON_COUNT):
        roster.append({**identity(i), "DOB": f"2008-09-{2 + (i * 7) % 26:02d}",
                       "Grade": str(6 + i % 4), "School": ("North", "South", "West")[i % 3],
                       "Support_Program": ("Speech", "Learning", "Mobility", "General")[i % 4],
                       "Assessment_Score": f"{(5500 + (i * 71) % 3500) / 100:.2f}"})
    iep = []
    for i in range(120):
        program = ("Speech", "Learning", "Mobility")[i % 3]
        if variant == "rare_iep":
            program = RARE_VOCABULARY if i == 0 else ("Learning", "Mobility")[i % 2]
        iep.append({**identity(i), "Support_Program": program,
                    "Service_Frequency": ("Weekly", "Monthly")[i % 2],
                    "Service_Minutes": f"{(1500 + (i * 137) % 9000) / 100:.2f}"})
    # The overlap is exactly student 0: a source membership signature that a
    # source-person pseudonym map would carry intact into a synthetic release.
    discipline_people = [0, *range(121, 240)]
    discipline = []
    for position, i in enumerate(discipline_people):
        for event in range(2 if position < 60 else 3):
            discipline.append({**identity(i), "Incident_Date": f"2024-10-{3 + event:02d}",
                               "Incident_Type": ("Tardy", "Classroom", "Attendance")[(i + event) % 3],
                               "Disposition": ("Conference", "Reminder")[(i + event) % 2],
                               "Instructional_Minutes": f"{(1000 + (i * 173 + event * 59) % 12000) / 100:.2f}"})
    events = []
    for position, i in enumerate(range(120)):
        count = 2 if position < 60 else 4
        if variant == "thirteen_events" and i == 0:
            count = 13
        for event in range(count):
            events.append({**identity(i), "Event_Date": f"2024-11-{3 + event:02d}",
                           "Event_Type": ("Review", "Checkin")[(i + event) % 2],
                           "Participation_Score": f"{(5000 + (i * 149 + event * 67) % 4000) / 100:.2f}"})
    tables = {"roster.csv": roster, "iep.csv": iep, "discipline.csv": discipline, "events.csv": events}
    if variant == "categorical_only":
        for rows in tables.values():
            for row in rows:
                for column in MEASURE_COLUMNS:
                    row.pop(column, None)
    return tables


def write_corpus(destination: Path, variant: str = "canonical") -> dict[str, str]:
    """Create only a fresh input directory, never replacing earlier evidence."""
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in corpus_rows(variant).items():
        with (destination / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return hash_inventory(destination)


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def hash_inventory(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def summarize(directory: Path) -> dict:
    tables = {}
    for path in sorted(directory.glob("*.csv")):
        rows = read_records(path)
        counts = Counter(row["Student_ID"] for row in rows)
        tables[path.name] = {"rows": len(rows), "people": len(counts),
                             "person_count_classes": dict(sorted(Counter(counts.values()).items()))}
    return {"provenance": PROVENANCE, "tables": tables, "sha256": hash_inventory(directory)}


def save_evidence(path: Path, original: Path, synthetic: Path, package: dict) -> None:
    """Save local provenance outside the release; do not export a seed or map."""
    evidence = {"original_path": str(original.resolve()), "synthetic_path": str(synthetic.resolve()),
                "original": summarize(original), "synthetic": summarize(synthetic), "package": package}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write("\n")
