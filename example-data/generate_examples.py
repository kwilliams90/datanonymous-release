"""Build wholly invented public examples without reading any existing dataset.

Run with Python 3.12+: python example-data/generate_examples.py
Only the three CSV files alongside this script are overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

DESTINATION = Path(__file__).resolve().parent


class Draws:
    """Portable deterministic draws from a named, independently chosen seed.

    SHA-256 counter bytes avoid a dependency on random-library versions. This
    is an example-data recipe, not a statistical or cryptographic generator.
    """

    def __init__(self, seed: str):
        self.seed = seed
        self.counter = 0

    def integer(self, low: int, high: int) -> int:
        self.counter += 1
        message = f"{self.seed}:{self.counter}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(message).digest(), "big")
        return low + value % (high - low + 1)

    def choose(self, values: tuple[str, ...]) -> str:
        return values[self.integer(0, len(values) - 1)]


def write_csv(name: str, header: tuple[str, ...], rows: list[list], newline: str) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=newline)
    writer.writerow(header)
    writer.writerows(rows)
    (DESTINATION / name).write_bytes(buffer.getvalue().encode("utf-8"))


def bounded_score(draws: Draws, baseline: int) -> int:
    return max(0, min(100, baseline + draws.integer(-22, 22)))


def performance() -> None:
    draws = Draws("invented-performance-v1-20261011")
    education = ("some high school", "high school", "some college", "associate's degree",
                 "bachelor's degree", "master's degree")
    rows = []
    for _ in range(1000):
        gender = draws.choose(("female", "male"))
        group = draws.choose(("group A", "group B", "group C", "group D", "group E"))
        parental = draws.choose(education)
        lunch = draws.choose(("standard", "free/reduced"))
        preparation = draws.choose(("none", "completed"))
        baseline = draws.integer(30, 90) + (5 if preparation == "completed" else 0)
        rows.append([gender, group, parental, lunch, preparation,
                     bounded_score(draws, baseline), bounded_score(draws, baseline),
                     bounded_score(draws, baseline)])
    write_csv("StudentsPerformance.csv", (
        "gender", "race/ethnicity", "parental level of education", "lunch",
        "test preparation course", "math score", "reading score", "writing score"), rows, "\r\n")


def scores() -> None:
    draws = Draws("invented-scores-v1-20261011")
    # Invented syllable combinations, independently authored for these fixtures.
    first_names = tuple(start + end for start in ("Zora", "Nelu", "Vemi", "Luro", "Tavi", "Soro", "Mela", "Piri")
                        for end in ("vi", "na", "lo", "ri", "mo", "sa", "tu", "ne"))
    last_names = tuple(start + end for start in ("Vell", "Zarn", "Kelv", "Dorn", "Felm", "Norr", "Pelv", "Tarn")
                       for end in ("avo", "eri", "ulo", "ani", "eva", "ori", "umi", "ela"))
    careers = ("Teacher", "Nurse", "Engineer", "Lawyer", "Doctor", "Artist", "Architect",
               "Scientist", "Accountant", "Pilot", "Chef", "Farmer", "Librarian", "Journalist",
               "Entrepreneur", "Software Engineer", "Social Worker")
    rows = []
    for identifier in range(1, 2001):
        first = draws.choose(first_names)
        last = draws.choose(last_names)
        email = f"{first.lower()}.{last.lower()}.{identifier}@example.invalid"
        gender = draws.choose(("female", "male"))
        job = draws.choose(("True", "False"))
        absence = draws.integer(0, 20)
        activities = draws.choose(("True", "False"))
        study = draws.integer(0, 30)
        career = draws.choose(careers)
        # Toy dependencies support tutorials about correlations. They were not
        # estimated from students: demographic labels do not set these scores.
        baseline = draws.integer(35, 75) + study // 3 - absence // 4
        marks = [bounded_score(draws, baseline) for _ in range(7)]
        rows.append([identifier, first, last, email, gender, job, absence,
                     activities, study, career, *marks])
    write_csv("student-scores.csv", (
        "id", "first_name", "last_name", "email", "gender", "part_time_job",
        "absence_days", "extracurricular_activities", "weekly_self_study_hours", "career_aspiration",
        "math_score", "history_score", "physics_score", "chemistry_score", "biology_score",
        "english_score", "geography_score"), rows, "\n")


def campus() -> None:
    draws = Draws("invented-campus-v1-20261011")
    years = ("first year", "second year", "third year", "fourth year")
    courses = ("ART101", "SCI202", "BUS303", "TEC404")
    # Generic direction words exercise literal trailing-space round trips.
    regions = ("Central ", "Eastern ", "Western", "Northern", "Southern", "Coast")
    grades = ("D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A")
    rows = []
    for i in range(62):
        year = i % 4
        course = courses[i % len(courses)]
        age = 18 + year + draws.integer(0, 8)
        height = draws.integer(145, 195)
        weight = draws.integer(45, 95)
        join_year = 2026 - year
        exam_year = join_year - draws.integer(1, 3)
        region = draws.choose(regions)
        # Currency/unit choices are arbitrary teaching values, with broad ranges
        # that exercise continuous-integer generation on a small table.
        semester = draws.integer(18000, 52000)
        accommodation = draws.integer(4000, 24000)
        grade = draws.choose(grades)
        rows.append([years[year], course, age, height, weight, exam_year,
                     join_year, region, semester, accommodation, grade])
    write_csv("students_data.csv", (
        "YearofStudy", "Course", "Age", "ApproxHeight", "ApproxWeight", "KCSE", "Yr_JoinCampus",
        "SitKCSE", "Expense_Semester", "Expense_Accommodation", "Previous_Exam_MeanGrade"), rows, "\n")


if __name__ == "__main__":
    performance()
    scores()
    campus()
