"""D37 (user, 2026-10-11): real IDs as bare-number names FAIL whatever the density of their namespace.

Standard: could a reasonable person identify a student from the output? Headers that are IDs are rare in the user's
data, so such files are flagged and skipped (the D34 FAIL; ``--keep-name`` overrides, one per flagged name).

The rule (``privacy.headers.pivot_flags``): sibling names (one table's headers, one workbook's sheets, one folder's
files and folders) that share the same text around a number no word of a numbered thing claims (a number alone,
``score1180``, ``_1180``, ``1180_score``; B5: text that differs only in runs of at most 3 letters, initials,
``FK (1180)``) are one set. The set FAILs when at least 2 of its distinct numbers (3 when the text differs in longer
words: surnames, ``Garcia (1180)``) are real IDs of one namespace (or of the run's IDs together) and at most
``max(1, n // 4)`` of its ``n`` distinct numbers are not (a few withdrawn students: at least three in four are real
IDs, or all but one), however many numbers of their length are real IDs. Still excused: every number reads as a date,
a clock time or a school year; the numbers count up in one constant step (three or more); a numbered parent context
(``rooms/``, ``enrollment_by_zip.xlsx``; A3: unless the IDs are spread like the namespace's). The B4 excuse (numbers
ending in two digits too low for a sample of the IDs: rooms and courses numbered within blocks) is gone: a set of round
numbers that are all real IDs FAILs unless it counts up in one constant step. A single bare number keeps the chance
logic (one number alone among dense IDs is chance; a labelled one is strong since K-01).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from conftest import run_cli
from datanonymous.privacy import headers as d34

SEED = "5"


class NS:
    def __init__(self, values: list[str]) -> None:
        self.raw_values = set(values)
        self.representatives = set(values)


def ids(lo: int, hi: int) -> list[str]:
    return [str(x) for x in range(lo, hi + 1)]


def keys_of(values: list[str]) -> d34.IdKeys:
    return d34.IdKeys([NS(values)])


CHARTER = ids(1001, 4500)  # a charter numbering its 3,500 students 1001-4500
DISTRICT = ids(20001, 60000)  # a district numbering its 40,000 students 20001-60000
GROUP_B = ["1180", "1349", "1518", "1561", "1792", "1929", "2014", "2518", "2538", "3075", "3887", "4323"]
CASELOAD = ["23187", "31942", "40518", "47206", "55733"]
WITHDRAWN = ["4602", "4655", "4718", "4790"]  # next to the charter's range: no ID of this run
OTHERS = ["4602", "4655", "4718", "4790", "4833", "4871", "4906", "4947", "4988"]  # no real ID, no date
ROOMS = ["1104", "1107", "1112", "1203", "1208", "1215", "1222", "2104", "2110", "2118", "2203", "2210"]
COURSES = ["1101", "1102", "1105", "1121", "1201", "1203", "1221", "1301", "1310", "1402", "1501", "1503", "1610",
           "1702", "1801", "2101", "2203", "2301"]


def picks(n: int, seed: int = 3, pool: list[str] = CHARTER) -> list[str]:
    """``n`` real IDs of ``pool`` that read as no date, time or school year (no per-number excuse)."""
    plain = [x for x in pool if not (d34.natural_number(x) or d34.date_like(x))]
    return sorted(random.Random(seed).sample(plain, n))


# =========================================================================== the attack repros now FAIL
def test_a_charter_reading_group_with_one_column_per_student_is_a_pivot() -> None:
    keys = keys_of(CHARTER)
    assert not any(keys.match(x) for x in GROUP_B)  # each alone is chance: 3,500 of 9,000 four-digit numbers
    flags = d34.id_group_flags(keys, ["Skill", *GROUP_B])
    assert set(flags) == set(range(1, 13)) and flags[1] == ("1180",)
    assert flags[7] == ("2014",)  # a real ID that reads as a year counts among real IDs


def test_a_district_caseload_with_five_real_ids_as_field_names_is_a_pivot() -> None:
    keys = keys_of(DISTRICT)  # 0.44 of all five-digit numbers are real IDs: 0.44 ** 5 was chance before D37
    assert set(d34.id_group_flags(keys, ["goal_area", *CASELOAD, "minutes"])) == set(range(1, 6))


@pytest.mark.parametrize("inside", [False, True])
def test_withdrawn_students_never_excuse_a_gradebook_of_real_ids(inside: bool) -> None:
    real = picks(22)
    if inside:  # the withdrawn students' numbers are gaps inside the range (they left the roster)
        gone = picks(4, seed=9, pool=[x for x in CHARTER if x not in real])
        keys = keys_of([x for x in CHARTER if x not in gone])
    else:  # next to the range (4602-4790)
        gone = WITHDRAWN
        keys = keys_of(CHARTER)
    heads = real + gone
    random.Random(1).shuffle(heads)
    flags = d34.id_group_flags(keys, ["Assignment", *heads])
    assert {heads[i - 1] for i in flags} == set(real)


def test_a_pair_of_real_ids_is_a_pivot_as_headers_sheets_files_or_folders() -> None:
    keys = keys_of(CHARTER)
    assert set(d34.id_group_flags(keys, ["1180", "1349"])) == {0, 1}
    assert set(d34.id_group_flags(keys, ["Name", "1180", "1349", "Total"])) == {1, 2}
    assert d34.id_group_flags(keys, ["Name", "1180", "Total"]) == {}  # one number alone: the chance logic


@pytest.mark.parametrize("make", ["score{}", "_{}", "{}_score", "{}.0", "FK ({})", "FK_{}", "{} JAS"])
def test_two_wrapped_real_ids_are_a_pivot(make: str) -> None:
    keys = keys_of(CHARTER)
    texts = ["Assignment", make.format("1180"), make.format("1349").replace("FK", "GB").replace("JAS", "MR")]
    assert set(d34.id_group_flags(keys, texts)) == {1, 2}, make


def test_real_ids_beside_differing_words_are_a_pivot_from_three() -> None:
    keys = keys_of(CHARTER)
    heads = ["Garcia (1180)", "Nguyen (1349)", "Okafor (1561)", "Smith (1792)"]  # (1180, 1349, 1518: one step)
    assert set(d34.id_group_flags(keys, ["Assignment", *heads])) == {1, 2, 3, 4}  # surnames around real IDs
    assert set(d34.id_group_flags(keys, ["Assignment", *heads[:3]])) == {1, 2, 3}
    # two differently worded names are no set (a per-student IEP file beside a roster named by one ID it holds)
    assert d34.id_group_flags(keys, ["Assignment", *heads[:2]]) == {}
    sparse_ids = [str(x) for x in random.Random(7).sample(range(100000, 1000000), 6000)]
    x, y = picks(2, seed=5, pool=sparse_ids)
    assert set(d34.id_group_flags(keys_of(sparse_ids), [f"IEP_{x}", f"Roster_{y}"])) == {0}  # the label's only


@pytest.mark.parametrize("n_real, n_other, flagged", [
    (2, 0, True), (2, 1, True), (3, 1, True), (6, 2, True), (8, 2, True), (22, 4, True), (24, 8, True),
    (1, 0, False), (1, 1, False), (2, 2, False), (3, 2, False), (6, 3, False), (12, 8, False), (24, 9, False),
])
def test_a_few_non_ids_never_excuse_a_set_that_is_otherwise_real_ids(n_real: int, n_other: int,
                                                                       flagged: bool) -> None:
    keys = keys_of(CHARTER)
    others = OTHERS[:n_other]
    assert len(set(others)) == n_other and not any(x in CHARTER for x in others)
    real = picks(n_real)
    flags = d34.id_group_flags(keys, ["Name", *real, *others])
    assert (set(flags) == set(range(1, n_real + 1))) if flagged else flags == {}, (n_real, n_other, flags)
    assert d34.mostly_real_ids(n_real, n_real + n_other) is flagged


# =========================================================================== round numbers: B4 is gone
def test_round_numbers_that_are_all_real_ids_fail_unless_they_count_up_in_one_step() -> None:
    keys = keys_of(CHARTER)
    assert set(d34.id_group_flags(keys, ["Grade", "1070", "1190", "2480", "3150"])) == {1, 2, 3, 4}
    # 15:00 and the year 2000 read as a time and a date, 3500 as neither: one that does not makes none of them one
    assert set(d34.id_group_flags(keys, ["Grade", "1500", "2000", "3500"])) == {1, 2, 3}
    # round numbers that all read as clock times on five-minute marks (10:50, 11:20, 12:30) are times: still excused
    assert d34.id_group_flags(keys, ["Grade", "1050", "1120", "1230"]) == {}
    students = keys_of(ids(1001, 2500))
    assert set(d34.pivot_flags(students, ROOMS)) == set(range(12))  # rooms within blocks: real IDs all the same
    assert set(d34.pivot_flags(students, COURSES)) == set(range(18))
    assert not hasattr(d34, "_numbered_within_blocks")
    for step in (["1050", "1100", "1150"], ["1000", "1100", "1200", "1300"], ["1010", "1020", "1030", "1040"]):
        assert d34.id_group_flags(keys, ["Grade", *step]) == {}, step  # a constant step: a numbered list


# =========================================================================== still excused
@pytest.mark.parametrize("numbers", [
    [str(y) for y in range(2019, 2026)],  # years
    ["2019", "2021", "2024"], ["2023", "2024"],  # years, not in one step
    ["2122", "2324", "2425"], ["2324", "2425"],  # school years
    ["1015", "1104", "1203"], ["1001", "1002", "1007", "1008", "1009"],  # MMDD dates
    ["1235", "1305", "1330", "1415"],  # clock times on five-minute marks (lunch periods)
    ["1001", "1002", "1003", "1004"], [str(x) for x in range(1100, 2001, 50)],  # a constant step
])
def test_dates_times_school_years_and_constant_steps_are_excused(numbers: list[str]) -> None:
    keys = keys_of(CHARTER)
    assert all(x in CHARTER for x in numbers)
    assert d34.id_group_flags(keys, ["Grade", *numbers]) == {}, numbers


def test_dates_of_an_entry_year_district_are_excused() -> None:
    eight = keys_of([f"{y}{s:04d}" for y in range(2019, 2026) for s in range(1, 1501)])
    for numbers in (["20241015", "20241016", "20241021"], ["20232024", "20242025"], ["202324", "202425"]):
        assert d34.id_group_flags(eight, ["Grade", *numbers]) == {}, numbers
    real = [f"2024{s:04d}" for s in (1342, 1377)]  # no date (month 13): the students
    assert set(d34.id_group_flags(eight, ["Grade", *real])) == {1, 2}


def test_numbered_words_and_a_numbered_parent_context_are_excused() -> None:
    keys = keys_of(CHARTER)
    for make in ("Q{}", "Item_{}", "Room {}", "Lexile_{}", "Course {}", "Period {}"):
        assert d34.id_group_flags(keys, [make.format(x) for x in GROUP_B[:4]]) == {}, make
    for parent in ("Rooms", "Course_Sections", "enrollment_by_zip"):
        context = d34.numbered_context(parent)
        assert context, parent
        assert d34.id_group_flags(keys, ROOMS, context=context) == {}, parent
    # A3: a numbered parent never hides real IDs spread like the namespace's (26 of 6,000 six-digit IDs)
    sparse_ids = [str(x) for x in random.Random(7).sample(range(100000, 1000000), 6000)]
    sparse = picks(26, seed=4, pool=sparse_ids)
    assert set(d34.id_group_flags(keys_of(sparse_ids), sparse, context=True)) == set(range(26))


def test_a_single_bare_number_keeps_the_chance_logic() -> None:
    keys = keys_of(CHARTER)
    assert d34.id_group_flags(keys, ["Name", "1180"]) == {}
    assert set(d34.id_group_flags(keys, ["Name", "Student 1180"])) == {1}  # a label: strong since K-01


# =========================================================================== end to end
def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def write_book(path: Path, sheets: dict[str, tuple[list[str], list[list[Any]]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, (header, rows) in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(header)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


def roster(src: Path, values: list[str], name: str = "roster.csv") -> None:
    rng = random.Random(4)
    write_csv(src / name, ["student_id", "grade", "score"],
              [[v, rng.choice(["6", "7", "8"]), rng.randint(0, 100)] for v in values])


def run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", SEED, "--force", *extra)
    return res, out, rep


def report(rep: Path) -> dict[str, Any]:
    return json.loads((rep / "report.json").read_text(encoding="utf-8"))


def statuses(rep: Path) -> dict[str, str]:
    return {f["source"]: f["status"] for f in report(rep)["files"]}


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([str(p) for d in dirs if d.exists() for p in sorted(d.rglob("*"))]
                     + [p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                        for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def assert_names_never_printed(numbers: list[str], blob: str) -> None:
    """A flagged number is never printed as a name: no run of the header, nor the number quoted or in a path."""
    assert ",".join(numbers[:3]) not in blob and ", ".join(numbers[:3]) not in blob
    for x in numbers:
        for form in (f"'{x}'", f'"{x}"', f"/{x}", f"{x}.csv", f"`{x}`", f"| {x} |"):
            assert form not in blob, form


def test_the_charter_reading_group_fails_and_keep_name_lets_the_whole_set_through(tmp_path: Path) -> None:
    src = tmp_path / "in"
    roster(src, CHARTER)
    rng = random.Random(2)
    rows = [[f"Skill {k}"] + [rng.randint(0, 4) for _ in GROUP_B] for k in range(1, 15)]
    write_csv(src / "Reading_Intervention_Group_B.csv", ["Skill", *GROUP_B], rows)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert "leak scan" not in res.output and "report this as a bug" not in res.output, res.output
    st = statuses(rep)
    assert st["Reading_Intervention_Group_B.csv"] == "FAIL" and st["roster.csv"] != "FAIL", st
    entry = next(f for f in report(rep)["files"] if f["source"] == "Reading_Intervention_Group_B.csv")
    reason = entry["reasons"][0]
    assert "the headers of 12 of 13 columns (columns 2-13; a real ID of this run: 12)" in reason, reason
    assert not any(x in reason for x in GROUP_B), reason  # located by column numbers, never quoted
    fixes = " ".join(entry.get("fixes") or [])
    assert "one --keep-name per flagged name" in fixes or "one --keep-name per flagged name" in res.output
    assert not (out / "Reading_Intervention_Group_B.csv").exists()
    assert_names_never_printed(GROUP_B, everything(out, rep, console=res.output))
    # --keep-name keeps one flagged name: the other eleven still FAIL the file
    some = [a for x in GROUP_B[1:] for a in ("--keep-name", x)]
    res2, out2, rep2 = run(src, tmp_path, *some, name="some")
    assert statuses(rep2)["Reading_Intervention_Group_B.csv"] == "FAIL", res2.output
    entry2 = next(f for f in report(rep2)["files"] if f["source"] == "Reading_Intervention_Group_B.csv")
    assert "the header of column 2 of 13 (a real ID of this run)" in entry2["reasons"][0], entry2["reasons"]
    every = [a for x in GROUP_B for a in ("--keep-name", x)]
    res3, out3, rep3 = run(src, tmp_path, *every, name="kept")
    assert res3.exit_code == 0, res3.output
    assert "FAIL" not in set(statuses(rep3).values())
    header = (out3 / "Reading_Intervention_Group_B.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(["Skill", *GROUP_B])
    # a config list keeps the whole set too
    cfg = tmp_path / "keep.yaml"
    cfg.write_text("keep_name: [" + ", ".join(f"'{x}'" for x in GROUP_B) + "]\n", encoding="utf-8")
    res4, out4, rep4 = run(src, tmp_path, "--config", cfg, name="cfg")
    assert res4.exit_code == 0, res4.output


def test_a_district_speech_caseload_fails_as_parquet_and_json(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    src = tmp_path / "in"
    roster(src, DISTRICT)
    rng = random.Random(6)
    goals = ["Artic", "Fluency", "Language", "Pragmatics", "Voice"]
    rows = [{"goal_area": g, **{x: rng.randint(0, 4) for x in CASELOAD}, "minutes": 30} for g in goals * 3]
    pd.DataFrame(rows).to_parquet(src / "Speech_Caseload.parquet", index=False)
    (src / "Speech_Caseload.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    st = statuses(rep)
    assert st["Speech_Caseload.parquet"] == "FAIL" and st["Speech_Caseload.json"] == "FAIL", st
    assert st["roster.csv"] != "FAIL", st
    blob = everything(out, rep, console=res.output)
    assert_names_never_printed(CASELOAD, blob)
    for x in CASELOAD:
        assert x not in blob, x


def test_a_gradebook_with_withdrawn_students_and_a_pair_of_columns_fail(tmp_path: Path) -> None:
    src = tmp_path / "in"
    roster(src, CHARTER)
    rng = random.Random(8)
    real = picks(22)
    heads = real + WITHDRAWN
    rng.shuffle(heads)
    write_csv(src / "Gradebook_P3.csv", ["Assignment", *heads],
              [[f"HW{k}"] + [rng.randint(50, 100) for _ in heads] for k in range(1, 13)])
    write_csv(src / "Buddy_Reading.csv", ["1180", "1349"], [[rng.randint(0, 20), rng.randint(0, 20)] for _ in range(9)])
    write_book(src / "Conference_Notes.xlsx", {x: (["goal", "progress"], [[f"G{k}", k] for k in range(5)])
                                               for x in ("1561", "1792")})
    for x in ("2518", "2538"):
        write_csv(src / "caseload" / f"{x}.csv", ["goal", "progress"], [[f"G{k}", 50] for k in range(4)])
        write_csv(src / "goals" / x / "notes.csv", ["date", "note"], [[f"2024-10-{d:02d}", "seen"] for d in range(1, 5)])
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    st = statuses(rep)
    assert st.pop("roster.csv") != "FAIL" and set(st.values()) == {"FAIL"} and len(st) == 7, st
    entry = next(f for f in report(rep)["files"] if f["source"] == "Gradebook_P3.csv")
    assert "the headers of 22 of 27 columns" in entry["reasons"][0], entry["reasons"]
    blob = everything(out, rep, console=res.output)
    assert_names_never_printed(real, blob)
    assert_names_never_printed(["1180", "1349", "1561", "1792", "2518", "2538"], blob)


def test_dated_stepped_and_numbered_parent_names_are_written_beside_dense_ids(tmp_path: Path) -> None:
    src = tmp_path / "in"
    roster(src, CHARTER)
    rng = random.Random(9)

    def table(cols: list[str], n: int = 6) -> list[list[Any]]:
        return [[g] + [rng.randint(0, 40) for _ in cols] for g in range(6, 6 + n)]

    years = [str(y) for y in range(2019, 2026)]
    write_csv(src / "Enrollment_Trend.csv", ["Grade", *years], table(years))
    sy = ["2122", "2223", "2324", "2425"]
    write_csv(src / "Enrollment_by_School_Year.csv", ["Grade", *sy], table(sy))
    lunch = ["1105", "1135", "1205", "1235", "1305"]
    write_csv(src / "Lunch_Counts.csv", ["Grade", *lunch], table(lunch))
    bands = ["1000", "1100", "1200", "1300", "1400"]
    write_csv(src / "Lexile_Bands.csv", ["Grade", *bands], table(bands))
    days = ["1001", "1002", "1003", "1006", "1007", "1008"]
    write_book(src / "attendance_oct.xlsx", {d: (["period", "present"], [[p, 380 + p] for p in range(1, 8)])
                                             for d in days})
    for r in ROOMS[:4]:
        write_csv(src / "Rooms" / f"{r}.csv", ["period", "course"], [[p, f"C{p}"] for p in range(1, 8)])
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "FAIL" not in set(statuses(rep).values()), statuses(rep)
    assert (out / "Enrollment_Trend.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(["Grade", *years])
    assert openpyxl.load_workbook(out / "attendance_oct.xlsx").sheetnames == days
    assert (out / "Rooms" / f"{ROOMS[0]}.csv").exists()
