"""Deterministic generator for the AREA D fixtures (IDs, entities, structure).

Run from the repository root::

    python tests/fixtures/ids/make_fixtures.py

Every file is written with LF line endings and no BOM; the same script always
writes the same bytes (fixed numpy seeds). All names and values are invented.

Files
-----
* ``long_ids.csv``: long format, 1-6 rows per student. ``student_id`` (7-digit,
  zero padded) + ``state_id`` (10 digits) + ``student_number`` (``S000123``)
  are 1:1 aliases; ``family_ident`` nests students (1-3 per family);
  ``school_id`` -> ``school_name`` holds in every row (row level); gender, race
  and dob are constant within a student; ``event_date`` is unique per
  (student, date); ``score`` varies per row with some ``NA``.
* ``ids.csv``: one row per record: zero-padded ``local_id``, patterned
  ``student_number`` (``S000123``), a UUID, MD5/SHA-1/SHA-256 hex hashes,
  sequential ``row_num``, fixed-width ``ticket`` codes (``AB-1234``), zero-padded
  ``badge`` codes and org codes ``school_code`` (577/530/612).
* ``multi_table/``: ``schools.csv`` (school_id -> school_name 1:1),
  ``students.csv`` (``student_id`` zero padded to 7 characters, ``StudentNumber``)
  and ``enrollments.csv`` (``student_id`` unpadded, ``student_number``,
  ``course_code`` -> ``course_title``; about 2% orphan rows whose student is not
  in ``students.csv``).
* ``sorted.csv``: sorted by school_code asc, exit_date asc with blanks first,
  student_id asc.
* ``leading_missing.csv``: not sorted; ``new_field`` is blank in exactly the
  first 60 rows.
* ``dups.csv``: low-cardinality table with exactly 15 duplicate rows.
* ``attendance.csv``: daily attendance panel (student_id, att_date, att_code),
  unique per (student, date), sorted by student and date.
* ``snapshots.csv``: three yearly snapshots per student; ``econ_dis`` changes
  across years for some students.
* ``aggregate.csv``: school x subgroup counts with ``*`` suppression.
* ``templates.csv``: ``email = lower(first).lower(last).id@district.example-isd.org``,
  ``username = initial(first) + last`` (lower case), ``full_name = 'Last, First'``.
* ``rules.csv``: ``start_year <= grad_year`` and ``credits_earned <= credits_attempted``
  hold in every row.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

FIRST = ["Avery", "Blake", "Carmen", "Dante", "Elena", "Felix", "Greta", "Hugo", "Iris", "Jonas", "Kira", "Leon",
         "Mira", "Nolan", "Opal", "Pavel", "Quinn", "Rosa", "Silas", "Tessa", "Umar", "Vera", "Wade", "Xena",
         "Yusuf", "Zara", "Anouk", "Bram", "Celia", "Dorian"]
LAST = ["Ashby", "Brennan", "Calloway", "Dunmore", "Eastwick", "Fairley", "Garrow", "Hollis", "Ingram", "Jessop",
        "Kendrick", "Lockhart", "Marlowe", "Northcott", "Oakley", "Pembroke", "Quarry", "Ramsden", "Sterling",
        "Thorne", "Underhill", "Vance", "Whitlock", "Yardley", "Zeller", "Abernathy", "Blackwood", "Crowther"]
SCHOOLS = {"577": "Lincoln Elementary", "530": "Adams Middle", "612": "Roosevelt High"}


def write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows)
    path.write_bytes(text.encode("utf-8"))


def school_days(n: int, start: np.datetime64 = np.datetime64("2024-08-19")) -> list[str]:
    days = []
    d = start
    while len(days) < n:
        if np.is_busday(d):
            days.append(str(d))
        d = d + np.timedelta64(1, "D")
    return days


def long_ids() -> None:
    rng = np.random.default_rng(101)
    n_students = 150
    sids = rng.choice(np.arange(10_000, 999_999), n_students, replace=False)
    state = rng.choice(np.arange(4_000_000_000, 9_999_999_999), n_students, replace=False)
    snum = rng.choice(np.arange(1, 999_999), n_students, replace=False)
    fam_sizes = []
    while sum(fam_sizes) < n_students:
        fam_sizes.append(int(rng.choice([1, 1, 2, 2, 3])))
    fam_sizes[-1] -= sum(fam_sizes) - n_students
    fam_ids = rng.choice(np.arange(100_000, 999_999), len(fam_sizes), replace=False)
    family = np.repeat(fam_ids, fam_sizes)
    days = school_days(40)
    rows = []
    for s in range(n_students):
        gender = "X" if s == 17 else str(rng.choice(["F", "M"]))
        race = str(rng.choice(["W", "B", "H", "A", "NA"], p=[0.4, 0.2, 0.25, 0.1, 0.05]))
        dob = f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{int(rng.integers(2008, 2013))}"
        k = int(rng.integers(1, 7))
        dates = sorted(rng.choice(days, k, replace=False).tolist())
        home = str(rng.choice(list(SCHOOLS)))
        for d in dates:
            school = home if rng.random() < 0.85 else str(rng.choice(list(SCHOOLS)))
            score = "NA" if rng.random() < 0.08 else str(int(rng.integers(200, 401)))
            rows.append([f"{sids[s]:07d}", str(state[s]), f"S{snum[s]:06d}", str(family[s]), gender, race, dob,
                         school, SCHOOLS[school], d, score])
    write(HERE / "long_ids.csv", ["student_id", "state_id", "student_number", "family_ident", "gender", "race",
                                  "dob", "school_id", "school_name", "event_date", "score"], rows)


def ids_file() -> None:
    rng = np.random.default_rng(202)
    n = 250
    local = rng.choice(np.arange(1, 9_999_999), n, replace=False)
    snum = rng.choice(np.arange(1, 999_999), n, replace=False)
    badge = rng.choice(np.arange(1, 99_999), n, replace=False)
    rows = []
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    tickets = set()
    while len(tickets) < n:
        tickets.add(f"{letters[int(rng.integers(len(letters)))]}{letters[int(rng.integers(len(letters)))]}-"
                    f"{int(rng.integers(0, 10_000)):04d}")
    tickets_l = sorted(tickets)
    order = rng.permutation(n)
    for i in range(n):
        seed_bytes = rng.bytes(16)
        u = str(uuid.UUID(bytes=seed_bytes, version=4))
        rows.append([
            f"{local[i]:08d}", f"S{snum[i]:06d}", u,
            hashlib.md5(seed_bytes).hexdigest(), hashlib.sha1(seed_bytes).hexdigest(),
            hashlib.sha256(seed_bytes).hexdigest(), str(i + 1), tickets_l[order[i]], f"{badge[i]:06d}",
            str(rng.choice(["577", "530", "612"])), str(rng.choice(["KG", "01", "02", "03", "04", "05"])),
        ])
    write(HERE / "ids.csv", ["local_id", "student_number", "record_uuid", "md5", "sha1", "sha256", "row_num",
                             "ticket", "badge", "school_code", "grade"], rows)


def multi_table() -> None:
    rng = np.random.default_rng(303)
    d = HERE / "multi_table"
    write(d / "schools.csv", ["school_id", "school_name"], [[k, v] for k, v in SCHOOLS.items()])
    n = 300
    sids = rng.choice(np.arange(10_000, 99_999), n, replace=False)
    snums = rng.choice(np.arange(200_000, 299_999), n, replace=False)
    grades = ["KG", "01", "02", "03", "04", "05", "06", "07", "08"]
    students = []
    for i in range(n):
        students.append([f"{sids[i]:07d}", str(snums[i]), str(rng.choice(["F", "M"])), str(rng.choice(grades)),
                         str(rng.choice(list(SCHOOLS)))])
    write(d / "students.csv", ["student_id", "StudentNumber", "gender", "grade_level", "school_id"], students)
    courses = {"MATH101": "Algebra I", "ENGL102": "English Composition", "SCI103": "Earth Science",
               "HIST104": "World History", "ART105": "Studio Art", "PE106": "Physical Education"}
    days = school_days(20)
    enr = []
    for i in range(n):
        k = int(rng.choice([1, 2, 2, 3, 3, 4]))
        for c in rng.choice(list(courses), k, replace=False):
            enr.append([str(sids[i]), str(snums[i]), str(rng.choice(list(SCHOOLS))), str(c), courses[str(c)],
                        str(rng.choice(days))])
    n_orphans = round(len(enr) * 0.02)
    orphan_ids = rng.choice(np.arange(100_000, 999_999), n_orphans, replace=False)
    orphan_nums = rng.choice(np.arange(300_000, 399_999), n_orphans, replace=False)
    for j in range(n_orphans):
        c = str(rng.choice(list(courses)))
        enr.append([str(orphan_ids[j]), str(orphan_nums[j]), str(rng.choice(list(SCHOOLS))), c, courses[c],
                    str(rng.choice(days))])
    perm = rng.permutation(len(enr))
    write(d / "enrollments.csv", ["student_id", "student_number", "school_id", "course_code", "course_title",
                                  "entry_date"], [enr[i] for i in perm])


def sorted_file() -> None:
    rng = np.random.default_rng(404)
    n = 300
    sids = rng.choice(np.arange(100_000, 999_999), n, replace=False)
    days = school_days(60)
    rows = []
    for i in range(n):
        exit_date = str(rng.choice(days)) if rng.random() < 0.3 else ""
        rows.append([str(rng.choice(["530", "577", "612"])), exit_date, str(sids[i]),
                     str(rng.choice(["F", "M"])), str(int(rng.integers(0, 101)))])
    rows.sort(key=lambda r: (r[0], r[1] != "", r[1], int(r[2])))
    write(HERE / "sorted.csv", ["school_code", "exit_date", "student_id", "gender", "score"], rows)


def leading_missing() -> None:
    rng = np.random.default_rng(505)
    n = 200
    rows = []
    for i in range(n):
        new = "" if i < 60 else str(rng.choice(["yes", "no", "pending"]))
        rows.append([str(rng.choice(["A", "B", "C", "D"])), str(int(rng.integers(0, 50))), new])
    write(HERE / "leading_missing.csv", ["group", "points", "new_field"], rows)


def dups() -> None:
    rng = np.random.default_rng(606)
    base = []
    seen = set()
    while len(base) < 285:
        r = (str(rng.choice(["F", "M"])), str(rng.choice(["W", "B", "H", "A"])),
             str(rng.choice(["KG", "01", "02", "03", "04", "05"])), str(int(rng.integers(0, 1000))))
        if r not in seen:
            seen.add(r)
            base.append(list(r))
    rows = base + [list(base[int(i)]) for i in rng.choice(len(base), 15, replace=False)]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    write(HERE / "dups.csv", ["gender", "race", "grade", "points"], rows)


def attendance() -> None:
    rng = np.random.default_rng(707)
    days = school_days(25)
    rows = []
    for s in sorted(rng.choice(np.arange(100_000, 999_999), 60, replace=False).tolist()):
        start = int(rng.integers(0, 6))
        for d in days[start:]:
            rows.append([str(s), d, str(rng.choice(["P", "A", "T", "E"], p=[0.85, 0.08, 0.05, 0.02]))])
    write(HERE / "attendance.csv", ["student_id", "att_date", "att_code"], rows)


def snapshots() -> None:
    rng = np.random.default_rng(808)
    rows = []
    sids = rng.choice(np.arange(1_000_000, 9_999_999), 200, replace=False)
    for s in sids:
        g0 = int(rng.integers(0, 6))
        econ = rng.random() < 0.6
        gender = str(rng.choice(["F", "M"]))
        for y, sy in enumerate(["2022-23", "2023-24", "2024-25"]):
            if rng.random() < 0.15:
                econ = not econ
            rows.append([str(s), sy, f"{g0 + y + 1:02d}", "Y" if econ else "N", gender])
    write(HERE / "snapshots.csv", ["student_id", "school_year", "grade_level", "econ_dis", "gender"], rows)


def aggregate() -> None:
    rng = np.random.default_rng(909)
    rows = []
    for school in SCHOOLS:
        for sg in ["All", "Female", "Male", "Hispanic", "White", "Black", "Asian", "EL", "SPED", "EconDis"]:
            n = int(rng.integers(3, 400)) if sg != "All" else int(rng.integers(400, 900))
            if sg in ("Asian", "EL") and school != "612":
                n = int(rng.integers(2, 10))  # small cells are suppressed
            pct = "*" if n < 10 else f"{rng.uniform(20, 90):.1f}"
            rows.append([school, sg, str(n), pct])
    write(HERE / "aggregate.csv", ["school_code", "subgroup", "n_tested", "pct_proficient"], rows)


def templates() -> None:
    rng = np.random.default_rng(1001)
    rows = []
    for i in range(1, 301):
        f, la = str(rng.choice(FIRST)), str(rng.choice(LAST))
        email = f"{f.lower()}.{la.lower()}.{i}@district.example-isd.org"
        user = f"{f[0].lower()}{la.lower()}"
        rows.append([str(i), f, la, email, user, f'"{la}, {f}"', str(rng.choice(["F", "M"]))])
    write(HERE / "templates.csv", ["id", "first_name", "last_name", "email", "username", "full_name", "gender"], rows)


def rules() -> None:
    rng = np.random.default_rng(1102)
    rows = []
    for _ in range(120):
        start = int(rng.integers(2014, 2021))
        grad = start + int(rng.choice([3, 4, 4, 4, 5]))
        attempted = int(rng.integers(10, 41))
        earned = attempted - int(rng.choice([0, 0, 1, 2, 3, 6]))
        rows.append([str(start), str(grad), str(earned), str(attempted), str(int(rng.integers(0, 100)))])
    write(HERE / "rules.csv", ["start_year", "grad_year", "credits_earned", "credits_attempted", "points"], rows)


def main() -> None:
    long_ids()
    ids_file()
    multi_table()
    sorted_file()
    leading_missing()
    dups()
    attendance()
    snapshots()
    aggregate()
    templates()
    rules()


if __name__ == "__main__":
    main()
