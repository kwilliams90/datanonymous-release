"""Round-5 adversarial review, name-levels cluster (R5-01), as superseded by D35 (user, 2026-10-05).

Bug class: the name screen of the copy gate judged a COLUMN, so one real person among ordinary labels
(``Dismissal``: ``Bus`` / ``Walk`` / ``John Smith``), or labels that carry a name (``John Smith (Counselor)``,
``Smith (Coach)``, ``Ms. Garcia - Office``, ``Pickup: Ana Lopez``), were copied. R5-01 added a per-VALUE name
screen (``detect.name_levels``) that replaced flagged values by fake names.

D35 replaced every content heuristic for values with an allowlist (``detect.text.unit_kept``): a value is written
verbatim only when it is a vocabulary word or held by >= ``--min-cell`` distinct entities with no personal-data
shape; any other value - a rare name, a rare label carrying a name - gets a same-shape pseudonym with its exact
count, on every path (factors, dictionaries, value labels, items, tokens, every output format). A name held by
>= k entities under a header that hints no personal data is D35's accepted residual risk.

``detect.name_levels.value_name_spans`` stays for the header and path screens (D34): its per-value rules and
false-positive controls are pinned below. The format tests check that rare names never reach an xlsx or json
output (tests/test_d35_allowlist.py covers CSV, Parquet/Feather dictionaries and Stata value labels).
Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from conftest import run_cli
from datanonymous.detect import name_levels as nl
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T

from test_area_a_support import detect, run_table


def cycle(values: list[Any], n: int = 120) -> list[Any]:
    return [values[i % len(values)] for i in range(n)]


def ids(n: int = 120) -> list[str]:
    return [f"S{i + 1000}" for i in range(n)]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "42"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def texts(*dirs: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", errors="replace")
                     for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file())


def found(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> list[str]:
    blob = texts(*dirs) + "\n" + console
    return [w for w in words if w in blob]


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def file_report(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    for t in file_report(rep, name)["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def check(rep: Path, name: str, key: str) -> str:
    return file_report(rep, name)["checks"][key]["result"]


def spans(value: str, thing: bool = False) -> list[str]:
    return [value[s:e] for s, e in nl.value_name_spans(value, thing)]


# =========================================================================== the per-value rules
@pytest.mark.parametrize("value,name", [
    # rule 3: given + surname from the name lists, either order, any case
    ("John Smith", "John Smith"), ("Smith John", "Smith John"), ("Mary Jones", "Mary Jones"),
    ("JOHN SMITH", "JOHN SMITH"), ("john smith", "john smith"), ("Maria de la Cruz", "Maria de la Cruz"),
    ("Jordan Lee", "Jordan Lee"), ("John Park", "John Park"), ("Sally Field", "Sally Field"),
    ("Smith, John", "Smith, John"), ("Smith-Jones, Ana", "Smith-Jones, Ana"),
    # rule 2: initials layouts with a name word
    ("Smith, J.", "Smith, J."), ("J. Smith", "J. Smith"), ("Smith J.", "Smith J."),
    ("Garcia Lopez, M. A.", "Garcia Lopez, M. A."), ("O'Brien, K.", "O'Brien, K."),
    # rule 1: titles
    ("Ms. Garcia", "Ms. Garcia"), ("Mrs. Quenby", "Mrs. Quenby"), ("Mr. Brown", "Mr. Brown"),
    ("Coach Taylor", "Coach Taylor"), ("Dr. Smith", "Dr. Smith"),
    # rule 4: a name beside a role, set off by brackets, a dash or a colon
    ("Smith (Coach)", "Smith"), ("John Smith (Counselor)", "John Smith"), ("Mary Jones (Teacher)", "Mary Jones"),
    ("Ms. Garcia - Office", "Ms. Garcia"), ("Counselor - Garcia", "Garcia"), ("Teacher: Nguyen", "Nguyen"),
    ("Nguyen (Aide)", "Nguyen"), ("Bus - Driver: Garcia", "Garcia"), ("Grandma - Lopez", "Lopez"),
    # rule 5 and pairs inside longer labels: pickup phrases
    ("Pickup: Ana Lopez", "Ana Lopez"), ("Pickup by Ana Lopez", "Ana Lopez"), ("Picked up by Smith", "Smith"),
    ("Ana Lopez Pickup", "Ana Lopez"), ("PICKED UP BY ANA LOPEZ", "ANA LOPEZ"), ("w/ Garcia", "Garcia"),
    ("Emergency Contact: Maria Garcia", "Maria Garcia"), ("Parent pickup - Maria Garcia", "Maria Garcia"),
    ("picked up by ana lopez", "ana lopez"),
    # a quoted nickname, accents, hyphens, an English given name with a rare surname (either layout)
    ('John "Jack" Smith', 'John "Jack" Smith'), ("José García", "José García"), ("JOSÉ GARCÍA", "JOSÉ GARCÍA"),
    ("Ana-Maria Lopez", "Ana-Maria Lopez"), ("John Quenby", "John Quenby"), ("Quenby, John", "Quenby, John"),
    ("Maria Szczepanski", "Maria Szczepanski"), ("Ms Ana Lopez", "Ms Ana Lopez"), ("Mr. & Mrs. Smith", "Mrs. Smith"),
])
def test_person_name_content_is_found_value_by_value(value: str, name: str) -> None:
    assert spans(value) == [name]


@pytest.mark.parametrize("value", [
    # the review's false-positive watch list
    "Lincoln High", "King Elementary", "Washington Hall", "Head Start", "Title I", "Route 12", "Grant Funded",
    "Rose Garden", "Will Call", "May", "June", "Blue", "Red", "Lady Eagles", "Tigers",
    # institutional names after people, places, programs, everyday labels that are also names somewhere
    "Martin Luther King Jr. Elementary", "Martin Luther King Day", "Dr. Phillips High", "Smith Hall",
    "Grace Hall", "Lincoln Park", "Mary Lyon Hall", "Booker T. Washington High", "Thurgood Marshall Academy",
    "Rosa Parks Elementary", "San Diego", "Santa Maria", "Fort Wayne", "Mount Vernon",
    "Walker", "Car Rider", "Bus Rider", "Parent Pickup", "Walker - Parent", "Bus - Walker", "Pickup by Bus",
    "Pickup by Parent", "Faith Christian", "Grace Baptist", "Holy Cross", "Sun Valley", "Moon Walk",
    "Coach Bus", "Sr. Project", "Miss America", "Plan B.", "Algebra, A", "Algebra, A.", "Section B.",
    "Building A", "Vitamin D", "Summer School", "Math Lab", "Cross Country", "Infant Toddler",
    "Strongly Agree", "Below Basic", "Two or More Races", "Hispanic or Latino", "Out of School Suspension",
    "Early Dismissal", "Main Office", "Room 101", "Bell Schedule", "Park and Ride", "Kiss and Ride",
    # an English given name that is also an everyday word, or an English derived word after a given name
    "Max Capacity", "Max Enrollment", "Mark Complete", "Gene Therapy", "Ray Tracing", "Tom Cat", "Jack Rabbit",
    "Grace Period", "Amber Alert", "Oscar Award", "Picked up by bus", "go with aide",
])
def test_ordinary_labels_hold_no_name(value: str) -> None:
    assert spans(value) == []


def test_thing_headers_judge_only_strong_layouts() -> None:
    # a house, team or program may be named after someone: bare pairs are not judged under such a header
    assert spans("Marie Curie", thing=True) == [] and spans("John Muir", thing=True) == []
    # titles, initials and roles still name a person
    assert spans("Ms. Garcia", thing=True) == ["Ms. Garcia"]
    assert spans("Smith (Coach)", thing=True) == ["Smith"]
    assert spans("Smith, J.", thing=True) == ["Smith, J."]


def test_same_person_alone_elsewhere_in_the_column() -> None:
    found_ = nl.screen_levels(["Smith (Coach)", "Smith", "Smith Hall", "Bus"])
    assert set(found_) == {"Smith (Coach)", "Smith"}  # 'Smith Hall' is a building, 'Bus' a label


@pytest.mark.parametrize("token,name", [
    ("Pickup: Ana Lopez", True), ("John Smith (Counselor)", True), ("Ms. Garcia", True),
    ("Not Tested", False), ("Medical Exempt", False), ("Lincoln High", False),
])
def test_number_column_tokens(token: str, name: bool) -> None:
    assert ("name" in dtext.token_kinds(token)) is name


# =========================================================================== D35: rare names in other formats
def _rare(values: list[str], n: int = 120, other: str = "Bus") -> list[str]:
    """Each of ``values`` held by 3 students, ``other`` (an ordinary label) by the rest."""
    return [values[i % len(values)] if i < 3 * len(values) else other for i in range(n)]


def test_force_factor_keeps_real_values() -> None:
    det = detect(_rare(["John Smith"]), "Dismissal", force_factor=["Dismissal"])
    assert det.type is T.FACTOR and det.source == "force-factor" and not det.params.get(dtext.PENDING_KEY)
    tp, cols = run_table({"student_id": ids(), "Dismissal": _rare(["John Smith"])}, force_factor=["Dismissal"])
    assert Counter(cols["Dismissal"].tolist())["John Smith"] == 3


def test_xlsx_factor_with_a_rare_name(tmp_path: Path) -> None:
    import openpyxl

    src = tmp_path / "in"
    src.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["student_id", "Dismissal"])
    for s_, v in zip(ids(), _rare(["Mrs. Quenby", "Walk"], other="Bus")):
        ws.append([s_, v])
    wb.save(src / "x.xlsx")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Quenby",), out, rep, console=res.output) == []
    back = Counter(c.value for c in openpyxl.load_workbook(out / "x.xlsx").active["B"][1:])
    assert back["Bus"] == 114 and len(back) == 3 and sorted(back.values()) == [3, 3, 114]
    assert check(rep, "x.xlsx", "leak_scan") == "PASS"


def test_json_and_accented_rare_names(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    rows = [{"student_id": s_, "Dismissal": v} for s_, v in zip(ids(), _rare(["José García", "Walk"]))]
    (src / "j.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("José", "García", "Garcia"), out, rep, console=res.output) == []
    back = Counter(r["Dismissal"] for r in json.loads((out / "j.json").read_text(encoding="utf-8")))
    assert back["Bus"] == 114 and len(back) == 3
    fake = next(v for v in back if v not in ("Bus",) and back[v] == 3 and " " in v)
    assert fake[1:4].islower() and fake[0].isupper() and len(fake) == len("José García")  # same shape
    assert check(rep, "j.json", "leak_scan") == "PASS"


# =========================================================================== no false refusals
@pytest.mark.parametrize("values", [
    ["Lincoln High", "King Elementary", "Washington Hall", "Rose Garden"],
    ["Head Start", "Title I", "Grant Funded", "Will Call"],
    ["Route 12", "Route 7", "Walker", "Car Rider", "Parent Pickup"],
    ["January", "May", "June", "August"], ["Red", "Blue", "Gold", "Rose"],
    ["Eagles", "Lady Eagles", "Tigers", "Lions"],
])
def test_label_columns_keep_every_real_value(values: list[str]) -> None:
    det = detect(cycle(values), "Category")
    assert det.type is T.FACTOR and not det.params.get(nl.NAME_LEVELS_KEY)
    _, cols = run_table({"Category": cycle(values)})  # D35: each held by >= 20 rows, so kept
    assert set(cols["Category"].tolist()) == set(values)
