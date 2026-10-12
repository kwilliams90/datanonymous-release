"""D34 and D35 together (integration of the two branches, 2026-10-06).

D35 is the one decision for real text VALUES (the copy gate's allowlist: kept when provably safe, otherwise a
same-shape pseudonym); D34 screens the NAMES a run writes (headers, sheet, file and folder names), which cannot
be pseudonymised. One run applies both: a file whose name holds a person's name FAILs (shown redacted), while the
other files keep their safe levels and pseudonymise the rest. The leak scanner holds both indexes: the D34 flagged
names (kind ``header``) and the D35 pseudonymised units (kind ``label``), and excuses only the kept levels.
Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.model import FileResult
from datanonymous.privacy import headers as d34
from datanonymous.privacy import leaks
from test_area_a_support import profile_of, run_table

#: rare values that must never be written, printed or reported (fixture data only)
SECRETS = ("Okafor", "Quenby", "Garcia_Ana", "Ana Garcia")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def roster_rows() -> list[list[Any]]:
    rng = random.Random(7)
    rows = []
    for i in range(40):
        homeroom = "Garcia" if i < 14 else "Okafor" if i < 16 else "Room 12" if i < 28 else "Nguyen"
        club = "Robotics with Mr. Quenby" if i == 3 else rng.choice(["Chess", "Band"])
        rows.append([100100 + i * 7, homeroom, rng.choice(["White", "Black", "Asian", "Hispanic"]), club,
                     rng.randint(50, 100)])
    return rows


def everything(*dirs: Path, console: str = "") -> str:
    parts = [console]
    for d in dirs:
        for p in sorted(d.rglob("*")):
            parts.append(p.relative_to(d).as_posix())
            if p.is_file():
                parts.append(p.read_bytes().decode("utf-8", errors="replace"))
    return "\n".join(parts).casefold()


def test_one_run_fails_a_flagged_file_name_and_pseudonymises_rare_levels_elsewhere(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "roster.csv", ["student_id", "Homeroom", "Race", "Club", "Score"], roster_rows())
    write_csv(src / "scores" / "Garcia_Ana.csv", ["Week", "Score"], [[k + 1, 50 + k] for k in range(12)])
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 3  # D34: the flagged file FAILs, the other file is written
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.csv")) == ["roster.csv"]
    rows = read_dicts(out / "roster.csv")
    homeroom = Counter(r["Homeroom"] for r in rows)
    # D35: 'Garcia' is held by 14 students (>= --min-cell) and kept, although a flagged file name holds the same
    # surname (D34 indexes the name as a whole, never its single words); 'Okafor' (2 students) is pseudonymised
    # with its exact count and shape
    assert homeroom["Garcia"] == 14 and homeroom["Nguyen"] == 12 and homeroom["Room 12"] == 12
    (pseudo, n), = [(v, c) for v, c in homeroom.items() if v not in {"Garcia", "Nguyen", "Room 12"}]
    assert n == 2 and len(pseudo) == len("Okafor") and pseudo[0].isupper() and pseudo[1:].islower()
    club = Counter(r["Club"] for r in rows)
    assert set(club) - {"Chess", "Band"} and sum(club.values()) == 40
    blob = everything(out, rep, console=res.output)
    assert not [s for s in SECRETS if s.casefold() in blob], "a flagged name or a rare level was printed"
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    text = json.dumps(data)
    assert "[redacted-1]" in text and "pseudonymized_levels" in text


def test_leak_index_holds_flagged_names_and_pseudonymised_units_and_excuses_only_kept_levels() -> None:
    n = 120
    values = ["Bus"] * 60 + ["Car Rider"] * 45 + ["After Care"] * 12 + ["John Smith"] * 3
    random.Random(3).shuffle(values)
    tp, _ = run_table({"Dismissal": values, "Score": [str(i * 3) for i in range(n)]})
    p = profile_of(tp, "Dismissal")
    assert set(dtext.pseudonymized(p)) == {"John Smith"}  # D35: 3 rows, no ID: rows count
    screen = d34.NameScreen()
    screen.add("Ana Garcia", "column", frozenset({"name"}), "column 2 of 4", written=True, number=2)
    screen.flags[0].file = "gradebook.csv"
    idx = leaks.SensitiveIndex.build([FileResult("f.csv", "f.csv", "csv", tables=[tp])], screen)
    kinds = {o.kind for o in idx.origins}
    assert {leaks.HEADER, leaks.LABEL} <= kinds  # both indexes in one run
    # the kept levels are excused (public tokens of the report), never the pseudonymised one nor the flagged name
    scanner = leaks.Scanner(idx, leaks.Allow(tokens={"bus", "car", "rider", "after", "care", "ana", "garcia",
                                                     "john", "smith"}, phrases=set()))
    assert scanner.hits_in_text("Bus, Car Rider and After Care") == []
    by_kind = {idx.origins[o].kind for h in scanner.hits_in_text("John Smith") for o in h.origins}
    assert by_kind == {leaks.LABEL}
    by_kind = {idx.origins[o].kind for h in scanner.hits_in_text("the column Ana Garcia") for o in h.origins}
    assert by_kind == {leaks.HEADER}
    assert idx.flagged_in("score of ana garcia") and not idx.flagged_in("John Smith")
    assert leaks.cross_column_values(p, ["Bus", "After Care", "John Smith"]) == ["John Smith"]
