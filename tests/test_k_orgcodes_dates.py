"""Known issues, cluster orgcodes-dates: D36 (person names inside organisational codes), K-07 (the date sentinel rule
had no k floor) and K-09 (org-code levels with a street-address or URL shape were kept verbatim).

D36 (user decision 2026-10-07, G3): an organisational code (D10) keeps its real values, but a section code often
carries its teacher's name (``ALG1-03 Garcia, Ana``). Such a value keeps everything except the name part, which gets a
consistent fake name of the same layout; the same real value gets the same synthetic value in every file and column of
the run (joins still match, the course part stays real); the fake name is never a word of any real text of the run;
the leak scanner indexes the real name parts as sensitive (it excused kept org-code levels before). Names are found
with the D34 checks of headers (explicit layouts, known first + last pairs), never new name lists; a bare single
surname follows D10 unchanged (accepted residual).

K-07 (G2): a date value held by >= 1% of the rows that lay >= 50 years from the median became a sentinel, written
verbatim with no floor: in a column of 100 rows or fewer one real far-off birth date was copied to the data, the docs
and the report. Now the far rule needs >= --min-cell entities and never applies to DOB-hinted (person-level) columns;
only the fixed sentinels (1900-01-01, 9999-12-31, 0/0/0) are unconditional; the exact min/max of D14 stay.

K-09 (G2): an org-code level with a street-address or URL shape (``Homebound - 1427 Juniper Ridge Rd`` under
``School_Name``) was written verbatim even when one student held it; it is now pseudonymised one by one like e-mail,
phone and SSN shapes, and the address shape test finds an address after a spaced dash.

Hardening (attack pass): names with middle initials and several given names are replaced whole (D36-MI); title
layouts the D34 check misses (particles, two-letter surnames, an eponym word or a sentence after the name) are read by
layouts of D36's own that keep D34's school/company/place exemptions (D36-HELPER-MISS); only the name parts of a code
written with a fake name are indexed, never its kept course words (D36-PHRASE-FAIL), and long descriptions are read in
runs of a few words; a code's name is replaced in every other column that would write it verbatim, also inside comma
lists (D36-CROSSCOL); the spaced-dash address start applies to organisational codes only (K09-DASH-FP); the zero days
of common date systems (Excel's 12/30/1899 ...) are fixed sentinels (K07-PLACEHOLDER-COUNTS).

Hardening 2 (second attack pass): a known first + last pair and 'Surname, Initial' with a common surname ('Lisa Miller',
'Miller, L') are names (D36-COMMON-SURNAME); 'Surname, Given' reads two-letter, apostrophe and hyphenated names ('Le,
Thanh', "D'Souza, Priya", 'Park, Ji-hoon': D36-COMMA-SHORT-SURNAMES); another column's shorter form, other order or
initial of a code's name is replaced too, and accented names are placed letter by letter (D36-CROSSCOL-PARTIAL);
'MS' / 'Ms' without a period is also a middle school, so 'MS HOPE SQUAD' and 'Ms Band' stay real
(D36-FALSE-POSITIVE-COURSES); a zero day with a time of day is a bare time, no placeholder (K07-DAY0-TIME-OF-DAY).

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from conftest import read_dicts, run_cli
from datanonymous import pipeline
from datanonymous.config import Settings
from datanonymous.detect import org_names
from datanonymous.detect import text as dtext
from datanonymous.generate import pseudonyms
from datanonymous.privacy import leaks
from datanonymous.privacy.leaks import Scanner, SensitiveIndex, build_allow

from test_area_a_support import detect, profile_of, run_table

SEED = "1"
#: Section codes of a school (D36): teachers' names in explicit layouts, and codes without one.
NAMED = ["ALG1-03 Garcia, Ana", "Algebra I - Ms. Okafor", "English 9 (A. Garcia)", "BIO-2 Nwosu, Chinedu",
         "HR 101 Mr Smith", "Sec 3 Garcia, Ana / Lopez, Maria", "Mrs. G", "ALGEBRA I GARCIA, ANA"]
PLAIN = ["GEO-01", "Lincoln Elementary", "Homeroom Garcia"]  # no layout: D10 (a bare surname is the residual)
#: The real name words of the fixture (none may be written).
NAME_WORDS = ("garcia", "okafor", "nwosu", "chinedu", "smith", "lopez", "maria")


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def cycle(values: list[Any], n: int) -> list[Any]:
    return [values[i % len(values)] for i in range(n)]


def sids(n: int, start: int = 100000) -> list[str]:
    return [f"S{start + i}" for i in range(n)]


def run(src: Path, tmp: Path, *extra: str, name: str = "run") -> tuple[Any, Path, Path]:
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", SEED, *extra)
    return res, out, rep


def everything(out: Path, rep: Path, console: str = "") -> str:
    """Every byte the run wrote (data, docs, report), case-folded, and the console output."""
    parts = [console]
    for root in (out, rep):
        for f in sorted(root.rglob("*")):
            if f.is_file():
                parts.append(f.read_bytes().decode("utf-8", errors="replace"))
    return "\n".join(parts).casefold()


def column(path: Path, name: str) -> list[str]:
    return [r[name] for r in read_dicts(path)]


def file_page(rep: Path, name: str) -> dict[str, Any]:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def report_column(rep: Path, file: str, name: str) -> dict[str, Any]:
    return next(c for t in file_page(rep, file)["tables"] for c in t["columns"] if c["name"] == name)


def school(tmp: Path, n: int = 220) -> Path:
    """enroll.csv (one row per student) and sections.csv (a code table) share the Section codes."""
    values = NAMED + PLAIN
    d = tmp / "in"
    write_csv(d / "enroll.csv", {"student_id": sids(n), "Section": cycle(values, n),
                                 "score": cycle(list(range(40, 100)), n)})
    write_csv(d / "sections.csv", {"Section": values, "room": [str(100 + i) for i in range(len(values))]})
    return d


# =========================================================================== D36: locating the name part
@pytest.mark.parametrize("value,parts,replaced", [
    ("ALG1-03 Garcia, Ana", ["Garcia, Ana"], [("Garcia", "last"), ("Ana", "first")]),
    ("Algebra I - Ms. Garcia", ["Ms. Garcia"], [("Garcia", "last")]),  # the title stays
    ("HR 101 Mr Smith", ["Mr Smith"], [("Smith", "last")]),
    ("English 9 (A. Garcia)", ["A. Garcia"], [("A", "initial"), ("Garcia", "last")]),
    ("Mrs. G", ["Mrs. G"], [("G", "initial")]),
    ("Period 2 - Smith, J.", ["Smith, J"], [("Smith", "last"), ("J", "initial")]),
    ("ELA 7 Ana Garcia", ["Ana Garcia"], [("Ana", "first"), ("Garcia", "last")]),
    ("Garcia Lopez, Maria - Rm 4", ["Garcia Lopez, Maria"],  # a compound surname: widened over 'Garcia'
     [("Garcia", "last"), ("Lopez", "last"), ("Maria", "first")]),
    ("Sec 3 Garcia, Ana / Lopez, Maria", ["Garcia, Ana", "Lopez, Maria"],
     [("Garcia", "last"), ("Ana", "first"), ("Lopez", "last"), ("Maria", "first")]),
    ("ALGEBRA I GARCIA, ANA", ["GARCIA, ANA"], [("GARCIA", "last"), ("ANA", "first")]),  # 'I' is the course's
    ("Mr. G's Class", ["Mr. G"], [("G", "initial")]),  # the possessive stays
])
def test_find_locates_the_name_part(value: str, parts: list[str], replaced: list[tuple[str, str]]) -> None:
    got = org_names.find(value)
    assert got is not None and got.tokens is not None
    assert list(got.parts) == parts
    assert [(value[a:b], role) for a, b, role in got.tokens] == replaced


@pytest.mark.parametrize("value", ["GEO-01", "Lincoln Elementary", "Homeroom Garcia", "Garcia", "MATH-101-01",
                                   "Jesse H Jones High School", "Rosa Parks Elementary", "Allen, Texas",
                                   "Smith Hall", "Martin Luther King Jr Elementary", "Section 04 - Rm 210",
                                   "AP US History", "Fall 2024", "Garcia P1"])
def test_find_leaves_codes_without_a_layout(value: str) -> None:
    # D34's exemptions (schools, places, namesakes, buildings) hold; a bare surname is the accepted residual (D10)
    assert org_names.find(value) is None


def test_names_in_other_spellings() -> None:
    # accents, apostrophes, camelCase and transliterated Cyrillic keep their places: the name pieces are located
    cyrillic = ("Иван", "Петров")
    for value, pieces in (("Sec 2 García, Ana", ["García", "Ana"]),
                          ("Sec 2 O'Brien, Sean", ["O'Brien", "Sean"]), ("Sec 4 McDonald, Ana", ["McDonald", "Ana"]),
                          ("Sec 1 " + " ".join(cyrillic), list(cyrillic)),
                          # folding that changes the length is followed letter by letter (D36-CROSSCOL-PARTIAL c)
                          ("Sec 1 Strauß, Anna", ["Strauß", "Anna"]),
                          ("ENG9-01 Æbelø, Søren", ["Æbelø", "Søren"]),
                          (unicodedata.normalize("NFD", "BIO-02 Núñez, José"),
                           [unicodedata.normalize("NFD", w) for w in ("Núñez", "José")])):
        got = org_names.find(value)
        assert got is not None and got.tokens is not None
        assert [value[a:b] for a, b, _ in got.tokens] == pieces
        assert got.parts and all(part in value for part in got.parts)
    # a word transliterated whole that its letters do not spell alone (all-caps Cyrillic: a word's 'Ж' reads 'ZH',
    # alone 'Zh'), or a name the Latin tokens cannot read (Chinese): no safe place, so the whole code is pseudonymised
    # (fail-safe)
    for value in ("Sec 1 ЖАННА ИВАНОВА", "ALG 张伟"):
        got = org_names.find(value)
        assert got is not None and got.tokens is None


def test_a_name_that_cannot_be_located_is_pseudonymised_whole() -> None:
    values = cycle(["Sec 1 ЖАННА ИВАНОВА", "Sec 2 Garcia, Ana", "GEO-01"], 60)
    tp, cols = run_table({"student_id": sids(60), "Section": values})
    p = profile_of(tp, "Section")
    assert p.params["org_code_names_replaced"] == 2
    written = set(cols["Section"].tolist())
    assert "GEO-01" in written and len(written) == 3
    assert not any(re.search(rf"\b{w}", v.casefold())
                   for v in written for w in ("жанна", "иванова", "garcia\\b"))
    assert any(v.startswith("Sec 2 ") for v in written)  # located: the course part stays


# =========================================================================== D36: the run
def test_names_in_section_codes_get_consistent_fake_names(tmp_path: Path) -> None:
    src = school(tmp_path)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    leaked = [w for w in NAME_WORDS if re.search(rf"\b{w}\b", blob)]
    # 'garcia' stays only in the bare-surname code (the accepted residual), nowhere else
    assert leaked == ["garcia"] and blob.count("homeroom garcia") == blob.count("garcia"), leaked
    enroll, sections = column(out / "enroll.csv", "Section"), column(out / "sections.csv", "Section")
    real = NAMED + PLAIN
    # N in = N out and exact counts: every real code maps to one written code with its count
    real_counts = Counter(cycle(real, 220))
    assert sorted(Counter(enroll).values()) == sorted(real_counts.values())
    # codes without a layout stay real; codes with one are never written as they are
    assert set(PLAIN) <= set(enroll) and not set(NAMED) & set(enroll)
    # the same synthetic code in both files: joins on Section still match
    assert set(enroll) == set(sections) and len(set(sections)) == len(real)
    by_prefix = {v.split(" ")[0]: v for v in sections}
    # the course part stays real, the layout is kept
    assert re.fullmatch(r"ALG1-03 [A-Z][a-z]+, [A-Z][a-z]+", by_prefix["ALG1-03"])
    assert re.fullmatch(r"Algebra I - Ms\. [A-Z][a-zA-Z]+", next(v for v in sections if v.startswith("Algebra I")))
    assert re.fullmatch(r"English 9 \([A-Z]\. [A-Z][a-zA-Z]+\)", by_prefix["English"])
    assert re.fullmatch(r"HR 101 Mr [A-Z][a-zA-Z]+", next(v for v in sections if v.startswith("HR 101")))
    assert re.fullmatch(r"Mrs\. [A-Z]", next(v for v in sections if v.startswith("Mrs.")))
    caps = next(v for v in sections if v.startswith("ALGEBRA I "))
    assert re.fullmatch(r"ALGEBRA I [A-Z]+, [A-Z]+", caps)
    # one fake per real name word: 'Garcia, Ana' is the same fake in every code that holds it
    fake = by_prefix["ALG1-03"].removeprefix("ALG1-03 ")
    assert next(v for v in sections if v.startswith("Sec 3 ")).startswith(f"Sec 3 {fake} / ")
    assert caps == f"ALGEBRA I {fake.upper()}"
    assert by_prefix["English"].endswith(f"{fake.split(',')[0]})")
    # the report gives counts only
    col = report_column(rep, "enroll.csv", "Section")
    assert col["type"] == "ORG_CODE"
    assert col["params"]["org_code_names_replaced"] == len(NAMED)
    assert col["params"]["pseudonymized_levels"] == len(NAMED)
    assert "org_code_names" in {m["code"] for m in col["messages"]}
    assert file_page(rep, "enroll.csv")["checks"]["leak_scan"]["result"] == "PASS"


def test_fake_names_never_equal_a_real_word_of_the_run(tmp_path: Path) -> None:
    # harvest the fake names one run draws, then plant them as real (kept) values of another column: the next run
    # must draw others (the draws are seeded, so without the avoidance they would come back)
    src = school(tmp_path / "a")
    res, out, _ = run(src, tmp_path, name="harvest")
    assert res.exit_code == 0, res.output
    first = column(out / "sections.csv", "Section")
    fake = next(v for v in first if v.startswith("ALG1-03 ")).removeprefix("ALG1-03 ")
    words = re.findall(r"[A-Za-z]+", fake)
    n = 220
    d = tmp_path / "b" / "in"
    write_csv(d / "enroll.csv", {"student_id": sids(n), "Section": cycle(NAMED + PLAIN, n),
                                 "score": cycle(list(range(40, 100)), n), "Club": cycle(words, n)})
    write_csv(d / "sections.csv", {"Section": NAMED + PLAIN, "room": [str(100 + i) for i in range(11)]})
    res, out, rep = run(d, tmp_path, name="planted")
    assert res.exit_code == 0, res.output
    assert set(column(out / "enroll.csv", "Club")) == set(words)  # held by >= 10 people: kept real (D35)
    second = column(out / "sections.csv", "Section")
    assert not any(re.search(rf"\b{w}\b", v) for v in second for w in words if not v.startswith(("GEO", "Lincoln")))


def test_the_same_seed_gives_the_same_output(tmp_path: Path) -> None:
    src = school(tmp_path)
    outs = []
    for name in ("one", "two"):
        res, out, _ = run(src, tmp_path, name=name)
        assert res.exit_code == 0, res.output
        outs.append([(out / f).read_bytes() for f in ("enroll.csv", "sections.csv")])
    assert outs[0] == outs[1]


def test_the_fixture_is_hot_without_d36(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without D36 the teachers' names are written verbatim and the scanner excuses them (kept org-code levels)."""
    monkeypatch.setattr(dtext, "org_code_names", lambda p, texts, gone: {})
    src = school(tmp_path)
    res, out, rep = run(src, tmp_path, name="hot")
    assert res.exit_code == 0, res.output
    assert set(NAMED) <= set(column(out / "sections.csv", "Section"))
    assert file_page(rep, "enroll.csv")["checks"]["leak_scan"]["result"] == "PASS"


def test_the_scanner_never_excuses_a_named_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a generator bug that writes the real code: the never-excused unit FAILs the file, and nothing real is printed
    def copy_named(forms, rep, named, names, drawer, drawn, mapping, pyr):
        for t in forms:
            mapping[t] = t
            drawn.add(t)

    monkeypatch.setattr(pseudonyms, "_draw_named", copy_named)
    src = school(tmp_path)
    res, out, rep = run(src, tmp_path, name="leaky")
    assert res.exit_code == 3, res.output
    assert not (out / "enroll.csv").exists() and not (out / "sections.csv").exists()
    blob = everything(out, rep, res.output)
    assert not any(re.search(rf"\b{w}\b", blob) for w in NAME_WORDS if w != "garcia")


def test_the_scanner_indexes_the_name_parts(tmp_path: Path) -> None:
    src = school(tmp_path)
    run_ = pipeline.execute(Settings(mode="fidelity", seed=1), src, tmp_path / "out", tmp_path / "rep")
    assert pipeline.exit_code(run_) == 0
    scanner = Scanner(SensitiveIndex.build(run_.files), build_allow(run_))
    for text in ("lesson plan by Garcia, Ana", "room of Ms. Okafor", "Nwosu, Chinedu"):
        assert scanner.hits_in_text(text), "a real name part is not indexed"
    # a bare surname is no layout: the code 'Homeroom Garcia' is written as it is, and so is 'Garcia' alone
    assert not scanner.hits_in_text("Homeroom Garcia")
    tp = next(fr for fr in run_.files if fr.rel_path == "enroll.csv").tables[0]
    p = profile_of(tp, "Section")
    assert {"Garcia, Ana", "Ms. Okafor", "Mrs. G"} <= set(dtext.org_name_parts(p))
    registry = leaks.run_texts(tp)
    assert registry is not None and registry.holds("Garcia, Ana") and registry.holds("ms. okafor")
    assert not registry.holds("Homeroom Garcia")


def test_force_factor_keeps_the_real_codes(tmp_path: Path) -> None:
    src = school(tmp_path)
    res, out, _ = run(src, tmp_path, "--force-factor", "Section", name="forced")
    assert res.exit_code == 0, res.output
    assert set(NAMED) <= set(column(out / "enroll.csv", "Section"))


def test_inspect_with_named_codes(tmp_path: Path) -> None:
    res = run_cli("inspect", school(tmp_path))
    assert res.exit_code == 0, res.output
    assert not any(re.search(rf"\b{w}\b", res.output.casefold()) for w in NAME_WORDS if w != "garcia")


def test_an_org_code_header_is_never_split_into_list_items() -> None:
    # two codes of three hold a comma: under a neutral header a comma list, under an org-code header whole codes, so
    # the teacher's surname and given name are not kept as two items held by many students (a full name put back)
    values = cycle(["ALG1 Garcia, Ana", "BIO Lopez, Maria", "GEO-01"], 60)
    assert detect(values, "Activities").type.value == "MULTI_SELECT"
    assert detect(values, "Section").type.value == "ORG_CODE"
    tp, cols = run_table({"student_id": sids(60), "Section": values})
    written = set(cols["Section"].tolist())
    assert "GEO-01" in written and len(written) == 3
    assert not any(re.search(rf"\b{w}\b", v.casefold()) for v in written for w in ("garcia", "ana", "lopez", "maria"))
    assert {v.split(" ")[0] for v in written} == {"ALG1", "BIO", "GEO-01"}


def test_a_code_pseudonymised_whole_elsewhere_is_pseudonymised_whole_everywhere() -> None:
    # the same text under a personal header (--org-code Teacher: nothing kept, R6-03) and under Section (D36): one
    # pseudonym for the run, the whole-value one, so no real fragment is written under the personal header
    values = cycle(["ALG1-03 Garcia, Ana", "GEO-01 Lopez, Maria"], 60)
    tp, cols = run_table({"student_id": sids(60), "Section": values, "Teacher": values}, org_code=["Teacher"])
    assert list(cols["Section"]) == list(cols["Teacher"])
    assert not any(v.startswith(("ALG1-03 ", "GEO-01 ")) for v in cols["Section"])


def test_a_personal_header_still_pseudonymises_every_level() -> None:
    # --org-code Teacher names the type only: every level is pseudonymised whole (R6-03), D36 does not apply
    values = cycle(["Garcia, Ana", "Lopez, Maria", "Smith, J."], 60)
    tp, cols = run_table({"student_id": sids(60), "Teacher": values}, org_code=["Teacher"])
    p = profile_of(tp, "Teacher")
    assert dtext.ORG_NAMES_KEY not in p.params and not set(cols["Teacher"]) & set(values)


# =========================================================================== K-09: shapes in organisational codes
def test_org_code_address_and_url_shapes_are_never_kept() -> None:
    assert not dtext.unit_kept("Homebound - 1427 Juniper Ridge Rd", 10_000, 10, org_code=True)
    assert not dtext.unit_kept("https://sites.example.edu/~kquenby/home", 10_000, 10, org_code=True)
    assert not dtext.unit_kept("Lincoln Elementary (123 Main St)", 10_000, 10, org_code=True)
    assert dtext.unit_kept("Lincoln Elementary", 1, 10, org_code=True)  # D10: a rare code keeps its value


@pytest.mark.parametrize("value,address", [
    ("Homebound - 1427 Juniper Ridge Rd", True),  # an address after a spaced dash (K-09)
    ("Building 2 – 300 Main Street", True), ("Homebound_-_1427_Juniper_Ridge_Rd", True),
    ("Grades 9 - 12", False), ("Period 1 - 2", False), ("K - 5 Elementary", False), ("Algebra I - 2 Way", False),
    ("Lincoln - 2nd Grade", False), ("Homebound 1427 Juniper", False),
])
def test_org_code_address_after_a_spaced_dash(value: str, address: bool) -> None:
    # the wider address start applies to organisational codes only (kept whatever their count): never to other
    # levels (kept only when >= --min-cell people hold them) nor to headers (D34 refuses a file on a shape)
    assert dtext.org_code_address(value) is address
    assert dtext.unit_kept(value, 10_000, 10, org_code=True) is not address
    assert "address" not in dtext.unit_shapes(value)


def test_k09_repro_address_and_url_levels(tmp_path: Path) -> None:
    n = 120
    school_name = cycle(["Lincoln Elementary", "Roosevelt Middle"], n - 1) + ["Homebound - 1427 Juniper Ridge Rd"]
    campus = ["https://sites.example.edu/~kquenby/home"] + cycle(["North", "South"], n - 1)
    src = write_csv(tmp_path / "in" / "c13_org.csv", {"student_id": sids(n), "School_Name": school_name,
                                                       "Campus": campus})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    assert "juniper" not in blob and "kquenby" not in blob
    written = column(out / "c13_org.csv", "School_Name")
    assert Counter(written)["Lincoln Elementary"] == 60 and Counter(written)["Roosevelt Middle"] == 59
    assert len(set(written)) == 3  # the address level has a pseudonym with its exact count
    assert set(column(out / "c13_org.csv", "Campus")) >= {"North", "South"}
    for name in ("School_Name", "Campus"):
        assert report_column(rep, "c13_org.csv", name)["params"]["pseudonymized_levels"] == 1


# =========================================================================== K-07: the date sentinel floor
def _dates(n: int, start: int = 2008) -> list[str]:
    return [f"{start + i % 5}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)]


def test_dob_hinted_columns_never_get_far_sentinels() -> None:
    far = _dates(300) + ["1952-03-17"] * 12
    det = detect(far, "birth_date")
    assert "sentinels" not in det.params and det.params.get("person_level") is True
    # the fixed sentinels stay unconditional
    det = detect(_dates(300) + ["1900-01-01"] * 3, "birth_date")
    assert det.params["sentinels"] == {"1900-01-01": 3}


def test_far_values_need_min_cell_rows() -> None:
    assert "sentinels" not in detect(_dates(300) + ["1850-06-01"] * 9, "exit_date").params
    assert detect(_dates(300) + ["1850-06-01"] * 10, "exit_date").params["sentinels"] == {"1850-06-01": 10}
    assert "sentinels" not in detect(_dates(300) + ["1850-06-01"] * 12, "exit_date", min_cell=20).params


def test_k07_repro_real_far_birth_dates_are_not_copied(tmp_path: Path) -> None:
    n = 60
    dob = _dates(n)
    real = {11: "1952-03-17", 26: "1955-11-02", 41: "1958-07-29"}
    for i, v in real.items():
        dob[i] = v
    src = write_csv(tmp_path / "in" / "students.csv", {"student_id": sids(n), "birth_date": dob,
                                                       "group": cycle(["A", "B", "C"], n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    written = column(out / "students.csv", "birth_date")
    assert written.count("1952-03-17") == 1  # the exact min (D14) only
    blob = everything(out, rep, res.output)
    assert "1955-11-02" not in blob and "1958-07-29" not in blob
    assert "sentinel_tokens" not in (out / "schema.json").read_text(encoding="utf-8")
    assert "sentinels" not in report_column(rep, "students.csv", "birth_date")["params"]


def test_a_far_value_of_one_student_is_read_as_a_date(tmp_path: Path) -> None:
    # 'exit_date' (no DOB hint), long format: one student holds 1890-05-05 on 12 rows (>= 1% and >= 10 rows, but one
    # person); the column is read again with it as a date: no sentinel token in the data, the docs or the report (an
    # ordinary date: the exact min of D14 and the interpolation over its rows)
    n = 300
    ids = sids(n // 3) * 3
    exit_dates = _dates(n)
    for i in range(12):
        exit_dates[i * 25] = "1890-05-05"
        ids[i * 25] = "S999999"
    src = write_csv(tmp_path / "in" / "exits.csv", {"student_id": ids, "exit_date": exit_dates,
                                                    "code": cycle(["W1", "W2", "G1"], n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    col = report_column(rep, "exits.csv", "exit_date")
    assert "sentinels" not in col["params"]
    assert "date_far_sentinels" in {m["code"] for m in col["messages"]}
    assert "sentinel_tokens" not in (out / "schema.json").read_text(encoding="utf-8")
    assert '"sentinels"' not in json.dumps(file_page(rep, "exits.csv"))


def test_a_far_placeholder_many_people_hold_stays_a_sentinel(tmp_path: Path) -> None:
    # the control: 2099-12-31 ('still enrolled') held by 30 students is a genuine placeholder, kept at its rate
    n = 300
    exit_dates = _dates(n)
    for i in range(30):
        exit_dates[i * 10] = "2099-12-31"
    src = write_csv(tmp_path / "in" / "exits.csv", {"student_id": sids(n), "exit_date": exit_dates})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert report_column(rep, "exits.csv", "exit_date")["params"]["sentinels"] == {"2099-12-31": 30}
    assert column(out / "exits.csv", "exit_date").count("2099-12-31") == 30


def test_far_sentinel_floor_in_profile_table() -> None:
    n = 300
    ids = sids(n // 3) * 3
    vals = _dates(n)
    for i in range(12):
        vals[i * 25] = "1890-05-05"
        ids[i * 25] = "S999999"
    tp, cols = run_table({"student_id": ids, "exit_date": vals})
    p = profile_of(tp, "exit_date")
    assert "sentinels" not in p.params and not p.params.get("_far_sentinels")
    xs = p.model.sorted_numeric
    assert xs[0] == xs[11] < xs[12]  # its 12 rows are dated rows of the model now
    assert "date_far_sentinels" in {m.code for m in p.messages} and "date_sentinels" not in {m.code for m in p.messages}
    # rows only (no ID column): 12 rows are 12 holders, the value stays a sentinel
    tp, cols = run_table({"exit_date": vals, "code": cycle(["A", "B"], n)})
    assert profile_of(tp, "exit_date").params["sentinels"] == {"1890-05-05": 12}
    assert list(cols["exit_date"]).count("1890-05-05") == 12


# =========================================================================== hardening: the attack pass
# D36-MI: a name with a middle initial or several given names is replaced whole (the D34 check reads 'Robert J' alone)
@pytest.mark.parametrize("value,parts,replaced", [
    ("ALG1-03 Thompson, Robert J", ["Thompson, Robert J"],
     [("Thompson", "last"), ("Robert", "first"), ("J", "initial")]),
    ("GEOM-02 Robert J Thompson", ["Robert J Thompson"], [("Robert", "first"), ("J", "initial"), ("Thompson", "last")]),
    ("ALG2-01 Mary K. Johnson", ["Mary K. Johnson"], [("Mary", "first"), ("K", "initial"), ("Johnson", "last")]),
    ("SPAN-03 Okafor, Chidi E.", ["Okafor, Chidi E"], [("Okafor", "last"), ("Chidi", "first"), ("E", "initial")]),
    ("ART-01 Abernathy, RJ", ["Abernathy, RJ"], [("Abernathy", "last"), ("R", "initial"), ("J", "initial")]),
    ("PE-04 Garcia, Ana Maria L", ["Garcia, Ana Maria L"],
     [("Garcia", "last"), ("Ana", "first"), ("Maria", "first"), ("L", "initial")]),
    # D36-HELPER-MISS: title layouts the D34 check does not read (particles, two-letter surnames, an eponym word
    # after the name, a sentence after it read as a company)
    ("SCI8 Ms. de la Cruz", ["Ms. de la Cruz"], [("Cruz", "last")]),
    ("ENG9 Ms. Le", ["Ms. Le"], [("Le", "last")]),
    ("HIST Mrs. Van Der Berg", ["Mrs. Van Der Berg"], [("Van", "last"), ("Der", "last"), ("Berg", "last")]),
    ("ESL Ms. Kim Soo-jin", ["Ms. Kim Soo-jin"], [("Kim", "first"), ("Soo", "first"), ("jin", "last")]),
    ("Civics Mr. De Leon", ["Mr. De Leon"], [("De", "last"), ("Leon", "last")]),
    ("Gov Mr. Vo", ["Mr. Vo"], [("Vo", "last")]), ("Gov Mr. Ng", ["Mr. Ng"], [("Ng", "last")]),
    ("Mr. Okafor Debate", ["Mr. Okafor"], [("Okafor", "last")]),
    ("Ms. Garcia Law 1", ["Ms. Garcia"], [("Garcia", "last")]),
    ("Lab with Mrs. Abernathy. Co-taught by Ms. Nguyen.", ["Mrs. Abernathy", "Ms. Nguyen"],
     [("Abernathy", "last"), ("Nguyen", "last")]),
    ("Lab with Dr. Patel. Co-taught by Ms. Nguyen.", ["Dr. Patel", "Ms. Nguyen"],
     [("Patel", "last"), ("Nguyen", "last")]),
])
def test_find_replaces_the_whole_name(value: str, parts: list[str], replaced: list[tuple[str, str]]) -> None:
    got = org_names.find(value)
    assert got is not None and got.tokens is not None
    assert list(got.parts) == parts
    assert [(value[a:b], role) for a, b, role in got.tokens] == replaced


@pytest.mark.parametrize("value", ["Mrs. Quenby's Academy", "Dr. Phillips High School", "Mr. Patel Brothers",
                                   "Austin, TX", "Washington, DC", "LINCOLN MS SCIENCE", "Coach Bus Schedule",
                                   "Nurse Joy", "Dr. Seuss Week", "Unit 1 - 3 Digit Place Value"])
def test_own_layouts_keep_the_d34_exemptions(value: str) -> None:
    # schools, companies and places named after people stay (D34's exemptions), and so do state codes and words that
    # are no names after a title
    assert org_names.find(value) is None


def test_find_reads_a_long_description_in_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # a course catalog's 200-word description: the D34 check reads runs of a few words, not the whole text per step
    from datanonymous.privacy import headers

    seen: list[int] = []
    real = headers.person_name_hit
    monkeypatch.setattr(headers, "person_name_hit", lambda text: seen.append(len(text)) or real(text))
    filler = "students explore data with graphs models and reasoning in lab projects review writing".split()
    words = [filler[i % len(filler)] for i in range(200)]
    words[60:60] = ["with", "Mr.", "Okafor."]
    words[140:140] = ["Co-taught", "by", "Ms.", "Nguyen."]
    text = "Course 7731 " + " ".join(words)
    got = org_names.find(text)
    assert got is not None and list(got.parts) == ["Mr. Okafor", "Ms. Nguyen"]
    assert sum(seen) < 5 * len(text), (len(seen), sum(seen), len(text))


def _q1(tmp: Path) -> Path:
    """q1_roster.csv (sections and their teacher), q1_marks.csv (the same sections) and staff.csv (the teachers)."""
    sections = {"ALG1-03 Thompson, Robert J": ("Thompson", "Robert"), "ENG9-01 Johnson, Mary K": ("Johnson", "Mary"),
                "BIO-02 Nguyen, Linh T": ("Nguyen", "Linh"), "GEOM-02 Robert J Okonkwo": ("Okonkwo", "Robert"),
                "SPAN-03 Okafor, Chidi E.": ("Okafor", "Chidi"), "ART-01 Abernathy, RJ": ("Abernathy", "Rhea")}
    n = 180
    codes = cycle(list(sections), n)
    d = tmp / "in"
    write_csv(d / "q1_roster.csv", {"Student_ID": sids(n), "Section_Name": codes,
                                    "Teacher": [f"{sections[c][0]}, {sections[c][1]}" for c in codes]})
    write_csv(d / "q1_marks.csv", {"Student_ID": sids(n), "Section_Name": codes, "Q1": cycle(list("ABCDF"), n)})
    staff = sorted(set(sections.values()))
    write_csv(d / "staff.csv", {"Staff_ID": [f"T{900 + i}" for i in range(len(staff))],
                                "First_Name": [g for _, g in staff], "Last_Name": [s for s, _ in staff]})
    return d


def test_middle_initials_in_codes_and_a_staff_file(tmp_path: Path) -> None:
    # D36-MI: before, 'Thompson, Robert J' became 'Thompson, Little H' and the files with the teacher's name in a PII
    # column FAILed on the shared DATA_DICTIONARY; now every name is replaced and every file is written
    res, out, rep = run(_q1(tmp_path), tmp_path)
    assert res.exit_code == 0, res.output
    assert all((out / f).exists() for f in ("q1_roster.csv", "q1_marks.csv", "staff.csv"))
    blob = everything(out, rep, res.output)
    leaked = [w for w in ("thompson", "robert", "johnson", "nguyen", "linh", "okonkwo", "okafor", "chidi", "abernathy",
                          "rhea") if re.search(rf"\b{w}\b", blob)]
    assert not leaked, leaked
    marks, roster = column(out / "q1_marks.csv", "Section_Name"), column(out / "q1_roster.csv", "Section_Name")
    assert set(marks) == set(roster) and len(set(marks)) == 6
    assert {v.split(" ")[0] for v in marks} == {"ALG1-03", "ENG9-01", "BIO-02", "GEOM-02", "SPAN-03", "ART-01"}
    assert any(re.fullmatch(r"ART-01 [A-Z][a-z]+, [A-Z]{2}", v) for v in marks)


def test_long_section_names_keep_their_course_words(tmp_path: Path) -> None:
    # D36-PHRASE-FAIL: 6 or more course words before the teacher made the scanner match the kept course words of the
    # synthetic code against the real code (indexed whole) and FAIL the file; only the name parts are indexed now
    sections = ["English 9 Honors Period 3 - Ms. Garcia", "Algebra 1 Period 2 Room 114 Mr. Okafor",
                "Biology Period 5 Lab B - Mrs. Abernathy", "AP English Literature and Composition S1 - Ms. Nwosu"]
    n = 4 * 28
    src = write_csv(tmp_path / "in" / "q1_grades.csv", {"Student_Number": sids(n), "Course_Section": cycle(sections, n),
                                                        "Q1_Grade": cycle(list("ABCDF"), n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    written = column(out / "q1_grades.csv", "Course_Section")
    blob = everything(out, rep, res.output)
    assert not any(re.search(rf"\b{w}\b", blob) for w in ("garcia", "okafor", "abernathy", "nwosu"))
    for real in sections:
        course = real.rsplit(" ", 1)[0]  # everything but the surname stays real
        assert sum(1 for v in written if v.startswith(course + " ")) == 28, course
    run_ = pipeline.execute(Settings(mode="fidelity", seed=1), src.parent, tmp_path / "o2", tmp_path / "r2")
    scanner = Scanner(SensitiveIndex.build(run_.files), build_allow(run_))
    assert scanner.hits_in_text("room of Ms. Garcia") and scanner.hits_in_text("Mrs. Abernathy")
    assert not scanner.hits_in_text("English 9 Honors Period 3 - Ms. Quill")


def test_dash_labels_are_no_address_outside_org_codes(tmp_path: Path) -> None:
    # K09-DASH-FP: a wide gradebook's assignment headers and a long gradebook's labels held by 60 students are no
    # street address ('3 Digit Place', '3 Milky Way'): the headers pass the D34 screen and the labels stay real
    labels = ["Unit 1 - 3 Digit Place Value", "Unit 4 - 2 Oregon Trail Project", "Science - 3 Milky Way",
              "Spelling Test 4"]
    n = 60
    wide: dict[str, list[Any]] = {"Student_ID": sids(n), "Grade_Level": ["03"] * n}
    wide.update({lab: cycle(list(range(55, 100)), n) for lab in labels})
    write_csv(tmp_path / "wide" / "gradebook_wide.csv", wide)
    res, out, _ = run(tmp_path / "wide", tmp_path, name="wide")
    assert res.exit_code == 0, res.output
    assert list(read_dicts(out / "gradebook_wide.csv")[0]) == list(wide)
    write_csv(tmp_path / "long" / "gradebook_long.csv", {"Student_ID": sids(n) * 4,
                                                         "Assignment": [lab for lab in labels for _ in range(n)],
                                                         "Score": cycle(list(range(55, 100)), 4 * n)})
    res, out, _ = run(tmp_path / "long", tmp_path, name="long")
    assert res.exit_code == 0, res.output
    assert Counter(column(out / "gradebook_long.csv", "Assignment")) == {lab: n for lab in labels}


def test_a_code_name_is_replaced_in_every_column_that_holds_it(tmp_path: Path) -> None:
    # D36-CROSSCOL: the section codes of gradebook.csv also sit under Class (a comma list to the detector), Homeroom
    # and Advisory of schedule.csv, kept there by D35 for their counts: the names are replaced with the same fake
    # names everywhere, so Class still joins Section_Name and no column keeps a teacher's name
    codes = ["ALG1-03 Garcia, Ana", "BIO-02 Okafor, Chidi", "ENG9-01 Brennan, Siobhan", "HIST-04 Ms. Abernathy"]
    names = [c.split(" ", 1)[1] for c in codes]
    n = 100
    sec = cycle(codes, n)
    d = tmp_path / "in"
    write_csv(d / "gradebook.csv", {"Student_ID": sids(n), "Section_Name": sec, "Score": cycle(list(range(50, 99)), n)})
    write_csv(d / "schedule.csv", {"Student_ID": sids(n), "Class": sec, "Homeroom": cycle(names, n),
                                   "Advisory": ["ADV " + x for x in cycle(names, n)],
                                   "Period": cycle(["1", "2", "3"], n)})
    res, out, rep = run(d, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    assert not any(re.search(rf"\b{w}\b", blob) for w in ("garcia", "ana", "okafor", "chidi", "brennan", "siobhan",
                                                          "abernathy"))
    section = Counter(column(out / "gradebook.csv", "Section_Name"))
    assert Counter(column(out / "schedule.csv", "Class")) == section and sorted(section.values()) == [25] * 4
    fakes = {v.split(" ", 1)[1] for v in section}
    assert set(column(out / "schedule.csv", "Homeroom")) == fakes
    assert set(column(out / "schedule.csv", "Advisory")) == {"ADV " + f for f in fakes}
    col = report_column(rep, "schedule.csv", "Homeroom")  # a comma list too: 'Garcia', 'Ana', ... 'Ms. Abernathy'
    assert col["params"]["org_code_names_replaced"] == 7 and "org_code_names" in {m["code"] for m in col["messages"]}


def test_spread_to_a_factor_keeps_its_other_words() -> None:
    # a level kept for its count under a neutral header ('Room 4 - Ms. Abernathy' under Homeroom) is written like
    # the code that names the same teacher: the same fake surname, the rest real, its exact count
    tp, cols = run_table({"student_id": sids(60), "Section": cycle(["ALG1 Ms. Abernathy", "GEO-01"], 60),
                          "Homeroom": cycle(["Room 4 - Ms. Abernathy", "Lab"], 60)})
    p = profile_of(tp, "Homeroom")
    assert p.type.value == "FACTOR" and p.params["org_code_names_replaced"] == 1
    fake = next(v for v in cols["Section"] if v != "GEO-01").removeprefix("ALG1 ")
    assert Counter(cols["Homeroom"]) == {f"Room 4 - {fake}": 30, "Lab": 30} and "Abernathy" not in fake


def test_spread_reads_a_list_value_whole() -> None:
    # the comma of 'Garcia, Ana' is the list's delimiter: the name's pieces land in two items, both replaced
    tp, cols = run_table({"student_id": sids(60), "Section": cycle(["ALG1 Garcia, Ana", "GEO-01"], 60),
                          "Clubs": cycle(["ALG1 Garcia, Ana", "Chess, Band", "Chess"], 60)})
    p = profile_of(tp, "Clubs")
    assert p.type.value == "MULTI_SELECT"
    assert p.params[dtext.ORG_NAMES_KEY] == {"ALG1 Garcia": [[5, 11, "last"]], "Ana": [[0, 3, "first"]]}
    assert {"Chess", "Band"}.isdisjoint(dtext.pseudonymized(p))
    assert set(cols["Clubs"]) & set(cols["Section"]) and not {"ALG1 Garcia, Ana"} & set(cols["Clubs"])


def test_a_rare_combination_never_keeps_a_named_item(tmp_path: Path) -> None:
    # K-11 x D36: under Clubs (written as combinations) the rare 'ALG1 Garcia, Ana, Chess' keeps the common
    # sub-combination 'ALG1 Garcia, Ana' real (COMBO_KEEP_KEY); D36 then names those items run-wide, so the rare
    # combination writes them with the code's fake name too, never the teacher's real name
    clubs = (["ALG1 Garcia, Ana"] * 40 + ["Chess"] * 30 + ["Band"] * 30 + ["Chess, Band"] * 15
             + ["ALG1 Garcia, Ana, Chess"] * 3)
    n = len(clubs)
    d = tmp_path / "in"
    write_csv(d / "clubs.csv", {"Student_ID": sids(n), "Section": cycle(["ALG1 Garcia, Ana", "GEO-01", "BIO-2"], n),
                                "Clubs": clubs})
    res, out, rep = run(d, tmp_path)
    assert res.exit_code == 0, res.output
    assert file_page(rep, "clubs.csv")["checks"]["leak_scan"]["result"] == "PASS"
    assert not any(re.search(rf"\b{w}\b", everything(out, rep, res.output)) for w in ("garcia", "ana"))
    fake = next(v for v in column(out / "clubs.csv", "Section") if v.startswith("ALG1 "))
    written = Counter(column(out / "clubs.csv", "Clubs"))
    assert written[fake] == 40 and sum(c for v, c in written.items() if v.startswith(fake + ", ")) == 3


def test_excel_zero_date_placeholders_keep_their_counts(tmp_path: Path) -> None:
    # K07-PLACEHOLDER-COUNTS: 12/30/1899 (Excel's day 0) as 'unknown' in a birth-date column of 400 students was an
    # ordinary date after K-07 (DOB columns never get far sentinels): 33 rows of it and invented 1984 birth dates came
    # out. Excel's day 0 is a fixed sentinel now, written at its exact count in any column
    n = 400
    dob = [f"{1 + i % 12:02d}/{1 + i % 28:02d}/{2008 + i % 5}" for i in range(n)]
    for i in range(34):
        dob[i * 11] = "12/30/1899"
    src = write_csv(tmp_path / "in" / "students.csv", {"Student_ID": sids(n), "Birthdate": dob,
                                                       "Grade": cycle(["06", "07", "08"], n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    written = column(out / "students.csv", "Birthdate")
    assert written.count("12/30/1899") == 34
    assert min(int(v[-4:]) for v in written if v != "12/30/1899") == 2008
    assert report_column(rep, "students.csv", "Birthdate")["params"]["sentinels"] == {"12/30/1899": 34}


@pytest.mark.parametrize("value", ["1899-12-30", "1899-12-31", "1904-01-01", "1753-01-01", "0001-01-01"])
def test_date_system_zero_days_are_fixed_sentinels(value: str) -> None:
    # one row is enough: no person's real date is a date system's zero day (K-07: fixed sentinels stay unconditional)
    det = detect(_dates(300) + [value], "birth_date")
    assert det.params["sentinels"] == {value: 1}


def test_a_real_epoch_date_is_no_fixed_sentinel() -> None:
    # 1970-01-01 (Unix) is somebody's birth date: in a staff file it is an ordinary date
    det = detect([f"{1960 + i % 30}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(300)] + ["1970-01-01"] * 3,
                 "birth_date")
    assert "sentinels" not in det.params


# =========================================================================== hardening 2: the second attack pass
#: The 50 most common US surnames (Census 2010 order): many are common English words or everyday surnames that the D34
#: pair check does not pair ('Miller', 'Brown', 'Young').
TOP_SURNAMES = ("Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
                "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis",
                "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
                "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter",
                "Roberts")


# D36-COMMON-SURNAME: a known first + last pair and 'Surname, Initial' with a common surname
@pytest.mark.parametrize("value,replaced", [
    ("ALG1-03 Lisa Miller", [("Lisa", "first"), ("Miller", "last")]),
    ("BIO-02 John Taylor", [("John", "first"), ("Taylor", "last")]),
    ("ENG9-01 Mark Brown", [("Mark", "first"), ("Brown", "last")]),  # both everyday words: two common US names
    ("GEOM-02 David Young", [("David", "first"), ("Young", "last")]),
    ("CHEM-01 Miller, L", [("Miller", "last"), ("L", "initial")]),
    ("PE-02 Jackson, E", [("Jackson", "last"), ("E", "initial")]),
    ("Period 3 Taylor, J", [("Taylor", "last"), ("J", "initial")]),
    ("SPED Lisa Miller/John Taylor", [("Lisa", "first"), ("Miller", "last"), ("John", "first"), ("Taylor", "last")]),
    ("ALG1-03-lisa.miller", [("lisa", "first"), ("miller", "last")]),  # a user name
    ("ALG1-03 MILLER LISA", [("MILLER", "first"), ("LISA", "last")]),  # surname first, no comma (roles by position)
])
def test_find_reads_common_surnames(value: str, replaced: list[tuple[str, str]]) -> None:
    got = org_names.find(value)
    assert got is not None and got.tokens is not None
    assert [(value[a:b], role) for a, b, role in got.tokens] == replaced


def test_find_reads_the_most_common_us_surnames() -> None:
    missed = [layout.format(s) for s in TOP_SURNAMES
              for layout in ("ALG1-03 Lisa {}", "ALG1-03 {}, L", "ALG1-03 {}, Lisa", "ALG1-03 Ms. {}")
              if (got := org_names.find(layout.format(s))) is None or not got.tokens
              or s not in [layout.format(s)[a:b] for a, b, _ in got.tokens]]
    assert not missed, missed


# D36-COMMA-SHORT-SURNAMES: 'Surname, Given' with two-letter, apostrophe and hyphenated names
@pytest.mark.parametrize("value,pieces", [
    ("ALG1-03 Le, Thanh", ["Le", "Thanh"]), ("BIO-02 Wu, Lin", ["Wu", "Lin"]),
    ("ENG9-01 D'Souza, Priya", ["D'Souza", "Priya"]), ("HIST-04 Park, Ji-hoon", ["Park", "Ji", "hoon"]),
    ("CHEM-01 Ng, Wei", ["Ng", "Wei"]), ("PE-02 Vo, Minh", ["Vo", "Minh"]),
    ("ART-01 D'Angelo, Maria", ["D'Angelo", "Maria"]),
    ("GEOM-02 Kim, Ji-hoon", ["Kim", "Ji", "hoon"]), ("ALG1-03 LE, THANH", ["LE", "THANH"]), ("NG, WEI", ["NG", "WEI"]),
    ("Sec 2 Park, Min", ["Park", "Min"]), ("Sec 2 West, Ana", ["West", "Ana"]),
    ("ESL Ji-hoon Kim", ["Ji", "hoon", "Kim"]),
    ("Sec 4 Do, Lan", ["Do", "Lan"]), ("Sec 4 Yoo, Jin", ["Yoo", "Jin"]), ("Sec 5 Ng, W.", ["Ng", "W"]),
    ("Math 7 - Priya D'Souza", ["Priya", "D'Souza"]), ("Math 7 - Fiona McAllister", ["Fiona", "Mc", "Allister"]),
])
def test_find_reads_short_and_apostrophe_surnames(value: str, pieces: list[str]) -> None:
    got = org_names.find(value)
    assert got is not None and got.tokens is not None
    assert [value[a:b] for a, b, _ in got.tokens] == pieces


# D36-FALSE-POSITIVE-COURSES: 'MS' / 'Ms' without a period is also a middle school
@pytest.mark.parametrize("value", [
    "MS HOPE SQUAD", "MS PEER TUTOR", "MS MARINE SCIENCE", "MS CHINA STUDIES", "MS BAND", "MS JUSTICE CIVICS",
    "Ms Band", "Ms Choir", "Ms Art", "Ms Science 7", "Ms Math 6", "MS KEY CLUB", "MS GREEN TEAM", "MS CROSS COUNTRY",
    "Ms Robotics", "MS PE", "Ms Of Mice and Men", "Ms Martin Luther King Day", "Peer Helpers", "Peer Leadership",
    # comma and pair layouts keep course words, schools, places, companies and namesakes
    "Biology, Honors", "French, Art", "French, I", "English, AP", "Reading, Math", "Algebra, I", "Lunch, A",
    "Smith, Inc.", "Garcia, PhD", "Lincoln, HS", "Houston, Texas", "Smith Hall", "Math Block", "Media Law",
    "Art History", "Jazz Band", "Mark Twain Elementary", "John Taylor Elementary", "Harriet Tubman Middle",
])
def test_course_titles_keep_their_words(value: str) -> None:
    assert org_names.find(value) is None


@pytest.mark.parametrize("value,pieces", [
    ("Ms Garcia", ["Garcia"]), ("MS GARCIA", ["GARCIA"]), ("Ms Le", ["Le"]), ("Ms Ng", ["Ng"]),
    ("ENG 9 Ms Brown", ["Brown"]), ("ENG 9 Ms Brown 7", ["Brown"]), ("Ms. Band", ["Band"]),  # an honorific's period
    ("Mr Band", ["Band"]), ("MS HOPE", ["HOPE"]), ("ALG1 Ms Lisa Brown", ["Lisa", "Brown"]),
])
def test_an_honorific_ms_still_names_a_teacher(value: str, pieces: list[str]) -> None:
    got = org_names.find(value)
    assert got is not None and got.tokens is not None
    assert [value[a:b] for a, b, _ in got.tokens] == pieces


def test_middle_school_courses_stay_real(tmp_path: Path) -> None:
    # the attacker's repro (fp2): 11 middle-school course titles held by 33-48 students each are org codes kept real
    # (D10); before, every one was rewritten with fake surnames ('MS HOPE SQUAD' -> 'MS VANESSA SNYDER')
    titles = ["MS HOPE SQUAD", "MS PEER TUTOR", "MS MARINE SCIENCE", "MS CHINA STUDIES", "MS BAND", "MS JUSTICE CIVICS",
              "Ms Band", "Ms Choir", "Ms Art", "Ms Science 7", "Ms Math 6", "ALG1 Ms Garcia"]
    n = 12 * 30
    src = write_csv(tmp_path / "in" / "course_marks.csv", {"Student_ID": sids(n), "Course_Title": cycle(titles, n),
                                                           "Mark": cycle(list("ABCDF"), n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    written = Counter(column(out / "course_marks.csv", "Course_Title"))
    assert {t: written[t] for t in titles[:-1]} == {t: 30 for t in titles[:-1]}
    assert "ALG1 Ms Garcia" not in written and "garcia" not in everything(out, rep, res.output)
    assert report_column(rep, "course_marks.csv", "Course_Title")["params"]["org_code_names_replaced"] == 1


def _staffed(tmp: Path) -> Path:
    """The attacker's repro (gl2): roster.csv and sections.csv share section names with teachers' common names;
    staff.csv holds the same teachers in PERSON_NAME columns."""
    sections = [("ALG1-03 Lisa Miller", "Miller", "Lisa"), ("BIO-02 John Taylor", "Taylor", "John"),
                ("ENG9-01 Mark Brown", "Brown", "Mark"), ("HIST-04 Karen Wilson", "Wilson", "Karen"),
                ("CHEM-01 Miller, L", "Miller", "Linda"), ("PE-02 Jackson, E", "Jackson", "Eric"),
                ("ART-01 Susan Martin", "Martin", "Susan"), ("GEOM-02 David Young", "Young", "David")]
    n = 8 * 25
    d = tmp / "in"
    codes = cycle([s for s, _, _ in sections], n)
    write_csv(d / "roster.csv", {"Student_ID": sids(n), "Section_Name": codes, "Mark": cycle(list("ABCDF"), n)})
    write_csv(d / "sections.csv", {"Section_ID": [f"S{i:03d}" for i in range(8)],
                                   "Section_Name": [s for s, _, _ in sections],
                                   "Room": [str(100 + i) for i in range(8)]})
    write_csv(d / "staff.csv", {"Staff_ID": [f"T{900 + i}" for i in range(8)],
                                "Teacher_Last": [last for _, last, _ in sections],
                                "Teacher_First": [first for _, _, first in sections]})
    return d


def test_common_surnames_in_section_names(tmp_path: Path) -> None:
    res, out, rep = run(_staffed(tmp_path), tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    real = ("lisa miller", "john taylor", "mark brown", "karen wilson", "miller, l", "jackson, e", "susan martin",
            "david young")
    assert not [x for x in real if x in blob]
    roster, sections = column(out / "roster.csv", "Section_Name"), column(out / "sections.csv", "Section_Name")
    assert set(roster) == set(sections) and sorted(Counter(roster).values()) == [25] * 8  # joins and counts hold
    assert {v.split(" ")[0] for v in sections} == {"ALG1-03", "BIO-02", "ENG9-01", "HIST-04", "CHEM-01", "PE-02",
                                                    "ART-01", "GEOM-02"}
    # one fake per real word: 'Miller' of 'Lisa Miller' and of 'Miller, L' is the same fake surname
    by = {v.split(" ")[0]: v.split(" ", 1)[1] for v in sections}
    assert by["CHEM-01"].split(",")[0] == by["ALG1-03"].split(" ")[1]
    assert report_column(rep, "roster.csv", "Section_Name")["params"]["org_code_names_replaced"] == 8
    assert file_page(rep, "roster.csv")["checks"]["leak_scan"]["result"] == "PASS"


def test_short_comma_surnames_in_codes(tmp_path: Path) -> None:
    # the attacker's repro (cm1): 'Surname, Given' with two-letter, apostrophe and hyphenated names
    codes = ["ALG1-03 Le, Thanh", "BIO-02 Wu, Lin", "ENG9-01 D'Souza, Priya", "HIST-04 Park, Ji-hoon",
             "CHEM-01 Ng, Wei", "PE-02 Vo, Minh", "ART-01 D'Angelo, Maria", "GEOM-02 Kim, Ji-hoon"]
    n = 8 * 25
    src = write_csv(tmp_path / "in" / "grades.csv", {"Student_ID": sids(n), "Course_Section": cycle(codes, n),
                                                     "Term": ["S1"] * n, "Mark": cycle(list("ABCDF"), n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    assert not [c for c in codes if c.split(" ", 1)[1].casefold() in blob]
    written = column(out / "grades.csv", "Course_Section")
    assert sorted(Counter(written).values()) == [25] * 8
    assert {v.split(" ")[0] for v in written} == {c.split(" ")[0] for c in codes}
    hoon = [v.split(" ", 1)[1].split(", ")[1] for v in set(written) if v.startswith(("HIST-04", "GEOM-02"))]
    assert len(set(hoon)) == 1 and "-" in hoon[0]  # 'Ji-hoon' is one fake hyphenated given name in both codes


def test_find_the_same_person_in_another_layout() -> None:
    # D36-CROSSCOL-PARTIAL: the code holds the longer form ('Thompson, Robert J'), another column the shorter one, or
    # the same name in another order or with an initial: the spread replaces it with the same fake words
    run_parts = dtext._RunParts(["Thompson, Robert J", "Garcia, Ana", "Ms. Abernathy"])
    for text, pieces in (("Thompson, Robert", ["Thompson", "Robert"]), ("Robert Thompson", ["Robert", "Thompson"]),
                         ("HR R. Thompson", ["R", "Thompson"]), ("Ana Garcia", ["Ana", "Garcia"]),
                         ("GARCIA ANA", ["GARCIA", "ANA"]), ("Garcia A", ["Garcia", "A"]),
                         ("Room 4 - Ms. Garcia", ["Garcia"]), ("Room 4 - Ms. Abernathy", ["Abernathy"])):
        got = run_parts.pieces(text)
        assert got is not None, text
        placed, unplaced, found = got
        assert not unplaced and [text[a:b] for a, b, _ in placed] == pieces, text
    for text in ("Thompson", "Homeroom Garcia", "Ana Lopez", "Abernathy Hall"):  # a bare surname: the residual
        got = run_parts.pieces(text)
        assert got is None or not got[0], text


def test_a_longer_name_in_the_code_is_replaced_in_other_columns(tmp_path: Path) -> None:
    # the attacker's repro (mf2, mf1): Homeroom and Advisory hold the teacher's shorter form, another order or an
    # initial; before, they were written verbatim with a fake code beside them (a 1:1 key to the real name)
    rows = [("ALG1-03 Thompson, Robert J", "Thompson, Robert", "R. Thompson"),
            ("BIO-02 Okafor, Chidi A", "Chidi Okafor", "Okafor C"),
            ("ENG9-01 Nwosu, Adaeze M.", "NWOSU ADAEZE", "Nwosu, A."),
            ("HIST-04 Hendricks, Paula Jean", "Paula Hendricks", "Hendricks P"),
            ("CHEM-01 Garcia Lopez, Ana", "Garcia, Ana", "Ana Garcia Lopez")]
    n = 5 * 25
    picked = cycle(rows, n)
    src = write_csv(tmp_path / "in" / "enrollments.csv", {
        "Student_ID": sids(n), "Section_Name": [r[0] for r in picked], "Homeroom": [r[1] for r in picked],
        "Advisory": [r[2] for r in picked], "Grade": cycle(["09", "10", "11", "12"], n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, res.output)
    assert not [w for w in ("thompson", "okafor", "chidi", "nwosu", "adaeze", "hendricks", "paula", "garcia")
                if re.search(rf"\b{w}\b", blob)]
    section = Counter(column(out / "enrollments.csv", "Section_Name"))
    assert sorted(section.values()) == [25] * 5
    fakes = {re.findall(r"[A-Za-z]+", v.split(" ", 1)[1])[0].casefold() for v in section}  # the fake surnames
    for name in ("Homeroom", "Advisory"):  # each teacher's other forms carry the code's fake surname, exact counts
        written = Counter(column(out / "enrollments.csv", name))
        assert sorted(written.values()) == [25] * 5, name
        assert all(any(f in re.findall(r"[a-z]+", v.casefold()) for f in fakes) for v in written), (name, written)


def test_accented_names_are_located_and_spread(tmp_path: Path) -> None:
    # the attacker's repro (ac1): folding that changes the length ('ß', 'Æ', NFD accents) left the code's name part
    # unknown, so Homeroom wrote 'Groß, Anna' verbatim; the name is located letter by letter now
    nfd = unicodedata.normalize("NFD", "Núñez, José")
    rows = [("ALG1-03 Groß, Anna", "Groß, Anna"), ("BIO-02 " + nfd, nfd),
            ("ENG9-01 Æbelø, Søren", "Æbelø, Søren")]
    n = 3 * 25
    picked = cycle(rows, n)
    src = write_csv(tmp_path / "in" / "enrollments.csv", {"Student_ID": sids(n),
                                                          "Section_Name": [r[0] for r in picked],
                                                          "Homeroom": [r[1] for r in picked]})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    blob = unicodedata.normalize("NFC", everything(out, rep, res.output))
    assert not [w for w in ("groß", "gross", "núñez", "josé", "æbelø", "søren", "anna") if w in blob]
    got = read_dicts(out / "enrollments.csv")
    assert {r["Section_Name"].split(" ")[0] for r in got} == {"ALG1-03", "BIO-02", "ENG9-01"}
    assert all(r["Section_Name"].split(" ", 1)[1] == r["Homeroom"] for r in got)


def test_the_scanner_indexes_both_orders_of_a_code_name(tmp_path: Path) -> None:
    rows = ["ALG1-03 Thompson, Robert J", "BIO-02 Lisa Miller", "GEO-01"]
    n = 60
    src = write_csv(tmp_path / "in" / "s.csv", {"Student_ID": sids(n), "Section": cycle(rows, n)})
    run_ = pipeline.execute(Settings(mode="fidelity", seed=1), src.parent, tmp_path / "out", tmp_path / "rep")
    assert pipeline.exit_code(run_) == 0
    scanner = Scanner(SensitiveIndex.build(run_.files), build_allow(run_))
    for text in ("taught by Robert Thompson", "Thompson Robert", "Miller, Lisa", "Lisa Miller"):
        assert scanner.hits_in_text(text), text
    assert not scanner.hits_in_text("Thompson") and not scanner.hits_in_text("GEO-01")


# K07-DAY0-TIME-OF-DAY: a zero day with a time of day is a bare time, no placeholder
@pytest.mark.parametrize("values", [
    [f"1899-12-30 {8 + i % 2:02d}:{i % 60:02d}:{(7 * i) % 60:02d}" for i in range(300)],
    [f"12/30/1899 {8 + i % 2}:{i % 60:02d}:{(7 * i) % 60:02d} AM" for i in range(300)],
    [f"1900-01-01 {8 + i % 2:02d}:{i % 60:02d}:00.000" for i in range(300)],  # SQL Server's time cast to datetime
])
def test_a_zero_day_with_a_time_is_a_bare_time(values: list[str]) -> None:
    det = detect(values, "Time_In")
    assert det.type.value == "DATETIME" and "sentinels" not in det.params


def test_zero_day_placeholders_stay_sentinels_as_dates() -> None:
    times = [f"2025-09-{1 + i % 28:02d} {8 + i % 9:02d}:{i % 60:02d}:00" for i in range(300)]
    det = detect(times + ["1899-12-30 00:00:00"] * 3 + ["9999-12-31 23:59:59"] * 2, "Check_In")
    assert det.params["sentinels"] == {"1899-12-30 00:00:00": 3, "9999-12-31 23:59:59": 2}


@pytest.mark.parametrize("fmt", ["parquet", "feather"])
def test_typed_time_of_day_columns_are_written(tmp_path: Path, fmt: str) -> None:
    # the attacker's repro: a nurse log whose Time_In is a typed timestamp on 1899-12-30 FAILed with exit 3 ('no
    # date/time generator accepted the column'); it is a datetime column again, written on day 0
    import datetime as dt

    import pandas as pd

    n = 200
    df = pd.DataFrame({"Student_ID": [500100 + i % 80 for i in range(n)],
                       "Time_In": [pd.Timestamp(dt.datetime(1899, 12, 30, 7, 50) + dt.timedelta(minutes=i % 95,
                                                                                               seconds=i % 60))
                                   for i in range(n)],
                       "Reason": cycle(["Headache", "Injury", "Medication"], n)})
    d = tmp_path / "in"
    d.mkdir(parents=True)
    if fmt == "parquet":
        df.to_parquet(d / "nurse.parquet", index=False)
    else:
        df.to_feather(d / "nurse.feather")
    res, out, _ = run(d, tmp_path)
    assert res.exit_code == 0, res.output
    back = pd.read_parquet(out / "nurse.parquet") if fmt == "parquet" else pd.read_feather(out / "nurse.feather")
    assert str(back["Time_In"].dtype).startswith("datetime64") and len(back) == n
    assert {t.date() for t in back["Time_In"]} == {dt.date(1899, 12, 30)}
