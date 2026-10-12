"""D35 (user decision, 2026-10-05): real text values are written verbatim only when provably safe.

Every path that would copy a real text value (factor, constant and flag levels, ordinal levels outside the
built-in vocabularies, multi-select items, native dictionary entries and Stata value labels, tokens among
numbers, NA tokens) goes through ONE decision, the copy gate's allowlist (``detect.text.unit_kept``):

* kept: booleans and numbers (their own rules), organisational codes (D10), the built-in vocabularies
  (``detect.vocab``: the standard education vocabulary, ordinal scales, boolean words, NA and suppression tokens),
  whatever their count; any other value held by >= ``--min-cell`` distinct entities with no personal-data shape
  (hardening pass 1: the minimum over every ID column of the table, a blank ID being no new person; the subject's
  names or e-mail addresses when the only IDs are record IDs; rows without an ID, a record repeated in every column
  counting once; pooled per column name over the run, a table with person IDs outweighing one without; hardening
  pass 2: teacher, staff, section and record IDs never prove people, an ID column turned into something else by a
  flag still counts, and a code table pools with the column that uses its codes under another name; hardening
  pass 3: the names of the row's own person always count, only an ID of the row's own person or a name under a
  student header proves people (never ``Entered_By``, ``UserName``, ``Provider_Name`` or a bare ``Name``), a
  student's record keys (``Student_Test_ID``) number records, a blank student ID never takes a record number, and
  a count without a key of the row's own person gets a warning);
  ``--force-factor COL`` (``--force-type`` and ``--org-code`` only name the type);
* never kept: PII-hinted columns (D12) and values with an e-mail, phone, SSN, URL or street-address shape;
* everything else gets a same-shape pseudonym with its exact count (``generate.pseudonyms``), one per value for the
  whole run, deterministic for a given ``--seed``, never equal to a real value of the run nor a vocabulary word;
* the leak scanner excuses exactly the kept values and indexes every pseudonymised one; the report gives counts
  only (``pseudonymized_levels``) and shows a pseudonymised level as ``[pseudonymized N]``; the docs list the
  pseudonyms (synthetic labels).

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
import pytest

from conftest import EXAMPLES, read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.detect import vocab
from datanonymous.generate import pseudonyms
from datanonymous.generate import text as gtext
from datanonymous.model import NEVER_COPY_TYPES
from datanonymous.model import ColumnType as T
from datanonymous.model import SyntheticColumn, SyntheticTable
from datanonymous.privacy import leaks

from test_area_a_support import profile_of, run_table

N = 120


def spread(spec: list[tuple[str, int]]) -> list[str]:
    """The values of ``spec`` (value, count) interleaved deterministically (no sort order to reproduce)."""
    out = [v for v, c in spec for _ in range(c)]
    assert len(out) == N, len(out)
    random.Random(7).shuffle(out)
    return out


#: Dismissal: frequent labels, a rare name, a rare sentence and a rare code (no ID column: rows count)
DISMISSAL = [("Bus", 60), ("Car Rider", 40), ("Aftercare Club", 12), ("John Smith", 3),
             ("Picked up by his grandmother after the second bell", 2), ("QX-17", 3)]
RARE_DISMISSAL = ("John Smith", "Picked up by his grandmother after the second bell", "QX-17")
RACE = [("White", 70), ("Hispanic or Latino", 29), ("Two or More Races", 9), ("Pacific Islander", 1),
        ("Black", 10), ("NA", 1)]
INFO = [("Front Office", 50), ("ana.lopez@school.org", 40), ("312-555-0187", 30)]
SCHOOL = [("Lincoln Elementary", 118), ("Smith Academy", 2)]
TEACHER = [("Ms. Garcia", 60), ("Mr. Lee", 60)]
STATUS = [("Enrolled", 100), ("Withdrawn", 1), ("Zebra Club", 19)]


def base_columns() -> dict[str, list[str]]:
    return {"Dismissal": spread(DISMISSAL), "Race": spread(RACE), "Info": spread(INFO),
            "School_Name": spread(SCHOOL), "Teacher": spread(TEACHER), "Status": spread(STATUS),
            "Score": [str(i * 3) for i in range(N)]}  # every row differs: rows count


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


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    data = json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
    for t in data["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def leak_check(rep: Path, name: str) -> str:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))["checks"]["leak_scan"]["result"]


def others(counts: Counter, kept: set[str]) -> Counter:
    return Counter({v: c for v, c in counts.items() if v not in kept})


def shape(s: str) -> str:
    """Character-class shape: letters by case, digits, everything else as itself."""
    return "".join("A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch for ch in s)


# =========================================================================== the vocabulary
def test_vocabulary_is_data_not_tool_text() -> None:
    assert len(vocab.STANDARD_EDUCATION_VOCABULARY) > 50  # one collection the leak scanner skips as data
    tool = leaks.tool_vocabulary()
    for word in ("kinyarwanda", "caucasian", "multiracial", "mckinney", "homebound", "marshallese"):
        assert vocab.is_standard_word(word) or any(word in vocab.vocab_key(w) for w in
                                                   vocab.STANDARD_EDUCATION_VOCABULARY)
        assert word not in tool, word


@pytest.mark.parametrize("value", [
    "Free / Reduced", "free/reduced", "F/R/P", "Paid", "pre k", "PreK", "KG", "k", "GRADE 9", "9th grade", "09", "12",
    "Ninth Grade", "Q3", "t2", "S1", "Semester 2", "Fall", "spring", "Title I", "title 1", "504 Plan", "IEP", "SPED",
    "LEP", "fep", "RFEP", "ELL", "Y", "no", "TRUE", "Two or more races", "American Indian/Alaska Native",
    "Hispanic or Latino", "Nonbinary", "N/A", "NA", "missing", "Unknown", "*", "<10", "n<10", "Tardy",
    "Excused Absence", "Withdrawn", "Transferred Out", "Dual Enrollment", "CTE", "Gifted", "GT", "AP", "IB",
    "Strongly Agree", "Below Basic", "B+", "2023-24", "Group C",
])
def test_vocabulary_words_are_kept_whatever_their_case_and_spacing(value: str) -> None:
    assert vocab.is_vocabulary(value)
    assert dtext.unit_kept(value, holders=1, min_cell=10)


@pytest.mark.parametrize("value", ["John Smith", "Pwani", "Zeta Cohort", "Ed", "Karen", "Walker", "June",
                                   "Robotics Club", "QX-17", "Rm 101"])
def test_other_values_are_not_vocabulary(value: str) -> None:
    assert not vocab.is_vocabulary(value)
    assert not dtext.unit_kept(value, holders=9, min_cell=10)
    assert dtext.unit_kept(value, holders=10, min_cell=10)


@pytest.mark.parametrize("value", ["ana@school.org", "Call 312-555-0187", "123-45-6789", "www.district.org/x",
                                   "12 Oak St"])
def test_personal_data_shapes_are_never_kept(value: str) -> None:
    assert not dtext.unit_kept(value, holders=10_000, min_cell=10)
    assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=True) or value in (
        "www.district.org/x", "12 Oak St")  # D10: a building's address or website in an org code is kept


def test_unit_key_is_the_leak_scanners_comparison_form() -> None:
    for s in ("Rift  Valley", "O’Brien", " CASE ", "Smith‐Jones", "x\ty"):
        assert dtext.unit_key(s) == leaks.norm_text(s)


# =========================================================================== entity counting
def test_entity_counting_with_and_without_an_id() -> None:
    sid = [f"S{1000 + i // 4}" for i in range(N)]  # 30 students x 4 rows
    club = ["Debate Society" if i < 12 else "Chess Guild" if i < 68 else "Drama Guild" for i in range(N)]
    with_id, cols = run_table({"student_id": sid, "Club": club})
    p = profile_of(with_id, "Club")
    assert p.role == "entity"
    assert set(dtext.pseudonymized(p)) == {"Debate Society"}  # 12 rows but only 3 students
    out = Counter(cols["Club"].tolist())
    assert out["Chess Guild"] == 56 and out["Drama Guild"] == 52 and "Debate Society" not in out
    assert sorted(others(out, {"Chess Guild", "Drama Guild"}).values()) == [12]
    holders = dtext.unit_holders(np.asarray(club, dtype=object), p, with_id.private.entity_index)
    assert holders == {"debate society": 3, "chess guild": 14, "drama guild": 13}
    # the same values without an ID column: rows count, so 12 rows keep the real value
    no_id, cols2 = run_table({"Club": club})
    assert dtext.pseudonymized(profile_of(no_id, "Club")) == {}
    assert Counter(cols2["Club"].tolist())["Debate Society"] == 12


def test_multi_select_items_are_counted_once_per_entity() -> None:
    sid = [f"S{1000 + i // 2}" for i in range(N)]
    vals = ["ELL;Robotics" if i < 24 else "ELL" if i % 2 else "SPED;ELL" for i in range(N)]
    tp, _ = run_table({"student_id": sid, "Services": vals})
    p = profile_of(tp, "Services")
    assert p.type is T.MULTI_SELECT
    assert dtext.unit_holders(np.asarray(vals, dtype=object), p, tp.private.entity_index)["robotics"] == 12
    assert dtext.unit_holders(np.asarray(vals, dtype=object), p, None)["robotics"] == 24


# =========================================================================== levels (run_table)
def test_rare_values_get_same_shape_pseudonyms_with_exact_counts() -> None:
    tp, cols = run_table(base_columns())
    p = profile_of(tp, "Dismissal")
    assert p.type is T.FACTOR and set(dtext.pseudonymized(p)) == set(RARE_DISMISSAL)
    assert p.params["pseudonymized_levels"] == 3 and p.flags["pseudonymized_levels"]
    out = Counter(cols["Dismissal"].tolist())
    kept = {"Bus", "Car Rider", "Aftercare Club"}
    assert {v: out[v] for v in kept} == {"Bus": 60, "Car Rider": 40, "Aftercare Club": 12}
    fakes = others(out, kept)
    assert not set(RARE_DISMISSAL) & set(out) and sorted(fakes.values()) == [2, 3, 3]
    want = {shape(v): c for v, c in DISMISSAL if v in RARE_DISMISSAL}
    assert {shape(v): c for v, c in fakes.items()} == want  # same length, words, case, punctuation, digits
    # the real side of the statistics shows placeholders, never a pseudonymised value
    assert "[pseudonymized 1]" in p.stats["levels"] and not set(RARE_DISMISSAL) & set(p.stats["levels"])


def test_frequent_and_vocabulary_values_keep_their_real_values() -> None:
    tp, cols = run_table(base_columns())
    race = Counter(cols["Race"].tolist())
    assert race == Counter(dict(RACE))  # 'Pacific Islander' and 'NA' (1 row each) are vocabulary words
    status = Counter(cols["Status"].tolist())
    assert status == Counter(dict(STATUS))  # 'Withdrawn' (vocabulary), 'Zebra Club' (19 rows)
    assert dtext.pseudonymized(profile_of(tp, "Race")) == {} == dtext.pseudonymized(profile_of(tp, "Status"))


def test_org_codes_keep_their_real_values() -> None:
    tp, cols = run_table(base_columns())
    p = profile_of(tp, "School_Name")
    assert p.type is T.ORG_CODE and dtext.pseudonymized(p) == {}
    assert Counter(cols["School_Name"].tolist()) == Counter(dict(SCHOOL))
    assert any(m.code == "rare_levels_kept" for m in p.messages)


def test_pii_hinted_columns_never_keep_a_value() -> None:
    tp, cols = run_table(base_columns())
    p = profile_of(tp, "Teacher")
    assert p.type in NEVER_COPY_TYPES  # D12: fake names (header hint), never a real value
    out = Counter(cols["Teacher"].tolist())
    assert not {"Ms. Garcia", "Mr. Lee"} & set(out) and sorted(out.values()) == [60, 60]


def test_personal_data_shapes_are_pseudonymised_even_when_frequent() -> None:
    tp, cols = run_table(base_columns())
    p = profile_of(tp, "Info")
    assert p.type is T.FACTOR and set(dtext.pseudonymized(p)) == {"ana.lopez@school.org", "312-555-0187"}
    out = Counter(cols["Info"].tolist())
    assert out["Front Office"] == 50 and not {"ana.lopez@school.org", "312-555-0187"} & set(out)
    emails = [v for v in out if "@" in v]
    assert len(emails) == 1 and out[emails[0]] == 40 and emails[0].rpartition("@")[2].startswith("example.")
    phones = [v for v in out if re.fullmatch(r"\d{3}-\d{3}-\d{4}", v)]
    assert len(phones) == 1 and out[phones[0]] == 30 and phones[0][4] in "23456789"


def test_force_factor_keeps_every_real_value() -> None:
    tp, cols = run_table(base_columns(), force_factor=["Dismissal"])
    assert dtext.pseudonymized(profile_of(tp, "Dismissal")) == {}
    assert Counter(cols["Dismissal"].tolist()) == Counter(dict(DISMISSAL))


def test_sentence_levels_are_no_longer_free_text() -> None:
    # D33 is retired: a sentence shared by >= --min-cell rows is kept, a rare one pseudonymised; the column stays a
    # factor either way
    long = "Return medication to the nurse after lunch"
    vals = spread([("Bus", 100), (long, 12), ("Go home with the neighbour on Fridays", 8)])
    tp, cols = run_table({"Dismissal": vals})
    p = profile_of(tp, "Dismissal")
    assert p.type is T.FACTOR and set(dtext.pseudonymized(p)) == {"Go home with the neighbour on Fridays"}
    out = Counter(cols["Dismissal"].tolist())
    assert out[long] == 12 and out["Bus"] == 100 and sorted(others(out, {"Bus", long}).values()) == [8]


def test_frequent_name_without_a_personal_header_is_the_accepted_residual() -> None:
    # D35 accepted residual risk: a name shared by >= k entities under a header that hints no personal data
    vals = spread([("Bus", 100), ("Smith", 20)])
    _, cols = run_table({"Homeroom": vals})
    assert Counter(cols["Homeroom"].tolist())["Smith"] == 20


def test_within_id_constant_and_case_forms() -> None:
    sid = [f"S{1000 + i // 3}" for i in range(N)]  # 40 students x 3 rows
    homeroom = [("Rm 101" if s < 2 else "Rm 202" if s < 4 else "Library") for s in (i // 3 for i in range(N))]
    homeroom = [v.upper() if (i // 3) == 1 else v for i, v in enumerate(homeroom)]  # 'RM 101' for one student
    tp, cols = run_table({"student_id": sid, "Homeroom": homeroom})
    p = profile_of(tp, "Homeroom")
    assert p.role == "entity" and set(dtext.pseudonymized(p)) == {"Rm 101", "RM 101", "Rm 202"}
    out = cols["Homeroom"].tolist()
    c = Counter(out)
    assert c["Library"] == 108 and not {"Rm 101", "RM 101", "Rm 202"} & set(c)
    assert sorted(others(c, {"Library"}).values()) == [3, 3, 6]
    fakes = set(others(c, {"Library"}))
    assert any(f.isupper() for f in fakes) and all(re.fullmatch(r"[A-Za-z]{2} \d{3}", f) for f in fakes)
    per_id: dict[str, set[str]] = {}
    for s, v in zip(cols["student_id"].tolist(), out):
        per_id.setdefault(s, set()).add(v)
    assert all(len(v) == 1 for v in per_id.values())  # still constant within each synthetic ID


def test_multi_select_items() -> None:
    vals = spread([("ELL;SPED", 40), ("ELL", 30), ("Robotics Club;ELL", 20), ("SPED", 27), ("Ana Lopez;SPED", 3)])
    tp, cols = run_table({"Services": vals})
    p = profile_of(tp, "Services")
    assert p.type is T.MULTI_SELECT and set(dtext.pseudonymized(p)) == {"Ana Lopez"}
    items = Counter(it for v in cols["Services"].tolist() for it in v.split(";"))
    assert items["ELL"] == 90 and items["SPED"] == 70 and items["Robotics Club"] == 20
    fakes = others(items, {"ELL", "SPED", "Robotics Club"})
    assert "Ana Lopez" not in items and list(fakes.values()) == [3]
    assert re.fullmatch(r"[A-Z][a-z]{2} [A-Z][a-z]{4}", next(iter(fakes)))
    combos = Counter(cols["Services"].tolist())
    assert combos["ELL;SPED"] == 40 and combos["Robotics Club;ELL"] == 20


def test_tokens_among_numbers_and_na_tokens() -> None:
    vals = [str(10 + i) for i in range(96)] + ["*"] * 5 + ["Zork"] * 12 + ["medical exemption"] * 3 + ["N/A"] * 4
    random.Random(3).shuffle(vals)
    tp, cols = run_table({"Minutes": vals})
    p = profile_of(tp, "Minutes")
    assert p.type is T.NUMERIC_WITH_TOKENS and set(dtext.pseudonymized(p)) == {"medical exemption"}
    out = Counter(cols["Minutes"].tolist())
    assert out["*"] == 5 and out["Zork"] == 12 and out["N/A"] == 4 and "medical exemption" not in out
    fakes = [v for v in out if not re.fullmatch(r"-?\d+(\.\d+)?", v) and v not in ("*", "Zork", "N/A")]
    assert len(fakes) == 1 and out[fakes[0]] == 3 and re.fullmatch(r"[a-z]{7} [a-z]{9}", fakes[0])
    assert p.stats["token_counts"].get("pseudonymized") == 3


def test_fd_dependent_follows_the_written_values() -> None:
    route = spread([("North Loop", 60), ("South Loop", 55), ("Zeta Run", 5)])
    desk = [{"North Loop": "Desk Alpha", "South Loop": "Desk Beta", "Zeta Run": "Desk Omega"}[r] for r in route]
    tp, cols = run_table({"Route": route, "Desk": desk})
    assert tp.structure.private.get("fds")
    pairs = Counter(zip(cols["Route"].tolist(), cols["Desk"].tolist()))
    assert pairs[("North Loop", "Desk Alpha")] == 60 and pairs[("South Loop", "Desk Beta")] == 55
    rest = [(r, d) for r, d in pairs if r not in ("North Loop", "South Loop")]
    assert len(rest) == 1 and pairs[rest[0]] == 5
    r, d = rest[0]
    assert r != "Zeta Run" and d != "Desk Omega" and shape(r) == shape("Zeta Run") and shape(d) == shape("Desk Omega")
    assert d == pseudonyms.mapped_value(profile_of(tp, "Desk"), "Desk Omega")  # the dependent's own pseudonym


def test_same_seed_same_pseudonyms_other_seed_other_pseudonyms() -> None:
    _, a = run_table(base_columns(), seed=1)
    _, b = run_table(base_columns(), seed=1)
    _, c = run_table(base_columns(), seed=2)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    fa = set(Counter(a["Dismissal"].tolist())) - {"Bus", "Car Rider", "Aftercare Club"}
    fc = set(Counter(c["Dismissal"].tolist())) - {"Bus", "Car Rider", "Aftercare Club"}
    assert fa != fc  # drawn from the seeded stream, not a hash of the value


def test_one_value_one_pseudonym_in_every_column_not_a_hash() -> None:
    vals = spread([("Bus", 117), ("John Smith", 3)])
    _, cols = run_table({"Dismissal": vals, "Pickup": list(vals)})
    fa = set(cols["Dismissal"].tolist()) - {"Bus"}
    fb = set(cols["Pickup"].tolist()) - {"Bus"}
    assert len(fa) == 1 and fa == fb  # one value, two columns, ONE pseudonym: equal columns stay equal
    assert list(cols["Dismissal"]) == list(cols["Pickup"])
    _, other = run_table({"Dismissal": vals, "Pickup": list(vals)}, seed=2)
    assert set(other["Dismissal"].tolist()) - {"Bus"} != fa  # drawn from the seed, not a hash of the value


def test_pseudonym_never_equals_a_real_value_of_the_run() -> None:
    tp, _ = run_table({"Region": spread([("Central", 100), ("Kitui", 20 - 15), ("Nairobi", 15)])})
    p = profile_of(tp, "Region")
    assert set(dtext.pseudonymized(p)) == {"Kitui"}
    lorem5 = frozenset(w.casefold() for w in gtext.lorem_words() if len(w) == 5)
    tp.private.extra[pseudonyms.RUN_TEXTS_KEY] = lorem5  # every 5-letter lorem word is a real value elsewhere
    pseudonyms.build_maps(tp, lambda q: np.random.default_rng(1))
    fake = pseudonyms.level_map(p)["Kitui"]
    assert len(fake) == 5 and fake.casefold() not in lorem5 and fake.casefold() != "kitui"


def test_pseudonym_shapes() -> None:
    pyr = random.Random(5)
    for real, rx in (("Rift Valley", r"[A-Z][a-z]{3} [A-Z][a-z]{5}"),
                     ("McKinley-Vento 2nd", r"[A-Z][a-z][A-Z][a-z]{5}-[A-Z][a-z]{4} \d[a-z]{2}"),
                     ("AB-0123", r"[A-Z]{2}-0\d{3}"), ("  spaced  ", r"  [a-z]{6}  ")):
        assert re.fullmatch(rx, pseudonyms.pseudonym(real, pyr)), real
    fake = pseudonyms.pseudonym("Mom: ana@district.edu", pyr)
    assert fake.startswith(("Mom", "Xxx")) or fake[3] == ":"
    assert "district" not in fake and fake.rpartition("@")[2].startswith("example.")


def test_numbers_keep_their_own_rules() -> None:
    # a plain number written as text is a number (D35: numbers follow their own rules), even when rare; a
    # leading-zero code is a code and follows the count rule
    tp, cols = run_table({"Year_Label": ["2024"] * 6, "Code": ["007"] * 3 + ["Active"] * 3})
    assert Counter(cols["Year_Label"].tolist()) == {"2024": 6}
    assert set(dtext.pseudonymized(profile_of(tp, "Code"))) == {"007"}  # 'Active' is a vocabulary word
    assert Counter(cols["Code"].tolist())["Active"] == 3 and "007" not in cols["Code"].tolist()


def test_tokens_under_a_personal_header_are_never_kept_for_their_count() -> None:
    # D12: under a person header no real text is kept for its count (vocabulary tokens such as '*' are, among
    # numbers that are no person numbers: Note_Count); hardening pass 2: the numbers under 'Teacher' are staff
    # numbers, so the whole column gets same-shape masks, NA tokens kept
    vals = [str(10 + i) for i in range(90)] + ["*"] * 6 + ["Garcia"] * 12 + ["N/A"] * 12
    random.Random(4).shuffle(vals)
    tp, cols = run_table({"Teacher": vals, "n": [str(i) for i in range(N)]})
    p = profile_of(tp, "Teacher")
    assert p.type is T.HIGH_CARD_CATEGORICAL and p.params.get("_numeric_mask")
    out = Counter(cols["Teacher"].tolist())
    assert not (set(out) - {"N/A"}) & set(vals) and out["N/A"] == 12
    assert sorted(out.values(), reverse=True)[:3] == [12, 12, 6]  # the frequency profile is kept
    tp, cols = run_table({"Note_Count": vals, "n": [str(i) for i in range(N)]})
    p = profile_of(tp, "Note_Count")
    assert p.type is T.NUMERIC_WITH_TOKENS and set(dtext.pseudonymized(p)) == {"Garcia"}  # 12 rows, still replaced
    out = Counter(cols["Note_Count"].tolist())
    assert "Garcia" not in out and out["*"] == 6 and out["N/A"] == 12


# =========================================================================== the leak scanner
def test_leak_scanner_excuses_exactly_the_kept_values() -> None:
    tp, cols = run_table(base_columns())
    p = profile_of(tp, "Dismissal")
    assert dtext.copy_cleared(p) and leaks.sensitive_kind(p) == leaks.LABEL
    real = leaks._real_values(tp, p, leaks.LABEL)
    assert set(real) == set(RARE_DISMISSAL)  # every pseudonymised real level is indexed, nothing else
    sets = leaks._cached_sets(tp, p, leaks.LABEL, real)
    assert leaks._same_column_hits(leaks.LABEL, sets, ["Bus", "Aftercare Club", "Car Rider"]) == 0
    assert leaks._same_column_hits(leaks.LABEL, sets, ["Bus", "john  SMITH"]) == 1  # a copy is a hit
    assert leaks.cross_column_values(p, ["Bus", "Aftercare Club", "John Smith", "Xyz"]) == ["John Smith", "Xyz"]
    # layer 1: a synthetic table that copies a pseudonymised real value FAILs before writing
    bad = cols["Dismissal"].copy()
    bad[0] = "QX-17"
    st = SyntheticTable(None, list(cols), N, [SyntheticColumn(q.index, q.name, bad if q.name == "Dismissal"
                                                              else cols[q.name]) for q in tp.columns], tp)
    assert any(m.code == "copied_value" for m in leaks.scan_table(tp, st))
    # a column whose values are all kept is not sensitive, and its decision is made (never pending)
    race = profile_of(tp, "Race")
    assert leaks.sensitive_kind(race) is None and dtext.copy_cleared(race)


def test_pending_decision_is_never_cleared() -> None:
    from test_area_a_support import detect

    det = detect(spread(DISMISSAL), "Dismissal")
    assert det.params.get(dtext.PENDING_KEY) and not dtext.copy_cleared(det)


# =========================================================================== end to end (CLI)
def test_cli_end_to_end(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "s.csv", base_columns())
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    rare = ("John Smith", "grandmother", "QX-17", "ana.lopez", "312-555-0187", "Ms. Garcia", "Mr. Lee")
    assert found(rare, out, rep, console=res.output) == []
    rows = read_dicts(out / "s.csv")
    c = Counter(r["Dismissal"] for r in rows)
    assert c["Bus"] == 60 and c["Aftercare Club"] == 12 and len(c) == 6
    col = column_report(rep, "s.csv", "Dismissal")
    assert col["params"]["pseudonymized_levels"] == 3 and col["flags"]["pseudonymized_levels"]
    assert "pseudonymized_levels" in [m["code"] for m in col["messages"]]
    for side in ("real", "synthetic"):  # the placeholders line up, so fidelity holds
        assert col[side]["level_shares"]["[pseudonymized 1]"] == pytest.approx(3 / N, abs=1e-6)
    assert col["fidelity"]["verdict"] == "ok"
    assert leak_check(rep, "s.csv") == "PASS"
    dd = (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "'Aftercare Club': 12" in dd and "pseudonymized_levels (3)" in dd and "[pseudonymized" not in dd
    for fake in set(c) - {"Bus", "Car Rider", "Aftercare Club"}:  # the docs list the written pseudonyms
        assert repr(fake) in dd
    res2, out2, _ = run(src.parent, tmp_path, name="again")
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)  # D26: same seed, same bytes


def test_cli_force_factor(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "s.csv",
                    {"Dismissal": spread(DISMISSAL), "Score": [str(i % 37) for i in range(N)]})
    res, out, rep = run(src.parent, tmp_path, "--force-factor", "Dismissal")
    assert res.exit_code == 0, res.output
    assert Counter(r["Dismissal"] for r in read_dicts(out / "s.csv")) == Counter(dict(DISMISSAL))


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("fmt", ["parquet", "feather"])
def test_arrow_dictionary(tmp_path: Path, fmt: str, ordered: bool) -> None:
    levels = ["Standard", "Honors", "Zeta Cohort"]
    idx = [0] * 60 + [1] * 57 + [2] * 3
    random.Random(1).shuffle(idx)
    arr = pa.DictionaryArray.from_arrays(pa.array(idx, pa.int8()), pa.array(levels), ordered=ordered)
    src = tmp_path / "in"
    src.mkdir()
    path = src / f"d.{fmt}"
    (pf.write_feather if fmt == "feather" else pq.write_table)(pa.table({"Track": arr}), path)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Zeta Cohort",), out, rep, console=res.output) == []
    back = (pf.read_table if fmt == "feather" else pq.read_table)(out / path.name)
    t = back.schema.field("Track").type
    assert pa.types.is_dictionary(t) and t.ordered is ordered
    entries = back["Track"].combine_chunks().dictionary.to_pylist()
    if ordered:  # the order is kept, the pseudonym in the real entry's place
        assert entries[:2] == ["Standard", "Honors"] and re.fullmatch(r"[A-Z][a-z]{3} [A-Z][a-z]{5}", entries[2])
    vals = Counter(back["Track"].to_pylist())
    assert vals["Standard"] == 60 and vals["Honors"] == 57
    assert sorted(others(vals, {"Standard", "Honors"}).values()) == [3]
    assert leak_check(rep, path.name) == "PASS"


def test_stata_value_labels(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    codes = [1] * 60 + [2] * 57 + [3] * 3
    random.Random(2).shuffle(codes)
    pd.DataFrame({"track": codes}).to_stata(src / "a.dta", write_index=False, version=118,
                                            value_labels={"track": {1: "Standard", 2: "Honors", 3: "Zeta Cohort"}})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Zeta Cohort",), out, rep, console=res.output) == []
    lab = next(iter(pd.io.stata.StataReader(out / "a.dta").value_labels().values()))
    assert lab[1] == "Standard" and lab[2] == "Honors" and re.fullmatch(r"[A-Z][a-z]{3} [A-Z][a-z]{5}", lab[3])
    back = pd.read_stata(out / "a.dta", convert_categoricals=False)
    assert Counter(back["track"].tolist()) == {1: 60, 2: 57, 3: 3}
    assert leak_check(rep, "a.dta") == "PASS"


def test_example_data_levels(example_run) -> None:  # noqa: ANN001 - the session fixture
    """Invented example labels retain exact quotas. In the 62-row campus fixture,
    Coast and Northern have fewer than 10 rows and are allowed vocabulary words;
    the other region labels have at least 10 rows. Course is an org code (D10)."""
    real = Counter(r["SitKCSE"] for r in read_dicts(EXAMPLES / "students_data.csv"))
    rows = read_dicts(example_run.out / "students_data.csv")
    assert Counter(r["SitKCSE"] for r in rows) == real
    for word in ("Coast", "Eastern ", "Western", "Central "):
        assert vocab.is_vocabulary(word), word
    real_course = Counter(r["Course"] for r in read_dicts(EXAMPLES / "students_data.csv"))
    assert Counter(r["Course"] for r in rows) == real_course  # an org code (D10)
    for name, col in (("StudentsPerformance.csv", "race/ethnicity"), ("student-scores.csv", "career_aspiration")):
        assert (Counter(r[col] for r in read_dicts(example_run.out / name))
                == Counter(r[col] for r in read_dicts(EXAMPLES / name)))


# =========================================================================== hardening pass 1
CANARY = "Visiting Abuela Xochiquetzal in Oaxaca"


def _attendance(first: str) -> dict[str, list[str]]:
    """40 students x 12 rows; one student's 12 rows hold CANARY. ``first`` (listed before ``student_id``) is a record
    ID (one per row) or a staff ID (12 different teachers for that student)."""
    n = 480
    sid = [f"S{5000 + i // 12}" for i in range(n)]
    reason = [CANARY if i < 12 else ("Appointment" if i % 3 else "Illness") for i in range(n)]
    other = [f"T{100 + i % 25}" for i in range(n)] if first in ("teacherid", "staffid") else \
        [str(900000 + i) for i in range(n)]
    return {first: other, "student_id": sid, "Reason": reason}


@pytest.mark.parametrize("first", ["id", "dcid", "teacherid", "staffid"])
def test_holders_are_counted_over_every_id_column(first: str) -> None:
    # attack35 leak 1: the primary ID (first default-list ID) may number records or staff, not the student
    tp, cols = run_table(_attendance(first))
    p = profile_of(tp, "Reason")
    assert CANARY in dtext.pseudonymized(p) and CANARY not in set(cols["Reason"].tolist())
    out = Counter(cols["Reason"].tolist())
    assert out["Appointment"] == 312 and out["Illness"] == 156  # vocabulary words stay
    assert sorted(others(out, {"Appointment", "Illness"}).values()) == [12]


def test_record_id_only_counts_the_subjects_names_not_other_roles() -> None:
    n = 240
    names = [f"{a} {b}" for a in ("Ana", "Ben", "Cleo", "Dev", "Eli") for b in ("Lopez", "Moss", "Ng", "Ortiz")]
    student = [names[i % 20] for i in range(n)]
    reason = [CANARY if student[i] == names[0] else "Appointment" for i in range(n)]
    teacher = [("Ms. Garcia", "Mr. Lee")[i % 2] for i in range(n)]
    club = ["Zebra Club" if i % 20 < 12 else "None" for i in range(n)]
    tp, cols = run_table({"dcid": [str(70000 + i) for i in range(n)], "Student_Name": student, "Teacher": teacher,
                          "Reason": reason, "Club": club})
    assert profile_of(tp, "Student_Name").type is T.PERSON_NAME
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason"))  # 12 records, ONE student name
    assert dtext.pseudonymized(profile_of(tp, "Club")) == {}  # 12 students hold it (2 teachers do not count)
    assert Counter(cols["Club"].tolist())["Zebra Club"] == 144


def test_blank_ids_are_one_holder() -> None:
    # attack35 leak 3: 12 rows with a blank student_id are not 12 people
    sid = [f"S{5000 + i // 3}" for i in range(120)] + [""] * 12
    pickup = [("Bus" if i % 2 else "Car Rider") for i in range(120)] + ["Pickup by uncle Thelonius Brackenridge"] * 12
    club = [("Zebra Club" if i < 36 else "Chess") for i in range(120)] + ["Zebra Club"] * 12
    tp, cols = run_table({"student_id": sid, "Pickup": pickup, "Club": club})
    assert "Pickup by uncle Thelonius Brackenridge" in dtext.pseudonymized(profile_of(tp, "Pickup"))
    assert "Thelonius" not in " ".join(cols["Pickup"].tolist())
    assert dtext.pseudonymized(profile_of(tp, "Club")) == {}  # 12 students with IDs (+ the blank-ID rows)


def _dismissal_log() -> dict[str, list[str]]:
    """No ID column: 200 rows (building, dismissal, minutes) plus one record exported 11 times."""
    rng = random.Random(11)
    building = [rng.choice(["Adams HS", "Lincoln HS"]) for _ in range(200)]
    dismissal = [rng.choice(["Bus", "Car Rider", "Walker"]) for _ in range(200)]
    dismissal[:12] = ["Zebra Shuttle"] * 12  # 12 different rows: 12 holders
    minutes = [str(rng.randint(1, 30)) for _ in range(200)]
    minutes[:12] = [str(31 + i) for i in range(12)]
    return {"building": building + ["Lincoln HS"] * 11,
            "Dismissal": dismissal + ["Grandma Philomena Szczepanik picks up"] * 11,
            "minutes": minutes + ["4"] * 11}


def test_identical_rows_of_a_table_without_ids_count_once() -> None:
    # attack35 leak 2: one record repeated (identical in every column) is one holder, not 11
    tp, cols = run_table(_dismissal_log())
    p = profile_of(tp, "Dismissal")
    assert set(dtext.pseudonymized(p)) == {"Grandma Philomena Szczepanik picks up"}
    out = Counter(cols["Dismissal"].tolist())
    assert out["Zebra Shuttle"] == 12 and "Grandma Philomena Szczepanik picks up" not in out


def test_rows_alike_by_nature_still_count_as_rows() -> None:
    # a narrow table (or a column that follows another) repeats its rows by nature: rows still count
    route = spread([("North Loop", 60), ("South Loop", 48), ("Zeta Run", 12)])
    desk = [{"North Loop": "Desk Alpha", "South Loop": "Desk Beta", "Zeta Run": "Desk Omega"}[r] for r in route]
    zebra = set(random.Random(3).sample(range(N), 12))
    club = ["Zebra Club" if i in zebra else "Chess" for i in range(N)]
    tp, cols = run_table({"Route": route, "Desk": desk, "Club": club})
    for name in ("Route", "Desk", "Club"):
        assert dtext.pseudonymized(profile_of(tp, name)) == {}, name
    assert Counter(cols["Desk"].tolist())["Desk Omega"] == 12


def test_cross_file_people_count_wins_over_rows(tmp_path: Path) -> None:
    # attack35 leak 2 (cross-file): a.csv shows ONE student holds the value; b.csv (no ID) repeats it on 12 rows
    src = tmp_path / "in"
    n = 60
    write_csv(src / "a.csv", {"student_id": [f"S{5000 + i // 2}" for i in range(n)],
                              "Pickup": ["Ottoline Quarrington" if i < 2 else ("Bus", "Walker")[i % 2]
                                         for i in range(n)]})
    write_csv(src / "b.csv", {"school": [("Lincoln HS", "Adams HS")[i % 2] for i in range(n)],
                              "Pickup": ["Ottoline Quarrington" if i < 12 else ("Bus", "Walker")[i % 2]
                                         for i in range(n)],
                              "minutes": [str(i) for i in range(n)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Quarrington",), out, rep, console=res.output) == []
    fa = {r["Pickup"] for r in read_dicts(out / "a.csv")} - {"Bus", "Walker"}
    fb = Counter(r["Pickup"] for r in read_dicts(out / "b.csv"))
    assert len(fa) == 1 and fb[next(iter(fa))] == 12  # the same pseudonym in both files, its exact count
    assert leak_check(rep, "a.csv") == leak_check(rep, "b.csv") == "PASS"


@pytest.mark.parametrize("value", ["4155550187", "14155550187", "123456789", "4155550187.0", "1234567"])
def test_long_digit_strings_are_not_exempt_numbers(value: str) -> None:
    # attack35 leak 4: a bare phone or SSN-like number held by one student is no "number" kept for free
    assert not dtext.short_number(value)
    assert not dtext.unit_kept(value, holders=1, min_cell=10)
    sid = [f"S{5000 + i // 2}" for i in range(N)]
    pickup = [value if i < 2 else "12" if i < 4 else ("Bus", "Car Rider", "Walker")[i % 3] for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Pickup": pickup})
    assert value in dtext.pseudonymized(profile_of(tp, "Pickup"))
    out = Counter(cols["Pickup"].tolist())
    assert value not in out and out["12"] == 2  # a short number keeps its own rules


def test_short_numbers_stay_numbers() -> None:
    for v in ("0", "12", "-3", "2.5", "123456", "+44"):
        assert dtext.short_number(v) and dtext.unit_kept(v, holders=1, min_cell=10), v
    for v in ("007", "1234567", "12a"):
        assert not dtext.short_number(v), v


def test_last_first_pairs_are_not_a_multi_select() -> None:
    # attack35 leak 5: 'Nguyen, John' is ONE value, not the items 'Nguyen' and 'John'
    rng = random.Random(5)
    common = ["Garcia, Maria", "Garcia, John", "Smith, Maria", "Smith, John"]
    vals = [rng.choice(common) for _ in range(N - 1)] + ["Smith, Linh"]
    vals[5] = "Garcia, Linh"
    vals[6:15] = ["Nguyen, Maria"] * 9
    vals[15] = "Nguyen, John"
    tp, cols = run_table({"student_id": [f"S{1000 + i}" for i in range(N)], "Authorized_Pickup": vals})
    p = profile_of(tp, "Authorized_Pickup")
    assert p.type is not T.MULTI_SELECT
    out = set(cols["Authorized_Pickup"].tolist())
    assert not {"Nguyen, John", "Nguyen, Maria", "Smith, Linh", "Garcia, Linh"} & out
    assert dtext.pair_layout(Counter(vals), ",")
    lists = Counter({"ELL;SPED": 40, "ELL": 30, "Robotics Club;ELL": 20, "SPED": 27})
    assert not dtext.pair_layout(lists, ";")  # a list repeats its items in any position


FORCED = {"student_id": [f"S{1000 + i // 2}" for i in range(N)],
          "Pickup": ["Grandma Philomena Szczepanik" if i < 2 else ("Bus", "Car Rider", "Walker")[i % 3]
                     for i in range(N)],
          "Acts": ["Band;Tutoring with Mr. Abernathy-Cole" if i < 2 else ("Band;Chess", "Chess", "Band")[i % 3]
                   for i in range(N)],
          "Score": ["absent - Dr. Vanterpool" if i < 2 else str(10 + i % 80) for i in range(N)]}


@pytest.mark.parametrize("flags", [
    {"force_type": ["Pickup=FACTOR", "Acts=MULTI_SELECT", "Score=NUMERIC_WITH_TOKENS"]},
    {"force_type": ["Pickup=ORDINAL", "Acts=FACTOR"]},
])
def test_force_type_names_a_type_not_the_values(flags: dict[str, list[str]]) -> None:
    # attack35 leak 6: only --force-factor keeps every real value (D35's explicit opt-in)
    tp, cols = run_table(dict(FORCED), **flags)
    for col, word in (("Pickup", "Szczepanik"), ("Acts", "Abernathy"), ("Score", "Vanterpool")):
        p = profile_of(tp, col)
        assert not any(word in str(v) for v in cols[col].tolist()), (col, p.type, p.source)
        assert dtext.copy_cleared(p) or not dtext.copies_real_values(p)
    assert Counter(cols["Pickup"].tolist())["Bus"] == 39
    tp, cols = run_table(dict(FORCED), force_factor=["Pickup"])
    assert Counter(cols["Pickup"].tolist())["Grandma Philomena Szczepanik"] == 2  # the explicit opt-in


def test_force_type_cli_passes_the_leak_scan(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "f.csv", dict(FORCED))
    res, out, rep = run(src.parent, tmp_path, "--force-type", "Pickup=FACTOR", "--force-type", "Acts=MULTI_SELECT")
    assert res.exit_code == 0, res.output
    assert found(("Szczepanik", "Abernathy", "Vanterpool"), out, rep, console=res.output) == []
    assert leak_check(rep, "f.csv") == "PASS"


def test_org_code_flag_still_never_keeps_personal_data_shapes() -> None:
    vals = ["front.office@lincoln.org" if i < 15 else "Lincoln HS" if i % 2 else "Adams HS" for i in range(N)]
    tp, cols = run_table({"Campus_Info": vals}, org_code=["Campus_Info"])
    p = profile_of(tp, "Campus_Info")
    assert p.type is T.ORG_CODE and set(dtext.pseudonymized(p)) == {"front.office@lincoln.org"}
    out = Counter(cols["Campus_Info"].tolist())
    assert "front.office@lincoln.org" not in out and out["Lincoln HS"] + out["Adams HS"] == N - 15
    # R6-03: under a header that hints a person ('Contact') --org-code keeps the type and no real value at all
    tp, cols = run_table({"Campus_Contact": vals}, org_code=["Campus_Contact"])
    p = profile_of(tp, "Campus_Contact")
    assert p.type is T.ORG_CODE and set(dtext.pseudonymized(p)) == set(vals)
    assert not set(cols["Campus_Contact"].tolist()) & set(vals)


def test_personal_header_keeps_only_placeholder_words() -> None:
    # attack35 leak 7: under a personal header a vocabulary word that is a surname is not kept
    vals = ([str(1000 + i) for i in range(100)] + ["White"] * 2 + ["Summer"] + ["Okonjo"] + ["*"] * 6
            + ["N/A"] * 4 + ["Unknown"] * 6)
    random.Random(4).shuffle(vals)
    # 'Teachers' (plural: a count of teachers, so its numbers stay numbers) is still a personal header
    tp, cols = run_table({"student_id": [f"S{1000 + i}" for i in range(N)], "Teachers": vals})
    p = profile_of(tp, "Teachers")
    assert p.type is T.NUMERIC_WITH_TOKENS
    assert set(dtext.pseudonymized(p)) == {"White", "Summer", "Okonjo"}
    out = Counter(cols["Teachers"].tolist())
    assert out["*"] == 6 and out["N/A"] == 4 and out["Unknown"] == 6 and not {"White", "Summer", "Okonjo"} & set(out)


def test_name_like_vocabulary_needs_its_section_in_the_column() -> None:
    # attack35 leak 8: 'White' and 'Summer' are surnames/given names under Homeroom, vocabulary beside 'Asian'/'Fall'
    sid = [f"S{1000 + i}" for i in range(N)]
    homeroom = ["White" if i == 0 else "Summer" if i == 1 else "Okonjo" if i < 60 else "Garcia" for i in range(N)]
    race = ["White" if i < 3 else "Black" if i < 5 else "Asian" if i < 60 else "Hispanic" for i in range(N)]
    term = ["Summer" if i < 2 else "Fall" if i < 60 else "Spring" for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Homeroom": homeroom, "Race": race, "Term": term})
    assert set(dtext.pseudonymized(profile_of(tp, "Homeroom"))) == {"White", "Summer"}
    assert not {"White", "Summer"} & set(cols["Homeroom"].tolist())
    assert Counter(cols["Race"].tolist()) == Counter(race) and Counter(cols["Term"].tolist()) == Counter(term)


@pytest.mark.parametrize("value", ["Ma Th", "E.L.", "Art-Ist", "Tu", "An", "Na", "White", "Summer", "Major",
                                   "English", "Parent", "Lao", "Mr", "Hi"])
def test_names_and_joined_letters_are_not_vocabulary(value: str) -> None:
    assert not vocab.is_vocabulary(value)
    assert not dtext.unit_kept(value, holders=9, min_cell=10)


def test_name_like_words_with_their_section() -> None:
    race = vocab.context_domains(["Asian", "Hispanic or Latino"])
    assert vocab.is_vocabulary("White", race) and vocab.is_vocabulary("Black", race)
    assert not vocab.is_vocabulary("Summer", race)
    assert vocab.is_vocabulary("Summer", vocab.context_domains(["Fall", "Spring"]))
    assert not vocab.context_domains(["White", "Black", "Okonjo"])  # name-like words corroborate nothing
    for code in ("GT", "SLD", "NA", "gt", "sld"):
        assert vocab.is_vocabulary(code), code  # abbreviations in one case are codes
    # hardening pass 2: codes that are surnames in an all-caps export need their section too
    attendance = vocab.context_domains(["Present", "Absent", "ISS"])
    race = vocab.context_domains(["AS", "WH", "HI"])
    for code, domains in (("AN", race), ("an", race), ("MENA", race), ("TU", attendance), ("tu", attendance),
                          ("AU", attendance), ("IP", vocab.context_domains(["Credit", "No Credit"]))):
        assert vocab.is_vocabulary(code, domains) and not vocab.is_vocabulary(code), code


@pytest.mark.parametrize("value", ["R", "L", "FDC", "EB", "Emergent Bilingual", "Non-EB", "JK", "SK", "3K", "4K",
                                   "K4", "EE", "Y1", "Shelters", "Hotels/Motels", "Separate School",
                                   "Residential Facility", "Regular Class 80%+", "Non-Binary", "Monitored Year 1"])
def test_common_sis_codes_are_vocabulary(value: str) -> None:
    assert vocab.is_vocabulary(value) and dtext.unit_kept(value, holders=1, min_cell=10)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_pseudonyms_are_no_vocabulary_words(seed: int) -> None:
    # 2-letter codes: the lorem words of that shape (ET, UT, ID, EX ...) are mostly attendance or SPED codes
    codes = ["QZ", "XV", "JW", "ZQ", "VX", "WJ", "KQ", "QK", "XZ", "ZX", "QJ", "JQ", "ZJ"]
    vals = codes + ["Bus"] * (N - len(codes))
    tp, cols = run_table({"Code": vals}, seed=seed)
    fakes = set(cols["Code"].tolist()) - {"Bus"}
    assert len(fakes) == len(codes) and not [f for f in fakes if vocab.is_vocabulary(f)]


def test_lookup_table_labels_follow_the_fact_table(tmp_path: Path) -> None:
    # attack35 utility 1: a code table (one row per program) keeps the labels its fact table holds by many people
    src = tmp_path / "in"
    progs = [("PRG-ROB", "Robotics Academy"), ("PRG-AVI", "Aviation Pathway"), ("PRG-NUR", "Nursing Pathway"),
             ("PRG-CUL", "Culinary Arts Pathway"), ("PRG-ZZZ", "Zeppelin Pathway")]
    write_csv(src / "programs.csv", {"program_code": [c for c, _ in progs], "program_name": [nm for _, nm in progs],
                                     "credits": ["4", "5", "6", "7", "8"]})
    n = 200
    write_csv(src / "enrollments.csv", {"student_id": [f"S{5000 + i}" for i in range(n)],
                                        "program_code": [progs[i % 4][0] for i in range(n)],
                                        "program_name": [progs[i % 4][1] for i in range(n)],
                                        "grade": [str(9 + i % 4) for i in range(n)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    lookup = read_dicts(out / "programs.csv")
    fact = read_dicts(out / "enrollments.csv")
    names = {r["program_name"] for r in lookup}
    assert {nm for _, nm in progs[:4]} <= names and "Zeppelin Pathway" not in names  # 0 students hold it
    assert {r["program_name"] for r in fact} <= names  # the join on the name still matches
    assert {r["program_code"] for r in fact} <= {r["program_code"] for r in lookup}


def test_pseudonyms_are_consistent_across_columns_and_files(tmp_path: Path) -> None:
    # attack35 utility 2: equal values stay equal wherever they are written
    src = tmp_path / "in"
    for name, other in (("a.csv", "Gym"), ("b.csv", "Library")):
        loc = ["Zephyr Hall" if i < 3 else (other, "Gym")[i % 2] for i in range(80)]
        write_csv(src / name, {"student_id": [f"S{5000 + i}" for i in range(80)], "Location": loc,
                               "Location2": list(loc), "n": [str(i) for i in range(80)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Zephyr",), out, rep, console=res.output) == []
    fakes = set()
    for name in ("a.csv", "b.csv"):
        rows = read_dicts(out / name)
        assert all(r["Location"] == r["Location2"] for r in rows)
        fakes |= {r["Location"] for r in rows} - {"Gym", "Library"}
    assert len(fakes) == 1
    res2, out2, _ = run(src, tmp_path, name="again")
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)  # D26: same seed, same bytes


def test_docs_list_pseudonyms_and_keep_the_level_order(tmp_path: Path) -> None:
    # attack35 utility 4: the docs describe the synthetic files: pseudonyms are synthetic labels
    levels = ["Chess Guild", "Garden Guild", "Zeta Cohort"]
    idx = [0] * 60 + [1] * 57 + [2] * 3
    random.Random(1).shuffle(idx)
    arr = pa.DictionaryArray.from_arrays(pa.array(idx, pa.int8()), pa.array(levels), ordered=True)
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({"Club": arr}), src / "c.parquet")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    entries = pq.read_table(out / "c.parquet")["Club"].combine_chunks().dictionary.to_pylist()
    fake = entries[2]
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    col = schema["files"][0]["tables"][0]["columns"][0]
    assert col["level_order"] == ["Chess Guild", "Garden Guild", fake]
    dd = (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert repr(fake) in dd and "[pseudonymized" not in dd and fake in dd.split("Ordered levels of")[1]
    assert found(("Zeta Cohort",), out, rep, console=res.output) == []


def test_inspect_and_run_decide_over_the_run(tmp_path: Path) -> None:
    # a value 2 students hold in a.csv and 12 in b.csv is shared by >= 10 people: kept in both files
    src = tmp_path / "in"
    write_csv(src / "a.csv", {"student_id": [f"S{100 + i}" for i in range(40)],
                              "Club": ["Zebra Club" if i < 2 else "Chess Club" for i in range(40)]})
    write_csv(src / "b.csv", {"student_id": [f"S{100 + i}" for i in range(40)],
                              "Club": ["Zebra Club" if i < 12 else "Chess Club" for i in range(40)]})
    res = run_cli("inspect", src, "--json")
    assert res.exit_code == 0, res.output
    cols = {f["source"]: c for f in json.loads(res.output)["files"] for t in f["tables"] for c in t["columns"]
            if c["name"] == "Club"}
    assert set(cols) == {"a.csv", "b.csv"}
    assert all("pseudonymized_levels" not in c["params"] for c in cols.values())
    res, out, _ = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert Counter(r["Club"] for r in read_dicts(out / "a.csv"))["Zebra Club"] == 2


def test_holders_pool_per_field_not_across_columns() -> None:
    # 'Jordan' held by 20 students as a country says nothing about 'Jordan' held by one student as a buddy's name
    sid = [f"S{1000 + i}" for i in range(N)]
    country = ["Jordan" if i < 20 else "Kenya" if i < 70 else "Peru" for i in range(N)]
    buddy = ["Jordan" if i == 7 else "Quinby" if i < 60 else "Thaddeus" for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Home_Country": country, "Buddy": buddy})
    assert dtext.pseudonymized(profile_of(tp, "Home_Country")) == {}
    assert set(dtext.pseudonymized(profile_of(tp, "Buddy"))) == {"Jordan"}
    assert "Jordan" not in set(cols["Buddy"].tolist()) and Counter(cols["Home_Country"].tolist())["Jordan"] == 20


def test_uncounted_vocabulary_rows_never_vouch_for_a_name() -> None:
    # 'White' is decided as a race word without a count: its 60 rows (12 students) must not keep a surname elsewhere
    sid = [f"S{1000 + i // 5}" for i in range(N)]
    race = ["White" if i < 60 else "Asian" for i in range(N)]
    homeroom = ["White" if i < 15 else "Okonjo" for i in range(N)]  # 3 students
    tp, cols = run_table({"student_id": sid, "Race": race, "Homeroom": homeroom})
    assert dtext.pseudonymized(profile_of(tp, "Race")) == {}
    assert set(dtext.pseudonymized(profile_of(tp, "Homeroom"))) == {"White"}
    assert Counter(cols["Race"].tolist())["White"] == 60 and "White" not in set(cols["Homeroom"].tolist())


# =========================================================================== hardening pass 2
ACCOM = "Zyxwell Bracewalk Plan"


def _roster(key: str, n_students: int = 30, teachers: int = 12, names: bool = True) -> dict[str, list[str]]:
    """``n_students`` students x ``teachers`` rows, keyed by ``key`` (a teacher/staff/section number per row); the
    first student's 12 rows hold ACCOM, everyone else 'Extended Time' or 'Small Group'."""
    rows = n_students * teachers
    student = [i // teachers for i in range(rows)]
    out = {key: [str(500 + (i % teachers) * 7 + (i // teachers) % 3) for i in range(rows)]}
    if names:
        full = [f"{last}, {first}" for last in ("Anderson", "Baker", "Chen", "Diaz", "Evans", "Fischer")
                for first in ("Ashley", "Brian", "Carmen", "Dmitri", "Elena")]
        out["Student_Name"] = [full[s] for s in student]
    out["Accommodation"] = [ACCOM if s == 0 else ("Extended Time" if s % 2 else "Small Group") for s in student]
    return out


@pytest.mark.parametrize("key", ["TeacherID", "teacherid", "StaffID", "EmpID", "Guardian_Number", "SID"])
def test_other_role_ids_never_make_one_student_many_holders(key: str) -> None:
    # attack35_2 leak 1: one student's value seen under 12 teachers (or sections, staff, guardians) is ONE holder
    tp, cols = run_table(_roster(key))
    p = profile_of(tp, "Accommodation")
    assert ACCOM in dtext.pseudonymized(p) and ACCOM not in set(cols["Accommodation"].tolist())
    out = Counter(cols["Accommodation"].tolist())
    assert out["Extended Time"] == 180 and out["Small Group"] == 168 and sorted(others(out, {
        "Extended Time", "Small Group"}).values()) == [12]


def test_names_count_beside_the_persons_own_id() -> None:
    # an SID that numbers sessions (one per row) gives each row one name, but one name many IDs: names count
    full = [f"{a}, {b}" for a in ("Anderson", "Baker", "Chen", "Diaz", "Evans") for b in ("Ana", "Ben", "Cleo", "Dev")]
    rows = [s for s in range(20) for _ in range(12)]
    cols = {"SID": [str(5000 + i) for i in range(len(rows))], "Student_Name": [full[s] for s in rows],
            "Plan": [ACCOM if s == 0 else ("Extended Time", "Small Group")[s % 2] for s in rows]}
    tp, out = run_table(cols)
    assert ACCOM in dtext.pseudonymized(profile_of(tp, "Plan")) and ACCOM not in set(out["Plan"].tolist())
    # hardening pass 3: names always count, beside an ID that looks one to one with them too; two namesakes holding
    # a value count once (the safe side: 10 IDs under 9 distinct names are 9 holders)
    sid = [f"S{1000 + i}" for i in range(N)]
    names = [f"{a}, {b}" for a in ("Adams", "Brook", "Cole", "Dunn", "Ellis", "Ford", "Gray", "Hale", "Ives", "Judd")
             for b in ("Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lia")]
    club = ["Zebra Club" if i < 10 else "Chess" for i in range(N)]
    tp, out = run_table({"student_id": sid, "Student_Name": names, "Club": club})
    assert dtext.pseudonymized(profile_of(tp, "Club")) == {}  # 10 students (10 IDs, 10 names)
    names[1] = names[0]
    tp, out = run_table({"student_id": sid, "Student_Name": names, "Club": club})
    assert set(dtext.pseudonymized(profile_of(tp, "Club"))) == {"Zebra Club"}  # 10 IDs, 9 names
    tp, out = run_table({"student_id": sid, "Student_Name": [names[i % 2 + 2] for i in range(N)], "Club": club})
    assert set(dtext.pseudonymized(profile_of(tp, "Club"))) == {"Zebra Club", "Chess"}  # 2 names: 2 holders at most


@pytest.mark.parametrize("flags", [{}, {"exclude": ["Student_Name"]}, {"null": ["Student_Name"]},
                                   {"force_factor": ["Student_Name"]}, {"force_text": ["Student_Name"]},
                                   {"pii_col": ["Student_Name=full_name"]}])
def test_flags_on_the_name_column_still_count_its_people(flags: dict[str, list[str]]) -> None:
    tp, cols = run_table(_roster("TeacherID"), **flags)
    assert ACCOM in dtext.pseudonymized(profile_of(tp, "Accommodation")), flags
    assert ACCOM not in set(cols["Accommodation"].tolist())


def test_last_first_name_column_counts_people_beside_a_staff_key() -> None:
    cols = _roster("teacherid", names=False)
    cols["Student_LastFirst"] = [f"Stu{i // 12:03d}, Kid" for i in range(len(cols["teacherid"]))]
    tp, out = run_table(cols)
    assert ACCOM in dtext.pseudonymized(profile_of(tp, "Accommodation"))


@pytest.mark.parametrize("name,role", [
    ("student_id", "subject"), ("SID", "subject"), ("LASID", "subject"), ("UPN", "subject"),
    ("Pupil_ID", "subject"), ("Family_ID", "subject"), ("TeacherID", "other"), ("staffid", "other"),
    ("EmployeeID", "other"), ("Guardian_Number", "other"), ("Parent_ID", "other"), ("id", "record"),
    ("DCID", "record")])
def test_id_roles(name: str, role: str) -> None:
    # attack35_2 leak 1: only an ID of the row's own person proves that holders are people
    from test_area_a_support import detect

    from datanonymous.profile import entity

    det = detect([f"{1000 + i}" for i in range(30)], name)
    assert det.type is T.IDENTIFIER, name
    p = type("P", (), {"name": name, "source": det.source, "type": det.type})()
    assert entity.id_role(p) == role


def test_flagged_record_ids_are_no_proof_of_people() -> None:
    from test_area_a_support import detect

    from datanonymous.profile import entity

    # K-02: 'kid' is a person word (Kid_Number numbers the row's child); Case_Number names no person
    for name, flag, role in (("Incident_ID", "id_col", "record"), ("Row", "id_col", "record"),
                             ("Incident_ID", "person_id", "subject"), ("Case_Number", "id_col", "record"),
                             ("Kid_Number", "id_col", "subject"),
                             ("Student_Key", "id_col", "subject"), ("Mentor_ID", "id_col", "other")):
        det = detect([str(i) for i in range(1, 31)], name, **{flag: [name]})
        p = type("P", (), {"name": name, "source": det.source, "type": det.type})()
        assert entity.id_role(p) == role, (name, flag)


def _incidents() -> dict[str, list[str]]:
    """40 students, 6 incidents each (one row per incident); the first student's 11 incidents hold CANARY."""
    names = [f"Stu{i:03d}, Kid" for i in range(40)]
    student = [0] * 11 + [1 + i // 6 for i in range(39 * 6)]
    return {"Incident_ID": [f"INC{i:05d}" for i in range(len(student))],
            "Student_Name": [names[s] for s in student],
            "Action_Taken": [CANARY if s == 0 else ("Warning", "Conference", "Detention")[i % 3]
                             for i, s in enumerate(student)]}


def test_id_col_on_a_record_id_still_counts_the_students() -> None:
    # attack35_2 leak 3: following the advice '--id-col Incident_ID' counted one holder per incident
    for flags in ({}, {"id_col": ["Incident_ID"]}, {"force_type": ["Incident_ID=IDENTIFIER"]}):
        tp, cols = run_table(_incidents(), **flags)
        assert CANARY in dtext.pseudonymized(profile_of(tp, "Action_Taken")), flags
        assert CANARY not in set(cols["Action_Taken"].tolist())


def _absences() -> dict[str, list[str]]:
    """30 students x 14 rows (a different date on every row); the first student's 14 rows hold CANARY."""
    import datetime as dt

    n = 420
    return {"student_id": [str(30000 + i // 14) for i in range(n)],
            "Att_Date": [(dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(n)],
            "Period": [str(1 + i % 7) for i in range(n)],
            "Reason": [CANARY if i < 14 else ("Illness", "Appointment", "Family Trip")[i % 3] for i in range(n)]}


@pytest.mark.parametrize("flags", [{"exclude": ["student_id"]}, {"null": ["student_id"]},
                                   {"force_factor": ["student_id"]}, {"force_text": ["student_id"]},
                                   {"force_type": ["student_id=FACTOR"]}, {"pii_col": ["student_id=zip"]}])
def test_flags_on_the_id_column_still_count_its_people(flags: dict[str, list[str]]) -> None:
    # attack35_2 leak 4: --exclude/--null/--force-* on the ID column made D35 count rows
    tp, cols = run_table(_absences(), **flags)
    p = profile_of(tp, "Reason")
    assert CANARY in dtext.pseudonymized(p), flags
    assert CANARY not in set(cols["Reason"].tolist())
    assert Counter(cols["Reason"].tolist())["Family Trip"] == 136  # held by many students: kept


def test_no_default_ids_still_counts_a_person_id_header() -> None:
    tp, cols = run_table(_absences(), no_default_ids=True)
    assert profile_of(tp, "student_id").type is T.IDENTIFIER  # student + id: a person-ID header (D10)
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason"))


def test_one_mis_keyed_file_never_clears_a_value_everywhere(tmp_path: Path) -> None:
    # attack35_2 leak 2: a teacher-keyed roster (12 teachers, one student) vouched for the value in every file
    src = tmp_path / "in"
    roster = _roster("TeacherID", names=False)
    write_csv(src / "roster.csv", roster)
    write_csv(src / "students.csv", {"student_id": [str(90000 + i) for i in range(60)],
                                     "Accommodation": [ACCOM if i == 0 else ("Extended Time", "Small Group")[i % 2]
                                                       for i in range(60)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Zyxwell",), out, rep, console=res.output) == []
    fake = {r["Accommodation"] for r in read_dicts(out / "students.csv")} - {"Extended Time", "Small Group"}
    assert len(fake) == 1 and Counter(r["Accommodation"] for r in read_dicts(out / "roster.csv"))[fake.pop()] == 12


@pytest.mark.parametrize("header", ["Pupil_ID", "UPN", "LASID", "Learner_Number"])
def test_unlisted_student_id_headers_are_person_ids(header: str) -> None:
    # attack35_2 leak 5: a student ID repeated on 5-15 rows was never unique, so never an ID candidate
    rng = random.Random(3)
    ids = [str(1001 + s) for s in range(60) for _ in range(rng.randint(5, 15))]
    tp, cols = run_table({header: ids, "Mark": [("X", "/", "L")[i % 3] for i in range(len(ids))]})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id"
    assert not set(cols[header].tolist()) & set(ids)


def test_numbers_under_a_person_header_are_never_written() -> None:
    # attack35_2 leak 6: staff numbers under Teacher / Counselor / Case_Manager were INTEGER levels
    rng = random.Random(8)
    staff = [str(rng.randint(40000, 49999)) for _ in range(30)]
    n = 400
    cols = {"student_id": [str(7000 + i) for i in range(n)],
            "Teacher": [staff[i % 30] for i in range(n)], "Counselor": [staff[i % 8] for i in range(n)],
            "Case_Manager": [staff[(i * 7) % 12] for i in range(n - 1)] + ["48888"],
            "Students": [str(18 + i % 9) for i in range(n)]}
    tp, out = run_table(cols)
    for name in ("Teacher", "Counselor", "Case_Manager"):
        p = profile_of(tp, name)
        assert p.type is T.HIGH_CARD_CATEGORICAL and p.params.get("_numeric_mask"), name
        assert not set(out[name].tolist()) & set(staff + ["48888"]), name
        assert len(set(out[name].tolist())) == len(set(cols[name])), name  # distinct count kept
        assert all(v.isdigit() and len(v) == 5 for v in out[name].tolist()), name
    assert profile_of(tp, "Students").type is T.INTEGER  # a plural role word counts people: a number
    assert profile_of(tp, "Students").params.get("_numeric_mask") is None


def test_bare_phone_numbers_are_masked_whatever_the_header() -> None:
    # attack35_2 leak 8: 10-digit phone numbers under 'Ph' / 'Number' kept their exact min and max (D14)
    rng = random.Random(5)
    phones = [f"{rng.randint(2, 9)}{rng.randint(0, 99):02d}{rng.randint(2, 9)}{rng.randint(0, 999999):06d}"
              for _ in range(120)]
    epochs = [str(1_700_000_000 + i * 3607) for i in range(120)]
    ids = [str(1_000_000_000 + i * 17) for i in range(120)]
    tp, out = run_table({"Ph": phones, "Number": phones[::-1], "Logged": epochs, "Ref": ids})
    for name in ("Ph", "Number"):
        p = profile_of(tp, name)
        if name == "Ph":  # hardening pass 3: a bare 'Ph' is a contact header: its phone numbers are PHONE values
            assert p.type is T.PHONE, name
        else:
            assert p.type is T.HIGH_CARD_CATEGORICAL and p.params.get("pii_shapes") == ["phone"], name
        assert not set(out[name].tolist()) & set(phones), name
    assert dtext.nanp_phone_values(Counter(epochs)) == {} and dtext.nanp_phone_values(Counter(ids)) == {}


def test_only_words_with_letters_vouch_for_a_section() -> None:
    # attack35_2 leak 9: room numbers 1-12 made 'Senior', 'Winter', 'Low' vocabulary; letters B/H made 'White' a race
    assert not vocab.context_domains([str(i) for i in range(1, 13)] + ["Q1", "P3", "B", "H", "W"])
    assert vocab.is_vocabulary("Senior", vocab.context_domains(["Freshman", "Sophomore"]))
    assert vocab.is_vocabulary("White", vocab.context_domains(["WH", "HI", "AS"]))
    assert vocab.is_vocabulary("Winter", vocab.context_domains(["Fall", "Spring"]))
    sid = [f"S{1000 + i}" for i in range(N)]
    words = ["Senior", "Good", "Winter", "Masters", "Summer", "Low", "Fair", "Major"]
    room = [words[i] if i < len(words) else str(1 + i % 12) for i in range(N)]
    letters = ["White" if i == 0 else "Black" if i == 1 else "BHCD"[i % 4] for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Homeroom": room, "Pod": letters})
    assert set(dtext.pseudonymized(profile_of(tp, "Homeroom"))) == set(words)
    assert set(dtext.pseudonymized(profile_of(tp, "Pod"))) == {"White", "Black"}
    assert not set(words) & set(cols["Homeroom"].tolist()) and not {"White", "Black"} & set(cols["Pod"].tolist())


@pytest.mark.parametrize("value", ["Holiday", "Montessori", "Poor", "Doctor", "Dance", "Tuesday", "Sunday", "Daya",
                                   "Daye", "Tiera", "MENA", "AU", "TU", "IP", "AN", "ED", "Karen"])
def test_names_among_vocabulary_words_need_their_section(value: str) -> None:
    # attack35_2 leak 10: vocabulary words that are names, joined phrases and all-caps surnames
    assert not vocab.is_vocabulary(value)
    assert not dtext.unit_kept(value, holders=1, min_cell=10)


def test_names_among_vocabulary_words_with_their_section() -> None:
    for value, section in (("Holiday", ["Present", "Absent"]), ("Montessori", ["Magnet", "STEM"]),
                           ("Poor", ["Excellent", "Satisfactory"]), ("Doctor", ["Tardy", "Excused"]),
                           ("Dance", ["Music", "Choir"]), ("Tuesday", ["Thursday", "Saturday"]),
                           ("Sunday", ["Saturday", "Monday"]), ("ED", ["SLD", "OHI", "AUT"]),
                           ("Karen", ["Spanish", "Somali"]), ("MENA", ["Asian", "Hispanic"])):
        assert vocab.is_vocabulary(value, vocab.context_domains(section)), value
    for joined in ("EconDis", "Grade9", "Tier1", "GenEd"):
        assert vocab.is_vocabulary(joined), joined


def test_a_name_in_either_position_is_still_a_pair() -> None:
    # attack35_2 leak 11: 'Scott, Ann' / 'Lee, Scott': a name first in one value and second in another
    spec = [("Lee, James", 50), ("James, Thomas", 45), ("Thomas, Lee", 44), ("Lee, Ann", 43),
            ("Allen, Maria", 42), ("Scott, Maria", 38), ("Allen, Ann", 37), ("Scott, Ann", 1)]
    vals = [v for v, c in spec for _ in range(c)]
    random.Random(2).shuffle(vals)
    assert dtext.pair_layout(Counter(vals), ",")
    tp, cols = run_table({"student_id": [str(8000 + i) for i in range(len(vals))], "Authorized_Pickup": vals})
    p = profile_of(tp, "Authorized_Pickup")
    assert p.type is not T.MULTI_SELECT and "Scott, Ann" in dtext.pseudonymized(p)
    assert "Scott, Ann" not in set(cols["Authorized_Pickup"].tolist())
    assert not dtext.pair_layout(Counter({"ELL;SPED": 40, "ELL": 30, "SPED;GT;ELL": 20, "SPED": 27}), ";")


def test_code_table_follows_the_fact_table_under_another_name(tmp_path: Path) -> None:
    # attack35_2 utility 1: programs.csv::Code and enrollments.csv::Program_Code are one field: the join still matches
    src = tmp_path / "in"
    codes = ["ESL-A1", "GT-ENR", "SPED-R", "TUT-AM", "TUT-PM", "BND-JZ", "ROB-CL", "CHS-CL"]
    write_csv(src / "programs.csv", {"Code": codes, "Credits": [str(i + 1) for i in range(8)]})
    n = 480
    write_csv(src / "enrollments.csv", {"student_id": [str(1000 + i // 2) for i in range(n)],
                                        "Program_Code": [codes[i % 8] for i in range(n)],
                                        "Enroll_Date": [f"2024-09-{1 + i % 28:02d}" for i in range(n)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert {r["Code"] for r in read_dicts(out / "programs.csv")} == set(codes)
    assert {r["Program_Code"] for r in read_dicts(out / "enrollments.csv")} == set(codes)
    assert leak_check(rep, "programs.csv") == leak_check(rep, "enrollments.csv") == "PASS"


def test_lookup_fields_need_most_of_the_codes() -> None:
    def col(name: str, values: list[str]) -> Any:
        tp, _ = run_table({name: values}, min_cell=1000)  # nothing kept for its count: the decision stays open
        return profile_of(tp, name)

    codes = ["QA1", "QB2", "QC3", "QD4", "QE5", "QF6", "QG7", "QH8"]
    code = col("Code", codes)
    use = col("Program_Code", codes[:4] * 30)
    few = col("Track_Code", codes[:3] * 40)
    other = col("Room_Code", codes[:4] * 29 + ["QZ9"] * 4)
    for p in (code, use, few, other):
        p.params[dtext.PENDING_KEY] = True
    assert dtext.lookup_fields([code, use]) == {"programcode": "code"} == dtext.lookup_fields([use, code])
    assert dtext.lookup_fields([code, few]) == {}  # 3 of 8 codes: fewer than half
    assert dtext.lookup_fields([code, other]) == {}  # a level the code table does not list
    assert dtext.lookup_fields([use, few]) == {}  # Program_Code holds each code on 30 rows: no code table
    code.params["_personal_header"] = True
    assert dtext.lookup_fields([code, use]) == {}


@pytest.mark.parametrize("value", ["National Guard", "Reserve", "Active Duty", "DR", "CT", "DAEP Placement", "Mam",
                                   "Oromo", "Tigrinya", "Eastern", "Western", "Coast", "Central", "Northeast",
                                   "Rural", "City: Large", "Suburb: Midsize"])
def test_more_standard_words(value: str) -> None:
    # attack35_2 utility 3-4: SIS codes and the region words of example-data
    assert vocab.is_vocabulary(value) and dtext.unit_kept(value, holders=1, min_cell=10)


def test_a_constant_initial_is_no_derived_template() -> None:
    from datanonymous.profile import templates

    rng = random.Random(4)
    a = [str(rng.randint(40000, 49999)) for _ in range(60)]
    b = [str(rng.randint(40000, 49999)) for _ in range(60)]
    assert templates.infer_template(a, {1: b}, 10, False) is None
    names = [f"{x}{y}" for x, y in zip("ABCDEFGHIJKL" * 5, range(60))]
    assert templates.infer_template([n[0].lower() + "123" for n in names], {1: names}, 10, False) is not None


# =========================================================================== hardening pass 3
ALLERGY = "Quoll pollen Vexmoor"
_FIRSTS = ("Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Gray", "Harper", "Indy", "Jules", "Kendall", "Logan")
_LASTS = ("Abernathy", "Bellweather", "Castellano", "Dunmore", "Eastwick", "Fairbanks", "Galloway", "Hollister",
          "Ingersoll", "Kingsbury", "Lockridge", "Mortensen")
_COMMON = ("Peanut", "Latex", "Bee Sting")


def _students_file(n: int = 60) -> dict[str, list[str]]:
    """One row per student (Student_ID); the first student alone holds ALLERGY."""
    return {"Student_ID": [str(70000 + i) for i in range(n)],
            "Allergy": [ALLERGY if i == 0 else _COMMON[i % 3] for i in range(n)]}


def _staff_value(kind: str, k: int) -> str:
    first, last = _FIRSTS[k], _LASTS[k]
    return {"full": f"{first} {last}", "lastfirst": f"{last}, {first}", "user": f"{first[0]}{last}".lower(),
            "email": f"{first}.{last}@district.org".lower()}[kind]


def _staff_log(header: str, kind: str, n_students: int = 30, per: int = 12) -> dict[str, list[str]]:
    """No student key: ``n_students`` x ``per`` rows, each entered by one of 12 staff members (``header``); the first
    student's 12 rows hold ALLERGY."""
    rows = n_students * per
    return {"Visit_Date": [f"2024-{1 + i % 12:02d}-{1 + (i * 7) % 28:02d}" for i in range(rows)],
            header: [_staff_value(kind, i % per) for i in range(rows)],
            "Allergy": [ALLERGY if i < per else _COMMON[(i // per) % 3] for i in range(rows)]}


@pytest.mark.parametrize("header,kind", [
    ("Entered_By", "full"), ("Referred_By", "full"), ("Last_Modified_By", "lastfirst"), ("EnteredBy", "full"),
    ("Created_By_Email", "email"), ("UserName", "user"), ("Provider_Name", "full"), ("Interpreter_Name", "full"),
    ("Sponsor_Name", "full"), ("Name", "full"), ("Email", "email")])
def test_staff_names_never_prove_people(tmp_path: Path, header: str, kind: str) -> None:
    # attack35_3 leak 1: a log without a student key counted the staff who wrote its rows as people, and that count
    # cleared a one-student value in every file of the run (run_holders' people-over-rows rule)
    src = tmp_path / "in"
    write_csv(src / "students.csv", _students_file())
    write_csv(src / "log.csv", _staff_log(header, kind))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert found(("Vexmoor",), out, rep, console=res.output) == []
    fake = {r["Allergy"] for r in read_dicts(out / "students.csv")} - set(_COMMON)
    assert len(fake) == 1 and Counter(r["Allergy"] for r in read_dicts(out / "log.csv"))[fake.pop()] == 12
    assert leak_check(rep, "students.csv") == leak_check(rep, "log.csv") == "PASS"


def test_own_names_and_headers() -> None:
    from datanonymous.profile import entity

    for name in ("Entered_By", "EnteredBy", "Last_Modified_By", "Created_By_Email", "UserName", "USERNAME",
                 "Provider_Name", "Interpreter_Name", "Sponsor_Name", "Teacher", "Guardian_Email", "Assigned_To"):
        assert entity.other_person_header(name), name
    for name in ("Name", "LastFirst", "Email", "Student_Name", "Stu_Last_Nm", "First_Name", "Preferred_Name"):
        assert not entity.other_person_header(name), name
    for name in ("Student_Name", "StudentName", "STUDENTNAME", "Stu_Last_Nm", "StuEmail", "Pupil_Email", "Child_Name"):
        assert entity.subject_named_header(name), name
    for name in ("Name", "LastFirst", "Email", "Study_Name", "First_Name"):
        assert not entity.subject_named_header(name), name


@pytest.mark.parametrize("name,role,flag", [
    ("Personnel_ID", "other", None), ("PersonnelNumber", "other", None), ("Student_Test_ID", "record", None),
    ("StudentSectionKey", "record", None), ("Student_Record_ID", "record", None),
    ("Student_Enrollment_ID", "record", None), ("State_Student_ID", "subject", None),
    ("Student_Local_ID", "subject", None), ("Perm ID", "subject", None), ("PermID", "subject", None),
    ("GTID", "subject", None), ("StudentUSI", "subject", None), ("TSDS ID", "subject", None),
    ("Stu_Num", "subject", None), ("Stu #", "subject", None), ("Student #", "subject", None),
    ("Unique ID", "record", None), ("StaffUSI", "other", None), ("Study_ID", "record", "id_col"),
    ("Stuff_Number", "record", "id_col"), ("Student_Key", "subject", "id_col"), ("Personnel_ID", "other", "id_col")])
def test_id_roles_pass_3(name: str, role: str, flag: str | None) -> None:
    # attack35_3 leaks 3-4: whole words only (Personnel, Study), a student's record keys number records, and the
    # student-ID headers of Aeries, Georgia, Ed-Fi and Texas are IDs of the row's own person
    from test_area_a_support import detect

    from datanonymous.profile import entity

    det = detect([f"{1000 + i // 3}" for i in range(30)], name, **({flag: [name]} if flag else {}))
    assert det.type is T.IDENTIFIER, name
    p = type("P", (), {"name": name, "source": det.source, "type": det.type})()
    assert entity.id_role(p) == role, name


def _cross(tmp_path: Path, name: str, log: dict[str, list[str]], *extra: str) -> None:
    """students.csv (one student holds ALLERGY) beside ``name``: ALLERGY is written nowhere."""
    src = tmp_path / "in"
    write_csv(src / "students.csv", _students_file())
    write_csv(src / name, log)
    res, out, rep = run(src, tmp_path, *extra)
    assert res.exit_code == 0, res.output
    assert found(("Vexmoor",), out, rep, console=res.output) == []


@pytest.mark.parametrize("key", ["Personnel_ID", "Student_Test_ID", "StudentSectionKey", "Study_ID"])
def test_record_and_staff_keys_never_prove_people(tmp_path: Path, key: str) -> None:
    rows = 360
    log = {key: [str(4100 + i % 12) if key == "Personnel_ID" else str(880000 + i) for i in range(rows)],
           "Allergy": [ALLERGY if i < 12 else _COMMON[(i // 12) % 3] for i in range(rows)]}
    _cross(tmp_path, "log.csv", log, *(["--id-col", key] if key == "Study_ID" else []))


def _visits(key: str, n_once: int = 100, visits: int = 12) -> dict[str, list[str]]:
    """``key`` numbers visits: ``n_once`` students with one visit, the first one with ``visits`` visits (ALLERGY)."""
    names = [f"{last}, {first}" for last in _LASTS for first in _FIRSTS][: n_once + 1]
    student = [0] * visits + list(range(1, n_once + 1))
    return {key: [str(5000 + i) for i in range(len(student))], "Student_Name": [names[s] for s in student],
            "Allergy": [ALLERGY if s == 0 else _COMMON[s % 3] for s in student]}


@pytest.mark.parametrize("key", ["SID", "Student_Test_ID", "Student_Enrollment_ID"])
def test_an_id_that_numbers_visits_never_hides_the_names(key: str) -> None:
    # attack35_3 leak 2: 100 students with one visit and one with 12 made the visit number look one to one with the
    # names (>= 90% of the names held one ID), so the names were dropped and 12 visits were 12 holders
    tp, cols = run_table(_visits(key))
    assert ALLERGY in dtext.pseudonymized(profile_of(tp, "Allergy")) and ALLERGY not in set(cols["Allergy"].tolist())
    assert Counter(cols["Allergy"].tolist())["Latex"] == 34  # held by 34 students: kept


def test_an_id_that_numbers_visits_never_clears_a_value_elsewhere(tmp_path: Path) -> None:
    _cross(tmp_path, "nurse_visits.csv", _visits("SID"))


def test_blank_student_ids_never_take_a_record_number(tmp_path: Path) -> None:
    # attack35_3 leak 5: 12 rows with a blank Student_ID took 12 record numbers and counted 12 people
    rows = 360
    cols = {"id": [str(1 + i) for i in range(rows)],
            "Student_ID": ["" if i < 12 else str(70000 + i // 12) for i in range(rows)],
            "Allergy": [ALLERGY if i < 12 else _COMMON[(i // 12) % 3] for i in range(rows)]}
    tp, out = run_table(cols)
    assert ALLERGY in dtext.pseudonymized(profile_of(tp, "Allergy"))
    _cross(tmp_path, "incidents.csv", cols)


@pytest.mark.parametrize("header", ["Perm ID", "PermID", "GTID", "StudentUSI", "TSDS ID", "Unique ID", "Stu #",
                                    "Stu_Num", "Student #"])
def test_more_student_id_headers_are_ids(header: str) -> None:
    # attack35_3 leak 3: the column was an interpolated INTEGER (every real ID written) and D35 counted rows
    rng = random.Random(3)
    ids = [str(4810000 + s * 37) for s in range(60) for _ in range(rng.randint(5, 15))]
    ids[:12] = [ids[0]] * 12
    reason = [ALLERGY if i < 12 else ("Illness", "Appointment", "Family Trip")[i % 3] for i in range(len(ids))]
    tp, cols = run_table({header: ids, "Reason": reason})
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id", header
    assert not set(cols[header].tolist()) & set(ids)
    assert ALLERGY in dtext.pseudonymized(profile_of(tp, "Reason"))


def test_id_like_integer_columns_get_id_advice() -> None:
    from test_area_a_support import detect

    rng = random.Random(3)
    ids = [str(4810000 + s * 37) for s in range(60) for _ in range(rng.randint(5, 15))]
    det = detect(ids, "Locker_ID")
    msg = next(m for m in det.messages if m.code == "possible_sparse_code")
    # never --force-factor first; K-04: --id-col for person or record IDs, --person-id only for the row's own person
    assert det.type is T.INTEGER and msg.fix.startswith("--id-col Locker_ID")


@pytest.mark.parametrize("value", ["Thai", "Newcomer", "Blank", "Kinder", "Monday", "Wednesday", "Main", "Charter",
                                   "AI"])
def test_more_names_among_vocabulary_words_need_their_section(value: str) -> None:
    # attack35_3 leak 6: surnames and given names among the vocabulary words, unused dictionary entries and Stata
    # labels included (io.arrowio.vocabulary_label)
    from datanonymous.io import arrowio

    assert not vocab.is_vocabulary(value) and not arrowio.vocabulary_label(value)
    assert not dtext.unit_kept(value, holders=2, min_cell=10)


def test_more_names_among_vocabulary_words_with_their_section() -> None:
    for value, section in (("Thai", ["Spanish", "Somali"]), ("Newcomer", ["ELL", "Bilingual"]),
                           ("Blank", ["Missing", "Unknown"]), ("Kinder", ["Freshman", "Pre-K"]),
                           ("Monday", ["Thursday", "Saturday"]), ("Wednesday", ["Thursday", "Saturday"]),
                           ("Main", ["Active", "Withdrawn"]), ("Charter", ["Active", "Home School"]),
                           ("AI", ["AS", "WH", "HI"])):
        assert vocab.is_vocabulary(value, vocab.context_domains(section)), value
    rare = ["Thai", "Newcomer", "Blank", "Kinder", "Monday", "Main", "Wednesday", "Charter", "AI"]
    home = ["Rm 101" if i % 2 else "Rm 102" for i in range(N)]
    for i, v in enumerate(rare):
        home[3 + i * 7] = v
    tp, cols = run_table({"Student_ID": [str(70000 + i) for i in range(N)], "Homeroom": home})
    assert set(dtext.pseudonymized(profile_of(tp, "Homeroom"))) == set(rare)
    assert not set(rare) & set(cols["Homeroom"].tolist())


def test_local_phone_numbers_under_a_bare_ph_header() -> None:
    # attack35_3 leak 7: seven-digit numbers under 'Ph' were an INTEGER column whose exact min and max were written
    from datanonymous import hints

    rng = random.Random(9)
    local = [f"{rng.randint(2, 9)}{rng.randint(0, 999999):06d}" for _ in range(N)]
    acid = [f"{rng.uniform(5.5, 8.5):.1f}" for _ in range(N)]
    tp, out = run_table({"student_id": [f"S{1000 + i}" for i in range(N)], "Ph": local, "pH": acid})
    assert profile_of(tp, "Ph").type is T.PHONE
    assert not set(out["Ph"].tolist()) & set(local)
    assert profile_of(tp, "pH").type is T.FLOAT
    assert hints.phone_header_hint("Ph") == hints.phone_header_hint("MOB") == "contact"
    assert hints.phone_header_hint("pH") is None and hints.phone_header_hint("Num_Ph") is None
    assert not dtext.contact_phone("5550187") and dtext.contact_phone("5550187", local=True)


@pytest.mark.parametrize("value", [
    "American Indian or Alaskan Native", "Native Hawaiian or Pacific Islander", "White, non-Hispanic",
    "Black, Non-Hispanic", "Hispanic/Latino of any race", "Two or More Races, Non-Hispanic", "Declined to Respond",
    "uri://ed-fi.org/RaceDescriptor#Asian", "uri://ed-fi.org/SexDescriptor#Female",
    "uri://ed-fi.org/GradeLevelDescriptor#Ninth grade", "Deaf/Hard of Hearing", "Speech Language Impairment",
    "Learning Disabled", "Cognitive Disability", "Emotional Behavioral Disability", "Speech Impaired",
    "Significant Developmental Delay", "Masters Grade Level", "Meets Grade Level", "Approaches Grade Level",
    "Did Not Meet Grade Level", "LoAvg", "Avg", "HiAvg", "Not Applied", "Free Meals", "Free - CEP", "Dyslexia",
    "Title I Part A", "Neglected or Delinquent", "Hospital/Homebound", "Fall 2023", "Spring 2024", "Novice",
    "Apprentice", "Exemplary", "Highly Proficient", "Beginning Learner", "Defiance/Insubordination",
    "Dress Code Violation", "Cell Phone Violation"])
def test_standard_spellings_pass_3(value: str) -> None:
    # attack35_3 utility 1-2: race, disability, state-assessment, lunch, program, term and discipline spellings
    assert vocab.is_vocabulary(value) and dtext.unit_kept(value, holders=1, min_cell=10)


def test_descriptor_uris_judge_their_code_value() -> None:
    assert vocab.descriptor_code("uri://ed-fi.org/RaceDescriptor#White") == "White"
    assert not vocab.is_vocabulary("uri://ed-fi.org/RaceDescriptor#White")  # a name-like code needs its section
    race = vocab.context_domains(["uri://ed-fi.org/RaceDescriptor#Asian"])
    assert vocab.is_vocabulary("uri://ed-fi.org/RaceDescriptor#White", race)
    assert not vocab.is_vocabulary("uri://ed-fi.org/RaceDescriptor#Okonjo")


def test_a_code_reads_its_section_from_its_header_or_its_description() -> None:
    # attack35_3 utility 3: single-letter codes vouch for no section, so a rare 'ED' (early dismissal) beside them
    # was pseudonymised; its header or the column describing it now names the section
    meaning = {"A": "Absent", "T": "Tardy", "E": "Excused", "U": "Unexcused", "X": "Exempt", "M": "Medical",
               "S": "Suspended", "I": "Illness", "B": "Bereavement", "R": "Religious", "C": "Court Appearance",
               "ED": "Early Dismissal"}
    letters = "ATEUXMSIBRC"
    codes = [letters[i % len(letters)] for i in range(N - 3)] + ["ED"] * 3
    sid = [f"S{1000 + i}" for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Code": codes})
    assert set(dtext.pseudonymized(profile_of(tp, "Code"))) == {"ED"}
    tp, cols = run_table({"student_id": sid, "Attendance_Code": codes})
    assert dtext.pseudonymized(profile_of(tp, "Attendance_Code")) == {}
    assert Counter(cols["Attendance_Code"].tolist())["ED"] == 3
    tp, cols = run_table({"student_id": sid, "Code": codes, "Description": [meaning[c] for c in codes]})
    assert dtext.pseudonymized(profile_of(tp, "Code")) == {} and Counter(cols["Code"].tolist())["ED"] == 3
    # a homeroom beside a race column describes nothing (not one to one): a teacher's surname stays a name
    home = ["White" if i < 2 else ("Okonjo", "Garcia", "Lee")[i % 3] for i in range(N)]
    race = [("Asian", "Hispanic", "Black or African American", "Two or More Races")[i % 4] for i in range(N)]
    tp, cols = run_table({"student_id": sid, "Homeroom": home, "Race": race})
    assert set(dtext.pseudonymized(profile_of(tp, "Homeroom"))) == {"White"}
    # ... nor in a tiny table where they happen to be one to one: a column of surnames is no code column
    tp, cols = run_table({"student_id": ["S1", "S2", "S3", "S4"], "Homeroom": ["White", "Okonjo", "Garcia", "Lee"],
                          "Race": ["Asian", "Hispanic", "Black or African American", "Two or More Races"]})
    assert "White" in dtext.pseudonymized(profile_of(tp, "Homeroom"))
    assert vocab.header_domains("Teacher_Language") == frozenset()
    assert vocab.header_domains("Home_Language") == {"#languages"}


def test_small_code_lists_under_post_words_stay_numbers() -> None:
    # attack35_3 utility 4: Transportation_Provider (1-4), Lunch_Provider (10-12) and Officer (0-4) are codes
    rng = random.Random(8)
    n = 200
    staff = [str(rng.randint(40000, 49999)) for _ in range(12)]
    cols = {"student_id": [str(7000 + i) for i in range(n)],
            "Transportation_Provider": [str(1 + i % 4) for i in range(n)],
            "Lunch_Provider": [str(10 + i % 3) for i in range(n)], "Officer": [str(i % 5) for i in range(n)],
            "Case_Manager": [staff[i % 12] for i in range(n)]}
    tp, out = run_table(cols)
    for name in ("Transportation_Provider", "Lunch_Provider", "Officer"):
        assert profile_of(tp, name).params.get("_numeric_mask") is None, name
        assert Counter(out[name].tolist()) == Counter(cols[name]), name
    assert profile_of(tp, "Case_Manager").params.get("_numeric_mask")
    assert not set(out["Case_Manager"].tolist()) & set(staff)


def test_counts_without_a_person_key_are_flagged() -> None:
    # attack35_3 other: a table whose keys number teachers or records (a6) counts one student's rows as several
    # holders; the report says so with the fix
    tp, _ = run_table(_roster("TeacherID", names=False))
    p = profile_of(tp, "Accommodation")
    msg = next(m for m in p.messages if m.code == "d35_rows_counted")
    assert msg.level == "warning" and msg.fix.startswith("--person-id")
    # a student ID, or a name or e-mail address on every row, shows the rows are people: no warning
    n = 200
    club = [("Chess Guild", "Robotics Guild")[i % 2] for i in range(n)]
    for cols in ({"student_id": [str(9000 + i) for i in range(n)], "Club": club},
                 {"id": [str(1 + i) for i in range(n)], "email": [f"kid{i}@school.org" for i in range(n)],
                  "Club": club}):
        tp, _ = run_table(cols)
        assert not any(m.code == "d35_rows_counted" for m in profile_of(tp, "Club").messages), list(cols)
