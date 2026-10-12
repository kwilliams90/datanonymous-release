"""Round-1 adversarial review, names/PII cluster (R1-02, R1-04, R1-05, R1-06).

Each test pins down one way a real person's name reached the output, the out-dir docs or the report:

* R1-02: a short leftover in a date column ("Zelmira Quenby") was written verbatim as a placeholder token.
* R1-04: synthetic first + last names (or a synthetic full name) could spell a real full name held by
  another file (or another column) of the run.
* R1-05: ``student_first`` / ``student_last`` headers were no name hint, so a small name column became a
  factor whose real levels were copied.
* R1-06: a headerless list of names was read with its first person as the header row.

Assertion messages never print a real value beyond the invented fixture names.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from conftest import read_dicts, run_cli
from datanonymous import hints
from datanonymous.detect import dates as D
from datanonymous.generate import pii
from datanonymous.io import sniff

REAL = ("Zelmira", "Quenby", "Balthazar", "Vosswyn", "Mirelda", "Xaverton", "Lysander", "Wexforth")


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _texts(*dirs: Path) -> dict[str, str]:
    return {str(p): p.read_text(encoding="utf-8", errors="replace")
            for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()}


def _assert_absent(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> None:
    texts = _texts(*dirs)
    texts["console"] = console
    found = {name: [w for w in words if w in text] for name, text in texts.items()}
    found = {k: v for k, v in found.items() if v}
    assert not found, found


def _run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}_out", tmp / f"{name}_rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "123", *extra)
    return res, out, rep


def _hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- R1-02 date placeholders
@pytest.mark.parametrize("value", ["Zelmira Quenby", "Zelmira", "Quenby, Z", "Jan Smith", "May June", "Zoë"])
def test_names_left_over_in_a_date_column_are_not_placeholders(value: str) -> None:
    assert D.leftover_kinds([value], ["%Y-%m-%d"]) == [None]
    assert D.leftover_kinds([value], ["%m/%d/%Y"]) == [None]


@pytest.mark.parametrize("value", ["TBD", "Unknown", "N/A", "n/a", "Not enrolled", "Pending review", "MM/DD/YYYY",
                                   "#VALUE!", "0", "000", "-", "?", "None"])
def test_known_placeholders_stay_tokens(value: str) -> None:
    assert D.leftover_kinds([value], ["%Y-%m-%d"]) == ["word"]


def test_name_in_a_date_column_is_never_copied(tmp_path: Path) -> None:
    # 250 distinct dates: the two leftovers stay under 1% of the distinct values (the column is a DATE)
    rows = [[(dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat(), i % 20] for i in range(250)]
    rows.append(["Zelmira Quenby", 17])
    rows.append(["TBD", 3])
    src = _write(tmp_path / "in" / "events.csv", ["event_date", "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "events.csv")
    assert len(syn) == len(rows)
    assert sum(r["event_date"] == "TBD" for r in syn) == 1  # a known placeholder keeps its count
    _assert_absent(["Zelmira", "Quenby"], out, rep, console=res.output)
    col = {c["name"]: c for c in json.loads((rep / "files" / "events.csv.json").read_text(encoding="utf-8"))
           ["tables"][0]["columns"]}["event_date"]
    assert col["type"] == "DATE"
    assert col["params"]["other_tokens"] == 1 and col["params"]["unparsed_replaced"] == 1


def test_na_value_keeps_an_unknown_placeholder(tmp_path: Path) -> None:
    rows = [[(dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat()] for i in range(150)] + [["Desconocido"]] * 2
    src = _write(tmp_path / "in" / "d.csv", ["fecha"], rows)
    res, out, _ = _run(src, tmp_path, name="plain")
    assert res.exit_code == 0, res.output
    assert not any(r["fecha"] == "Desconocido" for r in read_dicts(out / "d.csv"))  # unknown word: replaced
    res, out, _ = _run(src, tmp_path, "--na-value", "Desconocido", name="na")
    assert res.exit_code == 0, res.output
    # R8-05: declaring a missing value is no permission to publish it: held by 2 rows (< --min-cell), the token is
    # written as one consistent same-shape pseudonym, still a missing value
    syn = [r["fecha"] for r in read_dicts(out / "d.csv")]
    fakes = {v for v in syn if not v[:1].isdigit()}
    assert "Desconocido" not in syn and len(fakes) == 1 and syn.count(fakes.pop()) == 2
    res, out, _ = _run(src, tmp_path, "--na-value", "Desconocido", "--min-cell", "2", name="na2")
    assert res.exit_code == 0, res.output
    assert sum(r["fecha"] == "Desconocido" for r in read_dicts(out / "d.csv")) == 2  # held by >= k: kept verbatim


# --------------------------------------------------------------------------- R1-05 name-part headers
@pytest.mark.parametrize("header,key", [
    ("student_first", "first_name"), ("student_last", "last_name"), ("StuFirstNm", "first_name"),
    ("stu_last_nm", "last_name"), ("Guardian1First", "first_name"), ("Guardian_1_Last", "last_name"),
    ("First", "first_name"), ("LAST", "last_name"), ("Middle", "middle_name"), ("LegalFirst", "first_name"),
    ("Preferred_First", "first_name"), ("Teacher_Last", "last_name"), ("FirstInitial", "first_name"),
    ("LastFirst", "full_name"), ("LASTFIRST", "full_name"), ("Student_LastFirst", "full_name"),
])
def test_name_part_headers_are_name_hints(header: str, key: str) -> None:
    assert hints.header_hints(header).pii == key


@pytest.mark.parametrize("header", ["first_day", "last_login", "LastSchool", "is_first", "first_gen", "first_time",
                                    "FirstLanguage", "Last_Updated", "last_date", "firstlast_flag", "School",
                                    "Student_ID"])
def test_other_first_last_headers_are_not_name_hints(header: str) -> None:
    assert hints.person_name_part_hint(header) is None


def test_student_first_last_names_are_never_copied(tmp_path: Path) -> None:
    rows = [["Zelmira", "Quenby", i] if i % 2 else ["Balthazar", "Vosswyn", i] for i in range(30)]
    src = _write(tmp_path / "in" / "roster.csv", ["student_first", "student_last", "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "roster.csv")
    assert len(syn) == 30
    assert len({r["student_first"] for r in syn}) == 2 and len({r["student_last"] for r in syn}) == 2
    _assert_absent(list(REAL[:4]), out, rep, console=res.output)
    cols = {c["name"]: c for c in json.loads((rep / "files" / "roster.csv.json").read_text(encoding="utf-8"))
            ["tables"][0]["columns"]}
    assert (cols["student_first"]["type"], cols["student_first"]["subtype"]) == ("PERSON_NAME", "first")
    assert (cols["student_last"]["type"], cols["student_last"]["subtype"]) == ("PERSON_NAME", "last")


# --------------------------------------------------------------------------- R1-06 headerless name lists
def test_headerless_name_list_fails_instead_of_exposing_a_person(tmp_path: Path) -> None:
    src = _write(tmp_path / "in" / "names.csv", ["Zelmira", "Quenby"],
                 [["Balthazar", "Vosswyn"], ["Mirelda", "Xaverton"], ["Lysander", "Wexforth"]])
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert not (out / "names.csv").exists()
    _assert_absent(list(REAL), out, rep, console=res.output)
    entry = next(f for f in json.loads((rep / "report.json").read_text(encoding="utf-8"))["files"]
                 if f["source"] == "names.csv")
    assert entry["status"] == "FAIL"
    assert "--header-row 1" in json.dumps(entry)


def test_header_row_1_asserts_a_plain_word_header(tmp_path: Path) -> None:
    rows = [["Red", "Circle"], ["Blue", "Square"], ["Green", "Triangle"], ["Red", "Square"]] * 5
    src = _write(tmp_path / "in" / "shapes.csv", ["Tint", "Figure"], rows)
    res, _, _ = _run(src, tmp_path, name="refused")
    assert res.exit_code == 3, res.output  # indistinguishable from a record: refused
    res, out, _ = _run(src, tmp_path, "--header-row", "1", name="asserted")
    assert res.exit_code == 0, res.output
    assert (out / "shapes.csv").read_text(encoding="utf-8").splitlines()[0] == "Tint,Figure"
    assert len(read_dicts(out / "shapes.csv")) == len(rows)


@pytest.mark.parametrize("header,rows", [
    (["Region", "Program"], [["North", "Math"], ["South", "Reading"], ["East", "Math"]]),
    (["first_name", "last_name"], [["Balthazar", "Vosswyn"], ["Mirelda", "Xaverton"]]),
    (["Student", "Teacher"], [["Balthazar Vosswyn", "Mirelda Xaverton"], ["Lysander Wexforth", "Ana Ruiz"]]),
    (["Gender", "Race"], [["F", "White"], ["M", "Asian"]]),
    (["Name", "Score"], [["Balthazar", "12"], ["Mirelda", "14"]]),
    (["NAME", "CITY"], [["Balthazar", "Austin"], ["Mirelda", "Dallas"]]),
])
def test_recognisable_headers_are_still_headers(header: list[str], rows: list[list[str]]) -> None:
    assert sniff.header_problems(header, [list(c) for c in zip(*rows)]) is None


def test_record_like_rule() -> None:
    cols = [["Balthazar", "Mirelda", "Lysander"], ["Vosswyn", "Xaverton", "Wexforth"]]
    prob = sniff.header_problems(["Zelmira", "Quenby"], cols)
    assert prob is not None and prob.record_like and "Zelmira" not in prob.text()
    assert sniff.header_problems(["Zelmira"], [cols[0]]) is not None  # a one-column name list
    assert sniff.header_problems(["zelmira", "quenby"], [[v.lower() for v in c] for c in cols]) is not None
    # R2-05: letter case is no evidence of a header, and a numeric column does not vouch for a name cell
    assert sniff.header_problems(["ZELMIRA", "Quenby"], cols) is not None
    assert sniff.header_problems(["Zelmira", "Quenby"], [cols[0], ["12", "15", "9"]]) is not None
    # a recognisable column name over different values does
    assert sniff.header_problems(["Zelmira", "Score"], [cols[0], ["12", "15", "9"]]) is None
    assert not sniff.record_like(["Zelmira"], [])  # no column evidence (Excel title cells)


# --------------------------------------------------------------------------- R1-04 run-wide full names
def _two_runs(tmp_path: Path, people_header: list[str], people_rows: list[list[object]],
              other: str, other_header: list[str], make_other) -> tuple[list[dict], list[dict], Path, Path]:  # noqa: ANN001
    """Run people.csv alone, then again next to ``other`` built from that first synthetic output."""
    d = tmp_path / "in"
    _write(d / "people.csv", people_header, people_rows)
    res, out1, _ = _run(d, tmp_path, name="alone")
    assert res.exit_code == 0, res.output
    first = read_dicts(out1 / "people.csv")
    _write(d / other, other_header, make_other(first))
    res, out2, rep2 = _run(d, tmp_path, name="together")
    assert res.exit_code == 0, res.output
    return first, read_dicts(out2 / "people.csv"), out2, rep2


def test_first_last_pair_never_spells_a_real_full_name_of_another_file(tmp_path: Path) -> None:
    rows = [["Zelmira", "Quenby", i] for i in range(30)]
    first, second, out, rep = _two_runs(
        tmp_path, ["first_name", "last_name", "score"], rows, "staff.csv", ["full_name", "score"],
        lambda syn: [[f"{syn[0]['first_name']} {syn[0]['last_name']}", i] for i in range(30)])
    real_staff = f"{first[0]['first_name']} {first[0]['last_name']}"
    assert {f"{r['first_name']} {r['last_name']}" for r in second}.isdisjoint({real_staff})
    assert len(second) == 30 and len({(r["first_name"], r["last_name"]) for r in second}) == 1
    per_file = (rep / "files" / "people.csv.json").read_text(encoding="utf-8")
    assert '"name_pairs_redrawn"' in per_file  # the count is reported (info), never the name
    _assert_absent([real_staff, first[0]["first_name"] + "," + first[0]["last_name"]], out, rep)
    # the run is still byte-identical for the same seed
    res, out_again, _ = _run(tmp_path / "in", tmp_path, name="again")
    assert res.exit_code == 0
    assert _hashes(out_again) == _hashes(out)


def test_entity_names_and_email_template_follow_the_redraw(tmp_path: Path) -> None:
    """Within-ID constant names stay constant and derived e-mails are rebuilt from the redrawn names."""
    first_names = ["Zelmira", "Balthazar", "Mirelda", "Lysander", "Ottoline", "Peregrin", "Quillon", "Rosamund",
                   "Saffira", "Tobiah"]
    last_names = ["Quenby", "Vosswyn", "Xaverton", "Wexforth", "Ashgrove", "Brindle", "Corwick", "Dunmore",
                  "Elsworth", "Fennimore"]
    rows = []
    for s in range(10):
        f, l = first_names[s], last_names[s]
        for k in range(3):
            rows.append([f"S{1000 + s}", f, l, f"{f.lower()}.{l.lower()}@school.org", k + s])
    first, second, _, _ = _two_runs(
        tmp_path, ["student_id", "first_name", "last_name", "email", "score"], rows, "guardians.csv",
        ["guardian_id", "first_name", "last_name"],
        lambda syn: [[f"G{9000 + i}", r["first_name"], r["last_name"]]
                     for i, r in enumerate({(r["student_id"]): r for r in syn}.values())])
    real_pairs = {(r["first_name"], r["last_name"]) for r in first}  # now the guardians' real names
    assert not real_pairs & {(r["first_name"], r["last_name"]) for r in second}
    by_id: dict[str, set] = defaultdict(set)
    for r in second:
        by_id[r["student_id"]].add((r["first_name"], r["last_name"]))
    assert all(len(v) == 1 for v in by_id.values())  # D4: one name per synthetic student
    assert len({r["first_name"] for r in second}) == 10 and len({r["last_name"] for r in second}) == 10
    for r in second:  # the derived template (D5) uses the redrawn names
        assert r["email"].split("@")[0] == f"{r['first_name'].lower()}.{r['last_name'].lower()}", r
        assert r["email"].endswith(("example.org", "example.com", "example.net"))


def test_full_name_never_equals_a_real_first_last_pair_of_another_file(tmp_path: Path) -> None:
    rows = [["Zelmira Quenby", i] for i in range(20)] + [["Balthazar Vosswyn", i] for i in range(20)]
    first, second, _, _ = _two_runs(
        tmp_path, ["teacher_name", "score"], rows, "students.csv", ["first_name", "last_name"],
        lambda syn: [v.split(" ", 1) for v in dict.fromkeys(r["teacher_name"] for r in syn)])
    real_full = {r["teacher_name"].casefold() for r in first}  # now real students' first + last names
    assert not real_full & {r["teacher_name"].casefold() for r in second}
    assert Counter(Counter(r["teacher_name"] for r in second).values()) == Counter({20: 2})


def test_name_index_direction() -> None:
    """Known roles are one-directional (Gordon Paul is not Paul Gordon); full-name layouts are both ways."""
    index: dict[str, set[str]] = {}
    pii._link(index, "paul", "gordon")
    assert pii._pairs_real(index, ["gordon"], {"paul"})
    assert not pii._pairs_real(index, ["paul"], {"gordon"})
    assert pii._pairs_real(index, ["paul", "gordon"]) and pii._pairs_real(index, ["gordon", "paul"])
    pii._add_full(index, "Burton, Chelsea")
    assert pii._pairs_real(index, ["burton"], {"chelsea"}) and pii._pairs_real(index, ["chelsea"], {"burton"})
    assert pii.person_words("Mrs. Mary-Ann O'Brien Jr") == ["mary", "ann", "brien"]
