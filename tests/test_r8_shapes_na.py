"""Round 8, cluster shapes-na: the personal-data shape test is broad and fail-safe (R8-01, R8-02), and an
``--na-value`` token is no permission to publish text (R8-05).

R8-01 / R8-02, the bug class: the one shape test of a real text unit (``detect.text.unit_shapes``: the D35 copy gate's
never-kept rule, and D34's screen of headers, sheet, file and folder names) enumerated North American phone layouts
and ``+`` numbers, and ASCII e-mail addresses. A domestic number in another national layout (``020 7946 0958``,
``07911 123456``, ``01 23 45 67 89``) or an e-mail address with a letter outside ASCII (``josé@school.org``,
``张伟@school.org``) had no shape, so a factor level held by >= ``--min-cell`` students was copied verbatim and a JSON
key holding one was written as a header. The fix makes the test broad instead of enumerating countries:

* a phone number is any run of two or more digit groups joined by one space, dot, dash or slash (or set in
  parentheses, a ``+`` allowed) with 9-15 digits (a leading trunk ``0`` not counted), unless its groups say it is a
  date, year or clock time, a decimal number, a range or list of ranges, a count, a ZIP+4 code, an IPv4 address, the
  SSN layout (an SSN shape already) or an amount (``detect.text.phone_runs``);
* an e-mail address is any non-space local part, ``@`` and a domain of two or more dot-separated labels whose last
  label starts with a letter of any script.

A false alarm only costs a pseudonym (D35). D34 refuses a file, so a name keeps its documented exceptions: a phone
number in a non-North-American layout is excused after a word of a numbered thing (``Order 020 7946 0958``) or as a
list of numbers (``Cut_200_250_300``, ``05_10_15_20_25``), never after a phone or family word.

R8-05, the bug class: every configured NA token was exempt from the D35 decision (``t in na``), written verbatim in
missing slots, and excused run-wide by the leak scanner (``_kept_keys``). Now a ``--na-value`` token is decided like
any real text unit: kept when it is missing-value vocabulary (the default NA tokens, ``Unknown``, ``Missing``), a
short number or a symbol, or when it has no personal-data shape and >= ``--min-cell`` people hold it; otherwise it is
written as one consistent same-shape pseudonym (the run's pseudonym map), which is itself one of the column's NA
tokens, so missingness and its statistics are kept. The scanner excuses only the NA tokens a column actually writes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.privacy import headers as d34

SEED = "77"

PHONES = ["020 7946 0958", "(020) 7946 0958", "07911 123456", "+44 20 7946 0958", "01 23 45 67 89", "030 1234567",
          "030/1234567", "55 1234 5678", "098765 43210", "98765 43210", "0917 123 4567", "090 123 45 67",
          "0912.345.678", "06 12 34 56 78", "(11) 91234-5678", "03-1234-5678", "010-1234-5678", "0412 345 678",
          "138 0013 8000", "+52 55 1234 5678", "+91 98765 43210", "Call 020 7946 0958 after 3", "Mom: 07911 123456",
          "020 7946 0958,020 7946 0959", "Parent_020_7946_0958", "+44 (0)20 7946 0958"]
EMAILS = ["josé@school.org", "alice@müller.de", "张伟@school.org", "ana@xn--mller-kva.de", "ivan@пример.рф",
          "Front Office;josé@school.org", "ANA.LOPEZ@SCHOOL.ORG"]
#: Ordinary labels and numbers that hold no phone number (R8-01 exclusions) and no e-mail address.
ORDINARY = ["2021 2022 2023", "2022-23 2023-24", "2023 12 15 10 30", "20230915 1430", "1234567.89", "12345-6789",
            "10000-20000", "0-59 60-69 70-79", "01 02 03 04 05", "192.168.100.200", "10.0.19041.1234",
            "1 2 3 4 5 6 7 8 9", "$1 234 567 890", "123.456.789,00", "€ 123 456 789", "200 - 299", "10000 - 20000",
            "Score_2023_2024_2025", "Window_2023_09_01_2023_09_30", "14-AUG-24 01.45.07.123000 PM",
            "2023-11-16 19:23:46.721597-06:00", "PowerSchool_export_20250115_152233", "Grade_09_10_11_12",
            "Lexile 1000-1200", "09:00-15:30", "1.5 2.5 3.5 4.5 5.5", "5@3.50", "@handle.name", "x@y.1", "ab@c",
            "SY 2023-24", "2015-2016", "12/31/2023 1234567890", "Rate 3125550199.5", "Room 101, 2nd Floor",
            "Budget 2023 1500000", "Class of 2023 123456"]


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", SEED, *extra)
    return res, out, rep


def leak_scan(rep: Path, file: str) -> str:
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    entry = next(f for f in data["verdict"]["files"] if f["source"] == file)
    return entry["checks"]["leak_scan"]["result"]


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def column_report(rep: Path, file: str, col: str) -> dict:
    data = json.loads((rep / "files" / f"{file}.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in data["tables"][0]["columns"]}[col]


# --------------------------------------------------------------------------- R8-01, R8-02: the shape test
@pytest.mark.parametrize("value", PHONES)
def test_a_phone_number_in_any_national_layout_is_a_phone(value: str) -> None:
    assert "phone" in dtext.unit_shapes(value), value
    for org in (False, True):  # never kept for its count, nor as an organisational code
        assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=org), value


@pytest.mark.parametrize("value", EMAILS)
def test_an_email_address_in_any_script_is_an_email(value: str) -> None:
    assert "email" in dtext.unit_shapes(value), value
    assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=True), value


@pytest.mark.parametrize("value", ORDINARY)
def test_dates_ranges_numbers_and_amounts_hold_no_phone_or_email(value: str) -> None:
    assert not ({"phone", "email"} & dtext.unit_shapes(value)), value


def test_the_ssn_layout_is_an_ssn_not_also_a_phone() -> None:
    assert dtext.unit_shapes("123-45-6789") == {"ssn"}
    assert dtext.unit_shapes("123 45 6789") == {"ssn"}


@pytest.mark.parametrize("name", ["020 7946 0958", "Parent_020_7946_0958", "(020) 7946 0958", "07911_123456",
                                  "Tel 07911 123456", "+44 20 7946 0958", "01 23 45 67 89", "Mom 612 345 678"])
def test_d34_flags_a_phone_number_in_any_layout_in_a_name(name: str) -> None:
    assert "phone" in d34.name_kinds(name), name


@pytest.mark.parametrize("name", ["josé@school.org", "Contact_alice@müller.de", "张伟@school.org"])
def test_d34_flags_an_email_address_in_any_script_in_a_name(name: str) -> None:
    assert "email" in d34.name_kinds(name), name


@pytest.mark.parametrize("name", ["Order 020 7946 0958", "Invoice_2023_0001_2345", "Cut_200_250_300",
                                  "Pct_10_25_50_75_90", "05_10_15_20_25", "Grades_01_02_03_04_05",
                                  "Score_2023_2024_2025", "Window_2023_09_01_2023_09_30", "Bands 0-59 60-69 70-79",
                                  "PowerSchool_export_20250115_152233", "RIT_190_200_210", "Score_10000_20000",
                                  "Area_78701_1234", "Budget_2023_1500000"])
def test_d34_keeps_numbered_things_and_number_lists_in_names(name: str) -> None:
    assert "phone" not in d34.name_kinds(name), name


@pytest.mark.parametrize("value", PHONES + EMAILS)
def test_a_name_never_misses_a_shape_its_value_would_have(value: str) -> None:
    """R7-03 still holds: the name screen starts from the value test, and the D34 exceptions never excuse a phone
    number after a phone or family word, nor an e-mail address."""
    want = dtext.unit_shapes(value) & {"email", "phone"}
    assert want <= d34.name_kinds(value), value


# --------------------------------------------------------------------------- R8-01, R8-02: end to end
@pytest.mark.parametrize("value", ["020 7946 0958", "07911 123456", "(020) 7946 0958", "01 23 45 67 89",
                                   "josé@school.org", "alice@müller.de", "张伟@school.org"])
def test_a_shared_phone_or_email_level_is_never_copied(tmp_path: Path, value: str) -> None:
    # codex R8-01/R8-02: 12 of 60 students share the value, so the D35 count alone would keep it
    rows = [[f"S{61000 + j}", value if j < 12 else "Main Office"] for j in range(60)]
    src = write_csv(tmp_path / "in" / "contacts.csv", ["Student_ID", "Info"], rows)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = [r["Info"] for r in read_dicts(out / "contacts.csv")]
    assert len(syn) == 60 and value not in syn
    assert syn.count("Main Office") == 48  # the safe level is kept
    fakes = set(syn) - {"Main Office"}
    assert len(fakes) == 1 and syn.count(next(iter(fakes))) == 12  # one consistent pseudonym, exact count
    assert value not in everything(out, rep, console=res.output)
    assert leak_scan(rep, "contacts.csv") == "PASS"


def test_a_phone_number_header_fails_the_file(tmp_path: Path) -> None:
    # codex R8-01: a JSON key holding a domestic UK number (one column per phone number) is refused by D34
    src = tmp_path / "in" / "pivot.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps([{"Score": j, "020 7946 0958": j + 1} for j in range(40)]), encoding="utf-8")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    assert not (out / "pivot.json").exists()
    assert "020 7946 0958" not in everything(out, rep, console=res.output)
    entry = json.loads((rep / "files" / "pivot.json.json").read_text(encoding="utf-8"))
    assert entry["status"] == "FAIL" and "a phone number" in entry["reasons"][0]


# --------------------------------------------------------------------------- R8-05: --na-value tokens
def _na_table(tmp: Path) -> Path:
    # codex R8-05: a person's name declared missing in a number column and a constant column; 'Not Tested Yet' is
    # held by 12 students, 'Quinby Absent' by 3
    rows = []
    for i in range(60):
        score = "John Smith" if i < 2 else "Not Tested Yet" if i < 14 else "Quinby Absent" if i < 17 else str(i)
        rows.append([f"S{5000 + i}", score, "John Smith" if i < 2 else "Extended Time"])
    return write_csv(tmp / "in" / "na.csv", ["Student_ID", "Score", "Accommodation"], rows)


NA_FLAGS = ("--na-value", "John Smith", "--na-value", "Not Tested Yet", "--na-value", "Quinby Absent")


def test_a_declared_missing_value_is_no_permission_to_publish_it(tmp_path: Path) -> None:
    src = _na_table(tmp_path)
    res, out, rep = run(src, tmp_path, *NA_FLAGS)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "na.csv")
    assert len(syn) == 60
    blob = everything(out, rep, console=res.output)
    assert "John Smith" not in blob and "Quinby Absent" not in blob
    scores = [r["Score"] for r in syn]
    assert scores.count("Not Tested Yet") == 12  # no shape, held by >= --min-cell students: kept verbatim
    fakes = {v for v in scores if not v.isdigit() and v != "Not Tested Yet"}
    assert len(fakes) == 2  # 'John Smith' and 'Quinby Absent', each one pseudonym
    acc = [r["Accommodation"] for r in syn]
    fake_name = next(v for v in set(acc) if v != "Extended Time")
    assert acc.count(fake_name) == 2 and scores.count(fake_name) == 2  # one pseudonym in both columns
    # still missing values: the same missing counts, the real side shows a placeholder, the docs the written token
    for col in ("Score", "Accommodation"):
        c = column_report(rep, "na.csv", col)
        assert c["real"]["missing"] == c["synthetic"]["missing"], col
        assert all(k.startswith("[pseudonymized NA token") or k == "Not Tested Yet"
                   for k in c["real"]["missing_tokens"]), col
    manifest = json.loads((rep / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["na_value"].count("Not Tested Yet") == 1
    assert "readr" in (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert fake_name in (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")  # na = c(..., the pseudonym)
    assert leak_scan(rep, "na.csv") == "PASS"
    # seeded: the same run writes the same bytes
    res2, out2, _ = run(src, tmp_path, *NA_FLAGS, name="again")
    assert (out / "na.csv").read_bytes() == (out2 / "na.csv").read_bytes()


@pytest.mark.parametrize("token", ["Unknown", "Missing", "-999", "*"])
def test_missing_value_vocabulary_numbers_and_symbols_are_kept(tmp_path: Path, token: str) -> None:
    rows = [[f"S{2000 + i}", token if i < 2 else str(i)] for i in range(40)]
    src = write_csv(tmp_path / "in" / "s.csv", ["Student_ID", "Score"], rows)
    res, out, rep = run(src, tmp_path, "--na-value", token)
    assert res.exit_code == 0, res.output
    assert [r["Score"] for r in read_dicts(out / "s.csv")].count(token) == 2


def test_a_shared_na_token_with_a_phone_shape_is_pseudonymised(tmp_path: Path) -> None:
    rows = [[f"S{4000 + i}", "020 7946 0958" if i < 30 else str(i)] for i in range(60)]
    src = write_csv(tmp_path / "in" / "p.csv", ["Student_ID", "Score"], rows)
    res, out, rep = run(src, tmp_path, "--na-value", "020 7946 0958")
    assert res.exit_code == 0, res.output
    assert "020 7946 0958" not in everything(out, rep, console=res.output)
    assert column_report(rep, "p.csv", "Score")["synthetic"]["missing"] == 30


@pytest.mark.parametrize("header, values", [
    ("House", lambda i: ["Red", "Blue", "Green"][i % 3]),  # a factor: the token is one of its levels (D35 units)
    ("Clubs", lambda i: ["Art;Music", "Music", "Art;Drama", "Drama;Music", "Art"][i % 5]),  # a multi-select
    ("State", lambda i: ["TX", "CA", "NY"][i % 3]),  # a state column keeps its NA tokens verbatim
])
def test_text_policy_columns_follow_the_same_rule(tmp_path: Path, header: str, values: Any) -> None:
    rows = [[f"S{3000 + i}", "Zelda Quist" if i < 3 else values(i)] for i in range(60)]
    src = write_csv(tmp_path / "in" / "t.csv", ["Student_ID", header], rows)
    res, out, rep = run(src, tmp_path, "--na-value", "Zelda Quist")
    assert res.exit_code == 0, res.output
    syn = [r[header] for r in read_dicts(out / "t.csv")]
    assert len(syn) == 60 and "Zelda Quist" not in syn
    assert "Zelda Quist" not in everything(out, rep, console=res.output)
    assert leak_scan(rep, "t.csv") == "PASS"


def test_a_within_id_constant_writes_the_pseudonym_on_every_row(tmp_path: Path) -> None:
    rows = [[f"S{5000 + i}", "Ana Quist" if i < 2 else ("Female" if i % 2 else "Male"), str(r * 10 + i)]
            for i in range(30) for r in range(3)]
    src = write_csv(tmp_path / "in" / "e.csv", ["Student_ID", "Gender", "Score"], rows)
    res, out, rep = run(src, tmp_path, "--na-value", "Ana Quist")
    assert res.exit_code == 0, res.output
    syn = [r["Gender"] for r in read_dicts(out / "e.csv")]
    assert len(syn) == 90 and "Ana Quist" not in syn
    fakes = set(syn) - {"Female", "Male"}
    assert len(fakes) == 1 and syn.count(next(iter(fakes))) == 6


def test_the_scanner_excuses_only_the_na_tokens_a_column_writes(tmp_path: Path) -> None:
    """``_kept_keys`` no longer excuses every configured NA token in every column: a pseudonymised token is a real
    text of the run that no generator may write and the scan looks for."""
    from datanonymous.config import Settings
    from datanonymous.pipeline import execute
    from datanonymous.privacy import leaks

    src = _na_table(tmp_path)
    settings = Settings(mode="fidelity", na_value=["John Smith", "Not Tested Yet", "Quinby Absent", "Never Seen Token"], seed=7)
    run_ = execute(settings, src.parent, mode="inspect")
    texts = leaks.real_texts_of(run_.files)
    assert texts.holds("John Smith") and texts.holds("Quinby Absent")
    assert not texts.holds("Not Tested Yet")  # kept verbatim: public in the output anyway
    kept = set().union(*(leaks._kept_keys(p) for fr in run_.files for tp in fr.tables for p in tp.columns))
    assert "never seen token" not in kept and "john smith" not in kept and "not tested yet" in kept
