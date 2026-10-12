"""Round 6 (gate cluster): the copy gate's never-kept rules come first and never depend on the header, and the leak
scanner fails only the files it cannot verify.

* R6-01: a digit-only North American phone number is a personal-data shape in EVERY copied text unit (constant,
  flag, factor, organisational-code and ordinal levels, multi-select items, tokens) whatever the column's header
  (``detect.text.unit_shapes`` / ``bare_phone``); a header only widens the test. One such number among other
  numbers makes a number column same-shape masks (it would be a discrete level or the exact maximum).
* R6-03: no override that names a TYPE (``--org-code``, ``--force-type``) skips the personal-header or personal-shape
  rules (``detect.text.unit_kept`` checks them first); ``--force-factor`` is the one explicit opt-in that keeps every
  real value, and it warns with counts only when the header hints personal data or values have personal-data shapes.
* R6-04: a label without word tokens (a check mark, an emoji) no longer crashes the scanner's index, a unit without a
  letter or digit is kept (it identifies no one and its only same-shape text is itself), and an internal error of
  the scan FAILs only the file it concerns (D19): every other verified output stays.

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conftest import run_cli
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.privacy import leaks

from test_area_a_support import profile_of, run_table

N = 30
SID = [f"S{1000 + i}" for i in range(N)]
PHONE = "3125550187"


def _csv(path: Path, columns: dict[str, list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        w.writerows(zip(*columns.values()))
    return path


def _run(tmp_path: Path, src: Path, *extra: str, name: str = "run"):
    out, rep = tmp_path / f"{name}_out", tmp_path / f"{name}_rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "42", *extra)
    return res, out, rep


def _everything(out: Path, rep: Path) -> str:
    """Every text the run wrote (out dir and report dir)."""
    parts = []
    for root in (out, rep):
        for p in sorted(root.rglob("*")):
            if p.is_file():
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _file_report(rep: Path, rel: str) -> dict:
    return json.loads((rep / "files" / (rel + ".json")).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- R6-01: shapes whatever the header
@pytest.mark.parametrize("value", [PHONE, "13125550187", "+13125550187", "3125550187.0", " 3125550187 ",
                                   "3125550187, 4155550123", "3125550187 / 4155550123", "Home: 3125550187"])
def test_a_bare_north_american_number_is_a_phone_under_any_header(value: str) -> None:
    assert "phone" in dtext.unit_shapes(value), value
    for org in (False, True):  # an organisational code may hold its building's address, never a phone number
        assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=org), value


@pytest.mark.parametrize("value", ["1696342522",  # a Unix time starts with 1: no area code
                                   "9999999999",  # a common sentinel: the N9X area codes are reserved
                                   "2115550187", "3125110187",  # N11 service codes
                                   # R9-02: ``0123456789`` (one trunk 0, a UK 01234 number) is a phone number now;
                                   # two leading zeros read as zero padding
                                   "0012345678", "1234567890", "123456789", "31255501877", "-3125550187",
                                   "3125550187.25", "2024-03-15", "Room 3125"])
def test_other_digit_strings_are_no_phone_numbers(value: str) -> None:
    assert not dtext.bare_phone(value), value
    assert "phone" not in dtext.unit_shapes(value), value


@pytest.mark.parametrize("value", ["ana.lopez@school.org", "https://school.org/ana", "www.anas-page.com",
                                   "312-555-0187", "(312) 555-0187", "+44 20 7946 0958", "123-45-6789",
                                   "Pickup at 12 Oak Street", PHONE])
def test_every_shape_is_never_kept_without_a_personal_header(value: str) -> None:
    # e-mail, URL, separated phone, international phone, SSN, street address and (R6-01) a bare phone number: none of
    # them needs the header, so a neutral header ('Response') and any count keep none of them
    assert dtext.unit_shapes(value, digit_phone=False), value
    assert not dtext.unit_kept(value, holders=10_000, min_cell=10, digit_phone=False, personal_header=False), value


@pytest.mark.parametrize("values", [
    [PHONE] * 15 + ["Unavailable"] * 15,  # FACTOR
    [PHONE] * N,  # CONSTANT
    [PHONE] * 15 + [""] * 15,  # FLAG_SINGLE_TOKEN
    [f"{PHONE}, 4155550123"] * 15 + ["None given"] * 15,  # two numbers in one level
])
def test_phone_levels_held_by_many_people_are_pseudonymised(values: list[str]) -> None:
    tp, out = run_table({"student_id": SID, "Response": values})
    p = profile_of(tp, "Response")
    assert p.type in (T.FACTOR, T.CONSTANT, T.FLAG_SINGLE_TOKEN, T.MULTI_SELECT), p.type
    assert any(PHONE in t for t in dtext.pseudonymized(p))
    written = [v for v in out["Response"].tolist() if isinstance(v, str)]
    assert not any(PHONE in v for v in written)
    assert sum(1 for v in written if v.strip()) == sum(1 for v in values if v.strip())  # N in = N out, exact counts


def test_a_phone_level_of_an_organisational_code_and_a_multi_select_item_are_pseudonymised() -> None:
    school = [PHONE] * 15 + ["Lincoln"] * 15
    picks = [f"Art;{PHONE}" if i % 2 else "Art;Music" for i in range(N)]
    tp, out = run_table({"student_id": SID, "School": school, "Activities": picks})
    org = profile_of(tp, "School")
    assert org.type is T.ORG_CODE and PHONE in dtext.pseudonymized(org)
    assert "Lincoln" in set(out["School"].tolist())  # D10: the other codes keep their values
    assert not any(PHONE in str(v) for v in out["School"].tolist() + out["Activities"].tolist())


def test_the_pseudonym_of_a_bare_phone_number_is_a_fictitious_phone_number() -> None:
    tp, out = run_table({"student_id": SID, "Response": [PHONE] * 15 + ["Unavailable"] * 15})
    fakes = {v for v in out["Response"].tolist() if v != "Unavailable"}
    assert len(fakes) == 1
    fake = fakes.pop()
    assert len(fake) == 10 and fake.isdigit() and fake[3] in "23456789"  # D46: ordinary NXX exchange


def test_one_phone_number_among_other_numbers_is_never_written() -> None:
    # a phone typed into a score column was a discrete level (or would be the exact maximum, D14)
    scores = [str(i % 5) for i in range(N - 1)] + [PHONE]
    tp, out = run_table({"student_id": SID, "Response": scores})
    p = profile_of(tp, "Response")
    assert p.type is T.HIGH_CARD_CATEGORICAL and p.params.get("_numeric_mask"), p.type
    assert PHONE not in set(out["Response"].tolist())


def test_a_phone_number_as_the_exact_extreme_of_a_continuous_column_is_never_written() -> None:
    measures = [f"{50 + i * 0.37:.2f}" for i in range(N - 1)] + [PHONE]  # the phone is the exact maximum (D14)
    tp, out = run_table({"student_id": SID, "Measure": measures})
    assert profile_of(tp, "Measure").params.get("_numeric_mask")
    assert not any(PHONE in str(v) for v in out["Measure"].tolist())
    # big numbers inside the phone range whose written extremes are no phone numbers keep their rules: not whole,
    # or a round number in scientific notation (6.53396E+09 = 6533960000 has no ten written digits)
    amounts = [f"{3_000_000_000.5 + i * 1234.5678:.4f}" for i in range(N)]
    for vals in (amounts, ["6.53396E+09", "1.87254E+09", "2.5E+09"] * 10):
        tp, out = run_table({"student_id": SID, "Measure": vals})
        assert profile_of(tp, "Measure").params.get("_numeric_mask") is None, vals[0]


def test_numbers_that_are_no_phone_numbers_keep_their_rules() -> None:
    times = [str(1_696_000_000 + 3607 * i) for i in range(N)]  # Unix times
    sentinel = [str(i % 5) for i in range(N - 3)] + ["9999999999"] * 3
    tp, out = run_table({"student_id": SID, "Logged": times, "Score": sentinel,
                         "Account_Balance": [str(3125550000 + i) for i in range(N)]})
    assert profile_of(tp, "Logged").params.get("_numeric_mask") is None
    assert profile_of(tp, "Score").type is T.INTEGER and "9999999999" in set(out["Score"].tolist())
    assert profile_of(tp, "Account_Balance").params.get("_numeric_mask") is None  # R3: an amount header


def test_a_sentinel_level_is_kept() -> None:
    tp, out = run_table({"student_id": SID, "Response": ["9999999999"] * 15 + ["Unavailable"] * 15})
    assert not dtext.pseudonymized(profile_of(tp, "Response"))
    assert "9999999999" in set(out["Response"].tolist())


def test_cli_phone_level_under_a_neutral_header_is_never_written(tmp_path: Path) -> None:
    # Codex R6-01: 'Response' held 3125550187 on 15 rows and 'Unavailable' on 15 (30 students): copied everywhere
    src = _csv(tmp_path / "in" / "response.csv", {"student_id": SID, "Response": [PHONE] * 15 + ["Unavailable"] * 15})
    res, out, rep = _run(tmp_path, src)
    assert res.exit_code == 0, res.output
    assert PHONE not in _everything(out, rep)
    r = _file_report(rep, "response.csv")
    assert r["checks"]["leak_scan"]["result"] == "PASS"
    col = next(c for t in r["tables"] for c in t["columns"] if c["name"] == "Response")
    assert col["params"]["pseudonymized_levels"] == 1


# --------------------------------------------------------------------------- R6-03: overrides name the type only
def test_unit_kept_checks_the_never_kept_rules_before_every_exemption() -> None:
    # personal header first, then the organisational code (D10), then vocabulary and counts
    assert not dtext.unit_kept("John Smith", 10_000, 10, org_code=True, personal_header=True)
    assert not dtext.unit_kept("Lincoln", 10_000, 10, org_code=True, personal_header=True)
    assert dtext.unit_kept("N/A", 0, 10, org_code=True, personal_header=True)  # stands for no value
    assert dtext.unit_kept("101", 0, 10, org_code=True, personal_header=True)  # a short plain number
    assert not dtext.unit_kept(PHONE, 10_000, 10, org_code=True)
    # K-09: no shape tells a building's address from a student's (Homebound - 1427 Juniper Ridge Rd): never kept
    assert not dtext.unit_kept("12 Oak Street", 10_000, 10, org_code=True)
    assert dtext.unit_kept("Smith Academy", 1, 10, org_code=True)  # D10: rare codes keep their values


@pytest.mark.parametrize("flag,typ", [(("--org-code", "Teacher"), "ORG_CODE"),
                                      (("--force-type", "Teacher=ORG_CODE"), "ORG_CODE"),
                                      (("--force-type", "Teacher=FACTOR"), "FACTOR")])
def test_type_overrides_never_keep_values_under_a_personal_header(tmp_path: Path, flag: tuple[str, str],
                                                                   typ: str) -> None:
    # Codex R6-03: --org-code Teacher copied John Smith / Jane Jones (15 students each) with leak_scan PASS
    src = _csv(tmp_path / "in" / "teaching.csv",
               {"student_id": SID, "Teacher": ["John Smith"] * 15 + ["Jane Jones"] * 15})
    res, out, rep = _run(tmp_path, src, *flag)
    assert res.exit_code == 0, res.output
    text = _everything(out, rep)
    assert "John Smith" not in text and "Jane Jones" not in text
    r = _file_report(rep, "teaching.csv")
    col = next(c for t in r["tables"] for c in t["columns"] if c["name"] == "Teacher")
    assert col["type"] == typ  # the flag still sets the type
    assert col["params"]["pseudonymized_levels"] == 2
    assert "forced_type_values" in json.dumps(r)
    with (out / "teaching.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == N and len({r["Teacher"] for r in rows}) == 2  # same counts, same distinct count


def test_force_type_with_a_number_type_keeps_the_number_rules() -> None:
    phones = [str(3125550100 + i) for i in range(N)]
    for name in ("Parent_Phone", "Response"):
        tp, out = run_table({"student_id": SID, name: phones}, force_type=[f"{name}=INTEGER"])
        p = profile_of(tp, name)
        assert p.type is T.HIGH_CARD_CATEGORICAL and p.params.get("_numeric_mask"), (name, p.type)
        assert p.source == "force-type"
        assert not set(out[name].tolist()) & set(phones), name
    staff = [str(40000 + i % 7) for i in range(N)]
    tp, out = run_table({"student_id": SID, "Teacher": staff}, force_type=["Teacher=INTEGER"])
    assert profile_of(tp, "Teacher").params.get("_numeric_mask")  # D12: staff numbers under a person header
    assert not set(out["Teacher"].tolist()) & set(staff)
    tp, out = run_table({"student_id": SID, "Score": [str(i % 7) for i in range(N)]}, force_type=["Score=INTEGER"])
    assert profile_of(tp, "Score").type is T.INTEGER  # an ordinary number column keeps the forced type


def test_force_factor_is_the_explicit_opt_in_and_warns_with_counts_only() -> None:
    values = ["John Smith"] * 15 + ["Jane Jones"] * 15
    tp, out = run_table({"student_id": SID, "Teacher": values}, force_factor=["Teacher"])
    p = profile_of(tp, "Teacher")
    assert p.type is T.FACTOR and not dtext.pseudonymized(p)
    assert set(out["Teacher"].tolist()) == {"John Smith", "Jane Jones"}  # D35: the user's explicit word
    warn = [m for m in p.messages if m.code == "forced_copy"]
    assert len(warn) == 1 and "header hints personal data" in warn[0].text
    assert "John" not in warn[0].text and "Jones" not in warn[0].text
    tp, _ = run_table({"student_id": SID, "Response": [PHONE] * 15 + ["Unavailable"] * 15},
                      force_factor=["Response"])
    warn = [m for m in profile_of(tp, "Response").messages if m.code == "forced_copy"]
    assert len(warn) == 1 and "1 of 2 distinct value(s) hold personal data (phone)" in warn[0].text
    assert PHONE not in warn[0].text
    tp, _ = run_table({"student_id": SID, "Club": ["Chess"] * 15 + ["Robotics"] * 15}, force_factor=["Club"])
    assert not [m for m in profile_of(tp, "Club").messages if m.code == "forced_copy"]  # nothing to warn about


def test_header_hinted_organisational_codes_keep_their_values() -> None:
    # D10 control: a school code column keeps its rare values (no personal header, no personal-data shape)
    schools = ["Lincoln Elementary"] * 28 + ["Smith Academy"] * 2
    tp, out = run_table({"student_id": SID, "School": schools})
    assert profile_of(tp, "School").type is T.ORG_CODE
    assert set(out["School"].tolist()) == set(schools)


# --------------------------------------------------------------------------- R6-04: symbol labels and isolation
@pytest.mark.parametrize("value,expected", [("✓", True), ("✗", True), ("\U0001f7e2", True), ("+", True),
                                            ("—", True), ("!!", True), ("Ⓐ", False),  # circled A
                                            ("Ａ", False), ("N/A", False), ("A+", False), ("", False)])
def test_symbol_only(value: str, expected: bool) -> None:
    assert dtext.symbol_only(value) is expected
    if expected:
        assert dtext.unit_kept(value, holders=1, min_cell=10)
        assert dtext.unit_kept(value, holders=1, min_cell=10, personal_header=True)


def test_labels_without_word_tokens_are_indexed_without_an_error() -> None:
    terms = leaks._Terms()
    terms.add(leaks.LABEL, ["✓", "✗", "\U0001f7e2 \U0001f534", "--", "0,89", "Rift Valley"])
    assert ("rift", "valley") in terms.phrases[2]
    assert terms.number_literals[("0", "89")] == {"0,89"}
    real = leaks._RealSets.of(leaks.LABEL, ["✓"])
    assert leaks._same_column_hits(leaks.LABEL, real, ["✓"]) == 1  # the same-column check compares it whole


def test_cli_symbol_labels_never_remove_unrelated_outputs(tmp_path: Path) -> None:
    # Codex R6-04: IndexError at leaks.py while indexing ✓/✗ removed both outputs
    src = tmp_path / "in"
    _csv(src / "clean.csv", {"student_id": ["S1001", "S1002", "S1003"], "Score": ["10", "20", "30"]})
    _csv(src / "choices.csv", {"student_id": ["S1001", "S1002", "S1003"], "Response": ["✓", "✗", "✓"]})
    emoji = [chr(0x1F600 + i) for i in range(25)]  # more than --factor-max distinct: high-cardinality labels
    _csv(src / "mood.csv", {"student_id": [f"S{2000 + i}" for i in range(50)], "Mood": emoji * 2})
    res, out, rep = _run(tmp_path, src)
    assert res.exit_code == 0, res.output
    for name in ("clean.csv", "choices.csv", "mood.csv"):
        assert (out / name).is_file(), name
    with (out / "choices.csv").open(encoding="utf-8") as f:
        assert [r["Response"] for r in csv.DictReader(f)].count("✓") == 2  # a symbol is kept, counts exact
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert all(f["checks"]["leak_scan"] == "PASS" for f in data["files"])


def _two_files(tmp_path: Path) -> Path:
    src = tmp_path / "in"
    _csv(src / "a_clean.csv", {"student_id": SID, "Club": ["Chess"] * 15 + ["Robotics"] * 15})
    names = [f"Pat Quillon{i}" for i in range(N)]
    _csv(src / "b_notes.csv", {"student_id": SID, "Full_Name": names, "Score": [str(i) for i in range(N)]})
    return src


def _check_only_b_failed(res, out: Path, rep: Path, withheld: bool) -> None:  # noqa: ANN001
    assert res.exit_code == 3, res.output
    assert (out / "a_clean.csv").is_file()  # verified: kept
    assert not (out / "b_notes.csv").exists()  # unverified: removed (D19)
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    status = {f["source"]: f for f in data["files"]}
    assert status["a_clean.csv"]["status"] != "FAIL" and status["a_clean.csv"]["checks"]["leak_scan"] == "PASS"
    assert status["b_notes.csv"]["status"] == "FAIL" and status["b_notes.csv"]["checks"]["leak_scan"] == "FAIL"
    assert any(e["code"] == "leak_scan_incomplete" for e in data["errors"])
    b = _file_report(rep, "b_notes.csv")
    assert b["details_withheld"] is withheld
    if withheld:
        assert b["tables"] == []
    text = (rep / "REPORT.md").read_text(encoding="utf-8") + (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "Quillon" not in text and "scanner bug" not in text
    assert "a_clean.csv" in (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "b_notes.csv" not in (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")


def test_an_error_indexing_one_column_fails_only_its_file(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    real_add = leaks._Terms.add

    def add(self, kind, values, dt=None):  # noqa: ANN001, ANN202
        if any("Quillon" in v for v in values):
            raise RuntimeError("scanner bug")
        return real_add(self, kind, values, dt)

    monkeypatch.setattr(leaks._Terms, "add", add)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    _check_only_b_failed(res, out, rep, withheld=True)


def test_an_error_scanning_one_output_fails_only_that_file(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    real_scan = leaks._scan_data_file

    def scan(fr, out_dir, data):  # noqa: ANN001, ANN202
        if fr.rel_path == "b_notes.csv":
            raise RuntimeError("scanner bug")
        return real_scan(fr, out_dir, data)

    monkeypatch.setattr(leaks, "_scan_data_file", scan)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    _check_only_b_failed(res, out, rep, withheld=False)


def test_an_error_scanning_one_report_page_fails_only_its_file(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    real_hits = leaks.Scanner.hits_in_bytes

    def hits(self, name, data):  # noqa: ANN001, ANN202
        if name.startswith("files/b_notes.csv"):
            raise RuntimeError("scanner bug")
        return real_hits(self, name, data)

    monkeypatch.setattr(leaks.Scanner, "hits_in_bytes", hits)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    _check_only_b_failed(res, out, rep, withheld=True)


def test_an_error_on_a_run_level_page_still_leaves_nothing_unverified(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    real_hits = leaks.Scanner.hits_in_bytes

    def hits(self, name, data):  # noqa: ANN001, ANN202
        if name == "REPORT.md":
            raise RuntimeError("scanner bug")
        return real_hits(self, name, data)

    monkeypatch.setattr(leaks.Scanner, "hits_in_bytes", hits)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    assert res.exit_code == 3, res.output
    assert not (out / "a_clean.csv").exists() and not (out / "b_notes.csv").exists()  # no file can be tied to it
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert any(e["code"] == "leak_scan_incomplete" for e in data["errors"])


def test_an_unscannable_doc_is_removed_and_the_data_stays(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    real_hits = leaks.Scanner.hits_in_bytes

    def hits(self, name, data):  # noqa: ANN001, ANN202
        if name == "DATA_DICTIONARY.md":
            raise RuntimeError("scanner bug")
        return real_hits(self, name, data)

    monkeypatch.setattr(leaks.Scanner, "hits_in_bytes", hits)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    assert res.exit_code == 3, res.output
    assert (out / "a_clean.csv").is_file() and (out / "b_notes.csv").is_file()
    assert not (out / "DATA_DICTIONARY.md").exists() and (out / "README.md").is_file()
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert any(e["code"] == "leak_scan_incomplete" for e in data["errors"])


def test_a_file_failing_after_its_output_was_moved_into_place_is_not_written(tmp_path: Path,
                                                                             monkeypatch) -> None:  # noqa: ANN001
    # D19: an internal error after the self-check (statistics of the written values) FAILed the file but left it
    from datanonymous import pipeline

    real_stats = pipeline.column_stats

    def stats(values, p, min_cell, **kwargs):  # noqa: ANN001, ANN202
        if p.name == "Full_Name":
            raise RuntimeError("stats bug")
        return real_stats(values, p, min_cell, **kwargs)

    monkeypatch.setattr(pipeline, "column_stats", stats)
    res, out, rep = _run(tmp_path, _two_files(tmp_path))
    assert res.exit_code == 3, res.output
    assert (out / "a_clean.csv").is_file() and not (out / "b_notes.csv").exists()
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert not any(w["code"] == "stale_output" for w in data["warnings"])  # it is no copy from an earlier run


def test_the_redactor_fails_safe() -> None:
    idx = leaks.SensitiveIndex()
    scanner = leaks.Scanner(idx, leaks.Allow())

    def boom(text):  # noqa: ANN001, ANN202
        raise RuntimeError("scanner bug")

    scanner.hits_in_text = boom  # type: ignore[method-assign]
    assert scanner.redact_text("anything") == leaks.REDACTED  # text the matcher cannot read is never printed
    assert scanner.redact_obj(["a", 1]) == [leaks.REDACTED, leaks.REDACTED]
    assert json.loads(scanner.redact_bytes("x.json", b'{"a": "b"}')) == {leaks.REDACTED: leaks.REDACTED}
