"""D38 (user, 2026-10-11): state student-ID acronyms join the default ID list; unrecognised wide repeated keys FAIL.

* Known state student-ID acronyms (``NYSSIS``, ``OSIS``, ``STN``, ``UIC``, ``WISEid``, ``SUNS``, ``FLEID``, ``KSID``,
  ``MOSIS``, ``PAsecureID``, ``EDUID``; with an ID word ``EIS ID``, ``EMIS ID``, ``KIDS ID``, ``MARSS ID``,
  ``SAIS ID``, ``MSIS ID``, ``PEIMS ID``) are on the D7 default ID list, case-insensitively, with separators and ID
  words (``NYSSIS_ID``, ``STN Number``, ``NYSSIS #``), and inside longer headers as the other student-ID codes are
  (``Student NYSSIS``, ``Old_STN``: person-ID headers; ``Sibling NYSSIS``: the values decide). A recognised column is
  a header-proven subject ID for D35 holders (``profile.entity.subject_proof``) and its IDs are real IDs of the run
  (R9, ``privacy.realids``). The merge triage case: ``NYSSIS`` over 6-digit IDs in a multi-row table wrote all 60
  real IDs and a canary.
* In a table with no recognised ID column, a column of repeated keys of 8 or more digits under a header that names no
  measure, count or flag (D7 candidate rule 7) FAILs its file (exit 3, nothing written for it, other files continue)
  with a redacted message naming the fix: ``--person-id COL`` / ``--id-col COL`` for a person ID, ``--org-code COL``
  for a school or district code. Those flags, ``--exclude`` and ``--null`` settle it; ``--force-type COL=TYPE`` and
  ``--force-factor COL`` settle it as the user's explicit decision, with a ``d38_flagged_keys`` warning on every run
  (``--org-code``: an info note), and the report withholds the column's value statistics.
"""

from __future__ import annotations

import csv
import json
import random
import shlex
from pathlib import Path
from typing import Any

import pytest

from conftest import read_dicts, run_cli
from datanonymous import hints
from datanonymous.detect import ids as dids
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.profile import entity as entity_mod
from test_area_a_support import profile_of, run_table

CANARY = "Kiwi anaphylaxis Zorbek"
COMMON = ("Illness", "Appointment", "Family Trip")


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def keys(n: int = 60, digits: int = 10, seed: int = 4, prefix: str = "") -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        k = prefix + str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))
        if k not in out:
            out.append(k)
    return out


def attendance(header: str, ids: list[str], seed: int = 5) -> dict[str, list[str]]:
    """A multi-row attendance table: 5-15 rows per student, the first student's 12 rows hold the canary."""
    rng = random.Random(seed)
    rows = []
    for i, s in enumerate(ids):
        for k in range(12 if i == 0 else rng.randint(5, 15)):
            rows.append((s, f"2026-{1 + k % 9:02d}-{10 + rng.randint(0, 18):02d}", rng.choice("PAT"),
                         CANARY if i == 0 else COMMON[i % 3]))
    rng.shuffle(rows)
    return {header: [r[0] for r in rows], "Date": [r[1] for r in rows], "Status": [r[2] for r in rows],
            "Reason": [r[3] for r in rows]}


def run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "1", *extra)
    return res, out, rep


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def statuses(rep: Path) -> dict[str, str]:
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    return {f["source"]: f["status"] for f in data["files"]}


def column_report(rep: Path, rel: str, col: str) -> dict[str, Any]:
    for t in json.loads((rep / "files" / f"{rel}.json").read_text(encoding="utf-8"))["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


# =========================================================================== the default ID list
STATE_HEADERS = ["NYSSIS", "nyssis", "NYSSIS_ID", "NYSSIS ID", "NYSSIS #", "NYSSIS (NYSED)", "OSIS", "STN",
                 "STN_Number", "STN2", "UIC", "WISEid", "WISE ID", "WISE_ID", "SUNS", "SUNS ID", "FLEID", "KSID",
                 "MOSIS", "MOSIS ID", "PAsecureID", "EDUID", "EIS ID", "EMIS ID", "KIDS ID", "MARSS ID",
                 "MARSS Number", "SAIS ID", "MSIS ID", "PEIMS ID", "TSDS ID"]


@pytest.mark.parametrize("name", STATE_HEADERS)
def test_state_id_acronyms_are_on_the_default_list(name: str) -> None:
    h = hints.header_hints(name)
    assert h.default_id and h.org_code is None and h.pii is None, name


@pytest.mark.parametrize("name", ["Student NYSSIS", "Student_STN", "Old_NYSSIS", "Prior WISEid", "District UIC",
                                  "Student EIS ID", "Student OSIS"])
def test_state_ids_with_qualifiers_or_a_student_word_are_person_ids(name: str) -> None:
    assert hints.header_hints(name).person_id_like, name


@pytest.mark.parametrize("name", ["Sibling NYSSIS", "Missing STN", "Sibling WISEid", "Other EIS ID"])
def test_state_ids_after_another_word_let_the_values_decide(name: str) -> None:
    assert hints.weak_id_header(name) == "code", name
    assert not hints.header_hints(name).default_id, name


@pytest.mark.parametrize("name", ["EIS", "Kids", "Num_Kids", "Diagnosis", "Diagnosis Code", "Prognosis", "Quick",
                                  "Station", "Suns Score", "Has_STN", "Num_NYSSIS", "Osmosis", "Emission"])
def test_words_holding_an_acronym_are_no_ids(name: str) -> None:
    # whole words only: 'Diagnosis' holds no OSIS, 'Quick' no UIC; a system acronym needs its ID word ('EIS ID');
    # flags and counts never are
    h = hints.header_hints(name)
    assert not h.default_id and not h.person_id_like, name
    assert hints.weak_id_header(name) != "code", name


@pytest.mark.parametrize("header", ["NYSSIS", "STN", "UIC", "WISEid", "SUNS", "OSIS", "NYSSIS_ID", "EIS ID"])
@pytest.mark.parametrize("digits", [6, 10])
def test_state_id_columns_get_synthetic_ids_and_prove_people(header: str, digits: int) -> None:
    # the merge triage case (NYSSIS over 6-digit IDs wrote all 60 real IDs and the canary): an ID column by the
    # default list, a header-proven subject ID (K-04/K-06: the counts are of people), the canary pseudonymised
    ids = keys(digits=digits)
    tp, cols = run_table(attendance(header, ids))
    p = profile_of(tp, header)
    assert p.type is T.IDENTIFIER and p.role == "id" and p.source == "default-id", header
    assert entity_mod.id_role(p) == "subject" and entity_mod.subject_proof(p) == entity_mod.PROOF_HEADER, header
    assert not set(cols[header].tolist()) & set(ids), header
    assert CANARY in dtext.pseudonymized(profile_of(tp, "Reason")), header
    assert not [m for m in profile_of(tp, "Reason").messages if m.code == "d35_rows_counted"], header


def test_no_default_ids_turns_the_bare_acronyms_off() -> None:
    # D7: --no-default-ids turns the list off; a bare acronym is then a student-ID code whose values decide (K-02)
    h = hints.header_hints("NYSSIS")
    assert h.default_id and not h.person_id_like and hints.weak_id_header("NYSSIS") == "code"
    ids = keys()
    tp, cols = run_table(attendance("NYSSIS", ids), no_default_ids=True)
    p = profile_of(tp, "NYSSIS")
    assert p.type is T.IDENTIFIER and p.source == "person-id-hint"
    assert not set(cols["NYSSIS"].tolist()) & set(ids)


def test_the_triage_case_writes_no_real_id(tmp_path: Path) -> None:
    # tests/_tmp/triage/A/a07_NYSSIS_s7 (merge report): 60 students, 6-digit IDs, 5-15 rows each
    ids = [str(100100 + 7 * i + random.Random(i).randint(0, 3)) for i in range(60)]
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("NYSSIS", ids))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    text = everything(out, rep, console=res.output)
    assert not {t for t in ids if t in text}
    assert CANARY.lower() not in text.lower()
    assert column_report(rep, "att.csv", "NYSSIS")["type"] == "IDENTIFIER"


def test_state_ids_are_real_ids_of_the_run(tmp_path: Path) -> None:
    # R9-01: a roster's NYSSIS IDs are real IDs of the run, so a value of another file that equals one (one of ten
    # references: no key reference) is pseudonymised, never written
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"NYSSIS": ["482913", "583027"], "Grade": ["5", "6"]})
    refs = [197521, 235403, 381395, 482913, 591325, 651325, 739531, 847251, 935215, 953201]
    write_csv(src / "visits.csv", {"subject": [str(v) for v in refs for _ in range(2)],
                                   "Visit": [str(j) for _ in refs for j in range(2)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "482913" not in everything(out, rep, console=res.output)
    assert len(read_dicts(out / "visits.csv")) == 20


# =========================================================================== the FAIL rule
def test_unrecognised_wide_repeated_keys_fail_their_file(tmp_path: Path) -> None:
    ids = keys()
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    write_csv(src / "roster.csv", {"Student_ID": [str(500000 + 7 * i) for i in range(30)],
                                   "Grade": [str(9 + i % 4) for i in range(30)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    st = statuses(rep)
    assert st["att.csv"] == "FAIL" and st["roster.csv"] in ("PASS", "WARN")  # the other files continue (D19)
    assert not (out / "att.csv").exists() and (out / "roster.csv").is_file()
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    fail = next(f for f in data["failures"] if f["source"] == "att.csv")
    reason = " ".join(fail["reasons"])
    assert "att.csv::XREF" in reason and "8 or more digits" in reason and "no recognised ID column" in reason
    assert fail["fix_commands"] == ["--person-id att.csv::XREF", "--id-col att.csv::XREF", "--org-code att.csv::XREF"]
    # redacted: no value in the console, the report or the docs; the column's value statistics are withheld
    assert not {k for k in ids if k in everything(out, rep, console=res.output)}
    assert "redacted" in column_report(rep, "att.csv", "XREF")["real"]


def test_phone_numbers_under_an_unknown_header_name_pii_col_in_the_fix(tmp_path: Path) -> None:
    # a household's phone number on each of its rows looks like a repeated 10-digit key: the fix also names
    # --pii-col COL=phone, which settles it (fake phone numbers)
    rng = random.Random(9)
    phones = [f"{rng.choice((212, 315, 518, 607, 716))}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"
              for _ in range(60)]
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", phones))
    res, _, rep = run(src, tmp_path, name="fail")
    assert res.exit_code == 3, res.output
    commands = json.loads((rep / "report.json").read_text(encoding="utf-8"))["failures"][0]["fix_commands"]
    assert "--pii-col att.csv::XREF=phone" in commands
    res, out, rep = run(src, tmp_path, "--pii-col", "XREF=phone", name="fixed")
    assert res.exit_code == 0, res.output
    assert not {p for p in phones if p in everything(out, rep, console=res.output)}


def test_random_ten_digit_keys_do_not_name_pii_col(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", keys()))
    res, _, rep = run(src, tmp_path)
    assert res.exit_code == 3, res.output
    commands = json.loads((rep / "report.json").read_text(encoding="utf-8"))["failures"][0]["fix_commands"]
    assert not [c for c in commands if c.startswith("--pii-col")]


def test_inspect_shows_the_failure(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", keys()))
    res = run_cli("inspect", src, "--seed", "1")
    assert res.exit_code == 3, res.output
    assert "FAIL" in res.output and "--org-code" in res.output


@pytest.mark.parametrize("flag", ["--person-id", "--id-col"])
def test_an_id_flag_settles_it_with_synthetic_ids(tmp_path: Path, flag: str) -> None:
    # synthetic IDs, and the column's real IDs join the run's real IDs (R9-01): another file's value equal to one
    # (one of ten references, no key reference) is pseudonymised
    ids = keys(digits=12)  # twelve digits: no phone shape (a 10-digit level may be masked as one)
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    refs = keys(n=9, digits=12, seed=21) + [ids[3]]  # one row each: no repeated keys of its own
    write_csv(src / "visits.csv", {"subject": refs, "Visit": [str(j % 3) for j in range(len(refs))]})
    res, out, rep = run(src, tmp_path, flag, "att.csv::XREF")
    assert res.exit_code == 0, res.output
    assert column_report(rep, "att.csv", "XREF")["type"] == "IDENTIFIER"
    written = {r["XREF"] for r in read_dicts(out / "att.csv")}
    assert len(written) == 60 and not written & set(ids)
    assert not {k for k in ids if k in everything(out, rep, console=res.output)}
    assert len(read_dicts(out / "visits.csv")) == 10


def test_the_id_flag_fixture_is_hot_without_the_real_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # without the run's real IDs (R9-01) the reference above is written as it is: the test above is meaningful
    from datanonymous.privacy import realids

    monkeypatch.setattr(realids, "of_files", lambda files: realids.RealIds())
    ids = keys(digits=12)
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    refs = keys(n=9, digits=12, seed=21) + [ids[3]]
    write_csv(src / "visits.csv", {"subject": refs, "Visit": [str(j % 3) for j in range(len(refs))]})
    res, out, _ = run(src, tmp_path, "--person-id", "att.csv::XREF")
    assert res.exit_code == 0, res.output
    assert ids[3] in {r["subject"] for r in read_dicts(out / "visits.csv")}


def test_an_excluded_student_id_is_no_id_column_of_the_table(tmp_path: Path) -> None:
    # the written table has no ID column: wide repeated keys beside an --exclude'd student ID still FAIL the file
    ids = keys()
    table = attendance("XREF", ids)
    sid = {k: str(700000 + 13 * n) for n, k in enumerate(ids)}
    src = tmp_path / "in"
    write_csv(src / "att.csv", {"Student_ID": [sid[k] for k in table["XREF"]], **table})
    res, _, rep = run(src, tmp_path, "--exclude", "Student_ID")
    assert res.exit_code == 3, res.output
    assert statuses(rep)["att.csv"] == "FAIL"


def test_the_fix_commands_of_the_report_each_settle_it(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", keys()))
    res, _, rep = run(src, tmp_path, name="fail")
    assert res.exit_code == 3, res.output
    commands = json.loads((rep / "report.json").read_text(encoding="utf-8"))["failures"][0]["fix_commands"]
    assert len(commands) == 3
    for k, cmd in enumerate(commands):
        res, out, _ = run(src, tmp_path, *shlex.split(cmd), name=f"fix{k}")
        assert res.exit_code == 0, (cmd, res.output)
        assert (out / "att.csv").is_file(), cmd


def test_org_code_keeps_real_values_by_choice_and_the_report_lists_none(tmp_path: Path) -> None:
    # --org-code: the user says the keys are school or district codes; their values are kept (D10, the copy gate's
    # rules), the report withholds them, and their keys still count who holds a value (the canary of one key's rows).
    # D39: a header the tool knows as a school or district code (NCESSCH) is one without the flag: test_d39_known_codes
    ids = keys(digits=12, n=25)
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("SiteRef", ids))
    res, out, rep = run(src, tmp_path, "--org-code", "SiteRef")
    assert res.exit_code == 0, res.output
    col = column_report(rep, "att.csv", "SiteRef")
    assert col["type"] == "ORG_CODE"
    assert "redacted" in col["real"] and "level_shares" not in col["real"]
    note = next(m for m in col["messages"] if m["code"] == "d38_flagged_keys")
    assert note["level"] == "info"
    written = {r["SiteRef"] for r in read_dicts(out / "att.csv")}
    assert written & set(ids)  # kept by the user's explicit choice
    assert not {k for k in ids if k in everything(rep, console=res.output)}
    assert CANARY not in {r["Reason"] for r in read_dicts(out / "att.csv")}


@pytest.mark.parametrize("flag", ["--exclude", "--null"])
def test_dropping_the_column_settles_it(tmp_path: Path, flag: str) -> None:
    ids = keys()
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    res, out, rep = run(src, tmp_path, flag, "XREF")
    assert res.exit_code == 0, res.output
    assert not {k for k in ids if k in everything(out, rep, console=res.output)}
    # its keys still tell D35 that the canary's 12 rows are one holder
    assert CANARY.lower() not in everything(out, rep, console=res.output).lower()


@pytest.mark.parametrize("flags,typ", [(("--force-type", "XREF=INTEGER"), "INTEGER"),
                                       (("--force-factor", "XREF"), "FACTOR")])
def test_forcing_a_non_id_type_settles_it_never_silently(tmp_path: Path, flags: tuple[str, ...], typ: str) -> None:
    # the user's explicit type decides (no FAIL), but on every run a warning says the column looks like people's IDs
    # and names the ID flags; the report still withholds its value statistics
    ids = keys(digits=9)  # nine digits: no phone shape, so the forced type is applied as it is
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    res, out, rep = run(src, tmp_path, *flags)
    assert res.exit_code == 0, res.output
    assert statuses(rep)["att.csv"] == "WARN"
    col = column_report(rep, "att.csv", "XREF")
    assert col["type"] == typ
    warn = next(m for m in col["messages"] if m["code"] == "d38_flagged_keys")
    assert warn["level"] == "warning" and "--person-id XREF" in warn["fix"]
    assert "redacted" in col["real"]
    assert not {k for k in ids if k in everything(rep, console=res.output)}


def test_force_type_identifier_is_an_id_flag(tmp_path: Path) -> None:
    ids = keys()
    src = tmp_path / "in"
    write_csv(src / "att.csv", attendance("XREF", ids))
    res, out, rep = run(src, tmp_path, "--force-type", "XREF=IDENTIFIER")
    assert res.exit_code == 0, res.output
    assert not {r["XREF"] for r in read_dicts(out / "att.csv")} & set(ids)


def test_beside_an_id_column_the_rule_only_warns(tmp_path: Path) -> None:
    # D38 narrows D7 only for a table with no recognised ID column: beside one, the candidate warning stays
    ids = keys()
    table = attendance("XREF", ids)
    sid = {k: str(700000 + 13 * n) for n, k in enumerate(ids)}
    table = {"Student_ID": [sid[k] for k in table["XREF"]], **table}
    src = tmp_path / "in"
    write_csv(src / "att.csv", table)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert statuses(rep)["att.csv"] == "WARN"


def test_a_key_reference_is_an_id_column_not_a_failure(tmp_path: Path) -> None:
    # the run's key discovery comes first: a column holding another file's ID column's IDs is an ID column
    ids = keys()
    src = tmp_path / "in"
    write_csv(src / "students.csv", {"Student_ID": ids, "Grade": [str(9 + i % 4) for i in range(len(ids))]})
    write_csv(src / "att.csv", attendance("XREF", ids))
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert column_report(rep, "att.csv", "XREF")["type"] == "IDENTIFIER"
    assert not {r["XREF"] for r in read_dicts(out / "att.csv")} & set(ids)


@pytest.mark.parametrize("header,digits,unique", [
    ("Amount", 10, False),         # a measure header
    ("Num_Visits", 10, False),     # a count
    ("School_Code", 12, False),    # an organisational code (D10): real values kept, no FAIL
    ("Campus", 9, False),
    ("XREF", 7, False),            # narrower than 8 digits: D7 warns at most
    ("XREF", 10, True),            # one row per key: no repeated keys
])
def test_what_does_not_fail(tmp_path: Path, header: str, digits: int, unique: bool) -> None:
    ids = keys(digits=digits)
    table = ({header: ids, "Status": [("A", "B", "C")[i % 3] for i in range(len(ids))]} if unique
             else attendance(header, ids))
    src = tmp_path / "in"
    write_csv(src / "t.csv", table)
    res, _, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert statuses(rep)["t.csv"] in ("PASS", "WARN")


def test_org_code_on_an_org_code_header_changes_nothing() -> None:
    # an organisational code by its header is one without the flag too: --org-code on it adds no D38 note
    tp, _ = run_table(attendance("School_Code", keys(digits=12, n=25)), org_code=["School_Code"])
    p = profile_of(tp, "School_Code")
    assert p.type is T.ORG_CODE and not p.params.get(dids.FLAGGED_KEYS_KEY)
    assert not [m for m in p.messages if m.code == "d38_flagged_keys"]


def test_profile_level_marks(tmp_path: Path) -> None:
    # the table-level rule the pipeline applies (profile.table.unrecognised_key_columns)
    from datanonymous.profile.table import unrecognised_key_columns, unrecognised_keys_failure

    tp, _ = run_table(attendance("XREF", keys()))
    assert [p.name for p in unrecognised_key_columns(tp)] == ["XREF"]
    reason, fix = unrecognised_keys_failure("att.csv", [tp])
    assert "att.csv::XREF" in reason and "--org-code att.csv::XREF" in fix
    tp, _ = run_table(attendance("XREF", keys()), org_code=["XREF"])
    assert not unrecognised_key_columns(tp)
    assert profile_of(tp, "XREF").params.get(dids.FLAGGED_KEYS_KEY) == 10
