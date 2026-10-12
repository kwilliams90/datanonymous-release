"""Round 7, cluster avoid: no generator writes a real text value of ANY column of the run (R7-01, R7-02).

The bug class: each generator avoided only SOME real values of the run (its own column's, and run-wide only the
person-name columns' full names). A rare label that D35 pseudonymised in one column (``Roy Berger`` under
``Activity``), or a value of a column ``--exclude`` / ``--null`` keeps out of the output (a guardian's name, a note),
could come back verbatim from Faker, a pseudonym, filler text or an invented label in another column, and the leak
scanner, which compared names and labels within their own column only, passed it.

The fix is one run-wide registry (``privacy.texts``, planned by ``privacy.leaks.plan_real_texts``) of every real text
value the run does not write verbatim: D35-pseudonymised units of any column, every value of a person-name, e-mail,
address, city, IP/URL, free-text or high-cardinality column, text a date column replaces, native labels no row
writes, and the values of excluded and nulled columns (kept PRIVATE from pass 1). Values the run writes verbatim
somewhere (the D35 kept levels, items and tokens, NA tokens, states) are excused; a value without a letter is no
text (numbers, dates and codes; phones, SSNs and IDs have their own run-wide rules). Every generator avoids it (the
``generate.text.Drawer``, the name redraws, the template rebuilds, the pseudonym map), its short texts join the
R1-04 full-name index (no synthetic full name, nor first + last name on one row, spells one), and the leak scanner
checks every output text column against it, before writing and on the re-read files.

The fixtures are built from the tool's own output: a first run gives the synthetic values Faker (or the pseudonym
map, the label inventor, the template) draws for a column; a second run plants exactly those values as REAL values
of another column (excluded, nulled, a rare label, another file, a date column's leftover). Without the fix the
second run writes them again (the draws are seeded); with it they are redrawn. Assertion messages never print a real
value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.generate import pii
from datanonymous.generate import text as gtext
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType, FileResult, TableProfile
from datanonymous.privacy import leaks
from datanonymous.privacy import texts as rt

SEED = "31"
NAMES = ["Ana Garcia", "Liam Brown", "Noah Jones", "Mia Lopez", "Ella Moore", "Owen Clark", "Ava Lewis", "Leo Young",
         "Zoe Allen", "Eli King", "Ivy Scott", "Kai Green"]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", SEED, *extra)
    return res, out, rep


def column(out: Path, col: str, file: str = "input.csv") -> list[str]:
    path = out / file
    return [r[col] for r in read_dicts(path)] if path.exists() else []


def folded(values: list[str]) -> set[str]:
    return {" ".join(v.casefold().split()) for v in values if v.strip()}


def leak_scan(rep: Path, file: str = "input.csv") -> str:
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    entry = next(f for f in data["verdict"]["files"] if f["source"] == file)
    return entry["checks"]["leak_scan"]["result"]


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console]).casefold()


def roster(n: int, **cols: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {"student_id": [f"S{1001 + i}" for i in range(n)]}
    out.update(cols)
    out["score"] = [str(i + 1) for i in range(n)]
    return out


def no_avoidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generation without the run-wide registry (what the code did before R7): only the scanner stands."""
    monkeypatch.setattr(gtext, "real_text", lambda text: False)
    monkeypatch.setattr(pii, "NAME_MAX_WORDS", -1)


# =========================================================================== R7-02: excluded and nulled columns
@pytest.mark.parametrize("flag", ["--exclude", "--null"])
def test_excluded_or_nulled_names_are_never_generated(tmp_path: Path, flag: str) -> None:
    n = len(NAMES)
    dummy = [f"Qx{chr(97 + i)} Zzv" for i in range(n)]
    src = write_csv(tmp_path / "a" / "input.csv", roster(n, Full_Name=NAMES, Guardian_Name=dummy))
    res, out, _ = run(src, tmp_path, flag, "Guardian_Name", name="harvest")
    assert res.exit_code == 0, res.output
    fake = column(out, "Full_Name")
    assert len(folded(fake)) == n
    # the guardians ARE the names the tool would draw for the students
    src = write_csv(tmp_path / "b" / "input.csv", roster(n, Full_Name=NAMES, Guardian_Name=fake))
    res, out, rep = run(src, tmp_path, flag, "Guardian_Name", name="planted")
    assert res.exit_code == 0, res.output
    got = column(out, "Full_Name")
    assert not folded(got) & folded(fake), "a real guardian's name was written as a student's name"
    words = {frozenset(v.split()) for v in folded(fake)}
    assert not {frozenset(v.split()) for v in folded(got)} & words, "a real guardian's two names were written"
    assert leak_scan(rep) == "PASS"
    if flag == "--null":
        assert set(column(out, "Guardian_Name")) == {""}


def test_pseudonym_never_equals_a_value_of_an_excluded_column(tmp_path: Path) -> None:
    # R7-02 (codex pseudo_excluded): the pseudonym of a rare label must not be an excluded column's real value
    base = roster(3, Activity=["Quenby"] * 3, Notes=["Zzzzzz"] * 3)
    src = write_csv(tmp_path / "a" / "input.csv", base)
    res, out, _ = run(src, tmp_path, "--exclude", "Notes", name="harvest")
    assert res.exit_code == 0, res.output
    pseudo = set(column(out, "Activity"))
    assert len(pseudo) == 1 and pseudo != {"Quenby"}
    for flag in ("--exclude", "--null"):
        src = write_csv(tmp_path / flag / "input.csv", {**base, "Notes": sorted(pseudo) * 3})
        res, out, rep = run(src, tmp_path, flag, "Notes", name="planted" + flag)
        assert res.exit_code == 0, res.output
        again = set(column(out, "Activity"))
        assert len(again) == 1 and not folded(sorted(again)) & folded(sorted(pseudo)) and again != {"Quenby"}
        assert leak_scan(rep) == "PASS"


# =========================================================================== R7-01: rare labels D35 pseudonymised
def test_full_name_never_spells_a_pseudonymised_label(tmp_path: Path) -> None:
    names = NAMES[:3]
    dummy = ["Qxa Zzv", "Qxb Zzv", "Qxc Zzv"]
    src = write_csv(tmp_path / "a" / "input.csv", roster(3, Full_Name=names, Activity=dummy))
    res, out, _ = run(src, tmp_path, name="harvest")
    assert res.exit_code == 0, res.output
    fake = column(out, "Full_Name")
    src = write_csv(tmp_path / "b" / "input.csv", roster(3, Full_Name=names, Activity=fake))
    res, out, rep = run(src, tmp_path, name="planted")
    assert res.exit_code == 0, res.output
    col = json.loads((rep / "files" / "input.csv.json").read_text(encoding="utf-8"))
    activity = next(c for t in col["tables"] for c in t["columns"] if c["name"] == "Activity")
    assert activity["params"].get("pseudonymized_levels") == 3  # the D35 path: each label held by one student
    got = column(out, "Full_Name")
    assert not folded(got) & folded(fake), "a rare label's real text was written as a name"
    assert not folded(column(out, "Activity")) & folded(fake)
    assert leak_scan(rep) == "PASS"


def test_first_and_last_name_never_spell_a_pseudonymised_label(tmp_path: Path) -> None:
    # R7-01 (codex pair_rare_label): the label 'Dana Hughes' must not come back as First_Name + Last_Name on one row
    firsts, lasts = ["Ana", "Liam", "Noah"], ["Garcia", "Brown", "Jones"]
    src = write_csv(tmp_path / "a" / "input.csv",
                    roster(3, First_Name=firsts, Last_Name=lasts, Activity=["Qxa Zzv", "Qxb Zzv", "Qxc Zzv"]))
    res, out, _ = run(src, tmp_path, name="harvest")
    assert res.exit_code == 0, res.output
    pairs = [f"{f} {last}" for f, last in zip(column(out, "First_Name"), column(out, "Last_Name"))]
    src = write_csv(tmp_path / "b" / "input.csv", roster(3, First_Name=firsts, Last_Name=lasts, Activity=pairs))
    res, out, rep = run(src, tmp_path, name="planted")
    assert res.exit_code == 0, res.output
    rows = {frozenset((f.casefold(), last.casefold()))
            for f, last in zip(column(out, "First_Name"), column(out, "Last_Name"))}
    assert not rows & {frozenset(p.casefold().split()) for p in pairs}, "a rare label spelled by one row's names"
    assert leak_scan(rep) == "PASS"


# =========================================================================== the other generators
def test_invented_labels_and_filler_text_avoid_an_excluded_columns_values(tmp_path: Path) -> None:
    n = 30
    cols = roster(n, Career=[f"Career path number {i} here" for i in range(n)],
                  Comments=[f"Student {i} needs extra help with reading and writing at home." for i in range(n)],
                  Dropped=[f"Qx{i} Zzv" for i in range(n)])
    src = write_csv(tmp_path / "a" / "input.csv", cols)
    res, out, _ = run(src, tmp_path, "--exclude", "Dropped", name="harvest")
    assert res.exit_code == 0, res.output
    labels, notes = column(out, "Career"), column(out, "Comments")
    assert not folded(labels) & folded(cols["Career"]) and not folded(notes) & folded(cols["Comments"])
    planted = sorted(set(labels))[:15] + sorted(set(notes))[:15]
    src = write_csv(tmp_path / "b" / "input.csv", {**cols, "Dropped": planted})
    res, out, rep = run(src, tmp_path, "--exclude", "Dropped", name="planted")
    assert res.exit_code == 0, res.output
    assert not (folded(column(out, "Career")) | folded(column(out, "Comments"))) & folded(planted)
    assert leak_scan(rep) == "PASS"


def test_template_rebuild_avoids_an_excluded_columns_values(tmp_path: Path) -> None:
    n = 30
    firsts = ["Ana", "Liam", "Noah", "Mia", "Ella", "Owen", "Ava", "Leo", "Zoe", "Eli"] * 3
    lasts = ["Garcia", "Brown", "Jones", "Lopez", "Moore", "Clark", "Lewis", "Young", "Allen", "King"] * 3
    ids = [str(100 + i) for i in range(n)]
    cols = {"id": ids, "first_name": firsts, "last_name": lasts,
            "email": [f"{f.lower()}.{last.lower()}.{i}@school.org" for f, last, i in zip(firsts, lasts, ids)],
            "Alt": [f"qx{i}@zz.org" for i in range(n)]}
    src = write_csv(tmp_path / "a" / "input.csv", cols)
    res, out, _ = run(src, tmp_path, "--exclude", "Alt", name="harvest")
    assert res.exit_code == 0, res.output
    emails = column(out, "email")
    assert all(e.endswith(("example.org", "example.com", "example.net")) for e in emails)
    src = write_csv(tmp_path / "b" / "input.csv", {**cols, "Alt": emails})
    res, out, rep = run(src, tmp_path, "--exclude", "Alt", name="planted")
    assert res.exit_code == 0, res.output
    assert not folded(column(out, "email")) & folded(emails)
    assert leak_scan(rep) == "PASS"


def test_email_local_parts_are_real_usernames(tmp_path: Path) -> None:
    """The e-mail generator avoided only whole real addresses, while the scan compares local parts (a real local part
    is a real username, R5-02): local parts that are plain first names made the column FAIL. Run-wide now, also for
    the addresses of an excluded column."""
    from faker.providers.person.en_US import Provider

    names = sorted({n.lower() for n in Provider.first_names if n.isalpha() and len(n) == 4})[:120]
    cols = {"student_id": [f"S{1000 + i}" for i in range(len(names))], "Email": [f"{n}@school.org" for n in names],
            "Alt": [f"qx{i}@zz.org" for i in range(len(names))]}
    src = write_csv(tmp_path / "a" / "input.csv", cols)
    res, out, rep = run(src, tmp_path, "--exclude", "Alt", name="own")
    assert res.exit_code == 0, res.output
    fake = column(out, "Email")
    assert not {e.casefold().partition("@")[0] for e in fake} & set(names)
    assert leak_scan(rep) == "PASS"
    # the excluded column holds addresses with the local parts the tool drew (another domain)
    src = write_csv(tmp_path / "b" / "input.csv", {**cols, "Alt": [e.partition("@")[0] + "@home.net" for e in fake]})
    res, out, rep = run(src, tmp_path, "--exclude", "Alt", name="planted")
    assert res.exit_code == 0, res.output
    planted = {e.casefold().partition("@")[0] for e in fake}
    assert not {e.casefold().partition("@")[0] for e in column(out, "Email")} & planted
    assert leak_scan(rep) == "PASS"


def test_another_file_and_a_failed_file_count_too(tmp_path: Path) -> None:
    n = len(NAMES)
    roster_b = roster(n, Full_Name=NAMES)
    write_csv(tmp_path / "a" / "b.csv", roster_b)
    res, out, _ = run(tmp_path / "a", tmp_path, name="harvest")
    assert res.exit_code == 0, res.output
    fake = column(out, "Full_Name", "b.csv")[:3]
    # other.csv holds three of them as rare labels (pseudonymised, not a name column: the R1-04 name index alone
    # never saw them); in case b it FAILs (D34: a person's name as a header) before planning, and its people are
    # real people too
    write_csv(tmp_path / "b" / "b.csv", roster_b)
    write_csv(tmp_path / "b" / "other.csv", {"Activity": fake, "Garcia_Ana": ["1"] * 3})
    write_csv(tmp_path / "c" / "b.csv", roster_b)
    write_csv(tmp_path / "c" / "other.csv", {"Activity": fake, "Points": ["1"] * 3})
    for case, code in (("b", 3), ("c", 0)):
        res, out, rep = run(tmp_path / case, tmp_path, name="planted-" + case)
        assert res.exit_code == code, res.output
        got = column(out, "Full_Name", "b.csv")
        assert len(got) == n and not folded(got) & folded(fake)
        assert leak_scan(rep, "b.csv") == "PASS"
    page = json.loads((rep / "files" / "other.csv.json").read_text(encoding="utf-8"))
    activity = next(c for t in page["tables"] for c in t["columns"] if c["name"] == "Activity")
    assert activity["type"] != "PERSON_NAME" and activity["params"].get("pseudonymized_levels") == 3
    assert not folded(column(out, "Activity", "other.csv")) & folded(fake)


def test_text_a_date_column_replaces_is_avoided(tmp_path: Path) -> None:
    n = 120
    dates = [f"20{10 + i % 5}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)]
    names = [f"{a} {b}" for a, b in zip(["Ana", "Liam", "Noah", "Mia", "Ella", "Owen"] * 20,
                                         ["Garcia", "Brown", "Jones", "Lopez", "Moore", "Clark", "Lewis", "Young",
                                          "Allen", "King"] * 12)]
    names = [f"{nm} {chr(65 + i % 26)}{chr(97 + i // 26)}" for i, nm in enumerate(names)]  # 120 distinct
    src = write_csv(tmp_path / "a" / "input.csv", roster(n, Full_Name=names, DOB=dates))
    res, out, _ = run(src, tmp_path, name="harvest")
    assert res.exit_code == 0, res.output
    fake = column(out, "Full_Name")
    planted = list(dates)
    planted[7] = fake[0]  # a name typed into the date column: replaced by a generated date, never written
    src = write_csv(tmp_path / "b" / "input.csv", roster(n, Full_Name=names, DOB=planted))
    res, out, rep = run(src, tmp_path, name="planted")
    assert res.exit_code == 0, res.output
    col = json.loads((rep / "files" / "input.csv.json").read_text(encoding="utf-8"))
    dob = next(c for t in col["tables"] for c in t["columns"] if c["name"] == "DOB")
    assert dob["params"].get("unparsed_replaced") == 1
    assert fake[0].casefold() not in folded(column(out, "Full_Name"))
    assert fake[0].casefold() not in folded(column(out, "DOB"))


# =========================================================================== the scanner (both layers)
def _planted_guardians(tmp_path: Path, flag: str = "--exclude") -> tuple[Path, list[str]]:
    n = len(NAMES)
    src = write_csv(tmp_path / "a" / "input.csv", roster(n, Full_Name=NAMES, Guardian_Name=["Qxa Zzv"] * n))
    res, out, _ = run(src, tmp_path, flag, "Guardian_Name", name="harvest")
    assert res.exit_code == 0, res.output
    fake = column(out, "Full_Name")
    return write_csv(tmp_path / "b" / "input.csv", roster(n, Full_Name=NAMES, Guardian_Name=fake)), fake


@pytest.mark.parametrize("flag", ["--exclude", "--null"])
def test_scanner_fails_a_copy_from_another_column_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 flag: str) -> None:
    src, fake = _planted_guardians(tmp_path, flag)
    no_avoidance(monkeypatch)
    res, out, rep = run(src, tmp_path, flag, "Guardian_Name", name="leaky")
    assert res.exit_code == 3, res.output
    assert not (out / "input.csv").exists()
    page = json.loads((rep / "files" / "input.csv.json").read_text(encoding="utf-8"))
    assert "real text value held in the run" in json.dumps(page)
    blob = everything(out, rep, console=res.output)
    assert not any(v.casefold() in blob for v in fake), "a real value was printed"


def test_scanner_fails_a_copy_on_the_written_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, fake = _planted_guardians(tmp_path)
    no_avoidance(monkeypatch)
    monkeypatch.setattr(leaks, "run_texts", lambda tp: None)  # layer 1 sees no registry: the data scan must
    res, out, rep = run(src, tmp_path, "--exclude", "Guardian_Name", name="leaky")
    assert res.exit_code == 3, res.output
    assert not (out / "input.csv").exists()
    page = json.loads((rep / "files" / "input.csv.json").read_text(encoding="utf-8"))
    assert "rule: real text" in json.dumps(page)
    blob = everything(out, rep, console=res.output)
    assert not any(v.casefold() in blob for v in fake), "a real value was printed"


def test_the_fixture_is_hot_without_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the registry the planted run writes the planted names (the scanner then FAILs the file)."""
    src, fake = _planted_guardians(tmp_path)
    no_avoidance(monkeypatch)
    monkeypatch.setattr(leaks, "scan_table", lambda tp, st: [])
    monkeypatch.setattr(leaks, "scan_outputs", lambda run, payload: [])
    res, out, _ = run(src, tmp_path, "--exclude", "Guardian_Name", name="hot")
    assert res.exit_code == 0, res.output
    assert folded(column(out, "Full_Name")) == folded(fake)


# =========================================================================== no false refusals
def test_a_value_kept_verbatim_somewhere_is_excused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 'Grace Lee' is a club item of 12 students (D35 keeps it: the accepted residual) and one student's name; the
    # states are written as they are and are also values of an excluded column: neither is a leak
    n = 12
    names = ["Grace Lee"] + NAMES[1:]
    cols = roster(n, Full_Name=names, Clubs=["Chess;Grace Lee", "Grace Lee;Quenby"] * (n // 2),
                  State=["Texas", "Ohio", "Iowa"] * (n // 3), Consent=["Yes", "No"] * (n // 2),
                  Birth_State=["Texas", "Ohio", "Iowa", "Yes"] * (n // 4))
    src = write_csv(tmp_path / "in" / "input.csv", cols)
    res, out, rep = run(src, tmp_path, "--exclude", "Birth_State", name="kept")
    assert res.exit_code == 0, res.output
    items = {x for v in column(out, "Clubs") for x in v.split(";")}
    assert "Grace Lee" in items and not items & {"Chess", "Quenby"}  # kept, and two pseudonymised items
    assert "grace lee" not in folded(column(out, "Full_Name"))
    assert set(column(out, "State")) == {"Texas", "Ohio", "Iowa"}
    assert set(column(out, "Consent")) == {"Yes", "No"}
    assert leak_scan(rep) == "PASS"
    # without the excuse the scan would refuse the file: the fixture exercises it
    monkeypatch.setattr(leaks, "_kept_keys", lambda p: set())
    res, out, rep = run(src, tmp_path, "--exclude", "Birth_State", name="unexcused")
    assert res.exit_code == 3, res.output


def test_numbers_dates_and_codes_of_an_excluded_column_change_nothing(tmp_path: Path) -> None:
    n = 40
    cols = roster(n, Full_Name=[f"{a} {b}" for a, b in zip(NAMES * 4, reversed(NAMES * 4))],
                  Notes=[str(10 + i) for i in range(n)], Activity=[f"Club {i % 7}" for i in range(n)])
    outs = []
    variants = ([str(i) for i in range(n)], ["7"] * n, [f"2020-01-{1 + i % 28:02d}" for i in range(n)])
    for k, dropped in enumerate(variants):
        src = write_csv(tmp_path / str(k) / "input.csv", {**cols, "Dropped": dropped})
        res, out, rep = run(src, tmp_path, "--exclude", "Dropped", name=f"r{k}")
        assert res.exit_code == 0, res.output
        outs.append((out / "input.csv").read_bytes())
    assert outs[0] == outs[1] == outs[2]


def test_same_seed_same_output(tmp_path: Path) -> None:
    src, _ = _planted_guardians(tmp_path)
    blobs = []
    for k in range(2):
        res, out, _ = run(src, tmp_path, "--exclude", "Guardian_Name", name=f"det{k}")
        assert res.exit_code == 0, res.output
        blobs.append((out / "input.csv").read_bytes())
    assert blobs[0] == blobs[1]


# =========================================================================== units
def _profile(index: int, name: str, t: ColumnType, values: list[str], **params: Any) -> ColumnProfile:
    p = ColumnProfile(index=index, name=name, type=t, params=dict(params))
    p.model = ColumnModel(nonmissing_counts={v: 1 for v in values})
    p.model.real_values = frozenset(values)
    return p


def _table(*cols: ColumnProfile, dropped: dict[int, frozenset[str]] | None = None) -> TableProfile:
    tp = TableProfile(name=None, header=[p.name for p in cols], n_rows=3, columns=list(cols))
    if dropped:
        tp.private.extra[rt.DROPPED_KEY] = dropped
    return tp


def test_registry_keys_kinds_and_the_kept_excuse() -> None:
    from datanonymous.detect import text as dtext

    name = _profile(0, "Full_Name", ColumnType.PERSON_NAME, ["Roy  Berger", "O’Brien Kay"])
    city = _profile(1, "City", ColumnType.CITY, ["Springfield"])
    level = _profile(2, "Homeroom", ColumnType.FACTOR, ["Quenby", "Grace Lee"])
    level.params[dtext.COPY_GATE_KEY] = "allowlist"
    level.params[dtext.PSEUDO_KEY] = [["Quenby", "[pseudonymized 1]"]]
    dropped = ColumnProfile(index=3, name="Guardian", type=ColumnType.EMPTY, role="excluded")
    tp = _table(name, city, level, dropped, dropped={3: frozenset({"Emma Yoder", "12345", "Grace Lee"})})
    texts = leaks.real_texts_of([FileResult("f.csv", "f.csv", "csv", tables=[tp])])
    assert texts.holds("roy berger") and texts.holds("O'BRIEN KAY") and texts.holds(" Springfield ")
    assert texts.holds("Quenby") and texts.holds("emma   yoder")
    assert not texts.holds("Grace Lee")  # kept verbatim by Homeroom: public in the output anyway
    assert not texts.holds("12345")  # no letter: numbers, dates and codes name no one by themselves
    assert not texts.holds("Roy")  # whole values only
    kinds = {s.column: s.kind for s, _ in texts.sources}
    assert kinds == {"Full_Name": rt.NAME, "City": rt.TEXT, "Homeroom": rt.TEXT, "Guardian": rt.DROPPED}
    assert [s.column for s in texts.sources_of(rt.key("Emma Yoder"))] == ["Guardian"]
    assert set(texts.texts()) == {"springfield", "quenby", "emma yoder"}  # name columns are indexed by R1-04
    assert texts.hits(["ROY BERGER", "Grace Lee", "Lorem", "Emma Yoder"]) == {"roy berger", "emma yoder"}


def test_guard_and_drawer() -> None:
    texts = rt.RealTexts(frozenset({"roy berger"}), frozenset(), ())
    drawer = gtext.Drawer(set())
    assert drawer.ok("Roy Berger")  # no guard: the column's own values only (a table generated on its own)
    with rt.guard(texts):
        assert not drawer.ok("ROY  BERGER") and drawer.ok("Roy Bergers")
        assert pii._fresh("Roy Berger", set(), set()) is False
    assert rt.holds_real("Roy Berger") is False


def test_short_texts_join_the_full_name_index() -> None:
    texts = rt.RealTexts(frozenset({"roy berger", "quenby", "one two three four five six"}), frozenset(),
                         ((rt.Source("f.csv", None, "Activity"),
                           frozenset({"roy berger", "quenby", "one two three four five six"})),))
    tp = _table(_profile(0, "First", ColumnType.PERSON_NAME, ["Ana"], ), _profile(1, "Last", ColumnType.PERSON_NAME,
                                                                                ["Garcia"]))
    tp.columns[0].subtype, tp.columns[1].subtype = "first", "last"
    index = pii.plan_names([FileResult("f.csv", "f.csv", "csv", tables=[tp])], (), texts)
    assert "roy" in index.get("berger", set()) and "berger" in index.get("roy", set())
    assert "quenby" in index.get(pii.WHOLE, set())
    assert "one" not in index.get("two", set())  # six words: no person's name
    cols = {0: np.asarray(["Roy", "Ana"], dtype=object), 1: np.asarray(["Berger", "Smith"], dtype=object)}
    tp.private.extra["run_names"] = index
    assert pii.real_full_name_hits(tp, cols) == 1
