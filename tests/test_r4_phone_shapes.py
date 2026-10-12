"""Leak scan at scale: chance phone coincidences never FAIL a file, real copies always do.

Found at scale: on a 1M-row student file every run FAILed with "real phone-number value(s) ... in column
'Email'; rule: phone digits". The derived e-mail (``first.last + id``) glues a synthetic surname's random
digits to a synthetic ID (``bakdxvp16512760771@example.org``). About 50k of the 245k distinct e-mails held a
phone-length run, so a few equalled one of the ~1M real phone numbers by chance. Mimicked e-mail local parts
and notes with random phone-shaped digits do the same.

* The data rule compares phone-SHAPED tokens only (``privacy.digits.phone_tokens``): a whole run of 10-15
  digits, ``+`` notation, or digit groups laid out like a NANP number or a real phone of the run. It never
  compares digits glued across separators into another layout, or a window inside a longer run.
* The generators never write a value holding a real phone/SSN-shaped token (``RealDigitKeys``, planned
  run-wide). A drawn value is redrawn, and a derived-template row keeps its independently generated value.
* So the scan stays strict. A real phone copied into an e-mail local part, a note or a label still FAILs.

The end-to-end tests simulate the worst case "at scale": the real phone column holds exactly the digits the
generator draws. They also check that the generation guard is what turns that case from FAIL into PASS.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datanonymous import pipeline
from datanonymous.config import Settings
from datanonymous.generate import table as table_mod
from datanonymous.generate.text import Drawer
from datanonymous.model import ColumnProfile, ColumnType
from datanonymous.privacy import digits, leaks
from datanonymous.privacy.digits import NANP_LAYOUTS, RealDigitKeys, phone_keys, phone_tokens
from datanonymous.privacy.leaks import DataScanner, SensitiveIndex, cross_column_values

T = ColumnType


def _keys(text: str, layouts=NANP_LAYOUTS) -> list[str]:  # noqa: ANN001
    return [d for _, _, d in phone_tokens(text, layouts)]


# --------------------------------------------------------------------------- phone-shaped tokens
@pytest.mark.parametrize("text, expected", [
    ("3125550199", ["3125550199"]),
    ("jane3125550199@example.org", ["3125550199"]),  # a digit run bounded by non-digits, inside a local part
    ("bakdxvp16512760771@example.org", ["16512760771"]),  # 1 + 10 digits: compared as 6512760771 too
    ("(312) 555-0199", ["3125550199"]),
    ("312.555.0199", ["3125550199"]),
    ("312 555 0199", ["3125550199"]),
    ("call (312)555-0199!", ["3125550199"]),
    ("1-312-555-0199", ["13125550199"]),  # one number: its national part is not yielded again
    ("+1 (312) 555-0199", ["13125550199"]),
    ("+44 20 7946 0958", ["442079460958"]),  # international notation: the whole cluster
    ("312-555-0199 312-555-0200", ["3125550199", "3125550200"]),  # two numbers in one cluster
    ("Room 12 312-555-0199", ["3125550199"]),
    ("312-555-0199x12", ["3125550199"]),
    # not phone-shaped: groups glued across separators into another layout, decimals, long runs
    ("jsmith.1234567.2026@x.org", []),
    ("4.3 3.9 4.0 3.8 3.5", []),
    ("2024-03-15 10:30", []),
    ("2818.1709671234", []),
    ("020 7946 0958", []),  # a foreign layout only counts when a real phone of the run has it
    ("12345678901234567890", []),  # 20 digits: one number, never searched for a 10-digit window
    ("99312555019999", ["99312555019999"]),  # the whole run, not the phone inside it
])
def test_phone_tokens_are_phone_shaped(text: str, expected: list[str]) -> None:
    assert _keys(text) == expected


def test_real_layouts_make_a_formatted_phone() -> None:
    assert _keys("020 7946 0958", NANP_LAYOUTS | {(3, 4, 4)}) == ["02079460958"]
    assert phone_keys("16512760771") == ["16512760771", "6512760771"]


def test_holds_real_agrees_with_the_token_rules() -> None:
    """The fast paths (phone_numbers, the negative path of RealDigitKeys.holds_real) give the answers of the
    full token rules."""
    rng = random.Random(5)
    seps = ["", " ", "-", ".", ") ", "(", "x", "@", "a", "+", ", "]
    texts = []
    for _ in range(4000):
        parts = []
        for _ in range(rng.randint(1, 5)):
            parts.append("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 8))))
            parts.append(rng.choice(seps))
        texts.append("".join(parts))
    real = set()
    for t in texts[::3]:  # a third of the texts hold a real key (when they hold a token at all)
        for _, _, d in phone_tokens(t):
            real.update(phone_keys(d)[-1:])
    keys = RealDigitKeys(frozenset(real), frozenset({"900123456"}))
    for t in texts:
        for lays in (NANP_LAYOUTS, NANP_LAYOUTS | {(3, 4, 4), (2, 2, 2, 2, 2), (2, 4, 4)}):
            assert list(digits.phone_numbers(t, lays)) == [d for _, _, d in phone_tokens(t, lays)], t
        slow = any(k in real for _, _, d in phone_tokens(t) for k in phone_keys(d))
        slow = slow or any(digits.digits_of(m.group()) == "900123456" for m in digits.SSN_RE.finditer(t))
        assert keys.holds_real(t) == slow, t
    assert keys.holds_real("id 900-12-3456") and not keys.holds_real("id 900123456")


# --------------------------------------------------------------------------- the scan stays strict
def _profile(name: str, typ: ColumnType, index: int = 0) -> ColumnProfile:
    return ColumnProfile(index=index, name=name, type=typ, subtype=None, params={})


def _index(phones: list[str], ssns: list[str] = ()) -> SensitiveIndex:  # type: ignore[assignment]
    idx = SensitiveIndex()
    for o, (kind, vals) in enumerate(((leaks.PHONE, phones), (leaks.SSN, list(ssns)))):
        idx.origins.append(leaks.Origin("s.csv", None, kind, kind))
        terms = leaks._Terms()
        terms.add(kind, vals)
        idx._merge_terms(terms, o)
    return idx


def _phone_hits(idx: SensitiveIndex, p: ColumnProfile, values: list[str]) -> int:
    data = DataScanner(idx, leaks.Allow())
    fr = SimpleNamespace(rel_path="out.csv")
    hits = data.scan_column(fr, 0, p, np.array(values, dtype=object))  # type: ignore[arg-type]
    return sum(h.count for h in hits if h.rule in ("phone digits", "SSN"))


@pytest.mark.parametrize("typ", [T.EMAIL, T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL, T.PERSON_NAME, T.STREET_ADDRESS])
def test_a_real_phone_copied_into_any_text_column_fails(typ: ColumnType) -> None:
    idx = _index(["(312) 555-0199", "773.555.0142", "+44 20 7946 0958"], ["900-12-3456"])
    p = _profile("col", typ)
    planted = {
        "jane3125550199@example.org": 1,  # inside an e-mail local part, glued to letters
        "3125550199@vtext.example.com": 1,
        "j.773.555.0142@example.org": 1,
        "Called guardian at (773) 555-0142 about attendance": 1,
        "Call 1-312-555-0199 before noon": 1,
        "intl +44 20 7946 0958": 1,
        "ssn 900-12-3456 on file": 1,
    }
    for value, n in planted.items():
        assert _phone_hits(idx, p, [value]) == n, value
    # chance shapes that are not phone numbers: glued groups, windows inside longer runs, decimals
    for value in ["jsmith.3125550.199@example.org", "x93125550199123@example.org", "312.5550199",
                  "id 31255501991234567", "gpa 3.1 2.5 5.5 0.1 9.9"]:
        assert _phone_hits(idx, p, [value]) == 0, value


def test_phone_with_extension_and_foreign_layout_are_indexed() -> None:
    """A real phone with an extension is indexed by its number too; a real foreign layout is a phone shape."""
    dt = leaks.digit_terms(leaks.PHONE, ["312-555-0199 x12", "020 7946 0958", "(773) 555-0142"])
    assert {"3125550199", "312555019912", "02079460958", "7735550142"} <= dt.phones
    assert (3, 4, 4) in dt.layouts
    idx = _index(["312-555-0199 x12", "020 7946 0958"])
    notes = _profile("notes", T.FREE_TEXT)
    assert _phone_hits(idx, notes, ["call 312-555-0199"]) == 1
    assert _phone_hits(idx, notes, ["call 020 7946 0958"]) == 1


def test_id_and_numeric_columns_stay_out_of_the_phone_rule() -> None:
    """Area G's chance rule is unchanged: a synthetic ID or number equal to a real phone is not a copy."""
    from datanonymous.detect.text import COPY_GATE_KEY

    assert cross_column_values(_profile("sid", T.IDENTIFIER), ["3125550199"]) == []
    n = _profile("n", T.INTEGER)
    n.params[COPY_GATE_KEY] = "screened"  # a detected number column: its values passed the copy gate (R4)
    assert cross_column_values(n, ["3125550199"]) == []


# --------------------------------------------------------------------------- generators never draw one
def test_drawer_redraws_a_value_holding_a_real_phone() -> None:
    keys = RealDigitKeys(frozenset({"3125550199"}), frozenset({"900123456"}))
    d = Drawer(set())
    assert d.ok("call (312) 555-0199") and d.ok("900-12-3456")  # no guard: only real values of the column
    with digits.guard(keys):
        assert not d.ok("call (312) 555-0199") and not d.ok("x13125550199@example.org")
        assert not d.ok("900-12-3456") and d.ok("call (312) 555-0198")
    assert digits.active() is None


# --------------------------------------------------------------------------- end to end
FIRSTS = ["Ava", "Liam", "Noah", "Emma", "Mia", "Lucas", "Zoe", "Ethan", "Ivy", "Owen", "Ella", "Leo"]


def _students(n_students: int = 100, rows_per: int = 4, seed: int = 11) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    ids = sorted(int(v) for v in rng.choice(np.arange(1_000_000, 9_999_999), n_students, replace=False))
    rows = []
    for k, sid in enumerate(ids):
        first = FIRSTS[k % len(FIRSTS)]
        last = f"Surname{int(rng.integers(100, 999))}"  # 3 digits + a 7-digit ID: 10 glued digits
        contact = f"p{int(rng.integers(2 * 10**9, 9 * 10**9))}@mail.example.com"  # mimicked: random digits
        for _ in range(rows_per):
            a, b, c = (int(x) for x in (rng.integers(200, 999), rng.integers(200, 999), rng.integers(0, 9999)))
            rows.append({
                "student_id": str(sid), "first_name": first, "last_name": last,
                "email": f"{first.lower()}.{last.lower()}{sid}@school.example.org",
                "contact_email": contact,
                "notes": f"Called guardian at ({a}) {b}-{c:04d} about attendance on day {int(rng.integers(1, 180))}",
                "phone": f"({int(rng.integers(200, 999))}) {int(rng.integers(200, 999))}-"
                         f"{int(rng.integers(0, 9999)):04d}",
                "grade": str(int(rng.integers(1, 13))),
            })
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _run(tmp_path: Path, src: Path, name: str):  # noqa: ANN202
    out, rep = tmp_path / f"{name}_out", tmp_path / f"{name}_rep"
    run = pipeline.execute(Settings(mode="fidelity", seed=7), src, out, rep)
    return run, out, json.loads((rep / "report.json").read_text(encoding="utf-8"))


TEXT_COLS = ("email", "contact_email", "notes")


def _national(d: str) -> str | None:
    if len(d) == 10:
        return d
    if len(d) == 11 and d[0] == "1":
        return d[1:]
    return None


def _drawn_phones(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Per text column: the national numbers of the phone-shaped tokens the generator drew."""
    out: dict[str, list[str]] = {c: [] for c in TEXT_COLS}
    for r in rows:
        for c in TEXT_COLS:
            for _, _, d in phone_tokens(r[c]):
                n = _national(d)
                if n and n not in out[c]:
                    out[c].append(n)
    return out


def test_chance_phone_coincidences_at_scale_do_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worst case of scale: every real phone equals digits the generator draws. The guard keeps the file
    clean (PASS, no real phone in any text column). Without the guard the same data FAILs (the coincidence
    is real)."""
    rows = _students()
    src1 = tmp_path / "in1" / "students.csv"
    _write(src1, rows)
    run1, out1, rep1 = _run(tmp_path, src1.parent, "first")
    assert pipeline.exit_code(run1) == 0, rep1["files"][0]
    syn1 = _read(out1 / "students.csv")
    drawn = _drawn_phones(syn1)
    assert all(len(v) >= 20 for v in drawn.values()), {k: len(v) for k, v in drawn.items()}
    # the template e-mail held a 10-digit run (surname digits + ID) in every row: the 1M-row case
    assert all(phone_tokens(r["email"]) for r in syn1)
    # every other drawn number of each column becomes a real phone (the other rows keep their own phones)
    planted = [n for c in TEXT_COLS for n in drawn[c][::2]]
    assert len(planted) < len(rows)
    rows2 = [dict(r, phone=f"({n[:3]}) {n[3:6]}-{n[6:]}") if n else r
             for r, n in zip(rows, planted + [""] * (len(rows) - len(planted)))]
    src2 = tmp_path / "in2" / "students.csv"
    _write(src2, rows2)
    real = {digits.digits_of(r["phone"]) for r in rows2}
    assert all(any(n in real for n in drawn[c]) for c in TEXT_COLS)

    # without the generation guard: the drawn digits are the real phones now, and the scan FAILs
    with monkeypatch.context() as m:
        m.setattr(RealDigitKeys, "holds_real", lambda self, text: False)
        m.setattr(RealDigitKeys, "holds_phone", lambda self, text, **kw: False)
        # D46 also excludes run-wide bases before drawing from the phone deck.
        # Disable that independent protection for this intentionally leaky run.
        m.setattr(digits, "active", lambda: None)
        run0, out0, rep0 = _run(tmp_path, src2.parent, "unguarded")
    assert pipeline.exit_code(run0) == 3
    reasons = " ".join(rep0["files"][0]["reasons"])
    assert rep0["files"][0]["status"] == "FAIL" and "rule: phone digits" in reasons
    for c in TEXT_COLS:
        assert f"'{c}'" in reasons, c

    # with it (the real code): PASS, and no text column holds a real phone of the run
    run2, out2, rep2 = _run(tmp_path, src2.parent, "guarded")
    assert pipeline.exit_code(run2) == 0, rep2["files"][0]
    assert rep2["files"][0]["checks"]["leak_scan"] == "PASS"
    syn2 = _read(out2 / "students.csv")
    assert len(syn2) == len(rows2)
    for r in syn2:
        for c in TEXT_COLS:
            for _, _, d in phone_tokens(r[c]):
                assert not ({d, _national(d)} & real), c
    # the template rows that would have held a real phone keep their independently generated e-mail
    warn = [w["text"] for w in rep2["warnings"] if w["code"] == "template_partly_applied"]
    assert any("that would hold a real phone number or SSN of the run" in t for t in warn), warn
    kept = sum(r1["email"] == r2["email"] for r1, r2 in zip(syn1, syn2))
    assert 0 < kept < len(syn1)  # only those rows changed; the IDs and names are the same draws
    for r1, r2 in zip(syn1, syn2):
        if r1["email"] != r2["email"]:
            assert {_national(d) for _, _, d in phone_tokens(r1["email"])} & real
    assert [r["student_id"] for r in syn1] == [r["student_id"] for r in syn2]


def test_a_real_phone_written_into_a_text_column_still_fails(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """A generator bug that copies a real phone into an e-mail local part or a note FAILs the file."""
    rows = _students(n_students=40)
    src = tmp_path / "in" / "students.csv"
    _write(src, rows)
    real_phone = rows[5]["phone"]
    real_digits = digits.digits_of(real_phone)
    original = pipeline.structure_mod.finalize_rows

    def leaky(tp, st, *args):  # noqa: ANN001, ANN002, ANN202
        msgs = original(tp, st, *args)  # the last step before the pre-write check and the writer
        for col in st.columns:
            if col.name == "email":
                col.values[3] = f"x{real_digits}@example.org"
            elif col.name == "notes":
                col.values[8] = f"Called guardian at {real_phone} about attendance"
        return msgs

    monkeypatch.setattr(pipeline.structure_mod, "finalize_rows", leaky)
    run, out, rep = _run(tmp_path, src.parent, "leaky")
    assert pipeline.exit_code(run) == 3
    f = rep["files"][0]
    assert f["status"] == "FAIL" and not (out / "students.csv").exists()
    text = " ".join(f["reasons"])
    assert "rule: phone digits" in text and "'email'" in text and "'notes'" in text
    assert real_digits not in json.dumps(rep) and real_phone not in json.dumps(rep)


def test_generation_is_guarded_only_inside_a_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan hands every table the run's keys; the guard is active only while a table is generated."""
    rows = _students(n_students=20)
    src = tmp_path / "in" / "students.csv"
    _write(src, rows)
    seen: list[bool] = []
    original = table_mod._generate_table

    def spy(tp, ctx):  # noqa: ANN001, ANN202
        keys = digits.active()
        seen.append(keys is not None and digits.digits_of(rows[0]["phone"]) in keys.phones)
        return original(tp, ctx)

    monkeypatch.setattr(table_mod, "_generate_table", spy)
    run, _, _ = _run(tmp_path, src.parent, "spy")
    assert pipeline.exit_code(run) == 0 and seen == [True]
    assert digits.active() is None
