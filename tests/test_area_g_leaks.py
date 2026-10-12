"""AREA G leak scanner: sensitive columns, one test per matching rule, allowlist, redaction, data checks."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from datanonymous import pipeline
from datanonymous.config import Settings
from datanonymous.model import SyntheticColumn, SyntheticTable
from datanonymous.privacy import leaks
from datanonymous.privacy.leaks import (
    Allow,
    DataScanner,
    LeakMessage,
    Scanner,
    SensitiveIndex,
    build_allow,
    scan_table,
    sensitive_kind,
    text_view,
    tool_vocabulary,
)
from test_area_g_grep import SCHOOL, sensitive


@pytest.fixture(scope="module")
def school_run(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("leaks_school")
    run = pipeline.execute(Settings(mode="fidelity", seed=5), SCHOOL, base / "out", base / "rep")
    assert pipeline.exit_code(run) == 0
    return run


@pytest.fixture(scope="module")
def scanner(school_run) -> Scanner:
    return Scanner(SensitiveIndex.build(school_run.files), build_allow(school_run))


@pytest.fixture(scope="module")
def data_scanner(school_run) -> DataScanner:
    return DataScanner(SensitiveIndex.build(school_run.files), build_allow(school_run))


def _rules(hits) -> set[str]:
    return {h.rule for h in hits}


def _file(run, name: str):
    return next(fr for fr in run.files if fr.rel_path == name)


# --------------------------------------------------------------------------- which columns are sensitive
def test_sensitive_kinds(school_run) -> None:
    kinds = {p.name: sensitive_kind(p) for p in _file(school_run, "people.csv").tables[0].columns}
    assert kinds == {"student_id": "id", "first_name": "name", "last_name": "name", "email": "email",
                     "phone": "phone", "ssn": "ssn", "home_address": "address", "teacher_name": "name",
                     "counselor_comment": "free_text", "class_standing": None, "gender": None,
                     "lunch_status": None, "gpa": None}
    kinds = {p.name: sensitive_kind(p) for p in _file(school_run, "enrollments.csv").tables[0].columns}
    assert kinds["teacher_email"] == "email" and kinds["note"] == "free_text" and kinds["course"] is None
    assert {p.name: sensitive_kind(p) for p in _file(school_run, "staff.csv").tables[0].columns}["staffid"] == "id"


def test_index_holds_namespaces_across_files(school_run) -> None:
    idx = SensitiveIndex.build(school_run.files)
    s = sensitive()
    ns = idx.namespaces["studentid"]
    assert set(s["ids"]) <= ns.exact and not ns.identity
    assert {o.file for o in (idx.origins[i] for i in ns.origins)} == {"people.csv", "enrollments.csv"}
    assert all(i.lstrip("0") in ns.normalized for i in s["ids"])


# --------------------------------------------------------------------------- one test per rule (strict scan)
def test_ids_exact_whole_token(scanner: Scanner) -> None:
    sid = sensitive()["ids"][0]
    assert _rules(scanner.hits_in_text(f"student {sid} moved")) == {"id"}
    assert _rules(scanner.hits_in_text(f'{{"id": "{sid}"}}')) == {"id"}
    assert not scanner.hits_in_text(f"x{sid}y")  # not a whole token
    assert not scanner.hits_in_text(f"mean 0.{sid}")  # a decimal fraction is not an ID
    staff = sensitive()["staff_ids"][0]
    assert _rules(scanner.hits_in_text(f"({staff})")) == {"id"}


def test_names_are_case_insensitive_whole_tokens(scanner: Scanner) -> None:
    s = sensitive()
    first = s["first_names"][0]
    assert _rules(scanner.hits_in_text(f"hello {first.upper()}!")) == {"token"}
    assert _rules(scanner.hits_in_text(f"hello {first.casefold()}")) == {"token"}
    assert not scanner.hits_in_text(f"hello {first}x")  # part of a longer word
    apostrophe = next(n for n in s["last_names"] if "'" in n)
    assert scanner.hits_in_text(f"Dear {apostrophe.casefold()},")
    accent = next(n for n in s["first_names"] if not n.isascii())
    assert scanner.hits_in_text(accent.upper())
    teacher = s["teacher_names"][0]
    assert "phrase" in _rules(scanner.hits_in_text(f"taught by {teacher}"))


def test_emails_and_their_local_parts(scanner: Scanner) -> None:
    email = next(e for e in sensitive()["emails"] if e.count(".") >= 3)
    local = email.split("@")[0]
    assert "email" in _rules(scanner.hits_in_text(f"write to {email.upper()} today"))
    assert "email local part" in _rules(scanner.hits_in_text(f"user {local} logged in"))
    assert "email local part" in _rules(scanner.hits_in_text(f"{local}@other.example"))


def test_phones_are_digit_normalised(scanner: Scanner) -> None:
    phone = sensitive()["phones"][0]
    d = "".join(ch for ch in phone if ch.isdigit())
    for form in (phone, f"{d[:3]}.{d[3:6]}.{d[6:]}", d, f"+1 {d[:3]} {d[3:6]} {d[6:]}", f"1-{d[:3]}-{d[3:6]}-{d[6:]}"):
        assert "digits" in _rules(scanner.hits_in_text(f"tel {form} ext")), "phone form not found"
    assert not scanner.hits_in_text("tel " + d[:-1])


def test_ssns_are_digit_normalised(scanner: Scanner) -> None:
    ssn = sensitive()["ssns"][0]
    assert "digits" in _rules(scanner.hits_in_text(f"ssn {ssn}"))
    assert "digits" in _rules(scanner.hits_in_text("ssn " + ssn.replace("-", "")))


def test_addresses_are_token_sequences(scanner: Scanner) -> None:
    addr = sensitive()["addresses"][0]
    assert "phrase" in _rules(scanner.hits_in_text(f"lives at {addr.upper()}, apt 2"))
    street = addr.split(" ", 1)[1]
    assert not scanner.hits_in_text(f"on {street}")  # a street name alone is not the address


def test_free_text_sentences_and_ngrams(scanner: Scanner) -> None:
    comment = next(c for c in sensitive()["comments"] if len(c.split()) > 10)
    words = comment.rstrip(".").split()
    assert "phrase" in _rules(scanner.hits_in_text(" ".join(words[2:8])))  # any 6-word window
    assert "phrase" in _rules(scanner.hits_in_text(comment.upper()))
    assert not scanner.hits_in_text(" ".join(words[-4:]) + " xyzzy")  # 4 words inside a sentence: too short


def test_public_tokens_are_excused(school_run) -> None:
    idx = SensitiveIndex.build(school_run.files)
    sid = sensitive()["ids"][0]
    assert Scanner(idx, Allow()).hits_in_text(sid)
    assert not Scanner(idx, Allow(tokens={sid})).hits_in_text(sid)
    allow = build_allow(school_run)
    for word in ("student_id", "gpa", "freshman", "verdict", "synthetic", "missing"):
        assert word.split("_")[0] in allow.tokens
    vocab = tool_vocabulary()
    names = {t.casefold() for k in ("first_names", "last_names") for v in sensitive()[k] for t in v.split()}
    assert not names & vocab  # the tool's own words never excuse a fixture name


def test_tool_vocabulary_has_literals_not_docstrings() -> None:
    vocab = tool_vocabulary()
    assert {"verdict", "rare", "category", "adjusted", "identifier", "lorem"} <= vocab
    assert "vendored" not in vocab


# --------------------------------------------------------------------------- redaction
def test_redaction_text_json_and_bytes(scanner: Scanner) -> None:
    s = sensitive()
    email, first = s["emails"][0], s["first_names"][1]
    text = f"a {email} b {first} c"
    red = scanner.redact_text(text)
    assert email not in red and first not in red and red.count(leaks.REDACTED) == 2 and red.startswith("a ")
    obj = {"warnings": [{"text": f"value {first}"}], first: 1, "n": 3}
    out = scanner.redact_obj(obj)
    assert first not in json.dumps(out) and out["n"] == 3
    payload = {"r.json": (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode(),
               "R.md": text.encode(), "clean.md": b"nothing here"}
    new = scanner.redact_payload(payload)
    assert first.encode() not in new["r.json"] and json.loads(new["r.json"])["n"] == 3
    assert new["clean.md"] is payload["clean.md"]
    m = LeakMessage("error", "leak_found", "x", redactor=scanner)
    assert m.to_public() == {"level": "error", "code": "leak_found", "text": "x"}
    assert first not in m.redact_text(text)


# --------------------------------------------------------------------------- layer 1 (before writing)
def _synth_with(tp, values_by_name: dict[str, list[str]]) -> SyntheticTable:
    cols = []
    for p in tp.columns:
        vals = values_by_name.get(p.name, ["zz_placeholder"] * tp.n_rows)
        cols.append(SyntheticColumn(p.index, p.name, np.asarray(vals, dtype=object)))
    return SyntheticTable(tp.name, tp.header, tp.n_rows, cols, tp)


def test_scan_table_uses_type_aware_normalisation(school_run) -> None:
    tp = _file(school_run, "people.csv").tables[0]
    s = sensitive()
    real = {p.name: sorted(p.model.real_values or []) for p in tp.columns if p.model is not None}
    cases = {
        "first_name": [real["first_name"][0].upper()],
        "email": [real["email"][0].split("@")[0] + "@example.org"],  # same local part
        "phone": ["".join(ch for ch in real["phone"][0] if ch.isdigit())],
        "student_id": [s["ids"][0].lstrip("0")],  # D8 key equals a real ID
        "home_address": ["  " + real["home_address"][0].casefold() + " "],
    }
    for col, vals in cases.items():
        msgs = scan_table(tp, _synth_with(tp, {col: vals}))
        assert [m.column for m in msgs] == [col], col
        assert msgs[0].text.startswith(leaks.LEAK_PREFIX) and msgs[0].level == "error"
        assert all(v.strip() not in msgs[0].text for v in vals)
    assert scan_table(tp, _synth_with(tp, {})) == []


def test_identity_mapped_ids_are_not_sensitive(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    (d / "a.csv").write_text("id,g\n" + "".join(f"{i},{'F' if i % 2 else 'M'}\n" for i in range(1, 41)),
                             encoding="utf-8")
    run = pipeline.execute(Settings(mode="fidelity", seed=1), d, tmp_path / "o", tmp_path / "r")
    assert pipeline.exit_code(run) == 0
    assert SensitiveIndex.build(run.files).empty
    assert leaks.sensitive_columns(run.files) == []


# --------------------------------------------------------------------------- data checks
def test_distinctive_patterns_in_any_cell(data_scanner: DataScanner) -> None:
    s = sensitive()
    comment = next(c for c in s["comments"] if len(c.split()) > 10)
    phone_digits = "".join(ch for ch in s["phones"][0] if ch.isdigit())
    cases = {
        "email": [f"see {s['emails'][0]} now"],
        "phone digits": [f"{phone_digits[:3]}-{phone_digits[3:6]}-{phone_digits[6:]}"],
        "SSN": [f"id {s['ssns'][0]}"],
        "text sequence": ["x " + " ".join(comment.split()[1:8]), s["addresses"][1].upper()],
    }
    for rule, vals in cases.items():
        found = data_scanner.distinctive(vals)
        assert rule in {r for _, r, _ in found}, rule
    assert data_scanner.distinctive(["F", "M", "3.42", "lorem ipsum dolor sit amet consectetur"]) == []
    # names are not matched across columns of the data (invented names may share common names)
    assert data_scanner.distinctive([s["first_names"][0], s["last_names"][0]]) == []


def test_text_view_reads_zip_and_gzip(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", "<sst><si><t>Printed by a&amp;b</t></si></sst>")
    (tmp_path / "a.xlsx").write_bytes(buf.getvalue())
    (tmp_path / "b.csv.gz").write_bytes(gzip.compress("x,y\n1,hello\n".encode()))
    assert "Printed by a&b" in text_view(tmp_path / "a.xlsx")
    assert "hello" in text_view(tmp_path / "b.csv.gz")


def test_scan_outputs_clean_run_returns_nothing(school_run) -> None:
    payload = {"REPORT.md": b"# report\n\nOverall verdict: PASS\n", "report.json": b"{}\n"}
    assert leaks.scan_outputs(school_run, payload) == []
