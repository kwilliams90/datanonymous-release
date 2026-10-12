"""AREA G end to end: no real sensitive value in any out-dir or report file, and planted leaks are caught.

Every test greps every written file (and the console output) with the
independent grep of ``test_area_g_grep`` for every real ID, name, email (and
local part), phone (digit-normalised), SSN, address and free-text n-gram of
the adversarial fixtures. Assertion messages name kinds and files, never values.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import run_cli
from test_area_g_grep import SCHOOL, hits, read_texts, sensitive


def _run(tmp_path: Path, *extra: str, name: str = "run"):
    out, rep = tmp_path / f"{name}_out", tmp_path / f"{name}_rep"
    res = run_cli(SCHOOL, "--out-dir", out, "--report-dir", rep, "--seed", "11", *extra)
    return res, out, rep


def _assert_clean(*texts: tuple[str, str]) -> None:
    dirty = {name: hits(text) for name, text in texts if hits(text)}
    assert not dirty, dirty  # {file: [kinds]} - never the values


def _clean_dirs(res, out: Path, rep: Path, skip: tuple[str, ...] = ()) -> None:
    texts = [(k, v) for k, v in read_texts(out, rep).items() if not k.endswith(skip)]
    _assert_clean(("console", res.output), *texts)


def _report(rep: Path) -> dict:
    return json.loads((rep / "report.json").read_text(encoding="utf-8"))


def _file_entry(rep: Path, source: str) -> dict:
    return next(f for f in _report(rep)["files"] if f["source"] == source)


# --------------------------------------------------------------------------- the clean run
def test_fixture_run_has_no_real_value_anywhere(tmp_path: Path) -> None:
    res, out, rep = _run(tmp_path)
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["people.csv", "enrollments.csv", "staff.csv", "README.md", "DATA_DICTIONARY.md", "schema.json"])
    _clean_dirs(res, out, rep)
    data = _report(rep)
    assert data["verdict"]["overall"] in ("PASS", "WARN")
    assert all(f["checks"]["leak_scan"] == "PASS" for f in data["files"])
    # the per-file report says which columns the scan covered (names only)
    pj = json.loads((rep / "files" / "people.csv.json").read_text(encoding="utf-8"))
    kinds = {c["column"]: c["kind"] for c in pj["leak_scan"]["sensitive_columns"]}
    assert kinds["email"] == "email" and kinds["teacher_name"] == "name" and kinds["counselor_comment"] == "free_text"


def test_json_console_and_inspect_have_no_real_value(tmp_path: Path) -> None:
    res, out, rep = _run(tmp_path, "--json")
    assert res.exit_code == 0
    _assert_clean(("json", res.output))
    res = run_cli("inspect", SCHOOL)
    assert res.exit_code == 0
    _assert_clean(("inspect", res.output))
    res = run_cli("inspect", SCHOOL, "--json")
    _assert_clean(("inspect-json", res.output))


def test_same_seed_gives_a_byte_identical_out_dir(tmp_path: Path) -> None:
    _, out1, rep1 = _run(tmp_path, name="a")
    _, out2, rep2 = _run(tmp_path, name="b")
    files1 = {p.relative_to(out1).as_posix(): p.read_bytes() for p in out1.rglob("*") if p.is_file()}
    files2 = {p.relative_to(out2).as_posix(): p.read_bytes() for p in out2.rglob("*") if p.is_file()}
    assert files1 == files2
    a = (rep1 / "files" / "people.csv.json").read_bytes()
    b = (rep2 / "files" / "people.csv.json").read_bytes()
    assert a == b


# --------------------------------------------------------------------------- planted leaks
def _patch_writer(monkeypatch, target: str, column: str, value_fn) -> None:  # noqa: ANN001
    from datanonymous.io import registry

    h = registry.FORMATS["csv"]

    def write(req) -> None:  # noqa: ANN001
        if req.path.name.startswith(target):
            tables = []
            for t in req.tables:
                cols = list(t.columns)
                pos = t.header.index(column)
                vals = np.array(cols[pos], dtype=object, copy=True)
                vals[0] = value_fn(vals[0])
                cols[pos] = vals
                tables.append(dataclasses.replace(t, columns=cols))
            req = dataclasses.replace(req, tables=tables)
        h.write(req)

    monkeypatch.setitem(registry.FORMATS, "csv", dataclasses.replace(h, write=write))


def _assert_failed_by_leak(res, out: Path, rep: Path, source: str) -> None:
    assert res.exit_code == 3, res.output
    assert not (out / source).exists()
    entry = _file_entry(rep, source)
    assert entry["status"] == "FAIL" and entry["checks"]["leak_scan"] == "FAIL" and not entry["written"]
    assert any(r.startswith("leak scan:") for r in entry["reasons"])
    data = _report(rep)
    assert data["verdict"]["overall"] == "FAIL"
    assert any(e["code"] == "leak_found" for e in data["errors"])
    # the docs were rewritten without the FAILed file
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    assert source not in [f["path"] for f in schema["files"]]
    assert f"`{source}`" not in (out / "README.md").read_text(encoding="utf-8")


def test_real_email_in_another_column_fails_that_file(tmp_path: Path, monkeypatch) -> None:
    email = sensitive()["emails"][3]
    _patch_writer(monkeypatch, "people.csv", "gender", lambda v: email)
    res, out, rep = _run(tmp_path)
    _assert_failed_by_leak(res, out, rep, "people.csv")
    assert (out / "enrollments.csv").exists() and (out / "staff.csv").exists()
    _clean_dirs(res, out, rep)
    reason = _file_entry(rep, "people.csv")["reasons"][-1]
    assert "'gender'" in reason and "email" in reason


def test_case_changed_real_name_in_its_own_column_fails(tmp_path: Path, monkeypatch) -> None:
    people = (SCHOOL / "people.csv").read_text(encoding="utf-8").splitlines()
    real_first = people[5].split(",")[1]
    _patch_writer(monkeypatch, "people.csv", "first_name", lambda v: real_first.upper())
    res, out, rep = _run(tmp_path)
    _assert_failed_by_leak(res, out, rep, "people.csv")
    assert "same column" in _file_entry(rep, "people.csv")["reasons"][-1]
    _clean_dirs(res, out, rep)


def test_real_id_in_another_file_of_the_namespace_fails(tmp_path: Path, monkeypatch) -> None:
    sid = sensitive()["ids"][7]
    _patch_writer(monkeypatch, "enrollments.csv", "student_id", lambda v: sid)
    res, out, rep = _run(tmp_path)
    _assert_failed_by_leak(res, out, rep, "enrollments.csv")
    assert "ID namespace" in _file_entry(rep, "enrollments.csv")["reasons"][-1]
    assert (out / "people.csv").exists()
    _clean_dirs(res, out, rep)


def test_real_value_quoted_in_a_message_is_caught_and_redacted(tmp_path: Path, monkeypatch) -> None:
    from datanonymous import pipeline
    from datanonymous.model import Message

    s = sensitive()
    last, email = s["last_names"][4], s["emails"][9]
    orig = pipeline.profile_file

    def profile_file(f, settings, run):  # noqa: ANN001
        fr = orig(f, settings, run)
        if fr is not None and fr.rel_path == "people.csv":
            fr.tables[0].messages.append(Message("warning", "buggy", f"value {last} looks odd ({email})",
                                                 column="last_name"))
        return fr

    monkeypatch.setattr(pipeline, "profile_file", profile_file)
    res, out, rep = _run(tmp_path)
    _assert_failed_by_leak(res, out, rep, "people.csv")
    _clean_dirs(res, out, rep)
    assert "[redacted]" in (rep / "REPORT.md").read_text(encoding="utf-8")
    assert "[redacted]" in (rep / "files" / "people.csv.md").read_text(encoding="utf-8")
    assert "report file REPORT.md" in " ".join(_file_entry(rep, "people.csv")["reasons"])


def test_real_value_in_the_out_dir_docs_is_caught_and_redacted(tmp_path: Path, monkeypatch) -> None:
    from datanonymous.report import docs

    ssn = sensitive()["ssns"][2]
    orig = docs._readme
    monkeypatch.setattr(docs, "_readme", lambda *a, **k: orig(*a, **k) + f"\nContact SSN {ssn}\n")
    res, out, rep = _run(tmp_path)
    _assert_failed_by_leak(res, out, rep, "people.csv")
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "[redacted]" in readme
    _clean_dirs(res, out, rep)
    assert "out-dir doc README.md" in " ".join(_file_entry(rep, "people.csv")["reasons"])


def test_leak_caught_before_writing_never_reaches_disk(tmp_path: Path, monkeypatch) -> None:
    # integration: last_name is a PERSON_NAME generated by AREA C's Faker generator (D24), no longer by the
    # foundation's invented labels, so the leak is planted in whatever generator produces the column
    from datanonymous.generate import table

    orig = table.generate_values

    def generate(profile, model, n, rng, faker_factory=None):  # noqa: ANN001
        values, flags = orig(profile, model, n, rng, faker_factory)
        if profile.name == "last_name" and n:
            values[0] = sorted(model.real_values)[0].casefold()  # a real name in lower case
        return values, flags

    monkeypatch.setattr(table, "generate_values", generate)
    res, out, rep = _run(tmp_path)
    assert res.exit_code == 3
    assert not (out / "people.csv").exists()
    entry = _file_entry(rep, "people.csv")
    assert entry["checks"]["leak_scan"] == "FAIL" and entry["checks"]["layout_selfcheck"] == "N/A"
    _clean_dirs(res, out, rep)


def test_real_value_in_a_foreign_out_dir_file_is_a_run_error(tmp_path: Path) -> None:
    out = tmp_path / "run_out"
    (out / "notes").mkdir(parents=True)
    phone = sensitive()["phones"][1]
    (out / "notes" / "old.txt").write_text(f"call back {phone}\n", encoding="utf-8")
    res, out, rep = _run(tmp_path, "--force")
    assert res.exit_code == 3
    data = _report(rep)
    assert data["totals"]["FAIL"] == 0 and data["verdict"]["overall"] == "FAIL"
    errs = [e for e in data["errors"] if "notes/old.txt" in e["text"]]
    assert errs and errs[0]["fix"] == "delete notes/old.txt from the out dir"
    assert any(c["key"] == "run_errors" and c["result"] == "FAIL" for c in data["verdict"]["criteria"])
    _clean_dirs(res, out, rep, skip=("notes/old.txt",))


@pytest.mark.parametrize("flags", [(), ("--no-default-ids",)])
def test_no_real_value_with_other_detections(tmp_path: Path, flags: tuple[str, ...]) -> None:
    res, out, rep = _run(tmp_path, *flags, "--factor-max", "5")
    assert res.exit_code in (0, 3), res.output
    _clean_dirs(res, out, rep)
