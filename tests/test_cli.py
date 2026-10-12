"""CLI end to end (acceptance criterion 5) and option handling."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import EXAMPLE_FILES, EXAMPLES, REPO, read_dicts, run_cli


def listing(d: Path) -> list[str]:
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*"))


def test_example_run_outputs(example_run) -> None:
    assert sorted(p.name for p in example_run.out.iterdir()) == sorted(
        [*EXAMPLE_FILES, "README.md", "DATA_DICTIONARY.md", "schema.json"])
    rep = example_run.rep
    for f in ("REPORT.md", "report.json", "manifest.json"):
        assert (rep / f).is_file()
    for name in EXAMPLE_FILES:
        assert (rep / "files" / (name + ".md")).is_file()
        assert (rep / "files" / (name + ".json")).is_file()
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert data["totals"]["FAIL"] == 0 and data["totals"]["files"] == 3
    manifest = json.loads((rep / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 1 and manifest["versions"]["Faker"] == "40.40.0"
    schema = json.loads((example_run.out / "schema.json").read_text(encoding="utf-8"))
    sp = schema["files"][0]
    assert sp["path"] == "StudentsPerformance.csv"
    assert sp["tables"][0]["readr_col_types"] == "fffffiii"


def test_subprocess_entry_point(tmp_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    res = subprocess.run([sys.executable, "-m", "datanonymous", "--mode", "fidelity", str(EXAMPLES / "students_data.csv"),
                          "--out-dir", str(tmp_path / "o"), "--seed", "1", "-q"],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "o" / "students_data.csv").is_file()
    assert (tmp_path / "o-report" / "REPORT.md").is_file()  # default report dir is a sibling


def test_skip_unsupported_and_fail_broken(example_copy: Path, tmp_path: Path) -> None:
    (example_copy / "notes.pdf").write_bytes(b"%PDF-1.4 not a table")
    sub = example_copy / "sub"
    sub.mkdir()
    (sub / "broken.csv").write_bytes(b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    shutil.copy(EXAMPLES / "students_data.csv", sub / "students_data.csv")
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(example_copy, "--out-dir", out, "--report-dir", rep, "--seed", "4")
    assert res.exit_code == 3, res.output
    files = listing(out)
    assert "notes.pdf" not in files and "sub/broken.csv" not in files
    assert "sub/students_data.csv" in files and "StudentsPerformance.csv" in files
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert sorted(s["path"] for s in data["skipped"]) == ["README.md", "generate_examples.py", "notes.pdf"]
    fails = {f["source"]: f for f in data["failures"]}
    assert set(fails) == {"sub/broken.csv"} and fails["sub/broken.csv"]["fixes"]
    assert "FAIL" in (rep / "REPORT.md").read_text(encoding="utf-8")


def test_inspect_writes_nothing(tmp_path: Path) -> None:
    before = listing(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        res = run_cli("inspect", EXAMPLES)
    finally:
        os.chdir(cwd)
    assert res.exit_code == 0, res.output
    assert "IDENTIFIER" in res.output and "ORDINAL" in res.output
    assert listing(tmp_path) == before
    assert listing(EXAMPLES) == sorted([*EXAMPLE_FILES, "README.md", "generate_examples.py"])
    res = run_cli("inspect", EXAMPLES, "--json")
    payload = json.loads(res.output)
    assert len(payload["files"]) == 3


def test_directory_safety(example_copy: Path, tmp_path: Path) -> None:
    res = run_cli(example_copy, "--out-dir", example_copy / "out")
    assert res.exit_code == 2
    out = tmp_path / "o"
    res = run_cli(example_copy, "--out-dir", out, "--report-dir", out / "rep")
    assert res.exit_code == 2
    out.mkdir()
    (out / "keep.txt").write_text("x")
    res = run_cli(example_copy, "--out-dir", out, "--report-dir", tmp_path / "r")
    assert res.exit_code == 2 and "--force" in res.output
    res = run_cli(example_copy, "--out-dir", out, "--report-dir", tmp_path / "r", "--force", "-q")
    assert res.exit_code == 0


def test_usage_errors(tmp_path: Path) -> None:
    assert run_cli(EXAMPLES, "--out-dir", tmp_path / "o", "--force-type", "x=NOPE").exit_code == 2
    assert run_cli(EXAMPLES, "--out-dir", tmp_path / "o", "--factor-max", "0").exit_code == 2
    assert run_cli(tmp_path / "missing", "--out-dir", tmp_path / "o").exit_code == 2
    assert run_cli("inspect", EXAMPLES, "--config", tmp_path / "nope.yaml").exit_code == 2
    assert run_cli("--version").exit_code == 0


def test_config_precedence_and_qualifiers(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("factor_max: 10\nforce_factor:\n  - students_data.csv::Age\nseed: 9\n", encoding="utf-8")
    res = run_cli("inspect", EXAMPLES, "--config", cfg, "--json")
    assert res.exit_code == 0
    files = {f["source"]: f for f in json.loads(res.output)["files"]}
    cols = {c["name"]: c for c in files["students_data.csv"]["tables"][0]["columns"]}
    assert cols["Age"]["type"] == "FACTOR" and cols["Age"]["source"] == "force-factor"
    # factor_max 10 from config: career_aspiration (17 levels) is no longer a factor
    cols2 = {c["name"]: c for c in files["student-scores.csv"]["tables"][0]["columns"]}
    assert cols2["career_aspiration"]["type"] == "HIGH_CARD_CATEGORICAL"
    # CLI beats config
    res = run_cli("inspect", EXAMPLES, "--config", cfg, "--factor-max", "20", "--json")
    files = {f["source"]: f for f in json.loads(res.output)["files"]}
    cols2 = {c["name"]: c for c in files["student-scores.csv"]["tables"][0]["columns"]}
    assert cols2["career_aspiration"]["type"] == "FACTOR"
    bad = tmp_path / "bad.yaml"
    bad.write_text("factor_maxx: 3\n", encoding="utf-8")
    assert run_cli("inspect", EXAMPLES, "--config", bad).exit_code == 2


def test_null_exclude_and_unused_refs(tmp_path: Path) -> None:
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(EXAMPLES / "students_data.csv", "--out-dir", out, "--report-dir", rep, "--seed", "1",
                  "--null", "Age", "--exclude", "Course", "--id-col", "NoSuchColumn")
    assert res.exit_code == 0, res.output
    rows = read_dicts(out / "students_data.csv")
    assert "Course" not in rows[0] and all(r["Age"] == "" for r in rows)
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    texts = " ".join(w["text"] for w in data["warnings"])
    assert "NoSuchColumn" in texts
    assert any(w.get("column") == "Course" and "--exclude" in w["text"] for w in data["warnings"])


def test_default_command_and_json(tmp_path: Path) -> None:
    res = run_cli("run", EXAMPLES / "StudentsPerformance.csv", "--out-dir", tmp_path / "o", "--json", "--seed", "3")
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["files"][0]["status"] in ("PASS", "WARN") and data["seed"] == 3
    res = run_cli("--seed", "3", EXAMPLES / "StudentsPerformance.csv", "--out-dir", tmp_path / "o2", "-q")
    assert res.exit_code == 0 and res.output == ""


def test_random_seed_recorded(tmp_path: Path) -> None:
    res = run_cli(EXAMPLES / "students_data.csv", "--out-dir", tmp_path / "o")
    assert res.exit_code == 0 and "random" in res.output
    manifest = json.loads((tmp_path / "o-report" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_was_random"] and isinstance(manifest["seed"], int)


def test_force_type_unavailable_fails_file_not_run(tmp_path: Path) -> None:
    res = run_cli(EXAMPLES, "--out-dir", tmp_path / "o", "--seed", "1", "--force-type",
                  "students_data.csv::Age=DATE")
    assert res.exit_code == 3
    assert not (tmp_path / "o" / "students_data.csv").exists()
    assert (tmp_path / "o" / "StudentsPerformance.csv").exists()


# --------------------------------------------------------------------------- review fixes: IDs across files
def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def test_cross_file_row_number_ids_stay_identical(tmp_path: Path) -> None:
    d = tmp_path / "in"
    _write(d / "students.csv", ["student_id,gender"] + [f"{i},{'F' if i % 3 else 'M'}" for i in range(1, 201)])
    child = [str(5 + (i * 7) % 196) for i in range(400)]
    _write(d / "scores.csv", ["student_id,score"] + [f"{s},{i % 90}" for i, s in enumerate(child)])
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0, res.output
    assert [r["student_id"] for r in read_dicts(tmp_path / "o" / "scores.csv")] == child
    assert [r["student_id"] for r in read_dicts(tmp_path / "o" / "students.csv")] == [str(i) for i in range(1, 201)]


def test_generic_id_columns_are_scoped_per_file(tmp_path: Path) -> None:
    d = tmp_path / "in"
    _write(d / "students.csv", ["id,g"] + [f"{i},{'F' if i % 2 else 'M'}" for i in range(1, 101)])
    _write(d / "courses.csv", ["id,c"] + [f"{i},{'A' if i % 2 else 'B'}" for i in range(1001, 1051)])
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0, res.output
    assert [r["id"] for r in read_dicts(tmp_path / "o" / "students.csv")] == [str(i) for i in range(1, 101)]
    courses = [int(r["id"]) for r in read_dicts(tmp_path / "o" / "courses.csv")]
    assert all(1000 <= c < 10000 for c in courses) and not set(courses) & set(range(1001, 1051))


def test_dense_ids_with_a_deleted_row_pass(tmp_path: Path) -> None:
    src = (EXAMPLES / "student-scores.csv").read_bytes().split(b"\n")
    d = tmp_path / "in"
    d.mkdir()
    (d / "student-scores.csv").write_bytes(b"\n".join(src[:49] + src[50:]))
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0, res.output
    real = {r["id"] for r in read_dicts(d / "student-scores.csv")}
    syn = [r["id"] for r in read_dicts(tmp_path / "o" / "student-scores.csv")]
    assert not set(syn) & real and [int(s) for s in syn] == sorted(int(s) for s in syn)
    assert "id_width_changed" in (tmp_path / "r" / "report.json").read_text(encoding="utf-8")


def test_person_id_and_person_name_headers_never_copy_values(tmp_path: Path) -> None:
    d = tmp_path / "in"
    names = [f"{f} {l}" for f in ("Ann", "Bob", "Cara") for l in ("Smith", "Jones")]
    rows = ["student_id,Family_ID,Guardian_Number,Section_Teacher,Mother_Name,SchoolID"]
    for i in range(1, 121):
        rows.append(f"{i},{5000 + (i * 7) % 151},{1 + i % 40},{names[i % 6]},{names[(i + 1) % 6]},"
                    f"{577 if i % 2 else 530}")
    _write(d / "p.csv", rows)
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    assert res.exit_code == 0, res.output
    real, syn = read_dicts(d / "p.csv"), read_dicts(tmp_path / "o" / "p.csv")
    for c in ("Family_ID", "Guardian_Number", "Section_Teacher", "Mother_Name"):
        assert not {r[c] for r in real} & {s[c] for s in syn}, c
    assert {s["SchoolID"] for s in syn} == {"577", "530"}  # org code keeps real values (D10)
    text = (tmp_path / "o" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert not any(n in text for n in names)


# --------------------------------------------------------------------------- review fixes: usage errors, headers, prose
def test_contradictory_column_flags_exit_2_before_writing(tmp_path: Path) -> None:
    for flags in (["--force-factor", "gender", "--force-text", "gender"], ["--null", "gender", "--exclude", "gender"],
                  ["--force-type", "gender=FACTOR", "--force-type", "gender=BOOLEAN"],
                  ["--pii-col", "gender=bogus"]):
        out = tmp_path / ("o" + str(len(list(tmp_path.iterdir()))))
        res = run_cli(EXAMPLES / "StudentsPerformance.csv", "--out-dir", out, *flags)
        assert res.exit_code == 2, (flags, res.output)
        assert not out.exists() and not Path(str(out) + "-report").exists()
    assert run_cli("inspect", EXAMPLES / "StudentsPerformance.csv", "--null", "gender", "--exclude",
                   "gender").exit_code == 2


def test_headerless_file_fails_and_quotes_nothing(tmp_path: Path) -> None:
    d = tmp_path / "in"
    _write(d / "headerless.csv", ["0414003,Philippa,Quackenbush,philippa.q@realschool.k12.us,766-16-2186"]
           + [f"0{100000 + i},Ann,Lee,a{i}@realschool.k12.us,123-45-{6000 + i}" for i in range(20)])
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 3
    assert not (out / "headerless.csv").exists()
    texts = [res.output] + [p.read_text(encoding="utf-8") for p in rep.rglob("*") if p.is_file()]
    texts += [p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file()]
    for t in texts:
        assert "Philippa" not in t and "realschool" not in t and "766-16" not in t and "0414003" not in t
    assert "add a header row" in res.output


def test_prose_txt_is_skipped_and_prose_csv_fails(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    sentence = b"Contact Zebulon Abernathy at zebulon@realschool.k12.us for access."
    (d / "readme.txt").write_bytes(sentence)
    shutil.copy(EXAMPLES / "students_data.csv", d / "students_data.csv")
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 0, res.output
    assert not (out / "readme.txt").exists()
    data = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in data["skipped"]] == ["readme.txt"]
    for p in list(rep.rglob("*")) + list(out.rglob("*")):
        if p.is_file():
            assert "Zebulon" not in p.read_text(encoding="utf-8"), p
    (d / "readme.txt").unlink()
    (d / "notes.csv").write_bytes(sentence)
    res = run_cli(d, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "1")
    assert res.exit_code == 3 and "Zebulon" not in res.output


def test_stale_output_of_a_failed_file_is_reported(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(EXAMPLES / "students_data.csv", d / "students_data.csv")
    _write(d / "b.csv", ["a,b", "1,2", "3,4"])
    out, rep = tmp_path / "o", tmp_path / "r"
    assert run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1").exit_code == 0
    _write(d / "b.csv", ["a,b", "1,2", "3,4,5"])  # now ragged: FAIL
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1", "--force")
    assert res.exit_code == 3
    assert (out / "b.csv").exists()  # never deleted
    assert "stale_output" in (rep / "report.json").read_text(encoding="utf-8")
    assert "Not regenerated in this run" in (out / "README.md").read_text(encoding="utf-8")
    assert "b.csv" in res.output


# --------------------------------------------------------------------------- review fixes: config
def test_config_null_key_and_structure_switches(tmp_path: Path) -> None:
    from datanonymous.config import load_config
    from conftest import build_fidelity_settings as build_settings
    from datanonymous.errors import UsageError

    cfg = tmp_path / "c.yaml"
    cfg.write_text("null:\n  - gender\nstructure:\n  disable: [functional_dependencies]\n", encoding="utf-8")
    s = build_settings({}, load_config(cfg))
    assert [r.raw for r in s.null] == ["gender"]
    assert s.structure_disabled("functional_dependencies") and not s.structure_disabled("composite_keys")
    res = run_cli("inspect", EXAMPLES / "StudentsPerformance.csv", "--config", cfg)
    assert res.exit_code == 0 and "NULL" in res.output
    cfg.write_text("structure:\n  disable: [everything]\n", encoding="utf-8")
    with pytest.raises(UsageError):
        build_settings({}, load_config(cfg))


# --------------------------------------------------------------------------- review fixes: writer and leak-scan hooks
def test_writer_gets_the_seed_and_leak_scan_sees_the_report_before_it_is_written(tmp_path: Path, monkeypatch) -> None:
    import dataclasses

    from datanonymous.io import registry
    from datanonymous.model import Message
    from datanonymous.privacy import leaks

    seen: dict = {}
    h = registry.FORMATS["csv"]

    def write(req) -> None:  # noqa: ANN001
        seen["seed"] = req.seed
        seen["draw"] = int(req.rng_for("titles").integers(1 << 30))
        h.write(req)

    def scan(run, payload):  # noqa: ANN001
        seen["payload"] = sorted(payload)
        seen["written"] = (run.report_dir / "REPORT.md").exists()
        return [Message("error", "leak_found", "test leak in the report")]

    monkeypatch.setitem(registry.FORMATS, "csv", dataclasses.replace(h, write=write))
    monkeypatch.setattr(leaks, "scan_outputs", scan)
    rep = tmp_path / "r"
    res = run_cli(EXAMPLES / "students_data.csv", "--out-dir", tmp_path / "o", "--report-dir", rep, "--seed", "7")
    assert res.exit_code == 3
    assert seen["seed"] == 7 and not seen["written"]
    assert {"REPORT.md", "report.json", "manifest.json", "files/students_data.csv.md"} <= set(seen["payload"])
    assert "test leak in the report" in (rep / "REPORT.md").read_text(encoding="utf-8")
    first = seen["draw"]
    run_cli(EXAMPLES / "students_data.csv", "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2",
            "--seed", "7")
    assert seen["draw"] == first  # writer randomness is reproducible (D26)


def test_repeated_header_row_in_the_body_is_not_refused_on_self_check(tmp_path: Path) -> None:
    from datanonymous.io import csvio

    rows = ["grp,lvl"] + [("grp,lvl" if i == 150 else f"{'AB'[i % 2]},{'XY'[(i // 2) % 2]}") for i in range(300)]
    _write(tmp_path / "in" / "r.csv", rows)
    out = tmp_path / "o"
    # --min-cell 1: the two header cells held by one row stay real values (D35 would pseudonymise them)
    res = run_cli(tmp_path / "in", "--out-dir", out, "--report-dir", tmp_path / "r", "--seed", "1", "--min-cell", "1")
    assert res.exit_code == 0, res.output
    # the synthetic body repeats header cells without repeating the header row: only the source is checked
    with pytest.raises(csvio.HeaderLooksLikeData):
        csvio.parse_text_bytes((out / "r.csv").read_bytes())
