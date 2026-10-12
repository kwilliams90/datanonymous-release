"""Review round 1, docs/config cluster.

R1-07: a doc whose name (and ``_datanonymous_`` fallback name) is taken never overwrites a synthetic data
file (D20, D15). R1-15: CLI column options supersede ``--config`` entries for the same column, however
either reference is qualified (defaults < config < CLI, design 7; D22).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_cli
from datanonymous.config import load_config
from conftest import build_fidelity_settings as build_settings
from datanonymous.errors import UsageError
from datanonymous.report.docs import doc_paths


# --------------------------------------------------------------------------- R1-07 doc-name collisions
def _records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{path.name} is not a data file: {str(data)[:80]}"
    return data


def test_doc_paths_reapplies_the_prefix_until_the_name_is_free() -> None:
    names, msgs = doc_paths(["schema.json", "_DATANONYMOUS_schema.json", "sub/README.md"])
    assert names == {"README.md": "README.md", "DATA_DICTIONARY.md": "DATA_DICTIONARY.md",
                     "schema.json": "_datanonymous__datanonymous_schema.json"}
    assert [m.code for m in msgs] == ["doc_name_collision"]
    assert "_datanonymous__datanonymous_schema.json" in msgs[0].text
    names, _ = doc_paths(["Schema.JSON", "_datanonymous_schema.json", "_datanonymous__datanonymous_schema.json"])
    assert names["schema.json"] == "_datanonymous__datanonymous__datanonymous_schema.json"


def test_doc_paths_avoids_a_root_level_folder_of_the_same_name() -> None:
    names, msgs = doc_paths(["README.md/scores.csv", "other/schema.json"])
    assert names["README.md"] == "_datanonymous_README.md"
    assert names["schema.json"] == "schema.json"  # only root-level names collide
    assert len(msgs) == 1


def test_doc_never_overwrites_a_data_file_holding_the_prefixed_name(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    (d / "schema.json").write_text('[{"score": 1}, {"score": 2}]', encoding="utf-8")
    (d / "_datanonymous_schema.json").write_text('[{"score": 3}, {"score": 4}]', encoding="utf-8")
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 0, res.output
    for name in ("schema.json", "_datanonymous_schema.json"):  # N rows in = N rows out, still data
        rows = _records(out / name)
        assert len(rows) == 2 and all(set(r) == {"score"} for r in rows)
    doc = json.loads((out / "_datanonymous__datanonymous_schema.json").read_text(encoding="utf-8"))
    assert doc["synthetic"] is True
    assert sorted(f["path"] for f in doc["files"]) == ["_datanonymous_schema.json", "schema.json"]
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    assert report["docs"]["schema"] == "_datanonymous__datanonymous_schema.json"
    assert "_datanonymous__datanonymous_schema.json" in res.output
    # the same seed still gives byte-identical output
    out2, rep2 = tmp_path / "o2", tmp_path / "r2"
    assert run_cli(d, "--out-dir", out2, "--report-dir", rep2, "--seed", "1").exit_code == 0
    for p in out.iterdir():
        assert (out2 / p.name).read_bytes() == p.read_bytes(), p.name


def test_doc_never_overwrites_a_stale_data_file(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    (d / "schema.json").write_text('[{"score": 1}, {"score": 2}]', encoding="utf-8")
    (d / "f.csv").write_text("group\n" + "A\nB\n" * 20, encoding="utf-8")
    out, rep = tmp_path / "o", tmp_path / "r"
    assert run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1").exit_code == 0
    kept = (out / "schema.json").read_bytes()
    assert isinstance(json.loads(kept), list)
    (d / "schema.json").write_text('[{"score": 1}, {"score": ', encoding="utf-8")  # now unreadable: FAIL
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1", "--force")
    assert res.exit_code == 3, res.output
    assert (out / "schema.json").read_bytes() == kept  # the stale data file is neither deleted nor overwritten
    assert json.loads((out / "_datanonymous_schema.json").read_text(encoding="utf-8"))["synthetic"] is True
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "Not regenerated in this run" in readme and "`schema.json`" in readme


def test_doc_name_taken_by_a_mirrored_folder(tmp_path: Path) -> None:
    d = tmp_path / "in"
    (d / "README.md").mkdir(parents=True)
    (d / "README.md" / "scores.csv").write_text("group\n" + "A\nB\n" * 20, encoding="utf-8")
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 0, res.output
    assert (out / "README.md" / "scores.csv").is_file()
    assert (out / "_datanonymous_README.md").read_text(encoding="utf-8").startswith("# SYNTHETIC DATA")


# --------------------------------------------------------------------------- R1-15 CLI over config
def _settings(tmp_path: Path, yaml_text: str, **cli) -> object:  # noqa: ANN003
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    return build_settings({k: list(v) if isinstance(v, (list, tuple)) else v for k, v in cli.items()},
                          load_config(cfg))


@pytest.mark.parametrize("config_ref, cli_ref, other_file", [
    ("score", "data.csv::score", "FLOAT"),  # the CLI flag targets data.csv only; the config still holds elsewhere
    ("data.csv::score", "score", "INTEGER"),
    ("score", "score", "INTEGER"),
])
def test_cli_force_type_supersedes_config_whatever_the_qualification(tmp_path: Path, config_ref: str,
                                                                     cli_ref: str, other_file: str) -> None:
    s = _settings(tmp_path, f"force_type:\n  '{config_ref}': FLOAT\n", force_type=[f"{cli_ref}=INTEGER"])
    assert s.column_options("data.csv", None, "score").force_type == "INTEGER"
    assert s.column_options("other.csv", None, "score").force_type == other_file
    assert s.unused_refs() == []  # a superseded config entry matched a column: no "matched no column" warning


def test_cli_date_format_supersedes_config(tmp_path: Path) -> None:
    s = _settings(tmp_path, "date_format:\n  when: '%m/%d/%Y'\n", date_format=["data.csv::when=%d/%m/%Y"])
    assert s.column_options("data.csv", None, "when").date_format == "%d/%m/%Y"
    assert s.column_options("b.csv", None, "when").date_format == "%m/%d/%Y"


def test_cli_force_text_supersedes_config_force_factor(tmp_path: Path) -> None:
    s = _settings(tmp_path, "force_factor: [x, y]\n", force_text=["x"])
    o = s.column_options("data.csv", None, "x")
    assert o.force_text and not o.force_factor
    assert s.column_options("data.csv", None, "y").force_factor  # config still applies elsewhere


@pytest.mark.parametrize("yaml_text, cli, expect", [
    ("org_code: [x]\n", {"force_text": ["x"]}, "force_text"),
    ("force_text: [x]\n", {"force_factor": ["a.csv::x"]}, "force_factor"),
    ("id_col: [x]\n", {"force_factor": ["x"]}, "force_factor"),
    ("person_id: [x]\n", {"org_code": ["x"]}, "org_code"),
    ("force_type:\n  x: FLOAT\n", {"force_factor": ["x"]}, "force_factor"),
    ("pii_col:\n  x: name\n", {"force_text": ["x"]}, "force_text"),
])
def test_a_cli_type_option_supersedes_every_config_type_option(tmp_path: Path, yaml_text: str, cli: dict,
                                                               expect: str) -> None:
    o = _settings(tmp_path, yaml_text, **cli).column_options("a.csv", None, "x")
    on = {k for k in ("force_type", "pii_col", "id_col", "person_id", "org_code", "force_factor", "force_text")
          if getattr(o, k)}
    assert on == {expect}


def test_null_and_exclude_precedence(tmp_path: Path) -> None:
    o = _settings(tmp_path, "null: [x]\n", exclude=["x"]).column_options("a.csv", None, "x")
    assert o.exclude and not o.null
    o = _settings(tmp_path, "exclude: [x]\n", null=["a.csv::x"]).column_options("a.csv", None, "x")
    assert o.null and not o.exclude
    # a CLI type option never brings back a column the config drops or blanks
    o = _settings(tmp_path, "exclude: [x]\n", force_factor=["x"]).column_options("a.csv", None, "x")
    assert o.exclude and o.force_factor


def test_conflicts_within_one_source_stay_usage_errors(tmp_path: Path) -> None:
    s = _settings(tmp_path, "", force_factor=["x"], force_text=["a.csv::x"])
    with pytest.raises(UsageError, match="both --force-factor and --force-text"):
        s.column_options("a.csv", None, "x")
    s = _settings(tmp_path, "force_factor: [x]\nforce_text: [x]\n")
    with pytest.raises(UsageError, match="both --force-factor and --force-text"):
        s.column_options("a.csv", None, "x")
    s = _settings(tmp_path, "force_type:\n  score: FLOAT\n  a.csv::score: INTEGER\n")
    with pytest.raises(UsageError, match="conflicting --force-type"):
        s.column_options("a.csv", None, "score")
    s = _settings(tmp_path, "force_type:\n  score: FLOAT\n", force_type=["score=INTEGER", "a.csv::score=DATE"])
    with pytest.raises(UsageError, match="conflicting --force-type"):
        s.column_options("a.csv", None, "score")


def test_cli_overrides_config_end_to_end(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    rows = ["id,score,x"] + [f"{1000 + i},{i},secret value {i}" for i in range(30)]
    (d / "options.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("factor_max: 2\nforce_type:\n  score: FLOAT\nforce_factor: [x]\n", encoding="utf-8")
    res = run_cli("inspect", d / "options.csv", "--config", cfg, "--force-type", "options.csv::score=INTEGER",
                  "--force-text", "x")
    assert res.exit_code == 0, res.output
    lines = {ln.split()[0]: ln for ln in res.output.splitlines() if ln.startswith("   ") and ln.split()}
    assert "--force-type INTEGER" in lines["score"]
    assert "--force-text" in lines["x"] and "FACTOR" not in lines["x"]
    out, rep = tmp_path / "o", tmp_path / "r"
    res = run_cli(d, "--out-dir", out, "--report-dir", rep, "--seed", "3", "--config", cfg,
                  "--force-text", "options.csv::x")
    assert res.exit_code == 0, res.output
    text = (out / "options.csv").read_text(encoding="utf-8")
    assert "secret value" not in text  # --force-text won: no real string copied
    assert len(text.splitlines()) == 31
