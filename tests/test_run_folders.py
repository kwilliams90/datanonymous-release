"""Standing config: user config file, output/report roots and per-run folders (issue #10)."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous import rundirs
from datanonymous.cli import main
from datanonymous.config import user_config_path


@pytest.fixture
def user_config():
    """Write the (redirected) user config file."""
    def write(text: str) -> Path:
        path = user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    return write


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(rundirs, "now", lambda: datetime(2026, 3, 4, 5, 6, 7))


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "inbox" / "grades"
    folder.mkdir(parents=True)
    with open(folder / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "score"])
        for i in range(30):
            w.writerow([f"S{i:04d}", i % 7])
    return folder


def run(*args, seed="1"):
    return CliRunner().invoke(main, ["run", *map(str, args), "--mode", "fidelity", "--seed", seed])


def roots(tmp_path):
    return tmp_path / "out-root", tmp_path / "private" / "report-root"


def config_with_roots(tmp_path, user_config):
    out_root, rep_root = roots(tmp_path)
    return user_config(f"output_root: '{out_root.as_posix()}'\nreport_root: '{rep_root.as_posix()}'\n")


def test_user_config_roots_give_matching_timestamped_folders(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    res = run(source)
    assert res.exit_code == 0, res.output
    name = "grades-20260304-050607"
    assert (out_root / name / "scores.csv").is_file()
    assert (rep_root / name / "REPORT.md").is_file()
    assert [p.name for p in out_root.iterdir()] == [name]
    assert [p.name for p in rep_root.iterdir()] == [name]


def test_a_single_file_input_is_named_by_its_stem(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, _ = roots(tmp_path)
    assert run(source / "scores.csv").exit_code == 0
    assert (out_root / "scores-20260304-050607").is_dir()


def test_two_runs_in_the_same_second_get_numbered_folders(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    for _ in range(3):
        assert run(source).exit_code == 0
    names = sorted(p.name for p in out_root.iterdir())
    assert names == ["grades-20260304-050607", "grades-20260304-050607-2", "grades-20260304-050607-3"]
    assert sorted(p.name for p in rep_root.iterdir()) == names


def test_a_collision_in_either_root_moves_both_to_the_next_number(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    (rep_root / "grades-20260304-050607").mkdir(parents=True)
    assert run(source).exit_code == 0
    assert (out_root / "grades-20260304-050607-2" / "scores.csv").is_file()
    assert (rep_root / "grades-20260304-050607-2" / "REPORT.md").is_file()


def test_out_dir_overrides_output_root(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    explicit = tmp_path / "explicit"
    assert run(source, "--out-dir", explicit).exit_code == 0
    assert (explicit / "scores.csv").is_file()
    assert not out_root.exists()
    # report_root still says where this run's report goes
    assert (rep_root / "grades-20260304-050607" / "REPORT.md").is_file()


def test_out_dir_and_report_dir_override_both_roots(tmp_path, user_config, source):
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    assert run(source, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r").exit_code == 0
    assert (tmp_path / "o" / "scores.csv").is_file() and (tmp_path / "r" / "REPORT.md").is_file()
    assert not out_root.exists() and not rep_root.exists()


def test_config_option_overrides_the_user_config(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    other = tmp_path / "other"
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"output_root: '{(other / 'o').as_posix()}'\nreport_root: '{(other / 'r').as_posix()}'\n")
    assert run(source, "--config", cfg).exit_code == 0
    assert (other / "o" / "grades-20260304-050607" / "scores.csv").is_file()
    assert not roots(tmp_path)[0].exists()


def test_command_line_overrides_config_and_user_config(tmp_path, user_config, source):
    user_config("seed: 5\n")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("seed: 6\n")

    def invoke(label, *extra):
        res = CliRunner().invoke(main, ["run", str(source), "--out-dir", str(tmp_path / label),
                                        "--mode", "fidelity", *map(str, extra)])
        assert res.exit_code == 0, res.output
        return res.output

    assert "seed=5" in invoke("user")
    assert "seed=6" in invoke("config", "--config", cfg)
    assert "seed=7" in invoke("cli", "--config", cfg, "--seed", "7")


def test_summary_names_the_loaded_config_files(tmp_path, user_config, source):
    path = user_config("min_cell: 10\n")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("min_cell: 10\n")
    res = run(source, "--out-dir", tmp_path / "o", "--config", cfg)
    assert str(path) in res.output and str(cfg) in res.output


def test_summary_names_no_config_when_none_loaded(tmp_path, source):
    res = run(source, "--out-dir", tmp_path / "o")
    assert "Config loaded" not in res.output


def test_out_dir_is_still_required_without_output_root(tmp_path, source):
    res = run(source)
    assert res.exit_code == 2
    assert "--out-dir is required" in res.output


def test_report_root_alone_sets_where_reports_go(tmp_path, user_config, clock, source):
    user_config(f"report_root: '{(tmp_path / 'reports').as_posix()}'\n")
    assert run(source, "--out-dir", tmp_path / "o").exit_code == 0
    assert (tmp_path / "reports" / "grades-20260304-050607" / "REPORT.md").is_file()
    assert not (tmp_path / "o-report").exists()


def test_output_root_without_report_root_is_refused(tmp_path, user_config, source):
    user_config(f"output_root: '{(tmp_path / 'outs').as_posix()}'\n")
    for extra in ([], ["--out-dir", tmp_path / "o"]):
        res = run(source, *extra)
        assert res.exit_code == 2
        assert "output_root is set but report_root is not" in res.output
    assert not (tmp_path / "outs").exists() and not (tmp_path / "o").exists()


@pytest.mark.parametrize("out_rel,rep_rel", [("a", "a/b"), ("a/b", "a"), ("a", "a")])
def test_nested_roots_are_refused(tmp_path, user_config, source, out_rel, rep_rel):
    user_config(f"output_root: '{(tmp_path / out_rel).as_posix()}'\n"
                f"report_root: '{(tmp_path / rep_rel).as_posix()}'\n")
    res = run(source)
    assert res.exit_code == 2
    assert "inside one another" in res.output


@pytest.mark.parametrize("key", ["output_root", "report_root"])
def test_roots_inside_the_input_are_refused(tmp_path, user_config, source, key):
    inside = (source / "runs").as_posix()
    elsewhere = (tmp_path / "elsewhere").as_posix()
    other = "report_root" if key == "output_root" else "output_root"
    user_config(f"{key}: '{inside}'\n{other}: '{elsewhere}'\n")
    res = run(source)
    assert res.exit_code == 2
    assert f"{key} " in res.output and "inside the input" in res.output
    assert not (source / "runs").exists()


def test_a_taken_name_is_never_written_into(tmp_path, user_config, clock, source):
    config_with_roots(tmp_path, user_config)
    out_root, _ = roots(tmp_path)
    (out_root / "grades-20260304-050607").mkdir(parents=True)
    (out_root / "grades-20260304-050607" / "x.txt").write_text("keep")
    assert run(source).exit_code == 0
    assert [p.name for p in (out_root / "grades-20260304-050607").iterdir()] == ["x.txt"]


def test_sharing_run_uses_the_roots(tmp_path, user_config, clock):
    from release_corpus import write_corpus

    original = tmp_path / "school"
    write_corpus(original, "canonical")
    config_with_roots(tmp_path, user_config)
    out_root, rep_root = roots(tmp_path)
    for _ in range(2):
        res = CliRunner().invoke(main, ["run", str(original), "--seed", "573291"])
        assert res.exit_code == 0, res.output
    names = sorted(p.name for p in out_root.iterdir())
    assert names == ["school-20260304-050607", "school-20260304-050607-2"]
    assert sorted(p.name for p in rep_root.iterdir()) == names
    assert (out_root / names[1] / "README.md").is_file()
    assert (rep_root / names[1] / "REPORT.md").is_file()


def test_user_config_path_follows_the_platform_folders(monkeypatch):
    monkeypatch.delenv("DATANONYMOUS_CONFIG_DIR")
    monkeypatch.setenv("APPDATA", str(Path("appdata")))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(Path("xdg")))
    if sys.platform == "win32":
        assert user_config_path() == Path("appdata") / "datanonymous" / "config.yaml"
    elif sys.platform != "darwin":
        assert user_config_path() == Path("xdg") / "datanonymous" / "config.yaml"
