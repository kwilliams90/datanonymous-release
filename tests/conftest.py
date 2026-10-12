"""Shared fixtures. Outputs always go to pytest's tmp_path; example-data/ is read-only."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import build_settings

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "example-data"
EXAMPLE_FILES = ("StudentsPerformance.csv", "student-scores.csv", "students_data.csv")


def run_cli(*args: str):
    """Invoke the CLI with explicit fidelity compatibility for legacy tests."""
    argv = [str(a) for a in args]
    if argv not in (["--help"], ["--version"]) and not any(
        a == "--mode" or a.startswith("--mode=") for a in argv
    ):
        argv += ["--mode", "fidelity"]
    return CliRunner().invoke(main, argv, catch_exceptions=False)


def build_fidelity_settings(cli, config=None):
    """Build legacy fidelity settings while preserving configuration precedence."""
    return build_settings({"mode": "fidelity", **cli}, config)


def read_rows(path: Path, encoding: str = "utf-8") -> list[list[str]]:
    with open(path, newline="", encoding=encoding) as fh:
        return list(csv.reader(fh))


def read_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class ExampleRun:
    out: Path
    rep: Path
    result: object


@pytest.fixture(scope="session")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> ExampleRun:
    """One CLI run over example-data with --seed 1 (shared by many tests)."""
    base = tmp_path_factory.mktemp("example_run")
    out, rep = base / "out", base / "rep"
    res = run_cli(EXAMPLES, "--out-dir", out, "--report-dir", rep, "--seed", "1")
    assert res.exit_code == 0, res.output
    return ExampleRun(out, rep, res)


@pytest.fixture
def example_copy(tmp_path: Path) -> Path:
    """A writable copy of example-data inside tmp_path."""
    dst = tmp_path / "input"
    shutil.copytree(EXAMPLES, dst)
    return dst
