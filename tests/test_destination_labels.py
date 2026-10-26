"""Issue 19: every run labels its two destinations, at the start and at the end of the console output, and the
report folder carries the PRIVATE-DO-NOT-SHARE.txt marker. Tests run the CLI and read only what an operator sees."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous import rundirs
from datanonymous.cli import main
from datanonymous.config import user_config_path

from release_corpus import student_id

SHAREABLE = "Shareable synthetic files: "
PRIVATE = "PRIVATE report - do not share: "
MARKER = "PRIVATE-DO-NOT-SHARE.txt"


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(rundirs, "now", lambda: datetime(2026, 3, 4, 5, 6, 7))


@pytest.fixture
def user_config():
    def write(text: str) -> Path:
        path = user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    return write


def _roster(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "roster.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Student_ID", "Score", "Other_Score"])
        for i in range(60):
            writer.writerow([student_id(i), f"{(5100 + i * 73) / 100:.2f}", f"{(4000 + (i * 571) % 3999) / 100:.2f}"])
    return folder


def _run(*args, mode: str = "sharing"):
    return CliRunner().invoke(main, ["run", *map(str, args), "--mode", mode, "--seed", "573291"],
                              catch_exceptions=False)


def _labelled(lines: list[str], label: str) -> list[int]:
    return [i for i, line in enumerate(lines) if line.startswith(label)]


def _assert_labels_at_start_and_end(output: str, out: Path, report: Path) -> None:
    lines = output.splitlines()
    summary = next(i for i, line in enumerate(lines) if line.startswith("datanonymous "))
    shareable, private = _labelled(lines, SHAREABLE), _labelled(lines, PRIVATE)
    assert len(shareable) == 2 and len(private) == 2, output
    # the start pair comes before the summary, the end pair closes the output
    assert shareable[0] < summary and private[0] < summary, output
    assert shareable[1] == len(lines) - 2 and private[1] == len(lines) - 1, output
    assert str(out) in lines[shareable[0]] and str(out) in lines[shareable[1]], output
    assert str(report) in lines[private[0]] and str(report) in lines[private[1]], output


def test_sharing_run_prints_both_labelled_destinations_at_start_and_end(tmp_path):
    source = _roster(tmp_path / "inbox" / "roster")
    out, report = tmp_path / "synthetic", tmp_path / "private" / "report"
    res = CliRunner().invoke(main, [str(source), "--out-dir", str(out), "--report-dir", str(report),
                                    "--seed", "573291"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    _assert_labels_at_start_and_end(res.output, out, report)
    assert res.output.isascii()


def test_report_folder_has_the_marker_and_the_output_folder_does_not(tmp_path):
    source = _roster(tmp_path / "inbox" / "roster")
    out, report = tmp_path / "synthetic", tmp_path / "private" / "report"
    res = CliRunner().invoke(main, [str(source), "--out-dir", str(out), "--report-dir", str(report),
                                    "--seed", "573291"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    marker = (report / MARKER).read_text(encoding="utf-8")
    assert "real source data" in marker and "never share" in marker
    assert "PRIVATE-DO-NOT-SHARE.txt" in (report / "REPORT.md").read_text(encoding="utf-8")
    assert not any(p.name == MARKER for p in out.rglob("*"))


def test_standing_roots_label_the_resolved_run_folders(tmp_path, user_config, clock):
    out_root, rep_root = tmp_path / "out-root", tmp_path / "private" / "report-root"
    user_config(f"output_root: '{out_root.as_posix()}'\nreport_root: '{rep_root.as_posix()}'\n")
    source = _roster(tmp_path / "inbox" / "roster")
    res = _run(source)
    assert res.exit_code == 0, res.output
    name = "roster-20260304-050607"
    _assert_labels_at_start_and_end(res.output, out_root / name, rep_root / name)


def test_labels_screen_a_name_in_the_input_folder_name(tmp_path, user_config, clock):
    out_root, rep_root = tmp_path / "out-root", tmp_path / "private" / "report-root"
    user_config(f"output_root: '{out_root.as_posix()}'\nreport_root: '{rep_root.as_posix()}'\n")
    source = _roster(tmp_path / "in" / "Okonkwo_Chinedu")
    res = _run(source)
    assert res.exit_code == 0, res.output
    assert "Okonkwo" not in res.output and "Chinedu" not in res.output


def test_fidelity_runs_print_the_same_labels(tmp_path):
    source = _roster(tmp_path / "inbox" / "roster")
    out, report = tmp_path / "synthetic", tmp_path / "private" / "report"
    res = CliRunner().invoke(main, [str(source), "--out-dir", str(out), "--report-dir", str(report),
                                    "--mode", "fidelity", "--seed", "573291"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    _assert_labels_at_start_and_end(res.output, out, report)
    assert (report / MARKER).is_file() and not (out / MARKER).exists()
