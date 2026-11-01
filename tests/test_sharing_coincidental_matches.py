"""Issue 11: sharing mode tells chance whole-row matches from linkage.

Columns are generated independently from per-column classes many people hold, so in a table with few possible
combinations many synthetic rows equal a real one by chance. Sharing mode skips the check when there is only one
compared column group (no swap can change a match there), and otherwise compares the matches with a baseline made by
shuffling the column groups against each other. Matches no more common than that are written with
``row_matches_coincidental``; a planted copy of real rows is more common and still leaves the file out.

Every input here is invented.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from datanonymous.cli import main
from datanonymous.config import ColumnRef, Settings
from datanonymous.model import Message
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import finding_category

from planted_matches import plant_row_copies, write_survey
from release_corpus import read_records, student_id, write_corpus


def _rates(path: Path, with_id: bool = False) -> None:
    """The note's one-column table: 120 rows of percentages such as ``20.8%`` on a 0.1% grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["Student_ID", "Rate"] if with_id else ["Rate"])
        for i in range(120):
            rate = f"{15.0 + ((i * 37) % 201) / 10:.1f}%"
            writer.writerow([student_id(i), rate] if with_id else [rate])


def _run(original: Path, base: Path, seed: int = 573291, **settings):
    return execute(Settings(mode="sharing", seed=seed, **settings), original, base / "synthetic", base / "report")


def _file(run, name: str):
    return next(f for f in run.files if f.rel_path == name)


def _codes(fr) -> set[str]:
    return {m.code for m in fr.all_messages()}


@pytest.mark.parametrize("with_id", [False, True], ids=["one_column", "id_and_one_column"])
def test_a_table_with_one_compared_column_is_written_unchecked(tmp_path, with_id):
    original = tmp_path / "original"
    _rates(original / "rates.csv", with_id)
    run = _run(original, tmp_path)
    rates = _file(run, "rates.csv")
    assert rates.written and exit_code(run) == 0, rates.reasons
    assert not {c for c in _codes(rates) if c.startswith("row_matches_")}
    [priv] = rates.privacy
    assert not priv["checked"] and "fewer than two independent comparable column groups" in priv["note"]
    written = read_records(tmp_path / "synthetic" / "rates.csv")
    assert len(written) == 120 and all(row["Rate"].endswith("%") for row in written)


def test_a_table_cut_to_one_compared_column_by_null_is_written(tmp_path):
    original = tmp_path / "original"
    write_survey(original / "survey.csv")
    run = _run(original, tmp_path, null=[ColumnRef(c) for c in ("Track", "Lunch", "School")])
    survey = _file(run, "survey.csv")
    assert survey.written and exit_code(run) == 0, survey.reasons
    assert not survey.privacy[0]["checked"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_chance_matches_on_a_high_baseline_table_are_written(tmp_path, seed):
    original = tmp_path / "original"
    write_survey(original / "survey.csv")
    run = _run(original, tmp_path, seed)
    survey = _file(run, "survey.csv")
    assert survey.written and exit_code(run) == 0, survey.reasons
    assert "row_matches_coincidental" in _codes(survey)
    [priv] = survey.privacy
    assert priv["initial_rate"] > 0.20 and priv["coincidental"]
    assert 0.35 < priv["baseline"]["rate"] < 0.55 and priv["baseline"]["excess_risk"] <= 0.05


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_planted_ten_percent_copy_of_real_rows_is_still_refused(tmp_path, monkeypatch, seed):
    """The positive control: the same table with 10% of its synthetic rows copied from real rows matches more often
    than shuffling gives, so the file is left out as before."""
    original = tmp_path / "original"
    write_survey(original / "survey.csv")
    plant_row_copies(monkeypatch, 0.10, seed)
    run = _run(original, tmp_path, seed)
    survey = _file(run, "survey.csv")
    assert not survey.written and exit_code(run) == 3
    assert "row_matches_report_only" in _codes(survey) and "row_matches_coincidental" not in _codes(survey)
    assert not (tmp_path / "synthetic" / "survey.csv").exists()


def test_refuse_unsupported_still_leaves_out_tables_with_chance_matches(tmp_path):
    original = tmp_path / "original"
    write_corpus(original, "categorical_only")
    _rates(original / "rates.csv")
    run = _run(original, tmp_path, refuse_unsupported=True)
    refused = {f.rel_path for f in run.files if "row_matches_report_only" in _codes(f)}
    assert {"iep.csv", "roster.csv", "rates.csv"} <= refused
    assert not any(f.written for f in run.files if f.rel_path in refused)
    assert not any("row_matches_coincidental" in _codes(f) for f in run.files)
    assert exit_code(run) == 3


def test_coincidental_matches_are_a_utility_warning_explained_in_plain_words(tmp_path):
    assert finding_category(Message("warning", "row_matches_coincidental", "")) == "utility"
    original = tmp_path / "original"
    write_survey(original / "survey.csv")
    result = CliRunner().invoke(main, [str(original), "--out-dir", str(tmp_path / "synthetic"), "--report-dir",
                                       str(tmp_path / "report"), "--seed", "1"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    md = (tmp_path / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "few possible combinations" in md
    detail = (tmp_path / "report" / "files" / "survey.csv.md").read_text(encoding="utf-8")
    assert "Chance baseline:" in detail and "200 shuffles" in detail and ": coincidental" in detail
