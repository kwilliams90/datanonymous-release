"""Formatted numerics get the same rare-value protection as plain numbers (ADR-0002), seen at the run."""

from __future__ import annotations

import pytest

from datanonymous.config import Settings
from datanonymous.detect.numeric import parse_with_format
from datanonymous.pipeline import execute

from release_corpus import read_records
from test_sharing_release import _diagnostic, _write_csv

_FORMATS = {
    "plain-dollar": lambda v: f"${v}",
    "grouped-dollar": lambda v: f"${v:,}",
    "grouped": lambda v: f"{v:,}",
    "parenthesised": lambda v: f"({v:,})",
    "euro-suffix": lambda v: f"{v} €",
}
# common values are held by ~60 people each; 55,000 by 3 people, 5,000 (the low extreme) by 2, 125,000 (the high extreme) by 1
_COMMON = (20000, 40000, 60000, 80000, 100000)
_RARE = {55000: 3, 5000: 2, 125000: 1}


def _run(tmp_path, render, **settings):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    numbers = [v for v, c in _RARE.items() for _ in range(c)]
    numbers += [_COMMON[i % len(_COMMON)] for i in range(300 - len(numbers))]
    _write_csv(original / "pay.csv", [{"Student_ID": f"S{i:05d}", "Grade": str(6 + i % 4), "Score": f"{(i * 7919) % 30000 / 100:.2f}", "Amount": render(v)}
                                      for i, v in enumerate(numbers)])
    run = execute(Settings(mode="sharing", seed=1, **settings), original, output, report)
    return run, output


@pytest.mark.parametrize("name", list(_FORMATS))
def test_rare_formatted_values_are_suppressed_and_the_column_keeps_its_format(tmp_path, name):
    render = _FORMATS[name]
    run, output = _run(tmp_path, render)
    assert run.package["released"], (_diagnostic(run), [f.reasons for f in run.files])
    written = [r["Amount"] for r in read_records(output / "pay.csv")]
    assert len(written) == 300
    assert all(written), "the column is written with values, not nulled"
    allowed = {render(v) for v in _COMMON}
    assert set(written) <= allowed, set(written) - allowed
    assert not {render(v) for v in _RARE} & set(written)


def test_plain_discrete_integers_are_still_suppressed_and_written(tmp_path):
    run, output = _run(tmp_path, str)
    assert run.package["released"], (_diagnostic(run), [f.reasons for f in run.files])
    written = {r["Amount"] for r in read_records(output / "pay.csv")}
    assert written and written <= {str(v) for v in _COMMON}
