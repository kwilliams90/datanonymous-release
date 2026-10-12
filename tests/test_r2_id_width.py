"""R2-15 / D9 (user decision 2026-10-05): only an unpadded integer ID may change its number of digits when its
own width has too few non-real values; an ID written zero-padded anywhere keeps its padded width or FAILs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import run_cli
from datanonymous.errors import IdSpaceError
from datanonymous.generate.ids import allocate_digit_ids

DENSE = [str(i) for i in range(1, 2001) if i != 7]  # 1..2000 minus one gap: width 1 has only '0' and '7' free


def _alloc(pad_limit: dict[str, int] | None) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    real = np.asarray(sorted(int(k) for k in DENSE), dtype=np.int64)
    syn = allocate_digit_ids(DENSE, {k: len(k) for k in DENSE}, {}, real, np.random.default_rng(0), notes,
                             pad_limit=pad_limit)
    return syn, notes


def test_unpadded_integers_spill_to_another_width_with_a_note() -> None:
    syn, notes = _alloc(None)
    assert len(set(syn.tolist())) == len(DENSE)
    assert not set(syn.tolist()) & {int(k) for k in DENSE}
    assert notes and "different number of digits" in notes[0]


def test_padded_ids_never_take_more_digits_than_their_padding() -> None:
    # 1-digit keys also written as '01'..'09' somewhere: they may take 2 digits, never 4
    with pytest.raises(IdSpaceError, match="zero-padded IDs keep their width"):
        _alloc({k: 2 for k in DENSE if len(k) == 1})


def test_padding_wide_enough_still_allows_the_move() -> None:
    syn, _ = _alloc({k: 4 for k in DENSE})
    widths = {len(str(int(s))) for s in syn.tolist()}
    assert max(widths) <= 4
    assert not set(syn.tolist()) & {int(k) for k in DENSE}


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def test_cli_padded_dense_ids_fail_clearly_and_plain_ones_spill(tmp_path: Path) -> None:
    grades = ["A", "B", "C"]
    plain = tmp_path / "plain"
    plain.mkdir()
    _write(plain / "scores.csv", [["student_id", "grade"]] + [[k, grades[i % 3]] for i, k in enumerate(DENSE)])
    res = run_cli(plain, "--out-dir", tmp_path / "o1", "--report-dir", tmp_path / "r1", "--seed", "1")
    assert res.exit_code == 0, res.output
    out = [r["student_id"] for r in csv.DictReader((tmp_path / "o1" / "scores.csv").open(encoding="utf-8"))]
    assert len(out) == len(DENSE) and not set(out) & set(DENSE)

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write(mixed / "scores.csv", [["student_id", "grade"]] + [[k, grades[i % 3]] for i, k in enumerate(DENSE)])
    # the same students in a second export, the 1-digit IDs zero-padded to 2 characters
    _write(mixed / "roster.csv", [["student_id", "grade"]]
           + [[k.zfill(2) if len(k) == 1 else k, grades[i % 3]] for i, k in enumerate(DENSE)])
    res = run_cli(mixed, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "1")
    assert res.exit_code == 3, res.output
    assert "zero-padded IDs keep their width" in res.output
    assert not (tmp_path / "o2" / "roster.csv").exists() and not (tmp_path / "o2" / "scores.csv").exists()
