"""A planted linkage for the whole-row match gate (issue 11): copy real rows verbatim into the synthetic table.

Sharing mode writes whole-row matches that are no more common than shuffling its independently generated columns
gives (``row_matches_coincidental``). A generator that leaked real rows would match more often than that; this
helper plants such a leak, so tests can check that the gate still leaves the file out.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from datanonymous.privacy import matches

from release_corpus import student_id

SURVEY_ROWS = 1200


def survey_rows(n: int = SURVEY_ROWS) -> list[dict[str, str]]:
    """An invented table of common categories with no dates: about 44% of its independently generated rows equal a
    real row by chance (a high baseline), and with 1,200 rows a 10% verbatim copy stands well clear of it."""
    rows = []
    for i in range(n):
        grade, school = i % 4, (i // 4) % 3
        track = (grade + school + (i // 12) % 2) % 4  # tied to grade and school, so some combinations never occur
        lunch = (i // 24) % 2 if track < 2 else 0
        rows.append({"Student_ID": student_id(i), "Grade": str(6 + grade), "School": ("North", "South", "West")[school],
                     "Track": ("Arts", "STEM", "Sport", "Core")[track], "Lunch": ("Free", "Paid")[lunch]})
    return rows


def write_survey(path: Path, rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    rows = rows if rows is not None else survey_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plant_row_copies(monkeypatch, fraction: float, seed: int = 0) -> None:
    """Before the match gate runs, overwrite ``fraction`` of each table's synthetic rows, on the compared columns, with
    real rows picked at random (row-mode tables only)."""
    raw: dict[bytes, list[np.ndarray]] = {}
    real_keys, enforce = matches.real_keys, matches.enforce

    def keep(tp, raw_columns):
        real_keys(tp, raw_columns)
        if tp.private.real_row_keys is not None and len(tp.private.real_row_keys):
            raw[tp.private.real_row_keys.tobytes()] = [np.asarray(c, dtype=object) for c in raw_columns]

    def planted(tp, synth, rng, *args, **kwargs):
        mode, cols = matches.plan(tp)
        source = raw.get(tp.private.real_row_keys.tobytes()) if tp.private.real_row_keys is not None else None
        if mode == "row" and cols and source is not None:
            pick = np.random.default_rng(seed)
            n = synth.n_rows
            rows = pick.choice(n, size=round(fraction * n), replace=False)
            donors = pick.integers(len(source[cols[0]]), size=len(rows))
            by_index = {c.index: c for c in synth.columns}
            for i in cols:
                values = by_index[i].values.astype(object)
                values[rows] = source[i][donors]
                by_index[i].values = values
        return enforce(tp, synth, rng, *args, **kwargs)

    monkeypatch.setattr(matches, "real_keys", keep)
    monkeypatch.setattr(matches, "enforce", planted)
