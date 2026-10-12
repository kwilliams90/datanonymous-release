"""Chance full-row (or per-entity) match removal by swap-redraw (FOUNDATION; design D23, 5.2.4).

* Comparison columns: every generated column whose values can equal real values
  (never-copy types such as IDs, invented labels, lorem and PII are left out:
  they never match). A table with multi-row IDs and within-ID-constant columns
  is checked per entity on the constant columns ("a long-format row always
  matches on date and code"); every other table row by row.
* Real keys: 64-bit hashes of the real comparison rows/entity vectors, computed
  in pass 1 (:func:`real_keys`); generators never see real rows.
* The chance-match rate is estimated by the observed rate before any swap. If it
  is above 20% the tool reports and warns instead of swapping.
* Otherwise up to 10 rounds: each matching row (entity) swaps one comparison
  cell with a random partner whose cell has the same missing state (and, for
  entities, the same number of rows), so every column's marginal counts and
  every missingness pattern stay exactly the same. Columns grouped by
  ``structure.swap_groups`` (AREA D, e.g. a functional-dependency pair) move together.
* ``--max-row-matches PCT``: remaining matches above PCT percent fail the file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import PrivacyGateError
from ..model import NEVER_COPY_TYPES, ColumnProfile, ColumnType, Message, SyntheticTable, TableProfile
from ..profile.stats import missing_mask

MAX_ROUNDS = 10
REPORT_ONLY_RATE = 0.20


def comparable(p: ColumnProfile) -> bool:
    if p.role not in ("row", "entity"):
        return False
    if p.type in NEVER_COPY_TYPES or p.type is ColumnType.MULTI_SELECT:
        return False
    if p.type is ColumnType.DIGIT_CODE and not p.params.get("levels_mode"):
        return False
    return True


def plan(tp: TableProfile) -> tuple[str, list[int]]:
    """('entity' | 'row', comparison column indices)."""
    ent_cols = [i for i in tp.entity.constant_columns if comparable(tp.columns[i])]
    if tp.entity.multi_row_ids > 0 and ent_cols:
        return "entity", ent_cols
    return "row", [p.index for p in tp.columns if comparable(p)]


def row_hashes(columns: list[np.ndarray]) -> np.ndarray:
    if not columns:
        return np.zeros(0, dtype=np.uint64)
    from ..profile.codes import codes_of

    encoded = {}
    for i, values in enumerate(columns):
        codes, uniques = codes_of(np.asarray(values, dtype=object))
        if all(isinstance(u, str) or u is None for u in uniques):
            # pandas hashes object columns by factorizing into a Categorical.
            # Reuse the raw column's codes, with pandas' -1 null sentinel, then
            # let its existing hash combiner preserve the exact row hashes.
            valid = np.fromiter((u is not None for u in uniques), dtype=bool, count=len(uniques))
            remap = np.cumsum(valid, dtype=np.int64) - 1
            remap[~valid] = -1
            encoded[str(i)] = pd.Series(pd.Categorical.from_codes(remap[codes], categories=pd.Index(uniques[valid])))
        else:
            encoded[str(i)] = pd.Series(values, dtype=object)
    df = pd.DataFrame(encoded)
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)


def real_keys(tp: TableProfile, raw_columns: list[np.ndarray]) -> None:
    """Pass 1: store the hashed real comparison keys in ``tp.private``."""
    mode, cols = plan(tp)
    tp.private.compare_columns = cols
    if not cols:
        return
    if mode == "entity":
        vecs = [tp.private.extra["entity_values"][i] for i in cols]
        tp.private.real_entity_keys = np.unique(row_hashes(vecs))
    else:
        tp.private.real_row_keys = np.unique(row_hashes([raw_columns[i] for i in cols]))


def enforce(tp: TableProfile, synth: SyntheticTable, rng: np.random.Generator,
            max_row_matches: float | None) -> dict[str, Any]:
    """Swap-redraw chance matches in ``synth`` (in place); returns the privacy summary."""
    mode, cols = plan(tp)
    names = [tp.columns[i].name for i in cols]
    summary: dict[str, Any] = {"mode": mode, "columns": names, "checked": bool(cols)}
    if not cols:
        summary["note"] = "no column can reproduce real values; no check needed"
        return summary
    by_index = {c.index: c for c in synth.columns}
    role = "entity" if mode == "entity" else "row"
    carried, group_of = _groups(tp, cols, role)
    unit_cols = cols + carried
    profiles = [tp.columns[i] for i in unit_cols]
    if mode == "entity":
        index = tp.private.entity_index
        n_units = tp.entity.n_entities
        first = np.full(n_units, -1, dtype=np.int64)
        for r in range(len(index) - 1, -1, -1):
            first[index[r]] = r
        units = [by_index[i].values[first] for i in unit_cols]
        sizes = np.bincount(index, minlength=n_units)
        real = tp.private.real_entity_keys
    else:
        units = [by_index[i].values.copy() for i in unit_cols]
        sizes = None
        real = tp.private.real_row_keys
    n = len(units[0])
    missing = [missing_mask(u, p.missing_policy, p.na_tokens) for u, p in zip(units, profiles)]
    keys = units[: len(cols)]

    def matches() -> np.ndarray:
        return np.isin(row_hashes(keys), real)

    m = matches()
    initial = int(m.sum())
    rate = initial / n if n else 0.0
    summary.update(initial_matches=initial, initial_rate=round(rate, 6), units=n)
    rounds = swaps = 0
    if rate > REPORT_ONLY_RATE:
        summary["report_only"] = True
        summary["remaining_matches"] = initial
        summary["remaining_rate"] = round(rate, 6)
        synth.messages.append(Message(
            "warning", "row_matches_report_only",
            f"{initial} of {n} synthetic {'entities' if mode == 'entity' else 'rows'} ({rate:.1%}) equal a real one "
            f"on all compared columns by chance; above 20% the tool reports instead of swapping (D23)"))
    else:
        summary["report_only"] = False
        while m.any() and rounds < MAX_ROUNDS:
            rounds += 1
            for r in rng.permutation(np.flatnonzero(m)).tolist():
                if _swap_one(r, units, missing, group_of, sizes, rng):
                    swaps += 1
            m = matches()
        summary["remaining_matches"] = int(m.sum())
        summary["remaining_rate"] = round(float(m.mean()) if n else 0.0, 6)
    summary.update(rounds=rounds, swaps=swaps)
    # write back
    if mode == "entity":
        for i, u in zip(unit_cols, units):
            by_index[i].values = u[tp.private.entity_index]
    else:
        for i, u in zip(unit_cols, units):
            by_index[i].values = u
    remaining = summary["remaining_matches"]
    if remaining and not summary["report_only"]:
        synth.messages.append(Message("warning", "row_matches_remaining",
                                      f"{remaining} synthetic {'entities' if mode == 'entity' else 'rows'} still equal "
                                      f"a real one after {rounds} swap rounds"))
    if max_row_matches is not None and summary["remaining_rate"] * 100 > max_row_matches:
        raise PrivacyGateError(
            f"{remaining} chance matches ({summary['remaining_rate']:.2%}) exceed --max-row-matches {max_row_matches}%",
            fix="raise --max-row-matches, or add columns that vary (fewer exact matches are then possible)",
        )
    return summary


def _groups(tp: TableProfile, cols: list[int], role: str) -> tuple[list[int], list[list[int]]]:
    """Extra columns carried along by swap groups, and per key column the unit positions to swap."""
    pos = {c: i for i, c in enumerate(cols)}
    carried: list[int] = []
    group_of = [[i] for i in range(len(cols))]
    for group in tp.structure.swap_groups:
        members = [c for c in group if 0 <= c < len(tp.columns) and tp.columns[c].role == role]
        if not any(c in pos for c in members):
            continue
        positions = []
        for c in members:
            if c not in pos:
                pos[c] = len(cols) + len(carried)
                carried.append(c)
            positions.append(pos[c])
        for c in members:
            if pos[c] < len(cols):
                group_of[pos[c]] = sorted(set(group_of[pos[c]]) | set(positions))
    return carried, group_of


def _swap_one(r: int, units: list[np.ndarray], missing: list[np.ndarray], group_of: list[list[int]],
              sizes: np.ndarray | None, rng: np.random.Generator, tries: int = 25) -> bool:
    k = len(group_of)
    n = len(units[0])
    for _ in range(tries):
        c = int(rng.integers(k))
        s = int(rng.integers(n))
        group = group_of[c]
        if s == r or units[c][s] == units[c][r]:
            continue
        if any(missing[g][s] != missing[g][r] for g in group):
            continue
        if sizes is not None and sizes[s] != sizes[r]:
            continue
        for g in group:
            units[g][r], units[g][s] = units[g][s], units[g][r]
        return True
    return False
