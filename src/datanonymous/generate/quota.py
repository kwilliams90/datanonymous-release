"""Exact largest-remainder quota allocation with floors (FOUNDATION).

Exact quota (design 5.3) reproduces discrete shares exactly: at the same N every
level keeps its exact count. Floors (D14): every level present in the source
gets >= 1 row; when the floor changes a count the column is flagged
``rare_category_adjusted``. A floor is moved with minimal change: each zero level
takes one row from the level whose allocation most exceeds its exact share.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..errors import GenerationError


def largest_remainder(weights: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    """Integer counts summing to ``n`` proportional to ``weights`` (ties: lower index first)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        if n:
            raise GenerationError("cannot allocate rows to zero levels")
        return np.zeros(0, dtype=np.int64)
    total = w.sum()
    if total <= 0:
        raise GenerationError("quota weights sum to zero")
    raw = w / total * n
    base = np.floor(raw).astype(np.int64)
    rem = int(n - base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:rem]] += 1
    return base


def allocate(counts: Sequence[float] | np.ndarray, n: int, floor: bool = True) -> tuple[np.ndarray, bool]:
    """Exact quota of ``n`` rows over levels with source ``counts``; (allocation, adjusted)."""
    w = np.asarray(counts, dtype=float)
    alloc = largest_remainder(w, n)
    if not floor or n == 0:
        return alloc, False
    present = w > 0
    zero = np.flatnonzero(present & (alloc == 0))
    if zero.size == 0:
        return alloc, False
    if int(present.sum()) > n:
        raise GenerationError(
            f"{int(present.sum())} levels cannot each get a row in {n} rows (design Q11)",
            fix="this cannot happen at N rows in = N rows out; please report it",
        )
    exact = w / w.sum() * n
    for z in zero:
        surplus = np.where(alloc > 1, alloc - exact, -np.inf)
        donor = int(np.argmax(surplus))
        alloc[donor] -= 1
        alloc[z] += 1
    return alloc, True


def quota_sample(levels: Sequence[Any], counts: Sequence[float], n: int, rng: np.random.Generator,
                 floor: bool = True) -> tuple[np.ndarray, bool]:
    """``n`` values drawn by exact quota over ``levels`` and shuffled; (values, adjusted)."""
    alloc, adjusted = allocate(counts, n, floor)
    lv = np.empty(len(levels), dtype=object)
    lv[:] = list(levels)
    out = np.repeat(lv, alloc)
    return rng.permutation(out), adjusted
