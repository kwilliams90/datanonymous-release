"""Structural relationships: missingness, FDs, templates, keys, rules, panels, layout order (AREA D owns).

FOUNDATION implements the whole-row missingness model (D6), used by the pipeline:

* :func:`profile_missingness` - blank/non-blank pattern of every row over the
  row-level columns (missing = blank, or an NA token in a typed column);
  patterns held by >= ``min_cell`` rows are kept, rarer ones are pooled.
* :func:`sample_missing_mask` - exact largest-remainder quota over the kept
  patterns plus one pool; inside the pool each column gets exactly its pooled
  missing count at random rows, so per-column missing counts stay exact while
  rare (identifying) combinations are not reproduced.
* :func:`profile_entity_missingness` / :func:`sample_entity_mask` - the same
  for the within-ID-constant columns (D4), whose values are drawn once per
  entity: their joint blank/non-blank pattern is modelled and sampled over
  entities (exact quota, pooled below ``min_cell`` entities), so ID constants
  that are blank together stay blank together; the row-level patterns are then
  dealt to the rows of each entity pattern in their real proportions
  (``MissingnessModel.cross``), keeping every pattern's exact total.
* :func:`balance_entity_rows` / :func:`balance_entity_values` (R2-10): the
  entity patterns, and the levels and NA tokens of each D4 column, are dealt to
  the ID blocks so their row counts are the source's too, not only their
  entity counts (``MissingnessModel.entity_rows``; patterns and real values
  held by fewer than ``min_cell`` entities are not matched to block sizes, a
  D35 pseudonym is, since it writes no real value: R7-06).

AREA D (design D5, D16, sections 5.1 and 6). Detected on the real table in pass
1; public descriptions hold column names and counts only (D21), the real-derived
data generation needs (level mappings, rule gaps, domain counts) stays in
``StructureProfile.private``. Kept automatically when they hold in 100% of the
rows (``structure: {disable: [...]}`` in ``--config`` turns each off):

* **functional dependencies** (:func:`find_functional_dependencies`):
  ``school_id -> school_name``. Determinants are columns whose real values are
  written (factors, org codes, ordinals, school years, booleans, discrete
  integers/years, digit codes kept as levels) with at least 2 rows per value
  on average (org code -> org code pairs need no support: D10 keeps both).
  The dependent is regenerated from the synthetic determinant through the
  observed mapping; a dependent whose values are never copied (names, invented
  labels) is mapped consistently onto its own synthetic values. The determinant's
  synthetic values are first dealt to the rows whose sampled whole-row pattern
  wants the dependents' mapped blank/non-blank state (R2-09,
  ``generate.structure.align_fd_missingness``).
* **derived templates** (:func:`find_derived_templates`, ``profile.templates``):
  ``email = lower(first).lower(last).id@domain`` rebuilt from the synthetic
  inputs, with reserved domains (example.org / .com / .net). Missing-like
  placeholders in the target (``N/A``) are not data: the template must hold in
  every other row, and placeholder rows are left as they are. The target's value
  groups (units per value, input frequency ranks; no values) are kept privately so
  the rebuilt column keeps its distinct count and frequency profile (R5-04).
* **composite keys** (:func:`find_composite_keys`): (person ID, column) unique
  in the real table (no duplicate (student, date)) stays unique: the synthetic
  rows (their row-level columns, moving whole so missingness patterns, FDs and
  rules stay intact) are assigned to the ID blocks so no block repeats a key. When
  no single column is a key with the ID, (ID, A, B) pairs are tried too
  (attendance by (date, period), grades by (course, term)). Candidates are the
  columns whose synthetic values keep every real value's count (labels, booleans,
  years, codes, discrete integers such as visit/period/term numbers: R4-04),
  masked codes, invented labels and dates; never continuous numbers, nor integers
  with more values than factor-max and the most rows of one ID (scores: a
  coincidence, not a key). Generation rebuilds what moving rows cannot fix and
  reports any remaining repeat in the key's status (R5-03).
* **cross-file key coverage** (:func:`cross_file_structure`): ID columns
  linked across files share one synthetic mapping, so coverage and orphan
  rates are reproduced exactly (parents first is implied); the report gives
  key containment and orphan rates; ``0012345`` / ``12345`` and
  ``student_number`` / ``StudentNumber`` match (D8). ID columns are linked by
  their normalised name, and - whatever their headers (``student_id`` /
  ``StudentNumber``) - when >= 99% of one column's keys are found in the ID
  column(s) of another table or sheet (key discovery, :func:`_key_links`;
  never with kept row numbers 1..n). A non-ID column holding an ID column's
  real IDs (``subject_ref``, a self-reference ``mentor``) is profiled again as
  an ID column (source ``key-ref``, :func:`_key_reference_columns`; R2-01) and
  linked on the same evidence, so its real values are never written.
* **layout** (section 6): up to 3 leading sort keys (asc/desc, missing first or
  last), or leading blocks of missing values, and the duplicate-row count
  (switches ``sort_order`` and ``duplicate_rows``).

Only suggested (D5): inequality rules ``A <= B`` between numeric (plain or
formatted, ``$1,234.50``: R5-05) or same-kind date columns holding in 100% of
rows (:func:`suggest_inequality_rules`); a rule applies only when declared under
``rules:`` in ``--config`` (``"KCSE <= Yr_JoinCampus"`` or
``{rule: "...", file: "path.csv"}``; operators ``<=``, ``<``, ``>=``, ``>``),
and is built by construction, never by rejection:
every rule on one high column B is solved together (R2-11). B keeps its own
independently generated values (its source range, exact min/max, levels, date
layouts); the rules decide where they go, steered by the observed gaps between
B and its largest low value (once per entity for within-ID constants; R3-06,
``generate.structure``). Dates (date, datetime with or without offsets, time,
duration, year-month; :func:`rule_view`) compare in their model unit (R2-16).
A declared rule that cannot be built FAILs its file (D19) with the fix
instead of being ignored; every applied rule is re-checked on the final output
(``generate.structure.check_rules``). A functional dependency whose dependent is
a declared rule's high column is not applied (the declared rule wins); FDs and
rules otherwise run in column-dependency order.

Warned about (D16, :func:`detect_panels`): (ID, date) panels, multi-year
snapshots with attributes that change across years, aggregate-shaped tables.

Generation: :func:`apply_structure` (before privacy swap-redraw: declared
rules, FDs, templates, composite keys) and :func:`finalize_rows` (after it:
composite-key repair, duplicate rows, row order), implemented in
``generate.structure``. :func:`swap_groups` makes privacy swaps move linked
columns together. :func:`describe_structure` gives the report/inspect lines.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import Settings
from ..detect import dates as dates_hooks
from ..detect.dates_formats import US_DAY
from ..detect.ids import WEAK_ID_KEY, normalize_id
from ..errors import UsageError
from ..generate.quota import largest_remainder
from ..hints import header_hints, normalize
from ..model import (
    LEVEL_TYPES,
    NEVER_COPY_TYPES,
    TYPE_INFO,
    ColumnProfile,
    ColumnType,
    EntityProfile,
    Message,
    MissingnessModel,
    RawTable,
    StructureProfile,
)
from . import entity as entity_mod
from . import templates as tpl
from .codes import codes_of, has_nul, subset_codes
from .entity import EXPLICIT_SOURCES, GENERIC_ID_NAMES, base_namespace, entity_keys, is_functional
from .stats import is_missing, missing_mask, numeric_view_aligned, redact_candidate

T = ColumnType


def _pattern_model(masks: dict[int, np.ndarray], n_rows: int,
                   min_cell: int) -> tuple[MissingnessModel, np.ndarray | None]:
    """D6 pattern model of per-column missing masks, and each row's pattern code (the index of its kept
    pattern, or ``len(patterns)`` for the pool); None without columns or rows."""
    cols = sorted(masks)
    model = MissingnessModel(columns=cols)
    if not cols or n_rows == 0:
        return model, None
    m = np.column_stack([masks[c] for c in cols]).astype(bool)
    pats, inverse, counts = np.unique(m, axis=0, return_inverse=True, return_counts=True)
    code_of = np.empty(len(pats), dtype=np.int64)
    for u, (p, c) in enumerate(zip(pats, counts)):
        key = tuple(bool(x) for x in p)
        if c >= min_cell:
            code_of[u] = len(model.patterns)
            model.patterns.append((key, int(c)))
        else:
            code_of[u] = -1
            model.pooled.append((key, int(c)))
    code_of[code_of < 0] = len(model.patterns)
    return model, code_of[np.asarray(inverse).reshape(-1)]


def profile_missingness(masks: dict[int, np.ndarray], n_rows: int, min_cell: int) -> MissingnessModel:
    """Build the D6 pattern model from per-column boolean missing masks."""
    return _pattern_model(masks, n_rows, min_cell)[0]


def profile_entity_missingness(model: MissingnessModel, row_codes: np.ndarray | None,
                               entity_masks: dict[int, np.ndarray], entity_index: np.ndarray | None,
                               n_entities: int, min_cell: int) -> None:
    """D6 for within-ID-constant columns (D4), added to the row-level ``model`` in place.

    Their values are drawn once per entity, so their blank/non-blank patterns are
    modelled jointly over entities (``entity_masks``: per column, one flag per
    entity; patterns held by fewer than ``min_cell`` entities are pooled), and
    ``model.cross`` keeps how the row-level patterns are distributed inside each
    entity pattern (real row counts; combinations held by fewer than
    ``min_cell`` rows are dropped, as pooled whole-row patterns are). Both are
    sampled by :func:`sample_entity_mask` and :func:`sample_missing_mask`.
    ``model.entity_rows`` keeps the real rows of each entity-pattern group, so the
    patterns are dealt to ID blocks with these row totals (R2-10).
    """
    emodel, ecodes = _pattern_model(entity_masks, n_entities, min_cell)
    model.entity_columns, model.entity_patterns, model.entity_pooled = emodel.columns, emodel.patterns, emodel.pooled
    model.cross = None
    model.entity_rows = None
    if ecodes is not None and entity_index is not None and len(entity_index):
        # real rows per entity-pattern group: patterns are dealt to ID blocks of these sizes (R2-10)
        n_eg = len(emodel.patterns) + (1 if emodel.pooled else 0)
        model.entity_rows = np.bincount(ecodes[entity_index], minlength=n_eg).astype(np.int64).tolist()
    if ecodes is None or row_codes is None or entity_index is None:
        return
    n_groups = len(emodel.patterns) + (1 if emodel.pooled else 0)
    n_codes = len(model.patterns) + (1 if model.pooled else 0)
    if n_groups < 2 or n_codes < 2:
        return  # nothing to couple
    cross = np.zeros((n_groups, n_codes), dtype=float)
    np.add.at(cross, (ecodes[entity_index], row_codes), 1.0)
    cross[cross < min_cell] = 0.0
    model.cross = cross


def _sample_codes(patterns: list[tuple[tuple[bool, ...], int]], pooled: list[tuple[tuple[bool, ...], int]],
                  n: int, rng: np.random.Generator, cross: np.ndarray | None = None,
                  groups: np.ndarray | None = None) -> np.ndarray | None:
    """Pattern code of each of ``n`` rows by exact quota (``len(patterns)`` = the pool); None without weights.

    With ``cross`` and ``groups`` (each row's entity-pattern group), the codes are
    dealt to the groups by :func:`_coupled_codes` instead of shuffled over all rows.
    """
    pool_total = sum(c for _, c in pooled)
    weights = [c for _, c in patterns] + ([pool_total] if pool_total else [])
    if not weights:
        return None
    alloc = largest_remainder(np.asarray(weights, dtype=float), n)
    if (cross is not None and groups is not None and len(groups) == n and cross.shape[1] == len(alloc)
            and (not n or (int(groups.min()) >= 0 and int(groups.max()) < cross.shape[0]))):
        return _coupled_codes(cross, groups, alloc, rng)
    codes = np.repeat(np.arange(len(alloc)), alloc)
    return rng.permutation(codes)


def _coupled_codes(cross: np.ndarray, groups: np.ndarray, alloc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-pattern codes dealt to entity-pattern groups: the exact totals ``alloc`` of every code are kept, and
    each group's rows get the codes in the group's real proportions (``cross``) as closely as its synthetic
    size allows.

    Each group's rows are first split by exact quota over its real row-pattern counts (the overall
    shares when all of them were dropped as rare); then single codes move between rows of one group, from a
    code allocated too often to one allocated too rarely, in the group where that move departs least from
    its real proportions, until every code has exactly its total.
    """
    n_groups, n_codes = cross.shape
    sizes = np.bincount(groups, minlength=n_groups)[:n_groups]
    overall = np.asarray(alloc, dtype=float)
    target = np.zeros((n_groups, n_codes), dtype=float)
    take = np.zeros((n_groups, n_codes), dtype=np.int64)
    for g in range(n_groups):
        if not sizes[g]:
            continue
        w = cross[g] if cross[g].sum() > 0 else overall
        target[g] = w / w.sum() * sizes[g]
        take[g] = largest_remainder(w, int(sizes[g]))
    diff = take.sum(axis=0) - np.asarray(alloc, dtype=np.int64)
    while (diff > 0).any():
        a, b = int(np.argmax(diff)), int(np.argmin(diff))
        cand = np.flatnonzero(take[:, a] > 0)
        gain = (target[cand, b] - take[cand, b]) - (target[cand, a] - take[cand, a])
        g = int(cand[int(np.argmax(gain))])
        room = int(np.floor(target[g, b] - take[g, b]))
        k = max(1, min(int(diff[a]), int(-diff[b]), int(take[g, a]), room))
        take[g, a] -= k
        take[g, b] += k
        diff[a] -= k
        diff[b] += k
    codes = np.empty(len(groups), dtype=np.int64)
    for g in range(n_groups):
        rows = np.flatnonzero(groups == g)
        if rows.size:
            codes[rows] = rng.permutation(np.repeat(np.arange(n_codes), take[g]))
    return codes


def _fill_mask(patterns: list[tuple[tuple[bool, ...], int]], pooled: list[tuple[tuple[bool, ...], int]],
               codes: np.ndarray, out: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Write each row's pattern into ``out``; rows of the pool get each column's pooled missing count."""
    k = out.shape[1]
    for j, (pat, _) in enumerate(patterns):
        rows = codes == j
        out[rows] = np.asarray(pat, dtype=bool)
    pool_total = sum(c for _, c in pooled)
    if pool_total:
        pool_rows = np.flatnonzero(codes == len(patterns))
        size = len(pool_rows)
        per_col = np.zeros(k, dtype=float)
        for pat, c in pooled:
            per_col += np.asarray(pat, dtype=float) * c
        for col in range(k):
            want = per_col[col] / pool_total * size
            cnt = int(round(want)) if size != pool_total else int(per_col[col])
            cnt = min(max(cnt, 0), size)
            if per_col[col] > 0 and cnt == 0 and size > 0:
                cnt = 1  # floor: >= 1 missing when the source has any (section 10)
            if cnt:
                out[rng.choice(pool_rows, size=cnt, replace=False), col] = True
    return out


def sample_missing_mask(model: MissingnessModel, n: int, rng: np.random.Generator,
                        groups: np.ndarray | None = None) -> np.ndarray:
    """(n x len(columns)) boolean missing mask following ``model`` (see module doc).

    ``groups`` (each row's entity-pattern group from :func:`sample_entity_mask`)
    couples the row-level patterns to the within-ID constants' patterns through
    ``model.cross``; without it (or without ``cross``) the patterns are shuffled
    over all rows.
    """
    k = len(model.columns)
    out = np.zeros((n, k), dtype=bool)
    if k == 0 or n == 0:
        return out
    codes = _sample_codes(model.patterns, model.pooled, n, rng, model.cross if groups is not None else None, groups)
    if codes is None:
        return out
    return _fill_mask(model.patterns, model.pooled, codes, out, rng)


def sample_entity_mask(model: MissingnessModel, n_entities: int, rng: np.random.Generator,
                       sizes: np.ndarray | None = None, min_entities: int = 1,
                       report: list[dict[str, Any]] | None = None) -> tuple[np.ndarray | None, np.ndarray]:
    """(each entity's pattern group or None, (n_entities x len(entity_columns)) missing mask) for the
    within-ID-constant columns: exact quota over the entity patterns, the pool per column (D4, D6).

    With ``sizes`` (rows of each synthetic entity: its real ID's block, D8) the patterns are dealt so each
    pattern's rows come out as in the source (``model.entity_rows``, :func:`balance_entity_rows`; R2-10):
    a pattern held by short ID blocks in the source is not expanded onto long ones. Pattern rows that stay off
    are added to ``report`` (R9-04).
    """
    k = len(model.entity_columns)
    out = np.zeros((n_entities, k), dtype=bool)
    if k == 0 or n_entities == 0:
        return None, out
    codes = _sample_codes(model.entity_patterns, model.entity_pooled, n_entities, rng)
    if codes is None:
        return None, out
    if sizes is not None and model.entity_rows is not None and len(sizes) == n_entities:
        codes = balance_entity_rows(codes, sizes, model.entity_rows, rng, min_entities, report=report)
    return codes, _fill_mask(model.entity_patterns, model.entity_pooled, codes, out, rng)


def balance_entity_rows(labels: np.ndarray, sizes: np.ndarray, target: Any, rng: np.random.Generator,
                        min_entities: int = 1, real: Any = None, witness: Any = None,
                        report: list[dict[str, Any]] | None = None) -> np.ndarray:
    """Entity labels (pattern groups, levels, NA tokens) re-dealt to entities so each label's rows (the sum of
    its entities' block sizes) come as close as possible to the source's ``target`` rows (R2-10).

    D4 values and D6 entity patterns are drawn by exact quota over entities; expanding them onto ID blocks of
    unequal sizes would change their row counts (ten 2-row IDs blank, ten 5-row IDs filled: 20 blank rows,
    not 35). Labels are only swapped between two entities, so every label keeps its exact entity count.
    Starting from the random deal, an entity of the label with the most excess rows swaps with an entity of
    the label with the fewest (or the next pair that can), picking the size difference closest to half their
    gap (entities of those sizes at random); when no pair can, the rows move through a third label (two swaps
    that leave it unchanged). This stops when nothing lowers the sum of squared row errors.

    Every label is dealt this way except one that writes a REAL value held by fewer than ``min_entities``
    entities (``--min-cell``): it is never moved and its rows are not targeted, since matching them to block
    sizes would show which few real IDs hold that real value (as rare patterns are pooled, D6); the other
    labels share the remaining rows in proportion to their source rows. ``real`` (one flag per label; default:
    every label) says which labels write a real value. A D35 pseudonym writes none, so it is dealt like any
    other label and keeps its source level's rows however few entities hold it (R7-06; D35: a pseudonym keeps
    its level's exact count): such few-entity labels aim at their exact source rows, and only the labels held
    by many entities share what the frozen labels' blocks add or take.

    Such few-entity labels (often thousands of one-ID pseudonyms) are first made exact one at a time, before
    the descent (:func:`_exact_few`): each swaps an entity for one of exactly the size it needs, with a label
    held by many entities (which absorbs the difference) or with another few-entity label that is off too, or
    steps closer through a many-entity label; the exact ones then stay as they are and the descent balances the
    rest. This is linear in the entities, where the descent alone would be quadratic in the labels.

    R9-04: when the swaps stall with rows still off (often in a small table with nearly unique block sizes), an
    exact deal is searched (:func:`_exact_deal`): a complete, randomised search over how many entities of each
    block size every label holds, capped at :data:`_EXACT_NODES` steps; when it gives up, ``witness`` (the
    source's own (label, block size) pairs: the real assignment proves an exact deal exists) is used when it
    fits these entities, its labels moved around by random exchanges that keep every label's rows. Either way
    the entities keep as many of their dealt labels as the deal allows. Rows that stay off (no exact deal
    exists, or none was found) are added to ``report`` (counts only: labels off, rows moved too few) so the
    caller warns instead of passing. Consumes no randomness when the deal is already exact (equal block sizes).
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    sizes = np.asarray(sizes, dtype=np.int64)
    if len(labels) < 2:
        return labels
    tw = np.asarray(target, dtype=float)
    n_codes = max(len(tw), int(labels.max()) + 1)
    weights = np.zeros(n_codes, dtype=float)
    weights[:len(tw)] = tw
    counts = np.bincount(labels, minlength=n_codes)
    is_real = np.ones(n_codes, dtype=bool)
    if real is not None:
        flags = np.asarray(real, dtype=bool).reshape(-1)[:n_codes]
        is_real[:len(flags)] = flags
    few = counts < max(int(min_entities), 1)
    movable = (~few | ~is_real) & (weights > 0)
    if int(movable.sum()) < 2:
        return labels
    ents = np.flatnonzero(movable[labels])
    codes = np.flatnonzero(movable)
    goal = np.zeros(n_codes, dtype=np.int64)
    total = int(sizes[ents].sum())
    pseudo, rest = few & movable, ~few & movable
    fixed = int(np.rint(weights[pseudo]).sum())
    if pseudo.any() and rest.any() and fixed <= total:
        # the few-entity labels (D35 pseudonyms) keep their exact rows; the rows a frozen rare real label holds or
        # lacks are shared by the many-entity labels only (R7-06)
        goal[pseudo] = np.rint(weights[pseudo]).astype(np.int64)
        goal[rest] = largest_remainder(weights[rest], total - fixed)
    else:
        goal[codes] = largest_remainder(weights[codes], total)
    rows = np.bincount(labels[ents], weights=sizes[ents], minlength=n_codes).astype(np.int64)
    diff = rows - goal
    if not diff[codes].any():
        return labels
    buckets: dict[int, dict[int, list[int]]] = {int(c): {} for c in codes}
    for e in ents.tolist():
        buckets[int(labels[e])].setdefault(int(sizes[e]), []).append(e)
    for c in buckets:
        for b in buckets[c].values():
            b.sort()

    def best_pair(a: int, b: int, gap: int) -> tuple[int, int] | None:
        """Sizes (of a's entity, of b's entity) whose difference is in [1, gap - 1], closest to gap / 2."""
        sa = np.asarray(sorted(buckets[a]), dtype=np.int64)
        sb = np.asarray(sorted(buckets[b]), dtype=np.int64)
        if not len(sa) or not len(sb):
            return None
        at = np.searchsorted(sa, sb + gap // 2)
        best: tuple[int, int, int] | None = None
        for cand in (at - 1, at):
            ok = (cand >= 0) & (cand < len(sa))
            if not ok.any():
                continue
            x, y = sa[cand[ok]], sb[ok]
            delta = x - y
            valid = (delta > 0) & (delta < gap)
            if not valid.any():
                continue
            score = np.abs(2 * delta[valid] - gap)
            j = int(np.argmin(score))
            pick = (int(score[j]), int(x[valid][j]), int(y[valid][j]))
            if best is None or pick < best:
                best = pick
        return None if best is None else (best[1], best[2])

    def take(c: int, s: int) -> int:
        b = buckets[c][s]
        j = int(rng.integers(len(b)))
        e = b[j]
        b[j] = b[-1]
        b.pop()
        if not b:
            del buckets[c][s]
        return e

    def chain(a: int, b: int, gap: int) -> tuple[int, int, int, int, int] | None:
        """(c, s_x, s_z, s_z2, delta): move ``delta`` rows from a to b through a third label c - a's entity of
        size s_x swaps with c's of size s_z = s_x - delta, then c's of size s_z2 with b's of size s_z2 - delta
        (c keeps its rows) - when no direct swap exists (a's and b's sizes do not differ by delta)."""
        sa = np.asarray(sorted(buckets[a]), dtype=np.int64)
        sb = np.asarray(sorted(buckets[b]), dtype=np.int64)
        best: tuple[int, int, int, int, int, int] | None = None
        for c in sorted(int(x) for x in codes):
            if c in (a, b) or not buckets[c]:
                continue
            sc = np.asarray(sorted(buckets[c]), dtype=np.int64)
            d1 = sa[:, None] - sc[None, :]
            d2 = sc[:, None] - sb[None, :]
            common = np.intersect1d(d1[(d1 > 0) & (d1 < gap)], d2[(d2 > 0) & (d2 < gap)])
            found = None
            for delta in sorted(common.tolist(), key=lambda d: (abs(2 * d - gap), d)):
                hops = [(int(sa[i]), int(sc[j]), int(sc[k])) for i, j in zip(*np.nonzero(d1 == delta))
                        for k, _ in zip(*np.nonzero(d2 == delta))
                        # c's only entity of a size cannot leave twice
                        if not (k == j and len(buckets[c][int(sc[j])]) < 2)]
                if hops:
                    s_x, s_z, s_z2 = hops[0]
                    found = (abs(2 * delta - gap), c, s_x, s_z, s_z2, delta)
                    break
            if found is not None and (best is None or found < best):
                best = found
        return None if best is None else best[1:]

    def swap_pair(a: int, b: int, s_a: int, s_b: int) -> None:
        x, y = take(a, s_a), take(b, s_b)
        labels[x], labels[y] = b, a
        buckets[b].setdefault(s_a, []).append(x)
        buckets[a].setdefault(s_b, []).append(y)
        diff[a] -= s_a - s_b
        diff[b] += s_a - s_b

    all_codes = codes
    if few[codes].any():  # few-entity labels (D35 pseudonyms) first, exactly (R7-06)
        codes = _exact_few(codes, few, counts, diff, buckets, swap_pair)
    # each swap strictly lowers the sum of squared row errors (a move between two labels with errors da > db by
    # a size difference in (0, da - db)), so the loop ends; squares (not absolute errors) let an exact label pass
    # rows on (a long block from a surplus label to an exact one, and one of its own to a deficit label)
    while len(codes) >= 2 and diff[codes].any():
        order = sorted((int(c) for c in codes), key=lambda c: (-diff[c], c))
        swap = None
        for a in order:
            for b in reversed(order):
                gap = int(diff[a] - diff[b])
                if gap < 2:
                    break
                pair = best_pair(a, b, gap)
                if pair is not None:
                    swap = (a, b, *pair)
                    break
            if swap is not None or int(diff[a] - diff[order[-1]]) < 2:
                break
        if swap is not None:
            swap_pair(*swap)
            continue
        a, b = order[0], order[-1]
        hop = chain(a, b, int(diff[a] - diff[b])) if diff[a] - diff[b] >= 2 else None
        if hop is None:
            break
        c, s_x, s_z, s_z2, delta = hop
        swap_pair(a, c, s_x, s_z)
        swap_pair(c, b, s_z2, s_z2 - delta)
    if diff[all_codes].any():
        # R9-04: the swaps stalled with rows still off; an exact deal is searched over every movable label (and,
        # when the search gives up, built from the source's own (label, block size) counts, a deal that exists)
        if _exact_deal(labels, sizes, ents, all_codes, goal, rng, witness):
            diff[all_codes] = 0
    if report is not None and diff[all_codes].any():
        off = diff[all_codes]
        report.append({"labels": int(np.count_nonzero(off)), "rows": int(np.abs(off).sum()) // 2,
                       "codes": [int(c) for c in all_codes[off != 0]]})
    return labels


#: R9-04: the most steps of the exact search (:func:`_exact_deal`): a fixed count, so the deal never depends on time.
_EXACT_NODES = 60_000
#: R9-04: the exact search runs over at most this many distinct block sizes (beyond: the witness only).
_EXACT_MAX_SIZES = 1_000
#: R9-04: the most random exchanges that move a witness deal's labels around (:func:`_mix_counts`).
_MIX_ATTEMPTS = 20_000


def _exact_deal(labels: np.ndarray, sizes: np.ndarray, ents: np.ndarray, codes: np.ndarray, goal: np.ndarray,
                rng: np.random.Generator, witness: Any = None) -> bool:
    """R9-04: re-deal the labels ``codes`` among the entities ``ents`` (in place in ``labels``) so every label keeps
    its entity count and holds exactly ``goal`` rows; False (nothing changed) when no exact deal was found.

    The deal is a table of counts: how many entities of each block size every label holds (rows per size,
    entities per label and rows per label fixed). :func:`_search_counts` searches it completely, randomised and
    capped (:data:`_EXACT_NODES`); when the search gives up, ``witness`` (the source's (label, block size) pairs,
    codes and sizes) is the table when it fits these entities: the real assignment is an exact deal. Its labels
    are then moved around by random exchanges that keep every label's rows (:func:`_mix_counts`), so a label is
    not tied to the real holders' block sizes where other exact deals exist. Within a block size the entities keep
    the labels they were dealt as far as the table allows; the rest are dealt at random (:func:`_assign_counts`)."""
    ents = np.asarray(ents, dtype=np.int64)
    if len(ents) < 2:
        return False
    ent_sizes = np.asarray(sizes, dtype=np.int64)[ents]
    sz, inv = np.unique(ent_sizes, return_inverse=True)
    inv = np.asarray(inv, dtype=np.int64).reshape(-1)
    avail = np.bincount(inv, minlength=len(sz)).astype(np.int64)
    cur = labels[ents]
    code_list = sorted(int(c) for c in codes)
    held = np.bincount(cur, minlength=max(code_list) + 1)
    need = [(c, int(held[c]), int(goal[c])) for c in code_list]
    if sum(n for _, n, _ in need) != len(ents) or sum(g for _, _, g in need) != int(ent_sizes.sum()):
        return False
    table = None
    if len(sz) <= _EXACT_MAX_SIZES:
        table = _search_counts(sz.tolist(), avail.tolist(), need, rng)
    if table is None and witness is not None:
        table = _witness_counts(witness, sz, avail, need)
        if table is not None:
            table = _mix_counts(table, sz, rng, min(_MIX_ATTEMPTS, 20 * len(ents)))
    if table is None:
        return False
    _assign_counts(labels, ents, inv, cur, code_list, table, rng)
    return True


def _outward(v0: int, lo: int, hi: int) -> Any:
    """``v0``, then the values of ``[lo, hi]`` outward from it (``v0 + 1``, ``v0 - 1``, ``v0 + 2`` ...)."""
    yield v0
    for d in range(1, max(v0 - lo, hi - v0) + 1):
        if v0 + d <= hi:
            yield v0 + d
        if v0 - d >= lo:
            yield v0 - d


def _label_vectors(sz: list[int], av: list[int], r: int, t: int, rng: np.random.Generator,
                   budget: list[int]) -> Any:
    """Every way (lazily, randomised) for ONE label to hold ``r`` entities of exactly ``t`` rows, as counts per block
    size (``sz`` ascending, ``av`` entities available of each): the counts of the largest sizes first, each from
    the interval its bounds allow (the rows the remaining smaller sizes can still make: :func:`_exact_deal`), a
    random start inside it, then outward. Iterative (no recursion depth); each step spends one ``budget`` unit.
    The yielded list is reused: copy it."""
    m = len(sz)
    cc = [0] * (m + 1)  # entities, and rows, of the sizes below index i
    cw = [0] * (m + 1)
    for i in range(m):
        cc[i + 1] = cc[i] + av[i]
        cw[i + 1] = cw[i] + av[i] * sz[i]

    def low(i: int, n: int) -> int:  # the fewest rows of n entities of the sizes below index i
        if n <= 0:
            return 0
        p = bisect.bisect_left(cc, n, 0, i + 1)
        return cw[p - 1] + (n - cc[p - 1]) * sz[p - 1]

    def high(i: int, n: int) -> int:  # the most
        return cw[i] - low(i, cc[i] - n)

    def frame(i: int, n: int, rows: int) -> list[Any] | None:
        """Choices for the count of size index ``i - 1`` with ``n`` entities and ``rows`` rows still to place."""
        if i == 0:
            return None
        s = sz[i - 1]
        lo, hi = max(0, n - cc[i - 1]), min(av[i - 1], n, rows // s)
        if lo > hi:
            return None
        # rows - v*s >= low(i - 1, n - v) holds up to some v (its slack falls as v grows: the sizes below are
        # smaller), rows - v*s <= high(i - 1, n - v) from some v on: the feasible counts form one interval
        a, b = lo, hi
        while a < b:
            mid = (a + b + 1) // 2
            if rows - mid * s >= low(i - 1, n - mid):
                a = mid
            else:
                b = mid - 1
        vmax = a if rows - a * s >= low(i - 1, n - a) else lo - 1
        a, b = lo, vmax
        while a < b:
            mid = (a + b) // 2
            if rows - mid * s <= high(i - 1, n - mid):
                b = mid
            else:
                a = mid + 1
        vmin = a
        if vmin > vmax or rows - vmin * s > high(i - 1, n - vmin):
            return None
        v0 = vmin + int(rng.integers(vmax - vmin + 1)) if vmax > vmin else vmin
        return [i, n, rows, _outward(v0, vmin, vmax)]

    x = [0] * m
    if r == 0:
        if t == 0:
            yield x
        return
    if r > cc[m] or not low(m, r) <= t <= high(m, r):
        return
    first = frame(m, r, t)
    frames = [first] if first is not None else []
    while frames:
        if budget[0] < 0:
            return
        i, n, rows, values = frames[-1]
        v = next(values, None)
        if v is None:
            x[i - 1] = 0
            frames.pop()
            continue
        x[i - 1] = v
        n2, rows2 = n - v, rows - v * sz[i - 1]
        if n2 == 0:
            if rows2 == 0:
                yield x
            continue
        budget[0] -= 1
        nxt = frame(i - 1, n2, rows2)
        if nxt is not None:
            frames.append(nxt)


#: R9-04: the exact search restarts this many times (fresh random choices), each with its share of the steps: a
#: depth-first search stuck under one bad early choice is the common way to run out of steps.
_EXACT_RESTARTS = 4
#: R9-04: after each label's choice the bounds of at most this many labels after it are checked (the next ones and
#: the last, largest ones), so a step costs the same however many labels there are.
_BOUNDS_LABELS = 8


def _bounds_ok(sz: list[int], av: list[int], rest: list[tuple[int, int, int]]) -> bool:
    """Each label of ``rest`` can still get its rows from the entities left, by the bounds alone: between the rows of
    its count of the smallest and of the largest entities left."""
    cc, cw = [0], [0]
    for s, a in zip(sz, av):
        cc.append(cc[-1] + a)
        cw.append(cw[-1] + a * s)
    total, rows = cc[-1], cw[-1]

    def low(n: int) -> int:
        if n <= 0:
            return 0
        p = bisect.bisect_left(cc, n)
        return cw[p - 1] + (n - cc[p - 1]) * sz[p - 1]

    return all(n <= total and low(n) <= g <= rows - low(total - n) for _, n, g in rest)


def _search_counts(sz: list[int], avail: list[int], need: list[tuple[int, int, int]],
                   rng: np.random.Generator) -> np.ndarray | None:
    """R9-04: a table of counts (labels of ``need`` x block sizes ``sz``) giving every label ``(code, entities,
    rows)`` exactly, with each size's entities ``avail`` used up; None when none exists or the search gave up
    (:data:`_EXACT_NODES` steps over :data:`_EXACT_RESTARTS` randomised restarts). Depth-first over the labels,
    fewest entities first (one-entity labels have one choice), each choice kept only while every label after it
    can still be served (:func:`_bounds_ok`); the last label takes what is left, which fits by the totals.
    Deterministic for a given ``rng``."""
    order = sorted(range(len(need)), key=lambda q: (need[q][1], need[q][0]))
    m = len(sz)
    head = order[:-1]
    if not head:
        table = np.zeros((len(need), m), dtype=np.int64)
        table[order[0]] = avail
        return table
    for _ in range(_EXACT_RESTARTS):
        table = _search_once(sz, avail, need, order, rng, [_EXACT_NODES // _EXACT_RESTARTS])
        if table is not None:
            return table
    return None


def _search_once(sz: list[int], avail: list[int], need: list[tuple[int, int, int]], order: list[int],
                 rng: np.random.Generator, budget: list[int]) -> np.ndarray | None:
    """One depth-first pass of :func:`_search_counts` within ``budget`` steps (a step: one count tried; a label's
    choice costs steps in proportion to the block sizes, so the work is bounded whatever the table)."""
    head = order[:-1]
    av = list(avail)
    per_choice = 1 + (3 * len(sz) + 2 * _BOUNDS_LABELS) // 32

    def rest(pos: int) -> list[tuple[int, int, int]]:
        tail = order[pos:]
        if len(tail) > 2 * _BOUNDS_LABELS:
            tail = tail[:_BOUNDS_LABELS] + tail[-_BOUNDS_LABELS:]
        return [need[q] for q in tail]
    if not _bounds_ok(sz, av, [need[q] for q in order]):
        return None
    gens = [_label_vectors(sz, list(av), need[head[0]][1], need[head[0]][2], rng, budget)]
    chosen: list[list[int]] = []
    while gens:
        if budget[0] < 0:
            return None
        vec = next(gens[-1], None)
        if vec is None:  # this label has no (further) choice: the previous label's choice is taken back
            gens.pop()
            if chosen:
                for j, v in enumerate(chosen.pop()):
                    av[j] += v
            continue
        vec = list(vec)
        budget[0] -= per_choice
        for j, v in enumerate(vec):
            av[j] -= v
        if not _bounds_ok(sz, av, rest(len(chosen) + 1)):
            for j, v in enumerate(vec):
                av[j] += v
            continue
        chosen.append(vec)
        if len(chosen) == len(head):
            table = np.zeros((len(need), len(sz)), dtype=np.int64)
            for q, v in zip(head, chosen):
                table[q] = v
            table[order[-1]] = av
            return table
        q = head[len(chosen)]
        gens.append(_label_vectors(sz, list(av), need[q][1], need[q][2], rng, budget))
    return None


def _witness_counts(witness: Any, sz: np.ndarray, avail: np.ndarray,
                    need: list[tuple[int, int, int]]) -> np.ndarray | None:
    """R9-04: the table of counts of ``witness`` (the source's label codes and block sizes, one pair per real entity)
    when it is an exact deal of these entities (the same block sizes, entities and rows per label), else None."""
    codes, wsizes = (np.asarray(w, dtype=np.int64).reshape(-1) for w in witness)
    if len(codes) != len(wsizes) or len(codes) != int(avail.sum()):
        return None
    row = {c: q for q, (c, _, _) in enumerate(need)}
    rows = np.asarray([row.get(int(c), -1) for c in codes.tolist()], dtype=np.int64)
    cols = np.searchsorted(sz, wsizes)
    if (rows < 0).any() or (cols >= len(sz)).any() or (sz[np.minimum(cols, len(sz) - 1)] != wsizes).any():
        return None
    table = np.zeros((len(need), len(sz)), dtype=np.int64)
    np.add.at(table, (rows, cols), 1)
    ok = (np.array_equal(table.sum(axis=0), avail)
          and np.array_equal(table.sum(axis=1), np.asarray([n for _, n, _ in need], dtype=np.int64))
          and np.array_equal(table @ np.asarray(sz, dtype=np.int64), np.asarray([g for _, _, g in need], dtype=np.int64)))
    return table if ok else None


def _mix_counts(table: np.ndarray, sz: np.ndarray, rng: np.random.Generator, attempts: int) -> np.ndarray:
    """R9-04: random exchanges that keep every label's entities and rows (and every size's entities): label ``a``
    gives one entity of size ``i`` and one of size ``j`` for ``b``'s of sizes ``k`` and ``l``, ``i + j = k + l``.
    They move a witness deal (:func:`_witness_counts`) away from the real holders' block sizes wherever another
    exact deal is reachable."""
    t = table.copy()
    n_labels, m = t.shape
    if n_labels < 2 or m < 2:
        return t
    s = np.asarray(sz, dtype=np.int64)
    for _ in range(attempts):
        a, b = (int(v) for v in rng.choice(n_labels, 2, replace=False))
        ia, ib = np.flatnonzero(t[a]), np.flatnonzero(t[b])
        if not len(ia) or not len(ib):
            continue
        i, k = int(ia[rng.integers(len(ia))]), int(ib[rng.integers(len(ib))])
        if i == k:
            continue
        want = s[ia] + (s[i] - s[k])  # b's size l for each a's size j
        ls = np.searchsorted(s, want)
        ok = (ls < m) & (s[np.minimum(ls, m - 1)] == want)
        ls = np.minimum(ls, m - 1)
        ok &= (t[a][ia] >= np.where(ia == i, 2, 1)) & (t[b][ls] >= np.where(ls == k, 2, 1))
        ok &= ~((ia == k) & (ls == i))  # a change of nothing
        cand = np.flatnonzero(ok)
        if not len(cand):
            continue
        pick = int(cand[rng.integers(len(cand))])
        j, ll = int(ia[pick]), int(ls[pick])
        for q, d in ((i, -1), (j, -1), (k, 1), (ll, 1)):
            t[a, q] += d
            t[b, q] -= d
    return t


def _assign_counts(labels: np.ndarray, ents: np.ndarray, inv: np.ndarray, cur: np.ndarray, code_list: list[int],
                   table: np.ndarray, rng: np.random.Generator) -> None:
    """Write the deal ``table`` (labels ``code_list`` x block-size index ``inv``) into ``labels``: within a block
    size, entities keep the label they were dealt while that label still has room there (in a random order), and
    the others take the labels left, at random."""
    row = {c: q for q, c in enumerate(code_list)}
    room = table.copy()
    members = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[members], np.arange(table.shape[1] + 1))
    for j in range(table.shape[1]):
        pos = members[bounds[j]:bounds[j + 1]]
        if not len(pos):
            continue
        pos = pos[rng.permutation(len(pos))] if len(pos) > 1 else pos
        free: list[int] = []
        for p in pos.tolist():
            q = row[int(cur[p])]
            if room[q, j] > 0:
                room[q, j] -= 1
            else:
                free.append(p)
        if free:
            want = np.repeat(np.flatnonzero(room[:, j]), room[room[:, j] > 0, j])
            for p, q in zip(free, want.tolist()):
                labels[ents[p]] = code_list[q]
            room[:, j] = 0


#: At most this many passes over the few-entity labels (:func:`_exact_few`); each pass that changes nothing ends it.
_FEW_PASSES = 8
#: A closer step looks at most this many entity sizes on each side of the ideal one (:func:`_exact_few`).
_FEW_WINDOW = 64


def _exact_few(codes: np.ndarray, few: np.ndarray, counts: np.ndarray, diff: np.ndarray,
               buckets: dict[int, dict[int, list[int]]], swap_pair: Callable[[int, int, int, int], None]) -> np.ndarray:
    """:func:`balance_entity_rows`, first stage: make each movable label held by fewer than ``--min-cell``
    entities (a D35 pseudonym) exact, and return the labels left for the descent (the many-entity ones and the
    few-entity ones still off).

    A label ``c`` that is ``e`` rows over swaps one of its entities (size ``x``) for one of size exactly
    ``x - e`` held by a many-entity label (it absorbs the ``e`` rows; the one with the most entities) or by a
    few-entity label that is off too; with no such entity it steps closer through a many-entity label (the
    size nearest ``x - e`` that lowers its error) and tries again. A label made exact is never a partner again,
    so each exact swap leaves one label fewer off and every closer step lowers ``|e|``: the stage ends. Labels
    in order of entity count; sizes in increasing order; partners by fixed rules (the entity of a size is
    drawn at random by ``swap_pair``)."""
    many = {int(c) for c in codes.tolist() if not few[c]}
    holders: dict[int, set[int]] = {}  # size -> movable labels holding an entity of that size
    for c in codes.tolist():
        for s in buckets[c]:
            holders.setdefault(s, set()).add(c)

    def exact(c: int, e: int) -> tuple[int, int, int] | None:
        for x in sorted(buckets[c]):
            cands = [h for h in holders.get(x - e, ()) if h != c and (h in many or diff[h])]
            if cands:
                return min(cands, key=lambda h: (h not in many, -int(counts[h]), h)), x, x - e
        return None

    def helps(h: int, c: int, d: int) -> bool:
        """``h`` may take ``d`` rows from ``c``: a many-entity label, or a few-entity one that is off the other way
        and gets closer too."""
        return h != c and (h in many or (diff[h] * d < 0 and abs(diff[h] + d) < abs(diff[h])))

    def closer(c: int, e: int, all_sizes: list[int]) -> tuple[int, int, int] | None:
        """The swap that brings ``c`` nearest its rows (sizes outward from the ideal ``x - e``) with a partner that
        :func:`helps`; a few-entity partner first on a tie (both get closer)."""
        best: tuple[tuple[int, bool, int, int], int, int, int] | None = None
        for x in sorted(buckets[c]):
            i = bisect.bisect_left(all_sizes, x - e)
            lo, hi = i - 1, i
            for _ in range(2 * _FEW_WINDOW):
                if lo < 0 and hi >= len(all_sizes):
                    break
                if hi >= len(all_sizes) or (lo >= 0 and (x - e) - all_sizes[lo] <= all_sizes[hi] - (x - e)):
                    s, lo = all_sizes[lo], lo - 1
                else:
                    s, hi = all_sizes[hi], hi + 1
                gap = abs(s - (x - e))  # c's error after the swap
                if gap >= abs(e) or (best is not None and gap > best[0][0]):
                    break
                hs = [h for h in holders.get(s, ()) if helps(h, c, x - s)]
                if s != x and hs:
                    h = min(hs, key=lambda h: (h in many, -int(counts[h]), h))
                    key = (gap, h in many, s, x)
                    if best is None or key < best[0]:
                        best = (key, h, x, s)
                    break
        return None if best is None else best[1:]

    order = sorted((int(c) for c in codes.tolist() if few[c]), key=lambda c: (int(counts[c]), c))
    for _ in range(_FEW_PASSES):
        progress = False
        all_sizes = sorted(s for s, hs in holders.items() if hs)
        for c in order:
            while diff[c]:
                step = exact(c, int(diff[c])) or closer(c, int(diff[c]), all_sizes)
                if step is None:
                    break
                h, x, s = step
                swap_pair(c, h, x, s)
                for lab in (c, h):
                    for sz in (x, s):
                        if sz in buckets[lab]:
                            holders.setdefault(sz, set()).add(lab)
                        else:
                            holders.get(sz, set()).discard(lab)
                progress = True
        if not progress:
            break
    return np.asarray([c for c in codes.tolist() if c in many or diff[c]], dtype=np.int64)


def balance_entity_values(values: np.ndarray, sizes: np.ndarray, row_counts: dict[Any, int] | None,
                          missing_tokens: Any, levels: bool, rng: np.random.Generator,
                          min_entities: int = 1, pseudonyms: Any = frozenset(), witness: Any = None,
                          report: list[dict[str, Any]] | None = None) -> np.ndarray:
    """One synthetic value per entity (a D4 column), re-dealt among the entities so each level's and each NA
    token's rows come as close as possible to the source's (``row_counts``: written value -> real rows, from
    ``profile.entity.row_counts``, a pseudonymised level under its pseudonym; R2-10). Values move only between
    entities that are both missing or both non-missing, so the D6 entity patterns stay; non-missing values move
    only for level-generated columns (``levels``), whose values are the real levels or their pseudonyms
    (``pseudonyms``: the column's D35 pseudonyms, which write no real value however rare: R7-06). See
    :func:`balance_entity_rows`; ``witness`` (PRIVATE: each source entity's value as written, aligned with
    ``sizes``) proves an exact deal exists where it fits (R9-04), and rows left off are added to ``report``."""
    if not row_counts or len(values) < 2:
        return values
    codes, uniq = _factorize(values)
    toks = set(missing_tokens)
    is_tok = np.asarray([u in toks for u in uniq.tolist()], dtype=bool)
    out = np.array(values, dtype=object, copy=True)
    for part, on, tokens in ((~is_tok[codes], levels, False), (is_tok[codes], True, True)):
        idx = np.flatnonzero(part)
        if not on or len(idx) < 2:
            continue
        sub, inv = np.unique(codes[idx], return_inverse=True)
        written = uniq[sub].tolist()
        target = [row_counts.get(u, 0) for u in written]
        real = [u not in pseudonyms for u in written]
        new = balance_entity_rows(inv.reshape(-1), sizes[idx], target, rng, min_entities, real,
                                  witness=_part_witness(witness, sizes, written, toks, tokens), report=report)
        out[idx] = uniq[sub][new]
    return out


def _part_witness(witness: Any, sizes: np.ndarray, written: list[Any], toks: set[Any],
                  tokens: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """R9-04: the (label code, block size) pairs of the source entities whose written value falls in one part of
    :func:`balance_entity_values` (NA tokens, or values), or None (no witness, or a source value the synthetic part
    does not hold: then it proves nothing for this deal)."""
    if witness is None or len(witness) != len(sizes):
        return None
    code_of = {u: q for q, u in enumerate(written)}
    pairs = [(code_of.get(v, -1), int(s)) for v, s in zip(witness.tolist(), np.asarray(sizes).tolist())
             if (v in toks) == tokens]
    if not pairs or any(q < 0 for q, _ in pairs):
        return None
    return np.asarray([q for q, _ in pairs], dtype=np.int64), np.asarray([s for _, s in pairs], dtype=np.int64)


#: At most this many value bins for dealing a within-ID-constant number or date to ID blocks (R3-07).
ENTITY_ROW_BINS = 50


def _entity_numbers(values: np.ndarray, p: ColumnProfile) -> tuple[np.ndarray, np.ndarray] | None:
    """(entities with a readable number/date, their numeric view) of per-entity values; None without a view."""
    keep = [i for i, v in enumerate(values.tolist())
            if isinstance(v, str) and not is_missing(v, p.missing_policy, p.na_tokens)]
    x = numeric_view_aligned(p, [values[i].strip() for i in keep])
    if x is None:
        return None
    idx = np.asarray(keep, dtype=np.int64)
    ok = np.isfinite(x)
    return idx[ok], x[ok]


#: Consecutive IDs (by value) are grouped into at most this many groups for the bin search.
_BIN_GROUPS = 400


def entity_row_bins(entity_values: np.ndarray, sizes: np.ndarray, p: ColumnProfile,
                    min_cell: int) -> dict[str, Any] | None:
    """PRIVATE aggregate of a within-ID-constant number or date (D4; R3-07): consecutive bins of the source
    IDs ordered by value, each holding at least ``min_cell`` IDs (at most :data:`ENTITY_ROW_BINS`), with
    the number of IDs and the source rows (their block sizes) of each bin. Generation deals the per-entity
    values so each bin's rows come out as in the source (:func:`balance_entity_numbers`), as levels do
    (R2-10): a value held by long ID blocks in the source is not spread onto short ones. The bins are cut
    where the block sizes change (least squared deviation of the sizes within the bins), since inside a bin
    the values are dealt at random. None when fewer than two bins fit, or the column has no numeric view
    (levels use :func:`balance_entity_values`)."""
    got = _entity_numbers(np.asarray(entity_values, dtype=object), p)
    if got is None:
        return None
    idx, x = got
    k = max(int(min_cell), 1)
    n = len(x)
    nb = min(ENTITY_ROW_BINS, n // k)
    if nb < 2:
        return None
    s = np.asarray(sizes, dtype=float)[idx][np.argsort(x, kind="stable")]
    ids = _size_bins(s, k, nb)
    rows = np.add.reduceat(s, np.concatenate([[0], np.cumsum(ids)[:-1]]))
    return {"ids": [int(c) for c in ids], "rows": [int(round(r)) for r in rows.tolist()]}


def _size_bins(s: np.ndarray, k: int, nb: int) -> list[int]:
    """IDs per bin: ``nb`` consecutive bins of the sizes ``s`` (ordered by value), each >= ``k`` IDs, with the
    least total squared deviation of the sizes from their bin's mean (dynamic programming over at most
    :data:`_BIN_GROUPS` groups of consecutive IDs); equal sizes give equal-count bins."""
    n = len(s)
    g = max(1, -(-n // _BIN_GROUPS))  # IDs per group (the last group may be shorter)
    starts = np.arange(0, n, g)
    G = len(starts)
    cnt = np.diff(np.concatenate([starts, [n]])).astype(float)
    c1 = np.concatenate([[0.0], np.cumsum(np.add.reduceat(s, starts))])
    c2 = np.concatenate([[0.0], np.cumsum(np.add.reduceat(s * s, starts))])
    cc = np.concatenate([[0.0], np.cumsum(cnt)])
    i = np.arange(G + 1)[:, None]
    j = np.arange(G + 1)[None, :]
    m = cc[j] - cc[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = (c2[j] - c2[i]) - np.where(m > 0, (c1[j] - c1[i]) ** 2 / np.where(m > 0, m, 1), 0.0)
    cost = np.where((j > i) & (m >= k), np.maximum(cost, 0.0), np.inf)
    # equal-count bins win ties (a bin's size spread is zero whatever the cut): a tiny preference for balance
    cost = cost + 1e-9 * (m - n / nb) ** 2
    dp = np.full(G + 1, np.inf)
    dp[0] = 0.0
    back = np.zeros((nb + 1, G + 1), dtype=np.int64)
    for b in range(1, nb + 1):
        tot = dp[:, None] + cost
        back[b] = np.argmin(tot, axis=0)
        dp = tot[back[b], np.arange(G + 1)]
    if not np.isfinite(dp[G]):  # cannot happen when n >= nb * k; equal-count bins as a fallback
        return [n // nb + (1 if r < n % nb else 0) for r in range(nb)]
    cuts = [G]
    for b in range(nb, 0, -1):
        cuts.append(int(back[b][cuts[-1]]))
    cuts.reverse()
    return [int(cc[cuts[t + 1]] - cc[cuts[t]]) for t in range(nb)]


def balance_entity_numbers(values: np.ndarray, sizes: np.ndarray, bins: dict[str, Any] | None, p: ColumnProfile,
                           rng: np.random.Generator, row_counts: dict[Any, int] | None = None,
                           min_entities: int = 1, pseudonyms: Any = frozenset(),
                           report: list[dict[str, Any]] | None = None) -> np.ndarray:
    """One synthetic value per entity (a D4 number or date), re-dealt among the entities so each value bin's
    rows (``bins`` from :func:`entity_row_bins`) come as close as possible to the source's (R3-07): bins are
    labels for :func:`balance_entity_rows`, and the values of a bin move to the entities that took its label.
    Tokens that are not numbers (suppression marks such as ``*``, sentinel dates) are labels of their own,
    with their source rows from ``row_counts`` (written value -> rows: a pseudonymised token under its
    pseudonym); a real one held by fewer than ``min_entities`` IDs is not moved (as rare real levels, R2-10),
    a D35 pseudonym (``pseudonyms``) always is (R7-06). Blanks and NA tokens stay
    (:func:`balance_entity_values` deals them); every value is kept, so the per-entity distribution, the exact
    min/max and the D6 entity patterns are unchanged. The
    synthetic numbers are binned by rank with the source's number of IDs per bin (scaled to the synthetic
    count), since interpolated values near an edge could fall on either side of it; every bin holds at
    least ``--min-cell`` source IDs by construction, so each one moves."""
    if not bins or len(values) < 2:
        return values
    vals = np.asarray(values, dtype=object)
    got = _entity_numbers(vals, p)
    if got is None or len(got[0]) < 2:
        return values
    idx, x = got
    per_bin = largest_remainder(np.asarray(bins["ids"], dtype=float), len(x))
    nb = len(per_bin)
    lab = np.empty(len(x), dtype=np.int64)
    lab[np.argsort(x, kind="stable")] = np.repeat(np.arange(nb), per_bin)
    weights = [float(r) for r in bins["rows"]]
    filled = np.fromiter((isinstance(v, str) and not is_missing(v, p.missing_policy, p.na_tokens)
                          for v in vals.tolist()), dtype=bool, count=len(vals))
    filled[idx] = False
    tok_idx = np.flatnonzero(filled)  # filled, but not a number: a token
    tok_uniq = np.empty(0, dtype=object)
    if len(tok_idx) and row_counts:
        tok_codes, tok_uniq = _factorize(vals[tok_idx])
        held = np.bincount(tok_codes, minlength=len(tok_uniq))
        weights += [float(row_counts.get(u, 0)) if held[j] >= max(int(min_entities), 1) or u in pseudonyms
                    else 0.0 for j, u in enumerate(tok_uniq.tolist())]
        idx = np.concatenate([idx, tok_idx])
        lab = np.concatenate([lab, nb + tok_codes])
    off: list[dict[str, Any]] = []
    new = balance_entity_rows(lab, np.asarray(sizes, dtype=np.int64)[idx], weights, rng, 1, report=off)
    # R9-04: a token's rows (a level among the numbers) left off are reported; the value bins only approximate the
    # distribution, so theirs are not
    toks_off = [c for o in off for c in o["codes"] if c >= nb]
    if report is not None and toks_off:
        report.append({"labels": len(toks_off), "rows": off[0]["rows"], "codes": toks_off})
    if np.array_equal(new, lab):
        return values
    out = np.array(vals, dtype=object, copy=True)
    for b in np.unique(lab[new != lab]).tolist():
        src = idx[(lab == b) & (new != b)]
        dst = idx[(new == b) & (lab != b)]
        if b < nb:
            out[dst] = vals[src][rng.permutation(len(src))]
        else:
            out[dst] = tok_uniq[b - nb]
    return out


# --------------------------------------------------------------------------- shared helpers
_DET_LEVEL_TYPES = frozenset({T.FACTOR, T.ORG_CODE, T.ORDINAL, T.SCHOOL_YEAR, T.BOOLEAN, T.FLAG_SINGLE_TOKEN,
                              T.INTEGER_YEAR})
#: Dependents whose real values are never written: mapped onto their own synthetic values.
_TEXT_DEP_TYPES = frozenset((NEVER_COPY_TYPES - {T.IDENTIFIER}) | {T.CITY, T.STATE, T.ZIP, T.DIGIT_CODE})
_TEMPLATE_TARGETS = frozenset({T.EMAIL, T.PERSON_NAME, T.HIGH_CARD_CATEGORICAL, T.FREE_TEXT})
_TEMPLATE_INPUT_EXCLUDED = frozenset({T.FLOAT, T.FREE_TEXT, T.EMPTY, T.BINARY_NESTED, T.MULTI_SELECT, T.LAT, T.LON,
                                      T.DECIMAL, T.NUMERIC_FORMATTED, T.NUMERIC_WITH_TOKENS})
#: Composite-key columns besides the level-generated ones (:func:`_level_generated`) and dates: codes masked
#: shape for shape and invented labels, which keep their distinct counts.
_COMPOSITE_TYPES = frozenset({T.DIGIT_CODE, T.HIGH_CARD_CATEGORICAL})
#: Composite-key candidates paired for (ID, A, B) keys first: labels, codes and dates (then discrete numbers).
_COMPOSITE_PAIR_FIRST = frozenset({T.ORG_CODE, T.FACTOR, T.ORDINAL, T.SCHOOL_YEAR, T.INTEGER_YEAR, T.DIGIT_CODE,
                                   T.HIGH_CARD_CATEGORICAL})
#: Numbers a rule compares (:func:`rule_view`): plain numbers in any storage, Arrow decimals included (a declared
#: rule on Parquet decimals is built as on the same numbers in a text file, R4-05), and formatted numbers
#: (``$1,234.50``, ``12.5%``) with or without suppression tokens (R5-05), read through their learned format.
_FORMATTED_NUMBER_TYPES = frozenset({T.NUMERIC_FORMATTED, T.NUMERIC_WITH_TOKENS})
_NUMBER_TYPES = frozenset({T.INTEGER, T.INTEGER_YEAR, T.FLOAT, T.DECIMAL}) | _FORMATTED_NUMBER_TYPES
FD_SAMPLE = 20_000
MAX_SUGGESTIONS = 20
MAX_SORT_KEYS = 3
SORT_PREFIX = 2000
#: A suggested rule must fail for at least 5% of random value pairings (else the ranges imply it).
RANGE_IMPLIED = 0.95
_RULE_RE = re.compile(r"^\s*(.+?)\s*(<=|>=|=<|=>|<|>)\s*(.+?)\s*$")


def _is_date(p: ColumnProfile) -> bool:
    return TYPE_INFO[p.type].family == "date"


def _generated(p: ColumnProfile) -> bool:
    return p.role in ("row", "entity")


def _level_generated(p: ColumnProfile) -> bool:
    """Generated by exact quota over the real raw values, so every value keeps its count (the same test as
    ``generate.table.is_level_generated``): labels, booleans, years, codes kept as levels, discrete integers."""
    if p.type in LEVEL_TYPES:
        return True
    if p.type is T.INTEGER and p.subtype == "discrete":
        return True
    return p.type is T.DIGIT_CODE and bool(p.params.get("levels_mode"))


def _strip_all(values: np.ndarray) -> list[str | None]:
    return [None if v is None else v.strip() for v in values.tolist()]


def _stripped_codes(values: np.ndarray) -> np.ndarray:
    """Codes of the stripped values (whitespace variants share a code; None its own), stripping once per value."""
    codes, uniq = _factorize(values)
    stripped, _ = _factorize(np.asarray([u.strip() if isinstance(u, str) else None for u in uniq.tolist()] or [""],
                                        dtype=object))
    return stripped[codes] if len(uniq) else codes


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(codes, distinct raw values); a native null (None in typed formats) stays None, not NaN."""
    codes, uniques = codes_of(np.asarray(values, dtype=object))
    if all(isinstance(u, str) or u is None for u in uniques):
        return codes, uniques
    out = np.empty(len(uniques), dtype=object)
    out[:] = [u if isinstance(u, str) else None for u in uniques.tolist()]
    return np.asarray(codes, dtype=np.int64), out


def _entity_index(table: RawTable, entity: EntityProfile) -> np.ndarray | None:
    """Row -> entity number on the primary ID (blank IDs: own entity), as in ``profile_entity``."""
    if entity.primary is None:
        return None
    return entity_mod.entity_index(table.columns[entity.primary].values)[0]


def _disabled(settings: Settings, switch: str) -> bool:
    return settings.structure_disabled(switch)


def _status(settings: Settings, switch: str, ok: str) -> str:
    return f"disabled by config (structure: disable: [{switch}])" if _disabled(settings, switch) else ok


# --------------------------------------------------------------------------- profile entry point
def profile_structure(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile,
                      masks: dict[int, np.ndarray], settings: Settings,
                      entity_masks: dict[int, np.ndarray] | None = None,
                      entity_index: np.ndarray | None = None, file_rel: str | None = None) -> StructureProfile:
    """Structure profile: missingness (FOUNDATION) + every AREA D detector (see module docstring).

    ``masks``: missing flags per row of the row-level columns; ``entity_masks``: missing flags per entity of
    the within-ID-constant columns (D4), with ``entity_index`` (row -> entity). ``file_rel`` (the input-relative
    path) selects the declared rules qualified with ``file:`` (None: every rule).
    """
    mm, row_codes = _pattern_model(masks, table.n_rows, settings.min_cell)
    if entity_masks:
        profile_entity_missingness(mm, row_codes, entity_masks, entity_index, entity.n_entities, settings.min_cell)
    sp = StructureProfile(missingness=mm)
    fds, fd_private = _functional_dependencies(table, profiles, entity, settings)
    sp.functional_dependencies, sp.private["fds"] = fds, fd_private
    index = _entity_index(table, entity) if entity.multi_row_ids else None
    tmpl, tmpl_private = _derived_templates(table, profiles, entity, settings, index)
    sp.derived_templates, sp.private["templates"] = tmpl, tmpl_private
    keys, keys_private = _composite_keys(table, profiles, entity, settings, fd_private, index)
    sp.composite_keys, sp.private["composite"] = keys, keys_private
    sp.suggested_rules = suggest_inequality_rules(table, profiles, settings.min_cell)
    sp.private["rule_data"], sp.private["rule_targets"] = _rule_data(table, profiles, settings, index, file_rel)
    sp.panels = _panels(table, profiles, entity, index)
    sort_order, sort_private = _sort_order(table, profiles, settings)
    sp.sort_order, sp.private["sort"] = sort_order, sort_private
    sp.duplicate_rows = _duplicate_rows(table, profiles, settings)
    sp.swap_groups = swap_groups(sp, profiles)
    return sp


# --------------------------------------------------------------------------- functional dependencies
def _det_ok(p: ColumnProfile) -> bool:
    if not _generated(p) or p.params.get("id_candidate"):
        return False
    return (p.type in _DET_LEVEL_TYPES or (p.type is T.INTEGER and p.subtype == "discrete")
            or (p.type is T.DIGIT_CODE and bool(p.params.get("levels_mode"))))


def _dep_ok(p: ColumnProfile) -> bool:
    if not _generated(p) or p.params.get("id_candidate"):
        return False
    return _det_ok(p) or p.type in _TEXT_DEP_TYPES


def _fd_edges(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile
              ) -> tuple[set[tuple[int, int]], dict[int, int]]:
    """Every determinant -> dependent pair that holds in 100% of rows (support rules in the module doc)."""
    n = table.n_rows
    edges: set[tuple[int, int]] = set()
    distinct: dict[int, int] = {}
    if n < 2:
        return edges, distinct
    codes: dict[int, np.ndarray] = {}
    for p in profiles:
        if _det_ok(p) or _dep_ok(p):
            codes[p.index], u = _factorize(table.columns[p.index].values)
            distinct[p.index] = len(u)
    sample = None
    if n > FD_SAMPLE:
        sample = np.sort(np.random.default_rng(0).choice(n, FD_SAMPLE, replace=False))
    dets = [i for i in codes if _det_ok(profiles[i]) and distinct[i] >= 2]
    deps = [i for i in codes if _dep_ok(profiles[i]) and distinct[i] >= 2]
    for a in dets:
        pa = profiles[a]
        units = entity.n_entities if (pa.role == "entity" and entity.n_entities) else n
        supported = 2 * distinct[a] <= units
        org_a = pa.type is T.ORG_CODE
        if not supported and not org_a:
            continue
        ca = codes[a]
        order = None
        for b in deps:
            pb = profiles[b]
            if b == a or distinct[b] > distinct[a]:
                continue
            if not supported and pb.type is not T.ORG_CODE:
                continue
            if pb.role == "entity" and pa.role != "entity":
                continue
            if sample is not None and not is_functional(ca[sample], codes[b][sample]):
                continue
            if order is None:
                order = np.argsort(ca, kind="stable")
                sa = ca[order]
                new_run = np.empty(n, dtype=bool)
                new_run[0] = True
                new_run[1:] = sa[1:] != sa[:-1]
            sb = codes[b][order]
            change = np.zeros(n, dtype=bool)
            change[1:] = sb[1:] != sb[:-1]
            if not np.any(change & ~new_run):
                edges.add((a, b))
    return edges, distinct


def _select_fds(edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """One determinant per dependent, preferring root determinants; returned in application order."""
    nodes = sorted({x for e in edges for x in e})
    parent = {x: x for x in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if (b, a) in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[int, list[int]] = {}
    for x in nodes:
        members.setdefault(find(x), []).append(x)
    rep = {c: min(m) for c, m in members.items()}
    incoming = {c: False for c in members}
    outgoing = {c: False for c in members}
    for a, b in edges:
        ca, cb = find(a), find(b)
        if ca != cb:
            incoming[cb] = True
            outgoing[ca] = True
    root = {c: (not incoming[c]) for c in members}
    chosen: dict[int, int] = {}
    for b in nodes:
        cb = find(b)
        if root[cb] and b == rep[cb]:
            continue
        cands: list[int] = []
        if root[cb] and b != rep[cb] and (rep[cb], b) in edges:
            cands = [rep[cb]]
        if not cands:
            cands = [x for x in nodes if (x, b) in edges and find(x) != cb and root[find(x)] and x == rep[find(x)]]
        if not cands:
            cands = [x for x in nodes if (x, b) in edges and x != b and x not in chosen]
        if cands:
            chosen[b] = cands[0]
    # application order: determinants before their dependents; drop anything cyclic
    order: list[tuple[int, int]] = []
    done: set[int] = set()
    pending = dict(chosen)
    while pending:
        ready = [b for b, a in pending.items() if a not in pending]
        if not ready:
            break
        for b in sorted(ready):
            order.append((pending.pop(b), b))
            done.add(b)
    return order


def _functional_dependencies(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile,
                             settings: Settings) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges, distinct = _fd_edges(table, profiles, entity)
    if not edges:
        return [], []
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    status = _status(settings, "functional_dependencies", "applied")
    for a, b in _select_fds(edges):
        ca, ua = _factorize(table.columns[a].values)
        first = np.full(len(ua), -1, dtype=np.int64)
        first[ca[::-1]] = np.arange(len(ca) - 1, -1, -1)
        dep_raw = table.columns[b].values
        mapping = {ua[c]: dep_raw[first[c]] for c in range(len(ua))}
        pa, pb = profiles[a], profiles[b]
        public.append({
            "determinant": pa.name, "dependent": pb.name,
            "kind": "1:1" if (b, a) in edges else "many-to-one",
            "determinant_values": int(distinct[a]), "dependent_values": int(distinct[b]),
            "status": status, "text": f"{pa.name} -> {pb.name}",
        })
        private.append({"det": a, "dep": b, "mapping": mapping})
    return public, private


def find_functional_dependencies(table: RawTable, profiles: list[ColumnProfile]) -> list[dict[str, Any]]:
    """AREA D hook: exact functional dependencies holding in 100% of rows (school_id -> school_name, D5).

    Returns report-safe descriptions (column names, counts; no values), one per
    dependent with the determinant used to regenerate it (module docstring).
    """
    edges, distinct = _fd_edges(table, profiles, EntityProfile())
    return [{"determinant": profiles[a].name, "dependent": profiles[b].name,
             "kind": "1:1" if (b, a) in edges else "many-to-one",
             "determinant_values": int(distinct[a]), "dependent_values": int(distinct[b]),
             "text": f"{profiles[a].name} -> {profiles[b].name}"} for a, b in _select_fds(edges)]


# --------------------------------------------------------------------------- derived templates
def _input_ok(p: ColumnProfile) -> bool:
    return p.role in ("row", "entity", "id") and p.type not in _TEMPLATE_INPUT_EXCLUDED and not _is_date(p)


def _constant_within_primary(q: ColumnProfile, profiles: list[ColumnProfile], entity: EntityProfile) -> bool:
    if q.role == "entity":
        return True
    if q.role != "id" or entity.primary is None:
        return False
    if q.index == entity.primary:
        return True
    prim = profiles[entity.primary]
    return q.name in prim.params.get("alias_of", []) or q.name in prim.params.get("nested_in", [])


class _Stripped:
    """Lazy stripped view of ``values[rows]`` (None stays None); avoids copying every column."""

    def __init__(self, values: np.ndarray, rows: np.ndarray) -> None:
        self.values, self.rows = values, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, r: int) -> str | None:
        v = self.values[self.rows[r]]
        return None if v is None else v.strip()

    def array(self) -> np.ndarray:
        """All the stripped values as one object array (stripping once per distinct value)."""
        codes, uniq = subset_codes(self.values, self.rows)
        stripped = np.empty(len(uniq), dtype=object)
        stripped[:] = [u.strip() if isinstance(u, str) else None for u in uniq.tolist()]
        return stripped[codes]


def _template_groups(table: RawTable, profiles: list[ColumnProfile], info: dict[str, Any], rows: np.ndarray,
                     codes: np.ndarray, ustrip: list[str], local: list[str], usplit: list[Any] | None,
                     inputs: dict[int, Any], index: np.ndarray | None) -> dict[str, Any] | None:
    """PRIVATE (R5-04): the real value groups of a template's target, so the rebuilt column keeps its distinct
    count and frequency profile (``generate.structure.regroup_template``).

    A group is one distinct (stripped) target value; its units are the rows holding it (the IDs, for a column
    constant within the ID). Per group: its number of units (per unit, the ID's rows), its digit-suffix length
    and reserved-domain rank, and per input the *frequency rank* of its input value among that input's values
    (never the value). Only for templates whose inputs are all generated with the target's role (no ID input:
    an ID makes every value distinct already; a mixed role is rebuilt after the row moves instead)."""
    target = profiles[info["target"]]
    cols = sorted({p[1] for p in info["parts"] if p[0] == "f"})
    role = target.role
    if role not in ("row", "entity") or any(profiles[c].role != role for c in cols):
        return None
    per_entity = role == "entity" and index is not None and len(index) == table.n_rows
    stripped_codes, _ = codes_of(np.asarray(ustrip, dtype=object))
    gcode = stripped_codes[codes]
    if per_entity:
        assert index is not None
        ents, first = np.unique(index[rows], return_index=True)
        unit_pos = first  # the first template row of each ID
        sizes = np.bincount(index)[ents]
    else:
        unit_pos = np.arange(len(rows))
        sizes = None
    ugroup = gcode[unit_pos]
    n_groups = int(gcode.max()) + 1 if len(gcode) else 0
    if n_groups == 0:
        return None
    # each group's first unit gives its input values, digit length and domain
    g_first = np.full(n_groups, -1, dtype=np.int64)
    g_first[ugroup[::-1]] = np.arange(len(ugroup) - 1, -1, -1)
    rep = unit_pos[g_first]  # position (in rows) of each group's representative row
    ranks = np.zeros((n_groups, len(cols)), dtype=np.int32)
    combo = ugroup.copy()  # (group, input values) per unit: a lossy template merges several inputs in one group
    for j, c in enumerate(cols):
        vals = inputs[c].array() if hasattr(inputs[c], "array") else np.asarray(list(inputs[c]), dtype=object)
        vcode, _ = subset_codes(np.asarray(vals, dtype=object), unit_pos)
        vcode = np.asarray(vcode, dtype=np.int64)
        cnt = np.bincount(vcode)
        rank_of = np.empty(len(cnt), dtype=np.int64)
        rank_of[np.lexsort((np.arange(len(cnt)), -cnt))] = np.arange(len(cnt))
        ranks[:, j] = rank_of[vcode[g_first]]
        combo = np.asarray(pd.factorize(combo * (len(cnt) + 1) + vcode)[0], dtype=np.int64)
    if len(np.unique(combo)) > n_groups:
        # an initial or a case change merges different inputs into one value ('jsmith' for John and Jane Smith):
        # the synthetic inputs cannot be made to merge the same way, so the profile is not promised
        return {"lossy": True}
    out: dict[str, Any] = {"unit": "entity" if per_entity else "row", "inputs": cols, "ranks": ranks,
                           "units": np.bincount(ugroup, minlength=n_groups).astype(np.int64)}
    if per_entity:
        out["unit_group"] = ugroup.astype(np.int64)
        out["unit_size"] = np.asarray(sizes, dtype=np.int64)
    if "digits" in info:
        rendered, _ = tpl.render_array(info["parts"], {c: tpl.as_array(inputs[c]) for c in cols})
        out["dlen"] = np.fromiter((len(local[r]) - len(rendered[r]) for r in rep.tolist()), dtype=np.int64,
                                  count=n_groups)
    if usplit is not None and "domains" in info:
        dom: Counter = Counter()
        for s, c in zip(usplit, np.bincount(codes, minlength=len(usplit)).tolist()):
            dom[s[1].casefold()] += c
        order = {d: i for i, (d, _) in enumerate(sorted(dom.items(), key=lambda kv: (-kv[1], kv[0])))}
        out["dom"] = np.fromiter((order[usplit[int(codes[r])][1].casefold()] for r in rep.tolist()),
                                 dtype=np.int64, count=n_groups)
    return out


def _derived_templates(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile,
                       settings: Settings, index: np.ndarray | None = None
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = [p.name for p in profiles]
    found: dict[int, dict[str, Any]] = {}
    for p in profiles:
        if not _generated(p) or p.type not in _TEMPLATE_TARGETS or p.params.get("id_candidate"):
            continue
        # blanks and missing-like placeholders ('N/A', 'NULL' ...) are not data: the template must hold in
        # every other row (an e-mail column with a few 'N/A' still has its template)
        miss = missing_mask(table.columns[p.index].values, "typed", p.na_tokens)
        rows = np.flatnonzero(~miss)
        if len(rows) < max(settings.min_cell, 2):
            continue
        codes, uniq = subset_codes(table.columns[p.index].values, rows)
        ustrip = [u.strip() for u in uniq]
        usplit = [tpl.split_email(u) for u in ustrip]
        email = all(s is not None for s in usplit)
        if email:
            ulocal = [s[0] for s in usplit]  # type: ignore[index]
            local = [ulocal[c] for c in codes.tolist()]
        else:
            local = [ustrip[c] for c in codes.tolist()]
        inputs = {q.index: _Stripped(table.columns[q.index].values, rows)
                  for q in profiles if q.index != p.index and _input_ok(q)}
        parts = tpl.infer_template(local, inputs, settings.min_cell, email)
        if parts is None:
            continue
        info: dict[str, Any] = {"target": p.index, "parts": parts, "email": email, "rows": len(rows)}
        if email:
            ucount = np.bincount(codes, minlength=len(uniq))
            dom: Counter = Counter()
            for s, c in zip(usplit, ucount.tolist()):
                dom[s[1].casefold()] += c  # type: ignore[index]
            info["domains"] = tpl.reserved_domains(dom)
        if parts[-1][0] == "d":
            info["digits"] = dict(tpl.digit_suffix_lengths(parts, local, inputs))
        groups = _template_groups(table, profiles, info, rows, codes, ustrip, local, usplit if email else None,
                                  inputs, index)
        if groups is not None and groups.get("lossy"):
            info["lossy"] = True
        elif groups is not None:
            info["groups"] = groups
        found[p.index] = info
    # templates must not feed each other in a cycle: drop the later target of a cycle
    order: list[int] = []
    pending = dict(found)
    while pending:
        ready = [t for t, info in pending.items()
                 if not any(part[0] == "f" and part[1] in pending and part[1] != t for part in info["parts"])]
        if not ready:
            pending.pop(max(pending))
            continue
        for t in sorted(ready):
            order.append(t)
            pending.pop(t)
    public, private = [], []
    for t in order:
        info = found[t]
        p = profiles[t]
        inputs = sorted({part[1] for part in info["parts"] if part[0] == "f"})
        status = _status(settings, "derived_templates", "applied")
        if p.role == "entity" and not all(_constant_within_primary(profiles[i], profiles, entity) for i in inputs):
            status = "found, not applied (an input varies within the ID while the column is constant within it)"
        entry = {"column": p.name, "template": tpl.describe(info["parts"], names, info["email"]),
                 "inputs": [names[i] for i in inputs], "rows": info["rows"], "status": status}
        if info["email"]:
            entry["email_domains"] = len(info["domains"])
        public.append(entry)
        if status == "applied":
            private.append(info)
            # report.fidelity (R5-04): where the rebuilt column cannot keep the source's profile, its distinct count
            # is reported, not checked
            if info.get("lossy"):
                p.params["_template_profile"] = "the template merges different inputs into one value (an initial or " \
                                                "a case change), which the synthetic inputs cannot repeat"
            elif "groups" not in info and not any(profiles[i].role == "id" for i in inputs):
                p.params["_template_profile"] = "the template's inputs vary per row and per ID (rebuilt after the row " \
                                                "moves)"
    return public, private


def find_derived_templates(table: RawTable, profiles: list[ColumnProfile]) -> list[dict[str, Any]]:
    """AREA D hook: derived templates holding in 100% of rows (email = first.last.id@domain, D5).

    Report-safe descriptions (column names, transforms, separators); see
    ``profile.templates``.
    """
    return _derived_templates(table, profiles, EntityProfile(), Settings())[0]


# --------------------------------------------------------------------------- composite keys
def _composite_candidate(p: ColumnProfile) -> bool:
    """A row-level column that can complete a key with the ID (D5): every level-generated column - labels,
    booleans, years, codes, and discrete integers such as visit, period and term numbers (R4-04; not
    measurements, :func:`_measurement`) - plus masked codes, invented labels and dates. Continuous numbers
    (interpolated) are never keys."""
    if p.role != "row" or p.params.get("id_candidate"):
        return False
    return _level_generated(p) or p.type in _COMPOSITE_TYPES or _is_date(p)


def _measurement(p: ColumnProfile, limit: int) -> bool:
    """A discrete integer with more distinct values than ``limit`` (the larger of factor-max and the most rows
    of one ID): a measurement (score, minutes) whose uniqueness within the IDs is a coincidence. Integers that
    number an ID's rows (visit, period, term, sequence numbers) take no more values than that."""
    if p.type is not T.INTEGER or p.model is None:
        return False
    return len(p.model.nonmissing_counts) > limit


def _pair_rank(p: ColumnProfile) -> int:
    """Order of the (ID, A, B) pair candidates: labels, codes and dates first, then discrete numbers."""
    return 0 if p.type in _COMPOSITE_PAIR_FIRST or _is_date(p) else 1


def _composite_keys(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile, settings: Settings,
                    fd_private: list[dict[str, Any]] = (), index: np.ndarray | None = None
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if entity.primary is None or entity.multi_row_ids == 0:
        return [], []
    if index is None:
        index = _entity_index(table, entity)
    if index is None:
        return [], []
    blank_start = entity.n_entities - entity.blank_id_rows
    public, private = [], []
    status = _status(settings, "composite_keys", "enforced")
    fd_deps = {fd["dep"] for fd in fd_private}
    prim = profiles[entity.primary].name
    named = index < blank_start
    limit = max(settings.factor_max, int(np.bincount(index[named]).max()) if named.any() else 0)
    cols = _KeyColumns(table, profiles, index, named)
    rest: list[int] = []  # non-key candidates
    for p in profiles:
        if not _composite_candidate(p) or p.index in fd_deps or _measurement(p, limit):
            continue
        if cols.head_repeats([p.index]):  # most columns are not keys: rejected without a full pass
            rest.append(p.index)
            continue
        filled, codes = cols.full(p.index)
        rows = np.flatnonzero(filled)
        if not cols.varies(p.index):
            continue
        if _unique_within_ids(index, rows, [codes]):
            public.append({"columns": [prim, p.name], "rows": int(len(rows)),
                           "status": status, "text": f"({prim}, {p.name}) unique"})
            # 'pub': the public entry whose status generation updates (R5-03: never 'enforced' while violated)
            private.append({"x": p.index, "cols": [p.index], "pub": len(public) - 1})
        else:
            rest.append(p.index)
    # (ID, A, B): attendance by period, grades by (course, term) - when neither column alone is a key
    used: set[int] = set()
    pairable: list[int] = []
    for i in sorted(rest, key=lambda c: _pair_rank(profiles[c])):
        if len(pairable) >= MAX_PAIR_CANDIDATES:
            break
        if cols.varies(i):
            pairable.append(i)
    for i, a in enumerate(pairable):
        for b in pairable[i + 1:]:
            if a in used or b in used or cols.head_repeats([a, b]):
                continue
            (fa, ca), (fb, cb) = cols.full(a), cols.full(b)
            rows = np.flatnonzero(fa & fb)
            if len(rows) < 2 or not _unique_within_ids(index, rows, [ca, cb]):
                continue
            used.update((a, b))
            pa, pb = profiles[a], profiles[b]
            public.append({"columns": [prim, pa.name, pb.name], "rows": int(len(rows)), "status": status,
                           "text": f"({prim}, {pa.name}, {pb.name}) unique"})
            private.append({"x": a, "cols": [a, b], "pub": len(public) - 1})
    return public, private


#: At most this many non-key candidate columns are paired for (ID, A, B) keys (cost grows with the square).
MAX_PAIR_CANDIDATES = 12


class _KeyColumns:
    """Filled mask (named IDs only) and stripped codes of composite-key candidates, computed once per column and
    stage: on the rows of the first :data:`KEY_PRECHECK_IDS` IDs of a large table, then ten times as many (a
    repeat there rejects a key without a pass over every row; R4-04: discrete numbers add candidates whose pairs
    rarely repeat among a few IDs), and on every row only for what passes."""

    def __init__(self, table: RawTable, profiles: list[ColumnProfile], index: np.ndarray, named: np.ndarray) -> None:
        self.table, self.profiles, self.index, self.named = table, profiles, index, named
        self.stages: list[np.ndarray] = []  # row prefixes by entity number (entity numbers follow first appearance)
        if len(index) > KEY_PRECHECK_ROWS:
            for ids in (KEY_PRECHECK_IDS, 10 * KEY_PRECHECK_IDS):
                rows = np.flatnonzero(index < ids)
                if not 0 < len(rows) <= len(index) // 2:
                    break
                self.stages.append(rows)
        self._data_of: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}  # (column, stage or -1)

    def _data(self, i: int, stage: int) -> tuple[np.ndarray, np.ndarray]:
        if (i, stage) not in self._data_of:
            p = self.profiles[i]
            rows = None if stage < 0 else self.stages[stage]
            vals = self.table.columns[i].values if rows is None else self.table.columns[i].values[rows]
            named = self.named if rows is None else self.named[rows]
            self._data_of[(i, stage)] = (~missing_mask(vals, p.missing_policy, p.na_tokens) & named,
                                         _stripped_codes(vals))
        return self._data_of[(i, stage)]

    def full(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self._data(i, -1)

    def head_repeats(self, columns: list[int]) -> bool:
        """True when (ID, columns...) repeats among the first IDs' rows (so it is no key); False for a small table."""
        for s, srows in enumerate(self.stages):
            parts = [self._data(i, s) for i in columns]
            rows = np.flatnonzero(np.logical_and.reduce([f for f, _ in parts]))
            if len(rows) < 2:
                continue
            df = pd.DataFrame({"e": self.index[srows][rows], **{f"c{k}": c[rows] for k, (_, c) in enumerate(parts)}})
            if df.duplicated().any():
                return True
        return False

    def varies(self, i: int) -> bool:
        """At least 2 filled rows holding at least 2 distinct values (on the first IDs' rows when they show it)."""
        if self.stages and (i, -1) not in self._data_of:
            f, c = self._data(i, 0)
            if f.sum() >= 2 and c[f].min() != c[f].max():
                return True
        f, c = self.full(i)
        return bool(f.sum() >= 2) and c[f].min() != c[f].max()


def _unique_within_ids(index: np.ndarray, rows: np.ndarray, codes: list[np.ndarray]) -> bool:
    """True when (entity, codes...) is unique over ``rows`` and some ID has several of these rows."""
    ents = index[rows]
    if not np.any(np.bincount(ents) >= 2):
        return False
    if len(rows) > KEY_PRECHECK_ROWS:  # most columns are not keys: reject on the first IDs' rows first
        head = ents < KEY_PRECHECK_IDS  # entity numbers follow first appearance
        if 0 < int(head.sum()) < len(rows):
            df = pd.DataFrame({"e": ents[head], **{f"c{i}": c[rows][head] for i, c in enumerate(codes)}})
            if df.duplicated().any():
                return False
    df = pd.DataFrame({"e": ents, **{f"c{i}": c[rows] for i, c in enumerate(codes)}})
    return not bool(df.duplicated().any())


KEY_PRECHECK_ROWS = 50_000
KEY_PRECHECK_IDS = 2_000


def find_composite_keys(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile) -> list[dict[str, Any]]:
    """AREA D hook: composite-key uniqueness to keep (no duplicate (student, date), D5).

    (primary ID, column) pairs unique over the rows where the column is filled,
    for row-level date / code / label / discrete-number columns of tables with
    multi-row IDs (:func:`_composite_candidate`); then (primary ID, A, B) for two
    such columns that are not keys alone (each column in at most one key; the
    first :data:`MAX_PAIR_CANDIDATES` columns, labels, codes and dates first).
    """
    return _composite_keys(table, profiles, entity, Settings())[0]


# --------------------------------------------------------------------------- inequality rules
@dataclass
class RuleView:
    """Per-row values a declared rule compares (:func:`rule_view`)."""

    vals: np.ndarray  # float64 (numbers) or exact int64 (date-family model units)
    ok: np.ndarray  # filled and readable
    unit: str  # 'number', 'date', 'year_month', 'datetime:<ticks per us>', 'time:<..>', 'duration:<..>'
    kind: str  # 'number' or the date kind
    tpu: int = 1
    # layout=True: what re-rendering a date in its own row's layout needs
    cls: np.ndarray | None = None  # format index
    tz: np.ndarray | None = None  # offset/zone token
    digits: np.ndarray | None = None  # fraction digits as written (-1: none)
    lead: np.ndarray | None = None
    trail: np.ndarray | None = None


def _unit_text(unit: str) -> str:
    return unit.split(":")[0].replace("_", "-")


def rule_view(values: np.ndarray, p: ColumnProfile, layout: bool = False) -> RuleView | None:
    """Per-row values a declared rule compares (D5); None when the column is neither a number nor a date.

    Numbers (INTEGER, INTEGER_YEAR, FLOAT, DECIMAL, and NUMERIC_FORMATTED / NUMERIC_WITH_TOKENS read through the
    column's learned number format, the reading detection and generation use: ``$1,234.50`` is 1234.5, a
    suppression token such as ``*`` or ``<10`` is no number, R5-05) are float64 (unit ``number``). Date-family
    columns are exact int64 in their model unit (AREA A ``detect.dates.view``): days (``date``), months
    (``year_month``), or ticks (microseconds; nanoseconds for ``time_unit`` 'ns') of the UTC instant
    (``datetime:<ticks per us>``), of the time of day (``time:..``) or of the duration (``duration:..``).
    Blanks, NA tokens, sentinels and
    unreadable text are not ``ok``. Two columns are comparable only when their units are equal. ``layout``
    adds, per row, what re-rendering a date in that row's own layout needs (format, offset token, fraction
    digits, whitespace).
    """
    codes, uniq = _factorize(values)
    pres = np.fromiter((not is_missing(u, p.missing_policy, p.na_tokens) for u in uniq.tolist()), dtype=bool,
                       count=len(uniq))
    if p.type in _NUMBER_TYPES:
        unum = np.full(len(uniq), np.nan)
        if p.type in _FORMATTED_NUMBER_TYPES:
            idx = np.flatnonzero(pres)
            if len(idx):
                read = numeric_view_aligned(p, [str(uniq[i]).strip() for i in idx.tolist()])
                if read is not None:
                    unum[idx] = read
        else:
            for i in np.flatnonzero(pres).tolist():
                try:
                    unum[i] = float(uniq[i].strip())
                except ValueError:
                    pass
        x = unum[codes]
        return RuleView(x, np.isfinite(x), "number", "number")
    if not _is_date(p) or not (p.params or {}).get("formats"):
        return None
    present = np.flatnonzero(pres)
    raw = [uniq[i] for i in present.tolist()]
    v = dates_hooks.view(p.params, [u.strip() for u in raw])
    kind, tpu = v.kind, v.tpu
    if kind == "date":
        t = v.day
    elif kind == "year_month":
        t = v.mon
    elif kind == "datetime":
        t = v.day * (US_DAY * tpu) + v.sod
    elif kind == "time":
        t = v.sod
    else:
        t = v.dur
    uval = np.zeros(len(uniq), dtype=np.int64)
    uok = np.zeros(len(uniq), dtype=bool)
    if len(present):
        uval[present] = np.asarray(t, dtype=np.int64)
        uok[present] = v.cls >= 0
    unit = kind if kind in ("date", "year_month") else f"{kind}:{tpu}"
    rv = RuleView(uval[codes], uok[codes], unit, kind, tpu)
    if layout:
        ucls = np.full(len(uniq), -1, dtype=np.int64)
        utz = np.full(len(uniq), "", dtype=object)
        udig = np.full(len(uniq), -1, dtype=np.int64)
        ulead = np.full(len(uniq), "", dtype=object)
        utrail = np.full(len(uniq), "", dtype=object)
        if len(present):
            ucls[present] = v.cls
            utz[present] = [str(x or "") for x in v.tz.tolist()]
            if v.digits is not None:
                udig[present] = v.digits
            ulead[present] = [u[: len(u) - len(u.lstrip())] for u in raw]
            utrail[present] = [u[len(u.rstrip()):] for u in raw]
        rv.cls, rv.tz, rv.digits, rv.lead, rv.trail = ucls[codes], utz[codes], udig[codes], ulead[codes], utrail[codes]
    return rv


def rule_bounds(members: list[tuple[np.ndarray, np.ndarray, bool]], unit_of_row: np.ndarray | None, n_units: int,
                is_int: bool) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, bool]]]:
    """Lower bounds of a high column from every rule on it (R2-11): per unit (a row, or an entity when
    ``unit_of_row`` maps rows to entities), the largest filled value of each rule's low column over the unit's
    rows. ``members``: (low values, filled mask, strict) per rule. Returns (largest bound over the rules,
    unit has any bound, per rule (bound, unit has it, strict))."""
    fill = np.iinfo(np.int64).min if is_int else -np.inf
    dtype = np.int64 if is_int else float
    bound = np.full(n_units, fill, dtype=dtype)
    has_any = np.zeros(n_units, dtype=bool)
    per: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for vals, ok, strict in members:
        if unit_of_row is None:
            bnd = np.where(ok, vals, fill).astype(dtype)
            has = np.asarray(ok, dtype=bool).copy()
        else:
            bnd = np.full(n_units, fill, dtype=dtype)
            np.maximum.at(bnd, unit_of_row[ok], np.asarray(vals[ok], dtype=dtype))
            has = np.zeros(n_units, dtype=bool)
            has[unit_of_row[ok]] = True
        per.append((bnd, has, strict))
        bound = np.maximum(bound, bnd)
        has_any |= has
    return bound, has_any, per


def rule_holds(hi_vals: np.ndarray, per: list[tuple[np.ndarray, np.ndarray, bool]]) -> np.ndarray:
    """Per unit: every rule with a bound there holds for ``hi_vals``."""
    out = np.ones(len(hi_vals), dtype=bool)
    for bnd, has, strict in per:
        out &= ~has | ((hi_vals > bnd) if strict else (hi_vals >= bnd))
    return out


def _numbers(table: RawTable, p: ColumnProfile) -> tuple[np.ndarray, str] | None:
    """Per-row numbers (NaN where missing/unparseable) and the unit (:func:`rule_view`), or None."""
    rv = rule_view(table.columns[p.index].values, p)
    if rv is None:
        return None
    return np.where(rv.ok, rv.vals.astype(float), np.nan), rv.unit


def _rule_text(lo: str, hi: str, strict: bool) -> str:
    def q(name: str) -> str:
        return f'"{name}"' if (name != name.strip() or "<" in name or ">" in name or not name) else name
    return f"{q(lo)} {'<' if strict else '<='} {q(hi)}"


def suggest_inequality_rules(table: RawTable, profiles: list[ColumnProfile], min_cell: int = 10) -> list[dict[str, Any]]:
    """AREA D hook: inequality/threshold rules to *suggest* in the report (Yr_JoinCampus >= KCSE, D5).

    Pairs of numeric (or date) columns where ``A <= B`` holds in every row where
    both are filled (at least ``min_cell`` rows), the columns are not identical,
    and the rule is not implied by the value ranges alone: pairing the two
    columns' values at random would break it in at least 5% of the pairs
    (hours <= 50 vs scores >= 40 is scale, not a rule). Each suggestion carries
    the config text that applies it, so only pairs a declared rule can be built
    for are suggested: the same unit (numbers, or dates of the same kind: date,
    datetime, time, duration, year-month) and the same role (both constant
    within the ID, or both per row).
    Suggestions only; they apply only when declared under ``rules:``.
    """
    cand: list[tuple[ColumnProfile, np.ndarray, str]] = []
    for p in profiles:
        if not _generated(p) or p.params.get("id_candidate"):
            continue
        r = _numbers(table, p)
        if r is not None:
            cand.append((p, r[0], r[1]))
    out: list[dict[str, Any]] = []
    for i, (pa, va, ka) in enumerate(cand):
        for pb, vb, kb in cand[i + 1:]:
            if ka != kb or pa.role != pb.role:
                continue
            both = np.isfinite(va) & np.isfinite(vb)
            m = int(both.sum())
            if m < max(min_cell, 2):
                continue
            d = vb[both] - va[both]
            if np.all(d == 0):
                continue
            if np.all(d >= 0):
                lo, hi, lo_v, hi_v, gaps = pa, pb, va[both], vb[both], d
            elif np.all(d <= 0):
                lo, hi, lo_v, hi_v, gaps = pb, pa, vb[both], va[both], -d
            else:
                continue
            strict = bool(np.all(gaps > 0))
            if _pairing_share(lo_v, hi_v, strict) >= RANGE_IMPLIED:
                continue  # true by the value ranges alone (scale), not a rule
            rule = _rule_text(lo.name, hi.name, strict)
            config = 'rules: ["' + rule + '"]' if '"' not in rule else "rules: ['" + rule + "']"
            out.append({"rule": rule, "rows": m, "equal_rows": int(np.sum(gaps == 0)),
                        "kind": "number" if ka == "number" else "date", "config": config})
            if len(out) >= MAX_SUGGESTIONS:
                return out
    return out


def _pairing_share(lo: np.ndarray, hi: np.ndarray, strict: bool) -> float:
    """Share of all (lo, hi) value pairings that satisfy lo <= hi (lo < hi when strict)."""
    hs = np.sort(hi)
    if strict:
        ok = len(hs) - np.searchsorted(hs, lo, side="right")
    else:
        ok = len(hs) - np.searchsorted(hs, lo, side="left")
    return float(ok.sum()) / (len(lo) * len(hs))


def parse_rules(raw: Any) -> list[dict[str, Any]]:
    """Validate ``rules:`` from ``--config`` into ``{'text', 'lo', 'hi', 'strict', 'file'}`` dicts.

    Items are strings ``"A <= B"`` (also ``<``, ``>=``, ``>``) or mappings
    ``{rule: "A <= B", file: "path.csv"}``. Column names may be quoted.
    Raises :class:`UsageError` (exit 2) on anything else.
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict[str, Any]] = []
    for item in items:
        file = None
        if isinstance(item, dict):
            extra = set(item) - {"rule", "file"}
            if extra or "rule" not in item:
                raise UsageError(f"bad rule {item!r}: use a string 'A <= B' or {{rule: 'A <= B', file: 'path.csv'}}")
            text, file = item["rule"], item.get("file")
        else:
            text = item
        if not isinstance(text, str):
            raise UsageError(f"bad rule {item!r}: expected text such as 'start_year <= grad_year'")
        m = _RULE_RE.match(text)
        if not m:
            raise UsageError(f"bad rule {text!r}: expected 'A <= B', 'A < B', 'A >= B' or 'A > B'")
        a, op, b = m.group(1), m.group(2), m.group(3)
        a, b = _unquote(a), _unquote(b)
        if not a or not b or a == b:
            raise UsageError(f"bad rule {text!r}: name two different columns")
        op = {"=<": "<=", "=>": ">="}.get(op, op)
        lo, hi = (a, b) if op in ("<=", "<") else (b, a)
        out.append({"text": text.strip(), "lo": lo, "hi": hi, "strict": op in ("<", ">"),
                    "file": str(PurePosixPath(str(file).replace("\\", "/").strip("/"))) if file else None})
    return out


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        return s[1:-1]
    return s


def _fractional(p: ColumnProfile) -> bool:
    """Numbers written with decimals (a float, a decimal with a scale): whole-number gaps are no evidence of an
    integer rule there."""
    if p.type in _FORMATTED_NUMBER_TYPES:  # '$12.50' (cents generated); '$100.00' is integer-valued
        return not p.params.get("integer_valued") and int(p.params.get("decimals") or 0) > 0
    return p.type is T.FLOAT or (p.type is T.DECIMAL and int(p.params.get("scale") or 0) > 0)


def _gap_data(hv: RuleView, members: list[tuple[RuleView, bool]], phi: ColumnProfile, lows: list[ColumnProfile],
              index: np.ndarray | None) -> dict[str, Any]:
    """Real gaps between a high column and the largest of its low columns (:func:`rule_bounds`), per row - or
    per entity (its first row's value) when the high column is constant within the ID (D4), since it is then
    drawn once per synthetic entity. 'rows' = units with the high and a low value, 'held' = share of them
    where every rule holds, 'gaps' (sorted, from those units), 'integer'. The high column keeps its own
    generated values (its exact min/max included, D14); the gaps only steer where they go (R3-06)."""
    n = len(hv.vals)
    per_entity = phi.role == "entity" and index is not None and len(index) == n and n > 0
    if per_entity:
        assert index is not None
        n_units = int(index.max()) + 1
        first = np.full(n_units, -1, dtype=np.int64)
        first[index[::-1]] = np.arange(n - 1, -1, -1)
        has_row = first >= 0
        hi_v = hv.vals[np.maximum(first, 0)]
        hi_ok = hv.ok[np.maximum(first, 0)] & has_row
        unit_of_row = index
    else:
        n_units, hi_v, hi_ok, unit_of_row = n, hv.vals, hv.ok, None
    is_int = hv.vals.dtype.kind == "i"
    bound, has_any, per = rule_bounds([(m.vals, m.ok, s) for m, s in members], unit_of_row, n_units, is_int)
    valid = hi_ok & has_any
    holds = valid & rule_holds(hi_v, per)
    gaps = np.sort((hi_v - bound)[holds])
    d: dict[str, Any] = {"rows": int(valid.sum()), "held": float(holds[valid].mean()) if valid.any() else 0.0,
                         "gaps": gaps}
    if per_entity:
        d["per_entity"] = True
    d["integer"] = is_int or (bool(len(gaps)) and bool(np.all(np.round(gaps) == gaps))
                              and not any(_fractional(q) for q in [phi, *lows]))
    return d


def _rule_data(table: RawTable, profiles: list[ColumnProfile], settings: Settings,
               index: np.ndarray | None = None, file_rel: str | None = None
               ) -> tuple[dict[tuple, dict[str, Any]], dict[int, dict[str, Any]]]:
    """PRIVATE: (per declared rule whose two columns exist in this table, per high column).

    Per rule (key ``(lo, hi, strict)``): the column indexes, the real-data check (:func:`_gap_data`) and,
    when no rule can be built, ``unsupported`` (the run FAILs the file, see :func:`_resolve_rules`) or
    ``not_applicable`` (a column is ``--exclude``-d or ``--null``-ed: nothing to enforce). A rule compares two
    generated numbers or two dates of the same kind (:func:`rule_view`) with the same role.

    Per high column (``targets``): every supported rule on it that applies to this file (``file:`` equal to
    ``file_rel``, or none) in ``members`` as (low column, strict, rule key), and the real gaps between the
    column and the largest of its low columns, so one draw satisfies every rule on the column together
    (R2-11); ``unsupported`` when no real row satisfies them together.
    """
    try:
        rules = parse_rules(settings.rules)
    except UsageError:
        return {}, {}  # reported by cross_file_structure (exit 2)
    by_name: dict[str, ColumnProfile] = {}
    for p in profiles:
        by_name.setdefault(p.name, p)
    views: dict[int, RuleView | None] = {}

    def vw(p: ColumnProfile) -> RuleView | None:
        if p.index not in views:
            views[p.index] = rule_view(table.columns[p.index].values, p)
        return views[p.index]

    files: dict[tuple, set] = {}
    for r in rules:
        if r["lo"] in by_name and r["hi"] in by_name:
            files.setdefault((r["lo"], r["hi"], r["strict"]), set()).add(r["file"])
    out: dict[tuple, dict[str, Any]] = {}
    members: dict[int, list[tuple]] = {}
    for key, fset in files.items():
        plo, phi = by_name[key[0]], by_name[key[1]]
        d: dict[str, Any] = {"lo": plo.index, "hi": phi.index, "strict": key[2]}
        out[key] = d
        if {plo.role, phi.role} & {"excluded", "null"}:
            d["not_applicable"] = "a column is not generated (--exclude / --null)"
            continue
        a, b = vw(plo), vw(phi)
        if not (_generated(plo) and _generated(phi)) or a is None or b is None:
            d["unsupported"] = "both columns must be generated numbers (integer/decimal) or both dates"
            continue
        if a.unit != b.unit:
            d["unsupported"] = (f"the columns hold different kinds of values ({_unit_text(a.unit)} and "
                                f"{_unit_text(b.unit)})")
            continue
        if plo.role != phi.role:
            d["unsupported"] = "one column is constant within the ID and the other varies per row"
            continue
        d.update(_gap_data(b, [(a, key[2])], phi, [plo], index))
        if not len(d["gaps"]):
            d["unsupported"] = "no real row satisfies the rule"
            continue
        if file_rel is None or None in fset or file_rel in fset:
            members.setdefault(phi.index, []).append(key)
    targets: dict[int, dict[str, Any]] = {}
    for hi, keys in members.items():
        if len(keys) == 1:
            d = out[keys[0]]
            t = {k: d[k] for k in ("rows", "held", "gaps", "integer", "per_entity") if k in d}
        else:
            phi = profiles[hi]
            lows = [profiles[out[k]["lo"]] for k in keys]
            t = _gap_data(vw(phi), [(vw(q), k[2]) for q, k in zip(lows, keys)], phi, lows, index)  # type: ignore[misc]
            if not len(t["gaps"]):
                t["unsupported"] = "no real row satisfies all the rules on this column together"
        t["hi"] = hi
        t["members"] = [(out[k]["lo"], k[2], k) for k in keys]
        targets[hi] = t
    return out, targets


# --------------------------------------------------------------------------- panels (D16)
_SUPPRESSION = re.compile(r"^(\*+|<\s*\d+%?|>\s*\d+%?|n\s*<\s*\d+|--|DS|N/?A\*?)$", re.IGNORECASE)
_COUNT_WORDS = ("n", "count", "cnt", "total", "number", "num", "pct", "percent", "rate", "avg", "mean", "tested",
                "enrolled", "proficient")


def _panels(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile,
            index: np.ndarray | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if index is None and entity.multi_row_ids:
        index = _entity_index(table, entity)
    if index is not None and entity.primary is not None:
        prim = profiles[entity.primary].name
        multi = np.flatnonzero(np.bincount(index, minlength=entity.n_entities) > 1)
        in_multi = np.isin(index, multi)
        for p in profiles:
            if p.role != "row":
                continue
            h = header_hints(p.name)
            datey = _is_date(p) or bool(h.date)
            yeary = p.type in (T.SCHOOL_YEAR, T.INTEGER_YEAR) or bool(h.school_year) or bool(h.year)
            if not (datey or yeary):
                continue
            df = pd.DataFrame({"e": index[in_multi], "v": pd.Series(table.columns[p.index].values[in_multi],
                                                                     dtype=object)})
            varying = float((df.groupby("e", sort=False)["v"].nunique(dropna=False) > 1).mean()) if len(df) else 0.0
            if varying < 0.5:
                continue
            kind = "multi_year" if yeary and not datey else "panel"
            changing = _slowly_changing(table, profiles, index, in_multi)
            text = (f"({prim}, {p.name}) {'multi-year snapshots' if kind == 'multi_year' else 'panel'}: "
                    f"{len(multi)} IDs have several {p.name} values; time structure (spells, progression"
                    + (f", attributes that change within an ID: {', '.join(changing)}" if changing else "")
                    + ") is NOT preserved: rows are generated independently with composite-key uniqueness only "
                    "(D16)")
            entry = {"kind": kind, "columns": [prim, p.name], "ids_with_several_rows": int(len(multi)),
                     "text": text}
            if changing:
                entry["changing_attributes"] = changing
            out.append(entry)
            p.messages.append(Message("warning", "panel_detected" if kind == "panel" else "multi_year_snapshots",
                                      text, column=p.name))
            break
    if not entity.id_columns:
        group = [p for p in profiles if p.role == "row" and p.type in (T.ORG_CODE, T.FACTOR, T.ORDINAL,
                                                                          T.SCHOOL_YEAR, T.INTEGER_YEAR)]
        in_group = {p.index for p in group}
        counts = []
        suppressed = []
        for p in profiles:
            if p.role != "row" or p.index in in_group:
                continue
            toks = [v.strip() for v in (p.model.nonmissing_counts if p.model else {}) if isinstance(v, str)]
            numeric = sum(1 for v in toks if _is_number(v))
            supp = sum(1 for v in toks if _SUPPRESSION.match(v))
            words = header_hints(p.name)
            if toks and (numeric + supp) / len(toks) >= 0.9 and numeric:
                if supp:
                    suppressed.append(p.name)
                if supp or any(t in _COUNT_WORDS for t in _tokens(p.name)) or words.code_count:
                    counts.append(p.name)
        if group and counts and table.n_rows >= 2:
            cols = [table.columns[p.index].values for p in group]
            df = pd.DataFrame({str(i): pd.Series(c, dtype=object) for i, c in enumerate(cols)})
            if not df.duplicated().any():
                text = (f"aggregate-shaped table: one row per ({', '.join(p.name for p in group)}) with counts/rates "
                        f"({', '.join(counts)})" + (f" and suppression tokens ({', '.join(suppressed)})" if suppressed
                                                     else "")
                        + "; totals, subgroup <= total and suppression consistency are NOT preserved (D16)")
                out.append({"kind": "aggregate", "columns": [p.name for p in group] + counts, "text": text})
                group[0].messages.append(Message("warning", "aggregate_table", text, column=group[0].name))
    return out


def _tokens(name: str) -> tuple[str, ...]:
    from ..hints import tokens

    return tokens(name)


def _is_number(v: str) -> bool:
    try:
        float(v.replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


def _slowly_changing(table: RawTable, profiles: list[ColumnProfile], index: np.ndarray,
                     in_multi: np.ndarray) -> list[str]:
    """Row-level label columns constant within >= 50% (but < 99%) of the multi-row IDs."""
    out = []
    for p in profiles:
        if p.role != "row" or not (p.type in _DET_LEVEL_TYPES or p.type in (T.INTEGER,)):
            continue
        if p.type is T.SCHOOL_YEAR or p.type is T.INTEGER_YEAR:
            continue
        df = pd.DataFrame({"e": index[in_multi], "v": pd.Series(table.columns[p.index].values[in_multi],
                                                                 dtype=object)})
        if not len(df):
            continue
        share = float((df.groupby("e", sort=False)["v"].nunique(dropna=False) <= 1).mean())
        if 0.5 <= share < 0.99:
            out.append(p.name)
    return out


def detect_panels(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile) -> list[dict[str, Any]]:
    """AREA D hook: panels, slowly-changing attributes and aggregate shapes to warn about (D16).

    * (ID, date) panels and multi-year snapshots: a row-level date / year /
      school-year column with several values within >= 50% of the multi-row IDs;
      attributes constant within 50-99% of those IDs are listed as changing
      across years;
    * aggregate-shaped tables: no ID column, group columns unique per row, and
      count/rate columns (count words or suppression tokens such as ``*``).

    Each finding is also a warning on the column (counts and names only).
    """
    return _panels(table, profiles, entity)


# --------------------------------------------------------------------------- sort order and duplicates (section 6)
def order_values(values: np.ndarray, p: ColumnProfile, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(sortable float keys, missing mask) for ``mode`` 'numeric' or 'text'; None when not available.

    Numeric: numbers, date ordinals (AREA A hook), formatted numbers through their number format (AREA B
    reading) or plain digit strings. Text:
    the rank of the stripped string in Python order.
    """
    codes, uniq = _factorize(values)
    umiss = np.fromiter((is_missing(u, p.missing_policy, p.na_tokens) for u in uniq), dtype=bool, count=len(uniq))
    present = np.flatnonzero(~umiss)
    stripped = [uniq[i].strip() for i in present.tolist()]
    ukeys = np.zeros(len(uniq), dtype=float)
    if mode == "numeric":
        if _is_date(p):
            nums = dates_hooks.date_numbers(stripped, p)
            if nums is None:
                return None
            arr = np.asarray(nums, dtype=float)
        elif p.type in _FORMATTED_NUMBER_TYPES:  # '$1,234.50' in its number format (R5-05; a token: no order)
            read = numeric_view_aligned(p, stripped)
            if read is None:
                return None
            arr = np.asarray(read, dtype=float)
        else:
            try:
                arr = np.asarray([float(v) for v in stripped], dtype=float)
            except ValueError:
                return None
        if not np.all(np.isfinite(arr)):
            return None
        ukeys[present] = arr
    elif stripped:
        _, inv = np.unique(np.asarray(stripped, dtype=object).astype(str), return_inverse=True)
        ukeys[present] = inv
    return ukeys[codes], umiss[codes]


def _could_be_sorted(keys: np.ndarray, miss: np.ndarray, run_id: np.ndarray) -> bool:
    """False when the rows already contradict any sort direction / missing placement."""
    same = run_id[1:] == run_id[:-1]
    both = same & ~miss[:-1] & ~miss[1:]
    diff = keys[1:] - keys[:-1]
    if not (np.all(diff[both] >= 0) or np.all(diff[both] <= 0)):
        return False
    return not (np.any(same & miss[:-1] & ~miss[1:]) and np.any(same & ~miss[:-1] & miss[1:]))


def _check_key(keys: np.ndarray, miss: np.ndarray, run_id: np.ndarray,
               need_runs: int = 1) -> tuple[str, str | None] | None:
    """(direction, missing placement) if the column is sorted within every run, else None.

    ``need_runs``: how many runs must show a strict change (evidence against a
    column that is sorted within a few short runs by chance).
    """
    same = run_id[1:] == run_id[:-1]
    a_m, b_m = miss[:-1], miss[1:]
    both = same & ~a_m & ~b_m
    diff = keys[1:] - keys[:-1]
    up = bool(np.all(diff[both] >= 0))
    down = bool(np.all(diff[both] <= 0))
    if not (up or down) or (up and down):
        return None  # unsorted, or constant within every run (not a key)
    strict = both & (diff != 0)
    if len(np.unique(run_id[1:][strict])) < need_runs:
        return None
    to_present = bool(np.any(same & a_m & ~b_m))
    to_missing = bool(np.any(same & ~a_m & b_m))
    if to_present and to_missing:
        return None
    na = "first" if to_present else ("last" if to_missing else None)
    return ("asc" if up else "desc"), na


def _sort_order(table: RawTable, profiles: list[ColumnProfile], settings: Settings
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    n = table.n_rows
    if n < 3:
        return {}, {}
    run_id = np.zeros(n, dtype=np.int64)
    keys_found: list[tuple[int, str, str | None, str]] = []
    used: set[int] = set()
    cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray] | None] = {}
    pre_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray] | None] = {}
    m = min(n, SORT_PREFIX)
    for _ in range(MAX_SORT_KEYS):
        if len(np.unique(run_id)) == n:
            break
        hit = None
        for p in profiles:
            if p.index in used or p.role in ("excluded", "null"):
                continue
            for mode in ("numeric", "text"):
                ck = (p.index, mode)
                if ck not in cache:
                    if n > m:  # cheap rejection on the first rows (most columns are not sorted)
                        if ck not in pre_cache:
                            pre_cache[ck] = order_values(table.columns[p.index].values[:m], p, mode)
                        pre = pre_cache[ck]
                        if pre is None or not _could_be_sorted(pre[0], pre[1], run_id[:m]):
                            continue
                    cache[ck] = order_values(table.columns[p.index].values, p, mode)
                ov = cache[ck]
                if ov is None:
                    continue
                res = _check_key(ov[0], ov[1], run_id, need_runs=1 if not keys_found else 2)
                if res is not None:
                    hit = (p.index, res[0], res[1], mode)
                    break
            if hit:
                break
        if hit is None:
            break
        keys_found.append(hit)
        used.add(hit[0])
        kv, km = cache[(hit[0], hit[3])]  # type: ignore[misc]
        boundary = np.zeros(n, dtype=bool)
        boundary[1:] = (kv[1:] != kv[:-1]) | (km[1:] != km[:-1]) | (run_id[1:] != run_id[:-1])
        run_id = np.cumsum(boundary)
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    status = _status(settings, "sort_order", "reproduced")
    if keys_found:
        public["keys"] = [{"column": profiles[i].name, "direction": d, "missing": na, "order": mode}
                          for i, d, na, mode in keys_found]
        public["status"] = status
        private["keys"] = keys_found
        return public, private
    blocks = []
    for p in profiles:
        if p.role not in ("row", "entity"):
            continue
        miss = missing_mask(table.columns[p.index].values, p.missing_policy, p.na_tokens)
        m = int(miss.sum())
        if 2 <= m < n and bool(miss[:m].all()):
            blocks.append((p.index, m))
    if blocks:
        blocks.sort(key=lambda b: (-b[1], b[0]))
        public["leading_missing"] = [{"column": profiles[i].name, "rows": m} for i, m in blocks]
        public["status"] = status
        private["blocks"] = blocks
    return public, private


def _duplicate_rows(table: RawTable, profiles: list[ColumnProfile], settings: Settings) -> dict[str, Any]:
    """Real duplicate rows over the columns the output reproduces (not ``--exclude`` / ``--null`` ones)."""
    n = table.n_rows
    cols = [p.index for p in profiles if p.role not in ("excluded", "null")]
    if n < 2 or not cols:
        return {}
    encoded = {}
    for i in cols:
        values = table.columns[i].values
        codes, uniques = codes_of(values)
        # Object duplicate detection and string factorization disagree on NUL.
        # Also retain object behavior for mixed native nulls such as None/NaN.
        encoded[str(i)] = (codes if not has_nul(values)
                           and all(isinstance(u, str) or u is None for u in uniques)
                           else pd.Series(values, dtype=object))
    df = pd.DataFrame(encoded)
    dups = int(df.duplicated().sum())
    if not dups:
        return {"rows": 0, "rate": 0.0}
    return {"rows": dups, "rate": round(dups / n, 6), "status": _status(settings, "duplicate_rows", "reproduced")}


# --------------------------------------------------------------------------- swap groups
def swap_groups(structure: StructureProfile, profiles: list[ColumnProfile]) -> list[list[int]]:
    """AREA D hook: column groups that privacy swap-redraw must move together.

    An applied FD's determinant with its dependents, a template's column with its
    inputs, and both columns of a declared rule: swapping one of them alone would
    break the relationship. Overlapping groups are merged.
    """
    groups: list[set[int]] = []
    priv = structure.private
    by_det: dict[int, set[int]] = {}
    for fd in priv.get("fds", []):
        if not fd.get("overridden"):
            by_det.setdefault(fd["det"], {fd["det"]}).add(fd["dep"])
    groups.extend(by_det.values())
    for t in priv.get("templates", []):
        groups.append({t["target"]} | {p[1] for p in t["parts"] if p[0] == "f"})
    for key in priv.get("declared", []):
        d = priv.get("rule_data", {}).get(key)
        if d:
            groups.append({d["lo"], d["hi"]})
    merged: list[set[int]] = []
    for g in groups:
        g = set(g)
        keep = []
        for m in merged:
            if m & g:
                g |= m
            else:
                keep.append(m)
        merged = keep + [g]
    return [sorted(g) for g in merged if len(g) > 1]


# --------------------------------------------------------------------------- run level
def table_ref(rel: str, table: str | None, col: str) -> str:
    """``path::COL`` or ``book.xlsx::Sheet::COL``: the column-flag reference of one column."""
    return f"{rel}::{table}::{col}" if table else f"{rel}::{col}"


_table_ref = table_ref


_RULE_FIX = {
    "cycle": "remove one of the declared rules that form the cycle from rules: in --config",
    "default": "remove the rule from rules: in --config, or give both columns a number or date type of the same "
               "kind (--force-type COL=TYPE)",
    "data": "remove the rule from rules: in --config (the source data does not satisfy it)",
}


def _fail_file(fr: Any, reason: str, fix: str) -> None:
    if hasattr(fr, "fail"):
        fr.fail(reason, fix, "rule_unusable")
    else:  # minimal stand-ins (tests)
        fr.failed = True


def _resolve_rules(file_results: list[Any], settings: Settings) -> list[Message]:
    """Resolve ``rules:`` per table: statuses, the high columns to build (``declared_targets``, in dependency
    order) and the swap groups.

    A declared rule that names two columns of a table is either built by construction or the file FAILs
    (D19) with the fix: rules between columns that are not two numbers or two dates of the same kind, between a
    within-ID constant and a per-row column, rules the source never satisfies, and rules that form a cycle
    (R2-16: never a successful run that ignores a requested rule). A rule on an ``--exclude``-d / ``--null``-ed
    column has nothing to enforce (warning). Every rule on one high column is solved together (R2-11).
    """
    rules = parse_rules(settings.rules)  # UsageError -> exit 2 before anything is written
    msgs: list[Message] = []
    used = [False] * len(rules)
    for fr in file_results:
        if fr.failed:
            continue
        for tp in fr.tables:
            sp = tp.structure
            sp.declared_rules = []
            names = {p.name for p in tp.columns}
            data = sp.private.get("rule_data", {})
            all_targets = sp.private.get("rule_targets", {})
            entries: list[tuple[dict[str, Any], tuple, str]] = []
            for k, r in enumerate(rules):
                if r["file"] is not None and r["file"] != fr.rel_path:
                    continue
                if r["lo"] not in names or r["hi"] not in names:
                    continue
                used[k] = True
                entry: dict[str, Any] = {"rule": r["text"]}
                sp.declared_rules.append(entry)
                entries.append((entry, (r["lo"], r["hi"], r["strict"]), r["text"]))
            keys = list(dict.fromkeys(key for _, key, _ in entries))
            problems: dict[tuple, tuple[str, str, str]] = {}  # key -> (kind, reason, fix)
            for key in keys:
                d = data.get(key)
                if d is None:
                    problems[key] = ("fail", "the columns were not profiled", _RULE_FIX["default"])
                elif d.get("not_applicable"):
                    problems[key] = ("skip", d["not_applicable"], "")
                elif d.get("unsupported"):
                    fix = _RULE_FIX["data"] if "satisf" in d["unsupported"] else _RULE_FIX["default"]
                    problems[key] = ("fail", d["unsupported"], fix)
            ok_keys = [k for k in keys if k not in problems]
            targets: list[dict[str, Any]] = []
            for hi in dict.fromkeys(data[k]["hi"] for k in ok_keys):
                mine = [k for k in ok_keys if data[k]["hi"] == hi]
                t = all_targets.get(hi)
                if t is None or sorted(m[2] for m in t["members"]) != sorted(mine):
                    for k in mine:
                        problems[k] = ("fail", "its profile data does not match this file", _RULE_FIX["default"])
                elif t.get("unsupported"):
                    for k in mine:
                        problems[k] = ("fail", t["unsupported"], _RULE_FIX["data"])
                else:
                    targets.append(t)
            order, cyclic = _order_targets(targets)
            for t in cyclic:
                for m in t["members"]:
                    problems[m[2]] = ("fail", "the declared rules form a cycle", _RULE_FIX["cycle"])
            applied = [m[2] for t in order for m in t["members"]]
            partly: set[tuple] = set()
            for entry, key, text in entries:
                prob = problems.get(key)
                if prob is None:
                    d = data[key]
                    entry["status"] = "applied"
                    entry["real_rows_holding"] = round(d["held"], 6)
                    if d["held"] < 1.0 and key not in partly:
                        partly.add(key)
                        tp.messages.append(Message(
                            "warning", "rule_partly_true",
                            f"declared rule {text!r} holds in {d['held']:.1%} of the real rows where both "
                            "columns are filled; it is applied to every synthetic row"))
                    continue
                kind, why, fix = prob
                entry["status"] = f"not applied: {why}"
                if kind == "skip":
                    tp.messages.append(Message("warning", "rule_not_applied",
                                               f"declared rule {text!r} is not applied: {why}"))
                else:
                    _fail_file(fr, f"declared rule {text!r} cannot be applied: {why}", fix)
            sp.private["declared"] = applied
            sp.private["declared_targets"] = [t["hi"] for t in order]
            sp.private["declared_entries"] = [(i, key) for i, (_, key, _) in enumerate(entries) if key in applied]
            # a declared rule sets its high column: a functional dependency rebuilding that column would undo it
            highs = set(sp.private["declared_targets"])
            for fd, pub in zip(sp.private.get("fds", []), sp.functional_dependencies):
                fd["overridden"] = fd["dep"] in highs
                pub["status"] = ("not applied: the column is set by a declared rule" if fd["overridden"]
                                 else _status(settings, "functional_dependencies", "applied"))
                if fd["overridden"]:
                    tp.messages.append(Message("info", "fd_overridden_by_rule",
                                               f"functional dependency {pub['text']} is not applied: a declared rule "
                                               "sets that column", column=pub["dependent"]))
            sp.swap_groups = swap_groups(sp, tp.columns)
    for k, r in enumerate(rules):
        if not used[k]:
            msgs.append(Message("warning", "rule_unused", f"declared rule {r['text']!r} matched no table "
                                "(both column names must exist; 'file' must be the path relative to the input)"))
    return msgs


def _order_targets(targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(targets in build order, targets in a cycle): a high column whose rules read another target's high
    column is built after it (A <= B <= C)."""
    pending = list(targets)
    out: list[dict[str, Any]] = []
    while pending:
        highs = {t["hi"] for t in pending}
        ready = [t for t in pending if not any(m[0] in highs for m in t["members"])]
        if not ready:
            return out, pending
        out.extend(ready)
        pending = [t for t in pending if not any(t is r for r in ready)]
    return out, []


def _id_members(file_results: list[Any]) -> list[tuple[Any, Any, ColumnProfile]]:
    out = []
    for fr in file_results:
        if fr.failed:
            continue
        for tp in fr.tables:
            for i in tp.entity.id_columns:
                out.append((fr, tp, tp.columns[i]))
    return out


def _contained(keys: set[str], others: list[set[str]], share: float) -> float | None:
    """Share of ``keys`` found in any of ``others`` when it is >= ``share`` (None otherwise; stops at the
    first miss too many, so unrelated key sets cost about 1% of a pass)."""
    n = len(keys)
    if not n or not others:
        return None
    limit = n * (1.0 - share) + 1e-9
    misses = 0
    for k in keys:
        if not any(k in o for o in others):
            misses += 1
            if misses > limit:
                return None
    return (n - misses) / n


def _key_links(members: list[tuple[Any, Any, ColumnProfile]], base: dict[int, str],
               find: Callable[[str], str]) -> list[tuple[str, str]]:
    """Namespace pairs to merge: content-based ID key discovery across tables (D5, D8; section 10).

    The ID columns are grouped by their provisional namespace (name, identical aliases). A column whose D8
    keys are >= :data:`LINK_CONTAINMENT` contained in the keys of another group that has a column in another
    table (file or sheet) is linked to it: the same real ID then gets the same synthetic ID in both, so the
    join is kept even when the headers differ (``student_id`` / ``StudentNumber``). Merging keeps every real
    equality between the two columns (a coincidental one too, as it is in the real data) and never makes a
    synthetic ID real. Row-number namespaces (identity, D9) are never linked either way: their real 1..n
    values are kept, and a person ID linked to them would be written as it is. A column treated as an ID
    because it holds another ID column's keys (source ``key-ref``, :func:`_key_reference_columns`) is linked
    on that same evidence (:func:`_key_ref_share`, so its orphan references stay orphans and the rest still
    join), in its own table too. Each linked column gets an info message (names and the share only).
    """
    from ..generate.ids import IdRegistry, namespace_is_identity

    groups: dict[str, list[tuple[Any, Any, ColumnProfile]]] = {}
    for m in members:
        groups.setdefault(find(base[id(m[2])]), []).append(m)
    if len(groups) < 2:
        return []
    reg = IdRegistry()
    col_keys: dict[int, set[str]] = {}
    by_table: dict[str, dict[tuple[str, str | None], set[str]]] = {}  # group -> table -> its ID keys
    for root, ms in groups.items():
        for fr, tp, p in ms:
            raw = tp.private.id_raw[p.index]
            reg.register(root, fr.rel_path, tp.name, p.index, p.name, raw)
            col_keys[id(p)] = {k for k in entity_keys(codes_of(raw)[1]) if k is not None}
            by_table.setdefault(root, {}).setdefault((fr.rel_path, tp.name), set()).update(col_keys[id(p)])
    identity = {root: namespace_is_identity(reg.namespaces[root]) for root in groups}
    all_keys: dict[str, tuple[set[str], np.ndarray]] = {}

    def keys_of(root: str) -> tuple[set[str], np.ndarray]:
        if root not in all_keys:
            ks = set().union(*by_table[root].values())
            ints = np.asarray(sorted(int(k) for k in ks if k.isascii() and k.isdigit() and len(k) <= 18),
                              dtype=np.int64)
            all_keys[root] = (ks, ints)
        return all_keys[root]

    links: list[tuple[str, str]] = []
    for root in sorted(groups):
        if identity[root]:
            continue
        for fr, tp, p in groups[root]:
            found = []
            # a column treated as an ID because it holds an ID column's keys (source 'key-ref', R2-01), or because its
            # header and ID-like values say so (K-02: 'Admission No', 'Student_SK'), links to an ID column on the
            # key-ref evidence (orphan references included), in its own table too (mentor_ref -> student_id); other
            # same-table pairs are aliases/nesting (find_aliases_and_nesting)
            any_table = p.source == "key-ref" or bool(p.params.get(WEAK_ID_KEY))
            for other in sorted(groups):
                if other == root or identity[other]:
                    continue
                if any_table:
                    share = _key_ref_share(col_keys[id(p)], *keys_of(other), False)
                else:
                    others = [ks for t, ks in by_table[other].items() if t != (fr.rel_path, tp.name)]
                    share = _contained(col_keys[id(p)], others, LINK_CONTAINMENT)
                if share is None:
                    continue
                links.append((root, other))
                cols = sorted({_table_ref(ofr.rel_path, otp.name, op.name) for ofr, otp, op in groups[other]
                               if any_table or (ofr.rel_path, otp.name) != (fr.rel_path, tp.name)})
                found.append(f"{', '.join(cols)} ({share:.1%} of its keys found)")
            if found:
                p.messages.append(Message(
                    "info", "id_key_link",
                    f"ID column {p.name!r} shares one synthetic ID mapping with ID column(s) {'; '.join(found)}: "
                    "the same real ID gets the same synthetic ID, so joins between them are kept (D5/D8 key "
                    "discovery; config structure: {disable: [cross_file_keys]} turns it off)",
                    column=p.name))
    return links


def _assign_namespaces(file_results: list[Any], settings: Settings) -> dict[str, list[tuple[Any, Any, ColumnProfile]]]:
    members = _id_members(file_results)
    base: dict[int, str] = {}
    for fr, tp, p in members:
        p.params.pop("_namespace", None)
        p.messages[:] = [m for m in p.messages if m.code != "id_key_link"]  # recomputed below
        base[id(p)] = base_namespace(fr.rel_path, tp.name, p)
    if _disabled(settings, "cross_file_keys"):
        for fr, tp, p in members:
            b = base[id(p)]
            if normalize(p.name) not in GENERIC_ID_NAMES:
                p.params["_namespace"] = f"{b}@{fr.rel_path}" + (f"#{tp.name}" if tp.name else "")
    else:
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            a, b = find(x), find(y)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for fr, tp, p in members:
            for j in p.params.get("_same_values_as", []):
                union(base[id(p)], base_namespace(fr.rel_path, tp.name, tp.columns[j]))
        # D5/D8 key discovery: an ID column whose keys are >= 99% contained in the keys of an ID column of
        # another table (file or sheet) shares its synthetic IDs, whatever the headers (student_id / StudentNumber)
        for a, b in _key_links(members, base, find):
            union(a, b)
        for fr, tp, p in members:
            root = find(base[id(p)])
            if root != base[id(p)]:
                p.params["_namespace"] = root
    groups: dict[str, list[tuple[Any, Any, ColumnProfile]]] = {}
    for fr, tp, p in members:
        key = p.params.get("_namespace") or base[id(p)]
        groups.setdefault(key, []).append((fr, tp, p))
    return groups


def _coverage(groups: dict[str, list[tuple[Any, Any, ColumnProfile]]], settings: Settings) -> None:
    disabled = _disabled(settings, "cross_file_keys")
    for members in groups.values():
        for _, tp, _ in members:
            tp.structure.cross_file_keys = []
    for key, members in groups.items():
        if len({(fr.rel_path, tp.name) for fr, tp, _ in members}) < 2:
            continue
        info = []
        for fr, tp, p in members:
            keys = [k for k in entity_keys(tp.private.id_raw[p.index]) if k is not None]
            ks = set(keys)
            info.append((fr, tp, p, keys, ks, len(ks) == len(keys) and len(keys) > 0))
        parents = [m for m in info if m[5]]
        if not parents:
            continue
        for fr, tp, p, keys, ks, _ in info:
            if not keys:
                continue
            others = [m for m in parents if (m[0].rel_path, m[1].name) != (fr.rel_path, tp.name)]
            if not others:
                continue
            best = max(others, key=lambda m: (len(ks & m[4]), len(m[4])))
            pfr, ptp, pp, _, pkeys, _ = best
            common = len(ks & pkeys)
            if not common:
                continue  # unrelated files that happen to share a column name
            contained = common / len(ks)
            orphans = sum(1 for k in keys if k not in pkeys)
            tp.structure.cross_file_keys.append({
                "column": p.name, "parent": _table_ref(pfr.rel_path, ptp.name, pp.name),
                "key_containment": round(contained, 6), "orphan_rows": int(orphans),
                "orphan_rate": round(orphans / len(keys), 6), "foreign_key": contained >= 0.99,
                "status": ("not kept (structure: disable: [cross_file_keys])" if disabled
                           else "kept: same synthetic IDs as the parent file"),
            })


#: Detection params that quote real values (made private once a column is known to hold real IDs).
_VALUE_PARAMS = ("min", "max", "value", "token", "levels", "tokens")
#: An ID column whose keys are at least this share of another table's ID keys shares its mapping (D5: >= 99%).
LINK_CONTAINMENT = 0.99
#: Digit keys are evidence only where the namespace's keys are sparse over the column's value range.
LINK_MAX_DENSITY = 0.5
#: A non-ID column holds the IDs of an ID namespace when at least this share of its distinct values are keys of
#: it (and that is no coincidence, :func:`_key_ref_share`): it is treated as an ID column (R2-01).
KEY_REF_SHARE = 0.5
#: Digit keys: the largest accepted chance that the matched values are keys only by coincidence.
KEY_REF_CHANCE = 1e-3
#: Types never promoted to an ID column: PII and text shapes whose generators never write a real value (an SSN
#: used as a student ID stays an SSN in the invalid area range), booleans, coordinates, empty columns (dates too).
_NO_KEY_REF_TYPES = frozenset({T.FREE_TEXT, T.EMAIL, T.PERSON_NAME, T.PHONE, T.SSN, T.STREET_ADDRESS, T.IP, T.URL,
                               T.MULTI_SELECT, T.CITY, T.STATE, T.ZIP, T.LAT, T.LON, T.BOOLEAN, T.EMPTY,
                               T.BINARY_NESTED})


def _digit_keys_beyond_chance(digit_c: set[str], digit_m: set[str], ints: np.ndarray) -> bool:
    """The matched digit keys ``digit_m`` of a column's digit keys ``digit_c`` show it holds the namespace's IDs (sorted
    digit keys ``ints``): numbers drawn at random over the column's digit range (the namespace's range for 1-2
    values) are keys at the namespace's density there, so the matched share must be at least twice that density and
    the chance that every match is a coincidence at most :data:`KEY_REF_CHANCE`."""
    nums = sorted(int(k) for k in digit_c)
    if len(nums) >= 3:
        lo, hi = nums[0], nums[-1]
    elif len(ints):
        lo, hi = int(ints[0]), int(ints[-1])
    else:
        return False
    inside = int(np.searchsorted(ints, hi, side="right") - np.searchsorted(ints, lo))
    density = inside / (hi - lo + 1)
    return not (len(digit_m) / len(digit_c) < 2 * density or density ** len(digit_m) > KEY_REF_CHANCE)


def _digit_keys_beyond_width_chance(digit_m: set[str], ints: np.ndarray) -> set[str]:
    """The matched digit keys ``digit_m`` whose number of digits alone shows that the column holds the namespace's
    IDs (sorted digit keys ``ints``), however dense the namespace is over the column's own range (K-13 / R2-01): an
    arbitrary number of ``w`` digits is a key with chance ``d`` (the namespace's keys of ``w`` digits over all numbers
    of ``w`` digits), and ``d ** min(k, 2)`` must be at most :data:`KEY_REF_CHANCE` for the ``k`` matched keys of that
    width (two independent coincidences at most: a column's values cluster). 37 of 300 dense 6-digit student IDs
    (100001-100300) are; 1-5 (a grade) or 40-100 (a score) beside dense IDs 1..2000 are not."""
    out: set[str] = set()
    for w, k in Counter(len(m) for m in digit_m).items():
        lo, hi = (0 if w == 1 else 10 ** (w - 1)), 10 ** w - 1
        inside = int(np.searchsorted(ints, hi, side="right") - np.searchsorted(ints, lo))
        if (inside / (hi - lo + 1)) ** min(k, 2) <= KEY_REF_CHANCE:
            out |= {m for m in digit_m if len(m) == w}
    return out


#: Harden 2 (A2-5): matched digit keys spanning a namespace (:func:`_digit_keys_span_namespace`): at least this many,
#: leaving at most :data:`SPAN_GAPS` mean spacings of the namespace's keys outside their range, with at least
#: :data:`SPAN_INSIDE_SHARE` of the column's digit keys inside the namespace's range (a band of measures reaching past
#: the IDs is no sample of them).
SPAN_MIN_KEYS = 30
SPAN_GAPS = 6
SPAN_INSIDE_SHARE = 0.9


def _digit_keys_span_namespace(digit_c: set[str], digit_m: set[str], ints: np.ndarray) -> bool:
    """Harden 2 (A2-5): the matched digit keys ``digit_m`` of a column (digit keys ``digit_c``) are a sample of a
    namespace that is dense over its own range (sorted digit keys ``ints``: a warehouse's 1..N key, sequential IDs
    1001-1800), where no test over the column's range can tell IDs from numbers. A reference column's ``m`` values are
    drawn from the whole namespace of ``N`` keys, so they reach both of its ends: about ``N / m`` keys lie beyond each
    end, and at most :data:`SPAN_GAPS` ``* N / m`` in all is accepted (a sample fails that about 2% of the time). A band
    of measures inside the namespace (scale scores 2380-2620 beside IDs 1001-4000, Lexile 200-1700 or minutes 0-600
    beside IDs 1..6000) leaves a large part of it uncovered. Also needed: :data:`SPAN_MIN_KEYS` matched keys (the band
    must then cover more than 80% of the namespace), most of them wider than a code list or a count
    (``detect.ids.SMALL_NUMBER_DIGITS``: grade levels 1-12 are never IDs by this test), and
    :data:`SPAN_INSIDE_SHARE` of the column's digit keys inside the namespace's range. A reference to one contiguous
    block of the namespace (a cohort) is no such sample: a residual of dense 3-5 digit namespaces."""
    from ..detect.ids import SMALL_NUMBER_DIGITS

    n, m = len(ints), len(digit_m)
    if m < SPAN_MIN_KEYS or not n:
        return False
    if 2 * sum(len(k) > SMALL_NUMBER_DIGITS for k in digit_m) < m:
        return False
    lo_ns, hi_ns = int(ints[0]), int(ints[-1])
    if sum(lo_ns <= int(k) <= hi_ns for k in digit_c) < SPAN_INSIDE_SHARE * len(digit_c):
        return False
    nums = [int(k) for k in digit_m]
    outside = int(np.searchsorted(ints, min(nums))) + n - int(np.searchsorted(ints, max(nums), side="right"))
    return outside <= SPAN_GAPS * n / m


def _key_ref_share(keys_c: set[str], keys: set[str], ints: np.ndarray, by_name: bool) -> float | None:
    """Share of a column's distinct D8 keys ``keys_c`` that are keys of one ID namespace (``keys``; its sorted
    digit keys ``ints``), when that shows the column holds those IDs; None when it may be a coincidence.

    At least :data:`KEY_REF_SHARE` of the values must be keys. A column with the same (normalised) header as
    one of the namespace's ID columns needs nothing more. Otherwise the digit keys pass the chance test over the
    column's range (:func:`_digit_keys_beyond_chance`: a 40-100 score column is not a reference to IDs 1..2000; two
    values out of 5,000 sparse 6-digit IDs are) or over their number of digits
    (:func:`_digit_keys_beyond_width_chance`: 37 of 300 dense 6-digit IDs are), in a column of digits alone or among
    other values (K-13: the ``1``-``5`` of a ``KG``, ``1``-``5`` grade column beside dense IDs 1..N are a
    coincidence, never evidence); among other values they may also span a dense namespace
    (:func:`_digit_keys_span_namespace`, harden 2 A2-5: 268 of the dense IDs 1001-1800 among ``TBD``); matched digit
    keys a coincidence explains are dropped before the share is taken. Other values: at least 3 matched keys, or
    matched keys that contain a digit (``S100113``; a lone ``unknown`` that also appears in an ID column is not
    evidence).
    """
    matched = keys_c & keys
    if not matched:
        return None
    share = len(matched) / len(keys_c)
    if share < KEY_REF_SHARE:
        return None
    if by_name:
        return share
    digit_c = {k for k in keys_c if k.isascii() and k.isdigit() and len(k) <= 18}
    digit_m = matched & digit_c
    if digit_m and not _digit_keys_beyond_chance(digit_c, digit_m, ints) and not (
            digit_c != keys_c and _digit_keys_span_namespace(digit_c, digit_m, ints)):
        # K-13: coincidental digit matches are no evidence (unless their width alone is: dense 6-digit IDs; or, among
        # placeholders, they span a dense namespace: harden 2, A2-5)
        chance = digit_m - _digit_keys_beyond_width_chance(digit_m, ints)
        matched, digit_m = matched - chance, digit_m - chance
        share = len(matched) / len(keys_c)
        if not matched or share < KEY_REF_SHARE:
            return None
    if digit_m:
        return share  # digit keys beyond chance
    if digit_c == keys_c:
        return None  # every key is a digit key (none matched beyond chance)
    if len(matched) >= 3 or any(ch.isdigit() for k in matched for ch in k):
        return share
    return None


def _key_reference_columns(file_results: list[Any], groups: dict[str, list[tuple[Any, Any, ColumnProfile]]],
                           settings: Settings | None = None) -> None:
    """A non-ID column whose values are the real IDs of an ID column (D5/D8 key discovery; R2-01).

    Such a column would write real person IDs as they are (a factor, a discrete integer, the exact min/max of a
    numeric column) and would not get the synthetic IDs, so joins would break. It is found by its header (the
    normalised name of an ID column of the run) or by its content (:func:`_key_ref_share`), in any table of the
    run (another file, another sheet, or its own table: ``mentor_ref`` holding ``student_id`` values).
    Row-number namespaces kept as they are (1..n, D9) are skipped. The real IDs of a column a flag turned from an
    ID column into something else (``--exclude student_id``, ``--null``, ``--force-*``;
    ``profile.entity.shadow_namespaces``, R6-02) count too, for the columns no flag names as such.

    * The column is marked ``params['_key_ref']`` = ``{'of': ID column, 'share': share}`` (PRIVATE): the
      pipeline re-profiles its file with the column as an ID column (detection source ``key-ref``, the way
      ``--id-col`` would), so it gets synthetic IDs, shared with the ID column when >= 99% of its keys are
      found there (:func:`_key_links`). A column still marked when generation starts fails its file (D19).
    * A multi-valued column (MULTI_SELECT, ``S1001;S1002``) is checked on its items; one holding lists of real
      IDs is marked ``params['_key_ref_list']`` and its file fails (D19, the fix: ``--force-text``, ``--null``
      or ``--exclude``): a list of IDs cannot be one ID per cell, and its items would be written as they are.
    * An explicitly typed column (``--force-factor``, ``--org-code``, ``--force-type``, ``--force-text``,
      ``--pii-col``) and PII/text/date columns, whose generators never write a real value, keep their type,
      with a warning.
    * A column of a built-in ordinal vocabulary (grade levels ``KG``, ``1``-``12`` ...; ``hints.ORDINAL_VOCABS``) is
      never a key reference, whatever IDs its levels equal (K-13).

    Either way the column's value statistics are redacted from the report like an ID candidate's
    (``params['id_candidate']``). Messages hold names and shares only. Idempotent (the pipeline calls it again
    after re-profiling).
    """
    from ..generate.ids import IdRegistry, namespace_is_identity

    na = settings.na_tokens if settings is not None else frozenset()
    by_name: dict[str, list[int]] = {}  # normalised ID column name -> spaces holding it
    spaces: list[tuple[set[str], np.ndarray, str]] = []  # (keys, sorted int keys, ref)
    for key, members in groups.items():
        reg = IdRegistry()
        ks_all: set[str] = set()
        refs = []
        for fr, tp, p in members:
            raw = tp.private.id_raw[p.index]
            reg.register(key, fr.rel_path, tp.name, p.index, p.name, raw)
            ks_all |= {k for k in entity_keys(codes_of(raw)[1]) if k is not None}
            refs.append(_table_ref(fr.rel_path, tp.name, p.name))
        if not ks_all or namespace_is_identity(reg.namespaces[key]):
            continue  # row numbers 1..K: kept as they are (D9)
        ints = np.asarray(sorted(int(k) for k in ks_all if k.isascii() and k.isdigit() and len(k) <= 18),
                          dtype=np.int64)
        for _, _, p in members:
            nm = normalize(p.name)
            if nm not in GENERIC_ID_NAMES and len(spaces) not in by_name.setdefault(nm, []):
                by_name[nm].append(len(spaces))
        spaces.append((ks_all, ints, refs[0]))
    # R6-02: a column a flag turned from an ID column into something else (--exclude student_id, --null, --force-*)
    # still holds real IDs: a column holding them is an ID column too (but a flagged one is judged by its flag)
    n_id_spaces = len(spaces)
    for sns in entity_mod.shadow_namespaces(file_results, failed=False):
        ks_all = {normalize_id(v) for v in sns.raw_values}
        ints = np.asarray(sorted(int(k) for k in ks_all if k.isascii() and k.isdigit() and len(k) <= 18),
                          dtype=np.int64)
        for _, _, _, name in sns.members:
            nm = normalize(name)
            if nm not in GENERIC_ID_NAMES and len(spaces) not in by_name.setdefault(nm, []):
                by_name[nm].append(len(spaces))
        file_rel, table, _, name = sns.members[0]
        spaces.append((ks_all, ints, _table_ref(file_rel, table, name)))
    for fr in file_results:
        if fr.failed:
            continue
        for tp in fr.tables:
            for p in tp.columns:
                p.params.pop("_key_ref", None)
                p.params.pop("_key_ref_list", None)
                p.messages[:] = [m for m in p.messages if m.code != "unlinked_key_column"]  # recomputed below
                if p.role not in ("row", "entity") or p.model is None:
                    continue
                if p.type is T.ORDINAL and p.params.get("vocabulary"):
                    continue  # K-13: a built-in vocabulary (grade levels KG, 1-12 ...) holds no one's ID
                vals = {v.strip() for v in p.model.nonmissing_counts if isinstance(v, str) and v.strip()} - na
                if not vals:
                    continue
                # a multi-valued cell ('S1001;S1002') holds its items: each one can be a real ID
                delim = p.params.get("delimiter") if p.type is T.MULTI_SELECT else None
                if delim:
                    vals = {it.strip() for v in vals for it in v.split(delim) if it.strip()} - na
                    if not vals:
                        continue
                keys_c = {normalize_id(v) for v in vals}
                named = set(by_name.get(normalize(p.name), []))
                found: tuple[float, str, bool] | None = None
                flagged = entity_mod.shadow_id(p, settings)
                for s_i, (keys, ints, ref) in enumerate(spaces):
                    if flagged and s_i >= n_id_spaces:
                        continue  # two columns flagged by the user: each is judged by its own flag
                    share = _key_ref_share(keys_c, keys, ints, s_i in named)
                    if share is not None and (found is None or share > found[0]):
                        found = (share, ref, s_i >= n_id_spaces)
                if found is None:
                    continue
                share, ref, of_shadow = found
                # its values are real IDs: keep them out of the report like an ID candidate's (D7, D21)
                p.params.setdefault("id_candidate", f"holds the keys of ID column {ref}")
                redact_candidate(p.stats, str(p.params["id_candidate"]))
                for k in _VALUE_PARAMS:
                    if k in p.params:
                        p.params["_" + k] = p.params.pop(k)
                me = _table_ref(fr.rel_path, tp.name, p.name)
                if p.source in EXPLICIT_SOURCES:
                    p.messages.append(Message(
                        "warning", "unlinked_key_column",
                        f"column {p.name!r} holds the IDs of ID column {ref} ({share:.1%} of its distinct values) but "
                        f"was given its type explicitly ({p.source}): it is not treated as an ID column, so it does "
                        "not get the same synthetic IDs (joins between them break)"
                        + ("" if _values_never_real(p) else ", and its real values are written as they are"),
                        column=p.name, fix=f"--id-col {me}"))
                elif delim:
                    # a list of IDs cannot be an ID column (one ID per cell): its file is not written (D19)
                    p.params["_key_ref_list"] = {"of": ref, "share": round(share, 6)}
                    p.messages.append(Message(
                        "warning", "unlinked_key_column",
                        f"column {p.name!r} holds lists of the IDs of ID column {ref} ({share:.1%} of its distinct "
                        f"items): its items would be written as they are, so its file is not written",
                        column=p.name, fix=f"--force-text {me} (invented labels), or --null {me} / --exclude {me}"))
                elif p.type in _NO_KEY_REF_TYPES or _is_date(p):
                    p.messages.append(Message(
                        "warning", "unlinked_key_column",
                        f"column {p.name!r} holds the IDs of ID column {ref} ({share:.1%} of its distinct values) but "
                        f"is a {p.type.value} column: its values are replaced, but it does not get the same synthetic "
                        "IDs, so joins between them break",
                        column=p.name, fix=f"--id-col {me}"))
                else:
                    p.params["_key_ref"] = {"of": ref, "share": round(share, 6)}
                    if of_shadow:  # R6-02: the IDs of a column a flag made no ID column (no mapping to share)
                        p.params["_key_ref"]["flagged"] = True
                    p.messages.append(Message(
                        "warning", "unlinked_key_column",
                        f"column {p.name!r} holds the IDs of ID column {ref} ({share:.1%} of its distinct values) "
                        "but is not an ID column: the run treats it as one (synthetic IDs, never its real values)",
                        column=p.name, fix=f"--id-col {me}"))


def _values_never_real(p: ColumnProfile) -> bool:
    """The column's generator never writes a real value (never-copy types, same-shape code masks)."""
    return p.type in NEVER_COPY_TYPES or (p.type is T.DIGIT_CODE and not p.params.get("levels_mode"))


def _luhn_messages(groups: dict[str, list[tuple[Any, Any, ColumnProfile]]], settings: Settings) -> list[Message]:
    from ..generate.ids import luhn_eligible

    if settings.id_check != "luhn":
        return []
    msgs = []
    for key, members in sorted(groups.items()):
        reps: dict[str, str] = {}
        for _, tp, p in members:
            for v in tp.private.id_raw[p.index].tolist():
                if v is not None and v.strip():
                    reps.setdefault(normalize_id(v), v.strip())
        names = ", ".join(sorted({p.name for _, _, p in members}))
        if luhn_eligible(list(reps), reps):
            msgs.append(Message("info", "id_check", f"ID column(s) {names}: synthetic IDs get a valid Luhn check "
                                                    "digit (--id-check luhn)"))
        else:
            from ..detect.ids import luhn_valid

            bad = sum(1 for k, r in reps.items() if not (r.isascii() and r.isdigit() and luhn_valid(k)))
            msgs.append(Message("warning", "id_check_not_applied",
                                f"--id-check luhn is not applied to ID column(s) {names}: {bad} of {len(reps)} real "
                                "IDs are not digit IDs with a valid Luhn check digit"))
    return msgs


def cross_file_structure(file_results: list[Any], settings: Settings) -> list[Message]:
    """AREA D hook: cross-file key coverage and key discovery over the whole run (D5, D8).

    Called once after pass 1 with every FileResult (profiles only), in ``run``
    and ``inspect``:

    * validates ``rules:`` (a malformed rule is a usage error, exit 2) and
      resolves the declared rules per table (``declared_rules``, swap groups);
    * assigns ID namespaces: identical aliases share one mapping, and so do ID
      columns of different tables whose keys are >= 99% contained in one
      another (content-based key discovery, whatever the headers); with
      ``structure: {disable: [cross_file_keys]}`` every file gets its own;
    * reports key coverage of every ID column against the file where the same
      ID is unique (containment, orphan rows/rate) in ``cross_file_keys``;
    * finds non-ID columns holding the IDs of an ID column (``params['_key_ref']``: the pipeline re-profiles
      them as ID columns, R2-01) and warns about explicitly typed ones;
    * reports where ``--id-check luhn`` applies.
    """
    msgs = _resolve_rules(file_results, settings)
    groups = _assign_namespaces(file_results, settings)
    _coverage(groups, settings)
    _key_reference_columns(file_results, groups, settings)
    msgs.extend(_luhn_messages(groups, settings))
    return msgs


# --------------------------------------------------------------------------- generation hooks
def apply_structure(table_profile: Any, columns: dict[int, np.ndarray],
                    rng_for: Callable[[str], np.random.Generator], settings: Settings) -> list[Message]:
    """AREA D hook: generation-time structural switches (D5) and declared rules.

    Called by ``generate/table.py`` after IDs, within-ID constants, missingness and
    every independent column are generated (``columns``: column index -> rendered
    values, editable in place) and before privacy swap-redraw. ``rng_for(purpose)``
    returns a seeded Generator. Order: declared rules, functional dependencies,
    derived templates, composite keys (``generate.structure``).
    """
    from ..generate import structure as gs

    return gs.apply(table_profile, columns, rng_for, settings)


def finalize_rows(table_profile: Any, synth: Any, rng_for: Callable[[str], np.random.Generator],
                  settings: Settings) -> list[Message]:
    """AREA D hook: row-level structure after privacy swap-redraw (pipeline pass 2).

    Called by the pipeline right after ``privacy.matches.enforce`` and before the
    never-copy scan and writing. Repairs composite keys the swaps may have broken
    (rows that would then equal a real row are left alone), moves cells between
    rows (same column totals) towards the real duplicate-row count, and orders
    the rows by the detected sort keys (or leading missing blocks), updating
    ``table_profile.private.entity_index`` to the new row order.
    """
    from ..generate import structure as gs

    return gs.finalize(table_profile, synth, rng_for, settings)


# --------------------------------------------------------------------------- report lines
def describe_structure(tp: Any) -> list[str]:
    """Report-safe lines (names, counts, statuses) for ``inspect`` and the per-file report."""
    names = [c.name for c in tp.columns]
    sp = tp.structure
    lines: list[str] = []
    ent = tp.entity
    for g in ent.aliases:
        lines.append("ID aliases (1:1, kept 1:1): " + " = ".join(names[i] for i in g))
    for n in ent.nesting:
        cpp = n.get("children_per_parent", {})
        lines.append(f"ID nesting: {n['child']} within {n['parent']} ({n['n_child']} in {n['n_parent']}, "
                     f"{cpp.get('min')}-{cpp.get('max')} per {n['parent']}; kept)")
    for fd in sp.functional_dependencies:
        lines.append(f"functional dependency {fd['text']} ({fd['kind']}, {fd['determinant_values']} values): "
                     f"{fd['status']}")
    for t in sp.derived_templates:
        lines.append(f"derived template {t['column']} = {t['template']} ({t['rows']} rows): {t['status']}")
    for k in sp.composite_keys:
        lines.append(f"composite key {k['text']} ({k['rows']} rows): {k['status']}")
    for e in sp.cross_file_keys:
        lines.append(f"cross-file key {e['column']} -> {e['parent']}: {e['key_containment']:.1%} of keys found, "
                     f"{e['orphan_rows']} orphan rows ({e['orphan_rate']:.1%}): {e['status']}")
    for r in sp.declared_rules:
        lines.append(f"declared rule {r['rule']}: {r['status']}")
    for r in sp.suggested_rules:
        lines.append(f"suggested rule {r['rule']} (true in all {r['rows']} rows; to apply it add to --config: "
                     f"{r['config']})")
    for pnl in sp.panels:
        lines.append(pnl["text"])
    so = sp.sort_order
    if so.get("keys"):
        keys = ", ".join(f"{k['column']} {k['direction']}" + (f" (missing {k['missing']})" if k["missing"] else "")
                         for k in so["keys"])
        lines.append(f"sorted by {keys}: {so.get('status')}")
    elif so.get("leading_missing"):
        blocks = ", ".join(f"{b['column']} ({b['rows']} rows)" for b in so["leading_missing"])
        lines.append(f"leading blocks of missing values: {blocks}: {so.get('status')}")
    dr = sp.duplicate_rows
    if dr.get("rows"):
        lines.append(f"duplicate rows: {dr['rows']} ({dr['rate']:.1%}): {dr.get('status')}")
    return lines


__all__ = [
    "profile_missingness", "profile_entity_missingness", "sample_missing_mask", "sample_entity_mask",
    "profile_structure", "find_functional_dependencies",
    "find_derived_templates", "find_composite_keys", "suggest_inequality_rules", "detect_panels", "swap_groups",
    "cross_file_structure", "apply_structure", "finalize_rows", "describe_structure", "parse_rules",
    "order_values", "RuleView", "rule_view", "rule_bounds", "rule_holds",
]

