"""Generation-time structural switches and row-level layout (AREA D; design D5, D6, D16, section 6).

Called through ``profile.structure.apply_structure`` (inside ``generate/table.py``,
before privacy swap-redraw) and ``profile.structure.finalize_rows`` (pipeline,
after it). Works on rendered synthetic values only; the real-derived inputs are
the PRIVATE structure data from pass 1 (``StructureProfile.private``): FD level
mappings, template parts and reserved-domain counts, observed rule gaps.

:func:`apply` (before privacy), in order:

1. FD determinants vs whole-row missingness (:func:`align_fd_missingness`, R2-09):
   a dependent rebuilt from its determinant gets the determinant's mapped
   blank/non-blank state, so each root determinant's synthetic values are dealt
   to the rows (entities, for within-ID constants) whose sampled D6 pattern
   wants exactly that state for every dependent in its FD tree (values move
   only between rows of the same determinant missing state; marginals unchanged);
2. declared rules and functional dependencies, in column-dependency order (a
   step that writes a column runs before the steps that read it: an FD whose
   dependent is a rule's low column runs first, one whose determinant is a
   rule's high column after it):

   * declared rules (``rules:`` in ``--config``), one target per high column B
     (R2-11): B keeps its own independently generated values (D3: its source
     range, exact min/max (D14), only observed levels or calendar dates, each
     value's own written layout), and every rule on B together decides where
     they go (R3-06): each unit (row; entity for a within-ID constant)
     targets ``max(A1, A2, ...)`` plus a gap drawn from the observed gaps
     (exact quota for integers and dates, interpolated for decimals), and B's
     values are dealt in ascending order, each to the unit with the smallest
     target whose bounds it meets - by construction, never by rejection, and
     written verbatim so rendering cannot undo a rule. Whole-number gaps are
     then matched to their observed counts, a within-ID constant's values to
     their rows per ID block (R2-10, R3-07), and rows the deal made equal to a
     real row swap their B value (D23), always by swaps that keep every rule.
     A unit left without a value takes the smallest value of B that meets its
     bounds; only a low value above all of B is written past B's maximum
     (warned). Dates compare in their model unit (days, months, ticks; R2-16);
   * functional dependencies: the dependent is rebuilt from the synthetic
     determinant through the observed mapping; a never-copy dependent (names,
     invented labels) maps each real value onto one of its own synthetic values
     (same frequency rank), so no real value appears. Rows where the mapped
     blank/non-blank state still differs from the sampled pattern (pooled rare
     patterns, floors) follow the FD and are counted in a warning;
3. derived templates: rebuilt from the synthetic inputs (free trailing digits by
   their length distribution, reserved e-mail domains by exact quota); a value
   that would equal a real one of the column, or a real text of any column of the
   run that the run never writes (``privacy.texts``, R7-01/R7-02), keeps its
   independently generated value. When the
   rebuilt values do not have the source's distinct count and frequency profile
   (independent inputs pair up differently), the inputs are first regrouped to
   the source's value groups (R5-04, :func:`regroup_template`: each input keeps
   its values and counts, its frequency ranks shuffled among groups of one size);
4. composite keys: the column's synthetic values (with the columns that move
   with it) are reassigned so that no ID block repeats a value (greedy
   largest-remaining-capacity assignment; exact column totals); a tuple moves
   only to a slot with the same blank/non-blank state of every moved column.
   What moving rows cannot fix (too many copies of a key value, R5-03) is fixed
   by rebuilding a pair key block by block, swaps, and a free date for a repeated
   interpolated date (:func:`_settle_key`); the key's status says 'enforced' only
   when no synthetic row repeats it.

:func:`finalize` (after privacy):

1. composite keys broken by privacy swaps are repaired by swaps (whole rows, rows
   with a blank key, single key values) that do not create a row equal to a real
   one or change a row's missingness pattern; the key's final status is set;
2. templates with an input of another role (an ID, or a within-ID constant for
   a row-level target; :func:`pinned_templates`) are rebuilt: row moves carry
   only the columns of one role, which would split such a template;
3. duplicate rows: cells are swapped between rows (column totals and missing
   patterns unchanged) towards the real duplicate-row count, without creating
   rows equal to real ones; pinned template columns never move;
4. checks on the final values: every applied declared rule is re-checked
   (:func:`check_rules`; a violation FAILs the file unless ``--no-strict``,
   then it is a warning and the rule's status says so), and the blank/non-blank
   state of every column a structural step touched is compared with the state
   after step 2 of :func:`apply` (a warning names the columns whose patterns
   changed);
5. row order: the detected sort keys (direction, missing first/last), else the
   leading missing blocks; ``tp.private.entity_index`` follows the new order.
   Synthetic IDs are assigned at random (``generate.ids``), so a file sorted by
   ID is sorted here.
"""

from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..errors import GenerationError
from ..model import LEVEL_TYPES, TYPE_INFO, ColumnProfile, ColumnType, Message
from ..privacy import matches
from ..privacy import texts as real_texts
from ..profile import templates as tpl
from ..profile.stats import missing_mask
from . import pseudonyms
from .numeric import interp_icdf, render_numeric, render_numeric_extended, stratified_uniform
from .quota import largest_remainder, quota_sample

T = ColumnType
#: tp.private.extra key: blank/non-blank state of the structural columns after the FD/rule step (finalize check)
_MISSING_SNAPSHOT = "structure_missing_snapshot"
#: template info key set by sharing mode (``profile.sharing.REBUILT_KEY``, issue #30): the target is written from the
#: template only, a row it cannot fill safely being written empty
_SHARING_REBUILT = "_sharing_rebuilt_from"


def _label_generated(p: ColumnProfile) -> bool:
    """Generated by quota over the real raw values (real labels written)."""
    if p.type in LEVEL_TYPES:
        return True
    if p.type is T.INTEGER and p.subtype == "discrete":
        return True
    return p.type is T.DIGIT_CODE and bool(p.params.get("levels_mode"))


def _disabled(settings: Any, switch: str) -> bool:
    return bool(settings is not None and settings.structure_disabled(switch))


def _miss(tp: Any, columns: dict[int, np.ndarray], c: int) -> np.ndarray:
    p = tp.columns[c]
    return missing_mask(columns[c], p.missing_policy, p.na_tokens)


def _run_texts(tp: Any) -> Any:
    """The run's real texts that it never writes (``privacy.texts.RealTexts``, R7-01/R7-02; None without run
    planning): a rebuilt value equal to one keeps its independently generated value."""
    private = getattr(tp, "private", None)
    return private.extra.get(real_texts.RUN_KEY) if private is not None else None


# --------------------------------------------------------------------------- before privacy
def apply(tp: Any, columns: dict[int, np.ndarray], rng_for: Callable[[str], np.random.Generator],
          settings: Any) -> list[Message]:
    priv = tp.structure.private
    msgs: list[Message] = []
    tp.private.extra.pop(_MISSING_SNAPSHOT, None)
    if not priv:
        return msgs
    fds_on = bool(priv.get("fds")) and not _disabled(settings, "functional_dependencies")
    if fds_on:
        msgs += align_fd_missingness(tp, columns, rng_for("fd_missingness"))
    ops = _structural_ops(tp, columns, fds_on)
    if ops:
        rules_rng = rng_for("rules") if any(kind == "rule" for kind, _ in ops) else None
        for kind, x in ops:
            if kind == "rule":
                msgs += apply_rules(tp, columns, rules_rng, [x])
            else:
                msgs += apply_fds(tp, columns, [x])
    templates_on = bool(priv.get("templates")) and not _disabled(settings, "derived_templates")
    composite_on = bool(priv.get("composite")) and not _disabled(settings, "composite_keys")
    _snapshot_missing(tp, columns, templates_on, composite_on)
    if templates_on:
        msgs += apply_templates(tp, columns, rng_for("templates"))
    if composite_on:
        msgs += enforce_composite_keys(tp, columns, rng_for("composite"))
    return msgs


def _structural_ops(tp: Any, columns: dict[int, np.ndarray], fds_on: bool) -> list[tuple[str, dict[str, Any]]]:
    """Declared-rule targets and FDs in column-dependency order (writers before readers; ties keep the
    original order: rules first, in their order, then FDs in theirs; a cycle keeps the original order)."""
    priv = tp.structure.private
    ops: list[tuple[str, dict[str, Any], set[int], set[int]]] = []
    if priv.get("declared"):
        for hi in priv.get("declared_targets", []):
            t = priv.get("rule_targets", {}).get(hi)
            if t is not None:
                ops.append(("rule", t, {hi}, {m[0] for m in t["members"]}))
    if fds_on:
        for fd in priv.get("fds", []):
            if not fd.get("overridden") and fd["det"] in columns and fd["dep"] in columns:
                ops.append(("fd", fd, {fd["dep"]}, {fd["det"]}))
    if len(ops) < 2:
        return [(k, x) for k, x, _, _ in ops]
    n = len(ops)
    before = [{i for i in range(n) if i != j and ops[i][2] & ops[j][3]} for j in range(n)]
    done: list[int] = []
    left = list(range(n))
    while left:
        ready = next((j for j in left if not (before[j] - set(done))), None)
        if ready is None:  # a cycle: the rest in the original order (check_rules reports any violation)
            done += left
            break
        done.append(ready)
        left.remove(ready)
    return [(ops[j][0], ops[j][1]) for j in done]


# --------------------------------------------------------------------------- declared rules
def _first_rows(tp: Any) -> np.ndarray | None:
    index = tp.private.entity_index
    if index is None or tp.entity.n_entities == 0:
        return None
    first = np.full(tp.entity.n_entities, -1, dtype=np.int64)
    first[index[::-1]] = np.arange(len(index) - 1, -1, -1)
    return first


def apply_rules(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator,
                targets: list[dict[str, Any]] | None = None) -> list[Message]:
    """Build every declared rule's high column (module doc, step 2); ``targets`` default to all, in order."""
    priv = tp.structure.private
    if targets is None:
        targets = [priv["rule_targets"][hi] for hi in priv.get("declared_targets", [])
                   if hi in priv.get("rule_targets", {})]
    msgs: list[Message] = []
    for t in targets:
        msgs += _apply_target(tp, columns, t, rng)
    return msgs


def _apply_target(tp: Any, columns: dict[int, np.ndarray], t: dict[str, Any],
                  rng: np.random.Generator) -> list[Message]:
    """Deal the high column's own synthetic values so every rule on it holds (R3-06).

    The high column B was generated independently like any other column (D3): its values are in its source
    range with the exact min/max (D14), only observed levels for a discrete column, observed dates for a
    calendar column, its own layouts. A rule only decides *where* each value goes (units: rows; entities
    for a within-ID constant): each unit's target is its bound (the largest low value over the rules on B)
    plus a gap drawn from the observed gaps (exact quota for integers and dates), a unit without a low value
    targets its own draw, and B's values are dealt in ascending order, each to the unit with the smallest
    target among those whose bounds it meets (:func:`_deal_rule_values`) - so the gaps look like the
    source's, and every unit is served whenever any arrangement of B's values can serve them. B's values
    only move (its distribution is unchanged) and are written verbatim, so no rounding or re-rendering can
    undo a rule. Where the synthetic low values leave a unit without a value (rare: the low column's draw
    sits above B's), it takes the smallest value of B that meets its bounds (B's exact min/max are kept);
    only a low value above B's whole range forces a value past B's maximum, which is warned about.
    """
    from ..profile.structure import rule_bounds, rule_view

    hi, members = t["hi"], t["members"]
    gaps = t.get("gaps")
    if hi not in columns or any(lo not in columns for lo, _, _ in members) or gaps is None or not len(gaps):
        return []
    p_hi = tp.columns[hi]
    hv = rule_view(columns[hi], p_hi, layout=True)
    lvs = [rule_view(columns[lo], tp.columns[lo]) for lo, _, _ in members]
    if hv is None or any(v is None or v.unit != hv.unit for v in lvs):
        return []
    n = len(hv.vals)
    is_int = hv.vals.dtype.kind == "i"
    index = tp.private.entity_index
    first = _first_rows(tp) if t.get("per_entity") and index is not None and len(index) == n else None
    if first is not None:
        n_units, unit_of_row = len(first), index
        rep = np.maximum(first, 0)  # the row that represents each entity
        hi_ok = hv.ok[rep] & (first >= 0)
    else:
        n_units, unit_of_row, rep, hi_ok = n, None, np.arange(n), hv.ok
    bound, has_any, per = rule_bounds([(v.vals, v.ok, m[1]) for v, m in zip(lvs, members)],  # type: ignore[union-attr]
                                      unit_of_row, n_units, is_int)
    pool = np.flatnonzero(hi_ok)  # units whose B value is filled and readable: their values are dealt
    bounded = has_any[pool]
    if not bounded.any():
        return []
    k = len(pool)
    pvals = hv.vals[rep[pool]]
    order = np.argsort(pvals, kind="stable")
    sv = pvals[order]  # B's values, sorted
    texts = np.asarray(columns[hi], dtype=object)[rep[pool]][order]  # written verbatim, layout included
    lo_idx = np.zeros(k, dtype=np.int64)  # per unit: the first sorted value that meets every rule on B
    for bnd, has, strict in per:
        idx = np.searchsorted(sv, bnd[pool], side="right" if strict else "left")
        lo_idx = np.where(has[pool], np.maximum(lo_idx, idx), lo_idx)
    m = int(bounded.sum())
    if t.get("integer"):
        gv, gc = np.unique(gaps, return_counts=True)
        g, _ = quota_sample(gv.tolist(), gc.tolist(), m, rng, floor=False)
        g = np.asarray(g, dtype=float)
    else:
        g = interp_icdf(gaps, stratified_uniform(m, rng))
    # what each unit would like: its bound plus an observed gap; a unit without a low value its own draw
    target = np.asarray(pvals, dtype=float).copy()
    target[bounded] = np.asarray(bound[pool][bounded], dtype=float) + g
    take = _deal_rule_values(lo_idx, target, rng.permutation(k))
    # units left without a value that meets their bounds take the smallest value of B that does
    keep_ext = not p_hi.params.get("_no_exact_extremes")
    take, dropped = _settle_leftovers(take, lo_idx, sv, k, keep_ext)
    if t.get("integer"):
        take = _repair_gaps(take, lo_idx, sv, np.asarray(bound[pool]), bounded, g, rng)
    if first is not None:  # a within-ID constant: its values' rows per ID block too (R2-10, R3-07)
        take = _rebalance_rule_rows(tp, hi, pool, take, lo_idx, texts, rng,
                                    np.where(bounded, np.asarray(bound[pool], dtype=float), np.nan), sv,
                                    set(np.asarray(gaps, dtype=float).tolist()) if t.get("integer") else None)
        if t.get("integer"):  # gaps once more, between IDs of one block size (their rows stay)
            sizes = np.bincount(index, minlength=n_units)[pool]
            take = _repair_gaps(take, lo_idx, sv, np.asarray(bound[pool]), bounded, g, rng, same=sizes)
    new_text = np.empty(k, dtype=object)
    have = take >= 0
    new_text[have] = texts[take[have]]
    msgs: list[Message] = []
    over = np.flatnonzero(~have)  # every value of B is below their bounds: no value in B's range meets the rule
    if len(over):
        sb = [(bnd[pool][over], has[pool][over], strict) for bnd, has, strict in per]
        b = _minimal_rule_values(p_hi, hv, bound[pool][over], sb, is_int)
        res = _row_steps(p_hi, hv, rep[pool][over])
        new_text[over] = _render_rule_values(p_hi, hv, rep[pool][over], b, sb, res)
        unit = "ID(s)" if first is not None else "row(s)"
        msgs.append(Message("warning", "rule_out_of_range",
                            f"{len(over)} {unit}: the synthetic low value of a declared rule is above every value "
                            f"of {p_hi.name}, so the smallest value that meets the rule is written there (above "
                            f"the source maximum of {p_hi.name})", column=p_hi.name))
    if dropped:
        unit = "ID(s)" if first is not None else "row(s)"
        msgs.append(Message("info", "rule_values_shifted",
                            f"{dropped} {unit}: no free synthetic value of {p_hi.name} met the declared rule(s), "
                            f"so the smallest value of {p_hi.name} that does is repeated there (its distribution "
                            "shifts by that many values)", column=p_hi.name))
    if keep_ext and (dropped or len(over)):  # D14: a min/max no unit's bounds admit is gone
        held = set(take[have].tolist())
        lost = [w for w, v in (("minimum", sv[0]), ("maximum", sv[-1])) if not any(sv[q] == v for q in held)]
        if lost:  # a warning where the pipeline's D14 check does not look (levels, discrete integers)
            msgs.append(Message(
                "warning" if _label_generated(p_hi) else "info", "rule_extreme_dropped",
                f"the exact {' and '.join(lost)} of {p_hi.name} could not be placed: no synthetic row meets the "
                "declared rule(s) with it (in the source it occurs only where a rule does not hold, or with low "
                "values the independent synthetic ones do not reproduce)", column=p_hi.name))
    out = np.asarray(columns[hi], dtype=object).copy()
    if first is not None:
        pos = np.full(n_units, -1, dtype=np.int64)
        pos[pool] = np.arange(k)
        s = pos[unit_of_row]
        hit = s >= 0
        out[hit] = new_text[s[hit]]
    else:
        out[pool] = new_text
    columns[hi] = out
    if first is None:
        gset = set(np.asarray(gaps).tolist()) if t.get("integer") else None
        _swap_off_real_rows(tp, columns, hi, pool, take, lo_idx, rng,
                            None if gset is None else (sv.tolist(), np.where(bounded, bound[pool], 0).tolist(),
                                                       bounded.tolist(), gset))
    return msgs


def _swap_off_real_rows(tp: Any, columns: dict[int, np.ndarray], hi: int, pool: np.ndarray, take: np.ndarray,
                        lo_idx: np.ndarray, rng: np.random.Generator, gap_info: tuple | None = None) -> None:
    """Rows the deal made equal to a real row (D23) swap their B value with another row whose bounds both
    still hold. Privacy swap-redraw moves a rule's columns together (it cannot split them), so the deal
    removes its own chance matches; B's values only move, as in privacy (up to ``matches.MAX_ROUNDS``, and
    not at all above ``matches.REPORT_ONLY_RATE``: privacy then reports the rate instead, D23). With
    ``gap_info`` (sorted values, bounds, bounded flags, observed integer gaps) a partner that keeps both
    gaps observed is tried first."""
    def gaps_kept(u: int, v: int, tu: int, tv: int) -> bool:
        if gap_info is None:
            return True
        svl, bl, bdl, gset = gap_info
        return ((not bdl[u]) or (svl[tv] - bl[u]) in gset) and ((not bdl[v]) or (svl[tu] - bl[v]) in gset)

    mode, cols = matches.plan(tp)
    real = tp.private.real_row_keys
    k = len(pool)
    if mode != "row" or real is None or hi not in cols or k < 2 or any(c not in columns for c in cols):
        return
    col = columns[hi]
    unit_of = np.full(len(col), -1, dtype=np.int64)
    unit_of[pool] = np.arange(k)
    take = take.copy()
    for round_ in range(matches.MAX_ROUNDS):
        hit = np.isin(matches.row_hashes([columns[c] for c in cols]), real)
        if round_ == 0 and hit.mean() > matches.REPORT_ONLY_RATE:
            return
        units = unit_of[np.flatnonzero(hit)]
        units = units[units >= 0]
        units = units[take[units] >= 0]
        if not len(units):
            return
        moved = False
        for u in rng.permutation(units).tolist():
            for attempt in range(50):
                v = int(rng.integers(k))
                tu, tv = int(take[u]), int(take[v])
                if v == u or tv < 0 or tv < lo_idx[u] or tu < lo_idx[v]:
                    continue
                if attempt < 25 and not gaps_kept(u, v, tu, tv):
                    continue
                r, s = int(pool[u]), int(pool[v])
                if col[r] == col[s]:
                    continue
                col[r], col[s] = col[s], col[r]
                take[u], take[v] = tv, tu
                moved = True
                break
        if not moved:
            return


def _deal_rule_values(lo_idx: np.ndarray, target: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """Sorted position of B's value each unit takes (-1: none).

    B's values are dealt in ascending order, each to the unit with the smallest ``target`` among the units
    it suits (position >= the unit's ``lo_idx``; ties by ``tie``): a monotone matching of values to targets
    wherever the bounds allow. The eligible sets are nested (a value that suits a bound suits every lower
    one), so a value finds no unit only when no arrangement can serve every unit (Hall's condition); the
    fewest possible units are then left without a value.
    """
    import heapq

    k = len(lo_idx)
    take = np.full(k, -1, dtype=np.int64)
    order = np.lexsort((tie, lo_idx))
    lo_s = lo_idx[order].tolist()
    t_l, tie_l = target.tolist(), tie.tolist()
    units = order.tolist()
    heap: list[tuple[float, int, int]] = []
    nxt = 0
    for j in range(k):
        while nxt < k and lo_s[nxt] <= j:
            u = units[nxt]
            heapq.heappush(heap, (t_l[u], tie_l[u], u))
            nxt += 1
        if heap:
            take[heapq.heappop(heap)[2]] = j
    return take


#: Whole-number gaps are matched to their observed counts when they take few values (at most this many): a gap
#: such as Yr_JoinCampus - KCSE in {0, 1, 2, 5}, or an end date 1, 2, 7 or 30 days after a start date.
GAP_REPAIR_DISTINCT = 50


def _repair_gaps(take: np.ndarray, lo_idx: np.ndarray, sv: np.ndarray, bnd: np.ndarray, bounded: np.ndarray,
                 want: np.ndarray, rng: np.random.Generator, tries: int = 12,
                 same: np.ndarray | None = None) -> np.ndarray:
    """Whole-number gaps (numbers, days, months, ticks): bring the achieved gaps to the drawn ones (``want``,
    the exact quota of the observed gaps). A unit whose gap is over-represented (or never observed) swaps its
    value with a unit holding ``bound + g`` for an under-represented gap ``g`` (nearest first) when the swap
    lowers the total excess and both units still meet every rule. Values only swap (B's values unchanged);
    with ``same`` (a key per unit, e.g. its ID block size) only between units of equal key."""
    import bisect

    quota = Counter(np.asarray(want, dtype=float).tolist())
    if not quota or len(take) < 2:
        return take
    if len(quota) > GAP_REPAIR_DISTINCT:
        return take  # gaps spread over many values: their exact counts are no property worth swaps
    k = len(take)
    sv_l = np.asarray(sv, dtype=float).tolist()
    b_l = np.where(bounded, np.asarray(bnd, dtype=float), 0.0).tolist()
    bd_l, lo_l = bounded.tolist(), lo_idx.tolist()
    same_l = None if same is None else np.asarray(same).tolist()
    tk = take.tolist()
    glist = sorted(quota)

    def gap(u: int, pos: int) -> float | None:
        return sv_l[pos] - b_l[u] if bd_l[u] else None

    have: Counter = Counter(gap(u, tk[u]) for u in range(k) if bd_l[u] and tk[u] >= 0)

    def over(x: float | None) -> bool:
        return x is not None and have[x] > quota.get(x, 0)

    bad = [u for u in range(k) if bd_l[u] and tk[u] >= 0 and over(gap(u, tk[u]))]
    if not bad:
        return take
    holders: dict[float, list[int]] = defaultdict(list)
    for u in range(k):
        if tk[u] >= 0:
            holders[sv_l[tk[u]]].append(u)

    def nearest(d: float, only_short: bool) -> list[float]:
        """Observed gaps nearest to ``d`` (walking outward), only under-represented ones if ``only_short``."""
        out: list[float] = []
        r = bisect.bisect_left(glist, d)
        lft = r - 1
        steps = 0
        while len(out) < tries and (lft >= 0 or r < len(glist)) and steps < 8 * tries:
            steps += 1
            if r >= len(glist) or (lft >= 0 and d - glist[lft] <= glist[r] - d):
                x = glist[lft]
                lft -= 1
            else:
                x = glist[r]
                r += 1
            if not only_short or have[x] < quota[x]:
                out.append(x)
        return out

    def better(delta: dict) -> bool:
        """The swap lowers (units with a never-observed gap, units above their gap's quota)."""
        u0 = u1 = e0 = e1 = 0
        for x, s in delta.items():
            h = have[x]
            q = quota.get(x)
            if q is None:
                u0 += h
                u1 += h + s
            else:
                e0 += max(0, h - q)
                e1 += max(0, h + s - q)
        return (u1, e1) < (u0, e0)

    # work budget: a few dozen tries per unit of excess (a swap removes at least one)
    budget = 64 * sum(max(0, c - quota.get(x, 0)) for x, c in have.items()) + 10_000
    for _ in range(3):  # a few passes: a swap can open a later one
        # in random order, never-observed gaps first, then over-represented ones
        bad = [bad[i] for i in rng.permutation(len(bad)).tolist()]
        bad.sort(key=lambda u: gap(u, tk[u]) in quota)
        swapped = 0
        for u in bad:
            pu = tk[u]
            d = gap(u, pu)
            if not over(d):
                continue
            cands = nearest(d, True)  # type: ignore[arg-type]
            if d not in quota:
                cands += [x for x in nearest(d, False) if x not in cands]  # type: ignore[arg-type]
            done = False
            for g in cands:
                hs = holders.get(b_l[u] + g)
                if not hs:
                    continue
                picks = rng.permutation(len(hs)) if len(hs) <= 4 * tries else rng.integers(len(hs), size=4 * tries)
                for j in picks.tolist():
                    budget -= 1
                    w = hs[j]
                    pw = tk[w]
                    if w == u or pw < lo_l[u] or pu < lo_l[w] or (same_l is not None and same_l[w] != same_l[u]):
                        continue
                    delta: dict = {}
                    for x, s in ((d, -1), (g, 1), (gap(w, pw), -1), (gap(w, pu), 1)):
                        if x is not None:
                            delta[x] = delta.get(x, 0) + s
                    if not better(delta):
                        continue
                    for x, s in delta.items():
                        have[x] += s
                    tk[u], tk[w] = pw, pu
                    hs[j] = u
                    holders[sv_l[pu]] = [w if x == u else x for x in holders[sv_l[pu]]]
                    done = True
                    break
                if done:
                    swapped += 1
                    break
            if budget <= 0:
                break
        bad = [u for u in range(k) if bd_l[u] and tk[u] >= 0 and over(gap(u, tk[u]))]
        if not swapped or not bad or budget <= 0:
            break
    return np.asarray(tk, dtype=np.int64)


def _rebalance_rule_rows(tp: Any, hi: int, pool: np.ndarray, take: np.ndarray, lo_idx: np.ndarray,
                         texts: np.ndarray, rng: np.random.Generator, bnd: np.ndarray, sv: np.ndarray,
                         gset: set | None, max_swaps: int = 20_000) -> np.ndarray:
    """A within-ID-constant rule target (D4): after the deal, its values' rows (each entity's value on its ID
    block) are brought back to the source's per level (R2-10) or per value bin (R3-07) by swaps of two
    entities' values that keep every rule on both. The deal pairs values with bounds, not with block sizes.
    A swap moves rows from the label with the most excess to the one with the largest shortfall (a size
    difference below their gap, nearest half of it); it stops when no rule-keeping swap helps. Partners with
    the same bound go first (their gaps just trade places), then partners whose new gaps were both observed
    (whole-number gaps, ``gset``); any rule-keeping partner only when the gaps are not whole numbers."""
    from ..profile.structure import _factorize

    emodel = tp.private.entity_models.get(hi)
    index = tp.private.entity_index
    if emodel is None or index is None:
        return take
    units = np.flatnonzero(take >= 0)
    if len(units) < 2:
        return take
    sizes_all = np.bincount(index, minlength=tp.entity.n_entities)
    size = sizes_all[pool].astype(np.int64)
    k = len(texts)
    bins = emodel.extra.get("_row_bins")
    counts = emodel.extra.get("_row_counts")
    if bins:  # numbers and dates: rank bins with the source's number of IDs per bin
        per_bin = largest_remainder(np.asarray(bins["ids"], dtype=float), k)
        lab_of_pos = np.repeat(np.arange(len(per_bin)), per_bin)
        goal_w = np.asarray(bins["rows"], dtype=float)
    elif counts:  # levels: rows per raw value
        codes, uniq = _factorize(np.asarray(texts, dtype=object))
        lab_of_pos = codes
        goal_w = np.asarray([float(counts.get(u, 0)) for u in uniq.tolist()])
        if not goal_w.any():
            return take
    else:
        return take
    n_lab = len(goal_w)
    tk = take.copy()
    lab = lab_of_pos[tk[units]]
    rows = np.bincount(lab, weights=size[units], minlength=n_lab).astype(np.int64)
    goal = largest_remainder(goal_w, int(rows.sum())).astype(np.int64)
    diff = rows - goal
    members: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(n_lab)]
    for u, lb in zip(units.tolist(), lab.tolist()):
        members[lb][int(size[u])].append(u)
    lo_l = lo_idx.tolist()
    bnd_l = bnd.tolist()
    sv_l = np.asarray(sv, dtype=float).tolist()

    def same_bound(x: int, y: int) -> bool:
        bx, by = bnd_l[x], bnd_l[y]
        return (bx != bx and by != by) or bx == by  # both without a low value, or equal bounds

    def gaps_seen(x: int, y: int) -> bool:
        assert gset is not None
        bx, by = bnd_l[x], bnd_l[y]
        return (bx != bx or (sv_l[tk[y]] - bx) in gset) and (by != by or (sv_l[tk[x]] - by) in gset)

    tiers: list[Callable[[int, int], bool]] = [same_bound, gaps_seen] if gset is not None \
        else [same_bound, lambda x, y: True]
    for _ in range(max_swaps):
        order = np.argsort(-diff, kind="stable")
        done = False
        for a in order.tolist():
            if diff[a] <= 0:
                break
            for b in order[::-1].tolist():
                gap = int(diff[a] - diff[b])
                if gap < 2:
                    break
                pair = None
                for fits in tiers:
                    pair = _rule_swap_pair(members[a], members[b], gap, tk, lo_l, rng, fits)
                    if pair is not None:
                        break
                if pair is None:
                    continue
                x, y = pair
                sx, sy = int(size[x]), int(size[y])
                members[a][sx].remove(x)
                members[b][sy].remove(y)
                members[a][sy].append(y)
                members[b][sx].append(x)
                tk[x], tk[y] = tk[y], tk[x]
                diff[a] -= sx - sy
                diff[b] += sx - sy
                done = True
                break
            if done:
                break
        if not done:
            break
    return tk


def _rule_swap_pair(ma: dict[int, list[int]], mb: dict[int, list[int]], gap: int, tk: np.ndarray, lo_l: list[int],
                    rng: np.random.Generator, fits: Callable[[int, int], bool],
                    tries: int = 8) -> tuple[int, int] | None:
    """An entity x of label a and y of label b with 0 < size(x) - size(y) < gap (nearest gap / 2 first) whose
    values can trade places under every rule (each value meets the other's bounds) and ``fits``."""
    sa = sorted(s for s, v in ma.items() if v)
    sb = sorted(s for s, v in mb.items() if v)
    pairs = [(abs(2 * (x - y) - gap), x, y) for x in sa for y in sb if 0 < x - y < gap]
    pairs.sort()
    for _, s_x, s_y in pairs[:64]:
        xs, ys = ma[s_x], mb[s_y]
        for _ in range(tries):
            x = xs[int(rng.integers(len(xs)))]
            y = ys[int(rng.integers(len(ys)))]
            if tk[y] >= lo_l[x] and tk[x] >= lo_l[y] and fits(x, y):
                return x, y
    return None


def _settle_leftovers(take: np.ndarray, lo_idx: np.ndarray, sv: np.ndarray, k: int,
                      keep_extremes: bool) -> tuple[np.ndarray, int]:
    """Each unit left without a value takes the smallest value of B that meets its bounds (a repeat of B's
    own value, in its range); -1 when none does (its low value is above all of B). The values no unit took
    are dropped, except that B's only copy of its min or max moves to a unit it suits (D14). Returns (take,
    repeats written)."""
    short = np.flatnonzero(take < 0)
    if not len(short):
        return take, 0
    used = np.zeros(k, dtype=bool)
    used[take[take >= 0]] = True
    left = np.flatnonzero(~used)
    fit = short[lo_idx[short] < k]
    take[fit] = lo_idx[fit]
    if keep_extremes and len(left):
        take = _keep_dropped_extremes(take, left, lo_idx, sv, k)
    return take, int(len(fit))


def _keep_dropped_extremes(take: np.ndarray, left: np.ndarray, lo_idx: np.ndarray, sv: np.ndarray,
                           k: int) -> np.ndarray:
    """A dropped value that was B's only min or max moves to a unit it suits whose own value is neither."""
    held = take[take >= 0]
    for ext, pos in ((sv[0], 0), (sv[-1], k - 1)):
        if np.any(sv[held] == ext):
            continue
        at = left[sv[left] == ext]
        if not len(at):
            continue
        ok = (take >= 0) & (lo_idx <= int(at[0]))
        ok &= (sv[np.maximum(take, 0)] != sv[0]) & (sv[np.maximum(take, 0)] != sv[-1])
        cand = np.flatnonzero(ok)
        if len(cand):
            take[int(cand[0])] = int(at[0])
            held = take[take >= 0]
    return take


def _minimal_rule_values(p: ColumnProfile, hv: Any, bnd: np.ndarray, sb: list[tuple[np.ndarray, np.ndarray, bool]],
                         is_int: bool) -> np.ndarray:
    """The smallest value of B's kind that meets every rule (the bound, one step above a strict one)."""
    b = np.asarray(bnd, dtype=np.int64 if is_int else float).copy()
    if is_int:
        for bi, hm, strict in sb:
            low = hm & ((b <= bi) if strict else (b < bi))
            b[low] = bi[low] + (1 if strict else 0)
    else:
        dec = _rule_decimals(p)
        step = 10.0 ** -dec
        b = np.round(np.ceil(b / step - 1e-9) * step, dec)
        for bi, hm, strict in sb:
            low = hm & ((b <= bi) if strict else (b < bi))
            b[low] = np.round(b[low] + step, dec)
    if hv.kind == "time":
        b = np.minimum(b, _day_end(hv))
    return b


def _day_end(hv: Any) -> int:
    from ..detect.dates_formats import US_DAY

    return int(US_DAY * hv.tpu) - 1


def _row_steps(p: ColumnProfile, hv: Any, rows: np.ndarray) -> np.ndarray | None:
    """Per-row render step (model ticks) of a time/datetime/duration column's own layout; None otherwise."""
    if hv.kind not in ("datetime", "time", "duration"):
        return None
    from .dates import row_resolution

    fd = np.where(hv.digits[rows] >= 0, hv.digits[rows], -1).astype(np.int8)
    return np.maximum(row_resolution(p.params, hv.cls[rows], fd, hv.tpu), 1)


def _render_rule_values(p: ColumnProfile, hv: Any, rows: np.ndarray, b: np.ndarray,
                        sb: list[tuple[np.ndarray, np.ndarray, bool]], res: np.ndarray | None) -> np.ndarray:
    """Render the built high values, re-read them, and move any value the written form put below a rule's
    bound up by one step of its own layout (decimals, scientific mantissa, date/time grid); a few rounds."""
    from ..profile.structure import rule_view

    rendered = _render_once(p, hv, rows, b)
    for _ in range(4):
        back = rule_view(rendered, p)
        if back is None:
            break
        bad = np.zeros(len(b), dtype=bool)
        for bi, hm, strict in sb:
            bad |= hm & back.ok & ((back.vals <= bi) if strict else (back.vals < bi))
        if not bad.any():
            break
        if hv.kind == "number":
            step = _number_steps(p, back.vals[bad])
            b[bad] = np.maximum(back.vals[bad] + step, b[bad])
        else:
            b[bad] = back.vals[bad] + (res[bad] if res is not None else 1)
            if hv.kind == "time":
                b[bad] = np.minimum(b[bad], _day_end(hv))
        rendered[bad] = _render_once(p, hv, rows[bad], b[bad])
    return rendered


def _number_steps(p: ColumnProfile, v: np.ndarray) -> np.ndarray:
    sci = p.params.get("scientific")
    if sci:
        mag = np.floor(np.log10(np.where(v == 0, 1.0, np.abs(v))))
        return 10.0 ** (mag - int(sci.get("mantissa_decimals", 0)))
    return np.full(len(v), 10.0 ** -_rule_decimals(p))


#: Formatted numbers (``$1,234.50``), with or without suppression tokens: a rule value is written through the
#: column's learned number format (R5-05), as the generator writes them.
_FORMATTED = frozenset({ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS})


def _rule_decimals(p: ColumnProfile) -> int:
    """Decimals a built rule value is rounded to: the generator's (0 for integer-valued formatted numbers such as
    '$100.00', whose cents are always zero; else the observed decimals)."""
    if p.type in _FORMATTED and p.params.get("integer_valued"):
        return 0
    return int(p.params.get("decimals", 0) or 0)


def _render_once(p: ColumnProfile, hv: Any, rows: np.ndarray, b: np.ndarray) -> np.ndarray:
    if hv.kind == "number":
        if p.type in _FORMATTED:
            return render_numeric_extended(p, np.asarray(b, dtype=float))
        return render_numeric(p, b)
    from ..detect.dates_formats import US_DAY, days_from_civil
    from .dates import GEN_DTYPE, render_dates

    k = len(rows)
    arr = np.zeros(k, dtype=GEN_DTYPE)
    for f in ("text", "tz", "lead", "trail"):
        arr[f] = np.full(k, None, dtype=object)
    arr["fmt"] = hv.cls[rows]
    arr["fd"] = np.where(hv.digits[rows] >= 0, hv.digits[rows], -1)
    arr["tz"] = hv.tz[rows]
    arr["lead"] = hv.lead[rows]
    arr["trail"] = hv.trail[rows]
    ticks = np.asarray(b, dtype=np.int64)
    if hv.kind == "date":
        arr["day"] = ticks
    elif hv.kind == "year_month":
        arr["day"] = days_from_civil(1970 + ticks // 12, ticks % 12 + 1, np.ones(k, dtype=np.int64))
    elif hv.kind == "datetime":
        arr["day"], arr["us"] = np.divmod(ticks, US_DAY * hv.tpu)
    else:  # time of day, duration
        arr["us"] = ticks
    return render_dates(p, arr)


def check_rules(tp: Any, columns: dict[int, np.ndarray], settings: Any) -> list[Message]:
    """Re-check every applied declared rule on the final values (rows where both columns are filled).

    The rule's public status records any violation; a violation FAILs the file (GenerationError) unless
    ``--no-strict``, which turns it into a warning.
    """
    from ..profile.structure import rule_view

    priv = tp.structure.private
    entries = priv.get("declared_entries") or []
    if not entries:
        return []
    data = priv.get("rule_data", {})
    views: dict[int, Any] = {}

    def vw(c: int) -> Any:
        if c not in views:
            views[c] = rule_view(columns[c], tp.columns[c])
        return views[c]

    bad: dict[tuple, int] = {}
    for key in dict.fromkeys(k for _, k in entries):
        d = data.get(key)
        if d is None or d["lo"] not in columns or d["hi"] not in columns:
            continue
        a, b = vw(d["lo"]), vw(d["hi"])
        if a is None or b is None:
            continue
        both = a.ok & b.ok
        bad[key] = int(np.sum(both & ((b.vals <= a.vals) if d["strict"] else (b.vals < a.vals))))
    msgs: list[Message] = []
    texts = []
    for i, key in entries:
        entry = tp.structure.declared_rules[i] if i < len(tp.structure.declared_rules) else None
        if entry is None or key not in bad:
            continue
        nb = bad[key]
        entry.pop("synthetic_rows_violating", None)
        entry["status"] = "applied"
        if nb:
            entry["status"] = f"applied, but violated in {nb} synthetic row(s)"
            entry["synthetic_rows_violating"] = nb
            texts.append(f"{entry['rule']!r} ({nb} row(s))")
    if texts:
        text = "declared rule(s) violated in the synthetic output: " + ", ".join(texts)
        if settings is None or not getattr(settings, "no_strict", False):
            raise GenerationError(text, fix="--no-strict writes the file anyway (the violation becomes a warning), "
                                            "or remove the rule from rules: in --config")
        msgs.append(Message("warning", "rule_violated", text))
    return msgs

def _is_missing(v: Any, p: ColumnProfile) -> bool:
    if v is None:
        return True
    st = v.strip()
    return st == "" or (p.missing_policy == "typed" and st in p.na_tokens)


# --------------------------------------------------------------------------- FDs vs whole-row missingness (R2-09)
_NOT_MAPPED = object()


def align_fd_missingness(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator) -> list[Message]:
    """Deal each root determinant's values so its FD tree's rebuilt dependents keep the sampled D6 patterns.

    A dependent rebuilt from its determinant takes the mapped value, blank or not; the whole-row missingness
    pattern was sampled before, independently of the determinant. Each distinct value of a root determinant
    (one that no applied FD rebuilds) has a *state*: the blank/non-blank state of every dependent in its FD
    tree after the rebuild (None when a value has no observed mapping: those rows keep their own values).
    Each row (entity, for a within-ID-constant determinant; an entity whose rows want different states wants
    none) wants the state its sampled pattern gives those dependents. Rows whose value has another state
    swap values, by exact matching per state, with rows of the same determinant missing state only - the
    determinant's own pattern and every column total are unchanged. Rows left without a match (pooled rare
    patterns, floors) are counted by :func:`apply_fds`.
    """
    priv = tp.structure.private
    fds = [fd for fd in priv.get("fds", [])
           if not fd.get("overridden") and fd["det"] in columns and fd["dep"] in columns]
    if not fds:
        return []
    by_det: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fd in fds:
        by_det[fd["det"]].append(fd)
    deps = {fd["dep"] for fd in fds}
    for root in dict.fromkeys(fd["det"] for fd in fds if fd["det"] not in deps):
        _align_root(tp, columns, root, by_det, rng)
    return []


def _align_root(tp: Any, columns: dict[int, np.ndarray], root: int, by_det: dict[int, list[dict[str, Any]]],
                rng: np.random.Generator) -> None:
    tree: list[dict[str, Any]] = []  # the FD tree under ``root``, determinants before their dependents
    queue, seen = [root], {root}
    while queue:
        c = queue.pop(0)
        for fd in by_det.get(c, []):
            if fd["dep"] not in seen:
                seen.add(fd["dep"])
                tree.append(fd)
                queue.append(fd["dep"])
    if not tree or len(tree) > 60:
        return
    det = columns[root]
    n = len(det)
    codes, uniq = pd.factorize(np.asarray(det, dtype=object), use_na_sentinel=False)
    ustate = np.full(len(uniq), -1, dtype=np.int64)
    for i, u in enumerate(uniq.tolist()):
        val: dict[int, Any] = {root: u if isinstance(u, str) else None}
        bits = 0
        for j, fd in enumerate(tree):
            x = val.get(fd["det"], _NOT_MAPPED)
            if x is _NOT_MAPPED or x not in fd["mapping"]:
                bits = -1
                break
            y = fd["mapping"][x]
            val[fd["dep"]] = y
            if _is_missing(y, tp.columns[fd["dep"]]):
                bits |= 1 << j
        ustate[i] = bits
    state_row = ustate[np.asarray(codes, dtype=np.int64)]
    want_row = np.zeros(n, dtype=np.int64)
    for j, fd in enumerate(tree):
        want_row |= _miss(tp, columns, fd["dep"]).astype(np.int64) << j
    det_miss = _miss(tp, columns, root)
    index = tp.private.entity_index
    first = _first_rows(tp) if tp.columns[root].role == "entity" else None
    if first is not None and index is not None and len(index) == n:
        rep = np.maximum(first, 0)
        lo = np.full(len(first), np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(len(first), -1, dtype=np.int64)
        np.minimum.at(lo, index, want_row)
        np.maximum.at(hi, index, want_row)
        want = np.where((lo == hi) & (first >= 0), lo, -1)
        state, dmiss, vals = state_row[rep], det_miss[rep], det[rep]
    else:
        first = None
        want, state, dmiss, vals = want_row, state_row, det_miss, det
    new = vals.copy()
    moved = False
    for flag in (False, True):  # values move only between units of the same determinant missing state
        units = np.flatnonzero(dmiss == flag)
        w, s = want[units], state[units]
        if not np.any((w >= 0) & (s >= 0) & (w != s)):
            continue
        pool = units[(w < 0) | (s < 0) | (w != s)]
        src = _deal(want[pool], state[pool], rng)
        new[pool] = vals[pool][src]
        moved = True
    if moved:
        columns[root] = new[index] if first is not None else new


def _deal(want: np.ndarray, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each unit, the position of the value it takes (a permutation): exact matches per state first, then
    unmapped values (state -1) to units that want a state, then the rest."""
    k = len(want)
    src = np.full(k, -1, dtype=np.int64)
    used = np.zeros(k, dtype=bool)
    for st in sorted(set(want[want >= 0].tolist()) & set(state[state >= 0].tolist())):
        u = rng.permutation(np.flatnonzero(want == st))
        v = rng.permutation(np.flatnonzero(state == st))
        q = min(len(u), len(v))
        src[u[:q]] = v[:q]
        used[v[:q]] = True
    rest = np.flatnonzero(src < 0)
    if len(rest):
        spec = rng.permutation(rest[want[rest] >= 0])
        anyu = rng.permutation(rest[want[rest] < 0])
        free = np.flatnonzero(~used)
        wild = rng.permutation(free[state[free] < 0])
        known = rng.permutation(free[state[free] >= 0])
        q = min(len(spec), len(wild))
        src[spec[:q]] = wild[:q]
        src[np.concatenate([anyu, spec[q:]])] = np.concatenate([wild[q:], known])
    return src


# --------------------------------------------------------------------------- functional dependencies
def _rank_pairing(real: Counter, syn: Counter) -> dict[Any, Any]:
    """Real value -> synthetic value by frequency rank (ties: first-seen real order, sorted synthetic)."""
    order = {v: i for i, v in enumerate(real)}
    r_sorted = sorted(real.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    s_sorted = sorted(syn.items(), key=lambda kv: (-kv[1], str(kv[0])))
    if not s_sorted:
        return {}
    return {rv: s_sorted[i % len(s_sorted)][0] for i, (rv, _) in enumerate(r_sorted)}


def apply_fds(tp: Any, columns: dict[int, np.ndarray], fds: list[dict[str, Any]] | None = None) -> list[Message]:
    """Rebuild FD dependents from their synthetic determinants (module doc); ``fds`` default to all."""
    msgs: list[Message] = []
    for fd in (tp.structure.private.get("fds", []) if fds is None else fds):
        a, b, mapping = fd["det"], fd["dep"], fd["mapping"]
        if a not in columns or b not in columns or fd.get("overridden"):
            continue
        pb = tp.columns[b]
        det = columns[a]
        dep = columns[b]
        dep_miss = missing_mask(dep, pb.missing_policy, pb.na_tokens)
        if _label_generated(pb):
            g: dict[Any, Any] | None = None
        else:
            if pb.role == "entity" and (first := _first_rows(tp)) is not None:
                syn = Counter(v for v, mi in zip(dep[first].tolist(), dep_miss[first].tolist()) if not mi)
            else:
                syn = Counter(v for v, mi in zip(dep.tolist(), dep_miss.tolist()) if not mi)
            g = _rank_pairing(Counter(pb.model.nonmissing_counts) if pb.model else Counter(), syn)
        out = np.empty(len(det), dtype=object)
        unmapped = 0
        cache: dict[Any, Any] = {}
        for r, v in enumerate(det.tolist()):
            if v in cache:
                out[r] = cache[v]
                continue
            rv = pseudonyms.real_value(tp.columns[a], v)  # D35: a pseudonym stands for its real value
            if rv not in mapping:
                out[r] = dep[r]
                unmapped += 1
                continue
            real_b = mapping[rv]
            if g is None or _is_missing(real_b, pb):
                # the dependent's own written value (its pseudonym when D35 pseudonymised it), or a missing token
                new = real_b if _is_missing(real_b, pb) else pseudonyms.mapped_value(pb, real_b)
            elif real_b in g:
                new = g[real_b]
            else:  # never write a real value of a never-copy column
                out[r] = dep[r]
                unmapped += 1
                continue
            cache[v] = new
            out[r] = new
        columns[b] = out
        conflicts = int(np.sum(missing_mask(out, pb.missing_policy, pb.na_tokens) != dep_miss))
        if _label_generated(pb) and _counts_changed(tp, pb, out):
            pb.flags["rare_category_adjusted"] = True
        if unmapped:
            msgs.append(Message("warning", "fd_unmapped",
                                f"{unmapped} row(s) kept an independent value: their determinant value has no "
                                "observed mapping", column=pb.name))
        if conflicts:
            msgs.append(Message("warning", "fd_missingness_conflict",
                                f"{conflicts} row(s): the blank/non-blank state follows the functional dependency "
                                f"{tp.columns[a].name} -> {pb.name} instead of the sampled whole-row missingness "
                                "pattern (the synthetic determinant values could not all be matched to it)",
                                column=pb.name))
    return msgs


def _counts_changed(tp: Any, p: ColumnProfile, values: np.ndarray) -> bool:
    model = tp.private.entity_models.get(p.index) if p.role == "entity" else p.model
    if model is None:
        return False
    if p.role == "entity":
        first = _first_rows(tp)
        if first is None:
            return False
        values = values[first]
    real = Counter(pseudonyms.mapped_counts(model.nonmissing_counts, pseudonyms.level_map(p))) \
        + Counter(model.missing_counts)
    return Counter(values.tolist()) != real


def _units_for_entities(tp: Any, rows: np.ndarray) -> np.ndarray:
    index = tp.private.entity_index
    if index is None:
        return rows
    return index[rows]


def pinned_templates(tp: Any) -> list[dict[str, Any]]:
    """Templates with an input whose role differs from the target's (an ID, or a within-ID constant).

    Row-level moves (privacy swaps, composite keys, duplicate rows) carry the
    target with the inputs of its own role only, so such a template would break:
    it is rebuilt after the moves (:func:`finalize`) and its columns never move
    for duplicate rows.
    """
    out = []
    targets: set[int] = set()
    for info in tp.structure.private.get("templates", []):  # application order: inputs before their users
        role = tp.columns[info["target"]].role
        inputs = [p[1] for p in info["parts"] if p[0] == "f"]
        if any(tp.columns[c].role != role or c in targets for c in inputs):
            out.append(info)
            targets.add(info["target"])
    return out


def apply_templates(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator,
                    infos: list[dict[str, Any]] | None = None) -> list[Message]:
    """Rebuild every derived template's target from its synthetic inputs (module doc, step 3).

    R5-04: the rebuilt column keeps the source's distinct count and frequency profile. Inputs drawn independently
    pair up differently from the source (two staff members drawn the same first and last name share an e-mail;
    a repeated source e-mail splits over several name pairs), so when the rebuilt values' profile differs from
    the source's, the inputs are regrouped first (:func:`regroup_template`). With ``infos`` (the pinned
    templates' rebuild after the row moves, :func:`finalize`) the values are only re-rendered.
    """
    msgs: list[Message] = []
    todo = tp.structure.private.get("templates", []) if infos is None else infos
    fixed = _regroup_fixed_columns(tp) if infos is None else None
    done: list[dict[str, Any]] = []
    for info in todo:
        m, changed = _apply_template(tp, columns, info, rng, fixed)
        msgs += m
        if changed:
            assert fixed is not None
            fixed |= changed  # inputs are regrouped once
            for prev in done:  # an earlier template reading a regrouped input is rendered again from it
                if changed & {p[1] for p in prev["parts"] if p[0] == "f"}:
                    _apply_template(tp, columns, prev, rng, None)
        done.append(info)
    return msgs


def _regroup_fixed_columns(tp: Any) -> set[int]:
    """Columns a template regroup must not rewrite: IDs, FD columns, declared rule columns, composite-key columns
    and template targets (each keeps its own structure)."""
    priv = tp.structure.private
    fixed = {p.index for p in tp.columns if p.role not in ("row", "entity")}
    for fd in priv.get("fds", []):
        if not fd.get("overridden"):
            fixed |= {fd["det"], fd["dep"]}
    for key in priv.get("declared", []) or []:
        d = priv.get("rule_data", {}).get(key)
        if d:
            fixed |= {d["lo"], d["hi"]}
    for ck in priv.get("composite", []):
        fixed |= set(ck.get("cols") or [ck["x"]])
    fixed |= {t["target"] for t in priv.get("templates", [])}
    return fixed


def _apply_template(tp: Any, columns: dict[int, np.ndarray], info: dict[str, Any], rng: np.random.Generator,
                    fixed: set[int] | None) -> tuple[list[Message], set[int]]:
    """One template (see :func:`apply_templates`); returns (messages, input columns regrouped)."""
    msgs: list[Message] = []
    t = info["target"]
    if t not in columns:
        return msgs, set()
    pt = tp.columns[t]
    parts = info["parts"]
    cols = sorted({p[1] for p in parts if p[0] == "f"})
    if any(c not in columns for c in cols):
        if info.get(_SHARING_REBUILT):  # never its own generated values (issue #30)
            columns[t] = np.full(len(columns[t]), None, dtype=object)
        return msgs, set()
    vals = columns[t]
    miss = missing_mask(vals, "typed", pt.na_tokens)  # placeholders such as 'N/A' are kept as they are
    rows = np.flatnonzero(~miss)
    if not len(rows):
        return msgs, set()
    in_miss = {c: missing_mask(columns[c], tp.columns[c].missing_policy, tp.columns[c].na_tokens) for c in cols}
    # per-row draws (free digits, domains); per entity for a column constant within the ID
    per_entity = pt.role == "entity" and tp.private.entity_index is not None
    units = _units_for_entities(tp, rows) if per_entity else np.arange(len(rows))
    uniq, inv = np.unique(units, return_inverse=True)
    k = len(uniq)
    digits = None
    if "digits" in info:
        lens, cnts = zip(*sorted(info["digits"].items()))
        dl, _ = quota_sample(list(lens), list(cnts), k, rng, floor=False)
        digits = ["".join(str(int(x)) for x in rng.integers(0, 10, size=int(n))) for n in dl.tolist()]
    domains = None
    if info["email"]:
        names, cnts = zip(*info["domains"])
        domains, _ = quota_sample(list(names), list(cnts), k, rng, floor=False)
    real = pt.model.real_values if pt.model is not None and pt.model.real_values is not None else frozenset()
    real_cf = {v.casefold() for v in real}
    # a row whose text would hold a real phone number or SSN of the run keeps its independently generated
    # value (privacy.digits): the inputs' digits glued together (a synthetic surname's digits + a synthetic
    # ID) can equal one by chance
    digit_keys = tp.private.extra.get("run_digit_keys") if getattr(tp, "private", None) is not None else None
    out = vals.copy()
    # vectorised: transforms run once per distinct input value
    stripped = {c: _stripped(columns[c][rows]) for c in cols}
    text, _ = tpl.render_array(parts, stripped)
    if digits is not None:
        suffix = np.empty(k, dtype=object)
        suffix[:] = digits
        text = text + suffix[inv]
    if domains is not None:
        at = np.empty(k, dtype=object)
        at[:] = ["@" + str(d) for d in domains.tolist()]
        text = text + at[inv]
    skip = np.zeros(len(rows), dtype=bool)
    for c in cols:
        skip |= in_miss[c][rows]
    changed: set[int] = set()
    groups = info.get("groups")
    if fixed is not None and groups is not None and _profile_differs(tp, groups, text, rows, skip, per_entity):
        res = None
        if not set(cols) & fixed and (groups["unit"] == "entity") == per_entity:
            res = regroup_template(tp, columns, info, rows, skip, per_entity, rng, real_cf, digit_keys)
        if res is not None:
            text = res
            changed = set(cols)
        else:  # report.fidelity: reported, not checked
            pt.params["_template_profile"] = ("an input of the template keeps another structure (a functional "
                                              "dependency, declared rule, composite key or template)")
    codes, uniq = pd.factorize(text, use_na_sentinel=False)
    codes = np.asarray(codes, dtype=np.int64)
    uniq_l = uniq.tolist()
    # a rebuilt value equal to a real value of the column, or to a real text of any column of the run that the run
    # never writes (privacy.texts, R7-01/R7-02), keeps its independently generated value
    texts = _run_texts(tp)
    rebuilt = bool(info.get(_SHARING_REBUILT))
    # sharing mode (issue #30): nor may an address have the local part of a real one (the leak scan's e-mail rule)
    real_locals = {v.rpartition("@")[0] for v in real_cf if "@" in v} if rebuilt and info["email"] else set()
    hit_u = np.fromiter((u.casefold() in real_cf or (texts is not None and texts.holds(u))
                         or (bool(real_locals) and u.casefold().rpartition("@")[0] in real_locals) for u in uniq_l),
                        dtype=bool, count=len(uniq))
    hit = hit_u[codes] & ~skip
    held = np.zeros(len(rows), dtype=bool)
    if digit_keys:
        held_u = np.fromiter((digit_keys.holds_real(u) for u in uniq_l), dtype=bool, count=len(uniq))
        held = held_u[codes] & ~skip & ~hit
    skipped, collided, n_held = int(skip.sum()), int(hit.sum()), int(held.sum())
    ok = ~skip & ~hit & ~held
    out[rows[ok]] = text[ok]
    if rebuilt:
        # sharing mode writes this column from the template only: its own generated values are never released, so a
        # row the template cannot fill safely is written empty
        out[rows[~ok]] = None
    columns[t] = out
    # rows keeping an independent value (not from the template): the fidelity check allows for them
    pt.params["_template_rows_kept"] = skipped + collided + n_held
    if skipped or collided or n_held:
        also = f", {n_held} that would hold a real phone number or SSN of the run" if n_held else ""
        msgs.append(Message("warning", "template_partly_applied",
                            f"derived template not applied in {skipped + collided + n_held} row(s) "
                            f"({skipped} with a missing input, {collided} that would equal a real value{also}); "
                            + ("those rows are written empty" if rebuilt
                               else "those rows keep an independently generated value"), column=pt.name))
    return msgs, changed


def _unit_rows(tp: Any, rows: np.ndarray, per_entity: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """(position in ``rows`` of each unit's representative row, unit of each row of ``rows``); units are the rows
    themselves, or the IDs (in ID order) for a column constant within the ID."""
    if not per_entity:
        return np.arange(len(rows)), None
    ents = tp.private.entity_index[rows]
    _, first, inv = np.unique(ents, return_index=True, return_inverse=True)
    return first, inv


def _profile_differs(tp: Any, groups: dict[str, Any], text: np.ndarray, rows: np.ndarray, skip: np.ndarray,
                     per_entity: bool) -> bool:
    """True when the rebuilt values' frequency profile (units per distinct value, sorted) is not the source's
    (scaled to the number of synthetic units)."""
    ok = np.flatnonzero(~skip)
    if not len(ok):
        return False
    first, _ = _unit_rows(tp, rows[ok], per_entity)
    codes, _ = pd.factorize(np.asarray(text[ok][first], dtype=object), use_na_sentinel=False)
    syn = np.sort(np.bincount(np.asarray(codes, dtype=np.int64)))[::-1]
    real = np.asarray(groups["units"], dtype=np.int64)
    if int(real.sum()) != int(syn.sum()) and real.sum():
        real = largest_remainder(real.astype(float), int(syn.sum()))
    real = np.sort(real[real > 0])[::-1]
    return len(real) != len(syn) or bool(np.any(real != syn))


def regroup_template(tp: Any, columns: dict[int, np.ndarray], info: dict[str, Any], rows: np.ndarray,
                     skip: np.ndarray, per_entity: bool, rng: np.random.Generator, real_cf: set[str],
                     digit_keys: Any) -> np.ndarray | None:
    """Regroup a template's inputs so its rebuilt values keep the source's groups (R5-04); returns the new
    target text for ``rows`` (rows with a missing input keep theirs) after rewriting the inputs.

    The source's groups (one per distinct target value, ``profile.structure._template_groups``) are dealt to the
    synthetic units (rows, or IDs matched by block size for a column constant within the ID): every unit of a
    group gets the same inputs, so the target has the source's number of distinct values and units per value.
    Each input keeps its synthetic values and their counts: a group takes the synthetic value of the frequency
    rank its source value had. Which values share a group is not copied from the source: each input's ranks
    are shuffled among groups of the same size, then repeated combinations are split by swaps (two groups never
    share all their inputs). Digit suffixes keep each group's length and e-mail domains each group's reserved
    domain; a value equal to a real one (or holding a real phone number or SSN) gets new digits where it has
    them. Names stay synthetic values of their own columns, so no real value is written."""
    g = info["groups"]
    cols = list(g["inputs"])
    ok = np.flatnonzero(~skip)
    if not len(ok) or not cols:
        return None
    ok_rows = rows[ok]
    first, _ = _unit_rows(tp, ok_rows, per_entity)
    unit_rows = ok_rows[first]
    n_units = len(unit_rows)
    n_groups = len(g["units"])
    if per_entity:
        index = tp.private.entity_index
        usize = np.bincount(index)[index[unit_rows]]
        rsize, rgroup = np.asarray(g["unit_size"]), np.asarray(g["unit_group"])
        ro = np.lexsort((rng.random(len(rsize)), -rsize))
        so = np.lexsort((rng.random(n_units), -usize))
        unit_group = np.empty(n_units, dtype=np.int64)
        unit_group[so] = rgroup[ro[np.arange(n_units) % len(ro)]]
        cls = np.bincount(rgroup, minlength=n_groups)  # groups of the same number of IDs trade their inputs
    else:
        cnt = np.asarray(g["units"], dtype=np.int64)
        if int(cnt.sum()) != n_units:
            cnt = largest_remainder(cnt.astype(float), n_units)
        unit_group = rng.permutation(np.repeat(np.arange(n_groups), cnt))
        cls = cnt  # groups of the same number of rows trade their inputs (each input keeps its counts)
    used = np.unique(unit_group)
    source_ranks = np.asarray(g["ranks"], dtype=np.int64)
    ranks = source_ranks.copy()
    by_cls: dict[int, list[int]] = defaultdict(list)
    for gi in used.tolist():
        by_cls[int(cls[gi])].append(gi)
    members = [np.asarray(m, dtype=np.int64) for m in by_cls.values()]
    for j in range(len(cols)):  # decorrelate: an input's ranks move among groups of one size
        for m in members:
            if len(m) > 1:
                ranks[m, j] = ranks[rng.permutation(m), j]
    _split_shared_inputs(ranks, members, rng)

    def shared(r: np.ndarray) -> int:
        return len(used) - len({tuple(x) for x in r[used].tolist()})

    if shared(ranks) > shared(source_ranks):  # no split found: the source's combinations of ranks instead
        ranks = source_ranks.copy()
    # each input's synthetic values by frequency rank (counts over the units), with their raw forms
    gvals: list[np.ndarray] = []
    gstrip: dict[int, np.ndarray] = {}
    for j, c in enumerate(cols):
        raw = np.asarray(columns[c], dtype=object)[unit_rows]
        st = _stripped(raw)
        vcode, vuniq = pd.factorize(st, use_na_sentinel=False)
        vcode = np.asarray(vcode, dtype=np.int64)
        cnt_v = np.bincount(vcode)
        order = np.lexsort((np.arange(len(cnt_v)), -cnt_v))  # rank -> value code
        pools: list[list[Any]] = [[] for _ in range(len(cnt_v))]
        for u in rng.permutation(n_units).tolist():
            pools[int(vcode[u])].append(raw[u])
        code_of_group = order[ranks[:, j] % len(order)]
        gstrip[c] = np.asarray(vuniq, dtype=object)[code_of_group]
        # per unit: a raw form of its group's value (the value's own case/whitespace variants, reused when short)
        per_unit = np.empty(n_units, dtype=object)
        for u in range(n_units):
            cc = int(code_of_group[unit_group[u]])
            pool = pools[cc]
            per_unit[u] = pool.pop() if len(pool) > 1 else (pool[0] if pool else vuniq[cc])
        gvals.append(per_unit)
    base, _ = tpl.render_array(info["parts"], gstrip)
    dlen = g.get("dlen")
    dom = g.get("dom")
    domains = [d for d, _ in info.get("domains", [])]
    gtext = np.empty(n_groups, dtype=object)
    taken: set[str] = set()
    texts = _run_texts(tp)
    for gi in used.tolist():
        at = ("@" + domains[int(dom[gi]) % len(domains)]) if dom is not None and domains else ""
        n_dig = int(dlen[gi]) if dlen is not None else 0
        text = ""
        for _ in range(40):
            suffix = "".join(str(int(x)) for x in rng.integers(0, 10, size=n_dig)) if n_dig else ""
            text = str(base[gi]) + suffix + at
            bad = (text.casefold() in real_cf or text in taken or bool(digit_keys and digit_keys.holds_real(text))
                   or bool(texts is not None and texts.holds(text)))
            if not bad or not n_dig:
                break
        taken.add(text)
        gtext[gi] = text
    # write the inputs: every unit's rows (each ID's rows where the column is filled); the input's other filled
    # rows (IDs) take what the units left of its values, so the column keeps its synthetic values and counts
    if per_entity:
        index = tp.private.entity_index
        ent_unit = np.full(int(index.max()) + 1, -1, dtype=np.int64)
        ent_unit[index[unit_rows]] = np.arange(n_units)
        ru = ent_unit[index]
        for j, c in enumerate(cols):
            pc = tp.columns[c]
            filled = ~missing_mask(columns[c], pc.missing_policy, pc.na_tokens)
            ents, first_row = np.unique(index[filled], return_index=True)
            rep = np.flatnonzero(filled)[first_row]
            other = ents[ent_unit[ents] < 0]
            left = _leftover(np.asarray(columns[c], dtype=object)[rep], gvals[j], len(other), rng)
            sel = np.flatnonzero((ru >= 0) & filled)
            columns[c][sel] = gvals[j][ru[sel]]
            if left is not None and len(other):
                ent_val = np.empty(int(index.max()) + 1, dtype=object)
                ent_val[other] = left
                orows = np.flatnonzero(filled & np.isin(index, other))
                columns[c][orows] = ent_val[index[orows]]
        row_group = unit_group[ent_unit[index[ok_rows]]]
    else:
        for j, c in enumerate(cols):
            pc = tp.columns[c]
            filled = ~missing_mask(columns[c], pc.missing_policy, pc.na_tokens)
            is_unit = np.zeros(len(filled), dtype=bool)
            is_unit[unit_rows] = True
            other = np.flatnonzero(filled & ~is_unit)
            left = _leftover(np.asarray(columns[c], dtype=object)[np.flatnonzero(filled)], gvals[j], len(other), rng)
            columns[c][unit_rows] = gvals[j]
            if left is not None and len(other):
                columns[c][other] = left
        row_group = unit_group
    out = np.asarray(columns[info["target"]], dtype=object)[rows].copy()
    out[ok] = gtext[row_group]
    return out


def _leftover(all_vals: np.ndarray, used: np.ndarray, n_other: int, rng: np.random.Generator) -> np.ndarray | None:
    """The values of ``all_vals`` (a column's filled values) that ``used`` does not take, shuffled, when they are
    exactly ``n_other`` (else None: the other rows keep theirs)."""
    left = Counter(all_vals.tolist())
    left.subtract(Counter(used.tolist()))
    if any(v < 0 for v in left.values()) or sum(left.values()) != n_other:
        return None
    out = np.empty(n_other, dtype=object)
    out[:] = [v for v, c in left.items() for _ in range(c)]
    return out[rng.permutation(n_other)] if n_other else out


def _split_shared_inputs(ranks: np.ndarray, members: list[np.ndarray], rng: np.random.Generator) -> None:
    """Until no two groups share every input (in place, best effort; distinct groups keep distinct values): a
    group sharing its inputs with another swaps one input's rank with a group of its own size (so every input
    keeps its counts), creating no new shared combination."""
    k = ranks.shape[1]
    if k == 0:
        return
    groups = [int(gi) for m in members for gi in m.tolist()]
    cls_of = {int(gi): m for m in members for gi in m.tolist()}
    cnt: Counter = Counter(tuple(ranks[gi].tolist()) for gi in groups)
    holders: dict[tuple, list[int]] = defaultdict(list)
    for gi in groups:
        holders[tuple(ranks[gi].tolist())].append(gi)
    for tg0, hs in list(holders.items()):
        if len(hs) < 2:
            continue
        # every holder but one moves; a holder alone in its size class cannot, so the others go first
        for gi in sorted(hs, key=lambda x: -len(cls_of[x])):
            tg = tuple(ranks[gi].tolist())
            if cnt[tg] <= 1:
                break
            m = cls_of[gi]
            if len(m) < 2:
                continue
            for _ in range(60 * k):
                h = int(m[int(rng.integers(len(m)))])
                j = int(rng.integers(k))
                if h == gi or ranks[h, j] == ranks[gi, j]:
                    continue
                th = tuple(ranks[h].tolist())
                ng = list(tg)
                nh = list(th)
                ng[j], nh[j] = th[j], tg[j]
                ng_t, nh_t = tuple(ng), tuple(nh)
                if ng_t == nh_t or cnt[ng_t] > 0 or cnt[nh_t] > 0:
                    continue
                cnt[tg] -= 1
                cnt[th] -= 1
                cnt[ng_t] += 1
                cnt[nh_t] += 1
                ranks[gi, j], ranks[h, j] = ranks[h, j], ranks[gi, j]
                break


def _stripped(values: np.ndarray) -> np.ndarray:
    """Stripped copy of an object array of strings (None kept), stripping once per distinct value."""
    codes, uniq = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    s = np.empty(len(uniq), dtype=object)
    s[:] = [u.strip() if isinstance(u, str) else None for u in uniq.tolist()]
    return s[np.asarray(codes, dtype=np.int64)]


def _group_of(tp: Any, col: int, role: str) -> tuple[int, ...]:
    for g in tp.structure.swap_groups:
        if col in g:
            return tuple(c for c in g if tp.columns[c].role == role)
    return (col,)


def _key_setup(tp: Any, ck: dict[str, Any], columns: dict[int, np.ndarray]
               ) -> tuple[list[int], tuple[int, ...], np.ndarray, np.ndarray, str] | None:
    """(key columns, columns that move with them, slots where every key column is filled, key code per slot,
    label). Key codes number the stripped key tuples in order of first appearance (:func:`_key_codes`)."""
    cols = list(ck.get("cols") or [ck["x"]])
    if any(c not in columns for c in cols):
        return None
    unit = tuple(dict.fromkeys(c2 for c in cols for c2 in _group_of(tp, c, "row")))
    if any(c not in columns for c in unit):
        return None
    miss = np.zeros(len(columns[cols[0]]), dtype=bool)
    for c in cols:
        miss |= missing_mask(columns[c], tp.columns[c].missing_policy, tp.columns[c].na_tokens)
    slots = np.flatnonzero(~miss)
    codes = _key_codes([_col_codes(columns[c][slots]) for c in cols])
    label = f"({tp.columns[tp.entity.primary].name}, {', '.join(tp.columns[c].name for c in cols)})"
    return cols, unit, slots, codes, label


def _col_codes(values: np.ndarray) -> np.ndarray:
    """Code of each value's stripped text, numbered in order of first appearance."""
    codes, uniq = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    s = np.empty(len(uniq), dtype=object)
    s[:] = [u.strip() if isinstance(u, str) else None for u in uniq.tolist()]
    sc, _ = pd.factorize(s, use_na_sentinel=False)
    return np.asarray(sc, dtype=np.int64)[np.asarray(codes, dtype=np.int64)]


def _key_codes(parts: list[np.ndarray]) -> np.ndarray:
    """One code per row for the tuple of per-column codes (first-appearance order, as the tuples themselves)."""
    if len(parts) == 1:
        return parts[0]
    comb = parts[0].astype(np.int64)
    for p in parts[1:]:
        comb = comb * (int(p.max()) + 1 if len(p) else 1) + p
    codes, _ = pd.factorize(comb)
    return np.asarray(codes, dtype=np.int64)


def _dup_rows(index: np.ndarray, slots: np.ndarray, codes: np.ndarray) -> int:
    """Rows that repeat their ID block's key (each repeat after the first)."""
    if len(slots) < 2:
        return 0
    kc = np.asarray(codes, dtype=np.int64)
    pair = index[slots].astype(np.int64) * (int(kc.max()) + 1) + kc
    return int(len(pair) - len(np.unique(pair)))


def _gale_ryser_fails(codes: np.ndarray, blocks: np.ndarray) -> bool:
    """True when no assignment of these key values to the ID blocks (their slot counts) avoids a repeat: the k
    most frequent values need more slots than the blocks can give k distinct values (Gale-Ryser)."""
    c = np.sort(np.bincount(codes))[::-1]
    s = np.sort(np.bincount(blocks)[np.bincount(blocks) > 0])
    ks = np.arange(1, len(c) + 1)
    # sum over blocks of min(size, k), for every k
    j = np.searchsorted(s, ks, side="right")
    below = np.concatenate([[0], np.cumsum(s)])[j]
    cap = below + ks * (len(s) - j)
    return bool(np.any(np.cumsum(c) > cap))


def _key_status(tp: Any, ck: dict[str, Any], dups: int) -> None:
    """The key's public status (R5-03): 'enforced' only while no synthetic row repeats it."""
    i = ck.get("pub")
    keys = tp.structure.composite_keys
    if i is None or not 0 <= i < len(keys):
        return
    entry = keys[i]
    entry.pop("synthetic_rows_violating", None)
    entry["status"] = "enforced"
    if dups:
        entry["status"] = f"enforced, but violated in {dups} synthetic row(s)"
        entry["synthetic_rows_violating"] = int(dups)


def _key_warning(tp: Any, ck: dict[str, Any], label: str, dups: int, when: str = "") -> Message:
    cols = list(ck.get("cols") or [ck["x"]])
    return Message("warning", "composite_key_duplicates",
                   f"{dups} row(s) repeat a {label} combination{when}: the composite key is NOT kept in those rows "
                   "(no arrangement found of the synthetic key values, each kept at its count, that gives every ID "
                   "distinct values)", column=tp.columns[cols[0]].name)


def enforce_composite_keys(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator) -> list[Message]:
    """No ID block repeats a key value (see module doc), by construction where the values allow it (R5-03).

    1. Greedy assignment of whole rows to ID blocks (:func:`_assign_slots`): the rows move whole
       (:func:`_moved_columns`), so every row keeps its sampled blank/non-blank pattern and its functional
       dependencies, rules and templates (R2-09); a row only moves to a slot with the same state of the
       columns that stay (:func:`_missing_states`: within-ID constants, another composite key).
    2. What that leaves (:func:`_settle_key`): only moving rows cannot help when independent generation made
       too many copies of a key value - a (course, term) pair drawn 46 times for 40 students, or 17 distinct
       dates for one student's 20 visits. Pair keys are then rebuilt block by block from both columns'
       values (:func:`_construct_pairs`: every value keeps its count), rows and key values swap between
       blocks (:func:`_swap_repair`, rows with a blank key included), and an interpolated date that still
       repeats moves to the nearest free date (:func:`_redraw_key_values`).
    The public status says 'enforced' only when no synthetic row repeats the key; otherwise it gives the
    count and the file is warned about (never silent).
    """
    msgs: list[Message] = []
    index = tp.private.entity_index
    if index is None:
        return msgs
    for ck in tp.structure.private.get("composite", []):
        setup = _key_setup(tp, ck, columns)
        if setup is None:
            continue
        cols, _, slots, codes, label = setup
        if not _dup_rows(index, slots, codes):
            _key_status(tp, ck, 0)
            continue  # already unique: keep the independent draw
        blocks = index[slots]
        moved = _moved_columns(tp, columns, ck)
        # whole rows cannot help when too many rows hold the same pair (Gale-Ryser): go straight to rebuilding
        hopeless = len(cols) == 2 and _gale_ryser_fails(codes, blocks) and _pair_units(tp, ck, columns, slots)
        if not hopeless:
            pairs, _ = _assign_slots(slots, codes.tolist(), blocks, rng,
                                     _missing_states(tp, columns, _staying_columns(tp, columns, moved), slots))
            if pairs:
                dest = np.fromiter((p for p, _ in pairs), dtype=np.int64, count=len(pairs))
                src = slots[np.fromiter((i for _, i in pairs), dtype=np.int64, count=len(pairs))]
                for c in moved:
                    columns[c][dest] = columns[c][src].copy()
        left = _settle_key(tp, columns, ck, moved, rng, None)
        _key_status(tp, ck, left)
        if left:
            msgs.append(_key_warning(tp, ck, label, left))
    return msgs


def _settle_key(tp: Any, columns: dict[int, np.ndarray], ck: dict[str, Any], moved: tuple[int, ...],
                rng: np.random.Generator, checker: _RealRowChecker | None) -> int:
    """Remove the key repeats moving whole rows left (R5-03); returns the repeats that remain.

    Before privacy (``checker`` None) a pair key is first rebuilt block by block (:func:`_construct_pairs`,
    kept only when it leaves fewer repeats); then swaps (:func:`_swap_repair`) and, for interpolated dates,
    new values (:func:`_redraw_key_values`). After privacy every move is checked against the real rows.
    """
    index = tp.private.entity_index

    def count() -> int:
        s = _key_setup(tp, ck, columns)
        return 0 if s is None else _dup_rows(index, s[2], s[3])

    dups = count()
    if dups and checker is None and len(ck.get("cols") or []) == 2:
        saved = {c: columns[c].copy() for c in dict.fromkeys(c2 for c in ck["cols"] for c2 in _group_of(tp, c, "row"))
                 if c in columns}
        if _construct_pairs(tp, columns, ck, rng):
            after = count()
            if after > dups:  # never worse than moving whole rows
                for c, v in saved.items():
                    columns[c] = v
            else:
                dups = after
    if dups:
        _swap_repair(tp, columns, ck, moved, rng, checker)
        dups = count()
    if dups:
        _redraw_key_values(tp, columns, ck, checker)
        dups = count()
    return dups


def _proportional(rem: list[int], active: list[int], total: int, s: int, pyr: random.Random) -> dict[int, int]:
    """``s`` units over the ``active`` levels (those with values left) in proportion to their remaining counts
    ``rem`` (exact integer largest remainder, random ties); never more than a level has left."""
    s = min(s, total)
    if s <= 0:
        return {}
    out: dict[int, int] = {}
    frac: list[tuple[int, float, int]] = []
    given = 0
    for i in active:
        q, r = divmod(rem[i] * s, total)
        if q:
            out[i] = q
            given += q
        if r:
            frac.append((-r, pyr.random(), i))
    for _, _, i in heapq.nsmallest(s - given, frac):
        out[i] = out.get(i, 0) + 1
    return out


def _fit_margins(x: dict[int, int], y: dict[int, int], rem_x: list[int], rem_y: list[int],
                 act_x: list[int], act_y: list[int]) -> None:
    """Make one block's level counts of A (``x``) and B (``y``) pairable into distinct (A, B) pairs where the
    remaining values allow it (in place): no A level more often than the block has distinct B levels, and the
    other way round; the excess goes to the levels with the most values left."""
    for _ in range(6):
        changed = False
        for a, b, rem, act in ((x, y, rem_x, act_x), (y, x, rem_y, act_y)):
            cap = len(b)
            over = {i: c - cap for i, c in a.items() if c > cap}
            excess = sum(over.values())
            if not excess:
                continue
            changed = True
            for i, e in over.items():
                a[i] -= e
            for i in sorted(act, key=lambda j: a.get(j, 0) - rem[j]):
                room = min(rem[i], cap) - a.get(i, 0)
                if room > 0:
                    give = min(room, excess)
                    a[i] = a.get(i, 0) + give
                    excess -= give
                    if not excess:
                        break
            for i, e in over.items():  # nowhere else to go: kept (the pairing then repeats a pair)
                if not excess:
                    break
                back = min(e, excess)
                a[i] += back
                excess -= back
        if not changed:
            return


def _distinct_pairs(x: dict[int, int], y: dict[int, int], pyr: random.Random) -> tuple[list[int], list[int]]:
    """Pairs (A level, B level) with ``x[a]`` pairs per A level and ``y[b]`` per B level, all distinct when
    possible: each A level (most pairs first) takes the B levels with the most pairs left (Gale-Ryser greedy).
    What cannot be paired distinctly is paired anyway (a repeat)."""
    ycap = dict(y)
    pa: list[int] = []
    pb: list[int] = []
    short: list[int] = []
    for a in sorted(x, key=lambda i: (-x[i], pyr.random())):
        k = x[a]
        cand = [b for b, c in ycap.items() if c > 0]
        if len(cand) > k:
            cand = [b for _, _, b in heapq.nsmallest(k, ((-ycap[b], pyr.random(), b) for b in cand))]
        for b in cand:
            ycap[b] -= 1
        pa += [a] * len(cand)
        pb += cand
        short += [a] * (k - len(cand))
    if short:
        rest = [b for b, c in ycap.items() for _ in range(c)]
        pa += short[:len(rest)]
        pb += rest[:len(short)]
    return pa, pb


def _pair_units(tp: Any, ck: dict[str, Any], columns: dict[int, np.ndarray],
                slots: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """The two key columns of a pair key with the row-level columns of their swap groups, when each can move on
    its own (not tied to the other by a rule or template) and keeps every row's blank/non-blank pattern (one
    state over the key's slots); else None."""
    cols = list(ck.get("cols") or [])
    if len(cols) != 2:
        return None
    ua = tuple(dict.fromkeys(_group_of(tp, cols[0], "row")))
    ub = tuple(dict.fromkeys(_group_of(tp, cols[1], "row")))
    if set(ua) & set(ub) or any(c not in columns for c in ua + ub):
        return None
    if _missing_states(tp, columns, ua, slots) is not None or _missing_states(tp, columns, ub, slots) is not None:
        return None
    return ua, ub


def _construct_pairs(tp: Any, columns: dict[int, np.ndarray], ck: dict[str, Any], rng: np.random.Generator) -> bool:
    """Rebuild a pair key (ID, A, B) block by block (R5-03): each ID block takes its share of A's and of B's
    values (in proportion to what is left, largest blocks first) and pairs them into distinct (A, B) pairs.

    Only A's and B's values move (each with the row-level columns of its swap group, :func:`_pair_units`): both
    columns keep every value's count and every row its blank/non-blank pattern. Returns False when the key
    cannot be rebuilt this way."""
    setup = _key_setup(tp, ck, columns)
    if setup is None or len(setup[2]) < 2:
        return False
    cols, _, slots, _, _ = setup
    units = _pair_units(tp, ck, columns, slots)
    if units is None:
        return False
    ua, ub = units
    index = tp.private.entity_index
    pyr = random.Random(int(rng.integers(2**62)))
    ka = _col_codes(columns[cols[0]][slots])
    kb = _col_codes(columns[cols[1]][slots])
    rem_a = np.bincount(ka).tolist()
    rem_b = np.bincount(kb).tolist()
    perm = rng.permutation(len(slots)).tolist()
    pool_a: list[list[int]] = [[] for _ in range(len(rem_a))]
    pool_b: list[list[int]] = [[] for _ in range(len(rem_b))]
    ka_l, kb_l = ka.tolist(), kb.tolist()
    for i in perm:
        pool_a[ka_l[i]].append(i)
        pool_b[kb_l[i]].append(i)
    blk_codes, blk = np.unique(index[slots], return_inverse=True)
    sizes = np.bincount(blk, minlength=len(blk_codes))
    members: list[list[int]] = [[] for _ in range(len(blk_codes))]
    blk_l = blk.tolist()
    for i in perm:
        members[blk_l[i]].append(i)
    src_a = list(range(len(slots)))
    src_b = list(range(len(slots)))
    act_a = [i for i, c in enumerate(rem_a) if c]
    act_b = [i for i, c in enumerate(rem_b) if c]
    tot_a = tot_b = len(slots)
    for bi in np.lexsort((rng.random(len(sizes)), -sizes)).tolist():
        pos = members[bi]
        s = len(pos)
        x = _proportional(rem_a, act_a, tot_a, s, pyr)
        y = _proportional(rem_b, act_b, tot_b, s, pyr)
        _fit_margins(x, y, rem_a, rem_b, act_a, act_b)
        for i, c in x.items():
            rem_a[i] -= c
        for i, c in y.items():
            rem_b[i] -= c
        tot_a -= s
        tot_b -= s
        if any(rem_a[i] == 0 for i in x):
            act_a = [i for i in act_a if rem_a[i]]
        if any(rem_b[i] == 0 for i in y):
            act_b = [i for i in act_b if rem_b[i]]
        pa, pb = _distinct_pairs(x, y, pyr)
        for p, ac, bc in zip(pos, pa, pb):
            src_a[p] = pool_a[ac].pop()
            src_b[p] = pool_b[bc].pop()
    for unit, src in ((ua, src_a), (ub, src_b)):
        idx = np.asarray(src, dtype=np.int64)
        for c in unit:
            vals = np.asarray(columns[c], dtype=object)[slots]
            columns[c][slots] = vals[idx]
    return True


def _swap_repair(tp: Any, columns: dict[int, np.ndarray], ck: dict[str, Any], moved: tuple[int, ...],
                 rng: np.random.Generator, checker: _RealRowChecker | None, tries: int = 400) -> None:
    """Remove key repeats by swaps that create none (R5-03): a repeated row swaps whole with a row of another ID
    block - one with a blank key too, so a block can give away a filled key - or, for a pair key, swaps one key
    column's value (with its swap group) with any filled row. Every swap keeps the blank/non-blank patterns
    (states of what stays, or of the swapped unit, equal) and each column's values; after privacy
    (``checker``) a swap that would make a row equal a real one is undone."""
    index = tp.private.entity_index
    setup = _key_setup(tp, ck, columns)
    if index is None or setup is None:
        return
    cols, _, slots, _, _ = setup
    n = len(index)
    k = len(cols)
    comp: list[np.ndarray] = []  # per key column, a code per row (-1: not a slot)
    for c in cols:
        arr = np.full(n, -1, dtype=np.int64)
        arr[slots] = _col_codes(columns[c][slots])
        comp.append(arr)
    cnt: Counter = Counter()

    def tup(r: int) -> tuple | None:
        if comp[0][r] < 0:
            return None
        return tuple(int(c[r]) for c in comp)

    for r in slots.tolist():
        cnt[(int(index[r]), tup(r))] += 1
    stay = _missing_states(tp, columns, _staying_columns(tp, columns, moved), np.arange(n))
    units: list[tuple[int, ...] | None] = [None] * k
    unit_states: list[np.ndarray | None] = [None] * k
    if k == 2:
        ua = tuple(dict.fromkeys(_group_of(tp, cols[0], "row")))
        ub = tuple(dict.fromkeys(_group_of(tp, cols[1], "row")))
        if not set(ua) & set(ub) and all(c in columns for c in ua + ub):
            units = [ua, ub]
            unit_states = [_missing_states(tp, columns, u, np.arange(n)) for u in units]
    slot_list = slots.tolist()

    def delta(changes: list[tuple[int, tuple | None, tuple | None]]) -> int:
        """Change of the repeat count when rows of blocks move from one key to another."""
        d: Counter = Counter()
        for b, old, new in changes:
            if old is not None:
                d[(b, old)] -= 1
            if new is not None:
                d[(b, new)] += 1
        before = after = 0
        for key, s in d.items():
            c = cnt[key]
            before += max(c - 1, 0)
            after += max(c + s - 1, 0)
        return after - before

    def commit(changes: list[tuple[int, tuple | None, tuple | None]]) -> None:
        for b, old, new in changes:
            if old is not None:
                cnt[(b, old)] -= 1
            if new is not None:
                cnt[(b, new)] += 1

    todo = [r for r in slot_list if cnt[(int(index[r]), tup(r))] > 1]
    budget = max(200_000, 4 * len(slot_list))  # attempts in all (a key no swap can fix costs no more)
    for r in rng.permutation(np.asarray(todo, dtype=np.int64)).tolist():
        br = int(index[r])
        if cnt[(br, tup(r))] <= 1:
            continue
        if budget <= 0:
            break
        for attempt in range(tries):
            budget -= 1
            mode = attempt % 3 if units[0] is not None else 0
            if mode == 0:  # whole rows, between ID blocks
                s = int(rng.integers(n))
                bs = int(index[s])
                if bs == br or (stay is not None and stay[r] != stay[s]):
                    continue
                tr, ts = tup(r), tup(s)
                changes = [(br, tr, ts), (bs, ts, tr)]
                if delta(changes) >= 0:
                    continue
                _swap(columns, moved, r, s)
                if checker is not None and checker.any_real([r, s]):
                    _swap(columns, moved, r, s)
                    continue
                for c in comp:
                    c[r], c[s] = c[s], c[r]
                commit(changes)
                break
            j = mode - 1  # one key column's value (with its swap group), between any two filled rows
            s = slot_list[int(rng.integers(len(slot_list)))]
            us = unit_states[j]
            if s == r or comp[j][s] == comp[j][r] or (us is not None and us[r] != us[s]):
                continue
            bs = int(index[s])
            tr, ts = tup(r), tup(s)
            nr = tuple(int(comp[i][s]) if i == j else int(comp[i][r]) for i in range(k))
            ns = tuple(int(comp[i][r]) if i == j else int(comp[i][s]) for i in range(k))
            changes = [(br, tr, nr), (bs, ts, ns)]
            if delta(changes) >= 0:
                continue
            unit = units[j]
            assert unit is not None
            _swap(columns, unit, r, s)
            if checker is not None and checker.any_real([r, s]):
                _swap(columns, unit, r, s)
                continue
            comp[j][r], comp[j][s] = comp[j][s], comp[j][r]
            commit(changes)
            break


def _redraw_mode(p: ColumnProfile) -> str | None:
    """How a repeated key value of a date-family column may move (None: not at all). Interpolated values are no
    counts to keep: 'any' free value of the column's kind. A datetime whose dates are drawn by exact quota over
    the observed dates (``calendar``) keeps its date: 'same_day' moves only its interpolated time of day.
    Dates drawn by that quota are like levels (their counts are kept): they never move."""
    if TYPE_INFO[p.type].family != "date" or not p.params.get("formats"):
        return None
    calendar = bool(p.params.get("calendar")) and not p.params.get("person_level")
    if not calendar or p.params.get("kind") in ("time", "duration"):
        return "any"
    return "same_day" if p.params.get("kind") == "datetime" else None


def _redraw_key_values(tp: Any, columns: dict[int, np.ndarray], ck: dict[str, Any],
                       checker: _RealRowChecker | None) -> None:
    """A key repeat whose key column is an interpolated date takes the nearest free value of that column's own
    kind (R5-03: 20 visits of one student drew 17 distinct days): a date in the synthetic range on an observed
    weekday and the column's observed date grid, a time or instant on its row's render step and the time grid,
    written in the row's own layout. Another row of the block keeps the repeated value, so exact extremes stay."""
    from ..detect.dates_formats import US_DAY
    from ..profile.structure import rule_view

    index = tp.private.entity_index
    setup = _key_setup(tp, ck, columns)
    if index is None or setup is None:
        return
    cols = setup[0]
    for j, c in enumerate(cols):
        p = tp.columns[c]
        mode = _redraw_mode(p)
        if mode is None:
            continue
        hv = rule_view(columns[c], p, layout=True)
        if hv is None or not hv.ok.any():
            continue
        setup = _key_setup(tp, ck, columns)
        assert setup is not None
        slots = setup[2]
        key_l = list(zip(*[_stripped(columns[c2][slots]).tolist() for c2 in cols]))
        lo, hi = int(hv.vals[hv.ok].min()), int(hv.vals[hv.ok].max())
        grid = _date_candidates(p, hv, lo, hi)
        seen: dict[int, set] = defaultdict(set)
        repeats: list[int] = []
        block_size = Counter(index[slots].tolist())
        for pos, (r, kk) in enumerate(zip(slots.tolist(), key_l)):
            b = int(index[r])
            if kk in seen[b]:
                repeats.append(pos)
            seen[b].add(kk)
        for pos in repeats:
            r = int(slots[pos])
            b = int(index[r])
            if not hv.ok[r]:
                continue
            v = int(hv.vals[r])
            step = 1
            lo_r, hi_r = lo, hi
            if grid is None:
                res = _row_steps(p, hv, np.array([r]))
                gstep = _time_grid_step(p, hv)
                step = max(int(res[0]) if res is not None else 1, gstep)
            if mode == "same_day":  # the date was drawn by quota over the observed dates: keep it
                day_t = int(US_DAY * hv.tpu)
                start = (v // day_t) * day_t
                lo_r, hi_r = max(lo, start), min(hi, start + day_t - 1)
            # within (block size + 1) free steps a value the block does not hold exists, if the range has one
            for cand in _nearest(v, lo_r, hi_r, grid, step, limit=2 * int(block_size[b]) + 64):
                text = _render_once(p, hv, np.array([r]), np.array([cand]))[0]
                back = rule_view(np.array([text], dtype=object), p)
                if back is None or not back.ok[0] or int(back.vals[0]) != cand:
                    continue
                new_key = list(key_l[pos])
                new_key[j] = str(text).strip()
                nk = tuple(new_key)
                if nk in seen[b]:
                    continue
                old = columns[c][r]
                columns[c][r] = text
                if checker is not None and checker.any_real([r]):
                    columns[c][r] = old
                    continue
                seen[b].add(nk)
                key_l[pos] = nk
                break


def _date_candidates(p: ColumnProfile, hv: Any, lo: int, hi: int) -> np.ndarray | None:
    """Sorted values a date (days) or year-month (months) may take; None for times (stepped instead)."""
    from ..detect.dates_formats import weekday
    from .dates import grid_days

    if hv.kind == "date":
        wd = p.params.get("weekdays_observed") or list(range(7))
        g = grid_days(p.params, lo, hi, wd)
        if g is not None:
            return np.asarray(g, dtype=np.int64)
        if hi - lo > 200_000:
            return None
        days = np.arange(lo, hi + 1, dtype=np.int64)
        if 0 < len(set(wd)) < 7:
            days = days[np.isin(weekday(days), list(wd))]
        return days
    if hv.kind == "year_month":
        mons = np.arange(lo, hi + 1, dtype=np.int64)
        mg = p.params.get("month_grid")
        if mg:
            mons = mons[(mons - int(mg[1])) % int(mg[0]) == 0]
        return mons
    return None


def _time_grid_step(p: ColumnProfile, hv: Any) -> int:
    if hv.kind not in ("datetime", "time", "duration"):
        return 1
    from .dates import _grid

    return max(int(_grid(p.params, hv.tpu)[0]), 1)


def _nearest(v: int, lo: int, hi: int, grid: np.ndarray | None, step: int, limit: int):
    """Candidate values nearest to ``v`` first (never ``v``), within [lo, hi]: grid points, or ``v`` +- k steps."""
    if grid is not None:
        j = int(np.searchsorted(grid, v))
        left, right = j - 1, j
        if right < len(grid) and grid[right] == v:
            right += 1
        out = 0
        while (left >= 0 or right < len(grid)) and out < limit:
            if right >= len(grid) or (left >= 0 and v - grid[left] <= grid[right] - v):
                yield int(grid[left])
                left -= 1
            else:
                yield int(grid[right])
                right += 1
            out += 1
        return
    for i in range(1, limit + 1):
        for cand in (v - i * step, v + i * step):
            if lo <= cand <= hi:
                yield cand


def _moved_columns(tp: Any, columns: dict[int, np.ndarray], ck: dict[str, Any]) -> tuple[int, ...]:
    """Row-level columns a composite key's rows carry between ID blocks: the key columns with the row-level
    columns of their swap groups, then every other row-level column except those tied to an ID or a within-ID
    constant (a swap group holding one: they stay with their block) and those of another composite key (so
    enforcing one key never breaks another)."""
    cols = list(ck.get("cols") or [ck["x"]])
    unit = tuple(dict.fromkeys(c2 for c in cols for c2 in _group_of(tp, c, "row")))
    stay: set[int] = set()
    for other in tp.structure.private.get("composite", []):
        if other is ck:
            continue
        for c in other.get("cols") or [other["x"]]:
            stay |= set(_group_of(tp, c, "row")) | {c}
    for g in tp.structure.swap_groups:
        if any(tp.columns[c].role != "row" for c in g):
            stay |= set(g)
    rest = [p.index for p in tp.columns if p.role == "row" and p.index in columns and p.index not in unit
            and p.index not in stay]
    return unit + tuple(rest)


def _staying_columns(tp: Any, columns: dict[int, np.ndarray], moved: tuple[int, ...]) -> list[int]:
    """Generated columns that do not move with a composite key's rows (within-ID constants included)."""
    ms = set(moved)
    return [p.index for p in tp.columns if p.role in ("row", "entity") and p.index in columns and p.index not in ms]


def _missing_states(tp: Any, columns: dict[int, np.ndarray], cols: Any, slots: np.ndarray) -> np.ndarray | None:
    """Each slot's blank/non-blank state over ``cols`` (a state code); None when every slot has the same."""
    cols = list(cols)
    if not cols or len(slots) == 0:
        return None
    m = np.column_stack([missing_mask(columns[c][slots], tp.columns[c].missing_policy, tp.columns[c].na_tokens)
                         for c in cols])
    _, codes = np.unique(m, axis=0, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    return codes if len(np.unique(codes)) > 1 else None


def _greedy_assign(slots: np.ndarray, tuples: list[tuple], keys: list[Any], blocks: np.ndarray,
                   columns: dict[int, np.ndarray], unit: tuple[int, ...], rng: np.random.Generator,
                   states: np.ndarray | None = None) -> int:
    """Assign tuples to slots so no block gets one key twice (written into ``unit``); returns the number of
    unavoidable repeats. ``states``: see :func:`_assign_slots`."""
    pairs, residual = _assign_slots(slots, keys, blocks, rng, states)
    for pos, i in pairs:
        for c, v in zip(unit, tuples[i]):
            columns[c][pos] = v
    return residual


def _assign_slots(slots: np.ndarray, keys: list[Any], blocks: np.ndarray, rng: np.random.Generator,
                  states: np.ndarray | None = None) -> tuple[list[tuple[int, int]], int]:
    """(slot position, index of the slot whose content it takes) pairs, and the unavoidable repeats.

    With ``states`` (one code per slot), content only moves between slots of the same state: the states are
    assigned in turn, and a block never gets a key it already got in an earlier state.
    """
    if states is None or len(np.unique(states)) < 2:
        return _assign_part(slots, keys, blocks, rng, None)
    used: dict[int, set] = defaultdict(set)
    pairs: list[tuple[int, int]] = []
    residual = 0
    for st in np.unique(states).tolist():
        sel = np.flatnonzero(states == st)
        part, res = _assign_part(slots[sel], [keys[i] for i in sel.tolist()], blocks[sel], rng, used)
        pairs += [(pos, int(sel[i])) for pos, i in part]
        residual += res
    return pairs, residual


def _assign_part(slots: np.ndarray, keys: list[Any], blocks: np.ndarray, rng: np.random.Generator,
                 used: dict[int, set] | None) -> tuple[list[tuple[int, int]], int]:
    by_key: dict[Any, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    block_slots: dict[int, list[int]] = defaultdict(list)
    for s, b in zip(slots.tolist(), blocks.tolist()):
        block_slots[b].append(s)
    cap = {b: len(v) for b, v in block_slots.items()}
    buckets: dict[int, list[int]] = defaultdict(list)
    for b in rng.permutation(np.asarray(sorted(cap), dtype=np.int64)).tolist():
        buckets[cap[b]].append(b)
    max_cap = max(cap.values())
    key_list = list(by_key)
    order = rng.permutation(len(key_list))
    key_list = [key_list[i] for i in order]
    key_list.sort(key=lambda k: -len(by_key[k]))
    assigned: dict[int, list[int]] = defaultdict(list)
    residual = 0
    for k in key_list:
        items = by_key[k]
        need = len(items)
        taken: list[int] = []
        c = max_cap
        while len(taken) < need and c > 0:
            bucket = buckets.get(c)
            skipped: list[int] = []
            while bucket and len(taken) < need:
                j = int(rng.integers(len(bucket)))
                bucket[j], bucket[-1] = bucket[-1], bucket[j]
                b = bucket.pop()
                if used is not None and k in used[b]:
                    skipped.append(b)  # got this key in an earlier state
                    continue
                taken.append(b)
            if skipped:
                buckets[c].extend(skipped)
            c -= 1
        for b in taken:
            cap[b] -= 1
            if cap[b] > 0:
                buckets[cap[b]].append(b)
            if used is not None:
                used[b].add(k)
        for item, b in zip(items, taken):
            assigned[b].append(item)
        leftover = items[len(taken):]
        if leftover:
            residual += len(leftover)
            for item in leftover:  # unavoidable: any block with room
                for c2 in range(max_cap, 0, -1):
                    if buckets.get(c2):
                        b = buckets[c2].pop()
                        cap[b] -= 1
                        if cap[b] > 0:
                            buckets[cap[b]].append(b)
                        assigned[b].append(item)
                        if used is not None:
                            used[b].add(k)
                        break
        while max_cap > 0 and not buckets.get(max_cap):
            max_cap -= 1
    pairs: list[tuple[int, int]] = []
    for b, items in assigned.items():
        rows = block_slots[b]
        perm = rng.permutation(len(items))
        pairs += [(pos, items[i]) for pos, i in zip(rows, perm.tolist())]
    return pairs, residual


# --------------------------------------------------------------------------- after privacy
def _pattern_counts(tp: Any, columns: dict[int, np.ndarray], cols: list[int]) -> Counter:
    """Whole-row blank/non-blank pattern counts over ``cols`` (bit-packed rows)."""
    m = np.column_stack([_miss(tp, columns, c) for c in cols])
    packed = np.ascontiguousarray(np.packbits(m, axis=1))
    u, cnt = np.unique(packed, axis=0, return_counts=True)
    return Counter({bytes(row.tobytes()): int(c) for row, c in zip(u, cnt.tolist())})


def _snapshot_missing(tp: Any, columns: dict[int, np.ndarray], templates_on: bool, composite_on: bool) -> None:
    """Whole-row missingness pattern counts after the FD/rule step, when a later structural step writes or moves
    values (templates write only filled rows; composite keys move whole rows between slots of one state;
    privacy and duplicate-row swaps keep every column's state by design) - checked by :func:`_check_missing`."""
    if not (templates_on or composite_on):
        return
    cols = [p.index for p in tp.columns if p.role in ("row", "entity") and p.index in columns]
    n = len(columns[cols[0]]) if cols else 0
    if cols and n:
        tp.private.extra[_MISSING_SNAPSHOT] = (cols, n, _pattern_counts(tp, columns, cols))


def _check_missing(tp: Any, columns: dict[int, np.ndarray]) -> list[Message]:
    """Warn when the structural steps after the snapshot changed the whole-row missingness patterns (D6)."""
    snap = tp.private.extra.pop(_MISSING_SNAPSHOT, None)
    if not snap:
        return []
    cols, n, before = snap
    if any(c not in columns or len(columns[c]) != n for c in cols):
        return []
    after = _pattern_counts(tp, columns, cols)
    moved = sum((after - before).values())
    if not moved:
        return []
    return [Message("warning", "missingness_pattern_changed",
                    f"{moved} row(s) have a whole-row blank/non-blank pattern that the sampled patterns (D6) do not "
                    "hold: a structural step (derived template or composite key) changed it")]


def finalize(tp: Any, synth: Any, rng_for: Callable[[str], np.random.Generator], settings: Any) -> list[Message]:
    priv = tp.structure.private
    msgs: list[Message] = []
    if not priv:
        tp.private.extra.pop(_MISSING_SNAPSHOT, None)
        return msgs
    columns = {c.index: c.values for c in synth.columns}
    checker = _RealRowChecker(tp, columns)
    if priv.get("composite") and not _disabled(settings, "composite_keys"):
        msgs += repair_composite_keys(tp, columns, rng_for("composite"), checker)
    pinned = pinned_templates(tp)
    if pinned and not _disabled(settings, "derived_templates"):
        # rebuilt from the moved inputs (target values are never compared with real rows); messages were given once
        apply_templates(tp, columns, rng_for("templates"), pinned)
    if tp.structure.duplicate_rows and not _disabled(settings, "duplicate_rows"):
        msgs += adjust_duplicate_rows(tp, columns, rng_for("duplicates"), checker)
    msgs += _check_missing(tp, columns)
    msgs += check_rules(tp, columns, settings)
    perm = None
    if priv.get("sort") and not _disabled(settings, "sort_order"):
        perm = row_order(tp, columns)
    for c in synth.columns:
        c.values = columns[c.index] if perm is None else columns[c.index][perm]
    if perm is not None and tp.private.entity_index is not None:
        tp.private.entity_index = tp.private.entity_index[perm]
    return msgs


class _RealRowChecker:
    """Row-mode privacy check for single rows (same hashes as ``privacy.matches``)."""

    def __init__(self, tp: Any, columns: dict[int, np.ndarray]) -> None:
        mode, cols = matches.plan(tp)
        self.active = mode == "row" and tp.private.real_row_keys is not None and bool(cols)
        self.cols = cols
        self.real = tp.private.real_row_keys
        self.columns = columns

    def any_real(self, rows: list[int]) -> bool:
        if not self.active or not rows:
            return False
        arrays = []
        for c in self.cols:
            a = np.empty(len(rows), dtype=object)
            a[:] = [self.columns[c][r] for r in rows]
            arrays.append(a)
        return bool(np.isin(matches.row_hashes(arrays), self.real).any())


def repair_composite_keys(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator,
                          checker: _RealRowChecker) -> list[Message]:
    """Repair composite keys the privacy swaps broke (:func:`_settle_key` with the real-row check: whole rows,
    rows with a blank key, single key values; a date by a free date) and set each key's final public status:
    'enforced' only when no synthetic row repeats it (R5-03)."""
    msgs: list[Message] = []
    index = tp.private.entity_index
    if index is None:
        return msgs
    for ck in tp.structure.private.get("composite", []):
        setup = _key_setup(tp, ck, columns)
        if setup is None:
            continue
        cols, _, slots, keys, label = setup
        dups = _dup_rows(index, slots, keys)
        if dups:
            dups = _settle_key(tp, columns, ck, _moved_columns(tp, columns, ck), rng, checker)
        _key_status(tp, ck, dups)
        if dups:
            msgs.append(_key_warning(tp, ck, label, dups, " after the privacy swaps"))
    return msgs


def _swap(columns: dict[int, np.ndarray], unit: tuple[int, ...], r: int, s: int) -> None:
    for c in unit:
        col = columns[c]
        col[r], col[s] = col[s], col[r]


def _row_units(tp: Any, columns: dict[int, np.ndarray]) -> list[tuple[int, ...]]:
    """Row-level column groups that move together (swap groups, else single columns); key columns excluded."""
    row_cols = [p.index for p in tp.columns if p.role == "row" and p.index in columns]
    units: list[tuple[int, ...]] = []
    done: set[int] = set()
    for g in tp.structure.swap_groups:
        u = tuple(c for c in g if c in row_cols)
        if u:
            units.append(u)
            done.update(u)
    units += [(c,) for c in row_cols if c not in done]
    keys = {c for ck in tp.structure.private.get("composite", []) for c in (ck.get("cols") or [ck["x"]])}
    for info in pinned_templates(tp):  # tied to an ID / within-ID constant that does not move
        keys.add(info["target"])
        keys.update(p[1] for p in info["parts"] if p[0] == "f")
    return [u for u in units if not (set(u) & keys)]


class _Bag:
    """Rows holding one unit value: O(1) add/remove and random pick."""

    def __init__(self) -> None:
        self.items: list[int] = []
        self.pos: dict[int, int] = {}

    def add(self, x: int) -> None:
        self.pos[x] = len(self.items)
        self.items.append(x)

    def remove(self, x: int) -> None:
        i = self.pos.pop(x)
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def pick(self, rng: np.random.Generator, exclude: set[int]) -> int | None:
        for _ in range(20):
            if not self.items:
                return None
            x = self.items[int(rng.integers(len(self.items)))]
            if x not in exclude:
                return x
        return next((x for x in self.items if x not in exclude), None)


def _dup_count(columns: dict[int, np.ndarray], order: list[int]) -> int:
    df = pd.DataFrame({str(i): pd.Series(columns[c], dtype=object) for i, c in enumerate(order)})
    return int(df.duplicated().sum())


def adjust_duplicate_rows(tp: Any, columns: dict[int, np.ndarray], rng: np.random.Generator,
                          checker: _RealRowChecker) -> list[Message]:
    """Move the synthetic duplicate-row count towards the real one by cell swaps (see module doc)."""
    target = int(tp.structure.duplicate_rows.get("rows", 0))
    order = sorted(c for c in columns if tp.columns[c].role != "null")
    n = len(columns[order[0]]) if order else 0
    if n < 2:
        return []
    current = _dup_count(columns, order)
    if current == target:
        return []
    units = _row_units(tp, columns)
    if not units:
        return [Message("info", "duplicate_rows", f"{current} duplicate row(s) in the synthetic data vs {target} "
                        "in the real data (no row-level column can move)")]
    st = _DupState(columns, order, units, tp, rng, checker)
    if current < target:
        st.create(target)
    else:
        st.remove(target)
    current = st.dups()
    if current != target:
        return [Message("info", "duplicate_rows", f"{current} duplicate row(s) in the synthetic data vs {target} in "
                        "the real data (closest reachable by cell swaps)")]
    return []


class _DupState:
    """Row tuples, their counts and per-unit value bags for duplicate-row adjustment."""

    def __init__(self, columns: dict[int, np.ndarray], order: list[int], units: list[tuple[int, ...]], tp: Any,
                 rng: np.random.Generator, checker: _RealRowChecker) -> None:
        self.columns, self.order, self.units, self.rng, self.checker = columns, order, units, rng, checker
        self.n = len(columns[order[0]])
        self.rows = [tuple(columns[c][r] for c in order) for r in range(self.n)]
        self.counter = Counter(self.rows)
        self.miss = {c: missing_mask(columns[c], tp.columns[c].missing_policy, tp.columns[c].na_tokens)
                     for u in units for c in u}
        self.index = tp.private.entity_index

    def _val(self, u: tuple[int, ...], r: int) -> tuple:
        return tuple(self.columns[c][r] for c in u)

    def _swap(self, u: tuple[int, ...], a: int, b: int) -> None:
        _swap(self.columns, u, a, b)
        for c in u:
            m = self.miss[c]
            m[a], m[b] = m[b], m[a]

    def _refresh(self, changed: set[int]) -> None:
        for x in changed:
            old = self.rows[x]
            self.counter[old] -= 1
            if self.counter[old] == 0:
                del self.counter[old]
            self.rows[x] = tuple(self.columns[c][x] for c in self.order)
            self.counter[self.rows[x]] += 1

    def dups(self) -> int:
        return self.n - len(self.counter)

    def create(self, target: int) -> None:
        pattern = [tuple(bool(self.miss[c][r]) for u in self.units for c in u) for r in range(self.n)]
        groups: dict[tuple, list[int]] = defaultdict(list)
        for r in range(self.n):
            ent = int(self.index[r]) if self.index is not None else 0
            groups[(ent, pattern[r])].append(r)
        pools = [g for g in groups.values() if len(g) >= 2]
        if not pools:
            return
        bags: dict[tuple[int, ...], dict[tuple, _Bag]] = {}
        for u in self.units:
            d: dict[tuple, _Bag] = defaultdict(_Bag)
            for r in range(self.n):
                d[self._val(u, r)].add(r)
            bags[u] = d
        locked: set[int] = set()
        need = target - self.dups()
        attempts = 0
        while self.dups() < target and attempts < 30 * need + 100:
            attempts += 1
            g = pools[int(self.rng.integers(len(pools)))]
            r, s = (int(x) for x in self.rng.choice(g, size=2, replace=False))
            if r in locked or s in locked or self.rows[r] == self.rows[s]:
                continue
            ops: list[tuple[tuple[int, ...], int, int, tuple, tuple]] = []
            ok = True
            for u in self.units:
                rv, sv = self._val(u, r), self._val(u, s)
                if rv == sv:
                    continue
                t = bags[u][rv].pick(self.rng, locked | {r, s})
                if t is None:
                    ok = False
                    break
                self._swap(u, s, t)
                bags[u][rv].remove(t)
                bags[u][rv].add(s)
                bags[u][sv].remove(s)
                bags[u][sv].add(t)
                ops.append((u, s, t, rv, sv))
            if ok and self.checker.any_real(sorted({op[2] for op in ops} | {s})):
                ok = False
            if not ok:
                for u, a, b, av, bv in reversed(ops):
                    self._swap(u, a, b)
                    bags[u][av].remove(a)
                    bags[u][av].add(b)
                    bags[u][bv].remove(b)
                    bags[u][bv].add(a)
                continue
            self._refresh({s} | {op[2] for op in ops})
            locked.update((r, s))

    def remove(self, target: int) -> None:
        seen: set[tuple] = set()
        extra: list[int] = []
        for r in range(self.n):
            if self.rows[r] in seen:
                extra.append(r)
            seen.add(self.rows[r])
        for s in self.rng.permutation(np.asarray(extra, dtype=np.int64)).tolist():
            if self.dups() <= target:
                break
            if self.counter[self.rows[s]] <= 1:
                continue
            for _ in range(40):
                u = self.units[int(self.rng.integers(len(self.units)))]
                t = int(self.rng.integers(self.n))
                if t == s or any(self.miss[c][t] != self.miss[c][s] for c in u):
                    continue
                if self._val(u, t) == self._val(u, s):
                    continue
                before = self.dups()
                self._swap(u, s, t)
                self._refresh({s, t})
                if self.dups() >= before or self.checker.any_real([s, t]):
                    self._swap(u, s, t)
                    self._refresh({s, t})
                    continue
                break


def row_order(tp: Any, columns: dict[int, np.ndarray]) -> np.ndarray | None:
    """Row permutation for the detected sort keys or leading missing blocks (None: keep order)."""
    from ..profile.structure import order_values

    spec = tp.structure.private.get("sort") or {}
    n = len(next(iter(columns.values()))) if columns else 0
    if n < 2:
        return None
    keys: list[np.ndarray] = []
    if spec.get("keys"):
        for col, direction, na, mode in spec["keys"]:
            if col not in columns:
                continue
            ov = order_values(columns[col], tp.columns[col], mode)
            if ov is None:
                ov = order_values(columns[col], tp.columns[col], "text")
            vals, miss = ov  # type: ignore[misc]
            rank = -vals if direction == "desc" else vals
            na_key = np.where(miss, 0, 1) if na == "first" else np.where(miss, 1, 0)
            keys.append(na_key)
            keys.append(np.where(miss, 0, rank))
    elif spec.get("blocks"):
        for col, _ in spec["blocks"]:
            if col not in columns:
                continue
            p = tp.columns[col]
            miss = missing_mask(columns[col], p.missing_policy, p.na_tokens)
            keys.append(np.where(miss, 0, 1))
    if not keys:
        return None
    return np.lexsort(list(reversed(keys)))


__all__ = ["apply", "finalize", "align_fd_missingness", "apply_rules", "apply_fds", "check_rules", "apply_templates",
           "pinned_templates", "enforce_composite_keys",
           "repair_composite_keys", "adjust_duplicate_rows", "row_order"]
