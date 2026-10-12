"""Real-vs-synthetic fidelity verdicts per column (AREA G).

Thresholds (design section 10 "Fidelity flagging", D14):

* **missing rate**: flagged when the synthetic share of missing cells deviates
  from the source share by more than ``max(1 pp, 2 binomial SE)``;
* **discrete columns** (levels, booleans, ordered levels, discrete integers,
  digit codes kept as levels): flagged when *any* level's share deviates by
  more than ``max(1 pp, 2 binomial SE)`` (SE of the source share over the
  source's non-missing count);
* **continuous columns** (continuous integers, floats, dates and formatted
  numbers through their numeric view): flagged when the two-sample KS
  statistic exceeds 0.05; numbers mixed with tokens (``*``, ``<10``) also
  when a token's share (or the numbers') deviates by more than
  ``max(1 pp, 2 binomial SE)``;
* never-copy text columns (invented labels, filler text, PII, derived-template
  targets): shares are not compared (the values are invented on purpose), but
  the label-free frequency profile is (R5-04): the distinct count, the largest
  count and the number of values held once must match the source's up to
  the difference in filled rows and the rows a derived template kept
  independent; IDs, row-masked digit codes and multi-select items only report
  their distinct counts;
* level-generated columns whose statistics do not list the values (discrete
  integers with every value unique or more than 200 values, digit codes kept
  as levels, ID-like candidates): the level-share check is reported as not
  compared (``compared: false``) with the reason, never as a passed comparison.

Adjusted columns never raise drift for the adjusted aspect (D14):
``rare_category_adjusted`` (every source level / NA token kept at least once)
excuses level-share and missing-rate deviations; ``extreme_value_adjusted``
(the exact min and max emitted) widens the limit by the share of forced
extreme values (``2 / n``), for level shares and for KS. A deviation that
only an adjustment explains gets the verdict ``adjusted``, never ``drift``.

Within-ID-constant columns (D4) are judged per entity (``basis='entities'``):
the pipeline passes the per-entity real and synthetic statistics. Their written
rows (each ID's value on all its rows) are judged too (:func:`with_row_check`,
R2-10, R3-07): missing rate, level shares and, for numbers and dates, the KS
statistic against ``max(0.05, 2 SE)`` over the effective number of IDs plus
the largest ID block's share of the rows; the verdict is the worse of the two.

Every verdict carries a plain ``reason`` and every check its ``limit``, so a
reader (or an AI agent) can see why a column passed or drifted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..generate.table import is_level_generated
from ..model import NEVER_COPY_TYPES, ColumnProfile, ColumnType

KS_LIMIT = 0.05
SHARE_FLOOR = 0.01  # 1 percentage point
#: Number of values the exact-extremes rule may force (the min and the max).
FORCED_EXTREMES = 2

VERDICTS = ("ok", "adjusted", "drift", "n/a")


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max distance between the ECDFs)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def share_limit(p: float, n: int) -> float:
    """``max(1 pp, 2 binomial SE)`` for a source share ``p`` over ``n`` cells."""
    se = math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0
    return max(SHARE_FLOOR, 2 * se)


# backwards-compatible private name
_share_limit = share_limit


def _pp(x: float) -> str:
    return f"{100 * x:.2f} pp"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def compare_column(p: ColumnProfile, real: dict[str, Any], syn: dict[str, Any],
                   real_numbers: np.ndarray | None = None, syn_numbers: np.ndarray | None = None,
                   basis: str = "rows", block_share: float = 0.0, n_eff: float | None = None) -> dict[str, Any]:
    """Verdict for one column.

    Returns ``{'verdict': 'ok'|'adjusted'|'drift'|'n/a', 'basis': ..., 'reason': str,
    'checks': [...]}``. ``basis`` is 'rows', or 'entities' for within-ID-constant
    columns (D4), whose generation target is the per-entity distribution. Each
    check has ``check``, ``ok``, ``limit`` (where a threshold applies),
    ``adjusted`` and a ``reason``. For the row-level check of a within-ID
    constant (:func:`with_row_check`) the KS limit is ``max(0.05, 2 SE)`` - the
    SE of a share at the median over ``n_eff`` effective IDs (Kish: rows come in
    ID blocks, as the level shares' ``max(1 pp, 2 SE)``) - plus ``block_share``,
    the largest ID block's share of the rows (one value covers a whole block,
    and an interpolated value never sits exactly on the real one).
    """
    if p.role == "excluded":
        return {"verdict": "n/a", "basis": basis, "reason": "column dropped (--exclude)", "checks": []}
    if p.role == "null":
        return {"verdict": "n/a", "basis": basis, "reason": "written as all-missing (--null)", "checks": []}
    rare_adj = bool(p.flags.get("rare_category_adjusted"))
    ext_adj = bool(p.flags.get("extreme_value_adjusted"))
    checks: list[dict[str, Any]] = []
    n = int(real.get("n", 0) or 0)
    if n:
        checks.append(_missing_check(real, syn, n, rare_adj))
    n_nonmissing = n - int(real.get("missing", 0) or 0)
    if "level_shares" in real or (is_level_generated(p) and "pmf" in real):
        checks.append(_level_check(p, real, syn, n, rare_adj, ext_adj))
    elif real_numbers is not None and syn_numbers is not None:
        checks.append(_ks_check(real_numbers, syn_numbers, ext_adj, block_share, n_eff))
        if "token_shares" in real and "token_shares" in syn and n_nonmissing > 0:
            checks.append(_token_check(real, syn, n_nonmissing, rare_adj))
    elif is_level_generated(p) and p.type is not ColumnType.EMPTY and n_nonmissing > 0:
        # exact quota over the observed values, but the statistics do not list them: say so, never claim a check
        if "redacted" in real:
            why = "the column looks like an ID, so its values are left out of the statistics"
        elif "pmf_omitted" in real:
            why = "every source value is unique, so the statistics do not list the values"
        elif p.type is ColumnType.DIGIT_CODE:
            why = "codes are kept as levels, but a code column's statistics do not list its values"
        else:
            why = "the statistics do not list the values (more than 200 distinct values)"
        rd, sd = real.get("distinct"), syn.get("distinct")
        checks.append({"check": "level_shares", "compared": False, "real": rd, "synthetic": sd, "ok": True,
                       "adjusted": False,
                       "reason": f"level shares not compared: {why}; generated by exact quota over the observed "
                                 f"values (distinct count {rd} source vs {sd} synthetic)"})
    elif p.type in NEVER_COPY_TYPES or p.type in (ColumnType.DIGIT_CODE, ColumnType.MULTI_SELECT):
        checks.append(_distinct_check(p, real, syn, n, basis, rare_adj))
    verdict = _verdict(checks)
    return {"verdict": verdict, "basis": basis, "reason": _reason(verdict, checks, basis), "checks": checks}


#: Invented values whose distinct count and frequency profile the generator keeps (one synthetic value per real
#: distinct value, rows by exact quota; derived templates regrouped to the source's groups, R5-04). IDs (a 1:1
#: map over normalised keys), digit codes masked row by row and item-wise multi-select are only reported.
_PROFILE_KEPT = NEVER_COPY_TYPES - {ColumnType.IDENTIFIER}


def _values_kind(p: ColumnProfile) -> str:
    """What the written values are (K-11: a multi-select writes real values the D35 allowlist kept, never only
    invented ones)."""
    if p.type is not ColumnType.MULTI_SELECT:
        return "values are invented (never real)"
    from ..detect.text import multi_select_mode

    if multi_select_mode(p) == "combinations":
        return ("whole combinations are written by exact quota: real ones the D35 allowlist kept, the others as "
                "same-shape pseudonyms")
    return ("items are drawn by their real prevalence into new combinations: real items the D35 allowlist kept, the "
            "others as same-shape pseudonyms")


def _distinct_check(p: ColumnProfile, real: dict[str, Any], syn: dict[str, Any], n: int, basis: str,
                    rare_adj: bool) -> dict[str, Any]:
    """Invented values (R5-04): the distinct count and the label-free frequency profile (largest count, values
    held once) must match the source's, up to the rows that legitimately differ - the difference in
    filled rows (sampled missingness) and the rows a derived template could not rebuild (they keep an
    independently generated value). The written rows of a within-ID constant are judged per ID instead."""
    rd, sd = real.get("distinct"), syn.get("distinct")
    info = {"check": "distinct", "real": rd, "synthetic": sd, "ok": True, "adjusted": False,
            "reason": f"{_values_kind(p)}, so shares are not compared; distinct count {rd} source vs {sd} synthetic"}
    if (p.type not in _PROFILE_KEPT or rd is None or sd is None or (p.role == "entity" and basis == "rows")):
        return info
    why_not = p.params.get("_template_profile")
    if why_not:  # a derived template whose rebuilt values cannot keep the source's profile: reported only
        info["reason"] = (f"derived template, rebuilt from its synthetic inputs ({why_not}): distinct count {rd} "
                          f"source vs {sd} synthetic, not compared")
        return info
    n_real = n - int(real.get("missing", 0) or 0)
    n_syn = int(syn.get("n", n) or 0) - int(syn.get("missing", 0) or 0)
    slack = abs(n_syn - n_real) + int(p.params.get("_template_rows_kept", 0) or 0)
    rf, sf = real.get("frequency_profile") or {}, syn.get("frequency_profile") or {}
    devs = {"distinct": abs(int(sd) - int(rd)),
            "max_count": abs(int(sf.get("max_count", 0)) - int(rf.get("max_count", 0))),
            "singletons": abs(int(sf.get("singletons", 0)) - int(rf.get("singletons", 0)))}
    limits = {"distinct": slack, "max_count": slack, "singletons": 2 * slack}
    bad = [k for k in devs if devs[k] > limits[k]]
    profile = (f"distinct count {sd} vs {rd} in the source, largest count {sf.get('max_count', 0)} vs "
               f"{rf.get('max_count', 0)}, values held once {sf.get('singletons', 0)} vs "
               f"{rf.get('singletons', 0)}")
    out = {"check": "distinct", "real": rd, "synthetic": sd, "deviation": devs["distinct"], "limit": slack,
           "max_count": {"real": rf.get("max_count"), "synthetic": sf.get("max_count")},
           "singletons": {"real": rf.get("singletons"), "synthetic": sf.get("singletons")}}
    allowance = (f"(allowed: {slack}, the difference in filled rows plus the rows a derived template kept "
                 "independent)")
    if not bad:
        out.update(ok=True, adjusted=False, reason=f"invented values keep the source's profile: {profile} {allowance}")
    elif rare_adj:
        out.update(ok=True, adjusted=True,
                   reason=f"{profile} {allowance}, explained by rare_category_adjusted (every source value kept at "
                          "least once)")
    else:
        out.update(ok=False, adjusted=False,
                   reason=f"invented values drift from the source's frequency profile ({', '.join(bad)}): {profile} "
                          f"{allowance}")
    return out


def _missing_check(real: dict[str, Any], syn: dict[str, Any], n: int, rare_adj: bool) -> dict[str, Any]:
    rr = float(real.get("missing_rate", 0.0) or 0.0)
    sr = float(syn.get("missing_rate", 0.0) or 0.0)
    dev = abs(rr - sr)
    limit = share_limit(rr, n)
    within = dev <= limit
    adjusted = (not within) and rare_adj
    if within:
        why = f"missing {_pct(sr)} vs {_pct(rr)} in the source (deviation {_pp(dev)} <= limit {_pp(limit)})"
    elif adjusted:
        why = (f"missing {_pct(sr)} vs {_pct(rr)} in the source deviates {_pp(dev)} > limit {_pp(limit)}, "
               "explained by rare_category_adjusted (every source NA token kept at least once)")
    else:
        why = (f"missing {_pct(sr)} vs {_pct(rr)} in the source deviates {_pp(dev)} > limit {_pp(limit)} "
               "(max(1 pp, 2 SE))")
    return {"check": "missing_rate", "real": rr, "synthetic": sr, "deviation": round(dev, 6),
            "limit": round(limit, 6), "ok": within or adjusted, "adjusted": adjusted, "reason": why}


def _level_check(p: ColumnProfile, real: dict[str, Any], syn: dict[str, Any], n: int, rare_adj: bool,
                 ext_adj: bool) -> dict[str, Any]:
    r_sh, s_sh = shares(real), shares(syn)
    n_nm = max(n - int(real.get("missing", 0) or 0), 1)
    extra = FORCED_EXTREMES / n_nm if ext_adj else 0.0
    worst, worst_level, worst_limit = 0.0, None, share_limit(0.0, n_nm)
    plain_bad = excused_bad = False
    for level in sorted(set(r_sh) | set(s_sh), key=str):
        d = abs(r_sh.get(level, 0.0) - s_sh.get(level, 0.0))
        limit = share_limit(r_sh.get(level, 0.0), n_nm)
        if d > worst:
            worst, worst_level, worst_limit = d, level, limit
        if d > limit:
            if rare_adj or d <= limit + extra:
                excused_bad = True
            else:
                plain_bad = True
    ok = not plain_bad
    adjusted = excused_bad and not plain_bad
    if not excused_bad and not plain_bad:
        why = f"every level share within max(1 pp, 2 SE) of the source (worst deviation {_pp(worst)})"
    elif ok:
        cause = ("rare_category_adjusted (every source level kept at least once)" if rare_adj
                 else "extreme_value_adjusted (exact min and max emitted)")
        why = f"worst level share deviates {_pp(worst)} > limit {_pp(worst_limit)}, explained by {cause}"
    else:
        why = f"a level share deviates {_pp(worst)} > limit {_pp(worst_limit)} (max(1 pp, 2 SE))"
    out: dict[str, Any] = {"check": "level_shares", "max_abs_diff": round(worst, 6), "limit": round(worst_limit, 6),
                           "ok": ok, "adjusted": adjusted, "reason": why}
    if "level_shares" in real and worst_level is not None:  # label-safe: the label is reproduced anyway (D21)
        out["worst_level"] = worst_level
    return out


def _token_check(real: dict[str, Any], syn: dict[str, Any], n_nm: int, rare_adj: bool) -> dict[str, Any]:
    """Numbers mixed with tokens (``*``, ``<10``): the share of each token and of the numbers among the filled
    cells, within ``max(1 pp, 2 SE)`` of the source (R3-07: a within-ID constant's rows must keep them too).
    ``rare_category_adjusted`` (every source token kept at least once) excuses a deviation."""
    r = dict(real.get("token_shares") or {})
    s = dict(syn.get("token_shares") or {})
    r["(numbers)"] = float(real.get("numeric_share", 0.0) or 0.0)
    s["(numbers)"] = float(syn.get("numeric_share", 0.0) or 0.0)
    worst, worst_limit = 0.0, share_limit(0.0, n_nm)
    bad = False
    for key in sorted(set(r) | set(s), key=str):
        d = abs(r.get(key, 0.0) - s.get(key, 0.0))
        limit = share_limit(r.get(key, 0.0), n_nm)
        if d > worst:
            worst, worst_limit = d, limit
        bad |= d > limit
    adjusted = bad and rare_adj
    if not bad:
        why = f"every token share (and the numbers' share) within max(1 pp, 2 SE) of the source (worst {_pp(worst)})"
    elif adjusted:
        why = (f"a token share deviates {_pp(worst)} > limit {_pp(worst_limit)}, explained by rare_category_adjusted "
               "(every source token kept at least once)")
    else:
        why = f"a token share deviates {_pp(worst)} > limit {_pp(worst_limit)} (max(1 pp, 2 SE))"
    return {"check": "token_shares", "max_abs_diff": round(worst, 6), "limit": round(worst_limit, 6),
            "ok": (not bad) or adjusted, "adjusted": adjusted, "reason": why}


def _ks_check(real_numbers: np.ndarray, syn_numbers: np.ndarray, ext_adj: bool,
              block_share: float = 0.0, n_eff: float | None = None) -> dict[str, Any]:
    ks = ks_statistic(real_numbers, syn_numbers)
    m = max(int(np.asarray(syn_numbers).size), 1)
    extra = FORCED_EXTREMES / m if ext_adj else 0.0
    base = KS_LIMIT
    if n_eff:  # 2 SE of a share at the median over the effective number of IDs (rows come in ID blocks)
        base = max(KS_LIMIT, 1.0 / math.sqrt(n_eff))
    limit = base + max(float(block_share), 0.0)
    if block_share or n_eff:
        shown = (f"{limit:.4f} (max(0.05, 2 SE = {base:.4f} over {n_eff or 0:.0f} effective IDs) + "
                 f"{block_share:.4f}, the largest ID block's share of rows)")
    else:
        shown = f"{KS_LIMIT}"
    within = ks <= limit
    adjusted = (not within) and ks <= limit + extra
    if within:
        why = f"KS {ks:.4f} <= {shown}"
    elif adjusted:
        why = (f"KS {ks:.4f} > {shown} but <= {limit + extra:.4f}, explained by extreme_value_adjusted "
               "(exact min and max emitted)")
    else:
        why = f"KS {ks:.4f} > {shown}"
    out = {"check": "ks", "value": round(ks, 6), "limit": round(limit, 6) if (block_share or n_eff) else KS_LIMIT,
           "limit_with_adjustment": round(limit + extra, 6), "ok": within or adjusted, "adjusted": adjusted,
           "reason": why}
    if block_share or n_eff:
        out["block_share"] = round(float(block_share), 6)
        out["effective_ids"] = round(float(n_eff or 0.0), 3)
    return out


def with_row_check(entity_result: dict[str, Any], rows_result: dict[str, Any]) -> dict[str, Any]:
    """A within-ID constant's verdict (D4): the per-entity comparison plus the written rows' (R2-10, R3-07).

    The column is generated per entity, but every row of the file holds its ID's value, so the written
    per-row distribution (each ID's value counted once per row) is judged too: the verdict is the worse of
    the two, the reason names both, and ``rows`` holds the row-level checks.
    """
    out = dict(entity_result)
    order = {"n/a": 0, "ok": 1, "adjusted": 2, "drift": 3}
    rv = rows_result.get("verdict", "n/a")
    out["rows"] = {"verdict": rv, "checks": rows_result.get("checks", [])}
    if rows_result.get("checks"):
        row_reason = _reason(rv, rows_result["checks"], "rows")
        out["reason"] = f"{entity_result.get('reason', '')}; per row (each ID's value on all its rows): {row_reason}"
    if order.get(rv, 0) > order.get(out.get("verdict", "n/a"), 0):
        out["verdict"] = rv
    return out


def _verdict(checks: list[dict[str, Any]]) -> str:
    if any(not c["ok"] for c in checks):
        return "drift"
    if any(c.get("adjusted") for c in checks):
        return "adjusted"
    return "ok"


def _reason(verdict: str, checks: list[dict[str, Any]], basis: str) -> str:
    prefix = "per entity (one value per ID, D4): " if basis == "entities" else ""
    if not checks:
        return prefix + "nothing to compare"
    if verdict == "drift":
        parts = [c["reason"] for c in checks if not c["ok"]]
    elif verdict == "adjusted":
        parts = [c["reason"] for c in checks if c.get("adjusted")]
    else:
        parts = [c["reason"] for c in checks]
    return prefix + "; ".join(parts)


def shares(st: dict[str, Any]) -> dict[str, float]:
    """Level shares of a statistics dict (label shares, or a discrete integer pmf)."""
    if "level_shares" in st:
        return dict(st["level_shares"])
    pmf = st.get("pmf") or {}
    total = sum(pmf.values())
    return {k: v / total for k, v in pmf.items()} if total else {}


_shares = shares
