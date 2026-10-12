"""Per-column statistics, identical for real and synthetic values (FOUNDATION).

:func:`column_stats` returns **report-safe aggregates** (design 3 "Every column"
plus the per-type extras, design D21):

* every column: n, missing count/rate, each missing token's text and count
  (tokens are blanks or NA-vocabulary tokens, never data), distinct count and
  ratio, unique flag, whitespace-variant count, raw length min/median/max, the
  top character-class masks held by >= min_cell rows, frequency profile counts;
* label-safe types (FACTOR, ORDINAL, BOOLEAN, ORG_CODE, SCHOOL_YEAR, CONSTANT,
  FLAG): level counts/shares and the mode; a level the D35 allowlist did not keep
  (``detect.text.copy_gate_units``) and its synthetic pseudonym both show as the
  same placeholder (``[pseudonymized 1]``), so the shares line up and no real
  value is printed;
* numeric types: min, max, mean, sd, median, p1..p99, zero/negative shares, and
  the pmf for discrete integers (left out when every value is unique: such a
  pmf would list every real value, e.g. an undeclared ID);
* other types: hooks ``dates.date_stats`` (A), ``numeric.numeric_stats_extended``
  (B), ``text.text_stats`` (C), each given the stripped values as text. Never a
  real ID/name/free-text value.
* ID-like candidate columns (``params['id_candidate']``, D7): value-revealing
  keys (min/max/median/quantiles/pmf/levels/mode) are removed
  (:func:`redact_candidate`), because the tool itself suspects the values
  identify someone. So are those of a column of repeated wide keys a flag typed
  (D38, ``detect.ids.FLAGGED_KEYS_KEY``: ``--org-code`` keeps its values in the
  output, never in the report).

The same function computes the synthetic side for the fidelity comparison.
:func:`numeric_view` is the single numeric reading of a column used for
continuous generation models and the KS check: plain numbers for the
foundation's continuous types, the AREA A hook ``dates.date_numbers`` for date
types and the AREA B hook ``numeric.numbers_extended`` for formatted numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from ..detect import dates as dates_hooks
from ..detect import numeric as numeric_hooks
from ..detect import text as text_hooks
from ..detect.ids import FLAGGED_KEYS_KEY, char_mask
from ..model import LABEL_SAFE_TYPES, TYPE_INFO, ColumnProfile, ColumnType
from .codes import codes_of, counts_of

# Public list hooks remain replaceable. Their default implementations can use
# counts directly, while an override still receives the documented value list.
_DEFAULT_DATE_STATS = dates_hooks.date_stats
_DEFAULT_NUMERIC_STATS = numeric_hooks.numeric_stats_extended
_DEFAULT_TEXT_STATS = text_hooks.text_stats

#: D38: the flag behind a detection source, as the messages name it.
FLAG_NAMES = {"org-code": "--org-code", "force-type": "--force-type", "force-factor": "--force-factor"}

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
NUMERIC_TYPES = frozenset({ColumnType.INTEGER, ColumnType.INTEGER_YEAR, ColumnType.FLOAT, ColumnType.LAT,
                           ColumnType.LON})
_AREA_B = frozenset({ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL})
_CONTINUOUS = frozenset({ColumnType.FLOAT, ColumnType.LAT, ColumnType.LON})
#: Statistics that reveal real values; removed for ID-like candidate columns.
VALUE_REVEALING_KEYS = ("min", "max", "median", "quantiles", "pmf", "levels", "level_shares", "mode", "true_count",
                        "true_rate")


def has_numeric_view(profile: ColumnProfile) -> bool:
    """True for types whose distribution is compared by KS (continuous numbers, dates, AREA B)."""
    t = profile.type
    return (t in _CONTINUOUS or (t is ColumnType.INTEGER and profile.subtype == "continuous")
            or TYPE_INFO[t].family == "date" or t in _AREA_B)


def numeric_view(profile: ColumnProfile, values: list[str], counts: list[int] | None = None) -> np.ndarray | None:
    """Numbers of the stripped non-missing ``values`` (repeated by ``counts`` if given).

    Unparseable values (tokens) are dropped. None when the column type has no
    numeric view (levels, text, IDs) or the area hook returns None.
    """
    x = numeric_view_aligned(profile, values)
    if x is None:
        return None
    if counts is not None:
        x = np.repeat(x, np.asarray(counts, dtype=np.int64))
    return x[np.isfinite(x)]


def numeric_view_aligned(profile: ColumnProfile, values: list[str]) -> np.ndarray | None:
    """:func:`numeric_view` of each stripped value, aligned with ``values`` (NaN for a token); None when the
    column type has no numeric view."""
    t = profile.type
    if t in _CONTINUOUS or (t is ColumnType.INTEGER and profile.subtype == "continuous"):
        return np.fromiter((_to_float(v) for v in values), dtype=float, count=len(values))
    if TYPE_INFO[t].family == "date":
        r = dates_hooks.date_numbers(values, profile)
    elif t in _AREA_B:
        r = numeric_hooks.numbers_extended(values, profile)
    else:
        return None
    return None if r is None else np.asarray(r, dtype=float)


def _to_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def redact_candidate(st: dict[str, Any], reason: str) -> dict[str, Any]:
    """Remove value-revealing statistics of an ID-like candidate column (D7, D21)."""
    for k in VALUE_REVEALING_KEYS:
        st.pop(k, None)
    st["redacted"] = f"ID-like column ({reason}): values are not listed"
    return st


def token_display(raw: Any) -> str:
    return "<null>" if raw is None else raw


def is_missing(raw: Any, policy: str, na_tokens: frozenset[str]) -> bool:
    if raw is None:
        return True
    s = raw.strip()
    return s == "" or (policy == "typed" and s in na_tokens)


def missing_mask(values: np.ndarray, policy: str, na_tokens: frozenset[str]) -> np.ndarray:
    """Boolean missing flag per row (vectorised over distinct values)."""
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    codes, uniques = codes_of(np.asarray(values, dtype=object))
    # pd.factorize turns a native null (None, typed formats) into NaN: map it back to None
    flags = np.fromiter((is_missing(u if isinstance(u, str) else None, policy, na_tokens) for u in uniques),
                        dtype=bool, count=len(uniques))
    return flags[codes]


def split_values(values: np.ndarray, policy: str, na_tokens: frozenset[str],
                 counts: dict[Any, int] | None = None) -> tuple[Counter, Counter]:
    """(non-missing raw counts, missing raw counts), insertion-ordered.

    ``counts`` (raw value -> count) may be passed when already computed.
    """
    if counts is None:
        counts = counts_of(values)
    nonmissing: Counter = Counter()
    missing: Counter = Counter()
    for raw, c in counts.items():
        (missing if is_missing(raw, policy, na_tokens) else nonmissing)[raw] = c
    return nonmissing, missing


def _r(x: float) -> float | None:
    """Rounded to 6 decimals; None ('unavailable') for a non-finite value, so reports stay strict JSON."""
    f = float(x)
    return float(round(f, 6)) if np.isfinite(f) else None


def _moments(x: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(mean, sd, median, quantiles); finite values whose sum or squares overflow float64 (1e180 ...)
    are computed on ``x / max|x|`` and scaled back, so a finite column never gets an infinite/NaN statistic."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(x.mean())
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        median = float(np.median(x))
        q = np.quantile(x, QUANTILES)
        if np.all(np.isfinite([mean, sd, median, *q])) or not np.all(np.isfinite(x)):
            return mean, sd, median, q
        s = float(np.max(np.abs(x)))
        y = x / s
        return (float(y.mean()) * s, (float(y.std(ddof=1)) * s if x.size > 1 else 0.0), float(np.median(y)) * s,
                np.quantile(y, QUANTILES) * s)


def numeric_summary(x: np.ndarray) -> dict[str, Any]:
    if x.size == 0:
        return {}
    mean, sd, median, q = _moments(x)
    return {
        "min": _num(x.min()), "max": _num(x.max()), "mean": _r(mean),
        "sd": _r(sd), "median": _r(median),
        "quantiles": {f"p{int(round(p * 100))}": _r(v) for p, v in zip(QUANTILES, q)},
        "zero_share": _r(np.mean(x == 0)), "negative_share": _r(np.mean(x < 0)),
    }


def _num(v: Any) -> float | int | None:
    f = float(v)
    if not np.isfinite(f):
        return None  # 'unavailable': reports stay strict JSON
    return int(f) if f.is_integer() and abs(f) < 2**53 else f


def _wide_integers(t: ColumnType, keys: list[str], x: np.ndarray) -> dict[str, int] | None:
    """Exact integer of each integer text of an integer column whose values reach ``2**53`` (float64 rounds
    them: the min, max and pmf would merge neighbouring values, R2-12); None otherwise."""
    if t not in (ColumnType.INTEGER, ColumnType.INTEGER_YEAR) or not x.size or float(np.max(np.abs(x))) < 2**53:
        return None
    out: dict[str, int] = {}
    for k in keys:
        whole, _, frac = k.lstrip("+").partition(".")
        if frac.strip("0") or whole in ("", "-"):
            continue
        try:
            out[k] = int(whole)
        except ValueError:
            continue
    return out or None


def parse_numbers(stripped: list[str], counts: list[int]) -> np.ndarray:
    """Expand distinct numeric strings with their counts into a float array."""
    vals: list[float] = []
    cnts: list[int] = []
    for s, c in zip(stripped, counts):
        try:
            vals.append(float(s))
        except ValueError:
            continue
        cnts.append(c)
    return np.repeat(np.asarray(vals, dtype=float), np.asarray(cnts, dtype=np.int64))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of ``values`` repeated ``weights`` times (no expansion)."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    total = int(cum[-1])
    lo = v[np.searchsorted(cum, (total - 1) // 2 + 1)]
    hi = v[np.searchsorted(cum, total // 2 + 1)]
    return (float(lo) + float(hi)) / 2


def column_stats(values: np.ndarray | None, profile: ColumnProfile, min_cell: int,
                 counts: dict[Any, int] | None = None, n: int | None = None) -> dict[str, Any]:
    """Report-safe statistics of ``values`` interpreted with ``profile``'s type.

    ``counts`` (raw value -> count of ``values``) may be passed when already computed; with ``counts``, ``values``
    may be None when ``n`` (the number of values) is given.
    """
    t = profile.type
    n = int(len(values)) if n is None else int(n)
    nonmissing, missing = split_values(values, profile.missing_policy, profile.na_tokens, counts)
    n_missing = int(sum(missing.values()))
    n_nm = n - n_missing
    stripped: Counter = Counter()
    ws_variants = 0
    for raw, c in nonmissing.items():
        s = raw.strip()
        stripped[s] += c
        if s != raw:
            ws_variants += c
    distinct = len(stripped)
    st: dict[str, Any] = {
        "n": n,
        "missing": n_missing,
        "missing_rate": _r(n_missing / n) if n else 0.0,
        "missing_tokens": {token_display(k): int(v) for k, v in sorted(missing.items(), key=lambda kv: (-kv[1], str(kv[0])))},
        "distinct": distinct,
        "distinct_ratio": _r(distinct / n_nm) if n_nm else 0.0,
        "unique": bool(n_nm) and distinct == n_nm,
        "whitespace_variants": ws_variants,
    }
    if nonmissing:
        lens = np.fromiter((len(raw) for raw in nonmissing), dtype=np.int64, count=len(nonmissing))
        wts = np.fromiter(nonmissing.values(), dtype=np.int64, count=len(nonmissing))
        st["length"] = {"min": int(lens.min()), "median": _r(weighted_median(lens, wts)), "max": int(lens.max())}
    freq = np.asarray(sorted(stripped.values(), reverse=True)) if stripped else np.asarray([], dtype=int)
    if freq.size:
        st["frequency_profile"] = {"singletons": int((freq == 1).sum()), "max_count": int(freq[0]),
                                   "levels_below_min_cell": int((freq < min_cell).sum())}
    masks: Counter = Counter()
    for s, c in stripped.items():
        masks[char_mask(s)] += c
    st["top_masks"] = {m: int(c) for m, c in masks.most_common(5) if c >= min_cell}

    if t in LABEL_SAFE_TYPES:
        shown = text_hooks.display_map(profile)  # D35: a pseudonymised level shows its placeholder
        items: Any = nonmissing.items()
        if shown:
            merged: Counter = Counter()
            for k, v in nonmissing.items():
                merged[shown.get(k.strip(), k) if isinstance(k, str) else k] += v
            items = merged.items()
        levels = sorted(items, key=lambda kv: (-kv[1], str(kv[0])))
        st["levels"] = {token_display(k): int(v) for k, v in levels}
        st["level_shares"] = {token_display(k): _r(v / n_nm) for k, v in levels} if n_nm else {}
        if levels:
            st["mode"] = token_display(levels[0][0])
        if t is ColumnType.BOOLEAN and profile.params.get("pair"):
            true_tok = profile.params["pair"][0]
            tc = sum(c for s, c in stripped.items() if s.casefold() == true_tok)
            st["true_count"] = int(tc)
            st["true_rate"] = _r(tc / n_nm) if n_nm else None
    elif t in NUMERIC_TYPES:
        keys = list(stripped)
        x = parse_numbers(keys, [stripped[k] for k in keys])
        st.update(numeric_summary(x))
        discrete = t is ColumnType.INTEGER_YEAR or (t is ColumnType.INTEGER and profile.subtype == "discrete")
        wide = _wide_integers(t, keys, x)  # int64 from 2**53: float64 merges neighbours (R2-12)
        if wide is not None:
            st["min"], st["max"] = min(wide.values()), max(wide.values())
        if discrete and x.size:
            if wide is not None:
                exact: Counter = Counter()
                for k in keys:
                    if k in wide:
                        exact[wide[k]] += stripped[k]
                pmf_items = [(str(v), exact[v]) for v in sorted(exact)]
            else:
                vals, cnts = np.unique(x, return_counts=True)
                pmf_items = [(str(_num(v)), int(c)) for v, c in zip(vals, cnts)]
            if x.size > 1 and len(pmf_items) == x.size:
                st["pmf_omitted"] = "every value is unique"
            elif len(pmf_items) <= 200:
                st["pmf"] = {k: int(c) for k, c in pmf_items}
    elif TYPE_INFO[t].family == "date":
        st.update(_hook_stats(dates_hooks.date_stats, _DEFAULT_DATE_STATS,
                              dates_hooks.date_stats_counts, stripped, profile))
    elif t in _AREA_B:
        st.update(_hook_stats(numeric_hooks.numeric_stats_extended, _DEFAULT_NUMERIC_STATS,
                              numeric_hooks.numeric_stats_extended_counts, stripped, profile))
    elif TYPE_INFO[t].family in ("text", "pii", "id"):
        if stripped:
            total = sum(stripped.values())
            st["mean_words"] = _r(sum(len(s.split()) * c for s, c in stripped.items()) / total)
        st.update(_hook_stats(text_hooks.text_stats, _DEFAULT_TEXT_STATS,
                              text_hooks.text_stats_counts, stripped, profile))
    if profile.params.get("id_candidate"):
        redact_candidate(st, str(profile.params["id_candidate"]))
    elif profile.params.get(FLAGGED_KEYS_KEY):  # D38: wide repeated keys a flag typed: values kept, not listed
        redact_candidate(st, flagged_keys_reason(profile))
    return st


def flagged_keys_reason(profile: ColumnProfile) -> str:
    """D38: the report-safe reason a flagged column of repeated wide keys has its value statistics withheld."""
    return (f"repeated keys of up to {profile.params.get(FLAGGED_KEYS_KEY)} digits, typed by "
            f"{FLAG_NAMES.get(profile.source, '--' + str(profile.source))}")


def _hook_stats(legacy_hook: Callable, default_hook: Callable, counts_hook: Callable,
                counts: dict[str, int], profile: ColumnProfile) -> dict[str, Any]:
    """Use counts unless the documented list hook has been replaced."""
    if legacy_hook is default_hook:
        return counts_hook(counts, profile)
    return legacy_hook(_expand(counts), profile)


def _expand(counts: dict[str, int]) -> list[str]:
    """Legacy hook values, grouped in stripped first-appearance order."""
    return [value for value, count in counts.items() for _ in range(count)]
