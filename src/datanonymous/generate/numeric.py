"""Numeric generation and rendering (FOUNDATION core + AREA B).

FOUNDATION (design 3 #9-10, #15; 5.2.2; D14):

* discrete integers and years use exact quota over the observed raw values
  (``generate/table.py``), so only observed values appear;
* :func:`generate_continuous` - stratified interpolated inverse-CDF over the
  sorted real values (``u = (perm(n) + U(0,1)) / n``), rounded to the observed
  decimals, clipped to [min, max] (no extrapolation); the exact min and max are
  each emitted at least once (``extreme_value_adjusted`` when that changed a
  value; ``--no-exact-extremes COL`` turns it off) by replacing the generated
  value nearest to the missing extreme;
* :func:`render_numeric` - the source's text form: fixed or variable decimals
  (variable: each value keeps its own decimals, trailing zeros dropped down to
  the observed minimum), ``.0`` on integer-valued text, leading zero (``.5``),
  explicit ``+``, bare trailing point (``12.``, AREA B).
* :func:`round_decimals` (AREA B) - the generator's rounding: ``np.round`` that
  leaves a value alone where the requested decimals overflow it (no float64
  digit there) and handles more than 300 decimals (tiny scientific values).

AREA B (design 3 #15-#18; D14):

* :func:`render_numeric` also renders FLOAT columns of subtype ``scientific``
  in the source notation: mantissa decimals (fixed, or variable with trailing
  zeros dropped), exponent letter, explicit ``+`` and zero-padded exponent
  digits (``1.5e-05``, ``6.53396E+11``). In a column that mixes notations a
  value is written in scientific notation when its decimal exponent was only
  seen in scientific values, plainly when only seen in plain values, and in the
  majority notation otherwise. The continuous generator rounds such columns to
  absolute decimals fine enough for their smallest value; the renderer rounds
  to the mantissa digits.
* :func:`generate_numeric_extended`: NUMERIC_FORMATTED and DECIMAL use the
  FLOAT generator (:func:`generate_continuous` over the parsed numbers, rounded
  to the observed decimals, or to 0 decimals when every real value is
  integer-valued such as ``$25.00``; DECIMAL to its scale). NUMERIC_WITH_TOKENS
  is two-stage: an exact quota of each token (raw text, verbatim) against one
  numeric bucket, with a floor of one row per token present in the source
  (``rare_category_adjusted`` when the floor changed a count), then the numeric
  slots by the inverse-CDF clipped to the numeric part's own [min, max] with the
  exact extremes (D14). A token the D35 allowlist did not keep (rare, or holding
  personal data: ``absent, call 312-555-0199``) is written as its same-shape
  pseudonym (``generate.pseudonyms``, applied to the model before this generator
  runs), never verbatim.
* Within-ID-constant columns (D4) are generated from a per-entity model (each
  entity's most common value), where a token or an extreme held only as a
  minority value inside one entity is lost. For AREA B types the row-level
  model (``profile.model``) restores them: each lost token is floored in with
  weight 1 (``rare_category_adjusted``) and each lost exact min/max is added to
  the numbers (``extreme_value_adjusted``; not with ``--no-exact-extremes``).
* :func:`render_numeric_extended` re-applies the column's number format
  (:func:`format_number`: thousands separator, decimal mark, decimals, the
  positive/negative/zero skeletons such as ``'$#'`` / ``'($#)'``), writes tokens
  verbatim, and writes decimals with exactly ``scale`` decimals.
* Wide numbers (R1-14, R2-12): a float64 holds at most 15 significant digits
  exactly (integers up to ``2**53``), so a column whose values need more
  (``decimal128(38, 6)`` near its limit, ``decimal(38, 18)`` amounts, int64
  values from ``2**53``, decimal text with 16+ significant digits, formatted
  numbers that wide) is generated on **exact scaled integers** (``value *
  10**scale`` from the raw texts: :func:`exact_value`) by the same stratified
  interpolated inverse-CDF (same random draws), with the exact min and max (D14)
  and values that always fit the source type; they are returned as ``Decimal``
  and rendered exactly. This covers DECIMAL, INTEGER (continuous), FLOAT, LAT,
  LON (plain notation), NUMERIC_FORMATTED and the numbers of
  NUMERIC_WITH_TOKENS. Columns within float64's exact range keep the float
  generator, which is exact there; :func:`exact_extremes_missing` is the exact
  D14 self-check for wide columns.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation
from typing import Any

import numpy as np

from ..errors import UnsupportedColumnError
from ..model import ColumnModel, ColumnProfile, ColumnType
from . import pseudonyms
from .quota import allocate


def stratified_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.permutation(n) + rng.random(n)) / n


def interp_icdf(sorted_x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linear-interpolated inverse empirical CDF (numpy 'linear' / type 7)."""
    m = len(sorted_x)
    if m == 1:
        return np.full(len(u), float(sorted_x[0]))
    pos = u * (m - 1)
    lo = np.floor(pos).astype(np.int64)
    lo = np.clip(lo, 0, m - 1)
    hi = np.minimum(lo + 1, m - 1)
    frac = pos - lo
    return sorted_x[lo] + frac * (sorted_x[hi] - sorted_x[lo])


def round_decimals(y: np.ndarray, decimals: int) -> np.ndarray:
    """``np.round(y, decimals)`` that never overflows.

    ``np.round`` scales by ``10**decimals``: a value whose scaled form overflows
    has no float64 digit at that position and is left as it is (``np.round``
    would return inf), and more than 300 decimals (tiny scientific values such
    as ``1e-320``) are applied in two scaling steps.
    """
    d = int(decimals)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if d <= 300:
            r = np.round(y, d)
        else:
            k = 10.0 ** (d - 300)
            r = np.round(y * k, 300) / k
    bad = ~np.isfinite(r)
    if bad.any():
        r = np.where(bad, y, r)
    return r


def generate_continuous(model: ColumnModel, n: int, rng: np.random.Generator, decimals: int,
                        exact_extremes: bool = True) -> tuple[np.ndarray, bool]:
    """Continuous values for ``n`` slots; returns (float values, extreme_value_adjusted)."""
    xs = model.sorted_numeric
    if xs is None or len(xs) == 0:
        raise UnsupportedColumnError("no numeric values to model")
    if n == 0:
        return np.zeros(0), False
    lo, hi = float(xs[0]), float(xs[-1])
    y = interp_icdf(xs, stratified_uniform(n, rng))
    y = np.clip(round_decimals(y, decimals), lo, hi)
    adjusted = False
    if exact_extremes and lo != hi:
        if not np.any(y == lo):
            y[int(np.argmin(y))] = lo
            adjusted = True
        if not np.any(y == hi):
            cand = np.argsort(-y, kind="stable")
            for idx in cand:
                if y[idx] != lo or np.sum(y == lo) > 1:
                    y[idx] = hi
                    break
            adjusted = True
    return y, adjusted


def _fmt(v: float | Decimal, decimals: int) -> str:
    if isinstance(v, Decimal):  # exact wide values (R1-14, R2-12)
        return decimal_text(v, decimals)
    s = f"{v:.{decimals}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def _strip_decimals(s: str, min_d: int, mark: str = ".") -> str:
    """Drop trailing zeros of the decimals, keeping at least ``min_d`` of them."""
    if mark not in s:
        return s
    s = s.rstrip("0")
    have = len(s) - s.index(mark) - 1
    if have < min_d:
        s += "0" * (min_d - have)
    if s.endswith(mark):
        s = s[:-1]
    return s


def _plain_text(v: float, d: int, min_d: int, fixed: bool, leading_zero: bool, plus: bool,
                point: bool = False) -> str:
    s = _fmt(v, d)
    if not fixed:
        s = _strip_decimals(s, min_d)
    if not leading_zero:
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    if plus and not s.startswith("-") and v > 0:
        s = "+" + s
    if point and "." not in s:
        s += "."  # '12.' (SAS/Fortran style): readers infer a float, as for the source
    return s


def render_numeric(profile: ColumnProfile, values: np.ndarray) -> np.ndarray:
    """Render generated numbers in the column's observed text form (plain or scientific)."""
    p = profile.params
    if p.get("scientific"):
        return _render_scientific(p, values)
    d = int(p.get("decimals", 0))
    min_d = int(p.get("min_decimals", d))
    fixed = bool(p.get("fixed_decimals", True)) or p.get("float_text", False) and min_d == d
    leading_zero = p.get("leading_zero", True)
    plus = p.get("plus_sign", False)
    point = bool(p.get("trailing_point", False))
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values.tolist()):
        out[i] = _plain_text(v, d, min_d, fixed, leading_zero, plus, point)
    return out


# --------------------------------------------------------------------------- AREA B: scientific notation
def _exponent(v: float) -> int:
    return int(math.floor(math.log10(abs(v)))) if v != 0 and math.isfinite(v) else 0


def scientific_text(v: float, sci: dict[str, Any], plus: bool = False) -> str:
    """``v`` in the column's scientific notation (``params['scientific']``)."""
    d = int(sci["mantissa_decimals"])
    mant, _, exp = f"{v:.{d}e}".partition("e")
    if not sci.get("mantissa_fixed", True):
        mant = _strip_decimals(mant, int(sci.get("mantissa_min_decimals", 0)))
    if mant.startswith("-") and float(mant) == 0.0:
        mant = mant[1:]
    if plus and v > 0:
        mant = "+" + mant
    e = int(exp)
    sign = "-" if e < 0 else ("+" if sci.get("exp_plus", True) else "")
    return mant + sci.get("e_char", "e") + sign + str(abs(e)).zfill(int(sci.get("exp_digits", 1)))


def _render_scientific(p: dict[str, Any], values: np.ndarray) -> np.ndarray:
    sci = p["scientific"]
    plain = sci.get("plain")
    sci_exps, plain_exps = set(sci.get("sci_exponents") or ()), set(sci.get("plain_exponents") or ())
    majority_sci = float(sci.get("share", 1.0)) >= 0.5
    plus = bool(p.get("plus_sign", False))
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values.tolist()):
        use_sci = True
        if plain:
            e = _exponent(v)
            in_s, in_p = e in sci_exps, e in plain_exps
            use_sci = in_s if in_s != in_p else majority_sci
        if use_sci:
            out[i] = scientific_text(v, sci, plus)
        else:
            out[i] = _plain_in_scientific(v, plain, plus)
    return out


def _plain_in_scientific(v: float, plain: dict[str, Any], plus: bool) -> str:
    """A plain-notation value of a scientific column (significant digits when the source used them)."""
    max_d, min_d = int(plain["decimals"]), int(plain["min_decimals"])
    fixed = bool(plain["fixed_decimals"])
    d = max_d
    sig = plain.get("sig_digits")
    if sig and not fixed and v != 0:
        d = min(max(int(sig) - 1 - _exponent(v), 0), max_d)
    return _plain_text(v, d, min_d, fixed, bool(plain.get("leading_zero", True)), plus)


# --------------------------------------------------------------------------- AREA B: number formats
_SWAP_MARKS = str.maketrans({",": ".", ".": ","})


def format_number(v: float, fmt: dict[str, Any]) -> str:
    """One number in a learned number format (``params['format']`` of AREA B columns)."""
    d = int(fmt.get("decimals", 0))
    a = abs(v) if isinstance(v, Decimal) else abs(float(v))  # Decimal: exact wide values (R2-12)
    core = f"{a:,.{d}f}" if fmt.get("group_sep") else f"{a:.{d}f}"
    if not fmt.get("fixed_decimals", True):
        core = _strip_decimals(core, int(fmt.get("min_decimals", 0)))
    if not fmt.get("leading_zero", True) and core.startswith("0."):
        core = core[1:]
    if fmt.get("decimal_mark", ".") == ",":
        core = core.translate(_SWAP_MARKS)
    if not any(ch in "123456789" for ch in core):
        skeleton = fmt.get("zero") or fmt["positive"]
    elif v < 0:
        skeleton = fmt.get("negative") or ("-" + fmt["positive"].replace("+", ""))
    else:
        skeleton = fmt["positive"]
    return skeleton.replace("#", core)


def _rounding(p: dict[str, Any]) -> int:
    """Decimals the generator rounds to: 0 for integer-valued columns ($25.00), else the observed max."""
    return 0 if p.get("integer_valued") else int(p.get("decimals", 0))


def _is_token(raw: Any, tokens: set[str], fmt: dict[str, Any]) -> bool:
    from ..detect.numeric import parse_with_format

    s = str(raw).strip()
    return s in tokens or math.isnan(parse_with_format(s, fmt))


def _row_model(profile: ColumnProfile, model: ColumnModel) -> ColumnModel | None:
    """The row-level model when ``model`` is a per-entity one (D4 within-ID-constant column), else None."""
    row = profile.model
    return row if row is not None and row is not model else None


def _floor_row_extremes(profile: ColumnProfile, model: ColumnModel, exact_extremes: bool) -> tuple[ColumnModel, bool]:
    """D14 for a within-ID-constant column (D4): the row-level exact min/max stay reachable.

    The per-entity model keeps each entity's most common value, so an extreme
    held only as a minority value inside one entity would vanish. Each lost
    extreme is added to the per-entity sorted numbers (the generator then emits
    it at least once); returns (model to generate from, True when one was added).
    """
    row = _row_model(profile, model)
    if not exact_extremes or row is None or row.sorted_numeric is None or not len(row.sorted_numeric):
        return model, False
    xs = model.sorted_numeric if model.sorted_numeric is not None else np.zeros(0)
    lo, hi = float(row.sorted_numeric[0]), float(row.sorted_numeric[-1])
    add = [v for v in dict.fromkeys((lo, hi)) if not len(xs) or v < xs[0] or v > xs[-1]]
    if not add:
        return model, False
    floored = ColumnModel(nonmissing_counts=model.nonmissing_counts, missing_counts=model.missing_counts,
                          sorted_numeric=np.sort(np.concatenate([np.asarray(xs, dtype=float), add])),
                          extra=model.extra)
    return floored, True


def _generate_with_tokens(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
                          exact_extremes: bool) -> tuple[np.ndarray, dict[str, bool]]:
    p = profile.params
    fmt = p["format"]
    tokens = set(p.get("tokens") or p.get("_tokens") or ())
    levels: list[Any] = []
    counts: list[int] = []
    numeric_rows = 0
    for raw, c in model.nonmissing_counts.items():
        if _is_token(raw, tokens, fmt):
            levels.append(raw)
            counts.append(int(c))
        else:
            numeric_rows += int(c)
    floored = False
    row = _row_model(profile, model)
    if row is not None:
        # D14 in a within-ID-constant column (D4): a token (or the numbers) held only as a minority value
        # inside some entity is missing from the per-entity model; floor it in with weight 1
        have = set(levels)
        written = pseudonyms.level_map(profile)  # D35: a pseudonymised token is floored in as its pseudonym
        for raw in row.nonmissing_counts:  # every non-number of the column is one of the detected tokens
            out_raw = written.get(raw, raw)
            if out_raw not in have and str(raw).strip() in tokens:
                levels.append(out_raw)
                counts.append(1)
                have.add(out_raw)
                floored = True
        if not numeric_rows and row.sorted_numeric is not None and len(row.sorted_numeric):
            numeric_rows = 1
            floored = True
    if numeric_rows:
        counts.append(numeric_rows)
    if not counts:
        raise UnsupportedColumnError(f"column {profile.name!r}: no values to sample")
    alloc, adjusted = allocate(counts, n, floor=len(counts) <= n)
    out = np.empty(n, dtype=object)
    pos = 0
    for level, k in zip(levels, alloc.tolist()):
        out[pos:pos + k] = [level] * k
        pos += k
    flags = {"rare_category_adjusted": bool(adjusted or floored)}
    m = int(alloc[-1]) if numeric_rows else 0
    wide = _generate_exact(profile, model, m, rng, _rounding(p), exact_extremes,
                           skip=frozenset(str(t).strip() for t in tokens)) if m else None
    if wide is not None:  # numbers wider than float64: exact (R2-12)
        out[pos:pos + m] = list(wide[0])
        flags["extreme_value_adjusted"] = wide[1]
    elif m:
        num_model, extreme_floored = _floor_row_extremes(profile, model, exact_extremes)
        if num_model.sorted_numeric is None or not len(num_model.sorted_numeric):
            num_model = row if row is not None else num_model  # every entity value is a token
        y, adj = generate_continuous(num_model, m, rng, _rounding(p), exact_extremes)
        out[pos:pos + m] = [float(x) for x in y]
        flags["extreme_value_adjusted"] = adj or extreme_floored
    return rng.permutation(out), flags


# --------------------------------------------------------------------------- exact wide numbers (R1-14, R2-12)
#: Decimal arithmetic wide enough for any Arrow decimal (decimal256: 76 digits) and its scale.
DEC_CONTEXT = Context(prec=400, Emax=999999, Emin=-999999)
#: Integers below this (15 digits, DBL_DIG) round-trip through float64 exactly: the float generator is exact.
_FLOAT_EXACT_LIMIT = 10 ** 15
#: Integers below 2**53 are exact float64 values: integer columns keep the float generator below it.
_INT_EXACT_LIMIT = 2 ** 53
#: Column types with an exact numeric reading of their raw text (plain numbers, or AREA B number formats).
_EXACT_TYPES = frozenset({ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.LAT, ColumnType.LON, ColumnType.DECIMAL,
                          ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS})
_FORMAT_TYPES = (ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS)


def exact_value(raw: Any, profile: ColumnProfile | None = None) -> Decimal | None:
    """The exact value of one raw number text: plain (``9007199254740993``, ``-12.50``) or, for AREA B
    formatted columns, read with the column's number format (``$1,234.50``); None for anything else."""
    from ..detect.numeric import number_text

    s = str(raw).strip()
    fmt = profile.params.get("format") if profile is not None and profile.type in _FORMAT_TYPES else None
    if fmt:
        affix = fmt.get("affix") or (None, None)
        text = number_text(s, fmt.get("decimal_mark", "."), (affix[0], affix[1]))
        if text is None:
            return None
        s = text
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _scaled_counts(counts: dict[Any, int], scale: int, profile: ColumnProfile | None = None,
                   skip: frozenset[str] = frozenset()) -> dict[int, int]:
    """Exact ``value * 10**scale`` (integers) of raw number texts, with their counts (non-numbers and the
    ``skip`` tokens left out)."""
    out: dict[int, int] = {}
    for raw, c in counts.items():
        if raw is None or (skip and str(raw).strip() in skip):
            continue
        d = exact_value(raw, profile)
        if d is None:
            continue
        k = int(d.scaleb(scale, DEC_CONTEXT).to_integral_value(rounding=ROUND_HALF_EVEN, context=DEC_CONTEXT))
        out[k] = out.get(k, 0) + int(c)
    return out


def _generate_exact(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator, scale: int,
                    exact_extremes: bool, limit: int = _FLOAT_EXACT_LIMIT,
                    skip: frozenset[str] = frozenset()) -> tuple[np.ndarray, bool] | None:
    """Values on exact scaled integers when float64 cannot hold the column's values; None otherwise.

    Used when the largest ``|value| * 10**scale`` reaches ``limit`` (more than 15 significant digits for
    decimals, ``2**53`` for integers): float64 would round the values, and with them the exact min and max
    (D14), before generation. Mirrors :func:`generate_continuous` (same draws: :func:`stratified_uniform`;
    linear interpolation between neighbouring sorted values, rounded half up at the scale; exact min and
    max emitted at least once) and :func:`_floor_row_extremes` (D4: a row-level extreme lost by the
    per-entity model is added with weight 1). Returns (``Decimal`` values with exponent ``-scale``,
    extreme_value_adjusted).
    """
    counts = _scaled_counts(model.nonmissing_counts, scale, profile, skip)
    floored = False
    row = _row_model(profile, model)
    if row is not None:
        rc = _scaled_counts(row.nonmissing_counts, scale, profile, skip)
        if not counts:
            counts = dict(rc)  # every entity value is a token: the row-level numbers
        elif exact_extremes and rc:
            e_lo, e_hi = min(counts), max(counts)
            for v in dict.fromkeys((min(rc), max(rc))):
                if v < e_lo or v > e_hi:
                    counts[v] = counts.get(v, 0) + 1
                    floored = True
    if not counts:
        return None
    keys = sorted(counts)
    if max(abs(keys[0]), abs(keys[-1])) < limit:
        return None  # float64 is exact for these values: the float generator (unchanged behaviour)
    cum = np.cumsum(np.fromiter((counts[k] for k in keys), dtype=np.int64, count=len(keys)))
    m = int(cum[-1])
    u = stratified_uniform(n, rng)
    if m == 1:
        ys = [keys[0]] * n
    else:
        pos = u * (m - 1)
        lo_i = np.clip(np.floor(pos).astype(np.int64), 0, m - 1)
        hi_i = np.minimum(lo_i + 1, m - 1)
        frac = pos - lo_i
        j_lo = np.searchsorted(cum, lo_i, side="right")
        j_hi = np.searchsorted(cum, hi_i, side="right")
        ys = []
        for a_j, b_j, f in zip(j_lo.tolist(), j_hi.tolist(), frac.tolist()):
            a = keys[a_j]
            if a_j == b_j or f <= 0.0:
                ys.append(a)
                continue
            num, den = f.as_integer_ratio()  # exact: a + round_half_up(f * (b - a)), always within [a, b]
            ys.append(a + (2 * num * (keys[b_j] - a) + den) // (2 * den))
    adjusted = False
    lo, hi = keys[0], keys[-1]
    if exact_extremes and lo != hi and n:
        if lo not in ys:
            ys[ys.index(min(ys))] = lo
            adjusted = True
        if hi not in ys:
            i = ys.index(max(ys))
            if ys[i] != lo or ys.count(lo) > 1:
                ys[i] = hi
            adjusted = True
    out = np.empty(n, dtype=object)
    out[:] = [Decimal(k).scaleb(-scale, DEC_CONTEXT) for k in ys]
    return out, adjusted or floored


def _generate_decimal_exact(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
                            scale: int, exact_extremes: bool) -> tuple[np.ndarray, dict[str, bool]] | None:
    """DECIMAL values on exact scaled integers when float64 cannot hold them (R1-14); None otherwise."""
    wide = _generate_exact(profile, model, n, rng, scale, exact_extremes)
    return None if wide is None else (wide[0], {"extreme_value_adjusted": wide[1]})


def generate_continuous_wide(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
                             decimals: int, exact_extremes: bool = True) -> tuple[np.ndarray, bool] | None:
    """INTEGER (continuous) / FLOAT / LAT / LON values on exact scaled integers when float64 cannot hold the
    column's values (integers from ``2**53``, decimals with more than 15 significant digits: the exact min
    and max would be rounded away, R2-12); None otherwise (the float generator is exact there). Columns in
    scientific notation are floating point by nature and keep the float generator."""
    if n == 0 or profile.params.get("scientific"):
        return None
    limit = _INT_EXACT_LIMIT if profile.type is ColumnType.INTEGER else _FLOAT_EXACT_LIMIT
    return _generate_exact(profile, model, n, rng, max(int(decimals), 0), exact_extremes, limit)


def exact_extremes_missing(profile: ColumnProfile, values: Any, is_missing: Any) -> bool | None:
    """D14 self-check on exact values (R2-12): True when the column's exact real min or max is absent from the
    written ``values``; None when the float reading is exact (extremes of 15 significant digits or fewer) or
    the type has no exact reading (dates ...), so the caller's float check applies.

    ``is_missing(raw)`` tells missing cells; NUMERIC_WITH_TOKENS tokens are left out on both sides.
    """
    row = profile.model
    if (profile.type not in _EXACT_TYPES or row is None or not row.nonmissing_counts
            or profile.params.get("scientific") or _float_exact(profile)):
        return None
    skip = _token_set(profile)
    real = [exact_value(raw, profile) for raw in row.nonmissing_counts
            if raw is not None and not (skip and str(raw).strip() in skip)]
    real = [d for d in real if d is not None]
    if not real:
        return None
    lo, hi = min(real), max(real)
    if max(_sig_digits(lo), _sig_digits(hi)) <= 15:
        return None
    syn: set[Decimal] = set()
    for raw in {v for v in np.asarray(values, dtype=object).tolist() if isinstance(v, str)}:
        if is_missing(raw) or (skip and raw.strip() in skip):
            continue
        d = exact_value(raw, profile)
        if d is not None:
            syn.add(d)
    return not (lo in syn and hi in syn)


def exact_counts(profile: ColumnProfile, counts: dict[Any, int]) -> dict[Decimal, int] | None:
    """Exact value -> count of the numbers among raw ``counts`` (tokens left out) of a column wider than
    float64; None when the column type has no exact reading (dates, text ...) or float64 is exact for it (the
    caller's float reading applies). Equal values written differently (``1.50``, ``1.5``) are one."""
    if profile.type not in _EXACT_TYPES or profile.params.get("scientific") or _float_exact(profile):
        return None
    skip = _token_set(profile)
    out: dict[Decimal, int] = {}
    for raw, c in counts.items():
        if raw is None or (skip and str(raw).strip() in skip):
            continue
        d = exact_value(raw, profile)
        if d is not None:
            out[d] = out.get(d, 0) + int(c)
    return out


def _float_exact(profile: ColumnProfile) -> bool:
    """True when every value of the column has at most 15 significant digits, judged cheaply from the float
    extremes and the column's decimals (float64 holds such values exactly: no exact pass needed)."""
    xs = profile.model.sorted_numeric if profile.model is not None else None
    p = profile.params
    d = p.get("scale", p.get("decimals"))
    if d is None and isinstance(p.get("format"), dict):
        d = p["format"].get("decimals")
    if d is None:
        return False
    if xs is not None and len(xs):
        big = max(abs(float(xs[0])), abs(float(xs[-1])))
    else:  # discrete integers have no numeric view: the detected min/max
        lo, hi = p.get("min", p.get("_min")), p.get("max", p.get("_max"))
        if lo is None or hi is None:
            return False
        try:
            big = max(abs(float(lo)), abs(float(hi)))
        except (TypeError, ValueError, OverflowError):
            return False
    if not math.isfinite(big):
        return False
    return (len(str(int(big))) if big >= 1 else 0) + max(int(d), 0) <= 15


def _sig_digits(d: Decimal) -> int:
    """Significant digits of ``d`` (leading zeros of the digit tuple left out)."""
    digits = d.as_tuple().digits
    lead = next((i for i, x in enumerate(digits) if x), len(digits))
    return len(digits) - lead


def _token_set(profile: ColumnProfile) -> frozenset[str]:
    if profile.type is not ColumnType.NUMERIC_WITH_TOKENS:
        return frozenset()
    p = profile.params
    return frozenset(str(t).strip() for t in (p.get("tokens") or p.get("_tokens") or ()))


def decimal_text(v: Decimal, decimals: int) -> str:
    """Exact text of a ``Decimal`` with ``decimals`` decimals (``-0.00`` written ``0.00``)."""
    s = format(v.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN, context=DEC_CONTEXT), "f")
    if s.startswith("-") and not any(ch in "123456789" for ch in s):
        s = s[1:]
    return s


def generate_numeric_extended(profile: ColumnProfile, model: ColumnModel, n: int,
                              rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """AREA B hook: values for NUMERIC_FORMATTED / NUMERIC_WITH_TOKENS / DECIMAL columns.

    Called by ``generate/table.py`` for those types with ``n`` = the number of
    non-missing slots. Returns (generated values, flags such as
    ``{'extreme_value_adjusted': True, 'rare_category_adjusted': True}``): floats
    for NUMERIC_FORMATTED and DECIMAL (an object array of exact ``Decimal`` for a
    column wider than float64, R1-14/R2-12); for NUMERIC_WITH_TOKENS an object array
    of floats (or exact ``Decimal``) for the numeric slots and raw token strings, shuffled.
    :func:`render_numeric_extended` turns them into text. Rules: see the module
    docstring.
    """
    t = profile.type
    p = profile.params
    exact = not p.get("_no_exact_extremes", False)
    if t is ColumnType.NUMERIC_WITH_TOKENS:
        return _generate_with_tokens(profile, model, n, rng, exact)
    if t in (ColumnType.NUMERIC_FORMATTED, ColumnType.DECIMAL):
        if t is ColumnType.DECIMAL:  # more than 15 significant digits: exact scaled integers (R1-14)
            wide = _generate_decimal_exact(profile, model, n, rng, int(p.get("scale", p.get("decimals", 0))), exact)
            if wide is not None:
                return wide
        else:  # formatted numbers wider than float64 (R2-12)
            fwide = _generate_exact(profile, model, n, rng, _rounding(p), exact)
            if fwide is not None:
                return fwide[0], {"extreme_value_adjusted": fwide[1]}
        num_model, floored = _floor_row_extremes(profile, model, exact)
        decimals = _rounding(p) if t is ColumnType.NUMERIC_FORMATTED else int(p.get("scale", p.get("decimals", 0)))
        y, adj = generate_continuous(num_model, n, rng, decimals, exact)
        return y, {"extreme_value_adjusted": adj or floored}
    raise UnsupportedColumnError(
        f"column {profile.name!r}: {t.value} is not an area B type",
        fix=f"--force-text {profile.name} or --force-factor {profile.name}",
    )


#: Redraw rounds of :func:`avoid_real_ids` before a value is moved by steps of the column's resolution instead.
_REDRAW_ROUNDS = 20


def _nearest_free(d: Decimal, step: Decimal, forbidden: frozenset[Decimal] | set[Decimal],
                  bounds: tuple[Decimal, Decimal] | None) -> Decimal:
    """The value nearest ``d`` by steps of ``step`` that is not ``forbidden``: inside ``bounds`` (the column's real
    range) when one is left there, else outside it."""
    for inside_only in ((True, False) if bounds is not None else (False,)):
        k = 1
        while True:
            cands = [d + k * step, d - k * step]
            if inside_only:
                cands = [c for c in cands if bounds[0] <= c <= bounds[1]]  # type: ignore[index]
                if not cands:  # the whole range is taken
                    break
            for c in cands:
                if c not in forbidden:
                    return c
            k += 1
    raise AssertionError("unreachable: finitely many forbidden values")


def _resolution(profile: ColumnProfile) -> int:
    """Decimals the column's generator rounds to (0 for integers and integer-valued formatted numbers)."""
    p = profile.params
    if profile.type is ColumnType.INTEGER:
        return 0
    if profile.type in _FORMAT_TYPES:
        return _rounding(p)
    if profile.type is ColumnType.DECIMAL:
        return max(int(p.get("scale", p.get("decimals", 0))), 0)
    return max(int(p.get("decimals", 0)), 0)


def avoid_real_ids(profile: ColumnProfile, model: ColumnModel, values: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """R9-01: generated numbers of continuous column ``profile`` (floats or exact ``Decimal``; strings are tokens, left
    alone) with every one equal to a real number of the column that is a real person ID of the run
    (``privacy.realids.forbidden_numbers``: the exact min/max of D14, an interpolated value landing on one, a per-ID
    value) redrawn from the column's own distribution (``model``: the stratified inverse-CDF's interpolation, same
    rounding and range), then, if a few draws keep landing on one, moved by steps of the column's resolution. The
    shape of the distribution is kept and no real ID is written. Same array when nothing changes."""
    from ..privacy import realids

    forbidden = realids.forbidden_numbers(profile)
    if not forbidden or not len(values):
        return values
    fset = {float(d) for d in forbidden}
    ff = np.asarray(sorted(fset), dtype=float)
    vals = values.tolist()
    if values.dtype.kind == "f":  # plain floats: one vectorised look-up
        bad = np.flatnonzero(np.isin(values, ff)).tolist()
    else:
        bad = [i for i, v in enumerate(vals)
               if not isinstance(v, (str, bool)) and v is not None
               and ((v in forbidden) if isinstance(v, Decimal) else float(v) in fset)]
    if not bad:
        return values
    decimals = _resolution(profile)
    xs = model.sorted_numeric if model.sorted_numeric is not None and len(model.sorted_numeric) else \
        (profile.model.sorted_numeric if profile.model is not None else None)
    new = np.full(len(bad), np.nan)
    todo = np.arange(len(bad))
    bounds: tuple[Decimal, Decimal] | None = None
    if xs is not None and len(xs) and max(abs(float(xs[0])), abs(float(xs[-1]))) < 1e15:
        lo, hi = float(xs[0]), float(xs[-1])
        bounds = (Decimal(repr(lo)), Decimal(repr(hi)))
        for _ in range(_REDRAW_ROUNDS):
            if not len(todo):
                break
            y = np.clip(round_decimals(interp_icdf(np.asarray(xs, dtype=float), rng.random(len(todo))), decimals),
                        lo, hi)
            ok = ~np.isin(y, ff)
            new[todo[ok]] = y[ok]
            todo = todo[~ok]
    step = Decimal(1).scaleb(-decimals)
    out = np.array(values, copy=True)
    for j, i in enumerate(bad):
        v = vals[i]
        if np.isnan(new[j]):  # every draw landed on a real ID (or no numeric model): the nearest free value
            free = _nearest_free(v if isinstance(v, Decimal) else Decimal(repr(float(v))), step, forbidden, bounds)
            out[i] = free if isinstance(v, Decimal) else float(free)
        elif isinstance(v, Decimal):
            out[i] = Decimal(f"{new[j]:.{decimals}f}")
        else:
            out[i] = float(new[j])
    return out


def render_numeric_extended(profile: ColumnProfile, values: np.ndarray) -> np.ndarray:
    """AREA B hook: render values from :func:`generate_numeric_extended` as source text.

    NUMERIC_FORMATTED: :func:`format_number` with ``params['format']``;
    NUMERIC_WITH_TOKENS: tokens verbatim, numbers with the numeric part's format;
    DECIMAL: exactly ``scale`` decimals (``'-12.50'``), as Arrow casts it back.
    """
    t = profile.type
    p = profile.params
    out = np.empty(len(values), dtype=object)
    if t in (ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS):
        fmt = p["format"]
        for i, v in enumerate(values.tolist()):
            out[i] = v if isinstance(v, str) else format_number(v if isinstance(v, Decimal) else float(v), fmt)
        return out
    if t is ColumnType.DECIMAL:
        d = max(int(p.get("scale", p.get("decimals", 0))), 0)
        for i, v in enumerate(values.tolist()):
            out[i] = decimal_text(v, d) if isinstance(v, Decimal) else _fmt(float(v), d)
        return out
    raise UnsupportedColumnError(f"column {profile.name!r}: {t.value} rendering is not available (area B)")
