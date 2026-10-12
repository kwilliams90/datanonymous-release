"""Numeric detection: integers, years, floats, digit codes, formatted numbers (FOUNDATION core + AREA B).

FOUNDATION (design 3 #8-11, #15):

* :func:`is_digit_code` - all non-missing values are digit strings and at least
  one has a leading zero (``0\\d+``): never numeric (``000123``, ``02134``, ``01``).
* :func:`detect_integer_year` - every value is 4 digits within 1900-2100.
* :func:`detect_numeric` - all ``[+-]?\\d+`` -> INTEGER, *discrete* when
  distinct <= max(factor_max, 50), or density distinct/(max-min+1) >= 0.5, or a
  ``code`` header hint; *continuous* otherwise (flagged for review as a possible
  sparse code). Plain decimals (no exponent, no thousands separators) -> FLOAT,
  except integer-valued text such as ``72.0`` -> INTEGER keeping the ``.0`` form.
  AREA B refinements: the leading-zero style (``.5`` vs ``0.5``) and the explicit
  ``+`` are the row-weighted majority form (ties keep ``0.5`` / no ``+``); a bare
  trailing point (``12.``, the majority form of values without decimal digits)
  is kept (``params['trailing_point']``: pandas reads ``12.`` as float64 but ``12``
  as int64); only ASCII digits are numbers (``float()`` also reads fullwidth or
  Arabic-Indic digits, which would be rendered back as ASCII); with
  ``--decimal ,`` on a text file a ``.`` is never a decimal mark, so such values
  go to the formatted-number rule; a native Arrow decimal column is left to
  :func:`detect_numeric_extended` (DECIMAL).

AREA B (design 3 #15-#18, section 4, section 10) - :func:`detect_numeric_extended`
runs after :func:`detect_numeric` returned None (and before the FACTOR rule), in
this order:

1. **DECIMAL** (#16): a native Arrow decimal column. The *native contract*:
   ``RawColumn.native['arrow_type']`` (also accepted under ``'type'`` or
   ``'dtype'``) holds the pyarrow ``DataType`` or its text (``'decimal128(9, 2)'``)
   and the values are canonical decimal strings (``'123.45'``). A format area may
   also return :func:`decimal_detection` from its ``detect_native``. Params:
   ``precision``, ``scale``, ``arrow_type`` (``decimal128(p, s)``, or
   ``decimal256`` for a 256-bit source or a precision above 38), min/max;
   generated as FLOAT (inverse-CDF rounded to the scale) and rendered with
   exactly ``scale`` decimals, so the writer's cast back to the source type is
   exact. ``--force-type COL=DECIMAL`` on plain numbers derives precision/scale
   from the values.
2. **FLOAT, subtype scientific** (#15, section 4): every value is a plain or
   scientific-notation number and at least one uses an exponent. Params
   ``scientific``: exponent letter (``e``/``E``), explicit ``+`` on non-negative
   exponents, minimum exponent digits (``e-05``), mantissa decimals (fixed or
   variable), the share of values written in scientific notation and the
   exponents seen in each notation (a mixed column renders a value in the
   notation whose observed exponents contain it, else the majority notation).
   **Excel-mangled IDs** (``d.dddddE+XX``: upper-case ``E``, explicit ``+``, two
   exponent digits, at most 5 mantissa decimals = Excel's 6 significant digits,
   integer-valued, every exponent >= 11, where Excel's General format switches to
   scientific notation; plain values, if any, are integers of at most 11 digits,
   which Excel's General format writes in full) get a warning that precision was
   lost upstream; they are rendered in the same notation. As suspected IDs (D21)
   they are marked ``params['id_candidate']`` (the report's value-revealing
   statistics are redacted) and their min/max params are private. The generator's
   absolute rounding (``decimals``) is fine enough for the smallest value; it is
   not capped by the largest one (see ``generate.numeric.round_decimals``).
3. **NUMERIC_FORMATTED** (#17): every value parses with :func:`parse_number`
   and at least one shows a format feature (currency ``$``/``EUR``/euro/pound
   prefix or suffix, ``%`` suffix, thousands separators, parenthesised
   negatives, inner spaces, a decimal comma). Every value must carry the same
   currency/percent affix. The learned *number format* (``params['format']``)
   is the majority skeleton for positive, negative and zero values (the value
   with its number replaced by ``#``: ``'$#'``, ``'($#)'``, ``'-$#'``,
   ``'$-#'``, ``'-#%'``, ``'+#'``, ``'$ #'`` ...), the thousands separator
   (used when most values with >= 4 integer digits use it), max/min decimals and
   fixed vs variable decimals, and the leading-zero style. A decimal comma
   (``1.234,50``, ``12,5%``) is parsed **only** with ``--decimal ,`` and only in
   text formats (CSV/TSV, HTML tables): typed formats store canonical numbers.
   Without it such values do not parse and the column stays text/factor.
   Statistics are FLOAT statistics of the parsed numbers (``45%`` -> 45). A
   column whose only grouped values are ``1,234``-like (one separator, no
   decimals anywhere, no dollar/pound affix) is flagged *ambiguous* (thousands
   vs a decimal mark); detection itself is unchanged.
4. **NUMERIC_WITH_TOKENS** (#18): >= 50% of the non-missing rows parse as
   numbers (plain or one formatted template), 1-10 distinct leftover tokens, and
   either every token looks like suppression (:data:`SUPPRESSION_RE`: ``*``,
   ``<10``, ``n<10``, ``N<10``, ``>95%``, ``<5%``, ``DS``, ``--`` ...) or the
   column has more than ``--factor-max`` distinct values; otherwise None and the
   FACTOR/ORDINAL rules decide (this keeps ``PK``/``KG``/``01``-``12`` grade
   levels categorical). Numbers with a leading zero (``05``) are codes, never
   numbers. Missing-like tokens (``NA``, ``--`` ...) are missing as in every
   typed column. Token texts are public (report) only when every token is
   suppression-like; otherwise they are kept PRIVATE (``params['_tokens']``) and
   reported as ``token_1`` ... (they are still written verbatim to the output).

Statistics and docs hooks: :func:`numeric_stats_extended`,
:func:`numbers_extended` and :func:`doc_fields` (see their docstrings).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

from ..errors import UnsupportedColumnError
from ..hints import id_word_header
from ..model import ColumnType
from .context import DetectContext, Detection
from .ids import repeated_key_digits

# ASCII digits only: ``float()`` also reads other scripts' digits (fullwidth, Arabic-Indic ...), which the
# renderer would write back as ASCII digits; such columns stay levels/text, keeping their raw form.
INT_RE = re.compile(r"[+-]?\d+", re.ASCII)
FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)", re.ASCII)
DIGITS_RE = re.compile(r"\d+", re.ASCII)
LEADING_ZERO_RE = re.compile(r"0\d+", re.ASCII)

# --------------------------------------------------------------------------- AREA B vocabularies
#: Formats whose cells are stored as text: the only ones ``--decimal ,`` applies to.
TEXT_FORMATS = frozenset({"csv", "html_xls"})
#: Currency affixes recognised in formatted numbers (design 3 #17): EUR, $, euro sign, pound sign.
CURRENCIES = ("EUR", "$", "\u20ac", "\u00a3")
POUND = "\u00a3"
_CUR = "(?:EUR|\\$|\u20ac|\u00a3)"
_SP = "[ \u00a0]*"
#: Valid skeletons: a value with its number replaced by '#'.
SKELETON_RE = re.compile(
    rf"(?P<op>\()?{_SP}(?P<os>[+-])?{_SP}(?P<pre>{_CUR})?{_SP}(?P<is>[+-])?{_SP}(?P<ip>\()?{_SP}#"
    rf"{_SP}(?P<ipc>\))?{_SP}(?P<suf>%|{_CUR})?{_SP}(?P<opc>\))?")
_SPLIT = {".": re.compile(r"([^0-9.]*)([0-9.,]*[0-9])([^0-9]*)"),
          ",": re.compile(r"([^0-9,]*)([0-9.,]*[0-9])([^0-9]*)")}
_CORE = {".": re.compile(r"(?P<int>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)?(?:\.(?P<frac>[0-9]+))?"),
         ",": re.compile(r"(?P<int>[0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)?(?:,(?P<frac>[0-9]+))?")}
#: Thousands separator for each decimal mark.
GROUP_SEP = {".": ",", ",": "."}
#: Suppression-like tokens (design 3 #18; the prototype's rule).
SUPPRESSION_RE = re.compile(
    r"\*+|[<>\u2264\u2265]=?\s*[0-9]+(?:\.[0-9]+)?\s*%?|n\s*[<>]=?\s*[0-9]+|ds|s|--|n/?r|\^", re.IGNORECASE)
SCI_RE = re.compile(r"(?P<sign>[+-]?)(?:(?P<int>[0-9]+)(?:\.(?P<frac>[0-9]*))?|\.(?P<frac2>[0-9]+))"
                    r"(?P<e>[eE])(?P<esign>[+-]?)(?P<exp>[0-9]+)")
EXCEL_SCI_RE = re.compile(r"[0-9](?:\.[0-9]{1,5})?E\+[0-9]{2}")
EXCEL_SCI_MIN_EXPONENT = 11
#: Excel's General format writes integers below 1E+11 (up to 11 digits) in full.
EXCEL_PLAIN_RE = re.compile(r"[0-9]{1,11}")
#: Upper bound of the generator's absolute rounding (the smallest float64 subnormal needs about 340).
MAX_ROUND_DECIMALS = 400
MAX_TOKENS = 10
MIN_NUMERIC_SHARE = 0.5
_AREA_B_TYPES = frozenset({ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL})


# --------------------------------------------------------------------------- FOUNDATION rules
def is_digit_code(ctx: DetectContext) -> bool:
    """Design 3 #8: digit strings with a leading zero are codes, never numbers."""
    vals = ctx.typed_values
    if not vals:
        return False
    if not all(DIGITS_RE.fullmatch(v) for v in vals):
        return False
    return any(LEADING_ZERO_RE.fullmatch(v) for v in vals)


def digit_code_params(ctx: DetectContext) -> dict[str, Any]:
    widths: dict[int, int] = {}
    for v, c in ctx.typed_values.items():
        widths[len(v)] = widths.get(len(v), 0) + c
    return {"width_pmf": {str(k): v for k, v in sorted(widths.items())}, "fixed_width": len(widths) == 1}


def detect_integer_year(ctx: DetectContext) -> Detection | None:
    """Design 3 #11: 4 digits, all within 1900-2100 (KCSE, Yr_JoinCampus)."""
    vals = ctx.typed_values
    if not vals or not all(len(v) == 4 and v.isascii() and v.isdigit() and 1900 <= int(v) <= 2100 for v in vals):
        return None
    ints = [int(v) for v in vals]
    return Detection(
        ColumnType.INTEGER_YEAR,
        params={"min": min(ints), "max": max(ints)},
        reason="all values are 4-digit years within 1900-2100",
    )


def _int_part(v: str) -> int:
    """The exact integer part of plain number text (``'+72.0'`` -> 72, ``'-.0'`` -> 0)."""
    head = v.split(".", 1)[0].lstrip("+")
    return int(head) if head not in ("", "-") else 0


def _decimals(v: str) -> int:
    i = v.find(".")
    return 0 if i < 0 else len(v) - i - 1


def _integer_kind(ctx: DetectContext, ints: list[int]) -> tuple[str, str, list[Any]]:
    distinct = len(ints)
    lo, hi = min(ints), max(ints)
    density = distinct / (hi - lo + 1)
    limit = max(ctx.factor_max, 50)
    if distinct <= limit:
        return "discrete", f"{distinct} distinct values <= {limit}", []
    if density >= 0.5:
        return "discrete", f"density {density:.2f} >= 0.5", []
    if ctx.hints.code:
        return "discrete", f"code header hint '{ctx.hints.code}'", []
    return "continuous", f"{distinct} distinct values, density {density:.3f} < 0.5", [density]


def _text_style(values: dict[str, int]) -> tuple[bool, bool]:
    """(leading_zero, plus_sign) of plain number text: the row-weighted majority form.

    ``leading_zero``: values below 1 in magnitude written ``0.5`` vs ``.5`` (tie or
    none: ``0.5``). ``plus_sign``: positive values written ``+5`` vs ``5`` (tie or
    none: no ``+``).
    """
    kept = omitted = plus = noplus = 0
    for v, c in values.items():
        body = v.lstrip("+-")
        if body.startswith("."):
            omitted += c
        elif body.startswith("0."):
            kept += c
        if v.startswith("-"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if f > 0:
            if v.startswith("+"):
                plus += c
            else:
                noplus += c
    return kept >= omitted, plus > noplus


def _trailing_point(values: dict[str, int]) -> bool:
    """True when most values without decimal digits are written with a bare point (``12.``, SAS/Fortran style).

    The point is part of the shape: pandas reads ``12.`` as float64 but ``12`` as int64.
    """
    bare = whole = 0
    for v, c in values.items():
        if v.endswith("."):
            bare += c
        elif "." not in v:
            whole += c
    return bare > whole


def detect_numeric(ctx: DetectContext) -> Detection | None:
    """INTEGER (discrete/continuous) or FLOAT from plain numeric text; None otherwise."""
    from ..model import Message

    vals = list(ctx.typed_values)
    if not vals:
        return None
    if native_decimal(ctx.native) is not None:
        return None  # AREA B: native Arrow decimal -> DECIMAL (detect_numeric_extended)
    if all(INT_RE.fullmatch(v) for v in vals):
        _, plus_sign = _text_style(ctx.typed_values)
        ints = sorted({int(v) for v in vals})
        sub, why, extra = _integer_kind(ctx, ints)
        params: dict[str, Any] = {"min": ints[0], "max": ints[-1], "decimals": 0, "float_text": False,
                                  "plus_sign": plus_sign}
        det = Detection(ColumnType.INTEGER, subtype=sub, params=params, reason=f"all values are integers; {sub}: {why}")
        if sub == "continuous":
            # K-02: 'Admission No', 'Roll No' get ID advice too, and so do repeated keys of a state student ID's width
            # under a header no rule knows ('NYSSIS', harden 2 A2-3): --force-factor would write every one
            by_header = bool(ctx.hints.id_hint or id_word_header(ctx.name))
            if by_header or repeated_key_digits(ctx, det):
                # hardening pass 3: an ID-like header ('Perm ID', 'Student_Key') may hold person IDs, whose real
                # values the interpolation and the exact extremes write; --force-factor would write every one
                det.messages.append(Message(
                    "warning", "possible_sparse_code",
                    f"integer column with {len(ints)} distinct values and density {extra[0]:.3f} "
                    + ("under an ID-like header" if by_header else "holding repeated ID-like keys")
                    + ": generated as continuous (interpolated values, real extremes); if these are person or "
                    "record IDs, --id-col gives them synthetic IDs (--person-id only when the column numbers the "
                    "row's own person, never for incident or record numbers); if they are codes that identify no "
                    "one, --force-factor keeps only the observed values",
                    column=ctx.name, fix=f"--id-col {ctx.name} (person or record IDs; --person-id {ctx.name} only "
                                         "when it numbers the row's own person), "
                                         f"--force-factor {ctx.name} only for codes that identify no one",
                ))
            else:
                det.messages.append(Message(
                    "warning", "possible_sparse_code",
                    f"integer column with {len(ints)} distinct values and density {extra[0]:.3f}: generated as "
                    "continuous (interpolated values); if these are codes, --force-factor keeps only the observed "
                    "values",
                    column=ctx.name, fix=f"--force-factor {ctx.name}",
                ))
        return det
    if decimal_mark(ctx) != ".":
        return None  # --decimal , on a text file: '.' is a thousands separator (NUMERIC_FORMATTED)
    if all(FLOAT_RE.fullmatch(v) for v in vals):
        floats = [float(v) for v in vals]
        if not all(math.isfinite(f) for f in floats):
            return None
        leading_zero, plus_sign = _text_style(ctx.typed_values)
        decs = [_decimals(v) for v in vals]
        point = {"trailing_point": True} if _trailing_point(ctx.typed_values) else {}
        # integer-valued read from the text (only zeros after the point): float64 would call 9007199254740993.5
        # an integer and lose its fraction (R2-12)
        if all("." not in v or not v.split(".", 1)[1].strip("0") for v in vals):
            ints = sorted({_int_part(v) for v in vals})
            sub, why, extra = _integer_kind(ctx, ints)
            params = {"min": ints[0], "max": ints[-1], "decimals": max(decs), "min_decimals": min(decs),
                      "float_text": True, "leading_zero": leading_zero, "plus_sign": plus_sign, **point}
            det = Detection(ColumnType.INTEGER, subtype=sub, params=params,
                            reason=f"integer-valued decimals (e.g. 72.0, form kept); {sub}: {why}")
            if sub == "continuous":
                det.messages.append(Message(
                    "warning", "possible_sparse_code",
                    f"integer column with {len(ints)} distinct values and density {extra[0]:.3f}: generated as continuous",
                    column=ctx.name))
            return det
        modal: Counter = Counter()
        for v, d in zip(vals, decs):
            modal[d] += ctx.typed_values[v]
        params = {
            "min": min(floats), "max": max(floats), "decimals": max(decs), "min_decimals": min(decs),
            "modal_decimals": modal.most_common(1)[0][0],
            "fixed_decimals": len(set(decs)) == 1, "leading_zero": leading_zero, "plus_sign": plus_sign, **point,
        }
        return Detection(ColumnType.FLOAT, params=params,
                         reason=f"all values are decimal numbers (up to {max(decs)} decimals)")
    return None


def forced_numeric(ctx: DetectContext, t: ColumnType) -> Detection:
    """Parameters for ``--force-type COL=INTEGER|FLOAT|INTEGER_YEAR`` (fails on non-numbers)."""
    det = detect_numeric(ctx) if ctx.typed_values else Detection(ColumnType.INTEGER, subtype="discrete")
    if det is None and t is ColumnType.FLOAT:
        det = detect_scientific(ctx)
    if det is None:
        bad = sum(1 for v in ctx.typed_values if not FLOAT_RE.fullmatch(v))
        raise UnsupportedColumnError(
            f"column {ctx.name!r}: --force-type {t.value} but {bad} distinct value(s) are not plain numbers",
            fix=f"remove --force-type for {ctx.name!r} or use --force-factor {ctx.name}",
        )
    if t is ColumnType.FLOAT and det.type is ColumnType.INTEGER:
        p = det.params
        det = Detection(ColumnType.FLOAT, params={
            "min": float(p["min"]), "max": float(p["max"]), "decimals": p.get("decimals", 0),
            "min_decimals": p.get("min_decimals", p.get("decimals", 0)), "fixed_decimals": True,
            "leading_zero": p.get("leading_zero", True), "plus_sign": p.get("plus_sign", False),
            **({"trailing_point": True} if p.get("trailing_point") else {})})
    elif t is ColumnType.INTEGER and det.type is ColumnType.FLOAT:
        raise UnsupportedColumnError(
            f"column {ctx.name!r}: --force-type INTEGER but values have decimals",
            fix=f"use --force-type {ctx.name}=FLOAT")
    elif t is ColumnType.INTEGER_YEAR:
        det = Detection(ColumnType.INTEGER_YEAR, params=dict(det.params))
    det.source = "force-type"
    det.reason = f"--force-type {t.value}"
    return det


# --------------------------------------------------------------------------- AREA B: number templates
class ParsedNumber(NamedTuple):
    """One formatted number split into its value and its text form."""

    value: float
    skeleton: str  # the text with the number replaced by '#': '$#', '($#)', '-#%', '+#' ...
    negative: bool
    affix: tuple[str | None, str | None]  # (currency prefix, currency or % suffix)
    int_digits: int  # digits of the integer part (separators removed)
    grouped: bool  # the integer part uses the thousands separator
    decimals: int
    lead_zero: bool | None  # '0.5' True, '.5' False, None when |x| >= 1 or no decimals
    feature: bool  # anything a plain number does not have (affix, parens, grouping, spaces, decimal comma)
    code: bool  # integer part with a leading zero ('05'): a code, never a number


def decimal_mark(ctx: DetectContext) -> str:
    """',' when ``--decimal ,`` was given and the file stores cells as text; '.' otherwise."""
    mark = getattr(ctx.settings, "decimal", ".") or "."
    return "," if mark == "," and ctx.format in TEXT_FORMATS else "."


def parse_number(s: str, mark: str = ".") -> ParsedNumber | None:
    """Parse a stripped value as a (possibly formatted) number; None if it is not one.

    Accepted: optional sign or parentheses, an optional currency prefix
    (``$``, ``EUR``, euro, pound) with the sign before or after it, digits with
    optional thousands separators (``,`` or, with ``mark=','``, ``.``) and
    decimals, an optional ``%`` or currency suffix, spaces between the parts.
    """
    m = _SPLIT[mark].fullmatch(s)
    if m is None:
        return None
    lead, core, trail = m.groups()
    skeleton = lead + "#" + trail
    info = _skeleton_info(skeleton)
    if info is None:
        return None
    c = _CORE[mark].fullmatch(core)
    if c is None:
        return None
    int_part, frac = c.groups()
    if int_part is None and frac is None:
        return None
    negative, affix, plain_affixes = info
    digits = int_part.replace(GROUP_SEP[mark], "") if int_part else ""
    number = float((digits or "0") + ("." + frac if frac else ""))
    if not math.isfinite(number):
        return None
    lead_zero = (int_part is not None) if frac is not None and digits in ("", "0") else None
    grouped = int_part is not None and GROUP_SEP[mark] in int_part
    feature = not plain_affixes or grouped or (mark == "," and frac is not None)
    return ParsedNumber(-number if negative else number, skeleton, negative and number != 0, affix, len(digits),
                        grouped, len(frac) if frac else 0, lead_zero, feature,
                        len(digits) > 1 and digits[0] == "0")


@lru_cache(maxsize=4096)
def _skeleton_info(skeleton: str) -> tuple[bool, tuple[str | None, str | None], bool] | None:
    """(negative, (prefix, suffix), plain) of a valid skeleton, else None (cached: few per column).

    ``plain``: the skeleton is ``'#'``, ``'-#'`` or ``'+#'`` (no format feature).
    """
    k = SKELETON_RE.fullmatch(skeleton)
    if k is None:
        return None
    op, os_, pre, is_, ip, ipc, suf, opc = k.group("op", "os", "pre", "is", "ip", "ipc", "suf", "opc")
    # the closing ')' may match either slot ('($#)': op + ipc), so only the counts must agree
    n_open, n_close = bool(op) + bool(ip), bool(ipc) + bool(opc)
    if n_open != n_close or n_open > 1:
        return None
    paren = n_open == 1
    sign = os_ or is_
    if (os_ and is_) or (paren and sign) or (is_ and not pre) or (ip and not pre) or (pre and suf):
        return None
    return paren or sign == "-", (pre, suf), skeleton in ("#", "-#", "+#")


def number_value(s: str, mark: str, affix: tuple[str | None, str | None]) -> float:
    """Fast numeric reading of one value with a known format: NaN unless it parses with ``affix``."""
    m = _SPLIT[mark].fullmatch(s)
    if m is None:
        return math.nan
    lead, core, trail = m.groups()
    info = _skeleton_info(lead + "#" + trail)
    if info is None or info[1] != affix:
        return math.nan
    c = _CORE[mark].fullmatch(core)
    if c is None:
        return math.nan
    int_part, frac = c.groups()
    if int_part is None and frac is None:
        return math.nan
    digits = int_part.replace(GROUP_SEP[mark], "") if int_part else "0"
    number = float(digits + ("." + frac if frac else ""))
    return -number if info[0] else number


def number_text(s: str, mark: str, affix: tuple[str | None, str | None]) -> str | None:
    """Exact plain text of one value with a known format (``'$1,234.50'`` -> ``'1234.50'``, ``'(12)'`` ->
    ``'-12'``); None unless it parses with ``affix``. The exact counterpart of :func:`number_value` for numbers
    wider than float64 (R2-12)."""
    m = _SPLIT[mark].fullmatch(s)
    if m is None:
        return None
    lead, core, trail = m.groups()
    info = _skeleton_info(lead + "#" + trail)
    if info is None or info[1] != affix:
        return None
    c = _CORE[mark].fullmatch(core)
    if c is None:
        return None
    int_part, frac = c.groups()
    if int_part is None and frac is None:
        return None
    text = (int_part.replace(GROUP_SEP[mark], "") if int_part else "0") + ("." + frac if frac else "")
    return "-" + text if info[0] else text


def _majority(counter: Counter) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def _unsigned(skeleton: str) -> str:
    """A positive skeleton derived from a negative one: '($#)' -> '$#', '-$ #' -> '$ #'."""
    s = skeleton
    for ch in "()+-":
        s = s.replace(ch, "")
    return s.strip(" \u00a0")


def learn_format(items: list[tuple[ParsedNumber, int]], mark: str) -> dict[str, Any] | None:
    """The column's number format from parsed values with counts; None if affixes differ.

    Returns a JSON-safe dict (``params['format']``): ``decimal_mark``,
    ``group_sep`` (None = no grouping), ``decimals`` / ``min_decimals`` /
    ``fixed_decimals``, ``leading_zero``, ``affix`` ([prefix, suffix]) and the
    majority skeletons ``positive``, ``negative`` (None if no negative value)
    and ``zero``.
    """
    if not items or len({p.affix for p, _ in items}) != 1:
        return None
    pos: Counter = Counter()
    neg: Counter = Counter()
    zero: Counter = Counter()
    decs: Counter = Counter()
    grouped = ungrouped = kept = omitted = 0
    for p, c in items:
        (zero if p.value == 0 else neg if p.negative else pos)[p.skeleton] += c
        if p.int_digits >= 4:
            if p.grouped:
                grouped += c
            else:
                ungrouped += c
        decs[p.decimals] += c
        if p.lead_zero is True:
            kept += c
        elif p.lead_zero is False:
            omitted += c
    neg_sk = _majority(neg)
    pos_sk = _majority(pos) or _unsigned(neg_sk or _majority(zero) or "#")
    zero_sk = _majority(zero) or pos_sk.replace("+", "", 1).strip(" \u00a0")
    pre, suf = items[0][0].affix
    keys = sorted(decs)
    return {
        "decimal_mark": mark,
        "group_sep": GROUP_SEP[mark] if grouped and grouped >= ungrouped else None,
        "decimals": keys[-1],
        "min_decimals": keys[0],
        "fixed_decimals": len(keys) == 1,
        "leading_zero": kept >= omitted,
        "affix": [pre, suf],
        "positive": pos_sk,
        "negative": neg_sk,
        "zero": zero_sk,
    }


def parse_with_format(s: str, fmt: dict[str, Any]) -> float:
    """The number of ``s`` read with the column's format (NaN when it does not conform)."""
    affix = fmt.get("affix") or (None, None)
    return number_value(s, fmt.get("decimal_mark", "."), (affix[0], affix[1]))


def format_pattern(fmt: dict[str, Any]) -> str:
    """Excel-like display of a number format: ``'$#,##0.00;($#,##0.00)'``."""
    mark, grp = fmt["decimal_mark"], fmt.get("group_sep")
    core = f"#{grp}##0" if grp else ("0" if fmt.get("leading_zero", True) else "#")
    d, min_d = int(fmt["decimals"]), int(fmt["min_decimals"])
    if d:
        core += mark + ("0" * d if fmt["fixed_decimals"] else "0" * min_d + "#" * (d - min_d))
    parts = [fmt["positive"].replace("#", core)]
    if fmt.get("negative"):
        parts.append(fmt["negative"].replace("#", core))
    return ";".join(parts)


def _num(v: float) -> float | int:
    return int(v) if float(v).is_integer() and abs(v) < 2**53 else float(v)


def _format_params(items: list[tuple[ParsedNumber, int]], fmt: dict[str, Any]) -> dict[str, Any]:
    vals = [p.value for p, _ in items]
    lo, hi = min(vals), max(vals)
    integer_valued = all(float(v).is_integer() for v in vals)
    return {"min": _num(lo), "max": _num(hi), "decimals": fmt["decimals"], "integer_valued": integer_valued,
            "format": fmt, "template": format_pattern(fmt),
            "negative_style": fmt["negative"].replace("#", "x") if fmt.get("negative") else None}


def _ascii(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode("ascii")


# --------------------------------------------------------------------------- AREA B: DECIMAL
_DEC_TYPE_RE = re.compile(r"decimal(?:128|256)?\(\s*([0-9]+)\s*,\s*(-?[0-9]+)\s*\)", re.IGNORECASE)


def native_decimal(native: dict[str, Any] | None) -> tuple[int, int] | None:
    """(precision, scale) of a native Arrow decimal column, or None (see the module docstring)."""
    t = _native_decimal_type(native)
    if t is None:
        return None
    p, s = getattr(t, "precision", None), getattr(t, "scale", None)
    if isinstance(p, int) and isinstance(s, int):
        return p, s
    m = _DEC_TYPE_RE.fullmatch(str(t).strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def native_decimal_width(native: dict[str, Any] | None) -> int | None:
    """128 or 256 (the Arrow decimal width) when the native type says it, else None."""
    t = _native_decimal_type(native)
    text = str(t).lower() if t is not None else ""
    return 256 if "256" in text.split("(")[0] else 128 if "128" in text.split("(")[0] else None


def _native_decimal_type(native: dict[str, Any] | None) -> Any:
    if not native:
        return None
    for key in ("arrow_type", "type", "dtype"):
        t = native.get(key)
        if t is None:
            continue
        p, s = getattr(t, "precision", None), getattr(t, "scale", None)
        if isinstance(p, int) and isinstance(s, int) and "decimal" in str(t).lower():
            return t
        if _DEC_TYPE_RE.fullmatch(str(t).strip()):
            return t
    return None


def decimal_type_name(precision: int, scale: int, width: int | None = None) -> str:
    """``decimal128(p, s)``, or ``decimal256(p, s)`` for a 256-bit source or a precision above 38."""
    bits = width or (128 if precision <= 38 else 256)
    return f"decimal{bits}({precision}, {scale})"


def _float_or_nan(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def decimal_detection(ctx: DetectContext, precision: int, scale: int, source: str = "native",
                      width: int | None = None) -> Detection | None:
    """DECIMAL detection for a decimal(precision, scale) column; None if a value is not a number.

    Format areas may return this from their ``detect_native`` for Arrow
    ``decimal128``/``decimal256`` columns (values: canonical decimal strings).
    """
    floats = [_float_or_nan(v) for v in ctx.typed_values]
    if any(not math.isfinite(f) for f in floats):
        return None
    params: dict[str, Any] = {"precision": int(precision), "scale": int(scale), "decimals": max(int(scale), 0),
                              "fixed_decimals": True, "arrow_type": decimal_type_name(int(precision), int(scale), width)}
    if floats:
        params.update(min=_num(min(floats)), max=_num(max(floats)))
    reason = (f"Arrow {params['arrow_type']}" if source == "native"
              else f"--force-type DECIMAL: decimal({precision}, {scale}) from the values")
    return Detection(ColumnType.DECIMAL, params=params, source=source, reason=reason)


def _decimal_from_values(ctx: DetectContext) -> tuple[int, int] | None:
    """(precision, scale) implied by plain numbers (``--force-type COL=DECIMAL`` on text).

    Every value is written with ``scale`` decimals, so the precision is the most
    integer digits plus the scale (``12345.6`` and ``1.23`` need decimal(7, 2)).
    """
    int_digits = scale = 0
    for v in ctx.typed_values:
        if not FLOAT_RE.fullmatch(v):
            return None
        body = v.lstrip("+-")
        ip, _, fp = body.partition(".")
        int_digits = max(int_digits, len(ip.lstrip("0")))
        scale = max(scale, len(fp))
    return max(int_digits + scale, 1), scale


# --------------------------------------------------------------------------- AREA B: scientific notation
def _exponent(v: float) -> int:
    return int(math.floor(math.log10(abs(v)))) if v != 0 else 0


def _sig_digits(v: str) -> int:
    """Significant digits written in a plain number ('0.0253' -> 3, '12.50' -> 4, '1200' -> 2)."""
    body = v.lstrip("+-")
    digits = body.replace(".", "").lstrip("0") if "." in body else body.lstrip("0").rstrip("0")
    return max(len(digits), 1)


def _plain_style(items: list[tuple[str, int]]) -> dict[str, Any]:
    """Text style of the plain values of a scientific column.

    Variable decimals there come from significant-digit writers (Python
    ``repr``, R), so ``sig_digits`` (the most significant digits written) is
    kept and the renderer writes plain values to that many significant digits.
    """
    decs: Counter = Counter()
    for v, c in items:
        decs[_decimals(v)] += c
    keys = sorted(decs) or [0]
    lz, plus = _text_style(dict(items))
    style: dict[str, Any] = {"decimals": keys[-1], "min_decimals": keys[0], "fixed_decimals": len(keys) == 1,
                             "leading_zero": lz, "plus_sign": plus}
    if len(keys) > 1:
        style["sig_digits"] = max(_sig_digits(v) for v, _ in items)
    return style


def detect_scientific(ctx: DetectContext) -> Detection | None:
    """FLOAT (subtype ``scientific``): plain/scientific numbers with at least one exponent."""
    from ..model import Message

    if decimal_mark(ctx) != ".":
        return None
    tv = ctx.typed_values
    sci: list[tuple[str, re.Match[str], int]] = []
    plain: list[tuple[str, int]] = []
    for v, c in tv.items():
        m = SCI_RE.fullmatch(v)
        if m:
            sci.append((v, m, c))
        elif FLOAT_RE.fullmatch(v):
            plain.append((v, c))
        else:
            return None
    if not sci:
        return None
    floats = {v: float(v) for v in tv}
    if not all(math.isfinite(f) for f in floats.values()):
        return None
    upper = sum(c for _, m, c in sci if m.group("e") == "E")
    lower = sum(c for _, m, c in sci if m.group("e") == "e")
    plus = sum(c for v, m, c in sci if m.group("esign") == "+")
    noplus = sum(c for v, m, c in sci if m.group("esign") == "")
    mant_decs: Counter = Counter()
    for _, m, c in sci:
        mant_decs[len(m.group("frac") or m.group("frac2") or "")] += c
    mkeys = sorted(mant_decs)
    sci_exps = sorted({_exponent(floats[v]) for v, _, _ in sci})
    plain_exps = sorted({_exponent(floats[v]) for v, _ in plain})
    n_sci = sum(c for _, _, c in sci)
    n_all = sum(tv.values())
    plain_style = _plain_style(plain) if plain else None
    # absolute rounding for the continuous generator: fine enough for the smallest value (the generator's
    # rounding leaves values alone where a float64 has no digit at that position, so no cap by the largest
    # value is needed: a cap would round the small values of a wide-range column to zero)
    need = max([mkeys[-1] - e for e in sci_exps] + [plain_style["decimals"] if plain_style else 0])
    decimals = int(max(0, min(need, MAX_ROUND_DECIMALS)))
    lz, plus_sign = _text_style(tv)
    scientific = {
        "e_char": "E" if upper >= lower else "e",
        "exp_plus": plus >= noplus if plus + noplus else True,
        "exp_digits": min(len(m.group("exp")) for _, m, _ in sci),
        "mantissa_decimals": mkeys[-1],
        "mantissa_min_decimals": mkeys[0],
        "mantissa_fixed": len(mkeys) == 1,
        "share": round(n_sci / n_all, 6),
        "sci_exponents": sci_exps,
        "plain_exponents": plain_exps,
        "plain": plain_style,
    }
    vals = list(floats.values())
    # Excel General: integers up to 11 digits in full, longer ones as d.dddddE+XX (6 significant digits)
    excel = (all(EXCEL_SCI_RE.fullmatch(v) for v, _, _ in sci) and all(EXCEL_PLAIN_RE.fullmatch(v) for v, _ in plain)
             and all(f.is_integer() for f in vals) and min(sci_exps) >= EXCEL_SCI_MIN_EXPONENT)
    params: dict[str, Any] = {
        "min": min(vals), "max": max(vals), "decimals": decimals, "min_decimals": 0, "fixed_decimals": False,
        "leading_zero": lz, "plus_sign": plus_sign, "scientific": scientific, "excel_mangled": excel,
    }
    det = Detection(ColumnType.FLOAT, subtype="scientific", params=params,
                    reason=f"numbers in scientific notation ({round(100 * n_sci / n_all, 1)}% of rows; "
                           f"exponent style {scientific['e_char']}{'+' if scientific['exp_plus'] else ''}"
                           f"{'0' * scientific['exp_digits']})")
    if excel:
        # suspected IDs (D21): value-revealing statistics are left out of the report (profile.stats redacts
        # 'id_candidate' columns) and the real min/max are kept private
        params["id_candidate"] = "Excel-mangled long numbers (IDs?)"
        params["_min"], params["_max"] = params.pop("min"), params.pop("max")
        det.messages.append(Message(
            "warning", "excel_scientific_ids",
            f"values look like long numbers (IDs?) that Excel displayed in scientific notation (d.ddddE+XX, at most "
            f"{mkeys[-1] + 1} significant digits): precision was lost upstream; the synthetic values use the same "
            "notation and include the exact real min and max (D14); their values are left out of the report",
            column=ctx.name,
            fix=f"re-export the source with this column formatted as text (or as a number with 0 decimals) to keep "
                f"every digit; --no-exact-extremes {ctx.name} keeps the real min and max out of the output"))
    return det


# --------------------------------------------------------------------------- AREA B: formatted / with tokens
def detect_formatted(ctx: DetectContext, forced: bool = False) -> Detection | None:
    """NUMERIC_FORMATTED (design 3 #17) or None (see the module docstring)."""
    mark = decimal_mark(ctx)
    items: list[tuple[ParsedNumber, int]] = []
    for v, c in ctx.typed_values.items():
        p = parse_number(v, mark)
        if p is None:
            return None
        items.append((p, c))
    if not items or any(p.code for p, _ in items):
        return None
    if not forced and not any(p.feature for p, _ in items):
        return None
    fmt = learn_format(items, mark)
    if fmt is None:
        return None
    params = _format_params(items, fmt)
    det = Detection(ColumnType.NUMERIC_FORMATTED, params=params,
                    reason=f"formatted numbers, template {_ascii(params['template'])}"
                           + (" (decimal comma, --decimal ,)" if mark == "," else ""))
    _flag_ambiguous_grouping(det, items, fmt)
    return det


def _flag_ambiguous_grouping(det: Detection, items: list[tuple[ParsedNumber, int]], fmt: dict[str, Any]) -> None:
    """Flag ``1,234``-only columns: thousands under the decimal mark in use, or decimals under the other one.

    Ambiguous when no value has decimals and every grouped value has exactly one
    separator (4-6 integer digits); dollar and pound amounts are not flagged (their
    usual locales write '.' decimals). Detection is unchanged (the decimal mark
    comes from ``--decimal`` only); the flag shows in ``inspect`` and the report.
    """
    if not fmt.get("group_sep") or fmt["decimals"] != 0 or {"$", POUND} & set(fmt.get("affix") or ()):
        return
    if not all(4 <= p.int_digits <= 6 for p, _ in items if p.grouped):
        return
    det.ambiguous = True
    if fmt["decimal_mark"] == ".":
        det.ambiguity = ("values like 1,234 are read with ',' as the thousands separator (1234); if ',' is the "
                         "decimal mark here (1.234), use --decimal ,")
    else:
        det.ambiguity = ("values like 1.234 are read with '.' as the thousands separator (1234, --decimal ,); if "
                         "'.' is the decimal mark here (1.234), run this file without --decimal ,")


def is_suppression_token(token: str) -> bool:
    return SUPPRESSION_RE.fullmatch(token.strip()) is not None


def detect_with_tokens(ctx: DetectContext, forced: bool = False) -> Detection | None:
    """NUMERIC_WITH_TOKENS (design 3 #18) or None (see the module docstring)."""
    from ..model import Message

    mark = decimal_mark(ctx)
    items: list[tuple[ParsedNumber, int]] = []
    tokens: dict[str, int] = {}
    for v, c in ctx.typed_values.items():
        p = parse_number(v, mark)
        if p is None:
            tokens[v] = c
            if len(tokens) > MAX_TOKENS:
                return None  # stop early on text columns
        else:
            items.append((p, c))
    if not items or (not tokens and not forced):
        return None
    total = sum(ctx.typed_values.values())
    numeric_rows = sum(c for _, c in items)
    share = numeric_rows / total
    if share < MIN_NUMERIC_SHARE and not forced:
        return None
    if any(p.code for p, _ in items):
        return None  # leading-zero numbers are codes
    fmt = learn_format(items, mark)
    if fmt is None:
        return None
    suppression = all(is_suppression_token(t) for t in tokens)
    n_distinct = ctx.casefold_distinct()
    if not forced and not suppression and n_distinct <= ctx.factor_max:
        return None  # FACTOR/ORDINAL decide (grade levels PK/KG/01-12)
    params = _format_params(items, fmt)
    token_list = sorted(tokens)
    params.update(numeric_share=round(share, 6), n_tokens=len(tokens), suppression_like=suppression)
    params["tokens" if suppression else "_tokens"] = token_list
    if suppression:
        why = f"{len(tokens)} suppression-like token(s)"
    else:
        why = f"{len(tokens)} other token(s), {n_distinct} distinct values > factor-max {ctx.factor_max}"
    det = Detection(ColumnType.NUMERIC_WITH_TOKENS, params=params,
                    reason=f"{round(100 * share, 1)}% numeric rows (template {_ascii(params['template'])}) "
                           f"with {why}")
    if not suppression and tokens:
        det.messages.append(Message(
            "info", "numeric_tokens_verbatim",
            f"{len(tokens)} non-numeric token(s) that do not look like suppression codes are written at their exact "
            "counts: verbatim when the D35 allowlist keeps them, else as same-shape pseudonyms (their texts are left "
            "out of the report)", column=ctx.name))
    return det


def detect_numeric_extended(ctx: DetectContext) -> Detection | None:
    """AREA B hook: formatted numbers, numbers with tokens, decimals, scientific notation.

    Called by ``detect/types.py`` right after :func:`detect_numeric` returned None
    (and before the FACTOR rule), only when the column is not a digit code. Also
    called for ``--force-type`` of NUMERIC_FORMATTED / NUMERIC_WITH_TOKENS /
    DECIMAL with ``ctx.forced_type`` set. Returns a Detection of type
    NUMERIC_FORMATTED, NUMERIC_WITH_TOKENS, DECIMAL or FLOAT (scientific), or
    None. ``ctx.settings.decimal`` holds ``--decimal``. Rules and order: see the
    module docstring. Forced types relax the gates (a formatted column needs no
    format feature; a column with tokens needs no 50% share, no suppression-like
    token and may have no token at all) but the values must still parse, the
    numbers must share one affix, leading-zero codes are refused and at most 10
    distinct tokens are allowed (tokens are written verbatim).
    """
    forced = ctx.forced_type
    if forced is ColumnType.DECIMAL:
        nd = native_decimal(ctx.native)
        if nd is not None:
            return decimal_detection(ctx, *nd, width=native_decimal_width(ctx.native))
        nd = _decimal_from_values(ctx)
        return None if nd is None else decimal_detection(ctx, *nd, source="force-type")
    if forced is ColumnType.NUMERIC_FORMATTED:
        return detect_formatted(ctx, forced=True)
    if forced is ColumnType.NUMERIC_WITH_TOKENS:
        return detect_with_tokens(ctx, forced=True)
    nd = native_decimal(ctx.native)
    if nd is not None:
        return decimal_detection(ctx, *nd, width=native_decimal_width(ctx.native))
    return detect_scientific(ctx) or detect_formatted(ctx) or detect_with_tokens(ctx)


# --------------------------------------------------------------------------- AREA B: statistics and docs
def _token_label(profile: Any) -> tuple[list[str], bool]:
    p = profile.params
    if "tokens" in p:
        return list(p["tokens"]), True
    return list(p.get("_tokens") or []), False


def numbers_extended(values: list[str], profile: Any) -> np.ndarray | None:
    """AREA B hook: the numeric view of AREA B values (``$1,234.50`` -> 1234.5).

    Called through ``profile.stats.numeric_view`` for NUMERIC_FORMATTED,
    NUMERIC_WITH_TOKENS and DECIMAL columns: by ``profile.table.build_model``
    (the result, sorted, becomes ``ColumnModel.sorted_numeric``) and by the
    pipeline for the synthetic values (fidelity KS check). ``values`` are
    stripped non-missing strings (possibly only the distinct ones). Returns a
    float array aligned with ``values``: NaN for tokens (``*``, ``<10``) and for
    any value that does not conform to the column's number format.
    """
    t = profile.type
    if t not in _AREA_B_TYPES:
        return None
    fmt = profile.params.get("format")
    if t is ColumnType.DECIMAL or fmt is None:
        read = _float_or_nan
    else:
        mark, affix = fmt.get("decimal_mark", "."), tuple(fmt.get("affix") or (None, None))

        def read(v: str) -> float:
            return number_value(v, mark, affix)  # type: ignore[arg-type]

    if len(values) > 2 * len(set(values)):  # many repeats (synthetic rows): parse each distinct value once
        cache = {v: read(v) for v in set(values)}
        return np.fromiter((cache[v] for v in values), dtype=float, count=len(values))
    return np.fromiter((read(v) for v in values), dtype=float, count=len(values))


def numeric_stats_extended(values: list[str], profile: Any) -> dict[str, Any]:
    """AREA B hook: extra statistics for AREA B types (template, token shares ...).

    Called by ``profile/stats.py`` for NUMERIC_FORMATTED, NUMERIC_WITH_TOKENS and
    DECIMAL columns with the stripped non-missing values as TEXT (real values in
    pass 1, synthetic in pass 2). Values are parsed with the column's own format
    (``params['format']``). Returns report-safe aggregates only: the FLOAT
    summary of the parsed numbers (min/max/mean/median/sd/quantiles,
    zero/negative shares) plus, per type:

    * NUMERIC_FORMATTED: ``template``, ``format_conformance`` (share of values
      that parse with the template; 1.0 for real and synthetic data);
    * NUMERIC_WITH_TOKENS: ``numeric_share``, ``token_counts`` and
      ``token_shares`` (token texts only when every token is suppression-like,
      otherwise ``token_1`` ...; tokens the D35 allowlist did not keep and their
      pseudonyms as ``pseudonymized``), the summary being that of the numeric part;
    * DECIMAL: ``precision`` and ``scale``.
    """
    return numeric_stats_extended_counts(Counter(values), profile)


def numeric_stats_extended_counts(counts: dict[str, int], profile: Any) -> dict[str, Any]:
    """AREA B statistics from counts in first-appearance order, without a row list."""
    from ..profile.stats import numeric_summary  # lazy: profile.stats imports this module

    t = profile.type
    if t not in _AREA_B_TYPES:
        return {}
    keys = list(counts)
    cnt = np.fromiter((counts[k] for k in keys), dtype=np.int64, count=len(keys))
    nums = numbers_extended(keys, profile)
    if nums is None:
        return {}
    ok = np.isfinite(nums)
    total = int(cnt.sum())
    st: dict[str, Any] = dict(numeric_summary(np.repeat(nums[ok], cnt[ok])))
    numeric_rows = int(cnt[ok].sum())
    p = profile.params
    if t is ColumnType.NUMERIC_FORMATTED:
        if p.get("template"):
            st["template"] = p["template"]
        st["format_conformance"] = round(numeric_rows / total, 6) if total else 1.0
    elif t is ColumnType.NUMERIC_WITH_TOKENS:
        from .text import pseudonymized  # lazy: detect.text imports this module

        labels, public = _token_label(profile)
        masked = frozenset(pseudonymized(profile))  # D35: tokens written as same-shape pseudonyms
        tok_counts: Counter = Counter()
        for k, c, good in zip(keys, cnt.tolist(), ok.tolist()):
            if good:
                continue
            if k in masked or (masked and k not in labels):
                name = "pseudonymized"  # the real tokens the D35 allowlist did not keep, and their pseudonyms
            elif k in labels:
                name = k if public else f"token_{labels.index(k) + 1}"
            else:
                name = "other"
            tok_counts[name] += c
        st["numeric_share"] = round(numeric_rows / total, 6) if total else 0.0
        st["token_counts"] = {k: int(v) for k, v in sorted(tok_counts.items())}
        st["token_shares"] = {k: round(v / total, 6) for k, v in sorted(tok_counts.items())} if total else {}
    elif t is ColumnType.DECIMAL:
        st["precision"] = p.get("precision")
        st["scale"] = p.get("scale")
    return st


def _r_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def doc_fields(profile: Any, syn_stats: dict[str, Any]) -> dict[str, Any]:
    """AREA B hook: extra DATA_DICTIONARY / schema.json fields for an AREA B column.

    Called by ``report/docs.py`` with the column profile and the SYNTHETIC
    statistics. Returns report-safe fields; a ``'readr'`` key replaces the type's
    readr letter. The R class stays the catalog's (numeric for formatted numbers
    and decimals: ``readr::parse_number`` returns a double; character for
    numbers with tokens, as stored). Formatted numbers and numbers with tokens
    are text in the file (readr ``c``) with a ``parse`` note and the R class of
    the parsed numbers (``parsed_r_class``: integer when every real value is a
    whole number, else numeric); decimals are ``d`` with their Arrow type.
    """
    t = profile.type
    p = profile.params
    if t is ColumnType.DECIMAL:
        prec, scale = p.get("precision"), p.get("scale")
        return {"readr": "d", "precision": prec, "scale": scale,
                "arrow_type": p.get("arrow_type") or decimal_type_name(int(prec), int(scale))}
    if t not in (ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS):
        return {}
    fmt = p.get("format") or {}
    mark = fmt.get("decimal_mark", ".")
    grp = fmt.get("group_sep")
    args = ["x"]
    if mark == "," or grp == ".":
        args.append(f"locale = readr::locale(decimal_mark = {_r_string(mark)}, "
                    f"grouping_mark = {_r_string(grp or ('.' if mark == ',' else ','))})")
    fields: dict[str, Any] = {"readr": "c", "template": p.get("template"), "decimal_mark": mark,
                              "grouping_mark": grp,
                              "parsed_r_class": "integer" if p.get("integer_valued") else "numeric"}
    neg = p.get("negative_style")
    if neg:
        fields["negative_style"] = neg
    if t is ColumnType.NUMERIC_WITH_TOKENS:
        labels, public = _token_label(profile)
        if public:
            fields["tokens"] = labels
            args.append("na = c(" + ", ".join(_r_string(x) for x in labels) + ")")
        else:
            fields["tokens"] = f"{len(labels)} non-numeric token(s) (texts not listed)"
        note = (f"text column: numbers with tokens; R: readr::parse_number({', '.join(args)}) "
                "(tokens are not numbers: give them as na =)")
    else:
        note = f"text column with a number template; R: readr::parse_number({', '.join(args)})"
    if neg and "(" in neg:
        note += "; negatives are written in parentheses: negate the values whose text starts with '('"
    fields["parse"] = note
    return fields
