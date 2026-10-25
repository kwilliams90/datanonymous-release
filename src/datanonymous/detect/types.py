"""The fixed, ordered type-detection pipeline (FOUNDATION; shared file).

:func:`detect_column` decides one column's type. Order (design 3 "Ordering
decisions" and override order; first match wins):

Overrides
  1. ``--force-type COL=TYPE``
  2. ``--pii-col COL=SUBTYPE``                      -> text.pii_subtype_detection
  3. ``--id-col`` / ``--person-id``                 -> IDENTIFIER
  4. ``--org-code`` / ``--force-factor`` / ``--force-text`` (explicit flags beat
     the default ID list)
  5. default ID name list (D7; off with ``--no-default-ids``) -> IDENTIFIER
  5b. person-ID header (D10: person word + ID word, ``Family_ID``,
     ``Guardian_Number``, ``Student Code``, ``SIS ID``, ``Student ID (Aeries)``;
     ``hints.person_id_hint``) -> IDENTIFIER (source ``person-id-hint``;
     ``--org-code`` / ``--force-factor`` override it)
  6. native typed storage                           -> registry handler ``detect_native`` (AREA E/F):
     only what the values cannot show (Arrow dictionaries and Stata value labels -> FACTOR/ORDINAL,
     typed dates/times); numbers stored as numbers (int, float, decimal, Excel/Stata numbers) are typed
     by the value rules, so every header rule applies to them (a native decimal is DECIMAL at rule 17)
  6b. header rules on native detections (R4-05): a native detection that the value rules would only
     reach after rule 16 (factors, numbers, codes, text) under a D10 org-code header -> ORG_CODE, as
     for text; native dates and ordered dictionaries (before rule 16 in the value order) stay. ID
     headers are overrides (3-5b) and PII headers the D12 guard below, for native columns as for text.

Value rules (on stripped distinct values with counts)
  7. EMPTY                (no non-missing value)
  8. CONSTANT / FLAG_SINGLE_TOKEN (one distinct value; FLAG when blanks exist)
  9. BOOLEAN              (known pair; 0/1, Y/N ambiguous; code/count hints block 0/1)
 10. ORDINAL              (vocabulary subset, >= 3 levels; bypasses factor-max)
 11. SCHOOL_YEAR          (regex family, or ending-year integers + school_year hint)
 12. DIGIT_CODE mark      (leading zero -> never numeric or date-as-number)
 13. dates hook           -> dates.detect_dates            (AREA A)
 14. text/PII hook        -> text.detect_text_semantics     (AREA C)
 15. INTEGER_YEAR         -> numeric.detect_integer_year
 16. ORG_CODE             (D10 header hint ``hints.org_code_hint``: org word + optional ID suffix;
                          bypasses factor-max; never person IDs or person roles. D39: known
                          non-person codes ``NCESSCH``, ``LEA_ID``, ``CDS_Code``, ``CDC``,
                          ``GL_Account``, ``Term_CRN``, ``ISBN`` by header; ``Fund``,
                          ``Function``, ``Object`` only over code values: ``header_org_code``)
 17. numeric              -> numeric.detect_numeric, then numeric.detect_numeric_extended (AREA B)
 18. FACTOR               (<= factor-max distinct after strip + case-fold; DIGIT_CODE kept as codes)
 19. overflow text        -> DIGIT_CODE (masks) or text.classify_overflow_text (AREA C refines)

Then the D12 guard ``text.apply_pii_precedence`` is applied to value/native
detections (never to explicit overrides), and K-02's fail-safe mark
(``ids.mark_weak_id``): a header with a student word, a student-ID code or a final
ID word over ID-like keys (``Student_SK``, ``Stu``, ``CALPADS SSID``,
``Admission No``) may be an ID column, which its table decides
(``profile.table``). The PII guard ends in the copy gate
(``text.copy_gate``), whose D35 allowlist decides which real labels are written
verbatim (rare ones get same-shape pseudonyms once the run's entity counts are
known). ``--force-type`` and ``--org-code`` detections pass the copy gate too
(they name a type, not the values); only ``--force-factor`` keeps every real
value.
"""

from __future__ import annotations

import re

from ..errors import UnsupportedColumnError
from ..hints import (AMBIGUOUS_BOOLEAN_PAIRS, BOOLEAN_PAIRS, NUMERIC_BOOLEAN_PAIRS, ORDINAL_VOCABS, known_code_hint,
                     school_year_start)
from ..model import ColumnType, Message
from . import dates, ids, numeric, text
from .context import DetectContext, Detection

_DATE_TYPES = frozenset({ColumnType.DATE, ColumnType.DATE_MIXED_FORMATS, ColumnType.DATE_AS_INT, ColumnType.DATETIME,
                         ColumnType.DATETIME_TZ, ColumnType.YEAR_MONTH, ColumnType.TIME, ColumnType.DURATION})
_AREA_B_TYPES = frozenset({ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL})
_PII_TYPES = frozenset({ColumnType.EMAIL, ColumnType.PERSON_NAME, ColumnType.PHONE, ColumnType.SSN, ColumnType.ZIP,
                        ColumnType.STATE, ColumnType.CITY, ColumnType.STREET_ADDRESS, ColumnType.IP, ColumnType.URL,
                        ColumnType.MULTI_SELECT})


def detect_column(ctx: DetectContext) -> Detection:
    """Run the ordered pipeline for one column and return its Detection."""
    det = _overrides(ctx)
    if det is None:
        det = _native(ctx)
        det = _value_rules(ctx) if det is None else _native_header_rules(ctx, det)
        det = text.apply_pii_precedence(ctx, det)
        det = ids.mark_weak_id(ctx, det)  # K-02: its table decides (profile.table)
    elif det.source in text.GATED_EXPLICIT_SOURCES:
        det = text.copy_gate(ctx, det)  # D35: --force-type / --org-code name the type; the values still follow it
    elif det.source == "force-factor":
        det = text.forced_copy_warning(ctx, det)  # D35's explicit opt-in keeps every value: never silently (R6-03)
    return det


#: Value-rule outcomes that come after the D10 org-code rule (rule 16): numbers (17), factors and codes (18),
#: overflow text (19). A native detection of one of these types gives way to the org-code header as the value
#: rules do (R4-05); the writer restores the native storage (decimal, dictionary, value labels) from the values.
_AFTER_ORG_CODE = frozenset({ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.DECIMAL, ColumnType.NUMERIC_FORMATTED,
                             ColumnType.NUMERIC_WITH_TOKENS, ColumnType.FACTOR, ColumnType.DIGIT_CODE,
                             ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.FREE_TEXT})


#: D39: a value of a finance segment code made of digit groups joined by dashes or spaces (``199-11-6119-00``,
#: ``10 1000 6110``) or by two or more dots (``10.1000.6110``; one dot is a decimal number); digits alone are codes
#: when the whole column has one width (:func:`finance_code_values`).
_SEGMENTED_CODE_RE = re.compile(r"\d+(?:[- ]\d+)+|\d+(?:\.\d+){2,}")
_DIGIT_RE = re.compile(r"\d")


def finance_code_values(values: dict[str, int]) -> bool:
    """D39: every value (stripped, non-missing) is a finance code: ASCII digits of one width of 2 or more (``199``,
    ``6119``, ``199116119``) or digit groups joined by dashes, spaces or two or more dots, all of one layout
    (``199-11-6119-00``, ``10.1000.6110``). Free text, amounts (``6119.50``, ``-12``, ``1,200``), single digits and
    widths that vary (counts, amounts) are not."""
    vals = [v for v in values if v]
    if not vals or not all(v.isascii() for v in vals):
        return False
    if all(v.isdigit() for v in vals):
        return len({len(v) for v in vals}) == 1 and len(vals[0]) >= 2
    return len({_DIGIT_RE.sub("9", v) for v in vals}) == 1 and bool(_SEGMENTED_CODE_RE.fullmatch(vals[0]))


def header_org_code(ctx: DetectContext) -> str | None:
    """D10/D39: the org-code word that makes the column an organisational code by its header (real values kept), or
    None: an org-code header (``hints.org_code_hint``, D39's known non-person codes among them: ``NCESSCH``,
    ``Term_CRN``) or a finance segment header (``hints.finance_segment_hint``: ``Fund``, ``Object_Code``) over finance
    codes (:func:`finance_code_values`); never a person-ID header."""
    h = ctx.hints
    if h.person_id_like:
        return None
    if h.org_code:
        return h.org_code
    if h.finance_segment and finance_code_values(ctx.typed_values):
        return h.finance_segment
    return None


def _org_code_hint_rule(ctx: DetectContext, source: str, what: str = "") -> Detection | None:
    """Rule 16 (D10): ORG_CODE (real values kept) for an org-code header that is not a person ID
    (:func:`header_org_code`), else None."""
    word = header_org_code(ctx)
    if word is None:
        return None
    if not ctx.hints.org_code:
        why = f"header names the finance code segment '{word}' over code values (D39, D10): real values kept"
    elif known_code_hint(ctx.name):
        why = f"header names a known non-person code '{word}' (D39, D10): real values kept"
    else:
        why = f"header matches org-code hint '{word}' (D10): real values kept"
    return _org_code(ctx, source, why + (f" (wins over {what})" if what else ""))


def _native_header_rules(ctx: DetectContext, det: Detection) -> Detection:
    """6b: the header rules of the value order apply to a native detection too (R4-05).

    Native storage only decides what the values cannot show; a header rule that the value rules apply before
    the detected type wins over it: an org-code header turns a native factor (Arrow dictionary, Stata value
    labels) or number into ORG_CODE, keeping the native parameters (dictionary sizes ...). Native dates and
    ordered dictionaries (ordinal levels) come before the org-code rule in the value order and stay.
    """
    if det.type not in _AFTER_ORG_CODE:
        return det
    org = _org_code_hint_rule(ctx, det.source, f"native {det.type.value}")
    if org is None:
        return det
    if det.type is ColumnType.FACTOR:  # dictionary / value-label facts (sizes, unused levels) stay
        org.params = {**det.params, **org.params}
    org.messages = det.messages + org.messages
    return org


# --------------------------------------------------------------------------- overrides
def _overrides(ctx: DetectContext) -> Detection | None:
    o = ctx.options
    if o.force_type:
        return _forced(ctx, ColumnType.parse(o.force_type))
    if o.pii_col:
        return text.pii_subtype_detection(ctx, o.pii_col)
    if o.id_col or o.person_id:
        flag = "--id-col" if o.id_col else "--person-id"
        return ids.identifier_detection(ctx, "id-col" if o.id_col else "person-id", f"{flag} (synthetic IDs, D8/D9)")
    if o.org_code:
        return _org_code(ctx, "org-code", "--org-code: real values kept as a factor (D10)")
    if o.force_factor:
        return _factor(ctx, source="force-factor", reason="--force-factor: real levels kept")
    if o.force_text:
        det = text.classify_overflow_text(ctx)
        det.source = "force-text"
        det.reason = "--force-text; " + det.reason
        return det
    if ctx.hints.default_id and not ctx.settings.no_default_ids:
        return ids.identifier_detection(ctx, "default-id", "header is on the default ID name list (D7)")
    if ctx.hints.person_id_like:
        return ids.identifier_detection(ctx, "person-id-hint",
                                        "person-ID header (person word + ID word, D10): synthetic IDs")
    return None


def _forced(ctx: DetectContext, t: ColumnType) -> Detection:
    reason = f"--force-type {t.value}"
    if t is ColumnType.IDENTIFIER:
        return ids.identifier_detection(ctx, "force-type", reason)
    if t in (ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.INTEGER_YEAR):
        return numeric.forced_numeric(ctx, t)
    if t in (ColumnType.LAT, ColumnType.LON):
        det = numeric.forced_numeric(ctx, ColumnType.FLOAT)
        det.type = t
        return det
    if t in _DATE_TYPES or t in _AREA_B_TYPES:
        ctx.forced_type = t
        det = dates.detect_dates(ctx) if t in _DATE_TYPES else numeric.detect_numeric_extended(ctx)
        if det is None or det.type is not t:
            area = "A (dates)" if t in _DATE_TYPES else "B (numeric formats)"
            raise UnsupportedColumnError(
                f"column {ctx.name!r}: --force-type {t.value} cannot be applied (type not available in this build "
                f"or values do not match; area {area})",
                fix=f"remove --force-type for {ctx.name!r}, or use --force-factor {ctx.name} / --force-text {ctx.name}",
            )
        det.source = "force-type"
        return det
    if t is ColumnType.BINARY_NESTED:
        raise UnsupportedColumnError(f"column {ctx.name!r}: BINARY_NESTED columns cannot be generated",
                                     fix=f"--null {ctx.name} or --exclude {ctx.name}")
    if t is ColumnType.EMPTY:
        if ctx.typed_values:
            raise UnsupportedColumnError(f"column {ctx.name!r} has values; --force-type EMPTY does not apply",
                                         fix=f"--null {ctx.name} writes the column as all-missing")
        return Detection(t, source="force-type", reason=reason)
    if t is ColumnType.ORDINAL:
        det = _ordinal(ctx)
        if det is None:
            levels = sorted(ctx.typed_values)
            det = Detection(t, params={"vocabulary": None, "levels": levels, "order_source": "sorted (forced)"})
        det.source, det.reason = "force-type", reason
        return det
    if t is ColumnType.DIGIT_CODE:
        params = numeric.digit_code_params(ctx)
        params["levels_mode"] = ctx.casefold_distinct() <= ctx.factor_max
        return Detection(t, params=params, source="force-type", reason=reason)
    if t is ColumnType.FACTOR:
        return _factor(ctx, source="force-type", reason=reason)
    if t is ColumnType.ORG_CODE:
        return _org_code(ctx, "force-type", reason)
    if t is ColumnType.SCHOOL_YEAR:
        det = _school_year(ctx, forced=True)
        if det is None:
            raise UnsupportedColumnError(f"column {ctx.name!r}: values are not school years",
                                         fix=f"remove --force-type for {ctx.name!r}")
        det.source = "force-type"
        return det
    det = Detection(t, source="force-type", reason=reason)
    if t in (ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.FREE_TEXT, ColumnType.CONSTANT,
             ColumnType.FLAG_SINGLE_TOKEN, ColumnType.BOOLEAN) or t in _PII_TYPES:
        if t is ColumnType.BOOLEAN:
            b = _boolean(ctx)
            if b is None:
                raise UnsupportedColumnError(f"column {ctx.name!r}: values are not a known boolean pair",
                                             fix=f"use --force-factor {ctx.name}")
            b.source, b.reason = "force-type", reason
            return b
        return text.finish_forced(ctx, det)  # AREA C: --min-cell / --factor-max / NA policy for its types
    raise UnsupportedColumnError(f"column {ctx.name!r}: --force-type {t.value} is not supported",
                                 fix=f"remove --force-type for {ctx.name!r}")


def _native(ctx: DetectContext) -> Detection | None:
    from ..io import registry

    try:
        handler = registry.get(ctx.format)
    except Exception:
        return None
    det = handler.detect_native(ctx)
    if det is not None and det.source == "value":
        det.source = "native"
    return det


# --------------------------------------------------------------------------- value rules
def _value_rules(ctx: DetectContext) -> Detection:
    tv = ctx.typed_values
    if not tv:
        return Detection(ColumnType.EMPTY, reason="no non-missing values")
    if len(tv) == 1:
        value = next(iter(tv))
        if ctx.n_blank:
            return Detection(ColumnType.FLAG_SINGLE_TOKEN, params={"token": value},
                             reason="one distinct token plus blanks")
        return Detection(ColumnType.CONSTANT, params={"value": value}, reason="one distinct non-missing value")
    for rule in (_boolean, _ordinal, _school_year):
        det = rule(ctx)
        if det is not None:
            return det
    ctx.digit_code = numeric.is_digit_code(ctx)
    det = dates.detect_dates(ctx)
    if det is not None:
        return det
    det = text.detect_text_semantics(ctx)
    if det is not None:
        return det
    if not ctx.digit_code:
        det = numeric.detect_integer_year(ctx)
        if det is not None:
            return det
    det = _org_code_hint_rule(ctx, "value")
    if det is not None:
        return det
    if not ctx.digit_code:
        det = numeric.detect_numeric(ctx) or numeric.detect_numeric_extended(ctx)
        if det is not None:
            return det
    n_levels = ctx.casefold_distinct()
    if n_levels <= ctx.factor_max:
        if ctx.digit_code:
            params = numeric.digit_code_params(ctx)
            params["levels_mode"] = True
            return Detection(ColumnType.DIGIT_CODE, params=params,
                             reason=f"leading-zero digit codes, {n_levels} distinct <= factor-max: codes kept as levels")
        return _factor(ctx, source="value", reason=f"{n_levels} distinct values <= factor-max {ctx.factor_max}")
    if ctx.digit_code:
        params = numeric.digit_code_params(ctx)
        params["levels_mode"] = False
        det = Detection(ColumnType.DIGIT_CODE, params=params,
                        reason=f"leading-zero digit codes, {n_levels} distinct > factor-max: same-shape masks")
    else:
        det = text.classify_overflow_text(ctx)
    if ctx.hints.pii in text.GUARDED_PII_TEXT or ctx.hints.free_text:
        det.messages.append(Message("info", "factor_overflow",
                                    f"{n_levels} distinct values > factor-max {ctx.factor_max}; personal/free text: "
                                    "real values never copied", column=ctx.name))
    else:
        det.messages.append(Message(
            "warning", "factor_overflow",
            f"{n_levels} distinct values > factor-max {ctx.factor_max}: generated as non-factor text "
            "(real labels not kept)", column=ctx.name,
            # sharing mode refuses both options: the real labels are never kept there
            fix=None if getattr(ctx.settings, "mode", "") == "sharing" else
            f"--force-factor {ctx.name} (or --org-code {ctx.name}) keeps the real labels"))
    return det


def _boolean(ctx: DetectContext) -> Detection | None:
    low = {v.casefold() for v in ctx.typed_values}
    if len(low) != 2:
        return None
    for t_tok, f_tok in BOOLEAN_PAIRS:
        if low != {t_tok, f_tok}:
            continue
        pair = (t_tok, f_tok)
        if pair in NUMERIC_BOOLEAN_PAIRS and ctx.hints.code_count:
            return None
        true_count = sum(c for v, c in ctx.typed_values.items() if v.casefold() == t_tok)
        total = sum(ctx.typed_values.values())
        tokens = sorted({v for v in ctx.typed_values})
        det = Detection(
            ColumnType.BOOLEAN,
            params={"pair": list(pair), "tokens": tokens, "true_count": true_count,
                    "true_rate": round(true_count / total, 6) if total else None},
            reason=f"two values forming the boolean pair {t_tok}/{f_tok}",
        )
        if pair in AMBIGUOUS_BOOLEAN_PAIRS:
            det.ambiguous = True
            det.ambiguity = f"{t_tok}/{f_tok} may be a code or count rather than a boolean"
        return det
    return None


def _ordinal(ctx: DetectContext) -> Detection | None:
    low = {v.casefold() for v in ctx.typed_values}
    if len(low) < 3:
        return None
    all_numeric = all(v.lstrip("-").isdigit() for v in low)
    for vocab in ORDINAL_VOCABS:
        if not low <= vocab.ranks.keys():
            continue
        if all_numeric and vocab.numeric_needs_hint and not ctx.hints.grade:
            continue
        levels = vocab.order(list(ctx.typed_values))
        return Detection(ColumnType.ORDINAL, subtype=vocab.name,
                         params={"vocabulary": vocab.name, "levels": levels, "order_source": "vocabulary"},
                         reason=f"{len(low)} levels from the {vocab.name} vocabulary (bypasses factor-max)")
    return None


def _school_year(ctx: DetectContext, forced: bool = False) -> Detection | None:
    vals = list(ctx.typed_values)
    starts: dict[str, int] = {}
    forms: set[str] = set()
    for v in vals:
        r = school_year_start(v)
        if r is None:
            break
        forms.add(r[0])
        starts[v] = r[1]
    else:
        levels = sorted(vals, key=lambda v: (_full_year(starts[v]), v))
        return Detection(ColumnType.SCHOOL_YEAR, params={"forms": sorted(forms), "levels": levels},
                         reason=f"school-year values ({', '.join(sorted(forms))}) with consecutive years")
    # K-21: ASCII digits only ('20²5' is a digit string to str.isdigit() but no number to int())
    if (ctx.hints.school_year or forced) and all(len(v) == 4 and v.isascii() and v.isdigit() and 1900 <= int(v) <= 2100
                                                 for v in vals):
        levels = sorted(vals, key=int)
        return Detection(ColumnType.SCHOOL_YEAR, subtype="ending_year",
                         params={"forms": ["ending_year_integer"], "levels": levels},
                         reason=f"ending-year integers with school-year header hint '{ctx.hints.school_year}'")
    return None


def _full_year(y: int) -> int:
    """Two-digit years use the %y pivot of design section 10 (69 and above = 19xx)."""
    if y >= 100:
        return y
    return 1900 + y if y >= 69 else 2000 + y


def _org_code(ctx: DetectContext, source: str, reason: str) -> Detection:
    det = _factor(ctx, source=source, reason=reason)
    det.type = ColumnType.ORG_CODE
    return det


def _factor(ctx: DetectContext, *, source: str, reason: str) -> Detection:
    variants: dict[str, set[str]] = {}
    for raw in ctx.raw_counts:
        if raw is None or raw.strip() == "":
            continue
        variants.setdefault(raw.strip().casefold(), set()).add(raw)
    groups = sum(1 for v in variants.values() if len(v) > 1)
    return Detection(ColumnType.FACTOR, params={"n_levels": len(variants), "variant_groups": groups},
                     source=source, reason=reason)

