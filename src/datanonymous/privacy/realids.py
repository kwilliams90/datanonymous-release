"""The run's real person IDs, which no value the run writes verbatim may equal (R9-01; D9; PRIVATE, in memory only).

D9 keeps every synthetic ID away from every real ID of the run. A real ID can also reach the output by a path that
writes real values verbatim: a column whose values only partly overlap the IDs of an ID column is no key reference
(``profile.structure.KEY_REF_SHARE``: relationship inference needs a majority), so its values stay factor levels,
discrete numbers, the exact min/max of D14, organisational codes, tokens or NA tokens. One fail-safe rule, independent
of any relationship inference, closes that class:

* :class:`RealIds` holds the D8 keys (``detect.ids.normalize_id``: case, punctuation, leading zeros) of every real ID
  of the run: every ID column of every file (a FAILed file's people are real people too) and every column a flag
  turned from an ID column into something else (``profile.entity.shadow_namespaces``: ``--exclude student_id``,
  ``--null``, ``--force-*``). Row-number namespaces kept as 1..n (D9) are no person IDs and are left out.
  Chance-awareness, the leak scan's and the D34 screen's (:func:`of_namespaces`): only keys of at least
  :data:`MIN_KEY` characters with a digit count (``privacy.leaks.MIN_ID_TOKEN``), so small numbers (scores, counts,
  grade levels) and code words (``MATH``, ``SPED``) are never person IDs. A letter-bearing ID (``S482913``,
  ``AB-1234``) and a zero-padded one as written (``0482913``) always count. A plain number (an ID of digits; the
  digits of a letter-prefixed ID are another value, as D8 compares IDs: ``482913`` is not ``S482913``) counts unless
  an ordinary number of its shape is more likely than not a real ID (:data:`DENSE_SHARE`; the D34 screen's chance,
  ``privacy.headers.IdKeys.number_chance``: the share of the numbers of its digit count that are real IDs, for 4
  digits at least the share of the 100 numbers around it, read at the number and half a window to either side so
  that the edge of a dense block counts as dense): in a dense, row-number-like range (sequential 4-digit
  staff numbers 1001-1350, record numbers 1..5000 with a few gaps) equality is chance, as for row numbers 1..n, so a
  scale score or a Lexile of that range is no person ID (and could not be replaced inside its range anyway). Any
  sparser namespace counts, whatever its size (2 roster IDs or 50,000 six-digit student numbers).
* A real value of a column equals a real ID when its D8 key is one (:meth:`RealIds.key_of_text`), and a number when
  it is a whole number whose digits are one (:meth:`RealIds.key_of_number`: ``482913``, ``482913.0`` and
  ``$482,913`` are one number, ``4829.13`` is none: digits are never glued across a decimal mark). :func:`id_units`
  lists them per column, one source of truth for the generators and the leak scan.
* No such value is written as it is. A unit written by exact quota gets a same-shape pseudonym with its exact count,
  consistent over the run (``generate.pseudonyms``; the D35 machinery): the D35 allowlist never keeps one
  (``detect.text.unit_kept``, a never-kept rule like a personal-data shape: levels, organisational codes, multi-select
  items, tokens among numbers), and ``profile.table.finish_real_ids`` pseudonymises every other one (levels of a
  discrete number column, ``--na-value`` tokens). A pseudonym of a number is a free whole number inside the column's
  real range, drawn where the column's values are (``generate.pseudonyms._number_in_range``; else a number of the
  column's format), and never a real ID itself (:func:`holds`, active while pseudonyms are drawn), so the exact
  min/max of D14 stay unless one of them is a real ID. A continuous number column, whose values are interpolated,
  redraws every generated value equal to one of
  its real numbers that is a real ID from its own distribution (:data:`NUMBERS_KEY`,
  ``generate.numeric.avoid_real_ids``: the exact min/max of D14, an interpolated value landing on one), so its shape
  is kept. A number column with one has its value statistics withheld from the report (``params['id_candidate']``):
  its real min, max, quantiles and pmf would print the ID.
* The leak scan never excuses one (``privacy.leaks``: :func:`kept_hits` on every written column, before writing and
  on the re-read files; a column holding one is never public in the report allowlist).

Not checked, by type (:func:`exempt`): ID columns (the ID registry, D9), the never-copy types (their generators
write no real value), dates and times, years and school years, booleans, states, ZIP codes and coordinates (their
values are what their type says, as the D34 screen never takes a year or date for an ID), and ``--force-factor`` /
``--pii-col`` columns (the user's explicit opt-in, D35).
"""

from __future__ import annotations

import contextlib
import functools
import re
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Iterator

import numpy as np

from ..detect.ids import normalize_id
from ..model import NEVER_COPY_TYPES, TYPE_INFO, ColumnProfile, ColumnType

T = ColumnType

#: Fewest characters of a real ID's D8 key that counts: the leak scanner's ID-token rule
#: (``privacy.leaks.MIN_ID_TOKEN``; a 1-3 digit number is a count, a score or a grade, never evidence of an ID).
MIN_KEY = 4
#: A plain number equal to a real ID is no evidence of one when an ordinary number of its shape is a real ID with at
#: least this chance (``privacy.headers.IdKeys.number_chance``): a dense, row-number-like range (module docstring).
DENSE_SHARE = 0.5
#: PRIVATE ``TablePrivate.extra`` key: the run's :class:`RealIds` (:func:`plan`).
RUN_KEY = "run_real_ids"
#: PRIVATE params key: which real extremes of a number column are pseudonymised real IDs (``[min?, max?]``): D14 cannot
#: emit them (the self-check does not warn for them, ``pipeline``).
EXTREMES_KEY = "_real_id_extremes"
#: PRIVATE params key: the real numbers (as the column holds them) of a continuous number column that are real person
#: IDs; the generator redraws every generated value equal to one from the column's own distribution
#: (``generate.numeric.avoid_real_ids``), so none is written and the distribution keeps its shape.
NUMBERS_KEY = "_real_id_numbers"
#: How the report explains a number column whose value statistics are withheld (``params['id_candidate']``).
CANDIDATE_REASON = "holds real person IDs of the run"
#: Types whose values are never compared (module docstring): what their type says they are.
EXEMPT_TYPES = frozenset({T.INTEGER_YEAR, T.SCHOOL_YEAR, T.BOOLEAN, T.STATE, T.ZIP, T.LAT, T.LON, T.EMPTY,
                          T.CITY, T.BINARY_NESTED})
#: Number types whose values are compared by value (:meth:`RealIds.key_of_number`).
NUMBER_TYPES = frozenset({T.INTEGER, T.FLOAT, T.DECIMAL, T.NUMERIC_FORMATTED, T.NUMERIC_WITH_TOKENS})
#: Sources of the user's explicit opt-in to real values (``detect.text.EXPLICIT_COPY_SOURCES``).
_EXPLICIT = frozenset({"force-factor", "pii-col"})
_PLAIN_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _usable(k: str) -> bool:
    return len(k) >= MIN_KEY and any(ch.isdigit() for ch in k)


@dataclass(frozen=True)
class RealIds:
    """The run's real person IDs (module docstring; :func:`of_namespaces`). PRIVATE.

    ``keys``: the D8 keys that count (letter-bearing keys, and the digit keys of plain numbers that are no chance
    match); ``padded``: the zero-padded real IDs as written (``0482913``), which always count as written."""

    keys: frozenset[str] = frozenset()
    padded: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.keys or self.padded)

    def key_of_text(self, text: Any) -> str | None:
        """The real ID ``text`` is (its D8 key), or None: a zero-padded real ID as written; a plain number written as
        text that is one (:meth:`key_of_number`: by value, so its digits are never glued across a decimal mark);
        else a value of at least :data:`MIN_KEY` characters with a digit whose D8 key is a real ID's."""
        if not self or not isinstance(text, str):
            return None
        s = text.strip()
        if len(s) < MIN_KEY or not any(ch.isdigit() for ch in s):
            return None
        if s in self.padded:
            return normalize_id(s)
        if _PLAIN_NUMBER.fullmatch(s):
            try:
                return self.key_of_number(Decimal(s))
            except InvalidOperation:
                return None
        k = normalize_id(s)
        return k if k in self.keys else None

    def key_of_number(self, value: Any) -> str | None:
        """The real ID the number ``value`` is (a whole number whose digits are a real ID's D8 key), or None."""
        if not self.keys or value is None or isinstance(value, bool):
            return None
        try:
            d = value if isinstance(value, Decimal) else Decimal(str(value)) if isinstance(value, str) \
                else Decimal(float(value))
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            return None
        if not d.is_finite() or d != d.to_integral_value():
            return None
        k = str(abs(int(d)))
        return k if len(k) >= MIN_KEY and k in self.keys else None

    def holds(self, text: Any) -> bool:
        """``text`` is a real ID of the run (:meth:`key_of_text`)."""
        return self.key_of_text(text) is not None

    @functools.cached_property
    def float_keys(self) -> np.ndarray:
        """The digit keys exact in float64, as floats (a column's numbers are pre-filtered with them)."""
        return np.asarray(sorted(float(k) for k in self.keys if k.isascii() and k.isdigit() and len(k) <= 15),
                          dtype=float)


def of_namespaces(spaces: Iterable[Any]) -> RealIds:
    """The :class:`RealIds` of ID namespaces (objects with ``raw_values``: ``generate.ids.IdNamespace``,
    ``profile.entity.shadow_namespaces``; row-number namespaces are left out by the caller), chance-aware as the D34
    screen (module docstring): a letter-bearing key and a zero-padded ID as written always count; a digit key (a plain
    ID) only while an ordinary number of its shape is a real ID with a chance below :data:`DENSE_SHARE`
    (``privacy.headers.IdKeys.number_chance``, computed from the same namespaces)."""
    from .headers import LOCAL_WINDOW, IdKeys

    spaces = [ns for ns in spaces if getattr(ns, "raw_values", None)]
    keys: set[str] = set()
    digits: set[str] = set()
    padded: set[str] = set()
    for ns in spaces:
        for v in ns.raw_values:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if len(s) < MIN_KEY:
                continue
            k = normalize_id(s)
            if not _usable(k):
                continue
            if k.isdigit():
                digits.add(k)
                if s[0] == "0" and s.isdigit():
                    padded.add(s)
                continue
            keys.add(k)
    if digits:
        chance = IdKeys(spaces)
        by_len: dict[int, list[str]] = {}
        for k in digits:
            by_len.setdefault(len(k), []).append(k)
        for n_digits, ks in by_len.items():
            w = LOCAL_WINDOW.get(n_digits)
            if w:  # read locally: each number its own chance (the densest window around it, so a dense block's edge
                # is no sparser than its middle)
                keys.update(k for k in ks if max(
                    chance.number_chance(str(int(k) + off)) for off in (-(w // 2), 0, w // 2)
                    if len(str(int(k) + off)) == n_digits) < DENSE_SHARE)
            elif chance.number_chance(ks[0]) < DENSE_SHARE:  # one share for every number of that digit count
                keys.update(ks)
    return RealIds(frozenset(keys), frozenset(padded))


def of_files(files: Iterable[Any]) -> RealIds:
    """The :class:`RealIds` of ``files`` (``FileResult``s): every ID column of every file, FAILed ones too, and every
    shadow ID column (``profile.entity.shadow_namespaces``), grouped by registry key as the run's ID registry groups
    them (``profile.entity.id_namespace``), row-number namespaces (1..n, D9) left out (:func:`of_namespaces`)."""
    from ..generate.ids import IdRegistry, namespace_is_identity
    from ..profile import entity as entity_mod

    files = list(files)
    scratch = IdRegistry()
    for fr in files:
        for tp in fr.tables:
            for i in tp.entity.id_columns:
                raw = tp.private.id_raw.get(i)
                if raw is None:
                    continue
                key = entity_mod.id_namespace(fr.rel_path, tp.name, tp.columns[i])
                scratch.register(key, fr.rel_path, tp.name, i, tp.columns[i].name, raw)
    spaces: list[Any] = [ns for ns in scratch.namespaces.values() if ns.raw_values and not namespace_is_identity(ns)]
    spaces += list(entity_mod.shadow_namespaces(files))
    return of_namespaces(spaces)


def of_tables(tables: Iterable[Any], file_rel: str = "") -> RealIds:
    """:func:`of_files` for tables profiled on their own (one file)."""
    from types import SimpleNamespace

    return of_files([SimpleNamespace(rel_path=file_rel, tables=list(tables), failed=False)])


def plan(files: Iterable[Any]) -> RealIds:
    """Run-level planning (pipeline, after the key discovery and before the run's D35 decision): the run's
    :class:`RealIds`, handed to every table (``tp.private.extra[RUN_KEY]``)."""
    files = list(files)
    ids = of_files(files)
    for fr in files:
        for tp in fr.tables:
            tp.private.extra[RUN_KEY] = ids
    return ids


def run_ids(tp: Any) -> RealIds | None:
    """The run's :class:`RealIds` handed to ``tp`` in planning (None for a table profiled on its own)."""
    extra = getattr(getattr(tp, "private", None), "extra", None) or {}
    return extra.get(RUN_KEY)


_ACTIVE: ContextVar[RealIds | None] = ContextVar("datanonymous_real_ids", default=None)


@contextlib.contextmanager
def guard(ids: RealIds | None) -> Iterator[None]:
    """Make ``ids`` the real IDs no pseudonym may equal (:func:`holds`) inside the block."""
    token = _ACTIVE.set(ids if ids else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def holds(text: Any) -> bool:
    """True when ``text`` is a real ID of the run (:func:`guard`); False without a guard."""
    ids = _ACTIVE.get()
    return ids is not None and ids.holds(text)


# --------------------------------------------------------------------------- columns
def exempt(p: ColumnProfile) -> bool:
    """Column ``p`` is never compared with the run's real IDs (module docstring): it writes no real value verbatim,
    is an ID column, holds values of a type that says what they are, or is the user's explicit opt-in."""
    if getattr(p, "role", "row") in ("excluded", "null", "id") or p.type in NEVER_COPY_TYPES:
        return True
    if p.type in EXEMPT_TYPES or TYPE_INFO[p.type].family == "date" or getattr(p, "source", None) in _EXPLICIT:
        return True
    return p.type is T.DIGIT_CODE and not (p.params or {}).get("levels_mode")  # same-shape masks


def number_of(p: ColumnProfile, text: Any) -> Decimal | None:
    """The exact number a value of number column ``p`` writes (its format read: ``$1,234.50``), or None (a token)."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    fmt = (p.params or {}).get("format") if p.type in (T.NUMERIC_FORMATTED, T.NUMERIC_WITH_TOKENS) else None
    if fmt:
        from ..detect.numeric import number_text

        affix = fmt.get("affix") or (None, None)
        plain = number_text(s, fmt.get("decimal_mark", "."), (affix[0], affix[1]))
        if plain is None:
            return None
        s = plain
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def unit_texts(p: ColumnProfile, raw: Any) -> list[str]:
    """The texts one real value of ``p`` writes verbatim: its multi-select items, else the value (stripped)."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    s = raw.strip()
    d = (p.params or {}).get("delimiter") if p.type is T.MULTI_SELECT else None
    if d and s not in (p.na_tokens or ()):
        return list(dict.fromkeys(x.strip() for x in s.split(d) if x.strip()))
    return [s]


def _key_of(ids: RealIds, p: ColumnProfile, text: str) -> str | None:
    if p.type in NUMBER_TYPES:
        d = number_of(p, text)
        if d is not None:
            return ids.key_of_number(d)
    return ids.key_of_text(text)


#: PRIVATE ``ColumnModel.extra`` key: (the :class:`RealIds` it was computed for, :func:`id_units`).
_UNITS_KEY = "_real_id_units"
#: Below this magnitude a whole number is exact in float64: the numeric view pre-filters a column's numbers.
_FLOAT_EXACT = 1e15


def _number_candidates(p: ColumnProfile, ids: RealIds) -> set[float] | None:
    """The distinct real numbers of number column ``p`` that can be real IDs, read from its numeric view (one
    vectorised pass: a million interpolation-modelled values are not parsed one by one); None when the column has
    no numeric view (discrete levels: few values, read one by one) or numbers too wide for float64."""
    xs = p.model.sorted_numeric if p.model is not None else None
    if xs is None or not len(xs):
        return None
    if max(abs(float(xs[0])), abs(float(xs[-1]))) >= _FLOAT_EXACT:
        return None
    keys = ids.float_keys
    if not len(keys):
        return set()
    xs = np.asarray(xs, dtype=float)  # sorted: each key is looked up, the values are not sorted again
    out: set[float] = set()
    for sign in (1.0, -1.0):
        k = keys * sign
        pos = np.searchsorted(xs, k)
        ok = pos < len(xs)
        out.update(np.abs(k[ok][xs[pos[ok]] == k[ok]]).tolist())
    return out


def id_units(p: ColumnProfile, ids: RealIds | None) -> dict[str, str]:
    """Every real unit column ``p`` would write verbatim that is a real person ID of the run: stripped unit (a value,
    a multi-select item, a token, an NA token, as the column holds it) -> the ID's D8 key; numbers by value. Empty for
    an :func:`exempt` column. One source of truth for the generators (``profile.table.finish_real_ids``) and the leak
    scan (:func:`kept_hits`); computed once per column and run (cached in the PRIVATE model)."""
    if not ids or p.model is None or exempt(p):
        return {}
    cached = p.model.extra.get(_UNITS_KEY)
    if cached is not None and cached[0] is ids:
        return cached[1]
    cands = _number_candidates(p, ids) if p.type in NUMBER_TYPES else None
    tokens = frozenset(str(t).strip() for t in ((p.params or {}).get("tokens") or (p.params or {}).get("_tokens")
                                                 or ()))
    plain = p.type in (T.INTEGER, T.FLOAT, T.DECIMAL)
    out: dict[str, str] = {}
    for counts, missing in ((p.model.nonmissing_counts, False), (p.model.missing_counts, True)):
        for raw in counts:
            if cands is not None and not missing and not (isinstance(raw, str) and raw.strip() in tokens):
                if not cands:
                    continue  # a number, and no number of the column is a real ID
                if plain:
                    try:
                        if abs(float(raw)) not in cands:
                            continue  # (the exact reading below decides the candidates)
                    except (TypeError, ValueError):
                        pass
            for t in unit_texts(p, raw):
                if t in out:
                    continue
                k = _key_of(ids, p, t)
                if k is not None:
                    out[t] = k
    p.model.extra[_UNITS_KEY] = (ids, out)
    return out


def kept_hits(p: ColumnProfile, values: Iterable[Any], ids: RealIds | None) -> set[str]:
    """The written values of column ``p`` (stripped) that are a real value of the column that is a real person ID of
    the run (:func:`id_units`; numbers by value, multi-select items one by one): a real ID copied (R9-01). An
    interpolated number equal to a real ID that is no real value of the column is not a copy."""
    units = id_units(p, ids)
    if not units:
        return set()
    keys = set(units.values())
    distinct = list(dict.fromkeys(v for v in values if isinstance(v, str)))
    if p.type in NUMBER_TYPES and len(distinct) > 64:
        # numbers: one vectorised reading (``profile.stats.numeric_view_aligned``) picks the few values worth an exact
        # look (those equal to a real-ID number of the column, and tokens)
        from ..profile.stats import numeric_view_aligned

        nums = numeric_view_aligned(p, [v.strip() for v in distinct])
        if nums is not None:
            wanted = np.asarray(sorted({abs(float(d)) for d in (number_of(p, t) for t in units) if d is not None}),
                                dtype=float)
            keep = np.isnan(nums) | np.isin(np.abs(nums), wanted)
            distinct = [v for v, k in zip(distinct, keep.tolist()) if k]
    out: set[str] = set()
    for v in distinct:
        for t in unit_texts(p, v):
            k = units.get(t)
            if k is None:
                k = _key_of(ids, p, t) if ids is not None else None
            if k is not None and k in keys:
                out.add(v.strip())
                break
    return out


def continuous(p: ColumnProfile) -> bool:
    """Number column ``p`` is generated by interpolation (not by exact quota over its real values): its real-ID
    numbers are redrawn (:data:`NUMBERS_KEY`), not given pseudonyms."""
    return p.type in NUMBER_TYPES and not (p.type is T.INTEGER and p.subtype == "discrete")


def forbidden_numbers(p: ColumnProfile) -> frozenset[Decimal]:
    """The real numbers of continuous number column ``p`` that are real person IDs (:data:`NUMBERS_KEY`), with either
    sign (an ID is read without its sign: ``-482913`` is ``482913``, :meth:`RealIds.key_of_number`)."""
    texts = (p.params or {}).get(NUMBERS_KEY)
    if not texts:
        return frozenset()
    nums = numbers_of(p, texts)
    return frozenset(nums | {-d for d in nums})


def numbers_of(p: ColumnProfile, texts: Iterable[str]) -> set[Decimal]:
    """The numbers of the ``texts`` of number column ``p`` (:func:`number_of`; tokens left out)."""
    out: set[Decimal] = set()
    for t in texts:
        d = number_of(p, t)
        if d is not None:
            out.add(d)
    return out


def extreme_flags(p: ColumnProfile, numbers: set[Decimal]) -> list[bool]:
    """``[min, max]``: whether the real min / max of number column ``p`` is one of ``numbers`` (pseudonymised real
    IDs: D14 cannot emit it)."""
    if not numbers or p.model is None:
        return [False, False]
    xs = p.model.sorted_numeric
    if xs is not None and len(xs) and max(abs(float(xs[0])), abs(float(xs[-1]))) < _FLOAT_EXACT:
        nums = {float(d) for d in numbers}
        return [float(xs[0]) in nums, float(xs[-1]) in nums]
    real = [d for d in (number_of(p, raw) for raw in p.model.nonmissing_counts) if d is not None]
    if not real:
        return [False, False]
    return [min(real) in numbers, max(real) in numbers]


__all__ = ["CANDIDATE_REASON", "DENSE_SHARE", "EXEMPT_TYPES", "EXTREMES_KEY", "MIN_KEY", "NUMBERS_KEY", "NUMBER_TYPES",
           "RUN_KEY", "RealIds", "continuous", "exempt", "extreme_flags", "forbidden_numbers", "guard", "holds",
           "id_units", "kept_hits", "number_of", "numbers_of", "of_files", "of_namespaces", "of_tables", "plan",
           "run_ids", "unit_texts"]
