"""Same-shape pseudonyms for the real text units the copy gate did not keep (design D35).

A column whose real text units (levels, multi-select items, tokens among numbers) were not all kept by the D35
allowlist (``detect.text.copy_gate_units``: PRIVATE ``params['_pseudonymized']``) keeps its type and every kept
unit. Every pseudonymised unit gets ONE synthetic replacement (:func:`pseudonym`): letter runs become lorem words
of the same length with the same per-letter case, digit runs random digits (a leading zero only where the unit
has one), every other character (spaces, punctuation, separators) stays; an e-mail address, URL, phone number,
SSN or street address inside the unit follows its PII generator's rules (``generate.embedded``: reserved
domains, fresh phone bases, area 900-999 ...). So the length, word pattern, case, punctuation and digit pattern are
kept, and

* one real unit gets the same pseudonym in every column and file of the run (:func:`plan` draws ONE map for the
  run, ``RUN_MAP_KEY``; :func:`build_maps` falls back to one map per table when a table is generated on its
  own), so equalities between columns and joins between files hold for pseudonymised values as for kept ones;
  case forms of one value get the same pseudonym in the same case forms (``Rose Garden`` / ``ROSE GARDEN``), and
  surrounding whitespace is kept; counts are exact and the D14 floor holds: a rare level appears with its real
  count, under its pseudonym;
* a pseudonym equals no real value of the run (case-insensitive; :func:`plan` collects the run's real texts of
  the pseudonyms' lengths, the values of ``--exclude``d and ``--null``ed columns included, R7-02, and draws the map
  with the run's unwritten real texts active, ``privacy.texts``) and no other pseudonym, holds no real phone number
  or SSN of the run
  (``generate.text.Drawer``), and is no vocabulary word (``detect.vocab``: a code such as ``ID`` or ``EB`` would
  mislead code tested on the synthetic data) unless no other text of the shape exists; a pseudonymised token among
  numbers never reads as a number;
* the map is deterministic for a given ``--seed``: it is drawn from a seeded stream over the run's units in sorted
  order, never from a hash of the real value (a hash without a secret could be recomputed from a guessed name,
  and would not depend on the seed).

R9-01: a real unit that is a real person ID of the run (``privacy.realids``) is pseudonymised whatever its count, and
so is a level of a discrete number column that is one (``profile.table.finish_real_ids``): its pseudonym is a free
whole number inside the column's real range, drawn where the column's values are (:func:`_number_in_range`), else a
number of the column's format (new digits before a kept zero fraction). No pseudonym is a real person ID of the run
(``privacy.realids.guard`` is active while the map is drawn).

The map (real raw unit -> synthetic raw unit; a multi-select's items inside each value) is stored in the row
model's PRIVATE ``extra`` (``detect.text.PSEUDO_MAP_KEY``) and applied by every path that writes the column's
units: exact-quota generation (row and per-entity models, ``generate.table``), per-entity row balancing, numbers
with tokens (``generate.numeric``), functional-dependency rebuilds (``generate.structure``: a dependent level
written for a determinant is the dependent's own written value), and the writers' native label tables (Arrow
dictionaries, Stata value labels). The map never leaves memory.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
import re
from dataclasses import replace
from decimal import Decimal
from typing import AbstractSet, Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..detect.names import fold_word
from ..detect.org_names import FIRST, INITIAL
from ..detect.text import (COMBO_KEEP_KEY, ITEMS, NA_MAP_KEY, ORG_NAMES_KEY, PSEUDO_MAP_KEY, TOKENS, bare_phone,
                           d35_units, na_pseudonymized, pseudonymized, unit_key)
from ..detect.vocab import is_vocabulary
from ..model import ColumnModel, ColumnProfile, ColumnType
from ..privacy import realids
from ..privacy import texts as real_texts
from . import embedded
from . import text as textgen
from .rng import stream

#: PRIVATE ``TablePrivate.extra`` key: the run's real texts (stripped, case-folded) of the lengths of its
#: pseudonymised units, which no pseudonym may equal (:func:`plan`).
RUN_TEXTS_KEY = "run_real_texts"
#: PRIVATE ``TablePrivate.extra`` key: the run's one map, stripped real unit -> stripped pseudonym (:func:`plan`).
RUN_MAP_KEY = "run_pseudonyms"


def level_map(p: ColumnProfile) -> dict[Any, Any]:
    """The column's real -> pseudonym map (empty when the column has no pseudonymised unit or none was built)."""
    if p.model is None:
        return {}
    return p.model.extra.get(PSEUDO_MAP_KEY) or {}


# --------------------------------------------------------------------------- run-level planning
def _texts_of(p: ColumnProfile, lengths: set[int], out: set[str]) -> None:
    """The real texts of column ``p`` (whole values and multi-select items) whose stripped length is in
    ``lengths``, case-folded into ``out``."""
    if p.model is None:
        return
    d = (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None
    for counts in (p.model.nonmissing_counts, p.model.missing_counts):
        for raw in counts:
            if not isinstance(raw, str):
                continue
            s = raw.strip()
            if len(s) in lengths:
                out.add(s.casefold())
            if d and d in s:
                out.update(x.strip().casefold() for x in s.split(d) if len(x.strip()) in lengths)


def _pseudo_texts(p: ColumnProfile) -> list[str]:
    """The real texts of column ``p`` that get pseudonyms: its pseudonymised D35 units and --na-value tokens (R8-05)."""
    return [*pseudonymized(p), *na_pseudonymized(p)]


def _lengths(tables: Iterable[Any]) -> set[int]:
    """The lengths of the run's pseudonymised texts, and of the items of a pseudonymised whole combination (K-11:
    each item of its pseudonym avoids the real texts of its own length)."""
    out: set[int] = set()
    for tp in tables:
        for p in tp.columns:
            d = _delimiter(p)
            for t in _pseudo_texts(p):
                out.add(len(t))
                if d and d in t:
                    out.update(len(x.strip()) for x in t.split(d))
    return out


def _delimiter(p: ColumnProfile) -> str | None:
    """The delimiter of a multi-select whose items are its D35 units, else None."""
    return (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None


def _whole(raw: Any, d: str | None, mapping: dict[Any, Any]) -> bool:
    """K-11: ``raw`` is a whole multi-select combination the map rewrites as a whole (a pseudonymised combination of
    kept items, ``detect.text.combination_gate``): an item never holds the delimiter."""
    return bool(d) and isinstance(raw, str) and d in raw and raw in mapping


def _mapped_columns(tp: Any) -> list[ColumnProfile]:
    return [p for p in tp.columns if p.role not in ("excluded", "null") and p.model is not None and _pseudo_texts(p)]


def plan(files: Iterable[Any], seed: int | None = None) -> None:
    """Run-level planning (pipeline, after pass 1 and the run's D35 decision): the run's real texts that no
    pseudonym may equal and, with ``seed``, the run's ONE pseudonym map.

    The texts: every real value (and multi-select item) of every column of every file of the run, FAILed files
    included, whose length is the length of a pseudonymised unit (a pseudonym keeps its unit's length),
    case-folded. Every table gets the same set (``tp.private.extra[RUN_TEXTS_KEY]``) and the same map
    (``RUN_MAP_KEY``: stripped real unit -> stripped pseudonym, drawn from the run's ``__pseudonyms__`` stream).
    """
    tables = [tp for fr in files for tp in fr.tables]
    lengths = _lengths(tables)
    if not lengths:
        return
    texts: set[str] = set()
    for tp in tables:
        for p in tp.columns:
            _texts_of(p, lengths, texts)
        # the values of an excluded or nulled column are real values of the run too (R7-02)
        for vals in (tp.private.extra.get(real_texts.DROPPED_KEY) or {}).values():
            texts.update(v.casefold() for v in vals if len(v) in lengths)
    shared = frozenset(texts)
    for tp in tables:
        tp.private.extra[RUN_TEXTS_KEY] = shared
    if seed is None:
        return
    # and no pseudonym equals a real text of the run that it never writes, in any spelling (privacy.texts), nor a real
    # person ID of the run (privacy.realids, R9-01)
    registry = next((t for tp in tables if (t := tp.private.extra.get(real_texts.RUN_KEY)) is not None), None)
    ids = next((i for tp in tables if (i := realids.run_ids(tp)) is not None), None)
    cols = [p for tp in tables for p in _mapped_columns(tp)]
    # D36: the fake names inside organisational codes avoid every word of every real text of the run
    words = real_words(tables, registry) if any((p.params or {}).get(ORG_NAMES_KEY) for p in cols) else None
    with real_texts.guard(registry), realids.guard(ids):
        mapping = draw_map(cols, stream(seed, "__pseudonyms__"), shared, words)
    for tp in tables:
        tp.private.extra[RUN_MAP_KEY] = mapping


# --------------------------------------------------------------------------- building the maps
def build_maps(tp: Any, rng: np.random.Generator | Callable[[Any], np.random.Generator]) -> None:
    """Store the map of every column of ``tp`` with pseudonymised units (before generation): its raw units ->
    their pseudonyms from the run's map (:func:`plan`), or, for a table generated on its own, from one map drawn
    for the table from ``rng`` (a Generator, or a callable giving one)."""
    cols = _mapped_columns(tp)
    if not cols:
        return
    shared = tp.private.extra.get(RUN_MAP_KEY)
    if shared is None or any(t not in shared for p in cols for t in _pseudo_texts(p)):
        avoid = tp.private.extra.get(RUN_TEXTS_KEY)
        if avoid is None:  # no run-level planning (a table generated on its own): the table's own real texts
            texts: set[str] = set()
            lengths = _lengths([tp])
            for p in tp.columns:
                _texts_of(p, lengths, texts)
            avoid = frozenset(texts)
        gen = rng if isinstance(rng, np.random.Generator) else rng(None)
        words = real_words([tp]) if any((p.params or {}).get(ORG_NAMES_KEY) for p in cols) else None
        shared = draw_map(cols, gen, avoid, words)
    for p in cols:
        p.model.extra[PSEUDO_MAP_KEY] = column_map(p, shared)
        na = {t: shared[t] for t in na_pseudonymized(p) if t in shared}
        if na:  # R8-05: the pseudonym of an --na-value token is written in its place and read as missing too
            p.model.extra[NA_MAP_KEY] = na
            p.na_tokens = frozenset(p.na_tokens or ()) | frozenset(na.values())


def _units(p: ColumnProfile) -> dict[str, int]:
    """Raw units the map rewrites (as written: a multi-select item with the spacing around it) -> real rows."""
    pseudo = pseudonymized(p)
    kind = d35_units(p)
    d = (p.params or {}).get("delimiter") if kind == ITEMS else None
    na = p.na_tokens or frozenset()
    out: dict[str, int] = {}
    for raw, c in p.model.nonmissing_counts.items():
        if not isinstance(raw, str):
            continue
        if kind == ITEMS:
            if raw.strip() in na:
                continue
            # K-11: a pseudonymised whole combination is rewritten as a whole (its items are kept ones)
            parts = [raw] if d and d in raw and raw.strip() in pseudo else raw.split(d) if d else [raw]
        else:
            parts = [raw]
        for part in parts:
            if part.strip() in pseudo:
                out[part] = out.get(part, 0) + int(c)
    return out


def _na_units(p: ColumnProfile) -> dict[str, int]:
    """R8-05: raw values of column ``p`` that are one of its pseudonymised --na-value tokens -> rows."""
    na = na_pseudonymized(p)
    out: dict[str, int] = {}
    if not na:
        return out
    for counts in (p.model.nonmissing_counts, p.model.missing_counts):
        for raw, c in counts.items():
            if isinstance(raw, str) and raw.strip() in na:
                out[raw] = out.get(raw, 0) + int(c)
    return out


def apply_na_map(p: ColumnProfile, values: np.ndarray) -> np.ndarray:
    """R8-05: ``values`` (a generated column) with every pseudonymised --na-value token replaced by its pseudonym, the
    whitespace around it kept: the one place every path that writes an NA token (missing slots, per-ID values,
    structural rebuilds, a state or multi-select column's kept values) goes through."""
    mapping = (p.model.extra.get(NA_MAP_KEY) if p.model is not None else None) or {}
    if not mapping or not len(values):
        return values
    codes, uniq = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    new = []
    changed = False
    for u in uniq.tolist():
        fake = mapping.get(u.strip()) if isinstance(u, str) else None
        if fake is None:
            new.append(u)
            continue
        lead, trail = u[: len(u) - len(u.lstrip())], u[len(u.rstrip()):]
        new.append(lead + fake + trail)
        changed = True
    if not changed:
        return values
    out = np.empty(len(new), dtype=object)
    out[:] = new
    return out[np.asarray(codes, dtype=np.int64)]


def _real_units(p: ColumnProfile) -> set[str]:
    """Every real value of the column and, for a multi-select, every real item (stripped, case-folded)."""
    out = textgen.real_casefold(p, p.model)
    d = (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None
    if d:
        out |= {x.strip().casefold() for raw in p.model.nonmissing_counts if isinstance(raw, str)
                for x in raw.split(d) if x.strip()}
    return out


class _Drawer(textgen.Drawer):
    """:class:`generate.text.Drawer` with an extra acceptance test (a token among numbers stays a token), set per
    unit, and a first pass that also refuses vocabulary words (``detect.vocab.is_vocabulary``: a pseudonym ``ID``
    or ``EB`` would read as a code)."""

    def __init__(self, real_cf: set[str]) -> None:
        super().__init__(real_cf)
        self.accept: Callable[[str], bool] | None = None

    def ok(self, candidate: str) -> bool:
        # R9-01: a pseudonym is never a real person ID of the run (privacy.realids, active while the map is drawn)
        return (super().ok(candidate) and (self.accept is None or self.accept(candidate))
                and not realids.holds(candidate))

    def draw(self, make: Callable[[int], str], pyr: Any, attempts: int = textgen.MAX_ATTEMPTS) -> str:
        cand = ""
        for a in range(attempts):  # lorem words of the shape, then same-shape masks: no vocabulary word
            cand = embedded.guard(make(a), pyr)
            if self.ok(cand) and not is_vocabulary(cand):
                return self.take(cand)
        base = cand or "x"
        for _ in range(attempts):
            cand = embedded.guard(self.fallback(base, pyr), pyr)
            if self.ok(cand) and not is_vocabulary(cand):
                return self.take(cand)
        return super().draw(make, pyr, attempts)  # (a shape whose every text is a vocabulary word)


def _not_a_number(p: ColumnProfile) -> Callable[[str], bool]:
    """A pseudonym of a token among numbers must still be a token: it must not read as a number in the
    column's format."""
    from ..detect.numeric import parse_with_format

    fmt = (p.params or {}).get("format") or {}

    def accept(candidate: str) -> bool:
        try:
            return math.isnan(parse_with_format(candidate.strip(), fmt))
        except (TypeError, ValueError, KeyError):
            return True
    return accept


def _bounds(p: ColumnProfile) -> tuple[Decimal, Decimal] | None:
    """The real min and max of number column ``p`` (None without a number)."""
    xs = p.model.sorted_numeric if p.model is not None else None
    if xs is not None and len(xs):
        return Decimal(float(xs[0])), Decimal(float(xs[-1]))
    nums = realids.numbers_of(p, (raw for raw in (p.model.nonmissing_counts if p.model is not None else ())
                                  if isinstance(raw, str)))
    return (min(nums), max(nums)) if nums else None


#: Draws of one unit after which the pseudonym of a number may leave the column's real range (still a number of the
#: unit's shape: the drawer's same-shape masks come next, before any longer value), and after which any text is
#: accepted (a shape with no number left, never reached in practice: the drawer must always end).
_RANGE_DRAWS = textgen.MAX_ATTEMPTS
_NUMBER_DRAWS = 12 * textgen.MAX_ATTEMPTS


def _a_number(p: ColumnProfile, bounds: tuple[Decimal, Decimal] | None) -> Callable[[str], bool]:
    """R9-01: the pseudonym of a number of number column ``p`` (a real person ID) must read as a number in the
    column's format, inside the column's real range while one fits (the first draws: the D14 extremes stay put), so a
    level of a discrete column or an interpolated value keeps its kind. One test per unit (it counts its draws)."""
    calls = [0]

    def accept(candidate: str) -> bool:
        calls[0] += 1
        if calls[0] > _NUMBER_DRAWS:
            return True
        d = realids.number_of(p, candidate)
        if d is None:
            return False
        return bounds is None or calls[0] > _RANGE_DRAWS or bounds[0] <= d <= bounds[1]
    return accept


#: Draws of a whole number inside the column's range (R9-01) before its free numbers are walked.
_RANGE_TRIES = textgen.MAX_ATTEMPTS
#: Steps to either side of a drawn real value of the column within which a free number is looked for.
_NEAR_STEPS = 25
#: Most whole numbers walked from a random start to find a free one inside the range.
_RANGE_WALK = 200_000
_WHOLE = re.compile(r"[+-]?\d+")


def _real_spread(p: ColumnProfile) -> tuple[list[float], list[int]] | None:
    """The real numbers of number column ``p`` and their cumulative row counts (None without a number)."""
    if p.model is None:
        return None
    by_value: dict[float, int] = {}
    for raw, c in p.model.nonmissing_counts.items():
        d = realids.number_of(p, raw) if isinstance(raw, str) else None
        if d is not None:
            by_value[float(d)] = by_value.get(float(d), 0) + int(c)
    if not by_value:
        return None
    vals = sorted(by_value)
    return vals, list(itertools.accumulate(by_value[v] for v in vals))


def _number_in_range(rep: str, bounds: tuple[Decimal, Decimal], mark: str, pyr: random.Random,
                     drawer: "_Drawer", spread: tuple[list[float], list[int]] | None = None) -> str | None:
    """R9-01: the pseudonym of whole number ``rep`` of a number column (a real person ID): a free whole number inside
    the column's real range ``bounds`` (with ``rep``'s count of digits while one is free there, else any), written like
    ``rep`` (its sign, zero padding and zero fraction after ``mark``), that ``drawer`` accepts and that no
    personal-data guard would change; None for a formatted number (``$482,913``: the same-shape draws) or a range with
    no free number. It is drawn where the column's values are (the free number nearest a real value of the column
    drawn at random by its rows, ``spread``: :func:`_real_spread`), then uniformly, then by a walk from a random start:
    never from the real ID itself (the seed decides), and inside the range, so the column keeps its exact min/max and
    the level its place in the distribution far better than new digits of the same shape would (most fall outside the
    range)."""
    i = rep.rfind(mark)
    frac = rep[i:] if i > 0 and rep[i + 1:i + 2].isdigit() else ""
    whole = rep[:len(rep) - len(frac)]
    if not _WHOLE.fullmatch(whole):
        return None
    sign = whole[0] if whole[0] in "+-" else ""
    digits = whole[len(sign):]
    width = len(digits) if digits.startswith("0") and len(digits) > 1 else 0
    lo, hi = math.ceil(bounds[0]), math.floor(bounds[1])

    def render(x: int) -> str:
        return ("-" if x < 0 else "+" if sign == "+" else "") + str(abs(x)).zfill(width) + frac

    def free(x: int) -> str | None:
        cand = render(x)
        if drawer.ok(cand) and embedded.guard(cand, pyr) is cand:
            return drawer.take(cand)
        return None

    n = len(digits.lstrip("0") or "0")
    same = (max(lo, 10 ** (n - 1)), min(hi, 10 ** n - 1)) if sign != "-" else (max(lo, 1 - 10 ** n),
                                                                               min(hi, -(10 ** (n - 1))))
    for a, b in (same, (lo, hi)):
        if a > b:
            continue
        if spread is not None:
            vals, cum = spread
            for _ in range(_RANGE_TRIES):
                c = int(round(vals[bisect.bisect_right(cum, pyr.random() * cum[-1])]))
                for k in range(_NEAR_STEPS + 1):
                    got = next((g for x in ((c,) if k == 0 else (c - k, c + k)) if a <= x <= b
                                and (g := free(x)) is not None), None)
                    if got is not None:
                        return got
        for _ in range(_RANGE_TRIES):
            got = free(pyr.randint(a, b))
            if got is not None:
                return got
        start = pyr.randint(a, b)
        for k in range(min(b - a + 1, _RANGE_WALK)):  # walk on from a random start, wrapping round
            got = free(a + (start - a + k) % (b - a + 1))
            if got is not None:
                return got
    return None


def _case_like(base: str, variant: str) -> str | None:
    """``base`` (the pseudonym of one case form) in the per-letter case of ``variant`` (another case form of the same
    real value), or None when the two do not line up character by character."""
    if len(base) != len(variant):
        return None
    out = "".join(_cased(b, v) for b, v in zip(base, variant))
    return out if out.casefold() == base.casefold() else None


def draw_map(cols: list[ColumnProfile], rng: np.random.Generator, avoid: frozenset[str],
             words: AbstractSet[str] | None = None) -> dict[str, str]:
    """ONE map, stripped real unit -> stripped pseudonym, for every pseudonymised unit of ``cols``.

    Units are grouped by their case-folded text and drawn in sorted order from ``rng``: a group's most common form
    gets a fresh pseudonym (:func:`pseudonym`; the phone rules when any column holding it has a person/contact/notes
    header or it is a bare North American phone number, ``detect.text.bare_phone``; a token of a numbers column must
    not read as a number there), every other case form the same
    pseudonym in its own case. Nothing drawn equals a real text of ``avoid`` or of ``cols``, or another group's
    pseudonym.

    D36: an organisational code holding a person's name (``detect.text.ORG_NAMES_KEY``) keeps its text but the name
    pieces, which get the run's fake names (:class:`OrgNames`: one fake word per real name word, never a word of
    ``words``, the words of every real text of the run, :func:`real_words`; dotted initials another letter), so the
    same real value gets the same synthetic value everywhere and the course part stays real; a value whose name
    cannot be located, or whose rewrite the checks above refuse, gets a whole same-shape pseudonym instead.
    """
    rows: dict[str, int] = {}
    phone: set[str] = set()
    accept: dict[str, list[Callable[[str], bool]]] = {}
    marks: dict[str, str] = {}  # R9-01: a number's key -> the column's decimal mark (its fraction is kept)
    ranges: dict[str, tuple[Decimal, Decimal]] = {}  # R9-01: a number's key -> the real range of its column(s)
    spread: dict[str, Any] = {}  # R9-01: a number's key -> the real numbers of (one of) its column(s) (_real_spread)
    real: set[str] = set(avoid)
    combos: dict[str, str] = {}  # K-11: a whole combination's key -> its delimiter
    # K-11: a whole combination's key -> (free words, real items) written as they are, per column holding it
    keeps: dict[str, list[tuple[frozenset[str], tuple[str, ...]]]] = {}
    for p in cols:
        real |= _real_units(p)
        dp = bool((p.params or {}).get("_digit_phone"))
        token = d35_units(p) == TOKENS
        d = _delimiter(p)
        if d:
            keep = (p.params or {}).get(COMBO_KEEP_KEY) or {}
            for t in _units(p):
                if d in t:
                    key = t.strip().casefold()
                    combos[key] = d
                    free, kept = keep.get(unit_key(t)) or ([], [])
                    keeps.setdefault(key, []).append((frozenset(free), tuple(kept)))
        bounds = _bounds(p) if p.type in realids.NUMBER_TYPES else None
        for raw, c in _units(p).items():
            t = raw.strip()
            rows[t] = rows.get(t, 0) + c
            if dp:
                phone.add(t)
            if bounds is not None and realids.number_of(p, t) is not None:
                # R9-01: a number of a number column (a real person ID) gets a number of the column's format
                accept.setdefault(t.casefold(), []).append(_a_number(p, bounds))
                marks[t.casefold()] = str(((p.params or {}).get("format") or {}).get("decimal_mark", "."))
                had = ranges.get(t.casefold())
                both = (max(had[0], bounds[0]), min(had[1], bounds[1])) if had is not None else bounds
                ranges[t.casefold()] = both if both[0] <= both[1] else (had or bounds)
                if t.casefold() not in spread:
                    spread[t.casefold()] = _real_spread(p)
            elif token:
                accept.setdefault(t.casefold(), []).append(_not_a_number(p))
        for raw, c in _na_units(p).items():  # R8-05: the run's pseudonymised --na-value tokens, from the same map
            t = raw.strip()
            rows[t] = rows.get(t, 0) + c
            if dp:
                phone.add(t)
    stays = {key: _combo_stay(got) for key, got in keeps.items()}
    by_key: dict[str, list[str]] = {}
    for t in sorted(rows):
        by_key.setdefault(t.casefold(), []).append(t)
    named: dict[str, Any] = {}  # D36: real organisational code -> its name pieces (None: not located)
    whole: set[str] = set()  # ... unless another column pseudonymises the same text whole (D35: a personal header)
    for p in cols:
        own = (p.params or {}).get(ORG_NAMES_KEY) or {}
        named.update(own)
        whole.update(raw.strip().casefold() for raw in _units(p) if raw.strip() not in own)
    named = {t: v for t, v in named.items() if t.casefold() not in whole}
    pyr = textgen.py_random(rng)
    drawer = _Drawer(real)
    mapping: dict[str, str] = {}
    drawn: set[str] = set()  # the pseudonyms so far (R7-07: a set, not a scan of the map per case form)
    item_fakes: dict[str, str] = {}  # K-11 (H-7): an item replaced inside a whole combination -> its run-wide pseudonym

    def item_fake(core: str, dp: bool) -> str:
        """The pseudonym of item ``core`` inside a whole combination: a pseudonymised unit's own, else ONE drawn for
        the run (a case form of it in that case form), never equal to a real text or another pseudonym."""
        if core in mapping:
            return mapping[core]
        if core in item_fakes:
            return item_fakes[core]
        base = next((mapping[f] for f in by_key.get(core.casefold(), ()) if f in mapping),
                    next((f for c, f in item_fakes.items() if c.casefold() == core.casefold()), None))
        fake = _case_like(base, core) if base is not None else None  # (as the case forms of a unit below)
        if fake is None or fake in drawn:
            fake = drawer.draw(lambda a, r=core, d=dp: pseudonym(r, pyr, d), pyr)
        item_fakes[core] = fake
        drawn.add(fake)
        return fake

    names = OrgNames(pyr, words if words is not None else _words_of(real)) if named else None
    # every unit first, then the whole combinations (K-11: their items take the units' pseudonyms, D36 named items
    # their fake names)
    for key in sorted(by_key, key=lambda k: (k in combos, k)):
        forms = by_key[key]
        rep = max(forms, key=lambda t: rows[t])  # the most common form (first in sorted order on a tie)
        if names is not None and any(t in named for t in forms):
            _draw_named(forms, rep, named, names, drawer, drawn, mapping, pyr)
            continue
        tests = accept.get(key) or []
        drawer.accept = (lambda c, ts=tests: all(f(c) for f in ts)) if tests else None
        dp = any(t in phone or bare_phone(t) for t in forms)  # a bare phone number under any header (R6-01)
        mark = marks.get(key)
        if mark is not None:  # R9-01: a whole number that is a real ID: a free number inside its column's range,
            # else new digits before its (zero) fraction
            got = None if dp else _number_in_range(rep, ranges[key], mark, pyr, drawer, spread.get(key))
            base = got if got is not None else                 drawer.draw(lambda a, r=rep, d=dp, m=mark: _whole_number_pseudonym(r, pyr, d, m), pyr)
        elif key in combos:
            base = drawer.draw(lambda a, r=rep, d=dp, c=combos[key], s=stays.get(key, set()):
                               _combo_pseudonym(r, c, s, lambda core, d=d: item_fake(core, d)), pyr)
        else:
            base = drawer.draw(lambda a, r=rep, d=dp: pseudonym(r, pyr, d), pyr)
        mapping[rep] = base
        drawn.add(base)
        for t in forms:
            if t == rep:
                continue
            fake = _case_like(base, t)
            if fake is None or fake in drawn:
                fake = drawer.draw(lambda a, r=t, d=dp: pseudonym(r, pyr, d), pyr)
            mapping[t] = fake
            drawn.add(fake)
    drawer.accept = None
    return mapping


# --------------------------------------------------------------------------- D36: names inside organisational codes
#: A word of a real text: letter runs joined by apostrophes (``O'Brien``).
_WORD_RUN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_APOSTROPHE = re.compile(r"['’]")
_CAMEL = re.compile(r"(?<=[^\W\d_])(?=[A-Z][a-z])")
#: A short surname prefix before an apostrophe (``O'Brien``, ``D'Angelo``): kept, the name after it is replaced.
_NAME_PREFIX = re.compile(r"[^\W\d_]{1,2}['’](?=[^\W\d_])")
#: Draws of one fake name word (Faker names of the length, then blends of two names) before the value is
#: pseudonymised whole instead.
NAME_DRAWS = 200
#: Column types whose values the run's real-text registry does not hold (IDs and code masks have their own rules).
_UNREGISTERED = frozenset({ColumnType.IDENTIFIER, ColumnType.DIGIT_CODE})


def text_words(texts: Iterable[Any], out: set[str]) -> set[str]:
    """Add to ``out`` the words of ``texts`` (accent-folded and case-folded, ``detect.names.fold_word``; two letters or
    more): every word, every part of it between apostrophes and every camelCase part (``O'Brien``: ``obrien``,
    ``brien``; ``AnaGarcia``: ``anagarcia``, ``ana``, ``garcia``). Returns ``out``. One pass of the word pattern over
    all the texts at once, then each distinct word once (a free-text column holds millions of words, few distinct)."""
    blob = "\n".join(t for t in texts if isinstance(t, str))
    for w in set(_WORD_RUN.findall(blob)):
        pieces = {w, *_APOSTROPHE.split(w)}
        pieces.update(x for piece in list(pieces) for x in _CAMEL.split(piece))
        for piece in pieces:
            k = fold_word(piece)
            if len(k) >= 2:
                out.add(k)
    return out


def _words_of(real: Iterable[str]) -> frozenset[str]:
    return frozenset(text_words(real, set()))


def real_words(tables: Iterable[Any], registry: Any = None) -> frozenset[str]:
    """D36: every word of every real text of ``tables`` (:func:`text_words`): the headers, the values and NA tokens of
    every column (excluded and nulled ones too), the text outside the tables and the names of flagged headers and paths
    (D34, the run's full-name index: ``generate.pii.plan_names``). With the run's real-text ``registry``
    (``privacy.texts``: every text value of the run, written or not, except IDs and code masks) its texts stand for
    the values of the columns it covers, so a free-text column is read once. No fake name inside an organisational
    code is one of them, so it never equals a real name word of the run."""
    texts: set[str] = set()
    for tp in tables:
        texts.update(h for h in (getattr(tp, "header", None) or ()) if isinstance(h, str))
        for p in tp.columns:
            texts.add(str(p.name))
            m = p.model
            if m is None or not (registry is None or p.role == "id" or p.type in _UNREGISTERED):
                continue
            texts.update(k for counts in (m.nonmissing_counts, m.missing_counts) for k in counts if isinstance(k, str))
            texts.update(m.real_values or ())
        for vals in (tp.private.extra.get(real_texts.DROPPED_KEY) or {}).values():
            texts.update(vals)
        for w, partners in (tp.private.extra.get("run_names") or {}).items():
            texts.add(w)
            texts.update(partners)
    if registry is not None:
        texts.update(registry.keys)
        texts.update(registry.kept)
        for _, raw in registry.outside:
            texts.update(raw)
    return frozenset(text_words(texts, set()))


class OrgNames:
    """D36: the run's fake names for the person names inside organisational codes (PRIVATE, drawn from the seed).

    Each real name word (``detect.names.fold_word``) gets ONE fake word for the run, a Faker given name or surname of
    its length by its role (``generate.pii.name_word``; blends of two names when the lengths run out), never a word of
    ``avoid`` (the run's real words, :func:`real_words`) nor the fake of another word; written in the real word's case
    style, its accents carried (``generate.text.style_like``) and a short prefix before an apostrophe kept
    (``O'Brien`` -> ``O'Hughes``). An initial gets another letter (a random permutation of A-Z without fixed points).
    """

    def __init__(self, pyr: random.Random, avoid: AbstractSet[str]) -> None:
        self.pyr = pyr
        self.avoid = avoid
        self.words: dict[str, str] = {}
        self.used: set[str] = set()
        self.letters: dict[str, str] | None = None

    def word(self, key: str, role: str) -> str | None:
        """The fake word of real name word ``key`` (lower-case ASCII) as Faker writes it (``McCoy``), or None when
        none can be drawn."""
        got = self.words.get(key)
        if got is not None:
            return got
        from .pii import name_word

        kind = "first" if role == FIRST else "last"
        for a in range(NAME_DRAWS):
            name = name_word(kind, max(len(key), 2), a, self.pyr)
            cand = fold_word(name)
            if len(cand) >= 2 and cand != key and cand not in self.avoid and cand not in self.used and (
                    name.isascii() and name.isalpha()):
                self.words[key] = name
                self.used.add(cand)
                return name
        return None

    def initial(self, piece: str) -> str:
        """Another letter for initial ``piece``, in its case."""
        if self.letters is None:
            abc = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
            perm = abc[:]
            while any(a == b for a, b in zip(abc, perm)):
                self.pyr.shuffle(perm)
            self.letters = dict(zip(abc, perm))
        base = fold_word(piece)[:1].upper()
        out = self.letters.get(base, "X")
        return out.lower() if piece[:1].islower() else out

    def render(self, fake: str, piece: str) -> str:
        """Fake word ``fake`` written like real name word ``piece``: in its case style (a capitalised word as Faker
        writes the name: ``McCoy``), its accents carried, a short prefix before an apostrophe kept."""
        m = _NAME_PREFIX.match(piece)
        if m is not None:
            return m.group() + self.render(fake, piece[m.end():])
        if piece.isascii() and piece[:1].isupper() and not piece.isupper():
            return fake
        return textgen.style_like(fake, piece)

    def compose(self, text: str, tokens: Iterable[Iterable[Any]]) -> str | None:
        """``text`` with each name piece (``[start, end, role]``) replaced, or None when a fake word cannot be drawn."""
        out: list[str] = []
        pos = 0
        for a, b, role in tokens:
            a, b = int(a), int(b)
            out.append(text[pos:a])
            piece = text[a:b]
            if role == INITIAL:
                out.append(self.initial(piece))
            else:
                fake = self.word(fold_word(piece), str(role))
                if fake is None:
                    return None
                out.append(self.render(fake, piece))
            pos = b
        out.append(text[pos:])
        return "".join(out)


def _draw_named(forms: list[str], rep: str, named: dict[str, Any], names: OrgNames, drawer: "_Drawer",
                drawn: set[str], mapping: dict[str, str], pyr: random.Random) -> None:
    """D36: the pseudonyms of the case forms ``forms`` (``rep`` the most common) of one organisational code holding a
    person's name: the value with its name pieces replaced (:class:`OrgNames`), accepted like any pseudonym (no real
    text of the run, no earlier pseudonym, no personal-data shape to correct; another case form of the same rewrite is
    accepted as such), else a whole same-shape pseudonym (:func:`pseudonym`)."""
    first: str | None = None
    for t in [rep, *(f for f in forms if f != rep)]:
        tokens = named.get(t)
        fake = names.compose(t, tokens) if tokens else None
        if fake is not None and embedded.guard(fake, pyr) is fake and fake not in drawn and (
                (first is not None and fake.casefold() == first.casefold()) or drawer.ok(fake)):
            drawer.take(fake)
        else:
            fake = drawer.draw(lambda a, r=t: pseudonym(r, pyr), pyr)
        if first is None:
            first = fake
        mapping[t] = fake
        drawn.add(fake)


def column_map(p: ColumnProfile, shared: dict[str, str]) -> dict[str, str]:
    """Column ``p``'s map, raw unit (as written: whitespace around it kept) -> pseudonym, from ``shared``."""
    out: dict[str, str] = {}
    for raw in _units(p):
        t = raw.strip()
        fake = shared.get(t)
        if fake is None:
            continue
        lead = raw[: len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()):]
        out[raw] = lead + fake + trail
    return out


def _cased(ch: str, like: str) -> str:
    out = ch.upper() if like.isupper() else ch.lower() if like.islower() else ch
    return out if len(out) == 1 else ch


def _plain(template: str, pyr: random.Random) -> str:
    out: list[str] = []
    for tok in textgen._TOKEN_RE.findall(template):
        if tok[0].isalpha():
            word = textgen.lorem_word(len(tok), pyr, textgen.accent_bases(tok) if not tok.isascii() else None)
            word = textgen.style_like(word, tok)
            if len(word) == len(tok):  # the unit's per-letter case pattern (``McKinley`` -> ``EsSecillum``)
                word = "".join(_cased(w, t) for w, t in zip(word, tok))
            out.append(word)
        elif tok[0].isdigit():
            out.append(textgen.random_digits(tok, pyr))
        else:
            out.append(tok)
    return "".join(out)


def _whole_number_pseudonym(template: str, pyr: random.Random, digit_phone: bool, mark: str) -> str:
    """R9-01: the pseudonym of a whole number written in a number column (``482913.0``, ``$482,913.00``): the part
    before the decimal mark gets new digits (:func:`pseudonym`), the fraction (zeros: the number is whole) stays, so
    the column's decimals are kept."""
    i = template.rfind(mark)
    if i > 0 and template[i + 1:i + 2].isdigit():
        return pseudonym(template[:i], pyr, digit_phone) + template[i:]
    return pseudonym(template, pyr, digit_phone)


def _combo_stay(keeps: list[tuple[frozenset[str], tuple[str, ...]]]) -> set[str]:
    """K-11: the unit keys of the items a whole combination writes as they are, from what each column holding it keeps
    (``detect.text.COMBO_KEEP_KEY``: free words, real items). Columns that disagree (two fields deciding the same text
    apart) keep only the free words all of them keep and at most one real item all of them keep: a real sequence one
    column may write together need not be one the other may."""
    free = frozenset.intersection(*(f for f, _ in keeps))
    reals = {r for _, r in keeps}
    if len(reals) == 1:
        return set(free) | set(next(iter(reals)))
    both = set.intersection(*(set(r) for r in reals))
    return set(free) | set(sorted(both)[:1])


def _combo_pseudonym(template: str, delimiter: str, stay: set[str], fake_of: Callable[[str], str]) -> str:
    """K-11: the written form of a whole multi-select combination the D35 allowlist did not keep
    (``detect.text.combination_gate``): each item whose :func:`detect.text.unit_key` is in ``stay`` (its free words
    and the real items a common whole value holds together, or at most one, ``detect.text.COMBO_KEEP_KEY``) stays,
    every other item becomes ``fake_of(item)`` (H-7: the item's ONE pseudonym for the run, a pseudonymised unit's own
    one included), and so does an item that is a real person ID of the run whatever the column decided (R9-01:
    ``privacy.realids``, active while the map is drawn); separators and the spacing around the items stay. Distinct
    combinations get distinct forms: the item pseudonyms are distinct and no real text."""
    out: list[str] = []
    for part in template.split(delimiter):
        core = part.strip()
        if not core or (unit_key(core) in stay and not realids.holds(core)):
            out.append(part)
            continue
        out.append(part[: len(part) - len(part.lstrip())] + fake_of(core) + part[len(part.rstrip()):])
    return delimiter.join(out)


def pseudonym(template: str, pyr: random.Random, digit_phone: bool = False) -> str:
    """A same-shape pseudonym of one real unit (module docstring): ``Rift Valley`` -> ``Duis Tempora``, ``AB-12`` ->
    ``Ea-47``; personal-data shapes inside it follow their PII generators (``generate.embedded.rewrite``;
    ``digit_phone``: a whole 10/11-digit unit is a phone number, as under a person/contact/notes header)."""
    return embedded.rewrite(template, lambda part: _plain(part, pyr), pyr, digit_phone=digit_phone)


# --------------------------------------------------------------------------- applying the maps
def mapped_counts(counts: dict[Any, int], mapping: dict[Any, Any]) -> dict[Any, int]:
    """``counts`` with every pseudonymised real value replaced by its pseudonym (same counts)."""
    if not mapping:
        return counts
    out: dict[Any, int] = {}
    for raw, c in counts.items():
        key = mapping.get(raw, raw)
        out[key] = out.get(key, 0) + c
    return out


def _rewrite_items(raw: str, mapping: dict[Any, Any], d: str) -> str:
    return d.join(mapping.get(part, part) for part in raw.split(d))


def generation_model(p: ColumnProfile, model: ColumnModel) -> ColumnModel:
    """``model`` (row or per-entity) with its pseudonymised units replaced: what the generators sample from. A
    MULTI_SELECT has its pseudonymised items replaced inside each value (separators, item order, counts kept)."""
    mapping = level_map(p)
    if not mapping:
        return model
    d = (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None
    if d:  # (a multi-select without a delimiter has whole values as its units, like levels)
        counts: dict[Any, int] = {}
        for raw, c in model.nonmissing_counts.items():
            key = (mapping[raw] if _whole(raw, d, mapping) else _rewrite_items(raw, mapping, d)
                   if isinstance(raw, str) else raw)
            counts[key] = counts.get(key, 0) + c
        return replace(model, nonmissing_counts=counts)
    return replace(model, nonmissing_counts=mapped_counts(model.nonmissing_counts, mapping))


def mapped_value(p: ColumnProfile, raw: Any) -> Any:
    """The value written for real value ``raw`` of column ``p`` (``raw`` itself when nothing in it is
    pseudonymised)."""
    mapping = level_map(p)
    if not mapping or not isinstance(raw, str):
        return raw
    d = (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None
    if _whole(raw, d, mapping):
        return mapping[raw]
    return _rewrite_items(raw, mapping, d) if d else mapping.get(raw, raw)


#: PRIVATE ``ColumnModel.extra`` key: (the map it inverts, pseudonym -> real unit), built once per map
#: (:func:`inverse_map`).
_INVERSE_KEY = "_pseudonym_inverse"


def inverse_map(p: ColumnProfile) -> dict[Any, Any]:
    """The column's pseudonym -> real unit map, built once per map and cached beside it in the PRIVATE model
    (R7-07: rebuilding it for every looked-up value made functional-dependency rebuilds quadratic in the levels).
    A map replaced later (``build_maps``) gets a new inverse: the cache holds the map it was built from."""
    mapping = level_map(p)
    if not mapping:
        return {}
    cached = p.model.extra.get(_INVERSE_KEY)
    if cached is None or cached[0] is not mapping:
        cached = (mapping, {v: k for k, v in mapping.items()})
        p.model.extra[_INVERSE_KEY] = cached
    return cached[1]


def real_value(p: ColumnProfile, syn: Any) -> Any:
    """The real value a written value of column ``p`` stands for (inverse of :func:`mapped_value`)."""
    inverse = inverse_map(p)
    if not inverse or not isinstance(syn, str):
        return syn
    d = (p.params or {}).get("delimiter") if d35_units(p) == ITEMS else None
    if _whole(syn, d, inverse):
        return inverse[syn]
    return _rewrite_items(syn, inverse, d) if d else inverse.get(syn, syn)


__all__ = ["NAME_DRAWS", "OrgNames", "RUN_MAP_KEY", "RUN_TEXTS_KEY", "apply_na_map", "build_maps", "column_map",
           "draw_map", "generation_model", "inverse_map", "level_map", "mapped_counts", "mapped_value", "plan",
           "pseudonym", "real_value", "real_words", "text_words"]
