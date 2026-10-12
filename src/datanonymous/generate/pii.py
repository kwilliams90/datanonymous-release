"""PII value generation: names, emails, phones, SSNs, ZIP/state/city, addresses, IP/URL (AREA C).

Design D12, D24, section 3 #25-31, section 10. Faker (en_US, version pinned and
recorded in the manifest) is seeded per column through ``generate.rng.faker_for``
(``Faker.seed_instance``) and used where it fits: person names, city names and
street names (D24). Fields defined by their format stay mask-based so the
source shape is kept exactly.

Never-copy types (EMAIL, PERSON_NAME, PHONE, SSN, STREET_ADDRESS, IP, URL,
CITY) use :func:`generate.text.per_distinct`: one synthetic value per real
distinct value (grouped case-insensitively; raw case/whitespace variants kept),
rows by exact quota over the real counts, so the distinct ratio, singleton
share and frequency profile are kept exactly at the same N. A synthetic value
never equals a real value of the column (case-insensitive, trimmed): it is
redrawn. As a consequence separate first/last name columns can never recreate
a real (first, last) pair of their own rows: neither part is ever a real value
of its column. Run-wide (R1-04), :func:`avoid_real_full_names` redraws a
synthetic full name, or a synthetic middle/last name that pairs with a
synthetic earlier name part of its table, that spells a real person's full name
held anywhere in the run (another file, another column), including complete
names of a single distinct word (``Whitney``, ``Mr. Whitney``: R2-06);
:func:`real_name_pairs` (pass 1) and :func:`plan_names` (run planning) build
that private index, and :func:`real_full_name_hits` re-checks the final rows
before writing. Name words are compared without accents (``García`` is
``Garcia``), and a cell of two or more name words is a full name whatever its
column's subtype (a ``Nombre`` column may hold ``María García``: R3-04).

* PERSON_NAME (first/middle/last/full): each real value's layout is mimicked
  token by token (``Last, First M.``, ``First Middle Last``, hyphenated and
  ``O'``-prefixed parts, initials, titles such as ``Mrs.``, suffixes such as
  ``Jr``, particles such as ``de la``, couples joined by ``&``/``/``/``and``;
  mixed-case words such as ``McDonald`` keep a short prefix and their case
  style); every name word becomes a Faker first
  or last name of the same length where one exists (else of a close length,
  else a blend of two Faker names), in the word's case style, with the word's
  non-ASCII letters kept in place (accent share and encoding kept).
* CITY: a Faker city with the same number of words, in the value's case style;
  a ``City, ST`` value keeps its state code (digits after it are randomised).
* STREET_ADDRESS: house number by interpolated inverse-CDF over the real house
  numbers, Faker street-name words of the same length, the value's own street
  suffix, directionals, unit word and ordinal endings kept, other words and
  digits replaced (so suffix and unit shares are kept exactly).
* EMAIL (when no derived template applies; AREA D handles templates): the
  local part keeps its mask (letter runs become Faker name words of the same
  length, digit runs random digits, separators kept); each real domain maps to
  a reserved domain (example.org, example.com, example.net, then
  ``dN.example.org`` ...) by frequency rank, so the number of distinct domains
  and their shares are kept.
* PHONE: the value's own mask; the area code comes from the k-floored observed
  pmf: a code held by >= ``--min-cell`` rows is kept, a rarer one is replaced by
  one of the kept codes drawn in proportion to their counts (when no code
  reaches ``--min-cell``, each rare code gets an invented area code that is no
  real area code of the column); ordinary exchange and line number; random
  extension digits.
* SSN: the value's own mask (separators and masking characters kept), area
  900-999 (never issued), random group 01-99 and serial 0001-9999.
* ZIP: ZIP5 codes held by >= ``--min-cell`` rows are kept (exact quota, raw
  text kept); rarer ones are replaced by invented same-width codes that are no
  real ZIP5 of the column (one per rare code, small-cell policy); the +4 part
  is random per row (a ZIP+4 never equals a real one); leading zeros kept. In a
  dense column (more rare codes than free codes of the width) free codes are
  reused, and when every code of the width is real, rare codes are pooled into
  the kept ones: a rare real code is never written.
* STATE: real state codes/names by exact quota (floor 1 per level), like a factor;
  other values (forced types) become same-shape masks, never real.
* IP: IPv4 -> 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24, then
  198.18.0.0/15 (one address per real distinct address); IPv6 -> 2001:db8::/32.
* URL: scheme, ``www.`` prefix and port kept; hosts mapped by frequency rank
  to example.org/.com/.net (then ``hN.example.org``); path/query/fragment
  replaced by a same-shape mask.

Values that do not fit their type's pattern (``--pii-col``/``--force-type`` on
other values) become same-shape character-class masks (never real). A numeric
PHONE with an unsupported layout fails clearly instead of using a mask fallback.
"""

from __future__ import annotations

import ipaddress
import random
import re
from bisect import bisect_right
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..detect.text import (
    ADDRESS_RE,
    DIRECTIONALS,
    NAME_PARTICLES,
    NAME_SUFFIXES,
    NAME_TITLES,
    STREET_SUFFIXES,
    UNIT_WORDS,
    URL_PARSE_RE,
    US_STATE_ABBR,
    ZIP_HINTED_RE,
    case_style,
    is_state,
)
from ..detect.text import DEFAULT_MIN_CELL, ZIP_HOLDERS_KEY
from ..errors import GenerationError, UnsupportedColumnError
from ..hints import fold_accents
from ..model import ColumnModel, ColumnProfile, ColumnType, Message
from ..profile.codes import codes_of
from ..profile.stats import missing_mask
from ..privacy import digits as phone_privacy
from . import text as textgen
from .quota import quota_sample
from .text import Drawer, Group, accent_bases, mask_fill, mimic_words, per_distinct, random_digits, style_like

T = ColumnType

#: Reserved e-mail / URL domains (RFC 2606); subdomains of them are reserved too.
RESERVED_DOMAINS = ("example.org", "example.com", "example.net")
_ORDINALS = frozenset({"st", "nd", "rd", "th"})
_N11 = frozenset(f"{d}11" for d in range(2, 10))


# --------------------------------------------------------------------------- Faker data
@lru_cache(maxsize=1)
def name_pools() -> dict[str, tuple[tuple[str, ...], dict[int, tuple[str, ...]]]]:
    """Faker en_US first and last names (letters only): kind -> (all names, names by length)."""
    from faker.providers.person.en_US import Provider

    out = {}
    for kind, source in (("first", Provider.first_names), ("last", Provider.last_names)):
        names = tuple(dict.fromkeys(n for n in source if n.isalpha() and n.isascii()))
        by_len: dict[int, list[str]] = {}
        for nm in names:
            by_len.setdefault(len(nm), []).append(nm)
        out[kind] = (names, {k: tuple(v) for k, v in by_len.items()})
    return out


def name_word(kind: str, length: int, attempt: int, pyr: random.Random) -> str:
    """A Faker name of ``length`` letters (title case).

    Attempts 0-19 draw Faker names of exactly that length (names one letter
    longer/shorter join from attempt 6, two letters from attempt 12, or at once
    when fewer than 3 names fit); later attempts, and lengths no name is close
    to, blend two Faker names to the exact length. Length 1 is a random letter.
    """
    names, by_len = name_pools()[kind]
    if length <= 1:
        return pyr.choice(textgen._ASCII_UPPER)
    if attempt < 20:
        cands: list[str] = list(by_len.get(length, ()))
        for delta, start in ((1, 6), (2, 12)):
            if len(cands) < 3 or attempt >= start:
                cands += list(by_len.get(length + delta, ())) + list(by_len.get(length - delta, ()))
        if cands:
            return pyr.choice(cands)
    return blend_name(names, length, pyr)


def blend_name(names: tuple[str, ...], length: int, pyr: random.Random) -> str:
    """The start of one Faker name joined to the end of another, exactly ``length`` letters."""
    a = pyr.choice(names)
    b = pyr.choice(names).lower()
    cut = min(len(a), max(1, (length + 1) // 2))
    tail = (b[1:] + b) * (length // max(len(b), 1) + 2)
    s = a[:cut] + tail[: max(length - cut, 0)]
    return s[:1].upper() + s[1:].lower()


def _seeded_faker(rng: np.random.Generator) -> Any:
    from faker import Faker

    fk = Faker("en_US")
    fk.seed_instance(int(rng.integers(0, 2**32 - 1)))
    return fk


def _min_cell(profile: ColumnProfile) -> int:
    return int((profile.params or {}).get("_min_cell", DEFAULT_MIN_CELL))


# --------------------------------------------------------------------------- the AREA C hook
def generate_pii(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
                 faker_factory: Callable[[], Any]) -> tuple[np.ndarray, dict[str, bool]]:
    """AREA C hook: synthetic values for a PII-typed column.

    Called by ``generate/table.py`` for EMAIL, PERSON_NAME, PHONE, SSN, ZIP, STATE,
    CITY, STREET_ADDRESS, IP and URL columns with ``n`` = number of non-missing
    slots. ``faker_factory()`` returns a Faker seeded for this column (a Faker
    seeded from ``rng`` is used when it returns None). Returns (rendered
    strings, flags: ``rare_category_adjusted``, and ``never_copy_adjusted`` when
    a value had to be lengthened to avoid every real value). Never returns a
    real value of a never-copy type (see the module docstring).
    """
    if n == 0:
        return np.empty(0, dtype=object), {"rare_category_adjusted": False}
    t = profile.type
    if t is T.STATE:
        return _state(profile, model, n, rng)
    fk = faker_factory() if faker_factory is not None else None
    if fk is None:
        fk = _seeded_faker(rng)
    pyr: random.Random = fk.random
    if t is T.ZIP:
        return _zip(profile, model, n, rng, pyr)
    make = _MAKERS.get(t)
    if make is None:
        raise UnsupportedColumnError(f"column {profile.name!r}: {t.value} is not a PII type")
    groups = textgen.distinct_groups(model)
    if not groups:
        raise GenerationError(f"column {profile.name!r}: no real values to model the column on")
    base_maker = make(profile, model, groups, fk, pyr)
    drawer = (Drawer(textgen.real_casefold(profile, model), phone=True) if t is T.PHONE else None)
    return per_distinct(profile, model, n, rng, base_maker, pyr, drawer)


# --------------------------------------------------------------------------- PERSON_NAME
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+|.", re.S)


def fake_name(rep: str, subtype: str | None, attempt: int, pyr: random.Random,
              avoid: frozenset[str] = frozenset()) -> str:
    """A fake name with the layout of ``rep`` (see the module docstring).

    ``avoid``: case-folded name words of the column's real values; each fake word tries not to be one of
    them (best effort: 'Maritza de la Cruz' should not become 'Mackenzie de la Cruz'). Whole values are
    checked strictly by the caller (never a real value).
    """
    chunks = re.split(r"(\s+|,)", rep)
    # role of each chunk: 'last' before a comma (Last, First), else the last name chunk of a full name
    words_idx = [i for i, ch in enumerate(chunks) if ch and not ch.isspace() and ch != ","
                 and any(c.isalpha() for c in ch) and not _is_vocab_chunk(ch)]
    roles: dict[int, str] = {}
    comma = next((i for i, ch in enumerate(chunks) if ch == ","), None)
    for i in words_idx:
        if subtype in ("first", "middle"):
            roles[i] = "first"
        elif subtype == "last":
            roles[i] = "last"
        elif comma is not None:
            roles[i] = "last" if i < comma else "first"
        else:
            roles[i] = "last" if (i == words_idx[-1] and len(words_idx) > 1) or len(words_idx) == 1 else "first"
    out = []
    for i, ch in enumerate(chunks):
        if i not in roles:
            if ch and any(c.isalpha() for c in ch) and _is_initial(ch):
                out.append(_fake_initial(ch, pyr))
            else:
                out.append(ch)
            continue
        out.append(_fake_name_chunk(ch, roles[i], attempt, pyr, avoid))
    return "".join(out)


def _bare(ch: str) -> str:
    return ch.strip(".()\"'").casefold()


def _is_initial(ch: str) -> bool:
    letters = [c for c in ch if c.isalpha()]
    return len(letters) == 1 and len(ch.rstrip(".")) <= 2


#: Lower-case words joining two people in one cell ("Mr. and Mrs. Lee"); kept like titles.
_NAME_CONNECTORS = frozenset({"and"})


def _is_vocab_chunk(ch: str) -> bool:
    b = _bare(ch)
    lower_word = ch.islower() and (b in NAME_PARTICLES or b in _NAME_CONNECTORS)
    return b in NAME_TITLES or b in NAME_SUFFIXES or lower_word or _is_initial(ch)


def _fake_initial(ch: str, pyr: random.Random) -> str:
    out = []
    for c in ch:
        if c.isalpha():
            out.append(pyr.choice(textgen._ASCII_UPPER) if c.isupper() else pyr.choice(textgen._ASCII_LOWER))
        else:
            out.append(c)
    return "".join(out)


def _fake_name_chunk(chunk: str, role: str, attempt: int, pyr: random.Random,
                     avoid: frozenset[str] = frozenset()) -> str:
    toks = _NAME_TOKEN_RE.findall(chunk)
    out = []
    for j, tok in enumerate(toks):
        if not tok[0].isalpha():
            out.append(tok)
            continue
        nxt = toks[j + 1] if j + 1 < len(toks) else ""
        if len(tok) <= 2 and nxt in ("'", "\u2019") and j + 2 < len(toks):
            out.append(tok)  # O'Brien, D'Angelo: the prefix is kept, the name after it is replaced
            continue
        parts = _CAMEL_PARTS.findall(tok) if case_style(tok) == "mixed" else []
        if len(parts) > 1 and "".join(parts) == tok:
            # mixed case (McDonald, DeShawn, MacArthur): each capitalised part is replaced on its own and a
            # leading surname prefix (Mc, Mac, De, La ...: _NAME_PREFIXES) is kept, so the case style stays 'mixed'
            out.append("".join(p if i == 0 and p.casefold() in _NAME_PREFIXES
                               else _fake_name_word(p, role, attempt, pyr, avoid) for i, p in enumerate(parts)))
            continue
        out.append(_fake_name_word(tok, role, attempt, pyr, avoid))
    return "".join(out)


#: Capitalised parts of a mixed-case name word ('McDonald' -> 'Mc', 'Donald').
_CAMEL_PARTS = re.compile(r"[^\W\d_][^\W\d_A-Z]*")
#: Surname prefixes kept in a mixed-case word (like the O' of O'Brien); any other part is replaced.
_NAME_PREFIXES = frozenset({"mc", "mac", "de", "la", "le", "di", "da", "du", "van", "von", "st", "fitz"})


def _fake_name_word(tok: str, role: str, attempt: int, pyr: random.Random,
                    avoid: frozenset[str] = frozenset()) -> str:
    """A Faker name of ``tok``'s length in its case style, carrying its accented letters, preferably not
    one of the ``avoid`` words (real name words of the column)."""
    bases = accent_bases(tok) if not tok.isascii() else []
    out = ""
    for _ in range(_AVOID_TRIES if avoid else 1):
        cand = name_word(role, len(tok), attempt, pyr)
        for _ in range(12 if bases else 0):  # prefer a name that can carry the accent (an accented name keeps it)
            if all(b in cand.lower() for b in bases):
                break
            cand = name_word(role, len(tok), attempt, pyr)
        out = style_like(cand, tok)
        if out.casefold() not in avoid:
            break
    return out


#: Draws per name word to avoid the column's real name words (best effort; whole values are strict).
_AVOID_TRIES = 8


def name_words(values: Iterable[str]) -> frozenset[str]:
    """Case-folded letter runs (2+ letters) of real name values: the words a fake name avoids."""
    out: set[str] = set()
    for v in values:
        out.update(w.casefold() for w in _NAME_TOKEN_RE.findall(v) if len(w) > 1 and w[0].isalpha())
    return frozenset(out)


def _make_name(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
               pyr: random.Random) -> Callable[[Group, int], str]:
    sub = profile.subtype or "full"
    keep = NAME_TITLES | NAME_SUFFIXES | NAME_PARTICLES | _NAME_CONNECTORS  # kept verbatim anyway
    avoid = name_words(raw for g in groups for raw, _ in g.variants) - keep
    return lambda g, a: fake_name(g.rep, sub, a, pyr, avoid)


# --------------------------------------------------------------------------- run-wide real full names (R1-04)
#: Name words that never make a person (titles, suffixes, particles, "and"): not paired.
_PAIR_SKIP = NAME_TITLES | NAME_SUFFIXES | NAME_PARTICLES | _NAME_CONNECTORS
#: Order of the name parts in one person's name (first, then middle, then last).
_PART_RANK = {"first": 0, "middle": 1, "last": 2}

#: PRIVATE run-wide index of real full names: a name word used as a surname -> the real name words that
#: precede it in one person. From the first/middle and last cells of one real row the roles are known (one
#: direction: synthetic ``Gordon Paul`` is not the real ``Paul Gordon``); a full-name value's layout is not
#: (``Burton Chelsea`` may be last-first), so its words are linked both ways. The reserved key
#: :data:`WHOLE` holds the complete real names that have fewer than two distinct name words (``Whitney``,
#: ``Mr. Whitney``, ``Smith Smith``, ``J. K.``; :func:`whole_name_key`), which no word pair can represent
#: (R2-06). Never serialised.
NameIndex = dict[str, set[str]]
#: Key of the complete single-word real names in a :data:`NameIndex` (never a name word: those have >= 2 letters).
WHOLE = ""


def person_words(value: str) -> list[str]:
    """Case-folded, accent-folded name words of ``value`` (letter runs of >= 2 letters; titles, suffixes,
    particles out): ``García`` and ``Garcia`` are one name word (R3-04)."""
    # Keep the public list contract, without exposing the cached immutable tuple.
    return list(_person_words(value))


@lru_cache(maxsize=1 << 16)
def _person_words(value: str) -> tuple[str, ...]:
    out = []
    for w in _NAME_TOKEN_RE.findall(value):
        k = fold_accents(w).casefold()
        if len(k) > 1 and k[0].isalpha() and k not in _PAIR_SKIP:
            out.append(k)
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def whole_name_key(value: str) -> str | None:
    """Comparison key of a complete name that no word pair covers (R2-06), or None.

    One distinct name word (``Whitney``, ``Mr. Whitney``, ``Whitney Jr``, ``Smith Smith``, ``de la Cruz``):
    that word. No name word but initials (``J. K.``): ``#`` + the initials. Two or more distinct name words:
    None (the pair index covers every two of them). Titles, suffixes and particles alone: None (no name).
    """
    words = set(_person_words(value))
    if len(words) > 1:
        return None
    if words:
        return next(iter(words))
    initials = [w.casefold() for w in _NAME_TOKEN_RE.findall(value) if len(w) == 1 and w.isalpha()]
    return "#" + "".join(initials) if initials else None


def _link(index: NameIndex, first: str, last: str) -> None:
    if first != last:
        index.setdefault(last, set()).add(first)


def _add_full(index: NameIndex, value: str) -> None:
    words = list(dict.fromkeys(_person_words(value)))
    for i, a in enumerate(words):
        for b in words[i + 1:]:
            _link(index, a, b)
            _link(index, b, a)
    key = whole_name_key(value)
    if key is not None:
        index.setdefault(WHOLE, set()).add(key)


def _name_columns(profiles: Iterable[ColumnProfile]) -> list[ColumnProfile]:
    return [p for p in profiles if p.type is T.PERSON_NAME and p.role not in ("excluded", "null")]


def _part_columns(names: list[ColumnProfile]) -> list[ColumnProfile]:
    """First, middle and last name columns in name order (stable within a part)."""
    return sorted((p for p in names if p.subtype in _PART_RANK), key=lambda p: _PART_RANK[p.subtype])


def _distinct_name_rows(columns: list[np.ndarray]) -> Iterable[tuple[Any, ...]]:
    """Distinct name-part tuples in first-appearance order, grouped on integer codes."""
    encoded = [codes_of(values) for values in columns]
    n = min((len(codes) for codes, _ in encoded), default=0)
    frame = pd.DataFrame({i: codes[:n] for i, (codes, _) in enumerate(encoded)})
    for row in frame.drop_duplicates().itertuples(index=False, name=None):
        yield tuple(uniques[code] for code, (_, uniques) in zip(row, encoded))


def real_name_pairs(columns: list[np.ndarray], profiles: list[ColumnProfile]) -> NameIndex | None:
    """Pass 1 (``profile.table``): the real full names of one table as a :data:`NameIndex`, or None.

    Every pair of name words of a real full-name value, every (earlier part word, later part word) pair of the
    first/middle/last cells of one real row (separate name-part columns), and the complete names with fewer
    than two distinct name words (a full-name value, or the name cells of one row: ``Whitney`` with a blank
    last name) under :data:`WHOLE`. Kept PRIVATE in ``tp.private.extra``.
    """
    names = _name_columns(profiles)
    if not names:
        return None
    index: NameIndex = {}
    for p in names:
        if p.model is None:
            continue
        full = (p.subtype or "full") == "full"
        for raw in p.model.nonmissing_counts:
            # a name-part cell of two or more name words is a whole name too ('María García' under 'Nombre')
            if isinstance(raw, str) and (full or len(set(person_words(raw))) > 1):
                _add_full(index, raw)
    parts = _part_columns(names)
    if parts:
        nm = [p.model.nonmissing_counts if p.model is not None else {} for p in parts]
        # Keep the old set iteration semantics, inserting the same row tuples in
        # first-appearance order, without constructing a tuple for every row.
        for row in set(_distinct_name_rows([columns[p.index] for p in parts])):
            cells = [(p, v) for p, v, m in zip(parts, row, nm) if isinstance(v, str) and v in m]
            for i, (pa, va) in enumerate(cells):
                for pb, vb in cells[i + 1:]:
                    if _PART_RANK[pa.subtype] < _PART_RANK[pb.subtype]:
                        for a in person_words(va):
                            for b in person_words(vb):
                                _link(index, a, b)
            key = whole_name_key(" ".join(v for _, v in cells)) if cells else None
            if key is not None:
                index.setdefault(WHOLE, set()).add(key)
    return index or None


#: A text of the run's real-text registry with at most this many words (whitespace-separated) and name words may be
#: a person's full name: :func:`plan_names` links its name words like a full name's (R7-01, R7-02).
NAME_MAX_WORDS = 5


def plan_names(files: Iterable[Any], extra_names: Iterable[Iterable[str]] = (), texts: Any = None) -> NameIndex:
    """Run-level planning (pipeline, after pass 1): merge every table's real full names into one index and
    hand it to every table (``tp.private.extra['run_names']``), so no file writes a real person's full name
    from another file (R1-04). ``extra_names``: the name words of real person names found outside the data
    (D34: a header, sheet, file or folder name, ``privacy.headers``), linked as full names. ``texts``: the run's
    real texts that it never writes (``privacy.texts.RealTexts``, R7-01/R7-02): each one of at most
    :data:`NAME_MAX_WORDS` words that is not a person-name column's value (those are indexed by
    :func:`real_name_pairs`) is linked as a full name too, so a rare label ``Roy Berger`` pseudonymised in another
    column, or a guardian's name in an ``--exclude``d column, is never spelled by a synthetic name or by the first
    and last name on one row; a longer text outside the tables (an Excel title or note, R8-04) has each pair of
    adjacent name words linked."""
    index: NameIndex = {}
    tables = [tp for fr in files for tp in fr.tables]  # a FAILed file's people are real people too
    for tp in tables:
        for w, partners in (tp.private.extra.get("real_name_pairs") or {}).items():
            index.setdefault(w, set()).update(partners)
    for words in extra_names:
        _add_full(index, " ".join(words))
    if texts is not None:
        from ..privacy.texts import LAYOUT

        for value in texts.texts():
            if value.count(" ") < NAME_MAX_WORDS and len(person_words(value)) <= NAME_MAX_WORDS:
                _add_full(index, value)
        # R8-04: a longer text outside the tables (a title, a note, dropped metadata) names its people by two name
        # words side by side ('Class roster for Ms. Tara Walsh, printed ...'): each adjacent pair is linked
        for value in texts.texts(frozenset({LAYOUT})):
            if value.count(" ") >= NAME_MAX_WORDS or len(person_words(value)) > NAME_MAX_WORDS:
                words = person_words(value)
                for a, b in zip(words, words[1:]):
                    _link(index, a, b)
                    _link(index, b, a)
    for tp in tables:
        tp.private.extra["run_names"] = index
    return index


def _pairs_real(index: NameIndex, words: list[str], firsts: set[str] | None = None) -> bool:
    """A real person's (first, last) pair among ``words`` in either order (a full name), or between a word
    of ``firsts`` (first/middle names) and a word of ``words`` (a last name)."""
    if firsts is None:
        return any(a != b and a in index.get(b, ()) for a in words for b in words)
    return any(not index.get(w, set()).isdisjoint(firsts) for w in words if w != WHOLE)


def _run_index(tp: Any) -> NameIndex | None:
    return tp.private.extra.get("run_names") or tp.private.extra.get("real_name_pairs")


def _column_values(p: ColumnProfile, values: np.ndarray) -> tuple[set[str], bool]:
    """(distinct non-missing synthetic strings of a name column, whether it has a missing cell)."""
    _, uniques = codes_of(values)
    miss = missing_mask(uniques, p.missing_policy, p.na_tokens)
    vals = {v for v in uniques[~miss] if isinstance(v, str) and v.strip()}
    return vals, bool(miss.any())


def avoid_real_full_names(tp: Any, cols: dict[int, np.ndarray], rng: np.random.Generator) -> list[Message]:
    """Generation (``generate/table.py``, before the structural switches): no real person's full name.

    The run-wide :data:`NameIndex` (``run_names``; the table's own ``real_name_pairs`` without a run plan)
    holds every real full name of the run. Redrawn (every cell of the value gets one new fake value; distinct
    count, frequencies and within-ID constancy kept; never a real value of the column):

    * a synthetic full-name value whose words pair into a real full name, or that IS a real complete name of
      fewer than two distinct words (:func:`whole_name_key`: ``Whitney``; R2-06);
    * a synthetic middle or last name that pairs with any synthetic word of an earlier name part of the table
      (first before middle before last), so no row arrangement, swap or structural rebuild can put a real
      person's two names on one row; and a synthetic name-part cell whose own words pair into a real full
      name (R3-04: a ``Nombre`` column of whole names);
    * a synthetic name-part value that is a real complete single-word name when it can stand alone on a row
      (the table has no other name-part column, or one with a missing cell) or next to the same word in
      another part column (``Smith`` + ``Smith``).

    Returns an info message with the count (never a value).
    """
    index = _run_index(tp)
    names = [p for p in _name_columns(tp.columns) if p.index in cols]
    if not index or not names:
        return []
    whole = index.get(WHOLE, set())
    pyr = textgen.py_random(rng)
    redrawn = 0
    for p in names:
        if (p.subtype or "full") == "full":
            redrawn += _redraw_values(tp, p, cols, lambda v: not _pairs_real(index, person_words(v))
                                      and whole_name_key(v) not in whole, pyr)
    parts = _part_columns(names)
    for p in parts:
        rank = _PART_RANK[p.subtype]
        earlier: set[str] = set()  # words of earlier name parts (already final: parts are processed in order)
        alone = len(parts) == 1
        other_keys: set[str] = set()
        for q in parts:
            if q is p:
                continue
            vals, has_missing = _column_values(q, cols[q.index])
            alone = alone or has_missing
            for v in vals:
                if _PART_RANK[q.subtype] < rank:
                    earlier.update(person_words(v))
                k = whole_name_key(v)
                if k is not None:
                    other_keys.add(k)

        def ok(v: str, earlier: set[str] = earlier, alone: bool = alone, other_keys: set[str] = other_keys) -> bool:
            words = person_words(v)
            if earlier and _pairs_real(index, words, earlier):
                return False
            if _pairs_real(index, words):
                return False  # a cell of two name words spells a real full name by itself (R3-04)
            k = whole_name_key(v)
            return k is None or k not in whole or not (alone or k in other_keys)

        redrawn += _redraw_values(tp, p, cols, ok, pyr)
    if not redrawn:
        return []
    return [Message("info", "name_pairs_redrawn",
                    f"{redrawn} synthetic name value(s) redrawn: together with the run's other name columns they "
                    "would have spelled a real person's full name")]


def real_full_name_hits(tp: Any, columns: dict[int, np.ndarray]) -> int:
    """Final check (``privacy.leaks.scan_table``, before writing): synthetic full-name values and rows of
    name-part cells that spell a real full name of the run (word pairs and :data:`WHOLE` names), counted on
    the final rows. :func:`avoid_real_full_names` makes this 0 for any row arrangement; a hit FAILs the file."""
    index = _run_index(tp)
    names = [p for p in _name_columns(tp.columns) if p.index in columns]
    if not index or not names:
        return 0
    whole = index.get(WHOLE, set())
    hits = 0
    for p in names:
        vals, _ = _column_values(p, columns[p.index])
        if (p.subtype or "full") == "full":
            hits += sum(1 for v in vals if _pairs_real(index, person_words(v)) or whole_name_key(v) in whole)
        else:  # a name-part cell of two or more name words is a whole name too (R3-04)
            hits += sum(1 for v in vals if _pairs_real(index, person_words(v)))
    parts = _part_columns(names)
    if parts:
        n = min(len(columns[p.index]) for p in parts)
        frame = pd.DataFrame({j: np.where(missing_mask(columns[p.index], p.missing_policy, p.na_tokens), None,
                                          np.asarray(columns[p.index], dtype=object))[:n]
                              for j, p in enumerate(parts)})
        for row in frame.drop_duplicates().itertuples(index=False, name=None):
            cells = [(p, v) for p, v in zip(parts, row) if isinstance(v, str) and v.strip()]
            bad = any(_PART_RANK[pa.subtype] < _PART_RANK[pb.subtype]
                      and _pairs_real(index, person_words(vb), set(person_words(va)))
                      for i, (pa, va) in enumerate(cells) for pb, vb in cells[i + 1:])
            key = whole_name_key(" ".join(v for _, v in cells)) if cells else None
            if bad or (key is not None and key in whole):
                hits += 1
    return hits


def _redraw_values(tp: Any, p: ColumnProfile, cols: dict[int, np.ndarray], ok: Callable[[str], bool],
                   pyr: random.Random) -> int:
    """Give every synthetic value of column ``p`` that fails ``ok`` one new fake value (all its cells)."""
    values = cols[p.index]
    miss = missing_mask(values, p.missing_policy, p.na_tokens)
    codes, uniques = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    missing_codes = set(np.unique(codes[miss]).tolist()) if miss.any() else set()
    # distinct synthetic values grouped case-insensitively, in order of first appearance
    groups: dict[str, list[int]] = {}
    for c, v in enumerate(uniques.tolist()):
        if c not in missing_codes and isinstance(v, str) and v.strip():
            groups.setdefault(v.strip().casefold(), []).append(c)
    bad = [k for k, cs in groups.items() if not ok(uniques[cs[0]].strip())]
    if not bad:
        return 0
    cells = {k: np.flatnonzero(np.isin(codes, groups[k])).tolist() for k in bad}
    emodel = tp.private.entity_models.get(p.index) if p.role == "entity" else None
    real = textgen.real_casefold(p, emodel if emodel is not None else p.model)
    keep = NAME_TITLES | NAME_SUFFIXES | NAME_PARTICLES | _NAME_CONNECTORS
    avoid = name_words(r for r in (p.model.nonmissing_counts if p.model else {}) if isinstance(r, str)) - keep
    used = set(groups)
    sub = p.subtype or "full"
    def draw(rep: str) -> str:
        base = ""
        for a in range(4 * textgen.MAX_ATTEMPTS):
            cand = fake_name(rep, sub, a, pyr, avoid)
            if _fresh(cand, real, used) and ok(cand):
                base = cand
                break
        while not base:  # a longer blend: never a real value, never a real pair
            cand = textgen.lengthen(cand, pyr)
            if _fresh(cand, real, used) and ok(cand):
                base = cand
        used.add(base.strip().casefold())
        return base

    for k in bad:
        rows = cells[k]
        rep = textgen.split_ws(values[rows[0]])[1]
        base = draw(rep)
        # the value's case forms (McDonald / MCDONALD / McDONALD) stay apart (R2-14)
        styled = textgen.style_variants(base, [textgen.split_ws(values[r])[1] for r in rows], rep,
                                        lambda rep=rep: draw(rep))
        for r in rows:
            lead, core, trail = textgen.split_ws(values[r])
            values[r] = lead + styled[core] + trail
    return len(bad)


def _fresh(cand: str, real: set[str], used: set[str]) -> bool:
    """No real value of the column, no other synthetic value, no real phone/SSN of the run, and no real text of any
    column of the run that the run never writes (``privacy.texts``, R7-01/R7-02)."""
    key = cand.strip().casefold()
    return (bool(key) and key not in real and key not in used and not textgen.real_digits(cand)
            and not textgen.real_text(cand))


# --------------------------------------------------------------------------- CITY
#: 'City, ST' / 'City, ST 99999' values: the state code is kept (states are kept anyway), digits randomised.
_CITY_STATE_RE = re.compile(r"(?P<city>.*?\S)(?P<sep>\s*,\s*)(?P<state>[A-Z]{2})(?P<zip>\s+\d{5}(?:-\d{4})?)?")


def _make_city(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
               pyr: random.Random) -> Callable[[Group, int], str]:
    def make(g: Group, a: int) -> str:
        m = _CITY_STATE_RE.fullmatch(g.rep)
        if m and m.group("state").lower() in US_STATE_ABBR:
            tail = m.group("sep") + m.group("state") + "".join(
                pyr.choice(textgen._DIGITS) if c.isdigit() else c for c in (m.group("zip") or ""))
            return city(m.group("city"), a) + tail
        return city(g.rep, a)

    def city(rep: str, a: int) -> str:
        want_words, want_len = len(rep.split()), len(rep)
        # the closest of a few Faker cities: same number of words first, then the nearest length
        cand = min((fk.city() for _ in range(8)),
                   key=lambda c: (len(c.split()) != want_words, abs(len(c) - want_len)))
        if a >= 20:
            cand = name_word("last", max(4, len(rep)), a, pyr) + fk.city_suffix()
        st = case_style(rep)
        if st == "upper":
            return cand.upper()
        if st == "lower":
            return cand.lower()
        return cand

    return make


# --------------------------------------------------------------------------- STREET_ADDRESS
def _make_address(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
                  pyr: random.Random) -> Callable[[Group, int], str]:
    nums: list[float] = []
    weights: list[int] = []
    for g in groups:
        m = ADDRESS_RE.fullmatch(g.rep)
        if m:
            nums.append(float(m.group("num")))
            weights.append(g.count)
    if nums:
        order = np.argsort(nums, kind="stable")
        xs = np.asarray(nums)[order]
        cw = np.cumsum(np.asarray(weights, dtype=float)[order])
        cdf = (cw - 0.5 * np.asarray(weights, dtype=float)[order]) / cw[-1]
    keep_words = DIRECTIONALS | UNIT_WORDS | frozenset(STREET_SUFFIXES) | NAME_SUFFIXES

    def keep(tok: str, prev: str) -> bool:
        low = tok.casefold()
        if prev[:1].isdigit() and low in _ORDINALS:
            return True
        if tok.isupper() and len(tok) == 2 and low in US_STATE_ABBR:
            return True
        return low in keep_words

    def word(length: int, run: str) -> str:
        if length <= 2:
            return "".join(pyr.choices(textgen._ASCII_LOWER, k=length))
        return name_word("last", length, 0, pyr)

    def house_number(template: str) -> str:
        if not nums:
            return random_digits(template, pyr)
        u = pyr.random()
        v = int(round(float(np.interp(u, cdf, xs))))
        return str(max(v, 1))

    def make(g: Group, a: int) -> str:
        m = ADDRESS_RE.fullmatch(g.rep)
        if not m:
            return mimic_words(g.rep, pyr, word, keep)
        numsfx = m.group("numsfx")
        if numsfx:
            numsfx = pyr.choice(textgen._ASCII_UPPER if numsfx.isupper() else textgen._ASCII_LOWER)
        body = mimic_words(m.group("body"), pyr, word, keep)
        rest = mimic_words(m.group("rest"), pyr, word, keep)
        return (house_number(m.group("num")) + numsfx + m.group("sp1") + body + m.group("sp2") + m.group("suffix")
                + m.group("dot") + rest)

    return make


# --------------------------------------------------------------------------- EMAIL
def reserved_domain(rank: int) -> str:
    """The rank-th reserved domain: example.org, example.com, example.net, d4.example.org, ..."""
    if rank < len(RESERVED_DOMAINS):
        return RESERVED_DOMAINS[rank]
    return f"d{rank + 1}.{RESERVED_DOMAINS[rank % 3]}"


def _rank_map(counts: dict[str, int], make: Callable[[int], str]) -> dict[str, str]:
    order = sorted(counts, key=lambda k: -counts[k])  # stable: ties keep first appearance
    return {k: make(i) for i, k in enumerate(order)}


def _make_email(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
                pyr: random.Random) -> Callable[[Group, int], str]:
    domains: dict[str, int] = {}
    for g in groups:
        local, at, domain = g.rep.rpartition("@")
        if at:
            domains[domain.casefold()] = domains.get(domain.casefold(), 0) + g.count
    dmap = _rank_map(domains, reserved_domain)

    def make(g: Group, a: int) -> str:
        local, at, domain = g.rep.rpartition("@")
        if not at:
            return mask_fill(g.rep, pyr)
        runs = 0

        def word(length: int, run: str) -> str:
            nonlocal runs
            runs += 1
            if length == 1:
                return pyr.choice(textgen._ASCII_LOWER)
            return name_word("first" if runs == 1 else "last", length, a, pyr)

        new_local = mimic_words(local, pyr, word)
        new_domain = dmap[domain.casefold()]
        if case_style(domain) == "upper":
            new_domain = new_domain.upper()
        return new_local + "@" + new_domain

    return make


# --------------------------------------------------------------------------- PHONE
@lru_cache(maxsize=4096)
def phone_parts(value: str) -> tuple[str, str, str] | None:
    """Bounded generator-local memo, including extensions and float-zero suffixes."""
    return phone_privacy.phone_parts(value)


def _national(digits: str) -> str | None:
    if len(digits) == 11 and digits[0] == "1":
        return digits[1:]
    if len(digits) == 10 and digits[0] in "23456789":
        return digits
    return None


_PHONE_EXCHANGES = tuple(str(n) for n in range(200, 1000) if str(n) not in _N11)
_EXCHANGE_INDEX = {v: i for i, v in enumerate(_PHONE_EXCHANGES)}


class _PhoneSpace:
    """Uniform sparse shuffle of the exact free base × extension space of one phone shape.

    Source bases are removed before sampling, so a new extension cannot reuse a
    real line. The shuffle is column-scoped and stores only its consumed slots.
    """

    def __init__(self, digits: str, area: str | None, extension_digits: int,
                 blocked: set[str]) -> None:
        self.digits, self.area = digits, area
        self.nanp = area is not None or len(digits) == 7
        self.extension_digits = extension_digits
        self.extensions = 10 ** extension_digits
        self.start = 0 if digits.startswith("0") else 10 ** (len(digits) - 1)
        self.width = len(digits) - (1 if digits.startswith("0") else 0)
        size = len(_PHONE_EXCHANGES) * 10_000 if self.nanp else 10 ** self.width - self.start
        excluded: set[int] = set()
        for key in blocked:
            d: str | None = None
            if area is not None:
                if len(key) == 10 and key.startswith(area):
                    d = key[3:]
                elif key.startswith("L") and len(key) == 8:
                    d = key[1:]
            elif len(digits) == 7:
                if len(key) == 7 and key.isascii() and key.isdigit():
                    d = key
                elif key.startswith("S") and len(key) == 8:
                    d = key[1:]
            elif key.isascii() and key.isdigit():
                if len(key) == len(digits):
                    d = key
                elif len(digits) == 11 and len(key) == 10:
                    d = "1" + key  # country-code-1 alias of a national source base
            if d is None:
                continue
            if self.nanp:
                exchange = _EXCHANGE_INDEX.get(d[:3])
                if exchange is None:
                    continue
                slot = exchange * 10_000 + int(d[3:])
            else:
                slot = int(d) - self.start
                if digits.startswith("0") and not d.startswith("0"):
                    continue
            if 0 <= slot < size:
                excluded.add(slot)
        self.blocked = sorted(excluded)
        self.free = size - len(self.blocked)
        self.remaining = self.free * self.extensions
        self.swaps: dict[int, int] = {}

    def draw(self, pyr: random.Random) -> tuple[str, str]:
        if not self.remaining:
            raise GenerationError("phone number format has no unused collision-free values; "
                                  "all source bases are excluded, including their extensions")
        pick = pyr.randrange(self.remaining)
        self.remaining -= 1
        slot = self.swaps.pop(pick, pick)
        last = self.swaps.pop(self.remaining, self.remaining)
        if pick != self.remaining:
            self.swaps[pick] = last
        rank, extension = divmod(slot, self.extensions)
        lo, hi = rank, rank + len(self.blocked)
        while lo < hi:  # select the rank-th integer outside the sorted blocked bases
            mid = (lo + hi) // 2
            if mid - bisect_right(self.blocked, mid) >= rank:
                hi = mid
            else:
                lo = mid + 1
        if self.nanp:
            exchange, line = divmod(lo, 10_000)
            number = _PHONE_EXCHANGES[exchange] + f"{line:04d}"
            if self.area is not None:
                number = ("1" if len(self.digits) == 11 else "") + self.area + number
        else:
            number = str(lo + self.start).zfill(len(self.digits))
        return number, str(extension).zfill(self.extension_digits) if self.extension_digits else ""


def _make_phone(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
                pyr: random.Random) -> Callable[[Group, int], str]:
    k = _min_cell(profile)
    areas: dict[str, int] = {}
    parsed: dict[str, tuple | None] = {}
    spaces: dict[tuple, _PhoneSpace] = {}
    blocked = {key for g in groups for key in phone_privacy.semantic_phone_keys(g.rep, known=True, source=True)}
    active = phone_privacy.active()
    if active is not None:
        blocked.update(active.phone_bases)
        blocked.update(active.phones)
        blocked.update(active.ssns)

    def template(rep: str) -> tuple | None:
        if rep not in parsed:
            parts = phone_parts(rep)
            if parts is None:
                if phone_privacy.phone_numeric_main(rep) is not None:
                    raise GenerationError("declared PHONE value has an unsupported numeric layout")
                parsed[rep] = None
            else:
                main, ext, zeros = parts
                digits = phone_privacy.digits_of(phone_privacy.phone_text(main))
                base = phone_privacy.phone_base(rep)
                nat = _national(digits) if base is not None and base[2] else None
                parsed[rep] = (main, ext, zeros, digits, nat,
                               tuple(c.isdigit() for c in main), tuple(c.isdigit() for c in ext))
        return parsed[rep]

    for g in groups:
        info = template(g.rep)
        if info is None:
            continue
        nat = info[4]
        if nat:
            areas[nat[:3]] = areas.get(nat[:3], 0) + g.count
    real_areas = set(areas)
    kept = {a: c for a, c in areas.items() if c >= k}
    amap = {a: a for a in kept}
    rare = [a for a in areas if areas[a] < k]
    kept_codes, kept_weights = list(kept), list(kept.values())
    if not kept:  # no area code is held by >= k rows: invented codes, one per rare code, none a real one
        pool = [f"{x}" for x in range(200, 1000) if f"{x}" not in real_areas and f"{x}" not in _N11]
        pyr.shuffle(pool)
        for i, a in enumerate(rare):
            amap[a] = pool[i % len(pool)] if pool else "555"

    def area_code(code: str) -> str:
        # design 3 #27 / section 10: the area code comes from the k-floored observed pmf (codes held by
        # >= k rows); a value with a rarer code draws one of those in proportion to their counts
        if code in amap:
            return amap[code]
        return pyr.choices(kept_codes, weights=kept_weights)[0]

    def make(g: Group, a: int) -> str:
        info = template(g.rep)
        if info is None:
            return mask_fill(g.rep, pyr)
        main, ext, zeros, digits, nat, slots, ext_slots = info
        area = area_code(nat[:3]) if nat is not None else None
        shape = tuple("9" if c.isdigit() else c for c in main + ext + zeros)
        key = (shape, area, digits.startswith("0"), len(digits))
        space = spaces.get(key)
        if space is None:
            space = spaces[key] = _PhoneSpace(digits, area, sum(ext_slots), blocked)
        while True:
            new, extension = space.draw(pyr)
            it, eit = iter(new), iter(extension)
            main2 = "".join(next(it) if slot else c for slot, c in zip(slots, main))
            ext2 = "".join(next(eit) if slot else c for slot, c in zip(ext_slots, ext))
            candidate = main2 + zeros + ext2
            if not phone_privacy.holds_phone(candidate):
                return candidate

    return make


# --------------------------------------------------------------------------- SSN
_SSN_SLOT = re.compile(r"[\dXx*#]")


def _make_ssn(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
              pyr: random.Random) -> Callable[[Group, int], str]:
    def make(g: Group, a: int) -> str:
        slots = [i for i, c in enumerate(g.rep) if _SSN_SLOT.fullmatch(c)]
        if len(slots) != 9:
            return mask_fill(g.rep, pyr)
        new = list(g.rep)
        area = "9" + "".join(pyr.choices(textgen._DIGITS, k=2))
        group = f"{pyr.randint(1, 99):02d}"
        serial = f"{pyr.randint(1, 9999):04d}"
        for pos, d in zip(slots, area + group + serial):
            if g.rep[pos].isdigit():
                new[pos] = d
        return "".join(new)

    return make


# --------------------------------------------------------------------------- IP / URL
def _doc_ipv4() -> Any:
    """Documentation then benchmarking IPv4 addresses (never routed to real hosts)."""
    for net in ("192.0.2.", "198.51.100.", "203.0.113."):
        for h in range(1, 255):
            yield f"{net}{h}"
    for third in range(0, 512):
        for h in range(1, 255):
            yield f"198.{18 + third // 256}.{third % 256}.{h}"
    x = 0
    while True:  # 240.0.0.0/4 (reserved)
        yield f"{240 + (x >> 24) % 16}.{(x >> 16) & 255}.{(x >> 8) & 255}.{x & 255}"
        x += 1


def _make_ip(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
             pyr: random.Random) -> Callable[[Group, int], str]:
    first = [f"{net}{h}" for net in ("192.0.2.", "198.51.100.", "203.0.113.") for h in range(1, 255)]
    pyr.shuffle(first)
    gen = _doc_ipv4()
    for _ in range(len(first)):
        next(gen)
    pool = iter(first)

    def next_v4() -> str:
        try:
            return next(pool)
        except StopIteration:
            return next(gen)

    def make(g: Group, a: int) -> str:
        rep = g.rep
        if ":" in rep:
            try:
                ipaddress.IPv6Address(rep)
            except ValueError:
                return mask_fill(rep, pyr)
            groups6 = [f"{pyr.randint(0, 0xFFFF):x}" for _ in range(3)]
            addr = str(ipaddress.IPv6Address(f"2001:db8:{groups6[0]}:{groups6[1]}::{groups6[2]}"))
            return addr.upper() if case_style(rep) == "upper" else addr
        try:
            ipaddress.IPv4Address(rep)
        except ValueError:
            return mask_fill(rep, pyr)
        return next_v4()

    return make


def reserved_host(rank: int) -> str:
    if rank < len(RESERVED_DOMAINS):
        return RESERVED_DOMAINS[rank]
    return f"h{rank + 1}.{RESERVED_DOMAINS[rank % 3]}"


def _make_url(profile: ColumnProfile, model: ColumnModel, groups: list[Group], fk: Any,
              pyr: random.Random) -> Callable[[Group, int], str]:
    hosts: dict[str, int] = {}
    for g in groups:
        m = URL_PARSE_RE.fullmatch(g.rep)
        if m:
            h = m.group("host").casefold()
            h = h[4:] if h.startswith("www.") else h
            hosts[h] = hosts.get(h, 0) + g.count
    hmap = _rank_map(hosts, reserved_host)

    def make(g: Group, a: int) -> str:
        m = URL_PARSE_RE.fullmatch(g.rep)
        if not m:
            return mask_fill(g.rep, pyr)
        host = m.group("host")
        www = host[:4] if host.casefold().startswith("www.") else ""
        key = host.casefold()[len(www):]
        new_host = www + hmap.get(key, RESERVED_DOMAINS[0])
        if case_style(host) == "upper":
            new_host = new_host.upper()
        return (m.group("scheme") or "") + new_host + (m.group("port") or "") + mask_fill(m.group("rest"), pyr)

    return make


_MAKERS: dict[ColumnType, Callable[..., Callable[[Group, int], str]]] = {
    T.PERSON_NAME: _make_name,
    T.CITY: _make_city,
    T.STREET_ADDRESS: _make_address,
    T.EMAIL: _make_email,
    T.PHONE: _make_phone,
    T.SSN: _make_ssn,
    T.IP: _make_ip,
    T.URL: _make_url,
}


# --------------------------------------------------------------------------- STATE / ZIP
def _state(profile: ColumnProfile, model: ColumnModel, n: int,
           rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """Real US state codes/names by exact quota; any other value (``--pii-col COL=state`` or
    ``--force-type COL=STATE`` on other text) becomes a same-shape mask that is never a real value."""
    levels = [raw for raw in model.nonmissing_counts if isinstance(raw, str)]
    if not levels:
        raise GenerationError(f"column {profile.name!r}: no real values to model the column on")
    counts = [model.nonmissing_counts[raw] for raw in levels]
    labels = list(levels)
    drawer: Drawer | None = None
    keep = [is_state(raw) or raw.strip() in profile.na_tokens for raw in levels]  # NA tokens stay verbatim
    if not all(keep):
        pyr = textgen.py_random(rng)
        drawer = Drawer(textgen.real_casefold(profile, model))
        masks: dict[str, tuple[str, str]] = {}  # case-folded value -> (mask, the text it was drawn for)
        forms: dict[str, list[str]] = {}  # case-folded value -> its raw case forms
        for i, raw in enumerate(levels):
            if keep[i]:
                continue
            core = textgen.split_ws(raw)[1]
            key = core.casefold()
            if key not in masks:
                masks[key] = (drawer.draw(lambda a, core=core: mask_fill(core, pyr), pyr), core)
            forms.setdefault(key, []).append(core)
        # several case forms of one value stay apart (R2-14)
        styled = {key: textgen.style_variants(mask, forms[key], rep,
                                              lambda rep=rep: drawer.draw(lambda a: mask_fill(rep, pyr), pyr))
                  for key, (mask, rep) in masks.items()}
        for i, raw in enumerate(levels):
            if keep[i]:
                continue
            lead, core, trail = textgen.split_ws(raw)
            labels[i] = lead + styled[core.casefold()][core] + trail
    values, adjusted = quota_sample(labels, counts, n, rng, floor=len(levels) <= n)
    flags = {"rare_category_adjusted": adjusted}
    if drawer is not None and drawer.adjusted:
        flags["never_copy_adjusted"] = True
    return values, flags


def _zip_map(profile: ColumnProfile, z5_counts: dict[str, int], k: int, pyr: random.Random) -> dict[str, str]:
    """Real ZIP5 -> synthetic ZIP5 (small-cell policy).

    Codes held by >= k rows (k distinct entities when the model counts them) map to themselves, and only when at
    least k holders of the table hold them (K-08, ``detect.text.ZIP_HOLDERS_KEY``: distinct people, counted as D35
    counts them; one student's code on 12 rows is one holder). Each rarer code maps to its own invented code of the
    same width that is no real code of the column (random draws; once they start to collide, the remaining free
    codes are enumerated). When the width has fewer free codes than rare codes (a dense national file), the
    free codes are reused (the distinct count drops; never a real rare code); when every code of the width is
    real, rare codes are pooled into the kept codes in proportion to their counts.
    """
    real_z5 = set(z5_counts)
    holders = (profile.params or {}).get(ZIP_HOLDERS_KEY)
    kept = {z: c for z, c in z5_counts.items() if c >= k and (holders is None or int(holders.get(z, 0)) >= k)}
    zmap = {z: z for z in kept}
    used: set[str] = set(kept)
    free: dict[int, list[str]] = {}  # width -> shuffled free codes not used yet (built on demand)
    reuse: dict[int, list[str]] = {}  # width -> every free code (when they ran out)
    for z in z5_counts:
        if z in zmap:
            continue
        w = len(z)
        cand = None
        if w not in free:
            for _ in range(50):
                c = "".join(pyr.choices(textgen._DIGITS, k=w))
                if c not in real_z5 and c not in used:
                    cand = c
                    break
            if cand is None:
                pool = [c for c in (f"{i:0{w}d}" for i in range(10 ** w)) if c not in real_z5]
                reuse[w] = list(pool)
                free[w] = [c for c in pool if c not in used]
                pyr.shuffle(free[w])
        if cand is None and free.get(w):
            cand = free[w].pop()
        if cand is None and reuse.get(w):
            cand = pyr.choice(reuse[w])
        if cand is None and kept:
            cand = pyr.choices(list(kept), weights=list(kept.values()))[0]
        if cand is None:
            raise GenerationError(
                f"column {profile.name!r}: every {w}-digit ZIP code occurs in the column and none is held by "
                f">= {k} rows, so no code can be invented",
                fix=f"--null {profile.name} (all missing) or --exclude {profile.name}")
        zmap[z] = cand
        used.add(cand)
    return zmap


def _zip(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
         pyr: random.Random) -> tuple[np.ndarray, dict[str, bool]]:
    k = _min_cell(profile)
    real_full = textgen.real_casefold(profile, model)
    parsed: list[tuple[str, int, Any]] = []
    z5_counts: dict[str, int] = {}
    for raw, c in model.nonmissing_counts.items():
        if not isinstance(raw, str):
            continue
        m = ZIP_HINTED_RE.fullmatch(raw.strip())
        parsed.append((raw, int(c), m))
        if m:
            z5_counts[m.group("zip5")] = z5_counts.get(m.group("zip5"), 0) + int(c)
    if not parsed:
        raise GenerationError(f"column {profile.name!r}: no real values to model the column on")
    zmap = _zip_map(profile, z5_counts, k, pyr)
    drawer = Drawer(real_full)
    labels: list[tuple[str, str, str, str] | str] = []
    counts: list[int] = []
    for raw, c, m in parsed:
        lead, core, trail = textgen.split_ws(raw)
        if m is None:
            labels.append(lead + drawer.draw(lambda a, core=core: mask_fill(core, pyr), pyr) + trail)
        elif m.group("plus4"):
            labels.append((lead, zmap[m.group("zip5")], m.group("sep") or "", trail))
        else:
            labels.append(lead + zmap[m.group("zip5")] + trail)
        counts.append(c)
    idx, adjusted = quota_sample(list(range(len(labels))), counts, n, rng, floor=len(labels) <= n)
    out = np.empty(n, dtype=object)
    for r, i in enumerate(idx.tolist()):
        lab = labels[i]
        if isinstance(lab, str):
            out[r] = lab
            continue
        lead, z5, sep, trail = lab
        for _ in range(100):
            cand = z5 + sep + "".join(pyr.choices(textgen._DIGITS, k=4))
            if cand.casefold() not in real_full:
                break
        out[r] = lead + cand + trail
    flags = {"rare_category_adjusted": bool(adjusted)}
    if drawer.adjusted:
        flags["never_copy_adjusted"] = True
    return out, flags
