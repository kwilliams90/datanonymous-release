"""Non-factor text generation (AREA C): design D12, D13, D24, section 3 #8, #32, #34, #35.

Every never-copy text generator works **per real distinct value** (grouped by
stripped, case-folded text): each group gets one synthetic value built from the
group's shape, the group's raw variants keep their case style and surrounding
whitespace, and rows are allocated by exact quota over the real counts. Raw
case forms that share a case style (``aLiCe``, ``alICe``, ``AliCE``) still get
distinct renderings (:func:`style_variants`, R2-14). So at the same N the
distinct count, the frequency profile (singletons, skew) and the
case/whitespace variants are kept exactly. :func:`per_distinct` is that
framework (also used by ``generate/pii.py``); a synthetic value never equals a
real value of the column (case-insensitive, trimmed) and never repeats another
group's value. When the shape cannot satisfy that (e.g. every single-letter
initial is already a real value), the value is lengthened and the flag
``never_copy_adjusted`` is set.

* :func:`invented_labels` - HIGH_CARD_CATEGORICAL (and the PII fallback):
  labels ``<column>_<k>`` numbered in a seeded random order, of the real
  value's length (a shorter prefix, or a zero-padded number: length profile
  kept, D13); subtype ``code``: same-shape character-class masks
  (``AB-1234`` -> ``QX-0627``).
* :func:`lorem` - FREE_TEXT: the real value's skeleton is kept (length, word
  lengths, spaces, newlines, punctuation, case per word); letter runs become
  lorem words of the same length (Faker's Latin word list, D24) carrying the
  run's accented letters on the same base letter, digit runs random digits. So
  char/word length distributions and the newline/digit/punctuation/non-ASCII
  shares are kept; no real word survives. E-mail addresses, URLs and web
  domains, IP addresses, SSNs, phone numbers and street addresses inside a
  value are synthesised by their PII generators' rules instead (reserved
  domains, fresh phone bases, area 9xx ...; ``generate.embedded``, R4-07), one
  synthetic span per real span of the column.
* :func:`masked_codes` - DIGIT_CODE above factor-max: same width and leading
  zeros as a real code (templates by exact quota), never a real code; when every
  code of a shape is real, the column's next shape with free codes (else one
  digit wider) is used instead of failing.
* :func:`generate_text_extended` - MULTI_SELECT: combinations as a factor
  (exact quota over the raw combinations) when there are <= factor-max of them,
  else item-wise: items per value by exact quota, item totals by exact quota
  (floor 1 per item), items drawn per row in proportion to what is left and
  written in the observed order (experimental); an item's raw case forms
  (``ELL`` / ``ell``), the separator forms (``;`` vs
  ``; ``) and trailing separators are kept by exact quota; whole-value NA tokens
  ('None') and item-less values stay whole values at their exact counts. Items
  the D35 allowlist did not keep are replaced by their same-shape pseudonyms
  before this generator runs (``generate.pseudonyms``, item by item inside each
  value), so no such real item is written.
* numbers masked under a personal-data header (``params['_numeric_mask']``)
  keep their digits' shape; a leftover text token of such a column has its
  letters masked too.
* :func:`mask_fill` / :func:`digit_mask` - same-shape masks; a leading digit
  keeps its kind (zero padding kept, a non-zero first digit stays non-zero).
  They, :func:`mimic_words` and code masks never rebuild a personal-data shape
  character by character: an e-mail address keeps a reserved domain, a phone
  number gets fresh digits excluding source bases ... (``generate.embedded``); and every candidate a
  :class:`Drawer` accepts has passed ``embedded.guard`` (R4-07).

Random draws inside one column use a ``random.Random`` seeded from the
column's numpy stream (deterministic, fast per character).
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

from ..detect.text import case_style
from ..errors import GenerationError, UnsupportedColumnError
from ..model import ColumnModel, ColumnProfile, ColumnType
from ..privacy.digits import holds_real as _real_digits
from ..privacy.texts import holds_real as _real_text
from . import embedded
from .quota import allocate, quota_sample

LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore "
    "magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo "
    "consequat duis aute irure in reprehenderit voluptate velit esse cillum fugiat nulla pariatur excepteur sint "
    "occaecat cupidatat non proident sunt culpa qui officia deserunt mollit anim id est laborum"
).split()

_LABEL_CHARS = re.compile(r"[^0-9A-Za-z]+")
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|.", re.S)
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ASCII_UPPER = _ASCII_LOWER.upper()
_DIGITS = "0123456789"
_DROP_DIGITS = str.maketrans("", "", _DIGITS)
_DROP_NONLETTERS = str.maketrans("", "", "".join(chr(i) for i in range(128) if not chr(i).isalpha()))
MAX_ATTEMPTS = 40
#: :meth:`Drawer.draw` lengthens a candidate this many times before it puts a letter in front instead (K-21).
MAX_GROWTH = 8


def label_prefix(name: str) -> str:
    p = _LABEL_CHARS.sub("_", name).strip("_")
    return p or "value"


def py_random(rng: np.random.Generator) -> random.Random:
    """A ``random.Random`` seeded from the column's numpy stream (fast scalar draws)."""
    return random.Random(int(rng.integers(0, 2**63 - 1)))


@lru_cache(maxsize=1)
def lorem_words() -> tuple[str, ...]:
    """Faker's Latin lorem word list (D24), or the bundled list if Faker is unavailable."""
    try:
        from faker.providers.lorem.la import Provider

        words = tuple(w for w in Provider.word_list if w.isalpha() and w.isascii())
        if words:
            return words
    except Exception:  # pragma: no cover - Faker is a pinned runtime dependency
        pass
    return tuple(LOREM_WORDS)


@lru_cache(maxsize=1)
def _lorem_buckets() -> dict[int, tuple[str, ...]]:
    out: dict[int, list[str]] = {}
    for w in lorem_words():
        out.setdefault(len(w), []).append(w.lower())
    return {k: tuple(v) for k, v in out.items()}


def lorem_word(length: int, pyr: random.Random, letters: list[str] | None = None) -> str:
    """A lower-case lorem word of exactly ``length`` letters (long ones are joined and cut).

    ``letters``: prefer a word containing these base letters (to carry accents).
    """
    buckets = _lorem_buckets()
    if length in buckets:
        bucket = buckets[length]
        if letters:
            for _ in range(8):
                w = pyr.choice(bucket)
                if all(b in w for b in letters):
                    return w
        return pyr.choice(bucket)
    words = lorem_words()
    s = ""
    while len(s) < length:
        s += pyr.choice(words).lower()
    return s[:length]


# --------------------------------------------------------------------------- shape helpers
def split_ws(raw: str) -> tuple[str, str, str]:
    """(leading whitespace, stripped core, trailing whitespace)."""
    core = raw.strip()
    if not core:
        return raw, "", ""
    i = raw.index(core)
    return raw[:i], core, raw[i + len(core):]


def base_letter(ch: str) -> str:
    """ASCII base letter of an accented letter (``'\xe9'`` -> ``'e'``), or ``''``."""
    b = unicodedata.normalize("NFD", ch)[:1]
    return b.lower() if b.isascii() and b.isalpha() else ""


def accent_bases(template: str) -> list[str]:
    """Base letters of the template's non-ASCII letters (used to prefer words that can carry them)."""
    return [b for b in (base_letter(ch) for ch in template if not ch.isascii() and ch.isalpha()) if b]


def style_like(word: str, template: str) -> str:
    """``word`` in the case style of ``template`` (a letter run), carrying the template's non-ASCII letters.

    Each non-ASCII letter of the template replaces the same base letter in the
    word (``'\xe9'`` replaces an ``'e'`` of the same case, the one nearest the
    template's position), so the accent share and the source encoding are kept
    without odd spellings; a letter with no such base position is dropped.
    """
    if not word:
        return word
    if len(template) == 1:
        w = word.upper() if template.isupper() else word.lower()
    else:
        st = case_style(template)
        if st == "upper":
            w = word.upper()
        elif st == "lower":
            w = word.lower()
        else:
            w = word[:1].upper() + word[1:].lower()
    if not template.isascii():
        chars = list(w)
        taken: set[int] = set()
        for i, ch in enumerate(template):
            if ch.isascii() or not ch.isalpha():
                continue
            b = base_letter(ch)
            if not b:
                continue
            want = b.upper() if ch.isupper() else b
            spots = [j for j, c in enumerate(chars) if c == want and j not in taken]
            if spots:
                j = min(spots, key=lambda s: (abs(s - i), s))
                chars[j] = ch
                taken.add(j)
        w = "".join(chars)
    return w


def restyle(base: str, core: str, rep: str) -> str:
    """Render a group's synthetic ``base`` for one raw variant ``core`` of representative ``rep``.

    Only for a value with a single raw case form; several case forms of one
    value go through :func:`style_variants`, which keeps them apart (R2-14).
    """
    if core == rep:
        return base
    st = case_style(core)
    if st == case_style(rep):
        return base
    if st == "upper":
        return base.upper()
    if st == "lower":
        return base.lower()
    if st == "title":
        return base.title()
    return base


_ALPHA_RUN = re.compile(r"[^\W\d_]+")


def transfer_case(base: str, core: str) -> str:
    """``base`` with the per-letter case pattern of ``core``.

    Letter runs are paired in order when both have the same number of runs
    (the lorem/name/mask generators keep a value's runs), else all letters are
    paired in order; a base letter past the end of its core run takes the case
    of the run's last letter, and uncased core letters leave the base letter as
    it is.
    """
    b_pos = [list(range(*m.span())) for m in _ALPHA_RUN.finditer(base)]
    c_runs = [m.group() for m in _ALPHA_RUN.finditer(core)]
    if not b_pos or not c_runs:
        return base
    if len(b_pos) != len(c_runs):
        b_pos = [[i for run in b_pos for i in run]]
        c_runs = ["".join(c_runs)]
    chars = list(base)
    for pos, run in zip(b_pos, c_runs):
        for j, i in enumerate(pos):
            src = run[min(j, len(run) - 1)]
            ch = chars[i]
            new = ch.upper() if src.isupper() else ch.lower() if src.islower() else ch
            if len(new) == 1:
                chars[i] = new
    return "".join(chars)


def case_render(base: str, core: str) -> str:
    """``base`` in the case style of the raw form ``core``: upper / lower / title, or (mixed) its letter pattern."""
    st = case_style(core)
    if st == "upper":
        return base.upper()
    if st == "lower":
        return base.lower()
    if st == "title":
        return base.title()
    if st == "mixed":
        return transfer_case(base, core)
    return base


def _case_flips(text: str) -> Iterable[str]:
    """Other case patterns of ``text`` (letters flipped from the end; same letters, same case-folded text)."""
    pos = [i for i, ch in enumerate(text)
           if ch.upper() != ch.lower() and len(ch.upper()) == 1 and len(ch.lower()) == 1][::-1][:20]
    for t in range(1, 2 ** len(pos)):
        chars = list(text)
        for b, i in enumerate(pos):
            if t >> b & 1:
                chars[i] = chars[i].lower() if chars[i].isupper() else chars[i].upper()
        yield "".join(chars)


def style_variants(base: str, cores: Iterable[str], rep: str, fresh: Callable[[], str] | None) -> dict[str, str]:
    """One rendering of the synthetic ``base`` per distinct raw case form in ``cores`` of one value (R2-14).

    Distinct raw forms (``aLiCe``, ``alICe``, ``AliCE``) always get distinct
    renderings, so the distinct count and the frequency profile are kept. With
    a single form the generator's ``base`` is written as it is. With several,
    each form gets ``base`` in its own case style (:func:`case_render`; the
    representative ``rep`` keeps ``base`` when that already has its style); a
    rendering another form already took is replaced by the nearest unused case
    pattern of it, and when ``base`` has too few cased letters for every form,
    ``fresh()`` supplies another synthetic value (never real, unique). Every
    rendering differs from ``base`` only in letter case, so it is never a real
    value when ``base`` is not (case-insensitive check).
    """
    uniq = list(dict.fromkeys(cores))
    if len(uniq) <= 1:
        return {c: restyle(base, c, rep) for c in uniq}
    order = ([rep] if rep in uniq else []) + [c for c in uniq if c != rep]
    out: dict[str, str] = {}
    used: set[str] = set()

    def recased(value: str, core: str) -> str:
        """``value`` in ``core``'s case style, or ``value`` itself if re-casing would change its folded text."""
        r = case_render(value, core)
        return r if r.casefold() == value.casefold() else value

    fold = base.casefold()
    for c in order:
        cand: str | None = base if c == rep and case_style(base) == case_style(rep) else recased(base, c)
        if cand in used:
            cand = next((alt for alt in _case_flips(cand) if alt not in used and alt.casefold() == fold), None)
        while cand is None or cand in used:  # base has too few cased letters for every form
            if fresh is None:
                raise GenerationError("cannot keep every case form of a value apart: too few letters to vary")
            cand = recased(fresh(), c)  # another synthetic value: never real, unique case-insensitively
        used.add(cand)
        out[c] = cand
    return out


def random_digits(template: str, pyr: random.Random) -> str:
    """Digits of the same length; a leading zero only where the template has one."""
    if not template:
        return ""
    if len(template) == 1:
        return pyr.choice(_DIGITS)
    first = "0" if template[0] == "0" else pyr.choice("123456789")
    return first + "".join(pyr.choices(_DIGITS, k=len(template) - 1))


def _lead_digit(template: str) -> tuple[int, bool] | None:
    """(index, zero-padded) of the digit that opens the value (after an optional sign), or None.

    zero-padded: the template opens with ``0`` followed by another digit (``012``).
    """
    i = 1 if template[:1] in ("-", "+") else 0
    if not template[i:i + 1].isdigit():
        return None
    return i, template[i] == "0" and template[i + 1:i + 2].isdigit()


def _fix_lead(digits: list[str], template: str, pyr: random.Random) -> None:
    """A leading zero only where the template has one: ``123`` keeps a non-zero first digit, ``012`` a zero."""
    lead = _lead_digit(template)
    if lead is None:
        return
    i, padded = lead
    if padded:
        digits[i] = "0"
    elif template[i] != "0" and digits[i] == "0":
        digits[i] = pyr.choice(_DIGITS[1:])


def mask_fill(template: str, pyr: random.Random) -> str:
    """Same character-class mask: digit -> random digit, letter -> random letter of the same case.

    A value that opens with a digit keeps its kind of first digit (a leading zero only where the template
    has one, so a number keeps its width and a zero-padded code its padding). A personal-data shape inside the
    template (an e-mail address, URL or web domain, IP address, SSN, phone number) is synthesised by its PII
    rules instead of masked (``generate.embedded.rewrite``, R4-07): ``ana@district.edu`` -> ``xkq@example.org``.
    """
    return embedded.rewrite(template, lambda part: _mask_plain(part, pyr), pyr)


def _mask_plain(template: str, pyr: random.Random) -> str:
    n = len(template)
    digits = pyr.choices(_DIGITS, k=n)
    lower = pyr.choices(_ASCII_LOWER, k=n)
    _fix_lead(digits, template, pyr)
    out = []
    for i, ch in enumerate(template):
        if ch.isdigit():
            out.append(digits[i])
        elif ch.isalpha():
            out.append(lower[i].upper() if ch.isupper() else lower[i])
        else:
            out.append(ch)
    return "".join(out)


def digit_mask(template: str, pyr: random.Random) -> str:
    """Only the digits are randomised (``-12.50`` -> ``-83.07``, ``1e-05`` -> ``7e-03``): numbers stay numbers.

    A whole 10/11-digit number is a phone number here (numbers under a personal-data header): it follows the
    PHONE rules (fresh numbers excluding source bases; ``generate.embedded``, R4-07).
    """
    return embedded.rewrite(template, lambda part: _digit_mask_plain(part, pyr), pyr, digit_phone=True)


def _digit_mask_plain(template: str, pyr: random.Random) -> str:
    digits = [pyr.choice(_DIGITS) if ch.isdigit() else ch for ch in template]
    _fix_lead(digits, template, pyr)
    return "".join(digits)


def mimic_words(template: str, pyr: random.Random, word_for: Callable[[int, str], str],
                keep: Callable[[str, str], bool] | None = None) -> str:
    """Rebuild ``template`` token by token: letter runs -> ``word_for(length, run)`` styled like the run,
    digit runs -> random digits, every other character kept. ``keep(token, previous_token)`` keeps a
    letter run verbatim (vocabulary words such as street suffixes). A personal-data shape inside the template
    (e-mail address, URL or web domain, IP address, SSN, phone number) is synthesised by its PII rules instead
    (``generate.embedded.rewrite``, R4-07)."""
    return embedded.rewrite(template, lambda part: _mimic_plain(part, pyr, word_for, keep), pyr)


def _mimic_plain(template: str, pyr: random.Random, word_for: Callable[[int, str], str],
                 keep: Callable[[str, str], bool] | None = None) -> str:
    out: list[str] = []
    prev = ""
    for tok in _TOKEN_RE.findall(template):
        if tok[0].isalpha():
            if keep is not None and keep(tok, prev):
                out.append(tok)
            else:
                out.append(style_like(word_for(len(tok), tok), tok))
        elif tok[0].isdigit():
            out.append(random_digits(tok, pyr))
        else:
            out.append(tok)
        prev = tok
    return "".join(out)


def lengthen(value: str, pyr: random.Random) -> str:
    """Append one character of the last character's class (used when a shape is exhausted)."""
    last = value[-1:] if value else "a"
    if last.isdigit():
        return value + pyr.choice(_DIGITS)
    if last.isupper():
        return value + pyr.choice(_ASCII_UPPER)
    return value + pyr.choice(_ASCII_LOWER)


def _lengthen_safe(value: str, pyr: random.Random,
                  guard: Callable[[str, random.Random], str] | None = None) -> str:
    """:func:`lengthen`, unless the extra character would turn a reserved domain at the end into an unreserved
    one (``x@example.org`` -> ``x@example.orgq``): then one letter goes in front instead."""
    longer = lengthen(value, pyr)
    if (guard or embedded.guard)(longer, pyr) is longer:
        return longer
    return pyr.choice(_ASCII_LOWER) + value


# --------------------------------------------------------------------------- per-distinct framework
@dataclass
class Group:
    """One real distinct value (stripped, case-folded) with its raw variants."""

    key: str
    rep: str  # stripped text of the most common raw variant
    variants: list[tuple[str, int]]
    count: int


def real_casefold(profile: ColumnProfile, model: ColumnModel) -> set[str]:
    """Every real non-missing value of the column, stripped and case-folded (row model + this model)."""
    out: set[str] = set()
    for m in (profile.model, model):
        if m is None:
            continue
        for raw in m.nonmissing_counts:
            if isinstance(raw, str):
                out.add(raw.strip().casefold())
        if m.real_values:
            out.update(v.casefold() for v in m.real_values)
    return out


def distinct_groups(model: ColumnModel) -> list[Group]:
    """Groups of raw non-missing values by stripped, case-folded text (first-appearance order)."""
    by_key: dict[str, list[tuple[str, int]]] = {}
    for raw, c in model.nonmissing_counts.items():
        if not isinstance(raw, str) or c <= 0:
            continue
        by_key.setdefault(raw.strip().casefold(), []).append((raw, int(c)))
    groups = []
    for key, variants in by_key.items():
        best = max(variants, key=lambda rc: rc[1])  # first of the most common (stable)
        groups.append(Group(key, best[0].strip(), variants, sum(c for _, c in variants)))
    return groups


def real_digits(candidate: str) -> bool:
    """Skip digit regexes for short ASCII candidates; Unicode phone digits need canonicalization."""
    return (len(candidate) >= 7 and (not candidate.isascii()
            or len(candidate) - len(candidate.translate(_DROP_DIGITS)) >= 7)
            and _real_digits(candidate))


def real_text(candidate: str) -> bool:
    """Numeric ASCII candidates have no text key; e-mail local parts still count."""
    if candidate.isascii() and "@" not in candidate and not candidate.translate(_DROP_NONLETTERS):
        return False
    return _real_text(candidate)


class Drawer:
    """Accepts candidates that equal no real value and no earlier synthetic value (case-insensitive), hold
    no real phone number or SSN of the run (a phone/SSN-shaped token of its digits, ``privacy.digits``: random
    digits of a mimicked e-mail local part or note never equal a real phone by chance), and equal no real text of
    ANY column of the run that the run never writes (``privacy.texts``, R7-01/R7-02: a fake name is no real
    person's name of another column, a pseudonym no value of an excluded column ...).

    ``fallback`` makes the same-shape masks tried when ``make`` keeps failing (default :func:`mask_fill`).
    """

    def __init__(self, real_cf: set[str], fallback: Callable[[str, random.Random], str] | None = None,
                 *, memo_guard: bool = True, phone: bool = False) -> None:
        self.real_cf = real_cf
        self.used: set[str] = set()
        self.adjusted = False
        self.fallback = fallback or mask_fill
        self.guard = embedded.GuardMemo() if memo_guard else embedded.guard
        self.phone = phone

    def ok(self, candidate: str) -> bool:
        from ..privacy.digits import holds_phone

        key = candidate.strip().casefold()
        return (bool(key) and key not in self.real_cf and key not in self.used
                and not real_digits(candidate) and not (self.phone and holds_phone(candidate))
                and not real_text(candidate))

    def take(self, candidate: str) -> str:
        self.used.add(candidate.strip().casefold())
        return candidate

    def draw(self, make: Callable[[int], str], pyr: random.Random, attempts: int = MAX_ATTEMPTS) -> str:
        """``make(attempt)`` until a candidate is acceptable; then same-shape masks; then lengthen.

        Every candidate first passes ``embedded.guard`` (R4-07): an e-mail address or URL on an unreserved
        domain, an SSN outside area 9xx or a routable IPv4
        address in it is corrected, so no generator path writes one.
        """
        guard, ok, take = self.guard, self.ok, self.take
        cand = ""
        for a in range(attempts):
            cand = guard(make(a), pyr)
            if ok(cand):
                return take(cand)
        if self.phone:
            from ..privacy.digits import phone_base

            if phone_base(cand) is not None:
                # A phone maker has a finite shuffled format space. Exhaust it
                # explicitly instead of changing area codes or appending digits.
                while True:
                    cand = guard(make(attempts), pyr)
                    if ok(cand):
                        return take(cand)
        base = cand or "x"
        for _ in range(attempts):
            cand = guard(self.fallback(base, pyr), pyr)
            if ok(cand):
                return take(cand)
        self.adjusted = True
        cand = base
        grown = 0
        while True:  # one character longer at a time; several endings are tried at each length
            for _ in range(attempts):
                longer = _lengthen_safe(cand, pyr, guard)
                if ok(longer):
                    return take(longer)
            cand = _lengthen_safe(cand, pyr, guard)
            grown += 1
            if grown == MAX_GROWTH:
                # K-21: no ending of this shape is accepted (a token among numbers written in non-ASCII digits,
                # '１２' or '10²1', has only digit masks, and each reads as a number): a letter in front ends the loop
                cand = pyr.choice(_ASCII_LOWER) + base


def render_groups(groups: list[Group], bases: list[str],
                  fresh: Callable[[Group], str] | None = None) -> tuple[list[str], list[int]]:
    """Labels (one per raw variant, case style and whitespace kept) and their real counts.

    Distinct raw variants always get distinct labels (:func:`style_variants`;
    ``fresh(group)`` draws another synthetic value for a group whose base
    cannot carry all its case forms), so the distinct count and frequency
    profile are kept (R2-14).
    """
    labels: list[str] = []
    counts: list[int] = []
    for g, base in zip(groups, bases):
        if len(g.variants) == 1:  # one raw form (the common case)
            raw, c = g.variants[0]
            lead, core, trail = split_ws(raw)
            labels.append(lead + restyle(base, core, g.rep) + trail)
            counts.append(c)
            continue
        cores = [split_ws(raw)[1] for raw, _ in g.variants]
        styled = style_variants(base, cores, g.rep, None if fresh is None else (lambda g=g: fresh(g)))
        for raw, c in g.variants:
            lead, core, trail = split_ws(raw)
            labels.append(lead + styled[core] + trail)
            counts.append(c)
    if len(set(labels)) != len(labels):  # invariant: one synthetic label per distinct raw value (R2-14)
        raise GenerationError("internal error: two distinct source values were given the same synthetic value")
    return labels, counts


def allocate_labels(labels: list[str], counts: list[int], n: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Exact quota of ``n`` rows over ``labels`` (floor 1 per label when every label fits)."""
    if not labels:
        raise GenerationError("no real values to model the column on")
    idx, adjusted = quota_sample(list(range(len(labels))), counts, n, rng, floor=len(labels) <= n)
    lab = np.empty(len(labels), dtype=object)
    lab[:] = labels
    return lab[idx.astype(np.int64)] if len(idx) else np.empty(0, dtype=object), adjusted


def per_distinct(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator,
                 make_base: Callable[[Group, int], str], pyr: random.Random | None = None,
                 drawer: Drawer | None = None) -> tuple[np.ndarray, dict[str, bool]]:
    """One synthetic value per real distinct value (``make_base(group, attempt)``), exact quota over rows."""
    pyr = pyr or py_random(rng)
    groups = distinct_groups(model)
    if not groups:
        raise GenerationError(f"column {profile.name!r}: no real values to model the column on")
    drawer = drawer or Drawer(real_casefold(profile, model), memo_guard=profile.type is not ColumnType.EMAIL)
    bases = [drawer.draw(lambda a, g=g: make_base(g, a), pyr) for g in groups]
    labels, counts = render_groups(groups, bases, lambda g: drawer.draw(lambda a: make_base(g, a), pyr))
    values, adjusted = allocate_labels(labels, counts, n, rng)
    flags = {"rare_category_adjusted": adjusted}
    if drawer.adjusted:
        flags["never_copy_adjusted"] = True
    return values, flags


# --------------------------------------------------------------------------- HIGH_CARD_CATEGORICAL
def _length_label(prefix: str, num: str, length: int) -> str | None:
    """``<prefix>_<number>`` of exactly ``length`` characters, or None when the number does not fit.

    Longer targets zero-pad the number (``career_goal_00019``); shorter ones
    shorten the prefix (``career_19``).
    """
    if length < len(num) + 2:
        return None
    p = prefix[: length - len(num) - 1].rstrip("_") or prefix[:1]
    width = length - len(p) - 1
    if width < len(num):
        return None
    return f"{p}_{num.zfill(width)}"


def invented_labels(profile: ColumnProfile, model: ColumnModel, n: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Invented labels with the real distinct count, frequency profile and lengths; (values, adjusted).

    Subtype ``code``: same-shape character-class masks instead (D13).
    """
    values, flags = invented_labels_flags(profile, model, n, rng)
    return values, flags["rare_category_adjusted"]


def invented_labels_flags(profile: ColumnProfile, model: ColumnModel, n: int,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """:func:`invented_labels` returning the full flag dict (incl. ``never_copy_adjusted``)."""
    if n == 0:
        return np.empty(0, dtype=object), {"rare_category_adjusted": False}
    pyr = py_random(rng)
    if profile.subtype == "code":
        if (profile.params or {}).get("_numeric_mask"):  # numbers under a personal-data header (D12 guard)
            # numbers keep every non-digit (sign, point, exponent); a leftover text token of a numbers-with-tokens
            # column has its letters masked too (R3: it may name someone)
            def num_mask(template: str, r: random.Random) -> str:
                words = any(ch.isalpha() and ch not in "eE" for ch in template.replace("EUR", ""))
                return (mask_fill if words else digit_mask)(template, r)

            drawer = Drawer(real_casefold(profile, model), fallback=num_mask)
            synth = embedded.SpanSynth.build(profile, _templates(model), rng, digit_phone=True,
                                             avoid=_row_values(profile, model))
            if synth is None:
                return per_distinct(profile, model, n, rng, lambda g, a: num_mask(g.rep, pyr), pyr, drawer)
            return per_distinct(profile, model, n, rng,
                                lambda g, a: synth.rewrite(g.rep, lambda part: num_mask(part, pyr)), pyr, drawer)
        # personal-data shapes inside the codes (R3 copy gate): synthesised by their PII rules, one per real span
        synth = embedded.SpanSynth.build(profile, _templates(model), rng, avoid=_row_values(profile, model))
        if synth is None:
            return per_distinct(profile, model, n, rng, lambda g, a: mask_fill(g.rep, pyr), pyr)
        return per_distinct(profile, model, n, rng,
                            lambda g, a: synth.rewrite(g.rep, lambda part: mask_fill(part, pyr)), pyr)
    groups = distinct_groups(model)
    k = len(groups)
    if k == 0:
        return np.empty(0, dtype=object), {"rare_category_adjusted": False}
    width = len(str(k))
    prefix = label_prefix(profile.name)
    numbers = (rng.permutation(k) + 1).tolist()
    drawer = Drawer(real_casefold(profile, model))
    bases = []
    makes: dict[str, Callable[[int], str]] = {}
    for g, num in zip(groups, numbers):
        num_s = f"{num:0{width}d}"
        lab = _length_label(prefix, num_s, len(g.rep))

        def make(a: int, lab: str | None = lab, g: Group = g, num_s: str = num_s) -> str:
            if lab is not None:
                return lab if a == 0 else lab[:-len(num_s)] + mask_fill(num_s, pyr)
            return "".join(pyr.choices(_ASCII_LOWER, k=max(1, len(g.rep))))

        makes[g.key] = make
        if lab is not None and drawer.ok(lab):
            bases.append(drawer.take(lab))
        else:
            bases.append(drawer.draw(make, pyr))
    if case_style(prefix) != "upper":
        bases = [b if case_style(g.rep) != "upper" else b.upper() for b, g in zip(bases, groups)]
    # a label too short to carry every case form of its value gets another label (R2-14)
    labels, counts = render_groups(groups, bases, lambda g: drawer.draw(lambda a: makes[g.key](a + 1), pyr))
    values, adjusted = allocate_labels(labels, counts, n, rng)
    flags = {"rare_category_adjusted": adjusted}
    if drawer.adjusted:
        flags["never_copy_adjusted"] = True
    return values, flags


# --------------------------------------------------------------------------- FREE_TEXT
def lorem(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator) -> np.ndarray:
    """Lorem text keeping the distinct count, frequency profile and each value's shape (never real text).

    One string per real distinct value with the same skeleton (see module
    docstring), distinct from each other and from every real value; then exact
    quota over the real counts.
    """
    if n == 0:
        return np.empty(0, dtype=object)
    if not distinct_groups(model):
        return np.asarray(["Lorem"] * n, dtype=object)
    pyr = py_random(rng)

    def word(length: int, run: str) -> str:
        return lorem_word(length, pyr, accent_bases(run) if not run.isascii() else None)

    # R4-07: e-mail addresses, URLs, phone numbers ... inside the text follow their PII generators' rules (the
    # synth has found every span of every value: the text around them is plain lorem)
    synth = embedded.SpanSynth.build(profile, _templates(model), rng, digit_phone=True,
                                     avoid=_row_values(profile, model))
    if synth is None:
        values, _ = per_distinct(profile, model, n, rng, lambda g, a: _mimic_plain(g.rep, pyr, word), pyr)
    else:
        values, _ = per_distinct(profile, model, n, rng,
                                 lambda g, a: synth.rewrite(g.rep, lambda part: _mimic_plain(part, pyr, word)), pyr)
    return values


def _templates(model: ColumnModel) -> list[tuple[str, int]]:
    """(stripped value, rows) of each real raw value: what the generators rebuild (a group's representative is
    one of them)."""
    return [(raw.strip(), int(c)) for raw, c in model.nonmissing_counts.items() if isinstance(raw, str) and c > 0]


def _row_values(profile: ColumnProfile, model: ColumnModel) -> list[str]:
    """The row-level real values when ``model`` is a per-entity one (D4): their spans are real too."""
    row = profile.model
    if row is None or row is model:
        return []
    return [str(raw).strip() for raw in row.nonmissing_counts if isinstance(raw, str)]


# --------------------------------------------------------------------------- DIGIT_CODE masks
def masked_codes(profile: ColumnProfile, model: ColumnModel, n: int, rng: np.random.Generator) -> np.ndarray:
    """Same-shape digit codes (width and leading zeros kept), never a real code."""
    real = model.real_values or frozenset()
    templates: dict[tuple[int, int], int] = {}
    for raw, c in model.nonmissing_counts.items():
        s = raw.strip()
        lead = len(s) - len(s.lstrip("0")) if len(s) > 1 else 0
        lead = min(lead, len(s) - 1)
        key = (len(s), lead)
        templates[key] = templates.get(key, 0) + c
    keys = list(templates)
    picks, _ = quota_sample(list(range(len(keys))), [templates[k] for k in keys], n, rng, floor=False)
    out = np.empty(n, dtype=object)
    spaces: dict[tuple[int, int], list[str] | None] = {}  # template -> enumerated free codes (small spaces)
    for i, ti in enumerate(picks.tolist()):
        width, lead = keys[ti]
        s = _draw_digit_code(width, lead, real, rng, spaces)
        if s is None:  # every code of this shape is real: the next shape with free codes (never a real code)
            for w2, l2 in sorted({k for k in keys if k != (width, lead)}, key=lambda k: -templates[k]) + [
                    (width + j, lead) for j in range(1, 4)]:
                s = _draw_digit_code(w2, l2, real, rng, spaces)
                if s is not None:
                    break
        if s is None:
            raise GenerationError(
                f"column {profile.name!r}: cannot draw codes of the same shape that avoid every real code",
                fix=f"--force-factor {profile.name} keeps the real codes")
        out[i] = s
    return out


def _digit_code(width: int, lead: int, rng: np.random.Generator) -> str:
    free = width - lead
    # the first free digit is non-zero so the value keeps exactly `lead` leading zeros
    first = str(int(rng.integers(1, 10))) if width > 1 else str(int(rng.integers(0, 10)))
    rest = "".join(str(int(d)) for d in rng.integers(0, 10, size=max(free - 1, 0)))
    return "0" * lead + (first + rest if free > 0 else "")


def _draw_digit_code(width: int, lead: int, real: frozenset[str], rng: np.random.Generator,
                     spaces: dict[tuple[int, int], list[str] | None]) -> str | None:
    """A code of ``width`` digits with exactly ``lead`` leading zeros that is no real code, or None if the
    shape has none: random draws first; when they collide, the shape's free codes are enumerated (up to
    10^6 codes)."""
    key = (width, lead)
    if spaces.get(key) is None:
        for _ in range(50):
            s = _digit_code(width, lead, rng)
            if s not in real:
                return s
        free = width - lead
        size = 10 if width == 1 else (9 * 10 ** (free - 1) if free > 0 else 1)
        if size > 1_000_000:
            return None
        if width == 1:
            codes = [str(d) for d in range(10)]
        elif free <= 0:
            codes = ["0" * width]
        else:
            codes = ["0" * lead + str(v) for v in range(10 ** (free - 1), 10 ** free)]
        spaces[key] = [c for c in codes if c not in real]
    pool = spaces[key]
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


# --------------------------------------------------------------------------- MULTI_SELECT
def _infer_delimiter(model: ColumnModel) -> str | None:
    best, best_n = None, 0
    for d in (";", "|", ","):
        cnt = sum(c for raw, c in model.nonmissing_counts.items() if isinstance(raw, str) and d in raw)
        if cnt > best_n:
            best, best_n = d, cnt
    return best


def generate_text_extended(profile: ColumnProfile, model: ColumnModel, n: int,
                           rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """AREA C hook: values for MULTI_SELECT and other AREA C text types.

    Called by ``generate/table.py`` with ``n`` = number of non-missing slots;
    returns (rendered strings, flags such as ``{'rare_category_adjusted': True}``),
    like ``generate_pii``. MULTI_SELECT: see the module docstring; HIGH_CARD /
    FREE_TEXT are also accepted (invented labels / lorem).
    """
    t = profile.type
    if t is ColumnType.MULTI_SELECT:
        return multi_select(profile, model, n, rng)
    if t is ColumnType.HIGH_CARD_CATEGORICAL:
        return invented_labels_flags(profile, model, n, rng)
    if t is ColumnType.FREE_TEXT:
        return lorem(profile, model, n, rng), {}
    raise UnsupportedColumnError(f"column {profile.name!r}: {t.value} generation is not available (area C)")


def multi_select(profile: ColumnProfile, model: ColumnModel, n: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """MULTI_SELECT values: combinations as a factor, or item-wise sampling (module docstring)."""
    if n == 0:
        return np.empty(0, dtype=object), {"rare_category_adjusted": False}
    params = profile.params or {}
    d = params.get("delimiter") or _infer_delimiter(model)
    levels = [raw for raw in model.nonmissing_counts if isinstance(raw, str)]
    counts = [model.nonmissing_counts[raw] for raw in levels]
    if not levels:
        raise UnsupportedColumnError(f"column {profile.name!r}: no values to sample")
    n_combos = len({raw.strip().casefold() for raw in levels})
    mode = params.get("mode") or ("combinations" if n_combos <= int(params.get("_factor_max", 20)) else "items")
    if mode == "combinations" or not d:
        values, adjusted = quota_sample(levels, counts, n, rng, floor=len(levels) <= n)
        return values, {"rare_category_adjusted": adjusted}
    return _multi_select_items(model, n, rng, d, profile.na_tokens)


def _multi_select_items(model: ColumnModel, n: int, rng: np.random.Generator, d: str,
                        na_tokens: frozenset[str] = frozenset()) -> tuple[np.ndarray, dict[str, bool]]:
    """Item-wise MULTI_SELECT. Values with no item (only separators) and whole-value NA tokens ('None',
    'N/A': levels under the text missing policy) are kept as whole values at their exact counts (floor 1),
    never mixed into item lists."""
    sep_rx = re.compile(r"\s*" + re.escape(d) + r"\s*")
    seps: dict[str, int] = {}  # separator form between items (values with >= 2 items) -> rows
    trails: dict[str, int] = {}  # trailing separator form ('' = none; values with >= 1 item) -> rows
    forms: dict[str, dict[str, int]] = {}
    item_counts: dict[str, int] = {}
    pos_sum: dict[str, float] = {}
    pos_w: dict[str, int] = {}
    k_counts: dict[int, int] = {}
    zero_forms: dict[str, int] = {}
    for raw, c in model.nonmissing_counts.items():
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        items = [] if s in na_tokens else [p.strip() for p in s.split(d) if p.strip()]
        if items:
            body = s.rstrip(d + " \t\r\n\f\v")  # the value without its trailing separator(s)
            trail = s[len(body):]
            trails[trail] = trails.get(trail, 0) + c
            inner = Counter(sep_rx.findall(body))
            if len(items) > 1 and inner:
                form = inner.most_common(1)[0][0]
                seps[form] = seps.get(form, 0) + c
        k_counts[len(items)] = k_counts.get(len(items), 0) + c
        if not items:
            zero_forms[raw] = zero_forms.get(raw, 0) + c
        for j, it in enumerate(items):
            key = it.casefold()
            f = forms.setdefault(key, {})
            f[it] = f.get(it, 0) + c
            item_counts[key] = item_counts.get(key, 0) + c
            if len(items) > 1:
                pos_sum[key] = pos_sum.get(key, 0.0) + c * j / (len(items) - 1)
                pos_w[key] = pos_w.get(key, 0) + c
    keys = list(item_counts)
    canon = {k: max(forms[k].items(), key=lambda kv: kv[1])[0] for k in keys}
    order = {k: (pos_sum[k] / pos_w[k] if pos_w.get(k) else 0.5, i) for i, k in enumerate(keys)}
    k_levels = list(k_counts)
    ks, _ = quota_sample(k_levels, [k_counts[k] for k in k_levels], n, rng, floor=False)
    ks = ks.astype(np.int64)
    total_slots = int(ks.sum())
    floor = len(keys) <= total_slots
    alloc, adjusted = allocate([item_counts[k] for k in keys], total_slots, floor=floor)
    remaining = [int(a) for a in alloc]
    pyr = py_random(rng)
    # an item's raw forms ('ELL' / 'ell') by exact quota over its slots, never folded into one (R2-14)
    form_queue: dict[int, list[str]] = {}
    for i, k in enumerate(keys):
        fk = list(forms[k])
        if len(fk) > 1 and remaining[i] > 0:
            fa, fadj = allocate([forms[k][f] for f in fk], remaining[i], floor=len(fk) <= remaining[i])
            adjusted = adjusted or bool(fadj)
            q = [f for f, a in zip(fk, fa) for _ in range(int(a))]
            pyr.shuffle(q)
            form_queue[i] = q
    out = np.empty(n, dtype=object)
    n_zero = int((ks == 0).sum())
    zero_labels: list[str] = []
    if n_zero:
        zl = list(zero_forms)
        zv, zadj = quota_sample(zl, [zero_forms[z] for z in zl], n_zero, rng, floor=len(zl) <= n_zero)
        zero_labels = zv.tolist()
        adjusted = adjusted or zadj
    # separator and trailing-separator forms by exact quota over the values that have them
    sep_forms = list(seps) or [d]
    n_multi = int((ks > 1).sum())
    sep_rows = quota_sample(sep_forms, [seps.get(f, 1) for f in sep_forms], n_multi, rng, floor=False)[0].tolist()
    trail_forms = list(trails) or [""]
    n_items = int((ks > 0).sum())
    trail_rows = quota_sample(trail_forms, [trails.get(f, 1) for f in trail_forms], n_items, rng,
                              floor=False)[0].tolist()
    weights_all = [item_counts[k] for k in keys]
    for r in np.argsort(-ks, kind="stable").tolist():
        k = int(ks[r])
        if k == 0:
            out[r] = zero_labels.pop()
            continue
        chosen: list[int] = []
        written: dict[int, str] = {}
        avail = [i for i in range(len(keys)) if remaining[i] > 0]
        for _ in range(min(k, len(keys))):
            pool = [i for i in avail if i not in chosen]
            if pool:
                i = pyr.choices(pool, weights=[remaining[j] for j in pool])[0]
            else:
                pool = [i for i in range(len(keys)) if i not in chosen]
                i = pyr.choices(pool, weights=[weights_all[j] for j in pool])[0]
            chosen.append(i)
            q = form_queue.get(i)
            written[i] = q.pop() if q and remaining[i] > 0 else canon[keys[i]]
            if remaining[i] > 0:
                remaining[i] -= 1
        chosen.sort(key=lambda i: order[keys[i]])
        sep = sep_rows.pop() if len(chosen) > 1 else d
        out[r] = sep.join(written[i] for i in chosen) + trail_rows.pop()
    return out, {"rare_category_adjusted": bool(adjusted)}
