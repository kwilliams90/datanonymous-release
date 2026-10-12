"""Phone- and SSN-shaped tokens: what the leak scan matches and the generators never draw (AREA G).

The run-wide data rule of the leak scan (:mod:`privacy.leaks`) looks for a real phone number or SSN of any
column copied into a text column of the output. At district scale, with a million real phone numbers, digits
the tool draws itself equal some real phone number by chance in nearly every run. Examples are a synthetic
last name's digits glued to a synthetic ID in a derived e-mail address (``bakdxvp16512760771@example.org``),
and the random digits of a mimicked e-mail local part or note. Two rules keep such coincidences from FAILing a
file, and every real copy is still caught:

1. Only phone-SHAPED tokens are compared (:func:`phone_tokens`):

   * a whole run of 10-15 digits bounded by non-digits (``3125550199``, ``jane3125550199@x.org``). A longer
     run is one number, never searched for a window;
   * a separator-joined cluster of 10-15 digits in international notation (``+44 20 7946 0958``);
   * consecutive digit groups laid out like a phone number: the NANP layouts 3-3-4 and 1-3-3-4
     (``(312) 555-0199``, ``312.555.0199``, ``1-312-555-0199``), or the layout of a real phone number of
     the run (``020 7946 0958``). Groups with another layout are not one number glued across separators
     (``1234567.2026``, ``4.3 3.9 4.0 3.8 3.5``), and a plain decimal number is never a phone.

   An 11-digit key that opens with the country code 1 is also compared as its 10-digit national number
   (:func:`phone_keys`). An SSN-shaped token is the 3-2-4 layout bounded by non-digits, its groups joined by any
   short run of spaces, dots, slashes or dashes (:data:`SSN_RE`, R9-02: ``123-45-6789``, ``123.45.6789``,
   ``123 - 45 - 6789``; the copy gate's shape, ``detect.text.EMB_SSN_RE``).

2. The generators never write one (:class:`RealDigitKeys`, planned run-wide by
   ``privacy.leaks.plan_digit_keys`` and active while a table is generated, :func:`guard`). A synthetic
   value holding a phone- or SSN-shaped token equal to a real one of the run is redrawn, just like a value
   equal to a real value (``generate.text.Drawer``, ``generate.pii``). A derived-template row that would
   hold one keeps its independently generated value (``generate.structure.apply_templates``). This works
   the way IDs never equal a real ID (D9) and a full name never spells a real person's name (R1-04).

So the scan stays strict. A real phone number or SSN in a text column of the output is a copy and FAILs the
file, including one inside an e-mail local part or free text.
"""

from __future__ import annotations

import contextlib
import functools
import re
import unicodedata
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AbstractSet, Iterable, Iterator

#: Digit clusters: ASCII digits joined by at most two of ``space ( ) . -`` (an opening ``+`` / ``(`` kept).
CLUSTER_RE = re.compile(r"\+?\(?[0-9](?:[ ().\-]{0,2}[0-9])*")
#: The same with at least 10 digits: a whole cluster (the regex engine skips shorter ones).
CLUSTER10_RE = re.compile(r"\+?\(?[0-9](?:[ ().\-]{0,2}[0-9]){9,}")
_GROUP_RE = re.compile(r"[0-9]+")
_NON_DIGIT = re.compile(r"[^0-9]+")
#: A plain decimal number (one point between digits): never read as a phone/SSN digit run.
DECIMAL_NUMBER = re.compile(r"\d+\.\d+")
#: R9-02: an SSN-shaped token (module docstring, rule 1): ``ddd-dd-dddd`` with any separator run of up to five spaces,
#: dots, slashes or dashes between the groups, as the copy gate reads it (``detect.text.EMB_SSN_RE``).
SSN_RE = re.compile(r"(?<!\d)\d{3}[ ./\-]{1,5}\d{2}[ ./\-]{1,5}\d{4}(?!\d)")
MIN_PHONE = 10
MAX_PHONE = 15  # E.164
#: Digit-group layouts of a North American number (``312-555-0199``, ``1 (312) 555-0199``).
NANP_LAYOUTS: frozenset[tuple[int, ...]] = frozenset({(3, 3, 4), (1, 3, 3, 4)})
#: Same-length folding of typographic hyphens (as ``privacy.leaks`` folds text before matching).
_HYPHEN_FOLD = str.maketrans({"\u2010": "-", "\u2011": "-"})


def digits_of(v: str) -> str:
    return _NON_DIGIT.sub("", v)


def phone_keys(d: str) -> list[str]:
    """Comparison keys of a phone's digits: the digits, and the national number of ``1`` + 10 digits."""
    keys = [d]
    if len(d) == 11 and d[0] == "1":
        keys.append(d[1:])
    return keys


def layout_of(cluster: str) -> tuple[int, ...]:
    """Digit-group lengths of a digit cluster (``(312) 555-0199`` -> ``(3, 3, 4)``)."""
    return tuple(len(g) for g in _GROUP_RE.findall(cluster))


def is_phone_layout(layout: tuple[int, ...]) -> bool:
    """A layout of several groups that holds a whole phone number (10-15 digits)."""
    return len(layout) > 1 and MIN_PHONE <= sum(layout) <= MAX_PHONE


@functools.lru_cache(maxsize=4096)
def _token_plan(lens: tuple[int, ...], layouts: frozenset[tuple[int, ...]]) -> tuple[tuple[int, int], ...]:
    """Group ranges ``[i, j)`` of the phone-shaped runs of a cluster whose digit groups have these lengths: each
    group of 10-15 digits, then the consecutive groups laid out like a phone (the longest layout first, so a
    number's national part comes after the whole number)."""
    out = [(i, i + 1) for i, n in enumerate(lens) if MIN_PHONE <= n <= MAX_PHONE]
    if len(lens) > 1:
        for lay in sorted(layouts, key=lambda x: (-sum(x), x)):
            k = len(lay)
            if 2 <= k <= len(lens):
                out += [(i, i + k) for i in range(len(lens) - k + 1) if lens[i:i + k] == lay]
    return tuple(out)


def phone_tokens(text: str, layouts: AbstractSet[tuple[int, ...]] = NANP_LAYOUTS) -> Iterator[tuple[int, int, str]]:
    """``(start, end, digits)`` of every phone-shaped token of ``text`` (module docstring, rule 1).

    ``layouts`` are the digit-group layouts that make a formatted phone number (the NANP layouts and those of
    the run's real phone numbers). A number is yielded once per digit cluster, however many rules find it
    (``1-312-555-0199`` is ``13125550199``, whose national number ``3125550199`` is not yielded again).
    :func:`phone_numbers` yields the same digits without positions, faster.
    """
    lays = layouts if isinstance(layouts, frozenset) else frozenset(layouts)
    for m in CLUSTER10_RE.finditer(text):
        s = m.group()
        groups = list(_GROUP_RE.finditer(s))
        if len(groups) == 2 and DECIMAL_NUMBER.fullmatch(s):  # 2818.170967 is a number, not a phone
            continue
        base = m.start()
        seen: set[str] = set()
        if s[0] == "+" and len(groups) > 1:  # international notation: the whole cluster
            d = digits_of(s)
            if len(d) <= MAX_PHONE:
                seen.update(phone_keys(d))
                yield base, base + len(s), d
        for i, j in _token_plan(tuple(g.end() - g.start() for g in groups), lays):
            d = "".join(g.group() for g in groups[i:j])
            if d not in seen:
                seen.update(phone_keys(d))
                yield base + groups[i].start(), base + groups[j - 1].end(), d


def phone_numbers(text: str, layouts: AbstractSet[tuple[int, ...]] = NANP_LAYOUTS) -> Iterator[str]:
    """The digits of every phone-shaped token of ``text``: exactly those of :func:`phone_tokens`, in the same
    order, without their positions (the scan and the generation guard run it over millions of values)."""
    lays = layouts if isinstance(layouts, frozenset) else frozenset(layouts)
    for s in CLUSTER10_RE.findall(text):
        gs = _GROUP_RE.findall(s)
        if len(gs) == 1:  # a bare run (the common case)
            if len(gs[0]) <= MAX_PHONE:
                yield gs[0]
            continue
        if len(gs) == 2 and DECIMAL_NUMBER.fullmatch(s):
            continue
        plan = _token_plan(tuple(map(len, gs)), lays)
        if s[0] != "+" and len(plan) == 1:  # one number in the cluster (``(312) 555-0199``)
            i, j = plan[0]
            yield "".join(gs[i:j])
            continue
        seen: set[str] = set()
        if s[0] == "+":
            d = "".join(gs)
            if len(d) <= MAX_PHONE:
                seen.update(phone_keys(d))
                yield d
        for i, j in plan:
            d = "".join(gs[i:j])
            if d not in seen:
                seen.update(phone_keys(d))
                yield d


def phone_layouts(text: str) -> set[tuple[int, ...]]:
    """Layouts of the formatted phone-length digit clusters of ``text`` (a real phone value)."""
    out: set[tuple[int, ...]] = set()
    for m in CLUSTER_RE.finditer(text):
        s = m.group()
        lay = layout_of(s)
        if is_phone_layout(lay) and not (len(lay) == 2 and DECIMAL_NUMBER.fullmatch(s)):
            out.add(lay)
    return out


# D46: semantic phone comparison is separate from the general numeric-token rules
# above. Only a declared PHONE value is allowed to be a bare seven-digit local.
_PHONE_MAIN = re.compile(r"\+?[\d(][\d\s()./\-]*\d")
_PHONE_EXTENSION = re.compile(r"\s*(?:extension|ext\.?|x|#)\s*\d{1,6}\s*$", re.I)
_PHONE_FLOAT = re.compile(r"(\+?\d{7,15})(\.0+)")
_PHONE_FOLD = str.maketrans({**{c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\uff0d"},
                           "\uff08": "(", "\uff09": ")", "\uff0b": "+", "\uff0e": ".", "\uff0f": "/"})


def phone_text(text: str) -> str:
    """Fold Unicode decimal digits and common phone punctuation without changing positions."""
    if text.isascii():
        return text
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text).translate(_PHONE_FOLD)


def phone_parts(text: str) -> tuple[str, str, str] | None:
    """A known phone's (main, extension, float-zero suffix), preserving its original layout."""
    folded = phone_text(text)
    m = _PHONE_EXTENSION.search(folded)
    end = m.start() if m else len(text)
    main, extension = text[:end], text[end:]
    f = _PHONE_FLOAT.fullmatch(folded[:end])
    zeros = main[len(f.group(1)):] if f else ""
    if zeros:
        main = main[:-len(zeros)]
    normalized = phone_text(main)
    if not _PHONE_MAIN.fullmatch(normalized):
        return None
    if not 7 <= len(digits_of(normalized)) <= MAX_PHONE:
        return None
    return main, extension, zeros


def phone_numeric_main(text: str) -> str | None:
    """Numeric main of a declared phone, even when its punctuation cannot be synthesized.

    This only interprets the whole declared value, never searches general text.
    Keeping the fallback separate lets generation reject unsupported layouts while
    verification still catches their normalized source bases.
    """
    main = phone_text(text.strip())
    extension = _PHONE_EXTENSION.search(main)
    if extension is not None:
        main = main[:extension.start()]
    if not any(c.isdecimal() for c in main) or any(c.isalnum() and not c.isdecimal() for c in main):
        return None
    return main


def phone_base(text: str) -> tuple[str, bool, bool] | None:
    """Canonical base, local flag and NANP flag; an extension never changes the base."""
    parts = phone_parts(text.strip())
    if parts is None:
        main = phone_numeric_main(text)
        if main is None:
            return None
    else:
        main = phone_text(parts[0])
    d = digits_of(main)
    if not 7 <= len(d) <= MAX_PHONE:
        return None
    country_one = len(d) == 11 and d.startswith("1")
    national = country_one or (len(d) == 10 and d[0] in "23456789" and not main.startswith("+"))
    if country_one:
        d = d[1:]
    return d, len(d) == 7 and not main.startswith("+"), national


def phone_base_keys(base: tuple[str, bool, bool], *, source: bool = False) -> tuple[str, ...]:
    """Keep national numbers distinct across areas; compare local numbers to their suffixes."""
    d, local, national = base
    if local:
        return d, ("L" if source else "S") + d
    if national:
        return d, ("S" if source else "L") + d[-7:]
    return (d,)


def semantic_phones(text: str, *, known: bool = False,
                    layouts: AbstractSet[tuple[int, ...]] = NANP_LAYOUTS) -> Iterator[tuple[str, bool, bool]]:
    """Phone bases in a known PHONE value or in the established phone-shaped spans of text."""
    text = phone_text(text)
    if known:
        base = phone_base(text)
        if base is not None:
            yield base
            return
    seen: set[tuple[str, bool, bool]] = set()
    for b, e, _ in phone_tokens(text, layouts):
        base = phone_base(text[b:e])
        if base is not None and base not in seen:
            seen.add(base)
            yield base
    # The copy gate also supports long separator runs and international national
    # layouts. It explicitly excludes counts, ranges, decimal numbers and dates.
    from ..detect.text import EMB_INTL_PHONE_RE, phone_runs

    spans = phone_runs(text)
    if "+" in text:
        spans += list(EMB_INTL_PHONE_RE.finditer(text))
    for m in spans:
        base = phone_base(m.group())
        if base is not None and 8 <= len(base[0]) <= MAX_PHONE and base not in seen:
            seen.add(base)
            yield base


def semantic_phone_keys(text: str, *, known: bool = False, source: bool = False,
                        layouts: AbstractSet[tuple[int, ...]] = NANP_LAYOUTS) -> Iterator[str]:
    for base in semantic_phones(text, known=known, layouts=layouts):
        yield from phone_base_keys(base, source=source)


@dataclass(frozen=True)
class RealDigitKeys:
    """The run's real phone and SSN keys (PRIVATE, in memory only): what no generated value may hold."""

    phones: AbstractSet[str]
    ssns: AbstractSet[str]
    layouts: AbstractSet[tuple[int, ...]] = NANP_LAYOUTS
    phone_bases: AbstractSet[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.phones or self.ssns or self.phone_bases)

    def holds_phone(self, text: str, *, known: bool = True) -> bool:
        """D46: compare a phone's base, including local/NANP aliases and Unicode decimals."""
        if known:
            base = phone_base(text)
            if base is None:
                return self.holds_real(text)
            if any(k in self.phone_bases for k in phone_base_keys(base)):
                return True
            d = base[0]
            if d in self.ssns or d in self.phones or (base[2] and "1" + d in self.phones):
                return True
            # A forced phone layout can overlap an SSN layout; keep its original guard.
            return bool(self.ssns) and any(digits_of(m.group()) in self.ssns for m in SSN_RE.finditer(text))
        return bool(self.phone_bases) and any(
            k in self.phone_bases for k in semantic_phone_keys(text, known=known, layouts=self.layouts))

    def holds_real(self, text: str) -> bool:
        """True when ``text`` holds a phone-shaped token whose key is a real phone number, or an SSN-shaped
        token that is a real SSN (the leak scan's data rules)."""
        if not text:
            return False
        if self.phone_bases and self.holds_phone(text, known=False):
            return True
        d = digits_of(text)
        n = len(d)
        if n < 9:
            return False
        if not text.isascii():
            text = text.translate(_HYPHEN_FOLD)
        if self.ssns:
            for m in SSN_RE.finditer(text):
                if digits_of(m.group()) in self.ssns:
                    return True
        if n < MIN_PHONE or not self.phones:
            return False
        if n <= 11:
            # every token's key is the whole digits or 10 consecutive digits of 11 (fast negative path)
            if not any(c in self.phones for c in ((d,) if n == 10 else (d, d[1:], d[:10]))):
                return False
        phones = self.phones
        for tok in phone_numbers(text, self.layouts):
            if tok in phones or (len(tok) == 11 and tok[0] == "1" and tok[1:] in phones):
                return True
        return False


_ACTIVE: ContextVar[RealDigitKeys | None] = ContextVar("datanonymous_real_digit_keys", default=None)


@contextlib.contextmanager
def guard(keys: RealDigitKeys | None) -> Iterator[None]:
    """Make ``keys`` the real digit keys the generators avoid (:func:`holds_real`) inside the block."""
    token = _ACTIVE.set(keys if keys else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active() -> RealDigitKeys | None:
    return _ACTIVE.get()


def holds_real(text: str) -> bool:
    """True when ``text`` holds a real phone number or SSN of the run (:func:`guard`); False without a guard."""
    keys = _ACTIVE.get()
    return keys is not None and keys.holds_real(text)


def holds_phone(text: str) -> bool:
    """Compare a generated, known phone with every source phone base of the active run."""
    keys = _ACTIVE.get()
    return keys is not None and keys.holds_phone(text)


def union(parts: Iterable[AbstractSet[str]]) -> AbstractSet[str]:
    """Union of key sets, without copying when only one is non-empty (they can hold millions of keys)."""
    live = [p for p in parts if p]
    if not live:
        return frozenset()
    if len(live) == 1:
        return live[0]
    out: set[str] = set()
    for p in live:
        out |= p
    return out


__all__ = ["phone_tokens", "phone_numbers", "phone_keys", "phone_layouts", "RealDigitKeys", "guard", "active", "holds_real",
           "digits_of", "SSN_RE", "NANP_LAYOUTS", "CLUSTER_RE", "DECIMAL_NUMBER", "phone_text", "phone_parts",
           "phone_base", "phone_base_keys", "phone_numeric_main", "semantic_phones", "semantic_phone_keys", "holds_phone"]
