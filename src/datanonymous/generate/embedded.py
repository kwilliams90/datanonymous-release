"""Personal-data shapes inside generated text (AREA C; R4-07, design D12, D13, D24).

Free text (lorem), multi-select items, same-shape code masks and every fallback mask rebuild a real value's
skeleton. An e-mail address, URL or web domain, IPv4 address, SSN, phone number or street address inside such
a value is never rebuilt letter run by letter run (lorem words on an unreserved domain, digits that may
equal a source phone number or name a real SSN): it is synthesised by the rules of its PII generator
(``generate.pii``):

* e-mail addresses: name-like local parts (Faker name words of the same length, digits redrawn), domains
  mapped by frequency rank to the reserved example.org, example.com, example.net (D24);
* URLs and bare web domains (``www.district.org``, ``springfield.k12.il.us``): scheme, ``www.`` and port
  kept, hosts mapped by rank to the reserved hosts; the path stays text (and is rebuilt like the rest);
* phone numbers: the value's own layout, the k-floored area code, ordinary exchange and random line number
  (PHONE rules); international numbers get random digits;
* SSNs: area 900-999 (never issued); IPv4 addresses: the documentation ranges;
* street addresses: Faker street words, the street suffix, directionals and unit words kept, random house
  number.

:func:`pii_spans` finds them (the copy gate's patterns of ``detect.text``, plus Unicode local parts, bare web
domains and IPv4). :class:`SpanSynth` maps each distinct real span of a column to one synthetic span
(consistent across the column's values, never a real span of the column, never holding a real phone number
or SSN of the run) for the per-column generators (``generate.text.lorem``, multi-select items, code masks);
:meth:`SpanSynth.rewrite` rebuilds a value span by span and fills the text between them. Without a column
synth (``generate.text.mask_fill`` and ``mimic_words`` on any template), :func:`rewrite` gives each span a
stateless value by the same rules. :func:`guard` is the last check of every never-copy candidate
(``generate.text.Drawer``): an e-mail address or URL on an unreserved domain, an SSN outside area 900-999
or a routable IPv4 address in a candidate is corrected the same way. Phone candidates are checked separately
against every source base, including local/NANP aliases and extension variants (D46).
"""

from __future__ import annotations

import ipaddress
import random
import re
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np

from ..errors import GenerationError
from ..privacy import digits as phone_privacy

from ..detect.text import (
    _EMB_ADDRESS_START,
    _EMB_ADDRESS_SUFFIXES,
    EMB_ADDRESS_RE,
    EMB_BARE_PHONE_RE,
    EMB_INTL_PHONE_RE,
    EMB_PHONE_RE,
    EMB_SSN_RE,
    EMB_URL_RE,
    SEP_RUN_MAX,
    is_reserved_domain,
    phone_runs,
)

#: Reserved e-mail / URL domains (RFC 2606), as ``generate.pii.RESERVED_DOMAINS``.
RESERVED = ("example.org", "example.com", "example.net")
#: Span kinds, most specific first (an e-mail address wins over the domain inside it).
KINDS = ("email", "url", "ip", "ssn", "phone", "domain", "address")
_PRIORITY = {k: i for i, k in enumerate(KINDS)}

#: An e-mail address with any letters (``José.Núñez@colegio.mx``), unlike the ASCII copy-gate pattern.
EMAIL_SPAN_RE = re.compile(r"[\w.%+'\-]+@[\w\-]+(?:\.[\w\-]+)*\.[^\W\d_]{2,}")
#: Top-level domains that make ``name.tld`` a web domain inside text (common words such as ``in``, ``at``,
#: ``it``, ``is``, ``me``, ``to``, ``do``, ``be``, ``no`` are left out: ``ended.it`` is a typo, not a domain).
TLDS = ("com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "io", "co", "us", "uk", "ca", "au", "nz",
        "ie", "de", "fr", "es", "nl", "mx", "br", "jp", "cn", "ru", "ch", "se", "dk", "fi", "pl", "eu", "tv", "ly",
        "ai", "app", "dev", "online", "site", "school", "academy", "education", "email")
DOMAIN_SPAN_RE = re.compile(
    r"(?<![\w.@\-/:])(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?:"
    + "|".join(sorted(TLDS, key=len, reverse=True)) + r")(?![\w\-]|\.[A-Za-z0-9])", re.I)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_SPAN_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w]|\.\d)")
#: A whole value that is a digit-only phone number (notes and contact columns only), possibly stored as a
#: float (``3125550199.0``: the ``.0`` is kept as it is).
WHOLE_PHONE_RE = re.compile(r"\+?[01]?\d{10}(?:\.0+)?")
_FLOAT_ZEROS = re.compile(r"(.*?\d)(\.0+)")
_URL_TRAIL = ".,;:!?)]}'\">"
_SAFE_NETS = tuple(ipaddress.ip_network(n) for n in (
    "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "198.18.0.0/15", "240.0.0.0/4"))
_N11 = frozenset(f"{d}11" for d in range(2, 10))
_DIGITS = "0123456789"
#: Nine digits, a separator run (``detect.text.SEP_RUN_MAX``) and parentheses apart at most: a phone number or SSN may
#: hold them (R9-02: ``123 - 45 - 6789``, ``( 020 )  7946``).
_NINE_DIGITS = re.compile(rf"(?:\d\D{{0,{SEP_RUN_MAX + 2}}}){{9}}")
_ANY_DIGIT = re.compile(r"\d")
_DOTTED = re.compile(r"[A-Za-z0-9]\.[A-Za-z0-9]")
#: A necessary condition of ``detect.text.EMB_ADDRESS_RE`` (a number, 1-4 words, a street suffix), fast to test.
_ADDRESS_HINT = re.compile(r"\d[A-Za-z]?\s+(?:\S+\s+){1,4}?(?:"
                           + "|".join(sorted(_EMB_ADDRESS_SUFFIXES, key=len, reverse=True)) + r")\b", re.I)

Span = tuple[int, int, str]

# Keep repeated short templates/candidates, never arbitrarily large notes. Cached span
# tuples cannot be changed through the public list-returning API.
_SPAN_CACHE_SIZE = 1 << 17
_CACHE_TEXT_LIMIT = 4096


# --------------------------------------------------------------------------- finding spans
def _maybe(text: str) -> tuple[bool, bool, bool, bool]:
    """Cheap pre-checks: (has '@', has a URL marker, has 9+ digits, has a dot between letters/digits)."""
    low = text.casefold() if ("W" in text or "w" in text) else text
    return ("@" in text, "://" in text or "www." in low, bool(_NINE_DIGITS.search(text)),
            bool(_DOTTED.search(text)))


def pii_spans(text: str, digit_phone: bool = False, address: bool = True) -> list[Span]:
    """Non-overlapping ``(start, end, kind)`` of the personal-data shapes inside ``text``, in order.

    Kinds: :data:`KINDS`. Phone numbers and SSNs are the copy gate's (``detect.text.pii_shapes``, R9-02): North
    American layouts with separators, international ``+`` numbers of 8-15 digits, numbers in any national layout
    and any separator run (``detect.text.phone_runs``: ``020 - 7946 - 0958``), 10/11-digit runs inside words,
    SSNs 3-2-4 with any separators (``123.45.6789``), and with ``digit_phone`` (a notes or contact column, or a
    unit that is a bare phone number, ``detect.text.bare_phone``) the digit-only numbers of a value. Street addresses
    only with ``address``. Overlaps go to the more specific kind (an e-mail address over the domain in it).
    """
    if not text:
        return []
    if len(text) <= _CACHE_TEXT_LIMIT:
        return list(_cached_spans(text, digit_phone, address))
    return _find_spans(text, digit_phone, address)


@lru_cache(maxsize=_SPAN_CACHE_SIZE)
def _cached_spans(text: str, digit_phone: bool, address: bool) -> tuple[Span, ...]:
    return tuple(_find_spans(text, digit_phone, address))


def _find_spans(text: str, digit_phone: bool, address: bool) -> list[Span]:
    if not text:
        return []
    at, url, digits, dotted = _maybe(text)
    has_digit = digits or bool(_ANY_DIGIT.search(text))
    if not (at or url or has_digit or dotted):
        return []
    cands: list[tuple[int, int, int, str]] = []

    def add(s: int, e: int, kind: str) -> None:
        if e > s:
            cands.append((_PRIORITY[kind], s, e, kind))

    if at:
        for m in EMAIL_SPAN_RE.finditer(text):
            s = m.start()
            while s < m.end() and text[s] in ".'-%+":
                s += 1
            if text[s] != "@":  # a local part made only of punctuation is no address
                add(s, m.end(), "email")
    if url:
        for m in EMB_URL_RE.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in _URL_TRAIL:
                e -= 1
            add(s, e, "url")
    if dotted:
        if has_digit:
            for m in IPV4_SPAN_RE.finditer(text):
                add(m.start(), m.end(), "ip")
        for m in DOMAIN_SPAN_RE.finditer(text):
            tld = m.group().rsplit(".", 1)[1]
            if tld.islower() or m.group().isupper():  # 'ended.It' is a sentence, not a domain
                add(m.start(), m.end(), "domain")
    if digits:
        for m in EMB_SSN_RE.finditer(text):
            add(m.start(), m.end(), "ssn")
        for m in EMB_PHONE_RE.finditer(text):
            add(m.start(), m.end(), "phone")
        if "+" in text:
            for m in EMB_INTL_PHONE_RE.finditer(text):
                if 8 <= sum(ch.isdigit() for ch in m.group()) <= 15:
                    add(m.start(), m.end(), "phone")
        for m in phone_runs(text):  # R9-02: the copy gate's phone numbers in any national layout and separators
            add(m.start(), m.end(), "phone")
        if any(ch.isalpha() for ch in text):
            for m in EMB_BARE_PHONE_RE.finditer(text):
                add(m.start(), m.end(), "phone")
        elif digit_phone and WHOLE_PHONE_RE.fullmatch(text.strip()):
            s = len(text) - len(text.lstrip())
            add(s, s + len(text.strip()), "phone")
        elif digit_phone:  # R9-02: several bare numbers in one value (``07911123456, 07700900123``)
            for m in EMB_BARE_PHONE_RE.finditer(text):
                add(m.start(), m.end(), "phone")
    if address and has_digit and _ADDRESS_HINT.search(text):
        for m in _EMB_ADDRESS_START.finditer(text):
            a = EMB_ADDRESS_RE.match(text, m.end())
            if a:
                add(a.start(), a.end(), "address")
    if not cands:
        return []
    cands.sort()
    taken: list[Span] = []
    for _, s, e, kind in cands:
        if all(e <= ts or s >= te for ts, te, _ in taken):
            taken.append((s, e, kind))
    return sorted(taken)


# --------------------------------------------------------------------------- stateless rules
def _mask_chars(s: str, pyr: random.Random) -> str:
    """Letters -> random letters of the same case, digits -> random digits, everything else kept."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(pyr.choice(_DIGITS))
        elif ch.isalpha():
            c = pyr.choice("abcdefghijklmnopqrstuvwxyz")
            out.append(c.upper() if ch.isupper() else c)
        else:
            out.append(ch)
    return "".join(out)


def _reserved_like(domain: str, rank: int = 0) -> str:
    d = RESERVED[rank % len(RESERVED)]
    return d.upper() if domain.isupper() else d


def _put_digits(template: str, digits: str) -> str:
    it = iter(digits)
    return "".join(next(it) if ch.isdigit() else ch for ch in template)


def _area_code(pyr: random.Random) -> str:
    while True:
        a = str(pyr.randint(2, 9)) + pyr.choice(_DIGITS) + pyr.choice(_DIGITS)
        if a not in _N11:
            return a


def phone_digits(digits: str, pyr: random.Random, area: str | None = None, *, national: bool | None = None) -> str:
    """PHONE rules for one number's digits: North American numbers (10 digits opening with 2-9, or 11 opening with
    1) get ``area`` (else a random non-N11 area code), an ordinary exchange and a random line; other numbers random digits
    (the first one non-zero unless it was zero: a national trunk ``0`` stays, R9-02: ``0207946095``)."""
    line = "".join(pyr.choice(_DIGITS) for _ in range(4))
    if national is not False and len(digits) == 10 and digits[0] not in "01":
        return (area or _area_code(pyr)) + _area_code(pyr) + line
    if national is not False and len(digits) == 11 and digits[0] == "1":
        return "1" + (area or _area_code(pyr)) + _area_code(pyr) + line
    if len(digits) == 7:
        return _area_code(pyr) + line
    if not digits:
        return digits
    first = "0" if digits[0] == "0" else pyr.choice("123456789")
    return first + "".join(pyr.choice(_DIGITS) for _ in range(len(digits) - 1))


def ssn_digits(pyr: random.Random) -> str:
    """SSN rules: area 900-999 (never issued), group 01-99, serial 0001-9999."""
    return "9" + pyr.choice(_DIGITS) + pyr.choice(_DIGITS) + f"{pyr.randint(1, 99):02d}" + f"{pyr.randint(1, 9999):04d}"


def doc_ipv4(pyr: random.Random) -> str:
    return f"{pyr.choice(('192.0.2.', '198.51.100.', '203.0.113.'))}{pyr.randint(1, 254)}"


def _split_url(span: str) -> tuple[str, str, str]:
    """(scheme incl. '://', 'www.' or '', host) of a URL or domain span (port and path left in the host)."""
    scheme = ""
    m = re.match(r"(?i)(?:https?|ftp)://", span)
    if m:
        scheme = m.group()
    host = span[len(scheme):]
    www = host[:4] if host[:4].casefold() == "www." else ""
    return scheme, www, host[len(www):]


def stateless_span(kind: str, span: str, pyr: random.Random) -> str:
    """A synthetic value for one real span with no column context (the same rules as :class:`SpanSynth`)."""
    if kind == "email":
        local, _, domain = span.rpartition("@")
        return _mask_chars(local, pyr) + "@" + _reserved_like(domain)
    if kind in ("url", "domain"):
        scheme, www, host = _split_url(span)
        m = re.match(r"([^:/]*)(.*)", host)
        return scheme + www + _reserved_like(m.group(1)) + m.group(2)
    if kind == "ip":
        return doc_ipv4(pyr)
    if kind == "ssn":
        return _put_digits(span, ssn_digits(pyr))
    if kind == "phone":
        return _phone_span(span, pyr)
    return _mask_chars(span, pyr)  # address (only with a column synth) and unknown kinds


def _split_zeros(span: str) -> tuple[str, str]:
    """(number, '.0' suffix) of a float-stored digit-only phone number (``3125550199.0``), else (span, '')."""
    m = _FLOAT_ZEROS.fullmatch(span) if "." in span else None
    return (m.group(1), m.group(2)) if m and m.group(1).lstrip("+").isdigit() else (span, "")


def _phone_span(span: str, pyr: random.Random) -> str:
    """Draw a complete phone span; every route checks its base, independent of extension."""
    parts = phone_privacy.phone_parts(span)
    if parts is None:
        raise GenerationError("cannot synthesize an unsupported phone span")
    main, extension, zeros = parts
    d = phone_privacy.digits_of(phone_privacy.phone_text(main))
    base = phone_privacy.phone_base(span)
    national = base is not None and base[2]
    area = _area_code(pyr) if national else None
    own = set(phone_privacy.semantic_phone_keys(span, known=True, source=True))

    def accepted(candidate: str) -> bool:
        return (not own.intersection(phone_privacy.semantic_phone_keys(candidate, known=True))
                and not phone_privacy.holds_phone(candidate))

    for _ in range(40):
        candidate = (_put_digits(main, phone_digits(d, pyr, area, national=national)) + zeros
                     + "".join(pyr.choice(_DIGITS) if c.isdigit() else c for c in extension))
        if accepted(candidate):
            return candidate
    # Rejection sampling is only the fast path. Enumerate the actual free space
    # on a hard case, including all extension digits, then fail closed on exhaustion.
    from .pii import _PhoneSpace

    active = phone_privacy.active()
    blocked = own | (set(active.phone_bases) | set(active.phones) | set(active.ssns) if active is not None else set())
    space = _PhoneSpace(d, area, sum(c.isdigit() for c in extension), blocked)
    while True:
        number, ext = space.draw(pyr)
        candidate = _put_digits(main, number) + zeros + _put_digits(extension, ext)
        if accepted(candidate):
            return candidate


def rewrite(template: str, fill: Callable[[str], str], pyr: random.Random, digit_phone: bool = False) -> str:
    """``template`` rebuilt with every personal-data span (except street addresses, which ``fill`` rebuilds
    word by word anyway) given a stateless synthetic value and the text between spans given ``fill``."""
    spans = pii_spans(template, digit_phone, address=False)
    if not spans:
        return fill(template)
    out: list[str] = []
    pos = 0
    for s, e, kind in spans:
        if s > pos:
            out.append(fill(template[pos:s]))
        out.append(stateless_span(kind, template[s:e], pyr))
        pos = e
    if pos < len(template):
        out.append(fill(template[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- the last check (Drawer)
def _safe_ip(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return True
    return any(ip in net for net in _SAFE_NETS)


def guard(candidate: str, pyr: random.Random) -> str:
    """``candidate`` (a synthetic value) with every unsafe personal-data shape corrected (module docstring):
    e-mail domains and URL hosts outside the reserved names -> ``example.org``, SSNs -> area 9xx,
    routable IPv4 addresses -> documentation addresses. Phone collisions are checked separately (D46).
    A safe candidate is returned as it is (the same object)."""
    if not candidate or not isinstance(candidate, str):
        return candidate
    at, url, digits, dotted = _maybe(candidate)
    if not (at or url or digits or dotted):
        return candidate
    spans = pii_spans(candidate, address=False)
    return _guard_spans(candidate, pyr, spans)


def _guard_spans(candidate: str, pyr: random.Random, spans: Iterable[Span]) -> str:
    if not spans:
        return candidate
    out: list[str] = []
    pos = 0
    changed = False
    for s, e, kind in spans:
        span = candidate[s:e]
        new = span
        if kind == "email":
            local, _, domain = span.rpartition("@")
            if not is_reserved_domain(domain):
                new = local + "@" + _reserved_like(domain)
        elif kind == "url":
            scheme, www, host = _split_url(span)
            m = re.match(r"([^:/]*)(.*)", host)
            if not is_reserved_domain(m.group(1)):
                new = scheme + www + _reserved_like(m.group(1)) + m.group(2)
        elif kind == "ip":
            if not _safe_ip(span):
                new = doc_ipv4(pyr)
        elif kind == "ssn":
            d = "".join(ch for ch in span if ch.isdigit())
            if d[0] != "9":
                new = _put_digits(span, "9" + d[1:])
        if new != span:
            changed = True
        out.append(candidate[pos:s])
        out.append(new)
        pos = e
    if not changed:
        return candidate
    out.append(candidate[pos:])
    return "".join(out)


class GuardMemo:
    """A bounded, Drawer-owned cache of deterministic corrections.

    IP spans can draw random numbers, so they always run the guard. E-mail
    candidates bypass the memo because their mostly unique keys cost more than
    they save. True records an unchanged candidate, preserving object identity
    even when the next equal input is a different string object.
    """

    def __init__(self, maxsize: int = _SPAN_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self.cache: dict[str, str | bool] = {}

    def __call__(self, candidate: str, pyr: random.Random) -> str:
        if not candidate or not isinstance(candidate, str):
            return candidate
        if "@" in candidate or len(candidate) > _CACHE_TEXT_LIMIT:
            return guard(candidate, pyr)
        hit = self.cache.get(candidate)
        if hit is not None:
            return candidate if hit is True else hit
        spans = _cached_spans(candidate, False, False)
        out = _guard_spans(candidate, pyr, spans)
        if not any(kind == "ip" for _, _, kind in spans) and self.maxsize > 0:
            if len(self.cache) >= self.maxsize:
                self.cache.clear()
            self.cache[candidate] = True if out is candidate else out
        return out


# --------------------------------------------------------------------------- per-column synth
class SpanSynth:
    """One synthetic span per distinct real span of a column (module docstring), built by :meth:`build`."""

    def __init__(self, spans: dict[str, list[Span]], mapping: dict[str, dict[str, str]],
                 digit_phone: bool) -> None:
        self.spans = spans
        self.mapping = mapping
        self.digit_phone = digit_phone

    @classmethod
    def build(cls, profile: Any, templates: Iterable[tuple[str, int]], rng: np.random.Generator,
              digit_phone: bool = False, avoid: Iterable[str] = ()) -> "SpanSynth | None":
        """The synth of a column's ``templates`` ((value, rows) pairs), or None when no value holds a span.

        No random number is drawn unless some value holds one. ``avoid`` adds real values (whole values or
        items) whose spans the synthetic spans must not equal either.
        """
        from ..model import ColumnModel
        from . import pii
        from .text import distinct_groups

        spans: dict[str, list[Span]] = {}
        by_kind: dict[str, dict[str, int]] = {}
        for t, c in templates:
            if t in spans:
                continue
            sp = pii_spans(t, digit_phone)
            spans[t] = sp
            for s, e, kind in sp:
                k = "url" if kind == "domain" else kind  # one host map for URLs and bare domains
                d = by_kind.setdefault(k, {})
                d[t[s:e]] = d.get(t[s:e], 0) + int(c)
        if not by_kind:
            return None
        real: set[str] = {key.casefold() for d in by_kind.values() for key in d}
        for v in avoid:
            for s, e, _ in pii_spans(v, digit_phone):
                real.add(v[s:e].casefold())
        real_locals = {key.rpartition("@")[0].casefold() for key in by_kind.get("email", {})}
        fk = pii._seeded_faker(rng)
        pyr: random.Random = fk.random
        from ..privacy.digits import holds_real
        from ..privacy.texts import holds_real as real_text

        makers = {"email": pii._make_email, "url": pii._make_url, "ip": pii._make_ip, "ssn": pii._make_ssn,
                  "phone": pii._make_phone, "address": pii._make_address}
        used: set[str] = set()
        mapping: dict[str, dict[str, str]] = {}
        for kind in ("email", "url", "ip", "ssn", "phone", "address"):
            counts = by_kind.get(kind)
            if not counts:
                continue
            groups = distinct_groups(ColumnModel(nonmissing_counts=counts))
            if kind == "phone":  # a float-stored number keeps its '.0'; the area-code pmf reads the number
                bare = [replace(g, rep=_split_zeros(g.rep)[0]) for g in groups]
                base = makers[kind](profile, ColumnModel(nonmissing_counts=counts), bare, fk, pyr)

                def make(g: Any, a: int, base: Any = base) -> str:
                    number, zeros = _split_zeros(g.rep)
                    return base(replace(g, rep=number), a) + zeros
            else:
                # street addresses: no house-number pmf (random numbers: a note's house number is never kept)
                make = makers[kind](profile, ColumnModel(nonmissing_counts=counts),
                                    [] if kind == "address" else groups, fk, pyr)

            def ok(cand: str, kind: str = kind) -> bool:
                key = cand.casefold()
                # nor a real text of the run, nor (an e-mail span) one with a real local part (privacy.texts, R7)
                if not key or key in real or key in used or holds_real(cand) or real_text(cand):
                    return False
                if kind == "phone" and phone_privacy.holds_phone(cand):
                    return False
                return not (kind == "email" and cand.rpartition("@")[0].casefold() in real_locals)

            out = mapping.setdefault(kind, {})
            for g in groups:
                cand = ""
                for a in range(40):
                    cand = guard(make(g, a), pyr)
                    if ok(cand):
                        break
                else:
                    if kind == "phone":
                        # The phone maker owns a finite shuffled format space;
                        # continuing it preserves area codes and extension capacity.
                        while True:
                            cand = guard(make(g, 40), pyr)
                            if ok(cand):
                                break
                        used.add(cand.casefold())
                        out[g.key] = cand
                        continue
                    for _ in range(40):
                        cand = stateless_span(kind, g.rep, pyr)
                        if ok(cand):
                            break
                    else:
                        cand = _numbered(kind, g.rep, real | used, pyr)
                used.add(cand.casefold())
                out[g.key] = cand
        return cls(spans, mapping, digit_phone)

    def rewrite(self, template: str, fill: Callable[[str], str]) -> str:
        """``template`` with each span replaced by its synthetic span and the text between spans by ``fill``."""
        spans = self.spans.get(template)
        if spans is None:
            spans = pii_spans(template, self.digit_phone)
        if not spans:
            return fill(template)
        out: list[str] = []
        pos = 0
        for s, e, kind in spans:
            if s > pos:
                out.append(fill(template[pos:s]))
            k = "url" if kind == "domain" else kind
            new = self.mapping.get(k, {}).get(template[s:e].strip().casefold())
            if kind == "phone" and new is not None and phone_privacy.holds_phone(new):
                new = None
            out.append(new if new is not None else stateless_span(kind, template[s:e], random.Random(s)))
            pos = e
        if pos < len(template):
            out.append(fill(template[pos:]))
        return "".join(out)


def _numbered(kind: str, rep: str, taken: set[str], pyr: random.Random) -> str:
    """A last-resort synthetic span that is no real or used one: a numbered reserved host or local part (nor a real
    text or e-mail local part of the run, ``privacy.texts``)."""
    from ..privacy.texts import holds_real as real_text
    from ..privacy.digits import holds_real

    if kind == "phone":
        from ..model import ColumnModel, ColumnProfile, ColumnType
        from .pii import _make_phone
        from .text import Group

        group = Group(rep.casefold(), rep, [(rep, 1)], 1)
        profile = ColumnProfile(index=0, name="phone", type=ColumnType.PHONE, params={"_min_cell": 1})
        make = _make_phone(profile, ColumnModel(), [group], None, pyr)
        while True:
            cand = guard(make(group, 0), pyr)
            if (cand.casefold() not in taken and not real_text(cand)
                    and not phone_privacy.holds_phone(cand)):
                return cand

    for i in range(1, 100_000):
        if kind == "email":
            cand = f"user{i}@{RESERVED[0]}"
        elif kind == "url":
            scheme, www, _ = _split_url(rep)
            cand = f"{scheme}{www}h{i}.{RESERVED[0]}"
        else:
            cand = stateless_span(kind, rep, pyr)
        if (cand.casefold() not in taken and not real_text(cand) and not holds_real(cand)
                and not (kind == "phone" and phone_privacy.holds_phone(cand))):
            return cand
    raise GenerationError("cannot find an unused collision-free personal-data span")


__all__ = ["KINDS", "SpanSynth", "guard", "pii_spans", "phone_digits", "rewrite", "stateless_span"]
