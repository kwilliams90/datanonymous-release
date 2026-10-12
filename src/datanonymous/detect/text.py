"""Text and PII detection (AREA C): design D12, D13, D24, section 3 #25-35, section 10.

Value rules run on the stripped distinct values with their counts (shares are
row-weighted over the non-missing values; NA tokens are not values here).
:func:`detect_text_semantics` tries, in this order (first match wins):

==============  ==================================================================  ===========
type            rule                                                                share
==============  ==================================================================  ===========
EMAIL           ``local@domain.tld``                                                >= 95%
URL             ``http(s)://host...``, ``ftp://...`` or ``www.host.tld...``         >= 95%
IP              IPv4 dotted quad or IPv6                                            >= 95%
SSN             ``999-99-9999`` / ``999 99 9999``; with an SSN header hint also     >= 95%
                9 digits and partly masked forms (``XXX-XX-1234``)
PHONE           phone header hint + 7-15 digits in a phone layout (+ extension)     >= 90%
ZIP             ZIP header hint + ``99999``, ``99999-9999``, ``999999999`` (4-digit  >= 90%
                ZIPs that lost a leading zero in numeric storage are accepted)
LAT / LON       lat/lon header token + plain decimals inside [-90, 90] / [-180, 180] 100%
STREET_ADDRESS  house number + street words + a street suffix (St, Ave, Rd ...)     >= 80%
PERSON_NAME     header hint (first/last/middle/full name, person-role words, and    >= 90%
                their localized forms: ``Nombre``, ``Prénom``, ``Responsável``);
                values must look like names (letters, spaces, ``-'.,``). Header
                hint only: values are never judged by their content (D35)
CITY            city/town header hint; values look like place names                >= 90%
STATE           every distinct value is a US state/territory code or name, with a   100%
                state header hint or at least 3 distinct values
MULTI_SELECT    ``;`` ``|`` or ``,`` in >= 30% of the values; items are short       >= 30%
                labels (<= factor-max distinct items, <= 4 words); never with a
                PII/free-text or org-code header hint (D10/D36: an org code is
                whole); never thousands/decimal-comma numbers;
                never one pair per value (``Nguyen, Maria``: :func:`pair_layout`)
==============  ==================================================================  ===========

Never-copy PII types (EMAIL, PERSON_NAME, PHONE, SSN, STREET_ADDRESS, IP, URL,
CITY) and ZIP/LAT/LON use the **typed** missing policy: NA tokens such as ``N/A``
are missing markers, written back verbatim at their exact rates, while every
real value is replaced (design 3: tokens are always written back verbatim).
FREE_TEXT and HIGH_CARD_CATEGORICAL use it for the same reason.

:func:`apply_pii_precedence` is the D12 guard: PII and comment/alert/note header
hints beat every value-copying rule (CONSTANT, FLAG_SINGLE_TOKEN, FACTOR,
ORG_CODE, DIGIT_CODE kept as levels, numbers for phone/SSN/DOB hints, and
person numbers under ``Teacher`` or ``Case_Manager``: :func:`person_number_header`); a
value-validated PII type found by :func:`detect_text_semantics` wins over
CONSTANT/FLAG and native detections. Numbers and short single-token codes that
fall to the guard become same-shape masks (digits stay digits, so a numeric
column still reads as numbers); other values become invented labels. A header
naming a comment/note/alert column itself (``Notes``) wins over numeric typing
too (R3-01). Explicit overrides (``--force-factor`` ...) never reach the guard.

:func:`copy_gate` (R3, R4, D35) is then the one place that decides which real
values a detection may write verbatim, whatever its type (``--force-type`` and
``--org-code`` detections too; only ``--force-factor`` keeps every value). Real
TEXT values (levels of factors, constants, flags, non-vocabulary ordinals,
organisational codes and digit codes; native dictionary entries and Stata value
labels; multi-select items; tokens among numbers) follow the D35 allowlist
(:func:`unit_kept`): a value is written verbatim only when it is provably safe
(a word of the built-in vocabularies, ``detect.vocab``; an organisational code,
D10, whose person name in an explicit layout is replaced by a consistent fake
name, D36, ``detect.org_names``; a unit without a letter or digit; or held by
>= ``--min-cell`` distinct
people over the run), never with a personal-data shape (:func:`unit_shapes`,
whatever the header: a bare North American phone number is one in any column,
R6-01) nor under a personal header (also for ``--org-code`` and
``--force-type``, R6-03), nor when it is a real person ID of the run (R9-01,
``privacy.realids``); every other value gets a same-shape
pseudonym with its exact count, one per value for the whole run
(``generate.pseudonyms``). The decision needs entity counts, so the gate marks
the detection (:data:`PENDING_KEY`), ``profile.table`` counts each unit's
holders (:func:`count_units`: the fewest over every ID column of the table,
the ID columns a flag turned into something else and the names of the row's
own person, else rows with exported duplicates counted once;
``profile.entity.d35_keys``; only an ID of the row's own person or a name under
a student header proves people, hardening pass 3) and the pipeline decides over
the run (:func:`lookup_fields`, :func:`run_holders`, :func:`copy_gate_units`;
a count of rows or records without a key of the row's own person gets the
``d35_rows_counted`` warning). A name-like vocabulary word is read in its
section beside an unambiguous word of the section, under a header naming the
section, or beside the value describing it in a one-to-one partner column
(:func:`unit_domains`, :func:`partner_domains`). The
leak scanner excuses
exactly the kept values (:func:`pseudonymized`, :func:`copy_cleared`) and
indexes every pseudonymised one. Numbers keep their own rules (digit-only phone
numbers under a person, phone/contact or notes header become masks). Value
content heuristics (sentence rules, name lists and layouts) are not applied to
values (D35); :func:`value_name_detection`, :func:`is_sentence_like` and
:func:`token_kinds` stay for header and path screens (D34). Phone headers are
structural (``hints.phone_header_hint``): ``Parent_Cell`` / ``Tel_No`` /
``WhatsApp`` are phone hints; ``SMS`` / ``Text`` / ``Home`` / ``Primary`` and a
bare ``Ph`` / ``Mob`` are contact headers whose phone-number values make a PHONE
column (under a bare ``Ph``, seven-digit local numbers too: :func:`contact_phone`).

A header can carry several PII kinds (``city_state``: state and city);
:func:`pii_keys` lists them all and every one counts (``HeaderHints.pii`` keeps
only the first match).

:func:`classify_overflow_text` splits non-factor text into FREE_TEXT (mean >= 4
words, mean length > 40, or a free-text header hint) and HIGH_CARD_CATEGORICAL;
short single-token code columns (no whitespace, <= 16 characters, digits in at
least half of the distinct values) get the subtype ``code`` and are generated
as same-shape character-class masks (D13: ``AB-1234`` -> ``QX-0627``).

:func:`text_stats` returns report-safe aggregates only (lengths, word counts,
case and character shares, layouts, masks/area codes/street suffixes held by >=
``--min-cell`` rows, counts): never a value. E-mail domains, URL hosts, ZIP
codes and states appear as counts only (D21: real labels only where the output
reproduces them; real domains and hosts are replaced, so they are never named;
reserved ``example.*`` names of the synthetic side are).
:func:`doc_fields` adds DATA_DICTIONARY fields computed from SYNTHETIC
statistics. The private ``params['_min_cell']`` carries ``--min-cell`` to the
statistics and generators (``_`` keys never reach the report).
"""

from __future__ import annotations

import ipaddress
import itertools
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..errors import UsageError
from ..hints import (NAME_LIKE_THING_WORDS, PERSON_ROLE_WORDS, PII_HINTS, bare_short_phone_header, is_role_word,
                     phone_header_hint, pii_hint_matches, strip_role_trailers)
from ..hints import tokens as header_tokens
from ..model import NEVER_COPY_TYPES, TYPE_INFO, ColumnType, Message
from ..profile.codes import codes_of
from . import name_levels
from . import names as name_values
from . import numeric as numeric_detect
from .context import DetectContext, Detection
from .ids import char_mask
from .vocab import context_domains, header_domains, is_name_like, is_placeholder_word, is_vocabulary

T = ColumnType

#: PII hint kinds (keys of hints.PII_HINTS) whose real text values must never be copied.
GUARDED_PII_TEXT = frozenset({"first_name", "last_name", "middle_name", "full_name", "email", "phone", "ssn",
                              "address", "city", "dob"})
#: PII hint kinds whose numeric storage is itself the personal value.
GUARDED_PII_NUMERIC = frozenset({"phone", "ssn", "dob"})
#: Person-name hint kinds: numbers under such a header (``Teacher``, ``Counselor``, ``Case_Manager`` holding staff
#: numbers) are person numbers, never kept (D10/D12, :func:`person_number_header`).
PERSON_NAME_HINTS = frozenset({"first_name", "last_name", "middle_name", "full_name"})

_COPYING_TEXT_TYPES = frozenset({ColumnType.CONSTANT, ColumnType.FLAG_SINGLE_TOKEN, ColumnType.FACTOR,
                                 ColumnType.ORG_CODE, ColumnType.DIGIT_CODE})
#: Numbers stored as numbers (the masks keep numeric storage: ``params['numeric_storage']``).
_NUMERIC_STORAGE_TYPES = frozenset({ColumnType.INTEGER, ColumnType.INTEGER_YEAR, ColumnType.FLOAT,
                                    ColumnType.DECIMAL})
#: Every number type that writes real values (discrete levels, the exact min/max of D14, tokens) (R3-01).
_COPYING_NUMERIC_TYPES = _NUMERIC_STORAGE_TYPES | {ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS}
#: Semantic types whose generated values never equal a real value of the column.
NEVER_COPY_SEMANTIC = frozenset({T.EMAIL, T.PERSON_NAME, T.PHONE, T.SSN, T.STREET_ADDRESS, T.IP, T.URL, T.CITY})
#: Types detected by this module that use the typed missing policy (NA tokens written verbatim).
TYPED_POLICY_TYPES = NEVER_COPY_SEMANTIC | {T.ZIP, T.LAT, T.LON, T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL}

DEFAULT_MIN_CELL = 10

# --------------------------------------------------------------------------- thresholds (design 3, prototype)
EMAIL_SHARE = 0.95
URL_SHARE = 0.95
IP_SHARE = 0.95
SSN_SHARE = 0.95
PHONE_SHARE = 0.90
ZIP_SHARE = 0.90
ADDRESS_SHARE = 0.80
NAME_SHARE = 0.90
MULTI_SELECT_SHARE = 0.30
MAX_SCREEN_DISTINCT = 20_000  # design 10: above 20k distinct values, screen a sample first

# --------------------------------------------------------------------------- value patterns
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:(?:https?|ftp)://[^\s/?#]+|www\.[^\s/?#]+\.[A-Za-z]{2,})(?:[/?#:]\S*)?", re.I)
URL_PARSE_RE = re.compile(r"(?P<scheme>(?:https?|ftp)://)?(?P<host>[^\s/?#:]+)(?P<port>:\d+)?(?P<rest>.*)", re.I)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"{_OCTET}(?:\.{_OCTET}){{3}}")
SSN_RE = re.compile(r"\d{3}([- ])\d{2}\1\d{4}")
SSN_HINTED_RE = re.compile(r"[\dXx*#]{3}([- ]?)[\dXx*#]{2}\1\d{4}")
#: A phone: optional '+', digits and layout characters, then an optional extension.
PHONE_RE = re.compile(r"(?P<main>\+?[\d(][\d\s().\-/]*\d)(?P<ext>\s*(?:x|ext\.?|extension|#)\s*\d{1,6})?", re.I)
PHONE_MIN_DIGITS, PHONE_MAX_DIGITS = 7, 15
_NON_DIGIT = re.compile(r"\D+")
ZIP_RE = re.compile(r"(?P<zip5>\d{5})(?:(?P<sep>[- ]?)(?P<plus4>\d{4}))?")
ZIP_HINTED_RE = re.compile(r"(?P<zip5>\d{4,5})(?:(?P<sep>[- ]?)(?P<plus4>\d{4}))?")

#: Street suffixes (prototype list, extended with common USPS abbreviations).
STREET_SUFFIXES: tuple[str, ...] = (
    "street", "st", "avenue", "ave", "av", "road", "rd", "drive", "dr", "lane", "ln", "boulevard", "blvd",
    "court", "ct", "way", "place", "pl", "parkway", "pkwy", "circle", "cir", "highway", "hwy", "terrace", "ter",
    "trail", "trl", "loop", "square", "sq", "pike", "plaza", "plz", "expressway", "expy", "freeway", "fwy",
    "cove", "cv", "alley", "aly", "row", "run", "path", "crossing", "xing",
)
_SUFFIX_ALT = "|".join(sorted(STREET_SUFFIXES, key=len, reverse=True))
#: The body ends on a non-space character, so a long whitespace run is scanned once
#: (``\S.*?`` followed by ``\s+`` was quadratic in the run length).
ADDRESS_RE = re.compile(
    rf"(?P<num>\d+)(?P<numsfx>[A-Za-z]?)(?P<sp1>\s+)(?P<body>\S(?:.*?\S)?)(?P<sp2>\s+)(?P<suffix>{_SUFFIX_ALT})"
    r"(?P<dot>\.?)(?P<rest>(?:[\s,].*)?)", re.I | re.S)
DIRECTIONALS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
                          "northeast", "northwest", "southeast", "southwest"})
UNIT_WORDS = frozenset({"apt", "apartment", "unit", "ste", "suite", "bldg", "building", "fl", "floor", "rm", "room",
                        "lot", "trlr", "spc", "space", "dept", "box", "po", "p", "o"})

US_STATE_ABBR = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK "
    "OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR GU VI AS MP AA AE AP".lower().split())
US_STATE_NAMES = frozenset(s.lower() for s in (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
    "Puerto Rico", "Guam", "U.S. Virgin Islands", "Virgin Islands", "American Samoa", "Northern Mariana Islands"))

#: Name hint kinds -> PERSON_NAME subtype.
NAME_HINT_SUBTYPES = {"first_name": "first", "last_name": "last", "middle_name": "middle", "full_name": "full"}
#: Name punctuation; '&' and '/' join two people in one cell ("John & Mary Smith", "Ana / Luis Garcia").
_NAME_PUNCT = frozenset(" '\u2019`-.,()\"&/")
NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "vi"})
NAME_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "rev", "fr", "sra", "sr"})
NAME_PARTICLES = frozenset({"de", "del", "della", "la", "le", "van", "von", "der", "den", "da", "di", "du", "st",
                            "dos", "das", "bin", "ibn", "al", "el", "y"})

#: lat/lon header tokens (whole tokens; 'long' only as the last token: not 'long_jump').
LAT_TOKENS = frozenset({"lat", "latitude"})
LON_TOKENS = frozenset({"lon", "lng", "longitude"})

MULTI_DELIMITERS = (";", "|", ",")
_NUMBER_WITH_COMMAS = re.compile(r"[+-]?[$\u20ac\xa3]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|[+-]?\d+,\d+")
MULTI_MAX_ITEM_WORDS = 4
MULTI_MAX_ITEM_LENGTH = 40

#: Short single-token code columns (D13 masks).
CODE_MAX_LENGTH = 16
CODE_DIGIT_SHARE = 0.5

#: ``--pii-col`` subtypes -> (type, subtype).
PII_SUBTYPES: dict[str, tuple[ColumnType, str | None]] = {
    "email": (ColumnType.EMAIL, None),
    "email_address": (ColumnType.EMAIL, None),
    "first_name": (ColumnType.PERSON_NAME, "first"),
    "first": (ColumnType.PERSON_NAME, "first"),
    "last_name": (ColumnType.PERSON_NAME, "last"),
    "last": (ColumnType.PERSON_NAME, "last"),
    "surname": (ColumnType.PERSON_NAME, "last"),
    "middle_name": (ColumnType.PERSON_NAME, "middle"),
    "middle": (ColumnType.PERSON_NAME, "middle"),
    "full_name": (ColumnType.PERSON_NAME, "full"),
    "name": (ColumnType.PERSON_NAME, "full"),
    "phone": (ColumnType.PHONE, None),
    "phone_number": (ColumnType.PHONE, None),
    "ssn": (ColumnType.SSN, None),
    "zip": (ColumnType.ZIP, None),
    "zip_code": (ColumnType.ZIP, None),
    "postal_code": (ColumnType.ZIP, None),
    "state": (ColumnType.STATE, None),
    "city": (ColumnType.CITY, None),
    "address": (ColumnType.STREET_ADDRESS, None),
    "street_address": (ColumnType.STREET_ADDRESS, None),
    "ip": (ColumnType.IP, None),
    "ip_address": (ColumnType.IP, None),
    "url": (ColumnType.URL, None),
}


# --------------------------------------------------------------------------- small helpers
def _share(values: dict[str, int], pred: Any) -> float:
    """Row-weighted share of ``values`` (stripped value -> count) satisfying ``pred``.

    Above :data:`MAX_SCREEN_DISTINCT` distinct values a deterministic screen of
    the first ones (insertion order) runs first; the full set is checked only
    when the screen reaches the lowest threshold used here (design 10).
    """
    total = sum(values.values())
    if not total:
        return 0.0
    if len(values) > MAX_SCREEN_DISTINCT:
        head = dict(list(values.items())[:MAX_SCREEN_DISTINCT])
        h_total = sum(head.values())
        h_ok = sum(c for v, c in head.items() if pred(v))
        if h_total and h_ok / h_total < MULTI_SELECT_SHARE:
            return h_ok / h_total
    ok = sum(c for v, c in values.items() if pred(v))
    return ok / total


def _fullmatch(rx: re.Pattern[str]) -> Any:
    return lambda v: rx.fullmatch(v) is not None


def is_email(v: str) -> bool:
    return EMAIL_RE.fullmatch(v) is not None


def is_url(v: str) -> bool:
    return URL_RE.fullmatch(v) is not None


def is_ip(v: str) -> bool:
    if IPV4_RE.fullmatch(v):
        return True
    if ":" in v and len(v) <= 45:
        try:
            ipaddress.IPv6Address(v)
            return True
        except ValueError:
            return False
    return False


def phone_parts(v: str) -> tuple[str, str] | None:
    """(main, extension) of a phone-like value, or None."""
    m = PHONE_RE.fullmatch(v)
    if not m:
        return None
    main = m.group("main")
    nd = len(_NON_DIGIT.sub("", main))
    if not PHONE_MIN_DIGITS <= nd <= PHONE_MAX_DIGITS:
        return None
    return main, m.group("ext") or ""


def is_phone(v: str) -> bool:
    return phone_parts(v) is not None


def name_like(v: str) -> bool:
    """Letters plus name punctuation (spaces, ``-'.,()"&/``), no digits, at most 6 words."""
    if not v or not any(ch.isalpha() for ch in v):
        return False
    if not all(ch.isalpha() or ch in _NAME_PUNCT for ch in v):
        return False
    return len(v.split()) <= 6


def is_state(v: str) -> bool:
    key = v.strip().rstrip(".").casefold()
    return key in US_STATE_ABBR or key in US_STATE_NAMES


def latlon_hint(name: str) -> ColumnType | None:
    """LAT or LON when a header token is a latitude/longitude word."""
    toks = header_tokens(name)
    if any(t in LAT_TOKENS for t in toks):
        return T.LAT
    if any(t in LON_TOKENS for t in toks) or (toks and toks[-1] == "long"):
        return T.LON
    return None


def name_layout(v: str) -> str:
    """Report-safe layout category of a name value (never the value)."""
    s = v.strip()
    words = [w for w in re.split(r"[\s,]+", s) if w]
    has_initial = any(len(w.rstrip(".")) == 1 and w[0].isalpha() for w in words)
    if "&" in s or "/" in s or " and " in s.casefold():
        base = "two_people"
    elif "," in s:
        base = "last_comma_first"
    elif len(words) <= 1:
        base = "single"
    elif len(words) == 2:
        base = "first_last"
    else:
        base = "first_middle_last"
    return base + ("_initial" if has_initial else "")


def _det(t: ColumnType, ctx: DetectContext, params: dict[str, Any], reason: str,
         subtype: str | None = None) -> Detection:
    params = dict(params)
    params["_min_cell"] = int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))
    g = guarded_hint(ctx)
    if g is not None:
        params.setdefault("pii_hint", g)
    det = Detection(t, subtype=subtype, params=params, reason=reason)
    if t in TYPED_POLICY_TYPES:
        det.missing_policy = "typed"
    return det


@lru_cache(maxsize=4096)
def _pii_keys(name: str, first: str | None) -> tuple[str, ...]:
    keys = [first] if first else []
    keys += [k for k in PII_HINTS if k != first and pii_hint_matches(name, k)]
    return tuple(keys)


def contact_header(ctx: DetectContext) -> bool:
    """R4-02: a header that MAY hold phone numbers (``hints.phone_header_hint`` is ``'contact'``: ``SMS``,
    ``Text``, ``Home``, ``Primary``, ``Work_Number``, ``Phone_Type``): values that are phone numbers make it a
    PHONE column (:func:`detect_text_semantics`) and its digit-only phone numbers are never copied
    (:func:`digit_phone_header`); its ordinary labels and numbers stay."""
    return phone_header_hint(ctx.name) == "contact"


#: A local North American number without its area code (exchange 2-9, seven digits), possibly stored as a float.
LOCAL_PHONE_RE = re.compile(r"[2-9]\d{6}(?:\.0+)?")


def contact_phone(v: str, local: bool = False) -> bool:
    """A phone number under a contact header: a phone layout (:func:`phone_parts`) with >= 10 digits, or with
    separators (``555-0187``); a bare 7-9 digit number there may be anything, except under a header that is a short
    phone word alone (``local``: ``Ph``, ``Mob``; hardening pass 3), where a seven-digit local number is one."""
    if local and LOCAL_PHONE_RE.fullmatch(v.strip()):
        return True
    parts = phone_parts(v)
    if parts is None:
        return False
    main = parts[0]
    return len(_NON_DIGIT.sub("", main)) >= 10 or any(ch in main for ch in "-. ()/")


def pii_keys(ctx: DetectContext) -> tuple[str, ...]:
    """Every PII hint kind of the header, ``ctx.hints.pii`` first (``city_state`` -> ('state', 'city')).

    ``HeaderHints.pii`` keeps only the first matching kind; a header naming two kinds must be treated as
    both, or the second one's values would be copied (e.g. real cities in a ``city_state`` column).
    """
    return _pii_keys(ctx.name, ctx.hints.pii)


def guarded_hint(ctx: DetectContext, kinds: frozenset[str] = GUARDED_PII_TEXT) -> str | None:
    """The first PII hint kind of the header whose real values are never copied (or None)."""
    return next((k for k in pii_keys(ctx) if k in kinds), None)


# --------------------------------------------------------------------------- the AREA C hook
def detect_text_semantics(ctx: DetectContext) -> Detection | None:
    """AREA C hook: value-validated / header-hinted PII and other text semantics.

    When called: by ``detect/types.py`` after the dates hook and before
    INTEGER_YEAR, ORG_CODE, the numeric rules and FACTOR, so PII hints win over
    the factor rule (D12) and value-validated PII comes before ID candidates.
    Also called by :func:`apply_pii_precedence` for CONSTANT/FLAG and native
    detections.

    Inputs: ``ctx.typed_values`` (validation; NA tokens excluded),
    ``ctx.text_values`` (MULTI_SELECT), ``ctx.hints.pii`` and
    ``ctx.hints.free_text``, ``ctx.digit_code``.

    Returns a Detection (EMAIL, URL, IP, SSN, PHONE, ZIP, LAT, LON,
    STREET_ADDRESS, PERSON_NAME, CITY, STATE, MULTI_SELECT; order in the module
    docstring) or None.
    """
    tv = ctx.typed_values
    if not tv:
        return None
    h = ctx.hints
    keys = pii_keys(ctx)
    share = _share(tv, is_email)
    if share >= EMAIL_SHARE:
        return _det(T.EMAIL, ctx, {"match_share": round(share, 4)},
                    f"{share:.1%} of values are e-mail addresses (>= {EMAIL_SHARE:.0%})")
    share = _share(tv, is_url)
    if share >= URL_SHARE:
        return _det(T.URL, ctx, {"match_share": round(share, 4)}, f"{share:.1%} of values are URLs")
    share = _share(tv, is_ip)
    if share >= IP_SHARE:
        return _det(T.IP, ctx, {"match_share": round(share, 4)}, f"{share:.1%} of values are IP addresses")
    share = _share(tv, _fullmatch(SSN_RE))
    if share >= SSN_SHARE:
        return _det(T.SSN, ctx, {"match_share": round(share, 4)}, f"{share:.1%} of values have the SSN form 999-99-9999")
    if "ssn" in keys:
        share = _share(tv, lambda v: v.isdigit() and len(v) == 9 or SSN_HINTED_RE.fullmatch(v) is not None)
        if share >= SSN_SHARE:
            return _det(T.SSN, ctx, {"match_share": round(share, 4)},
                        f"SSN header hint and {share:.1%} of values are 9-digit SSNs")
    if "phone" in keys:
        share = _share(tv, is_phone)
        if share >= PHONE_SHARE:
            return _det(T.PHONE, ctx, {"match_share": round(share, 4)},
                        f"phone header hint and {share:.1%} of values are phone numbers")
    elif contact_header(ctx):  # R4-02: SMS, Text, Home, Primary ... holding phone numbers
        local = bare_short_phone_header(ctx.name)
        share = _share(tv, lambda v: contact_phone(v, local))
        if share >= PHONE_SHARE:
            return _det(T.PHONE, ctx, {"match_share": round(share, 4), "contact_header": True},
                        f"contact header and {share:.1%} of values are phone numbers (R4-02): fake numbers, "
                        "never real ones")
    if "zip" in keys:
        share = _share(tv, _fullmatch(ZIP_HINTED_RE))
        if share >= ZIP_SHARE:
            return _det(T.ZIP, ctx, {"match_share": round(share, 4)},
                        f"ZIP header hint and {share:.1%} of values are ZIP/ZIP+4 codes")
    ll = latlon_hint(ctx.name)
    if ll is not None:
        det = _latlon(ctx, ll)
        if det is not None:
            return det
    share = _share(tv, _fullmatch(ADDRESS_RE))
    if share >= ADDRESS_SHARE:
        return _det(T.STREET_ADDRESS, ctx, {"match_share": round(share, 4)},
                    f"{share:.1%} of values are street addresses (>= {ADDRESS_SHARE:.0%})")
    name_key = next((k for k in keys if k in NAME_HINT_SUBTYPES), None)
    if name_key is not None:
        share = _share(tv, name_like)
        if share >= NAME_SHARE:
            layouts = Counter()
            for v, c in tv.items():
                layouts[name_layout(v)] += c
            sub = NAME_HINT_SUBTYPES[name_key]
            return _det(T.PERSON_NAME, ctx, {"name_like_share": round(share, 4),
                                             "layouts": dict(layouts.most_common())},
                        f"person-name header hint '{name_key}' (D12; header hint only): fake names, never real ones",
                        subtype=sub)
    if "city" in keys:
        share = _share(tv, name_like)
        if share >= NAME_SHARE:
            return _det(T.CITY, ctx, {"name_like_share": round(share, 4)},
                        "city header hint: fake city names (D24), never real ones")
    if all(is_state(v) for v in tv):
        n_states = len({v.strip().rstrip(".").casefold() for v in tv})
        if "state" in keys or n_states >= 3:
            why = "state header hint" if "state" in keys else f"{n_states} distinct values"
            return _det(T.STATE, ctx, {"n_states": n_states},
                        f"every value is a US state code/name ({why}): real values kept by exact quota")
    # MULTI_SELECT items follow the D35 allowlist of the copy gate (rare items get same-shape pseudonyms)
    return _multi_select(ctx)


def multi_select_items(values: dict[str, int], d: str, na_tokens: Iterable[str] = ()) -> dict[str, int]:
    """Item -> number of rows holding it (stripped, case-folded; whole-value NA tokens are not items)."""
    na = frozenset(na_tokens)
    items: Counter = Counter()
    for v, c in values.items():
        if v.strip() in na:
            continue
        for p in {p.strip().casefold() for p in v.split(d) if p.strip()}:
            items[p] += c
    return dict(items)


def _latlon(ctx: DetectContext, t: ColumnType) -> Detection | None:
    det = numeric_detect.detect_numeric(ctx)
    if det is None or det.type is not T.FLOAT:
        return None
    lo, hi = float(det.params["min"]), float(det.params["max"])
    limit = 90.0 if t is T.LAT else 180.0
    if lo < -limit or hi > limit:
        return None
    params = dict(det.params)
    params["_min_cell"] = int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))
    word = "latitude" if t is T.LAT else "longitude"
    out = Detection(t, params=params, reason=f"{word} header hint and decimals within +/-{limit:.0f}")
    out.messages.append(Message(
        "warning", "lat_lon_independent",
        f"{word} generated as independent decimals inside the source bounding box: synthetic points need not be "
        "real places and are not paired with the other coordinate", column=ctx.name))
    return out


#: Share of the values (rows) of a delimited column that must have exactly two parts to make it a pair layout.
PAIR_SHARE = 0.9


def pair_layout(values: dict[str, int], d: str, na_tokens: Iterable[str] = ()) -> bool:
    """The delimiter splits each value into ONE pair, not a list of items: >= 90% of the values (rows) have exactly
    two parts, and either the delimiter is a comma (the ``Last, First`` layout: ``Nguyen, Maria``, ``Austin, TX``,
    also ``Scott, Ann`` / ``Lee, Scott`` where one name stands first in one value and second in another, hardening
    pass 2) or no part ever stands first in one value and second in another. A list varies in length (``ELL;SPED``,
    ``SPED``, ``Band;ELL;GT``) or repeats its items in any position.

    Such a column is not a MULTI_SELECT: splitting it would count each half as an item of its own, and two halves
    each held by many people would put back together a whole value that one person holds (D35 decides whole
    values; R4-01's layout is kept structurally, never by judging the parts as names). A comma list whose values all
    have two items loses nothing as whole values: each combination is decided, and written, as one level."""
    na = frozenset(na_tokens)
    total = two = 0
    first: set[str] = set()
    second: set[str] = set()
    for v, c in values.items():
        if v.strip() in na:
            continue
        parts = [p for p in (p.strip().casefold() for p in v.split(d)) if p]
        if not parts:
            continue
        total += c
        if len(parts) == 2:
            two += c
            first.add(parts[0])
            second.add(parts[1])
    return bool(total) and two >= PAIR_SHARE * total and (d == "," or not (first & second))


def _multi_select(ctx: DetectContext) -> Detection | None:
    """MULTI_SELECT: a delimiter in >= 30% of the values and short label items (design 3 #32). Never under an org-code
    header (D10 keeps an organisational code whole: the comma of ``ALG1-03 Garcia, Ana`` is part of the code, and
    split into items its teacher's surname and given name would each be kept for their counts, putting the full name
    back together; whole, D36 replaces the name: ``copy_gate_units``)."""
    h = ctx.hints
    if h.free_text or guarded_hint(ctx) is not None or (h.org_code and not h.person_id_like):
        return None
    vals = ctx.text_values
    total = sum(vals.values())
    if not total or len(vals) < 2:
        return None
    best: tuple[float, str] | None = None
    for d in MULTI_DELIMITERS:
        if d == "," and getattr(ctx.settings, "decimal", ".") == ",":
            continue
        share = sum(c for v, c in vals.items() if d in v) / total
        if share >= MULTI_SELECT_SHARE and (best is None or share > best[0]):
            best = (share, d)
    if best is None:
        return None
    share, d = best
    if d == "," and all(_NUMBER_WITH_COMMAS.fullmatch(v) for v in vals if "," in v):
        return None
    if pair_layout(vals, d, ctx.na_tokens):
        return None
    items: Counter = Counter()
    for v, c in vals.items():
        if v in ctx.na_tokens:
            continue  # a whole-value NA token ('None', 'N/A') is not an item; it is written back verbatim
        for p in (p.strip() for p in v.split(d)):
            if not p:
                continue
            if len(p) > MULTI_MAX_ITEM_LENGTH or len(p.split()) > MULTI_MAX_ITEM_WORDS:
                return None
            items[p.casefold()] += c
    if len(items) < 2 or len(items) > ctx.factor_max:
        return None
    if not any(any(ch.isalpha() for ch in it) for it in items):
        return None
    n_combos = ctx.casefold_distinct()
    mode = "combinations" if n_combos <= ctx.factor_max else "items"
    sep_name = {";": "semicolon", "|": "pipe", ",": "comma"}[d]
    det = _det(T.MULTI_SELECT, ctx, {"delimiter": d, "n_items": len(items), "n_combinations": n_combos,
                                    "mode": mode, "delimiter_share": round(share, 4),
                                    "_factor_max": ctx.factor_max},
               f"{sep_name} delimiter in {share:.1%} of values (>= {MULTI_SELECT_SHARE:.0%}); {len(items)} items, "
               f"{n_combos} combinations: " + ("combinations kept as a factor" if mode == "combinations"
                                               else "item-wise sampling (prevalence and order)"))
    return det


# --------------------------------------------------------------------------- R3-04 names by their values
# D35: value content (names, sentences) no longer decides what a column writes: the copy gate's allowlist does.
# :func:`value_name_detection`, :func:`name_screen_applies` and :func:`_thing_header` are kept for the header and
# path screens (D34), which cannot pseudonymise and still judge text by its content.
#: Detections a column of person names can fall to without a name header: real labels written verbatim
#: (factors, constants, flags, native ordered dictionaries; multi-selects whose comma splits ``Smith, J.``,
#: R4-01), or invented labels (high-cardinality text: as PERSON_NAME its real names also join the run-wide
#: full-name index, R1-04, so no other file writes them).
_NAME_BACKSTOP_TYPES = frozenset({T.FACTOR, T.CONSTANT, T.FLAG_SINGLE_TOKEN, T.HIGH_CARD_CATEGORICAL, T.ORDINAL,
                                  T.MULTI_SELECT})


def vocabulary_levels(det: Detection) -> bool:
    """The detection's real levels come from one of the tool's vocabularies or patterns (boolean pairs, ordinal
    scales, school years, US states, 4-digit years), so they quote nobody: the copy gate needs no screen."""
    t = det.type
    if t is T.ORDINAL:
        return bool(det.params.get("vocabulary"))  # an Arrow ordered dictionary has none: any text
    return t in (T.BOOLEAN, T.SCHOOL_YEAR, T.STATE, T.INTEGER_YEAR)


def name_screen_applies(ctx: DetectContext) -> bool:
    """The value-name screen (R3-04/R4-01) judges this column: no name, PII, free-text, org-code, ID, date,
    year, grade or code header hint, and no header of a thing named after people or places."""
    h = ctx.hints
    return not (h.pii or h.free_text or h.org_code or h.person_id_like or h.default_id or h.id_hint or h.date
                or h.time or h.duration or h.year or h.school_year or h.grade or h.code or latlon_hint(ctx.name)
                or _thing_header(ctx.name))


#: Last header words naming the PERSON tied to a thing (``Award_Recipient``, ``Team_Captain``, ``Club_President``,
#: ``Scholarship_Winner``): the thing word before them does not make the column a thing column (R4-01).
_PERSON_HEADS = frozenset({
    "recipient", "winner", "nominee", "honoree", "awardee", "holder", "owner", "member", "leader", "captain",
    "president", "champion", "author", "presenter", "speaker", "sponsor", "chair", "chairperson", "lead",
    "representative", "rep", "delegate", "officer", "manager", "director", "monitor", "buddy", "partner"})


def _thing_header(name: str) -> bool:
    """A header token naming a thing that borrows people's or places' names (``School``, ``Team_House``,
    ``Nombre_Escuela``, ``Month``): its values are never judged person names by themselves, unless the last
    word names the person tied to the thing (``Award_Recipient``, ``Team_Captain``: R4-01)."""
    toks = strip_role_trailers(header_tokens(name))
    if toks and len(toks) > 1:
        head = toks[-1]
        if head in _PERSON_HEADS or (head.endswith("s") and head[:-1] in _PERSON_HEADS) or is_role_word(head):
            return False
    for t in toks:
        if t in NAME_LIKE_THING_WORDS or (len(t) > 3 and t.endswith("s") and t[:-1] in NAME_LIKE_THING_WORDS):
            return True
    return False


def _name_like_beside_room(v: str) -> bool:
    """:func:`name_like` once a room number beside the name is set aside (``Smith 101``, R4-01)."""
    return name_like(name_values.strip_room(v))


def value_name_detection(ctx: DetectContext, det: Detection) -> Detection | None:
    """D12 value-validated person names (R3-04): PERSON_NAME for a column without a name, PII, free-text,
    org-code, ID, date, year, grade or code header hint whose values are person names
    (:func:`detect.names.name_value_verdict`), else None.

    Called by :func:`apply_pii_precedence` for (non-code) HIGH_CARD_CATEGORICAL and by the copy gate
    (:func:`copy_gate`, screen 2) for FACTOR, CONSTANT, FLAG_SINGLE_TOKEN, native (non-vocabulary) ORDINAL and
    MULTI_SELECT value/native detections (``Smith, J.`` splits on its comma, R4-01; ``Smith 101``: a room
    number beside the name is set aside, ``detect.names.strip_room``); headers of things named after people or places
    (``hints.NAME_LIKE_THING_WORDS``: school, team, house, street, month, course ...) are skipped, and so are
    org codes (D10 keeps their real values). The detection carries a ``value_names`` warning whose fix
    (``--force-factor COL``) keeps the real values when they are not personal data.
    """
    if det.type not in _NAME_BACKSTOP_TYPES or det.subtype == "code" or vocabulary_levels(det):
        return None
    if not name_screen_applies(ctx):
        return None
    tv = ctx.typed_values
    share = _share(tv, _name_like_beside_room) if tv else 0.0
    if share < NAME_SHARE:
        return None
    sub = name_values.name_value_verdict(tv)
    if sub is None:
        return None
    layouts: Counter = Counter()
    for v, c in tv.items():
        layouts[name_layout(v)] += c
    new = _det(T.PERSON_NAME, ctx, {"name_like_share": round(share, 4),
                                    "layouts": dict(layouts.most_common()), "value_validated": True},
               f"values are person names (known given/family names; D12 value-validated PII, R3-04): fake names, "
               f"never real ones (would otherwise be {det.type.value})", subtype=sub)
    new.source = det.source
    new.messages.append(Message(
        "warning", "value_names",
        f"the values look like person names although the header names no person: generated as fake names "
        f"(would otherwise be {det.type.value})", column=ctx.name,
        fix=f"--force-factor {ctx.name} keeps the real values (only if they are not personal data)"))
    return new


# --------------------------------------------------------------------------- D12 guard
def apply_pii_precedence(ctx: DetectContext, det: Detection) -> Detection:
    """D12 guard applied to every value-rule (and native) detection.

    Called by ``detect/types.py`` after the value rules picked ``det`` (never for
    explicit overrides such as ``--force-factor``). Returns ``det`` or a
    replacement:

    * HIGH_CARD_CATEGORICAL / FREE_TEXT: keep, noting ``params['pii_hint']`` (their values are never copied;
      D35: they are not judged by their content).
    * A value-copying detection (CONSTANT, FLAG_SINGLE_TOKEN, FACTOR, ORG_CODE,
      DIGIT_CODE; numbers for phone/SSN/DOB hints, and staff or student numbers under a person header,
      :func:`person_number_header`) with a PII or free-text
      header hint, and CONSTANT/FLAG/native detections whose values validate as
      never-copy PII (also numbers under a contact header that are phone numbers, R4-02): the AREA C
      semantic type when the values validate (:func:`detect_text_semantics`), else FREE_TEXT (free-text hints)
      or HIGH_CARD_CATEGORICAL with ``params['pii_hint']`` (invented labels keeping
      the distinct count and frequency profile). Real values are never copied.
    * Numbers (any number type: discrete levels and the exact min/max would be
      written) under a header that names a comment/note/alert/narrative column
      itself (:func:`free_text_column`: ``Notes``, ``Teacher_Comments``; not
      ``Note_Count``, ``Alert_Level``) or a phone (``hints.phone_header_hint``: ``SMS_Number``, ``Parent_Cell``,
      R4-02): same-shape masks (D12 over numeric typing, R3-01).

    Every detection that still writes real values verbatim then passes the copy
    gate (:func:`copy_gate`, R3/R4, D35).
    """
    h = ctx.hints
    if det.source not in ("value", "native"):
        return det
    hint = guarded_hint(ctx)  # every PII hint of the header counts ('city_state' is a city header too)
    if det.type in (ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.FREE_TEXT):
        if hint is not None:
            det.params["pii_hint"] = hint
        return det
    num_hint = guarded_hint(ctx, GUARDED_PII_NUMERIC)
    if num_hint is None and det.type in _COPYING_NUMERIC_TYPES and person_number_header(ctx):
        # staff or student numbers under Teacher / Case_Manager (D10: person IDs are never kept)
        num_hint = guarded_hint(ctx, PERSON_NAME_HINTS) or "person_number"
    guard_free = h.free_text is not None and det.type in _COPYING_TEXT_TYPES
    guard_pii = hint is not None and det.type in _COPYING_TEXT_TYPES
    guard_num = num_hint is not None and det.type in _COPYING_NUMERIC_TYPES
    free_num = h.free_text is not None and det.type in _COPYING_NUMERIC_TYPES and free_text_column(ctx)
    if guard_num and not guard_pii:
        hint = num_hint
    # the AREA C hook runs inside the value rules after CONSTANT/FLAG, and never for native detections
    hook_skipped = det.source == "native" or det.type in (ColumnType.CONSTANT, ColumnType.FLAG_SINGLE_TOKEN)
    zip_num = "zip" in pii_keys(ctx) and det.type in _COPYING_NUMERIC_TYPES
    phone_num = det.type in _COPYING_NUMERIC_TYPES and contact_header(ctx)  # R4-02: SMS numbers stored as numbers
    if hook_skipped and (det.type in _COPYING_TEXT_TYPES or guard_num or zip_num or phone_num):
        sem = detect_text_semantics(ctx)
        # never-copy PII, and ZIP codes (small-cell policy) stored as numbers or constants
        if sem is not None and (sem.type in NEVER_COPY_SEMANTIC or sem.type is ColumnType.ZIP):
            sem.source = det.source
            sem.reason = f"{sem.reason} (wins over {det.type.value}, D12)"
            sem.messages.append(Message("info", "pii_hint",
                                        f"personal data: real values are never copied (would otherwise be "
                                        f"{det.type.value})", column=ctx.name,
                                        fix=f"--force-factor {ctx.name} keeps real values (only if they are not "
                                            "personal data)"))
            return copy_gate(ctx, sem)  # stamps the ZIP codes it keeps (small-cell policy)
    if not (guard_free or guard_pii or guard_num or free_num):
        return copy_gate(ctx, det)
    if guard_free and not guard_pii:
        new = Detection(ColumnType.FREE_TEXT, params={"header_hint": h.free_text},
                        reason=f"free-text header hint '{h.free_text}' wins over {det.type.value} (D12): filler text")
    elif det.type in _COPYING_NUMERIC_TYPES or is_code_like(ctx.typed_values):
        # D13/D24: numbers and short single-token codes keep their shape as same-shape masks (digits stay
        # digits, so a numeric column still reads as numbers); invented labels would turn them into words
        params: dict[str, Any] = {"pii_hint": hint} if hint is not None else {"header_hint": h.free_text}
        if det.type in _COPYING_NUMERIC_TYPES:
            params["_numeric_mask"] = True
            if det.type in _NUMERIC_STORAGE_TYPES:
                params["numeric_storage"] = det.type.value
        what = f"PII header hint '{hint}'" if hint is not None else f"free-text header hint '{h.free_text}'"
        new = Detection(ColumnType.HIGH_CARD_CATEGORICAL, subtype="code", params=params,
                        reason=f"{what} wins over {det.type.value} (D12): same-shape masks, real values never copied")
    else:
        new = Detection(ColumnType.HIGH_CARD_CATEGORICAL, params={"pii_hint": hint},
                        reason=f"PII header hint '{hint}' wins over {det.type.value} (D12): invented labels, "
                               "real values never copied")
    new.params["_min_cell"] = int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))
    new.missing_policy = "typed"
    new.messages.append(Message("info", "pii_hint",
                                f"PII/free-text header hint: real values are never copied "
                                f"(would otherwise be {det.type.value})", column=ctx.name,
                                fix=f"--force-factor {ctx.name} keeps real values (only if they are not personal data)"))
    return new


def person_number_header(ctx: DetectContext) -> bool:
    """D10/D12 (hardening pass 2): numbers under this header are person numbers (staff or student numbers), never
    kept: a person-name header (``Teacher``, ``Counselor``, ``Student_Name``) whose last word names ONE person, or
    a header ending in a word for the person who holds a post (:data:`_PERSON_NUMBER_HEADS`: ``Case_Manager``,
    ``Club_Sponsor``). A plural role word counts people (``Students``, ``Teachers``: a count stays a number)."""
    toks = header_tokens(ctx.name)
    if toks and toks[0] in _COUNT_PREFIXES:
        return False
    core = strip_role_trailers(toks)
    if not core:
        return False
    last = core[-1]
    if last in _PERSON_NUMBER_HEADS:
        # hardening pass 3: a post word names an organisation as often as a person (Transportation_Provider,
        # Lunch_Provider, Officer): numbers of one or two digits are a code list there, not staff numbers
        return guarded_hint(ctx, PERSON_NAME_HINTS) is not None or not _small_codes(ctx.typed_values)
    if guarded_hint(ctx, PERSON_NAME_HINTS) is None:
        return False
    return not (is_role_word(last) and last not in _SINGULAR_ROLE_WORDS)


#: A number of one or two digits (``3``, ``12``, ``-1``, ``4.0``): a code under a post word (:func:`_small_codes`).
_SMALL_CODE_RE = re.compile(r"[+-]?\d{1,2}(?:\.0+)?")


def _small_codes(typed_values: dict[str, int]) -> bool:
    """Every value is a number of at most two digits: a code list (``Transportation_Provider`` 1-4), not staff
    numbers (:func:`person_number_header`)."""
    return bool(typed_values) and all(_SMALL_CODE_RE.fullmatch(v.strip()) for v in typed_values)


#: Person-role words as listed (singular): ``hints.is_role_word`` also accepts their plurals.
_SINGULAR_ROLE_WORDS = frozenset(PERSON_ROLE_WORDS)
#: Last header words for the person who holds a post: numbers under them are staff numbers (Case_Manager = 42069).
#: Their text values are not personal by header (D35 decides them value by value): this list is for numbers only.
_PERSON_NUMBER_HEADS = frozenset({"manager", "director", "coordinator", "captain", "president", "officer",
                                  "representative", "sponsor", "leader", "owner", "liaison", "specialist",
                                  "clinician", "provider", "administrator", "worker"})


#: Last header words that name a comment/note/alert/narrative column itself (:func:`free_text_column`).
_FREE_TEXT_HEADS = frozenset({"comment", "comments", "note", "notes", "alert", "alerts", "narrative", "narratives"})
#: First header words of a count, share or flag (``Num_Notes``, ``Total_Alerts``, ``Has_Comments``).
_COUNT_PREFIXES = frozenset({"is", "has", "was", "num", "no", "n", "nbr", "number", "count", "cnt", "total", "pct",
                             "percent", "qty"})


def free_text_column(ctx: DetectContext) -> bool:
    """The header names a comment/note/alert/narrative column itself: its last word, numbering ignored
    (``Notes``, ``Teacher_Notes``, ``MedicalAlert``, ``Comments_2``), not a count, level, date, flag or ID of
    one (``Note_Count``, ``Num_Notes``, ``Alert_Level``, ``NoteDate``, ``NoteID``). Such a column's numbers
    are never copied (D12 wins over numeric typing, R3-01)."""
    toks = header_tokens(ctx.name)
    core = strip_role_trailers(toks)
    return bool(core) and core[-1] in _FREE_TEXT_HEADS and toks[0] not in _COUNT_PREFIXES


# --------------------------------------------------------------------------- copy gate (R3)
#: Personal-data shapes searched INSIDE values that would be written verbatim (:func:`pii_shapes`).
#: R8-02: an e-mail address in any script, broad on purpose (fail-safe): a local part without a space or ``@``, an
#: ``@`` and a domain of two or more dot-separated labels (``josé@school.org``, ``alice@müller.de``,
#: ``张伟@school.org``, ``ana@xn--mller-kva.de``); :func:`_has_email` also wants a letter that starts its last label.
EMB_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@.]+(?:\.[^\s@.]+)+")
EMB_URL_RE = re.compile(r"(?<![\w.@\-])(?:(?:https?|ftp)://[^\s/?#]+|www\.[^\s/?#]+\.[A-Za-z]{2,})", re.I)
#: North American layouts with separators: ``312-555-0199``, ``(312) 555-0199``, ``+1 312.555.0199``.
EMB_PHONE_RE = re.compile(r"(?<![\w+])(?:\+?1[\s.\-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\w)")
#: International layouts: ``+44 20 7946 0958`` (8-15 digits, checked by :func:`_intl_phone`).
EMB_INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.\-]?\(?\d{1,5}\)?){2,6}(?!\w)")
_SP = " \u00a0\u2009\u202f"  # spaces inside a number: space, no-break, thin and narrow no-break spaces
_SEP = _SP + r"./\-\u2010-\u2013"  # spaces, dots, slashes and dashes (any hyphen) join the digit groups of a number
#: R9-02: the most characters that join two digit groups of ONE number: any run of spaces, dots, slashes and dashes,
#: spaced or doubled (``020  7946  0958``, ``020 - 7946 - 0958``, ``123 . 45 . 6789``), not a wider gap.
SEP_RUN_MAX = 5
_SEP_RUN = rf"[{_SEP}]{{1,{SEP_RUN_MAX}}}"
#: R9-02: the SSN layout 3-2-4 with ANY separator run between the groups (``123-45-6789``, ``123 45 6789``,
#: ``123.45.6789``, ``123/45/6789``, ``123 - 45 - 6789``; one shape test for the copy gate and the generators,
#: ``generate.embedded.pii_spans``).
EMB_SSN_RE = re.compile(rf"(?<![\w\-\u2010-\u2013])\d{{3}}{_SEP_RUN}\d{{2}}{_SEP_RUN}\d{{4}}(?![\w\-\u2010-\u2013])")
#: A digit-only phone number (10 digits, or 11 with a leading 1 or 0), possibly stored as a float (``.0``):
#: personal only under a person/contact/notes header (:func:`digit_phone_header`).
DIGIT_PHONE_RE = re.compile(r"\+?[01]?\d{10}(?:\.0+)?")
#: A digit-only North American number (area code and exchange 2-9, an optional leading 1): a column of them is a
#: column of phone numbers under any header that does not say otherwise (``Ph``, ``Number``; not ``nces_id`` or
#: ``amount``: :func:`not_phone_header`) when they are at least :data:`NANP_SHARE` of its distinct numbers
#: (hardening pass 2; Unix times start with 1, IDs rarely keep both 2-9 digits).
NANP_DIGIT_PHONE_RE = re.compile(r"\+?1?[2-9]\d\d[2-9]\d{6}(?:\.0+)?")
NANP_SHARE = 0.9
#: The same digits standing alone inside words (``Call mom 3125550199``): a phone number in any column.
EMB_BARE_PHONE_RE = re.compile(r"(?<![\w.,+\-/])\+?[01]?\d{10}(?![\w.,\-/])")
#: Street suffixes that make a house number + words a street address inside other text (a strict subset of
#: :data:`STREET_SUFFIXES`: ``Run``, ``Path``, ``Loop``, ``Row`` ... also end ordinary labels).
_EMB_ADDRESS_SUFFIXES = ("street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd", "drive", "dr", "lane",
                         "ln", "court", "ct", "parkway", "pkwy", "highway", "hwy", "circle", "cir", "terrace", "ter",
                         "place", "pl", "way", "trail", "trl")
EMB_ADDRESS_RE = re.compile(
    r"\d{1,6}[A-Za-z]?\s+(?:[A-Za-z0-9][\w'.\-]*\s+){1,4}(?:"
    + "|".join(sorted(_EMB_ADDRESS_SUFFIXES, key=len, reverse=True)) + r")\.?(?=$|[\s,.;:)])", re.I)
#: Where an embedded street address may start: the value's start, after a separator, or after a word that
#: introduces a place (``lives at 12 Oak St``); never after ``Grade``/``Room``/``Level`` numbers.
_EMB_ADDRESS_START = re.compile(r"(?:^|[:;|(\n,]|\b(?:at|on|to|from|near|address|addr|lives|located|meet|visit)\b)"
                                r"\s*(?=\d)", re.I)
#: K-09: in an organisational code an address may also start after a spaced dash (``Homebound - 1427 Juniper Ridge
#: Rd`` under ``School_Name``; :func:`org_code_address`). Only there: a code is kept whatever its count, so its shape is
#: its only guard and a false alarm costs one code's pseudonym; elsewhere a level is kept only when >= --min-cell people
#: hold it, and the header screen (D34) refuses a file on a shape, so a gradebook's ``Unit 1 - 3 Digit Place Value`` or
#: ``Science - 3 Milky Way`` is no address there (K09-DASH-FP).
_ORG_ADDRESS_START = re.compile(r"\s[-‐-―]{1,2}\s*(?=\d)")
#: Header words that make a digit-only 10/11-digit number a phone number besides the PII/free-text hints.
_CONTACT_WORDS = ("emergency", "emerg")
_GATE_TEXT_TYPES = _COPYING_TEXT_TYPES | {ColumnType.ORDINAL}
# --------------------------------------------------------------------------- D33: sentence-like text
# D35 retired the D33 rule for values (a sentence-like level is pseudonymised when it is rare, kept when it is
# shared by >= --min-cell entities); :func:`is_sentence_like` stays for the header and path screens (D34).
#: Lowercase function words that mark a clause rather than a label (D33): articles and demonstratives,
#: pronouns and possessives, the prepositions of instructions and narratives, subordinators and auxiliaries.
#: Left out on purpose, because they join the nouns of ordinary labels: ``and``, ``or``, ``nor``, ``but``,
#: ``of``, ``than``, ``not``, ``per``, ``vs``, ``via`` and quantifiers (``some``, ``all``, ``no``, ``once``):
#: ``Extended time on tests and quizzes``, ``Neither agree nor disagree``, ``Less than once a month``,
#: ``Does not meet grade level expectations`` stay labels.
SENTENCE_FUNCTION_WORDS: frozenset[str] = frozenset((
    # articles and demonstratives
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns, possessives and relatives
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "himself", "herself", "themselves", "who", "whom", "whose", "which",
    # prepositions
    "to", "at", "after", "before", "with", "without", "for", "from", "by", "on", "in", "into", "onto", "upon",
    "about", "during", "until", "since", "through", "between", "within", "toward", "towards", "near", "around",
    # subordinators
    "when", "while", "if", "because", "unless", "although", "whether", "where",
    # auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "will", "would", "should",
    "can", "could", "may", "might", "must", "does", "did", "do",
))
#: Words a Title Case label leaves lowercase (headline style: articles, coordinating conjunctions and
#: prepositions, plus name particles). Pronouns, auxiliaries and subordinators are capitalized in Title Case,
#: so a lowercase ``his``/``is``/``when`` makes a value sentence case.
TITLE_MINOR_WORDS: frozenset[str] = frozenset((
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "as", "at", "by", "in", "of", "off", "on",
    "onto", "out", "per", "to", "up", "via", "vs", "with", "without", "from", "into", "upon", "over", "under",
    "after", "before", "about", "during", "until", "since", "through", "between", "within", "toward", "towards",
    "near", "around", "than", "de", "la", "del", "des", "du", "le", "van", "von", "y", "en",
))
SENTENCE_MIN_WORDS = 6
SENTENCE_MIN_FUNCTION_WORDS = 2
_WORD_EDGE_PUNCT = "\"'`()[]{}<>.,;:!?\u2018\u2019\u201c\u201d\u00ab\u00bb"


def _bare_words(v: str) -> list[str]:
    """Whitespace words with their surrounding punctuation stripped; words without a letter or digit dropped."""
    out = []
    for w in v.split():
        w = w.strip(_WORD_EDGE_PUNCT)
        if any(ch.isalnum() for ch in w):
            out.append(w)
    return out


def is_title_case(v: str) -> bool:
    """Headline-style Title Case: every word that starts with a letter is capitalized unless it is a minor
    word (:data:`TITLE_MINOR_WORDS`: articles, conjunctions, prepositions) written in lowercase; at least one
    word is capitalized. ``Introduction to the History of the United States`` is Title Case; ``Return
    medication to the office`` and ``Call His Mother when the Bus Is Late`` are not."""
    capital = False
    for w in _bare_words(v):
        first = w[0]
        if not first.isalpha():
            continue  # '2nd', '504'
        if first.isupper():
            capital = True
        elif w not in TITLE_MINOR_WORDS:
            return False
    return capital


def is_sentence_like(v: str) -> bool:
    """D33: a level that reads like a sentence: >= 6 words, >= 2 lowercase function words
    (:data:`SENTENCE_FUNCTION_WORDS`, counted per occurrence) and not Title Case (:func:`is_title_case`).

    ``Return medication to the nurse after the second class`` is; ``Introduction to the History of the
    United States`` (Title Case), ``Extended time on tests and quizzes`` (one function word) and ``some high
    school`` (3 words) are not. All-caps text has no lowercase function words, so it is never sentence-like.
    """
    words = _bare_words(v)
    if len(words) < SENTENCE_MIN_WORDS:
        return False
    if sum(1 for w in words if w in SENTENCE_FUNCTION_WORDS) < SENTENCE_MIN_FUNCTION_WORDS:
        return False
    return not is_title_case(v)


def sentence_levels(values: dict[str, int]) -> tuple[int, int]:
    """(number of distinct levels that are sentence-like, rows holding them) over ``values`` (value -> count)."""
    n = rows = 0
    for v, c in values.items():
        if is_sentence_like(v):
            n += 1
            rows += c
    return n, rows


def _intl_phone(v: str) -> bool:
    return any(8 <= sum(ch.isdigit() for ch in m.group()) <= 15 for m in EMB_INTL_PHONE_RE.finditer(v))


def _has_email(v: str) -> bool:
    """R8-02: ``v`` holds an e-mail address (:data:`EMB_EMAIL_RE`) whose domain's last label starts with a letter of
    any script and has two or more characters (``5@3.50`` is a price, ``v2@1.0`` a version)."""
    for m in EMB_EMAIL_RE.finditer(v):
        last = m.group().rsplit(".", 1)[1].rstrip(_WORD_EDGE_PUNCT)
        if len(last) >= 2 and last[0].isalpha():
            return True
    return False


# --------------------------------------------------------------------------- R8-01: phone numbers in any layout
#: R8-01, R9-02: a run of two or more digit groups that may be ONE phone number in any national layout (``020 7946
#: 0958``, ``(020) 7946 0958``, ``07911 123456``, ``01 23 45 67 89``, ``030/1234567``, ``55 1234 5678``,
#: ``0917-123-4567``): each group joined to the next by a separator run (:data:`_SEP_RUN`: any short run of spaces,
#: dots, slashes and dashes, so spaced and doubled separators too: ``020 - 7946 - 0958``, ``020  7946  0958``) or set
#: in parentheses (spaces inside allowed), a leading ``+`` allowed. A comma, a letter or a wider gap ends the run; a
#: run glued to a word, a number, a clock time or a sign (``AB12 345``, ``-312 555``, ``3.14 159``,
#: ``19:23:46.721597``) or followed by a decimal tail (``123 456,78``) is none. :func:`phone_runs` keeps the runs (and,
#: in a run too long for one number, the parts) whose digits and groups are a phone number's.
PHONE_RUN_RE = re.compile(
    rf"(?<![\w+/\-\u2010-\u2013])(?<!\d[.,:])(?P<plus>\+[{_SP}]?)?"
    rf"(?:\([{_SP}]*\d{{1,6}}[{_SP}]*\)(?:{_SEP_RUN})?|\d{{1,8}}{_SEP_RUN})+"
    rf"(?:\([{_SP}]*\d{{1,6}}[{_SP}]*\)|\d{{1,8}})(?![\w/\-\u2010-\u2013]|[.,]\d{{1,2}}(?!\d))")
_DIGIT_GROUP_RE = re.compile(r"\d+")
#: The fewest and most digits of a phone number in a run (a leading trunk ``0`` not counted; E.164 has at most 15).
PHONE_RUN_MIN, PHONE_RUN_MAX = 9, 15
#: Currency signs that make a digit run beside them an amount (``$1 234 567 890``, ``123.456.789 €``).
_CURRENCY = frozenset("$€£¥₹₩₦¢₱₫₺₽")


def _year(g: str) -> bool:
    return len(g) == 4 and 1900 <= int(g) <= 2100


def _clock(g: str) -> bool:
    """``HHMM`` or ``HHMMSS``."""
    return len(g) in (4, 6) and int(g[:2]) <= 23 and int(g[2:4]) <= 59 and (len(g) == 4 or int(g[4:]) <= 59)


def _date_groups(groups: list[str]) -> bool:
    """R8-01: digit groups that read as dates, years and times, not a phone number: a year, a ``YYYYMMDD`` date or a
    ``YYYYMM`` month among groups of at most two digits and clock times (``2023 12 15 10 30``, ``2022-23 2023-24``,
    ``2021 2022 2023``, ``20230915 1430``, ``20250115_152233``), or a clock time with its fraction of a second
    (``01.45.07.123000``)."""
    if 3 <= len(groups) <= 4 and all(len(g) == 2 for g in groups[:3]) and int(groups[0]) <= 23 \
            and int(groups[1]) <= 59 and int(groups[2]) <= 59 and (len(groups) == 3 or len(groups[3]) >= 3):
        return True
    dated = False
    for g in groups:
        if _year(g) or (len(g) in (6, 8) and _year(g[:4]) and 1 <= int(g[4:6]) <= 12
                        and (len(g) == 6 or 1 <= int(g[6:]) <= 31)):
            dated = True
        elif not (len(g) <= 2 or _clock(g)):
            return False
    return dated


def trunk_zero(groups: list[str]) -> bool:
    """The first digit group opens with a national trunk ``0`` (``020``, ``07911``, ``01``; a lone ``0`` is zero)."""
    return len(groups[0]) >= 2 and groups[0][0] == "0"


#: A dash or slash with spaces around it (``0 - 59``): the separator of ONE range (:func:`_ranges`, R9-02).
_SPACED_RANGE_SEP = re.compile(rf"[{_SP}]*([\-\u2010-\u2013/])[{_SP}]*")


def _ranges(s: str) -> bool:
    """Space-separated ranges ``a-b`` with ``a < b`` (``0-59 60-69 70-79``, ``0 - 59  60 - 69``): score bands, not a
    phone number."""
    chunks = _SPACED_RANGE_SEP.sub(r"\1", s).split()
    if len(chunks) < 2:
        return False
    for c in chunks:
        parts = re.split(r"[\-\u2010-\u2013/]", c)
        # K-21: isdecimal(), not isdigit(): int() refuses the non-decimal digits isdigit() accepts (superscript two)
        if len(parts) != 2 or not all(p.isdecimal() for p in parts) or int(parts[0]) >= int(parts[1]):
            return False
    return True


def _phone_run(v: str, m: re.Match) -> bool:
    """R8-01: a run of :data:`PHONE_RUN_RE` is a phone number unless its digits and groups say otherwise (broad on
    purpose: a false alarm only costs a pseudonym, D35): it has :data:`PHONE_RUN_MIN` to :data:`PHONE_RUN_MAX`
    digits (a leading trunk ``0`` not counted, :func:`trunk_zero`), no one-digit group after the first (``1 2 3 4 5 6
    7 8 9``, ``10.0.19041.1234``) and it is no SSN layout (an SSN shape already), decimal number (``1234567.89``),
    ZIP+4 code (``12345-6789``), range or list of ranges (``10000-20000``, ``0-59 60-69 70-79``), count (``01 02 03 04
    05``), year and number (``2023 1500000``), run of dates, years and times (:func:`_date_groups`), IPv4 address or
    amount (a currency sign beside it)."""
    s = m.group()
    groups = _DIGIT_GROUP_RE.findall(s)
    plus = m.group("plus") is not None
    n = sum(map(len, groups)) - (1 if not plus and trunk_zero(groups) else 0)
    if not PHONE_RUN_MIN <= n <= PHONE_RUN_MAX or any(len(g) == 1 for g in groups[1:]):
        return False
    if plus or "(" in s:
        return True  # international notation, an area code in parentheses
    if [len(g) for g in groups] == [3, 2, 4]:
        return False  # the SSN layout (``123-45-6789``, ``123 45 6789``): an SSN shape already (EMB_SSN_RE)
    if len(groups) == 2:
        sep = s[len(groups[0]):len(s) - len(groups[1])].strip(_SP)
        if sep in (".", ","):
            return False  # a decimal number
        if sep:  # a dash or a slash
            a, b = groups
            if (len(a), len(b)) == (5, 4) or (len(a) == len(b) and int(a) <= int(b)):
                return False  # a ZIP+4 code, a range
    nums = [int(g) for g in groups]
    if len({len(g) for g in groups}) == 1 and all(b == a + 1 for a, b in zip(nums, nums[1:])):
        return False  # a count (``01 02 03 04 05``, ``100 101 102``)
    if len(groups) == 2 and _year(groups[0]):
        return False  # a year and a number (``2023 1500000``, ``Class of 2023 123456``)
    if _ranges(s) or _date_groups(groups):
        return False
    if len(groups) == 4 and "." in s and all(len(g) <= 3 and int(g) <= 255 for g in groups):
        return False  # an IPv4 address
    before, after = v[:m.start()].rstrip(_SP), v[m.end():].lstrip(_SP)
    return not ((before and before[-1] in _CURRENCY) or (after and after[0] in _CURRENCY))  # an amount


class _Part:
    """A part of a :data:`PHONE_RUN_RE` run (consecutive digit groups of it), read like a match: ``start()``,
    ``end()``, ``group()`` and ``group("plus")``."""

    __slots__ = ("_v", "_s", "_e", "_plus")

    def __init__(self, v: str, s: int, e: int, plus: str | None) -> None:
        self._v, self._s, self._e, self._plus = v, s, e, plus

    def start(self) -> int:
        return self._s

    def end(self) -> int:
        return self._e

    def group(self, name: int | str = 0) -> str | None:
        return self._plus if name == "plus" else self._v[self._s:self._e]


#: A digit group of a run, an area code in parentheses included (:func:`_parts`).
_RUN_UNIT_RE = re.compile(rf"\([{_SP}]*\d+[{_SP}]*\)|\d+")
#: The most digit groups one phone number spans (15 digits, no one-digit group after the first).
_PART_MAX_GROUPS = 8


def _parts(v: str, m: re.Match) -> list[_Part]:
    """R9-02: the phone numbers among the parts of a run too long for ONE number: with any run of separators
    joining the groups, two numbers side by side (``07911 123456 / 07700 900123``) are one run of 22 digits, so its
    consecutive groups are tested on their own (:func:`_phone_run`), and the parts that together cover the most
    digits are kept (the fewest parts on a tie; a deterministic dynamic program over at most
    :data:`_PART_MAX_GROUPS` groups per part)."""
    base = m.start()
    units = [(base + u.start(), base + u.end()) for u in _RUN_UNIT_RE.finditer(m.group())]
    digits = [sum(ch.isdigit() for ch in v[s:e]) for s, e in units]
    plus = m.group("plus")
    # best[i]: (digits covered, -parts) and the parts, over the first i groups
    best: list[tuple[tuple[int, int], list[_Part]]] = [((0, 0), [])]
    for i in range(1, len(units) + 1):
        top = best[i - 1]
        for s in range(max(0, i - _PART_MAX_GROUPS), i - 1):
            part = _Part(v, units[s][0] if s else m.start(), units[i - 1][1], plus if s == 0 else None)
            score = (best[s][0][0] + sum(digits[s:i]), best[s][0][1] - 1)
            if score > top[0] and _phone_run(v, part):
                top = (score, best[s][1] + [part])
        best.append(top)
    return best[-1][1]


def phone_runs(v: str) -> list[re.Match | _Part]:
    """R8-01: the runs of digit groups in ``v`` that are a phone number in any national layout (:func:`_phone_run`);
    in a run with more digits than one number holds, its parts that are (:func:`_parts`, R9-02)."""
    out: list[re.Match | _Part] = []
    for m in PHONE_RUN_RE.finditer(v):
        if _phone_run(v, m):
            out.append(m)
            continue
        groups = _DIGIT_GROUP_RE.findall(m.group())
        if sum(map(len, groups)) - (1 if m.group("plus") is None and trunk_zero(groups) else 0) > PHONE_RUN_MAX:
            out.extend(_parts(v, m))
    return out


def _has_address(v: str) -> bool:
    return any(EMB_ADDRESS_RE.match(v, m.end()) for m in _EMB_ADDRESS_START.finditer(v))


def org_code_address(text: str) -> bool:
    """K-09: an organisational-code value holds a street address after a spaced dash (``Homebound - 1427 Juniper Ridge
    Rd``; :data:`_ORG_ADDRESS_START`), an underscore counting as a space. :func:`unit_shapes` finds every other
    address; this wider start applies to organisational codes only (:func:`unit_kept`)."""
    if not any(ch.isdigit() for ch in text):
        return False
    variants = (text, text.replace("_", " ")) if "_" in text else (text,)
    return any(EMB_ADDRESS_RE.match(v, m.end()) for v in variants for m in _ORG_ADDRESS_START.finditer(v))


def pii_shapes(value: str, digit_phone: bool = False) -> frozenset[str]:
    """Kinds of personal data found inside ``value``: ``email``, ``url``, ``phone``, ``ssn``, ``address``.

    Embedded forms count (``Call Ana at 312-555-0199``, ``Call mom 3125550199``, ``Front Office;ana@x.edu``).
    ``digit_phone`` (a person/contact/notes header) also makes a whole digit-only 10/11-digit value a phone
    number. Broad on purpose (R8-01, R8-02, R9-02): an e-mail address in any script (:func:`_has_email`), a phone
    number in any national layout and with any separator run (:func:`phone_runs`: ``020 7946 0958``, ``07911 123456``,
    ``01 23 45 67 89``, ``020 - 7946 - 0958``) and the SSN layout with any separators (:data:`EMB_SSN_RE`:
    ``123.45.6789``). (:func:`unit_shapes` adds bare digit-only numbers: :func:`bare_phone`.)
    """
    v = value.strip()
    found: set[str] = set()
    if "@" in v and _has_email(v):
        found.add("email")
    if ("://" in v or "www." in v.casefold()) and EMB_URL_RE.search(v):
        found.add("url")
    if any(ch.isdigit() for ch in v):
        if (EMB_PHONE_RE.search(v) or ("+" in v and _intl_phone(v))
                or (any(ch.isalpha() for ch in v) and EMB_BARE_PHONE_RE.search(v)) or phone_runs(v)):
            found.add("phone")
        if EMB_SSN_RE.search(v):
            found.add("ssn")
        if _has_address(v):
            found.add("address")
        if digit_phone and DIGIT_PHONE_RE.fullmatch(v):
            found.add("phone")
    return frozenset(found)


def nanp_digits(d: str) -> bool:
    """Ten digits that are a valid North American number (R6-01): area code and exchange start with 2-9 and are no
    ``N11`` service code, and the area code is no reserved ``N9X`` code (so the sentinel ``9999999999`` is none, and
    neither is a Unix time, which starts with 1)."""
    return (len(d) == 10 and d.isdigit() and d[0] not in "01" and d[3] not in "01" and d[1:3] != "11"
            and d[4:6] != "11" and d[1] != "9")


#: A whole digit-only number of 10 digits (11 with a leading 1, a ``+`` allowed), possibly stored as a float (``.0``).
_BARE_NUMBER_RE = re.compile(r"\+?1?(\d{10})(?:\.0+)?")


def trunk_digits(d: str) -> bool:
    """R9-02: 10 or 11 digits opening with ONE national trunk ``0`` (``07911123456``, ``02079460958``,
    ``0298765432``, ``0612345678``): a national phone number written without separators. Two leading zeros
    (``0012345678``) read as zero padding, not a trunk prefix."""
    return len(d) in (10, 11) and d[0] == "0" and d[1] != "0" and d.isdigit()


def bare_phone(text: str) -> bool:
    """R6-01: ``text`` holds a digit-only phone number, WHATEVER the column's header: the whole value
    (``3125550187``, ``13125550187``, ``+13125550187``, ``3125550187.0``) or a 10/11-digit run standing alone in it
    (``3125550187, 3125550199``), its ten digits a valid North American number (:func:`nanp_digits`) or (R9-02) a
    national number with its trunk ``0`` (:func:`trunk_digits`: ``07911123456``, ``02079460958``)."""
    s = text.strip()
    if len(s) < 10:
        return False
    m = _BARE_NUMBER_RE.fullmatch(s)
    if m is not None and nanp_digits(m.group(1)):
        return True
    if trunk_digits(s):
        return True
    for m in EMB_BARE_PHONE_RE.finditer(s):
        d = _NON_DIGIT.sub("", m.group())
        if trunk_digits(d) and m.group()[0] != "+":
            return True
        if len(d) == 11 and d[0] == "1":
            d = d[1:]
        if nanp_digits(d):
            return True
    return False


def unit_shapes(text: str, digit_phone: bool = False) -> frozenset[str]:
    """THE personal-data shape test of one real text unit (D35) and of a header, sheet, file or folder name (D34,
    ``privacy.headers``, R7-03): :func:`pii_shapes` plus a digit-only phone number under ANY header (:func:`bare_phone`:
    a North American number, R6-01, or one with its national trunk ``0``, R9-02), an underscore counting as a space
    (``Office_3125550187``, ``123_45_6789``,
    ``12_Oak_Street``). A person/contact/notes header (``digit_phone``) only widens the test (any whole 10/11-digit
    unit is a phone number there); no header narrows it, so the shapes of a unit never depend on the column it sits
    in. Broad and fail-safe (R8-01, R8-02): an e-mail address in any script, a phone number in any national layout
    (:func:`phone_runs`); a false alarm only costs a pseudonym. (D34 narrows only a phone shape of a name, and only
    when every one is an excused bare run of digits or a non-North-American run after a word of a numbered thing or
    reading as a list of numbers: ``privacy.headers._phone_in_name``.)"""
    variants = (text, text.replace("_", " ")) if "_" in text else (text,)
    kinds = frozenset().union(*(pii_shapes(v, digit_phone) for v in variants))
    if "phone" not in kinds and any(ch.isdigit() for ch in text) and any(bare_phone(v) for v in variants):
        kinds = kinds | {"phone"}
    return kinds


def symbol_only(text: str) -> bool:
    """A unit without a single letter or digit (``✓``, ``✗``, ``+``, ``—``, an emoji; compatibility forms folded
    first, so circled or full-width letters count as letters): it can name, number or address no one, and its only
    same-shape text is itself (a pseudonym keeps punctuation and symbols), so the D35 allowlist keeps it (R6-04)."""
    s = text.strip()
    if not s:
        return False
    if not s.isascii():
        s = unicodedata.normalize("NFKC", s)
    return not any(ch.isalnum() for ch in s)


def digit_phone_header(ctx: DetectContext) -> bool:
    """A person, phone/contact (``hints.phone_header_hint``: ``SMS``, ``Text``, ``Home``, ``Phone_Type`` ...;
    R4-02) or notes header: there a digit-only 10/11-digit number is a phone number."""
    if guarded_hint(ctx) is not None or ctx.hints.free_text is not None or phone_header_hint(ctx.name):
        return True
    return any(t.startswith(w) for t in header_tokens(ctx.name) for w in _CONTACT_WORDS)


def _screen(values: Iterable[str], digit_phone: bool,
            ignore: frozenset[str] = frozenset()) -> tuple[frozenset[str], int]:
    """(kinds of personal data found, number of values holding any) over ``values``; ``ignore``d kinds
    do not count."""
    kinds: set[str] = set()
    n = 0
    for v in values:
        k = pii_shapes(v, digit_phone) - ignore
        if k:
            kinds |= k
            n += 1
    return frozenset(kinds), n


#: Params key the copy gate stamps on a detection whose generator writes real values verbatim: ``"allowlist"``
#: (real text units decided one by one by the D35 allowlist), ``"vocabulary"`` (levels of a tool vocabulary),
#: ``"numbers"``, ``"dates"`` or ``"zip_policy"`` (their own rules). PRIVATE (``_``).
COPY_GATE_KEY = "_copy_gate"
#: Sources of explicit overrides that keep real values on the user's word (never screened): ``--force-factor``
#: (D35's explicit opt-in) and ``--pii-col`` (fake values of the named kind).
EXPLICIT_COPY_SOURCES = frozenset({"force-factor", "pii-col"})
#: Sources of explicit overrides that name a TYPE, not the real values: ``--force-type COL=TYPE`` and ``--org-code``
#: (D10). Their real text units still follow the D35 allowlist (:func:`copy_gate`): an organisational code keeps its
#: values (except personal-data shapes), a forced factor, ordinal, multi-select or token column keeps the safe ones.
GATED_EXPLICIT_SOURCES = frozenset({"force-type", "org-code"})


def copies_real_values(p: Any) -> bool:
    """The column's generator writes real source values verbatim (a profile or a detection): levels (constants,
    flags, booleans, factors, ordinals, school years, org codes, discrete integers, digit codes kept as levels),
    numbers (the exact min/max of D14, discrete levels), dates (exact extremes, sentinels, placeholder tokens),
    numeric tokens, multi-select items and combinations, kept ZIP codes and states. (A column whose units are all
    pseudonymised by D35 still counts: :func:`pseudonymized` tells which units are written.) False for the
    never-copy types (IDs, invented labels, filler text, fake PII, same-shape masks), and empty, excluded or null
    columns."""
    if getattr(p, "role", "row") in ("excluded", "null"):
        return False
    t = p.type
    if t in NEVER_COPY_TYPES or t in (T.CITY, T.EMPTY, T.BINARY_NESTED):
        return False
    params = p.params or {}
    return not (t is T.DIGIT_CODE and not params.get("levels_mode"))


def copy_cleared(p: Any) -> bool:
    """The real values a column writes verbatim passed the copy gate (``params[COPY_GATE_KEY]``, with its D35
    decision made: :data:`PENDING_KEY` gone), or an explicit override keeps them on the user's word
    (:data:`EXPLICIT_COPY_SOURCES`).

    One source of truth for the gate and the leak scanner (``privacy.leaks``): the scanner excuses a real level
    (in the report and docs, and in the data scan) only for a column whose values this gate cleared, and never
    a unit the gate pseudonymised (:func:`pseudonymized`), so a path that writes real values without passing the
    gate is never excused (R4, D35)."""
    if getattr(p, "source", None) in EXPLICIT_COPY_SOURCES:
        return True
    params = p.params or {}
    return bool(params.get(COPY_GATE_KEY)) and not params.get(PENDING_KEY)


def copy_gate(ctx: DetectContext, det: Detection) -> Detection:
    """The copy gate (R3, R4, D35): the one decision of which real values a detection writes verbatim.

    Called by :func:`apply_pii_precedence` for every value/native detection that the D12 header guard left
    unchanged, and by ``detect.types`` for ``--force-type`` and ``--org-code`` (:data:`GATED_EXPLICIT_SOURCES`: the
    flag names the type, not the values, so the type stays and its text units follow the allowlist; only
    ``--force-factor`` keeps every real value). It is type-agnostic: every path whose output holds real source values
    is listed here with its rule.

    ==============================================  =================================================================
    values written verbatim                         rule
    ==============================================  =================================================================
    FACTOR / CONSTANT / FLAG_SINGLE_TOKEN levels    D35 allowlist per level (:func:`unit_kept`): kept when a word of
    (value rules, Arrow dictionaries, Stata value   the built-in vocabularies (``detect.vocab``) or held by
    labels), ORDINAL levels outside the built-in    >= ``--min-cell`` distinct entities (IDs, or rows without an ID)
    vocabularies, DIGIT_CODE levels                 with no personal-data shape; every other level gets a same-shape
                                                    pseudonym (``generate.pseudonyms``)
    ORG_CODE levels                                 kept (D10), except personal-data shapes (e-mail, phone, SSN, URL,
                                                    street address: K-09) and, under a personal header (``--org-code
                                                    Teacher``), every level (pseudonymised); a level holding a person's
                                                    name in an explicit layout keeps all but the name part, which gets
                                                    a consistent fake name (D36, ``detect.org_names``)
    MULTI_SELECT items                              D35 allowlist per item; a combination is rewritten item by item;
                                                    written as combinations, a rare one of kept items is
                                                    pseudonymised whole (K-11, :func:`combination_gate`)
    NUMERIC_WITH_TOKENS leftover tokens             D35 allowlist per token (suppression tokens are vocabulary)
    numbers: discrete levels, D14 extremes          digit-only phone numbers under a person, phone/contact
    (INTEGER, FLOAT, DECIMAL, NUMERIC_*)            (``hints.phone_header_hint``) or notes header, and a column
                                                    holding a North American phone number (:func:`nanp_phone_values`,
                                                    :func:`bare_phone`: one is enough) under a header naming no ID,
                                                    code, account or amount, become same-shape masks (numeric storage
                                                    kept); person numbers under a person header are masked by the D12
                                                    guard (:func:`person_number_header`); otherwise numbers keep their
                                                    rules; a ``--force-type`` number type follows the same rules
    vocabulary levels: BOOLEAN pairs, ORDINAL       kept: a tool vocabulary or pattern quotes nobody
    scales, SCHOOL_YEAR, STATE, INTEGER_YEAR        (:func:`vocabulary_levels`)
    dates: exact extremes, sentinels, tokens        dates' own rules; leftover tokens only from the placeholder
                                                    vocabulary (``detect.dates.leftover_kinds``)
    ==============================================  =================================================================

    Also covered: functional-dependency dependents (``generate.structure``) write the dependent column's own
    written values (its pseudonyms included); derived templates are rebuilt from synthetic inputs; NA tokens: the
    default list is vocabulary, and an ``--na-value`` token follows the D35 allowlist like any unit (it says what is
    missing, not what may be published: :func:`decide_na_tokens`, R8-05); dictionary entries and value labels no row
    holds are written only when they are vocabulary words (``io.arrowio.vocabulary_label`` ->
    ``detect.vocab.is_vocabulary``).

    The D35 decision needs the number of distinct entities holding each unit, which is known once the table's
    IDs are profiled: this stage marks the detection (:data:`PENDING_KEY`) and ``profile.table`` runs the second
    stage (:func:`copy_gate_units`). A detection that writes real values is stamped ``params[COPY_GATE_KEY]``
    (:func:`copy_cleared`): the leak scanner excuses exactly the kept units of such a column and indexes every
    pseudonymised one (:func:`pseudonymized`).
    """
    if det.source in GATED_EXPLICIT_SOURCES:
        out = _gate_explicit(ctx, det)
    elif det.source in ("value", "native"):
        out = _gate(ctx, det)
    else:
        return det
    if copies_real_values(out):
        out.params[COPY_GATE_KEY] = _clearance(out)
    return out


def _clearance(det: Detection) -> str:
    """How a value-copying detection was cleared (the :data:`COPY_GATE_KEY` stamp): its text units follow the D35
    allowlist, its levels come from a tool vocabulary, its values are dates (leftover tokens are placeholder words
    only), ZIP codes follow the small-cell policy, or its values are numbers (their own rules)."""
    if det.params.get(PENDING_KEY):
        return "allowlist"
    if vocabulary_levels(det):
        return "vocabulary"
    if TYPE_INFO[det.type].family == "date":
        return "dates"
    if det.type is T.ZIP:
        return "zip_policy"
    return "numbers"


def _gate(ctx: DetectContext, det: Detection) -> Detection:
    """The first stage of :func:`copy_gate` for one detection: ``det`` (marked for the D35 decision when it writes
    real text units) or, for numbers that are phone numbers under a person/contact/notes header, same-shape
    masks."""
    t = det.type
    if vocabulary_levels(det):
        return det
    if t in _COPYING_NUMERIC_TYPES:
        tokens = set(det.params.get("tokens") or det.params.get("_tokens") or ())
        # the numbers themselves are personal only as digit-only phone numbers: any such number under a
        # person/contact/notes header, and a column of North American numbers whatever its header (``Ph``, ``Number``)
        if digit_phone_header(ctx):
            values = {v: c for v, c in ctx.typed_values.items() if v not in tokens}
            n = sum(1 for v in values if DIGIT_PHONE_RE.fullmatch(v))
        elif not_phone_header(ctx.name):
            values, n = {}, 0  # an ID, code, account or amount header says what its 10-digit numbers are (R3)
        else:
            values = nanp_phone_values(ctx.typed_values, tokens)
            n = len(values)
            if not n:
                # R6-01: ONE valid North American number among the numbers the column writes verbatim is a phone
                # number too (a phone typed into a score column is a discrete level or the exact maximum, D14)
                values = {v: c for v, c in ctx.typed_values.items() if v not in tokens}
                n = written_phone_numbers(det, values)
        if not n:
            if tokens:
                _mark_pending(ctx, det)  # the tokens among the numbers follow the D35 allowlist
            return det
        kinds = frozenset({"phone"})
        params: dict[str, Any] = {"_numeric_mask": True}
        if t in _NUMERIC_STORAGE_TYPES:
            params["numeric_storage"] = t.value
        new = Detection(ColumnType.HIGH_CARD_CATEGORICAL, subtype="code", params=params,
                        reason=f"values hold personal data ({', '.join(sorted(kinds))}) that a {t.value} column "
                               "would copy (R3 copy gate, D12): same-shape masks, real values never copied")
        new.params["pii_shapes"] = sorted(kinds)
        new.params["_min_cell"] = int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))
        hint = guarded_hint(ctx)
        if hint is not None:
            new.params.setdefault("pii_hint", hint)
        new.source = det.source
        new.missing_policy = "typed"
        new.messages.append(_pii_message(ctx, kinds, n, len(values), "distinct value(s)",
                                         f"generated as {new.type.value} (would otherwise be {t.value})"))
        return new
    if t is T.MULTI_SELECT or t in _GATE_TEXT_TYPES:
        if t is T.DIGIT_CODE and not det.params.get("levels_mode"):
            return det  # same-shape masks already
        _mark_pending(ctx, det)
    return det


#: Header words that say what a column's 10-digit numbers are when they are not phone numbers (R3: an NCES ID, a
#: district code, an account number, an amount): such a column keeps its numbers' own rules.
_NOT_PHONE_WORDS = frozenset({
    "id", "ids", "ident", "identifier", "key", "code", "codes", "cd", "account", "accounts", "acct", "amount", "amt",
    "balance", "total", "sum", "cost", "price", "budget", "salary", "revenue", "fee", "fees", "payment", "expense",
    "expenses", "income", "nces", "fips", "ein", "tin", "isbn", "serial", "count", "cnt", "qty", "quantity",
    "population", "enrollment", "time", "timestamp", "epoch", "seconds", "ms", "date", "dt", "hash", "checksum"})


def not_phone_header(name: str) -> bool:
    """A header naming something whose 10-digit numbers are no phone numbers (:data:`_NOT_PHONE_WORDS`)."""
    return any(t in _NOT_PHONE_WORDS for t in header_tokens(name))


def nanp_phone_values(typed_values: dict[str, int], tokens: Iterable[str] = ()) -> dict[str, int]:
    """The numbers of a column (``tokens`` left out) when at least :data:`NANP_SHARE` of its distinct numbers are
    digit-only North American phone numbers (:data:`NANP_DIGIT_PHONE_RE`), else ``{}``. The first 64 distinct values
    are checked first, so an ordinary number column costs next to nothing."""
    skip = set(tokens)
    head = [v for v in itertools.islice(typed_values, 64) if v not in skip]
    if not head or sum(1 for v in head if NANP_DIGIT_PHONE_RE.fullmatch(v)) < NANP_SHARE * len(head):
        return {}
    values = {v: c for v, c in typed_values.items() if v not in skip}
    n = sum(1 for v in values if NANP_DIGIT_PHONE_RE.fullmatch(v))
    return values if values and n >= NANP_SHARE * len(values) else {}


#: The smallest and largest numbers that are North American phone numbers (:func:`nanp_digits`; 11 digits with a
#: leading 1): a number column whose values all lie outside needs no closer look (:func:`written_phone_numbers`).
_PHONE_LOW, _PHONE_HIGH = 2_002_000_000, 19_899_999_999


def _phone_number(x: Any) -> bool:
    """A number (int, float, Decimal or its text) that is a whole North American phone number."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    if not (_PHONE_LOW <= f <= _PHONE_HIGH) or not f.is_integer():
        return False
    d = str(int(f))
    return nanp_digits(d[1:] if len(d) == 11 and d[0] == "1" else d)


def written_phone_numbers(det: Detection, values: dict[str, int]) -> int:
    """R6-01: how many of the numbers a number column writes verbatim are North American phone numbers written as
    one (:func:`bare_phone`: all ten digits, so ``6.53396E+09`` is a round number, not a phone): its exact minimum
    and maximum (D14) and, for a discrete integer column (exact quota over the real values), every value.
    Interpolated numbers are never copies. A column whose range misses the phone range costs nothing; without a
    known range every value is checked."""
    lo, hi = det.params.get("min"), det.params.get("max")
    try:
        low, high = float(lo), float(hi)
    except (TypeError, ValueError):
        low, high = None, None
    if low is not None and high is not None:
        if high < _PHONE_LOW or low > _PHONE_HIGH:
            return 0
        if not (det.type is T.INTEGER and det.subtype == "discrete"):
            extremes = {float(x) for x in (lo, hi) if _phone_number(x)}
            if not extremes:
                return 0
            return sum(1 for v in values if len(v) >= 10 and bare_phone(v) and _number_of(v) in extremes)
    return sum(1 for v in values if len(v) >= 10 and bare_phone(v))


def _number_of(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _gate_explicit(ctx: DetectContext, det: Detection) -> Detection:
    """The first stage of :func:`copy_gate` for ``--force-type COL=TYPE`` and ``--org-code``: the flag names the type,
    not the values (R6-03), so no rule that keeps real values from being written is skipped.

    Text: the type stays and its real text units are marked for the D35 decision like a detected column's, a
    personal header included (:func:`unit_kept` then keeps none of them; a ``forced_type_values`` warning says so).
    Numbers: the rules of a detected number column (the D12 guard over numbers under a phone/SSN/DOB, person-number
    or notes header, and the phone numbers of :func:`_gate`) give same-shape masks with the numeric storage kept;
    otherwise the type stays."""
    t = det.type
    if vocabulary_levels(det):
        return det
    if t in _COPYING_NUMERIC_TYPES:
        source = det.source
        det.source = "value"  # numbers keep their own rules, exactly as for a detected column
        try:
            out = apply_pii_precedence(ctx, det)
        finally:
            det.source = source
        out.source = source
        return out
    if t is T.MULTI_SELECT or t in _GATE_TEXT_TYPES:
        if t is T.DIGIT_CODE and not det.params.get("levels_mode"):
            return det  # same-shape masks already
        _mark_pending(ctx, det)
        if det.params.get("_personal_header"):
            flag = "--org-code" if det.source == "org-code" else "--force-type"
            det.messages.append(Message(
                "warning", "forced_type_values",
                f"{flag} sets the column's type, not its values: the header hints personal data (D12), so no real "
                "value is written (each gets a same-shape pseudonym, D35)", column=ctx.name,
                fix=f"--force-factor {ctx.name} keeps the real values (only if they cannot identify anyone)"))
    return det


def forced_copy_warning(ctx: DetectContext, det: Detection) -> Detection:
    """``--force-factor`` is D35's explicit opt-in: it keeps every real value, also under a header that hints personal
    data (D12) and with personal-data shapes (R6-03; documented in the README). Such a column gets a
    ``forced_copy`` warning that says why, with counts only (never a value), so the opt-in is never silent."""
    personal = guarded_hint(ctx) is not None or ctx.hints.free_text is not None
    digit_phone = digit_phone_header(ctx)
    kinds: set[str] = set()
    n = 0
    for v in ctx.typed_values:
        k = unit_shapes(v, digit_phone)
        if k:
            kinds |= k
            n += 1
    if not personal and not n:
        return det
    why = []
    if personal:
        why.append("its header hints personal data (D12)")
    if n:
        why.append(f"{n} of {len(ctx.typed_values)} distinct value(s) hold personal data "
                   f"({', '.join(sorted(kinds))})")
    det.messages.append(Message(
        "warning", "forced_copy",
        f"--force-factor writes every real value verbatim although {' and '.join(why)}: use it only when the values "
        "cannot identify anyone (D35: the explicit opt-in)", column=ctx.name,
        fix=f"remove --force-factor {ctx.name}: rare values then get same-shape pseudonyms and personal data is "
            "never written"))
    return det


def _mark_pending(ctx: DetectContext, det: Detection) -> None:
    det.params[PENDING_KEY] = True
    if digit_phone_header(ctx):
        det.params["_digit_phone"] = True  # a whole 10/11-digit unit is a phone number here (pii_shapes)
    if guarded_hint(ctx) is not None or ctx.hints.free_text is not None:
        # D12: a personal or notes header (numbers with tokens under ``Teacher`` / ``Note_Count``): no real text
        # unit is kept for its count, only vocabulary words (``*``, ``N/A``)
        det.params["_personal_header"] = True


# --------------------------------------------------------------------------- D35: the allowlist (second stage)
#: PRIVATE detection/profile params key: the detection's real text units (levels, multi-select items, tokens among
#: numbers) await the D35 decision, which needs the table's entity counts (:func:`copy_gate_units`, run by
#: ``profile.table``). While it is set, nothing about the column is cleared (:func:`copy_cleared`).
PENDING_KEY = "_d35_pending"
#: PRIVATE params key: the real units the D35 allowlist did not keep, each ``[text, placeholder]`` (text = the unit
#: stripped as the column holds it; the placeholder is how the report and the docs show it, on both sides).
PSEUDO_KEY = "_pseudonymized"
#: PRIVATE model key (``ColumnModel.extra`` of the row model): real raw unit -> its pseudonym (``generate.pseudonyms``).
PSEUDO_MAP_KEY = "_pseudonym_map"
#: PRIVATE model key (``ColumnModel.extra`` of the row model): stripped pseudonym -> rows of the written column holding
#: it (set by the pipeline from the synthetic values; the docs list the pseudonyms, synthetic labels, with it).
PSEUDO_WRITTEN_KEY = "_pseudonyms_written"
#: Public params key and flag: how many distinct units of the column are pseudonymised (a count, never a value).
PSEUDO_COUNT_KEY = "pseudonymized_levels"
#: D36: PRIVATE params key of an organisational-code column: real level (stripped, as the column holds it) -> the
#: pieces of it that are a person's name, ``[[start, end, role], ...]`` (``detect.org_names.OrgName.tokens``), or
#: None when the name part cannot be located (the level is pseudonymised whole). Such a level is also a
#: pseudonymised unit (:data:`PSEUDO_KEY`): it is never written as it is (``generate.pseudonyms`` writes it with a
#: fake name in place of the real one).
ORG_NAMES_KEY = "_org_code_names"
#: D36: PRIVATE params key: the name layouts found in the column's organisational codes, as written (``Garcia, Ana``,
#: ``Ms. Garcia``): real person names, which the leak scanner indexes as sensitive text (:func:`org_name_parts`).
ORG_NAME_PARTS_KEY = "_org_code_name_parts"
#: D36: public params key and flag: how many distinct levels of an organisational-code column hold a person's name
#: that is replaced (a count, never a value).
ORG_NAMES_COUNT_KEY = "org_code_names_replaced"
#: How the report and the docs show a pseudonymised unit, real and synthetic side alike.
PLACEHOLDER = "[pseudonymized {}]"
_PLACEHOLDER_RE = re.compile(r"\[pseudonymized \d+\]")
#: A plain number written as text (no leading zero: ``007`` is a code): numbers keep their own rules (D35) when they
#: have at most :data:`SHORT_NUMBER_DIGITS` digits before the decimal point (:func:`short_number`).
_PLAIN_NUMBER = re.compile(r"[+-]?(?P<int>0|[1-9]\d*)(?:\.\d+)?")
SHORT_NUMBER_DIGITS = 6
#: Kinds of units a value-copying column writes (:func:`d35_units`).
LEVELS, ITEMS, TOKENS = "levels", "items", "tokens"
_UNIT_WORDS = {LEVELS: "level(s)", ITEMS: "item(s)", TOKENS: "token(s)"}
#: Typographic apostrophes and hyphens folded like the leak scanner's matching form (``privacy.leaks.fold``), so
#: a unit decision and the scanner's comparison see the same text.
_UNIT_FOLD = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'", "‐": "-",
                            "‑": "-"})
_LEVEL_UNIT_TYPES = frozenset({T.FACTOR, T.CONSTANT, T.FLAG_SINGLE_TOKEN, T.ORDINAL, T.ORG_CODE})


def unit_key(text: str) -> str:
    """The comparison form of a unit: typographic apostrophes/hyphens folded, case-folded, whitespace collapsed (the
    leak scanner's ``norm_text``). One D35 decision covers every unit with the same key."""
    s = text if text.isascii() else text.translate(_UNIT_FOLD)
    return " ".join(s.casefold().split())


def multi_select_mode(p: Any) -> str:
    """How a MULTI_SELECT column is written (``generate.text.multi_select``): ``'combinations'`` (whole real values by
    exact quota: no delimiter, or at most ``--factor-max`` distinct combinations) or ``'items'`` (items drawn by
    their prevalence into new combinations)."""
    params = getattr(p, "params", None) or {}
    if not params.get("delimiter"):
        return "combinations"
    mode = params.get("mode")
    if mode in ("combinations", "items"):
        return mode
    model = getattr(p, "model", None)
    if model is None:
        return "items"
    n = len({raw.strip().casefold() for raw in model.nonmissing_counts if isinstance(raw, str)})
    return "combinations" if n <= int(params.get("_factor_max", 20)) else "items"


def d35_units(p: Any) -> str | None:
    """``'levels'``, ``'items'`` or ``'tokens'``: the real text units a value-copying column writes verbatim and the
    D35 allowlist decides (factor, constant, flag, non-vocabulary ordinal, org-code and digit-code levels;
    multi-select items; tokens among numbers), or None. (A multi-select written as combinations also has its whole
    combinations decided: :func:`combination_gate`, K-11.)"""
    t = p.type
    if t is T.MULTI_SELECT:
        return ITEMS
    if t is T.NUMERIC_WITH_TOKENS:
        return TOKENS
    if t in _LEVEL_UNIT_TYPES or (t is T.DIGIT_CODE and (p.params or {}).get("levels_mode")):
        return LEVELS
    return None


def _token_set(p: Any) -> frozenset[str]:
    params = p.params or {}
    return frozenset(str(t).strip() for t in (params.get("tokens") or params.get("_tokens") or ()))


def unit_texts(p: Any, raw: Any) -> list[str]:
    """The units (stripped, as the column holds them) one raw value of column ``p`` writes: the value itself for
    levels, its items for a multi-select (a whole-value NA token is no item), the value when it is one of the
    column's tokens among numbers; nothing for a missing value."""
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    kind = d35_units(p)
    na = getattr(p, "na_tokens", None) or frozenset()
    if kind == ITEMS:
        if s in na:
            return []
        d = (p.params or {}).get("delimiter")
        return list(dict.fromkeys(x.strip() for x in s.split(d) if x.strip())) if d else [s]
    if getattr(p, "missing_policy", "text") == "typed" and s in na:
        return []
    if kind == TOKENS:
        return [s] if s in _token_set(p) else []
    return [s] if kind == LEVELS else []


def unit_holders(values: Any, p: Any, entity_index: np.ndarray | None = None,
                 texts: Callable[[Any], list[str]] | None = None) -> dict[str, int]:
    """:func:`unit_key` -> number of distinct entities holding the unit in ``values`` (the column's raw values in
    row order): distinct values of ``entity_index`` (row -> entity number: the table's person IDs, a blank ID
    being its own entity), or rows when the table has no ID (``entity_index`` None). A multi-select value counts
    once for each of its items. ``texts`` (raw value -> its units) defaults to :func:`unit_texts` of ``p``."""
    vals = np.asarray(values, dtype=object)
    if not len(vals):
        return {}
    texts_of = texts if texts is not None else (lambda u: unit_texts(p, u))
    codes, uniq = codes_of(vals)
    key_ids: dict[str, int] = {}
    per_code: list[tuple[int, ...]] = []
    for u in uniq.tolist():
        ids = (key_ids.setdefault(unit_key(t), len(key_ids)) for t in texts_of(u)) if isinstance(u, str) else ()
        per_code.append(tuple(dict.fromkeys(ids)))
    if not key_ids:
        return {}
    lens = np.fromiter((len(k) for k in per_code), dtype=np.int64, count=len(per_code))
    flat = np.fromiter((k for ks in per_code for k in ks), dtype=np.int64, count=int(lens.sum()))
    codes = np.asarray(codes, dtype=np.int64)
    if entity_index is None:
        rows = np.bincount(codes, minlength=len(per_code))
        out = np.zeros(len(key_ids), dtype=np.int64)
        np.add.at(out, flat, np.repeat(rows, lens))
    else:
        ent = np.asarray(entity_index, dtype=np.int64)
        n_e = int(ent.max()) + 1 if len(ent) else 1
        pairs = np.unique(codes * n_e + ent)  # distinct (value, entity)
        pc, pe = np.divmod(pairs, n_e)
        rep = lens[pc]
        pc_r, pe_r = np.repeat(pc, rep), np.repeat(pe, rep)
        within = np.arange(len(pc_r), dtype=np.int64) - np.repeat(np.cumsum(rep) - rep, rep)
        starts = np.cumsum(lens) - lens
        kid = flat[starts[pc_r] + within]
        kept = np.unique(kid * n_e + pe_r)  # distinct (unit, entity)
        out = np.bincount(kept // n_e, minlength=len(key_ids))
    return {k: int(out[i]) for k, i in key_ids.items()}


def short_number(text: str) -> bool:
    """A plain number written as text with at most :data:`SHORT_NUMBER_DIGITS` digits before the decimal point (no
    leading zero: ``007`` is a code). Numbers keep their own rules (D35); a longer digit string is not exempt, since
    a bare phone number (``4155550187``) or an SSN-like number (``123456789``) is one too."""
    m = _PLAIN_NUMBER.fullmatch(text.strip())
    return m is not None and len(m.group("int")) <= SHORT_NUMBER_DIGITS


def unit_rows(values: Any, p: Any, wanted: Iterable[str],
              texts: Callable[[Any], list[str]] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The rows holding each of the ``wanted`` units (:func:`unit_key`), as pairs: (row numbers, unit numbers,
    the units' keys by number), each (row, unit) once, rows ascending within a unit. R7-07: one pass over the
    rows for every unit (a boolean row mask per unit was quadratic in a column of thousands of units). ``texts``:
    as for :func:`unit_holders`."""
    want = set(wanted)
    texts_of = texts if texts is not None else (lambda u: unit_texts(p, u))
    vals = np.asarray(values, dtype=object)
    codes, uniq = codes_of(vals)
    key_ids: dict[str, int] = {}
    per_code: list[tuple[int, ...]] = []
    for u in uniq.tolist():
        ks = dict.fromkeys(unit_key(t) for t in texts_of(u)) if isinstance(u, str) else {}
        per_code.append(tuple(key_ids.setdefault(k, len(key_ids)) for k in ks if k in want))
    empty = np.zeros(0, dtype=np.int64)
    if not key_ids:
        return empty, empty, []
    codes = np.asarray(codes, dtype=np.int64)
    lens = np.fromiter((len(k) for k in per_code), dtype=np.int64, count=len(per_code))
    flat = np.fromiter((k for ks in per_code for k in ks), dtype=np.int64, count=int(lens.sum()))
    starts = np.cumsum(lens) - lens
    rep = lens[codes]
    rows = np.repeat(np.arange(len(codes), dtype=np.int64), rep)
    within = np.arange(len(rows), dtype=np.int64) - np.repeat(np.cumsum(rep) - rep, rep)
    unit = flat[np.repeat(starts[codes], rep) + within]
    order = np.lexsort((rows, unit))
    return rows[order], unit[order], list(key_ids)


def unit_kept(text: str, holders: int, min_cell: int, *, org_code: bool = False, digit_phone: bool = False,
              personal_header: bool = False, domains: Iterable[str] | None = None, real_id: bool = False) -> bool:
    """THE D35 decision for one real text unit: written verbatim only when provably safe.

    The never-kept rules come first, so no exemption below can skip them (R6-03): a unit that is a real person ID of
    the run (``real_id``: ``privacy.realids``, R9-01; an organisational code, a short number or a count of holders
    never makes a real ID public) and a unit with a personal-data shape
    (:func:`unit_shapes`: e-mail address, phone number, also a bare North American one whatever the header, SSN,
    URL, street address; an organisational code too, K-09: ``Homebound - 1427 Juniper Ridge Rd`` under
    ``School_Name`` is one student's address, and no shape tells a building's address or website from a person's; in
    an organisational code an address after a spaced dash counts too, :func:`org_code_address`) are never kept;
    under a personal header (``personal_header``, D12: also a column that ``--org-code`` or
    ``--force-type`` names) nothing is kept but words that stand for no value (``detect.vocab.is_placeholder_word``:
    NA, unknown, boolean and suppression tokens), short plain numbers and units without a letter or digit
    (:func:`symbol_only`). Only then: an organisational code (D10) is kept (a person's name inside it is replaced,
    D36: :func:`copy_gate_units`); a word of the built-in vocabularies (``detect.vocab.is_vocabulary``; a word that
    is also a common name only beside an unambiguous word of its section, ``domains``), a short plain number
    (:func:`short_number`) and a unit without a letter or digit are kept whatever their count; any other unit held
    by >= ``min_cell`` distinct entities (``holders``, counted over the run: ``profile.table.finish_copy_gate``).
    Everything else is pseudonymised. (``--force-factor``, the explicit opt-in, never reaches this decision.)
    """
    if real_id or unit_shapes(text, digit_phone) or (org_code and org_code_address(text)):
        return False
    s = text.strip()
    if personal_header:
        return is_placeholder_word(s) or short_number(s) or symbol_only(s)
    if org_code:
        return True
    if is_vocabulary(s, domains) or short_number(s) or symbol_only(s):
        return True
    return holders >= min_cell


def free_unit(text: str, domains: Iterable[str] | None = None) -> bool:
    """A unit kept whatever its count unless a never-kept rule of :func:`unit_kept` applies: a vocabulary word (the
    default NA tokens among them), a short plain number or a unit without a letter or digit (its rows count no
    holders). R8-05: an ``--na-value`` token is no exemption: it says which values are missing, not that they may be
    published, so it is kept like any other unit (vocabulary, or held by >= ``--min-cell`` entities)."""
    return is_vocabulary(text, domains) or short_number(text) or symbol_only(text)


#: PRIVATE params key: :func:`unit_key` -> distinct entities of the column's table holding the unit (counted only
#: where the count can decide: a unit held by fewer rows than ``--min-cell`` keeps its row count, an upper bound).
HOLDERS_KEY = "_d35_holders"
#: PRIVATE params key: whether the table's counts are of people (IDs, names or e-mail addresses of the row's own
#: person), not of records, rows or other people's IDs (``profile.entity.d35_keys``): :data:`PEOPLE_PROVEN` (a header
#: or a student-name column proves them), :data:`PEOPLE_CLAIMED` (only ``--person-id`` says so, or the counts are of
#: a bare ``SID`` that may number the students or their records, :data:`SID_KEY`) or 0; such counts decide a unit
#: the whole run holds (:func:`run_holders`).
PEOPLE_KEY = "_d35_people"
PEOPLE_PROVEN, PEOPLE_CLAIMED = 2, 1
#: PRIVATE params key (H-1, H-6): the headers of the bare ``SID`` columns a table with no person key counts its
#: holders by (``profile.entity.HolderKeys.sids``). Over the run such counts rank as a claim (:func:`run_holders`:
#: the fewest wins; never pooled with records and rows, which would let a sibling's 11 incidents outvote the one SID
#: that holds a value); nobody said that SID numbers the row's own person, so a unit they alone keep gets the
#: ``d35_rows_counted`` warning (:func:`copy_gate_units`).
SID_KEY = "_d35_sid"
#: PRIVATE params key: what the table's counts are of, for the messages (``people (IDs)``, ``rows ...``).
BASIS_KEY = "_d35_basis"
#: PRIVATE params key: the table has ID or name keys (an ID column, a flagged one, a name column), whatever they number,
#: and no name or e-mail column that differs on every row (``profile.entity.HolderKeys.own_rows``).
KEYED_KEY = "_d35_keyed"


def unit_groups(p: Any) -> tuple[dict[str, list[str]], Counter]:
    """(:func:`unit_key` -> the column's real unit texts with that key, :func:`unit_key` -> rows holding it)."""
    texts: dict[str, list[str]] = {}
    rows: Counter = Counter()
    model = getattr(p, "model", None)
    for raw, c in (model.nonmissing_counts.items() if model is not None else ()):
        for t in unit_texts(p, raw):
            k = unit_key(t)
            group = texts.setdefault(k, [])
            if t not in group:
                group.append(t)
            rows[k] += int(c)
    return texts, rows


#: PRIVATE params key: :func:`unit_key` -> the vocabulary sections that the unit's partner value names
#: (:func:`partner_domains`).
PARTNER_KEY = "_d35_partner_domains"


def unit_domains(p: Any, texts: dict[str, list[str]]) -> Any:
    """``unit_key -> sections`` in which a name-like vocabulary word of column ``p`` is read as vocabulary
    (``detect.vocab.is_vocabulary``): the sections of the column's unambiguous words (``context_domains``), of its
    header (``detect.vocab.header_domains``: ``Attendance_Code``, ``Race``; hardening pass 3) and of the unit's own
    partner value in a column that describes this one (:func:`partner_domains`: ``ED`` beside ``Early Dismissal``)."""
    base = context_domains(t for group in texts.values() for t in group) | header_domains(getattr(p, "name", ""))
    partner = (getattr(p, "params", None) or {}).get(PARTNER_KEY) or {}
    return lambda k: base | frozenset(partner.get(k, ()))


#: Fewest distinct pairs of a column and the column that describes it (:func:`partner_domains`).
PARTNER_MIN_PAIRS = 3


def partner_domains(columns: list[tuple[Any, Any]]) -> None:
    """D35 (hardening pass 3): a code column and the column that describes it (``Code`` = ``ED`` and ``Description`` =
    ``Early Dismissal`` on the same rows), both marked by :func:`copy_gate` and written as levels, are one to one over
    the rows where both are filled (at least :data:`PARTNER_MIN_PAIRS` pairs). A name-like vocabulary word of one of
    them (``ED``, ``TU``, ``White``) is then read in the sections its partner value names when that value is an
    unambiguous word of the vocabulary with two or more letters: single-letter codes beside ``ED`` vouch for nothing
    (``context_domains``), but ``Early Dismissal`` does. Stored PRIVATELY in ``params[PARTNER_KEY]``.

    ``columns``: (profile, raw values in row order) of one table. Only a code column reads its words this way: at
    least half of its other words are vocabulary words (single-letter codes count), so a column of surnames
    (``Homeroom``) never reads a teacher's surname as a race from a race column beside it, even in a tiny table."""
    cols = [(p, v) for p, v in columns if (p.params or {}).get(PENDING_KEY) and d35_units(p) == LEVELS
            and not p.params.get("_personal_header")]
    if len(cols) < 2:
        return
    # only a code column needs a partner: one holding a name-like word among words that are mostly vocabulary (single
    # letters too: A, T, E ...); a column of surnames (Homeroom) never reads its 'White' from a race column
    needy: dict[int, set[str]] = {}
    for p, _ in cols:
        keys = list(unit_groups(p)[0])
        named = {k for k in keys if is_name_like(k)}
        rest = [k for k in keys if k not in named]
        if named and len(rest) >= PARTNER_MIN_PAIRS and 2 * sum(1 for k in rest if is_vocabulary(k)) >= len(rest):
            needy[id(p)] = named
    if not any(needy.values()):
        return
    keyed: list[tuple[Any, np.ndarray, np.ndarray]] = []  # (profile, row -> value code, value code -> unit key)
    for p, values in cols:
        codes, uniq = codes_of(np.asarray(values, dtype=object))
        keys = np.empty(len(uniq), dtype=object)
        for i, u in enumerate(uniq.tolist()):
            t = unit_texts(p, u) if isinstance(u, str) else []
            keys[i] = unit_key(t[0]) if t else None
        keyed.append((p, np.asarray(codes, dtype=np.int64), keys))
    for p, codes, keys in keyed:
        wanted = needy.get(id(p))
        if not wanted:
            continue
        found: dict[str, set[str]] = {}
        filled = np.asarray([k is not None for k in keys], dtype=bool)[codes]
        for q, q_codes, q_keys in keyed:
            if q is p:
                continue
            both = filled & np.asarray([k is not None for k in q_keys], dtype=bool)[q_codes]
            if not both.any():
                continue
            width = int(q_codes.max()) + 1
            pa, pb = np.divmod(np.unique(codes[both] * width + q_codes[both]), width)
            links = set(zip(keys[pa].tolist(), q_keys[pb].tolist()))
            one_to_one = len({x for x, _ in links}) == len({y for _, y in links}) == len(links)
            if len(links) < PARTNER_MIN_PAIRS or not one_to_one:
                continue  # not one to one: no description of this column
            for x, y in links:
                if x in wanted:
                    found.setdefault(x, set()).update(context_domains([y]))
        found = {k: v for k, v in found.items() if v}
        if found:
            p.params[PARTNER_KEY] = {k: sorted(v) for k, v in sorted(found.items())}


def count_units(p: Any, values: Any, keys: Any, min_cell: int) -> None:
    """D35, first half of the copy gate's second stage (``profile.table``, while the raw values are at hand): the
    number of distinct entities of the table holding each real text unit of a column marked by :func:`copy_gate`
    (:data:`PENDING_KEY`), stored PRIVATELY in ``params[HOLDERS_KEY]`` with the table's basis.

    ``keys`` (``profile.entity.HolderKeys``) holds the table's numberings: a unit is held by the MINIMUM over them of
    the distinct entities among its rows (:func:`unit_holders`); with no person key, rows count and a record
    repeated in every column counts once (``HolderKeys.duplicates``). Counting is skipped where it cannot decide: a
    unit held by fewer rows than ``min_cell`` keeps its row count (fewer entities too); a unit with a personal-data
    shape, a column under a personal header, and vocabulary words (the default NA tokens among them) and short numbers
    get no entry (their rows count no holders, so they never vouch for the same text in a column of the same name
    elsewhere). An ``--na-value`` token written as a level is counted like any other unit (R8-05).
    """
    params = p.params
    if not params.get(PENDING_KEY):
        return
    digit_phone = bool(params.get("_digit_phone"))
    personal = bool(params.get("_personal_header"))
    org = p.type is T.ORG_CODE
    texts, rows = unit_groups(p)
    domains = unit_domains(p, texts)
    # a unit held by fewer rows than min_cell is held by fewer entities too: its rows are a safe bound. A unit this
    # column decides without a count (a vocabulary word here, a personal-data shape, a personal header) gets no
    # entry: its rows are no count of holders and must not make the same text safe in another column
    holders: dict[str, int] = {k: rows[k] for k in texts if rows[k] < min_cell}

    def needs_count(k: str) -> bool:
        group = texts[k]
        if rows[k] < min_cell or personal or any(unit_shapes(t, digit_phone) for t in group):
            return False
        return org or not all(free_unit(t, domains(k)) for t in group)

    holders.update(_counted(p, values, keys, [k for k in texts if needs_count(k)], rows))
    if not personal and writes_combinations(p):
        # K-11: a multi-select written as combinations writes each whole real value: its own holders decide it
        # (combination_gate), never only the holders of its items
        whole, whole_rows = combination_groups(p)
        holders.update({k: whole_rows[k] for k in whole if whole_rows[k] < min_cell})
        need = [k for k in whole if whole_rows[k] >= min_cell
                and _combination_needs_count(p, whole[k][0], domains, digit_phone)]
        holders.update(_counted(p, values, keys, need, whole_rows, lambda raw: _whole_combination(p, raw)))
    params[HOLDERS_KEY] = holders
    params[PEOPLE_KEY] = people_level(keys)
    if not getattr(keys, "people", False) and getattr(keys, "sids", ()):
        params[SID_KEY] = list(keys.sids)
    params[BASIS_KEY] = str(getattr(keys, "basis", "rows (the table has no ID column)"))
    params[KEYED_KEY] = bool(getattr(keys, "keys", None)) and not getattr(keys, "own_rows", False)


#: PRIVATE params key of a ZIP column: ZIP5 -> distinct entities of the table holding it (K-08,
#: :func:`count_zip_holders`), read by the small-cell policy (``generate.pii._zip_map``).
ZIP_HOLDERS_KEY = "_zip5_holders"


def _zip5_of(raw: Any) -> list[str]:
    m = ZIP_HINTED_RE.fullmatch(raw.strip()) if isinstance(raw, str) else None
    return [m.group("zip5")] if m else []


def count_zip_holders(p: Any, values: Any, keys: Any, min_cell: int) -> None:
    """K-08: the holders of each real ZIP5 code of ZIP column ``p`` (``values``: its raw values in row order), counted
    as D35 counts a text unit (:func:`_counted`: the fewest over the table's numberings ``keys``,
    ``profile.entity.HolderKeys``; rows, a record exported several times once, without a person key), stored
    PRIVATELY in ``params[ZIP_HOLDERS_KEY]``. A code held by fewer rows than ``min_cell`` keeps its row count (fewer
    holders too). The small-cell policy keeps a code only when at least ``min_cell`` holders hold it: one student's
    ZIP code on 12 rows is one holder."""
    if getattr(p, "model", None) is None:
        return
    rows: Counter = Counter()
    for raw, c in p.model.nonmissing_counts.items():
        for z in _zip5_of(raw):
            rows[z] += int(c)
    held = {z: c for z, c in rows.items() if c < min_cell}
    if keys is not None:
        held.update(_counted(p, values, keys, [z for z in rows if rows[z] >= min_cell], rows, _zip5_of))
    else:
        held.update({z: c for z, c in rows.items() if c >= min_cell})
    p.params[ZIP_HOLDERS_KEY] = {z: int(c) for z, c in held.items()}


def people_level(keys: Any) -> int:
    """:data:`PEOPLE_KEY` of a table's ``keys`` (``profile.entity.HolderKeys``): :data:`PEOPLE_PROVEN`,
    :data:`PEOPLE_CLAIMED` (an ``--person-id`` claim, or H-1 the counts of a bare ``SID``, ``HolderKeys.sids``: they
    are the students' own if SID is their ID, so no record or row count of another table may outvote them) or 0."""
    if not getattr(keys, "people", False):
        return PEOPLE_CLAIMED if getattr(keys, "sids", ()) else 0
    return PEOPLE_PROVEN if getattr(keys, "proven", True) else PEOPLE_CLAIMED


def _counted(p: Any, values: Any, keys: Any, wanted: list[str], rows: Counter,
             texts: Callable[[Any], list[str]] | None = None) -> dict[str, int]:
    """:func:`unit_key` -> distinct entities holding it, for the ``wanted`` units of column ``p`` (``values``: its raw
    values in row order; ``rows``: rows holding each unit): the minimum over the table's numberings (``keys``,
    ``profile.entity.HolderKeys``) and, with no person key, one record repeated in every column counted once
    (``HolderKeys.duplicates``). ``texts``: as for :func:`unit_holders`."""
    if not wanted:
        return {}
    counted = {k: rows[k] for k in wanted}  # rows count where no key numbers the entities
    for index in list(getattr(keys, "keys", None) or []):
        got = unit_holders(values, p, index, texts)
        counted = {k: min(c, got.get(k, c)) for k, c in counted.items()}
    if getattr(keys, "table", None) is not None:
        # no person key: one record repeated in every column is one holder (profile.entity.HolderKeys.duplicates)
        for k, dup in keys.duplicates(p.index, values, unit_rows(values, p, wanted, texts)).items():
            counted[k] = min(counted[k], dup)
    return counted


def holder_field(p: Any) -> str:
    """The field a column's D35 counts pool with over the run: its normalised name (``Program_Name`` in a lookup
    table and ``program name`` in its fact table are one field; ``Race`` and ``Homeroom`` are two)."""
    from ..hints import normalize

    name = str(getattr(p, "name", ""))
    return normalize(name) or name


#: Fewest codes of a code table (one row per code) whose column joins the field of the column using its codes
#: (:func:`lookup_fields`).
LOOKUP_MIN_CODES = 3


def lookup_fields(columns: Iterable[Any]) -> dict[str, str]:
    """D35 over the run (hardening pass 2): field (:func:`holder_field`) -> the field it pools with, for the code
    columns of code tables whose codes another column uses under another name (``programs.csv::Code`` and
    ``enrollments.csv::Program_Code``): a lookup table follows the table that uses its codes, so the join still
    matches.

    A code column holds each of its (at least :data:`LOOKUP_MIN_CODES`) levels on exactly one row; a column using
    its codes holds at least :data:`LOOKUP_MIN_CODES` levels, all of them among the code column's, and at least half
    of them (a fact table rarely uses fewer than half of a code list; two unrelated columns rarely share most of
    their values). Fields joined this way share their counts (:func:`run_holders`); each field joins the
    alphabetically first field of its group, so the result does not depend on the order of the files."""
    cols = [p for p in columns if (getattr(p, "params", None) or {}).get(PENDING_KEY)
            and not p.params.get("_personal_header") and d35_units(p) == LEVELS]
    info = []
    for p in cols:
        texts, rows = unit_groups(p)
        info.append((holder_field(p), frozenset(texts), rows))
    parent: dict[str, str] = {}

    def find(f: str) -> str:
        while parent.setdefault(f, f) != f:
            f = parent[f]
        return f

    for f_code, codes, rows in info:
        if len(codes) < LOOKUP_MIN_CODES or any(c != 1 for c in rows.values()):
            continue
        for f_use, used, _ in info:
            if f_use == f_code or len(used) < LOOKUP_MIN_CODES or not used <= codes or 2 * len(used) < len(codes):
                continue
            a, b = find(f_code), find(f_use)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {f: find(f) for f in list(parent) if find(f) != f}


def run_holders(columns: Iterable[Any], groups: dict[str, str] | None = None,
                people_units: dict[str, set[str]] | None = None) -> dict[str, dict[str, int]]:
    """D35 over the run: field (:func:`holder_field`, or the field it pools with in ``groups``,
    :func:`lookup_fields`) -> :func:`unit_key` -> distinct entities holding the unit, for every unit of the marked
    columns of ``columns`` (every table of the run).

    A field is one kind of value wherever the run holds it (the same column name in several files: a lookup table
    and the table that uses its codes, or one export per school); another column holding the same text means
    something else (``White`` under ``Race`` and under ``Homeroom``), so its holders never count here. Within a field,
    the strongest evidence of who the people are decides, and weaker evidence never outvotes it (K-04, K-05: no
    run-level maximum lets a weak key clear a value that a proven count shows held by few):

    1. counts of proven people (:data:`PEOPLE_PROVEN`: an ID whose header names the row's own person, or names under a
       student header; never teacher, staff or record IDs): the most people any one such table shows (a value shared
       by >= ``--min-cell`` people somewhere singles no one out anywhere, and a lookup table's one row per code does
       not make its codes rare);
    2. only for a unit no such table counts: counts of claimed people (:data:`PEOPLE_CLAIMED`: an ID that only
       ``--person-id`` calls the row's own person, such as an incident or badge number, and the counts of a bare
       ``SID`` that nothing ties to the students, :data:`SID_KEY`, H-1): the FEWEST any such table shows (the claim
       may be wrong, and the smaller count is the safe side);
    3. only for a unit no table counts people for: records or rows (the largest such count).

    So a value that a file with person IDs shows held by one student is never kept because a sibling file without IDs
    repeats it on many rows, or numbers its rows with an ``--person-id`` that is no student's, and a value one SID
    holds is never kept because a staff, incident or row-numbered sibling repeats it (H-1). Columns under a personal
    header count for nothing: their values are never kept for a count, and their counts say who is listed there, not
    who shares the value. ``people_units`` (when given) receives field -> the units some table counts people for
    (proven, or claimed by ``--person-id``; a bare SID's counts are no person's for the ``d35_rows_counted`` warning,
    :func:`copy_gate_units`).
    """
    proven: dict[str, dict[str, int]] = {}
    claimed: dict[str, dict[str, int]] = {}
    asserted: dict[str, set[str]] = {}  # the units an --person-id claim counts (a bare SID's are not asserted)
    other: dict[str, dict[str, int]] = {}
    for p in columns:
        params = getattr(p, "params", None) or {}
        if not params.get(PENDING_KEY) or params.get("_personal_header"):
            continue
        field = holder_field(p)
        field = (groups or {}).get(field, field)
        level = params.get(PEOPLE_KEY)
        level = PEOPLE_PROVEN if level is True else int(level or 0)
        if level == PEOPLE_CLAIMED:
            agg = claimed.setdefault(field, {})
            for k, c in (params.get(HOLDERS_KEY) or {}).items():
                agg[k] = min(agg.get(k, int(c)), int(c))
            if not params.get(SID_KEY):
                asserted.setdefault(field, set()).update(params.get(HOLDERS_KEY) or {})
            continue
        agg = (proven if level else other).setdefault(field, {})
        for k, c in (params.get(HOLDERS_KEY) or {}).items():
            agg[k] = max(agg.get(k, 0), int(c))
    fields = set(proven) | set(claimed) | set(other)
    if people_units is not None:
        for f in fields:
            units = set(proven.get(f, {})) | asserted.get(f, set())
            if units:
                people_units.setdefault(f, set()).update(units)
    return {f: {**other.get(f, {}), **claimed.get(f, {}), **proven.get(f, {})} for f in fields}


def copy_gate_units(p: Any, holders: dict[str, int], min_cell: int, people_units: Iterable[str] | None = None,
                    real_id: Callable[[str], bool] | None = None) -> int:
    """D35, second half of the copy gate's second stage (``profile.table.finish_copy_gate``, once every table of the
    run is counted): decide every real text unit of a column marked by :func:`copy_gate` (:data:`PENDING_KEY`)
    with :func:`unit_kept`, given ``holders`` (the column's field in :func:`run_holders`; a unit missing there counts
    its own rows).

    Records the units that are not kept in PRIVATE ``params[PSEUDO_KEY]`` with their placeholders (numbered by
    rows, most frequent first; one decision and one placeholder per :func:`unit_key`), the public count
    ``params['pseudonymized_levels']`` and flag, and an info message with counts only; public params that quote a
    unit (an ordinal's ``levels``, a constant's ``value``, a flag's ``token``) show the placeholder. Returns the
    number of pseudonymised units (0 when the column was not marked).

    D36: a level of an organisational code that the allowlist keeps but that holds a person's name in an explicit
    layout (:func:`org_code_names`) is a pseudonymised unit too (placeholder, ``pseudonymized_levels``), whose
    pseudonym keeps everything but the name part (``generate.pseudonyms``): its name pieces are recorded PRIVATELY
    (:data:`ORG_NAMES_KEY`, :data:`ORG_NAME_PARTS_KEY`), with the public count ``org_code_names_replaced`` and an info
    message with counts only.

    ``people_units`` (the units of the field some table of the run counts people for, :func:`run_holders`): a unit
    kept for its count where no table counted people, in a table whose ID or name columns all number something else
    than the row's own person (teacher IDs, record IDs, bare names), gets a ``d35_rows_counted`` warning with the fix
    (hardening pass 3): one person's repeated rows may have been counted as several holders.

    ``real_id`` (R9-01, ``privacy.realids``): tells a unit that is a real person ID of the run; such a unit is never
    kept (:func:`unit_kept`), whatever its count, vocabulary or organisational-code status.
    """
    params = p.params
    if not params.pop(PENDING_KEY, None):
        return 0
    digit_phone = bool(params.get("_digit_phone"))  # kept: the pseudonyms follow the phone rules too
    personal = bool(params.pop("_personal_header", False))
    own = params.pop(HOLDERS_KEY, None) or {}
    people = bool(params.pop(PEOPLE_KEY, None))
    sids = params.pop(SID_KEY, None) or []
    keyed = bool(params.pop(KEYED_KEY, None))
    who = params.pop(BASIS_KEY, None) or "rows (the table has no ID column)"
    org = p.type is T.ORG_CODE
    kind = d35_units(p) or LEVELS
    texts, rows = unit_groups(p)
    domains = unit_domains(p, texts)
    params.pop(PARTNER_KEY, None)
    count = {k: int(holders.get(k, own.get(k, rows[k]))) for k in texts}
    ids = {k for k, group in texts.items() if real_id is not None and any(real_id(t) for t in group)}
    gone = [k for k, group in texts.items()
            if not all(unit_kept(t, count[k], min_cell, org_code=org, digit_phone=digit_phone,
                                 personal_header=personal, domains=domains(k), real_id=k in ids) for t in group)]
    combos, combo_rows, combo_keep = combination_gate(p, set(gone), holders, own, min_cell, domains, count)
    # D36: a kept organisational code holding a person's name in an explicit layout keeps all but the name part
    named = org_code_names(p, texts, set(gone)) if org and not personal else {}
    if org:
        rare = sum(1 for k in texts if k not in gone and k not in named and own.get(k, rows[k]) < min_cell)
        if rare:
            p.messages.append(Message(
                "info", "rare_levels_kept",
                f"{rare} of {len(texts)} level(s) are held by fewer than {min_cell} {who}; they keep their real "
                "values because the column is an organisational code (D10)", column=p.name))
    if keyed and (not people or sids) and not personal and not org and people_units is not None:
        by_people = set(people_units)
        counted = [k for k, group in texts.items() if k not in gone and k not in by_people
                   and not all(free_unit(t, domains(k)) for t in group)]
        if counted and sids:  # H-6: a bare SID may well be the students' own ID; the text says what is unknown
            sid = sids[0]
            p.messages.append(Message(
                "warning", "d35_rows_counted",
                f"{len(counted)} of {len(texts)} {_UNIT_WORDS[kind]} keep their real values because at least "
                f"{min_cell} distinct {sid} values hold them, but nothing shows that {sid} numbers the row's own "
                f"person: the header {sid} numbers sessions, sections or visits on some exports, no student-name "
                "column agrees with it, and its values are not shown to be the run's student IDs (at least half of "
                "them IDs of a Student_ID-like column of the run; numbers of a dense range such as 1001-1300 prove "
                "nothing, any numbering there matches by chance). If it numbers records, one person's records may "
                "count as several holders (D35)", column=p.name,
                fix=f"--person-id {sid} if {sid} is the student's own ID (its values then count as people and this "
                    f"warning goes); --force-text {p.name} writes invented labels instead"))
        elif counted:
            p.messages.append(Message(
                "warning", "d35_rows_counted",
                f"{len(counted)} of {len(texts)} {_UNIT_WORDS[kind]} keep their real values because at least "
                f"{min_cell} records or rows hold them, but no ID or student-name column of this table numbers the "
                "row's own person: one person's repeated rows may count as several holders (D35)", column=p.name,
                fix="--person-id COL for the column that numbers the row's own person (its people are then counted); "
                    f"--force-text {p.name} writes invented labels instead"))
    if not gone and not named and not combos:
        return 0
    order = sorted([*gone, *named], key=lambda k: (-rows[k], k))
    number = {k: i + 1 for i, k in enumerate(order)}
    rec = [[t, PLACEHOLDER.format(number[k])] for k in order for t in sorted(texts[k])]
    # K-11: the rare whole combinations of kept items, numbered after the units
    c_order = sorted(combos, key=lambda k: (-combo_rows[k], k))
    rec += [[t, PLACEHOLDER.format(len(order) + i + 1)] for i, k in enumerate(c_order) for t in sorted(combos[k])]
    params[PSEUDO_KEY] = rec
    params[PSEUDO_COUNT_KEY] = len(order) + len(combos)
    p.flags[PSEUDO_COUNT_KEY] = True
    if combos:
        params[COMBO_KEEP_KEY] = {k: combo_keep[k] for k in combos}
        total = sum(int(c) for raw, c in p.model.nonmissing_counts.items() if isinstance(raw, str)) or 1
        p.reason += (f"; D35: {len(combos)} rare combination(s) keep only real items a common value holds "
                     "together, the others as same-shape pseudonyms (K-11)")
        p.messages.append(Message(
            "info", "pseudonymized_combinations",
            f"{len(combos)} of {len(combo_rows)} combination(s) of several items "
            f"({sum(combo_rows[k] for k in combos) / total:.0%} of the values) would write two or more real items "
            f"together, but are held by fewer than {min_cell} {who} as a whole: each keeps its vocabulary words and "
            "only real items that a common whole value of the column holds together (else the most widely held "
            "one), and every other item is written as that item's consistent same-shape pseudonym, with the "
            "combination's exact count, so no rare whole value is rebuilt from common items (D35)",
            column=p.name,
            fix=f"--force-factor {p.name} keeps the real values (only if they cannot identify anyone)"))
    if not gone and not named:
        return len(combos)
    shown = {t: ph for t, ph in rec}
    if isinstance(params.get("levels"), list):
        params["levels"] = [shown.get(v.strip(), v) if isinstance(v, str) else v for v in params["levels"]]
    for key in ("value", "token"):
        v = params.get(key)
        if isinstance(v, str) and v.strip() in shown:
            params[key] = shown[v.strip()]
    what = _UNIT_WORDS[kind]
    total_rows = sum(rows.values()) or 1
    if named:
        found = {t: names for group in named.values() for t, names in group.items()}
        params[ORG_NAMES_KEY] = {t: (None if o is None or o.tokens is None else [list(x) for x in o.tokens])
                                 for t, o in sorted(found.items())}
        params[ORG_NAME_PARTS_KEY] = sorted({part for o in found.values() if o is not None for part in o.parts})
        params[ORG_NAMES_COUNT_KEY] = len(named)
        p.flags[ORG_NAMES_COUNT_KEY] = True
        p.reason += f"; D36: {len(named)} of {len(texts)} {what} hold a person's name, written with a fake name"
        p.messages.append(Message(
            "info", "org_code_names",
            f"{len(named)} of {len(texts)} {what} ({sum(rows[k] for k in named) / total_rows:.0%} of the values) hold "
            "a person's name in an explicit layout (surname and given name, a title or initial before a surname, a "
            "known first and last name): the name part is written as a consistent fake name of the same layout, the "
            "rest of the code stays real and every file of the run writes the same value for it (D36); a value whose "
            "name cannot be located is written as a same-shape pseudonym", column=p.name,
            fix=f"--force-factor {p.name} keeps the real values (only if they name no one)"))
    if not gone:
        return len(order) + len(combos)
    share = sum(rows[k] for k in gone) / total_rows
    p.reason += f"; D35: {len(gone)} of {len(texts)} {what} written as same-shape pseudonyms"
    why = ("the column's header hints personal data (D12)" if personal else
           f"neither vocabulary words nor held by at least {min_cell} {who} in the run")
    n_ids = len(ids & set(gone))
    as_ids = f"; {n_ids} of them equal a real person ID of the run (R9-01)" if n_ids else ""
    p.messages.append(Message(
        "info", "pseudonymized_levels",
        f"{len(gone)} of {len(texts)} {what} ({share:.0%} of the values) are written as consistent same-shape "
        f"pseudonyms with their exact counts (D35): {why}, or they hold personal data{as_ids}", column=p.name,
        fix=f"--force-factor {p.name} keeps the real values (only if they cannot identify anyone)"))
    return len(order) + len(combos)


def writes_combinations(p: Any) -> bool:
    """A multi-select with a delimiter written as combinations (:func:`multi_select_mode`): its whole real values are
    written by exact quota (K-11)."""
    return (getattr(p, "type", None) is T.MULTI_SELECT and bool((getattr(p, "params", None) or {}).get("delimiter"))
            and multi_select_mode(p) == "combinations")


def _whole_combination(p: Any, raw: Any) -> list[str]:
    """``[raw]`` (stripped) when it holds two or more distinct items of multi-select ``p``, else ``[]``."""
    items = unit_texts(p, raw)
    return [raw.strip()] if len({unit_key(t) for t in items}) >= 2 else []


def combination_groups(p: Any) -> tuple[dict[str, list[str]], Counter]:
    """K-11: (:func:`unit_key` of a whole value with two or more items -> its real forms, -> rows holding it)."""
    texts: dict[str, list[str]] = {}
    rows: Counter = Counter()
    model = getattr(p, "model", None)
    for raw, c in (model.nonmissing_counts.items() if model is not None else ()):
        for t in _whole_combination(p, raw):
            k = unit_key(t)
            group = texts.setdefault(k, [])
            if t not in group:
                group.append(t)
            rows[k] += int(c)
    return texts, rows


def _combination_needs_count(p: Any, text: str, domains: Callable[[str], Any], digit_phone: bool) -> bool:
    """A whole combination whose decision may need its holders: no item has a personal-data shape (that item is
    pseudonymised anyway) and not every item is a free word (:func:`free_unit`)."""
    items = unit_texts(p, text)
    if any(unit_shapes(t, digit_phone) for t in items):
        return False
    return not all(free_unit(t, domains(unit_key(t))) for t in items)


#: PRIVATE params key (K-11, H-7): :func:`unit_key` of a whole combination :func:`combination_gate` did not keep ->
#: ``[free, real]``, the :func:`unit_key` of each of its items written as it is: its free words, and the real items a
#: whole value held by >= ``--min-cell`` writes together anyway (or at most one); every other item is written as that
#: item's run-wide pseudonym (``generate.pseudonyms``), and the leak scanner excuses exactly these
#: (``privacy.leaks``; :func:`combination_stay`).
COMBO_KEEP_KEY = "_d35_combo_keep"


def combination_stay(p: Any, key: str) -> set[str]:
    """K-11: the :func:`unit_key` of the items a whole combination (``key``) the D35 allowlist did not keep writes as
    they are (:data:`COMBO_KEEP_KEY`)."""
    got = ((getattr(p, "params", None) or {}).get(COMBO_KEEP_KEY) or {}).get(key) or ([], [])
    return {k for part in got for k in part}


def _subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    rest = iter(long)
    return all(any(x == y for y in rest) for x in short)


def combination_gate(p: Any, gone: set[str], holders: dict[str, int], own: dict[str, int], min_cell: int,
                     domains: Callable[[str], Any], count: dict[str, int] | None = None
                     ) -> tuple[dict[str, list[str]], Counter, dict[str, list[list[str]]]]:
    """K-11 (D35): the whole combinations a multi-select written as combinations (:func:`writes_combinations`) may not
    write as they are, the rows of every combination of two or more items, and the items each such combination
    writes as they are (:data:`COMBO_KEEP_KEY`).

    Such a column writes each whole real value by exact quota, so two items each held by many people (``Okafor`` and
    ``Tuan``, the halves of ``Okafor, Tuan`` under ``Pickup_List``) would rebuild a whole value one student holds. The
    rule, an allowlist: a combination held by fewer than ``min_cell`` entities as a whole (``holders``, the run's count
    of its field, else the column's own, ``own``, else its rows) never writes itself, and writes real items that are
    not free words (:func:`free_unit`: vocabulary, short numbers, symbols: ``ELL;SPED`` is written as it is) together
    only when they are, in its order, the real items of a whole value of the column that D35 keeps as it is (held by
    >= ``min_cell``: that text is written anyway; ``Extended time;Read aloud`` inside a rare ``Extended time;Read
    aloud;Small group``; ``Okafor, Tuan`` never, unless a whole value ``Okafor, Tuan`` is that common). So a
    combination is written item by item as before when its kept items are at most one, or such a common sequence,
    beside an item pseudonymised anyway (``gone``: that pseudonym breaks the value); every other one is returned
    (:func:`unit_key` -> its real forms, and -> ``[free, real]``: its free words, and the longest common sequence of
    its other kept items that leaves at least one of them out, else the one held by the most entities, ``count``, the
    least telling). H-7: every other item becomes its run-wide item pseudonym (``generate.pseudonyms``), so item
    prevalence holds under the pseudonyms and the vocabulary gains one pseudonym per replaced item, not one per
    combination."""
    if not writes_combinations(p):
        return {}, Counter(), {}
    texts, rows = combination_groups(p)
    count = count or {}

    def split(form: str) -> tuple[set[str], tuple[str, ...], bool]:
        items: dict[str, str] = {}
        for t in unit_texts(p, form):
            items.setdefault(unit_key(t), t)
        free = {ik for ik, t in items.items() if ik not in gone and free_unit(t, domains(ik))}
        kept = tuple(ik for ik in items if ik not in gone and ik not in free)
        return free, kept, any(ik in gone for ik in items)

    parts = {k: split(group[0]) for k, group in texts.items()}
    whole = {k: int(holders.get(k, own.get(k, rows[k]))) for k in texts}
    # H-7: the kept-item sequences that a whole value D35 keeps (no item pseudonymised) writes together anyway
    common: dict[tuple[str, ...], int] = {}
    for k, (_, kept, broken) in parts.items():
        if len(kept) >= 2 and not broken and whole[k] >= min_cell:
            common[kept] = max(common.get(kept, 0), whole[k])
    out: dict[str, list[str]] = {}
    keep: dict[str, list[list[str]]] = {}
    for k, group in texts.items():
        free, kept, broken = parts[k]
        if not kept or (broken and (len(kept) == 1 or kept in common)):
            continue
        if whole[k] >= min_cell:
            continue
        out[k] = group
        fits = sorted((t for t in common if len(t) < len(kept) and _subsequence(t, kept)),
                      key=lambda t: (-len(t), -common[t], t))
        stay = (fits[0] if fits else
                sorted(kept, key=lambda ik: (-int(count.get(ik, 0)), ik))[:1] if len(kept) >= 2 else [])
        keep[k] = [sorted(free), list(stay)]
    return out, rows, keep


def org_code_names(p: Any, texts: dict[str, list[str]], gone: set[str]) -> dict[str, dict[str, Any]]:
    """D36: :func:`unit_key` -> {real level text -> its ``detect.org_names.OrgName``, or None for a case form that holds
    no layout of its own} for every level of organisational-code column ``p`` that the D35 allowlist kept (not in
    ``gone``) and that holds a person's name in an explicit layout (``detect.org_names.find``: the D34 checks of
    headers). One case form with a layout makes the whole group a named level (fail-safe); a form without one reuses
    the name part of a form of the same length, else it is pseudonymised whole."""
    from .org_names import OrgName, find

    out: dict[str, dict[str, Any]] = {}
    for k, group in texts.items():
        if k in gone:
            continue
        found = {t: find(t) for t in group}
        located = [(t, o) for t, o in found.items() if o is not None and o.tokens is not None]
        if not any(o is not None for o in found.values()):
            continue
        for t, o in list(found.items()):
            if o is None:  # the same words in another case or spacing: the name pieces must sit at the same places
                base = next((b for u, b in located if len(u) == len(t) and all(
                    t[x:y].casefold() == u[x:y].casefold() for x, y, _ in b.tokens)), None)
                found[t] = OrgName(base.tokens if base is not None else None, ())
        out[k] = found
    return out


def org_name_parts(p: Any) -> list[str]:
    """D36: the person-name layouts found in the organisational codes of column ``p`` (``Garcia, Ana``, ``Ms. Garcia``),
    as written: real names the leak scanner indexes as sensitive text, never excused (``privacy.leaks``)."""
    return list((getattr(p, "params", None) or {}).get(ORG_NAME_PARTS_KEY) or ())


def org_name_phrases(p: Any) -> list[str]:
    """D36: what the leak scanner indexes for the person names found in the organisational codes of column ``p``
    (never excused): each name part as written (:func:`org_name_parts`) and its surname and given names side by side in
    both orders (``detect.org_names.name_orders``: ``Robert Thompson`` for ``Thompson, Robert J``), so another
    column or the report writing the same person in another layout is found too."""
    from .org_names import name_orders

    parts = org_name_parts(p)
    return list(dict.fromkeys([*parts, *(x for part in parts for x in name_orders(part))]))


def org_named_texts(p: Any) -> set[str]:
    """D36: the real units of column ``p`` written with a fake name in place of the real one, the rest of their text
    real (:data:`ORG_NAMES_KEY` with located name pieces; not those pseudonymised whole)."""
    named = (getattr(p, "params", None) or {}).get(ORG_NAMES_KEY) or {}
    return {t for t, tokens in named.items() if tokens is not None}


# --------------------------------------------------------------------------- D36 over the run
#: Word tokens of a text read for the name parts of the run (accent-folded, lower case: :func:`_word_spans`).
_SEQ_WORD = re.compile(r"[a-z0-9]+")
#: What may stand between two words of one person's name in another column (``Thompson, Robert``, ``R. Thompson``,
#: ``Garcia-Lopez``, ``AnaGarcia``): spaces, a comma, semicolon, period or hyphen, at most three characters.
_NAME_SEP = re.compile(r"[\s,.;\-‐‑]{0,3}")


@lru_cache(maxsize=1 << 14)
def _aligned(text: str) -> tuple[str, list[int], list[int]] | None:
    """(``text`` as the D34 checks read it, and where each of its characters comes from in ``text``:
    ``detect.org_names.align``), or None when folding cannot be followed character by character."""
    from ..privacy.headers import clean_text
    from .org_names import align

    s = clean_text(text)
    where = align(text, s)
    return None if where is None else (s, where[0], where[1])


def _word_spans(text: str) -> list[tuple[str, int, int]] | None:
    """(word, start, end) of the word tokens of ``text`` (accent-folded like the D34 checks, lower case; start and end
    in ``text``), or None when folding cannot be followed (no safe place for a name piece: the unit is pseudonymised
    whole)."""
    got = _aligned(text)
    if got is None:
        return None
    s, starts, ends = got
    low = s.lower()
    if len(low) != len(s):
        return None
    return [(m.group(), starts[m.start()], ends[m.end() - 1]) for m in _SEQ_WORD.finditer(low)]


def _occurrences(words: list[str], seq: tuple[str, ...]) -> list[int]:
    n = len(seq)
    return [i for i in range(len(words) - n + 1) if tuple(words[i:i + n]) == seq] if n else []


@dataclass(frozen=True)
class _Person:
    """The name words of one name part of the run's organisational codes (folded keys): its surname words, given
    names and initials (a given name's first letter too), and the part as written."""

    last: frozenset[str]
    given: frozenset[str]
    initials: frozenset[str]
    part: str


class _RunParts:
    """The name parts of the run's organisational codes (:func:`spread_org_names`): each part's word sequence and its
    name pieces (``detect.org_names.find`` of the part: what the code replaces, with roles), and the person each part
    names (:class:`_Person`), whose name another column may write in another layout or order."""

    def __init__(self, parts: Iterable[str]) -> None:
        from .names import fold_word
        from .org_names import FIRST, INITIAL, LAST, find

        self.seqs: dict[tuple[str, ...], list[tuple[str, list[tuple[str, int, int]], Any]]] = {}
        self.people: list[_Person] = []
        self.by_word: dict[str, list[int]] = {}
        for part in sorted(set(parts)):
            spans = _word_spans(part)
            own = find(part)
            if spans:
                self.seqs.setdefault(tuple(w for w, _, _ in spans), []).append(
                    (part, spans, own.tokens if own is not None else None))
            if own is None or not own.tokens:
                continue
            words: dict[str, set[str]] = {LAST: set(), FIRST: set(), INITIAL: set()}
            for a, b, role in own.tokens:
                k = fold_word(part[a:b])
                if k:
                    words[role].add(k[:1] if role == INITIAL else k)
            if not words[LAST]:
                continue  # an initial after a title ('Mrs. G'): its own sequence only
            person = _Person(frozenset(words[LAST]), frozenset(words[FIRST]),
                             frozenset(words[INITIAL] | {g[:1] for g in words[FIRST]}), part)
            for k in person.last | person.given:
                self.by_word.setdefault(k, []).append(len(self.people))
            self.people.append(person)
        self.firsts = {seq[0] for seq in self.seqs}

    def pieces(self, text: str) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]], list[str]] | None:
        """(the name pieces of the run's parts found in ``text``, as (start, end, role) in ``text``; the spans of
        the occurrences whose pieces could not be placed; the parts found), or None when ``text`` holds none. A
        piece is placed where the text spells it as the part does, case aside (``GARCIA,ANA`` for ``Garcia, Ana``),
        and where the text holds the same person's name in another layout or order (:meth:`_people`: ``Thompson,
        Robert`` or ``Robert Thompson`` for ``Thompson, Robert J``, ``Ana Garcia`` or ``Garcia A`` for ``Garcia,
        Ana``, ``Ms. Garcia``)."""
        placed: dict[tuple[int, int], str] = {}
        unplaced: list[tuple[int, int]] = []
        found: list[str] = []
        spans = _word_spans(text)
        if spans is None:
            from ..privacy.headers import clean_text

            words = _SEQ_WORD.findall(clean_text(text).lower())
            if self.firsts.intersection(words):
                hits = [p for seq, ps in self.seqs.items() if _occurrences(words, seq) for p, _, _ in ps]
                if hits:
                    found += hits
                    unplaced.append((0, len(text)))
        else:
            words = [w for w, _, _ in spans]
            for seq, entries in (self.seqs.items() if self.firsts.intersection(words) else ()):
                for i in _occurrences(words, seq):
                    for part, pspans, tokens in entries:
                        found.append(part)
                        occurrence = (spans[i][1], spans[i + len(seq) - 1][2])
                        if tokens is None:
                            unplaced.append(occurrence)
                            continue
                        for a, b, role in tokens:
                            k1 = next(k for k, (_, ws, we) in enumerate(pspans) if a < we)
                            k2 = max(k for k, (_, ws, we) in enumerate(pspans) if ws < b)
                            ta = spans[i + k1][1] + (a - pspans[k1][1])
                            tb = spans[i + k2][2] - (pspans[k2][2] - b)
                            if text[ta:tb].casefold() == part[a:b].casefold():
                                placed.setdefault((ta, tb), role)
                            else:
                                unplaced.append(occurrence)
        self._people(text, placed, unplaced, found)
        if not found:
            return None
        return [(a, b, r) for (a, b), r in sorted(placed.items())], unplaced, list(dict.fromkeys(found))

    def _people(self, text: str, placed: dict[tuple[int, int], str], unplaced: list[tuple[int, int]],
                found: list[str]) -> None:
        """Add to ``placed`` (or ``unplaced``, ``found``) the names of the run's people (:class:`_Person`) that
        ``text`` writes in any layout: a run of words of one person, with nothing but spaces, commas, periods or
        hyphens between, that holds a surname of that person and another piece of the name (a given name or an
        initial: ``Thompson, Robert``, ``Robert Thompson``, ``Garcia A``, ``R. Abernathy``), or a surname after a title
        (``Ms. Garcia``, ``Coach Garcia``). A bare surname alone stays (the accepted residual, D10)."""
        from ..privacy import headers as names
        from .org_names import FIRST, INITIAL, LAST, _core_end, place

        if not self.people:
            return
        got = _aligned(text)
        s = got[0] if got is not None else names.clean_text(text)
        if not any(ch.isalpha() for ch in s):
            return
        toks = names._tokens(s)
        ids = sorted({x for t in toks for x in self.by_word.get(t.key, ())})
        if not ids:
            return
        titles = names._TITLES | names._ROLE_TITLES
        caps = not any(ch.islower() for ch in s)
        for pid in ids:
            who = self.people[pid]

            def member(t: Any, who: _Person = who) -> bool:
                k = t.key
                return k in who.last or k in who.given or (
                    len(k) == 1 and k in who.initials and (caps or t.raw.isupper()))

            i, n = 0, len(toks)
            while i < n:
                if not member(toks[i]):
                    i += 1
                    continue
                j = i
                while j + 1 < n and member(toks[j + 1]) and _NAME_SEP.fullmatch(s[toks[j].end:toks[j + 1].start]):
                    j += 1
                run = toks[i:j + 1]
                surname = any(t.key in who.last for t in run)
                other = any(t.key not in who.last for t in run)
                titled = i > 0 and toks[i - 1].key in titles and s[toks[i - 1].end:toks[i].start].strip() in ("", ".")
                if surname and (other or titled):
                    found.append(who.part)
                    spots = None
                    if got is not None:
                        _, starts, ends = got
                        spots = [place(text, s, starts, ends, t.start, _core_end(t)) for t in run]
                    if spots is None or any(x is None for x in spots):
                        unplaced.append((0, len(text)))
                    else:
                        if any(t.key in who.given for t in run):  # indexed as written ('Thompson, Robert'); a form
                            found.append(text[spots[0][0]:spots[-1][1]])  # with an initial only by its part
                        for t, spot in zip(run, spots):
                            role = INITIAL if len(t.key) == 1 else LAST if t.key in who.last else FIRST
                            placed.setdefault(spot, role)
                i = j + 1


def spread_org_names(columns: Iterable[Any]) -> list[Any]:
    """D36 over the run (``profile.table.finish_copy_gate``, after the D35 decision of ``columns``, every column of the
    run): a name part found in an organisational code (``Garcia, Ana`` of ``ALG1-03 Garcia, Ana``, ``Ms. Abernathy``)
    is a real person's name wherever the run holds it. Every unit another column would write verbatim (a level the
    D35 allowlist kept for its count, under ``Class``, ``Homeroom``, ``Advisory`` or a pivoted ``Period_3``) that
    holds one, word for word with case and punctuation aside, or the same person's name in another layout, order or
    length (``Thompson, Robert``, ``Robert Thompson``, ``R. Thompson`` for ``Thompson, Robert J``; a bare surname
    stays: :meth:`_RunParts.pieces`), is written like the code: its name pieces replaced from
    the run's one word map (``generate.pseudonyms``: the same fake name as in the code, so ``Class`` and ``Section``
    still join), the rest of its text real (it was kept), or as a whole pseudonym when the name cannot be located
    (:func:`_locate_parts`). Such a unit is a pseudonymised unit with its exact count (placeholder,
    ``pseudonymized_levels``, ``org_code_names_replaced``, an info message with counts only); the leak scanner indexes
    its name parts and never excuses it. A bare surname without the layout (``Abernathy``) is the accepted residual.
    Returns the columns changed."""
    from .org_names import OrgName

    cols = list(columns)
    run = _RunParts(part for p in cols for part in org_name_parts(p))
    if not run.seqs:
        return []
    changed = []
    for p in cols:
        params = getattr(p, "params", None) or {}
        kind = d35_units(p)
        if kind is None or params.get(PENDING_KEY) or getattr(p, "model", None) is None:
            continue
        pseudo = pseudonymized(p)
        pieces: dict[str, list[tuple[int, int, str]]] = {}  # kept unit -> its name pieces
        whole: set[str] = set()  # kept units holding a part whose pieces cannot be placed
        parts: dict[str, list[str]] = {}
        for carrier, items in _carriers(p, kind):
            got = run.pieces(carrier)
            if got is None:
                continue
            placed, unplaced, found = got
            for unit, start in items:
                if unit in pseudo:
                    continue  # pseudonymised whole already (D35)
                end = start + len(unit)
                mine = [(a - start, b - start, r) for a, b, r in placed if start <= a and b <= end]
                if any(a < end and start < b for a, b in unplaced):
                    whole.add(unit)
                if mine or unit in whole:
                    pieces.setdefault(unit, []).extend(mine)
                    parts.setdefault(unit, []).extend(found)
        if not pieces:
            continue
        texts, rows = unit_groups(p)
        found_named: dict[str, dict[str, Any]] = {}
        for k, group in texts.items():
            if not any(t in pieces for t in group):
                continue
            entry: dict[str, Any] = {}
            for t in group:
                tokens = tuple(sorted(set(pieces.get(t, ()))))
                ok = t in pieces and t not in whole and tokens and all(
                    x[1] <= y[0] for x, y in zip(tokens, tokens[1:]))
                entry[t] = OrgName(tokens if ok else None, tuple(dict.fromkeys(parts.get(t, ()))))
            found_named[k] = entry
        _add_named(p, found_named, texts, rows)
        changed.append(p)
    return changed


def _carriers(p: Any, kind: str) -> list[tuple[str, list[tuple[str, int]]]]:
    """The texts of column ``p`` that the run's name parts are looked for in, each with the units it writes and where
    each starts in it: a level or token is its own text; a multi-select value is read whole, since a name's comma may
    be its delimiter (``ALG1-03 Garcia, Ana`` is the items ``ALG1-03 Garcia`` and ``Ana``)."""
    model = p.model
    if kind != ITEMS:
        texts, _ = unit_groups(p)
        return [(t, [(t, 0)]) for group in texts.values() for t in group]
    d = (p.params or {}).get("delimiter")
    na = getattr(p, "na_tokens", None) or frozenset()
    out = []
    for raw in model.nonmissing_counts:
        if not isinstance(raw, str) or not raw.strip() or raw.strip() in na:
            continue
        items: list[tuple[str, int]] = []
        pos = 0
        for x in (raw.split(d) if d else [raw]):
            if x.strip():
                items.append((x.strip(), pos + len(x) - len(x.lstrip())))
            pos += len(x) + (len(d) if d else 0)
        out.append((raw, items))
    return out


def _add_named(p: Any, found: dict[str, dict[str, Any]], texts: dict[str, list[str]], rows: Counter) -> None:
    """Record the units ``found`` (:func:`unit_key` -> {text -> ``OrgName``}) of column ``p`` as D36 named units after
    its D35 decision (:func:`spread_org_names`): pseudonymised units with the next placeholders (most rows first),
    their name pieces and parts PRIVATELY, the public counts and an info message with counts only."""
    params = p.params
    rec = list(params.get(PSEUDO_KEY) or [])
    placeholders = {r[1] for r in rec}
    shown: dict[str, str] = {}
    for k in sorted(found, key=lambda k: (-rows[k], k)):
        ph = PLACEHOLDER.format(len(placeholders) + 1)
        placeholders.add(ph)
        for t in sorted(texts[k]):
            rec.append([t, ph])
            shown[t] = ph
    params[PSEUDO_KEY] = rec
    params[PSEUDO_COUNT_KEY] = len(placeholders)
    p.flags[PSEUDO_COUNT_KEY] = True
    if isinstance(params.get("levels"), list):
        params["levels"] = [shown.get(v.strip(), v) if isinstance(v, str) else v for v in params["levels"]]
    for key in ("value", "token"):
        v = params.get(key)
        if isinstance(v, str) and v.strip() in shown:
            params[key] = shown[v.strip()]
    named = dict(params.get(ORG_NAMES_KEY) or {})
    parts = set(params.get(ORG_NAME_PARTS_KEY) or ())
    for group in found.values():
        for t, o in group.items():
            named[t] = None if o.tokens is None else [list(x) for x in o.tokens]
            parts.update(o.parts)
    params[ORG_NAMES_KEY] = dict(sorted(named.items()))
    params[ORG_NAME_PARTS_KEY] = sorted(parts)
    params[ORG_NAMES_COUNT_KEY] = int(params.get(ORG_NAMES_COUNT_KEY) or 0) + len(found)
    p.flags[ORG_NAMES_COUNT_KEY] = True
    if params.get(COMBO_KEEP_KEY):
        # K-11: a rare whole combination never writes a named item as it is (it takes the item's fake name, as the
        # common value it was kept beside does), and the leak scanner no longer excuses it there (combination_stay)
        params[COMBO_KEEP_KEY] = {c: [[k for k in part if k not in found] for part in keep]
                                  for c, keep in params[COMBO_KEEP_KEY].items()}
    kind = d35_units(p) or LEVELS
    total_rows = sum(rows.values()) or 1
    p.reason += (f"; D36: {len(found)} of {len(texts)} {_UNIT_WORDS[kind]} hold a person's name an organisational code "
                 "of the run holds, written with the same fake name")
    p.messages.append(Message(
        "info", "org_code_names",
        f"{len(found)} of {len(texts)} {_UNIT_WORDS[kind]} ({sum(rows[k] for k in found) / total_rows:.0%} of the "
        "values) hold a person's name that an organisational code of the run holds in an explicit layout: the name "
        "part is written as the same fake name as in that code, the rest stays real (D36); a value whose name cannot "
        "be located is written as a same-shape pseudonym", column=p.name,
        fix=f"--force-factor {p.name} keeps the real values (only if they name no one)"))


def pseudonymized(p: Any) -> dict[str, str]:
    """Real unit (stripped, as the column holds it) -> placeholder, for every unit of the column the D35 allowlist
    did not keep (empty when there is none), and every whole multi-select combination it did not keep
    (:func:`combination_gate`, K-11). The leak scanner's source of truth for what is never excused."""
    rec = (getattr(p, "params", None) or {}).get(PSEUDO_KEY) or ()
    return {r[0]: r[1] for r in rec}


def is_placeholder(text: Any) -> bool:
    """``text`` is a D35 placeholder (``[pseudonymized 3]``)."""
    return isinstance(text, str) and _PLACEHOLDER_RE.fullmatch(text.strip()) is not None


def display_map(p: Any) -> dict[str, str]:
    """Stripped text -> report label for a column with pseudonymised units: each real unit and its pseudonym (the
    generation map in ``p.model.extra``) show as the same placeholder, so the real and synthetic level shares line
    up and no real value is printed (D21, D35)."""
    rec = pseudonymized(p)
    if not rec:
        return {}
    out = dict(rec)
    model = getattr(p, "model", None)
    mapping = (model.extra.get(PSEUDO_MAP_KEY) if model is not None else None) or {}
    for real, fake in mapping.items():
        ph = rec.get(str(real).strip())
        if ph is not None:
            out.setdefault(str(fake).strip(), ph)
    return out


# --------------------------------------------------------------------------- R8-05: --na-value tokens (D35)
#: PRIVATE params key: the ``--na-value`` tokens the column writes as missing values (or verbatim outside its D35
#: units) that the D35 allowlist did not keep, each ``[text, placeholder]`` (text = the token as the column holds it,
#: stripped).
NA_PSEUDO_KEY = "_na_pseudonymized"
#: Public params key and flag: how many distinct ``--na-value`` tokens of the column are pseudonymised (a count).
NA_PSEUDO_COUNT_KEY = "pseudonymized_na_tokens"
#: How the report shows a pseudonymised NA token of the source (the written pseudonym is synthetic text).
NA_PLACEHOLDER = "[pseudonymized NA token {}]"
#: PRIVATE model key (``ColumnModel.extra``): stripped real NA token -> stripped pseudonym (``generate.pseudonyms``).
NA_MAP_KEY = "_na_pseudonym_map"


def custom_na(p: Any) -> frozenset[str]:
    """R8-05: the column's ``--na-value`` tokens, i.e. its NA tokens that are not the default missing-value vocabulary
    (``hints.DEFAULT_NA_TOKENS``, always kept) nor non-finite numbers its storage holds natively (an Arrow float's
    ``inf``: numbers, always kept, K-20)."""
    from ..hints import DEFAULT_NA_TOKENS
    from ..model import numeric_specials

    kept = frozenset(DEFAULT_NA_TOKENS) | numeric_specials(getattr(p, "native", None))
    return frozenset(t for t in (getattr(p, "na_tokens", None) or ()) if t not in kept)


def na_unit_texts(p: Any) -> Callable[[Any], list[str]]:
    """R8-05: raw value -> ``[token]`` when column ``p`` writes that ``--na-value`` token itself (as a missing value,
    a multi-select's whole value, a state column's kept value ...), else ``[]``. A token the column writes as one of
    its D35 units (a factor level: :func:`copy_gate_units` decides it) or under ``--force-factor`` is not one."""
    custom = custom_na(p)
    params = getattr(p, "params", None) or {}
    by_units = params.get(COPY_GATE_KEY) == "allowlist" or getattr(p, "source", None) in EXPLICIT_COPY_SOURCES

    def texts(raw: Any) -> list[str]:
        if not isinstance(raw, str):
            return []
        s = raw.strip()
        if s not in custom or (by_units and unit_texts(p, raw)):
            return []
        return [s]
    return texts


def decide_na_tokens(p: Any, values: Any, keys: Any, min_cell: int, *, personal_header: bool = False,
                     digit_phone: bool = False) -> int:
    """R8-05, D35 for ``--na-value`` tokens: declaring a value missing is no permission to publish it. A token column
    ``p`` writes itself (:func:`na_unit_texts`) is decided like any real text unit (:func:`unit_kept`): kept when it
    is vocabulary (the default NA tokens, ``Unknown``, ``Missing`` ...), a short number or a symbol, or when it has no
    personal-data shape and >= ``min_cell`` distinct entities of the table hold it (counted over the table's
    ``keys``, :func:`_counted`); under a personal header only words that stand for no value are kept. Every other
    token is recorded PRIVATELY (:data:`NA_PSEUDO_KEY`) and gets one same-shape pseudonym for the run
    (``generate.pseudonyms``), still read as missing. Returns the number of pseudonymised tokens."""
    if p.model is None or not custom_na(p) or getattr(p, "role", "row") in ("excluded", "null", "id"):
        return 0
    texts_of = na_unit_texts(p)
    groups: dict[str, list[str]] = {}
    rows: Counter = Counter()
    for counts in (p.model.nonmissing_counts, p.model.missing_counts):
        for raw, c in counts.items():
            for t in texts_of(raw):
                k = unit_key(t)
                group = groups.setdefault(k, [])
                if t not in group:
                    group.append(t)
                rows[k] += int(c)
    if not groups:
        return 0

    def needs_count(k: str) -> bool:
        return (rows[k] >= min_cell and not personal_header and not any(unit_shapes(t, digit_phone) for t in groups[k])
                and not all(free_unit(t) for t in groups[k]))

    holders = {k: rows[k] for k in groups}
    holders.update(_counted(p, values, keys, [k for k in groups if needs_count(k)], rows, texts_of))
    gone = [k for k, group in groups.items()
            if not all(unit_kept(t, holders[k], min_cell, digit_phone=digit_phone, personal_header=personal_header)
                       for t in group)]
    if not gone:
        return 0
    order = sorted(gone, key=lambda k: (-rows[k], k))
    rec = [[t, NA_PLACEHOLDER.format(i + 1)] for i, k in enumerate(order) for t in sorted(groups[k])]
    p.params[NA_PSEUDO_KEY] = rec
    p.params[NA_PSEUDO_COUNT_KEY] = len(gone)
    p.flags[NA_PSEUDO_COUNT_KEY] = True
    shown = {t: ph for t, ph in rec}
    stats = getattr(p, "stats", None) or {}
    if isinstance(stats.get("missing_tokens"), dict):  # the report shows a placeholder, never the token
        merged: Counter = Counter()
        for tok, c in stats["missing_tokens"].items():
            merged[shown.get(tok.strip(), tok) if isinstance(tok, str) else tok] += c
        stats["missing_tokens"] = dict(merged)
    p.messages.append(Message(
        "info", "pseudonymized_na_tokens",
        f"{len(gone)} of {len(groups)} --na-value token(s) are written as consistent same-shape pseudonyms, which "
        f"stay missing-value tokens of the column (D35): a missing-value token is published only when it is "
        f"missing-value vocabulary or held by at least {min_cell} people (or rows) and holds no personal data",
        column=p.name,
        fix="write these missing values as a standard token (NA, N/A, NULL, Unknown) in the source to keep them"))
    return len(gone)


def na_pseudonymized(p: Any) -> dict[str, str]:
    """R8-05: real ``--na-value`` token (stripped) -> placeholder, for every token of the column the D35 allowlist
    did not keep (:func:`decide_na_tokens`); empty when there is none."""
    rec = (getattr(p, "params", None) or {}).get(NA_PSEUDO_KEY) or ()
    return {r[0]: r[1] for r in rec}


def multi_select_written_items(values: dict[str, int], d: str, na_tokens: Iterable[str] = ()) -> dict[str, int]:
    """Item (stripped, as written) -> number of rows holding it; the case-folded form is
    :func:`multi_select_items`. The name screen reads capitals (``Voss, A.``)."""
    na = frozenset(na_tokens)
    items: Counter = Counter()
    for v, c in values.items():
        if v.strip() in na:
            continue
        for p in {p.strip() for p in v.split(d) if p.strip()}:
            items[p] += c
    return dict(items)


def token_kinds(token: str) -> frozenset[str]:
    """Personal-data shapes, a person's name on its own (``detect.names.stray_name``: ``Smith, J.``, ``Quillon
    Varga``) or a sentence (D33) inside one short text. Not applied to values since D35 (their units follow the
    allowlist of :func:`unit_kept`); kept for the header and path screens (D34)."""
    kinds = set(pii_shapes(token))
    if name_values.stray_name(token) or name_levels.value_name_spans(token):  # R5-01: 'Pickup: Ana Lopez'
        kinds.add("name")
    if is_sentence_like(token):
        kinds.add("sentence")
    return frozenset(kinds)


def _pii_message(ctx: DetectContext, kinds: frozenset[str], n: int, total: int, what: str, done: str) -> Message:
    """The ``pii_values`` warning: kinds and counts only, never a value."""
    return Message("warning", "pii_values",
                   f"{n} of {total} {what} hold personal data ({', '.join(sorted(kinds))}): real values are never "
                   f"copied; {done}", column=ctx.name,
                   fix=f"--force-factor {ctx.name} keeps the real values (only if they are not personal data)")


# --------------------------------------------------------------------------- overflow text
def is_code_like(values: dict[str, int]) -> bool:
    """Short single-token codes: no whitespace, <= 16 characters, digits in >= half the distinct values."""
    if not values:
        return False
    with_digit = 0
    for v in values:
        if len(v) > CODE_MAX_LENGTH or any(ch.isspace() for ch in v):
            return False
        if any(ch.isdigit() for ch in v):
            with_digit += 1
    return with_digit >= CODE_DIGIT_SHARE * len(values)


def classify_overflow_text(ctx: DetectContext) -> Detection:
    """Text that is not a factor (> factor_max or ``--force-text``): FREE_TEXT or HIGH_CARD.

    Called by ``detect/types.py`` for non-digit-code text columns with more
    distinct values than ``--factor-max`` and for ``--force-text``. FREE_TEXT
    when the row-weighted mean is >= 4 words or > 40 characters, or the header
    has a comment/note/alert/narrative hint; else HIGH_CARD_CATEGORICAL, with
    the subtype ``code`` for short single-token code columns (same-shape masks,
    D13). Both use the typed missing policy (NA tokens written verbatim).
    """
    tv = ctx.typed_values
    words, length = ctx.mean_words_and_length(tv if tv else None)
    params: dict[str, Any] = {"mean_words": round(words, 2), "mean_length": round(length, 2),
                              "_min_cell": int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))}
    hint = guarded_hint(ctx)
    if hint is not None:
        params["pii_hint"] = hint
    if ctx.hints.free_text or words >= 4 or length > 40:
        why = (f"free-text header hint '{ctx.hints.free_text}'" if ctx.hints.free_text
               else f"mean {words:.1f} words / {length:.0f} characters")
        det = Detection(ColumnType.FREE_TEXT, params=params, reason=f"non-factor text; {why}")
        det.missing_policy = "typed"
        return det
    n = ctx.casefold_distinct()
    pii = f"; PII header hint '{hint}': real values never copied" if hint is not None else ""
    if is_code_like(tv):
        det = Detection(ColumnType.HIGH_CARD_CATEGORICAL, subtype="code", params=params,
                        reason=f"non-factor short codes with {n} distinct values (factor-max {ctx.factor_max}): "
                               f"same-shape masks{pii}")
    else:
        det = Detection(ColumnType.HIGH_CARD_CATEGORICAL, params=params,
                        reason=f"non-factor text with {n} distinct values (factor-max {ctx.factor_max}){pii}")
    det.missing_policy = "typed"
    return det


def finish_forced(ctx: DetectContext, det: Detection) -> Detection:
    """``--force-type COL=<type>`` for the text/PII types: the private parameters value detection sets.

    Called by ``detect/types.py`` (``_forced``) for CONSTANT, FLAG_SINGLE_TOKEN,
    HIGH_CARD_CATEGORICAL, FREE_TEXT and the AREA C types. Adds ``_min_cell``
    (``--min-cell``: area-code/ZIP retention and report lists; without it a
    forced column used 10), ``_factor_max`` for MULTI_SELECT, and the typed
    missing policy of :data:`TYPED_POLICY_TYPES` (NA tokens written verbatim, as
    with ``--pii-col``). Other types are returned unchanged.
    """
    if det.type in TYPED_POLICY_TYPES or det.type in (T.STATE, T.MULTI_SELECT):
        det.params.setdefault("_min_cell", int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL)))
    if det.type is T.MULTI_SELECT:
        det.params.setdefault("_factor_max", ctx.factor_max)
        if not det.params.get("delimiter"):
            # the delimiter the generator would infer (most rows): the D35 allowlist decides the same items
            rows = {d: sum(c for v, c in ctx.text_values.items() if d in v) for d in MULTI_DELIMITERS}
            best = max(MULTI_DELIMITERS, key=lambda d: rows[d])  # ties: ';' before '|' before ','
            if rows[best]:
                det.params["delimiter"] = best
    if det.type in TYPED_POLICY_TYPES:
        det.missing_policy = "typed"
    return det


def pii_subtype_detection(ctx: DetectContext, subtype: str) -> Detection:
    """Detection for ``--pii-col COL=SUBTYPE`` (subtypes: :data:`PII_SUBTYPES`).

    The column gets the PII type whatever its values look like; generators fall
    back to same-shape masks for values that do not fit the type.
    """
    key = subtype.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PII_SUBTYPES:
        raise UsageError(f"--pii-col {ctx.name}={subtype}: unknown subtype; use one of {', '.join(sorted(PII_SUBTYPES))}")
    t, sub = PII_SUBTYPES[key]
    params: dict[str, Any] = {"pii_subtype": key,
                              "_min_cell": int(getattr(ctx.settings, "min_cell", DEFAULT_MIN_CELL))}
    det = Detection(t, subtype=sub, params=params, source="pii-col", reason=f"--pii-col {key}")
    if t in TYPED_POLICY_TYPES:
        det.missing_policy = "typed"
    return det


# --------------------------------------------------------------------------- statistics (report-safe)
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def _wquantiles(values: np.ndarray, weights: np.ndarray, qs: Iterable[float] = QUANTILES) -> dict[str, float]:
    if values.size == 0:
        return {}
    order = np.argsort(values, kind="stable")
    v, w = values[order].astype(float), weights[order].astype(float)
    cum = np.cumsum(w)
    total = cum[-1]
    out = {}
    for q in qs:
        idx = int(np.searchsorted(cum, q * total, side="left"))
        out[f"p{int(round(q * 100))}"] = float(v[min(idx, v.size - 1)])
    return out


def _r(x: float) -> float:
    return float(round(float(x), 6))


def case_style(s: str) -> str:
    """'upper' | 'lower' | 'title' | 'mixed' | 'none' (no cased letters)."""
    if s.isupper():
        return "upper"
    if s.islower():
        return "lower"
    if s.istitle():
        return "title"
    if s.upper() == s.lower():
        return "none"
    return "mixed"


_PUNCT_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\u2019\u2018\u201c\u201d"
_PUNCT_SEARCH = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
_DIGIT_SEARCH = re.compile(r"\d")
_ASCII_DIGIT = re.compile(r"[0-9]")
_LOWER_RUN = re.compile(r"[a-z]+")
_UPPER_RUN = re.compile(r"[A-Z]+")


def shape_stats(counts: Counter) -> dict[str, Any]:
    """Length/word quantiles, case styles and character-class shares (row-weighted)."""
    vals = list(counts)
    w = np.fromiter((counts[v] for v in vals), dtype=np.int64, count=len(vals))
    total = int(w.sum())
    if not total:
        return {}
    chars = np.fromiter((len(v) for v in vals), dtype=np.int64, count=len(vals))
    words = np.fromiter((len(v.split()) for v in vals), dtype=np.int64, count=len(vals))
    cases: Counter = Counter()
    flags = {"newline": 0, "digit": 0, "punctuation": 0, "non_ascii": 0, "hyphen": 0, "apostrophe": 0}
    for v, c in zip(vals, w.tolist()):
        cases[case_style(v)] += c
        if "\n" in v or "\r" in v:
            flags["newline"] += c
        if _DIGIT_SEARCH.search(v):
            flags["digit"] += c
        if _PUNCT_SEARCH.search(v):
            flags["punctuation"] += c
        if not v.isascii():
            flags["non_ascii"] += c
        if "-" in v:
            flags["hyphen"] += c
        if "'" in v or "\u2019" in v:
            flags["apostrophe"] += c
    return {
        "char_length": _wquantiles(chars, w),
        "word_count": _wquantiles(words, w),
        "mean_chars": _r((chars * w).sum() / total),
        "case": {k: _r(v / total) for k, v in sorted(cases.items())},
        "shares": {k: _r(v / total) for k, v in flags.items()},
    }


def _kept(counter: Counter, k: int, limit: int = 50) -> tuple[dict[str, int], int]:
    """Entries held by >= k rows (most common first, at most ``limit``), and the count of the others."""
    kept = {key: int(c) for key, c in counter.most_common() if c >= k}
    if len(kept) > limit:
        kept = dict(list(kept.items())[:limit])
    return kept, len(counter) - len(kept)


#: RFC 2606 reserved second-level domains and top-level domains (never a real organisation).
RESERVED_SLD = ("example.org", "example.com", "example.net")
RESERVED_TLD = ("test", "example", "invalid", "localhost")


def is_reserved_domain(domain: str) -> bool:
    """True for RFC 2606 reserved names (``example.org``, ``d4.example.org``, ``*.test`` ...)."""
    d = domain.casefold().rstrip(".")
    return (any(d == s or d.endswith("." + s) for s in RESERVED_SLD)
            or any(d == t or d.endswith("." + t) for t in RESERVED_TLD))


def _domain_counts(counter: Counter, k: int) -> dict[str, Any]:
    """E-mail domains / URL hosts, report-safe (D21): real names are not reproduced in the output, so only
    counts are reported; reserved names (the synthetic side) are listed by name."""
    held = sorted((int(c) for c in counter.values() if c >= k), reverse=True)
    return {"n": len(counter), "count_profile": held[:50], "below_min_cell": sum(1 for c in counter.values() if c < k),
            "reserved": {d: int(c) for d, c in counter.most_common() if c >= k and is_reserved_domain(d)}}


def local_mask(local: str) -> str:
    """Generalised e-mail local-part mask: letter runs -> 'a'/'A', digits -> '9' each, others kept."""
    if local.isascii():
        return _UPPER_RUN.sub("A", _LOWER_RUN.sub("a", _ASCII_DIGIT.sub("9", local)))
    out = []
    prev = ""
    for ch in local:
        if ch.isalpha():
            cls = "A" if ch.isupper() else "a"
            if prev != cls:
                out.append(cls)
            prev = cls
        else:
            out.append("9" if ch.isdigit() else ch)
            prev = ""
    return "".join(out)


def text_stats(values: list[str], profile: Any) -> dict[str, Any]:
    """AREA C hook: extra statistics for text/PII types (case, punctuation, domain counts ...).

    Called by ``profile/stats.py`` for text and PII columns with the stripped
    non-missing values (real or synthetic). Report-safe aggregates only: never
    a value. Lists of masks, e-mail domains, phone area codes, URL hosts and
    street suffixes (design 3 #25-31) keep only entries held by >= ``--min-cell``
    rows (``params['_min_cell']``, default 10). ZIP codes and states are not
    label-safe types (D21): only their counts are reported, never the codes.
    """
    return text_stats_counts(Counter(values), profile)


def text_stats_counts(counts: dict[str, int], profile: Any) -> dict[str, Any]:
    """AREA C statistics from counts in first-appearance order, without a row list."""
    t = profile.type
    if t in (T.IDENTIFIER, T.DIGIT_CODE) or not counts:
        return {}
    k = int(profile.params.get("_min_cell", DEFAULT_MIN_CELL)) if getattr(profile, "params", None) else DEFAULT_MIN_CELL
    total = sum(counts.values())
    out: dict[str, Any] = {}
    if t in (T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL, T.PERSON_NAME, T.CITY, T.STREET_ADDRESS, T.MULTI_SELECT):
        out["text_shape"] = shape_stats(counts)
    if t in (T.HIGH_CARD_CATEGORICAL, T.FREE_TEXT, T.PERSON_NAME, T.CITY):
        freq = np.asarray(sorted(counts.values(), reverse=True), dtype=float)
        out["frequency_skew"] = {"top1_share": _r(freq[0] / total), "top5_share": _r(freq[:5].sum() / total),
                                 "singleton_share": _r(float((freq == 1).sum()) / len(freq))}
    if t is T.PERSON_NAME:
        layouts: Counter = Counter()
        for v, c in counts.items():
            layouts[name_layout(v)] += c
        out["layouts"] = {key: _r(c / total) for key, c in layouts.most_common()}
    elif t is T.EMAIL:
        masks: Counter = Counter()
        domains: Counter = Counter()
        for v, c in counts.items():
            local, at, domain = v.rpartition("@")
            if not at:
                masks["<not an e-mail>"] += c
                continue
            masks[local_mask(local)] += c
            domains[domain.casefold()] += c
        kept, other = _kept(masks, k)
        out["local_masks"] = kept
        out["local_masks_below_min_cell"] = other
        dc = _domain_counts(domains, k)
        out["n_domains"] = dc["n"]
        out["domain_count_profile"] = dc["count_profile"]
        out["domains_below_min_cell"] = dc["below_min_cell"]
        out["domains"] = dc["reserved"]  # D21: only reserved (synthetic) domains are named
    elif t is T.PHONE:
        masks = Counter()
        areas: Counter = Counter()
        ext = 0
        ex555 = 0
        for v, c in counts.items():
            masks[char_mask(v)] += c
            parts = phone_parts(v)
            if parts is None:
                continue
            if parts[1]:
                ext += c
            digits = _NON_DIGIT.sub("", parts[0])
            if len(digits) == 11 and digits[0] == "1":
                digits = digits[1:]
            if len(digits) == 10:
                areas[digits[:3]] += c
                ex555 += c if digits[3:6] == "555" else 0
            elif len(digits) == 7:
                ex555 += c if digits[:3] == "555" else 0
        km, om = _kept(masks, k)
        ka, oa = _kept(areas, k)
        out.update({"masks": km, "masks_below_min_cell": om, "n_area_codes": len(areas), "area_codes": ka,
                    "area_codes_below_min_cell": oa, "extension_share": _r(ext / total),
                    "exchange_555_share": _r(ex555 / total)})
    elif t is T.SSN:
        masks = Counter()
        invalid_area = 0
        for v, c in counts.items():
            masks[char_mask(v)] += c
            d = "".join(ch for ch in v if ch.isdigit())
            if len(d) == 9 and (d[0] == "9" or d[:3] in ("000", "666")):
                invalid_area += c
        km, om = _kept(masks, k)
        out.update({"masks": km, "masks_below_min_cell": om, "invalid_area_share": _r(invalid_area / total)})
    elif t is T.ZIP:
        z5: Counter = Counter()
        plus4 = 0
        widths: Counter = Counter()
        for v, c in counts.items():
            m = ZIP_HINTED_RE.fullmatch(v)
            if not m:
                widths["other"] += c
                continue
            z5[m.group("zip5")] += c
            widths[str(len(m.group("zip5")))] += c
            if m.group("plus4"):
                plus4 += c
        # D21: ZIP codes are not label-safe, so only counts (never the codes) are reported
        out.update({"n_zip5": len(z5), "zip5_held_by_min_cell": sum(1 for c in z5.values() if c >= k),
                    "zip5_below_min_cell": sum(1 for c in z5.values() if c < k),
                    "zip5_rows_below_min_cell": int(sum(c for c in z5.values() if c < k)),
                    "zip5_count_profile": sorted((int(c) for c in z5.values() if c >= k), reverse=True)[:50],
                    "plus4_share": _r(plus4 / total), "zip5_widths": dict(sorted(widths.items()))})
    elif t is T.STATE:
        st: Counter = Counter()
        for value, count in counts.items():
            st[value.strip().upper()] += count
        # D21: STATE is not a label-safe type, so the report gets counts only (the output keeps the states)
        out.update({"n_states": len(st), "states_below_min_cell": sum(1 for c in st.values() if c < k),
                    "state_count_profile": sorted((int(c) for c in st.values() if c >= k), reverse=True)})
    elif t is T.STREET_ADDRESS:
        suffixes: Counter = Counter()
        units = 0
        houses: list[float] = []
        hw: list[int] = []
        matched = 0
        for v, c in counts.items():
            m = ADDRESS_RE.fullmatch(v)
            if not m:
                continue
            matched += c
            suffixes[m.group("suffix").casefold() + m.group("dot")] += c
            houses.append(float(m.group("num")))
            hw.append(c)
            rest = m.group("rest").casefold()
            if any(re.search(rf"(?<![a-z]){re.escape(u)}\b", rest) for u in ("apt", "unit", "suite", "ste", "#")):
                units += c
        ks, os_ = _kept(suffixes, k)
        out.update({"match_share": _r(matched / total), "suffixes": ks, "suffixes_below_min_cell": os_,
                    "unit_share": _r(units / total)})
        if houses:
            out["house_number"] = _wquantiles(np.asarray(houses), np.asarray(hw), (0.10, 0.25, 0.50, 0.75, 0.90))
    elif t is T.IP:
        v4 = sum(c for v, c in counts.items() if IPV4_RE.fullmatch(v))
        priv = 0
        for v, c in counts.items():
            try:
                if ipaddress.ip_address(v).is_private:
                    priv += c
            except ValueError:
                pass
        masks = Counter()
        for v, c in counts.items():
            masks[char_mask(v)] += c
        km, om = _kept(masks, k)
        out.update({"ipv4_share": _r(v4 / total), "private_share": _r(priv / total), "masks": km,
                    "masks_below_min_cell": om})
    elif t is T.URL:
        hosts: Counter = Counter()
        schemes: Counter = Counter()
        for v, c in counts.items():
            m = URL_PARSE_RE.fullmatch(v)
            if not m:
                continue
            hosts[m.group("host").casefold()] += c
            schemes[(m.group("scheme") or "").casefold().rstrip("/:") or "none"] += c
        hc = _domain_counts(hosts, k)
        out.update({"n_hosts": hc["n"], "host_count_profile": hc["count_profile"],
                    "hosts_below_min_cell": hc["below_min_cell"], "hosts": hc["reserved"],
                    "schemes": {s: _r(c / total) for s, c in schemes.most_common()}})
    elif t is T.MULTI_SELECT:
        d = profile.params.get("delimiter") if getattr(profile, "params", None) else None
        if d:
            na = getattr(profile, "na_tokens", frozenset()) or frozenset()
            per: Counter = Counter()
            items: set[str] = set()
            for v, c in counts.items():
                parts = [] if v.strip() in na else [p.strip().casefold() for p in v.split(d) if p.strip()]
                per[len(parts)] += c
                items.update(parts)
            out.update({"delimiter": d, "n_items": len(items), "n_combinations": len(counts),
                        "items_per_value": {str(n): int(c) for n, c in sorted(per.items())}})
    return out


# --------------------------------------------------------------------------- data dictionary
_DOC_NOTES = {
    T.EMAIL: "synthetic addresses on reserved domains (example.org/.com/.net); never a real address",
    T.PERSON_NAME: "fake names (Faker en_US); never a real name",
    T.PHONE: "ordinary numbers in the source layout; never a source phone base",
    T.SSN: "fake SSNs in the invalid area range 900-999; never a real SSN",
    T.ZIP: "ZIP codes held by >= min-cell people (rows without an ID) kept; rarer ones replaced; +4 randomised; read "
           "as character",
    T.STATE: "real state codes/names kept by exact quota",
    T.CITY: "fake city names (Faker en_US); never a real city of the column",
    T.STREET_ADDRESS: "fake street addresses (Faker street names, source suffixes and unit forms)",
    T.IP: "documentation/benchmark address ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24, 198.18.0.0/15, "
          "2001:db8::/32)",
    T.URL: "hosts replaced by example.org/.com/.net; paths are same-shape random text",
    T.MULTI_SELECT: "delimited multi-select items",
    T.LAT: "independent decimals inside the source bounding box; points need not be real places",
    T.LON: "independent decimals inside the source bounding box; points need not be real places",
    T.FREE_TEXT: "lorem filler text matched to the length/word/punctuation shape; never real text",
    T.HIGH_CARD_CATEGORICAL: "invented labels keeping the distinct count, frequency skew and lengths",
}


def doc_fields(profile: Any, syn_stats: dict[str, Any]) -> dict[str, Any]:
    """AREA C hook: extra DATA_DICTIONARY / schema.json fields for text/PII columns.

    Called by ``report/docs.py`` with the column profile and the SYNTHETIC
    statistics (e.g. ``{'mask': '(999) 999-9999'}``); a ``'readr'`` key replaces
    the type's readr letter. Report-safe only: every field comes from the
    synthetic statistics or from value-free detection parameters (delimiter,
    name part). All text/PII types are R ``character`` (readr ``c``; ZIP codes
    stay character so leading zeros survive); LAT/LON are ``numeric``.
    """
    t = profile.type
    out: dict[str, Any] = {}
    note = _DOC_NOTES.get(t)
    params = getattr(profile, "params", {}) or {}
    if t is T.HIGH_CARD_CATEGORICAL and getattr(profile, "subtype", None) == "code":
        note = "same-shape random codes (character-class masks); never a real code"
        if params.get("numeric_storage"):
            note = "same-shape random numbers (personal-data header; digits masked); never a real value"
            short = int(((syn_stats or {}).get("length") or {}).get("max", 0) or 0) <= 9  # fits a 32-bit int
            integer = params["numeric_storage"] in (ColumnType.INTEGER.value, ColumnType.INTEGER_YEAR.value)
            out["readr"] = "i" if integer and short else "d"
    if note:
        out["note"] = note
    syn = syn_stats or {}
    masks = syn.get("masks") or syn.get("top_masks") or {}
    if t in (T.PHONE, T.SSN, T.IP) and masks:
        out["mask"] = next(iter(masks))
    if t is T.EMAIL and syn.get("domains"):
        out["domains"] = sorted(syn["domains"])
    if t is T.ZIP:
        out["readr"] = "c"
        widths = syn.get("zip5_widths") or {}
        out["format"] = "99999-9999" if syn.get("plus4_share") else "99999"
        if widths and set(widths) - {"5", "other"}:
            out["format_note"] = "some codes have 4 digits (leading zero lost upstream)"
    if t is T.PERSON_NAME and getattr(profile, "subtype", None):
        out["name_part"] = profile.subtype
        if syn.get("layouts"):
            out["layout"] = next(iter(syn["layouts"]))
    if t is T.MULTI_SELECT:
        if params.get("delimiter"):
            out["delimiter"] = params["delimiter"]
        if "n_items" in syn:
            out["n_items"] = syn["n_items"]
    if t in (T.LAT, T.LON):
        if "min" in syn and "max" in syn:
            out["bounding_box"] = [syn["min"], syn["max"]]
        out["readr"] = "d"
    if t in (T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL, T.PERSON_NAME, T.CITY, T.STREET_ADDRESS):
        shape = syn.get("text_shape") or {}
        if shape.get("char_length"):
            out["char_length_p50"] = shape["char_length"].get("p50")
    return out
