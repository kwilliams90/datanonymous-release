"""Identifier detection, ID-candidate warnings and check digits (FOUNDATION core; AREA D extends).

FOUNDATION implements (design D7, D8, D9, section 3 #13):

* IDs come only from ``--id-col`` / ``--person-id`` / the default ID name list
  (section 10 and its ID stems with any ID word, ``hints.default_id_hint``;
  ``--no-default-ids`` turns it off) / person-ID headers (D10,
  ``hints.person_id_hint``) / K-02, fail-safe: a header with a student word, a
  student-ID code or a final ID word over ID-like keys (``Student_SK``, ``Stu``,
  ``CALPADS SSID``, ``Admission No``; :func:`mark_weak_id`, decided per table by
  ``profile.table``). :func:`identifier_detection` builds their
  parameters: shape (digits vs patterned), widths, zero padding, and the
  subtype ``sequential`` for *row numbers*: unique values (one row per ID) whose
  integers are exactly 1..n (D9 keeps those 1..n). Repeated 1..k values are
  person keys, not row numbers, and are remapped.
* :func:`normalize_id` - the D8 matching key (case, punctuation, leading zeros).

AREA D (design section 3 #13, D7):

* :func:`id_candidate_reason` - why a column that is NOT an ID looks like one
  (the D7 warning lists it and the report redacts its value statistics; it is
  never promoted to an ID: that needs ``--id-col`` / ``--person-id``). Rules, in
  order:

  1. every value is a UUID (8-4-4-4-12 hex digits);
  2. every value is a 32/40/64-character hex hash (MD5/SHA-1/SHA-256 shape);
  3. unique sequential integers (FOUNDATION);
  4. unique values with an ID-like header (FOUNDATION);
  5. unique fixed-width codes (one value length) and
  6. >= 95% unique values with one character-shape mask covering >= 90% of rows;
  7. repeated keys of at least :data:`REPEATED_KEY_DIGITS` digits under a header
     that names no measure (harden 2, A2-3: a state student ID on each of a
     student's rows under a header no rule knows; D38 put the known acronyms,
     ``NYSSIS``, ``STN``, ``FLEID``, on the default ID list). Such a column also
     counts who holds a value (D35) in a table with no ID of the row's own
     person, and D38: in a table with no recognised ID column it FAILs its file
     until a flag names it (``profile.table.unrecognised_key_columns``).

  Repeated person keys under an ID header (``Admission No``, ``Aeries ID``) are
  no candidates: they are ID columns (K-02, :func:`mark_weak_id`).

  Rules 5-6 apply to code-like text (FACTOR, DIGIT_CODE, non-factor text
  labels) and to integer columns (a unique 6-digit badge number without a
  leading zero is an INTEGER; a discrete one writes every real value), not to
  decimals. An all-unique measurement of mixed widths (Expense_Semester: 4-5
  digits, the widest mask on 86% of rows) does not reach the 90% shape share.
  Missing-like placeholders (``N/A``, ``NULL`` ..., the typed NA tokens) are
  not counted as values by any rule. Organisational codes (D10, real values
  kept), value-validated or header-hinted personal data (handled by their own
  generators) and date/time columns are never ID candidates.
* :func:`is_uuid`, :func:`hex_hash_length` - value shapes shared with the
  synthetic ID generator (hex IDs keep their hex character class).
* :func:`luhn_check_digit` / :func:`luhn_valid` - the opt-in ``--id-check luhn``
  (design section 4: check digits are not detected; when asked, synthetic digit
  IDs get a valid Luhn check digit).

R5-02 (e-mail addresses and usernames as IDs), shared by the ID generator and
the leak scanner so both apply one rule:

* :func:`is_email_id` - an ID value that is a whole e-mail address
  (``jsmith@district.edu``): its synthetic ID is an e-mail address too, with a
  name-like local part on a reserved domain (D24), never a mask of the real one;
* :func:`username_stem` - the *username* of an ID value: the case-folded
  letters of an e-mail address's local part, or of the value without its
  namespace's kept code prefix (``jsmith23`` -> ``jsmith``, ``ana.lopez`` ->
  ``analopez``; ``S104019`` with the kept prefix ``S`` has none). No synthetic
  ID has the username of a real ID of the run;
* :func:`name_like` - a letter run that reads as a person's name or username
  (a known given name or surname, ``jsmith``, ``johnsmith``): never kept as a
  common ID prefix;
* :func:`kept_prefix` / :func:`checked_prefix` - the letter prefix patterned
  synthetic IDs keep: the common leading letters of **every** ID of the
  namespace; two or more letters only when at least ``--min-cell`` distinct
  IDs share them, they are not name-like and no real ID, else the first letter
  (``S`` of ``S104019``). The leak scan re-checks the prefix the generator used
  before it reads usernames without it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import AbstractSet, Any, Iterable

from ..model import TYPE_INFO, ColumnType
from .context import DetectContext, Detection

_PUNCT = re.compile(r"[^0-9a-z]+")
UUID_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")
_HEX_RE = re.compile(r"[0-9A-Fa-f]+")
#: Hex-hash lengths and the hash they look like (design section 3 #13).
HASH_LENGTHS = {32: "MD5", 40: "SHA-1", 64: "SHA-256"}
#: Candidate rules 5-6 (shape rules) apply to code-like text and to integers (unique fixed-width integers such
#: as an unpadded badge number are IDs as often as codes; a discrete integer column writes every real value).
_SHAPE_TYPES = frozenset({ColumnType.FACTOR, ColumnType.DIGIT_CODE, ColumnType.HIGH_CARD_CATEGORICAL,
                          ColumnType.INTEGER})
#: Never ID candidates: org codes keep real values (D10); personal data and dates have their own generators.
_NEVER_CANDIDATE_FAMILIES = frozenset({"pii", "date", "empty"})
CANDIDATE_UNIQUE_SHARE = 0.95
CANDIDATE_MASK_SHARE = 0.90


def is_uuid(value: str) -> bool:
    """True for an 8-4-4-4-12 hex UUID string."""
    return UUID_RE.fullmatch(value) is not None


def hex_hash_length(value: str) -> int | None:
    """32/40/64 for a hex string of that length (MD5/SHA-1/SHA-256 shape), else None."""
    n = len(value)
    if n in HASH_LENGTHS and _HEX_RE.fullmatch(value):
        return n
    return None


def is_hex_id(value: str) -> bool:
    """A UUID or a hex hash: synthetic replacements keep the hex character class."""
    return is_uuid(value) or hex_hash_length(value) is not None


#: R5-02: a whole ID value that is an e-mail address: one '@', no whitespace, a dotted domain ending in a
#: top-level domain of >= 2 letters (``jsmith@district.edu``; ``S12@2021`` is no address).
EMAIL_ID_RE = re.compile(r"[^\s@]+@[\w\-]+(?:\.[\w\-]+)*\.[^\W\d_]{2,}")
#: Fewest letters of a username part (:func:`username_stem`) compared as a username.
MIN_USERNAME_LETTERS = 3


def is_email_id(value: str) -> bool:
    """R5-02: True when the (stripped) ID value is a whole e-mail address."""
    return "@" in value and EMAIL_ID_RE.fullmatch(value.strip()) is not None


def username_stem(value: str, prefix: str = "") -> str | None:
    """R5-02 username rule: the case-folded letters of an ID value's user part, or None.

    The user part is an e-mail address's local part, else the value without ``prefix`` (its namespace's kept
    code prefix, case-folded, as :func:`kept_prefix` returns it). None when that part has fewer than
    :data:`MIN_USERNAME_LETTERS` letters, or for a UUID / hex hash. ``jsmith23`` -> ``jsmith``,
    ``Ana.Lopez@district.edu`` -> ``analopez``, ``S104019`` (prefix ``s``) -> None.
    """
    s = value.strip()
    if s.isascii() and len(s) - len(s.translate(_NO_ASCII_LETTERS)) < MIN_USERNAME_LETTERS:
        return None  # fast exit (digit IDs, ``S104019``): too few letters anywhere
    if "@" in s and is_email_id(s):
        s = s.rpartition("@")[0]
    else:
        if is_hex_id(s):
            return None
        if prefix and s[:len(prefix)].casefold() == prefix:
            s = s[len(prefix):]
    letters = "".join(ch for ch in s.casefold() if ch.isalpha())
    return letters if len(letters) >= MIN_USERNAME_LETTERS else None


#: ASCII letters, deleted by ``str.translate`` to count a value's letters fast.
_NO_ASCII_LETTERS = str.maketrans("", "", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def name_like(word: str) -> bool:
    """R5-02: a letter run that reads as a person's name or username.

    Everyday words (``detect.names.common_words``: ``case``, ``test``, ``teacher``) and runs of one or two
    letters never are. Otherwise: a known given name or surname (``detect.names.known_names``: the pinned Faker
    lists), an initial and a surname (``jsmith``), a given name and an initial (``annak``), or two known names
    run together (``johnsmith``, ``analopez``).
    """
    from .names import common_words, known_names

    w = word.casefold()
    if len(w) < 3 or w in common_words():
        return False
    first, last = known_names()
    if w in first or w in last:
        return True
    if len(w) >= 4 and (w[1:] in last or w[1:] in first or w[:-1] in first or w[:-1] in last):
        return True
    return any((w[:i] in first or w[:i] in last) and (w[i:] in first or w[i:] in last)
               for i in range(2, len(w) - 1))


#: Distinct IDs that must share a kept prefix of two or more letters (the ``--min-cell`` default).
PREFIX_MIN_IDS = 10


def common_letter_prefix(values: Iterable[str]) -> str:
    """Longest common leading run of letters of ``values``, compared case-insensitively (case-folded)."""
    folded = [v.casefold() for v in values]
    if not folded:
        return ""
    lo, hi = min(folded), max(folded)  # the common prefix of all values is the one of the extremes
    i = 0
    while i < min(len(lo), len(hi)) and lo[i] == hi[i] and lo[i].isalpha():
        i += 1
    return lo[:i]


def checked_prefix(prefix: str, n_ids: int | None = None, min_ids: int = PREFIX_MIN_IDS) -> str:
    """R5-02: ``prefix`` (case-folded letters) if a synthetic ID may keep it, else its first letter.

    One letter is always fine. Two or more letters must not be name-like (:func:`name_like`) and, when
    ``n_ids`` (the distinct IDs sharing it) is given, be shared by at least ``min_ids`` IDs: a code of the ID
    scheme (``STU``), not one person's username (``jsmith``) or a stem shared by a few accounts.
    """
    if len(prefix) <= 1:
        return prefix
    if name_like(prefix) or (n_ids is not None and n_ids < min_ids):
        return prefix[:1]
    return prefix


def kept_prefix(reps: list[str], banned: AbstractSet[str] = frozenset(), min_ids: int = PREFIX_MIN_IDS) -> str:
    """R5-02 (D9 'prefix'): the case-folded letter prefix the patterned synthetic IDs of one namespace keep.

    ``reps``: one value per distinct patterned ID (no digit IDs, no e-mail addresses). The common leading
    letters of **every** value, checked by :func:`checked_prefix` (``min_ids``: ``--min-cell``); a prefix whose
    D8 key is in ``banned`` (a real ID of the run) is cut to its first letter, and that letter dropped if it is
    one too. UUIDs and hex hashes keep no prefix (hex digits throughout).
    """
    if not reps or all(is_hex_id(r) for r in reps):
        return ""
    pre = common_letter_prefix(reps)
    if len(pre) > 1 and normalize_id(pre) in banned:
        pre = pre[:1]
    pre = checked_prefix(pre, len(reps), min_ids)
    if pre and normalize_id(pre) in banned:
        pre = ""
    return pre


def luhn_check_digit(body: str) -> int:
    """The Luhn check digit to append to the digit string ``body``."""
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = ord(ch) - 48
        if i % 2 == 0:  # these positions are doubled once the check digit is appended
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def luhn_valid(digits: str) -> bool:
    """True if the digit string ends with a valid Luhn check digit (leading zeros do not matter)."""
    if len(digits) < 2 or not (digits.isascii() and digits.isdigit()):
        return False
    return luhn_check_digit(digits[:-1]) == ord(digits[-1]) - 48


def normalize_id(value: str) -> str:
    """D8 key: case-folded, punctuation removed, leading zeros stripped. An integer written as a float
    (``100234.0``: an integer ID column with a missing value stored as float64) is that integer (``100234``), never
    the digits glued across the point (K-02)."""
    s = value.strip()
    if s.isascii() and s.isdigit():  # fast path for plain digit IDs
        return s.lstrip("0") or "0"
    if "." in s:
        m = _INTEGRAL_FLOAT_KEY_RE.fullmatch(s)
        if m:
            return m.group(1).lstrip("0") or "0"
    key = _PUNCT.sub("", s.casefold()).lstrip("0")
    return key or "0"


#: An integer written as a float (``100234.0``): its D8 key and its synthetic ID are the integer's (K-02).
_INTEGRAL_FLOAT_KEY_RE = re.compile(r"([0-9]+)\.0+")
#: Harden 2: an integer written with thousands separators (``100,226``: an Excel 'Number' format saved to CSV); its D8
#: key is the integer's (the separators are punctuation) and its synthetic ID is written the same way.
_THOUSANDS_INT_RE = re.compile(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+")


def integral_float_parts(value: str) -> tuple[str, str] | None:
    """K-02: (integer digits, the rest) of an integer written as a float (``100234.0`` -> ``('100234', '.0')``)."""
    m = _INTEGRAL_FLOAT_KEY_RE.fullmatch(value)
    return (m.group(1), value[m.end(1):]) if m else None


def thousands_digits(value: str) -> str | None:
    """Harden 2: the digits of an integer written with thousands separators (``100,226`` -> ``100226``), else None."""
    return value.replace(",", "") if _THOUSANDS_INT_RE.fullmatch(value) else None


def id_text(value: str) -> bool:
    """Harden 2: an ID cell (stripped) that can name someone: it holds a letter or a digit. A cell without one (``-``,
    ``?``, ``***``, a dash) is a placeholder: an ID column writes it as it is, at its rows (it names no one, as D35
    keeps such values), and it is no key of the namespace, so it never FAILs the file for want of a same-shape value
    (D9)."""
    return any(ch.isalnum() for ch in value)


_MASK_TABLE = str.maketrans(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", "9" * 10 + "A" * 26 + "a" * 26)


def char_mask(value: str) -> str:
    """Character-class mask: digit -> 9, upper -> A, lower -> a, others kept."""
    if value.isascii():
        return value.translate(_MASK_TABLE)
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    return "".join(out)


def analyze_id_shape(values: Iterable[str]) -> dict[str, Any]:
    """Shape of a set of real ID values (stripped, non-missing).

    Returns ``kind`` ('digits' | 'pattern'), ``widths`` (width -> count),
    ``padded`` (any leading zero), ``fixed_width``, ``sequential`` (digits whose
    integer set is exactly 1..k; uniqueness is checked by the caller) and, for
    patterns, the number of masks (count only).
    """
    vals = [v for v in values]
    widths = Counter(len(v) for v in vals)
    info: dict[str, Any] = {"widths": {str(k): c for k, c in sorted(widths.items())}, "fixed_width": len(widths) == 1}
    # K-21: ASCII digits only, as normalize_id: '10²1' is a digit to str.isdigit() but no number to int();
    # K-02: an integer stored as a float ('100234.0') is a digit ID
    digits = [v if v.isascii() and v.isdigit() else (integral_float_parts(v) or ("",))[0] for v in vals]
    if vals and all(digits):
        ints = {int(v) for v in digits}
        info.update(
            kind="digits",
            padded=any(len(v) > 1 and v.startswith("0") for v in vals),
            # a lone '1' is not a sequence: row numbers need at least 1..2 (a one-row extract keeps no real ID)
            sequential=len(ints) >= 2 and min(ints) == 1 and max(ints) == len(ints),
        )
    else:
        masks = Counter(char_mask(v) for v in vals)
        info.update(kind="pattern", padded=False, sequential=False, n_masks=len(masks))
    return info


def identifier_detection(ctx: DetectContext, source: str, reason: str) -> Detection:
    """IDENTIFIER detection for a column named by the ID list or flags."""
    vals = list(ctx.text_values)
    shape = analyze_id_shape(vals)
    n_nonblank = sum(ctx.text_values.values())
    params = {
        "shape": shape,
        "n_ids": len(vals),
        "rows_per_id_max": max(ctx.text_values.values()) if vals else 0,
        "unique": len(vals) == n_nonblank,
    }
    row_numbers = bool(shape.get("sequential")) and params["unique"]
    sub = "sequential" if row_numbers else shape["kind"]
    return Detection(ColumnType.IDENTIFIER, subtype=sub, params=params, source=source, reason=reason)


#: K-02: an ID-like value: ASCII digits (``100234``, ``000123``), an integer stored as a float (``100234.0``: the
#: pandas, Stata and Parquet storage of an integer ID column with a missing value), or letters and digits in at most
#: three parts joined by ``-``, ``_``, ``/`` or one space (``S100234``, ``STU-0012``, ``ADM/2019/0045``,
#: ``2016-0100``, ``S 100234``). Decimals, ranges (``10-12``) and fractions are not; dates, SSNs and phone numbers
#: have their own types before this test.
_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/ ][A-Za-z0-9]+){0,2}")
_DIGIT_RUN_RE = re.compile(r"[0-9]+")
#: Digit values of at most this many digits are a code list or a count, not person IDs (``1``-``60``).
SMALL_NUMBER_DIGITS = 2
#: K-02: other values may stand among the IDs of an ID column found by its values on at most this share of the rows:
#: placeholders of people without an ID (``TBD``, ``Pending``), a warehouse's unknown member (``-1``) and data-entry
#: leftovers (``100234, 100567``, ``100234 (dup)``, ``100234?``). Harden 2 (A2-1): whatever they hold (digits too),
#: they never stop the column from being an ID column, whose generator writes none of them (a same-shape code each).
ID_PLACEHOLDER_SHARE = 0.2


def id_value_shape(v: str) -> tuple[int, bool] | None:
    """K-02: (ASCII digits, has letters) of an ID-like value ``v`` (stripped; see :data:`_ID_TOKEN_RE`), else None."""
    if not v.isascii():
        return None
    if v.isdigit():
        return len(v), False
    whole = integral_float_parts(v)  # 100234.0: an integer stored as a float
    if whole:
        return len(whole[0]), False
    grouped = thousands_digits(v)  # 100,226: an integer in a number format
    if grouped:
        return len(grouped), False
    if not _ID_TOKEN_RE.fullmatch(v):
        return None
    if " " in v and any(a.isalpha() and b.isalpha() for a, b in zip(v.split(" "), v.split(" ")[1:])):
        return None  # two words in a row: text, not an ID ('Room A 204')
    runs = _DIGIT_RUN_RE.findall(v)
    if not runs:
        return None
    letters = any(ch.isalpha() for ch in v)
    if not letters and max(len(r) for r in runs) <= SMALL_NUMBER_DIGITS:
        return None  # '10-12', '3/4': a range or a fraction
    return sum(len(r) for r in runs), letters


def id_like_value(v: str) -> bool:
    """K-02: ``v`` (stripped) reads as one ID (:func:`id_value_shape`)."""
    return id_value_shape(v) is not None


def id_keys_width(values: dict[str, int], min_distinct: int) -> int:
    """K-02: the widest key (in ASCII digits) when ``values`` (stripped non-missing value -> rows) are keys of
    people, else 0.

    Keys of people: at least ``min_distinct`` distinct ID-like values (:func:`id_value_shape`), most of them wide
    (3 or more digits, or 2 digits among 5 or more letters and digits: ``jsmith23``; a list of 1-2 digit numbers or
    of short codes such as ``EL01`` is a code list or a count), on at least ``1 -`` :data:`ID_PLACEHOLDER_SHARE` of
    the rows. The other values (``TBD``, ``-1``, ``100234, 100567``) are a minority whatever they hold (harden 2,
    A2-1: one odd cell never makes a column of 60 students' IDs a column of numbers that writes them all); a column
    of measures, ranges or text has few ID-like values on its rows.
    """
    keys = wide = width = odd_rows = 0
    for v, c in values.items():
        shape = id_value_shape(v)
        if shape is None:
            odd_rows += c
            continue
        digits, letters = shape
        keys += 1
        if digits > SMALL_NUMBER_DIGITS or (letters and digits == SMALL_NUMBER_DIGITS and sum(map(str.isalnum, v)) >= 5):
            wide += 1
        width = max(width, digits)
    if keys < max(min_distinct, 1) or 2 * wide < keys:
        return 0
    if odd_rows > ID_PLACEHOLDER_SHARE * sum(values.values()):
        return 0  # measures (12.5), ranges or text: no ID column
    return width


#: K-02: PRIVATE detection param of a column whose header (``hints.weak_id_header``) and values may make it an ID
#: column; its table decides (``profile.table``): ``{'form': 'person' | 'code' | 'id', 'width': widest key}``.
WEAK_ID_KEY = "_weak_id"
#: Detected types whose generators write real values (numbers, levels, tokens): an ID column typed so leaks. Harden 2
#: (A2-1): one decimal or formatted cell among integer IDs (``12.5``, ``100,226``, ``(100107)``) makes the column a
#: FLOAT or a formatted number, whose generator writes real values (interpolated, exact extremes) too.
_WEAK_ID_TYPES = frozenset({ColumnType.INTEGER, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.FACTOR,
                            ColumnType.DIGIT_CODE, ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.CONSTANT,
                            ColumnType.FLAG_SINGLE_TOKEN, ColumnType.FLOAT, ColumnType.NUMERIC_FORMATTED,
                            ColumnType.DECIMAL})


def mark_weak_id(ctx: DetectContext, det: Detection) -> Detection:
    """K-02, fail-safe: mark (:data:`WEAK_ID_KEY`) a value-rule detection whose header names no ID by itself but holds
    a student word, a student-ID code or a final ID word (``hints.weak_id_header``: ``Student_SK``, ``Stu``,
    ``Student (District)``, ``CALPADS SSID``, ``Admission No``) when its values are keys of people
    (:func:`id_keys_width`). ``profile.table`` decides with the rest of the table whether the column is an ID column
    (:func:`weak_id_detection`); the detected type itself is unchanged.

    Harden 2 (A2-2): however few the keys are. An intervention group or a caseload of 8 students under ``Other ID`` is
    as much an ID column as a roster of 600, and ``--min-cell`` (a privacy threshold: raising it must never write
    more real values) plays no part in it."""
    from ..hints import weak_id_header

    if det.type not in _WEAK_ID_TYPES:
        return det
    form = weak_id_header(ctx.name)
    if form is None:
        return det
    shapes = set(det.params.get("pii_shapes") or ())
    if shapes and (form == "id" or shapes - {"phone"}):
        # personal-data shapes are masked as such ('Number' holding phone numbers); under a student word or a
        # student-ID code ten digits are an ID (a CALPADS SSID), not a phone number
        return det
    width = weak_id_width(ctx.typed_values)
    if width:
        det.params[WEAK_ID_KEY] = {"form": form, "width": width}
    return det


def weak_id_width(values: dict[str, int]) -> int:
    """K-02: the widest key of ``values`` (stripped non-missing value -> rows) under a weak ID header when they are keys
    of people (:func:`id_keys_width`, any number of them) and no row numbers (unique 1..n, which D9 keeps), else 0."""
    width = id_keys_width(values, 1)
    return 0 if not width or _row_numbers(values) else width


def _row_numbers(values: dict[str, int]) -> bool:
    """Unique integers exactly 1..n (``row_num``): row numbers, which D9 keeps as they are and D7 lists (rule 3)."""
    if any(c != 1 for c in values.values()) or not all(v.isascii() and v.isdigit() for v in values):
        return False
    ints = {int(v) for v in values}
    return len(ints) == len(values) and min(ints) == 1 and max(ints) == len(ints)


#: Harden 2 (A2-3, D7): integers (or letter-prefixed codes) of at least this many digits that fill several rows each
#: are keys, whatever the header: a state student ID under its bare acronym (``NYSSIS``, ``STN``, ``UIC``,
#: ``WISEid``, ``SUNS``: 10 digits; ``FLEID``: ``FL`` + 12). Scores, amounts and counts that wide rarely repeat exactly;
#: narrower keys cannot be told from them by their values.
REPEATED_KEY_DIGITS = 8
#: PRIVATE profile param of a column of repeated wide keys (:func:`repeated_key_digits`): its widest key. ``profile``
#: lists it as an ID-like candidate (D7) and counts its keys as holders of a value (D35) when the table has no ID of
#: the row's own person. D38: such a column FAILs its file when its table has no recognised ID column, unless a flag
#: names it (``profile.table.unrecognised_key_columns``).
REPEATED_KEYS_KEY = "_repeated_keys"
#: D38: PRIVATE profile param of a column of repeated wide keys (:func:`repeated_keys_width`) that an explicit flag typed
#: so that it writes real values (``--org-code``, ``--force-type COL=TYPE``, ``--force-factor``): its widest key. The
#: file is written (the flag is the user's decision), with a message, and the report withholds its value statistics.
FLAGGED_KEYS_KEY = "_flagged_repeated_keys"


def repeated_key_digits(ctx: DetectContext, det: Detection) -> int:
    """Harden 2 (A2-3): the widest key of a non-ID column whose values are keys of people (:func:`id_keys_width`) at
    least :data:`REPEATED_KEY_DIGITS` digits wide (most of them), with two or more keys and some on several rows, under
    a header that names no measure, count or flag (``hints.measure_header``), else 0. Ten-digit keys may look like
    phone numbers (masked as such: a state ID all the same, and a phone number on each of a person's rows counts
    their household); personal data of any other shape or under a personal-data header and other types never are."""
    if det.type not in _WEAK_ID_TYPES or set(det.params.get("pii_shapes") or ()) - {"phone"} or det.params.get(
            "pii_hint"):
        return 0
    return repeated_keys_width(ctx)


def repeated_keys_width(ctx: DetectContext) -> int:
    """The header and value part of :func:`repeated_key_digits`, whatever the column's type (D38: also for a column a
    flag typed, ``--org-code`` included): the widest key, or 0."""
    from ..hints import measure_header

    if ctx.hints.pii is not None or ctx.hints.free_text is not None or measure_header(ctx.name):
        return 0
    vals = ctx.typed_values
    if len(vals) < 2 or all(c == 1 for c in vals.values()):
        return 0
    width = id_keys_width(vals, 2)
    if width < REPEATED_KEY_DIGITS:
        return 0
    shapes = [id_value_shape(v) for v in vals]
    keys = [s for s in shapes if s is not None]
    return width if 2 * sum(s[0] >= REPEATED_KEY_DIGITS for s in keys) >= len(keys) else 0


def weak_id_detection(ctx: DetectContext, form: str) -> Detection:
    """K-02: the IDENTIFIER detection of a marked column (:func:`mark_weak_id`) that its table makes an ID column:
    source ``person-id-hint`` (with a note) for a student word or a student-ID code (``Student_SK``, ``Stu``,
    ``CALPADS SSID``), ``id-shape`` with a warning for a final ID word alone (``Admission No``: it may number things,
    not people)."""
    from ..model import Message

    if form != "id":
        det = identifier_detection(ctx, "person-id-hint", "header names the student (or a student-ID code) over "
                                                          "ID-like values (D10, K-02): synthetic IDs")
        det.params[WEAK_ID_KEY] = form  # PRIVATE: found by its values; it links to another ID column like a key-ref
        det.messages.append(Message(
            "info", "id_by_header_values",
            "treated as a student ID column: the header names the student (or a student-ID code) and the values are "
            "ID-like keys, so it gets synthetic IDs of the same shape (never its real values)",
            column=ctx.name, fix=f"--force-type {ctx.name}=INTEGER (or --force-factor {ctx.name}) only if it holds "
                                 "no one's ID: its real values are then written"))
        return det
    det = identifier_detection(ctx, "id-shape", "header ends in an ID word over ID-like keys (K-02): synthetic IDs")
    det.params[WEAK_ID_KEY] = form
    det.messages.append(Message(
        "warning", "id_by_header_values",
        f"treated as an ID column: the header ends in an ID word and the column holds {len(ctx.typed_values)} "
        "distinct ID-like keys, so it gets synthetic IDs of the same shape, consistent across files (never its real "
        "values)",
        # K-04: --person-id only for the row's own person (an incident or record number would count each record as a
        # person); --id-col for records or other people
        column=ctx.name, fix=f"--person-id {ctx.name} if it numbers the row's own person, --id-col {ctx.name} if it "
                             f"numbers records or other people (either: no warning), or --org-code {ctx.name} to "
                             "keep its real values if it numbers things such as rooms, buses or devices"))
    return det


def id_candidate_reason(ctx: DetectContext, det: Detection) -> str | None:
    """AREA D hook: why this (non-ID) column looks like an ID, or None.

    Called by ``profile`` for every non-IDENTIFIER column of every table (the D7
    warning lists the candidates; a candidate's real values are left out of the
    report and docs). Must not change the column's type. Rules (module
    docstring): UUIDs, 32/40/64-hex hashes, unique sequential integers, unique
    values with an ID-like header, repeated wide keys (:func:`repeated_key_digits`),
    unique fixed-width codes, and >= 95% unique values with one character-shape
    mask covering >= 90% of the rows. The returned reason is report-safe (shapes
    and shares, never a value).
    """
    if det.type is ColumnType.IDENTIFIER or det.type is ColumnType.ORG_CODE:
        return None
    if TYPE_INFO[det.type].family in _NEVER_CANDIDATE_FAMILIES:
        return None
    # missing-like placeholders ('N/A', 'NULL' ...) are not ID values: a UUID column with a few 'N/A' is still one
    vals = ctx.typed_values
    n = sum(vals.values())
    if n < 2:
        return None
    distinct = len(vals)
    unique = distinct == n
    if all(is_uuid(v) for v in vals):
        return "UUID values"
    lengths = {len(v) for v in vals}
    if len(lengths) == 1:
        width = next(iter(lengths))
        if width in HASH_LENGTHS and all(_HEX_RE.fullmatch(v) for v in vals):
            return f"{width}-character hex hash values ({HASH_LENGTHS[width]} shape)"
    if unique and all(v.isascii() and v.isdigit() for v in vals):
        ints = sorted(int(v) for v in vals)
        if ints[0] in (0, 1) and ints[-1] - ints[0] + 1 == len(ints):
            return "unique sequential integers"
    if unique and ctx.hints.id_hint:
        return f"unique values and ID-like header ('{ctx.hints.id_hint}')"
    wide = repeated_key_digits(ctx, det)
    if wide:  # harden 2 (A2-3): a person's ID on each of their rows, under a header no rule knows
        return f"repeated ID-like keys of up to {wide} digits (a person's ID on each of their rows)"
    personal = bool(det.params.get("pii_hint")) or ctx.hints.pii is not None or ctx.hints.free_text is not None
    if det.type not in _SHAPE_TYPES or personal:
        return None
    if unique and len(lengths) == 1:
        return f"unique fixed-width codes ({next(iter(lengths))} characters)"
    share_unique = distinct / n
    if share_unique >= CANDIDATE_UNIQUE_SHARE:
        masks: Counter = Counter()
        for v, c in vals.items():
            masks[char_mask(v)] += c
        top = masks.most_common(1)[0][1] / n
        if top >= CANDIDATE_MASK_SHARE:
            return f"{share_unique:.0%} unique values; one character shape covers {top:.0%} of rows"
    return None
