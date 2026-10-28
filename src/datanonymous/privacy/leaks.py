"""Copied-value leak scanning (AREA G; design 5.2.3, 5.2.6, D9, D19-D21).

Two layers make sure no real ID, name, email, phone, SSN, address, IP/URL,
free-text or high-cardinality text value reaches anything the run writes.

1. :func:`scan_table` (before writing, per column): no synthetic value of a
   never-copy column equals a real value of that column. Comparison is
   type-aware: names, labels, addresses and free text case-insensitively with
   collapsed whitespace, emails also by local part, phones/SSNs by their digits,
   IDs exactly and by their D8 key (case, punctuation, leading zeros) and
   (R5-02) by the e-mail and username rules: an ID that is an e-mail address
   never has a real local part, and no ID has the username of a real ID
   (``detect.ids.username_stem``, read without the column's kept ID prefix,
   ``params['_id_prefix']``). ID
   namespaces that keep the real row numbers 1..n (D9,
   ``params['_identity_mapped']``) are skipped. Run-wide, no full-name value and
   no row of first/middle/last cells spells a real person's full name held
   anywhere in the run (``generate.pii.real_full_name_hits``: word pairs and
   single-word complete names), and (R7-01, R7-02) no output text equals a
   real text of ANY column of the run that the run never writes
   (``privacy.texts``, :func:`plan_real_texts`). A hit FAILs the file before
   it is written.

2. :func:`scan_outputs` (after writing, directory-wide): every file in the out
   dir and every file of the report payload (built, not yet written).

Sensitive columns (:func:`sensitive_kind`): IDENTIFIER, PERSON_NAME, EMAIL,
PHONE, SSN, STREET_ADDRESS, IP/URL, FREE_TEXT, HIGH_CARD_CATEGORICAL (a PII
header hint selects the name/email/phone/SSN/address rules), DIGIT_CODE
masks, and every value-copying column with units the copy gate's D35 allowlist
did not keep (``detect.text.pseudonymized``: levels, multi-select items compared
item by item, tokens among numbers). Exactly those pseudonymised real units are
indexed (as labels), and so are the person names found inside organisational
codes (D36, ``detect.text.org_name_parts``: ``Garcia, Ana``, ``Ms. Garcia``),
whose codes are written with fake names; the column's kept units are excused,
from the same source of truth (:func:`detect.text.copy_cleared`), and never
anything else.
Their real values are the in-memory private models of the run; they are never
written or quoted. E-mail addresses, phone numbers (>= 10 digits) and SSNs
inside text values are indexed as such too (``312-555-0199`` inside a note
matches ``3125550199`` anywhere).

Flagged names (D34, ``privacy.headers``): a header, sheet, file or folder name
that holds personal data is indexed too (kind ``header``: its word sequence,
the words of the person's name in it, its whole text, the ID parts of a real
ID and the e-mails, phones and SSNs inside it). It is never public (the
allowlist leaves it out although headers are public otherwise), a written
output whose re-read header or sheet name holds one FAILs (rule
``header/name``), and the out-dir paths the scan prints are shown redacted.
The report, docs and console are already redacted by the D34 redactor, so a
hit here is a backstop for a path that escaped it.

Matching rules:

* IDs and codes: exact whole tokens (report/docs: tokens of >= 4 characters;
  digits right after ``<digit>.`` are a decimal fraction, not an ID);
  in the data: exact and D8-normalised values against the real IDs of every ID
  namespace of the run (its own and every other one, except kept row numbers
  1..n), so a synthetic ID never equals a real one (D9); and (R5-02) the e-mail
  local parts and usernames of those real IDs, and an ID that is an e-mail
  address against every real e-mail address of the run. Real IDs that are
  e-mail addresses are indexed as e-mail addresses too (report and docs);
* names: case-insensitive whole tokens (and the full name as a token sequence);
* emails: the full address (case-insensitive) and its local part as a whole token;
* phones and SSNs: digit-normalised (``(312) 555-0199`` = ``312.555.0199`` =
  ``3125550199``; a leading country code ``1`` is ignored);
* street addresses, multi-word labels and full names: case-insensitive token
  sequences; free text: the whole value and every sentence of >= 4 words and
  every 6-word n-gram. A label that is a number written as text (only digit
  tokens: ``0,89``, ``2.396,27``, ``56,1%`` in a decimal-comma file read without
  ``--decimal ,``) matches only with its own separators, so a statistic of the
  report in another notation (a share ``0.89``) is not a hit; the real text is.

Where they apply:

* report payload and out-dir docs (aggregates only, D20/D21): every rule. A
  single token is excused when it is *public*: a word or number of the tool's
  own text (string and number literals of this package), a path, file or
  column name, a label of a label-safe column (reproduced anyway, D21), an NA
  token, a run setting, or an aggregate of a non-sensitive column (statistics
  of a sensitive column only through their value-free keys and their counts).
  Pure-digit tokens of fewer than 4 digits and decimal fractions are never
  matched as IDs (they are counts and statistics). A token sequence is excused
  only when it is part of a header or a label-safe level.
* synthetic data files of this run (re-read with the output format's reader):
  same-column values, ID namespaces, and run-wide the distinctive patterns
  (full emails, phone-shaped tokens of >= 10 digits, SSN-formatted numbers,
  free-text and street-address token sequences) in every text column. A phone
  match is a phone-SHAPED token (:func:`privacy.digits.phone_tokens`: a whole
  digit run of 10-15 digits, ``+`` international notation, or digit groups laid
  out like a NANP number or a real phone of the run), never digits glued
  across separators into another layout or a window inside a longer run. The
  generators never write a real phone/SSN-shaped token into a value
  (:func:`plan_digit_keys`, ``privacy.digits.guard``): digits the tool draws
  (synthetic IDs glued into a derived e-mail, random digits of a mimicked local
  part or note) are redrawn rather than left to equal a real phone by chance,
  so a hit here is a copy and FAILs the file
  (:func:`cross_column_values`: not in ID/code columns, whose values come from
  their own shape and are checked against their own namespace, not in numeric
  or date columns, and not for the real levels of level-generated columns,
  which are reproduced by design); Excel XML text outside the cells (number
  formats included), Arrow metadata, and label text no cell shows (dictionary
  entries and Stata value labels no row holds, Stata variable/data labels:
  :func:`hidden_label_text`) too. Run-wide (R7-01, R7-02), every value of
  those text columns (a multi-select item by item) is also compared as a
  whole with the run's real texts that the run never writes
  (``privacy.texts``: a pseudonymised label, a name, address, free text or
  high-cardinality label of ANY column, a value of an excluded or nulled
  column, text a date column replaced), excusing only the real texts the run
  writes verbatim somewhere (the D35 kept levels, items and tokens, NA tokens,
  states: :func:`real_texts_of`); an output value that is an e-mail address
  never has the local part of a real address or e-mail ID of the run (a
  username), and (R8-03) no value of an ID column is a real e-mail address
  of the run, or has the local part or username of one, wherever it is held
  (``privacy.texts.RealTexts.user_hits``; the ID registry avoids them all),
  nor (R9-03) equals one of the registry's texts as a whole (:func:`id_texts`;
  ``generate.ids.RealUsers.text_taken`` never draws one).
  R9-01: no output column holds a real value of its own that is a real person
  ID of the run (``privacy.realids.kept_hits``: a level, a number by value,
  an organisational code, a token, an item, an NA token; every ID column of
  the run, FAILed, excluded and nulled ones too; row numbers 1..n, keys of
  fewer than 4 characters or without a digit and plain numbers of a range
  where most numbers of their shape are real IDs, the D34 screen's chance,
  are no person IDs), whatever
  the D35 allowlist or the key discovery decided; such a value is never
  excused (the generators pseudonymise every one, ``profile.table
  .finish_real_ids``), in the data (both layers) and in the report allowlist
  (a column holding one is never public).
  The source text outside the tables (R8-04: Excel and HTML titles, notes,
  footers and side notes written as placeholders, HTML comments and dropped
  attributes, free-form Arrow metadata, Stata variable and data labels:
  :func:`outside_texts`) is registered the same way, and the e-mail
  addresses, phone numbers and SSNs inside it are indexed for the data rules
  (every output text and the file metadata view). The generators never draw one
  (``privacy.texts.guard``), so a hit is a copy and FAILs the file. Single
  name WORDS are not matched across columns: an invented full name may share a
  first or last name with a real person of another column (never the whole
  value, nor a real person's two names: ``generate.pii``).
* any other out-dir file: an output left from an earlier run whose source
  FAILed in this run is scanned like a data file (when it can be re-read);
  anything else (not written by the tool) gets the distinctive patterns.

Text is matched in a folded form: typographic apostrophes and hyphens become
``'`` and ``-`` (``O’Brien`` = ``O'Brien``) and case is folded; the parts of a
hyphenated or apostrophe name (``Okonkwo`` of ``Okonkwo-Vance``) are names too.
Short free-text values (2-3 words, e.g. a medical alert) are matched as token
sequences in the report and docs; token sequences inside one string literal of
the tool's own source are excused like its single words.

On a hit the message never quotes the value. A synthetic file with a hit FAILs
and is deleted (D19: a FAILed file is not written). A hit in a per-file report
FAILs that file; a hit in the run report or the out-dir docs FAILs the file(s)
whose column the value came from. The out-dir docs are rewritten without the
FAILed files and every occurrence is redacted. Any hit adds a run-level error
(exit 3), a :class:`LeakMessage` whose private redactor the report builder uses
to redact the rebuilt report and the console output. A file elsewhere in the
out dir with a hit is a run-level error naming the file.

Extreme exposure (D14 exact min/max, design 5.2.5) is reported by the report
builder as information.
"""

from __future__ import annotations

import ast
import bz2
import functools
import gzip
import hashlib
import html
import io
import json
import lzma
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import AbstractSet, Any, Callable, Iterable

import numpy as np

from ..detect.dates import REPLACED_KEY as REPLACED_DATE_KEY
from ..detect.ids import PREFIX_MIN_IDS, checked_prefix, id_text, is_email_id, normalize_id, username_stem
from ..detect.text import (combination_stay, copies_real_values, copy_cleared, na_pseudonymized, org_name_parts,
                           org_name_phrases, org_named_texts, pseudonymized, unit_key)
from ..generate import pii
from ..generate.table import is_level_generated
from ..model import (
    LABEL_SAFE_TYPES,
    NEVER_COPY_TYPES,
    TYPE_INFO,
    ColumnProfile,
    ColumnType,
    FileResult,
    Message,
    SyntheticTable,
    TableProfile,
)
from ..profile.stats import VALUE_REVEALING_KEYS
from ..profile.codes import codes_of
from . import realids
from . import texts as real_texts
from .digits import (
    DECIMAL_NUMBER,
    MIN_PHONE,
    NANP_LAYOUTS,
    SSN_RE,
    RealDigitKeys,
    phone_keys,
    phone_layouts,
    phone_numbers,
    phone_tokens,
    semantic_phone_keys,
    semantic_phones,
    phone_base_keys,
    phone_base,
    phone_text,
    union,
)

LEAK_PREFIX = "leak scan: "

# kinds of sensitive columns
ID, CODE, NAME, EMAIL, PHONE, SSN, ADDRESS, NET, FREE_TEXT, LABEL = (
    "id", "code", "name", "email", "phone", "ssn", "address", "net", "free_text", "label")
#: D34: a header, sheet, file or folder name that holds personal data (``privacy.headers``)
HEADER = "header"
#: R7-01, R7-02: a real text value of the run that it never writes (``privacy.texts``; any column, excluded and
#: nulled ones included)
REAL_TEXT = "real_text"
#: R8-04: source text outside the tables that the run never writes (titles, notes, footers, dropped metadata)
LAYOUT = "layout"
KIND_TEXT = {
    ID: "ID", CODE: "code", NAME: "person-name", EMAIL: "email", PHONE: "phone-number", SSN: "SSN",
    ADDRESS: "street-address", NET: "IP/URL", FREE_TEXT: "free-text", LABEL: "high-cardinality text",
    HEADER: "header/name", REAL_TEXT: "unwritten real text", LAYOUT: "title/notes/metadata text",
}
_NAME_HINTS = frozenset({"first_name", "last_name", "middle_name", "full_name", "name"})
_HINT_KIND = {"email": EMAIL, "phone": PHONE, "ssn": SSN, "address": ADDRESS, "street_address": ADDRESS}

#: Word tokens: letters/digits, joined by apostrophes or hyphens (``O'Brien``, ``Smith-Jones``).
WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
#: Same-length folding of typographic apostrophes and hyphens before matching, so ``O’Brien`` (curly, common
#: in cp1252 exports) and ``O'Brien`` are one name. One character to one character: match spans stay valid.
_PUNCT_FOLD = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u2032": "'", "\u2010": "-",
                             "\u2011": "-"})
#: Joiners inside a word token; parts of a joined token are matched too (``Okonkwo`` in ``Okonkwo-Vance``).
_JOINERS = re.compile(r"['\-]")
MIN_PART = 3


def fold(s: str) -> str:
    """Matching form of text: typographic apostrophes/hyphens folded (same length), then case-folded."""
    return (s if s.isascii() else s.translate(_PUNCT_FOLD)).casefold()


def word_parts(tok: str) -> list[str]:
    """Parts of a joined word token (``o'shaughnessy`` -> ``shaughnessy``) of at least MIN_PART characters."""
    if "'" not in tok and "-" not in tok:
        return []
    return [p for p in _JOINERS.split(tok) if len(p) >= MIN_PART and not p.isdigit()]
#: ID/URL-like tokens: split on whitespace, quotes, brackets and list separators.
TOKEN_RE = re.compile(r"[^\s,;|`'\"()\[\]{}<>]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+'\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
LOCAL_RE = re.compile(r"[A-Za-z0-9._%+'\-]+")
#: Phone-like digit runs: digits separated by at most two of ``space ( ) . -`` (report/docs rule).
DIGITS_RE = re.compile(r"\+?\(?\d(?:[ ().\-]{0,2}\d){6,14}")
#: The same with at least 10 digits: indexes phone numbers inside real text values (the data scan compares
#: phone-SHAPED tokens only, :func:`privacy.digits.phone_tokens`).
PHONE10_RE = re.compile(r"\+?\(?\d(?:[ ().\-]{0,2}\d){9,14}")
#: A plain decimal number (one point between digits): never read as a phone/SSN digit run.
_DECIMAL_NUMBER = DECIMAL_NUMBER
SENTENCE_RE = re.compile(r"[.!?;\n\r]+")
NGRAM = 6
MIN_SENTENCE_WORDS = 4
#: Report/docs scan only: sentences of 3 words inside a free-text value are phrases too.
MIN_SHORT_SENTENCE = 3
#: Fewest characters of an ID token (report/docs rule): one rule with ``privacy.realids.MIN_KEY`` (R9-01).
MIN_ID_TOKEN = realids.MIN_KEY
REDACTED = "[redacted]"
#: Statistics keys of a sensitive column that never hold a value (public aggregates).
_VALUE_FREE_STATS = frozenset({"n", "missing", "missing_rate", "missing_tokens", "distinct", "distinct_ratio",
                               "unique", "whitespace_variants", "length", "frequency_profile", "top_masks",
                               "mean_words", "redacted", "pmf_omitted"})
#: Message codes whose text the tool builds from counts, rates and column names only (their format strings
#: interpolate no data value), so the plain numbers in them are counts. A count can equal a real numeric ID
#: when IDs are dense (``8980 synthetic ID(s) have a different number of digits`` with IDs 1000-9989); such a
#: number is excused. An area adding a counts-only message code adds it here.
COUNT_ONLY_CODES = frozenset({
    "line_endings_mixed", "blank_lines_dropped", "quoting_inconsistent", "factor_overflow",
    "possible_sparse_code", "id_width_changed", "row_matches_report_only", "row_matches_remaining",
    "fidelity_drift", "dtype_mismatch",
    # added at integration (the areas were built in parallel): counts plus column/sheet names and formats only
    "date_mixed_formats", "date_sentinels", "date_tokens", "date_unparsed_replaced",  # A
    "composite_key_duplicates", "fd_unmapped", "template_partly_applied", "id_check_not_applied",  # D
    "panel_detected", "multi_year_snapshots", "aggregate_table",  # D
    "excel_formulas", "excel_side_notes", "excel_merged_dropped", "excel_range_dropped", "excel_extras_dropped",  # E
    "stata_extended_missing",  # F
    "name_pairs_redrawn",  # names-pii (R1-04)
    "real_full_name",  # names-headers (R2-06)
    "real_text_copied",  # the run's real-text registry (R7-01, R7-02): counts only
    "real_id_values",  # values equal to a real person ID, pseudonymised (R9-01): counts only
    "pii_values",  # copy gate (R3): kinds and counts only
    "pseudonymized_levels", "rare_levels_kept",  # D35 allowlist: counts and shares only
    "org_code_names",  # D36 person names inside organisational codes: counts and shares only
    "date_far_sentinels",  # K-07 far-from-median sentinels read as dates: counts only
    "pseudonymized_na_tokens",  # D35 over --na-value tokens (R8-05): counts only
    "forced_copy",  # --force-factor over personal data (R6-03): kinds and counts only
    "leak_scan_incomplete",  # the files the scan could not verify (R6-04): counts only
    "entity_rows_inexact",  # within-ID constants' rows the ID blocks could not match exactly (R9-04): counts only
})
#: Detection params that may quote a value (never public tokens for a sensitive column).
_VALUE_PARAMS = frozenset({"min", "max", "value", "token", "tokens", "levels", "pair", "mode"})
_MAX_TEXT_VIEW = 256 * 1024 * 1024


# --------------------------------------------------------------------------- column classification
def sensitive_kind(p: ColumnProfile) -> str | None:
    """Leak-scan kind of a column, or None when its real values may appear (labels, numbers, dates)."""
    if p.role in ("excluded", "null"):
        return None
    t = p.type
    hint = p.params.get("pii_hint") or p.params.get("pii_subtype")
    if t is ColumnType.IDENTIFIER:
        return ID
    if t is ColumnType.PERSON_NAME:
        return NAME
    if t is ColumnType.EMAIL:
        return EMAIL
    if t is ColumnType.PHONE:
        return PHONE
    if t is ColumnType.SSN:
        return SSN
    if t is ColumnType.STREET_ADDRESS:
        return ADDRESS
    if t in (ColumnType.IP, ColumnType.URL):
        return NET
    if t in (ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.FREE_TEXT):
        # a PII header hint picks the PII rules (long emails can be classified as free text)
        if hint in _NAME_HINTS:
            return NAME
        if hint in _HINT_KIND:
            return _HINT_KIND[hint]
        return FREE_TEXT if t is ColumnType.FREE_TEXT else LABEL
    if t is ColumnType.DIGIT_CODE and not p.params.get("levels_mode"):
        return CODE
    if pseudonymized(p):
        # D35: the levels, multi-select items or tokens the allowlist did not keep are written as pseudonyms; those
        # real units are indexed as labels and never excused (the column's kept units are)
        return LABEL
    return None


def _item_strings(p: ColumnProfile, values: Iterable[str]) -> list[str]:
    """The strings of ``values`` the same-column check compares for ``p``.

    A multi-select with pseudonymised items (D35) has each item compared on its own as well as each whole value;
    NA tokens and item-less values (written as they are) are left out.
    """
    vals = list(values)
    d = p.params.get("delimiter") if p.type is ColumnType.MULTI_SELECT and pseudonymized(p) else None
    if not d:
        return vals
    na = p.na_tokens or frozenset()
    out: dict[str, None] = {}
    for v in vals:
        s = v.strip()
        if not s or s in na:
            continue
        items = [it.strip() for it in s.split(d) if it.strip()]
        if not items:
            continue
        out[s] = None
        for it in items:
            out[it] = None
    return list(out)


def sensitive_columns(files: Iterable[FileResult]) -> list[tuple[str, str | None, str, str]]:
    """(file, table, column, kind) of every sensitive column whose real values the scan checks."""
    out = []
    for fr in files:
        for tp in fr.tables:
            for p in tp.columns:
                kind = sensitive_kind(p)
                if kind is None or (kind == ID and p.params.get("_identity_mapped")):
                    continue
                out.append((fr.rel_path, tp.name, p.name, kind))
    return out


_NON_DIGIT = re.compile(r"[^0-9]+")


def norm_text(v: str) -> str:
    return " ".join(fold(v).split())


def digits_of(v: str) -> str:
    return _NON_DIGIT.sub("", v)


_phone_keys = phone_keys


def _norm_digits(v: str) -> str:
    """Phone/SSN comparison key: the digits (a leading country code 1 of an 11-digit phone dropped)."""
    d = _NON_DIGIT.sub("", v)
    if len(d) < 7:
        return norm_text(v)
    return d[1:] if len(d) == 11 and d[0] == "1" else d


def _normalizer(kind: str) -> Callable[[str], str]:
    """Same-column comparison form of a value: exact (IDs, codes), digits (phones, SSNs) or folded text."""
    if kind in (ID, CODE):
        return str.strip
    if kind == PHONE:
        return lambda v: base[0] if (base := phone_base(v)) is not None else _norm_digits(v)
    if kind == SSN:
        return _norm_digits
    return norm_text


def _local_part(v: str) -> str:
    return fold(v.strip()).partition("@")[0]


def _id_local(v: str) -> str | None:
    """R5-02: the folded local part of an ID value that is an e-mail address, else None."""
    s = v.strip()
    return fold(s).rpartition("@")[0] if "@" in s and is_email_id(s) else None


def id_users(values: Iterable[str], prefix: str = "") -> tuple[set[str], set[str]]:
    """R5-02: (e-mail local parts, usernames) of ID values: the folded local part of every value that is an
    e-mail address, and every :func:`detect.ids.username_stem` (``prefix``: the column's kept ID prefix)."""
    locs: set[str] = set()
    stems: set[str] = set()
    for v in values:
        loc = _id_local(v)
        if loc is not None:
            locs.add(loc)
        st = username_stem(v, prefix)
        if st:
            stems.add(st)
    return locs, stems


@dataclass(frozen=True)
class _RealSets:
    """Normalised real values of one column for the same-column check (PRIVATE)."""

    norm: frozenset[str]
    locals: frozenset[str] = frozenset()  # EMAIL: local parts; ID: local parts of e-mail IDs (R5-02)
    keys: frozenset[str] = frozenset()  # ID: D8 keys
    stems: frozenset[str] = frozenset()  # ID: usernames (R5-02)
    prefix: str = ""  # ID: the kept ID prefix the usernames are read without

    @classmethod
    def of(cls, kind: str, real: Iterable[str], prefix: str = "") -> "_RealSets":
        real = list(real)
        if kind == ID:
            locs, stems = id_users(real, prefix)
            return cls(frozenset(map(_normalizer(kind), real)), frozenset(locs),
                       frozenset(map(normalize_id, real)), frozenset(stems), prefix)
        return cls(frozenset(map(_normalizer(kind), real)),
                   frozenset(_local_part(v) for v in real if "@" in v) if kind == EMAIL else frozenset(),
                   frozenset())


def _same_column_hits(kind: str, real: "_RealSets", syn: Iterable[str]) -> int:
    """Number of distinct synthetic values equal to a real value of the column (normalised per kind)."""
    syn_distinct = set(syn)
    hits = set(map(_normalizer(kind), syn_distinct)) & real.norm
    if kind == ID:
        hits |= set(map(normalize_id, syn_distinct)) & real.keys
        if real.locals or real.stems:  # R5-02: e-mail and username rules for e-mail addresses and usernames
            hits |= {"@" + v for v in syn_distinct if _id_local(v) in real.locals}
            hits |= {"#" + v for v in syn_distinct if username_stem(v, real.prefix) in real.stems}
    if kind == EMAIL:
        hits |= {_local_part(v) for v in syn_distinct if "@" in v} & real.locals
    return len(hits)


def _real_values(tp: TableProfile, p: ColumnProfile, kind: str) -> list[str]:
    if kind == ID:
        raw = tp.private.id_raw.get(p.index)
        if raw is not None:
            # a placeholder without a letter or digit ('-', '?') is no ID: the ID column writes it as it is
            return list({v.strip() for v in codes_of(raw)[1].tolist()
                         if isinstance(v, str) and id_text(v)})
    pseudo = pseudonymized(p)
    if pseudo:
        # D35: the real units the allowlist did not keep (levels, items or tokens); the kept ones are written
        # verbatim by design, and the numbers of a token column are interpolated (exact extremes, D14). D36: and the
        # person names found inside organisational codes ('Garcia, Ana', 'Ms. Garcia'), never excused
        return list(pseudo) + [t for t in org_name_phrases(p) if t not in pseudo]
    m = p.model
    if m is None:
        return []
    vals = m.real_values if m.real_values is not None else {
        str(k).strip() for k in m.nonmissing_counts if k is not None}
    return _item_strings(p, (v for v in vals if v))


_SETS_KEY = "_leak_real_sets"


def _cached_sets(tp: TableProfile, p: ColumnProfile, kind: str, real: list[str]) -> "_RealSets":
    """Normalised real sets of a column, cached in its PRIVATE model (computed once per run)."""
    holder = p.model.extra if p.model is not None else {}
    # R5-02: the prefix the generator kept, re-checked (never a name-like prefix: a kept username is not excused)
    prefix = checked_prefix(str(p.params.get("_id_prefix") or "")) if kind == ID else ""
    cached = holder.get(_SETS_KEY)
    if cached is not None and cached[0] == (kind, prefix):
        return cached[1]
    sets = _RealSets.of(kind, real, prefix)
    if p.model is not None:
        holder[_SETS_KEY] = ((kind, prefix), sets)
    return sets


_DIGITS_KEY = "_leak_digit_terms"


def _cached_digit_terms(p: ColumnProfile, kind: str, real: list[str]) -> "DigitTerms":
    """The column's :func:`digit_terms`, cached in its PRIVATE model: planned before generation
    (:func:`plan_digit_keys`) and reused by the scan's index (computed once per run)."""
    holder = p.model.extra if p.model is not None else {}
    cached = holder.get(_DIGITS_KEY)
    if cached is not None and cached[0] == kind:
        return cached[1]
    dt = digit_terms(kind, real, phone_bases=_source_phone_bases(p) if _known_phone(p) else None)
    if p.model is not None:
        holder[_DIGITS_KEY] = (kind, dt)
    return dt


def plan_digit_keys(files: Iterable[FileResult]) -> RealDigitKeys:
    """Run-level planning (pipeline, after pass 1): the real phone and SSN keys of the run, which no generator
    may write (:mod:`privacy.digits`).

    The keys are exactly those the scan's data rules compare (:class:`SensitiveIndex`): phone numbers and SSNs
    of every sensitive column of every file (a FAILed file's people are real people too), including those
    inside text values, and those inside the values of an excluded or nulled column (R7-02: real values all the
    same) and inside the source text outside the tables that the run does not write verbatim (R8-04: titles,
    notes, dropped metadata; read from the run's real-text registry, so :func:`plan_real_texts` comes first). Every
    table gets them (``tp.private.extra['run_digit_keys']``), and generation avoids them while it builds the table
    and while its file is written (title placeholders: ``privacy.digits.guard``).
    """
    files = list(files)
    phones: list[AbstractSet[str]] = []
    bases: list[AbstractSet[str]] = []
    ssns: list[AbstractSet[str]] = []
    layouts: set[tuple[int, ...]] = set(NANP_LAYOUTS)
    for fr in files:
        for tp in fr.tables:
            for p in tp.columns:
                bases.append(_source_phone_bases(p))
                kind = sensitive_kind(p)
                if kind is None or kind in (ID, CODE, NET):
                    continue
                real = _real_values(tp, p, kind)
                if not real:
                    continue
                dt = _cached_digit_terms(p, kind, real)
                phones.append(dt.phones)
                bases.append(dt.phone_bases)
                ssns.append(dt.ssns)
                layouts |= dt.layouts
            for dt in _dropped_digit_terms(tp).values():
                phones.append(dt.phones)
                bases.append(dt.phone_bases)
                ssns.append(dt.ssns)
    # R8-04: and those inside the source text outside the tables that the run does not write verbatim (the run's
    # real-text registry, planned before: ``plan_real_texts``)
    texts = next((t for fr in files for tp in fr.tables if (t := run_texts(tp)) is not None), None)
    for _, raw in (texts.outside if texts is not None else ()):
        dt = outside_digit_terms(raw)
        phones.append(dt.phones)
        bases.append(dt.phone_bases)
        ssns.append(dt.ssns)
    keys = RealDigitKeys(union(phones), union(ssns), frozenset(layouts), union(bases))
    for fr in files:
        for tp in fr.tables:
            tp.private.extra["run_digit_keys"] = keys
    return keys


_DROPPED_DIGITS_KEY = "_leak_dropped_digit_terms"


def _dropped_digit_terms(tp: TableProfile) -> dict[int, "DigitTerms"]:
    """R7-02: the :class:`DigitTerms` (phone numbers and SSNs inside the values) of every excluded or nulled column
    of ``tp``, cached in its PRIVATE extra (computed once per run)."""
    dropped = tp.private.extra.get(real_texts.DROPPED_KEY) or {}
    if not dropped:
        return {}
    cached = tp.private.extra.get(_DROPPED_DIGITS_KEY)
    if cached is None:
        from ..hints import phone_header_hint

        declared = tp.private.extra.get("_dropped_phone_columns", ())
        cached = {i: digit_terms(PHONE if (i in declared or _known_phone(tp.columns[i])
                                         or phone_header_hint(tp.columns[i].name) is not None) else LABEL, sorted(vals))
                  for i, vals in dropped.items()}
        tp.private.extra[_DROPPED_DIGITS_KEY] = cached
    return cached


# --------------------------------------------------------------------------- run-wide real texts (R7-01, R7-02)
def _source_kind(p: ColumnProfile) -> str:
    if p.role in ("excluded", "null"):
        return real_texts.DROPPED
    return real_texts.NAME if p.type is ColumnType.PERSON_NAME else real_texts.TEXT


def _keys_of(values: Iterable[Any]) -> frozenset[str]:
    """Registry keys (:func:`privacy.texts.key`) of the text values (with a letter) of ``values``."""
    out = set()
    for v in values:
        if isinstance(v, str):
            k = real_texts.key(v)
            if k and real_texts.is_text(k):
                out.add(k)
    return frozenset(out)


def _native_labels(p: ColumnProfile) -> list[str]:
    """The label texts of a column's native metadata (Arrow dictionary entries, Stata value labels): only those
    the column writes are kept (``detect.text.copy_gate``); the others are real texts it never writes."""
    nat = p.native or {}
    texts: list[Any] = []
    if nat.get("kind") == "dictionary":
        texts = list(nat.get("dictionary_values") or [])
    elif nat.get("family") == "stata" and nat.get("labels"):
        texts = list(nat["labels"].values())
    return [str(x) for x in texts if x is not None]


def _unkept_keys(tp: TableProfile, p: ColumnProfile) -> AbstractSet[str]:
    """The registry keys of the real text values of column ``p`` that it does not write verbatim (R7-01, R7-02):
    every value of an excluded or nulled column; the units the D35 allowlist pseudonymised; every value of a
    person-name, e-mail, address, IP/URL, free-text, high-cardinality or city column (IDs and digit codes have
    their own namespaces and rules, phone numbers and SSNs :mod:`privacy.digits`: only their text with a letter);
    a state or ZIP column's values that are no state or ZIP code (masked); the text a date column replaces with
    generated dates; native labels (kept ones are excused by the kept set)."""
    if p.role in ("excluded", "null"):
        return _keys_of((tp.private.extra.get(real_texts.DROPPED_KEY) or {}).get(p.index, ()))
    out: list[AbstractSet[str]] = []
    na_pseudo = na_pseudonymized(p)
    if na_pseudo:  # R8-05: --na-value tokens the D35 allowlist did not keep are written as pseudonyms
        out.append(_keys_of(na_pseudo))
    labels = _native_labels(p)
    if labels:
        out.append(_keys_of(labels))
    kind = sensitive_kind(p)
    m = p.model
    if kind in (NAME, EMAIL, ADDRESS, NET, FREE_TEXT, LABEL):
        real = _real_values(tp, p, kind)
        if real:
            # the same-column check's normalised set (computed once per run, shared): norm_text = privacy.texts.key
            out.append(_cached_sets(tp, p, kind, real).norm)
    elif kind in (PHONE, SSN):
        out.append(_keys_of(_real_values(tp, p, kind)))
    elif m is not None and p.type is ColumnType.CITY:
        out.append(_keys_of(m.nonmissing_counts))
    elif m is not None and p.type is ColumnType.STATE:
        from ..detect.text import is_state

        out.append(_keys_of(v for v in m.nonmissing_counts if isinstance(v, str) and not is_state(v)))
    elif m is not None and p.type is ColumnType.ZIP:
        from ..detect.text import ZIP_HINTED_RE

        out.append(_keys_of(v for v in m.nonmissing_counts
                            if isinstance(v, str) and not ZIP_HINTED_RE.fullmatch(v.strip())))
    replaced = (p.params or {}).get(REPLACED_DATE_KEY)
    if replaced:
        out.append(_keys_of(replaced))
    return union(out)


def _kept_keys(p: ColumnProfile) -> set[str]:
    """The registry keys of the real text values column ``p`` writes verbatim (the D35 excuse list): the NA tokens it
    holds and writes (R8-05: never one the D35 allowlist pseudonymised, nor a configured token it does not hold),
    and, for a column whose real values passed the copy gate (``detect.text.copy_cleared``; one source of truth with
    the gate and :func:`cross_column_values`), its kept levels, multi-select items (and the combinations made of
    kept items) and tokens, the placeholder tokens and sentinels of a date column, and the states of a state
    column; never a pseudonymised unit."""
    if p.role in ("excluded", "null"):
        return set()
    m = p.model
    if m is None:
        return set()
    na, gone = p.na_tokens or frozenset(), {**na_pseudonymized(p), **pseudonymized(p)}
    out = set(_keys_of(k for k in (*m.missing_counts, *m.nonmissing_counts)
                       if isinstance(k, str) and k.strip() in na and k.strip() not in gone))
    out |= _keys_of(k for k in m.missing_counts if isinstance(k, str) and k.strip() not in gone)
    if not (copies_real_values(p) and copy_cleared(p)):
        return out
    pseudo = pseudonymized(p)
    params = p.params or {}
    family = TYPE_INFO[p.type].family
    if p.type is ColumnType.MULTI_SELECT:
        d = params.get("delimiter")
        for raw in m.nonmissing_counts:
            if not isinstance(raw, str):
                continue
            if raw.strip() in pseudo:  # K-11: a whole combination the allowlist did not keep excuses only the items
                stay = combination_stay(p, unit_key(raw.strip()))  # it writes as they are (H-7)
                out |= _keys_of([x.strip() for x in raw.split(d) if x.strip() and unit_key(x.strip()) in stay]
                                if d else [])
                continue
            items = [x.strip() for x in raw.split(d)] if d else [raw.strip()]
            items = [x for x in items if x]
            kept = [x for x in items if x not in pseudo]
            out |= _keys_of(kept)
            if len(kept) == len(items):
                out |= _keys_of([raw])
    elif family == "date":
        out |= _keys_of(str(t) for t in (params.get("_other_tokens") or ()))
        out |= _keys_of(str(t) for t in (params.get("sentinels") or {}))
    elif family == "numeric":
        tokens = params.get("tokens") or params.get("_tokens") or ()
        out |= _keys_of(str(t).strip() for t in tokens if str(t).strip() not in pseudo)
    elif p.type is ColumnType.STATE:
        from ..detect.text import is_state

        out |= _keys_of(v for v in m.nonmissing_counts if isinstance(v, str) and is_state(v))
    elif p.type is not ColumnType.ZIP:
        out |= _keys_of(v for v in m.nonmissing_counts if isinstance(v, str) and v.strip() not in pseudo)
    return out


def dropped_texts(fr: FileResult) -> list[str]:
    """R8-04: the source text of a file outside its tables that no output holds: HTML comments, script text and
    attributes the writer drops (``io.excel_html.dropped_texts``), free-form Arrow schema and field metadata
    (``io.arrowio.dropped_metadata_texts``), Stata variable labels and the data label (PRIVATE)."""
    extra = (fr.layout.extra or {}) if fr.layout is not None else {}
    out: list[str] = []
    if extra.get("_events"):
        from ..io.excel_html import dropped_texts as html_dropped

        out.extend(html_dropped(extra["_events"]))
    schema = extra.get("_arrow_schema")
    if schema is not None:
        from ..io.arrowio import dropped_metadata_texts

        out.extend(dropped_metadata_texts(schema))
    if isinstance(extra.get("_data_label"), str):
        out.append(extra["_data_label"])
    for info in (extra.get("_columns") or {}).values():
        if isinstance(info, dict) and isinstance(info.get("var_label"), str):
            out.append(info["var_label"])
    return out


def outside_texts(fr: FileResult, keep_titles: bool = False) -> tuple[list[str], list[str]]:
    """R8-04: (the source text outside the tables of ``fr`` that the run does not write verbatim, the text it writes
    verbatim). Title, notes, footer and side-note text (``privacy.texts.layout_texts``) is written as placeholders,
    or verbatim with ``--keep-titles``; a footer's "Total"-vocabulary label is always written verbatim; the text no
    output holds (:func:`dropped_texts`) never is. PRIVATE."""
    from ..io.excel_write import _KEEP_LABEL_RE

    unwritten: list[str] = [t for t in dropped_texts(fr) if t.strip()]
    written: list[str] = []
    for text, _ in real_texts.layout_texts(fr):
        (written if keep_titles or _KEEP_LABEL_RE.match(text) else unwritten).append(text)
    return list(dict.fromkeys(unwritten)), list(dict.fromkeys(written))


def outside_digit_terms(texts: Iterable[str]) -> "DigitTerms":
    """The phone numbers and SSNs inside source text outside the tables (:func:`outside_texts`)."""
    return digit_terms(LABEL, sorted(set(texts)))


def real_texts_of(files: Iterable[FileResult], keep_titles: bool = False) -> "real_texts.RealTexts":
    """The run's :class:`privacy.texts.RealTexts` (module docstring of :mod:`privacy.texts`): the real text values
    of every column of every file (a FAILed file's people are real people too) that the run does not write
    verbatim, the source text outside the tables (:func:`outside_texts`; ``keep_titles``: ``--keep-titles``), and the
    ones it writes verbatim in the files that are written (which are never hits)."""
    from ..profile import entity as entity_mod

    sources: list[tuple[real_texts.Source, AbstractSet[str]]] = []
    kept: set[str] = set()
    id_emails: set[str] = set()
    outside_raw: list[tuple[str, tuple[str, ...]]] = []
    for fr in files:
        unwritten, written = outside_texts(fr, keep_titles and not fr.failed)  # a FAILed file writes nothing
        outside_kept = _keys_of(written)
        outside = frozenset(_keys_of(unwritten) - outside_kept)
        if outside:
            sources.append((real_texts.Source(fr.rel_path, None, real_texts.LAYOUT_WHERE, real_texts.LAYOUT),
                            outside))
        if unwritten:
            outside_raw.append((fr.rel_path, tuple(unwritten)))
        if not fr.failed:
            kept |= outside_kept
        for tp in fr.tables:
            for p in tp.columns:
                keys = _unkept_keys(tp, p)
                if keys:
                    sources.append((real_texts.Source(fr.rel_path, tp.name, p.name, _source_kind(p)), keys))
                if not fr.failed:
                    kept |= _kept_keys(p)
            # the e-mail addresses among the run's real IDs (ID columns, and those a flag turned into something else)
            for raw in entity_mod.privacy_id_raw(tp).values():
                id_emails.update(real_texts.key(v) for v in codes_of(raw)[1].tolist()
                                 if isinstance(v, str) and "@" in v)
    # the local parts of the real e-mail addresses (a username): never the local part of a synthetic address
    emails = (k for _, ks in sources for k in ks if "@" in k and k not in kept)
    locals_ = real_texts.local_parts(emails) | real_texts.local_parts(id_emails)
    return real_texts.RealTexts(union([ks for _, ks in sources]), frozenset(kept), tuple(sources),
                                frozenset(locals_), tuple(outside_raw))


def plan_real_texts(files: Iterable[FileResult], keep_titles: bool = False,
                    texts: "real_texts.RealTexts | None" = None) -> "real_texts.RealTexts":
    """Run-level planning (pipeline, after the run's D35 decision and the D34 screen): the run's real texts that no
    generator may write (:func:`real_texts_of`; ``keep_titles``: ``--keep-titles``; ``texts``: computed already for
    the same files in the same state), handed to every table (``tp.private.extra[privacy.texts.RUN_KEY]``):
    generation and the title placeholders avoid them (``privacy.texts.guard``) and the scan checks every output text
    column, and every ID column's e-mail addresses and usernames, against them."""
    files = list(files)
    if texts is None:
        texts = real_texts_of(files, keep_titles)
    for fr in files:
        for tp in fr.tables:
            tp.private.extra[real_texts.RUN_KEY] = texts
    return texts


def run_texts(tp: TableProfile) -> "real_texts.RealTexts | None":
    """The run's real-text registry handed to ``tp`` in planning (None for a table profiled on its own)."""
    return tp.private.extra.get(real_texts.RUN_KEY)


def id_user_hits(texts: "real_texts.RealTexts | None", p: ColumnProfile, values: list[str],
                 prefix: str | None = None) -> set[str]:
    """R8-03: the values of an ID column (not one that keeps its real row numbers) that are a real e-mail address of
    the run, or hold the local part or the username of one (``privacy.texts.RealTexts.user_hits``; ``prefix``: the
    column's kept ID prefix, re-checked when not given). Other columns: none (their values are checked as texts)."""
    if not texts or not values or sensitive_kind(p) != ID or p.params.get("_identity_mapped"):
        return set()
    if prefix is None:
        prefix = checked_prefix(str(p.params.get("_id_prefix") or ""))
    return texts.user_hits(values, prefix)


def cross_text_values(p: ColumnProfile, values: list[str]) -> list[str]:
    """The texts of an output column checked against the run's real-text registry: the values the run-wide patterns
    apply to (:func:`cross_column_values`: not IDs, codes, numbers, dates or the real levels a column keeps), and the
    items of a multi-select (its combinations are made of kept items and pseudonyms)."""
    return _with_items(p, cross_column_values(p, values))


def id_texts(p: ColumnProfile, values: list[str]) -> list[str]:
    """R9-03: the values of a synthetic ID column (not one that keeps its real row numbers 1..n) that are checked
    against the run's real-text registry as a whole: the ID registry never draws one of its texts
    (``generate.ids.RealUsers.text_taken``), so a hit is a copy. Other columns: none (:func:`cross_text_values`)."""
    if not values or sensitive_kind(p) != ID or p.params.get("_identity_mapped"):
        return []
    return values


def _with_items(p: ColumnProfile, vals: list[str]) -> list[str]:
    """``vals`` with each multi-select combination replaced by its items (other columns: ``vals``)."""
    d = (p.params or {}).get("delimiter") if p.type is ColumnType.MULTI_SELECT else None
    if not d or not vals:
        return vals
    out: dict[str, None] = {}
    for v in vals:
        if d in v:
            out.update((x, None) for x in (s.strip() for s in v.split(d)) if x)
        else:
            out[v] = None
    return list(out)


# --------------------------------------------------------------------------- layer 1: per column, before writing
def scan_table(tp: TableProfile, synth: SyntheticTable) -> list[Message]:
    """Per-column never-copy check for one synthetic table (called before writing)."""
    out: list[Message] = []
    by_index = {c.index: c for c in synth.columns}
    distinct: dict[int, list[str]] = {}

    def values_of(index: int) -> list[str]:
        if index not in distinct:
            distinct[index] = _distinct_strings(by_index[index].values)
        return distinct[index]
    flagged: set[int] = set()  # columns with a copy of their own real value (one error per column)
    for p in tp.columns:
        if p.role in ("excluded", "null") or p.index not in by_index:
            continue
        kind = sensitive_kind(p)
        never = p.type in NEVER_COPY_TYPES or kind is not None
        if not never:
            continue
        kind = kind or LABEL
        if kind == ID and p.params.get("_identity_mapped"):
            continue
        real = _real_values(tp, p, kind)
        if not real:
            continue
        hits = _same_column_hits(kind, _cached_sets(tp, p, kind, real),
                                 _item_strings(p, values_of(p.index)))
        if hits:
            flagged.add(p.index)
            what = (f"a never-copy {KIND_TEXT[kind]} column" if p.type in NEVER_COPY_TYPES or not pseudonymized(p)
                    else "the column that the D35 allowlist pseudonymised")
            out.append(Message("error", "copied_value",
                               f"{LEAK_PREFIX}{hits} synthetic value(s) equal a real value of {what}",
                               column=p.name,
                               fix=f"--null {p.name} (or --exclude {p.name}) keeps the column out until the cause "
                                   "is fixed; please report this as a bug"))
    # R9-01: no column writes a real value of its own that is a real person ID of the run (a level, a number, an
    # organisational code, a token ...), whatever the D35 allowlist or the key discovery decided: never excused
    ids = realids.run_ids(tp)
    if ids:
        for p in tp.columns:
            if p.role in ("excluded", "null") or p.index not in by_index or p.index in flagged:
                continue
            n_ids = len(realids.kept_hits(p, values_of(p.index), ids))
            if n_ids:
                flagged.add(p.index)
                out.append(Message("error", "copied_value",
                                   f"{LEAK_PREFIX}{n_ids} synthetic value(s) are a real value of the column that is a "
                                   "real person ID of the run",
                                   column=p.name,
                                   fix=f"--null {p.name} (or --exclude {p.name}) keeps the column out until the "
                                       "cause is fixed; please report this as a bug"))
    # run-wide (R7-01, R7-02): no output text equals a real text of ANY column of the run that the run does not write
    # verbatim (a pseudonymised label, a name, free text, a value of an excluded or nulled column ...); no ID is, or
    # has the local part or username of, a real e-mail address of the run (R8-03), nor equals such a text (R9-03)
    texts = run_texts(tp)
    if texts:
        for p in tp.columns:
            if p.role in ("excluded", "null") or p.index not in by_index or p.index in flagged:
                continue
            vals = values_of(p.index)
            n_users = len(id_user_hits(texts, p, vals))
            if n_users:
                out.append(Message("error", "copied_value",
                                   f"{LEAK_PREFIX}{n_users} synthetic ID(s) are, or hold the local part or username "
                                   "of, a real e-mail address held in the run",
                                   column=p.name,
                                   fix=f"--null {p.name} (or --exclude {p.name}) keeps the column out until the "
                                       "cause is fixed; please report this as a bug"))
            n = len(texts.hits(cross_text_values(p, vals) or id_texts(p, vals)))
            if n:
                out.append(Message("error", "real_text_copied",
                                   f"{LEAK_PREFIX}{n} synthetic value(s) equal a real text value held in the run "
                                   "(in this or another column or file) that the run never writes",
                                   column=p.name,
                                   fix=f"--null {p.name} (or --exclude {p.name}) keeps the column out until the "
                                       "cause is fixed; please report this as a bug"))
    # run-wide: no name value or row of name parts spells a real person's full name of the run (R1-04, R2-06)
    n_names = pii.real_full_name_hits(tp, {c.index: c.values for c in synth.columns})
    if n_names:
        out.append(Message("error", "real_full_name",
                           f"{LEAK_PREFIX}{n_names} synthetic name value(s) or row(s) spell a real person's full "
                           "name held in the run",
                           fix="please report this as a bug; --null COL (or --exclude COL) keeps a name column out "
                               "until the cause is fixed"))
    return out


# --------------------------------------------------------------------------- sensitive value index
@dataclass(frozen=True)
class Origin:
    """Where a sensitive value came from (file, table, column) - names only, never values."""

    file: str
    table: str | None
    column: str
    kind: str

    def ref(self) -> str:
        return f"{self.file}::{self.table}::{self.column}" if self.table else f"{self.file}::{self.column}"


@dataclass
class _ColumnTerms:
    origin: int
    kind: str
    real: _RealSets


class Unverified:
    """The files whose outputs the scan could not verify because of an internal error (R6-04, D19): each of them
    FAILs on its own (its output is removed), while every other file is still verified and kept.

    ``files``: source rel path -> [the first error, whether its report details are withheld]. Details are withheld
    (``FileResult.withheld``) when the error may have left real values of the file out of the index (indexing one of
    its columns) or kept its report page unscanned, so nothing unverified about it is printed."""

    def __init__(self) -> None:
        self.files: dict[str, list[Any]] = {}

    def add(self, rel: str, exc: BaseException, withhold: bool = False) -> None:
        entry = self.files.setdefault(rel, [exc, False])
        entry[1] = entry[1] or withhold

    def __bool__(self) -> bool:
        return bool(self.files)


#: Start of the FAIL reason of a file the scan could not verify (``report.report`` withholds its details).
UNVERIFIED_TEXT = "the leak scan could not verify this file"


def error_place(exc: BaseException) -> str:
    """``IndexError at leaks.py:826``: the error's type and where it was raised (never its message, which may quote
    a value: ``diagnostics.error_place``)."""
    from .. import diagnostics

    return diagnostics.error_place(exc)


@dataclass
class _Namespace:
    exact: set[str] = field(default_factory=set)
    normalized: set[str] = field(default_factory=set)
    origins: set[int] = field(default_factory=set)
    identity: bool = False
    locals: set[str] = field(default_factory=set)  # R5-02: local parts of the e-mail addresses among its IDs
    stems: set[str] = field(default_factory=set)  # R5-02: usernames of its IDs (detect.ids.username_stem)
    kept: set[str] = field(default_factory=set)  # R5-02: the prefix(es) the generator kept (params['_id_prefix'])
    prefix: str = ""  # R5-02: that prefix re-checked (:func:`namespace_users`)


def namespace_users(values: AbstractSet[str], kept: AbstractSet[str],
                    min_cell: int = PREFIX_MIN_IDS) -> tuple[str, set[str]]:
    """R5-02: (the prefix the usernames of one ID namespace are read without, the usernames of ``values``).

    The prefix is the one the generator kept (``kept``: the longest when columns differ), re-checked on its
    own (``detect.ids.checked_prefix``) with the number of distinct patterned IDs of the namespace that start
    with it: a name-like prefix, or one fewer than ``min_cell`` IDs share, is cut to its first letter, so a
    real username kept as a "prefix" is still a hit.
    """
    given = max(sorted(kept), key=len) if kept else ""
    if len(given) > 1:
        keys = {normalize_id(v) for v in values
                if v[:len(given)].casefold() == given and not ("@" in v and is_email_id(v))}
        given = checked_prefix(given, len(keys), min_cell)
    _, stems = id_users(values, given)
    return given, stems


def _id_scannable(tok: str) -> bool:
    return len(tok) >= MIN_ID_TOKEN


def _word_scannable(tok: str) -> bool:
    if len(tok) < 2:
        return False
    return not (tok.isdigit() and len(tok) < MIN_ID_TOKEN)


def _phrases_into(target: dict[int, set[tuple[str, ...]]], toks: list[str]) -> None:
    """A token sequence of 2-6 tokens as itself, a longer one as its 6-grams."""
    n = len(toks)
    if n < 2:
        return
    if n <= NGRAM:
        target[n].add(tuple(toks))
        return
    grams = target[NGRAM]
    for i in range(n - NGRAM + 1):
        grams.add(tuple(toks[i:i + NGRAM]))


@dataclass(frozen=True)
class DigitTerms:
    """Phone/SSN terms of one column's real values (PRIVATE, in memory only).

    ``digits``: phone/SSN keys of >= 7 digits (report/docs rule); ``phones``: phone keys of >= 10 digits and
    ``ssns``: SSN keys (data rules, and what the generators never draw: :mod:`privacy.digits`); ``layouts``:
    digit-group layouts of the column's real phone numbers (``(3, 4, 4)`` for ``020 7946 0958``);
    ``short``: values of a phone/SSN column with fewer than 7 digits (indexed as labels);
    ``phone_bases``: D46 canonical phone bases and local/NANP aliases, with extensions removed.
    """

    digits: frozenset[str] = frozenset()
    phones: frozenset[str] = frozenset()
    ssns: frozenset[str] = frozenset()
    layouts: frozenset[tuple[int, ...]] = frozenset()
    short: tuple[str, ...] = ()
    phone_bases: frozenset[str] = frozenset()


def _known_phone(p: ColumnProfile) -> bool:
    return p.type is ColumnType.PHONE or p.params.get("pii_hint") == "phone"


def _source_phone_bases(p: ColumnProfile) -> frozenset[str]:
    """All source phone spans, including levels an explicit fidelity flag keeps."""
    m = p.model
    if m is None or (not _known_phone(p) and
                     (p.type is ColumnType.IDENTIFIER or TYPE_INFO[p.type].family in ("numeric", "date"))):
        return frozenset()
    key = "_semantic_phone_bases"
    if key not in m.extra:
        m.extra[key] = frozenset(k for v in (*m.nonmissing_counts, *m.missing_counts)
                                 if isinstance(v, str) and any(c.isdecimal() for c in v)
                                 for k in semantic_phone_keys(v, known=_known_phone(p), source=True))
    return m.extra[key]


_NINES = str.maketrans("0123456789", "9999999999")
_ANY_DIGIT = re.compile(r"\d")


def _shape_terms(shape: str) -> tuple[list[tuple[int, int]], set[tuple[int, ...]], bool]:
    """(phone-shaped token spans, phone layouts, whether one token holds every digit) of a phone value's
    shape (its digits written as ``9``): values of one shape share them."""
    lays = phone_layouts(shape)
    spans = [(b, e) for b, e, _ in phone_tokens(shape, NANP_LAYOUTS | lays)]
    n_digits = shape.count("9")
    whole = len(spans) == 1 and shape[spans[0][0]:spans[0][1]].count("9") == n_digits
    return spans, lays, whole


def digit_terms(kind: str, values: list[str], *, phone_bases: frozenset[str] | None = None) -> DigitTerms:
    """The :class:`DigitTerms` of a sensitive column of ``kind`` with these real values.

    A phone column gives the digits of each value (country code 1 dropped too) and, worked out once per
    value shape, the phone-shaped tokens inside it (the number of ``312-555-0199 x12``) and its layouts. A
    text column (name, label, address, free text, an e-mail column's values without ``@``) gives the phone
    numbers of >= 10 digits and SSNs inside its values (R3), so a copy in any layout (``312.555.0199`` for
    ``312-555-0199``) or in another column is found.
    """
    if kind in (ID, CODE, NET):
        return DigitTerms()
    digits: set[str] = set()
    phones: set[str] = set()
    ssns: set[str] = set()
    layouts: set[tuple[int, ...]] = set()
    bases = (phone_bases if phone_bases is not None else
             frozenset(k for v in values if any(c.isdecimal() for c in v)
                       for k in semantic_phone_keys(v, known=kind == PHONE, source=True))
             if kind != SSN else frozenset())
    digits.update(k for k in bases if k.isdigit())
    if kind in (PHONE, SSN):
        short: list[str] = []
        shapes: dict[str, tuple[list[tuple[int, int]], set[tuple[int, ...]], bool]] = {}
        for v in values:
            d = digits_of(phone_text(v) if kind == SSN else v)
            if len(d) < 7:
                short.append(v)
                continue
            for k in phone_keys(d):
                digits.add(k)
                if kind == PHONE and len(k) >= MIN_PHONE:
                    phones.add(k)
            if kind == SSN and len(d) == 9:
                ssns.add(d)
            if kind == PHONE and len(d) >= MIN_PHONE:
                shape = v.translate(_NINES)
                info = shapes.get(shape)
                if info is None:
                    info = shapes[shape] = _shape_terms(shape)
                    layouts |= info[1]
                if not info[2]:
                    for b, e in info[0]:
                        for k in phone_keys(digits_of(v[b:e])):
                            digits.add(k)
                            phones.add(k)
        return DigitTerms(frozenset(digits), frozenset(phones), frozenset(ssns), frozenset(layouts), tuple(short), bases)
    texts = [v for v in values if "@" not in v] if kind == EMAIL else values
    for v in texts:
        if not _ANY_DIGIT.search(v):
            continue
        t = v.translate(_PUNCT_FOLD)
        for m in PHONE10_RE.finditer(t):
            if _DECIMAL_NUMBER.fullmatch(m.group()):
                continue
            for k in phone_keys(digits_of(m.group())):
                digits.add(k)
                phones.add(k)
        for _, _, tok in phone_tokens(t):  # a NANP number inside a longer digit cluster (``Rm 12 312-555-0199``)
            for k in phone_keys(tok):
                digits.add(k)
                phones.add(k)
        for m in SSN_RE.finditer(t):
            d = digits_of(m.group())
            digits.add(d)
            ssns.add(d)
    return DigitTerms(frozenset(digits), frozenset(phones), frozenset(ssns), phone_bases=bases)


def numbered_labels(values: Iterable[str], header: str) -> set[str]:
    """The values of a label column that are its header's words and a number (``Week 36`` under ``Week``,
    ``Question 12`` under ``Item Question``, ``Day 180`` under ``Day``): a numbered label names no one, and the
    report prints the header beside a count (``[Week] 36 distinct values``) that would read as the value."""
    head = {t for t in WORD_RE.findall(fold(str(header))) if not t.isdigit()}
    if not head:
        return set()
    out: set[str] = set()
    for v in values:
        toks = WORD_RE.findall(fold(v))
        letters = [t for t in toks if not t.isdigit()]
        if letters and len(letters) < len(toks) <= NGRAM and set(letters) <= head:
            out.add(v)
    return out


class _Terms:
    """The terms of one column before they are merged into the index under its origin."""

    def __init__(self) -> None:
        self.words: set[str] = set()
        self.exact: set[str] = set()
        self.net: set[str] = set()
        self.emails: set[str] = set()
        self.locals: set[str] = set()
        self.digits: set[str] = set()
        self.phone_digits: set[str] = set()
        self.phone_bases: set[str] = set()
        self.ssn_digits: set[str] = set()
        self.phone_layouts: set[tuple[int, ...]] = set()
        self.phrases: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        self.data_phrases: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        #: all-digit token sequences of whole label values (``0,89``, ``2.396,27``, ``56,1%``) -> their text
        #: from the first to the last digit; see :func:`Scanner.hits_in_text`
        self.number_literals: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.number_strict: set[tuple[str, ...]] = set()  # 6-grams of longer values: always matched

    def add(self, kind: str, values: list[str], dt: DigitTerms | None = None) -> None:
        """Index ``values`` of a column of ``kind`` (``dt``: its :func:`digit_terms` when already computed)."""
        if kind in (ID, CODE):
            self.exact.update(v for v in values if _id_scannable(v))
            if kind == ID:  # R5-02: e-mail addresses used as IDs are e-mail addresses too
                for v in values:
                    loc = _id_local(v)
                    if loc is not None:
                        self.emails.add(fold(v.strip()))
                        if len(loc) >= 3:
                            self.locals.add(loc)
            return
        if kind == NET:
            self.net.update(fold(v) for v in values if len(v) >= MIN_ID_TOKEN)
            return
        if dt is None:
            dt = digit_terms(kind, values)
        # the column's key sets are shared, not copied (a phone column can hold a million keys)
        self.digits = self.digits | dt.digits if self.digits else dt.digits  # type: ignore[assignment]
        self.phone_digits = self.phone_digits | dt.phones if self.phone_digits else dt.phones  # type: ignore[assignment]
        self.phone_bases = self.phone_bases | dt.phone_bases if self.phone_bases else dt.phone_bases  # type: ignore[assignment]
        self.ssn_digits = self.ssn_digits | dt.ssns if self.ssn_digits else dt.ssns  # type: ignore[assignment]
        self.phone_layouts |= dt.layouts
        if kind in (PHONE, SSN):
            self._text(LABEL, list(dt.short))
            return
        if kind == EMAIL:
            rest: list[str] = []
            for v in values:
                e = fold(v)
                if "@" in e:
                    self.emails.add(e)
                    local = e.partition("@")[0]
                    if len(local) >= 3:
                        self.locals.add(local)
                else:
                    rest.append(v)
            self._text(LABEL, rest)
            return
        self._text(kind, values)

    def _embedded(self, values: list[str]) -> None:
        """E-mail addresses inside text values (R3): the addresses (and local parts) in a label, free-text or
        multi-select value are indexed as such too, so a copy in another column is found (phone numbers and
        SSNs inside text values: :func:`digit_terms`)."""
        for v in values:
            if "@" in v:
                for e in EMAIL_RE.findall(v):
                    e = fold(e)
                    self.emails.add(e)
                    local = e.partition("@")[0]
                    if len(local) >= 3:
                        self.locals.add(local)

    def _text(self, kind: str, values: list[str]) -> None:
        findall = WORD_RE.findall
        self._embedded(values)
        for v in values:
            fv = fold(v)
            toks = findall(fv)
            if kind == NAME:
                self.words.update(t for t in toks if _word_scannable(t) and not t.isdigit())
                for t in toks:  # each part of a hyphenated / apostrophe name is a name too
                    self.words.update(word_parts(t))
                _phrases_into(self.phrases, toks)
            elif kind == LABEL:
                if not toks:
                    # R6-04: a label without word tokens (a check mark, an emoji) gives no text term; the same-column
                    # check still compares it whole (``_RealSets``)
                    continue
                if len(toks) == 1:
                    if _word_scannable(toks[0]):
                        self.words.add(toks[0])
                else:
                    _phrases_into(self.phrases, toks)
                    if len(toks) <= NGRAM and all(t.isdigit() for t in toks):
                        # a number written as text (decimal comma, thousands dots): its digits match a report
                        # statistic in another notation (0,89 vs a share 0.89) only by chance
                        spans = [m.span() for m in WORD_RE.finditer(fv)]
                        self.number_literals[tuple(toks)].add(fv[spans[0][0]:spans[-1][1]])
                    elif len(toks) > NGRAM:
                        self.number_strict.update(tuple(toks[i:i + NGRAM]) for i in range(len(toks) - NGRAM + 1))
            elif kind == ADDRESS:
                _phrases_into(self.phrases, toks)
                if len(toks) >= 3 and any(ch.isdigit() for t in toks for ch in t):
                    _phrases_into(self.data_phrases, toks)
            elif kind == FREE_TEXT:
                parts = [toks]
                # report/docs only: a short whole value (``Severe peanut allergy``) is a phrase too
                if 2 <= len(toks) < MIN_SENTENCE_WORDS:
                    _phrases_into(self.phrases, toks)
                segments = [x for x in SENTENCE_RE.split(fv) if x.strip()] if SENTENCE_RE.search(fv) else []
                if len(segments) > 1:  # one segment is the whole value again
                    sentences = [findall(x) for x in segments]
                    for s in sentences:
                        if MIN_SHORT_SENTENCE <= len(s) < MIN_SENTENCE_WORDS:
                            _phrases_into(self.phrases, s)
                    parts += sentences
                for part in parts:
                    if len(part) >= MIN_SENTENCE_WORDS:
                        _phrases_into(self.phrases, part)
                        _phrases_into(self.data_phrases, part)


class TermTable:
    """Terms of every sensitive column: one set per origin plus their union for fast negative lookups.

    ``table.get(term)`` returns the set of origins holding ``term`` (None when no column does);
    iterating yields every term. Columns are few, so resolving a hit's origins is cheap.
    """

    __slots__ = ("union", "parts")

    def __init__(self) -> None:
        self.union: set[Any] = set()
        self.parts: list[tuple[int, set[Any]]] = []

    def add(self, terms: AbstractSet[Any], origin: int) -> None:
        if terms:
            if not self.parts:
                self.union = terms  # type: ignore[assignment]  # one part: shared, not copied
            else:
                if self.union is self.parts[0][1]:
                    self.union = set(self.union)  # a second part: the union gets its own set
                self.union |= terms
            self.parts.append((origin, terms))  # type: ignore[arg-type]

    def get(self, term: Any) -> set[int] | None:
        if term not in self.union:
            return None
        return {o for o, terms in self.parts if term in terms}

    def __contains__(self, term: Any) -> bool:
        return term in self.union

    def __iter__(self):  # noqa: ANN204
        return iter(self.union)

    def __len__(self) -> int:
        return len(self.union)

    def __bool__(self) -> bool:
        return bool(self.union)


class SensitiveIndex:
    """Every sensitive real value of the run, indexed for the matching rules (PRIVATE, in memory only)."""

    def __init__(self) -> None:
        self.origins: list[Origin] = []
        self.columns: dict[tuple[str, int, int], _ColumnTerms] = {}
        self.column_namespace: dict[tuple[str, int, int], str] = {}
        self.namespaces: dict[str, _Namespace] = {}
        #: run-wide real IDs of the non-identity namespaces: exact, D8 keys, e-mail local parts, usernames
        self._id_union: tuple[set[str], set[str], set[str], set[str]] | None = None
        self.words = TermTable()
        self.exact = TermTable()
        self.net = TermTable()
        self.phrases: dict[int, TermTable] = {}
        self.data_phrases: dict[int, TermTable] = {}
        self.data_vocab: set[str] = set()  # every token of a data phrase
        #: all-digit phrases that come only from whole numeric label values -> their real texts; any other
        #: origin of the same digits (a 6-gram, a name, an address ...) puts the phrase in ``number_strict``
        self.number_literals: dict[tuple[str, ...], set[str]] = {}
        self.number_strict: set[tuple[str, ...]] = set()
        self.emails = TermTable()
        self.locals = TermTable()
        self.digits = TermTable()
        self.phone_digits = TermTable()
        self.phone_bases = TermTable()
        self.ssn_digits = TermTable()
        #: digit-group layouts that make a formatted phone number (NANP and the run's real phones)
        self.phone_layouts: set[tuple[int, ...]] = set(NANP_LAYOUTS)
        #: D34: the run's flagged names (``privacy.headers.NameScreen``) and the origin of each flagged text
        self.screen: Any = None
        self.flag_origins: dict[str, int] = {}
        #: R7-01, R7-02: the run's real texts that it does not write verbatim (``privacy.texts``; data scan) and the
        #: origin of each of its sources (same order)
        self.texts: real_texts.RealTexts | None = None
        self.text_origins: list[int] = []
        #: R9-01: the run's real person IDs (``privacy.realids``): no column writes a real value of its own that is one
        self.real_ids: realids.RealIds | None = None

    @property
    def empty(self) -> bool:
        return not self.origins

    def flagged_in(self, text: str, parent: str | None = None) -> set[int]:
        """Origins of the flagged names (D34) that ``text`` holds (a written header or sheet name; ``parent``: its
        table's or workbook's name). The same matcher as D34's step 3 (``NameScreen.repeats``, K-16, K-18), so a name
        D34 let through is never a leak here."""
        if self.screen is None or not text:
            return set()
        from .headers import numbered_context

        # a name flagged only beside another flagged name (one word of a person's name in a typed glob) is not
        # personal data on its own: a clean header that equals it is no leak
        found = self.screen.repeats(text, numbered_context(parent))
        return {self.flag_origins[t] for t in found if t in self.flag_origins}

    def real_id_hits(self, syn: set[str], prefix: str = "") -> tuple[int, set[int]]:
        """(number of ``syn`` values equal to a real ID of any non-identity namespace, exactly or by D8 key, or
        holding the e-mail local part or the username of one (R5-02; ``prefix``: the column's kept ID prefix);
        the origins of the namespaces they match)."""
        live = [ns for ns in self.namespaces.values() if not ns.identity and ns.exact]
        if not live or not syn:
            return 0, set()
        if self._id_union is None:
            exact: set[str] = set()
            normalized: set[str] = set()
            locs: set[str] = set()
            stems: set[str] = set()
            for ns in live:
                exact |= ns.exact
                normalized |= ns.normalized
                locs |= ns.locals
                stems |= ns.stems
            self._id_union = (exact, normalized, locs, stems)
        exact, normalized, locs, stems = self._id_union
        keyed = {v: normalize_id(v) for v in syn}
        hit = {v: k for v, k in keyed.items() if v in exact or k in normalized}
        users: dict[str, tuple[str | None, str | None]] = {}
        if locs or stems:  # R5-02: the e-mail and username rules
            for v in syn:
                loc, st = _id_local(v), username_stem(v, prefix)
                if (loc is not None and loc in locs) or (st is not None and st in stems):
                    users[v] = (loc, st)
        origins: set[int] = set()
        for ns in live:
            if any(v in ns.exact or k in ns.normalized for v, k in hit.items()) or any(
                    (loc is not None and loc in ns.locals) or (st is not None and st in ns.stems)
                    for loc, st in users.values()):
                origins |= ns.origins
        return len(set(hit) | set(users)), origins

    # ---------------------------------------------------------------- building
    @classmethod
    def build(cls, files: Iterable[FileResult], screen: Any = None,
              min_cell: int = PREFIX_MIN_IDS, broken: "Unverified | None" = None) -> "SensitiveIndex":
        """The index of every sensitive real value of ``files`` (``min_cell``: ``--min-cell``, which the kept ID
        prefixes are re-checked with, R5-02) and, when ``screen`` (``privacy.headers.NameScreen``) is given, of
        every flagged header, sheet, file or folder name (D34): its word sequence, its person name words, its
        whole text and the ID parts of a real ID (:meth:`privacy.headers.Flag.terms`), plus the e-mail addresses,
        phone numbers and SSNs inside it.

        ``broken`` (R6-04, D19): when given, an internal error while one column is indexed marks that column's file
        (and an error in an ID namespace, every file of the namespace) as unverified there instead of ending the
        build: that file FAILs, every other file is still verified. What the column indexed before the error stays
        in the index (its e-mail addresses, phone numbers and SSNs are indexed first), so it still guards the rest of
        the run. Without ``broken`` the error propagates."""
        idx = cls()
        files = list(files)  # read twice: the columns, then the shadow IDs (R6-02)
        broken_ns: dict[str, BaseException] = {}  # ID namespaces with a column that could not be indexed
        for fr in files:
            for t_i, tp in enumerate(fr.tables):
                for p in tp.columns:
                    try:
                        idx._add_column(fr, t_i, tp, p)
                        bases = _source_phone_bases(p)
                        if bases:
                            ct = idx.columns.get((fr.rel_path, t_i, p.index))
                            o = ct.origin if ct is not None else len(idx.origins)
                            if ct is None:
                                idx.origins.append(Origin(fr.rel_path, tp.name, p.name, PHONE))
                            idx.phone_bases.add(bases, o)
                    except Exception as exc:  # noqa: BLE001 - one file's column never stops the others' scan
                        if broken is None:
                            raise
                        broken.add(fr.rel_path, exc, withhold=True)
                        key = idx.column_namespace.get((fr.rel_path, t_i, p.index))
                        if key is not None:
                            broken_ns.setdefault(key, exc)
        idx._add_shadow_ids(files)
        idx._add_real_texts(files, broken)
        # R9-01: the run's real person IDs, planned (privacy.realids.plan) or read from the files for a run scanned
        # without planning
        idx.real_ids = next((i for fr in files for tp in fr.tables if (i := realids.run_ids(tp)) is not None), None)
        if idx.real_ids is None:
            idx.real_ids = realids.of_files(files)
        for key, ns in idx.namespaces.items():
            if not ns.identity and ns.exact:
                try:
                    ns.prefix, ns.stems = namespace_users(ns.exact, ns.kept, min_cell)
                except Exception as exc:  # noqa: BLE001
                    if broken is None:
                        raise
                    broken_ns.setdefault(key, exc)
        if broken is not None:  # the files sharing an ID namespace are checked against its IDs: none is verified
            for (rel, _, _), key in idx.column_namespace.items():
                if key in broken_ns:
                    broken.add(rel, broken_ns[key], withhold=True)
        if screen is not None and screen.flags:
            idx.screen = screen
            for f in screen.flags:
                if f.text in idx.flag_origins:
                    continue
                o = len(idx.origins)
                idx.origins.append(Origin(f.file or "(input)", None, f.where, HEADER))
                idx.flag_origins[f.text] = o
                words, seqs, exact = f.terms()
                terms = _Terms()
                terms.words.update(words)
                for seq in seqs:
                    terms.phrases[len(seq)].add(seq)
                terms.exact.update(exact)
                terms._embedded([f.text])
                dt = digit_terms(LABEL, [f.text, f.text.replace("_", " ")])
                terms.digits, terms.phone_digits, terms.ssn_digits = set(dt.digits), set(dt.phones), set(dt.ssns)
                terms.phone_bases = set(dt.phone_bases)
                idx._merge_terms(terms, o)
        return idx

    def _add_shadow_ids(self, files: Iterable[FileResult]) -> None:
        """R6-02: the real IDs of every column a flag turned from an ID column into something else
        (``--exclude student_id``, ``--null``, ``--force-*``: ``profile.entity.shadow_namespaces``) are real IDs of
        the run: indexed like an ID column's (the report, the docs and every synthetic ID column are checked for
        them), in namespaces of their own that no output column is checked against as its own (the column writes
        nothing, or what its flag asked for)."""
        from ..profile import entity as entity_mod

        for sns in entity_mod.shadow_namespaces(files):
            real = sorted(sns.raw_values)
            ns = self.namespaces.setdefault(sns.key, _Namespace())
            ns.exact.update(real)
            ns.normalized.update(map(normalize_id, real))
            ns.locals |= {loc for loc in map(_id_local, real) if loc is not None}
            terms = _Terms()
            terms.add(ID, real)
            for file, table, _, name in sns.members:
                o = len(self.origins)
                self.origins.append(Origin(file, table, name, ID))
                ns.origins.add(o)
                self._merge_terms(terms, o)

    def _add_real_texts(self, files: list[FileResult], broken: "Unverified | None") -> None:
        """R7-01, R7-02: the run's real-text registry (``privacy.texts``: planned by :func:`plan_real_texts`, or
        computed here for a run scanned without planning), with one origin per source column, for the data scan's
        cross-column rule; and the e-mail addresses, phone numbers and SSNs inside the values of an excluded or
        nulled column (data rules: the full addresses in the report and docs too). An internal error while one
        file's columns are read marks that file unverified (``broken``), as in :meth:`build`."""
        texts = next((t for fr in files for tp in fr.tables if (t := run_texts(tp)) is not None), None)
        if texts is None:
            parts: list[real_texts.RealTexts] = []
            for fr in files:
                try:
                    parts.append(real_texts_of([fr]))
                except Exception as exc:  # noqa: BLE001 - one file's columns never stop the others' scan
                    if broken is None:
                        raise
                    broken.add(fr.rel_path, exc, withhold=True)
            sources = tuple(s for t in parts for s in t.sources)
            texts = real_texts.RealTexts(union([ks for _, ks in sources]),
                                         frozenset().union(*(t.kept for t in parts)), sources,
                                         frozenset().union(*(t.locals for t in parts)),
                                         tuple(o for t in parts for o in t.outside))
        self.texts = texts
        where: dict[tuple[str, str | None, str], int] = {}
        for src, _ in texts.sources:
            ref = (src.file, src.table, src.column)
            if ref not in where:
                where[ref] = len(self.origins)
                self.origins.append(Origin(src.file, src.table, src.column,
                                           LAYOUT if src.kind == real_texts.LAYOUT else REAL_TEXT))
            self.text_origins.append(where[ref])
        # R8-04: the e-mail addresses, phone numbers and SSNs inside the source text outside the tables that the run
        # does not write verbatim (a title's contact address, a note's phone number) are real data: indexed for the
        # data rules (every output text and file metadata) and the full addresses for the report and docs too
        for rel, raw in texts.outside:
            try:
                dt = outside_digit_terms(raw)
                emails = {fold(e) for v in raw if "@" in v for e in EMAIL_RE.findall(v)}
            except Exception as exc:  # noqa: BLE001
                if broken is None:
                    raise
                broken.add(rel, exc, withhold=True)
                continue
            if not (dt.phones or dt.phone_bases or dt.ssns or emails):
                continue
            ref = (rel, None, real_texts.LAYOUT_WHERE)
            if ref not in where:
                where[ref] = len(self.origins)
                self.origins.append(Origin(rel, None, real_texts.LAYOUT_WHERE, LAYOUT))
            o = where[ref]
            self.phone_digits.add(dt.phones, o)
            self.phone_bases.add(dt.phone_bases, o)
            self.ssn_digits.add(dt.ssns, o)
            self.emails.add(emails, o)
        for fr in files:
            for tp in fr.tables:
                try:
                    terms = _dropped_digit_terms(tp)
                except Exception as exc:  # noqa: BLE001
                    if broken is None:
                        raise
                    broken.add(fr.rel_path, exc, withhold=True)
                    continue
                dropped = tp.private.extra.get(real_texts.DROPPED_KEY) or {}
                for i, dt in terms.items():
                    p = tp.columns[i]
                    ref = (fr.rel_path, tp.name, p.name)
                    emails = {fold(e) for v in dropped.get(i, ()) if "@" in v for e in EMAIL_RE.findall(v)}
                    if not (dt.phones or dt.phone_bases or dt.ssns or emails):
                        continue
                    if ref not in where:
                        where[ref] = len(self.origins)
                        self.origins.append(Origin(fr.rel_path, tp.name, p.name, REAL_TEXT))
                    o = where[ref]
                    self.phone_digits.add(dt.phones, o)
                    self.phone_bases.add(dt.phone_bases, o)
                    self.ssn_digits.add(dt.ssns, o)
                    self.emails.add(emails, o)

    def text_hits(self, p: ColumnProfile, values: list[str]) -> tuple[int, set[int]]:
        """(number of distinct texts among ``values`` that equal a real text of the run it never writes, their
        origins): the data scan's cross-column rule (R7-01, R7-02). ``values``: what :func:`cross_column_values`
        leaves of an output column (multi-select combinations are read item by item)."""
        texts = self.texts
        if not texts or not values:
            return 0, set()
        found = texts.hits(_with_items(p, values))
        if not found:
            return 0, set()
        origins = {self.text_origins[j] for j, (_, ks) in enumerate(texts.sources) if not found.isdisjoint(ks)}
        return len(found), origins

    def user_hits(self, p: ColumnProfile, values: list[str], prefix: str | None) -> tuple[int, set[int]]:
        """(number of distinct ID values that are, or hold the local part or username of, a real e-mail address of
        the run: :func:`id_user_hits`, R8-03; the origins of the registry sources holding those addresses)."""
        found = id_user_hits(self.texts, p, values, prefix)
        if not found:
            return 0, set()
        assert self.texts is not None
        locs: set[str] = set()
        stems: set[str] = set()
        for v in found:
            if "@" in v and is_email_id(v):
                locs.add(real_texts.key(v).rpartition("@")[0])
            st = username_stem(v, prefix or "")
            if st:
                stems.add(st)
        origins: set[int] = set()
        for j, (_, ks) in enumerate(self.texts.sources):
            for loc in real_texts.local_parts(k for k in ks if "@" in k):
                if loc in locs or "".join(ch for ch in loc if ch.isalpha()) in stems:
                    origins.add(self.text_origins[j])
                    break
        return len(found), origins

    def _add_column(self, fr: FileResult, t_i: int, tp: TableProfile, p: ColumnProfile) -> None:
        """Index the real values of one sensitive column (:meth:`build`)."""
        from ..profile import entity as entity_mod

        kind = sensitive_kind(p)
        if kind is None:
            return
        if kind == ID:
            key = entity_mod.id_namespace(fr.rel_path, tp.name, p)
            ns = self.namespaces.setdefault(key, _Namespace())
            self.column_namespace[(fr.rel_path, t_i, p.index)] = key
            if p.params.get("_identity_mapped"):
                ns.identity = True
                return
        real = _real_values(tp, p, kind)
        if not real:
            return
        o = len(self.origins)
        self.origins.append(Origin(fr.rel_path, tp.name, p.name, kind))
        terms = _Terms()
        try:
            if kind == ID:
                ns = self.namespaces[self.column_namespace[(fr.rel_path, t_i, p.index)]]
                ns.exact.update(real)
                ns.normalized.update(map(normalize_id, real))
                ns.origins.add(o)
                ns.locals |= {loc for loc in map(_id_local, real) if loc is not None}  # R5-02
                if p.params.get("_id_prefix"):
                    ns.kept.add(str(p.params["_id_prefix"]))
            else:
                self.columns[(fr.rel_path, t_i, p.index)] = _ColumnTerms(o, kind, _cached_sets(tp, p, kind, real))
            if kind == LABEL:  # 'Week 36' under 'Week': the report prints '[Week] 36 distinct values'
                header = tp.header[p.index] if p.index < len(tp.header) else p.name
                numbered = numbered_labels(real, header)
                # D36: a code written with a fake name in place of its real one keeps the rest of its text real by
                # design ('English 9 Honors Period 3 - Ms. Garcia' -> '... - Ms. Hendricks'): its name parts are
                # indexed ('Ms. Garcia', in ``real``), never the kept words around them (6 of them in a row would
                # match the synthetic code); the same-column and run-wide checks above still hold the whole text
                named = org_named_texts(p) - set(org_name_parts(p))
                if numbered or named:
                    real = [v for v in real if v not in numbered and v not in named]
            terms.add(kind, real, None if kind in (ID, CODE, NET) else _cached_digit_terms(p, kind, real))
        finally:
            self._merge_terms(terms, o)  # what was indexed before an internal error still guards the run (R6-04)

    def _merge_terms(self, t: _Terms, o: int) -> None:
        for name in ("words", "exact", "net", "emails", "locals", "digits", "phone_digits", "phone_bases", "ssn_digits"):
            getattr(self, name).add(getattr(t, name), o)
        self.phone_layouts |= t.phone_layouts
        for L, phrases in t.phrases.items():
            self.phrases.setdefault(L, TermTable()).add(phrases, o)
            for ph in phrases:
                if ph in t.number_literals and ph not in t.number_strict:
                    self.number_literals.setdefault(ph, set()).update(t.number_literals[ph])
                elif all(x.isdigit() for x in ph):
                    self.number_strict.add(ph)
        for L, phrases in t.data_phrases.items():
            self.data_phrases.setdefault(L, TermTable()).add(phrases, o)
            for ph in phrases:
                self.data_vocab.update(ph)


# --------------------------------------------------------------------------- public tokens (allowlist)
#: Literal collections longer than this are data (word lists), not the tool's own text.
_MAX_LITERAL_LIST = 50


def _tool_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _literal_checksum(texts: list[str]) -> str:
    encoded = json.dumps(texts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_literal_cache(path: Path, source_key: str) -> tuple[str, ...] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        texts = cached["literals"]
        if (cached.get("version") == 1 and cached.get("source") == source_key
                and isinstance(texts, list) and all(type(text) is str for text in texts)
                and cached.get("checksum") == _literal_checksum(texts)):
            return tuple(texts)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        pass
    return None


def _write_literal_cache(path: Path, source_key: str, texts: list[str]) -> None:
    temporary: str | None = None
    try:
        path.parent.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="tool_literals-", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            json.dump({"version": 1, "source": source_key, "literals": texts,
                       "checksum": _literal_checksum(texts)}, file, ensure_ascii=True)
        os.replace(temporary, path)
    except (OSError, UnicodeError, ValueError):
        pass  # a read-only installed package still computes the exact vocabulary
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


@functools.lru_cache(maxsize=1)
def _tool_literals() -> tuple[str, ...]:
    """Literal texts (:func:`literal_texts`) of every module of this package, plus the module names."""
    root = _tool_source_root()
    sources: list[tuple[Path, bytes]] = []
    fingerprint = hashlib.sha1()
    complete = True
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_bytes()
        except OSError:
            complete = False  # do not persist a vocabulary missing unreadable sources
            continue
        name = path.relative_to(root).as_posix().encode("utf-8")
        fingerprint.update(len(name).to_bytes(8, "big"))
        fingerprint.update(name)
        fingerprint.update(len(source).to_bytes(8, "big"))
        fingerprint.update(source)
        sources.append((path, source))
    source_key = fingerprint.hexdigest()
    cache_path = root / "__pycache__" / f"tool_literals-{source_key}.json"
    cached = _read_literal_cache(cache_path, source_key) if complete else None
    if cached is not None:
        return cached
    texts: list[str] = []
    for path, source in sources:
        try:
            # read_text used universal-newline translation; keep that exact AST
            # input so escaped CRLF literals and line continuations behave alike.
            decoded = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            texts.extend(literal_texts(decoded))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        texts.append(path.stem)
    if complete:
        _write_literal_cache(cache_path, source_key, texts)
    return tuple(texts)


def literal_texts(source: str) -> list[str]:
    """String and number literals and class names of Python ``source``: the text the tool itself writes.

    Left out: docstrings, and the elements of literal collections (list, tuple, set, dict displays) of more
    than ``_MAX_LITERAL_LIST`` items. Such a collection is DATA (a name, city or state list a detector
    validates values against), not text the tool writes, so its words must not excuse a real name.
    """
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    skip: set[int] = set()
    for node in nodes:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                skip.add(id(body[0].value))
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > _MAX_LITERAL_LIST:
            skip.update(id(e) for e in node.elts)
        elif isinstance(node, ast.Dict) and len(node.keys) > _MAX_LITERAL_LIST:
            skip.update(id(e) for e in node.keys + node.values if e is not None)
    texts: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Constant) and id(node) not in skip:
            v = node.value
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                texts.append(str(v))
        elif isinstance(node, ast.ClassDef):
            texts.append(node.name)
    return texts


@functools.lru_cache(maxsize=1)
def tool_vocabulary() -> frozenset[str]:
    """Casefolded tokens of the tool's own text: string and number literals of this package (not docstrings),
    class names and module names. A report word found here was written by the tool, not copied from data."""
    words: set[str] = set()
    for s in _tool_literals():
        words.update(_public_tokens_of(s))
    return frozenset(words)


@functools.lru_cache(maxsize=1)
def tool_phrases() -> frozenset[tuple[str, ...]]:
    """Token sequences (2-6 words) inside one literal of the tool's own text: the tool wrote them."""
    out: set[tuple[str, ...]] = set()
    for s in _tool_literals():
        if " " in s:
            out.update(_phrases_of(s))
    return frozenset(out)


def _public_tokens_of(s: str) -> set[str]:
    s = s.translate(_PUNCT_FOLD)
    out = {t.casefold() for t in WORD_RE.findall(s)}
    for t in list(out):
        out.update(word_parts(t))
    out.update(t.strip(".:*!?#=").casefold() for t in TOKEN_RE.findall(s))
    out.update(t.strip(".'-").casefold() for t in LOCAL_RE.findall(s))
    out.discard("")
    return out


def _phrases_of(s: str) -> set[tuple[str, ...]]:
    toks = WORD_RE.findall(fold(s))
    out = set()
    for i in range(len(toks)):
        for j in range(i + 2, min(len(toks), i + NGRAM) + 1):
            out.add(tuple(toks[i:j]))
    return out


def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _numbers_only(obj: Any) -> list[str]:
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            out.append(str(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return out


@dataclass
class Allow:
    """Public tokens and token sequences that are never a leak (see the module docstring)."""

    tokens: set[str] = field(default_factory=set)
    phrases: set[tuple[str, ...]] = field(default_factory=set)


def build_allow(run: Any, payload: dict[str, bytes] | None = None, broken: Unverified | None = None) -> Allow:
    """The public tokens and token sequences of the run (module docstring). ``broken`` (R6-04): when given, an
    internal error while one file's public text is collected marks that file unverified instead of ending the scan
    (its public text is then not excused: stricter for every other file); without it, the error propagates."""
    from .headers import screen_of

    screen = screen_of(getattr(run, "messages", []))
    flagged = screen.texts if screen is not None else frozenset()
    allow = Allow(tokens=set(tool_vocabulary()), phrases=set(tool_phrases()))
    texts: list[str] = []
    for attr in ("input", "out_dir", "report_dir"):
        v = getattr(run, attr, None)
        if v is not None:
            texts.append(str(v))
            texts.append(Path(v).as_posix())
    texts.append(str(getattr(run, "seed", "")))
    texts.extend(str(v) for v in (getattr(run, "docs", None) or {}).values())
    for s in getattr(run, "skipped", []):
        texts.append(s.rel_path)
    if payload and "manifest.json" in payload:
        try:
            man = json.loads(payload["manifest.json"].decode("utf-8"))
            for key in ("versions", "settings", "seed", "created", "input", "out_dir", "report_dir", "tool"):
                texts.append(_json_text(man.get(key)))
        except (ValueError, UnicodeDecodeError, AttributeError):
            pass
    phrase_sources: list[str] = []
    for fr in getattr(run, "files", []):
        file_texts: list[str] = []
        file_phrases: list[str] = []
        try:
            _public_texts_of(fr, flagged, file_texts, file_phrases)
        except Exception as exc:  # noqa: BLE001 - one file's public text never stops the others' scan
            if broken is None:
                raise
            broken.add(fr.rel_path, exc)
            continue
        texts += file_texts
        phrase_sources += file_phrases
    # the numbers of messages whose text is built from counts only (a count may equal a dense numeric ID)
    messages = list(getattr(run, "messages", []))
    for fr in getattr(run, "files", []):
        messages += fr.all_messages()
    for m in messages:
        if m.code in COUNT_ONLY_CODES:
            texts += [t for t in WORD_RE.findall(m.text) if t.isdigit()]
    for s in texts:
        allow.tokens.update(_public_tokens_of(s))
    for s in phrase_sources:
        allow.phrases.update(_phrases_of(s))
    if screen is not None:  # D34: what the index holds for a flagged name is never excused
        for f in screen.flags:
            words, seqs, exact = f.terms()
            allow.tokens.difference_update(words)
            allow.tokens.difference_update(e.casefold() for e in exact)
            allow.phrases.difference_update(seqs)
    return allow


def _public_texts_of(fr: FileResult, flagged: AbstractSet[str], texts: list[str], phrase_sources: list[str]) -> None:
    """The public text of one file (:func:`build_allow`) into ``texts`` (tokens) and ``phrase_sources`` (sequences)."""
    texts += [fr.rel_path, fr.out_rel_path, fr.format]
    if fr.layout is not None:
        texts.append(_json_text(fr.layout.to_public()))
    if fr.out_layout is not None:
        texts.append(_json_text(fr.out_layout.to_public()))
    for priv in fr.privacy:
        texts.append(_json_text(priv))
    texts += [" ".join(map(str, d or [])) for d in fr.source_dtypes + fr.out_dtypes]
    for t_i, tp in enumerate(fr.tables):
        if tp.name and tp.name not in flagged:  # a flagged name (D34) is never public
            texts.append(tp.name)
            phrase_sources.append(tp.name)
        for h in tp.header:
            if h not in flagged:
                texts.append(h)
                phrase_sources.append(h)
        names = [c.name for c in tp.columns]
        texts += _numbers_only(tp.entity.to_public(names)) + _numbers_only(tp.structure.to_public(names))
        syn = fr.synthetic_stats[t_i] if t_i < len(fr.synthetic_stats) else {}
        fid = fr.fidelity[t_i] if t_i < len(fr.fidelity) else {}
        ids = realids.run_ids(tp)
        for p in tp.columns:
            if p.name not in flagged:
                texts.append(p.name)
            kind = sensitive_kind(p)
            # R9-01: a real value that is a real person ID of the run is never excused, whatever wrote it
            id_units = realids.id_units(p, ids)
            # real values a column writes verbatim are public only when the copy gate cleared them (one source
            # of truth with detect.text.copy_gate, R4): a path that skipped the gate is never excused
            public = kind is None and (copy_cleared(p) or not copies_real_values(p)) and not id_units
            if p.model is not None:
                na_pseudo = na_pseudonymized(p)  # R8-05: a pseudonymised --na-value token is never reproduced
                texts += [str(k) for k in p.model.missing_counts if k is not None and str(k).strip() not in na_pseudo
                          and str(k).strip() not in id_units]
                if p.type in LABEL_SAFE_TYPES and copy_cleared(p):  # labels reproduced in the output (D21)
                    # exactly the levels the D35 allowlist kept: a pseudonymised one is never reproduced
                    pseudo = pseudonymized(p)
                    levels = [str(k) for k in p.model.nonmissing_counts if k is not None and str(k).strip()
                              not in pseudo and str(k).strip() not in id_units]
                    texts += levels
                    phrase_sources += levels
            public_params = {k: v for k, v in p.params.items() if not str(k).startswith("_")}
            for st in (p.stats, syn.get(p.index)):
                if not st:
                    continue
                if public:
                    texts.append(_json_text(st))
                else:  # a sensitive or uncleared column: value-free statistics, and the counts of the others
                    texts.append(_json_text({k: v for k, v in st.items() if k in _VALUE_FREE_STATS}))
                    texts += _numbers_only({k: v for k, v in st.items() if k not in VALUE_REVEALING_KEYS})
            if public:
                texts.append(_json_text(public_params))
            else:
                texts += _numbers_only({k: v for k, v in public_params.items() if k not in _VALUE_PARAMS})
            if p.index in fid:
                texts.append(_json_text(fid[p.index]))


# --------------------------------------------------------------------------- strict scan (report payload, docs)
@dataclass
class Hit:
    origins: frozenset[int]
    rule: str
    start: int
    end: int


class Scanner:
    """Strict matcher over report/doc text with the public-token allowlist; also redacts."""

    def __init__(self, index: SensitiveIndex, allow: Allow) -> None:
        self.index = index
        self.allow = allow
        self.phrase_first = {L: {k[0] for k in table} for L, table in index.phrases.items()}

    def hits_in_text(self, text: str) -> list[Hit]:
        idx, allow = self.index, self.allow
        hits: list[Hit] = []
        text = text.translate(_PUNCT_FOLD)  # same length: spans stay valid for the original text
        words = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
        folded = [w.casefold() for w, _, _ in words]
        for (w, s, e), f in zip(words, folded):
            o = idx.words.get(f)
            if o and f not in allow.tokens:
                hits.append(Hit(frozenset(o), "token", s, e))
            elif idx.words:
                for part in word_parts(f):  # a name inside a joined token (``Smith-Jones``, ``Smith's``)
                    o = idx.words.get(part)
                    if o and part not in allow.tokens:
                        hits.append(Hit(frozenset(o), "token", s, e))
                        break
            o = idx.exact.get(w)
            if o and self._id_counts(text, w, s):
                hits.append(Hit(frozenset(o), "id", s, e))
        for L, table in idx.phrases.items():
            firsts = self.phrase_first[L]
            for i in range(len(folded) - L + 1):
                if folded[i] not in firsts:
                    continue
                key = tuple(folded[i:i + L])
                o = table.get(key)
                if o and key not in allow.phrases:
                    s, e = words[i][1], words[i + L - 1][2]
                    lits = idx.number_literals.get(key)
                    if lits is not None and key not in idx.number_strict and text[s:e].casefold() not in lits:
                        # the digits of a numeric label value in another notation: a statistic of the report
                        # (``0.89``) that only shares digits with the real text ``0,89``; the real text itself
                        # (same separators) is still a hit
                        continue
                    hits.append(Hit(frozenset(o), "phrase", s, e))
        if idx.exact or idx.net:
            for m in TOKEN_RE.finditer(text):
                raw = m.group()
                tok = raw.strip(".:*!?#=")
                if not tok:
                    continue
                s = m.start() + raw.index(tok)
                o = idx.exact.get(tok)
                if o and self._id_counts(text, tok, s):
                    hits.append(Hit(frozenset(o), "id", s, s + len(tok)))
                o = idx.net.get(tok.casefold())
                if o and tok.casefold() not in allow.tokens:
                    hits.append(Hit(frozenset(o), "net", s, s + len(tok)))
        if idx.emails and "@" in text:
            for m in EMAIL_RE.finditer(text):
                o = idx.emails.get(m.group().casefold())
                if o:
                    hits.append(Hit(frozenset(o), "email", m.start(), m.end()))
        if idx.locals:
            for m in LOCAL_RE.finditer(text):
                raw = m.group()
                tok = raw.strip(".'-")
                if not tok:
                    continue
                f = tok.casefold()
                o = idx.locals.get(f)
                if o and f not in allow.tokens:
                    s = m.start() + raw.index(tok)
                    hits.append(Hit(frozenset(o), "email local part", s, s + len(tok)))
        if idx.digits:
            seen: set[tuple[int, int, frozenset[int]]] = set()
            for m in DIGITS_RE.finditer(text):
                if _DECIMAL_NUMBER.fullmatch(m.group()):
                    continue
                for k in _phone_keys(digits_of(m.group())):
                    o = idx.digits.get(k)
                    if o:
                        hit = (m.start(), m.end(), frozenset(o))
                        seen.add(hit)
                        hits.append(Hit(hit[2], "digits", hit[0], hit[1]))
                        break
            # Canonicalization extends PHONE comparison only. IDs, SSNs and
            # numeric counts retain the original digit-token rules above.
            if idx.phone_bases:
                for m in DIGITS_RE.finditer(phone_text(text)):
                    if _DECIMAL_NUMBER.fullmatch(m.group()) and not re.fullmatch(r"\d{7,15}\.0+", m.group()):
                        continue
                    base = phone_base(m.group())
                    if base is None:
                        continue
                    origins = set().union(*(idx.phone_bases.get(k) or set() for k in phone_base_keys(base)))
                    if origins:
                        hit = (m.start(), m.end(), frozenset(origins))
                        if hit not in seen:
                            seen.add(hit)
                            hits.append(Hit(hit[2], "digits", hit[0], hit[1]))
        return hits

    def _id_counts(self, text: str, tok: str, start: int) -> bool:
        if tok.casefold() in self.allow.tokens:
            return False
        # the fractional part of a decimal number (``0.0451``) is not an ID
        if tok.isdigit() and start >= 2 and text[start - 1] == "." and text[start - 2].isdigit():
            return False
        return True

    # ---------------------------------------------------------------- per file kind
    def hits_in_json(self, obj: Any) -> list[Hit]:
        hits: list[Hit] = []
        for s in _json_strings(obj):
            hits.extend(self.hits_in_text(s))
        return hits

    def hits_in_bytes(self, name: str, data: bytes) -> list[Hit]:
        text = _decode(data)
        if name.lower().endswith(".json"):
            try:
                return self.hits_in_json(json.loads(text))
            except ValueError:
                pass
        return self.hits_in_text(text)

    def redact_text(self, text: str) -> str:
        try:
            hits = self.hits_in_text(text)
        except Exception:  # noqa: BLE001 - text the matcher cannot read is never printed (R6-04: fail safe)
            return REDACTED
        if not hits:
            return text
        spans = sorted((h.start, h.end) for h in hits)
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out, pos = [], 0
        for s, e in merged:
            out.append(text[pos:s])
            out.append(REDACTED)
            pos = e
        out.append(text[pos:])
        return "".join(out)

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {self.redact_text(k) if isinstance(k, str) else k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(v) for v in obj)
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if self.redact_text(str(obj)) != str(obj):
                return REDACTED
        return obj

    def redact_bytes(self, name: str, data: bytes) -> bytes:
        text = _decode(data)
        if name.lower().endswith(".json"):
            try:
                obj = json.loads(text)
            except ValueError:
                obj = None
            if obj is not None:
                try:
                    clean = not self.hits_in_json(obj)
                except Exception:  # noqa: BLE001 - redacted string by string below (redact_text fails safe)
                    clean = False
                if clean:
                    return data
                return (json.dumps(self.redact_obj(obj), indent=2, ensure_ascii=False, default=str) + "\n").encode(
                    "utf-8")
        new = self.redact_text(text)
        return data if new == text else new.encode("utf-8")

    def redact_payload(self, payload: dict[str, bytes]) -> dict[str, bytes]:
        return {k: self.redact_bytes(k, v) for k, v in payload.items()}


def _json_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _json_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _json_strings(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield str(obj)


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


# --------------------------------------------------------------------------- data files
@dataclass
class _DataHit:
    origins: set[int]
    rule: str
    column: str | None
    count: int


class DataScanner:
    """Column-aware scan of re-read synthetic tables, and distinctive patterns in any text."""

    def __init__(self, index: SensitiveIndex, allow: Allow) -> None:
        self.index = index
        self.allow = allow
        self.vocab = index.data_vocab

    def scan_column(self, fr: FileResult, t_i: int, p: ColumnProfile, values: np.ndarray) -> list[_DataHit]:
        vals = _distinct_strings(values)
        if not vals:
            return []
        idx = self.index
        hits: list[_DataHit] = []
        ct = idx.columns.get((fr.rel_path, t_i, p.index))
        if ct is not None:
            n = _same_column_hits(ct.kind, ct.real, _item_strings(p, vals))
            if n:
                hits.append(_DataHit({ct.origin}, "same column", p.name, n))
        key = idx.column_namespace.get((fr.rel_path, t_i, p.index))
        if key is not None:
            ns = idx.namespaces.get(key)
            if ns is not None and not ns.identity:
                # D9: never equal to ANY real ID of the run - its own namespace and every other one (a synthetic
                # student ID must not be a real staff ID); kept row numbers 1..n (identity) are not person IDs.
                # R5-02: nor holding a real ID's e-mail local part or username
                n, origins = idx.real_id_hits({v.strip() for v in vals}, ns.prefix)
                if n:
                    hits.append(_DataHit(origins, "ID namespace", p.name, n))
                # R5-02: an e-mail address used as an ID is never a real address of the run (any column)
                if idx.emails:
                    origins, n = set(), 0
                    for e in {fold(v.strip()) for v in vals if "@" in v and _id_local(v) is not None}:
                        o = idx.emails.get(e)
                        if o:
                            origins |= o
                            n += 1
                    if n:
                        hits.append(_DataHit(origins, "email", p.name, n))
        if key is not None:
            # R8-03: nor is an ID, or has the local part or username of, a real e-mail address held anywhere in the
            # run (an excluded column, a pseudonymised label, a title ...)
            ns = idx.namespaces.get(key)
            n, origins = idx.user_hits(p, vals, ns.prefix if ns is not None else None)
            if n:
                hits.append(_DataHit(origins, "real e-mail username", p.name, n))
        cross = cross_column_values(p, vals)
        if cross:
            for origins, rule, n in self.distinctive(cross, known_phone=_known_phone(p)):
                hits.append(_DataHit(origins, rule, p.name, n))
        # R7-01, R7-02: no output text equals a real text of any column of the run that the run never writes; R9-03:
        # nor does a synthetic ID
        n, origins = idx.text_hits(p, cross or id_texts(p, vals))
        if n:
            hits.append(_DataHit(origins, "real text", p.name, n))
        # R9-01: no column writes a real value of its own that is a real person ID of the run (never excused)
        kept = realids.kept_hits(p, vals, idx.real_ids)
        if kept:
            keys = {k for k in (idx.real_ids.key_of_text(v) for v in kept) if k is not None} if idx.real_ids else set()
            _, origins = idx.real_id_hits(keys)
            hits.append(_DataHit(origins, "real person ID", p.name, len(kept)))
        return hits

    def distinctive(self, values: list[str], joined: str | None = None, *,
                    known_phone: bool = False) -> list[tuple[set[int], str, int]]:
        """Run-wide distinctive patterns: full emails, phones (>= 10 digits), SSNs, free-text/address phrases."""
        idx = self.index
        out: list[tuple[set[int], str, int]] = []
        if joined is None:
            joined = "\n".join(values)
        joined = joined.translate(_PUNCT_FOLD)
        if idx.emails and "@" in joined:
            found = {m.casefold() for m in EMAIL_RE.findall(joined)}
            origins: set[int] = set()
            n = 0
            for e in found:
                o = idx.emails.get(e)
                if o:
                    origins |= o
                    n += 1
            if n:
                out.append((origins, "email", n))
        if idx.phone_bases:
            origins, n = set(), 0
            # Keep cells separate: a PHONE column may hold local seven-digit values.
            # Stale files and metadata supply only their already-joined text view.
            for value in values if values else [joined]:
                for base in semantic_phones(value, known=known_phone, layouts=idx.phone_layouts):
                    found: set[int] = set()
                    for k in phone_base_keys(base):
                        found |= idx.phone_bases.get(k) or set()
                    if found:
                        origins |= found
                        n += 1
            if n:
                out.append((origins, "phone digits", n))
        elif idx.phone_digits:
            # phone-SHAPED tokens only (privacy.digits): not digits glued across separators into another
            # layout, not a window inside a longer digit run
            origins, n = set(), 0
            for d in phone_numbers(joined, idx.phone_layouts):
                for k in phone_keys(d):
                    o = idx.phone_digits.get(k)
                    if o:
                        origins |= o
                        n += 1
                        break
            if n:
                out.append((origins, "phone digits", n))
        if idx.ssn_digits:
            origins, n = set(), 0
            for value in values if known_phone else [joined]:
                # A declared phone's whole base is never a source SSN, even
                # when its layout is bare digits rather than the SSN shape.
                base = phone_base(value) if known_phone else None
                o = idx.ssn_digits.get(base[0]) if base is not None else None
                if o:
                    origins |= o
                    n += 1
                    continue
                for m in SSN_RE.finditer(value.translate(_PUNCT_FOLD) if known_phone else value):
                    o = idx.ssn_digits.get(digits_of(m.group()))
                    if o:
                        origins |= o
                        n += 1
            if n:
                out.append((origins, "SSN", n))
        if idx.data_phrases:
            folded = joined.casefold()
            present = set(WORD_RE.findall(folded)) & self.vocab
            # only phrases whose every token occurs in this text can match
            cands = {L: {k for k in table if present.issuperset(k) and k not in self.allow.phrases}
                     for L, table in idx.data_phrases.items()} if present else {}
            cands = {L: c for L, c in cands.items() if c}
            if cands:
                firsts = {L: {k[0] for k in c} for L, c in cands.items()}
                origins, n = set(), 0
                for line in folded.split("\n"):
                    toks = WORD_RE.findall(line)
                    for L, c in cands.items():
                        fs = firsts[L]
                        for i in range(len(toks) - L + 1):
                            if toks[i] in fs:
                                key = tuple(toks[i:i + L])
                                if key in c:
                                    origins |= idx.data_phrases[L].get(key) or set()
                                    n += 1
                if n:
                    out.append((origins, "text sequence", n))
        return out


def cross_column_values(p: ColumnProfile, values: list[str]) -> list[str]:
    """The values of an output column that the run-wide distinctive patterns apply to.

    The patterns (full emails, phone digits, SSNs, free-text/address sequences) look for a real value of
    ANOTHER column copied into this one, so they apply to columns whose generator emits text. They do not
    apply to:

    * ID and code columns (``sensitive_kind`` ID/CODE): their values are drawn from the column's own shape by
      the ID registry or a mask and checked against their own namespace / column; a 10-digit synthetic state
      ID that happens to equal a real phone's digits is a coincidence, not a copy;
    * numeric and date columns: interpolated numbers and dates, never a copied string (``2024-03-15 10:30``
      is not the phone number 202-403-1510);
    * the real levels of a level-generated column (factor, org code, discrete integer ...): they are the
      source's own labels, reproduced by design (D21); any other value of such a column is still checked.

    Numbers, dates and real levels are excused only for a column whose real values passed the copy gate
    (``detect.text.copy_cleared``: one source of truth with the gate, R4); a column that writes real values
    without that clearance has every value checked.
    """
    if not values:
        return values
    if sensitive_kind(p) in (ID, CODE) or p.type is ColumnType.IDENTIFIER:
        return []
    cleared = copy_cleared(p) or not copies_real_values(p)
    if TYPE_INFO[p.type].family in ("numeric", "date") and cleared:
        return []
    if is_level_generated(p) and p.model is not None and cleared:
        pseudo = pseudonymized(p)  # D35: a pseudonymised level is never written, so never excused
        real = {str(k) for k in p.model.nonmissing_counts if k is not None and str(k).strip() not in pseudo}
        na_pseudo = na_pseudonymized(p)  # R8-05: a pseudonymised --na-value token is never written, never excused
        real |= {str(k) for k in p.model.missing_counts if k is not None and str(k).strip() not in na_pseudo}
        return [v for v in values if v not in real]
    return values


def _distinct_strings(values: np.ndarray) -> list[str]:
    if len(values) == 0:
        return []
    uniq = codes_of(values)[1]
    return [v for v in uniq.tolist() if isinstance(v, str) and v.strip()]


def text_view(path: Path) -> str:
    """Readable text of any out-dir file: decoded text, decompressed streams, zip (xlsx) XML members."""
    try:
        size = path.stat().st_size
        if size > _MAX_TEXT_VIEW:
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    return _bytes_view(data)


def _bytes_view(data: bytes, depth: int = 0) -> str:
    if depth > 2:
        return ""
    try:
        if data.startswith(b"PK\x03\x04"):
            parts = []
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist()[:2000]:
                    if info.file_size > _MAX_TEXT_VIEW:
                        continue
                    member = zf.read(info)
                    text = _bytes_view(member, depth + 1)
                    if info.filename.lower().endswith((".xml", ".rels", ".vml")):
                        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
                    parts.append(text)
            return "\n".join(parts)
        if data.startswith(b"\x1f\x8b"):
            return _bytes_view(gzip.decompress(data), depth + 1)
        if data.startswith(b"BZh"):
            return _bytes_view(bz2.decompress(data), depth + 1)
        if data.startswith(b"\xfd7zXZ"):
            return _bytes_view(lzma.decompress(data), depth + 1)
    except (OSError, EOFError, zipfile.BadZipFile, lzma.LZMAError, ValueError):
        pass
    return _decode(data)


_XML_TEXT = re.compile(r">([^<]+)<")
_XML_FORMAT_CODE = re.compile(r"\bformatCode=\"([^\"]*)\"")


def xlsx_text_view(path: Path, covered: set[str] = frozenset()) -> str:  # type: ignore[assignment]
    """Text of an .xlsx output that the cell scan does not see: titles, document properties, comments, notes.

    Every XML text node of the workbook is kept unless its stripped text is one of ``covered`` (the cell
    values already judged column by column with their own rules), so a synthetic ID cell is not re-checked
    against the run-wide patterns of other columns. Non-XML members are read with :func:`text_view`'s rules.
    """
    try:
        if path.stat().st_size > _MAX_TEXT_VIEW:
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    if not data.startswith(b"PK\x03\x04"):
        return _bytes_view(data)
    parts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist()[:2000]:
                if info.file_size > _MAX_TEXT_VIEW:
                    continue
                member = zf.read(info)
                if not info.filename.lower().endswith((".xml", ".rels", ".vml")):
                    parts.append(_bytes_view(member, 1))
                    continue
                xml = _decode(member)
                for m in _XML_TEXT.finditer(xml):
                    chunk = html.unescape(m.group(1))
                    if chunk.strip() and chunk.strip() not in covered:
                        parts.append(chunk)
                # number formats live in an attribute (styles.xml): their literal text is masked by the writer
                parts.extend(html.unescape(m.group(1)) for m in _XML_FORMAT_CODE.finditer(xml))
    except (OSError, EOFError, zipfile.BadZipFile, ValueError):
        return _bytes_view(data)
    return "\n".join(parts)


def hidden_label_text(rr: Any) -> str:
    """Label text of a re-read output that no cell shows: Arrow dictionary entries and Stata value labels that
    no row holds, Stata variable labels and the data label (R2-03).

    The writers never write such text (an unused label was never checked), so this is the last line of
    defence: the run-wide distinctive patterns are applied to it like other file metadata.
    """
    parts: list[str] = []
    for t in getattr(rr, "tables", []) or []:
        for col in getattr(t, "columns", []) or []:
            nat = col.native or {}
            texts: list[Any] = []
            if nat.get("kind") == "dictionary":
                texts = list(nat.get("dictionary_values") or [])
            elif nat.get("family") == "stata" and nat.get("labels"):
                texts = list(nat["labels"].values())
            if texts:
                held = {v for v in col.values.tolist() if isinstance(v, str)}
                parts.extend(str(x) for x in texts if x is not None and str(x) not in held)
            if nat.get("var_label"):
                parts.append(str(nat["var_label"]))
    layout = getattr(rr, "layout", None)
    data_label = ((layout.extra or {}) if layout is not None else {}).get("_data_label")
    if data_label:
        parts.append(str(data_label))
    return "\n".join(parts)


def _arrow_metadata_view(path: Path, fmt: str) -> str:
    """Schema and field metadata of a Parquet/Feather output (text that the cell scan does not see)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if fmt == "parquet":
            schema = pq.read_schema(path)
        else:
            import pyarrow.feather as pf

            schema = pf.read_table(path).schema
    except Exception:  # noqa: BLE001 - metadata is optional
        return ""
    parts: list[str] = []
    metas = [schema.metadata] + [f.metadata for f in schema]
    for meta in metas:
        for k, v in (meta or {}).items():
            parts.append(k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k))
            parts.append(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
    del pa
    return "\n".join(parts)


# --------------------------------------------------------------------------- run-level message and redaction
@dataclass
class LeakMessage(Message):
    """A leak-scan error. Carries a PRIVATE redactor (never serialised: :meth:`to_public`
    is inherited) that the report builder applies to the rebuilt report and the console."""

    redactor: Any = field(default=None, repr=False, compare=False)

    def redact_payload(self, payload: dict[str, bytes]) -> dict[str, bytes]:
        return self.redactor.redact_payload(payload) if self.redactor is not None else payload

    def redact_text(self, text: str) -> str:
        return self.redactor.redact_text(text) if self.redactor is not None else text

    def redact_obj(self, obj: Any) -> Any:
        return self.redactor.redact_obj(obj) if self.redactor is not None else obj


# --------------------------------------------------------------------------- layer 2: the directory-wide scan
@dataclass
class _Finding:
    where: str  # out-dir or report-dir relative path
    area: str  # 'data' | 'docs' | 'report' | 'other'
    origins: set[int]
    rules: set[str]
    count: int
    columns: set[str] = field(default_factory=set)  # data: output columns holding the values
    result: FileResult | None = None


def scan_outputs(run_result: Any, report_payload: dict[str, bytes]) -> list[Message]:
    """AREA G hook: scan every written output, the out-dir docs and the report for real non-label values.

    Called by the pipeline after every synthetic file and the out-dir docs are
    written and the report has been BUILT but not yet written:
    ``report_payload`` maps each report-dir relative path (``REPORT.md``,
    ``report.json``, ``manifest.json``, ``files/<rel>.md`` ...) to its bytes. The
    out dir is ``run_result.out_dir`` (docs in ``run_result.docs``). Returns
    run-level messages; any 'error' makes the exit code 3, and the pipeline then
    rebuilds the report so it includes them (see the module docstring for the
    rules, what FAILs and the redaction). Returns [] when nothing leaked.

    If the scan itself breaks (an internal error), nothing it should have
    verified is left behind as if it were clean, and nothing it did verify is
    removed (D19, R6-04): an error tied to one file (indexing one of its
    columns, collecting its public text, re-reading its output, scanning its
    report page) FAILs that file only (its output is removed and, when real
    values of it may be missing from the index, its report details are
    withheld); an out-dir doc that cannot be scanned is removed; a file this run
    did not write is reported, never deleted. Only an error no file can be tied
    to (a run-level report page) FAILs every written output. Each case returns a
    run-level ``leak_scan_incomplete`` error (exit 3); the traceback goes to
    ``run.debug`` (stderr with ``-v`` only).
    """
    try:
        return _scan_outputs(run_result, report_payload)
    except Exception as exc:  # noqa: BLE001 - an unverified output is never kept
        return _scan_incomplete(run_result, report_payload, exc)
    finally:
        for fr in run_result.files:
            fr.clear_reread()


def _scan_incomplete(run: Any, payload: dict[str, bytes], exc: BaseException) -> list[Message]:
    """An internal error no file can be tied to (:func:`scan_outputs`): every written output is unverified."""
    from ..report import docs as docs_mod

    place = error_place(exc)
    run.debug.append("leak scan: " + _traceback_text(exc))
    out_dir = Path(run.out_dir) if run.out_dir is not None else None
    fix = "re-run with -v to see the traceback, and report it"
    removed = 0
    for fr in run.files:
        if not fr.written:
            continue
        fr.fail(f"{LEAK_PREFIX}the leak scan did not complete (internal error {place}); "
                "the output could not be verified and was removed", fix, "leak_scan_incomplete")
        if out_dir is not None:
            _remove_output(out_dir, fr.out_rel_path)
        removed += 1
    if out_dir is not None and run.docs:
        try:
            docs_mod.rewrite_docs(run, _min_cell(payload))
        except Exception:  # noqa: BLE001 - unverifiable docs are removed too
            for name in run.docs.values():
                (out_dir / name).unlink(missing_ok=True)
    return [Message("error", "leak_scan_incomplete",
                    f"{LEAK_PREFIX}the leak scan did not complete (internal error {place}); "
                    f"{removed} written file(s) were removed as unverified",
                    fix=fix)]


def _scan_outputs(run_result: Any, report_payload: dict[str, bytes]) -> list[Message]:
    from types import SimpleNamespace

    from .headers import IdKeys, NameScreen, screen_of

    run = run_result
    screen = screen_of(run.messages)
    # D19 (R6-04): an internal error tied to one file (indexing its columns, collecting its public text, re-reading
    # its output, scanning its report page) FAILs that file only; only an error no file can be tied to (a run-level
    # report page) leaves every output unverified (scan_outputs)
    broken = Unverified()
    index = SensitiveIndex.build(run.files, screen, _min_cell(report_payload), broken)
    if index.empty and not broken:
        return []
    if screen is None:  # D34: an out-dir path this run did not write is screened before it is printed
        screen = NameScreen(IdKeys([SimpleNamespace(raw_values=ns.exact, representatives=ns.normalized)
                                    for ns in index.namespaces.values() if not ns.identity and ns.exact]))
    allow = build_allow(run, report_payload, broken)
    scanner = Scanner(index, allow)
    data = DataScanner(index, allow)
    out_dir = Path(run.out_dir) if run.out_dir is not None else None
    findings: list[_Finding] = []
    extra: list[Message] = []  # run-level errors of out-dir files the scan could not verify
    bad_docs: list[str] = []
    known: set[str] = set()
    if out_dir is not None:
        for fr in run.files:
            if not fr.written:
                continue
            known.add(fr.out_rel_path.casefold())
            try:
                findings.extend(_scan_data_file(fr, out_dir, data))
            except Exception as exc:  # noqa: BLE001 - this file is unverified, the others are not
                broken.add(fr.rel_path, exc)
        doc_names = [n for n in (run.docs or {}).values()]
        known.update(n.casefold() for n in doc_names)
        for name in doc_names:
            path = out_dir / name
            if path.is_file():
                try:
                    hits = scanner.hits_in_bytes(name, path.read_bytes())
                except Exception as exc:  # noqa: BLE001 - an unverified doc is removed (_drop_docs)
                    run.debug.append(f"leak scan ({name}): " + _traceback_text(exc))
                    bad_docs.append(name)
                    continue
                if hits:
                    findings.append(_finding_from_hits(name, "docs", hits))
        # an output left from an earlier run whose source FAILed now: column-aware when it can be re-read. It
        # is found at its real path and named by its shown one (a flagged folder or file name is redacted, D34);
        # any other path is screened before it is printed
        stale = {fr.out_rel_on_disk.casefold(): fr for fr in run.files if not fr.written and fr.tables}
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(out_dir).as_posix()
            if rel.casefold() in known:
                continue
            old = stale.get(rel.casefold())
            shown = old.out_rel_path if old is not None else None
            try:
                shown = shown or screen.display_path(rel)
                if old is not None:
                    old_findings = _scan_data_file(old, out_dir, data)
                    if not any("unreadable" in f.rules for f in old_findings):
                        for f in old_findings:
                            findings.append(_Finding(shown, "other", f.origins, f.rules, f.count))
                        continue
                found = data.distinctive([], joined=text_view(path))
            except Exception as exc:  # noqa: BLE001 - not this run's output: never deleted, reported instead
                run.debug.append("leak scan (an out-dir file): " + _traceback_text(exc))
                where = shown or "an out-dir file this run did not write"
                extra.append(Message(
                    "error", "leak_scan_incomplete",
                    f"{LEAK_PREFIX}the out-dir file {where} (not written by this run) could not be verified "
                    f"(internal error {error_place(exc)})",
                    fix=f"delete {where} from the out dir" if shown else "delete the files this run did not write "
                                                                       "from the out dir"))
                continue
            if found:
                findings.append(_Finding(shown, "other", set().union(*(o for o, _, _ in found)),
                                         {r for _, r, _ in found}, sum(n for _, _, n in found)))
    pages = {str(PurePosixPath("files", fr.rel_path)) + suffix: fr for fr in run.files for suffix in (".md", ".json")}
    for rel in sorted(report_payload):
        try:
            hits = scanner.hits_in_bytes(rel, report_payload[rel])
        except Exception as exc:
            fr = pages.get(rel)
            if fr is None:
                raise  # a run-level page shows every file: none of them is verified (scan_outputs)
            broken.add(fr.rel_path, exc, withhold=True)  # the rebuilt page withholds the file's details
            continue
        if hits:
            findings.append(_finding_from_hits(rel, "report", hits))
    messages: list[Message] = []
    if findings or broken:
        messages = _act(run, index, scanner, findings, report_payload, broken)
    return messages + extra + _drop_docs(run, out_dir, bad_docs)


def _traceback_text(exc: BaseException) -> str:
    """The ``-v`` traceback of a scan error: classes and code locations only, never a message (R7-04)."""
    from .. import diagnostics

    return diagnostics.safe_traceback(exc)


def _drop_docs(run: Any, out_dir: Path | None, names: list[str]) -> list[Message]:
    """Remove the out-dir docs the scan could not verify (R6-04): the data files stay, they were verified."""
    if out_dir is None or not names:
        return []
    for name in names:
        (out_dir / name).unlink(missing_ok=True)
    run.docs = {k: v for k, v in (run.docs or {}).items() if v not in names}
    return [Message("error", "leak_scan_incomplete",
                    f"{LEAK_PREFIX}{len(names)} out-dir doc(s) could not be verified and were removed "
                    f"({', '.join(sorted(names))}); the data files were verified",
                    fix="re-run with -v to see the traceback, and report it")]


def _finding_from_hits(where: str, area: str, hits: list[Hit]) -> _Finding:
    origins: set[int] = set()
    for h in hits:
        origins |= h.origins
    return _Finding(where, area, origins, {h.rule for h in hits}, len(hits))


def reread_key(path: Path) -> tuple[int, int, int]:
    """Identity of the written bytes whose private verification parse is kept."""
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _scan_data_file(fr: FileResult, out_dir: Path, data: DataScanner) -> list[_Finding]:
    try:
        return _scan_data_file_inner(fr, out_dir, data)
    finally:
        fr.clear_reread()


def _scan_data_file_inner(fr: FileResult, out_dir: Path, data: DataScanner) -> list[_Finding]:
    from ..config import Settings
    from ..io import registry
    from ..io.registry import ReadRequest
    from .headers import split_name

    path = out_dir / Path(*fr.out_rel_on_disk.split("/"))
    if not path.is_file():
        return []
    finding = _Finding(fr.out_rel_path, "data", set(), set(), 0, result=fr)
    try:
        handler = registry.output_handler(fr.format)
        settings = Settings()
        layout = fr.out_layout or fr.layout
        if layout is not None:
            settings.encoding = layout.encoding
            if layout.text is not None and layout.text.delimiter:
                settings.delimiter = layout.text.delimiter
        rr = fr._reread if fr._reread_key == reread_key(path) else None
        if rr is None:
            rr = handler.read(ReadRequest(path, fr.out_rel_path, handler.key, layout.compression if layout else None,
                                          settings, purpose="selfcheck"))
    except Exception:  # noqa: BLE001 - an output we cannot re-read cannot be verified: conservative FAIL
        finding.rules.add("unreadable")
        return [finding]
    is_xlsx = registry.output_handler(fr.format).key == "xlsx"
    covered: set[str] = set()  # xlsx: cell values already judged column by column
    stem = split_name(PurePosixPath(fr.out_rel_on_disk).name)[0]  # the context of its sheet names (D34 step 3)
    for ot in rr.tables:
        # D34: a written header or sheet name holding a flagged name (never named: the name itself is the value)
        hit: set[int] = set()
        for name in ot.header:
            hit |= data.index.flagged_in(str(name), ot.name or stem)
        hit |= data.index.flagged_in(str(ot.name or ""), stem)
        if hit:
            finding.origins |= hit
            finding.rules.add("header/name")
            finding.count += len(hit)
    for t_i, ot in enumerate(rr.tables):
        tp = fr.tables[t_i] if t_i < len(fr.tables) else None
        out_cols = [p for p in tp.columns if p.role != "excluded"] if tp is not None else []
        if len(out_cols) == len(ot.columns):
            pairs: list[tuple[ColumnProfile | None, Any]] = list(zip(out_cols, ot.columns))
        else:  # fall back to header names; an output column without a profile gets the run-wide patterns
            by_name = {p.name: p for p in out_cols}
            pairs = [(by_name.get(h), c) for h, c in zip(ot.header, ot.columns)]
        for p, oc in pairs:
            if p is None:
                vals = _distinct_strings(oc.values)
                found = [_DataHit(o, r, None, n) for o, r, n in data.distinctive(vals)] if vals else []
                texts = data.index.texts
                hit = texts.hits(vals) if texts and vals else set()
                if hit:  # R7-01, R7-02
                    origins = {data.index.text_origins[j] for j, (_, ks) in enumerate(texts.sources)
                               if not hit.isdisjoint(ks)}
                    found.append(_DataHit(origins, "real text", None, len(hit)))
            else:
                found = data.scan_column(fr, t_i, p, oc.values)
            for h in found:
                finding.origins |= h.origins
                finding.rules.add(h.rule)
                finding.count += h.count
                if h.column:
                    finding.columns.add(h.column)
            if is_xlsx:
                covered.update(v.strip() for v in _distinct_strings(oc.values))
    meta = ""
    if fr.format in ("parquet", "feather", "sas"):
        meta = _arrow_metadata_view(path, "parquet" if fr.format == "sas" else fr.format)
    elif is_xlsx:
        meta = xlsx_text_view(path, covered)
    hidden = hidden_label_text(rr)
    if hidden:
        meta = meta + "\n" + hidden if meta else hidden
    if meta:
        for origins, rule, n in data.distinctive([], joined=meta):
            finding.origins |= origins
            finding.rules.add(rule + " (file metadata)")
            finding.count += n
    return [finding] if finding.rules else []


def _describe(index: SensitiveIndex, origins: set[int]) -> tuple[str, str, Origin | None]:
    """(kinds text, ' from file::col, ...' text, first origin) - column names only, never values (a flagged
    name, D34, by its location)."""
    kinds = sorted({KIND_TEXT[index.origins[o].kind] for o in origins}) or ["sensitive"]
    refs = sorted({index.origins[o].ref() for o in origins})
    src = (f" from {', '.join(refs[:3])}" + (f" and {len(refs) - 3} more" if len(refs) > 3 else "")) if refs else ""
    columns = sorted(o for o in origins if index.origins[o].kind not in (HEADER, LAYOUT))
    first = index.origins[columns[0]] if columns else (index.origins[min(origins)] if origins else None)
    return "/".join(kinds), src, first


def _bug_fix(ref: str | None) -> str:
    if not ref or "::" not in ref or ref.startswith("(input)"):
        return "please report this as a bug"
    return (f"--null {ref} (or --exclude {ref}) keeps the column out until the cause is fixed; please report this "
            "as a bug")


def _act(run: Any, index: SensitiveIndex, scanner: Scanner, findings: list[_Finding],
         payload: dict[str, bytes], broken: Unverified | None = None) -> list[Message]:
    """FAIL and delete the affected outputs (the files with findings and the files the scan could not verify,
    ``broken``: R6-04, D19), rewrite and redact the docs, return the run-level errors."""
    from ..report import docs as docs_mod

    by_rel = {fr.rel_path: fr for fr in run.files}
    report_rel = {str(PurePosixPath("files", fr.rel_path)): fr for fr in run.files}
    failures: dict[str, tuple[FileResult, list[str], str]] = {}
    messages: list[Message] = []
    #: report/doc findings grouped per FAILed file: rel -> [places, origins, rules, count]
    grouped: dict[str, list[Any]] = {}

    def fail(fr: FileResult, reason: str, fix: str) -> None:
        entry = failures.setdefault(fr.rel_path, (fr, [], fix))
        entry[1].append(reason)

    for f in findings:
        kinds, src, _ = _describe(index, f.origins)
        rules = ", ".join(sorted(f.rules))
        if f.area == "data" and f.result is not None:
            if "unreadable" in f.rules:
                fail(f.result, f"{LEAK_PREFIX}the written file could not be re-read to verify it holds no real values",
                     "re-run with -v to see the traceback, and report it")
                continue
            cols = sorted(f.columns)
            col_ref = f"{f.result.rel_path}::{cols[0]}" if cols else None
            fail(f.result,
                 f"{LEAK_PREFIX}the synthetic output held {f.count} real {kinds} value(s){src} "
                 f"(in column(s) {', '.join(repr(c) for c in cols) or 'file metadata'}; rule: {rules}); "
                 "the file was not written", _bug_fix(col_ref))
        elif f.area in ("report", "docs"):
            targets: list[FileResult] = []
            if f.area == "report":
                for suffix in (".md", ".json"):
                    if f.where.endswith(suffix) and f.where[: -len(suffix)] in report_rel:
                        targets = [report_rel[f.where[: -len(suffix)]]]
            if not targets:
                targets = [by_rel[o.file] for o in (index.origins[i] for i in sorted(f.origins)) if o.file in by_rel]
            place = ("report file " if f.area == "report" else "out-dir doc ") + f.where
            for fr in {t.rel_path: t for t in targets}.values():
                g = grouped.setdefault(fr.rel_path, [[], set(), set(), 0])
                g[0].append(place)
                g[1] |= f.origins
                g[2] |= f.rules
                g[3] += f.count
        else:
            messages.append(LeakMessage(
                "error", "leak_found",
                f"{LEAK_PREFIX}the out-dir file {f.where} (not written by this run) holds {f.count} real "
                f"{kinds} value(s){src} (rule: {rules})",
                fix=f"delete {f.where} from the out dir", redactor=scanner))
    for rel, (places, origins, rules, count) in grouped.items():
        kinds, src, first = _describe(index, origins)
        fail(by_rel[rel],
             f"{LEAK_PREFIX}{count} real {kinds} value(s){src} would have appeared in the {', '.join(places)} "
             f"(rule: {', '.join(sorted(rules))}); the occurrences were redacted and the file was not written",
             _bug_fix(first.ref() if first is not None and first.kind not in (HEADER, LAYOUT) else None))
    leaked = len(failures)  # files FAILed by a finding
    unverified: dict[str, str] = {}  # rel -> where the internal error happened
    for rel, (exc, withhold) in (broken.files.items() if broken else ()):
        fr = by_rel.get(rel)
        if fr is None:
            continue
        place = error_place(exc)
        run.debug.append(f"leak scan ({rel}): " + _traceback_text(exc))
        unverified[rel] = place
        if withhold:
            fr.withheld = True  # nothing unverified about it is printed (report, console)
        fail(fr, f"{LEAK_PREFIX}{UNVERIFIED_TEXT} (internal error {place}); its output was not written"
                 + ("; its details are withheld from the report" if withhold else ""),
             "re-run with -v to see the traceback, and report it")
    removed = False
    out_dir = Path(run.out_dir) if run.out_dir is not None else None
    for fr, reasons, fix in failures.values():
        was_written = fr.written
        for k, r in enumerate(reasons):
            fr.fail(r, fix if k == 0 else None,
                    "leak_scan_incomplete" if UNVERIFIED_TEXT in r else "real_values_in_output")
        if was_written and out_dir is not None:
            _remove_output(out_dir, fr.out_rel_path)
            removed = True
    if out_dir is not None and run.docs:
        try:
            if removed:
                docs_mod.rewrite_docs(run, _min_cell(payload))
            for name in run.docs.values():
                path = out_dir / name
                if path.is_file():
                    data = path.read_bytes()
                    new = scanner.redact_bytes(name, data)
                    if new != data:
                        path.write_bytes(new)
        except Exception as exc:  # noqa: BLE001 - docs that cannot be rewritten or redacted are removed
            run.debug.append("leak scan (docs): " + _traceback_text(exc))
            messages.extend(_drop_docs(run, out_dir, list(run.docs.values())))
    if unverified:
        places = sorted(set(unverified.values()))
        messages.append(Message(
            "error", "leak_scan_incomplete",
            f"{LEAK_PREFIX}the leak scan could not verify {len(unverified)} file(s) (internal error "
            f"{', '.join(places[:3])}); they FAILed and were not written; every other file was verified",
            fix="re-run with -v to see the traceback, and report it"))
    if not leaked:
        return messages
    n_hits = sum(f.count for f in findings)
    places = sorted({f.where for f in findings})
    messages.insert(0, LeakMessage(
        "error", "leak_found",
        f"{LEAK_PREFIX}{n_hits} real value(s) of ID/PII/free-text/high-cardinality columns found in "
        f"{len(places)} place(s) ({', '.join(places[:5])}{' ...' if len(places) > 5 else ''}); "
        f"{leaked} file(s) FAILed and were not written; no value is quoted and the report and docs "
        "were redacted",
        fix="see the FAIL reasons of each file; please report this as a bug", redactor=scanner))
    return messages


def _remove_output(out_dir: Path, rel: str) -> None:
    path = out_dir / Path(*rel.split("/"))
    path.unlink(missing_ok=True)
    parent = path.parent
    root = out_dir.resolve()
    while parent.resolve() != root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def _min_cell(payload: dict[str, bytes]) -> int:
    try:
        man = json.loads(payload["manifest.json"].decode("utf-8"))
        return int(man["settings"]["min_cell"])
    except (KeyError, ValueError, TypeError, UnicodeDecodeError):
        return 10


__all__ = ["scan_table", "scan_outputs", "sensitive_kind", "sensitive_columns", "SensitiveIndex", "Scanner",
           "DataScanner", "LeakMessage", "LEAK_PREFIX", "build_allow", "tool_vocabulary", "plan_digit_keys",
           "digit_terms", "DigitTerms", "plan_real_texts", "real_texts_of", "run_texts", "cross_text_values",
           "dropped_texts", "outside_texts", "outside_digit_terms", "id_user_hits"]
