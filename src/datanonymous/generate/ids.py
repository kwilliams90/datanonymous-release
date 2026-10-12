"""Synthetic ID values and the run-level ID registry (FOUNDATION core; AREA D extends).

Design D8/D9/section 3 #13:

* One :class:`IdNamespace` per registry key (``profile.entity.id_namespace``):
  every real ID seen in any file of the run (D8 key: case, punctuation and
  leading zeros normalised) maps to exactly one synthetic ID, so the same real
  ID gets the same synthetic ID in every file and keeps its exact row count
  (the synthetic ID is written at the real row positions). Every relationship
  between ID columns (1:1 aliases, student -> family nesting, cross-file
  coverage and orphan rates) is therefore kept row by row. The mapping lives
  only in memory and is never written.
* Identity (D9 "sequential 1..n row numbers stay 1..n") is decided per
  namespace: the real IDs are kept only when every key is an integer, the keys
  are exactly 1..K, and at least one member column is a *row-number column*
  (unique values that are exactly 1..K). Child files whose IDs reference those
  row numbers keep the same values (D8 consistency). Repeated 1..k values alone
  (a long-format person key) are remapped. :attr:`IdNamespace.identity` tells
  the privacy scan which columns legitimately hold real values.
* Same shape, never real (D9). Keys whose representative is all digits (up to
  18 digits) are *digit IDs*; every other key is a *patterned ID*. Both kinds
  can share a namespace and never collide. "Never real" is run-wide: a
  namespace's synthetic IDs also avoid the real IDs (D8 keys) of every other
  namespace of the run (a synthetic ``student_id`` is never a real
  ``staff_id``), except the row numbers of identity namespaces; the mappings
  themselves stay separate.

  - Digit IDs keep the number of significant digits of the key they replace
    (so ``0012345`` -> ``0067890`` keeps its two leading zeros and ``12345`` ->
    ``67890`` its five digits), excluding every real integer of the namespace.
  - **Per raw form** (AREA D): a column holding a zero-padded value
    (``0012345``) writes every digit value zero-padded to that value's own raw
    width; a column without padding writes the plain number. The same ID can
    therefore appear as ``0012345`` in one file and ``12345`` in another (D8:
    the same entity) and each keeps its own width.
  - When a width has too few non-real values (dense keys such as 1..2000 with
    one gap), keys that only ever appear zero-padded move first to another
    number of significant digits that still fits their padded width (invisible
    in the output); the remaining shortfall of **unpadded** integers spills to
    the nearest other width inside the observed range (wider first, then
    narrower), then past the widest observed width when the whole observed range
    is full (a warehouse's dense 1..N key; harden 2, A2-6: D9 lets an unpadded
    integer take another digit count, it never FAILs for want of one), and the
    number of IDs whose visible width changed is reported
    (:attr:`IdNamespace.notes`). A key written zero-padded in any column never
    takes more digits than its padding holds (D9, user decision 2026-10-05:
    a plain integer ID getting longer has no real consequence, a fixed-width
    one does). When the allowed widths are too small, :class:`IdSpaceError`
    fails the files using it.
  - Synthetic values are assigned **at random** within each width: an
    order-preserving mapping could be rebuilt from the real ID list by rank
    (and puts each synthetic ID next to its real one), which D8 forbids ("never
    written"). A file sorted by ID is re-sorted on the synthetic IDs by the
    sort-order step (``profile.structure``, section 6).
  - Patterned IDs keep each real value's character-class mask and the common
    letter prefix (compared case-insensitively) of **every** patterned ID of
    the namespace (``detect.ids.kept_prefix``, R5-02). A prefix of two or more
    letters is kept only when at least ``--min-cell`` distinct IDs share it (a
    code of the ID scheme, ``STU``; a username stem shared by one person's few
    accounts is not), it is not name-like (``detect.ids.name_like``:
    ``jsmith``) and it is no real ID of the run; otherwise only its first
    letter is kept (``S``). The other letters are drawn at random. UUIDs and
    32/40/64-hex hashes keep the **hex** character class (AREA D). When random
    draws keep hitting real or used values (a nearly full mask space such as
    ``T00``-``T99``), the free values of the mask are enumerated, so the run
    fails only when the space is really too small (D9).
  - **E-mail addresses** used as IDs (``detect.ids.is_email_id``; R5-02) get
    e-mail addresses built like an EMAIL column's (D24, ``generate.pii``): the
    local part's letter runs become Faker name words of the same length, its
    digits random digits, its separators are kept; each real domain maps to a
    reserved domain (example.org, example.com, example.net, then
    ``dN.example.org``; never a real domain of the namespace) by its rows. The
    mapping stays 1:1 and consistent across files like any other ID.
  - **Usernames** (R5-02): no synthetic ID has the username of a real ID of the
    run (``detect.ids.username_stem``: the letters of an e-mail local part or of
    the value without the kept prefix, ``jsmith23`` -> ``jsmith``), and no
    synthetic e-mail local part is a real e-mail local part or a real ID of the
    run (whole, case-insensitive, or by its D8 key), nor the local part or the
    username of a real address held anywhere in the run (``avoid_emails`` of
    :meth:`IdRegistry.build`: every address of the run's real-text registry,
    R8-03: EMAIL columns, excluded and nulled columns, labels, free text,
    titles; :class:`RealUsers`).
  - **Real texts** (R9-03): no synthetic ID equals a text of the run's
    real-text registry (``avoid_texts`` of :meth:`IdRegistry.build`: a
    pseudonymised label, a name, free text, a value of an excluded or nulled
    column, a title ..., and the texts the run writes verbatim), case-folded
    (:meth:`RealUsers.text_taken`). Only IDs with a letter can be one (the
    registry holds texts with a letter); the leak scan checks every ID column
    against the registry.
  - ``--id-check luhn`` (design section 4, opt-in; AREA D): in an all-digit
    namespace whose real IDs all carry a valid Luhn check digit, the synthetic
    ID is a non-real body plus its recomputed check digit.
* Rendering is per raw form: a synthetic ID takes the case, punctuation and
  surrounding whitespace of the raw value it replaces (``S79521`` and
  ``s79521`` share one mapping but keep their own case). A synthetic e-mail
  address takes the case and punctuation of the raw local part and the case of
  the raw domain (an upper-case domain stays upper-case).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Callable, Iterable

import numpy as np

from ..detect.ids import (
    MIN_USERNAME_LETTERS,
    PREFIX_MIN_IDS,
    analyze_id_shape,
    char_mask,
    id_text,
    integral_float_parts,
    is_email_id,
    is_hex_id,
    kept_prefix,
    luhn_check_digit,
    luhn_valid,
    normalize_id,
    thousands_digits,
    username_stem,
)
from ..errors import IdSpaceError
from ..profile.codes import codes_of, has_nul
from ..privacy.texts import is_text as _is_text
from ..privacy.texts import key as _text_key
from .rng import stream

_ENUMERATE_LIMIT = 2_000_000
_MAX_DIGIT_WIDTH = 18  # wider digit strings are handled as patterns (int64 range)
_UPPER = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", dtype=np.uint8)
_LOWER = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)
_DIGIT = np.frombuffer(b"0123456789", dtype=np.uint8)
_HEX = "0123456789abcdef"
ID_CHECKS = ("luhn",)


@dataclass
class IdNamespace:
    key: str
    members: list[tuple[str, str | None, int, str]] = field(default_factory=list)  # (file, table, col, name)
    representatives: dict[str, str] = field(default_factory=dict)  # PRIVATE: D8 key -> first raw value
    raw_values: set[str] = field(default_factory=set)  # PRIVATE: stripped raw values
    row_number_sizes: list[int] = field(default_factory=list)  # n of each member that is a unique 1..n column
    #: PRIVATE: digit key -> narrowest raw width among its forms in zero-padded columns
    pad_bound: dict[str, int] = field(default_factory=dict)
    #: PRIVATE: digit keys written plainly somewhere (their number of digits is visible)
    fixed_keys: set[str] = field(default_factory=set)
    check: str | None = None  # 'luhn' when the opt-in check digit is applied (--id-check)
    shape: dict[str, Any] = field(default_factory=dict)
    mapping: dict[str, Any] = field(default_factory=dict)  # PRIVATE: D8 key -> canonical synthetic value
    #: PRIVATE: D8 key -> rows holding it (every member column; ranks the e-mail domains, R5-02)
    key_rows: dict[str, int] = field(default_factory=dict)
    #: PRIVATE: D8 keys whose representative is an e-mail address (R5-02); their synthetic IDs are addresses too
    email_keys: set[str] = field(default_factory=set)
    #: PRIVATE: the case-folded letter prefix the patterned synthetic IDs keep (``detect.ids.kept_prefix``)
    prefix: str = ""
    identity: bool = False  # real IDs kept (row numbers 1..n, D9)
    notes: list[str] = field(default_factory=list)  # report-safe notes (counts only)
    error: IdSpaceError | None = None

    def describe(self) -> str:
        """Report-safe description of the member columns (names and files only)."""
        names = sorted({m[3] for m in self.members})
        files = sorted({m[0] for m in self.members})
        cols = ", ".join(repr(n) for n in names) or repr(self.key)
        return f"ID column {cols} ({', '.join(files)})" if files else f"ID column {cols}"

    def fix_hint(self) -> str:
        if not self.members:
            return "--null COL or --exclude COL"
        file, table, _, name = self.members[0]
        ref = f"{file}::{table}::{name}" if table else f"{file}::{name}"
        return (f"if {name!r} holds record numbers rather than person IDs, --force-factor {ref} keeps its real values; "
                f"otherwise --null {ref} or --exclude {ref}")


def column_is_padded(values: list[str]) -> bool:
    """True when a column's stripped digit values include a zero-padded one (``007``)."""
    return any(len(v) > 1 and v[0] == "0" and v.isascii() and v.isdigit() for v in values)


def luhn_eligible(keys: list[str], reps: dict[str, str]) -> bool:
    """``--id-check luhn`` applies: every key is a plain digit ID of >= 2 digits that passes Luhn."""
    if not keys:
        return False
    for k in keys:
        rep = reps[k]
        if not _is_digit_rep(rep) or len(k) < 2 or not luhn_valid(k):
            return False
    return True


class IdRegistry:
    """Run-level registry of ID namespaces (built after pass 1).

    ``id_check`` is the opt-in ``--id-check`` value (``'luhn'`` or None); ``min_cell`` (``--min-cell``) is the
    number of distinct IDs a kept prefix of two or more letters needs (``detect.ids.kept_prefix``).
    """

    def __init__(self, id_check: str | None = None, min_cell: int = PREFIX_MIN_IDS) -> None:
        self.namespaces: dict[str, IdNamespace] = {}
        self.id_check = id_check
        self.min_cell = min_cell

    def register(self, key: str, file_rel: str, table: str | None, col: int, name: str, raw: np.ndarray) -> None:
        ns = self.namespaces.setdefault(key, IdNamespace(key))
        ns.members.append((file_rel, table, col, name))
        codes, uniq = codes_of(np.asarray(raw, dtype=object))
        counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=len(uniq)) if len(uniq) else []
        forms: dict[str, int] = {}  # stripped raw form -> rows (first-seen order)
        for u, c in zip(uniq.tolist(), list(counts)):
            if not isinstance(u, str):  # a native null (factorize turns None into NaN)
                continue
            s = u.strip()
            if s and id_text(s):  # a placeholder without a letter or digit ('-', '?') is no ID (harden 2)
                forms[s] = forms.get(s, 0) + int(c)
        padded = column_is_padded(list(forms))
        keys: dict[str, int] = {}
        all_digits = True
        for s, c in forms.items():
            ns.raw_values.add(s)
            k = normalize_id(s)
            rep = ns.representatives.get(k)
            if rep is None or (not _is_digit_rep(rep) and _is_digit_rep(s)):
                # a digit ID stays a digit ID when a data-entry form of it ('100234?', '100-234') comes first
                ns.representatives[k] = s
            keys[k] = keys.get(k, 0) + c
            ns.key_rows[k] = ns.key_rows.get(k, 0) + c
            plain_digits = s.isascii() and s.isdigit()
            all_digits = all_digits and plain_digits
            if plain_digits and padded:
                ns.pad_bound[k] = min(ns.pad_bound.get(k, len(s)), len(s))
            else:
                ns.fixed_keys.add(k)
        # a row-number column: one row per ID, integers exactly 1..n (D9)
        if keys and all_digits and all(c == 1 for c in keys.values()):
            ints = [int(k) for k in keys]
            if len(ints) >= 2 and min(ints) == 1 and max(ints) == len(ints):  # a lone '1' is no sequence
                ns.row_number_sizes.append(len(ints))

    def build(self, seed: int, avoid_emails: Iterable[str] = (), avoid_ids: Iterable[str] = (),
              avoid_texts: Any = None) -> None:
        """Build every namespace's mapping (see the module docstring).

        ``avoid_emails``: real e-mail addresses held anywhere in the run (the pipeline reads every address of the
        run's real-text registry, R8-03): no synthetic ID takes one of them, one of their local parts or their
        usernames (R5-02).
        ``avoid_ids``: real IDs of ID columns that are not registered (files that FAILed before planning, and
        columns a flag turned from ID columns into something else, R6-02): no synthetic ID equals one (digit IDs
        too, by D8 key), takes its local part or its username.
        ``avoid_texts`` (R9-03): the run's real-text registry (``privacy.texts.RealTexts``): no synthetic ID equals
        one of its texts (:func:`text_taken`: a real text the run does not write, or one it writes verbatim, so a
        file that FAILs later never turns a drawn ID into a copy), whatever column holds it.
        """
        # D9 "never equal to ANY real ID": every namespace's synthetic IDs avoid the real IDs of every other
        # namespace of the run too (a synthetic student ID must not be a real staff ID); mappings stay separate.
        # Row-number namespaces (identity, D9) are not person IDs and keep 1..n: they are not excluded.
        avoid_ids = list(avoid_ids)
        run_keys: set[str] = set()
        live = [ns for ns in self.namespaces.values() if not namespace_is_identity(ns)]
        for ns in live:
            run_keys.update(ns.representatives)
        avoid_keys = {normalize_id(s) for s in (str(v).strip() for v in avoid_ids) if s}
        # R6-02: the unregistered real IDs are real IDs of the run too, digit IDs included
        run_int = _digit_key_array(run_keys | avoid_keys)
        # R5-02: each namespace's kept prefix, then the run's real usernames and e-mail local parts
        for ns in live:
            ns.prefix = kept_prefix(_pattern_reps(ns), run_keys | avoid_keys, self.min_cell)
        users = RealUsers.of(live, avoid_emails, avoid_ids, avoid_texts)
        for key in sorted(self.namespaces):
            ns = self.namespaces[key]
            try:
                build_mapping(ns, stream(seed, "__ids__", key), id_check=self.id_check, run_keys=run_keys,
                              run_int=run_int, min_cell=self.min_cell, users=users, prefix=ns.prefix)
            except IdSpaceError as e:
                ns.error = IdSpaceError(f"{ns.describe()}: {e.message}", fix=ns.fix_hint())

    def render(self, key: str, raw: np.ndarray) -> np.ndarray:
        """Synthetic IDs at the real row positions; blank IDs, and placeholders without a letter or digit (``-``,
        ``?``: ``detect.ids.id_text``, harden 2), are kept verbatim."""
        ns = self.namespaces[key]
        if ns.error is not None:
            raise ns.error
        raw = np.asarray(raw, dtype=object)
        codes, uniques = codes_of(raw)
        if has_nul(raw):
            # The legacy renderer visited exact raw forms. pandas codes can
            # collapse distinct NUL-containing forms, including unmapped IDs.
            positions = {value: i for i, value in enumerate(dict.fromkeys(raw.tolist()))}
            uniques = np.asarray(list(positions), dtype=object)
            codes = np.fromiter((positions[value] for value in raw), dtype=np.int64, count=len(raw))
        padded = column_is_padded([v.strip() for v in uniques.tolist() if v is not None])
        out = np.empty(len(uniques), dtype=object)
        for i, v in enumerate(uniques.tolist()):
            if v is None or not v.strip() or not id_text(v):
                out[i] = v
                continue
            s = v.strip()
            k = normalize_id(s)
            syn = ns.mapping[k]
            if isinstance(syn, (int, np.integer)):
                core = str(int(syn))
                if s.isascii() and s.isdigit():
                    body = core.zfill(len(s)) if padded else core
                elif integral_float_parts(s) is not None:
                    body = core + integral_float_parts(s)[1]  # K-02: an integer stored as a float, 583920.0
                elif thousands_digits(s) is not None:
                    body = f"{int(core):,}"  # harden 2: an integer in a number format, 583,920
                else:
                    n_alnum = sum(ch.isalnum() for ch in s)
                    first = next((ch for ch in s if ch.isalnum()), "")
                    if first == "0" and len(core) < n_alnum:
                        core = core.zfill(n_alnum)
                    body = _conform(core, s) or core
            elif k in ns.email_keys:
                body = _conform_email(syn, s)
            else:
                body = _conform(syn, s) or syn
            lead, trail = v[: len(v) - len(v.lstrip())], v[len(v.rstrip()):]
            out[i] = lead + body + trail
        return out[codes]


def _conform(core: str, form: str) -> str | None:
    """``core``'s characters placed into ``form``'s layout (case, punctuation); None if lengths differ."""
    positions = [i for i, ch in enumerate(form) if ch.isalnum()]
    chars = [ch for ch in core if ch.isalnum()]
    if len(chars) != len(positions):
        return None
    out = list(form)
    for c, i in zip(chars, positions):
        f = form[i]
        if c.isalpha() and f.isalpha():
            c = c.upper() if f.isupper() else c.lower()
        out[i] = c
    return "".join(out)


def _conform_email(syn: str, form: str) -> str:
    """A synthetic e-mail ID in the layout of the raw form ``form`` (R5-02): the local part takes the raw local
    part's case and punctuation (when it has as many letters and digits; else only an all-upper or all-lower
    case), the domain the raw domain's case (an upper-case domain stays upper-case; any other is written in
    lower case)."""
    local, _, domain = syn.rpartition("@")
    raw_local, at, raw_domain = form.rpartition("@")
    if not at:  # a raw form without '@' that shares the address's D8 key: the canonical address
        return syn
    domain = domain.upper() if raw_domain.isupper() else domain.lower()
    shaped = _conform(local, raw_local)
    if shaped is None:
        shaped = local.upper() if raw_local.isupper() else local.lower() if raw_local.islower() else local
    return shaped + "@" + domain


def _is_digit_rep(rep: str) -> bool:
    """A digit ID (up to 18 digits), also when stored as a float (``100234.0``, K-02) or written with thousands
    separators (``100,226``, harden 2)."""
    if rep.isascii() and rep.isdigit():
        return len(rep) <= _MAX_DIGIT_WIDTH
    whole = integral_float_parts(rep)
    if whole is not None:
        return len(whole[0]) <= _MAX_DIGIT_WIDTH
    grouped = thousands_digits(rep)
    return grouped is not None and len(grouped) <= _MAX_DIGIT_WIDTH


def _digit_key_array(keys: Any) -> np.ndarray:
    """Sorted distinct integers of the all-digit D8 ``keys`` (up to 18 digits)."""
    return np.unique(np.fromiter((int(k) for k in keys if k.isascii() and k.isdigit() and len(k) <= _MAX_DIGIT_WIDTH),
                                 dtype=np.int64))


def namespace_is_identity(ns: IdNamespace) -> bool:
    """D9: the namespace's real IDs are row numbers 1..K that are kept (see module docstring)."""
    keys = list(ns.representatives)
    if not keys or not all(_is_digit_rep(ns.representatives[k]) for k in keys):
        return False
    ints = sorted(int(k) for k in keys)
    return ints[0] == 1 and ints[-1] == len(ints) and len(ints) in ns.row_number_sizes


def build_mapping(ns: IdNamespace, rng: np.random.Generator, id_check: str | None = None,
                  run_keys: set[str] | frozenset[str] | None = None, run_int: np.ndarray | None = None,
                  min_cell: int = PREFIX_MIN_IDS, users: "RealUsers | None" = None,
                  prefix: str | None = None) -> None:
    """Fill ``ns.mapping`` / ``ns.identity`` / ``ns.notes`` (see module docstring).

    ``run_keys`` / ``run_int`` (from :meth:`IdRegistry.build`) are the D8 keys and the sorted integer keys of
    every real ID of the run outside row-number namespaces: synthetic IDs avoid them as well as the
    namespace's own real IDs (D9: never equal to any real ID). ``users`` (R5-02) holds the run's real
    usernames and e-mail local parts (the namespace's own when None); ``prefix`` the namespace's kept prefix
    (``detect.ids.kept_prefix`` with ``min_cell`` when None).
    """
    keys = list(ns.representatives)
    ns.shape = analyze_id_shape(sorted(ns.raw_values))
    ns.mapping, ns.identity, ns.notes, ns.check = {}, False, [], None
    if not keys:
        return
    digit_keys = [k for k in keys if _is_digit_rep(ns.representatives[k])]
    pattern_keys = [k for k in keys if not _is_digit_rep(ns.representatives[k])]
    if namespace_is_identity(ns):
        ns.identity = True
        ns.mapping = {k: int(k) for k in digit_keys}
        return
    # R5-02: e-mail addresses get e-mail addresses (D24), the other patterned IDs same-shape masks
    ns.email_keys = {k for k in pattern_keys if "@" in ns.representatives[k] and is_email_id(ns.representatives[k])}
    if ns.email_keys:
        pattern_keys = [k for k in pattern_keys if k not in ns.email_keys]
    real_keys = set(keys)  # every raw value's D8 key is a representatives key
    foreign: set[str] | frozenset[str] = run_keys if run_keys is not None else frozenset()
    if prefix is None:
        prefix = kept_prefix([ns.representatives[k] for k in pattern_keys], real_keys | set(foreign), min_cell)
    ns.prefix = prefix
    if users is None:
        users = RealUsers.of([ns])
    if users.id_keys:  # real IDs of files that FAILed before planning are real IDs of the run too
        foreign = set(foreign) | users.id_keys
    real_int = _digit_key_array(real_keys)
    if run_int is not None and len(run_int):
        real_int = np.union1d(real_int, np.asarray(run_int, dtype=np.int64))
    used: set[str] = set()
    if digit_keys:
        order = sorted(digit_keys, key=int)
        if id_check == "luhn" and not pattern_keys and not ns.email_keys and luhn_eligible(order, ns.representatives):
            ns.check = "luhn"
            bodies = [k[:-1] for k in order]
            # a synthetic body plus its check digit must not be any real ID: exclude every real ID's body
            real_bodies = np.union1d(np.asarray(sorted({int(b) for b in bodies}), dtype=np.int64),
                                     real_int[real_int >= 10] // 10)
            nat = {b: len(b) for b in bodies}
            bound = {b: ns.pad_bound[k] - 1 for b, k in zip(bodies, order)
                     if k not in ns.fixed_keys and k in ns.pad_bound}
            limit = {b: ns.pad_bound[k] - 1 for b, k in zip(bodies, order) if k in ns.pad_bound}
            syn_bodies = allocate_digit_ids(bodies, nat, bound, real_bodies, rng, ns.notes, lowest=1,
                                            pad_limit=limit)
            syn = [int(b) * 10 + luhn_check_digit(str(int(b))) for b in syn_bodies.tolist()]
        else:
            nat = {k: len(k) for k in order}
            bound = {k: ns.pad_bound[k] for k in order if k not in ns.fixed_keys and k in ns.pad_bound}
            limit = {k: ns.pad_bound[k] for k in order if k in ns.pad_bound}
            syn = allocate_digit_ids(order, nat, bound, real_int, rng, ns.notes, pad_limit=limit).tolist()
        ns.mapping.update({k: int(s) for k, s in zip(order, syn)})
        used.update(str(int(s)) for s in syn)
    if pattern_keys:
        _pattern_mapping(ns, pattern_keys, real_keys, used, rng, foreign, prefix, users.stems,
                         users.text_taken if users.texts else None)
    if ns.email_keys:
        _email_mapping(ns, [k for k in keys if k in ns.email_keys], real_keys, used, rng, foreign, users)


# --------------------------------------------------------------------------- R5-02: prefixes, usernames
def _pattern_reps(ns: IdNamespace) -> list[str]:
    """Representatives of the namespace's patterned keys (neither digit IDs nor e-mail addresses)."""
    return [r for r in ns.representatives.values() if not _is_digit_rep(r) and not ("@" in r and is_email_id(r))]


@dataclass(frozen=True)
class RealUsers:
    """PRIVATE: the run's real usernames and e-mail addresses no synthetic ID may take (R5-02).

    ``stems``: :func:`detect.ids.username_stem` of every real ID (with its namespace's kept prefix) and of the
    local part of every real e-mail address of the run; ``locals``: case-folded real e-mail local parts;
    ``local_keys``: their D8 keys; ``emails``: case-folded real e-mail addresses of the run's other columns;
    ``id_keys``: D8 keys of real IDs outside the registry (ID columns of files that FAILed before planning:
    their people are real people too).
    """

    stems: frozenset[str] = frozenset()
    locals: frozenset[str] = frozenset()
    local_keys: frozenset[str] = frozenset()
    emails: frozenset[str] = frozenset()
    id_keys: frozenset[str] = frozenset()
    #: R9-03: the run's real-text registry (``privacy.texts.RealTexts``), or None
    texts: Any = None

    def text_taken(self, value: str) -> bool:
        """R9-03: ``value`` (a candidate synthetic ID) equals a text of the run's real-text registry: a real text the
        run does not write (``privacy.texts.RealTexts.holds``, what the leak scan checks every ID column for) or one
        it writes verbatim (a kept label; a file that FAILs after the IDs are drawn writes it no more)."""
        t = self.texts
        if not t:
            return False
        k = _text_key(value)
        return t.holds_key(k) or (k in t.kept and _is_text(k))

    @classmethod
    def of(cls, namespaces: Iterable[IdNamespace], avoid_emails: Iterable[str] = (),
           avoid_ids: Iterable[str] = (), avoid_texts: Any = None) -> "RealUsers":
        stems: set[str] = set()
        locs: set[str] = set()
        emails: set[str] = set()
        ids: set[str] = set()

        def add(value: str, prefix: str = "") -> None:
            if "@" in value and is_email_id(value):
                locs.add(value.rpartition("@")[0].casefold())
                st = username_stem(value)
            else:
                st = username_stem(value, prefix)
            if st:
                stems.add(st)

        for ns in namespaces:
            for rep in ns.raw_values:  # every raw form (the leak scan reads them all)
                if not _is_digit_rep(rep):
                    add(rep, ns.prefix)
        for v in avoid_emails:
            s = v.strip()
            if "@" in s and is_email_id(s):
                emails.add(s.casefold())
                add(s)
            elif s.count("@") == 1 and s[0] != "@":
                # R8-03: any real address the run's registry holds (one source of truth with the leak scan, whatever
                # its domain looks like): its local part and username too
                emails.add(s.casefold())
                local = s.partition("@")[0].casefold()
                locs.add(local)
                letters = "".join(ch for ch in local if ch.isalpha())
                if len(letters) >= MIN_USERNAME_LETTERS:
                    stems.add(letters)
        for v in avoid_ids:
            s = v.strip()
            if s:
                ids.add(normalize_id(s))
                if not _is_digit_rep(s):
                    add(s)
        return cls(frozenset(stems), frozenset(locs), frozenset(normalize_id(x) for x in locs), frozenset(emails),
                   frozenset(ids), avoid_texts if avoid_texts else None)


def _width_bounds(w: int, lowest: int = 0) -> tuple[int, int]:
    return (lowest if w == 1 else 10 ** (w - 1)), 10**w


def _width_of(v: int) -> int:
    return len(str(int(v)))


def allocate_digit_ids(keys: list[str], nat: dict[str, int], bound: dict[str, int], real: np.ndarray,
                       rng: np.random.Generator, notes: list[str], lowest: int = 0,
                       pad_limit: dict[str, int] | None = None) -> np.ndarray:
    """Distinct non-real synthetic integers for the digit ``keys``, aligned with ``keys``.

    ``nat[k]`` is the key's number of significant digits; ``bound[k]`` (only for
    keys that always appear zero-padded) is the widest number of significant
    digits that still fits the padded width. Every key gets a target width: its
    own when that width has room; on a shortfall, padded-only keys move first to
    another width their padding hides (wider up to their bound, or narrower:
    invisible), then remaining keys to the nearest observed width, and past the
    widest observed width when every observed one is full (visible, noted; a
    dense 1..N key, harden 2 A2-6). ``pad_limit[k]`` (keys written zero-padded in any column) caps that
    visible move: only an unpadded integer may change its number of digits, a
    zero-padded ID keeps its width or the namespace fails (D9, user decision
    2026-10-05). Each key keeps its target width exactly (a plain key written inside a
    padded column keeps its own raw width). Values are drawn per width excluding
    every real integer and assigned **at random** within the width, so the
    mapping cannot be rebuilt from the real ID list by rank (D8: the mapping is
    never recoverable); a sorted source is re-sorted by ``profile.structure``'s
    sort-order step. ``lowest`` is the smallest allowed 1-digit value (0, or 1
    for Luhn bodies); 0 is used only when the other 1-digit values do not
    suffice, since an all-zero ID (``0`` / ``0000000``) reads as a missing one
    (``if student_id:`` drops it). Raises :class:`IdSpaceError` when no
    allocation exists.
    """
    if not keys:
        return np.empty(0, dtype=np.int64)
    pad_limit = pad_limit or {}
    widths_needed = sorted({nat[k] for k in keys})
    max_w = max([*widths_needed, *bound.values()]) if bound else widths_needed[-1]
    max_w = min(max_w, _MAX_DIGIT_WIDTH)
    # D9 (harden 2, A2-6): an unpadded integer may also take more digits than the widest real ID when every observed
    # width is full (a warehouse's dense 1..N student key, Stu = 101..650): it is noted, never a FAIL
    top = _MAX_DIGIT_WIDTH
    cap: dict[int, int] = {}
    for w in range(1, top + 1):
        lo, hi = _width_bounds(w, lowest)
        cap[w] = (hi - lo) - int(np.searchsorted(real, hi) - np.searchsorted(real, lo))
    groups: dict[int, list[str]] = {w: [] for w in range(1, top + 1)}
    for k in keys:
        groups[nat[k]].append(k)
    spare = {w: cap[w] - len(groups[w]) for w in groups}
    order = {k: i for i, k in enumerate(keys)}
    lo_w, hi_w = widths_needed[0], widths_needed[-1]

    def move(w: int, t: int, chosen: list[str]) -> None:
        gone = set(chosen)
        groups[w] = [k for k in groups[w] if k not in gone]
        groups[t].extend(chosen)
        spare[w] += len(chosen)
        spare[t] -= len(chosen)

    for w in widths_needed:
        deficit = -spare[w]
        if deficit <= 0:
            continue
        # padded-only keys: another width their padding hides - wider (up to their bound) first, then narrower
        for t in [*range(w + 1, max_w + 1), *range(w - 1, 0, -1)]:
            if not deficit:
                break
            if spare[t] <= 0:
                continue
            elig = [k for k in groups[w] if k in bound and (t < w or bound[k] >= t)]
            # wider: the tightest sufficient bounds first, so wider bounds stay free for wider targets
            elig.sort(key=lambda k: (bound[k], order[k]) if t > w else (-bound[k], order[k]))
            chosen = elig[:min(deficit, spare[t])]
            if chosen:
                move(w, t, chosen)
                deficit -= len(chosen)
        # the nearest observed width (wider first, then narrower), then a width past the widest observed one -
        # visible. D9 (user decision 2026-10-05): only an unpadded integer may change its number of digits; a key
        # written zero-padded anywhere keeps its padded width, so it may only take a width up to that padding
        for t in [*range(w + 1, hi_w + 1), *range(w - 1, lo_w - 1, -1), *range(hi_w + 1, top + 1)]:
            if not deficit:
                break
            if spare[t] <= 0:
                continue
            rest = sorted((k for k in groups[w] if t <= pad_limit.get(k, t)), key=order.__getitem__)
            take = min(deficit, spare[t], len(rest))
            if not take:
                continue
            move(w, t, rest[len(rest) - take:] if t > w else rest[:take])
            deficit -= take
        if deficit:
            free = sum(max(cap[x], 0) for x in range(lo_w, hi_w + 1))
            padded = any(k in pad_limit for k in groups[w])
            raise IdSpaceError(
                f"{len(keys)} synthetic IDs of {lo_w}-{hi_w} digits cannot avoid all real IDs "
                f"(only {free} free values" + ("; zero-padded IDs keep their width" if padded else "") + ")")
    value: dict[str, int] = {}
    changed = 0
    for w in sorted(groups):
        members = groups[w]
        if not members:
            continue
        members.sort(key=order.__getitem__)
        lo, hi = _width_bounds(w, lowest)
        zero_free = not bool(np.any(real == 0))
        if lo == 0 and len(members) <= cap[w] - zero_free:
            lo = 1  # enough non-zero values: no all-zero synthetic ID
        drawn = rng.permutation(_draw_excluding(lo, hi, len(members), real, rng))
        for k, v in zip(members, drawn.tolist()):
            value[k] = int(v)
            changed += (w > bound[k]) if k in bound else (w != nat[k])
    if changed:
        notes.append(f"{changed} synthetic ID(s) have a different number of digits than the real ID they replace "
                     "(too few non-real values of the real width; IDs stay non-real)")
    return np.asarray([value[k] for k in keys], dtype=np.int64)


def _draw_excluding(lo: int, hi: int, k: int, exclude: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``k`` distinct integers in [lo, hi) not in the sorted array ``exclude``."""
    excl = np.unique(exclude[(exclude >= lo) & (exclude < hi)])
    available = (hi - lo) - len(excl)
    if k > available:
        raise IdSpaceError(f"{k} synthetic IDs of the same shape cannot avoid all real IDs "
                           f"(only {available} free values in [{lo}, {hi}))")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if hi - lo <= _ENUMERATE_LIMIT:
        pool = np.setdiff1d(np.arange(lo, hi, dtype=np.int64), excl)
        return rng.choice(pool, size=k, replace=False)
    out = np.empty(0, dtype=np.int64)
    while len(out) < k:
        cand = rng.integers(lo, hi, size=int((k - len(out)) * 1.2) + 16, dtype=np.int64)
        cand = np.unique(cand)
        cand = cand[~np.isin(cand, excl)]
        cand = np.setdiff1d(cand, out)
        cand = rng.permutation(cand)
        out = np.concatenate([out, cand[: k - len(out)]])
    return out


def _fill_mask(mask: str, prefix: str, rng: np.random.Generator) -> str:
    out = list(prefix)
    for ch in mask[len(prefix):]:
        if ch == "9":
            out.append(chr(int(_DIGIT[rng.integers(10)])))
        elif ch == "A":
            out.append(chr(int(_UPPER[rng.integers(26)])))
        elif ch == "a":
            out.append(chr(int(_LOWER[rng.integers(26)])))
        else:
            out.append(ch)
    return "".join(out)


def _fill_hex(rep: str, rng: np.random.Generator) -> str:
    """Random hex digits in every alphanumeric position of ``rep`` (letters in the value's case)."""
    upper = any(ch.isupper() for ch in rep)
    out = []
    for ch in rep:
        if ch.isalnum():
            c = _HEX[int(rng.integers(16))]
            out.append(c.upper() if upper else c)
        else:
            out.append(ch)
    return "".join(out)


def _slots(rep: str, keep: str, hex_mode: bool) -> list[tuple[str, str]]:
    """Per-character (fixed text or '', alphabet) of the synthetic shape of ``rep`` (see :func:`_fill_mask`)."""
    out: list[tuple[str, str]] = [(ch, "") for ch in keep]
    upper = any(ch.isupper() for ch in rep)
    for ch in rep[len(keep):]:
        if hex_mode and ch.isalnum():
            out.append(("", _HEX.upper() if upper else _HEX))
        elif hex_mode:
            out.append((ch, ""))
        else:
            m = char_mask(ch)
            alphabet = {"9": "0123456789", "A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        "a": "abcdefghijklmnopqrstuvwxyz"}.get(m, "")
            out.append(("", alphabet) if alphabet else (ch, ""))
    return out


def _enumerate_free(slots: list[tuple[str, str]], need: int, banned: set[str],
                    rng: np.random.Generator, reject: Callable[[str], bool] | None = None) -> list[str] | None:
    """Up to ``need`` values of the shape ``slots`` whose D8 key is not in ``banned`` (and that ``reject``
    does not refuse: a real username, R5-02), in random order.

    Walks a random permutation of the whole space (only when it holds at most
    ``_ENUMERATE_LIMIT`` values); None when the space is too large to enumerate.
    """
    sizes = [len(a) for _, a in slots if a]
    space = 1
    for s in sizes:
        space *= s
        if space > _ENUMERATE_LIMIT:
            return None
    out: list[str] = []
    for idx in rng.permutation(space).tolist():
        chars = []
        rest = idx
        for fixed, alphabet in reversed(slots):
            if alphabet:
                rest, d = divmod(rest, len(alphabet))
                chars.append(alphabet[d])
            else:
                chars.append(fixed)
        v = "".join(reversed(chars))
        nk = normalize_id(v)
        if nk in banned or (reject is not None and reject(v)):
            continue
        banned.add(nk)
        out.append(v)
        if len(out) == need:
            break
    return out


def _pattern_mapping(ns: IdNamespace, keys: list[str], real_keys: set[str], used: set[str],
                     rng: np.random.Generator, foreign: set[str] | frozenset[str] = frozenset(),
                     prefix: str | None = None, stems: AbstractSet[str] = frozenset(),
                     text_taken: Callable[[str], bool] | None = None) -> None:
    """Same-shape synthetic values for the patterned ``keys``, avoiding ``real_keys``, ``used`` and ``foreign``,
    keeping ``prefix`` (``detect.ids.kept_prefix`` when None), never the username of a real ID (``stems``, R5-02)
    and never a text of the run's real-text registry (``text_taken``: :meth:`RealUsers.text_taken`, R9-03)."""
    reps = [ns.representatives[k] for k in keys]
    hex_mode = all(is_hex_id(r) for r in reps)
    if prefix is None:
        prefix = kept_prefix(reps, real_keys | set(foreign))
    if hex_mode:
        prefix = ""
    n_pre = len(prefix)
    mapping: dict[str, str] = {}
    pending = [(k, ns.representatives[k]) for k in keys]

    def keep_of(rep: str) -> str:
        return "" if hex_mode else (rep[:n_pre] if rep[:n_pre].casefold() == prefix else "")

    def real_user(v: str) -> bool:  # R5-02: the username of a real ID of the run; R9-03: a real text of the run
        return (bool(stems) and username_stem(v, prefix) in stems) or (text_taken is not None and text_taken(v))

    for _round in range(50):
        retry = []
        for k, rep in pending:
            v = _fill_hex(rep, rng) if hex_mode else _fill_mask(char_mask(rep), keep_of(rep), rng)
            nk = normalize_id(v)
            if nk in real_keys or nk in used or nk in foreign or real_user(v):
                retry.append((k, rep))
                continue
            used.add(nk)
            mapping[k] = v
        pending = retry
        if not pending:
            break
    if pending:
        # a nearly full shape: enumerate the free values of each pending shape (D9 fails only when none is left)
        by_shape: dict[tuple, list[str]] = {}
        for k, rep in pending:
            by_shape.setdefault(tuple(_slots(rep, keep_of(rep), hex_mode)), []).append(k)
        banned = set(real_keys) | used | set(foreign)
        for slots, ks in by_shape.items():
            free = _enumerate_free(list(slots), len(ks), banned, rng,
                                   real_user if stems or text_taken is not None else None)
            if free is None or len(free) < len(ks):
                raise IdSpaceError("cannot draw synthetic IDs of the same shape that avoid every real ID and every "
                                   f"real text of the run ({len(ks)} ID(s) of one shape found no free value)")
            for k, v in zip(ks, free):
                used.add(normalize_id(v))
                mapping[k] = v
    ns.mapping.update(mapping)


# --------------------------------------------------------------------------- R5-02: e-mail addresses as IDs
#: Name-like local parts tried per address, then as many same-shape masks (then one character longer at a time).
_EMAIL_ATTEMPTS = 60
_LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _reserved_domain_map(rows: dict[str, int]) -> dict[str, str]:
    """Real domain (case-folded) -> reserved domain by rows, most first (ties: first seen): example.org,
    example.com, example.net, then ``dN.example.org`` ... (``generate.pii.reserved_domain``), never one of the
    real domains (D24)."""
    from .pii import reserved_domain

    out: dict[str, str] = {}
    rank = 0
    for d in sorted(rows, key=lambda x: -rows[x]):
        cand = reserved_domain(rank)
        while cand in rows:
            rank += 1
            cand = reserved_domain(rank)
        out[d] = cand
        rank += 1
    return out


def _fake_local(local: str, attempt: int, pyr: random.Random) -> str:
    """A name-like local part with the shape of ``local``, built like an EMAIL column's (``generate.pii``,
    D24): letter runs become Faker name words of the same length (a first name, then last names; a single
    letter stays a random letter), digit runs random digits, separators are kept."""
    from . import pii
    from . import text as textgen

    runs = 0

    def word(length: int, run: str) -> str:
        nonlocal runs
        runs += 1
        if length == 1:
            return pyr.choice(_LOWER_LETTERS)
        return pii.name_word("first" if runs == 1 else "last", length, attempt, pyr)

    return textgen.mimic_words(local, pyr, word)


def _mask_local(local: str, pyr: random.Random) -> str:
    """Random letters (same case) and digits in the letter and digit positions of ``local``."""
    out = []
    for ch in local:
        if ch.isdigit():
            out.append(pyr.choice("0123456789"))
        elif ch.isalpha():
            c = pyr.choice(_LOWER_LETTERS)
            out.append(c.upper() if ch.isupper() else c)
        else:
            out.append(ch)
    return "".join(out)


def _email_mapping(ns: IdNamespace, keys: list[str], real_keys: set[str], used: set[str],
                   rng: np.random.Generator, foreign: AbstractSet[str], users: RealUsers) -> None:
    """Synthetic e-mail addresses for the e-mail ``keys`` (module docstring, R5-02).

    Each is accepted only when its D8 key is no real ID of the run and no other synthetic ID, it is no real
    e-mail address of the run, and its local part is no real local part (case-insensitive or by D8 key), no
    real ID and has no real username (``users``). Never fails: after the name-like and masked local parts,
    the local part grows one character at a time.
    """
    from . import text as textgen

    pyr = textgen.py_random(rng)
    rows: dict[str, int] = {}
    for k in keys:
        d = ns.representatives[k].rpartition("@")[2].casefold()
        rows[d] = rows.get(d, 0) + ns.key_rows.get(k, 1)
    dmap = _reserved_domain_map(rows)

    def ok(cand: str) -> bool:
        nk = normalize_id(cand)
        if nk in real_keys or nk in used or nk in foreign or cand.casefold() in users.emails:
            return False
        local = cand.rpartition("@")[0]
        lk = normalize_id(local)
        if local.casefold() in users.locals or lk in users.local_keys or lk in real_keys or lk in foreign:
            return False
        st = username_stem(cand)
        return not (st and st in users.stems) and not users.text_taken(cand)  # R9-03

    for k in keys:
        local, _, domain = ns.representatives[k].rpartition("@")
        dom = dmap[domain.casefold()]
        if domain.isupper():
            dom = dom.upper()
        cand = ""
        for a in range(_EMAIL_ATTEMPTS):
            cand = _fake_local(local, a, pyr) + "@" + dom
            if ok(cand):
                break
        else:
            for _ in range(_EMAIL_ATTEMPTS):
                cand = _mask_local(local, pyr) + "@" + dom
                if ok(cand):
                    break
            else:
                grow = local or "x"
                while not ok(cand):
                    grow = textgen.lengthen(grow, pyr)
                    cand = _mask_local(grow, pyr) + "@" + dom
        used.add(normalize_id(cand))
        ns.mapping[k] = cand


__all__ = ["IdNamespace", "IdRegistry", "build_mapping", "allocate_digit_ids", "column_is_padded", "luhn_eligible",
           "namespace_is_identity", "ID_CHECKS", "RealUsers"]
