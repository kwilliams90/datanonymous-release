"""Excel cell storage per column: the mix of cell types and number formats, kept by exact quota (AREA E helper).

The reader (``excel_read.finalize_column``) counts a column's non-empty data
cells per storage combination, cell type (``s`` text, ``n`` number, ``d``
date, ``b`` boolean, ``e`` error) and number format (literal text masked,
:mod:`.numfmt`): ``native["storage"] = {type: {format: count}}``. When the
column holds more than one combination it also keeps, PRIVATELY, the
combinations of each distinct value (``native["value_storage"]``, at most
``excel_read.VALUE_TYPES_CAP`` values; real values, never reported).

Writing (:func:`plan_column`), design 5.3 exact quota applied to storage:

* each synthetic value has at most one non-text type it can be stored as
  (:func:`cell_option`): an error token the source column holds, a number,
  a date or TRUE/FALSE. When the source column also holds text cells, a number
  or a date must be in the reader's canonical form (``2134``, ``0.5``,
  ``2024-01-05``), so a text value such as ``02134``, ``1.0`` or ``20240105``
  is never turned into a number or a date (its text would change);
* target cells per type: the source counts scaled to the number of non-empty
  synthetic cells (largest remainder: the source counts themselves at the same
  count). Every type gets ``min(target, cells that can take it)`` cells and the
  rest are text; a column with no text cells in the source stores every value
  as its own type (a value that can take none is written as text);
* within each type the number formats are dealt by exact quota of the
  source formats of that type (the minority ``0%`` cells beside ``0.00`` ones
  stay); a synthetic value that is also a source value first takes the
  storage its source cells had, in their proportions (a grade level ``KG``
  stored as text stays text, ``5`` stays a number); the other cells are dealt
  at random from the seeded stream.

Self-check (:func:`storage_issues`): from the re-read output's per-cell types
and values, every type must hold exactly ``min(target, cells that could take
it)`` cells and every type its exact format quota. A dropped or invented
minority type or format is therefore a layout mismatch, while the part of a
type change that the synthetic values force (labels where the source had
numbers) is left to the dominant-type rule of ``excelio.layout_diff``.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .excel_read import canon_date, canon_number

#: cell types in their fixed order (also the option precedence after errors: see :func:`cell_option`)
TYPES = "sndbe"
TYPE_NAMES = {"s": "text", "n": "number", "d": "date", "b": "boolean", "e": "error"}

_INT_RE = re.compile(r"[+-]?\d+")
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_DUR_RE = re.compile(r"(-)?(\d+):(\d{2})(?::(\d{2})(?:\.(\d+))?)?")
_BOOLS = ("TRUE", "FALSE")


# --------------------------------------------------------------------------- typing synthetic strings
def parse_date(s: str, kind: str | None) -> Any:
    """ISO text (the reader's canonical form) -> datetime / time / timedelta; None if not a date."""
    st = s.strip()
    if kind == "duration" or (kind is None and _DUR_RE.fullmatch(st) and ":" in st and "-" not in st[1:]):
        m = _DUR_RE.fullmatch(st)
        if m:
            sign, h, mi, sec, frac = m.groups()
            td = dt.timedelta(hours=int(h), minutes=int(mi), seconds=int(sec or 0),
                              microseconds=int((frac or "0")[:6].ljust(6, "0")))
            return -td if sign else td
        if kind == "duration":
            return None
    if kind == "time":
        try:
            return dt.time.fromisoformat(st)
        except ValueError:
            pass
    try:
        v = dt.datetime.fromisoformat(st)
    except ValueError:
        return None
    if v.tzinfo is not None:
        v = v.replace(tzinfo=None)
    if kind == "time":
        return v.time()
    return v


def convert(s: str, t: str, kind: str | None) -> Any:
    """The Python value of string ``s`` stored as cell type ``t`` (None when it cannot be)."""
    if t == "s":
        return s
    st = s.strip()
    if t == "n":
        if _INT_RE.fullmatch(st):
            return int(st)
        if _NUM_RE.fullmatch(st):
            return float(st)
        return None
    if t == "b":
        u = st.upper()
        return True if u == "TRUE" else False if u == "FALSE" else None
    if t == "d":
        return parse_date(st, kind)
    if t == "e":
        return s
    return None


def is_canonical_number(s: str) -> bool:
    """True when ``s`` is how the reader renders a number cell (``canon_number``): '2134', '0.5', '-3'.

    '02134', '+5', '1.50' or '1e3' can only come from a text cell, so in a
    column that mixes text and number cells they stay text (a ZIP code '02134'
    keeps its leading zero).
    """
    if _INT_RE.fullmatch(s):
        return s == str(int(s)) and not s.startswith("+") and s != "-0"
    if not _NUM_RE.fullmatch(s):
        return False
    try:
        return canon_number(float(s)) == s
    except (ValueError, OverflowError):
        return False


def cell_option(s: str, types: Mapping[str, int] | set[str], strict: bool, kind: str | None,
                errors: set[str] | frozenset[str]) -> str | None:
    """The non-text cell type string ``s`` can be stored as in a column whose source holds ``types``.

    The first of: ``e`` (an error token of the source column), ``n`` (a
    number), ``d`` (a date), ``b`` (TRUE/FALSE); None = text only. ``strict``
    (the source column also holds text cells): a number or a date must be in
    the reader's canonical form (``2134``; ``2024-01-05`` for a date column,
    ``2024-01-05 08:00:00`` for a datetime one) and a boolean exactly
    ``TRUE``/``FALSE``, so storing the value as that type keeps its text.
    """
    if "e" in types and s in errors:
        return "e"
    st = s.strip()
    if "n" in types and _NUM_RE.fullmatch(st) and (not strict or (st == s and is_canonical_number(st))):
        return "n"
    if "d" in types:
        v = parse_date(st, kind)
        if v is not None and (not strict or canon_date(v, kind) == s):
            return "d"
    if "b" in types and (s in _BOOLS if strict else st.upper() in _BOOLS):
        return "b"
    return None


# --------------------------------------------------------------------------- quotas
def scaled(counts: Mapping[Any, int], n: int) -> dict[Any, int]:
    """Exact quota of ``n`` over ``counts`` (largest remainder in exact integer arithmetic; ties: larger
    count, then label order), shared by the writer and the self-check. At ``n`` = the total of ``counts``
    it is ``counts`` itself."""
    labels = sorted((k for k, v in counts.items() if v > 0), key=lambda k: (-counts[k], str(k)))
    if not labels:
        return {}
    if n <= 0:
        return {k: 0 for k in labels}
    total = sum(int(counts[k]) for k in labels)
    base = [int(counts[k]) * n // total for k in labels]
    frac = [int(counts[k]) * n % total for k in labels]
    for i in sorted(range(len(labels)), key=lambda i: (-frac[i], i))[:n - sum(base)]:
        base[i] += 1
    return dict(zip(labels, base))


def storage_of(nat: Mapping[str, Any] | None) -> dict[str, dict[str, int]]:
    """``{type: {format: count}}`` of a native Excel description (older descriptions: main formats only)."""
    if not nat:
        return {}
    st = nat.get("storage")
    if isinstance(st, dict):
        return {t: {str(f): int(c) for f, c in fm.items() if c} for t, fm in st.items() if t in TYPES and fm}
    formats = nat.get("formats") or {}
    return {t: {str(formats.get(t) or "General"): int(c)} for t, c in (nat.get("types") or {}).items()
            if t in TYPES and c}


def type_counts(storage: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {t: sum(fm.values()) for t, fm in storage.items() if sum(fm.values()) > 0}


def _deal(cells: list[int], quotas: dict[Any, int], strings: Sequence[str | None],
          pref: Callable[[str], dict[Any, int] | None] | None, rng: np.random.Generator) -> dict[Any, list[int]]:
    """Cells (row indices) split by exact ``quotas`` (summing to ``len(cells)``).

    A cell whose value ``pref`` knows takes the labels of that value first
    (its counts scaled to the value's cells, within the remaining quotas); the
    other cells are dealt at random.
    """
    labels = [k for k, q in quotas.items() if q > 0]
    out: dict[Any, list[int]] = {k: [] for k in labels}
    if len(labels) <= 1:
        if labels:
            out[labels[0]] = list(cells)
        return out
    order = [cells[j] for j in rng.permutation(len(cells))]
    remaining = {k: quotas[k] for k in labels}
    taken: set[int] = set()
    if pref is not None:
        by_value: dict[str, list[int]] = {}
        for i in order:
            by_value.setdefault(strings[i] or "", []).append(i)
        for s, lst in by_value.items():
            p = pref(s)
            if not p:
                continue
            p = {k: c for k, c in p.items() if c > 0 and remaining.get(k, 0) > 0}
            pos = 0
            for k, w in scaled(p, len(lst)).items():
                take = min(w, remaining[k])
                if take > 0:
                    out[k].extend(lst[pos:pos + take])
                    taken.update(lst[pos:pos + take])
                    remaining[k] -= take
                    pos += take
    rest = [i for i in order if i not in taken] if taken else order
    pos = 0
    for k in labels:
        q = remaining[k]
        out[k].extend(rest[pos:pos + q])
        pos += q
    return out


# --------------------------------------------------------------------------- writer plan
@dataclass
class ColumnPlan:
    """Storage of every synthetic cell of one column: ``codes[i]`` indexes ``combos`` (-1 = empty cell)."""

    strings: list[str | None]
    codes: np.ndarray
    combos: list[tuple[str, str | None]] = field(default_factory=list)
    kind: str | None = None

    def cell(self, i: int) -> tuple[Any, str, str | None]:
        """(Python value, cell type, number format or None) of row ``i``; (None, '', None) = empty."""
        c = int(self.codes[i])
        if c < 0:
            return None, "", None
        t, fmt = self.combos[c]
        s = self.strings[i]
        assert s is not None
        v = convert(s, t, self.kind)
        if v is None:  # cannot happen (cell_option checked it): keep the text
            return s, "s", None
        return v, t, fmt


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    if s and _ILLEGAL.search(s):
        s = _ILLEGAL.sub("", s)  # openpyxl cannot store control characters
    return s or None


_ILLEGAL = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


def plan_column(values: Sequence[Any], nat: Mapping[str, Any] | None, rng: np.random.Generator) -> ColumnPlan:
    """Cell type and number format of each synthetic value (see the module docstring)."""
    n = len(values)
    strings = [_clean(values[i]) for i in range(n)]
    codes = np.full(n, -1, dtype=np.int32)
    nonempty = [i for i, s in enumerate(strings) if s is not None]
    plan = ColumnPlan(strings, codes, [], (nat or {}).get("date_kind"))
    storage = storage_of(nat)
    types = type_counts(storage)
    if not nonempty:
        return plan
    if not types:
        plan.combos.append(("s", None))
        codes[nonempty] = 0
        return plan
    strict = "s" in types
    errors = frozenset((nat or {}).get("errors") or ())
    vs: dict[str, Counter] = (nat or {}).get("value_storage") or {}
    memo: dict[str, str | None] = {}
    groups: dict[str, list[int]] = {}
    for i in nonempty:
        s = strings[i]
        o = memo.get(s, "?")
        if o == "?":
            o = memo[s] = cell_option(s, types, strict, plan.kind, errors)
        groups.setdefault(o or "s", []).append(i)
    by_type: dict[str, list[int]] = {"s": groups.pop("s", [])}
    target = scaled(types, len(nonempty))
    for t in sorted(groups, key=TYPES.index):
        cells = groups[t]
        if not strict:
            by_type.setdefault(t, []).extend(cells)
            continue
        x = min(target.get(t, 0), len(cells))

        def type_pref(s: str, t: str = t) -> dict[str, int] | None:
            c = vs.get(s)
            if not c:
                return None
            return {t: sum(k for (tt, _), k in c.items() if tt == t),
                    "s": sum(k for (tt, _), k in c.items() if tt == "s")}

        dealt = _deal(cells, {t: x, "s": len(cells) - x}, strings, type_pref if vs else None, rng)
        for k, lst in dealt.items():
            by_type.setdefault(k, []).extend(lst)
    for t in TYPES:
        cells = sorted(by_type.get(t) or [])
        if not cells:
            continue
        fmts = storage.get(t)
        if not fmts:
            plan.combos.append((t, None))
            codes[cells] = len(plan.combos) - 1
            continue

        def fmt_pref(s: str, t: str = t) -> dict[str, int] | None:
            c = vs.get(s)
            return {f: k for (tt, f), k in c.items() if tt == t} if c else None

        dealt = _deal(cells, scaled(fmts, len(cells)), strings, fmt_pref if vs else None, rng)
        for f, lst in dealt.items():
            if lst:
                plan.combos.append((t, f))
                codes[lst] = len(plan.combos) - 1
    _keep_first(plan, nonempty, (nat or {}).get("first"), memo, strict, vs)
    return plan


def _keep_first(plan: ColumnPlan, nonempty: list[int], first: Any, memo: dict[str, str | None],
                strict: bool, vs: Mapping[str, Counter]) -> None:
    """Give the first non-empty cell the storage of the source's first one, by swapping with a later cell
    (counts unchanged): pandas types a column of booleans and TRUE/FALSE text by its first cell. A swap
    never gives a source value a storage none of its source cells had."""
    if not first or not nonempty:
        return
    want = (str(first[0]), str(first[1]))
    if want not in plan.combos:
        return
    cw = plan.combos.index(want)
    i0 = nonempty[0]
    c0 = int(plan.codes[i0])
    if c0 == cw:
        return

    def can(i: int, combo: tuple[str, str | None]) -> bool:
        s = plan.strings[i] or ""
        t = combo[0]
        if not ((t == "s" and strict) or memo.get(s) == t):
            return False
        known = vs.get(s)
        return not known or known.get(combo, 0) > 0

    have = plan.combos[c0]
    if not can(i0, want):
        return
    for i in reversed(nonempty[1:]):
        if int(plan.codes[i]) == cw and can(i, have):
            plan.codes[i0], plan.codes[i] = cw, c0
            return


# --------------------------------------------------------------------------- self-check
def storage_issues(name: str, src: Mapping[str, Any], values: Sequence[Any], out_native: Mapping[str, Any] | None
                   ) -> list[str]:
    """Differences between a source column's storage mix (``src``: the layout entry built by the reader:
    ``cells``, ``date_kind``, ``errors``) and the re-read output column (canonical ``values`` and its
    native description with ``cell_types``). Empty when the writer reproduced it by exact quota."""
    cells = storage_of({"storage": src.get("cells") or {}})
    types = type_counts(cells)
    nat = out_native or {}
    out = storage_of(nat)
    o_types = type_counts(out)
    n_out = sum(o_types.values())
    if not types or not n_out:
        return []
    strict = "s" in types
    kind = src.get("date_kind")
    errors = frozenset(src.get("errors") or ())
    can: Counter = Counter()
    per_cell = nat.get("cell_types")
    if per_cell is not None:
        for v, t in zip(values, per_cell):
            if not t:
                continue
            tc = chr(t)
            if tc != "s":
                can[tc] += 1
            elif isinstance(v, str) and v:
                o = cell_option(v, types, strict, kind, errors)
                if o:
                    can[o] += 1
    else:  # no per-cell types: what the values allow
        for t, c in o_types.items():
            if t != "s":
                can[t] += c
    target = scaled(types, n_out)
    issues: list[str] = []
    n_src = sum(types.values())
    for t in TYPES:
        if t == "s":
            continue
        have = o_types.get(t, 0)
        if t not in types:
            if have:
                issues.append(f"column {name!r}: {have} {TYPE_NAMES[t]} cell(s), a type the source column "
                              "does not hold")
            continue
        want = min(target.get(t, 0), can[t]) if strict else can[t]
        if have != want:
            issues.append(f"column {name!r}: {have} {TYPE_NAMES[t]} cell(s) of {n_out}, expected {want} "
                          f"(the source holds {types[t]} of {n_src})")
    for t, fm in out.items():
        if t not in cells or not o_types.get(t):
            continue
        want_f = {f: c for f, c in scaled(cells[t], o_types[t]).items() if c}
        if fm != want_f:
            issues.append(f"column {name!r}: number formats of the {TYPE_NAMES[t]} cells {_fmt_counts(fm)}, "
                          f"expected {_fmt_counts(want_f)}")
    return issues


def _fmt_counts(fm: Mapping[str, int]) -> str:
    return "{" + ", ".join(f"{f!r}: {c}" for f, c in sorted(fm.items(), key=lambda kv: (-kv[1], kv[0]))) + "}"
