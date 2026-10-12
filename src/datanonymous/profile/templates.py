"""Derived-template inference: a text column built from other columns in every row (AREA D; design D5).

Example: ``email = lower(first_name) + '.' + lower(last_name) + '.' + id + '@' + domain``
(student-scores: 2000 of 2000 rows). A template is a sequence of parts:

* ``('f', column_index, transform)`` - another column's stripped value through
  one of :data:`TRANSFORMS` (case changes, alphanumerics only, initials, a
  zero-padded ID written without its padding);
* ``('s', separator)`` - a literal separator from :data:`SEPARATORS`;
* ``('d',)`` - a free run of digits at the end (``jsmith2``), whose length
  distribution is reproduced (lengths only, never the digits).

For e-mail-like columns (every value ``local@domain``) the template covers the
local part; the domain is replaced by reserved domains (example.org / .com /
.net, the number of distinct domains kept; design D24/section 10). A template
must hold in 100% of the rows where the column is filled, on at least
``--min-cell`` rows. Inference proposes candidates from the first row,
filters them on a deterministic sample and verifies the best one on every row
(vectorised: :func:`render_array` transforms each distinct input value once).
Only column names, transforms and separators reach the report.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

SEPARATORS = (".", "_", "-", " ", ", ", ",", "+")


def _alnum(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def _no_leading_zeros(s: str) -> str:
    return (s.lstrip("0") or "0") if s.isascii() and s.isdigit() else s


#: Transforms in order of preference: when several describe the real data, the
#: first is used (alphanumeric variants first, so a synthetic value with a space
#: or apostrophe still yields a valid e-mail local part).
TRANSFORMS: dict[str, Callable[[str], str]] = {
    "lower_alnum": lambda s: _alnum(s).lower(),
    "lower": str.lower,
    "upper_alnum": lambda s: _alnum(s).upper(),
    "upper": str.upper,
    "alnum": _alnum,
    "as_is": lambda s: s,
    # a zero-padded ID written without its padding (student_id '0012345' in 'jsmith.12345@...', D8)
    "no_leading_zeros": _no_leading_zeros,
    "initial_lower": lambda s: s[:1].lower(),
    "initial_upper": lambda s: s[:1].upper(),
    "initial": lambda s: s[:1],
}
_TF_RANK = {name: i for i, name in enumerate(TRANSFORMS)}
_SAMPLE = 200
_MAX_CANDIDATES = 4000
_MAX_STEPS = 200_000

Part = tuple  # ('f', col, tf) | ('s', sep) | ('d',)


def split_email(value: str) -> tuple[str, str] | None:
    """(local, domain) when ``value`` has exactly one '@' with text on both sides."""
    if value.count("@") != 1:
        return None
    local, domain = value.split("@")
    if not local or not domain or " " in value:
        return None
    return local, domain


def render(parts: Sequence[Part], row: dict[int, str]) -> str:
    """The template's text for one row of input values (free digits rendered as empty)."""
    out = []
    for p in parts:
        if p[0] == "f":
            out.append(TRANSFORMS[p[2]](row[p[1]]))
        elif p[0] == "s":
            out.append(p[1])
    return "".join(out)


def matches(parts: Sequence[Part], row: dict[int, str], target: str) -> bool:
    """True if ``target`` is the template's text for ``row`` (a trailing digit run is free)."""
    prefix = render(parts, row)
    if parts and parts[-1][0] == "d":
        return target.startswith(prefix) and (len(target) == len(prefix) or target[len(prefix):].isdigit())
    return target == prefix


def describe(parts: Sequence[Part], names: Sequence[str], email: bool) -> str:
    """Report-safe text: column names, transforms and separators only."""
    out = []
    for p in parts:
        if p[0] == "f":
            out.append(names[p[1]] if p[2] == "as_is" else f"{p[2]}({names[p[1]]})")
        elif p[0] == "s":
            out.append(repr(p[1]))
        else:
            out.append("<digits>")
    text = " + ".join(out)
    return text + (" + '@' + <reserved domain>" if email else "")


def _segment(s: str, fields: list[tuple[int, str, str]]) -> list[tuple[Part, ...]]:
    """Every way to write ``s`` as fields, separators and an optional trailing digit run."""
    out: list[tuple[Part, ...]] = []
    by_first: dict[str, list[tuple[int, str, str]]] = {}
    for f in fields:
        by_first.setdefault(f[2][0], []).append(f)
    n = len(s)
    steps = [0]

    def rec(pos: int, parts: list[Part]) -> None:
        steps[0] += 1
        if len(out) >= _MAX_CANDIDATES or steps[0] > _MAX_STEPS:
            return
        if pos == n:
            if parts and parts[-1][0] == "f":
                out.append(tuple(parts))
            return
        for col, tf, text in by_first.get(s[pos], ()):
            if s.startswith(text, pos):
                parts.append(("f", col, tf))
                rec(pos + len(text), parts)
                parts.pop()
        if parts and parts[-1][0] == "f":
            for sep in SEPARATORS:
                if s.startswith(sep, pos):
                    parts.append(("s", sep))
                    rec(pos + len(sep), parts)
                    parts.pop()
            if s[pos:].isdigit():
                out.append(tuple(parts) + (("d",),))

    rec(0, [])
    return out


def _preference(parts: tuple[Part, ...]) -> tuple:
    fields = [p for p in parts if p[0] == "f"]
    return (parts[-1][0] == "d", sum(_TF_RANK[p[2]] for p in fields), len(parts),
            [p[1] for p in fields])


def infer_template(targets: list[str], inputs: dict[int, list[str | None]], min_rows: int,
                   email: bool) -> tuple[Part, ...] | None:
    """Best template for ``targets`` (stripped values; local parts for e-mails) or None.

    ``inputs`` maps a column index to its stripped values aligned with
    ``targets`` (None/'' where missing). The template must hold in every row.
    """
    n = len(targets)
    if n < max(min_rows, 2):
        return None
    step = max(1, n // _SAMPLE)
    sample = list(range(0, n, step))[:_SAMPLE]
    # usable inputs: filled in every target row of the sample
    usable = [c for c, vals in inputs.items() if all(vals[r] for r in sample)]
    if not usable:
        return None
    # transforms per column, de-duplicated over the sample (keep the preferred one of identical outputs)
    fields_tf: list[tuple[int, str]] = []
    for c in usable:
        seen: dict[tuple[str, ...], str] = {}
        # 'as_is' first: a transform that changes nothing on the sample is described as the value itself
        for tf in ("as_is", *(t for t in TRANSFORMS if t != "as_is")):
            fn = TRANSFORMS[tf]
            outs = tuple(fn(inputs[c][r] or "") for r in sample)
            if any(o == "" for o in outs) or outs in seen:
                continue
            if tf.startswith("initial") and len(sample) > 1 and len(set(outs)) == 1:
                continue  # one initial on every row ('4' of 5-digit staff numbers) is a constant, not a derivation
            seen[outs] = tf
            fields_tf.append((c, tf))
    first = sample[0]
    fields = [(c, tf, TRANSFORMS[tf](inputs[c][first] or "")) for c, tf in fields_tf]
    cands = _segment(targets[first], fields)
    if not cands:
        return None
    # a template may end in free digits that the first row happens not to have (jsmith, jsmith2)
    extra = [c + (("d",),) for c in cands if c[-1][0] == "f"]
    cands = list(dict.fromkeys(cands + extra))
    for r in sample[1:]:
        row = {c: inputs[c][r] or "" for c in usable}
        cands = [c for c in cands if matches(c, row, targets[r])]
        if not cands:
            return None
    cands.sort(key=_preference)
    arrays: dict[int, np.ndarray] = {}
    target_arr = np.empty(n, dtype=object)
    target_arr[:] = targets
    for cand in cands:
        for c in {p[1] for p in cand if p[0] == "f"}:
            if c not in arrays:
                arrays[c] = as_array(inputs[c])
        rendered, missing = render_array(cand, arrays)
        if missing.any():
            continue
        if cand[-1][0] != "d":
            ok = bool(np.all(rendered == target_arr))
        else:
            ok = all(t.startswith(p) and (len(t) == len(p) or t[len(p):].isdigit())
                     for p, t in zip(rendered.tolist(), targets))
        if ok:
            return cand
    return None


def as_array(values: Any) -> np.ndarray:
    """Object array of the stripped values (``values`` may be a sequence or offer ``array()``)."""
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "array"):
        return values.array()
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def render_array(parts: Sequence[Part], arrays: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`render` over aligned input arrays: (texts, rows with a missing input).

    Transforms run once per distinct input value; a missing input is None or ''.
    """
    n = len(next(iter(arrays.values()))) if arrays else 0
    out = np.empty(n, dtype=object)
    out[:] = ""
    missing = np.zeros(n, dtype=bool)
    for p in parts:
        if p[0] == "f":
            codes, uniq = pd.factorize(arrays[p[1]], use_na_sentinel=False)
            fn = TRANSFORMS[p[2]]
            texts = np.empty(len(uniq), dtype=object)
            miss_u = np.zeros(len(uniq), dtype=bool)
            for i, u in enumerate(uniq.tolist()):
                if not isinstance(u, str) or u == "":
                    miss_u[i] = True
                    texts[i] = ""
                else:
                    texts[i] = fn(u)
            codes = np.asarray(codes, dtype=np.int64)
            out = out + texts[codes]
            missing |= miss_u[codes]
        elif p[0] == "s":
            out = out + p[1]
    return out, missing


def digit_suffix_lengths(parts: Sequence[Part], targets: list[str], inputs: dict[int, Any]) -> Counter:
    """Length distribution of the free trailing digit run (counts per length, incl. 0)."""
    cols = {p[1] for p in parts if p[0] == "f"}
    rendered, _ = render_array(parts, {c: as_array(inputs[c]) for c in cols})
    return Counter(len(t) - len(p) for p, t in zip(rendered.tolist(), targets))


RESERVED_DOMAINS = ("example.org", "example.com", "example.net")


def reserved_domains(real_counts: Counter) -> list[tuple[str, int]]:
    """Reserved replacement domains, one per real domain (most frequent first), never a real one.

    The first three are example.org / example.com / example.net (design D24),
    then ``mailN.example.org``. Returns (reserved domain, real row count) pairs.
    """
    real = {d.casefold() for d in real_counts}
    pool: list[str] = [d for d in RESERVED_DOMAINS if d not in real]
    i = 2
    out: list[tuple[str, int]] = []
    for _, count in sorted(real_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        while not pool:
            cand = f"mail{i}.example.org"
            i += 1
            if cand not in real:
                pool.append(cand)
        out.append((pool.pop(0), int(count)))
    return out


__all__ = ["SEPARATORS", "TRANSFORMS", "infer_template", "render", "render_array", "as_array", "matches", "describe",
           "split_email",
           "digit_suffix_lengths", "reserved_domains", "RESERVED_DOMAINS"]
