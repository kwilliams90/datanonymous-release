"""Out-dir documentation computed from the SYNTHETIC data only (AREA G; design D20).

Writes ``README.md``, ``DATA_DICTIONARY.md`` and ``schema.json`` at the out-dir
root. They travel with the synthetic files (an AI skill's data folder), so
they hold no real-derived statistic: every count, level and summary comes from
the synthetic files; types and formats come from detection metadata. The
output is byte-identical for a given seed (no timestamps, no input paths).
Sharing mode omits the seed and reproduction details, labels its privacy
contract separately, and limits ordered vocabulary to labels written in the
output. The detailed exact-fidelity behavior below applies to fidelity mode,
which is labelled for invented inputs and local testing.

* ``README.md``: a SYNTHETIC DATA banner; how the folder was generated
  (version, Faker pin, seed, how to reproduce); a per-file summary (rows,
  columns, ID columns, columns constant within an ID, dropped/nulled columns);
  what IS preserved (layout, per-column distributions, ID blocks, within-ID
  constants, missingness patterns, structural rules applied in this run;
  row-number ID columns 1..n that keep their numbers, D9, are named so the
  "no real ID" statement stays true) and what is NOT (relationships between
  independent columns: correlations, group differences, cross-column logic;
  panel/time structure, with the panels detected per file, D16; real personal
  values; formal privacy); files left from an earlier run whose source FAILed.
* ``DATA_DICTIONARY.md``: per file the layout (format, encoding, BOM,
  delimiter, line ending, quoting), a readr ``col_types`` string and read
  snippets for R and pandas; per column the original header, type, R class,
  readr letter, the dtype pandas infers from the synthetic file, format details
  (decimals, padding, value shapes, date formats from the area hooks) and the
  written NA tokens, missing %, distinct count, levels (label-safe columns
  only, as written: the real levels the D35 allowlist kept and the pseudonyms
  of the others, which are synthetic labels; ``pseudonymized_levels`` counts
  them) or a synthetic numeric/length summary, and the adjusted flags. Counts
  below ``--min-cell`` k are shown as ``<k``. A ``--null`` column of a text
  file is documented with the empty cells it holds; R string literals escape
  control characters (``delim = "\\t"``).
* ``schema.json``: types and formats only (no statistics): per file the
  layout, per table the readr ``col_types``, per column name, type, R class,
  readr letter, pandas dtype, NA tokens, role, format details and the level
  order of ordered factors.

Per-type dictionary fields come from the area hooks ``detect.dates.doc_fields``
(A), ``detect.numeric.doc_fields`` (B) and ``detect.text.doc_fields`` (C): their
keys are added to the column entry, and ``'readr'`` / ``'r_class'`` keys replace
the type's readr letter / R class. A's R parsing ``note`` is published as
``parse`` (B's key for the same instruction). schema.json keeps the hook fields
that describe types and formats only: statistics and prose
(:data:`SCHEMA_HOOK_EXCLUDED`) stay in the dictionary, and mixed date formats
are listed without their row shares. If a doc name collides with a synthetic data file at the out-dir
root, the doc gets the ``_datanonymous_`` prefix and a warning (applied again while the prefixed name is
taken too: a doc never overwrites a data file, :func:`doc_paths`).
:func:`rewrite_docs` rebuilds the docs from the run results alone (used by the
leak scan after it removed a FAILed output). A header, sheet, file or folder
name that holds personal data (D34, ``privacy.headers``) is never printed: the
stale-output list uses the redacted paths, and the run's redactor is applied to
every doc before it is written.
"""

from __future__ import annotations

import importlib.metadata as md
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..detect import dates as dates_hooks
from ..detect import numeric as numeric_hooks
from ..detect import text as text_hooks
from ..model import (
    LABEL_SAFE_TYPES,
    TYPE_INFO,
    ColumnProfile,
    ColumnType,
    FileResult,
    Message,
    SyntheticTable,
    TableProfile,
)
from ..paths import replace_bytes
from ..privacy.headers import screen_of

DOC_NAMES = ("README.md", "DATA_DICTIONARY.md", "schema.json")
DOC_KINDS = {"README.md": "readme", "DATA_DICTIONARY.md": "data_dictionary", "schema.json": "schema"}
DOC_PREFIX = "_datanonymous_"
#: How the statistics show a native null (``profile.stats.token_display``).
NULL_DISPLAY = "<null>"
READR_LOGICAL = frozenset({"TRUE", "FALSE", "T", "F", "true", "false", "True", "False"})
#: Column-entry keys that schema.json keeps (types and formats only).
SCHEMA_KEYS = ("name", "type", "subtype", "r_class", "readr", "pandas_dtype", "na_tokens", "role", "decimals",
               "float_text", "plus_sign", "zero_padded", "width", "shapes", "boolean_tokens", "level_order",
               "row_numbers_kept")


@dataclass
class WrittenFile:
    result: FileResult
    tables: list[SyntheticTable]
    synthetic_stats: list[dict[int, dict[str, Any]]]
    dtypes: list[list[str] | None]


@dataclass
class _DocTable:
    name: str | None
    n_rows: int
    profile: TableProfile
    out_columns: list[int]  # source column indices of the output columns, in output order
    stats: dict[int, dict[str, Any]]
    dtypes: list[str] | None


def _tables_of_written(w: WrittenFile) -> list[_DocTable]:
    out = []
    for t_i, st in enumerate(w.tables):
        out.append(_DocTable(st.name, st.n_rows, st.profile, [c.index for c in st.columns],
                             w.synthetic_stats[t_i] if t_i < len(w.synthetic_stats) else {},
                             w.dtypes[t_i] if t_i < len(w.dtypes) else None))
    return out


def _tables_of_result(fr: FileResult) -> list[_DocTable]:
    out = []
    for t_i, tp in enumerate(fr.tables):
        out.append(_DocTable(tp.name, tp.n_rows, tp, [p.index for p in tp.columns if p.role != "excluded"],
                             fr.synthetic_stats[t_i] if t_i < len(fr.synthetic_stats) else {},
                             fr.out_dtypes[t_i] if t_i < len(fr.out_dtypes) else None))
    return out


def doc_paths(out_rel_paths: list[str]) -> tuple[dict[str, str], list[Message]]:
    """Doc name -> file name at the out-dir root, never one a data file (or its folder) already holds (D20).

    A doc whose name is taken gets the ``_datanonymous_`` prefix; when the prefixed name is taken too (the
    input held a data file of that name), the prefix is applied again until the name is free, so a doc
    never overwrites a synthetic data file. Names compare case-insensitively (Windows, macOS).
    """
    taken: set[str] = set()
    for p in out_rel_paths:
        rel = str(p).replace("\\", "/").strip("/")
        taken.add(rel.casefold())
        taken.add(rel.split("/", 1)[0].casefold())  # a root-level folder of the mirrored tree blocks the name too
    names: dict[str, str] = {}
    msgs: list[Message] = []
    for d in DOC_NAMES:
        name = d
        while name.casefold() in taken:
            name = DOC_PREFIX + name
        taken.add(name.casefold())
        names[d] = name
        if name != d:
            msgs.append(Message("warning", "doc_name_collision",
                                f"a synthetic data file is named {d}; the doc is written as {name}"))
    return names, msgs


def r_class(p: ColumnProfile, syn: dict[str, Any]) -> str:
    if p.role == "null":
        return "logical"
    if p.type is ColumnType.INTEGER and p.params.get("float_text"):
        return "numeric"
    if p.type is ColumnType.CONSTANT and _numeric_value(p.params.get("value", p.params.get("_value"))):
        return "numeric"
    return TYPE_INFO[p.type].r_class


def readr_letter(p: ColumnProfile, syn: dict[str, Any]) -> str:
    if p.role == "null":
        return "l"
    t = p.type
    if t is ColumnType.BOOLEAN:
        tokens = {str(k).strip() for k in syn.get("levels", {})}
        return "l" if tokens and tokens <= READR_LOGICAL else "c"
    if t in (ColumnType.INTEGER, ColumnType.INTEGER_YEAR):
        if p.params.get("float_text"):
            return "d"
        try:
            big = max(abs(float(syn.get("min", 0) or 0)), abs(float(syn.get("max", 0) or 0)))
        except (TypeError, ValueError):
            big = 0.0
        return "i" if big < 2**31 else "d"
    if t is ColumnType.CONSTANT:
        return "d" if _numeric_value(p.params.get("value", p.params.get("_value"))) else "c"
    return TYPE_INFO[t].readr


def _numeric_value(v: Any) -> bool:
    if v is None:
        return False
    try:
        float(str(v))
        return True
    except (TypeError, ValueError):
        return False


def small(count: int, k: int) -> int | str:
    """A count for the shared docs: counts below the small-cell k are shown as ``<k``."""
    return f"<{k}" if count < k else int(count)


_small = small

_AREA_B = frozenset({ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL})


def area_doc_fields(p: ColumnProfile, syn: dict[str, Any]) -> dict[str, Any]:
    """Per-type dictionary fields from the area hooks (A dates, B formatted numbers, C text/PII).

    AREA A's R parsing ``note`` is published under AREA B's key for the same thing, ``parse`` (how to
    read the text column in R), so every reading instruction has one key in the docs and schema.json.
    """
    if p.role == "null":
        return {}
    family = TYPE_INFO[p.type].family
    if family == "date":
        fields = dict(dates_hooks.doc_fields(p, syn) or {})
        if "note" in fields:
            fields["parse"] = fields.pop("note")
        return fields
    if p.type in _AREA_B:
        return dict(numeric_hooks.doc_fields(p, syn) or {})
    if family in ("text", "pii") or p.type in (ColumnType.LAT, ColumnType.LON):
        return dict(text_hooks.doc_fields(p, syn) or {})
    return {}


def _format_fields(p: ColumnProfile, syn: dict[str, Any]) -> dict[str, Any]:
    """Format details from detection metadata and the synthetic values (never real values)."""
    out: dict[str, Any] = {}
    prm = p.params
    if p.type in (ColumnType.FLOAT, ColumnType.LAT, ColumnType.LON) and "decimals" in prm:
        out["decimals"] = prm["decimals"]
    if p.type in (ColumnType.INTEGER, ColumnType.FLOAT) and prm.get("float_text"):
        out["float_text"] = True
    if prm.get("plus_sign"):
        out["plus_sign"] = True
    if p.type is ColumnType.BOOLEAN and isinstance(syn.get("levels"), dict):
        out["boolean_tokens"] = sorted(str(k) for k in syn["levels"])
    if p.type in _SHAPED:
        length = syn.get("length")
        if isinstance(length, dict) and length.get("min") is not None and length.get("min") == length.get("max"):
            out["width"] = length["min"]
        masks = syn.get("top_masks") or {}
        if masks:
            out["shapes"] = list(masks)[:3]  # character-class masks of synthetic values held by >= k rows
        shape = prm.get("shape")
        if p.type is ColumnType.DIGIT_CODE or (isinstance(shape, dict) and shape.get("padded")):
            out["zero_padded"] = True
    return out


#: Types whose values have a fixed shape worth documenting (width, character-class masks, zero padding).
_SHAPED = frozenset({ColumnType.IDENTIFIER, ColumnType.DIGIT_CODE, ColumnType.PHONE, ColumnType.SSN, ColumnType.ZIP})


def _written_tokens(p: ColumnProfile, syn: dict[str, Any], text_output: bool) -> dict[str, int]:
    """Missing-value tokens as written: a ``--null`` column of a text file holds empty cells, not ``<null>``."""
    toks = {str(t): int(c) for t, c in (syn.get("missing_tokens") or {}).items()}
    if text_output and p.role == "null" and NULL_DISPLAY in toks:
        toks[""] = toks.get("", 0) + toks.pop(NULL_DISPLAY)
    return toks


def _entry_parts(p: ColumnProfile, syn: dict[str, Any], dtype: str | None, k: int,
                 text_output: bool = False, *, mode: str = "fidelity") -> tuple[dict[str, Any], list[str]]:
    tokens = _written_tokens(p, syn, text_output)
    entry: dict[str, Any] = {
        "name": p.name,
        "type": p.type.value,
        "subtype": p.subtype,
        "r_class": r_class(p, syn),
        "readr": readr_letter(p, syn),
        "pandas_dtype": dtype,
        "na_tokens": sorted(tokens),
        "na_token_counts": {t: small(c, k) for t, c in tokens.items()},
        "missing_rate": syn.get("missing_rate"),
        "distinct": syn.get("distinct"),
        "role": p.role,
        "rare_category_adjusted": bool(p.flags.get("rare_category_adjusted")),
        "extreme_value_adjusted": bool(p.flags.get("extreme_value_adjusted")),
    }
    pseudo = int(p.params.get(text_hooks.PSEUDO_COUNT_KEY) or 0) if p.role != "null" else 0
    if pseudo:
        entry["pseudonymized_levels"] = pseudo  # D35: how many levels are written as pseudonyms (listed below)
    entry.update(_format_fields(p, syn))
    if mode != "sharing" and p.role == "id" and p.params.get("_identity_mapped"):
        entry["row_numbers_kept"] = True  # Fidelity: the original 1..n assignments remain.
    if p.type in LABEL_SAFE_TYPES and isinstance(syn.get("levels"), dict) and p.role != "null":
        # the levels as written: a level the D35 allowlist pseudonymised shows as a placeholder in the (real vs
        # synthetic) statistics and as its pseudonym here, since the docs describe the synthetic files
        labels = _pseudonym_labels(p)
        if mode == "sharing":
            labels = {ph: [(fake, count) for fake, count in fakes if count > 0]
                      for ph, fakes in labels.items()}
            entry.pop("pseudonymized_levels", None)
            written_pseudonyms = {fake for fakes in labels.values() for fake, _ in fakes}
            if written_pseudonyms:
                entry["pseudonymized_levels"] = len(written_pseudonyms)
        levels: dict[Any, int] = {}
        for lv, c in syn["levels"].items():
            fakes = labels.get(lv) if text_hooks.is_placeholder(lv) else None
            if fakes is None:
                levels[lv] = c
            elif len(fakes) == 1:
                levels[fakes[0][0]] = c
            else:
                levels.update(dict(fakes))
        order = _written_order(p.params.get("levels"), labels)
        if mode == "sharing" and order:
            written_levels = {str(lv).strip() for lv in levels}
            order = [lv for lv in order if str(lv).strip() in written_levels]
        if order:
            rank = {}  # level -> its first place in the order (R7-07: a dict, not a list scan per level)
            for i, lv in enumerate(order):
                rank.setdefault(lv, i)
            ordered = sorted((lv for lv in levels if str(lv).strip() in rank), key=lambda lv: rank[str(lv).strip()])
            placed = set(ordered)
            ordered += [lv for lv in levels if lv not in placed]
        else:
            ordered = list(levels)
        entry["levels"] = {lv: small(levels[lv], k) for lv in ordered}
        if p.type in (ColumnType.ORDINAL, ColumnType.SCHOOL_YEAR) and order:
            entry["level_order"] = list(order)
    for key in ("min", "max", "mean", "median", "sd"):
        if key in syn:
            entry[key] = syn[key]
    if "length" in syn and p.type not in LABEL_SAFE_TYPES:
        entry["length"] = syn["length"]
    if "redacted" in syn:
        entry["note"] = syn["redacted"]
    fields = area_doc_fields(p, syn)
    if "readr" in fields:
        entry["readr"] = str(fields.pop("readr"))
    if "r_class" in fields:  # e.g. AREA A: epoch integers are instants (POSIXct), not the type's default Date
        entry["r_class"] = str(fields.pop("r_class"))
    hook_keys = []
    for key, val in fields.items():
        if key not in entry:
            entry[key] = val
            hook_keys.append(key)
    return entry, hook_keys


def note_pseudonyms(p: ColumnProfile, values: Any, counts: dict | None = None) -> None:
    """Record PRIVATELY how many rows of the written column ``values`` hold each pseudonym of ``p`` (D35), so the
    docs list the pseudonyms as the synthetic labels they are (:func:`_pseudonym_labels`)."""
    if p.type not in LABEL_SAFE_TYPES or p.model is None or not text_hooks.pseudonymized(p):
        return
    fakes = {str(f).strip() for f in (p.model.extra.get(text_hooks.PSEUDO_MAP_KEY) or {}).values()}
    if counts is None:
        counts = Counter(values.tolist() if hasattr(values, "tolist") else values)
    written: Counter = Counter()
    for value, count in counts.items():
        if isinstance(value, str) and value.strip() in fakes:
            written[value.strip()] += count
    p.model.extra[text_hooks.PSEUDO_WRITTEN_KEY] = dict(written)


def _pseudonym_labels(p: ColumnProfile) -> dict[str, list[tuple[str, int]]]:
    """Placeholder (``[pseudonymized 2]``) -> the pseudonyms written for it, each with its written rows (several
    for the case forms of one value), from the column's PRIVATE map; empty when the column has none."""
    rec = text_hooks.pseudonymized(p)
    if not rec or p.model is None:
        return {}
    mapping = p.model.extra.get(text_hooks.PSEUDO_MAP_KEY) or {}
    written = p.model.extra.get(text_hooks.PSEUDO_WRITTEN_KEY) or {}
    out: dict[str, list[tuple[str, int]]] = {}
    for real, fake in mapping.items():
        ph = rec.get(str(real).strip())
        fake = str(fake).strip()
        if ph is not None and fake not in (f for f, _ in out.get(ph, ())):
            out.setdefault(ph, []).append((fake, int(written.get(fake, 0))))
    return {ph: sorted(v) for ph, v in out.items()}


def _written_order(order: Any, labels: dict[str, list[tuple[str, int]]]) -> list[str] | None:
    """An ordinal's level order as written: each placeholder replaced by its pseudonym(s)."""
    if not order:
        return None
    out: list[str] = []
    for lv in order:
        if isinstance(lv, str) and text_hooks.is_placeholder(lv):
            if lv not in labels:
                return None  # no map (not generated): no usable order
            out += [f for f, _ in labels[lv]]
        else:
            out.append(lv)
    return out


def column_entry(p: ColumnProfile, syn: dict[str, Any], dtype: str | None, k: int) -> dict[str, Any]:
    """Dictionary entry of one column from its profile (types) and SYNTHETIC statistics."""
    return _entry_parts(p, syn, dtype, k)[0]


_R_ESCAPES = {"\\": "\\\\", '"': '\\"', "\t": "\\t", "\n": "\\n", "\r": "\\r"}


def _r_string(s: str) -> str:
    """An R string literal (tab, newline and other control characters escaped, e.g. ``delim = "\\t"``)."""
    out = []
    for ch in s:
        if ch in _R_ESCAPES:
            out.append(_R_ESCAPES[ch])
        elif ord(ch) < 32 or ord(ch) == 127:
            out.append(f"\\x{ord(ch):02x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _code(s: Any) -> str:
    """Markdown code span that survives backticks inside ``s``."""
    s = str(s)
    longest = run = 0
    for ch in s:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if s.startswith("`") or s.endswith("`") or not s.strip() else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


R_ENCODINGS = {"cp1252": "windows-1252", "latin-1": "ISO-8859-1", "utf-16-le": "UTF-16LE", "utf-16-be": "UTF-16BE",
               "utf-16": "UTF-16", "utf-8-sig": "UTF-8"}


def _out_format(fmt: str) -> str:
    try:
        from ..io import registry

        return registry.output_handler(fmt).key
    except Exception:  # noqa: BLE001 - unknown formats keep their key
        return fmt


def write_docs(out_dir: Path, written: list[WrittenFile], run_messages: list[Message], seed: int,
               min_cell: int, stale: list[str] | None = None, names: dict[str, str] | None = None,
               *, mode: str = "fidelity", not_written: tuple[int, int] = (0, 0)) -> dict[str, str]:
    """Write README.md, DATA_DICTIONARY.md and schema.json; returns doc kind -> file name.

    ``stale``: out-dir relative paths of files left from an earlier run whose
    source FAILed in this run (listed in the README; never deleted). ``names``
    (doc name -> file name) keeps the names of an earlier write. ``not_written``:
    (input files not written, input files found), counted in the sharing README.
    """
    files = [(w.result, _tables_of_written(w)) for w in written]
    return _write(out_dir, files, run_messages, seed, min_cell, stale or [], names, screen_of(run_messages), mode=mode,
                  not_written=not_written)


def not_written_count(run: Any) -> tuple[int, int]:
    """(input files not written, input files found): every FAILed file and every skipped file of the run."""
    skipped = len(getattr(run, "skipped", ()))  # an older run object may have none
    return (sum(1 for fr in run.files if fr.failed) + skipped, len(run.files) + skipped)


def rewrite_docs(run: Any, min_cell: int) -> dict[str, str]:
    """Rewrite the docs from the run results alone (written files only; same doc names)."""
    out_dir = Path(run.out_dir)
    files = [(fr, _tables_of_result(fr)) for fr in run.files if fr.written]
    # found at the real path, listed with the shown one (a flagged folder or file name is redacted, D34)
    written_now = {fr.out_rel_on_disk.casefold() for fr in run.files if fr.written}
    stale = [fr.out_rel_path for fr in run.files
             if fr.failed and fr.out_rel_on_disk.casefold() not in written_now
             and (out_dir / Path(*fr.out_rel_on_disk.split("/"))).is_file()]
    names = None
    if run.docs:
        inv = {v: k for k, v in DOC_KINDS.items()}
        names = {inv[k]: v for k, v in run.docs.items() if k in inv}
        if set(names) != set(DOC_NAMES):
            names = None
    return _write(out_dir, files, [], run.seed, min_cell, stale, names, screen_of(run.messages),
                  mode=getattr(run, "data_mode", "fidelity"), not_written=not_written_count(run))


def _write(out_dir: Path, files: list[tuple[FileResult, list[_DocTable]]], run_messages: list[Message], seed: int,
           min_cell: int, stale: list[str], names: dict[str, str] | None, screen: Any = None,
           *, mode: str = "fidelity", not_written: tuple[int, int] = (0, 0)) -> dict[str, str]:
    if names is None:
        # stale outputs (kept from an earlier run, listed in the README) are data files too: never overwritten
        names, msgs = doc_paths([fr.out_rel_path for fr, _ in files] + list(stale))
        run_messages.extend(msgs)
    schema: dict[str, Any] = {"generator": f"datanonymous {__version__}", "synthetic": True, "seed": seed,
                              "files": []}
    if mode == "sharing":
        schema.pop("seed")
        schema["data_mode"] = "sharing"
    dd: list[str] = [
        "# Data dictionary (SYNTHETIC DATA)", "",
        "Every statistic below is computed from the synthetic files in this folder, not from real data. "
        f"Counts below {min_cell} are shown as `<{min_cell}`. Levels are listed only for category-like columns "
        "(factor, ordered factor, boolean, organisational code, school year, constant, flag), whose labels are "
        "reproduced as they are; other text columns hold invented values. A real label is reproduced only when "
        f"it is a vocabulary word or shared by at least {min_cell} people in the run (distinct IDs; identical "
        "rows count once in a table without IDs); every other one is written as a consistent same-shape "
        "pseudonym, listed here as written and counted in `pseudonymized_levels`.", "",
    ]
    if mode == "sharing":
        dd = [
            "# Data dictionary (SYNTHETIC DATA)", "",
            "**SHARING MODE.** Statistics and listed labels describe only the synthetic output. "
            f"Counts below {min_cell} are displayed as `<{min_cell}`. Category-like columns list their "
            "written values; ordered levels include only values present in this output. Rare source labels "
            "and exact counts are privacy-transformed, including standard vocabulary. A value's popularity "
            "across other files does not authorize its retention in a sensitive fact table. "
            "Missing categories can still be valid in production; use a reviewed non-identifying codebook "
            "and independently invented edge-case fixtures.", "",
        ]
    else:
        dd.insert(2, "**FIDELITY MODE.** For invented-data and local testing; this folder is not an approved sharing package.")
        dd.insert(3, "")
    for fr, tables in files:
        file_entry = _file_layout(fr)
        file_entry["tables"] = []
        dd += [f"## {_code(fr.out_rel_path)}", ""]
        dd += _layout_lines(file_entry, fr)
        for dt in tables:
            entries, hooks = [], []
            for pos, idx in enumerate(dt.out_columns):
                p = dt.profile.columns[idx]
                dtype = dt.dtypes[pos] if dt.dtypes and pos < len(dt.dtypes) else None
                e, hk = _entry_parts(p, dt.stats.get(idx, {}), dtype, min_cell,
                                     file_entry["format"] == "csv", mode=mode)
                entries.append(e)
                hooks.append(hk)
            col_types = "".join(e["readr"] for e in entries)
            na = sorted(({tok for e in entries for tok in e["na_tokens"]} | {""}) - {NULL_DISPLAY})
            file_entry["tables"].append({
                "name": dt.name, "rows": dt.n_rows, "readr_col_types": col_types,
                "columns": [_schema_entry(e, hk) for e, hk in zip(entries, hooks)],
            })
            if dt.name:
                dd += [f"### Table {_code(dt.name)}", ""]
            dd += [f"- Rows: {dt.n_rows}; columns: {len(entries)}",
                   f"- readr `col_types`: {_code(col_types) if col_types else '(none)'}",
                   f"- Missing-value tokens written: {', '.join(_code(repr(t)) for t in na)}", "",
                   "| # | Column (original header) | Type | R class | readr | pandas dtype | Format / NA tokens "
                   "| Missing % | Distinct | Levels or summary (synthetic) | Flags |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for i, e in enumerate(entries, 1):
                miss = "" if e["missing_rate"] is None else f"{100 * float(e['missing_rate']):.1f}"
                dd.append("| " + " | ".join([
                    str(i), _md(_code(e["name"])), e["type"] + (f" ({e['subtype']})" if e["subtype"] else ""),
                    e["r_class"], e["readr"], e["pandas_dtype"] or "", _md(_format_cell(e)), miss,
                    "" if e["distinct"] is None else str(e["distinct"]), _md(_summary(e)), _flags(e),
                ]) + " |")
            dd.append("")
            dd += _snippets(fr, dt, file_entry, col_types, na)
            for e in entries:
                if e.get("level_order"):
                    levels = ", ".join(_r_string(str(x)) for x in e["level_order"])
                    dd += [f"Ordered levels of {_code(e['name'])} (R): "
                           f"{_code(f'factor(x, levels = c({levels}), ordered = TRUE)')}", ""]
        schema["files"].append(file_entry)
    readme = _readme(files, seed, names, stale, min_cell, mode=mode, not_written=not_written)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = {"README.md": readme.encode("utf-8"), "DATA_DICTIONARY.md": "\n".join(dd).encode("utf-8"),
            "schema.json": (json.dumps(schema, indent=2, ensure_ascii=False) + "\n").encode("utf-8")}
    for doc, data in docs.items():
        if screen is not None:  # D34: a flagged header, sheet, file or folder name is never printed
            data = screen.redact_bytes(doc, data)
        # replace_bytes: an entry left at a doc name (a link, a hard link) is replaced, never written through
        replace_bytes(out_dir / names[doc], data)
    return {DOC_KINDS[d]: names[d] for d in DOC_NAMES}


#: Area hook fields that are statistics or prose, not types or formats: DATA_DICTIONARY.md only.
#: (C: the median length, item count and bounding box of the synthetic values, and the generation note.)
SCHEMA_HOOK_EXCLUDED = frozenset({"note", "char_length_p50", "n_items", "bounding_box"})


def _schema_entry(e: dict[str, Any], hook_keys: list[str]) -> dict[str, Any]:
    """The schema.json entry: types and formats only (D20/D21: no statistics, synthetic or real)."""
    out = {k: e[k] for k in SCHEMA_KEYS if k in e}
    for k in hook_keys:
        if k in SCHEMA_HOOK_EXCLUDED:
            continue
        v = e[k]
        if k == "formats" and isinstance(v, dict):
            v = list(v)  # AREA A mixed formats: the formats, without their (synthetic) row shares
        out[k] = v
    return out


def _file_layout(fr: FileResult) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": fr.out_rel_path, "format": _out_format(fr.format)}
    lay_src = fr.out_layout or fr.layout
    if lay_src is not None:
        lay = lay_src.to_public()
        entry["encoding"] = lay.get("encoding")
        entry["bom"] = bool(lay.get("bom"))
        entry["compression"] = lay.get("compression")
        if "text" in lay:
            t = lay["text"]
            entry["delimiter"] = t["delimiter"]
            entry["line_ending"] = t["line_ending"]
            entry["quoting"] = t["quoting"]
            entry["header_row"] = t["has_header"]
        footer = (lay.get("extra") or {}).get("n_footer_lines")
        if footer:  # K-17: note lines below the data of a CSV (placeholder text), not data rows
            entry["footer_lines"] = int(footer)
    return entry


def _layout_lines(f: dict[str, Any], fr: FileResult) -> list[str]:
    parts = [f"format {f['format']}"]
    if f.get("encoding"):
        parts.append(f"encoding {f['encoding']}")
    if "bom" in f:
        parts.append("byte-order mark" if f["bom"] else "no byte-order mark")
    if f.get("compression"):
        parts.append(f"compression {f['compression']}")
    if "delimiter" in f:
        d = f["delimiter"]
        parts.append(f"delimiter {_code(repr(d)) if d is not None else 'none (one column)'}")
        parts.append(f"line ending {f['line_ending']}")
        parts.append(f"quoting {f['quoting']}")
    if f.get("footer_lines"):
        parts.append(f"{f['footer_lines']} footer note line(s) below the data (no data rows: the read code below "
                     "stops after the last data row)")
    return [f"- File: {', '.join(parts)}"]


def _snippets(fr: FileResult, dt: _DocTable, f: dict[str, Any], col_types: str, na: list[str]) -> list[str]:
    path = fr.out_rel_path
    fmt = f["format"]
    enc = (f.get("encoding") or "utf-8").lower()
    if fmt == "csv":
        delim = f.get("delimiter") or ","
        na_r = ", ".join(_r_string(x) for x in na)
        locale = ""
        if enc not in ("utf-8", "utf8", "ascii", "utf-8-sig"):
            locale = f", locale = readr::locale(encoding = {_r_string(R_ENCODINGS.get(enc, enc))})"
        py_enc = "" if enc in ("utf-8", "utf8", "ascii") else f", encoding={enc!r}"
        # K-17: footer note lines below the data are no rows: read exactly the data rows (blank lines are skipped)
        r_rows, py_rows = (f", n_max = {dt.n_rows}", f", nrows={dt.n_rows}") if f.get("footer_lines") else ("", "")
        return ["R (readr):", "", "```r",
                f"df <- readr::read_delim({_r_string(path)}, delim = {_r_string(delim)}, "
                f"col_types = {_r_string(col_types)}, na = c({na_r}){r_rows}{locale})", "```", "",
                "pandas (all columns as text, NA tokens kept as written):", "", "```python",
                f"df = pd.read_csv({path!r}, sep={delim!r}, dtype=str, keep_default_na=False{py_rows}{py_enc})",
                "```", ""]
    sheet = dt.name
    snippets = {
        "xlsx": (f"df <- readxl::read_excel({_r_string(path)}" + (f", sheet = {_r_string(sheet)}" if sheet else "")
                 + ")", f"df = pd.read_excel({path!r}" + (f", sheet_name={sheet!r}" if sheet else "") + ", dtype=str)"),
        "parquet": (f"df <- arrow::read_parquet({_r_string(path)})", f"df = pd.read_parquet({path!r})"),
        "feather": (f"df <- arrow::read_feather({_r_string(path)})", f"df = pd.read_feather({path!r})"),
        "dta": (f"df <- haven::read_dta({_r_string(path)})", f"df = pd.read_stata({path!r})"),
        "json": (f"df <- jsonlite::fromJSON({_r_string(path)})", f"df = pd.read_json({path!r}, dtype=False)"),
        "jsonl": (f"df <- jsonlite::stream_in(file({_r_string(path)}))",
                  f"df = pd.read_json({path!r}, lines=True, dtype=False)"),
        "html_xls": (f"df <- rvest::html_table(rvest::read_html({_r_string(path)}))[[1]]",
                     f"df = pd.read_html({path!r})[0]"),
    }
    if fmt not in snippets:
        return []
    r, py = snippets[fmt]
    return ["R:", "", "```r", r, "```", "", "pandas:", "", "```python", py, "```", ""]


def _format_cell(e: dict[str, Any]) -> str:
    parts = []
    if e.get("format"):
        parts.append(f"format {_code(e['format'])}")
    if isinstance(e.get("formats"), dict):  # AREA A mixed formats with their synthetic row shares
        parts.append("formats " + " / ".join(
            _code(k) + (f" ({100 * float(v):.0f}%)" if isinstance(v, (int, float)) else "")
            for k, v in e["formats"].items()))
    if "decimals" in e:
        parts.append(f"{e['decimals']} decimal(s)")
    if e.get("float_text"):
        parts.append("written with a decimal point (e.g. 72.0)")
    if e.get("plus_sign"):
        parts.append("leading + kept")
    if "width" in e:
        parts.append(f"width {e['width']}")
    if e.get("zero_padded"):
        parts.append("zero-padded")
    if e.get("shapes"):
        parts.append("shape " + " / ".join(_code(s) for s in e["shapes"]))
    if e.get("boolean_tokens"):
        parts.append("tokens " + " / ".join(_code(t) for t in e["boolean_tokens"]))
    if e.get("na_token_counts"):
        toks = ", ".join(f"{_code(repr(t))} ({c})" for t, c in e["na_token_counts"].items())
        parts.append(f"NA: {toks}")
    return "; ".join(parts)


def _flags(e: dict[str, Any]) -> str:
    out = []
    if e["role"] == "id":
        out.append("ID (row numbers 1..n kept)" if e.get("row_numbers_kept") else "ID (synthetic)")
    elif e["role"] == "entity":
        out.append("constant within ID")
    if e["rare_category_adjusted"]:
        out.append("rare_category_adjusted")
    if e["extreme_value_adjusted"]:
        out.append("extreme_value_adjusted")
    if e.get("pseudonymized_levels"):
        out.append(f"pseudonymized_levels ({e['pseudonymized_levels']})")
    return ", ".join(out)


def _summary(e: dict[str, Any]) -> str:
    if "levels" in e:
        items = list(e["levels"].items())
        text = "; ".join(f"{k!r}: {v}" for k, v in items[:12])
        text += f"; ... ({len(items)} levels)" if len(items) > 12 else ""
        return text
    if "min" in e:
        return f"min {e['min']}, median {e.get('median')}, mean {e.get('mean')}, max {e['max']}"
    if "length" in e and isinstance(e["length"], dict):
        return f"length {e['length'].get('min')}-{e['length'].get('max')}"
    if "note" in e:
        return str(e["note"])
    return ""


# --------------------------------------------------------------------------- README
def _sharing_readme(files: list[tuple[FileResult, list[_DocTable]]], names: dict[str, str], min_cell: int,
                    not_written: tuple[int, int] = (0, 0)) -> str:
    """Describe the sharing contract without exporting source profiles or reproduction details.

    ``not_written`` is counted, never named: an input file name can hold personal data.
    """
    lines = [
        "# SYNTHETIC DATA", "",
        "> **SYNTHETIC DATA - NOT REAL RECORDS. SHARING MODE.** People, table memberships, and person "
        "relationships are fictional. Use these files for coding fixtures, not student decisions or "
        "real-data analysis. A successful run does not automatically establish FERPA de-identification "
        "or authorize disclosure.", "",
        "## How it was generated", "",
        f"- Generator: datanonymous {__version__} (Faker {_faker_version()} pinned), sharing mode.",
        "- Inputs were processed locally. Source reports, source profiles, mappings, assignment information, "
        "and reproduction settings are excluded from this folder.",
        f"- `{names['DATA_DICTIONARY.md']}` and `{names['schema.json']}` describe output values, types, "
        "and formats; their statistics do not describe the original records.", "",
        "## Files", "",
    ]
    if files:
        lines += ["| File | Format | Rows | Columns |", "|---|---|---|---|"]
        for fr, tables in files:
            rows = ", ".join(str(t.n_rows) for t in tables)
            cols = ", ".join(str(len(t.out_columns)) for t in tables)
            lines.append(f"| {_md(_code(fr.out_rel_path))} | {_out_format(fr.format)} | {rows} | {cols} |")
    else:
        lines.append("No synthetic data file was written in this run.")
    missing, found = not_written
    if missing:
        lines += ["", f"**{missing} of {found} input files were not written** to this folder (they could not be "
                  "generated safely, or are not supported tables). Fictional keys and joins hold only among "
                  "the files in this folder."]
    changed = _changed_lines(files)
    if changed:
        lines += ["", "## What changed", "", "Deliberate changes that make these files differ from real data:", ""] + changed
    lines += [
        "", "## Sharing contract", "",
        "- Each released table keeps its row total: N output equals N input. Safe paths, headers, column "
        "order, types, formats, and missing-value semantics support production readers.",
        "- Fictional keys retain valid joins, one-to-one aliases, nesting, and composite uniqueness. "
        "Within-person constants remain constant for fictional people. Original memberships and "
        "family or staff relationships are regenerated.",
        "- Primary-person counts are reassigned independently per fact table; secondary relationships "
        "are regenerated. A count that identifies its original donor must be blurred or refused.",
        f"- Rare facts held by fewer than {min_cell} distinct people in a fact table lose exact counts and "
        "rare real labels, including standard vocabulary. Person-level dates exclude exact source extrema; "
        "month and academic season are preserved where feasible.",
        "- Row-number IDs 1..n have randomized assignment. Personal-data fields are regenerated; synthetic "
        "phones must not match a source phone anywhere in the run.",
        "- An input file with a privacy warning or unverified output is left out of this folder. Only a "
        "validated inventory of synthetic data and reviewed documentation is written here.", "",
        "## Limits and code use", "",
        "- Fields are generally generated independently. Ordinary correlations, subgroup effects, source "
        "timelines, and original person relationships are not analytical evidence. Only documented "
        "structural contracts and declared rules support code compatibility.",
        "- Validate production readers, joins, date logic, and missing values separately. Handle valid "
        "categories absent from this copy using a reviewed non-identifying codebook and invented fixtures; "
        "do not derive permanent enums or filters solely from observed synthetic values.",
        "- The tool does not provide formal differential privacy or an automated assessment of repeated "
        "releases. Joined indirect identifiers, counts, numeric extremes, metadata, prior copies, agent "
        "transcripts, and other reasonably available information still require disclosure review.",
        "- The institution determines authorized source use, recipients, retention, and whether this exact "
        "folder meets the applicable disclosure standard. Automated checks are technical evidence, "
        "not automatic FERPA approval.", "",
    ]
    return "\n".join(lines)


def _changed_lines(files: list[tuple[FileResult, list[_DocTable]]]) -> list[str]:
    """The transformations visible in the output, one bullet each: no counts, reasons or source facts."""
    from .changes import file_changes

    out = []
    for fr, _ in files:
        if fr.withheld:
            continue
        for column, text in file_changes(fr, counts=False):
            out.append(f"- {_md(_code(fr.out_rel_path))}" + (f" {_md(_code(column))}" if column else "") + f": {text}")
    return out


def _faker_version() -> str:
    try:
        return md.version("Faker")
    except md.PackageNotFoundError:  # pragma: no cover
        return "not installed"


def _kept_row_numbers(files: list[tuple[FileResult, list[_DocTable]]]) -> list[str]:
    """``file / column`` of every ID column that keeps its row numbers 1..n (D9: they carry no identity)."""
    out = []
    for fr, tables in files:
        for t in tables:
            for idx in t.out_columns:
                p = t.profile.columns[idx]
                if p.role == "id" and p.params.get("_identity_mapped"):
                    out.append(f"{fr.out_rel_path}{' / ' + t.name if t.name else ''} :: {p.name}")
    return out


def _readme(files: list[tuple[FileResult, list[_DocTable]]], seed: int, names: dict[str, str],
            stale: list[str], min_cell: int = 10, *, mode: str = "fidelity",
            not_written: tuple[int, int] = (0, 0)) -> str:
    if mode == "sharing":
        return _sharing_readme(files, names, min_cell, not_written)
    kept_ids = _kept_row_numbers(files)
    lines = [
        "# SYNTHETIC DATA",
        "",
        "> **SYNTHETIC DATA - NOT REAL RECORDS.** Every data file in this folder was generated by datanonymous. "
        "These generated fixtures are for invented-data and local testing. Personal-data fields are regenerated"
        + (" (row-number ID columns that run 1..n keep those numbers; listed below)"
           if kept_ids else "")
        + ". Validate code compatibility separately; results computed from them are not real-data analysis. "
        "**FIDELITY MODE.** Exact source counts and extrema may survive. This folder is not an approved "
        "sharing package and these checks do not establish de-identification.",
        "",
        "## How it was generated",
        "",
        f"- Generator: datanonymous {__version__} (Faker {_faker_version()} pinned), seed {seed}. The same tool "
        f"versions, the same input and `--seed {seed}` reproduce these files byte for byte.",
        "- Each source file was profiled column by column and replaced by a synthetic copy with the same relative "
        "path and file name, the same header text and column order, the same number of rows (N rows in = N rows "
        "out) and the same file layout (encoding, delimiter, quoting, line endings, missing-value tokens).",
        "- Every non-ID column was generated on its own from the source column's distribution: exact counts for "
        "categories, flags and missing-value tokens; interpolated values within the observed range for continuous "
        "numbers and dates; invented same-shape values for IDs, names, emails, phone numbers and other personal "
        "text; filler text for free text.",
        f"- These docs ({names['README.md']}, {names['DATA_DICTIONARY.md']}, {names['schema.json']}) were computed "
        "from the synthetic files only.",
        "",
        "## Files",
        "",
    ]
    if files:
        lines += ["| File | Format | Rows | Columns | Notes |", "|---|---|---|---|---|"]
        for fr, tables in files:
            rows = ", ".join(str(t.n_rows) for t in tables)
            cols = ", ".join(str(len(t.out_columns)) for t in tables)
            lines.append(f"| {_md(_code(fr.out_rel_path))} | {_out_format(fr.format)} | {rows} | {cols} | "
                         f"{_md(_file_notes(tables, _file_layout(fr).get('footer_lines')))} |")
    else:
        lines.append("No synthetic data file was written in this run.")
    lines += [
        "",
        f"Column types, formats and synthetic summaries: `{names['DATA_DICTIONARY.md']}`; machine-readable "
        f"types and formats: `{names['schema.json']}`.",
        "",
        "## What IS preserved",
        "",
        "- File names and relative paths, header text (exactly), column order and row counts.",
        "- File layout: encoding, byte-order mark, delimiter, quoting, line endings, sheet/table structure, and "
        "the missing-value tokens (`NA`, blanks, ...) at their exact rates.",
        "- Value formats: decimals, zero padding, date and time formats, boolean words, whitespace variants.",
        "- Each column's own distribution: exact level counts for categories, booleans and ordered levels; "
        "observed values only for discrete integers; interpolated values within the observed range for "
        "continuous numbers and dates (the exact minimum and maximum appear, unless `--no-exact-extremes` turned "
        "this off for a column).",
        "- Every source category level and missing-value token appears at least once (`rare_category_adjusted` "
        "marks columns where this changed a count).",
        "- Replacement IDs keep their shape (width, zero padding, prefix) and avoid detected source IDs"
        + (" (except row-number values listed below)" if kept_ids else "")
        + "; each synthetic ID has the row "
        "count of one real ID, and the same real ID gets the same synthetic ID in every file of the run, so joins "
        "between files work.",
        "- Structure that holds in every source row: ID aliases and nesting, the same IDs across files (join and "
        "orphan rates), no repeated (ID, date) pairs, the sort order and the number of duplicate rows.",
    ]
    if kept_ids:
        lines.append("- Row-number ID columns (values exactly 1..n) keep their numbers and can retain assignment fingerprints: "
                     + ", ".join(_code(k) for k in kept_ids) + ".")
    lines += [
        "- Columns that are constant within an ID (such as gender or date of birth) stay constant within each "
        "synthetic ID.",
        "- Whole-row patterns of missing values: columns that are blank together stay blank together (patterns "
        f"held by fewer than {min_cell} source rows are pooled).",
    ]
    kept = _structure_lines(files)
    lines += kept
    lines += [
        "",
        "## What is NOT preserved",
        "",
        "- Relationships between columns: every non-ID column is generated independently of the others. "
        "Correlations (between scores), group differences (a score gap by gender or lunch status) and "
        "cross-column logic (a later year being >= an earlier year, a school name matching a school ID, an age "
        "plausible for a grade) do not hold" + (", except the structural rules listed above." if kept else "."),
        "- Panel and time structure: attendance spells, attributes that change across years and the order of "
        "events within an ID are not modelled.",
    ]
    lines += _detected_not_kept(files)
    lines += [
        "- Cross-column rules that the report only suggests (they apply only when declared under `rules:` in "
        "`--config`), and aggregate suppression rules (small counts hidden in a source).",
        "- Real IDs, names, emails, phone numbers, SSNs, addresses and free text: replaced by same-shape IDs, "
        "invented values or filler text. Real labels of high-cardinality text columns: replaced by invented "
        "labels with the same frequency profile.",
        f"- Rare real labels: a text value is kept only when it is a vocabulary word or shared by at least {min_cell} "
        "people in the run (distinct IDs; identical rows count once in a table without IDs); any other value is "
        "written as a same-shape pseudonym with its exact count, the same pseudonym in every column and file.",
        "- This is not formal differential privacy. The exact minimum and maximum of numeric and date columns are "
        "emitted, so a unique extreme value can equal one real person's value.",
    ]
    lines += _not_reproduced(files)
    if stale:
        lines += ["", "## Not regenerated in this run", "",
                  "**Not regenerated in this run** (left from an earlier run; their source failed this time):", ""]
        lines += [f"- {_code(p)}" for p in stale]
    lines.append("")
    return "\n".join(lines)


def _file_notes(tables: list[_DocTable], footer_lines: int | None = None) -> str:
    notes = []
    if footer_lines:  # K-17: a CSV's note lines below the data are no rows
        notes.append(f"{footer_lines} footer note line(s) below the data (not rows)")
    for t in tables:
        cols = t.profile.columns
        prefix = f"{t.name}: " if t.name else ""
        ids = [p.name for p in cols if p.role == "id"]
        const = [p.name for p in cols if p.role == "entity"]
        if ids:
            notes.append(prefix + "ID " + ", ".join(ids))
        if const:
            notes.append(prefix + "constant within ID: " + ", ".join(const))
        adjusted = sum(1 for p in cols if p.role != "excluded" and any(p.flags.values()))
        if adjusted:
            notes.append(prefix + f"{adjusted} adjusted column(s)")
    return "; ".join(notes)


def _structure_lines(files: list[tuple[FileResult, list[_DocTable]]]) -> list[str]:
    out = []
    for fr, tables in files:
        for t in tables:
            s = t.profile.structure
            kinds = [(len(s.functional_dependencies), "functional dependency(ies) (one column determines another)"),
                     (len(s.derived_templates), "derived template(s) (a column built from others)"),
                     (len(s.composite_keys), "composite key(s) kept unique")]
            found = [f"{n} {what}" for n, what in kinds if n]
            if found:
                where = fr.out_rel_path + (f" / {t.name}" if t.name else "")
                out.append(f"- Structural rules applied in {_code(where)}: " + "; ".join(found) + ".")
    return out


def _detected_not_kept(files: list[tuple[FileResult, list[_DocTable]]]) -> list[str]:
    """Per-file panels / slowly-changing patterns detected in the source (D16: listed as not preserved; counts)."""
    out = []
    for fr, tables in files:
        for t in tables:
            s = t.profile.structure
            where = fr.out_rel_path + (f" / {t.name}" if t.name else "")
            if s.panels:  # D16: the README lists them as not preserved
                out.append(f"  - {_code(where)}: {len(s.panels)} panel / slowly-changing pattern(s) detected in the "
                           "source; generated under the independent model (only composite-key uniqueness is kept).")
    return out


def _not_reproduced(files: list[tuple[FileResult, list[_DocTable]]]) -> list[str]:
    out = []
    for fr, tables in files:
        for t in tables:
            dropped = [p.name for p in t.profile.columns if p.role == "excluded"]
            nulled = [p.name for p in t.profile.columns if p.role == "null"]
            where = fr.out_rel_path + (f" / {t.name}" if t.name else "")
            if dropped:
                out.append(f"- {_code(where)}: dropped with `--exclude` (not in the file): "
                           + ", ".join(_code(n) for n in dropped))
            if nulled:
                out.append(f"- {_code(where)}: written as all-missing with `--null`: "
                           + ", ".join(_code(n) for n in nulled))
    return (["", "## Columns not reproduced", ""] + out) if out else []
