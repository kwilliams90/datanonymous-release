"""Flat JSON (top-level array of records) and JSON-lines reader/writer.

AREA F owns this module (design 6 "JSON/JSONL", section 4 nested values).

Reading
-------
The text is decoded like CSV (BOM, strict UTF-8, cp1252, latin-1; ``--encoding``)
and parsed with the standard ``json`` decoder, never through pandas type
inference: numbers keep their exact source text (``12``, ``1.50``, ``1e5``),
so integer vs float spelling survives. Columns are the record keys in order of
first appearance. Cell text: strings as they are; numbers as their source text;
``true``/``false``; JSON ``null`` -> ``None``; a key absent from a record ->
:data:`MISSING_KEY` (a blank token, so it counts as missing and is written back
as an absent key at its exact rate). Per column the JSON value kinds are kept
(``string``/``number``/``bool``/``const`` for ``NaN``/``Infinity``; the constants
are numbers stored natively, ``model.NUMERIC_SPECIALS_KEY``, kept verbatim at
their exact counts like ``NaN`` and left out of the statistics, K-20); the writer
emits a value bare (unquoted) only when its column held that kind and the text
has that kind's grammar, so ``"0123"`` stays a string and ``12`` stays a number.
A column holding both a number and a string that looks like a JSON number (or
``true`` and ``"true"``) is ambiguous and FAILs with a fix.

Nested values (objects/arrays) are never stringified or copied: the file FAILs
before writing with ``--null COL`` / ``--exclude COL`` (a column whose values are
all null/absent is read). A top-level object or an array of scalars is not a
table (listed as skipped, D18); invalid JSON, a non-object record or a repeated
key FAILs. A ``.json`` file holding JSON lines (NDJSON exports) is read and
written back as JSON lines under its own name (info ``json_lines_content``).
JSON lines may mix CRLF and LF (warning; the majority ending is written).

Formatting is mirrored from the source text: the characters before the first
record (``[`` plus indentation), between records, after the last one (``]``,
trailing newline), and inside a record (after ``{``, around ``:``, between items,
before ``}``) come from the first records, together with ``ensure_ascii``,
escaped ``/`` and the hex case of ``\\u`` escapes, so ``json.dumps(indent=2)``,
compact ``separators=(',', ':')``, pandas ``to_json`` and one-record-per-line
arrays are reproduced. ``format_exact`` in the layout says whether re-serialising
the source with that template gives the source text back byte for byte (if not,
an info message says the formatting is approximate).

JSON lines may be compressed (``.jsonl.gz`` ... via ``io/compress.py``).
"""

from __future__ import annotations

import heapq
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import NotTabular, ReadError, UnsupportedColumnError, WriteError
from ..model import NUMERIC_SPECIALS_KEY, FileLayout, Message, RawColumn, RawTable, ReadResult
from . import arrowio, compress, sniff

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ReadRequest, WriteRequest

#: Cell text of a key absent from a record (whitespace for ``str.strip``: a blank, i.e. missing, token).
MISSING_KEY = "\x1f"
STYLE_KEY = "_json_style"
KINDS_KEY = "_json_kinds"
NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
CONSTANTS = ("NaN", "Infinity", "-Infinity")
_WS = re.compile(r"[ \t\n\r]*")
_ITEM_SEP = re.compile(r"[ \t\n\r]*,[ \t\n\r]*")
_KEY_SEP = re.compile(r"[ \t\n\r]*:[ \t\n\r]*")
_PANDAS_COMPRESSION = {"gz": "gzip", "zip": "zip", "bz2": "bz2", "xz": "xz"}


class _Num(str):
    """A JSON number, kept as its exact source text."""


class _Const(str):
    """``NaN`` / ``Infinity`` / ``-Infinity`` (Python's json extension)."""


class _Pairs(list):
    """A JSON object as its (key, value) pairs in source order."""


def _decoder() -> json.JSONDecoder:
    return json.JSONDecoder(parse_float=_Num, parse_int=_Num, parse_constant=_Const, object_pairs_hook=_Pairs)


def _decode_error(e: json.JSONDecodeError, what: str = "JSON") -> ReadError:
    return ReadError(f"invalid {what}: {e.msg} at line {e.lineno} column {e.colno}",
                     fix="repair the file (it must be valid JSON) and re-run")


# --------------------------------------------------------------------------- reading
def _scan_array(text: str, dec: json.JSONDecoder) -> tuple[list[tuple[Any, int, int]], list[str], str, str]:
    """Records with their spans, the separators between them, and the text before/after them."""
    pos = _WS.match(text, 0).end()
    if pos >= len(text):
        raise ReadError("the file holds no JSON value", fix="repair the file (it must be valid JSON) and re-run")
    if text[pos] != "[":
        try:
            dec.raw_decode(text, pos)
        except json.JSONDecodeError as e:
            raise _decode_error(e) from None
        raise NotTabular("not a table: the JSON value is not an array of records")
    ws = _WS.match(text, pos + 1).end()
    if text[ws:ws + 1] == "]":
        rest = text[ws + 1:]
        if rest.strip():
            raise ReadError("data after the JSON array", fix="repair the file (one JSON array of records)")
        return [], [], text[:ws], text[ws:]
    records: list[tuple[Any, int, int]] = []
    seps: list[str] = []
    cur = ws
    while True:
        try:
            obj, end = dec.raw_decode(text, cur)
        except json.JSONDecodeError as e:
            raise _decode_error(e) from None
        if not isinstance(obj, _Pairs):
            if not records:
                raise NotTabular("not a table: a JSON array of values, not of records")
            raise ReadError(f"element {len(records) + 1} of the JSON array is not a record (object)",
                            fix="every element must be a flat JSON object")
        records.append((obj, cur, end))
        m = _ITEM_SEP.match(text, end)
        if m:
            seps.append(m.group())
            cur = m.end()
            continue
        close = _WS.match(text, end).end()
        if text[close:close + 1] != "]":
            raise ReadError("malformed JSON array (missing ',' or ']')", fix="repair the file and re-run")
        if text[close + 1:].strip():
            raise ReadError("data after the JSON array", fix="repair the file (one JSON array of records)")
        return records, seps, text[:ws], text[end:]


def _holds_json_lines(text: str, dec: json.JSONDecoder) -> bool:
    """True when ``text`` is JSON lines (an object, a line break, another object ...) rather than one value."""
    pos = _WS.match(text, 0).end()
    if text[pos:pos + 1] != "{":
        return False
    try:
        obj, end = dec.raw_decode(text, pos)
    except json.JSONDecodeError:
        return False
    nxt = _WS.match(text, end).end()
    return isinstance(obj, _Pairs) and text[nxt:nxt + 1] == "{" and "\n" in text[end:nxt]


def line_ending(text: str) -> tuple[str, bool]:
    """(majority line ending, mixed?) of a text: CRLF vs bare LF counted in the text, a tie -> the first one."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    if crlf != lf:
        return ("\r\n" if crlf > lf else "\n"), bool(crlf and lf)
    first = text.find("\n")
    return ("\r\n" if first > 0 and text[first - 1] == "\r" else "\n"), bool(crlf and lf)


def _scan_lines(text: str, dec: json.JSONDecoder) -> tuple[list[tuple[Any, int, int]], list[str], str, str, int]:
    """JSON lines: records with spans, separators, prefix, suffix and the number of dropped blank lines.

    Lines end at LF; a CR right before it belongs to the line ending (CRLF and LF may be mixed).
    """
    records: list[tuple[Any, int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        line_end = n if nl < 0 else nl
        if line_end > pos and text[line_end - 1] == "\r":
            line_end -= 1
        line = text[pos:line_end]
        if line.strip():
            start = pos + (len(line) - len(line.lstrip(" \t")))
            try:
                obj, end = dec.raw_decode(text, start)
            except json.JSONDecodeError as e:
                raise _decode_error(e, "JSON lines") from None
            if end > line_end:
                raise ReadError(f"the record starting on line {text.count(chr(10), 0, pos) + 1} spans several "
                                "lines", fix="write one record per line (JSON lines), or one JSON array of records "
                                             "([{...}, {...}]) in a .json file")
            if text[end:line_end].strip(" \t"):
                raise ReadError(f"line {text.count(chr(10), 0, pos) + 1} holds more than one JSON value",
                                fix="put exactly one JSON object on each line")
            if not isinstance(obj, _Pairs):
                if not records:
                    raise NotTabular("not a table: JSON lines that are not records (objects)")
                raise ReadError(f"line {text.count(chr(10), 0, pos) + 1} is not a JSON object",
                                fix="put exactly one flat JSON object on each line")
            records.append((obj, start, end))
        pos = nl + 1 if nl >= 0 else n
    if not records:
        return [], [], "", text, 0
    seps = [text[records[i][2]:records[i + 1][1]] for i in range(len(records) - 1)]
    dropped = sum(max(s.count("\n") - 1, 0) for s in seps)
    return records, seps, text[:records[0][1]], text[records[-1][2]:], dropped


def _record_seps(text: str, start: int, end: int, dec: json.JSONDecoder) -> dict[str, str | None]:
    """Whitespace/punctuation inside one record: after '{', around ':', between items, before '}'."""
    p = start + 1
    w = _WS.match(text, p).end()
    out: dict[str, str | None] = {"open_ws": text[p:w], "key_sep": None, "item_sep": None, "close_ws": None}
    if text[w:w + 1] == "}":
        out["empty"] = text[start:end]
        return out
    _, kend = dec.raw_decode(text, w)
    m = _KEY_SEP.match(text, kend)
    if m is None:
        return out
    out["key_sep"] = m.group()
    _, vend = dec.raw_decode(text, m.end())
    m2 = _ITEM_SEP.match(text, vend)
    if m2:
        out["item_sep"] = m2.group()
    j = end - 1
    i = j
    while i > start and text[i - 1] in " \t\n\r":
        i -= 1
    out["close_ws"] = text[i:j]
    return out


def _style(text: str, records: list[tuple[Any, int, int]], seps: list[str], prefix: str, suffix: str,
           dec: json.JSONDecoder, lines: bool) -> dict[str, Any]:
    style: dict[str, Any] = {"prefix": prefix, "suffix": suffix, "empty": "{}", "open_ws": "", "key_sep": ": ",
                             "item_sep": None, "close_ws": ""}
    if lines:  # one record per line (majority line ending); blank lines between records are dropped (warned)
        style["record_sep"] = line_ending(text)[0]
    else:
        style["record_sep"] = Counter(seps).most_common(1)[0][0] if seps else ","
    got_key = got_item = got_empty = False
    for _, start, end in records:
        r = _record_seps(text, start, end, dec)
        if "empty" in r and not got_empty:
            style["empty"] = r["empty"]
            got_empty = True
            continue
        if r["key_sep"] is not None and not got_key:
            style.update({"open_ws": r["open_ws"], "key_sep": r["key_sep"], "close_ws": r["close_ws"] or ""})
            got_key = True
        if r["item_sep"] is not None and not got_item:
            style["item_sep"] = r["item_sep"]
            got_item = True
        if got_key and got_item and got_empty:
            break
    if style["item_sep"] is None:
        ow = style["open_ws"]
        style["item_sep"] = ("," + ow) if "\n" in ow else (", " if style["key_sep"].endswith(" ") else ",")
    non_ascii = any(ord(ch) > 127 for ch in text)
    escaped = re.search(r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})", text)
    style["ensure_ascii"] = (not non_ascii) if (non_ascii or escaped) else True
    style["escape_slash"] = "\\/" in text
    hexes = re.findall(r"\\u([0-9a-fA-F]{4})", text)
    style["upper_hex"] = bool(hexes) and any(c in "ABCDEF" for h in hexes for c in h) and not any(
        c in "abcdef" for h in hexes for c in h)
    return style


def _cell(v: Any) -> tuple[str | None, str | None]:
    """(cell text, JSON kind) of one value; kind 'nested' for objects/arrays."""
    if v is None:
        return None, None
    if isinstance(v, list):  # _Pairs or array
        return None, "nested"
    if v is True or v is False:
        return ("true" if v else "false"), "bool"
    if isinstance(v, _Const):
        return str(v), "const"
    if isinstance(v, _Num):
        return str(v), "number"
    return str(v), "string"


def _read_text(req: "ReadRequest") -> tuple[str, FileLayout, list[Message]]:
    data = compress.read_bytes(req.path, req.compression) if req.compression else Path(req.path).read_bytes()
    enc = sniff.sniff_encoding(data, req.settings.encoding)
    text = sniff.decode(data, enc)
    layout = FileLayout(format="", encoding=enc.encoding, bom=enc.bom, encoding_confidence=enc.confidence,
                        compression=req.compression)
    messages = list(enc.messages)
    if req.compression:
        layout.extra.update(compress.container_layout(req.path, req.compression))
        if req.purpose == "source":  # a stored gzip/zip name is free text: never copied
            messages.extend(compress.container_messages(layout.extra))
    return text, layout, messages


def _read(req: "ReadRequest", lines: bool) -> ReadResult:
    text, layout, messages = _read_text(req)
    layout.format = "jsonl" if lines else "json"
    dec = _decoder()
    le, mixed = line_ending(text)
    if not lines and _holds_json_lines(text, dec):
        # NDJSON saved as .json (common for exports): read and written back as JSON lines, same name
        lines = True
        messages.append(Message("info", "json_lines_content",
                                "the .json file holds JSON lines (one record per line): it is read and written "
                                "as JSON lines"))
    if lines:
        records, seps, prefix, suffix, dropped = _scan_lines(text, dec)
        if dropped:
            messages.append(Message("warning", "blank_lines_dropped",
                                    f"{dropped} blank line(s) between records were dropped"))
        if mixed:
            messages.append(Message("warning", "line_endings_mixed",
                                    "the file mixes CRLF and LF line endings; the output uses the majority ending ("
                                    + ("CRLF" if le == "\r\n" else "LF") + ") between records"))
    else:
        records, seps, prefix, suffix = _scan_array(text, dec)
    header, order_ok = _key_order(records)
    index = {key: j for j, key in enumerate(header)}
    n = len(records)
    k = len(header)
    cells = [np.full(n, MISSING_KEY, dtype=object) for _ in range(k)]
    kinds: list[set[str]] = [set() for _ in range(k)]
    consts: list[set[str]] = [set() for _ in range(k)]  # NaN / Infinity / -Infinity held (K-20)
    nested = [False] * k
    for r_i, (rec, _, _) in enumerate(records):
        for key, v in rec:
            j = index[key]
            text_v, kind = _cell(v)
            if kind == "nested":
                nested[j] = True
                cells[j][r_i] = None  # never stringified or copied
                continue
            if text_v == MISSING_KEY:
                raise ReadError(f"record {r_i + 1} holds a string made of the control character U+001F, which "
                                "marks absent keys internally", fix="remove the control character from the data")
            cells[j][r_i] = text_v
            if kind:
                kinds[j].add(kind)
                if kind == "const":
                    consts[j].add(text_v)
    unsupported = [(header[j], "nested object/array values") for j in range(k) if nested[j]]
    arrowio.check_unsupported(req, None, unsupported)
    for j in range(k):
        # strings that read like the column's own numbers/booleans: the writer could not tell them apart
        if "string" in kinds[j] and kinds[j] & {"number", "bool", "const"} and _string_clash(
                records, header[j], kinds[j]):
            opts = req.settings.column_options(req.rel_path, None, header[j])
            if not (opts.null or opts.exclude):
                raise UnsupportedColumnError(
                    f"column {header[j]!r} mixes JSON strings and numbers/booleans that read the same, so the "
                    "value types cannot be mirrored",
                    fix=f"--null {header[j]} or --exclude {header[j]}, or store the column as one JSON type")
    style = _style(text, records, seps, prefix, suffix, dec, lines)
    # K-20: the NaN / Infinity constants are numbers stored natively, kept as such (model.NUMERIC_SPECIALS_KEY)
    columns = [RawColumn(header[j], cells[j], {"family": "json", "kind": "unsupported" if nested[j] else "json",
                                                "json_kinds": sorted(kinds[j]),
                                                **({NUMERIC_SPECIALS_KEY: sorted(consts[j])} if consts[j] else {})})
               for j in range(k)]
    exact = order_ok and len(set(seps)) <= 1 and _serialize(style, header, [c.values for c in columns],
                                                            [sorted(x) for x in kinds], n) == text
    if not exact:
        messages.append(Message("info", "json_format_approximate",
                                "the JSON formatting is mirrored approximately (whitespace or key order differs "
                                "between records); values, types and key order are kept"))
    if not order_ok:
        messages.append(Message("info", "json_key_order",
                                "records list their keys in different orders; the output uses one order"))
    layout.extra.update({
        "json_layout": "lines" if lines else "array", "records": n, "format_exact": bool(exact),
        "ensure_ascii": style["ensure_ascii"], "escape_slash": style["escape_slash"],
        "line_ending": "CRLF" if le == "\r\n" else "LF", "trailing_newline": text.endswith("\n"),
        STYLE_KEY: style, KINDS_KEY: {header[j]: sorted(kinds[j]) for j in range(k)},
    })
    raw = RawTable(None, list(header), columns, n)
    return ReadResult(layout, [raw], messages)


def _key_order(records: list[tuple[Any, int, int]]) -> tuple[list[str], bool]:
    """Column order: one key order every record follows (records may omit keys), ties broken by first
    appearance; (first-appearance order, False) when the records disagree."""
    first: dict[str, int] = {}
    after: dict[str, set[str]] = {}
    for r_i, (rec, _, _) in enumerate(records):
        seen: set[str] = set()
        prev = None
        for key, _ in rec:
            if key in seen:
                raise ReadError(f"record {r_i + 1} repeats a key", fix="remove the repeated key from the record")
            seen.add(key)
            first.setdefault(key, len(first))
            after.setdefault(key, set())
            if prev is not None:
                after[prev].add(key)
            prev = key
    indegree = {k: 0 for k in first}
    for k, nxt in after.items():
        for m in nxt:
            indegree[m] += 1
    ready =[(first[k], k) for k, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, k = heapq.heappop(ready)
        order.append(k)
        for m in after[k]:
            indegree[m] -= 1
            if indegree[m] == 0:
                heapq.heappush(ready, (first[m], m))
    if len(order) != len(first):
        return sorted(first, key=first.__getitem__), False
    return order, True


def _string_clash(records: list[tuple[Any, int, int]], key: str, kinds: set[str]) -> bool:
    for rec, _, _ in records:
        for k, v in rec:
            if k != key or not isinstance(v, str) or isinstance(v, (_Num, _Const)):
                continue
            if ("number" in kinds and NUMBER_RE.fullmatch(v)) or ("bool" in kinds and v in ("true", "false")) or (
                    "const" in kinds and v in CONSTANTS):
                return True
    return False


def read_json(req: "ReadRequest") -> ReadResult:
    """Read a JSON file holding a top-level array of flat records."""
    return _read(req, lines=False)


def read_jsonl(req: "ReadRequest") -> ReadResult:
    """Read JSON lines (one flat record per line)."""
    return _read(req, lines=True)


# --------------------------------------------------------------------------- writing
def _enc_str(s: str, style: dict[str, Any]) -> str:
    out = json.dumps(s, ensure_ascii=bool(style.get("ensure_ascii", True)))
    if style.get("escape_slash"):
        out = out.replace("/", "\\/")
    if style.get("upper_hex"):
        out = re.sub(r"\\u([0-9a-f]{4})", lambda m: "\\u" + m.group(1).upper(), out)
    return out


def _bare(s: str, kinds: list[str]) -> bool:
    """True when ``s`` is written unquoted in a column holding these JSON kinds."""
    return bool(("number" in kinds and NUMBER_RE.fullmatch(s)) or ("const" in kinds and s in CONSTANTS)
                or ("bool" in kinds and s in ("true", "false")))


def _enc_value(v: Any, kinds: list[str], style: dict[str, Any]) -> str:
    if v is None:
        return "null"
    s = str(v)
    if _bare(s, kinds):
        return s
    return _enc_str(s, style)


def _serialize(style: dict[str, Any], header: list[str], columns: list[np.ndarray], kinds: list[list[str]],
               n: int) -> str:
    keys = [_enc_str(h, style) + style["key_sep"] for h in header]
    cols = [c.tolist() for c in columns]
    open_ws, item_sep, close_ws = style["open_ws"], style["item_sep"], style["close_ws"]
    recs = []
    for r in range(n):
        items = [keys[j] + _enc_value(cols[j][r], kinds[j], style) for j in range(len(cols))
                 if cols[j][r] != MISSING_KEY]
        recs.append("{" + open_ws + item_sep.join(items) + close_ws + "}" if items else style["empty"])
    return style["prefix"] + style["record_sep"].join(recs) + style["suffix"]


def _write(req: "WriteRequest", lines: bool) -> None:
    if len(req.tables) != 1:
        raise WriteError("the JSON writer needs exactly one table", fix="report this as a bug")
    layout = req.layout
    style = layout.extra.get(STYLE_KEY)
    if style is None:
        raise WriteError("the source JSON layout is missing", fix="report this as a bug")
    t = req.tables[0]
    kinds: list[list[str]] = []
    columns: list[np.ndarray] = []
    for k, si in enumerate(t.source_index):
        prof = t.profile.columns[si] if t.profile is not None else None
        native = (prof.native if prof is not None else None) or {}
        vals = t.columns[k]
        if native.get("kind") == "unsupported" and any(v is not None and v != MISSING_KEY for v in vals.tolist()):
            raise WriteError(f"column {t.header[k]!r}: values present in a nested column",
                             fix=f"--null {t.header[k]} or --exclude {t.header[k]}")
        if prof is not None and prof.role == "null":
            vals = np.full(len(vals), None, dtype=object)  # --null: the key is kept, every value null
        col_kinds = list(native.get("json_kinds") or [])
        if col_kinds and "string" not in col_kinds and any(
                v is not None and v != MISSING_KEY and not _bare(str(v), col_kinds) for v in vals.tolist()):
            # a number/boolean column would get JSON strings (e.g. text generated for numbers the value
            # rules could not read, such as 1.5e-05 without AREA B): FAIL before writing, never quoting values
            name = t.header[k]
            what = prof.type.value if prof is not None else "text"
            raise WriteError(f"column {name!r}: synthetic {what} values cannot be written as JSON "
                             f"{'/'.join(col_kinds)} values (the source column holds no strings)",
                             fix=arrowio.value_fix(name, prof))
        kinds.append(col_kinds)
        columns.append(vals)
    text = _serialize(style, list(t.header), columns, kinds, t.n_rows)
    codec = layout.encoding or "utf-8"
    try:
        data = layout.bom + text.encode(codec)
    except UnicodeEncodeError as e:
        raise WriteError(f"a synthetic value cannot be encoded as {codec}: {e.reason}",
                         fix="pass --encoding utf-8 or report this as a bug") from None
    if layout.compression:
        compress.write_bytes(req.path, data, layout.compression, layout)
    else:
        Path(req.path).write_bytes(data)


def write_json(req: "WriteRequest") -> None:
    """Write flat JSON records mirroring the source layout."""
    _write(req, lines=False)


def write_jsonl(req: "WriteRequest") -> None:
    """Write JSON lines mirroring the source layout."""
    _write(req, lines=True)


# --------------------------------------------------------------------------- detection / self-check
def detect_native(ctx: Any) -> Any:
    """JSON value kinds need no native type: the value rules decide (bool/number texts are typed values).

    Nested columns never reach detection (the reader FAILs them); a defensive error is raised if one does.
    """
    native = ctx.native or {}
    if native.get("family") == "json" and native.get("kind") == "unsupported" and ctx.text_values:
        raise UnsupportedColumnError(f"column {ctx.name!r}: nested JSON values are never copied",
                                     fix=f"--null {ctx.name} or --exclude {ctx.name}")
    return None


def layout_diff(source: FileLayout, output: FileLayout, source_tables: list[dict[str, Any]],
                output_tables: list[Any]) -> list[str]:
    """JSON layout (template, escaping), per-column value kinds and container differences."""
    s, o = source.extra or {}, output.extra or {}
    issues: list[str] = []
    ss, os_ = s.get(STYLE_KEY) or {}, o.get(STYLE_KEY) or {}
    # escaping settings are copied verbatim by the writer and are only observable when a value needs them;
    # the structural template is compared
    for key in ("prefix", "suffix", "open_ws", "key_sep", "close_ws"):
        if key in ss and key in os_ and ss[key] != os_[key]:
            issues.append(f"JSON formatting ({key.replace('_', ' ')}) differs from the source")
    if s.get("json_layout") != o.get("json_layout"):
        issues.append("JSON layout (array vs lines) differs from the source")
    sk, ok = s.get(KINDS_KEY) or {}, o.get(KINDS_KEY) or {}
    for name, kinds in ok.items():
        src_kinds = set(sk.get(name, []))
        extra_kinds = set(kinds) - src_kinds
        if extra_kinds:
            issues.append(f"column {name!r}: JSON value type(s) {', '.join(sorted(extra_kinds))} not in the source")
    issues.extend(compress.container_diff(source, output))
    return issues


def inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes pandas infers with default ``read_json`` (``lines=True`` for JSONL)."""
    import pandas as pd

    lines = layout.format == "jsonl" or (layout.extra or {}).get("json_layout") == "lines"
    enc = layout.encoding or "utf-8"
    if layout.bom and enc == "utf-8":
        enc = "utf-8-sig"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_json(path, lines=lines, encoding=enc,
                              compression=_PANDAS_COMPRESSION.get(layout.compression or "", None))
    except Exception:
        return None
    header = list((layout.extra or {}).get(KINDS_KEY, {}).keys())
    if header and len(set(df.columns)) == len(df.columns) and set(header) <= set(map(str, df.columns)):
        return [str(df[h].dtype) for h in header]
    return [str(t) for t in df.dtypes]
