"""Stata .dta reader/writer and SAS .sas7bdat/.xpt reader.

AREA F owns this module (design 6 "Stata", section 4 "SAS", D17).

Stata
-----
Read with pandas' ``StataReader`` without conversions (``convert_dates``,
``convert_categoricals`` off; ``convert_missing`` and ``preserve_dtypes`` on), so
the stored codes, storage types and missing values are seen as they are. The
file's version, byte order, data label, time stamp, every variable's storage
type, display format, variable label and value-label table are kept
(``FileLayout.extra`` private ``_`` keys and ``RawColumn.native``,
``family='stata'``). Canonical cell text:

* strings as stored (Stata has no string null: ``""`` is its missing value);
* numbers as their shortest text (float storage at its own precision); the
  system missing ``.`` and the extended ``.a``-``.z`` are ``None`` (pandas cannot
  write extended missing values: they come back as ``.``, with a warning);
* a column with a value-label table (labels unique and non-blank, no label equal
  to an unlabelled code present) shows the **label text** for labelled codes and
  the number for the others, and is detected as FACTOR (native): labelled ints
  behave as categorical with their labels; the writer maps labels back to codes.
  Real labels are written back only for label-safe types (FACTOR, ORDINAL ...),
  and only the labels of codes the synthetic rows hold (a label the D35
  allowlist did not keep as its pseudonym): a label no row holds was
  never checked (it can quote anything) and is written only when it is a
  vocabulary word (``arrowio.vocabulary_label``: grade levels, ordinal scales,
  the standard education vocabulary ...); a table whose
  labels could not be shown as values (duplicate or blank labels: only the codes
  were checked) is not written, the column keeps its codes.
  A labelled column generated as something else (a D12 PII/free-text header
  hint, ``--force-text``, an ID) gets a new value-label table built from its
  synthetic labels on new codes (or, when every synthetic value is a number, no
  table); a ``--null`` column gets no table (it was never detected, so its
  labels are not known to be safe);
* ``%td`` (days since 1960) as ``YYYY-MM-DD`` and ``%tc`` (milliseconds since
  1960) as ``YYYY-MM-DD HH:MM:SS[.fff]``, detected through AREA A's
  ``detect_dates`` (``io.arrowio.native_date_detection``); other ``%t`` formats
  and non-integral values stay numbers.

Display formats are kept, with the literal text of ``%t`` formats (``!c``
characters) masked (:func:`safe_stata_format`).

The **data label and variable labels are free text** (a dataset description, a
note on a variable: they can quote real data) and are **never written** to the
synthetic file; the source's are counted in the warning ``stata_labels_dropped``
(never quoted) and the self-check expects them blank in the output.

Written with ``DataFrame.to_stata`` at the same version (115 and older formats
are written as 114, the oldest pandas writes, with a warning), byte order, time
stamp and value-label tables (as above), each numeric column
in its source storage type (byte/int/long/float/double, nullable for missing;
a synthetic value outside the type's non-missing range, e.g. a synthetic ID
above 100 in a byte column, FAILs the file with ``--null`` / ``--exclude``),
``strL`` columns as ``strL``; then the display formats (``%td``, ``%9.2f``,
``%-20s`` ...) are written back into the file's format list byte for byte.
Not reproduced (listed by the self-check only when it changes something a
reader sees): value-label table *names* (pandas names each table after its
column), characteristics/notes and the sort order; string widths follow the
synthetic values.

SAS
---
Read with ``pd.read_sas`` (``encoding='infer'`` for .sas7bdat, latin-1 for
.xpt), converted to an Arrow table and handled like Parquet input
(``io.arrowio``); no SAS writer exists, so the registry writes it with
``arrowio.write_parquet`` as ``<name>.parquet`` (warning ``sas_as_parquet``).
"""

from __future__ import annotations

import datetime as _dt
import io
import math
import struct
import warnings
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ReadError, UnsupportedColumnError, WriteError
from ..model import LABEL_SAFE_TYPES, ColumnType, FileLayout, Message, RawColumn, RawTable, ReadResult
from . import arrowio
from .numfmt import mask_literal

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ReadRequest, WriteRequest

WRITABLE_VERSIONS = (114, 117, 118, 119)
STATA_EPOCH_DAYS = -3653  # 1960-01-01 in days since 1970-01-01
_NUMERIC_TYPES = {"b": "Int8", "h": "Int16", "l": "Int32", "f": "float32", "d": "float64"}
#: Non-missing value ranges of Stata's integer storage types (larger values are the missing codes).
STATA_INT_RANGES = {"b": (-127, 100), "h": (-32767, 32740), "l": (-2147483647, 2147483620)}
#: ``TableProfile.layout`` key (PRIVATE) where :func:`write_dta` records the columns whose value-label
#: table was deliberately not copied ('synthetic': rebuilt from the synthetic labels; 'dropped': no
#: table, for ``--null`` and for non-label columns written as plain numbers), so :func:`layout_diff`
#: (which receives the table layouts) does not expect the source labels there.
LABEL_POLICY_KEY = "_stata_label_policy"
_FMT_WIDTH = {114: 49, 115: 49, 117: 49, 118: 57, 119: 57}


# --------------------------------------------------------------------------- reading
def safe_stata_format(fmt: str) -> str:
    """A Stata display format with its literal text masked.

    A ``%t`` date format shows any character written as ``!c`` (``%tdDD_Mon_CCYY!_!(!J!.!S!.!)``), so a
    format can quote a name. Runs of escaped characters are masked like Excel format literals
    (``io.numfmt.mask_literal``: letters ``x``/``X`` outside the unit/currency words, digits ``0``,
    punctuation kept); the codes and every other format are kept. Idempotent, so the self-check compares
    the rebuilt source format with the re-read output.
    """
    if "!" not in fmt or not fmt.lstrip("%-").startswith("t"):
        return fmt
    out: list[str] = []
    i, n = 0, len(fmt)
    while i < n:
        if fmt[i] == "!" and i + 1 < n:
            chars = []
            while i + 1 < n and fmt[i] == "!":
                chars.append(fmt[i + 1])
                i += 2
            out.extend("!" + c for c in mask_literal("".join(chars)))
        else:
            out.append(fmt[i])
            i += 1
    return "".join(out)


def _date_kind(fmt: str) -> str | None:
    f = fmt.lstrip("%").lstrip("-")
    if f.startswith(("td", "d")):  # %td, and %d (dates before Stata 10)
        return "date"
    if f.startswith("tc"):
        return "datetime"
    return None


def _num_text(v: Any, typ: Any) -> str:
    if typ in ("b", "h", "l"):
        return str(int(v))
    return arrowio._float_text(float(v), 32 if typ == "f" else 64)


def _stata_reader(data: bytes) -> Any:
    import pandas as pd

    return pd.io.stata.StataReader(io.BytesIO(data), convert_dates=False, convert_categoricals=False,
                                   convert_missing=True, preserve_dtypes=True, order_categoricals=False)


def _render_stata_column(raw: list[Any], typ: Any, fmt: str, labels: dict[int, str] | None
                         ) -> tuple[np.ndarray, dict[str, Any], int]:
    """Canonical text of one Stata column; returns (values, native details, extended-missing count)."""
    from pandas.io.stata import StataMissingValue

    n = len(raw)
    out = np.empty(n, dtype=object)
    info: dict[str, Any] = {}
    if isinstance(typ, int) or typ == "Q":  # str# / strL
        info["kind"] = "strl" if typ == "Q" else "str"
        for i, v in enumerate(raw):
            if isinstance(v, bytes):
                raise arrowio._CannotRender("binary strL")
            out[i] = "" if v is None else str(v)
        return out, info, 0
    nums: list[float | None] = []
    extended = 0
    for v in raw:
        if isinstance(v, StataMissingValue):
            nums.append(None)
            if v.string != ".":
                extended += 1
        elif v is None or (isinstance(v, float) and v != v):
            nums.append(None)
        else:
            nums.append(float(v))
    present = [x for x in nums if x is not None]
    integral = all(x == int(x) for x in present)
    dkind = _date_kind(fmt) if not labels else None
    if dkind == "date" and integral:
        info["kind"] = "date"
        days = np.array([int(x) + STATA_EPOCH_DAYS for x in present], dtype=np.int64)
        texts = iter(arrowio.format_dates(days))
        for i, x in enumerate(nums):
            out[i] = None if x is None else next(texts)
        return out, info, extended
    if dkind == "datetime" and integral:
        info["kind"] = "datetime"
        ms = np.array([int(x) for x in present], dtype=np.int64)
        secs = np.floor_divide(ms, 1000)
        frac = ms - secs * 1000
        digits = 3 if np.any(frac) else 0
        info["frac_digits"] = digits
        texts = iter(arrowio.format_datetimes(secs + STATA_EPOCH_DAYS * 86400, frac, digits))
        for i, x in enumerate(nums):
            out[i] = None if x is None else next(texts)
        return out, info, extended
    info["kind"] = "int" if typ in ("b", "h", "l") else "float"
    use_labels = False
    if labels:
        codes_present = {int(x) for x in present if x == int(x)}
        unlabelled_texts = {str(c) for c in codes_present if c not in labels}
        label_texts = list(labels.values())
        # a blank label would read as a missing value and lose its code: codes are shown then
        use_labels = (len(set(label_texts)) == len(label_texts) and not (set(label_texts) & unlabelled_texts)
                      and all(t.strip() for t in label_texts))
    info["labels_rendered"] = use_labels
    if labels:
        info["unused_labels"] = sum(1 for c in labels if c not in codes_present)
        info["unused_label_texts"] = [lbl for c, lbl in labels.items() if c not in codes_present]  # PRIVATE
    for i, x in enumerate(nums):
        if x is None:
            out[i] = None
        elif use_labels and x == int(x) and int(x) in labels:  # type: ignore[operator]
            out[i] = labels[int(x)]  # type: ignore[index]
        else:
            out[i] = _num_text(x, typ)
    return out, info, extended


def _version_of(data: bytes) -> int | None:
    if data.startswith(b"<stata_dta>"):
        i = data.find(b"<release>")
        try:
            return int(data[i + 9:i + 12])
        except ValueError:
            return None
    return data[0] if data else None


def read_dta(req: "ReadRequest") -> ReadResult:
    """Read a Stata .dta file keeping labels, formats, storage types and version."""
    data = Path(req.path).read_bytes()
    messages: list[Message] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _stata_reader(data) as r:
                df = r.read()
                value_labels = {str(k): {int(c): str(lbl) for c, lbl in v.items()} for k, v in r.value_labels().items()}
                var_labels = r.variable_labels()
                version = int(getattr(r, "_format_version", None) or _version_of(data) or 118)
                typlist = list(getattr(r, "_typlist"))
                fmtlist = [str(f) for f in getattr(r, "_fmtlist")]
                lbllist = [str(x) for x in getattr(r, "_lbllist")]
                byteorder = str(getattr(r, "_byteorder", "<"))
                data_label = str(r.data_label or "")
                time_stamp = str(r.time_stamp or "")
    except Exception as e:  # pandas raises many types for a bad file
        raise ReadError(f"not a readable Stata .dta file ({type(e).__name__})",
                        fix="check the file (it may be truncated or not Stata)") from None
    write_version = version if version in WRITABLE_VERSIONS else 114
    if write_version != version:
        messages.append(Message("warning", "stata_version",
                                f"Stata format {version} cannot be written (pandas writes 114, 117, 118, 119): the "
                                "output is format 114"))
    columns: list[RawColumn] = []
    unsupported: list[tuple[str, str]] = []
    extended_total = 0
    col_info: dict[str, Any] = {}
    for i, name in enumerate(df.columns):
        name = str(name)
        typ, fmt = typlist[i], safe_stata_format(fmtlist[i])  # literal '!c' text masked (R2-02 sibling)
        lbl_name = lbllist[i] if i < len(lbllist) else ""
        labels = value_labels.get(lbl_name) if lbl_name else None
        native: dict[str, Any] = {"family": "stata", "typ": typ, "fmt": fmt, "label_name": lbl_name,
                                  "labels": labels, "var_label": str(var_labels.get(name, "") or "")}
        try:
            values, info, extended = _render_stata_column(df[name].tolist(), typ, fmt, labels)
            native.update(info)
            extended_total += extended
        except arrowio._CannotRender:
            native["kind"] = "unsupported"
            values = np.empty(len(df), dtype=object)
            unsupported.append((name, "binary strL"))
        columns.append(RawColumn(name, values, native))
        col_info[name] = {"typ": typ, "fmt": fmt, "labels": labels, "var_label": native["var_label"]}
    arrowio.check_unsupported(req, None, unsupported)
    if extended_total:
        messages.append(Message("warning", "stata_extended_missing",
                                f"{extended_total} extended missing value(s) (.a-.z) are written as system "
                                "missing '.' (pandas cannot write extended missing values)"))
    n_var_labels = sum(1 for c in col_info.values() if c["var_label"].strip())
    if req.purpose == "source" and (data_label.strip() or n_var_labels):
        parts = (["the data label"] if data_label.strip() else []) + (
            [f"{n_var_labels} variable label(s)"] if n_var_labels else [])
        messages.append(Message("warning", "stata_labels_dropped",
                                "free-text Stata labels are not copied to the synthetic file (they can quote real "
                                "data): " + " and ".join(parts) + " left out"))
    extra: dict[str, Any] = {
        "stata_version": version, "write_version": write_version, "byteorder": byteorder,
        "value_label_tables": len(value_labels),
        "_data_label": data_label, "_time_stamp": time_stamp, "_columns": col_info,
    }
    layout = FileLayout(format="dta", extra=extra)
    raw = RawTable(None, [str(c) for c in df.columns], columns, len(df))
    return ReadResult(layout, [raw], messages)


# --------------------------------------------------------------------------- writing
def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _stored_numbers(name: str, nums: list[float | None], typ: Any, profile: Any) -> Any:
    """``nums`` in the column's Stata storage type (nullable for missing); FAILs cleanly when they do not fit."""
    import pandas as pd

    dtype = _NUMERIC_TYPES.get(str(typ), "float64")
    if any(x is not None and not math.isfinite(x) for x in nums):
        raise _value_error(name, profile)
    if dtype.startswith("Int"):
        lo, hi = STATA_INT_RANGES[str(typ)]
        if any(x is not None and (x != int(x) or not lo <= x <= hi) for x in nums):
            raise _value_error(name, profile)
        return pd.array([None if x is None else int(x) for x in nums], dtype=dtype)
    return np.array([np.nan if x is None else x for x in nums], dtype=dtype)


def _numeric_column(name: str, vals: list[Any], native: dict[str, Any], profile: Any) -> Any:
    """The synthetic texts of a numeric column as numbers in its storage type (labels mapped to codes)."""
    return _stored_numbers(name, _parse_numbers(name, vals, native, profile), native.get("typ"), profile)


def _parse_numbers(name: str, vals: list[Any], native: dict[str, Any], profile: Any) -> list[float | None]:
    """Stata numbers of canonical texts (label texts -> codes, %td/%tc dates); raises WriteError otherwise."""
    kind = native.get("kind")
    labels = native.get("labels") or {}
    reverse: dict[str, int] = {}
    if native.get("labels_rendered"):
        for code, label in sorted(labels.items()):
            reverse.setdefault(label, code)
    nums: list[float | None] = []
    for v in vals:
        if _blank(v):
            nums.append(None)
            continue
        s = str(v)
        if s in reverse:
            nums.append(float(reverse[s]))
            continue
        if kind == "date":
            r = arrowio.parse_datetime(s)
            if r is None or (r[3] and (r[0] % 86400 or r[1])):
                raise _value_error(name, profile)
            nums.append(float(r[0] // 86400 - STATA_EPOCH_DAYS))
            continue
        if kind == "datetime":
            r = arrowio.parse_datetime(s)
            if r is None:
                raise _value_error(name, profile)
            nums.append(float((r[0] - STATA_EPOCH_DAYS * 86400) * 1000 + r[1] // 10**6))
            continue
        try:
            nums.append(float(s.strip()))
        except ValueError:
            raise _value_error(name, profile) from None
    return nums


def _relabelled_column(name: str, vals: list[Any], native: dict[str, Any], profile: Any
                       ) -> tuple[Any, dict[int, str]]:
    """A labelled column generated as non-label values (D12 PII hint, ``--force-text``, IDs ...).

    Every synthetic value becomes a label of a new value-label table (sorted values on the
    source's sorted codes, then the codes after the largest one), so the column stays a
    labelled integer column and no real label is ever written.
    """
    uniq = sorted({str(v) for v in vals if not _blank(v)})
    codes = sorted(int(c) for c in (native.get("labels") or {}))
    nxt = (codes[-1] + 1) if codes else 1
    while len(codes) < len(uniq):
        codes.append(nxt)
        nxt += 1
    code_of = dict(zip(uniq, codes))
    nums = [None if _blank(v) else float(code_of[str(v)]) for v in vals]
    return _stored_numbers(name, nums, native.get("typ"), profile), {code_of[u]: u for u in uniq}


def _value_error(name: str, profile: Any = None) -> WriteError:
    what = f"synthetic {profile.type.value} value" if getattr(profile, "type", None) is not None else "synthetic value"
    return WriteError(f"column {name!r}: a {what} cannot be stored in the column's Stata storage type",
                      fix=arrowio.value_fix(name, profile))


def _time_stamp(text: str) -> _dt.datetime:
    try:
        return _dt.datetime.strptime(text.strip(), "%d %b %Y %H:%M")
    except ValueError:
        return _dt.datetime(1960, 1, 1)  # deterministic stand-in for a missing/unparsable stamp (D26)


def _format_offset(data: bytes, version: int) -> tuple[int, int, int] | None:
    """(offset of the first format entry, entry width, number of variables) in a written .dta file."""
    width = _FMT_WIDTH.get(version)
    if width is None:
        return None
    if version >= 117:
        pos = len(b"<stata_dta><header><release>") + 3 + len(b"</release><byteorder>")
        little = data[pos:pos + 3] == b"LSF"
        e = "<" if little else ">"
        pos += 3 + len(b"</byteorder><K>")
        if version == 119:
            k = struct.unpack_from(e + "I", data, pos)[0]
            pos += 4
        else:
            k = struct.unpack_from(e + "H", data, pos)[0]
            pos += 2
        pos += len(b"</K><N>") + (4 if version == 117 else 8)
        pos += len(b"</N><label>")
        if version == 117:
            llen = data[pos]
            pos += 1
        else:
            llen = struct.unpack_from(e + "H", data, pos)[0]
            pos += 2
        pos += llen + len(b"</label><timestamp>")
        pos += 1 + data[pos] + len(b"</timestamp></header>")
        if data[pos:pos + 5] != b"<map>":
            return None
        fmt_pos = struct.unpack_from(e + "Q", data, pos + 5 + 8 * 5)[0]
        if data[fmt_pos:fmt_pos + 9] != b"<formats>":
            return None
        return fmt_pos + 9, width, k
    e = "<" if data[1] == 2 else ">"
    k = struct.unpack_from(e + "H", data, 4)[0]
    return 109 + k + k * 33 + 2 * (k + 1), width, k


def patch_formats(path: Path, version: int, formats: list[str]) -> None:
    """Write the source display formats into a .dta file written by pandas (fixed-width entries)."""
    data = bytearray(Path(path).read_bytes())
    loc = _format_offset(bytes(data), version)
    if loc is None:
        raise WriteError("could not locate the format list in the written .dta file", fix="report this as a bug")
    start, width, k = loc
    if k != len(formats):
        raise WriteError("variable count mismatch while writing Stata formats", fix="report this as a bug")
    codec = "utf-8" if version >= 118 else "latin-1"
    for i, fmt in enumerate(formats):
        b = fmt.encode(codec, "replace")[: width - 1]
        data[start + i * width:start + (i + 1) * width] = b + b"\x00" * (width - len(b))
    Path(path).write_bytes(bytes(data))


def write_dta(req: "WriteRequest") -> None:
    """Write a Stata .dta file with the source version, labels, storage types and formats."""
    import pandas as pd

    if len(req.tables) != 1:
        raise WriteError("the Stata writer needs exactly one table", fix="report this as a bug")
    t = req.tables[0]
    ex = req.layout.extra
    frame: dict[str, Any] = {}
    value_labels: dict[str, dict[int, str]] = {}
    strl: list[str] = []
    formats: list[str] = []
    policy: dict[str, str] = {}
    for k, si in enumerate(t.source_index):
        prof = t.profile.columns[si] if t.profile is not None else None
        native = (prof.native if prof is not None else None) or {}
        name = t.header[k]
        vals = t.columns[k].tolist()
        kind = native.get("kind")
        if kind == "unsupported":
            if any(v is not None and v != "" for v in vals):
                raise WriteError(f"column {name!r}: binary strL values cannot be written",
                                 fix=f"--null {name} or --exclude {name}")
            vals = [None] * len(vals)
            kind = "strl"
        labels = native.get("labels")
        names = arrowio._level_map(prof)
        renamed = bool(labels and names)
        if renamed:
            # D35: a value label the allowlist did not keep is written as its pseudonym on the same code
            labels = {c: names.get(lbl, lbl) for c, lbl in labels.items()}
            native = {**native, "labels": labels}
        null = prof is not None and prof.role == "null"
        if kind in ("str", "strl"):
            frame[name] = pd.Series(["" if v is None else str(v) for v in vals], dtype=object)
            if kind == "strl" and int(ex.get("write_version", 118)) >= 117:
                strl.append(name)
        elif labels and not null and prof is not None and prof.type not in LABEL_SAFE_TYPES:
            # not a label column (D12 PII hint, --force-text, IDs, --force-type): the real labels are not
            # copied. Numbers stay numbers (no label table); text gets new codes with its synthetic labels.
            try:
                nums = _parse_numbers(name, vals, native, prof)
            except WriteError:
                frame[name], synthetic = _relabelled_column(name, vals, native, prof)
                if synthetic:
                    value_labels[name] = synthetic
                policy[name] = "synthetic"
            else:
                frame[name] = _stored_numbers(name, nums, native.get("typ"), prof)
                policy[name] = "dropped"
        else:
            frame[name] = _numeric_column(name, vals, native, prof)
            if labels and null:
                policy[name] = "dropped"  # --null: the column was not detected, so its labels are not known safe
            elif labels and not native.get("labels_rendered"):
                # labels that could not be shown as values (duplicate/blank labels): detection saw only the
                # codes, so no label text was ever checked; the column is written as plain codes (R2-03)
                policy[name] = "dropped"
            elif labels:
                # the labels of codes the synthetic rows hold: a label no row holds was never checked (it can
                # quote anything) and is written only when it is a tool-vocabulary word (a grade level, R2-03)
                held = {int(x) for x in pd.Series(frame[name]).dropna().tolist() if float(x) == int(x)}
                kept = {int(c): str(lbl) for c, lbl in labels.items()
                        if int(c) in held or arrowio.vocabulary_label(lbl)}
                if kept:
                    value_labels[name] = kept
                    if renamed:
                        policy[name] = "names"
                else:
                    policy[name] = "dropped"
        # the variable label (free text) is never copied
        formats.append(str(native.get("fmt") or ""))
    if t.profile is not None:
        t.profile.layout[LABEL_POLICY_KEY] = policy
    df = pd.DataFrame(frame)
    version = int(ex.get("write_version", 118))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df.to_stata(req.path, write_index=False, version=version, byteorder=str(ex.get("byteorder", "<")),
                        time_stamp=_time_stamp(str(ex.get("_time_stamp", ""))),
                        data_label=None, variable_labels=None,  # free text: never copied
                        value_labels=value_labels or None, convert_strl=strl or None)
    except Exception as e:  # pandas raises ValueError / UnicodeEncodeError ... with messages quoting values
        raise WriteError(f"Stata writer error ({type(e).__name__})",
                         fix="check that the synthetic text fits the file's Stata version (e.g. latin-1 for "
                             "format 114/117), or report this as a bug") from None
    if all(formats):
        patch_formats(Path(req.path), version, formats)


# --------------------------------------------------------------------------- SAS
def _sas_format(name: str) -> str:
    return "xport" if PurePosixPath(name).suffix.lower() == ".xpt" else "sas7bdat"


def parquet_storable(table: Any) -> Any:
    """``table`` with the Arrow types Parquet stores unchanged.

    pandas reads SAS dates as ``datetime64[s]`` (Arrow ``timestamp[s]``), a unit Parquet does not
    have: written as Parquet they come back as ``timestamp[ms]``, which the self-check would see as
    schema drift. Second-resolution timestamps/times are cast to milliseconds (lossless) up front.
    """
    import pyarrow as pa
    import pyarrow.types as pt

    fields = []
    for f in table.schema:
        t = f.type
        if pt.is_timestamp(t) and t.unit == "s":
            f = f.with_type(pa.timestamp("ms", tz=t.tz))
        elif pt.is_time32(t) and t.unit == "s":
            f = f.with_type(pa.time32("ms"))
        fields.append(f)
    schema = pa.schema(fields, metadata=table.schema.metadata)
    return table if schema.equals(table.schema) else table.cast(schema)


def read_sas_frame(path: Path, fmt: str, encoding: str | None) -> Any:
    """``pd.read_sas`` (one place, so tests can stub SAS files that cannot be created here)."""
    import pandas as pd

    return pd.read_sas(path, format=fmt, encoding=encoding)


def read_sas(req: "ReadRequest") -> ReadResult:
    """Read SAS .sas7bdat/.xpt; the output is written as Parquet (D17)."""
    import pyarrow as pa

    fmt = _sas_format(req.rel_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = read_sas_frame(Path(req.path), fmt, "infer" if fmt == "sas7bdat" else "latin-1")
        table = parquet_storable(pa.Table.from_pandas(df, preserve_index=False))
    except Exception as e:  # pandas raises many types for a bad file
        raise ReadError(f"not a readable SAS {fmt} file ({type(e).__name__})",
                        fix="export the dataset from SAS to .csv (or Parquet) and re-run") from None
    n = len(table.schema)
    extra: dict[str, Any] = {
        "sas_format": fmt, "written_as": "parquet", "parquet_version": "2.6", "codec": "snappy",
        "row_groups": None, "row_group_rows": None, "arrow_schema_stored": True,
        "_codecs": ["SNAPPY"] * n, "_use_dictionary": [True] * n, "_write_statistics": [True] * n,
    }
    messages = [Message("warning", "sas_as_parquet",
                        "SAS files are read-only (no SAS writer exists): the output is written as Parquet "
                        f"({PurePosixPath(req.rel_path).stem}.parquet), design D17")]
    return arrowio.table_result(table, req, "sas", extra, messages)


# --------------------------------------------------------------------------- detection / self-check
def detect_native(ctx: Any) -> Any:
    """Type implied by Stata storage (value labels -> FACTOR, %td/%tc -> dates) or, for SAS, by Arrow."""
    native = ctx.native or {}
    family = native.get("family")
    if family == "arrow":
        return arrowio.detect_native(ctx)
    if family != "stata":
        return None
    kind = native.get("kind")
    if kind == "unsupported" and ctx.text_values:
        raise UnsupportedColumnError(f"column {ctx.name!r}: binary strL values are never copied",
                                     fix=f"--null {ctx.name} or --exclude {ctx.name}")
    if not ctx.typed_values and not ctx.text_values:
        return None
    if native.get("labels") and kind in ("int", "float"):
        from ..detect.context import Detection

        labels = native["labels"]
        variants: dict[str, set[str]] = {}
        for raw in ctx.raw_counts:
            if raw is None or raw.strip() == "":
                continue
            variants.setdefault(raw.strip().casefold(), set()).add(raw)
        how = "label text" if native.get("labels_rendered") else "codes (labels not unique)"
        unused = int(native.get("unused_labels") or 0)
        det = Detection(ColumnType.FACTOR, params={
            "n_levels": len(variants), "variant_groups": sum(1 for v in variants.values() if len(v) > 1),
            "value_labels": len(labels), "unused_labels": unused},
            source="native", reason=f"Stata value labels ({len(labels)} labels; values as {how}): categorical")
        dropped = sum(1 for lbl in (native.get("unused_label_texts") or []) if not arrowio.vocabulary_label(lbl))
        if dropped and native.get("labels_rendered"):  # never written: a label no row holds was never checked
            det.messages.append(Message(
                "info", "value_labels_unused",
                f"{dropped} value label(s) belong to codes held by no row; they are not written (a label that no "
                "row holds is never checked, so only tool-vocabulary labels such as grade levels are kept unused)",
                column=ctx.name))
        if not native.get("labels_rendered"):
            det.messages.append(Message(
                "info", "value_labels_not_written",
                "the value labels cannot be shown as values (duplicate or blank labels), so only the codes were "
                "checked: the column is written as codes without its value-label table", column=ctx.name))
        return det
    if not ctx.typed_values:
        return None
    if kind == "date":
        return arrowio.native_date_detection(ctx, ColumnType.DATE, f"Stata {native.get('fmt')} date")
    if kind == "datetime":
        return arrowio.native_date_detection(ctx, ColumnType.DATETIME, f"Stata {native.get('fmt')} datetime")
    return None


def _type_family(typ: Any) -> str:
    if isinstance(typ, int):
        return "str"
    return str(typ)


def layout_diff(source: FileLayout, output: FileLayout, source_tables: list[dict[str, Any]],
                output_tables: list[Any]) -> list[str]:
    """Stata version, byte order, data label, storage types, formats, variable and value labels."""
    if output.format != "dta":
        return []
    s, o = source.extra or {}, output.extra or {}
    issues: list[str] = []
    if s.get("write_version") != o.get("stata_version"):
        issues.append(f"Stata format {s.get('write_version')} -> {o.get('stata_version')}")
    if s.get("byteorder") != o.get("byteorder"):
        issues.append("byte order differs from the source")
    if (o.get("_data_label") or "").strip():  # free text: the source's is never copied
        issues.append("a data label was written (the source's free-text labels are never copied)")
    s_cols, o_cols = s.get("_columns") or {}, o.get("_columns") or {}
    policy: dict[str, str] = {}
    for tl in source_tables or []:
        policy.update((tl or {}).get(LABEL_POLICY_KEY) or {})
    for name, oc in o_cols.items():
        sc = s_cols.get(name)
        if sc is None:
            issues.append(f"column {name!r} is not in the source")
            continue
        if _type_family(sc["typ"]) != _type_family(oc["typ"]):
            issues.append(f"column {name!r}: Stata storage type {sc['typ']} -> {oc['typ']}")
        if sc["fmt"] != oc["fmt"]:
            issues.append(f"column {name!r}: display format {sc['fmt']} -> {oc['fmt']}")
        if (oc["var_label"] or "").strip():
            issues.append(f"column {name!r}: a variable label was written (free-text labels are never copied)")
        how = policy.get(name)
        if how == "dropped":  # --null or plain numbers: no value-label table (the real labels are not copied)
            if oc["labels"]:
                issues.append(f"column {name!r}: a value-label table was written for a column whose labels are "
                              "not copied")
        elif how == "synthetic":  # rebuilt from synthetic labels: never the source's labels
            if set((oc["labels"] or {}).values()) & set((sc["labels"] or {}).values()):
                issues.append(f"column {name!r}: a source value label was written for a non-label column")
        elif how == "names" and (not set(oc["labels"] or {}) <= set(sc["labels"] or {}) or any(
                lbl != sc["labels"][c] and lbl in set(sc["labels"].values()) for c, lbl in oc["labels"].items())):
            # D35: labels the allowlist did not keep are written as their pseudonyms on their own codes, never as
            # another code's source label
            issues.append(f"column {name!r}: value labels differ from the source")
        elif how != "names" and (not set((oc["labels"] or {}).items()) <= set((sc["labels"] or {}).items()) or (
                bool(sc["labels"]) != bool(oc["labels"]))):
            issues.append(f"column {name!r}: value labels differ from the source")
        else:  # the source's labels of the codes the rows hold (and unused tool-vocabulary labels) only (R2-03)
            ocol = next((c for t in output_tables or [] for c in getattr(t, "columns", []) or [] if c.name == name),
                        None)
            texts = ((ocol.native or {}) if ocol is not None else {}).get("unused_label_texts") or []
            unused = sum(1 for lbl in texts if not arrowio.vocabulary_label(lbl))
            if unused:
                issues.append(f"column {name!r}: {unused} value label(s) for codes no row holds (a label no row "
                              "holds is never written)")
    return issues


def inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes pandas infers with default ``read_stata`` / ``read_sas`` (None if it cannot read the file)."""
    import pandas as pd

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if layout.format == "sas":
                df = read_sas_frame(Path(path), str(layout.extra.get("sas_format", "sas7bdat")), None)
            else:
                df = pd.read_stata(path)
    except Exception:
        return None
    return [str(t) for t in df.dtypes]
