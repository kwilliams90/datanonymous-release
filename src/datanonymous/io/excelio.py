"""Excel reader/writer: .xlsx/.xlsm, .xls (via xlrd, written as .xlsx), HTML tables saved as .xls (AREA E).

Registry hooks (design D17, section 6 "Excel", section 4 Excel rows, section 10
Excel defaults); the work happens in the helper modules:

* :func:`read_xlsx` / :func:`read_xls` (:mod:`.excel_read`, :mod:`.excel_book`):
  one :class:`~datanonymous.model.RawTable` per worksheet in workbook order,
  hidden sheets included (``--sheet NAME`` reads one sheet). Header row by
  the design heuristic (``--header-row N`` / ``--range [Sheet!]A5:H200``
  override), title/notes rows above it, footer total/note rows below the
  data, headerless side notes beside the table and notes-only sheets kept as
  layout (written as placeholders), cell values as canonical strings with the
  per-column cell storage in ``RawColumn.native``. Excel extras (pivots, charts,
  comments, validation, conditional formats, named ranges, hyperlinks,
  protection, external links, macros, document properties ...) are counted and
  listed in the report, never copied. ``.xlsm`` macros are never copied
  (written as ``.xlsx`` with a warning); a real ``.xls`` is written as
  ``.xlsx`` (discovery adds the warning).
* :func:`write_xlsx` (:mod:`.excel_write`): a fresh workbook mirroring sheet
  names/order/visibility, title/footer rows (placeholders unless
  ``--keep-titles``), header row (number/date/boolean header cells keep
  their type), per-column cell types and number formats (their mix by exact
  quota),
  widths, freeze panes, autofilter, merged title ranges and the date system;
  footer SUM/COUNT/AVERAGE recomputed; formulas as static values; neutral
  document properties; byte-identical for the same seed.
* :func:`read_html_table` / :func:`write_html_table` (:mod:`.excel_html`): SIS
  "Excel" exports that are really HTML, mirrored as HTML (stdlib
  ``html.parser``; CSS comments/urls/text, scripts and banner tables of
  notes are never copied). Disguised tab/comma text ``.xls`` files are read and
  mirrored by the CSV handler (discovery classifies them).
* :func:`detect_native`: Excel date cells are typed by the AREA A date hook
  before the value rules; every other storage type is left to the value rules
  (the writer restores the cell type from ``native`` whatever the detected type).
* :func:`layout_diff`: sheet names/order/visibility, date system, header row,
  title/footer row counts, column span, per-column mix of cell types and
  number formats (exact quota, :mod:`.excel_storage`) and main cell type,
  freeze panes, autofilter, widths, merged ranges; for HTML "Excel" tables
  each column's mix of cell templates per content class (exact quota).
* :func:`inferred_dtypes`: pandas dtype inference from the same native cells
  already scanned for a plain XLSX sheet, with ``pd.read_excel`` as the fallback
  for other layouts; :func:`inferred_dtypes_html`: ``pd.read_html`` (None when
  pandas lacks its HTML parser, as in the default install).

Assumed date contract (AREA A, developed in parallel): Excel date cells are
read as ISO 8601 text per column (``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM:SS``,
``HH:MM:SS``, durations ``[-]H:MM:SS``), so the date detector sees one ISO
format and its renderer writes the same ISO text back; the writer parses that
text into real date cells with the column's source number format. If AREA A
cannot type such a column, the value rules apply and a value that is not an
ISO date is written as text (the self-check then reports the type change).

Anything that may quote real data (title/notes/footer cells, comments,
document properties) stays in ``_``-prefixed layout keys (never reported).
``WriteRequest.rng_for`` seeds the placeholder text.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..model import FileLayout, ReadResult
from . import excel_html, excel_read, excel_storage, excel_write

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ReadRequest, WriteRequest


def read_xlsx(req: "ReadRequest") -> ReadResult:
    """Read an .xlsx/.xlsm workbook (one RawTable per worksheet; see the module docstring)."""
    return excel_read.read_xlsx(req)


def read_xls(req: "ReadRequest") -> ReadResult:
    """Read a real (OLE2/BIFF) .xls workbook with xlrd; it is written as .xlsx."""
    return excel_read.read_xls(req)


def write_xlsx(req: "WriteRequest") -> None:
    """Write a fresh .xlsx workbook mirroring the source layout (see :mod:`.excel_write`)."""
    excel_write.write_xlsx(req)


def read_html_table(req: "ReadRequest") -> ReadResult:
    """Read an HTML table disguised as .xls (see :mod:`.excel_html`)."""
    return excel_html.read_html(req)


def write_html_table(req: "WriteRequest") -> None:
    """Write the synthetic table back as the same disguised HTML (see :mod:`.excel_html`)."""
    excel_html.write_html(req)


def detect_native(ctx: Any) -> Any:
    """Type implied by Excel cell storage.

    Called by ``detect/types.py`` after the explicit overrides and before the
    value rules. Columns whose cells are mostly Excel date cells (canonical ISO
    text, see the module docstring) go to the AREA A date detector first, so a
    date column is a date column whatever its cardinality. Every other storage
    type returns None (no opinion): text, number and boolean cells are typed by
    the value rules on their canonical strings, and the writer restores each
    cell's storage type from ``native``.
    """
    nat = ctx.native
    if not nat or nat.get("kind") != "excel" or nat.get("main") != "d":
        return None
    from ..detect import dates

    det = dates.detect_dates(ctx)
    if det is None:
        return None
    det.reason = f"Excel date cells ({nat.get('date_kind') or 'date'}); " + (det.reason or "")
    return det


def layout_diff(source: FileLayout, output: FileLayout, source_tables: list[dict[str, Any]],
                output_tables: list[Any]) -> list[str]:
    """Excel-specific layout differences for the self-check.

    ``source_tables``: the source table layouts (``RawTable.layout`` dicts);
    ``output_tables``: the re-read output RawTables. Compared: sheet names,
    order and visibility, date system, and per sheet the header row, the
    number of title and footer rows, the column span, freeze panes, autofilter,
    column widths, merged ranges, and per column the counts of each cell type
    and of each number format within a type (the source mix scaled to the
    written cells, :func:`.excel_storage.storage_issues`) and the main cell
    type (columns matched by header in order, so ``--exclude`` is allowed; a
    column with no written value cannot show a type and is skipped). HTML
    "Excel" tables: each column's cell templates per content class
    (:func:`.excel_html.template_issues`).
    """
    issues: list[str] = []
    s_ex, o_ex = source.extra or {}, output.extra or {}
    s_sheets = [(s.get("name"), s.get("state")) for s in s_ex.get("sheets") or []]
    o_sheets = [(s.get("name"), s.get("state")) for s in o_ex.get("sheets") or []]
    if [n for n, _ in s_sheets] != [n for n, _ in o_sheets]:
        issues.append("sheet names/order differ from the source")
    elif s_sheets != o_sheets:
        issues.append("sheet visibility differs from the source")
    if s_ex.get("date_system") != o_ex.get("date_system"):
        issues.append(f"date system: source {s_ex.get('date_system')}, output {o_ex.get('date_system')}")
    for sl, ot in zip(source_tables, output_tables):
        ol = getattr(ot, "layout", {}) or {}
        where = f"sheet {getattr(ot, 'name', None)!r}: "
        for key, label in (("notes_only", "notes-only sheet"), ("header_row", "header row"),
                           ("n_title_rows", "title rows"), ("n_footer_rows", "footer rows"),
                           ("first_col", "first column"), ("freeze_panes", "freeze panes"),
                           ("autofilter", "autofilter range"), ("widths", "column widths"),
                           ("merged", "merged ranges")):
            a, b = sl.get(key), ol.get(key)
            if key in ("merged", "widths"):
                a, b = sorted(map(str, a or [])), sorted(map(str, b or []))
            if key in ("autofilter", "widths") and a != b and _excluded_shift(sl, ot):
                continue  # --exclude moved the columns (the writer shifts the range and the widths)
            if a != b and not (key == "notes_only" and not a and not b):
                issues.append(f"{where}{label}: source {a!r}, output {b!r}")
        issues.extend(where + i for i in _column_diff(sl.get("columns") or [], ot))
        issues.extend(where + i for i in _html_template_diff(sl, ot))
    return issues


def _html_template_diff(sl: dict[str, Any], ot: Any) -> list[str]:
    """HTML "Excel" tables: each column's mix of cell templates (class/style, i.e. Excel's number format) per
    content class, by exact quota (:func:`.excel_html.template_issues`); columns matched by header in order."""
    ol = getattr(ot, "layout", {}) or {}
    stb, otb = sl.get("_html_table"), ol.get("_html_table")
    if stb is None or otb is None or sl.get("notes_only") or ol.get("notes_only"):
        return []
    h, oh = int(sl.get("header_row") or 1), int(ol.get("header_row") or 1)
    s_rows = stb.rows[h:int(sl.get("data_end") or h)]
    o_rows = otb.rows[oh:int(ol.get("data_end") or oh)]
    s_names = [c.text for c in stb.rows[h - 1].cells] if 0 < h <= len(stb.rows) else []
    out: list[str] = []
    j = 0
    for k, name in enumerate(getattr(ot, "header", []) or []):
        while j < len(s_names) and s_names[j] != name:
            j += 1
        if j >= len(s_names):
            break
        out.extend(excel_html.template_issues(name, excel_html.column_templates(s_rows, j),
                                              excel_html.column_templates(o_rows, k)))
        j += 1
    return out


def _excluded_shift(sl: dict[str, Any], ot: Any) -> bool:
    return len(sl.get("columns") or []) != len(getattr(ot, "columns", []) or [])


def _column_diff(src_cols: list[list[Any]], ot: Any) -> list[str]:
    """Per column: the mix of cell types and number formats (exact quota, :mod:`.excel_storage`), then the
    main cell type (a change the synthetic values force, e.g. labels where the source had numbers)."""
    out: list[str] = []
    j = 0
    for col in getattr(ot, "columns", []) or []:
        while j < len(src_cols) and src_cols[j][0] != col.name:
            j += 1
        if j >= len(src_cols):
            break
        name, s_type, s_fmt, share, storage = (list(src_cols[j]) + [None] * 5)[:5]
        j += 1
        nat = col.native or {}
        mix = (excel_storage.storage_issues(name, storage, col.values.tolist(), nat)
               if isinstance(storage, dict) else [])
        out.extend(mix)
        o_type = nat.get("main")
        if mix or s_type is None or o_type is None:
            continue
        if share is not None and share < 0.6:
            continue  # no dominant storage type in the source
        o_counts = nat.get("types") or {}
        tie = bool(o_counts.get(s_type)) and o_counts.get(s_type) == o_counts.get(o_type)
        if s_type != o_type and not tie:
            out.append(f"column {name!r}: cell type {s_type} -> {o_type}")
            continue
        if isinstance(storage, dict):
            continue  # the formats of every type were compared exactly above
        o_type = s_type  # a tie between the source's main type and another one is not a change
        o_fmt = (nat.get("formats") or {}).get(o_type)
        if (s_fmt or "General") != (o_fmt or "General"):
            out.append(f"column {name!r}: number format {s_fmt!r} -> {o_fmt!r}")
    return out


def inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes ``pd.read_excel`` infers for sheet ``table`` read at its header row over the data rows.

    Plain XLSX sheets retain only their inferred dtype strings in private
    layout extras. Reuse requires unchanged file identity, window and pandas
    string inference options. None when unknown (notes-only/empty sheet,
    unreadable file).
    """
    info = next((s for s in (layout.extra or {}).get("sheets") or [] if s.get("name") == table), None)
    if info is None or info.get("notes_only") or not info.get("header_row"):
        return None
    import pandas as pd

    cached = layout.extra.get("_xlsx_dtypes") or {}
    sheet = (cached.get("sheets") or {}).get(table)
    if (sheet is not None and cached.get("file") == excel_read.dtype_file_key(path)
            and cached.get("infer_string") == pd.options.future.infer_string
            and sheet.get("window") == excel_read.dtype_window(info)):
        return list(sheet["dtypes"])
    first, last = int(info["first_col"]), int(info["last_col"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as fh:
                df = pd.read_excel(fh, sheet_name=table, header=int(info["header_row"]) - 1,
                                   nrows=int(info.get("n_rows") or 0), usecols=list(range(first - 1, last)))
    except Exception:  # noqa: BLE001 - unknown, not an error
        return None
    return [str(t) for t in df.dtypes]


def inferred_dtypes_html(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes ``pd.read_html`` infers for an HTML table saved as .xls (None if pandas cannot parse HTML)."""
    return excel_html.inferred_dtypes(path, layout, table)
