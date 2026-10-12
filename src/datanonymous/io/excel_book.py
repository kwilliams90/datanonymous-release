"""OOXML package inspection for the Excel reader (AREA E helper).

openpyxl's read-only mode streams cell values but does not expose the sheet
layout (column widths, freeze panes, autofilter, merged ranges) or formulas
together with their cached values. This module reads those parts straight from
the ``.xlsx/.xlsm`` zip with ``xml.etree.ElementTree.iterparse`` (streaming,
so large sheets stay cheap):

* :func:`read_book` - sheet list (name, state, kind, part path) in workbook
  order, the 1900/1904 date system, the active tab, and the workbook-level
  Excel extras that the fresh-workbook writer drops (pivot tables and caches,
  charts, drawings, comments, external links, macros, connections, named
  ranges, custom properties, document properties, workbook protection).
* :func:`read_sheet_xml` - one worksheet's layout (freeze panes, column widths,
  autofilter range, merged ranges), its sheet-level extras (hyperlinks, data
  validation, conditional formatting, protection, Excel tables, autofilter
  criteria, page header/footer text ...), and its formula cells: per-column
  row numbers, the count of formula cells without a cached value, and the
  formula text (shared formulas translated) of the last formula-bearing rows,
  which is all the footer-total recomputation needs.

Nothing here is reported verbatim except counts and coordinates: extras are
counted by kind, never quoted (comments and properties can hold real names).
"""

from __future__ import annotations

import bisect
import io
import posixpath
import re
import zipfile
from array import array
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

#: formula-bearing rows whose formula text is kept (footer detection looks at the last rows only)
FORMULA_TAIL_ROWS = 64

_CELL_REF = re.compile(r"([A-Za-z]{1,3})(\d+)$")


def local(tag: str) -> str:
    """Element/attribute name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def split_ref(ref: str) -> tuple[int, int] | None:
    """'AB12' -> (row 12, column 28); None if not a cell reference."""
    m = _CELL_REF.match(ref.replace("$", ""))
    if not m:
        return None
    return int(m.group(2)), col_index(m.group(1))


@dataclass
class SheetPart:
    """One sheet of the workbook, in workbook order."""

    name: str
    state: str  # 'visible' | 'hidden' | 'veryHidden'
    kind: str  # 'worksheet' | 'chartsheet' | 'dialogsheet' | 'macrosheet'
    path: str | None  # zip member


@dataclass
class BookInfo:
    sheets: list[SheetPart]
    date1904: bool = False
    active_tab: int = 0
    extras: Counter = field(default_factory=Counter)
    has_macros: bool = False


@dataclass
class SheetXml:
    """Layout, extras and formula facts of one worksheet part."""

    freeze_panes: str | None = None
    widths: list[list[Any]] = field(default_factory=list)  # [min_col, max_col, width, hidden]
    default_width: float | None = None
    autofilter: str | None = None
    merged: list[str] = field(default_factory=list)
    extras: Counter = field(default_factory=Counter)
    formula_rows: dict[int, array] = field(default_factory=dict)  # column -> rows holding a formula
    n_formulas: int = 0
    n_formulas_uncached: int = 0
    uncached: set = field(default_factory=set)  # (row, col) of formula cells without a cached value
    formula_tail: deque = field(default_factory=lambda: deque(maxlen=FORMULA_TAIL_ROWS))  # (row, {col: text})


_EXTRA_PARTS = (
    ("xl/pivotTables/", "pivot tables"),
    ("xl/pivotCache/pivotCacheDefinition", "pivot caches"),
    ("xl/charts/chart", "charts"),
    ("xl/chartsheets/", None),  # counted from the sheet list
    ("xl/drawings/drawing", "drawings (images, shapes, charts)"),
    ("xl/media/", "embedded media"),
    ("xl/threadedComments/", "threaded comments"),
    ("xl/externalLinks/externalLink", "external links"),
    ("xl/connections.xml", "data connections"),
    ("xl/queryTables/", "query tables"),
    ("xl/slicers/", "slicers"),
    ("xl/timelines/", "timelines"),
    ("xl/embeddings/", "embedded objects"),
    ("customXml/item", "custom XML parts"),
    ("docProps/custom.xml", "custom document properties"),
)


def _rels(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Relationship id -> (type suffix, absolute target) for ``part``."""
    d, base = posixpath.split(part)
    rels_path = posixpath.join(d, "_rels", base + ".rels")
    try:
        root = ET.fromstring(zf.read(rels_path))
    except KeyError:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for rel in root:
        target = rel.get("Target") or ""
        if rel.get("TargetMode") == "External":
            continue
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join(d, target))
        out[rel.get("Id") or ""] = ((rel.get("Type") or "").rsplit("/", 1)[-1], path)
    return out


def _workbook_part(zf: zipfile.ZipFile) -> str:
    for rid, (typ, path) in _rels(zf, "").items():
        if typ == "officeDocument":
            return path
    return "xl/workbook.xml"


def read_book(zf: zipfile.ZipFile) -> BookInfo:
    """Sheets, date system, active tab and workbook-level extras of an .xlsx/.xlsm package."""
    names = zf.namelist()
    wb_part = _workbook_part(zf)
    rels = _rels(zf, wb_part)
    root = ET.fromstring(zf.read(wb_part))
    info = BookInfo(sheets=[])
    view_seen = False
    for el in root.iter():
        tag = local(el.tag)
        if tag == "workbookPr":
            info.date1904 = (el.get("date1904") or "").lower() in ("1", "true")
        elif tag == "workbookView" and not view_seen:
            view_seen = True
            try:
                info.active_tab = int(el.get("activeTab") or 0)
            except ValueError:
                info.active_tab = 0
        elif tag == "sheet":
            rid = next((v for k, v in el.attrib.items() if local(k) == "id"), "")
            typ, path = rels.get(rid, ("", None))
            kind = {"worksheet": "worksheet", "chartsheet": "chartsheet", "dialogsheet": "dialogsheet",
                    "xlMacrosheet": "macrosheet", "xlIntlMacrosheet": "macrosheet"}.get(typ, "worksheet")
            info.sheets.append(SheetPart(el.get("name") or "", el.get("state") or "visible", kind, path))
        elif tag == "definedName":
            name = el.get("name") or ""
            if not name.startswith("_xlnm."):
                info.extras["named ranges"] += 1
        elif tag == "workbookProtection" and any(v not in ("0", "false") for v in el.attrib.values()):
            info.extras["workbook protection"] += 1
    for prefix, label in _EXTRA_PARTS:
        if label is None:
            continue
        n = sum(1 for n_ in names if n_.startswith(prefix) and not n_.endswith(".rels") and not n_.endswith("/"))
        if n:
            info.extras[label] += n
    n_comments = 0
    for n_ in names:
        if re.match(r"xl/comments\d*\.xml$", n_) or re.match(r"xl/comments/comment\d*\.xml$", n_):
            n_comments += len(re.findall(rb"<(?:\w+:)?comment[\s>]", zf.read(n_)))
    if n_comments:
        info.extras["comments"] += n_comments
    if any(n_.lower().endswith("vbaproject.bin") for n_ in names):
        info.has_macros = True
        info.extras["macros (VBA)"] += 1
    if "docProps/core.xml" in names:
        core = ET.fromstring(zf.read("docProps/core.xml"))
        if any((el.text or "").strip() for el in core.iter()
               if local(el.tag) in ("creator", "lastModifiedBy", "title", "subject", "description", "keywords",
                                    "category")):
            info.extras["document properties (author, title ...)"] += 1
    n_charts = sum(1 for s in info.sheets if s.kind == "chartsheet")
    if n_charts:
        info.extras["chart sheets"] += n_charts
    n_other = sum(1 for s in info.sheets if s.kind in ("dialogsheet", "macrosheet"))
    if n_other:
        info.extras["dialog/macro sheets"] += n_other
    return info


_SHEETDATA_OPEN = re.compile(rb"<(?:[\w.-]+:)?sheetData(?:\s[^>]*)?(/?)>")
_SHEETDATA_CLOSE = re.compile(rb"</(?:[\w.-]+:)?sheetData\s*>")
_FORMULA_TAG = re.compile(rb"<(?:[\w.-]+:)?f[\s>/]")


def read_sheet_xml(zf: zipfile.ZipFile, path: str) -> SheetXml:
    """One worksheet part: layout, extras and formula facts (see module docstring).

    A sheet without formula cells is parsed without its ``sheetData`` (only the
    layout elements are needed), which keeps large plain exports cheap.
    """
    data = zf.read(path)
    m = _SHEETDATA_OPEN.search(data)
    if m is not None and not m.group(1):
        end = _SHEETDATA_CLOSE.search(data, m.end())
        if end is not None and _FORMULA_TAG.search(data, m.end(), end.start()) is None:
            data = data[:m.end()] + data[end.start():]
    return _parse_sheet(io.BytesIO(data))


def _parse_sheet(fh: Any) -> SheetXml:
    out = SheetXml()
    shared: dict[str, tuple[str, str]] = {}  # si -> (formula text, origin cell)
    row_no = 0
    col_no = 0
    cur_row_formulas: dict[int, str] = {}
    sheet_data = None
    in_view = 0
    views_seen = 0
    for event, el in ET.iterparse(fh, events=("start", "end")):
        tag = local(el.tag)
        if event == "start":
            if tag == "sheetData":
                sheet_data = el
            elif tag == "row":
                r = el.get("r")
                row_no = int(r) if r else row_no + 1
                col_no = 0
                cur_row_formulas = {}
            elif tag == "sheetView":
                in_view = views_seen == 0
                views_seen += 1
            continue
        # ---- end events
        if tag == "c":
            ref = el.get("r")
            pos = split_ref(ref) if ref else None
            if pos is not None:
                col_no = pos[1]
            else:
                col_no += 1
            f_el = None
            v_el = None
            for child in el:
                ct = local(child.tag)
                if ct == "f":
                    f_el = child
                elif ct == "v":
                    v_el = child
            if f_el is not None:
                out.n_formulas += 1
                if v_el is None or not (v_el.text or ""):
                    out.n_formulas_uncached += 1
                    out.uncached.add((row_no, col_no))
                rows = out.formula_rows.get(col_no)
                if rows is None:
                    rows = out.formula_rows[col_no] = array("I")
                rows.append(row_no)
                cur_row_formulas[col_no] = _formula_text(f_el, ref, shared)
        elif tag == "row":
            if cur_row_formulas:
                out.formula_tail.append((row_no, cur_row_formulas))
            if sheet_data is not None:
                sheet_data.clear()
        elif tag == "pane" and in_view:
            if (el.get("state") or "") in ("frozen", "frozenSplit"):
                tl = el.get("topLeftCell")
                if not tl:
                    xs = int(float(el.get("xSplit") or 0))
                    ys = int(float(el.get("ySplit") or 0))
                    tl = _cell_name(ys + 1, xs + 1)
                out.freeze_panes = tl
        elif tag == "sheetView":
            in_view = False
        elif tag == "sheetFormatPr":
            dw = el.get("defaultColWidth")
            out.default_width = float(dw) if dw else None
        elif tag == "col":
            try:
                lo, hi = int(el.get("min") or 0), int(el.get("max") or 0)
            except ValueError:
                continue
            width = el.get("width")
            hidden = (el.get("hidden") or "") in ("1", "true")
            if lo and hi and (width is not None or hidden):
                out.widths.append([lo, min(hi, 16384), float(width) if width else None, hidden])
        elif tag == "autoFilter":
            out.autofilter = el.get("ref")
            n_crit = sum(1 for ch in el if local(ch.tag) == "filterColumn")
            if n_crit:
                out.extras["autofilter criteria"] += n_crit
        elif tag == "sortState":
            out.extras["sort state"] += 1
        elif tag == "mergeCell":
            ref = el.get("ref")
            if ref:
                out.merged.append(ref)
        elif tag == "hyperlink":
            out.extras["hyperlinks"] += 1
        elif tag == "dataValidation":
            out.extras["data validation"] += 1
        elif tag == "conditionalFormatting":
            out.extras["conditional formatting"] += 1
        elif tag in ("sheetProtection", "protectedRange"):
            out.extras["sheet protection"] += 1
        elif tag == "tablePart":
            out.extras["Excel tables (ListObjects)"] += 1
        elif tag in ("oleObject", "control"):
            out.extras["embedded objects/controls"] += 1
        elif tag == "picture":
            out.extras["background picture"] += 1
        elif tag == "customSheetView":
            out.extras["custom views"] += 1
        elif tag == "scenario":
            out.extras["scenarios"] += 1
        elif tag in ("oddHeader", "oddFooter", "evenHeader", "evenFooter", "firstHeader", "firstFooter"):
            if (el.text or "").strip():
                out.extras["page header/footer text"] += 1
    return out


def _cell_name(row: int, col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def _formula_text(f_el: ET.Element, ref: str | None, shared: dict[str, tuple[str, str]]) -> str:
    """'=...' text of a formula element; shared formulas are translated from their master cell."""
    text = f_el.text or ""
    ftype = f_el.get("t")
    if ftype == "shared":
        si = f_el.get("si") or ""
        if text:
            shared[si] = (text, ref or "")
        elif si in shared and ref:
            master, origin = shared[si]
            try:
                from openpyxl.formula.translate import Translator

                return Translator("=" + master, origin=origin).translate_formula(ref)
            except Exception:  # noqa: BLE001 - an untranslatable formula is just not recomputed
                return "="
    return "=" + text


def formula_count_between(sx: SheetXml, col: int, lo: int, hi: int) -> int:
    """Formula cells of column ``col`` within rows ``lo..hi`` (inclusive)."""
    rows = sx.formula_rows.get(col)
    if not rows:
        return 0
    return bisect.bisect_right(rows, hi) - bisect.bisect_left(rows, lo)
