"""HTML tables saved as ``.xls`` (common SIS "Excel" exports): read and mirror (AREA E helper; design 4).

Reading uses the standard library ``html.parser`` (lxml/bs4 are not
dependencies). The document is tokenised once; every top-level ``<table>`` is
one table (named ``None`` when the file has a single table, else ``Table 1``,
``Table 2`` ...). Cell text is the browser-visible text: entities decoded,
whitespace runs collapsed, ``&nbsp;`` as blank. Within a table, the header
row, title rows above it and total/note rows below the data are found with the
same heuristics as Excel sheets (:class:`~.excel_read.SheetScan`; numeric-looking
cell text counts as a number cell). ``colspan``/``rowspan`` in the header or data
rows and nested tables are refused with a fix (save as .csv/.xlsx).

Writing mirrors the disguised format: same encoding and BOM, the document
skeleton in the same order (a doctype with a W3C public identifier kept, any
other neutralized to ``<!DOCTYPE html>``; the XML declaration kept, other
processing instructions dropped: their text can name a person;
``<html>/<head>/<style>`` CSS kept, ``<script>`` and comments dropped,
attributes reduced to a formatting allow-list such as
``style``/``class``/``align``/``width`` whose values are formatting tokens
only; Excel round-trip value attributes such as ``x:num`` dropped; ``id`` is
never copied; classes and CSS selectors are an allowlist (D34:
:func:`safe_class`, :func:`safe_selector`): Excel's generated classes
(``xl65``, ``font5``) and classes of ordinary formatting words, HTML element
names and a few pseudo-classes, so a selector or class holding a name
(``td.quixote``, ``Thibodeaux {...}``) is dropped with its rule;
``lang``/``dir``/``xmlns`` keep only formatting values, ``font-family`` only
known fonts, and a declaration holding free text (a custom property, a
counter or list-style name, a quoted string other than a known font or a
number format) is dropped), text outside the tables, title
rows, captions and footer text as same-shape placeholders (``--keep-titles``:
verbatim; "Total" labels kept; footer numbers blank), the header row verbatim,
and the N synthetic data rows with the most common source row tag and row
separator. Each column's cell templates (tag and class/style attributes,
which carry Excel's ``mso-number-format``) keep their source mix by exact
quota per content class (empty, text, number, number whose text a number
cell would change such as ``004100``): a minority template is never dropped,
and a zero-padded ID keeps the text-format template such IDs had (R3-05).
"""

from __future__ import annotations

import html
import re
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ReadError, WriteError
from ..model import FileLayout, Message, RawColumn, RawTable, ReadResult
from . import excel_storage, sniff
from .excel_read import _WRITTEN, VALUE_TYPES_CAP, SheetScan, XCell, notes_like, path_key, register_written
from .excel_storage import is_canonical_number
from .excel_write import _KEEP_LABEL_RE, Placeholders
from .numfmt import safe_css_number_format

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ReadRequest, WriteRequest

_NUMERIC_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?%?|[+-]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_ALLOWED_ATTRS = frozenset({
    "style", "class", "align", "valign", "width", "height", "nowrap", "bgcolor", "border", "cellpadding",
    "cellspacing", "colspan", "rowspan", "span", "dir", "lang", "charset", "http-equiv", "type", "media", "frame",
    "rules", "xmlns", "xmlns:o", "xmlns:x", "xmlns:v", "xmlns:html", "x:str", "scope", "bordercolor",
})
_VOID = frozenset({"br", "meta", "link", "img", "hr", "input", "col", "area", "base", "wbr", "param", "source"})
#: Excel's HTML CSS can carry the printed page header/footer text (``mso-header-data:"&LPrinted by ..."``)
_PAGE_TEXT_RE = re.compile(r"mso-(?:header|footer)-data\s*:\s*(?:\"[^\"]*\"|'[^']*'|[^;}]*)\s*;?", re.I)
#: CSS comments (``/* exported for jsmith */``) are free text: dropped
_CSS_COMMENT_RE = re.compile(r"/\*.*?(?:\*/|$)", re.S)
#: ``@import url(...)`` statements (paths can name the exporting user): dropped
_CSS_IMPORT_RE = re.compile(r"@import\b(?:\"[^\"]*\"|'[^']*'|[^;{}])*;?", re.I)
#: one CSS declaration ``name: value;`` (quoted values may hold ';')
_CSS_DECL_RE = re.compile(r"(?<![\w-])([A-Za-z-]+)\s*:\s*((?:\"[^\"]*\"|'[^']*'|[^;{}\"'])*)(;?)")
#: declarations whose value is text, a name or a location, not formatting
_CSS_TEXT_PROPS = frozenset({"content", "quotes", "mso-header-data", "mso-footer-data", "src", "counter-reset",
                             "counter-increment", "counter-set", "list-style", "list-style-type", "animation",
                             "animation-name", "grid-area", "grid-template-areas", "string-set", "bookmark-label"})
#: declarations whose quoted strings are checked on their own (fonts, number formats, Excel's style names)
_CSS_QUOTED_OK = frozenset({"font-family", "font", "mso-number-format", "mso-style-name", "mso-font-charset"})
_CSS_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
#: Words of Excel's built-in cell style names (``mso-style-name:Normal``, ``"Comma \[0\]"``, ``"20% - Accent1"``):
#: any other style name is free text (a user's style can be named after a person) and is dropped.
_EXCEL_STYLE_WORDS = frozenset({
    "normal", "comma", "currency", "percent", "hyperlink", "followed", "good", "bad", "neutral", "title", "heading",
    "total", "input", "output", "calculation", "check", "cell", "linked", "note", "warning", "text", "explanatory",
    "accent", "default", "style", "standard",
})


def clean_css(css: str) -> str:
    """CSS of a ``<style>`` block with free-text parts removed (formatting such as mso-number-format kept).

    Dropped: comments, ``@import``, the page header/footer text
    (``mso-header-data``), and declarations whose value is text or a location
    (``content``, ``quotes``, ``src``, anything with ``url(...)``). An
    ``mso-number-format`` keeps its formatting with its literal text masked
    (``io.numfmt``: ``"\\0022Zelmira Quenby\\0022 0"`` can quote a name).
    """
    css = _CSS_COMMENT_RE.sub("", css)
    css = _CSS_IMPORT_RE.sub("", css)
    css = _PAGE_TEXT_RE.sub("", css)

    def decl(m: re.Match) -> str:
        prop = m.group(1).lower()
        if prop in _CSS_TEXT_PROPS or prop.startswith("--") or "url(" in m.group(2).lower():
            return ""
        if prop == "font-family":
            fonts = _known_fonts(m.group(2))
            return f"{m.group(1)}:{fonts}{m.group(3)}" if fonts else ""
        if prop == "font" and _CSS_QUOTED.search(m.group(2)):
            value = re.sub(r"\s{2,}", " ", _FONT_FAMILY_RE.sub(_font_family, m.group(2))).strip()
            return f"{m.group(1)}:{value}{m.group(3)}" if value else ""
        if prop == "mso-style-name":
            words = re.findall(r"[A-Za-z]+", m.group(2))
            return m.group(0) if all(w.casefold() in _EXCEL_STYLE_WORDS for w in words) else ""
        if prop not in _CSS_QUOTED_OK and any(ch.isalpha() for q in _CSS_QUOTED.findall(m.group(2)) for ch in q):
            return ""  # a quoted string with letters is free text ('"\."' of a decimal separator is not)
        if m.group(1).lower() == "mso-number-format":
            value = m.group(2)
            safe = safe_css_number_format(value)
            if safe != value:
                return f"{m.group(1)}:{safe}{m.group(3)}"
        return m.group(0)

    css = _CSS_RULE_RE.sub(_rule_without_names, css)
    return _CSS_DECL_RE.sub(decl, css)


#: Words of font family names (``Calibri``, ``Times New Roman``, ``MS Sans Serif``, ``Arial Unicode MS``): a
#: ``font-family`` keeps the families made of these words only; any other family (``"Philomena Quixote
#: Script"``: free text that can hold a name) is dropped (D34). One literal collection of more than 50 entries.
_FONT_WORDS: frozenset[str] = frozenset({
    "arial", "calibri", "cambria", "times", "new", "roman", "courier", "verdana", "tahoma", "helvetica", "georgia",
    "garamond", "segoe", "ui", "aptos", "consolas", "lucida", "console", "sans", "serif", "mono", "monospace",
    "unicode", "ms", "gothic", "pgothic", "mincho", "pmincho", "meiryo", "yu", "simsun", "simhei", "nsimsun",
    "narrow", "black", "bold", "light", "semibold", "semilight", "condensed", "symbol", "wingdings", "webdings",
    "trebuchet", "franklin", "century", "book", "antiqua", "palatino", "linotype", "bookman", "old", "style",
    "comic", "impact", "candara", "constantia", "corbel", "ebrima", "gadugi", "leelawadee", "malgun",
    "microsoft", "jhenghei", "yahei", "himalaya", "nirmala", "sylfaen", "batang", "gulim", "dotum", "gungsuh",
    "mingliu", "pmingliu", "dengxian", "fangsong", "kaiti", "inherit", "initial", "system", "cursive", "fantasy",
    "emoji", "math", "display", "text", "medium", "heavy", "extra", "regular", "italic", "body", "headings",
    "minor", "major", "latin", "theme", "noto", "roboto", "open", "lato", "montserrat", "source", "pro", "code",
    "inter", "poppins", "raleway", "ubuntu", "oswald", "nunito", "merriweather", "playfair", "rockwell",
    "baskerville", "bodoni", "didot", "futura", "gill", "optima", "perpetua", "elephant", "castellar",
    "algerian", "broadway", "stencil", "haettenschweiler", "papyrus", "harrington", "chiller", "jokerman",
})


#: A quoted family of a ``font`` shorthand with the comma before or after it.
_FONT_FAMILY_RE = re.compile(r"(\s*,\s*)?(\"[^\"]*\"|'[^']*')(\s*,)?")


def _font_family(m: re.Match) -> str:
    """A quoted family of a ``font`` shorthand: kept when it is a known font, else dropped with one comma."""
    if _known_fonts(m.group(2)):
        return m.group()
    return ", " if m.group(1) and m.group(3) else " "


@lru_cache(maxsize=1024)
def _known_fonts(value: str) -> str:
    """The families of a ``font-family`` value made of known font words only (quoted as written)."""
    keep = []
    for fam in value.split(","):
        name = fam.strip().strip("\"'").strip()
        words = re.findall(r"[A-Za-z]+|\d+", name)
        if name and words and all(w.casefold() in _FONT_WORDS or (w.isdigit() and len(w) <= 2) for w in words):
            keep.append(fam.strip())
    return ",".join(keep)


#: one CSS rule ``selectors { declarations }`` (no nesting)
_CSS_RULE_RE = re.compile(r"([^{}]*)\{([^{}]*)\}")
#: one token of a selector: a combinator, ``*``, an element name, a class, an id, a pseudo-class or -element, an
#: attribute selector or an at-rule keyword
_SELECTOR_TOKEN = re.compile(r"\s*(?:<!--|-->|[>+~,]|\*|[A-Za-z][A-Za-z0-9]*|\.-?[A-Za-z_][\w-]*|#-?[A-Za-z_][\w-]*"
                             r"|::?[A-Za-z-]+(?:\([^()]*\))?|\[[^\]]*\]|@[A-Za-z-]+)\s*")
#: HTML element names a copied selector may name (Excel writes ``td``, ``br``, ``col``, ``tr``, ``ruby``, ``rt``).
_HTML_TAGS = frozenset({
    "html", "head", "body", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "col", "colgroup", "caption", "br",
    "p", "div", "span", "a", "font", "b", "i", "u", "s", "strong", "em", "sub", "sup", "small", "big", "center",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "img", "hr", "pre", "code", "ruby",
    "rt", "rp", "nobr", "wbr", "label", "blockquote", "abbr", "address", "mark", "q", "cite", "del", "ins", "kbd",
    "samp", "var", "time", "section", "article", "header", "footer", "main", "nav", "aside", "figure",
    "figcaption", "title", "meta", "link", "style", "input", "button", "select", "option", "textarea", "form",
})
#: Pseudo-classes and -elements a copied selector may use.
_PSEUDOS = frozenset({"link", "visited", "hover", "active", "focus", "first-child", "last-child", "first-line",
                      "first-letter", "before", "after", "nth-child", "nth-of-type", "first", "left", "right",
                      "only-child", "empty"})


def safe_selector(selector: str) -> bool:
    """A CSS selector made of formatting names only, an allowlist (D34): HTML element names, classes that
    :func:`safe_class` keeps (Excel's ``xl65``, ``font5``, ``style0``; ordinary words such as ``header``), a few
    pseudo-classes, combinators and the ``@page`` / ``@font-face`` rules. An id, an attribute selector
    (``td[title="..."]``) or any other name (``Thibodeaux``, ``td.quixote``) is free text that can hold a person's
    name: the rule is dropped."""
    pos = 0
    sel = selector.strip()
    while pos < len(sel):
        m = _SELECTOR_TOKEN.match(sel, pos)
        if m is None or m.end() == pos:
            return False
        tok = m.group().strip()
        pos = m.end()
        if not tok or tok in ">+~,*" or tok in ("<!--", "-->"):  # HTML comment marks around Excel's CSS
            continue
        if tok.startswith("@"):
            if tok.lower() not in ("@page", "@font-face"):
                return False
        elif tok.startswith("."):
            if safe_class(tok[1:]) != tok[1:]:
                return False
        elif tok.startswith(("#", "[")):
            return False
        elif tok.startswith(":"):
            name = tok.lstrip(":").split("(", 1)[0].lower()
            arg = tok.split("(", 1)[1][:-1] if "(" in tok else ""
            if name not in _PSEUDOS or not re.fullmatch(r"\s*(?:odd|even|[0-9n+\-\s]*)\s*", arg):
                return False
        elif tok.lower() not in _HTML_TAGS and not re.fullmatch(r"(?:Word)?Section[0-9]{1,3}", tok):
            return False  # an element name off the list ('Thibodeaux'), an @page name off the list
    return True


def _rule_without_names(m: re.Match) -> str:
    """A CSS rule whose selector is not made of formatting names only (:func:`safe_selector`) is dropped (the
    elements lose a dropped class too: :func:`safe_class`); an HTML comment mark before it stays."""
    if not safe_selector(m.group(1)):
        lead = m.group(1)[:len(m.group(1)) - len(m.group(1).lstrip())]
        return lead + ("<!--" if m.group(1).lstrip().startswith("<!--") else "")
    return m.group(0)


_CELL = ("td", "th")


# --------------------------------------------------------------------------- tokenising
class _Tokens(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[Any, ...]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.events.append(("start", tag, attrs))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.events.append(("startend", tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        self.events.append(("end", tag))

    def handle_data(self, data: str) -> None:
        if self.events and self.events[-1][0] == "data":
            self.events[-1] = ("data", self.events[-1][1] + data)
        else:
            self.events.append(("data", data))

    def handle_comment(self, data: str) -> None:
        self.events.append(("comment", data))

    def handle_decl(self, decl: str) -> None:
        self.events.append(("decl", decl))

    def handle_pi(self, data: str) -> None:
        self.events.append(("pi", data))

    def unknown_decl(self, data: str) -> None:
        self.events.append(("comment", data))


@dataclass
class _Cell:
    tag: str
    attrs: list[tuple[str, str | None]]
    text: str
    closed: bool
    span: int = 1  # colspan
    rowspan: int = 1


@dataclass
class _Row:
    start: int  # event index of <tr>
    end: int  # event index after the row (exclusive)
    attrs: list[tuple[str, str | None]]
    cells: list[_Cell]
    closed: bool


@dataclass
class _Table:
    start: int  # event index of <table>
    end: int  # event index after </table>
    rows: list[_Row] = field(default_factory=list)


def _visible(text: str) -> str:
    return " ".join(text.split())


def _parse_tables(events: list[tuple[Any, ...]]) -> list[_Table]:
    tables: list[_Table] = []
    depth = 0
    cur: _Table | None = None
    row: _Row | None = None
    cell: _Cell | None = None
    buf: list[str] = []

    def close_cell(i: int) -> None:
        nonlocal cell
        if cell is not None and row is not None:
            cell.text = _visible("".join(buf))
            row.cells.append(cell)
        cell = None

    def close_row(i: int, explicit: bool) -> None:
        nonlocal row
        close_cell(i)
        if row is not None and cur is not None:
            row.end = i + 1 if explicit else i
            row.closed = explicit
            cur.rows.append(row)
        row = None

    for i, ev in enumerate(events):
        kind = ev[0]
        if kind == "start" and ev[1] == "table":
            if depth >= 1:
                raise ReadError("nested HTML tables are not supported",
                                fix="open the file in Excel and save it as .xlsx or .csv")
            depth += 1
            cur = _Table(start=i, end=i)
            continue
        if cur is None:
            continue
        if kind == "end" and ev[1] == "table":
            close_row(i, False)
            cur.end = i + 1
            tables.append(cur)
            cur = None
            depth = 0
            continue
        if kind == "start" and ev[1] == "tr":
            close_row(i, False)
            row = _Row(start=i, end=i, attrs=ev[2], cells=[], closed=False)
            continue
        if kind == "end" and ev[1] == "tr":
            close_row(i, True)
            continue
        if kind in ("start", "startend") and ev[1] in _CELL:
            close_cell(i)
            if row is None:
                row = _Row(start=i, end=i, attrs=[], cells=[], closed=False)
            spans = {k: int(v) for k, v in ev[2] if k in ("colspan", "rowspan") and v and v.strip().isdecimal()}
            cell = _Cell(ev[1], ev[2], "", ev[0] == "startend", max(1, spans.get("colspan", 1)),
                         max(1, spans.get("rowspan", 1)))
            buf = []
            if ev[0] == "startend":
                close_cell(i)
            continue
        if kind == "end" and ev[1] in _CELL:
            if cell is not None:
                cell.closed = True
            close_cell(i)
            continue
        if kind in ("start", "end") and ev[1] in ("thead", "tbody", "tfoot", "caption"):
            if ev[1] != "caption":
                close_row(i, False)
            continue
        if cell is not None:
            if kind == "data":
                buf.append(ev[1])
            elif kind in ("start", "startend") and ev[1] == "br":
                buf.append(" ")
    if cur is not None:  # unterminated table
        close_row(len(events), False)
        cur.end = len(events)
        tables.append(cur)
    return tables


# --------------------------------------------------------------------------- reading
def _decode(data: bytes, override: str | None) -> tuple[str, str, bytes]:
    if not override and not data.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        m = re.search(rb"charset\s*=\s*[\"']?([A-Za-z0-9_\-]+)", data[:4096], re.I)
        if m:
            cs = m.group(1).decode("ascii").lower()
            try:
                text = data.decode(cs)
                return text, cs, b""
            except (LookupError, UnicodeDecodeError):
                pass
    info = sniff.sniff_encoding(data, override)
    return sniff.decode(data, info), info.encoding, info.bom


def _notes_table(tb: _Table) -> bool:
    """A table of single-cell rows of multi-word text (a title banner): notes (``excel_read.notes_like``)."""
    texts: list[str] = []
    for row in tb.rows:
        cells = [c.text for c in row.cells if c.text]
        if len(cells) > 1:
            return False
        texts.extend(cells)
    return bool(texts) and notes_like(texts[0], texts[1:])


def _row_cells(row: _Row) -> dict[int, XCell]:
    cells: dict[int, XCell] = {}
    col = 1
    for c in row.cells:
        if c.text:
            t = "n" if _NUMERIC_RE.fullmatch(c.text) else "s"
            cells[col] = XCell(c.text, t, "General")
        col += c.span
    return cells


def read_html(req: "ReadRequest") -> ReadResult:
    data = Path(req.path).read_bytes()
    text, encoding, bom = _decode(data, req.settings.encoding)
    p = _Tokens()
    try:
        p.feed(text)
        p.close()
    except Exception as e:  # noqa: BLE001 - html.parser is lenient; anything else is a broken file
        raise ReadError(f"HTML could not be parsed ({type(e).__name__})", fix="save the file as .csv") from None
    events = p.events
    tables = _parse_tables(events)
    tables = [t for t in tables if t.rows]
    if not tables:
        raise ReadError("the HTML file has no table rows", fix="save the file as .csv or .xlsx")
    known_book = _WRITTEN.pop(path_key(Path(req.path)), None) if req.purpose == "selfcheck" else None
    messages: list[Message] = []
    raw_tables: list[RawTable] = []
    summaries: list[dict[str, Any]] = []
    for k, tb in enumerate(tables):
        name = None if len(tables) == 1 else f"Table {k + 1}"
        known = (known_book or {}).get(str(k))
        if (known or {}).get("notes_only") or (known is None and req.purpose != "selfcheck" and _notes_table(tb)):
            n_rows_tb = len(tb.rows)
            layout = {"notes_only": True, "header_row": n_rows_tb + 1, "n_title_rows": n_rows_tb,
                      "n_footer_rows": 0, "data_end": n_rows_tb + 1, "first_col": 1, "table_index": k,
                      "n_columns": 0, "_html_table": tb}
            raw_tables.append(RawTable(name, [], [], 0, layout))
            summaries.append({"name": name, "notes_only": True, "n_rows": 0, "n_title_rows": n_rows_tb,
                              "n_columns": 0})
            if req.purpose != "selfcheck":
                messages.append(Message("warning", "html_notes_table",
                                        f"HTML table {k + 1} holds notes (one column of short multi-word text), "
                                        "not data: written back as same-shape placeholder text"))
            continue
        scan = SheetScan(name or "table", None, None, known)
        for r, row in enumerate(tb.rows, 1):
            scan.feed(r, _row_cells(row))
        s = scan.finish(None)
        if s.header_row is None or s.first_col is None:
            raise ReadError("the HTML table has no header row", fix="save the file as .csv or .xlsx")
        h = s.header_row
        for r in range(h, s.data_end + 1):
            if any(c.span != 1 or c.rowspan != 1 for c in tb.rows[r - 1].cells):
                raise ReadError("an HTML table with merged cells (colspan/rowspan) in its header or data rows",
                                fix="open the file in Excel and save it as .xlsx or .csv")
        n_cols = max((len(tb.rows[r - 1].cells) for r in range(h, s.data_end + 1)), default=0)
        header = [(tb.rows[h - 1].cells[j].text if j < len(tb.rows[h - 1].cells) else "") for j in range(n_cols)]
        columns: list[RawColumn] = []
        for j in range(n_cols):
            vals = np.empty(s.n_rows, dtype=object)
            for i in range(s.n_rows):
                cells = tb.rows[h + i].cells
                vals[i] = cells[j].text if j < len(cells) else ""
            columns.append(RawColumn(header[j], vals, None))
        layout = {"header_row": h, "n_title_rows": h - 1, "n_footer_rows": len(tb.rows) - s.data_end,
                  "data_end": s.data_end, "first_col": 1, "table_index": k, "n_columns": n_cols,
                  "_html_table": tb}
        if req.purpose != "selfcheck" and known is None:
            cols = [[v for v in c.values.tolist() if v][:5000] for c in columns]
            prob = sniff.header_problems(header, cols)
            if prob is not None:
                raise ReadError("the HTML table's header row does not look like a header: " + "; ".join(prob.reasons),
                                fix="save the file as .csv with a header row")
        raw_tables.append(RawTable(name, header, columns, s.n_rows, layout))
        summaries.append({"name": name, "header_row": h, "n_rows": s.n_rows, "n_title_rows": h - 1,
                          "n_footer_rows": len(tb.rows) - s.data_end, "n_columns": n_cols})
        if req.purpose != "selfcheck" and (h > 1 or s.data_end < len(tb.rows)):
            messages.append(Message("info", "html_table_layout",
                                    f"HTML table {k + 1}: header row {h}, {h - 1} title row(s), "
                                    f"{len(tb.rows) - s.data_end} footer row(s) (written as placeholders)"))
    if req.purpose != "selfcheck":
        n_comments = sum(1 for e in events if e[0] == "comment")
        n_scripts = sum(1 for e in events if e[0] == "start" and e[1] == "script")
        dropped = [f"{what} ({n})" for what, n in (("comments", n_comments), ("scripts", n_scripts)) if n]
        if dropped:
            messages.append(Message("warning", "html_extras_dropped",
                                    "HTML parts not copied to the synthetic file: " + ", ".join(dropped)))
    layout = FileLayout(format="html_xls", encoding=encoding, bom=bom, encoding_confidence="strict",
                        extra={"html_tables": summaries, "_events": events, "_tables": tables})
    return ReadResult(layout, raw_tables, messages)


# --------------------------------------------------------------------------- writing
def _attr_value(v: str) -> str:
    return v.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


_CLASS_SPLIT = re.compile(r"[^A-Za-z]+|(?<=[a-z])(?=[A-Z])")
#: a generated style class (Excel's ``xl65``, ``font5``, ``style0``; LibreOffice's ``ce1``; Google's ``s0``):
#: a short number only (a longer one can be a real ID: ``s482913``)
_GENERATED_CLASS = re.compile(r"(?:xl|font|style|ce|ro|co|ta|gr|pr)[0-9]{1,4}|[scrx][0-9]{1,3}", re.I)


#: Words of ordinary class names (``header``, ``odd``, ``total-row``, ``text-center``): with everyday and header
#: words, the only words a copied class may hold. One word list of more than 50 entries (data).
_CLASS_WORDS: frozenset[str] = frozenset({
    "header", "heading", "head", "body", "foot", "footer", "row", "rows", "col", "cols", "column", "cell", "cells",
    "odd", "even", "alt", "title", "subtitle", "caption", "label", "total", "totals", "subtotal", "sum", "bold",
    "italic", "underline", "center", "centered", "left", "right", "top", "bottom", "middle", "border", "borders",
    "bordered", "highlight", "highlighted", "hidden", "visible", "selected", "active", "inactive", "disabled",
    "data", "text", "num", "number", "numeric", "date", "time", "money", "currency", "percent", "pct", "grid",
    "table", "tbl", "main", "content", "container", "wrapper", "section", "group", "first", "last", "nowrap",
    "wrap", "small", "large", "big", "red", "green", "blue", "yellow", "gray", "grey", "black", "white", "warning",
    "error", "info", "success", "note", "notes", "comment", "link", "sort", "sorted", "sortable", "asc", "desc",
    "sticky", "fixed", "blank", "empty", "spacer", "gridlines", "mso", "normal", "default", "style", "report",
    "page", "print", "screen", "name", "value", "values", "key", "item", "items", "list", "entry", "field", "line",
    "lines", "indent", "pad", "padding", "merge", "merged", "span", "wide", "narrow", "dark", "light", "shade",
    "shaded", "stripe", "striped", "zebra", "band", "banded", "excel", "sheet", "frozen", "freeze", "pane",
})


@lru_cache(maxsize=4096)
def safe_class(value: str) -> str:
    """The class names of an element without those that could hold personal data (D34), an allowlist, never a
    name list: Excel's generated classes (``xl65``, ``font5``, ``style0``; LibreOffice's ``ce1``) stay, and so
    does a class made only of ordinary formatting words (:data:`_CLASS_WORDS`: ``header``, ``odd``,
    ``total-row``) and pieces of one or two letters (``n``, ``x``); any other class is dropped: one with a digit
    (``stu482913``, ``sid-0042``: a real ID) or any other word (``Abernathy-Zelda``, ``row_Okonkwo``,
    ``quixote``). The source's and the output's cell templates are both compared without them."""
    keep = []
    for c in value.split():
        if _GENERATED_CLASS.fullmatch(c):
            keep.append(c)
            continue
        if any(ch.isdigit() for ch in c) or not c.isascii():
            continue
        pieces = [x.lower() for x in _CLASS_SPLIT.split(c) if x]
        if pieces and all(len(x) < 3 or x in _CLASS_WORDS for x in pieces):
            keep.append(c)
    return " ".join(keep)


#: Attribute values that are formatting, never text: a BCP 47 language tag, a direction, the namespaces Excel
#: and Word write. Any other ``lang``/``dir``/``xmlns`` value is dropped (D34: free text can hold a name).
_LANG_RE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3}")
_XMLNS = frozenset({
    "urn:schemas-microsoft-com:office:office", "urn:schemas-microsoft-com:office:excel",
    "urn:schemas-microsoft-com:vml", "urn:schemas-microsoft-com:office:word", "http://www.w3.org/TR/REC-html40",
    "http://www.w3.org/1999/xhtml", "http://www.w3.org/1998/Math/MathML",
})
#: Sizes and counts (``width``, ``height``, ``border``, ``colspan`` ...): a number with an optional unit.
_ATTR_SIZE_RE = re.compile(r"\s*[0-9]{1,4}(?:\.[0-9]{1,4})?(?:px|pt|em|%|in|cm|mm)?\s*|\s*\*\s*")
#: Colours: a hex colour or a CSS colour keyword.
_ATTR_COLOR_RE = re.compile(r"\s*#[0-9A-Fa-f]{3,8}\s*")
_COLOR_WORDS = frozenset({
    "white", "black", "red", "green", "blue", "yellow", "gray", "grey", "silver", "maroon", "navy", "purple",
    "teal", "olive", "lime", "aqua", "fuchsia", "orange", "transparent", "windowtext", "window", "buttonface",
    "auto", "none", "inherit",
})
#: The formatting words each enumerated attribute may hold.
_ATTR_WORDS: dict[str, frozenset[str]] = {
    "align": frozenset({"left", "right", "center", "justify", "char", "top", "middle", "bottom", "absmiddle",
                        "baseline", "general"}),
    "valign": frozenset({"top", "middle", "bottom", "baseline", "center"}),
    "nowrap": frozenset({"nowrap", "true", ""}),
    "frame": frozenset({"void", "above", "below", "hsides", "lhs", "rhs", "vsides", "box", "border"}),
    "rules": frozenset({"none", "groups", "rows", "cols", "all"}),
    "scope": frozenset({"row", "col", "rowgroup", "colgroup"}),
    "media": frozenset({"all", "screen", "print", "handheld", "projection", "tv", "speech"}),
    "http-equiv": frozenset({"content-type", "x-ua-compatible", "content-language", "content-style-type"}),
}
_ATTR_SIZES = frozenset({"width", "height", "border", "cellpadding", "cellspacing", "colspan", "rowspan", "span"})


@lru_cache(maxsize=4096)
def _safe_attr(k: str, v: str) -> bool:
    """The value of a copied attribute is formatting, not free text that could hold a name or an ID (D34), an
    allowlist per attribute: sizes, colours, the enumerated words of ``align``/``valign``/``frame`` ..., an
    encoding name, a MIME type, a language tag, a direction, Excel's and Word's namespaces."""
    import codecs

    v0 = v.strip()
    if k == "lang":
        return bool(_LANG_RE.fullmatch(v0))
    if k == "dir":
        return v0.lower() in ("ltr", "rtl", "auto")
    if k.startswith("xmlns"):
        return v0 in _XMLNS
    if k in _ATTR_SIZES:
        return bool(_ATTR_SIZE_RE.fullmatch(v))
    if k in ("bgcolor", "bordercolor"):
        return bool(_ATTR_COLOR_RE.fullmatch(v)) or v0.lower() in _COLOR_WORDS
    if k in _ATTR_WORDS:
        return v0.lower() in _ATTR_WORDS[k]
    if k == "charset":
        try:
            return bool(re.fullmatch(r"[A-Za-z0-9._:-]{1,40}", v0)) and codecs.lookup(v0) is not None
        except LookupError:
            return False
    if k == "type":
        return bool(re.fullmatch(r"(?:text|application)/[a-z0-9.+-]{1,40}", v0.lower()))
    return False


def _attrs(attrs: list[tuple[str, str | None]], tag: str) -> str:
    out = []
    for k, v in attrs:
        k = k.lower()
        if k not in _ALLOWED_ATTRS:
            continue
        if k == "x:str" and v:
            continue
        if k == "class" and v:
            v = safe_class(v)
            if not v:
                continue
        elif k == "style" and v:
            v = clean_css(v)  # inline CSS: no url(...) / content text either
        elif v and not _safe_attr(k, v):
            continue
        out.append(f" {k}" if v is None else f' {k}="{_attr_value(v)}"')
    if tag == "meta":
        d = {k.lower(): v for k, v in attrs}
        if (d.get("http-equiv") or "").lower() == "content-type" and d.get("content"):
            out.append(f' content="{_attr_value(d["content"] or "")}"')
    return "".join(out)


def _start(tag: str, attrs: list[tuple[str, str | None]], selfclose: bool = False) -> str:
    a = _attrs(attrs, tag)
    if tag == "meta" and not a:
        return ""  # a <meta> left without attributes (author, description ...) is dropped
    return f"<{tag}{a}{' /' if selfclose else ''}>"


def _esc(v: str) -> str:
    return html.escape(v, quote=False)


def _common_sep(events: list[tuple[Any, ...]], rows: list[_Row]) -> str:
    seps = Counter()
    for a, b in zip(rows, rows[1:]):
        gap = events[a.end:b.start]
        if all(e[0] == "data" and not e[1].strip() for e in gap):
            seps["".join(e[1] for e in gap)] += 1
    return seps.most_common(1)[0][0] if seps else "\n"


def cell_class(text: str) -> str:
    """Content class of an HTML data cell, for its formatting template (R3-05).

    Excel reads a number out of a cell unless the cell's style makes it text,
    so a number whose text a number cell would change (``004100``, ``2.10``,
    ``1,234``) is a class of its own: it keeps the templates (``mso-number-format``
    ``\\@``, ``0.00`` ...) that such source cells had.
    """
    t = _visible(text)
    if not t:
        return "empty"
    if not _NUMERIC_RE.fullmatch(t):
        return "text"
    return "number" if is_canonical_number(t) else "number_text"


def _template_key(cell: _Cell) -> str:
    return cell.tag + _attrs(cell.attrs, cell.tag) + ("/" if cell.closed else "")


@dataclass
class ColumnTemplates:
    """Cell templates of one column's data cells: counts per content class and overall, PRIVATELY the
    templates of each distinct value, and key -> (tag, attributes, closed)."""

    keep: dict[str, tuple[str, list[tuple[str, str | None]], bool]] = field(default_factory=dict)
    by_class: dict[str, Counter] = field(default_factory=dict)
    by_value: dict[str, Counter] = field(default_factory=dict)
    overall: Counter = field(default_factory=Counter)

    def expected(self, class_counts: dict[str, int]) -> dict[str, dict[str, int]]:
        """Template counts per content class for ``class_counts`` written cells: the class's source mix
        by exact quota (the column's whole mix for a class the source column does not have)."""
        return {cls: {k: c for k, c in excel_storage.scaled(self.by_class.get(cls) or self.overall, n).items()
                      if c}
                for cls, n in class_counts.items() if n}


def column_templates(rows: list[_Row], j: int) -> ColumnTemplates:
    ct = ColumnTemplates()
    for row in rows:
        if j < len(row.cells):
            cell = row.cells[j]
            key = _template_key(cell)
            ct.overall[key] += 1
            ct.keep.setdefault(key, (cell.tag, cell.attrs, cell.closed))
            ct.by_class.setdefault(cell_class(cell.text), Counter())[key] += 1
            if cell.text and (cell.text in ct.by_value or len(ct.by_value) < VALUE_TYPES_CAP):
                ct.by_value.setdefault(cell.text, Counter())[key] += 1
    if len(ct.overall) <= 1:
        ct.by_value = {}
    return ct


def plan_templates(ct: ColumnTemplates, values: Any, rng: np.random.Generator) -> list[str | None]:
    """Template key of each written cell: per content class, the source templates of that class by exact
    quota; a value that is also a source value takes its source templates first (excel_storage._deal)."""
    n = len(values)
    if not ct.overall:
        return [None] * n
    if len(ct.overall) == 1:
        return [next(iter(ct.overall))] * n
    strings = ["" if values[i] is None else str(values[i]) for i in range(n)]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strings):
        groups.setdefault(cell_class(s), []).append(i)
    out: list[str | None] = [None] * n
    for cls in sorted(groups):
        cells = groups[cls]
        quotas = ct.expected({cls: len(cells)})[cls]

        def pref(s: str, keys: dict = quotas) -> dict[str, int] | None:
            c = ct.by_value.get(s)
            return {k: v for k, v in c.items() if k in keys} if c else None

        for key, lst in excel_storage._deal(cells, quotas, strings, pref if ct.by_value else None, rng).items():
            for i in lst:
                out[i] = key
    return out


def template_issues(name: str, src: ColumnTemplates, out: ColumnTemplates) -> list[str]:
    """Self-check: the written cells' templates per content class must be the source mix by exact quota.
    Templates are not quoted (attribute values are never reported): counts only."""
    have = {cls: dict(c) for cls, c in out.by_class.items()}
    want = src.expected({cls: sum(c.values()) for cls, c in out.by_class.items()})
    issues = []
    for cls in sorted(set(have) | set(want)):
        h, w = have.get(cls, {}), want.get(cls, {})
        if h != w:
            other = "; not the source's styles" if set(h) != set(w) else ""
            issues.append(f"column {name!r}: {sum(h.values())} {cls.replace('_', ' ')} cell(s) written with "
                          f"{len(h)} cell style(s) {sorted(h.values(), reverse=True)}, expected {len(w)} "
                          f"{sorted(w.values(), reverse=True)} (the source mix of cell styles{other})")
    return issues


def _inner_sep(events: list[tuple[Any, ...]], row: _Row) -> tuple[str, str, str]:
    """Whitespace after <tr>, between cells, before </tr> in a source row."""
    seg = events[row.start + 1:row.end - (1 if row.closed else 0)]
    first = seg[0][1] if seg and seg[0][0] == "data" and not seg[0][1].strip() else ""
    last = seg[-1][1] if seg and seg[-1][0] == "data" and not seg[-1][1].strip() else ""
    mids = Counter()
    for a, b, c in zip(seg, seg[1:], seg[2:]):
        if a[0] == "end" and a[1] in _CELL and b[0] == "data" and not b[1].strip() and c[0] == "start":
            mids[b[1]] += 1
    return first, (mids.most_common(1)[0][0] if mids else ""), last


def write_html(req: "WriteRequest") -> None:
    layout = req.layout
    events: list[tuple[Any, ...]] = layout.extra.get("_events") or []
    tables: list[_Table] = layout.extra.get("_tables") or []
    if not events or len(tables) != len(req.tables):
        raise WriteError("HTML layout missing for the synthetic table", fix="report this as a bug")
    rng = req.rng_for("html-placeholders") if req.rng_for else np.random.default_rng(req.seed)
    ph = Placeholders(rng, bool(getattr(req.settings, "keep_titles", False)))

    def rng_for(purpose: str) -> np.random.Generator:
        return req.rng_for(purpose) if req.rng_for else np.random.default_rng([req.seed, zlib.crc32(purpose.encode())])

    out: list[str] = []
    written: dict[str, dict[str, Any]] = {}
    i = 0
    for t_i, (tb, ot) in enumerate(zip(tables, req.tables)):
        i = _emit_plain(events, i, tb.start, out, ph)
        written[str(t_i)] = _emit_table(events, tb, ot, out, ph, rng_for, t_i)
        i = tb.end
    _emit_plain(events, i, len(events), out, ph)
    text = "".join(out)
    codec = layout.encoding or "utf-8"
    data = layout.bom + text.encode(codec, errors="xmlcharrefreplace")
    req.path.write_bytes(data)
    register_written(req.path, written)


def _emit_plain(events: list[tuple[Any, ...]], a: int, b: int, out: list[str], ph: Placeholders) -> int:
    """Document parts outside the tables: skeleton kept, text placeholdered, scripts/comments dropped."""
    in_script = False
    in_style = False
    for ev in events[a:b]:
        kind = ev[0]
        if kind == "start":
            if ev[1] == "script":
                in_script = True
                continue
            if ev[1] == "style":
                in_style = True
            if ev[1] in ("link",):
                continue
            out.append(_start(ev[1], ev[2]))
        elif kind == "startend":
            if ev[1] in ("link", "script"):
                continue
            out.append(_start(ev[1], ev[2], selfclose=True))
        elif kind == "end":
            if ev[1] == "script":
                in_script = False
                continue
            if ev[1] == "style":
                in_style = False
            if ev[1] in _VOID:
                continue
            out.append(f"</{ev[1]}>")
        elif kind == "data":
            if in_script:
                continue
            if in_style:
                out.append(clean_css(ev[1]))  # CSS kept; comments, page header/footer text, urls dropped
            elif not ev[1].strip():
                out.append(ev[1])
            else:
                out.append(_esc(ph.text(ev[1])))
        elif kind == "decl":
            out.append(f"<!{safe_doctype(ev[1])}>")
        elif kind == "pi":
            if _XML_DECL_RE.fullmatch(ev[1]):  # other processing instructions can name a person: dropped
                out.append(f"<?{ev[1]}>")
    return b


#: The XML declaration (``<?xml version="1.0" encoding="utf-8"?>``) and Office's ``mso-application`` instruction:
#: the only processing instructions written (D34: any other one, ``<?xml-stylesheet href="Garcia_Ana.xsl"?>``, can
#: name a person).
_XML_DECL_RE = re.compile(
    r"xml\s+version\s*=\s*[\"']1\.[01][\"'](?:\s+encoding\s*=\s*[\"'][A-Za-z0-9._-]{1,40}[\"'])?"
    r"(?:\s+standalone\s*=\s*[\"'](?:yes|no)[\"'])?\s*\??"
    r"|mso-application\s+progid\s*=\s*[\"'][A-Za-z.]{1,40}[\"']\s*\??")
#: A document type declaration with a W3C public identifier (``DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0
#: Transitional//EN"``, with the W3C DTD address): kept as written; any other one is written as ``DOCTYPE html``.
_DOCTYPE_RE = re.compile(
    r"doctype\s+html(?:\s+public\s+\"-//W3C//DTD [A-Za-z0-9 .]{1,40}//EN\""
    r"(?:\s+\"https?://www\.w3\.org/[A-Za-z0-9/._-]{1,80}\")?)?\s*", re.I)


def safe_doctype(decl: str) -> str:
    """A document type declaration as written when it is the bare HTML one or a W3C public identifier, else
    ``DOCTYPE html`` (D34: a public identifier or system address is free text that can name a person)."""
    if _DOCTYPE_RE.fullmatch(decl):
        return decl
    return decl[:7] + " html" if decl[:7].lower() == "doctype" else "DOCTYPE html"


_CSS_URL_RE = re.compile(r"url\(([^)]*)\)", re.I)
_CSS_NAME_RE = re.compile(r"[.#]-?[A-Za-z_][\w-]*")


def css_dropped_texts(css: str) -> list[str]:
    """The free text of CSS that :func:`clean_css` drops or never trusts: comments, quoted strings with letters
    (page header/footer text, ``content``, font and style names), ``url(...)`` locations, and the selectors (and
    their class and id names) of rules :func:`safe_selector` drops."""
    out = [m.group(0).strip("/*").strip() for m in _CSS_COMMENT_RE.finditer(css)]
    out += [q[1:-1] for q in _CSS_QUOTED.findall(css) if any(ch.isalpha() for ch in q)]
    out += [m.group(1).strip(" '\"") for m in _CSS_URL_RE.finditer(css)]
    for m in _CSS_RULE_RE.finditer(_CSS_COMMENT_RE.sub("", css)):
        if not safe_selector(m.group(1)):
            out.append(m.group(1).replace("<!--", "").replace("-->", "").strip())
            out.extend(t[1:] for t in _CSS_NAME_RE.findall(m.group(1)))
    return [t for t in out if t]


def dropped_texts(events: list[tuple[Any, ...]]) -> list[str]:
    """R8-04 audit: the source text of an HTML-disguised .xls that the writer never writes: comments, script text,
    processing instructions, the values of attributes outside the formatting allow-list (``title``, ``alt``,
    ``href``, ``id``, ``content`` of a ``<meta>`` ...), the class names :func:`safe_class` drops and the free text of
    CSS (:func:`css_dropped_texts`: style blocks and ``style`` attributes). Real text no output holds (PRIVATE): the
    run's real-text registry (``privacy.texts``) keeps generators from writing it."""
    out: list[str] = []
    in_script = in_style = False
    for ev in events:
        kind = ev[0] if isinstance(ev, (tuple, list)) and ev else None
        if kind in ("start", "startend") and len(ev) > 2:
            in_script = in_script or (kind == "start" and ev[1] == "script")
            in_style = in_style or (kind == "start" and ev[1] == "style")
            for k, v in ev[2] or ():
                k = str(k).lower()
                if not v or k == "x:num":  # a cell's number again (its text is the cell's value)
                    continue
                if k not in _ALLOWED_ATTRS or k == "x:str":
                    out.append(v)
                elif k == "class":
                    kept = set(safe_class(v).split())
                    out.extend(c for c in v.split() if c not in kept)
                elif k == "style":
                    out.extend(css_dropped_texts(v))
        elif kind == "end":
            in_script = in_script and ev[1] != "script"
            in_style = in_style and ev[1] != "style"
        elif kind in ("comment", "pi") or (kind == "data" and in_script):
            if isinstance(ev[1], str) and ev[1].strip():
                out.append(ev[1])
        elif kind == "data" and in_style and isinstance(ev[1], str):
            out.extend(css_dropped_texts(ev[1]))
    return list(dict.fromkeys(out))


def _emit_row(events: list[tuple[Any, ...]], row: _Row, out: list[str], texts: list[str | None]) -> None:
    """Re-emit a source row with its own structure and the given cell texts (None = keep cell empty)."""
    out.append(_start("tr", row.attrs))
    first, mid, last = _inner_sep(events, row)
    out.append(first)
    for j, cell in enumerate(row.cells):
        if j:
            out.append(mid)
        out.append(_start(cell.tag, cell.attrs))
        txt = texts[j] if j < len(texts) else None
        if txt:
            out.append(_esc(txt))
        if cell.closed:
            out.append(f"</{cell.tag}>")
    out.append(last)
    if row.closed:
        out.append("</tr>")


def _emit_table(events: list[tuple[Any, ...]], tb: _Table, ot: Any, out: list[str], ph: Placeholders,
                rng_for: Any, t_i: int) -> dict:
    tl = ot.layout or {}
    h = int(tl.get("header_row") or 1)
    data_end = int(tl.get("data_end") or h)
    rows = tb.rows
    data_rows = rows[h:data_end]
    sep = _common_sep(events, data_rows) if len(data_rows) > 1 else _common_sep(events, rows)
    row_tpl = Counter(_start("tr", r.attrs) + ("/" if r.closed else "") for r in data_rows).most_common(1)
    src_index = list(ot.source_index)
    # each column's cell templates (class / style: Excel's number format) by exact quota per content class
    templates: list[tuple[dict[str | None, tuple[str, list, bool]], list[str | None] | str | None]] = []
    for k, j in enumerate(src_index):
        ct = column_templates(data_rows, j)
        keep: dict[str | None, tuple[str, list, bool]] = {None: ("td", [], True), **ct.keep}
        if len(ct.overall) <= 1:
            templates.append((keep, next(iter(ct.overall), None)))  # one template for the whole column
        else:
            templates.append((keep, plan_templates(ct, ot.columns[k], rng_for(f"html-templates:{t_i}:{k}"))))
    tpl_row = data_rows[0] if data_rows else rows[min(h, len(rows)) - 1]
    first_ws, mid_ws, last_ws = _inner_sep(events, tpl_row)
    tr_attrs = next((r.attrs for r in data_rows if _start("tr", r.attrs) + ("/" if r.closed else "")
                     == row_tpl[0][0]), []) if row_tpl else []
    tr_closed = row_tpl[0][0].endswith("/") if row_tpl else True
    row_starts = {r.start: k for k, r in enumerate(rows, 1)}
    i = tb.start
    emitted_data = False
    in_script = in_style = False  # script text is dropped, style text cleaned (as outside the tables)
    while i < tb.end:
        ev = events[i]
        k = row_starts.get(i)
        if k is not None:
            row = rows[k - 1]
            if k < h:
                _emit_row(events, row, out, [ph.text(c.text) if c.text else None for c in row.cells])
            elif k == h:
                _emit_header(events, row, out, ot.header, src_index)
            elif k <= data_end:
                if not emitted_data:
                    _emit_data(out, ot, templates, tr_attrs, tr_closed, sep, first_ws, mid_ws, last_ws)
                    emitted_data = True
                # skip the source data row and the whitespace after it
                i = row.end
                while (k < data_end and i < rows[k].start and events[i][0] == "data"
                       and not events[i][1].strip()):
                    i += 1
                continue
            else:
                texts = []
                for c in row.cells:
                    if not c.text:
                        texts.append(None)
                    elif _KEEP_LABEL_RE.match(c.text):
                        texts.append(c.text)
                    elif _NUMERIC_RE.fullmatch(c.text):
                        texts.append(None)  # static totals are left blank
                    else:
                        texts.append(ph.text(c.text))
                _emit_row(events, row, out, texts)
            i = row.end
            continue
        kind = ev[0]
        if kind in ("start", "startend"):
            if ev[1] == "script":
                in_script = kind == "start"
            else:
                in_style = in_style or (ev[1] == "style" and kind == "start")
                out.append(_start(ev[1], ev[2], selfclose=kind == "startend"))
        elif kind == "end":
            if ev[1] == "script":
                in_script = False
            elif ev[1] not in _VOID:
                in_style = in_style and ev[1] != "style"
                out.append(f"</{ev[1]}>")
        elif kind == "data" and not in_script:
            if in_style:
                out.append(clean_css(ev[1]))
            else:
                out.append(ev[1] if not ev[1].strip() else _esc(ph.text(ev[1])))
        i += 1
    if not emitted_data and ot.n_rows:
        _emit_data(out, ot, templates, tr_attrs, tr_closed, sep, first_ws, mid_ws, last_ws)
    if tl.get("notes_only"):
        return {"notes_only": True}
    return {"header_row": h, "data_end": h + ot.n_rows, "first_col": 1, "last_col": len(src_index)}


def _emit_header(events: list[tuple[Any, ...]], row: _Row, out: list[str], header: list[str],
                 src_index: list[int]) -> None:
    out.append(_start("tr", row.attrs))
    first, mid, last = _inner_sep(events, row)
    out.append(first)
    for k, j in enumerate(src_index):
        cell = row.cells[j] if j < len(row.cells) else _Cell("td", [], "", True)
        if k:
            out.append(mid)
        out.append(_start(cell.tag, cell.attrs))
        if header[k]:
            out.append(_esc(header[k]))
        if cell.closed:
            out.append(f"</{cell.tag}>")
    out.append(last)
    if row.closed:
        out.append("</tr>")


def _emit_data(out: list[str], ot: Any, templates: list[tuple[dict, Any]], tr_attrs: list,
               tr_closed: bool, sep: str, first_ws: str, mid_ws: str, last_ws: str) -> None:
    # per column: {template key: (start tag, end tag)} and the key of every row (or one key for all rows)
    starts = [{key: (_start(tag, attrs), f"</{tag}>" if closed else "") for key, (tag, attrs, closed) in keep.items()}
              for keep, _ in templates]
    plans = [keys for _, keys in templates]
    tr = _start("tr", tr_attrs)
    tr_end = "</tr>" if tr_closed else ""
    cols = ot.columns
    for i in range(ot.n_rows):
        if i:
            out.append(sep)
        parts = [tr, first_ws]
        for k, col in enumerate(cols):
            if k:
                parts.append(mid_ws)
            v = col[i]
            keys = plans[k]
            st, en = starts[k][keys[i] if isinstance(keys, list) else keys]
            parts.append(st)
            if v:
                parts.append(_esc(str(v)))
            parts.append(en)
        parts.append(last_ws)
        parts.append(tr_end)
        out.append("".join(parts))


def inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes of ``pd.read_html`` for the table (None when pandas has no HTML parser installed)."""
    import pandas as pd

    tables = (layout.extra or {}).get("html_tables") or []
    k = next((i for i, t in enumerate(tables) if t.get("name") == table), 0)
    info = tables[k] if tables else {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfs = pd.read_html(str(path), header=int(info.get("header_row") or 1) - 1,
                               encoding=layout.encoding)
    except Exception:  # noqa: BLE001 - lxml/bs4 absent or unparseable: unknown
        return None
    if k >= len(dfs):
        return None
    return [str(t) for t in dfs[k].dtypes]
