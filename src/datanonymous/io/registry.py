"""Format registry: format key -> reader/writer handler (FOUNDATION).

Every v1 format of design D17 is registered here once, pointing at functions in
its area-owned module. Area agents implement those functions in their own
module and never edit this file:

========== ============================ ======
key        module                       area
========== ============================ ======
csv        io/csvio.py                  FOUNDATION (CSV/TSV/TXT, disguised text .xls)
xlsx       io/excelio.py                E
xls        io/excelio.py (out .xlsx)    E
html_xls   io/excelio.py                E (HTML table saved as .xls)
parquet    io/arrowio.py                F
feather    io/arrowio.py                F
dta        io/statio.py                 F
sas        io/statio.py (out .parquet)  F (read-only; written as Parquet)
json       io/jsonio.py                 F
jsonl      io/jsonio.py                 F
========== ============================ ======

Compression (``.gz/.zip/.bz2/.xz``) is applied by the format's reader/writer
through ``io/compress.py`` (AREA F); ``compress.COMPRESSIBLE_FORMATS`` says which
inner formats may be compressed.

``out_format``: formats written in another format (``xls`` -> ``xlsx``, ``sas``
-> ``parquet``). The self-check re-reads the output with the *output* format's
handler and uses its ``layout_diff`` and ``inferred_dtypes``.

Handler contract
----------------
``read(ReadRequest) -> ReadResult``
    Parse the file into :class:`~datanonymous.model.FileLayout` + raw tables.
    Raise :class:`~datanonymous.errors.ReadError` (bad file) or
    :class:`~datanonymous.errors.NotImplementedFormat` (stub).
``write(WriteRequest) -> None``
    Write ``tables`` to ``request.path`` mirroring ``request.layout``.
``layout_diff(source, output, source_tables, output_tables) -> list[str]``
    Format-specific layout differences for the self-check (empty = identical).
    ``source``/``output`` are the FileLayouts; ``source_tables`` is the list of
    source table layouts (``RawTable.layout`` dicts, as kept in
    ``TableProfile.layout``) and ``output_tables`` the re-read output
    :class:`~datanonymous.model.RawTable` objects (sheet order/visibility, header
    rows, cell types, number formats ...).
``detect_native(ctx: DetectContext) -> Detection | None``
    Type decided by native typed storage (Parquet/Excel/Stata), consulted after
    the explicit overrides and before the value rules (design 3). None = no opinion.
``inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None``
    dtypes pandas infers with default reading of one table (self-check warning,
    schema.json); None when unknown.

``WriteRequest`` carries the run ``seed`` and ``rng_for(purpose)`` (a seeded
Generator per purpose for this file), so a writer that must invent content
(Excel title placeholders, section 10) stays reproducible (D26).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import Settings
from ..errors import NotImplementedFormat
from ..model import FileLayout, ReadResult, TableProfile
from . import arrowio, csvio, excelio, jsonio, statio


@dataclass
class ReadRequest:
    """What a reader gets. ``purpose`` is 'source' (pass 1) or 'selfcheck' (re-reading
    our own output: content checks on the source, such as header plausibility,
    are skipped)."""

    path: Path
    rel_path: str
    format: str
    compression: str | None
    settings: Settings
    purpose: str = "source"


@dataclass
class OutTable:
    """A synthetic table ready to write.

    ``columns`` holds one rendered object array per *output* column (``None`` =
    native null); ``source_index`` maps each output column to its source column
    (columns removed with ``--exclude`` are absent).
    """

    name: str | None
    header: list[str]
    columns: list[np.ndarray]
    source_index: list[int]
    n_rows: int
    layout: dict[str, Any] = field(default_factory=dict)
    profile: TableProfile | None = None


@dataclass
class WriteRequest:
    """What a writer gets. ``rng_for(purpose)`` returns a Generator seeded from
    (seed, file, purpose); ``tables[k].profile.columns[i].native`` holds each
    source column's native details (cell types, Arrow types, labels)."""

    path: Path
    layout: FileLayout
    tables: list[OutTable]
    settings: Settings
    seed: int = 0
    rng_for: Callable[[str], np.random.Generator] | None = None


def _no_native(ctx: Any) -> None:
    return None


def _no_layout_diff(source: FileLayout, output: FileLayout, source_tables: list[dict[str, Any]],
                    output_tables: list[Any]) -> list[str]:
    return []


def _no_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    return None


@dataclass(frozen=True)
class FormatHandler:
    key: str
    description: str
    area: str
    read: Callable[[ReadRequest], ReadResult]
    write: Callable[[WriteRequest], None]
    layout_diff: Callable[[FileLayout, FileLayout, list[dict[str, Any]], list[Any]], list[str]] = _no_layout_diff
    detect_native: Callable[[Any], Any] = _no_native
    inferred_dtypes: Callable[[Path, FileLayout, str | None], list[str] | None] = _no_dtypes
    out_format: str | None = None  # key of the format the output is written in (None = same)


FORMATS: dict[str, FormatHandler] = {
    "csv": FormatHandler("csv", "delimited text (CSV/TSV/TXT)", "FOUNDATION", csvio.read, csvio.write,
                         csvio.layout_diff, _no_native, csvio.inferred_dtypes),
    "xlsx": FormatHandler("xlsx", "Excel workbook (.xlsx/.xlsm)", "E", excelio.read_xlsx, excelio.write_xlsx,
                          excelio.layout_diff, excelio.detect_native, excelio.inferred_dtypes),
    "xls": FormatHandler("xls", "Excel 97-2003 workbook (.xls, written as .xlsx)", "E", excelio.read_xls,
                         excelio.write_xlsx, excelio.layout_diff, excelio.detect_native, excelio.inferred_dtypes,
                         out_format="xlsx"),
    "html_xls": FormatHandler("html_xls", "HTML table saved as .xls", "E", excelio.read_html_table,
                              excelio.write_html_table, excelio.layout_diff, _no_native,
                              excelio.inferred_dtypes_html),
    "parquet": FormatHandler("parquet", "Apache Parquet", "F", arrowio.read_parquet, arrowio.write_parquet,
                             arrowio.layout_diff, arrowio.detect_native, arrowio.inferred_dtypes),
    "feather": FormatHandler("feather", "Feather / Arrow IPC", "F", arrowio.read_feather, arrowio.write_feather,
                             arrowio.layout_diff, arrowio.detect_native, arrowio.inferred_dtypes),
    "dta": FormatHandler("dta", "Stata .dta", "F", statio.read_dta, statio.write_dta, statio.layout_diff,
                         statio.detect_native, statio.inferred_dtypes),
    "sas": FormatHandler("sas", "SAS .sas7bdat/.xpt (read-only; written as Parquet)", "F", statio.read_sas,
                         arrowio.write_parquet, statio.layout_diff, statio.detect_native, statio.inferred_dtypes,
                         out_format="parquet"),
    "json": FormatHandler("json", "flat JSON records", "F", jsonio.read_json, jsonio.write_json,
                          jsonio.layout_diff, jsonio.detect_native, jsonio.inferred_dtypes),
    "jsonl": FormatHandler("jsonl", "JSON lines", "F", jsonio.read_jsonl, jsonio.write_jsonl,
                           jsonio.layout_diff, jsonio.detect_native, jsonio.inferred_dtypes),
}

#: File extension (lower case) -> format key. ``.xls`` is re-classified by magic bytes.
EXTENSION_FORMATS: dict[str, str] = {
    ".csv": "csv", ".tsv": "csv", ".txt": "csv",
    ".xlsx": "xlsx", ".xlsm": "xlsx", ".xls": "xls",
    ".parquet": "parquet", ".feather": "feather",
    ".dta": "dta", ".sas7bdat": "sas", ".xpt": "sas",
    ".json": "json", ".jsonl": "jsonl",
}

#: Output suffix changes (design D17 and section 4); every other file keeps its name.
OUTPUT_SUFFIX: dict[str, str] = {".xlsm": ".xlsx", ".sas7bdat": ".parquet", ".xpt": ".parquet"}

#: Tabular formats that v1 does not support: skipped with conversion advice.
UNSUPPORTED_ADVICE: dict[str, str] = {
    ".xlsb": "Excel binary workbook: save it as .xlsx or .csv",
    ".ods": "OpenDocument spreadsheet: save it as .xlsx or .csv",
    ".sav": "SPSS file: export to .csv (or .dta)",
    ".rds": "R data file: write.csv() it (or arrow::write_parquet())",
    ".rdata": "R data file: write.csv() each data frame",
    ".rda": "R data file: write.csv() each data frame",
}

COMPRESSION_SUFFIXES: dict[str, str] = {".gz": "gz", ".zip": "zip", ".bz2": "bz2", ".xz": "xz"}

#: File extensions of other files (documents, images, media, archives, code and data files): with the tabular
#: formats above, the suffixes that are an extension at all. Any other suffix (``Ana.Garcia``,
#: ``notes.Okonkwo``) is part of the file's name (D34 screens it with the name; a skip reason never quotes it).
#: One literal collection of more than 50 entries (data).
OTHER_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".doc", ".docx", ".docm", ".dot", ".dotx", ".rtf", ".odt", ".ods", ".odp", ".txt", ".md", ".pages",
    ".numbers", ".key", ".ppt", ".pptx", ".pps", ".ppsx", ".xls", ".xlsx", ".xlsm", ".xlsb", ".xlt", ".xltx",
    ".csv", ".tsv", ".tab", ".dat", ".json", ".jsonl", ".ndjson", ".xml", ".html", ".htm", ".mht", ".mhtml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log", ".sql", ".db", ".sqlite", ".mdb", ".accdb",
    ".sav", ".zsav", ".por", ".dta", ".sas7bdat", ".sas7bcat", ".xpt", ".rds", ".rdata", ".rda", ".parquet",
    ".feather", ".arrow", ".avro", ".orc", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".mat", ".py",
    ".r", ".rmd", ".qmd", ".ipynb", ".do", ".sas", ".sps", ".js", ".css", ".png", ".jpg", ".jpeg", ".gif",
    ".bmp", ".tif", ".tiff", ".svg", ".heic", ".webp", ".ico", ".mp3", ".mp4", ".m4a", ".mov", ".avi",
    ".wav", ".wmv", ".zip", ".gz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".tar", ".tgz", ".msg", ".eml",
    ".pst", ".ics", ".vcf", ".lnk", ".url", ".bak", ".tmp", ".old", ".orig", ".exe", ".msi", ".dll", ".bat",
    ".cmd", ".ps1", ".sh", ".one", ".vsdx", ".pub", ".epub", ".mobi", ".odg", ".wpd", ".wps", ".gdoc",
    ".gsheet", ".gslides", ".xps", ".oxps", ".dbf", ".shp", ".kml", ".kmz", ".geojson", ".gpx", ".spv",
})


def known_extension(suffix: str) -> bool:
    """``suffix`` (``.csv``, ``.PDF``) is a file extension: a format of :data:`EXTENSION_FORMATS`,
    :data:`UNSUPPORTED_ADVICE`, :data:`COMPRESSION_SUFFIXES` or :data:`OTHER_EXTENSIONS`, or 1-3 digits
    (``.001``)."""
    low = suffix.lower()
    return (low in EXTENSION_FORMATS or low in UNSUPPORTED_ADVICE or low in COMPRESSION_SUFFIXES
            or low in OTHER_EXTENSIONS or (low[1:].isdigit() and 1 <= len(low) - 1 <= 3))


def get(key: str) -> FormatHandler:
    try:
        return FORMATS[key]
    except KeyError:
        raise NotImplementedFormat(f"no handler for format {key!r}") from None


def output_handler(key: str) -> FormatHandler:
    """Handler of the format the output of ``key`` is written in (re-read, layout_diff, dtypes)."""
    h = get(key)
    return get(h.out_format) if h.out_format else h
