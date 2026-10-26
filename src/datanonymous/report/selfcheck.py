"""Self-check: re-read the written file and compare its layout with the source (AREA G).

The pipeline re-reads every written file with the OUTPUT format's handler and
calls :func:`compare` (design 6 "Self-check", D20). Compared:

* generic (every format): byte-order mark, encoding, compression, table count,
  raw header text and order, row count (N rows in = N rows out), and per column
  the set of written missing tokens (exactly the source's tokens);
* the format's own hook ``layout_diff(source, output, source_tables,
  output_tables)`` of the OUTPUT handler (for CSV: delimiter, quote character,
  line ending, trailing newline, trailing blank lines, header row present; for
  Excel/Arrow/Stata/JSON the per-table layouts of their areas);
* text formats (CSV/TSV/TXT): header-cell quoting and each column's quoting
  policy (where the output has a value the policy applies to).

Deliberate normalisations are not differences: a source with mixed line
endings is written with its majority line ending and interior blank lines are
dropped (both are warned about when the source is read).

The output is compared with what the written table can show, never with facts it
cannot carry (R6-06). For text formats the source dialect is first narrowed to
the written table (:func:`~datanonymous.io.csvio.written_dialect`: a one-column
output has no trailing blank lines, ends with a line break when all its values
are blank and quotes a blank header; the pipeline re-reads it with the source's
delimiter, which it cannot show). A column's quoting policy differs only when the
two policies write its values differently: a column whose values all need quotes
cannot show whether the source quoted every value.

Any difference is a layout mismatch: the file FAILs unless ``--no-strict``
(then warnings). The pandas dtypes inferred from source and output are compared
separately by :func:`dtype_messages` (warnings, including an output that pandas
cannot read with default settings while it reads the source).

:func:`compare` also records per-table facts for the report's verdict in the
table's private extras (``TableProfile.private.extra['selfcheck']``: output row
count, output column count, whether the output header equals the source
header). They are layout facts, never data values.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..detect.text import NA_MAP_KEY
from ..io import csvio
from ..model import ColumnQuoting, FileLayout, RawTable, SyntheticTable, TextDialect
from ..profile.codes import codes_of
from ..profile.stats import is_missing

#: What the self-check compares for CSV/TSV/TXT output (the report lists it in its verdict).
CSV_ASPECTS = (
    "encoding", "byte-order mark", "compression", "delimiter", "quote character", "line ending",
    "trailing newline", "trailing blank lines", "header row present", "header text and order",
    "header quoting", "per-column quoting", "missing-value tokens per column", "row count",
)
#: Generic aspects compared for every format (plus the format's ``layout_diff`` hook).
GENERIC_ASPECTS = (
    "encoding", "byte-order mark", "compression", "table count", "header text and order", "row count",
    "missing-value tokens per column",
)
FACTS_KEY = "selfcheck"


def coverage(fmt: str | None) -> list[str]:
    """Aspects the self-check compares for an output format (report text)."""
    if fmt == "csv":
        return list(CSV_ASPECTS)
    return list(GENERIC_ASPECTS) + [f"{fmt} layout (format hook)" if fmt else "format layout (format hook)"]


def compare(src: FileLayout, out: FileLayout, synth_tables: list[SyntheticTable], out_tables: list[RawTable],
            layout_diff: Any) -> list[str]:
    """Layout differences between the source and the re-read output (empty = identical)."""
    issues: list[str] = []
    if src.text is not None and len(synth_tables) == 1:  # what the written table can carry (R6-06)
        st0 = synth_tables[0]
        text = csvio.written_dialect(src.text, list(st0.header), [c.values for c in st0.columns],
                                     [c.index for c in st0.columns])
        if text is not src.text:
            src = dataclasses.replace(src, text=text)
    if src.bom != out.bom:
        issues.append(f"byte-order mark: source {src.bom.hex() or 'none'}, output {out.bom.hex() or 'none'}")
    if (src.encoding or "") != (out.encoding or ""):
        issues.append(f"encoding: source {src.encoding}, output {out.encoding}")
    if src.compression != out.compression:
        issues.append(f"compression: source {src.compression}, output {out.compression}")
    issues.extend(layout_diff(src, out, [st.profile.layout for st in synth_tables], out_tables))
    for st, ot in zip(synth_tables, out_tables):
        _record_facts(st, ot)
    if len(synth_tables) != len(out_tables):
        issues.append(f"table count: expected {len(synth_tables)}, output {len(out_tables)}")
        return issues
    for st, ot in zip(synth_tables, out_tables):
        where = f"table {st.name!r}: " if st.name else ""
        if st.header != ot.header:
            issues.append(f"{where}header differs from the source header")
        if st.n_rows != ot.n_rows:
            issues.append(f"{where}row count: expected {st.n_rows}, output {ot.n_rows}")
        if src.text is not None and out.text is not None:
            idx = [c.index for c in st.columns]
            s_hq = [src.text.header_quoted[i] if i < len(src.text.header_quoted) else False for i in idx]
            if s_hq != list(out.text.header_quoted[: len(idx)]):
                issues.append(f"{where}header quoting differs")
            for pos, i in enumerate(idx):
                if i < len(src.text.column_quoting) and pos < len(out.text.column_quoting):
                    a, b = src.text.column_quoting[i], out.text.column_quoting[pos]
                    if (a.policy, a.quote_empty) != (b.policy, b.quote_empty) and _quoting_shows(st, pos, a, b,
                                                                                                 src.text):
                        issues.append(f"{where}column {st.header[pos]!r}: quoting {a.policy} -> {b.policy}")
        if st.header == ot.header and st.n_rows == ot.n_rows:
            prof = st.profile
            for pos, sc in enumerate(st.columns):
                p = prof.columns[sc.index]
                if p.role == "null" or p.model is None:
                    continue
                out_missing = [v for v in codes_of(ot.columns[pos].values)[1]
                               if is_missing(v, p.missing_policy, p.na_tokens)]
                na_map = p.model.extra.get(NA_MAP_KEY) or {}  # R8-05: a pseudonymised NA token is written as such
                src_tokens = {("" if k is None else _na_written(k, na_map)) for k in p.model.missing_counts}
                out_tokens = {("" if k is None else k) for k in out_missing}
                if src_tokens != out_tokens:
                    issues.append(f"{where}column {p.name!r}: missing tokens differ "
                                  f"({len(src_tokens)} in source, {len(out_tokens)} in output)")
    return issues


def _na_written(token: str, na_map: dict[str, str]) -> str:
    """The NA token as written: an --na-value token the D35 allowlist did not keep is written as its pseudonym (R8-05),
    the whitespace around it kept."""
    fake = na_map.get(token.strip())
    if fake is None:
        return token
    return token[: len(token) - len(token.lstrip())] + fake + token[len(token.rstrip()):]


def _record_facts(st: SyntheticTable, ot: RawTable) -> None:
    """Store output row/column counts and header identity for the report verdict (no data values)."""
    tp = st.profile
    try:
        tp.private.extra[FACTS_KEY] = {
            "rows_out": int(ot.n_rows),
            "columns_out": len(ot.header),
            "header_matches_written": list(st.header) == list(ot.header),
            "header_matches_source": list(tp.header) == list(ot.header),
        }
    except AttributeError:  # pragma: no cover - defensive: a profile without private extras
        pass


def facts(tp: Any) -> dict[str, Any] | None:
    """Self-check facts recorded for a table profile (None when the self-check did not run)."""
    try:
        return tp.private.extra.get(FACTS_KEY)
    except AttributeError:
        return None


def _quoting_shows(st: SyntheticTable, pos: int, a: ColumnQuoting, b: ColumnQuoting, t: TextDialect) -> bool:
    """The source's quoting ``a`` and the re-read quoting ``b`` write this column's values differently.

    Only then can the output show the difference: a policy that applies to no value of the column (no text
    for ``nonnumeric``, no blank for quoted empties), or a column whose values all need quotes anyway (one
    column left of a file that quoted everything), reads back as another policy that writes the same bytes.
    """
    vals = st.columns[pos].values
    b = ColumnQuoting(b.policy, b.quote_empty, a.bare_quotes)
    return csvio.column_text(vals, a, t) != csvio.column_text(vals, b, t)


def dtype_drift(src: list[str] | None, out: list[str] | None, header: list[str]) -> list[tuple[str | None, str]]:
    """pandas dtype drift between source and output as (column, text), the column None when the drift is the whole
    file's (warnings, never a FAIL)."""
    if src is not None and out is None:
        return [(None, "pandas reads the source with default settings but cannot read the output")]
    if src is None or out is None or len(src) != len(out):
        return []
    return [(h, f"column {h!r}: pandas reads {a} from the source but {b} from the output")
            for h, a, b in zip(header, src, out) if a != b]


def dtype_messages(src: list[str] | None, out: list[str] | None, header: list[str]) -> list[str]:
    """pandas dtype drift between source and output (warnings, never a FAIL)."""
    return [text for _, text in dtype_drift(src, out, header)]
