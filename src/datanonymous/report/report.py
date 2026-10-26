"""Report directory writer and console output (AREA G).

Report dir (design D21), aggregates only:

* ``REPORT.md`` + ``report.json`` (``schema_version`` 1.0, stable keys): an AI
  agent verifying many files confirms the whole run in one read. REPORT.md
  starts with a machine-checkable **Verdict** section: the line
  ``Overall verdict: PASS|WARN|FAIL``, then one row per criterion (stable keys:
  ``written``, ``layout_selfcheck``, ``rows``, ``header``, ``dtypes``,
  ``fidelity``, ``leak_scan``, ``chance_matches``, ``warnings``,
  ``adjusted_flags``, plus the run-level ``run_errors`` and ``run_warnings``)
  with counts of PASS/WARN/FAIL/N/A files, and a per-file table of those
  results. Then the run facts, the per-file table, skipped files with reasons,
  failures with their exact fixes, run errors, warnings and notes (adjusted
  flags explained, extreme-value exposure). ``report.json`` mirrors it
  (``verdict.overall``, ``verdict.criteria``, ``verdict.files``).
* ``manifest.json``: seed, tool/Python/library versions (incl. Faker), settings,
  the source -> output file map and a timestamp (the only non-reproducible field).
* ``files/<relpath>.md`` + ``.json`` per file: status with reasons and fixes, the
  per-file checks, layout, per-column real vs synthetic statistics with flags,
  each column's fidelity verdict with its reason and thresholds, IDs, constants
  and structure, privacy (leak scan, chance matches, extreme exposure) and the
  self-check result.

Result values: ``PASS``, ``WARN``, ``FAIL``, ``N/A`` (the check did not run,
e.g. the file FAILed earlier) and ``INFO`` (informational: adjusted flags).
Overall: FAIL when a file FAILed or a run-level error exists (exit code 3);
WARN when everything was written but something needs review; else PASS. The
leak scan runs after the report is built: when it finds anything the pipeline
rebuilds the report (so a written report that says ``leak_scan: PASS`` was
scanned clean).

Fixes render as a command in code plus a plain explanation
(``Fix: `--force-text Course` replaces the labels ...``); report.json adds the
shell-ready ``fix_commands``. Notes name concrete commands, never a
placeholder (``--no-exact-extremes data.csv::Expense``). The warnings of a
file include the pandas dtype drift of a table with ``--exclude`` columns
(:func:`excluded_dtype_messages`; the pipeline's positional comparison cannot
pair those), so the ``dtypes`` and ``warnings`` criteria agree.

Only public aggregates are written (``ColumnProfile.stats`` and the synthetic
stats): never a real ID/name/email/phone/address/free-text value; real labels
only for label-safe types. When the leak scan found something, its private
redactor (``privacy.leaks.LeakMessage``) redacts the rebuilt report and the
console text. Console text is ASCII only (cp1252 consoles).

D34 (``privacy.headers``): a header, sheet, file or folder name that holds
personal data is never printed. A file that FAILs for one shows its reasons
only (``FileResult.withheld``: no column table, sheet names, layout or column
messages, :func:`shown_tables`), its paths are shown redacted, and the run's
``privacy.headers.NameScreenMessage`` redacts every remaining occurrence in the
report, the console summary, ``inspect`` (text and ``--json``) and ``-v``
tracebacks (:func:`redact_console`).

:func:`build_report` returns every file as bytes without writing, so the AREA G
leak scan can check the report before :func:`write_report_files` writes it.
"""

from __future__ import annotations

import csv
from collections import Counter
import datetime as _dt
import functools
import importlib.metadata as md
import io
import json
import platform
import re
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from .. import __version__
from .. import messages as catalogue
from ..config import Settings
from ..detect.text import na_pseudonymized, pseudonymized
from ..generate import numeric as numeric_gen
from ..generate.table import is_level_generated
from ..model import FileResult, Message, RunResult, Status, TableProfile
from ..paths import replace_bytes
from ..privacy.sharing_policy import names_refused_option
from ..privacy.leaks import KIND_TEXT, LEAK_PREFIX, UNVERIFIED_TEXT, sensitive_columns
from ..profile import structure as structure_mod
from . import selfcheck

LIBRARIES = ("pandas", "numpy", "pyarrow", "openpyxl", "xlrd", "click", "PyYAML", "Faker")
SCHEMA_VERSION = "1.0"

PASS, WARN, FAIL, NA, INFO = "PASS", "WARN", "FAIL", "N/A", "INFO"
_RANK = {FAIL: 4, WARN: 3, PASS: 2, INFO: 1, NA: 0}
OVERALL_RULE = ("FAIL = at least one file FAILed (not written) or a run-level error exists (exit code 3); "
                "WARN = every file was written, with warnings to review; PASS = every file written and every "
                "check clean")

#: (key, title, what it checks) of every per-file criterion, in report order.
CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("written", "Synthetic file written",
     "the file was generated, passed every gate and is in the out dir (a FAILed file is not written)"),
    ("layout_selfcheck", "Layout self-check",
     "the written file was re-read with the output format's reader and its layout compared with the source "
     "(aspects listed below); a difference FAILs the file unless --no-strict"),
    ("rows", "N rows in = N rows out", "the re-read output has exactly the source's row count in every table"),
    ("header", "Header identical",
     "the re-read output has the source's column names (raw text) in the same order"),
    ("dtypes", "pandas dtypes",
     "pandas infers the same dtype per column from the source and from the output (a difference is a warning)"),
    ("fidelity", "Per-column fidelity",
     "every column's synthetic distribution is within the thresholds: level shares and missing rate within "
     "max(1 pp, 2 SE), KS <= 0.05; within-ID constants judged per entity; adjusted aspects never drift"),
    ("leak_scan", "Leak scan",
     "no real ID, name, email, phone, SSN, address, free-text or high-cardinality text value in the output, "
     "the out-dir docs or the report"),
    ("chance_matches", "Chance row matches",
     "no synthetic row (or ID's entity vector) equals a real one on all compared columns after swap-redraw"),
    ("warnings", "Warnings to review", "the file has no warnings (each is listed below with its fix)"),
    ("adjusted_flags", "Adjusted flags",
     "informational: rare_category_adjusted = every source level / NA token kept at least once, which changed "
     "a count; extreme_value_adjusted = the exact source min and max were emitted. Adjusted aspects never "
     "raise fidelity errors"),
)
RUN_CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("run_errors", "Run-level errors", "no run-level error (e.g. a leak in an out-dir file not written by this run)"),
    ("run_warnings", "Run-level warnings", "no run-level warning (unused column flags, stale outputs, ...)"),
)


def ascii_text(s: Any) -> str:
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def versions() -> dict[str, str]:
    out = {"datanonymous": __version__, "python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:  # pragma: no cover
            out[lib] = "not installed"
    return out


def _posix(p: Path | None) -> str | None:
    return None if p is None else Path(p).as_posix()


def finite_json(obj: Any) -> Any:
    """``obj`` with every non-finite float (inf, NaN) as None: report JSON stays strict JSON (R1-16)."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    return obj


def _json(obj: Any) -> str:
    return json.dumps(finite_json(obj), indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n"


def file_report_rel(rel_path: str) -> str:
    return str(PurePosixPath("files") / (rel_path + ".md"))


def link_target(rel: str) -> str:
    """A relative path as a Markdown link target (spaces, ``#``, ``?``, ``(`` ... percent-encoded)."""
    return urllib.parse.quote(rel, safe="/")


def _messages(fr: FileResult, levels: tuple[str, ...]) -> list[Message]:
    return [m for m in file_messages(fr) if m.level in levels]


def file_messages(fr: FileResult) -> list[Message]:
    """Every message of a file, plus the dtype drift of columns the pipeline could not pair (``--exclude``).

    A file whose names hold personal data (D34, ``FileResult.withheld``) shows its FAIL reasons only: its
    reader, table and column messages name its columns and sheets."""
    if fr.withheld:
        return [m for m in fr.messages if m.level == "error"]
    return fr.all_messages() + excluded_dtype_messages(fr)


def shown_tables(fr: FileResult) -> list[TableProfile]:
    """The tables whose details the report and the console show (none for a D34-withheld file)."""
    return [] if fr.withheld else fr.tables


def _dtype_pairs(fr: FileResult, t_i: int) -> list[tuple[str, str, str]] | None:
    """(column, source dtype, output dtype) per output column, or None when the dtypes cannot be paired."""
    tp = fr.tables[t_i]
    src = fr.source_dtypes[t_i] if t_i < len(fr.source_dtypes) else None
    out = fr.out_dtypes[t_i] if t_i < len(fr.out_dtypes) else None
    if src is None or out is None:
        return None
    kept = [i for i, p in enumerate(tp.columns) if p.role != "excluded"]
    if len(src) != len(tp.columns) or len(out) != len(kept):
        return None
    return [(tp.columns[i].name, src[i], out[pos]) for pos, i in enumerate(kept)]


def excluded_dtype_messages(fr: FileResult) -> list[Message]:
    """dtype-drift warnings for tables with ``--exclude`` columns.

    The pipeline compares source and output dtypes by position
    (``selfcheck.dtype_messages``), which it cannot do once a column is dropped;
    the report pairs them through the kept columns so the drift is a listed
    warning, as for every other file.
    """
    out: list[Message] = []
    for t_i, tp in enumerate(fr.tables):
        if not any(p.role == "excluded" for p in tp.columns):
            continue
        for name, a, b in _dtype_pairs(fr, t_i) or []:
            if a != b:
                out.append(Message("warning", "dtype_mismatch",
                                   f"column {name!r}: pandas reads {a} from the source but {b} from the output"))
    return out


def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _period(s: str) -> str:
    """End a sentence with exactly one period (a text ending in . ! ? is kept)."""
    s = s.rstrip()
    return s if s.endswith((".", "!", "?")) else s + "."


def code_span(s: Any) -> str:
    """Markdown code span that survives backticks inside ``s``."""
    s = str(s)
    longest = run = 0
    for ch in s:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if s.startswith("`") or s.endswith("`") or not s.strip() else ""
    return f"{fence}{pad}{s}{pad}{fence}"


# --------------------------------------------------------------------------- fix rendering
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*|-[vq])(?![\w-])")
_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*(?:\.[A-Za-z]+)?(?:=[A-Z0-9_]+)?$")
_PROSE = frozenset({
    "a", "add", "all", "an", "and", "are", "as", "be", "but", "each", "every", "for", "if", "in", "is", "it", "its",
    "keeps", "makes", "no", "not", "of", "off", "on", "one", "only", "or", "pass", "please", "replaces", "see", "so",
    "that", "the", "then", "this", "to", "turns", "until", "use", "when", "with", "writes",
})
#: Flags whose value is not a column (a lower-case word such as ``tab`` or ``utf-8`` is their value).
_VALUE_FLAGS = frozenset({"--delimiter", "--encoding", "--sheet", "--na-value", "--decimal", "--include",
                          "--exclude-path", "--keep-name", "--config", "--seed", "--min-cell", "--factor-max",
                          "--max-row-matches", "--header-row", "--out-dir", "--report-dir", "--range", "--id-check"})
_FALLBACK_WITH_ARG = frozenset({
    "--factor-max", "--force-factor", "--force-text", "--force-type", "--pii-col", "--dayfirst", "--date-cols",
    "--date-format", "--calendar", "--na-value", "--decimal", "--id-col", "--org-code", "--person-id", "--id-check",
    "--no-exact-extremes", "--min-cell", "--null", "--exclude", "--encoding", "--delimiter", "--sheet",
    "--header-row", "--range", "--include", "--exclude-path", "--keep-name", "--seed", "--config", "--out-dir",
    "--report-dir", "--max-row-matches"})


@functools.lru_cache(maxsize=1)
def cli_flags_with_value() -> frozenset[str]:
    """CLI options that take a value (read from the click commands; a fixed list if that fails)."""
    try:
        import click

        from .. import cli

        out: set[str] = set()
        for cmd in (cli.run_cmd, cli.inspect_cmd):
            for prm in cmd.params:
                if isinstance(prm, click.Option) and not prm.is_flag and not prm.count:
                    out.update(o for o in prm.opts + prm.secondary_opts if o.startswith("-"))
        return frozenset(out) or _FALLBACK_WITH_ARG
    except Exception:  # noqa: BLE001 - rendering must never fail
        return _FALLBACK_WITH_ARG


def shell_quote(arg: str) -> str:
    """Quote a flag value for a shell when it holds spaces or shell characters (double quotes)."""
    if arg and re.fullmatch(r"[A-Za-z0-9_./:=+@%,-]+", arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`") + '"'


def _parse_arg(text: str, start: int, flag: str, cands: list[str]) -> tuple[str | None, int]:
    rest = text[start:]
    quoted = re.match(r'"([^"\s][^"]*)"', rest)  # a quoted value (--range "Roster!A3:H33")
    if quoted:
        return quoted.group(1), start + quoted.end()
    for c in cands:
        for form in (c, repr(c)):
            if not rest.startswith(form):
                continue
            after = start + len(form)
            if after < len(text) and text[after] == "=":
                m = re.match(r"=[^\s,;)]+", text[after:])
                if m:
                    return c + m.group(), after + len(m.group())
            if after == len(text) or text[after] in " ,;:).":
                return c, after
    m = re.match(r"\S+", rest)
    if not m:
        return None, start
    tok = m.group().rstrip(",;:.)")
    if not tok or tok[0] in "(/'\"-" or tok.lower() in _PROSE:
        return None, start
    if _PLACEHOLDER.match(tok) or flag in _VALUE_FLAGS or not (tok.isalpha() and tok.islower()):
        return tok, start + len(tok)
    return None, start


def parse_fix(fix: str, candidates: Iterable[str] = ()) -> list[tuple[str, str | None]]:
    """Split a fix into segments: (original text, None) for prose, (original text, shell command) for a flag.

    ``candidates`` are column names (and qualified ``file::column`` refs) that may follow a flag; a
    multi-word column name is recognised as one value and quoted in the command.
    """
    cands = sorted({c for c in candidates if c}, key=len, reverse=True)
    with_value = cli_flags_with_value()
    segs: list[tuple[str, str | None]] = []
    pos = 0
    for m in _FLAG_RE.finditer(fix):
        if m.start() < pos:
            continue
        flag = m.group(1)
        end = m.end()
        arg = None
        if flag in with_value and end < len(fix) and fix[end] in " =":
            arg, arg_end = _parse_arg(fix, end + 1, flag, cands)
            if arg is not None:
                end = arg_end
        if m.start() > pos:
            segs.append((fix[pos:m.start()], None))
        segs.append((fix[m.start():end], flag if arg is None else f"{flag} {shell_quote(arg)}"))
        pos = end
    if pos < len(fix):
        segs.append((fix[pos:], None))
    return segs


def fix_commands(fix: str | None, candidates: Iterable[str] = ()) -> list[str]:
    """Shell-ready commands (flag plus value) named in a fix."""
    if not fix:
        return []
    return [cmd for _, cmd in parse_fix(fix, candidates) if cmd]


def fix_markdown(fix: str | None, candidates: Iterable[str] = ()) -> str:
    """A fix as Markdown: each command in a code span, the explanation as plain text."""
    if not fix:
        return ""
    out = []
    for text, cmd in parse_fix(fix, candidates):
        out.append(_md(text) if cmd is None else code_span(cmd))
    return "".join(out).strip()


def fix_plain(fix: str | None, candidates: Iterable[str] = ()) -> str:
    """A fix for the console: commands with shell quoting, the rest as is."""
    if not fix:
        return ""
    return "".join(text if cmd is None else cmd for text, cmd in parse_fix(fix, candidates)).strip()


def _candidates(fr: FileResult | None) -> list[str]:
    if fr is None:
        return []
    from ..privacy.headers import quoted_names

    out: list[str] = []
    for tp in shown_tables(fr):
        for p in tp.columns:
            out.append(p.name)
            out.append(f"{fr.rel_path}::{p.name}")
            if tp.name:
                out.append(f"{fr.rel_path}::{tp.name}::{p.name}")
    # a file that FAILed while it was read has no table: the names its reasons quote are the values of its fix
    # (``--null Okonkwo, Chinedu``), parsed as one value so the D34 redactor finds them whole
    if not fr.tables:
        for r in fr.reasons:
            out.extend(quoted_names(r))
    return out


def _run_candidates(run: RunResult) -> list[str]:
    out: list[str] = []
    for fr in run.files:
        out.extend(_candidates(fr))
    return out


def message_public(m: Message, candidates: Iterable[str] = ()) -> dict[str, Any]:
    out = m.to_public()
    if m.fix:
        out["fix_commands"] = fix_commands(m.fix, candidates)
    return out


# --------------------------------------------------------------------------- plain-language findings
def plain_finding(code: str, level: str, text: str, fix: str | None, sharing: bool,
                  cands: Iterable[str] = (), markdown: bool = True) -> tuple[str, str]:
    """(text, fix) of a finding as the operator should read it; the fix is '' when there is none.

    In sharing mode a blocking or failure code that the message catalogue knows is shown as its plain sentence
    and plain fix, with the code next to the sentence in parentheses. Anything else keeps its own text, and a fix
    that names an option sharing mode refuses is never shown.
    """
    if sharing and level in ("warning", "error"):
        shown = catalogue.render(code, _md(text) if markdown else text)
        if shown is not None:
            sentence, plain_fix = shown
            if code == "fail" and fix and not names_refused_option(fix):
                plain_fix = fix_markdown(fix, cands) if markdown else fix_plain(fix, cands)
            elif not markdown and plain_fix:
                plain_fix = plain_fix.replace("`", "")
            if not markdown:
                sentence = sentence.replace("`", "")
            coded = f"{sentence.rstrip('.')} ({code_span(code) if markdown else code})."
            return coded, plain_fix or ""
    if sharing and names_refused_option(fix):
        fix = None
    if markdown:
        return _period(fix_markdown(text, cands)), fix_markdown(fix, cands)
    return text, fix_plain(fix, cands)


def failure_findings(fr: FileResult, sharing: bool, cands: Iterable[str] = (),
                     markdown: bool = True) -> list[tuple[str, str]]:
    """(text, fix) for each reason a file FAILed, with the catalogue's plain wording in sharing mode."""
    errors = {m.text: m for m in fr.messages if m.level == "error"}
    out = []
    for i, reason in enumerate(fr.reasons):
        m = errors.get(reason)
        if m is None:
            fix = fr.fixes[i] if i < len(fr.fixes) and len(fr.fixes) == len(fr.reasons) else None
            m = Message("error", "fail", reason, fix=fix)
        out.append(plain_finding(m.code, "error", reason, m.fix, sharing, cands, markdown))
    return out


# --------------------------------------------------------------------------- per-file checks (verdict)
def _worst(results: Iterable[str]) -> str:
    best = NA
    for r in results:
        if _RANK[r] > _RANK[best]:
            best = r
    return best


def _check(result: str, detail: str = "") -> dict[str, str]:
    return {"result": result, "detail": detail}


def _leak_failed(fr: FileResult) -> bool:
    return any(r.startswith(LEAK_PREFIX) for r in fr.reasons)


def file_checks(fr: FileResult) -> dict[str, dict[str, str]]:
    """Per-file criteria (see :data:`CRITERIA`): ``{key: {'result', 'detail'}}``."""
    checks: dict[str, dict[str, str]] = {}
    facts = [selfcheck.facts(tp) for tp in fr.tables]
    have_facts = bool(fr.tables) and all(f is not None for f in facts)
    if fr.written:
        checks["written"] = _check(PASS, "written to the out dir")
    elif fr.failed:
        checks["written"] = _check(FAIL, "not written: " + "; ".join(fr.reasons))
    else:  # inspect (preview): nothing is generated
        checks["written"] = _check(NA, "not generated (inspect writes nothing)")
    # layout self-check
    selfcheck_failed = any(r.startswith("self-check") for r in fr.reasons)
    out_fmt = fr.out_layout.format if fr.out_layout is not None else fr.format
    cov = ", ".join(selfcheck.coverage(out_fmt))
    if fr.selfcheck:
        res = FAIL if selfcheck_failed else WARN
        checks["layout_selfcheck"] = _check(res, "; ".join(fr.selfcheck[:5])
                                            + ("" if selfcheck_failed else " (accepted with --no-strict)"))
    elif have_facts and _guessed_delimiter(fr):
        # R7-05: the self-check re-reads the output with the source's delimiter, so it agrees with the guess by
        # construction; an ambiguous guess is flagged here instead of passing silently
        checks["layout_selfcheck"] = _check(WARN, "compared: " + cov + "; but the source delimiter was guessed "
                                            "among several that split every line consistently ("
                                            + _guessed_delimiter(fr) + "), and the output is re-read with the "
                                            "same one: pass --delimiter to confirm it")
    elif have_facts:
        checks["layout_selfcheck"] = _check(PASS, "compared: " + cov)
    else:
        checks["layout_selfcheck"] = _check(NA, "not run (the file FAILed before it was written)")
    # rows and header
    if have_facts:
        pairs = [(tp.n_rows, f["rows_out"]) for tp, f in zip(fr.tables, facts)]
        ok = all(a == b for a, b in pairs)
        checks["rows"] = _check(PASS if ok else FAIL, ", ".join(f"{a} in -> {b} out" for a, b in pairs))
        dropped = sum(1 for tp in fr.tables for p in tp.columns if p.role == "excluded")
        if all(f["header_matches_source"] for f in facts):
            checks["header"] = _check(PASS, "same column names in the same order")
        elif dropped and all(f["header_matches_written"] for f in facts):
            checks["header"] = _check(WARN, f"{dropped} column(s) dropped with --exclude; the other names and "
                                            "their order are identical")
        else:
            checks["header"] = _check(FAIL, "the output header differs from the source header")
    else:
        checks["rows"] = _check(NA, "not checked (no output)")
        checks["header"] = _check(NA, "not checked (no output)")
    checks["dtypes"] = _dtype_check(fr)
    # fidelity
    verdicts = [v.get("verdict") for fid in fr.fidelity for v in fid.values() if isinstance(v, dict)]
    if verdicts:
        counts = {k: verdicts.count(k) for k in ("ok", "adjusted", "drift", "n/a")}
        detail = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        drifted = [p.name for t_i, tp in enumerate(fr.tables) for p in tp.columns
                   if t_i < len(fr.fidelity) and (fr.fidelity[t_i].get(p.index) or {}).get("verdict") == "drift"]
        if drifted:
            detail += "; drift: " + ", ".join(drifted)
        checks["fidelity"] = _check(WARN if counts["drift"] else PASS, detail)
    else:
        checks["fidelity"] = _check(NA, "not compared (no output)")
    # leak scan
    sens = sensitive_columns([fr])
    if _leak_failed(fr):
        checks["leak_scan"] = _check(FAIL, next(r for r in fr.reasons if r.startswith(LEAK_PREFIX)))
    elif fr.written:
        checks["leak_scan"] = _check(PASS, (f"{len(sens)} ID/PII/text column(s) of this file checked; " if sens else
                                            "no ID/PII/text column of its own; ")
                                     + "the output, the docs and the report hold no real value of any sensitive "
                                       "column of the run")
    else:
        checks["leak_scan"] = _check(NA, "not run (no output)")
    checks["chance_matches"] = _match_check(fr)
    n_warn = len(_messages(fr, ("warning",)))
    checks["warnings"] = _check(WARN if n_warn else PASS, f"{n_warn} warning(s)" if n_warn else "none")
    adjusted = adjusted_columns(fr)
    if adjusted:
        checks["adjusted_flags"] = _check(INFO, "; ".join(f"{c} ({', '.join(f)})" for c, f in adjusted))
    else:
        checks["adjusted_flags"] = _check(PASS, "no adjusted column")
    return checks


def _guessed_delimiter(fr: FileResult) -> str:
    """``'semicolon used; comma also consistent'`` for a delimited source whose delimiter was an ambiguous guess
    (``io.sniff.sniff_delimiter``, R7-05), else ''."""
    from ..io.sniff import delimiter_name

    t = fr.layout.text if fr.layout is not None else None
    if t is None or not t.delimiter_alternatives or t.delimiter is None:
        return ""
    return (f"{delimiter_name(t.delimiter)} used; "
            + ", ".join(delimiter_name(d) for d in t.delimiter_alternatives) + " also consistent")


def _dtype_check(fr: FileResult) -> dict[str, str]:
    results, details = [], []
    for t_i, tp in enumerate(fr.tables):
        src = fr.source_dtypes[t_i] if t_i < len(fr.source_dtypes) else None
        out = fr.out_dtypes[t_i] if t_i < len(fr.out_dtypes) else None
        if src is not None and out is None and fr.written:
            results.append(WARN)
            details.append("pandas reads the source but cannot read the output with default settings")
            continue
        pairs = _dtype_pairs(fr, t_i)
        if pairs is None:
            continue
        diff = [name for name, a, b in pairs if a != b]
        results.append(WARN if diff else PASS)
        details.append(f"differs for {', '.join(diff)}" if diff else "identical")
    if not results:
        return _check(NA, "pandas dtypes not available for this format")
    return _check(_worst(results), "; ".join(details))


def _match_check(fr: FileResult) -> dict[str, str]:
    if not fr.privacy:
        if any("--max-row-matches" in r for r in fr.reasons):
            return _check(FAIL, "remaining chance matches exceed --max-row-matches")
        return _check(NA, "not run")
    results, details = [], []
    for priv in fr.privacy:
        if not priv.get("checked"):
            results.append(NA)
            details.append(priv.get("note", "not checked"))
            continue
        unit = "entities" if priv.get("mode") == "entity" else "rows"
        rem = int(priv.get("remaining_matches") or 0)
        if priv.get("report_only"):
            results.append(WARN)
            details.append(f"{rem} of {priv.get('units')} {unit} match a real one by chance (expected rate above "
                           "20%: reported, not swapped)")
        elif rem:
            results.append(WARN)
            details.append(f"{rem} {unit} still match after {priv.get('rounds')} swap round(s)")
        else:
            results.append(PASS)
            details.append(f"{priv.get('initial_matches', 0)} {unit} matched before swap-redraw, 0 after")
    return _check(_worst(results), "; ".join(details))


def adjusted_columns(fr: FileResult) -> list[tuple[str, list[str]]]:
    out = []
    for tp in shown_tables(fr):
        for p in tp.columns:
            flags = [k for k, v in p.flags.items() if v]
            if flags and p.role not in ("excluded", "null"):
                out.append((p.name, flags))
    return out


def extreme_exposure(fr: FileResult, min_cell: int = 10) -> list[dict[str, Any]]:
    """Columns whose exact source min/max (held by fewer than ``min_cell`` source rows) is in the output (D14)."""
    out: list[dict[str, Any]] = []
    if not fr.synthetic_stats or fr.withheld:
        return out
    for t_i, tp in enumerate(fr.tables):
        syn = fr.synthetic_stats[t_i] if t_i < len(fr.synthetic_stats) else {}
        for p in tp.columns:
            if p.role in ("excluded", "null", "id") or p.model is None:
                continue
            s, r = syn.get(p.index) or {}, p.stats or {}
            for side in ("min", "max"):
                if side not in r or side not in s or r[side] != s[side]:
                    continue
                held = _rows_at_extreme(p, side)
                if held is None or held >= min_cell:
                    continue
                forced = bool(p.flags.get("extreme_value_adjusted"))
                how = "forced" if forced else ("kept" if is_level_generated(p) else "drawn")
                out.append({"table": tp.name, "column": p.name, "extreme": side, "held_by_fewer_than": min_cell,
                            "forced": forced, "how": how})
    return out


#: How an exposed extreme got into the output (report text).
EXPOSURE_HOW = {
    "forced": "forced by the exact-extremes rule (extreme_value_adjusted); --no-exact-extremes COL turns it off",
    "kept": "kept because every observed value of a discrete column is reproduced with its exact count",
    "drawn": "drawn by interpolation, not forced",
}


def _rows_at_extreme(p: Any, side: str) -> int | None:
    m = p.model
    exact = numeric_gen.exact_counts(p, m.nonmissing_counts)  # plain/formatted numbers: exact (R2-12)
    if exact:
        target = min(exact) if side == "min" else max(exact)
        return exact[target]
    arr = m.sorted_numeric
    if arr is not None and len(arr):
        if side == "min":
            return int(np.searchsorted(arr, arr[0], side="right"))
        return int(len(arr) - np.searchsorted(arr, arr[-1], side="left"))
    vals: list[tuple[float, int]] = []
    for raw, c in m.nonmissing_counts.items():
        try:
            vals.append((float(str(raw).strip()), int(c)))
        except (TypeError, ValueError):
            return None
    if not vals:
        return None
    target = min(v for v, _ in vals) if side == "min" else max(v for v, _ in vals)
    return sum(c for v, c in vals if v == target)


def verdict(run: RunResult, files_checks: list[dict[str, dict[str, str]]] | None = None,
            min_cell: int = 10) -> dict[str, Any]:
    """Machine-checkable run verdict (mirrored in REPORT.md and report.json)."""
    if files_checks is None:
        files_checks = [file_checks(fr) for fr in run.files]
    errors = [m for m in run.messages if m.level == "error"]
    warnings = [m for m in run.messages if m.level == "warning"]
    if run.any_failed or errors:
        overall = FAIL
    elif any(fr.status is Status.WARN for fr in run.files) or warnings:
        overall = WARN
    else:
        overall = PASS
    criteria = []
    for key, title, what in CRITERIA:
        results = [fc[key]["result"] for fc in files_checks]
        counts = {r: results.count(r) for r in (PASS, WARN, FAIL, NA, INFO)}
        criteria.append({"key": key, "title": title, "description": what, "result": _worst(results),
                         "counts": counts})
    criteria.append({"key": "run_errors", "title": RUN_CRITERIA[0][1], "description": RUN_CRITERIA[0][2],
                     "result": FAIL if errors else PASS, "counts": {"errors": len(errors)}})
    criteria.append({"key": "run_warnings", "title": RUN_CRITERIA[1][1], "description": RUN_CRITERIA[1][2],
                     "result": WARN if warnings else PASS, "counts": {"warnings": len(warnings)}})
    fmts = sorted({(fr.out_layout.format if fr.out_layout is not None else fr.format) for fr in run.files})
    return {
        "overall": overall,
        "rule": OVERALL_RULE,
        "exit_code": 3 if overall == FAIL else 0,
        "criteria": criteria,
        "files": [{"source": fr.rel_path, "status": fr.status.value, "checks": fc}
                  for fr, fc in zip(run.files, files_checks)],
        "selfcheck_coverage": {f: selfcheck.coverage(f) for f in fmts},
        "min_cell": min_cell,
    }


# --------------------------------------------------------------------------- per-file report
def file_json(fr: FileResult, min_cell: int = 10) -> dict[str, Any]:
    cands = _candidates(fr)
    tables = []
    for t_i, tp in enumerate(shown_tables(fr)):
        pub = tp.to_public()
        syn = fr.synthetic_stats[t_i] if t_i < len(fr.synthetic_stats) else {}
        fid = fr.fidelity[t_i] if t_i < len(fr.fidelity) else {}
        for col, p in zip(pub["columns"], tp.columns):
            col["real"] = col.pop("stats")
            col["synthetic"] = syn.get(col["index"])
            col["fidelity"] = fid.get(col["index"])
            col["messages"] = [message_public(m, cands) for m in p.messages]
        pub["messages"] = [message_public(m, cands) for m in tp.messages]
        pub["privacy"] = fr.privacy[t_i] if t_i < len(fr.privacy) else None
        pub["selfcheck_facts"] = selfcheck.facts(tp)
        tables.append(pub)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": fr.rel_path,
        "output": fr.out_rel_path,
        "format": fr.format,
        "status": fr.status.value,
        "written": fr.written,
        "reasons": fr.reasons,
        "fixes": fr.fixes,
        "fix_commands": [c for f in fr.fixes for c in fix_commands(f, cands)],
        "checks": file_checks(fr),
        "messages": [message_public(m, cands) for m in (file_messages(fr) if fr.withheld else fr.messages)],
        "layout": fr.layout.to_public() if fr.layout and not fr.withheld else None,
        "output_layout": fr.out_layout.to_public() if fr.out_layout and not fr.withheld else None,
        "tables": tables,
        "details_withheld": fr.withheld,
        "selfcheck": fr.selfcheck,
        "leak_scan": {"sensitive_columns": [{"table": t, "column": c, "kind": k} for _, t, c, k in
                                            (sensitive_columns([fr]) if not fr.withheld else [])]},
        "extreme_exposure": extreme_exposure(fr, min_cell),
    }


def _summary_cell(st: dict[str, Any] | None) -> str:
    if not st:
        return ""
    if "level_shares" in st:
        items = list(st["level_shares"].items())
        s = "; ".join(f"{k!r} {v:.3f}" for k, v in items[:6])
        return s + ("; ..." if len(items) > 6 else "")
    if "min" in st:
        return f"min {st['min']}, med {st.get('median')}, mean {st.get('mean')}, sd {st.get('sd')}, max {st['max']}"
    if "length" in st:
        return f"len {st['length']['min']}-{st['length']['max']}, words {st.get('mean_words', '')}"
    if "redacted" in st:
        return str(st["redacted"])
    return ""


def file_markdown(fr: FileResult, min_cell: int = 10, sharing: bool = False) -> str:
    cands = _candidates(fr)
    lines = [f"# {fr.rel_path}", "", f"- Status: **{fr.status.value}**",
             f"- Output: {code_span(fr.out_rel_path)}" + ("" if fr.written else " (not written)"),
             f"- Format: {fr.format}"]
    if fr.withheld:
        why = ("the leak scan could not verify this file" if any(UNVERIFIED_TEXT in r for r in fr.reasons)
               else "a name this file would write holds personal data (D34)")
        lines.append(f"- Details withheld: {why}, so its column table, sheet names, layout and column messages are "
                     "left out of the report")
    if fr.layout is not None and not fr.withheld:
        lay = fr.layout.to_public()
        t = lay.get("text") or {}
        lines.append(f"- Layout: encoding {lay['encoding']}, BOM {'yes' if lay['bom'] else 'no'}"
                     + (f", delimiter {t.get('delimiter')!r}, line ending {t.get('line_ending')}, quoting "
                        f"{t.get('quoting')}" if t else ""))
    if sharing:
        for text, fix in failure_findings(fr, True, cands):
            lines.append(f"- Reason: {text}")
            if fix:
                lines.append(f"- Fix: {fix}")
    else:
        for r in fr.reasons:
            lines.append(f"- Reason: {_md(r)}")
        for f in fr.fixes:
            lines.append(f"- Fix: {fix_markdown(f, cands)}")
    lines += ["", "## Checks", "", "| Check | Key | Result | Detail |", "|---|---|---|---|"]
    checks = file_checks(fr)
    for key, title, _ in CRITERIA:
        c = checks[key]
        lines.append(f"| {title} | `{key}` | {c['result']} | {_md(c['detail'])} |")
    lines.append("")
    for t_i, tp in enumerate(shown_tables(fr)):
        lines += _table_markdown(fr, t_i, tp)
    lines += _privacy_markdown(fr, min_cell)
    if fr.selfcheck:
        lines += ["## Self-check", ""] + [f"- {_md(s)}" for s in fr.selfcheck] + [""]
    msgs = file_messages(fr)
    if msgs:
        lines += ["## Messages", ""]
        for m in msgs:
            col = f" [{_md(m.column)}]" if m.column else ""
            text, fix = plain_finding(m.code, m.level, m.text, m.fix, sharing, cands)
            lines.append(f"- {m.level.upper()}{col} {text}" + (f" Fix: {fix}" if fix else ""))
        lines.append("")
    return "\n".join(lines)


def _table_markdown(fr: FileResult, t_i: int, tp: TableProfile) -> list[str]:
    syn = fr.synthetic_stats[t_i] if t_i < len(fr.synthetic_stats) else {}
    fid = fr.fidelity[t_i] if t_i < len(fr.fidelity) else {}
    title = f"## Table {tp.name}" if tp.name else "## Columns"
    lines = [title, "", f"{tp.n_rows} rows, {len(tp.columns)} columns.", "",
             "| # | Column | Type | Role | Missing % real / syn | Distinct real / syn | Real | Synthetic | Flags | Fidelity |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in tp.columns:
        s = syn.get(p.index) or {}
        r = p.stats or {}
        flags = ", ".join(k for k, v in p.flags.items() if v)
        verdict_ = (fid.get(p.index) or {}).get("verdict", "")
        typ = p.type.value + (f" ({p.subtype})" if p.subtype else "")
        miss = f"{100 * r.get('missing_rate', 0):.1f} / " + (f"{100 * s.get('missing_rate', 0):.1f}" if s else "-")
        dist = f"{r.get('distinct', '')} / {s.get('distinct', '-') if s else '-'}"
        lines.append("| " + " | ".join([str(p.index + 1), _md(p.name), typ, p.role, miss, dist,
                                         _md(_summary_cell(r)), _md(_summary_cell(s)), flags, verdict_]) + " |")
    lines.append("")
    if fid:
        lines += ["Column fidelity (verdict: reason):", ""]
        for p in tp.columns:
            v = fid.get(p.index)
            if not v:
                continue
            lines.append(f"- {code_span(p.name)}: **{v.get('verdict')}** - {_md(v.get('reason', ''))}")
        lines.append("")
    lines += ["Detection reasons:", ""]
    for p in tp.columns:
        amb = f" (ambiguous: {_md(p.ambiguity)})" if p.ambiguous else ""
        lines.append(f"- {code_span(p.name)}: {_md(p.reason)} [{p.source}]{amb}")
    lines.append("")
    names = [c.name for c in tp.columns]
    ent = tp.entity.to_public(names)
    lines += ["IDs and entities:", ""]
    if ent["id_columns"]:
        lines.append(f"- ID columns: {', '.join(ent['id_columns'])}; {ent['n_entities']} entities; "
                     f"rows per ID: {ent['rows_per_id']}")
        lines.append(f"- Within-ID constant columns: {', '.join(ent['within_id_constant_columns']) or 'none'}")
    else:
        lines.append("- No ID column: each row is its own entity.")
    struct = tp.structure.to_public(names)
    mm = struct["missingness"]
    lines.append(f"- Missingness: {mm['n_patterns']} whole-row patterns ({mm['n_pooled_patterns']} pooled, "
                 f"{mm['pooled_rows']} rows)")
    for key in ("functional_dependencies", "derived_templates", "composite_keys", "suggested_rules", "panels"):
        lines.append(f"- {key.replace('_', ' ').capitalize()}: {len(struct[key]) or 'none found'}")
    lines += [f"  - {_md(d)}" for d in structure_mod.describe_structure(tp)]  # AREA D detail lines
    priv = fr.privacy[t_i] if t_i < len(fr.privacy) else None
    if priv:
        if priv.get("checked"):
            lines.append(f"- Chance matches ({priv['mode']}): {priv.get('initial_matches')} before, "
                         f"{priv.get('remaining_matches')} after {priv.get('rounds')} swap rounds"
                         + (" (report only)" if priv.get("report_only") else ""))
        else:
            lines.append(f"- Chance matches: {priv.get('note', 'not checked')}")
    lines.append("")
    return lines


def _privacy_markdown(fr: FileResult, min_cell: int) -> list[str]:
    sens = sensitive_columns([fr]) if not fr.withheld else []
    lines = ["## Privacy", ""]
    if sens:
        lines.append("- Leak scan covers these columns (their real values must appear nowhere): "
                     + ", ".join(f"{code_span(c)} ({KIND_TEXT[k]})" for _, _, c, k in sens))
    else:
        lines.append("- Leak scan: no ID/PII/free-text/high-cardinality text column in this file.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in extreme_exposure(fr, min_cell):
        grouped.setdefault(e["column"], []).append(e)
    for col, es in grouped.items():
        sides = " and ".join("minimum" if e["extreme"] == "min" else "maximum" for e in es)
        how = es[0]["how"]
        fix = (f" Fix if unwanted: {code_span('--no-exact-extremes ' + shell_quote(col))}" if how == "forced"
               else "")
        lines.append(f"- INFO extreme exposure: the exact source {sides} of {code_span(col)}, held by fewer than "
                     f"{es[0]['held_by_fewer_than']} source rows, appear(s) in the synthetic data "
                     f"({_md(EXPOSURE_HOW[how].split(';')[0])}).{fix}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- run report
def run_json(run: RunResult, min_cell: int = 10) -> dict[str, Any]:
    cands = _run_candidates(run)
    fcs = [file_checks(fr) for fr in run.files]
    files = []
    for fr, fc in zip(run.files, fcs):
        warns = _messages(fr, ("warning",))
        rows_out = None
        facts = [selfcheck.facts(tp) for tp in fr.tables]
        if fr.written and facts and all(f is not None for f in facts):
            rows_out = sum(f["rows_out"] for f in facts)
        fcands = _candidates(fr)
        files.append({
            "source": fr.rel_path, "output": fr.out_rel_path if fr.written else None, "format": fr.format,
            "status": fr.status.value, "written": fr.written, "reasons": fr.reasons, "fixes": fr.fixes,
            "fix_commands": [c for f in fr.fixes for c in fix_commands(f, fcands)],
            "rows": sum(t.n_rows for t in fr.tables) if fr.tables else None, "rows_out": rows_out,
            "columns": sum(len(t.columns) for t in fr.tables) if fr.tables else None,
            "warnings": len(warns), "report": file_report_rel(fr.rel_path),
            "checks": {k: v["result"] for k, v in fc.items()},
        })
    warnings = [message_public(m, cands) | {"file": None} for m in run.messages if m.level == "warning"]
    for fr in run.files:
        fcands = _candidates(fr)
        for m in _messages(fr, ("warning",)):
            warnings.append(message_public(m, fcands) | {"file": fr.rel_path})
    notes = [message_public(m, cands) | {"file": None} for m in run.messages if m.level == "info"]
    for fr in run.files:
        for e in extreme_exposure(fr, min_cell):
            note = {"level": "info", "code": "extreme_exposure", "file": fr.rel_path, "column": e["column"],
                    "extreme": e["extreme"], "forced": e["forced"], "how": e["how"],
                    "text": f"the exact source {'minimum' if e['extreme'] == 'min' else 'maximum'}, held by "
                            f"fewer than {e['held_by_fewer_than']} source rows, appears in the synthetic data "
                            f"({EXPOSURE_HOW[e['how']].split(';')[0]})"}
            if e["how"] == "forced":
                ref = f"{fr.rel_path}::{e['column']}"
                note["fix"] = f"--no-exact-extremes {ref}"
                note["fix_commands"] = [f"--no-exact-extremes {shell_quote(ref)}"]
            notes.append(note)
    data = {
        "schema_version": SCHEMA_VERSION,
        "tool": "datanonymous", "version": __version__, "mode": run.mode, "seed": run.seed,
        "seed_was_random": run.seed_was_random,
        "verdict": verdict(run, fcs, min_cell),
        "totals": run.totals(), "files": files,
        "skipped": [{"path": s.rel_path, "reason": s.reason, "kind": skip_kind(s)} for s in run.skipped],
        "failures": [{"source": f.rel_path, "reasons": f.reasons, "fixes": f.fixes,
                      "fix_commands": [c for x in f.fixes for c in fix_commands(x, _candidates(f))]}
                     for f in run.files if f.failed],
        "errors": [message_public(m, cands) for m in run.messages if m.level == "error"],
        "warnings": warnings,
        "notes": notes,
        "docs": run.docs,
    }
    if run.data_mode == "sharing":
        data["data_mode"] = run.data_mode
        data["sharing_package"] = run.package
    return _redact_obj(run, data)


def _exposure_by_column(fr: FileResult, data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(column, 'min and max', how) of the extreme-exposure notes of one file."""
    sides: dict[str, list[str]] = {}
    hows: dict[str, str] = {}
    for n in data.get("notes") or []:
        if n.get("code") == "extreme_exposure" and n.get("file") == fr.rel_path:
            sides.setdefault(n["column"], []).append(n["extreme"])
            hows[n["column"]] = n.get("how", "forced")
    return [(c, " and ".join(v), hows[c]) for c, v in sides.items()]


def _redactors(run: RunResult) -> list[Any]:
    return [m for m in run.messages if callable(getattr(m, "redact_payload", None))]


def _redact_obj(run: RunResult, obj: Any) -> Any:
    for m in _redactors(run):
        obj = m.redact_obj(obj)
    return obj


def _redact_text(run: RunResult, text: str) -> str:
    for m in _redactors(run):
        text = m.redact_text(text)
    return text


def redact_console(run: RunResult, text: str) -> str:
    """Console text (``-v`` tracebacks ...) with every redaction of the run applied (leak scan, D34 names)."""
    return _redact_text(run, text)


def redact_console_obj(run: RunResult, obj: Any) -> Any:
    """A JSON-able console payload (``inspect --json``) with every redaction of the run applied."""
    return _redact_obj(run, obj)


#: How the manifest shows an ``--na-value`` token the run pseudonymised (R8-05).
NA_VALUE_SHOWN = "[pseudonymized NA token]"


def _public_settings(run: RunResult, settings: Settings) -> dict[str, Any]:
    """The settings as the manifest shows them: an ``--na-value`` token that the D35 allowlist pseudonymised in any
    column of the run is real text the run never writes, so it shows as :data:`NA_VALUE_SHOWN` (R8-05)."""
    pub = settings.to_public()
    gone = {t for fr in run.files for tp in fr.tables for p in tp.columns
            for t in (*na_pseudonymized(p), *pseudonymized(p))}
    if gone and pub.get("na_value"):
        pub["na_value"] = [NA_VALUE_SHOWN if isinstance(v, str) and v.strip() in gone else v for v in pub["na_value"]]
    return pub


#: sharing mode: the input files not written to the output folder (report folder only: a path can hold a name)
FAILURES_CSV = "sharing-failures.csv"


def failures_csv(run: RunResult) -> str:
    """One row per input file not written: ``input_path, outcome (failed | skipped), reason``."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["input_path", "outcome", "reason"])
    for fr in run.files:
        if fr.failed:
            plain = [text if not fix else f"{text} {fix}" for text, fix in failure_findings(fr, True, markdown=False)]
            writer.writerow([fr.rel_path, "failed", "; ".join(dict.fromkeys(plain))])
    for s in run.skipped:
        writer.writerow([s.rel_path, "skipped", s.reason])
    return buf.getvalue()


def build_report(run: RunResult, settings: Settings) -> dict[str, bytes]:
    """Every report-dir file as {posix relative path: bytes}; nothing is written.

    The pipeline builds the report first so the AREA G leak scan
    (``privacy.leaks.scan_outputs``) can check it, then writes it with
    :func:`write_report_files`. When the scan found something, the rebuilt
    report is redacted by the scan's private redactor.
    """
    min_cell = settings.min_cell
    files: dict[str, bytes] = {}
    data = run_json(run, min_cell)
    files["report.json"] = _json(data).encode("utf-8")
    files["REPORT.md"] = run_markdown(run, data).encode("utf-8")
    files[PRIVATE_MARKER] = PRIVATE_MARKER_TEXT.encode("utf-8")
    manifest = {
        "tool": "datanonymous", "schema_version": SCHEMA_VERSION, "versions": versions(), "seed": run.seed,
        "seed_was_random": run.seed_was_random,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "input": _posix(run.input), "out_dir": _posix(run.out_dir), "report_dir": _posix(run.report_dir),
        "settings": _public_settings(run, settings),
        "files": [{"source": f.rel_path, "output": f.out_rel_path if f.written else None, "format": f.format,
                   "status": f.status.value} for f in run.files],
        "skipped": [s.rel_path for s in run.skipped],
        "docs": run.docs,
    }
    if run.data_mode == "sharing":
        manifest["sharing_package"] = run.package
    files["manifest.json"] = _json(manifest).encode("utf-8")
    if run.data_mode == "sharing" and run.mode == "run":
        files[FAILURES_CSV] = failures_csv(run).encode("utf-8")
    for fr in run.files:
        base = str(PurePosixPath("files", fr.rel_path))
        files[base + ".json"] = _json(file_json(fr, min_cell)).encode("utf-8")
        files[base + ".md"] = file_markdown(fr, min_cell, run.data_mode == "sharing").encode("utf-8")
    for m in _redactors(run):
        files = m.redact_payload(files)
    return files


def write_report_files(report_dir: Path, files: dict[str, bytes]) -> None:
    """Write a built report (see :func:`build_report`) into ``report_dir``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    for rel, payload in files.items():
        dest = report_dir / Path(*PurePosixPath(rel).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        replace_bytes(dest, payload)  # an entry left at that name (a link, a hard link) is replaced, not followed


def write_report(run: RunResult, settings: Settings) -> None:
    """Build and write the report in one step."""
    assert run.report_dir is not None
    write_report_files(run.report_dir, build_report(run, settings))


def nulled_columns(run: RunResult) -> list[tuple[str, str, bool, str, str]]:
    """(file, column, nulled automatically, plain reason, alternative or '') of every Nulled column of the files
    written: by the operator (``--null``) or by sharing mode (ADR-0001)."""
    from ..profile.sharing import AUTO_NULL_ALTERNATIVE_KEY, AUTO_NULL_SOURCE

    out = []
    for fr in run.files:
        if not fr.written or fr.withheld:
            continue
        for tp in fr.tables:
            for p in tp.columns:
                if p.role != "null":
                    continue
                automatic = p.source == AUTO_NULL_SOURCE
                reason = p.reason.split(" because ", 1)[1] if automatic and " because " in p.reason else ""
                out.append((fr.rel_path, p.name, automatic, reason, p.params.get(AUTO_NULL_ALTERNATIVE_KEY) or ""))
    return out


#: First line of REPORT.md and the console's private-folder wording: the report describes the source data
PRIVATE_BANNER = "PRIVATE – DO NOT SHARE"

#: The console's labels of the run's two destinations (ASCII; the start and the end of every run, issue 19)
SHAREABLE_LABEL = "Shareable synthetic files"
PRIVATE_LABEL = "PRIVATE report - do not share"

#: The report folder's marker file: it says why the folder is private (never written to the output folder)
PRIVATE_MARKER = "PRIVATE-DO-NOT-SHARE.txt"
PRIVATE_MARKER_TEXT = (
    "PRIVATE - DO NOT SHARE\n\n"
    "This folder is the private report of a datanonymous run. It describes the real source data, so never share, "
    "upload or send it to anyone.\n"
    "Share only the output folder, which holds the synthetic files.\n"
)


def skip_kind(s: Any) -> str:
    """``deselected`` (left out by --include / --exclude-path), ``routine`` (not a table format at all: a PDF, an
    image, a Word document) or ``actionable`` (looks tabular but could not be read or is not supported)."""
    return "deselected" if s.deselected else "routine" if s.routine else "actionable"


def routine_skip_line(run: RunResult) -> str | None:
    """One line for every routine skipped input, with counts by extension: ``4 non-table files skipped (3 .pdf,
    1 .png)``; None when there were none."""
    routine = [s for s in run.skipped if skip_kind(s) == "routine"]
    if not routine:
        return None
    counts = Counter(s.ext or "other" for s in routine)
    parts = ", ".join(f"{n} {ext}" for ext, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    n = len(routine)
    return f"{n} non-table file{'s' if n != 1 else ''} skipped ({parts})"


def deselected_line(run: RunResult) -> str | None:
    n = sum(1 for s in run.skipped if skip_kind(s) == "deselected")
    return f"{n} input{'s' if n != 1 else ''} left out by --include / --exclude-path" if n else None


def written_files(run: RunResult) -> list[tuple[str, int, int]]:
    """(output path, rows, columns) of every file written to the output folder."""
    out = []
    for fr in run.files:
        if not fr.written or fr.withheld:
            continue
        facts = [selfcheck.facts(tp) for tp in fr.tables]
        rows = (sum(f["rows_out"] for f in facts) if facts and all(f is not None for f in facts)
                else sum(t.n_rows for t in fr.tables))
        out.append((fr.out_rel_path, rows, sum(len(t.columns) for t in fr.tables)))
    return out


def written_line(run: RunResult) -> str:
    n = len(written_files(run))
    return f"{n} file{'s' if n != 1 else ''} written" if n else "No files were written"


#: how an operator can act on a column sharing mode nulled on its own
AUTO_NULL_FIX_MD = ("If it holds IDs, declare it with `--id-col`; if you do not want an empty column, leave it out "
                    "with `--exclude`.")
AUTO_NULL_FIX_PLAIN = AUTO_NULL_FIX_MD.replace("`", "")


def attention_items(run: RunResult, data: dict[str, Any], markdown: bool = True) -> list[tuple[str, str, list[str]]]:
    """What needs the operator's attention, as (where, text, fixes): files left out, actionable skipped inputs, run
    errors, columns nulled automatically and (sharing mode) warnings that block a file. Routine skipped inputs and
    inputs deselected on purpose are not here."""
    sharing = run.data_mode == "sharing"
    where_of = (lambda p: code_span(_md(p))) if markdown else (lambda p: p)
    esc = _md if markdown else (lambda x: x)
    by_source = {fr.rel_path: fr for fr in run.files}
    items: list[tuple[str, str, list[str]]] = []
    for f in data["failures"]:
        fr = by_source.get(f["source"])
        cands = _candidates(fr)
        if sharing and fr is not None:
            found = failure_findings(fr, True, cands, markdown)
            text = " ".join(dict.fromkeys(t for t, _ in found))
            fixes = list(dict.fromkeys(fx for _, fx in found if fx))
        else:
            text = esc("; ".join(f["reasons"]))
            fixes = [fix_markdown(fx, cands) if markdown else fix_plain(fx, cands) for fx in dict.fromkeys(f["fixes"])]
        items.append((where_of(f["source"]), f"not written. Reason: {text}", fixes))
    for s in run.skipped:
        if skip_kind(s) != "actionable":
            continue
        text, fix = plain_finding(s.code or "skipped", "warning", s.reason, None, sharing, (), markdown)
        items.append((where_of(s.rel_path), text, [fix] if fix else []))
    run_cands = _run_candidates(run)
    for e in data["errors"]:
        text, fix = plain_finding(e["code"], "error", e["text"], e.get("fix"), sharing, run_cands, markdown)
        items.append(("run", text, [fix] if fix else []))
    for file, column, automatic, reason, alternative in nulled_columns(run):
        if automatic:
            why = (f"nulled automatically (written with every cell empty) because {esc(reason)}" if reason
                   else "nulled automatically (written with every cell empty)")
            items.append((f"{where_of(file)} {code_span(column) if markdown else column}", why,
                          [esc(alternative) if alternative else AUTO_NULL_FIX_MD if markdown else AUTO_NULL_FIX_PLAIN]))
    for w in data["warnings"]:
        if not _shown_warning(w, sharing):
            continue
        fr = by_source.get(w.get("file")) if w.get("file") else None
        cands = _candidates(fr) if fr is not None else run_cands
        text, fix = plain_finding(w["code"], "warning", w["text"], w.get("fix"), True, cands, markdown)
        where = where_of(w["file"]) if w.get("file") else "run"
        if w.get("column"):
            where += f" [{esc(w['column'])}]"
        items.append((where, text, [fix] if fix else []))
    return items


def _shown_warning(w: dict[str, Any], sharing: bool) -> bool:
    """True when a warning blocks a file in sharing mode: it is listed under "What needs attention", not in the
    Appendix's warnings."""
    entry = catalogue.lookup(w["code"])
    return sharing and entry is not None and entry.kind == catalogue.PROBLEM


def run_markdown(run: RunResult, data: dict[str, Any]) -> str:
    sharing = run.data_mode == "sharing"
    v = data["verdict"]
    t = data["totals"]
    run_cands = _run_candidates(run)
    done = written_files(run)
    lines = [f"# {PRIVATE_BANNER}", "",
             "This is the private report of a datanonymous run. It describes the source data: keep it out of "
             f"anything you share, and send only the synthetic files. Read {PRIVATE_MARKER} in this folder for why.",
             "",
             "## What was generated", "",
             f"{written_line(run)} to the output folder {code_span(_posix(run.out_dir))}." if done
             else f"{written_line(run)}."]
    if done:
        lines += ["", "| File | Rows | Columns |", "|---|---|---|"]
        lines += [f"| {code_span(_md(path))} | {rows} | {cols} |" for path, rows, cols in done]
    for extra in (routine_skip_line(run), deselected_line(run)):
        if extra:
            lines += ["", extra + "."]
    lines += ["", "<!-- What changed: reserved for the \"What changed\" section (a later ticket fills it); "
                  "omitted until it has content -->", "", "## What needs attention", ""]
    items = attention_items(run, data)
    if not items:
        lines += ["Nothing needs attention.", ""]
    for where, text, fixes in items:
        lines.append(f"- {where}: {text}")
        lines += [f"  - Fix: {fx}" for fx in fixes]
    if items:
        lines.append("")
    lines += ["## Appendix", "",
              "The details behind the sections above, for review: the verdict and its criteria, per-file checks, "
              "statistics, utility warnings, notes and every skipped input.", "",
              "### Verdict", "", f"Overall verdict: {v['overall']}", "",
              f"Rule: {v['rule']}. Exit code: {v['exit_code']}.", "",
              "Counts are numbers of files; N/A = the check did not run (the file FAILed earlier); INFO = "
              "informational only.", "",
              "| Check | Key | Result | PASS | WARN | FAIL | N/A | INFO | What it checks |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in v["criteria"]:
        n = c["counts"]
        if c["key"] in ("run_errors", "run_warnings"):
            what = next(iter(n))
            lines.append(f"| {c['title']} | `{c['key']}` | {c['result']} | - | - | - | - | - | "
                         f"{_md(c['description'])} ({n[what]} {what}) |")
            continue
        lines.append(f"| {c['title']} | `{c['key']}` | {c['result']} | {n.get(PASS, 0)} | {n.get(WARN, 0)} | "
                     f"{n.get(FAIL, 0)} | {n.get(NA, 0)} | {n.get(INFO, 0)} | {_md(c['description'])} |")
    lines.append("")
    if v["files"]:
        keys = [k for k, _, _ in CRITERIA]
        lines += ["### Per-file checks", "",
                  "| File | Status | " + " | ".join(f"`{k}`" for k in keys) + " |",
                  "|---|---|" + "---|" * len(keys)]
        for f in v["files"]:
            cells = []
            for k in keys:
                c = f["checks"][k]
                cell = c["result"]
                if k == "rows" and c["result"] in (PASS, FAIL):
                    cell += f" ({c['detail']})"
                cells.append(_md(cell))
            lines.append(f"| {code_span(_md(f['source']))} | {f['status']} | " + " | ".join(cells) + " |")
        lines.append("")
    if v["selfcheck_coverage"]:
        lines += ["Self-check coverage (what is compared after re-reading each output):", ""]
        for fmt, aspects in v["selfcheck_coverage"].items():
            lines.append(f"- {fmt}: {', '.join(aspects)}; pandas dtypes are compared separately (warnings).")
        lines.append("")
    lines += ["Adjusted flags never fail a check: `rare_category_adjusted` means every source level and NA token "
              "was kept at least once (which changed a count); `extreme_value_adjusted` means the exact source "
              "minimum and maximum were emitted (D14). Their deviation is reported, not an error.", ""]
    lines += ["### Run", "",
              f"- Input: {code_span(_posix(run.input))}", f"- Out dir: {code_span(_posix(run.out_dir))}",
              f"- Report dir: {code_span(_posix(run.report_dir))}",
              f"- Seed: {run.seed}" + (" (random; pass --seed to reproduce)" if run.seed_was_random else ""),
              f"- Files: {t['files']} tabular ({t['PASS']} PASS, {t['WARN']} WARN, {t['FAIL']} FAIL), "
              f"{t['skipped']} skipped", "",
              "### Files", "",
              "| Status | Source | Output | Rows in | Rows out | Columns | Warnings | Details |",
              "|---|---|---|---|---|---|---|---|"]
    for f in data["files"]:
        lines.append(f"| {f['status']} | {code_span(_md(f['source']))} | "
                     f"{code_span(_md(f['output'])) if f['output'] else '-'} "
                     f"| {f['rows'] if f['rows'] is not None else '-'} "
                     f"| {f['rows_out'] if f['rows_out'] is not None else '-'} "
                     f"| {f['columns'] if f['columns'] is not None else '-'} "
                     f"| {f['warnings']} | [{_md(f['report'])}]({link_target(f['report'])}) |")
    lines.append("")
    if data["skipped"]:
        lines += ["### Skipped inputs (not copied)", ""]
        for s in data["skipped"][:500]:
            lines.append(f"- {code_span(_md(s['path']))} ({s['kind']}): {fix_markdown(s['reason'])}")
        if len(data["skipped"]) > 500:
            lines.append(f"- ... {len(data['skipped']) - 500} more in report.json")
        lines.append("")
    by_source = {fr.rel_path: fr for fr in run.files}
    other_warnings = [w for w in data["warnings"] if not _shown_warning(w, sharing)]
    if other_warnings:
        lines += ["### Warnings", ""]
        for w in other_warnings:
            where = code_span(_md(w["file"])) if w.get("file") else "run"
            col = f" [{_md(w['column'])}]" if w.get("column") else ""
            cands = _candidates(by_source.get(w.get("file"))) if w.get("file") else run_cands
            text, fix = plain_finding(w["code"], "warning", w["text"], w.get("fix"), sharing, cands)
            lines.append(f"- {where}{col}: {text}" + (f" Fix: {fix}" if fix else ""))
        lines.append("")
    nulled = nulled_columns(run)
    if nulled:
        lines += ["### Nulled columns", "",
                  "Written with their header and position kept and every cell empty; their real values appear "
                  "nowhere in the output.", ""]
        for file, column, automatic, reason, alternative in nulled:
            how = (f"nulled automatically because {_md(reason)}" if automatic
                   else "nulled by you (--null)")
            if alternative:
                how += f". Instead: {_md(alternative)}"
            lines.append(f"- {code_span(_md(file))} {code_span(column)}: {how}")
        lines.append("")
    notes = [n for n in data.get("notes") or [] if n.get("code") != "extreme_exposure"]
    out_notes: list[str] = []
    for fr in run.files:
        cols = adjusted_columns(fr)
        if cols:
            out_notes.append(f"- INFO {code_span(_md(fr.rel_path))}: adjusted columns (expected, never an error): "
                             + "; ".join(f"{code_span(c)} ({', '.join(fl)})" for c, fl in cols))
        exposed = _exposure_by_column(fr, data)
        if exposed:
            forced = [c for c, _, how in exposed if how == "forced"]
            out_notes.append(
                f"- INFO {code_span(_md(fr.rel_path))}: extreme exposure - the exact source minimum/maximum, "
                f"held by fewer than {data['verdict'].get('min_cell', 10)} source rows, appears in the synthetic "
                "data (D14): " + "; ".join(f"{code_span(c)} ({sides}, {how})" for c, sides, how in exposed)
                + ". forced = exact-extremes rule, kept = discrete column with every observed value, drawn = "
                "interpolation."
                + (" Fix if unwanted (forced only): "
                   + ", ".join(code_span(f"--no-exact-extremes {shell_quote(fr.rel_path + '::' + c)}")
                               for c in forced) if forced else ""))
    for n in notes:
        where = code_span(_md(n["file"])) if n.get("file") else "run"
        col = f" [{_md(n['column'])}]" if n.get("column") else ""
        fix = f" Fix: {fix_markdown(n['fix'], run_cands)}" if n.get("fix") else ""
        out_notes.append(f"- INFO {where}{col}: {_period(_md(n['text']))}{fix}")
    if out_notes:
        lines += ["### Notes", ""] + out_notes + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- console
def console_summary(run: RunResult) -> list[str]:
    fcs = [file_checks(fr) for fr in run.files]
    v = verdict(run, fcs)
    lines = [f"datanonymous {__version__}  seed={run.seed}"
             + ("  (random; pass --seed to reproduce)" if run.seed_was_random else ""),
             "What was generated"]
    for fr in run.files:
        if fr.failed:
            continue
        rows = sum(t.n_rows for t in fr.tables) if fr.tables else 0
        ncol = sum(len(t.columns) for t in fr.tables) if fr.tables else 0
        n_warn = len(_messages(fr, ("warning",)))
        lines.append(f"  {fr.status.value:4}  {fr.rel_path} -> {fr.out_rel_path}  ({rows} rows, {ncol} cols)"
                     + (f"  {n_warn} warning(s)" if n_warn else ""))
    lines.append(f"  {written_line(run)}")
    for extra in (routine_skip_line(run), deselected_line(run)):
        if extra:
            lines.append(f"  {extra}")
    lines.append("What needs attention")
    items = attention_items(run, run_json(run), markdown=False)
    if not items:
        lines.append("  Nothing needs attention.")
    for where, text, fixes in items:
        lines.append(f"  {where}: {text}" + (f"  fix: {'; '.join(fixes)}" if fixes else ""))
    sharing = run.data_mode == "sharing"
    others = [m for m in run.messages if m.level == "warning" and not _shown_warning({"code": m.code}, sharing)]
    if others:
        lines.append("Other run warnings (details in the report's Appendix)")
        for m in others:
            lines.append(f"  {m.text}")
    t = run.totals()
    lines.append(f"Totals: {t['files']} file(s): {t['PASS']} PASS, {t['WARN']} WARN, {t['FAIL']} FAIL; "
                 f"{t['skipped']} skipped")
    bad = [c for c in v["criteria"] if c["result"] in (WARN, FAIL)]
    lines.append(f"Verdict: {v['overall']}" + (" (" + ", ".join(f"{c['key']} {c['result']}" for c in bad) + ")"
                                                if bad else ""))
    if run.out_dir is not None:
        lines.append(f"Out dir: {run.out_dir}")
    if run.report_dir is not None:
        lines.append(f"Report: {run.report_dir / 'REPORT.md'}")
        not_written = sum(1 for fr in run.files if fr.failed) + len(run.skipped)
        if run.data_mode == "sharing" and not_written:
            lines.append(f"{not_written} input file(s) were not written to the output folder; "
                         f"see {run.report_dir / FAILURES_CSV}")
    if run.out_dir is not None and run.report_dir is not None:
        lines.append(f"{SHAREABLE_LABEL}: {run.out_dir}")
        lines.append(f"{PRIVATE_LABEL}: {run.report_dir}")
    return [ascii_text(_redact_text(run, x)) for x in lines]


def inspect_lines(run: RunResult, verbose: int = 0) -> list[str]:
    lines = [f"datanonymous {__version__} inspect (nothing is written)"]
    for fr in run.files:
        cands = _candidates(fr)
        lines.append("")
        lines.append(f"== {fr.rel_path}  [{fr.format}]  {fr.status.value if fr.failed else ''}".rstrip())
        if fr.failed:
            fixes = [fix_plain(f, cands) for f in dict.fromkeys(fr.fixes)]
            lines.append(f"   FAIL: {'; '.join(fr.reasons)}" + (f"  fix: {'; '.join(fixes)}" if fixes else ""))
            continue
        for tp in fr.tables:
            if tp.name:
                lines.append(f"   -- table {tp.name}")
            lines.append(f"   {tp.n_rows} rows, {len(tp.columns)} columns")
            width = max([len(p.name) for p in tp.columns] + [6])
            for p in tp.columns:
                typ = p.type.value + (f"({p.subtype})" if p.subtype else "")
                role = {"id": "ID", "entity": "CONST-IN-ID", "row": "", "null": "NULL", "excluded": "EXCLUDED"}[p.role]
                amb = "  AMBIGUOUS: " + p.ambiguity if p.ambiguous and p.ambiguity else ""
                miss = p.stats.get("missing_rate", 0) * 100 if p.stats else 0
                lines.append(f"   {p.name:<{width}}  {typ:<34} {role:<11} miss {miss:5.1f}%  {p.reason}{amb}")
            names = [c.name for c in tp.columns]
            ent = tp.entity.to_public(names)
            if ent["id_columns"]:
                lines.append(f"   IDs: {', '.join(ent['id_columns'])}; {ent['n_entities']} entities; rows per ID "
                             f"{ent['rows_per_id']}; within-ID constants: "
                             f"{', '.join(ent['within_id_constant_columns']) or 'none'}")
            struct = tp.structure.to_public(names)
            mm = struct["missingness"]
            found = {k: len(struct[k]) for k in ("functional_dependencies", "derived_templates", "composite_keys",
                                                 "suggested_rules", "panels")}
            lines.append(f"   Structure: {mm['n_patterns']} missingness pattern(s) ({mm['n_pooled_patterns']} pooled); "
                         + ", ".join(f"{k.replace('_', ' ')}: {v or 'none found'}" for k, v in found.items()))
            lines += [f"     {d}" for d in structure_mod.describe_structure(tp)]  # AREA D detail lines
        for m in fr.all_messages():
            if m.level != "info" or verbose:
                col = f" [{m.column}]" if m.column else ""
                fix = f"  fix: {fix_plain(m.fix, cands)}" if m.fix else ""
                lines.append(f"   {m.level.upper()}{col}: {m.text}{fix}")
    if run.skipped:
        lines.append("")
        for s in run.skipped:
            lines.append(f"   SKIP  {s.rel_path}  ({s.reason})")
    for m in run.messages:
        lines.append(f"   {m.level.upper()}: {m.text}")
    return [ascii_text(_redact_text(run, x)) for x in lines]
