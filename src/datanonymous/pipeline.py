"""Run orchestration (FOUNDATION; shared file).

``run``:  discover -> pass 1 per file (read, detect, profile; a per-file error
becomes a FAIL entry with its fix and the run continues, D19; a usage error
such as contradictory column flags stops the run with exit 2 before anything is
written; a file that is not a table, e.g. a prose ``.txt``, is listed as
skipped) -> run-level planning (cross-file structure hook; a file with a column
holding another ID column's real IDs is profiled again with that column as an
ID column, :func:`plan_key_refs`; a file with a table that has no recognised ID column but repeated keys of 8 or
more digits under a header that names no measure, count or flag FAILs with the flags that settle it,
:func:`fail_unrecognised_keys` (D38); the run's real person IDs, which no value
written verbatim may equal, ``privacy.realids.plan`` (R9-01); the D35 decision of every real text unit over
the whole run, ``profile.table.finish_copy_gate``, and every other real value that is a real person ID
pseudonymised, ``profile.table.finish_real_ids``; ID registry shared by every file, D8;
identity-mapped ID columns and ID-width notes are marked on the profiles; D34
screen of every header, sheet, file and folder name the run would write
(``privacy.headers``: a name holding personal data FAILs its files, which are
then shown with redacted paths and withheld details); the run's real texts that
it never writes, of any column, excluded and nulled ones included, which no
generator writes and the leak scan looks for in every output text column
(``privacy.texts``, ``leaks.plan_real_texts``); the run's real full names (and
the person names of flagged names and of those texts) and real phone/SSN keys,
which no generator writes, and the run's real texts no D35 pseudonym may equal,
and the run's one pseudonym map) ->
pass 2 per file (generate, privacy swap-redraw, row-level
structure (``structure.finalize_rows``: composite keys, duplicate rows, sort
order), leak check, render, write to a temporary file, self-check by re-reading with the OUTPUT
format's handler, move into place; a native date/time column's statistics and fidelity come from the re-read
values, with a ``written_values_differ`` warning when the writer stored a value differently, R4-06) ->
out-dir docs (synthetic-only, D20) ->
report built in memory -> AREA G leak scan over the outputs and the report
payload -> report dir written (aggregates only, D21).

``inspect``: discover + pass 1 + the D34 screen of names, then print the detection table; writes nothing.

Extension points
================
Every hook below is a documented function with a stable signature and a trivial
default in the area's own module. Area agents edit only their own modules.

====  ===============================  ===================================  ==========================================
Area  Module                           Function                             Must return
====  ===============================  ===================================  ==========================================
A     detect/dates.py                  detect_dates(ctx)                    Detection (DATE, DATE_MIXED_FORMATS,
                                                                            DATE_AS_INT, DATETIME[_TZ], YEAR_MONTH,
                                                                            TIME, DURATION) or None
A     detect/dates.py                  date_stats(values, profile)          dict of report-safe aggregates
A     detect/dates.py                  date_numbers(values, profile)        float array aligned with the value
                                                                            strings (day/second ordinals; NaN for
                                                                            tokens) or None: model.sorted_numeric
                                                                            and the fidelity KS check
A     detect/dates.py                  doc_fields(profile, syn_stats)       dict of dictionary fields ('readr'
                                                                            replaces the readr letter)
A     generate/dates.py                generate_dates(profile, model, n,    (values, flags dict)
                                       rng)
A     generate/dates.py                render_dates(profile, values)        object array of source-format strings
B     detect/numeric.py                detect_numeric_extended(ctx)         Detection (NUMERIC_FORMATTED,
                                                                            NUMERIC_WITH_TOKENS, DECIMAL, FLOAT
                                                                            scientific) or None
B     detect/numeric.py                numeric_stats_extended(values,       dict of report-safe aggregates; values
                                       profile)                             are the stripped value STRINGS
B     detect/numeric.py                numbers_extended(values, profile)    float array aligned with the strings
                                                                            (NaN for tokens) or None (KS, model)
B     detect/numeric.py                doc_fields(profile, syn_stats)       dict of dictionary fields
B     generate/numeric.py              generate_numeric_extended(profile,   (values, flags dict)
                                       model, n, rng)
B     generate/numeric.py              render_numeric_extended(profile,     object array of strings
                                       values)
C     detect/text.py                   detect_text_semantics(ctx)           Detection (EMAIL, PERSON_NAME, PHONE,
                                                                            SSN, ZIP, STATE, CITY, STREET_ADDRESS,
                                                                            IP, URL, MULTI_SELECT, LAT, LON,
                                                                            FREE_TEXT) or None
C     detect/text.py                   apply_pii_precedence(ctx, det)       Detection (D12 guard; default provided)
C     detect/text.py                   classify_overflow_text(ctx)          Detection (FREE_TEXT or
                                                                            HIGH_CARD_CATEGORICAL; default provided)
C     detect/text.py                   pii_subtype_detection(ctx, subtype)  Detection for --pii-col
C     detect/text.py                   text_stats(values, profile)          dict of report-safe aggregates
C     detect/text.py                   doc_fields(profile, syn_stats)       dict of dictionary fields
C     generate/pii.py                  generate_pii(profile, model, n, rng, (strings, flags); never a real value
                                       faker_factory)
C     generate/text.py                 generate_text_extended(profile,      (strings, flags) (MULTI_SELECT ...)
                                       model, n, rng)
D     detect/ids.py                    id_candidate_reason(ctx, det)        str reason or None (D7 warning)
D     profile/entity.py                id_namespace(file, table, profile)   registry key (str) shared across files
D     profile/entity.py                find_aliases_and_nesting(table,      (alias groups, nesting list)
                                       profiles, entity)
D     profile/structure.py             find_functional_dependencies(table,  list of dicts (names/counts only)
                                       profiles)
D     profile/structure.py             find_derived_templates(table,        list of dicts
                                       profiles)
D     profile/structure.py             find_composite_keys(table, profiles, list of dicts
                                       entity)
D     profile/structure.py             suggest_inequality_rules(table,      list of dicts (suggestions only)
                                       profiles)
D     profile/structure.py             detect_panels(table, profiles,       list of dicts (D16 warnings)
                                       entity)
D     profile/structure.py             swap_groups(structure, profiles)     list of column-index groups
D     profile/structure.py             cross_file_structure(files,          list of run-level Messages
                                       settings)
D     profile/structure.py             apply_structure(table_profile,       list of Messages (edits columns in place)
                                       columns, rng_for, settings)
D     profile/structure.py             finalize_rows(table_profile, synth,  list of Messages (after privacy swap-redraw:
                                       rng_for, settings)                   composite keys, duplicate rows, sort order;
                                                                            reorders synth rows in place)
E     io/excelio.py                    read_xlsx / read_xls /               ReadResult
                                       read_html_table(req)
E     io/excelio.py                    write_xlsx / write_html_table(req)   None (file written; req.seed/rng_for
                                                                            for placeholders; native details in
                                                                            req.tables[k].profile.columns[i].native)
E     io/excelio.py                    detect_native(ctx)                   Detection or None
E     io/excelio.py                    layout_diff(src, out, src_tables,    list of issue strings
                                       out_tables)
E     io/excelio.py                    inferred_dtypes(path, layout, table) list of dtype strings or None
E     io/excelio.py                    inferred_dtypes_html(path, layout,   list of dtype strings or None
                                       table)
F     io/arrowio.py                    read_parquet(req), read_feather(req) ReadResult
F     io/arrowio.py                    write_parquet(req),                  None (file written; also SAS output)
                                       write_feather(req)
F     io/arrowio.py                    detect_native(ctx)                   Detection or None
F     io/arrowio.py                    layout_diff(src, out, src_tables,    list of issue strings
                                       out_tables)
F     io/arrowio.py                    inferred_dtypes(path, layout, table) list of dtype strings or None
F     io/statio.py                     read_dta(req), read_sas(req)         ReadResult
F     io/statio.py                     write_dta(req)                       None (file written)
F     io/statio.py                     detect_native(ctx),                  Detection or None / list of strings /
                                       layout_diff(src, out, src_tables,    dtype strings or None
                                       out_tables), inferred_dtypes(...)
F     io/jsonio.py                     read_json(req), read_jsonl(req)      ReadResult
F     io/jsonio.py                     write_json(req), write_jsonl(req)    None (file written)
F     io/jsonio.py                     detect_native(ctx),                  Detection or None / list of strings /
                                       layout_diff(src, out, src_tables,    dtype strings or None
                                       out_tables), inferred_dtypes(...)
F     io/compress.py                   read_bytes(path, compression)        decompressed bytes
F     io/compress.py                   write_bytes(path, data,              None; plus the COMPRESSIBLE_FORMATS set
                                       compression, layout)
G     privacy/leaks.py                 scan_outputs(run_result,             list of run-level Messages ('error' ->
                                       report_payload)                      exit 3); report_payload = {report-dir
                                                                            relative path: bytes}, not yet written
G     report/report.py, report/        build_report, write_report_files,    (polish; same inputs)
      fidelity.py, report/docs.py      file_markdown, compare_column,
                                       write_docs
====  ===============================  ===================================  ==========================================

Shared contracts the areas rely on:

* ``ColumnProfile.native`` (PRIVATE) carries ``RawColumn.native`` to writers for
  every column, including overridden, ``--null`` and ``--exclude`` columns.
* ``FileLayout.extra`` / ``RawTable.layout`` keys starting with ``_`` are
  private (never in the report); use them for anything that may quote real data.
* ``registry.FormatHandler.out_format`` (``xls`` -> ``xlsx``, ``sas`` ->
  ``parquet``): the self-check re-reads with the output format's handler.
* ``profile.stats.numeric_view`` is the one numeric reading of a column
  (continuous numbers, A ``date_numbers``, B ``numbers_extended``), used for
  ``ColumnModel.sorted_numeric`` and the KS check.
* ``Settings.structure_disabled(switch)`` (config ``structure: {disable: [...]}``)
  and ``Settings.rules`` (config ``rules:``) for AREA D's structural switches.
* ID planning (``plan_ids``) sets ``params['_identity_mapped']`` (PRIVATE; the
  never-copy scan skips such columns), ``params['_id_prefix']`` (PRIVATE: the
  letter prefix the synthetic IDs kept, which the leak scan's username rule
  re-checks, R5-02) and ``params['id_mapping']`` (public) on every ID column
  from its namespace, and turns ``IdNamespace.notes`` (e.g. IDs whose digit
  count had to change) into warnings. Identity is a namespace property
  (``generate.ids``), never a single column's. The registry also gets every
  real e-mail address held in the run (R8-03: any column, excluded and nulled
  ones, addresses inside labels and free text, titles; read from the run's
  real-text registry, ``privacy.leaks.real_texts_of``), every other text of
  that registry (R9-03: no synthetic ID equals one) and the real IDs of
  files that FAILed before planning, which no synthetic ID may take (R5-02).

Hook call order inside ``detect/types.detect_column`` is documented there;
``generate/table.generate_table`` documents the generation order.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import shutil
from pathlib import Path

import numpy as np

from . import diagnostics, paths
from .config import Settings, validate
from .detect import dates as dates_detect
from .discover import check_dirs, default_report_dir, discover
from .errors import ColumnRefusal, DatanonymousError, GenerationError, NotTabular, UsageError
from .generate import numeric as numeric_gen
from .generate import pii, pseudonyms
from .generate.ids import IdRegistry
from .generate.rng import stream
from .generate.table import GenContext, generate_table
from .io import registry
from .io.registry import OutTable, ReadRequest, WriteRequest
from .model import TYPE_INFO, DiscoveredFile, FileResult, Message, RunResult, SkippedFile, SyntheticTable
from .privacy import digits as real_digits
from .privacy import headers as names_screen
from .privacy import leaks, matches
from .privacy import realids as real_ids
from .privacy import texts as real_texts
from .profile import entity as entity_mod
from .profile import structure as structure_mod
from .profile.codes import codes_of, counts_of, has_nul, subset_codes
from .profile.stats import column_stats, has_numeric_view, is_missing, numeric_view, numeric_view_aligned
from .profile.table import finish_copy_gate, finish_real_ids, profile_table, unrecognised_keys_failure
from .report import docs, fidelity, selfcheck
from .report import report as report_mod

TMP_SUFFIX = ".datanonymous-tmp"
_REREAD_CELL_BUDGET = 50_000_000


def _keep_reread(fr: FileResult, rr, path: Path, files: list[FileResult],
                 expected_key: tuple[int, int, int] | None) -> None:  # noqa: ANN001
    """Keep a verified disk parse within the run's total cell budget."""
    fr.clear_reread()
    cells = sum(table.n_rows * len(table.columns) for table in rr.tables)
    if expected_key is None or cells + sum(file._reread_cells for file in files) > _REREAD_CELL_BUDGET:
        return
    try:
        key = leaks.reread_key(path)
    except OSError:
        return  # the scan re-reads (and fails conservatively) if the file is unavailable
    if key != expected_key:
        return
    fr._reread, fr._reread_key, fr._reread_cells = rr, key, cells


def _internal_failure(fr: FileResult, run: RunResult, exc: BaseException, stage: str) -> None:
    # Never print the exception's text anywhere (report, console, -v traceback): its message, arguments, notes and
    # chained exceptions may quote a real value, and a read error comes before any index could redact it (R7-04)
    fr.fail(f"internal error during {stage} ({diagnostics.error_place(exc)})",
            "re-run with -v to see the traceback, and report it", code="internal_error")
    run.debug.append(f"{fr.rel_path}: " + diagnostics.safe_traceback(exc))


def profile_file(f: DiscoveredFile, settings: Settings, run: RunResult,
                 key_refs: dict[int, dict[int, dict]] | None = None) -> FileResult | None:
    """Pass 1 for one file: read and profile every table.

    Returns None (and lists the file in ``run.skipped``) when the content is not
    a table (:class:`~datanonymous.errors.NotTabular`). A
    :class:`~datanonymous.errors.UsageError` (contradictory or invalid column
    flags) propagates: the run stops with exit code 2 before anything is written.
    ``key_refs`` (table number -> column index -> ``{'of', 'share'}``) names the
    columns to profile as ID columns because they hold another ID column's IDs
    (:func:`plan_key_refs`).
    """
    fr = FileResult(rel_path=f.rel_path, out_rel_path=f.out_rel_path, format=f.format, messages=list(f.notes))
    if f.preflight_error:
        fr.fail(*f.preflight_error)
        return fr
    try:
        handler = registry.get(f.format)
        rr = handler.read(ReadRequest(f.path, f.rel_path, f.format, f.compression, settings))
        fr.layout = rr.layout
        fr.messages.extend(rr.messages)
        for t_i, t in enumerate(rr.tables):
            fr.tables.append(profile_table(t, f.rel_path, f.format, settings, (key_refs or {}).get(t_i),
                                           defer_d35=True, auto_null=f.auto_null.get(t_i)))
            if settings.mode == "sharing":
                from .profile.sharing import record_evidence

                record_evidence(fr.tables[-1], t, settings)
            fr.source_dtypes.append(handler.inferred_dtypes(f.path, rr.layout, t.name))
    except UsageError:
        raise
    except NotTabular as e:
        run.skipped.append(SkippedFile(f.rel_path, e.message, code="skipped_not_a_table"))
        return None
    except DatanonymousError as e:
        fr.tables = []
        fr.fail(e.message, e.fix, e.code, e.column)
    except Exception as e:  # noqa: BLE001 - per-file isolation (D19)
        fr.tables = []
        _internal_failure(fr, run, e, "reading/profiling")
    return fr


def plan_key_refs(run: RunResult, files: dict[str, DiscoveredFile], settings: Settings) -> list[Message]:
    """Run-level structure (``structure.cross_file_structure``), with the columns that hold an ID column's IDs
    profiled as ID columns (R2-01; D5/D8/D10). Returns the run-level messages.

    A column found to hold the real IDs of an ID column (``params['_key_ref']``) would write them as they are:
    its file is read and profiled again with that column as an ID column (synthetic IDs; the ID column's
    mapping when >= 99% of its keys are found there), and the run-level structure is recomputed, until no new
    column is found (promotion only ever adds ID columns, so this ends). A column still marked afterwards
    fails its file (D19): its real IDs are never written. So does a multi-valued column whose items are real
    IDs (``params['_key_ref_list']``: one cell is not one ID), with the fix to add.
    """
    msgs = structure_mod.cross_file_structure(run.files, settings)
    key_refs: dict[str, dict[int, dict[int, dict]]] = {}
    while True:
        changed: set[str] = set()
        for fr in run.files:
            if fr.failed or fr.rel_path not in files:
                continue
            for t_i, tp in enumerate(fr.tables):
                done = key_refs.get(fr.rel_path, {}).get(t_i, {})
                for p in tp.columns:
                    ref = p.params.get("_key_ref")
                    if ref and p.index not in done:
                        key_refs.setdefault(fr.rel_path, {}).setdefault(t_i, {})[p.index] = dict(ref)
                        changed.add(fr.rel_path)
        if not changed:
            break
        for k, fr in enumerate(run.files):
            if fr.rel_path in changed:
                again = profile_file(files[fr.rel_path], settings, run, key_refs=key_refs[fr.rel_path])
                if again is not None:
                    run.files[k] = again
        msgs = structure_mod.cross_file_structure(run.files, settings)
    for fr in run.files:
        stuck = sorted({structure_mod.table_ref(fr.rel_path, tp.name, p.name)
                        for tp in fr.tables for p in tp.columns if p.params.get("_key_ref")})
        lists = sorted({structure_mod.table_ref(fr.rel_path, tp.name, p.name)
                        for tp in fr.tables for p in tp.columns if p.params.get("_key_ref_list")})
        if stuck and not fr.failed:
            fr.fail(f"column(s) {', '.join(stuck)} hold the real IDs of an ID column but could not be treated as "
                    "ID columns; the file is not written", "--id-col " + " --id-col ".join(stuck),
                    code="unlinked_key_column")
        if lists and not fr.failed:
            fr.fail(f"column(s) {', '.join(lists)} hold lists of the real IDs of an ID column, which would be "
                    "written as they are; the file is not written",
                    " ".join(f"--force-text {c}" for c in lists) + " (invented labels), or --null / --exclude",
                    code="unlinked_key_column")
    return msgs


def fail_unrecognised_keys(files: list[FileResult]) -> None:
    """D38 (user, 2026-10-11): a file with a table that has no recognised ID column but a column of repeated keys of 8
    or more digits under a header that names no measure, count or flag (D7 candidate rule 7) FAILs (flag and skip,
    D19): its real keys may be people's IDs. The fix names the flags that settle it (``--person-id`` / ``--id-col``
    for a person ID, ``--org-code`` for a school or district code); ``profile.table.unrecognised_key_columns``. Run
    after the key discovery (``plan_key_refs``), which may make such a column an ID column, and before the run's real
    IDs are planned. The FAILed file's tables stay (its people are real people too)."""
    for fr in files:
        if fr.failed:
            continue
        failure = unrecognised_keys_failure(fr.rel_path, fr.tables)
        if failure is not None:
            fr.fail(*failure)


def plan_ids(files: list[FileResult], seed: int, settings: Settings | None = None,
             texts: real_texts.RealTexts | None = None) -> tuple[IdRegistry, dict[tuple[str, int], dict[int, str]]]:
    """Register every ID column of every profiled file and build the run-level mappings.

    No synthetic ID is, or takes the local part or username of, a real e-mail address held anywhere in the run
    (R8-03), and no synthetic ID equals any real text of the run that the run does not write (R9-03: a pseudonymised
    label, a name, free text, a value of an excluded or nulled column, a title ...): the run's real-text registry
    (``texts``, else ``privacy.leaks.real_texts_of``: every column's values the run does not write verbatim, excluded
    and nulled columns, e-mail addresses inside pseudonymised labels and free text, the text outside the tables, and
    the values it does write) is read before any ID is drawn, and every candidate ID is refused that it holds
    (``privacy.texts.RealTexts.holds``, the leak scan's own test of ID columns)."""
    reg = IdRegistry(id_check=settings.id_check if settings is not None else None,
                     min_cell=settings.min_cell if settings is not None else 10)
    namespaces: dict[tuple[str, int], dict[int, str]] = {}
    for fr in files:
        if fr.failed:
            continue
        for t_i, tp in enumerate(fr.tables):
            nsmap: dict[int, str] = {}
            for i in tp.entity.id_columns:
                key = entity_mod.id_namespace(fr.rel_path, tp.name, tp.columns[i])
                reg.register(key, fr.rel_path, tp.name, i, tp.columns[i].name, tp.private.id_raw[i])
                nsmap[i] = key
            namespaces[(fr.rel_path, t_i)] = nsmap
    # R5-02: no synthetic ID is a real address of the run's e-mail columns or takes one's local part or username,
    # nor equals, or takes the username of, a real ID of a file that FAILed before planning (a FAILed file's people
    # are real people too; the leak scan reads every file); R6-02: nor a real ID of a column that a flag turned
    # from an ID column into something else (--exclude, --null, --force-*: entity_mod.shadow_namespaces)
    avoid_ids = {v for fr in files if fr.failed for tp in fr.tables for raw in tp.private.id_raw.values()
                 for v in np.asarray(raw, dtype=object).tolist() if isinstance(v, str)}
    for ns in entity_mod.shadow_namespaces(files):
        avoid_ids |= ns.raw_values
    # R8-03: every real e-mail address of the run, wherever it is held (an EMAIL column, an excluded or nulled column,
    # a pseudonymised label, free text, a title ...), kept or not: one source of truth with the leak scan's registry
    if texts is None:
        texts = leaks.real_texts_of(files)
    emails = real_texts.email_addresses(texts)
    # R9-03: nor any other real text of the run that it does not write (the whole registry, not only its e-mails)
    reg.build(seed, avoid_emails=sorted(emails), avoid_ids=avoid_ids, avoid_texts=texts)
    for fr in files:
        if fr.failed:
            continue
        for t_i, tp in enumerate(fr.tables):
            for i, key in namespaces.get((fr.rel_path, t_i), {}).items():
                ns = reg.namespaces[key]
                p = tp.columns[i]
                # identity is decided per namespace (D8/D9): child files of a 1..n row-number key keep it too
                p.params["_identity_mapped"] = ns.identity
                p.params["_id_prefix"] = ns.prefix  # PRIVATE: the kept prefix (the leak scan's username rule)
                p.params["id_mapping"] = ("kept: row numbers 1..n (D9)" if ns.identity
                                          else "synthetic: same shape, never a real ID (D9)" + (
                                              "; e-mail addresses get synthetic ones on reserved domains (D24)"
                                              if ns.email_keys else ""))
                for note in ns.notes:
                    tp.messages.append(Message("warning", "id_width_changed", note, column=p.name))
    for key, ns in reg.namespaces.items():
        if ns.error is None:
            continue
        members = {m[0] for m in ns.members}
        for fr in files:
            if fr.rel_path in members and not fr.failed:
                fr.fail(ns.error.message, ns.error.fix, ns.error.code, ns.error.column)
    return reg, namespaces


def _synthetic_numbers(values: np.ndarray, p, coded=None) -> np.ndarray | None:  # noqa: ANN001
    """Numeric view (``profile.stats.numeric_view``) of the non-missing synthetic values."""
    if not has_numeric_view(p):
        return None
    codes, uniques = codes_of(values) if coded is None else coded
    if has_nul(values):
        stripped = [v.strip() for v in values.tolist() if not is_missing(v, p.missing_policy, p.na_tokens)]
        return numeric_view(p, stripped)
    keep = np.asarray([not is_missing(v, p.missing_policy, p.na_tokens) for v in uniques], dtype=bool)
    numbers = numeric_view_aligned(p, [v.strip() for v in uniques[keep]])
    if numbers is None:
        return None
    aligned = np.full(len(uniques), np.nan)
    aligned[keep] = numbers
    expanded = aligned[codes]
    return expanded[np.isfinite(expanded)]


def _extremes_missing(values: np.ndarray, p, syn: np.ndarray | None = None,
                      uniques: np.ndarray | None = None) -> bool:  # noqa: ANN001
    """D14 self-check: the column's exact real min or max is absent from its written values.

    Checked on the rendered rows (numeric view ``syn`` when already computed) of every column with a numeric
    view, within-ID constants included (their fidelity is otherwise judged per entity), unless
    ``--no-exact-extremes``.
    """
    real = p.model.sorted_numeric if p.model is not None else None
    if p.role not in ("row", "entity") or p.params.get("_no_exact_extremes") or real is None or not len(real):
        return False
    lo_gone, hi_gone = p.params.get(real_ids.EXTREMES_KEY) or (False, False)
    if lo_gone or hi_gone:
        # R9-01: a real extreme that is a real person ID is never written (redrawn or pseudonymised; D14 yields)
        if lo_gone and hi_gone:
            return False
        if syn is None:
            syn = _synthetic_numbers(values, p)
        if syn is None:
            return False
        return not np.any(syn == (real[-1] if lo_gone else real[0]))
    # wide numbers (int64 from 2**53, 16+ significant digits): float64 rounds the real and the written extremes
    # alike, so they are compared exactly (R2-12)
    exact_values = values if has_nul(values) else codes_of(values)[1] if uniques is None else uniques
    exact = numeric_gen.exact_extremes_missing(
        p, exact_values,
        lambda raw: is_missing(raw, p.missing_policy, p.na_tokens))
    if exact is not None:
        return exact
    if syn is None:
        syn = _synthetic_numbers(values, p)
    if syn is None:
        return False
    return not (np.any(syn == real[0]) and np.any(syn == real[-1]))


def _written_values(st: SyntheticTable, out_table, pos: int, p) -> tuple[np.ndarray, int]:  # noqa: ANN001
    """The values a column's statistics and fidelity describe, and how many rows the writer stored differently.

    A native date/time column (Excel date cells, Arrow/Parquet/Feather temporal types, Stata %td/%tc) is
    judged on its re-read output: the writer converts the generated text into the native type (a timestamp
    into its column's time zone, R4-06), so the statistics describe what is stored. Every other column is
    stored as generated (text formats verbatim).
    """
    gen = st.columns[pos].values
    if (out_table is None or pos >= len(out_table.columns) or int(out_table.n_rows) != int(st.n_rows)
            or list(out_table.header) != list(st.header) or dates_detect.native_temporal(p.native) is None
            or TYPE_INFO[p.type].family != "date"):
        return gen, 0
    written = np.asarray(out_table.columns[pos].values, dtype=object)
    if len(written) != len(gen):
        return gen, 0

    def norm(v: object) -> object:
        return None if v is None or (isinstance(v, str) and not v.strip()) else str(v).strip()

    a = np.fromiter((norm(v) for v in gen.tolist()), dtype=object, count=len(gen))
    b = np.fromiter((norm(v) for v in written.tolist()), dtype=object, count=len(written))
    return written, int(np.count_nonzero(a != b))


def _first_rows(entity_index: np.ndarray | None, n_entities: int) -> np.ndarray | None:
    """Row index of each entity's first row (None without entities)."""
    if entity_index is None or n_entities == 0:
        return None
    first = np.full(n_entities, -1, dtype=np.int64)
    if not len(entity_index):
        return first
    # Negative labels historically use NumPy indexing: -1 aliases the last
    # entity. Normalize those aliases before selecting the first occurrence.
    labels = np.arange(n_entities)[entity_index] if np.any(entity_index < 0) else entity_index
    entities, rows = np.unique(labels, return_index=True)
    # Assign by entity label (including NumPy's negative-index behavior), rather
    # than returning indices in row order; absent entities retain the old -1.
    first[entities] = rows
    return first


def _request_null_after_refusal(fr: FileResult, settings: Settings, run: RunResult, t_i: int | None,
                                refusal: GenerationError) -> None:
    """ADR-0001, a refusal raised while table ``t_i`` of ``fr`` was generated (a :class:`ColumnRefusal`, or another
    :class:`GenerationError` that names one column): when its column is not a key, not in a structural relationship
    and not nulled already, record a null request (``RunResult.null_requests``), so ``execute`` profiles and
    generates the whole run again with the column nulled, as for a column-scoped finding. Otherwise (or with
    ``--refuse-unsupported``) nothing is recorded and the file is left out, as before."""
    if settings.mode != "sharing" or settings.refuse_unsupported or t_i is None or not refusal.column:
        return
    from .profile.sharing import catalogue_fix, key_or_related

    tp = fr.tables[t_i]
    if isinstance(refusal, ColumnRefusal):
        i, plain = refusal.index, refusal.plain
        if not (0 <= i < len(tp.columns)) or tp.columns[i].name != refusal.column:
            return
    else:
        hits = [p.index for p in tp.columns if p.name == refusal.column]
        if len(hits) != 1:
            return
        i, plain = hits[0], "its synthetic values could not be generated"
    if tp.columns[i].role in ("null", "excluded") or key_or_related(tp, i):
        return
    run.null_requests.setdefault(fr.rel_path, {}).setdefault(t_i, {})[i] = (plain, catalogue_fix(refusal.code))


def _request_nulls_for_findings(fr: FileResult, settings: Settings, run: RunResult,
                                findings: dict[int, list[Message]]) -> None:
    """ADR-0001, the blocking findings made while ``fr`` was generated ({table number: messages}: ``verify_values``
    errors, leak-scan hits): when every one is about one column that is not a key, not in a structural relationship
    and not nulled already (``profile.sharing.generation_null_targets``), record null requests for those columns, so
    ``execute`` generates the whole run again with them nulled. Otherwise (or with ``--refuse-unsupported``) nothing
    is recorded and the file is left out, as before."""
    if settings.mode != "sharing" or settings.refuse_unsupported or not findings:
        return
    from .profile.sharing import generation_null_targets

    targets = {t_i: generation_null_targets(fr.tables[t_i], messages) for t_i, messages in findings.items()}
    if any(columns is None for columns in targets.values()):
        return
    for t_i, columns in targets.items():
        run.null_requests.setdefault(fr.rel_path, {}).setdefault(t_i, {}).update(columns)


def generate_file(fr: FileResult, settings: Settings, run: RunResult, reg: IdRegistry,
                  namespaces: dict[tuple[str, int], dict[int, str]]) -> docs.WrittenFile | None:
    """Pass 2 for one file: generate, protect, write, self-check. Returns docs input or None.

    A :class:`ColumnRefusal` (or another :class:`GenerationError` naming one column) raised while a table is
    generated, a ``verify_values`` error or a leak-scan hit leaves the file out of this attempt; in sharing mode a
    non-key column they are all about is requested nulled for the next one (:func:`_request_null_after_refusal`,
    :func:`_request_nulls_for_findings`)."""
    assert run.out_dir is not None and fr.layout is not None
    generating: int | None = None
    findings: dict[int, list[Message]] = {}  # blocking findings made while values were generated, by table
    dest = run.out_dir / Path(*fr.out_rel_path.split("/"))
    tmp = dest.with_name(dest.name + TMP_SUFFIX)
    try:
        handler = registry.get(fr.format)
        out_handler = registry.output_handler(fr.format)
        synth: list[SyntheticTable] = []
        for t_i, tp in enumerate(fr.tables):
            generating = t_i
            ctx = GenContext(run.seed, fr.rel_path, reg, namespaces.get((fr.rel_path, t_i), {}), settings)
            generation_profile = tp
            if settings.mode == "sharing":
                from .profile.sharing import CHANGES_KEY, change_records, protect_profile

                generation_profile = protect_profile(tp, settings)
                tp.private.extra[CHANGES_KEY] = change_records(generation_profile)
                # the deliberate changes protection made (ADR-0004 broader classes) are reported with the table
                tp.messages.extend(m for m in generation_profile.messages[len(tp.messages):]
                                   if m.level == "info" and m not in tp.messages)
            st = generate_table(generation_profile, ctx)
            generated_profile = st.profile
            priv = matches.enforce(generated_profile, st, stream(run.seed, fr.rel_path, tp.name or "", "__privacy__"),
                                   settings.max_row_matches)
            fr.privacy.append(priv)
            st.messages.extend(structure_mod.finalize_rows(
                generated_profile, st, lambda purpose, _t=tp.name or "": stream(run.seed, fr.rel_path, _t, "__finalize__", purpose),
                settings))
            tp.messages.extend(st.messages)
            if settings.mode == "sharing":
                from .profile.sharing import verify_values

                for column in st.columns:
                    for message in verify_values(st.profile.columns[column.index], column.values):
                        findings.setdefault(t_i, []).append(message)
                        fr.fail(message.text, message.fix, message.code, message.column)
            for column in st.columns:
                # Every generation/privacy repair has finished. Writers render
                # copies and never mutate these final synthetic column arrays.
                column.values.flags.writeable = False
            for m in leaks.scan_table(tp, st):
                findings.setdefault(t_i, []).append(m)
                fr.fail(m.text + (f" (column {m.column})" if m.column else ""), m.fix, m.code, m.column)
            synth.append(st)
        generating = None
        if fr.failed:
            _request_nulls_for_findings(fr, settings, run, findings)
            return None
        out_tables = [OutTable(st.name, st.header, [c.values for c in st.columns], [c.index for c in st.columns],
                               st.n_rows, dict(st.profile.layout), st.profile) for st in synth]
        dest.parent.mkdir(parents=True, exist_ok=True)
        paths.remove_entry(tmp)  # a link or hard link left at the temporary name is never written through
        # what a writer generates (title, notes and footer placeholders) avoids the run's real texts, phone numbers
        # and SSNs like the generators do (R8-04, R8-06): the run's registries are active while the file is written
        plan = next(((tp.private.extra.get("run_digit_keys"), leaks.run_texts(tp)) for tp in fr.tables), (None, None))
        with real_digits.guard(plan[0]), real_texts.guard(plan[1]):
            handler.write(WriteRequest(tmp, fr.layout, out_tables, settings, seed=run.seed,
                                       rng_for=lambda purpose: stream(run.seed, fr.rel_path, "__writer__", purpose)))
        # re-read the way the source was read: its encoding and, for delimited text, its delimiter (a one-column
        # output cannot show it, and synthetic values must not make a sniffer guess another one: R6-06)
        dialect = fr.layout.text
        check_settings = dataclasses.replace(settings, encoding=fr.layout.encoding,
                                             delimiter=dialect.delimiter if dialect is not None else settings.delimiter)
        rr_key = leaks.reread_key(tmp)
        rr = out_handler.read(ReadRequest(tmp, fr.out_rel_path, out_handler.key, fr.layout.compression,
                                          check_settings, purpose="selfcheck"))
        if leaks.reread_key(tmp) != rr_key:
            rr_key = None
        issues = selfcheck.compare(fr.layout, rr.layout, synth, rr.tables, out_handler.layout_diff)
        fr.selfcheck = issues
        if issues and not settings.no_strict:
            tmp.unlink(missing_ok=True)
            fr.fail("self-check: the written file does not have the source layout: " + "; ".join(issues[:5]),
                    "--no-strict writes it anyway (layout differences become warnings)")
            return None
        for issue in issues:
            fr.messages.append(Message("warning", "layout_mismatch", issue))
        os.replace(tmp, dest)
        fr.written = True
        fr.out_layout = rr.layout
        _keep_reread(fr, rr, dest, run.files, rr_key)
        dtypes: list[list[str] | None] = []
        for t_i, st in enumerate(synth):
            stats_map = {}
            fid_map = {}
            tp = st.profile
            first_rows = _first_rows(tp.private.entity_index, tp.entity.n_entities)
            block_share, n_eff = 0.0, None
            if first_rows is not None and st.n_rows:
                blocks = np.bincount(tp.private.entity_index).astype(float)
                block_share = float(blocks.max()) / st.n_rows
                n_eff = float(blocks.sum()) ** 2 / float((blocks ** 2).sum())  # Kish effective number of IDs
            out_table = rr.tables[t_i] if t_i < len(rr.tables) else None
            for pos, c in enumerate(st.columns):
                p = tp.columns[c.index]
                source_profile = fr.tables[t_i].columns[c.index]
                values, changed = _written_values(st, out_table, pos, p)
                if settings.mode == "sharing":
                    from .profile.sharing import verify_values

                    for message in verify_values(p, values):
                        findings.setdefault(t_i, []).append(message)
                        fr.fail(message.text, message.fix, message.code, message.column)
                    if fr.failed:
                        dest.unlink(missing_ok=True)
                        _request_nulls_for_findings(fr, settings, run, findings)
                        return None
                coded = codes_of(values)
                counts = counts_of(values)
                if changed:
                    fr.messages.append(Message(
                        "warning", "written_values_differ",
                        f"{changed} value(s) were stored differently from the generated values (the format "
                        "converted them); the statistics and fidelity describe the stored values", column=p.name))
                stats_map[c.index] = column_stats(values, p, settings.min_cell, counts=counts, n=len(values))
                docs.note_pseudonyms(p, values, counts=counts)  # D35: written synthetic labels
                nums = None
                if p.role == "entity" and first_rows is not None:
                    real_v = tp.private.extra["entity_values"][c.index]
                    syn_v = values[first_rows]
                    syn_coded = subset_codes(values, first_rows)
                    if has_nul(values):
                        syn_counts = counts_of(syn_v)
                    else:
                        frequencies = np.bincount(syn_coded[0], minlength=len(syn_coded[1]))
                        syn_counts = dict(zip(syn_coded[1], map(int, frequencies)))
                    emodel = fr.tables[t_i].private.entity_models[c.index]
                    ent_fid = fidelity.compare_column(
                        p, column_stats(real_v, p, settings.min_cell, counts=counts_of(real_v)),
                        column_stats(syn_v, p, settings.min_cell, counts=syn_counts, n=len(syn_v)),
                        emodel.sorted_numeric,
                        _synthetic_numbers(syn_v, p, syn_coded) if emodel.sorted_numeric is not None else None,
                        basis="entities")
                    # the written rows (each ID's value on all its rows) must match per row too: missing rate,
                    # level shares (R2-10) and, for numbers and dates, the row-level KS (R3-07)
                    real_rows = source_profile.model.sorted_numeric if source_profile.model is not None else None
                    nums = _synthetic_numbers(values, p, coded) if real_rows is not None else None
                    rows_fid = fidelity.compare_column(p, p.stats, stats_map[c.index], real_rows, nums,
                                                       block_share=block_share, n_eff=n_eff)
                    fid_map[c.index] = fidelity.with_row_check(ent_fid, rows_fid)
                    if rows_fid["verdict"] == "drift":
                        fr.messages.append(Message(
                            "warning", "fidelity_drift",
                            "within-ID constant: expanded to its ID's rows, the synthetic column drifts from the "
                            f"source ({', '.join(ch['check'] for ch in rows_fid['checks'] if not ch['ok'])}, per "
                            "row; a value or blank pattern held by fewer than --min-cell IDs, or a declared rule "
                            "on the column, is not matched to the sizes of their ID blocks)", column=p.name))
                    own = ent_fid  # the per-entity drift gets the general message below
                else:
                    real_nums = source_profile.model.sorted_numeric if source_profile.model is not None else None
                    nums = _synthetic_numbers(values, p, coded) if real_nums is not None else None
                    fid_map[c.index] = own = fidelity.compare_column(p, p.stats, stats_map[c.index], real_nums, nums)
                if _extremes_missing(values, p, nums, coded[1]):
                    fr.messages.append(Message("warning", "extremes_not_emitted",
                                               "the exact source min or max does not appear in the synthetic "
                                               "column (D14)", column=p.name))
                if own["verdict"] == "drift":
                    fr.messages.append(Message("warning", "fidelity_drift",
                                               "synthetic distribution drifts from the source "
                                               f"({', '.join(ch['check'] for ch in own['checks'] if not ch['ok'])})",
                                               column=p.name))
            fr.synthetic_stats.append(stats_map)
            fr.fidelity.append(fid_map)
            dt = out_handler.inferred_dtypes(dest, rr.layout, st.name)
            dtypes.append(dt)
            fr.out_dtypes.append(dt)
            src_dt = fr.source_dtypes[t_i] if t_i < len(fr.source_dtypes) else None
            for column, text in selfcheck.dtype_drift(src_dt, dt, st.header):
                fr.messages.append(Message("warning", "dtype_mismatch", text, column=column))
        return docs.WrittenFile(fr, synth, fr.synthetic_stats, dtypes)
    except GenerationError as e:
        tmp.unlink(missing_ok=True)
        if fr.written:
            dest.unlink(missing_ok=True)
        _request_null_after_refusal(fr, settings, run, generating, e)
        fr.fail(e.message, e.fix, e.code, e.column)
    except DatanonymousError as e:
        tmp.unlink(missing_ok=True)
        if fr.written:  # D19: a file that FAILs after its output was moved into place is not written either
            dest.unlink(missing_ok=True)
        fr.fail(e.message, e.fix, e.code, e.column)
    except Exception as e:  # noqa: BLE001 - per-file isolation (D19)
        tmp.unlink(missing_ok=True)
        if fr.written:
            dest.unlink(missing_ok=True)
        _internal_failure(fr, run, e, "generation/writing")
    return None


def _execute(settings: Settings, input_path: Path, out_dir: Path | None = None, report_dir: Path | None = None,
             mode: str = "run", *, _dirs_checked: bool = False,
             _left_out: dict[str, FileResult] | None = None,
             _auto_null: dict[str, dict[int, dict[int, tuple[str, str | None]]]] | None = None) -> RunResult:
    """Run (``mode='run'``) or inspect (``mode='inspect'``) and return the RunResult.

    ``_left_out`` (sharing): files an earlier attempt planned but could not write; they FAIL right after
    profiling, with the earlier reason, so every plan is made over the files written. ``_auto_null`` (sharing,
    ADR-0001): the columns earlier attempts found column-scoped blocking findings about after generation
    (``RunResult.null_requests``); they are profiled as Nulled columns, so every plan sees them as nulled.
    """
    input_path = Path(input_path)
    if mode == "run":
        if out_dir is None:
            raise DatanonymousError("--out-dir is required")
        out_dir = Path(out_dir)
        report_dir = Path(report_dir) if report_dir is not None else default_report_dir(out_dir)
        if not input_path.exists():
            raise UsageError(f"input not found: {input_path}")
        if not _dirs_checked:
            check_dirs(input_path, out_dir, report_dir, settings.force)
    disc = discover(input_path, settings,
                    avoid=(("--out-dir", out_dir), ("--report-dir", report_dir)) if mode == "run" else ())
    if settings.seed is not None:
        seed, was_random = settings.seed, False
    else:
        seed, was_random = secrets.randbelow(2**32), True
    run = RunResult(mode=mode, input=input_path, out_dir=out_dir if mode == "run" else None,
                    report_dir=report_dir if mode == "run" else None, seed=seed, seed_was_random=was_random,
                    skipped=list(disc.skipped), messages=list(disc.messages), data_mode=settings.mode)
    for f in disc.files:
        f.auto_null = (_auto_null or {}).get(f.rel_path, {})
        fr = profile_file(f, settings, run)
        if fr is not None:
            if _left_out and fr.rel_path in _left_out and not fr.failed:
                earlier = _left_out[fr.rel_path]
                reason = earlier.reasons[0] if earlier.reasons else "not written in an earlier attempt"
                fr.fail(reason, earlier.fixes[0] if earlier.fixes else None,
                        next((m.code for m in earlier.messages if m.level == "error" and m.text == reason), None))
            run.files.append(fr)
    if disc.files and not run.files:
        run.messages.append(Message("warning", "no_tabular_files", "no supported tabular files found in the input"))
    for ref in settings.unused_refs():
        run.messages.append(Message("warning", "unused_column_flag", f"column reference {ref!r} matched no column"))
    run.messages.extend(plan_key_refs(run, {f.rel_path: f for f in disc.files}, settings))
    fail_unrecognised_keys(run.files)  # D38: once the key discovery has made every ID column it can
    if settings.mode == "sharing" and mode == "run" and not settings.refuse_unsupported:
        from .profile.sharing import AUTO_NULL_ALTERNATIVE_KEY, AUTO_NULL_SOURCE, null_unprotectable_columns

        # ADR-0001: a non-key column sharing mode cannot protect is nulled the way --null does it: the run is profiled
        # again from the start with the column nulled (``_auto_null``), so every profile-time record (the row-match
        # keys, D35 holders, structure) and every run-wide plan (real texts, IDs, pseudonyms, population) sees it as
        # nulled. Nothing of this attempt is planned or written.
        for fr in run.files:
            if fr.failed:
                continue
            for t_i, tp in enumerate(fr.tables):
                null_unprotectable_columns(tp, settings)
                before = (_auto_null or {}).get(fr.rel_path, {}).get(t_i, {})
                for p in tp.columns:
                    if p.source == AUTO_NULL_SOURCE and p.index not in before:
                        reason = p.reason.split(" because ", 1)[1] if " because " in p.reason else p.reason
                        run.null_requests.setdefault(fr.rel_path, {}).setdefault(t_i, {})[p.index] = (
                            reason, p.params.get(AUTO_NULL_ALTERNATIVE_KEY))
        if run.null_requests:
            return run
    # R9-01: the run's real person IDs (every ID column of every file, FAILed and flagged ones too), which no value
    # written verbatim may equal, whatever the key discovery inferred
    ids = real_ids.plan(run.files)
    tables = [tp for fr in run.files for tp in fr.tables]
    # D35: every real text unit is decided over the whole run (the most people any file shows holding it); a real
    # person ID is never kept, and every other real unit that is one is pseudonymised too (numbers, NA tokens)
    finish_copy_gate(tables, settings.min_cell, ids)
    finish_real_ids(tables, settings.min_cell, ids)
    if mode == "inspect":
        run.messages.extend(names_screen.screen_run(run, settings, disc.root))  # D34 (scratch ID registry)
        for fr in run.files:
            fr.finalize_status()
        return run

    assert out_dir is not None and report_dir is not None
    # run-wide real texts the run never writes (any column, excluded and nulled ones too, and the text outside the
    # tables: titles, notes, dropped metadata): no synthetic ID takes one of their e-mail addresses or usernames
    # (R8-03), no generated value or title placeholder equals one, and the leak scan checks every output text column
    # and every ID column against them (R7-01, R7-02, R8-04). Read before any ID is drawn.
    keep_titles = bool(settings.keep_titles)
    texts = leaks.real_texts_of(run.files, keep_titles)
    failed = [fr.failed for fr in run.files]
    reg, namespaces = plan_ids(run.files, seed, settings, texts)
    # D34: no header, sheet, file or folder name the run writes holds personal data (needs the run's real IDs)
    run.messages.extend(names_screen.screen_run(run, settings, disc.root, reg))
    screen = names_screen.screen_of(run.messages)
    # a file that FAILed since writes nothing: the texts it would have written verbatim no longer excuse anything
    texts = leaks.plan_real_texts(run.files, keep_titles, texts if [fr.failed for fr in run.files] == failed else None)
    # run-wide real full names: no file writes one from another file (R1-04), nor a flagged name's person (D34), nor
    # one of the run's unwritten real texts (R7-01, R7-02)
    pii.plan_names(run.files, screen.name_word_lists() if screen is not None else (), texts)
    leaks.plan_digit_keys(run.files)  # run-wide real phones/SSNs: no generated value holds one, even by chance
    pseudonyms.plan(run.files, seed)  # D35: one pseudonym per real unit in the run, never a real value of it
    if settings.mode == "sharing":
        from .generate.population import PopulationRefusal, plan_population

        while True:
            try:
                plan_population(run.files, seed, settings, reg, namespaces)
            except PopulationRefusal as refusal:
                if not settings.all_or_nothing:
                    # leave the refused tables out and plan the others again (each pass leaves one out)
                    for fr in run.files:
                        error = next((e for (path, _), e in refusal.refused.items() if path == fr.rel_path), None)
                        if error is not None:
                            fr.fail(error.message, error.fix, error.code, error.column)
                    continue
                error = refusal
            except DatanonymousError as caught:  # a refusal no table answers for: nothing is planned
                error = caught
            else:
                for fr in run.files:
                    for t_i, tp in enumerate(fr.tables):
                        for i, key in namespaces.get((fr.rel_path, t_i), {}).items():
                            if tp.columns[i].params.get("_identity_mapped") and not reg.namespaces[key].identity:
                                # row numbers whose roster was left out: real IDs the leak scan checks (every file)
                                tp.columns[i].params["_identity_mapped"] = False
                                tp.columns[i].params["id_mapping"] = "synthetic: same shape, never a real ID (D9)"
                        if tp.private.extra.get("_sharing_population") is not None:
                            tp.messages = [message for message in tp.messages if message.code != "id_columns_unrelated"]
                            for column in tp.columns:
                                column.messages = [message for message in column.messages if message.code != "id_columns_unrelated"]
                break
            for fr in run.files:
                if not fr.failed:
                    fr.fail(error.message, error.fix, error.code, error.column)
            break
    planned = {fr.rel_path for fr in run.files if not fr.failed}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[docs.WrittenFile] = []
    for fr in run.files:
        if fr.failed:
            continue
        w = generate_file(fr, settings, run, reg, namespaces)
        if w is not None:
            written.append(w)
    if settings.mode == "sharing" and not settings.refuse_unsupported:
        from .privacy.sharing_policy import finding_category
        from .profile.sharing import column_scoped_findings

        # ADR-0001: a written file whose only blocking findings are about non-key columns is generated again (same
        # seed) with those columns nulled; any other finding still withholds it (below, and in package.finish)
        for fr in run.files:
            if fr.written and not fr.withheld:
                scoped = column_scoped_findings(fr, finding_category)
                if scoped:
                    run.null_requests[fr.rel_path] = scoped
    if settings.mode == "sharing" and not settings.all_or_nothing:
        from .privacy import package

        # a file whose findings block sharing is left out before the docs describe the output
        written = package.withhold_blocked(run, out_dir, written)
    run.late_failures = [fr for fr in run.files if fr.failed and fr.rel_path in planned]
    stale_files = stale_outputs(run.files, out_dir)
    for fr in stale_files:
        run.messages.append(Message(
            "warning", "stale_output",
            f"{fr.out_rel_path}: the out dir still holds a copy from an earlier run; it was NOT regenerated "
            "because its source FAILed in this run (not deleted)"
            + ("; its name or headers hold personal data (D34), so delete it before the out dir is shared"
               if fr.withheld else "")))
    stale = [fr.out_rel_path for fr in stale_files]  # shown paths: a flagged name is redacted (D34)
    run.docs = docs.write_docs(out_dir, written, run.messages, seed, settings.min_cell, stale, mode=settings.mode,
                               not_written=docs.not_written_count(run))
    for fr in run.files:
        fr.finalize_status()
    payload = report_mod.build_report(run, settings)
    leak_messages = leaks.scan_outputs(run, payload)
    if leak_messages:
        run.messages.extend(leak_messages)
        payload = report_mod.build_report(run, settings)
    report_mod.write_report_files(report_dir, payload)
    return run


def execute(settings: Settings, input_path: Path, out_dir: Path | None = None, report_dir: Path | None = None,
            mode: str = "run") -> RunResult:
    """Sharing writes the verified files to the output folder and leaves out (and lists in the report folder's
    sharing-failures.csv) the ones that cannot be written safely; fidelity retains local workflow."""
    validate(settings)
    if mode == "run":
        from . import rundirs

        out_dir, report_dir = rundirs.resolve(settings, Path(input_path), out_dir, report_dir)
        if out_dir is None:
            raise UsageError("--out-dir is required",
                             fix="pass --out-dir, or set output_root and report_root in a config file")
    if settings.mode != "sharing" or mode != "run":
        return _execute(settings, input_path, out_dir, report_dir, mode)
    from .privacy import package

    destination = Path(out_dir)
    local_report = Path(report_dir) if report_dir is not None else default_report_dir(destination)
    if not Path(input_path).exists():
        raise UsageError(f"input not found: {input_path}")
    check_dirs(Path(input_path), destination, local_report, False, sharing=True)
    package.check_fresh_destination(destination)
    left_out: dict[str, FileResult] = {}
    auto_null: dict[str, dict[int, dict[int, tuple[str, str | None]]]] = {}
    was_random = False
    while True:
        stage = package.create_staging(local_report)
        run = _execute(settings, input_path, stage, local_report, mode, _dirs_checked=True, _left_out=left_out,
                       _auto_null=auto_null)
        requests = run.null_requests
        if not requests and (settings.all_or_nothing or not run.late_failures):
            break
        shutil.rmtree(stage)
        # ADR-0001: columns whose blocking findings were their file's only ones are nulled and the whole run planned
        # and written again (same seed), so every run-wide plan sees them as nulled. Each attempt nulls columns not
        # nulled before (a finding on a column already nulled withholds its file), so the attempts end.
        from .profile.sharing import with_lost_links

        by_path = {fr.rel_path: fr for fr in run.files}
        for path, tables in requests.items():
            for t_i, columns in tables.items():
                # the relationships inside the table each nulled column loses, read before the next attempt's profile
                # finds none of them (ADR-0001 amendment)
                tp = by_path[path].tables[t_i] if path in by_path and t_i < len(by_path[path].tables) else None
                auto_null.setdefault(path, {}).setdefault(t_i, {}).update(
                    (i, with_lost_links(tp, i, request) if tp is not None and i < len(tp.columns) else request)
                    for i, request in columns.items())
        # a file left out after planning: plan and write the others again without it (same seed), so no plan
        # (the row numbers 1..n kept, D9; the real texts it would have written) assumes it was written
        if not settings.all_or_nothing:
            left_out.update((fr.rel_path, fr) for fr in run.late_failures if fr.rel_path not in requests)
        if settings.seed is None:
            settings = dataclasses.replace(settings, seed=run.seed)
            was_random = True
    for fr in run.files:
        # the report keeps what the attempt that generated a file left out found (its reasons and warnings); its
        # profiles stay this run's, planned over the files written
        earlier = left_out.get(fr.rel_path)
        if earlier is not None:
            fr.reasons, fr.fixes, fr.messages = earlier.reasons, earlier.fixes, earlier.messages
            for tp, earlier_tp in zip(fr.tables, earlier.tables):
                tp.messages = earlier_tp.messages
    run.seed_was_random = run.seed_was_random or was_random
    try:
        run.package = package.finish(run, destination, stage, settings.all_or_nothing, settings)
    except (OSError, UsageError):
        run.package = {"mode": "sharing", "released": False, "blocking_codes": ["sharing_promotion"]}
        run.messages.append(Message("error", "sharing_promotion", "the output could not be moved into the output folder, so nothing was written to it",
                                    fix="choose a new or empty output folder on the same drive as the report folder"))
        for fr in run.files:
            fr.fail("not written: the output could not be moved into the output folder", code="sharing_promotion")
        run.docs = {}
    run.out_dir = destination
    report_mod.write_report_files(local_report, report_mod.build_report(run, settings))
    return run


def stale_outputs(files: list[FileResult], out_dir: Path) -> list[FileResult]:
    """FAILed files whose output path in the out dir still holds a copy from an earlier run (found at the real
    path, ``FileResult.out_rel_on_disk``, even when the shown path is redacted, D34)."""
    written_now = {fr.out_rel_on_disk.casefold() for fr in files if fr.written}
    return [fr for fr in files
            if fr.failed and fr.out_rel_on_disk.casefold() not in written_now
            and (out_dir / Path(*fr.out_rel_on_disk.split("/"))).is_file()]


def exit_code(run: RunResult) -> int:
    """0 = every tabular file written (PASS or WARN); 3 = at least one FAIL (design 0.2).

    ``inspect`` uses the same rule: 3 when a file would FAIL (it cannot be read/profiled).
    """
    if run.any_failed or any(m.level == "error" for m in run.messages):
        return 3
    return 0


__all__ = ["execute", "exit_code", "profile_file", "generate_file", "plan_key_refs", "fail_unrecognised_keys",
           "plan_ids"]
