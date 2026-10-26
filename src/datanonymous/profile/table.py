"""Pass-1 profiling of one raw table (FOUNDATION).

For every column: resolve its options, run the ordered detection pipeline,
build the PRIVATE generation model and the public statistics, and keep the
reader's native column details (``ColumnProfile.native``) for the writers. The
columns whose header and ID-like values may make them ID columns (K-02,
``detect.ids.mark_weak_id``: ``Student_SK``, ``Stu``, ``Admission No``) are
decided with the rest of the table (:func:`weak_id_columns`), an
``--exclude``d or ``--null``ed one too (harden 2: it stays a shadow ID).
Then
the entity structure (IDs, rows per ID, within-ID constants, with the D14 floor
applied to the per-entity models), the K-07 floor of far-from-median date
sentinels (:func:`floor_far_sentinels`: one written verbatim must be held by >=
``--min-cell`` entities, else the column is read again with it as a date), the
D35 count of the copy gate for every
real text unit a column writes verbatim (``detect.text.count_units``: the
distinct people holding it, the minimum over every ID column of the table, the
columns a flag turned from IDs into something else, and the names or e-mail
addresses of the row's own person; else its distinct rows;
``profile.entity.d35_keys``) and its decision (:func:`finish_copy_gate`: per
table here, or once over the whole run by the pipeline with ``defer_d35``), every
other real value a column would write verbatim that is a real person ID
pseudonymised too (:func:`finish_real_ids`, R9-01; the same per table or over the run), the
structure profile (missingness patterns
+ AREA D detectors), the ID-candidate warning (D7, for every table; a
candidate's value-revealing statistics are redacted) and the hashed real
comparison keys for privacy. An ``--exclude``d or ``--null``ed column gets no
model, but its distinct values are kept PRIVATE (:func:`dropped_values`): they
are real values of the run that no generator may write and the leak scan looks
for (``privacy.texts``, R7-02). The raw table can be dropped afterwards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..detect import text as text_detect
from ..detect.context import DetectContext, Detection
from ..detect.ids import (
    FLAGGED_KEYS_KEY,
    REPEATED_KEY_DIGITS,
    REPEATED_KEYS_KEY,
    id_candidate_reason,
    repeated_key_digits,
    repeated_keys_width,
)
from ..detect.types import detect_column, header_org_code
from ..model import (
    LABEL_SAFE_TYPES,
    NEVER_COPY_TYPES,
    ColumnModel,
    ColumnProfile,
    ColumnType,
    Message,
    RawTable,
    TableProfile,
)
from ..generate import pii
from ..generate.table import is_level_generated
from ..hints import DEFAULT_NA_TOKENS
from ..privacy import matches
from ..privacy import realids
from ..privacy import texts as real_texts
from . import entity as entity_mod
from . import structure as structure_mod
from .codes import codes_of
from .stats import (
    FLAG_NAMES,
    column_stats,
    flagged_keys_reason,
    missing_mask,
    numeric_view,
    numeric_view_aligned,
    redact_candidate,
    split_values,
)


#: Detection params that quote real values (made private for ID-like candidate columns).
_VALUE_PARAMS = ("min", "max", "value", "token", "levels", "tokens")
#: K-04: the fix for ID-like candidates. ``--id-col`` (synthetic IDs, counted as records by D35) is the advice for any
#: person or record ID; ``--person-id`` claims that the column numbers the row's own person, which an incident or
#: record number does not (each record would count as a person).
ID_FIX_EACH = ("--id-col COL for each person or record ID column (synthetic IDs); --person-id COL only for the one "
               "that numbers the row's own person")


def build_model(values: np.ndarray, p: ColumnProfile, counts: dict[Any, int] | None = None) -> ColumnModel:
    """PRIVATE generation model of ``values`` under profile ``p``.

    ``sorted_numeric`` holds the sorted :func:`~datanonymous.profile.stats.numeric_view`
    of the column (continuous numbers, date ordinals from AREA A, formatted
    numbers from AREA B), or None for types without a numeric view.
    """
    nonmissing, missing = split_values(values, p.missing_policy, p.na_tokens, counts)
    model = ColumnModel(nonmissing_counts=dict(nonmissing), missing_counts=dict(missing))
    t = p.type
    cnts = np.fromiter(nonmissing.values(), dtype=np.int64, count=len(nonmissing))
    nums = numeric_view(p, [raw.strip() for raw in nonmissing], cnts.tolist())
    if nums is not None:
        model.sorted_numeric = np.sort(nums)
    if t in NEVER_COPY_TYPES or t in (ColumnType.DIGIT_CODE, ColumnType.MULTI_SELECT, ColumnType.FREE_TEXT):
        model.real_values = frozenset(raw.strip() for raw in nonmissing)
        lens = np.fromiter((len(raw.strip()) for raw in nonmissing), dtype=float, count=len(nonmissing))
        model.lengths = np.sort(np.repeat(lens, cnts))
    return model


def _forced_identifier(force_type: str | None) -> bool:
    """``--force-type COL=IDENTIFIER`` (an unknown type name is reported where the type is applied)."""
    if not force_type:
        return False
    try:
        return ColumnType.parse(force_type) is ColumnType.IDENTIFIER
    except ValueError:
        return False


def _profile_from(det: Detection, index: int, name: str, na_tokens: frozenset[str]) -> ColumnProfile:
    return ColumnProfile(
        index=index, name=name, type=det.type, subtype=det.subtype, params=dict(det.params), reason=det.reason,
        source=det.source, ambiguous=det.ambiguous, ambiguity=det.ambiguity, missing_policy=det.policy(),
        na_tokens=na_tokens, messages=list(det.messages),
    )


def profile_table(raw: RawTable, file_rel: str, fmt: str, settings: Settings,
                  key_refs: dict[int, dict[str, Any]] | None = None, defer_d35: bool = False,
                  auto_null: dict[int, tuple[str, str | None]] | None = None) -> TableProfile:
    """Profile one table. ``key_refs`` (column index -> ``{'of', 'share'}``, from the run-level key discovery,
    R2-01) names columns found to hold another ID column's IDs: they are profiled as ID columns
    (``profile.entity.key_ref_detection``). ``defer_d35``: the D35 units are counted but not decided; the pipeline
    decides them over the whole run (:func:`finish_copy_gate`), so a value is judged by everyone the run shows
    holding it and gets one pseudonym everywhere. ``auto_null`` (column index -> (plain reason, alternative), sharing
    mode, ADR-0001) names the columns sharing mode nulls because a generated copy of the file had blocking findings
    about them: they are profiled exactly as ``--null`` would (every plan of the run sees them as nulled), and
    labelled as nulled automatically (``profile.sharing.mark_auto_nulled``)."""
    na_tokens = settings.na_tokens
    profiles: list[ColumnProfile] = []
    contexts: dict[int, tuple[DetectContext, Detection]] = {}
    dropped: dict[int, frozenset[str]] = {}
    dropped_phones: set[int] = set()
    dropped_marks: dict[int, dict[str, Any]] = {}
    for i, col in enumerate(raw.columns):
        opts = settings.column_options(file_rel, raw.name, col.name)
        auto = (auto_null or {}).get(i)
        if opts.exclude or opts.null or auto is not None:
            if (opts.pii_col or "").lower() == "phone" or (opts.force_type or "").upper() == "PHONE":
                dropped_phones.add(i)
            # R7-02: the column is not written, but its values are real people's data: no generator of the run may
            # write one, and the leak scan looks for them (privacy.texts; PRIVATE, never serialised); an --na-value
            # token is real text like any other (R8-05): only the default NA vocabulary is left out
            dropped[i] = dropped_values(col.values, frozenset(DEFAULT_NA_TOKENS))
            role = "excluded" if opts.exclude else "null"
            p = ColumnProfile(index=i, name=col.name, type=ColumnType.EMPTY, role=role, na_tokens=na_tokens,
                              source="exclude" if opts.exclude else "null",
                              reason="--exclude: column dropped" if opts.exclude else "--null: written as all-missing")
            if opts.exclude:
                p.messages.append(Message("warning", "column_excluded",
                                          "column dropped (--exclude): the output schema differs from the source",
                                          column=col.name))
            p.model = ColumnModel()
            p.stats = {"n": raw.n_rows}
            p.native = col.native
            if auto is not None and not (opts.exclude or opts.null):
                from .sharing import mark_auto_nulled

                mark_auto_nulled(p, *auto)
            flagged = ("person-id" if opts.person_id else "id-col" if opts.id_col
                       else "force-type" if _forced_identifier(opts.force_type) else None)
            if flagged:  # R6-02: an ID column by flag, excluded or nulled: its real IDs stay real IDs of the run
                p.params[entity_mod.FLAGGED_ID_KEY] = flagged
            else:  # A2-7: an ID column by its header and values (K-02) is one too; its table decides (weak_id_columns)
                dctx = DetectContext(index=i, name=col.name, values=col.values, file_rel=file_rel, table=raw.name,
                                     format=fmt, options=opts, settings=settings, native=col.native)
                mark = dropped_weak_mark(dctx)
                if mark:
                    dropped_marks[i] = mark
                # D38: dropped repeated wide keys (the flag that settles the FAIL) still tell D35 who holds a value
                # when the table has no ID of the row's own person (profile.entity.d35_keys)
                wide = repeated_keys_width(dctx)
                if wide:
                    p.params[REPEATED_KEYS_KEY] = wide
            profiles.append(p)
            continue
        ctx = DetectContext(index=i, name=col.name, values=col.values, file_rel=file_rel, table=raw.name, format=fmt,
                            options=opts, settings=settings, native=col.native)
        det = detect_column(ctx)
        if key_refs and i in key_refs:
            det = entity_mod.key_ref_detection(ctx, det, key_refs[i])
        profiles.append(_column_profile(det, ctx, col, settings))
        contexts[i] = (ctx, det)
    # K-02: a column whose header and ID-like values may make it an ID column is one when its table says so
    for i, det in weak_id_columns(raw, profiles, contexts, settings, dropped_marks).items():
        ctx = contexts[i][0]
        profiles[i] = _column_profile(det, ctx, raw.columns[i], settings)
        contexts[i] = (ctx, det)
    # harden 2 (A2-3): repeated wide keys under a header no rule knows ('NYSSIS'): an ID-like candidate (D7, below)
    # whose keys tell D35 who holds a value when the table has no ID of the row's own person (profile.entity.d35_keys)
    for i, (ctx, det) in contexts.items():
        if profiles[i].role != "id":
            wide = repeated_key_digits(ctx, det)
            if wide:
                profiles[i].params[REPEATED_KEYS_KEY] = wide
                if text_detect.nanp_phone_values(ctx.typed_values):  # D38's fix names --pii-col too
                    profiles[i].params[_REPEATED_PHONES_KEY] = True
            flagged_repeated_keys(profiles[i], ctx, wide)  # D38: a flag typed it: written, said aloud

    tp = TableProfile(name=raw.name, header=list(raw.header), n_rows=raw.n_rows, columns=profiles,
                      layout=dict(raw.layout))
    # entity structure (D4, D7, D8)
    ent, private = entity_mod.profile_entity(raw, profiles, settings)
    aliases, nesting = entity_mod.find_aliases_and_nesting(raw, profiles, ent)
    ent.aliases, ent.nesting = aliases, nesting
    tp.entity = ent
    tp.private.id_raw = private["id_raw"]
    tp.private.shadow_id_raw = private["shadow_id_raw"]
    tp.private.entity_index = private["entity_index"]
    tp.private.extra["entity_values"] = private["entity_values"]
    if dropped:
        tp.private.extra[real_texts.DROPPED_KEY] = dropped
        tp.private.extra["_dropped_phone_columns"] = frozenset(dropped_phones)
    # K-07: a far-from-median date sentinel is written verbatim only when >= --min-cell entities hold it
    holder_keys = floor_far_sentinels(raw, profiles, contexts, ent, settings)
    # D35, the copy gate's second stage: count the distinct people (or rows) holding each real text unit a column
    # would write verbatim; the keep-or-pseudonymise decision follows (here, or over the run: defer_d35)
    pending = [p for p in profiles if p.params.get(text_detect.PENDING_KEY)]
    zips = [p for p in profiles if p.type is ColumnType.ZIP and p.role not in ("excluded", "null")]
    if pending or zips:
        if holder_keys is None:  # (K-07 computed them when a column has far-from-median date sentinels)
            holder_keys = entity_mod.d35_keys(raw, profiles, ent, settings)
        # K-04/K-06: the ID columns whose subject role is only claimed; the run may still prove them
        # (entity.corroborated_tables, finish_copy_gate)
        tp.private.extra[entity_mod.CLAIMED_KEY] = list(holder_keys.claimed)
    for p in zips:  # K-08: the ZIP small-cell policy counts holders as D35 does (generate.pii._zip_map)
        text_detect.count_zip_holders(p, raw.columns[p.index].values, holder_keys, settings.min_cell)
    if pending:
        text_detect.partner_domains([(p, raw.columns[p.index].values) for p in pending])
        for p in pending:
            text_detect.count_units(p, raw.columns[p.index].values, holder_keys, settings.min_cell)
        if not defer_d35:
            finish_copy_gate([tp], settings.min_cell)
    # R8-05: an --na-value token the output would hold is decided by D35 like any real text (kept only when it is
    # vocabulary or held by >= --min-cell people, with no personal-data shape), else pseudonymised, still missing
    for i, (ctx, _det) in contexts.items():
        p = profiles[i]
        if not text_detect.custom_na(p):
            break  # one setting for every column
        if holder_keys is None:
            holder_keys = entity_mod.d35_keys(raw, profiles, ent, settings)
        text_detect.decide_na_tokens(
            p, raw.columns[i].values, holder_keys, settings.min_cell,
            personal_header=text_detect.guarded_hint(ctx) is not None or ctx.hints.free_text is not None,
            digit_phone=text_detect.digit_phone_header(ctx))
    if not defer_d35:  # R9-01: no value written verbatim is a real person ID (over the run: the pipeline)
        finish_real_ids([tp], settings.min_cell)
    sizes = entity_mod.entity_sizes(private["entity_index"], ent.n_entities)
    for i in ent.constant_columns:
        profiles[i].role = "entity"
        emodel = build_model(private["entity_values"][i], profiles[i])
        floor_entity_levels(profiles[i], emodel)
        floor_entity_extremes(profiles[i], emodel)
        if sizes is not None:  # rows per value, kept when the values are dealt to ID blocks (R2-10)
            emodel.extra["_row_counts"] = entity_mod.row_counts(private["entity_values"][i], sizes)
            if not is_level_generated(profiles[i]):  # numbers and dates: rows per value bin (R3-07)
                bins = structure_mod.entity_row_bins(private["entity_values"][i], sizes, profiles[i],
                                                     settings.min_cell)
                if bins is not None:
                    emodel.extra["_row_bins"] = bins
        tp.private.entity_models[i] = emodel
    if len(ent.id_columns) > 1:
        names = ", ".join(profiles[i].name for i in ent.id_columns)
        primary = profiles[ent.primary].name if ent.primary is not None else "the first"
        tp.messages.append(Message("info", "multiple_ids",
                                   f"several ID columns ({names}); {primary} defines the entity, the others are "
                                   "remapped independently"))
    cands: list[str] = []  # real values written (levels, interpolated numbers/dates with the exact min/max)
    masked: list[str] = []  # values already replaced (invented labels, same-shape masks, fake PII)
    for i, (ctx, det) in contexts.items():
        if profiles[i].role == "id":
            continue
        why = id_candidate_reason(ctx, det)
        if why:
            (masked if _values_replaced(profiles[i]) else cands).append(f"{profiles[i].name} ({why})")
            profiles[i].params["id_candidate"] = why
            redact_candidate(profiles[i].stats, why)
            for key in _VALUE_PARAMS:  # detection params that quote real values become private
                if key in profiles[i].params:
                    profiles[i].params["_" + key] = profiles[i].params.pop(key)
    groups = []
    if cands:
        groups.append("their real values are written as-is, including the exact min/max: " + "; ".join(cands))
    if masked:
        groups.append("their values are replaced (invented same-shape values) but are not kept consistent per "
                      "entity or across files like IDs: " + "; ".join(masked))
    if not ent.id_columns:
        if groups:
            tp.messages.append(Message(
                "warning", "no_id_column",
                "no ID column matched; each row is its own entity. ID-like candidates: " + ". ".join(groups),
                fix=ID_FIX_EACH))
        else:
            tp.messages.append(Message("info", "no_id_column",
                                       "no ID column matched; each row is its own entity (D7)"))
    elif groups:
        tp.messages.append(Message(
            "warning", "id_candidates",
            "column(s) look like IDs but are not treated as IDs; " + ". ".join(groups),
            fix=ID_FIX_EACH))
    # missingness patterns (D6): row-level columns over rows, within-ID constants (D4) over entities
    masks: dict[int, np.ndarray] = {}
    for p in profiles:
        if p.role != "row":
            continue
        masks[p.index] = missing_mask(raw.columns[p.index].values, p.missing_policy, p.na_tokens)
    entity_masks = {i: missing_mask(private["entity_values"][i], profiles[i].missing_policy, profiles[i].na_tokens)
                    for i in ent.constant_columns}
    tp.structure = structure_mod.profile_structure(raw, profiles, ent, masks, settings, entity_masks,
                                                   private["entity_index"], file_rel=file_rel)
    # privacy: hashed real comparison keys
    matches.real_keys(tp, [c.values for c in raw.columns])
    # privacy: the table's real full names (first + last cells of a row too), checked run-wide (R1-04)
    tp.private.extra["real_name_pairs"] = pii.real_name_pairs([c.values for c in raw.columns], profiles)
    return tp


def _column_profile(det: Detection, ctx: DetectContext, col: Any, settings: Settings) -> ColumnProfile:
    """The profile of one detected column: its detection, generation model and public statistics. Its NA tokens are
    the context's: the run's (``--na-value`` included) and the non-finite numbers its storage holds natively (K-20)."""
    p = _profile_from(det, ctx.index, col.name, ctx.na_tokens)
    p.native = col.native
    if p.type is ColumnType.IDENTIFIER:
        p.role = "id"
    if ctx.options.no_exact_extremes:
        p.params["_no_exact_extremes"] = True
    p.model = build_model(col.values, p, ctx.raw_counts)
    p.stats = column_stats(col.values, p, settings.min_cell, ctx.raw_counts)
    return p


#: K-02 (harden 2, A2-4): beside an ID of the row's own person, a marked column (``detect.ids.mark_weak_id``) is an ID
#: column when at least this share of its distinct keys are each held by fewer than ``--min-cell`` people of that ID:
#: keys as fine-grained as people (another ID of the student, ``Statewide ID``, ``Lunch Number``; the IDs of other
#: people of the row, ``Sibling SSID``, whose keys each belong to a family's 1-3 students), not a code that
#: ``--min-cell`` or more people share (a test, a bus, a class), which cannot single anyone out (D35's rule).
WEAK_ID_PERSON_KEY_SHARE = 0.5


def weak_id_columns(raw: RawTable, profiles: list[ColumnProfile],
                    contexts: dict[int, tuple[DetectContext, Detection]], settings: Settings,
                    dropped_marks: dict[int, dict[str, Any]] | None = None) -> dict[int, Detection]:
    """K-02, fail-safe: the IDENTIFIER detections of the marked columns of a table (``detect.ids.mark_weak_id``: a
    header with a student word, a student-ID code or a final ID word over ID-like keys) that are ID columns.

    With no ID column of the row's own person (``profile.entity.id_role`` ``'subject'``), a person's ID repeats on
    each of their rows and nothing else in the table numbers them: every marked column is an ID column
    (``Student_SK``, ``Stu``, ``CALPADS SSID``, ``Admission No``, ``KIDS ID``; harden 2, A2-2: 3-digit keys too, a
    small school's admission numbers cannot be told from room or bus numbers by their values, and the ID warning
    names ``--org-code`` to keep real values of things). Beside such an ID, a header that names a student
    (``hints.person_head_hint``: ``Mentor Student``) is an ID column, and any other marked column when its keys are as
    fine-grained as people (:data:`WEAK_ID_PERSON_KEY_SHARE`: another ID of the student, ``Statewide ID``,
    ``Lunch PIN``, or of other people, ``Sibling SSID``); a code ``--min-cell`` or more students share (a test, a
    class, a bus, ``Student Homeroom``) keeps its type. Student-word and student-ID-code columns are decided first, so
    they count as the person's ID for the others.

    ``dropped_marks`` (harden 2, A2-7): the marks an ``--exclude``d or ``--null``ed column would have (column index ->
    mark, :func:`dropped_weak_mark`). Such a column is decided the same way; when it would be an ID column its
    profile gets ``profile.entity.FLAGGED_ID_KEY`` (the source it would have): a shadow ID whose real IDs are still
    real IDs of the run (D9, the leak scan) and still tell D35 who holds a value."""
    from ..detect import ids as dids
    from ..hints import person_head_hint

    marked = {i: ctx_det[1].params[dids.WEAK_ID_KEY] for i, ctx_det in contexts.items()
              if profiles[i].type is not ColumnType.IDENTIFIER and ctx_det[1].params.get(dids.WEAK_ID_KEY)}
    dropped_marks = dict(dropped_marks or {})
    if not marked and not dropped_marks:
        return {}
    subjects = [p.index for p in profiles if p.type is ColumnType.IDENTIFIER and p.role == "id"
                and entity_mod.id_role(p) == "subject"]
    out: dict[int, Detection] = {}
    shadows: list[int] = []
    for phase in ("person", "id"):
        for i, mark in [*marked.items(), *dropped_marks.items()]:
            if (mark["form"] == "id") != (phase == "id"):
                continue
            # beside the person's ID: a thing many students share ('Student Homeroom', 'Bus No') is no ID column; a
            # header that names a student ('Mentor Student', 'Stu') holds students, IDs by its values
            if subjects and not person_head_hint(raw.columns[i].name) and not any(
                    _person_keys(raw.columns[i].values, raw.columns[j].values, settings.min_cell) for j in subjects):
                continue
            if i in dropped_marks:
                profiles[i].params[entity_mod.FLAGGED_ID_KEY] = "id-shape" if mark["form"] == "id" else "person-id-hint"
                shadows.append(i)
            else:
                out[i] = dids.weak_id_detection(contexts[i][0], str(mark["form"]))
        if phase == "person":
            subjects += [i for i, det in out.items() if det.source == "person-id-hint"
                         and entity_mod.id_role(_profile_from(det, i, raw.columns[i].name, settings.na_tokens))
                         == "subject"]
            subjects += [i for i in shadows if entity_mod.id_role(profiles[i]) == "subject"]
    for i in marked:
        if i not in out:
            profiles[i].params.pop(dids.WEAK_ID_KEY, None)
    return out


def _person_keys(values: np.ndarray, person_ids: np.ndarray, min_cell: int) -> bool:
    """K-02 (harden 2, A2-4): at least :data:`WEAK_ID_PERSON_KEY_SHARE` of the distinct D8 keys of ``values`` (over the
    rows where both columns are filled) are each held by fewer than ``min_cell`` (at least 2) person IDs."""
    keys = pd.Series(entity_mod.entity_keys(values), dtype=object)
    owners = pd.Series(entity_mod.entity_keys(person_ids), dtype=object)
    both = keys.notna() & owners.notna()
    if not both.any():
        return False
    per_key = pd.DataFrame({"k": keys[both], "o": owners[both]}).groupby("k", sort=False)["o"].nunique()
    return int((per_key < max(min_cell, 2)).sum()) >= WEAK_ID_PERSON_KEY_SHARE * len(per_key)


#: D38: PRIVATE param of a column of repeated wide keys of which nearly all are North American phone numbers
#: (``detect.text.nanp_phone_values``): its FAIL's fix names ``--pii-col COL=phone`` too.
_REPEATED_PHONES_KEY = "_repeated_keys_phones"
#: D38: the flags that type a column so that its real values may be written (``--org-code``, ``--force-type
#: COL=TYPE``, ``--force-factor``): on a column of repeated wide keys they are the user's decision, said aloud.
D38_TYPING_SOURCES = frozenset({"org-code", "force-type", "force-factor"})


def flagged_repeated_keys(p: ColumnProfile, ctx: DetectContext, wide: int = 0) -> None:
    """D38: mark (``detect.ids.FLAGGED_KEYS_KEY``) a column of repeated wide keys (``detect.ids.repeated_keys_width``,
    candidate rule 7's header and value test; ``wide``: rule 7's own width when known) that a flag typed so that its
    real values may be written (:data:`D38_TYPING_SOURCES`, a type the generators may copy values of). Its file is
    written whatever the table holds (the flag is the user's decision); the report and the docs withhold its value
    statistics, and its keys still count D35 holders where the table has no ID of the row's own person
    (``detect.ids.REPEATED_KEYS_KEY``). ``--org-code`` (or ``--force-type COL=ORG_CODE``), the fix D38's FAIL names
    for a school or district code, gets an info note; ``--force-type`` to another type and ``--force-factor`` a
    warning on every run, because the column looks like people's IDs and its real values are written where that type
    writes them (they never silently pass for something else)."""
    if p.source not in D38_TYPING_SOURCES or p.type in NEVER_COPY_TYPES:
        return
    if header_org_code(ctx):
        return  # an organisational code by its header (D10, D39) without the flag too: the flag decides nothing here
    width = wide or repeated_keys_width(ctx)
    if not width:
        return
    p.params[FLAGGED_KEYS_KEY] = width
    # its keys still tell D35 who holds a value where the table has no ID of the row's own person (the safe side: an
    # organisational code counts fewer holders than there are people, never more)
    p.params.setdefault(REPEATED_KEYS_KEY, width)
    redact_candidate(p.stats, flagged_keys_reason(p))
    flag = FLAG_NAMES.get(p.source, "a flag")
    if p.type is ColumnType.ORG_CODE:
        p.messages.append(Message(
            "info", "d38_flagged_keys",
            f"repeated keys of up to {width} digits kept as an organisational code by {flag} (D10, D38): its real "
            "values are written as they are; the report does not list them", column=p.name))
        return
    p.messages.append(Message(
        "warning", "d38_flagged_keys",
        f"repeated keys of up to {width} digits, typed {p.type.value} by {flag}: the file is written (D38 refuses it "
        "only without a flag), and real values are written where that type writes them; they look like people's IDs",
        column=p.name, fix=f"--person-id {p.name} (or --id-col {p.name}) if it holds person IDs (synthetic IDs), "
                           f"--org-code {p.name} if it is a school or district code"))


def unrecognised_key_columns(tp: TableProfile) -> list[ColumnProfile]:
    """D38 (user, 2026-10-11): the columns that FAIL the file of a table: candidate rule 7's repeated keys of
    ``detect.ids.REPEATED_KEY_DIGITS`` or more digits under a header that names no measure, count or flag
    (``detect.ids.REPEATED_KEYS_KEY``) in a table with no recognised ID column (``tp.entity.id_columns``: none by the
    default ID list, a person-ID header, K-02's header and values, the run's key discovery or an ID flag). Such keys may
    be people's IDs under a header the tool does not know, and the table shows nothing else of who its people are, so
    the file is flagged and skipped rather than written on a chance threshold (D7 only warns about them beside an ID
    column). A column an explicit flag typed (``--org-code``, ``--force-type``, ``--force-factor``, ``--force-text``,
    ``--pii-col``; ``profile.entity.EXPLICIT_SOURCES``) is the user's decision (:func:`flagged_repeated_keys`), and
    ``--person-id`` / ``--id-col`` make it an ID column, ``--exclude`` / ``--null`` drop it."""
    if tp.entity.id_columns:
        return []
    return [p for p in tp.columns if p.params.get(REPEATED_KEYS_KEY) and p.role not in ("id", "excluded", "null")
            and p.source not in entity_mod.EXPLICIT_SOURCES]


def unrecognised_keys_failure(rel_path: str, tables: list[TableProfile]) -> tuple[str, str] | None:
    """D38: the (reason, fix) a file FAILs with when a table of it has :func:`unrecognised_key_columns`, else None.
    Names and digit counts only: never a value."""
    from .structure import table_ref

    found = [(table_ref(rel_path, tp.name, p.name), int(p.params[REPEATED_KEYS_KEY]),
              bool(p.params.get(_REPEATED_PHONES_KEY))) for tp in tables for p in unrecognised_key_columns(tp)]
    if not found:
        return None
    refs = ", ".join(r for r, _, _ in found)
    width = max(w for _, w, _ in found)
    reason = (f"column(s) {refs}: keys of {REPEATED_KEY_DIGITS} or more digits (up to {width}) repeated across rows, "
              "under a header that names no ID, measure, count or flag, in a table with no recognised ID column; they "
              "may be people's IDs (such as a state student ID) or school or district codes, which their values cannot "
              "tell apart, so the file is not written (D38)")
    # nearly all North American phone numbers (a household's number on each of its rows): --pii-col says so
    fix = "; ".join(f"--person-id {r} (or --id-col {r}) if it is a person ID, --org-code {r} if it is a school or "
                    "district code" + (f", --pii-col {r}=phone if it holds phone numbers" if phone else "")
                    for r, _, phone in found)
    return reason, fix


def dropped_weak_mark(ctx: DetectContext) -> dict[str, Any] | None:
    """K-02 (harden 2, A2-7): the mark (``detect.ids.WEAK_ID_KEY``) an ``--exclude``d or ``--null``ed column would have,
    detected as if it were written (its header and ID-like values: ``Student_SK``, ``Stu``, ``CALPADS SSID``,
    ``Admission No``), or None. Dropping the column writes none of its values, but its real IDs are still real IDs of
    the run and still tell who holds a value (D35)."""
    from ..detect import ids as dids
    from ..errors import DatanonymousError
    from ..hints import weak_id_header

    if weak_id_header(ctx.name) is None:
        return None
    try:
        det = detect_column(ctx)
    except DatanonymousError:
        return None  # a column the run would refuse is not written either: no ID by its values
    if det.type is ColumnType.IDENTIFIER:
        return None  # an ID by its header or a flag: profile.entity.shadow_id already says so
    mark = det.params.get(dids.WEAK_ID_KEY)
    return dict(mark) if mark else None


def dropped_values(values: np.ndarray, na_tokens: frozenset[str] = frozenset()) -> frozenset[str]:
    """R7-02: the distinct non-missing values (stripped; blanks and ``na_tokens`` out) of a column that ``--exclude``
    or ``--null`` keeps out of the output: real values of the run all the same (``privacy.texts``)."""
    vals = np.asarray(values, dtype=object)
    if not len(vals):
        return frozenset()
    out = {v.strip() for v in codes_of(vals)[1] if isinstance(v, str)}
    out.discard("")
    return frozenset(out - set(na_tokens))


def floor_far_sentinels(raw: RawTable, profiles: list[ColumnProfile],
                        contexts: dict[int, tuple[DetectContext, Detection]], ent: Any,
                        settings: Settings) -> entity_mod.HolderKeys | None:
    """K-07: a date sentinel found by the far-from-median rule (``detect.dates.FAR_SENTINELS_KEY``) is written
    verbatim, so it must be held by >= ``--min-cell`` distinct entities of the table, counted like the D35 holders
    (``detect.dates.far_sentinel_holders``; ``profile.entity.d35_keys``). A column with a far value fewer hold is read
    again with only the others as far sentinels (``detect.dates.far_sentinels_only``): that value is an ordinary date
    (only the exact min/max of D14 is written as it is), and the column's model and statistics are rebuilt. The fixed
    sentinels and zero dates are unaffected. Returns the table's holder keys when they were computed (else None)."""
    from ..detect import dates as dates_detect

    keys = None
    for i, (ctx, _det) in list(contexts.items()):
        p = profiles[i]
        far = p.params.get(dates_detect.FAR_SENTINELS_KEY)
        if not far or p.role != "row":
            continue
        if keys is None:
            keys = entity_mod.d35_keys(raw, profiles, ent, settings)
        held = dates_detect.far_sentinel_holders(p, raw.columns[i].values, keys)
        keep = sorted(s for s in far if held.get(s, 0) >= settings.min_cell)
        if len(keep) == len(far):
            continue
        with dates_detect.far_sentinels_only(keep):
            det = detect_column(ctx)
        fresh = _profile_from(det, i, p.name, p.na_tokens)
        for attr in ("type", "subtype", "params", "reason", "source", "ambiguous", "ambiguity", "missing_policy",
                     "messages"):
            setattr(p, attr, getattr(fresh, attr))
        if ctx.options.no_exact_extremes:
            p.params["_no_exact_extremes"] = True
        p.messages.append(Message(
            "info", "date_far_sentinels",
            f"{len(far) - len(keep)} value(s) far from the column's other dates are held by fewer than "
            f"{settings.min_cell} {keys.basis}: they are read as dates, not written as sentinel tokens (K-07)",
            column=p.name))
        p.model = build_model(raw.columns[i].values, p, ctx.raw_counts)
        p.stats = column_stats(raw.columns[i].values, p, settings.min_cell, ctx.raw_counts)
        contexts[i] = (ctx, det)
    return keys


def finish_copy_gate(tables: list[TableProfile], min_cell: int, real_ids: realids.RealIds | None = None) -> int:
    """D35: decide every counted real text unit of ``tables`` (one table, or every table of the run) and return the
    number of pseudonymised units.

    A unit's holders are counted over the columns of the same field in all of ``tables`` (``detect.text.run_holders``:
    the same column name in several files, and a code table's code column with the column that uses its codes under
    another name, ``detect.text.lookup_fields``; the most people any table with proven people shows, else the fewest
    any ``--person-id`` claim or bare ``SID`` shows, records or rows only where neither counts; a claimed subject ID
    whose keys are the run's proven student IDs proves people, ``profile.entity.corroborated_tables``), so a field
    gets the same decision in every file and a lookup table follows the table that uses its codes; every
    pseudonymised value gets one pseudonym for the run (``generate.pseudonyms.plan``). Statistics that list levels or
    tokens are recomputed so that each pseudonymised one shows as its placeholder (D21).

    A unit that is a real person ID of the run (``real_ids``: the run's, ``privacy.realids.plan``; else the IDs of
    ``tables`` themselves) is never kept (R9-01): an organisational code, a short number or a count of holders never
    makes a real ID public.

    D36 over the run (``detect.text.spread_org_names``): a person's name found in an organisational code is replaced
    with the same fake name in every other unit of ``tables`` that would be written verbatim and holds it (``Class``,
    ``Homeroom``, a comma list read whole), so no column keeps it and joins on the code still match.
    """
    columns = [(tp, p) for tp in tables for p in tp.columns if p.params.get(text_detect.PENDING_KEY)]
    if not columns:
        return 0
    ids = real_ids if real_ids is not None else realids.of_tables(tables)
    # K-06 (K-04/K-05): a table whose claimed subject IDs (a bare SID, an --person-id on a header naming no person of
    # the row) are the run's proven student IDs counts proven people
    for tp in entity_mod.corroborated_tables(tables):
        for p in tp.columns:
            if p.params.get(text_detect.PENDING_KEY) and p.params.get(text_detect.PEOPLE_KEY) != \
                    text_detect.PEOPLE_PROVEN:
                p.params[text_detect.PEOPLE_KEY] = text_detect.PEOPLE_PROVEN
                p.params[text_detect.BASIS_KEY] = "people (IDs of the run's students)"
                p.params.pop(text_detect.SID_KEY, None)  # a proven SID: no d35_rows_counted warning (H-6)
    groups = text_detect.lookup_fields(p for _, p in columns)
    people_units: dict[str, set[str]] = {}
    holders = text_detect.run_holders((p for _, p in columns), groups, people_units)
    total = 0
    changed: dict[int, tuple[TableProfile, ColumnProfile]] = {}
    for tp, p in columns:
        field = text_detect.holder_field(p)
        field = groups.get(field, field)
        n = text_detect.copy_gate_units(p, holders.get(field, {}), min_cell, people_units.get(field, set()),
                                        real_id=None if realids.exempt(p) or not ids else ids.holds)
        total += n
        if n:
            changed[id(p)] = (tp, p)
    # D36 over the run: a name found in an organisational code is written with its fake name wherever another column
    # would write it verbatim (Class, Homeroom, a pivoted Period_3), so joins hold and no column keeps the real name
    table_of = {id(p): tp for tp, p in columns}
    before = {id(p): int(p.params.get(text_detect.PSEUDO_COUNT_KEY) or 0) for _, p in columns}
    for p in text_detect.spread_org_names(p for _, p in columns):
        total += int(p.params.get(text_detect.PSEUDO_COUNT_KEY) or 0) - before[id(p)]
        changed[id(p)] = (table_of[id(p)], p)
    for tp, p in changed.values():
        if p.model is not None and (p.type in LABEL_SAFE_TYPES or p.type is ColumnType.NUMERIC_WITH_TOKENS):
            counts = {**p.model.nonmissing_counts, **p.model.missing_counts}
            p.stats = column_stats(None, p, min_cell, counts, n=tp.n_rows)
    return total


def finish_real_ids(tables: list[TableProfile], min_cell: int, real_ids: realids.RealIds | None = None) -> int:
    """R9-01: after the D35 decision (:func:`finish_copy_gate`), no real unit of ``tables`` that is a real person ID of
    the run (``privacy.realids.id_units``: the run's ``real_ids``, else the IDs of ``tables`` themselves) is left to be
    written verbatim. Levels, tokens, list items and discrete numbers no D35 decision covered are pseudonymised like a
    D35 unit (exact count, one pseudonym for the run; a number gets a number of the column's format inside its range:
    ``generate.pseudonyms``); ``--na-value`` tokens too (still read as missing); the numbers of a continuous number
    column are recorded PRIVATELY (``privacy.realids.NUMBERS_KEY``) and every generated value equal to one is redrawn
    from the column's own distribution (``generate.numeric.avoid_real_ids``: the exact min/max of D14, an
    interpolated value landing on one). A number column with one gets its value statistics withheld from the report
    (its real min, max, quantiles and pmf would print the ID: ``params['id_candidate']``) and, when its real min or
    max is one, the D14 self-check skips that extreme (``privacy.realids.EXTREMES_KEY``). Returns the number of units
    handled here.
    """
    ids = real_ids if real_ids is not None else realids.of_tables(tables)
    for tp in tables:  # a table profiled on its own: its pseudonyms avoid its own IDs too (generate.pseudonyms)
        tp.private.extra.setdefault(realids.RUN_KEY, ids)
    if not ids:
        return 0
    total = 0
    for tp in tables:
        for p in tp.columns:
            units = realids.id_units(p, ids)
            if not units:
                continue
            done = (set(text_detect.pseudonymized(p)) | set(text_detect.na_pseudonymized(p))
                    | set(p.params.get(realids.NUMBERS_KEY) or ()))
            missing = {k.strip() for k in p.model.missing_counts if isinstance(k, str)}
            rows: dict[str, int] = {}
            for counts in (p.model.nonmissing_counts, p.model.missing_counts):
                for raw, c in counts.items():
                    for t in realids.unit_texts(p, raw):
                        rows[t] = rows.get(t, 0) + int(c)
            fresh = sorted((t for t in units if t not in done), key=lambda t: (-rows.get(t, 0), t))
            if not fresh:
                continue
            na = [t for t in fresh if t in missing]
            redraw = realids.continuous(p)
            numbers: list[str] = []
            if p.type in realids.NUMBER_TYPES:
                numbers = [t for t in fresh if t not in missing and realids.number_of(p, t) is not None]
            redrawn = set(numbers) if redraw else set()
            levels = [t for t in fresh if t not in missing and t not in redrawn]
            if levels:
                _pseudonymize(p, levels)
            if redraw and numbers:
                p.params[realids.NUMBERS_KEY] = sorted({*(p.params.get(realids.NUMBERS_KEY) or ()), *numbers})
            new: dict[str, str] = {}
            if na:
                rec_na = list(p.params.get(text_detect.NA_PSEUDO_KEY) or [])
                base = len({r[1] for r in rec_na})
                new = {t: text_detect.NA_PLACEHOLDER.format(base + i + 1) for i, t in enumerate(na)}
                rec_na += [[t, ph] for t, ph in new.items()]
                p.params[text_detect.NA_PSEUDO_KEY] = rec_na
                p.params[text_detect.NA_PSEUDO_COUNT_KEY] = len({r[1] for r in rec_na})
                p.flags[text_detect.NA_PSEUDO_COUNT_KEY] = True
            ends = [False, False]
            if numbers:
                ends = realids.extreme_flags(p, realids.numbers_of(p, numbers))
                if any(ends):
                    p.params[realids.EXTREMES_KEY] = ends
                    p.flags["extreme_value_adjusted"] = True
                p.params["id_candidate"] = realids.CANDIDATE_REASON
                for key in _VALUE_PARAMS:  # detection params that quote real values become private
                    if key in p.params:
                        p.params["_" + key] = p.params.pop(key)
            if p.type in LABEL_SAFE_TYPES or p.type is ColumnType.NUMERIC_WITH_TOKENS or na or numbers:
                counts = {**p.model.nonmissing_counts, **p.model.missing_counts}
                st = column_stats(None, p, min_cell, counts, n=tp.n_rows)
                if na and isinstance(st.get("missing_tokens"), dict):  # the report shows a placeholder, never the token
                    merged: dict[Any, int] = {}
                    for tok, c in st["missing_tokens"].items():
                        k = new.get(tok.strip(), tok) if isinstance(tok, str) else tok
                        merged[k] = merged.get(k, 0) + c
                    st["missing_tokens"] = merged
                p.stats = st
            total += len(fresh)
            how = ("each is replaced by another value drawn from the column's distribution" if redraw and numbers
                   else "each is written as a consistent same-shape pseudonym with its exact count")
            p.messages.append(Message(
                "warning" if numbers else "info", "real_id_values",
                f"{len(fresh)} {'value(s)' if len(fresh) > len(na) else '--na-value token(s)'} of the column equal a "
                "real person ID of the run (an ID column of this or another file, or one a flag turned into something "
                f"else): {how}, never written as it is (R9-01)"
                + ("; the column's value statistics are withheld from the report" if numbers else "")
                + ("; the exact min/max that is one cannot be emitted (D14)" if any(ends) else ""),
                column=p.name,
                fix=f"--force-factor {p.name} keeps the real values (only if they are not person IDs)"))
    return total


def _pseudonymize(p: ColumnProfile, texts: list[str]) -> None:
    """Record real units ``texts`` of column ``p`` as pseudonymised (``detect.text.PSEUDO_KEY``, like a D35 unit: a
    placeholder per unit, shared by the texts of one D35 key; params that quote one show its placeholder)."""
    rec = list(p.params.get(text_detect.PSEUDO_KEY) or [])
    shown = {r[0]: r[1] for r in rec}
    placeholders = {r[1] for r in rec}
    by_key: dict[str, str] = {}
    for t in texts:
        key = t if p.type in realids.NUMBER_TYPES else text_detect.unit_key(t)
        ph = by_key.get(key)
        if ph is None:
            ph = text_detect.PLACEHOLDER.format(len(placeholders) + 1)
            placeholders.add(ph)
            by_key[key] = ph
        rec.append([t, ph])
        shown[t] = ph
    p.params[text_detect.PSEUDO_KEY] = rec
    p.params[text_detect.PSEUDO_COUNT_KEY] = len(placeholders)
    p.flags[text_detect.PSEUDO_COUNT_KEY] = True
    if isinstance(p.params.get("levels"), list):
        p.params["levels"] = [shown.get(v.strip(), v) if isinstance(v, str) else v for v in p.params["levels"]]
    for key in ("value", "token"):
        v = p.params.get(key)
        if isinstance(v, str) and v.strip() in shown:
            p.params[key] = shown[v.strip()]


def _values_replaced(p: ColumnProfile) -> bool:
    """The column's generator never writes a real value (AREA C text/PII types, same-shape code masks)."""
    return p.type in NEVER_COPY_TYPES or (p.type is ColumnType.DIGIT_CODE and not p.params.get("levels_mode"))


def floor_entity_levels(p: ColumnProfile, emodel: ColumnModel) -> int:
    """D14 floor for a within-ID-constant column (D4): keep every source level.

    The per-entity model keeps each entity's most common value, so a level (or
    NA token) that only occurs as a minority value inside some entity would
    vanish. Each such level is added to the per-entity model with weight 1, so
    exact quota gives it at least one synthetic entity; the number of floored
    levels is stored in ``emodel.extra['floored_levels']`` and generation sets
    ``rare_category_adjusted``. Levels are floored for level-generated types;
    NA tokens for every type. Returns the number of floored levels.
    """
    row = p.model
    if row is None:
        return 0
    floored = 0
    if is_level_generated(p):
        for level in row.nonmissing_counts:
            if level not in emodel.nonmissing_counts:
                emodel.nonmissing_counts[level] = 1
                floored += 1
    for tok in row.missing_counts:
        if tok not in emodel.missing_counts:
            emodel.missing_counts[tok] = 1
            floored += 1
    if floored:
        emodel.extra["floored_levels"] = floored
    return floored


def floor_entity_extremes(p: ColumnProfile, emodel: ColumnModel) -> int:
    """D14 for a within-ID-constant column (D4): keep the column's exact min and max.

    The per-entity model keeps each entity's most common value, so a min or max
    held only by a minority row of some entity (``999.99`` in the second row of
    one ID) would vanish. Each lost extreme's raw value is added to the
    per-entity model with weight 1 (and to its sorted numbers), so the
    generator's exact-extremes rule emits it at least once; the number added is
    stored in ``emodel.extra['floored_extremes']`` and generation sets
    ``extreme_value_adjusted``. Applies to columns with a numeric view
    (continuous numbers, dates, AREA B formatted numbers) unless
    ``--no-exact-extremes``; discrete integers keep theirs through
    :func:`floor_entity_levels`. Returns the number of extremes added.
    """
    row = p.model
    if (row is None or row.sorted_numeric is None or not len(row.sorted_numeric) or is_level_generated(p)
            or p.params.get("_no_exact_extremes")):
        return 0
    xs = np.asarray(emodel.sorted_numeric if emodel.sorted_numeric is not None else [], dtype=float)
    lo, hi = float(row.sorted_numeric[0]), float(row.sorted_numeric[-1])
    lost = [v for v in dict.fromkeys((lo, hi)) if not len(xs) or v < xs[0] or v > xs[-1]]
    if not lost:
        return 0
    raws = [r for r in row.nonmissing_counts if isinstance(r, str)]
    nums = numeric_view_aligned(p, [r.strip() for r in raws])
    if nums is None:
        return 0
    added = []
    for v in lost:
        at = np.flatnonzero(nums == v)
        if not at.size:
            continue
        raw = raws[int(at[0])]
        emodel.nonmissing_counts[raw] = emodel.nonmissing_counts.get(raw, 0) + 1
        added.append(v)
    if added:
        emodel.sorted_numeric = np.sort(np.concatenate([xs, np.asarray(added, dtype=float)]))
        emodel.extra["floored_extremes"] = len(added)
    return len(added)

