"""ID/entity structure: rows per ID, within-ID constants, aliases, nesting (AREA D owns).

FOUNDATION implements a correct basic version used by the pipeline:

* :func:`profile_entity` - the primary ID column defines the entity of each row
  (D8 normalised key; a row with a blank ID is its own entity): the first
  IDENTIFIER column named by ``--id-col`` / ``--person-id`` / ``--force-type``,
  else the first on the default ID list, else the first person-ID header (D10);
  rows-per-ID pmf; within-ID-constant columns (D4: one distinct raw value,
  missing included, within >= 99% of the multi-row IDs) with one value per real
  entity (the entity's most common raw value) for generation.
* :func:`id_namespace` - the run-level ID registry key (D8 consistency across
  files): the normalised column name, except the generic bare ``id``, which is
  scoped to its own file and table (``id`` in ``students.csv`` and ``id`` in
  ``courses.csv`` are different entities).

AREA D:

* :func:`find_aliases_and_nesting` - relationships between the ID columns of a
  table, on D8 keys over the rows where both are filled: 1:1 pairs are
  **aliases** (``student_id`` + ``state_id`` + ``student_number``), many-to-one
  pairs are **nesting** (students inside ``family_ident``), anything else gets a
  warning. Every relationship is kept row by row because each ID column is
  remapped 1:1 at the real row positions (``generate.ids``); aliases whose
  values are identical in every row (``local_id`` = ``student_id``) are marked
  (``params['_same_values_as']``, PRIVATE) so ``profile.structure.cross_file_structure``
  gives them one shared synthetic mapping and they stay identical.
* :func:`id_namespace` honours ``params['_namespace']`` (set by
  ``cross_file_structure``: shared namespaces of identical aliases, or per-file
  namespaces when ``structure: {disable: [cross_file_keys]}``).
* :func:`key_ref_detection` (R2-01): a column found by the run-level key
  discovery to hold another ID column's real IDs is profiled again as an ID
  column (source ``key-ref``, lowest primary rank), so it gets synthetic IDs.
* :func:`row_counts` (R2-10): rows per value of a within-ID constant, so its
  values are dealt to ID blocks with the source's row counts.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Settings
from ..detect.context import DetectContext, Detection
from ..detect.ids import REPEATED_KEYS_KEY, id_text, identifier_detection, normalize_id
from ..hints import STATE_ID_CODES, STATE_ID_SYSTEMS, is_role_word, normalize, tokens
from ..model import ColumnProfile, ColumnType, EntityProfile, Message, RawTable
from .codes import codes_of

CONSTANT_SHARE = 0.99
#: Primary-ID preference by detection source (lower wins; ties: column order). ``key-ref``: a column treated as
#: an ID because it holds another ID column's keys (R2-01), and ``id-shape``: an ID-word header over ID-like keys
#: (K-02, ``Admission No``), define the entity only when nothing else does.
PRIMARY_RANK = {"id-col": 0, "person-id": 0, "force-type": 0, "default-id": 1, "person-id-hint": 2, "key-ref": 3,
                "id-shape": 3}
#: Normalised ID names too generic to link files by name alone (scoped per file/table).
GENERIC_ID_NAMES = frozenset({"id"})
#: Detection sources of explicit column flags: such a column keeps the type it was given (never a ``key-ref``).
EXPLICIT_SOURCES = frozenset({"force-type", "pii-col", "force-factor", "force-text", "org-code"})


def key_ref_detection(ctx: DetectContext, det: Detection, ref: dict[str, Any]) -> Detection:
    """R2-01: the detection of a column found to hold another ID column's IDs (``ref`` = ``{'of', 'share'}``
    from ``profile.structure._key_reference_columns``): an IDENTIFIER (source ``key-ref``), the way
    ``--id-col`` would make it, so it gets synthetic IDs instead of its real values. An explicitly typed column
    (or one that is already an ID) keeps ``det``. The message holds names and the share only.
    """
    if det.type is ColumnType.IDENTIFIER or det.source in EXPLICIT_SOURCES:
        return det
    of, share = str(ref["of"]), float(ref["share"])
    me = f"{ctx.file_rel}::{ctx.table}::{ctx.name}" if ctx.table else f"{ctx.file_rel}::{ctx.name}"
    out = identifier_detection(ctx, "key-ref", f"holds the IDs of ID column {of} ({share:.1%} of its distinct "
                                               "values): synthetic IDs (D5/D8/D10)")
    # R6-02: the IDs of a column a flag made no ID column (--exclude, --null, --force-*) are still real IDs
    shared = (" (that column was excluded, nulled or given another type by a flag, so there is no mapping to "
              "share)" if ref.get("flagged") else " from that column's mapping, so joins between them are kept")
    out.messages.append(Message(
        "warning", "id_key_ref",
        f"column {ctx.name!r} holds the IDs of ID column {of} ({share:.1%} of its distinct values), so it is treated "
        "as an ID column: it gets synthetic IDs of the same shape (never its real values)" + shared,
        column=ctx.name, fix=f"--force-type {me}=TYPE (or --force-factor {me}) if it does not hold person IDs: "
                             "its real values are then written"))
    return out


def entity_keys(values: np.ndarray) -> list[str | None]:
    """D8 key per row (None for a blank/missing ID, or a placeholder without a letter or digit: ``-``, ``?``;
    ``detect.ids.id_text``)."""
    if len(values) == 0:
        return []
    codes, uniques = codes_of(np.asarray(values, dtype=object))
    # a native null (None in typed formats) comes back from factorize as NaN: not a string, no ID
    keys = [None if (not isinstance(u, str) or not id_text(u)) else normalize_id(u) for u in uniques]
    return np.asarray(keys, dtype=object)[codes].tolist()


def entity_index(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Row entity codes, named-entity count and blank rows; blanks follow named IDs."""
    raw, uniques = codes_of(np.asarray(values, dtype=object))
    keys = np.asarray([None if not isinstance(u, str) or not id_text(u) else normalize_id(u)
                       for u in uniques], dtype=object)
    normalized, names = pd.factorize(keys)
    index = np.asarray(normalized, dtype=np.int64)[raw]
    blank = index < 0
    n_blank, n_named = int(blank.sum()), len(names)
    index[blank] = np.arange(n_named, n_named + n_blank, dtype=np.int64)
    return index, n_named, n_blank


def profile_entity(table: RawTable, profiles: list[ColumnProfile], settings: Settings) -> tuple[EntityProfile, dict[str, Any]]:
    """Basic entity profile; returns (public EntityProfile, private dict).

    The private dict holds ``entity_index`` (row -> entity number), ``entity_values``
    (D4 column index -> per-entity raw value array), ``id_raw`` and ``shadow_id_raw``
    (R6-02: the real values of every column that would be an ID column but a flag made
    it something else, :func:`shadow_id`).
    """
    ent = EntityProfile()
    private: dict[str, Any] = {"entity_index": None, "entity_values": {}, "id_raw": {}, "shadow_id_raw": {}}
    id_cols = [p.index for p in profiles if p.type is ColumnType.IDENTIFIER and p.role == "id"]
    ent.id_columns = id_cols
    for i in id_cols:
        private["id_raw"][i] = table.columns[i].values
    for p in profiles:
        if shadow_id(p, settings):
            private["shadow_id_raw"][p.index] = table.columns[p.index].values
    if not id_cols:
        return ent, private
    primary_candidates = id_cols
    if getattr(settings, "mode", "fidelity") == "sharing":
        # A record key must not silently displace the person's repeated key.
        primary_candidates = [i for i in id_cols if id_role(profiles[i]) == "subject"] or id_cols
    primary = min(primary_candidates, key=lambda i: (PRIMARY_RANK.get(profiles[i].source, 1), i))
    ent.primary = primary
    index, n_named, blank = entity_index(table.columns[primary].values)
    ent.n_entities = n_named + blank
    ent.blank_id_rows = blank
    sizes = np.bincount(index[index < n_named], minlength=n_named) if n_named else np.zeros(0, dtype=np.int64)
    ent.rows_per_id = {int(k): int(v) for k, v in sorted(Counter(sizes.tolist()).items())}
    multi = np.flatnonzero(sizes > 1)
    ent.multi_row_ids = int(len(multi))
    private["entity_index"] = index
    if len(multi) == 0:
        return ent, private
    mask_rows = np.isin(index, multi)
    sub_index = index[mask_rows]
    for p in profiles:
        if p.role != "row":
            continue
        vals = table.columns[p.index].values
        codes, uniques = codes_of(vals)
        width = len(uniques)
        if n_named <= np.iinfo(np.int64).max // max(width, 1):
            pairs = np.unique(sub_index * width + codes[mask_rows])
            holders = pairs // width
        else:
            # Avoid packed-key overflow even for an exceptionally large input.
            holders = np.unique(np.column_stack((sub_index, codes[mask_rows])), axis=0)[:, 0]
        nun = np.bincount(holders, minlength=n_named)[multi]
        share = float((nun <= 1).mean())
        if share >= CONSTANT_SHARE:
            ent.constant_columns.append(p.index)
            ent.constant_shares[p.index] = share
            private["entity_values"][p.index] = per_entity_values(index, vals, ent.n_entities)
    return ent, private


def per_entity_values(index: np.ndarray, values: np.ndarray, n_entities: int) -> np.ndarray:
    """Read-only source values per entity: most common raw value, first occurrence breaks ties."""
    out = np.empty(n_entities, dtype=object)
    if len(index) == 0:
        out.flags.writeable = False
        return out
    # vectorised (1M rows x many D4 columns): count (entity, value) pairs, keep the most common, earliest first
    vals = np.asarray(values, dtype=object)
    codes, uniq = codes_of(vals)
    uniq_arr = np.empty(len(uniq), dtype=object)
    uniq_arr[:] = [u if isinstance(u, str) else None for u in uniq.tolist()]  # a native null stays None
    df = pd.DataFrame({"e": np.asarray(index, dtype=np.int64), "v": np.asarray(codes, dtype=np.int64),
                       "r": np.arange(len(index), dtype=np.int64)})
    g = df.groupby(["e", "v"], sort=False)["r"].agg(["size", "min"]).reset_index()
    g = g.sort_values(["e", "size", "min"], ascending=[True, False, True], kind="stable")
    best = g.drop_duplicates("e")
    out[best["e"].to_numpy()] = uniq_arr[best["v"].to_numpy()]
    out.flags.writeable = False
    return out


# --------------------------------------------------------------------------- D35: who holds a value
#: Default ID names that may number records rather than people (PowerSchool ``DCID``, a bare ``id`` or
#: ``Unique ID``): no proof of people (D35).
RECORD_ID_NAMES = frozenset({"id", "dcid", "uniqueid"})
#: Header words (token prefixes) of an ID or name column of ANOTHER person than the row's own (a teacher, a staff
#: member, a guardian, a user account, the person who entered or provided something): one student seen by 12 teachers
#: is 12 teachers but ONE holder (D35, hardening passes 2 and 3). ``emp`` covers ``EmpID`` / ``EmployeeID``,
#: ``personnel`` ``Personnel_ID`` (never the ``person`` of ``PersonID``), ``user`` ``UserName`` / ``User_ID``.
_OTHER_PERSON_WORDS = (
    "teacher", "tchr", "staff", "emp", "personnel", "guardian", "parent", "counselor", "counsellor", "advisor",
    "adviser", "instructor", "principal", "contact", "caseworker", "coach", "tutor", "mentor", "manager", "nurse",
    "driver", "proctor", "sponsor", "user", "login", "author", "creator", "editor", "operator", "clerk", "secretary",
    "interpreter", "translator", "provider", "admin", "specialist", "clinician", "worker", "officer", "director",
    "coordinator", "liaison", "representative", "leader", "owner", "captain", "president", "physician", "therapist",
    "psychologist", "evaluator", "examiner", "supervisor", "reviewer", "approver", "recorder", "reporter",
    "submitter", "assessor", "observer", "facilitator", "volunteer")
#: Person words of an ID header that numbers the row's own person (``Student_ID``, ``Stu_Num``, ``Pupil_No``,
#: ``PersonID``, ``Family_ID``: a family count never exceeds the count of its members), qualifiers that may stand
#: before or after them (``State_Student_ID``, ``Student_Local_ID``) and the ID words that end such a header. A
#: header with any other word after the person word numbers that person's RECORDS (``Student_Test_ID``,
#: ``StudentSectionKey``, ``Student_Enrollment_ID``: hardening pass 3).
_SUBJECT_PERSON = "students|student|stu|pupils|pupil|learners|learner|children|child|persons|person|family"
_ID_QUALIFIERS = ("state|local|district|school|perm|permanent|unique|alt|alternate|sis|legal|official|primary|"
                  "secondary|other|old|new|prior|previous|former|current|tx|texas")
#: D38: the state student-ID acronyms (``hints.STATE_ID_CODES``: ``NYSSIS``, ``STN``, ``UIC``, ``WISEid`` ...) and the
#: state systems that name one with an ID word (``hints.STATE_ID_SYSTEMS``: ``EIS ID``, ``KIDS ID``).
_STATE_CODES = "|".join(STATE_ID_CODES)
_STATE_SYSTEMS = "|".join(STATE_ID_SYSTEMS)
_ID_ENDINGS = ("identifier|ident|ids|id|number|num|nbr|nr|no|key|code|ref|dcid|usi|uid|ssid|sasid|lasid|upn|gtid|sid|"
               + _STATE_CODES)
_SUBJECT_ID_RE = re.compile(rf"(?:{_ID_QUALIFIERS})*(?:{_SUBJECT_PERSON})(?:{_ID_QUALIFIERS}|{_ID_ENDINGS})*\d*")
#: Student-ID headers without a person word (``SID``, ``SSID``, ``LASID``, ``UPN``, ``GTID``, ``Perm ID``,
#: ``State_ID``, ``Local_ID``, ``TSDS Unique ID``, ``USI``; D38: ``NYSSIS``, ``STN``, ``WISEid``, ``EIS ID``).
_SUBJECT_CODE_RE = re.compile(
    rf"(?:{_ID_QUALIFIERS})*(?:(?:ssid|sasid|lasid|upn|gtid|usi|sid|{_STATE_CODES})(?:{_ID_QUALIFIERS}|{_ID_ENDINGS})*"
    rf"|(?:state|local|perm|permanent|tsds|{_STATE_SYSTEMS})(?:{_ID_QUALIFIERS})*(?:{_ID_ENDINGS})+)\d*")
#: A bare ``SID`` (``SID``, ``SID_No``, ``SID2``): a student ID on many exports, but sessions, sections, visits and
#: staff are numbered ``SID`` too, so the header alone proves no person (K-06, :func:`subject_proof`).
_BARE_SID_RE = re.compile(rf"sid(?:{_ID_ENDINGS})*\d*")
#: Detection sources of a column the user flagged as an ID without saying whose (``--id-col``, ``--force-type
#: COL=IDENTIFIER``): it numbers the row's own person only when its header says so, else it may number records
#: (``Incident_ID``, ``Row``).
_FLAGGED_ID_SOURCES = frozenset({"id-col", "force-type"})
#: Sources of explicit flags that turn an ID column into something else (its real IDs are still in the table).
_FLAG_SOURCES = frozenset({"force-factor", "force-text", "force-type", "pii-col", "org-code"})
#: Name hint kinds of a person-name column (``HeaderHints.pii``).
_NAME_HINTS = frozenset({"first_name", "last_name", "middle_name", "full_name"})


def other_person_header(name: str) -> bool:
    """D35 (hardening pass 3): the header names a person other than the row's own: a person-role word other than the
    subject's (``Teacher``, ``Guardian_Email``), an actor (``Entered_By``, ``Last_Modified_By``, ``CreatedBy``,
    ``Created_By_Email``, ``Assigned_To``) or a word for an account or a post (:data:`_OTHER_PERSON_WORDS`:
    ``UserName``, ``Provider_Name``, ``Interpreter_Name``, ``Sponsor_Name``, ``Personnel_ID``)."""
    from ..hints import PERSON_ACTOR_VERBS

    for t in tokens(name):
        if t in _SUBJECT_WORDS:
            continue
        if t == "by" or t in PERSON_ACTOR_VERBS or (t.endswith("by") and t[:-2] in PERSON_ACTOR_VERBS):
            return True
        if is_role_word(t) or any(t.startswith(w) for w in _OTHER_PERSON_WORDS):
            return True
    return False


def id_role(p: ColumnProfile) -> str:
    """D35: whom an ID column (or a column that would be one, :func:`shadow_id`) numbers: ``'subject'`` (the row's own
    person: ``student_id``, ``SID``, ``LASID``, ``Perm ID``, ``Pupil_ID``, ``Student_Ref``, ``--person-id``),
    ``'other'`` (another person: a teacher, staff, personnel or guardian ID, :func:`other_person_header`, or a column
    holding another ID column's IDs under a header that names no person of the row, ``key-ref``, or an ID word alone,
    ``id-shape``: ``Admission No``) or ``'record'``
    (``id``, ``DCID``, ``Unique ID``, a student's record key such as ``Student_Test_ID``, an ``--id-col`` whose header
    names no person: ``Incident_ID``, ``Study_ID``, ``Row``).

    A person word must be a whole word followed only by ID words and qualifiers (``Student_Local_ID``): ``Personnel``
    and ``Study`` are other words, and ``Student_Test_ID`` numbers tests (hardening pass 3). Only a subject ID proves
    that a value's holders are people: the others are still counted (a value is held by the minimum over the keys,
    :func:`d35_keys`), but one student's rows under 12 teachers or 12 incident numbers are one holder. Beside a
    subject ID an ``'other'`` key never counts (R6-05): one teacher of a whole class is one teacher, not one
    student."""
    from ..hints import DEFAULT_ID_SET, subject_id_hint

    norm = normalize(p.name)
    # an excluded or nulled column an ID flag names too has the role that flag gives it (R6-02)
    source = (getattr(p, "params", None) or {}).get(FLAGGED_ID_KEY) or p.source
    if norm in RECORD_ID_NAMES:
        return "record"
    if other_person_header(p.name):
        return "other"
    if _SUBJECT_ID_RE.fullmatch(norm) or subject_id_hint(p.name):  # K-02: CALPADS SSID, SIS ID, Kid ID
        return "subject"
    if source == "person-id":
        return "subject"  # --person-id: the user says it numbers the row's own person
    if source == "key-ref":
        return "other"  # another ID column's people (``subject_ref``, ``Buddy``): perhaps not the row's own
    if source in _FLAGGED_ID_SOURCES and norm not in DEFAULT_ID_SET:
        return "record"
    if _SUBJECT_CODE_RE.fullmatch(norm):
        return "subject"
    if source == "id-shape":
        # K-02: an ID word alone (``Admission No``, ``Asset No``) may number people, records or things: beside the
        # subject's ID it never lowers a count (R6-05), alone it counts (the safe side)
        return "other"
    return "record"  # Student_Test_ID, StudentSectionKey: a key of the person's records, not of the person


#: How an ID of the row's own person is known to be one (:func:`subject_proof`).
PROOF_HEADER, PROOF_FLAG, PROOF_SID = "header", "flag", "sid"


def subject_proof(p: ColumnProfile) -> str | None:
    """K-04/K-06: why an ID of :func:`id_role` ``'subject'`` numbers the row's own person, or None for another role.

    :data:`PROOF_HEADER`: its header says so (``Student_ID``, ``Pupil_No``, ``LASID``, ``Perm ID``, ``State_ID``, and
    the K-02 student-ID headers, ``hints.subject_id_hint``: ``CALPADS SSID``, ``SIS ID``, ``Student_SK``, ``Kid_Ref``,
    a header naming the student such as ``Stu``), with or without ``--person-id``;
    :data:`PROOF_FLAG`: only ``--person-id`` says so, on a header that names no person of the row (``Incident_ID``,
    ``Badge_No``: the report's own advice once suggested it for record numbers, and staff numbers have no role word);
    :data:`PROOF_SID`: a bare ``SID`` (:data:`_BARE_SID_RE`), which numbers sessions, sections or visits as often as
    students. Only a header proof (or a corroboration, :func:`d35_keys` and :func:`corroborated_tables`) lets a table's
    counts decide as counts of people over the run (``detect.text.run_holders``); weaker evidence never outweighs
    it (fail-safe: fewer holders win)."""
    from ..hints import subject_id_hint

    if id_role(p) != "subject":
        return None
    norm = normalize(p.name)
    if _SUBJECT_ID_RE.fullmatch(norm) or (not _BARE_SID_RE.fullmatch(norm) and (
            _SUBJECT_CODE_RE.fullmatch(norm) or subject_id_hint(p.name))):
        return PROOF_HEADER
    source = (getattr(p, "params", None) or {}).get(FLAGGED_ID_KEY) or p.source
    if source == "person-id":
        return PROOF_FLAG
    if _BARE_SID_RE.fullmatch(norm):
        return PROOF_SID
    return PROOF_HEADER


#: PRIVATE params key of an excluded or nulled column that an ID flag names too (``--id-col``, ``--person-id``,
#: ``--force-type COL=IDENTIFIER``), or whose header and ID-like values would make it an ID column (K-02 harden 2,
#: A2-7: ``--exclude Student_SK``; ``profile.table.weak_id_columns``): the detection source it would have had
#: (``id-col``, ``person-id``, ``force-type``, ``person-id-hint``, ``id-shape``); it would be an ID column
#: (:func:`shadow_id`, :func:`id_role`).
FLAGGED_ID_KEY = "_flagged_id"


def shadow_id(p: ColumnProfile, settings: Settings | None) -> bool:
    """A column that WOULD be an ID column (the default ID list, D7, a person-ID header, D10, or an ID flag on an
    excluded or nulled column or its header and ID-like values, K-02, :data:`FLAGGED_ID_KEY`) but a flag made it
    something else (``--exclude``, ``--null``,
    ``--force-factor``, ``--force-text``, ``--force-type``, ``--pii-col``, ``--org-code``): its real IDs are still
    real IDs of the run. D35: they still tell who holds a value. R6-02: they stay in every index of the run's real
    IDs (:func:`shadow_namespaces`: D9, the D34 screen, the leak scan, the key discovery), whatever the column
    writes."""
    from ..hints import header_hints

    if p.type is ColumnType.IDENTIFIER:
        return False
    if p.role not in ("excluded", "null") and p.source not in _FLAG_SOURCES:
        return False
    if p.params.get(FLAGGED_ID_KEY):
        return True
    h = header_hints(p.name)
    return h.person_id_like or (h.default_id and not getattr(settings, "no_default_ids", False))


def privacy_id_raw(tp: Any) -> dict[int, np.ndarray]:
    """R6-02: column index -> real raw values of every ID-like column of a table profile: its ID columns
    (``private.id_raw``) and its shadow IDs (``private.shadow_id_raw``, :func:`shadow_id`), PRIVATE."""
    private = getattr(tp, "private", None)
    out = dict(getattr(private, "shadow_id_raw", None) or {})
    out.update(getattr(private, "id_raw", None) or {})
    return out


def shadow_namespaces(files: Iterable[Any], failed: bool = True) -> list[Any]:
    """R6-02: the real IDs of the run's shadow ID columns (:func:`shadow_id`: ``--exclude student_id``, ``--null``,
    ``--force-factor`` ...), registered in a scratch ID registry (``generate.ids.IdRegistry``) by namespace
    (:func:`id_namespace`, under a ``shadow:`` key of their own); row-number namespaces (1..n, kept as they are,
    D9) are left out, as for ID columns. No synthetic ID is generated for them, but their real IDs are real IDs of
    the run: no synthetic ID equals one (D9), the D34 screen and the leak scan look for them, and a column holding
    them is an ID column (key discovery, R2-01). ``failed``: FAILed files count too. PRIVATE, in memory only."""
    from ..generate.ids import IdRegistry, namespace_is_identity

    reg = IdRegistry()
    for fr in files:
        if fr.failed and not failed:
            continue
        for tp in fr.tables:
            for i, raw in (getattr(tp.private, "shadow_id_raw", None) or {}).items():
                p = tp.columns[i]
                reg.register("shadow:" + id_namespace(fr.rel_path, tp.name, p), fr.rel_path, tp.name, i, p.name, raw)
    return [ns for ns in reg.namespaces.values() if ns.raw_values and not namespace_is_identity(ns)]


#: The duplicate-row rule (:meth:`HolderKeys.duplicates`) applies only where at most this share of the pairs of rows
#: that share a value are identical in every other column: elsewhere rows are alike by nature.
DUPLICATE_SHARE = 0.05


@dataclass
class HolderKeys:
    """What D35 counts the holders of a real text value of one table over (:func:`d35_keys`).

    ``keys``: one int64 row -> entity numbering per counting key (a value is held by the MINIMUM over the keys of
    the distinct entities among its rows; no key: rows count); ``people``: a key is an ID, name or e-mail address of
    the row's own person (False: record numbers, other people's IDs or rows); ``basis``: how the messages name what
    was counted; ``table`` (with the key columns in ``skip``): a table without a person key, whose other columns the
    duplicate-row rule reads (:meth:`duplicates`; their codes are computed once, on first use); ``own_rows``: a name
    or e-mail column of the row's own person (a bare one too) holds a different value on (nearly) every row, so its
    rows are no person's repeated rows (``detect.text.copy_gate_units`` then gives no ``d35_rows_counted`` warning);
    ``proven``: the people are proven by a header (an ID of :func:`subject_proof` :data:`PROOF_HEADER`, or names under
    a student header), not only claimed by ``--person-id`` (K-04, K-05); ``claimed``: the indexes of the table's ID
    columns whose subject role is unproven (``--person-id`` on a header naming no person of the row, a bare ``SID``):
    their keys may still prove people over the run (:func:`corroborated_tables`); ``sids``: the headers of the bare
    ``SID`` columns the counts of a table with no person key rest on (H-1: such counts may be of the students
    themselves, so over the run they rank as a claim, the fewest wins, never pooled with records and rows;
    ``detect.text.people_level``).
    """

    keys: list[np.ndarray]
    people: bool
    basis: str
    table: RawTable | None = None
    own_rows: bool = False
    skip: frozenset[int] = frozenset()
    proven: bool = False
    claimed: tuple[int, ...] = ()
    sids: tuple[str, ...] = ()
    _columns: dict[int, np.ndarray] | None = None
    _rows: np.ndarray | None = None

    @property
    def columns(self) -> dict[int, np.ndarray]:
        """Column index -> row value codes of every varying column of :attr:`table` not in :attr:`skip`."""
        if self._columns is None:
            self._columns = _varying_columns(self.table, set(self.skip)) if self.table is not None else {}
        return self._columns

    def _combined(self, skip: int | None = None) -> np.ndarray:
        """Row -> number of its content over :attr:`columns` (except ``skip``): identical rows share a number."""
        combined = np.zeros(len(next(iter(self.columns.values()))), dtype=np.int64)
        for i, codes in self.columns.items():
            if i != skip:
                combined = np.asarray(pd.factorize(combined * (int(codes.max()) + 1) + codes)[0], dtype=np.int64)
        return combined

    def duplicates(self, own: int, own_values: np.ndarray,
                   units: tuple[np.ndarray, np.ndarray, list[str]]) -> dict[str, int]:
        """Unit -> distinct rows among its rows (``units``: the (row, unit) pairs of the units of column ``own``,
        ``detect.text.unit_rows``), for the units whose rows are identical in every column far more often than rows
        alike in the rest of the table (one record exported several times is one holder); every other unit is left
        out (its rows count).

        The yardstick is the table itself: among the rows that share another value of column ``own``, the share of
        pairs that are identical in every other column (a narrow table, or columns that follow ``own`` such as a
        name beside its code, makes most pairs identical; a table with dates, minutes or scores makes few). Only a
        table where at most :data:`DUPLICATE_SHARE` of such pairs are identical tells duplicates apart; there the
        unit's identical pairs count as duplicates when they exceed that share of its pairs by more than three
        standard deviations (Poisson) plus one. Every unit is weighed in one pass over the pairs (R7-07: a pass over
        the whole table per unit was quadratic in a column of thousands of units).
        """
        row_of, unit_of, keys = units
        if not self.columns or not len(row_of):
            return {}
        if self._rows is None:
            self._rows = self._combined()
        n_units = len(keys)
        r = np.bincount(unit_of, minlength=n_units)  # rows holding each unit
        # identical rows among each unit's rows: (unit, row content) groups
        content = self._rows[row_of]
        c_width = int(content.max()) + 1
        grp, same = np.unique(unit_of * c_width + content, return_counts=True)
        g_unit = grp // c_width
        distinct = np.bincount(g_unit, minlength=n_units)
        found = (r >= 2) & (distinct < r)
        if not found.any():  # every unit's rows differ: nothing to weigh
            return {}
        own_codes, _ = codes_of(np.asarray(own_values, dtype=object))
        n_groups = int(own_codes.max()) + 1 if len(own_codes) else 0
        sizes = np.bincount(own_codes, minlength=n_groups).astype(float)
        pairs = sizes * (sizes - 1)  # ordered pairs of rows sharing a value of ``own``
        rest = self._combined(own)
        width = int(rest.max()) + 1 if len(rest) else 1
        joint, counts = np.unique(own_codes * width + rest, return_counts=True)
        alike = np.zeros(n_groups, dtype=float)  # ... that are identical in every other column
        np.add.at(alike, joint // width, counts.astype(float) * (counts - 1))
        # each unit's own values ("mine"): their pairs and alike pairs
        mine = np.unique(unit_of * max(n_groups, 1) + own_codes[row_of])
        m_unit, m_code = np.divmod(mine, max(n_groups, 1))
        pairs_mine = np.zeros(n_units, dtype=float)
        alike_mine = np.zeros(n_units, dtype=float)
        np.add.at(pairs_mine, m_unit, pairs[m_code])
        np.add.at(alike_mine, m_unit, alike[m_code])
        observed_all = np.zeros(n_units, dtype=float)
        np.add.at(observed_all, g_unit, same * (same - 1) / 2)
        pairs_sum, alike_sum = pairs.sum(), alike.sum()
        out: dict[str, int] = {}
        for u in np.flatnonzero(found).tolist():
            ru = int(r[u])
            base = float(pairs_sum - pairs_mine[u])
            if base > 0:
                share = float(alike_sum - alike_mine[u]) / base
            else:  # no other value of ``own`` repeats: the other columns' content shares outside the unit
                mask = np.ones(len(rest), dtype=bool)
                mask[row_of[unit_of == u]] = False
                outside = np.bincount(rest[mask], minlength=width) / max(1, len(rest) - ru)
                share = float((outside * outside).sum())
            if share > DUPLICATE_SHARE:
                continue  # rows alike by nature (a narrow or low-variety table): identical rows prove nothing
            observed = float(observed_all[u])
            expected = ru * (ru - 1) / 2 * share
            if observed > expected + 3 * expected ** 0.5 + 1:
                out[keys[u]] = int(distinct[u])
        return out


#: Role words that name the row's own subject (``Student_Name``): every other role word (``Teacher``, ``Guardian``)
#: names someone else, whose count says nothing about who holds the row's values.
_SUBJECT_WORDS = frozenset({"student", "students", "pupil", "pupils", "learner", "learners"})
#: What may follow ``stu`` in one header word naming the student's name or e-mail address (``StuName``,
#: ``stu_last_nm``, ``StuEmail``).
_STU_NAME_ENDINGS = frozenset({"name", "names", "nm", "first", "last", "middle", "fname", "lname", "mname", "lastfirst",
                               "firstlast", "fullname", "firstname", "lastname", "middlename", "firstnm", "lastnm",
                               "email", "mail"})


def _subject_header(name: str) -> bool:
    """A person-name or e-mail header of the row's own person (``Name``, ``LastFirst``, ``Student_Email``), not of
    another person (:func:`other_person_header`: ``Teacher``, ``Parent_Email``, ``Entered_By``, ``UserName``,
    ``Provider_Name``)."""
    return not other_person_header(name)


def subject_named_header(name: str) -> bool:
    """D35 (hardening pass 3): the header itself says the names are the STUDENT's (``Student_Name``, ``StudentName``,
    ``Stu_Last_Nm``, ``Pupil_Email``, ``Child_Name``). Only such names prove that a table's counts are of people: a
    bare ``Name``, ``LastFirst`` or ``Email`` may hold the staff member who wrote the row, and one table counting
    staff as people would vouch for a value in every file of the run (``detect.text.run_holders``)."""
    for t in tokens(name):
        if t in ("stu", "child", "children") or t.startswith(("student", "pupil", "learner")):
            return True
        if t.startswith("stu") and t[3:] in _STU_NAME_ENDINGS:
            return True
    return False


def subject_name_columns(profiles: list[ColumnProfile]) -> tuple[list[int], list[int]]:
    """D35: (whole-name and e-mail columns, name-part columns) of the row's own person (:func:`_subject_header`):
    PERSON_NAME and EMAIL columns, columns whose person-name or e-mail header made them invented labels, and
    excluded, nulled or flagged (``--force-factor`` ...) columns under such a header (their real values are still in
    the table). Name parts (first, middle, last) count together, as one name."""
    from ..hints import header_hints

    whole: list[int] = []
    parts: list[int] = []
    for p in profiles:
        if not _subject_header(p.name):
            continue
        if p.role in ("excluded", "null") or (p.source in _FLAG_SOURCES and p.type not in (ColumnType.EMAIL,
                                                                                           ColumnType.PERSON_NAME)):
            hint = header_hints(p.name).pii
            kind = "email" if hint == "email" else ("full" if hint == "full_name" else
                                                     "part" if hint in _NAME_HINTS else None)
        elif p.type is ColumnType.EMAIL:
            kind = "email"
        elif p.type is ColumnType.PERSON_NAME:
            kind = "full" if p.subtype in (None, "full") else "part"
        elif p.type in (ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.FREE_TEXT):
            hint = p.params.get("pii_hint")
            kind = "email" if hint == "email" else ("full" if hint == "full_name" else
                                                     "part" if hint in _NAME_HINTS else None)
        else:
            kind = None
        if kind == "part":
            parts.append(p.index)
        elif kind is not None:
            whole.append(p.index)
    return whole, parts


def _joined_codes(codes: list[np.ndarray]) -> np.ndarray:
    """One key over several name-part columns (first + last as one name; -1 where every part is blank)."""
    out = np.zeros(len(codes[0]), dtype=np.int64)
    blank = np.ones(len(codes[0]), dtype=bool)
    if not len(out):
        return out
    for c in codes:
        out = np.asarray(pd.factorize(out * (int(c.max()) + 2) + (c + 1))[0], dtype=np.int64)
        blank &= c < 0
    out[blank] = -1
    return out


def _key_codes(values: np.ndarray, id_like: bool) -> np.ndarray:
    """Codes of one key column (-1 for a blank cell): D8 keys for an ID column, else the case-folded text (one
    normalisation per distinct value)."""
    raw, uniq = codes_of(np.asarray(values, dtype=object))
    keys = [None if not isinstance(u, str) or not u.strip() or (id_like and not id_text(u))
            else normalize_id(u) if id_like else " ".join(u.casefold().split()) for u in uniq.tolist()]
    codes, _ = pd.factorize(np.asarray([k if k is not None else "" for k in keys], dtype=object))
    codes = np.asarray(codes, dtype=np.int64)
    codes[np.fromiter((k is None for k in keys), dtype=bool, count=len(keys))] = -1
    return codes[np.asarray(raw, dtype=np.int64)] if len(raw) else np.zeros(0, dtype=np.int64)


def _with_fallback(codes: list[np.ndarray], j: int, pool: list[int] | None = None) -> np.ndarray:
    """Key ``j`` with each blank cell filled by the row's first non-blank other key of ``pool`` (default: every key;
    each in its own namespace); rows blank in every key of the pool share ONE entity (a blank ID is not a new person
    each time). A key of the row's own person takes only another such key (hardening pass 3): a record number on a
    row with a blank student ID is no new student."""
    n = len(codes[j])
    out = np.full(n, -1, dtype=np.int64)
    offset = 0
    others = range(len(codes)) if pool is None else pool
    for i in [j] + [i for i in others if i != j]:
        c = codes[i]
        fill = (out < 0) & (c >= 0)
        out[fill] = c[fill] + offset
        offset += int(c.max()) + 1 if len(c) and c.max() >= 0 else 0
    out[out < 0] = offset
    return np.asarray(pd.factorize(out)[0], dtype=np.int64)


def _varying_columns(table: RawTable, skip: set[int]) -> dict[int, np.ndarray]:
    """Column index -> row value codes of every column not in ``skip`` that holds more than one value."""
    out: dict[int, np.ndarray] = {}
    for i, c in enumerate(table.columns):
        if i in skip:
            continue
        codes, _ = codes_of(np.asarray(c.values, dtype=object))
        codes = np.asarray(codes, dtype=np.int64)
        if len(codes) and codes.max() > 0:
            out[i] = codes
    return out


def d35_keys(table: RawTable, profiles: list[ColumnProfile], ent: EntityProfile,
             settings: Settings | None = None) -> HolderKeys:
    """D35: what counts who holds a real text value of this table. A value is held by the MINIMUM, over every key
    below, of the distinct keys among its rows (each key can only over-count the people holding it):

    * every ID column of the table (``ent.id_columns``), not only the primary one: a record ID (``id``, ``DCID``,
      ``--id-col Incident_ID``) or a staff ID listed before ``student_id`` must not turn one student's rows into many
      holders. A blank ID is no new person: it takes the row's other key, or joins the one entity of the rows blank
      in every key;
    * every column that would be an ID column but a flag changed it (:func:`shadow_id`: ``--exclude student_id``,
      ``--null``, ``--force-factor`` ...): its real IDs still tell who holds a value;
    * with no ID of the row's own person, an ID-like candidate of repeated wide keys (``detect.ids.REPEATED_KEYS_KEY``,
      harden 2 A2-3: a state student ID under a header no rule knows): it may number the rows' people, so its keys
      count too (the safe side); D38: also when a flag settled its file's FAIL by typing it (``--org-code``,
      ``--force-*``) or dropping it (``--exclude``, ``--null``);
    * except, beside an ID of the row's own person (:func:`id_role` ``'subject'``), the IDs of other people
      (``'other'``: a teacher, staff, guardian or ``key-ref`` ID; R6-05): they number those people, not the
      subject, so they can count FEWER holders than there are (one teacher of a whole class makes every activity
      of the class one holder) and say nothing the subject ID does not. The minimum stays over the subject's keys
      and the keys of unknown role (record IDs: ``id``, ``DCID``, ``--id-col``); with no subject ID every key
      counts (the safe side: nothing shows who the rows' people are);
    * the person-name and e-mail columns of the row's own person (:func:`subject_name_columns`: ``Name``,
      ``Student_LastFirst``, ``Student_Email``; not ``Teacher``, ``Entered_By``, ``UserName`` or ``Provider_Name``):
      a student's rows under 12 teacher IDs (or under a ``SID`` that numbers sections, sessions or visits) still carry
      one name. They always count, beside the person's own ID too (hardening pass 3): an ID that numbers one
      student's 12 visits looks one to one with the names when every other student has one visit, so only the names
      show that the visits are one holder (two namesakes holding a value count once: the safe side). Beside such an
      ID a lone first-name or last-name column never counts.

    The counts are of people (``HolderKeys.people``) only when a key numbers the row's own person: an ID of
    :func:`id_role` ``'subject'``, or names under a header that says they are the student's
    (:func:`subject_named_header`: ``Student_Name``; a bare ``Name`` or ``Email`` may hold the staff member who wrote
    the row). Teacher, staff and guardian IDs, IDs of another ID column (``key-ref``), record IDs and bare names prove
    nothing about who holds the value, so they never outweigh a count of people elsewhere in the run
    (``detect.text.run_holders``). The people are proven (``HolderKeys.proven``) only by a header
    (:func:`subject_proof`) or by such names: an ID that only ``--person-id`` calls the row's own person is a claim
    (K-04, K-05: counts of people that never outweigh proven ones), and a bare ``SID`` is a record key in its table
    (K-06: it numbers sessions and visits too; ``HolderKeys.sids``: over the run its counts still rank as a claim,
    H-1, since it may be the students' own ID), unless a name or e-mail column of the row's own person agrees with it
    one to one (:func:`_one_to_one`) or its keys are the run's proven student IDs (:func:`corroborated_tables`). A
    blank cell of a key of the row's own person takes only another such key, never a record number
    (:func:`_with_fallback`). With no person key, rows identical in every other column count once when chance alone
    would hardly make them identical (:meth:`HolderKeys.duplicates`: one record exported several times is one holder).
    """
    id_cols = [i for i in ent.id_columns if profiles[i].role == "id"]
    key_cols = id_cols + [p.index for p in profiles if p.index not in id_cols and shadow_id(p, settings)]
    roles = [id_role(profiles[i]) for i in key_cols]
    whole, parts = subject_name_columns(profiles)
    whole_codes = [_key_codes(table.columns[i].values, False) for i in whole]
    part_codes = [_key_codes(table.columns[i].values, False) for i in parts]
    id_codes = {i: _key_codes(table.columns[i].values, True) for i in key_cols}
    # K-04/K-06: an ID whose subject role only a flag or a bare SID claims proves people when a name or e-mail column
    # of the row's own person agrees with it one to one; a bare SID that no name backs numbers records (sessions,
    # sections, visits), and a claim stays a claim (its keys may still match the run's subject IDs:
    # corroborated_tables)
    names = whole_codes + ([_joined_codes(part_codes)] if len(part_codes) >= 2 else [])
    proven_ids = False
    claimed: list[int] = []
    for j, i in enumerate(key_cols):
        proof = subject_proof(profiles[i]) if roles[j] == "subject" else None
        if proof is None:
            continue
        if proof == PROOF_HEADER or any(_one_to_one(id_codes[i], n) for n in names):
            proven_ids = True
            continue
        claimed.append(i)
        if proof == PROOF_SID:
            roles[j] = "record"
    if "subject" in roles:  # R6-05: other people's IDs never lower a count of the subject's people
        key_cols = [i for i, r in zip(key_cols, roles) if r != "other"]
        roles = [r for r in roles if r != "other"]
    else:  # harden 2 (A2-3): an ID-like candidate of repeated wide keys ('NYSSIS') may number the rows' people
        extra = [p.index for p in profiles if p.params.get(REPEATED_KEYS_KEY) and p.index not in key_cols]
        key_cols += extra
        roles += ["record"] * len(extra)
        id_codes.update((i, _key_codes(table.columns[i].values, True)) for i in extra)
    subject_ids = [j for j, r in enumerate(roles) if r == "subject"]
    codes = [id_codes[i] for i in key_cols]
    if subject_ids and len(parts) < 2:
        parts, part_codes = [], []  # beside the person's own ID a lone first or last name only merges people
    named = any(subject_named_header(profiles[i].name) for i in whole + parts)
    own = list(subject_ids) + list(range(len(codes), len(codes) + len(whole) + (1 if parts else 0)))
    codes += whole_codes
    if parts:
        codes.append(_joined_codes(part_codes))
    keys = [_with_fallback(codes, j, own if j in own else None) for j in range(len(codes))]
    subject = bool(subject_ids)
    if subject or named:
        ids = "IDs" if proven_ids else "IDs that --person-id names"
        what = ", ".join(w for w, ok in ((ids, subject), ("names or e-mail addresses", named)) if ok)
        return HolderKeys(keys, True, f"people ({what})", proven=proven_ids or named, claimed=tuple(claimed))
    # H-1: no key proves or claims people here; each claimed column is a bare SID (a --person-id claim is a subject
    # key), so the counts are of SID values: the students' if SID is their ID, records' if it numbers sessions
    sids = tuple(profiles[i].name for i in claimed)
    basis = (f"{sids[0]} values (students or records: the header alone proves neither)" if sids
             else "records or rows (no ID or student-name column of the row's own person)" if keys
             else "rows (the table has no ID column)")
    own_rows = any(_distinct_per_row(codes[j]) for j in own)
    return HolderKeys(keys, False, basis, table=table, skip=frozenset(key_cols), own_rows=own_rows,
                      claimed=tuple(claimed), sids=sids)


#: Share of an ID's distinct keys (and of a name column's distinct names) that must pair with exactly one value of
#: the other column for the two to agree (:func:`_one_to_one`).
AGREE_SHARE = 0.9


def _one_to_one(a: np.ndarray, b: np.ndarray) -> bool:
    """K-06: two key columns (codes, -1 blank) number the same entities: over the rows where both are filled, at least
    :data:`AGREE_SHARE` of the values of each pair with exactly one value of the other (an ID of visits gives one name
    many IDs; a staff name beside a student ID gives one name many IDs too)."""
    both = (a >= 0) & (b >= 0)
    if not both.any():
        return False
    pairs = np.unique(np.stack([a[both], b[both]], axis=1), axis=0)
    for col in (0, 1):
        per = np.unique(pairs[:, col], return_counts=True)[1]
        if (per == 1).sum() < AGREE_SHARE * len(per):
            return False
    return True


#: PRIVATE ``TablePrivate.extra`` key: the indexes of the table's ID columns whose subject role is only claimed
#: (``HolderKeys.claimed``), read by :func:`corroborated_tables`.
CLAIMED_KEY = "d35_claimed_ids"
#: Share of a claimed ID column's distinct keys that must be IDs of a header-proven subject-ID column of the run
#: (:func:`corroborated_tables`).
CLAIM_OVERLAP_SHARE = 0.5


def _distinct_texts(raw: Any) -> list[str]:
    """The distinct non-blank values of a raw column, stripped."""
    out = {v.strip() for v in codes_of(np.asarray(raw, dtype=object))[1] if isinstance(v, str)}
    out.discard("")
    return sorted(out)


def corroborated_tables(tables: Iterable[Any]) -> list[Any]:
    """K-06 (and K-04/K-05): the tables among ``tables`` whose claimed subject IDs (:data:`CLAIMED_KEY`: a bare
    ``SID``, an ``--person-id`` on a header that names no person of the row) are IDs of a subject-ID column the run
    proves by its header (``Student_ID``, ``LASID`` ...; :func:`subject_proof`, ID columns and the ID columns a flag
    turned into something else, :func:`privacy_id_raw`, of any table): at least :data:`CLAIM_OVERLAP_SHARE` of the
    distinct values of one claimed column. Equality is read as R9-01 reads it (``privacy.realids.RealIds``: D8 keys
    of at least 4 characters with a digit, and a plain number only outside a dense, row-number-like range), so
    session or visit numbers 5000-5243 never match student IDs 5000-5079 by chance. Such a table counts people as
    surely as the table that proves them (``profile.table.finish_copy_gate``); session, visit, incident or badge
    numbers never are a student-ID namespace, so they stay records or claims."""
    from types import SimpleNamespace

    from ..privacy import realids

    tables = list(tables)
    claims = [(tp, list(tp.private.extra.get(CLAIMED_KEY) or ())) for tp in tables]
    if not any(c for _, c in claims):
        return []
    spaces = [SimpleNamespace(raw_values=_distinct_texts(raw)) for tp in tables
              for i, raw in privacy_id_raw(tp).items() if subject_proof(tp.columns[i]) == PROOF_HEADER]
    proven = realids.of_namespaces(spaces) if spaces else None
    if not proven:
        return []
    out = []
    for tp, cols in claims:
        raw = privacy_id_raw(tp)
        for i in cols:
            vals = _distinct_texts(raw[i]) if i in raw else []
            hits = sum(1 for v in vals if proven.key_of_text(v) is not None)
            if vals and hits >= CLAIM_OVERLAP_SHARE * len(vals):
                out.append(tp)
                break
    return out


#: Share of the filled rows that must hold distinct values (:func:`_distinct_per_row`).
DISTINCT_ROW_SHARE = 0.9


def _distinct_per_row(codes: np.ndarray) -> bool:
    """A name or e-mail column holds a different value on (nearly) every filled row, and fills most rows: each row is
    a different person, as far as the table shows."""
    filled = codes[codes >= 0]
    if len(filled) < max(2, len(codes) // 2):
        return False
    return len(np.unique(filled)) >= DISTINCT_ROW_SHARE * len(filled)


def entity_sizes(index: np.ndarray | None, n_entities: int) -> np.ndarray | None:
    """Rows of each entity (its ID block; a blank-ID row is a 1-row entity); None without entities."""
    if index is None or n_entities == 0:
        return None
    return np.bincount(np.asarray(index, dtype=np.int64), minlength=n_entities)[:n_entities]


def row_counts(entity_values: np.ndarray, sizes: np.ndarray) -> dict[Any, int]:
    """PRIVATE aggregate of a within-ID-constant column (D4): raw value -> rows of the entities holding it
    (the entity's value expanded to its ID block). Generation deals the per-entity values so these row counts
    are kept (``profile.structure.balance_entity_values``, R2-10)."""
    if len(entity_values) == 0:
        return {}
    codes, uniq = codes_of(np.asarray(entity_values, dtype=object))
    rows = np.bincount(np.asarray(codes, dtype=np.int64), weights=np.asarray(sizes, dtype=float),
                       minlength=len(uniq))
    return {(u if isinstance(u, str) else None): int(r) for u, r in zip(uniq.tolist(), rows.tolist())}


def id_namespace(file_rel: str, table_name: str | None, profile: ColumnProfile) -> str:
    """AREA D hook: key of the run-level ID registry for this ID column.

    Columns with the same namespace share one real-ID -> synthetic-ID mapping
    across every file of the run (D8). Called by the pipeline's planning step
    for every IDENTIFIER column. FOUNDATION: the normalised column name, so
    ``student_number`` and ``StudentNumber`` share a mapping; the generic bare
    ``id`` (:data:`GENERIC_ID_NAMES`) is scoped to its file and table. AREA D:
    ``params['_namespace']`` (set by ``profile.structure.cross_file_structure``)
    overrides the name-based key.
    """
    override = profile.params.get("_namespace")
    if override:
        return str(override)
    return base_namespace(file_rel, table_name, profile)


def base_namespace(file_rel: str, table_name: str | None, profile: ColumnProfile) -> str:
    """The name-based registry key (FOUNDATION rule, without the ``_namespace`` override)."""
    key = normalize(profile.name) or f"column{profile.index}"
    if key in GENERIC_ID_NAMES:
        return f"{key}@{file_rel}" + (f"#{table_name}" if table_name else "")
    return key


def find_aliases_and_nesting(table: RawTable, profiles: list[ColumnProfile], entity: EntityProfile) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """AREA D hook: ID columns that are 1:1 aliases, and many-to-one nesting.

    Called after :func:`profile_entity`. Returns (alias groups as column-index
    lists, nesting descriptions). Every pair of ID columns is compared on D8
    keys over the rows where both are filled:

    * 1:1 -> alias (groups are the connected components); identical keys in
      every such row also set ``params['_same_values_as']`` (PRIVATE) so the
      columns later share one synthetic mapping and stay identical;
    * many-to-one -> nesting ``{'child', 'parent', 'n_child', 'n_parent',
      'children_per_parent': {'min', 'max'}}`` (names and counts only);
    * otherwise a warning on the second column: it is remapped with its own
      mapping (the row-level relationship is still kept exactly).

    Each alias/nesting column also gets public ``params['alias_of']`` /
    ``params['nested_in']`` (column names).
    """
    ids = list(entity.id_columns)
    if len(ids) < 2:
        return [], []
    codes: dict[int, np.ndarray] = {}
    keyed: dict[int, list[str | None]] = {}
    for i in ids:
        keys = entity_keys(table.columns[i].values)
        keyed[i] = keys
        codes[i] = _key_codes(table.columns[i].values, id_like=True)
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nesting: list[dict[str, Any]] = []
    for a_pos, a in enumerate(ids):
        for b in ids[a_pos + 1:]:
            both = (codes[a] >= 0) & (codes[b] >= 0)
            if not both.any():
                continue
            ca, cb = codes[a][both], codes[b][both]
            a_to_b = is_functional(ca, cb)
            b_to_a = is_functional(cb, ca)
            pa, pb = profiles[a], profiles[b]
            if a_to_b and b_to_a:
                parent[find(b)] = find(a)
                ka = [keyed[a][r] for r in np.flatnonzero(both).tolist()]
                kb = [keyed[b][r] for r in np.flatnonzero(both).tolist()]
                if ka == kb:
                    pb.params.setdefault("_same_values_as", []).append(a)
                    pa.params.setdefault("_same_values_as", []).append(b)
            elif a_to_b or b_to_a:
                child, par = (a, b) if a_to_b else (b, a)
                cc, pc = (ca, cb) if a_to_b else (cb, ca)
                per_parent = pd.Series(cc).groupby(pc).nunique()
                desc = {"child": profiles[child].name, "parent": profiles[par].name,
                        "n_child": int(len(np.unique(cc))), "n_parent": int(len(np.unique(pc))),
                        "children_per_parent": {"min": int(per_parent.min()), "max": int(per_parent.max())}}
                nesting.append(desc)
                profiles[child].params.setdefault("nested_in", []).append(profiles[par].name)
            else:
                pb.messages.append(Message(
                    "warning", "id_columns_unrelated",
                    f"ID columns {pa.name!r} and {pb.name!r} are neither aliases (1:1) nor nested (many-to-one); "
                    "each is remapped with its own mapping (their row-level pairing is still kept exactly)",
                    column=pb.name))
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    aliases = [sorted(g) for g in groups.values() if len(g) > 1]
    aliases.sort()
    for g in aliases:
        for i in g:
            profiles[i].params["alias_of"] = [profiles[j].name for j in g if j != i]
    # one nesting line per alias group (state_id within family_ident repeats student_id's)
    first = {profiles[i].name: profiles[min(g)].name for g in aliases for i in g}
    seen: set[tuple[str, str]] = set()
    kept = []
    for d in nesting:
        key = (first.get(d["child"], d["child"]), first.get(d["parent"], d["parent"]))
        if key not in seen:
            seen.add(key)
            kept.append(d)
    return aliases, kept


def is_functional(a: np.ndarray, b: np.ndarray) -> bool:
    """True if every code in ``a`` maps to a single code in ``b`` (same-length int arrays)."""
    if len(a) == 0:
        return True
    order = np.argsort(a, kind="stable")
    sa, sb = a[order], b[order]
    new_run = np.empty(len(sa), dtype=bool)
    new_run[0] = True
    new_run[1:] = sa[1:] != sa[:-1]
    change = np.zeros(len(sb), dtype=bool)
    change[1:] = sb[1:] != sb[:-1]
    return not bool(np.any(change & ~new_run))
