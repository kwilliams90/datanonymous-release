"""Header vocabularies and value vocabularies from the design (FOUNDATION; shared file).

Header matching
---------------
Headers are compared case-insensitively after removing non-alphanumerics
(:func:`normalize`). Three match modes exist:

* ``exact``: the whole normalised header equals the hint (default ID list, D7).
* ``prefix``: the normalised header starts with the hint.
* ``token``: the hint matches at the start of a header *token*. Tokens come from
  splitting on non-alphanumerics, lower->Upper camelCase and letter/digit
  boundaries; the hint may span consecutive tokens (``exitcode`` matches
  ``Exit_Code``) and may be a prefix of a token (``school`` matches
  ``SchoolID``), but never starts inside a token (``site`` does not match
  ``composite_score``; ``term`` does not match ``midterm``).

Three header rules need more than a vocabulary and have their own functions:

* :func:`org_code_hint` (D10): the normalised header is one or more org-code
  words followed by nothing, an identifier suffix (id, code, name, number,
  desc ...) and/or digits: ``Course``, ``SchoolID``, ``school_name``,
  ``section_number``, ``ExitCodeDesc`` match; ``Section_Teacher``,
  ``School_Counselor``, ``Term_Advisor``, ``school_year``, ``midterm``,
  ``test preparation course`` and ``Yr_JoinCampus`` do not. A person-role word
  therefore never yields an org code. D39 adds the words of known non-person
  codes (:data:`KNOWN_CODE_WORDS`: ``NCESSCH``, ``LEA_ID``, ``CDS_Code``,
  ``CDC``, ``GL_Account``, ``Account Code``, ``Term_CRN``, ``ISBN13``; not a
  bare ``Account``); the generic finance words ``Fund``, ``Function`` and
  ``Object`` (:func:`finance_segment_hint`) make one only over code values
  (``detect.types``).
* :func:`person_id_hint` (D10): a person word (student, staff, family, teacher,
  employee, person, guardian, parent, pupil, learner, child; ``stu`` and ``kid``
  as whole words) plus an ID word as the *last* token (``Family_ID``,
  ``Guardian_Number``, ``StudentLocalID``, ``staff_no``, ``StudentUSI``,
  ``Stu_Num``, ``Kid_ID``) or a trailing ``#`` (``Student #``, ``Stu #``), or a
  last token that is a person word joined to an ID word (``familyid``,
  ``stuid``). K-02, structural: trailing numbering and ID qualifiers never hide
  the ID word (``Student_ID_2``, ``Student Number (Local)``), a code or reference
  word right after a student word is an ID word (``Student Code``,
  ``Stu_Code``, ``Pupil Code``, ``Learner Ref``; not ``Student_Status_Code``),
  and a student-ID code inside a longer header is one (``CALPADS SSID``,
  ``EdFi USI``, ``SIS ID``). :func:`default_id_hint` adds the default list's ID
  stems with any ID word (``ID Number``, ``Perm_No``) and D38's state student-ID
  acronyms (:data:`STATE_ID_CODES`: ``NYSSIS``, ``STN``, ``WISEid``, ``NYSSIS_ID``;
  :data:`STATE_ID_SYSTEMS` with an ID word: ``EIS ID``, ``KIDS ID``), and
  :func:`person_head_hint` names a header that IS the student (``Student``,
  ``Stu``, ``Pupil``): integer or code values under it are student IDs.
* :func:`person_name_hint` (D12): headers ending in ``name`` whose prefix is not
  a non-person noun (``Mother_Name``, ``AdvisorName``; not ``SchoolName``,
  ``course_name``), headers whose last token is a person-role word, singular or
  plural (``Teacher``, ``Section_Teacher``, ``EmergencyContact``, ``Student``,
  ``Guardians``), and ``<verb>_By`` actor headers (``Referred_By``). Trailing
  numbering and ranking never hides the role or name word (R2-04:
  ``Guardian1``, ``Contact_2``, ``Parent(s)``, ``Guardian_Primary``,
  ``Teacher_of_Record``, ``Name_2``, ``MotherName1``).
* :func:`person_name_part_hint` (D12): a name part after person-role or
  qualifier words (``Student_First``, ``stu_last_nm``, ``Guardian1First``,
  ``LegalLast``, bare ``First``/``Last``/``Middle``, ``Name_First``; ``LastFirst``
  and ``Name (Last, First)`` are full names), so such headers never fall
  through to copied factors.
* :func:`localized_name_hint` (D12, R3-04): person-name and person-role headers
  of the other languages of school data exports (Spanish, Portuguese, French,
  Italian, German, Dutch, Tagalog, Haitian Creole, Somali, Vietnamese, Chinese,
  Japanese, Korean, Russian, Ukrainian, Arabic, Hindi, Urdu): ``Nombre``,
  ``Apellido_Paterno``, ``Nombre del Padre``,
  ``Prénom``, ``Nom de famille``, ``Sobrenome``, ``Responsável``,
  ``Director_Escuela``, ``Họ và tên``, ``姓名``. Romance word order is read
  head first: a name word followed by a role names that person, a name word
  followed by a thing names the thing (``Nombre_Escuela``, ``Nome_Turma``), and
  an attribute before a role is that attribute (``Sexo_Alumno``,
  ``Telefono_Padre``). :func:`localized_pii_hint` adds whole-word phone, e-mail,
  address, city, ZIP, birth-date and national-ID headers (``Teléfono``,
  ``Correo``, ``Dirección``, ``Ciudad``, ``Código Postal``, ``CURP``). Both read
  run-together database names split into known words that cover the whole
  token (``apellidopaterno``, ``fechanacimiento``; ``renaissance`` stays whole).

* :func:`phone_header_hint` (D12, R4-02): phone headers are one structural rule
  instead of a prefix vocabulary: a phone word anywhere (``Phone``, ``homephone``,
  ``Telephone``, ``Mobile``, ``Fax``, ``Pager``, ``WhatsApp``) or a short phone
  word in a phone position (``Parent_Cell``, ``Tel_No``, ``Home_Ph``,
  ``SMS_Number``) is a ``phone`` PII hint; messaging and contact-qualifier headers
  (``SMS``, ``Text``, ``Home``, ``Primary``, ``Phone_Type``) are ``contact``
  headers (value-validated); ``Phoneme``, ``Text_Level``, ``Day_Number`` are
  neither.

Headers are folded to ASCII before matching (:func:`fold_accents`: ``Prénom`` ->
``Prenom``, ``Responsável`` -> ``Responsavel``), so accented letters never split
or hide a word.

Value vocabularies (ordinal levels, boolean pairs, school-year forms, NA tokens)
are compared on stripped, case-folded values unless stated otherwise.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

# --------------------------------------------------------------------------- header normalisation
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
#: Letters NFKD does not decompose (``Đặng`` -> ``Dang``, ``Øster`` -> ``Oster``, ``Straße`` -> ``Strasse``).
_FOLD_EXTRA = str.maketrans({"đ": "d", "Đ": "D", "\xf8": "o", "\xd8": "O", "ł": "l", "Ł": "L",
                             "\xe6": "ae", "\xc6": "AE", "œ": "oe", "Œ": "OE", "\xdf": "ss",
                             "ı": "i", "\xf0": "d", "\xd0": "D", "\xfe": "th", "\xde": "TH"})


@lru_cache(maxsize=8192)
def fold_accents(text: str) -> str:
    """``text`` without accents or ligatures, case kept (``Prénom`` -> ``Prenom``, ``Họ`` -> ``Ho``); letters
    of other scripts (``姓名``) are kept as they are."""
    if text.isascii():
        return text
    s = unicodedata.normalize("NFKD", text.translate(_FOLD_EXTRA))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize(name: str) -> str:
    """Lower-case, fold accents (:func:`fold_accents`) and drop every non-alphanumeric character."""
    return _NON_ALNUM.sub("", fold_accents(name).casefold())


@lru_cache(maxsize=4096)
def tokens(name: str) -> tuple[str, ...]:
    """Split a header into lower-case word tokens (see module docstring); accents are folded first."""
    parts: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", fold_accents(name)):
        if not chunk:
            continue
        parts.extend(p.lower() for p in _CAMEL.split(chunk) if p)
    return tuple(parts)


def matches(name: str, hint: str, mode: str = "token") -> bool:
    """True if ``hint`` (already normalised) matches header ``name``."""
    if mode == "exact":
        return normalize(name) == hint
    if mode == "prefix":
        return normalize(name).startswith(hint)
    toks = tokens(name)
    for i in range(len(toks)):
        if "".join(toks[i:]).startswith(hint):
            return True
    return False


def match_any(name: str, vocab: "Vocabulary") -> str | None:
    """Return the first hint of ``vocab`` that matches ``name`` (or None)."""
    for hint, mode in vocab.entries:
        if matches(name, hint, mode):
            return hint
    return None


@dataclass(frozen=True)
class Vocabulary:
    """Named list of (normalised hint, mode) pairs."""

    name: str
    entries: tuple[tuple[str, str], ...]


def _vocab(
    name: str, token: tuple[str, ...] = (), exact: tuple[str, ...] = (), prefix: tuple[str, ...] = ()
) -> Vocabulary:
    entries = (
        tuple((normalize(h), "exact") for h in exact)
        + tuple((normalize(h), "prefix") for h in prefix)
        + tuple((normalize(h), "token") for h in token)
    )
    return Vocabulary(name, entries)


# --------------------------------------------------------------------------- ID and org-code names
#: D38 (user, 2026-10-11): state student-ID acronyms that are student IDs as bare headers. An attendance or assessment
#: export repeats them on each of a student's rows, so the D7 candidate rules never see them unique, and 6-7 digit keys
#: cannot be told from counts by their values: New York's NYSSIS and New York City's OSIS, Indiana's and Oklahoma's
#: STN, Michigan's UIC, Wisconsin's WISEid, South Carolina's SUNS, Florida's FLEID, KSID, Missouri's MOSIS,
#: Pennsylvania's PAsecureID and Idaho's EDUID. Each is on the default ID list alone and with ID words (``NYSSIS ID``,
#: ``STN_Number``, :func:`default_id_hint`); after ID qualifiers or a student word it makes a person-ID header
#: (``Student NYSSIS``, ``Old_STN``: :func:`student_id_code_hint`); after another word the values decide
#: (``Sibling NYSSIS``: :func:`weak_id_header`, as whole words only: ``Diagnosis`` holds no ``OSIS``).
STATE_ID_CODES: tuple[str, ...] = ("nyssis", "osis", "stn", "uic", "wiseid", "suns", "fleid", "ksid", "mosis",
                                   "pasecureid", "eduid")
#: D38: state data systems whose acronym names a student ID only with an ID word after it (Tennessee's ``EIS ID``,
#: Ohio's ``EMIS ID``, Kansas' ``KIDS ID``, Minnesota's ``MARSS ID``, Arizona's ``SAIS ID``, Mississippi's ``MSIS ID``,
#: Texas' ``PEIMS ID`` and ``TSDS ID``); a bare ``Kids`` counts children and ``Kids_Num`` is a count.
STATE_ID_SYSTEMS: tuple[str, ...] = ("eis", "emis", "kids", "marss", "sais", "msis", "peims", "tsds")
#: D38: the normalised forms of :data:`STATE_ID_CODES` and :data:`STATE_ID_SYSTEMS` that name a student ID: a code
#: alone or with ID words, a system with ``id``/``ident``/``identifier`` (numbering kept: ``STN2``).
_STATE_ID_FORM_RE = re.compile("(?:(?:" + "|".join(STATE_ID_CODES) + ")(?:id|ident|identifier|number|num|no|nbr)*"
                               "|(?:" + "|".join(STATE_ID_SYSTEMS) + r")(?:id|ident|identifier)+)\d*")

#: Design section 10 default ID name list (exact match after normalisation).
DEFAULT_ID_NAMES: tuple[str, ...] = (
    "id", "studentid", "student_id", "stuid", "sid", "studentnumber", "student_number",
    "stateid", "state_id", "statestudentnumber", "state_studentnumber", "sasid", "ssid",
    "localid", "local_id", "studentsdcid", "dcid", "personid", "person_id", "familyident",
    "family_ident", "staffid", "teacherid", "teachernumber", "empid", "employeeid",
    # hardening pass 2 (D10: person IDs are synthesised): Massachusetts' locally assigned student ID and the
    # English unique pupil number; a multi-row table repeats them, so the D7 candidate rules never see them unique
    "lasid", "upn",
    # hardening pass 3: Aeries' permanent ID, Georgia's testing ID, Texas' TSDS unique ID and a bare unique ID
    # (``Perm ID``, ``GTID``, ``TSDS ID``, ``Unique ID``): an attendance table repeats them on many rows, so they were
    # written in full as interpolated numbers and D35 counted rows
    "permid", "permanentid", "permnumber", "gtid", "tsdsid", "tsdsuniqueid", "uniqueid",
    # D38: state student-ID acronyms (STATE_ID_CODES, STATE_ID_SYSTEMS) as bare headers and with ``ID``
    *STATE_ID_CODES, *(c + "id" for c in STATE_ID_CODES if not c.endswith("id")),
    *(s + "id" for s in STATE_ID_SYSTEMS), "marssnumber",
)
DEFAULT_ID_SET: frozenset[str] = frozenset(normalize(n) for n in DEFAULT_ID_NAMES)

#: D10 organisational-code words: real values kept as factors whatever --factor-max is.
ORG_CODE_WORDS: tuple[str, ...] = ("school", "site", "campus", "building", "course", "section", "term", "district",
                                   "exitcode", "entrycode", "ellcode")
#: What may follow the org-code word(s) in the normalised header (besides nothing and digits).
ORG_CODE_SUFFIXES: tuple[str, ...] = ("id", "code", "cd", "name", "nm", "number", "num", "no", "nbr", "key", "desc",
                                      "description", "title", "abbr", "abbreviation")
_ORG_SUFFIX_ALT = "|".join(sorted(ORG_CODE_SUFFIXES, key=len, reverse=True))
_ORG_CODE_RE = re.compile("^(" + "|".join(ORG_CODE_WORDS) + ")(?:" + "|".join(ORG_CODE_WORDS) + ")*(?:"
                          + _ORG_SUFFIX_ALT + r")?\d*$")
#: D39 (user, 2026-10-11): words of long codes that cannot identify a student. They join :data:`ORG_CODE_WORDS` in
#: :func:`org_code_hint` (D10: real values kept, whatever ``--factor-max`` is), so such a column is never a D38 key
#: column: NCES school and district IDs (``NCES``, ``NCESSCH``, ``NCES_SCH_ID``, ``School_NCES_ID``,
#: ``NCES School ID``, ``NCES_ID``, ``NCESID``, ``LEAID``, ``LEA_ID``, ``NCES_LEAID``; the Common Core of Data's
#: ``SCHID``, ``ST_SCHID`` and ``ST_LEAID``), California CDS codes (``CDS``, ``CDS Code``, ``CDSCode``,
#: ``County_District_School_Code``), Texas county-district(-campus) numbers (``CDC``, ``County-District-Campus
#: Number``, ``County_District``; ``Campus Number`` and ``District Number`` are D10 org codes already),
#: general-ledger account codes (``GL_Account``, ``GL Account Number``, ``Account Code``, ``Acct_Cd``), course
#: reference numbers (``CRN``, ``Term_CRN``, ``Course Reference Number``) and ISBNs (``ISBN``, ``ISBN13``,
#: ``ISBN-10``, ``Textbook_ISBN``). A bare ``Account`` (or ``Account_Number``) is not one: a lunch, fee or library
#: account number is a person's. Kept out of :data:`ORG_CODE_WORDS`, whose words are also the header words D34
#: never reads as names (``Lea`` is a given name). Deliberately left out (D39), so they still FAIL under D38:
#: census tracts and ``GEOID`` (where a student lives), provider NPIs (a person's ID) and ``Barcode`` (it may be a
#: student ID card).
KNOWN_CODE_WORDS: tuple[str, ...] = (
    "nces", "ncessch", "schid", "stschid", "cds", "countydistrict", "cdc", "crn", "coursereference", "isbn",
    "textbookisbn", "bookisbn", "glaccount", "glacct", "generalledgeraccount", "ledgeraccount", "accountcode",
    "acctcode", "accountcd", "acctcd")
#: D39: ``LEA`` (a local education agency: a district) only before an ID suffix or ``NCES`` (``LEAID``, ``LEA_Code``,
#: ``ST_LEAID``, ``LEA_NCES_ID``); bare ``Lea`` is a given name, ``Leano`` a surname, and ``LEA_Name`` stays a name
#: header (D12: ``lea`` is no non-person noun).
_LEA_RE = "(?:st)?lea(?=id|ident|code|cd|number|num|nbr|key|nces)"
_KNOWN_CODE_ALT = "|".join((*ORG_CODE_WORDS, *KNOWN_CODE_WORDS, _LEA_RE))
_KNOWN_CODE_RE = re.compile("^(" + _KNOWN_CODE_ALT + ")(?:" + _KNOWN_CODE_ALT + ")*(?:" + _ORG_SUFFIX_ALT + r")?\d*$")


def org_code_hint(name: str) -> str | None:
    """D10 org-code hint (see module docstring): the first org-code word, or None. D39's known non-person codes
    (:data:`KNOWN_CODE_WORDS`, :func:`known_code_hint`) are org-code words here."""
    norm = normalize(name)
    m = _ORG_CODE_RE.match(norm) or _KNOWN_CODE_RE.match(norm)
    return m.group(1) if m else None


def known_code_hint(name: str) -> bool:
    """D39: the header is an org code (:func:`org_code_hint`) only by a word of a known non-person code
    (:data:`KNOWN_CODE_WORDS`: ``NCESSCH``, ``LEA_ID``, ``CDS_Code``, ``Term_CRN``, ``ISBN13``, ``GL_Account``)."""
    norm = normalize(name)
    return not _ORG_CODE_RE.match(norm) and bool(_KNOWN_CODE_RE.match(norm))


#: D39: finance segment words (``Fund``, ``Function``, ``Object``; ``Fund_Code``, ``Obj_Cd``, ``Fund_Function_Object``).
#: Generic words, so the header alone never makes an org code: ``detect.types`` keeps such a column real (D10) only
#: when every value is a code (digits of one width, or digit groups joined by dashes, spaces or two or more dots:
#: ``199``, ``6119``, ``199-11-6119-00``), so a free-text ``Function``, a ``Fund`` of amounts or a ``Func`` of single
#: digits is unchanged.
FINANCE_SEGMENT_WORDS: tuple[str, ...] = ("fund", "function", "func", "object", "obj")
_FINANCE_SEGMENT_RE = re.compile("(" + "|".join(FINANCE_SEGMENT_WORDS) + ")(?:" + "|".join(FINANCE_SEGMENT_WORDS)
                                 + r")*(?:code|cd)?\d*")


def finance_segment_hint(name: str) -> str | None:
    """D39: the first finance segment word of a header made of them (:data:`FINANCE_SEGMENT_WORDS`, then optionally
    ``code``/``cd`` and digits: ``Fund``, ``FUND_CD``, ``Object Code``, ``Fund_Function_Object``), or None
    (``Functional``, ``Objective``, ``Funding``, ``Fund_Balance``, ``Fund_ID``)."""
    m = _FINANCE_SEGMENT_RE.fullmatch(normalize(name))
    return m.group(1) if m else None


#: D10: person-ID words; together with an ID word they make a person ID (synthesised).
PERSON_ID_WORDS: tuple[str, ...] = ("student", "staff", "family", "teacher", "employee", "person", "guardian", "parent",
                                    "pupil", "learner", "child")
#: ``usi``: Ed-Fi's unique student/staff identifier (``StudentUSI``, ``StaffUSI``; hardening pass 3).
ID_WORDS: tuple[str, ...] = ("id", "ident", "identifier", "number", "num", "no", "nbr", "dcid", "key", "usi")
#: Short person words matched as a whole token only (``Stu_Num``, ``Stu #``, ``StuID``, ``Kid_ID``; never
#: ``Study_ID`` or ``Kidney``).
_EXACT_PERSON_ID_WORDS: tuple[str, ...] = ("stu", "kid")
#: K-02: identifier words that make a person-ID header by the header alone only right after a student word that
#: starts the header, ID qualifiers aside (:data:`SUBJECT_PERSON_WORDS`: ``Student_Code``, ``Stu_Code``,
#: ``Pupil Code``, ``Learner Ref``, ``StudentRef``, ``Local Student Code``). After another word the header may name
#: a status (``English Learner Code``, ``Homeless Student Code``): the values decide (:func:`weak_id_header`).
#: Elsewhere they name the code of an attribute (``Student_Status_Code``, ``Grade_Level_Code``), and a staff or
#: teacher code is often the person's initials, whose values D35 decides (``Teacher_Code``).
PERSON_REF_WORDS: tuple[str, ...] = ("code", "cd", "ref", "reference")
#: K-02: student-ID codes. First in the header, ID qualifiers and a student word aside, they make a person-ID header
#: (``SIS ID``, ``USI``, ``SSID_Number``, ``Student_SSID``, ``Old_LASID``); after another word (``CALPADS SSID``,
#: ``EdFi USI``, ``Missing SSID``) or run together with one (``CALPADSSSID``) the values decide
#: (:func:`weak_id_header`). The bare codes on the default ID list (``SSID``, ``LASID``, ``UPN``, ``GTID``) stay
#: with that list (D7, ``--no-default-ids``).
STUDENT_ID_CODES: tuple[str, ...] = ("ssid", "sasid", "lasid", "usi", "sisid", "upn", "gtid")
#: K-02: words for the row's own person, the student (``Student``, ``Stu``, ``Pupil``, ``Learner``, ``Child``,
#: ``Kid``; singular: ``Students`` counts them). Integer or code values under a header holding one are that person's
#: IDs unless the header names a measure (:func:`weak_id_header`).
SUBJECT_PERSON_WORDS: tuple[str, ...] = ("student", "stu", "pupil", "learner", "child", "kid")
#: Student words long enough to be read at the start of a run-together token (``STUDENTSK``, ``studentuid``).
_LONG_SUBJECT_WORDS: tuple[str, ...] = ("student", "pupil", "learner")
#: K-02: qualifiers of an ID that may follow it (``Student Number Local``, ``Student_ID_State``, ``SSID_Old``); they
#: never hide the ID word.
_ID_TRAILING_QUALIFIERS = frozenset({"state", "local", "sis", "perm", "permanent", "unique", "alt", "alternate",
                                     "primary", "secondary", "old", "new", "prior", "previous", "former", "current",
                                     "legacy", "official", "legal"})
#: K-02: qualifiers that may stand before the person or ID word of a person-ID header (``District Student Code``).
_ID_LEADING_QUALIFIERS = _ID_TRAILING_QUALIFIERS | {"district"}
#: A student-ID code (:data:`STUDENT_ID_CODES`, D38's :data:`STATE_ID_CODES`) and ID words, or a D38 state system with
#: an ``id`` word (``EIS ID``, ``KIDS ID``).
_STUDENT_ID_CODE_RE = re.compile("(?:(?:" + "|".join(STUDENT_ID_CODES + STATE_ID_CODES) + ")(?:" + "|".join(ID_WORDS)
                                 + ")*|(?:" + "|".join(STATE_ID_SYSTEMS) + r")(?:id|ident|identifier)+)\d*")
#: K-02: a bracketed group or a spaced-dash suffix qualifies the header (``Student ID (Aeries)``, ``SSID (CALPADS)``,
#: ``Student ID - District``, ``Student (District)``): the ID rules read the header without it.
_BRACKETED_RE = re.compile(r"\s*[\(\[\{][^\(\)\[\]\{\}]*[\)\]\}]")
_DASH_SUFFIX_RE = re.compile("\\s+[-\u2013\u2014]+\\s+.*$")


def _id_core(toks: tuple[str, ...]) -> tuple[str, ...]:
    """``toks`` without trailing numbering (``2``) and trailing ID qualifiers (``Local``, ``State``), at least one
    token kept (K-02)."""
    out = list(toks)
    while len(out) > 1 and (out[-1].isdigit() or out[-1] in _ID_TRAILING_QUALIFIERS):
        out.pop()
    return tuple(out)


@lru_cache(maxsize=4096)
def unbracketed(name: str) -> str:
    """K-02: the header without bracketed groups and a spaced-dash suffix (``Student ID (Aeries)`` -> ``Student ID``,
    ``Student ID - District`` -> ``Student ID``), or the header itself when nothing would be left."""
    s = _DASH_SUFFIX_RE.sub("", _BRACKETED_RE.sub(" ", name)).strip()
    return s if tokens(s) else name


@lru_cache(maxsize=4096)
def id_header_core(name: str) -> tuple[str, ...]:
    """K-02: the header's tokens as the ID rules read them (:func:`unbracketed`, then without trailing numbering and
    ID qualifiers: ``Student_ID_2``, ``Student Number (Local)``); at least one token, or none for an empty header."""
    toks = tokens(unbracketed(name))
    return _id_core(toks) if toks else ()


def _leading_qualifiers(toks: tuple[str, ...]) -> bool:
    return all(t in _ID_LEADING_QUALIFIERS for t in toks)


def person_id_hint(name: str) -> bool:
    """D10 person-ID header, by the header alone (see module docstring). A trailing ``#`` is an ID word
    (``Student #``, ``Stu #``: hardening pass 3).

    K-02: a bracketed or spaced-dash qualifier, trailing numbering and ID qualifiers never hide the ID word
    (:func:`id_header_core`: ``Student_ID_2``, ``Student Number (Local)``, ``Student ID (Aeries)``,
    ``Student ID - District``); a code or reference word right after a student word that starts the header is an
    ID word (:data:`PERSON_REF_WORDS`: ``Student Code``, ``Stu_Code``, ``Pupil Code``, ``Learner Ref``); and a
    student-ID code that starts the header is one (:func:`student_id_code_hint`: ``SIS ID``, ``USI``,
    ``Student_SSID``). Other headers with a student word or a student-ID code (``English Learner Code``,
    ``CALPADS SSID``, ``Missing SSID``, ``Student_SK``) are IDs only by their values (:func:`weak_id_header`), and
    flags and counts (``Has_Student_Code``, ``Num_SSID``) never are.
    """
    toks = tokens(name)
    if not toks:
        return False

    def person(t: str) -> bool:
        return t in _EXACT_PERSON_ID_WORDS or any(t.startswith(pw) for pw in PERSON_ID_WORDS)

    core = id_header_core(name)
    last = core[-1]
    if last in ID_WORDS and any(person(t) for t in core[:-1]):
        return True
    if name.rstrip().endswith("#") and any(person(t) for t in toks):
        return True
    if any(last == pw + iw for pw in PERSON_ID_WORDS + _EXACT_PERSON_ID_WORDS for iw in ID_WORDS):
        return True
    if toks[0] in _FLAG_COUNT_PREFIXES:
        return False
    lead = core[:-1]
    # the student's own code or reference (a staff or teacher code is often initials: D35 decides those values)
    if last in PERSON_REF_WORDS and lead and lead[-1] in SUBJECT_PERSON_WORDS and _leading_qualifiers(lead[:-1]):
        return True  # Student_Code, Pupil Code, Learner Ref (Student_Status_Code is a status code)
    if _leading_qualifiers(lead) and any(last == pw + rw for pw in SUBJECT_PERSON_WORDS for rw in PERSON_REF_WORDS):
        return True  # StudentCode, StuRef
    return student_id_code_hint(name)


def student_id_code_hint(name: str) -> bool:
    """K-02: a student-ID code (:data:`STUDENT_ID_CODES`) that starts the header (ID qualifiers and a student word
    aside), followed only by ID words and numbering (``SIS ID``, ``SSID_Number``, ``Student_SSID``, ``USI``,
    ``Old_LASID``, ``USI (EdFi)``); not a bare code on the default ID list (``SSID``, ``LASID``:
    :func:`default_id_hint`), never a flag or count (``Has_SSID``), and not after another word (``CALPADS SSID``,
    ``Missing SSID``: :func:`weak_id_header`)."""
    core = id_header_core(name)
    toks = tokens(name)
    if not core or toks[0] in _FLAG_COUNT_PREFIXES or "".join(core) in DEFAULT_ID_SET:
        return False
    for i in range(len(core)):
        if not all(t in _ID_LEADING_QUALIFIERS or t in SUBJECT_PERSON_WORDS for t in core[:i]):
            return False
        if _STUDENT_ID_CODE_RE.fullmatch("".join(core[i:])):
            return True
    return False


#: K-02: the default list's own ID stems take any ID word (``ID Number``, ``ID_No``, ``IDNum``, ``Identification
#: Number``, ``Perm_No``, ``Permanent Number``, ``Perm #``): the list itself spells ``id``, ``permid``,
#: ``permnumber`` and ``permanentid``.
_DEFAULT_ID_FORM_RE = re.compile(r"(?:id|ident|identifier|identification|perm|permanent)"
                                 r"(?:id|ident|identifier|number|num|no|nbr|key)+\d*")
_DEFAULT_ID_STEMS = frozenset({"id", "ident", "identifier", "identification", "perm", "permanent"})


def default_id_hint(name: str) -> bool:
    """D7 default ID name list (section 10, exact after normalisation), plus its ID stems with any other ID word
    (K-02: ``ID Number``, ``Perm_No``, ``Perm #``); a bracketed or spaced-dash qualifier is read away
    (``SSID (CALPADS)``, ``Student_ID (PowerSchool)``)."""
    for norm in dict.fromkeys((normalize(name), normalize(unbracketed(name)))):
        if norm in DEFAULT_ID_SET or _DEFAULT_ID_FORM_RE.fullmatch(norm) or _STATE_ID_FORM_RE.fullmatch(norm):
            return True  # D38: a state student-ID acronym with ID words too (NYSSIS ID, STN_Number, EIS ID)
    if not name.rstrip().endswith("#"):
        return False
    stem = normalize(name).rstrip("0123456789")
    return stem in _DEFAULT_ID_STEMS or bool(_STATE_ID_FORM_RE.fullmatch(stem))  # Perm #, NYSSIS #


#: K-02: words of a header that names a measure of the student or an attribute counted in numbers, never the
#: student's ID (``Student_Count``, ``Student Scale Score``, ``Student_Teacher_Ratio``, ``Student_Age``,
#: ``Mean Scale Score Student``). Fail-safe: a measure word missing here costs fidelity (the column gets synthetic
#: IDs of the same shape), never privacy.
MEASURE_WORDS: frozenset[str] = frozenset({
    "count", "cnt", "total", "sum", "amount", "amt", "cost", "price", "fee", "paid", "charged", "owed", "balance",
    "budget", "spend", "spent", "expenditure", "expense", "revenue", "salary", "wage", "income", "score", "scale",
    "raw", "rit", "lexile", "quantile", "percentile", "pctl", "sgp", "stanine", "nce", "mean", "avg", "average",
    "median", "min", "max", "ratio", "rate", "pct", "percent", "percentage", "rank", "gpa", "credit", "hours", "hrs",
    "hour", "minutes", "mins", "minute", "seconds", "secs", "days", "day", "weeks", "months", "years", "year", "yr",
    "age", "weight", "height", "size", "enrolled", "enrollment", "enrolment", "population", "capacity", "absences",
    "absent", "tardies", "tardy", "attendance", "points", "pts", "growth", "fte", "distance", "miles", "grade",
    "grades", "level", "levels", "wcpm", "wpm", "pages", "words"})
#: A student word after one of these is a divisor or an actor (``Cost per Student``, ``Amount Paid by Student``).
_PER_WORDS = frozenset({"per", "by", "each", "every"})
#: K-02: ID words a header may end with when it holds keys (``Admission No``, ``Roll_Number``, ``Aeries ID``,
#: ``Lunch PIN``, ``Learner UID``, ``Aeries Perm``, a warehouse's surrogate key ``Student_SK``).
_ID_HEADER_END_WORDS = frozenset(ID_WORDS + ("nr", "ref", "reference", "pin", "uid", "guid", "uuid", "sk", "perm"))


def _measure_token(t: str) -> bool:
    """A measure word (:data:`MEASURE_WORDS`) or its plural; D39: a known code word is never one (``NCES`` is the
    National Center for Education Statistics' school ID, no plural of ``NCE``)."""
    return t in MEASURE_WORDS or (t.endswith("s") and t[:-1] in MEASURE_WORDS and t not in KNOWN_CODE_WORDS)


def measure_header(name: str) -> bool:
    """K-02: a header word names a measure (:data:`MEASURE_WORDS`: ``Amount``, ``Total_Cost``, ``Scale Score``) or the
    header is a count or flag (``Num_Students``, ``Has_ID``)."""
    toks = tokens(name)
    return bool(toks) and (toks[0] in _FLAG_COUNT_PREFIXES or any(_measure_token(t) for t in toks))


def _subject_token(t: str) -> bool:
    """A header token naming the student: a student word (:data:`SUBJECT_PERSON_WORDS`) or a long student word run
    together with more (``studentsk``, ``studentuid``), never a plural (``Students``, ``studentsenrolled``) or a
    run-together measure (``studentcount``)."""
    if t in SUBJECT_PERSON_WORDS:
        return True
    for w in _LONG_SUBJECT_WORDS:
        if t.startswith(w):
            rest = t[len(w):]
            if len(rest) < 2 or not rest.isalpha():
                return False  # students, pupils, student2
            if rest.startswith("s") and len(rest) > 2:
                return False  # a plural run together with more: studentsenrolled (studentsk is one student's key)
            return not any(rest.startswith(m) for m in MEASURE_WORDS if len(m) >= 3)  # studentcount, studentage
    return False


@lru_cache(maxsize=4096)
def weak_id_header(name: str) -> str | None:
    """K-02, fail-safe: a header that does not name an ID by itself (:func:`person_id_hint`,
    :func:`default_id_hint`) but whose ID-like values make the column an ID (``detect.ids.weak_id_detection``):

    * ``'person'``: a student word (:data:`SUBJECT_PERSON_WORDS`) whatever follows it (``Student``, ``Stu``,
      ``Student_SK``, ``Student UID``, ``Student PIN``, ``Student Lunch PIN``, ``Student (District)``,
      ``Student_ID_Status``, ``English Learner Code``), unless the header names a measure (:data:`MEASURE_WORDS`:
      ``Student_Count``, ``Student Scale Score``, ``Cost per Student``) or the student's organisation (an org-code
      word after the student word and before any ID word: ``Student_School``, ``Student_District_ID``);
    * ``'code'``: a student-ID code inside the header (``CALPADS SSID``, ``EdFi USI``, ``CALPADSSSID``,
      ``Missing SSID``), or a D38 state student-ID acronym as whole words (``Sibling NYSSIS``,
      :func:`state_id_code_in`);
    * ``'id'``: a header that ends in an ID word (``Admission No``, ``Roll_Number``, ``Aeries ID``, ``KIDS ID``,
      ``Lunch Number``, ``Statewide ID``, ``Badge #``);

    None for flags and counts (``Has_Student_Code``, ``Num_Absences``) and every other header. The values decide:
    a status code (``EL``, ``RFEP``), a flag (``Y``/``N``) or a short code list never makes an ID.
    """
    toks = tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES:
        return None
    core = id_header_core(name)
    if any(_measure_token(t) for t in core):
        return None
    for i, t in enumerate(core):
        if not _subject_token(t):
            continue
        if any(u in _PER_WORDS for u in core[:i]):
            return None  # Cost per Student, Amount Paid by Student: a measure
        for u in core[i + 1:]:
            if u in _ID_HEADER_END_WORDS or u in PERSON_REF_WORDS:
                break  # Student_ID_District: what follows the ID word qualifies the ID
            if _ORG_CODE_RE.match(u):
                # Student_School, Student_District_ID: the student's organisation (D10). Fail-safe: not D39's known
                # codes, so 'Student_CRN' (a customer reference number may be a person's) stays the student's
                return None
        return "person"
    if any(t.endswith(c) for t in core for c in STUDENT_ID_CODES) or state_id_code_in(core):
        return "code"
    last = tokens(unbracketed(name))[-1]  # 'perm' is an ID word last ('Aeries Perm'), a qualifier after another
    if core[-1] in _ID_HEADER_END_WORDS or last in _ID_HEADER_END_WORDS or unbracketed(name).rstrip().endswith("#"):
        return "id"
    return None


def state_id_code_in(toks: tuple[str, ...]) -> bool:
    """D38: a run of whole header tokens names a state student ID (:data:`STATE_ID_CODES` alone or with ID words,
    :data:`STATE_ID_SYSTEMS` with an ``id`` word: ``Sibling NYSSIS``, ``Prior WISEid``, ``Other EIS ID``). Whole tokens
    only: ``Diagnosis`` holds no ``OSIS`` and ``Quick`` no ``UIC``."""
    return any(_STATE_ID_FORM_RE.fullmatch("".join(toks[i:j]))
               for i in range(len(toks)) for j in range(i + 1, len(toks) + 1))


def person_head_hint(name: str) -> bool:
    """K-02: the header names the row's own person (a :data:`SUBJECT_PERSON_WORDS` word as its last word, numbering,
    ID qualifiers and bracketed qualifiers aside: ``Student``, ``Stu``, ``Pupil``, ``Local_Student``, ``Child_2``,
    ``Student (District)``). A plural (``Students``), a count or flag (``Num_Student``, ``Has_Child``), a measure
    (``Cost per Student``) or another last word (``Student_Count``, ``Student_Name``) is not."""
    core = id_header_core(name)
    return bool(core) and core[-1] in SUBJECT_PERSON_WORDS and weak_id_header(name) == "person"


def subject_id_hint(name: str) -> bool:
    """K-02 (D35 ``profile.entity.id_role``): an ID header of the row's own person beyond the person-word + ID-word
    forms: a student-ID code in the header (``CALPADS SSID``, ``EdFi USI``, ``SIS ID``), a header naming the student
    (:func:`person_head_hint`: ``Kid``, ``Stu``), or a student word with one ID or code word (``Kid ID``,
    ``Kid_Code``, ``Student_SK``, ``Student PIN``, ``Learner UID``)."""
    if student_id_code_hint(name) or person_head_hint(name) or weak_id_header(name) == "code":
        return True
    toks = tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES:
        return False
    core = id_header_core(name)
    return (len(core) == 2 and core[0] in SUBJECT_PERSON_WORDS
            and (core[1] in _ID_HEADER_END_WORDS or core[1] in PERSON_REF_WORDS))


def id_word_header(name: str) -> str | None:
    """K-02: the ID word a header ends with (``Admission No``, ``Roll_Number``, ``Aeries ID``, ``Network ID``,
    ``Candidate Ref``, ``Badge #``; numbering, ID qualifiers and bracketed qualifiers aside), or None. Flags and
    counts (``Num_Absences``, ``Has_ID``) have none."""
    toks = tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES:
        return None
    if unbracketed(name).rstrip().endswith("#"):
        return "#"
    last = id_header_core(name)[-1]
    return last if last in _ID_HEADER_END_WORDS else None


#: D12: role words whose column holds a person's name ("Teacher", "Section_Teacher", "EmergencyContact").
PERSON_ROLE_WORDS: tuple[str, ...] = (
    "teacher", "counselor", "counsellor", "advisor", "adviser", "instructor", "principal", "guardian", "parent",
    "mother", "father", "contact", "student", "staff", "caseworker", "coach", "tutor", "mentor",
)
#: R2-04: more words naming a person in school, health and case data ("role": the column holds that person's
#: name; "actor": ``<verb>_By`` holds the person who did it: ``Referred_By``, ``EnteredBy``). One word list of
#: more than 50 entries, so the leak scanner treats it as data, never as text the tool writes.
_PERSON_WORD_KINDS: dict[str, str] = {
    "tchr": "role", "mom": "role", "dad": "role", "grandparent": "role", "grandmother": "role",
    "grandfather": "role", "stepparent": "role", "stepmother": "role", "stepfather": "role", "caregiver": "role",
    "pupil": "role", "learner": "role", "spouse": "role", "sibling": "role", "employee": "role",
    "supervisor": "role", "physician": "role", "doctor": "role", "nurse": "role", "therapist": "role",
    "psychologist": "role", "driver": "role", "proctor": "role", "examiner": "role", "evaluator": "role",
    "interviewer": "role", "chaperone": "role", "volunteer": "role", "paraprofessional": "role", "aide": "role",
    "victim": "role", "witness": "role", "offender": "role", "referrer": "role", "respondent": "role",
    "applicant": "role", "patient": "role", "alias": "role", "aka": "role", "entered": "actor", "created": "actor",
    "updated": "actor", "modified": "actor", "changed": "actor", "edited": "actor", "added": "actor",
    "recorded": "actor", "reported": "actor", "referred": "actor", "submitted": "actor", "approved": "actor",
    "reviewed": "actor", "verified": "actor", "signed": "actor", "completed": "actor", "assigned": "actor",
    "taught": "actor", "administered": "actor", "scored": "actor", "graded": "actor", "issued": "actor",
    "logged": "actor", "requested": "actor", "authorized": "actor", "authorised": "actor", "observed": "actor",
    "evaluated": "actor", "prepared": "actor", "witnessed": "actor", "interviewed": "actor", "supervised": "actor",
    "processed": "actor", "closed": "actor", "opened": "actor", "resolved": "actor",
}
PERSON_ROLE_WORDS += tuple(w for w, kind in _PERSON_WORD_KINDS.items() if kind == "role")
#: Verbs of ``<verb>_by`` headers whose column holds the person who did it.
PERSON_ACTOR_VERBS: frozenset[str] = frozenset(w for w, kind in _PERSON_WORD_KINDS.items() if kind == "actor")
#: R3-03: plural words for the people a cell lists (``Participants``, ``Recipients``: names or lists of them).
#: Plural only: a singular ``Participant`` header is usually a status (``Yes``/``No``/``Withdrawn``).
PERSON_ROLE_WORDS += ("participants", "recipients", "attendees", "invitees", "assignees", "nominees", "awardees",
                      "honorees", "approvers", "reviewers")
#: Trailing header tokens that only number or rank a person column (``Guardian1``, ``Guardian_2``,
#: ``Contact #3``, ``Parent 1st``, ``Parent(s)``, ``Guardian_Primary``, ``Teacher_of_Record``, ``Parent_A``):
#: stripped before the role-word and name-word rules (R2-04).
_ROLE_TRAILERS = frozenset({"primary", "secondary", "tertiary", "alt", "alternate", "s"})
_ORDINAL_SUFFIXES = frozenset({"st", "nd", "rd", "th"})
#: Nouns before "name" that do not denote a person (``school_name`` is not personal data).
NON_PERSON_NOUNS: tuple[str, ...] = (
    "school", "course", "section", "district", "site", "building", "campus", "program", "programme", "term", "class",
    "subject", "department", "dept", "test", "assessment", "exam", "file", "sheet", "table", "product", "item",
    "event", "activity", "team", "club", "organization", "organisation", "org", "company", "vendor", "field",
    "column", "col", "variable", "var", "category", "group", "state", "city", "country", "county", "street",
    "month", "day", "period", "room", "grade", "level", "award", "plan", "service", "agency", "facility",
    "location", "place", "project", "model", "brand", "unit", "track", "cohort", "pathway", "lunch", "meal",
    "bus", "route", "sport", "language", "race", "ethnicity", "title", "job", "position", "role", "region", "zone",
    "area", "host", "server", "domain", "app", "application", "device", "report", "form", "type", "status",
)


_FLAG_COUNT_PREFIXES = frozenset({"is", "has", "was", "num", "no", "n", "count", "total", "pct", "percent",
                                  # R3-04: Spanish/Portuguese flags and counts (Tiene_Tutor, Cantidad_Alumnos)
                                  "tiene", "tem", "cantidad", "numero", "nro", "quantidade", "qtd"})


_ROLE_SET = frozenset(PERSON_ROLE_WORDS)


def is_role_word(tok: str) -> bool:
    """A person-role token, singular or plural (``teacher``, ``Teachers``, ``coaches``)."""
    return tok in _ROLE_SET or (tok.endswith("s") and (tok[:-1] in _ROLE_SET or (
        tok.endswith("es") and tok[:-2] in _ROLE_SET)))


def strip_role_trailers(toks: tuple[str, ...]) -> tuple[str, ...]:
    """``toks`` without trailing tokens that only number or rank a person column (R2-04): digits (``1``,
    ``02``), ordinal endings after digits (``1st``), ``primary``/``secondary``/``alt`` ..., a plural ``(s)``,
    ``of record``, and one letter right after a role word (``Parent_A``)."""
    out = list(toks)
    while len(out) > 1:
        last = out[-1]
        if last.isdigit() or last in _ROLE_TRAILERS:
            out.pop()
        elif last in _ORDINAL_SUFFIXES and out[-2].isdigit():
            out.pop()
        elif last == "record" and out[-2] == "of" and len(out) > 2:
            del out[-2:]
        elif len(last) == 1 and last.isalpha() and is_role_word(out[-2]):
            out.pop()
        else:
            break
    return tuple(out)


def person_name_hint(name: str) -> bool:
    """D12 person-name header beyond the fixed PII phrases (see module docstring).

    Trailing numbering and ranking (:func:`strip_role_trailers`: ``Guardian1``, ``Contact_2``, ``Parent(s)``,
    ``Guardian_Primary``, ``Name_2``, ``MotherName1``) never hides the role or name word (R2-04).
    """
    toks = tokens(name)
    if toks and toks[0] in _FLAG_COUNT_PREFIXES:
        return False  # is_parent, num_teachers, count_student: flags and counts, not names
    core = strip_role_trailers(toks)
    if not core:
        return False
    if is_role_word(core[-1]):
        return True
    if len(core) >= 2 and core[-1] == "by" and core[-2] in PERSON_ACTOR_VERBS:
        return True  # Referred_By, EnteredBy: the person who did it
    n = "".join(core)
    if n in ("name", "names", "fullname"):
        return True  # Name1, Name_2, Names
    for suffix in ("names", "name"):
        if n.endswith(suffix) and len(n) > len(suffix):
            prefix = n[: -len(suffix)]
            return not any(prefix.endswith(w) for w in NON_PERSON_NOUNS)
    return False


#: Words that may stand before a name part (``Student_First``, ``Guardian1Last``, ``LegalFirst``): person
#: roles plus a few qualifiers of a person's name.
NAME_PART_QUALIFIERS: tuple[str, ...] = tuple(dict.fromkeys(
    PERSON_ROLE_WORDS + PERSON_ID_WORDS + (
        "stu", "pupil", "child", "learner", "emp", "member", "user", "spouse", "sibling", "applicant", "patient",
        "client", "legal", "preferred", "pref", "nick", "alias", "maiden", "birth", "emergency", "emerg", "primary",
        "secondary", "alt", "alternate", "other", "household", "head", "hh", "ec", "sped", "case", "manager",
        "homeroom", "hr", "physician", "doctor", "nurse", "provider", "driver", "owner", "author", "reporter",
    )))
#: Name parts -> PII_HINTS key (``LastFirst`` / ``Last_First`` hold "Last, First": a full name).
_NAME_PARTS: dict[str, str] = {"first": "first_name", "given": "first_name", "fname": "first_name",
                               "last": "last_name", "surname": "last_name", "lname": "last_name",
                               "middle": "middle_name", "mname": "middle_name",
                               "lastfirst": "full_name", "firstlast": "full_name",
                               "lastfirstmiddle": "full_name", "firstmiddlelast": "full_name"}
_NAME_PART_RE = re.compile(
    "^(?:(?:" + "|".join(sorted(NAME_PART_QUALIFIERS, key=len, reverse=True)) + r")\d*)*"
    r"(?:name|nm)?"
    "(?P<part>" + "|".join(sorted(_NAME_PARTS, key=len, reverse=True)) + r")"
    r"(?:nm|nme|name|names|initial|init)?\d*$")


def person_name_part_hint(name: str) -> str | None:
    """D12 name-part header (R1-05): the PII_HINTS name key of ``Student_First``, ``stu_last_nm``,
    ``Guardian1First``, ``First``, ``LASTFIRST`` (-> full_name), ``LegalLast``; or None.

    The normalised header must be person-role/qualifier words (each optionally numbered), then optionally
    ``name`` (``Name_First``, ``Name (Last, First)``: R2-04), then a name part (first, last, middle, given,
    surname, fname, lname, mname, lastfirst ...), then optionally ``nm``/``name``/``initial`` and digits;
    trailing numbering or ranking is ignored (``Guardian_First_Primary``). ``first_day``, ``last_login``,
    ``LastSchool`` and ``is_first`` do not match.
    """
    toks = tokens(name)
    if toks and toks[0] in _FLAG_COUNT_PREFIXES:
        return None
    m = _NAME_PART_RE.match("".join(strip_role_trailers(toks)))
    return _NAME_PARTS[m.group("part")] if m else None


# --------------------------------------------------------------------------- R3-04: names beyond English
#: Accent-folded header words of person-name headers in the other languages of school data exports, by kind:
#: "given" (``Nombre``, ``Prénom``, ``Nome``: the given name(s)), "family" (``Apellido``, ``Sobrenome``,
#: ``Nom``: the surname(s)), "full" (``Naam``, ``Nominativo``), "role" (a person's role: ``Padre``,
#: ``Responsável``, ``Docente``; "role_accent": a role alone only with its accent, ``Mãe`` but not ``MAE``),
#: qualifiers ("q_full" ``completo``, "q_middle" ``segundo``, "q_first" ``primer``/``pila``, "q" other
#: qualifiers such as ``paterno``, ``legal``, ``emergencia``), "connective" (articles and prepositions,
#: ignored: ``Nombre del Padre``), "flag" (a leading flag/count word: ``Tiene_Tutor``, ``Cantidad_Alumnos``)
#: and "thing" (non-person nouns: ``Nombre_Escuela`` is the school's name). One word list of more than 50
#: entries, so the leak scanner treats it as data, never as text the tool writes.
_LOCAL_WORDS: dict[str, str] = {
    # given names: es, pt, it, fr, de, nl, tl, ht
    "nombre": "given", "nombres": "given", "nome": "given", "nomes": "given", "prenom": "given",
    "prenoms": "given", "prenome": "given", "prenomes": "given", "vorname": "given", "vornamen": "given",
    "rufname": "given", "voornaam": "given", "voornamen": "given", "pangalan": "given", "prenon": "given",
    # surnames: es, pt, it, fr, de, nl, tl, ht
    "apellido": "family", "apellidos": "family", "sobrenome": "family", "sobrenomes": "family",
    "apelido": "family", "apelidos": "family", "cognome": "family", "cognomi": "family", "nom": "family",
    "noms": "family", "nachname": "family", "familienname": "family", "zuname": "family", "achternaam": "family",
    "apelyido": "family", "siyati": "family",
    # a whole name: nl, it, so
    "naam": "full", "nominativo": "full", "magac": "full", "magaca": "full",
    # roles: es
    "padre": "role", "padres": "role", "madre": "role", "madres": "role", "tutora": "role", "tutoras": "role",
    "tutores": "role", "encargado": "role", "encargada": "role", "encargados": "role", "acudiente": "role",
    "acudientes": "role", "apoderado": "role", "apoderada": "role", "representante": "role",
    "responsable": "role", "responsables": "role", "maestro": "role", "maestra": "role", "maestros": "role",
    "profesor": "role", "profesora": "role", "profesores": "role", "docente": "role", "docentes": "role",
    "consejero": "role", "consejera": "role", "orientador": "role", "orientadora": "role", "director": "role",
    "directora": "role", "alumno": "role", "alumna": "role", "alumnos": "role", "alumnas": "role",
    "estudiante": "role", "estudiantes": "role", "contacto": "role", "contactos": "role", "abuelo": "role",
    "abuela": "role", "hermano": "role", "hermana": "role", "medico": "role", "enfermera": "role",
    "enfermero": "role", "jefe": "role", "jefa": "role", "titular": "role", "coordinador": "role",
    "coordinadora": "role", "psicologo": "role", "psicologa": "role", "trabajador": "role",
    # roles: pt
    "pai": "role", "mae": "role_accent", "responsavel": "role", "responsaveis": "role", "encarregado": "role",
    "encarregada": "role", "professora": "role", "professores": "role", "aluno": "role", "aluna": "role",
    "alunos": "role", "estudante": "role", "estudantes": "role", "contato": "role", "coordenador": "role",
    "coordenadora": "role",
    # roles: fr, it, de, tl
    "pere": "role", "mere": "role", "tuteur": "role", "tutrice": "role", "eleve": "role", "eleves": "role",
    "etudiant": "role", "etudiante": "role", "enseignant": "role", "enseignante": "role",
    "professeur": "role", "conseiller": "role", "conseillere": "role", "directeur": "role",
    "directrice": "role", "medecin": "role", "infirmier": "role", "infirmiere": "role", "genitore": "role",
    "genitori": "role", "tutore": "role", "insegnante": "role", "alunno": "role", "alunna": "role",
    "studente": "role", "vater": "role", "mutter": "role", "eltern": "role", "lehrer": "role",
    "lehrerin": "role", "schuler": "role", "schueler": "role", "schulerin": "role", "magulang": "role",
    "guro": "role", "estudyante": "role",
    # qualifiers
    "completo": "q_full", "completa": "q_full", "complet": "q_full", "complete": "q_full", "voller": "q_full",
    "vollstandiger": "q_full", "volledige": "q_full", "segundo": "q_middle", "segunda": "q_middle",
    "seconde": "q_middle", "deuxieme": "q_middle", "medio": "q_middle", "meio": "q_middle", "autres": "q_middle",
    "outros": "q_middle", "otros": "q_middle", "zweiter": "q_middle", "gitnang": "q_middle", "primer": "q_first",
    "primero": "q_first", "primera": "q_first", "primeiro": "q_first", "primeira": "q_first",
    "premier": "q_first", "pila": "q_first", "propio": "q_first", "usuel": "q_first", "unang": "q_first",
    "paterno": "q", "paterna": "q", "materno": "q", "materna": "q", "famille": "q", "naissance": "q",
    "usage": "q", "jeune": "q", "fille": "q", "soltera": "q", "casada": "q", "ultimo": "q", "ultima": "q",
    "oficial": "q", "emergencia": "q", "urgencia": "q", "urgence": "q", "principal": "q", "secundario": "q",
    "alterno": "q", "adicional": "q", "otro": "q", "otra": "q", "autre": "q", "outro": "q", "outra": "q",
    "familia": "q", "legal": "q",
    # articles, prepositions and conjunctions (ignored)
    "de": "connective", "del": "connective", "da": "connective", "do": "connective", "das": "connective",
    "dos": "connective", "du": "connective", "des": "connective", "d": "connective", "l": "connective",
    "la": "connective", "el": "connective", "le": "connective", "les": "connective", "los": "connective",
    "las": "connective", "lo": "connective", "y": "connective", "e": "connective", "et": "connective",
    "o": "connective", "ou": "connective", "u": "connective", "al": "connective", "au": "connective",
    "aux": "connective", "di": "connective", "della": "connective", "dello": "connective", "degli": "connective",
    "und": "connective", "ng": "connective", "ak": "connective",
    # leading flag and count words (``Tiene_Tutor``, ``Cantidad_Alumnos``, ``Numero_Docentes``)
    "tiene": "flag", "tem": "flag", "es": "flag", "esta": "flag", "cantidad": "flag", "numero": "flag",
    "nro": "flag", "nb": "flag", "quantidade": "flag", "qtd": "flag",
    # things named like people: es, pt, fr, it, de (``Nombre_Escuela``, ``Nome_Turma``, ``Nom_Ecole``)
    "escuela": "thing", "escuelas": "thing", "colegio": "thing", "plantel": "thing", "instituto": "thing",
    "institucion": "thing", "liceo": "thing", "centro": "thing", "cct": "thing", "sede": "thing",
    "curso": "thing", "cursos": "thing", "materia": "thing", "asignatura": "thing", "clase": "thing",
    "grupo": "thing", "seccion": "thing", "grado": "thing", "nivel": "thing", "programa": "thing",
    "distrito": "thing", "municipio": "thing", "estado": "thing", "pais": "thing",
    "provincia": "thing", "departamento": "thing", "region": "thing", "zona": "thing", "edificio": "thing",
    "aula": "thing", "salon": "thing", "periodo": "thing", "semestre": "thing", "trimestre": "thing",
    "ciclo": "thing", "turno": "thing", "jornada": "thing", "archivo": "thing", "hoja": "thing",
    "tabla": "thing", "columna": "thing", "categoria": "thing", "tipo": "thing", "producto": "thing",
    "evento": "thing", "actividad": "thing", "equipo": "thing", "organizacion": "thing", "empresa": "thing",
    "proveedor": "thing", "servicio": "thing", "agencia": "thing", "ubicacion": "thing", "lugar": "thing",
    "proyecto": "thing", "modelo": "thing", "unidad": "thing", "ruta": "thing", "deporte": "thing",
    "idioma": "thing", "lengua": "thing", "titulo": "thing", "puesto": "thing", "premio": "thing",
    "examen": "thing", "prueba": "thing", "evaluacion": "thing", "barrio": "thing", "localidad": "thing",
    "comuna": "thing", "facultad": "thing", "universidad": "thing", "beca": "thing", "documento": "thing",
    "formulario": "thing", "reporte": "thing", "informe": "thing", "especialidad": "thing", "taller": "thing",
    "entidad": "thing", "delegacion": "thing", "parroquia": "thing", "iglesia": "thing", "escola": "thing",
    "escolas": "thing", "disciplina": "thing", "turma": "thing", "bairro": "thing",
    "arquivo": "thing", "planilha": "thing", "tabela": "thing", "coluna": "thing", "produto": "thing",
    "atividade": "thing", "equipe": "thing", "clube": "thing", "organizacao": "thing", "servico": "thing",
    "unidade": "thing", "regiao": "thing", "projeto": "thing", "prova": "thing", "exame": "thing",
    "avaliacao": "thing", "funcao": "thing", "lingua": "thing", "instituicao": "thing", "modalidade": "thing",
    "ecole": "thing", "ecoles": "thing", "etablissement": "thing", "lycee": "thing", "cours": "thing",
    "matiere": "thing", "classe": "thing", "groupe": "thing", "niveau": "thing", "programme": "thing",
    "commune": "thing", "departement": "thing", "pays": "thing", "fichier": "thing", "feuille": "thing",
    "tableau": "thing", "colonne": "thing", "categorie": "thing", "produit": "thing", "evenement": "thing",
    "activite": "thing", "organisation": "thing", "entreprise": "thing", "societe": "thing", "unite": "thing",
    "projet": "thing", "epreuve": "thing", "langue": "thing", "titre": "thing", "rapport": "thing",
    "filiere": "thing", "specialite": "thing", "academie": "thing", "scuola": "thing", "istituto": "thing",
    "corso": "thing", "sezione": "thing", "plesso": "thing", "comune": "thing", "regione": "thing",
    "schule": "thing", "klasse": "thing", "kurs": "thing", "fach": "thing", "gruppe": "thing", "stadt": "thing",
    "gemeinde": "thing", "bezirk": "thing",
    # whole folded headers (articles and numbering dropped, words joined) of a name part: "=<PII_HINTS key>"
    "hoten": "=full_name", "hovaten": "=full_name", "hodem": "=last_name", "holot": "=last_name",
    "tendem": "=middle_name", "tenlot": "=middle_name", "appaterno": "=last_name", "apmaterno": "=last_name",
    "appat": "=last_name", "apmat": "=last_name", "apepat": "=last_name", "apemat": "=last_name",
    "apepaterno": "=last_name", "apematerno": "=last_name", "nompere": "=last_name",
    # other PII kinds, whole words ("pii:<key>": ``Teléfono``, ``Correo``, ``Dirección``, ``Ciudad``, ``CEP``,
    # ``Nacimiento``, ``CURP``; ``Movilidad`` is no phone) and phrases found anywhere in the joined header
    # ("pii=<key>": ``Código Postal``, ``Seguro Social``)
    "telefono": "pii:phone", "telefonos": "pii:phone", "telefone": "pii:phone", "telefones": "pii:phone",
    "celular": "pii:phone", "celulares": "pii:phone", "movil": "pii:phone", "cellulaire": "pii:phone",
    "telemovel": "pii:phone", "telepono": "pii:phone", "correo": "pii:email", "correos": "pii:email",
    "courriel": "pii:email", "direccion": "pii:address", "domicilio": "pii:address", "endereco": "pii:address",
    "adresse": "pii:address", "morada": "pii:address", "calle": "pii:address", "rua": "pii:address",
    "logradouro": "pii:address", "ciudad": "pii:city", "cidade": "pii:city", "ville": "pii:city",
    "cep": "pii:zip", "nacimiento": "pii:dob", "nascimento": "pii:dob", "naissance": "pii:dob",
    "geburtsdatum": "pii:dob", "curp": "pii:ssn", "cpf": "pii:ssn", "nss": "pii:ssn",
    "codigopostal": "pii=zip", "codpostal": "pii=zip", "codepostal": "pii=zip", "segurosocial": "pii=ssn",
    "securitesociale": "pii=ssn", "segurancasocial": "pii=ssn",
}
#: Non-ASCII headers (NFC, case-folded, spaces/punctuation/digits dropped) of name columns: Vietnamese with its
#: diacritics (bare ``Ho``/``Ten`` are too ambiguous without them), Chinese, Japanese, Korean.
_SCRIPT_NAME_HEADERS: dict[str, str] = {
    "họ": "last_name", "tên": "first_name", "họtên": "full_name", "họvàtên": "full_name", "tênđệm": "middle_name",
    "tênlót": "middle_name", "họđệm": "last_name", "họlót": "last_name",
    "姓名": "full_name", "名字": "full_name", "氏名": "full_name", "名前": "full_name", "성명": "full_name",
    "이름": "full_name", "姓": "last_name", "姓氏": "last_name", "苗字": "last_name", "성": "last_name",
    "성씨": "last_name", "名": "first_name", "家长": "full_name", "家長": "full_name", "父亲": "full_name",
    "母亲": "full_name", "监护人": "full_name", "監護人": "full_name", "老师": "full_name", "教师": "full_name",
    "教師": "full_name", "班主任": "full_name", "学生": "full_name", "學生": "full_name", "联系人": "full_name",
    "保護者": "full_name", "학부모": "full_name", "보호자": "full_name", "교사": "full_name", "학생": "full_name",
    "담임": "full_name",
    # Russian, Ukrainian (apostrophes dropped: ім'я -> імя), Arabic, Hindi, Urdu
    "имя": "first_name", "фамилия": "last_name", "отчество": "middle_name", "фио": "full_name",
    "полноеимя": "full_name", "родитель": "full_name", "мать": "full_name", "отец": "full_name",
    "опекун": "full_name", "учитель": "full_name", "ученик": "full_name", "імя": "first_name",
    "прізвище": "last_name", "побатькові": "middle_name", "піб": "full_name",
    "الاسم": "full_name", "اسم": "full_name", "الاسمالأول": "first_name", "الاسمالاول": "first_name",
    "اسمالعائلة": "last_name", "اللقب": "last_name", "الاسمالكامل": "full_name", "اسمالطالب": "full_name",
    "وليالأمر": "full_name", "नाम": "full_name", "उपनाम": "last_name", "نام": "full_name",
    # person-name words that name a person wherever they stand in the header (``学生姓名``): "~<word>"
    "~姓名": "full_name", "~氏名": "full_name", "~성명": "full_name", "~фамилия": "last_name",
    "~отчество": "middle_name", "~прізвище": "last_name",
}



def _script_key(text: str) -> str:
    """NFC, case-folded letters and marks only (spaces, punctuation, digits dropped; vowel signs kept)."""
    return "".join(ch for ch in unicodedata.normalize("NFC", text).casefold()
                   if unicodedata.category(ch)[0] in "LM")


_SCRIPT_KEYS: dict[str, str] = {_script_key(k): v for k, v in _SCRIPT_NAME_HEADERS.items() if k[0] != "~"}
_SCRIPT_PART_KEYS: dict[str, str] = {_script_key(k[1:]): v for k, v in _SCRIPT_NAME_HEADERS.items()
                                      if k[0] == "~"}


def _script_name_hint(name: str) -> str | None:
    if name.isascii():
        return None
    key = _script_key(name)
    if key in _SCRIPT_KEYS:
        return _SCRIPT_KEYS[key]
    return next((v for p, v in _SCRIPT_PART_KEYS.items() if p in key), None)


#: Words of a run-together header (``apellidopaterno``, ``fechanacimiento``) besides the localized ones.
_DATE_WORDS = frozenset({"fecha", "data", "date", "dt"})


@lru_cache(maxsize=1)
def _segment_vocab() -> frozenset[str]:
    words = {w for w in _LOCAL_WORDS if len(w) > 1 and not _LOCAL_WORDS[w].startswith("pii=")}
    return frozenset(words | _ROLE_SET | _DATE_WORDS | frozenset(NON_PERSON_NOUNS))


@lru_cache(maxsize=4096)
def _segment(token: str) -> tuple[str, ...]:
    """A run-together header token split into known words covering it entirely, fewest words first
    (``apellidopaterno`` -> apellido, paterno; ``nombredelpadre`` -> nombre, del, padre); the token itself when
    no such split exists (``renaissance`` never yields ``naissance``)."""
    vocab = _segment_vocab()
    if token in vocab or len(token) < 5 or not token.isalpha():
        return (token,)
    n = len(token)
    best: list[tuple[str, ...] | None] = [None] * (n + 1)
    best[0] = ()
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 2, n + 1):
            piece = token[i:j]
            if piece in vocab and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + (piece,)
    return best[n] if best[n] and len(best[n]) > 1 else (token,)


def _local_tokens(name: str) -> tuple[str, ...]:
    """Header tokens with run-together localized words split (:func:`_segment`)."""
    return tuple(p for t in tokens(name) for p in _segment(t))


def _local_kind(word: str) -> str | None:
    kind = _LOCAL_WORDS.get(word)
    if kind is None and len(word) > 3 and word.endswith("s"):
        kind = _LOCAL_WORDS.get(word[:-1])  # plurals: ``Responsables``, ``Docentes``
        kind = kind if kind in ("role", "thing") else None
    return kind


def localized_name_hint(name: str) -> str | None:
    """D12 (R3-04): the PII_HINTS name key of a person-name or person-role header in another language, or None.

    ``Nombre`` / ``Prénom`` / ``Nome`` -> first_name; ``Apellido(s)`` / ``Sobrenome`` / ``Nom`` /
    ``Primer_Apellido`` -> last_name; ``Segundo_Nombre`` -> middle_name; ``Nombre_Completo`` / ``Nombre y
    Apellidos`` / ``Nom_Prenom`` -> full_name; a role (``Padre``, ``Responsável``, ``Tutor_Legal``,
    ``Director_Escuela``) -> full_name; ``Họ và tên``, ``姓名`` -> full_name.

    The folded header is read word by word, articles dropped, head first (Romance order): a name or role word
    may be followed by name words, roles and qualifiers; after a role, a thing (``Director_Escuela``: that
    school's director); before any role, a thing makes it the thing's name (``Nombre_Escuela``: None). An
    attribute or unknown word makes it no name header (``Sexo_Alumno``, ``Telefono_Padre``, ``Numero_Alumno``).
    At least one word must come from the localized vocabulary; English headers are left to the other rules.
    """
    script = _script_name_hint(name)
    if script is not None:
        return script
    toks = _local_tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES or _LOCAL_WORDS.get(toks[0]) == "flag":
        return None
    if toks[:2] == ("nombre", "d"):
        return None  # French "Nombre d'élèves": a count
    words = [t for t in strip_role_trailers(toks) if not t.isdigit() and _LOCAL_WORDS.get(t) != "connective"]
    if not words:
        return None
    joined = "".join(words)
    phrase = _LOCAL_WORDS.get(joined, "")
    if phrase.startswith("="):
        return phrase[1:]
    seen: set[str] = set()
    local = role_seen = False
    for i, w in enumerate(words):
        kind = _local_kind(w)
        if kind is not None and kind.startswith(("pii", "=")):
            return None  # Telefono_Padre, Correo_Alumno: that attribute (localized_pii_hint)
        if kind == "role_accent":
            if len(words) == 1 and name.isascii():
                return None  # a bare 'MAE' (mean absolute error) is no 'Mãe' (mother)
            kind = "role"
        if kind is not None:
            local = True
        elif is_role_word(w) or w in NAME_PART_QUALIFIERS:
            kind = "role"
        elif w in NON_PERSON_NOUNS:
            kind = "thing"
        else:
            return None  # an attribute (sexo, telefono, id) or an unknown word: not a name header
        if i == 0 and kind not in ("given", "family", "full", "role", "q_first", "q_middle", "q_full"):
            return None
        if kind == "thing" and not role_seen:
            return None  # Nombre_Escuela, Nome_Turma: the name of a thing
        if kind == "role":
            role_seen = True
        seen.add(kind)
    if not local or not seen & {"given", "family", "full", "role"}:
        return None
    if "q_full" in seen or "full" in seen or ("given" in seen and "family" in seen):
        return "full_name"
    if "family" in seen:
        return "last_name"
    if "given" in seen:
        return "middle_name" if "q_middle" in seen and "q_first" not in seen else "first_name"
    return "full_name"  # a role: Padre, Responsavel, Tutor_Legal, Director_Escuela


def localized_pii_hint(name: str) -> str | None:
    """R3-04: the PII_HINTS key of a phone/e-mail/address/city/ZIP/birth-date/national-ID header in another
    language (whole folded words or joined phrases, articles dropped), or None."""
    toks = _local_tokens(name)
    words = [t for t in toks if _LOCAL_WORDS.get(t) != "connective"]
    joined = "".join(words)
    for phrase, kind in _LOCAL_WORDS.items():
        if kind.startswith("pii=") and phrase in joined:
            return kind[4:]
    for t in words:
        kind = _LOCAL_WORDS.get(t, "")
        if kind.startswith("pii:"):
            return kind[4:]
    return None


@lru_cache(maxsize=1)
def localized_header_words() -> frozenset[str]:
    """Header words of the localized vocabularies that sniffing may take for column names (R3-04: a Spanish
    ``Nombre,Apellido`` header line is a header). Words that are also given names or surnames
    (``detect.names.known_names``: ``Mae``, ``Campos``, ``Calle``) are left out, so a headerless record of such
    a person is still refused (R2-05)."""
    from .detect.names import known_names

    first, last = known_names()
    taken = first | last
    out = set()
    for w, kind in _LOCAL_WORDS.items():
        if kind in ("given", "family", "full", "role", "thing") or kind.startswith("pii:"):
            forms = {w, w + "s", w[:-1] if w.endswith("s") else w}
            if len(w) > 2 and not forms & taken:
                out.add(w)
    return frozenset(out)


#: Header words of things that borrow people's or places' names (schools, teams, houses, streets, months ...):
#: a column headed by one is never judged a column of person names by its values alone (R3-04).
NAME_LIKE_THING_WORDS: frozenset[str] = frozenset({
    "school", "schools", "campus", "building", "site", "facility", "center", "centre", "academy", "college",
    "university", "institute", "district", "city", "town", "county", "state", "country", "nation", "nationality",
    "street", "road", "avenue", "address", "location", "place", "region", "zone", "area", "neighborhood",
    "neighbourhood", "team", "club", "house", "mascot", "tribe", "hall", "dorm", "residence", "church", "parish",
    "park", "lake", "river", "bus", "route", "vendor", "publisher", "company", "organization", "organisation",
    "org", "agency", "brand", "product", "model", "award", "scholarship", "course", "subject", "month", "day",
    "weekday", "season", "holiday", "language", "religion", "color", "colour", "hospital", "clinic", "library",
    "birthplace", "hometown", "origin", "destination", "ethnicity", "race", "program", "programme", "sport",
    "event", "venue", "store", "shop", "restaurant", "station", "airport", "port", "province", "territory",
    "feeder", "zip", "postal", "career", "occupation", "job", "profession", "major", "title", "position",
    "team", "conference", "league", "division",
} | {w for w, kind in _LOCAL_WORDS.items() if kind == "thing"})

#: ID-ish words used for basic ID-candidate warnings (AREA D extends).
ID_HINTS = _vocab("id_hint", token=("id", "ident", "guid", "uuid", "key", "lasid", "sasid", "ssid", "number"))

# --------------------------------------------------------------------------- PII and free text
PII_HINTS: dict[str, Vocabulary] = {
    "first_name": _vocab("first_name", token=("first name", "firstname", "fname", "given name", "forename", "preferred name", "nickname")),
    "last_name": _vocab("last_name", token=("last name", "lastname", "lname", "surname", "family name")),
    "middle_name": _vocab("middle_name", token=("middle name", "middlename", "middle initial", "mname")),
    "full_name": _vocab(
        "full_name",
        exact=("name",),
        token=("full name", "student name", "teacher name", "staff name", "guardian name",
               "parent name", "contact name", "counselor name", "person name"),
    ),
    "email": _vocab("email", token=("email", "e mail")),
    # R4-02: phone headers are structural (:func:`phone_header_hint`, through :func:`pii_hint_matches`); a
    # prefix vocabulary took ``Phoneme_Segmentation`` and ``Contact_Notes`` for phones and missed ``SMS``,
    # ``Parent_Cell`` and ``Tel_No``
    "phone": _vocab("phone"),
    "ssn": _vocab("ssn", token=("ssn", "social security", "socsec", "social sec no", "social sec num",
                                "social sec nbr")),
    "zip": _vocab("zip", token=("zip", "postal", "postcode")),
    "state": _vocab("state", token=("state",)),
    "city": _vocab("city", token=("city", "town")),
    "address": _vocab("address", token=("address", "addr", "street")),
    "dob": _vocab("dob", token=("dob", "birth", "date of birth")),
}

FREE_TEXT_HINTS = _vocab("free_text", token=("comment", "note", "alert", "narrative"))

# --------------------------------------------------------------------------- R4-02: phone and contact headers
#: Number words after a phone word (``Cell_No``, ``SMS_Number``, ``TelNr``, ``Contact #`` -> ``contact``).
PHONE_NUMBER_WORDS: frozenset[str] = frozenset({"number", "numbers", "num", "nums", "no", "nos", "nbr", "nr",
                                                "numero"})
#: Words that name a phone line wherever they stand (whole header tokens, two adjacent tokens such as
#: ``Whats``+``App``, or parts of a run-together token such as ``homephone``, ``phonenumber``, ``telno``): the
#: header is a phone field. ``Phoneme`` and ``Saxophone`` are other words.
PHONE_LINE_WORDS: frozenset[str] = frozenset({
    "phone", "phones", "telephone", "telephones", "tel", "tels", "telno", "telnr", "cellphone", "cellphones",
    "cellular", "smartphone", "mobile", "mobiles", "mobilephone", "fax", "faxes", "telefax", "pager", "pagers",
    "beeper", "beepers", "whatsapp", "landline"})
#: Short or ambiguous phone words: a phone field only in a phone position (``Cell``/``Parent_Cell``/``Cell_No``;
#: ``Home_Ph``/``Ph_No``; ``Mob_No``; ``SMS_Number``/``Txt_No``); elsewhere a messaging word (``SMS``,
#: ``Txt``, ``Text``, ``Cell_Carrier``) makes the header a contact header at most, and ``pH``, ``Text_Level``,
#: ``Informational_Text`` stay ordinary headers.
PHONE_SHORT_WORDS: frozenset[str] = frozenset({"cell", "ph", "mob", "sms", "txt", "text", "texting"})
#: Qualifiers of a phone line or contact: a header made of them alone (``Home``, ``Work``, ``Primary``,
#: ``Alternate``, ``Emergency``, plus numbering or a number word: ``Home_Number``) MAY hold phone numbers.
CONTACT_QUALIFIERS: frozenset[str] = frozenset({
    "home", "work", "business", "office", "daytime", "evening", "primary", "secondary", "alternate", "alt",
    "emergency", "emerg", "contact"})
#: Last header words naming an attribute OF a phone, not the number (``Phone_Type``, ``Cell_Carrier``,
#: ``SMS_Opt_In``, ``Contact_Notes``): such a header is a contact header at most. An extension (``Phone_Ext``)
#: is part of the number: still a phone header.
PHONE_ATTRIBUTE_WORDS: frozenset[str] = frozenset({
    "type", "types", "kind", "status", "carrier", "provider", "plan", "opt", "optin", "optout", "in", "out",
    "consent", "flag", "count", "cnt", "total", "verified", "valid", "confirmed", "date", "dt", "time",
    "country", "code", "area", "language", "lang", "preference", "pref", "method", "sent", "received",
    "message", "messages", "msg", "note", "notes", "comment", "comments", "level", "lvl", "score", "desc",
    "description", "listed", "unlisted", "permission", "allowed"})
#: Words a run-together phone header may be made of (``homephone``, ``cellno``, ``dayphone``, ``parentcell``).
_PHONE_SEGMENT_WORDS: frozenset[str] = (PHONE_LINE_WORDS | PHONE_SHORT_WORDS | PHONE_NUMBER_WORDS
                                        | CONTACT_QUALIFIERS | frozenset({"day", "night", "eve", "after", "hours"}))


@lru_cache(maxsize=4096)
def _phone_segments(token: str) -> tuple[str, ...]:
    """A run-together header token split into phone-header words covering it entirely, when one of them is a
    phone word (``homephone`` -> home, phone; ``smsno`` -> sms, no); the token itself otherwise
    (``saxophone``, ``phoneme``, ``context`` stay whole)."""
    if token in _PHONE_SEGMENT_WORDS or len(token) < 4 or not token.isalpha():
        return (token,)
    n = len(token)
    best: list[tuple[str, ...] | None] = [None] * (n + 1)
    best[0] = ()
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 2, n + 1):
            piece = token[i:j]
            if piece in _PHONE_SEGMENT_WORDS and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + (piece,)
    split = best[n]
    if split and len(split) > 1 and any(w in PHONE_LINE_WORDS or w in PHONE_SHORT_WORDS for w in split):
        return split
    return (token,)


def _contact_word(word: str) -> bool:
    """A word that may stand before a short phone word (``Home_Ph``, ``Parent_Text``, ``Student_Ph``)."""
    return word in CONTACT_QUALIFIERS or word in PERSON_ID_WORDS or is_role_word(word) or word in (
        "day", "night", "eve")


@lru_cache(maxsize=4096)
def phone_header_hint(name: str) -> str | None:
    """R4-02: the phone reading of a header, one structural rule for every phone/messaging header.

    * ``'phone'``: the header names a phone-number field whatever its values (a PII hint, D12: real values are
      never copied): a phone word anywhere (:data:`PHONE_LINE_WORDS`: ``Phone``, ``Home_Phone``, ``Telephone``,
      ``Mobile``, ``Fax``, ``Pager``, ``WhatsApp``, ``Cellular``, ``Parent_Tel``, ``homephone``,
      ``phonenumber``) or a short phone word in a phone position (:data:`PHONE_SHORT_WORDS`: ``Cell``,
      ``Guardian_Cell``, ``Cell_No``, ``Home_Ph``, ``Ph_No``, ``Mob_No``, ``SMS_Number``, ``Txt_No``), or
      ``Contact_Number``.
    * ``'contact'``: the header MAY hold phone numbers (``detect.text``: values that are phone numbers make the
      column a PHONE column, and digit-only phone numbers are never copied; ordinary labels stay): a
      messaging word anywhere (``SMS``, ``SMS_Contact``, ``Txt``, ``Cell_Carrier``), ``Text`` alone or after a
      person/qualifier word (``Parent_Text``, ``Text_Number``), a header of contact qualifiers alone
      (:data:`CONTACT_QUALIFIERS`: ``Home``, ``Work``, ``Primary``, ``Alternate``, ``Emergency``,
      ``Home_Number``), or a phone header naming an attribute of the phone (:data:`PHONE_ATTRIBUTE_WORDS`:
      ``Phone_Type``, ``SMS_Opt_In``).
      A short phone word alone (``Ph``, ``Mob``: :func:`bare_short_phone_header`) is a contact header too.
    * None otherwise, and for flags and counts (``Has_Cell``, ``Num_Phones``): ``Phoneme_Segmentation``,
      ``Text_Level``, ``Informational_Text``, ``Day_Number``, ``Primary_Language``, ``Contact_Notes``, ``pH``.

    Numbering and ranking never hide the phone word (``Cell_2``, ``Phone_Primary``, ``Guardian1_Cell``).
    """
    toks = tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES:
        return None
    words = [w for t in toks for w in _phone_segments(t)]
    core = list(strip_role_trailers(tuple(words)))
    has_num = False
    while len(core) > 1 and core[-1] in PHONE_NUMBER_WORDS:
        core.pop()
        has_num = True
    attribute = len(core) > 1 and core[-1] in PHONE_ATTRIBUTE_WORDS
    if attribute:
        core = core[:-1]
    last = core[-1]
    pairs = {a + b for a, b in zip(core, core[1:])}
    strong = (any(w in PHONE_LINE_WORDS for w in core) or bool(pairs & PHONE_LINE_WORDS)
              or last == "cell"
              or (last in ("ph", "mob") and (has_num or (len(core) >= 2 and _contact_word(core[-2]))))
              or (last in ("sms", "txt", "contact") and has_num and not attribute))
    if strong:
        return "contact" if attribute else "phone"
    if any(w in ("sms", "txt", "texting", "cell") for w in core):
        return "contact"
    if attribute:
        return None
    if last == "text" and (len(core) == 1 or _contact_word(core[-2])):
        return "contact"
    if all(w in CONTACT_QUALIFIERS for w in core):
        return "contact"
    if bare_short_phone_header(name):
        return "contact"  # hardening pass 3: a bare 'Ph' or 'Mob' holding phone numbers (never a pH value)
    return None


def bare_short_phone_header(name: str) -> bool:
    """A header that is a short phone word alone (``Ph``, ``PH``, ``Mob``; numbering aside): it MAY hold phone numbers,
    local seven-digit ones included (``detect.text.contact_phone``), and ``pH`` values stay numbers."""
    toks = tokens(name)
    if not toks or toks[0] in _FLAG_COUNT_PREFIXES:
        return False
    return list(strip_role_trailers(toks)) in (["ph"], ["mob"])


def pii_hint_matches(name: str, key: str) -> bool:
    """True when header ``name`` carries the PII hint ``key`` of :data:`PII_HINTS` (``phone`` is structural:
    :func:`phone_header_hint` is ``'phone'``)."""
    if key == "phone":
        return phone_header_hint(name) == "phone"
    return match_any(name, PII_HINTS[key]) is not None

# --------------------------------------------------------------------------- dates, years, codes
DATE_HINTS = _vocab(
    "date",
    exact=("dt",),
    token=("date", "dob", "birth", "timestamp", "datetime", "created", "updated", "modified",
           "enroll", "entry", "exit", "withdraw", "start", "end"),
)
TIME_HINTS = _vocab("time", token=("time", "clock", "hour"))
DURATION_HINTS = _vocab("duration", token=("duration", "elapsed", "minutes", "mins", "hours", "hrs", "seconds", "length of"))
YEAR_HINTS = _vocab("year", exact=("yy",), token=("year", "yr", "cohort", "class of"))
SCHOOL_YEAR_HINTS = _vocab(
    "school_year",
    exact=("sy",),
    token=("school year", "schoolyear", "academic year", "acad year", "sch year", "school yr", "schyr"),
)
#: Grade/level/code/count words: block 0/1 and -1/0 booleans (section 3 #4),
#: make integers discrete (#9).
CODE_COUNT_HINTS = _vocab(
    "code_count",
    exact=("n", "gr"),
    token=("grade", "grd", "level", "lvl", "code", "count", "cnt", "number", "num", "total", "qty", "quantity"),
)
GRADE_HINTS = _vocab("grade", exact=("gr",), token=("grade", "grd", "grade level"))
#: "Code hint" of section 3 #9: integer code columns keep only observed values (discrete).
CODE_HINTS = _vocab("code", token=("code",))

# --------------------------------------------------------------------------- value vocabularies
#: Missing-like tokens: case-sensitive, missing only in typed columns (section 3).
DEFAULT_NA_TOKENS: tuple[str, ...] = (
    "NA", "N/A", "NULL", "null", "NaN", "#N/A", "None", ".", "-", "--", "#VALUE!", "#DIV/0!",
)

#: Known boolean pairs (case-folded). Value = (true token, false token).
BOOLEAN_PAIRS: tuple[tuple[str, str], ...] = (
    ("true", "false"), ("t", "f"), ("yes", "no"), ("y", "n"), ("1", "0"),
    ("1.0", "0.0"), ("-1", "0"), ("on", "off"),
)
#: Pairs flagged ambiguous (0/1 and Y/N, plus the other numeric encodings of 0/1).
AMBIGUOUS_BOOLEAN_PAIRS = frozenset({("1", "0"), ("1.0", "0.0"), ("-1", "0"), ("y", "n")})
#: Pairs blocked by a grade/level/code/count header hint.
NUMERIC_BOOLEAN_PAIRS = frozenset({("1", "0"), ("1.0", "0.0"), ("-1", "0")})


@dataclass(frozen=True)
class OrdinalVocab:
    """An ordered level vocabulary. ``ranks`` maps case-folded label -> order key."""

    name: str
    ranks: dict[str, float]
    numeric_needs_hint: bool = False  # all-numeric value sets need GRADE_HINTS

    def order(self, labels: list[str]) -> list[str]:
        return sorted(labels, key=lambda s: (self.ranks[s.strip().casefold()], s))


def _ordered(name: str, labels: list[str]) -> OrdinalVocab:
    return OrdinalVocab(name, {lab.casefold(): float(i) for i, lab in enumerate(labels)})


def _grade_levels() -> OrdinalVocab:
    ranks: dict[str, float] = {"pk3": -3.0, "pk4": -2.0, "pk": -1.5, "tk": -0.5, "kg": 0.0, "k": 0.0,
                               "ug": 100.0, "13": 13.0}
    for g in range(1, 13):
        ranks[f"{g:02d}"] = float(g)
        ranks[str(g)] = float(g)
    for g in (-2, -1, 0):
        ranks[str(g)] = float(g)
    return OrdinalVocab("grade_level", ranks, numeric_needs_hint=True)


#: Section 3 #6 vocabularies, in priority order. Matching needs >= 3 levels.
ORDINAL_VOCABS: tuple[OrdinalVocab, ...] = (
    _grade_levels(),
    _ordered("letter_grade", ["F", "E", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]),
    _ordered("likert", ["strongly disagree", "disagree", "somewhat disagree", "neither agree nor disagree",
                        "neutral", "somewhat agree", "agree", "strongly agree"]),
    _ordered("very_low_very_high", ["very low", "low", "medium", "high", "very high"]),
    _ordered("year_of_study", ["first year", "second year", "third year", "fourth year"]),
    _ordered("class_standing", ["freshman", "sophomore", "junior", "senior"]),
    _ordered("parental_education", ["some high school", "high school", "some college", "associate's degree",
                                    "bachelor's degree", "master's degree"]),
    _ordered("term_season", ["fall", "winter", "spring"]),
    _ordered("performance_naep", ["below basic", "basic", "proficient", "advanced"]),
    _ordered("performance_ok", ["unsatisfactory", "limited knowledge", "proficient", "advanced"]),
    _ordered("performance_met", ["standard not met", "standard nearly met", "standard met", "standard exceeded"]),
    _ordered("performance_expectations", ["did not meet", "partially met", "approached", "met", "exceeded"]),
)

# --------------------------------------------------------------------------- school years (section 3 #7)
_SY = r"(?:[Ss][Yy]\s?)?"
SCHOOL_YEAR_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("YYYY-YYYY", re.compile(_SY + r"(\d{4})\s*[-/]\s*(\d{4})")),
    ("YYYY-YY", re.compile(_SY + r"(\d{4})\s*[-/]\s*(\d{2})")),
    ("YY-YY", re.compile(_SY + r"(\d{2})\s*[-/]\s*(\d{2})")),
    ("SYYYYY", re.compile(r"[Ss][Yy]\s?(\d{2})(\d{2})")),
    ("YYYY_concat", re.compile(r"(\d{2})(\d{2})")),
)


def school_year_start(value: str) -> tuple[str, int] | None:
    """Return (form, start year as written) if ``value`` is a valid school year.

    The second year must be the first plus one (mod 100 for 2-digit years).
    """
    v = value.strip()
    for form, rx in SCHOOL_YEAR_FORMS:
        m = rx.fullmatch(v)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        ai, bi = int(a), int(b)
        if len(a) == 4 and len(b) == 4:
            ok = bi == ai + 1
        else:
            ok = bi == (ai + 1) % 100
        if ok:
            return form, ai
    return None


# --------------------------------------------------------------------------- convenience
@dataclass(frozen=True)
class HeaderHints:
    """All header-hint matches of one column header (computed once per column)."""

    name: str
    normalized: str
    default_id: bool
    org_code: str | None
    person_id_like: bool
    pii: str | None  # PII_HINTS key
    free_text: str | None
    date: str | None
    time: str | None
    duration: str | None
    year: str | None
    school_year: str | None
    code_count: str | None
    grade: str | None
    code: str | None
    id_hint: str | None
    #: D39: a finance segment word (:func:`finance_segment_hint`): an org code only when its values are codes
    finance_segment: str | None = None


def header_hints(name: str) -> HeaderHints:
    pii = None
    for key in PII_HINTS:
        if pii_hint_matches(name, key):
            pii = key
            break
    person_id_like = person_id_hint(name)
    if pii is None and not person_id_like:
        # R3-04: localized name headers before the English '<x>name' rule (``Vorname`` is a given name)
        pii = (person_name_part_hint(name) or localized_name_hint(name)
               or ("full_name" if person_name_hint(name) else None) or localized_pii_hint(name))
    return HeaderHints(
        name=name,
        normalized=normalize(name),
        default_id=default_id_hint(name),
        org_code=None if person_id_like else org_code_hint(name),
        person_id_like=person_id_like,
        pii=pii,
        free_text=match_any(name, FREE_TEXT_HINTS),
        date=match_any(name, DATE_HINTS),
        time=match_any(name, TIME_HINTS),
        duration=match_any(name, DURATION_HINTS),
        year=match_any(name, YEAR_HINTS),
        school_year=match_any(name, SCHOOL_YEAR_HINTS),
        code_count=match_any(name, CODE_COUNT_HINTS),
        grade=match_any(name, GRADE_HINTS),
        code=match_any(name, CODE_HINTS),
        id_hint=match_any(name, ID_HINTS),
        finance_segment=None if person_id_like else finance_segment_hint(name),
    )
