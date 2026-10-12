"""Person names recognised by their values (AREA C, R3-04): design D12 "value-validated PII".

A column whose header names no person (``Homeroom``, ``Assigned``, ``Responsible``, a header in a language the
hint lists do not cover) can still hold real people's names; as a factor (<= ``--factor-max`` distinct values)
its real levels would be written verbatim (above it, invented labels would keep its names out of the run-wide
full-name index). :func:`name_value_verdict` judges the distinct values alone and
returns a PERSON_NAME subtype (``first``, ``last``, ``full``) when they are mostly known given/family names,
so D12 (PII wins over the factor rule) applies to them too; ``detect.text.apply_pii_precedence`` calls it.

Known names are the given-name and surname lists of the pinned Faker version (D2/D24) for the locales of US
school populations (English, Spanish, Portuguese, French, Vietnamese, Chinese/Japanese romanised, African,
South Asian, German, Italian ...), folded to lower-case ASCII (:func:`known_names`; deterministic: the same
Faker version gives the same lists).

False-positive controls (a factor of real labels keeps its labels):

* every word of a value must be a known name (titles, suffixes, particles and initials aside): ``Lincoln
  Elementary``, ``Algebra II``, ``Software Engineer`` are no names;
* words that are also everyday words are *common* (:func:`common_words`: the Faker English word list, month,
  weekday, season, colour, language, country and US-state names in several languages, big US cities, famous
  namesakes of schools such as ``Washington`` or ``Lincoln``, virtues, flowers, animals, occupations): they
  never count as evidence, only as consistent with names;
* words of two letters or fewer are common (``Li``, ``Le``, ``An``);
* values that start like a place (``San``/``Santa``/``Saint``/``Los``/``Las``/``El``/``Fort``/``Mount``) are
  no names;
* at least :data:`MIN_DISTINCTIVE` distinct values must hold a distinctive known name, and they must be at least
  :data:`DISTINCTIVE_SHARE` of the distinct values, and :data:`NAMEISH_SHARE` of the distinct values must be
  name-shaped (known names, common words or unknown capitalised words such as a rare surname);
* the caller skips headers of things named after people or places (school, team, house, street, month,
  colour, course ...: ``hints.NAME_LIKE_THING_WORDS``) and org-code columns (D10).

A single-value column (CONSTANT / FLAG) qualifies only as a full name of two or more words with at least
two distinctive known names (``Maria Garcia``).
"""

from __future__ import annotations

import importlib
import re
from functools import lru_cache
from typing import Any

from ..hints import fold_accents

#: Distinct values with a distinctive known name needed (columns of two or more distinct values).
MIN_DISTINCTIVE = 2
#: Share of the distinct values that must hold a distinctive known name.
DISTINCTIVE_SHARE = 0.5
#: Share of the distinct values that must be name-shaped (known names, common words, unknown capitalised words).
NAMEISH_SHARE = 0.9
#: Distinct values judged at most (the most frequent first; a high-cardinality column may have more).
MAX_JUDGED = 5000

#: Faker provider modules read (``faker.providers.<key>``) and what they give: "names" (given-name and surname
#: lists of the person locales of US school populations; romanised lists only), and common words: "lorem"
#: (the English word list), "dates" (month and weekday names), "colors" and "places" (state and country
#: names); language names come from the English person provider. A dict of more than 50 entries, so the leak
#: scanner treats the locale codes as data, never as text the tool writes.
FAKER_SOURCES: dict[str, str] = {
    "person.en_US": "names", "person.en": "names", "person.en_GB": "names", "person.en_IE": "names",
    "person.en_NZ": "names", "person.en_IN": "names", "person.en_PK": "names", "person.en_KE": "names",
    "person.en_NG": "names", "person.es_ES": "names", "person.es_MX": "names", "person.es_CO": "names",
    "person.es_AR": "names", "person.es_CL": "names", "person.pt_BR": "names", "person.pt_PT": "names",
    "person.fr_FR": "names", "person.fr_CA": "names", "person.fr_DZ": "names", "person.it_IT": "names",
    "person.de_DE": "names", "person.vi_VN": "names", "person.zh_CN": "names", "person.zh_TW": "names",
    "person.ja_JP": "names", "person.sw": "names", "person.yo_NG": "names", "person.ig_NG": "names",
    "person.ha_NG": "names", "person.tw_GH": "names",
    "lorem.en_US": "lorem", "date_time.en_US": "dates", "date_time.es": "dates", "date_time.es_ES": "dates",
    "date_time.pt_BR": "dates", "date_time.pt_PT": "dates", "date_time.fr_FR": "dates", "date_time.it_IT": "dates",
    "date_time.de_DE": "dates", "date_time.vi_VN": "dates", "color.en_US": "colors", "color.es": "colors",
    "color.es_ES": "colors", "color.pt_BR": "colors", "color.fr_FR": "colors", "color.de_DE": "colors",
    "color.vi_VN": "colors", "address.en_US": "places", "address.es_ES": "places", "address.es_MX": "places",
    "address.pt_BR": "places", "address.fr_FR": "places",
}
#: Provider attributes read per kind of source (person lists by prefix: every given/middle/surname list).
_SOURCE_ATTRS: dict[str, tuple[str, ...]] = {
    "lorem": ("word_list",), "dates": ("DAY_NAMES", "MONTH_NAMES"), "colors": ("all_colors",),
    "places": ("states", "countries"),
}
_FIRST_PREFIXES = ("first_name", "first_romanized", "middle_name")
_LAST_PREFIXES = ("last_name", "last_romanized")

#: Everyday words that are also given names or surnames, beyond the Faker word lists (seasons, virtues,
#: flowers and gems, animals, directions, ranks, occupations, US cities and states, school namesakes ...).
#: One word list of more than 50 entries, so the leak scanner treats it as data, never as text the tool writes.
_COMMON_EXTRA: frozenset[str] = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "spring", "summer", "autumn", "fall", "winter", "primavera", "verano", "otono", "invierno", "hope", "faith",
    "joy", "grace", "charity", "honor", "honour", "patience", "mercy", "trinity", "destiny", "harmony",
    "liberty", "justice", "unity", "glory", "promise", "serenity", "felicity", "prudence", "constance",
    "temperance", "verity", "chastity", "unique", "precious", "angel", "heaven", "rose", "lily", "daisy",
    "iris", "ivy", "holly", "jasmine", "poppy", "violet", "ruby", "pearl", "jade", "amber", "crystal", "opal",
    "diamond", "coral", "willow", "hazel", "olive", "laurel", "heather", "blossom", "flora", "fern", "sage",
    "basil", "clover", "rosemary", "saffron", "rain", "river", "sky", "storm", "dawn", "sunny", "forest",
    "meadow", "brook", "cliff", "dale", "glen", "heath", "rock", "stone", "wood", "woods", "field", "fields",
    "eagle", "falcon", "hawk", "owl", "wolf", "fox", "bear", "lion", "tiger", "panther", "jaguar", "cougar",
    "bulldog", "mustang", "bronco", "ram", "raven", "cardinal", "robin", "sparrow", "wren", "finch", "crane",
    "swan", "dove", "lamb", "bird", "fish", "north", "south", "east", "west", "northeast", "northwest",
    "southeast", "southwest", "central", "upper", "lower", "middle", "king", "queen", "prince", "princess",
    "duke", "earl", "baron", "knight", "bishop", "pope", "major", "general", "marshal", "sergeant", "captain",
    "baker", "butler", "carpenter", "carter", "cook", "cooper", "farmer", "fisher", "fowler", "gardner",
    "hunter", "mason", "miller", "porter", "potter", "shepherd", "turner", "weaver", "taylor", "tailor",
    "sawyer", "slater", "thatcher", "chandler", "draper", "fletcher", "banker", "lawyer", "doctor", "nurse",
    "teacher", "writer", "singer", "dancer", "driver", "pilot", "judge", "chef", "artist", "washington",
    "lincoln", "jefferson", "roosevelt", "kennedy", "madison", "monroe", "franklin", "hamilton", "jackson",
    "adams", "grant", "wilson", "truman", "eisenhower", "reagan", "obama", "douglass", "tubman", "edison",
    "whitman", "emerson", "thoreau", "hawthorne", "austin", "dallas", "houston", "denver", "phoenix", "orlando",
    "charlotte", "jordan", "tyler", "eugene", "florence", "chester", "savannah", "raleigh", "columbus",
    "aurora", "irving", "arlington", "lafayette", "logan", "marion", "salem", "camden", "clinton", "warren",
    "kent", "dayton", "lansing", "trenton", "lowell", "tacoma", "augusta", "helena", "montgomery", "richmond",
    "albany", "boston", "cleveland", "georgia", "virginia", "carolina", "dakota", "indiana", "montana",
    "nevada", "florida", "maryland", "louisiana", "victoria", "paris", "london", "sydney", "chelsea",
    "brooklyn", "bronx", "jersey", "kingston", "america", "asia", "africa", "europe", "india", "china", "japan",
    "israel", "kenya", "chad", "peru", "chile", "cuba", "english", "french", "spanish", "german", "dutch",
    "irish", "scottish", "welsh", "mercedes", "porsche", "lexus", "chevy", "ford", "dodge", "light", "dark",
    "bright", "gray", "grey", "brown", "green", "white", "black", "red", "blue", "gold", "silver", "bronze",
    "unknown", "other", "none", "various", "multiple", "staff", "vacant", "pending", "tbd", "tba", "unassigned",
    "male", "female", "boy", "girl", "man", "woman", "ell", "esl", "sped", "iep", "gifted", "honors", "regular",
    "standard", "special", "basic", "advanced", "proficient", "novice", "expert", "master", "junior", "senior",
    "freshman", "sophomore", "active", "inactive", "yes", "no", "si", "sim", "oui", "non",
})
#: Value words that never make or break a name ("particle": titles, suffixes, particles, connectors) and first
#: words of place names ("place": ``San Diego``, ``Santa Rosa``, ``Los Angeles``, ``Fort Wayne``).
_VALUE_WORDS: dict[str, str] = {
    "mr": "particle", "mrs": "particle", "ms": "particle", "miss": "particle", "mx": "particle",
    "dr": "particle", "prof": "particle", "profa": "particle", "rev": "particle", "fr": "particle",
    "sr": "particle", "sra": "particle", "srta": "particle", "don": "particle", "dona": "particle",
    "lic": "particle", "ing": "particle", "mme": "particle", "mlle": "particle", "jr": "particle",
    "ii": "particle", "iii": "particle", "iv": "particle", "de": "particle", "del": "particle",
    "della": "particle", "la": "particle", "le": "particle", "van": "particle", "von": "particle",
    "der": "particle", "den": "particle", "da": "particle", "di": "particle", "du": "particle",
    "dos": "particle", "das": "particle", "do": "particle", "bin": "particle", "ibn": "particle",
    "al": "particle", "y": "particle", "e": "particle", "and": "particle", "et": "particle", "mc": "particle",
    "mac": "particle", "san": "place", "santa": "place", "santo": "place", "sao": "place", "saint": "place",
    "st": "place", "ste": "place", "sainte": "place", "los": "place", "las": "place", "el": "place",
    "fort": "place", "ft": "place", "mount": "place", "mt": "place", "port": "place", "lake": "place",
    "new": "place", "north": "place", "south": "place", "east": "place", "west": "place",
}
_SPLIT = re.compile(r"[\s,/&]+")
_STRIP = ".()\"'`’"


def fold_word(word: str) -> str:
    """Lower-case ASCII form of a name word (``Hernández`` -> ``hernandez``, ``O'Brien`` -> ``obrien``)."""
    w = fold_accents(word).casefold()
    return "".join(ch for ch in w if ch.isalpha())


def _provider(module: str) -> Any:  # noqa: ANN401 - a Faker provider class
    try:
        return importlib.import_module("faker.providers." + module).Provider
    except Exception:  # a locale missing from the pinned Faker: no words
        return None


def _words_of(items: Any) -> set[str]:  # noqa: ANN401 - a Faker list, tuple or (weighted) dict of strings
    out: set[str] = set()
    for item in (list(items) if isinstance(items, dict) else items if isinstance(items, (list, tuple)) else ()):
        if isinstance(item, str):
            for part in re.split(r"[\s\-]+", item):
                w = fold_word(part)
                if w.isascii() and len(w) >= 2:
                    out.add(w)
    return out


@lru_cache(maxsize=1)
def known_names() -> tuple[frozenset[str], frozenset[str]]:
    """(given names, surnames) of the "names" sources of :data:`FAKER_SOURCES`, folded (every given, middle
    and surname list of the locale; ``*_romanized_*`` lists for Chinese and Japanese)."""
    first: set[str] = set()
    last: set[str] = set()
    for module, kind in FAKER_SOURCES.items():
        prov = _provider(module) if kind == "names" else None
        if prov is None:
            continue
        for attr in sorted(vars(prov)):
            if attr.startswith(_FIRST_PREFIXES):
                first |= _words_of(getattr(prov, attr))
            elif attr.startswith(_LAST_PREFIXES):
                last |= _words_of(getattr(prov, attr))
    return frozenset(first), frozenset(last)


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """Everyday words that are also names somewhere (module docstring); never evidence of a name column."""
    words: set[str] = set(_COMMON_EXTRA)
    for module, kind in FAKER_SOURCES.items():
        prov = _provider(module) if kind != "names" else None
        if prov is None:
            continue
        for attr in _SOURCE_ATTRS[kind]:
            words |= _words_of(getattr(prov, attr, ()))
    en = _provider("person.en_US")
    if en is not None:  # language names ("English", "French": surnames too)
        words |= _words_of(getattr(en, "language_names", ()))
    return frozenset(words)


#: R4-01: a room or period number written beside a person's name in a homeroom/advisory label (``Smith 101``,
#: ``Garcia - Rm 104``, ``Nguyen (Room 12)``, ``101 - Smith``): set apart by a space or separator, at most 4
#: digits and one letter. The name is judged without it (:func:`strip_room`).
_ROOM_TAIL = re.compile(r"(?:\s*[-–,/:|(]\s*|\s+)(?:(?:rm|room|hr|per|period|p|#)\.?\s*)?#?\d{1,4}[A-Za-z]?"
                        r"\s*\)?\s*$", re.I)
_ROOM_HEAD = re.compile(r"^\s*(?:(?:rm|room|hr|per|period|p|#)\.?\s*)?#?\d{1,4}[A-Za-z]?(?:\s*[-–,/:|]\s*|\s+)",
                        re.I)


def strip_room(value: str) -> str:
    """``value`` without one leading and one trailing room/period number (``Smith 101`` -> ``Smith``,
    ``Garcia - Rm 104`` -> ``Garcia``, ``101 - Nguyen`` -> ``Nguyen``); unchanged when nothing with a letter
    would be left."""
    v = _ROOM_TAIL.sub("", value)
    v = _ROOM_HEAD.sub("", v)
    return v if any(ch.isalpha() for ch in v) else value


def _value_words(value: str) -> list[str] | None:
    """Folded name words of ``value`` (titles, suffixes, particles and initials dropped; a room number beside
    the name too, :func:`strip_room`), or None when the value is no name shape (digits, symbols, more than 6
    words, a place name)."""
    value = strip_room(value)
    if any(ch.isdigit() for ch in value) or not any(ch.isalpha() for ch in value):
        return None
    parts = [p.strip(_STRIP) for p in _SPLIT.split(value.strip()) if p.strip(_STRIP)]
    if not parts or len(parts) > 6:
        return None
    words: list[str] = []
    for p in parts:
        for piece in p.split("-"):
            if not piece:
                continue
            if not all(ch.isalpha() or ch in _STRIP for ch in piece):
                return None
            if "." in piece and all(len(s) <= 1 for s in piece.split(".")):
                continue  # initials written together ('J.K', 'M.A'): no name word
            w = fold_word(piece)
            if not w or len(w) == 1 or _VALUE_WORDS.get(w) == "particle":
                continue
            words.append(w)
    if words and _VALUE_WORDS.get(fold_word(parts[0])) == "place" and len(parts) > 1:
        return None
    return words


#: :func:`classify_value` results.
DISTINCTIVE, CONSISTENT, UNKNOWN, OTHER = "distinctive", "consistent", "unknown", "other"

#: R4-01: the layouts of a person's name written with initials or a title, which ordinary labels do not use:
#: ``Smith, J.`` / ``Smith, J`` / ``Garcia Lopez, M. A.`` (surname, comma, initials), ``J. Smith`` / ``J.K.
#: Smith`` (initials with periods, surname), ``Smith J.`` (surname, initial with a period), and ``Mr. Smith`` /
#: ``Mrs Quenby`` / ``Dr. Voss`` (a title, then one to three name words). ``Building A``, ``Day B``, ``Plan B``
#: and ``Vitamin D`` have none of them.
_W = r"[^\W\d_][^\W\d_'’\-]*"
_INITIAL_LAYOUTS = (
    re.compile(rf"(?P<n>{_W}(?:[ \-]{_W})?)\s*,\s*(?P<i>(?:[^\W\d_]\.?\s*){{1,2}})"),
    re.compile(rf"(?P<i>(?:[^\W\d_]\.\s*){{1,2}})(?P<n>{_W}(?:[ \-]{_W})?)"),
    re.compile(rf"(?P<n>{_W})\s+(?P<i>[^\W\d_]\.(?:\s*[^\W\d_]\.)?)"),
)
_TITLE_LAYOUT = re.compile(rf"(?P<t>{_W})\.?\s+(?P<n>{_W}(?:\s+{_W}){{0,2}})")
_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "rev", "fr", "sr", "sra", "srta", "mme", "mlle",
                     "coach"})


def name_layout_kind(value: str) -> str | None:
    """``'initials'`` or ``'title'`` when ``value`` is written in a person-name layout of
    :data:`_INITIAL_LAYOUTS` / :data:`_TITLE_LAYOUT` with capitalised name words (R4-01), else None."""
    v = strip_room(value).strip()
    if not v or len(v) > 60 or any(ch.isdigit() for ch in v):
        return None
    for rx in _INITIAL_LAYOUTS:
        m = rx.fullmatch(v)
        if m and all(w[0].isupper() for w in re.split(r"[ \-]+", m.group("n")) if w) \
                and all(ch.isupper() for ch in m.group("i") if ch.isalpha()):
            # an initial without a period after a comma ('Smith, J') only with a known surname: 'Algebra, A' is a
            # course section
            if rx is _INITIAL_LAYOUTS[0] and "." not in m.group("i"):
                first, last = known_names()
                if not any(fold_word(w) in last or fold_word(w) in first for w in re.split(r"[ \-]+", m.group("n"))):
                    return None
            return "initials"
    m = _TITLE_LAYOUT.fullmatch(v)
    if m and fold_word(m.group("t")) in _TITLES and all(w[0].isupper() for w in m.group("n").split()):
        return "title"
    return None


def classify_value(value: str) -> str:
    """:data:`DISTINCTIVE` (every word a known name, one of them no common word; or a name written with
    initials or a title, :func:`name_layout_kind`: ``Voss, A.``, ``Mrs. Quenby``), :data:`CONSISTENT` (every
    word a known name or common word, none distinctive), :data:`UNKNOWN` (name-shaped words not on the lists:
    a rare name), or :data:`OTHER` (no name shape, or a common word beside an unknown one: ``Lincoln
    Elementary``)."""
    words = _value_words(value)
    if words is None:
        return OTHER
    if not words:
        return CONSISTENT  # only a particle, title or initial ('Le', 'J.'): consistent with names, no evidence
    first, last = known_names()
    common = common_words()
    n_common = n_name = n_unknown = 0
    for w in words:
        if w in common or len(w) <= 2:
            n_common += 1
        elif w in first or w in last:
            n_name += 1
        else:
            n_unknown += 1
    if n_unknown:
        # a rare name ('Zelmira Quenby') or a label ('Lincoln Elementary', 'Software Engineer'): only unknown
        # words alone, or with known names, stay name-shaped; written with initials or a title ('Quenby, Z.',
        # 'Mrs. Quenby') it is a person's name (R4-01)
        if n_common:
            return OTHER
        return DISTINCTIVE if name_layout_kind(value) else UNKNOWN
    if n_name or name_layout_kind(value):  # 'Mr. Brown', 'Green, G.': a common surname with a title/initial
        return DISTINCTIVE
    return CONSISTENT


def person_value(value: str) -> bool:
    """One value that is a person's name on its own (a constant, a token of a number column: R4-01): a
    distinctive name (:func:`classify_value`) written with initials or a title (``Smith, J.``, ``Mrs. Quenby``)
    or holding at least two distinctive known name words (``Maria Garcia``)."""
    if classify_value(value) != DISTINCTIVE:
        return False
    if name_layout_kind(value):
        return True
    first, last = known_names()
    common = common_words()
    words = _value_words(value) or []
    distinctive = sum(1 for w in words if w not in common and len(w) > 2 and (w in first or w in last))
    return len(words) >= 2 and distinctive >= 2


_TITLE_WORD = re.compile(r"[^\W\d_][^\W\d_A-Z]*(?:['’\-][^\W\d_][^\W\d_]*)*")


def stray_name(value: str) -> bool:
    """A text left among numbers (a token of a number column) that is a person's name: :func:`person_value`, or
    two or three capitalised words, none a common word, one of them a distinctive known name (``Quillon
    Varga``: a rare given name beside a known surname). Among numbers no label reads like that, so the looser
    rule holds there (R4-01); a column of such labels is judged by :func:`name_value_verdict`."""
    if person_value(value):
        return True
    parts = value.split()
    if not 2 <= len(parts) <= 3 or not all(_TITLE_WORD.fullmatch(p) and p[0].isupper() for p in parts):
        return False
    if classify_value(value) not in (DISTINCTIVE, UNKNOWN):
        return False
    first, last = known_names()
    common = common_words()
    return any(w not in common and len(w) > 2 and (w in first or w in last) for w in _value_words(value) or [])


def _part(value: str) -> str | None:
    """'first' / 'last' when a one-word distinctive value is only on one list, else None."""
    words = _value_words(value) or []
    if len(words) != 1:
        return None
    first, last = known_names()
    w = words[0]
    if w in first and w not in last:
        return "first"
    if w in last and w not in first:
        return "last"
    return None


def name_value_verdict(values: dict[str, int]) -> str | None:
    """PERSON_NAME subtype (``first``, ``last``, ``full``) when the distinct ``values`` (stripped value ->
    row count; NA tokens left out) are person names by the rules of the module docstring, else None."""
    distinct: dict[str, int] = {}
    rep: dict[str, str] = {}  # case-folded value -> a written form (the layout rules read capitals)
    for v, c in values.items():
        key = v.strip()
        if key:
            folded = key.casefold()
            distinct[folded] = distinct.get(folded, 0) + c
            rep.setdefault(folded, key)
    if not distinct:
        return None
    judged = [rep[k] for k in sorted(distinct, key=lambda k: (-distinct[k], k))[:MAX_JUDGED]]
    kinds = {v: classify_value(v) for v in judged}
    named = [v for v in judged if kinds[v] == DISTINCTIVE]
    n = len(judged)
    if n == 1:
        if not person_value(judged[0]):  # 'Maria Garcia', 'Smith, J.', 'Mrs. Quenby' (R4-01)
            return None
    elif len(named) < MIN_DISTINCTIVE or len(named) < DISTINCTIVE_SHARE * n:
        return None
    elif len({tuple(sorted(_value_words(v) or [])) for v in named}) < MIN_DISTINCTIVE:
        return None  # one name with different numbers ('Hall 1', 'Hall 2') or initials is no list of people
    nameish = sum(1 for v in judged if kinds[v] != OTHER)
    if nameish < NAMEISH_SHARE * n:
        return None
    multi = sum(1 for v in named if len(_value_words(v) or []) > 1 or name_layout_kind(v))
    if multi * 2 >= len(named):
        return "full"
    parts = [_part(v) for v in named]
    n_first, n_last = parts.count("first"), parts.count("last")
    if n_first > n_last:
        return "first"
    if n_last > n_first:
        return "last"
    return "full"
