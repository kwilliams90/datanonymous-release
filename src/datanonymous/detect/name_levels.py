"""Person names inside single values (R5-01): the copy gate's per-VALUE name screen.

The column-wide screen (``detect.names.name_value_verdict``, called by ``detect.text.value_name_detection``)
turns a column into PERSON_NAME only when most of its values are names. A value-copying column of ordinary
labels can still hold one real person among them (``Dismissal``: ``Bus`` / ``Walk`` / ``John Smith``), or
labels that carry a name (``Smith (Coach)``, ``Ms. Garcia - Office``, ``Pickup: Ana Lopez``). This module
judges every value on its own and returns the character spans of person-name content in it
(:func:`screen_levels`). The copy gate (``detect.text.copy_gate``) records the flagged values in the
detection's PRIVATE ``params['_name_levels']`` (:data:`NAME_LEVELS_KEY`): the column keeps its type and its
other levels (real values, exact counts, D11/D14 floors), and generation replaces each flagged value by the
same value with every name span replaced by a fake name of the span's layout (``generate.text.name_level_map``;
the whole value when the value is the name). The report and the docs show a flagged level as a placeholder
(``[person name 1]``, :func:`placeholder_of`) on both the real and the synthetic side, and the leak scanner
indexes the real name spans as person names (``privacy.leaks``: never excused as labels).

Rules (a *name word* is a word of the given-name or surname lists of ``detect.names.known_names``; it is
*distinctive* when it is no everyday word, ``detect.names.common_words``, and longer than two letters):

1. **title**: ``Mr.``/``Mrs.``/``Ms.``/``Miss``/``Mx.`` ... and one to three capitalised name words (the first
   may be a rare name: ``Mrs. Quenby``); weaker titles (``Dr.``, ``Prof.``, ``Rev.``, ``Fr.``, ``Sr.``,
   ``Coach``) only before a name word (``Coach Taylor``, not ``Coach Bus`` or ``Sr. Project``);
2. **initials**: ``J. Smith``, ``J.K. Smith``, ``Smith J.``, ``Smith, J.`` / ``Garcia Lopez, M. A.`` with a
   name word (not ``Plan B.``, ``Algebra, A.``);
3. **pair**: two to five name words in a row (particles and initials between them), one on the given-name
   list and another on the surname list, at least one distinctive, in either order (``John Smith``,
   ``Smith John``, ``Ana Lopez``, ``Maria de la Cruz``); inside a longer label too (``Pickup by Ana Lopez``,
   ``Ana Lopez Pickup``, ``John "Jack" Smith``); ``Surname, Given`` across a comma (``Smith, John``); a whole
   value in lower case only when both words are distinctive (``john smith``); an English given name that is no
   everyday word beside one capitalised word on no list and no English derived word (``John Quenby``,
   ``Quenby, John``; not ``Max Capacity``, ``Gene Therapy``);
4. **role**: one name word (or a pair) in a part of the value set off by brackets, a dash, a colon, a comma or
   a slash, when another part names a person's role (``Smith (Coach)``, ``Counselor - Garcia``, ``Teacher:
   Nguyen``, ``John Smith (Counselor)``);
5. **by**: a name word or pair right after ``by`` / ``with`` / ``w/`` / ``per`` (``Picked up by Garcia``);
6. **same person**: once a value of the column is flagged, its distinctive name words alone elsewhere in the
   column are that person too (``Smith`` beside ``Smith (Coach)``).

False-positive controls: a run of capitalised words holding an institutional or place word (:data:`THING_WORDS`:
``Lincoln High``, ``King Elementary``, ``Washington Hall``, ``Rose Garden``, ``Martin Luther King Day``) names a
thing after a person and is never a name (a two-word ``Given Hall`` / ``Given Park`` with a distinctive given
name is still a person: ``John Park``); everyday words never count as evidence (months, colours, ``Grant
Funded``, ``Will Call``, ``Head Start``); values that start like a place (``San``, ``Fort``, ``Mount``) are
places. Under a header of a thing named after people or places (``detect.text._thing_header``: ``House``,
``Team``, ``Program`` ...) bare pairs (rule 3) are not judged: the thing may be named after someone (``Marie
Curie`` house); titles, initials, roles, ``by`` phrases and the same person's words (rule 6) still are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from ..hints import is_role_word
from . import names as name_lists

#: PRIVATE detection/profile params key: the flagged values, each ``[value, [[start, end], ...], placeholder]``
#: (value = stripped text as the column holds it; spans of person-name content in it; the report label).
NAME_LEVELS_KEY = "_name_levels"
#: PRIVATE model key (``ColumnModel.extra`` of the row model): real raw value -> synthetic raw value of every
#: flagged value, built once per generation (``generate.text.name_level_map``).
NAME_MAP_KEY = "_name_level_map"
#: How the report and the docs show a flagged level (both sides): the name spans replaced by this.
PLACEHOLDER = "[person name {}]"

#: Honorifics that only ever precede a person's name.
_STRONG_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "mme", "mlle", "sra", "srta", "mstr", "messrs"})
#: Titles that also start other labels (``Sr. Project``, ``Coach Bus``, ``Dr. Seuss Day``): a name word must follow.
_WEAK_TITLES = frozenset({"dr", "prof", "rev", "fr", "sr", "coach", "pastor", "sister", "brother", "father",
                          "madam", "sir", "dame", "judge", "officer", "nurse", "dean", "principal", "supt"})
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
_PARTICLES = frozenset({"de", "del", "della", "la", "le", "van", "von", "der", "den", "da", "di", "du", "dos",
                        "das", "do", "bin", "ibn", "al", "y", "e", "mc", "mac", "ter", "ten", "st"})
#: Prepositions that introduce the person who did something (``Picked up by Garcia``).
_BY_WORDS = frozenset({"by", "with", "w", "per", "attn"})
#: Person-role nouns beside a name in a label, beyond ``hints.is_role_word`` (teacher, counselor, coach, nurse,
#: aide, driver, parent, contact ...).
_ROLE_NOUNS = frozenset({
    "sub", "substitute", "para", "secretary", "librarian", "custodian", "admin", "administrator", "registrar",
    "clerk", "receptionist", "dean", "director", "manager", "coordinator", "specialist", "liaison", "officer",
    "sro", "interpreter", "translator", "psych", "slp", "intern", "sponsor", "grandma", "grandpa", "aunt",
    "uncle", "sister", "brother", "cousin", "neighbor", "neighbour", "babysitter", "sitter", "nanny", "friend",
    "stepdad", "stepmom", "worker", "superintendent", "supt", "pastor", "rep", "representative",
})
#: Words that only qualify a role noun in such a part (``Bus Driver``, ``Asst. Coach``, ``Emergency Contact``).
_ROLE_MODIFIERS = frozenset({
    "bus", "head", "lead", "co", "case", "social", "vice", "assistant", "asst", "homeroom", "emergency",
    "pickup", "school", "class", "sped", "ell", "esl", "primary", "secondary", "acting", "interim", "of",
    "record", "the", "and", "main", "front", "office", "attendance", "health", "family", "home",
})
#: Institutional and place words: a run of capitalised words holding one names a thing (often after a person:
#: ``Lincoln High``, ``Washington Hall``, ``Martin Luther King Jr. Elementary``), never a person. One word
#: list of more than 50 entries, so the leak scanner treats it as data, never as text the tool writes.
THING_WORDS: frozenset[str] = frozenset({
    "school", "schools", "elementary", "elem", "middle", "high", "hs", "academy", "prep", "preparatory",
    "college", "university", "univ", "institute", "campus", "center", "centre", "hall", "halls", "building",
    "bldg", "wing", "annex", "library", "gym", "gymnasium", "field", "fields", "stadium", "arena", "park",
    "pavilion", "auditorium", "house", "lodge", "lab", "laboratory", "clinic", "hospital", "church", "chapel",
    "cathedral", "temple", "mosque", "synagogue", "street", "avenue", "ave", "road", "rd", "boulevard", "blvd",
    "drive", "lane", "ln", "way", "court", "ct", "place", "plaza", "square", "parkway", "pkwy", "highway",
    "hwy", "circle", "terrace", "trail", "bridge", "tower", "towers", "station", "terminal", "airport", "county",
    "city", "town", "village", "township", "district", "memorial", "award", "awards", "prize", "scholarship",
    "scholarships", "fellowship", "foundation", "fund", "trust", "society", "association", "club", "league",
    "cup", "trophy", "day", "days", "week", "festival", "program", "programs", "programme", "project", "magnet",
    "charter", "montessori", "kindergarten", "preschool", "daycare", "cafe", "store", "shop", "market", "farm",
    "ranch", "estates", "apartments", "apts", "heights", "gardens", "garden", "manor", "commons", "creek",
    "lake", "river", "mountain", "valley", "ridge", "hills", "hill", "woods", "forest", "beach", "bay",
    "island", "point", "springs", "falls", "harbor", "harbour", "port", "complex", "residence", "dorm",
    "lounge", "studio", "theater", "theatre", "museum", "zoo", "aquarium", "cemetery", "pool", "rink", "track",
    "room", "rm", "suite", "floor", "dept", "department", "institution", "conference", "summit", "camp",
    "classic", "invitational", "tournament", "series", "lecture", "series", "chair", "endowment", "grant",
    "challenge", "games", "olympics", "relay", "run", "walk", "race", "fair", "expo", "gala", "ball", "bowl",
    "tigers", "eagles", "lions", "bears", "panthers", "wildcats", "knights", "warriors", "falcons", "hawks",
    "rams", "mustangs", "bulldogs", "cougars", "spartans", "trojans", "titans", "pirates", "vikings", "raiders",
})
#: Everyday label words that are also on the name lists of some locale (``Walker``, ``Car Rider``, ``Faith
#: Christian``, ``Block Schedule``): never distinctive in a value, so they alone never make a name (a pair
#: with a distinctive name still does: ``Christian Smith``). One word list of more than 50 entries (data).
_LABEL_WORDS: frozenset[str] = frozenset({
    "walker", "walkers", "rider", "riders", "christian", "catholic", "baptist", "lutheran", "methodist",
    "episcopal", "cross", "moon", "sun", "math", "metro", "infant", "scarlet", "cousin", "plaza", "mountain",
    "wing", "van", "bridge", "block", "bell", "chip", "dash", "echo", "fair", "golden", "grove", "hill", "hood",
    "horn", "journey", "lane", "lord", "love", "lucky", "mills", "oak", "rush", "tank", "wise", "worth", "zion",
    "earnest", "honest", "ring", "rice", "nail", "sole", "penny", "pepper", "candy", "doll", "bliss",
    "blessing", "jewel", "lyric", "misty", "rocky", "rusty", "sandy", "dusty", "hay", "marsh", "moss", "buck",
    "bud", "lark", "fawn", "garnet", "ember", "flint", "pike", "carpool", "transfer", "general", "regular",
})
#: First words of place names (``San Diego``, ``Fort Wayne``, ``Mount Vernon``): such a run is a place.
_PLACE_STARTS = frozenset({"san", "santa", "santo", "sao", "saint", "st", "ste", "sainte", "los", "las", "el",
                           "fort", "ft", "mount", "mt", "port", "lake", "new", "north", "south", "east", "west",
                           "upper", "lower", "great", "little", "old"})

_TOKEN_RE = re.compile(r"(?P<w>[^\W\d_]+(?:['’\-][^\W\d_]+)*)(?P<dot>\.)?|(?P<d>\d+)|(?P<p>[^\w\s])")
#: Punctuation that sets off parts of a value (rule 4) and ends a run of capitalised words.
_SEPARATORS = frozenset("()[]{}:;|/\\,–—-•·@#&+=<>!?*~")
#: Quotes around a nickname inside a name (``John "Jack" Smith``): they do not end a run.
_QUOTES = frozenset("\"'“”‘’«»`")


@dataclass
class _Tok:
    text: str
    start: int
    end: int  # without a trailing dot
    dot: bool
    kind: str  # "w" word, "d" digits, "p" punctuation

    @property
    def stop(self) -> int:
        """End including the trailing dot."""
        return self.end + (1 if self.dot else 0)


@dataclass
class _Word:
    tok: _Tok
    fold: str
    cap: bool
    given: bool
    surname: bool
    common: bool
    thing: bool
    role: bool
    title: str | None  # "strong" / "weak"
    initial: bool
    particle: bool
    suffix: bool

    @property
    def name(self) -> bool:
        return self.given or self.surname

    @property
    def distinctive(self) -> bool:
        return self.name and not self.common and len(self.fold) > 2


def _lists() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    first, last = name_lists.known_names()
    return first, last, name_lists.common_words()


def _word(tok: _Tok) -> _Word:
    first, last, common = _lists()
    raw = tok.text
    f = name_lists.fold_word(raw)
    letters = [ch for ch in raw if ch.isalpha()]
    initial = len(letters) == 1
    given = f in first
    surname = f in last
    if not (given or surname) and ("-" in raw or "'" in raw or "’" in raw):
        parts = [name_lists.fold_word(p) for p in re.split(r"[-'’]", raw) if p]
        long_parts = [p for p in parts if len(p) > 1]
        if long_parts and all(p in first or p in last for p in long_parts):
            surname = any(p in last for p in long_parts)  # Smith-Jones, Garcia-Lopez
            given = not surname or any(p in first for p in long_parts)
        elif len(parts) == 2 and len(parts[0]) <= 2 and parts[1] in last:
            surname = True  # O'Brien, D'Angelo
    title = "strong" if f in _STRONG_TITLES else "weak" if f in _WEAK_TITLES else None
    return _Word(tok=tok, fold=f, cap=raw[:1].isupper(), given=given and not initial, surname=surname and not initial,
                 common=f in common or len(f) <= 2 or f in _LABEL_WORDS, thing=f in THING_WORDS, role=_is_role(f),
                 title=title, initial=initial,
                 particle=(f in _PARTICLES and not raw[:1].isupper()) or f in ("mc", "mac"), suffix=f in _SUFFIXES)


@lru_cache(maxsize=1)
def _us_given_names() -> frozenset[str]:
    """Given names of the English (US) person list of the pinned Faker: a ``Given Hall`` / ``Given Park`` run
    with one of them is a person (``John Park``), and so is ``Given Rare`` (``John Quenby``); the multi-locale
    lists also hold ``Smith``, ``Park`` ... as given names."""
    try:
        from faker.providers.person.en_US import Provider

        src = list(getattr(Provider, "first_names", ()))
    except Exception:  # pragma: no cover - Faker is a pinned dependency
        src = []
    return frozenset(name_lists.fold_word(n) for n in src if isinstance(n, str))


def _is_role(f: str) -> bool:
    return f in _ROLE_NOUNS or is_role_word(f)


def _tokens(value: str) -> list[_Tok]:
    out = []
    for m in _TOKEN_RE.finditer(value):
        if m.group("w") is not None:
            out.append(_Tok(m.group("w"), m.start("w"), m.end("w"), m.group("dot") is not None, "w"))
        elif m.group("d") is not None:
            out.append(_Tok(m.group("d"), m.start(), m.end(), False, "d"))
        else:
            out.append(_Tok(m.group("p"), m.start(), m.end(), False, "p"))
    return out


def _runs(toks: list[_Tok]) -> list[list[_Word]]:
    """Maximal runs of capitalised words (titles, initials and lower-case particles between them), split at
    punctuation, digits, lower-case words and a period after a long word (a sentence end)."""
    runs: list[list[_Word]] = []
    cur: list[_Word] = []
    for i, t in enumerate(toks):
        if t.kind == "p" and t.text in _QUOTES and cur:
            continue  # a quoted nickname inside a name
        if t.kind != "w":
            if cur:
                runs.append(cur)
            cur = []
            continue
        w = _word(t)
        if w.cap or (w.particle and cur and _capital_after_particles(toks, i)):
            cur.append(w)
            if t.dot and not (w.initial or w.title or w.suffix) and len(w.fold) > 2:
                runs.append(cur)  # 'Smith.' ends a sentence
                cur = []
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _capital_after_particles(toks: list[_Tok], i: int) -> bool:
    """A capitalised word follows the lower-case particles from token ``i`` on (``de la Cruz``)."""
    j = i + 1
    while j < len(toks) and toks[j].kind == "w" and toks[j].text.casefold() in _PARTICLES \
            and not toks[j].text[:1].isupper():
        j += 1
    return j < len(toks) and toks[j].kind == "w" and toks[j].text[:1].isupper()


def _thing_run(run: list[_Word]) -> bool:
    """The run names a thing (an institutional or place word in it, or a place start), not a person;
    a two-word ``Given Hall`` with a distinctive English given name is a person (``John Park``, ``Sally
    Field``)."""
    names = [w for w in run if not (w.title or w.initial or w.particle or w.suffix)]
    if names and names[0].fold in _PLACE_STARTS and len(names) > 1:
        return True
    things = [w for w in names if w.thing]
    if not things:
        return False
    if len(names) == 2 and names[1].thing and names[1].surname and not names[0].thing and not names[0].common \
            and names[0].fold in _us_given_names():
        return False
    return True


Span = tuple[int, int]


def _pair_ok(words: list[_Word]) -> bool:
    """Two to five name words, one given and another a surname, at least one distinctive."""
    names = [w for w in words if not (w.initial or w.particle or w.suffix or w.title)]
    if not 2 <= len(names) <= 5 or not all(w.name for w in names):
        return False
    if not any(w.distinctive for w in names):
        return False
    return any(a.given and b.surname for a in names for b in names if a is not b)


#: English given names that are also everyday words (``Max Capacity``, ``Mark Complete``, ``Gene Therapy``,
#: ``Ray Tracing``, ``Tom Cat``): never the given name of a ``Given Rare`` pair.
_EVERYDAY_GIVEN = frozenset({
    "bob", "chase", "dean", "don", "drew", "frank", "gene", "harry", "jack", "lance", "mark", "max", "misty",
    "penny", "ray", "roy", "sue", "tom", "will", "bill", "pat", "rob", "art", "guy", "nick", "kit", "val",
    "buck", "rusty", "sandy", "candy", "ginger", "honey", "cherry", "skip", "sonny", "buddy", "junior", "cliff",
    "dale", "glen", "heath", "brook", "ash", "reed", "wade", "hunter", "mason", "tucker", "cooper", "carter",
    "price", "sterling", "miles", "grant", "hugh", "rich", "rocky", "colt", "lee", "jean", "joan", "gail",
    "spencer", "chuck", "dusty", "jewel", "royal", "noble", "prince", "king", "duke", "earl", "major",
})
#: Endings of English derived words (``Enrollment``, ``Capacity``, ``Tracing``): such a word is no rare surname.
_ENGLISH_ENDINGS = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship", "ing", "ics", "ology", "ism",
                    "ist", "ive", "ous", "ful", "less", "able", "ible", "ary", "ery", "ory", "ure", "ed", "ize",
                    "ise", "al", "ly", "hood", "dom", "ward", "wise", "logy", "graphy", "phy", "cy")


def _rare(w: _Word) -> bool:
    """A capitalised word on no list (a rare surname such as ``Quenby``): no everyday, thing or role word, no
    English derived word."""
    return (w.cap and not (w.name or w.common or w.thing or w.role or w.title or w.initial or w.particle
                           or w.suffix) and len(w.fold) >= 3 and w.tok.text.replace("-", "").replace("'", "").isalpha()
            and not w.fold.endswith(_ENGLISH_ENDINGS))


def _given_and_rare(a: _Word, b: _Word) -> bool:
    """A two-word run of a distinctive English given name and a rare word (``John Quenby``)."""
    return (a.given and a.distinctive and a.fold in _us_given_names() and a.fold not in _EVERYDAY_GIVEN
            and not a.thing and _rare(b))


def _windows(run: list[_Word]) -> list[list[_Word]]:
    """Maximal stretches of name words (with initials, particles and suffixes between them) in a run."""
    out: list[list[_Word]] = []
    cur: list[_Word] = []
    for w in run:
        if (w.name and not (w.title or (w.role and not w.distinctive))) or w.initial or w.particle or (
                w.suffix and cur):
            cur.append(w)
        else:
            if cur:
                out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    # trim particles/initial-less edges ('de' at an end is no name part)
    trimmed = []
    for win in out:
        while win and (win[0].particle or win[0].suffix):
            win = win[1:]
        while win and win[-1].particle:
            win = win[:-1]
        if win:
            trimmed.append(win)
    return trimmed


def _span(words: list[_Word]) -> Span:
    return words[0].tok.start, words[-1].tok.stop if (words[-1].initial or words[-1].title or words[-1].suffix) \
        else words[-1].tok.end


def _title_spans(run: list[_Word]) -> list[Span]:
    """Rule 1 inside a run: a title and the name words after it."""
    out = []
    i = 0
    while i < len(run) - 1:
        t = run[i]
        if not t.title:
            i += 1
            continue
        j = i + 1
        taken: list[_Word] = []
        while j < len(run) and len([w for w in taken if not w.initial]) < 3:
            w = run[j]
            if w.initial or w.particle:
                taken.append(w)
            elif w.name and not w.thing and not (w.role and not w.distinctive) and (w.surname or not w.common):
                taken.append(w)  # an everyday word only as a surname ('Mr. Brown', not 'Miss America')
            elif not taken and t.title == "strong" and not (w.common or w.thing or w.role or w.title):
                taken.append(w)  # a rare name after an honorific ('Mrs. Quenby')
            elif w.suffix and taken:
                taken.append(w)
                j += 1
                break
            else:
                break
            j += 1
        named = [w for w in taken if not (w.initial or w.particle or w.suffix)]
        if named and (t.title == "strong" or any(w.name for w in named)):
            if t.title == "weak" and _thing_run(run):
                break
            out.append((t.tok.start, _span(taken)[1]))
            i = j
        else:
            i += 1
    return out


def _initial_spans(run: list[_Word]) -> list[Span]:
    """Rule 2 inside a run: ``J. Smith``, ``J.K. Smith``, ``Smith J.``."""
    out = []
    n = len(run)
    # 'J. Smith', 'J.K. Smith', 'J. K. Smith Jr.'
    i = 0
    while i < n:
        if run[i].initial and run[i].tok.dot:
            j = i
            while j < n and run[j].initial and run[j].tok.dot and j - i < 2:
                j += 1
            k = j
            while k < n and run[k].name and not run[k].thing and k - j < 2:
                k += 1
            if k > j and not (k < n and run[k].thing):
                end = run[k - 1].tok.end
                if k < n and run[k].suffix:
                    end = run[k].tok.stop
                out.append((run[i].tok.start, end))
                i = k
                continue
        i += 1
    # 'Smith J.' at the end of a run
    if n >= 2 and run[-1].initial and run[-1].tok.dot and run[-2].name and not run[-2].thing:
        start = run[-2].tok.start
        if n >= 3 and run[-3].name and not run[-3].thing:
            start = run[-3].tok.start
        if not any(s <= start < e for s, e in out):
            out.append((start, run[-1].tok.stop))
    return out


def _surname_comma_spans(toks: list[_Tok], value: str, thing: bool) -> list[Span]:
    """``Smith, J.`` / ``Garcia Lopez, M. A.`` (any header) and ``Smith, John`` (not under a thing header):
    a part of the value set off by separators."""
    out = []
    for ci, t in enumerate(toks):
        if t.kind != "p" or t.text != ",":
            continue
        # left: 1-3 capitalised name words right before the comma (a label word before them ends the name:
        # 'Pickup Smith, J.')
        left: list[_Word] = []
        k = ci - 1
        while k >= 0 and toks[k].kind == "w" and len(left) < 3:
            w = _word(toks[k])
            if not w.cap or w.title or not (w.name or w.particle) or w.thing or w.initial:
                break
            left.insert(0, w)
            k -= 1
        while left and left[0].particle:
            left = left[1:]
        if not left and not thing and ci >= 1 and toks[ci - 1].kind == "w" and _rare(_word(toks[ci - 1])) \
                and (ci < 2 or toks[ci - 2].kind != "w") and ci + 2 <= len(toks) and toks[ci + 1].kind == "w":
            # 'Quenby, John': a rare surname before an English given name, the whole part of the value
            g = _word(toks[ci + 1])
            if (ci + 2 == len(toks) or (toks[ci + 2].kind == "p" and toks[ci + 2].text in _SEPARATORS)) \
                    and _given_and_rare(g, _word(toks[ci - 1])):
                out.append((toks[ci - 1].start, toks[ci + 1].end))
            continue
        if not left or (k >= 0 and toks[k].kind == "w" and _word(toks[k]).thing):
            continue
        # right: initials (with or without dots) or 1-2 given names, up to a separator or the end
        right: list[_Word] = []
        k = ci + 1
        while k < len(toks) and toks[k].kind == "w" and len(right) < 3:
            w = _word(toks[k])
            if not w.cap:
                break
            right.append(w)
            k += 1
        if not right or (k < len(toks) and not (toks[k].kind == "p" and toks[k].text in _SEPARATORS)):
            continue
        end = right[-1].tok.stop if (right[-1].initial or right[-1].suffix) else right[-1].tok.end
        if all(w.initial for w in right):
            if not any(w.surname or w.given for w in left):
                continue
            if not any(w.tok.dot for w in right) and not any(w.surname and w.distinctive for w in left):
                continue  # 'Algebra, A' is a course section
            out.append((left[0].tok.start, end))
        elif not thing:
            names = [w for w in right if not (w.initial or w.suffix)]
            if names and all(w.given for w in names) and any(w.surname for w in left) \
                    and any(w.distinctive for w in left + names):
                out.append((left[0].tok.start, end))
    return out


def _segments(toks: list[_Tok]) -> list[list[_Tok]]:
    segs: list[list[_Tok]] = []
    cur: list[_Tok] = []
    for t in toks:
        if t.kind == "p" and t.text in _SEPARATORS:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def _role_spans(toks: list[_Tok]) -> list[Span]:
    """Rule 4: a part holding one name word beside a part naming a role (``Smith (Coach)``)."""
    segs = _segments(toks)
    if len(segs) < 2:
        return []
    words = [[_word(t) for t in seg if t.kind == "w"] for seg in segs]
    role_seg = [bool(ws) and len(ws) <= 3 and any(w.role for w in ws)
                and all(w.role or w.fold in _ROLE_MODIFIERS for w in ws)
                and all(t.kind == "w" for t in seg) for seg, ws in zip(segs, words)]
    if not any(role_seg):
        return []
    out = []
    for seg, ws, is_role in zip(segs, words, role_seg):
        if is_role or len(seg) != len(ws):
            continue
        core = [w for w in ws if not (w.title or w.initial or w.particle or w.suffix)]
        if len(core) == 1 and core[0].cap and core[0].name and not core[0].thing and not core[0].role \
                and len(core[0].fold) > 2 and core[0].fold not in _LABEL_WORDS:
            out.append((ws[0].tok.start, _span(ws)[1]))
        elif len(core) > 1 and all(w.cap for w in core) and not any(w.thing for w in core) and _pair_ok(ws):
            out.append((ws[0].tok.start, _span(ws)[1]))  # a pair beside a role: a person under any header
    return out


def _by_spans(toks: list[_Tok]) -> list[Span]:
    """Rule 5: a name word or pair right after ``by`` / ``with`` / ``w/`` / ``per`` (also inside an all-caps
    value: ``PICKED UP BY SMITH``), not a thing named after someone (``by Lincoln High``)."""
    out = []
    for i, t in enumerate(toks):
        if t.kind != "w" or t.text.casefold() not in _BY_WORDS or t.dot:
            continue
        j = i + 1
        if t.text.casefold() == "w":
            if j >= len(toks) or toks[j].text != "/":
                continue
            j += 1
        win: list[_Word] = []
        while j < len(toks) and toks[j].kind == "w" and len(win) < 5:
            w = _word(toks[j])
            lower_name = not w.cap and w.distinctive and not w.particle  # 'picked up by ana lopez'
            if not (w.cap or lower_name) or not (w.name or w.initial or w.particle) or w.title or w.thing:
                break
            win.append(w)
            if toks[j].dot and not w.initial:
                break
            j += 1
        if not win or (j < len(toks) and toks[j].kind == "w" and _word(toks[j]).thing):
            continue
        names = [w for w in win if not (w.initial or w.particle or w.suffix)]
        if any(not w.cap for w in names) and not (len(names) >= 2 and all(w.distinctive for w in names)):
            continue  # lower case: only a pair of distinctive names
        if (len(names) == 1 and names[0].distinctive) or _pair_ok(win):
            out.append(_span(win))
    return out


def _merge(spans: Iterable[Span]) -> list[Span]:
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def value_name_spans(value: str, thing: bool = False) -> list[Span]:
    """Character spans of person-name content in ``value`` by rules 1-5 of the module docstring (``thing``: a
    header of a thing named after people or places, where bare pairs are not judged). Empty when none."""
    v = value
    if not v or not any(ch.isalpha() for ch in v) or len(v) > 500:
        return []
    toks = _tokens(v)
    runs = _runs(toks)
    spans: list[Span] = []
    for run in runs:
        spans += _title_spans(run)
        spans += _initial_spans(run)
        if not thing and not _thing_run(run):
            for win in _windows(run):
                if _pair_ok(win):
                    spans.append(_span(win))
            if len(run) == 2 and _given_and_rare(run[0], run[1]):
                spans.append(_span(run))  # 'John Quenby': an English given name and a rare surname
    spans += _surname_comma_spans(toks, v, thing)
    spans += _role_spans(toks)
    spans += _by_spans(toks)
    if not spans and not thing:
        spans += _lower_pair(v)
    return _merge(spans)


def _lower_pair(value: str) -> list[Span]:
    """A whole lower-case value of two or three name words, every one distinctive (``john smith``)."""
    s = value.strip()
    words = s.split()
    if not 2 <= len(words) <= 3 or not all(w.isalpha() and w.islower() for w in words):
        return []
    first, last, common = _lists()
    folded = [name_lists.fold_word(w) for w in words]
    if not all((f in first or f in last) and f not in common and len(f) > 2 for f in folded):
        return []
    if not any(a in first and b in last for a in folded for b in folded if a != b):
        return []
    start = value.index(s)
    return [(start, start + len(s))]


def _span_words(value: str, span: Span) -> set[str]:
    out = set()
    for t in _tokens(value[span[0]:span[1]]):
        if t.kind != "w":
            continue
        w = _word(t)
        if w.distinctive and not (w.title or w.initial or w.particle or w.suffix):
            out.add(w.fold)
    return out


def _same_person_spans(value: str, words: set[str]) -> list[Span]:
    """Rule 6: runs whose every name word is a distinctive name word of a flagged value of the column."""
    out = []
    for run in _runs(_tokens(value)):
        if _thing_run(run):
            continue
        for win in _windows(run):
            core = [w for w in win if not (w.title or w.initial or w.particle or w.suffix)]
            if core and all(w.fold in words for w in core):
                out.append(_span(win))
    return out


def screen_levels(values: Iterable[str], thing: bool = False) -> dict[str, list[Span]]:
    """Flagged values (as given) -> their person-name spans, over the distinct ``values`` of one column
    (rules 1-6 of the module docstring)."""
    vals = list(dict.fromkeys(v for v in values if isinstance(v, str)))
    found: dict[str, list[Span]] = {}
    for v in vals:
        sp = value_name_spans(v, thing)
        if sp:
            found[v] = sp
    if found:
        words: set[str] = set()
        for v, sp in found.items():
            for s in sp:
                words |= _span_words(v, s)
        if words:
            for v in vals:
                extra = _same_person_spans(v, words)
                if extra:
                    found[v] = _merge(found.get(v, []) + extra)
    return found


# --------------------------------------------------------------------------- the private record
def record(values: dict[str, int], found: dict[str, list[Span]]) -> list[list[Any]]:
    """The :data:`NAME_LEVELS_KEY` record: ``[value, spans, placeholder]`` per flagged value, the most frequent
    first. Case forms of one value (``JOHN SMITH`` / ``John Smith``) share one placeholder number."""
    groups: dict[str, int] = {}
    for v in found:
        groups[v.casefold()] = groups.get(v.casefold(), 0) + int(values.get(v, 0))
    order = sorted(groups, key=lambda k: (-groups[k], k))
    number = {k: i + 1 for i, k in enumerate(order)}
    out = []
    for v in sorted(found, key=lambda x: (number[x.casefold()], x)):
        n = number[v.casefold()]
        out.append([v, [list(s) for s in found[v]], placeholder_text(v, found[v], n)])
    return out


def placeholder_text(value: str, spans: list[Span], n: int) -> str:
    """``value`` with each name span replaced by :data:`PLACEHOLDER` numbered ``n``."""
    out, pos = [], 0
    for s, e in spans:
        out.append(value[pos:s])
        out.append(PLACEHOLDER.format(n))
        pos = e
    out.append(value[pos:])
    return "".join(out)


def flagged(p: Any) -> dict[str, tuple[list[Span], str]]:
    """value (stripped, as the column holds it) -> (spans, placeholder) of a profile's or detection's flagged
    values; empty for a column without any."""
    rec = (getattr(p, "params", None) or {}).get(NAME_LEVELS_KEY) or ()
    return {r[0]: ([tuple(s) for s in r[1]], r[2]) for r in rec}


def name_texts(p: Any) -> list[str]:
    """The real person-name text of every flagged span of a column (titles left out: ``Ms. Garcia`` ->
    ``Garcia``): what the leak scanner indexes as person names (``privacy.leaks``)."""
    out: dict[str, None] = {}
    for v, (spans, _) in flagged(p).items():
        for s, e in spans:
            text = v[s:e]
            toks = [t for t in _tokens(text) if t.kind == "w"]
            while toks and _word(toks[0]).title and len(toks) > 1:
                text = text[toks[1].start:]
                toks = toks[1:]
            text = text.strip()
            if text:
                out[text] = None
    return list(out)


def display_map(p: Any) -> dict[str, str]:
    """Stripped text -> report label for a column with flagged values: each real flagged value and its synthetic
    replacement (``params`` + the generation map in ``p.model.extra``) show as the same placeholder, so the
    real and synthetic level shares line up and no name is printed (D21)."""
    fl = flagged(p)
    if not fl:
        return {}
    out = {v: ph for v, (_, ph) in fl.items()}
    model = getattr(p, "model", None)
    mapping = (model.extra.get(NAME_MAP_KEY) if model is not None else None) or {}
    for real, fake in mapping.items():
        ph = fl.get(str(real).strip(), (None, None))[1]
        if ph is not None:
            out.setdefault(str(fake).strip(), ph)
    return out


def kept_levels(p: Any) -> set[str]:
    """The stripped real values of a column with flagged values that are written verbatim (not flagged)."""
    fl = flagged(p)
    model = getattr(p, "model", None)
    if model is None:
        return set()
    return {str(k).strip() for k in model.nonmissing_counts if k is not None and str(k).strip() not in fl}
