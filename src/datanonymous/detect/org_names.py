"""D36: person names inside organisational codes (AREA C; design 0.3 D36, user decision 2026-10-07).

An organisational code (D10) keeps its real values (``detect.text.unit_kept``), but a section or homeroom code often
carries its teacher's name: ``ALG1-03 Garcia, Ana``, ``Algebra I - Ms. Garcia``, ``English 9 (A. Garcia)``. Such a
value keeps everything except the name part, which gets a consistent fake name of the same layout (``ALG1-03 Lopez,
Maria``, ``Algebra I - Ms. Lopez``): the course part stays real and joins on the code still match, because the same
real value gets the same synthetic value in every file and column of the run (``generate.pseudonyms``).

What counts as a person name is decided by layouts, never by new name lists:

* the D34 checks of headers and paths (``privacy.headers.person_name_hit``: explicit person layouts, ``Garcia, Ana``,
  ``Smith, J.``, ``A. Garcia``, ``Ms. Garcia``, ``Mr Smith``, ``Mrs. G``, ``Counselor Nwosu``, user names such as
  ``jgarcia``, and known first + last name pairs, ``Ana Garcia``; with its exemptions for schools, places, namesakes
  and programs: ``Jesse H Jones High School``, ``Rosa Parks Elementary``, ``Allen, Texas``);
* and layouts of their own (:meth:`_Scan.layout`), because a false alarm here only costs a fake word in a code,
  not a refused file as for a header, so the D34 check's precision rules (no common English word, no everyday
  surname) give way where the layout itself says "a person": ``Surname, Given`` with any name of the lists before the
  comma, common ones too (``Miller, Lisa``, ``West, Ana``, ``Le, Thanh``, ``Wu, Lin``), or a word off the lists before
  a known given name (``D'Souza, Priya``, ``Ng, Wei``), and after it a given name, a hyphenated one (``Park,
  Ji-hoon``) or a capital initial (``Miller, L``) (:meth:`_Scan.comma_name`); a known given name and a surname side by
  side when one of the two is a distinctive name, or two common US names (``Lisa Miller``, ``David Young``, ``Mark
  Brown``, ``lisa.miller``: :meth:`_Scan.pair`), and a hyphenated given name before an East Asian surname (``Ji-hoon
  Kim``); an honorific that names a person whatever follows it (``Mr``, ``Mrs``, ``Ms.``, ``Mx``, ``Sra``, ``Herr``
  ...: ``privacy.headers._INITIAL_TITLES``) before a capitalised word or particle, also a surname the name lists do
  not hold or read as an everyday or eponym word (``Ms. Le``, ``Mr. Vo``, ``Ms. de la Cruz``, ``Mrs. Van Der Berg``,
  ``Mr. Okafor Debate``); ``Ms`` / ``MS`` without a period is a middle school as often (``MS Band``, ``Ms Science 7``,
  ``MS HOPE SQUAD``), so only a surname of the lists counts after it, an everyday one only where the name ends
  (:meth:`_Scan.ms_surname`: ``Ms Garcia``, ``Ms Brown``), and a window of the D34 check that reads such a course as a
  title is dropped (:meth:`_Scan.middle_school`); another title of the D34 check (``Dr``, ``Coach``, ``Counselor``
  ...) before a distinctive known name, as there; and a distinctive surname, a comma and two or three capital
  initials run together (``Abernathy, RJ``; not a state code: ``Austin, TX``). Course words stay (``Biology,
  Honors``, ``French, I``, ``Math Block``, ``Smith Hall``). Each keeps D34's school, company, place and namesake
  exemptions (``Mrs. Quenby's Academy``, ``Mr. Patel Brothers``, ``John Taylor Elementary``:
  ``privacy.headers._place_after`` / ``_place_before``), but only across plain spaces: in a longer text a period or
  comma ends the name (``with Dr. Patel. Co-taught by ...`` names a teacher; the D34 check reads the sentence after
  it as a company).

A bare single surname without a layout (``Homeroom`` = ``Garcia``) is no layout and follows D10 unchanged: the
accepted residual risk, as in D35.

:func:`find` locates the name part of one value (:class:`OrgName`): the smallest run of words the D34 check still
reads as a person's name (dropping words from either end while it does), or a layout of its own, then widened
fail-safe over the whole name (:meth:`_Scan.extent`: replacing a word too many only costs a fake word in the code,
one too few writes a real name): the name words, particles, initials and hyphenated parts right beside it (``Robert J
Thompson``, ``Mary K. Johnson``, ``Garcia Lopez, Maria``, ``De La Cruz, Ana``, ``Okonkwo-Adeyemi, Ana``), the
surname before a comma when the part found is the given name (``Thompson, Robert J``: the D34 check reads ``Robert J``
alone), the given names and initials after the comma (``Garcia, Ana Maria L``), and after an honorific its surname,
with the given name before it (``Ms. Kim Soo-jin``); then the next name in the rest of the value (``Garcia, Ana /
Lopez, Maria``), up to :data:`MAX_NAMES`. Inside a name part, titles (``Ms``, ``Coach``, ``Counselor``), suffixes
(``Jr``), lower-case particles (``de la``) and digits stay; every other word is replaced (role ``first`` or
``last``: before a comma the surname, after it the given names; otherwise the last word is the surname) and so is
every initial (role ``initial``). The pieces are placed on the value's own characters letter by letter (:func:`align`:
an NFD accent, a ``ß`` that reads ``ss``, an ``Æ`` that reads ``AE``); a value whose name part cannot be placed so (a
name in another script, a word transliterated whole) gets ``tokens`` None: it is pseudonymised whole like a D35 unit
(fail-safe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

#: Most names located in one value (co-teachers); a value with more is pseudonymised whole.
MAX_NAMES = 6
#: Roles of a replaced token.
FIRST, LAST, INITIAL = "first", "last", "initial"
#: Token kinds of ``privacy.headers`` that stay as they are inside a name part (``Ms``, ``Coach``, ``Jr``, ``12``).
_KEPT_KINDS = frozenset({"title", "suffix", "digit"})
_HYPHENS = frozenset({"-", "‐", "‑"})
#: Separators that join two words of one name (``Garcia Lopez``, ``Okonkwo-Adeyemi``, ``AnaGarcia``).
_NAME_JOINS = frozenset({"", " "}) | _HYPHENS
#: Token kinds a surname right after an honorific may have: a name, a word off the lists, a particle (``Ms. Le``),
#: and an everyday, institution, place or eponym word (``Ms. Price``, ``Mr. Hall``, ``Mr. Okafor Debate``).
_SURNAME_KINDS = frozenset({"name", "other", "particle", "word", "inst", "place", "program"})
_POSSESSIVE = re.compile(r"['’]s$")
_LETTERS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
#: A separator between two names written without a comma (``Lisa Miller/John Taylor``, ``Ana Garcia & Luis Ortiz``).
_SEPARATOR = re.compile(r"[^\s.'’\-‐‑]")


@dataclass(frozen=True)
class OrgName:
    """The person-name content of one organisational-code value (PRIVATE: it locates real names).

    ``tokens``: (start, end, role) of each piece of the value to replace, in order (role :data:`FIRST`, :data:`LAST`
    or :data:`INITIAL`), or None when the name part cannot be located (the value is pseudonymised whole); ``parts``:
    the name layouts as written (``Garcia, Ana``, ``Ms. Garcia``), which the leak scanner indexes as sensitive text.
    """

    tokens: tuple[tuple[int, int, str], ...] | None
    parts: tuple[str, ...]


def find(text: str) -> OrgName | None:
    """The person name(s) inside organisational-code value ``text`` (module docstring), or None when it holds no
    explicit person layout (the value then keeps its real text, D10)."""
    if not isinstance(text, str) or not text.strip():
        return None
    return _find(text)


@lru_cache(maxsize=1 << 16)
def _find(text: str) -> OrgName | None:
    from ..privacy import headers as names

    d34 = names.person_name_hit(text) is not None
    s = names.clean_text(text)
    toks = names._tokens(s) if any(ch.isalpha() for ch in s) else []
    if not toks:
        return OrgName(None, ()) if d34 else None  # a name the Latin tokens cannot read ('ALG 张伟')
    sc = _Scan(s, toks, names._token_kinds(toks, s), names)
    active = [True] * len(toks)
    if not d34 and sc.layout(active) is None:
        return None
    where = align(text, s)
    if where is None:  # folding that no character-by-character reading reproduces: no safe location
        return OrgName(None, ())
    windows: list[tuple[int, int]] = []
    masked = s

    def mask(lo: int, hi: int) -> None:
        nonlocal masked
        for k in range(lo, hi + 1):
            active[k] = False
        masked = masked[:toks[lo].start] + " " * (toks[hi].end - toks[lo].start) + masked[toks[hi].end:]

    def take(lo: int, hi: int) -> bool:
        lo, hi = sc.extent(lo, hi, active)
        if not all(active[lo:hi + 1]):  # around a name found before: no clean place for this one
            return False
        windows.append((lo, hi))
        mask(lo, hi)
        return True

    hit = d34  # the D34 layouts (the value as cleaned reads like the value itself)
    vetoed = False
    while hit:
        w = _window(masked, toks, active, names, sc.case_info) if len(windows) < MAX_NAMES else None
        if w is None:
            return OrgName(None, ())
        if sc.middle_school(*w):  # 'MS HOPE SQUAD', 'Ms Band': a middle school's course, no honorific
            mask(*w)
            vetoed = True
        elif not take(*w):
            return OrgName(None, ())
        hit = names.person_name_hit(masked) is not None
    while (w := sc.layout(active)) is not None:  # the layouts of D36's own
        if len(windows) >= MAX_NAMES or not take(*w):
            return OrgName(None, ())
    if vetoed and not windows:
        return None
    starts, ends = where
    out: list[tuple[int, int, str]] = []
    parts: list[str] = []
    for lo, hi in sorted(windows):
        got: list[tuple[int, int, str]] = []
        for a, b, role in sc.roles(lo, hi):
            spot = place(text, s, starts, ends, a, b)
            if spot is None:  # accent folding moved characters: no safe location
                return OrgName(None, ())
            oa, ob = spot
            if got and oa < got[-1][1]:  # two pieces from one letter ('Æbelø' reads 'A' 'Ebelo'): one piece
                poa, pob, prole = got.pop()
                oa, ob = poa, max(ob, pob)
                role = LAST if LAST in (role, prole) else FIRST if FIRST in (role, prole) else INITIAL
            got.append((oa, ob, role))
        out.extend(got)
        parts.append(text[starts[toks[lo].start]:ends[_core_end(toks[hi]) - 1]])
    if not out:
        return OrgName(None, tuple(parts))
    return OrgName(tuple(out), tuple(parts))


def place(text: str, s: str, starts: list[int], ends: list[int], a: int, b: int) -> tuple[int, int] | None:
    """Where the piece ``s[a:b]`` of ``s`` (``text`` as the D34 checks read it, :func:`align` giving ``starts`` and
    ``ends``) sits in ``text``, widened to whole letters (``Ebelo`` of ``AEbelo`` is ``Æbelø``), or None when the text
    there does not read as that piece."""
    from ..privacy.headers import clean_text

    while a > 0 and starts[a - 1] == starts[a]:  # a piece that starts inside a letter's reading
        a -= 1
    while b < len(s) and ends[b] == ends[b - 1]:
        b += 1
    oa, ob = starts[a], ends[b - 1]
    return (oa, ob) if clean_text(text[oa:ob]) == s[a:b] else None


def align(text: str, s: str | None = None) -> tuple[list[int], list[int]] | None:
    """Where each character of ``s``, ``text`` as the D34 checks read it (``privacy.headers.clean_text``: accents and
    ligatures folded, look-alike letters made Latin), comes from in ``text``: (start, end) lists, one entry per
    character of ``s``, of the piece of ``text`` that gives it (a letter with its combining marks: an NFD ``u`` and
    acute, a ``ß`` that reads ``ss``, an ``Æ`` that reads ``AE``). None when ``text`` cleaned piece by piece does not
    give ``s`` (a word transliterated whole whose letters read otherwise alone): no safe place for a name piece."""
    from ..privacy.headers import clean_text

    if s is None:
        s = clean_text(text)
    if text.isascii():
        return (list(range(len(s))), list(range(1, len(s) + 1))) if s == text else None
    import unicodedata

    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and (unicodedata.combining(text[j]) or unicodedata.category(text[j]) == "Cf"):
            j += 1
        piece = clean_text(text[i:j])
        starts.extend([i] * len(piece))
        ends.extend([j] * len(piece))
        i = j
    if len(starts) != len(s) or "".join(clean_text(text[a:b]) for a, b in dict.fromkeys(zip(starts, ends))) != s:
        return None
    return starts, ends


#: A long value (a course description) is first read in overlapping runs of this many active words (a person's name
#: with its title, initials and particles is far shorter), so each D34 check reads a few words, not the whole text.
LOCAL_WORDS = 12


def _window(masked: str, toks: list[Any], active: list[bool], names: Any,
            case_info: bool = False) -> tuple[int, int] | None:
    """The smallest run of the active tokens that ``person_name_hit`` still reads as a person's name: words are dropped
    from the left while it does, then from the right (a window's text runs to the next token, so a trailing period or
    comma stays with it). Big steps first, halved when a step loses the name, so a long description costs a few
    dozen checks, not one per word. In a value of more than twice :data:`LOCAL_WORDS` words the runs of that many
    words (half overlapping) are read first, and the first that holds a name is narrowed: a run read out of its
    value can only find more (fail-safe: a fake word more in a code), and when none holds one the whole value is
    narrowed. A run without a capital letter in a value written in mixed case is skipped (the value's case says its
    lower-case words are no names; the whole value is read when only such a run holds one: ``jgarcia``)."""
    idx = [i for i, a in enumerate(active) if a]
    if not idx:
        return None

    def hit(a: int, b: int) -> bool:
        end = toks[idx[b + 1]].start if b + 1 < len(idx) else len(masked)
        return names.person_name_hit(masked[toks[idx[a]].start:end]) is not None

    if len(idx) > 2 * LOCAL_WORDS:
        for a in range(0, len(idx) - 1, LOCAL_WORDS // 2):
            b = min(len(idx) - 1, a + LOCAL_WORDS - 1)
            run_text = masked[toks[idx[a]].start:toks[idx[b]].end]
            if (not case_info or any(ch.isupper() for ch in run_text)) and hit(a, b):
                return _narrow(a, b, hit, idx)
            if b == len(idx) - 1:
                break
    a, b = 0, len(idx) - 1
    if not hit(a, b):
        return None
    return _narrow(a, b, hit, idx)


def _narrow(a: int, b: int, hit: Any, idx: list[int]) -> tuple[int, int]:
    """Run ``a..b`` (positions in ``idx``, the active tokens) narrowed while ``hit`` still holds (:func:`_window`)."""
    step = 1 << max(0, (b - a).bit_length() - 1)
    while step:
        if a + step <= b and hit(a + step, b):
            a += step
        else:
            step //= 2
    step = 1 << max(0, (b - a).bit_length() - 1)
    while step:
        if b - step >= a and hit(a, b - step):
            b -= step
        else:
            step //= 2
    return idx[a], idx[b]


def _state_code(raw: str) -> bool:
    """``raw`` is a US state or territory code (``TX``, ``DC``; ``detect.text.is_state``)."""
    from .text import is_state

    return len(raw) == 2 and is_state(raw)


def _core_end(tok: Any) -> int:
    """Where token ``tok`` ends without a possessive ``'s`` (``G's`` -> ``G``)."""
    return tok.end - 2 if len(tok.raw) > 2 and _POSSESSIVE.search(tok.raw) else tok.end


class _Scan:
    """The tokens of one value (``privacy.headers._tokens`` and ``_token_kinds``) and the layout rules over them."""

    def __init__(self, s: str, toks: list[Any], kinds: list[str], names: Any) -> None:
        self.s, self.toks, self.kinds, self.names = s, toks, kinds, names
        self.n = len(toks)
        self.case_info = any(ch.islower() for ch in s) and any(ch.isupper() for ch in s)
        self.given, self.surnames = names.known_names()

    # ------------------------------------------------------------------ token tests
    def sep(self, i: int) -> str:
        """The text between token ``i`` and the next one (or the end)."""
        t = self.toks
        return self.s[t[i].end:t[i + 1].start] if i + 1 < self.n else self.s[t[i].end:]

    def word(self, i: int) -> bool:
        """Token ``i`` is letters (apostrophes inside: ``O'Brien``)."""
        return _LETTERS.fullmatch(self.toks[i].raw) is not None

    def capital(self, i: int) -> bool:
        return self.toks[i].raw[:1].isupper()

    def initial(self, i: int) -> bool:
        """An initial: one capital letter (any letter in a value without lower case), not running into digits."""
        return self.kinds[i] == "initial" and (self.capital(i) or not self.case_info) and not (
            i + 1 < self.n and self.kinds[i + 1] == "digit" and self.sep(i) == "")

    def name_word(self, i: int, upper: bool | None) -> bool:
        """A known name word written like the name beside it (``upper``: that name is in capitals): ``GEO`` beside
        ``Garcia`` is a course code, not a name word."""
        return self.kinds[i] == "name" and self.word(i) and self.capital(i) and self.styled(i, upper)

    def styled(self, i: int, upper: bool | None) -> bool:
        return not self.case_info or upper is None or self.toks[i].raw.isupper() == upper

    def strong(self, i: int) -> bool:
        """A name word the D34 pair check reads (a known name that is neither a common English word nor an everyday
        surname: ``Lisa``, ``Garcia``; not ``Miller``, ``Brown``, ``Mark``)."""
        key = self.toks[i].key
        return self.kinds[i] == "name" and self.names._distinctive(key) and not self.names._everyday(key)

    def ends_name(self, j: int) -> bool:
        """The name ends at token ``j``: the value ends, or a digit or punctuation follows (``Ms Brown``, ``Ms Brown
        7``, ``Ms Brown - Rm 4``), not another word (``MS Hope Squad``, ``MS Key Club``)."""
        return j + 1 >= self.n or self.sep(j).strip() != "" or self.kinds[j + 1] == "digit"

    def ms_surname(self, j: int) -> bool:
        """Token ``j`` reads as a surname after ``Ms`` / ``MS`` without a period, which is also a middle school
        (``MS Band``, ``Ms Science 7``, ``MS HOPE SQUAD``): a surname of the lists that is no header word (a name, or a
        capitalised particle: ``Ms Garcia``, ``Ms Le``), an everyday one only where the name ends (``Ms Brown``; not
        ``MS Hope Squad``, ``MS Key Club``); or a two-letter word off the lists in Title case (``Ms Ng``)."""
        tok, kind = self.toks[j], self.kinds[j]
        if not (self.word(j) and self.capital(j)):
            return False
        if kind == "other":
            return self.case_info and len(tok.key) == 2 and tok.raw[1:].islower() \
                and tok.key not in self.names.name_lists.common_words()  # not 'MS Of Mice and Men'
        if kind not in ("name", "particle") or tok.key not in self.surnames:
            return False
        return tok.key not in self.names._EVERYDAY_SURNAMES or self.ends_name(j)

    def ambiguous_ms(self, i: int) -> bool:
        """Token ``i`` is ``Ms`` / ``MS`` without a period: an honorific or a middle school."""
        return self.kinds[i] == "title" and self.toks[i].key == "ms" and not self.sep(i).startswith(".")

    def middle_school(self, lo: int, hi: int) -> bool:
        """A window of the D34 check that is ``Ms`` / ``MS`` without a period and a word that is no surname there
        (:meth:`ms_surname`): a middle school's course (``MS CHINA STUDIES``, ``MS HOPE SQUAD``), no person."""
        j = lo + 1
        return self.ambiguous_ms(lo) and j <= hi and not self.ms_surname(j) and not (
            j + 1 < self.n and self.pair(j, j + 1, self.sep(j)))  # 'Ms Lisa Brown' names a teacher

    # ------------------------------------------------------------------ D36's own layouts
    def layout(self, active: list[bool]) -> tuple[int, int] | None:
        """The first person-name layout of D36's own among the active tokens (module docstring), as a window."""
        names = self.names
        toks, kinds = self.toks, self.kinds
        for i in range(self.n - 1):
            j = i + 1
            if not (active[i] and active[j]):
                continue
            between = self.sep(i)
            if kinds[i] == "title" and between.strip() in ("", "."):
                if (self.case_info and toks[i].raw.islower() and not between.startswith(".")) or not self.word(j):
                    continue
                if toks[i].key in names._INITIAL_TITLES:
                    # an honorific: 'Ms. Le', 'Mrs. Van Der Berg'; 'Ms' / 'MS' without a period may be a middle
                    # school ('MS Band', 'LINCOLN MS SCIENCE'): only before a surname then; another all-caps honorific
                    # without a period only before a known name
                    caps = toks[i].raw.isupper() and len(toks[i].key) > 1 and not between.startswith(".")
                    if self.ambiguous_ms(i):
                        ok = self.ms_surname(j)
                    elif caps:
                        ok = kinds[j] == "name" and self.capital(j)
                    else:
                        ok = kinds[j] == "particle" or (self.capital(j) and kinds[j] in _SURNAME_KINDS)
                else:
                    # another title of the D34 check ('Dr', 'Coach', 'Prof', 'Counselor' ...): before a distinctive
                    # known name, as there ('Dr. Patel'; not 'Coach Bus', 'Nurse Joy'), so a sentence after the name
                    # cannot exempt it ('with Dr. Patel. Co-taught by ...' is no company: :meth:`exempt`)
                    key = toks[j].key
                    ok = kinds[j] == "name" and self.capital(j) and names._distinctive(key) \
                        and key not in names._EVERYDAY_SURNAMES
                if ok:
                    last = self.after_title(i, active)
                    if last > i and not self.exempt(i, last):
                        return i, last
            elif between.strip() in (",", ";"):
                if self.comma_name(i, j) and not self.exempt(i, j):
                    return i, j
            elif self.pair(i, j, between) and not self.exempt(i, j) and not self.names.namesakes.is_namesake(
                    (toks[i].key, toks[j].key)):
                return i, j
            elif between == " " and i >= 1 and active[i - 1] and self.sep(i - 1) in _HYPHENS and self.word(i - 1) \
                    and self.capital(i - 1) and self.word(i) and kinds[i - 1] in ("name", "other") \
                    and kinds[i] in ("name", "other") and self.word(j) and self.capital(j) \
                    and toks[j].key in names.world_names.ASIAN_SURNAMES and not self.exempt(i - 1, j):
                return i - 1, j  # a hyphenated given name before an East Asian surname: 'Ji-hoon Kim', 'Soo-jin Park'
            elif between == " " and kinds[j] == "particle" and j + 1 < self.n and active[j + 1] \
                    and self.sep(j) == "" and self.capital(j) and self.word(j + 1) and self.capital(j + 1) \
                    and self.strong(i) and toks[i].key in self.given \
                    and toks[i].key not in self.surnames and not self.exempt(i, j + 1):
                return i, j + 1  # a distinctive given name before a surname run into its prefix: 'Fiona McAllister'
        return None

    def comma_name(self, i: int, j: int) -> bool:
        """``Surname, Given`` at tokens ``i``, ``j`` (a comma or semicolon between), the layout D36 names first: the
        comma says which word is the surname, so any name of the lists counts, common ones too (``Miller, Lisa``,
        ``West, Ana``, ``Le, Thanh``, ``Wu, Lin``, ``D'Angelo, Maria``), and so does a word off the lists before a known
        given name (``D'Souza, Priya``, ``Ng, Wei``, ``Vo, Minh``; a two-letter one that is an English word too before
        a distinctive one: ``Do, Lan``). After the comma: a known given name (``Park, Min``), another name of the lists
        beside a distinctive one (``Yoo, Jin``), a word off the lists hyphenated (``Park, Ji-hoon``) or after a
        distinctive surname, or a capital initial (``Miller, L``, ``Jackson, E``, ``Hall, L``: after a surname of the
        lists; after any surname with a period, ``Ng, W.``). Never a course word before a given name (``French, Art``: a
        header word on the lists only before a distinctive given name), a US state, a company or credential (``Allen,
        Texas``, ``Smith, Inc.``), a Roman numeral (``French, I``), and, two or three capitals run together
        (``Abernathy, RJ``), only after a distinctive surname, never a state code or a school (``Austin, TX``,
        ``Lincoln, HS``)."""
        names, toks, kinds = self.names, self.toks, self.kinds
        left, right = toks[i], toks[j]
        if not (self.word(i) and self.capital(i)) or len(left.key) < 2 or kinds[i] in ("title", "initial", "suffix"):
            return False
        common = names.name_lists.common_words()
        header = kinds[i] == "word"
        known = not header and (left.key in self.surnames or kinds[i] == "name")
        off = kinds[i] == "other" and left.key not in self.surnames and (
            left.key not in common or (len(left.key) == 2 and self.strong(j)))
        if not (known or off or (header and left.key in self.surnames)):
            return False
        if j + 1 < self.n and self.sep(j) == "":  # 'P1', 'AB12', 'PhD': a code or an abbreviation, no name
            return False
        if kinds[j] == "initial" and right.raw.isupper():
            dotted = self.sep(j).startswith(".")
            if right.key in ("i", "v", "x") and not dotted:
                return False
            return (known and left.key in self.surnames) or dotted
        if not (self.word(j) and self.capital(j)) or right.key in names.us_states(given_names_too=True) \
                or right.key in names._COMPANY_WORDS or kinds[j] == "title":
            return False
        if kinds[j] == "other" and right.raw.isalpha() and right.raw.isupper() and 2 <= len(right.raw) <= 3:
            # initials run together ('Abernathy, RJ'), in a value written in mixed case
            return self.case_info and kinds[i] == "name" and self.strong(i) and left.key in self.surnames \
                and not left.raw.isupper() and not _state_code(right.raw) \
                and right.key not in names._INITIAL_SCHOOLS
        if right.key in self.given:
            return not header or self.strong(j)
        if header:
            return False
        if kinds[j] == "name":
            return self.strong(i) or self.strong(j)
        hyphenated = j + 1 < self.n and self.sep(j) in _HYPHENS and self.word(j + 1)
        return kinds[j] == "other" and right.key not in common and len(right.key) >= 2 and known \
            and (hyphenated or self.strong(i))

    def pair(self, i: int, j: int, between: str) -> bool:
        """A given name and a surname side by side (``Lisa Miller``, ``David Young``, ``MILLER LISA``; a user name
        ``lisa.miller``): the known first + last pair D36 names, also when one of the two is a common word the D34
        check does not pair (``Miller``, ``Young``): one of them a distinctive name (:meth:`strong`), the other on the
        other list, either order; in that order, a distinctive given name that is no surname and a word off the lists
        that is no English word, plural or word ending (``Priya D'Souza``; not ``Peer Helpers``, ``Peer Leadership``);
        or, in that order, a common US given name that is no common surname and one of the 1,000 most common US
        surnames (``Mark Brown``; not ``Smith Hall``). Both written as names (capitalised alike in a value written in
        mixed case; in lower case only run together by ``.`` or ``_``)."""
        names, toks, kinds = self.names, self.toks, self.kinds
        skip = ("title", "initial", "suffix", "particle", "program", "digit")  # 'Harriet Tubman', 'John Deere'
        if not (self.word(i) and self.word(j)) or kinds[i] in skip or kinds[j] in skip or len(toks[i].key) < 2 \
                or len(toks[j].key) < 2:
            return False
        if between == " ":
            if not (self.capital(i) and self.capital(j)) or (self.case_info and (
                    toks[i].raw.isupper() != toks[j].raw.isupper())):
                return False
        elif not (between in (".", "_") and toks[i].raw.islower() and toks[j].raw.islower()):
            return False
        a, b = toks[i].key, toks[j].key
        states = names.us_states()
        if a in states or b in states:
            return False
        popular_first, popular_last = names.popular_names()

        def given(x: int) -> bool:  # a given name more than a surname ('Lisa', 'John'; not 'Smith' of 'Smith Hall')
            k = toks[x].key
            return k in self.given and kinds[x] != "word" and (k not in self.surnames or k in popular_first)

        if b in self.surnames and given(i) and (self.strong(i) or self.strong(j)):
            return True
        if a in self.surnames and given(j) and self.strong(j):  # 'MILLER LISA'
            return True
        if a in self.given and a not in self.surnames and self.strong(i) and kinds[j] == "other" \
                and between == " " and len(b) >= 3 and b not in names.name_lists.common_words() \
                and not b.endswith(names._EVERYDAY_ENDINGS) and not b.endswith("s") \
                and (not self.case_info or not toks[j].raw.isupper()):  # not 'Peer Helpers', 'Peer Leadership'
            return True  # a distinctive given name that is no surname before a surname off the lists ("Priya D'Souza")
        return a in popular_first and a not in popular_last and popular_last.get(b, 1000) < 1000

    def exempt(self, lo: int, hi: int) -> bool:
        """D34's exemptions around a layout of D36's own, ``lo..hi`` (``privacy.headers._place_after`` and
        ``_place_before``, the checks ``person_name_hit`` makes): a school name form or company word right after it
        (``Mrs. Quenby's Academy``, ``Mr. Patel Brothers``) or a place word right before it, each only with nothing
        but spaces between (in a longer text a period or comma ends the name: ``with Mrs. Abernathy. Co-taught by``
        names a teacher, not a company)."""
        names = self.names
        after = hi + 1 < self.n and self.sep(hi).strip() == "" and names._place_after(
            self.toks, self.kinds, hi, self.case_info)
        before = lo > 0 and self.sep(lo - 1).strip() == "" and names._place_before(self.toks, self.kinds, lo)
        return after or before

    # ------------------------------------------------------------------ the whole name
    def after_title(self, t: int, active: list[bool]) -> int:
        """The last token of the name after title ``t`` (``t`` when none follows): initials and particles
        (``Dr. J. Lee``, ``Ms. de la Cruz``), the surname (a lone particle is one: ``Ms. Le``), its hyphenated parts
        (``Garcia-Lopez``, ``Soo-jin``), more known name words (``Ms. Ana Maria Garcia``), a word off the lists after a
        given name (``Ms. Kim Soo-jin``) and suffixes; not a course word after the surname (``Ms. Garcia Law 1``)."""
        kinds, n = self.kinds, self.n
        loose = not self.ambiguous_ms(t)  # after 'MS' without a period no course word: 'MS KIM ROBOTICS'
        k, last = t + 1, t
        while k < n and active[k] and self.sep(k - 1).strip() in ("", "."):
            if self.initial(k):
                last = k
                k += 1
                if not self.sep(last).startswith(".") or _POSSESSIVE.search(self.toks[last].raw):
                    return last  # a bare initial ends the name: 'Mrs. G', 'Mr. G's Class'
            elif kinds[k] == "particle" and self.word(k):
                last = k
                k += 1
            else:
                break
        if not (k < n and active[k] and self.sep(k - 1).strip() in ("", ".") and self.word(k) and self.capital(k)
                and kinds[k] in _SURNAME_KINDS):
            return last
        last = k
        upper = self.toks[k].raw.isupper()
        k += 1
        while k < n and active[k]:
            sp = self.sep(k - 1)
            if sp in _HYPHENS and self.word(k):
                pass
            elif sp == " " and (self.name_word(k, upper) or (
                    loose and kinds[k] == "other" and self.word(k) and self.capital(k) and self.styled(k, upper)
                    and self.toks[k - 1].key in self.given)):
                pass
            elif kinds[k] == "suffix" and sp.strip(" ,") == "":
                pass
            else:
                break
            last = k
            k += 1
        return last

    def extent(self, lo: int, hi: int, active: list[bool]) -> tuple[int, int]:
        """Window ``lo..hi`` widened over the whole name (module docstring): a title before it with the name after
        that title (:meth:`after_title`), the name words, particles, initials and hyphenated parts beside it, the
        surname before a comma, and the given names and initials after it."""
        kinds, n = self.kinds, self.n
        t = lo if kinds[lo] == "title" else (
            lo - 1 if lo > 0 and active[lo - 1] and kinds[lo - 1] == "title" and self.sep(lo - 1).strip() in ("", ".")
            else None)
        if t is not None:
            hi = max(hi, self.after_title(t, active))
            if hi == t:
                return t, hi
        first = lo if t is None else t + 1
        ref = [k for k in range(first, hi + 1) if kinds[k] not in ("initial", "title", "digit", "particle", "suffix")
               and self.word(k)]
        upper = all(self.toks[k].raw.isupper() for k in ref) if ref else None
        lo = first
        comma = any(any(c in self.sep(k) for c in ",;") for k in range(lo, hi))
        changed = True
        while changed:
            changed = False
            k = lo - 1
            if t is None and k >= 0 and active[k]:
                sp = self.sep(k)
                if (sp in _NAME_JOINS and (self.name_word(k, upper) or (kinds[k] == "particle" and self.word(k)))) or (
                        kinds[k] == "other" and sp in _HYPHENS and self.word(k)) or (
                        self.initial(k) and sp.startswith(".") and sp.strip(". ") == ""):
                    lo, changed = k, True
                elif not comma and sp.strip() in (",", ";") and kinds[k] in ("name", "other", "program") \
                        and self.word(k) and self.capital(k) and self.styled(k, upper) and len(self.toks[k].key) > 1:
                    # the part found is the given name ('Thompson, Robert J': D34 reads 'Robert J'): its surname
                    lo, comma, changed = k, True, True
            k = hi + 1
            if k < n and active[k]:
                sp = self.sep(hi)
                if (sp in _NAME_JOINS and self.name_word(k, upper)) or (
                        kinds[k] == "other" and sp in _HYPHENS and self.word(k)) or (
                        self.initial(k) and sp in ("", " ")) or (kinds[k] == "suffix" and sp.strip(" ,") == ""):
                    hi, changed = k, True
                elif comma and sp == " " and kinds[k] in ("name", "other") and self.word(k) and self.capital(k) \
                        and self.styled(k, upper) and all(kinds[x] != "title" for x in range(lo, k)):
                    hi, changed = k, True  # a given name after the comma off the lists ('Garcia, Ana Xochitl')
                elif kinds[k] == "particle" and sp in _NAME_JOINS and k + 1 < n and active[k + 1] \
                        and self.sep(k) in _NAME_JOINS and self.name_word(k + 1, upper):
                    hi, changed = k + 1, True
        return (lo if t is None else t), hi

    # ------------------------------------------------------------------ what is replaced
    def roles(self, lo: int, hi: int) -> list[tuple[int, int, str]]:
        """(start, end, role) of each token of name part ``lo..hi`` to replace: titles, suffixes, digits and lower-case
        particles stay; before a comma (or semicolon) the words are the surname, after it the given names; otherwise
        the last word is the surname and the words before it given names, the last word before a separator too (``Lisa
        Miller/John Taylor``; a capitalised particle is part of the surname: ``Van Der Berg``); capitals run together
        after a comma are initials (``Abernathy, RJ``)."""
        kinds, toks = self.kinds, self.toks
        comma = next((k for k in range(lo, hi) if any(c in self.sep(k) for c in ",;")), None)
        words = [k for k in range(lo, hi + 1) if kinds[k] not in _KEPT_KINDS and kinds[k] != "initial"
                 and not (kinds[k] == "particle" and toks[k].raw.islower())]
        ends = {w for x, w in enumerate(words) if x + 1 == len(words) or any(
            _SEPARATOR.search(self.sep(k)) for k in range(w, words[x + 1]))}
        out: list[tuple[int, int, str]] = []
        for k in range(lo, hi + 1):
            tok = toks[k]
            end = _core_end(tok)  # 'G's Class': the 's stays
            if kinds[k] == "initial":
                out.append((tok.start, end, INITIAL))
            elif k in words:
                if comma is not None and k > comma and self.case_info and kinds[k] != "name" \
                        and tok.raw.isalpha() and tok.raw.isupper() and 2 <= len(tok.raw) <= 3:
                    out.extend((x, x + 1, INITIAL) for x in range(tok.start, end))
                    continue
                if comma is not None:
                    role = LAST if k <= comma else FIRST
                else:
                    role = LAST if k in ends or kinds[k] == "particle" else FIRST
                out.append((tok.start, end, role))
        return out


def name_orders(part: str) -> list[str]:
    """The surname and each given name of name part ``part`` side by side in both orders (``Thompson Robert``,
    ``Robert Thompson`` for ``Thompson, Robert J``), as another text may write the same person: the leak scanner
    indexes them with the part (``detect.text.org_name_phrases``)."""
    got = find(part)
    if got is None or not got.tokens:
        return []
    last = [part[a:b] for a, b, role in got.tokens if role == LAST]
    given = [part[a:b] for a, b, role in got.tokens if role == FIRST]
    return list(dict.fromkeys(x for s in last for g in given for x in (f"{s} {g}", f"{g} {s}")))


__all__ = ["FIRST", "INITIAL", "LAST", "MAX_NAMES", "OrgName", "align", "find", "name_orders", "place"]
