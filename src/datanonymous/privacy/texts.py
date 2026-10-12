"""The run's real texts that no generated value may equal (R7-01, R7-02; AREA G; PRIVATE, in memory only).

D35 writes a real text value verbatim only when it is provably safe. Every other real text value of the run is
replaced: a unit the allowlist pseudonymised (any column), every value of a person-name, e-mail, address, city,
IP/URL, free-text or high-cardinality column, the text a date column replaces with generated dates, a native label
no row writes, and every value of an ``--exclude``d or ``--null``ed column (the column is not written, but its
values are real people's data all the same). Each generator always avoided the real values of its OWN column; this
registry makes that rule run-wide, as :mod:`privacy.digits` does for phone numbers and SSNs and the ID namespaces do
for IDs (D9, R6-02):

* :class:`RealTexts` holds the comparison keys (:func:`key`: the leak scanner's ``norm_text``) of those values, per
  source column (names only, for messages), and the keys of the real text values the run DOES write verbatim
  somewhere (kept levels, multi-select items and tokens, kept tokens of date columns, states, NA tokens): a value
  kept somewhere is public in the output anyway (the D35 excuse list), so it is never a hit. Only text counts
  (:func:`is_text`, a letter in the value): a number, a date or a digit code names no one by itself; the run's real
  phone numbers and SSNs are guarded by :mod:`privacy.digits` and its real IDs by the ID namespaces. It also holds
  the local parts of the real e-mail addresses among those values and among the run's real IDs: a synthetic value
  that is an e-mail address never has one (a real local part is a real username; the leak scan compares local
  parts, R5-02).
* :func:`guard` makes a registry active while a table (or the run's pseudonym map) is generated. Every acceptance
  test of a synthetic text value refuses a candidate the active registry holds (:func:`holds_real`): the
  ``generate.text.Drawer`` behind fake PII, filler text, invented labels, masks and pseudonyms, the name redraws of
  ``generate.pii`` and the template rebuilds of ``generate.structure`` (which read the table's registry directly).
  ``generate.pii.plan_names`` adds the registry's short texts to the run's full-name index (R1-04), so a synthetic
  full name, or the first and last name on one row, never spells one either.
* the leak scanner (``privacy.leaks``) checks every text column of the output against it, not only against the
  column's own real values: before writing (``scan_table``) and on the re-read files (``DataScanner``); and every
  ID column's values against its e-mail addresses and usernames (:meth:`RealTexts.user_hits`, R8-03).

Source text outside the table counts the same (R8-04, kind :data:`LAYOUT`): the title, notes, footer and side-note
text of a workbook or an HTML-disguised .xls (:func:`layout_texts`; written as placeholders, so real text the run
does not write; with ``--keep-titles`` written verbatim, so kept), and what no output holds at all (HTML comments,
script text and attributes the writer drops, free-form Arrow metadata, Stata variable and data labels:
``privacy.leaks.dropped_texts``). The ID registry is built from it too: no synthetic ID is, or has the local part
or username of, a real e-mail address held anywhere in the run (``pipeline.plan_ids``, :func:`email_addresses`), nor
equals any of its texts (R9-03: ``generate.ids.RealUsers.text_taken``; the leak scan checks every ID column against
it, ``privacy.leaks.id_texts``).

``privacy.leaks.plan_real_texts`` builds it in run planning and hands it to every table
(``TablePrivate.extra[RUN_KEY]``). The values of an excluded or nulled column are kept from pass 1 for it
(``profile.table``: ``TablePrivate.extra[DROPPED_KEY]``). Never serialised. The guard is active while a table is
generated and while its file is written (the placeholder text of titles avoids it too).
"""

from __future__ import annotations

import contextlib
import functools
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AbstractSet, Any, Iterable, Iterator

from ..detect.ids import MIN_USERNAME_LETTERS, is_email_id, username_stem
from ..detect.text import unit_key

#: PRIVATE ``TablePrivate.extra`` key: the run's :class:`RealTexts` (``privacy.leaks.plan_real_texts``).
RUN_KEY = "run_avoid_texts"
#: PRIVATE ``TablePrivate.extra`` key: column index -> the distinct non-missing values (stripped) of a column that
#: ``--exclude`` or ``--null`` keeps out of the output (``profile.table``).
DROPPED_KEY = "_dropped_values"
#: Kinds of a :class:`Source`: a person-name column (its full names are indexed by ``generate.pii.real_name_pairs``
#: already), an excluded or nulled column, any other column, and the source text outside the table of a file
#: (titles, notes, footers, side notes, dropped metadata: R8-04).
NAME, DROPPED, TEXT, LAYOUT = "name", "dropped", "text", "layout"
#: The ``column`` of a :data:`LAYOUT` source (a location, never a value).
LAYOUT_WHERE = "(text outside the table: titles, notes, footers, metadata)"

_LETTER = re.compile(r"[^\W\d_]")
#: A whole value that is one e-mail address (a key: case-folded, no inner space).
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
#: E-mail addresses inside a key (their local parts are registered).
EMAIL_IN_TEXT = re.compile(r"[a-z0-9._%+'\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+")
#: Deletes every ASCII character that is no letter (a username is the letters of a local part).
_ASCII_NON_LETTERS = str.maketrans("", "", "".join(chr(i) for i in range(128) if not chr(i).isalpha()))


def key(text: str) -> str:
    """The comparison key of a text value: typographic apostrophes and hyphens folded, case-folded, whitespace
    collapsed (``detect.text.unit_key``, the leak scanner's ``norm_text``)."""
    return unit_key(text)


def is_text(k: str) -> bool:
    """A value with at least one letter: what the registry holds and checks (a number, a date or a digit code
    names no one by itself; real phone numbers, SSNs and IDs have their own run-wide rules)."""
    return _LETTER.search(k) is not None


@dataclass(frozen=True)
class Source:
    """Where registered texts came from (names only, never values)."""

    file: str
    table: str | None
    column: str
    kind: str = TEXT


@dataclass(frozen=True)
class RealTexts:
    """The run's real text values that it does not write verbatim (module docstring). PRIVATE.

    ``keys``: the keys of every registered value (all sources; shared, not copied, when only one source has any);
    ``kept``: the keys of the real text values written verbatim somewhere in the run, which are never a hit;
    ``sources``: (source column, its keys) per column; ``locals``: the local parts (case-folded) of the real e-mail
    addresses among them and of the run's real e-mail IDs: a synthetic value that IS an e-mail address never has one
    (a real local part is a real username; the leak scan compares local parts, R5-02); ``outside``: (file, its
    source texts outside the tables that the run does not write verbatim, as read: a digit-only title is no key but
    its phone number is indexed, R8-04).
    """

    keys: AbstractSet[str] = frozenset()
    kept: AbstractSet[str] = frozenset()
    sources: tuple[tuple[Source, AbstractSet[str]], ...] = ()
    locals: AbstractSet[str] = frozenset()
    outside: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.keys or self.locals)

    def holds_key(self, k: str) -> bool:
        """``k`` (a :func:`key`) is a real text of the run that the run does not write verbatim, or an e-mail address
        with a real local part."""
        if k in self.keys and k not in self.kept and is_text(k):
            return True
        return bool(self.locals) and "@" in k and self._local_hit(k)

    def _local_hit(self, k: str) -> bool:
        return _EMAIL.fullmatch(k) is not None and k.partition("@")[0] in self.locals

    def holds(self, text: object) -> bool:
        """``text`` equals (as a :func:`key`) a real text of the run that the run does not write verbatim, or is an
        e-mail address with a real local part of the run."""
        return isinstance(text, str) and bool(self) and self.holds_key(key(text))

    def hits(self, texts: Iterable[str]) -> set[str]:
        """The keys of ``texts`` that the registry holds (one set intersection; the rare hits are then checked)."""
        if not self:
            return set()
        found = {key(t) for t in texts if isinstance(t, str)}
        out = {k for k in found & self.keys if k not in self.kept and is_text(k)}  # type: ignore[operator]
        if self.locals:
            out |= {k for k in found if "@" in k and self._local_hit(k)}
        return out

    def sources_of(self, k: str) -> list[Source]:
        """The source columns holding key ``k``."""
        return [s for s, ks in self.sources if k in ks]

    @functools.cached_property
    def user_stems(self) -> frozenset[str]:
        """The usernames (``detect.ids.username_stem``: the letters of a local part) of :attr:`locals`."""
        out = set()
        for loc in self.locals:
            letters = loc.translate(_ASCII_NON_LETTERS) if loc.isascii() else "".join(ch for ch in loc if ch.isalpha())
            if len(letters) >= MIN_USERNAME_LETTERS:
                out.add(letters)
        return frozenset(out)

    def user_hits(self, values: Iterable[str], prefix: str = "") -> set[str]:
        """R8-03: the values of an ID column (stripped) that are a real e-mail address of the run the run does not
        write verbatim, or an e-mail address with a real local part, or whose username
        (``detect.ids.username_stem``, read without the column's kept ID ``prefix``) is the username of one. The ID
        registry never draws one (``pipeline.plan_ids`` avoids every e-mail address of the registry), so a hit is a
        copy."""
        if not self.locals:
            return set()
        stems = self.user_stems
        out: set[str] = set()
        for v in values:
            if not isinstance(v, str):
                continue
            if "@" in v:
                s = v.strip()
                if is_email_id(s) and self.holds_key(key(s)):
                    out.add(s)
                    continue
            if stems:
                st = username_stem(v, prefix)
                if st is not None and st in stems:
                    out.add(v.strip())
        return out

    def texts(self, kinds: AbstractSet[str] = frozenset({TEXT, DROPPED, LAYOUT})) -> Iterator[str]:
        """The registered keys of sources of ``kinds`` that are hits (not kept, with a letter); each once."""
        seen: set[str] = set()
        for s, ks in self.sources:
            if s.kind not in kinds:
                continue
            for k in ks:
                if k not in seen and k not in self.kept and is_text(k):
                    seen.add(k)
                    yield k


_ACTIVE: ContextVar[RealTexts | None] = ContextVar("datanonymous_real_texts", default=None)


@contextlib.contextmanager
def guard(texts: RealTexts | None) -> Iterator[None]:
    """Make ``texts`` the run's real texts that the generators avoid (:func:`holds_real`) inside the block."""
    token = _ACTIVE.set(texts if texts else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active() -> RealTexts | None:
    return _ACTIVE.get()


def holds_real(text: str) -> bool:
    """True when ``text`` equals a real text of the run that the run does not write verbatim (:func:`guard`);
    False without a guard."""
    texts = _ACTIVE.get()
    return texts is not None and texts.holds(text)


def email_addresses(texts: RealTexts) -> set[str]:
    """R8-03: every e-mail address held anywhere in the registry (case-folded): whole values and addresses inside
    texts, of every source and of the texts the run writes verbatim too. The ID registry avoids them all
    (``pipeline.plan_ids``): no synthetic ID is one, nor has its local part or username."""
    out: set[str] = set()
    for keys in (texts.keys, texts.kept):
        for k in keys:
            if "@" not in k:
                continue
            if _EMAIL.fullmatch(k):  # a whole address (any characters in its local part)
                out.add(k)
            else:
                out.update(EMAIL_IN_TEXT.findall(k))
    return out


def layout_texts(fr: Any) -> list[tuple[str, str]]:
    """(text, location) of every source text a workbook or an HTML-disguised .xls keeps outside its tables' header
    and data rows and writes back: title and notes rows (every row of a notes sheet), footer rows, side notes, and
    the text of an HTML document outside the header and data rows (script and style text left out). The writers
    replace each with a same-shape placeholder (``io.excel_write.Placeholders``), or copy it verbatim with
    ``--keep-titles`` (D34 screens it then: ``privacy.headers``). PRIVATE: the texts are real data."""
    out: list[tuple[str, str]] = []
    n_tables = len(fr.tables)
    for t_i, tp in enumerate(fr.tables):
        tl = tp.layout or {}
        where = f"sheet {t_i + 1} of {n_tables}" if n_tables > 1 or tp.name else "the table"
        for k, what in (("_title_rows", "a title/notes row"), ("_footer_rows", "a footer row")):
            for item in tl.get(k) or []:
                cells = item[1] if isinstance(item, (tuple, list)) and len(item) == 2 else item
                for c in _cell_texts(cells):
                    out.append((c, f"{what} of {where}"))
        for cells in (tl.get("_side_cells") or {}).values():
            for c in _cell_texts(cells):
                out.append((c, f"a side note of {where}"))
    extra = (fr.layout.extra or {}) if fr.layout is not None else {}
    events = extra.get("_events") or []
    if events:  # HTML-disguised .xls: every text but the header and data rows (and script/style text)
        skip = html_table_rows(fr)
        hidden = 0
        for idx, ev in enumerate(events):
            if not isinstance(ev, (tuple, list)) or len(ev) < 2:
                continue
            if ev[0] == "start" and ev[1] in ("script", "style"):
                hidden += 1
            elif ev[0] == "end" and ev[1] in ("script", "style"):
                hidden = max(0, hidden - 1)
            elif ev[0] == "data" and not hidden and idx not in skip and isinstance(ev[1], str) and ev[1].strip():
                out.append((ev[1].strip(), "a text outside the table's header and data rows"))
    return out


def html_table_rows(fr: Any) -> set[int]:
    """The event indexes of the header and data rows of an HTML-disguised .xls's tables (not notes tables)."""
    skip: set[int] = set()
    for tp in fr.tables:
        tl = tp.layout or {}
        tb = tl.get("_html_table")
        if tb is None or tl.get("notes_only"):
            continue
        h = int(tl.get("header_row") or 1)
        rows = getattr(tb, "rows", []) or []
        for r in range(h - 1, min(int(tl.get("data_end") or h), len(rows))):
            skip.update(range(rows[r].start, rows[r].end))
    return skip


def _cell_texts(cells: Any) -> list[str]:
    out: list[str] = []
    values = cells.values() if isinstance(cells, dict) else cells if isinstance(cells, (list, tuple)) else []
    for c in values:
        v = getattr(c, "v", c)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def local_parts(keys: Iterable[str]) -> set[str]:
    """The local parts of the e-mail addresses inside ``keys`` (:func:`key` forms: already case-folded)."""
    out: set[str] = set()
    for k in keys:
        if "@" not in k:
            continue
        if _EMAIL.fullmatch(k):  # a whole address (any characters in its local part)
            out.add(k.partition("@")[0])
        else:
            out.update(e.partition("@")[0] for e in EMAIL_IN_TEXT.findall(k))
    out.discard("")
    return out


__all__ = ["DROPPED", "DROPPED_KEY", "LAYOUT", "LAYOUT_WHERE", "NAME", "RUN_KEY", "TEXT", "RealTexts", "Source",
           "active", "email_addresses", "guard", "holds_real", "html_table_rows", "is_text", "key", "layout_texts",
           "local_parts"]
