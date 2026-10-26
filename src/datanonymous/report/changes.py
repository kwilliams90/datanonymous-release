"""What changed: the deliberate privacy changes made to each written file, in plain language.

One source for the private report (with counts and the operator-or-automatic distinction), the console summary and
the shareable README (neither counts nor reasons nor source facts). The changes come from three places, so a new one
usually needs no code here:

* the change records protection leaves on each table (``profile.sharing.change_records``);
* the Nulled columns of a table (by the operator or, in sharing mode, automatically);
* every message whose code is registered as a ``change`` in the message catalogue.

These are deliberate changes, not problems: they are never listed under "What needs attention".
"""

from __future__ import annotations

from typing import Iterable

from .. import messages as catalogue
from ..model import FileResult

#: change codes that are reported through the table's Nulled columns instead of as a generic message
_HANDLED = frozenset({"column_nulled_automatically", "column_excluded"})


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _record_text(kind: str, count: int, counts: bool, numeric: bool = False) -> str:
    if kind == "dates":
        return "dates shifted; month patterns kept"
    if kind == "numeric":
        return "numeric extremes removed"
    if kind == "categories" and numeric:
        # a number column's rare values are numbers, not categories
        if not counts:
            return "rare values replaced with common ones"
        return f"{count} {_plural(count, 'rare value', 'rare values')} replaced with common ones"
    what = "rare categories" if kind == "categories" else "rare missing-value spellings"
    if not counts:
        return f"{what} merged into common ones"
    one = what.replace("categories", "category").replace("spellings", "spelling")
    return f"{count} {_plural(count, one, what)} merged into {_plural(count, 'a common one', 'common ones')}"


def file_changes(fr: FileResult, counts: bool = True) -> list[tuple[str | None, str]]:
    """(column or None, plain text) of every deliberate change made to ``fr``, each listed once."""
    from ..generate import sharing_numeric
    from ..profile.sharing import AUTO_NULL_SOURCE, CHANGES_KEY, LINK_NOT_KEPT_CODE, TEMPLATE_REBUILT_CODE

    out: list[tuple[str | None, str]] = []
    for tp in fr.tables:
        numeric = {p.name for p in tp.columns if p.type in sharing_numeric.TYPES}
        for column, kind, count in tp.private.extra.get(CHANGES_KEY) or ():
            out.append((column, _record_text(kind, count, counts, numeric=column in numeric)))
        for p in tp.columns:
            if p.role == "excluded":
                # the shareable README must not name it: the operator left it out, so its header is not for recipients
                out.append((p.name, "column left out by you") if counts else (None, "one or more columns are left out of the file"))
                continue
            if p.role != "null":
                continue
            if not counts:
                text = "column is empty"
            elif p.source == AUTO_NULL_SOURCE:
                text = "nulled automatically (written with every cell empty)"
            else:
                text = "nulled by you (written with every cell empty)"
            out.append((p.name, text))
    seen: set[tuple[str | None, str]] = set()
    links: set[str] = set()
    for m in fr.all_messages():
        entry = catalogue.lookup(m.code)
        if entry is not None and m.code in (LINK_NOT_KEPT_CODE, TEMPLATE_REBUILT_CODE):
            # the private report names the link (each once, under the first column nulled in it, or the column rebuilt
            # from the fictional values); the README only says that one is not kept, or rebuilt: which columns were
            # linked in the source is a source fact
            if counts and m.text not in links:
                links.add(m.text)
                out.append((m.column, m.text))
            elif not counts:
                out.append((None, entry.sentence))
            continue
        if entry is None or entry.kind != catalogue.CHANGE or m.code in _HANDLED or (m.column, m.code) in seen:
            continue
        seen.add((m.column, m.code))
        out.append((m.column, entry.sentence))
    return list(dict.fromkeys(out))


def run_changes(files: Iterable[FileResult], counts: bool = True) -> list[tuple[str, str | None, str]]:
    """(file, column or None, plain text) of every deliberate change made to the files written."""
    out = []
    for fr in files:
        if not fr.written or fr.withheld:
            continue
        out += [(fr.rel_path, column, text) for column, text in file_changes(fr, counts)]
    return out


__all__ = ["file_changes", "run_changes"]
