"""Filesystem path safety: real locations, overlap tests and writes that never follow links (FOUNDATION).

The input, the out dir and the report dir must not overlap (D20, D21). A
spelling of a path says little about where it lands: a symlink, a Windows
junction or volume mount point, an 8.3 short name, a ``\\\\?\\`` prefix, an
admin share (``\\\\host\\C$``), a bind mount or a case-insensitive file system
can each make two different spellings name one folder. So:

* :func:`real_path` resolves every link and short name (``os.path.realpath``,
  which also resolves the existing part of a path that does not exist yet) and
  case-folds where the OS does, and :func:`is_within` compares those real
  paths and, as a fallback, the file identity (device + inode / file index) of
  the parent against every existing folder on the child's real path. Two
  names of one folder are therefore always "within" each other.
* :func:`escaping_links` lists the links (symlinks, junctions) inside an
  existing out or report dir that lead out of it: writing through such a link
  would land outside the folder the user named (for example in the input).
* :func:`replace_bytes` and :func:`remove_entry` never write through an
  existing directory entry: the bytes go to a fresh temporary file that is then
  moved over the destination, so a symlink or hard link left at the
  destination name is replaced, and its target (perhaps a real input file) is
  left untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

#: suffix of the temporary file a destination is written to before it is moved into place
TMP_SUFFIX = ".datanonymous-tmp"


def _strip_extended(p: str) -> str:
    """``\\\\?\\C:\\x`` -> ``C:\\x`` and ``\\\\?\\UNC\\h\\s`` -> ``\\\\h\\s`` (Windows extended-length spellings)."""
    if os.name != "nt":
        return p
    if p[:8].upper() == "\\\\?\\UNC\\":
        return "\\\\" + p[8:]
    if p.startswith("\\\\?\\") and len(p) >= 6 and p[5] == ":":
        return p[4:]
    return p


def shown_real(p: Path | str) -> str:
    """Where ``p`` really lands (links, junctions and short names resolved), as the OS spells it."""
    return _strip_extended(os.path.realpath(os.path.abspath(os.fspath(p))))


def real_path(p: Path | str) -> str:
    """Canonical comparison form of ``p``: absolute, every link resolved, case-folded where the OS folds case."""
    return os.path.normcase(shown_real(p))


def contains(parent_real: str, child_real: str) -> bool:
    """``child_real`` is ``parent_real`` or below it (both from :func:`real_path`)."""
    if child_real == parent_real:
        return True
    try:
        return os.path.commonpath([child_real, parent_real]) == parent_real
    except ValueError:  # different drives / UNC vs drive letter
        return False


def _identity(p: str) -> os.stat_result | None:
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    # a file system without stable file ids (FAT, some network shares) reports 0: no identity test there
    return st if st.st_ino else None


def same_folder(a: str, b: str) -> bool:
    """``a`` and ``b`` are one existing folder (or file) by identity, whatever their spelling."""
    sa, sb = _identity(a), _identity(b)
    return sa is not None and sb is not None and os.path.samestat(sa, sb)


def same_folder_on_path(child_real: str, parent_real: str) -> bool:
    """Some existing folder on ``child_real`` (itself or an ancestor) IS the ``parent_real`` folder.

    Catches the aliases a path string cannot show: bind mounts, admin shares, case-insensitive volumes on an OS
    that does not fold case in :func:`os.path.normcase`.
    """
    target = _identity(parent_real)
    if target is None:
        return False
    cur = child_real
    while True:
        st = _identity(cur)
        if st is not None and os.path.samestat(st, target):
            return True
        up = os.path.dirname(cur)
        if up == cur:
            return False
        cur = up


def is_within(child: Path | str, parent: Path | str) -> bool:
    """``child`` is ``parent`` or inside it, wherever links, junctions or other aliases make either land."""
    c, p = real_path(child), real_path(parent)
    return contains(p, c) or same_folder_on_path(c, p)


def is_link(p: Path | str) -> bool:
    """A symlink, or a Windows junction / volume mount point (``os.path.islink`` misses those)."""
    return os.path.islink(p) or os.path.isjunction(p)


def escaping_links(folder: Path) -> list[tuple[str, str]]:
    """Links inside the existing ``folder`` whose target is outside it: ``[(relative posix path, target)]``.

    Real subfolders are searched; links are never followed. A dangling link counts by where it points.
    """
    if not folder.is_dir():
        return []
    root = real_path(folder)
    found: list[tuple[str, str]] = []
    stack = [str(folder)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                link = e.is_symlink() or e.is_junction()
                is_dir = not link and e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if link:
                if not contains(root, real_path(e.path)):
                    found.append((Path(e.path).relative_to(folder).as_posix(), shown_real(e.path)))
            elif is_dir:
                stack.append(e.path)
    return sorted(found)


def remove_entry(path: Path) -> None:
    """Remove a file, symlink or hard link at ``path`` (the entry only, never its target); folders are kept."""
    if os.path.lexists(path) and (is_link(path) or not os.path.isdir(path)):
        os.unlink(path)  # on Windows this removes a directory symlink or junction itself, not its target


def replace_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` without ever writing through an existing entry.

    The bytes go to ``<dest>.datanonymous-tmp``, created exclusively after any entry left at that name is
    removed, and that file is then moved over ``dest``. A symlink or hard link already at ``dest`` is
    replaced, never followed, so whatever it pointed to stays untouched.
    """
    dest = Path(dest)
    tmp = dest.with_name(dest.name + TMP_SUFFIX)
    remove_entry(tmp)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


__all__ = ["TMP_SUFFIX", "real_path", "shown_real", "contains", "same_folder", "same_folder_on_path",
           "is_within", "is_link", "escaping_links", "remove_entry", "replace_bytes"]
