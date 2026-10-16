"""Input discovery, file classification and output paths (FOUNDATION).

* INPUT is a file or a directory; a directory is walked recursively in sorted
  order (D18). ``--include GLOB`` / ``--exclude-path GLOB`` filter the walk: a
  pattern containing ``/`` is matched against the posix path relative to the
  input root, otherwise against the file name or the relative path.
* Each file is classified by extension plus a magic-byte sniff into a registry
  format key. Unsupported files are listed as skipped and never copied.
  ``.xls`` is OLE2 (real .xls, written as .xlsx), a zip (an .xlsx in disguise),
  HTML (``html_xls``) or text (read as delimited text, mirrored with its name).
  A content/extension mismatch (e.g. a zip named .csv) fails the file. An
  Office owner or lock file of an open document (``~$Roster.xlsx``,
  ``.~lock.Roster.xlsx#``) and a macOS AppleDouble file (``._Roster.csv``, also
  under ``__MACOSX/``: left behind when a zip made on a Mac is extracted) are
  skipped, never read.
* Output path = same relative path and file name, except ``.xls`` (OLE2) and
  ``.xlsm`` -> ``.xlsx`` and SAS -> ``.parquet`` (D17).
* Directory safety (D20, D21): the input folder (for a file INPUT: its folder,
  and the folder its real file is in), the out dir and the report dir must be
  separate trees: none may be, contain or lie inside another, except that the
  report dir may lie inside the input (the walk skips it; the input may not lie
  inside the report dir, whose files could replace input files). Every test
  compares REAL locations (:mod:`datanonymous.paths`), so a symlink, junction,
  short name or other alias cannot smuggle the out dir into the input or the
  report into the shareable out dir. An existing out or report dir may not hold
  a link that leads out of it, and a non-empty one needs ``--force`` (refused
  in sharing mode: there both must be new or empty).
* The walk follows linked folders (symlinks and junctions alike) but never one
  that leads back to a folder above it (a loop: listed as skipped), and it
  refuses (exit 2) any folder or file whose real location is inside the out
  dir or the report dir, because writing the mirror would replace it.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import paths
from .config import Settings
from .errors import UsageError
from .io import compress, registry
from .model import DiscoveredFile, Message, SkippedFile
from .paths import is_within

MAGIC_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
MAGIC_ZIP = b"PK\x03\x04"
MAGIC_PARQUET = b"PAR1"
MAGIC_GZIP = b"\x1f\x8b"
MAGIC_BZ2 = b"BZh"
MAGIC_XZ = b"\xfd7zXZ\x00"
MAGIC_ARROW = b"ARROW1"
MAGIC_FEATHER1 = b"FEA1"


@dataclass
class Discovery:
    root: Path  # input root directory
    single_file: bool
    files: list[DiscoveredFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


def _non_empty(d: Path) -> bool:
    return d.exists() and d.is_dir() and any(d.iterdir())


def default_report_dir(out_dir: Path) -> Path:
    out = Path(os.path.abspath(str(out_dir)))
    return out.parent / (out.name + "-report")


def _named(flag: str, d: Path) -> str:
    """``--out-dir X`` plus where X really lands when a link or alias moves it."""
    real = paths.shown_real(d)
    if os.path.normcase(real) == os.path.normcase(os.path.abspath(str(d))):
        return f"{flag} {d}"
    return f"{flag} {d} (which resolves to {real})"


def input_folders(input_path: Path) -> list[tuple[str, Path]]:
    """``(label, folder)`` the input lives in: the INPUT folder, or a file's folder and its real file's folder."""
    if input_path.is_dir():
        return [("the input folder", input_path)]
    folders = [("the input folder", input_path.parent)]
    real_parent = Path(paths.shown_real(input_path)).parent
    if paths.real_path(real_parent) != paths.real_path(input_path.parent):
        folders.append(("the folder of the real input file", real_parent))
    return folders


def check_dirs(input_path: Path, out_dir: Path, report_dir: Path, force: bool, sharing: bool = False) -> None:
    """Directory-safety rules (raise UsageError, exit 2) on real locations: see the module docstring.

    ``sharing``: a non-empty folder cannot be forced (sharing mode refuses ``--force``), so say what to do instead.
    """
    dirs = (("--out-dir", out_dir, "out dir"), ("--report-dir", report_dir, "report dir"))
    for flag, d, _ in dirs:
        if d.exists() and not d.is_dir():
            raise UsageError(f"{flag} {d} exists and is not a directory")
    for label, root in input_folders(input_path):
        for flag, d, what in dirs:
            # the local report may sit inside the input (the walk skips it); the input inside the report dir may
            # not: report files written there could replace real input files
            if is_within(d, root) and flag != "--report-dir":
                raise UsageError(f"{_named(flag, d)} is inside {label} {root}; choose a folder outside it")
            if is_within(root, d):
                raise UsageError(f"the input must not be inside the {what}: {label} {root} is inside "
                                 f"{_named(flag, d)}; choose a folder outside the input")
    if is_within(report_dir, out_dir):
        raise UsageError(f"{_named('--report-dir', report_dir)} must not be the out dir or inside it "
                         "(the out dir holds only synthetic files)")
    if is_within(out_dir, report_dir):
        raise UsageError(f"{_named('--out-dir', out_dir)} must not be inside the report dir: report files "
                         "would land in the shareable out dir; choose two separate folders")
    for flag, d, _ in dirs:
        if _non_empty(d) and not force:
            if sharing:
                raise UsageError(f"{flag} {d} already contains files; sharing mode needs a new or empty folder "
                                 "(clear it or choose another)")
            raise UsageError(f"{flag} {d} is not empty; pass --force to write into it")
    for flag, d, _ in dirs:
        for rel, target in paths.escaping_links(d)[:1]:
            raise UsageError(f"{flag} {d} holds a link {rel} that leads out of it (to {target}); writing through "
                             "it would land outside the folder; remove the link or choose another folder")


class _Walk:
    """Real-location bookkeeping for the input walk (loops, links into the out/report dirs)."""

    def __init__(self, root: Path, avoid: tuple[tuple[str, Path], ...]) -> None:
        self.root = root
        self.avoid = [(flag, d, paths.real_path(d)) for flag, d in avoid]
        #: the report dir, which may sit inside the input: that folder is not walked (never read as input)
        self.skip = [paths.real_path(d) for flag, d in avoid if flag == "--report-dir"]
        self.chain: dict[str, tuple[str, ...]] = {str(root): (paths.real_path(root),)}

    def skipped(self, real: str) -> bool:
        """The walked folder at real location ``real`` is the report dir or inside it: not part of the input."""
        return any(paths.contains(d_real, real) or paths.same_folder_on_path(real, d_real) for d_real in self.skip)

    def guard(self, p: Path, real: str, linked: bool = True) -> None:
        """Refuse a walked folder or file whose real location is inside the out dir or the report dir.

        A linked entry is compared on its whole real path (and by file identity along it); a plain subfolder
        of an already checked folder only needs its own identity compared (a bind mount, say).
        """
        for flag, d, d_real in self.avoid:
            hit = paths.contains(d_real, real)
            if not hit:
                hit = (paths.same_folder_on_path(real, d_real) if linked
                       else paths.same_folder(str(p), d_real))
            if hit:
                rel = p.relative_to(self.root).as_posix() if p != self.root else "."
                raise UsageError(f"the input entry {rel} is (or links to) {paths.shown_real(p)}, which is inside "
                                 f"{_named(flag, d)}; the synthetic mirror would replace it; remove the link or "
                                 "choose a folder outside the input")

    def prune(self, dirpath: str, dirnames: list[str], skipped: list[SkippedFile]) -> None:
        """Keep the subfolders to walk: a linked folder that leads back above itself is a loop (skipped)."""
        chain = self.chain.get(dirpath) or (paths.real_path(dirpath),)
        keep = []
        for name in sorted(dirnames):
            p = os.path.join(dirpath, name)
            linked = paths.is_link(p)
            real = paths.real_path(p) if linked else os.path.join(chain[-1], os.path.normcase(name))
            if self.skip and self.skipped(real if linked else paths.real_path(p)):
                continue
            if linked and any(paths.contains(real, a) for a in chain):
                skipped.append(SkippedFile(Path(p).relative_to(self.root).as_posix() + "/",
                                           "linked folder leads back to a folder above it (a loop); not followed"))
                continue
            if self.avoid:
                self.guard(Path(p), real, linked)
            self.chain[p] = chain + (real,)
            keep.append(name)
        dirnames[:] = keep

    def check_file(self, p: Path) -> None:
        """A linked file whose real location is inside the out or report dir would be replaced: refuse it."""
        if self.avoid and paths.is_link(p):
            self.guard(p, paths.real_path(p))


def _glob_match(rel: str, pattern: str) -> bool:
    if "/" in pattern:
        return fnmatch.fnmatch(rel, pattern)
    return fnmatch.fnmatch(PurePosixPath(rel).name, pattern) or fnmatch.fnmatch(rel, pattern)


def _head(path: Path, n: int = 512) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _binary_magic(head: bytes) -> str | None:
    for magic, name in ((MAGIC_OLE2, "ole2"), (MAGIC_ZIP, "zip"), (MAGIC_PARQUET, "parquet"), (MAGIC_GZIP, "gzip"),
                        (MAGIC_BZ2, "bz2"), (MAGIC_XZ, "xz"), (MAGIC_ARROW, "arrow"), (MAGIC_FEATHER1, "feather")):
        if head.startswith(magic):
            return name
    return None


def _looks_html(head: bytes) -> bool:
    h = head.lstrip(b"\xef\xbb\xbf").lstrip().lower()
    return h.startswith((b"<html", b"<!doctype", b"<table", b"<?xml", b"<meta", b"<head", b"<body", b"<"))


#: An Office owner file (``~$Roster.xlsx``) or a LibreOffice lock file (``.~lock.Roster.xlsx#``): a temporary
#: file of a document open in an editor, never a table.
_LOCK_FILE_RE = re.compile(r"^(?:~\$.+|\.~lock\..+#)$")
#: A macOS AppleDouble file (``._Roster.csv``: the resource fork and Finder data of ``Roster.csv``, written next to
#: it on non-Mac file systems and in zips made on a Mac): never a table.
_APPLEDOUBLE_RE = re.compile(r"^\._.+$")


def classify(path: Path, rel: str) -> tuple[DiscoveredFile | None, SkippedFile | None]:
    name = PurePosixPath(rel).name
    if _LOCK_FILE_RE.match(name):
        return None, SkippedFile(rel, "an Office owner or lock file of an open document (temporary, not a table)")
    if _APPLEDOUBLE_RE.match(name):
        return None, SkippedFile(rel, "a macOS AppleDouble file (resource fork and Finder data, not a table)")
    suffixes = [s.lower() for s in PurePosixPath(name).suffixes]
    if not suffixes:
        return None, SkippedFile(rel, "not a supported tabular file type (no extension)")
    ext = suffixes[-1]
    compression = None
    if ext in registry.COMPRESSION_SUFFIXES and len(suffixes) >= 2:
        inner = suffixes[-2]
        fmt = registry.EXTENSION_FORMATS.get(inner)
        if fmt in compress.COMPRESSIBLE_FORMATS:
            compression = registry.COMPRESSION_SUFFIXES[ext]
            return DiscoveredFile(path=path, rel_path=rel, format=fmt, out_rel_path=rel, compression=compression), None
        return None, SkippedFile(rel, f"compressed {inner or 'file'} is not a supported tabular file")
    if ext in registry.UNSUPPORTED_ADVICE:
        return None, SkippedFile(rel, "unsupported tabular format: " + registry.UNSUPPORTED_ADVICE[ext])
    fmt = registry.EXTENSION_FORMATS.get(ext)
    if fmt is None:
        if not registry.known_extension(ext):  # 'Ana.Garcia': the suffix is part of the name (D34)
            return None, SkippedFile(rel, "not a supported tabular file type (no known extension)")
        return None, SkippedFile(rel, f"not a supported tabular file type ({ext})")
    out_rel = rel
    if ext in registry.OUTPUT_SUFFIX:
        out_rel = str(PurePosixPath(rel).with_suffix(registry.OUTPUT_SUFFIX[ext]))
    head = _head(path)
    magic = _binary_magic(head)
    df = DiscoveredFile(path=path, rel_path=rel, format=fmt, out_rel_path=out_rel)
    if fmt == "xls":
        if magic == "ole2":
            df.out_rel_path = str(PurePosixPath(rel).with_suffix(".xlsx"))
            df.notes.append(Message("warning", "xls_as_xlsx", ".xls is written as .xlsx (no .xls writer, D17)"))
        elif magic == "zip":
            df.format = "xlsx"
            df.notes.append(Message("info", "disguised_xls", ".xls file is really an .xlsx workbook; name kept"))
        elif _looks_html(head):
            df.format = "html_xls"
            df.notes.append(Message("info", "disguised_xls", ".xls file is really an HTML table; mirrored as such"))
        elif magic is None:
            df.format = "csv"
            df.notes.append(Message("info", "disguised_xls", ".xls file is really delimited text; mirrored as such"))
        else:
            df.preflight_error = (f".xls file content is {magic}, not a workbook", "rename the file to its real type")
        return df, None
    expected = {"csv": None, "xlsx": "zip", "parquet": "parquet", "feather": ("arrow", "feather")}
    if fmt == "csv" and magic is not None:
        df.preflight_error = (f"file content looks like {magic} data, not delimited text",
                              "rename the file to its real extension")
    elif fmt in ("xlsx", "parquet", "feather"):
        want = expected[fmt]
        ok = magic in want if isinstance(want, tuple) else magic == want
        if not ok and head:
            df.preflight_error = (f"file content does not match its {ext} extension",
                                  "rename the file to its real extension")
    return df, None


def discover(input_path: Path, settings: Settings, avoid: tuple[tuple[str, Path], ...] = ()) -> Discovery:
    """Walk ``input_path`` and classify every file (no writes).

    ``avoid`` = ``(flag, folder)`` pairs (the out dir and report dir of a run): an input folder or file whose
    real location is inside one of them is a usage error (exit 2).
    """
    if not input_path.exists():
        raise UsageError(f"input not found: {input_path}")
    if input_path.is_file():
        root = input_path.parent
        disc = Discovery(root=root, single_file=True)
        walk = _Walk(root, avoid)
        if avoid:
            walk.guard(input_path, paths.real_path(input_path))
        candidates = [(input_path, input_path.name)]
    else:
        root = input_path
        disc = Discovery(root=root, single_file=False)
        walk = _Walk(root, avoid)
        if avoid:
            walk.guard(root, paths.real_path(root))
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            walk.prune(dirpath, dirnames, disc.skipped)
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                walk.check_file(p)
                rel = p.relative_to(root).as_posix()
                candidates.append((p, rel))
    for path, rel in candidates:
        if settings.include and not any(_glob_match(rel, g) for g in settings.include):
            disc.skipped.append(SkippedFile(rel, "not matched by --include"))
            continue
        if any(_glob_match(rel, g) for g in settings.exclude_path):
            disc.skipped.append(SkippedFile(rel, "excluded by --exclude-path"))
            continue
        f, s = classify(path, rel)
        if f is not None:
            disc.files.append(f)
        else:
            disc.skipped.append(s)  # type: ignore[arg-type]
    # output name collisions (e.g. a.xls and a.xlsx both -> a.xlsx)
    seen: dict[str, DiscoveredFile] = {}
    for f in sorted(disc.files, key=lambda f: (f.out_rel_path != f.rel_path, f.rel_path)):
        key = f.out_rel_path.casefold()
        if key in seen:
            f.preflight_error = (f"output name {f.out_rel_path} collides with {seen[key].rel_path}",
                                 "rename one of the two input files")
        else:
            seen[key] = f
    if not disc.files:
        disc.messages.append(Message("warning", "no_tabular_files", "no supported tabular files found in the input"))
    return disc
