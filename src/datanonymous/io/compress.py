"""Transparent .gz/.zip/.bz2/.xz compression for text formats (design D17).

AREA F owns this module. The CSV family and JSON lines may be compressed
(:data:`COMPRESSIBLE_FORMATS`); discovery treats ``name.<inner ext>.<gz|zip|bz2|xz>``
as compressed only for these inner formats, and the output keeps the same file
name and container.

* :func:`read_bytes` returns the decompressed payload. A zip archive must hold
  exactly one file member (directory entries are ignored); several members FAIL
  the file with the fix "extract the members". A corrupt or encrypted container
  is a :class:`~datanonymous.errors.ReadError`.
* :func:`container_layout` describes the source container so the writer can
  mirror it; readers store the result in ``FileLayout.extra`` (the key starts
  with ``_``: the member name is never reported).
* :func:`write_bytes` compresses like the source:

  - gzip: our own single-member writer reproducing the header fields that carry
    no free text (``MTIME``, ``XFL`` -> compression level, ``OS``, and whether a
    file name ``FNAME`` is stored); comment/extra fields are not copied;
  - zip: one member with the source member's date/time, method and attributes
    (archive comment not copied);
  - bz2: the source block size (``BZh1``..``BZh9``) as compression level;
  - xz: the source integrity check (none/CRC32/CRC64/SHA-256), default preset.

  The **stored names are free text** (``Private <patient name> export.csv``) and
  are never copied (R1-01): a stored gzip ``FNAME`` and the zip member are named
  after the output file (``students.csv.zip`` -> ``students.csv``), whose name is
  the source's file name anyway (D20). :func:`container_messages` warns
  (``container_name_replaced``) when the source's stored name differed, and
  :func:`container_diff` checks that the output holds the output's own name.

  Copying ``MTIME``/date-time from the source keeps the output byte-identical
  for the same seed and input (D26).
"""

from __future__ import annotations

import bz2
import lzma
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

from ..errors import ReadError, WriteError
from ..model import FileLayout, Message

#: Inner format keys that may be compressed (discovery uses this set).
COMPRESSIBLE_FORMATS: frozenset[str] = frozenset({"csv", "jsonl"})

#: ``FileLayout.extra`` key holding :func:`container_layout` (private: names a file member).
LAYOUT_KEY = "_container"

_TMP_SUFFIX = ".datanonymous-tmp"  # pipeline.TMP_SUFFIX (not imported: io never imports the pipeline)
_SUFFIX = {"gz": ".gz", "zip": ".zip", "bz2": ".bz2", "xz": ".xz"}
_XZ_CHECKS = {0: lzma.CHECK_NONE, 1: lzma.CHECK_CRC32, 4: lzma.CHECK_CRC64, 10: lzma.CHECK_SHA256}
_GZ_FLAGS = {"FTEXT": 1, "FHCRC": 2, "FEXTRA": 4, "FNAME": 8, "FCOMMENT": 16}


def _bad(compression: str, what: str = "") -> ReadError:
    return ReadError(f"the file is not valid {compression} data" + (f" ({what})" if what else ""),
                     fix="check that the extension matches the content, or decompress the file and re-run")


# --------------------------------------------------------------------------- reading
def read_bytes(path: Path, compression: str) -> bytes:
    """Return the decompressed bytes of ``path``."""
    path = Path(path)
    if compression == "zip":
        return _read_zip(path)
    raw = path.read_bytes()
    try:
        if compression == "gz":
            if not raw.startswith(b"\x1f\x8b"):
                raise _bad("gzip", "no gzip header")
            import gzip

            return gzip.decompress(raw)
        if compression == "bz2":
            return bz2.decompress(raw)
        if compression == "xz":
            return lzma.decompress(raw)
    except ReadError:
        raise
    except (OSError, EOFError, ValueError, lzma.LZMAError, zlib.error) as e:
        raise _bad(compression, type(e).__name__) from None
    raise ReadError(f"unknown compression {compression!r}", fix="decompress the file and re-run")


def _file_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [i for i in zf.infolist() if not i.is_dir()]


def _read_zip(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as zf:
            members = _file_members(zf)
            if len(members) != 1:
                raise ReadError(
                    f"the zip archive holds {len(members)} files; only a single-file zip is supported",
                    fix="extract the archive and run datanonymous on the extracted files (one table per file)")
            info = members[0]
            if info.flag_bits & 0x1:
                raise ReadError("the zip member is encrypted",
                                fix="extract the archive (with its password) and re-run on the extracted file")
            try:
                return zf.read(info)
            except NotImplementedError:
                raise ReadError("the zip member uses an unsupported compression method",
                                fix="extract the archive and re-run on the extracted file") from None
    except ReadError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, zlib.error, lzma.LZMAError, RuntimeError) as e:
        raise _bad("zip", type(e).__name__) from None


def _gzip_header(raw: bytes) -> dict[str, Any]:
    if len(raw) < 10 or raw[:3] != b"\x1f\x8b\x08":
        return {}
    flg = raw[3]
    mtime = struct.unpack_from("<I", raw, 4)[0]
    info: dict[str, Any] = {"mtime": mtime, "xfl": raw[8], "os": raw[9], "fname": None}
    pos = 10
    try:
        if flg & _GZ_FLAGS["FEXTRA"]:
            xlen = struct.unpack_from("<H", raw, pos)[0]
            pos += 2 + xlen
        if flg & _GZ_FLAGS["FNAME"]:
            end = raw.index(b"\x00", pos)
            info["fname"] = raw[pos:end].decode("latin-1")
    except (ValueError, struct.error):
        pass
    return info


def container_layout(path: Path, compression: str | None) -> dict[str, Any]:
    """Describe the source container (what :func:`write_bytes` mirrors); {} when not compressed.

    ``derived_name`` is the member name the writer uses: the file's name without the
    compression suffix (and without the pipeline's temporary suffix).
    """
    if not compression:
        return {}
    path = Path(path)
    derived = _inner_name(path, compression)
    if compression == "gz":  # the gzip header stores latin-1 (as the writer encodes it)
        derived = derived.encode("latin-1", "replace").decode("latin-1")
    info: dict[str, Any] = {"type": compression, "derived_name": derived}
    try:
        if compression == "gz":
            with open(path, "rb") as fh:
                head = fh.read(4096)
            info.update(_gzip_header(head))
        elif compression == "bz2":
            with open(path, "rb") as fh:
                head = fh.read(4)
            if head[:3] == b"BZh" and head[3:4].isdigit():
                info["level"] = int(head[3:4])
        elif compression == "xz":
            with open(path, "rb") as fh:
                head = fh.read(12)
            if len(head) >= 8:
                info["check"] = head[7] & 0x0F
        elif compression == "zip":
            with zipfile.ZipFile(path) as zf:
                members = _file_members(zf)
                if members:
                    m = members[0]
                    info.update({"name": m.filename, "date_time": list(m.date_time),
                                 "compress_type": m.compress_type, "create_system": m.create_system,
                                 "external_attr": m.external_attr})
    except (OSError, zipfile.BadZipFile):
        pass
    return {LAYOUT_KEY: info}


def container_messages(extra: dict[str, Any] | None) -> list[Message]:
    """The ``container_name_replaced`` warning when the source's stored name is not the output's (never quoted)."""
    info = (extra or {}).get(LAYOUT_KEY) or {}
    derived = info.get("derived_name")
    if info.get("type") == "gz" and info.get("fname") and info["fname"] != derived:
        return [Message("warning", "container_name_replaced",
                        "the file name stored in the gzip header is not copied (it can hold free text); the "
                        f"output stores {derived!r}, the output file's own name")]
    if info.get("type") == "zip" and info.get("name") and info["name"] != derived:
        return [Message("warning", "container_name_replaced",
                        "the zip member's name is not copied (it can hold free text); the output's member is "
                        f"named {derived!r} after the output file (code that opens the member by name must use "
                        "this name)")]
    return []


# --------------------------------------------------------------------------- writing
def _inner_name(path: Path, compression: str) -> str:
    name = path.name
    if name.endswith(_TMP_SUFFIX):
        name = name[: -len(_TMP_SUFFIX)]
    suf = _SUFFIX.get(compression, "")
    if suf and name.lower().endswith(suf):
        name = name[: -len(suf)]
    return name


def _gzip_bytes(data: bytes, info: dict[str, Any], default_name: str) -> bytes:
    xfl = int(info.get("xfl", 0)) if info else 0
    level = 9 if xfl == 2 else 1 if xfl == 4 else 6
    # a stored name is written when the source had one, but never the source's (free text): the output's own
    fname = default_name if (not info or info.get("fname")) else None
    flg = _GZ_FLAGS["FNAME"] if fname else 0
    header = b"\x1f\x8b\x08" + bytes([flg]) + struct.pack("<I", int(info.get("mtime", 0)) & 0xFFFFFFFF)
    header += bytes([xfl & 0xFF, int(info.get("os", 255)) & 0xFF])
    if fname:
        header += fname.encode("latin-1", "replace") + b"\x00"
    comp = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    body = comp.compress(data) + comp.flush()
    trailer = struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF)
    return header + body + trailer


def write_bytes(path: Path, data: bytes, compression: str, layout: FileLayout) -> None:
    """Write ``data`` to ``path`` compressed like the source container."""
    path = Path(path)
    info: dict[str, Any] = dict((layout.extra or {}).get(LAYOUT_KEY) or {}) if layout is not None else {}
    inner = _inner_name(path, compression)
    if compression == "gz":
        path.write_bytes(_gzip_bytes(data, info, inner))
    elif compression == "bz2":
        level = int(info.get("level", 9))
        path.write_bytes(bz2.compress(data, compresslevel=min(max(level, 1), 9)))
    elif compression == "xz":
        check = _XZ_CHECKS.get(int(info.get("check", 4)), lzma.CHECK_CRC64)
        path.write_bytes(lzma.compress(data, format=lzma.FORMAT_XZ, check=check))
    elif compression == "zip":
        name = inner  # never the source member's name (free text): the output file's own name
        date_time = tuple(info.get("date_time") or (1980, 1, 1, 0, 0, 0))
        method = int(info.get("compress_type", zipfile.ZIP_DEFLATED))
        if method not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
            method = zipfile.ZIP_DEFLATED
        zi = zipfile.ZipInfo(name, date_time=date_time)  # type: ignore[arg-type]
        zi.compress_type = method
        if "create_system" in info:
            zi.create_system = int(info["create_system"])
        if "external_attr" in info:
            zi.external_attr = int(info["external_attr"])
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(zi, data)
    else:
        raise WriteError(f"unknown compression {compression!r}", fix="report this as a bug")


def container_diff(source: FileLayout, output: FileLayout) -> list[str]:
    """Container differences for the self-check (gzip time/level and stored name, zip method and member name).

    Stored names are never the source's: a gzip ``FNAME`` (present when the source had one)
    and the zip member must be the output file's own name (``derived_name``).
    """
    s = (source.extra or {}).get(LAYOUT_KEY) or {}
    o = (output.extra or {}).get(LAYOUT_KEY) or {}
    if not s or not o:
        return []
    issues: list[str] = []
    typ = str(s.get("type"))
    keys = {"gz": ("mtime", "xfl"), "zip": ("date_time", "compress_type"),
            "bz2": ("level",), "xz": ("check",)}.get(typ, ())
    for k in keys:
        if s.get(k) != o.get(k):
            issues.append(f"compressed container: {k} differs from the source")
    if typ == "gz":
        if bool(s.get("fname")) != bool(o.get("fname")):
            issues.append("compressed container: a stored file name is present in only one of source and output")
        elif o.get("fname") and o.get("fname") != o.get("derived_name"):
            issues.append("compressed container: the stored file name differs from the output file's name")
    elif typ == "zip" and o.get("name") != o.get("derived_name"):
        issues.append("compressed container: the member name differs from the output file's name")
    return issues
