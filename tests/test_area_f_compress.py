"""AREA F: .gz/.zip/.bz2/.xz for the CSV family and JSON lines (container mirrored on output)."""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import lzma
import struct
import zipfile
from pathlib import Path

import pytest

from datanonymous.errors import ReadError
from datanonymous.io import compress, csvio
from datanonymous.model import FileLayout
from test_area_f_util import FIX, all_bytes, read, report, run_dir, text_of

PAYLOAD = b"student_id,first_name,score\r\n1001,Ana,90\r\n1002,Ben,80\r\n"


def _layout(path: Path, compression: str) -> FileLayout:
    return FileLayout("csv", compression=compression, extra=compress.container_layout(path, compression))


def _gz_header(data: bytes) -> dict:
    return compress._gzip_header(data)


# --------------------------------------------------------------------------- unit: read/write mirror
def test_gzip_header_mirrored(tmp_path: Path) -> None:
    src = tmp_path / "a.csv.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(filename="original name.csv", mode="wb", fileobj=buf, mtime=1700000000, compresslevel=1) as g:
        g.write(PAYLOAD)
    src.write_bytes(buf.getvalue())
    assert compress.read_bytes(src, "gz") == PAYLOAD
    lay = _layout(src, "gz")
    info = lay.extra[compress.LAYOUT_KEY]
    assert info["fname"] == "original name.csv" and info["mtime"] == 1700000000 and info["xfl"] == 4
    out = tmp_path / "a.csv.gz.datanonymous-tmp"
    compress.write_bytes(out, b"x,y\n1,2\n", "gz", lay)
    h = _gz_header(out.read_bytes())
    # the stored name is free text: never copied, the output's own name instead (R1-01)
    assert (h["fname"], h["mtime"], h["xfl"], h["os"]) == ("a.csv", 1700000000, 4, info["os"])
    assert gzip.decompress(out.read_bytes()) == b"x,y\n1,2\n"
    assert compress.container_diff(lay, _layout(out, "gz")) == []


def test_gzip_without_name(tmp_path: Path) -> None:
    src = tmp_path / "b.csv.gz"
    src.write_bytes(gzip.compress(PAYLOAD, mtime=5))
    lay = _layout(src, "gz")
    out = tmp_path / "o.gz"
    compress.write_bytes(out, PAYLOAD, "gz", lay)
    assert out.read_bytes() == src.read_bytes()  # same header fields + level 9 deflate: identical bytes


def test_zip_member_mirrored(tmp_path: Path) -> None:
    src = tmp_path / "c.csv.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("exports/", b"")  # a directory entry is ignored
        zf.writestr(zipfile.ZipInfo("exports/real_name.csv", date_time=(2021, 5, 6, 7, 8, 10)), PAYLOAD,
                    compress_type=zipfile.ZIP_STORED)
    assert compress.read_bytes(src, "zip") == PAYLOAD
    lay = _layout(src, "zip")
    out = tmp_path / "c.csv.zip.datanonymous-tmp"
    compress.write_bytes(out, b"q\n", "zip", lay)
    with zipfile.ZipFile(out) as zf:
        (info,) = zf.infolist()
        # the member name is free text: never copied, named after the output file instead (R1-01)
        assert info.filename == "c.csv" and info.date_time == (2021, 5, 6, 7, 8, 10)
        assert info.compress_type == zipfile.ZIP_STORED and zf.read(info) == b"q\n"
    assert compress.container_diff(lay, _layout(out, "zip")) == []


def test_zip_fallback_member_name(tmp_path: Path) -> None:
    out = tmp_path / "students.csv.zip.datanonymous-tmp"
    compress.write_bytes(out, PAYLOAD, "zip", FileLayout("csv"))
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["students.csv"]


@pytest.mark.parametrize("n_members", [0, 2])
def test_zip_must_hold_one_file(tmp_path: Path, n_members: int) -> None:
    src = tmp_path / "m.csv.zip"
    with zipfile.ZipFile(src, "w") as zf:
        for i in range(n_members):
            zf.writestr(f"part{i}.csv", PAYLOAD)
    with pytest.raises(ReadError) as ei:
        compress.read_bytes(src, "zip")
    assert f"holds {n_members} files" in ei.value.message and "extract" in ei.value.fix


def test_bz2_and_xz_settings(tmp_path: Path) -> None:
    b = tmp_path / "x.csv.bz2"
    b.write_bytes(bz2.compress(PAYLOAD, compresslevel=3))
    lay = _layout(b, "bz2")
    assert lay.extra[compress.LAYOUT_KEY]["level"] == 3
    out = tmp_path / "o.bz2"
    compress.write_bytes(out, PAYLOAD, "bz2", lay)
    assert out.read_bytes() == b.read_bytes()
    x = tmp_path / "x.csv.xz"
    x.write_bytes(lzma.compress(PAYLOAD, format=lzma.FORMAT_XZ, check=lzma.CHECK_SHA256))
    lay = _layout(x, "xz")
    assert lay.extra[compress.LAYOUT_KEY]["check"] == lzma.CHECK_SHA256
    out = tmp_path / "o.xz"
    compress.write_bytes(out, PAYLOAD, "xz", lay)
    assert lzma.decompress(out.read_bytes()) == PAYLOAD and out.read_bytes()[7] & 0x0F == lzma.CHECK_SHA256


@pytest.mark.parametrize("compression,content", [("gz", b"plain text"), ("bz2", b"BZh9 nope"),
                                                 ("xz", b"\xfd7zXZ\x00junk"), ("zip", b"PK\x03\x04junk")])
def test_corrupt_containers_fail(tmp_path: Path, compression: str, content: bytes) -> None:
    p = tmp_path / f"bad.csv.{compression}"
    p.write_bytes(content)
    with pytest.raises(ReadError):
        compress.read_bytes(p, compression)


def test_compressible_formats() -> None:
    assert compress.COMPRESSIBLE_FORMATS == frozenset({"csv", "jsonl"})


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("name", ["students.csv.gz", "students.csv.zip"])
def test_cli_compressed_csv(tmp_path: Path, name: str) -> None:
    res, out, rep = run_dir(FIX / name, tmp_path)
    assert res.exit_code == 0, res.output
    comp = name.rsplit(".", 1)[1]
    src_text = compress.read_bytes(FIX / name, comp).decode("utf-8")
    out_text = compress.read_bytes(out / name, comp).decode("utf-8")
    real = list(csv.reader(io.StringIO(src_text, newline="")))
    fake = list(csv.reader(io.StringIO(out_text, newline="")))
    assert fake[0] == real[0] and len(fake) == len(real)
    assert out_text.endswith("\r\n") and "\r\n" in out_text  # CRLF mirrored inside the container
    if comp == "gz":
        assert _gz_header((out / name).read_bytes())["fname"] == "students.csv"
        assert _gz_header((out / name).read_bytes())["mtime"] == _gz_header((FIX / name).read_bytes())["mtime"]
    else:
        with zipfile.ZipFile(out / name) as zf:
            assert [(i.filename, i.date_time) for i in zf.infolist()] == [("students.csv", (2026, 10, 4, 9, 30, 0))]
    # privacy: the IDs are synthetic and the first names never real
    real_ids = {r[0] for r in real[1:]}
    real_names = {r[1] for r in real[1:]}
    assert not real_ids & {r[0] for r in fake[1:]}
    assert not real_names & {r[1] for r in fake[1:]}
    txt = out_text + text_of(rep)
    for v in real_ids:
        assert v not in txt
    data = report(rep)
    assert data["files"][0]["output"] == name
    lay_src = read(FIX / name, "csv", compression=comp).layout
    lay_out = read(out / name, "csv", compression=comp, purpose="selfcheck").layout
    assert compress.container_diff(lay_src, lay_out) == []
    assert csvio.inferred_dtypes(out / name, lay_out) == csvio.inferred_dtypes(FIX / name, lay_src)


@pytest.mark.parametrize("ext", ["bz2", "xz"])
def test_cli_bz2_xz_tsv(tmp_path: Path, ext: str) -> None:
    src = tmp_path / "in"
    src.mkdir()
    tsv = PAYLOAD.replace(b",", b"\t") + b"1003\tCy\t70\r\n"
    data = bz2.compress(tsv) if ext == "bz2" else lzma.compress(tsv, format=lzma.FORMAT_XZ)
    (src / f"scores.tsv.{ext}").write_bytes(data)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    back = compress.read_bytes(out / f"scores.tsv.{ext}", ext)
    assert back.splitlines()[0] == b"student_id\tfirst_name\tscore" and len(back.splitlines()) == 4


def test_unsupported_compressed_formats_are_skipped(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    (src / "t.parquet.gz").write_bytes(gzip.compress((FIX / "types.parquet").read_bytes()))
    (src / "r.json.gz").write_bytes(gzip.compress((FIX / "records.json").read_bytes()))
    (src / "ok.csv.gz").write_bytes(gzip.compress(PAYLOAD))
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert sorted(s["path"] for s in report(rep)["skipped"]) == ["r.json.gz", "t.parquet.gz"]
    assert (out / "ok.csv.gz").is_file() and not (out / "t.parquet.gz").exists()


def test_multi_member_zip_fails_in_cli(tmp_path: Path) -> None:
    src = tmp_path / "in"
    src.mkdir()
    with zipfile.ZipFile(src / "two.csv.zip", "w") as zf:
        zf.writestr("a.csv", PAYLOAD)
        zf.writestr("b.csv", PAYLOAD)
    res, out, rep = run_dir(src, tmp_path)
    assert res.exit_code == 3
    f = report(rep)["failures"][0]
    assert f["source"] == "two.csv.zip" and "extract the archive" in f["fixes"][0]
    assert b"Ana" not in all_bytes(rep)


def test_compressed_same_seed_same_bytes(tmp_path: Path) -> None:
    for name in ("students.csv.gz", "students.csv.zip"):
        r1, o1, _ = run_dir(FIX / name, tmp_path / name / "a", seed="4")
        r2, o2, _ = run_dir(FIX / name, tmp_path / name / "b", seed="4")
        assert r1.exit_code == r2.exit_code == 0
        assert (o1 / name).read_bytes() == (o2 / name).read_bytes()


def test_gzip_header_parser_handles_extra_field() -> None:
    raw = b"\x1f\x8b\x08" + bytes([4 | 8]) + struct.pack("<I", 9) + b"\x00\x03" + b"\x02\x00ab" + b"n.csv\x00"
    assert compress._gzip_header(raw)["fname"] == "n.csv"
