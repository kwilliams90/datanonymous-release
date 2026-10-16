"""Review round 2, paths cluster: the input, the out dir and the report dir are three separate REAL trees.

R2-07: an out dir that is a junction (or symlink) to the input passed the lexical checks and the run replaced
the real input files. R2-08: an out dir inside the report dir let the per-file report replace a synthetic file
and put report files into the shareable out dir.

The bug class is "two spellings, one folder": every overlap test now compares real locations (links,
junctions, 8.3 short names, ``\\\\?\\`` prefixes and admin shares resolved, file identity as a fallback) in
both containment directions, an existing out/report dir may not hold a link that leads out of it, the input
walk refuses folders or files that really live in the out/report dir and skips link loops, and docs, report
files and temporary files never write through an entry already at their name (symlink or hard link).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from conftest import run_cli
from datanonymous import paths
from datanonymous.discover import check_dirs, discover
from conftest import build_fidelity_settings as build_settings
from datanonymous.errors import UsageError

WINDOWS = sys.platform == "win32"


# --------------------------------------------------------------------------- helpers
def _input(base: Path, sub: str = "") -> Path:
    d = base / "in" / sub if sub else base / "in"
    d.mkdir(parents=True)
    rows = ["first_name,score"] + [f"Zelmira{i},{i}" for i in range(30)]
    (d / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return base / "in"


def _snapshot(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def _junction(link: Path, target: Path) -> None:
    if not WINDOWS:
        pytest.skip("junctions are Windows-only")
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def _symlink(link: Path, target: Path, is_dir: bool) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError) as e:  # Windows without the symlink privilege
        pytest.skip(f"cannot create symlinks here: {e}")


def _dir_link(link: Path, target: Path, kind: str) -> None:
    if kind == "junction":
        _junction(link, target)
    else:
        _symlink(link, target, True)


LINK_KINDS = ["junction", "symlink"]


def _refused(base: Path, *args, watch: Path, needle: str) -> str:
    before = _snapshot(watch)
    res = run_cli(*args, "--seed", "1")
    assert res.exit_code == 2, res.output
    assert needle in res.output, res.output
    assert _snapshot(watch) == before  # nothing replaced, nothing added
    return res.output


# --------------------------------------------------------------------------- R2-07 and its aliasing variants
@pytest.mark.parametrize("kind", LINK_KINDS)
def test_out_dir_link_to_the_input_is_refused_and_the_input_is_untouched(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    _dir_link(tmp_path / "alias", src, kind)
    out = _refused(tmp_path, src, "--out-dir", tmp_path / "alias", "--report-dir", tmp_path / "rep", "--force",
                   watch=src, needle="inside the input folder")
    assert "resolves to" in out
    assert not (tmp_path / "rep").exists()


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_report_dir_link_to_the_input_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    _dir_link(tmp_path / "alias", src, kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "alias", "--force",
             watch=src, needle="--report-dir")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_new_folder_under_a_link_to_the_input_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    _dir_link(tmp_path / "alias", src, kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "alias" / "new" / "out", "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")
    assert not (src / "new").exists()


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_linked_parent_folder_in_the_out_path_is_resolved(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "x").mkdir()
    _dir_link(tmp_path / "x" / "up", tmp_path, kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "x" / "up" / "in" / "o", "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_out_dir_link_to_an_ancestor_of_the_input_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "x").mkdir()
    _dir_link(tmp_path / "x" / "alias", tmp_path, kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "x" / "alias", "--report-dir", tmp_path / "rep", "--force",
             watch=tmp_path / "in", needle="the input must not be inside the out dir")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_input_given_through_a_link_into_the_out_dir_is_refused(tmp_path: Path, kind: str) -> None:
    real = _input(tmp_path / "out")
    _dir_link(tmp_path / "in", real, kind)
    _refused(tmp_path, tmp_path / "in", "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=real, needle="the input must not be inside the out dir")


def test_single_file_input_whose_real_file_lives_in_the_out_dir_is_refused(tmp_path: Path) -> None:
    src = _input(tmp_path)
    (tmp_path / "real").mkdir()
    (src / "data.csv").replace(tmp_path / "real" / "data.csv")
    _symlink(src / "data.csv", tmp_path / "real" / "data.csv", False)
    _refused(tmp_path, src / "data.csv", "--out-dir", tmp_path / "real", "--report-dir", tmp_path / "rep",
             "--force", watch=tmp_path / "real", needle="the folder of the real input file")


@pytest.mark.skipif(not WINDOWS, reason="Windows path spellings")
def test_extended_length_prefix_spelling_is_resolved(tmp_path: Path) -> None:
    src = _input(tmp_path)
    _refused(tmp_path, src, "--out-dir", "\\\\?\\" + str(src / "o"), "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")


@pytest.mark.skipif(not WINDOWS, reason="Windows 8.3 short names")
def test_short_name_spelling_is_resolved(tmp_path: Path) -> None:
    src = _input(tmp_path)
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(src), buf, 1024)
    if not buf.value or os.path.normcase(buf.value) == os.path.normcase(str(src)):
        pytest.skip("no 8.3 short names on this volume")
    _refused(tmp_path, src, "--out-dir", buf.value + "\\o", "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")


@pytest.mark.skipif(not WINDOWS, reason="Windows admin shares")
def test_admin_share_spelling_is_caught_by_file_identity(tmp_path: Path) -> None:
    src = _input(tmp_path)
    a = os.path.abspath(src)
    unc = "\\\\localhost\\" + a[0] + "$" + a[2:]
    if not os.path.isdir(unc):
        pytest.skip("admin share not reachable")
    assert paths.real_path(unc) != paths.real_path(src)  # the strings differ: identity decides
    assert paths.is_within(unc + "\\o", src) and paths.is_within(src, unc)
    _refused(tmp_path, src, "--out-dir", unc + "\\o", "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")


def test_case_variant_spelling_on_a_case_insensitive_volume(tmp_path: Path) -> None:
    src = _input(tmp_path)
    upper = tmp_path / "IN"
    if not upper.is_dir():
        pytest.skip("case-sensitive file system")
    _refused(tmp_path, src, "--out-dir", upper / "o", "--report-dir", tmp_path / "rep",
             watch=src, needle="inside the input folder")


# --------------------------------------------------------------------------- R2-08 and the out/report overlap
def test_out_dir_inside_the_report_dir_is_refused_before_writing(tmp_path: Path) -> None:
    """The reviewer's case: data.csv and data.csv.json, --out-dir REP/files --report-dir REP."""
    src = _input(tmp_path)
    (src / "data.csv.json").write_text(json.dumps([{"amount": i} for i in range(30)]), encoding="utf-8")
    rep = tmp_path / "rep"
    out = _refused(tmp_path, src, "--out-dir", rep / "files", "--report-dir", rep, watch=src,
                   needle="must not be inside the report dir")
    assert "shareable out dir" in out
    assert not rep.exists()


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_out_dir_linked_into_the_report_dir_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "rep" / "files").mkdir(parents=True)
    _dir_link(tmp_path / "oalias", tmp_path / "rep" / "files", kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "oalias", "--report-dir", tmp_path / "rep", "--force",
             watch=tmp_path / "rep", needle="must not be inside the report dir")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_report_dir_linked_to_the_out_dir_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "out").mkdir()
    _dir_link(tmp_path / "ralias", tmp_path / "out", kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "ralias", "--force",
             watch=tmp_path / "out", needle="must not be the out dir or inside it")


def test_out_dir_equal_to_report_dir_and_report_inside_out_still_refused(tmp_path: Path) -> None:
    src = _input(tmp_path)
    out = tmp_path / "out"
    for rep in (out, out / "rep", out / "." / "a" / ".." / "rep"):
        res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "1")
        assert res.exit_code == 2 and "must not be the out dir or inside it" in res.output, res.output
    assert not out.exists()


def test_check_dirs_is_symmetric_for_out_and_report(tmp_path: Path) -> None:
    src = _input(tmp_path)
    base = tmp_path / "share"
    for out, rep in ((base, base / "r"), (base / "o", base), (base, base)):
        with pytest.raises(UsageError):
            check_dirs(src, out, rep, force=True)
    check_dirs(src, base / "o", base / "r", force=True)  # siblings are fine


def test_report_dir_inside_the_input_is_allowed(tmp_path: Path) -> None:
    src = _input(tmp_path)
    check_dirs(src, tmp_path / "out", src / "report", force=False)
    check_dirs(src / "data.csv", tmp_path / "out", src / "report", force=False)  # a file input: its folder


@pytest.mark.parametrize("mode", ["fidelity", "sharing"])
def test_a_run_never_reads_its_report_dir_inside_the_input(tmp_path: Path, mode: str) -> None:
    """In sharing mode the staging folder already sits in the report dir while the input is walked."""
    src = _input(tmp_path)
    run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", src / "report", "--mode", mode, "--seed", "1")
    report = json.loads((src / "report" / "report.json").read_text(encoding="utf-8"))
    assert [f["source"] for f in report["files"]] == ["data.csv"]
    assert not report["skipped"]


def test_a_non_empty_report_dir_inside_the_input_is_refused_even_with_force(tmp_path: Path) -> None:
    """--force must never let report files replace input files."""
    src = _input(tmp_path)
    (src / "reports").mkdir()
    (src / "reports" / "report.json").write_text('[{"real": 1}]', encoding="utf-8")
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", src / "reports", "--force",
             watch=src, needle="a report folder inside the input must be new or empty")


def test_input_inside_the_report_dir_is_still_refused(tmp_path: Path) -> None:
    """Report files written into the input root could replace real input files."""
    src = _input(tmp_path)
    for rep in (src, tmp_path):
        with pytest.raises(UsageError, match="the input must not be inside the report dir"):
            check_dirs(src, tmp_path / "out", rep, force=True)


# --------------------------------------------------------------------------- links inside an existing out/report dir
@pytest.mark.parametrize("kind", LINK_KINDS)
def test_link_inside_the_out_dir_leading_into_the_input_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path, "sub")
    (tmp_path / "out").mkdir()
    _dir_link(tmp_path / "out" / "sub", src / "sub", kind)  # the mirror of sub/data.csv would land in the input
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=src, needle="holds a link sub that leads out of it")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_link_inside_the_report_dir_leading_into_the_input_is_refused(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "rep").mkdir()
    _dir_link(tmp_path / "rep" / "files", src, kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=src, needle="holds a link files that leads out of it")


def test_file_symlink_at_a_doc_name_is_refused(tmp_path: Path) -> None:
    src = _input(tmp_path)
    (tmp_path / "out").mkdir()
    _symlink(tmp_path / "out" / "schema.json", src / "data.csv", False)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=src, needle="holds a link schema.json")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_link_that_stays_inside_the_out_dir_is_allowed(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    (tmp_path / "out" / "keep").mkdir(parents=True)
    _dir_link(tmp_path / "out" / "alias", tmp_path / "out" / "keep", kind)
    res = run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force", "--seed", "1")
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "data.csv").is_file()


def test_hard_links_at_output_names_are_replaced_never_written_through(tmp_path: Path) -> None:
    """A hard link to an input file left at the temporary name, a doc name or a report name."""
    src = _input(tmp_path)
    out, rep = tmp_path / "out", tmp_path / "rep"
    out.mkdir()
    rep.mkdir()
    for p in (out / "data.csv.datanonymous-tmp", out / "README.md", out / "data.csv", rep / "REPORT.md",
              rep / "report.json"):
        try:
            os.link(src / "data.csv", p)
        except OSError as e:
            pytest.skip(f"no hard links here: {e}")
    before = _snapshot(src)
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--force", "--seed", "1")
    assert res.exit_code == 0, res.output
    assert _snapshot(src) == before
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# SYNTHETIC DATA")
    assert json.loads((rep / "report.json").read_text(encoding="utf-8"))["mode"] == "run"
    assert not (out / "data.csv.datanonymous-tmp").exists()
    assert "Zelmira" not in (out / "data.csv").read_text(encoding="utf-8")


def test_replace_bytes_and_remove_entry_never_touch_a_link_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"real")
    link = tmp_path / "hard.txt"
    os.link(target, link)
    paths.replace_bytes(link, b"synthetic")
    assert target.read_bytes() == b"real" and link.read_bytes() == b"synthetic"
    tmp = tmp_path / ("x" + paths.TMP_SUFFIX)
    os.link(target, tmp)
    paths.remove_entry(tmp)
    assert not tmp.exists() and target.read_bytes() == b"real"
    folder = tmp_path / "folder"
    folder.mkdir()
    paths.remove_entry(folder)  # a real folder is kept
    assert folder.is_dir()
    _symlink(tmp_path / "sym.txt", target, False)
    paths.replace_bytes(tmp_path / "sym.txt", b"new")
    assert target.read_bytes() == b"real" and not (tmp_path / "sym.txt").is_symlink()


# --------------------------------------------------------------------------- the input walk
@pytest.mark.parametrize("kind", LINK_KINDS)
def test_input_folder_linking_into_the_out_dir_is_refused(tmp_path: Path, kind: str) -> None:
    """in/sub -> out/sub: its real files would be replaced by their own mirror."""
    src = _input(tmp_path)
    (tmp_path / "out" / "sub").mkdir(parents=True)
    (tmp_path / "out" / "sub" / "data.csv").write_bytes((src / "data.csv").read_bytes())
    _dir_link(src / "sub", tmp_path / "out" / "sub", kind)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=tmp_path / "out" / "sub", needle="the input entry sub")


def test_input_file_linking_into_the_report_dir_is_refused(tmp_path: Path) -> None:
    src = _input(tmp_path)
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep" / "x.csv").write_bytes((src / "data.csv").read_bytes())
    _symlink(src / "x.csv", tmp_path / "rep" / "x.csv", False)
    _refused(tmp_path, src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--force",
             watch=tmp_path / "rep", needle="the input entry x.csv")


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_link_loop_in_the_input_is_skipped_not_walked_forever(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path, "a")
    _dir_link(src / "a" / "loop", src, kind)
    _dir_link(src / "self", src, kind)
    res = run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--seed", "1")
    assert res.exit_code == 0, res.output
    report = json.loads((tmp_path / "rep" / "report.json").read_text(encoding="utf-8"))
    assert {s["path"] for s in report["skipped"]} == {"a/loop/", "self/"}
    assert [f["source"] for f in report["files"]] == ["a/data.csv"]
    disc = discover(src, build_settings({}, None))  # inspect walks the same way
    assert [f.rel_path for f in disc.files] == ["a/data.csv"]


@pytest.mark.parametrize("kind", LINK_KINDS)
def test_linked_folder_in_the_input_is_followed_and_mirrored(tmp_path: Path, kind: str) -> None:
    src = _input(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "e.csv").write_bytes((src / "data.csv").read_bytes())
    _dir_link(src / "linked", other, kind)
    res = run_cli(src, "--out-dir", tmp_path / "out", "--report-dir", tmp_path / "rep", "--seed", "1")
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "linked" / "e.csv").is_file()
    assert not paths.is_link(tmp_path / "out" / "linked")  # a real folder in the out dir


def test_is_within_unit_cases(tmp_path: Path) -> None:
    a = tmp_path / "a"
    (a / "b").mkdir(parents=True)
    assert paths.is_within(a / "b", a) and paths.is_within(a, a) and not paths.is_within(a, a / "b")
    assert paths.is_within(a / "missing" / "deeper", a)  # not created yet
    assert not paths.is_within(tmp_path / "ab", a)  # a name prefix is not containment
    assert paths.is_within(str(a / "b" / ".." / "b"), a)
