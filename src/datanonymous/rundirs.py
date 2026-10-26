"""Standing output and report roots: one fresh, matching folder per run under each root.

``output_root`` / ``report_root`` (config keys) name two trees. A run with no ``--out-dir`` writes to
``<output_root>/<input-name>-<YYYYMMDD-HHMMSS>/`` and its report to ``<report_root>/<same name>/``; a name that is
taken in either root gets ``-2``, ``-3``, ... in both. An explicit ``--out-dir`` / ``--report-dir`` wins.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from . import paths
from .config import Settings
from .errors import UsageError


def now() -> datetime:
    """The clock for run-folder names (tests replace it)."""
    return datetime.now()


def _root(value: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def _input_name(input_path: Path) -> str:
    p = Path(os.path.abspath(str(input_path)))
    name = p.stem if p.is_file() else p.name
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "input"


def check_roots(settings: Settings, input_path: Path) -> None:
    """Refuse a configuration whose roots cannot keep the private report apart from the shareable output."""
    if settings.output_root and not settings.report_root:
        raise UsageError("output_root is set but report_root is not",
                         fix="set report_root too, so the private report never lands next to the shareable "
                             "output by default")
    if not (settings.output_root and settings.report_root):
        return
    out_root, rep_root = _root(settings.output_root), _root(settings.report_root)
    if paths.is_within(out_root, rep_root) or paths.is_within(rep_root, out_root):
        raise UsageError(f"output_root ({out_root}) and report_root ({rep_root}) must not be the same folder or "
                         "sit inside one another",
                         fix="choose two separate folders for them")
    folder = input_path if input_path.is_dir() else input_path.parent
    for root, key in ((out_root, "output_root"), (rep_root, "report_root")):
        if paths.is_within(root, folder):
            raise UsageError(f"{key} ({root}) is inside the input folder {folder}",
                             fix=f"choose a {key} outside the input")


def resolve(settings: Settings, input_path: Path, out_dir: Path | None,
            report_dir: Path | None) -> tuple[Path | None, Path | None]:
    """The run's folders: explicit ones kept, the others taken from the roots (a fresh timestamped name)."""
    input_path = Path(input_path)
    check_roots(settings, input_path)
    need_out = out_dir is None and bool(settings.output_root)
    need_rep = report_dir is None and bool(settings.report_root)
    if not (need_out or need_rep):
        return out_dir, report_dir
    stem = f"{_input_name(input_path)}-{now().strftime('%Y%m%d-%H%M%S')}"
    out_root = _root(settings.output_root) if need_out else None
    rep_root = _root(settings.report_root) if need_rep else None
    n, name = 1, stem
    while any((r / name).exists() for r in (out_root, rep_root) if r is not None):
        n += 1
        name = f"{stem}-{n}"
    if out_root is not None:
        out_dir = out_root / name
    if rep_root is not None:
        report_dir = rep_root / name
    return out_dir, report_dir
