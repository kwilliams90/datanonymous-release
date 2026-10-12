"""Helpers shared by the AREA F format tests (no tests here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from conftest import run_cli
from datanonymous.config import Settings
from datanonymous.io import registry
from datanonymous.io.registry import OutTable, ReadRequest, WriteRequest
from datanonymous.model import ColumnProfile, ColumnType, ReadResult, TableProfile

FIX = Path(__file__).resolve().parent / "fixtures" / "formats"
#: Columns of types.parquet / types.feather that need AREA A (dates) or AREA B (DECIMAL) generators.
AB_COLUMNS = ("balance", "enrolled_at", "dob", "start_time")


def read(path: Path, fmt: str, rel: str | None = None, compression: str | None = None,
         settings: Settings | None = None, purpose: str = "source") -> ReadResult:
    return registry.get(fmt).read(ReadRequest(Path(path), rel or Path(path).name, fmt, compression,
                                              settings or Settings(mode="fidelity"), purpose=purpose))


def identity_write(rr: ReadResult, fmt: str, out: Path, columns: list[np.ndarray] | None = None,
                   keep: list[int] | None = None, types: dict[int, ColumnType] | None = None) -> Path:
    """Write the read values back unchanged (no generation): the I/O identity round trip."""
    t = rr.tables[0]
    keep = list(range(len(t.columns))) if keep is None else keep
    profiles = []
    for i, c in enumerate(t.columns):
        p = ColumnProfile(index=i, name=c.name, type=(types or {}).get(i, ColumnType.FACTOR))
        p.native = c.native
        profiles.append(p)
    tp = TableProfile(t.name, list(t.header), t.n_rows, profiles)
    cols = columns if columns is not None else [t.columns[i].values for i in keep]
    ot = OutTable(t.name, [t.header[i] for i in keep], cols, keep, t.n_rows, dict(t.layout), tp)
    registry.get(fmt).write(WriteRequest(Path(out), rr.layout, [ot], Settings(mode="fidelity"), seed=1))
    return Path(out)


def cli(*args: Any) -> Any:
    return run_cli(*args)


def run_dir(src: Path, tmp: Path, *flags: str, seed: str = "7") -> tuple[Any, Path, Path]:
    out, rep = tmp / "out", tmp / "rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *flags)
    return res, out, rep


def report(rep: Path) -> dict[str, Any]:
    return json.loads((rep / "report.json").read_text(encoding="utf-8"))


def all_bytes(*dirs: Path) -> bytes:
    """Every file under the given directories, concatenated (for leak assertions)."""
    chunks = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                chunks.append(p.read_bytes())
    return b"\n".join(chunks)


def text_of(*dirs: Path) -> str:
    """Decoded text of every non-binary file under the directories."""
    out = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".md", ".json", ".csv", ".jsonl", ".txt"):
                out.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)
