"""Shared helpers for the AREA A (dates) tests. Holds no tests itself."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import dates as D
from datanonymous.detect.context import DetectContext
from datanonymous.detect.types import detect_column
from datanonymous.generate.ids import IdRegistry
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.model import RawColumn, RawTable, TableProfile

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dates"


def arr(values: list[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def ctx_for(values: list[Any], name: str = "col", native: dict[str, Any] | None = None, **cli: Any) -> DetectContext:
    s = build_settings(cli)
    return DetectContext(index=0, name=name, values=arr(values), file_rel="f.csv", table=None, format="csv",
                         options=s.column_options("f.csv", None, name), settings=s, native=native)


def detect(values: list[Any], name: str = "col", native: dict[str, Any] | None = None, **cli: Any):
    """Full detection pipeline (detect/types.py) for one column."""
    return detect_column(ctx_for(values, name, native, **cli))


def run_table(columns: dict[str, list[Any]], seed: int = 1, **cli: Any) -> tuple[TableProfile, dict[str, np.ndarray]]:
    """Profile and generate one table with the FOUNDATION pipeline (no writer)."""
    s = build_settings(cli)
    n = len(next(iter(columns.values())))
    raw = RawTable(None, list(columns), [RawColumn(k, arr(v)) for k, v in columns.items()], n)
    from datanonymous.profile.table import profile_table

    tp = profile_table(raw, "f.csv", "csv", s)
    reg = IdRegistry()
    namespaces: dict[int, str] = {}
    from datanonymous.profile import entity as entity_mod

    for i in tp.entity.id_columns:
        key = entity_mod.id_namespace("f.csv", None, tp.columns[i])
        reg.register(key, "f.csv", None, i, tp.columns[i].name, tp.private.id_raw[i])
        namespaces[i] = key
    reg.build(seed)
    st = generate_table(tp, GenContext(seed, "f.csv", reg, namespaces, s))
    return tp, {c.name: c.values for c in st.columns}


def profile_of(tp: TableProfile, name: str):
    return next(p for p in tp.columns if p.name == name)


def days(start: dt.date, n: int, rng: np.random.Generator, span: int = 300) -> list[dt.date]:
    return [start + dt.timedelta(days=int(x)) for x in rng.integers(0, span, n)]


def parse_all(profile: Any, values: list[str]) -> Any:
    """ColumnView of the stripped non-missing values under the profile's params."""
    vals = [v.strip() for v in values if isinstance(v, str) and v.strip() and v.strip() not in profile.na_tokens]
    return D.view(profile.params, vals), vals
