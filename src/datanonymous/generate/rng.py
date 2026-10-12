"""Seeded random streams (FOUNDATION).

Every random draw comes from a stream keyed by (seed, file, table, column,
purpose): ``SeedSequence(seed, spawn_key=(crc32(...), ...))``. Python's ``hash()``
is salted per process and is never used (design 7, D26). The same seed, input
and library versions give byte-identical output.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np


def _key(part: str | int | None) -> int:
    if part is None:
        return 0
    if isinstance(part, int):
        if part < 0:
            raise ValueError("stream key parts must be non-negative")
        return part
    return zlib.crc32(str(part).encode("utf-8"))


def seed_sequence(seed: int, *parts: str | int | None) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed, spawn_key=tuple(_key(p) for p in parts))


def stream(seed: int, *parts: str | int | None) -> np.random.Generator:
    """A Generator for the given key parts (str parts are crc32-hashed)."""
    return np.random.Generator(np.random.PCG64(seed_sequence(seed, *parts)))


class RngFactory:
    """Streams for one table of one file."""

    def __init__(self, seed: int, file_rel: str, table: str | None) -> None:
        self.seed = seed
        self.file_rel = file_rel
        self.table = table or ""

    def column(self, index: int, name: str, purpose: str = "values") -> np.random.Generator:
        return stream(self.seed, self.file_rel, self.table, index, name, purpose)

    def table_stream(self, purpose: str) -> np.random.Generator:
        return stream(self.seed, self.file_rel, self.table, "__table__", purpose)

    def int_seed(self, index: int, name: str, purpose: str) -> int:
        """A 32-bit integer seed (for libraries such as Faker)."""
        ss = seed_sequence(self.seed, self.file_rel, self.table, index, name, purpose)
        return int(ss.generate_state(1, dtype=np.uint32)[0])


def faker_for(seed: int, *parts: str | int | None, locale: str = "en_US") -> Any:
    """A Faker instance seeded from the key parts (AREA C uses it; D24).

    Faker is imported lazily (slow import). The Faker version is pinned and
    recorded in the manifest because its output for a seed may change.
    """
    from faker import Faker

    fk = Faker(locale)
    fk.seed_instance(int(seed_sequence(seed, *parts).generate_state(1, dtype=np.uint32)[0]))
    return fk
