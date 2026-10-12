"""Private, array-lifetime codes for immutable raw string columns.

Codes and counts preserve first appearance. Only read-only ``str``/``None``
object arrays with no writable ndarray base are cached; other inputs retain
pandas factorization and Python Counter semantics independently. Cached entries
never own their source ndarray, and an identity check protects against id reuse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import weakref

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pandas._libs import lib


@dataclass
class _Entry:
    source: weakref.ReferenceType[np.ndarray]
    has_nul: bool = False
    arrow: pa.Array | pa.ChunkedArray | None = None
    result: tuple[np.ndarray, np.ndarray] | None = None
    counts: dict[str | None, int] | None = None


_CACHE: dict[int, _Entry] = {}


def _immutable(values: np.ndarray) -> bool:
    current = values
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        current = current.base
    return True


def _string_kind(values: np.ndarray) -> int:
    # skipna=True also accepts NaN/pd.NA; those must retain Counter's distinct
    # null semantics. Exact types also exclude subclasses with custom equality.
    if (values.ndim != 1 or values.dtype != object
            or (len(values) and not lib.is_string_array(values, skipna=True))):
        return 0
    kind = 1
    for value in values:
        if type(value) is str:
            if "\x00" in value:
                kind = 2
        elif value is not None:
            return 0
    return kind


def _strings(values: np.ndarray) -> bool:
    return bool(_string_kind(values))


def _entry(values: np.ndarray) -> _Entry | None:
    entry = _CACHE.get(id(values))
    if entry is not None:
        if entry.source() is values and _immutable(values):
            return entry
        _CACHE.pop(id(values), None)
    return None


def _new_entry(values: np.ndarray, contains_nul: bool = False) -> _Entry:
    key = id(values)

    def discard(ref: weakref.ReferenceType[np.ndarray]) -> None:
        # A stale callback must never remove a newer array using the same id.
        entry = _CACHE.get(key)
        if entry is not None and entry.source is ref:
            _CACHE.pop(key, None)

    entry = _Entry(weakref.ref(values, discard), has_nul=contains_nul)
    _CACHE[key] = entry
    return entry


def _string_entry(values: np.ndarray) -> _Entry | None:
    entry = _entry(values)
    if entry is not None:
        return entry
    if not _immutable(values):
        return None
    kind = _string_kind(values)
    return _new_entry(values, contains_nul=kind == 2) if kind else None


def has_nul(values: np.ndarray) -> bool:
    """Whether exact raw-string distinctions need the legacy consumer's path.

    pandas' string factorizer can equate strings after their first NUL, unlike
    Counter or single-column duplicate detection. Reader arrays retain this
    eligibility fact with their codes, so consumers need no extra row scan.
    """
    entry = _entry(values)
    if entry is not None:
        return entry.has_nul
    return any(isinstance(value, str) and "\x00" in value for value in values)


def register_arrow(values: np.ndarray, array: pa.Array | pa.ChunkedArray) -> None:
    """Retain a reader's matching Arrow string column for cheap factorization.

    The reader must pass the exact same row values in both representations.
    Writable inputs (including writable bases) are deliberately not registered.
    Call after constructing RawColumn, or after explicitly freezing ``values``.
    Arrow buffers are immutable and live only as long as the source ndarray.
    """
    if len(values) != len(array):
        raise ValueError("Arrow column and values must have equal lengths")
    if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
        raise TypeError("Arrow column must contain strings")
    entry = _string_entry(values)
    if entry is None or entry.has_nul:
        return
    entry.arrow = array


def _readonly(codes: np.ndarray, uniques: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Bytes-backed codes cannot have writeability re-enabled with setflags.
    codes = np.frombuffer(np.asarray(codes, dtype=np.int64).tobytes(), dtype=np.int64)
    uniques = np.asarray(uniques, dtype=object)
    uniques.flags.writeable = False
    return codes, uniques


def _factorize(values: np.ndarray, strings: bool) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    uniques = np.asarray(uniques, dtype=object)
    if strings:
        # pandas represents native None as NaN in the uniques array.
        uniques = np.array([value if isinstance(value, str) else None for value in uniques], dtype=object)
    return _readonly(codes, uniques)


def _arrow_codes(array: pa.Array | pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    encoded = pc.dictionary_encode(array)
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    codes = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    uniques = encoded.dictionary.to_pylist()
    missing = codes == -1
    if missing.any():
        # Arrow excludes null from its dictionary. Insert it where its first
        # occurrence belongs, shifting later dictionary codes by one.
        first = int(np.flatnonzero(missing)[0])
        position = int(codes[:first].max()) + 1 if first else 0
        codes[(codes >= position) & ~missing] += 1
        codes[missing] = position
        uniques.insert(position, None)
    return _readonly(codes, np.asarray(uniques, dtype=object))


def codes_of(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read-only int64 codes and object uniques, in first-appearance order."""
    entry = _entry(values)
    if entry is not None and entry.result is not None:
        return entry.result
    entry = entry or _string_entry(values)
    if entry is not None:
        entry.result = (_arrow_codes(entry.arrow) if entry.arrow is not None
                        else _factorize(values, True))
        return entry.result
    return _factorize(values, _strings(values))


def counts_of(values: np.ndarray) -> dict:
    """A fresh dict with exactly Counter(values.tolist()) counts and order."""
    entry = _string_entry(values)
    if entry is None:
        return dict(Counter(values.tolist()))
    if entry.counts is None:
        if entry.has_nul:
            entry.counts = dict(Counter(values.tolist()))
        else:
            codes, uniques = codes_of(values)
            frequencies = np.bincount(codes, minlength=len(uniques))
            entry.counts = {value: int(count) for value, count in zip(uniques, frequencies)}
    return entry.counts.copy()


def subset_codes(values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factorize a row selection through immutable parent's integer codes."""
    entry = _string_entry(values)
    if entry is None or entry.has_nul:
        selected_values = values[rows]
        return _factorize(selected_values, _strings(selected_values))
    codes, uniques = codes_of(values)
    selected, labels = pd.factorize(codes[rows], use_na_sentinel=False)
    return _readonly(selected, uniques[labels])
