"""Detection context and result passed to every detector and area hook (FOUNDATION).

Detection works on the **distinct stripped values with their counts** (design
3), never on rows. :class:`DetectContext` precomputes them once per column in
two views, following the missing-token rule (design 3, section 10):

* ``typed_values``: stripped value -> count, excluding blanks/``None`` **and**
  the case-sensitive NA tokens (``NA``, ``N/A``, ``NULL`` ... + ``--na-value``,
  + the non-finite numbers the column stores natively, ``model.numeric_specials``:
  an Arrow float's ``inf``, kept verbatim like ``NaN``, K-20).
  Use it for typed checks (numbers, dates, booleans, ordinals).
* ``text_values``: stripped value -> count excluding only blanks/``None``. Use it
  for text checks: in text/factor columns NA tokens stay levels (the race code
  ``NA`` and the StudentsPerformance level ``none``).

Hooks return a :class:`Detection` or ``None`` (no match: the pipeline falls
through to the next rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ColumnOptions, Settings
from ..hints import HeaderHints, header_hints
from ..model import TYPE_INFO, ColumnType, Message, numeric_specials
from ..profile.codes import counts_of


@dataclass
class Detection:
    """Result of one detector.

    ``params`` are type-specific detection parameters that generators/renderers
    need (formats, token pairs, level order, decimals ...). Keys starting with
    ``_`` are private (never written to reports). ``missing_policy`` defaults to
    the type's policy in :data:`~datanonymous.model.TYPE_INFO`.
    """

    type: ColumnType
    subtype: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    source: str = "value"
    ambiguous: bool = False
    ambiguity: str | None = None
    missing_policy: str | None = None
    messages: list[Message] = field(default_factory=list)

    def policy(self) -> str:
        return self.missing_policy or TYPE_INFO[self.type].missing_policy


def is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


class DetectContext:
    """Everything a detector may look at for one column."""

    def __init__(
        self,
        *,
        index: int,
        name: str,
        values: np.ndarray,
        file_rel: str,
        table: str | None,
        format: str,
        options: ColumnOptions,
        settings: Settings,
        native: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.name = name
        self.values = values
        self.file_rel = file_rel
        self.table = table
        self.format = format
        self.options = options
        self.settings = settings
        self.native = native
        self.n = len(values)
        # K-20: non-finite numbers stored natively (Arrow float inf, JSON Infinity) are kept like NaN, never text
        self.na_tokens: frozenset[str] = settings.na_tokens | numeric_specials(native)
        self.hints: HeaderHints = header_hints(name)
        self.digit_code = False
        self.forced_type: ColumnType | None = None
        self.raw_counts: dict[Any, int] = counts_of(values)
        self.blank_raw: dict[Any, int] = {}
        self.na_raw: dict[Any, int] = {}
        text: dict[str, int] = {}
        typed: dict[str, int] = {}
        for raw, c in self.raw_counts.items():
            if is_blank(raw):
                self.blank_raw[raw] = c
                continue
            s = raw.strip()
            text[s] = text.get(s, 0) + c
            if s in self.na_tokens:
                self.na_raw[raw] = self.na_raw.get(raw, 0) + c
            else:
                typed[s] = typed.get(s, 0) + c
        self.text_values = text
        self.typed_values = typed

    # ------------------------------------------------------------------ helpers
    @property
    def factor_max(self) -> int:
        return self.settings.factor_max

    @property
    def n_blank(self) -> int:
        return sum(self.blank_raw.values())

    @property
    def n_na_tokens(self) -> int:
        return sum(self.na_raw.values())

    def casefold_distinct(self, values: dict[str, int] | None = None) -> int:
        """Distinct count after strip + case-fold (factor threshold counting)."""
        vals = self.text_values if values is None else values
        return len({v.casefold() for v in vals})

    def is_missing(self, raw: Any, policy: str) -> bool:
        if is_blank(raw):
            return True
        return policy == "typed" and raw.strip() in self.na_tokens

    def split_counts(self, policy: str) -> tuple[dict[Any, int], dict[Any, int]]:
        """(non-missing raw -> count, missing raw -> count) under ``policy``."""
        nonmissing: dict[Any, int] = {}
        missing: dict[Any, int] = {}
        for raw, c in self.raw_counts.items():
            (missing if self.is_missing(raw, policy) else nonmissing)[raw] = c
        return nonmissing, missing

    def mean_words_and_length(self, values: dict[str, int] | None = None) -> tuple[float, float]:
        """Row-weighted mean word count and mean length of non-blank values."""
        vals = self.text_values if values is None else values
        total = sum(vals.values())
        if not total:
            return 0.0, 0.0
        words = sum(len(v.split()) * c for v, c in vals.items()) / total
        length = sum(len(v) * c for v, c in vals.items()) / total
        return words, length
