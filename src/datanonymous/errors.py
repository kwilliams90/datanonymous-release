"""Exception types shared by every stage (FOUNDATION).

Every error the tool raises on purpose is a :class:`DatanonymousError`. It
carries a ``fix`` hint: the exact flag or action that resolves it. The pipeline
turns a per-file error into a FAIL entry with that fix (design D19), and the CLI
maps :class:`UsageError` to exit code 2.
"""

from __future__ import annotations


class DatanonymousError(Exception):
    """Base class. ``fix`` is a short, user-facing remedy (usually a flag)."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class UsageError(DatanonymousError):
    """Bad command line, bad config file, or refused directory layout (exit 2)."""


class ReadError(DatanonymousError):
    """The input file could not be parsed into a table."""


class NotTabular(DatanonymousError):
    """The file has a tabular extension but its content is not a table (e.g. a prose ``.txt``).

    The pipeline lists such a file as skipped (design D18: non-tabular files are
    never copied) instead of failing it.
    """


class NotImplementedFormat(DatanonymousError):
    """The file format is recognised but its reader/writer is not in this build."""


class UnsupportedColumnError(DatanonymousError):
    """A column cannot be detected, generated or written (design section 4)."""


class IdSpaceError(DatanonymousError):
    """Synthetic IDs of the same shape cannot avoid every real ID (design D9)."""


class GenerationError(DatanonymousError):
    """Generation failed for a reason other than the ID space."""


class WriteError(DatanonymousError):
    """The synthetic file could not be written in the source format."""


class SelfCheckError(DatanonymousError):
    """The written file does not have the source layout (strict mode)."""


class PrivacyGateError(DatanonymousError):
    """Remaining chance row matches exceed ``--max-row-matches``."""
