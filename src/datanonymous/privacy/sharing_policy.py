"""Settings and finding gates for packages intended to leave the machine.

Unknown warnings block release. A small, explicit utility allowlist keeps an
approved privacy transformation's distribution drift from being mistaken for
a successful privacy check.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..errors import UsageError
from ..model import Message
from ..model import ColumnType

POLICY_VERSION = "1"
#: id_width_changed: invented IDs had to take another digit count (never a real ID); dtype_mismatch: pandas reads a
#: column's output as another dtype (a blanked --null column, say). Both are counts and formats, not source values.
UTILITY_WARNINGS = frozenset({"fidelity_drift", "extremes_not_emitted", "panel_detected",
                              "id_width_changed", "dtype_mismatch"})


def validate_settings(settings: Any) -> None:
    """Reject bypasses consistently for the CLI, YAML and Python API."""
    mode = getattr(settings, "mode", "sharing")
    if mode not in {"sharing", "fidelity"}:
        raise UsageError("--mode must be sharing or fidelity")
    if mode == "fidelity":
        return
    unsafe: list[str] = []
    for field, flag in (
        ("force_factor", "--force-factor"), ("keep_name", "--keep-name"),
        ("keep_titles", "--keep-titles"), ("force", "--force"),
        ("no_strict", "--no-strict"), ("no_default_ids", "--no-default-ids"),
        ("org_code", "--org-code"),
    ):
        if getattr(settings, field, False):
            unsafe.append(flag)
    if settings.min_cell < 10:
        unsafe.append("--min-cell below 10")
    if settings.max_row_matches is not None:
        unsafe.append("--max-row-matches")
    safe_types = {ColumnType.IDENTIFIER, ColumnType.EMAIL, ColumnType.PERSON_NAME,
                  ColumnType.PHONE, ColumnType.SSN, ColumnType.STREET_ADDRESS,
                  ColumnType.IP, ColumnType.URL, ColumnType.FREE_TEXT}
    if any(ColumnType.parse(kind) not in safe_types for _, kind in getattr(settings, "force_type", ())):
        unsafe.append("--force-type that can copy source values")
    disabled = (getattr(settings, "structure", None) or {}).get("disable") or []
    if disabled:
        unsafe.append("structure.disable")
    if unsafe:
        raise UsageError(
            "sharing mode refuses privacy or validation bypasses: " + ", ".join(unsafe),
            fix="remove these options; use --mode fidelity only for invented data kept locally",
        )


def finding_category(message: Message) -> str:
    """Classify conservatively; new warning codes require an explicit review."""
    if message.level == "info":
        return "informational"
    if message.level == "warning" and message.code in UTILITY_WARNINGS:
        return "utility"
    return "blocking"


def all_findings(run: Any) -> Iterable[Message]:
    yield from run.messages
    for file in run.files:
        yield from file.all_messages()


def blocking_findings(run: Any) -> list[Message]:
    return [message for message in all_findings(run) if finding_category(message) == "blocking"]


__all__ = ["POLICY_VERSION", "validate_settings", "finding_category", "blocking_findings"]
