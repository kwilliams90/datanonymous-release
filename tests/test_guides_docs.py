"""The published guides and README describe the sharing usability behaviour (issue 20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datanonymous.cli import main
from datanonymous.config import _SCALARS

REPO = Path(__file__).resolve().parents[1]
GUIDES = REPO / "guides"
#: options every operator-facing option list needs not name (they only change how much is printed)
_UNLISTED = {"--verbose", "--quiet"}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_run_option_is_in_the_option_reference() -> None:
    reference = _text(GUIDES / "fidelity-reference.md")
    options = {o for p in main.commands["run"].params for o in p.opts if o.startswith("--")}
    assert "--refuse-unsupported" in options
    assert sorted(o for o in options - _UNLISTED if o not in reference) == []


@pytest.mark.parametrize("name", ["output_root", "report_root", "refuse_unsupported"])
def test_new_config_keys_are_documented(name: str) -> None:
    assert name in _SCALARS
    for doc in (REPO / "README.md", GUIDES / "fidelity-reference.md", GUIDES / "release-notes.md"):
        assert f"`{name}`" in _text(doc), (doc.name, name)


def test_readme_and_operations_describe_the_user_config_and_private_report() -> None:
    for doc in (REPO / "README.md", GUIDES / "operations.md"):
        text = _text(doc)
        for phrase in ("DATANONYMOUS_CONFIG_DIR", "PRIVATE-DO-NOT-SHARE.txt", "--refuse-unsupported"
                       if doc.name == "README.md" else "What needs attention"):
            assert phrase in text, (doc.name, phrase)
    operations = _text(GUIDES / "operations.md")
    for heading in ("What was generated", "What changed", "What needs attention"):
        assert heading in operations


def test_release_notes_cover_every_user_visible_change_of_rc3() -> None:
    notes = " ".join(_text(GUIDES / "release-notes.md").split("# Release candidate 0.1.0rc2")[0].split())
    for phrase in ("Nulled column", "--refuse-unsupported", "rebalanced", "ID format class", "text shape",
                   "date format", "Currency", "What changed", "output_root", "report_root", "config.yaml",
                   "PRIVATE-DO-NOT-SHARE.txt", "message catalogue"):
        assert phrase in notes, phrase


def test_sharing_policy_no_longer_says_what_the_adrs_replaced() -> None:
    policy = _text(GUIDES / "sharing-policy.md")
    for stale in ("or require refusal", "the count must be blurred or the table refused",
                  "(a rare rows-per-person count class, a rare identifier format, an infeasible join)"):
        assert stale not in policy, stale
    for current in ("ADR-0001", "ADR-0002", "ADR-0003", "ADR-0004", "`--refuse-unsupported`", "Nulled column"):
        assert current in policy, current
    assert "Rare count or identifier-format" not in _text(GUIDES / "limitations.md")
    assert "refuses rare count or identifier-format classes" not in _text(GUIDES / "limitations.md")
