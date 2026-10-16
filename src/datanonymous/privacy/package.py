"""Fresh, privately staged sharing packages with a closed file inventory.

By default a file that cannot be written safely is left out on its own (its output removed from the stage, the
docs written without it) and every other file is written to the output folder; only a run-level problem
withholds everything. ``all_or_nothing`` writes nothing unless every input file was written and none skipped.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile
from xml.etree import ElementTree

from ..errors import UsageError
from ..model import Message
from ..paths import is_link, is_within
from .sharing_policy import POLICY_VERSION, blocking_findings, finding_category


def check_fresh_destination(destination: Path) -> None:
    """The out dir is new or an empty real folder; a file, link or folder with files in it is never a target."""
    if os.path.lexists(destination):
        if is_link(destination) or not destination.is_dir():
            raise UsageError(f"--out-dir {destination} exists and is not a plain folder; "
                             "sharing mode needs a new or empty folder")
        if any(destination.iterdir()):
            raise UsageError(f"--out-dir {destination} already contains files; sharing mode needs a new or empty "
                             "folder (clear it or choose another)")
    current = destination.parent
    while True:
        if is_link(current):
            raise UsageError("sharing output must not have a linked parent directory")
        if current == current.parent:
            break
        current = current.parent


def _blocking(messages: Any) -> list[Message]:
    return [message for message in messages if finding_category(message) == "blocking"]


def _remove_staged(stage: Path, rel: str) -> None:
    """Remove a staged output and the folders it leaves empty (an empty folder would fail the inventory)."""
    path = stage / Path(*rel.split("/"))
    if os.path.lexists(path) and not path.is_dir():
        os.unlink(path)
    parent = path.parent
    while parent != stage and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def withhold_blocked(run: Any, stage: Path, written: list[Any]) -> list[Any]:
    """Leave out every file whose own findings block sharing; returns the ``written`` entries still written.

    A FAILed file's output (or the empty folders its generation left) is removed from the stage too.
    """
    for file in run.files:
        codes = sorted({message.code for message in _blocking(file.all_messages())})
        if file.written and codes:
            file.fail(f"not written to the output folder: its findings block sharing ({', '.join(codes)})",
                      "see this file's warnings; fix the input or its column options and run again")
    for file in run.files:
        if file.failed:
            _remove_staged(stage, file.out_rel_on_disk)
    return [w for w in written if w.result.written]


def _unread(run: Any) -> list[str]:
    """Input files that FAILed before any column was read: a column flag or rule may have been meant for them."""
    return [file.rel_path for file in run.files if file.failed and not file.tables]


def _scopes(settings: Any, code: str) -> list[tuple[str, str | None]]:
    """(text quoted in a ``code`` message, the input file it is limited to or None) for each option of the run."""
    if code == "rule_unused":
        from ..profile.structure import parse_rules

        return [(rule["text"], rule["file"]) for rule in parse_rules(getattr(settings, "rules", None))]
    from ..config import _KV_COL, _LIST_COL

    refs = [ref for name in _LIST_COL for ref in getattr(settings, name, ())]
    refs += [ref for name in _KV_COL for ref, _ in getattr(settings, name, ())]
    return [(ref.raw, ref.file.replace("\\", "/").strip("/") if ref.file else None) for ref in refs]


def _may_name_unread(message: Message, run: Any, settings: Any) -> bool:
    """The unused column option or rule of ``message`` may have been meant for a file that was never read."""
    unread = _unread(run)
    if not unread:
        return False
    if settings is None:
        return True
    for text, file in _scopes(settings, message.code):
        if repr(text) in message.text:
            return file is None or file in unread
    return True


def _blocks_partial(message: Message, run: Any, settings: Any) -> bool:
    """A run-level finding that still withholds the whole output when failed files are only left out."""
    if message.code == "leak_found":
        return False  # a summary: the files holding the values FAILed and were removed
    if message.code == "leak_scan_incomplete" and "every other file was verified" in message.text:
        return False  # the unverified files FAILed and were removed; a scan that did not complete still blocks
    if message.code in ("unused_column_flag", "rule_unused"):
        return not _may_name_unread(message, run, settings)
    return True


def create_staging(report_dir: Path) -> Path:
    """Keep incomplete files with local diagnostics, outside the release folder."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sharing-staging-", dir=report_dir))
    os.chmod(stage, 0o700)
    return stage


def _xlsx_inventory(path: Path) -> None:
    """Refuse active/external parts; parse every XML part, including properties."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise ValueError("duplicate workbook members")
        for member in members:
            name = member.filename
            parts = name.replace("\\", "/").split("/")
            if name.startswith(("/", "\\")) or ".." in parts or ":" in parts[0]:
                raise ValueError("unsafe workbook member path")
            lowered = name.casefold()
            if any(token in lowered for token in ("vbaproject", "externallinks/", "embeddings/", "customxml/")):
                raise ValueError("unapproved workbook part")
            contents = archive.read(member)  # CRC and truncation must be checked for every member.
            if lowered.endswith((".xml", ".rels")):
                tree = ElementTree.fromstring(contents)
                if lowered.endswith(".rels"):
                    for element in tree.iter():
                        if element.attrib.get("TargetMode", "").casefold() == "external":
                            raise ValueError("external workbook relationship")


def inventory(stage: Path, expected: set[str]) -> tuple[list[Message], dict[str, str]]:
    """Read every included byte and reject foreign entries and all links."""
    findings: list[Message] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    stack = [stage]
    try:
        if is_link(stage):
            raise ValueError("linked staging directory")
        while stack:
            folder = stack.pop()
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    rel = path.relative_to(stage).as_posix()
                    if entry.is_symlink() or entry.is_junction() or is_link(path):
                        raise ValueError("linked package entry")
                    if entry.is_dir(follow_symlinks=False):
                        if not any(name.startswith(rel + "/") for name in expected):
                            raise ValueError("foreign package directory")
                        stack.append(path)
                        continue
                    if not entry.is_file(follow_symlinks=False) or rel not in expected:
                        raise ValueError("foreign package entry")
                    stat = path.stat()
                    if stat.st_nlink != 1:
                        raise ValueError("hard-linked package entry")
                    digest = hashlib.sha256()
                    signature = b""
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            if not signature:
                                signature = chunk[:4]
                            digest.update(chunk)
                    after = path.stat()
                    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
                    ):
                        raise ValueError("package changed during verification")
                    if path.suffix.casefold() == ".xlsx" or signature == b"PK\x03\x04":
                        _xlsx_inventory(path)
                    hashes[rel] = digest.hexdigest()
                    seen.add(rel)
        if seen != expected:
            raise ValueError("incomplete package inventory")
    except (OSError, ValueError, RuntimeError, NotImplementedError, zipfile.BadZipFile, ElementTree.ParseError):
        # Exception text can contain a source-derived filename or container text.
        findings.append(Message("error", "sharing_inventory", "the contents of the output folder could not be verified, so nothing was written to it",
                                fix="resolve incomplete, foreign, linked or unreadable artifacts and regenerate"))
        hashes = {}
    return findings, hashes


def finish(run: Any, destination: Path, stage: Path, all_or_nothing: bool = False,
           settings: Any = None) -> dict[str, Any]:
    """Move the validated output into ``destination``; return a local evidence record."""
    expected = {file.out_rel_on_disk for file in run.files if file.written}
    expected.update(run.docs.values())
    if all_or_nothing:
        blockers = blocking_findings(run)
        if run.any_failed or not run.files or any(not file.written for file in run.files):
            blockers.append(Message("error", "sharing_incomplete",
                                    "--all-or-nothing: an input file was not written, so none is"))
        # Skips may conceal a sensitive input or an unsupported portion of a dataset.
        if run.skipped:
            blockers.append(Message("error", "sharing_skipped_inputs",
                                    "--all-or-nothing: an input file was skipped, so none is written",
                                    fix="select a supported input set explicitly and regenerate"))
    else:
        blockers = [message for message in _blocking(run.messages) if _blocks_partial(message, run, settings)]
        # every written file was cleared by withhold_blocked; a finding added since withholds everything
        blockers += _blocking(message for file in run.files if file.written for message in file.all_messages())
        if not any(file.written for file in run.files):
            blockers.append(Message("error", "sharing_nothing_written",
                                    "no input file could be written to the output folder"))
    findings, hashes = inventory(stage, expected)
    blockers.extend(findings)
    evidence: dict[str, Any] = {
        "mode": "sharing", "policy_version": POLICY_VERSION, "released": False,
        "inventory_sha256": hashes,
        "blocking_codes": sorted({message.code for message in blockers}),
    }
    if blockers:
        run.messages.extend(message for message in blockers if message not in run.messages)
        run.messages.append(Message("error", "sharing_package_blocked",
                                    "nothing was written to the output folder; see the report's failures and run again"))
        for file in run.files:
            if not file.failed:
                file.fail("not written to the output folder: a run-level problem withheld every file")
        run.docs = {}
        return evidence
    check_fresh_destination(destination)
    if is_within(destination, stage) or is_within(stage, destination):
        raise UsageError("sharing staging and output must not overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    emptied = os.path.lexists(destination)
    if emptied:
        os.rmdir(destination)  # checked empty just above; rmdir refuses anything else
    # Reports and stage use the same filesystem as the destination in typical
    # runs. Cross-device promotion refuses rather than exposing a partial copy.
    try:
        os.rename(stage, destination)
    except OSError:
        if emptied:
            destination.mkdir(exist_ok=True)  # the user's empty folder is left as it was
        raise
    evidence["released"] = True
    return evidence


__all__ = ["check_fresh_destination", "create_staging", "withhold_blocked", "inventory", "finish"]
