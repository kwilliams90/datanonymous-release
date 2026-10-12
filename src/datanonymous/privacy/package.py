"""Fresh, privately staged sharing packages with a closed file inventory."""

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
from .sharing_policy import POLICY_VERSION, blocking_findings


def check_fresh_destination(destination: Path) -> None:
    """An existing directory, dangling link or alias is never a release target."""
    if os.path.lexists(destination):
        raise UsageError("sharing mode requires an output path that does not yet exist",
                         fix="choose a fresh --out-dir outside the input and report directories")
    current = destination.parent
    while True:
        if is_link(current):
            raise UsageError("sharing output must not have a linked parent directory")
        if current == current.parent:
            break
        current = current.parent


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
        findings.append(Message("error", "sharing_inventory", "sharing package inventory or container verification failed",
                                fix="resolve incomplete, foreign, linked or unreadable artifacts and regenerate"))
        hashes = {}
    return findings, hashes


def finish(run: Any, destination: Path, stage: Path) -> dict[str, Any]:
    """Promote only a fully validated package; return a local evidence record."""
    expected = {file.out_rel_on_disk for file in run.files if file.written}
    expected.update(run.docs.values())
    blockers = blocking_findings(run)
    if run.any_failed or not run.files or any(not file.written for file in run.files):
        blockers.append(Message("error", "sharing_incomplete", "sharing requires every discovered table to succeed"))
    # Skips may conceal a sensitive input or an unsupported portion of a dataset.
    if run.skipped:
        blockers.append(Message("error", "sharing_skipped_inputs", "sharing requires an explicitly selected input with no skipped files",
                                fix="select a supported input set explicitly and regenerate"))
    findings, hashes = inventory(stage, expected)
    blockers.extend(findings)
    evidence: dict[str, Any] = {
        "mode": "sharing", "policy_version": POLICY_VERSION, "released": False,
        "inventory_sha256": hashes,
        "blocking_codes": sorted({message.code for message in blockers}),
    }
    if blockers:
        run.messages.extend(message for message in blockers if message not in run.messages)
        run.messages.append(Message("error", "sharing_package_blocked", "sharing package was withheld; review the local report and regenerate"))
        for file in run.files:
            file.fail("sharing package withheld because release gates did not pass")
        run.docs = {}
        return evidence
    check_fresh_destination(destination)
    if is_within(destination, stage) or is_within(stage, destination):
        raise UsageError("sharing staging and output must not overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reports and stage use the same filesystem as the destination in typical
    # runs. Cross-device promotion refuses rather than exposing a partial copy.
    os.rename(stage, destination)
    evidence["released"] = True
    return evidence


__all__ = ["check_fresh_destination", "create_staging", "inventory", "finish"]
