"""ADR-0004 at the run seam: rare ID format classes are written in the column's most common held format class."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from datanonymous.config import Settings
from datanonymous.pipeline import execute, exit_code
from datanonymous.privacy.sharing_policy import blocking_findings

from release_corpus import read_records

HELD = re.compile(r"S\d{6}")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _layout(value: str) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", value))


def _mixed_roster_and_services(original: Path) -> tuple[list[str], list[str]]:
    """Invented: a roster whose Student_ID mostly reads S000000, with a year-number minority and one odd layout,
    and a services table that refers to every minority holder and to 40 others."""
    original.mkdir()
    people = [f"S{410003 + i * 37:06d}" for i in range(100)] + ["2024-0045", "2024-0187", "XX-901234"]
    _write_csv(original / "roster.csv", [
        {"Student_ID": sid, "Grade": str(6 + i % 4), "School": ("North", "South")[i % 2],
         "Score": f"{(5500 + (i * 71) % 3500) / 100:.2f}"} for i, sid in enumerate(people)])
    served = people[:40] + people[100:]
    _write_csv(original / "services.csv", [
        {"Student_ID": sid, "Frequency": ("Weekly", "Monthly")[(i + visit) % 2],
         "Minutes": f"{(1500 + (i * 137 + visit * 59) % 9000) / 100:.2f}"}
        for i, sid in enumerate(served) for visit in range(2)])
    return people, served


def test_a_mixed_layout_id_column_is_written_in_the_held_layout_and_its_join_holds(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    people, served = _mixed_roster_and_services(original)
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    assert not blocking_findings(run), [(m.code, m.text) for m in blocking_findings(run)]
    assert run.package["released"]
    assert exit_code(run) == 0
    roster, services = read_records(output / "roster.csv"), read_records(output / "services.csv")
    assert len(roster) == len(people) and len(services) == 2 * len(served)
    roster_ids = [r["Student_ID"] for r in roster]
    service_ids = [r["Student_ID"] for r in services]
    # every ID is in the held layout: no rare source format class appears in either file
    assert all(HELD.fullmatch(v) for v in roster_ids + service_ids), Counter(map(_layout, roster_ids + service_ids))
    assert len(set(roster_ids)) == len(people)
    # the join holds: services refer to fictional roster people, as many as in the source, never a real ID
    assert set(service_ids) <= set(roster_ids)
    assert len(set(service_ids)) == len(served)
    assert not set(roster_ids + service_ids) & set(people)
    # the operator is told, in plain words, on each file's column
    for name in ("roster.csv", "services.csv"):
        text = (report / "files" / f"{name}.md").read_text(encoding="utf-8")
        assert "IDs in a rare format were written in the column's most common format" in text, name


def test_an_id_column_with_no_held_format_class_is_still_refused_by_name(tmp_path):
    original, output, report = tmp_path / "original", tmp_path / "synthetic", tmp_path / "local-report"
    original.mkdir()
    # twelve layouts of at most nine IDs each: no format class is held by --min-cell (10) people
    ids = [f"{'S' * (1 + i % 12)}{410003 + i * 37:06d}" for i in range(100)]
    _write_csv(original / "roster.csv", [
        {"Student_ID": sid, "Grade": str(6 + i % 4), "Score": f"{(5500 + (i * 71) % 3500) / 100:.2f}"}
        for i, sid in enumerate(ids)])
    run = execute(Settings(mode="sharing", seed=573291), original, output, report)
    roster = next(f for f in run.files if f.rel_path == "roster.csv")
    assert not roster.written
    assert any("column 'Student_ID'" in r and "format class" in r for r in roster.reasons), roster.reasons
    assert not (output / "roster.csv").exists()
