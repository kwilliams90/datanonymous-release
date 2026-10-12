"""Report and out-dir docs never contain real non-label values (acceptance criterion 6, D21)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import EXAMPLES, read_dicts

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _doc_files(example_run) -> list[Path]:
    files = [p for p in example_run.rep.rglob("*") if p.is_file()]
    files += [example_run.out / n for n in ("README.md", "DATA_DICTIONARY.md", "schema.json")]
    return files


def test_no_real_names_or_emails_in_report_or_docs(example_run) -> None:
    real = read_dicts(EXAMPLES / "student-scores.csv")
    names = {r["first_name"] for r in real} | {r["last_name"] for r in real}
    emails = {r["email"] for r in real}
    local_parts = {e.split("@")[0] for e in emails}
    for path in _doc_files(example_run):
        text = path.read_text(encoding="utf-8")
        words = set(WORD.findall(text))
        hits = names & words
        assert not hits, (path.name, sorted(hits)[:10])
        # character-class masks such as 'aaaa.aaaaa.9@aaaa.aaa' are shapes, not values
        assert not any(e in text for e in emails), path.name
        assert not any(lp in text for lp in local_parts), path.name


def test_report_json_has_no_levels_for_non_label_columns(example_run) -> None:
    data = json.loads((example_run.rep / "files" / "student-scores.csv.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in data["tables"][0]["columns"]}
    for c in ("id", "first_name", "last_name", "email"):
        for side in ("real", "synthetic"):
            st = cols[c][side]
            assert "levels" not in st and "mode" not in st and "min" not in st, (c, side)
    # label-safe columns may show levels (they are reproduced in the output anyway)
    assert "levels" in cols["career_aspiration"]["real"]


def test_out_dir_contains_only_synthetic_files_and_docs(example_run) -> None:
    names = sorted(p.name for p in example_run.out.iterdir())
    assert names == sorted(["StudentsPerformance.csv", "student-scores.csv", "students_data.csv",
                            "README.md", "DATA_DICTIONARY.md", "schema.json"])
    readme = (example_run.out / "README.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in readme and "NOT preserved" in readme
