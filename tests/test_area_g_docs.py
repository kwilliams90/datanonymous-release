"""AREA G out-dir docs: README banner and preserved/not-preserved, data dictionary, types-only schema."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from conftest import EXAMPLES, read_dicts, run_cli
from test_area_g_grep import SCHOOL

STAT_KEYS = {"min", "max", "mean", "median", "sd", "levels", "missing_rate", "distinct", "length", "note",
             "na_token_counts", "rare_category_adjusted", "extreme_value_adjusted"}


def _walk_keys(obj) -> set[str]:  # noqa: ANN001
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _walk_keys(v)
    return keys


def test_readme_banner_generation_and_preserved_lists(example_run) -> None:
    readme = (example_run.out / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()
    assert lines[0] == "# SYNTHETIC DATA"
    assert "**SYNTHETIC DATA - NOT REAL RECORDS.**" in readme
    assert "seed 1" in readme and "`--seed 1`" in readme and "Faker 40.40.0" in readme
    for heading in ("## How it was generated", "## Files", "## What IS preserved", "## What is NOT preserved"):
        assert heading in readme
    not_preserved = readme.split("## What is NOT preserved")[1]
    for phrase in ("generated independently", "Correlations", "group differences", "cross-column logic",
                   "Panel and time structure", "not formal differential privacy"):
        assert phrase in not_preserved
    preserved = readme.split("## What IS preserved")[1].split("## What is NOT")[0]
    for phrase in ("row counts", "missing-value tokens", "constant within an ID", "blank together"):
        assert phrase in preserved
    assert "| `students_data.csv` | csv | 62 | 11 |" in readme
    assert "| `student-scores.csv` | csv | 2000 | 17 | ID id |" in readme
    assert "example-data" not in readme  # no input path in the shareable folder


def test_data_dictionary_columns_levels_and_snippets(example_run) -> None:
    dd = (example_run.out / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert dd.startswith("# Data dictionary (SYNTHETIC DATA)")
    assert "| # | Column (original header) | Type | R class | readr | pandas dtype | Format / NA tokens |" in dd
    assert "readr `col_types`: `fffffiii`" in dd
    assert "| `parental level of education` | ORDINAL (parental_education) | ordered factor | f | str |" in dd
    assert "readr::read_delim(\"StudentsPerformance.csv\"" in dd and "pd.read_csv('students_data.csv'" in dd
    assert "factor(x, levels = c(\"some high school\"" in dd
    scores = dd.split("## `student-scores.csv`")[1].split("## `")[0]
    first = next(line for line in scores.splitlines() if line.startswith("| 2 | `first_name`"))
    assert "length" in first and "'" not in first.split("|")[10]  # no levels for a non-label column
    source_count = Counter(row["career_aspiration"] for row in read_dicts(EXAMPLES / "student-scores.csv"))["Software Engineer"]
    assert source_count > 0
    assert f"'Software Engineer': {source_count}" in scores  # label-safe levels with exact synthetic counts
    assert "tokens `False` / `True`" in scores  # boolean format
    assert "| ID (row numbers 1..n kept) |" in scores.split("| 1 | `id`")[1].splitlines()[0] + "|"


def test_small_cells_are_shown_as_below_k(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    # 'Tardy' is a vocabulary word (D35: kept whatever its count), 'rare' is not (pseudonymised below --min-cell)
    rows = ["grp,score"] + [f"{'Tardy' if i < 3 else 'rare' if i < 6 else ('a' if i % 2 else 'b')},{i % 50}"
                            for i in range(200)]
    (d / "f.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "2")
    assert res.exit_code == 0, res.output
    dd = (tmp_path / "o" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "'Tardy': <10" in dd and "'a': 97" in dd and "'rare'" not in dd and "pseudonymized_levels (1)" in dd
    res = run_cli(d, "--out-dir", tmp_path / "o2", "--report-dir", tmp_path / "r2", "--seed", "2",
                  "--min-cell", "2")
    dd2 = (tmp_path / "o2" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "'Tardy': 3" in dd2 and "'rare': 3" in dd2


def test_schema_is_types_and_formats_only(example_run) -> None:
    schema = json.loads((example_run.out / "schema.json").read_text(encoding="utf-8"))
    assert schema["synthetic"] is True and schema["seed"] == 1
    assert not _walk_keys(schema) & STAT_KEYS
    files = {f["path"]: f for f in schema["files"]}
    sp = files["StudentsPerformance.csv"]
    assert sp["line_ending"] == "CRLF" and sp["delimiter"] == "," and sp["encoding"] == "utf-8"
    cols = {c["name"]: c for c in sp["tables"][0]["columns"]}
    assert cols["parental level of education"]["level_order"][0] == "some high school"
    assert cols["math score"] == {"name": "math score", "type": "INTEGER", "subtype": "discrete",
                                  "r_class": "integer", "readr": "i", "pandas_dtype": "int64", "na_tokens": [],
                                  "role": "row"}
    ss = files["student-scores.csv"]["tables"][0]["columns"]
    assert next(c for c in ss if c["name"] == "part_time_job")["boolean_tokens"] == ["False", "True"]


def test_docs_list_null_and_excluded_columns(tmp_path: Path) -> None:
    res = run_cli(EXAMPLES / "students_data.csv", "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r",
                  "--seed", "1", "--null", "Age", "--exclude", "Course")
    assert res.exit_code == 0
    readme = (tmp_path / "o" / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Columns not reproduced")[1]
    assert "`Course`" in section and "--exclude" in section and "`Age`" in section and "--null" in section
    schema = json.loads((tmp_path / "o" / "schema.json").read_text(encoding="utf-8"))
    names = [c["name"] for c in schema["files"][0]["tables"][0]["columns"]]
    assert "Course" not in names and "Age" in names
    age = next(c for c in schema["files"][0]["tables"][0]["columns"] if c["name"] == "Age")
    assert age["role"] == "null" and age["readr"] == "l"


def test_docs_mention_ids_zero_padding_and_shapes(tmp_path: Path) -> None:
    res = run_cli(SCHOOL, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "3")
    assert res.exit_code == 0, res.output
    schema = json.loads((tmp_path / "o" / "schema.json").read_text(encoding="utf-8"))
    people = next(f for f in schema["files"] if f["path"] == "people.csv")
    sid = next(c for c in people["tables"][0]["columns"] if c["name"] == "student_id")
    assert sid["type"] == "IDENTIFIER" and sid["zero_padded"] is True and sid["width"] == 6
    assert sid["shapes"] == ["999999"]
    readme = (tmp_path / "o" / "README.md").read_text(encoding="utf-8")
    assert "| `enrollments.csv` | csv | 362 | 6 | ID student_id |" in readme
    dd = (tmp_path / "o" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "`'NA'` (" in dd  # the written NA token of the score column, with its count


def test_doc_name_collision_keeps_prefixed_docs(tmp_path: Path) -> None:
    d = tmp_path / "in"
    d.mkdir()
    (d / "schema.json").write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]\n', encoding="utf-8")
    (d / "f.csv").write_text("g\n" + "a\nb\n" * 20, encoding="utf-8")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--report-dir", tmp_path / "r", "--seed", "1")
    names = sorted(p.name for p in (tmp_path / "o").iterdir())
    assert "README.md" in names and "DATA_DICTIONARY.md" in names
    if "_datanonymous_schema.json" in names:  # only when the JSON reader wrote schema.json
        assert json.loads((tmp_path / "o" / "_datanonymous_schema.json").read_text(encoding="utf-8"))["synthetic"]
    assert res.exit_code in (0, 3)
