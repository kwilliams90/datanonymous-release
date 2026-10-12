"""Sharing documentation excludes reproduction details and absent source vocabulary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datanonymous.detect import text
from datanonymous.model import (
    ColumnModel,
    ColumnProfile,
    ColumnType,
    FileLayout,
    FileResult,
    SyntheticColumn,
    SyntheticTable,
    TableProfile,
    TextDialect,
)
from datanonymous.report import docs


def _written() -> docs.WrittenFile:
    columns = [
        ColumnProfile(0, "student_id", ColumnType.IDENTIFIER, role="id",
                      params={"_identity_mapped": True}),
        ColumnProfile(1, "program", ColumnType.ORDINAL,
                      params={"levels": ["source_only_label", "Common", "Invented"]},
                      stats={"levels": {"source_only_label": 1, "Common": 23}}),
    ]
    profile = TableProfile(None, [p.name for p in columns], 24, columns)
    stats = {
        0: {"missing_rate": 0.0, "distinct": 24},
        1: {"missing_rate": 0.0, "distinct": 2, "levels": {"Common": 21, "Invented": 3}},
    }
    result = FileResult("private_source_path.csv", "events.csv", "csv", written=True,
                        layout=FileLayout("csv", encoding="utf-8", text=TextDialect(delimiter=",")),
                        tables=[profile], synthetic_stats=[stats], out_dtypes=[["str", "str"]])
    synthetic = SyntheticTable(None, profile.header, 24, [
        SyntheticColumn(0, "student_id", np.arange(1, 25).astype(str)),
        SyntheticColumn(1, "program", np.array(["Common"] * 21 + ["Invented"] * 3)),
    ], profile)
    return docs.WrittenFile(result, [synthetic], [stats], [["str", "str"]])


def _contents(out: Path) -> tuple[str, str, dict]:
    return ((out / "README.md").read_text(encoding="utf-8"),
            (out / "DATA_DICTIONARY.md").read_text(encoding="utf-8"),
            json.loads((out / "schema.json").read_text(encoding="utf-8")))


def test_sharing_docs_omit_seed_source_labels_and_identity_mapping(tmp_path: Path) -> None:
    docs.write_docs(tmp_path, [_written()], [], 919283, 10, mode="sharing")
    readme, dictionary, schema = _contents(tmp_path)
    all_docs = readme + dictionary + json.dumps(schema)
    assert "SHARING MODE" in readme and "SHARING MODE" in dictionary
    assert schema["data_mode"] == "sharing" and "seed" not in schema
    assert "919283" not in all_docs and "--seed" not in all_docs
    assert "private_source_path" not in all_docs and "source_only_label" not in all_docs
    exported = schema["files"][0]["tables"][0]
    assert exported["rows"] == 24
    assert [c["name"] for c in exported["columns"]] == ["student_id", "program"]
    assert "row_numbers_kept" not in exported["columns"][0]
    assert exported["columns"][1]["level_order"] == ["Common", "Invented"]
    assert "'Common': 21" in dictionary and "'Invented': <10" in dictionary
    assert "exact level counts" not in readme and "same real ID gets" not in readme
    for phrase in ("memberships", "relationships", "valid joins", "N output equals N input",
                   "privacy-transformed", "not automatic FERPA approval", "Source reports"):
        assert phrase in readme + dictionary


def test_sharing_order_without_any_written_source_level_is_not_exported(tmp_path: Path) -> None:
    written = _written()
    written.tables[0].profile.columns[1].params["levels"] = ["absent_original"]
    docs.write_docs(tmp_path, [written], [], 3, 10, mode="sharing")
    readme, dictionary, schema = _contents(tmp_path)
    column = schema["files"][0]["tables"][0]["columns"][1]
    assert "level_order" not in column
    assert "absent_original" not in readme + dictionary + json.dumps(schema)
    assert "'Common': 21" in dictionary


def test_sharing_excludes_unwritten_pseudonyms_and_source_pseudonym_count(tmp_path: Path) -> None:
    written = _written()
    column = written.tables[0].profile.columns[1]
    placeholder = "[pseudonymized 1]"
    column.params.update({"levels": ["Common", placeholder], text.PSEUDO_COUNT_KEY: 7,
                          text.PSEUDO_KEY: [["private_label", placeholder], ["PRIVATE_LABEL", placeholder]]})
    column.model = ColumnModel(extra={text.PSEUDO_MAP_KEY: {"private_label": "Invented", "PRIVATE_LABEL": "Unwritten"},
                                      text.PSEUDO_WRITTEN_KEY: {"Invented": 3, "Unwritten": 0}})
    written.synthetic_stats[0][1]["levels"] = {"Common": 21, placeholder: 3}
    docs.write_docs(tmp_path, [written], [], 3, 10, mode="sharing")
    readme, dictionary, schema = _contents(tmp_path)
    assert "Unwritten" not in dictionary + json.dumps(schema)
    assert "private_label" not in readme + dictionary + json.dumps(schema)
    assert "pseudonymized_levels (1)" in dictionary
    assert "pseudonymized_levels (7)" not in dictionary
    assert "Invented" in dictionary


@pytest.mark.parametrize("data_mode", [None, "fidelity", "sharing"])
def test_rewrite_docs_preserves_mode_and_old_run_compatibility(tmp_path: Path, data_mode: str | None) -> None:
    written = _written()
    run = SimpleNamespace(out_dir=tmp_path, files=[written.result], seed=53, docs={}, messages=[])
    if data_mode is not None:
        run.data_mode = data_mode
    run.docs = docs.rewrite_docs(run, 10)
    readme, dictionary, schema = _contents(tmp_path)
    assert set(run.docs) == {"readme", "data_dictionary", "schema"}
    if data_mode == "sharing":
        assert "SHARING MODE" in readme and "seed" not in schema
        assert "source_only_label" not in dictionary + json.dumps(schema)
    else:
        assert "FIDELITY MODE" in readme and "FIDELITY MODE" in dictionary
        assert schema["seed"] == 53
        assert "source_only_label" in json.dumps(schema)


def test_write_docs_default_remains_fidelity_for_existing_callers(tmp_path: Path) -> None:
    docs.write_docs(tmp_path, [_written()], [], 42, 10)
    readme, dictionary, schema = _contents(tmp_path)
    assert "FIDELITY MODE" in readme and "invented-data and local testing" in readme
    assert "not an approved sharing package" in dictionary
    assert schema["seed"] == 42 and "data_mode" not in schema
    assert "--seed 42" in readme
