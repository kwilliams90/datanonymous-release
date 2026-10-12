"""Architecture contracts: module docstrings, area hooks, import boundaries (acceptance criterion 7)."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import datanonymous
from datanonymous import pipeline

SRC = Path(datanonymous.__file__).resolve().parent

#: (module, function, area) for every documented extension hook.
HOOKS = [
    ("datanonymous.detect.dates", "detect_dates", "A"),
    ("datanonymous.detect.dates", "date_stats", "A"),
    ("datanonymous.detect.dates", "date_numbers", "A"),
    ("datanonymous.detect.dates", "doc_fields", "A"),
    ("datanonymous.generate.dates", "generate_dates", "A"),
    ("datanonymous.generate.dates", "render_dates", "A"),
    ("datanonymous.detect.numeric", "detect_numeric_extended", "B"),
    ("datanonymous.detect.numeric", "numeric_stats_extended", "B"),
    ("datanonymous.detect.numeric", "numbers_extended", "B"),
    ("datanonymous.detect.numeric", "doc_fields", "B"),
    ("datanonymous.generate.numeric", "generate_numeric_extended", "B"),
    ("datanonymous.generate.numeric", "render_numeric_extended", "B"),
    ("datanonymous.detect.text", "detect_text_semantics", "C"),
    ("datanonymous.detect.text", "apply_pii_precedence", "C"),
    ("datanonymous.detect.text", "classify_overflow_text", "C"),
    ("datanonymous.detect.text", "pii_subtype_detection", "C"),
    ("datanonymous.detect.text", "text_stats", "C"),
    ("datanonymous.detect.text", "doc_fields", "C"),
    ("datanonymous.generate.pii", "generate_pii", "C"),
    ("datanonymous.generate.text", "generate_text_extended", "C"),
    ("datanonymous.detect.ids", "id_candidate_reason", "D"),
    ("datanonymous.profile.entity", "id_namespace", "D"),
    ("datanonymous.profile.entity", "find_aliases_and_nesting", "D"),
    ("datanonymous.profile.structure", "find_functional_dependencies", "D"),
    ("datanonymous.profile.structure", "find_derived_templates", "D"),
    ("datanonymous.profile.structure", "find_composite_keys", "D"),
    ("datanonymous.profile.structure", "suggest_inequality_rules", "D"),
    ("datanonymous.profile.structure", "detect_panels", "D"),
    ("datanonymous.profile.structure", "swap_groups", "D"),
    ("datanonymous.profile.structure", "cross_file_structure", "D"),
    ("datanonymous.profile.structure", "apply_structure", "D"),
    ("datanonymous.io.excelio", "read_xlsx", "E"),
    ("datanonymous.io.excelio", "read_xls", "E"),
    ("datanonymous.io.excelio", "write_xlsx", "E"),
    ("datanonymous.io.excelio", "read_html_table", "E"),
    ("datanonymous.io.excelio", "write_html_table", "E"),
    ("datanonymous.io.excelio", "detect_native", "E"),
    ("datanonymous.io.excelio", "layout_diff", "E"),
    ("datanonymous.io.excelio", "inferred_dtypes", "E"),
    ("datanonymous.io.excelio", "inferred_dtypes_html", "E"),
    ("datanonymous.io.arrowio", "read_parquet", "F"),
    ("datanonymous.io.arrowio", "write_parquet", "F"),
    ("datanonymous.io.arrowio", "read_feather", "F"),
    ("datanonymous.io.arrowio", "write_feather", "F"),
    ("datanonymous.io.arrowio", "inferred_dtypes", "F"),
    ("datanonymous.io.statio", "inferred_dtypes", "F"),
    ("datanonymous.io.jsonio", "inferred_dtypes", "F"),
    ("datanonymous.io.statio", "read_dta", "F"),
    ("datanonymous.io.statio", "write_dta", "F"),
    ("datanonymous.io.statio", "read_sas", "F"),
    ("datanonymous.io.jsonio", "read_json", "F"),
    ("datanonymous.io.jsonio", "write_jsonl", "F"),
    ("datanonymous.io.compress", "read_bytes", "F"),
    ("datanonymous.io.compress", "write_bytes", "F"),
    ("datanonymous.privacy.leaks", "scan_outputs", "G"),
    ("datanonymous.report.report", "build_report", "G"),
    ("datanonymous.report.report", "write_report_files", "G"),
]


def all_modules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages([str(SRC)], prefix="datanonymous.")]


def test_every_module_has_a_docstring() -> None:
    for name in all_modules() + ["datanonymous"]:
        mod = importlib.import_module(name)
        assert (mod.__doc__ or "").strip(), name


def test_hooks_exist_documented_and_listed() -> None:
    doc = pipeline.__doc__ or ""
    for module, func, area in HOOKS:
        fn = getattr(importlib.import_module(module), func)
        assert callable(fn), (module, func)
        assert (fn.__doc__ or "").strip(), (module, func)
        assert func in doc, f"{func} missing from pipeline 'Extension points'"
        assert inspect.signature(fn) is not None


def test_generate_never_imports_io() -> None:
    for path in (SRC / "generate").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not (node.level >= 2 and mod.split(".")[0] == "io"), (path.name, mod)
                assert not mod.startswith("datanonymous.io"), (path.name, mod)
                if node.level >= 2 and mod == "":
                    assert all(a.name != "io" for a in node.names), path.name
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("datanonymous.io") for a in node.names), path.name


def test_registry_covers_v1_formats() -> None:
    from datanonymous.io import registry

    assert {"csv", "xlsx", "xls", "html_xls", "parquet", "feather", "dta", "sas", "json", "jsonl"} <= set(
        registry.FORMATS)
    for ext in (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".feather", ".dta", ".sas7bdat", ".xpt",
                ".json", ".jsonl"):
        assert ext in registry.EXTENSION_FORMATS


# --------------------------------------------------------------------------- contracts for the area agents
def test_hook_signatures() -> None:
    from datanonymous.detect import dates, numeric, text
    from datanonymous.io import arrowio, csvio, excelio, jsonio, statio
    from datanonymous.privacy import leaks

    assert list(inspect.signature(numeric.numeric_stats_extended).parameters) == ["values", "profile"]
    assert list(inspect.signature(dates.date_numbers).parameters) == ["values", "profile"]
    assert list(inspect.signature(numeric.numbers_extended).parameters) == ["values", "profile"]
    for mod in (dates, numeric, text):
        assert list(inspect.signature(mod.doc_fields).parameters) == ["profile", "syn_stats"]
    for mod in (csvio, excelio, arrowio, statio, jsonio):
        assert list(inspect.signature(mod.layout_diff).parameters) == [
            "source", "output", "source_tables", "output_tables"], mod.__name__
    assert list(inspect.signature(leaks.scan_outputs).parameters) == ["run_result", "report_payload"]


def test_output_format_handlers() -> None:
    from datanonymous.io import registry

    assert registry.output_handler("xls").key == "xlsx"
    assert registry.output_handler("sas").key == "parquet"
    assert registry.output_handler("csv").key == "csv"
    for key in ("parquet", "feather", "dta", "sas", "json", "jsonl", "html_xls"):
        assert registry.FORMATS[key].inferred_dtypes is not registry._no_dtypes, key


def test_layout_private_extra_keys_are_not_public() -> None:
    from datanonymous.model import FileLayout

    lay = FileLayout(format="xlsx", extra={"_title_rows": ["Printed by jsmith"], "sheets": ["S1"]})
    assert lay.to_public()["extra"] == {"sheets": ["S1"]}


def test_selfcheck_passes_table_layouts_to_layout_diff() -> None:
    from datanonymous.model import FileLayout, RawTable, SyntheticTable, TableProfile
    from datanonymous.report import selfcheck

    calls = []
    tp = TableProfile(name="S1", header=["a"], n_rows=0, columns=[], layout={"header_row": 3})
    st = SyntheticTable("S1", ["a"], 0, [], tp)
    out = [RawTable("S1", ["a"], [], 0)]
    selfcheck.compare(FileLayout("xlsx"), FileLayout("xlsx"), [st], out,
                      lambda *a: calls.append(a) or [])
    assert calls and calls[0][2] == [{"header_row": 3}] and calls[0][3] is out


def test_dtype_message_when_output_unreadable() -> None:
    from datanonymous.report import selfcheck

    assert selfcheck.dtype_messages(["int64"], None, ["a"])
    assert not selfcheck.dtype_messages(None, None, ["a"])


def test_doc_fields_hook_and_date_readr_defaults(monkeypatch) -> None:
    from datanonymous.detect import dates
    from datanonymous.model import ColumnProfile, ColumnType
    from datanonymous.report import docs

    p = ColumnProfile(index=0, name="d", type=ColumnType.DATE)
    assert docs.column_entry(p, {}, None, 10)["readr"] == "c"  # safe default for any date format
    p_int = ColumnProfile(index=0, name="d", type=ColumnType.DATE_AS_INT)
    assert docs.column_entry(p_int, {}, None, 10)["readr"] == "i"
    monkeypatch.setattr(dates, "doc_fields", lambda profile, syn: {"format": "%Y-%m-%d", "readr": "D"})
    e = docs.column_entry(p, {}, None, 10)
    assert e["format"] == "%Y-%m-%d" and e["readr"] == "D"


def test_column_profile_native_is_private() -> None:
    from datanonymous.model import ColumnProfile, ColumnType

    p = ColumnProfile(index=0, name="x", type=ColumnType.FACTOR, native={"number_format": "@"})
    assert "native" not in p.to_public()
