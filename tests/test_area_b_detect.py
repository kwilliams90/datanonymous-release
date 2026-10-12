"""AREA B detection: formatted numbers, numbers with tokens, scientific notation, decimals (design 3 #15-#18, 4)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import numeric as N
from datanonymous.detect.context import DetectContext
from datanonymous.detect.types import detect_column
from datanonymous.errors import UnsupportedColumnError
from datanonymous.model import ColumnType as T
from datanonymous.model import RawColumn, RawTable
from datanonymous.profile.stats import column_stats
from datanonymous.profile.table import profile_table
from datanonymous.report import docs

EURO, POUND = "\u20ac", "\u00a3"


def _arr(values) -> np.ndarray:
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def ctx_for(values, name="col", fmt="csv", native=None, **cli) -> DetectContext:
    s = build_settings(dict(cli))
    return DetectContext(index=0, name=name, values=_arr(values), file_rel="f.csv", table=None, format=fmt,
                         options=s.column_options("f.csv", None, name), settings=s, native=native)


def detect(values, name="col", fmt="csv", native=None, **cli):
    return detect_column(ctx_for(values, name, fmt, native, **cli))


def profile(values, name="col", fmt="csv", native=None, **cli):
    s = build_settings(dict(cli))
    raw = RawTable(None, [name], [RawColumn(name, _arr(values), native)], len(values))
    return profile_table(raw, "f.csv", fmt, s).columns[0]


# --------------------------------------------------------------------------- parser
@pytest.mark.parametrize("text,value,skeleton", [
    ("$1,234.50", 1234.5, "$#"),
    ("($25.00)", -25.0, "($#)"),
    ("(1,234.50)", -1234.5, "(#)"),
    ("-$12.50", -12.5, "-$#"),
    ("$-12.50", -12.5, "$-#"),
    ("$(5.00)", -5.0, "$(#)"),
    ("-1,234.5", -1234.5, "-#"),
    ("93.5%", 93.5, "#%"),
    ("-3.2 %", -3.2, "-# %"),
    ("$ 12.50", 12.5, "$ #"),
    ("EUR 1,234", 1234.0, "EUR #"),
    (f"1,234.50 {EURO}", 1234.5, f"# {EURO}"),
    (f"{POUND}12,345", 12345.0, f"{POUND}#"),
    ("+5", 5.0, "+#"),
    (".5", 0.5, "#"),
    ("( 5.00 )", -5.0, "( # )"),
])
def test_parse_number_forms(text, value, skeleton) -> None:
    p = N.parse_number(text)
    assert p is not None and p.value == value and p.skeleton == skeleton
    assert p.negative == (value < 0)


@pytest.mark.parametrize("text", [
    "5.", "1,23", "12,5", "1.234,50", "$5%", f"$5{EURO}", "(-5)", "((5))", "(5", "5)", "1e5", "<10", ">95%",
    "*", "5-", "1-2", "$", "abc", "#5", "1,2345", "5%%",
])
def test_parse_number_rejects(text) -> None:
    assert N.parse_number(text) is None


def test_parse_number_decimal_comma_mark() -> None:
    p = N.parse_number("1.234,50", ",")
    assert p.value == 1234.5 and p.grouped and p.decimals == 2 and p.feature
    assert N.parse_number("12,5%", ",").value == 12.5
    assert N.parse_number(f"1.234,50 {EURO}", ",").skeleton == f"# {EURO}"
    assert N.parse_number("3.5", ",") is None  # '.' is a thousands separator under ','
    assert N.parse_number("1.234", ",").value == 1234.0
    assert N.parse_number("007").code and not N.parse_number("0.07").code


# --------------------------------------------------------------------------- NUMERIC_FORMATTED
def test_currency_with_parenthesised_negatives() -> None:
    d = detect(["$1,234.50", "($25.00)", "$3.00", "$999.99", "($1,000.00)"])
    assert d.type is T.NUMERIC_FORMATTED
    f = d.params["format"]
    assert (f["positive"], f["negative"], f["group_sep"], f["decimals"], f["fixed_decimals"]) == (
        "$#", "($#)", ",", 2, True)
    assert d.params["template"] == "$#,##0.00;($#,##0.00)"
    assert d.params["negative_style"] == "($x)"
    assert d.params["min"] == -1000 and d.params["max"] == 1234.5


@pytest.mark.parametrize("values,style", [
    (["-1,234.5", "12.0", "2,000.0"], "-x"),
    (["(1,234.5)", "12.0", "2,000.0"], "(x)"),
    (["-$1,234.50", "$12.00", "$2,000.00"], "-$x"),
    (["$-1,234.50", "$12.00", "$2,000.00"], "$-x"),
    (["($1,234.50)", "$12.00", "$2,000.00"], "($x)"),
])
def test_negative_styles(values, style) -> None:
    d = detect(values)
    assert d.type is T.NUMERIC_FORMATTED and d.params["negative_style"] == style


def test_percent_euro_pound_and_inner_spaces() -> None:
    d = detect(["45.5%", "-3.2%", "100.0%", "0.0%"])
    assert d.type is T.NUMERIC_FORMATTED and d.params["format"]["affix"] == [None, "%"]
    assert d.params["template"] == "0.0%;-0.0%" and d.params["max"] == 100
    d = detect([f"{EURO}12.50", f"{EURO}1,300.00", f"{EURO}7.25"])
    assert d.params["format"]["positive"] == f"{EURO}#"
    d = detect([f"{POUND}12,345", f"{POUND}4,000", f"{POUND}912"])
    assert d.params["integer_valued"] and d.params["template"] == f"{POUND}#,##0"
    d = detect(["EUR 12.50", "EUR 1,300.00", "EUR 7.25", "EUR  3.00"])
    assert d.params["format"]["positive"] == "EUR #"  # majority spacing
    d = detect(["12.5 %", "3.0 %", "99.9 %"])
    assert d.params["format"]["positive"] == "# %"


def test_thousands_only_and_majority_grouping() -> None:
    d = detect(["1,234", "56", "250,000", "7"])
    assert d.type is T.NUMERIC_FORMATTED and d.params["integer_valued"]
    assert d.params["format"]["group_sep"] == "," and d.params["template"] == "#,##0"
    d = detect(["$1234.00", "$5,678.00", "$9999.00", "$12.00"])  # 2 of 3 large values ungrouped
    assert d.params["format"]["group_sep"] is None


def test_formatted_needs_one_affix_and_a_feature() -> None:
    assert detect(["$5.00", "5%", "$7.00"]).type is not T.NUMERIC_FORMATTED
    assert detect(["$5.00", "5.00", "$7.00", "$1.00"]).type is not T.NUMERIC_FORMATTED
    assert detect(["3.50", "2.75", "4.00"]).type is T.FLOAT  # plain floats stay FLOAT


def test_variable_decimals_and_leading_zero_style() -> None:
    d = detect(["$.50", "$.25", "$1.5", "$2"])
    f = d.params["format"]
    assert not f["fixed_decimals"] and (f["min_decimals"], f["decimals"]) == (0, 2)
    assert f["leading_zero"] is False


def test_decimal_comma_only_with_flag() -> None:
    vals = [f"{i}.{i:03d},{i % 100:02d}" for i in range(1, 40)]
    assert detect(vals).type is T.HIGH_CARD_CATEGORICAL
    d = detect(vals, decimal=",")
    assert d.type is T.NUMERIC_FORMATTED
    f = d.params["format"]
    assert (f["decimal_mark"], f["group_sep"]) == (",", ".") and d.params["template"] == "#.##0,00"
    assert detect(["12,5%", "3,0%", "100,0%"], decimal=",").params["template"] == "0,0%"
    assert detect(["12,5", "3,25", "0,75"], decimal=",").type is T.NUMERIC_FORMATTED
    assert detect(["12,5", "3,25", "0,75"]).type is T.FACTOR  # no --decimal: text/factor
    assert detect(["12", "3", "7"], decimal=",").type is T.INTEGER


def test_decimal_comma_flag_applies_to_text_formats_only() -> None:
    assert detect(["3.5", "2.25", "1.0"], decimal=",").type is T.FACTOR  # '.' cannot be a decimal mark
    assert detect(["3.5", "2.25", "1.0"], fmt="parquet", decimal=",").type is T.FLOAT  # canonical numbers


# --------------------------------------------------------------------------- NUMERIC_WITH_TOKENS
def test_numbers_with_suppression_tokens() -> None:
    vals = [str(i) for i in range(10, 60)] * 2 + ["*"] * 5 + ["<10"] * 3 + ["n<10"]
    d = detect(vals, name="n_tested")
    assert d.type is T.NUMERIC_WITH_TOKENS
    assert d.params["tokens"] == ["*", "<10", "n<10"] and d.params["suppression_like"]
    assert d.params["numeric_share"] == round(100 / 109, 6)
    assert (d.params["min"], d.params["max"]) == (10, 59)


def test_tokens_with_few_distinct_values_still_numeric_when_suppression() -> None:
    d = detect([str(i) for i in range(1, 16)] + ["*"])
    assert d.type is T.NUMERIC_WITH_TOKENS


def test_formatted_numeric_part_with_tokens() -> None:
    vals = [f"{i}.5%" for i in range(5, 95)] + [">95%", "<5%", "*"]
    d = detect(vals)
    assert d.type is T.NUMERIC_WITH_TOKENS and d.params["format"]["affix"] == [None, "%"]
    assert d.params["tokens"] == ["*", "<5%", ">95%"]


def test_na_tokens_are_missing_not_tokens() -> None:
    d = detect(["1.5", "2.5", "--", "NA", "N<10", "3.5"] * 5)
    assert d.type is T.NUMERIC_WITH_TOKENS and d.params["tokens"] == ["N<10"]
    p = profile(["1.5", "2.5", "--", "NA", "N<10", "3.5"] * 5)
    assert set(p.model.missing_counts) == {"--", "NA"}
    assert p.model.sorted_numeric.tolist() == sorted([1.5, 2.5, 3.5] * 5)


def test_tokens_rules_fall_through_to_factor() -> None:
    # non-suppression tokens with <= factor-max distinct values: FACTOR (fixes grade_level-like columns)
    assert detect([str(i) for i in range(1, 13)] + ["Ungraded", "Other"]).type is T.FACTOR
    assert detect(["PK", "KG"] + [f"{g:02d}" for g in range(1, 13)], name="grade_level").type is T.ORDINAL
    # leading-zero numbers are codes
    assert detect([f"{i:02d}" for i in range(1, 30)] + ["*"]).type is not T.NUMERIC_WITH_TOKENS
    # < 50% numeric
    assert detect(["1", "2", "*", "<10", "*"]).type is not T.NUMERIC_WITH_TOKENS
    # > 10 distinct tokens
    toks = [f"<{i}" for i in range(11)]
    assert detect([str(i) for i in range(100)] + toks).type is not T.NUMERIC_WITH_TOKENS


def test_other_tokens_above_factor_max_are_private() -> None:
    vals = [str(i) for i in range(0, 181)] + ["Pending"] * 4 + ["Withdrawn"] * 2
    d = detect(vals, name="days")
    assert d.type is T.NUMERIC_WITH_TOKENS and not d.params["suppression_like"]
    assert "tokens" not in d.params and d.params["_tokens"] == ["Pending", "Withdrawn"]
    assert any(m.code == "numeric_tokens_verbatim" for m in d.messages)
    p = profile(vals, name="days")
    pub = p.to_public()
    assert "Pending" not in str(pub) and "Withdrawn" not in str(pub)
    assert p.stats["token_counts"] == {"token_1": 4, "token_2": 2}


@pytest.mark.parametrize("token", ["*", "**", "<10", "< 10", "<=10", ">95%", "<5%", "n<10", "N<10", "n < 10",
                                   "DS", "ds", "--", "N/R", "NR", "^", "S", "\u226410"])
def test_suppression_tokens(token) -> None:
    assert N.is_suppression_token(token)


@pytest.mark.parametrize("token", ["Pending", "N/A", "abc", "<", "x", "10<"])
def test_not_suppression_tokens(token) -> None:
    assert not N.is_suppression_token(token)


# --------------------------------------------------------------------------- scientific notation
def test_python_style_mixed_scientific() -> None:
    vals = ["0.0253", "0.018", "2.62e-06", "4.11e-05", "0.5", "1e-07", "0.000123"]
    d = detect(vals, name="p_value")
    assert (d.type, d.subtype) == (T.FLOAT, "scientific")
    s = d.params["scientific"]
    assert (s["e_char"], s["exp_plus"], s["exp_digits"]) == ("e", True, 2)
    assert (s["mantissa_min_decimals"], s["mantissa_decimals"], s["mantissa_fixed"]) == (0, 2, False)
    assert s["sci_exponents"] == [-7, -6, -5] and s["plain_exponents"] == [-4, -2, -1]
    assert s["plain"]["sig_digits"] == 3 and not d.params["excel_mangled"]
    assert d.params["min"] == 1e-07 and d.params["max"] == 0.5
    assert not d.messages


def test_scientific_style_variants() -> None:
    d = detect(["1.5E-3", "2.0E-4", "3.25E-3"])
    s = d.params["scientific"]
    assert (s["e_char"], s["exp_digits"]) == ("E", 1)
    d = detect(["1.5e5", "2.5e6", "3e7"])
    assert d.params["scientific"]["exp_plus"] is False


def test_excel_mangled_ids_warn() -> None:
    vals = ["6.53396E+11", "1.87254E+11", "6.14118E+11", "5.5127E+11", "1E+12"]
    d = detect(vals, name="ext_ref")
    assert d.type is T.FLOAT and d.params["excel_mangled"]
    msg = [m for m in d.messages if m.code == "excel_scientific_ids"]
    assert msg and msg[0].level == "warning" and msg[0].fix
    for v in vals:
        assert v not in msg[0].text  # no real value in the message
    assert not detect(["6.53396e+11", "1.87254e+11"]).params["excel_mangled"]  # lower-case e: not Excel
    assert not detect(["6.53396E+09", "1.87254E+09"]).params["excel_mangled"]  # Excel shows these in full
    assert not detect(["6.53396E+11", "1.5E+11", "0.5"]).params["excel_mangled"]


def test_force_float_accepts_scientific() -> None:
    d = detect(["1e5", "2.5e5", "1000"], force_type=["col=FLOAT"])
    assert d.type is T.FLOAT and d.source == "force-type"


# --------------------------------------------------------------------------- DECIMAL
@pytest.mark.parametrize("native", [{"arrow_type": pa.decimal128(9, 2)}, {"arrow_type": "decimal128(9, 2)"},
                                    {"type": "decimal(9,2)"}, {"dtype": pa.decimal256(9, 2)}])
def test_native_decimal(native) -> None:
    d = detect(["123.45", "-0.50", "1000.00", "7.10"], fmt="parquet", native=native)
    assert d.type is T.DECIMAL and d.source == "native"
    assert (d.params["precision"], d.params["scale"]) == (9, 2)
    assert (d.params["min"], d.params["max"]) == (-0.5, 1000)


def test_native_decimal_contract_helpers() -> None:
    assert N.native_decimal({"arrow_type": pa.float64()}) is None
    assert N.native_decimal({"number_format": "0.00"}) is None
    assert N.native_decimal(None) is None
    ctx = ctx_for(["1.5", "2.25"], fmt="parquet")
    d = N.decimal_detection(ctx, 5, 2)
    assert d.type is T.DECIMAL and d.params["scale"] == 2
    assert N.decimal_detection(ctx_for(["1.5", "x"]), 5, 2) is None


def test_force_decimal_on_text() -> None:
    d = detect(["123.45", "-0.5", "10"], force_type=["col=DECIMAL"])
    assert d.type is T.DECIMAL and d.source == "force-type"
    assert (d.params["precision"], d.params["scale"]) == (5, 2)
    with pytest.raises(UnsupportedColumnError):
        detect(["$5", "x"], force_type=["col=DECIMAL"])


def test_forced_area_b_types() -> None:
    d = detect(["1.5", "2.5", "3"], force_type=["col=NUMERIC_FORMATTED"])
    assert d.type is T.NUMERIC_FORMATTED and d.params["template"] == "0.#" and d.source == "force-type"
    d = detect(["1", "2", "x"], force_type=["col=NUMERIC_WITH_TOKENS"])
    assert d.type is T.NUMERIC_WITH_TOKENS and d.params["_tokens"] == ["x"]
    with pytest.raises(UnsupportedColumnError):
        detect(["a", "b", "c"], force_type=["col=NUMERIC_FORMATTED"])


# --------------------------------------------------------------------------- FLOAT refinements
def test_float_style_majorities() -> None:
    d = detect([".5", ".25", "0.75", ".125"])
    assert d.type is T.FLOAT and d.params["leading_zero"] is False
    d = detect(["3.5", "3.25", "4", "2.75", "1.5", "1.5"])
    assert (d.params["min_decimals"], d.params["modal_decimals"], d.params["decimals"]) == (0, 1, 2)
    assert not d.params["fixed_decimals"]
    d = detect(["0.5", ".25", "0.75"])
    assert d.params["leading_zero"] is True
    d = detect(["+2.5", "-1.0", "+3.0", "0.0", "1.5"])
    assert d.params["plus_sign"] is True
    d = detect(["+2.5", "1.0", "3.5"])
    assert d.params["plus_sign"] is False
    d = detect(["72.0", "81.0", "90.0"])
    assert d.type is T.INTEGER and d.params["float_text"]


# --------------------------------------------------------------------------- stats, numeric view, docs
def test_stats_and_numeric_view_formatted() -> None:
    p = profile(["$1,234.50", "($25.00)", "$3.00", "", "NA"])
    assert p.stats["min"] == -25 and p.stats["max"] == 1234.5 and p.stats["missing"] == 2
    assert p.stats["template"] == "$#,##0.00;($#,##0.00)" and p.stats["format_conformance"] == 1.0
    assert p.model.sorted_numeric.tolist() == [-25.0, 3.0, 1234.5]
    syn = column_stats(_arr(["$5.00", "oops", "($1.00)"]), p, 10)
    assert syn["format_conformance"] == round(2 / 3, 6) and syn["min"] == -1
    assert N.numbers_extended(["$1.00", "5%"], p).tolist()[0] == 1.0
    assert np.isnan(N.numbers_extended(["$1.00", "5%"], p)[1])


def test_stats_with_tokens() -> None:
    vals = [str(i) for i in range(10, 60)] + ["*"] * 4 + ["<10"] * 2
    p = profile(vals)
    st = p.stats
    assert st["token_counts"] == {"*": 4, "<10": 2} and st["numeric_share"] == round(50 / 56, 6)
    assert st["token_shares"]["*"] == round(4 / 56, 6)
    assert (st["min"], st["max"]) == (10, 59)
    assert "level_shares" not in st  # the fidelity check stays KS on the numeric part


def test_decimal_stats() -> None:
    p = profile(["1.50", "2.25", "-3.00"], fmt="parquet", native={"arrow_type": pa.decimal128(5, 2)})
    assert p.type is T.DECIMAL and (p.stats["precision"], p.stats["scale"]) == (5, 2)
    assert p.stats["min"] == -3 and p.model.sorted_numeric.tolist() == [-3.0, 1.5, 2.25]


def test_doc_fields() -> None:
    p = profile(["$1,234.50", "($25.00)", "$3.00"])
    f = N.doc_fields(p, p.stats)
    assert f["readr"] == "c" and f["template"] == "$#,##0.00;($#,##0.00)" and "parse_number" in f["parse"]
    assert "parentheses" in f["parse"] and f["parsed_r_class"] == "numeric"
    assert N.doc_fields(profile(["$1,234", "$5", "$70"]), {})["parsed_r_class"] == "integer"
    e = docs.column_entry(p, p.stats, "str", 10)
    assert e["readr"] == "c" and e["r_class"] == "numeric" and e["template"]
    p = profile(["12,5", "3,25", "1.000,75"], decimal=",")
    assert 'decimal_mark = ","' in N.doc_fields(p, p.stats)["parse"]
    p = profile([str(i) for i in range(10, 60)] + ["*", "<10"])
    f = N.doc_fields(p, p.stats)
    assert f["tokens"] == ["*", "<10"] and 'na = c("*", "<10")' in f["parse"]
    p = profile([str(i) for i in range(0, 181)] + ["Pending", "Withdrawn"])
    assert "Pending" not in str(N.doc_fields(p, p.stats))
    p = profile(["1.50", "2.25"], fmt="parquet", native={"arrow_type": pa.decimal128(5, 2)})
    assert N.doc_fields(p, p.stats) == {"readr": "d", "precision": 5, "scale": 2, "arrow_type": "decimal128(5, 2)"}
    assert docs.column_entry(p, p.stats, None, 10)["readr"] == "d"
