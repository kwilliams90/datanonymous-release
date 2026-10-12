"""AREA B hardening: regressions for issues found by adversarial review (detection and generation units)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import given, settings, strategies as st

from datanonymous.detect import numeric as N
from datanonymous.generate import numeric as G
from datanonymous.generate.table import generate_values
from datanonymous.model import ColumnModel, ColumnType as T
from datanonymous.report import docs

from test_area_b_detect import detect, profile

FULLWIDTH = str.maketrans("0123456789", "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19")
ARABIC_INDIC = str.maketrans("0123456789", "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")


def gen(p, n=None, seed=0, model=None):
    model = p.model if model is None else model
    n = model.n_nonmissing if n is None else n
    vals, flags = generate_values(p, model, n, np.random.default_rng(seed))
    return vals.tolist(), flags


# --------------------------------------------------------------------------- non-ASCII digits
def test_non_ascii_digits_are_not_numbers() -> None:
    """float() reads Arabic-Indic/fullwidth digits; the renderer would write ASCII digits back."""
    real = [f"{x:.1f}".translate(ARABIC_INDIC) for x in np.linspace(1, 9, 12)] * 5
    p = profile(real)
    assert p.type is T.FACTOR  # levels: the raw texts are written back verbatim
    syn, _ = gen(p)
    assert Counter(syn) == Counter(real)
    years = [str(y).translate(FULLWIDTH) for y in (1999, 2001, 2005, 2010)] * 3
    assert detect(years).type is not T.INTEGER_YEAR
    ints = [str(x).translate(FULLWIDTH) for x in range(100, 400)]
    assert detect(ints).type is not T.INTEGER
    assert detect([str(x) for x in range(100, 400)]).type is T.INTEGER  # ASCII unchanged


# --------------------------------------------------------------------------- '12.' trailing point
def test_trailing_point_kept_on_integer_valued_text() -> None:
    real = [f"{x}." for x in range(1000, 61000, 97)]
    p = profile(real)
    assert p.type is T.INTEGER and p.subtype == "continuous" and p.params["trailing_point"]
    syn, _ = gen(p)
    assert all(v.endswith(".") and v[:-1].isdigit() for v in syn)  # pandas reads float64 as for the source


def test_trailing_point_in_variable_decimal_floats() -> None:
    real = [f"{x}." for x in range(10, 90)] + [f"{x + 0.25:.2f}" for x in range(10, 90)]
    p = profile(real)
    assert p.type is T.FLOAT and p.params["trailing_point"]
    syn, _ = gen(p)
    assert all(v.endswith(".") or "." in v[:-1] for v in syn)
    assert any(v.endswith(".") for v in syn)
    assert G.render_numeric(p, np.array([41.0, 3.5])).tolist() == ["41.", "3.5"]


def test_no_trailing_point_param_for_ordinary_numbers() -> None:
    for real in ([f"{x:.2f}" for x in np.linspace(0, 4, 90)], [f"{x}.0" for x in range(100, 900, 3)]):
        assert "trailing_point" not in profile(real).params


def test_forced_float_keeps_trailing_point() -> None:
    p = profile([f"{x}." for x in range(1, 40)], force_type=["col=FLOAT"])
    assert p.type is T.FLOAT and p.params["trailing_point"]
    assert G.render_numeric(p, np.array([7.0])).tolist() == ["7."]


# --------------------------------------------------------------------------- scientific rounding
@pytest.mark.parametrize("lo,hi", [(-200, 200), (-300, -1), (-320, -300), (100, 300)])
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # profile.stats sd of values > 1e154
def test_wide_range_scientific_values_do_not_collapse(lo, hi) -> None:
    """A rounding cap by the largest value used to round the small values to zero (then clip to the min)."""
    real = [f"{x:.2e}" for x in np.geomspace(10.0 ** lo, 10.0 ** hi, 200)]
    p = profile(real)
    assert p.type is T.FLOAT and p.subtype == "scientific"
    syn, flags = gen(p)
    counts = Counter(syn)
    assert counts.most_common(1)[0][1] <= 2 and len(counts) >= 190
    nums = [float(v) for v in syn]
    assert min(nums) == float(real[0]) and max(nums) == float(real[-1])
    assert all(math.isfinite(x) for x in nums)


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=-1e12, max_value=1e12, allow_nan=False), st.integers(min_value=-3, max_value=12))
def test_round_decimals_matches_numpy_in_range(x, d) -> None:
    y = np.array([x])
    assert G.round_decimals(y, d)[0] == np.round(y, d)[0]


def test_round_decimals_overflow_and_tiny() -> None:
    y = np.array([1e300, 1.5, -2e305])
    r = G.round_decimals(y, 20)  # np.round would give inf for the huge values
    assert r.tolist() == [1e300, 1.5, -2e305]
    tiny = np.array([1.234e-320, 5.5e-310, 0.0])
    r = G.round_decimals(tiny, 323)
    assert np.all(np.isfinite(r)) and r[2] == 0.0
    assert [f"{v:.2e}" for v in r[:2]] == ["1.23e-320", "5.50e-310"]


# --------------------------------------------------------------------------- Excel-mangled IDs
def _excel(x: int) -> str:
    return str(x) if x < 10**11 else f"{x:.5E}"


def test_excel_ids_mixed_with_full_eleven_digit_integers_warn() -> None:
    rng = np.random.default_rng(3)
    real = [_excel(int(x)) for x in rng.integers(10**10, 10**12, 200)]
    assert any("E" not in v for v in real) and any("E" in v for v in real)
    p = profile(real, name="acct")
    assert p.type is T.FLOAT and p.subtype == "scientific" and p.params["excel_mangled"]
    assert [m.code for m in p.messages] == ["excel_scientific_ids"]
    assert "--no-exact-extremes acct" in p.messages[0].fix
    # suspected IDs (D21): no real value among the public params or the report statistics
    pub = p.to_public()
    assert "min" not in pub["params"] and "max" not in pub["params"] and p.params["id_candidate"]
    for key in ("min", "max", "median", "quantiles"):
        assert key not in p.stats
    text = repr(pub)
    assert not any(v in text for v in real)
    syn, _ = gen(p)
    assert all(v.isdigit() and len(v) <= 11 or N.EXCEL_SCI_RE.fullmatch(v) for v in syn)


@pytest.mark.parametrize("plain", ["123456789012", "12345678901.5", "-12345678901"])
def test_excel_rule_needs_excel_general_plain_values(plain) -> None:
    """Excel General writes 12+ digit integers in scientific notation, and never these plain forms with E+11 IDs."""
    real = ["6.53396E+11", "1.87254E+11", "6.14118E+11", "5.5127E+11", plain] * 3
    p = profile(real)
    assert p.type is T.FLOAT and p.subtype == "scientific" and not p.params["excel_mangled"]
    assert not p.messages and "id_candidate" not in p.params


# --------------------------------------------------------------------------- ambiguous grouping
@pytest.mark.parametrize("real,ambiguous", [
    ([f"{x:,}" for x in range(1000, 99999, 517)], True),
    ([f"{x:,} \u20ac" for x in range(1000, 99999, 517)], True),
    ([f"${x:,}" for x in range(1000, 99999, 517)], False),
    ([f"\u00a3{x:,}" for x in range(1000, 99999, 517)], False),
    ([f"{x:,.2f}" for x in range(1000, 99999, 517)], False),
    ([f"{x:,}" for x in range(1000, 9999999, 49999)], False),
])
def test_ambiguous_thousands_grouping_is_flagged(real, ambiguous) -> None:
    p = profile(real)
    assert p.type is T.NUMERIC_FORMATTED and p.ambiguous is ambiguous
    if ambiguous:
        assert "--decimal ," in p.ambiguity and p.ambiguity.isascii()
        assert not any(v in p.ambiguity for v in real)


def test_ambiguous_grouping_under_decimal_comma() -> None:
    p = profile([f"{x:,}".replace(",", ".") for x in range(1000, 99999, 517)], decimal=",")
    assert p.type is T.NUMERIC_FORMATTED and p.ambiguous and "without --decimal ," in p.ambiguity


# --------------------------------------------------------------------------- DECIMAL type names
def test_decimal256_is_documented_as_decimal256() -> None:
    p = profile(["1.50", "2.25", "-3.00"], fmt="parquet", native={"arrow_type": pa.decimal256(40, 2)})
    assert p.type is T.DECIMAL and p.params["arrow_type"] == "decimal256(40, 2)"
    assert N.doc_fields(p, p.stats)["arrow_type"] == "decimal256(40, 2)"
    assert "decimal256(40, 2)" in p.reason
    q = profile(["1.50", "2.25"], fmt="parquet", native={"arrow_type": "decimal128(5, 2)"})
    assert q.params["arrow_type"] == "decimal128(5, 2)"
    big = profile(["1" * 40 + ".5", "2" * 39 + ".25"], force_type=["col=DECIMAL"])
    assert big.params["precision"] == 42 and big.params["arrow_type"] == "decimal256(42, 2)"


def test_forced_decimal_precision_holds_every_value_at_the_scale() -> None:
    """'12345.6' with '1.23' is written '12345.60': decimal(7, 2), not decimal(6, 2)."""
    p = profile(["12345.6", "1.23", "-0.5", "007.10"], force_type=["col=DECIMAL"])
    assert (p.params["precision"], p.params["scale"]) == (7, 2)
    syn, _ = gen(p)
    arr = pa.array([__import__("decimal").Decimal(v) for v in syn]).cast(pa.decimal128(7, 2))
    assert len(arr) == 4


# --------------------------------------------------------------------------- D4 per-entity models (D14 floors)
def _entity_model(p, drop) -> ColumnModel:
    """A per-entity model like profile.table builds (most common value per entity) that lost ``drop``."""
    counts = {k: v for k, v in p.model.nonmissing_counts.items() if k not in drop}
    nums = N.numbers_extended([k.strip() for k in counts], p)
    xs = np.sort(np.repeat(nums, list(counts.values())))
    return ColumnModel(nonmissing_counts=counts, missing_counts=dict(p.model.missing_counts),
                       sorted_numeric=xs[np.isfinite(xs)])


def test_entity_model_floors_a_lost_token_and_extreme() -> None:
    real = [f"{x:.1f}" for x in np.linspace(20, 90, 80)] + ["*"] * 6 + ["<10"] + ["99.9"]
    p = profile(real)
    assert p.type is T.NUMERIC_WITH_TOKENS
    em = _entity_model(p, {"<10", "99.9"})
    syn, flags = gen(p, n=40, model=em)
    assert len(syn) == 40 and "<10" in syn and "*" in syn and "99.9" in syn
    assert flags["rare_category_adjusted"] and flags["extreme_value_adjusted"]
    # nothing lost: no floor beyond the usual ones
    em2 = _entity_model(p, set())
    syn2, flags2 = gen(p, model=em2)
    assert Counter(syn2)["<10"] == 1 and not flags2["rare_category_adjusted"]


def test_entity_model_of_tokens_only_keeps_the_numbers() -> None:
    real = ["*"] * 30 + ["<10"] * 5 + ["12.5", "40.0"]
    p = profile(real, force_type=["col=NUMERIC_WITH_TOKENS"])
    em = _entity_model(p, {"12.5", "40.0"})
    syn, flags = gen(p, n=20, model=em)
    assert len({"12.5", "40.0"} & set(syn)) == 1 and flags["rare_category_adjusted"]  # one numeric slot


def test_entity_model_floors_extremes_of_formatted_and_decimal() -> None:
    real = [f"${x:,.2f}" for x in np.linspace(5, 4000, 120)] + ["$9,999.99"]
    p = profile(real)
    assert p.type is T.NUMERIC_FORMATTED
    em = _entity_model(p, {"$9,999.99"})
    syn, flags = gen(p, n=30, model=em)
    assert "$9,999.99" in syn and "$5.00" in syn and flags["extreme_value_adjusted"]
    p.params["_no_exact_extremes"] = True  # --no-exact-extremes: never added
    syn, flags = gen(p, n=30, model=em)
    assert "$9,999.99" not in syn and not flags["extreme_value_adjusted"]
    d = profile(["1.50", "2.25", "7.75", "-3.00"] * 5 + ["99.99"], fmt="parquet",
                native={"arrow_type": pa.decimal128(5, 2)})
    syn, flags = gen(d, n=10, model=_entity_model(d, {"99.99"}))
    assert "99.99" in syn and "-3.00" in syn and flags["extreme_value_adjusted"]


def test_row_level_generation_is_unchanged_by_entity_floors() -> None:
    real = [f"{x:.1f}" for x in np.linspace(20, 90, 80)] + ["*"] * 6 + ["<10"]
    p = profile(real)
    a, fa = gen(p, seed=4)
    b, fb = gen(p, seed=4, model=p.model)
    assert a == b and fa == fb and not fa["rare_category_adjusted"]
    assert Counter(a)["<10"] == 1 and Counter(a)["*"] == 6


def test_docs_read_the_decimal_type_from_params() -> None:
    p = profile(["1.50", "2.25"], fmt="parquet", native={"arrow_type": pa.decimal128(5, 2)})
    entry = docs.column_entry(p, p.stats, "object", 10)
    assert entry["arrow_type"] == "decimal128(5, 2)" and entry["readr"] == "d"
