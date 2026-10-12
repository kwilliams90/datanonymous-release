"""AREA B generation and rendering: templates re-applied, tokens by quota, notations kept (design 3 #15-#18, D14)."""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import given, settings, strategies as st

from datanonymous.detect import numeric as N
from datanonymous.errors import UnsupportedColumnError
from datanonymous.generate import numeric as G
from datanonymous.generate.table import generate_values
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType as T

from test_area_b_detect import EURO, POUND, profile


def gen(p, n=None, seed=0):
    n = p.model.n_nonmissing if n is None else n
    vals, flags = generate_values(p, p.model, n, np.random.default_rng(seed))
    return vals.tolist(), flags


def nonmissing(values):
    return [v for v in values if v.strip() and v.strip() not in ("NA", "--")]


# --------------------------------------------------------------------------- NUMERIC_FORMATTED
def _money(x: float) -> str:
    return f"(${-x:,.2f})" if x < 0 else f"${x:,.2f}"


def test_formatted_values_conform_and_keep_extremes() -> None:
    rng = np.random.default_rng(1)
    real = [_money(x) for x in np.round(rng.normal(300, 900, 400), 2)]
    p = profile(real, name="fee")
    assert p.type is T.NUMERIC_FORMATTED
    syn, flags = gen(p)
    nums = [N.parse_with_format(v, p.params["format"]) for v in syn]
    assert not any(math.isnan(x) for x in nums)  # 100% format conformance
    lo, hi = min(N.parse_number(v).value for v in real), max(N.parse_number(v).value for v in real)
    assert min(nums) == lo and max(nums) == hi
    assert _money(lo) in syn and _money(hi) in syn  # exact extremes in the source text
    assert any(v.startswith("($") for v in syn) and any("," in v for v in syn)
    assert all(re.fullmatch(r"\(?\$[0-9]{1,3}(,[0-9]{3})*\.[0-9]{2}\)?", v) for v in syn)
    unseen = np.mean([v not in set(real) for v in syn])
    assert unseen > 0.8


def test_all_negative_and_repeated_values() -> None:
    p = profile([f"(${x:,.2f})" for x in np.linspace(5, 1500, 60)])
    f = p.params["format"]
    assert (f["positive"], f["negative"], f["zero"]) == ("$#", "($#)", "$#")
    syn, _ = gen(p)
    assert all(v.startswith("($") for v in syn)
    many = syn * 5  # the numeric view of repeated synthetic values parses each distinct value once
    x = N.numbers_extended(many, p)
    assert len(x) == len(many) and not np.isnan(x).any() and (x < 0).all()
    assert G.format_number(-0.001, f) == "$0.00"  # rounds to zero: the zero form, never '($0.00)'


def test_formatted_integer_valued_rounds_to_whole_numbers() -> None:
    real = [f"${x:,}.00" for x in range(5, 2000, 7)]
    p = profile(real)
    assert p.params["integer_valued"]
    syn, _ = gen(p)
    assert all(v.endswith(".00") for v in syn)


@pytest.mark.parametrize("real,pattern", [
    ([f"{x:.1f}%" for x in np.linspace(-20, 100, 90)], r"-?[0-9]+\.[0-9]%"),
    ([f"-${-x:,.2f}" if x < 0 else f"${x:,.2f}" for x in np.linspace(-500, 3000, 90)], r"-?\$[0-9,]+\.[0-9]{2}"),
    ([f"$-{-x:,.2f}" if x < 0 else f"${x:,.2f}" for x in np.linspace(-500, 3000, 90)], r"\$-?[0-9,]+\.[0-9]{2}"),
    ([f"{x:,.2f} {EURO}" for x in np.linspace(1, 5000, 90)], rf"[0-9,]+\.[0-9]{{2}} {EURO}"),
    ([f"{POUND}{x:,.0f}" for x in np.linspace(1000, 90000, 90)], rf"{POUND}[0-9]{{1,2}},[0-9]{{3}}"),
    ([f"{x:,.0f}" for x in np.linspace(10, 500000, 90)], r"[0-9]{1,3}(,[0-9]{3})*"),
])
def test_formatted_rendering_patterns(real, pattern) -> None:
    p = profile(real)
    assert p.type is T.NUMERIC_FORMATTED
    syn, _ = gen(p)
    bad = [v for v in syn if not re.fullmatch(pattern, v)]
    assert not bad, bad[:5]


def test_decimal_comma_rendering() -> None:
    real = [f"{x:,.2f}".translate(str.maketrans({",": ".", ".": ","})) for x in np.linspace(-900, 25000, 80)]
    p = profile(real, decimal=",")
    syn, _ = gen(p)
    assert all(re.fullmatch(r"-?[0-9]{1,3}(\.[0-9]{3})*,[0-9]{2}", v) for v in syn)
    assert min(N.parse_with_format(v, p.params["format"]) for v in syn) == -900


def test_variable_decimals_and_no_leading_zero() -> None:
    p = profile(["$.50", "$.25", "$1.5", "$2", "$.125", "$3.75"] * 5)
    syn, _ = gen(p)
    assert all(not v.startswith("$0.") for v in syn)
    assert all(len(v.partition(".")[2]) <= 3 for v in syn)


def test_no_exact_extremes_flag() -> None:
    p = profile([_money(x) for x in np.linspace(-50, 50, 101)], no_exact_extremes=["col"])
    assert p.params["_no_exact_extremes"]
    _, flags = gen(p, seed=3)
    assert not flags.get("extreme_value_adjusted")


# --------------------------------------------------------------------------- NUMERIC_WITH_TOKENS
def test_tokens_exact_quota_and_numeric_part() -> None:
    real = [str(i) for i in range(10, 200)] * 2 + ["*"] * 7 + ["<10"] * 3 + [" n<10"] + ["--"] * 4
    p = profile(real, name="n_tested")
    assert p.type is T.NUMERIC_WITH_TOKENS
    syn_nm, flags = gen(p)
    c = Counter(syn_nm)
    assert c["*"] == 7 and c["<10"] == 3 and c[" n<10"] == 1  # verbatim, raw whitespace kept
    nums = [int(v) for v in syn_nm if v.strip() not in ("*", "<10", "n<10")]
    assert len(nums) == 380 and min(nums) == 10 and max(nums) == 199
    assert not flags["rare_category_adjusted"]


def test_tokens_floor_when_fewer_rows() -> None:
    real = [str(i) for i in range(10, 500)] + ["*"] * 2 + ["<10"]
    p = profile(real)
    syn, flags = gen(p, n=20)
    assert "*" in syn and "<10" in syn and flags["rare_category_adjusted"]
    assert len(syn) == 20


def test_tokens_with_formatted_numeric_part() -> None:
    real = [f"{x:.1f}%" for x in np.linspace(5, 95, 120)] + [">95%"] * 6 + ["<5%"] * 4 + ["*"] * 2
    p = profile(real)
    syn, _ = gen(p)
    toks = Counter(v for v in syn if v in (">95%", "<5%", "*"))
    assert toks == {">95%": 6, "<5%": 4, "*": 2}
    for v in syn:
        if v not in toks:
            assert re.fullmatch(r"[0-9]+\.[0-9]%", v), v
            assert 5 <= float(v[:-1]) <= 95


def test_private_tokens_written_verbatim() -> None:
    real = [str(i) for i in range(0, 181)] + ["Pending"] * 4 + ["Withdrawn"] * 2
    p = profile(real)
    syn, _ = gen(p)
    assert Counter(syn)["Pending"] == 4 and Counter(syn)["Withdrawn"] == 2


# --------------------------------------------------------------------------- DECIMAL
def test_decimal_generation_scale() -> None:
    real = [f"{x:.2f}" for x in np.linspace(-1000, 99999.99, 150)]
    p = profile(real, fmt="parquet", native={"arrow_type": pa.decimal128(7, 2)})
    assert p.type is T.DECIMAL
    syn, flags = gen(p)
    assert all(re.fullmatch(r"-?[0-9]+\.[0-9]{2}", v) for v in syn)
    vals = [float(v) for v in syn]
    assert min(vals) == -1000 and max(vals) == 99999.99 and "extreme_value_adjusted" in flags
    arr = pa.array([Decimal(v) for v in syn]).cast(pa.decimal128(7, 2))  # the writer's cast back is exact
    assert [str(x) for x in arr.to_pylist()] == syn


def test_decimal_zero_and_negative_scale_render() -> None:
    p = ColumnProfile(index=0, name="d", type=T.DECIMAL, params={"scale": 0, "precision": 5})
    assert G.render_numeric_extended(p, np.array([12.0, -0.0, 3.0])).tolist() == ["12", "0", "3"]
    p = ColumnProfile(index=0, name="d", type=T.DECIMAL, params={"scale": -2, "precision": 5})
    m = ColumnModel(sorted_numeric=np.array([100.0, 1200.0, 5500.0]))
    y, _ = G.generate_numeric_extended(p, m, 10, np.random.default_rng(0))
    assert all(v % 100 == 0 for v in y)


# --------------------------------------------------------------------------- FLOAT (scientific, styles)
def test_scientific_text_styles() -> None:
    py = {"e_char": "e", "exp_plus": True, "exp_digits": 2, "mantissa_decimals": 2, "mantissa_min_decimals": 0,
          "mantissa_fixed": False}
    assert G.scientific_text(2.62e-06, py) == "2.62e-06"
    assert G.scientific_text(1e-07, py) == "1e-07"
    assert G.scientific_text(2.5e16, py) == "2.5e+16"
    excel = {"e_char": "E", "exp_plus": True, "exp_digits": 2, "mantissa_decimals": 5, "mantissa_fixed": True}
    assert G.scientific_text(653396000000.0, excel) == "6.53396E+11"
    java = {"e_char": "E", "exp_plus": False, "exp_digits": 1, "mantissa_decimals": 1, "mantissa_fixed": True}
    assert G.scientific_text(-0.00015, java) == "-1.5E-4"
    assert G.scientific_text(-0.0, java) == "0.0E0"


def test_mixed_scientific_column_rendering() -> None:
    rng = np.random.default_rng(5)
    real = [repr(float(f"{x:.3g}")) for x in 10 ** rng.uniform(-7, 0, 400)]
    p = profile(real, name="p_value")
    assert p.subtype == "scientific"
    syn, _ = gen(p)
    for v in syn:
        x = float(v)
        if x < 1e-4:
            assert re.fullmatch(r"[0-9](\.[0-9]{1,2})?e-0[5-7]", v), v
        else:
            assert "e" not in v and len(v.lstrip("0.")) <= 3, v
    assert min(syn, key=float) == min(real, key=float)


def test_excel_scientific_ids_same_notation() -> None:
    rng = np.random.default_rng(2)
    real = [f"{x:.5E}" for x in rng.integers(10**11, 10**12, 200)]
    p = profile(real, name="ext_ref")
    syn, _ = gen(p)
    assert all(N.EXCEL_SCI_RE.fullmatch(v) for v in syn)
    assert max(real, key=float) in syn and min(real, key=float) in syn


def test_float_style_rendering() -> None:
    p = profile([f"{x:.3f}"[1:] for x in np.linspace(0.05, 0.95, 50)])
    syn, _ = gen(p)
    assert all(v.startswith(".") for v in syn)
    p = profile([f"{x:+.1f}" if round(x, 1) else "0.0" for x in np.linspace(-3, 3, 61)])
    syn, _ = gen(p)
    assert all(v.startswith(("+", "-")) or float(v) == 0 for v in syn)
    p = profile(["3.5", "4", "2.75", "3.25", "1", "0.5"] * 10)
    syn, _ = gen(p)
    assert any("." not in v for v in syn) and all(len(v.partition(".")[2]) <= 2 for v in syn)
    p = profile(["3.50", "4.00", "2.75", "0.00"] * 10)
    syn, _ = gen(p)
    assert all(re.fullmatch(r"[0-9]\.[0-9]{2}", v) for v in syn) and "4.00" in syn and "0.00" in syn


def test_foundation_integer_float_text_unchanged() -> None:
    p = profile([f"{x}.0" for x in range(150, 1000, 3)])
    assert p.type is T.INTEGER and p.subtype == "continuous"
    syn, _ = gen(p)
    assert all(re.fullmatch(r"[0-9]+\.0", v) for v in syn)


# --------------------------------------------------------------------------- hooks and properties
def test_extended_hooks_reject_other_types() -> None:
    p = ColumnProfile(index=0, name="x", type=T.FLOAT)
    with pytest.raises(UnsupportedColumnError):
        G.generate_numeric_extended(p, ColumnModel(), 1, np.random.default_rng(0))
    with pytest.raises(UnsupportedColumnError):
        G.render_numeric_extended(p, np.array([1.0]))
    assert N.numbers_extended(["1"], p) is None and N.numeric_stats_extended(["1"], p) == {}


def test_render_is_deterministic() -> None:
    p = profile([_money(x) for x in np.linspace(-50, 5000, 80)])
    a, _ = gen(p, seed=11)
    b, _ = gen(p, seed=11)
    c, _ = gen(p, seed=12)
    assert a == b and a != c


_SKELETONS = ["#", "$#", "$ #", f"{EURO}#", f"# {EURO}", "EUR #", "#%", "# %", f"{POUND}#"]


@st.composite
def formats(draw):
    pos = draw(st.sampled_from(_SKELETONS))
    neg_kind = draw(st.sampled_from(["minus", "paren", "inner"]))
    pre = pos.split("#")[0]
    if neg_kind == "minus":
        neg = "-" + pos
    elif neg_kind == "paren":
        neg = "(" + pos + ")"
    else:
        neg = (pre + "-" + "#" + pos.split("#")[1]) if pre else "-" + pos
    mark = draw(st.sampled_from([".", ","]))
    d = draw(st.integers(0, 4))
    fixed = draw(st.booleans())
    affix = N.parse_number(pos.replace("#", "1"), mark).affix
    return {"decimal_mark": mark, "group_sep": draw(st.sampled_from([None, N.GROUP_SEP[mark]])), "decimals": d,
            "min_decimals": d if fixed else draw(st.integers(0, d)), "fixed_decimals": fixed,
            "leading_zero": draw(st.booleans()), "affix": list(affix), "positive": pos, "negative": neg,
            "zero": pos}


@settings(max_examples=300, deadline=None)
@given(fmt=formats(), x=st.floats(-1e9, 1e9, allow_nan=False))
def test_format_number_round_trip(fmt, x) -> None:
    text = G.format_number(x, fmt)
    back = N.parse_with_format(text, fmt)
    assert not math.isnan(back), (text, fmt)
    assert back == pytest.approx(round(x, fmt["decimals"]), abs=10 ** -fmt["decimals"] / 2 + 1e-6)


@settings(max_examples=200, deadline=None)
@given(x=st.floats(-1e12, 1e12, allow_nan=False).filter(lambda v: v != 0),
       d=st.integers(0, 6), e=st.sampled_from(["e", "E"]), plus=st.booleans(), digits=st.integers(1, 3))
def test_scientific_text_round_trip(x, d, e, plus, digits) -> None:
    sci = {"e_char": e, "exp_plus": plus, "exp_digits": digits, "mantissa_decimals": d, "mantissa_fixed": True}
    text = G.scientific_text(x, sci)
    m = N.SCI_RE.fullmatch(text)
    assert m and m.group("e") == e and len(m.group("exp")) >= digits
    assert float(text) == pytest.approx(x, rel=10 ** -d)
