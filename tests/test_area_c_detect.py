"""AREA C detection: one table-driven test per text/PII rule (design 3 #25-35, D12, D13)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import text as dtext
from datanonymous.detect.context import DetectContext
from datanonymous.detect.types import detect_column
from datanonymous.errors import UsageError
from datanonymous.model import ColumnProfile, ColumnType as T, RawColumn, RawTable
from datanonymous.profile.table import profile_table


def _arr(values: list) -> np.ndarray:
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def ctx(values: list, name: str = "col", **cli) -> DetectContext:
    s = build_settings(cli)
    return DetectContext(index=0, name=name, values=_arr(values), file_rel="f.csv", table=None, format="csv",
                         options=s.column_options("f.csv", None, name), settings=s)


def detect(values: list, name: str = "col", **cli):
    return detect_column(ctx(values, name, **cli))


def profile(values: list, name: str = "col", **cli) -> ColumnProfile:
    s = build_settings(cli)
    tp = profile_table(RawTable(None, [name], [RawColumn(name, _arr(values))], len(values)), "f.csv", "csv", s)
    return tp.columns[0]


EMAILS = [f"user{i}.name@mail{i % 3}.com" for i in range(100)]
PHONES = [f"(214) {300 + i}-{1000 + i:04d}" for i in range(60)]
ADDRESSES = [f"{100 + 7 * i} {w} {s}" for i, (w, s) in enumerate(
    [("Oak", "St"), ("Elm", "Ave"), ("Cedar Ridge", "Dr"), ("Main", "Street"), ("Pecan", "Blvd")] * 12)]
NAMES = [f"{f} {l}" for f in ("Tavish", "Brennagh", "Xiomara", "Thaddeus", "Perpetua", "Lysander")
         for l in ("Okonkwo", "Abernathy", "Zielinski", "Thistlewood", "Lindqvist")]


# --------------------------------------------------------------------------- value-validated PII
def test_email_needs_95_percent() -> None:
    assert detect(EMAILS, name="contact").type is T.EMAIL  # no header hint needed
    assert detect(EMAILS[:95] + [f"not an email {i}" for i in range(5)], name="contact").type is T.EMAIL
    d = detect(EMAILS[:94] + [f"not an email {i}" for i in range(6)], name="contact")
    assert d.type is not T.EMAIL


def test_email_beats_a_name_hint_and_uses_the_typed_missing_policy() -> None:
    d = detect(EMAILS + ["N/A"] * 5 + [""] * 3, name="parent_name")
    assert d.type is T.EMAIL and d.policy() == "typed"
    p = profile(EMAILS + ["N/A"] * 5 + [""] * 3, name="parent_email")
    assert p.stats["missing"] == 8 and p.stats["missing_tokens"] == {"N/A": 5, "": 3}
    assert "N/A" not in p.model.real_values


def test_url_and_ip() -> None:
    urls = [f"https://www.site{i}.org/page/{i}" for i in range(30)] + [f"www.home{i}.net" for i in range(10)]
    assert detect(urls, name="link").type is T.URL
    ips = [f"10.0.{i}.{i + 1}" for i in range(40)] + ["2001:db8::1", "fe80::1ff:fe23:4567:890a"]
    assert detect(ips, name="source").type is T.IP
    assert detect([f"10.0.{i}.{i + 1}" for i in range(18)] + ["n/a", "unknown"], name="x").type is not T.IP
    assert detect(["999.1.1.1", "300.2.2.2"] * 10, name="x").type is not T.IP  # octets out of range


def test_ssn_regex_or_nine_digits_with_a_hint() -> None:
    dashed = [f"{100 + i:03d}-{10 + i % 80:02d}-{1000 + i:04d}" for i in range(40)]
    assert detect(dashed, name="code").type is T.SSN  # value regex alone
    nine = [f"{123450000 + 7 * i}" for i in range(40)]
    assert detect(nine, name="code").type is not T.SSN  # 9 digits without a hint: a number
    assert detect(nine, name="SSN").type is T.SSN
    mixed = dashed[:30] + nine[:10]
    assert detect(mixed, name="ssn").type is T.SSN
    masked = [f"XXX-XX-{1000 + i}" for i in range(20)]
    assert detect(masked, name="Social Security Number").type is T.SSN


def test_phone_needs_a_hint_and_90_percent() -> None:
    assert detect(PHONES, name="parent_phone").type is T.PHONE
    assert detect(PHONES, name="Contact_Number").type is T.PHONE
    d = detect(PHONES, name="col")
    assert d.type is not T.PHONE  # no hint
    forms = ["214-555-0101 x12", "+1 (972) 555-0102", "214.555.0103", "2145550104", "555-0105"] * 10
    assert detect(forms, name="home phone").type is T.PHONE
    d = detect(PHONES[:50] + [f"ask office {i}" for i in range(10)], name="phone")
    assert d.type is T.HIGH_CARD_CATEGORICAL and d.params["pii_hint"] == "phone"  # D12 guard, never copied


def test_zip_hint_regex_leading_zeros_and_plus4() -> None:
    zips = ["02134"] * 5 + ["75201"] * 10 + ["75201-1234", "07030"]
    d = detect(zips, name="zip")
    assert d.type is T.ZIP and d.policy() == "typed"
    assert detect(["75201", "75202", "76102"] * 5, name="Postal Code").type is T.ZIP
    assert detect(["2134", "7030", "75201"] * 5, name="zip_code").type is T.ZIP  # lost leading zero
    assert detect(["75201", "75202", "76102"] * 5, name="area").type is T.INTEGER  # no hint


def test_state_value_set() -> None:
    assert detect(["TX", "OK"] * 10, name="home_state").type is T.STATE
    assert detect(["TX", "OK", "LA", "NM"] * 5, name="region").type is T.STATE  # >= 3 distinct, no hint
    assert detect(["TX", "OK"] * 10, name="region").type is not T.STATE
    assert detect(["Texas", "Oklahoma", "New Mexico", "tx"] * 5, name="x").type is T.STATE
    assert detect(["TX", "OK", "ZZ"] * 5, name="state").type is not T.STATE


def test_city_needs_a_hint_and_place_like_values() -> None:
    cities = ["Dallas", "Fort Worth", "St. Louis", "Winston-Salem", "O'Fallon"] * 4
    d = detect(cities, name="city")
    assert d.type is T.CITY and d.policy() == "typed"
    assert detect(cities, name="destination").type is T.FACTOR
    d = detect(["C1", "C2", "C3"] * 5, name="Home City")
    assert d.type is T.HIGH_CARD_CATEGORICAL and d.params["pii_hint"] == "city"


def test_street_address_needs_80_percent() -> None:
    assert detect(ADDRESSES, name="where").type is T.STREET_ADDRESS
    with_po = ADDRESSES[:50] + [f"PO Box {i}" for i in range(10)]  # 83% match
    assert detect(with_po, name="where").type is T.STREET_ADDRESS
    low = ADDRESSES[:40] + [f"PO Box {i}" for i in range(20)]  # 67%
    assert detect(low, name="where").type is not T.STREET_ADDRESS
    assert detect(["12 Main St, Apt 4", "9 N Oak Ave Unit 2", "77 Elm Rd #5"] * 10, name="x").type \
        is T.STREET_ADDRESS


# --------------------------------------------------------------------------- PERSON_NAME (header hint only)
@pytest.mark.parametrize("header,sub", [("first_name", "first"), ("LastName", "last"), ("Middle Initial", "middle"),
                                        ("Student Name", "full"), ("Teacher", "full"), ("Guardian_Name", "full"),
                                        ("surname", "last"), ("Preferred Name", "first")])
def test_person_name_by_header_hint(header: str, sub: str) -> None:
    d = detect(NAMES, name=header)
    assert (d.type, d.subtype) == (T.PERSON_NAME, sub), header
    assert d.policy() == "typed" and d.params["layouts"]


def test_person_name_needs_the_hint_and_name_like_values() -> None:
    assert detect(NAMES, name="player").type is not T.PERSON_NAME  # header hint only (design 3 #26)
    d = detect([f"T{i:03d}" for i in range(40)], name="teacher_name")
    assert d.type is T.HIGH_CARD_CATEGORICAL and d.params["pii_hint"] == "full_name"
    layouts = detect(["O'Brien, José", "Smith-Jones, Mary-Kate", "de la Cruz, Zoë A."] * 4, name="student_name")
    assert layouts.type is T.PERSON_NAME
    assert set(layouts.params["layouts"]) == {"last_comma_first", "last_comma_first_initial"}


def test_d12_teacher_name_with_three_values_is_a_person_name() -> None:
    vals = ["Mrs. Alvarez"] * 100 + ["Mr. Thibodeaux"] * 90 + ["Ms. O'Neil"] * 50
    d = detect(vals, name="teacher_name")
    assert d.type is T.PERSON_NAME  # PII hint wins over the factor rule (3 <= factor-max)
    d = detect(vals, name="teacher_name", force_factor=["teacher_name"])
    assert d.type is T.FACTOR and d.source == "force-factor"  # --force-factor overrides


def test_d12_constant_and_flag_columns() -> None:
    d = detect(["Mrs. Alvarez"] * 20, name="Teacher")
    assert d.type is T.PERSON_NAME  # would be CONSTANT
    d = detect(["Tavish Okonkwo"] * 5 + [""] * 5, name="guardian name")
    assert d.type is T.PERSON_NAME  # would be FLAG_SINGLE_TOKEN
    d = detect(["office@school.org"] * 10, name="reply_to")
    assert d.type is T.EMAIL  # value-validated PII beats CONSTANT (no hint needed)
    assert detect(["Y"] * 10, name="reply_to").type is T.CONSTANT


def test_d12_guard_for_native_detections() -> None:
    from datanonymous.detect.context import Detection

    c = ctx([str(2145550100 + i) for i in range(30)], name="parent_phone")  # phones stored as numbers
    out = dtext.apply_pii_precedence(c, Detection(T.INTEGER, subtype="continuous", source="native"))
    assert out.type is T.PHONE and out.source == "native"
    c = ctx(["2134", "75201", "75201"] * 5, name="zip")  # Excel numbers that lost a leading zero
    out = dtext.apply_pii_precedence(c, Detection(T.INTEGER, subtype="discrete", source="native"))
    assert out.type is T.ZIP
    c = ctx(["Tavish", "Dagny"] * 5, name="first_name")
    out = dtext.apply_pii_precedence(c, Detection(T.FACTOR, source="native"))
    assert out.type is T.PERSON_NAME
    c = ctx(["1.5", "2.5"] * 5, name="score")
    native = Detection(T.FLOAT, source="native")
    assert dtext.apply_pii_precedence(c, native) is native
    assert detect(["02134"] * 3, name="zip").type is T.ZIP  # would be CONSTANT: small-cell policy applies


def test_d12_free_text_hints_beat_the_factor_rule() -> None:
    d = detect(["Peanut allergy", "Asthma", ""] * 10 + ["Seizure plan on file"], name="Medical Alert")
    assert d.type is T.FREE_TEXT and d.policy() == "typed"
    d = detect(["call home", "ok", "ok"], name="teacher_comments")
    assert d.type is T.FREE_TEXT


# --------------------------------------------------------------------------- MULTI_SELECT
def test_multi_select_combinations_and_items() -> None:
    vals = ["ELL", "SPED", "ELL;SPED", "GT", "504;SPED"] * 6 + [""] * 10
    d = detect(vals, name="programs")
    assert d.type is T.MULTI_SELECT and d.params["mode"] == "combinations" and d.params["delimiter"] == ";"
    items = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate"]
    combos = ["|".join(items[i:i + k]) for i in range(8) for k in (1, 2, 3)]
    d = detect(combos * 2, name="activities")
    assert d.type is T.MULTI_SELECT and d.params["mode"] == "items" and d.params["n_items"] == 8


def test_multi_select_threshold_and_guards() -> None:
    few = ["ELL;SPED"] * 2 + ["ELL"] * 5 + ["SPED"] * 3  # 20% < 30%
    assert detect(few, name="programs").type is T.FACTOR
    assert detect(["ELL;SPED"] * 3 + ["ELL"] * 7, name="programs").type is T.MULTI_SELECT  # 30%
    names = [f"Smith{i}, John{i}" for i in range(40)]  # many distinct items: not a multi-select
    assert detect(names, name="x").type is not T.MULTI_SELECT
    assert detect(["Good job, keep going", "Late, missing work"] * 5, name="Teacher Notes").type is T.FREE_TEXT
    assert detect(["1,234", "12,500", "3,000"] * 5, name="amount").type is not T.MULTI_SELECT
    assert detect(["3,5", "4,25", "1,0"] * 5, name="score", decimal=",").type is not T.MULTI_SELECT
    long_items = ["a very long item made of many many words;another very long item made of words"] * 10
    assert detect(long_items, name="x").type is not T.MULTI_SELECT


def test_state_and_multi_select_follow_the_d35_allowlist() -> None:
    # D35 replaced the small-file warning: states are a vocabulary (kept whatever their count); multi-select
    # items await the allowlist decision of the copy gate (rare non-vocabulary items get pseudonyms)
    d = detect(["TX"] * 20 + ["OK"] * 2 + ["LA"], name="state")
    assert d.type is T.STATE and d.params[dtext.COPY_GATE_KEY] == "vocabulary"
    assert not d.params.get(dtext.PENDING_KEY) and not [m for m in d.messages if m.code == "small_file_levels"]
    d = detect(["ELL;SPED"] * 10 + ["ELL"] * 10 + ["GT;ELL"], name="programs")
    assert d.type is T.MULTI_SELECT and d.params.get(dtext.PENDING_KEY)
    assert not [m for m in d.messages if m.code == "small_file_levels"]


# --------------------------------------------------------------------------- LAT / LON
def test_lat_lon_need_a_hint_floats_and_range() -> None:
    lat = [f"{32.6 + i / 1000:.6f}" for i in range(50)]
    lon = [f"{-97.0 + i / 1000:.6f}" for i in range(50)]
    d = detect(lat, name="home_lat")
    assert d.type is T.LAT and d.params["decimals"] == 6
    assert any(m.code == "lat_lon_independent" and m.level == "warning" for m in d.messages)
    assert detect(lon, name="Longitude").type is T.LON
    assert detect(lon, name="lng").type is T.LON
    assert detect(lon, name="geo_long").type is T.LON
    assert detect(lat, name="long_jump").type is T.FLOAT  # 'long' only as the last token
    assert detect([f"{100 + i / 10:.1f}" for i in range(50)], name="latitude").type is T.FLOAT  # out of range
    assert detect(lat, name="score").type is T.FLOAT


# --------------------------------------------------------------------------- overflow text (D13)
def test_overflow_text_codes_labels_and_free_text() -> None:
    codes = [f"{chr(65 + i % 8)}-{1000 + i}" for i in range(60)]
    d = detect(codes, name="locker")
    assert (d.type, d.subtype) == (T.HIGH_CARD_CATEGORICAL, "code") and d.policy() == "typed"
    labels = [f"career number {chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(30)]
    d = detect(labels, name="career")
    assert (d.type, d.subtype) == (T.HIGH_CARD_CATEGORICAL, None)
    long = [f"this is a fairly long sentence number {i}" for i in range(25)]
    assert detect(long, name="thing").type is T.FREE_TEXT
    d = detect(["a", "b", "a"], name="x", force_text=["x"])
    assert d.type is T.HIGH_CARD_CATEGORICAL and d.source == "force-text"


def test_pii_col_subtypes() -> None:
    d = detect(["Room 1", "Room 2"] * 5, name="homeroom", pii_col=["homeroom=last"])
    assert (d.type, d.subtype, d.source) == (T.PERSON_NAME, "last", "pii-col") and d.policy() == "typed"
    assert detect(["x"] * 3, name="c", pii_col=["c=postal-code"]).type is T.ZIP
    assert detect(["x"] * 3, name="c", pii_col=["c=ip_address"]).type is T.IP
    with pytest.raises(UsageError):
        detect(["x"], name="c", pii_col=["c=bogus"])


# --------------------------------------------------------------------------- report-safe statistics
def _stats_text(values: list, name: str, **cli) -> tuple[dict, str]:
    p = profile(values, name, **cli)
    return p.stats, json.dumps(p.stats, ensure_ascii=False)


def test_text_stats_never_contain_values() -> None:
    cases = {
        "student_name": NAMES * 2,
        "parent_email": EMAILS,
        "parent_phone": PHONES,
        "ssn": [f"{100 + i:03d}-{10 + i % 80:02d}-{1000 + i:04d}" for i in range(40)],
        "home_address": ADDRESSES,
        "comments": [f"Spoke with Mrs. Thistlewood about item {i}; follow up next week please." for i in range(30)],
        "city": ["Dallas", "Fort Worth", "St. Louis"] * 4,
    }
    for name, vals in cases.items():
        st, text = _stats_text(vals, name)
        for v in set(vals):
            assert v not in text, (name, v)
        for word in ("Thistlewood", "Okonkwo", "Dallas", "Worth", "Pecan"):
            assert word not in text, (name, word)


def test_text_stats_keep_only_entries_held_by_min_cell_rows() -> None:
    emails = [f"a{i}@big.org" for i in range(30)] + ["solo@private-family.net"] + [f"b{i}@mid.com" for i in range(9)]
    st, text = _stats_text(emails, "email")
    # D21: real domains are replaced in the output, so the report never names them (counts only)
    assert st["domains"] == {} and st["domain_count_profile"] == [30]
    assert st["domains_below_min_cell"] == 2 and st["n_domains"] == 3
    assert "private-family" not in text and "mid.com" not in text and "big.org" not in text
    st, _ = _stats_text(emails, "email", min_cell=5)
    assert st["domain_count_profile"] == [30, 9] and st["domains_below_min_cell"] == 1
    st, _ = _stats_text([f"x{i}@example.org" for i in range(12)] + ["y@d4.example.org"] * 10, "email")
    assert st["domains"] == {"example.org": 12, "d4.example.org": 10}  # reserved names (the synthetic side)
    phones = [f"(214) 3{i:02d}-{1000 + i}" for i in range(20)] + ["(817) 301-2222"]
    st, text = _stats_text(phones, "phone")
    assert st["area_codes"] == {"214": 20} and st["area_codes_below_min_cell"] == 1 and "817" not in text
    st, text = _stats_text(["75201"] * 12 + ["02134"] * 2 + ["75201-1234"], "zip")
    assert st["zip5_held_by_min_cell"] == 1 and st["zip5_count_profile"] == [13]
    assert st["zip5_below_min_cell"] == 1 and st["plus4_share"] > 0
    assert "75201" not in text and "02134" not in text  # ZIP codes are not label-safe (D21)
    st, text = _stats_text(["TX"] * 20 + ["OK"] * 3, "state")
    assert st["n_states"] == 2 and st["state_count_profile"] == [20] and '"TX"' not in text


def test_text_stats_shapes() -> None:
    vals = ["O'Brien, José", "Smith-Jones, Mary", "LEE, ANN"] * 10
    st, _ = _stats_text(vals, "student_name")
    shape = st["text_shape"]
    assert shape["shares"]["apostrophe"] == pytest.approx(1 / 3, abs=1e-6)
    assert shape["shares"]["hyphen"] == pytest.approx(1 / 3, abs=1e-6)
    assert shape["shares"]["non_ascii"] == pytest.approx(1 / 3, abs=1e-6)
    assert shape["case"]["upper"] == pytest.approx(1 / 3, abs=1e-6)
    assert st["layouts"] == {"last_comma_first": 1.0}
    st, _ = _stats_text(["Line one.\nLine 2", "plain words here ok"] * 10, "notes")
    assert st["text_shape"]["shares"]["newline"] == 0.5 and st["text_shape"]["shares"]["digit"] == 0.5


def test_local_mask_and_case_style() -> None:
    assert dtext.local_mask("John.Smith42") == "Aa.Aa99"
    assert dtext.local_mask("josé_99") == "a_99"
    assert [dtext.case_style(s) for s in ("ABC", "abc", "Abc Def", "aBc", "123")] == [
        "upper", "lower", "title", "mixed", "none"]


def test_doc_fields() -> None:
    zp = ColumnProfile(index=0, name="zip", type=T.ZIP)
    f = dtext.doc_fields(zp, {"plus4_share": 0.1, "zip5_widths": {"5": 10}})
    assert f["readr"] == "c" and f["format"] == "99999-9999"
    lat = ColumnProfile(index=0, name="lat", type=T.LAT)
    f = dtext.doc_fields(lat, {"min": 32.6, "max": 33.0})
    assert f["readr"] == "d" and f["bounding_box"] == [32.6, 33.0]
    em = ColumnProfile(index=0, name="e", type=T.EMAIL)
    assert dtext.doc_fields(em, {"domains": {"example.org": 20, "example.com": 12}})["domains"] == [
        "example.com", "example.org"]
    ms = ColumnProfile(index=0, name="p", type=T.MULTI_SELECT, params={"delimiter": ";"})
    assert dtext.doc_fields(ms, {"n_items": 4}) == {"note": "delimited multi-select items", "delimiter": ";",
                                                     "n_items": 4}
    ph = ColumnProfile(index=0, name="p", type=T.PHONE)
    assert dtext.doc_fields(ph, {"masks": {"(999) 999-9999": 50}})["mask"] == "(999) 999-9999"
