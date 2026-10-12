"""AREA C generation: never-copy, distinct/frequency profiles, shapes, masks, reserved ranges (D12, D13, D24)."""

from __future__ import annotations

import ipaddress
import re
from collections import Counter

import numpy as np
import pytest

from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import text as dtext
from datanonymous.detect.ids import char_mask
from datanonymous.generate import pii, text as gtext
from datanonymous.generate.rng import faker_for, stream
from datanonymous.generate.table import generate_values
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType as T, RawColumn, RawTable
from datanonymous.profile.table import profile_table

NA = frozenset({"", "N/A", "NA", "NULL", "null", "None", "-", "--", ".", "NaN", "#N/A"})


def _arr(values: list) -> np.ndarray:
    a = np.empty(len(values), dtype=object)
    a[:] = list(values)
    return a


def profile(values: list, name: str, **cli) -> ColumnProfile:
    s = build_settings(cli)
    tp = profile_table(RawTable(None, [name], [RawColumn(name, _arr(values))], len(values)), "f.csv", "csv", s)
    return tp.columns[0]


def synth(values: list, name: str, seed: int = 1, **cli) -> tuple[list[str], dict, ColumnProfile]:
    """Synthetic non-missing values for one column (the table generator's call)."""
    p = profile(values, name, **cli)
    n = p.model.n_nonmissing
    out, flags = generate_values(p, p.model, n, stream(seed, "t", name), lambda: faker_for(seed, "t", name))
    return list(out), flags, p


def real_cf(values: list) -> set[str]:
    return {v.strip().casefold() for v in values if v.strip() not in NA}


def assert_never_real(values: list, syn: list) -> None:
    hits = real_cf(values) & {s.strip().casefold() for s in syn}
    assert not hits, sorted(hits)[:5]


def assert_same_profile(values: list, syn: list) -> None:
    """Same distinct count and frequency profile (sorted counts) over the non-missing raw values."""
    rc = Counter(v for v in values if v.strip() not in NA)
    sc = Counter(syn)
    assert sorted(rc.values()) == sorted(sc.values())


FIRST = ["Tavish", "Brennagh", "Xiomara", "Thaddeus", "Perpetua", "Lysander", "Ottoline", "Seraphina", "Ezekiel",
         "Ignatius", "Marisol", "Dagny", "Soledad", "Anouk", "Leopold"]
LAST = ["Okonkwo", "Abernathy", "Zielinski", "Thistlewood", "Lindqvist", "Haverbrook", "Ravensworth", "Kowalczyk",
        "Achterberg", "Fitzwilliam", "Oyelaran", "Szczepanski", "Delacroix", "Mbeki", "Quintanilla"]


def name_column(rng_seed: int = 3, n: int = 300) -> list[str]:
    r = np.random.default_rng(rng_seed)
    return [f"{LAST[i % 15]}, {FIRST[(i * 7 + int(r.integers(3))) % 15]}" for i in range(n)]


# --------------------------------------------------------------------------- PERSON_NAME
def test_names_never_real_and_keep_distinct_and_frequency_profile() -> None:
    vals = [FIRST[min(int(x), 14)] for x in np.random.default_rng(0).exponential(3, 400)]
    syn, flags, p = synth(vals, "first_name")
    assert p.type is T.PERSON_NAME and len(syn) == 400
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    assert all(re.fullmatch(r"[A-Z][a-z]+", s) for s in syn)
    assert flags.get("rare_category_adjusted") is False


def test_names_case_and_whitespace_variants_are_kept() -> None:
    vals = ["Okonkwo"] * 10 + ["OKONKWO"] * 3 + [" Okonkwo "] * 2 + ["mbeki"] * 4 + ["Mbeki"] * 6
    syn, _, _ = synth(vals, "last_name")
    c = Counter(syn)
    assert sorted(c.values()) == [2, 3, 4, 6, 10]
    upper = [s for s in c if s.isupper()]
    padded = [s for s in c if s != s.strip()]
    assert len(upper) == 1 and c[upper[0]] == 3
    assert len(padded) == 1 and padded[0].startswith(" ") and padded[0].endswith(" ") and c[padded[0]] == 2
    assert len({s.strip().casefold() for s in c}) == 2  # two real names -> two synthetic names
    assert_never_real(vals, syn)


def test_name_layouts_titles_suffixes_and_particles() -> None:
    vals = (["O'Brien, Mary-Kate"] * 5 + ["Thistlewood, Anouk J."] * 4 + ["Mrs. Alvarez-Whitcomb"] * 6
            + ["Leopold Fitzwilliam Jr"] * 3 + ["de la Cruz, Soledad"] * 2 + ["Ezekiel M Okonkwo"] * 3)
    syn, _, p = synth(vals, "student_name")
    assert p.subtype == "full"
    assert_never_real(vals, syn)
    by_count = {c: s for s, c in Counter(syn).items()}
    assert re.fullmatch(r"O'[A-Z][a-z]+, [A-Z][a-z]+-[A-Z][a-z]+", by_count[5])
    assert re.fullmatch(r"[A-Z][a-z]+, [A-Z][a-z]+ [A-Z]\.", by_count[4])
    assert re.fullmatch(r"Mrs\. [A-Z][a-z]+-[A-Z][a-z]+", by_count[6])
    assert re.fullmatch(r"de la [A-Z][a-z]+, [A-Z][a-z]+", by_count[2])
    three = [s for s, c in Counter(syn).items() if c == 3]
    assert any(re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+ Jr", s) for s in three)
    assert any(re.fullmatch(r"[A-Z][a-z]+ [A-Z] [A-Z][a-z]+", s) for s in three)


def test_name_accents_are_kept_and_encodable() -> None:
    vals = ["José"] * 6 + ["Zoë"] * 4 + ["Renée"] * 3 + ["Marisol"] * 7
    syn, _, _ = synth(vals, "first_name")
    assert_never_real(vals, syn)
    accented = sum(1 for s in syn if not s.isascii())
    assert accented >= 6  # most accented values keep an accent (on a matching base letter)
    for s in syn:
        s.encode("cp1252")
        assert not re.search(r"[éë][^aeiouy\s]*[éë]", s)  # no doubled odd accents


def test_name_lengths_follow_the_real_ones() -> None:
    vals = [f"{FIRST[i % 15]}" for i in range(150)]
    syn, _, _ = synth(vals, "first_name")
    real_len = sorted(len(v) for v in vals)
    syn_len = sorted(len(v) for v in syn)
    assert abs(np.median(real_len) - np.median(syn_len)) <= 1
    assert abs(np.mean(real_len) - np.mean(syn_len)) <= 1.0


def test_exhausted_initials_are_lengthened_and_flagged() -> None:
    letters = [chr(65 + i) for i in range(26)]
    vals = letters * 3
    syn, flags, _ = synth(vals, "middle_initial")
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    assert flags.get("never_copy_adjusted") is True
    assert all(len(s) == 2 for s in syn)


def test_names_with_na_tokens_write_tokens_verbatim() -> None:
    p = profile(["Okonkwo", "N/A", "", "Mbeki", "N/A"] * 4, "middle_name")
    assert p.model.missing_counts == {"N/A": 8, "": 4}
    assert p.model.real_values == frozenset({"Okonkwo", "Mbeki"})


# --------------------------------------------------------------------------- EMAIL
def test_emails_reserved_domains_keep_domain_count_shares_and_local_masks() -> None:
    vals = ([f"t.okonkwo{i}@gmail.com" for i in range(40)] + [f"jzielinski{i}@yahoo.com" for i in range(25)]
            + [f"Anouk.Mbeki@DISTRICT.ORG"] * 3 + ["solo_99@family-net.net"] + [f"x{i}@school.edu" for i in range(5)])
    syn, _, p = synth(vals, "parent_email")
    assert p.type is T.EMAIL
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    domains = Counter(s.rsplit("@", 1)[1] for s in syn)
    assert set(d.casefold() for d in domains) <= {"example.org", "example.com", "example.net", "d4.example.org",
                                                  "d5.example.com"}
    assert sorted(domains.values()) == [1, 3, 5, 25, 40]  # 5 distinct domains with the same shares
    assert domains["example.org"] == 40 and domains["example.com"] == 25
    assert domains["D4.EXAMPLE.ORG".lower()] == 0 and any(d.isupper() for d in domains)
    assert Counter(dtext.local_mask(s.rsplit("@", 1)[0]) for s in syn) == Counter(
        dtext.local_mask(v.rsplit("@", 1)[0]) for v in vals)
    assert Counter(char_mask(s.rsplit("@", 1)[0]) for s in syn) == Counter(
        char_mask(v.rsplit("@", 1)[0]) for v in vals)


# --------------------------------------------------------------------------- PHONE
def test_phones_keep_masks_floor_area_codes_and_use_ordinary_exchanges() -> None:
    vals = ([f"(214) {300 + i}-{1000 + i:04d}" for i in range(30)] + [f"972-{400 + i}-{2000 + i:04d}" for i in range(12)]
            + ["(817) 301-2222", "940.302.3333 x12", "+1 469 303 4444", "331-5555", "N/A"])
    syn, _, p = synth(vals, "parent_phone")
    assert p.type is T.PHONE and len(syn) == len(vals) - 1  # N/A is a missing token
    assert_never_real(vals, syn)
    assert Counter(char_mask(s) for s in syn) == Counter(char_mask(v) for v in vals if v != "N/A")
    areas = Counter()
    for s in syn:
        digits = re.sub(r"\D", "", s.split(" x")[0])
        if len(digits) == 11:
            digits = digits[1:]
        if len(digits) == 10:
            assert digits[3] in "23456789" and digits[3:6] not in pii._N11, s
            areas[digits[:3]] += 1
        else:
            assert digits[0] in "23456789" and digits[:3] not in pii._N11, s
    # k-floored pmf (design 3 #27): 214/972 are held by >= min-cell rows and kept; the 3 rarer codes
    # (817, 940, 469) are redrawn from the kept codes, so only kept codes appear
    assert set(areas) == {"214", "972"} and sum(areas.values()) == 45
    assert areas["214"] >= 30 and areas["972"] >= 12
    syn2, _, _ = synth(vals[42:], "parent_phone")  # no code reaches min-cell: invented, never a real code
    codes = {re.sub(r"\D", "", s.split(" x")[0])[-10:][:3] for s in syn2 if len(re.sub(r"\D", "", s)) >= 10}
    assert len(codes) == 3 and not codes & {"817", "940", "469"}
    assert any(s.startswith("+1 ") for s in syn) and any(re.search(r" x\d\d$", s) for s in syn)


# --------------------------------------------------------------------------- SSN
def test_ssn_keeps_mask_and_uses_the_invalid_area_range() -> None:
    vals = [f"{100 + i:03d}-{10 + i % 80:02d}-{1000 + i:04d}" for i in range(30)] + [f"{234567890 + i}" for i in
                                                                                       range(5)] + ["XXX-XX-4321"]
    syn, _, p = synth(vals, "ssn")
    assert p.type is T.SSN
    assert_never_real(vals, syn)
    assert Counter(char_mask(s) for s in syn) == Counter(char_mask(v) for v in vals)
    for s in syn:
        d = re.sub(r"\D", "", s)
        if len(d) == 9:
            assert d[0] == "9" and d[3:5] != "00" and d[5:] != "0000"
        else:
            assert s.startswith("XXX-XX-")


# --------------------------------------------------------------------------- ZIP / STATE / CITY
def test_zip_small_cell_policy_leading_zeros_and_plus4() -> None:
    vals = (["75201"] * 20 + ["07030"] * 12 + ["02134"] * 3 + ["75203"] + [f"75201-{1000 + i}" for i in range(6)]
            + ["7030"] * 2)
    syn, flags, p = synth(vals, "zip")
    assert p.type is T.ZIP and len(syn) == len(vals)
    z5 = Counter(re.match(r"\d+", s).group() for s in syn)
    assert z5["75201"] == 26 and z5["07030"] == 12  # kept (>= min-cell rows) with exact counts
    assert z5["02134"] == 0 and z5["75203"] == 0 and z5["7030"] == 0  # rare codes replaced
    assert sorted(len(z) for z in z5.elements()) == sorted(len(re.match(r"\d+", v).group()) for v in vals)
    plus4 = [s for s in syn if "-" in s]
    assert len(plus4) == 6 and not set(plus4) & set(vals)
    assert all(s.startswith("75201-") for s in plus4)
    invented = set(z5) - {"75201", "07030"}
    assert len(invented) == 3 and not invented & {v[:5] for v in vals} | {"7030"} & invented
    assert flags["rare_category_adjusted"] is False
    syn, _, _ = synth(vals, "zip", min_cell=2)
    assert Counter(syn)["02134"] == 3 and Counter(syn)["7030"] == 2  # a smaller k keeps them


def test_state_exact_quota() -> None:
    vals = ["TX"] * 30 + ["OK"] * 3 + ["LA"] + ["Texas"] * 2
    syn, flags, p = synth(vals, "state")
    assert p.type is T.STATE and Counter(syn) == Counter(vals) and flags["rare_category_adjusted"] is False


def test_city_fake_names_keep_distinct_counts() -> None:
    vals = ["Dallas"] * 20 + ["Fort Worth"] * 8 + ["GRAND PRAIRIE"] * 4 + ["Irving"] * 2 + ["Cedar Hill"]
    syn, _, p = synth(vals, "city")
    assert p.type is T.CITY
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    c = Counter(syn)
    upper = [s for s in c if s.isupper()]
    assert len(upper) == 1 and c[upper[0]] == 4


# --------------------------------------------------------------------------- STREET_ADDRESS
def test_addresses_keep_suffix_unit_and_house_number_ranges() -> None:
    r = np.random.default_rng(2)
    vals = []
    for i in range(80):
        num = int(r.integers(10, 9000))
        suffix = ["St", "Ave", "Dr", "Blvd"][i % 4]
        unit = f" Apt {i % 9}B" if i % 5 == 0 else ""
        vals.append(f"{num} {'N ' if i % 7 == 0 else ''}Pecan Hollow {suffix}{unit}")
    vals += ["PO Box 1234", "77 Martin Luther King Jr Blvd"]
    syn, _, p = synth(vals, "home_address")
    assert p.type is T.STREET_ADDRESS
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)

    def parts(v: str):
        m = dtext.ADDRESS_RE.fullmatch(v)
        return (m.group("suffix"), "Apt" in v, v.split()[1] == "N") if m else None

    assert Counter(parts(v) for v in vals) == Counter(parts(s) for s in syn)
    real_nums = [int(v.split()[0]) for v in vals if v[0].isdigit()]
    syn_nums = [int(s.split()[0]) for s in syn if s[0].isdigit()]
    assert min(real_nums) <= min(syn_nums) and max(syn_nums) <= max(real_nums)
    assert any(re.fullmatch(r"PO Box \d{4}", s) for s in syn)
    assert any(re.fullmatch(r"\d+ [A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ Jr Blvd", s) for s in syn)
    assert not any("Pecan" in s or "Hollow" in s for s in syn)


# --------------------------------------------------------------------------- IP / URL
def test_ip_addresses_use_reserved_ranges() -> None:
    vals = [f"10.0.{i}.{i + 1}" for i in range(50)] + ["10.0.0.1"] * 4 + ["2600:1700:abcd::1", "fe80::1"]
    syn, _, p = synth(vals, "last_ip")
    assert p.type is T.IP
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    nets = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "198.18.0.0/15",
                                              "2001:db8::/32")]
    for s in syn:
        a = ipaddress.ip_address(s)
        assert any(a in n for n in nets if n.version == a.version), s


def test_urls_use_reserved_hosts_and_keep_scheme_and_shape() -> None:
    vals = ([f"https://sites.google.com/view/user{i}" for i in range(12)] + [f"https://www.okonkwofamily.net/p{i}"
                                                                             for i in range(3)]
            + ["http://github.com/tavish-okonkwo"])
    syn, _, p = synth(vals, "portfolio_url")
    assert p.type is T.URL
    assert_never_real(vals, syn)
    hosts = Counter(re.match(r"https?://([^/]+)", s).group(1) for s in syn)
    assert hosts == Counter({"example.org": 12, "www.example.com": 3, "example.net": 1})
    assert sorted(len(s) - len(h) for s, h in ((s, re.match(r"https?://([^/]+)", s).group(1)) for s in syn)) == \
        sorted(len(v) - len(re.match(r"https?://([^/]+)", v).group(1)) for v in vals)
    assert not any("okonkwo" in s or "user" in s for s in syn)


# --------------------------------------------------------------------------- MULTI_SELECT
def test_multi_select_combinations_exact() -> None:
    vals = ["ELL"] * 9 + ["SPED"] * 7 + ["ELL;SPED"] * 6 + ["GT"] * 3 + ["504;SPED"] * 2 + ["ELL;GT"]
    syn, flags, p = synth(vals, "programs")
    assert p.type is T.MULTI_SELECT and Counter(syn) == Counter(vals) and flags == {"rare_category_adjusted": False}


def test_multi_select_items_mode() -> None:
    items = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate"]
    r = np.random.default_rng(4)
    vals = []
    for _ in range(300):
        k = int(r.integers(1, 5))
        pick = sorted(r.choice(8, size=k, replace=False).tolist())
        vals.append(" | ".join(items[i] for i in pick))
    syn, flags, p = synth(vals, "activities")
    assert p.type is T.MULTI_SELECT and p.params["mode"] == "items"
    split = [[x.strip() for x in s.split("|")] for s in syn]
    assert all(" | " in s for s in syn if "|" in s)  # the observed separator form
    assert Counter(len(x) for x in split) == Counter(len(v.split("|")) for v in vals)  # items per value exact
    assert {i for x in split for i in x} == set(items)  # every item appears (floor)
    real_n = Counter(i.strip() for v in vals for i in v.split("|"))
    syn_n = Counter(i for x in split for i in x)
    for it in items:
        assert abs(real_n[it] - syn_n[it]) <= 0.1 * real_n[it] + 3, it
    for x in split:  # observed order (alphabetical by the source's item order)
        assert x == sorted(x, key=items.index)


# --------------------------------------------------------------------------- FREE_TEXT / HIGH_CARD
def test_free_text_keeps_shape_and_never_real_words() -> None:
    vals = ([f"Spoke with Mrs. Thistlewood about attendance on {i}/12." for i in range(10)]
            + ["Called home.\nNo answer, left voicemail."] * 4 + ["Excellent participation this week."] * 6
            + ["IEP review with José Okonkwo and counselor."] * 2)
    syn, _, p = synth(vals, "comments")
    assert p.type is T.FREE_TEXT
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    assert sorted(len(s) for s in syn) == sorted(len(v) for v in vals)
    assert sorted(len(s.split()) for s in syn) == sorted(len(v.split()) for v in vals)
    assert sum("\n" in s for s in syn) == 4
    assert sum(bool(re.search(r"\d", s)) for s in syn) == 10
    assert sum(not s.isascii() for s in syn) == 2
    words = {w.casefold() for v in vals for w in re.findall(r"[^\W\d_]{4,}", v)}
    syn_words = {w.casefold() for s in syn for w in re.findall(r"[^\W\d_]{4,}", s)}
    assert not words & syn_words


def test_high_card_labels_keep_lengths_and_skew() -> None:
    careers = ["Veterinarian", "Marine Biologist", "Electrician", "Pediatric Nurse", "Architect", "Game Designer",
               "Pharmacist", "Firefighter", "Paralegal", "Astronomer", "Welder", "Chef", "Physical Therapist",
               "Translator", "Pilot", "Software Developer", "Dental Hygienist", "Journalist", "Carpenter",
               "Geologist", "Social Worker", "Accountant", "Photographer", "Zookeeper", "Undecided"]
    vals = [c for i, c in enumerate(careers) for _ in range(30 - i)]
    syn, _, p = synth(vals, "career_goal")
    assert p.type is T.HIGH_CARD_CATEGORICAL
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    assert sorted(len(s) for s in set(syn)) == sorted(len(c) for c in careers)
    assert all(re.fullmatch(r"[a-z_]+_\d+", s) for s in syn)
    assert {s.split("_")[0] for s in syn} <= {"c", "ca", "car", "care", "caree", "career"}


def test_high_card_code_masks() -> None:
    vals = [f"{chr(65 + i % 8)}-{1000 + 37 * i:04d}" for i in range(60)] + ["A-1000"] * 3
    syn, _, p = synth(vals, "locker")
    assert p.subtype == "code"
    assert_never_real(vals, syn)
    assert_same_profile(vals, syn)
    assert Counter(char_mask(s) for s in syn) == Counter(char_mask(v) for v in vals)


# --------------------------------------------------------------------------- determinism and robustness
def test_same_seed_same_values_other_seed_other_values() -> None:
    vals = name_column()
    a, _, _ = synth(vals, "student_name", seed=5)
    b, _, _ = synth(vals, "student_name", seed=5)
    c, _, _ = synth(vals, "student_name", seed=6)
    assert a == b and a != c


def test_generators_without_a_faker_factory() -> None:
    p = profile(["Okonkwo", "Mbeki", "Okonkwo"], "last_name")
    out, _ = pii.generate_pii(p, p.model, 3, np.random.default_rng(1), lambda: None)
    assert len(out) == 3 and not {"okonkwo", "mbeki"} & {s.casefold() for s in out}


@pytest.mark.parametrize("t", [T.EMAIL, T.PERSON_NAME, T.PHONE, T.SSN, T.ZIP, T.CITY, T.STREET_ADDRESS, T.IP, T.URL])
def test_forced_pii_types_handle_any_values(t: T) -> None:
    vals = ["alpha 1", "Beta-22", "gamma", "Beta-22", "N/A", "#7"]
    p = ColumnProfile(index=0, name="x", type=t, params={"_min_cell": 10})
    p.model = ColumnModel(nonmissing_counts=dict(Counter(vals)), real_values=frozenset(vals))
    out, _ = pii.generate_pii(p, p.model, len(vals), np.random.default_rng(1), lambda: faker_for(1, "x"))
    assert len(out) == len(vals)
    if t is not T.ZIP:
        assert not {v.casefold() for v in vals} & {s.casefold() for s in out}


def test_generate_text_extended_dispatch() -> None:
    model = ColumnModel(nonmissing_counts={"aa-1": 2, "bb-2": 1}, real_values=frozenset({"aa-1", "bb-2"}))
    p = ColumnProfile(index=0, name="c", type=T.HIGH_CARD_CATEGORICAL, subtype="code")
    out, flags = gtext.generate_text_extended(p, model, 3, np.random.default_rng(0))
    assert sorted(Counter(out.tolist()).values()) == [1, 2] and flags["rare_category_adjusted"] is False
    p = ColumnProfile(index=0, name="c", type=T.FREE_TEXT)
    out, _ = gtext.generate_text_extended(p, model, 3, np.random.default_rng(0))
    assert len(out) == 3
