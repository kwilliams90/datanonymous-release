"""AREA C hardening: regression tests for the adversarial review (one test per issue found).

Each test names the defect it pins down. Inputs are in-memory tables or the deterministic
``tests/fixtures/text/adversarial_crlf.csv`` (UTF-8, CRLF; written by ``make_fixtures.py``).
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import run_cli
from conftest import build_fidelity_settings as build_settings
from datanonymous.detect import text as dtext
from datanonymous.generate import text as gtext
from datanonymous.generate.rng import faker_for, stream
from datanonymous.generate.table import generate_values
from datanonymous.model import ColumnProfile, ColumnType as T, RawColumn, RawTable
from datanonymous.profile.table import profile_table

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "text"
ADV = FIXTURES / "adversarial_crlf.csv"
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
    p = profile(values, name, **cli)
    n = p.model.n_nonmissing
    out, flags = generate_values(p, p.model, n, stream(seed, "t", name), lambda: faker_for(seed, "t", name))
    return list(out), flags, p


def _cf(values) -> set[str]:
    return {v.strip().casefold() for v in values if v.strip() not in NA}


# --------------------------------------------------------------------------- forced STATE copied any text
def test_pii_col_state_masks_values_that_are_not_states() -> None:
    """--pii-col COL=state on other text wrote every real value verbatim (only state labels are kept)."""
    vals = ["Lives with grandmother", "TX", "Custody: father only", "OK", "N/A"] * 4
    out, _, p = synth(vals, "notes", pii_col=["notes=state"])
    assert p.type is T.STATE
    real_other = _cf(v for v in vals if not dtext.is_state(v))
    assert not real_other & _cf(out)
    assert Counter(o for o in out if o in ("TX", "OK")) == Counter({"TX": 4, "OK": 4})  # states kept exactly
    assert Counter(out)["N/A"] == 4  # NA tokens are levels of a STATE column: verbatim
    masked = [o for o in out if o not in ("TX", "OK", "N/A")]
    assert len(masked) == 8 and all(len(m) in (22, 20) for m in masked)  # same-shape masks


# --------------------------------------------------------------------------- numbers under PII headers
def test_numbers_under_phone_or_ssn_headers_stay_numbers() -> None:
    """The D12 guard turned numeric columns (phone_ext, ssn_last4) into letter labels ('j', 's_02')."""
    r = random.Random(3)
    ext = [str(r.randrange(100, 999)) for _ in range(80)]
    last4 = [str(r.randrange(1000, 9999)) for _ in range(80)]
    for name, vals in (("phone_ext", ext), ("ssn_last4", last4)):
        out, _, p = synth(vals, name)
        assert (p.type, p.subtype) == (T.HIGH_CARD_CATEGORICAL, "code"), name
        assert p.params["numeric_storage"] == "INTEGER" and p.params["pii_hint"] in ("phone", "ssn")
        assert all(re.fullmatch(r"[1-9]\d*", o) and len(o) == len(vals[0]) for o in out), name
        assert not _cf(vals) & _cf(out), name
        assert sorted(Counter(out).values()) == sorted(Counter(vals).values())  # frequency profile kept
        doc = dtext.doc_fields(p, {"length": {"max": 4}})
        assert doc["readr"] == "i" and "numbers" in doc["note"]
    out, _, p = synth(["-12.50", "3.75", "0.25", "-1.00"] * 3, "ssn_amount")
    assert p.params["numeric_storage"] == "FLOAT"
    assert all(re.fullmatch(r"-?\d+\.\d\d", o) for o in out) and dtext.doc_fields(p, {})["readr"] == "d"


def test_code_like_text_under_a_pii_header_gets_masks() -> None:
    """Short codes under a name header (usernames <= factor-max) became 'username_01' labels, not masks."""
    vals = ["jsmith42", "mlee7", "aking19", "bwu3"] * 5
    out, _, p = synth(vals, "username")
    assert (p.type, p.subtype) == (T.HIGH_CARD_CATEGORICAL, "code")
    assert all(re.fullmatch(r"[a-z]+\d+", o) for o in out) and not _cf(vals) & _cf(out)
    out, _, p = synth(["Y", "N", "Unknown"] * 5, "email_opt_in")  # not code-like: invented labels as before
    assert p.subtype is None and not _cf(["Y", "N", "Unknown"]) & _cf(out)


def test_mask_fill_keeps_a_nonzero_leading_digit() -> None:
    pyr = random.Random(0)
    for _ in range(300):
        assert gtext.mask_fill("5", pyr) != "0"
        assert re.fullmatch(r"[1-9]\d\d-[A-Z]{2}", gtext.mask_fill("123-AB", pyr))
        assert re.fullmatch(r"-[1-9]\d", gtext.digit_mask("-42", pyr))
        assert re.fullmatch(r"0\d\d", gtext.mask_fill("012", pyr))  # a leading zero stays where it was
        sci = gtext.digit_mask("1e-05", pyr)
        assert re.fullmatch(r"[1-9]e-\d\d", sci) and float(sci) > 0  # only digits change: still a number


# --------------------------------------------------------------------------- --force-type on AREA C types
def test_force_type_pii_columns_follow_min_cell_and_the_na_policy() -> None:
    """--force-type COL=PHONE ignored --min-cell (area codes held by 10-99 rows kept and listed with
    --min-cell 100) and turned the NA token 'N/A' into a masked value instead of writing it verbatim."""
    vals = [f"(214) 3{i % 90:02d}-{1000 + i}" for i in range(150)] + [f"(972) 4{i:02d}-{2000 + i}" for i in range(60)]
    vals += ["N/A"] * 5
    for cli in ({"min_cell": 100}, {"min_cell": 100, "force_type": ["parent_phone=PHONE"]}):
        out, _, p = synth(vals, "parent_phone", **cli)
        assert p.type is T.PHONE and p.missing_policy == "typed", cli
        assert p.params["_min_cell"] == 100 and p.stats["area_codes"] == {"214": 150}, cli
        assert len(out) == 210 and not any(o.startswith("(972)") for o in out), cli  # 60 rows < k: redrawn
    _, _, p = synth(["ELL;SPED", "GT", "ELL", "SPED;GT;504"] * 4, "programs", factor_max=3,
                    force_type=["programs=MULTI_SELECT"])
    assert p.params["_factor_max"] == 3


# --------------------------------------------------------------------------- MULTI_SELECT item mode
def test_multi_select_items_keep_na_tokens_whole() -> None:
    """Item-wise MULTI_SELECT treated 'None' as an item and wrote 'Choir;Robotics;None'."""
    r = random.Random(5)
    items = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate", "Math Club"]
    vals = []
    for _ in range(300):
        k = r.choice([0, 1, 1, 2, 2, 3, 4])
        vals.append(r.choice(["None", "N/A", ";"]) if k == 0 else ";".join(sorted(r.sample(items, k),
                                                                              key=items.index)))
    out, _, p = synth(vals, "activities")
    assert p.type is T.MULTI_SELECT and p.params["mode"] == "items" and p.params["n_items"] == len(items)
    for tok in ("None", "N/A", ";"):
        assert Counter(out)[tok] == Counter(vals)[tok], tok  # whole values at their exact counts
    assert not any(("None" in o or "N/A" in o) and o not in ("None", "N/A") for o in out)
    st = dtext.text_stats([v for v in vals], p)
    assert st["n_items"] == len(items) and st["items_per_value"]["0"] == sum(Counter(vals)[t] for t in
                                                                           ("None", "N/A", ";"))


def test_multi_select_items_keep_separator_and_trailing_forms() -> None:
    """Item-wise values were all joined with the single most common separator and lost trailing ';'."""
    r = random.Random(8)
    items = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate"]
    vals = []
    for _ in range(400):
        k = r.choice([1, 2, 2, 3, 4])
        v = r.choice([";", "; "]).join(sorted(r.sample(items, k), key=items.index))
        vals.append(v + (";" if r.random() < 0.1 else ""))
    out, _, p = synth(vals, "activities")
    assert p.params["mode"] == "items"

    def forms(vs: list[str]) -> tuple[int, int, int]:
        multi = [v.rstrip(";") for v in vs if v.rstrip(";").count(";") >= 1]
        return sum("; " in v for v in multi), sum("; " not in v for v in multi), sum(v.endswith(";") for v in vs)

    assert forms(out) == forms(vals)  # '; ' vs ';' between items and trailing ';' kept by exact quota


def test_multi_select_items_mode_pseudonymises_rare_items() -> None:
    """Items are written like factor levels, so they follow the D35 allowlist: vocabulary items (ELL, SPED, GT,
    504, Band, Choir, Art, Drama) are kept whatever their count; Chess (5 rows) and Robotics (1 row) get
    pseudonyms; the message gives counts only."""
    base = ["ELL;SPED", "GT;504", "ELL", "SPED;GT", "504;ELL;GT"]
    vals = [f"{a};{b}" for a in base for b in ("Band", "Choir", "Art", "Chess", "Drama")] + ["ELL;Robotics"]
    p = profile(vals, "programs")
    assert p.type is T.MULTI_SELECT and p.params["mode"] == "items"
    assert set(dtext.pseudonymized(p)) == {"Chess", "Robotics"}
    msgs = [m for m in p.messages if m.code == "pseudonymized_levels"]
    assert msgs and "2 of 10 item(s)" in msgs[0].text and "Robotics" not in msgs[0].text


# --------------------------------------------------------------------------- report: domains / hosts
def test_report_never_names_real_email_domains_or_url_hosts() -> None:
    """Real e-mail domains / URL hosts (not reproduced in the output) were listed in the report (D21)."""
    emails = [f"u{i}@lakeview-isd.org" for i in range(20)] + [f"p{i}@gmail.com" for i in range(12)]
    p = profile(emails, "parent_email")
    text = json.dumps(p.stats)
    assert "lakeview" not in text and "gmail" not in text
    assert p.stats["domain_count_profile"] == [20, 12] and p.stats["n_domains"] == 2
    urls = [f"https://www.smithfamily.net/p{i}" for i in range(15)]
    p = profile(urls, "portfolio_url")
    assert "smithfamily" not in json.dumps(p.stats) and p.stats["host_count_profile"] == [15]
    assert dtext.is_reserved_domain("d4.example.org") and not dtext.is_reserved_domain("example.org.evil.com")


# --------------------------------------------------------------------------- huge / odd values
def test_address_regex_is_linear_on_long_whitespace_runs() -> None:
    """'1 a' + many spaces + 'x' took ~1 s per 3,000 spaces (quadratic backtracking)."""
    t0 = time.perf_counter()
    assert dtext.ADDRESS_RE.fullmatch("1 a" + " " * 50_000 + "x") is None
    assert time.perf_counter() - t0 < 1.0
    m = dtext.ADDRESS_RE.fullmatch("123   Cedar  Ridge   Dr.,  Apt 4")
    assert m and (m.group("body"), m.group("suffix"), m.group("dot")) == ("Cedar  Ridge", "Dr", ".")


# --------------------------------------------------------------------------- names: couples, two hints
def test_couples_in_a_name_column_are_names() -> None:
    """'John & Mary Smith' / 'Ana / Luis Garcia' failed name_like, so the column became 'Guardian_01' labels."""
    vals = ["John & Mary Smith", "Ana / Luis García", "Mr. and Mrs. Lee", "Pat O'Neil"] * 6
    out, _, p = synth(vals, "Guardian")
    assert p.type is T.PERSON_NAME and not _cf(vals) & _cf(out)
    assert dtext.name_layout("John & Mary Smith") == "two_people"
    assert all(" & " in o for o in out if "&" in o) and sum("&" in o for o in out) == 6
    assert sum(" / " in o for o in out) == 6
    assert sum(o.startswith("Mr. and Mrs. ") for o in out) == 6  # title and connector kept


def test_full_names_avoid_the_columns_real_name_words() -> None:
    """'Maritza de la Cruz' became 'Mackenzie de la Cruz': a real surname of the column reappeared (Faker
    draws common names); fake name words now avoid the column's real name words (best effort)."""
    firsts = ["Maria", "James", "Linda", "David", "Susan", "Kevin", "Laura", "Brian"]
    lasts = ["Cruz", "Smith", "Jones", "Brown", "Davis", "Lopez", "Garcia", "Miller", "de la Cruz"]
    vals = [f"{f} {l}" for f in firsts for l in lasts] * 2
    out, _, p = synth(vals, "guardian_name", seed=4)
    real_words = {w.casefold() for v in vals for w in re.findall(r"[A-Za-z]{2,}", v)} - {"de", "la"}
    syn_words = {w.casefold() for o in out for w in re.findall(r"[A-Za-z]{2,}", o)}
    assert p.type is T.PERSON_NAME and not real_words & syn_words, sorted(real_words & syn_words)


def test_mixed_case_names_keep_their_case_style() -> None:
    """'McDonald' / 'DeShawn' became plain title case ('Harrison'): the 'mixed' case-style share was lost."""
    vals = ["McDonald", "DeShawn", "MacArthur", "LeBlanc", "Smith", "O'Brien", "MCKAY"] * 3
    out, _, p = synth(vals, "last_name")
    assert not _cf(vals) & _cf(out)
    styles = Counter(dtext.case_style(v) for v in vals)
    assert Counter(dtext.case_style(o) for o in out) == styles
    assert sum(o.startswith("Mc") for o in out) == 3 and sum(o.startswith("Mac") for o in out) == 3
    assert all(len(o) == len(v) for o, v in zip(sorted(out, key=len), sorted(vals, key=len)))
    st = p.stats["text_shape"]["case"]
    assert st["mixed"] > 0


def test_a_header_with_two_pii_kinds_is_both() -> None:
    """'city_state' matched only the first PII kind ('state'), so real cities were copied as MULTI_SELECT."""
    vals = ["Dallas, TX", "Fort Worth, TX", "Tulsa, OK", "Irving, TX"] * 5
    out, _, p = synth(vals, "city_state")
    assert p.type is T.CITY
    assert not _cf(vals) & _cf(out)
    assert not {"dallas", "fort worth", "tulsa", "irving"} & {o.split(",")[0].casefold() for o in out}
    assert Counter(o.rsplit(", ", 1)[1] for o in out) == Counter({"TX": 15, "OK": 5})  # state part kept
    ctx_keys = dtext._pii_keys("city_state", "state")
    assert ctx_keys == ("state", "city")


# --------------------------------------------------------------------------- exhausted code spaces
def test_dense_zip_column_never_fails_and_never_copies_rare_codes() -> None:
    """A dense ZIP column (more rare codes than free codes of the width) FAILed the whole file."""
    codes = [f"{i:04d}" for i in range(10_000)] + ["1234"] * 20  # every 4-digit code is real
    out, _, p = synth(codes, "zip")
    assert p.type is T.ZIP and len(out) == len(codes)
    assert set(out) == {"1234"}  # nothing to invent: rare codes pooled into the kept code
    r = random.Random(2)
    codes = [f"{r.randrange(10_000):04d}" for _ in range(12_000)]  # ~7k rare codes, ~3k free codes
    rare = {z for z, c in Counter(codes).items() if c < 10}
    out, _, p = synth(codes, "zip")
    assert not rare & set(out) and all(re.fullmatch(r"\d{4}", z) for z in out)


def test_digit_code_with_an_exhausted_shape_does_not_fail() -> None:
    """A zero-padded code whose whole 01-99 space is real FAILed the file ('cannot draw codes')."""
    vals = [f"{i:02d}" for i in range(1, 100)] * 2
    out, _, p = synth(vals, "seat")
    assert p.type is T.DIGIT_CODE and len(out) == len(vals)
    assert not set(vals) & set(out) and all(o.isdigit() for o in out)


# --------------------------------------------------------------------------- end to end on the fixture
def test_adversarial_fixture_is_deterministic() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_text_fixtures_adv", FIXTURES / "make_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\r\n").writerows(mod.make_adversarial(random.Random(4242)))
    assert buf.getvalue().encode("utf-8") == ADV.read_bytes()


def test_adversarial_fixture_end_to_end(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copyfile(ADV, inp / ADV.name)
    out, rep = tmp_path / "out", tmp_path / "rep"
    res = run_cli(inp, "--out-dir", out, "--report-dir", rep, "--seed", "13")
    assert res.exit_code == 0, res.output
    src_bytes, out_bytes = ADV.read_bytes(), (out / ADV.name).read_bytes()
    assert out_bytes.split(b"\r\n", 1)[0] == src_bytes.split(b"\r\n", 1)[0]
    with open(ADV, encoding="utf-8", newline="") as fh:
        real = list(csv.DictReader(fh))
    with open(out / ADV.name, encoding="utf-8", newline="") as fh:
        syn = list(csv.DictReader(fh))
    assert len(syn) == len(real) == 120
    data = json.loads((rep / "files" / f"{ADV.name}.json").read_text(encoding="utf-8"))
    cols = {c["name"]: c for c in data["tables"][0]["columns"]}
    assert {k: (c["type"], c["subtype"]) for k, c in cols.items()} == {
        "Guardian": ("PERSON_NAME", "full"), "city_state": ("CITY", None), "activities": ("MULTI_SELECT", None),
        "phone_ext": ("HIGH_CARD_CATEGORICAL", "code"), "ssn_last4": ("HIGH_CARD_CATEGORICAL", "code"),
        "notes": ("FREE_TEXT", None), "seat": ("DIGIT_CODE", None)}
    for c in ("Guardian", "city_state", "phone_ext", "ssn_last4", "notes", "seat"):
        assert not _cf(r[c] for r in real) & _cf(s[c] for s in syn), c
    assert Counter(s["activities"] for s in syn)["None"] == Counter(r["activities"] for r in real)["None"]
    assert not any("None" in s["activities"] and s["activities"] != "None" for s in syn)
    # numbers stay numbers: the reader infers the same dtypes, and no dtype warning is raised
    src_df, out_df = pd.read_csv(ADV), pd.read_csv(out / ADV.name)
    for c in ("phone_ext", "ssn_last4"):
        assert src_df[c].dtype == out_df[c].dtype == np.int64, c
    assert not [m for m in data["messages"] if m["code"] == "dtype_mismatch"]
    # embedded line breaks survive in the free text, and no real word of the notes does
    assert any("\r\n" in s["notes"] for s in syn) and any("\n" in s["notes"] for s in syn)
    real_words = {w.casefold() for r in real for w in re.findall(r"[^\W\d_]{4,}", r["notes"] + " " + r["Guardian"])}
    syn_words = {w.casefold() for s in syn for w in re.findall(r"[^\W\d_]{4,}", s["notes"])}
    assert not real_words & syn_words, sorted(real_words & syn_words)[:5]
    # nothing real (names, cities) in the out-dir docs or the report
    sensitive = {r[c].strip() for r in real for c in ("Guardian", "city_state") if len(r[c].strip()) >= 4}
    sensitive |= {r["city_state"].split(",")[0] for r in real}
    docs = [p for p in rep.rglob("*") if p.is_file()] + [p for p in out.iterdir() if p.suffix in (".md", ".json")]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert not [v for v in sensitive if v in text], path.name
