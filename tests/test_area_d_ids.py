"""AREA D: ID candidates, synthetic ID shapes (padding per raw form, hex, Luhn), aliases and nesting."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as hst

from datanonymous.config import Settings
from datanonymous.detect.context import DetectContext
from datanonymous.detect.ids import (
    char_mask,
    hex_hash_length,
    id_candidate_reason,
    is_uuid,
    luhn_check_digit,
    luhn_valid,
    normalize_id,
)
from datanonymous.detect.types import detect_column
from datanonymous.errors import IdSpaceError
from datanonymous.generate.ids import IdRegistry, allocate_digit_ids, build_mapping
from datanonymous.io import csvio
from datanonymous.profile.table import profile_table


def _arr(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def _ctx(name: str, values: list[str], settings: Settings | None = None) -> DetectContext:
    s = settings or Settings(mode="fidelity")
    return DetectContext(index=0, name=name, values=_arr(values), file_rel="t.csv", table=None, format="csv",
                         options=s.column_options("t.csv", None, name), settings=s)


def _reason(name: str, values: list[str]) -> str | None:
    ctx = _ctx(name, values)
    return id_candidate_reason(ctx, detect_column(ctx))


def _profile(text: str, settings: Settings | None = None, name: str = "t.csv"):
    r = csvio.parse_text_bytes(text.encode("utf-8"))
    return profile_table(r.tables[0], name, "csv", settings or Settings(mode="fidelity"))


# --------------------------------------------------------------------------- candidate rules (D7, section 3 #13)
def test_uuid_and_hash_candidates() -> None:
    rng = np.random.default_rng(1)
    raw = [rng.bytes(16) for _ in range(30)]
    assert _reason("record_key2", [str(uuid.UUID(bytes=b, version=4)) for b in raw]) == "UUID values"
    for fn, width in ((hashlib.md5, 32), (hashlib.sha1, 40), (hashlib.sha256, 64)):
        why = _reason("digest", [fn(b).hexdigest() for b in raw])
        assert why is not None and f"{width}-character hex hash" in why
    assert is_uuid("7c42a60b-6245-4094-baa0-4d30f538e710") and not is_uuid("7c42a60b62454094baa04d30f538e710")
    assert hex_hash_length("ab" * 16) == 32 and hex_hash_length("zz" * 16) is None


def test_fixed_width_and_mask_candidates() -> None:
    tickets = [f"{a}{b}-{i:04d}" for i, (a, b) in enumerate(zip("ABCDEFGHJKLMNPQRSTUVWXYZ", "ZYXWVUTSRQPNMLKJHGFEDCBA"))]
    assert _reason("ticket", tickets) == "unique fixed-width codes (7 characters)"
    # >= 95% unique, one shape covering >= 90% of rows (two values repeat, lengths differ by one)
    codes = [f"AB{i:05d}" for i in range(95)] + ["AB00001", "AB00002", "X1", "Y22", "Z333"]
    why = _reason("voucher", codes)
    assert why is not None and "unique values; one character shape covers" in why


def test_sequential_and_hinted_candidates_kept_from_foundation() -> None:
    assert _reason("row_num", [str(i) for i in range(1, 60)]) == "unique sequential integers"
    assert _reason("badge_id", [str(1000 + 7 * i) for i in range(40)]).startswith("unique values and ID-like header")


def test_no_candidate_for_measurements_org_codes_and_personal_data() -> None:
    rng = np.random.default_rng(3)
    # like the real Expense_Semester: unique, but 4-5 digits (one width on < 90% of rows) - a measurement
    expenses = [str(v) for v in rng.choice(np.arange(10_000, 49_999), 55, replace=False)] + [
        str(v) for v in rng.choice(np.arange(5_000, 9_999), 7, replace=False)]
    assert _reason("Expense_Semester", expenses) is None
    assert _reason("school_code", [f"{i:03d}" for i in range(500, 560)]) is None  # D10 org code keeps values
    phones = [f"555-01{i:02d}" for i in range(60)]
    assert _reason("phone", phones) is None  # personal data: own generator
    assert _reason("gender", ["F", "M"] * 20) is None


def test_luhn_helpers() -> None:
    assert luhn_valid("79927398713") and not luhn_valid("79927398714")
    assert luhn_check_digit("7992739871") == 3
    assert luhn_valid("0079927398713")  # leading zeros do not change the check digit
    assert not luhn_valid("7") and not luhn_valid("12a4")


# --------------------------------------------------------------------------- per-raw-form padding (D8/D9)
def test_padded_and_unpadded_forms_keep_their_own_width() -> None:
    reg = IdRegistry()
    a = _arr(["0012345", "0067890", "0099999", "0012345"])  # students: zero padded to 7
    b = _arr(["12345", "67890", "99999", "55555"])  # enrollments: plain 5 digits
    reg.register("studentid", "students.csv", None, 0, "student_id", a)
    reg.register("studentid", "enrollments.csv", None, 0, "student_id", b)
    reg.build(5)
    ra, rb = reg.render("studentid", a).tolist(), reg.render("studentid", b).tolist()
    assert all(len(v) == 7 and v.startswith("00") for v in ra)
    assert all(len(v) == 5 and v[0] != "0" for v in rb)
    assert normalize_id(ra[0]) == normalize_id(rb[0]) and normalize_id(ra[1]) == normalize_id(rb[1])
    assert ra[0] == ra[3]
    real = {normalize_id(v) for v in a.tolist() + b.tolist()}
    assert not {normalize_id(v) for v in ra + rb} & real


def test_mixed_padding_within_one_column() -> None:
    reg = IdRegistry()
    col = _arr(["00123", "4567", "123", "00042", "98765"])
    reg.register("k", "f.csv", None, 0, "student_id", col)
    reg.build(2)
    out = reg.render("k", col).tolist()
    assert [len(v) for v in out] == [5, 4, 3, 5, 5]
    assert out[0].startswith("00") and out[3].startswith("000")
    assert normalize_id(out[0]) == normalize_id(out[2])  # '00123' and '123' are the same ID (D8)


def test_padded_only_keys_move_widths_invisibly() -> None:
    real = [f"{i:07d}" for i in range(1, 2001) if i != 50]  # dense, but always written 7 wide
    reg = IdRegistry()
    col = _arr(real)
    reg.register("k", "f.csv", None, 0, "student_id", col)
    reg.build(3)
    ns = reg.namespaces["k"]
    assert not ns.notes  # no visible width change: the padding hides the number of significant digits
    out = reg.render("k", col).tolist()
    assert all(len(v) == 7 for v in out)
    assert not set(out) & set(real)
    assert len(set(out)) == len(real)
    assert [int(v) for v in out] != sorted(int(v) for v in out)  # no rank linkage (D8)


def test_allocate_keeps_significant_digits() -> None:
    keys = ["7", "42", "123", "9999", "55555"]
    nat = {k: len(k) for k in keys}
    out = allocate_digit_ids(keys, nat, {}, np.asarray(sorted(int(k) for k in keys)), np.random.default_rng(0), [])
    assert [len(str(v)) for v in out.tolist()] == [1, 2, 3, 4, 5]
    # a full width: unpadded integers take another digit count (D9, noted; K-02 harden 2 A2-6), padded ones FAIL
    nine = [str(i) for i in range(1, 10)]
    notes: list[str] = []
    out = allocate_digit_ids(nine, {k: 1 for k in nine}, {}, np.arange(0, 10), np.random.default_rng(0), notes)
    assert all(len(str(v)) == 2 for v in out.tolist()) and notes
    with pytest.raises(IdSpaceError):
        allocate_digit_ids(nine, {k: 1 for k in nine}, {}, np.arange(0, 10), np.random.default_rng(0), [],
                           pad_limit={k: 1 for k in nine})


# --------------------------------------------------------------------------- hex IDs, Luhn
def test_uuid_and_hash_ids_keep_the_hex_class() -> None:
    rng = np.random.default_rng(9)
    uuids = [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(50)]
    hashes = [hashlib.md5(rng.bytes(8)).hexdigest().upper() for _ in range(50)]
    for vals in (uuids, hashes):
        reg = IdRegistry()
        col = _arr(vals)
        reg.register("k", "f.csv", None, 0, "record_id", col)
        reg.build(1)
        out = reg.render("k", col).tolist()
        assert not set(out) & set(vals)
        for r, s in zip(vals, out):
            assert len(r) == len(s) and all(c in "0123456789abcdefABCDEF-" for c in s)
            assert [c == "-" for c in r] == [c == "-" for c in s]
            assert (s == s.upper()) == (r == r.upper())
    reg = IdRegistry()
    reg.register("k", "f.csv", None, 0, "record_id", _arr(uuids))
    reg.build(4)
    assert all(is_uuid(v) for v in reg.render("k", _arr(uuids)).tolist())


def _luhn_ids(n: int, width: int, rng: np.random.Generator) -> list[str]:
    out = set()
    while len(out) < n:
        body = str(int(rng.integers(10 ** (width - 2), 10 ** (width - 1))))
        out.add(body + str(luhn_check_digit(body)))
    return sorted(out)


def test_luhn_check_digits_on_request() -> None:
    real = _luhn_ids(200, 9, np.random.default_rng(5))
    col = _arr(real)
    reg = IdRegistry(id_check="luhn")
    reg.register("k", "f.csv", None, 0, "card_id", col)
    reg.build(6)
    assert reg.namespaces["k"].check == "luhn"
    out = reg.render("k", col).tolist()
    assert all(luhn_valid(v) and len(v) == 9 for v in out)
    assert not set(out) & set(real) and len(set(out)) == len(real)
    # without the flag the check digit is not kept
    reg2 = IdRegistry()
    reg2.register("k", "f.csv", None, 0, "card_id", col)
    reg2.build(6)
    assert reg2.namespaces["k"].check is None
    assert sum(luhn_valid(v) for v in reg2.render("k", col).tolist()) < 60


def test_luhn_not_applied_when_real_ids_fail_it() -> None:
    real = [str(10_000 + 13 * i) for i in range(100)]
    ns_reg = IdRegistry(id_check="luhn")
    ns_reg.register("k", "f.csv", None, 0, "card_id", _arr(real))
    ns_reg.build(1)
    assert ns_reg.namespaces["k"].check is None


def test_build_mapping_default_signature_unchanged() -> None:
    reg = IdRegistry()
    reg.register("k", "f.csv", None, 0, "student_id", _arr([f"S{i:05d}" for i in range(1, 40)]))
    ns = reg.namespaces["k"]
    build_mapping(ns, np.random.default_rng(0))
    assert all(char_mask(v) == "A99999" for v in ns.mapping.values())


# --------------------------------------------------------------------------- aliases and nesting
def _long_text() -> str:
    rows = ["student_id,state_id,family_ident,other_id,score"]
    rng = np.random.default_rng(4)
    for s in range(60):
        fam = 900 + s // 2
        for k in range(1 + s % 3):
            other = int(rng.integers(1, 30))
            rows.append(f"{1000 + s},{55_000 + 7 * s},{fam},{other},{int(rng.integers(0, 100))}")
    return "\n".join(rows) + "\n"


def test_aliases_nesting_and_unrelated_ids() -> None:
    from conftest import build_fidelity_settings as build_settings

    # other_id is not on the default list: declare it
    settings = build_settings({"id_col": ["other_id", "student_id", "state_id", "family_ident"]})
    tp = _profile(_long_text(), settings)
    names = [c.name for c in tp.columns]
    assert tp.entity.aliases == [[names.index("student_id"), names.index("state_id")]]
    nest = tp.entity.nesting
    assert len(nest) == 1 and nest[0]["child"] == "student_id" and nest[0]["parent"] == "family_ident"
    assert nest[0]["children_per_parent"] == {"min": 2, "max": 2}
    other = tp.columns[names.index("other_id")]
    assert any(m.code == "id_columns_unrelated" for c in tp.columns for m in c.messages)
    assert tp.columns[names.index("state_id")].params["alias_of"] == ["student_id"]
    assert other.role == "id"


def test_identical_aliases_share_one_mapping(tmp_path) -> None:
    from conftest import read_dicts, run_cli

    d = tmp_path / "in"
    d.mkdir()
    rows = ["student_id,local_id,score"] + [f"{4000 + 3 * i},{4000 + 3 * i},{i % 7}" for i in range(80)]
    (d / "a.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")
    (d / "b.csv").write_text("local_id,flag\n" + "".join(f"{4000 + 3 * i},{i % 2}\n" for i in range(0, 80, 2)),
                             encoding="utf-8", newline="")
    res = run_cli(d, "--out-dir", tmp_path / "o", "--seed", "3")
    assert res.exit_code == 0, res.output
    a = read_dicts(tmp_path / "o" / "a.csv")
    b = read_dicts(tmp_path / "o" / "b.csv")
    assert all(r["student_id"] == r["local_id"] for r in a)  # identical aliases stay identical
    assert {r["local_id"] for r in b} <= {r["local_id"] for r in a}  # cross-file link through the alias
    assert not {r["student_id"] for r in a} & {str(4000 + 3 * i) for i in range(80)}


@hsettings(max_examples=60, deadline=None)
@given(ints=hst.lists(hst.integers(min_value=0, max_value=99_999), min_size=1, max_size=60, unique=True),
       pad_a=hst.integers(min_value=5, max_value=8), seed=hst.integers(min_value=0, max_value=2**31))
def test_digit_id_invariants(ints: list[int], pad_a: int, seed: int) -> None:
    """Never real, one synthetic per key, each raw form keeps its width (D8/D9)."""
    a = _arr([f"{v:0{pad_a}d}" for v in ints])  # zero padded in one file
    b = _arr([str(v) for v in ints[::2]])  # plain in another
    reg = IdRegistry()
    reg.register("k", "a.csv", None, 0, "student_id", a)
    reg.register("k", "b.csv", None, 0, "student_id", b)
    reg.build(seed)
    ns = reg.namespaces["k"]
    if ns.error is not None or ns.identity:
        return  # tiny dense spaces may be too small (D9: fail clearly); 1..n row numbers stay 1..n (D9)
    ra, rb = reg.render("k", a).tolist(), reg.render("k", b).tolist()
    real_keys = {normalize_id(v) for v in a.tolist()}
    assert not {normalize_id(v) for v in ra} & real_keys
    assert len({normalize_id(v) for v in ra}) == len(ints)
    assert dict(zip(a.tolist()[::2], ra[::2])) and all(
        normalize_id(x) == normalize_id(y) for x, y in zip(ra[::2], rb))
    if not ns.notes:
        assert all(len(x) == len(y) for x, y in zip(a.tolist(), ra))
        assert all(len(x) == len(y) for x, y in zip(b.tolist(), rb))


def test_rows_per_id_profile_counts_every_row() -> None:
    tp = _profile(_long_text())
    assert sum(k * v for k, v in tp.entity.rows_per_id.items()) == tp.n_rows
    assert Counter(tp.entity.rows_per_id).total() == 60
