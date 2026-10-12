"""Round 9, cluster shapes-counts: one general phone/SSN shape test (R9-02, G3) and exact per-level row counts for
within-ID constants, or a WARN (R9-04, G4).

R9-02, the bug class: the shared shape test (``detect.text``: ``phone_runs``, ``bare_phone``, ``EMB_SSN_RE``) joined
digit groups by exactly ONE separator character, read a bare digit-only number as a phone only when it was a North
American one, and read the SSN layout only with dashes or spaces. A UK number written bare (``07911123456``) or with
doubled or spaced separators (``020  7946  0958``, ``020 - 7946 - 0958``) and a dotted SSN (``123.45.6789``) had no
shape, so D35 kept a value shared by >= ``--min-cell`` students and the leak scanner excused it. The fix makes the
separator rule general instead of listing layouts:

* digit groups are joined by ANY short run of spaces, dots, slashes and dashes (``detect.text.SEP_RUN_MAX``), or set
  in parentheses with spaces inside; a run with more digits than one number holds is split into the phone numbers
  side by side in it (``07911 123456 / 07700 900123``);
* a bare 10-11-digit run opening with ONE national trunk ``0`` is a phone number (``detect.text.trunk_digits``);
  two leading zeros read as zero padding;
* the SSN layout 3-2-4 is an SSN with any separator run (``EMB_SSN_RE``; the leak scan's and the generators' SSN
  token too, ``privacy.digits.SSN_RE``).

The exclusions stay (dates, times, decimals, ranges, spaced ranges too, versions, amounts, IPv4, ZIP+4, counts) and
so do D34's narrow excuses of a name (a bare non-North-American run, numbered things, number lists). The generators
recognise the same spans (``generate.embedded.pii_spans``), so a pseudonym follows the phone rules (a trunk ``0``
kept, ordinary exchange for a North American number) and the SSN rules (area 9xx). A false alarm only costs a pseudonym.

R9-04, the bug class: dealing a within-ID constant's values to ID blocks of unequal sizes (``profile.structure.
balance_entity_rows``) stopped when no improving swap was left, with rows still off, and the file PASSed. Now the
deal is exact whenever an exact deal is found: a complete, randomised, capped search over how many entities of each
block size every label holds, then (when the search gives up) the source's own (level, block size) counts, which
prove an exact deal exists, moved around by row-keeping exchanges. Rows that still miss the source's are reported
(``entity_rows_inexact``, counts only) so the file is WARN, never PASS.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import datanonymous.profile.structure as st
from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.generate import embedded, pseudonyms
from datanonymous.privacy import headers as d34
from datanonymous.privacy.digits import RealDigitKeys

SEED = "9"

#: R9-02: the layouts codex found copied, and their kin (doubled, spaced, slashed separators; brackets with spaces).
PHONES = ["07911123456", "02079460958", "0298765432", "020  7946  0958", "020 - 7946 - 0958", "020 / 7946 / 0958",
          "020--7946--0958", "( 020 ) 7946 0958", "01  23  45  67  89", "312 - 555 - 0199", "312  555  0199",
          "07911 123456 / 07700 900123", "07911123456, 07700900123", "Call 07911123456", "Mom: 020 - 7946 - 0958"]
SSNS = ["123.45.6789", "123/45/6789", "123 - 45 - 6789", "123 . 45 . 6789", "123-45-6789", "123 45 6789",
        "SSN 123.45.6789"]
#: Still no phone number or SSN: dates, times, decimals, ranges (spaced too), versions, amounts, IPv4, ZIP+4, counts,
#: zero padding.
ORDINARY = ["2023-09-15 - 2023-09-20", "0 - 59  60 - 69  70 - 79", "100000 - 200000", "12345 - 6789", "1.5 - 2.5",
            "10:30 - 11:45", "2022-23 - 2023-24", "192.168.100.200", "$1 234 567 890", "0012345678", "1234567.89",
            "01 02 03 04 05", "2023 1500000", "20230915 - 20230920", "12/15/2023 - 01/10/2024", "Grade 9 - 12",
            "10.0.19041.1234", "2021  2022  2023", "100 - 199", "15.06.2023 - 20.06.2023", "20.06.2023  14.30",
            "2023 - 2024", "0612345", "Room 101 - 2nd Floor"]


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = SEED):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, "--force", *extra)
    return res, out, rep


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def file_entry(rep: Path, file: str) -> dict:
    return json.loads((rep / "files" / f"{file}.json").read_text(encoding="utf-8"))


def messages(entry: Any) -> list[dict]:
    """Every message of a per-file report entry (file, table and column messages)."""
    if isinstance(entry, dict):
        own = [entry] if "code" in entry and "level" in entry else []
        return own + [m for v in entry.values() for m in messages(v)]
    if isinstance(entry, list):
        return [m for v in entry for m in messages(v)]
    return []


# --------------------------------------------------------------------------- R9-02: the shape test
@pytest.mark.parametrize("value", PHONES)
def test_a_phone_number_with_any_separators_or_a_trunk_zero_is_a_phone(value: str) -> None:
    assert "phone" in dtext.unit_shapes(value), value
    for org in (False, True):  # never kept for its count, nor as an organisational code
        assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=org), value


@pytest.mark.parametrize("value", SSNS)
def test_the_ssn_layout_with_any_separators_is_an_ssn(value: str) -> None:
    assert "ssn" in dtext.unit_shapes(value), value
    assert "phone" not in dtext.unit_shapes(value), value  # one shape, not two
    assert not dtext.unit_kept(value, holders=10_000, min_cell=10, org_code=True), value


@pytest.mark.parametrize("value", ORDINARY)
def test_dates_ranges_numbers_and_amounts_stay_no_phone_or_ssn(value: str) -> None:
    assert not ({"phone", "ssn"} & dtext.unit_shapes(value)), value


def test_two_numbers_side_by_side_are_found_one_by_one() -> None:
    runs = [m.group() for m in dtext.phone_runs("07911 123456 / 07700 900123")]
    assert runs == ["07911 123456", "07700 900123"]
    assert [m.group() for m in dtext.phone_runs("020 - 7946 - 0958")] == ["020 - 7946 - 0958"]


def test_trunk_digits_are_one_leading_zero() -> None:
    assert dtext.trunk_digits("07911123456") and dtext.trunk_digits("0207946095")
    assert not dtext.trunk_digits("0012345678") and not dtext.trunk_digits("791112345")
    assert not dtext.bare_phone("00791112345")  # zero padding


@pytest.mark.parametrize("name", ["Parent 020 - 7946 - 0958", "Tel 020  7946  0958", "Mom_07911_123456",
                                  "Contact 020 / 7946 / 0958"])
def test_d34_flags_the_new_layouts_after_a_phone_word(name: str) -> None:
    assert "phone" in d34.name_kinds(name), name


@pytest.mark.parametrize("name", ["SSN 123.45.6789", "Student 123 - 45 - 6789"])
def test_d34_flags_an_ssn_with_any_separators(name: str) -> None:
    assert "ssn" in d34.name_kinds(name), name


@pytest.mark.parametrize("name", ["Export_0123456789", "Score 0 - 59  60 - 69", "Invoice 020 - 7946 - 0958",
                                  "Cut 200 - 250 - 300", "Window 2023 - 09 - 01", "Bands 100000 - 200000"])
def test_d34_narrow_excuses_still_hold(name: str) -> None:
    """D34 is frozen (best effort): a bare non-North-American run, a numbered thing and a list of numbers are
    excused in a name as before; only the value test (D35) is widened."""
    assert "phone" not in d34.name_kinds(name), name


# --------------------------------------------------------------------------- R9-02: the generator side
@pytest.mark.parametrize("value", PHONES[3:11] + ["Call 07911123456"])
def test_the_generators_find_the_same_phone_spans(value: str) -> None:
    assert any(kind == "phone" for _, _, kind in embedded.pii_spans(value)), value


@pytest.mark.parametrize("value", SSNS)
def test_the_generators_find_the_same_ssn_spans(value: str) -> None:
    assert any(kind == "ssn" for _, _, kind in embedded.pii_spans(value)), value


@pytest.mark.parametrize("value", SSNS[:4])
def test_an_ssn_pseudonym_follows_the_ssn_rules(value: str) -> None:
    for seed in range(20):
        fake = pseudonyms.pseudonym(value, random.Random(seed))
        assert len(fake) == len(value) and fake != value
        assert [c.isdigit() for c in fake] == [c.isdigit() for c in value]  # the same layout
        assert [c for c in fake if not c.isdigit()] == [c for c in value if not c.isdigit()]
        assert next(c for c in fake if c.isdigit()) == "9"  # area 900-999: never issued


@pytest.mark.parametrize("value", ["07911123456", "0207946095", "020 - 7946 - 0958", "01  23  45  67  89"])
def test_a_phone_pseudonym_keeps_its_trunk_zero_and_layout(value: str) -> None:
    for seed in range(20):
        fake = pseudonyms.pseudonym(value, random.Random(seed), digit_phone=dtext.bare_phone(value))
        assert len(fake) == len(value) and fake != value
        assert [c.isdigit() for c in fake] == [c.isdigit() for c in value]
        assert fake[0] == "0" or not fake[0].isdigit()


def test_a_north_american_pseudonym_keeps_shape_with_ordinary_exchange() -> None:
    for seed in range(20):
        fake = pseudonyms.pseudonym("312 - 555 - 0199", random.Random(seed))
        assert fake[3:6] == " - " and fake[9:12] == " - " and fake[6] in "23456789", fake


def test_phone_digits_keep_a_trunk_zero() -> None:
    for seed in range(20):
        d = embedded.phone_digits("0207946095", random.Random(seed))
        assert len(d) == 10 and d[0] == "0"
        nanp = embedded.phone_digits("3125550199", random.Random(seed))
        assert nanp[3] in "23456789" and nanp[3:6] not in embedded._N11


def test_the_guard_corrects_a_dotted_ssn_and_leaves_a_trunk_number() -> None:
    assert embedded.guard("ID 482.19.3301", random.Random(1)) == "ID 982.19.3301"
    assert embedded.guard("482 / 19 / 3301", random.Random(1)) == "982 / 19 / 3301"
    trunk = "020 7946 095"  # ten digits opening with a trunk 0: no North American line, left alone
    assert embedded.guard(trunk, random.Random(1)) == trunk


def test_the_leak_keys_read_an_ssn_with_any_separators() -> None:
    keys = RealDigitKeys(frozenset(), frozenset({"123456789"}))
    for text in ("123-45-6789", "123.45.6789", "x 123 / 45 / 6789 y", "123 - 45 - 6789"):
        assert keys.holds_real(text), text
    assert not keys.holds_real("1234-5-6789") and not keys.holds_real("123456789")


# --------------------------------------------------------------------------- R9-02: end to end (codex probe4)
@pytest.mark.parametrize("value", ["07911123456", "02079460958", "020  7946  0958", "020 - 7946 - 0958",
                                   "123.45.6789", "123 / 45 / 6789"])
def test_a_shared_phone_or_ssn_value_is_never_copied(tmp_path: Path, value: str) -> None:
    # codex R9-02: 20 students share the value under a neutral header, so the D35 count alone would keep it
    rows = [[f"S{1000 + i}", value, i % 5 + 50] for i in range(20)]
    src = write_csv(tmp_path / "in" / "input.csv", ["student_id", "Value", "Score"], rows)
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = [r["Value"] for r in read_dicts(out / "input.csv")]
    assert len(syn) == 20 and len(set(syn)) == 1 and syn[0] != value  # one consistent pseudonym, exact count
    assert [c.isdigit() for c in syn[0]] == [c.isdigit() for c in value]
    assert value not in everything(out, rep, console=res.output)
    if "." in value or "/" in value:
        assert next(c for c in syn[0] if c.isdigit()) == "9"  # the SSN rules
    else:
        assert syn[0][0] == "0"  # the trunk 0 kept


# --------------------------------------------------------------------------- R9-04: exact rows per level
CODEX_SIZES = np.asarray([9, 4, 16, 12, 14, 2, 6, 15, 16, 9, 16, 16])
CODEX_LABELS = np.asarray([0, 1, 1, 1, 0, 2, 2, 2, 1, 0, 0, 2])


def _rows(labels: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels, weights=sizes, minlength=k).astype(np.int64)


@pytest.mark.parametrize("witness", [False, True])
def test_the_codex_deal_is_exact_for_every_start(witness: bool) -> None:
    target = _rows(CODEX_LABELS, CODEX_SIZES, 3)
    for seed in range(300):
        deal = np.random.default_rng(seed).permutation(CODEX_LABELS)
        report: list[dict[str, int]] = []
        out = st.balance_entity_rows(deal, CODEX_SIZES, target, np.random.default_rng(seed), 10, [False] * 3,
                                     witness=(CODEX_LABELS, CODEX_SIZES) if witness else None, report=report)
        assert np.array_equal(np.bincount(out, minlength=3), np.bincount(CODEX_LABELS, minlength=3)), seed
        assert np.array_equal(_rows(out, CODEX_SIZES, 3), target), seed
        assert report == [], seed


def test_the_exact_deal_is_not_always_the_real_one() -> None:
    """Two partitions of the codex blocks are exact; the search, randomised, finds both (it does not rebuild the real
    assignment where another exact deal exists)."""
    target = _rows(CODEX_LABELS, CODEX_SIZES, 3)
    seen = set()
    for seed in range(200):
        deal = np.random.default_rng(seed).permutation(CODEX_LABELS)
        out = st.balance_entity_rows(deal, CODEX_SIZES, target, np.random.default_rng(seed), 10, [False] * 3)
        seen.add(tuple(sorted(CODEX_SIZES[out == 2].tolist())))
    assert seen == {(2, 6, 15, 16), (6, 9, 9, 15)}


def _random_table(rng: np.random.Generator, n: int, n_many: int, n_few: int, size_hi: int):
    sizes = rng.integers(1, size_hi + 1, n)
    few = rng.integers(1, 10, n_few)
    labels = np.concatenate([np.full(c, n_many + i) for i, c in enumerate(few)]
                            + [rng.integers(0, n_many, max(n - int(few.sum()), 0))])[:n]
    rng.shuffle(labels)
    k = int(labels.max()) + 1
    real = (np.bincount(labels, minlength=k) >= 10).tolist()  # rarer real values are D35 pseudonyms
    return sizes, labels, k, real


@pytest.mark.parametrize("shape", [(12, 3, 0, 19), (20, 2, 3, 30), (30, 3, 5, 40), (50, 4, 10, 60), (100, 5, 20, 100),
                                   (40, 6, 4, 200), (300, 5, 40, 200)])
def test_small_tables_with_unequal_blocks_get_exact_rows(shape: tuple[int, int, int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    for trial in range(25):
        sizes, labels, k, real = _random_table(rng, *shape)
        target = _rows(labels, sizes, k)
        report: list[dict[str, int]] = []
        out = st.balance_entity_rows(rng.permutation(labels), sizes, target, np.random.default_rng(trial), 10, real,
                                     witness=(labels, sizes), report=report)
        assert np.array_equal(np.bincount(out, minlength=k), np.bincount(labels, minlength=k)), trial
        assert np.array_equal(_rows(out, sizes, k), target), trial
        assert report == []


def test_an_impossible_deal_is_reported_with_counts() -> None:
    sizes = np.asarray([1, 1, 5, 5])
    labels = np.asarray([0, 0, 1, 1])
    report: list[dict[str, int]] = []
    out = st.balance_entity_rows(labels, sizes, [4, 8], np.random.default_rng(1), 1, report=report)
    assert np.array_equal(np.bincount(out, minlength=2), [2, 2])  # entity counts kept
    assert report and report[0]["labels"] == 2 and report[0]["rows"] >= 1 and report[0]["codes"] == [0, 1]


def test_the_search_is_deterministic_and_capped() -> None:
    sizes, labels, k, real = _random_table(np.random.default_rng(5), 100, 5, 20, 100)
    target = _rows(labels, sizes, k)
    deal = np.random.default_rng(6).permutation(labels)
    a = st.balance_entity_rows(deal, sizes, target, np.random.default_rng(7), 10, real)
    b = st.balance_entity_rows(deal, sizes, target, np.random.default_rng(7), 10, real)
    assert np.array_equal(a, b)
    # an infeasible problem ends within the step cap (parity: odd rows from even blocks)
    need = [(0, 25, 51), (1, 25, 49)]
    assert st._search_counts([2, 4], [30, 20], need, np.random.default_rng(1)) is None


def test_a_witness_that_does_not_fit_is_not_used() -> None:
    sz = np.asarray([2, 3, 5])
    avail = np.asarray([1, 1, 1])
    need = [(0, 1, 5), (1, 2, 5)]
    assert st._witness_counts((np.asarray([0, 1, 1]), np.asarray([5, 2, 3])), sz, avail, need) is not None
    assert st._witness_counts((np.asarray([0, 1, 1]), np.asarray([5, 2, 2])), sz, avail, need) is None  # other sizes
    assert st._witness_counts((np.asarray([0, 1]), np.asarray([5, 2])), sz, avail, need) is None


def test_mixing_keeps_every_label_exact() -> None:
    rng = np.random.default_rng(3)
    sz = np.arange(1, 21)
    table = rng.integers(0, 4, (5, 20))
    mixed = st._mix_counts(table, sz, np.random.default_rng(4), 2000)
    assert np.array_equal(mixed.sum(axis=0), table.sum(axis=0))
    assert np.array_equal(mixed.sum(axis=1), table.sum(axis=1))
    assert np.array_equal(mixed @ sz, table @ sz)
    assert (mixed >= 0).all() and not np.array_equal(mixed, table)


# --------------------------------------------------------------------------- R9-04: end to end (codex probe)
def _codex_rows() -> list[list[str]]:
    names = ["Quenby", "Zelwick", "Brixham"]
    return [[f"S{1000 + i}", names[CODEX_LABELS[i]], str(j)] for i in range(12) for j in range(int(CODEX_SIZES[i]))]


def _level_rows(path: Path) -> list[int]:
    return sorted(Counter(r["Activity"] for r in read_dicts(path)).values())


@pytest.mark.parametrize("seed", ["1", "2", "3", "4"])
def test_pseudonyms_keep_their_exact_rows_on_unequal_blocks(tmp_path: Path, seed: str) -> None:
    src = write_csv(tmp_path / "in" / "input.csv", ["student_id", "Activity", "Visit"], _codex_rows())
    res, out, rep = run(src, tmp_path, seed=seed)
    assert res.exit_code == 0, res.output
    assert _level_rows(out / "input.csv") == [39, 48, 48]
    entry = file_entry(rep, "input.csv")
    assert entry["status"] == "PASS"
    assert not any(m["code"] == "entity_rows_inexact" for m in messages(entry))


def test_rows_left_off_warn_with_counts(tmp_path: Path, monkeypatch) -> None:
    """With the exact deal undone, a level whose rows miss the source's makes the file WARN (never PASS), and the
    warning gives counts only; an exact deal gives no warning."""
    monkeypatch.setattr(st, "_exact_deal", lambda *a, **k: False)
    src = write_csv(tmp_path / "in" / "input.csv", ["student_id", "Activity", "Visit"], _codex_rows())
    off = 0
    for seed in ("1", "2", "3", "4", "5", "6"):
        res, out, rep = run(src, tmp_path, seed=seed, name=f"s{seed}")
        assert res.exit_code == 0, res.output
        entry = file_entry(rep, "input.csv")
        warned = [m for m in messages(entry) if m["code"] == "entity_rows_inexact"]
        exact = _level_rows(out / "input.csv") == [39, 48, 48]
        assert bool(warned) == (not exact), seed
        if warned:
            off += 1
            assert entry["status"] == "WARN" and warned[0]["column"] == "Activity"
            report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
            scan = next(f for f in report["verdict"]["files"] if f["source"] == "input.csv")
            assert scan["checks"]["leak_scan"]["result"] == "PASS"  # a counts-only message
            for name in ("Quenby", "Zelwick", "Brixham"):
                assert name not in warned[0]["text"]
    assert off  # the swaps alone stall for some seed (codex: seeds 2 and 4)


def test_exact_deals_are_deterministic(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "input.csv", ["student_id", "Activity", "Visit"], _codex_rows())
    outs = [(run(src, tmp_path, seed="2", name=f"d{k}")[1] / "input.csv").read_bytes() for k in range(2)]
    assert outs[0] == outs[1]
