"""D46: ordinary phone space, run-wide base collisions, and final on-disk verification."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import run_cli
from datanonymous import pipeline
from datanonymous.errors import GenerationError
from datanonymous.generate import embedded, pii, text
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType, TableProfile
from datanonymous.privacy import digits, leaks, texts


def keys(*values):
    dt = leaks.digit_terms(leaks.PHONE, list(values))
    return digits.RealDigitKeys(dt.phones, dt.ssns, digits.NANP_LAYOUTS | dt.layouts, dt.phone_bases)


@pytest.mark.parametrize("source,candidate", [
    ("312-244-5678 x12", "+1 (312) 244-5678 extension 999"),
    ("+1 (312) 244-5678", "3122445678.0"),
    ("３１２－２４４－５６７８", "312.244.5678"),
    ("٣١٢-٢٤٤-٥٦٧٨", "3122445678"),
    ("＋１（３１２）２３４－５６７８", "312-234-5678"),
    ("3122345678", "＋１（３１２）２３４－５６７８"),
    ("312:234:5678 ext. 12", "+1 (312) 234-5678 x99"),
    ("3122345678", "312:234:5678"),
    ("244-5678 x1", "(773) 244-5678 x2"),
    ("773-244-5678", "2445678.00"),
    ("+44 20 7946 0958 ext. 12", "+44-20-7946-0958 #99"),
    ("020 / 7946 / 0958", "02079460958"),
])
def test_semantic_phone_bases_ignore_layout_extensions_and_unicode(source, candidate):
    registry = keys(source)
    with digits.guard(registry):
        assert digits.holds_phone(candidate)
        assert not text.Drawer(set(), phone=True).ok(candidate)
    idx = leaks.SensitiveIndex()
    idx.phone_bases.add(registry.phone_bases, 0)
    p = ColumnProfile(index=0, name="phone", type=ColumnType.PHONE)
    hits = leaks.DataScanner(idx, leaks.Allow()).scan_column(
        SimpleNamespace(rel_path="out.csv"), 0, p, np.array([candidate], dtype=object))
    assert any(h.rule == "phone digits" for h in hits)


@pytest.mark.parametrize("source,candidate", [
    ("３１２－２４４－５６７８ ext. 12", "3122445678.0"),
    ("244-5678 x12", "2445678"),
    ("3122445678", "３１２－２４４－５６７８"),
    ("＋１（３１２）２３４－５６７８", "312-234-5678"),
    ("3122345678", "＋１（３１２）２３４－５６７８"),
])
def test_reports_and_same_column_checks_share_phone_base_canonicalization(source, candidate):
    terms = leaks._Terms()
    terms.add(leaks.PHONE, [source])
    idx = leaks.SensitiveIndex()
    idx._merge_terms(terms, 0)
    scanner = leaks.Scanner(idx, leaks.Allow())
    assert scanner.hits_in_text(candidate)
    assert candidate not in scanner.redact_text(candidate)
    real = leaks._RealSets.of(leaks.PHONE, [source])
    assert leaks._same_column_hits(leaks.PHONE, real, [candidate]) == 1


def test_local_aliases_do_not_merge_two_national_area_codes_or_reclassify_counts():
    registry = keys("312-244-5678")
    with digits.guard(registry):
        assert not digits.holds_phone("773-244-5678")
        assert digits.holds_phone("244-5678")
        assert not digits.holds_real("2445678")
        for value in ("244.5678", "2024-03-15 10:30", "4.3 3.9 4.0 3.8 3.5", "12312445678999"):
            assert not digits.holds_real(value)


@pytest.mark.parametrize("candidate", ["900-12-3456", "900\u201012\u20103456", "900\u201112\u20113456"])
def test_phone_drawer_preserves_the_existing_ssn_guard(candidate):
    registry = digits.RealDigitKeys(phones=set(), ssns={"900123456"}, phone_bases={"3122345678"})
    with digits.guard(registry):
        assert text.real_digits(candidate)
        assert embedded.guard(candidate, random.Random(51)) == candidate
        assert not text.Drawer(set(), phone=True).ok(candidate)


@pytest.mark.parametrize("source", ["900-12-3456", "９００－１２－３４５６", "٩٠٠-١٢-٣٤٥٦"])
@pytest.mark.parametrize("candidate", ["900123456", "900123456.0", "900 123 456", "900-12-3456", "９００１２３４５６"])
def test_known_phone_base_never_equals_a_declared_source_ssn(source, candidate):
    terms = leaks.digit_terms(leaks.SSN, [source])
    registry = digits.RealDigitKeys(terms.phones, terms.ssns)
    with digits.guard(registry):
        assert digits.holds_phone(candidate)
        assert not text.Drawer(set(), phone=True).ok(candidate)
        # The known-PHONE extension does not reinterpret generic counts as SSNs.
        assert not digits.holds_real("900123456")
        assert not digits.holds_real("900123456.0")
    idx = leaks.SensitiveIndex()
    idx.ssn_digits.add(terms.ssns, 0)
    p = ColumnProfile(index=0, name="Phone", type=ColumnType.PHONE)
    hits = leaks.DataScanner(idx, leaks.Allow()).scan_column(
        SimpleNamespace(rel_path="out.csv"), 0, p, np.array([candidate], dtype=object))
    assert any(h.rule == "SSN" for h in hits)
    assert not leaks.DataScanner(idx, leaks.Allow()).distinctive(["900123456", "900123456.0"])


def test_phone_decks_exclude_source_ssns_before_drawing(monkeypatch):
    class FirstSlot(random.Random):
        draws = 0

        def randrange(self, stop):
            self.draws += 1
            return 0

    registry = digits.RealDigitKeys(phones=set(), ssns={"100000000"})
    group = text.Group("123456789", "123456789", [("123456789", 1)], 1)
    profile = ColumnProfile(index=0, name="Phone", type=ColumnType.PHONE)
    rng = FirstSlot(51)
    with digits.guard(registry):
        make = pii._make_phone(profile, ColumnModel(), [group], None, rng)
        assert make(group, 0) == "100000001"
        assert rng.draws == 1
        # Exhaust the stateless random fast path, then use its exact free deck.
        monkeypatch.setattr(embedded, "phone_digits", lambda *args, **kwargs: "100000000")
        rng = FirstSlot(51)
        assert embedded.stateless_span("phone", "123456789", rng) == "100000001"
        assert rng.draws == 1


@pytest.mark.parametrize("source,view,found", [
    ("312-234-5678", "Call 312-234-5678\n", True),
    ("312-234-5678", "Note,callback\nPhone,＋１（３１２）２３４－５６７８ ext. 12\n", True),
    ("+44 20 7946 0958", "<note>Call +44-20-7946-0958</note>", True),
    ("234-5678", "metadata: +1 (312) 234-5678", True),
    ("234-5678", "count: 2345678", False),
    ("312-234-5678", "count: 31223.45678", False),
    ("312-234-5678", "id: 99312234567899", False),
    ("312-234-5678", "2024-03-15 10:30", False),
])
def test_generic_phone_scan_checks_stale_file_and_metadata_text_views(source, view, found):
    terms = leaks._Terms()
    terms.add(leaks.PHONE, [source])
    idx = leaks.SensitiveIndex()
    idx._merge_terms(terms, 0)
    scanner = leaks.DataScanner(idx, leaks.Allow())
    in_cells = scanner.distinctive([view])
    in_view = scanner.distinctive([], joined=view)
    assert any(rule == "phone digits" for _, rule, _ in in_view) is found
    assert in_view == in_cells


def test_report_phone_normalization_does_not_expand_the_general_digit_registry():
    idx = leaks.SensitiveIndex()
    idx.digits.add({"3122345678", "31223456780"}, 0)
    scanner = leaks.Scanner(idx, leaks.Allow())
    assert not scanner.hits_in_text("3122345678.0")
    assert not scanner.hits_in_text("＋１（３１２）２３４－５６７８")


@pytest.mark.parametrize("value", ["312:234:5678", "312|234|5678 ext. 12", "123456", "1234567890123456"])
def test_unsupported_declared_numeric_phone_layouts_fail_clearly(value):
    model = ColumnModel(nonmissing_counts={value: 1}, real_values=frozenset({value}))
    profile = ColumnProfile(index=0, name="Code", type=ColumnType.PHONE)
    profile.model = model
    with pytest.raises(GenerationError, match="unsupported numeric layout") as error:
        pii.generate_pii(profile, model, 1, np.random.default_rng(51), lambda: None)
    assert value not in str(error.value)


def test_phone_deck_counts_every_extension_and_fails_only_when_exhausted(monkeypatch):
    monkeypatch.setattr(pii, "_PHONE_EXCHANGES", ("222",))
    monkeypatch.setattr(pii, "_EXCHANGE_INDEX", {"222": 0})
    blocked = {f"312222{i:04d}" for i in range(9997)}
    pool = pii._PhoneSpace("3122445678", "312", 1, blocked)
    rng = random.Random(51)
    got = {pool.draw(rng) for _ in range(30)}
    assert got == {(f"312222{i:04d}", str(ext)) for i in range(9997, 10000) for ext in range(10)}
    with pytest.raises(GenerationError, match="no unused collision-free"):
        pool.draw(rng)
    # Blocking the base excludes all its extensions, regardless of the source extension.
    blocked |= {f"312222{i:04d}" for i in range(9997, 10000)}
    exhausted = pii._PhoneSpace("3122445678", "312", 6, blocked)
    assert exhausted.remaining == 0
    with pytest.raises(GenerationError):
        exhausted.draw(rng)


def test_more_than_ten_thousand_numbers_keep_the_area_pmf_shape_and_seed():
    values = [f"(312) {240 + i // 10000}-{i % 10000:04d}" for i in range(12_050)]
    values += [f"415.266.{i:04d} ext. 12" for i in range(31)]
    model = ColumnModel(nonmissing_counts=dict.fromkeys(values, 1), real_values=frozenset(values))
    profile = ColumnProfile(index=0, name="Phone", type=ColumnType.PHONE, params={"_min_cell": 10})
    profile.model = model
    registry = keys(*values)

    def draw():
        with digits.guard(registry):
            return pii.generate_pii(profile, model, len(values), np.random.default_rng(51), lambda: None)[0].tolist()

    first = draw()
    assert first == draw()
    assert len(set(first)) == len(values)
    assert Counter(digits.phone_base(v)[0][:3] for v in first) == {"312": 12_050, "415": 31}
    assert Counter(re.sub(r"\d", "9", v) for v in first) == Counter(re.sub(r"\d", "9", v) for v in values)
    assert len({digits.phone_base(v)[0][3:6] for v in first}) > 500
    assert not any(registry.holds_phone(v) for v in first)


def test_stateless_numbered_and_mapped_phone_routes_reject_source_bases(monkeypatch):
    registry = keys("773-244-5678")
    original = embedded.phone_digits
    draws = iter(["7732445678", "7732449999"])
    monkeypatch.setattr(embedded, "phone_digits", lambda *a, **kw: next(draws))
    with digits.guard(registry):
        assert embedded.stateless_span("phone", "312-299-0000", random.Random(1)) == "773-244-9999"
    monkeypatch.setattr(embedded, "phone_digits", original)
    with digits.guard(registry):
        assert not registry.holds_phone(embedded._numbered("phone", "773-299-0000", set(), random.Random(1)))
        synth = embedded.SpanSynth({"312-299-0000": [(0, 12, "phone")]},
                                   {"phone": {"312-299-0000": "773-244-5678"}}, False)
        assert not registry.holds_phone(synth.rewrite("312-299-0000", lambda s: s))


def test_plan_collects_cross_file_dropped_outside_and_forced_local_phone_bases():
    p = ColumnProfile(index=0, name="Code", type=ColumnType.PHONE)
    p.model = ColumnModel(nonmissing_counts={"244-5678": 1})
    a = TableProfile(name=None, header=["Code"], n_rows=1, columns=[p])
    dropped = ColumnProfile(index=0, name="Hidden", type=ColumnType.EMPTY, role="excluded")
    b = TableProfile(name=None, header=["Hidden"], n_rows=1, columns=[dropped])
    b.private.extra[texts.DROPPED_KEY] = {0: frozenset({"３１２-２６６-５４３２ ext. 12", "2999999"})}
    b.private.extra["_dropped_phone_columns"] = {0}
    # The outside registry is the same input used by title/metadata generation.
    registry = texts.RealTexts(outside=[("b.csv", ["Call +44 20 7946 0958"])])
    a.private.extra[texts.RUN_KEY] = registry
    b.private.extra[texts.RUN_KEY] = registry
    planned = leaks.plan_digit_keys([SimpleNamespace(tables=[a]), SimpleNamespace(tables=[b])])
    assert planned.holds_phone("(415) 244-5678")
    assert planned.holds_phone("3122665432")
    assert planned.holds_phone("(312) 299-9999")
    assert planned.holds_phone("+44-20-7946-0958")


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows(rows)


@pytest.mark.parametrize("drop", ["--exclude", "--null"])
def test_written_forced_phone_is_checked_against_another_files_dropped_local(tmp_path, monkeypatch, drop):
    source = tmp_path / "input"
    _write(source / "a.csv", [["Student_ID", "Code"], *[[f"S{i:03d}", f"312-244-{i:04d}"] for i in range(20)]])
    _write(source / "b.csv", [["Student_ID", "Hidden_Phone"], ["T001", "266-5432 ext. 12"]])
    original = pipeline.structure_mod.finalize_rows

    def plant(tp, st, *args):
        messages = original(tp, st, *args)
        for col in st.columns:
            if col.name == "Code":
                col.values[0] = "+1 (415) 266-5432 x99"
        return messages

    monkeypatch.setattr(pipeline.structure_mod, "finalize_rows", plant)
    out, rep = tmp_path / "out", tmp_path / "rep"
    result = run_cli(source, "--out-dir", out, "--report-dir", rep, "--pii-col", "Code=phone",
                     drop, "b.csv::Hidden_Phone", "--seed", "51")
    assert result.exit_code == 3, result.output
    report = json.loads((rep / "report.json").read_text(encoding="utf-8"))
    a = next(f for f in report["files"] if f["source"] == "a.csv")
    assert any("phone digits" in reason for reason in a["reasons"])
    assert not (out / "a.csv").exists()
    assert "266-5432" not in json.dumps(report)
