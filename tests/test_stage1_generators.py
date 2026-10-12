"""Stage 1 generator changes preserve outputs, RNG state, and privacy predicates."""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest

from datanonymous.generate import embedded, pii, text
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType
from datanonymous.privacy import digits, texts


def _uncached_phone(profile, groups, pyr):
    """D46 deliberately changes phones; compare its cached and uncached parser paths.

    The original Stage 1 oracle remains in its validated milestone commit.
    """
    with patch.object(pii, "phone_parts", digits.phone_parts):
        make = pii._make_phone(profile, ColumnModel(), groups, None, pyr)

    def draw(group, attempt):
        with patch.object(pii, "phone_parts", digits.phone_parts):
            return make(group, attempt)

    return draw


@pytest.mark.parametrize("seed", [1, 91, 20261011])
@pytest.mark.parametrize("minimum", [1, 10, 1000])
def test_phone_templates_keep_each_rng_draw(seed, minimum):
    values = ["(312) 244-5678 ext. 123", "+1 415.278.3398", "972-442-5678", "244-5678",
              "+44 20 7946 0958", "not a phone"]
    groups = [text.Group(value, value, [(value, count)], count)
              for value, count in zip(values, [50, 20, 1, 5, 2, 3])]
    profile = ColumnProfile(index=0, name="phone", type=ColumnType.PHONE, params={"_min_cell": minimum})
    old_rng, new_rng = random.Random(seed), random.Random(seed)
    old = _uncached_phone(profile, groups, old_rng)
    new = pii._make_phone(profile, ColumnModel(), groups, None, new_rng)
    assert old_rng.getstate() == new_rng.getstate()
    for attempt in range(45):
        for group in groups:
            assert new(group, attempt) == old(group, attempt)
            assert old_rng.getstate() == new_rng.getstate()


def test_occupied_555_space_keeps_cached_output_and_rng():
    value = "312-244-5678"
    group = text.Group(value, value, [(value, 10)], 10)
    profile = ColumnProfile(index=0, name="phone", type=ColumnType.PHONE, params={"_min_cell": 10})
    old_rng, new_rng = random.Random(37), random.Random(37)
    old_make = _uncached_phone(profile, [group], old_rng)
    new_make = pii._make_phone(profile, ColumnModel(), [group], None, new_rng)
    old, new = text.Drawer({value}, memo_guard=False), text.Drawer({value})
    # Occupying the old fictional exchange leaves the ordinary exchange space.
    occupied = {f"312-555-{line:04d}" for line in range(10_000)}
    old.used.update(occupied)
    new.used.update(occupied)
    for _ in range(12):
        assert new.draw(lambda a: new_make(group, a), new_rng) == old.draw(lambda a: old_make(group, a), old_rng)
        assert new_rng.getstate() == old_rng.getstate()
        assert new.used == old.used


def test_guard_memo_preserves_rng_and_bypasses_ips_and_emails():
    memo = embedded.GuardMemo(maxsize=3)
    old_rng, new_rng = random.Random(131), random.Random(131)
    candidates = ["312-244-5678", "plain filler", "123-45-6789", "call 8.8.8.8",
                  "192.0.2.1", "alfa@example.org", "alfa@district.edu", "312-555-1234"]
    for value in candidates * 4:
        assert memo(value, new_rng) == embedded.guard(value, old_rng)
        assert new_rng.getstate() == old_rng.getstate()
        assert len(memo.cache) <= 3
        if "@" in value or any(kind == "ip" for _, _, kind in embedded.pii_spans(value, address=False)):
            assert value not in memo.cache


def test_guard_memo_returns_current_unchanged_object():
    memo = embedded.GuardMemo()
    first = "".join(["312-555-", "0199"])
    equal = "".join(["312-", "555-0199"])
    assert first == equal and first is not equal
    pyr = random.Random(1)
    assert memo(first, pyr) is first
    assert memo(equal, pyr) is equal
    ordinary = "312-244-0199"
    assert memo(ordinary, pyr) is ordinary
    assert text.Drawer(set()).guard is not text.Drawer(set()).guard


def test_lengthening_uses_memo_identity_without_changing_rng():
    memo = embedded.GuardMemo()
    old_rng, new_rng = random.Random(79), random.Random(79)
    for value in ["312-555-0199", "x@example.org", "name", "8.8.8.8"] * 4:
        assert text._lengthen_safe(value, new_rng, memo) == text._lengthen_safe(value, old_rng)
        assert new_rng.getstate() == old_rng.getstate()


def test_span_cache_keeps_options_and_cannot_be_mutated_through_public_api():
    embedded._cached_spans.cache_clear()
    for digit_phone in (False, True):
        for address in (False, True):
            value = "3125550199; 41 Main St; person@district.org"
            expected = embedded._find_spans(value, digit_phone, address)
            actual = embedded.pii_spans(value, digit_phone, address)
            assert actual == expected
            actual.clear()
            assert embedded.pii_spans(value, digit_phone, address) == expected
    before = embedded._cached_spans.cache_info().currsize
    value = "x " * (embedded._CACHE_TEXT_LIMIT + 1)
    assert embedded.pii_spans(value) == []
    assert embedded._cached_spans.cache_info().currsize == before


def test_name_word_cache_returns_fresh_lists_and_folds_names():
    value = "Mrs. María-Ann O'Brien Jr"
    assert pii.person_words(value) == ["maria", "ann", "brien"]
    pii.person_words(value).clear()
    assert pii.person_words(value) == ["maria", "ann", "brien"]
    assert pii.whole_name_key("Smith Smith") == "smith"
    assert pii.whole_name_key("J. K.") == "#jk"
    assert pii.whole_name_key(value) is None


def test_name_rows_keep_first_appearance_and_legacy_set_order():
    rows = [("María", "García"), ("Alex", None), ("María", "García"), ("Alex", "Smith"),
            (None, "Smith"), ("Alex", None), ("", "")]
    columns = [np.asarray(values, dtype=object) for values in zip(*rows)]
    for column in columns:
        column.flags.writeable = False
    distinct = list(pii._distinct_name_rows(columns))
    assert distinct == list(dict.fromkeys(rows))
    assert list(set(distinct)) == list(set(rows))


def test_cheap_privacy_negatives_agree_with_active_registries():
    digit_keys = digits.RealDigitKeys(frozenset({"3125550199", "02079460958"}), frozenset({"123456789"}),
                                    digits.NANP_LAYOUTS | {(3, 4, 4)})
    registry = texts.RealTexts(keys=frozenset({"ana garcía", "iⅰ", "plain filler"}),
                              locals=frozenset({"123", "first.last"}))
    candidates = ["", "1", "12345678", "123-45-6789", "312-555-0199", "020 7946 0958",
                  "１２３４５６７８９", "١٢٣٤٥٦٧٨٩", "Ana García", "IⅠ", "plain filler", "120.25",
                  "123@456.789", "first.last@example.org", "label3125550199", "123‐45‐6789"]
    with digits.guard(digit_keys), texts.guard(registry):
        for value in candidates:
            assert text.real_digits(value) == digits.holds_real(value)
            assert text.real_text(value) == texts.holds_real(value)
