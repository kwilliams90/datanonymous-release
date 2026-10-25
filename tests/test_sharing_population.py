"""Invented adversarial and utility fixtures for the D47--D49 population planner."""

from collections import Counter
import numpy as np
import pytest

from datanonymous.config import Settings
from datanonymous.detect.ids import normalize_id
from datanonymous.errors import UnsupportedColumnError
from datanonymous.generate.ids import IdRegistry
from datanonymous.generate.population import PLAN_KEY, _fold_widths, generation_profile, plan_population
from datanonymous.generate.table import GenContext, generate_table
from datanonymous.model import (ColumnProfile, ColumnType, EntityProfile, FileResult, RawColumn, RawTable,
                               TablePrivate, TableProfile)
from datanonymous.profile.entity import entity_index
from datanonymous.profile.table import profile_table


def _file(name, columns, *, aliases=(), nesting=(), links=(), sheet=None):
    names = list(columns)
    raw = {i: np.asarray(columns[n], dtype=object) for i, n in enumerate(names)}
    n = len(raw[0])
    index, named, blank = entity_index(raw[0])
    sizes = np.bincount(index)
    profiles = [ColumnProfile(i, name, ColumnType.IDENTIFIER, role="id", source="person-id")
                for i, name in enumerate(names)]
    entity = EntityProfile(id_columns=list(raw), primary=0, n_entities=named + blank,
                           blank_id_rows=blank, rows_per_id=dict(Counter(sizes.tolist())),
                           multi_row_ids=int((sizes > 1).sum()), aliases=list(aliases), nesting=list(nesting))
    tp = TableProfile(sheet, names, n, profiles, entity=entity,
                      private=TablePrivate(id_raw=raw, entity_index=index))
    tp.structure.cross_file_keys = list(links)
    return FileResult(name, name, "csv", tables=[tp])


def _registry(files):
    registry = IdRegistry()
    namespaces = {}
    for fr in files:
        for ti, tp in enumerate(fr.tables):
            ns = {}
            for col in tp.entity.id_columns:
                name = tp.columns[col].name
                ns[col] = name
                registry.register(name, fr.rel_path, tp.name, col, name, tp.private.id_raw[col])
            namespaces[(fr.rel_path, ti)] = ns
    # Fidelity mappings are intentionally absent: the sharing planner must not read them.
    return registry, namespaces


def _plan(files, seed=512, minimum=10):
    registry, namespaces = _registry(files)
    plans = plan_population(files, seed, Settings(min_cell=minimum), registry, namespaces)
    return plans, registry, namespaces


def _ids(n=80, prefix="S"):
    return [f"{prefix}{100000 + i * 37}" for i in range(n)]


def _normalized(values):
    return [normalize_id(str(v)) for v in values]


def _foreign(column="student_id", parent="roster.csv::student_id"):
    return {"column": column, "parent": parent, "key_containment": 1.0, "orphan_rows": 0,
            "foreign_key": True}


def test_independent_counts_memberships_and_valid_parent_join():
    people = _ids()
    roster = _file("roster.csv", {"student_id": people})
    counts = [2] * 20 + [4] * 20
    first = _file("first.csv", {"student_id": np.repeat(people[:40], counts)}, links=[_foreign()])
    second = _file("second.csv", {"student_id": np.repeat(people[:40], counts)}, links=[_foreign()])
    plans, _, _ = _plan([roster, first, second])
    parent = set(plans[("roster.csv", 0)].id_values[0])
    a, b = plans[("first.csv", 0)], plans[("second.csv", 0)]
    for plan in (a, b):
        assert len(plan.id_values[0]) == 120
        assert set(plan.id_values[0]) <= parent
        assert Counter(Counter(plan.id_values[0]).values()) == {2: 20, 4: 20}
        assert not set(plan.id_values[0]) & set(people)
    assert set(a.id_values[0]) != set(b.id_values[0])
    common = set(a.id_values[0]) & set(b.id_values[0])
    assert any(Counter(a.id_values[0])[k] != Counter(b.id_values[0])[k] for k in common)


def test_row_permutation_and_identity_relabeling_do_not_change_population():
    people = _ids()
    originals = np.repeat(people, [2] * 40 + [3] * 40)
    first, _, _ = _plan([_file("events.csv", {"student_id": originals})])
    rename = dict(zip(people, people[::-1]))
    changed = np.random.default_rng(33).permutation([rename[p] for p in originals])
    second, _, _ = _plan([_file("events.csv", {"student_id": changed})])
    a, b = first[("events.csv", 0)], second[("events.csv", 0)]
    np.testing.assert_array_equal(a.id_values[0], b.id_values[0])
    np.testing.assert_array_equal(a.entity_index, b.entity_index)
    np.testing.assert_array_equal(a.slots[0], b.slots[0])


def test_known_seed_cannot_replay_or_influence_a_source_mapping():
    file = _file("people.csv", {"student_id": _ids()})
    registry, namespaces = _registry([file])
    registry.namespaces["student_id"].mapping = {k: "FORBIDDEN" for k in _ids()}
    first = plan_population([file], 42, Settings(), registry, namespaces)
    registry.namespaces["student_id"].mapping = {k: f"DIFFERENT{i}" for i, k in enumerate(_ids())}
    second = plan_population([file], 42, Settings(), registry, namespaces)
    np.testing.assert_array_equal(first[("people.csv", 0)].id_values[0], second[("people.csv", 0)].id_values[0])
    assert "FORBIDDEN" not in first[("people.csv", 0)].id_values[0]


def test_unique_thirteen_event_donor_blocks_before_attaching_any_plan():
    roster = _file("roster.csv", {"student_id": _ids()})
    events = _file("events.csv", {"student_id": np.repeat(_ids(), [2] * 79 + [13])})
    with pytest.raises(UnsupportedColumnError, match="count class.*donors"):
        _plan([roster, events])
    assert PLAN_KEY not in roster.tables[0].private.extra
    assert PLAN_KEY not in events.tables[0].private.extra


def test_rare_id_format_donor_is_refused_and_named():
    with pytest.raises(UnsupportedColumnError, match="column 'student_id'.*format class"):
        _plan([_file("people.csv", {"student_id": _ids(79) + ["XX-123456"]})])


def test_rare_unpadded_id_width_is_folded_into_a_common_width():
    values = [str(1000 + i) for i in range(70)] + [str(100000 + i) for i in range(12)] + ["9000000", "12345678"]
    plans, _, _ = _plan([_file("people.csv", {"student_id": values})])
    out = plans[("people.csv", 0)].id_values[0]
    assert sorted(Counter(len(v) for v in out).items()) == [(4, 70), (6, 14)]
    assert not set(out) & set(values)


@pytest.mark.parametrize("widths, folded", [({4: 70, 6: 12, 7: 1, 8: 1}, {4: 70, 6: 14}),
                                            ({4: 3, 5: 3, 6: 40}, {6: 46}),
                                            ({4: 20, 6: 2, 8: 20}, {4: 20, 8: 22}),
                                            ({4: 4}, {4: 4})])
def test_fold_widths(widths, folded):
    assert _fold_widths(Counter(widths), 10) == Counter(folded)


def test_sequential_keys_keep_set_and_shuffle_rows_and_references():
    people = [str(i) for i in range(1, 81)]
    roster = _file("roster.csv", {"student_id": people})
    events = _file("events.csv", {"student_id": np.repeat(people[:40], 2)}, links=[_foreign()])
    plans, _, _ = _plan([roster, events])
    out = plans[("roster.csv", 0)].id_values[0].tolist()
    assert set(out) == set(people)
    assert out != people
    assert set(plans[("events.csv", 0)].id_values[0]) <= set(out)


def test_aliases_join_across_files_with_different_key_names():
    people, state = _ids(), _ids(prefix="T")
    roster = _file("roster.csv", {"student_id": people, "state_id": state}, aliases=[[0, 1]])
    events = _file("events.csv", {"state_id": np.repeat(state[:40], 2)},
                   links=[_foreign("state_id", "roster.csv::state_id")])
    plans, _, _ = _plan([roster, events])
    p = plans[("roster.csv", 0)]
    assert len(set(zip(p.id_values[0], p.id_values[1]))) == 80
    assert len(set(p.id_values[0])) == len(set(p.id_values[1])) == 80
    np.testing.assert_array_equal(p.slots[0], p.slots[1])
    assert set(plans[("events.csv", 0)].id_values[0]) <= set(p.id_values[1])


def test_padding_is_column_format_not_source_person_assignment():
    people = [str(10000 + i * 31) for i in range(80)]
    roster = _file("roster.csv", {"student_id": [s.zfill(8) for s in people]})
    events = _file("events.csv", {"student_id": np.repeat(people[:40], 2)}, links=[_foreign()])
    plans, _, _ = _plan([roster, events])
    a, b = plans[("roster.csv", 0)].id_values[0], plans[("events.csv", 0)].id_values[0]
    assert all(len(v) == 8 for v in a)
    assert set(_normalized(b)) <= set(_normalized(a))


def test_nesting_is_fictional_consistent_across_tables_and_joins():
    people, families = _ids(), _ids(20, "F")
    relation = {"child": "student_id", "parent": "family_id"}
    roster = _file("roster.csv", {"student_id": people, "family_id": np.repeat(families, 4)}, nesting=[relation])
    family_file = _file("families.csv", {"family_id": families})
    roster.tables[0].structure.cross_file_keys = [_foreign("family_id", "families.csv::family_id")]
    # Secondary family cardinalities may differ after membership regeneration (D48).
    events = _file("events.csv", {"student_id": np.repeat(people[:40], 2),
                                  "family_id": np.repeat(np.repeat(families[:10], 4), 2)},
                   nesting=[relation], links=[_foreign(), _foreign("family_id", "families.csv::family_id")])
    plans, _, _ = _plan([roster, family_file, events])
    p, e = plans[("roster.csv", 0)], plans[("events.csv", 0)]
    relation = dict(zip(p.id_values[0], p.id_values[1]))
    assert all(relation[s] == f for s, f in zip(e.id_values[0], e.id_values[1]))
    assert set(p.id_values[1]) <= set(plans[("families.csv", 0)].id_values[0])
    # Original families are contiguous in source-index order; new edges are not.
    assert len(np.unique(p.slots[1][:4])) > 1


def test_many_to_many_edges_preserve_composite_key_uniqueness():
    people, teachers = _ids(), _ids(20, "T")
    src_people = np.repeat(people, 2)
    src_teachers = [teachers[(i + j) % 20] for i in range(80) for j in range(2)]
    file = _file("links.csv", {"student_id": src_people, "teacher_id": src_teachers})
    plans, _, _ = _plan([file])
    p = plans[("links.csv", 0)]
    assert len(set(zip(p.id_values[0], p.id_values[1]))) == 160
    assert Counter(Counter(p.id_values[0]).values()) == {2: 80}


def test_nonzero_orphans_are_refused_with_explicit_contract_message():
    roster = _file("roster.csv", {"student_id": _ids()})
    link = _foreign()
    link["orphan_rows"] = 1
    events = _file("events.csv", {"student_id": np.repeat(_ids(40), 2)}, links=[link])
    with pytest.raises(UnsupportedColumnError, match="orphan.*contract"):
        _plan([roster, events])


def test_incompatible_alias_cardinalities_are_refused():
    aliases = _file("aliases.csv", {"student_id": _ids(40), "state_id": _ids(40, "T")}, aliases=[[0, 1]])
    full = _file("full.csv", {"student_id": _ids(80)})
    with pytest.raises(UnsupportedColumnError, match="alias namespaces.*cardinalities"):
        _plan([aliases, full])


def test_generation_copy_preserves_source_indices_and_drops_source_order_and_duplicates():
    file = _file("people.csv", {"student_id": _ids()})
    source = file.tables[0]
    source.structure.private["sort"] = {"keys": [(0, "asc", "first", "text")]}
    source.structure.duplicate_rows = {"rows": 7}
    plans, _, _ = _plan([file])
    before = source.private.entity_index.copy()
    generated = generation_profile(source, plans[("people.csv", 0)])
    assert generated is not source
    np.testing.assert_array_equal(source.private.entity_index, before)
    assert "sort" not in generated.structure.private
    assert generated.structure.duplicate_rows == {}
    assert source.structure.duplicate_rows == {"rows": 7}


def test_generator_uses_population_and_does_not_call_fidelity_renderer(monkeypatch):
    raw = RawTable(None, ["student_id", "score"],
                   [RawColumn("student_id", np.asarray(np.repeat(_ids(), 2), dtype=object)),
                    RawColumn("score", np.asarray([str(20 + i % 40) for i in range(160)], dtype=object))], 160)
    settings = Settings()
    tp = profile_table(raw, "data.csv", "csv", settings)
    file = FileResult("data.csv", "data.csv", "csv", tables=[tp])
    plans, registry, namespaces = _plan([file])
    before = tp.private.entity_index.copy()

    def forbidden(*_):
        raise AssertionError("sharing generation accessed the original identity mapper")

    monkeypatch.setattr(registry, "render", forbidden)
    generated = generate_table(tp, GenContext(512, "data.csv", registry, namespaces[("data.csv", 0)], settings))
    np.testing.assert_array_equal(generated.columns[0].values, plans[("data.csv", 0)].id_values[0])
    np.testing.assert_array_equal(tp.private.entity_index, before)
    assert generated.profile is not tp
    assert generated.n_rows == raw.n_rows


def test_source_row_and_identity_permutation_leave_family_plan_unchanged():
    people, families = _ids(), _ids(20, "F")
    relation = {"child": "student_id", "parent": "family_id"}
    a = _file("people.csv", {"student_id": people, "family_id": np.repeat(families, 4)}, nesting=[relation])
    b = _file("people.csv", {"student_id": people[::-1], "family_id": np.tile(families[::-1], 4)}, nesting=[relation])
    first, _, _ = _plan([a])
    second, _, _ = _plan([b])
    for col in (0, 1):
        np.testing.assert_array_equal(first[("people.csv", 0)].id_values[col], second[("people.csv", 0)].id_values[col])


def test_empty_identifier_table_can_be_planned():
    file = _file("empty.csv", {"student_id": []})
    plans, _, _ = _plan([file])
    assert len(plans[("empty.csv", 0)].id_values[0]) == 0


def test_transitive_nesting_is_composed_instead_of_independently_redrawn():
    relations = [{"child": "student_id", "parent": "family_id"},
                 {"child": "family_id", "parent": "household_id"},
                 {"child": "student_id", "parent": "household_id"}]
    file = _file("people.csv", {"student_id": _ids(80), "family_id": np.repeat(_ids(40, "F"), 2),
                                "household_id": np.repeat(_ids(20, "H"), 4)}, nesting=relations)
    plans, _, _ = _plan([file])
    p = plans[("people.csv", 0)]
    mapping = {}
    for family, household in zip(p.id_values[1], p.id_values[2]):
        assert family not in mapping or mapping[family] == household
        mapping[family] = household
    assert len(mapping) == 40


def test_luhn_contract_is_preserved_without_original_assignment():
    from datanonymous.detect.ids import luhn_check_digit, luhn_valid

    values = [str(i) + str(luhn_check_digit(str(i))) for i in range(10000, 10080)]
    file = _file("people.csv", {"student_id": values})
    registry, namespaces = _registry([file])
    registry.namespaces["student_id"].check = "luhn"
    plans = plan_population([file], 512, Settings(), registry, namespaces)
    output = plans[("people.csv", 0)].id_values[0]
    assert all(luhn_valid(value) for value in output)
    assert not set(output) & set(values)


def test_uuid_random_character_classes_do_not_create_false_rare_formats():
    import uuid

    values = [str(uuid.UUID(int=i * 8273689927 + 238763937462352356512351513)) for i in range(80)]
    file = _file("people.csv", {"student_id": values})
    plans, _, _ = _plan([file])
    output = plans[("people.csv", 0)].id_values[0]
    assert all(str(uuid.UUID(value)) == value for value in output)
    assert not set(output) & set(values)


def test_primary_parent_and_unique_children_preserve_nesting():
    relation = {"child": "student_id", "parent": "family_id"}
    file = _file("people.csv", {"family_id": np.repeat(_ids(20, "F"), 4), "student_id": _ids(80)},
                 nesting=[relation])
    plans, _, _ = _plan([file])
    p = plans[("people.csv", 0)]
    assert len(set(p.id_values[1])) == 80
    assert Counter(Counter(p.id_values[0]).values()) == {4: 20}


def test_new_original_key_values_do_not_change_fictional_membership_or_count_assignment():
    first = _file("people.csv", {"student_id": np.repeat(_ids(80), [2] * 40 + [3] * 40)})
    other_keys = [f"S{700000 + i * 13}" for i in range(80)]
    second = _file("people.csv", {"student_id": np.repeat(other_keys, [3] * 40 + [2] * 40)})
    a, _, _ = _plan([first])
    b, _, _ = _plan([second])
    np.testing.assert_array_equal(a[("people.csv", 0)].slots[0], b[("people.csv", 0)].slots[0])
    np.testing.assert_array_equal(a[("people.csv", 0)].entity_index, b[("people.csv", 0)].entity_index)


def test_non_id_composite_keys_are_repaired_on_fictional_blocks():
    from datanonymous.generate.rng import stream
    from datanonymous.profile.structure import finalize_rows

    ids = np.asarray(np.repeat(_ids(), 2), dtype=object)
    dates = np.asarray(["2024-09-03", "2024-09-04"] * 80, dtype=object)
    raw = RawTable(None, ["student_id", "event_date"],
                   [RawColumn("student_id", ids), RawColumn("event_date", dates)], 160)
    settings = Settings()
    tp = profile_table(raw, "events.csv", "csv", settings)
    file = FileResult("events.csv", "events.csv", "csv", tables=[tp])
    _, registry, namespaces = _plan([file])
    assert tp.structure.composite_keys
    generated = generate_table(tp, GenContext(512, "events.csv", registry, namespaces[("events.csv", 0)], settings))
    messages = finalize_rows(generated.profile, generated, lambda purpose: stream(512, purpose), settings)
    assert not any(m.code == "composite_key_duplicates" for m in messages)
    assert len(set(zip(generated.columns[0].values, generated.columns[1].values))) == 160


def test_failed_namespace_is_excluded_from_planning_but_not_collision_exclusion():
    live = _file("live.csv", {"student_id": _ids()})
    failed = _file("failed.csv", {"staff_id": [str(10000 + i) for i in range(80)]})
    registry, namespaces = _registry([live, failed])
    failed.fail("invented failed input")
    plans = plan_population([live, failed], 512, Settings(), registry, namespaces)
    assert set(plans) == {("live.csv", 0)}
    assert PLAN_KEY not in failed.tables[0].private.extra


def test_inverse_roster_coverage_is_not_mistaken_for_a_foreign_key():
    roster = _file("roster.csv", {"student_id": _ids()}, links=[
        {"column": "student_id", "parent": "iep.csv::student_id", "key_containment": 0.5,
         "orphan_rows": 40, "foreign_key": False}])
    subset = _file("iep.csv", {"student_id": _ids(40)}, links=[_foreign()])
    plans, _, _ = _plan([roster, subset])
    all_people, iep = plans[("roster.csv", 0)], plans[("iep.csv", 0)]
    assert len(iep.id_values[0]) == 40
    assert set(iep.id_values[0]) <= set(all_people.id_values[0])
    assert any(slot >= 40 for slot in iep.slots[0])


@pytest.mark.parametrize("actual_fk, reverse", [(True, True), (False, False)])
def test_partial_join_without_proved_inverse_coverage_remains_a_refusal(actual_fk, reverse):
    roster = _file("roster.csv", {"student_id": _ids()}, links=[
        {"column": "student_id", "parent": "iep.csv::student_id", "key_containment": 0.5,
         "orphan_rows": 40, "foreign_key": actual_fk}])
    subset = _file("iep.csv", {"student_id": _ids(40)}, links=[_foreign()] if reverse else [])
    with pytest.raises(UnsupportedColumnError, match="orphan.*contract"):
        _plan([roster, subset])
