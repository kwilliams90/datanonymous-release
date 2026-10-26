"""Sharing D42: local person support, suppression, immutable source evidence."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from datanonymous.config import Settings
from datanonymous.errors import GenerationError
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType as T, EntityProfile, RawColumn, RawTable, TableProfile
from datanonymous.profile import sharing
from datanonymous.profile.structure import sample_missing_mask


def table(values, ids=None, kind=T.FACTOR, missing=(), name="Status", params=None, subtype=None):
    n = len(values)
    cols, rawcols = [], []
    if ids is not None:
        cols.append(ColumnProfile(0, "Student_ID", T.IDENTIFIER, role="id", source="default-id"))
        rawcols.append(RawColumn("Student_ID", np.array(ids, dtype=object)))
    i = len(cols)
    nm, ms = Counter(), Counter()
    for v in values:
        (ms if v in missing or v is None or v == "" else nm)[v] += 1
    p = ColumnProfile(i, name, kind, subtype=subtype, params=params or {}, stats={"n": n}, na_tokens=frozenset(missing))
    p.model = ColumnModel(nonmissing_counts=dict(nm), missing_counts=dict(ms), real_values=frozenset(str(v) for v in values))
    cols.append(p)
    rawcols.append(RawColumn(name, np.array(values, dtype=object)))
    tp = TableProfile(None, [c.name for c in cols], n, cols)
    if ids is not None:
        tp.entity = EntityProfile(id_columns=[0], primary=0, n_entities=len(set(ids)))
        tp.private.id_raw[0] = rawcols[0].values
    raw = RawTable(None, tp.header, rawcols, n)
    settings = Settings()
    sharing.record_evidence(tp, raw, settings)
    return tp, settings


def test_vocab_is_suppressed_by_people_not_repeated_rows_and_source_is_unchanged():
    values = ["IEP"] * 20 + ["General"] * 20
    ids = ["S000"] * 20 + [f"S{i:03d}" for i in range(1, 21)]
    source, settings = table(values, ids)
    before = deepcopy(source.to_public())
    protected = sharing.protect_profile(source, settings)
    p = protected.columns[1]
    assert p.model.nonmissing_counts == {"General": 40}
    assert source.columns[1].model.nonmissing_counts == {"IEP": 20, "General": 20}
    assert source.to_public() == before
    assert p.model.real_values is source.columns[1].model.real_values
    assert p.model.extra[sharing.SOURCE_MODEL_KEY] is source.columns[1].model


def test_other_table_and_lookup_counts_do_not_support_rare_fact_membership():
    fact, settings = table(["504"] + ["General"] * 20, [f"S{i:03d}" for i in range(21)])
    fact.columns[1].params["_d35_holders"] = {"504": 9999}
    protected = sharing.protect_profile(fact, settings)
    assert "504" not in protected.columns[1].model.nonmissing_counts
    lookup, _ = table(["504", "General", "IEP"])
    lookup.private.extra[sharing.LOOKUP_KEY] = True
    assert sharing.protect_profile(lookup, settings).columns[0].model.nonmissing_counts == lookup.columns[0].model.nonmissing_counts


def test_no_id_is_not_assumed_to_mean_one_person_per_row():
    source, settings = table(["General"] * 40)
    with pytest.raises(GenerationError, match="distinct-person"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("name,kind,actors", [
    ("Parent_Name", T.PERSON_NAME, ["Clara O'Baxen", "Daria D'Buxen", "Elena D'Bixen"]),
    ("Guardian_Email", T.EMAIL, ["baaan@first.test", "babcn@other.test", "bacdn@other.test"]),
    ("Teacher_Phone", T.PHONE, ["212-555-0101", "312-555-0102", "312-555-0103"]),
    ("Staff_Address", T.STREET_ADDRESS, ["11 Birch St", "12 Cedar Rd", "13 Maple Rd"]),
    ("Parent_City", T.CITY, ["Austin", "Boston", "Denver"]),
    ("Guardian_ZIP", T.ZIP, ["60601", "60602", "60603"]),
    ("Staff_DOB", T.DATE, ["1978-01-02", "1981-01-15", "1982-01-24"]),
    ("Teacher_Age", T.INTEGER, ["31", "42", "53"]),
    ("Parent_Education", T.FACTOR, ["Graduate", "College", "Secondary"]),
    ("Guardian_Income", T.FLOAT, ["42000.50", "52000.75", "62000.25"]),
    ("Teacher_Salary", T.FLOAT, ["42000.50", "52000.75", "62000.25"]),
    ("Staff_Race", T.FACTOR, ["Group A", "Group B", "Group C"]),
    ("Staff_Employment_Status", T.FACTOR, ["Permanent", "Temporary", "Contract"]),
    ("Teacher_Certification", T.FACTOR, ["Level A", "Level B", "Level C"]),
    # Header ownership survives an explicit type override or fallback typing.
    ("Parent_Name", T.HIGH_CARD_CATEGORICAL, ["Clara", "Daria", "Elena"]),
    ("Guardian1First", T.FACTOR, ["Clara", "Daria", "Elena"]),
    ("MotherName1", T.FREE_TEXT, ["Clara", "Daria", "Elena"]),
    ("Created_By_Email", T.EMAIL, ["baaan@first.test", "babcn@other.test", "bacdn@other.test"]),
    # One context per actor bypasses the repeated embedded-relationship guard.
    ("Teacher_Comments", T.FREE_TEXT, ["Contact baaan@first.test", "Contact babcn@other.test", "Contact bacdn@other.test"]),
    ("Parent_Notes", T.FREE_TEXT, ["Call 212-555-0101", "Call 312-555-0102", "Call 312-555-0103"]),
    ("Guardian_Comments", T.FREE_TEXT, ["Visit 11 Birch St", "Visit 12 Cedar Rd", "Visit 13 Maple Rd"]),
    ("Staff_Notes", T.FACTOR, ["Contact baaan@first.test", "Contact babcn@other.test", "Contact bacdn@other.test"]),
])
def test_intrinsic_other_person_attributes_cannot_count_students_as_actors(name, kind, actors):
    values = [actor for actor in actors for _ in range(20)]
    source, settings = table(values, [f"S{i:03d}" for i in range(60)], kind=kind, name=name)
    evidence = source.private.extra[sharing.EVIDENCE_KEY]
    assert evidence["people"]  # sixty students, only three attribute owners
    assert all(1 not in evidence[key] for key in ("counts", "classes", "numbers", "date_buckets"))
    before = deepcopy(source.to_public())
    with pytest.raises(GenerationError, match="intrinsic attribute of another person") as error:
        sharing.protect_profile(source, settings)
    assert "--person-id" not in error.value.fix
    assert source.to_public() == before


def test_actor_attribute_refusal_is_not_bypassed_by_lookup_or_inflated_evidence():
    source, settings = table(["College"] * 60, [f"S{i:03d}" for i in range(60)], name="Parent_Education")
    source.private.extra[sharing.LOOKUP_KEY] = True
    source.private.extra[sharing.EVIDENCE_KEY]["counts"][1] = {"u:college": 9999}
    with pytest.raises(GenerationError, match="support for that actor"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("role", ["null", "excluded"])
def test_null_or_excluded_actor_attributes_do_not_block_sharing(role):
    source, settings = table(["College"] * 60, [f"S{i:03d}" for i in range(60)], name="Parent_Education")
    source.columns[1].role = role
    assert sharing.protect_profile(source, settings).columns[1].role == role


@pytest.mark.parametrize("name", ["Teacher_Comments", "Parent_Notes", "Guardian_Student_Comments", "Staff_Comment"])
def test_other_person_role_word_does_not_make_student_comments_an_actor_attribute(name):
    source, settings = table(["Completed work"] * 60, [f"S{i:03d}" for i in range(60)], kind=T.FREE_TEXT, name=name)
    protected = sharing.protect_profile(source, settings)
    assert protected.columns[1].model.nonmissing_counts == {"Completed work": 60}


@pytest.mark.parametrize("name", ["Name", "Student_Name", "StudentName", "Pupil_Name"])
def test_own_subject_names_keep_their_distinct_person_support(name):
    names = [f"Ba{chr(97 + i)}an Wilson" for i in range(20)]
    source, settings = table(names, [f"S{i:03d}" for i in range(20)], kind=T.PERSON_NAME, name=name)
    assert set(source.private.extra[sharing.EVIDENCE_KEY]["classes"][1].values()) == {20}
    assert sharing.protect_profile(source, settings).columns[1].model.nonmissing_counts == Counter(names)


def test_student_contact_events_do_not_require_actor_attribute_support():
    source, settings = table(["2024-10-04"] * 60, [f"S{i:03d}" for i in range(60)],
                             kind=T.DATE, name="Parent_Contact_Date")
    assert sharing.protect_profile(source, settings).columns[1].params[sharing.PERSON_DATE_KEY]


def test_a_constant_held_by_one_person_is_refused_even_with_many_rows():
    source, settings = table(["IEP"] * 50, ["S001"] * 50, kind=T.CONSTANT)
    with pytest.raises(GenerationError, match="no category"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("missing", ["", "N/A", None])
def test_rare_missing_tokens_cannot_be_restored_by_source_patterns(missing):
    source, settings = table([missing] + ["General"] * 20, [f"S{i:03d}" for i in range(21)], missing=("N/A",))
    protected = sharing.protect_profile(source, settings)
    assert protected.columns[1].model.missing_counts == {}
    mask = sample_missing_mask(protected.structure.missingness, 21, np.random.default_rng(3))
    assert mask.shape == (21, 1) and not mask.any()
    assert source.columns[1].model.n_missing == 1


def test_multiselect_items_and_rare_combinations_are_suppressed():
    values = ["Reading;Math"] * 20 + ["Reading;IEP"] * 20
    ids = [f"S{i:03d}" for i in range(20)] + ["S999"] * 20
    source, settings = table(values, ids, kind=T.MULTI_SELECT, params={"delimiter": ";", "mode": "combinations"})
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.model.nonmissing_counts == {"Reading;Math": 40}


def test_fd_cannot_reintroduce_a_suppressed_label():
    source, settings = table(["IEP"] + ["General"] * 20, [f"S{i:03d}" for i in range(21)])
    source.structure.private["fds"] = [{"det": 0, "dep": 1, "mapping": {"S000": "IEP"}}]
    with pytest.raises(GenerationError, match="structural"):
        sharing.protect_profile(source, settings)


def test_one_row_roster_date_is_person_level_without_entity_role():
    source, settings = table([f"2001-02-{i % 28 + 1:02d}" for i in range(84)], [f"S{i:03d}" for i in range(84)],
                             kind=T.DATE, name="Enrollment_Date")
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.role == "row" and p.params[sharing.PERSON_DATE_KEY]


def test_rare_native_vocabulary_labels_are_removed_from_unused_label_metadata():
    source, settings = table(["IEP"] + ["General"] * 20, [f"S{i:03d}" for i in range(21)])
    source.columns[1].native = {"labels": {1: "IEP", 2: "General", 3: "504"}}
    protected = sharing.protect_profile(source, settings)
    assert protected.columns[1].native["labels"] == {2: "General"}
    assert source.columns[1].native["labels"] == {1: "IEP", 2: "General", 3: "504"}


def test_final_verification_uses_original_missing_and_combination_units():
    source, settings = table(["N/A", "Reading;IEP"] + ["Reading;Math"] * 20,
                             [f"S{i:03d}" for i in range(22)], kind=T.MULTI_SELECT,
                             missing=("N/A",), params={"delimiter": ";", "mode": "items"})
    protected = sharing.protect_profile(source, settings).columns[1]
    assert not sharing.verify_values(protected, ["Reading;Math"] * 22)
    assert sharing.verify_values(protected, ["N/A"])[0].code == "sharing_rare_unit_restored"
    assert sharing.verify_values(protected, [" n/a "])[0].code == "sharing_rare_unit_restored"
    assert sharing.verify_values(protected, ["Reading;IEP"])[0].code == "sharing_rare_unit_restored"


def test_multiselect_rare_combination_cannot_return_in_another_order():
    from datanonymous.generate.table import generate_values

    values = ["Reading;Math"] + ["Reading;Art"] * 20 + ["Math;Art"] * 20
    source, settings = table(values, [f"S{i:03d}" for i in range(41)], kind=T.MULTI_SELECT,
                             params={"delimiter": ";", "mode": "items"})
    protected = sharing.protect_profile(source, settings).columns[1]
    assert protected.params["mode"] == "combinations"
    assert sharing.verify_values(protected, ["Math; Reading"])[0].code == "sharing_rare_unit_restored"
    out = generate_values(protected, protected.model, 100, np.random.default_rng(3))[0]
    assert not sharing.verify_values(protected, out)


def test_arrow_dictionary_does_not_reexport_unused_rare_vocabulary():
    pa = pytest.importorskip("pyarrow")
    from datanonymous.io.arrowio import _dictionary_array

    source, settings = table(["IEP"] + ["General"] * 20, [f"S{i:03d}" for i in range(21)])
    labels = ["General", "IEP", "504"]
    source.columns[1].native = {"dictionary_values": labels, "dictionary": pa.array(labels)}
    protected = sharing.protect_profile(source, settings).columns[1]
    field = pa.field("Status", pa.dictionary(pa.int8(), pa.string(), ordered=True))
    written = _dictionary_array(np.array(["General"] * 21, dtype=object), field, protected.native, protected)
    assert written.dictionary.to_pylist() == ["General"]
    assert source.columns[1].native["dictionary_values"] == labels


def test_independent_missingness_replaces_rare_joint_patterns_without_rare_marginals():
    from datanonymous.model import MissingnessModel

    source, settings = table([""] * 20 + ["General"] * 20, [f"S{i:03d}" for i in range(40)])
    source.structure.missingness = MissingnessModel(columns=[1], patterns=[((False,), 20), ((True,), 20)])
    source.structure.missingness.cross = {0: "source-coupling"}
    protected = sharing.protect_profile(source, settings)
    assert protected.structure.missingness.patterns == []
    assert protected.structure.missingness.cross is None
    assert sample_missing_mask(protected.structure.missingness, 40, np.random.default_rng(3)).sum() == 20
    assert source.structure.missingness.cross == {0: "source-coupling"}


def test_private_entity_floors_and_public_level_metadata_cannot_restore_rare_units():
    source, settings = table(["IEP"] + ["General"] * 20, [f"S{i:03d}" for i in range(21)],
                             params={"levels": ["IEP", "General"], "level_order": ["IEP", "General"]})
    source.columns[1].role = "entity"
    source.private.entity_models[1] = ColumnModel(nonmissing_counts={"IEP": 1, "General": 20},
                                                  extra={"_row_counts": {"IEP": 9}, "floored_levels": 1})
    protected = sharing.protect_profile(source, settings)
    assert protected.columns[1].params["levels"] == ["General"]
    assert protected.columns[1].params["level_order"] == ["General"]
    assert "IEP" not in protected.private.entity_models[1].nonmissing_counts
    assert "floored_levels" not in protected.private.entity_models[1].extra
    assert "_row_counts" not in protected.private.entity_models[1].extra
    assert source.private.entity_models[1].extra["floored_levels"] == 1


def _broadened(source, settings):
    """The protected model of column 1, after checking that every class it writes is a held source class."""
    p = sharing.protect_profile(source, settings).columns[1]
    held = source.private.extra[sharing.EVIDENCE_KEY]["classes"][1]
    signatures = sharing._generation_classes(p)
    assert signatures and all(held.get(sig, 0) >= 10 for sig in signatures.values())
    assert p.flags.get("sharing_format_broadened")
    return p.model.nonmissing_counts


def test_rare_name_shape_or_frequency_refuses_unique_donor():
    source, settings = table(["Elena"] * 20 + ["Bea"], [f"S{i:03d}" for i in range(21)], kind=T.PERSON_NAME)
    with pytest.raises(GenerationError, match="format/frequency class"):
        sharing.protect_profile(source, settings)
    values = [f"v{i:03d}" for i in range(20)] + ["v100"] * 3
    ids = [f"S{i:03d}" for i in range(20)] + ["S100"] * 3
    source, settings = table(values, ids, kind=T.PERSON_NAME)
    with pytest.raises(GenerationError, match="format/frequency class"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("kind", [T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL])
def test_rare_invented_text_shape_or_frequency_moves_to_a_held_class(kind):
    # ADR-0004: the unique donor's class is never written; its rows take a held class
    source, settings = table(["Elena"] * 20 + ["Bea"], [f"S{i:03d}" for i in range(21)], kind=kind)
    assert _broadened(source, settings) == {"Elena": 20}  # one row is too few for another value of 20 rows
    values = [f"v{i:03d}" for i in range(20)] + ["v100"] * 3
    ids = [f"S{i:03d}" for i in range(20)] + ["S100"] * 3
    source, settings = table(values, ids, kind=kind)
    counts = _broadened(source, settings)
    assert "v100" not in counts and set(counts.values()) == {1} and len(counts) == 23
    assert all(sharing._shape(v) == "a000" for v in counts)


@pytest.mark.parametrize("rare,common", [("O'", "D'"), ("O\u2019", "D\u2019"), ("Ab'", "De'"), ("Mc", "De")])
def test_literal_surname_prefix_is_a_donor_class_even_with_the_same_generic_shape(rare, common):
    suffixes = ["B" + chr(97 + i // 26) + chr(97 + i % 26) + "an" for i in range(40)]
    values = ["Zorax " + (rare if i == 3 else common) + suffix for i, suffix in enumerate(suffixes)]
    assert len({sharing._shape(v) for v in values}) == 1
    source, settings = table(values, [f"S{i:06d}" for i in range(40)], kind=T.PERSON_NAME)
    with pytest.raises(GenerationError, match="format/frequency class"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("kind,values", [
    (T.PHONE, [f"312-234-{i:04d}" for i in range(20)]),
    (T.EMAIL, [f"v{i:03d}@school.org" for i in range(20)]),
    (T.STREET_ADDRESS, [f"{i:03d} Oak Road" for i in range(100, 120)]),
    (T.HIGH_CARD_CATEGORICAL, [f"v{i:03d}" for i in range(20)]),
])
def test_common_invented_formats_have_ambiguous_statistical_donors(kind, values):
    source, settings = table(values, [f"S{i:03d}" for i in range(20)], kind=kind)
    assert sharing.protect_profile(source, settings).columns[1].model.nonmissing_counts == source.columns[1].model.nonmissing_counts


def test_rare_phone_area_and_email_domain_classes_refuse():
    cases = [(T.PHONE, [f"312-234-{i:04d}" for i in range(20)] + ["773-234-1234"]),
             (T.EMAIL, [f"v{i:03d}@school.org" for i in range(20)] + ["v100@private.io"])]
    for kind, values in cases:
        source, settings = table(values, [f"S{i:03d}" for i in range(21)], kind=kind)
        with pytest.raises(GenerationError, match="format/frequency class"):
            sharing.protect_profile(source, settings)


def test_rare_embedded_phone_frequency_is_a_donor_class_even_in_distinct_notes():
    values = [f"Call 312-234-{i:04d} code {i:03d}" for i in range(20)]
    values += [f"Call 312-777-7777 code {i:03d}" for i in range(100, 110)]
    source, settings = table(values, [f"S{i:03d}" for i in range(20)] + ["S100"] * 10, kind=T.FREE_TEXT)
    with pytest.raises(GenerationError, match="embedded PII relationship"):
        sharing.protect_profile(source, settings)


def test_person_date_unique_month_donor_refuses_but_repeated_month_is_supported():
    values = [f"2001-02-{i + 1:02d}" for i in range(20)] + ["2001-05-01"]
    source, settings = table(values, [f"S{i:03d}" for i in range(21)], kind=T.DATE, name="DOB")
    with pytest.raises(GenerationError, match="distinct people"):
        sharing.protect_profile(source, settings)


def test_even_common_person_fd_and_template_grouping_are_refused():
    source, settings = table(["General"] * 20, [f"S{i:03d}" for i in range(20)])
    source.structure.private["fds"] = [{"det": 0, "dep": 1, "mapping": {}}]
    with pytest.raises(GenerationError, match="source person dependency"):
        sharing.protect_profile(source, settings)
    source.structure.private = {"templates": [{"target": 1, "parts": [("f", 0)], "groups": {"n": 20}}]}
    with pytest.raises(GenerationError, match="source template"):
        sharing.protect_profile(source, settings)


def test_parseable_missing_date_extreme_is_not_restored_by_missing_generation():
    from datanonymous.generate.table import generate_independent
    from test_sharing_dates import profile as date_profile

    values = ["2004-02-29"] * 20 + ["1900-01-01"] * 20
    params = date_profile(values).params
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.DATE,
                             missing=("1900-01-01",), name="DOB", params=params)
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.model.missing_counts == {} and p.model.n_nonmissing == 40
    out = generate_independent(p, p.model, 40, np.random.default_rng(5))[0]
    assert not sharing.verify_values(p, out)
    restored = sharing.verify_values(p, ["1900-01-01", "2004-02-29"])
    assert {m.code for m in restored} == {"sharing_rare_unit_restored", "sharing_date_extreme_restored", "sharing_date_cover_restored"}


def test_different_date_offsets_and_formats_need_local_person_support():
    from test_sharing_dates import profile as date_profile

    values = ["2004-02-03 10:15:00+00:00"] * 20 + ["2004-02-03 10:15:00-05:00"]
    params = date_profile(values).params
    source, settings = table(values, [f"S{i:03d}" for i in range(21)], kind=T.DATETIME, name="DOB", params=params)
    # ADR-0005: the unsupported offset is not written; its row takes the column's most common held value
    assert _broadened(source, settings) == {"2004-02-03 10:15:00+00:00": 21}


@pytest.mark.parametrize("rare,common,kind,missing,params", [
    ("SPEECH", "Speech", T.FACTOR, (), {}),
    (" Speech ", "Speech", T.CONSTANT, (), {}),
    ("na", "NA", T.FACTOR, ("na", "NA"), {}),
    ("Mobility;Speech", "Speech;Mobility", T.MULTI_SELECT, (), {"delimiter": ";", "mode": "combinations"}),
])
def test_exact_copied_variants_are_suppressed_despite_common_normalized_support(rare, common, kind, missing, params):
    source, settings = table([rare] + [common] * 39, [f"S{i:03d}" for i in range(40)],
                             kind=kind, missing=missing, params=params)
    p = sharing.protect_profile(source, settings).columns[1]
    assert rare not in p.model.nonmissing_counts and rare not in p.model.missing_counts
    assert sharing.verify_values(p, [rare])[0].code == "sharing_rare_unit_restored"


def test_a_rare_spelling_moves_with_its_whole_value_to_held_classes():
    # one person writes v000 and V000; the spelling set is rare, so both spellings take held classes (ADR-0005)
    values = [f"v{i:03d}" for i in range(1, 21)] + ["v000", "V000"]
    ids = [f"S{i:03d}" for i in range(1, 21)] + ["S000"] * 2
    source, settings = table(values, ids, kind=T.FREE_TEXT)
    counts = _broadened(source, settings)
    assert not {"v000", "V000"} & set(counts) and len(counts) == 22
    assert all(sharing._shape(v) == "a000" for v in counts)


def test_whole_value_leading_zero_is_a_retained_text_donor_class_and_is_broadened():
    values = [f"{41000+i} reference baaan" for i in range(39)] + ["00123 reference baaan"]
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.FREE_TEXT)
    counts = _broadened(source, settings)
    assert len(counts) == 40 and not [v for v in counts if v.startswith("0")]


def test_whole_name_variant_set_needs_local_person_support():
    values, ids = [], []
    for i in range(40):
        name = "Zorax Ba" + chr(97 + i // 26) + chr(97 + i % 26) + "n"
        forms = (name.lower(), name.title()) if i == 0 else (name.lower(), name.upper()) if i < 20 else (name.title(), name.upper())
        values.extend(forms)
        ids.extend([f"S{i:03d}"] * 2)
    source, settings = table(values, ids, kind=T.PERSON_NAME)
    with pytest.raises(GenerationError, match="format/frequency class"):
        sharing.protect_profile(source, settings)


def test_discrete_interior_donor_is_suppressed_and_supported_extrema_remain():
    from datanonymous.generate.table import generate_values

    values = ["0"] * 10 + ["1"] * 10 + ["2"] + ["3"] * 19
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.INTEGER, subtype="discrete")
    p = sharing.protect_profile(source, settings).columns[1]
    assert set(p.model.nonmissing_counts) == {"0", "1", "3"}
    assert set(generate_values(p, p.model, 40, np.random.default_rng(3))[0]) == {"0", "1", "3"}
    assert sharing.verify_values(p, [2])[0].code == "sharing_rare_unit_restored"
    assert sharing.verify_values(p, ["2.0"])[0].code == "sharing_rare_unit_restored"
    assert not sharing.verify_values(p, ["0.00", "3.000"])


def test_extended_discrete_interior_donor_is_suppressed_and_never_interpolated_back():
    from datanonymous.generate.table import generate_values
    from test_area_b_detect import detect

    values = ["$0.00"] * 10 + ["$1.00"] * 10 + ["$2.00"] + ["$3.00"] * 19
    detection = detect(values)
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.NUMERIC_FORMATTED, params=detection.params)
    p = sharing.protect_profile(source, settings).columns[1]
    assert set(p.model.nonmissing_counts) == {"$0.00", "$1.00", "$3.00"}
    out = generate_values(p, p.model, 400, np.random.default_rng(3))[0]
    assert set(out) == {"$0.00", "$1.00", "$3.00"}
    assert sharing.verify_values(p, ["$2.00"])[0].code == "sharing_rare_unit_restored"


def test_continuous_numeric_model_and_final_verifier_exclude_original_endpoints():
    from datanonymous.generate.table import generate_values

    values = [f"{i / 4:.2f}" for i in range(40)]
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.FLOAT, params={"decimals": 2})
    p = sharing.protect_profile(source, settings).columns[1]
    out = generate_values(p, p.model, 120, np.random.default_rng(5))[0]
    assert min(map(float, out)) > 0 and max(map(float, out)) < 9.75
    assert not sharing.verify_values(p, out)
    assert sharing.verify_values(p, [0.0, "9.7500"])[0].code == "sharing_numeric_extreme_restored"


def test_numeric_columns_without_strict_interior_support_refuse():
    source, settings = table(["0"] * 20 + ["1"] * 20, ["S001"] * 20 + ["S002"] * 20, kind=T.FLOAT)
    with pytest.raises(GenerationError, match="unsupported original extrema"):
        sharing.protect_profile(source, settings)


def test_numeric_endpoint_support_is_local_people_and_shared_by_entity_model():
    from datanonymous.generate import sharing_numeric

    values = ["0"] * 20 + ["1"] * 10 + ["2"] * 10
    ids = ["S001"] * 20 + [f"S{i:03d}" for i in range(10, 30)]
    source, settings = table(values, ids, kind=T.FLOAT)
    source.private.entity_models[1] = ColumnModel(nonmissing_counts={"0": 1, "1": 10, "2": 10})
    source.columns[1].params["_global_holders"] = {"0": 1000}
    ptable = sharing.protect_profile(source, settings)
    p = ptable.columns[1]
    assert set(p.model.nonmissing_counts) == {"1", "2"}
    assert p.model.extra[sharing_numeric.FORBIDDEN_KEY] == frozenset({"0"})
    assert ptable.private.entity_models[1].extra[sharing_numeric.FORBIDDEN_KEY] == frozenset({"0"})
    assert not sharing.verify_values(p, [2.0])
    assert sharing.verify_values(p, ["0.000"])[0].code == "sharing_numeric_extreme_restored"
    assert sharing_numeric.source_extremes(p) == frozenset({"0", "2e0"})


def test_numeric_common_constant_year_and_two_supported_extrema_are_feasible():
    from datanonymous.generate.table import generate_values

    for kind, values in ((T.INTEGER_YEAR, ["2024"] * 20), (T.FLOAT, ["0"] * 10 + ["1"] * 10)):
        source, settings = table(values, [f"S{i:03d}" for i in range(20)], kind=kind)
        p = sharing.protect_profile(source, settings).columns[1]
        assert set(p.model.nonmissing_counts) == set(values)
        out = generate_values(p, p.model, 20, np.random.default_rng(3))[0]
        assert not sharing.verify_values(p, out)
        assert not sharing.verify_values(p, values)


def test_numeric_endpoint_uses_the_more_protective_configured_threshold():
    source, settings = table(["0"] * 10 + ["1"] * 20 + ["2"] * 20,
                             [f"S{i:03d}" for i in range(50)], kind=T.FLOAT)
    settings.min_cell = 15
    p = sharing.protect_profile(source, settings).columns[1]
    assert "0" not in p.model.nonmissing_counts and "2" in p.model.nonmissing_counts


def test_numeric_endpoint_support_uses_normalized_aliases_and_numeric_missing_tokens():
    values = ["0"] * 5 + ["0.00"] * 5 + ["1"] * 10 + ["999"] * 10
    source, settings = table(values, [f"S{i:03d}" for i in range(30)], kind=T.FLOAT, missing=("999",))
    p = sharing.protect_profile(source, settings).columns[1]
    assert set(p.model.nonmissing_counts) == {"0", "0.00", "1"}
    assert p.model.missing_counts == {"999": 10}
    assert not sharing.verify_values(p, [0.0, 999.0])


def test_nine_endpoint_holders_fail_despite_repeated_rows_and_ten_holders_pass():
    low_ids = [f"S{i:03d}" for i in range(9)] * 3
    high_ids = [f"S{i:03d}" for i in range(20, 30)] * 2
    values = ["0"] * len(low_ids) + ["1"] * 20 + ["2"] * len(high_ids)
    ids = low_ids + [f"S{i:03d}" for i in range(40, 60)] + high_ids
    source, settings = table(values, ids, kind=T.FLOAT)
    p = sharing.protect_profile(source, settings).columns[1]
    assert "0" not in p.model.nonmissing_counts and "2" in p.model.nonmissing_counts
    assert sharing.verify_values(p, [0])[0].code == "sharing_numeric_extreme_restored"
    assert not sharing.verify_values(p, [2])


def test_wide_decimal_extreme_normalization_never_rounds_through_float():
    from decimal import Decimal
    from datanonymous.generate.table import generate_values

    values = [f"1234567890123456789012345.{i:03d}" for i in range(40)]
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.DECIMAL,
                             params={"scale": 3, "decimals": 3, "precision": 30})
    p = sharing.protect_profile(source, settings).columns[1]
    out = generate_values(p, p.model, 80, np.random.default_rng(3))[0]
    assert not sharing.verify_values(p, out)
    assert sharing.verify_values(p, [Decimal(values[0]), Decimal(values[-1])])[0].code == "sharing_numeric_extreme_restored"


def test_event_dates_with_safe_months_allow_composite_uniqueness_repair():
    values = [f"2001-02-{i % 28 + 1:02d}" for i in range(168)]
    source, settings = table(values, [f"S{i // 2:03d}" for i in range(168)], kind=T.DATE, name="Event_Date")
    source.structure.private["composite"] = [{"x": 1, "cols": [1]}]
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.params[sharing.PERSON_DATE_KEY]


@pytest.mark.parametrize("people,expected", [(9, False), (10, True)])
def test_excluded_date_bucket_support_counts_local_people_not_repeated_rows(people, expected):
    from datanonymous.generate import sharing_dates

    values = ["2008-01-01"] * (people * 3) + ["2008-01-15"] * 30
    ids = [f"S{i:03d}" for i in range(people)] * 3 + [f"S{i:03d}" for i in range(20, 50)]
    source, settings = table(values, ids, kind=T.DATE, name="DOB")
    source.private.entity_models[1] = ColumnModel(nonmissing_counts={"2008-01-01": people, "2008-01-15": 30})
    if not expected:
        with pytest.raises(GenerationError, match="excluded endpoint calendar bucket"):
            sharing.protect_profile(source, settings)
        return
    protected = sharing.protect_profile(source, settings)
    p = protected.columns[1]
    key = sharing_dates.calendar_bucket_keys(p, ["2008-01-01"])[0]
    assert p.model.extra[sharing_dates.BUCKET_HOLDERS_KEY][key] == people
    assert protected.private.entity_models[1].extra[sharing_dates.BUCKET_HOLDERS_KEY][key] == people
    out = sharing_dates.generate_dates(p, protected.private.entity_models[1], 100, np.random.default_rng(3))[0]
    assert not sharing.verify_values(p, out)
    assert sharing_dates.BUCKET_HOLDERS_KEY not in source.columns[1].model.extra


def test_singleton_date_bucket_donor_refuses_even_when_month_has_forty_people():
    source, settings = table(["2008-01-01"] + ["2008-01-15"] * 39,
                             [f"S{i:03d}" for i in range(40)], kind=T.DATE, name="DOB")
    with pytest.raises(GenerationError, match="excluded endpoint calendar bucket"):
        sharing.protect_profile(source, settings)


def test_calendar_bucket_person_support_is_not_borrowed_from_other_tables_or_row_counts():
    source, settings = table(["2008-01-01"] * 30 + ["2008-01-15"] * 30, kind=T.DATE, name="DOB")
    source.columns[0].params["_global_holders"] = {"2008-01-01": 500, "2008-01-15": 500}
    with pytest.raises(GenerationError, match="distinct people"):
        sharing.protect_profile(source, settings)


def test_date_bucket_threshold_uses_configured_k_and_ignores_off_domain_sentinel_extreme():
    values = ["1900-01-01"] * 20 + ["2008-01-01"] * 10 + ["2008-01-15"] * 20
    ids = [f"S{i:03d}" for i in range(50)]
    source, settings = table(values, ids, kind=T.DATE, name="DOB", missing=("1900-01-01",))
    settings.min_cell = 15
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.params[sharing.PERSON_DATE_KEY]
    # The January 2008 first bucket is not an actual endpoint bucket: the
    # source minimum is the out-of-domain missing sentinel from 1900.
    source, settings = table(["2008-01-01"] * 10 + ["2008-01-15"] * 20,
                             [f"S{i:03d}" for i in range(30)], kind=T.DATE, name="DOB")
    settings.min_cell = 15
    with pytest.raises(GenerationError, match="excluded endpoint calendar bucket"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("kind,rare,common", [(T.STATE, "AK", "CA"), (T.ZIP, "60601", "60602")])
def test_rare_geographic_levels_are_suppressed_locally(kind, rare, common):
    source, settings = table([rare] + [common] * 20, [f"S{i:03d}" for i in range(21)], kind=kind)
    p = sharing.protect_profile(source, settings).columns[1]
    assert p.model.nonmissing_counts == {common: 21}


def test_city_state_suffix_needs_local_donor_support():
    values = ["A" + chr(97 + i) + "a, CA" for i in range(20)] + ["Zza, AK"]
    source, settings = table(values, [f"S{i:03d}" for i in range(21)], kind=T.CITY)
    with pytest.raises(GenerationError, match="format/frequency class"):
        sharing.protect_profile(source, settings)


def test_embedded_span_whole_context_set_needs_local_donor_support():
    values, ids = [], []
    words = {7: "Related", 8: "Reported", 9: "Submitted"}
    for i in range(40):
        stem = "ba" + chr(97 + i // 26) + chr(97 + i % 26) + "n"
        sizes = (7, 8) if i == 0 else (7, 9) if i < 20 else (8, 9)
        values.extend(f"{words[size]} {stem}@commx.test" for size in sizes)
        ids.extend([f"S{i:03d}"] * 2)
    source, settings = table(values, ids, kind=T.FREE_TEXT)
    with pytest.raises(GenerationError, match="embedded PII relationship"):
        sharing.protect_profile(source, settings)


def test_regular_embedded_cooccurrence_graph_refuses_even_with_equal_classes():
    emails = [f"v{i:03d}@school.org" for i in range(20)]
    values = [f"From {emails[i]} to {emails[(i + step) % 20]}" for i in range(20) for step in (1, 7)]
    source, settings = table(values, [f"S{i:03d}" for i in range(40)], kind=T.FREE_TEXT)
    assert len(set(sharing._generation_classes(source.columns[1]).values())) == 1
    with pytest.raises(GenerationError, match="source graph"):
        sharing.protect_profile(source, settings)


@pytest.mark.parametrize("native", [
    {"kind": "excel", "storage": {"n": {"0.00": 39, "0.00000": 1}}},
    {"kind": "excel", "storage": {"n": {"0.00": 20, "0.00000": 20}}},
    {"kind": "excel", "storage": {"s": {"General": 39}, "n": {"General": 1}}},
    {"kind": "excel", "types": {"s": 39, "n": 1}, "formats": {"s": "General", "n": "0"}},
])
def test_mixed_excel_storage_refuses_without_joint_person_support(native):
    source, settings = table(["General"] * 40, [f"S{i:03d}" for i in range(40)])
    source.columns[1].native = native
    with pytest.raises(GenerationError, match="mixed Excel"):
        sharing.protect_profile(source, settings)


def test_excel_storage_guard_covers_person_ids_even_without_an_id_model():
    source, settings = table(["General"] * 40, [f"S{i:03d}" for i in range(40)])
    source.columns[0].native = {"kind": "excel", "storage": {"s": {"General": 39, "@": 1}}}
    with pytest.raises(GenerationError, match="mixed Excel"):
        sharing.protect_profile(source, settings)


def test_uniform_excel_storage_is_preserved_and_null_excluded_columns_skip_guard():
    source, settings = table(["General"] * 40, [f"S{i:03d}" for i in range(40)])
    source.columns[1].native = {"kind": "excel", "storage": {"s": {"General": 40, "@": 0}}}
    before = deepcopy(source.columns[1].native)
    assert sharing.protect_profile(source, settings).columns[1].native == before
    assert source.columns[1].native == before
    source.columns[1].native["storage"]["s"]["@"] = 1
    for role in ("null", "excluded"):
        source.columns[1].role = role
        assert sharing.protect_profile(source, settings).columns[1].role == role


def test_an_unprotectable_key_column_is_never_nulled_automatically():
    # ADR-0001: a key with a column refusal still refuses; only a non-key column is nulled
    source, settings = table(["General"] * 40, [f"S{i:03d}" for i in range(40)])
    source.columns[0].native = {"kind": "excel", "storage": {"s": {"General": 39, "@": 1}}}
    assert sharing.null_unprotectable_columns(source, settings) == []
    assert source.columns[0].role == "id"
    with pytest.raises(GenerationError, match="mixed Excel"):
        sharing.protect_profile(source, settings)
    source, settings = table(["General"] * 40, [f"S{i:03d}" for i in range(40)])
    source.columns[1].native = {"kind": "excel", "storage": {"n": {"0.00": 39, "0.00000": 1}}}
    assert sharing.null_unprotectable_columns(source, settings) == ["Status"]
    assert source.columns[1].role == "null" and source.columns[1].messages[-1].code == "column_nulled_automatically"
    sharing.protect_profile(source, settings)
