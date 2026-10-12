"""Type detection: example acceptance (criterion 3) and table-driven rule tests (design 8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from datanonymous import hints
from conftest import build_fidelity_settings as build_settings
from datanonymous.detect.context import DetectContext
from datanonymous.detect.types import detect_column
from datanonymous.errors import UnsupportedColumnError
from datanonymous.io import csvio
from datanonymous.model import ColumnType as T
from datanonymous.profile.table import profile_table

from conftest import EXAMPLES


def settings(**cli):
    return build_settings({k: v for k, v in cli.items()})


def detect(values, name="col", **cli):
    s = settings(**cli)
    arr = np.empty(len(values), dtype=object)
    arr[:] = list(values)
    ctx = DetectContext(index=0, name=name, values=arr, file_rel="f.csv", table=None, format="csv",
                        options=s.column_options("f.csv", None, name), settings=s)
    return detect_column(ctx)


def profile_example(name: str, **cli):
    s = settings(**cli)
    r = csvio.parse_text_bytes((EXAMPLES / name).read_bytes())
    tp = profile_table(r.tables[0], name, "csv", s)
    return {p.name: p for p in tp.columns}, tp


# --------------------------------------------------------------------------- acceptance (criterion 3)
def test_students_performance_types() -> None:
    cols, _ = profile_example("StudentsPerformance.csv")
    assert cols["gender"].type is T.FACTOR
    assert cols["race/ethnicity"].type is T.FACTOR
    assert cols["parental level of education"].type is T.ORDINAL
    assert cols["parental level of education"].params["levels"] == [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    assert cols["lunch"].type is T.FACTOR
    tpc = cols["test preparation course"]
    assert tpc.type is T.FACTOR
    assert set(tpc.model.nonmissing_counts) == {"none", "completed"} and not tpc.model.missing_counts
    assert tpc.stats["missing"] == 0
    for c in ("math score", "reading score", "writing score"):
        assert (cols[c].type, cols[c].subtype) == (T.INTEGER, "discrete")


def test_student_scores_types() -> None:
    cols, tp = profile_example("student-scores.csv")
    assert cols["id"].type is T.IDENTIFIER and cols["id"].subtype == "sequential"
    assert tp.entity.id_columns == [0]
    for c in ("first_name", "last_name", "email"):
        assert cols[c].type not in (T.FACTOR, T.ORG_CODE, T.CONSTANT)  # never real values
    assert cols["first_name"].params.get("pii_hint") == "first_name"
    assert cols["last_name"].params.get("pii_hint") == "last_name"
    assert cols["gender"].type is T.FACTOR
    for c in ("part_time_job", "extracurricular_activities"):
        assert cols[c].type is T.BOOLEAN
        assert set(cols[c].model.nonmissing_counts) == {"True", "False"}
    for c in ("absence_days", "weekly_self_study_hours", "math_score", "history_score", "physics_score",
              "chemistry_score", "biology_score", "english_score", "geography_score"):
        assert (cols[c].type, cols[c].subtype) == (T.INTEGER, "discrete"), c
    assert cols["career_aspiration"].type is T.FACTOR
    assert cols["career_aspiration"].stats["distinct"] == 17


def test_students_data_types() -> None:
    cols, _ = profile_example("students_data.csv")
    assert cols["YearofStudy"].type is T.ORDINAL
    assert cols["Course"].type in (T.FACTOR, T.ORG_CODE)
    for c in ("Age", "ApproxHeight", "ApproxWeight"):
        assert (cols[c].type, cols[c].subtype) == (T.INTEGER, "discrete"), c
    assert cols["KCSE"].type is T.INTEGER_YEAR
    assert cols["Yr_JoinCampus"].type is T.INTEGER_YEAR
    sit = cols["SitKCSE"]
    assert sit.type is T.FACTOR
    assert "Central " in sit.model.nonmissing_counts and "Eastern " in sit.model.nonmissing_counts
    for c in ("Expense_Semester", "Expense_Accommodation"):
        assert (cols[c].type, cols[c].subtype) == (T.INTEGER, "continuous"), c
    assert cols["Previous_Exam_MeanGrade"].type is T.ORDINAL


# --------------------------------------------------------------------------- rule tests
def test_none_level_is_not_missing() -> None:
    d = detect(["none", "completed", "none"], name="test preparation course")
    assert d.type is T.FACTOR and d.policy() == "text"


def test_race_code_na_stays_a_level() -> None:
    d = detect(["W", "B", "NA", "H", "NA"], name="race")
    assert d.type is T.FACTOR and d.policy() == "text"


def test_na_tokens_missing_in_typed_columns() -> None:
    d = detect(["1", "2", "NA", "3", "N/A", ""], name="n_things")
    assert d.type is T.INTEGER and d.policy() == "typed"


def test_grade_level_vocabulary() -> None:
    d = detect(["PK", "KG", "01", "02", "03", "12", "05"], name="grade_level")
    assert d.type is T.ORDINAL and d.subtype == "grade_level"
    assert d.params["levels"][:3] == ["PK", "KG", "01"]


def test_numeric_grade_needs_hint() -> None:
    assert detect([str(i) for i in range(0, 11)], name="absence_days").type is T.INTEGER
    assert detect([str(i) for i in range(-2, 13)], name="grade").type is T.ORDINAL


def test_flag_single_token() -> None:
    assert detect(["Yes", "", "Yes", ""]).type is T.FLAG_SINGLE_TOKEN


def test_constant_and_empty() -> None:
    assert detect(["x", "x", "x"]).type is T.CONSTANT
    assert detect(["", "NA", " "]).type is T.EMPTY


@pytest.mark.parametrize("vals,ambiguous", [
    (["True", "False"], False), (["yes", "no", "Yes"], False), (["Y", "N"], True), (["1", "0"], True),
    (["on", "off"], False), (["t", "f"], False), (["-1", "0"], True),
])
def test_boolean_pairs(vals, ambiguous) -> None:
    d = detect(vals, name="flag")
    assert d.type is T.BOOLEAN and d.ambiguous is ambiguous


def test_code_count_hint_blocks_01_boolean() -> None:
    d = detect(["0", "1", "1", "0"], name="grade")
    assert d.type is not T.BOOLEAN


@pytest.mark.parametrize("vals", [
    ["2023-2024", "2024-2025"], ["2023-24", "2024-25"], ["23-24", "24-25"], ["SY2324", "SY2425"],
    ["SY 2023-24", "SY 2024-25"], ["2324", "2425"],
])
def test_school_year_forms(vals) -> None:
    assert detect(vals, name="sy_label").type is T.SCHOOL_YEAR


def test_school_year_ending_integer_with_hint() -> None:
    assert detect(["2023", "2024", "2025"], name="school_year").type is T.SCHOOL_YEAR
    assert detect(["2023", "2024", "2025"], name="KCSE").type is T.INTEGER_YEAR


def test_digit_codes() -> None:
    d = detect(["01", "02", "10", "02"], name="code_x")
    assert d.type is T.DIGIT_CODE and d.params["levels_mode"]
    d = detect([f"{i:05d}" for i in range(30)], name="zone")
    assert d.type is T.DIGIT_CODE and not d.params["levels_mode"]


def test_integer_discrete_continuous_and_floats() -> None:
    assert detect([str(i) for i in range(100)]).subtype == "discrete"
    sparse = [str(i * 1000 + 7) for i in range(80)]
    d = detect(sparse, name="amount")
    assert (d.type, d.subtype) == (T.INTEGER, "continuous")
    assert any(m.code == "possible_sparse_code" for m in d.messages)
    d = detect(["72.0", "81.0", "90.0"])
    assert d.type is T.INTEGER and d.params["float_text"] and d.params["decimals"] == 1
    d = detect(["3.50", "2.75", "4.00"])
    assert d.type is T.FLOAT and d.params["decimals"] == 2 and d.params["fixed_decimals"]


def test_factor_overflow_and_free_text() -> None:
    vals = [f"label {i}" for i in range(25)]
    d = detect(vals, name="thing")
    assert d.type is T.HIGH_CARD_CATEGORICAL
    assert any(m.code == "factor_overflow" and m.level == "warning" for m in d.messages)
    long = [f"this is a fairly long sentence number {i}" for i in range(25)]
    assert detect(long, name="thing").type is T.FREE_TEXT
    assert detect([f"l{i}" for i in range(25)], name="thing", factor_max=30).type is T.FACTOR


def test_pii_hints_win_over_factor_rule() -> None:
    d = detect(["Ann", "Bob", "Ann"], name="first_name")
    assert d.type is T.PERSON_NAME and d.params["pii_hint"] == "first_name"  # AREA C: fake names (D12)
    d = detect(["call home", "ok", "ok"], name="teacher_comments")
    assert d.type is T.FREE_TEXT
    d = detect(["Ann", "Bob", "Ann"], name="first_name", force_factor=["first_name"])
    assert d.type is T.FACTOR and d.source == "force-factor"
    assert detect(["yes", "no"], name="email_opt_in").type is T.BOOLEAN


def test_org_codes() -> None:
    d = detect(["577", "530", "577", "101"], name="school_id")
    assert d.type is T.ORG_CODE
    assert detect(["1", "2", "3"], name="student_school_id").type is not T.ORG_CODE
    assert detect(["none", "completed"], name="test preparation course").type is T.FACTOR


def test_overrides() -> None:
    vals = [str(i) for i in range(1, 30)]
    assert detect(vals, name="id").type is T.IDENTIFIER
    assert detect(vals, name="id", no_default_ids=True).type is T.INTEGER
    assert detect(vals, name="rownum", id_col=["rownum"]).type is T.IDENTIFIER
    assert detect(vals, name="id", org_code=["id"]).type is T.ORG_CODE  # explicit beats the default list
    assert detect(vals, name="x", force_type=["x=FACTOR"]).type is T.FACTOR
    assert detect(["a", "b"], name="x", force_text=["x"]).type in (T.HIGH_CARD_CATEGORICAL, T.FREE_TEXT)
    assert detect(vals, name="x", force_type=["f.csv::x=FLOAT"]).type is T.FLOAT
    assert detect(vals, name="x", force_type=["other.csv::x=FLOAT"]).type is T.INTEGER
    with pytest.raises(UnsupportedColumnError):
        detect(["a", "b"], name="x", force_type=["x=INTEGER"])
    assert detect(["2024-01-01"], name="x", force_type=["x=DATE"]).type is T.DATE  # area A
    with pytest.raises(UnsupportedColumnError):
        detect(["a", "b"], name="x", force_type=["x=DATE"])  # values are not dates


def test_header_hint_matching() -> None:
    assert hints.tokens("YearofStudy") == ("yearof", "study")
    assert hints.header_hints("SchoolID").org_code == "school"
    assert hints.header_hints("composite_score").org_code is None
    assert hints.header_hints("Yr_JoinCampus").org_code is None
    assert hints.header_hints("Student_First_Name").pii == "first_name"
    assert hints.header_hints("school_name").pii is None
    assert hints.header_hints("StudentNumber").default_id
    assert not hints.header_hints("student_number_2").default_id


def test_every_catalog_type_present() -> None:
    from datanonymous.model import TYPE_INFO, ColumnType

    assert len(ColumnType) == 40  # 36 catalog rows; INTEGER spans 9-10; 21, 29, 31, 33 split
    assert set(TYPE_INFO) == set(ColumnType)
    rows = {r for t in ColumnType for r in TYPE_INFO[t].catalog.split("-")}
    assert {str(i) for i in range(1, 37)} <= rows


# --------------------------------------------------------------------------- person IDs, org codes, person names (D10, D12)
PERSON_ID_VALUES = [str(5000 + i) for i in range(40)] * 2


@pytest.mark.parametrize("name", ["Family_ID", "Guardian_Number", "StudentLocalID", "staff_number", "familyid",
                                  "Teacher_No", "Student_ID_Number"])
def test_person_id_headers_are_synthesised_identifiers(name: str) -> None:
    d = detect(PERSON_ID_VALUES, name=name)
    assert d.type is T.IDENTIFIER and d.source == "person-id-hint", name


@pytest.mark.parametrize("name", ["student_notes", "parent_notification", "num_students", "student_count",
                                  "student_paid", "LASID"])
def test_person_words_without_an_id_word_are_not_ids(name: str) -> None:
    assert not hints.person_id_hint(name)


def test_person_id_header_overrides() -> None:
    assert detect(PERSON_ID_VALUES, name="Family_ID", org_code=["Family_ID"]).type is T.ORG_CODE
    assert detect(PERSON_ID_VALUES, name="Family_ID", force_factor=["Family_ID"]).type is T.FACTOR


@pytest.mark.parametrize("name,expected", [
    ("Course", "course"), ("SchoolID", "school"), ("school_name", "school"), ("section_number", "section"),
    ("ExitCodeDesc", "exitcode"), ("ELL_Code", "ellcode"), ("Course1", "course"), ("Section_Teacher", None),
    ("School_Counselor", None), ("Term_Advisor", None), ("school_year", None), ("terminated", None),
    ("midterm", None), ("test preparation course", None), ("Yr_JoinCampus", None),
])
def test_org_code_hint_needs_an_identifier_suffix(name: str, expected: str | None) -> None:
    assert hints.org_code_hint(name) == expected


@pytest.mark.parametrize("header", ["Section_Teacher", "School_Counselor", "Term_Advisor", "Teacher", "Counselor",
                                    "Mother_Name", "Advisor_Name", "EmergencyContact", "Student"])
def test_person_name_headers_never_keep_real_values(header: str) -> None:
    names = [f"Pat Example{i}" for i in range(5)] * 10
    d = detect(names, name=header)
    assert d.type not in (T.FACTOR, T.ORG_CODE, T.CONSTANT, T.DIGIT_CODE), header
    assert d.params.get("pii_hint") == "full_name", header


@pytest.mark.parametrize("name", ["SchoolName", "course_name", "test_name", "is_parent", "state_name", "filename"])
def test_non_person_name_headers(name: str) -> None:
    assert hints.header_hints(name).pii != "full_name"


def test_contact_number_is_a_phone_hint() -> None:
    assert hints.header_hints("Contact_Number").pii == "phone"


def test_rare_levels_await_the_d35_decision_without_labels() -> None:
    # D35 replaced the small-file warning: detection marks the levels for the copy gate's allowlist, which
    # profile.table decides with the entity counts (rare levels get pseudonyms; counts only in the message)
    from datanonymous.detect import text as dtext

    vals = ["Alpha"] * 20 + ["Beta"] * 20 + ["Zed Quill"] + ["Yarrow"] * 2
    d = detect(vals, name="program")
    assert d.type is T.FACTOR and d.params.get(dtext.PENDING_KEY)
    assert not [m for m in d.messages if m.code == "small_file_levels"]
    from datanonymous.model import RawColumn, RawTable

    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    p = profile_table(RawTable(None, ["program"], [RawColumn("program", arr)], len(vals)), "f.csv", "csv",
                      settings()).columns[0]
    assert set(dtext.pseudonymized(p)) == {"Zed Quill", "Yarrow"}
    msg = next(m for m in p.messages if m.code == "pseudonymized_levels")
    assert "2 of 4 level(s)" in msg.text and "Zed" not in msg.text and "Yarrow" not in msg.text
    forced = detect(vals, name="program", force_factor=["program"])
    assert not forced.params.get(dtext.PENDING_KEY) and dtext.copy_cleared(forced)  # explicit: kept as they are


def test_repeated_1_to_n_ids_are_not_row_numbers() -> None:
    d = detect([str(i) for i in range(1, 51)] * 2, name="student_id")
    assert d.type is T.IDENTIFIER and d.subtype == "digits"
    assert detect([str(i) for i in range(1, 51)], name="student_id").subtype == "sequential"


# --------------------------------------------------------------------------- profile contracts
def _arr(values):
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def test_native_details_reach_every_column_profile() -> None:
    import json

    from datanonymous.model import RawColumn, RawTable

    cols = [RawColumn("id", _arr(["1", "2", "3"]), native={"cell": "s"}),
            RawColumn("x", _arr(["1.5", "2.5", "3.5"]), native={"fmt": "0.00"}),
            RawColumn("drop", _arr(["a", "b", "c"]), native={"k": 1}),
            RawColumn("blank", _arr(["a", "b", "c"]), native={"k": 2})]
    tp = profile_table(RawTable(None, ["id", "x", "drop", "blank"], cols, 3), "f.csv", "csv",
                       settings(exclude=["drop"], null=["blank"]))
    assert [p.native for p in tp.columns] == [{"cell": "s"}, {"fmt": "0.00"}, {"k": 1}, {"k": 2}]
    assert "native" not in json.dumps(tp.to_public(), default=str)


def test_id_candidates_warned_and_redacted_even_when_an_id_matched() -> None:
    from datanonymous.model import RawColumn, RawTable

    n = 30
    # ('Badge_Number' would be an ID column: an ID word over keys that each belong to one student, K-02)
    cols = [RawColumn("StudentNumber", _arr([f"S{i:05d}" for i in range(1, n + 1)])),
            RawColumn("Badge", _arr([str(70000 + 3 * i) for i in range(1, n + 1)])),
            RawColumn("score", _arr([str(i % 7) for i in range(n)]))]
    tp = profile_table(RawTable(None, ["StudentNumber", "Badge", "score"], cols, n), "f.csv", "csv", settings())
    msgs = [m for m in tp.messages if m.code == "id_candidates"]
    assert tp.entity.id_columns == [0] and len(msgs) == 1 and "Badge" in msgs[0].text
    st = tp.columns[1].stats
    assert "redacted" in st and not {"min", "max", "pmf", "quantiles", "median", "levels"} & set(st)
    assert "70003" not in str(tp.to_public())


def test_unique_discrete_integers_have_no_pmf() -> None:
    from datanonymous.model import ColumnProfile
    from datanonymous.profile.stats import column_stats

    p = ColumnProfile(index=0, name="n", type=T.INTEGER, subtype="discrete")
    st = column_stats(_arr([str(i) for i in range(10)]), p, 10)
    assert "pmf" not in st and st["pmf_omitted"]
    st = column_stats(_arr(["1", "1", "2"]), p, 10)
    assert st["pmf"] == {"1": 2, "2": 1}


# --------------------------------------------------------------------------- area hooks wired into the foundation
def test_date_numbers_hook_feeds_model_and_ks(monkeypatch) -> None:
    from datanonymous.detect import dates as dates_hooks
    from datanonymous.model import ColumnProfile
    from datanonymous.profile.stats import numeric_view
    from datanonymous.profile.table import build_model
    from datanonymous.report import fidelity

    def ordinals(values, profile):
        return np.array([float(v.replace("-", "")) if v[0].isdigit() and "/" not in v else np.nan for v in values])

    monkeypatch.setattr(dates_hooks, "date_numbers", ordinals)
    p = ColumnProfile(index=0, name="d", type=T.DATE, missing_policy="typed", na_tokens=frozenset({"NA"}))
    model = build_model(_arr(["2024-01-03", "2024-01-01", "NA", "0/0/0", "2024-01-02"]), p)
    assert model.sorted_numeric.tolist() == [20240101.0, 20240102.0, 20240103.0]
    syn = numeric_view(p, ["2024-01-01", "2024-01-01", "2024-01-01"])
    res = fidelity.compare_column(p, {"n": 5, "missing_rate": 0.2}, {"n": 5, "missing_rate": 0.2},
                                  model.sorted_numeric, syn)
    ks = [c for c in res["checks"] if c["check"] == "ks"]
    assert ks and not ks[0]["ok"] and res["verdict"] == "drift"


def test_area_b_hooks_get_value_strings(monkeypatch) -> None:
    from datanonymous.detect import numeric as numeric_hooks
    from datanonymous.model import ColumnProfile
    from datanonymous.profile.stats import column_stats, numeric_view

    seen: dict[str, list[str]] = {}

    def stats_hook(values, profile):
        seen["stats"] = list(values)
        return {"min": -25.0}

    monkeypatch.setattr(numeric_hooks, "numeric_stats_extended", stats_hook)
    monkeypatch.setattr(numeric_hooks, "numbers_extended",
                        lambda values, profile: np.array([1.0 if v.startswith("$") else np.nan for v in values]))
    p = ColumnProfile(index=0, name="amt", type=T.NUMERIC_FORMATTED)
    st = column_stats(_arr(["$1,234.50", "($25.00)", "45%", ""]), p, 10)
    assert sorted(seen["stats"]) == sorted(["$1,234.50", "($25.00)", "45%"]) and st["min"] == -25.0
    assert numeric_view(p, ["$1", "x", "$2"], [2, 1, 1]).tolist() == [1.0, 1.0, 1.0]
