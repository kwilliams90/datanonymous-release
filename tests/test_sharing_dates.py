"""Sharing D43: extrema are excluded after rendering with month/type fidelity."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from datanonymous.detect import dates as D
from datanonymous.config import Settings
from datanonymous.errors import GenerationError
from datanonymous.generate import dates as ordinary, sharing_dates
from datanonymous.model import ColumnModel, ColumnProfile, ColumnType, EntityProfile, RawColumn, RawTable, TableProfile
from datanonymous.profile import sharing

from test_area_a_support import detect


def profile(values, name="DOB", native=None, **options):
    detection = detect(values, name=name, native=native, **options)
    kind = detection.type if detection.type in D.DATE_TYPES else ColumnType.DATE
    p = ColumnProfile(0, name, kind, params=deepcopy(detection.params), native=native)
    # These rendering fixtures have ten invented, distinct people per supplied
    # date. Separate integration tests exercise unsupported people counts.
    proof_values = list(values) * 10
    p.model = ColumnModel(nonmissing_counts=dict(Counter(proof_values)), real_values=frozenset(values))
    p.params["_sharing_person_date"] = True
    ids = np.array([f"S{i:06d}" for i in range(len(proof_values))], dtype=object)
    identity = ColumnProfile(1, "Student_ID", ColumnType.IDENTIFIER, role="id", source="default-id")
    tp = TableProfile(None, [name, "Student_ID"], len(proof_values), [p, identity])
    tp.entity = EntityProfile(id_columns=[1], primary=1, n_entities=len(proof_values))
    tp.private.id_raw[1] = ids
    raw = RawTable(None, tp.header, [RawColumn(name, np.array(proof_values, dtype=object)), RawColumn("Student_ID", ids)], len(proof_values))
    settings = Settings()
    sharing.record_evidence(tp, raw, settings)
    return sharing.protect_profile(tp, settings).columns[0]


def generated(p, n=120, seed=7):
    return sharing_dates.generate_dates(p, p.model, n, np.random.default_rng(seed))[0]


def test_singleton_month_uses_public_days_and_preserves_ambiguous_endpoint_cover():
    p = profile(["2004-02-29"] * 20)
    before = deepcopy(p.params)
    out = generated(p)
    assert all(v.startswith("2004-02-") and v != "2004-02-29" for v in out)
    assert len(set(out)) > 1
    assert len({int(v[-2:]) for v in out}) <= 29 - sharing_dates.MIN_BUCKET_DAYS
    assert p.params == before
    assert out.tolist() == generated(p).tolist()


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"])
def test_real_normalized_extremes_never_survive_rendering(fmt):
    import datetime as dt

    values = [(dt.date(2004, 2, 1) + dt.timedelta(days=i)).strftime(fmt) for i in range(29)] * 2
    values += [(dt.date(2004, 3, 1) + dt.timedelta(days=i)).strftime(fmt) for i in range(31)]
    p = profile(values, date_format=[f"DOB={fmt}"])
    out = generated(p, len(values))
    assert not (set(sharing_dates.normalized_keys(p, out)) & sharing_dates.source_extremes(p))
    parsed = D.view(p.params, out.tolist())
    assert Counter(parsed.mon.tolist()) == Counter(D.view(p.params, values).mon.tolist())
    assert np.all(parsed.cls >= 0)


def test_person_month_precision_and_infeasible_rigid_calendar_refuse():
    month = profile(["2001-03"] * 20)
    with pytest.raises(GenerationError, match="month-preserving"):
        generated(month)
    p = profile(["2004-02-29"] * 20)
    p.params.update({"calendar": True, "calendar_reason": "--calendar"})
    with pytest.raises(GenerationError, match="month-preserving"):
        generated(p)


def test_named_timezone_native_datetime_stays_typed_and_avoids_exact_instants():
    values = [f"2024-03-{day:02d} 10:15:00-05:00" for day in range(12, 30)]
    native = {"temporal": {"kind": "datetime", "tz": "America/Chicago", "unit": "us"}}
    p = profile(values, name="Student_Created", native=native)
    out = generated(p)
    assert not (set(sharing_dates.normalized_keys(p, out)) & sharing_dates.source_extremes(p))
    typed = ordinary.to_native(p, out)
    assert all(v is not None and v.month == 3 and str(v.tzinfo) == "America/Chicago" for v in typed)
    assert all(v.hour == 10 and v.minute == 15 for v in typed)


def test_mixed_datetime_precision_is_checked_after_rendering():
    values = ["2004-02-03 10:15", "2004-02-03 10:15:30", "2004-02-27 11:30", "2004-02-27 11:30:59"] * 10
    p = profile(values, name="Student_Date")
    out = generated(p)
    assert not (set(sharing_dates.normalized_keys(p, out)) & sharing_dates.source_extremes(p))
    assert np.all(D.view(p.params, out.tolist()).cls >= 0)


def test_normalized_verification_matches_an_extreme_across_timezone_spellings():
    p = profile(["2004-02-03 10:15:00+00:00", "2004-02-27 10:15:00-05:00"] * 20)
    lo = sharing_dates.normalized_keys(p, ["2004-02-03 05:15:00-05:00"])[0]
    assert lo in sharing_dates.source_extremes(p)


@pytest.mark.parametrize("year,month,last", [(2023, 3, 31), (2025, 4, 30), (2024, 2, 29)])
def test_saturated_calendar_never_exposes_exactly_the_original_endpoint_holes(year, month, last):
    values = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(2, 28)]
    p = profile(values)
    out = generated(p, n=5000)
    days = {int(v[-2:]) for v in out}
    holes = set(range(1, last + 1)) - days
    assert {2, 27} <= holes
    assert len(holes) >= 2 * sharing_dates.MIN_BUCKET_DAYS
    assert len(days) >= sharing_dates.MIN_BUCKET_DAYS
    assert not (set(sharing_dates.normalized_keys(p, out)) & sharing_dates.source_extremes(p))
    assert out.tolist() == generated(p, n=5000).tolist()


@pytest.mark.parametrize("datetime", [False, True])
@pytest.mark.parametrize("length", [28, 29, 30, 31])
def test_calendar_bucket_support_has_an_endpoint_pair_ambiguity_bound(datetime, length):
    from itertools import combinations

    p = profile([f"2023-03-{day:02d}" for day in range(2, 28)])
    days = np.arange(length, dtype=np.int64)
    equivalent = Counter()
    for pair in combinations(days, 2):
        try:
            pool = sharing_dates._covered_pool(p, days, set(pair), datetime=datetime)
        except GenerationError:
            continue  # Some datetime bucket pairs exhaust the valid cover.
        equivalent[tuple(pool)] += 1
    assert min(equivalent.values()) >= sharing_dates.MIN_ENDPOINT_PAIRS
    buckets = sharing_dates._calendar_buckets(days)
    assert all(len(bucket) >= sharing_dates.MIN_BUCKET_DAYS for bucket in buckets)
    assert np.array_equal(np.concatenate(buckets), days)


def test_inferred_weekdays_and_date_grids_do_not_constrain_the_public_cover():
    import datetime as dt

    values = [f"2023-03-{day:02d}" for day in range(1, 32) if dt.date(2023, 3, day).weekday() < 5]
    p = profile(values)
    out = generated(p, n=2000)
    assert any(dt.date.fromisoformat(v).weekday() >= 5 for v in out)
    before = set(out)
    p.params["day_grid"] = [7, int(D.view(p.params, [values[0]]).day[0]) % 7]
    assert set(generated(p, n=2000)) == before


def test_short_source_weekday_range_uses_a_source_independent_calendar_cover():
    # March 4-7, 2023 spans Sat/Sun/Mon/Tue. Those four weekdays each occur
    # four times in March. Retaining that weekday set could make the exact
    # 4/7 pair inferable from a four-day excluded bucket, so sharing drops it.
    p = profile([f"2023-03-{day:02d}" for day in range(4, 8)] * 10)
    out = generated(p, n=2000)
    assert len(set(D.weekday(D.view(p.params, out.tolist()).day))) == 7


def test_source_membership_calendar_and_cross_day_endpoint_aliases_refuse():
    p = profile([f"2023-03-{day:02d}" for day in range(2, 28)])
    p.params.update({"calendar": True, "calendar_reason": "--calendar"})
    with pytest.raises(GenerationError, match="month-preserving date cover"):
        generated(p)
    p = profile([f"2024-03-{day:02d} 00:15:00+05:00" for day in range(2, 28)])
    with pytest.raises(GenerationError, match="month-preserving date cover"):
        generated(p)


def test_final_cover_verification_rejects_restored_decoys_and_preserves_native_dates():
    p = profile([f"2023-03-{day:02d}" for day in range(2, 28)])
    out = generated(p, n=2000)
    assert sharing_dates.cover_violations(p, out) == 0
    typed = ordinary.to_native(p, out)
    assert sharing_dates.cover_violations(p, typed) == 0
    blocked = ["2023-03-01", "2023-03-02", "2023-03-28"]
    assert sharing_dates.cover_violations(p, blocked) == 3
    assert sharing_dates.cover_violations(p, ordinary.to_native(p, blocked)) == 3


def test_generator_and_final_cover_fail_closed_if_person_bucket_proof_is_missing():
    p = profile(["2008-01-01", "2008-01-15"])
    p.model.extra.pop(sharing_dates.BUCKET_HOLDERS_KEY)
    with pytest.raises(GenerationError, match="excluded endpoint calendar bucket"):
        generated(p)
    assert sharing_dates.cover_violations(p, ["2008-01-20"]) == 1
