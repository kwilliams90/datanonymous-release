"""AREA G fidelity thresholds (section 10): max(1 pp, 2 SE) shares, KS 0.05, adjusted aspects, per-entity basis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from datanonymous.model import ColumnProfile, ColumnType
from datanonymous.report.fidelity import KS_LIMIT, compare_column, ks_statistic, share_limit


def _levels(shares: dict[str, float], n: int, missing_rate: float = 0.0) -> dict:
    return {"n": n, "missing": int(round(missing_rate * n)), "missing_rate": missing_rate,
            "level_shares": shares, "distinct": len(shares)}


def _factor(**flags: bool) -> ColumnProfile:
    p = ColumnProfile(index=0, name="lunch", type=ColumnType.FACTOR)
    p.flags.update(flags)
    return p


def test_share_limit_is_max_of_1pp_and_2se() -> None:
    assert share_limit(0.5, 100) == pytest.approx(2 * math.sqrt(0.25 / 100))  # 10 pp
    assert share_limit(0.01, 10_000) == 0.01  # 2 SE = 0.2 pp < 1 pp
    assert share_limit(0.0, 0) == 0.01


def test_discrete_share_deviation_drifts_beyond_the_limit() -> None:
    real = _levels({"a": 0.5, "b": 0.5}, 10_000)
    ok = compare_column(_factor(), real, _levels({"a": 0.505, "b": 0.495}, 10_000))
    assert ok["verdict"] == "ok" and "within" in ok["reason"]
    bad = compare_column(_factor(), real, _levels({"a": 0.53, "b": 0.47}, 10_000))
    assert bad["verdict"] == "drift" and "deviates 3.00 pp" in bad["reason"]
    lvl = next(c for c in bad["checks"] if c["check"] == "level_shares")
    assert lvl["limit"] == 0.01 and lvl["worst_level"] in ("a", "b") and not lvl["ok"]
    # small n: 2 SE dominates (0.5 at n=100 -> 10 pp), so 5 pp is fine
    assert compare_column(_factor(), _levels({"a": 0.5, "b": 0.5}, 100),
                          _levels({"a": 0.55, "b": 0.45}, 100))["verdict"] == "ok"


def test_rare_category_adjusted_never_drifts_on_shares_or_missing() -> None:
    real = _levels({"a": 0.9, "b": 0.1}, 10_000, missing_rate=0.0)
    syn = _levels({"a": 0.8, "b": 0.2}, 10_000, missing_rate=0.02)
    assert compare_column(_factor(), real, syn)["verdict"] == "drift"
    adj = compare_column(_factor(rare_category_adjusted=True), real, syn)
    assert adj["verdict"] == "adjusted" and "rare_category_adjusted" in adj["reason"]
    assert all(c["ok"] for c in adj["checks"])


def test_missing_rate_is_checked_like_a_share() -> None:
    real = _levels({"a": 1.0}, 10_000, missing_rate=0.10)
    assert compare_column(_factor(), real, _levels({"a": 1.0}, 10_000, 0.105))["verdict"] == "ok"
    res = compare_column(_factor(), real, _levels({"a": 1.0}, 10_000, 0.13))
    assert res["verdict"] == "drift" and "missing" in res["reason"]


def test_continuous_ks_threshold() -> None:
    rng = np.random.default_rng(0)
    real = np.sort(rng.normal(size=5000))
    p = ColumnProfile(index=0, name="x", type=ColumnType.FLOAT)
    st = {"n": 5000, "missing": 0, "missing_rate": 0.0}
    same = compare_column(p, st, st, real, rng.permutation(real))
    assert same["verdict"] == "ok" and "KS 0.0000" in same["reason"]
    shifted = compare_column(p, st, st, real, real + 0.5)
    assert shifted["verdict"] == "drift" and ks_statistic(real, real + 0.5) > KS_LIMIT


def test_extreme_value_adjusted_widens_ks_and_shares_by_the_forced_values() -> None:
    real = np.arange(100, dtype=float)
    syn = real.copy()
    syn[:6] = 50.5  # KS 0.06: above 0.05 but within 0.05 + 2/n (the two forced extreme values)
    ks = ks_statistic(real, syn)
    assert KS_LIMIT < ks <= KS_LIMIT + 2 / len(syn)
    st = {"n": 100, "missing": 0, "missing_rate": 0.0}
    p = ColumnProfile(index=0, name="x", type=ColumnType.FLOAT)
    assert compare_column(p, st, st, real, syn)["verdict"] == "drift"
    p.flags["extreme_value_adjusted"] = True
    res = compare_column(p, st, st, real, syn)
    assert res["verdict"] == "adjusted" and "extreme_value_adjusted" in res["reason"]
    ks_check = next(c for c in res["checks"] if c["check"] == "ks")
    assert ks_check["limit"] == KS_LIMIT and ks_check["limit_with_adjustment"] > KS_LIMIT


def test_discrete_integer_pmf_with_forced_extremes() -> None:
    p = ColumnProfile(index=0, name="age", type=ColumnType.INTEGER, subtype="discrete")
    real = {"n": 62, "missing": 0, "missing_rate": 0.0, "pmf": {"18": 30, "22": 30, "34": 2}}
    syn = {"n": 62, "missing": 0, "missing_rate": 0.0, "pmf": {"18": 31, "22": 31}}
    assert compare_column(p, real, syn)["verdict"] in ("ok", "drift")
    p.flags["extreme_value_adjusted"] = True
    assert compare_column(p, real, syn)["verdict"] in ("ok", "adjusted")


def test_entity_basis_and_not_applicable_roles() -> None:
    real = _levels({"F": 0.5, "M": 0.5}, 200)
    res = compare_column(_factor(), real, real, basis="entities")
    assert res["basis"] == "entities" and res["reason"].startswith("per entity")
    p = _factor()
    p.role = "null"
    assert compare_column(p, real, real)["verdict"] == "n/a"
    p.role = "excluded"
    assert compare_column(p, real, real)["verdict"] == "n/a"


def test_never_copy_columns_are_not_share_compared() -> None:
    p = ColumnProfile(index=0, name="first_name", type=ColumnType.HIGH_CARD_CATEGORICAL)
    real = {"n": 100, "missing": 0, "missing_rate": 0.0, "distinct": 40}
    res = compare_column(p, real, {"n": 100, "missing": 0, "missing_rate": 0.0, "distinct": 40})
    assert res["verdict"] == "ok"
    distinct = next(c for c in res["checks"] if c["check"] == "distinct")
    assert "invented" in distinct["reason"]


def test_pipeline_warns_only_on_drift(example_run) -> None:
    import json

    data = json.loads((example_run.rep / "report.json").read_text(encoding="utf-8"))
    drift_warnings = [w for w in data["warnings"] if w["code"] == "fidelity_drift"]
    verdicts = []
    for name in ("StudentsPerformance.csv", "student-scores.csv", "students_data.csv"):
        pj = json.loads((example_run.rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
        verdicts += [c["fidelity"]["verdict"] for c in pj["tables"][0]["columns"] if c["fidelity"]]
    assert len(drift_warnings) == verdicts.count("drift")
    assert set(verdicts) <= {"ok", "adjusted", "drift", "n/a"}
