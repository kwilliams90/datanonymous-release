"""Round-3 review fixes, rules-fidelity cluster (R3-06, R3-07).

* R3-06: a declared rule's high column B used to be rebuilt as ``max(lows) + an observed gap``, which wrote
  values outside B's source range (a 0-100 score reached 175, and 6 below its minimum of 7), values a discrete
  column never had, dates on weekdays or calendar days B never has, and drifted B's distribution. Now B keeps
  its own independently generated values (range, exact min/max, observed levels and dates, layouts) and every
  rule on it decides only where they go: a monotone deal of B's values to targets ``bound + observed gap``
  under the rule bounds (optimal: every unit is served whenever any arrangement can), whole-number gaps
  matched to their observed counts, rows equal to real ones swapped off (privacy cannot split a rule's
  columns). A low value above all of B is the only case written past B's maximum, with a warning.
* R3-07: a within-ID-constant number or date (D4) is drawn per entity; expanded to unequal ID blocks its
  written rows drifted (KS 0.67) while the fidelity verdict, judged per entity, said PASS. Now the values are
  dealt to the ID blocks by value bins (>= --min-cell IDs each, cut where the block sizes change) so their rows
  match the source, and the verdict judges the written rows too: row-level KS against max(0.05, 2 SE over the
  effective number of IDs) plus the largest block's share of rows (the analogue of the level shares'
  max(1 pp, 2 SE): rows come in ID blocks, and one value covers a whole block).
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.generate import structure as gs
from datanonymous.profile import structure as st
from datanonymous.report import fidelity
from test_r2_structure_gen import _codes, _report, _write


def _run(tmp_path: Path, case: str, header: list[str], rows: list[list[object]], rules: list | None = None,
         seed: int = 1, *extra: str):
    d = tmp_path / case
    _write(d / "in" / "data.csv", header, rows)
    args = [d / "in", "--out-dir", d / "out", "--report-dir", d / "rep", "--seed", str(seed), "--force", *extra]
    if rules is not None:
        cfg = d / "rules.yaml"
        cfg.write_text("rules:\n" + "".join(f"  - {json.dumps(r)}\n" for r in rules), encoding="utf-8")
        args += ["--config", cfg]
    res = run_cli(*args)
    out = d / "out" / "data.csv"
    return res, (read_dicts(out) if out.exists() else []), d / "rep"


def _column(rep: Path, name: str) -> dict:
    for c in _report(rep)["tables"][0]["columns"]:
        if c["name"] == name:
            return c
    raise AssertionError(name)


def _holds(rows: list[dict[str, str]], lo: str, hi: str, strict: bool, parse) -> bool:
    return all((parse(r[hi]) > parse(r[lo])) if strict else (parse(r[hi]) >= parse(r[lo]))
               for r in rows if r[lo].strip() != "" and r[hi].strip() != "")


def _ks(a, b) -> float:
    return fidelity.ks_statistic(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


# =========================================================================== R3-06: rules stay in range
def _bounded_rows(seed: int = 42, n: int = 300) -> list[list[object]]:
    """Codex's fixture: baseline <= followup, followup in 7..100."""
    r = random.Random(seed)
    rows = []
    for i in range(n):
        a = r.randrange(0, 91)
        rows.append([f"S{i:06}", a, r.randrange(a, 101)])
    return rows


@pytest.mark.parametrize("seed", [123, 1, 2])
def test_codex_bounded_rule_keeps_the_high_column_in_range(tmp_path: Path, seed: int) -> None:
    rows = _bounded_rows()
    res, out, rep = _run(tmp_path, f"b{seed}", ["student_id", "baseline", "followup"], rows,
                         ["baseline <= followup"], seed)
    assert res.exit_code == 0, res.output
    real = [r[2] for r in rows]
    syn = [int(r["followup"]) for r in out]
    assert min(syn) == min(real) == 7 and max(syn) == max(real) == 100  # in range, exact extremes (D14)
    assert Counter(syn) == Counter(real)  # a discrete column: its exact quota, observed values only
    assert all(int(r["baseline"]) <= int(r["followup"]) for r in out)
    assert Counter(int(r["baseline"]) for r in out) == Counter(r[1] for r in rows)  # the low side untouched
    assert _column(rep, "followup")["fidelity"]["verdict"] == "ok"
    assert [d["status"] for d in _report(rep)["tables"][0]["structure"]["declared_rules"]] == ["applied"]
    codes = _codes(rep)
    assert "rule_out_of_range" not in codes and "rule_violated" not in codes
    # the gaps still look like the source's (rows swapped off real rows take a random fitting partner)
    real_gaps = np.asarray([r[2] - r[1] for r in rows])
    syn_gaps = np.asarray([int(r["followup"]) - int(r["baseline"]) for r in out])
    assert _ks(real_gaps, syn_gaps) <= 0.25, _ks(real_gaps, syn_gaps)
    # the deal swaps off rows equal to real ones itself (privacy moves the rule's columns together)
    assert "row_matches_remaining" not in codes


def _school_days(start: dt.date, n: int) -> list[dt.date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _iso(s: str) -> dt.date:
    return dt.date.fromisoformat(s.strip())


def _hm(s: str) -> int:
    h, m = s.strip().split(":")
    return int(h) * 60 + int(m)


def _rule_case(kind: str, rng: random.Random):
    """(header, rows, rules, low columns, high column, strict, parse, observed values only)."""
    if kind == "continuous_int":
        rows = []
        for i in range(400):
            a = rng.randrange(0, 50_000)
            rows.append([f"S{i:05}", a, min(a + int(rng.expovariate(1 / 6000)), 61_000)])
        return ["student_id", "fee_due", "fee_paid"], rows, ["fee_due <= fee_paid"], ["fee_due"], "fee_paid", \
            False, int, False
    if kind == "float":
        rows = []
        for i in range(300):
            a = round(rng.uniform(0, 8), 2)
            rows.append([f"S{i:05}", f"{a:.2f}", f"{min(a + rng.uniform(0, 3), 10.0):.2f}"])
        return ["student_id", "hours_planned", "hours_logged"], rows, ["hours_planned <= hours_logged"], \
            ["hours_planned"], "hours_logged", False, float, False
    if kind == "strict":
        rows = []
        for i in range(300):
            a = rng.randrange(0, 90)
            rows.append([f"S{i:05}", a, rng.randrange(a + 1, 101)])
        return ["student_id", "pre", "post"], rows, ["pre < post"], ["pre"], "post", True, int, True
    if kind == "multi":
        rows = []
        for i in range(300):
            a, b = rng.randrange(0, 80), rng.randrange(0, 80)
            rows.append([f"S{i:05}", a, b, rng.randrange(max(a, b), 101)])
        return ["student_id", "q1", "q2", "final"], rows, ["q1 <= final", "final >= q2"], ["q1", "q2"], \
            "final", False, int, True
    if kind == "weekday_dates":
        days = _school_days(dt.date(2023, 8, 21), 180)
        rows = []
        for i in range(250):
            j = rng.randrange(0, 170)
            rows.append([f"S{i:05}", days[j].isoformat(), days[min(j + rng.randrange(0, 10), 179)].isoformat()])
        return ["student_id", "referral_date", "action_date"], rows, ["referral_date <= action_date"], \
            ["referral_date"], "action_date", False, _iso, False
    if kind == "calendar_dates":
        tests = _school_days(dt.date(2024, 1, 8), 60)[::5]  # 12 test days, 20 rows each
        rows = []
        for i in range(240):
            b = tests[i % 12]
            rows.append([f"S{i:05}", (b - dt.timedelta(days=rng.randrange(0, 30))).isoformat(), b.isoformat()])
        return ["student_id", "registered", "test_date"], rows, ["registered <= test_date"], ["registered"], \
            "test_date", False, _iso, True
    if kind == "time_of_day":
        rows = []
        for i in range(200):
            a = rng.randrange(7 * 60, 14 * 60)
            b = min(a + rng.choice([45, 50, 90]), 15 * 60 + 30)
            rows.append([f"S{i:05}", f"{a // 60:02d}:{a % 60:02d}", f"{b // 60:02d}:{b % 60:02d}"])
        return ["student_id", "period_start", "period_end"], rows, ["period_start < period_end"], \
            ["period_start"], "period_end", True, _hm, False
    if kind == "per_entity":
        rows = []
        for s in range(120):
            a = rng.randrange(2014, 2021)
            b = min(a + rng.choice([0, 1, 1, 2, 5]), 2022)
            for _ in range(rng.randrange(1, 6)):
                rows.append([f"S{s:05}", a, b, rng.randrange(0, 100)])
        return ["student_id", "KCSE", "Yr_JoinCampus", "score"], rows, ["KCSE <= Yr_JoinCampus"], ["KCSE"], \
            "Yr_JoinCampus", False, int, True
    raise AssertionError(kind)


RULE_KINDS = ["continuous_int", "float", "strict", "multi", "weekday_dates", "calendar_dates", "time_of_day",
              "per_entity"]


@pytest.mark.parametrize("kind", RULE_KINDS)
def test_rule_variants_keep_the_high_columns_own_values(tmp_path: Path, kind: str) -> None:
    header, rows, rules, lows, hi, strict, parse, observed_only = _rule_case(kind, random.Random(7))
    h = header.index(hi)
    real = [parse(str(r[h])) for r in rows]
    for seed in (3, 4):
        res, out, rep = _run(tmp_path, f"{kind}{seed}", header, rows, rules, seed)
        assert res.exit_code == 0, res.output
        syn = [parse(r[hi]) for r in out]
        assert min(syn) == min(real) and max(syn) == max(real), (kind, seed)  # range and exact extremes
        if observed_only:
            assert set(syn) <= set(real), (kind, seed)  # levels / calendar dates B has
        for lo in lows:
            assert _holds(out, lo, hi, strict, parse), (kind, lo, seed)
        if kind == "weekday_dates":
            assert all(d.weekday() < 5 for d in syn)  # never on a weekday B never has
        if kind == "float":
            assert all(len(r[hi].split(".")[1]) == 2 for r in out)
        if kind == "per_entity":
            per: dict[str, set] = {}
            for r in out:
                per.setdefault(r["student_id"], set()).add(r[hi])
            assert all(len(v) == 1 for v in per.values())  # still one value per ID (D4)
        statuses = [d["status"] for d in _report(rep)["tables"][0]["structure"]["declared_rules"]]
        assert statuses == ["applied"] * len(rules), statuses
        assert "rule_out_of_range" not in _codes(rep)
        # the rule only decides where B's values go: the same values as without the rule (same seed; one per
        # ID for a within-ID constant)
        if "rule_values_shifted" not in _codes(rep):
            _, free, _ = _run(tmp_path, f"{kind}{seed}free", header, rows, None, seed)
            if kind == "per_entity":
                vals = lambda rs: Counter({r["student_id"]: r[hi] for r in rs}.values())  # noqa: E731
            else:
                vals = lambda rs: Counter(r[hi] for r in rs)  # noqa: E731
            assert vals(out) == vals(free), (kind, seed)


def test_whole_number_gaps_keep_their_observed_counts(tmp_path: Path) -> None:
    """Yr_JoinCampus - KCSE in {0, 1, 2, 5}: the deal and its gap repair reproduce the observed gaps."""
    rng = random.Random(11)
    rows = []
    for _ in range(150):
        a = rng.randrange(2014, 2021)
        rows.append([a, a + rng.choice([0, 1, 1, 2, 5]), rng.randrange(0, 100)])
    real = Counter(r[1] - r[0] for r in rows)
    for seed in (1, 2, 3):
        res, out, _ = _run(tmp_path, f"g{seed}", ["KCSE", "Yr_JoinCampus", "score"], rows,
                           ["KCSE <= Yr_JoinCampus"], seed)
        assert res.exit_code == 0, res.output
        syn = Counter(int(r["Yr_JoinCampus"]) - int(r["KCSE"]) for r in out)
        off = sum((syn - real).values())
        assert off <= 6, (seed, syn, real)  # at most a few rows off their observed gap counts
        assert Counter(r["Yr_JoinCampus"] for r in out) == Counter(str(r[1]) for r in rows)


def test_a_short_high_column_repeats_its_smallest_fitting_value(tmp_path: Path) -> None:
    """B is 50 in most rows; A's values 51-55 occur only where B is blank in the source. Synthetic blanks are
    independent of A, so some rows get A > 50 with too few B values above 50: they repeat B's smallest
    fitting value (in range, info message), never extrapolate."""
    rng = random.Random(5)
    rows = []
    for i in range(300):
        if i < 90:
            rows.append([f"S{i:05}", rng.randrange(51, 56), ""])
        else:
            b = 50 if rng.random() < 0.93 else rng.randrange(51, 61)
            rows.append([f"S{i:05}", rng.randrange(0, 51), b])
    seen = False
    for seed in (1, 2, 3, 4):
        res, out, rep = _run(tmp_path, f"s{seed}", ["student_id", "a", "b"], rows, ["a <= b"], seed)
        assert res.exit_code == 0, res.output
        syn = [int(r["b"]) for r in out if r["b"] != ""]
        real = [r[2] for r in rows if r[2] != ""]
        assert min(syn) == min(real) and max(syn) == max(real)
        assert set(syn) <= set(real)
        assert _holds(out, "a", "b", False, int)
        assert sum(r["b"] == "" for r in out) == 90  # blanks stay where the patterns put them
        assert "rule_out_of_range" not in _codes(rep)
        seen |= "rule_values_shifted" in _codes(rep)
    assert seen


def test_a_low_value_above_all_of_the_high_column_is_warned(tmp_path: Path) -> None:
    """A's values 90-99 occur only where B is blank; B's maximum is 60. A synthetic row with A in the 90s and B
    filled can meet the rule only above B's range: written there with a warning, the rule holds, exit 0."""
    rng = random.Random(6)
    rows = []
    for i in range(200):
        if i < 50:
            rows.append([f"S{i:05}", rng.randrange(90, 100), ""])
        else:
            a = rng.randrange(0, 51)
            rows.append([f"S{i:05}", a, a + rng.randrange(0, 10)])
    warned = False
    for seed in (1, 2, 3):
        res, out, rep = _run(tmp_path, f"o{seed}", ["student_id", "a", "b"], rows, ["a <= b"], seed)
        assert res.exit_code == 0, res.output
        assert _holds(out, "a", "b", False, int)
        over = [r for r in out if r["b"] != "" and int(r["b"]) > 59]
        assert all(int(r["b"]) == int(r["a"]) for r in over)  # the smallest value that meets the rule
        if over:
            warned = True
            assert "rule_out_of_range" in _codes(rep)
    assert warned


def test_an_extreme_only_a_broken_rule_admits_is_warned(tmp_path: Path) -> None:
    """The rule holds in most real rows only; B's exact min sits on a row that breaks it. Applied to every
    synthetic row, the rule leaves that min nowhere to go: the file says so (D14), never silently."""
    rng = random.Random(8)
    rows = []
    for i in range(300):
        a = rng.randrange(150, 260)
        rows.append([f"S{i:05}", a, min(a + rng.randrange(0, 60), 300)])
    rows[0] = ["S99999", 160, 140]  # the min of B, below every low value
    res, out, rep = _run(tmp_path, "brk", ["student_id", "boy", "eoy"], rows, ["boy <= eoy"], 1)
    assert res.exit_code == 0, res.output
    assert _holds(out, "boy", "eoy", False, int)
    syn = [int(r["eoy"]) for r in out]
    assert min(syn) > 140 and max(syn) == 300
    codes = _codes(rep)
    assert "rule_partly_true" in codes and "rule_extreme_dropped" in codes


def test_rule_runs_are_deterministic(tmp_path: Path) -> None:
    rows = _bounded_rows()
    outs = []
    for k in range(2):
        _run(tmp_path, f"det{k}", ["student_id", "baseline", "followup"], rows, ["baseline <= followup"], 9)
        outs.append((tmp_path / f"det{k}" / "out" / "data.csv").read_bytes())
    assert outs[0] == outs[1]


def _min_unserved(lo: list[int], k: int) -> int:
    """Hall's condition for nested eligibility (unit u may take positions lo[u]..k-1)."""
    return max(max(0, sum(x >= j for x in lo) - (k - j)) for j in range(k + 1))


def test_deal_serves_every_unit_whenever_possible() -> None:
    rng = np.random.default_rng(0)
    for _ in range(400):
        k = int(rng.integers(1, 9))
        lo = rng.integers(0, k + 1, size=k)
        target = rng.normal(size=k)
        take = gs._deal_rule_values(lo.astype(np.int64), target, rng.permutation(k))
        served = take[take >= 0]
        assert len(set(served.tolist())) == len(served)  # each value once
        assert np.all(take[take >= 0] >= lo[take >= 0])  # every rule bound met
        assert int((take < 0).sum()) == _min_unserved(lo.tolist(), k)  # the fewest units left over


def test_deal_is_a_monotone_matching_without_bounds() -> None:
    rng = np.random.default_rng(1)
    target = rng.normal(size=50)
    take = gs._deal_rule_values(np.zeros(50, dtype=np.int64), target, rng.permutation(50))
    assert np.array_equal(np.argsort(take), np.argsort(target, kind="stable"))


# =========================================================================== R3-07: written rows of D4 numbers
def _codex_panel() -> list[list[object]]:
    """Codex's fixture: 100 IDs with a constant expense; 90 hold one row, 10 hold 50 rows."""
    rows = []
    for i in range(100):
        for j in range(1 if i < 90 else 50):
            rows.append([f"S{i:06}", 1000 + i * 100, j + 1])
    return rows


def _row_numbers(rows, col, parse) -> list[float]:
    return [parse(r[col]) for r in rows if r[col].strip() != ""]


def _check_entity_rows(real_rows: list[dict], out: list[dict], rep: Path, col: str, parse) -> float:
    """Written-row KS within the block tolerance, constant within each ID, verdict not drift; returns KS."""
    sizes = Counter(r["student_id"] for r in real_rows)
    share = max(sizes.values()) / len(real_rows)
    ks = _ks(_row_numbers(real_rows, col, parse), _row_numbers(out, col, parse))
    assert ks <= fidelity.KS_LIMIT + share, (col, ks, share)
    per: dict[str, set] = {}
    for r in out:
        per.setdefault(r["student_id"], set()).add(r[col])
    assert all(len(v) == 1 for v in per.values())  # D4
    fid = _column(rep, col)["fidelity"]
    assert fid["verdict"] != "drift", fid["reason"]
    assert any(c["check"] == "ks" for c in fid["rows"]["checks"])  # the written rows are judged
    assert "per row" in fid["reason"]
    return ks


@pytest.mark.parametrize("seed", [123, 1, 2])
def test_codex_entity_continuous_rows_match_the_source(tmp_path: Path, seed: int) -> None:
    rows = _codex_panel()
    res, out, rep = _run(tmp_path, f"e{seed}", ["student_id", "expense", "visit"], rows, None, seed)
    assert res.exit_code == 0, res.output
    real = read_dicts(tmp_path / f"e{seed}" / "in" / "data.csv")
    assert _column(rep, "expense")["role"] == "entity"
    ks = _check_entity_rows(real, out, rep, "expense", float)
    assert ks < 0.1  # was 0.6661
    rm = np.mean([float(r["expense"]) for r in real])
    sm = np.mean([float(r["expense"]) for r in out])
    assert abs(sm - rm) / rm < 0.02, (rm, sm)  # was 9687 -> 6645
    below = lambda rs: np.mean([float(r["expense"]) < 10_000 for r in rs])  # noqa: E731
    # was 15% -> 82%; now at most one 50-row block (8.5% of the rows) apart: an interpolated value at the
    # bin's edge can fall just below 10000
    assert abs(below(out) - below(real)) <= 50 / 590 + 0.01
    assert "fidelity_drift" not in _codes(rep)


def _entity_case(kind: str, rng: random.Random):
    """(header, rows, column, parse): a D4 number/date whose values go with the ID's block size."""
    rows: list[list[object]] = []
    for s in range(160):
        big = s < 30
        k = rng.randrange(20, 31) if big else rng.randrange(1, 3)
        if kind == "float":
            v: object = f"{rng.uniform(3.0, 4.0) if big else rng.uniform(1.0, 2.5):.2f}"
        elif kind == "dob":
            d = dt.date(2008, 1, 1) + dt.timedelta(days=rng.randrange(0, 700) + (0 if big else 1500))
            v = f"{d.month}/{d.day}/{d.year}"
        elif kind == "currency":
            v = f"${rng.uniform(2000, 5000) if big else rng.uniform(10, 900):,.2f}"
        elif kind == "blanks":
            v = "" if s % 10 == 5 else f"{rng.uniform(60, 90) if big else rng.uniform(20, 50):.1f}"
        else:
            raise AssertionError(kind)
        for j in range(k):
            rows.append([f"S{s:05}", v, rng.randrange(0, 100)])
    parse = {"float": float, "blanks": float,
             "dob": lambda x: dt.datetime.strptime(x.strip(), "%m/%d/%Y").toordinal(),
             "currency": lambda x: float(x.strip().replace("$", "").replace(",", ""))}[kind]
    return ["student_id", "attr", "score"], rows, "attr", parse


@pytest.mark.parametrize("kind", ["float", "dob", "currency", "blanks"])
def test_entity_number_variants_keep_their_row_distribution(tmp_path: Path, kind: str) -> None:
    header, rows, col, parse = _entity_case(kind, random.Random(13))
    for seed in (1, 2):
        res, out, rep = _run(tmp_path, f"{kind}{seed}", header, rows, None, seed)
        assert res.exit_code == 0, res.output
        real = read_dicts(tmp_path / f"{kind}{seed}" / "in" / "data.csv")
        assert _column(rep, col)["role"] == "entity", kind
        ks = _check_entity_rows(real, out, rep, col, parse)
        assert ks <= 0.1, (kind, seed, ks)
        assert "fidelity_drift" not in _codes(rep), kind
        if kind == "blanks":  # blank IDs keep their rows (R2-10)
            assert sum(r[col] == "" for r in out) == sum(r[col] == "" for r in real)


def test_unbalanced_entity_numbers_are_reported_per_row(tmp_path: Path, monkeypatch) -> None:
    """With the bin deal undone, the written rows drift (codex: KS 0.67) and the verdict says so."""
    monkeypatch.setattr(st, "balance_entity_numbers", lambda values, *a, **k: values)
    res, out, rep = _run(tmp_path, "undone", ["student_id", "expense", "visit"], _codex_panel(), None, 123)
    assert res.exit_code == 0, res.output
    fid = _column(rep, "expense")["fidelity"]
    assert fid["verdict"] == "drift" and fid["rows"]["verdict"] == "drift", fid
    assert "per row" in fid["reason"] and "KS" in fid["reason"]
    msgs = _report(rep)["messages"]
    assert any(m["code"] == "fidelity_drift" and m.get("column") == "expense" and "per row" in m["text"]
               and "ks" in m["text"] for m in msgs)
    assert _report(rep)["checks"]["fidelity"]["result"] == "WARN"


@pytest.mark.parametrize("kind", ["float", "dob", "currency", "blanks"])
def test_unbalanced_variants_are_reported_per_row(tmp_path: Path, monkeypatch, kind: str) -> None:
    """The row-level limit (max(0.05, 2 SE over the effective IDs) + the largest block's share) still catches
    a block-size-driven drift of every kind."""
    monkeypatch.setattr(st, "balance_entity_numbers", lambda values, *a, **k: values)
    header, rows, col, parse = _entity_case(kind, random.Random(13))
    res, out, rep = _run(tmp_path, f"u{kind}", header, rows, None, 1)
    assert res.exit_code == 0, res.output
    fid = _column(rep, col)["fidelity"]
    assert fid["verdict"] == "drift" and fid["rows"]["verdict"] == "drift", (kind, fid["reason"])
    assert "fidelity_drift" in _codes(rep)


def _suppressed_panel() -> list[list[object]]:
    """A within-ID-constant score with a suppression token held mostly by long ID blocks."""
    rng = random.Random(21)
    rows = []
    for s in range(160):
        big = s < 30
        k = rng.randrange(20, 31) if big else rng.randrange(1, 3)
        v = "*" if (big and s < 14) or s in (40, 50, 60) else f"{rng.uniform(10, 90):.1f}"
        for _ in range(k):
            rows.append([f"S{s:05}", v, rng.randrange(0, 100)])
    return rows


@pytest.mark.parametrize("balanced", [True, False])
def test_entity_tokens_keep_their_rows(tmp_path: Path, monkeypatch, balanced: bool) -> None:
    """Suppression tokens of a within-ID-constant number are dealt to ID blocks too, and their written share is
    judged (R3-07 sibling: '*' held by long blocks must not land on short ones)."""
    if not balanced:
        monkeypatch.setattr(st, "balance_entity_numbers", lambda values, *a, **k: values)
    rows = _suppressed_panel()
    res, out, rep = _run(tmp_path, f"tok{balanced}", ["student_id", "score", "visit"], rows, None, 1)
    assert res.exit_code == 0, res.output
    col = _column(rep, "score")
    assert col["role"] == "entity" and col["type"] == "NUMERIC_WITH_TOKENS"
    real_star = sum(r[1] == "*" for r in rows)
    syn_star = sum(r["score"] == "*" for r in out)
    checks = {c["check"]: c for c in col["fidelity"]["rows"]["checks"]}
    assert "token_shares" in checks
    if balanced:
        assert abs(syn_star - real_star) <= 30, (syn_star, real_star)  # within one long block
        assert checks["token_shares"]["ok"] and col["fidelity"]["verdict"] != "drift", col["fidelity"]["reason"]
        assert sum(r["score"] == "*" for r in {r["student_id"]: r for r in out}.values()) == 17  # IDs kept
    else:
        assert not checks["token_shares"]["ok"] and col["fidelity"]["verdict"] == "drift"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_block_sizes_independent_of_values_raise_no_warning(tmp_path: Path, seed: int) -> None:
    """A typical long file (1-8 rows per student, DOB and GPA unrelated to the row count): no false drift."""
    rng = random.Random(100 + seed)
    rows = []
    for s in range(300):
        d = dt.date(2009, 1, 1) + dt.timedelta(days=rng.randrange(0, 1500))
        gpa = f"{rng.uniform(1.5, 4.0):.2f}"
        for j in range(rng.randrange(1, 9)):
            rows.append([f"S{s:05}", d.isoformat(), gpa, j + 1])
    res, out, rep = _run(tmp_path, f"ind{seed}", ["student_id", "dob", "gpa", "visit"], rows, None, seed)
    assert res.exit_code == 0, res.output
    for col in ("dob", "gpa"):
        fid = _column(rep, col)["fidelity"]
        assert fid["verdict"] != "drift", (col, fid["reason"])
    assert "fidelity_drift" not in _codes(rep)


def test_entity_runs_are_deterministic(tmp_path: Path) -> None:
    outs = []
    for k in range(2):
        _run(tmp_path, f"edet{k}", ["student_id", "expense", "visit"], _codex_panel(), None, 5)
        outs.append((tmp_path / f"edet{k}" / "out" / "data.csv").read_bytes())
    assert outs[0] == outs[1]


def test_with_row_check_takes_the_worse_verdict() -> None:
    ent = {"verdict": "ok", "basis": "entities", "reason": "per entity (one value per ID, D4): KS 0.01 <= 0.05",
           "checks": [{"check": "ks", "ok": True, "adjusted": False, "reason": "KS 0.01 <= 0.05"}]}
    rows = {"verdict": "drift", "checks": [{"check": "ks", "ok": False, "adjusted": False,
                                            "reason": "KS 0.6661 > 0.1347"}]}
    out = fidelity.with_row_check(ent, rows)
    assert out["verdict"] == "drift" and out["basis"] == "entities" and out["rows"]["verdict"] == "drift"
    assert out["reason"].endswith("per row (each ID's value on all its rows): KS 0.6661 > 0.1347")
    ok_rows = {"verdict": "ok", "checks": [{"check": "ks", "ok": True, "adjusted": False, "reason": "KS 0.02"}]}
    assert fidelity.with_row_check(ent, ok_rows)["verdict"] == "ok"


def test_block_share_widens_only_the_row_level_ks_limit() -> None:
    a, b = np.arange(100.0), np.arange(100.0) + 8
    plain = fidelity._ks_check(a, b, False)
    wide = fidelity._ks_check(a, b, False, block_share=0.05)
    assert not plain["ok"] and wide["ok"] and wide["limit"] == pytest.approx(0.1)
    assert "largest ID block" in wide["reason"]
    few = fidelity._ks_check(a, b, False, block_share=0.0, n_eff=25.0)  # 2 SE = 1 / sqrt(25) = 0.2
    assert few["ok"] and few["limit"] == pytest.approx(0.2) and few["effective_ids"] == 25.0
    many = fidelity._ks_check(a, b, False, block_share=0.0, n_eff=10_000.0)  # 2 SE = 0.01: 0.05 stays
    assert not many["ok"] and many["limit"] == pytest.approx(0.05)
