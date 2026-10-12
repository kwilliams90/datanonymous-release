"""Round 9, cluster ids: no real person ID in any output (G1), whatever path writes a value verbatim.

The bug class: a real ID of the run reached the output by a path that writes real values verbatim, because the
protection hung on an inference or a partial list rather than on the value itself.

* R9-01: a column whose values only partly overlap the IDs of an ID column (below ``KEY_REF_SHARE``) is no key
  reference, so its values stayed discrete numbers, factor levels, organisational codes, tokens, NA tokens or the
  exact min/max of D14, and the leak scanner excused them as kept values. Now one fail-safe rule decides, list-free
  and independent of the key discovery (``privacy.realids``): every real value a column would write verbatim that
  is a real person ID of the run (an ID column of any file, FAILed, excluded and nulled ones too; row numbers 1..n
  are no person IDs; the scanner's ID-token rule: at least 4 characters with a digit; the D34 screen's chance: a
  plain number in a range where most numbers of its shape are real IDs is no evidence) is written as a consistent
  same-shape pseudonym with its exact count (numbers: a number of the column's format inside its range), a number
  column's value statistics are withheld from the report, and the scan never excuses one (both layers).
* R9-03: ID generation avoided only the e-mail addresses of the run's real-text registry, and the scan never checked
  ID columns against the registry, so a synthetic patterned ID could equal a pseudonymised label or a value of an
  excluded or nulled column. Now every candidate ID is refused that equals a text of the registry, and the scan
  checks every synthetic ID column against it (both layers); a shape too small to avoid them FAILs clearly (D9).

Assertion messages never print a real value beyond the invented fixture data.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import read_dicts, run_cli
from datanonymous.detect import text as dtext
from datanonymous.generate import pseudonyms
from datanonymous.generate.ids import IdRegistry, RealUsers
from datanonymous.model import FileResult, SyntheticColumn, SyntheticTable
from datanonymous.privacy import leaks, realids
from datanonymous.privacy import texts as rt
from test_area_a_support import profile_of, run_table

SEED = "1"
ROSTER_IDS = ["482913", "583027"]
REFS = [197521, 235403, 381395, 482913, 591325, 651325, 739531, 847251, 935215, 953201]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = SEED):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def everything(*dirs: Path, console: str = "") -> str:
    return "\n".join([p.read_bytes().decode("utf-8", errors="replace") for d in dirs if d.exists()
                      for p in sorted(d.rglob("*")) if p.is_file()] + [console])


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    for t in json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def column(path: Path, name: str) -> list[str]:
    return [r[name] for r in read_dicts(path)]


def _partial(src: Path) -> None:
    # the reference column's header holds no ID word: under 'subject_ref' (a final ID word over ID-like keys) K-02
    # makes the column an ID column whatever its values share, and every value gets a synthetic ID
    write_csv(src / "roster.csv", {"student_id": ROSTER_IDS, "Grade": ["5", "6"]})
    write_csv(src / "visits.csv", {"subject": [str(v) for v in REFS for _ in range(2)],
                                   "Visit": [str(j) for _ in REFS for j in range(2)]})


# =========================================================================== R9-01: partial references
@pytest.mark.parametrize("flags", [[], ["--exclude", "student_id"], ["--null", "student_id"]])
def test_a_partial_reference_never_writes_a_real_id(tmp_path: Path, flags: list[str]) -> None:
    # the codex probe: one of ten six-digit references is a roster ID (10% < KEY_REF_SHARE): it was written as a
    # discrete level and excused by the scan, in the data and in the report
    src = tmp_path / "in"
    _partial(src)
    res, out, rep = run(src, tmp_path, *flags)
    assert res.exit_code == 0, res.output
    assert "482913" not in everything(out, rep, console=res.output)
    refs = column(out / "visits.csv", "subject")
    assert len(refs) == 20  # N in = N out
    assert sorted(Counter(refs).values()) == [2] * 10  # each level keeps its exact count
    fake = sorted(set(refs) - {str(v) for v in REFS})
    assert len(fake) == 1 and len(fake[0]) == 6 and min(REFS) <= int(fake[0]) <= max(REFS)
    assert set(refs) - set(fake) == {str(v) for v in REFS} - {"482913"}  # the other levels stay as they are
    col = column_report(rep, "visits.csv", "subject")
    assert col["type"] == "INTEGER" and col["params"]["pseudonymized_levels"] == 1
    assert "real_id_values" in [m["code"] for m in col["messages"]]
    assert "pmf" not in col["real"] and "min" not in col["real"]  # value statistics withheld (they print the ID)


def test_the_partial_fixture_is_hot_without_the_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # without the run's real IDs (what the code did before R9-01) the fixture writes the roster ID verbatim
    monkeypatch.setattr(realids, "of_files", lambda files: realids.RealIds())
    src = tmp_path / "in"
    _partial(src)
    res, out, _ = run(src, tmp_path, name="hot")
    assert res.exit_code == 0, res.output
    assert "482913" in column(out / "visits.csv", "subject")


def test_the_fix_is_seeded(tmp_path: Path) -> None:
    src = tmp_path / "in"
    _partial(src)
    _, a, _ = run(src, tmp_path, name="a")
    _, b, _ = run(src, tmp_path, name="b")
    assert (a / "visits.csv").read_bytes() == (b / "visits.csv").read_bytes()


def test_levels_org_codes_and_tokens_that_are_real_ids_are_pseudonymised(tmp_path: Path) -> None:
    # a level held by many rows (kept by D35 for its count), an organisational code (kept by D10) and a short number
    # (kept whatever its count) never make a real ID public
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": ["S482913", "583027", "771204"], "Grade": ["5", "6", "7"]})
    n = 60
    mentor = ["S482913" if i % 3 == 0 else ("Band", "Choir")[i % 2] for i in range(n)]
    write_csv(src / "visits.csv", {
        "Club": mentor,
        "Building": ["771204" if i % 4 == 0 else ("210", "330")[i % 2] for i in range(n)],
        "Room": ["583027" if i % 5 == 0 else ("12", "14")[i % 2] for i in range(n)],
        "Visit": [str(i % 4) for i in range(n)]})
    res, out, rep = run(src, tmp_path, "--org-code", "Building")
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, console=res.output)
    for real in ("S482913", "s482913", "771204", "583027"):
        assert real not in blob, real
    rows = read_dicts(out / "visits.csv")
    assert len(rows) == n
    written = Counter(r["Club"] for r in rows)
    assert sorted(written.values()) == sorted(Counter(mentor).values()) and {"Band", "Choir"} <= set(written)
    assert {"210", "330"} <= {r["Building"] for r in rows}  # the other organisational codes stay as they are
    assert {"12", "14"} <= {r["Room"] for r in rows}
    assert column_report(rep, "visits.csv", "Building")["type"] == "ORG_CODE"


def test_a_real_id_as_the_exact_maximum_is_never_emitted(tmp_path: Path) -> None:
    src = tmp_path / "in"
    amounts = [str(100000 + 7919 * i) for i in range(80)] + ["482913"] * 3 + ["983027"]
    write_csv(src / "roster.csv", {"student_id": ["983027", "482913"], "Grade": ["5", "6"]})
    write_csv(src / "amounts.csv", {"Amount": amounts, "Visit": [str(i % 3) for i in range(len(amounts))]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert "983027" not in everything(out, rep) and "482913" not in everything(out, rep)
    vals = [int(v) for v in column(out / "amounts.csv", "Amount")]
    assert len(vals) == len(amounts) and min(vals) == 100000  # the exact minimum (no ID) is still emitted (D14)
    col = column_report(rep, "amounts.csv", "Amount")
    assert col["type"] == "INTEGER" and col["flags"]["extreme_value_adjusted"]
    codes = [m["code"] for m in col["messages"]]
    assert "real_id_values" in codes and "extremes_not_emitted" not in codes


def test_na_tokens_items_and_per_id_values(tmp_path: Path) -> None:
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": ROSTER_IDS, "Grade": ["5", "6"]})
    write_csv(src / "scores.csv", {"Score": ["482913" if i % 7 == 0 else str(50 + i) for i in range(40)],
                                   "Grade": [str(5 + i % 3) for i in range(40)]})
    write_csv(src / "groups.csv", {"Members": ["583027;Band;Choir", "Band;Choir", "Choir;Band", "Band"] * 5,
                                   "Room": ["A", "B"] * 10})
    write_csv(src / "cases.csv", {"case_id": [f"C{100 + i}" for i in range(30) for _ in range(3)],
                                  "Code": [str(482913 if i == 0 else 10000 + 37 * i) for i in range(30) for _ in
                                           range(3)],
                                  "Visit": [str(j) for _ in range(30) for j in range(3)]})
    res, out, rep = run(src, tmp_path, "--na-value", "482913", "--id-col", "case_id")
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, console=res.output)
    assert "482913" not in blob and "583027" not in blob
    assert len(read_dicts(out / "scores.csv")) == 40 and len(read_dicts(out / "cases.csv")) == 90
    codes = column(out / "cases.csv", "Code")
    assert all(len(set(codes[3 * k:3 * k + 3])) == 1 for k in range(30))  # still one value per ID (D4)


def test_continuous_numbers_are_redrawn_from_their_own_distribution() -> None:
    # interpolated numbers: a value landing on a real ID (or its negative) is drawn again from the column's
    # distribution, so no ID is written and the shape stays (no pseudonym point mass, no uniform smear)
    rng = np.random.default_rng(3)
    n = 3000
    amounts = [str(v) for v in rng.integers(-500000, 500000, n)]
    amounts[::50] = ["482913"] * len(amounts[::50])
    amounts[1::97] = ["-482913"] * len(amounts[1::97])
    cols = {"student_id": [("482913", "583027", "771204")[i % 3] for i in range(n)], "Amount": amounts}
    tp, out = run_table(cols)
    p = profile_of(tp, "Amount")
    assert p.type.value == "INTEGER" and p.subtype == "continuous"
    assert set(p.params[realids.NUMBERS_KEY]) == {"482913", "-482913"} and not dtext.pseudonymized(p)
    vals = [int(v) for v in out["Amount"].tolist()]
    assert len(vals) == n and not {482913, -482913} & set(vals)
    real = np.asarray([int(v) for v in amounts], dtype=float)
    assert abs(np.median(vals) - np.median(real)) < 0.05 * (real.max() - real.min())


# =========================================================================== R9-01: what is never an ID
def test_small_numbers_and_row_numbers_are_never_ids() -> None:
    # 1..n row numbers are kept as they are (D9: no person IDs), and a 1-3 digit number is a count, a score or a
    # grade, never evidence of an ID (the scanner's ID-token rule)
    cols = {"id": [str(i) for i in range(1, 41)], "Score": [str(i % 30 + 1) for i in range(40)],
            "staff": [str(101 + i % 5) for i in range(40)], "Points": [str(101 + i % 7) for i in range(40)]}
    tp, out = run_table(cols, id_col=["id", "staff"])
    assert not realids.of_tables([tp])  # 101-105 have 3 digits; 1..40 are row numbers
    for name in ("Score", "Points"):
        assert dtext.pseudonymized(profile_of(tp, name)) == {}, name
    assert Counter(out["Points"].tolist()) == Counter(cols["Points"])


def test_code_words_and_dates_are_never_ids() -> None:
    ids = realids.RealIds(frozenset({"math", "482913", "s482913", "20240115"}))
    assert ids.key_of_text("MATH") is None  # no digit: a code word, never a person ID
    assert ids.key_of_text("s-482913") == "s482913" and ids.key_of_text("0482913") == "482913"
    assert ids.key_of_text("482913.0") == "482913" and ids.key_of_number(482913.0) == "482913"
    assert ids.key_of_number("4829.13") is None and ids.key_of_number(4829.13) is None
    p = profile_of(run_table({"When": ["2024-01-15", "2024-02-01"] * 10})[0], "When")
    assert realids.exempt(p)  # a date column holds dates, whatever an ID looks like


def _spaces(*groups: Any) -> list[Any]:
    from types import SimpleNamespace

    return [SimpleNamespace(raw_values=set(g)) for g in groups]


def test_chance_awareness_is_the_d34_screens() -> None:
    # a plain number counts unless an ordinary number of its shape is more likely than not a real ID
    # (privacy.headers.IdKeys.number_chance); letter-bearing and zero-padded IDs always count as written
    sparse = realids.of_namespaces(_spaces(["482913", "583027"]))
    assert sparse.key_of_number(482913) == "482913" and sparse.key_of_text("$482,913") == "482913"
    assert sparse.key_of_text("4829.13") is None and sparse.key_of_text("482-913") == "482913"  # no glued decimals
    rng = np.random.default_rng(0)
    big = sorted({str(v) for v in rng.choice(np.arange(100000, 1000000), 50000, replace=False)})
    assert realids.of_namespaces(_spaces(big)).key_of_number(int(big[7])) == big[7]  # 5.6% of six-digit numbers
    seq = realids.of_namespaces(_spaces([str(v) for v in range(100001, 100501)]))
    assert seq.key_of_number(100007) == "100007"  # sequential, but six digits: rare among six-digit numbers
    staff = realids.of_namespaces(_spaces([str(v) for v in range(1001, 1351)]))
    assert not staff and staff.key_of_number(1200) is None  # every 4-digit number around 1200 is a staff number
    records = realids.of_namespaces(_spaces([str(v) for v in range(1, 5060) if v % 97]))  # 1..n with gaps
    assert records.key_of_number(1500) is None and records.key_of_number(4321) is None
    pre = realids.of_namespaces(_spaces(["S482913"] + [f"S{v}" for v in range(1001, 1351)]))
    assert pre.key_of_text("s-482913") == "s482913" and pre.key_of_text("S1200") == "s1200"  # dense or not
    assert pre.key_of_number(482913) is None  # D8: the digits alone are another value
    pad = realids.of_namespaces(_spaces([f"{v:07d}" for v in range(1001, 1351)]))
    assert pad.key_of_text("0001200") == "1200" and pad.key_of_text("1200") is None


def test_dense_short_ids_never_turn_ordinary_numbers_into_ids(tmp_path: Path) -> None:
    # sequential 4-digit staff numbers beside 4-digit scale scores and Lexiles, and record numbers 1..n with gaps
    # (no row numbers) beside a 4-digit count: the numbers stay as they are, statistics and exact extremes included
    rng = np.random.default_rng(0)
    n = 600
    scale = [str(v) for v in np.clip(np.round(rng.normal(1550, 120, n)), 1100, 1950).astype(int)]
    lexile = [str(v) for v in rng.choice(np.arange(600, 1450, 50), n)]
    rec = [v for v in range(1, 1400) if v % 97]
    minutes = [str(v) for v in rng.integers(1000, 1350, len(rec))]
    src = tmp_path / "in"
    write_csv(src / "staff.csv", {"staff_id": [str(v) for v in range(1001, 1351)], "Dept": ["Math", "Art"] * 175})
    write_csv(src / "scores.csv", {"student_id": [str(400000 + 7 * i) for i in range(n)], "ScaleScore": scale,
                                   "Lexile": lexile})
    write_csv(src / "log.csv", {"id": [str(v) for v in rec], "Minutes": minutes})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    for name, col, real in (("scores.csv", "ScaleScore", scale), ("scores.csv", "Lexile", lexile),
                            ("log.csv", "Minutes", minutes)):
        info = column_report(rep, name, col)
        assert "real_id_values" not in [m["code"] for m in info["messages"]], col
        assert "redacted" not in info["real"] and "min" in info["real"], col
        vals = [int(v) for v in column(out / name, col)]
        assert len(vals) == len(real) and min(vals) == min(map(int, real)) and max(vals) == max(map(int, real))
    assert Counter(column(out / "scores.csv", "Lexile")) == Counter(lexile)


def test_a_number_pseudonym_stays_inside_the_range_where_the_values_are(tmp_path: Path) -> None:
    # sparse random 4-digit staff numbers beside 4-digit scale scores: the few scores that equal one are pseudonymised
    # (fail-safe), each by a free number inside the column's range near its values, so the exact min/max, every
    # level's count and the shape stay (new digits of the same shape would mostly land outside the range)
    rng = np.random.default_rng(5)
    staff = sorted(rng.choice(np.arange(1000, 10000), 200, replace=False).tolist())
    scores = np.clip(np.round(rng.normal(1550, 120, 1500)), 1100, 1950).astype(int).tolist()
    hit = set(scores) & set(staff)
    assert hit  # the fixture is hot
    src = tmp_path / "in"
    write_csv(src / "staff.csv", {"staff_id": [str(s) for s in staff], "Dept": ["Math", "Art"] * 100})
    write_csv(src / "scores.csv", {"student_id": [str(400000 + i) for i in range(len(scores))],
                                   "ScaleScore": [str(v) for v in scores]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    info = column_report(rep, "scores.csv", "ScaleScore")
    assert info["type"] == "INTEGER" and "real_id_values" in [m["code"] for m in info["messages"]]
    vals = [int(v) for v in column(out / "scores.csv", "ScaleScore")]
    assert not set(vals) & hit
    assert min(vals) == min(scores) and max(vals) == max(scores)
    assert sorted(Counter(vals).values()) == sorted(Counter(scores).values())
    assert abs(float(np.median(vals)) - float(np.median(scores))) <= 10


def test_sequential_and_prefixed_ids_are_still_never_written(tmp_path: Path) -> None:
    # dense locally but rare among six-digit numbers: still IDs; a letter-prefixed ID counts however dense its digits
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": [str(v) for v in range(100001, 100501)], "Grade": ["5"] * 500})
    write_csv(src / "staff.csv", {"staff_id": [f"T{v}" for v in range(1001, 1351)], "Dept": ["Math", "Art"] * 175})
    refs = [100007, 100123, 731221, 845512, 912345, 653321]  # under a header without an ID word (K-02: see _partial)
    write_csv(src / "visits.csv", {"subject": [str(v) for v in refs for _ in range(2)],
                                   "Club_Ref": ["T1207"] * 10 + ["Band", "Choir"],
                                   "Visit": [str(j) for _ in refs for j in range(2)]})
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    blob = everything(out, rep, console=res.output).casefold()
    for real in ("100007", "100123", "t1207"):
        assert real not in blob, real
    written = column(out / "visits.csv", "subject")
    assert sorted(Counter(written).values()) == [2] * len(refs)
    assert {"731221", "845512", "912345", "653321"} <= set(written) and all(len(v) == 6 for v in written)
    assert sorted(Counter(column(out / "visits.csv", "Club_Ref")).values()) == [1, 1, 10]


def test_the_last_resort_stays_inside_the_range() -> None:
    from decimal import Decimal

    from datanonymous.generate.numeric import _nearest_free

    forbidden = {Decimal(v) for v in (99, 100, 101, 98)}
    assert _nearest_free(Decimal(100), Decimal(1), forbidden, (Decimal(90), Decimal(101))) == Decimal(97)
    assert _nearest_free(Decimal(100), Decimal(1), forbidden, (Decimal(98), Decimal(101))) == Decimal(102)
    assert _nearest_free(Decimal(100), Decimal(1), forbidden, None) == Decimal(102)


def test_force_factor_is_the_explicit_opt_in(tmp_path: Path) -> None:
    src = tmp_path / "in"
    _partial(src)
    res, out, _ = run(src, tmp_path, "--force-factor", "subject")
    assert res.exit_code == 0, res.output
    assert "482913" in column(out / "visits.csv", "subject")  # D35: the user's word keeps real values


# =========================================================================== R9-01: the scanner never excuses one
def _visits_table() -> tuple[Any, dict[str, np.ndarray]]:
    cols = {"student_id": ["S1001", "S1002", "482913", "S1004"] * 15,
            "Buddy": [str(v) for v in REFS] * 6, "Visit": [str(i % 3) for i in range(60)]}
    return run_table(cols)


def test_the_generators_never_write_one_and_the_pseudonym_is_no_real_id() -> None:
    tp, out = _visits_table()
    p = profile_of(tp, "Buddy")
    assert "482913" in dtext.pseudonymized(p) and "482913" not in out["Buddy"].tolist()
    fake = pseudonyms.level_map(p)["482913"]
    assert not realids.run_ids(tp).holds(fake) and len(fake) == 6


def test_scan_table_flags_a_real_id_written_as_a_kept_value() -> None:
    tp, out = _visits_table()
    planted = out["Buddy"].copy()
    planted[0] = "482913"
    synth = SyntheticTable(None, list(out), len(planted),
                           [SyntheticColumn(p.index, p.name, planted if p.name == "Buddy" else out[p.name])
                            for p in tp.columns], tp)
    msgs = leaks.scan_table(tp, synth)
    assert any(m.level == "error" and m.column == "Buddy" for m in msgs)
    assert "482913" not in " ".join(m.text for m in msgs)


def test_data_scan_and_report_allowlist_never_excuse_one() -> None:
    tp, out = _visits_table()
    fr = FileResult("visits.csv", "visits.csv", "csv", tables=[tp])
    idx = leaks.SensitiveIndex.build([fr])
    data = leaks.DataScanner(idx, leaks.Allow(tokens=set(), phrases=set()))
    p = profile_of(tp, "Buddy")
    planted = np.asarray(out["Buddy"].tolist() + ["482913"], dtype=object)
    assert any(h.rule == "real person ID" for h in data.scan_column(fr, 0, p, planted))
    assert not any(h.rule == "real person ID" for h in data.scan_column(fr, 0, p, out["Buddy"]))
    allow = leaks.build_allow(type("Run", (), {"files": [fr], "messages": []})())
    assert "482913" not in allow.tokens


# =========================================================================== R9-03: IDs vs the real-text registry
@pytest.mark.parametrize("flags", [[], ["--exclude", "labels.csv::Activity"], ["--null", "labels.csv::Activity"]])
def test_synthetic_ids_avoid_every_real_text(tmp_path: Path, flags: list[str]) -> None:
    # the codex probe: two-letter IDs took the pseudonymised (or excluded / nulled) labels AV and AN
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": ["AB", "AC"], "Score": ["60", "70"]})
    write_csv(src / "labels.csv", {"Activity": ["AV", "AN", "Extra"], "Value": ["1", "2", "3"]})
    for seed in ("1", "2", "3", "4"):
        res, out, rep = run(src, tmp_path, *flags, name=f"s{seed}", seed=seed)
        assert res.exit_code == 0, res.output
        ids = {v.casefold() for v in column(out / "roster.csv", "student_id")}
        assert len(ids) == 2 and not ids & {"ab", "ac", "av", "an", "extra"}, seed


def test_the_text_fixture_is_hot_without_the_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # without the registry in ID planning and the scan of ID columns (before R9-03), seed 1 draws AV and AN
    monkeypatch.setattr(RealUsers, "text_taken", lambda self, value: False)
    monkeypatch.setattr(leaks, "scan_table", lambda tp, st: [])
    monkeypatch.setattr(leaks, "scan_outputs", lambda run, payload: [])
    src = tmp_path / "in"
    write_csv(src / "roster.csv", {"student_id": ["AB", "AC"], "Score": ["60", "70"]})
    write_csv(src / "labels.csv", {"Activity": ["AV", "AN", "Extra"], "Value": ["1", "2", "3"]})
    res, out, _ = run(src, tmp_path, name="hot")
    assert res.exit_code == 0, res.output
    assert {v.casefold() for v in column(out / "roster.csv", "student_id")} & {"av", "an"}


def test_the_id_registry_refuses_every_registry_text() -> None:
    taken = rt.RealTexts(frozenset({"av", "an", "zq"}), frozenset({"gt"}))
    for seed in range(12):
        reg = IdRegistry()
        reg.register("sid", "f.csv", None, 0, "student_id", np.asarray(["AB", "AC", "AD"], dtype=object))
        reg.build(seed, avoid_texts=taken)
        got = {v.casefold() for v in reg.namespaces["sid"].mapping.values()}
        assert not got & {"av", "an", "zq", "gt", "ab", "ac", "ad"}, seed


def test_a_shape_too_small_for_the_real_texts_fails_clearly(tmp_path: Path) -> None:
    src = tmp_path / "in"
    alpha = [a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    write_csv(src / "roster.csv", {"student_id": ["AB", "AC"], "Score": ["60", "70"]})
    write_csv(src / "labels.csv", {"Description": [s for s in alpha if s not in ("AB", "AC")]})
    res, out, rep = run(src, tmp_path, "--exclude", "labels.csv::Description")
    assert res.exit_code == 3, res.output
    assert not (out / "roster.csv").exists()
    page = json.loads((rep / "files" / "roster.csv.json").read_text(encoding="utf-8"))
    assert page["status"] == "FAIL" and "real text" in " ".join(page["reasons"])


def test_both_scan_layers_check_id_columns_against_the_registry() -> None:
    tp, out = run_table({"student_id": ["AB", "AC", "AD"] * 4, "Score": [str(60 + i) for i in range(12)]})
    registry = rt.RealTexts(frozenset({"zz"}), frozenset(), (
        (rt.Source("labels.csv", None, "Activity", rt.DROPPED), frozenset({"zz"})),))
    tp.private.extra[rt.RUN_KEY] = registry
    p = profile_of(tp, "student_id")
    planted = out["student_id"].copy()
    planted[0] = "ZZ"
    synth = SyntheticTable(None, list(out), len(planted),
                           [SyntheticColumn(q.index, q.name, planted if q.name == "student_id" else out[q.name])
                            for q in tp.columns], tp)
    assert any(m.code == "real_text_copied" and m.column == "student_id" for m in leaks.scan_table(tp, synth))
    fr = FileResult("roster.csv", "roster.csv", "csv", tables=[tp])
    idx = leaks.SensitiveIndex.build([fr])
    data = leaks.DataScanner(idx, leaks.Allow(tokens=set(), phrases=set()))
    assert any(h.rule == "real text" for h in data.scan_column(fr, 0, p, planted))
    assert not any(h.rule == "real text" for h in data.scan_column(fr, 0, p, out["student_id"]))
