"""Sentence-like prose in factor columns: D33 (codex R3-02, option B), superseded by D35 (user, 2026-10-05).

Bug class: a low-cardinality text column of PII-free prose (``Return medication to Althea Voss after the second
class``) was detected as a FACTOR, so its real sentences - and the names in them - were written verbatim to the
output and listed in the report. D33 converted any column with a sentence-like level to FREE_TEXT. D35 replaced
that content rule (and every other value heuristic) with an allowlist: a real level is written verbatim only when
it is a vocabulary word or held by >= ``--min-cell`` distinct entities with no personal-data shape; every other
level gets a same-shape pseudonym with its exact count, and the column stays a factor. So a RARE sentence (the
identifying one: a single ``Picked up by his grandmother ...`` among ``Bus`` / ``Walker``) is never copied, while
a sentence shared by >= k entities is kept (D35's accepted residual risk). ``--force-factor COL`` (CLI or config)
keeps every real level.

:func:`detect.text.is_sentence_like` stays (the header and path screens of D34 still judge text by its content);
its unit tests stay here. The false-conversion tests pin that ordinary labels stay real levels.
Assertion messages never print a real value beyond invented fixture data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import read_dicts, run_cli
from datanonymous import pipeline
from datanonymous.config import Settings
from datanonymous.detect import text as dtext
from datanonymous.model import ColumnType as T
from datanonymous.model import SyntheticColumn, SyntheticTable
from datanonymous.privacy import leaks

from test_area_a_support import detect, profile_of, run_table

#: Invented narratives with no personal-data shape (no e-mail, phone, SSN, address, URL): only their count keeps
#: them from being copied (D35).
PROSE = ["Return medication to Althea Voss after the second class",
         "Contact Floriana Drexel when the student needs transport",
         "Picked up by his grandmother Zelmira Quenby on Fridays"]
REAL_WORDS = ("Althea", "Voss", "Floriana", "Drexel", "Zelmira", "Quenby", "grandmother", "medication",
              "transport", "Fridays")
MIXED_LEVELS = {"Bus": 50, "Car rider": 40, "Walks home": 29,
                "Picked up by his grandmother Zelmira Quenby after school": 1}


def cycle(values: list[str], n: int = 120) -> list[str]:
    return [values[i % len(values)] for i in range(n)]


def expand(levels: dict[str, int]) -> list[str]:
    return [v for v, c in levels.items() for _ in range(c)]


def ids(n: int = 120) -> list[str]:
    return [f"S{i:05d}" for i in range(n)]


def write_csv(path: Path, columns: dict[str, list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(columns))
        w.writerows(zip(*columns.values()))
    return path


def run(src: Path, tmp: Path, *extra: str, name: str = "run", seed: str = "7"):
    out, rep = tmp / f"{name}-out", tmp / f"{name}-rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", seed, *extra)
    return res, out, rep


def texts(*dirs: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file())


def found_words(words: tuple[str, ...], *dirs: Path, console: str = "") -> int:
    blob = (texts(*dirs) + "\n" + console).casefold()
    return sum(1 for w in words if w.casefold() in blob)


def hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def column_report(rep: Path, name: str, col: str) -> dict[str, Any]:
    data = json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))
    for t in data["tables"]:
        for c in t["columns"]:
            if c["name"] == col:
                return c
    raise KeyError(col)


def codes(det) -> list[str]:
    return [m.code for m in det.messages]


# --------------------------------------------------------------------------- the sentence test
@pytest.mark.parametrize("value", [
    *PROSE,
    "Student rides the bus with her older brother",
    "Meets with the counselor before the first period",
    "Needs a quiet room when he takes a test",
    "Return medication to the nurse after lunch.",
    "Call his mother if the bus is late",  # lowercase 'his' / 'is' / 'if' never stay lowercase in Title Case
    "Call His Mother if the Bus Is Late",
])
def test_sentence_like(value: str) -> None:
    assert dtext.is_sentence_like(value)


@pytest.mark.parametrize("value", [
    # example-data levels
    "some high school", "associate's degree", "bachelor's degree", "Software Engineer", "Real Estate Developer",
    # Likert, frequency and performance scales
    "Strongly agree", "Neither agree nor disagree", "Less than once a month or never",
    "Does not meet grade level expectations", "Did Not Yet Meet Expectations",
    # course titles (Title Case) and long noun-phrase labels
    "Introduction to the History of the United States", "AP English Language and Composition",
    "Advanced Placement Computer Science Principles", "History of the Americas from the Colonial Era to Today",
    # accommodation and discipline labels
    "Extended time on tests", "Preferential seating", "Extended time on tests and quizzes",
    "Preferential seating near the front", "Fighting - Mutual Combat", "ISS - 1 day", "Tardy (3+)",
    "Free/Reduced Lunch", "Room 101, 2nd Floor",
    # all-caps text has no lowercase function words (D33 counts lowercase ones only)
    "INTRODUCTION TO THE HISTORY OF THE UNITED STATES",
])
def test_not_sentence_like(value: str) -> None:
    assert not dtext.is_sentence_like(value)


def test_title_case_rule() -> None:
    assert dtext.is_title_case("Introduction to the History of the United States")
    assert dtext.is_title_case("History of the Americas from the Colonial Era to Today")
    assert dtext.is_title_case("Grade 12: The (Optional) Senior Project")
    assert not dtext.is_title_case("Return medication to the office")
    assert not dtext.is_title_case("Call His Mother when the Bus Is Late")  # 'when' is not a minor word
    assert not dtext.is_title_case("to the of")  # no capitalized word


def test_function_word_list_is_explicit_and_lowercase() -> None:
    fw = dtext.SENTENCE_FUNCTION_WORDS
    assert {"to", "the", "at", "after", "before", "when", "with", "for", "his", "her", "their"} <= fw
    assert not fw & {"and", "or", "nor", "of", "than", "not", "some", "all", "no", "once"}  # label connectors
    assert all(w == w.lower() and w.isalpha() for w in fw)


# --------------------------------------------------------------------------- D35: rare prose is pseudonymised
#: Each sentence held by 3 students, the rest an ordinary vocabulary level: every sentence is rare.
RARE_PROSE = [PROSE[i % 3] if i < 9 else "None" for i in range(120)]


@pytest.mark.parametrize("values, header, was", [
    (RARE_PROSE, "accommodation", T.FACTOR),
    ([PROSE[1] if i < 4 else "Bus" for i in range(60)], "pickup", T.FACTOR),
    ([PROSE[0]] * 6, "accommodation", T.CONSTANT),
    ([PROSE[1] if i < 5 else "" for i in range(60)], "transport", T.FLAG_SINGLE_TOKEN),
])
def test_rare_sentence_levels_are_pseudonymised(values: list[str], header: str, was: T) -> None:
    tp, cols = run_table({header: values, "n": [str(i) for i in range(len(values))]})
    p = profile_of(tp, header)
    assert p.type is was  # the column keeps its type (D33's FREE_TEXT conversion is retired)
    assert set(dtext.pseudonymized(p)) == {v for v in values if v in PROSE}
    msg = next(m for m in p.messages if m.code == "pseudonymized_levels")
    assert not any(w in msg.text or w in p.reason for w in REAL_WORDS)  # counts only, never a value
    out = cols[header].tolist()
    assert len(out) == len(values) and not set(out) & set(PROSE)
    assert sum(1 for v in out if not v.strip()) == sum(1 for v in values if not v.strip())  # blanks kept
    assert sorted(Counter(v for v in out if v not in values).values()) == sorted(
        Counter(v for v in values if v in PROSE).values())  # exact counts under pseudonyms
    for fake in {v for v in out if v not in values}:  # same word count and word lengths
        assert any([len(w) for w in fake.split()] == [len(w) for w in real.split()] for real in PROSE)
    assert leaks.sensitive_kind(p) == leaks.LABEL


def test_frequent_sentences_are_kept_d35_residual() -> None:
    # D35's accepted residual risk: a sentence shared by >= --min-cell entities under a header that hints no
    # personal data is kept (D33 would have converted the column)
    values = expand({PROSE[0]: 70, PROSE[1]: 35, PROSE[2]: 15})
    tp, cols = run_table({"accommodation": values})
    p = profile_of(tp, "accommodation")
    assert p.type is T.FACTOR and dtext.pseudonymized(p) == {}
    assert Counter(cols["accommodation"].tolist()) == Counter(values)


def test_na_tokens_are_written_back_at_their_counts() -> None:
    values = [PROSE[1] if i < 3 else PROSE[2] if i < 6 else "N/A" if i % 2 else "Bus" for i in range(120)]
    tp, cols = run_table({"pickup": values})
    out = cols["pickup"].tolist()
    assert out.count("N/A") == values.count("N/A") and out.count("Bus") == values.count("Bus")
    assert not set(out) & set(PROSE)


def test_one_sentence_level_is_pseudonymised_and_the_labels_stay() -> None:
    values = expand(MIXED_LEVELS)
    tp, cols = run_table({"dismissal": values})
    p = profile_of(tp, "dismissal")
    assert p.type is T.FACTOR and set(dtext.pseudonymized(p)) == {k for k, v in MIXED_LEVELS.items() if v == 1}
    out = Counter(cols["dismissal"].tolist())
    assert {k: out[k] for k in ("Bus", "Car rider", "Walks home")} == {"Bus": 50, "Car rider": 40, "Walks home": 29}
    assert sorted(out.values()) == sorted(MIXED_LEVELS.values()) and len(out) == len(MIXED_LEVELS)


def test_mixed_column_without_a_sentence_stays_a_factor() -> None:
    levels = {k: v for k, v in MIXED_LEVELS.items() if v > 1} | {"Picked up by grandmother": 1}
    det = detect(expand(levels), "dismissal")
    assert det.type is T.FACTOR and "sentence_levels" not in codes(det)


def test_long_labels_keep_their_levels_without_a_warning() -> None:
    # the D33 'prose_levels' warning is retired: the allowlist decides (these labels are held by 40 rows each)
    values = cycle(["Advanced Placement Computer Science Principles", "International Baccalaureate Diploma Programme",
                    "Career and Technical Education Pathway Completer"])
    det = detect(values, "program_track")
    assert det.type is T.FACTOR and "prose_levels" not in codes(det) and "sentence_levels" not in codes(det)
    tp, cols = run_table({"program_track": values})
    assert Counter(cols["program_track"].tolist()) == Counter(values)


# --------------------------------------------------------------------------- non-conversions
@pytest.mark.parametrize("values, header, expected", [
    (cycle(["some college", "associate's degree", "high school", "some high school", "bachelor's degree",
            "master's degree"]), "parental level of education", T.ORDINAL),
    (cycle(["Software Engineer", "Doctor", "Real Estate Developer", "Government Officer", "Unknown"]),
     "career_aspiration", T.FACTOR),
    (cycle(["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"]), "q1",
     None),
    (cycle(["Introduction to the History of the United States", "AP English Language and Composition",
            "Algebra I Honors"]), "class_title", None),
    (cycle(["Extended time on tests", "Preferential seating", "Frequent breaks"]), "accommodation", T.FACTOR),
    (cycle(["Fighting - Mutual Combat", "Disruptive Behavior", "ISS - 1 day", "Tardy (3+)"]), "incident", T.FACTOR),
    # D10: an org code keeps its real values, sentence-case course titles included
    (cycle(["Introduction to the history of the united states", "Reading for the gifted and talented",
            "Algebra I"]), "Course", T.ORG_CODE),
])
def test_ordinary_labels_are_not_converted(values: list[str], header: str, expected: T | None) -> None:
    det = detect(values, header)
    assert det.type not in (T.FREE_TEXT, T.HIGH_CARD_CATEGORICAL)
    if expected is not None:
        assert det.type is expected
    assert "sentence_levels" not in codes(det)


# --------------------------------------------------------------------------- explicit overrides
def test_force_factor_keeps_real_levels() -> None:
    det = detect(cycle(PROSE), "accommodation", force_factor=["accommodation"])
    assert det.type is T.FACTOR and det.source == "force-factor"
    assert "sentence_levels" not in codes(det)
    tp, cols = run_table({"accommodation": cycle(PROSE)}, force_factor=["accommodation"])
    assert set(cols["accommodation"].tolist()) == set(PROSE)


def test_force_factor_cli_and_config(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "in" / "roster.csv", {"student_id": ids(), "accommodation": RARE_PROSE,
                                                    "dismissal": expand(MIXED_LEVELS)})
    res, out, _ = run(src.parent, tmp_path, "--force-factor", "accommodation", name="cli")
    assert res.exit_code == 0, res.output
    rows = read_dicts(out / "roster.csv")
    assert Counter(r["accommodation"] for r in rows) == Counter(RARE_PROSE)  # explicitly kept
    assert not {r["dismissal"] for r in rows} & {k for k, v in MIXED_LEVELS.items() if v == 1}  # not named
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("force_factor: [dismissal]\n", encoding="utf-8")
    res, out, _ = run(src.parent, tmp_path, "--config", cfg, name="cfg")
    assert res.exit_code == 0, res.output
    rows = read_dicts(out / "roster.csv")
    assert Counter(r["dismissal"] for r in rows) == Counter(MIXED_LEVELS)
    assert not {r["accommodation"] for r in rows} & set(PROSE)


def test_force_text_still_gives_filler() -> None:
    det = detect(cycle(PROSE), "accommodation", force_text=["accommodation"])
    assert det.type is T.FREE_TEXT and det.source == "force-text"


# --------------------------------------------------------------------------- end to end: nothing rare copied
def test_end_to_end_no_rare_sentence_reaches_output_report_or_docs(tmp_path: Path) -> None:
    n = 120
    src = write_csv(tmp_path / "in" / "roster.csv", {
        "student_id": ids(n), "accommodation": RARE_PROSE, "dismissal": expand(MIXED_LEVELS),
        "arrival": [PROSE[0] if i < 2 else "" for i in range(n)], "grade": cycle(["5", "6", "7"], n)})
    res, out, rep = run(src.parent, tmp_path)
    assert res.exit_code == 0, res.output
    assert found_words(REAL_WORDS, out, rep, console=res.output) == 0
    rows = read_dicts(out / "roster.csv")
    assert len(rows) == n and list(rows[0]) == ["student_id", "accommodation", "dismissal", "arrival", "grade"]
    for col in ("accommodation", "dismissal", "arrival"):
        c = column_report(rep, "roster.csv", col)
        assert c["type"] in ("FACTOR", "FLAG_SINGLE_TOKEN"), col
        assert "pseudonymized_levels" in [m["code"] for m in c["messages"]], col
    assert {r["grade"] for r in rows} == {"5", "6", "7"}  # ordinary columns unchanged
    data = json.loads((rep / "files" / "roster.csv.json").read_text(encoding="utf-8"))
    assert data["checks"]["leak_scan"]["result"] == "PASS"
    res2, out2, _ = run(src.parent, tmp_path, name="again")  # same seed: byte-identical
    assert res2.exit_code == 0 and hashes(out) == hashes(out2)


def test_end_to_end_native_dictionary_column(tmp_path: Path) -> None:
    n = 60
    src = tmp_path / "in"
    src.mkdir()
    pq.write_table(pa.table({"student_id": ids(n),
                             "accommodation": pa.array(RARE_PROSE[:n]).dictionary_encode()}), src / "a.parquet")
    res, out, rep = run(src, tmp_path)
    assert res.exit_code == 0, res.output
    vals = pq.read_table(out / "a.parquet").column("accommodation").to_pylist()
    assert len(vals) == n and not {str(v).casefold() for v in vals} & {p.casefold() for p in PROSE}
    assert found_words(REAL_WORDS, out, rep, console=res.output) == 0


# --------------------------------------------------------------------------- the leak scanner indexes them
@pytest.fixture(scope="module")
def prose_run(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("d33_leaks")
    src = write_csv(base / "in" / "roster.csv", {"student_id": ids(), "accommodation": RARE_PROSE,
                                                 "dismissal": expand(MIXED_LEVELS)})
    run_ = pipeline.execute(Settings(mode="fidelity", seed=5), src.parent, base / "out", base / "rep")
    assert pipeline.exit_code(run_) == 0
    return run_


def test_pseudonymised_columns_are_sensitive(prose_run) -> None:
    fr = next(f for f in prose_run.files if f.rel_path == "roster.csv")
    kinds = {p.name: leaks.sensitive_kind(p) for p in fr.tables[0].columns}
    assert kinds["accommodation"] == leaks.LABEL and kinds["dismissal"] == leaks.LABEL


def test_report_and_docs_scan_catch_a_real_sentence(prose_run) -> None:
    scanner = leaks.Scanner(leaks.SensitiveIndex.build(prose_run.files), leaks.build_allow(prose_run))
    for sentence in [*PROSE, "Picked up by his grandmother Zelmira Quenby after school"]:
        assert scanner.hits_in_text(f"note: {sentence.upper()} (copied)"), "a real sentence was not caught"
    assert not scanner.hits_in_text("Lorem ipsum dolor sit amet consectetur adipiscing elit")
    assert not scanner.hits_in_text("Bus / Car rider / Walks home")  # the kept levels are excused


def test_same_column_scan_fails_a_copied_level(prose_run) -> None:
    fr = next(f for f in prose_run.files if f.rel_path == "roster.csv")
    tp = fr.tables[0]
    p = profile_of(tp, "accommodation")
    synth = np.asarray(cycle(["Lorem ipsum dolor sit amet consectetur adipiscing"], tp.n_rows), dtype=object)
    synth[3] = PROSE[2]  # one real sentence copied
    st = SyntheticTable(tp.name, tp.header, tp.n_rows, [SyntheticColumn(p.index, p.name, synth)], tp)
    assert any(m.code == "copied_value" and m.column == "accommodation" for m in leaks.scan_table(tp, st))
