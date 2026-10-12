"""Build the AREA B numeric fixtures (deterministic; every value is fabricated).

Usage: ``python tests/fixtures/numeric/make_fixtures.py [OUT_DIR]`` (default: this folder).

* ``floats.csv`` (UTF-8, LF): fixed 2dp GPA with trailing zeros and missing
  tokens, variable-decimal GPA, percents, accounting dollars ``$1,234.50`` /
  ``($25.00)``, ``-$x`` and ``$-x`` negatives, euro and pound amounts,
  proportions with and without the leading zero, explicit ``+`` signs, a
  Python-style mixed plain/scientific p-value, Excel-mangled scientific IDs,
  lognormal income, a bimodal measure, integer-valued ``72.0`` text, ``-9``/``999``
  sentinels in a discrete integer, thousands-separated counts, and a person ID.
* ``suppressed.csv`` (UTF-8, LF): suppression tokens inside numeric columns
  (``*``, ``<10``, ``n<10``, ``>95%``, ``<5%``, ``DS``, ``N<10``, ``--``),
  formatted numbers with tokens, grade levels and short codes that must stay
  categorical, and numbers with non-suppression tokens above factor-max.
* ``decimal_comma.csv`` (cp1252, CRLF, ``;``): European decimal comma
  (``1.234,50``, ``12,5%``, ``1.234,50 EUR-sign``), parsed only with ``--decimal ,``.
* ``adversarial.csv`` (UTF-8 with BOM, CRLF): long-format edge cases (see
  :func:`adversarial`): within-ID-constant columns whose only token/maximum is
  a minority value inside one student, ``12.`` numbers, scientific values over
  300 orders of magnitude, Excel General long numbers, fullwidth digits,
  ambiguous ``1,234`` counts and fabricated name-like tokens in a numeric column.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _write(path: Path, header: list[str], cols: list[list[str]], delimiter: str = ",", newline: str = "\n",
           encoding: str = "utf-8") -> None:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter, lineterminator=newline, quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    for row in zip(*cols):
        w.writerow(row)
    path.write_bytes(buf.getvalue().encode(encoding))


def _money(x: float, style: str, symbol: str = "$") -> str:
    a = f"{abs(x):,.2f}"
    if x >= 0:
        return f"{symbol}{a}"
    return {"paren": f"({symbol}{a})", "minus_outer": f"-{symbol}{a}", "minus_inner": f"{symbol}-{a}"}[style]


def _short(x: float, d: int) -> str:
    """R-style shortest text of a value rounded to d decimals ('3.5', '4', '2.75')."""
    s = f"{x:.{d}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def floats(rng: np.random.Generator, n: int = 300) -> tuple[list[str], list[list[str]]]:
    cols: dict[str, list[str]] = {}
    cols["student_id"] = [f"S{v:06d}" for v in rng.choice(np.arange(100000, 999999), n, replace=False)]
    gpa = np.clip(4 - rng.gamma(2, 0.45, n), 0, 4)
    gpa[:6] = [4, 4, 0, 4, 3.5, 2]
    g = [f"{x:.2f}" for x in gpa]
    for i in rng.choice(np.arange(6, n), 12, replace=False):
        g[i] = "" if i % 2 else "NA"
    cols["gpa"] = g
    cols["gpa_r"] = [_short(x, 2) for x in np.clip(4 - rng.gamma(2, 0.45, n), 0, 4)]
    cols["attendance_rate"] = [f"{x:.1f}%" for x in np.clip(rng.normal(92, 6, n), 40, 100)]
    cols["fee_balance"] = [_money(x, "paren") for x in np.round(rng.normal(300, 900, n), 2)]
    cols["refund"] = [_money(x, "minus_outer") for x in np.round(rng.normal(-20, 60, n), 2)]
    cols["adjustment"] = [_money(x, "minus_inner") for x in np.round(rng.normal(0, 40, n), 2)]
    cols["price_eur"] = [_money(x, "paren", "\u20ac") for x in np.round(rng.lognormal(5, 1.2, n), 2)]
    cols["tuition_gbp"] = [f"\u00a3{x:,.0f}" for x in np.round(rng.uniform(4000, 18000, n), -1)]
    cols["proportion"] = [f"{x:.3f}" for x in rng.beta(2, 5, n)]
    cols["stata_prop"] = [f"{x:.3f}"[1:] for x in rng.uniform(0.05, 0.95, n)]
    cols["change"] = [f"{x:+.1f}" if x != 0 else "0.0" for x in np.round(rng.normal(0.5, 2, n), 1)]
    cols["p_value"] = [repr(float(f"{x:.3g}")) for x in 10 ** rng.uniform(-7, 0, n)]
    cols["sci_id"] = [f"{x:.5E}" for x in rng.integers(10**11, 10**12, n)]
    cols["income"] = [f"{x:.2f}" for x in rng.lognormal(10.5, 0.8, n)]
    bim = np.concatenate([rng.normal(40, 5, n * 3 // 5), rng.normal(75, 8, n - n * 3 // 5)])
    cols["bimodal"] = [f"{x:.1f}" for x in rng.permutation(bim)]
    cols["score_x10"] = [f"{x}.0" for x in rng.integers(150, 1000, n)]
    hours = rng.integers(0, 40, n).astype(object)
    hours[rng.choice(n, 15, replace=False)] = -9
    hours[rng.choice(n, 8, replace=False)] = 999
    cols["sentinel_hours"] = [str(x) for x in hours]
    cols["big_count"] = [f"{x:,}" for x in rng.integers(50, 250000, n)]
    return list(cols), list(cols.values())


def suppressed(rng: np.random.Generator, n: int = 300) -> tuple[list[str], list[list[str]]]:
    cols: dict[str, list[str]] = {}
    cols["school_id"] = [str(x) for x in rng.choice(np.arange(101, 121), n)]
    cols["subgroup"] = list(rng.choice(["All", "Female", "Male", "EcoDis", "ELL", "SPED"], n))
    tested = rng.integers(0, 160, n)
    cols["n_tested"] = [str(x) if x >= 10 else str(rng.choice(["*", "<10", "n<10"])) for x in tested]
    pct = rng.uniform(0, 100, n)
    cols["pct_proficient"] = [(">95%" if x > 95 else "<5%" if x < 5 else f"{x:.1f}%") if t >= 10 else "*"
                              for x, t in zip(pct, tested)]
    avg = rng.normal(250, 18, n)
    r = rng.random(n)
    cols["avg_score"] = [("--" if u < 0.05 else "N<10" if u < 0.1 else "DS" if u < 0.13 else f"{x:.1f}")
                         for x, u in zip(avg, r)]
    cols["grade_level"] = list(rng.choice(["PK", "KG"] + [f"{g:02d}" for g in range(1, 13)], n))
    cols["status"] = list(rng.choice(["1", "2", "3", "4", "5", "Pending"], n))
    days = rng.integers(0, 181, n)
    r = rng.random(n)
    cols["days_enrolled"] = [("Pending" if u < 0.04 else "Withdrawn" if u < 0.07 else str(x)) for x, u in zip(days, r)]
    fee = np.round(rng.uniform(5, 80, n), 2)
    r = rng.random(n)
    cols["fee"] = [("*" if u < 0.1 else f"${x:,.2f}") for x, u in zip(fee, r)]
    return list(cols), list(cols.values())


def decimal_comma(rng: np.random.Generator, n: int = 200) -> tuple[list[str], list[list[str]]]:
    def de(x: float, d: int = 2, group: bool = True) -> str:
        s = f"{x:,.{d}f}" if group else f"{x:.{d}f}"
        return s.translate(str.maketrans({",": ".", ".": ","}))

    cols: dict[str, list[str]] = {}
    cols["id"] = [str(i) for i in range(1, n + 1)]
    cols["betrag"] = [de(x) for x in np.round(rng.normal(1500, 1200, n), 2)]
    cols["anteil"] = [de(x, 1, False) + "%" for x in rng.uniform(0, 100, n)]
    cols["preis"] = [de(x) + " \u20ac" for x in rng.lognormal(6, 1, n)]
    cols["menge"] = [str(x) for x in rng.integers(0, 40, n)]
    cols["faktor"] = [de(x, 2, False) for x in rng.uniform(0, 2, n)]
    return list(cols), list(cols.values())


#: Fabricated name-like tokens typed into a numeric column (never public: not suppression-like).
FAKE_NAME_TOKENS = ("Quillon Varga", "Tess Okonkwo-Lind")
#: Entities (0-based) whose minority row holds the only '<10' token / the only maximum balance.
TOKEN_ENTITY, EXTREME_ENTITY = 50, 60
MAX_BALANCE = "$9,999.99"


def adversarial(rng: np.random.Generator, n_students: int = 150) -> tuple[list[str], list[list[str]]]:
    """Long-format edge cases (UTF-8 BOM, CRLF): 2-4 rows per student.

    * ``score_supp``/``balance``: constant within each student (D4) except one
      student whose minority row holds the column's only ``<10`` token / the
      only maximum balance (D14 floors on the per-entity model);
    * ``sas_amount``: integer-valued numbers written ``12.`` (pandas float64);
    * ``wide_sci``: scientific values from 1e-150 to 1e+150;
    * ``account_no``: Excel General long numbers (11-digit integers in full,
      12-digit ones as ``d.dddddE+11``);
    * ``fw_level``: fullwidth-digit levels (never numbers);
    * ``count_amb``: ``1,234``-like counts without decimals (ambiguous grouping);
    * ``visits``: numbers with fabricated name-like tokens (private, verbatim).
    """
    cols: dict[str, list[str]] = {k: [] for k in ("student_id", "term", "score_supp", "balance", "sas_amount",
                                                    "wide_sci", "account_no", "fw_level", "count_amb", "visits")}
    fw = str.maketrans("0123456789", "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19")
    for s in range(n_students):
        k = int(rng.integers(2, 5))
        score = "*" if s % 15 == 0 else f"{rng.uniform(20, 99):.1f}"
        balance = f"${rng.uniform(5, 5000):,.2f}"
        for j in range(k):
            cols["student_id"].append(f"Q{70000 + 7 * s}")
            cols["term"].append(f"T{j + 1}")
            cols["score_supp"].append("<10" if s == TOKEN_ENTITY and j == k - 1 else score)
            cols["balance"].append(MAX_BALANCE if s == EXTREME_ENTITY and j == k - 1 else balance)
            cols["sas_amount"].append(f"{int(rng.integers(0, 90000))}.")
            cols["wide_sci"].append(f"{10 ** rng.uniform(-150, 150):.3e}")
            acct = int(rng.integers(10**10, 10**12))
            cols["account_no"].append(str(acct) if acct < 10**11 else f"{acct:.5E}")
            cols["fw_level"].append(str(int(rng.integers(1, 9))).translate(fw))
            cols["count_amb"].append(f"{int(rng.integers(1000, 99999)):,}")
            u = rng.random()
            cols["visits"].append(FAKE_NAME_TOKENS[0] if u < 0.03 else FAKE_NAME_TOKENS[1] if u < 0.05
                                  else str(int(rng.integers(0, 200))))
    return list(cols), list(cols.values())


def main(out: Path = HERE) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h, c = floats(np.random.default_rng(20261004))
    _write(out / "floats.csv", h, c)
    h, c = suppressed(np.random.default_rng(20261005))
    _write(out / "suppressed.csv", h, c)
    h, c = decimal_comma(np.random.default_rng(20261006))
    _write(out / "decimal_comma.csv", h, c, delimiter=";", newline="\r\n", encoding="cp1252")
    h, c = adversarial(np.random.default_rng(20261007))
    _write(out / "adversarial.csv", h, c, newline="\r\n", encoding="utf-8-sig")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
