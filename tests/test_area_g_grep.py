"""Independent grep of output text for real sensitive values (helper for the AREA G privacy tests).

Deliberately simpler than ``privacy.leaks``: it greps text for every value of
``tests/fixtures/privacy/sensitive_values.json`` and reports the KIND of each
hit, never the value (assertion messages must not print real values). The
tests at the bottom check the grep itself on planted values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "privacy"
SCHOOL = FIXTURES / "school"
WORD = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")
DIGIT_RUN = re.compile(r"\+?\(?\d(?:[ ().\-]{0,2}\d){6,14}")
TITLES = {"ms", "mr", "dr", "mrs"}


def sensitive() -> dict[str, list[str]]:
    return json.loads((FIXTURES / "sensitive_values.json").read_text(encoding="utf-8"))


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def read_texts(*dirs: Path) -> dict[str, str]:
    out = {}
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[p.relative_to(d.parent).as_posix()] = p.read_bytes().decode("utf-8", errors="replace")
    return out


def hits(text: str, sens: dict[str, list[str]] | None = None) -> list[str]:
    """Kinds of real values found in ``text`` (one entry per kind; never the value)."""
    sens = sens or sensitive()
    found: list[str] = []
    folded = text.casefold()
    tokens = {t.casefold() for t in WORD.findall(text)}
    for kind in ("ids", "staff_ids"):
        for v in sens[kind]:
            if re.search(rf"(?<![0-9A-Za-z.]){re.escape(v)}(?![0-9A-Za-z])", text):
                found.append(kind)
                break
    name_tokens = set()
    for kind in ("first_names", "last_names", "teacher_names", "staff_names"):
        for v in sens[kind]:
            name_tokens.update(t.casefold() for t in WORD.findall(v) if t.casefold() not in TITLES)
    if tokens & name_tokens:
        found.append("names")
    for v in sens["emails"]:
        local = v.split("@")[0].casefold()
        if v.casefold() in folded or re.search(rf"(?<![a-z0-9._%+\-]){re.escape(local)}(?![a-z0-9_%+\-])", folded):
            found.append("emails")
            break
    phone_digits = {_digits(v) for v in sens["phones"]}
    ssn_digits = {_digits(v) for v in sens["ssns"]}
    runs = {_digits(m.group()) for m in DIGIT_RUN.finditer(text)}
    if runs & phone_digits or any(v in text for v in sens["phones"]):
        found.append("phones")
    if runs & ssn_digits or any(v in text for v in sens["ssns"]):
        found.append("ssns")
    if any(v.casefold() in folded for v in sens["addresses"]):
        found.append("addresses")
    words = [t.casefold() for t in WORD.findall(text)]
    grams = {tuple(words[i:i + 6]) for i in range(len(words) - 5)}
    for kind in ("comments", "notes"):
        for v in sens[kind]:
            w = [t.casefold() for t in WORD.findall(v)]
            if v.casefold() in folded or any(tuple(w[i:i + 6]) in grams for i in range(len(w) - 5)):
                found.append(kind)
                break
    return found


# --------------------------------------------------------------------------- the grep itself
def test_grep_finds_every_kind_in_the_sources() -> None:
    texts = read_texts(SCHOOL)
    assert set(hits(texts["school/people.csv"])) == {"ids", "names", "emails", "phones", "ssns", "addresses",
                                                    "comments"}
    assert {"ids", "names", "emails", "notes"} <= set(hits(texts["school/enrollments.csv"]))
    assert {"staff_ids", "names", "emails", "phones"} <= set(hits(texts["school/staff.csv"]))


def test_grep_normalises_like_the_rules() -> None:
    s = sensitive()
    phone = s["phones"][0]
    assert hits("call " + "-".join([_digits(phone)[:3], _digits(phone)[3:6], _digits(phone)[6:]])) == ["phones"]
    assert hits("x " + s["first_names"][0].upper() + " y") == ["names"]
    assert hits("mean 0." + s["ids"][0]) == []  # a decimal fraction is not an ID
    assert hits("nothing to see: 0.452, group C, 1000 rows") == []


def test_fixtures_are_deterministic(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("mk", FIXTURES / "make_fixtures.py")
    mk = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mk)
    for rel, lines in mk.make().items():
        assert (FIXTURES / rel).read_bytes() == ("\n".join(lines) + "\n").encode("utf-8"), rel
    total = sum(p.stat().st_size for p in FIXTURES.rglob("*") if p.is_file())
    assert total < 300_000


def test_fixture_names_are_outside_faker_lists() -> None:
    """An invented (Faker) name can never coincide with a fixture name, so a hit is always a real leak."""
    from faker.providers.person.en_US import Provider

    faker_names = set()
    for attr in ("first_names", "first_names_male", "first_names_female", "first_names_nonbinary", "last_names"):
        faker_names |= {n.casefold() for n in getattr(Provider, attr, ())}
    s = sensitive()
    ours = {t.casefold() for k in ("first_names", "last_names") for v in s[k] for t in WORD.findall(v)}
    assert not ours & faker_names
