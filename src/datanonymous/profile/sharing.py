"""Private generation views for sharing: local-person rare-unit suppression (D42/43).

Capture evidence while raw rows exist. Never infer people from row counts or
borrow holders from a sibling fact table or a code lookup. Source profiles and
registries stay intact for the independent verification stage.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import re
from typing import Any

import numpy as np

from ..detect import dates as date_detect
from ..detect import text as text_detect
from ..errors import ColumnRefusal
from ..hints import header_hints, tokens
from ..model import LEVEL_TYPES, NEVER_COPY_TYPES, ColumnModel, ColumnProfile, ColumnType, Message, MissingnessModel, RawTable, TableProfile
from ..generate.quota import largest_remainder
from ..generate import sharing_numeric
from . import entity

EVIDENCE_KEY = "_sharing_local_holders"
SOURCE_MODEL_KEY = "_sharing_source_model"
SOURCE_COLUMN_KEY = "_sharing_source_column"
SUPPRESSED_KEY = "_sharing_suppressed_units"
PERSON_DATE_KEY = "_sharing_person_date"
LOOKUP_KEY = "_sharing_lookup"


def _other_person_attribute(p: ColumnProfile) -> bool:
    """Recognizable intrinsic attributes need support from their own actors.

    Student keys cannot establish how many parents supplied a name, income or
    birth date: twenty children can share one parent. Neither a role-labelled
    ID nor a matching attribute by itself proves the actor ownership needed
    for these support counts. Until that contract is established, sharing
    refuses such attributes, including in putative lookup tables (D48/D49).

    A role word alone is not enough: ordinary Teacher_Comments and
    Parent_Contact_Date can describe the student's record. Recognizable PII
    inside those comments still needs support from its own actors.
    """
    if not entity.other_person_header(p.name):
        return False
    if header_hints(p.name).pii is not None or p.type in {
        ColumnType.PERSON_NAME, ColumnType.EMAIL, ColumnType.PHONE, ColumnType.SSN,
        ColumnType.STREET_ADDRESS, ColumnType.CITY, ColumnType.STATE, ColumnType.ZIP,
        ColumnType.IP, ColumnType.URL, ColumnType.LAT, ColumnType.LON,
    }:
        return True
    attributes = {
        "age", "birth", "birthday", "dob", "gender", "sex", "race", "ethnicity",
        "ethnic", "nationality", "citizenship", "language", "marital", "education",
        "educational", "degree", "qualification", "qualifications", "occupation",
        "employment", "employer", "income", "salary", "wage", "wages", "earnings",
        "credential", "credentials", "certification", "license", "licence",
    }
    if attributes.intersection(tokens(p.name)):
        return True
    if p.model is not None:
        from ..generate.embedded import pii_spans

        return any(isinstance(raw, str) and pii_spans(raw, digit_phone=True)
                   for raw in p.model.nonmissing_counts)
    return False


def _level_column(p: ColumnProfile) -> bool:
    # Discrete numeric measures keep their separate numeric policy; a numeric
    # category explicitly typed FACTOR still follows this level policy.
    return (p.type in LEVEL_TYPES and p.type is not ColumnType.EMPTY or p.type in (ColumnType.STATE, ColumnType.ZIP)
            or p.type is ColumnType.DIGIT_CODE and bool(p.params.get("levels_mode")))


def _key(raw: Any) -> str:
    return "\x00native-null" if raw is None else text_detect.unit_key(str(raw))


def _variant(raw: Any) -> str:
    # unit_holders case-folds its keys; hex preserves the exact raw spelling,
    # whitespace, punctuation and case that quota generation would retain.
    return "v:" + repr(None if raw is None else str(raw)).encode("utf-8").hex()


def _date_tokens(p: ColumnProfile) -> set[str]:
    return set(p.params.get("sentinels") or ()) | set(p.params.get("_other_tokens") or ())


def _invented(p: ColumnProfile) -> bool:
    return (p.type in NEVER_COPY_TYPES - {ColumnType.IDENTIFIER} or p.type is ColumnType.CITY
            or p.type is ColumnType.DIGIT_CODE and not p.params.get("levels_mode"))


def _shape(raw: str) -> str:
    # Keep word lengths, case, digits, punctuation, whitespace and non-ASCII
    # details: all can survive a same-shape pseudonym and identify its donor.
    return "".join("A" if "A" <= c <= "Z" else "a" if "a" <= c <= "z"
                   else "0" if "0" <= c <= "9" else c for c in raw)


def _retained_semantics(kind: str, raw: str) -> str:
    """Source subdomains or literal vocabulary retained by the PII makers."""
    if kind == "email":
        return raw.rpartition("@")[2].casefold()
    if kind in ("url", "domain"):
        return raw.split("://", 1)[-1].split("/", 1)[0].casefold()
    if kind == "phone":
        from ..privacy.digits import phone_base

        base = phone_base(raw)
        return repr((base[0][:3] if base[2] else base[0][:-7], base[1:])) if base else "unsupported"
    if kind == "person_name":
        from ..generate.pii import _CAMEL_PARTS, _NAME_PREFIXES, _NAME_TOKEN_RE
        from ..generate.text import case_style

        kept = text_detect.NAME_PARTICLES | text_detect.NAME_SUFFIXES | text_detect.NAME_TITLES | {"and"}
        literals = [("word", w.casefold()) for w in re.findall(r"[^\W\d_]+", raw) if w.casefold() in kept]
        for chunk in re.split(r"(\s+|,)", raw):
            tokens = _NAME_TOKEN_RE.findall(chunk)
            for i, token in enumerate(tokens):
                if not token[0].isalpha():
                    continue
                following = tokens[i + 1] if i + 1 < len(tokens) else ""
                if len(token) <= 2 and following in ("'", "\u2019") and i + 2 < len(tokens):
                    literals.append(("apostrophe", token + following))
                parts = _CAMEL_PARTS.findall(token) if case_style(token) == "mixed" else []
                if len(parts) > 1 and "".join(parts) == token and parts[0].casefold() in _NAME_PREFIXES:
                    literals.append(("surname-prefix", parts[0]))
        return repr(literals)
    if kind in ("street_address", "address"):
        kept = (set(text_detect.STREET_SUFFIXES) | text_detect.DIRECTIONALS | text_detect.UNIT_WORDS
                | text_detect.US_STATE_ABBR | text_detect.NAME_SUFFIXES | {"st", "nd", "rd", "th"})
        return repr([w.casefold() for w in re.findall(r"[^\W\d_]+", raw) if w.casefold() in kept])
    if kind == "city":
        from ..generate.pii import _CITY_STATE_RE

        match = _CITY_STATE_RE.fullmatch(raw.strip())
        return match.group("state") if match else ""
    if kind == "digit_code":
        return str(len(raw.strip()) - len(raw.strip().lstrip("0")))
    return ""


def _generation_classes(p: ColumnProfile) -> dict[Any, str]:
    """Private source classes whose exact shape/frequency the generator preserves."""
    if p.model is None:
        return {}
    values = [v for v in p.model.nonmissing_counts if isinstance(v, str)]
    if not values:
        return {}
    if _invented(p):
        from ..generate.embedded import pii_spans
        from ..generate.text import _lead_digit

        spans = {v: [(v[a:b], kind) for a, b, kind in pii_spans(v, digit_phone=True)] for v in values}
        span_counts: Counter = Counter()
        group_counts: Counter = Counter()
        for raw in values:
            count = int(p.model.nonmissing_counts[raw])
            group_counts[raw.strip().casefold()] += count
            for part, kind in spans[raw]:
                span_counts[kind, part.strip().casefold()] += count
        bases = {}
        variants: dict[str, list[tuple]] = {}
        span_variants: dict[tuple[str, str], list[tuple]] = {}
        for raw in values:
            embedded = [(kind, _shape(part), span_counts[kind, part.strip().casefold()],
                         _retained_semantics(kind, part)) for part, kind in spans[raw]]
            base = (_shape(raw), int(p.model.nonmissing_counts[raw]), _lead_digit(raw.strip()),
                    _retained_semantics(p.type.value.lower(), raw), embedded)
            bases[raw] = base
            variants.setdefault(raw.strip().casefold(), []).append(base)
            for part, kind in spans[raw]:
                span_variants.setdefault((kind, part.strip().casefold()), []).append(base)
        return {raw: repr((base, group_counts[raw.strip().casefold()],
                           sorted(variants[raw.strip().casefold()], key=repr),
                           [(kind, sorted(span_variants[kind, part.strip().casefold()], key=repr))
                            for part, kind in spans[raw]])).encode("utf-8").hex()
                for raw, base in bases.items()}
    if p.type in date_detect.DATE_TYPES:
        params = _date_params(p)
        view = date_detect.view(params, [v.strip() for v in values])
        return {raw: repr((int(view.sod[i]) if view.kind == "time" else int(view.dur[i]) if view.kind == "duration"
                           else int(view.mon[i]), int(view.cls[i]), str(view.tz[i]), int(view.digits[i]),
                           raw[:len(raw) - len(raw.lstrip())], raw[len(raw.rstrip()):])).encode("utf-8").hex()
                for i, raw in enumerate(values) if view.cls[i] >= 0}
    return {}


def _date_params(p: ColumnProfile) -> dict[str, Any]:
    params = {k: v for k, v in p.params.items() if k not in ("sentinels", "_other_tokens")}
    if not params.get("formats"):
        from ..generate.dates import _params

        params = _params(replace(p, params=deepcopy(params)), p.model)
    return params


# --------------------------------------------------------------------------- ADR-0004: broader held classes
#: The informational code of a column whose rare text shapes or date formats were written in a broader held class
BROADENED_CODE = "sharing_format_broadened"
#: Invented text whose shape is a layout, not personal information: a rare shape may climb the ladder. Names,
#: e-mail addresses, phone numbers, addresses, cities ... keep classes defined by personal information (ADR-0004).
_BROADENED_TEXT_TYPES = frozenset({ColumnType.FREE_TEXT, ColumnType.HIGH_CARD_CATEGORICAL, ColumnType.DIGIT_CODE})


def _pattern(shape: str) -> str:
    """A character mask without length detail: each run of letters of one case, or of digits, becomes one."""
    return re.sub(r"(A)A+|(a)a+|(0)0+", lambda m: m.group(1) or m.group(2) or m.group(3), shape)


def _text_base(p: ColumnProfile, raw: str) -> tuple:
    """The per-value part of :func:`_generation_classes` of a value with no embedded personal data."""
    from ..generate.text import _lead_digit

    return (_shape(raw), int(p.model.nonmissing_counts[raw]), _lead_digit(raw.strip()),
            _retained_semantics(p.type.value.lower(), raw), [])


def _template(donor: str, base: tuple, kind: str, taken: set[str], rnd: Any) -> str | None:
    """A made-up value of the donor's class ``base`` (shape, leading digit, retained semantics, no personal data)
    whose case-folded text is in no ``taken``; None when none is found."""
    from ..generate.embedded import pii_spans
    from ..generate.text import _lead_digit

    zeros = len(donor.strip()) - len(donor.strip().lstrip("0"))
    start = len(donor) - len(donor.lstrip())
    for _ in range(200):
        out = []
        for i, ch in enumerate(donor):
            if "0" <= ch <= "9":
                out.append("0" if i - start < zeros else rnd.choice("0123456789"))
            elif "A" <= ch <= "Z":
                out.append(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))
            elif "a" <= ch <= "z":
                out.append(rnd.choice("abcdefghjkmnpqrstuvwxyz"))
            else:
                out.append(ch)
        text = "".join(out)
        if (text.strip().casefold() not in taken and not pii_spans(text, digit_phone=True)
                and (_shape(text), _lead_digit(text.strip()), _retained_semantics(kind, text)) == (base[0], base[2], base[3])):
            return text
    return None


def _remap(model: ColumnModel, chosen: dict[Any, str], groups: dict[str, list[tuple[Any, int]]],
           exact: bool) -> ColumnModel:
    """``model`` with each key of ``chosen`` replaced by the new values of its group: with their own counts when
    ``exact``, else sharing out the replaced rows in proportion to those counts."""
    nm: dict[Any, int] = {}
    rows: Counter = Counter()
    for raw, count in model.nonmissing_counts.items():
        if raw in chosen:
            rows[chosen[raw]] += int(count)
        else:
            nm[raw] = nm.get(raw, 0) + int(count)
    for key in sorted(rows):
        values = groups.get(key) or []
        shares = [c for _, c in values] if exact else largest_remainder([c for _, c in values], rows[key]) if values else []
        for (value, _), share in zip(values, shares):
            if share:
                nm[value] = nm.get(value, 0) + int(share)
    return replace(model, nonmissing_counts=nm)


def _broaden_text(p: ColumnProfile, source: ColumnProfile, rare: list[str], held: dict[str, int],
                  signatures: dict[Any, str], k: int) -> tuple[dict[Any, str], dict[str, list[tuple[str, int]]]] | None:
    """ADR-0004 ladder for rare invented text shapes: (rare value -> its class, class -> made-up values and counts).

    Each rare value moves to the first held class on the ladder: its exact class (not held, or it would not be
    rare), then the held classes of the same pattern without leading-digit and length detail, then the column's
    most common held class. The made-up values are of a held class exactly (shape, leading digit, retained
    semantics and frequency), so the generator writes them in that class. None when a rare value cannot be
    broadened: its class is defined by personal information inside it, or no held class exists.
    """
    import random

    from ..generate.embedded import pii_spans

    if p.type not in _BROADENED_TEXT_TYPES:
        return None
    nm = p.model.nonmissing_counts
    groups = Counter(raw.strip().casefold() for raw in nm if isinstance(raw, str))

    def plain(raw: str) -> bool:  # one spelling and no personal data inside it
        return groups[raw.strip().casefold()] == 1 and not pii_spans(raw, digit_phone=True)

    if any(pii_spans(raw, digit_phone=True) for raw in rare):
        return None  # a class defined by personal data inside the value is never broadened
    # ADR-0005: skew rather than refuse. A rare spelling moves with every other spelling of its value (the
    # spellings' class is their set), each spelling taking a held class of its own.
    moving = {raw.strip().casefold() for raw in rare}
    rare = [raw for raw in nm if isinstance(raw, str) and raw.strip().casefold() in moving]
    if any(pii_spans(raw, digit_phone=True) for raw in rare):
        return None
    targets: dict[str, tuple[int, str, tuple]] = {}  # repr(class) -> (holders, a donor value, class)
    for raw, signature in signatures.items():
        if raw in nm and int(held.get(signature, 0)) >= k and plain(raw):
            base = _text_base(source, raw)
            targets.setdefault(repr(base), (int(held[signature]), raw, base))
    if not targets:
        return None
    ranked = sorted(targets, key=lambda key: (-targets[key][0], key))

    def rung(raw: str) -> str:  # same pattern without leading-digit and length detail, else most common held
        pattern = _pattern(_shape(raw))
        return next((key for key in ranked if _pattern(targets[key][2][0]) == pattern), ranked[0])

    chosen = {raw: rung(raw) for raw in rare}
    rows: Counter = Counter()
    for raw, key in chosen.items():
        rows[key] += int(nm[raw])
    taken = {raw.strip().casefold() for raw in (*nm, *source.model.nonmissing_counts) if isinstance(raw, str)}
    rnd = random.Random(f"sharing-broaden:{p.name}:{p.index}")
    made: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for key in sorted(rows):
        _, donor, base = targets[key]
        count = max(1, int(base[1]))  # each made-up value keeps the held class's frequency
        made[key] = []
        for _ in range(int(round(rows[key] / count))):
            value = _template(donor, base, p.type.value.lower(), taken, rnd)
            if value is None:
                break  # ADR-0005: the remaining rows go to the held values rather than refuse the column
            taken.add(value.strip().casefold())
            made[key].append(value)
            counts[value] = count
    # rows too few for one more value of a class's frequency go to the held values when the column is allocated
    return chosen, {key: [(value, counts[value]) for value in values] for key, values in made.items()}


def _broaden_dates(p: ColumnProfile, rare: list[str], held: dict[str, int], signatures: dict[Any, str],
                   k: int) -> tuple[dict[Any, str], dict[str, list[tuple[str, int]]]] | None:
    """ADR-0004 for dates: a value in a rare date format is written in the column's most common held format.

    A format is held when a held class uses it. A rare value whose format is held (it is rare for another reason,
    such as its month or offset), or that cannot be written in the held format keeping its date, time and offset,
    goes to the most common held value of that format instead (ADR-0005: skew a distribution rather than refuse).
    None when no held format exists.
    """
    import numpy as np

    from ..generate import dates as gen_dates

    params = _date_params(p)
    values = [v for v in p.model.nonmissing_counts if isinstance(v, str)]
    view = date_detect.view(params, [v.strip() for v in values])
    at = {raw: i for i, raw in enumerate(values)}

    def layout(i: int) -> tuple[int, int]:
        return int(view.cls[i]), int(view.digits[i])

    rows: Counter = Counter()
    for raw, signature in signatures.items():
        if raw in at and int(held.get(signature, 0)) >= k:
            rows[layout(at[raw])] += int(p.model.nonmissing_counts[raw])
    if not rows:
        return None
    target = min(rows, key=lambda f: (-rows[f], f))
    # ADR-0005: a value that cannot be written in the held format keeping its date goes to the most common held
    # value of that format instead (a skewed distribution, never a refused column)
    fallback = max((raw for raw, signature in signatures.items() if raw in at and int(held.get(signature, 0)) >= k
                    and layout(at[raw]) == target), key=lambda raw: (int(p.model.nonmissing_counts[raw]), raw))
    kind = view.kind
    profile = replace(p, params=params)

    def component(v: Any, i: int) -> int:
        return int(v.sod[i]) if kind == "time" else int(v.dur[i]) if kind == "duration" else int(v.mon[i])

    chosen: dict[Any, str] = {}
    groups: dict[str, list[tuple[str, int]]] = {}
    for raw in rare:
        i = at.get(raw)
        if i is None:
            return None
        if layout(i) in rows:  # rare for another reason (its month, its offset): ADR-0005 skews, not refuses
            chosen[raw] = fallback
            groups[fallback] = [(fallback, 1)]
            continue
        cell = gen_dates._empty(1)
        cell["fmt"] = target[0]
        cell["day"] = int(view.day[i])
        cell["us"] = int(view.dur[i]) if kind == "duration" else int(view.sod[i])
        cell["tz"] = np.asarray([str(view.tz[i] or "")], dtype=object)
        cell["fd"] = target[1] if target[1] >= 0 else -1
        cell["lead"] = np.asarray([raw[:len(raw) - len(raw.lstrip())]], dtype=object)
        cell["trail"] = np.asarray([raw[len(raw.rstrip()):]], dtype=object)
        written = str(gen_dates.render_dates(profile, cell)[0])
        check = date_detect.view(params, [written.strip()])
        # the value must read back in the held format with the same date, time and offset
        if ((int(check.cls[0]), int(check.digits[0])) != target or component(check, 0) != component(view, i)
                or str(check.tz[0] or "") != str(view.tz[i] or "")
                or kind in ("date", "datetime") and int(check.day[0]) != int(view.day[i])
                or kind == "datetime" and int(check.sod[0]) != int(view.sod[i])):
            written = fallback
        chosen[raw] = written
        groups[written] = [(written, 1)]
    return chosen, groups


def _broaden(p: ColumnProfile, source: ColumnProfile, private: Any, held: dict[str, int],
             signatures: dict[Any, str], k: int) -> int:
    """Move the rare text shapes or date formats of ``p`` into broader held classes (ADR-0004), in its generation
    model and its per-entity model. Returns how many source values moved (0: none could, the column refuses)."""
    rare = [raw for raw, signature in signatures.items()
            if raw in p.model.nonmissing_counts and int(held.get(signature, 0)) < k]
    if not rare:
        return 0
    found = (_broaden_text(p, source, rare, held, signatures, k) if _invented(p)
             else _broaden_dates(p, rare, held, signatures, k) if p.type in date_detect.DATE_TYPES else None)
    if found is None:
        return 0
    chosen, groups = found
    before = p.model
    p.model = _remap(p.model, chosen, groups, exact=_invented(p))
    if p.type in date_detect.DATE_TYPES and p.params.get("formats"):
        # a format no value is written in any more is not named in the shared schema or data dictionary either
        def used(model: ColumnModel) -> set[int]:
            texts = [v.strip() for v in model.nonmissing_counts if isinstance(v, str)]
            return {int(c) for c in date_detect.view(_date_params(p), texts).cls.tolist() if c >= 0}

        gone = used(before) - used(p.model)
        p.params["formats"] = [f for i, f in enumerate(p.params["formats"]) if i not in gone]
        if p.type is ColumnType.DATE_MIXED_FORMATS and len(p.params["formats"]) == 1:
            p.type = ColumnType.DATE  # nor does the type say that another format existed
    if p.index in private.entity_models:
        private.entity_models[p.index] = _remap(private.entity_models[p.index], chosen, groups, exact=False)
    p.flags["sharing_format_broadened"] = True
    return len(rare)


def _check_embedded_relationships(columns: list[ColumnProfile]) -> None:
    """Repeated span identities can preserve an identifiable co-occurrence graph.

    Marginal frequencies and even whole context multisets cannot prove a graph
    has multiple possible donors. Sharing currently refuses those contracts.
    """
    from ..generate.embedded import pii_spans

    seen: dict[tuple[str, str], tuple[int, str, bool]] = {}
    for p in columns:
        if p.role in ("excluded", "null", "id") or p.model is None or not _invented(p):
            continue
        for raw in p.model.nonmissing_counts:
            if not isinstance(raw, str):
                continue
            for a, b, kind in pii_spans(raw, digit_phone=True):
                part = raw[a:b]
                embedded = bool(raw[:a].strip() or raw[b:].strip())
                key = kind, part.strip().casefold()
                previous = seen.get(key)
                if previous and (previous[0], previous[1]) != (p.index, raw) and (embedded or previous[2]):
                    _refuse(p, "has a repeated embedded PII relationship whose source graph cannot be safely retained",
                            "the same e-mail address, phone number or name appears in more than one place, which could link records", "sharing_repeated_embedded")
                seen[key] = (p.index, raw, embedded or bool(previous and previous[2]))


def _units(p: ColumnProfile, raw: Any) -> list[str]:
    if p.model is None:
        return []
    if raw in p.model.missing_counts:
        return ["m:" + _key(raw), _variant(raw)]
    if sharing_numeric.discrete(p) and (key := sharing_numeric.normalized_key(p, raw)) is not None:
        return ["n:" + key, _variant(raw)]
    if not isinstance(raw, str):
        return []
    if _level_column(p):
        return ["u:" + _key(raw), _variant(raw)]
    if p.type is ColumnType.MULTI_SELECT:
        d = p.params.get("delimiter")
        items = [t.strip() for t in raw.split(d) if t.strip()] if d else [raw.strip()]
        return ["u:" + _key(t) for t in dict.fromkeys(items)] + ["c:" + repr(tuple(sorted({_key(t) for t in items}))), _variant(raw)]
    if p.type is ColumnType.NUMERIC_WITH_TOKENS:
        tokens = text_detect.unit_texts(p, raw)
        return ["u:" + _key(t) for t in tokens] + ([_variant(raw)] if tokens else [])
    if p.type in date_detect.DATE_TYPES and raw.strip() in _date_tokens(p):
        return ["u:" + _key(raw), _variant(raw)]
    return []


def record_evidence(tp: TableProfile, raw: RawTable, settings: Any) -> None:
    """Capture exact local distinct-person counts after profiling, before dropping raw rows.

    All units count, including standard vocabulary and missing spellings. Blank
    person keys share one unknown holder through d35_keys, never one per row.
    This metadata is private and does not change source stats or public params.
    """
    keys = entity.d35_keys(raw, tp.columns, tp.entity, settings)
    counts: dict[int, dict[str, int]] = {}
    classes: dict[int, dict[str, int]] = {}
    numbers: dict[int, dict[str, int]] = {}
    date_buckets: dict[int, dict[str, int]] = {}
    for p in tp.columns:
        if p.model is None or p.role in ("excluded", "null", "id"):
            continue
        if p.model.n_nonmissing and _other_person_attribute(p):
            # These keys count the row's subject, not the attribute's actor.
            # Do not store those counts as evidence for another person's data.
            continue
        values = raw.columns[p.index].values
        # unit_holders ignores native None; an internal marker lets missing
        # native nulls participate in the same per-person counting operation.
        nullable = any(v is None for v in p.model.missing_counts)
        counted_values = np.asarray(["\x00native-null" if v is None else v for v in values], dtype=object) if nullable else values

        def units(v: Any, p: ColumnProfile = p, nullable: bool = nullable) -> list[str]:
            return _units(p, None if nullable and v == "\x00native-null" else v)

        signatures = _generation_classes(p)

        def signature(v: Any) -> list[str]:
            return [signatures[v]] if v in signatures else []

        numeric_keys = ({v: sharing_numeric.normalized_key(p, v)
                         for v in (*p.model.nonmissing_counts, *p.model.missing_counts)}
                        if p.type in sharing_numeric.TYPES else {})

        def numeric(v: Any) -> list[str]:
            key = numeric_keys.get(v)
            return ["n:" + key] if key is not None else []

        bucket_keys = {}
        if p.type in date_detect.DATE_TYPES - {ColumnType.TIME, ColumnType.DURATION}:
            from ..generate.sharing_dates import calendar_bucket_keys

            bucket_values = [*p.model.nonmissing_counts, *p.model.missing_counts]
            bucket_keys = dict(zip(bucket_values, calendar_bucket_keys(p, bucket_values)))

        def date_bucket(v: Any) -> list[str]:
            key = bucket_keys.get(v)
            return [key] if key is not None else []

        for destination, extract in ((counts, units), (classes, signature), (numbers, numeric), (date_buckets, date_bucket)):
            held = text_detect.unit_holders(counted_values, p, texts=extract)
            if not held:
                continue
            for index in keys.keys:
                got = text_detect.unit_holders(counted_values, p, index, extract)
                held = {k: min(c, got.get(k, 0)) for k, c in held.items()}
            destination[p.index] = held
    one_per_person = bool(keys.people and keys.keys and all(len(np.unique(k)) == raw.n_rows for k in keys.keys))
    tp.private.extra[EVIDENCE_KEY] = {"people": bool(keys.people), "counts": counts, "classes": classes, "numbers": numbers,
                                      "date_buckets": date_buckets if keys.people else {},
                                      "one_row_per_person": one_per_person}


def _clone_model(model: ColumnModel | None) -> ColumnModel | None:
    if model is None:
        return None
    extra = dict(model.extra)
    extra[SOURCE_MODEL_KEY] = model.extra.get(SOURCE_MODEL_KEY, model)
    # These source-specific allocations must not restore suppressed levels or
    # impose source donor counts on the fictional population.
    for key in ("_row_counts", "_row_bins", "_generation_model", "_pseudonym_inverse", "floored_levels", "floored_extremes"):
        extra.pop(key, None)
    return replace(model, nonmissing_counts=dict(model.nonmissing_counts), missing_counts=dict(model.missing_counts), extra=extra)


#: Plain reasons (ADR-0001: the report says why a column was nulled)
_NO_PEOPLE = "no column says which person each row belongs to, so sharing mode cannot count how many people share its values"
_TIED = "it is tied to another column by a key, dependency, template or rule that cannot be regenerated safely"


def _refuse(p: ColumnProfile, reason: str, plain: str, code: str) -> None:
    raise ColumnRefusal(f"sharing mode: column {p.name!r} {reason}",
                        fix=f"--null {p.name} or --exclude {p.name}; use --person-id COL when a column identifies the row's person",
                        code=code, column=p.name, index=p.index, plain=plain)


def _rescale(counts: dict[Any, int], n: int) -> dict[Any, int]:
    if not counts:
        return {}
    allocated = largest_remainder(list(counts.values()), n)
    return {k: int(c) for k, c in zip(counts, allocated) if c}


def _protect_model(p: ColumnProfile, model: ColumnModel, rare: set[str]) -> ColumnModel:
    nm: Counter = Counter()
    missing = {v: c for v, c in model.missing_counts.items() if "m:" + _key(v) not in rare and _variant(v) not in rare}
    for raw, count in model.nonmissing_counts.items():
        units = _units(p, raw)
        if any(u in rare for u in units):
            continue
        nm[raw] += int(count)
    if not nm and not missing and model.n_nonmissing + model.n_missing:
        _refuse(p, "has no category or missing value supported by enough distinct people in this table",
                "too few people share each of its values to keep any of them", "sharing_no_support")
    total = model.n_nonmissing + model.n_missing
    # Suppressed units get zero rows. Reassign their mass within the same
    # missing/nonmissing group, or to the only remaining supported group.
    nm_target = model.n_nonmissing if missing else total
    missing_target = model.n_missing if nm else total
    return replace(model, nonmissing_counts=_rescale(dict(nm), nm_target),
                   missing_counts=_rescale(missing, missing_target))


def _related_columns(tp: TableProfile, *, composite: bool = True) -> set[int]:
    """Columns whose source relationship could restore suppressed units."""
    names = {p.name: p.index for p in tp.columns}
    involved: set[int] = set()
    for fd in tp.structure.private.get("fds", ()):
        involved.update(int(fd[k]) for k in ("det", "dep") if k in fd)
    if composite:
        for key in tp.structure.private.get("composite", ()):
            involved.update(int(i) for i in key.get("cols", ()))
    for entry in tp.structure.derived_templates:
        if entry.get("column") in names:
            involved.add(names[entry["column"]])
        for name in entry.get("inputs", ()):
            if name in names:
                involved.add(names[name])
    for entry in tp.structure.private.get("templates", ()):
        involved.add(int(entry["target"]))
        involved.update(int(part[1]) for part in entry.get("parts", ()) if part[0] == "f")
    for target, entry in tp.structure.private.get("rule_targets", {}).items():
        involved.add(int(target))
        involved.update(int(member[0]) for member in entry.get("members", ()))
    # Declared inequalities may regenerate one side from source residuals.
    for entry in tp.structure.declared_rules:
        for value in entry.values():
            if isinstance(value, str) and value in names:
                involved.add(names[value])
    return involved


def _independent_missingness(columns: list[ColumnProfile], n: int) -> MissingnessModel:
    rows = [p for p in columns if p.role == "row" and p.model is not None]
    model = MissingnessModel(columns=[p.index for p in rows])
    counts = [p.model.n_missing for p in rows]
    cuts = sorted({0, n, *counts})
    # The pool sampler independently draws each column's marginal. These
    # artificial interval patterns encode totals, never source row patterns.
    model.pooled = [(tuple(c > a for c in counts), b - a) for a, b in zip(cuts, cuts[1:]) if b > a]
    return model


def _trim_native(p: ColumnProfile, rare: set[str]) -> None:
    if not p.native:
        return
    removed = {u[2:] for u in rare}
    source = p.model.extra.get(SOURCE_COLUMN_KEY, p)

    def kept(value: Any) -> bool:
        return _key(value) not in removed and not set(_units(source, value)).intersection(rare)

    native = p.native
    labels = native.get("labels")
    if isinstance(labels, dict):
        native["labels"] = {k: v for k, v in labels.items() if kept(v)}
    values = native.get("dictionary_values")
    if isinstance(values, list):
        keep = [i for i, v in enumerate(values) if kept(v)]
        native["dictionary_values"] = [values[i] for i in keep]
        dictionary = native.get("dictionary")
        if dictionary is not None:
            native["dictionary"] = dictionary.take(np.asarray(keep, dtype=np.int64))


def _native_units(p: ColumnProfile) -> set[str]:
    if not _level_column(p) and p.type is not ColumnType.MULTI_SELECT:
        return set()  # non-label dictionaries are rebuilt from synthetic values by the writers
    native = p.native or {}
    values = list((native.get("labels") or {}).values()) + list(native.get("dictionary_values") or ())
    return {u for v in values if v is not None for u in (_units(p, v) or ["u:" + _key(v)])}


def _check_native_storage(p: ColumnProfile) -> None:
    """Exact Excel format quotas have no local-person or joint-format proof."""
    native = p.native or {}
    if native.get("kind") != "excel":
        return
    storage = native.get("storage")
    if isinstance(storage, dict):
        classes = {(kind, fmt) for kind, formats in storage.items()
                   for fmt, count in formats.items() if count > 0}
    else:
        formats = native.get("formats") or {}
        classes = {(kind, formats.get(kind) or "General") for kind, count in (native.get("types") or {}).items()
                   if count > 0}
    if len(classes) > 1:
        _refuse(p, "has mixed Excel cell types or number formats without distinct-person support for their retained quotas",
                "its cells mix Excel cell types or number formats that too few people share", "sharing_mixed_cell_types")


def _trim_params(p: ColumnProfile, rare: set[str]) -> None:
    removed = {u[2:] for u in rare}
    source = p.model.extra.get(SOURCE_COLUMN_KEY, p)

    def kept(value: Any) -> bool:
        return _key(value) not in removed and not set(_units(source, value)).intersection(rare)

    for key in ("levels", "level_order", "tokens", "_tokens", "_other_tokens", "pair"):
        values = p.params.get(key)
        if isinstance(values, (list, tuple)):
            p.params[key] = [v for v in values if kept(v)]
    for key in ("sentinels", "variant_groups"):
        values = p.params.get(key)
        if isinstance(values, dict):
            p.params[key] = {v: c for v, c in values.items() if kept(v)}
    for key in ("value", "token"):
        if key in p.params and not kept(p.params[key]):
            p.params.pop(key)


def _structural_contracts(tp: TableProfile, dates: set[int]) -> None:
    sensitive = {p.index for p in tp.columns if p.role == "id" or _invented(p)} | dates
    for fd in tp.structure.private.get("fds", ()):
        if not fd.get("overridden") and sensitive.intersection((fd["det"], fd["dep"])):
            _refuse(tp.columns[fd["dep"]], "has a source person dependency that cannot preserve fictional relationships", _TIED, "sharing_person_dependency")
    # A template's source grouping carries the original distinct/frequency
    # relationship even after its input labels become fictional.
    for template in tp.structure.private.get("templates", ()):
        if template.get("groups") or sensitive.intersection(
                {template["target"]} | {part[1] for part in template.get("parts", ()) if part[0] == "f"}):
            _refuse(tp.columns[template["target"]], "has a source template relationship that cannot be safely regenerated", _TIED, "sharing_source_template")


def verify_values(profile: ColumnProfile, values: Any) -> list[Message]:
    """Check final rendered/re-read values after every structural and privacy repair."""
    if profile.model is None or profile.role in ("null", "excluded"):
        return []
    source = profile.model.extra.get(SOURCE_COLUMN_KEY) or replace(
        profile, model=profile.model.extra.get(SOURCE_MODEL_KEY, profile.model))
    suppressed = set(profile.model.extra.get(SUPPRESSED_KEY, ()))
    raw = list(values)
    # Rereaders may normalize the case/spacing of a missing token. Its source
    # spelling need not still occur in the protected missing-count dictionary.
    hits = sum(bool((set(_units(source, value)) | {"u:" + _key(value), "m:" + _key(value), _variant(value)}) & suppressed)
               for value in raw) if suppressed else 0
    messages = []
    if hits:
        messages.append(Message("error", "sharing_rare_unit_restored",
                                f"{hits} generated value(s) restored a locally suppressed category or missing value",
                                column=profile.name, fix="null or exclude this column"))
    if profile.params.get(PERSON_DATE_KEY):
        from ..generate.sharing_dates import cover_violations, normalized_keys, source_extremes

        forbidden = source_extremes(profile)
        dates = sum(k in forbidden for k in normalized_keys(profile, raw) if k is not None)
        if dates:
            messages.append(Message("error", "sharing_date_extreme_restored",
                                    f"{dates} generated value(s) restored an original normalized person-date extreme",
                                    column=profile.name, fix="null or exclude this column"))
        restored_cover = cover_violations(profile, raw)
        if restored_cover:
            messages.append(Message("error", "sharing_date_cover_restored",
                                    f"{restored_cover} generated value(s) restored a masked calendar group",
                                    column=profile.name, fix="null or exclude this column"))
    if profile.params.get(sharing_numeric.MARKER):
        forbidden = sharing_numeric.forbidden_extremes(profile)
        hits = sum(sharing_numeric.normalized_key(source, value) in forbidden for value in raw)
        if hits:
            messages.append(Message("error", "sharing_numeric_extreme_restored",
                                    f"{hits} generated value(s) restored an original normalized numeric extreme",
                                    column=profile.name, fix="null or exclude this column"))
    return messages


def protect_profile(tp: TableProfile, settings: Any) -> TableProfile:
    """Clone a source profile and protect the generation models, preserving source evidence.

    Caller must capture record_evidence while raw rows exist. A planner-proven
    lookup may set LOOKUP_KEY; this never supplies holders to a fact table.
    """
    evidence = tp.private.extra.get(EVIDENCE_KEY) or {}
    lookup = tp.private.extra.get(LOOKUP_KEY) is True
    k = max(10, int(getattr(settings, "min_cell", 10)))
    columns = [replace(p, params=deepcopy(p.params), flags=dict(p.flags), messages=list(p.messages),
                       model=_clone_model(p.model), native=deepcopy(p.native)) for p in tp.columns]
    private = replace(tp.private, extra=dict(tp.private.extra), entity_models={i: _clone_model(m) for i, m in tp.private.entity_models.items()})
    structure = deepcopy(tp.structure)
    out = replace(tp, header=list(tp.header), entity=deepcopy(tp.entity), layout=deepcopy(tp.layout),
                  columns=columns, private=private, structure=structure, messages=list(tp.messages))
    related = _related_columns(tp)
    value_contracts = _related_columns(tp, composite=False)
    changed: set[int] = set()
    person_dates: set[int] = set()
    _check_embedded_relationships(columns)
    for p in columns:
        if p.role in ("excluded", "null"):
            continue
        _check_native_storage(p)
        if p.model is None or p.role == "id":
            continue
        if p.model.n_nonmissing and _other_person_attribute(p):
            raise ColumnRefusal(
                f"sharing mode: column {p.name!r} contains an intrinsic attribute of another person; "
                "distinct-person support for that actor has not been established",
                fix=f"--null {p.name} or --exclude {p.name}", code="sharing_other_person_attribute", column=p.name, index=p.index,
                plain="it describes another person (such as a parent or teacher), and sharing mode cannot count "
                      "how many of those people share each value",
            )
        source = tp.columns[p.index]
        p.model.extra[SOURCE_COLUMN_KEY] = source
        requested = {u for raw in (*source.model.nonmissing_counts, *source.model.missing_counts) for u in _units(source, raw)} | _native_units(source)
        held = (evidence.get("counts") or {}).get(p.index, {})
        if requested and not lookup and not evidence.get("people"):
            _refuse(p, "cannot establish distinct-person support for its categories or missing values", _NO_PEOPLE, "sharing_no_people")
        rare = {u for u in requested if int(held.get(u, 0)) < k} if not lookup else set()
        if rare:
            if p.index in related:
                _refuse(p, "has rare units in a structural key, dependency or rule that cannot be safely regenerated", _TIED, "sharing_rare_in_structure")
            p.model = _protect_model(source, p.model, rare)
            if p.index in private.entity_models:
                private.entity_models[p.index] = _protect_model(source, private.entity_models[p.index], rare)
            p.model.extra[SUPPRESSED_KEY] = frozenset(rare)
            _trim_native(p, rare)
            _trim_params(p, rare)
            if p.type is ColumnType.MULTI_SELECT:
                # Sampling items independently could rebuild an eliminated
                # source combination (including another ordering of its items).
                p.params["mode"] = "combinations"
            changed.add(p.index)
            p.flags["sharing_rare_units_suppressed"] = True
            if p.type in (ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL) and any(u.startswith("n:") for u in rare):
                # ADR-0002: same suppression as plain discrete numbers; generated by quota over the kept values, in the column's format
                p.params[sharing_numeric.LEVELS_KEY] = True
        if not lookup and p.type in sharing_numeric.TYPES and p.model.n_nonmissing:
            if not evidence.get("people"):
                _refuse(p, "cannot establish distinct-person support for its numeric measures", _NO_PEOPLE, "sharing_no_people")
            if p.index in related:
                _refuse(p, "has numeric extrema in a structural contract that cannot be safely regenerated", _TIED, "sharing_numeric_structure")
            number_holders = (evidence.get("numbers") or {}).get(p.index, {})
            forbidden = frozenset(key for key in sharing_numeric.source_extremes(source)
                                  if int(number_holders.get("n:" + key, 0)) < k)
            p.params[sharing_numeric.MARKER] = True
            if forbidden:
                p.params["_no_exact_extremes"] = True
            p.model.extra[sharing_numeric.FORBIDDEN_KEY] = forbidden
            p.model = sharing_numeric.protect_model(source, p.model, forbidden)
            if p.index in private.entity_models:
                private.entity_models[p.index].extra[sharing_numeric.FORBIDDEN_KEY] = forbidden
                private.entity_models[p.index] = sharing_numeric.protect_model(source, private.entity_models[p.index], forbidden)
            p.flags["sharing_numeric_extremes_removed"] = bool(forbidden)
            changed.add(p.index)
        person_date = (p.type in date_detect.DATE_TYPES and p.type not in (ColumnType.TIME, ColumnType.DURATION)
                       and (p.params.get("person_level") or header_hints(p.name).pii == "dob"
                            or p.role == "entity" or evidence.get("people")))
        if person_date:
            from ..generate.sharing_dates import BUCKET_HOLDERS_KEY, BUCKET_MIN_KEY, validate_bucket_support

            bucket_holders = dict((evidence.get("date_buckets") or {}).get(p.index, {}))
            p.model.extra[BUCKET_HOLDERS_KEY] = bucket_holders
            p.model.extra[BUCKET_MIN_KEY] = k
            if p.index in private.entity_models:
                private.entity_models[p.index].extra[BUCKET_HOLDERS_KEY] = bucket_holders
                private.entity_models[p.index].extra[BUCKET_MIN_KEY] = k
            validate_bucket_support(p)
            person_dates.add(p.index)
            p.params[PERSON_DATE_KEY] = True
            p.params["_no_exact_extremes"] = True
            if p.index in value_contracts:
                _refuse(p, "has a protected person date in a structural contract that cannot be safely regenerated", _TIED, "sharing_date_structure")
            from ..generate.sharing_dates import normalized_keys, source_extremes

            missing = list(source.model.missing_counts)
            extrema = source_extremes(source)
            date_missing = {"m:" + _key(v) for v, key in zip(missing, normalized_keys(source, missing)) if key in extrema}
            if date_missing:
                p.model = _protect_model(source, p.model, date_missing)
                p.model.extra[SUPPRESSED_KEY] = frozenset(set(p.model.extra.get(SUPPRESSED_KEY, ())) | date_missing)
                if p.index in private.entity_models:
                    private.entity_models[p.index] = _protect_model(source, private.entity_models[p.index], date_missing)
                changed.add(p.index)
        if not lookup and p.model.n_nonmissing and (_invented(p) or p.type in date_detect.DATE_TYPES):
            if not evidence.get("people"):
                _refuse(p, "cannot establish distinct-person support for its retained date, text or PII classes", _NO_PEOPLE, "sharing_no_people")
            signatures = _generation_classes(source)
            held_classes = (evidence.get("classes") or {}).get(p.index, {})
            moved = _broaden(p, source, private, held_classes, signatures, k)
            if moved:
                # every class the generator now writes must be held by itself, as before
                signatures = _generation_classes(replace(source, model=p.model, params=p.params))
                out.messages.append(Message(
                    "info", BROADENED_CODE,
                    f"{moved} value(s) of column {p.name!r} had a "
                    + ("date format, month or offset" if p.type in date_detect.DATE_TYPES else "layout")
                    + " too few people share, so they were written in "
                    + ("the column's most common held date format" if p.type in date_detect.DATE_TYPES
                       else "a broader layout that enough people share"),
                    column=p.name))
            if any(int(held_classes.get(signature, 0)) < k for raw, signature in signatures.items()
                   if raw in p.model.nonmissing_counts):
                _refuse(p, "has a date, text or PII format/frequency class supported by fewer than the required distinct people",
                        "some of its value layouts are shared by too few people", "sharing_rare_class")
    _structural_contracts(out, person_dates)
    # Source joint missingness can identify one pattern donor even when every
    # marginal is common. Only independent protected marginals survive.
    structure.missingness = _independent_missingness(columns, tp.n_rows)
    if changed:
        private.extra["_sharing_suppressed_columns"] = frozenset(changed)
    private.extra["_sharing_source_profile"] = tp
    return out


#: The warning a column nulled by sharing mode carries (utility: it releases nothing real, ADR-0001)
AUTO_NULL_CODE = "column_nulled_automatically"
#: ``ColumnProfile.source`` of a column sharing mode nulled (``--null`` sets ``null``)
AUTO_NULL_SOURCE = "auto-null"
#: ``ColumnProfile.params`` key (PRIVATE) of what the operator can do instead of an automatic null
AUTO_NULL_ALTERNATIVE_KEY = "_null_alternative"


def _key_or_related(tp: TableProfile, index: int) -> bool:
    """A key (ID column, alias, nesting, a column holding another ID column's keys) or a column in a structural
    relationship (join, dependency, template, composite key, declared rule): nulling it would break what the
    synthetic copy exists to keep, so its refusal still leaves the file out (ADR-0001)."""
    from .entity import FLAGGED_ID_KEY

    p = tp.columns[index]
    ent = tp.entity
    if p.role == "id" or index in ent.id_columns or index == ent.primary:
        return True
    if any(index in group for group in ent.aliases):
        return True
    if any(p.name in (n.get("child"), n.get("parent")) for n in ent.nesting):
        return True
    if any(p.params.get(key) for key in ("_key_ref", "_key_ref_list", FLAGGED_ID_KEY)):
        return True
    return index in _related_columns(tp)


def null_column(tp: TableProfile, index: int, reason: str, *, automatic: bool = True) -> ColumnProfile:
    """Turn column ``index`` of ``tp`` into a Nulled column, the way ``--null`` does: header and position kept,
    every cell written missing, and its real values recorded as dropped (``privacy.texts.DROPPED_KEY``) so no
    generator writes one and the leak scan looks for them. ``reason`` is the plain reason; ``automatic`` says
    sharing mode (not the operator) nulled it, which the column records and reports as a utility warning.
    Returns the new column profile (it replaces the old one in ``tp.columns``)."""
    from ..hints import DEFAULT_NA_TOKENS
    from ..privacy import texts as real_texts

    old = tp.columns[index]
    values = [*old.model.nonmissing_counts, *old.model.missing_counts] if old.model is not None else []
    dropped = {v.strip() for v in values if isinstance(v, str)} - {""} - set(DEFAULT_NA_TOKENS)
    extra = tp.private.extra
    extra[real_texts.DROPPED_KEY] = {**(extra.get(real_texts.DROPPED_KEY) or {}), index: frozenset(dropped)}
    if old.type is ColumnType.PHONE or header_hints(old.name).pii == "phone":
        extra["_dropped_phone_columns"] = frozenset(extra.get("_dropped_phone_columns") or ()) | {index}
    extra.pop("_leak_dropped_digit_terms", None)  # privacy.leaks caches the dropped columns' digit terms
    new = ColumnProfile(index=index, name=old.name, type=ColumnType.EMPTY, role="null", na_tokens=old.na_tokens,
                        source="null", reason=f"--null: written as all-missing because {reason}",
                        model=ColumnModel(), stats={"n": tp.n_rows}, native=old.native)
    new.flags["nulled_automatically"] = automatic
    if automatic:
        # an ID-like column (D7 candidate) may be an ID the operator can declare instead (ADR-0001)
        mark_auto_nulled(new, reason, _id_alternative(old.name) if old.params.get("id_candidate") else None)
    tp.columns[index] = new
    if old.params.get("id_candidate") and not any(p.params.get("id_candidate") for p in tp.columns):
        # the table's ID-like candidates (D7) were all nulled: none of their real values is written any more
        tp.messages = [m for m in tp.messages if m.code != "id_candidates"]
        tp.messages = [Message("info", "no_id_column", "no ID column matched; each row is its own entity (D7)")
                       if m.code == "no_id_column" else m for m in tp.messages]
    ent = tp.entity
    if index in ent.constant_columns:
        ent.constant_columns = [i for i in ent.constant_columns if i != index]
    ent.constant_shares.pop(index, None)
    tp.private.entity_models.pop(index, None)
    return new


def mark_auto_nulled(p: ColumnProfile, reason: str, alternative: str | None = None) -> None:
    """Label the Nulled column ``p`` as nulled by sharing mode (ADR-0001), not by the operator: its source, its plain
    ``reason``, and the utility warning that reports it. ``alternative`` is what the operator can do instead to keep
    the column (for an ID-like column: declare it as an ID), shown with it in the report."""
    p.source = AUTO_NULL_SOURCE
    p.reason = f"nulled automatically: written as all-missing because {reason}"
    p.flags["nulled_automatically"] = True
    if alternative:
        p.params[AUTO_NULL_ALTERNATIVE_KEY] = alternative
    instead = (alternative + "; " if alternative else "") + f"--exclude {p.name} drops the column instead"
    p.messages.append(Message(
        "warning", AUTO_NULL_CODE,
        f"column {p.name!r} was nulled automatically (written with every cell empty) because {reason}",
        column=p.name, fix=f"{instead}; --refuse-unsupported leaves the file out instead"))


#: findings about whole rows or the whole file: nulling a column never answers them, so they still withhold the file
_ROW_OR_FILE_CODES = ("row_matches_", "leak_scan_incomplete")
#: findings that a non-key column looks like an ID or an ID-like code (``id_candidates`` and ``no_id_column`` name
#: the table's candidate columns, ``params['id_candidate']``; ``possible_sparse_code`` is the column's own)
_ID_LIKE_CODES = ("id_candidates", "no_id_column", "possible_sparse_code")


def _id_alternative(name: str) -> str:
    """What the operator can do instead of nulling the ID-like column ``name``: declare it as an ID."""
    return (f"if it is an ID, declare it with --id-col {name} to get synthetic IDs (--person-id {name} when it is "
            "the ID of the row's own person)")


def _post_generation_reason(name: str, codes: list[str]) -> tuple[str, str | None]:
    """(plain reason, alternative) for a column nulled because of the blocking findings ``codes`` about it."""
    if any(code in _ID_LIKE_CODES for code in codes):
        return ("it looks like an ID or a code number but is not used as a key, so its real values could not be "
                "shared as ordinary numbers or labels", _id_alternative(name))
    from ..messages import lookup

    entry = lookup(codes[0])
    found = f" ({entry.sentence.rstrip('.')})" if entry is not None and "{text}" not in entry.sentence else ""
    return f"a privacy check of the generated file found a problem with it{found}", None


def column_scoped_findings(fr: Any, finding_category: Any) -> dict[int, dict[int, tuple[str, str | None]]] | None:
    """ADR-0001, second trigger: when every blocking finding (``finding_category``) of the generated file ``fr`` is
    about one or more columns that are not keys, not in a structural relationship and not nulled already, the
    columns to null before generating the file again, as {table number: {column index: (plain reason,
    alternative)}}; else None (the file is withheld as before).

    A finding about whole rows or the whole file (``row_matches_*``, an incomplete leak scan, any finding that names
    no column) is not scoped to a column. A finding on a column already nulled is one that nulling did not clear: the
    file is withheld, so the rounds of nulling end."""
    targets: dict[tuple[int, int], list[str]] = {}

    def add(t_i: int, index: int, code: str) -> bool:
        tp = fr.tables[t_i]
        if tp.columns[index].role in ("null", "excluded") or _key_or_related(tp, index):
            return False
        targets.setdefault((t_i, index), []).append(code)
        return True

    def scoped(message: Message) -> bool:
        return not message.code.startswith(_ROW_OR_FILE_CODES)

    def blocking(messages: Any) -> list[Message]:
        return [m for m in messages if finding_category(m) == "blocking"]

    for message in blocking(fr.messages):
        hits = [(t_i, p.index) for t_i, tp in enumerate(fr.tables) for p in tp.columns if p.name == message.column]
        if not scoped(message) or not message.column or len(hits) != 1 or not add(*hits[0], message.code):
            return None
    for t_i, tp in enumerate(fr.tables):
        for message in blocking(tp.messages):
            if not scoped(message):
                return None
            if message.code in ("id_candidates", "no_id_column") and not message.column:
                hits = [p.index for p in tp.columns if p.params.get("id_candidate")]
            else:
                hits = [p.index for p in tp.columns if message.column and p.name == message.column]
                hits = hits if len(hits) == 1 else []
            if not hits or not all(add(t_i, i, message.code) for i in hits):
                return None
        for p in tp.columns:
            for message in blocking(p.messages):
                if not scoped(message) or not add(t_i, p.index, message.code):
                    return None
    if not targets:
        return None
    out: dict[int, dict[int, tuple[str, str | None]]] = {}
    for (t_i, index), codes in sorted(targets.items()):
        out.setdefault(t_i, {})[index] = _post_generation_reason(fr.tables[t_i].columns[index].name, codes)
    return out


def null_unprotectable_columns(tp: TableProfile, settings: Any) -> list[str]:
    """ADR-0001: null every column whose protection refuses (:class:`ColumnRefusal`) and that is not a key or in a
    structural relationship, until the table's profile can be protected. Returns the nulled columns' names.

    A refusal on a key or relationship column, or any other error, stops here: the file is left out when it is
    generated, with today's reason."""
    nulled: list[str] = []
    while True:
        try:
            protect_profile(tp, settings)
        except ColumnRefusal as refusal:
            i = refusal.index
            if not (0 <= i < len(tp.columns)) or tp.columns[i].role in ("null", "excluded") or _key_or_related(tp, i):
                return nulled
            null_column(tp, i, refusal.plain, automatic=True)
            nulled.append(tp.columns[i].name)
        except Exception:  # noqa: BLE001 - generation reports it for the file (D19)
            return nulled
        else:
            return nulled


__all__ = ["record_evidence", "protect_profile", "verify_values", "null_column", "null_unprotectable_columns",
           "mark_auto_nulled", "column_scoped_findings", "AUTO_NULL_CODE", "AUTO_NULL_ALTERNATIVE_KEY", "AUTO_NULL_SOURCE", "EVIDENCE_KEY", "SOURCE_MODEL_KEY", "SUPPRESSED_KEY", "PERSON_DATE_KEY", "LOOKUP_KEY"]
