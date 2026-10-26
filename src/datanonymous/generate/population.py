"""Fictional populations for sharing (D44, D47--D49).

Profiling reads original keys only to extract cardinalities, format domains and
relationship *contracts*. Generation uses numbered fictional slots. No original
key-to-output assignment or original row order is used. The original keys remain
an exclusion set for drawing non-real identifiers.

The first sharing policy refuses rare identifier-format classes rather than
claiming that assigning their statistics to another person hides their unique
donor. Rare primary rows-per-person count classes are rebalanced into held
classes with an exact row total (ADR-0003), or refused when that is impossible.
Secondary degrees are regenerated, not preserved (D48).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import reduce
from math import gcd
from typing import Any

import numpy as np

from ..detect.ids import (char_mask, id_text, integral_float_parts, is_hex_id, luhn_check_digit,
                          normalize_id, thousands_digits)
from ..errors import UnsupportedColumnError
from ..model import Message, TableProfile
from ..privacy import texts as real_texts
from ..profile.entity import entity_keys, shadow_namespaces
from .ids import (IdNamespace, IdRegistry, RealUsers, _conform, _is_digit_rep,
                  _pattern_mapping, allocate_digit_ids, namespace_is_identity)
from .rng import stream

PLAN_KEY = "_sharing_population"
#: the informational message of a table whose rare rows-per-person count classes were rebalanced (ADR-0003)
REBALANCED_CODE = "count_classes_rebalanced"


@dataclass(frozen=True)
class TablePopulation:
    """Only fictional assignments; arrays have no source-person indices."""

    id_values: dict[int, np.ndarray]
    entity_index: np.ndarray | None
    n_entities: int
    slots: dict[int, np.ndarray]


@dataclass(frozen=True)
class _Shape:
    kind: str
    template: str = ""
    width: int = 0
    lead: str = ""
    trail: str = ""

    def render(self, values: np.ndarray) -> np.ndarray:
        def one(value: Any) -> str:
            if self.kind == "integer":
                body = str(int(value)).zfill(self.width)
            elif self.kind == "float":
                body = str(int(value)) + self.template
            elif self.kind == "grouped":
                body = f"{int(value):,}"
            else:
                body = _conform(str(value), self.template)
                if body is None:
                    _refuse("an identifier cannot be rendered in every referring column's format")
            return self.lead + body + self.trail

        return np.asarray([one(v) for v in values], dtype=object)


@dataclass
class _Table:
    key: tuple[str, int]
    profile: TableProfile
    namespaces: dict[int, str]
    shapes: dict[int, _Shape]
    cardinalities: dict[int, int]
    groups: list[list[int]]
    counts: np.ndarray
    unique_pairs: list[tuple[int, int]]
    #: people in the table minus people in the source, when rare count classes were rebalanced (else None)
    rebalanced: int | None = None


class PopulationRefusal(UnsupportedColumnError):
    """A refusal traced to particular tables; the other tables can be planned again without them."""

    def __init__(self, refused: dict[tuple[str, int], UnsupportedColumnError]) -> None:
        first = next(iter(refused.values()))
        super().__init__(first.message, first.fix)
        self.refused = refused


def _refuse(reason: str) -> None:
    raise UnsupportedColumnError(
        "sharing population: " + reason,
        "Use a sufficiently large reviewed population, simplify the key contracts, or exclude the affected "
        "table from this sharing run. Use fidelity mode only for independently invented data.")


@contextmanager
def _blame(key: tuple[str, int]) -> Iterator[None]:
    """Attribute a refusal raised for one table to it (a nested attribution to another table stands)."""
    try:
        yield
    except PopulationRefusal:
        raise
    except UnsupportedColumnError as error:
        raise PopulationRefusal({key: error}) from error


def _shape(raw: np.ndarray, minimum: int) -> _Shape:
    forms: dict[_Shape, set[str]] = defaultdict(set)
    strings = [v for v in set(np.asarray(raw, dtype=object).tolist()) if isinstance(v, str) and id_text(v)]
    padded = any(v.strip().isascii() and v.strip().isdigit() and len(v.strip()) > 1
                 and v.strip().startswith("0") for v in strings)
    for value in strings:
        s = value.strip()
        lead, trail = value[:len(value) - len(value.lstrip())], value[len(value.rstrip()):]
        if s.isascii() and s.isdigit():
            f = _Shape("integer", width=len(s) if padded else 0, lead=lead, trail=trail)
        elif (parts := integral_float_parts(s)) is not None:
            f = _Shape("float", parts[1], lead=lead, trail=trail)
        elif thousands_digits(s) is not None:
            f = _Shape("grouped", lead=lead, trail=trail)
        elif is_hex_id(s):
            f = _Shape("pattern", "".join(("A" if s.upper() == s else "a") if c.isalnum() else c for c in s),
                       lead=lead, trail=trail)
        elif "@" in s:
            _refuse("e-mail identifier layouts need an explicit sharing key-format policy")
        else:
            # The template contains only character classes and punctuation, never a name or ID.
            f = _Shape("pattern", char_mask(s), lead=lead, trail=trail)
        forms[f].add(normalize_id(s))
    if not forms:
        _refuse("an identifier column has no nonmissing keys")
    if any(len(keys) < minimum for keys in forms.values()):
        _refuse("an identifier format class is held by fewer than --min-cell people")
    if len(forms) != 1:
        _refuse("mixed identifier layouts cannot yet be assigned independently with valid joins")
    return next(iter(forms))


def _table_spec(fr: Any, ti: int, namespaces: dict[int, str], minimum: int) -> _Table:
    tp = fr.tables[ti]
    keys: dict[int, list[str | None]] = {}
    shapes, cards = {}, {}
    for col in tp.entity.id_columns:
        raw = tp.private.id_raw[col]
        keys[col] = entity_keys(raw)
        if any(k is None for k in keys[col]):
            _refuse("missing identifiers need a reviewed fictional orphan/missing-key contract")
        cards[col] = len(set(keys[col]))
        if tp.n_rows:
            try:
                shapes[col] = _shape(raw, minimum)
            except UnsupportedColumnError as error:
                raise UnsupportedColumnError(error.message.replace(
                    "sharing population: ", f"sharing population: column {tp.columns[col].name!r}: ", 1),
                    error.fix) from error
    primary = tp.entity.primary
    counts = np.empty(0, dtype=np.int64)
    groups = [list(group) for group in tp.entity.aliases]
    seen = {c for group in groups for c in group}
    groups += [[c] for c in tp.entity.id_columns if c not in seen]
    rebalanced = None
    if primary is not None and tp.n_rows:
        source = Counter(Counter(keys[primary]).values())
        classes = _rebalance_counts(source, minimum)
        if classes is None:
            _refuse("a primary-person rows-per-ID count class has fewer than --min-cell donors and cannot be "
                    "moved into a held count class with an exact row total; its count cannot be released")
        counts = np.asarray([k for k, h in sorted(classes.items()) for _ in range(h)], dtype=np.int64)
        if classes != source:
            rebalanced = len(counts) - sum(source.values())
            # the primary person (and its one-to-one aliases) now has as many fictional people as there are counts
            for col in next(group for group in groups if primary in group):
                cards[col] = len(counts)
    pairs = []
    for j, a in enumerate(tp.entity.id_columns):
        for b in tp.entity.id_columns[j + 1:]:
            if len(set(zip(keys[a], keys[b]))) == tp.n_rows:
                pairs.append((a, b))
    return _Table((fr.rel_path, ti), tp, namespaces, shapes, cards, groups, counts, pairs, rebalanced)


def _fold_widths(widths: Counter[int], minimum: int) -> Counter[int]:
    """Give the IDs of each digit width held by fewer than ``minimum`` IDs the nearest held width.

    Only a width that ``minimum`` real IDs hold on their own receives IDs (the longer on a tie), so a released
    width always has ``minimum`` real donors; with no such width the counts are returned as they are (refused).
    """
    held = [w for w, c in widths.items() if c >= minimum]
    if not held:
        return Counter(widths)
    out: Counter[int] = Counter()
    for width, count in widths.items():
        out[width if width in held else min(held, key=lambda w: (abs(w - width), -w))] += count
    return out


def _rebalance_counts(classes: Counter[int], minimum: int) -> Counter[int] | None:
    """Move the people of each rows-per-person count class held by fewer than ``minimum`` people into held classes.

    ``classes`` maps a row count to the number of people holding it. Every count in the result is one that
    ``minimum`` source people hold, and the row total is exact (ADR-0003). Each rare person takes the nearest held
    count (the larger on a tie, as for ID widths); the rows that leaves over or short are made up by moving people
    one held class up or down, next to the rare counts first, so the number of people stays the source's where it
    can. Only when no such move fits is a person dropped (from the held class nearest the rare counts) or added,
    so the number of people may differ from the source. None when no class is held or no exact total exists.
    """
    held = sorted(k for k, h in classes.items() if h >= minimum)
    rare = sorted(k for k, h in classes.items() if h < minimum)
    if not rare:
        return Counter(classes)
    total = sum(k * h for k, h in classes.items())
    if not held or total % reduce(gcd, held):
        return None

    def distance(k: int) -> int:
        return min(abs(k - c) for c in rare)

    out: Counter[int] = Counter({k: classes[k] for k in held})
    for c in rare:
        out[min(held, key=lambda k: (abs(k - c), -k))] += classes[c]
    while True:
        gap = _shift(out, held, total - sum(k * h for k, h in out.items()), distance)
        if not gap:
            return +out
        if gap > 0 and (parts := _parts(gap, held, distance)) is not None:
            out.update(parts)
            return +out
        donors = [k for k in held if out[k]]
        if not donors:
            return None
        out[min(donors, key=lambda k: (distance(k), -k))] -= 1


def _shift(out: Counter[int], held: list[int], gap: int, distance: Callable[[int], int]) -> int:
    """Close ``gap`` rows by moving people between neighbouring held classes, nearest the rare counts first; the
    rows still left over (positive) or in excess (negative) are returned."""
    while gap:
        moves = [(a, b) for a, b in zip(held, held[1:]) if b - a <= abs(gap) and out[a if gap > 0 else b]]
        if not moves:
            return gap
        a, b = min(moves, key=lambda m: (min(distance(m[0]), distance(m[1])), -m[1]))
        source, target = (a, b) if gap > 0 else (b, a)
        moved = min(out[source], abs(gap) // (b - a))
        out[source] -= moved
        out[target] += moved
        gap -= moved * (target - source)
    return gap


def _parts(rows: int, held: list[int], distance: Callable[[int], int]) -> Counter[int] | None:
    """The fewest new people whose held counts add up to ``rows`` (nearest the rare counts on a tie), or None."""
    best: list[tuple[int, int, int] | None] = [(0, 0, 0)] + [None] * rows
    for value in range(1, rows + 1):
        options = [(best[value - k][0] + 1, best[value - k][1] + distance(k), k)
                   for k in held if k <= value and best[value - k] is not None]
        best[value] = min(options, key=lambda o: (o[0], o[1], -o[2])) if options else None
    if best[rows] is None:
        return None
    parts: Counter[int] = Counter()
    while rows:
        k = best[rows][2]
        parts[k] += 1
        rows -= k
    return parts


def _columns(ns: IdNamespace) -> str:
    return ", ".join(sorted({repr(name) for _, _, _, name in ns.members})) or repr(ns.key)


def _mint(ns: IdNamespace, n: int, minimum: int, seed: int, banned: set[str],
          users: RealUsers, shapes: list[_Shape]) -> np.ndarray:
    """Draw a bag from an aggregate format domain, never from ns.mapping."""
    rng = stream(seed, "sharing_population", ns.key, "identifier_pool")
    if not n:
        return np.empty(0, dtype=object)
    if namespace_is_identity(ns):
        return rng.permutation(np.arange(1, n + 1, dtype=np.int64)).astype(object)
    reps = list(ns.representatives.values())
    if all(_is_digit_rep(v) for v in reps):
        widths = Counter(len(normalize_id(v)) for v in reps)
        padded = [s.width for s in shapes if s.kind == "integer" and s.width]
        all_padded = all(s.kind == "integer" and s.width for s in shapes)
        if not all_padded:
            widths = _fold_widths(widths, minimum)
            if any(c < minimum for c in widths.values()):
                _refuse(f"identifier column {_columns(ns)} has fewer than --min-cell ({minimum}) IDs")
        # Padding hides significant width, and a rare unpadded width is folded into a common one, so no rare
        # original width is retained.
        sizes = ([min(padded)] * n if all_padded else
                 [w for w, count in sorted(widths.items()) for _ in range(count)])
        if len(sizes) != n:
            _refuse("aliased identifier domains have incompatible cardinalities")
        tokens = [f"slot:{i}" for i in range(n)]
        nat = dict(zip(tokens, sizes))
        limit = {k: min(padded) for k in tokens} if padded else {}
        bounds = limit if all_padded else {}
        real = np.asarray(sorted({int(k) for k in banned if k.isascii() and k.isdigit() and len(k) <= 18}),
                          dtype=np.int64)
        notes: list[str] = []
        if ns.check == "luhn":
            nat = {key: width - 1 for key, width in nat.items()}
            bounds = {key: width - 1 for key, width in bounds.items()}
            limit = {key: width - 1 for key, width in limit.items()}
            bodies = allocate_digit_ids(tokens, nat, bounds, np.unique(real[real >= 10] // 10), rng,
                                        notes, lowest=1, pad_limit=limit)
            values = np.asarray([int(body) * 10 + luhn_check_digit(str(int(body))) for body in bodies],
                                dtype=np.int64)
        else:
            values = allocate_digit_ids(tokens, nat, bounds, real, rng, notes, pad_limit=limit)
        return rng.permutation(values).astype(object)
    if any(_is_digit_rep(v) for v in reps):
        _refuse("mixed numeric and patterned identifier domains are unsupported in sharing")
    hex_mode = all(is_hex_id(v) for v in reps)
    masks = Counter(char_mask("".join(c for c in v if c.isalnum())).lower() for v in reps)
    if not hex_mode and (len(masks) != 1 or any(c < minimum for c in masks.values())):
        _refuse("patterned identifier classes cannot be protected while preserving these joins")
    if hex_mode:
        # UUID/hash character classes vary with the random digits, not with a donor.
        rep = "".join("0" if c.isalnum() else c for c in sorted(reps)[0])
        prefix = ""
    else:
        prefix = ns.prefix
        mask = next(iter(masks))
        rep = prefix + mask[len(prefix):]
    invented = IdNamespace(ns.key, representatives={f"slot:{i}": rep for i in range(n)})
    _pattern_mapping(invented, list(invented.representatives), banned, set(), rng,
                     prefix=prefix, stems=users.stems, text_taken=users.text_taken)
    return rng.permutation(np.asarray(list(invented.mapping.values()), dtype=object))


def _balanced(pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if size and not len(pool):
        _refuse("a required fictional key domain is empty")
    return rng.permutation(np.resize(rng.permutation(pool), size))


def plan_population(files: list[Any], seed: int, settings: Any, reg: IdRegistry,
                    namespaces: dict[tuple[str, int], dict[int, str]]) -> dict[tuple[str, int], TablePopulation]:
    """Plan the complete directory atomically, then attach plans to its profiles.

    An unsupported contract raises before any new plan is attached: a
    ``PopulationRefusal`` when it concerns particular tables (every table whose
    own contract is refused, else the first table refused later), so the caller
    can leave those out and plan the rest again. FAILed files are not planned:
    a join to one of them is dropped, and a key domain holds only the keys of
    the tables planned. Every source key and row assignment is confined to
    extracting the aggregate specification and collision exclusion sets;
    fictional membership uses independent streams.
    """
    minimum = max(int(settings.min_cell), 1)
    tables, refused = [], {}
    for fr in files:
        for ti in range(len(fr.tables)) if not fr.failed else ():
            try:
                tables.append(_table_spec(fr, ti, namespaces.get((fr.rel_path, ti), {}), minimum))
            except UnsupportedColumnError as error:
                refused[(fr.rel_path, ti)] = error
    if refused:
        raise PopulationRefusal(refused)
    live_namespaces = {key for table in tables for key in table.namespaces.values()}
    # A table left out takes its keys with it: a domain (its format classes, and whether it is the row numbers
    # 1..n that are kept, D9) is decided over the planned tables' keys alone. Row numbers whose roster is left out
    # are ordinary real IDs: they are minted like any other and the leak scan checks them (``reg`` is updated below).
    held = IdRegistry()
    for table in tables:
        for col, key in table.namespaces.items():
            held.register(key, table.key[0], table.profile.name, col, table.profile.columns[col].name,
                          table.profile.private.id_raw[col])
    planned = {}
    for key, ns in reg.namespaces.items():
        if key in live_namespaces:
            here = held.namespaces[key]
            keep = {k: v for k, v in ns.representatives.items() if k in here.representatives}
            if len(keep) == len(ns.representatives):
                planned[key] = ns
            else:
                shrunk = replace(ns, representatives=keep, row_number_sizes=list(here.row_number_sizes))
                planned[key] = replace(shrunk, identity=namespace_is_identity(shrunk))
    parent = {key: key for key in live_namespaces}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for table in tables:
        for group in table.groups:
            for col in group[1:]:
                union(table.namespaces[group[0]], table.namespaces[col])
    sizes: dict[str, int] = {}
    for key, ns in planned.items():
        r, size = root(key), len(ns.representatives)
        if r in sizes and sizes[r] != size:
            _refuse("one-to-one alias namespaces have different population cardinalities")
        sizes[r] = size

    def ref(rel_path: str, tp: TableProfile, col: int) -> str:
        return "::".join([rel_path, *([tp.name] if tp.name is not None else []), tp.columns[col].name])

    by_ref = {ref(table.key[0], table.profile, col): (table.key, col) for table in tables for col in table.namespaces}
    left_out = {ref(fr.rel_path, tp, col) for fr in files if fr.failed for tp in fr.tables
                for col in range(len(tp.columns))}
    specs = {table.key: table for table in tables}

    def inverse_coverage(table: _Table, col: int, target_key: tuple[str, int], target_col: int,
                         link: dict[str, Any]) -> bool:
        """A complete roster's partial overlap with its own child is not a FK.

        Profiling records coverage in both directions when a child happens to
        have unique keys (e.g. one IEP row per person). Only the child's full,
        zero-orphan link is a membership constraint. Require that reverse proof
        and a complete unique roster; ambiguous partial joins still fail.
        """
        if link.get("foreign_key") is not False:
            return False
        target = specs[target_key]
        domain = root(table.namespaces[col])
        if (table.cardinalities[col] != table.profile.n_rows
                or table.cardinalities[col] != sizes[domain]
                or target.cardinalities[target_col] >= table.cardinalities[col]):
            return False
        return any(reverse.get("column") == target.profile.columns[target_col].name
                   and by_ref.get(reverse.get("parent")) == (table.key, col)
                   and reverse.get("foreign_key") is True
                   and reverse.get("key_containment") == 1
                   and reverse.get("orphan_rows", 0) == 0
                   for reverse in target.profile.structure.cross_file_keys)

    # Every zero-orphan source FK is an inclusion constraint, not a source subset.
    dependencies: dict[tuple[tuple[str, int], str], list[tuple[tuple[str, int], str]]] = defaultdict(list)
    for table in tables:
        names = {p.name: p.index for p in table.profile.columns}
        for link in table.profile.structure.cross_file_keys:
            if link.get("parent") in left_out:
                continue  # its target is not written, so there is no join to keep
            with _blame(table.key):
                col = names.get(link.get("column"))
                target = by_ref.get(link.get("parent"))
                if col not in table.namespaces or target is None:
                    _refuse("a detected join target could not be resolved")
                target_key, target_col = target
                domain = root(table.namespaces[col])
                if domain != root(specs[target_key].namespaces[target_col]):
                    _refuse("a foreign key and its target have different fictional namespaces")
                if inverse_coverage(table, col, target_key, target_col, link):
                    continue
                if link.get("orphan_rows", 0) or link.get("key_containment", 1) < 1:
                    _refuse("nonzero source orphan rates need an explicit fictional orphan contract")
            dependencies[(table.key, domain)].append((target_key, domain))
    membership: dict[tuple[tuple[str, int], str], np.ndarray] = {}
    active: set[tuple[tuple[str, int], str]] = set()

    def members(table: _Table, domain: str) -> np.ndarray:
        with _blame(table.key):
            return table_members(table, domain)

    def table_members(table: _Table, domain: str) -> np.ndarray:
        node = (table.key, domain)
        if node in membership:
            return membership[node]
        cardinalities = {n for col, n in table.cardinalities.items() if root(table.namespaces[col]) == domain}
        if len(cardinalities) != 1:
            _refuse("multiple roles of the same namespace require incompatible local memberships")
        n = next(iter(cardinalities), 0)
        if node in active:
            # Mutual complete rosters impose no restriction beyond the full domain.
            if n == sizes[domain]:
                return np.arange(n, dtype=np.int64)
            _refuse("cyclic subset joins require a reviewed fictional membership contract")
        active.add(node)
        available = np.arange(sizes[domain], dtype=np.int64)
        for target_key, _ in sorted(set(dependencies.get(node, []))):
            available = np.intersect1d(available, members(specs[target_key], domain), assume_unique=True)
        if len(available) < n:
            _refuse("the independently invented memberships cannot satisfy all required parent joins")
        rng = stream(seed, "sharing_population", table.key[0], table.key[1], domain, "membership")
        membership[node] = rng.permutation(available)[:n]
        active.remove(node)
        return membership[node]

    for table in tables:
        for key in sorted(set(table.namespaces.values())):
            members(table, root(key))
    edges: set[tuple[str, str]] = set()
    for table in tables:
        names = {p.name: p.index for p in table.profile.columns}
        for relation in table.profile.entity.nesting:
            with _blame(table.key):
                child, par = names.get(relation["child"]), names.get(relation["parent"])
                if child not in table.namespaces or par not in table.namespaces:
                    _refuse("a nesting contract does not identify two known key columns")
                a, b = root(table.namespaces[child]), root(table.namespaces[par])
                if a == b:
                    _refuse("self-nesting keys need an explicit acyclic relationship contract")
            edges.add((a, b))
    for a, b in edges:
        todo, seen = [b], set()
        while todo:
            v = todo.pop()
            if v == a:
                _refuse("cyclic nesting cannot be regenerated as a fictional hierarchy")
            if v not in seen:
                seen.add(v)
                todo.extend(y for x, y in edges if x == v)
    def path(a: str, b: str, available: set[tuple[str, str]]) -> list[str] | None:
        todo = [(a, [a])]
        while todo:
            current, visited = todo.pop()
            if current == b:
                return visited
            todo.extend((target, [*visited, target]) for source, target in sorted(available)
                        if source == current and target not in visited)
        return None

    # Profiling reports transitive FDs too (student -> family -> household and
    # student -> household). Invent only the hierarchy's edges, then compose the
    # implied relationships rather than independently drawing contradictory ones.
    direct = {edge for edge in edges if path(edge[0], edge[1], edges - {edge}) is None}
    relations = {(a, b): _balanced(np.arange(sizes[b], dtype=np.int64), sizes[a],
                                  stream(seed, "sharing_population", a, b, "nesting")) for a, b in sorted(direct)}
    for a, b in sorted(edges - direct):
        route = path(a, b, direct)
        if route is None:
            _refuse("a transitive nesting relationship cannot be resolved")
        values = np.arange(sizes[a], dtype=np.int64)
        for source, target in zip(route, route[1:]):
            values = relations[(source, target)][values]
        relations[(a, b)] = values
    # Original values are used from here only for collision exclusion and format domains.
    shadows = list(shadow_namespaces(files))
    all_ns = [*reg.namespaces.values(), *shadows]
    other_ids = {v.strip() for fr in files for tp in fr.tables for raw in tp.private.id_raw.values()
                 for v in set(np.asarray(raw, dtype=object).tolist()) if isinstance(v, str) and id_text(v)}
    banned = {normalize_id(v) for ns in all_ns for v in ns.raw_values} | {normalize_id(v) for v in other_ids}
    texts = next((t.profile.private.extra.get(real_texts.RUN_KEY) for t in tables
                  if t.profile.private.extra.get(real_texts.RUN_KEY) is not None), None)
    users = RealUsers.of(all_ns, avoid_emails=real_texts.email_addresses(texts) if texts else (),
                         avoid_ids=other_ids, avoid_texts=texts)
    pools = {}
    for key, ns in sorted(planned.items()):
        forms = [table.shapes[col] for table in tables for col, k in table.namespaces.items()
                 if k == key and col in table.shapes]
        pools[key] = _mint(ns, sizes[root(key)], minimum, seed, banned, users, forms)
    plans = {}
    for table in tables:
        with _blame(table.key):
            tp, n = table.profile, table.profile.n_rows
            if not table.namespaces or not n:
                empty = {col: np.empty(0, dtype=object) for col in table.namespaces}
                plans[table.key] = TablePopulation(empty, None, 0, {})
                continue
            primary = tp.entity.primary
            if primary is None:
                _refuse("an identifier table has no primary key contract")
            group_of = {c: min(g) for g in table.groups for c in g}
            primary_group = group_of[primary]
            group_col = {min(g): g[0] for g in table.groups}
            primary_domain = root(table.namespaces[primary])
            primary_members = members(table, primary_domain)
            rng = stream(seed, "sharing_population", table.key[0], table.key[1], "counts")
            counts = rng.permutation(table.counts)
            index = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
            order = stream(seed, "sharing_population", table.key[0], table.key[1], "rows").permutation(n)
            index = index[order]
            assigned = {primary_group: primary_members[index]}
            pending = set(group_col) - set(assigned)
            local_edges = []
            names = {p.name: p.index for p in tp.columns}
            for relation in tp.entity.nesting:
                a, b = names[relation["child"]], names[relation["parent"]]
                local_edges.append((group_of[a], group_of[b], root(table.namespaces[a]), root(table.namespaces[b])))
            while pending:
                progress = False
                for a, b, da, db in local_edges:
                    if a in assigned and b in pending:
                        assigned[b] = relations[(da, db)][assigned[a]]
                        pending.remove(b)
                        progress = True
                    elif b in assigned and a in pending:
                        out = np.empty(n, dtype=np.int64)
                        relation = relations[(da, db)]
                        eligible = members(table, da)
                        for par in np.unique(assigned[b]):
                            rows = np.flatnonzero(assigned[b] == par)
                            candidates = eligible[relation[eligible] == par]
                            if not len(candidates):
                                _refuse("a fictional parent has no child eligible for its required table")
                            out[rows] = _balanced(candidates, len(rows), stream(
                                seed, "sharing_population", table.key[0], table.key[1], a, int(par), "children"))
                        assigned[a] = out
                        pending.remove(a)
                        progress = True
                if not progress:
                    a = min(pending)
                    col = group_col[a]
                    assigned[a] = _balanced(members(table, root(table.namespaces[col])), n,
                                            stream(seed, "sharing_population", table.key[0], table.key[1], a, "secondary"))
                    pending.remove(a)
            for a, b, da, db in local_edges:
                if not np.array_equal(relations[(da, db)][assigned[a]], assigned[b]):
                    _refuse("the fictional nesting contracts disagree in this table")
            constrained = {g for a, b, _, _ in local_edges for g in (a, b)} | {primary_group}
            # A many-to-many relation may require unique pairs. Invent edges directly
            # inside each partner block; do not import the source incidence matrix.
            for attempt in range(8):
                changed = False
                for ca, cb in table.unique_pairs:
                    a, b = group_of[ca], group_of[cb]
                    if len(np.unique(np.column_stack((assigned[a], assigned[b])), axis=0)) == n:
                        continue
                    if b in constrained or table.cardinalities[group_col[b]] == n:
                        a, b = b, a
                    if b in constrained or table.cardinalities[group_col[b]] == n:
                        continue
                    col = group_col[b]
                    pool = members(table, root(table.namespaces[col]))
                    draw = stream(seed, "sharing_population", table.key[0], table.key[1], b, attempt, "pair_repair")
                    out = np.empty(n, dtype=np.int64)
                    for value in np.unique(assigned[a]):
                        rows = np.flatnonzero(assigned[a] == value)
                        if len(rows) > len(pool):
                            _refuse("a required unique relationship has too few fictional partner keys")
                        out[rows] = draw.choice(pool, len(rows), replace=False)
                    assigned[b] = out
                    changed = True
                if not changed:
                    break
            slots = {col: assigned[group_of[col]] for col in table.namespaces}
            for col, key in table.namespaces.items():
                node = (table.key, root(key))
                for target_key, _ in dependencies.get(node, []):
                    if not np.isin(slots[col], members(specs[target_key], root(key))).all():
                        _refuse("a regenerated secondary relationship cannot satisfy its required parent join")
                if table.cardinalities[col] == n and len(np.unique(slots[col])) != n:
                    _refuse("a regenerated secondary key cannot preserve required uniqueness")
            for a, b in table.unique_pairs:
                if len(np.unique(np.column_stack((slots[a], slots[b])), axis=0)) != n:
                    _refuse("regenerated relationships cannot preserve the required composite-key uniqueness")
            values = {col: table.shapes[col].render(pools[key][slots[col]]) for col, key in table.namespaces.items()}
            for array in [*values.values(), *slots.values(), index]:
                array.flags.writeable = False
            plans[table.key] = TablePopulation(values, index, len(counts), slots)
    by_path = {fr.rel_path: fr for fr in files}
    for table in tables:
        table.profile.private.extra[PLAN_KEY] = plans[table.key]
        if table.rebalanced is not None:
            fr = by_path[table.key[0]]
            name = f"table {table.profile.name!r}: " if table.profile.name is not None else ""
            people = ("the same number of people as the source" if not table.rebalanced else
                      f"{abs(table.rebalanced)} {'more' if table.rebalanced > 0 else 'fewer'} "
                      f"{'person' if abs(table.rebalanced) == 1 else 'people'} than the source")
            text = (f"{name}rows per person rebalanced: people with a rare number of rows were given common row "
                    f"counts and the row total kept; {people}")
            fr.messages = [m for m in fr.messages if not (m.code == REBALANCED_CODE and m.text == text)]
            fr.messages.append(Message("info", REBALANCED_CODE, text))
    for key, ns in planned.items():
        reg.namespaces[key].identity = ns.identity
    return plans


def generation_profile(tp: TableProfile, plan: TablePopulation) -> TableProfile:
    """A generation-only copy; keep original source models/indices for reports."""
    extra = dict(tp.private.extra)
    extra[PLAN_KEY] = plan
    private = replace(tp.private, entity_index=plan.entity_index, extra=extra)
    mm = replace(tp.structure.missingness, entity_rows=None, cross=None)
    structural = dict(tp.structure.private)
    structural.pop("sort", None)
    # Exact source duplicate frequencies and original sort ties are not contracts.
    sp = replace(tp.structure, private=structural, missingness=mm, duplicate_rows={})
    entity = replace(tp.entity, n_entities=plan.n_entities)
    columns = [replace(p, params=dict(p.params), flags=dict(p.flags),
                       messages=[m for m in p.messages if m.code != "id_columns_unrelated"]) for p in tp.columns]
    return replace(tp, columns=columns, private=private, entity=entity, structure=sp,
                   messages=[*tp.messages, Message("info", "sharing_population",
                             "Fictional memberships and key relationships were generated from aggregate contracts.")])
