"""Run settings, column qualifiers and ``--config`` YAML loading (FOUNDATION; shared file).

Every CLI option of design section 0.2 has a field here. Flags whose behaviour
belongs to a later area (dates, decimal comma, Excel sheet, ...) are parsed and
stored; the area hook reads them from :class:`ColumnOptions` / :class:`Settings`.

Precedence (design 7): defaults < ``--config`` file < CLI flags. Scalars from
the CLI replace config values; repeatable column/list options are merged (config
entries first, CLI entries added; each column reference keeps its provenance).
Column options are resolved per matched column by :meth:`Settings.column_options`:
a CLI option supersedes the config entries that compete with it for that column
(the type-deciding options as one group, ``null``/``exclude`` as another, and the
value of each ``COL=VALUE`` option), however either reference is qualified.

Config-only keys: ``rules`` (declared cross-column rules, D5; AREA D; e.g.
``rules: ["KCSE <= Yr_JoinCampus"]``, validated by ``profile.structure``) and
``structure`` (``{disable: [...]}`` turns off automatic structural switches, D5:
``functional_dependencies``, ``derived_templates``, ``composite_keys``,
``cross_file_keys``, and the layout switches ``sort_order`` and
``duplicate_rows`` (section 6); read with :meth:`Settings.structure_disabled`). In YAML the
key ``null`` parses as a null key; it is accepted as the ``null`` option.

Column references
-----------------
A ``COL`` may be qualified: ``relative/path.csv::COL`` applies to one file and
``book.xlsx::Sheet::COL`` to one sheet. Paths are compared as posix paths
relative to the input root (for a single-file INPUT, the file name). Column
names are matched exactly (raw header text). ``COL=VALUE`` is split on the last
``=``. References that never match a column produce a run-level warning.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import UsageError
from .model import ColumnType


# --------------------------------------------------------------------------- column references
@dataclass(frozen=True)
class ColumnRef:
    column: str
    file: str | None = None
    sheet: str | None = None
    raw: str = ""
    #: provenance: True for an entry from the ``--config`` file (CLI flags take precedence, design 7)
    from_config: bool = False

    def matches(self, file_rel: str, sheet: str | None, column: str) -> bool:
        if self.column != column:
            return False
        if self.file is not None and self.file.replace("\\", "/").strip("/") != file_rel:
            return False
        if self.sheet is not None and self.sheet != sheet:
            return False
        return True


def parse_column_ref(text: str, from_config: bool = False) -> ColumnRef:
    """Parse ``COL``, ``path::COL`` or ``path::Sheet::COL``."""
    if not isinstance(text, str) or text == "":
        raise UsageError(f"empty column reference: {text!r}")
    parts = text.split("::")
    if len(parts) == 1:
        return ColumnRef(column=parts[0], raw=text, from_config=from_config)
    if len(parts) == 2:
        return ColumnRef(column=parts[1], file=parts[0], raw=text, from_config=from_config)
    if len(parts) == 3:
        return ColumnRef(column=parts[2], file=parts[0], sheet=parts[1], raw=text, from_config=from_config)
    raise UsageError(f"bad column reference {text!r}: use COL, path::COL or book.xlsx::Sheet::COL")


def parse_column_value(text: str, what: str, from_config: bool = False) -> tuple[ColumnRef, str]:
    """Parse ``COL=VALUE`` (split on the last '=')."""
    col, sep, val = str(text).rpartition("=")
    if not sep or not col or val == "":
        raise UsageError(f"expected COL={what}, got {text!r}")
    return parse_column_ref(col, from_config), val


# --------------------------------------------------------------------------- settings
_LIST_COL = (
    "force_factor", "force_text", "dayfirst", "date_cols", "id_col", "org_code", "person_id",
    "no_exact_extremes", "null", "exclude", "calendar",
)
_KV_COL = ("force_type", "pii_col", "date_format")
_LIST_STR = ("na_value", "include", "exclude_path", "range", "keep_name")
_SCALARS = (
    "mode",
    "factor_max", "decimal", "no_default_ids", "min_cell", "encoding", "delimiter", "sheet",
    "header_row", "seed", "force", "max_row_matches", "no_strict", "id_check", "keep_titles",
)
_CONFIG_ONLY = ("rules", "structure")
#: Options that decide what a column IS (one of them wins, ``detect/types.py``). A CLI option of this
#: group that matches a column supersedes every config entry of the group for that column.
_TYPE_GROUP = ("force_type", "pii_col", "id_col", "person_id", "org_code", "force_factor", "force_text")
#: ``--null`` / ``--exclude``: a CLI one supersedes a config one. A CLI type option never undoes a
#: config ``null``/``exclude`` (the column stays dropped or blanked: the privacy-preserving reading).
_ROLE_GROUP = ("null", "exclude")
#: D5 structural switches that ``structure: {disable: [...]}`` can turn off (AREA D applies them).
STRUCTURE_SWITCHES = ("functional_dependencies", "derived_templates", "composite_keys", "cross_file_keys",
                      "sort_order", "duplicate_rows")
#: ``--id-check`` values (design section 4: opt-in check digits; AREA D).
ID_CHECKS = ("luhn",)


@dataclass
class Settings:
    """All run options (defaults per design sections 0 and 10)."""

    # types
    factor_max: int = 20
    force_factor: list[ColumnRef] = field(default_factory=list)
    force_text: list[ColumnRef] = field(default_factory=list)
    force_type: list[tuple[ColumnRef, str]] = field(default_factory=list)
    pii_col: list[tuple[ColumnRef, str]] = field(default_factory=list)  # AREA C
    dayfirst: list[ColumnRef] = field(default_factory=list)  # AREA A
    date_cols: list[ColumnRef] = field(default_factory=list)  # AREA A
    date_format: list[tuple[ColumnRef, str]] = field(default_factory=list)  # AREA A
    calendar: list[ColumnRef] = field(default_factory=list)  # AREA A: dates drawn from observed dates only
    na_value: list[str] = field(default_factory=list)
    decimal: str = "."  # AREA B
    # IDs and codes
    id_col: list[ColumnRef] = field(default_factory=list)
    no_default_ids: bool = False
    org_code: list[ColumnRef] = field(default_factory=list)
    person_id: list[ColumnRef] = field(default_factory=list)
    id_check: str | None = None  # AREA D: 'luhn' recomputes a check digit on synthetic digit IDs (section 4)
    # floors
    no_exact_extremes: list[ColumnRef] = field(default_factory=list)
    min_cell: int = 10
    # columns
    null: list[ColumnRef] = field(default_factory=list)
    exclude: list[ColumnRef] = field(default_factory=list)
    # input
    encoding: str | None = None
    delimiter: str | None = None
    sheet: str | None = None  # AREA E
    header_row: int | None = None  # AREA E
    #: AREA E: ``--range [Sheet!]A5:H200`` (repeatable): the table region of a sheet
    range: list[str] = field(default_factory=list)
    #: AREA E: ``--keep-titles`` copies Excel title/notes rows verbatim (default: placeholders)
    keep_titles: bool = False
    include: list[str] = field(default_factory=list)
    exclude_path: list[str] = field(default_factory=list)
    #: D34: header, sheet, file or folder names that the personal-data screen of names flagged but that are not
    #: personal data (exact text; a file name with or without its extension): kept as they are
    keep_name: list[str] = field(default_factory=list)
    # run
    seed: int | None = None
    config_path: str | None = None
    force: bool = False
    max_row_matches: float | None = None
    no_strict: bool = False
    verbosity: int = 0
    json_output: bool = False
    # config-only
    rules: Any = None  # AREA D (declared cross-column rules, D5)
    structure: dict[str, Any] | None = None  # AREA D: {'disable': [switch, ...]} (D5)
    #: Sharing packages use fictional populations and blocking release gates.
    mode: str = "sharing"

    # ------------------------------------------------------------------ helpers
    def __post_init__(self) -> None:
        self._used: set[str] = set()

    @property
    def na_tokens(self) -> frozenset[str]:
        from .hints import DEFAULT_NA_TOKENS

        return frozenset(DEFAULT_NA_TOKENS) | frozenset(self.na_value)

    def column_options(self, file_rel: str, sheet: str | None, column: str) -> "ColumnOptions":
        """Resolve every column option for one column; raises UsageError on conflicts.

        Precedence is resolved per matched column before any conflict check (defaults < ``--config`` <
        CLI, design 7), whatever the qualification of the references (``score`` and
        ``data.csv::score`` name the same column):

        * a CLI option of :data:`_TYPE_GROUP` that matches the column supersedes every config entry of
          that group for the column (e.g. config ``force_factor: [x]`` + ``--force-text x`` -> text;
          config ``force_type: {score: FLOAT}`` + ``--force-type data.csv::score=INTEGER`` -> INTEGER);
        * likewise within :data:`_ROLE_GROUP` (``--null`` / ``--exclude``);
        * for every ``COL=VALUE`` option a CLI value supersedes the config value.

        Conflicts left within one source (two CLI flags, or two config entries) stay usage errors.
        A superseded config entry still counts as used (it matched a column).
        """
        opts = ColumnOptions()
        hits = {name: [r for r in getattr(self, name) if r.matches(file_rel, sheet, column)] for name in _LIST_COL}
        kv_hits = {name: [(r, v) for r, v in getattr(self, name) if r.matches(file_rel, sheet, column)]
                   for name in _KV_COL}
        for refs in hits.values():
            self._used.update(r.raw for r in refs)
        for pairs in kv_hits.values():
            self._used.update(r.raw for r, _ in pairs)

        def refs_of(name: str) -> list[ColumnRef]:
            return hits[name] if name in hits else [r for r, _ in kv_hits[name]]

        for group in (_TYPE_GROUP, _ROLE_GROUP):
            if any(not r.from_config for name in group for r in refs_of(name)):
                for name in group:
                    if name in hits:
                        hits[name] = [r for r in hits[name] if not r.from_config]
                    else:
                        kv_hits[name] = [(r, v) for r, v in kv_hits[name] if not r.from_config]
        for name, pairs in kv_hits.items():
            if any(not r.from_config for r, _ in pairs):
                kv_hits[name] = [(r, v) for r, v in pairs if not r.from_config]

        def value(name: str, what: str) -> str | None:
            vals = [v for _, v in kv_hits[name]]
            if len(set(vals)) > 1:
                raise UsageError(f"conflicting {what} values for column {column!r} in {file_rel}: {sorted(set(vals))}")
            return vals[-1] if vals else None

        for name in _LIST_COL:
            setattr(opts, name, bool(hits[name]))
        if opts.org_code and opts.force_text:
            raise UsageError(f"column {column!r} in {file_rel} is both --org-code and --force-text")
        opts.force_type = value("force_type", "--force-type")
        opts.pii_col = value("pii_col", "--pii-col")
        opts.date_format = value("date_format", "--date-format")

        # Different override kinds from the same source resolve by precedence in detect/types.py
        # (--force-type > --pii-col > --id-col/--person-id > --org-code/--force-factor/--force-text);
        # only directly contradictory pairs are usage errors.
        if opts.force_factor and opts.force_text:
            raise UsageError(f"column {column!r} in {file_rel} is both --force-factor and --force-text")
        if opts.null and opts.exclude:
            raise UsageError(f"column {column!r} in {file_rel} is both --null and --exclude")
        return opts

    def structure_disabled(self, switch: str) -> bool:
        """True if the config disables the D5 structural switch ``switch`` (see STRUCTURE_SWITCHES)."""
        return bool(self.structure) and switch in (self.structure.get("disable") or [])

    def unused_refs(self) -> list[str]:
        refs: list[str] = []
        for name in _LIST_COL:
            refs.extend(r.raw for r in getattr(self, name))
        for name in _KV_COL:
            refs.extend(r.raw for r, _ in getattr(self, name))
        return sorted({r for r in refs if r not in self._used})

    def to_public(self) -> dict[str, Any]:
        """JSON-safe dict for the manifest (user-supplied names only)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in _LIST_COL:
                v = [r.raw for r in v]
            elif f.name in _KV_COL:
                v = [f"{r.raw}={val}" for r, val in v]
            out[f.name] = v
        return out


@dataclass
class ColumnOptions:
    """Options resolved for one column (all False/None by default)."""

    force_factor: bool = False
    force_text: bool = False
    force_type: str | None = None
    pii_col: str | None = None
    dayfirst: bool = False
    date_cols: bool = False
    date_format: str | None = None
    calendar: bool = False
    id_col: bool = False
    org_code: bool = False
    person_id: bool = False
    no_exact_extremes: bool = False
    null: bool = False
    exclude: bool = False


# --------------------------------------------------------------------------- building settings
def _as_list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, int, float)):
        return [value]
    raise UsageError(f"config key {key!r} must be a list or a single value")


def _parse_kv_items(value: Any, key: str, what: str) -> list[tuple[ColumnRef, str]]:
    """Config-file ``COL=VALUE`` entries (a mapping or a list of ``COL=VALUE`` strings)."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [(parse_column_ref(str(k), from_config=True), str(v)) for k, v in value.items()]
    return [parse_column_value(str(v), what, from_config=True) for v in _as_list(value, key)]


def _merge_kv(base: list[tuple[ColumnRef, str]], extra: list[tuple[ColumnRef, str]]) -> list[tuple[ColumnRef, str]]:
    over = {r.raw for r, _ in extra}
    return [p for p in base if p[0].raw not in over] + list(extra)


def _yaml_problem(e: Exception) -> str:
    """A YAML error without the lines of the file it quotes (they can hold a name, D34): the problem and where it
    is (line and column)."""
    problem = getattr(e, "problem", None) or type(e).__name__
    mark = getattr(e, "problem_mark", None) or getattr(e, "context_mark", None)
    where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
    context = getattr(e, "context", None)
    return (f"{context}: " if context else "") + f"{problem}{where}"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML (or JSON) config file into a raw dict."""
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UsageError(f"config file not found: {p}") from None
    except yaml.YAMLError as e:
        raise UsageError(f"config file {p} is not valid YAML: {_yaml_problem(e)}") from None
    except UnicodeDecodeError as e:
        raise UsageError(f"config file {p} is not valid YAML: {e.reason} at byte {e.start}") from None  # R7-04
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise UsageError(f"config file {p} must contain a mapping of option names")
    norm: dict[str, Any] = {}
    allowed = set(_LIST_COL) | set(_KV_COL) | set(_LIST_STR) | set(_SCALARS) | set(_CONFIG_ONLY)
    for k, v in data.items():
        key = "null" if k is None else str(k).replace("-", "_")  # YAML reads a bare `null:` key as None
        if key not in allowed:
            raise UsageError(f"unknown option {k!r} in config file {p}")
        norm[key] = v
    return norm


def build_settings(cli: dict[str, Any], config: dict[str, Any] | None = None) -> Settings:
    """Combine defaults < config < CLI into :class:`Settings`.

    ``cli`` uses Settings field names; scalars are None when not given, lists are
    lists of raw strings.
    """
    config = config or {}
    s = Settings()
    # lists of column refs
    for name in _LIST_COL:
        refs = [parse_column_ref(str(v), from_config=True) for v in _as_list(config.get(name), name)]
        refs += [parse_column_ref(str(v)) for v in cli.get(name) or ()]
        setattr(s, name, refs)
    what = {"force_type": "TYPE", "pii_col": "SUBTYPE", "date_format": "FMT"}
    for name in _KV_COL:
        base = _parse_kv_items(config.get(name), name, what[name])
        extra = [parse_column_value(str(v), what[name]) for v in cli.get(name) or ()]
        setattr(s, name, _merge_kv(base, extra))
    for name in _LIST_STR:
        vals = [str(v) for v in _as_list(config.get(name), name)] + [str(v) for v in cli.get(name) or ()]
        setattr(s, name, vals)
    for name in _SCALARS:
        val = cli.get(name)
        if val is None or val is False:
            val = config.get(name, val)
        if val is None:
            continue
        setattr(s, name, val)
    s.rules = config.get("rules")
    s.structure = config.get("structure")
    for name in ("config_path", "verbosity", "json_output"):
        if cli.get(name) is not None:
            setattr(s, name, cli[name])
    validate(s)
    return s


def validate(s: Settings) -> None:
    """Type/range checks that click cannot do for config-file values."""
    try:
        s.factor_max = int(s.factor_max)
        s.min_cell = int(s.min_cell)
        if s.seed is not None:
            s.seed = int(s.seed)
        if s.header_row is not None:
            s.header_row = int(s.header_row)
        if s.max_row_matches is not None:
            s.max_row_matches = float(s.max_row_matches)
    except (TypeError, ValueError) as e:
        raise UsageError(f"bad option value: {e}") from None
    if s.factor_max < 1:
        raise UsageError("--factor-max must be >= 1")
    if s.min_cell < 1:
        raise UsageError("--min-cell must be >= 1")
    if s.seed is not None and not (0 <= s.seed < 2**63):
        raise UsageError("--seed must be a non-negative integer below 2**63")
    if s.header_row is not None and s.header_row < 1:
        raise UsageError("--header-row must be >= 1")
    if s.max_row_matches is not None and not (0 <= s.max_row_matches <= 100):
        raise UsageError("--max-row-matches must be a percentage between 0 and 100")
    if s.decimal not in (".", ","):
        raise UsageError("--decimal must be '.' or ','")
    if s.id_check is not None:
        s.id_check = str(s.id_check).strip().lower()
        if s.id_check not in ID_CHECKS:
            raise UsageError(f"--id-check must be one of {', '.join(ID_CHECKS)}")
    if s.delimiter is not None:
        s.delimiter = normalize_delimiter(s.delimiter)
    for ref, t in s.force_type:
        try:
            ColumnType.parse(t)
        except ValueError:
            names = ", ".join(ct.value for ct in ColumnType)
            raise UsageError(f"--force-type {ref.raw}={t}: unknown type; use one of {names}") from None
    for b in ("no_default_ids", "force", "no_strict", "keep_titles"):
        setattr(s, b, bool(getattr(s, b)))
    from .detect.text import PII_SUBTYPES  # lazy: detect imports config

    for ref, sub in s.pii_col:
        if sub.strip().lower().replace("-", "_").replace(" ", "_") not in PII_SUBTYPES:
            raise UsageError(f"--pii-col {ref.raw}={sub}: unknown subtype; "
                             f"use one of {', '.join(sorted(PII_SUBTYPES))}")
    if s.structure is not None:
        if not isinstance(s.structure, dict) or set(s.structure) - {"disable"}:
            raise UsageError("config key 'structure' must be a mapping with only the key 'disable'")
        disable = s.structure.get("disable") or []
        if not isinstance(disable, list) or any(d not in STRUCTURE_SWITCHES for d in disable):
            raise UsageError(f"structure: disable must list some of {', '.join(STRUCTURE_SWITCHES)}")
    from .privacy.sharing_policy import validate_settings

    validate_settings(s)


def normalize_delimiter(d: str) -> str:
    aliases = {"\\t": "\t", "tab": "\t", "TAB": "\t", "comma": ",", "pipe": "|", "semicolon": ";"}
    d = aliases.get(d, d)
    if len(d) != 1 or d in ('"', "\r", "\n"):
        raise UsageError(f"--delimiter must be a single character (or tab), got {d!r}")
    return d
