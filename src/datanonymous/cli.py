"""Command line (FOUNDATION; shared file).

::

    datanonymous [run] INPUT --out-dir OUT [--report-dir REP] [options]   # file or directory
    datanonymous inspect INPUT [options]                                   # detection preview, writes nothing

``run`` is the default command: when the first argument is not a subcommand,
``run`` is inserted. Every flag of design section 0.2 is accepted and stored in
:class:`~datanonymous.config.Settings`; flags whose feature belongs to a later
area are passed to that area's hook. Column flags are repeatable (no comma
lists) and accept ``path::COL`` / ``book.xlsx::Sheet::COL`` qualifiers.

Exit codes: 0 every tabular file written (PASS or WARN); 1 unexpected error;
2 usage/config error; 3 one or more files FAILED (not written; in sharing mode the
others are still written unless ``--all-or-nothing``).
Logging goes to stderr; stdout carries the summary (or JSON with ``--json``).
Console output is ASCII only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import click

from . import __version__, diagnostics
from .config import build_settings, load_config
from .errors import DatanonymousError, UsageError

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_FAILED = 0, 1, 2, 3


#: The arguments of the current invocation: a usage error that repeats one is screened (D34, :func:`_screened`).
_ARGV: list[str] = []


class DefaultGroup(click.Group):
    """``datanonymous INPUT ...`` behaves like ``datanonymous run INPUT ...``. Click's own usage errors (a bad
    ``--header-row`` value) are screened like the tool's before they are printed (D34)."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in ("--help", "-h", "--version"):
            args = ["run", *args]
        return super().parse_args(ctx, args)

    def main(self, args: Any = None, prog_name: str | None = None, complete_var: str | None = None,
             standalone_mode: bool = True, **extra: Any) -> Any:
        global _ARGV
        _ARGV = [str(a) for a in (sys.argv[1:] if args is None else args)]
        if not standalone_mode:
            return super().main(args, prog_name, complete_var, standalone_mode, **extra)
        try:
            rv = super().main(args, prog_name, complete_var, standalone_mode=False, **extra)
        except click.exceptions.Abort:
            click.echo("Aborted!", err=True)
            sys.exit(EXIT_ERROR)
        except click.ClickException as e:
            raw = e.format_message
            e.format_message = lambda: _screened(raw())  # type: ignore[method-assign]
            e.show()
            sys.exit(e.exit_code)
        except SystemExit as e:
            if e.code is None or isinstance(e.code, int):
                raise
            sys.exit(EXIT_ERROR)  # Python would print a non-numeric exit code (a library's text): never (R7-04)
        except Exception as e:  # noqa: BLE001 - never Python's own traceback, which prints the messages (R7-04)
            _unexpected(e, _verbose_argv())
            sys.exit(EXIT_ERROR)
        sys.exit(rv if isinstance(rv, int) else EXIT_OK)


def _echo(text: str, err: bool = False) -> None:
    click.echo(text.encode("ascii", "backslashreplace").decode("ascii"), err=err)


def common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Options shared by ``run`` and ``inspect`` (design 0.2)."""
    opts = [
        click.option("--mode", type=click.Choice(["sharing", "fidelity"]), default=None,
                     help="Sharing applies release safeguards (default); fidelity is for invented data kept locally."),
        # types
        click.option("--factor-max", type=click.IntRange(1), default=None, help="Max distinct values for a text factor (default 20)."),
        click.option("--force-factor", multiple=True, metavar="COL", help="Keep the column's real levels as a factor."),
        click.option("--force-text", multiple=True, metavar="COL", help="Treat the column as non-factor text."),
        click.option("--force-type", multiple=True, metavar="COL=TYPE", help="Force a column type (catalog name)."),
        click.option("--pii-col", multiple=True, metavar="COL=SUBTYPE", help="Mark a column as PII of a subtype."),
        click.option("--dayfirst", multiple=True, metavar="COL", help="Ambiguous dates are day-first."),
        click.option("--date-cols", multiple=True, metavar="COL", help="Allow date-as-number detection for the column."),
        click.option("--date-format", multiple=True, metavar="COL=FMT", help="Explicit date format for a column."),
        click.option("--calendar", multiple=True, metavar="COL",
                     help="Calendar-like date column: draw only observed dates (never for DOB)."),
        click.option("--na-value", multiple=True, metavar="TOKEN", help="Extra missing-value token (case-sensitive)."),
        click.option("--decimal", type=click.Choice([".", ","]), default=None, help="Decimal mark for numbers."),
        # IDs and codes
        click.option("--id-col", multiple=True, metavar="COL", help="Person/record ID column (synthesised)."),
        click.option("--no-default-ids", is_flag=True, default=False, help="Do not use the default ID name list."),
        click.option("--org-code", multiple=True, metavar="COL", help="Organisational code: keep real values."),
        click.option("--person-id", multiple=True, metavar="COL", help="Person ID: synthesise even if org-code-like."),
        click.option("--id-check", type=click.Choice(["luhn"]), default=None,
                     help="Give synthetic digit IDs a valid check digit (when every real ID has one)."),
        # floors
        click.option("--no-exact-extremes", multiple=True, metavar="COL", help="Do not force the exact min/max."),
        click.option("--min-cell", type=click.IntRange(1), default=None, help="Small-cell threshold k (default 10)."),
        # columns
        click.option("--null", "null", multiple=True, metavar="COL", help="Write the column as all-missing."),
        click.option("--exclude", multiple=True, metavar="COL", help="Drop the column (schema changes)."),
        # input
        click.option("--encoding", default=None, help="Input encoding (Python codec name)."),
        click.option("--delimiter", default=None, help="Input delimiter (single character, or 'tab')."),
        click.option("--sheet", default=None, metavar="NAME", help="Excel sheet to read."),
        click.option("--header-row", type=click.IntRange(1), default=None, metavar="N", help="Excel header row."),
        click.option("--range", "range", multiple=True, metavar="SHEET!A5:H200",
                     help="Excel table region of a sheet (header = its first row)."),
        click.option("--include", multiple=True, metavar="GLOB", help="Only files matching the glob."),
        click.option("--exclude-path", multiple=True, metavar="GLOB", help="Skip files matching the glob."),
        click.option("--keep-name", multiple=True, metavar="TEXT",
                     help="A header, sheet, file or folder name flagged as personal data that is not (kept as is)."),
        # run
        click.option("--seed", type=click.IntRange(0, 2**63 - 1), default=None, help="Random seed (random if absent)."),
        click.option("--config", "config_path", type=click.Path(dir_okay=False), default=None, metavar="FILE.yaml",
                     help="Standing settings (YAML)."),
        click.option("-v", "--verbose", count=True, help="More detail on stderr."),
        click.option("-q", "--quiet", is_flag=True, default=False, help="Only errors."),
        click.option("--json", "json_output", is_flag=True, default=False, help="Print the summary as JSON."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _settings(kw: dict[str, Any]):  # noqa: ANN202
    config = load_config(kw["config_path"]) if kw.get("config_path") else {}
    cli = dict(kw)
    cli["verbosity"] = -1 if cli.pop("quiet", False) else cli.pop("verbose", 0)
    cli.pop("verbose", None)
    return build_settings(cli, config)


def _screened(text: str) -> str:
    """A usage or run error printed before any report exists: a quoted name holding personal data (a sheet or
    range typed with ``--sheet``/``--range``, a column reference) is never printed (D34)."""
    from .privacy.headers import redact_message

    return redact_message(text, typed=_ARGV)


def _unexpected(e: BaseException, verbose: bool) -> None:
    """An unexpected error: its class, and with ``-v`` a traceback of classes and code locations only. Never its
    message, arguments, notes or chained exceptions' text: they may quote a source value (R7-04)."""
    _echo(f"datanonymous: unexpected error: {type(e).__name__} (re-run with -v for the traceback)", err=True)
    if verbose:
        _echo(diagnostics.safe_traceback(e), err=True)


def _verbose_argv() -> bool:
    """``-v``/``-vv``/``--verbose`` among the arguments (for an error raised outside a command)."""
    return any(a == "--verbose" or (len(a) >= 2 and a[0] == "-" and set(a[1:]) == {"v"}) for a in _ARGV)


def _run_guarded(fn: Callable[[], int], verbose: bool) -> None:
    try:
        # a library's warning (or an error Python ignores) is printed with -v only, never with its text (R7-04)
        with diagnostics.quiet_diagnostics((lambda line: _echo(line, err=True)) if verbose else None):
            code = fn()
    except UsageError as e:
        _echo(_screened(f"datanonymous: error: {e.message}" + (f"\n  fix: {e.fix}" if e.fix else "")), err=True)
        sys.exit(EXIT_USAGE)
    except DatanonymousError as e:
        _echo(_screened(f"datanonymous: error: {e.message}" + (f"\n  fix: {e.fix}" if e.fix else "")), err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:  # noqa: BLE001
        _unexpected(e, verbose)
        sys.exit(EXIT_ERROR)
    sys.exit(code)


@click.group(cls=DefaultGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="datanonymous")
def main() -> None:
    """Profile tabular files and write shape-faithful synthetic copies."""


@main.command("run")
@click.argument("input", type=click.Path(path_type=Path))
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Folder for the synthetic files (+ README, DATA_DICTIONARY, schema). Must be a new or empty "
                   "folder outside the input and the report folder (it is created if missing).")
@click.option("--report-dir", "report_dir", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Folder for the report, kept locally (default: <out-dir>-report). A new or empty folder; it may "
                   "be inside the input.")
@click.option("--force", is_flag=True, default=False,
              help="Fidelity mode only: write into non-empty out/report folders.")
@click.option("--all-or-nothing", "all_or_nothing", is_flag=True, default=False,
              help="Sharing mode: write nothing unless every input file can be written (default: leave out the "
                   "files that cannot and list them in <report-dir>/sharing-failures.csv).")
@click.option("--refuse-unsupported", "refuse_unsupported", is_flag=True, default=False,
              help="Sharing mode: leave a file out when one of its non-key columns cannot be protected (default: "
                   "write that column with every cell empty and list it in the report).")
@click.option("--max-row-matches", type=click.FloatRange(0, 100), default=None, metavar="PCT",
              help="Fail a file whose remaining chance row matches exceed PCT percent.")
@click.option("--no-strict", is_flag=True, default=False, help="Layout self-check differences only warn.")
@click.option("--keep-titles", is_flag=True, default=False,
              help="Copy Excel title/notes rows and CSV footer notes verbatim (default: same-shape placeholders).")
@common_options
def run_cmd(input: Path, out_dir: Path, report_dir: Path | None, **kw: Any) -> None:
    """Profile INPUT (file or directory) and write synthetic copies to --out-dir."""
    verbose = bool(kw.get("verbose"))

    def go() -> int:
        from . import pipeline
        from .report.report import console_summary, finite_json, redact_console, run_json

        settings = _settings(kw)
        result = pipeline.execute(settings, input, out_dir, report_dir, mode="run")
        if settings.json_output:
            click.echo(json.dumps(finite_json(run_json(result)), indent=2, ensure_ascii=True, allow_nan=False))
        elif settings.verbosity >= 0:
            for line in console_summary(result):
                _echo(line)
        if verbose:
            for d in result.debug:
                _echo(redact_console(result, d), err=True)
        return pipeline.exit_code(result)

    _run_guarded(go, verbose)


@main.command("inspect")
@click.argument("input", type=click.Path(path_type=Path))
@common_options
def inspect_cmd(input: Path, **kw: Any) -> None:
    """Preview detection for INPUT (types, IDs, org codes, constants, structure, warnings); writes nothing."""
    verbose = bool(kw.get("verbose"))

    def go() -> int:
        from . import pipeline
        from .report.report import file_json, finite_json, inspect_lines, redact_console, redact_console_obj

        settings = _settings(kw)
        result = pipeline.execute(settings, input, mode="inspect")
        if settings.json_output:
            payload = {"files": [file_json(f) for f in result.files],
                       "skipped": [{"path": s.rel_path, "reason": s.reason} for s in result.skipped],
                       "messages": [m.to_public() for m in result.messages]}
            click.echo(json.dumps(finite_json(redact_console_obj(result, payload)), indent=2, ensure_ascii=True,
                                  default=str, allow_nan=False))
        else:
            for line in inspect_lines(result, settings.verbosity if settings.verbosity > 0 else 0):
                _echo(line)
        if verbose:
            for d in result.debug:
                _echo(redact_console(result, d), err=True)
        return pipeline.exit_code(result)

    _run_guarded(go, verbose)


if __name__ == "__main__":  # pragma: no cover
    main()
