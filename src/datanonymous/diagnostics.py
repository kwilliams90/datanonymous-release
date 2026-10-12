"""Diagnostics that never print a source value (R7-04).

An exception's message and arguments may quote a real cell value (``could not convert string to float: '<the
cell>'``, a parser quoting the row it stopped on), and so may the exceptions chained to it, its notes, its frames'
local variables and a library's warning text. A read error happens before any sensitive-value index exists, so
nothing could redact such a value afterwards. The rule is therefore structural, not a redaction: everything the
tool prints about an exception or a warning is built from CODE facts only, whoever raised it:

* :func:`safe_traceback` (``-v`` tracebacks of internal errors: ``pipeline``, the leak scan, the CLI's top-level
  handler): per exception of the chain (cause, context, exception-group members) its class, and per frame the
  code file, line number and function name. Never a message, an argument, a note, a local variable or a source
  line;
* :func:`error_place` (the FAIL reason of an internal error): ``ValueError at _reader.py:83``;
* :func:`quiet_diagnostics` (around a CLI command): a warning of a library is printed only with ``-v``, as its
  category and code location (:func:`warning_text`), never its message; an error Python would print and ignore (in
  a finalizer or a thread) likewise, as :func:`safe_traceback`.

A tool error (``errors.DatanonymousError``) carries the tool's own text (its message and fix), which the CLI and the
report print as such; its message never quotes a foreign exception's text either.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import traceback
import warnings
from types import TracebackType
from typing import Callable, Iterator

#: File suffixes of code: a frame or warning location is printed only when it is a code file (a module path).
_CODE_SUFFIXES = (".py", ".pyw", ".pyx", ".pxd", ".pxi")
#: How deep exception groups and chains are followed (a cycle is cut by identity anyway).
_MAX_DEPTH = 8


def exception_class(exc: BaseException) -> str:
    """The exception's class as code names it (``ValueError``, ``pyarrow.lib.ArrowInvalid``): never its text."""
    t = type(exc)
    mod = getattr(t, "__module__", "") or ""
    name = getattr(t, "__qualname__", "") or t.__name__
    return name if mod in ("builtins", "__main__", "") else f"{mod}.{name}"


def code_file(filename: str) -> str:
    """A code file as its import path (``datanonymous/io/excel_read.py``, ``openpyxl/worksheet/_reader.py``): the
    path below the longest ``sys.path`` entry holding it, else its base name; ``<code>`` for anything that is no
    code file."""
    name = str(filename or "")
    if not name.casefold().endswith(_CODE_SUFFIXES):
        return "<code>"
    try:
        full = os.path.abspath(name)
    except (OSError, ValueError):
        return os.path.basename(name)
    key = os.path.normcase(full)
    best = 0
    for entry in sys.path:
        if not isinstance(entry, str) or not entry:
            continue
        try:
            base = os.path.normcase(os.path.abspath(entry)).rstrip("\\/")
        except (OSError, ValueError):
            continue
        if len(base) > best and key.startswith(base) and key[len(base):len(base) + 1] in ("\\", "/"):
            best = len(base)
    rel = full[best + 1:] if best else os.path.basename(full)
    return rel.replace("\\", "/")


def _frames(tb: TracebackType | None) -> list[tuple[str, int, str]]:
    """(code file, line, function) of every frame of ``tb``, read from the code objects (no source line, no local
    variable is looked at)."""
    out: list[tuple[str, int, str]] = []
    for frame, lineno in traceback.walk_tb(tb):
        code = frame.f_code
        out.append((code.co_filename, int(lineno or 0), code.co_name))
    return out


def error_place(exc: BaseException) -> str:
    """``ValueError at _reader.py:83``: the error's class and the code line that raised it (never its message)."""
    frames = _frames(exc.__traceback__)
    if not frames:
        return f"{type(exc).__name__} at ?"
    file, line, _ = frames[-1]
    base = code_file(file).rsplit("/", 1)[-1]
    return f"{type(exc).__name__} at {base}:{line}"


def _one(exc: BaseException, indent: str, depth: int, seen: set[int]) -> list[str]:
    lines = [f"{indent}Traceback (most recent call last; messages, arguments, notes and variables are not printed):"]
    frames = _frames(exc.__traceback__)
    k = 0
    while k < len(frames):  # a frame repeated in a row (a recursion) is printed once, with its count
        file, line, func = frames[k]
        n = 1
        while k + n < len(frames) and frames[k + n] == frames[k]:
            n += 1
        lines.append(f"{indent}  {code_file(file)}:{line} in {func}")
        if n > 1:
            lines.append(f"{indent}  [the frame above repeated {n - 1} more time(s)]")
        k += n
    lines.append(f"{indent}{exception_class(exc)}")
    members = getattr(exc, "exceptions", None) if isinstance(exc, BaseExceptionGroup) else None
    if members and depth < _MAX_DEPTH:
        for k, sub in enumerate(members, 1):
            if not isinstance(sub, BaseException):
                continue
            lines.append(f"{indent}  +-- member {k} of {len(members)}:")
            lines += _chain(sub, indent + "  | ", depth + 1, seen)
    return lines


def _chain(exc: BaseException, indent: str, depth: int, seen: set[int]) -> list[str]:
    """``exc`` with its cause or context first, the way Python prints a chain (oldest first)."""
    if id(exc) in seen:
        return [f"{indent}(the same {exception_class(exc)} again)"]
    seen.add(id(exc))
    lines: list[str] = []
    if depth < _MAX_DEPTH:
        if exc.__cause__ is not None:
            lines += _chain(exc.__cause__, indent, depth + 1, seen)
            lines += ["", f"{indent}The above exception was the direct cause of the following exception:", ""]
        elif exc.__context__ is not None and not exc.__suppress_context__:
            lines += _chain(exc.__context__, indent, depth + 1, seen)
            lines += ["", f"{indent}During handling of the above exception, another exception occurred:", ""]
    return lines + _one(exc, indent, depth, seen)


def safe_traceback(exc: BaseException) -> str:
    """A traceback of ``exc`` (and the exceptions chained to it) that holds no value: the class of each exception
    and the code file, line and function of each frame (module docstring)."""
    try:
        return "\n".join(_chain(exc, "", 0, set())) + "\n"
    except Exception:  # noqa: BLE001 - a diagnostic never fails the run, and never falls back to the message
        return f"{type(exc).__name__} (traceback unavailable)\n"


def warning_text(category: type, filename: str, lineno: int) -> str:
    """A warning as its category and code location: ``warning: UserWarning at openpyxl/reader/excel.py:12``."""
    name = getattr(category, "__name__", "Warning")
    return f"warning: {name} at {code_file(filename)}:{lineno} (message not printed)"


@contextlib.contextmanager
def quiet_diagnostics(echo: Callable[[str], None] | None) -> Iterator[None]:
    """Within the block, Python itself never prints an exception's or a warning's text either: a warning goes to
    ``echo`` as :func:`warning_text` (``-v``) or nowhere (``echo`` None; warning filters are left as they are, so an
    ``error`` filter still raises), and so does an error Python would report and ignore (raised in a finalizer or a
    thread: ``sys.unraisablehook``, ``threading.excepthook``), as :func:`safe_traceback`."""
    def show(message: object, category: type, filename: str, lineno: int, file: object = None,
             line: str | None = None) -> None:
        if echo is not None:
            echo(warning_text(category, filename, lineno))

    def ignored(exc: BaseException | None) -> None:
        if echo is not None and exc is not None:
            echo("an error was ignored (in a finalizer or a thread):\n" + safe_traceback(exc))

    old_unraisable, old_thread = sys.unraisablehook, threading.excepthook
    sys.unraisablehook = lambda u: ignored(u.exc_value)
    threading.excepthook = lambda a: ignored(a.exc_value)
    try:
        with warnings.catch_warnings():
            warnings.showwarning = show
            yield
    finally:
        sys.unraisablehook, threading.excepthook = old_unraisable, old_thread


__all__ = ["code_file", "error_place", "exception_class", "quiet_diagnostics", "safe_traceback", "warning_text"]
