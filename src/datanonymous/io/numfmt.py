"""Excel number formats without free text (privacy; D13, D20, D21).

An Excel number format is structural formatting (``#,##0.00``, ``mm/dd/yyyy``,
``[Red]-0``) plus optional **literal text**: quoted strings (``"Zelmira Quenby "0``),
backslash escapes (``\\Z\\e\\l``), letters outside quotes that are not format
codes, and the text of a ``[$text-LCID]`` currency tag. Literal text can quote
anything (a teacher's name, a phone number), and the workbook keeps it in
``xl/styles.xml``, not in a cell. Every format the tool reads (``.xlsx``,
``.xls``) or writes (``.xlsx`` cells, the ``mso-number-format`` CSS of an HTML
"Excel" file) goes through :func:`safe_number_format`, which rebuilds it from an
allow-list:

* kept as they are: digit placeholders ``0 # ?``, ``. , % / E+ E-``, the date
  and time codes (letters ``y m d h s e b g`` in any case), ``AM/PM``, ``A/P``,
  ``General``, ``@``, section separators ``;``, spacing/fill pairs ``_x``/``*x``
  (a letter or digit there becomes ``x``/``0``), and the bracket codes for
  colours, conditions (``[<=9999999]``), elapsed time (``[h]``), locales and
  numeral systems (``[$-409]``, ``[DBNum1]``); any other bracket is dropped;
* literal text keeps its punctuation, spacing and symbols (``$ € £ ¥ °``) and
  the words of :data:`SAFE_WORDS` (units, currencies, ordinal suffixes, yes/no
  words, CJK date units); every other letter becomes ``x``/``X`` (case kept)
  and every digit ``0``, so ``"Zelmira Quenby "0`` is written
  ``"Xxxxxxx Xxxxxx "0`` and ``0.0" kg"`` stays as it is. A word spelled over
  several escapes or quoted pieces is judged as one word. Letters outside
  quotes that are not format codes are masked the same way and quoted.

The result is deterministic and idempotent (``safe(safe(f)) == safe(f)``): the
self-check compares the re-read output format with the rebuilt source format,
the report never quotes a source literal, and the same seed gives
byte-identical files (D26).
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Literal words kept in number formats (casefolded): units of measure and time, multipliers, currencies,
#: ordinal suffixes, yes/no words and the CJK date/time units. Everything else literal is masked.
SAFE_WORDS = frozenset(w.casefold() for w in (
    # counts, shares and multipliers
    "pct", "percent", "pt", "pts", "point", "points", "x", "k", "m", "b", "bn", "mn", "mm", "mil", "mio", "mrd",
    "thousand", "thousands", "million", "millions", "billion", "billions", "trillion", "per", "of", "each", "ea",
    "unit", "units", "item", "items", "no", "nr", "n", "na", "q", "qtr", "fy", "approx", "ca",
    # measures
    "kg", "g", "mg", "lb", "lbs", "oz", "t", "ton", "tons", "tonne", "tonnes", "km", "cm", "mi", "ft", "in", "yd",
    "l", "ml", "dl", "cl", "gal", "qt", "sq", "deg", "c", "f", "kwh", "kw", "w", "v", "a", "mph", "kph", "kmh",
    # time
    "h", "hr", "hrs", "hour", "hours", "min", "mins", "minute", "minutes", "s", "sec", "secs", "second", "seconds",
    "ms", "d", "day", "days", "wk", "wks", "week", "weeks", "mo", "mos", "month", "months", "yr", "yrs", "year",
    "years", "am", "pm", "st", "nd", "rd", "th",
    # currencies
    "usd", "eur", "gbp", "jpy", "cny", "rmb", "inr", "cad", "aud", "nzd", "chf", "sek", "nok", "dkk", "mxn", "brl",
    "zar", "kes", "ngn", "ghs", "egp", "krw", "hkd", "sgd", "rub", "pln", "czk", "huf", "ils", "aed", "sar", "thb",
    "idr", "php", "myr", "vnd", "pkr", "bdt", "try", "kr", "zł", "zl", "kč", "lei", "lev", "rs", "rp", "r", "tl",
    "dr", "cr", "fr", "sfr", "dm", "rm", "ksh", "bs", "kn", "din", "лв", "руб", "р", "credit", "credits",
    # words of boolean-style sections ("Yes";"Yes";"No")
    "yes", "true", "false", "on", "off",
    # CJK date/time units and currencies (one character each)
    "年", "月", "日", "时", "時", "分", "秒", "号", "號", "元", "円", "圓", "년", "월", "일", "시", "분", "초", "원",
))

#: Letters that are format codes outside quotes (dates/times, scientific E, era/Buddhist years).
_CODE_LETTERS = frozenset("yYmMdDhHsSeEbBgG")
_WORD_RE = re.compile(r"AM/PM|A/P|General", re.I)
#: Kana, CJK ideographs and Hangul syllables: each character is a word of its own (\u5e74, \u6708, \u65e5 ...).
_CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_COLOR_RE = re.compile(r"(?:black|blue|cyan|green|magenta|red|white|yellow|colou?r\s*\d{1,2})", re.I)
_COND_RE = re.compile(r"(?:<=|>=|<>|<|>|=)\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?", re.I)
_ELAPSED_RE = re.compile(r"h+|m+|s+", re.I)
_SYSTEM_RE = re.compile(r"(?:dbnum|natnum)\d{1,2}|\$-x-sys(?:date|time)", re.I)
#: ``[$text-LCID]``: currency text, then a hex LCID (``-409``) or a language tag (``-en-US``, ``-zh-Hans-CN``).
#: (an LCID holds a digit: a name made of hex letters such as ``Abe`` is text, not a locale)
_LOCALE_RE = re.compile(r"\$(.*?)(-(?:(?=[0-9A-Fa-f]*\d)[0-9A-Fa-f]{1,8}|[a-z]{2,3}(?:-[A-Z][a-z]{3})?"
                        r"(?:-(?:[A-Z]{2}|\d{3}))?))?", re.S)


def _pieces(text: str) -> list[str]:
    """Literal text split into words (runs of letters; a CJK character alone), digits and other characters."""
    out: list[str] = []
    for ch in text:
        word = ch.isalpha() and not _CJK_RE.match(ch)
        if out and word and out[-1][-1].isalpha() and not _CJK_RE.match(out[-1][-1]):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def mask_literal(text: str) -> str:
    """Literal format text with every word outside :data:`SAFE_WORDS` masked (``x``/``X``, case kept) and
    every digit ``0`` (same length; punctuation, spacing and symbols such as ``$ € ° ²`` kept)."""
    parts = []
    for s in _pieces(text):
        if s.isalpha() and s.casefold() not in SAFE_WORDS:
            s = "".join("X" if ch.isupper() else "x" for ch in s)
        elif s.isdecimal():
            s = "0"
        parts.append(s)
    return "".join(parts)


def _mask_char(ch: str) -> str:
    if ch.isdecimal():
        return "0"
    if ch.isalpha():
        return "X" if ch.isupper() else "x"
    return ch


def _bracket(body: str) -> str:
    """A ``[...]`` code kept (colours, conditions, elapsed time, locales, numeral systems) or dropped ('')."""
    if _COLOR_RE.fullmatch(body) or _COND_RE.fullmatch(body) or _ELAPSED_RE.fullmatch(body) \
            or _SYSTEM_RE.fullmatch(body):
        return f"[{body}]"
    m = _LOCALE_RE.fullmatch(body)
    if m is not None:  # [$text-LCID]: the currency text is literal
        return f"[${mask_literal(m.group(1))}{m.group(2) or ''}]"
    return ""


def _tokens(fmt: str) -> list[list[str]]:
    """[kind, text] pieces: 'code' (kept), 'q' (quoted literal), 'e' (escaped character), 'u' (unquoted
    letters that are not format codes)."""
    out: list[list[str]] = []
    i, n = 0, len(fmt)
    while i < n:
        ch = fmt[i]
        if ch == '"':
            j = fmt.find('"', i + 1)
            j = n if j < 0 else j
            out.append(["q", fmt[i + 1:j]])
            i = j + 1
        elif ch == "\\":
            if i + 1 < n:
                out.append(["e", fmt[i + 1]])
            i += 2
        elif ch in "_*":
            out.append(["code", ch + (_mask_char(fmt[i + 1]) if i + 1 < n else "")])
            i += 2
        elif ch == "[":
            j = fmt.find("]", i + 1)
            if j < 0:
                break  # an unterminated bracket: the rest is dropped
            out.append(["code", _bracket(fmt[i + 1:j])])
            i = j + 1
        elif ch.isalpha():
            m = _WORD_RE.match(fmt, i)
            if m is not None:
                out.append(["code", m.group()])
                i = m.end()
                continue
            j = i + 1
            while j < n and fmt[j].isalpha() and _WORD_RE.match(fmt, j) is None:
                j += 1
            run = fmt[i:j]
            # date codes are one case per run (yyyymmdd, HH, E); a mixed-case run of code letters is a word (Bess)
            code = set(run) <= _CODE_LETTERS and (run.islower() or run.isupper())
            out.append(["code" if code else "u", run])
            i = j
        else:
            out.append(["code", ch])
            i += 1
    return out


@lru_cache(maxsize=4096)
def _safe(fmt: str) -> str:
    toks = _tokens(fmt)
    # literal pieces next to each other are one text: a word spelled over escapes/quotes is judged whole
    k = 0
    while k < len(toks):
        if toks[k][0] == "code":
            k += 1
            continue
        j = k
        while j < len(toks) and toks[j][0] != "code":
            j += 1
        masked = mask_literal("".join(t[1] for t in toks[k:j]))
        pos = 0
        for t in toks[k:j]:
            new = masked[pos:pos + len(t[1])]
            pos += len(t[1])
            if t[0] == "u" and new != t[1]:
                t[0] = "q"  # masked letters outside quotes are quoted (never read as date codes)
            t[1] = new
        k = j
    parts = []
    for kind, text in toks:
        parts.append(f'"{text}"' if kind == "q" else f"\\{text}" if kind == "e" else text)
    return "".join(parts)


def safe_number_format(fmt: str | None) -> str | None:
    """``fmt`` with its free text masked (see the module docstring); ``None``/``''``/``General`` unchanged."""
    if not fmt or not isinstance(fmt, str) or fmt == "General":
        return fmt
    return _safe(fmt)


# --------------------------------------------------------------------------- CSS (HTML "Excel" files)
_IMPORTANT_RE = re.compile(r"\s*!\s*important\s*$", re.I)
_CSS_ESCAPE_RE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|(.))", re.S)
#: Excel's named formats in ``mso-number-format`` (kept as they are).
_NAMED_FORMATS = frozenset(s.casefold() for s in (
    "General", "General Number", "General Date", "Long Date", "Medium Date", "Short Date", "Long Time",
    "Medium Time", "Short Time", "Currency", "Euro Currency", "Fixed", "Standard", "Percent", "Scientific",
    "Yes/No", "True/False", "On/Off"))


def _css_unescape(s: str) -> str:
    def one(m: re.Match) -> str:
        if m.group(1) is not None:
            cp = int(m.group(1), 16)
            return chr(cp) if 0 < cp <= 0x10FFFF else "�"
        return m.group(2)

    return _CSS_ESCAPE_RE.sub(one, s)


def _css_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', "\\0022 ").replace("\n", "\\a ") + '"'


def safe_css_number_format(value: str) -> str:
    """The value of an ``mso-number-format`` CSS declaration with its format's free text masked.

    The original text is returned when the format has no literal to mask (no output change);
    otherwise the masked format as a CSS string.
    """
    v = value.strip()
    important = _IMPORTANT_RE.search(v)
    if important is not None:
        v = v[:important.start()].rstrip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        fmt = _css_unescape(v[1:-1])
    else:
        fmt = _css_unescape(v)
    if fmt.strip().casefold() in _NAMED_FORMATS:
        return value
    safe = safe_number_format(fmt)
    if safe == fmt:
        return value
    return _css_string(safe or "") + (" !important" if important is not None else "")
