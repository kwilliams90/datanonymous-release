# Release candidate 0.1.0rc3

Sharing mode now tries to make each file work while still never releasing a
real value. The decisions are in `docs/adr/` and the terms in `GLOSSARY.md`.

Columns it cannot protect
- A non-key column that sharing mode cannot protect (for example a parent's
  education level, a column with too few people behind its values, a number
  that looks like an ID but is not used as a key) is written as a Nulled
  column: header and position kept, every cell empty. Before, one such column
  left the whole file out. The column is listed under "What needs attention"
  with a plain reason and, for an ID-like column, the suggestion to declare it
  with `--id-col`. The report tells columns you nulled with `--null` from
  columns nulled automatically.
- A blocking finding about one such column that appears only after the file is
  generated (a possible ID or code, a leak-scan hit in one column) nulls that
  column and generates the file again with the same seed. Findings about whole
  rows or the whole file still withhold the file, and so do findings on a key
  or on a column in a join, dependency, template, composite key or declared
  rule. If nulling does not clear the findings, the file is left out.
- A run whose only problems are automatically Nulled columns exits 0 and
  counts as written for `--all-or-nothing`.
- New `--refuse-unsupported` option (and config key `refuse_unsupported`):
  leave the file out instead, as in earlier releases (exit 3; with
  `--all-or-nothing`, nothing is written).

Fewer files and tables left out
- A table with a rows-per-person count class held by fewer than `--min-cell`
  people is rebalanced instead of left out: the rows of rare count classes move
  to the nearest classes that enough people hold, and the row total stays
  exact. The number of fictional people in the table may differ from the
  source; the private report says by how many. The table is still left out
  when no count class is held or the row total cannot be met exactly.
- A rare ID format class, including a minority layout in a mixed ID column, is
  written in the column's most common held format class, and joins stay valid.
  A column with no held format class, or made of e-mail addresses, is still
  refused.
- A rare text shape (such as `AA-0000`) is written without its leading-digit
  and length detail, or else in the column's most common held shape. A rare
  date format is written in the column's most common held date format. A shape
  defined by embedded personal information is not broadened: a non-key column
  like that is nulled.
- Currency, grouped, suffixed, parenthesised-negative and zero-scale decimal
  columns get the same rare-value suppression and extreme removal as plain
  integers, and are written in their own format, instead of refusing. A
  possible-ID number column that is not a key is nulled and the report suggests
  declaring it as an ID.

Reports and run folders
- `REPORT.md` and the console summary have the same three headings: "What was
  generated", "What changed" and "What needs attention". Every problem is a
  plain sentence with a plain fix; message codes, the verdict and criteria
  table, per-file checks, statistics and the full skip list moved to an
  Appendix, and the code stays next to the sentence it explains.
- Non-table inputs (PDFs, images, Word documents) are one routine line with
  counts by type. An input that looks like a table but could not be read is
  listed under "What needs attention".
- "What changed" lists the deliberate privacy changes per column (dates
  shifted, rare categories merged, numeric extremes removed, counts
  rebalanced, formats broadened, Nulled columns). The output folder's README
  gains a short "What changed" list without counts, reasons or source facts.
- A user config file (`config.yaml` in the `datanonymous` folder of your
  platform's user config folder, or in `DATANONYMOUS_CONFIG_DIR`) is loaded
  automatically. Precedence: defaults, user config, `--config`, command line.
  The run summary names the config file it loaded.
- New config keys `output_root` and `report_root`: a run without `--out-dir`
  writes to `<output_root>/<input-name>-<YYYYMMDD-HHMMSS>/` and its report to
  `<report_root>/<same name>/`, with `-2`, `-3` on a name collision.
  `--out-dir` and `--report-dir` override them. `output_root` without
  `report_root` is refused, as are roots that sit inside one another or inside
  the input.
- Both destinations are printed at the start and end of a run, labelled
  `Shareable synthetic files` and `PRIVATE report - do not share`. `REPORT.md`
  opens with a `PRIVATE – DO NOT SHARE` banner and the report folder holds a
  `PRIVATE-DO-NOT-SHARE.txt` marker (never written to the output folder).
- Every blocking or failure message has a plain-language entry in a message
  catalogue, enforced by a test.

Earlier rc3 changes

- A numeric ID column whose IDs differ in length (say 4 to 8 digits) no longer
  fails in sharing mode when some length is held by fewer than `--min-cell`
  IDs: those IDs get the nearest length that enough IDs hold.
- An ID-format refusal in sharing mode names the column.
- Three warnings that never carry source values no longer keep a file out of
  the sharing output: `id_width_changed` (fictional IDs needed another number
  of digits), `dtype_mismatch` (pandas reads an output column as another
  type, such as a column blanked with `--null`) and `factor_overflow` (a
  column has too many labels to keep, so it is written as generated text).
  They are still reported.
- When a file fails or is held back after the run has been planned (during
  generation or for its warnings), sharing mode plans and writes the other
  files again without it. Before, a roster of row numbers 1..n that failed
  this late still let the other tables keep those real numbers as IDs.
- `--all-or-nothing` no longer counts the files that `--include` or
  `--exclude-path` leave out as skipped inputs; an unsupported file in the
  selection still blocks the output.
- Sharing mode no longer suggests options it refuses (`--force-factor`,
  `--org-code`, `--keep-name`).
- A formatted number column with decimals, such as `$1,234.50` or `12.5%`, no
  longer stops sharing mode with an internal error.

# Release candidate 0.1.0rc2

- Sharing mode leaves out the input files it cannot write safely and writes the
  rest. The files left out are listed in `sharing-failures.csv` in the report
  folder (never in the output folder) and counted, without names, in the
  output's README; the run exits with code 3. `--all-or-nothing` restores the
  earlier behaviour of writing nothing unless every file succeeds.
- A key-population refusal in sharing mode (a rare rows-per-person count class,
  a rare identifier format) now leaves out only the tables it concerns; the
  other tables are planned again without them, dropping any join to a table
  left out.
- Row-number IDs (exactly 1 through n) whose full set belonged to a table that
  was left out are no longer kept as they are: the kept tables get fictional
  IDs for them, checked by the leak scan like any other identifier.
- `--out-dir` may be an existing empty folder; a folder with files in it is
  refused with a message saying so. The folder itself is kept (a mount point
  or a folder with its own permissions), and the files are copied into it when
  it is on another drive.
- The report folder may sit inside the input; the input scan skips it. It must
  be new or empty there, even with `--force`, so report files never replace
  input files.
- A declared rule or column option that matched nothing still blocks the whole
  output unless it was meant for a file that could not be read.

# Release candidate 0.1.0rc1

Sharing is now the default workflow. It generates fictional table memberships
and relationships, independently assigns supported primary-person counts, and
promotes a complete package only after verification. Existing output paths,
unsafe overrides, partial results and unresolved privacy warnings are refused.

Dates use protected public calendar groups and retain supported months and
seasons. Rare categories, missing spellings and donor-identifying patterns may
disappear or require refusal. Numeric extrema require at least ten local person
holders. These controls can change distributions and supported input layouts;
see the [sharing policy](sharing-policy.md) and [limitations](limitations.md).

Phone generation permits ordinary-looking numbers while excluding source
phones across the entire run, including embedded values and alternate formats.
The three public example CSVs and their demonstrations were replaced with
independently invented records and an explicit generation recipe. Public source
release starts with fresh history; earlier development history stays private.

The earlier generation behavior is available through `--mode fidelity` for
invented inputs kept locally. Existing scripts that depend on exact counts or
original relationship structure must choose that mode explicitly. Sharing
refusals require remediation rather than a fidelity-mode workaround.

Stage 1 improves profiling and generation performance while preserving the
earlier output bytes before these deliberate behavior changes. Further
performance stages remain outside this release candidate.

Software tests and adversarial assessments support the documented scope. They
do not establish that every generated dataset is de-identified. Review the
complete output, relevant outside information and related releases before
disclosure.
