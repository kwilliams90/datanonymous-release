# Release candidate 0.1.0rc3

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
