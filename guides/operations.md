# Operating and adopting datanonymous

Run datanonymous inside the authorized local environment and give coding agents only a separately reviewed synthetic package. Keep the real source, source-derived reports, configuration, and temporary artifacts outside the agent workspace. The [sharing policy](sharing-policy.md) defines the implemented safeguards and required disclosure review. Fidelity mode is for invented-data work kept locally.

## Prepare the local run

1. Confirm authorized source access and identify the institution's disclosure decision owner, intended coding task, and proposed recipient. Determine applicable state, local, and IDEA requirements with the institution's responsible staff. Provider terms, permitted use, redisclosure, retention, and deletion belong to that deployment. [Department of Education third-party provider guidance](https://studentprivacy.ed.gov/sites/default/files/resource_document/file/Vendor%20FAQ.pdf)
2. Inventory the source tables, workbook sheets, formats, identifiers, secondary person relationships, keys, dates, sensitive categories, and paths. Review `inspect` results locally. Use `--person-id` to identify the subject's key when inference is ambiguous; no recognized ID does not mean no people are present. Sharing refuses forced copying overrides such as `--org-code`; resolve classification without weakening the sharing policy.
3. Review filenames, directory names, headers, sheet names, title rows, and metadata for identifying content. Reshape or safely rename identifying schema before generation, recording production-code changes locally. Exclude unsupported artifacts deliberately.
4. Install and record dependencies before working with records. Verify a representative offline run with invented inputs on the intended host. Use separate source, output, and report directories with restricted access; the output folder must be new or empty and never within the source or reports. The report folder is the private report and is never shared; you can set standing folders once (see below).

For an invented-input trial, the existing command is:

```powershell
datanonymous inspect .\invented-input --mode fidelity
datanonymous .\invented-input --mode fidelity --out-dir .\trial-output --report-dir .\private-trial-report
```

These commands exercise fidelity on invented records. For source-derived work use the default sharing mode, a new or empty output folder, and the supported person and join contracts. Review the installed release's help and evidence. Do not bypass a privacy refusal by switching modes.

## Set your folders once

Instead of naming two new folders for every run, set standing roots in a config file and let each run create its own folders under them.

1. Create `config.yaml` in the `datanonymous` folder of your user config folder: `%APPDATA%\datanonymous\config.yaml` on Windows, `~/Library/Application Support/datanonymous/config.yaml` on macOS, and `$XDG_CONFIG_HOME/datanonymous/config.yaml` (or `~/.config/datanonymous/config.yaml`) on Linux. Setting `DATANONYMOUS_CONFIG_DIR` to a folder makes datanonymous read `config.yaml` from that folder instead. This file is loaded automatically; the run summary prints `Config loaded:` with its path (and the `--config` file's, if you passed one), so standing settings are never invisible.
2. Name two roots in it:

   ```yaml
   output_root: D:\datanonymous\shareable
   report_root: D:\datanonymous\private-reports
   ```

3. Run `datanonymous <input>` with no folder options. The run writes the synthetic files to `<output_root>/<input-name>-<YYYYMMDD-HHMMSS>/` and the private report to `<report_root>/<same name>/`. A name that is already taken in either root gets `-2`, `-3` and so on in both, so two runs in the same second do not collide, and the two folders of a run always share a name.

Precedence is defaults, then the user config, then `--config`, then the command line. An explicit `--out-dir` or `--report-dir` overrides the roots; without `output_root`, `--out-dir` is still required. Any other option can live in the config too, including `refuse_unsupported`.

The roots are checked so the private report cannot land beside shareable output by accident. `output_root` without `report_root` is refused. `report_root` alone is allowed; it only sets where reports go. Roots that are the same folder or sit inside one another, or inside the input folder, are refused, and the usual new-or-empty and not-inside-each-other checks apply to the folders a run resolves.

## Read the run summary and the private report

Unless you pass `-q` or `--json`, a run prints both destinations before it starts and again at the end, labelled `Shareable synthetic files:` (the output folder) and `PRIVATE report - do not share:` (the private report). Send only the output folder. `REPORT.md` opens with a `PRIVATE – DO NOT SHARE` banner, and the report folder holds `PRIVATE-DO-NOT-SHARE.txt` to warn anyone who opens it. Keep this folder local: it describes the source.

The console summary and `REPORT.md` use the same three headings, in this order:

- **What was generated:** the files written to the output folder, with their rows and columns, and one line counting routine skips. A routine skip is an input that is not a table format at all, such as a PDF, an image or a Word document; the line gives counts by type (for example, `4 non-table files skipped (3 .pdf, 1 .png)`). Inputs you left out with `--include` or `--exclude-path` are counted on their own line.
- **What changed:** the deliberate privacy changes made to each written file, per column, such as dates shifted with month patterns kept, rare categories merged into common ones, numeric extremes removed, rows-per-person counts rebalanced, ID or text formats written in a more common format, and Nulled columns. These are not problems; they are why the copy differs from the source. In the private report, they carry counts and say whether you or sharing mode nulled a column. The same list, without counts, reasons or source facts, is in the output folder's README for the recipient.
- **What needs attention:** the things you may need to act on, each in one plain sentence with a plain fix: files left out, skipped inputs that look like tables but could not be read (an actionable skip, as opposed to a routine one), run-level errors, columns that sharing mode nulled automatically, and warnings that block a file. A column nulled automatically appears here and under "What changed". It gives the reason, and for an ID-like column suggests declaring it with `--id-col`. If nothing needs action it says so.

The report's Appendix holds what used to open it: the verdict and criteria table, per-file checks, message codes, statistics, utility warnings and the full skip list. A code also appears next to the plain sentence in the report, so you can look it up. `sharing-failures.csv` in the report folder still lists every file left out, with its reason.

A rebalanced table may have more or fewer fictional people than the source, with the same row total; the report says by how many.

Exit codes are unchanged: 0 when no file was left out, including when files have Nulled columns, and 3 when a file was left out. With `--refuse-unsupported`, a file that holds a column sharing mode cannot protect is left out and the run exits 3.

## Review and transfer

Files that fail or carry unresolved privacy warnings are left out of the output folder and listed in the report folder's `sharing-failures.csv`; review that list (it can name real files, so keep it local) and decide whether the remaining files are enough, or run with `--all-or-nothing` to require every file. Review every Nulled column in the report's "What needs attention": decide whether an empty column is acceptable, declare the column as an ID if it is one, exclude it, or run with `--refuse-unsupported` if you would rather have the file left out than a column nulled for you. Review other warnings against their utility criteria. Inspect every proposed artifact, including its metadata and documentation, and resolve skipped files. Validate intended readers, types, leading zeros, row totals, fictional joins, composite uniqueness, dates, and production handling of categories absent from the copy.

Assess direct and partial linkage, donor counts, rare membership, relationships, extrema, and joined fields with relevant public information and earlier releases. Keep source comparison reports local even if their direct identifiers are redacted: source-derived aggregates can disclose facts. The tool does not perform a combined-release review.

For a real-data deployment, the institution records its reasonable determination for the exact recipient and package. Keep the file inventory or cryptographic digests, tool and policy versions, dependency versions, configuration, utility results, disclosure assessment, considered prior releases, remediations, residual-risk rationale, decision, and retention plan in a restricted local record. Avoid identifiers in the approval summary. Reassess when inputs, settings, tool, recipient, or relevant outside information changes.

Transfer only the approved synthetic inventory and reviewed synthetic-derived documentation. Give the agent a separate workspace with no access to sources, reports, backups, temporary files, or revealing configuration. Review agent transcripts, generated reports, charts, and new exports under the same boundary. Do not attach original records to support tickets or model prompts.

Use stubbed delivery services when testing contact fields. Ordinary-looking synthetic phone numbers can belong to people outside the input; they are coding fixtures, not contact destinations.

## Retention and incidents

Define local access and retention for inputs, profiles, reports, temporary files, abandoned output, agent artifacts, and backups. Clean up according to the institution's approved procedures; ordinary deletion and Python exit do not securely erase storage or memory. Include swap, crash dumps, snapshots, and backup copies in the plan. [Department of Education data destruction guidance](https://studentprivacy.ed.gov/resources/best-practices-data-destruction)

Stop transfer if a package exposes identifying content or demonstrable source linkage. Isolate it, preserve restricted evidence, and use the institution's incident procedures. Report software defects using an independently invented reproduction and safe diagnostics as described in [SECURITY.md](../SECURITY.md).

These institutional procedures guide adoption. Obtaining an actual district approval, legal review, contract, or workplace signoff is not a prerequisite for the project's public software release.
