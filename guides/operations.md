# Operating and adopting datanonymous

Run datanonymous inside the authorized local environment and give coding agents only a separately reviewed synthetic package. Keep the real source, source-derived reports, configuration, and temporary artifacts outside the agent workspace. The [sharing policy](sharing-policy.md) defines the implemented safeguards and required disclosure review. Fidelity mode is for invented-data work kept locally.

## Prepare the local run

1. Confirm authorized source access and identify the institution's disclosure decision owner, intended coding task, and proposed recipient. Determine applicable state, local, and IDEA requirements with the institution's responsible staff. Provider terms, permitted use, redisclosure, retention, and deletion belong to that deployment. [Department of Education third-party provider guidance](https://studentprivacy.ed.gov/sites/default/files/resource_document/file/Vendor%20FAQ.pdf)
2. Inventory the source tables, workbook sheets, formats, identifiers, secondary person relationships, keys, dates, sensitive categories, and paths. Review `inspect` results locally. Use `--person-id` to identify the subject's key when inference is ambiguous; no recognized ID does not mean no people are present. Sharing refuses forced copying overrides such as `--org-code`; resolve classification without weakening the sharing policy.
3. Review filenames, directory names, headers, sheet names, title rows, and metadata for identifying content. Reshape or safely rename identifying schema before generation, recording production-code changes locally. Exclude unsupported artifacts deliberately.
4. Install and record dependencies before working with records. Verify a representative offline run with invented inputs on the intended host. Use separate source, fresh output, and report directories with restricted access; do not place output within the source or reports.

For an invented-input trial, the existing command is:

```powershell
datanonymous inspect .\invented-input --mode fidelity
datanonymous .\invented-input --mode fidelity --out-dir .\trial-output --report-dir .\private-trial-report
```

These commands exercise fidelity on invented records. For source-derived work use the default sharing mode, a fresh output path, and the supported person and join contracts. Review the installed release's help and evidence. Do not bypass a privacy refusal by switching modes.

## Review and transfer

Treat failures, incomplete directories, and unresolved privacy warnings as blocked release. Review other warnings against their utility criteria. Inspect every proposed artifact, including its metadata and documentation, and resolve skipped files. Validate intended readers, types, leading zeros, row totals, fictional joins, composite uniqueness, dates, and production handling of categories absent from the copy.

Assess direct and partial linkage, donor counts, rare membership, relationships, extrema, and joined fields with relevant public information and earlier releases. Keep source comparison reports local even if their direct identifiers are redacted: source-derived aggregates can disclose facts. The tool does not perform a combined-release review.

For a real-data deployment, the institution records its reasonable determination for the exact recipient and package. Keep the file inventory or cryptographic digests, tool and policy versions, dependency versions, configuration, utility results, disclosure assessment, considered prior releases, remediations, residual-risk rationale, decision, and retention plan in a restricted local record. Avoid identifiers in the approval summary. Reassess when inputs, settings, tool, recipient, or relevant outside information changes.

Transfer only the approved synthetic inventory and reviewed synthetic-derived documentation. Give the agent a separate workspace with no access to sources, reports, backups, temporary files, or revealing configuration. Review agent transcripts, generated reports, charts, and new exports under the same boundary. Do not attach original records to support tickets or model prompts.

Use stubbed delivery services when testing contact fields. Ordinary-looking synthetic phone numbers can belong to people outside the input; they are coding fixtures, not contact destinations.

## Retention and incidents

Define local access and retention for inputs, profiles, reports, temporary files, abandoned output, agent artifacts, and backups. Clean up according to the institution's approved procedures; ordinary deletion and Python exit do not securely erase storage or memory. Include swap, crash dumps, snapshots, and backup copies in the plan. [Department of Education data destruction guidance](https://studentprivacy.ed.gov/resources/best-practices-data-destruction)

Stop transfer if a package exposes identifying content or demonstrable source linkage. Isolate it, preserve restricted evidence, and use the institution's incident procedures. Report software defects using an independently invented reproduction and safe diagnostics as described in [SECURITY.md](../SECURITY.md).

These institutional procedures guide adoption. Obtaining an actual district approval, legal review, contract, or workplace signoff is not a prerequisite for the project's public software release.
