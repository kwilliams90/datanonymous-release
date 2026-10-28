# Threat model

The protected information is student identity and information that links a released record or fact back to a source person. Sensitive membership and attributes can be exposed through joins, summaries, and relationships even when direct identifiers are replaced. The [sharing policy](sharing-policy.md) defines the required transformations; their implementation and acceptance evidence are in progress.

## What an observer can access

Assume the generated package, its documentation, public source code, algorithms, and published examples are available to the observer. Plausible auxiliary information includes school rosters, public event reports, school and course codes, academic calendars, demographic information, and earlier related disclosures. Do not rely on hidden code, recipient goodwill, or the destruction of an in-memory mapping to prevent identification.

FERPA's disclosure determination considers multiple releases and other reasonably available information. The school-community observer lacks personal knowledge of the relevant circumstances; the project also uses stronger invented-data attacks with the original available to uncover structural weaknesses. [Department of Education FERPA regulations](https://studentprivacy.ed.gov/ferpa)

## Disclosure paths

| Released feature | Plausible disclosure | Required evaluation |
| --- | --- | --- |
| Values, free text, paths, headers, titles, and metadata | Names or identifiers survive outside ordinary data cells. | Scan the complete package and explicitly review schema and names. Include unusual spellings, encoded values, and container metadata. |
| Person row counts across tables | A unique attendance, discipline, or assessment count identifies the donor after ID replacement. | Test individual counts and combined count vectors. Rebalance a rare count class into held classes, or leave the table out; independently reassigning it is insufficient. |
| File membership and relationships | A rare program table or family graph reveals an original person's membership. | Test fictional membership and regenerated relationships, including secondary IDs, aliases, nesting, and many-to-many joins. |
| Categories, constants, and missingness | A rare label or exact frequency identifies a group or sensitive fact. | Count distinct primary people per fact table and assess totals, complements, rare vocabulary, and patterns together. |
| Dates, numeric extrema, and combinations | A well-publicized event, oldest student, or unusual score links a row. | Exclude person-date source extremes and test partial matches, timelines, retained numeric facts, and public calendars. |
| Approved joins and dependencies | Safe-looking fields combine into an identifying fingerprint. | Test only documented structures survive and assess joined combinations of school, grade, program, geography, and demographics. |
| Prior copies, reports, and derived outputs | Several releases reveal suppressed facts or enable reconstruction. | Review related releases locally, including agent transcripts and charts; the tool has no automated combined-release assessment. |

Replacing names can also mislead a reader into attributing a synthetic sensitive fact to an actual person. Generated names and ordinary phone numbers are not claims about their real-world owners. Synthetic data must be labelled and must not be used to contact people or make student-level decisions.

## Boundaries and evidence

Original inputs, source profiles, comparison reports, mappings, diagnostics, and backups stay in the authorized local environment. Coding agents receive only the separately reviewed package. Local processing narrows exposure but does not secure the host, validate provider behavior, or audit dependency network access. See [operations](operations.md) and [security](../SECURITY.md).

Single-package evidence must exercise direct identifiers, partial row matches, unique donor counts, sensitive membership, secondary relationships, joined indirect identifiers, extrema, metadata, stale artifacts, and unsafe overrides. Tests must record a false-match baseline and explain any apparent match. No complete-row match does not establish protection from partial linkage. Any demonstrable identity or source-donor link in the canonical invented-data attack blocks release.

The separate certainty assessment and owner judgment described in the [sharing policy](sharing-policy.md) remain part of acceptance. These exercises provide bounded technical evidence. They do not establish safety for unknown auxiliary information, future releases, or every institution's dataset.
