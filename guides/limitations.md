# Limitations

datanonymous creates synthetic coding fixtures, not authoritative student records or analytical substitutes. It does not certify FERPA compliance. A successful run, a PASS status, no complete-row matches, or a small-cell setting of 10 cannot establish that a whole package is safe to disclose.

## Implementation status

The [sharing policy](sharing-policy.md) is implemented with adversarial regression coverage. Its release assessment remains subject to independent certainty review. Fidelity mode can retain exact category counts, person count vectors, numerical or date extrema, and original relationships; it also allows overrides and partial output. Use fidelity only with invented inputs kept locally.

Sharing mode first tries to make a file work: a column it cannot protect becomes a Nulled column, rare rows-per-person count classes are rebalanced into count classes that enough people hold, and rare identifier formats, text shapes and date formats are written in a broader format class that enough people hold. Formatted numbers (currency, grouping, suffixes, parenthesised negatives) get the same rare-value protection as plain numbers. These changes protect people at a cost in fidelity: a Nulled column is empty, a rebalanced table can have a different number of fictional people and a different rows-per-person distribution, and minority formats disappear into the common one. Do not build production code on distributions or formats seen only in the copy.

A table with no person column counts each row as a different person when nothing suggests that a person has several rows ([ADR-0006](../docs/adr/0006-rows-count-as-people-without-repeats.md)). A table that repeats people with no visible sign (no ID-like, name or e-mail column that repeats, or repeated names beside an e-mail that differs on every row) defeats this: a value one person holds on many rows can pass as common and be released. The private report lists every table where the assumption was used under "What needs attention"; if someone can have several rows there, name their ID column with `--person-id` and run again.

Sharing still leaves a file out when the problem concerns a key or a column in a relationship (a join, dependency, template, composite key or declared rule), because nulling it would break the joins the copy exists to keep. It also leaves out a file or table for missing or ambiguous orphan-key contracts, infeasible joins, an identifier column with no held format class or made of e-mail addresses, a table with no held rows-per-person count class or whose row total cannot be met exactly, and findings about whole rows or the whole file. `--refuse-unsupported` extends this to every column it would otherwise null. These refusals protect the supported contract; they can limit code compatibility. A column with repeated embedded personal identifiers across text contexts, mixed Excel storage formats or types, or a numerical or date representation with no safe supported result is nulled when it is not a key and not in a relationship. A saturated categorical domain no longer withholds a file by itself: whole-row matches no more common than shuffling the independently generated columns gives are written with a warning, and only matches above that baseline withhold the file. That baseline covers whole rows only, not attribute inference on parts of rows. A table left out takes its joins with it: a table that referred to it is written without that join.

Recognizable intrinsic attributes of other people require their own ownership
evidence. The current sharing implementation conservatively nulls such
parent, guardian, teacher and staff fields (leaving the file out when they are
keys or in a relationship, or under `--refuse-unsupported`), including a putative
staff roster, rather than counting student subjects as their donors. Ordinary student-record
comments or contact-event dates do not become actor attributes merely because
their headers contain a role word. Review their contents and the reported
classification; an unfamiliar header can escape this safeguard.

## Detection and disclosure

Type and identifier detection use heuristics. Unusual names, single surnames, non-Latin scripts, unexpected column names, embedded identifiers, and misleading organization-code labels can escape detection. Review paths, headers, sheets, constants, free text, and metadata explicitly. Common vocabulary and organizational codes can contribute to identification in combination; a sufficient holder count for one value is not a global anonymity result.

Exact total rows and useful structural contracts remain observable. A unique donor count, residual relationship graph, rare membership, numerical extreme, or joined set of indirect identifiers can disclose information without a full-row match. Documented retained structures still need disclosure testing. The tool does not provide differential privacy or a formal bound against arbitrary linkage or inference.

The date calendar cover can reveal coarse intervals containing original extrema. Its local distinct-person support check protects the revealed boundary group; it does not establish ten holders of the exact extreme or anonymity for joined attributes. Dates may move outside their original day bounds within the retained month, including onto other weekdays. Repeated releases or outside knowledge can narrow those intervals.

The tool does not check or establish protection across repeated releases. Different seeds, fresh mappings, independent reruns, source summaries, agent transcripts, generated charts, and other public information may combine to reveal facts that a single copy obscures. Unavailable prior releases and unknown outside information limit the assessment. Each institution must review related disclosures; a previous approval does not automatically cover a rerun.

## Code compatibility and realism

Rows and fields are regenerated; ordinary correlations, subgroup effects, original student trajectories, attendance spells, and family or staff relationships are not preserved as source facts. Fictional joins and declared rules support software development, but plausible charts do not validate an analysis. Test ingestion, schema, joins, missing values, and academic-date logic separately from visual resemblance.

Privacy transformations can remove categories and alter frequencies. Use a reviewed, non-identifying codebook when available and handle unknown valid categories in production. Do not build permanent filters or enums from only the values observed in the synthetic copy. Use independently invented fixtures to test rare or absent categories.

Format support does not mean every container feature is retained. Spreadsheet macros, unsupported attachments, complex nested data, and other skipped artifacts require explicit handling. Conversions and privacy-required schema changes must be recorded and tested with the intended production reader. An infeasible join, uniqueness, date, identifier-space, or privacy constraint requires refusal rather than silent weakening.

## Local and institutional boundaries

Processing is intended to stay local. This does not audit the host, dependencies, sync software, backups, swap, crash dumps, or a provider's use of received files. Process exit is not secure erasure. See the [operator guide](operations.md) and [security guidance](../SECURITY.md).

The institution determines authorized source use, the recipient, applicable requirements, retention, and the disclosure decision for its exact package. Project evidence supports that decision; the open-source project does not approve providers or administer institutional access.
