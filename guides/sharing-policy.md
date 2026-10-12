# Sharing policy

datanonymous is intended to provide synthetic files for coding work while original education records stay local. Privacy takes precedence over exact source fidelity. A generated copy requires review of its complete contents and the disclosure context before it is shared.

The requirements below define the implemented sharing release contract. Release acceptance requires independent adversarial evidence and the owner's certainty review. Do not use successful generation or a PASS status as permission to disclose source-derived files. Fidelity mode retains earlier behavior for invented-data work. See the [limitations](limitations.md).

## Disclosure target

The target is that a reasonable person in the school community, without personal knowledge of the circumstances, cannot identify a student with reasonable certainty. FERPA requires removal of personally identifiable information and a reasonable determination considering single or multiple releases and other reasonably available information. Recipient restrictions alone cannot establish that a copy is de-identified. [Department of Education FERPA regulations, sections 99.3 and 99.31(b)(1)](https://studentprivacy.ed.gov/ferpa)

The threshold of 10 distinct people is a project safeguard for individual facts, not a universal legal threshold or a guarantee for joined combinations. The Department does not prescribe a universal disclosure-avoidance method or threshold. [PTAC Disclosure Avoidance FAQ](https://studentprivacy.ed.gov/sites/default/files/resource_document/file/FAQs_disclosure_avoidance_0.pdf)

Fictional joins do not authorize tracking original students. FERPA's coded-record provision in section 99.31(b)(2) specifically concerns education research and imposes additional conditions; ordinary coding work cannot assume that provision applies. [Department of Education FERPA regulations](https://studentprivacy.ed.gov/ferpa)

## Sharing transformations

The sharing mode must be the default and must refuse options that weaken privacy, including forced copying of real factor values, retained titles or flagged names, and a lowered small-cell threshold. There is no policy-exception bypass. Fidelity mode is separately labelled for invented-data work.

Sharing must meet these conditions:

- Each table keeps its row total: N output equals N input. People, table membership, and person relationships are fictional. Keys retain valid joins within a run; aliases, nesting, and composite uniqueness remain valid among fictional entities.
- Each fact table independently reassigns primary-person row counts and equivalent one-to-one aliases, preserving the rows-per-person distribution where it is non-identifying. Secondary person relationships are regenerated rather than inheriting their source count fingerprints. If a unique count identifies its source donor, the count must be blurred or the package refused; merely assigning it to another ID is insufficient.
- Categorical facts held by fewer than 10 distinct people in a fact table lose their exact counts and rare real labels, including standard educational vocabulary. Merge, suppress, or blur such facts without removing table rows. Reviewed labels with sufficient holders may remain, but their combinations still require assessment.
- Dates attached to personal records exclude whole public calendar groups containing source extrema, with enough local distinct-person support for each revealed boundary group. Month and season classes remain where supported. Source-inferred weekdays and day grids are discarded; exact source-membership calendars and infeasible constraints are refused. Coarse endpoint intervals remain observable. Numeric minima and maxima may remain only when at least 10 distinct people in that fact table hold them (use a higher configured threshold when set). Rare interior discrete numeric values also lose their exact counts or require refusal.
- IDs whose values are exactly 1 through n keep that set with randomized assignment and consistent fictional joins. Other replacement identifiers must avoid source identifiers within their applicable namespace.
- Synthetic phones may look ordinary and use exchanges other than 555. No generated phone may match any source phone anywhere in the run, including alternate formatting or embedded numbers. Preserve useful formatting and area-code distributions only when collision-free generation permits it.
- Existing protections for names, email addresses, personal identifiers, addresses, free text, and metadata continue to apply. An unrecognized identifying value is a defect to remediate, not an approved exception.

Person support must describe the person who owns the attribute. Twenty students
sharing one parent's name do not supply twenty parent donors. Sharing currently
refuses recognizable intrinsic attributes of parents, guardians, teachers,
staff and other people when actor ownership has not been established. Null or
exclude those fields as directed. A subject-key flag alone cannot establish
ownership of another person's attributes.

The intended retained joint structures are fictional join coverage, one-to-one key aliases, nesting, composite-key uniqueness, regenerated within-person constants, approved dependencies and templates, safe missingness patterns, and declared temporal rules. They support code compatibility; source membership, original family or staff relationships, source timelines, and undocumented person fingerprints are not part of the contract.

## Package requirements

Use a fresh output directory outside the inputs and reports. Release only a validated inventory of synthetic tables and reviewed synthetic-derived documentation. Source profiles, mappings, assignment details, source comparison reports, temporary files, stale files, and unrelated artifacts are excluded. Read and validate every included artifact, including filenames, headers, sheet names, container metadata, workbook properties, value labels, and compressed members. Unsupported, unreadable, unverified, or unreviewed artifacts prevent release.

Unresolved privacy warnings block packaging. Other warnings must satisfy their stated utility or validation criteria. Generation failure or partial success cannot approve a directory. The operator must resolve skipped inputs explicitly and review schema and paths because automated name screening has gaps.

## Technical acceptance and deployment

Release evaluation uses a canonical wholly invented original. An adversary receives that original, its generated copy, and the public repository and attempts direct and partial linkage across files. Any demonstrable source identity or donor link blocks release and requires remediation. Coincidental equality must be distinguished from linkage using recorded attacks and false-match baselines.

A separate gpt-6-astra assessment receives the synthetic copy and repository and states what it can determine with certainty about the original. The owner judges those results against the reasonable-person target. This judgment remains required; automated checks do not replace it. Record versions, transformations, tested scenarios, failures, and uncertainty as technical evidence.

Public software release requires implementation, reproducible evidence, useful code contracts, and accurate documentation. It does not require a district customer, institutional legal approval, or vendor agreement. Institutions adopting the tool remain responsible for their own authorized access, recipients, applicable rules, and disclosure decisions. Follow the [operator guide](operations.md) for those deployment steps.
