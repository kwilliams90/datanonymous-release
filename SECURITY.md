# Security and privacy

datanonymous processes sensitive inputs locally. Keep original records, profiles, source comparison reports, configuration containing identifying information, and failed or temporary outputs inside the authorized local environment. Give coding agents only a separately reviewed synthetic package. See the [operator guide](guides/operations.md), [sharing policy](guides/sharing-policy.md), and [limitations](guides/limitations.md).

## Reporting a problem

Report a reproducible software problem through the [repository issues](https://github.com/kwilliams90/datanonymous-release/issues) using wholly invented records. Include the affected version, operating system, file format, redacted command, expected behavior, and an invented example that demonstrates the failure. Describe whether it affects identifiers, linkage, metadata, package validation, or code compatibility.

Public issues are public. Do not attach student records, source reports, identifiers, production paths, seeds paired with real inputs, crash dumps, or screenshots containing them. The project has no published dedicated private security contact or guaranteed response time. If a report cannot be made safely with an invented example, keep the sensitive evidence local and request a private reporting channel without disclosing it.

## Local safeguards

Restrict access to source directories, reports, temporary files, backups, and abandoned runs. Use institution-managed storage and encryption where required by the deployment's policy. Install dependencies before processing records; verify offline execution on the intended host. Local processing does not audit the host, dependencies, synchronization clients, or agent permissions.

Python process exit does not securely erase data from memory, swap, crash dumps, storage snapshots, or backups. Apply the institution's retention and destruction procedures to those copies as well as the visible files. Ordinary deletion may leave recoverable data. [Department of Education data destruction guidance](https://studentprivacy.ed.gov/resources/best-practices-data-destruction)

## Responding to exposure

Stop generation and distribution when an identifying value, demonstrable link to a source person or donor, or unverified artifact is found. Isolate the affected package and preserve the minimum evidence needed in restricted storage. Record the version, configuration, affected artifact inventory, recipients, and timing without putting identifiers into public diagnostics.

Use the institution's incident contact and procedures for containment, recipient follow-up, retention, and any required notifications. Reproduce software defects with invented data for the project. A corrected rerun does not undo an earlier disclosure; reassess related releases before distributing a replacement.
