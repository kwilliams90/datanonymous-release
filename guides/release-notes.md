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
