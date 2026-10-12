# Production code compatibility

The public [consumer example](../tests/release_workflow.py) runs the same CSV
reader, student/state-alias joins, family checks, academic-date grouping,
service summaries, and chart-data preparation against the independently
invented canonical corpus and its sharing replacement. It uses Python's
standard library. From the repository root:

```bash
python tests/release_workflow.py path-to-invented-original
python tests/release_workflow.py path-to-sharing-copy --join-key State_Student_ID
python -m pytest -q tests/test_release_workflow.py
```

The [fixture recipe](../tests/release_corpus.py) contains no student records or
source-derived fixture. Tests generate fresh input, sharing output, and local
reports. The example expects the four CSV tables and named columns in that
recipe; it is a sample consumer contract, rather than a universal district
schema. It checks required headers, typed numbers and dates, unique roster
keys, many-to-one fact joins, matching aliases/families, and student/date
composites before producing summaries. Additional columns are accepted.

The small program codebook is independently invented public example content.
Known programs route to declared groups; additional nonempty categories route
to `Other`, and blank categories route to `Missing`. The consumer keeps their
rows in totals. It never builds permanent enums or filters from observed fake
levels. Use a reviewed non-identifying schema/codebook for an actual district,
and invent additional test cases independently. Do not export a private source
category list merely to improve the fake copy.

The regressions cover a valid invented category absent from the canonical
copy, and a rare `Speech` fact suppressed by sharing despite its common
presence in the roster. The suppression case gives common programs both
service frequencies; an unsupported source dependency must be refused rather
than regenerated in a way that restores the rare label. Missing measures keep
their records in counts and are excluded from means using an explicit measured
row denominator.

`SchoolCalendar` is an explicit consumer policy: by default the academic year
starts in September, Spring is January–May, Summer June–August, and Fall
September–December. Its year boundary, month partition, and institution
timezone are configurable in Python; the CLI also accepts
`--year-start-month`. Date-only strings represent local calendar dates;
timestamps require an offset and are converted to the institution timezone
before grouping. Tests check summer/fall/spring summaries after sharing,
year boundaries, and a timezone conversion across midnight. This reader policy
does not add a generator guarantee tying an independently generated term field
to a date, or retaining an original timeline. Declared ordering/duration rules
and any district-specific term constraints need their own generation and
consumer checks.

Chart data uses ordinary labelled counts suitable for a bar chart. Tests
recompute those counts from parsed records, verify total conservation, and
check known invented summary answers. No graphical artifact is required to
verify those logical inputs. Schema compatibility, valid joins, numerical
correctness, and temporal grouping are assessed separately from visual
resemblance. Synthetic subgroup counts, correlations, means, and plots need
not equal the original, and cannot support decisions about real students.

This workflow tests CSV ingestion. The format support and release validation
of other readers are separate evidence; do not infer that this CSV consumer
tested every supported input/output format.
