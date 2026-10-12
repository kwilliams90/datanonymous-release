# Wholly invented example data

These three CSV files contain fictional teaching records generated from the
standalone recipe in [generate_examples.py](generate_examples.py). They are not
student records and were not produced by perturbing, sampling, shuffling,
redacting, or fitting a model to real people. No source data are inputs to this
recipe. A coincidental match to a real name or value is possible.

The historical filenames, column headers, row counts, and selected CSV layout
features remain for tutorial and test compatibility. Those choices do not imply
that the invented values describe the datasets formerly stored at these paths.
The replacement recipe and all row values were authored independently on
2026-10-11. Public release uses a clean source snapshot with fresh Git history.
Earlier development history, which contains the former files, stays private.

| File | Rows | Columns | Line endings | Independent recipe seed |
| --- | ---: | ---: | --- | --- |
| `StudentsPerformance.csv` | 1000 | 8 | CRLF | `invented-performance-v1-20261011` |
| `student-scores.csv` | 2000 | 17 | LF | `invented-scores-v1-20261011` |
| `students_data.csv` | 62 | 11 | LF | `invented-campus-v1-20261011` |

All files use UTF-8 without a BOM, comma delimiters, minimal CSV quoting, and a
final newline. `Central ` and `Eastern ` deliberately include a trailing space
so tutorials and tests can exercise faithful serialization.

Run this command from the repository root with Python 3.12 or later:

```sh
python example-data/generate_examples.py
```

The script uses only Python's standard library. It overwrites the three CSVs
next to itself, never reads their previous contents, and does not open or alter
other project files. Its SHA-256 counter draws, integer arithmetic, fixed seeds,
and explicit CSV line endings produce the same bytes across platforms. Change a
seed or a specification only when intentionally revising the public fixtures.

The recipe specifications are intentionally simple:

- `StudentsPerformance.csv`: equally selectable demographic and meal labels,
  standard parental-education vocabulary, and 0–100 integer marks drawn around
  an arbitrary shared baseline. Completed preparation adds five toy baseline
  points. Demographic labels do not determine marks. `race/ethnicity` contains
  generic group labels, not measured racial or ethnic membership.
- `student-scores.csv`: sequential row IDs from 1 to 2000, names assembled from
  newly authored syllable lists, and `first.last.id@example.invalid` email
  strings. The `.invalid` domain is for example addresses. Boolean labels,
  0–20 absence days, 0–30 study hours, and 17 independently chosen career labels
  exercise detection. Seven 0–100 marks share an invented baseline with toy
  study/absence effects; no parameters were learned from actual students.
- `students_data.csv`: four study years, invented course codes, 18–29 ages,
  145–195 heights, 45–95 weights, and generic region labels. Campus join year
  and prior exam year follow a fictional 2026 teaching calendar. Expense values
  are arbitrary integers in 18000–52000 and 4000–24000; grade labels use a
  standard ordinal vocabulary. Historical headers such as `KCSE` are retained
  as schema examples and do not represent measurements from an institution.

These files support software demonstrations and fixture tests. Their numbers,
category shares, and correlations have no empirical meaning and should not be
used for research conclusions or population comparisons. The recipe supplies
no claim about legal compliance or the privacy of any other dataset processed
by the application.
