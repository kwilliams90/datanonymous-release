# Data dictionary (SYNTHETIC DATA)

**FIDELITY MODE.** For invented-data and local testing; this folder is not an approved sharing package.

Every statistic below is computed from the synthetic files in this folder, not from real data. Counts below 10 are shown as `<10`. Levels are listed only for category-like columns (factor, ordered factor, boolean, organisational code, school year, constant, flag), whose labels are reproduced as they are; other text columns hold invented values. A real label is reproduced only when it is a vocabulary word or shared by at least 10 people in the run (distinct IDs; identical rows count once in a table without IDs); every other one is written as a consistent same-shape pseudonym, listed here as written and counted in `pseudonymized_levels`.

## `StudentsPerformance.csv`

- File: format csv, encoding utf-8, no byte-order mark, delimiter `','`, line ending CRLF, quoting minimal
- Rows: 1000; columns: 8
- readr `col_types`: `fffffiii`
- Missing-value tokens written: `''`

| # | Column (original header) | Type | R class | readr | pandas dtype | Format / NA tokens | Missing % | Distinct | Levels or summary (synthetic) | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `gender` | FACTOR | factor | f | str |  | 0.0 | 2 | 'female': 502; 'male': 498 |  |
| 2 | `race/ethnicity` | FACTOR | factor | f | str |  | 0.0 | 5 | 'group A': 212; 'group C': 205; 'group D': 205; 'group E': 204; 'group B': 174 |  |
| 3 | `parental level of education` | ORDINAL (parental_education) | ordered factor | f | str |  | 0.0 | 6 | 'some high school': 160; 'high school': 175; 'some college': 167; "associate's degree": 179; "bachelor's degree": 141; "master's degree": 178 |  |
| 4 | `lunch` | FACTOR | factor | f | str |  | 0.0 | 2 | 'free/reduced': 500; 'standard': 500 |  |
| 5 | `test preparation course` | FACTOR | factor | f | str |  | 0.0 | 2 | 'completed': 503; 'none': 497 |  |
| 6 | `math score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 11, median 62.5, mean 62.132, max 100 |  |
| 7 | `reading score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 90 | min 8, median 61.0, mean 61.461, max 100 |  |
| 8 | `writing score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 11, median 63.0, mean 62.127, max 100 |  |

R (readr):

```r
df <- readr::read_delim("StudentsPerformance.csv", delim = ",", col_types = "fffffiii", na = c(""))
```

pandas (all columns as text, NA tokens kept as written):

```python
df = pd.read_csv('StudentsPerformance.csv', sep=',', dtype=str, keep_default_na=False)
```

Ordered levels of `parental level of education` (R): `factor(x, levels = c("some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"), ordered = TRUE)`

## `student-scores.csv`

- File: format csv, encoding utf-8, no byte-order mark, delimiter `','`, line ending LF, quoting minimal
- Rows: 2000; columns: 17
- readr `col_types`: `ccccflilifiiiiiii`
- Missing-value tokens written: `''`

| # | Column (original header) | Type | R class | readr | pandas dtype | Format / NA tokens | Missing % | Distinct | Levels or summary (synthetic) | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `id` | IDENTIFIER (sequential) | character | c | int64 | shape `9999` / `999` / `99` | 0.0 | 2000 | length 1-4 | ID (row numbers 1..n kept) |
| 2 | `first_name` | PERSON_NAME (first) | character | c | str |  | 0.0 | 64 | length 6-6 |  |
| 3 | `last_name` | PERSON_NAME (last) | character | c | str |  | 0.0 | 64 | length 7-7 |  |
| 4 | `email` | EMAIL | character | c | str |  | 0.0 | 2000 | length 28-31 |  |
| 5 | `gender` | FACTOR | factor | f | str |  | 0.0 | 2 | 'male': 1029; 'female': 971 |  |
| 6 | `part_time_job` | BOOLEAN | logical | l | bool | tokens `False` / `True` | 0.0 | 2 | 'False': 1006; 'True': 994 |  |
| 7 | `absence_days` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 21 | min 0, median 10.0, mean 10.041, max 20 |  |
| 8 | `extracurricular_activities` | BOOLEAN | logical | l | bool | tokens `False` / `True` | 0.0 | 2 | 'True': 1015; 'False': 985 |  |
| 9 | `weekly_self_study_hours` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 31 | min 0, median 15.0, mean 15.1385, max 30 |  |
| 10 | `career_aspiration` | FACTOR | factor | f | str |  | 0.0 | 17 | 'Librarian': 131; 'Scientist': 129; 'Social Worker': 127; 'Journalist': 126; 'Software Engineer': 124; 'Teacher': 124; 'Accountant': 122; 'Doctor': 122; 'Entrepreneur': 120; 'Artist': 118; 'Chef': 118; 'Pilot': 113; ... (17 levels) |  |
| 11 | `math_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 13, median 58.0, mean 58.3515, max 100 |  |
| 12 | `history_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 89 | min 12, median 58.0, mean 57.8985, max 100 |  |
| 13 | `physics_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 11, median 57.0, mean 57.487, max 100 |  |
| 14 | `chemistry_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 87 | min 13, median 57.0, mean 57.704, max 100 |  |
| 15 | `biology_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 12, median 58.0, mean 57.563, max 100 |  |
| 16 | `english_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 88 | min 11, median 58.0, mean 58.0315, max 100 |  |
| 17 | `geography_score` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 89 | min 12, median 58.0, mean 58.3155, max 100 |  |

R (readr):

```r
df <- readr::read_delim("student-scores.csv", delim = ",", col_types = "ccccflilifiiiiiii", na = c(""))
```

pandas (all columns as text, NA tokens kept as written):

```python
df = pd.read_csv('student-scores.csv', sep=',', dtype=str, keep_default_na=False)
```

## `students_data.csv`

- File: format csv, encoding utf-8, no byte-order mark, delimiter `','`, line ending LF, quoting minimal
- Rows: 62; columns: 11
- readr `col_types`: `ffiiiiifiif`
- Missing-value tokens written: `''`

| # | Column (original header) | Type | R class | readr | pandas dtype | Format / NA tokens | Missing % | Distinct | Levels or summary (synthetic) | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `YearofStudy` | ORDINAL (year_of_study) | ordered factor | f | str |  | 0.0 | 4 | 'first year': 16; 'second year': 16; 'third year': 15; 'fourth year': 15 |  |
| 2 | `Course` | ORG_CODE | factor | f | str |  | 0.0 | 4 | 'ART101': 16; 'SCI202': 16; 'BUS303': 15; 'TEC404': 15 |  |
| 3 | `Age` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 11 | min 19, median 24.0, mean 23.83871, max 29 |  |
| 4 | `ApproxHeight` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 37 | min 146, median 171.0, mean 170.032258, max 193 |  |
| 5 | `ApproxWeight` | INTEGER (discrete) | integer | i | int64 |  | 0.0 | 37 | min 46, median 71.5, mean 71.516129, max 95 |  |
| 6 | `KCSE` | INTEGER_YEAR | integer | i | int64 |  | 0.0 | 6 | min 2020, median 2022.0, mean 2022.451613, max 2025 |  |
| 7 | `Yr_JoinCampus` | INTEGER_YEAR | integer | i | int64 |  | 0.0 | 4 | min 2023, median 2025.0, mean 2024.532258, max 2026 |  |
| 8 | `SitKCSE` | FACTOR | factor | f | str |  | 0.0 | 6 | 'Central ': 14; 'Western': 13; 'Southern': 11; 'Eastern ': 10; 'Coast': <10; 'Northern': <10 |  |
| 9 | `Expense_Semester` | INTEGER (continuous) | integer | i | int64 |  | 0.0 | 62 | length 5-5 | extreme_value_adjusted |
| 10 | `Expense_Accommodation` | INTEGER (continuous) | integer | i | int64 |  | 0.0 | 62 | min 4429, median 15148.5, mean 15031.048387, max 23362 | extreme_value_adjusted |
| 11 | `Previous_Exam_MeanGrade` | ORDINAL (letter_grade) | ordered factor | f | str |  | 0.0 | 10 | 'D': <10; 'D+': <10; 'C-': <10; 'C': <10; 'C+': <10; 'B-': <10; 'B': <10; 'B+': <10; 'A-': <10; 'A': <10 |  |

R (readr):

```r
df <- readr::read_delim("students_data.csv", delim = ",", col_types = "ffiiiiifiif", na = c(""))
```

pandas (all columns as text, NA tokens kept as written):

```python
df = pd.read_csv('students_data.csv', sep=',', dtype=str, keep_default_na=False)
```

Ordered levels of `YearofStudy` (R): `factor(x, levels = c("first year", "second year", "third year", "fourth year"), ordered = TRUE)`

Ordered levels of `Previous_Exam_MeanGrade` (R): `factor(x, levels = c("D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A"), ordered = TRUE)`
