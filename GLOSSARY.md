# datanonymous

datanonymous makes synthetic copies of tabular files so code can be written against them while the original records stay local. Sharing mode is the default and must never release real values.

## Language

**Nulled column**:
A column written with its header and position kept and every cell missing, either because the operator asked for it or because sharing mode could not protect it.
_Avoid_: Blanked column, suppressed column, dropped column

**File left out**:
An input file that sharing mode did not write to the output folder, listed with its reason in the private report.
_Avoid_: Refused file, rejected file

**Skipped input**:
An input file that was never attempted because it is not a supported table (for example a PDF or an image).
_Avoid_: Failed file

**Count class**:
All the people in a table who hold the same number of rows.
_Avoid_: Count bucket, frequency class

**Format class**:
All the values in a column that share one layout (character pattern, width, punctuation or date format).
_Avoid_: Template, pattern, shape class

**Held class**:
A count class or format class that at least `--min-cell` distinct source people hold, so it may appear in sharing output.
_Avoid_: Common class, supported class

**Private report**:
The report folder for a run. It describes the source and must never be shared.
_Avoid_: Run report, log folder

**Output folder**:
The folder of synthetic files, README and data dictionary that a run produces for sharing.
_Avoid_: Synthetic tree, package, out dir
