# datanonymous

Build and test code against synthetic files while the original education
records stay on your machine. datanonymous preserves useful columns, formats,
row totals and joins, and creates fictional people and relationships.

Sharing mode is the default. It checks the complete output before making it
available and refuses inputs it cannot safely support. Synthetic data still
needs review before sharing; a successful run is not FERPA certification.

## Install

Clone this repository and run these commands from its folder. Python 3.12 or
newer is required.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

The commands below use Windows paths. On macOS or Linux, replace
`.venv/Scripts/python.exe` with `.venv/bin/python`.

## Try the invented examples

The CSVs in `example-data/` are [wholly invented](example-data/README.md).
Generate a copy and inspect the results in `demo-output`:

```powershell
.venv/Scripts/python.exe -m datanonymous example-data --mode fidelity --include "*.csv" --out-dir demo-output --seed 1
```

Fidelity mode is for invented data kept locally. It can retain statistics and
relationships that sharing mode changes. The checked-in
`example-output/synthetic/` shows an invented-data demonstration.

## Generate a sharing copy

```powershell
.venv/Scripts/python.exe -m datanonymous local-input --person-id Student_ID --out-dir new-sharing-copy --report-dir local-report
```

Use the column identifying each row's subject, including repeated event rows.
Keep inputs, output and reports in separate folders; the output folder must
not already exist.

Review the synthetic tables and their data dictionary before sharing them.
Keep original records, comparison reports and temporary files local. If a run
is refused, follow its explanation; switching to fidelity mode does not make
the result safe to share.

## Further reading

- [Sharing policy](guides/sharing-policy.md), [threat model](guides/threat-model.md) and [limits](guides/limitations.md)
- [File formats](guides/formats.md) and [production code examples](guides/code-compatibility.md)
- [Operator guide](guides/operations.md) and [security reporting](SECURITY.md)
- [Detailed fidelity options](guides/fidelity-reference.md) and [release notes](guides/release-notes.md)

For development, run `.venv/Scripts/python.exe -m pytest -q`.

Licensed under [Apache-2.0](LICENSE).
