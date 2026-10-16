# datanonymous

Build and test code against synthetic files while the original education
records stay on your machine. datanonymous preserves useful columns, formats,
row totals and joins, and creates fictional people and relationships.

Sharing mode is the default. It checks every output file before writing it to
the output folder. Input files it cannot safely support are left out and listed
in the report folder's `sharing-failures.csv`; pass `--all-or-nothing` to write
nothing unless every file succeeds. Synthetic data still
needs review before sharing; a successful run is not FERPA certification.

## Install

Python 3.12 or newer is required. The current release is a release candidate,
so pip needs `--pre` to find it:

```shell
pip install --pre datanonymous
```

Install it into a virtual environment: it pins an exact Faker version so the
same seed keeps producing the same output, which may conflict with other
projects. Run `datanonymous --help` to see the options.

### From source

To try the invented examples or run the tests, clone this repository and run
these commands from its folder:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

The commands below assume a source checkout on Windows. On macOS or Linux,
replace `.venv/Scripts/python.exe` with `.venv/bin/python`. With a pip install,
replace `.venv/Scripts/python.exe -m datanonymous` with `datanonymous`.

## Try the invented examples

The CSVs in `example-data/` are [wholly invented](https://github.com/kwilliams90/datanonymous-release/blob/main/example-data/README.md).
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
The output folder must be new or empty, and outside the input and report
folders; the report folder stays local and may sit inside the input.

Review the synthetic tables and their data dictionary before sharing them.
Keep original records, comparison reports and temporary files local. If a run
is refused, follow its explanation; switching to fidelity mode does not make
the result safe to share.

## Further reading

- [Sharing policy](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/sharing-policy.md), [threat model](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/threat-model.md) and [limits](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/limitations.md)
- [File formats](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/formats.md) and [production code examples](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/code-compatibility.md)
- [Operator guide](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/operations.md) and [security reporting](https://github.com/kwilliams90/datanonymous-release/blob/main/SECURITY.md)
- [Detailed fidelity options](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/fidelity-reference.md) and [release notes](https://github.com/kwilliams90/datanonymous-release/blob/main/guides/release-notes.md)

For development, run `.venv/Scripts/python.exe -m pytest -q`.

Licensed under [Apache-2.0](https://github.com/kwilliams90/datanonymous-release/blob/main/LICENSE).
