# File formats and reader contracts

The routing below describes supported readers and output conversions. Sharing
still requires credible person evidence, supported fictional joins and complete
artifact verification. An extension alone does not establish that a particular
file can be released.

| Input | Output | Reader and contract |
| --- | --- | --- |
| CSV, TSV, delimited text | Same format | Preserve supported headers, delimiter, encoding, quoting and line endings; reread rows and values |
| CSV/text in gzip, ZIP, bzip2 or xz | Same compression | Validate container members as well as the parsed table; ambiguous members refuse |
| XLSX, XLSM | XLSX | Rebuild supported sheets and cells; macros, formulas, comments, links and author metadata are not copied as active content |
| BIFF XLS or supported text/HTML disguised as XLS | XLSX | Explicit conversion; test consumers against XLSX rather than expecting the old extension |
| Parquet | Parquet | Reread native logical types and metadata; nested/binary columns require explicit null or exclusion |
| Feather | Feather | Reread native logical types and metadata; unsupported representations refuse |
| Stata DTA | DTA | Check storage types, missing values, formats, labels and reader compatibility |
| Flat JSON records | JSON | Preserve supported field order and scalar types; complex nesting requires explicit handling |
| JSON lines | JSON lines | Validate each record and the complete row count |
| Supported SAS files | Parquet | Explicit conversion; unsupported source features require handling before release |

Sharing may discard a rare missing spelling or refuse mixed storage formats.
Do not assume a privacy transformation preserves every literal sentinel, native
cell style or source-only category. Read the generated dictionary and local
diagnostics, then check the required logical/native types, leading-zero IDs,
missing semantics and joins using the intended production reader. Source
comparison reports remain local.

The [production consumer example](code-compatibility.md) demonstrates CSV
contracts and academic-date grouping. Public format fixtures and regression
tests exercise encodings, compression, workbook conversions, Stata labels and
native Arrow types. A release smoke test covers representative invented files;
these checks do not establish support for every feature of each container.
