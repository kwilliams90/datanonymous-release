"""The message catalogue: one plain-language sentence (and fix) for each code a sharing run can report.

Operators of sharing mode do not know the tool's internals, so a blocking finding or a failure is shown as the
catalogue's sentence and fix, with its code kept next to it in parentheses for anyone who needs to look it up.

Adding a code is one line: ``problem(code, sentence, fix)`` for something that needs action, ``change(code,
description)`` for an informational code that describes a deliberate privacy change (no fix). A sentence or fix may
hold ``{text}``, which stands for the finding's own (report-safe) text where the detail matters, for example which
rule or column reference matched nothing. ``tests/test_message_catalogue.py`` reads every code the source can
raise and fails when a blocking one has no entry here.

Fixes never name an option that sharing mode refuses (see ``privacy.sharing_policy.REFUSED_SUGGESTIONS``).
"""

from __future__ import annotations

from dataclasses import dataclass

#: the two kinds of entry: something that needs the operator's attention, and a deliberate privacy change
PROBLEM = "problem"
CHANGE = "change"


@dataclass(frozen=True)
class Entry:
    kind: str
    sentence: str
    fix: str | None = None


CATALOGUE: dict[str, Entry] = {}


def problem(code: str, sentence: str, fix: str) -> None:
    """Register a blocking or failure code: what is wrong, and what the operator can do about it."""
    CATALOGUE[code] = Entry(PROBLEM, sentence, fix)


def change(code: str, description: str) -> None:
    """Register an informational code that describes a deliberate privacy change (a description, no fix)."""
    CATALOGUE[code] = Entry(CHANGE, description, None)


def lookup(code: str) -> Entry | None:
    return CATALOGUE.get(code)


def render(code: str, text: str = "") -> tuple[str, str | None] | None:
    """The plain (sentence, fix) of ``code``, or None when the catalogue does not know it."""
    entry = CATALOGUE.get(code)
    if entry is None:
        return None
    sentence = entry.sentence.replace("{text}", text.rstrip(". "))
    fix = entry.fix.replace("{text}", text.rstrip(". ")) if entry.fix else None
    return sentence, fix


# --------------------------------------------------------------------------- run and file failures
problem("fail", "This file could not be written safely: {text}.",
        "Read the reason, correct the file or the column options, and run again.")
problem("withheld_by_findings", "This file was left out because a privacy check found a problem in it.",
        "Fix the findings listed for this file, or leave the file out with `--exclude-path`, and run again.")
problem("withheld_by_run", "This file was left out because a problem with the whole run stopped everything being written.",
        "Fix the run-level problem listed in the report, then run again.")
problem("internal_error", "This file hit an unexpected error and was left out.",
        "Run again with `-v` to see the details, and report the problem.")
problem("sharing_nothing_written", "No file could be written to the output folder.",
        "Fix the problems listed for each file, or leave the problem files out with `--exclude-path`.")
problem("sharing_package_blocked", "A problem was found, so nothing was written to the output folder.",
        "Fix the problems listed in the report, then run again.")
problem("sharing_incomplete", "`--all-or-nothing` is on and at least one input file was not written, so no file was.",
        "Fix or leave out the files that were not written, or run without `--all-or-nothing`.")
problem("sharing_skipped_inputs", "`--all-or-nothing` is on and at least one input file was skipped, so no file was written.",
        "Choose the input files explicitly with `--include`, or run without `--all-or-nothing`.")
problem("sharing_inventory", "The files for the output folder could not be checked, so nothing was written.",
        "Remove odd files, links or unreadable files from the input and output folders, then run again.")
problem("sharing_promotion", "The finished files could not be moved into the output folder, so nothing was written.",
        "Choose a new or empty output folder on the same drive as the report folder.")
problem("no_tabular_files", "No table files (CSV, Excel and the like) were found in the input folder.",
        "Point the run at a folder that holds table files.")

# --------------------------------------------------------------------------- sharing: columns that cannot be protected
problem("sharing_unprotectable", "This column cannot be protected well enough to share.",
        "Leave it out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_no_support", "Too few different people share this column's values, so none of them can be shared safely.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_no_people", "This table has no person column, so sharing mode cannot tell how many people hold each value.",
        "Name the column that identifies each row's person with `--person-id`, or leave the column out with `--exclude`.")
problem("sharing_rare_in_structure", "This column holds rare values that are tied to a key, rule or dependency between columns.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_rare_numeric", "This number column has rare values that could be recovered from the shared copy.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_mixed_cell_types", "This Excel column mixes cell types or number formats that are too rare to share.",
        "Make the column one type in the source, or leave it out with `--exclude`.")
problem("sharing_repeated_embedded", "This column repeats personal details inside its text in a way that would link people.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_person_dependency", "This column depends on a person or date in a way that cannot be copied without revealing the source.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_source_template", "This column is built from other columns in a way that cannot be copied safely.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_numeric_structure", "This number column has extreme values that a rule or key between columns would expose.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_date_structure", "This person date is tied to a rule or key between columns, so it cannot be changed safely.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_date_no_cover", "This date column has no safe set of dates to write that hides the real earliest and latest dates.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_date_out_of_range", "A date in this column falls outside what the file format can store once it is changed.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_numeric_no_support", "This number column has no safe range of values left once its rare extremes are removed.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_rare_unit_restored", "Some rare categories or blanks in this column came back into the shared copy.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_numeric_extreme_restored", "A rare extreme number from the source came back into the shared copy.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_date_extreme_restored", "A rare earliest or latest date from the source came back into the shared copy.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_date_cover_restored", "A rare group of dates from the source came back into the shared copy.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")

# --------------------------------------------------------------------------- leak scan
problem("copied_value", "A value in the shared copy of this column is the same as a real value in the source.",
        "Leave the column out with `--exclude` or write it as an empty column with `--null`, and report the problem.")
problem("real_text_copied", "Text in the shared copy of this column is the same as real text in the source.",
        "Leave the column out with `--exclude` or write it as an empty column with `--null`, and report the problem.")
problem("real_full_name", "The shared copy spells out the full name of a real person.",
        "Leave the name column out with `--exclude` or write it as an empty column with `--null`, and report the problem.")
problem("leak_scan_incomplete", "The check that looks for real values in the shared files could not finish, so these files cannot be trusted.",
        "Run again with `-v` to see the details, and report the problem.")
problem("real_id_values", "Some values in this column are real person IDs from your data.",
        "Declare the column as an ID with `--id-col`, or leave it out with `--exclude`.")

# --------------------------------------------------------------------------- IDs, keys and links between columns
problem("id_candidates", "Some columns look like IDs, but sharing mode is not treating them as IDs.",
        "Declare each one with `--person-id` (a person's own ID) or `--id-col` (any other ID), or leave it out with `--exclude`.")
problem("no_id_column", "No column was recognised as an ID, so every row is treated as its own person.",
        "Name the column that identifies each person with `--person-id`.")
problem("id_columns_unrelated", "Two ID columns are not linked to each other, so their pairing in the shared copy is only kept row by row.",
        "Check that both columns are the IDs you mean, and declare any that are not with `--id-col`.")
problem("id_key_ref", "This column holds the values of an ID column, so sharing mode treats it as an ID too.",
        "If it is not an ID, leave it out with `--exclude`.")
problem("id_by_header_values", "This column was treated as an ID because of its name and values.",
        "Declare it yourself with `--person-id` or `--id-col`, or leave it out with `--exclude`.")
problem("unlinked_key_column", "This column holds the IDs of another column but is not set up as an ID, so its link would break.",
        "Declare it with `--id-col`, or leave it out with `--exclude`.")
problem("d38_flagged_keys", "This column repeats ID-like numbers that were not declared as IDs.",
        "Declare it with `--person-id` or `--id-col`, or leave it out with `--exclude`.")
problem("possible_sparse_code", "This number column looks like a code or ID, which cannot be shared as an ordinary number.",
        "Declare it with `--id-col` (or `--person-id` for a person's own ID), or leave it out with `--exclude`.")
problem("id_check_not_applied", "The ID check digit you asked for was not applied because the real IDs do not all have one.",
        "Check the IDs in the source, or remove the `--id-check` option.")
problem("id_width_changed", "The IDs in the shared copy have a different length from the real ones.",
        "No action is needed unless you need the original length; then leave the column out with `--exclude`.")
problem("composite_key_duplicates", "Some rows repeat a combination of columns that is meant to be unique.",
        "Check the key columns in the source, or leave one of them out with `--exclude`.")
problem("excel_scientific_ids", "Long ID numbers were shown by Excel in scientific notation, so digits were already lost.",
        "Save the source column as text and run again, or leave it out with `--exclude`.")
problem("entity_rows_inexact", "Some people in the shared copy do not get exactly the same number of rows as in the source.",
        "Check the ID columns, or leave a column out with `--exclude` if the row counts matter.")

# --------------------------------------------------------------------------- personal data in columns
problem("pii_values", "Some values in this column look like personal data, so they were replaced with made-up ones.",
        "If the column should not be shared at all, leave it out with `--exclude` or write it as an empty column with `--null`.")
problem("value_names", "The values in this column look like people's names, so they were replaced with made-up names.",
        "If the column should not be shared at all, leave it out with `--exclude` or write it as an empty column with `--null`.")
problem("d35_rows_counted", "This column keeps real values, but sharing mode cannot show that enough different people hold them.",
        "Name the student ID column with `--person-id`, or leave the column out with `--exclude`.")
problem("forced_copy", "This column would copy its real values unchanged.",
        "Remove the option that keeps real values, or leave the column out with `--exclude`.")
problem("forced_type_values", "A column type you set does not change that this column is personal data, so its values are still replaced.",
        "No action is needed unless the column should not be shared; then use `--exclude` or `--null`.")
problem("factor_overflow", "This column has too many different values to keep as categories, so its real labels were not kept.",
        "No action is needed unless the labels are required; then leave the column out with `--exclude`.")
problem("lat_lon_independent", "Location values were made up independently, so shared points are not real places and latitude and longitude do not match.",
        "No action is needed unless locations matter; then leave the columns out with `--exclude`.")
problem("column_excluded", "A column was left out with `--exclude`, so the shared file has fewer columns than the source.",
        "No action is needed; remove `--exclude` for that column if you meant to share it.")
problem("unused_column_flag", "An option you gave names a column that was not found: {text}.",
        "Check the spelling of the column name, or remove the option.")

# --------------------------------------------------------------------------- rules, structure and chance matches
problem("rule_not_applied", "A rule you declared was not applied: {text}.",
        "Check the rule's column names, or remove the rule.")
problem("rule_unused", "A rule you declared matched no table: {text}.",
        "Check the rule's column names and file path, or remove the rule.")
problem("rule_violated", "A rule you declared is broken in the source data.",
        "Correct the data or the rule, then run again.")
problem("rule_partly_true", "A rule you declared holds for only some of the real rows, but is applied to every row of the shared copy.",
        "Check the rule and the source data.")
problem("rule_out_of_range", "A rule could only be met by writing values outside the real range.",
        "Check the rule, or remove it.")
problem("rule_extreme_dropped", "A real earliest or highest value could not be kept while meeting a rule you declared.",
        "Check the rule, or remove it.")
problem("template_partly_applied", "A column built from other columns could not be rebuilt for some rows.",
        "Check the source columns, or leave the built column out with `--exclude`.")
problem("fd_unmapped", "Some rows could not follow a column-to-column dependency in the source.",
        "No action is needed unless the dependency matters; then leave the column out with `--exclude`.")
problem("fd_missingness_conflict", "Blank cells in some rows could not follow both the dependency between columns and the usual blank pattern.",
        "No action is needed unless the blanks matter.")
problem("missingness_pattern_changed", "The pattern of blank cells changed in some rows of the shared copy.",
        "No action is needed unless the blanks matter.")
problem("aggregate_table", "This table looks like a summary of other data, so sharing it may reveal the source totals.",
        "Check that sharing a summary table is intended; leave it out with `--exclude-path` if it is not.")
problem("multi_year_snapshots", "This table seems to hold the same people in several years.",
        "No action is needed unless the yearly structure matters.")
problem("row_matches_remaining", "Some rows in the shared copy are the same as real rows by chance, even after retrying.",
        "Leave out a column that makes rows easy to match with `--exclude`, or add variety to the source columns.")
problem("row_matches_report_only", "A share of rows in the shared copy are the same as real rows by chance.",
        "Leave out a column that makes rows easy to match with `--exclude`, or add variety to the source columns.")

# --------------------------------------------------------------------------- dates
problem("ambiguous_date_order", "Dates in this column could be day-first or month-first.",
        "Say which with `--dayfirst`, or give the exact format with `--date-format`.")
problem("calendar_ignored", "The calendar option was ignored for personal dates such as birth dates.",
        "No action is needed; remove the option to avoid this message.")
problem("date_subus_truncated", "Very fine time detail in this column was written as zeros.",
        "No action is needed unless exact times matter; then leave the column out with `--exclude`.")

# --------------------------------------------------------------------------- reading the source files
problem("blank_lines_dropped", "Blank lines between records in the source were dropped.",
        "Remove the blank lines from the source and run again if they matter.")
problem("line_endings_mixed", "The source mixes different line endings, and the shared copy uses the most common one.",
        "Convert the source to one line ending and run again if that matters.")
problem("delimiter_ambiguous", "The column separator could not be told apart with confidence.",
        "Check the columns in the report, then confirm the separator with `--delimiter`.")
problem("header_row_ignored", "The header-row option does not apply to a text file and was ignored.",
        "Remove `--header-row` for this file.")
problem("encoding_low_confidence", "The file's text encoding could not be identified with confidence.",
        "If the text looks wrong, set the encoding with `--encoding`.")
problem("quoting_inconsistent", "This file quotes some values in a column but not others, and the shared copy quotes consistently.",
        "No action is needed unless exact quoting matters.")
problem("xls_as_xlsx", "This old Excel file (.xls) is written as a newer one (.xlsx).",
        "Save the source as .xlsx if the original file type matters.")
problem("xlsm_macros", "This Excel file holds macros, which are never copied; it is written without them.",
        "No action is needed; save the source as .xlsx to avoid this message.")
problem("excel_extras_dropped", "Excel extras such as comments or images were not copied.",
        "No action is needed unless they matter.")
problem("excel_formulas_uncached", "The workbook's formulas have no stored results, so those cells read as empty.",
        "Open the workbook in Excel, save it, and run again.")
problem("excel_hidden_selected", "Only hidden sheets were selected, and the sheet is written visible.",
        "Select a visible sheet with `--sheet`, or leave it as is.")
problem("excel_merged_dropped", "Merged cells inside the data were not merged in the shared copy.",
        "No action is needed unless the merging matters.")
problem("excel_notes_sheet", "This sheet holds notes rather than a table, and is written as placeholder text.",
        "Leave the sheet out with `--sheet` if it is not needed.")
problem("excel_range_dropped", "Cells outside the range you set were not written.",
        "Widen the range with `--range` if they are needed.")
problem("excel_range_unused", "A range option names a sheet that does not exist.",
        "Check the sheet name in `--range`.")
problem("excel_sheets_not_written", "Only the sheet you chose was written, so other sheets are not in the output.",
        "Remove `--sheet` to write every sheet.")
problem("excel_side_notes", "Cells beside the table were treated as notes and written as placeholder text.",
        "Set the table's range with `--range` if they belong to the table.")
problem("html_extras_dropped", "Parts of the HTML file that are not data were not copied.",
        "No action is needed unless they matter.")
problem("html_notes_table", "An HTML table holds notes rather than data, and is written as placeholder text.",
        "No action is needed unless the notes matter.")
problem("arrow_metadata_dropped", "Free-form metadata in the file was not copied, because it can quote real data.",
        "No action is needed.")
problem("parquet_codec", "The file's compression is not available for writing, so another one is used.",
        "No action is needed.")
problem("sas_as_parquet", "SAS files can only be read, so this one is written as a Parquet file.",
        "No action is needed unless you need another format.")
problem("stata_extended_missing", "Stata's extended missing values were all written as one plain missing value.",
        "No action is needed unless they matter.")
problem("stata_labels_dropped", "Free-text Stata labels were not copied, because they can quote real data.",
        "No action is needed.")
problem("stata_version", "This Stata version cannot be written, and an older one is used.",
        "No action is needed.")
problem("container_name_replaced", "The file name stored inside a compressed file was replaced, because it can hold free text.",
        "No action is needed; programs that open the member by name must use its new name.")
problem("doc_name_collision", "A data file has the same name as one of the documents written with the output.",
        "Rename the data file in the source.")
problem("stale_output", "The output folder holds an older copy of a file that was not rewritten in this run.",
        "Use a new or empty output folder.")

# --------------------------------------------------------------------------- checks on the written copy
problem("dtype_mismatch", "A column's data type differs between the source and the shared copy.",
        "Check the column in the report; leave it out with `--exclude` if the type matters.")
problem("layout_mismatch", "The written file's layout differs from the source: {text}.",
        "Check the file in the report; the layout rarely matters for the data.")
problem("written_values_differ", "The file format changed some values as it saved them.",
        "No action is needed unless exact values matter.")

# --------------------------------------------------------------------------- failures raised while writing a file
problem("sharing_other_person_attribute", "This column describes a parent or teacher, not the student, so it can't be shared safely.",
        "Leave it out with `--exclude`, or write it as an empty column with `--null`.")
problem("sharing_rare_class", "This column has a date, text or personal-data layout that too few people share to release.",
        "Leave the column out with `--exclude`, or write it as an empty column with `--null`.")
problem("real_values_in_output", "Real values from the source would have appeared in this file's output, so the file was left out.",
        "Leave the column that holds them out with `--exclude` or write it as an empty column with `--null`, and report the problem.")
problem("name_holds_personal_data", "A name this file would write (its file, sheet or column name) holds personal data, so the file was left out.",
        "Rename the file, sheet or column in the source, or leave the file out with `--exclude-path`.")
problem("rule_unusable", "A rule you declared cannot be used with this file, so the file was left out.",
        "Correct the rule's column names, or remove the rule.")
