# Sharing mode nulls an unprotectable non-key column instead of leaving the file out

When sharing mode cannot protect a column that is not a key and has no structural relationship (join, alias, nesting, dependency or declared rule), it writes that column as a Nulled column and reports it under "What needs attention", instead of leaving the whole file out. Before this, one such column cost the file, and operators had to rerun with `--null` themselves. Nulling releases nothing real and keeps the rest of the file usable. `--refuse-unsupported` restores the old refusal for operators who want it. Keys and columns in a relationship still leave the file out, because nulling them would break the joins the copy exists to keep.

## Consequences

- A run whose only problems are auto-nulled columns exits 0 and counts the file as written for `--all-or-nothing`. With `--refuse-unsupported`, the file is left out and counts toward exit 3, as before.
- A numeric column that looks like an identifier but is not used as a key is nulled the same way. The report suggests declaring it as an ID instead.

## Amendment: relationships inside one table don't stop nulling

A column whose only relationship is inside its own table (a dependency, such as a join year fixed by the year of study, or a template input, such as the name parts of a built e-mail address) is nulled like any other non-key column, and the report says which link is no longer kept. Nulling it breaks no join; it only loses the pattern, which ADR-0005 ranks below writing the file. Keys, cross-file joins, aliases, nesting and rules the operator declared still refuse.
