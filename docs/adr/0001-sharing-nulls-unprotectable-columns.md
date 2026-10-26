# Sharing mode nulls an unprotectable non-key column instead of leaving the file out

When sharing mode cannot protect a column that is not a key and has no structural relationship (join, alias, nesting, dependency or declared rule), it writes that column as a Nulled column and reports it under "What needs attention", instead of leaving the whole file out. Before this, one such column cost the file, and operators had to rerun with `--null` themselves. Nulling releases nothing real and keeps the rest of the file usable. `--refuse-unsupported` restores the old refusal for operators who want it. Keys and columns in a relationship still leave the file out, because nulling them would break the joins the copy exists to keep.

## Consequences

- A run whose only problems are auto-nulled columns exits 0 and counts the file as written for `--all-or-nothing`. With `--refuse-unsupported`, the file is left out and counts toward exit 3, as before.
- A numeric column that looks like an identifier but is not used as a key is nulled the same way. The report suggests declaring it as an ID instead.
