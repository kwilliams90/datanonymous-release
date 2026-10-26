# Rows count as people in a table with no person column, unless something suggests repeats

Sharing mode used to refuse to count people in a table with no column that identifies each row's person, because nothing shows whose rows they are. That left a common one-row-per-person file with every column nulled. Now, within that one table, each row counts as a different person, but only when nothing suggests the same person appears on several rows:
- no column looks like a repeated ID (repeated wide keys);
- no person-name or e-mail column repeats a value;
- exact duplicate rows still count once, as before.

The run reports the assumption under "What needs attention" and points to `--person-id` for tables where a person can have several rows.

## Consequences

- The assumption never crosses files. A row in one table says nothing about the people in another.
- When there is a sign of repeated people, the table keeps the old behaviour: its columns that cannot be protected are nulled ([ADR-0001](0001-sharing-nulls-unprotectable-columns.md)).
- The risk that remains is a table that repeats people with no visible sign: no ID, name or e-mail column. In that case a value held by one person on many rows could pass as common. That is why the assumption is reported every time it is used.
