# Rare ID formats and text shapes take broader classes that enough people hold

A rare ID format class (including a minority layout in a mixed column), a rare text shape (such as `AA-0000`) and a rare date format used to refuse the column or table, because writing them would show that a few source people use that format. Sharing mode now writes them in a broader class that at least `--min-cell` people hold:
- **ID formats:** the column's most common held format class.
- **Text shapes:** first the same shape without leading-digit and length detail, then the column's most common held shape.
- **Date formats:** the column's most common held date format.

If no held class exists, the column or table is refused as before. Shape classes defined by embedded personal information are not broadened; when they cannot be protected, the column is nulled ([ADR-0001](0001-sharing-nulls-unprotectable-columns.md)). E-mail-address IDs still refuse.
