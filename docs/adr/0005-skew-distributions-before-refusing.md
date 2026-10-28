# When protection costs fidelity, skew the distribution before refusing

Sharing-mode copies exist so code can be written against them. When a protection forces a trade-off, the order is:
1. Release no real or rare value.
2. Keep keys and joins valid.
3. Write the file.
4. Only then, keep statistical distributions close to the source.

So we accept skewed distributions, such as count classes reshaped by rebalancing ([ADR-0003](0003-rebalance-rare-count-classes.md)), minority formats collapsed into the common one ([ADR-0004](0004-rare-formats-take-broader-held-classes.md)) and values drawn only from the supported ones ([ADR-0002](0002-one-rare-value-rule-for-all-numerics.md)). We do not accept leaving a file out, or keeping an identifiable row, to preserve a distribution. A broken join costs a developer more than a histogram that differs from the source.
