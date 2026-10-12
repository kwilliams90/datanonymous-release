"""Joint model between columns (FOUNDATION seam; design D3, 5.1).

v1 generates every non-ID column independently: exact quota for discrete
columns, stratified inverse-CDF for continuous ones, within the ID blocks,
within-ID constants and missingness patterns. No copula or stratify mode ships.

Seam for a later version: :func:`joint_model` is consulted by nothing yet; a
future Gaussian copula (numpy-only, design 5.1) or ``--stratify-by`` mode would
return a :class:`JointPlan` whose ``uniforms`` replace each column's independent
stratified uniforms in ``generate/numeric.py`` and ``generate/quota.py`` (rank
coupling), leaving marginals, floors and rendering unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODEL = "independent"


@dataclass
class JointPlan:
    """Correlated uniforms per column index (empty = independent generation)."""

    model: str = MODEL
    uniforms: dict[int, np.ndarray] = field(default_factory=dict)


def joint_model(table_profile: Any, n: int, rng: np.random.Generator) -> JointPlan:
    """Return the joint plan for a table. v1: independent (no coupling)."""
    return JointPlan()
