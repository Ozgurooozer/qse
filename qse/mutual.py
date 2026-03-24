"""
qse.mutual
----------
Mutual information and subadditivity utilities.

Theorem T-MI
~~~~~~~~~~~~
For any bipartition of B into B₁, B₂ (rows of M split accordingly),
the quantum mutual information satisfies

    I(B₁ ; B₂) = S(B₁) + S(B₂) − S(B₁B₂) ≥ 0 .

This is a consequence of strong subadditivity of von Neumann entropy.
"""

from __future__ import annotations

import numpy as np

from .entropy import t14_formula


def mutual_information(
    thetas: list[float],
    M: np.ndarray,
    split: int | None = None,
) -> float:
    """Mutual information I(B₁ ; B₂) via the T14 formula.

    Parameters
    ----------
    thetas : sequence of float, length NA
        A-qubit rotation angles.
    M : np.ndarray of int, shape (NB, NA)
        Binary connectivity matrix.  Rows 0 … split-1 belong to B₁;
        rows split … NB-1 belong to B₂.
    split : int, optional
        Row index at which to partition M.  Defaults to NB // 2.

    Returns
    -------
    float
        I(B₁ ; B₂) ≥ 0.
    """
    M = np.asarray(M, dtype=int)
    n_B = M.shape[0]
    if split is None:
        split = n_B // 2

    M1 = M[:split, :]
    M2 = M[split:, :]

    s_b1 = t14_formula(thetas, M1)
    s_b2 = t14_formula(thetas, M2)
    s_b  = t14_formula(thetas, M)

    return max(0.0, s_b1 + s_b2 - s_b)
