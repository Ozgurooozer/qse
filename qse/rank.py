"""
qse.rank
--------
Gaussian elimination over GF(2) (the field with two elements, F₂).

The rank of the binary connectivity matrix M over F₂ is a central
quantity in QSE theory:

    S(B) ≤ rank_F₂(M)          (T-RANK)
    S(B) = rank_F₂(M)  at θ=π/2 (T-OPT)
"""

import numpy as np


def f2_rank(M: np.ndarray) -> int:
    """Compute the rank of matrix *M* over GF(2).

    Parameters
    ----------
    M : array-like of int, shape (m, n)
        Binary matrix (entries are taken mod 2).

    Returns
    -------
    int
        rank_F₂(M).

    Examples
    --------
    >>> f2_rank([[1, 0], [0, 1]])
    2
    >>> f2_rank([[1, 1], [1, 1]])
    1
    >>> f2_rank([[0, 0], [0, 0]])
    0
    """
    m = np.asarray(M, dtype=int) % 2
    rows, cols = m.shape
    rank = 0

    for col in range(cols):
        # find pivot
        pivot = next((r for r in range(rank, rows) if m[r, col]), None)
        if pivot is None:
            continue
        # swap pivot row into position
        m[[rank, pivot]] = m[[pivot, rank]]
        # eliminate all other rows
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] = (m[r] + m[rank]) % 2
        rank += 1

    return rank
