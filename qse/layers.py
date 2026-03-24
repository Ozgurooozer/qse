"""
qse.layers
----------
Multi-layer CNOT circuits and the XOR composition theorem (T15A).

Theorem T15A
~~~~~~~~~~~~
For a circuit consisting of *k* sequential CNOT layers with binary
connectivity matrices M₁, M₂, …, M_k, the effective connectivity
matrix is their element-wise XOR (sum mod 2) over F₂:

    M_eff = M₁ ⊕ M₂ ⊕ … ⊕ M_k  (mod 2)

The von Neumann entropy of the output state equals t14_formula(θ, M_eff),
independently of the order of the layers.

Corollary (T18 – period-2)
~~~~~~~~~~~~~~~~~~~~~~~~~~
Since M ⊕ M = 0, applying the same layer twice is equivalent to the
identity, giving an entropy sequence with period 2.
"""

from __future__ import annotations

import numpy as np

from .core import rx, cx
from .entropy import t14_formula, vne_statevector


def effective_matrix(*matrices: np.ndarray) -> np.ndarray:
    """Compute the effective F₂ connectivity matrix for a sequence of layers.

    Parameters
    ----------
    *matrices : np.ndarray of int, each shape (NB, NA)
        Layer connectivity matrices in application order.

    Returns
    -------
    np.ndarray of int, shape (NB, NA)
        M_eff = (M₁ + M₂ + … + M_k) mod 2.

    Examples
    --------
    >>> import numpy as np
    >>> M1 = np.array([[1, 0], [0, 1]])
    >>> M2 = np.array([[1, 1], [0, 1]])
    >>> effective_matrix(M1, M2)
    array([[0, 1],
           [0, 0]])
    """
    M_eff = np.zeros_like(matrices[0], dtype=int)
    for M in matrices:
        M_eff = (M_eff + np.asarray(M, dtype=int)) % 2
    return M_eff


def multilayer_vne(
    thetas: list[float],
    *matrices: np.ndarray,
    method: str = "formula",
) -> float:
    """Von Neumann entropy after applying multiple CNOT layers.

    Parameters
    ----------
    thetas : sequence of float, length NA
        A-qubit rotation angles.
    *matrices : np.ndarray of int, shape (NB, NA)
        Layer connectivity matrices in application order.
    method : {"formula", "statevector"}
        "formula"    → use T15A (fast, exact).
        "statevector"→ direct simulation (slower, for verification).

    Returns
    -------
    float
        S(B) in bits.
    """
    if method == "formula":
        M_eff = effective_matrix(*matrices)
        return t14_formula(thetas, M_eff)

    # statevector reference
    n_B, n_A = matrices[0].shape
    N = n_A + n_B

    psi_A = rx(thetas[0])
    for t in thetas[1:]:
        psi_A = np.kron(psi_A, rx(t))

    psi_B = np.zeros(2 ** n_B, dtype=complex)
    psi_B[0] = 1.0
    psi = np.kron(psi_A, psi_B)

    for M in matrices:
        M = np.asarray(M, dtype=int)
        for j in range(n_B):
            for i in range(n_A):
                if M[j, i]:
                    psi = cx(psi, i, n_A + j, N)

    return vne_statevector(psi, n_B)
