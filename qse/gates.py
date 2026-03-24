"""
qse.gates
---------
CZ-gate equivalence theorem (CZ-T14).

Theorem CZ-T14
~~~~~~~~~~~~~~
Replacing each CNOT(A_i → B_j) with a CZ gate, and initialising each
B-qubit in |+⟩ = (|0⟩ + |1⟩)/√2 instead of |0⟩, yields the same von
Neumann entropy as the T14 CNOT circuit.  That is, the two circuits are
entropy-equivalent up to a local Hadamard on B.
"""

from __future__ import annotations

import numpy as np

from .entropy import t14_formula, vne_statevector
from .core import rx


def cz_vne(thetas: list[float], M: np.ndarray) -> float:
    """Von Neumann entropy of B under CZ-gate circuit with B initialised in |+⟩^⊗NB.

    Parameters
    ----------
    thetas : sequence of float, length NA
        A-qubit rotation angles.
    M : np.ndarray of int, shape (NB, NA)
        Binary connectivity matrix (same interpretation as T14).

    Returns
    -------
    float
        S(B) in bits.  Equals t14_formula(thetas, M) by CZ-T14.
    """
    M = np.asarray(M, dtype=int)
    n_B, n_A = M.shape
    N = n_A + n_B

    psi_A = rx(thetas[0])
    for t in thetas[1:]:
        psi_A = np.kron(psi_A, rx(t))

    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    psi_B = plus.copy()
    for _ in range(n_B - 1):
        psi_B = np.kron(psi_B, plus)

    psi = np.kron(psi_A, psi_B)

    # apply CZ(i, n_A+j) for each (i,j) with M[j,i]=1
    for j in range(n_B):
        for i in range(n_A):
            if M[j, i]:
                for idx in range(2 ** N):
                    b = format(idx, f"0{N}b")
                    if b[i] == "1" and b[n_A + j] == "1":
                        psi[idx] *= -1

    return vne_statevector(psi, n_B)
