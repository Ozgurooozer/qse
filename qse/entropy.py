"""
qse.entropy
-----------
Von Neumann entropy computations.

Key results implemented
~~~~~~~~~~~~~~~~~~~~~~~
**T14 (Walsh-Hadamard formula)**
    For a product input state ⊗ᵢ Rₓ(θᵢ)|0⟩ on subsystem A and |0…0⟩ on B,
    connected by CNOT gates encoded in a binary matrix M ∈ {0,1}^{NB×NA},
    the von Neumann entropy of subsystem B equals the Shannon entropy of a
    probability distribution computed via a Walsh-Hadamard transform:

        S(B) = H({ p_b }) ,

    where

        p_b = (1/2^NB) Σ_s (-1)^{b·s} ∏_{i: (Mᵀs)_i=1} cos(θᵢ) .

**T-OPT**
    At θ = π/2 (all qubits) the formula reduces to S(B) = rank_F₂(M).

**T-RANK**
    S(B) ≤ rank_F₂(M) for all θ.
"""

from __future__ import annotations

import numpy as np

from .core import bits, rx, cx


# ──────────────────────────────────────────────────────────────────────────────
# Shannon entropy
# ──────────────────────────────────────────────────────────────────────────────

def shannon(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits.

    Parameters
    ----------
    p : array-like of non-negative floats
        Probability vector (automatically normalised).

    Returns
    -------
    float
        H(p) = −Σᵢ pᵢ log₂ pᵢ  (0 log 0 ≡ 0).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-15, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


# ──────────────────────────────────────────────────────────────────────────────
# Von Neumann entropy via partial trace
# ──────────────────────────────────────────────────────────────────────────────

def vne_statevector(psi: np.ndarray, n_B: int) -> float:
    """Von Neumann entropy of subsystem B for a pure bipartite state |ψ⟩.

    The Hilbert space is partitioned as H_A ⊗ H_B where dim(H_B) = 2^n_B
    and the B qubits occupy the *last* n_B positions.

    Parameters
    ----------
    psi : np.ndarray, shape (2**N,)
        Normalised state vector.
    n_B : int
        Number of qubits in subsystem B.

    Returns
    -------
    float
        S(B) = −Tr[ρ_B log₂ ρ_B].
    """
    N = int(round(np.log2(len(psi))))
    n_A = N - n_B
    d_A, d_B = 2 ** n_A, 2 ** n_B

    rho = np.outer(psi, psi.conj())
    rho_B = np.zeros((d_B, d_B), dtype=complex)
    for a in range(d_A):
        rho_B += rho[a * d_B : (a + 1) * d_B, a * d_B : (a + 1) * d_B]

    ev = np.linalg.eigvalsh(rho_B.real)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev))) if len(ev) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# T14 – analytic Walsh-Hadamard formula
# ──────────────────────────────────────────────────────────────────────────────

def t14_formula(thetas: list[float], M: np.ndarray) -> float:
    """Compute S(B) via the T14 Walsh-Hadamard formula (Theorem 14).

    Parameters
    ----------
    thetas : sequence of float, length NA
        Rotation angles θᵢ for each A-qubit.
    M : np.ndarray of int, shape (NB, NA)
        Binary connectivity matrix. M[j, i] = 1 iff a CNOT gate connects
        A-qubit i to B-qubit j.

    Returns
    -------
    float
        S(B) in bits.

    Notes
    -----
    The formula is:

        p_b = (1/2^NB) Σ_{s ∈ {0,1}^NB} (-1)^{b·s} φ(s)

    where φ(s) = ∏_{i : (Mᵀ s)_i = 1} cos(θᵢ).

    This equals the Shannon entropy H({|p_b|}) after normalisation.
    """
    M = np.asarray(M, dtype=int)
    if not M.any():
        return 0.0

    n_B = M.shape[0]
    cos_t = np.cos(np.asarray(thetas, dtype=float))
    probs = np.empty(2 ** n_B)

    for b_int in range(2 ** n_B):
        b = bits(b_int, n_B)
        p = 0.0
        for s_int in range(2 ** n_B):
            s = bits(s_int, n_B)
            MT_s = (M.T @ s) % 2
            phi = float(np.prod(cos_t[MT_s == 1])) if MT_s.any() else 1.0
            p += (-1) ** int(np.dot(b, s) % 2) * phi
        probs[b_int] = p / (2 ** n_B)

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total < 1e-15:
        return 0.0
    return shannon(probs / total)


# ──────────────────────────────────────────────────────────────────────────────
# State-vector reference implementation of T14 setup
# ──────────────────────────────────────────────────────────────────────────────

def t14_statevector(thetas: list[float], M: np.ndarray) -> float:
    """Compute S(B) directly via state-vector simulation (reference/test).

    Prepares ψ_A = ⊗ᵢ Rₓ(θᵢ)|0⟩, ψ_B = |0…0⟩, applies CNOTs according
    to M, then computes the exact von Neumann entropy of subsystem B.

    Parameters
    ----------
    thetas : sequence of float, length NA
        Rotation angles for A-qubits.
    M : np.ndarray of int, shape (NB, NA)
        Binary connectivity matrix.

    Returns
    -------
    float
        S(B) in bits.
    """
    M = np.asarray(M, dtype=int)
    n_B, n_A = M.shape
    N = n_A + n_B

    psi_A = rx(thetas[0])
    for t in thetas[1:]:
        psi_A = np.kron(psi_A, rx(t))

    psi_B = np.zeros(2 ** n_B, dtype=complex)
    psi_B[0] = 1.0
    psi = np.kron(psi_A, psi_B)

    for j in range(n_B):
        for i in range(n_A):
            if M[j, i]:
                psi = cx(psi, i, n_A + j, N)

    return vne_statevector(psi, n_B)
