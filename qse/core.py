"""
qse.core
--------
Low-level quantum state primitives.

All states are represented as numpy arrays of complex amplitudes in
the computational basis, ordered as |q₀ q₁ … q_{N-1}⟩ with q₀ the
most-significant bit.
"""

import numpy as np


def bits(n: int, width: int) -> np.ndarray:
    """Return the binary representation of *n* as a length-*width* int array.

    Parameters
    ----------
    n : int
        Non-negative integer to convert.
    width : int
        Number of bits (zero-padded on the left).

    Returns
    -------
    np.ndarray of dtype int, shape (width,)
    """
    return np.array([int(c) for c in format(n, f"0{width}b")], dtype=int)


def rx(theta: float) -> np.ndarray:
    """Single-qubit Rₓ(θ) state: Rₓ(θ)|0⟩ = cos(θ/2)|0⟩ − i sin(θ/2)|1⟩.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray of dtype complex128, shape (2,)
    """
    return np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)], dtype=complex)


def cx(psi: np.ndarray, ctrl: int, tgt: int, n_qubits: int) -> np.ndarray:
    """Apply a CNOT gate to state *psi*.

    Parameters
    ----------
    psi : np.ndarray, shape (2**n_qubits,)
        Input state vector (need not be normalised).
    ctrl : int
        Index of the control qubit (0 = most significant).
    tgt : int
        Index of the target qubit.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray of dtype complex128, shape (2**n_qubits,)
        New state vector after applying CNOT(ctrl, tgt).
    """
    out = np.zeros_like(psi)
    for idx in range(2 ** n_qubits):
        b = list(format(idx, f"0{n_qubits}b"))
        if b[ctrl] == "1":
            b[tgt] = "0" if b[tgt] == "1" else "1"
        out[int("".join(b), 2)] += psi[idx]
    return out


def cx_matrix(ctrl: int, tgt: int, n_qubits: int) -> np.ndarray:
    """Return the 2ⁿ × 2ⁿ unitary matrix for CNOT(ctrl, tgt).

    Parameters
    ----------
    ctrl : int
        Control qubit index.
    tgt : int
        Target qubit index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray of dtype float64, shape (2**n_qubits, 2**n_qubits)
    """
    dim = 2 ** n_qubits
    U = np.eye(dim)
    for idx in range(dim):
        b = list(format(idx, f"0{n_qubits}b"))
        if b[ctrl] == "1":
            b2 = list(b)
            b2[tgt] = "0" if b2[tgt] == "1" else "1"
            j = int("".join(b2), 2)
            U[j, idx] = 1.0
            U[idx, idx] = 0.0
    return U
