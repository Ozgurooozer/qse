"""
tests/test_theorems.py
======================
Numerical verification of QSE theorems T14–T20.

Run with:
    pytest tests/test_theorems.py -v

Each test corresponds to a theorem in the paper.  All comparisons are
against exact state-vector simulation; passing tolerance is 1e-10 (well
above numerical noise ~1e-13).
"""

import numpy as np
import pytest

from qse.core import bits, cx, cx_matrix, rx
from qse.entropy import (
    shannon,
    t14_formula,
    t14_statevector,
    vne_statevector,
)
from qse.rank import f2_rank
from qse.layers import effective_matrix, multilayer_vne
from qse.gates import cz_vne
from qse.mutual import mutual_information

RNG = np.random.default_rng(42)
TOL = 1e-10
N_RANDOM = 50          # samples for most tests
N_RANK   = 200         # samples for inequality tests


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def rand_angles(rng, n):
    return list(rng.uniform(0.1, np.pi - 0.1, n))

def rand_matrix(rng, nb, na):
    return rng.integers(0, 2, (nb, na))

def rand_params(rng):
    na = int(rng.integers(1, 4))
    nb = int(rng.integers(1, 3))
    return na, nb, rand_angles(rng, na), rand_matrix(rng, nb, na)


# ──────────────────────────────────────────────────────────────────────────────
# T14 – Walsh-Hadamard formula
# ──────────────────────────────────────────────────────────────────────────────

def test_t14_formula_matches_statevector():
    """T14: analytic formula agrees with state-vector to machine precision."""
    errs = []
    for _ in range(N_RANDOM):
        na, nb, th, M = rand_params(RNG)
        errs.append(abs(t14_formula(th, M) - t14_statevector(th, M)))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T-RANK – entropy bounded by F₂ rank
# ──────────────────────────────────────────────────────────────────────────────

def test_t_rank_vne_leq_f2_rank():
    """T-RANK: S(B) ≤ rank_F₂(M) for all θ."""
    violations = 0
    for _ in range(N_RANK):
        na = int(RNG.integers(1, 5))
        nb = int(RNG.integers(1, 4))
        th = rand_angles(RNG, na)
        M  = rand_matrix(RNG, nb, na)
        if t14_formula(th, M) > f2_rank(M) + TOL:
            violations += 1
    assert violations == 0, f"{violations}/{N_RANK} violations"


# ──────────────────────────────────────────────────────────────────────────────
# T-OPT – maximum entropy at θ = π/2
# ──────────────────────────────────────────────────────────────────────────────

def test_t_opt_max_entropy_at_halfpi():
    """T-OPT: S(B)|_{θ=π/2} = rank_F₂(M)."""
    errs = []
    for _ in range(N_RANDOM * 2):
        na = int(RNG.integers(1, 4))
        nb = int(RNG.integers(1, 3))
        M  = rand_matrix(RNG, nb, na)
        errs.append(abs(t14_formula([np.pi / 2] * na, M) - f2_rank(M)))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T15A – XOR composition of layers
# ──────────────────────────────────────────────────────────────────────────────

def test_t15a_multilayer_xor():
    """T15A: k-layer entropy equals T14 with M_eff = XOR of layer matrices."""
    errs = []
    for _ in range(N_RANDOM):
        na, nb, th, _ = rand_params(RNG)
        k      = int(RNG.integers(2, 5))
        layers = [rand_matrix(RNG, nb, na) for _ in range(k)]
        sv  = multilayer_vne(th, *layers, method="statevector")
        fml = multilayer_vne(th, *layers, method="formula")
        errs.append(abs(fml - sv))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T-EVRENSEL – mixed input state ρ_A
# ──────────────────────────────────────────────────────────────────────────────

def test_t_evrensel_mixed_input():
    """T-EVRENSEL: formula extends to mixed ρ_A (off-diagonal terms irrelevant)."""
    errs = []
    for _ in range(40):
        na = int(RNG.integers(1, 3))
        nb = int(RNG.integers(1, 3))
        M  = rand_matrix(RNG, nb, na)

        # random mixed state ρ_A
        A   = (RNG.standard_normal((2**na, 2**na))
               + 1j * RNG.standard_normal((2**na, 2**na)))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        diag = np.real(np.diag(rho))

        # formula: diagonal only (off-diagonal irrelevant)
        probs = np.zeros(2**nb)
        for x in range(2**na):
            xv    = bits(x, na)
            b_int = int("".join(map(str, (M @ xv) % 2)), 2)
            probs[b_int] += diag[x]
        formula = shannon(probs)

        # state-vector reference (spectral decomposition)
        ev, evec = np.linalg.eigh(rho)
        n_tot = na + nb
        rho_B_tot = np.zeros((2**nb, 2**nb), dtype=complex)
        for i, lam in enumerate(ev):
            if lam < 1e-12:
                continue
            psi_a = evec[:, i]
            psi_b = np.zeros(2**nb, dtype=complex); psi_b[0] = 1.0
            psi   = np.kron(psi_a, psi_b)
            for j in range(nb):
                for ii in range(na):
                    if M[j, ii]:
                        psi = cx(psi, ii, na + j, n_tot)
            rho_f = np.outer(psi, psi.conj())
            rho_B = np.zeros((2**nb, 2**nb), dtype=complex)
            for a in range(2**na):
                rho_B += rho_f[a * 2**nb : (a+1) * 2**nb,
                                a * 2**nb : (a+1) * 2**nb]
            rho_B_tot += lam * rho_B

        ev2 = np.linalg.eigvalsh(rho_B_tot.real)
        ev2 = ev2[ev2 > 1e-12]
        sv  = float(-np.sum(ev2 * np.log2(ev2))) if len(ev2) else 0.0
        errs.append(abs(formula - sv))

    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T17 – general pure state + blindness condition
# ──────────────────────────────────────────────────────────────────────────────

def test_t17_general_pure_state():
    """T17: formula holds for arbitrary pure state on A."""
    errs = []
    for _ in range(N_RANDOM):
        na, nb, _, M = rand_params(RNG)
        psi_a = (RNG.standard_normal(2**na)
                 + 1j * RNG.standard_normal(2**na))
        psi_a /= np.linalg.norm(psi_a)

        probs = np.zeros(2**nb)
        for x in range(2**na):
            if abs(psi_a[x]) < 1e-12:
                continue
            b_int = int("".join(map(str, (M @ bits(x, na)) % 2)), 2)
            probs[b_int] += abs(psi_a[x])**2
        formula = shannon(probs)

        psi_b = np.zeros(2**nb, dtype=complex); psi_b[0] = 1.0
        psi   = np.kron(psi_a, psi_b)
        for j in range(nb):
            for i in range(na):
                if M[j, i]:
                    psi = cx(psi, i, na + j, na + nb)
        sv = vne_statevector(psi, nb)
        errs.append(abs(formula - sv))

    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


def test_t17_blindness_condition():
    """T17 blindness: M maps support differences to 0 ⟺ S(B) = 0."""
    n_correct = 0
    n_total   = N_RANDOM

    for _ in range(n_total):
        na, nb, _, M = rand_params(RNG)
        psi_a = (RNG.standard_normal(2**na)
                 + 1j * RNG.standard_normal(2**na))
        psi_a /= np.linalg.norm(psi_a)

        psi_b = np.zeros(2**nb, dtype=complex); psi_b[0] = 1.0
        psi   = np.kron(psi_a, psi_b)
        for j in range(nb):
            for i in range(na):
                if M[j, i]:
                    psi = cx(psi, i, na + j, na + nb)
        sv = vne_statevector(psi, nb)

        support = [x for x in range(2**na) if abs(psi_a[x]) > 1e-8]
        blind_pred = all(
            not np.any((M @ bits(support[i] ^ support[j], na)) % 2)
            for i in range(len(support))
            for j in range(i + 1, len(support))
        )
        if blind_pred == (sv < 1e-10):
            n_correct += 1

    assert n_correct == n_total, f"{n_correct}/{n_total} correct"


# ──────────────────────────────────────────────────────────────────────────────
# T18 – period-2 entropy sequence
# ──────────────────────────────────────────────────────────────────────────────

def test_t18_period_two():
    """T18: applying the same layer twice returns to original entropy (period 2)."""
    errs = []
    for _ in range(N_RANDOM * 2):
        na, nb, th, M = rand_params(RNG)
        if not M.any():
            continue
        # k=1: M_eff = M;  k=3: M_eff = M⊕M⊕M = M
        v1 = multilayer_vne(th, M, method="formula")
        v3 = multilayer_vne(th, M, M, M, method="formula")
        errs.append(abs(v1 - v3))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T20 – bidirectional CNOT circuit, period 3
# ──────────────────────────────────────────────────────────────────────────────

def _build_t20_unitary():
    """Return U = U_bwd @ U_fwd for the bidirectional CNOT circuit on 4 qubits."""
    N = 4
    U_fwd = cx_matrix(1, 3, N) @ cx_matrix(0, 2, N)
    U_bwd = cx_matrix(3, 1, N) @ cx_matrix(2, 0, N)
    return U_bwd @ U_fwd


def test_t20_unitary_period_three():
    """T20: U³ = I for the bidirectional CNOT circuit."""
    U   = _build_t20_unitary()
    I16 = np.eye(2**4)
    err = np.max(np.abs(U @ U @ U - I16))
    assert err < TOL, f"max|U³ − I| = {err:.2e}"


def test_t20_unitary_not_period_two():
    """T20: U² ≠ I, confirming the minimal period is exactly 3."""
    U   = _build_t20_unitary()
    I16 = np.eye(2**4)
    err = np.max(np.abs(U @ U - I16))
    assert err > 1e-6, "U² = I unexpectedly — period is 2, not 3"


def _t20_vne(t_A: float, t_B: float, k: int) -> float:
    """VNE after k rounds of the bidirectional 2+2 qubit circuit."""
    M  = np.eye(2, dtype=int)
    na = nb = 2
    N  = 4

    psi = np.kron(np.kron(rx(t_A), rx(t_A)), np.kron(rx(t_B), rx(t_B)))

    for _ in range(k - 1):
        for i in range(na):
            for j in range(nb):
                if M[j, i]:
                    psi = cx(psi, i, na + j, N)
        for j in range(nb):
            for i in range(na):
                if M[j, i]:
                    psi = cx(psi, na + j, i, N)

    for i in range(na):
        for j in range(nb):
            if M[j, i]:
                psi = cx(psi, i, na + j, N)

    return vne_statevector(psi, nb)


def test_t20_entropy_period_three():
    """T20: VNE(k) = VNE(k+3) for bidirectional circuit."""
    errs = []
    for _ in range(20):
        t_A = float(RNG.uniform(0.2, np.pi - 0.2))
        t_B = float(RNG.uniform(0.2, np.pi - 0.2))
        errs.append(abs(_t20_vne(t_A, t_B, 1) - _t20_vne(t_A, t_B, 4)))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T-MI – mutual information non-negativity
# ──────────────────────────────────────────────────────────────────────────────

def test_t_mi_non_negative():
    """T-MI: I(B₁ ; B₂) ≥ 0 for all θ, M."""
    violations = 0
    for _ in range(N_RANK):
        na = int(RNG.integers(1, 5))
        th = rand_angles(RNG, na)
        M  = rand_matrix(RNG, 2, na)
        if mutual_information(th, M) < -TOL:
            violations += 1
    assert violations == 0, f"{violations}/{N_RANK} violations"


# ──────────────────────────────────────────────────────────────────────────────
# CZ-T14 – CZ gate equivalence
# ──────────────────────────────────────────────────────────────────────────────

def test_cz_t14_equivalence():
    """CZ-T14: CZ circuit with |+⟩ initialisation equals T14 formula."""
    errs = []
    for _ in range(30):
        na, nb, th, M = rand_params(RNG)
        errs.append(abs(t14_formula(th, M) - cz_vne(th, M)))
    assert max(errs) < TOL, f"max error = {max(errs):.2e}"
