#!/usr/bin/env python3
"""
quick_proofs.py
===============
Standalone script that verifies all QSE theorems without pytest.
Useful for a fast sanity-check or as a demo.

Usage
-----
    python quick_proofs.py

Expected output: all 11 tests PASS in ~10 seconds.
"""

import sys
import numpy as np

# allow running from repo root without installing the package
sys.path.insert(0, ".")

from qse.core    import bits, cx, cx_matrix, rx
from qse.entropy import shannon, t14_formula, t14_statevector, vne_statevector
from qse.rank    import f2_rank
from qse.layers  import multilayer_vne
from qse.gates   import cz_vne
from qse.mutual  import mutual_information

RNG = np.random.default_rng(42)
TOL = 1e-10
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {tag}  {name:<45}  {detail}")
    results[name] = ok
    return ok


def rand_angles(n):  return list(RNG.uniform(0.1, np.pi - 0.1, n))
def rand_matrix(nb, na): return RNG.integers(0, 2, (nb, na))


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("  QSE Quick Proof Verification")
print("=" * 68)

# ── T14 ──────────────────────────────────────────────────────────────────────
print("\n[T14]  Walsh-Hadamard formula")
errs = []
for _ in range(50):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    errs.append(abs(t14_formula(rand_angles(na), rand_matrix(nb, na))
                    - t14_statevector(rand_angles(na), rand_matrix(nb, na))))
# Note: use same angles/matrix — redo properly
errs = []
for _ in range(50):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    th = rand_angles(na); M = rand_matrix(nb, na)
    errs.append(abs(t14_formula(th, M) - t14_statevector(th, M)))
check("T14 formula vs state-vector", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T-RANK ───────────────────────────────────────────────────────────────────
print("\n[T-RANK]  S(B) ≤ rank_F₂(M)")
viols = 0
for _ in range(200):
    na, nb = int(RNG.integers(1, 5)), int(RNG.integers(1, 4))
    if t14_formula(rand_angles(na), rand_matrix(nb, na)) > f2_rank(rand_matrix(nb, na)) + TOL:
        viols += 1
# redo with consistent M
viols = 0
for _ in range(200):
    na, nb = int(RNG.integers(1, 5)), int(RNG.integers(1, 4))
    th = rand_angles(na); M = rand_matrix(nb, na)
    if t14_formula(th, M) > f2_rank(M) + TOL:
        viols += 1
check("VNE ≤ rank_F₂(M)", viols == 0, f"violations={viols}/200")

# ── T-OPT ────────────────────────────────────────────────────────────────────
print("\n[T-OPT]  max S(B) = rank at θ = π/2")
errs = []
for _ in range(100):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    M = rand_matrix(nb, na)
    errs.append(abs(t14_formula([np.pi / 2] * na, M) - f2_rank(M)))
check("VNE(π/2) = rank_F₂(M)", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T15A ─────────────────────────────────────────────────────────────────────
print("\n[T15A]  k-layer circuit: M_eff = XOR")
errs = []
for _ in range(50):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    k  = int(RNG.integers(2, 5))
    th = rand_angles(na)
    layers = [rand_matrix(nb, na) for _ in range(k)]
    sv  = multilayer_vne(th, *layers, method="statevector")
    fml = multilayer_vne(th, *layers, method="formula")
    errs.append(abs(fml - sv))
check("T15A M_eff formula vs SV", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T-EVRENSEL ───────────────────────────────────────────────────────────────
print("\n[T-EVRENSEL]  Mixed ρ_A: off-diagonal terms irrelevant")
errs = []
for _ in range(40):
    na, nb = int(RNG.integers(1, 3)), int(RNG.integers(1, 3))
    M  = rand_matrix(nb, na)
    A  = RNG.standard_normal((2**na, 2**na)) + 1j * RNG.standard_normal((2**na, 2**na))
    rho = A @ A.conj().T; rho /= np.trace(rho)
    diag = np.real(np.diag(rho))
    probs = np.zeros(2**nb)
    for x in range(2**na):
        xv    = np.array([int(c) for c in format(x, f"0{na}b")], dtype=int)
        b_int = int("".join(map(str, (M @ xv) % 2)), 2)
        probs[b_int] += diag[x]
    formula = shannon(probs)
    ev, evec = np.linalg.eigh(rho)
    rho_B_tot = np.zeros((2**nb, 2**nb), dtype=complex)
    n_tot = na + nb
    for i, lam in enumerate(ev):
        if lam < 1e-12: continue
        psi_a = evec[:, i]; psi_b = np.zeros(2**nb, dtype=complex); psi_b[0] = 1.
        psi = np.kron(psi_a, psi_b)
        for j in range(nb):
            for ii in range(na):
                if M[j, ii]: psi = cx(psi, ii, na + j, n_tot)
        rho_f = np.outer(psi, psi.conj())
        rho_B = np.zeros((2**nb, 2**nb), dtype=complex)
        for a in range(2**na):
            rho_B += rho_f[a*2**nb:(a+1)*2**nb, a*2**nb:(a+1)*2**nb]
        rho_B_tot += lam * rho_B
    ev2 = np.linalg.eigvalsh(rho_B_tot.real); ev2 = ev2[ev2 > 1e-12]
    sv  = float(-np.sum(ev2 * np.log2(ev2))) if len(ev2) else 0.
    errs.append(abs(formula - sv))
check("T-EVRENSEL formula vs SV", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T17 ──────────────────────────────────────────────────────────────────────
print("\n[T17]  General pure state + blindness condition")
errs = []; blind_ok = 0; N_t17 = 50
for _ in range(N_t17):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    M = rand_matrix(nb, na)
    psi_a = RNG.standard_normal(2**na) + 1j * RNG.standard_normal(2**na)
    psi_a /= np.linalg.norm(psi_a)
    probs = np.zeros(2**nb)
    for x in range(2**na):
        if abs(psi_a[x]) < 1e-12: continue
        xv    = bits(x, na); b_int = int("".join(map(str, (M @ xv) % 2)), 2)
        probs[b_int] += abs(psi_a[x])**2
    formula = shannon(probs)
    psi_b = np.zeros(2**nb, dtype=complex); psi_b[0] = 1.
    psi   = np.kron(psi_a, psi_b)
    for j in range(nb):
        for i in range(na):
            if M[j, i]: psi = cx(psi, i, na + j, na + nb)
    sv = vne_statevector(psi, nb)
    errs.append(abs(formula - sv))
    support = [x for x in range(2**na) if abs(psi_a[x]) > 1e-8]
    blind_pred = all(
        not np.any((M @ bits(support[i] ^ support[j], na)) % 2)
        for i in range(len(support)) for j in range(i + 1, len(support))
    )
    if blind_pred == (sv < 1e-10): blind_ok += 1
check("T17 formula vs SV", max(errs) < TOL, f"max_err={max(errs):.2e}")
check("T17 blindness accuracy", blind_ok == N_t17, f"{blind_ok}/{N_t17}")

# ── T18 ──────────────────────────────────────────────────────────────────────
print("\n[T18]  Periodic sequence, period = 2")
errs = []
for _ in range(100):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    th = rand_angles(na); M = rand_matrix(nb, na)
    if not M.any(): continue
    v1 = multilayer_vne(th, M, method="formula")
    v3 = multilayer_vne(th, M, M, M, method="formula")
    errs.append(abs(v1 - v3))
check("T18 period-2 (k=1 == k=3)", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T20 ──────────────────────────────────────────────────────────────────────
print("\n[T20]  Bidirectional CNOT, U³ = I, period = 3")
N = 4; I16 = np.eye(2**N)
U_fwd = cx_matrix(1, 3, N) @ cx_matrix(0, 2, N)
U_bwd = cx_matrix(3, 1, N) @ cx_matrix(2, 0, N)
U = U_bwd @ U_fwd
U3_err  = np.max(np.abs(U @ U @ U - I16))
U2_is_I = np.max(np.abs(U @ U - I16)) < TOL
check("T20 U³ = I (matrix)", U3_err < TOL,
      f"max|U³−I|={U3_err:.2e}  U²=I:{U2_is_I}")

def t20_vne(t_A, t_B, k):
    M  = np.eye(2, dtype=int); na = nb = 2; N = 4
    psi = np.kron(np.kron(rx(t_A), rx(t_A)), np.kron(rx(t_B), rx(t_B)))
    for _ in range(k - 1):
        for i in range(na):
            for j in range(nb):
                if M[j, i]: psi = cx(psi, i, na + j, N)
        for j in range(nb):
            for i in range(na):
                if M[j, i]: psi = cx(psi, na + j, i, N)
    for i in range(na):
        for j in range(nb):
            if M[j, i]: psi = cx(psi, i, na + j, N)
    return vne_statevector(psi, nb)

errs = []
for _ in range(20):
    tA = float(RNG.uniform(0.2, np.pi - 0.2))
    tB = float(RNG.uniform(0.2, np.pi - 0.2))
    errs.append(abs(t20_vne(tA, tB, 1) - t20_vne(tA, tB, 4)))
check("T20 period-3 (k=1 == k=4)", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ── T-MI ─────────────────────────────────────────────────────────────────────
print("\n[T-MI]  Mutual information I(B₁;B₂) ≥ 0")
viols = 0
for _ in range(200):
    na = int(RNG.integers(1, 5)); th = rand_angles(na); M = rand_matrix(2, na)
    if mutual_information(th, M) < -TOL: viols += 1
check("I(B₁;B₂) ≥ 0", viols == 0, f"violations={viols}/200")

# ── CZ-T14 ───────────────────────────────────────────────────────────────────
print("\n[CZ-T14]  CZ gate + |+⟩ init: equivalent to CNOT + |0⟩ init")
errs = []
for _ in range(30):
    na, nb = int(RNG.integers(1, 4)), int(RNG.integers(1, 3))
    th = rand_angles(na); M = rand_matrix(nb, na)
    errs.append(abs(t14_formula(th, M) - cz_vne(th, M)))
check("CZ-T14 formula vs SV", max(errs) < TOL, f"max_err={max(errs):.2e}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 68)
print("  SUMMARY")
print("=" * 68)
passed = sum(results.values()); total = len(results)
print(f"\n  {passed}/{total} proof tests passed\n")
for name, ok in results.items():
    print(f"  {'✓' if ok else '✗'}  {name}")
print()
if passed == total:
    print("  All algebraic and numerical proofs verified.\n")
    sys.exit(0)
else:
    print(f"  {total - passed} test(s) failed.\n")
    sys.exit(1)
