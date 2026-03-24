# QSE — Quantum Stabilizer Entropy

[![Tests](https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

Analytic von Neumann entropy for CNOT-based quantum circuits, via Walsh–Hadamard transforms and **F₂ linear algebra**.

## Overview

For a quantum circuit that applies `Rₓ(θᵢ)` rotations on subsystem `A` and connects to `B` via a binary CNOT matrix `M`, the von Neumann entropy of `B` has an **exact closed form** (Theorem T14):

```
S(B) = H({ p_b })

where  p_b = (1/2^NB) Σ_s (-1)^{b·s} ∏_{i: (Mᵀs)_i=1} cos(θᵢ)
```

This package implements T14 and the surrounding theorem family:

| Theorem     | Statement                                                        |
|-------------|------------------------------------------------------------------|
| **T14**     | Walsh–Hadamard closed-form entropy formula                       |
| **T-RANK**  | `S(B) ≤ rank_F₂(M)` for all θ                                   |
| **T-OPT**   | `S(B) = rank_F₂(M)` at θ = π/2 (maximal entanglement)          |
| **T15A**    | Multi-layer: `M_eff = M₁ ⊕ M₂ ⊕ … ⊕ Mₖ` (mod 2)              |
| **T17**     | Extends T14 to arbitrary pure states on A                        |
| **T18**     | Period-2 entropy oscillation under repeated layer application    |
| **T20**     | Bidirectional CNOT: `U³ = I`, period-3 entropy dynamics          |
| **T-MI**    | Mutual information `I(B₁;B₂) ≥ 0` (subadditivity)              |
| **CZ-T14**  | CZ gates + `|+⟩` init are entropy-equivalent to CNOT + `|0⟩`   |

## Quickstart

```bash
git clone https://github.com/[username]/qse
cd qse
pip install numpy          # only dependency
python quick_proofs.py     # verify all 11 theorems (~10 seconds)
```

Expected output:
```
================================================================
  QSE Quick Proof Verification
================================================================

[T14]  Walsh-Hadamard formula
  ✓ PASS  T14 formula vs state-vector             max_err=1.03e-13
[T-RANK]  S(B) ≤ rank_F₂(M)
  ✓ PASS  VNE ≤ rank_F₂(M)                        violations=0/200
...
  11/11 proof tests passed
  All algebraic and numerical proofs verified.
```

## Installation

```bash
pip install .                      # install the qse package
pip install ".[dev]"               # + pytest for running tests
```

## Usage

```python
import numpy as np
from qse import t14_formula, f2_rank

# 2 A-qubits, 2 B-qubits, random connectivity
thetas = [np.pi / 3, np.pi / 4]
M = np.array([[1, 0],
              [0, 1]])

# Analytic entropy (fast)
S = t14_formula(thetas, M)
print(f"S(B) = {S:.6f} bits")

# Upper bound from F₂ rank
print(f"rank_F₂(M) = {f2_rank(M)}")   # S(B) ≤ this

# Maximum entropy at θ = π/2
S_max = t14_formula([np.pi/2, np.pi/2], M)
print(f"S_max = {S_max:.6f} = rank = {f2_rank(M)}")
```

### Multi-layer circuits (T15A)

```python
from qse import multilayer_vne
import numpy as np

M1 = np.array([[1, 0, 1]])
M2 = np.array([[0, 1, 1]])
thetas = [0.5, 1.0, 1.5]

# XOR composition: M_eff = M1 ⊕ M2
S = multilayer_vne(thetas, M1, M2, method="formula")
```

### Mutual information (T-MI)

```python
from qse import mutual_information
import numpy as np

M = np.array([[1, 0, 1],
              [0, 1, 1]])
thetas = [0.8, 1.2, 0.4]

I_B1_B2 = mutual_information(thetas, M, split=1)  # always ≥ 0
```

## Running the Test Suite

```bash
pytest tests/ -v
```

All 11 theorems are verified against exact state-vector simulation.
Observed numerical errors are consistently at machine precision (~1e-13),
well below the pass threshold of 1e-10.

## Repository Structure

```
qse/
├── qse/
│   ├── __init__.py      # public API
│   ├── core.py          # rx(), cx(), cx_matrix(), bits()
│   ├── entropy.py       # shannon(), vne_statevector(), t14_formula(), t14_statevector()
│   ├── rank.py          # f2_rank()
│   ├── layers.py        # effective_matrix(), multilayer_vne()
│   ├── gates.py         # cz_vne()
│   └── mutual.py        # mutual_information()
├── tests/
│   └── test_theorems.py # pytest suite for T14–T20
├── quick_proofs.py      # standalone verification script
├── pyproject.toml
└── README.md
```

## Paper

This library accompanies the paper:

> **Quantum Stabilizer Entropy: Analytic von Neumann Entropy for CNOT-Based Circuits**  
> via Walsh–Hadamard Transforms and F₂ Linear Algebra  
> [Authors], [Year]. [arXiv link]

The LaTeX source is in `arxiv_paper/qse_paper.tex`.

## License

MIT
