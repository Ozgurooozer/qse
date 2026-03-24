"""
QSE – Quantum Stabilizer Entropy
=================================
Analytic von Neumann entropy for CNOT-based quantum circuits.

Modules
-------
- core   : state-vector simulation primitives
- entropy : T14 Walsh-Hadamard formula and VNE computation
- rank   : F₂ matrix rank
- layers : multi-layer XOR composition (T15A)
- gates  : CZ-gate equivalent (CZ-T14)
- mutual : mutual information utilities
"""

from .core    import rx, cx, cx_matrix
from .entropy import shannon, vne_statevector, t14_formula, t14_statevector
from .rank    import f2_rank
from .layers  import effective_matrix, multilayer_vne
from .gates   import cz_vne
from .mutual  import mutual_information

__version__ = "1.0.0"
__all__ = [
    "rx", "cx", "cx_matrix",
    "shannon", "vne_statevector", "t14_formula", "t14_statevector",
    "f2_rank",
    "effective_matrix", "multilayer_vne",
    "cz_vne",
    "mutual_information",
]
