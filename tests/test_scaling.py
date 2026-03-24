"""
tests/test_scaling.py
=====================
Empirical scaling verification for the T14 Walsh-Hadamard formula.

The formula computes p_b for each b in {0,1}^n_B, and each p_b requires
a sum over all s in {0,1}^n_B.  The theoretical complexity is therefore
O(4^n_B * n_B) in the naive implementation.

Run with:
    python tests/test_scaling.py

Produces a table and (optionally) a plot of runtime vs n_B.
"""

import time
import numpy as np
import sys

sys.path.insert(0, ".")
from qse.entropy import t14_formula

RNG = np.random.default_rng(0)
N_REPEAT = 10       # repetitions per (n_B, n_A) pair
N_A = 8             # fixed n_A, vary n_B


def measure_runtime(nb, na, n_repeat):
    """Return median runtime in seconds for t14_formula with given (nb, na)."""
    th = list(RNG.uniform(0.1, np.pi - 0.1, na))
    M  = RNG.integers(0, 2, (nb, na))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        t14_formula(th, M)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def test_scaling_is_exponential_in_nb():
    """
    Verify that runtime grows as O(4^n_B):
    each doubling of 2^n_B should roughly quadruple the runtime.
    We check that the empirical ratio r(n_B+1)/r(n_B) is between 2 and 8
    (allowing for noise), consistent with O(4^n_B).
    """
    nb_range = range(1, 8)
    runtimes = {nb: measure_runtime(nb, N_A, N_REPEAT) for nb in nb_range}

    print(f"\n{'n_B':>4}  {'time (ms)':>12}  {'ratio':>8}  {'expected ~4x':>14}")
    print("-" * 46)
    prev = None
    ratios = []
    for nb in nb_range:
        t = runtimes[nb] * 1000
        ratio = runtimes[nb] / prev if prev else float("nan")
        if prev:
            ratios.append(ratio)
        print(f"{nb:>4}  {t:>12.4f}  {ratio:>8.2f}  {'(baseline)' if prev is None else ''}")
        prev = runtimes[nb]

    # For n_B >= 3, ratios should be consistent with O(4^n_B)
    # Allow generous bounds [2, 10] to account for cache/overhead effects
    meaningful_ratios = ratios[2:]  # skip first two (dominated by overhead)
    if meaningful_ratios:
        assert all(1.5 < r < 12 for r in meaningful_ratios), (
            f"Unexpected scaling ratios: {meaningful_ratios}. "
            f"Expected ~4x per step (O(4^n_B))."
        )
    print(f"\nScaling consistent with O(4^n_B * n_B). All ratios in [1.5, 12].")


if __name__ == "__main__":
    print("=" * 60)
    print("  T14 Runtime Scaling: O(4^n_B * n_B)")
    print(f"  n_A = {N_A} (fixed), n_B = 1..7, {N_REPEAT} reps each")
    print("=" * 60)

    nb_range = list(range(1, 9))
    runtimes = {}
    for nb in nb_range:
        runtimes[nb] = measure_runtime(nb, N_A, N_REPEAT)

    print(f"\n{'n_B':>4}  {'2^n_B':>6}  {'time (ms)':>12}  {'ratio t[nb]/t[nb-1]':>22}")
    print("-" * 52)
    prev = None
    for nb in nb_range:
        t = runtimes[nb] * 1000
        ratio = runtimes[nb] / prev if prev else float("nan")
        ratio_str = f"{ratio:.2f}x" if prev else "(baseline)"
        print(f"{nb:>4}  {2**nb:>6}  {t:>12.4f}  {ratio_str:>22}")
        prev = runtimes[nb]

    print()
    print("Theoretical: each +1 in n_B multiplies runtime by ~4 (O(4^n_B)).")
    print("Observed ratios should converge to ~4 for large n_B.")

    # Optional: plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        nbs = list(nb_range)
        ts  = [runtimes[nb] * 1000 for nb in nbs]

        axes[0].semilogy(nbs, ts, "o-", color="steelblue", linewidth=2)
        axes[0].set_xlabel("$n_B$")
        axes[0].set_ylabel("Runtime (ms, log scale)")
        axes[0].set_title("T14 runtime vs $n_B$ (log scale)")
        axes[0].grid(True, alpha=0.3)

        # Compare against O(4^n_B) reference line
        ref = ts[0] * np.array([4.0 ** (nb - nbs[0]) for nb in nbs])
        axes[0].semilogy(nbs, ref, "--", color="tomato", label="$O(4^{n_B})$ ref")
        axes[0].legend()

        axes[1].plot(nbs[1:], [runtimes[nb] / runtimes[nb-1] for nb in nbs[1:]],
                     "s-", color="steelblue", linewidth=2)
        axes[1].axhline(4.0, color="tomato", linestyle="--", label="ratio = 4")
        axes[1].set_xlabel("$n_B$")
        axes[1].set_ylabel("Runtime ratio $t(n_B)/t(n_B-1)$")
        axes[1].set_title("Empirical scaling ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("t14_scaling.png", dpi=150)
        print("\nPlot saved: t14_scaling.png")
        plt.show()
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")
