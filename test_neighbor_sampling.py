"""Sanity test for exact uniform neighbor sampling.

The Monte Carlo update must choose one of the focal node's actual neighbors
uniformly.  This test checks representative degrees that do not necessarily
divide a common maximum degree.
"""

from __future__ import annotations

import numpy as np


def sample_indices(deg: int, draws: int, seed: int = 20260622) -> np.ndarray:
    rng = np.random.default_rng(seed + deg)
    u = rng.random(draws)
    return np.minimum((u * deg).astype(np.int64), deg - 1)


def main() -> None:
    draws = 300_000
    tolerance = 0.006
    for deg in (3, 5, 6, 7):
        idx = sample_indices(deg, draws)
        freq = np.bincount(idx, minlength=deg) / draws
        expected = 1.0 / deg
        max_abs = float(np.max(np.abs(freq - expected)))
        print(f"degree={deg}: max_abs_deviation={max_abs:.6f}")
        if max_abs > tolerance:
            raise SystemExit(
                f"Neighbor sampling failed for degree {deg}: "
                f"max_abs_deviation={max_abs:.6f} > {tolerance:.6f}"
            )
    print("neighbor_sampling_uniformity=PASS")


if __name__ == "__main__":
    main()

