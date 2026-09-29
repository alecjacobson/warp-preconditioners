"""Load the dragon's full Laplacian for the library's factored operator."""

import numpy as np
import warp as wp
from dirichlet_inputs import load_laplacian
from dragon import upload

from warp_preconditioners import SquaredLaplacianOperator


def load_squared(perm, lanes=1):
    d = np.load("data/dirichlet/fields.npz")
    L = load_laplacian()
    free = d["free"][perm].astype(np.int32)
    return SquaredLaplacianOperator(
        upload(L, "cuda:0"),
        wp.array(d["mass"], dtype=wp.float64, device="cuda:0"),
        wp.array(free, dtype=int, device="cuda:0"),
        row_lanes=lanes,
    )
