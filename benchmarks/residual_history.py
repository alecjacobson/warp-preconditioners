"""Sample Warp CG/CR without restarting their Krylov recurrences.

The adapter changes only the private loop driver in Warp 1.15, reusing
Warp's native GPU loop and do_cycle closure across observation intervals.
Solver vectors, search directions and reductions remain alive throughout.
It is process-local, restored on exit, and is only for benchmark diagnostics.
"""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp
import warp._src.optim.linear as implementation


def scalar(value):
    return float(value.numpy().reshape(-1)[0]) if isinstance(value, wp.array) else float(value)


@contextmanager
def sample_iterations(checkpoints, observe):
    native = implementation._run_capturable_loop
    points = sorted(set(int(i) for i in checkpoints if i > 0))

    def driver(
        do_cycle, r_norm_sq, maxiter, atol_sq, callback, check_every, use_cuda_graph, cycle_size=1
    ):
        if callback is not None or cycle_size != 1 or r_norm_sq.size != 1:
            raise ValueError("Sampler supports unbatched CG/CR without an additional callback")
        total = 0
        observe(0, np.sqrt(scalar(r_norm_sq)))
        for stop in sorted(set([p for p in points if p <= maxiter] + [int(maxiter)])):
            requested = stop - total
            result = native(
                do_cycle, r_norm_sq, requested, atol_sq, None, 0, use_cuda_graph, cycle_size
            )
            steps = int(scalar(result[0]))
            total += steps
            residual_sq = scalar(r_norm_sq)
            observe(total, np.sqrt(residual_sq))
            if steps < requested or residual_sq <= scalar(atol_sq):
                break
        return total, np.sqrt(scalar(r_norm_sq)), np.sqrt(scalar(atol_sq))

    with patch.object(implementation, "_run_capturable_loop", driver):
        yield
