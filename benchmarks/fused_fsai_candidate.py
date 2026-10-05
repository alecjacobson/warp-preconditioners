"""Experimental recomputation-based fusion of G.T @ (G @ x)."""

from functools import lru_cache

import warp as wp

from warp_preconditioners import FSAI


@lru_cache(None)
def kernel(dtype, storage):
    @wp.kernel(enable_backward=False, module="unique")
    def apply(
        goff: wp.array(dtype=int),
        gcol: wp.array(dtype=int),
        gv: wp.array(dtype=storage),
        toff: wp.array(dtype=int),
        tcol: wp.array(dtype=int),
        tv: wp.array(dtype=storage),
        x: wp.array(dtype=dtype),
        y: wp.array(dtype=dtype),
        z: wp.array(dtype=dtype),
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        total = dtype(0)
        if alpha != dtype(0):
            for e in range(toff[row], toff[row + 1]):
                r = tcol[e]
                inner = dtype(0)
                for f in range(goff[r], goff[r + 1]):
                    inner += dtype(gv[f]) * x[gcol[f]]
                total += dtype(tv[e]) * inner
        total *= alpha
        if beta != dtype(0):
            total += beta * y[row]
        z[row] = total

    return apply


class FusedFSAI(FSAI):
    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(self.scalar_type).flatten() for v in (x, y, z))
        if x.ptr == z.ptr:
            wp.copy(self._tmp, x)
            x = self._tmp
        wp.launch(
            kernel(self.scalar_type, self.factor_dtype),
            self.G.nrow,
            [
                self.G.offsets,
                self.G.columns,
                self.G.values,
                self.GT.offsets,
                self.GT.columns,
                self.GT.values,
                x,
                y,
                z,
                self.scalar_type(alpha),
                self.scalar_type(beta),
            ],
            device=self.device,
        )
