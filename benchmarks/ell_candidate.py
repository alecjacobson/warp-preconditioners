"""Experimental coalesced ELL/CSR factor application; construction stays in Warp."""

from functools import lru_cache

import warp as wp

from warp_preconditioners import FSAI


@lru_cache(None)
def kernels(dtype):
    @wp.kernel(enable_backward=False)
    def pack(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        ec: wp.array2d(dtype=int),
        ev: wp.array2d(dtype=dtype),
    ):
        row = wp.tid()
        beg, end = offsets[row], offsets[row + 1]
        for slot in range(ec.shape[0]):
            e = beg + slot
            if e < end:
                ec[slot, row] = columns[e]
                ev[slot, row] = values[e]
            else:
                ec[slot, row] = -1
                ev[slot, row] = dtype(0)

    @wp.kernel(enable_backward=False)
    def mv(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        ec: wp.array2d(dtype=int),
        ev: wp.array2d(dtype=dtype),
        x: wp.array(dtype=dtype),
        y: wp.array(dtype=dtype),
        z: wp.array(dtype=dtype),
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        result = dtype(0)
        if alpha != dtype(0):
            beg, end = offsets[row], offsets[row + 1]
            count = end - beg
            if count <= ec.shape[0]:
                for slot in range(count):
                    result += ev[slot, row] * x[ec[slot, row]]
            else:
                for e in range(beg, end):
                    result += values[e] * x[columns[e]]
            result *= alpha
        if beta != dtype(0):
            result += beta * y[row]
        z[row] = result

    return pack, mv


class ELLFSAI(FSAI):
    def __init__(self, A, **kwargs):
        super().__init__(A, **kwargs)
        pack, self._mv = kernels(self.scalar_type)
        width = kwargs.get("max_row_size", 8)
        self._ell = []
        for factor, slots in [(self.G, width), (self.GT, min(64, max(16, width * 2)))]:
            ec = wp.empty((slots, factor.nrow), dtype=int, device=self.device)
            ev = wp.empty((slots, factor.nrow), dtype=self.scalar_type, device=self.device)
            wp.launch(
                pack,
                factor.nrow,
                [factor.offsets, factor.columns, factor.values, ec, ev],
                device=self.device,
            )
            self._ell.append((ec, ev))

    def _apply(self, x, y, z, alpha, beta):
        x = x.view(self.scalar_type).flatten()
        y = y.view(self.scalar_type).flatten()
        z = z.view(self.scalar_type).flatten()
        for f, (ec, ev), xx, yy, zz, aa, bb in [
            (self.G, self._ell[0], x, self._tmp, self._tmp, 1.0, 0.0),
            (self.GT, self._ell[1], self._tmp, y, z, alpha, beta),
        ]:
            wp.launch(
                self._mv,
                f.nrow,
                [
                    f.offsets,
                    f.columns,
                    f.values,
                    ec,
                    ev,
                    xx,
                    yy,
                    zz,
                    self.scalar_type(aa),
                    self.scalar_type(bb),
                ],
                device=self.device,
            )
