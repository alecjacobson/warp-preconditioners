"""Experimental block-adaptive FSAI; kept outside the supported API until measured."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp
from warp.optim.linear import LinearOperator

from warp_preconditioners.fsai_pack import pack_factor
from warp_preconditioners.sparse_operator import _matvec


@lru_cache(None)
def kernels(dtype, block, max_blocks):
    width = block * max_blocks
    mat = wp.types.matrix((width, width), dtype)
    rhs_mat = wp.types.matrix((width, block), dtype)
    diag_mat = wp.types.matrix((block, block), dtype)
    pattern_type = wp.types.vector(max_blocks, int)
    block_type = wp.types.matrix((block, block), dtype)

    @wp.func
    def entry(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=block_type),
        i: int,
        j: int,
    ):
        lo = offsets[i // wp.static(block)]
        hi = offsets[i // wp.static(block) + 1]
        end = hi
        while lo < hi:
            mid = (lo + hi) // 2
            if columns[mid] < j // wp.static(block):
                lo = mid + 1
            else:
                hi = mid
        val = dtype(0)
        if lo < end:
            if columns[lo] == j // wp.static(block):
                val = values[lo][i % wp.static(block), j % wp.static(block)]
        return val

    @wp.kernel(enable_backward=False, module="unique")
    def diagonal(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=block_type),
        scale: wp.array(dtype=dtype),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        d = entry(offsets, columns, values, i, i)
        if d > dtype(0) and wp.isfinite(d):
            scale[i] = dtype(1) / wp.sqrt(d)
        else:
            wp.atomic_add(status, 0, 1)
            scale[i] = dtype(0)

    @wp.kernel(enable_backward=False, module="unique")
    def build(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=block_type),
        scale: wp.array(dtype=dtype),
        cols: wp.array(dtype=int),
        vals: wp.array(dtype=dtype),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        pattern = pattern_type(-1)
        pattern[0] = i
        size = int(1)
        z = rhs_mat(dtype(0))
        for step in range(wp.static(max_blocks)):
            chol = mat(dtype(0))
            # Factor the selected, equilibrated principal submatrix.
            count = size * wp.static(block)
            for p in range(count):
                r = pattern[p // wp.static(block)] * wp.static(block) + p % wp.static(block)
                for q in range(p + 1):
                    c = pattern[q // wp.static(block)] * wp.static(block) + q % wp.static(block)
                    v = entry(offsets, columns, values, r, c) * scale[r] * scale[c]
                    for k in range(q):
                        v -= chol[p, k] * chol[q, k]
                    if p == q:
                        if v <= dtype(0) or not wp.isfinite(v):
                            wp.atomic_add(status, 0, 1)
                            v = dtype(1)
                        chol[p, q] = wp.sqrt(v)
                    else:
                        chol[p, q] = v / chol[q, q]
            # Solve B Z = [I, 0]^T: one RHS for each component of this block.
            for c in range(wp.static(block)):
                for p in range(count):
                    v = dtype(0)
                    if p == c:
                        v = dtype(1)
                    for q in range(p):
                        v -= chol[p, q] * z[q, c]
                    z[p, c] = v / chol[p, p]
                for rev in range(count):
                    p = count - rev - 1
                    v = z[p, c]
                    for q in range(p + 1, count):
                        v -= chol[q, p] * z[q, c]
                    z[p, c] = v / chol[p, p]
            if step == wp.static(max_blocks - 1):
                break
            best = int(-1)
            best_score = dtype(0)
            for p in range(size):
                row = pattern[p]
                for e in range(offsets[row], offsets[row + 1]):
                    candidate = columns[e]
                    selected = bool(candidate >= i)
                    for q in range(size):
                        if pattern[q] == candidate:
                            selected = True
                    if not selected:
                        score = dtype(0)
                        for r in range(wp.static(block)):
                            cr = candidate * wp.static(block) + r
                            for c in range(wp.static(block)):
                                residual = dtype(0)
                                for q in range(count):
                                    j = pattern[q // wp.static(block)] * wp.static(
                                        block
                                    ) + q % wp.static(block)
                                    residual += (
                                        entry(offsets, columns, values, cr, j)
                                        * scale[cr]
                                        * scale[j]
                                        * z[q, c]
                                    )
                                score += residual * residual
                        if score > best_score or (score == best_score and candidate < best):
                            best, best_score = candidate, score
            if best < 0 or best_score == dtype(0):
                break
            pattern[size] = best
            size += 1
        # Normalize the block row: G = chol(Z_ii)^-1 Z^T D^-1/2.
        norm = diag_mat(dtype(0))
        for r in range(wp.static(block)):
            for c in range(r + 1):
                v = z[r, c]
                for k in range(c):
                    v -= norm[r, k] * norm[c, k]
                if r == c:
                    norm[r, c] = wp.sqrt(v)
                else:
                    norm[r, c] = v / norm[c, c]
        for q in range(size * wp.static(block)):
            for r in range(wp.static(block)):
                v = z[q, r]
                for k in range(r):
                    v -= norm[r, k] * z[q, k]
                z[q, r] = v / norm[r, r]
        for r in range(wp.static(block)):
            for q in range(wp.static(width)):
                e = (i * wp.static(block) + r) * wp.static(width) + q
                cols[e] = -1
                vals[e] = dtype(0)
                if q < size * wp.static(block):
                    j = pattern[q // wp.static(block)] * wp.static(block) + q % wp.static(block)
                    cols[e] = j
                    vals[e] = z[q, r] * scale[j]

    return diagonal, build


class BlockFSAI(LinearOperator):
    def __init__(self, a, max_blocks=2, apply_lanes=1, factor_dtype=wp.float32):
        if a.row_counts is not None:
            a = sp.bsr_copy(a)
        block = a.block_shape[0]
        assert block == a.block_shape[1] and block > 1
        n, dtype, device = a.shape[0], a.scalar_type, a.device
        width = block * max_blocks
        scale = wp.empty(n, dtype=dtype, device=device)
        status = wp.zeros(1, dtype=int, device=device)
        cols = wp.empty(n * width, dtype=int, device=device)
        vals = wp.empty(n * width, dtype=dtype, device=device)
        diagonal, build = kernels(dtype, block, max_blocks)
        wp.launch(diagonal, n, [a.offsets, a.columns, a.values, scale, status], device=device)
        wp.launch(
            build,
            a.nrow,
            [a.offsets, a.columns, a.values, scale, cols, vals, status],
            device=device,
        )
        if status.numpy()[0]:
            raise ValueError("Nonpositive block FSAI pivot")
        self.G = pack_factor(n, width, cols, vals)
        if factor_dtype != dtype:
            self.G = sp.bsr_copy(self.G, scalar_type=factor_dtype)
        self.GT = sp.bsr_transposed(self.G)
        self._tmp = wp.empty(n, dtype=dtype, device=device)
        self.apply_lanes = apply_lanes
        super().__init__(a.shape, a.dtype, device, self._apply)

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(self.scalar_type).flatten() for v in (x, y, z))
        _matvec(self.G, x, self._tmp, self._tmp, 1.0, 0.0, self.apply_lanes)
        _matvec(self.GT, self._tmp, y, z, alpha, beta, self.apply_lanes)


@lru_cache(None)
def block_product(dtype, storage, block):
    matrix = wp.types.matrix((block, block), storage)
    vector = wp.types.vector(block, dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def apply(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=matrix),
        x: wp.array(dtype=vector),
        y: wp.array(dtype=vector),
        z: wp.array(dtype=vector),
        alpha: dtype,
        beta: dtype,
    ):
        row = wp.tid()
        result = vector(dtype(0))
        if alpha != dtype(0):
            for e in range(offsets[row], offsets[row + 1]):
                xx = x[columns[e]]
                value = values[e]
                for r in range(wp.static(block)):
                    for c in range(wp.static(block)):
                        result[r] += dtype(value[r, c]) * xx[c]
        result *= alpha
        if beta != dtype(0):
            result += beta * y[row]
        z[row] = result

    return apply


class BsrBlockFSAI(BlockFSAI):
    def __init__(self, a, **kw):
        super().__init__(a, **kw)
        block = a.block_shape[0]
        self.B = sp.bsr_copy(self.G, block_shape=(block, block))
        self.BT = sp.bsr_transposed(self.B)
        self._vec = wp.types.vector(block, a.scalar_type)
        self._mv = block_product(a.scalar_type, self.B.scalar_type, block)

    def _apply(self, x, y, z, alpha, beta):
        x, y, z = (v.view(self._vec).flatten() for v in (x, y, z))
        tmp = self._tmp.reshape((-1, self.B.block_shape[0])).view(self._vec).flatten()
        for a, u, v, out, aa, bb in [
            (self.B, x, tmp, tmp, 1.0, 0.0),
            (self.BT, tmp, y, z, alpha, beta),
        ]:
            wp.launch(
                self._mv,
                a.nrow,
                [
                    a.offsets,
                    a.columns,
                    a.values,
                    u,
                    v,
                    out,
                    self.scalar_type(aa),
                    self.scalar_type(bb),
                ],
                device=self.device,
            )
