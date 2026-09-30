"""Pack unique short FSAI supports directly into canonical CSR on device."""

from functools import lru_cache

import warp as wp
import warp.sparse as sp


@lru_cache(None)
def kernels(dtype, width):
    ivec = wp.types.vector(length=width, dtype=int)
    vec = wp.types.vector(length=width, dtype=dtype)

    @wp.kernel(enable_backward=False, module="unique")
    def sort_rows(
        columns: wp.array(dtype=int), values: wp.array(dtype=dtype), counts: wp.array(dtype=int)
    ):
        i = wp.tid()
        cols = ivec(-1)
        vals = vec(dtype(0))
        size = int(0)
        for p in range(wp.static(width)):
            c = columns[i * wp.static(width) + p]
            if c >= 0:
                v = values[i * wp.static(width) + p]
                q = size
                while q > 0:
                    if cols[q - 1] <= c:
                        break
                    cols[q] = cols[q - 1]
                    vals[q] = vals[q - 1]
                    q -= 1
                cols[q] = c
                vals[q] = v
                size += 1
        counts[i + 1] = size
        for p in range(size):
            columns[i * wp.static(width) + p] = cols[p]
            values[i * wp.static(width) + p] = vals[p]

    @wp.kernel(enable_backward=False, module="unique")
    def pack(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        outcols: wp.array(dtype=int),
        outvals: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        for e in range(offsets[i], offsets[i + 1]):
            p = i * wp.static(width) + e - offsets[i]
            outcols[e] = columns[p]
            outvals[e] = values[p]

    return sort_rows, pack


def pack_factor(n, width, columns, values):
    result = sp.bsr_zeros(n, n, values.dtype, device=values.device)
    counts = wp.zeros(n + 1, dtype=int, device=values.device)
    sort, pack = kernels(values.dtype, width)
    wp.launch(sort, n, [columns, values, counts], device=values.device)
    wp.utils.array_scan(counts, result.offsets)
    result.notify_nnz_changed()
    wp.launch(
        pack,
        n,
        [result.offsets, columns, values, result.columns, result.values],
        device=values.device,
    )
    return result


@lru_cache(None)
def scalarize_kernel(dtype, block_size):
    @wp.kernel(enable_backward=False, module="unique")
    def scalarize(
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        outoff: wp.array(dtype=int),
        outcol: wp.array(dtype=int),
        outval: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        r = i // wp.static(block_size)
        component = i % wp.static(block_size)
        start = offsets[r]
        count = offsets[r + 1] - start
        base = start * wp.static(block_size * block_size) + component * count * wp.static(
            block_size
        )
        outoff[i] = base
        for p in range(count):
            for c in range(wp.static(block_size)):
                dest = base + p * wp.static(block_size) + c
                outcol[dest] = columns[start + p] * wp.static(block_size) + c
                outval[dest] = values[
                    (start + p) * wp.static(block_size * block_size)
                    + component * wp.static(block_size)
                    + c
                ]
        if i == outoff.shape[0] - 2:
            outoff[i + 1] = base + count * wp.static(block_size)

    return scalarize


def scalarize_compact(A):
    """Expand canonical compact square BSR directly, retaining numerical zeros."""
    b = A.block_shape[0]
    result = sp.bsr_zeros(A.shape[0], A.shape[1], A.scalar_type, device=A.device)
    count = A.nnz_sync() * b * b
    result.columns = wp.empty(count, dtype=int, device=A.device)
    result.values = wp.empty(count, dtype=A.scalar_type, device=A.device)
    wp.launch(
        scalarize_kernel(A.scalar_type, b),
        result.nrow,
        [
            A.offsets,
            A.columns,
            A.values.view(A.scalar_type).flatten(),
            result.offsets,
            result.columns,
            result.values,
        ],
        device=A.device,
    )
    result.notify_nnz_changed(nnz=count)
    return result
