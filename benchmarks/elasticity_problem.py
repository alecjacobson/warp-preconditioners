"""P1 tetrahedral isotropic elasticity: Warp assembly and Dirichlet elimination.

Each vertex has a 3-vector displacement and each stiffness entry is a 3x3
BSR block. The constrained vertices have zero displacement. No penalty,
mass regularizer, or external numerical solver is used.
"""

import numpy as np
import warp as wp
import warp.sparse as sp

mat43d = wp.types.matrix(shape=(4, 3), dtype=wp.float64)


@wp.kernel(enable_backward=False)
def element_blocks(
    vertices: wp.array(dtype=wp.vec3d),
    tets: wp.array(dtype=wp.vec4i),
    free_index: wp.array(dtype=int),
    lame: wp.float64,
    mu: wp.float64,
    material_scale: wp.array(dtype=wp.float64),
    fiber: wp.array(dtype=wp.vec3d),
    reinforcement: wp.array(dtype=wp.float64),
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    values: wp.array(dtype=wp.mat33d),
    nodal_volume: wp.array(dtype=wp.float64),
):
    e = wp.tid()
    tet = tets[e]
    origin = vertices[tet[0]]
    dm = wp.matrix_from_cols(
        vertices[tet[1]] - origin, vertices[tet[2]] - origin, vertices[tet[3]] - origin
    )
    volume = wp.abs(wp.determinant(dm)) / wp.float64(6)
    inv = wp.inverse(dm)
    gradients = mat43d(wp.float64(0))
    gradients[0] = -(inv[0] + inv[1] + inv[2])
    gradients[1] = inv[0]
    gradients[2] = inv[1]
    gradients[3] = inv[2]
    for i in range(4):
        wp.atomic_add(nodal_volume, tet[i], volume / wp.float64(4))
        gi = gradients[i]
        for j in range(4):
            out = 16 * e + 4 * i + j
            ri = int(free_index[tet[i]])
            cj = int(free_index[tet[j]])
            rows[out] = wp.max(ri, 0)
            cols[out] = wp.max(cj, 0)
            block = wp.mat33d(wp.float64(0))
            if ri >= 0 and cj >= 0:
                gj = gradients[j]
                block = (
                    volume
                    * material_scale[e]
                    * (
                        lame * wp.outer(gi, gj)
                        + mu * wp.outer(gj, gi)
                        + mu * wp.dot(gi, gj) * wp.identity(n=3, dtype=wp.float64)
                        + reinforcement[e]
                        * wp.dot(fiber[e], gi)
                        * wp.dot(fiber[e], gj)
                        * wp.outer(fiber[e], fiber[e])
                    )
                )
            values[out] = block


@wp.kernel(enable_backward=False)
def gravity_load(
    free: wp.array(dtype=int),
    volume: wp.array(dtype=wp.float64),
    density: wp.float64,
    gravity: wp.vec3d,
    rhs: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    rhs[i] = density * volume[free[i]] * gravity


def assemble(
    vertices,
    tets,
    fixed,
    young=1e7,
    poisson=0.35,
    density=1000.0,
    device="cuda:0",
    material_scale=None,
    fiber=None,
    reinforcement=None,
):
    # Additional energy density: tau/2 * (q^T epsilon q)^2, tau >= 0.
    # This is directional reinforcement, not a nearly incompressible material.
    if fiber is None:
        fiber = np.tile([1.0, 0.0, 0.0], (len(tets), 1))
    fiber = np.asarray(fiber, dtype=np.float64)
    if (
        fiber.shape != (len(tets), 3)
        or not np.all(np.isfinite(fiber))
        or np.any(np.linalg.norm(fiber, axis=1) == 0)
    ):
        raise ValueError("fiber must contain one finite nonzero direction per tet")
    fiber = fiber / np.linalg.norm(fiber, axis=1)[:, None]
    if reinforcement is None:
        reinforcement = np.zeros(len(tets))
    reinforcement = np.broadcast_to(
        np.asarray(reinforcement, dtype=np.float64), (len(tets),)
    ).copy()
    if not np.all(np.isfinite(reinforcement)) or np.any(reinforcement < 0):
        raise ValueError("reinforcement must be finite and nonnegative")
    if material_scale is None:
        material_scale = np.ones(len(tets))
    material_scale = np.asarray(material_scale, dtype=np.float64)
    if (
        material_scale.shape != (len(tets),)
        or not np.all(np.isfinite(material_scale))
        or np.any(material_scale <= 0)
    ):
        raise ValueError("material_scale must contain one finite positive multiplier per tet")
    if not young > 0 or not -1 < poisson < 0.5 or not density > 0:
        raise ValueError("Expected positive Young's modulus/density and -1 < Poisson ratio < 0.5")
    free = np.flatnonzero(~fixed).astype(np.int32)
    index = np.full(len(vertices), -1, dtype=np.int32)
    index[free] = np.arange(len(free), dtype=np.int32)
    with wp.ScopedDevice(device):
        rows = wp.empty(16 * len(tets), dtype=int)
        cols = wp.empty_like(rows)
        values = wp.empty(16 * len(tets), dtype=wp.mat33d)
        volume = wp.zeros(len(vertices), dtype=wp.float64)
        wp.launch(
            element_blocks,
            len(tets),
            [
                wp.array(vertices, dtype=wp.vec3d),
                wp.array(tets, dtype=wp.vec4i),
                wp.array(index, dtype=int),
                young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
                young / (2 * (1 + poisson)),
                wp.array(material_scale, dtype=wp.float64),
                wp.array(fiber, dtype=wp.vec3d),
                wp.array(reinforcement, dtype=wp.float64),
                rows,
                cols,
                values,
                volume,
            ],
        )
        matrix = sp.bsr_from_triplets(len(free), len(free), rows, cols, values)
        rhs = wp.empty(len(free), dtype=wp.vec3d)
        wp.launch(
            gravity_load,
            len(free),
            [wp.array(free, dtype=int), volume, density, wp.vec3d(0.0, 0.0, -9.81), rhs],
        )
    return matrix, rhs, volume, free


def cpu_matrix(matrix):
    """Download for independent diagnostics only; never used by a solver."""
    import scipy.sparse as ss

    nnz = matrix.nnz_sync()
    return ss.bsr_matrix(
        (matrix.values.numpy()[:nnz], matrix.columns.numpy()[:nnz], matrix.offsets.numpy()),
        shape=matrix.shape,
    ).tocsr()


def independent_force(vertices, tets, displacement, young, poisson, fiber=None, reinforcement=None):
    """Element strain/stress contraction, independent of the BSR assembly kernel."""
    out = np.zeros_like(vertices)
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    for start in range(0, len(tets), 20000):
        t = tets[start : start + 20000]
        edges = vertices[t[:, 1:]] - vertices[t[:, :1]]
        inv = np.linalg.inv(edges.transpose(0, 2, 1))
        g = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)
        grad = np.einsum("eia,eib->eab", displacement[t], g)
        stress = mu * (grad + grad.transpose(0, 2, 1))
        stress += lame * np.trace(grad, axis1=1, axis2=2)[:, None, None] * np.eye(3)
        if fiber is not None:
            q = fiber[start : start + len(t)]
            q = q / np.linalg.norm(q, axis=1)[:, None]
            tau = np.broadcast_to(reinforcement, (len(tets),))[start : start + len(t)]
            axial = np.einsum("ea,eab,eb->e", q, grad, q)
            stress += (tau * axial)[:, None, None] * np.einsum("ea,eb->eab", q, q)
        force = np.einsum("eab,eib->eia", stress, g)
        force *= (np.abs(np.linalg.det(edges)) / 6)[:, None, None]
        np.add.at(out, t.ravel(), force.reshape(-1, 3))
    return out
