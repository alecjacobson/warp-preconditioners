# Nonlinear data smoothing on the dragon

The figure contains **four dragons in one Blender scene**, under a single
orthographic camera with shared lighting and ground. The centers are 2.08
scene units apart. Left to right:

1. Warp CG + adaptive FSAI (width 8), **converged at $k=28{,}688$ iterations**.
2. Warp CR + Jacobi, **$k=28{,}688$ iterations**.
3. Warp CR + Jacobi, **$10k=286{,}880$ iterations**.
4. Warp CR + Jacobi, **$100k=2{,}868{,}800$ iterations**.

**All four start from the same data function, $u_0=f$.** A separate
[initialization experiment](initialization.md) compares this with zero:
FSAI uses about 2.7% fewer iterations, while the early Jacobi fields are
substantially closer to the converged solution. This is evidence for the
choice in this example, not a claim that data initialization always wins.

## A nonlinear target and stronger smoothing

A linear combination of $x,y,z$ would still be affine. Instead, normalize
each coordinate to $[-1,1]$ using the original mesh bounding box, for example
$X=2(x-x_{\min})/(x_{\max}-x_{\min})-1$, and define

$$
\begin{aligned}
f(X,Y,Z)={}&\sin\bigl(\pi(0.9X+0.45Y-0.65Z)\bigr)\\
&+0.65\cos\bigl(\pi(0.35X-0.8Y+0.9Z)\bigr)\\
&+0.35\sin\bigl(2\pi(XZ+0.2Y)\bigr).
\end{aligned}
$$

This combines broad spatial oscillations with an $XZ$ interaction. The
normalization defines the signal only: geometry and the operator remain in
the original mesh coordinates. The target and its bounding-box parameters
are stored explicitly alongside the results; no random field is used.

We minimize

$$
E(u)=\frac{\alpha}{2}(u-f)^T M(u-f)
     +\frac{1}{2}(Ku)^T M^{-1}(Ku),
\qquad
(\alpha M+KM^{-1}K)u=\alpha Mf.
$$

$M$ is the diagonal mass matrix and $K$ is the positive cotangent stiffness
matrix. **The data weight is $\alpha=10^{-4}$**, down from $10^{-3}$ in the
previous height-field example; the smoothing weight stays 1. The
smoothing/data ratio is therefore ten times larger. Coefficients use the
original geometry units and are not invariant under rescaling the mesh.
The displayed scalar values are dimensionless.

The target's mass-weighted best-affine-fit residual is 87.3% of its standard
deviation; the **converged solution remains 85.6% non-affine** by the same
measure. In the previous converged height example this fraction was only
4.6%. Bending energy falls from about **44.13 to 0.04932**, a **99.89%**
reduction, while the RMS change from the data is about 0.1365. The curves
in the new field therefore belong to the converged solution, rather than
being an artifact of stopping the solver early. Affine fitting is only a
diagnostic and is never rendered or subtracted from the field.

## Budgets and convergence checks

The script determines $k$ afresh from FSAI reaching Warp's recursive relative
residual tolerance $10^{-8}$. Jacobi's budgets are derived from that actual
returned count, not hard-coded. Each displayed field is the final iterate
of one uninterrupted solve from the data, with no restarts or best-checkpoint
selection. Jacobi uses zero stopping tolerance to reach its requested
budgets; actual counts are verified. Setup and kernels are warmed before
timing. Setup time is recorded separately. Solve times refer to one scalar
right-hand side and exclude initialization copies and diagnostic checks.

The FSAI label means **the solver reached its stated stopping tolerance**.
Its recursive relative residual is about $8.89\times10^{-9}$; independently
computing $\|b-Au\|_2/\|b\|_2$ gives about $1.28\times10^{-3}$, and its
componentwise backward error is about $9.11\times10^{-13}$. The figure shows
the independent residual, not the smaller internal estimate. This
ill-conditioned system amplifies floating-point errors in residual evaluation.

An additional, unrendered FSAI run with budget $2k$ checks field stability.
Its mass-weighted relative difference from the displayed FSAI field must be
below $10^{-6}$; the recorded change is only **$3.34\times10^{-11}$**, with
maximum absolute change $1.16\times10^{-10}$. This run is separate from the
displayed timing. Jacobi's field differences from FSAI are also recorded,
without treating the FSAI field as an exact-arithmetic solution.

| Method | Iterations | Solve time (s) | Relative mass-norm difference from FSAI |
| --- | ---: | ---: | ---: |
| FSAI-CG | 28,688 | 10.04 | 0 |
| Jacobi-CR, k | 28,688 | 6.25 | 3.93e-2 |
| Jacobi-CR, 10k | 286,880 | 66.34 | 1.08e-2 |
| Jacobi-CR, 100k | 2,868,800 | 673.47 | 8.12e-9 |

## Checking what is visualized

- All three original benchmark RHS columns are checked against $M$ times the
  original mesh coordinates, confirming the vertex ordering. The **new** RHS
  is then assembled explicitly as $\alpha Mf$.
- The dumped stiffness agrees with an independent cotangent assembly on the
  displayed mesh (relative maximum difference about $1.05\times10^{-8}$,
  allowing for roundoff on very thin triangles across libigl builds).
- The original dumped matrix agrees with $M+KM^{-1}K$ to relative maximum
  error below $10^{-12}$; its mass coefficient is then adjusted to $\alpha$.
- Every field is downloaded directly from its Warp solve in float64. Its
  SHA-256 is verified before assigning it to the corresponding dragon.
- Blender's actual per-vertex `biharmonic_u` attribute is read back and checked
  against that field's float32 conversion. Conversion errors are recorded.

Each dragon uses the original 360,757 vertices and 721,510 triangles,
positioned by uniform scaling and translation only. No solution is replaced
with raw data, filtering, or an analytic approximation. Smooth shading
changes lighting, not the scalar data or geometry.

## Colors and rendering

All dragons share their combined scalar extrema and the same 26 intervals.
Scalars are interpolated over each original triangle **before** the constant
color ramp. No object has its own normalization, and prequantized vertex
RGB values are not interpolated.

The palette reproduces
`isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))`: an OKLab hue arc with
alternate interval lightness multiplied by 0.9. Its source functions are in
[gptoolbox at dd355405](https://github.com/alecjacobson/gptoolbox/tree/dd3554053237cd02d305b78db298d913a6196508),
under `imageprocessing/` and `mesh/isolines_stripe_map.m`. The translation's
license is retained in `third_party/gptoolbox-LICENSE-MIT.txt`.

Cycles/OptiX renders the four meshes simultaneously with shared contact
shadows. Composition adds labels and an unlit sRGB colorbar to this **single
render**, rather than combining separate dragon renders. Camera settings,
object positions, target definition, field hashes, and diagnostics are in
`assets/dragon-biharmonic-comparison.json`.

## Reproduce

Generate the C++ benchmark dumps using the main README instructions, then:

```bash
python -m pip install -e '.[visualization]'
OPENBLAS_NUM_THREADS=1 python visualization/solve_fields.py \
  --dir /tmp/dump --mesh /path/to/xyzrgb_dragon-720K.ply \
  --data-weight 0.0001 --initial-guess data --fsai-rtol 1e-8
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --width 4800 --height 1120 --samples 64
python visualization/compose.py
```

Use `--audit-initial-guess` for the [separate initialization test](initialization.md).
The optional `--audit-weights` mode tests its specified positive coefficients
on the **new nonlinear target**, independently of the displayed fields.

Solving uses `cuda:0`; rendering uses an OptiX GPU and Blender 4.5.3. The
editable scene is `data/visualization/four_dragons.blend`. Intermediate
fields (including the input `target`) and renders are also in ignored
`data/visualization/`. Published PNGs use Git LFS. The numerical solver
library still depends only on Warp; Matplotlib is an optional dependency
for the separate initialization plot.
