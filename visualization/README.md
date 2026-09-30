# Nonlinear data smoothing on the dragon

For interpolation without a data term, see the separate
[head/tail Dirichlet comparison](dirichlet.md).

The figure contains **five dragons in one Blender scene**, under a single
orthographic camera with shared lighting and ground. The centers are 2.08
scene units apart. Left to right:

1. **Raw data $f$**, before any solve or smoothing.
2. Warp CG + tuned adaptive FSAI (width 48), **converged at $k=9{,}482$ iterations**.
3. Warp CR + Jacobi, **$k=9{,}482$ iterations**.
4. Warp CR + Jacobi, **$10k=94{,}820$ iterations**.
5. Warp CR + Jacobi, **$100k=948{,}200$ iterations**.

**All four solvers start from the same data function, $u_0=f$.** A separate
[initialization experiment](initialization.md), using the original width-8 setup, compares this with zero:
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

## How much does smoothing change the data?

The first dragon shows the stored raw input `target` directly. It is next to
the converged FSAI result, and **all five dragons use one scalar range**, now
including the raw data extrema. No separate normalization or clipping hides
the input's larger range. The target, geometry, smoothing weight, and data weight match the previous
figure. All four solver fields have been regenerated using the factored
operator and budgets derived from the tuned FSAI solve.

| Measure | Raw data | Converged FSAI result |
| --- | ---: | ---: |
| Scalar range | −1.5784 to 1.8346 | −1.2218 to 1.7389 |
| Mass-weighted standard deviation | 0.71254 | 0.66299 |
| Bending energy | 44.12668 | 0.04932 |

The mass-weighted RMS change is **0.13652**, or **19.16% of the input's
standard deviation**. The maximum vertexwise absolute change is **0.71507**,
while the overall standard deviation decreases by only **6.95%**.

The broad signal is therefore substantially preserved. The large bending
energy reduction measures curvature suppression, not the amount of visible
change in the field's overall values. High-frequency or localized changes
can account for a large energy reduction. The earlier 99.89% figure should
not be interpreted as evidence of a comparably dramatic visual change.
These direct input/output measurements are stored under `raw_data_comparison`
in the figure JSON and summarized below the shared legend.

## Budgets and convergence checks

The script determines $k$ afresh from FSAI reaching Warp's recursive relative
residual tolerance $10^{-8}$. Jacobi's budgets are derived from that actual
returned count, not hard-coded. Each displayed field is the final iterate
of one uninterrupted solve from the data, with no restarts or best-checkpoint
selection. Jacobi uses zero stopping tolerance to reach its requested
budgets; actual counts are verified. Setup and kernels are warmed before
timing. Setup time is recorded separately. Solve times refer to one scalar
right-hand side and exclude initialization copies and diagnostic checks.

The FSAI label means **the solver reached its stated recursive stopping
tolerance**. The recursive relative residual is 9.44e-9; independently
recomputing the factored energy gradient in CPU float64 gives 3.35e-3.
The figure shows the latter. The initial data field has relative residual
2.15e7, and the small right-hand side makes this normalization particularly
sensitive to roundoff. The stopping tolerance is not a claim of an independently
verified 1e-8 residual; field accuracy is checked separately.

An additional unrendered run for $2k$ checks stability: its relative mass-norm
change is **1.95e-11**, with maximum absolute change 6.92e-11. This run is
excluded from displayed solve times.

An independent reference uses long-double evaluation of the factored energy
gradient with six equilibrated Cholesky correction solves. The tuned FSAI
field's relative mass-norm error is **1.83e-11**, versus **1.76e-6** for the
original assembled-operator FSAI field. The tuned Jacobi field at $100k$
has error **4.74e-11**. Target values, vertices, and triangles are bitwise
identical between old and new inputs. This checks field accuracy independently
of the drifting recursive residual.
[Reference verification](../results/smoothing-verification.json).

| Method | Iterations | Solve time (s) | Setup (s) | Relative mass-norm difference from FSAI |
| --- | ---: | ---: | ---: | ---: |
| Tuned FSAI-CG | 9,482 | 2.86 | 1.173 | 0 |
| Jacobi-CR, k | 9,482 | 1.27 | 0.002 | 0.104 |
| Jacobi-CR, 10k | 94,820 | 12.64 | 0.002 | 0.0164 |
| Jacobi-CR, 100k | 948,200 | 127.54 | 0.001 | 5.21e-11 |

## Transfer of the Dirichlet tuning

The transferred settings are width 48, `kap_tolerance=0.003`, float32 factor
storage with float64 arithmetic, and four cooperating CUDA lanes per sparse
row. Both FSAI and Jacobi use `SquaredLaplacianOperator` with all vertices free
and `mass_weight=0.0001`. The diagonal mass term is fused into the final
Laplacian product. The explicitly assembled matrix is used to build the
preconditioners; solver products use the factored energy.

Three warmed trials, alternating configuration order, measure:

| Configuration | Iterations | Median solve (s) | Median setup (s) | Median total (s) |
| --- | ---: | ---: | ---: | ---: |
| Original width-8 FSAI, assembled operator | 28,688 | 10.14 | 0.029 | 10.17 |
| Tuned width-48 FSAI, factored operator | 9,482 | 2.80 | 1.144 | 3.96 |

This is **3.62× faster for the solve and 2.57× including setup**, at the
same recursive tolerance and initial guess. It measures the combined tuning
and operator change. Setup includes the operator and preconditioner;
both configurations exclude CPU assembly and matrix upload.
[Full repeated measurements](../results/smoothing-tuning.json).
The figure labels use its own fresh run, so they differ slightly from these
medians. The old width-8 image and its metadata remain in
[the previous revision](https://github.com/alecjacobson/warp-preconditioners/tree/3ecbebcd1435382e67806a6514c46be8acd8b7af/assets).

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

The raw-data dragon and all four solved dragons share their combined scalar extrema and the same 26 intervals.
Scalars are interpolated over each original triangle **before** the constant
color ramp. No object has its own normalization, and prequantized vertex
RGB values are not interpolated.

The palette reproduces
`isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))`: an OKLab hue arc with
alternate interval lightness multiplied by 0.9. Its source functions are in
[gptoolbox at dd355405](https://github.com/alecjacobson/gptoolbox/tree/dd3554053237cd02d305b78db298d913a6196508),
under `imageprocessing/` and `mesh/isolines_stripe_map.m`. The translation's
license is retained in `third_party/gptoolbox-LICENSE-MIT.txt`.

Cycles/OptiX renders all five meshes simultaneously with shared contact
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
  --output data/visualization-tuned --tuned \
  --data-weight 0.0001 --initial-guess data --fsai-rtol 1e-8
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --data data/visualization-tuned --width 6000 --height 1120 --samples 64
python visualization/compose.py --data data/visualization-tuned
```

To repeat the original/tuned measurements and reference checks, retain the
original width-8 inputs in `data/visualization/` and run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/compare_smoothing.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/verify_smoothing.py
```

The reference check additionally requires scikit-sparse and SuiteSparse;
these are verification dependencies, not dependencies of the Warp solvers.

Omit `--tuned` and use `--audit-initial-guess` to reproduce the [separate initialization test](initialization.md).
The optional `--audit-weights` mode tests its specified positive coefficients
on the **new nonlinear target**, independently of the displayed fields.

Solving uses `cuda:0`; rendering uses an OptiX GPU and Blender 4.5.3. The
editable scene is `data/visualization-tuned/five_dragons.blend`. Intermediate
fields (including the input `target`) and renders are also in ignored
`data/visualization-tuned/`. Published PNGs use Git LFS. The numerical solver
library still depends only on Warp; Matplotlib is an optional dependency
for the separate initialization plot.
