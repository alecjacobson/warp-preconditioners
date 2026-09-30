# Tetrahedral dragon: linear elasticity under gravity

This comparison uses scalar Jacobi, the block-Jacobi implementation from
[NVIDIA/warp PR #1890](https://github.com/NVIDIA/warp/pull/1890), this
repository's block Jacobi, and adaptive FSAI. Assembly, preconditioner setup,
the verified reference, and all iterative solves run in **pure Warp on the
NVIDIA L40**. No multigrid or external sparse factorization is involved.
NumPy/SciPy handle mesh preparation and independent diagnostics, not solver
or preconditioner numerical work.

![Displacement and energy errors against iterations and wall time.](../assets/dragon-elasticity-convergence.png)

![Actual iterates at a shared setup-plus-solve budget, alongside the verified reference.](../assets/dragon-elasticity-comparison.png)

## Results

| Selected configuration | Iterations | Median setup (s) | Median solve (s) | Median total (s) | Total range, 3 trials (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar Jacobi + CG | 6,117 | 0.000397 | 0.393027 | 0.393424 | 0.393407–0.393660 |
| PR direct block Jacobi + CG | 5,026 | 0.000379 | 0.327184 | 0.327532 | 0.327411–0.327890 |
| FSAI width 8, κ = 0.003, float64 factors, four lanes + CG | 2,109 | 0.033383 | 0.180877 | **0.214849** | 0.210557–0.231925 |

Component medians need not sum to the median total. FSAI is **1.52× faster
than block Jacobi** and **1.83× faster than scalar Jacobi** including setup.
Nearby width-six/compressed-factor choices are close in measured total time;
the selected storage/width combination is not a universal recommendation.
Wider factors reduce iteration counts but lose this single-RHS race because
their construction costs dominate. CG wins the selected field-accuracy
comparison in each family; all CR trials remain in the raw results.

At the **0.214849 s** rendering budget:

| Method | Actual iterations | Median setup + actual solve (s) | Relative mass-norm displacement error | Relative energy error |
| --- | ---: | ---: | ---: | ---: |
| Scalar Jacobi-CG | 3,322 | 0.215459 | 1.048e-2 | 1.097e-2 |
| PR block-Jacobi-CG | 3,277 | 0.214845 | 3.440e-3 | 4.331e-3 |
| FSAI-CG | 2,109 | 0.217466 | 4.822e-5 | 9.082e-5 |

All snapshot times are within 1.3% of the target budget. The geometric
differences are subtle because both Jacobi fields are already reasonably
close; the saved errors and common colormap report those differences
without artificially amplifying one method's error.

## Problem and discretization

The original 721,510-triangle dragon is simplified to 60,000 triangles with
libigl's intersection-blocking decimation, then filled using
[TetGen](https://www.wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html).
The volume has **54,720 vertices, 225,770 linear tetrahedra, and 69,244 boundary
triangles**. All tetrahedra have positive orientation. The render uses this
volume mesh's boundary, rather than pretending the simplified geometry is
the original high-resolution surface.

The dragon is one metre long. Material parameters are Young's modulus
10 MPa, Poisson ratio 0.35, and density 1,000 kg/m³. Gravity is
`(0, 0, -9.81)` m/s². This is ordinary compressible, small-strain, isotropic
elasticity, without a mass regularizer or near-incompressible parameter
chosen to favor a method. The maximum reference displacement is **6.309 mm**;
the maximum element strain Frobenius norm is approximately **0.0219**.

The head and raised tail surface patches have zero displacement in all
three coordinates. In the original mesh coordinates the selections are
`x > 70 and z > 55` and `x < -70 and z > 45`, restricted to boundary vertices.
Eliminating **4,034 fixed vertices** leaves **152,058 free scalar DOFs**.
The other surfaces are traction-free. There is no artificial ground contact.

For linear tetrahedral shape gradients $g_i$, the element stiffness blocks are

$$
K^e_{ij}=V_e\left[\lambda g_i g_j^T + \mu g_j g_i^T
                  +\mu(g_i^Tg_j)I\right],
\qquad f^e_i=\rho V_e g/4.
$$

[`elasticity_problem.py`](../benchmarks/elasticity_problem.py) builds these
3×3 blocks and the consistent constant-gravity load in Warp, and sums
triplets with `warp.sparse.bsr_from_triplets`. Each free vertex carries one
`wp.vec3d` displacement. The reduced stiffness matrix is SPD. Every method
uses the same native 3×3 BSR product; a separate operator microbenchmark
found it faster here than scalar CSR products with 4, 8, or 16 cooperating
lanes. Every solve starts from zero.

## Comparison with PR #1890

The benchmark vendors the PR's block-Jacobi code unchanged, apart from
imports, from commit
[`0b58bcf2320b77e548bbbac0292f7f4415455eec`](https://github.com/NVIDIA/warp/blob/0b58bcf2320b77e548bbbac0292f7f4415455eec/warp/_src/optim/linear.py#L286-L762).
The PR is closed and was not merged at the time of this experiment. The
snapshot is [benchmark-only](../benchmarks/warp_pr1890.py), with its Apache
license retained; it does not replace the installed Warp 1.15 solver.

For this SPD matrix, all variants apply the same mathematical operator,
$\operatorname{blockdiag}(K)^{-1}$. The maximum measured relative difference
from our implementation on a random vector is **2.60e-16**. This establishes
equivalence on the tested blocks, not identical floating-point arithmetic
or identical singular-block behavior.

| Implementation | Inversion/application | Warm setup (ms) | Application (µs) |
| --- | --- | ---: | ---: |
| This repository | Scaled explicit 3×3 inverse | 4.410 | 3.40 |
| PR `auto` | Selects `direct` for 3×3 | 0.431 | 3.40 |
| PR `direct` | QR inverse, then block multiplication | 0.300 | 3.41 |
| PR `sequential` | LDLᵀ triangular solves | 1.122 | 5.67 |
| PR `tile` | Tile Cholesky solves | 0.566 | 79.24 |

Setup numbers above are single warm measurements; application numbers are
medians of five graph replays with 100 applications per replay. Full-solve
selection uses additional repeated timings. Our setup copies/validates the
sparse matrix; it is not faster than the PR's diagonal extraction here.
The PR's assertion in its code comments that tile application was fastest
on its tested block sizes does **not** carry over to this 3×3 case on the L40.
The reported block baseline is selected from the measured implementations,
including the PR versions.

## Tuning, timing, and accuracy

The sweep retains all **36 tested configurations**, including slower ones,
in [`elasticity.json`](elasticity.json). This is a measured search, not a
claim of a global optimum. It tests CG and CR for every family, all four
block implementations, and FSAI widths 4, 6, 8, 12, 16, 32, and 48. Nearby
FSAI trials vary the adaptive stopping parameter (0.001, 0.003, 0.01),
application lanes (1, 4, 8), and factor storage (float32 or float64).
Matrix and solve arithmetic remain float64.

Convergence requires **both**

$$
\frac{\|u-u_*\|_{M_l}}{\|u_*\|_{M_l}}\le10^{-4},\qquad
\frac{\|u-u_*\|_K}{\|u_*\|_K}\le10^{-4},
$$

where $M_l$ contains lumped nodal volumes, repeated for the three coordinates.
The constant density cancels from the mass-norm ratio. The plotted quantities
are forward displacement and energy errors, not recursive residuals. True
residuals are also retained in the JSON; they need not rank the methods in
the same order as field errors.

The initial sweep locates accuracy crossings with sampled, uninterrupted
trajectories. Up to six candidates within 12% of the best preliminary time
in each family are refined in iteration count and measured three times.
The fastest median selects that family's settings. Crossings are resolved
within the sampled bracket, rather than claiming an exact first iteration.
The selected configurations receive another three warm measurements.

Times include **preconditioner construction plus the solve**, including
solver graph-capture and host-launch costs. Common assembly, input uploads,
first-use JIT, independent CPU diagnostics, mesh generation, and rendering
are excluded. Each line-plot time sample is a separate timed native solve
from zero to that iteration; diagnostic callbacks do not contaminate those
times. The iteration curves sample one live recurrence without restarting.
The clock measurements are retained as measured, without monotonic filtering.

The frozen rendering uses the fastest finalist's median setup-plus-solve
time as the common budget. Each other method receives a fixed iteration
count calibrated using actual uninstrumented solves. The displayed time is
that method's median setup plus its actual snapshot solve time; small timing
differences are reported, not hidden. FSAI pays its setup cost before it can
iterate. The winning snapshot uses the accuracy-qualified iteration count.

All four rendered dragons share the same 26-band **von Mises stress**
colormap, white background, lighting, and camera. Geometry uses the **actual
physical displacement, with no exaggeration**. Colors use one linear scale
in kPa across all four fields. Dark gray marks
fully clamped surface triangles. The first dragon is the tightly converged
reference, followed by the three actual saved iterates. Blender vertex
positions and scalar attributes are read back and checked, and saved fields
are SHA-256 checked before rendering.

Linear tetrahedra have constant element strain and stress. For visualization,
[`elasticity_stress.py`](../benchmarks/elasticity_stress.py) volume-averages
the element stress **tensors** at each vertex, then computes
$\sigma_{\rm VM}=\sqrt{\tfrac32\operatorname{dev}(\sigma):\operatorname{dev}(\sigma)}$.
Both operations run in Warp using each method's actual saved displacement.
This nodal recovery smooths element discontinuities; it is not the raw peak
element stress. The reference maxima are 100.398 kPa recovered nodal stress
and 182.391 kPa element stress. Affine-strain and hydrostatic-stress tests
check the invariant independently on CPU and CUDA. The recovered scalar is
interpolated over surface triangles before the crisp 26-band color lookup.

## Verification

An independent CPU strain/stress contraction agrees with the GPU-assembled
matrix action to **9.04e-16 relative error**. The assembled matrix is symmetric
to the reported precision. Tests additionally check the six rigid modes of
a free tetrahedron, analytic affine-strain energy, total gravitational force,
Dirichlet elimination, and PR/local block-inverse agreement on CPU and CUDA.

The reference is a pure-Warp, tightly converged FSAI-CG solution, corrected
using a separately built block-Jacobi-CG solve. The correction changes its
mass-norm displacement by **4.38e-13 relative**. Independent element-force
evaluation gives **2.03e-11 relative residual**. No external sparse
factorization supplies the reference or any compared iterate.

## Reproduction

```bash
python -m pip install -e '.[elasticity,test]'
# Use the original benchmark dragon PLY, or an existing fields.npz:
OPENBLAS_NUM_THREADS=1 python benchmarks/tetmesh_dragon.py \
  --surface /path/to/xyzrgb_dragon-720K.ply
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage sweep
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage sweep \
  --configs results/elasticity-nearby-configs.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage final
python benchmarks/plot_elasticity.py
python benchmarks/elasticity_stress.py
blender -b --factory-startup --python visualization/render_elasticity.py
python visualization/compose_elasticity.py
pytest -q tests/test_elasticity.py tests/test_residual_history.py
```

The sweep resumes completed matching candidates. For a fresh measurement
using the commands above, move the existing `results/elasticity.json` aside
first. Meshing and volume solution files live under
`data/elasticity/`. The committed JSON regenerates the line plot directly;
the PNG assets are tracked with Git LFS. Blender 4.5/OptiX is used for the
rendering commands. Input hashes, mesh parameters, configurations, raw
timings, and accuracy samples are recorded with the results.
