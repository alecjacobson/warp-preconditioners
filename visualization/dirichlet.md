# Head-to-tail biharmonic interpolation

![Prescribed regions, FSAI-CG, and Jacobi-CG at k, 10k, and 100k with refreshed timings](../assets/dragon-dirichlet-comparison.png)

The dragon rendering uses **one FSAI method, CG**, and **Jacobi-CG**.
It was rerun after the FSAI setup/packing changes. FSAI-CG and FSAI-CR were
compared at the same independently checked field-accuracy target, then the
faster qualifying method was retained in the rendering and residual plot.
The residual plot also includes **Jacobi-CR** alongside Jacobi-CG.
The former four-method residual plot and Jacobi-CR rendering remain in
[the repository history](https://github.com/alecjacobson/warp-preconditioners/tree/f01f138).
The [original width-eight experiment](dirichlet-original.md) is also archived.

## Problem and constraints

We minimize

$$
E(u)=\tfrac12 u^T L M^{-1}L u,
\qquad u_i=-1\text{ on the tail},\quad u_i=+1\text{ on the head}.
$$

There is no data term or diagonal regularizer. Free variables start at zero,
and constrained values are imposed exactly by elimination. The first dragon
marks the prescribed regions: blue tail, red head, gray free vertices.
The other four dragons share one scalar range and 26 crisp OKLab color bands;
overshoot is retained. This is a scalar field on the original geometry.

| Patch | Selection in original coordinates | Vertices | Value |
| --- | --- | ---: | ---: |
| Tail | Largest connected component of x < −70 and z > 45 | 12,684 | −1 |
| Head | Largest connected component of x > 70 and z > 55 | 27,578 | +1 |

There are 360,757 vertices and 721,510 triangles, with **320,495 free
unknowns**. The connected mesh and nonempty fixed regions remove the constant
nullspace. The full squared operator is formed before elimination, retaining
all rows of the Laplacian, including rows at constrained vertices. The
assembled reduced matrix has 6,525,939 stored entries and supplies both
preconditioners. Every Krylov matrix product uses the same float64 factored
operator `Rᵀ Lᵀ M⁻¹ L R`, with four cooperating CUDA lanes per row.

## Choosing the FSAI method fairly

FSAI settings remain width 48, κ=.003, float32 factor storage, float64
accumulation, and four lanes per row. We compare CG and CR at relative
lumped-mass field error **≤1e-8**, measured against the saved independently
refined energy reference:

$$
\frac{\sqrt{(u-u_*)^T M (u-u_*)}}{\sqrt{u_*^T M u_*}}\le10^{-8}.
$$

This replaces the older choice of k from a recursive residual tolerance.
Equal recursive tolerances had given substantially different field errors
for CG and CR. Each solver's trajectory is sampled every 500 iterations,
then every 25 within the first qualifying interval. The chosen endpoint is
checked again with an uninstrumented solve. These are sampled qualifying
stops, not claims of exact first crossings or a new search over FSAI widths.

| FSAI solver | Selected iterations | Field error | Median setup + solve |
| --- | ---: | ---: | ---: |
| CG | 33,125 | 9.32e-9 | **10.1437 s** |
| CR | 33,425 | 6.60e-9 | 10.6070 s |

Three interleaved warm repetitions per method select **FSAI-CG**, about 4.6%
faster in this run. The figure uses the median-total trial: **9.0374 s solve + 1.1063 s setup**.
Keeping those components paired makes their sum equal the displayed total.
The rendered FSAI field is the last validated repeat; all repetitions meet
the target.

[Selection samples, all timing repetitions and provenance](../results/dirichlet-refresh.json).

## Fresh rendering measurements

| Method | Iterations | Setup (s) | Solve (s) | Total (s) | Relative field error | True relative residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSAI-CG | 33,125 | 1.106 | 9.04 | 10.14 | 9.320e-09 | 1.165e-09 |
| Jacobi-CG, 1k | 33,125 | 0.003 | 3.82 | 3.82 | 8.235e-01 | 2.393e-05 |
| Jacobi-CG, 10k | 331,250 | 0.002 | 38.07 | 38.07 | 8.965e-01 | 2.012e-06 |
| Jacobi-CG, 100k | 3,312,500 | 0.003 | 383.05 | 383.05 | 8.953e-12 | 3.315e-07 |

All numerical solves and preconditioners run in Warp on NVIDIA L40. FSAI
labels use three-repeat medians; each Jacobi image uses one warm,
uninterrupted run at its stated iteration budget. Total time includes
factored-operator and preconditioner setup. Uploads, host assembly, JIT,
verification and rendering are excluded. These are **iteration-budget
comparisons**, not snapshots at equal wall time.

The displayed field errors use the same independently refined reference for
both methods. True residuals independently evaluate the full energy gradient
`(Lᵀ((L u)/mass))[free]`, divided by the norm of the eliminated right-hand
side. Neither a small recursive residual nor a small true residual alone is
used as proof of field accuracy on this ill-conditioned problem.

## Residual history

![Fresh trajectories for selected FSAI-CG, Jacobi-CG and Jacobi-CR](../assets/dragon-dirichlet-residuals.png)

The plot contains **FSAI-CG, Jacobi-CG and Jacobi-CR**. Solid curves independently
recompute the relative residual; dashed curves show Warp's internal
recursive residual. All start with true relative residual 1 at iteration
zero, which is omitted from the logarithmic axis. FSAI runs to its selected
field-accuracy stop k; both Jacobi methods run to the same 100k budget.
The FSAI stopping target is a field error, not a horizontal residual threshold.

Samples retain each solver's live Krylov state, without restarts or smoothing.
The FSAI-CG and Jacobi-CG sampler endpoints are checked against their
separately timed rendered fields. Jacobi-CR appears only in the plot and has
no corresponding rendered field. Sampling wall times include transfers, graph
captures and CPU checks and are excluded from benchmark solve timings. The temporary sampler
uses Warp 1.15's private loop driver; CPU/CUDA regression tests check its
endpoints against native CG and CR.

The sampled FSAI-CG and Jacobi-CG final fields match the separately timed
figure fields **exactly**
(zero relative L2 difference). The true residual and reference-field error
also agree at every rendered k, 10k and 100k checkpoint. FSAI ends at true
relative residual **1.16e-9**, versus recursive **1.47e-10**. Jacobi-CG ends at
true residual **3.31e-7**, versus recursive **3.18e-30**, while its field error
is **8.95e-12**. The large residual gap is reported directly.

The restored Jacobi-CR trace uses the same current operator and **3,312,500**
iteration budget. It ends at true residual **8.44e-08**,
recursive residual **8.84e-17**, and relative
field error **3.59e-11**. Its smaller final
recomputed residual than Jacobi-CG does not imply smaller field error.
The existing FSAI-CG and Jacobi-CG records are unchanged by this addition.

[Raw trajectories and endpoint checks](../results/dirichlet-residuals.json).
The independent reference is the unchanged long-double energy-gradient
refinement, with diagonally equilibrated float64 AMD Cholesky corrections,
from the earlier audit. Its array hash is verified before this experiment;
no external solver produces any rendered iterative field.

## Reproduction

Prepare the original `data/dirichlet-tuned/` fields and independently refined
reference using the [earlier commands](https://github.com/alecjacobson/warp-preconditioners/blob/f01f138/visualization/dirichlet.md#reproduction).
The benchmark Laplacian dump remains in `/tmp/dump/k4_Q.mtx`. Then:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/refresh_dirichlet.py --stage select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/refresh_dirichlet.py --stage fields
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/trace_dirichlet.py
python benchmarks/plot_dirichlet_residuals.py
# Blender 4.5 with OptiX; run after solver timing measurements finish:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 blender -b --factory-startup \
  --python visualization/render_dragon.py -- \
  --data data/dirichlet-current --width 6000 --height 1120 --samples 64
python visualization/compose_dirichlet.py
```

Fresh fields and intermediate renders live in ignored `data/dirichlet-current/`.
[Figure metadata](../assets/dragon-dirichlet-comparison.json) includes current
timings, selected FSAI measurements, independent field errors and render hashes.
PNGs are tracked with Git LFS. The [data-smoothing comparison](README.md)
and [SimJEB elasticity experiments](../results/elasticity.md) are separate problems.
