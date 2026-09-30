# Head-to-tail biharmonic interpolation

![Prescribed regions, tuned FSAI-CG at convergence, and Jacobi-CR at k, 10k, and 100k.](../assets/dragon-dirichlet-comparison.png)

This regenerated comparison uses the [tuned FSAI recipe](../results/dirichlet-tuning/README.md):
maximum width 48, `kap_tolerance=0.003`, float32 factor storage with float64
accumulation, and four cooperating CUDA lanes per row. **Both solvers use
the same factored squared-Laplacian operator.** The earlier width-eight,
explicitly assembled comparison is [archived here](dirichlet-original.md).

The problem and mesh are unchanged:

$$
E(u)=\frac12 u^TLM^{-1}Lu,
\qquad u_i=-1\text{ on the tail},\quad u_i=1\text{ on the head}.
$$

There is **no data term or diagonal regularizer**. All free values start at
zero. The first dragon shows the fixed regions; the next four show the
actual final iterates of independent, uninterrupted solves. All solution
dragons use the same 26-band striped colormap and scalar range, without
clipping overshoot. Geometry, camera, lights, and spacing match the original
scene.

## Regions and reduced system

The leftmost dragon identifies the constrained vertices: **blue is the tail
at −1, red is the head at +1, and gray is free**. This view uses categorical
colors, not the solution colorbar. The remaining four dragons share one
26-band striped scalar colormap, including all their extrema.

Selections use the original mesh coordinates:

| Patch | Selection | Vertices | Value |
| --- | --- | ---: | ---: |
| Raised tail | Largest connected component of $x<-70$ and $z>45$ | 12,684 | −1 |
| Upper head | Largest connected component of $x>70$ and $z>55$ | 27,578 | +1 |

Both candidate selections already form single connected patches. The
original mesh has 360,757 vertices and 721,510 triangles. After fixing
40,262 vertices, the reduced system has **320,495 unknowns and 6,525,939
stored nonzeros**.

Let $b$ denote fixed indices and $f$ free indices. We form the **full**
squared operator before eliminating fixed values:

$$
Q_{ff}u_f=-Q_{fb}u_b.
$$

In particular, the energy retains all rows of $L$, including those at fixed
vertices. Squaring an already restricted Laplacian generally gives a
different problem. The mesh is connected, so the nullspace of $L$, and
therefore of $Q$, consists of constants. Fixing these nonempty patches
removes that nullspace: $Q_{ff}$ is symmetric positive definite. An
independent Cholesky factorization with AMD ordering checks this without
any diagonal shift.

## Updated solver comparison

FSAI-CG reaches recursive relative tolerance $10^{-12}$ at
**$k=33{,}374$ iterations**. Jacobi-CR runs for exactly $k$, $10k=333{,}740$,
and $100k=3{,}337{,}400$ iterations. These budgets are smaller than in the
original figure because $k$ now comes from the faster FSAI method.

| Method | Iterations | Solve (s) | Setup (s) | Relative mass-norm difference from FSAI | Energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuned FSAI-CG | 33,374 | 8.56 | 1.073 | 0 | 0.0002063421512 |
| Jacobi-CR, $k$ | 33,374 | 4.01 | 0.001 | 0.8374 | 0.04425449381 |
| Jacobi-CR, $10k$ | 333,740 | 39.95 | 0.001 | 0.8403 | 0.006240283998 |
| Jacobi-CR, $100k$ | 3,337,400 | 403.77 | 0.002 | 1.284e-09 | 0.0002063421512 |

Timings are single runs on the L40 after warmup. Figure labels show solve
time; the FSAI label also reports setup. Setup includes construction of the
factored operator and preconditioner. Host assembly, uploads, RHS construction,
verification, and rendering are excluded. For repeated timing estimates,
see the tuning report.

The factored operator applies

$$
Ax=R^TL^TM^{-1}LRx,
\qquad b=-R^TL^TM^{-1}Lu_b,
$$

where $R$ inserts free values into a full vector and $u_b$ contains the
prescribed values with zeros on free vertices. All rows of $L$ participate.
The explicitly assembled $Q_{ff}$ is used to construct the preconditioners;
the Krylov products use the factored form. No materialized squared matrix
is used in those products. The problem remains float64; only the FSAI
factor storage is compressed.

## Numerical and rendering checks

The FSAI field differs from the independently refined energy reference by
**1.29e-09** in relative mass norm. A separate $2k$ run changes it
by **1.29e-09**. The Jacobi-$100k$ field differs
from the refined reference by **3.59e-11**. All fields satisfy the
prescribed values exactly. FSAI's range is **-1.151402890 to 1.020900690**;
this genuine overshoot is retained in the shared color scale.

Field errors are relative mass norms of full fields. The figure's residuals
are independently recomputed on the CPU from the full energy gradient:

$$
\frac{\|(L^TM^{-1}Lu)_f\|_2}{\|b\|_2}.
$$

These differ from the internal recursive residuals. Neither a small residual
nor a lower bending energy guarantees that an unconverged iterate is close
to the final field in mass norm. In particular, Jacobi's field error need
not decrease monotonically with iteration budget.

The refined reference evaluates stationarity in long double, using
AMD Cholesky correction solves. Agreement with the old explicitly assembled
Cholesky solution is also recorded, but is not used as the accuracy gate:
that matrix has a documented cancellation-error limit. The verification
checks exact constraints, the full-energy restriction, energies, independent
residuals, and agreement of FSAI with the refined reference.

Every rendered field is SHA-256 checked. Blender's actual scalar attribute
is read back and compared exactly with the float32 conversion of the saved
float64 field. Colors are assigned after scalar interpolation on each
triangle, yielding crisp isointervals rather than interpolated vertex colors.

## Residual history

![Measured log-log residual histories.](../assets/dragon-dirichlet-residuals.png)

The plot uses the same tuned FSAI-CG and Jacobi-CR configurations, full
factored operator, prescribed values, and zero free initial guess as the
figure. Its vertical quantity is a **relative Euclidean residual norm**,
not squared residual loss, bending energy, or forward field error:

$$
\rho_k=\frac{\|(L^T M^{-1} L u_k)_f\|_2}{\|b\|_2}.
$$

Solid curves recompute this quantity independently in CPU float64 at each
sample. Dashed curves show the residual maintained by Warp's iterative
recurrence. The horizontal tolerance applies to the recursive FSAI residual;
it is not a guarantee that the independently recomputed residual reaches
that value. Residual drift and finite precision explain the visible gap
near convergence. Field accuracy is checked separately as described above.

Each curve comes from a single live Krylov recurrence. The sampling driver
pauses the native GPU iteration loop at logarithmically spaced checkpoints
and at the figure's $k$, $10k$, and $100k$. It preserves all search
directions and reduction buffers; it never calls a fresh solve at a
checkpoint. Iteration zero is saved with relative residual 1, but omitted
from the log axis. Lines connect actual samples, without smoothing or
cumulative-minimum filtering; unsampled intermediate oscillations are not
shown. FSAI stops at its original tolerance, while Jacobi runs through
$100k$ with tolerance zero, matching the figure's budget.

[`results/dirichlet-residuals.json`](../results/dirichlet-residuals.json)
contains the sample values, configuration, input hash, and endpoint check
against the saved figure fields. Instrumented wall times include graph
captures, downloads, and CPU verification; they are **not solver benchmark
timings**. The diagnostic adapter uses Warp 1.15's private loop driver in
a temporary, process-local context and restores it afterward; the library
and its public solver API are unchanged. CPU/CUDA tests compare sampled
and native endpoints exactly for both CG and CR.

After preparing `data/dirichlet-tuned/` below, collect and plot with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/trace_dirichlet.py
python benchmarks/plot_dirichlet_residuals.py
```

Only the plot command is needed to regenerate the image from the committed
JSON. The PNG is tracked with Git LFS.

## Reproduction

Generate the benchmark dumps in `/tmp/dump` as in the main README, then:

```bash
OPENBLAS_NUM_THREADS=1 python visualization/solve_dirichlet.py \
  --dir /tmp/dump --mesh /path/to/xyzrgb_dragon-720K.ply \
  --output data/dirichlet-tuned --tuned
# Independent reference requires SuiteSparse and scikit-sparse.
OPENBLAS_NUM_THREADS=1 python benchmarks/refine_dirichlet.py \
  --data data/dirichlet-tuned --output data/dirichlet-tuned
OPENBLAS_NUM_THREADS=1 python visualization/verify_dirichlet.py \
  --data data/dirichlet-tuned \
  --refined-reference data/dirichlet-tuned/refined_reference.npy
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --data data/dirichlet-tuned --width 6000 --height 1120 --samples 64
python visualization/compose_dirichlet.py --data data/dirichlet-tuned
```

The saved fields and editable Blender scene are in ignored
`data/dirichlet-tuned/`. Published [figure metadata](../assets/dragon-dirichlet-comparison.json)
contains timings, diagnostics, hashes, the Cholesky/refined-reference audit,
and render settings. The PNG uses Git LFS. The
[nonlinear data-smoothing comparison](README.md) is a separate experiment.
