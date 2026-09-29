# Tuning FSAI for the unregularized dragon

The original width-eight parameters are **not the best tested choice** for
this problem. Wider adaptive supports, cooperative sparse products, and a
factored squared-Laplacian application substantially reduce the total time.
There is a tradeoff between cheapest iteration and fastest complete solve.

The [regenerated dragon figure](../../visualization/dirichlet.md) uses the
width-48 recipe, with Jacobi on the same factored operator at its new
$k$, $10k$, and $100k$ budgets.

![Controlled mass-term experiment, repeated solve timings, per-iteration costs, and parameter sweep.](../../assets/dragon-dirichlet-tuning.png)

## Why 0.1 seconds became 43 seconds

Removing the data term changes the low-frequency conditioning dramatically.
This is more consequential than changing the right-hand side. To isolate it,
we kept the mesh, head/tail constraints, right-hand side, zero initial guess,
width-eight FSAI-CG, and relative stopping tolerance $10^{-12}$ fixed, and solved

$$
(Q_{ff}+\alpha M_{ff})u_f=-Q_{fb}u_b,
\qquad Q=LM^{-1}L.
$$

| Mass weight $\alpha$ | Iterations | Solve time |
| --- | ---: | ---: |
| 1 | 394 | 0.125 s |
| $10^{-4}$ | 22,799 | 7.109 s |
| 0 | 138,856 | 43.501 s |

These are controlled diagnostic problems; adding this term would change the
requested interpolation. **The optimized interpolation below still has no
data term or diagonal shift.** In mass-normalized coordinates, adding
$\alpha M_{ff}$ shifts every eigenvalue by $\alpha$, lifting the small
eigenvalues that local FSAI struggles with. The mesh coordinates and units
are unchanged. [Raw controlled measurements](regularization.json).

The earlier height benchmark also used a different stopping schedule and
right-hand side. Rechecking that original equation gives 0.177 s for 500
iterations and 0.237 s for the tighter $10^{-12}$ stopping rule. Neither
the schedule nor tolerance alone explains the jump to 43 s. The original
README's 0.708 s measurement covers **three** right-hand sides and checks;
it is not a single uninterrupted solve.

## Repeated measurements

All repeated trials use the same NVIDIA L40, Warp 1.15.0, float64 equation
and vectors, 320,495 free unknowns, natural ordering, and zero free initial
guess. Each recipe is warmed, then rebuilt and solved in one uninterrupted
call. Three rounds interleave the recipes. The table is generated from
[all trials](repeated.json); ranges and medians are in [summary.json](summary.json).
GPU clocks were not locked; the table includes the observed timing ranges.
Input hashes and software versions are recorded in [environment.json](environment.json).

| Recipe | Iterations | Setup (s) | Solve (s) | Total (s), range | µs / iteration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FSAI-CG, width 8 | 138,856 | 0.027 | 43.334 | 43.361 (43.082–43.828) | 312.1 |
| Tuned FSAI-CG, width 48 | 33,374 | 1.052 | 9.142 | 10.193 (9.901–10.464) | 273.9 |
| Tuned FSAI-CG, width 64 | 30,107 | 1.940 | 8.409 | 10.415 (10.163–10.537) | 279.3 |
| Cheaper-step FSAI-CG | 89,859 | 0.197 | 15.147 | 15.344 (15.105–15.403) | 168.6 |
| Original Jacobi-CR, budget only | 5,000 | 0.000 | 0.858 | 0.859 (0.840–0.861) | 171.6 |
| Factored Jacobi-CR, budget only | 5,000 | 0.002 | 0.601 | 0.602 (0.600–0.603) | 120.1 |

The tuned recipes apply the factored operator with four lanes per row,
construct FSAI in float64, store its completed factors in float32, and
accumulate factor products in float64 with four lanes per row. Width 48
has lower setup cost; width 64 has lower solve cost and is attractive when
reusing the factors. The cheaper-step recipe uses width 64 and
`kap_tolerance=0.03`; the other tuned recipes use `0.003`.

"Total" includes operator construction, FSAI construction, and the solve.
It excludes host matrix assembly, input transfer, factored RHS construction,
warmup/JIT compilation, field downloads, and independent verification.
The assembled reduced matrix is **still needed to construct FSAI**. This
is a factored Krylov application, not a fully assembly-free pipeline.

Jacobi-CR costs are measured over 5,000 uninterrupted iterations from zero,
using the original and optimized operator respectively. Those fields are
**not converged**. FSAI retains two additional sparse products per iteration;
it does not become as cheap as Jacobi with the same optimized operator.
We did not rerun the original 13.9-million-step Jacobi convergence experiment.

## What helped, and what did not

At the original application settings, increasing width from 8 to 16, 32,
and 64 reduced total setup-plus-solve time from 43.08 s to 32.57 s, 26.68 s,
and 22.95 s. Width 64 used 33,991 iterations, but each cost 603 microseconds.
Simply minimizing iteration count is the wrong objective.

Disabling adaptive early stopping was worse: width 64 with
`kap_tolerance=0` needed only 29,595 iterations, but took **49.78 s including
setup**, with 16.48 million factor entries. The default $10^{-3}$ and
slightly looser $3\times10^{-3}$ criteria avoid expensive, marginally useful
entries. The pivot safeguard did not activate in any measured candidate;
there is no evidence to change `pivot_floor` here.

Morton and reverse Cuthill–McKee ordering, ELL factor storage, and FSAI-CR
were tested. None beat the final recipes. They remain in the raw results,
including the slower configurations. Cooperative CSR application alone
reduced width-eight time per iteration from about 310 to 242 microseconds.
For users who only have the assembled matrix, width 64 with eight lanes
for both matrix and factor products took **15.07 s including FSAI setup**
in the exploratory sweep, without changing the stored equation.

Isolated warmed products measured approximately 105 microseconds for the
original matrix, 53 for both FSAI factors, and 4 for Jacobi. Whole-iteration
measurements are more informative: the matrix and factors together exceed
the L40's 96 MiB L2 cache, whereas repeatedly applying an isolated factor
can reuse that cache. Reducing factor storage and assigning several lanes
to each short sparse row help memory access. The factored operator also
uses the smaller Laplacian stencil instead of the squared stencil.

The float32 option rounds **only the preconditioner**, after constructing
it in float64. Its transpose is built from the same rounded factor, so
$P=G^TG$ retains its Gram form and positive definiteness in exact arithmetic.
Conversion checks reject nonfinite entries or a lost positive diagonal.
The equation and all Krylov vectors remain float64. This option needs an
accuracy check on other problems; it is not a universal default.

## Accuracy: the old Cholesky reference had a limit

The optimized operator applies

$$
A x=R^TL^TM^{-1}LRx,
\qquad b=-R^TL^TM^{-1}Lu_b,
$$

where $R$ inserts free values into a full vector and $u_b$ is zero on free
vertices. Here $L$ is symmetric. All rows of $L$ participate, including
rows at fixed vertices. Squaring $L_{ff}$ would solve a different problem.

Explicitly multiplying the matrices in float64 introduces cancellation
error. The old Cholesky comparison independently solved that rounded
matrix, so it could not detect this source of error. We built a stronger
reference by promoting the input $L,M$ to long double before assembly,
using equilibrated AMD Cholesky for corrections, and evaluating the
stationarity residual as $-L^T(M^{-1}Lu)$ in long double. The last relative
mass-norm correction was $3.94\times10^{-17}$; long double has epsilon
$1.08\times10^{-19}$ on this machine. This is an empirical numerical
reference for the supplied discrete $L,M$, not an exact or continuum solution.

Against it, the previous Cholesky field differs by $1.03\times10^{-4}$ and
the previously displayed FSAI field by $1.38\times10^{-4}$ in relative mass
norm. The factored tuned solutions differ by roughly $10^{-9}$. This also
explains why a new, more accurate solution narrowly fails the old
$10^{-4}$ agreement gate with the old Cholesky field. We preserve both
checks in the raw data instead of silently redefining that old gate.
[Refinement audit](refinement.json).

All FSAI runs stop at recursive relative residual $10^{-12}$, but this is
not a forward-error guarantee. The runner also recomputes the applied
operator's residual and measures error to the refined reference. In the
JSON, `relative_residual` and `backward_error` use the explicitly stored
matrix (and the configured RHS); `operator_relative_residual` uses the
actual Krylov operator. `mass_relative_error` uses the old Cholesky field;
`physical_mass_relative_error` uses the refined energy reference. The mass
norm denominator includes the full field, including fixed vertices.

## Recommended usage

```python
import warp as wp
from warp.optim.linear import cg
from warp_preconditioners import FSAI, SquaredLaplacianOperator

# L: full symmetric scalar Warp BSR Laplacian, float64.
# mass: full positive lumped masses, float64 Warp array.
# free: distinct free indices, int32 Warp array.
# prescribed: full float64 vector (free entries are ignored).
# Qff: explicitly assembled (L M^-1 L)[free, free], for FSAI construction.
A = SquaredLaplacianOperator(L, mass, free, row_lanes=4)
b = A.rhs(prescribed)
P = FSAI(Qff, max_row_size=48, kap_tolerance=0.003,
         apply_lanes=4, factor_dtype=wp.float32)
x = wp.zeros_like(b)
cg(A, b, x, M=P, tol=1e-12, maxiter=200000, check_every=0)
```

For repeated right-hand sides, try width 64 and reuse both operators. For
lower per-iteration cost, try `kap_tolerance=0.01` or `0.03`, and compare
**total time at verified accuracy**. The library defaults remain unchanged:
these are problem-specific measured settings. Local FSAI still takes tens
of thousands of iterations here; no mesh-independent convergence or global
parameter optimum has been established. A global coarse correction would
be a separate algorithmic investigation.

An optional [ARPACK spectral diagnostic](../../benchmarks/dirichlet_spectrum.py)
uses Cholesky shift-invert for the smallest generalized eigenvalue.
Its [recorded endpoints](spectrum.json) are exploratory: the smallest
eigenvector has componentwise backward error about 0.053. We do not use
that estimate as a certified condition number or as evidence of accuracy;
the controlled mass-term experiment establishes the timing effect directly.

## Reproduction

Generate the original matrix dumps as in the main README. These benchmarks
expect `/tmp/dump/` and `data/dirichlet/`. On a fresh checkout, prepare the
same patches without running the long Jacobi comparison:

```bash
python -m pip install -e '.[test,visualization]'
# Independent references additionally require SuiteSparse and scikit-sparse.
mkdir -p data/tuning
OPENBLAS_NUM_THREADS=1 python visualization/solve_dirichlet.py \
  --mesh /path/to/xyzrgb_dragon-720K.ply --prepare-only
OPENBLAS_NUM_THREADS=1 python visualization/verify_dirichlet.py
OPENBLAS_NUM_THREADS=1 python benchmarks/refine_dirichlet.py
OPENBLAS_NUM_THREADS=1 python benchmarks/tune_dirichlet.py \
  --configs results/dirichlet-tuning/repeated-config.json \
  --output data/tuning/repeated.json
OPENBLAS_NUM_THREADS=1 python benchmarks/regularization_audit.py
OPENBLAS_NUM_THREADS=1 python benchmarks/profile_dirichlet.py \
  --output data/tuning/profile8.json
pytest -q
```

`--prepare-only` writes its output directory: use it on a fresh dataset or
choose `--output` when preserving an earlier full visualization run.
Derived tuning caches are invalidated on changed input arrays or matrix
file metadata. Experiment JSON records each configuration; the runner
accepts a JSON list of those configurations. The other committed config
files reproduce the exploratory sweeps. Timings vary with hardware and
kernel/compiler versions. Plot the committed measurements with
`python benchmarks/plot_dirichlet_tuning.py`.

Tests cover independent matrix/energy references, CPU/CUDA, both scalar
precisions, compressed factors, block-vector integration, alpha/beta and
aliasing behavior, CUDA graph replay, and invalid inputs.
