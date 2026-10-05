# Stronger tests for blocking

The controlled problem gives a clear win for block Jacobi. On the rotated tensor
system at κ=10⁶, it takes **6.08 ms** total,
versus **788.84 ms** for scalar Jacobi:
about **130× faster**.
The reinforced bracket tests below distinguish that benefit from the additional
cost of adaptive block FSAI. These are experimental implementations and
measurements; the supported preconditioner APIs are unchanged.

## Controlled component coupling

We assemble `A = K ⊗ C` on the GPU, where `K` is the seven-point Dirichlet
Laplacian on 32³ interior nodes (98,304 displacement unknowns), and
`C = R diag(1,1,κ) Rᵀ`. κ ranges from 1 to 10⁶. The exact solution contains
both smooth and broadband components; it and the RHS rotate consistently with
`R`. This compares equivalent problems under a component-coordinate change.
The zero initial guess is shared.

With block Jacobi, `M⁻¹ A = (diag(K)⁻¹ K) ⊗ I`. Its exact spectrum is independent
of `C`. Independent small-matrix eigenvalue tests verify this identity on CPU
and CUDA, and each large case checks the block inverse action. At every tested
contrast and orientation, block-Jacobi CG reaches the target at the sampled
**100-iteration** checkpoint. Scalar Jacobi is also effective when `C` is
aligned and diagonal; rotation exposes its missing component coupling.

Median **milliseconds**, including setup, five warmed trials per candidate:

| κ (rotated) | Scalar Jacobi | Block Jacobi | Scalar FSAI | Block FSAI |
|---:|---:|---:|---:|---:|
| 1 | 5.71 | 5.62 | 6.56 | 8.77 |
| 100 | 28.91 | 6.03 | 10.54 | 10.09 |
| 10000 | 255.36 | 5.95 | 11.19 | 10.13 |
| 1e+06 | 788.84 | 6.08 | 10.42 | 11.79 |

![Controlled coupling timing and actual solution error](../assets/block-coupling.png)

Both relative solution error and energy error must be ≤10⁻⁶. CG and CR are
both tested. The plot includes the common zero start using `iterations + 1`
on its logarithmic axis. The curves show actual forward error, not the
solver's recursive residual. Forward error in this norm need not decrease
monotonically. Shaded timing bands are the minimum and maximum trials.

At κ=10⁶ the public direct block inverse needs the explicitly reported
`pivot_tolerance=1e-15`; its default determinant threshold conservatively
rejects the aligned SPD block. After rotation, the cofactor inverse has
relative action error **5.79e-7**. A benchmark-only SPD Cholesky inverse reduces
this to **2.73e-11**. Both reach the required final solve accuracy. We include
both implementations in tuning; the small timing differences do not justify
replacing the general public inverse with an SPD-only implementation.

An [untimed rotation audit](block-rotation-audit.json) additionally compares
adaptive selection with fixed-block-pattern refits on an 8³ tensor grid,
using two rotations, contrasts 1/100/10⁶, and two-/four-block factors.
**No selected block rows change** under either rotation in this control.
After transforming the preconditioner action back, the largest relative
covariance error is **1.67e-10 with float64 factors** and **9.21e-8 with
float32 factors**. Fixed-pattern refits agree at the same scale. Thus this
separable control gives no evidence of a block-pattern rotation bug; it
does not establish rotation invariance for arbitrary nonseparable matrices.

## Directionally reinforced SimJEB bracket

The fTetWild mesh, bolt-hole constraints, and challenge load are unchanged:
351,711 tets and 204,231 free scalar unknowns. Poisson ratio remains 0.342.
The new energy density is

`W = W_iso + τ/2 (qᵀ ε q)²`, with `τ = (contrast − 1)(λ + 2μ)`.

Thus contrast is the ratio of axial **uniaxial-strain stiffness**, not a
Young's-modulus ratio under stress-free lateral contraction. This positive
term preserves SPD after constraint elimination. Its element contribution is
`τ V (q·g_i)(q·g_j) q qᵀ`. Assembly runs entirely in Warp. Tests independently
check affine energy, element forces, rigid modes and the zero-reinforcement
limit. Directions are global +Z (`aligned`), a fixed oblique rotation
(`rotated`), or a smooth spatially twisting field (`varying`). Unlike the
coordinate-change control above, these change the physical material relative
to the fixed mesh and load.

Each material gets a new float64 Warp-CG reference, corrected with a different
preconditioner and checked by independent element strain/stress contractions.
Acceptance requires **both relative lumped-mass displacement and energy
errors ≤10⁻⁴**. Fresh force residuals are recorded too, but a 10⁻⁴ force
residual is not the stopping target. These are finite-element solution errors,
not continuum discretization errors.

Median **seconds**, including setup, five trials:

| Reinforcement contrast | Direction | Scalar Jacobi | Block Jacobi | Scalar FSAI | Block FSAI |
|---:|---|---:|---:|---:|---:|
| 1 | rotated | 0.1705 | 0.1537 | 0.1311 | 0.1598 |
| 10 | rotated | 0.3425 | 0.2604 | 0.2114 | 0.2599 |
| 100 | aligned | 0.5792 | 0.5844 | 0.3662 | 0.5497 |
| 100 | rotated | 0.9115 | 0.5783 | 0.3926 | 0.6457 |
| 100 | varying | 0.8142 | 0.4815 | 0.3342 | 0.4858 |
| 1000 | aligned | 0.9621 | 0.9704 | 0.5737 | 0.8761 |
| 1000 | rotated | 2.1041 | 0.8708 | 0.5842 | 1.1389 |
| 1000 | varying | 1.8744 | 0.9483 | 0.6003 | 0.8505 |

![Reinforced bracket timing and actual displacement error](../assets/block-elasticity.png)

For spatially varying reinforcement at contrast 1,000:

| Family | Selected settings | Iterations | Setup (s) | Solve (s) | Total (s) | Retained factors (MB) |
|---|---|---:|---:|---:|---:|---:|
| Scalar Jacobi | CG | 23175 | 0.0005 | 1.8739 | 1.8744 | 1.63 |
| Block Jacobi | CG, direct | 11475 | 0.0005 | 0.9478 | 0.9483 | 4.90 |
| Scalar FSAI | CG, width 6, batch 1 | 5175 | 0.0164 | 0.5842 | 0.6003 | 17.76 |
| Block FSAI | CG, 4 blocks | 6650 | 0.0327 | 0.8168 | 0.8505 | 221.30 |


## Wider-support check

The best block setting reached the four-block edge of the initial sweep on
some cases, so we additionally tested **eight-block supports** and **24-entry
scalar factors** at contrast 1,000. CG and CR are both tested, with five
interleaved trials and repeated six-entry scalar/four-block controls.
The same reference and error checks apply. The larger factors reduce
iterations but increase total time: **six-entry scalar FSAI still wins** on
both directions. Small control-time differences between runs are ordinary
run variation; the figures retain the original sweep's measurements.
[Additional raw measurements](block-wide-check.json).

| Direction | Settings (best tested solver) | Iterations | Setup (s) | Solve (s) | Total (s) |
|---|---|---:|---:|---:|---:|
| aligned | Scalar FSAI 6, batch 1, CG | 4775 | 0.0185 | 0.5396 | 0.5581 |
| aligned | Scalar FSAI 12, batch 1, CG | 3500 | 0.0956 | 0.6808 | 0.7777 |
| aligned | Scalar FSAI 24, batch 1, CG | 2575 | 0.4682 | 0.6550 | 1.1232 |
| aligned | Scalar FSAI 24, batch 2, CG | 2550 | 0.3296 | 0.7484 | 1.0770 |
| aligned | Block FSAI 4 blocks, CG | 6725 | 0.0298 | 0.8537 | 0.8834 |
| aligned | Block FSAI 8 blocks, CG | 4750 | 0.2033 | 1.2727 | 1.4763 |
| varying | Scalar FSAI 6, batch 1, CG | 5175 | 0.0162 | 0.5883 | 0.6045 |
| varying | Scalar FSAI 12, batch 1, CG | 3900 | 0.0626 | 0.6047 | 0.6662 |
| varying | Scalar FSAI 24, batch 1, CG | 2975 | 0.3234 | 0.5480 | 0.8687 |
| varying | Scalar FSAI 24, batch 2, CG | 3275 | 0.2233 | 0.8657 | 1.0897 |
| varying | Block FSAI 4 blocks, CG | 6650 | 0.0299 | 0.8879 | 0.9195 |
| varying | Block FSAI 8 blocks, CG | 4800 | 0.2022 | 1.2751 | 1.4780 |

Scalar FSAI wins all eight bracket cases in the original sweep. A six-entry
scalar row can select components from more vertices than a two-block row.
That greater spatial reach is a plausible explanation for its advantage;
these measurements do not isolate one cause or rule out improved block
algorithms. They support keeping this block-FSAI implementation experimental.

## Storage and numerical updates

The sweep includes scalar FSAI widths 4, 6, 12 with growth batches 1 and 2,
and block FSAI supports of 2 and 4 vertices. A block support of size `b`
has at most `3b` scalar entries per scalar factor row, so the sweep contains
matched maximum-entry budgets (6 versus 2 blocks; 12 versus 4 blocks).
Adaptive scalar rows may stop early at `kap_tolerance=.003`. All completed
FSAI factors use float32 storage with float64 equations and accumulation.

We also apply a **common actual allocation ceiling of 116.14 MB**
to the spatially varying contrast-1,000 case. This is the measured retained
factor allocation of the two-block prototype. Every scalar candidate fits
under this ceiling; entries below are the fastest tested settings that fit,
not an exhaustive search over every possible sparsity pattern.

| Family | Best measured total within common allocation ceiling (s) | Actual factors (MB) |
|---|---:|---:|
| Scalar Jacobi | 1.8744 | 1.63 |
| Block Jacobi | 0.9483 | 4.90 |
| Scalar FSAI | 0.6003 | 17.76 |
| Block FSAI | 0.9013 | 116.14 |


Storage includes G, Gᵀ and the block prototype's additional BSR copies,
including their allocated spare capacity. The block prototype is particularly
wasteful here: it retains scalar copies and conversion overallocations.
These numbers describe this implementation, not an inherent block-FSAI memory
requirement. They exclude the common system matrix, solver workspace, and
refit workspace; they are **factor allocations**, not peak device memory.

For reuse, we increase contrast by 10% while keeping the matrix graph and
loads fixed (the manufactured tensor RHS changes consistently). We compare
fresh adaptive construction with numerical refits of supports chosen at the
old values. Each new solve starts at zero and must pass the same error checks.
The supported scalar refit retains its buffers. The experimental block refit
performs local Cholesky solves in Warp but still repacks and transposes its
factors, so it requires graph recapture. Both costs are included.

| Problem | Family | Fresh build + solve (s) | Refit + solve (s) | Refit setup alone (s) |
|---|---|---:|---:|---:|
| Tensor, κ=10⁶ rotated | Block Jacobi | 0.0060 | 0.0058 | 0.0003 |
| Tensor, κ=10⁶ rotated | Scalar FSAI | 0.0105 | 0.0082 | 0.0008 |
| Tensor, κ=10⁶ rotated | Block FSAI | 0.0115 | 0.0106 | 0.0031 |
| Bracket, 1 rotated | Block Jacobi | 0.1562 | 0.1561 | 0.0003 |
| Bracket, 1 rotated | Scalar FSAI | 0.1333 | 0.1254 | 0.0010 |
| Bracket, 1 rotated | Block FSAI | 0.1634 | 0.1634 | 0.0039 |
| Bracket, 1000 rotated | Block Jacobi | 0.9526 | 0.9530 | 0.0003 |
| Bracket, 1000 rotated | Scalar FSAI | 0.6136 | 0.5894 | 0.0012 |
| Bracket, 1000 rotated | Block FSAI | 1.0936 | 1.1003 | 0.0040 |
| Bracket, 1000 varying | Block Jacobi | 0.9516 | 0.9506 | 0.0004 |
| Bracket, 1000 varying | Scalar FSAI | 0.6183 | 0.6010 | 0.0014 |
| Bracket, 1000 varying | Block FSAI | 0.8792 | 0.8719 | 0.0075 |


These are repeated timings of one numerical transition per case, not a long
material-evolution trajectory. Jacobi is rebuilt in both update modes;
block Jacobi recomputes its diagonal inverses. Setup-once/multiple-RHS use can
also be assessed from the separately recorded setup and solve times.

## Refinement check

A separate nested Freudenthal-tet cantilever test holds domain, body load,
material direction and Poisson ratio fixed. It checks independently converged
solutions on successively refined meshes; it is a discretization check, not
a solver timing comparison. Compliance should increase toward the continuum
limit under these nested displacement spaces.

| Contrast | Compliance, h=1/8 | Compliance, h=1/16 | Increase with refinement |
|---:|---:|---:|---:|
| 1 | 0.00587107 | 0.00621039 | 5.8% |
| 10 | 0.00531875 | 0.00574725 | 8.1% |
| 100 | 0.00419053 | 0.00518888 | 23.8% |
| 1000 | 0.00161306 | 0.00330364 | 104.8% |


The contrast-1,000 beam's compliance more than doubles in the last refinement.
That is substantial mesh-dependent stiffening despite tiny algebraic errors.
The high-contrast bracket results therefore test algebraic solver performance;
they do **not** establish mesh-converged physical displacements or stresses.
Holding Poisson ratio fixed avoids confounding the experiment with a sweep
toward the incompressible limit, but it does not eliminate discretization
problems from strong directional constraints.

## Measurement and reproduction

NVIDIA L40, Warp 1.15, GPU float64 assembly/reference/iteration and mixed
precision FSAI application. No multigrid, external sparse factorization or
SciPy numerical backend is used. NumPy handles geometry, exact fields and
independent verification outside timings. Fresh-build candidates are warmed before
five interleaved forward/reverse timing rounds. The numerical-update modes
use five warmed trials per mode in separate groups, so small differences in
their totals are less conclusive than the large setup-cost differences. Wall time includes
preconditioner construction, solver graph capture and synchronized native
solve. Matrix assembly, JIT, diagnostics and transfers are excluded. Setup and solve
medians are reported separately; total time is the median of per-trial sums.
A native warm solve precedes every trial. Other GPU compute processes are
checked before and after timing; overlapping trials are discarded and retried.
Clocks are unlocked, so tiny differences and single-trial outliers are not
interpreted as algorithmic improvements.

Iteration targets come from the existing state-preserving CG/CR sampler,
with 25-iteration resolution after the initial checkpoints. Each timed native
endpoint is independently verified. Full trajectories, per-trial errors,
settings, source hashes, input fingerprints and reference checks are in
[the tensor results](block-coupling.json),
[oblique bracket results](block-elasticity-rotated.json), and
[other bracket directions](block-elasticity-directions.json).
The tensor run resumed after three completed cases when another GPU job
interrupted it; both runner provenance records are retained. The resumed
runner adds wait/retry handling without changing numerical kernels or settings.
The first runner hashed complete BSR allocations, including unused tail capacity;
those digests are explicitly labeled allocation hashes. Input file/recipe hashes
are also recorded, and the current runner hashes only active matrix entries.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/block_challenge.py \
  --problem coupling --contrasts 1 100 10000 1000000 \
  --directions aligned rotated --updates --output results/block-coupling.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/block_challenge.py \
  --problem elasticity --contrasts 1 10 100 1000 --directions rotated \
  --target 1e-4 --updates --output results/block-elasticity-rotated.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/block_challenge.py \
  --problem elasticity --contrasts 100 1000 --directions aligned varying \
  --target 1e-4 --updates --output results/block-elasticity-directions.json
python benchmarks/block_wide_check.py
python benchmarks/block_rotation_audit.py
python benchmarks/block_refinement.py
python benchmarks/block_refinement.py --contrasts 10 100 --levels 4 8 16 \
  --output results/block-refinement-intermediate.json
python benchmarks/plot_block_challenge.py results/block-coupling.json \
  --output assets/block-coupling.png
python benchmarks/plot_block_challenge.py results/block-elasticity-rotated.json \
  results/block-elasticity-directions.json --directions rotated varying \
  --output assets/block-elasticity.png
```

Use `--resume` to retain completed cases after an interruption. The bracket
requires the existing prepared `data/simjeb-ftetwild` inputs. Regression tests
cover CPU/CUDA tensor spectra, reinforced element energy/forces, fixed-support
block refits against independent dense local solves, aliases, graph replay,
and failed SPD-update rollback, alongside the existing full suite.

Final validation: **198 tests passed** (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`), including CPU and CUDA. The [result audit](block-validation.json) checks every recorded native endpoint and trial count.
