# Warp preconditioners

Adaptive factorized sparse approximate inverse (FSAI) for NVIDIA Warp's BSR
matrices and `warp.optim.linear` solvers. Setup and application run entirely
in Warp on CPU or CUDA. There is no multigrid, external sparse factorization,
SciPy numerical backend, or matrix download in the library.

On the L40 dragon biharmonic problem, width-eight FSAI-CG solves all three
right-hand sides in **0.708 seconds median total**, including setup, at
backward error **8.83e-11**. See [measurements and caveats](results/README.md)
for repeated trials, Jacobi comparisons, and the upstream timing schedule.

![Five dragons in one scene: raw input f, converged FSAI-CG at k, and Jacobi-CR at k, 10k and 100k. All use the same 26 crisp OKLab intervals.](assets/dragon-biharmonic-comparison.png)

The first dragon shows **raw input f**; the next four show solutions of `(α M + K M^-1 K)u = α Mf`, with
**α = 0.0001** and a **nonlinear combination of x, y, z and xz** as the data
function. The converged field remains strongly non-affine while bending
energy falls by about **99.89%**. The raw-versus-smoothed comparison shows
that broad variation remains: RMS change is **19.2% of the input standard
deviation**, while standard deviation decreases by only **7.0%**. The energy
reduction alone is not a measure of visible change. All methods start from the same data
function; a [separate initialization experiment](visualization/initialization.md)
finds modest FSAI iteration savings and substantially better early Jacobi fields.

The Dirichlet tuning also improves this smoothing problem: three warmed
trials reduce median FSAI solve time from **10.14 s to 2.80 s**, and total
setup-plus-solve time from **10.17 s to 3.96 s (2.57× faster)**.
Both configurations use the same data, initial guess, and stopping tolerance.
[Repeated measurements](results/smoothing-tuning.json).

Tuned FSAI-CG reaches its relative stopping tolerance of **1e-8 at k = 9,482**.
Left to right after the raw data: **FSAI-CG at k; Jacobi-CR at k, 10k = 94,820, and
100k = 948,200**. Both solvers now use the same factored operator including
the data term. Each is one uninterrupted solve. All five share a scene,
scalar range, and `isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))` palette.
The figure shows independently recomputed residuals; a longer FSAI run
verifies field stability, and a [higher-precision reference](results/smoothing-verification.json)
checks field accuracy. This nonlinear, stronger-smoothing example
is a different problem from the α = 1 height benchmark timings above.
[Target formula, convergence checks, and reproduction](visualization/README.md).

![Head and tail constraints, followed by selected FSAI-CG and Jacobi-CG at k, 10k and 100k, with refreshed timings.](assets/dragon-dirichlet-comparison.png)

This comparison prescribes **−1 on the tail and +1 on the head**, then
minimizes $u^TLM^{-1}Lu/2$ with **no data term**. Free vertices start at zero.
The first dragon marks the fixed regions; the next four show **FSAI-CG** and
**Jacobi-CG at k, 10k and 100k**. Both use the same factored operator, with
exact constraint elimination and 320,495 free unknowns.

The image has been rerun after the FSAI setup/packing changes. We compared
FSAI-CG and FSAI-CR at **relative mass-weighted field error ≤1e-8** against the
independently refined reference. **CG wins: 10.14 s median setup + solve**,
versus **10.61 s for CR**, across three interleaved warm repeats. The retained
width-48 FSAI-CG reaches the target at **k = 33,125** sampled iterations.
Labels show total time, separate solve/setup components, field error and
independently recomputed residual. Jacobi timings are single warm runs.

[Fresh selection measurements](results/dirichlet-refresh.json),
[problem, accuracy checks and reproduction](visualization/dirichlet.md).
The [original width-eight figure](visualization/dirichlet-original.md) is archived.

![Fresh log-log residual trajectories for selected FSAI-CG, Jacobi-CG and Jacobi-CR.](assets/dragon-dirichlet-residuals.png)

The residual plot compares **FSAI-CG, Jacobi-CG and Jacobi-CR**. The dragon
rendering above retains Jacobi-CG as its single baseline. Solid curves recompute
$\|r_k\|_2/\|b\|_2$ from the full biharmonic energy gradient; dashed curves
show Warp's recursive residual. They can diverge on this ill-conditioned
system. Samples preserve Krylov state without restarts or smoothing, and
all initial residuals are 1. Iteration zero is omitted from the log axis.
The selected FSAI endpoint is based on verified **field accuracy**, not a
recursive residual threshold. Both Jacobi methods run through **3,312,500 iterations**.

The [earlier FSAI tuning study](results/dirichlet-tuning/README.md) explains
why wider factors and a factored squared-Laplacian operator help this
unregularized problem, and why assembling the squared matrix limits accuracy.
Its historical timings are separate from the refreshed measurements above.

![FSAI tuning: conditioning, total time, per-iteration cost, and parameter tradeoffs.](assets/dragon-dirichlet-tuning.png)

A [follow-up performance study](results/fsai-followup.md) adds optional
**batched support growth**: `max_step_size=2` with the width-48 recipe reduces
median dragon setup-plus-solve time from **10.19 s to 9.46 s (7.2% less)**
in five isolated GPU trials at the same verified field accuracy. Single-entry
growth remains the default. Separate factor thread counts, a fused application,
and a block-FSAI prototype did not produce repeatable wins and remain
experiments. The figures above retain their recorded single-entry settings.

## Stronger tests for blocking

A [controlled coupling and anisotropic-elasticity study](results/block-challenge.md)
tests scalar Jacobi, block Jacobi, scalar FSAI and block FSAI with CG and CR.
On `A = K ⊗ C`, rotating a component contrast of **10⁶** exposes a large
blocking benefit: block Jacobi takes **6.08 ms**, versus **788.84 ms** for
scalar Jacobi, including setup—about **130× faster** at the same verified
solution and energy accuracy. Block Jacobi needs 100 sampled iterations
at every tested contrast and orientation.

![Controlled component coupling: total time and verified solution error.](assets/block-coupling.png)

The obliquely reinforced fTetWild SimJEB bracket still favors **scalar FSAI**.
At reinforcement contrast **1,000**, its median total is **0.584 s**, versus
**0.871 s** for block Jacobi, **1.139 s** for block FSAI, and **2.104 s** for
scalar Jacobi. These are five warmed trials at both relative displacement
and energy errors ≤10⁻⁴, with the original mesh, constraints and load.

![Reinforced bracket: total time and verified displacement error.](assets/block-elasticity.png)

The study also measures fixed-pattern numerical refits and actual factor
allocations. Independent reference and element-energy checks are included.
A separate cantilever refinement test shows substantial mesh-dependent
stiffening under strong reinforcement, so the high-contrast results establish
algebraic solver performance, not mesh-converged physical stresses. The block prototypes
remain under `benchmarks/`; no additional supported API is introduced.

## Indefinite mixed systems

The new `MixedHarmonicSystem` and `ShiftedBlock` preconditioner solve the
original mixed biharmonic dragon's **three coordinate RHSs in 0.615 s median
total**, using Warp GMRES. A harder nonlinear case with data weight **1e-4**
solves in **2.250 s median total** after symmetric block equilibration;
its original-system relative residual is **4.25e-9**. Both totals include
operator construction and preconditioner setup, excluding input uploads.
These use a fixed GMRES restart, not the upstream adaptive-chunk schedule.

Pure-Warp `BlockJacobi` and parallel `BlockILU0` are also available for
scalar and square BSR blocks of size 1–4. ILU is a tested alternative, but
FSAI in the first-order block construction was faster on these cases.
The actual mixed triharmonic benchmark **remains unconverged**: its dumped
signs give an indefinite Schur complement, unlike the positive triharmonic
energy. Failed trials and manufactured-solution errors are reported.

![Indefinite preconditioner results: successful biharmonic solves and unresolved triharmonic residuals.](assets/indefinite-preconditioners.png)

[Algorithms, exact matrix signs, accuracy checks, API examples, and reproduction](results/indefinite.md).

## Tetrahedral linear elasticity

The current example is [SimJEB #225](https://simjeb.github.io/), Michael
Jenkins's GE bracket: **215,219 original tetrahedra and 131,079 free DOFs**.
It uses the supplied titanium material, fixed bolt holes, and **35.6 kN
vertical pin load**. All assembly, preconditioners, references, and compared
solves run in Warp on NVIDIA L40. Both relative displacement and energy
errors must reach **1e-4**.

The input boundary conditions are shown explicitly below: **orange nodes have
u_x=u_y=u_z=0**; **blue nodes receive the distributed pin load and remain free
to move**. The see-through view exposes all selected nodes and the coupling
centers. The gray base plate is free outside the four bolt-hole regions.

![Exact fixed node sets and applied pin load, with top, underside, and see-through views.](assets/simjeb-boundary-conditions.png)

| Best tested configuration | Iterations | Median setup + solve |
| --- | ---: | ---: |
| Default scalar Jacobi + CG | 2,035 | 0.1123 s |
| Block Jacobi + CG | 1,846 | 0.1038 s |
| FSAI, width 4, κ=.003, float32 factors, one lane + CG | 1,132 | **0.0820 s** |

FSAI is **1.27× faster than block Jacobi**, including setup. Our streamlined
block Jacobi and the actual [Warp PR #1890](https://github.com/NVIDIA/warp/pull/1890)
direct implementation apply the same inverse to within **2.39e-16** and have
practically tied total times. CG wins over CR for these finalists.

![Bracket displacement and energy errors versus iterations and setup-plus-solve wall time.](assets/simjeb-elasticity-convergence.png)

The rendering freezes all three methods near **0.0820 s**, alongside the
verified reference. It shows **actual displacement**, colored by recovered
**von Mises stress in MPa**, with a shared linear scale and crisp intervals.
Maximum reference displacement is **0.798806 mm**, matching the published
SimJEB metadata to its reported precision.

The image distinguishes **time to target** from **snapshot time**. All three
snapshots stop near the same budget; only FSAI meets both error targets there.
Block Jacobi's 0.0814 s snapshot still has 0.0479% displacement error and
0.145% energy error, above the 0.01% target. Its time to target is 0.1038 s,
compared with FSAI's 0.0820 s.

![Reference and actual bracket iterates frozen near the first winner's wall-clock time.](assets/simjeb-elasticity-comparison.png)

The peak stress is on the **underside of a fixed bolt hole**. The
[boundary-condition and stress audit](results/elasticity.md#boundary-condition-and-stress-audit)
checks the source deck, reaction balance, and an independent stress calculation.
An [underside view](assets/simjeb-elasticity-underside.png) exposes those concentrations;
stress colors now remain visible on constrained surfaces.

[Problem, tuning, PR comparison, verification, attribution, and reproduction](results/elasticity.md).
The [earlier dragon elasticity results](results/archive/dragon-elasticity.md) remain archived.

### Does better tetrahedral mesh quality help?

We also remeshed the bracket with **fTetWild**. The original bracket uses
SimJEB's HyperMesh tetrahedra; TetGen was used for the earlier dragon.
The worst dihedral angle improves from **5.36° to 14.31°**, with no elements
below 10°. The new mesh has **351,711 tetrahedra** and preserves the tagged
bolt/pin regions, material, total pin force and reference moment.

The stress concentrations persist at the same bolt holes. Maximum displacement
changes from **0.7988 to 0.8214 mm**; the area-weighted surface stress difference
is **7.74%**. This improves element quality but does not establish stress
convergence. Element count and equal-weight nodal load sampling also change.
FSAI still wins the tested settings: **0.1331 s**, versus **0.1533 s** for block
Jacobi and **0.1693 s** for scalar Jacobi, including setup on NVIDIA L40.

![Original and fTetWild converged stresses, with top, underside and all fixed/loaded boundary nodes.](assets/simjeb-mesh-comparison.png)

[Mesh-quality plots, boundary-transfer checks, convergence curves and reproduction](results/simjeb-mesh-comparison.md).

### Displacement and larger loads

Increasing the pin force to **2×, 5× and 10×** on the fTetWild mesh gives the
same sampled iterations to relative accuracy: **2,059 for Jacobi, 1,834 for
block Jacobi, and 1,182 for FSAI**. FSAI takes **0.131–0.133 s**, including
setup, across these loads. Scaling the force in this linear model scales the
displacement while leaving the stiffness matrix and conditioning unchanged.

The figure below colors **displacement magnitude** and shows actual geometry
at **1× display scale**, with the unloaded silhouette in gray. Maximum
displacement grows from **0.8214 mm to 8.2138 mm**. Each row has its own labeled
color range. The model remains small-strain linear elasticity at both loads;
deformation-dependent stiffness would require a nonlinear experiment.

![Actual displacement snapshots at 1x and 10x force, colored by displacement magnitude, with unloaded outlines.](assets/simjeb-displacement.png)

![Relative convergence and five-repeat timing comparison as force increases.](assets/simjeb-load-scaling.png)

[Load-scaling measurements, verification and reproduction](results/simjeb-load-scaling.md).

## Fixed sparsity, changing values

`BlockJacobi.update(A)` recomputes only diagonal inverses. FSAI can cache the
selected factor supports, source-value mapping, and transpose permutation:

```python
P = FSAI(A, max_row_size=8, reuse_pattern=True)
# Update A.values in place, or supply another matrix with identical topology.
P.update(A)
cg(A, b, x, M=P, tol=1e-8)
```

Refits run entirely in Warp and keep factor buffers stable for captured
applications. Invalid topology or local pivots raise before replacing the
previous factors. FSAI reuse currently requires compact BSR storage; use
`warp.sparse.bsr_copy` to canonicalize padded matrices first. Setup and updates
synchronize for validation; application is graph-capturable.

A refit keeps the original supports—it does not repeat adaptive pattern
selection. Monitor convergence and rebuild if that pattern stops working
well. Optional `P.quality(A)` estimates the normalized Frobenius defect of
`G A G.T`; it is a diagnostic heuristic, not a condition-number bound.

On ten spatially varying stiffness fields of the bracket, width-eight FSAI
refits take **1.0 ms versus 25.5 ms** for fresh setup. Including the initial
plan and all solves, refitting takes **0.950 s**, versus **1.168 s** for fresh
builds and **1.460 s** for updated block Jacobi. The final material contrast
is 263×; fresh and refitted width-eight factors both need 1,321 iterations.

![FSAI numerical refits versus fresh builds, stale factors, and updated block Jacobi.](assets/simjeb-numerical-updates.png)

[Changing-material benchmarks, measured update costs, and API details](results/numerical-updates.md).

```bash
python -m pip install -e '.[test]'
pytest -q
```

```python
import warp as wp
from warp.optim.linear import cg
from warp_preconditioners import FSAI

# A: fully stored symmetric positive definite warp.sparse.BsrMatrix.
# b, x: Warp arrays on A.device (x is the initial guess and output).
P = FSAI(A, max_row_size=8)
with wp.ScopedDevice(A.device):
    iterations, residual, tolerance = cg(A, b, x, M=P, tol=1e-10)
```

Scalar CSR and square BSR blocks work in float16, float32 or float64. Block inputs
are scalarized **on the device** during setup; vector-valued solver arrays
are supported. This is scalar FSAI on a BSR input, not a dense block-FSAI
algorithm. Use float64 for the poorly conditioned biharmonic benchmark.
Float16 uses native half-precision construction, factors and application.
The [Warp handoff PR](https://github.com/alecjacobson/warp/pull/16) also fixes
half-precision CR compilation; older Warp releases may require that fix.

`FSAI` is a `LinearOperator` implementing `z = alpha * P*x + beta*y`,
including aliased buffers and `beta=0`. Apply uses two sparse products,
`G*x` and `G.T*tmp`, with an explicitly stored transpose. It is compatible
with CG, CR, BiCGSTAB and GMRES when the system meets each solver's own
requirements. Apply supports CUDA graph capture. Setup must run outside
capture: it reads two diagnostic integers to the host. Each instance owns
scratch storage; use separate instances for concurrent streams. Rebuild
when the input matrix changes. Setup is not differentiable.

## Algorithm and parameters

For SPD `A`, equilibrate with `D = sqrt(diag(A))`. For each row `i`, grow a
support `S` containing `i` and selected indices below `i`. Solve the small
principal system

```
(D^-1 A D^-1)[S,S] z = e_i
G[i,S] = z^T D[S]^-1 / sqrt(z_i)
P = G^T G
```

Each row starts with only its diagonal. A greedy search selects the largest
absolute residual entries from the graph frontier of the current support.
The local Cholesky factor is extended by bordering; the forward solve is
extended with each new entry, then a backward solve updates `z`. Rows run independently; neither setup nor
apply requires a global triangular solve. There is no bounded candidate
hash table: all frontier entries are considered. Candidates already in the
current top selection are skipped; other duplicates may be evaluated again. The resulting lower triangular `G` has a positive diagonal, so
`G^T G` is SPD and suitable for CG.

- `max_row_size=8`: maximum entries including the diagonal, range 1–64.
  Eight is a useful starting point for the dragon benchmark. One recovers
  Jacobi exactly. Denser rows increase setup and application cost.
- `max_step_size=1`: maximum entries added per search, range 1–64. The
  default preserves single-entry growth. Batching can reduce setup and total
  time but changes the supports; larger batches are not always faster.
- `kap_tolerance=1e-3`: stop after relative improvement in row energy
  `psi=1/z_i` drops below this value. Checked after each batch; retune when
  changing batch size. Zero disables this early stopping.
- `pivot_floor`: stop growing a row if an equilibrated local pivot is too
  small (defaults: `1e-12` in float64, `1e-6` in float32, `1e-3` in float16). The accepted
  positive factor is retained; `P.truncated_rows` counts these events.
- `apply_lanes=1`: CUDA threads cooperating on each factor row; accepts
  1, 2, 4, 8, 16, or 32. Four helps the tuned dragon. CPU uses ordinary CSR.
- `factor_dtype=None`: defaults to matrix precision. For float64 input,
  `wp.float32` stores the completed factors in float32 while accumulating in float64.
  The rounded factor is validated before constructing its transpose;
  this option adds a setup synchronization. Validate solution accuracy
  for the intended problem.
- `P.G`, `P.GT`: the scalar Warp BSR factors, available for inspection.

This is an independent implementation inspired by [hypre's adaptive
FSAI](https://hypre.readthedocs.io/en/latest/solvers-fsai.html). Differences
include diagonal equilibration, single-entry growth by default, and
bordering local Cholesky factors. It is not a bitwise port or a comparison against
hypre's compiled implementation. Positive diagonals are checked; full
symmetry and positive definiteness remain caller preconditions. Supply
both triangular halves of `A`, with canonical sorted BSR topology.

A sparse approximate inverse is not a mesh-independent method: increasing
the mesh resolution or worsening the low-frequency spectrum can still
increase iteration counts. Large supports/high-valence graphs also make
the serial work within each setup row expensive.

`SparseOperator(A, row_lanes=4)` exposes the cooperative matrix product as
a Warp linear operator. For a full symmetric scalar Laplacian and lumped
mass, `SquaredLaplacianOperator(L, mass, free_indices, row_lanes=4)` applies
the reduced squared energy through two Laplacian products; its
`rhs(prescribed)` method eliminates fixed values while retaining all rows
of the full energy. Both support CUDA graph capture after construction.
For smoothing, pass every vertex as free and set `mass_weight=alpha` to
apply `L M^-1 L + alpha M`; supply `alpha M f` as the solver RHS.
See the [tested Dirichlet recipe](results/dirichlet-tuning/README.md#recommended-usage)
for FSAI parameters, input conventions, and accuracy checks.

## Biharmonic structure experiment

The linked benchmark solves `Q = M + K M^-1 K`, where `K=-L` is the positive
semidefinite stiffness matrix and `M` is positive diagonal. The package
also implements an equivalent first-order formulation:

```python
from warp_preconditioners import BiharmonicSystem
from warp.optim.linear import gmres

system = BiharmonicSystem(K, mass, tau=1.0)  # Q=M+tau*K*M^-1*K
rhs = system.rhs(b)                        # interleaved (b,0)
x2 = wp.zeros_like(rhs)
P2 = system.preconditioner(max_row_size=8)
with wp.ScopedDevice(K.device):
    gmres(system.matrix, rhs, x2, M=P2, tol=1e-12)
u = system.solution(x2)
```

The 2×2-block BSR matrix is `[[M,-sqrt(tau)*K],[sqrt(tau)*K,M]]` in
interleaved ordering. Eliminating its second component recovers `Q`
exactly. `P2` applies FSAI of `H=M+sqrt(tau)*K` independently to the two
components. Use a nonsymmetric solver, **not CG/CR**. Check the original
`Q` residual after extracting `u`: a small lifted residual can be amplified
by the elimination.

With *exact* `H` inverses, `H^-1 M H^-1` would precondition `Q` with spectrum
in `[1/2,1]` (diagonalize `M^-1/2 K M^-1/2`; the eigenvalue is
`(1+t^2)/(1+t)^2`). This bound does not apply to a sparse FSAI approximation.
The lifted formulation is experimental; direct FSAI-CG was substantially
better in the initial dragon tests, while lifted BiCGSTAB diverged.

This API requires the actual `K` and `mass`; it does not infer them from
`Q`. For constrained problems, pass operators for which the stated
factorization holds: slicing an already squared operator is generally
different from squaring its sliced stiffness matrix.

## Reproduce the dragon experiment

Use the [`alecjacobson/refresh` benchmark](https://github.com/alecjacobson/sparse-solver-benchmark/tree/alecjacobson/refresh)
and its `xyzrgb_dragon-720K.ply`. Dump the exact C++ systems:

```bash
/path/to/sparse_solver_benchmark /path/to/xyzrgb_dragon-720K.ply \
    --dump-matrices /tmp/dump --dump-only
python benchmarks/dragon.py --dir /tmp/dump --width 8 \
    --methods jacobi-cg jacobi-cr fsai-cg fsai-cr \
    --schedule upstream --seconds 60 --output results/dragon.json
python benchmarks/synthetic.py --size 32 --output results/grid32.json
```

Use `--k 1` or `--k 3` with explicit Jacobi/FSAI methods to test the
harmonic or triharmonic SPD matrices. Lifted methods apply only to `--k 2`.

The runner verifies the biharmonic factorization against the dumped
matrices. SciPy/NumPy are used only in benchmarks/tests for I/O, input
construction and independent verification. Preconditioner construction
and all solves use Warp. Setup includes all library construction costs
(and lifted matrix assembly when applicable), after a setup/solve warmup.
Transfers of input matrices are excluded. Solve timing includes graph
capture on subsequent solver calls, external residual checks, and output
transfers. All three coordinate RHS columns start from zero, sharing a
single preconditioner.

`--schedule upstream` follows the upstream adaptive chunks: start at 10,
target approximately five seconds per chunk, cap at 5,000, and stop after
five chunks without a 1% improvement. `--schedule fixed --chunk 500` is
also available. Each chunk restarts the Krylov method from its current
solution. The configurable time limit is per RHS; reaching it is not
reported as convergence. The external target is the same componentwise
backward error `max(abs(b-Qx)/(abs(Q)*abs(x)+abs(b)))`, default `1e-8`.
Normwise relative residual is reported too. Backward error alone is not a
forward-error guarantee on an ill-conditioned system.

Tests compare factors with independent dense/local solves, exercise
CPU/CUDA and float16/float32/float64, check block input and solver integration,
verify alpha/beta aliasing and graph replay, and validate the lifted
system against the original biharmonic solve.
