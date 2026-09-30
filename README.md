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

![Head and tail constraints, followed by FSAI-CG and Jacobi-CR biharmonic interpolation at k, 10k and 100k.](assets/dragon-dirichlet-comparison.png)

A second comparison prescribes **−1 on the tail and +1 on the head**, then
minimizes $u^TLM^{-1}Lu/2$ with **no data term**. Free vertices start at zero.
The first dragon marks the fixed regions; the next four compare converged
FSAI-CG with Jacobi-CR at matched iteration budgets. The reduced system is
SPD with 320,495 unknowns; constraints are enforced exactly by elimination.
The regenerated figure uses tuned width-48 FSAI and the same factored
operator for both solvers. Here **k = 33,374**: FSAI takes **8.56 s to solve
plus 1.07 s setup**. Jacobi's relative field error is **83.7% at k**, **84.0%
at 10k**, and **1.28e-9 at 100k** (403.77 s). The nonmonotonic early field
error is reported directly. A long-double energy reference with AMD
Cholesky corrections independently checks the result; the
[original width-eight figure](visualization/dirichlet-original.md) is archived.
[Region definitions, convergence checks, and reproduction](visualization/dirichlet.md).

![Log-log residual convergence for the tuned dragon Dirichlet problem: tuned FSAI-CG, tuned FSAI-CR, Jacobi-CR, and Jacobi-CG, with independently recomputed and recursive residuals.](assets/dragon-dirichlet-residuals.png)

The log–log plot measures relative residual norm, $\|r_k\|_2/\|b\|_2$.
Solid curves independently recompute the full biharmonic energy gradient;
dashed curves show Warp's recursive residual. They can diverge near
convergence on this ill-conditioned system. Samples preserve each solver's
Krylov state, with no restarts or smoothing. The zero-iteration residual is
1 and is omitted from the logarithmic axis.
Both Jacobi-CG and Jacobi-CR run through the same 3,337,400-iteration budget,
so their convergence can be compared directly.
Tuned FSAI-CR uses the same preconditioner as FSAI-CG and reaches the same
recursive tolerance in **33,256 iterations**, versus CG's **33,374**.
Their independently recomputed final residuals are **6.20e-9** and **8.82e-10**,
respectively; equal recursive tolerances do not imply equal final accuracy.
[Convergence data and reproduction](visualization/dirichlet.md#residual-history).

The [FSAI tuning study](results/dirichlet-tuning/README.md) reduces
this unregularized solve to about **10 seconds including setup**, using
wider adaptive factors and a factored squared-Laplacian operator. A sparser
alternative roughly halves the original cost per iteration. Controlled
tests explain the large timing change when the data term is removed, and a
higher-precision reference exposes an accuracy limit in the previously
assembled squared matrix.

![FSAI tuning: conditioning, total time, per-iteration cost, and parameter tradeoffs.](assets/dragon-dirichlet-tuning.png)

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

![Reference and actual bracket iterates frozen near the first winner's wall-clock time.](assets/simjeb-elasticity-comparison.png)

[Problem, tuning, PR comparison, verification, attribution, and reproduction](results/elasticity.md).
The [earlier dragon elasticity results](results/archive/dragon-elasticity.md) remain archived.

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

Scalar CSR and square BSR blocks work in float32 or float64. Block inputs
are scalarized **on the device** during setup; vector-valued solver arrays
are supported. This is scalar FSAI on a BSR input, not a dense block-FSAI
algorithm. Use float64 for the poorly conditioned biharmonic benchmark.

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

Each row starts with only its diagonal. A greedy step selects the largest
absolute residual entry from the graph frontier of the current support.
The local Cholesky factor is extended by bordering, and `z` is recomputed
by two local triangular solves. Rows run independently; neither setup nor
apply requires a global triangular solve. There is no bounded candidate
hash table: all frontier entries are considered, with duplicates evaluated
again. The resulting lower triangular `G` has a positive diagonal, so
`G^T G` is SPD and suitable for CG.

- `max_row_size=8`: maximum entries including the diagonal, range 1–64.
  Eight is a useful starting point for the dragon benchmark. One recovers
  Jacobi exactly. Denser rows increase setup and application cost.
- `kap_tolerance=1e-3`: stop after relative improvement in row energy
  `psi=1/z_i` drops below this value. Zero disables this early stopping.
- `pivot_floor`: stop growing a row if an equilibrated local pivot is too
  small (defaults: `1e-12` in float64, `1e-6` in float32). The accepted
  positive factor is retained; `P.truncated_rows` counts these events.
- `apply_lanes=1`: CUDA threads cooperating on each factor row; accepts
  1, 2, 4, 8, 16, or 32. Four helps the tuned dragon. CPU uses ordinary CSR.
- `factor_dtype=None`: defaults to matrix precision. `wp.float32` stores
  the completed factors in float32 while accumulating in matrix precision.
  The rounded factor is validated before constructing its transpose;
  this option adds a setup synchronization. Validate solution accuracy
  for the intended problem.
- `P.G`, `P.GT`: the scalar Warp BSR factors, available for inspection.

This is an independent implementation inspired by [hypre's adaptive
FSAI](https://hypre.readthedocs.io/en/latest/solvers-fsai.html). Differences
include diagonal equilibration, adding one entry per step, and bordering
local Cholesky factors. It is not a bitwise port or a comparison against
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
CPU/CUDA and float32/float64, check block input and solver integration,
verify alpha/beta aliasing and graph replay, and validate the lifted
system against the original biharmonic solve.
