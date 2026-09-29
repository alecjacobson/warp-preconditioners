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

FSAI-CG reaches its relative stopping tolerance of **1e-8 at k = 28,688**.
Left to right after the raw data: **FSAI-CG at k; Jacobi-CR at k, 10k = 286,880, and
100k = 2,868,800**. Each is one uninterrupted solve. All five share a scene,
scalar range, and `isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))` palette.
The figure shows independently recomputed residuals; a longer FSAI run
also verifies field stability. This nonlinear, stronger-smoothing example
is a different problem from the α = 1 height benchmark timings above.
[Target formula, convergence checks, and reproduction](visualization/README.md).

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
