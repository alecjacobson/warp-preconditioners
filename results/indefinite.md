# Indefinite mixed dragon systems

Pure-Warp experiments on the NVIDIA L40, Warp 1.15.0, float64, using the
360,757-vertex dragon. The mixed biharmonic matrix has 721,514 scalar
unknowns; mixed triharmonic has 1,082,271. Preconditioner construction and
solves use no multigrid, external sparse factorization, or SciPy solver. NumPy/SciPy in the runner handle
input files, construction of independent checks, and residual evaluation.

![Indefinite preconditioner measurements](../assets/indefinite-preconditioners.png)

**Validated improvement:** the mixed biharmonic system solves all three
original coordinate RHSs in **0.615 s median total**. A harder nonlinear
case with data weight 1e-4 solves in **2.250 s median total**, with symmetric
block equilibration. **Unresolved:** the actual mixed triharmonic benchmark
still fails the accuracy criteria. Residual reduction alone is not counted
as success.

## The actual matrices and a sign distinction

Let K=-L be the positive cotangent stiffness and M the positive diagonal
lumped mass. The dumped benchmark matrices are exactly

```
A2 = [ alpha M  -K ]     A3 = [ alpha M    0     -K ]
     [   -K    -M ]          [    0      -K     -M ]
                             [   -K     -M      0 ]
```

The original benchmark uses alpha=1. Our BSR layout interleaves each
vertex's two or three unknowns, an exact permutation of the dumped block
ordering. `MixedHarmonicSystem` assembles these matrices on device from
scalar BSR K and a Warp mass array.

Eliminating the auxiliary fields gives

- A2: **alpha M + K M^-1 K**, an SPD Schur complement.
- A3: **alpha M - K M^-1 K M^-1 K**, an indefinite Schur complement.

The second sign differs from the positive triharmonic energy described in
the upstream prose. We preserve the actual dumped matrix and
[`build_mixed_system` signs](https://github.com/alecjacobson/sparse-solver-benchmark/blob/ab2c9c76a2fd15c7d77e7d7f9ae1fc75f1182324/main.cpp#L741);
we do not silently replace its middle block by +K. The runner verifies both
matrices against the dumps. The triharmonic Schur complement has troublesome
modes when a generalized stiffness eigenvalue approaches alpha^(1/3).
This is a different difficulty from merely squaring an SPD Laplacian.

## What was implemented

- **BlockJacobi:** invert each diagonal scalar/2x2/3x3/4x4 BSR block in Warp.
  Coupled blocks can be nonsingular even when scalar diagonal entries are
  zero, as in the triharmonic multiplier block.
- **BlockILU0:** synchronous parallel incomplete block LU on the existing
  BSR pattern, followed by fixed Jacobi sweeps for both triangular solves.
  The setup follows the parallel iterative-factorization idea of
  [Chow and Patel](https://doi.org/10.1137/140968896). This implementation uses
  synchronous old/new buffers, not asynchronous updates. There is no fill,
  reordering, or pivoting between blocks. Bad local pivots are rejected.
- **MatchingSchur:** block elimination with a repeated approximation to
  H^-1, where H=K+alpha^(1/p)M. This applies H^-1 M H^-1 for p=2, or its
  signed three-factor analogue for the actual p=3 benchmark. The initial
  trials were poor; composing inaccurate inverses is not automatically a win.
- **ShiftedBlock:** a first-order block approximation using the same H,
  with either FSAI or the new scalar ILU(0) as its approximate inverse.
  This was the successful approach for mixed biharmonic.

All application paths are fixed linear maps, so ordinary **right GMRES**
is appropriate. No tolerance-driven inner CG is hidden inside the
preconditioner. These indefinite/nonsymmetric preconditioners must not be
substituted into CG. Setup reads validation counters only; factors and
matrix values remain on device. Applications support aliasing and CUDA
graph capture and own scratch buffers; do not share an instance between
concurrent streams.

For biharmonic, equilibrate with D=diag(alpha^-1/4,alpha^1/4), so that

```
D A2 D = [ S  -K ],  S=sqrt(alpha) M,  H=S+K.
         [-K  -S ]

P = 1/2 [ H^-1  -H^-1 ]
        [-H^-1  -H^-1 ]
```

With exact H inverses the preconditioned eigenvalues are
(1 +/- i*t)/2, with |t|<=1. This spectral statement is not a guarantee for
an approximate FSAI/ILU inverse or for Euclidean GMRES convergence.
It explains why the first-order block construction is worth testing
instead of composing two approximate inverses.

Equilibration solves `(D A D)y=D b`, then returns `x=D y`; it changes neither
the problem nor the residual used for final verification. For triharmonic,
D=diag(alpha^-1/3,1,alpha^1/3). Its leading-K block inverse uses three H
inverses but leaves the resonant modes unresolved.

## Original mixed biharmonic: all three RHSs

Three warmed repeats, alternating method order, zero initial guesses,
GMRES restart 31, solver relative tolerance 1e-10. Both FSAI variants use
kap=0.003, four lanes per sparse row, float32 factor storage and float64
arithmetic. Total includes on-device operator construction, preconditioner
setup and all three solves; input uploads and independent CPU verification
are excluded. Common operator construction is measured once per case and
included in each reported total. This is not the upstream adaptive-chunk
schedule, and direct solvers were not rerun.

| Preconditioner | Iterations per RHS | Median total, 3 RHSs (s) | Worst true relative residual | Worst componentwise backward error |
| --- | ---: | ---: | ---: | ---: |
| ShiftedBlock + FSAI width 8 | 186 | 0.615 | 2.28e-11 | 1.13e-9 |
| ShiftedBlock + FSAI width 48 | 124 | 0.629 | 1.56e-12 | 2.25e-10 |

Width 48 reduces iterations, but its setup cost offsets that saving here.
[All repeat measurements](indefinite-original-repeats.json).

For context, a one-RHS probe with a 3,100-iteration budget gave relative
residual 3.16e-3 for block Jacobi and 3.59e-1 for direct block ILU with 32
triangular sweeps. Neither converged. See
[the first-order comparison](indefinite-second-probe.json) and
[initial Schur/ILU experiments](indefinite-probe.json).
The initial probe records Warp's raw device iteration counter separately
from the corrected Arnoldi-step count; later runs use host checks every
restart and need no correction.

## Harder mixed biharmonic: nonlinear RHS, alpha=1e-4

The target is the same deterministic nonlinear field used by the
[smoothing image](../visualization/README.md): b=(alpha M f,0).
All methods start at zero. The first unbalanced tests stalled even with
FSAI; the balanced system is essential for this restarted GMRES comparison.

| Preconditioner on balanced system | Iterations | Median total, 1 RHS (s) | Original-system relative residual | Relative mass-norm error in u |
| --- | ---: | ---: | ---: | ---: |
| ShiftedBlock + FSAI width 8 | 5,952 | 5.050 | 5.92e-9 | 7.36e-11 |
| ShiftedBlock + FSAI width 48 | 2,139 | 2.250 | 4.25e-9 | 6.17e-11 |

Three warmed repeats; the last column uses the existing independently
refined smoothing reference, only for verification. The solver's residual
is measured in balanced coordinates; the table recomputes the residual in
the **original** coordinates. These norms need not be equal.
[Repeats](indefinite-hard-repeats.json),
[balanced comparison including baselines](indefinite-balanced.json),
[unbalanced comparison](indefinite-biharmonic.json).

The non-FSAI alternative, ShiftedBlock with scalar ILU(0), successfully
solved the alpha=1 nonlinear case in about 6.43 s with eight triangular
sweeps. It did **not** solve the alpha=1e-4 case within 12,400 iterations.
Direct block ILU and scalar ILU of H are different preconditioners; the
latter was the more useful of the two here. Neither beat FSAI.

## Mixed triharmonic: improvement, not a validated solve

On the original alpha=1 first coordinate RHS, at 3,100 GMRES iterations
and restart 31:

| Preconditioner | Relative residual | Converged? |
| --- | ---: | --- |
| Block Jacobi | 4.81e-2 | No |
| ShiftedBlock + FSAI width 8 | 8.00e-4 | No |
| ShiftedBlock + FSAI width 48 | 4.00e-4 | No |

Increasing the restart to 93 and running 6,231 iterations reduces the
width-48 residual only to 2.22e-4, with backward error 2.58e-2. The shifted
ILU alternative also fails. [Longer-restart trials](indefinite-triharmonic-long.json).
No times in this table are presented as times to convergence, and there is
no claim of a correct triharmonic solution. Near-resonant-mode treatment or
a stronger indefinite factorization remains needed.

## Verification and reproduction

A run passes only if independently recomputed original-system relative
residual **and** componentwise backward error are below 1e-8. Manufactured
runs additionally require relative forward error below 1e-6. For these runs,
all three/two fields are known analytic vectors, and the RHS is formed from
the original matrix; no reference factorization is needed.

The manufactured biharmonic cases pass (forward errors 1.23e-12 at alpha=1
and 2.08e-8 at alpha=1e-4). Both triharmonic cases fail; the hard case's
forward error is about 3.14, confirming that a partially reduced residual
is not a solution. [Manufactured checks](indefinite-manufactured.json).
Tests also compare block factors with independent sequential ILU, check
full-pattern inverses, fixed-sweep linearity, CPU/CUDA, float32/float64,
small-system GMRES solutions, aliasing, graph capture, exact block signs,
and congruence transformations.

```python
import warp as wp
from warp_preconditioners import MixedHarmonicSystem, ShiftedBlock
from warp.optim.linear import gmres

# K is positive scalar BSR stiffness; mass and b0 are scalar Warp arrays.
# b0 is the original first-block RHS, e.g. alpha*M*f.
system = MixedHarmonicSystem(K, mass, order=2, data_weight=1e-4, equilibrate=True)
P = ShiftedBlock(system, max_row_size=48, kap_tolerance=.003,
                 apply_lanes=4, factor_dtype=wp.float32)
b = system.rhs(b0)
x = wp.zeros_like(b)
gmres(system.matrix, b, x, M=P, restart=31, maxiter=12400,
      tol=1e-10, check_every=31)
u = system.solution(x)
original_full_solution = system.transform(x)
```

The independent block preconditioners also accept a BSR matrix directly:
`BlockJacobi(A)` or `BlockILU0(A, factor_sweeps=20, solve_sweeps=8)`.
Choose blocks that group the coupled variables; scalar ILU cannot use an
unshifted zero multiplier pivot. Check the original residual explicitly.

With the existing dump files and smoothing input data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/indefinite.py \
  --orders 2 --weights 1 --rhs coordinates --rhs-count 3 \
  --methods shift8 shift48 --maxiter 3100 --repeats 3 \
  --output results/indefinite-original-repeats.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/indefinite.py \
  --orders 2 --weights 0.0001 --rhs nonlinear --equilibrate \
  --methods shift8 shift48 --maxiter 12400 --repeats 3 \
  --output results/indefinite-hard-repeats.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/indefinite.py \
  --orders 2 3 --weights 1 0.0001 --rhs manufactured --equilibrate \
  --methods shift48 --maxiter 6200 --output results/indefinite-manufactured.json
python benchmarks/plot_indefinite.py
```

Use `--methods jacobi block-jacobi ilu4 ilu8 ilu32 schur8 schur48 shift-ilu8`
to reproduce the other candidates. GMRES can complete the last restart past
its requested budget; actual counts are recorded. The figure PNG uses Git LFS.
