# Measurements, 2026-09-29

NVIDIA L40, dual Intel Xeon Platinum 8362, Warp 1.15.0, float64.
The dragon has 360,757 vertices; its biharmonic matrix has 7,356,259 stored
entries. All three coordinate right-hand sides are solved from zero.

The recommended configuration is `FSAI(Q, max_row_size=8)` with CG. The
final implementation's three final trials give **0.708 s median total**
(approximately **0.031 s setup** and **0.676 s solve**), at worst componentwise backward error
**8.83e-11**. Each column takes 500 iterations with fixed chunks of 500.
The factor has 2,396,149 entries (6.64 per row on average), and no rows
were stopped by the pivot safeguard. Repeat totals range from 0.701 s to
0.717 s on this shared server.

## Fixed 500-iteration chunks

| Method | Setup (s) | All 3 solves (s) | Worst backward error | Reached 1e-8? |
|---|---:|---:|---:|---|
| Jacobi-CG | 0.00035 | 105.18 | 5.22e-5 | No, time limit |
| Jacobi-CR | 0.00037 | 105.16 | 4.77e-8 | No, time limit |
| FSAI-CG, width 4 | 0.013 | 1.232 | 5.22e-12 | Yes |
| FSAI-CG, width 8, median of final repeats | 0.031 | 0.676 | 8.83e-11 | Yes |
| FSAI-CG, width 16 | 0.161 | 0.747 | 2.90e-15 | Yes |
| FSAI-CG, width 32 | 0.963 | 0.773 | 3.03e-15 | Yes |
| FSAI-CR, width 8, initial implementation | 0.030 | 0.656 | 8.54e-12 | Yes |
| Lifted FSAI-GMRES, width 8, initial implementation | 0.016 | 1.739 | 3.87e-10 | Yes |

The Jacobi rows are budgeted single runs, capped at 35 seconds per column;
they are not times to convergence. They used the same fixed restart
schedule as the FSAI rows. `dragon-width8.json` contains the initial
comparison; `dragon-width8-final*.json` contains the final repeats. The library
removes an extra output combination kernel; the runner computes diagnostic
norms with NumPy sums to avoid launching BLAS thread teams between timed
GPU solves. Earlier `dragon-width8-repeat*.json` trials used BLAS-backed
norm reporting and are retained as intermediate measurements. Their
numerical answers agree. Timings include host residual checks and graph-capture
costs after warmup; host scheduling variability remains visible.

An initial lifted BiCGSTAB experiment diverged (nonfinite result on the
first RHS after 2,179 iterations); see `explore-width8.json`. Use GMRES
for that optional formulation. Direct FSAI-CG was the better choice here.

## Upstream adaptive schedule

`dragon-upstream-width8.json` repeats the successful methods using the
upstream adaptive schedule (10 initial iterations, chunks targeting five
seconds, capped at 5,000). All three methods succeed. FSAI-CG takes
**0.038 s setup + 1.260 s solve**, with worst backward error **3.91e-15**
and 759–763 iterations per column. CR takes 3.811 s solve and lifted GMRES
7.532 s. These longer times reflect over-solving within a chunk, solver
capture costs, and in GMRES's case different restart behavior for a small
initial iteration budget. Do not substitute fixed-chunk timings into the
original leaderboard without stating the schedule change.

For context, the [published leaderboard](https://github.com/alecjacobson/sparse-solver-benchmark/blob/alecjacobson/refresh/leaderboards/l40-2x-xeon-platinum-8362.md)
reports 85 s for Warp-CR, failed Warp-CG, and 2.45 s total for CHOLMOD on
the biharmonic problem. Those are historical measurements, not direct
solvers rerun in this experiment. This work does not claim equal forward
accuracy to a direct factorization. The dragon's normwise residual can
remain around 1e-7 even at tiny componentwise backward error because of
strongly varying row scales and cancellation; both metrics are recorded.

## Grid check

Synthetic Dirichlet grids use `Q=h^2 I + K^2/h^2`, with known random exact
solutions and a normwise CG tolerance of `1e-11`. No Krylov restarts are
introduced by the convergence checks.

| Grid | Jacobi iterations | FSAI width 8 iterations | Jacobi total (s) | FSAI total (s) |
|---|---:|---:|---:|---:|
| 32×32 | 600 | 175 | 0.0150 | 0.0096 |
| 64×64 | 2,175 | 600 | 0.0480 | 0.0208 |

All tested configurations have normwise residual below 1e-11 and relative
forward error below 4e-8. See `grid32.json` and `grid64.json`. FSAI improves
convergence here too, but is not mesh-independent.

## Input provenance and reproduction

The source benchmark checkout inspected was
`193812a89fba7d881603fa8f78ef1883912e847c` on `alecjacobson/refresh`.
Existing MatrixMarket dumps from its C++ benchmark were used; the runner
independently checks `Q=M+K M^-1 K` and `H=M+K` against the dumped mixed
system. No changes were made to the upstream repository or leaderboard.

SHA-256:

```
xyzrgb_dragon-720K.ply
36de4549989dd359bfb9d703cd7431da30b24b223db3c1027c5ac4551b4e51d0
k2_Q.mtx
a45d5ddf4db0fa063c4488b9de405a7cd92b5e08bd91a925cd59cc7e1f1eccce
k2_rhs.mtx
a20d64ff757d6be75fca900ccf2e610b4299fd20f356e74a514a5eb7a902b52e
```

Run from the repository root after generating dumps as described in the
main README:

```bash
python benchmarks/dragon.py --dir /tmp/dump --width 8 \
  --methods fsai-cg --schedule fixed --chunk 500 \
  --output results/repeat.json
python benchmarks/dragon.py --dir /tmp/dump --width 8 \
  --methods fsai-cg fsai-cr lifted-gmres --schedule upstream \
  --output results/upstream.json
python benchmarks/synthetic.py --size 32 --output results/grid32.json
python benchmarks/synthetic.py --size 64 --output results/grid64.json
```

Library validation: 14 passing tests, including CPU/CUDA, float32/float64,
full-pattern inverse agreement, independently solved sparse supports,
BSR vector integration, alpha/beta aliasing, CUDA graph replay, guarded
pivots, and equivalence of the lifted biharmonic system.

## Other orders

`dragon-harmonic.json` records a fixed-chunk harmonic check: Jacobi-CG
uses 1,000 iterations per column, versus 76 for width-eight FSAI-CG.
Measured totals are 0.513 s and 0.362 s respectively, with both reaching
the target. The small solve times are particularly sensitive to host
scheduling overhead, so the iteration reduction is the clearer finding.

`dragon-triharmonic.json` records a 15-second-per-column check of
width-eight FSAI-CG and FSAI-CR. Neither consistently reaches the 1e-8
backward-error target on all columns (CR reaches it on one). CR's worst
backward error is 6.01e-8 after 36.8 seconds total solve, but its normwise
residuals remain about 0.96–1.58. This is not a validated high-accuracy
triharmonic solution; stronger preconditioning or a different formulation
remains necessary there. No claim is made for the indefinite mixed
triharmonic system.
