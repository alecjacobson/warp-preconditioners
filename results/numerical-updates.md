# Fixed sparsity with changing stiffness values

The preconditioners now separate numerical updates from symbolic structure.
All construction, updates, applications, and numerical diagnostics use Warp;
no SciPy solver, sparse factorization, multigrid, or host matrix staging is
used. The [SimJEB bracket](elasticity.md) supplies the test matrix.

## What changed

- **Block Jacobi:** read and invert diagonal blocks directly. No full BSR copy,
  entry-to-row map, or ILU initialization. `update(A)` reuses allocations and
  recomputes only diagonal inverses. It accepts different off-diagonal topology
  as long as the matrix remains canonical and shape/type/device agree.
- **Initial FSAI setup:** borrow compact scalar CSR during construction,
  expand compact BSR with a direct Warp kernel, and pack unique short factor
  supports directly into CSR. Adaptive selection and the resulting Gram
  preconditioner are unchanged. Source `nnz` is settled before allocation,
  avoiding oversized temporaries from assembly's conservative count bound.
- **FSAI refits:** an opt-in plan snapshots source topology, caches the BSR-to-CSR
  value map and factor-transpose permutation, and retains the chosen supports.
  An update gathers new values, equilibrates, solves each small local SPD
  system, validates, then commits coefficients into the existing G/GT buffers.
  There is no new frontier search, sparse sort, or transpose construction.

Local refits use row-parallel dense Cholesky on the selected supports, with
binary searches in cached CSR topology to obtain entries. We deliberately
avoid a per-row dense entry map requiring O(n × width²) persistent memory.
The transpose always comes from the same rounded factor, retaining `G.T @ G`.

## API and limitations

```python
import warp as wp
from warp.optim.linear import cg
from warp_preconditioners import BlockJacobi, FSAI

P = FSAI(A, max_row_size=8, kap_tolerance=.003,
         factor_dtype=wp.float64, apply_lanes=4, reuse_pattern=True)
D = BlockJacobi(A)

# A_new has the same dimensions, block type, device and compact BSR topology.
P.update(A_new)
D.update(A_new)
x.zero_()  # Or supply a deliberate warm start.
cg(A_new, b, x, M=P, tol=1e-8)

# Optional: four deterministic Rademacher probes, plus synchronization.
defect = P.quality(A_new, probes=4, seed=17)
```

Updating `A.values` in place followed by `P.update()` is also supported.
`reuse_pattern=True` must be requested at construction. It requires compact
BSR; canonicalize padded input with `warp.sparse.bsr_copy` first. Active
source columns and row offsets must match the snapshot exactly; spare array
capacity is ignored. Shape, type, device, or topology changes require a new
plan. `BlockILU0.update` explicitly rejects updates; rebuild its factors.

Updates preserve factor buffer addresses, so a captured preconditioner
application sees the new coefficients. A captured **whole solve** also needs
its operator to read the updated matrix: changing Python's `A` binding does
not redirect a previously captured graph. Update the captured matrix's value
buffer in place or recapture the operator/solve as appropriate.

Validation failures leave the old factors intact. That does **not** make
an invalid changed input matrix usable; restore or correct the input before
solving. Setup and updates synchronize for validation and are not capture
operations. Applications remain capturable. Do not overlap update/application
or apply one instance concurrently on independent streams.

A fixed-pattern refit need not match a fresh adaptive factor: changing
coefficients may change which supports would be selected. Rebuild explicitly
when the cached support loses effectiveness. Matrix SPD remains the caller's
precondition; passing local Cholesky checks is not a global SPD proof.

## Experiment

Ten matrices have the same tetrahedra, constraints, load, and BSR pattern.
The Young's modulus multiplier at a tet centroid is

```
exp(a * sin(2*pi*x) * cos(pi*y) * sin(pi*z))
a = [0, .1, .2, .3, .4, .5, 1, 2, 3, 4]
```

Here x/y/z are normalized coordinates in the mesh bounding box. This changes
relative local stiffnesses, rather than uniformly scaling the whole matrix.
At the strongest change, multipliers range from **0.0424 to 11.136**, a
**263× spatial contrast**. This is a numerical stress test, not a claim that
the original titanium bracket has those physical material properties.

We compare a fresh adaptive FSAI build, a refit of the first pattern, a stale
unchanged FSAI, and updated block Jacobi. All use zero-start Warp CG with
relative recursive tolerance **1e-8**, with independently recomputed true
residuals checked below **1e-7** and solution differences checked against the
fresh solve. This is a stricter, different stopping target from the single
solve visualization's displacement/energy target.

Each solve time is the median of three native runs. Per-matrix setup/update
is timed separately. Aggregate totals include the initial plan or factor
construction where needed. Matrix assembly, uploads, and JIT are excluded.
The initial-setup microbenchmark uses five warmed, interleaved repetitions
against code loaded from commit `164d848`, settling source counts consistently
for both versions. It verifies identical initial factor topology and values.
We test widths 4, 8, and 16 at κ=.003, float64 factor storage and four product
lanes; this is a controlled width study, not an exhaustive tuning claim.

## Measurements on L40

| FSAI width | Median fresh setup | Median numerical refit | Ten fresh builds + solves | Initial plan + ten refits + solves |
| --- | ---: | ---: | ---: | ---: |
| 4 | 5.34 ms | 0.721 ms | 1.150 s | 1.106 s |
| 8 | 25.45 ms | 1.006 ms | 1.168 s | **0.950 s** |
| 16 | 168.85 ms | 2.395 ms | 2.531 s | 1.033 s |

Width-eight refits are **25× cheaper than fresh numerical setup**, with
**1.23× lower total time** than rebuilding the same width for this sequence.
They are **1.54× faster overall than updated block Jacobi** (1.460 s).
Width 16 benefits more from eliminating adaptive searches (**2.45× overall
versus rebuilding width 16**), but its initial plan still makes width 8 the
best of these three for ten matrices. Width 4 is best for the single-solve
visualization; amortization changes the preferred width.

At width 8, checking quality after every refit costs a median **1.92 ms** and
raises the total from **0.950 to 0.969 s**. An individual block-Jacobi update
costs about **0.16–0.20 ms**, but its solves need more iterations.

At the final 263× material contrast, fresh/refitted/stale width-eight factors
require **1,321 / 1,321 / 3,667 iterations**. Updated block Jacobi needs 2,770.
Across all three width experiments, solutions agree with fresh-build solves
to **1.11e-10 relative or better**. Refit and fresh iteration counts can differ
slightly even when factors represent the same local solves, because their
Cholesky order and floating-point rounding differ.

![Setup, iteration counts, and cumulative time for changing material values](../assets/simjeb-numerical-updates.png)

The initial-setup microbenchmark also compares the previous implementation:

| Constructor | Previous setup | New setup |
| --- | ---: | ---: |
| Block Jacobi (width-eight experiment) | 1.272 ms | **0.274 ms** |
| FSAI from BSR, width 4 | 12.480 ms | **4.940 ms** |
| FSAI from BSR, width 8 | 30.693 ms | **25.450 ms** |
| FSAI from BSR, width 16 | 172.827 ms | **166.564 ms** |
| FSAI from scalar CSR, width 4 | 9.098 ms | **3.744 ms** |

Setup savings matter most for small factors; at wider supports the adaptive
search dominates. These are warm five-trial medians, with raw samples retained;
they do not include JIT, assembly or upload. Each initial factor matches the
previous implementation's topology and values within the recorded tolerance.


## Quality monitoring

`quality()` estimates `||G A G.T - I||_F / sqrt(n)` using fixed Rademacher
probes. The same seed allows direct comparisons between stale, fresh, and
refitted factors. It reports a **heuristic**, not a condition number, residual
bound, or reliable predictor of Krylov iterations. A few poorly conditioned
modes can be hidden by this Frobenius average. Always monitor actual solve
behavior as well; there is no automatic rebuild threshold in the library.

The experiments separately time this diagnostic, including allocations and
host synchronization, and report totals both with and without checking on
every update. A practical application may check less frequently. In this
sequence, unchanged numerical factors deteriorate while refitted supports
stay effective even for the strong changes. At width 8, the final defect is
**1.511** for stale factors versus **0.347** after refitting. This does not establish that
all changing matrices can safely retain the original supports.

## Verification and reproduction

Tests compare refits against independent dense solves on each selected
support, across CPU/CUDA, float32/float64 source matrices, scalar/3×3 BSR,
and float32 factor storage. They cover invalid updates, topology changes,
in-place changes, transpose consistency, exact full-support inverses, quality
on exact factors, and applications captured before a numerical update.
Separate elasticity tests check changing element moduli by superposition.

```bash
OPENBLAS_NUM_THREADS=1 python benchmarks/simjeb_problem.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/numerical_updates.py --width 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/numerical_updates.py --width 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/numerical_updates.py --width 16
python benchmarks/plot_numerical_updates.py
pytest -q
```

Raw results: [width 4](numerical-updates-w4.json),
[width 8](numerical-updates-w8.json), [width 16](numerical-updates-w16.json).
