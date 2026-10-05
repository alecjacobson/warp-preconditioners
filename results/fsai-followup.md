# FSAI follow-up: retain measured improvements

This study tests three general-purpose changes against FSAI from commit
`8f25bf7`: factor application, adaptive construction, and block-aware FSAI.
The supported library gains only an optional `max_step_size` argument and
its shared construction implementation. Single-entry growth remains the
default. The application and block prototypes remain under `benchmarks/`;
they are not exported library APIs.

## Controlled comparisons

All numerical construction, applications, and CG solves use pure Warp on
NVIDIA L40, with float64 equations/vectors and float32 completed factors.
There is no multigrid, external sparse factorization, or SciPy numerical
backend in any preconditioner. CPU NumPy/SciPy load the archived dragon
operator and compute independent errors outside timing. Elasticity assembly
uses Warp and the existing SimJEB boundary conditions and loads.

The baseline constructor is loaded from its pinned Git source. The matrix,
operator, RHS, initial guess, storage precision, and solver are shared with
its candidates in each process. JIT warmup is excluded. Timings include
preconditioner construction, solver graph capture, and the uninterrupted
solve, with synchronization around each interval. Common operator assembly,
input transfer, diagnostics, and downloads are excluded. Each final trial
also has an untimed native solve immediately beforehand. Trials alternate
forward and reverse order; GPU clocks are not locked. The final runner rejects
other GPU compute processes at startup and before/after each timed trial.
Runs that overlapped unrelated rendering were excluded and repeated. Short solves still
show wall-time outliers, so compare paired experiments rather than absolute
times from separate runs.

The dragon accuracy target is relative lumped-mass field error <= 1e-8
against the saved refined reference. SimJEB requires both relative
lumped-mass displacement and energy errors <= 1e-4. Selection samples every
25 iterations for the dragon and every 10 for elasticity, preserving Krylov
state. Timed native endpoints independently pass the same criteria.
Recursive residuals are not the acceptance criterion; fresh residuals are
also recorded in final trials. At the elasticity field/energy target, the
force residual can still be around 1e-3; this is not a claim of a 1e-4
relative force residual.

## Adaptive construction: accepted

`FSAI(A, max_step_size=2, ...)` selects up to two distinct largest-residual
candidates per frontier search, borders the local Cholesky factor for each,
and solves the enlarged local problem once per batch. The forward solve
retains its existing entries when bordering. Already-selected top candidates
are not rescored if encountered again in the frontier. All candidates remain
eligible; no bounded hash table truncates the search.

`max_row_size` still caps the total support, including the diagonal. The
energy-improvement stopping test runs after each batch. Thus changing batch
size changes the support and the interpretation of the same `kap_tolerance`;
it is an explicit tuning choice, not a universal replacement default. This
follows the multiple-additions-per-step approach supported by
[hypre FSAI](https://hypre.readthedocs.io/en/latest/solvers-fsai.html).

| Dragon configuration | Iterations | Setup (s) | Solve (s) | Total (s), range |
| --- | ---: | ---: | ---: | ---: |
| Pinned baseline, width 48 | 33,125 | 1.055 | 9.132 | **10.187** (9.966–10.215) |
| Width 48, batch two | 27,200 | 0.634 | 8.822 | **9.456** (9.325–9.545) |

Five isolated, interleaved repeats give **7.2% lower median
setup-plus-solve time**. Batch two is faster in all five paired rounds. Its
field error is 6.84e-09, versus
9.32e-09 for the baseline; both pass the same 1e-8 target.
Columns are separate medians; total uses paired setup-plus-solve samples.
[Final isolated measurements](fsai-final-dragon.json).

```python
P = FSAI(A, max_row_size=48, max_step_size=2, kap_tolerance=.003,
         apply_lanes=4, factor_dtype=wp.float32)
```

The elasticity checks show no material advantage from batching at width four:

| Mesh | Baseline total (s) | Current default (s) | Batch two (s) |
| --- | ---: | ---: | ---: |
| Original SimJEB | 0.08140 | 0.08117 | 0.08078 |
| fTetWild SimJEB | 0.13127 | 0.13105 | 0.13024 |

Differences are below 1%; the elasticity recipe stays at batch one. All trials
pass the same field and energy targets. The default factors are bit-identical
to baseline. [Original mesh](fsai-batch-original-elasticity.json),
[fTetWild mesh](fsai-batch-elasticity.json).

The initial prototype five-repeat sweep tested batches 1, 2, 3, 4, and 6 at width 48
and kappa=.003. Batch two was fastest. Larger batches reduced iteration
counts further, but denser factors increased solve cost: batch six needed
25,000 iterations and took 10.57 s median total, versus 27,200 iterations
and 9.57 s for batch two. The baseline in that sweep took 10.39 s.
[Full exploratory sweep](fsai-batch-dragon.json). That prototype capped search
rounds; regression review changed the final implementation to continue through
short frontiers until the actual support cap or stopping criterion. The final
comparison above was rerun after that fix.

The default (`max_step_size=1`) is checked for bit-identical factor offsets,
columns, and coefficients against the pinned implementation on the real
dragon and fTetWild elasticity matrices, including scalar support widths
4, 8, and 16 on elasticity.
[Dragon](fsai-default-dragon.json), [width 4](fsai-default4-elasticity.json),
[width 8](fsai-default8-elasticity.json), [width 16](fsai-default16-elasticity.json).
The default dragon setup fell from 1.043 to 0.875 s; application is unchanged.
Independently implemented small-matrix greedy
selection tests cover multiple batch sizes, stored zeros, scaling, both
precisions, and CPU/CUDA. Full-support inverses, partial batches, guarded
pivots, aliasing, captures, and fixed-pattern numerical refits are checked.

A separate attempt to deduplicate *every* frontier candidate using prior-row
binary searches was rejected: median dragon setup increased from about
1.05 to 1.21 s despite identical factors. Retaining the forward solve alone
was effectively tied. Those ablations are reproducible with
`--stage setup_ablation`; their exploratory data are
[deduplication](fsai-setup-dedup-dragon.json) and
[forward-solve reuse](fsai-setup-forward-dragon.json).

## Factor application: rejected

The existing four-lane dragon and one-lane elasticity recipes remained the
best within variation when testing all 25 combinations of 1, 2, 4, 8, and
16 lanes independently for G and its transpose. These screens used 2,000
iterations with unchanged factors; they measure iteration cost, not a
converged solution. No separate-lane public option was added.
[Dragon](fsai-lanes-dragon.json), [elasticity](fsai-lanes-elasticity.json).

A fused kernel that recomputes Gx inside the transpose product preserves the
Gram operation without explicitly multiplying the factors. It avoids the
intermediate-vector launch but repeats work. On width-four fTetWild
elasticity, its isolated median total was 0.1737 s versus 0.1307 s for the baseline,
with the same iterations and field errors. It was rejected.
[Measurements](fsai-fused-elasticity.json).

## Block FSAI: tested on elasticity, not promoted

The prototype grows whole 3x3 displacement blocks using block residual
scores, solves the selected local principal system for three RHSs, then
normalizes its block row to satisfy G_i A G_i^T = I. It preserves G^T G,
uses only the supplied BSR structure and entries, and has no geometry or
boundary-condition heuristic. Widths one through eight blocks were tested;
width one reproduces block Jacobi and full support reproduces the dense
inverse in independent tests.

Whole-block supports did not win overall. The initial scalar-storage sweep
reduced iterations from 1,180 (scalar FSAI) to 950 at eight blocks, but total
time rose from 0.1295 to 0.6035 s. We also tested cooperative scalar products
and native BSR factors with float64 accumulation. Native BSR reduced product
cost, but the best two-block case did not produce a repeatable total-time
win over scalar FSAI. In the final isolated comparison, native BSR took
**0.1609 s versus 0.1310 s** for scalar FSAI, about 23% longer.
This rejects this implementation/settings, not every possible block FSAI.
[Support sweep](fsai-block-elasticity.json),
[application variants](fsai-block-apply-elasticity.json),
[final paired comparison](fsai-block-final-elasticity.json).

## Reproduce

The existing `data/dirichlet-current`, `data/simjeb`, and
`data/simjeb-ftetwild` benchmark inputs and references are required. The
archived dragon Laplacian is loaded directly after checking that its input
stamp matches the fields; the obsolete `/tmp/dump` export is not required.
The pinned baseline commit must be available in the Git clone.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python benchmarks/fsai_experiments.py --stage batch --steps 2 --problem dragon --output results/fsai-final-dragon.json
python benchmarks/fsai_experiments.py --stage batch --steps 1 2 --problem simjeb --output results/fsai-batch-original-elasticity.json
python benchmarks/fsai_experiments.py --stage block_apply --block-widths 2 --problem simjeb-ftetwild --output results/fsai-block-final-elasticity.json
python benchmarks/fsai_experiments.py --stage lanes --problem dragon --output results/fsai-lanes-dragon.json
python benchmarks/fsai_experiments.py --stage fused --problem simjeb-ftetwild --output results/fsai-fused-elasticity.json
python benchmarks/fsai_experiments.py --stage setup --width 16 --problem simjeb-ftetwild --output results/fsai-default16-elasticity.json
pytest -q
```

Validation: **188 tests passed** on CPU and CUDA (`pytest -q`), including
both floating-point precisions, independent dense references, graph captures,
aliasing, numerical refits and rollback, and narrow-frontier batch growth.
Ruff and `git diff --check` also pass.
