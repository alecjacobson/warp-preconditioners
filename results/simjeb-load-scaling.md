# Displacement visualization and load scaling

Increasing the pin force from 1× to 10× **does not separate the preconditioners
further** in the current linear-elasticity model. Every tested load reaches
the same sampled iteration count for each method. FSAI remains fastest, with
roughly the same advantage over block Jacobi.

![Actual displacement at 1x and 10x load, with unloaded outlines](../assets/simjeb-displacement.png)

The figure shows actual displacement, without display amplification, at 1×
and 10× force on the fTetWild mesh. Gray curves mark the unloaded silhouette.
Colors show **displacement magnitude in millimeters**. Each row has its own
physical color range, shared by all four methods in that row; the ranges are
approximately 0–0.821 mm and 0–8.214 mm. The three iterative snapshots are
frozen near the fastest method's measured time to target for that load.
Their labels distinguish time to target from the displayed snapshot's time
and show whether both accuracy requirements are met.

## Measured results

All solves use the 351,711-tet fTetWild mesh, the original fixed bolt regions
and pin-force distribution, and the previously selected configurations:
scalar Jacobi-CG, block Jacobi-CG, and FSAI-CG with width 4, κ=.003, one lane
and float32 factors. Solver arithmetic and the matrix are float64. Holding
these settings fixed isolates the effect of force magnitude; this is not a
new FSAI parameter search.

| Pin load multiplier | Total force | Max reference displacement | Jacobi total | Block Jacobi total | FSAI total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1× | 35.586 kN | 0.8214 mm | 0.16919 s | 0.15478 s | **0.13101 s** |
| 2× | 71.172 kN | 1.6428 mm | 0.17042 s | 0.15275 s | **0.13175 s** |
| 5× | 177.929 kN | 4.1069 mm | 0.16897 s | 0.15392 s | **0.13250 s** |
| 10× | 355.858 kN | 8.2138 mm | 0.17013 s | 0.15417 s | **0.13167 s** |

Iteration counts at **every** load are **2,059 / 1,834 / 1,182**, respectively.
The stopping criterion requires both relative lumped-mass displacement error
and relative energy error ≤1e-4 against a tightly converged reference. We
sample every four iterations near the crossing and validate the chosen
stop with an uninstrumented solve. These sampled counts are four iterations
below the earlier remeshing report, which used a different sampling grid.
They are not claims of exact first-crossing iteration counts.

![Relative error curves and setup-plus-solve time as force increases](../assets/simjeb-load-scaling.png)

Times are medians of **five randomly interleaved repeats** of each method/load
pair on NVIDIA L40. Every repeat builds a fresh preconditioner and includes
solver graph capture. Assembly, uploads, allocation of the solution vector,
JIT compilation, reference computation, meshing and CPU diagnostics are
excluded. The plot shows min–max timing whiskers. FSAI's median is
0.1310–0.1325 seconds across all loads; block Jacobi/FSAI speedup stays around
1.16–1.18×. Timing variation does not establish a load-dependent trend.

## Why the convergence stays the same

The system is `K u = f`. In this small-strain model, the stiffness `K` depends
on the reference geometry and material, not on the solved displacement.
Multiplying the load by `s` gives `K (s u) = s f`. It leaves `K`, its condition
number, and all three preconditioners unchanged. With a zero initial guess,
exact-arithmetic preconditioned CG iterates scale by the same factor.
Relative displacement, energy and residual errors therefore follow the same
curves. Floating-point arithmetic can produce small differences, but the
measured crossing counts agree here.

A fixed **absolute** error threshold would instead become relatively tighter
as the load grows. This experiment keeps relative accuracy fixed so that
larger units alone do not create an apparent conditioning challenge.

The 10× image is the prediction of the same **small-strain linear model**.
At larger displacements this model omits geometric changes, plasticity and
contact. To test whether increasing deformation changes preconditioner
performance, a useful next experiment is finite-deformation elasticity with
load stepping and a displacement-dependent tangent stiffness. Its fixed
sparsity would also exercise cached FSAI numerical updates between steps.

## Verification and reproduction

- A hash of the assembled stiffness values is identical before and after the
  sweep; one matrix is used for all loads.
- Each load receives a separate pure-Warp tightly converged FSAI-CG reference
  and block-Jacobi-CG residual correction. Independently evaluated element
  forces give relative residuals below **9.89e-12**.
- Dividing the solved reference displacement by the force multiplier recovers
  the baseline field to **1.20e-13 relative error** or better.
- All 60 timed solves pass both accuracy checks. Renderer hashes match the
  saved physical fields, and constrained displacements are exactly zero.
- CPU NumPy/SciPy calculations are independent diagnostics; numerical solves
  and preconditioning run in Warp on the GPU.

Start with the [fTetWild experiment](simjeb-mesh-comparison.md) and its ignored
mesh/reference files, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/simjeb_load_scaling.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/plot_simjeb_load_scaling.py
# Blender 4.5 with OptiX; run after solver timings finish:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 blender -b --factory-startup \
  --python visualization/render_simjeb_displacement.py
python visualization/compose_simjeb_displacement.py
```

[Raw measurements, all repeats and convergence samples](simjeb-load-scaling.json).
Saved fields and intermediate renders are in ignored
`data/simjeb-ftetwild/load-scaling/`. PNG figures are tracked with Git LFS.
