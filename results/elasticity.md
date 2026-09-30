# SimJEB bracket elasticity

The current volume benchmark is **SimJEB design #225**, Michael Jenkins's
[Improved GE Bracket](https://grabcad.com/library/improved-ge-bracket-1), from
[SimJEB](https://simjeb.github.io/) by Eamon Whalen, Azariah Beyene and Caitlin
Mueller. It is a representative bracket with four mounting holes and two pin
lugs, rather than a claim about the competition's best design.

We import the original **215,219 first-order tetrahedra**, 44,121 physical
vertices, material, fixed bolt-hole surfaces, and vertical pin load from its
OptiStruct/Nastran `.fem` deck. After eliminating 428 fixed vertices, there
are **131,079 free displacement DOFs** in a 3×3 BSR matrix. No remeshing,
penalty constraints, multigrid, or external factorization is involved.
The earlier [dragon experiment](archive/dragon-elasticity.md) is archived;
its code and reproduction commands correspond to commit `164d848`.

## Loads and material

The original deck uses mm, N, MPa; the importer converts to m, N, Pa.
Young's modulus is **113.8 GPa**, Poisson's ratio **0.342**. The vertical
load is **35,585.77 N** in +Z. This benchmark uses that pin load, not gravity.
All translations on each RBE2 bolt-hole surface are fixed, equivalent to
fixing its rigid spider's six center DOFs. There are 753 load-interface nodes.

For the equal-weight RBE3 load interface, we distribute the reference force
by the transpose of a rigid-motion least-squares fit. With centered node
positions `r`, the nodal force is `F/n + omega × r`, where
`J omega = (x_ref - centroid) × F` and
`J = sum((r·r) I - r rᵀ)`. The small dense inverse and force distribution run
in Warp. Tests check total force, moment, and work under arbitrary rigid
motion. We do not claim to have validated every detail of OptiStruct's RBE3
implementation. The reference maximum displacement is **0.798805994 mm**,
matching **0.798806 mm** in the [official metadata](https://simjeb.github.io/data/all_bracket_metadata.csv).

The original deck is downloaded on demand from a
[pinned public mirror](https://raw.githubusercontent.com/nora-alshareef/jeb-dataset-forge/e571104fc20c7cd34dc9be5b7f038b56fbef4981/data/raw/225.fem),
with SHA-256 verification. Attribution and provenance are preserved in
`mesh.json` and the results. SimJEB uses ODC-By; the original design remains
subject to GrabCAD's terms. The geometry is not relicensed as project code.

## Best tested settings

All assembly, preconditioners, reference solves, and compared solves execute
in Warp on NVIDIA L40, with float64 stiffness and solver arithmetic. CPU
NumPy/SciPy calculations independently check the assembled operator and
errors; they never supply a solution or preconditioner. Every solve starts
from zero. The target is **both relative lumped-mass displacement error and
relative energy error ≤ 1e-4**, against the independently checked reference.

| Method | Iterations | Setup | Solve | Total |
| --- | ---: | ---: | ---: | ---: |
| Scalar Jacobi + CG | 2,035 | 0.00039 s | 0.11187 s | 0.11226 s |
| Block Jacobi + CG | 1,846 | 0.00032 s | 0.10342 s | 0.10384 s |
| FSAI width 4, κ=.003, float32 factor, one lane + CG | 1,132 | 0.00565 s | 0.07588 s | **0.08198 s** |

Each timing column is a median; the total uses paired setup-plus-solve trials,
so it need not equal the sum of the other two medians.

FSAI is **1.27× faster than block Jacobi**, **1.37× faster than scalar Jacobi**
including setup. These modest gains are specific to this mesh and target.
Small FSAI patterns win single solves; wider patterns lower iteration counts
but cost more to construct. See the separate [numerical-update experiment](numerical-updates.md)
for repeated solves with changing stiffness.

The 48-configuration sweep includes CG and CR, all three PR block algorithms and ours,
FSAI widths 2–48, nearby stopping tolerances, product lane counts, and factor
precision. Raw settings and measurements are in [elasticity.json](elasticity.json).
Leading candidates within 12% of each family's initial fastest total are
repeated after refining their accuracy crossing (at most six candidates).
Finalists receive three further native setup/solve trials. Timings include
solver graph capture and validation synchronizations, but exclude JIT,
assembly, mesh I/O, transfers, and independent error diagnostics.

### Warp PR #1890 comparison

[`warp_pr1890.py`](../benchmarks/warp_pr1890.py) contains the actual direct-QR,
sequential-LDLT, and tile-Cholesky functions from
[PR #1890](https://github.com/NVIDIA/warp/pull/1890), pinned at
`0b58bcf2320b77e548bbbac0292f7f4415455eec`, with compatibility imports and
its original license retained. The PR's auto strategy selects direct for
3×3 blocks. Our block Jacobi computes the same mathematical inverse; its
application differs from PR direct by **2.39e-16 relative** on this problem.

After removing the matrix copy and ILU topology work from our constructor,
our block Jacobi and PR direct have effectively tied solve totals. Our
implementation narrowly won the repeated selection (0.10300 vs 0.10328 s);
that difference is too small to call a meaningful speedup. The final table
uses the selected implementation. Unlike the PR snapshot, ours validates
singular/nonfinite pivots and provides transactional numerical updates.

## Curves and frozen iterates

![Error versus iterations and wall clock](../assets/simjeb-elasticity-convergence.png)

Solid curves measure lumped-mass displacement error; dashed curves measure
energy error. Both are relative to the reference. Iteration histories preserve
Krylov state. Wall-clock curves use native, uninstrumented prefix solves
from zero; independent diagnostics are excluded from timing. This is an
error plot, not a recursive-residual stopping plot: the residual can remain
around 1e-3 when field and energy errors meet this target.

![Actual displacement colored by von Mises stress](../assets/simjeb-elasticity-comparison.png)

The common budget is **0.08198 s**, including setup. Actual saved snapshot
times are 0.08293, 0.08142, and 0.08160 s. Their relative displacement errors
are **0.0785%**, **0.0479%**, and **0.00176%** respectively. Errors are small
on this problem, so the rendered shapes and stresses should look similar.

Geometry uses the **actual physical displacement, with no amplification**.
All four brackets share camera, white background, lighting, and a linear
**MPa** stress scale with 26 crisp `isolines_stripe_map(okloop(...))` bands.
Stress colors are retained on fixed surface triangles. The reference is shown first.
The shader interpolates the recovered scalar before the discrete lookup.

Linear-tet stress is constant per element. For visualization, Warp
volume-averages element stress tensors at vertices, then computes the von
Mises invariant. This smooths discontinuities; the displayed nodal maximum
is not the peak raw element stress (reference: **1012.56 MPa** nodal,
**1299.15 MPa** element). Affine/hydrostatic tests verify the
invariant. Saved displacements and stress fields are hash checked; Blender
coordinates and scalar attributes are read back and checked before rendering.

## Boundary-condition and stress audit

The imported vertical case was rechecked against the original deck and
[SimJEB's simulation description, §3.4](https://arxiv.org/html/2105.03534).
It uses the same four fixed bolt interfaces and pin-load arrangement described
by the [challenge rules](https://blog.grabcad.com/ge-terms-of-service/).
SimJEB notes that its detailed modeling assumptions may differ from those
used by the challenge or individual designers.

- `SUBCASE 1` selects `SPC=1`, `LOAD=2`.
- Four RBE2 centers (44122–44125) have all six DOFs fixed. Their dependent
  physical nodes are exactly the **428 fixed vertices** in our imported mesh;
  saved displacements are identically zero there.
- `FORCE 2` applies **35,585.77 N in +Z** at node 44126. Its RBE3 connects
  exactly the **753 loaded physical vertices** used here. An independent
  six-DOF least-squares construction checks the force distribution.
- Reactions computed from independent element forces balance the applied
  force within **1e-8 N**, and its moment within **1e-9 N·m**.
- Independent NumPy tensor recovery agrees with the saved Warp nodal von
  Mises field to **3.83e-15 relative error**. This checks indexing, gradients,
  units, averaging and the invariant, without calling the Warp stress kernel.

The recovered peak is **1012.56 MPa at fixed GRID 284**, on the underside
at approximately `(-1.05, -143.08, 0)` mm. All 19 vertices in the highest
0.1% of surface stresses lie within **1.74 mm** of a fixed node; 13 are fixed.
There are also concentrations around the loaded pin interface. Meanwhile,
90% of surface vertices are below **190.41 MPa**, so most of the model uses
the low end of the shared 0–1013 MPa color scale.

The original figure made the concentrations hard to see: the top view hid
the underside, and a gray boundary-condition material replaced stress colors
on 712 constrained surface triangles. That mask has been removed. The new
view below is a rigid display rotation of the same saved fields, with the
same scale and no displacement amplification.

![Underside stress concentrations around the fixed bolt holes](../assets/simjeb-elasticity-underside.png)

The remaining mottling is consistent with stress variation from first-order
tetrahedra and nodal recovery, emphasized by alternating crisp color bands.
Each tet has constant strain/stress; volume-averaging at vertices does not
provide a mesh-independent smooth stress field. The
[SimJEB paper](https://arxiv.org/html/2105.03534) likewise distinguishes robust
displacement prediction from stress accuracy, which can benefit from better
mesh quality, higher-order elements and fillets. We have **not** performed a
stress mesh-convergence study, so these checks do not establish that every
small-scale feature is physically resolved. No additional smoothing or
numerical changes were made to produce the new images.

[Reproducible audit](../benchmarks/audit_simjeb_stress.py),
[raw checks and per-bolt reactions](simjeb-stress-audit.json):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/audit_simjeb_stress.py
blender -b --factory-startup --python visualization/render_elasticity.py -- --underside
python visualization/compose_elasticity.py --underside
```

## Verification and reproduction

Independent element strain/stress contraction agrees with GPU matrix action
to **6.27e-16 relative error**. The pure-Warp FSAI-CG reference receives a
separate block-Jacobi-CG residual correction, changing its displacement by
**1.98e-13 relative**. Independent element forces yield **1.03e-11 relative
residual**. Tests also cover rigid modes, affine strain energy, elimination,
nonuniform material superposition, and load force/moment/work.

```bash
python -m pip install -e '.[elasticity,test]'
OPENBLAS_NUM_THREADS=1 python benchmarks/simjeb_problem.py
# Move existing results/elasticity.json aside for a fresh timing run.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage sweep
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage sweep \
  --configs results/simjeb-nearby-configs.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py --stage final
python benchmarks/plot_elasticity.py
python benchmarks/elasticity_stress.py
# Blender 4.5 with OptiX:
blender -b --factory-startup --python visualization/render_elasticity.py
python visualization/compose_elasticity.py
pytest -q
```

Downloaded mesh and saved solution fields live in ignored `data/simjeb/`.
The committed JSON regenerates the line plot without mesh downloads.
PNG assets are tracked with Git LFS.
