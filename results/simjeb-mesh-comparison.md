# SimJEB #225: original mesh versus fTetWild

fTetWild improves the poor-element tail substantially, while the main stress
concentrations remain at the same bolt holes. The recovered peak occurs at the
same fixed location on both meshes. Maximum displacement changes by 2.83%; the
area-weighted surface stress difference is 7.74%. Mesh quality contributes to
the uncertainty, but this comparison does not support attributing the entire
stress pattern to a few bad tetrahedra.

The baseline uses **SimJEB's original HyperMesh tetrahedra**. TetGen was used
for the earlier dragon example. Both bracket meshes use linear tetrahedral
elasticity, the supplied titanium material, four fixed bolt-hole regions and
the case-1 vertical pin load.

![Converged stress fields and all boundary nodes on the two meshes](../assets/simjeb-mesh-comparison.png)

The first two rows show tightly converged references, actual displacement
without amplification, and one shared 26-band von Mises scale. The underside
is a rigid display rotation. Stress recovery is unchanged: volume-average
element stress tensors at vertices, then evaluate von Mises. There is no
additional field smoothing. The third row shows undeformed transparent
geometry, every fixed node in orange, every loaded node in blue, and the
35,585.77 N resultant in +Z. Loaded nodes remain free to move.

## Mesh quality and geometry

The [fTetWild Python wrapper](https://wildmeshing.github.io/python/),
`wildmeshing==0.4.1`, remeshes the original faceted boundary with a 2 mm target
edge length and 0.02 mm surface envelope. These are converted to fractions
of the bounding-box diagonal for the API. Meshing uses four CPU threads;
measured wall time was 535.47 seconds. This offline geometry operation is
excluded from the GPU solver timings. The wrapper's raw statistics contain
an invalid winding-number timer; the reported runtime comes from Python's
wall clock.

| Quantity | Original SimJEB | fTetWild |
| --- | ---: | ---: |
| Vertices | 44,121 | 68,507 |
| Tetrahedra | 215,219 | 351,711 |
| Minimum dihedral angle | 5.36° | **14.31°** |
| 1st percentile minimum dihedral | 14.12° | **34.99°** |
| Tets below 10° | 0.415% | **0%** |
| Worst mean-ratio quality (1 = regular) | 0.208 | **0.432** |
| Median mean-ratio quality | 0.857 | **0.900** |
| Median edge length | 2.008 mm | 1.932 mm |
| Volume | 277,422.26 mm³ | 277,423.72 mm³ |

The original mesh has a good median quality and a poor tail. fTetWild also
changes element count, size distribution and connectivity, so this is a
combined shape-quality and discretization comparison. It preserves the
original faceted geometry rather than reconstructing CAD surfaces.

![Element quality distributions and matched surface stress values](../assets/simjeb-mesh-quality.png)

## Boundary transfer

Original surface triangles are tagged using the exact deck node sets. Each
new boundary triangle inherits the tag of the closest original triangle at
its centroid; incident vertices form the new fixed and loaded sets. The four
bolt regions retain zero displacement in all three components. Other base
plate vertices remain free. No contact or gravity is added.

| Region | Original nodes | New nodes | Surface-area change |
| --- | ---: | ---: | ---: |
| Bolt 1 | 106 | 106 | −0.0107% |
| Bolt 2 | 109 | 111 | −0.0064% |
| Bolt 3 | 108 | 108 | −0.0130% |
| Bolt 4 | 105 | 105 | −0.0100% |
| Pin | 753 | 724 | −0.0283% |

The maximum sampled surface distance is 0.01454 mm, checking vertices in
both directions and new face centroids against the original. This sampling
is a geometry check, not an exact Hausdorff-distance calculation.

The original pin reference point, force resultant and zero moment about that
point are preserved by the same pure-Warp equal-weight load-distribution
kernel. Force and moment balance pass absolute 1e-8 checks in N and N·m.
Remeshing changes node locations and equal-weight RBE3 sampling, so individual
nodal loads change. This is another source of discretization differences.
The figure makes the transferred node sets inspectable.

## Reference solutions and solver performance

| Quantity | Original SimJEB | fTetWild |
| --- | ---: | ---: |
| Maximum displacement | 0.798806 mm | 0.821377 mm |
| Compliance, fᵀu | 19.5068 N·m | 20.0392 N·m |
| Recovered nodal peak von Mises | 1012.56 MPa | 1035.09 MPa |
| Element peak von Mises | 1299.15 MPa | 1243.10 MPa |
| Independent element-force relative residual | 1.03e-11 | 9.84e-12 |

Both recovered maxima occur at the fixed location `(-1.04908, -143.076, 0)` mm.
On the new mesh, the hottest 0.1% of surface nodes are all within 2.52 mm of
a fixed node. The main underside concentrations persist after remeshing.

Projecting the new recovered fields to the original boundary and using its
lumped surface areas as weights gives **2.87% relative displacement difference**
and **7.74% relative stress difference**. More than 5 mm from fixed nodes,
the stress difference remains **6.70%**. These differences are much larger
than the verified algebraic errors. A mesh-refinement study, and potentially
higher-order elements, would be needed to establish stress convergence.
Two meshes alone do not identify which local stress value is more accurate.

The new mesh was tested with CG and CR for scalar Jacobi, our block Jacobi,
PR #1890's direct block Jacobi, and FSAI widths 4, 8 and 16 at κ=.003. FSAI
uses float32 factors with double-precision solves; width 4 uses one row lane,
widths 8/16 use four. These are the **best tested** settings in this targeted
12-configuration sweep, followed by repeated timing of nearby finalists.

| Best tested on fTetWild | Iterations | Median setup + solve |
| --- | ---: | ---: |
| Scalar Jacobi + CG | 2,063 | 0.16929 s |
| Block Jacobi + CG | 1,838 | 0.15331 s |
| FSAI width 4 + CG | 1,186 | **0.13311 s** |

FSAI remains **1.15× faster than block Jacobi**, including setup. Our block
Jacobi agrees with the PR's direct application to 2.42e-16 relative error.
All assembly, preconditioning and solves run in Warp on the NVIDIA L40.
The target requires both relative lumped-mass displacement and energy error
≤1e-4. Warm timings include setup and solver graph capture; they exclude
meshing, assembly, uploads, JIT compilation and independent diagnostics.
NumPy/SciPy/libigl are used for geometry and independent diagnostics only;
no external sparse solver or factorization supplies the reference.

![Errors versus iterations and wall time on the fTetWild mesh](../assets/simjeb-ftetwild-convergence.png)

## Reproduction

The [original problem report](elasticity.md) contains source attribution,
download provenance and the original-mesh solve commands. Start with its
`data/simjeb/mesh.npz` and saved reference/comparison fields. Move an existing
`results/simjeb-ftetwild.json` aside before a fresh timing run.

```bash
python -m pip install -e '.[remesh]'
python benchmarks/export_simjeb_boundary.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/remesh_simjeb.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/prepare_simjeb_remesh.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py \
  --data data/simjeb-ftetwild --output results/simjeb-ftetwild.json --stage reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py \
  --data data/simjeb-ftetwild --output results/simjeb-ftetwild.json --stage sweep \
  --configs results/simjeb-remesh-configs.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/elasticity.py \
  --data data/simjeb-ftetwild --output results/simjeb-ftetwild.json --stage final
python benchmarks/elasticity_stress.py \
  --data data/simjeb-ftetwild --output results/simjeb-ftetwild.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/compare_simjeb_meshes.py
python benchmarks/plot_elasticity.py \
  --input results/simjeb-ftetwild.json --output assets/simjeb-ftetwild-convergence.png
# Blender 4.5 with OptiX:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 blender -b --factory-startup \
  --python visualization/render_simjeb_mesh_comparison.py
python visualization/compose_simjeb_mesh_comparison.py
```

[Raw quality, transfer and solution comparisons](simjeb-mesh-comparison.json),
[full solver measurements](simjeb-ftetwild.json),
[render hashes and settings](../assets/simjeb-mesh-comparison.json).
Mesher output and numerical fields are in ignored `data/simjeb-ftetwild/`;
PNG figures are committed using Git LFS.
