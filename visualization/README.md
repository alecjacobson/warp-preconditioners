# Dragon scalar-field comparison

The figure contains **four dragons in one Blender scene**, under a single
orthographic camera with shared lighting and ground. From left to right:

1. Warp CG + adaptive FSAI (width 8), **500 iterations**.
2. Warp CR + Jacobi, **500 iterations**.
3. Warp CR + Jacobi, **5,000 iterations**.
4. Warp CR + Jacobi, **50,000 iterations**.

Each field is the final iterate of one uninterrupted solve from zero. There
are no 500-step restarts and no selection of a best intermediate iterate.
The actual iteration counts returned by Warp are checked against these
budgets. Setup and solver kernels are warmed before timing; setup times
are recorded separately. The comparison is by iteration budget, not equal
wall-clock time. Timings refer to one scalar right-hand side.

## Equation and stronger smoothing

We minimize the screened biharmonic energy

$$
E(u)=\frac{\alpha}{2}(u-z)^T M(u-z)
     +\frac{1}{2}(Ku)^T M^{-1}(Ku),
\qquad
(\alpha M+KM^{-1}K)u=\alpha Mz.
$$

Here $z$ is original mesh height, $M$ is the diagonal vertex mass matrix,
and $K$ is the positive cotangent stiffness matrix. Geometry and target
height are unchanged. **The data weight is reduced from 1 to 0.001**;
the smoothing weight stays 1. Equivalently, the smoothing/data ratio is
increased by a factor of 1,000. These coefficients use the original mesh
units; they are not scale independent.

This is the surface Laplacian-squared energy from the solver benchmark.
Ambient affine functions are not generally in its nullspace on a curved
surface. With the original data weight, however, its solution stayed very
close to the affine target height: an area-weighted affine-fit residual
was approximately 0.5% of the field's standard deviation. A 5,000-step
FSAI diagnostic at the new weight increases that fraction to approximately
4.6%. The weight sweep and its residuals are included in the figure JSON.

**These displayed fields are finite-budget iterates, not converged reference
solutions.** Stronger smoothing also makes the linear system harder. In
particular, the extra variation in the 500-step FSAI field includes iteration
error; it must not all be interpreted as the converged smoothing effect.
Both true relative residual and componentwise backward error are printed
under each dragon. Small backward error alone can conceal a substantial
relative residual in this poorly conditioned system.

## Checking what is visualized

The workflow verifies all of the following:

- All three original RHS columns equal $M$ times the original mesh coordinates,
  and the visualized scalar RHS is column 2 (zero based).
- The dumped stiffness agrees with an independent cotangent assembly on the
  displayed mesh (relative maximum difference about $1.05\times10^{-8}$,
  allowing for roundoff on very thin triangles across libigl builds).
- The original dumped biharmonic matrix agrees with $M+KM^{-1}K$ to relative
  maximum error below $10^{-12}$; only the mass coefficient and RHS are changed.
- Every displayed field is downloaded directly from its Warp solve in float64.
  Its SHA-256 is checked before assigning it to the corresponding dragon.
- Blender's actual per-vertex `biharmonic_u` attribute is read back and checked
  against the float32 conversion of that field. Conversion errors are recorded.

All four use the original 360,757-vertex, 721,510-triangle mesh, positioned
by uniform scaling and translation only. No solution is replaced by raw
height, affine fitting, smoothing, or filtering. The affine fit above is a
numerical diagnostic only. Smooth shading changes lighting, not scalar data.

## Colors and rendering

All dragons share the combined scalar extrema and the same 26 color
intervals. Values are interpolated over each original triangle **before**
applying the constant color ramp. No object has its own normalization, and
colors are not interpolated from prequantized vertex RGB values.

The palette reproduces
`isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))`: an OKLab hue arc with
alternate interval lightness multiplied by 0.9. The source functions are in
[gptoolbox at dd355405](https://github.com/alecjacobson/gptoolbox/tree/dd3554053237cd02d305b78db298d913a6196508),
under `imageprocessing/` and `mesh/isolines_stripe_map.m`. The translation's
license notice is retained in `third_party/gptoolbox-LICENSE-MIT.txt`.

Cycles/OptiX renders all four meshes simultaneously, including shared contact
shadows. Composition adds labels and an unlit sRGB colorbar to this **single
render**; it does not combine separate dragon renders. Camera settings,
object positions, field hashes, and solver diagnostics are retained in
`assets/dragon-biharmonic-comparison.json`.

## Reproduce

Generate the C++ benchmark dumps using the main README instructions, then:

```bash
python -m pip install -e '.[visualization]'
OPENBLAS_NUM_THREADS=1 python visualization/solve_fields.py \
  --dir /tmp/dump --mesh /path/to/xyzrgb_dragon-720K.ply \
  --data-weight 0.001
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --width 4800 --height 1120 --samples 64
python visualization/compose.py
```

To reproduce the optional weight diagnostic, run the solve script with
`--audit-weights 1 0.1 0.01 0.001 0.0001` first. This writes a sweep without
replacing the four displayed fields. Longer diagnostic runs can terminate
before their budget if Warp's recursive residual reaches numerical zero;
actual counts and independently recomputed residuals are recorded.

The solve uses `cuda:0`; rendering uses an OptiX GPU and Blender 4.5.3.
The editable four-object scene is saved as
`data/visualization/four_dragons.blend`. Intermediate fields and renders are
also in ignored `data/visualization/`. The published PNG uses Git LFS.
The numerical solver library still depends only on Warp.
