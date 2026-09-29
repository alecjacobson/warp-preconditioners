# Dragon scalar-field comparison

The README PNG shows actual Warp solutions of the benchmark's scalar
biharmonic equation `(M + K M^-1 K) u = M z`, where `z` is the original
mesh height. It uses column 2 (zero based) of `k2_rhs.mtx`, and the original
360,757-vertex, 721,510-triangle dragon geometry.

- **Left:** Warp CR with Jacobi, the strongest tested configuration without
  FSAI. Run the original z-column budget of 118,500 iterations in chunks of
  500; retain the checkpoint with the smallest independently computed
  componentwise backward error.
- **Right:** Warp CG with width-eight adaptive FSAI, the fastest successful
  configuration in the support-size study, after 500 iterations.

“Without FSAI” retains the benchmark's Jacobi preconditioner; it does not
mean an identity preconditioner. “Best” refers to the configurations and
budgets measured in this repository, not a search over all possible Warp
solvers or iteration budgets. The comparison is not an equal-time run.
The labels report this single scalar RHS, not three-RHS benchmark totals.
The FSAI solution is not claimed to be an exact reference solution.

The two solutions share the same physical scalar range, computed from their
combined extrema. Neither is individually normalized, filtered, smoothed,
or substituted with the original height field. Scalar values are
interpolated over each original triangle and only then passed through a
26-entry, constant-interpolation color ramp. This avoids the smeared
boundaries that interpolating prequantized vertex RGB colors would cause.
Smooth shading affects lighting only, not geometry or scalar data.

The palette reproduces
`isolines_stripe_map(okloop(26,-4/3*pi,-1/2*pi))`: an OKLab hue arc at the
original lightness/chroma, with alternate interval lightness multiplied by
0.9. The source functions inspected are in
[gptoolbox at dd355405](https://github.com/alecjacobson/gptoolbox/tree/dd3554053237cd02d305b78db298d913a6196508),
under `imageprocessing/` and `mesh/isolines_stripe_map.m`. The translation's
license notice is retained in `third_party/gptoolbox-LICENSE-MIT.txt`.

Blender Cycles/OptiX renders identical orthographic cameras and soft area
lights. A shadow catcher preserves contact shadows over the white canvas.
Lighting modulates the palette on the surface; the shared colorbar shows
the unlit sRGB colors. The shipped render uses Blender 4.5.3, 128 samples,
and 1600×1240 pixels per panel. `assets/dragon-biharmonic-comparison.json`
records solver, palette, camera, and mesh provenance.

## Reproduce

Generate the C++ benchmark dumps using the main README instructions, then:

```bash
python -m pip install -e '.[visualization]'
OPENBLAS_NUM_THREADS=1 python visualization/solve_fields.py \
  --dir /tmp/dump --mesh /path/to/xyzrgb_dragon-720K.ply
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --width 1600 --height 1240 --samples 128
python visualization/compose.py
```

The solve currently targets `cuda:0`, and the render targets an OptiX GPU.
Composition uses the DejaVu Sans fonts under `/usr/share/fonts/truetype/`.
The intermediate fields and renders are stored in ignored
`data/visualization/`; the PNG is tracked through Git LFS. The numerical
solver library still depends only on Warp.
