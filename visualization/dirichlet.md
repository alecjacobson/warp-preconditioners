# Head-to-tail biharmonic interpolation

![Prescribed head and tail regions, followed by FSAI-CG and Jacobi-CR solutions at k, 10k and 100k.](../assets/dragon-dirichlet-comparison.png)

This comparison has **no data term**. On the original dragon surface mesh,
we minimize

$$
E(u)=\frac{1}{2}u^TQu,
\qquad Q=LM^{-1}L,
\qquad u_i=-1\text{ on the tail},\quad u_i=1\text{ on the head}.
$$

Here $L$ is the positive cotangent stiffness matrix and $M$ is the positive
diagonal mass matrix from the benchmark. Reversing the sign convention for
$L$ leaves $Q$ unchanged. Every solve starts from **zero on the free
vertices**, with the prescribed values already assigned on the fixed
vertices. There is no diagonal regularizer or penalty approximation to the
constraints.

## Regions and reduced system

The leftmost dragon identifies the constrained vertices: **blue is the tail
at −1, red is the head at +1, and gray is free**. This view uses categorical
colors, not the solution colorbar. The remaining four dragons share one
26-band striped scalar colormap, including all their extrema.

Selections use the original mesh coordinates:

| Patch | Selection | Vertices | Value |
| --- | --- | ---: | ---: |
| Raised tail | Largest connected component of $x<-70$ and $z>45$ | 12,684 | −1 |
| Upper head | Largest connected component of $x>70$ and $z>55$ | 27,578 | +1 |

Both candidate selections already form single connected patches. The
original mesh has 360,757 vertices and 721,510 triangles. After fixing
40,262 vertices, the reduced system has **320,495 unknowns and 6,525,939
stored nonzeros**.

Let $b$ denote fixed indices and $f$ free indices. We form the **full**
squared operator before eliminating fixed values:

$$
Q_{ff}u_f=-Q_{fb}u_b.
$$

In particular, the energy retains all rows of $L$, including those at fixed
vertices. Squaring an already restricted Laplacian generally gives a
different problem. The mesh is connected, so the nullspace of $L$, and
therefore of $Q$, consists of constants. Fixing these nonempty patches
removes that nullspace: $Q_{ff}$ is symmetric positive definite. An
independent Cholesky factorization with AMD ordering checks this without
any diagonal shift.

## Solver comparison and convergence

All displayed fields come directly from one uninterrupted float64 Warp
solve from zero. FSAI uses adaptive width 8 with CG. Its actual stopping
iteration sets $k$; Warp CR with Jacobi runs for exactly $k$, $10k$, and
$100k$. Setup and kernels are warmed before timing. Solve times exclude
setup, field downloads, and verification.

The FSAI recursive relative residual tolerance is **$10^{-12}$** for this
example. A trial at $10^{-8}$ stopped at 97,746 iterations but differed from
a longer solve by **0.895% in relative mass norm**. That threshold was not
sufficient to declare the field converged. The published run uses the
tighter threshold and checks both a $2k$ FSAI solve and an independent
Cholesky solution.

The final FSAI run stops at **$k=139{,}136$**, in **43.29 seconds**. A separate
$2k$ run changes the field by only **$3.35\times10^{-8}$** in relative mass
norm. The FSAI field differs from the independent Cholesky result by
**$4.53\times10^{-5}$** in relative mass norm, with maximum absolute
difference $6.98\times10^{-5}$. Its energy differs by about
$2.00\times10^{-7}$ relatively. These numbers describe the accuracy actually
checked, rather than interpreting the stopping tolerance as a field-error
bound.

| Method | Iterations | Solve time (s) | Relative mass-norm difference from FSAI | Energy |
| --- | ---: | ---: | ---: | ---: |
| FSAI-CG | 139,136 | 43.29 | 0 | 0.0002063423 |
| Jacobi-CR, $k$ | 139,136 | 23.94 | 0.7657 | 0.01005694 |
| Jacobi-CR, $10k$ | 1,391,360 | 242.65 | 0.1430 | 0.0002694790 |
| Jacobi-CR, $100k$ | 13,913,600 | 2,425.32 | 1.223e-7 | 0.0002063423 |

At the same iteration count, Jacobi is faster per iteration but its field
is still **76.6%** away from FSAI. At ten times the iterations the difference
is **14.3%**. At one hundred times the iterations it agrees visually and to
about $1.22\times10^{-7}$ in relative mass norm. These are single-run GPU
timings on the L40, not repeated-trial performance estimates.

The independently recomputed relative residuals are $2.97\times10^{-9}$
for FSAI and $1.50\times10^{-6}$, $5.53\times10^{-8}$, and
$9.94\times10^{-8}$ for the three Jacobi budgets. In particular, the last
Jacobi run has a slightly larger recomputed residual than the $10k$ run,
even though its field is much more accurate. Its internal recursive
residual is $9.26\times10^{-17}$; finite-precision residual drift makes
that number an unreliable standalone accuracy measure here. The figure
shows the independently recomputed residuals.

All four fields satisfy the constraints **exactly**. The FSAI solution
ranges from **−1.151744 to 1.020902**. The common colorbar extends to 1.156399
to include the larger overshoot in the unconverged Jacobi-$k$ field.

The field errors use the mass norm

$$
\frac{\|u-u_{\mathrm{ref}}\|_M}{\|u_{\mathrm{ref}}\|_M},
\qquad \|v\|_M=\sqrt{v^TMv}.
$$

Independently recomputed residuals and the internal stopping residual are
reported separately in the figure metadata. Reaching a small recursive
residual is not, by itself, a guarantee of small solution error on this
ill-conditioned problem. The Cholesky comparison is an independent
floating-point check, not an exact-arithmetic reference.

Biharmonic interpolation has no general maximum principle: the solution
can exceed the prescribed interval $[-1,1]$. We preserve this overshoot
in the fields and in the shared color range.

## Verification and reproduction

The script checks the dumped stiffness against a fresh cotangent assembly,
mesh connectivity, matrix symmetry, exact constraints, solver iteration
counts, and stability of the converged FSAI field. The optional Cholesky
check also verifies the energy as $\frac12\sum_i (Lu)_i^2/M_{ii}$ and free
stationarity against the full squared operator. Every rendered field is
hash checked, and Blender's actual scalar attribute is read back and
compared with the float32 conversion of the saved float64 solution.

Generate the benchmark matrix dumps as described in the main README, then:

```bash
OPENBLAS_NUM_THREADS=1 python visualization/solve_dirichlet.py \
  --dir /tmp/dump --mesh /path/to/xyzrgb_dragon-720K.ply
# Optional independent verification; requires scikit-sparse and SuiteSparse:
OPENBLAS_NUM_THREADS=1 python visualization/verify_dirichlet.py
blender -b --factory-startup --python visualization/render_dragon.py -- \
  --data data/dirichlet --width 6000 --height 1120 --samples 64
python visualization/compose_dirichlet.py
```

The numerical data and editable scene are saved in ignored
`data/dirichlet/`, including `fields.npz`, `solve.json`, `cholesky.json`, and
`five_dragons.blend`. Published measurements are in
[`dragon-dirichlet-comparison.json`](../assets/dragon-dirichlet-comparison.json),
and the figure uses Git LFS. The existing
[nonlinear data-smoothing comparison](README.md) is a separate experiment.
