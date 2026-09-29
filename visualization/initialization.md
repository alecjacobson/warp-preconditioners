# Does starting from the data help?

**For this example, yes—especially for early field accuracy.** Starting from
the data reduces the FSAI convergence count by about 2.7%, and produces much
better Jacobi-CR fields at equal iteration budgets. This is a separate
paired experiment, not a conclusion inferred from the main comparison figure.

![Independent initial-guess comparison](../assets/dragon-initialization-comparison.png)

The equation, nonlinear target, data weight $10^{-4}$, preconditioner
(width-eight FSAI or Jacobi), and relative stopping tolerance $10^{-8}$ are
identical within each pair. Only the initial guess changes. Runs are
independent and uninterrupted. Kernels are warmed; setup and solve are
timed separately. The target is defined in the [visualization notes](README.md).

| Measurement | Start from zero | Start from data |
| --- | ---: | ---: |
| FSAI iterations to its stopping tolerance | 29,481 | 28,688 |
| FSAI solve time, this paired run | 10.33 s | 10.03 s |
| Initial relative field difference | 100% | 17.23% |
| Jacobi field difference after 1,000 iterations | 99.59% | 16.58% |
| Jacobi field difference after 28,688 iterations | 29.99% | 3.93% |

Field differences use $\|u-u_*\|_M/\|u_*\|_M$, where
$\|v\|_M=\sqrt{v^TMv}$ and $u_*$ is the converged FSAI run initialized from
the data. It is an approximate reference, not an exact solution. The two
converged FSAI runs agree to approximately $3.94\times10^{-7}$ in this norm.
The timing difference is a single paired observation, not a statistically
established universal speedup. The iteration counts and field errors give
the more useful evidence here.

## Why this can help without a small initial residual

Write $B=KM^{-1}K$. The smoothing problem is

$$
(\alpha M+B)u=\alpha Mf.
$$

Starting from $f$ gives $r_0=-Bf$; starting from zero gives $r_0=\alpha Mf$.
The data can already be close to the smoothed field while its curvature
produces a large residual. Here the data initialization's relative residual
is about $2.15\times10^7$, versus 1 for zero initialization, yet its initial
field difference is much smaller. Its initial bending energy is also larger.
A smaller residual is therefore not the justification for using the data.

Using $u_0=f$ is a reasonable default for this data-smoothing example because
it preserves a useful field before convergence. It is not guaranteed to
reduce Krylov iterations for every signal or smoothing weight; the spectral
content of the initial error matters. For extremely strong smoothing or
noisy data, another initial guess can be preferable.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python visualization/solve_fields.py \
  --mesh /path/to/xyzrgb_dragon-720K.ply --audit-initial-guess
python visualization/plot_initialization.py
```

The experiment writes its own `initialization_audit.json` and
`initialization_fields.npz` under `data/visualization/`; it does not overwrite
the four displayed fields. The published plot and full measurements are
[PNG](../assets/dragon-initialization-comparison.png) and
[JSON](../assets/dragon-initialization-comparison.json).
