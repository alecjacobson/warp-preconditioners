"""Untimed rotation audit: separate adaptive-pattern changes from factor precision."""

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from block_challenge import coupled, rotation
from block_fsai_candidate import BsrBlockFSAI


def pattern(pre):
    off, col = pre.G.offsets.numpy(), pre.G.columns.numpy()
    return [tuple(np.unique(col[off[3 * i] : off[3 * i + 1]] // 3)) for i in range(pre.G.nrow // 3)]


def action(pre, probe):
    x = wp.array(probe, dtype=wp.vec3d)
    y = wp.empty_like(x)
    pre.matvec(x, y, y, 1.0, 0.0)
    return y.numpy()


def main():
    wp.init()
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    probe = np.random.default_rng(391).normal(size=(8**3, 3))
    output = dict(
        description="Untimed 8^3 tensor-grid audit. Rotate the same probe and transform output back. Fixed-pattern refit keeps the aligned block supports.",
        source_sha256={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [__file__, "benchmarks/block_fsai_candidate.py"]
        },
        cases=[],
    )
    for contrast in [1, 100, 1e6]:
        a, *_ = coupled(8, contrast, "aligned")
        for blocks in [2, 4]:
            for storage in [wp.float64, wp.float32]:
                base = BsrBlockFSAI(a, max_blocks=blocks, factor_dtype=storage, reuse_pattern=True)
                expected, original = action(base, probe), pattern(base)
                for direction in ["rotated", "rotated2"]:
                    r = rotation(direction)
                    rotated, *_ = coupled(8, contrast, direction)
                    adaptive = BsrBlockFSAI(rotated, max_blocks=blocks, factor_dtype=storage)
                    base.update(rotated)
                    fixed_action = action(base, probe @ r.T) @ r
                    adaptive_action = action(adaptive, probe @ r.T) @ r
                    changed = np.mean([x != y for x, y in zip(original, pattern(adaptive))])
                    fixed_error = np.linalg.norm(fixed_action - expected) / np.linalg.norm(expected)
                    row = dict(
                        contrast=contrast,
                        blocks=blocks,
                        storage=str(storage),
                        direction=direction,
                        changed_block_rows_fraction=float(changed),
                        adaptive_action_rotation_error=float(
                            np.linalg.norm(adaptive_action - expected) / np.linalg.norm(expected)
                        ),
                        fixed_pattern_action_rotation_error=float(fixed_error),
                    )
                    if storage == wp.float64:
                        assert fixed_error < 1e-8
                    output["cases"].append(row)
                    print(row, flush=True)
    Path("results/block-rotation-audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
