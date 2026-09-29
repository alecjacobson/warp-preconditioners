"""Benchmark adapter for float32 factor storage with float64 accumulation."""

import warp as wp

from warp_preconditioners import FSAI


class CompactFSAI(FSAI):
    def __init__(self, A, **kwargs):
        super().__init__(A, factor_dtype=wp.float32, **kwargs)
