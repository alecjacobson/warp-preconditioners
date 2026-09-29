"""Benchmark adapters for the library's cooperative CSR implementation."""

from warp_preconditioners import FSAI, SparseOperator
from warp_preconditioners.sparse_operator import _grouped_mv as grouped_mv

__all__ = ["GroupedOperator", "GroupedFSAI", "grouped_mv"]


class GroupedOperator(SparseOperator):
    def __init__(self, a, lanes=4):
        super().__init__(a, row_lanes=lanes)


class GroupedFSAI(FSAI):
    def __init__(self, a, lanes=4, **kwargs):
        super().__init__(a, apply_lanes=lanes, **kwargs)
