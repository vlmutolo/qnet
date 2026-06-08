
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qbp_sim.core.types import Array1D, FloatMatrix, IntArray1D, IntMatrix

def _build_pair_index(n_nodes: int) -> tuple[IntArray1D, IntArray1D]:
    pair_u: list[int] = []
    pair_v: list[int] = []
    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            pair_u.append(x)
            pair_v.append(y)
    return np.asarray(pair_u, dtype=np.int64), np.asarray(pair_v, dtype=np.int64)


def _build_swap_candidates(n_nodes: int) -> tuple[IntArray1D, IntArray1D, IntArray1D]:
    swap_i: list[int] = []
    swap_y: list[int] = []
    swap_z: list[int] = []
    for i in range(n_nodes):
        others = [j for j in range(n_nodes) if j != i]
        for left in range(len(others)):
            for right in range(left + 1, len(others)):
                y = others[left]
                z = others[right]
                swap_i.append(i)
                swap_y.append(y)
                swap_z.append(z)
    return (
        np.asarray(swap_i, dtype=np.int64),
        np.asarray(swap_y, dtype=np.int64),
        np.asarray(swap_z, dtype=np.int64),
    )


def _build_pair_lookup(n_nodes: int, pair_u: IntArray1D, pair_v: IntArray1D) -> IntMatrix:
    lookup = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    for idx, (x, y) in enumerate(zip(pair_u, pair_v, strict=True)):
        lookup[x, y] = idx
        lookup[y, x] = idx
    return lookup


def _build_swap_node_starts(n_nodes: int, swap_i: IntArray1D) -> IntArray1D:
    starts = np.zeros(n_nodes + 1, dtype=np.int64)
    if swap_i.shape[0] == 0:
        return starts
    current_node = 0
    for idx in range(swap_i.shape[0]):
        node = int(swap_i[idx])
        while current_node < node:
            starts[current_node + 1] = idx
            current_node += 1
    while current_node < n_nodes:
        starts[current_node + 1] = swap_i.shape[0]
        current_node += 1
    return starts


def _build_swap_lookup(
    n_nodes: int,
    swap_i: IntArray1D,
    swap_y: IntArray1D,
    swap_z: IntArray1D,
) -> NDArray[np.int64]:
    lookup = np.full((n_nodes, n_nodes, n_nodes), -1, dtype=np.int64)
    for idx, (i, y, z) in enumerate(zip(swap_i, swap_y, swap_z, strict=True)):
        lookup[i, y, z] = idx
        lookup[i, z, y] = idx
    return lookup


def _matrix_to_pair_vector(matrix: FloatMatrix, pair_u: IntArray1D, pair_v: IntArray1D) -> Array1D:
    values = np.zeros(pair_u.shape[0], dtype=np.float64)
    for idx, (x, y) in enumerate(zip(pair_u, pair_v, strict=True)):
        values[idx] = float(matrix[x, y])
    return values


def _init_state_matrix(n_nodes: int, value: IntMatrix | None) -> IntMatrix:
    if value is None:
        return np.zeros((n_nodes, n_nodes), dtype=np.int64)
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.shape != (n_nodes, n_nodes):
        raise ValueError("Initial state matrices must match the configured node count.")
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Initial state matrices must be symmetric.")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def _init_swap_counter(size: int, value: IntArray1D | None) -> IntArray1D:
    if value is None:
        return np.zeros(size, dtype=np.int64)
    counter = np.asarray(value, dtype=np.int64)
    if counter.shape != (size,):
        raise ValueError("Initial swap counters must match the number of swap candidates.")
    if np.any(counter < 0):
        raise ValueError("Initial swap counters must be non-negative.")
    return counter.copy()


def _upper_triangle_sum(matrix: IntMatrix) -> int:
    return int(np.triu(matrix, k=1).sum())
