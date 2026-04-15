"""
LP benchmark model for quantum-pair routing.

Important:
- This file does not implement the same stochastic event-driven backpressure model used by
  the Gillespie simulator in `src/qbp_sim/`.
- It solves a global average-rate linear program, so its outputs should be treated as an
  optimal benchmark or comparison target, not as a drop-in controller or replayable event log.
- Future AI agents should preserve that distinction when comparing LP outputs to backpressure
  runs. "Convergence to LP" means convergence of aggregate rates or summary statistics, not
  equivalence of the underlying simulation model.
- Third-party dependencies used here should be managed at the project level in
  `pyproject.toml` rather than hidden as one-off assumptions inside this file.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.optimize import linprog
from tqdm import tqdm

from qbp_sim.config import SimulationInputConfig
from qbp_sim.progress import should_use_progress


def emit_single_run_lp_json(
    *,
    payload: dict,
    json_output_path: str | None,
    json_pretty: bool,
) -> None:
    """
    Emit the single-run LP JSON payload either to a file (default) or stdout.

    Args:
        payload: JSON-serializable dictionary.
        json_output_path: If provided, writes JSON to this path and ensures parent dir exists.
                          If None, prints JSON to stdout.
        json_pretty: If True, pretty-print with indentation.
    """
    if json_output_path:
        parent = os.path.dirname(json_output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2 if json_pretty else None, sort_keys=False)
        print(f"Wrote JSON solution record to: {json_output_path}")
    else:
        print(json.dumps(payload, indent=2 if json_pretty else None, sort_keys=False))


def emit_single_run_cycle_json(
    *,
    payload: dict,
    json_output_path: str | None,
    json_pretty: bool,
) -> None:
    """
    Backward-compatible wrapper for older cycle-only callers.
    """
    emit_single_run_lp_json(
        payload=payload,
        json_output_path=json_output_path,
        json_pretty=json_pretty,
    )


def emit_simulation_input_json(
    *,
    payload: SimulationInputConfig,
    json_output_path: str | None,
    json_pretty: bool,
) -> None:
    if json_output_path:
        parent = os.path.dirname(json_output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        Path(json_output_path).write_text(
            payload.model_dump_json(indent=2 if json_pretty else None),
            encoding="utf-8",
        )
        print(f"Wrote simulation config JSON to: {json_output_path}")
    else:
        print(payload.model_dump_json(indent=2 if json_pretty else None))


def build_simulation_input_config(
    *,
    generation_graph: np.ndarray,
    consumption_graph: np.ndarray,
    swap_rates: list[float] | np.ndarray,
) -> SimulationInputConfig:
    return SimulationInputConfig(
        generation_rates=np.asarray(generation_graph, dtype=float).tolist(),
        consumption_rates=np.asarray(consumption_graph, dtype=float).tolist(),
        swap_rates=np.asarray(swap_rates, dtype=float).tolist(),
    )


def lp_node_swap_rates(spec: "LinearSpec", lp_result) -> np.ndarray:
    """Aggregate the LP swap solution into one continuous-time swap rate per node."""
    if lp_result.status != 0:
        raise ValueError(f"LP not solved to optimality (status={lp_result.status}).")

    swap_rates = np.zeros(spec.num_nodes, dtype=float)
    for k in range(spec.num_nodes):
        total = 0.0
        for i in range(spec.num_nodes):
            for j in range(i + 1, spec.num_nodes):
                idx = spec._offsets.swap + spec._triple_to_idx((k, i, j))
                total += lp_result.x[idx]
        swap_rates[k] = total
    return swap_rates


def build_lp_solution_simulation_input_config(
    *,
    spec: "LinearSpec",
    lp_result,
) -> SimulationInputConfig:
    """Build a BP simulation config from the LP's optimal average-rate solution."""
    return build_simulation_input_config(
        generation_graph=spec.generate_matrix(lp_result),
        consumption_graph=spec.consume_matrix(lp_result),
        swap_rates=lp_node_swap_rates(spec, lp_result),
    )


class LinearSpec:
    def __init__(self, num_nodes: int):
        self._offsets = Offsets(num_nodes)

        self.a_ub: list[NDArray[np.float64]] = []
        self.b_ub: list[float] = []
        # Sparse equality constraint storage
        self.eq_rows: list[int] = []
        self.eq_cols: list[int] = []
        self.eq_data: list[float] = []
        self.b_eq: list[float] = []
        self.lb: list[None | float] = [0.0 for _ in range(self._offsets.total_num_vars)]
        self.ub: list[None | float] = [
            None for _ in range(self._offsets.total_num_vars)
        ]
        self.num_nodes = num_nodes
        self._zero_out_degenerate_swaps()

    def solve(self, objective: str = "min_sum_generate"):
        """
        Solve the LP with a selectable objective.

        Args:
            objective:
                - "min_sum_generate": minimize total generation sum_{i,j} g(i,j)
                - "min_total_swaps": minimize total swap rate sum_{k,i,j} sigma(k,i,j)
        """
        if objective == "min_sum_generate":
            c = self.cost_min_sum_generate()
        elif objective == "min_total_swaps":
            c = self.cost_min_total_swaps()
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Expected 'min_sum_generate' or 'min_total_swaps'."
            )

        self._add_flow_and_symmetry_constraints()

        # Build sparse equality matrix from accumulated triplets
        if self.b_eq:
            A_eq = sp.coo_matrix(
                (self.eq_data, (self.eq_rows, self.eq_cols)),
                shape=(len(self.b_eq), self._offsets.total_num_vars),
            )
            b_eq = np.array(self.b_eq)
        else:
            A_eq = None
            b_eq = None

        # Inequalities remain optional and are left as-is if provided elsewhere
        a_ub = np.array(self.a_ub) if self.a_ub else None
        b_ub = np.array(self.b_ub) if self.b_ub else None

        bounds = list(zip(self.lb, self.ub))

        return linprog(
            c, A_ub=a_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )

    def add_generate_constraints(self, generate_adj_matrix: NDArray[np.float64]):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                var_idx = self._offsets.generate + self._edge_to_idx((i, j))
                self._add_univariate_ub_constraint(var_idx, generate_adj_matrix[i, j])

    def add_generate_zero_constraints(self, generate_adj_matrix: NDArray[np.float64]):
        """
        Apply only zero constraints for generation: if capacity(i,j) == 0, force g(i,j) = 0;
        otherwise leave g(i,j) unbounded above to allow fair comparison to path routing.
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if generate_adj_matrix[i, j] <= 0:
                    var_idx = self._offsets.generate + self._edge_to_idx((i, j))
                    self._add_univariate_ub_constraint(var_idx, 0.0)

    def add_consume_constraints(self, consume_adj_matrix: NDArray[np.float64]):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                lhs = self._var_consume((i, j))
                rhs = consume_adj_matrix[i, j]
                self._add_eq_constraint(lhs, rhs)

    def cost_min_sum_generate(self) -> NDArray[np.float64]:
        a = np.zeros(self._offsets.total_num_vars)
        a[self._offsets.generate : self._offsets.consume] = 1
        return a

    def cost_min_total_swaps(self) -> NDArray[np.float64]:
        """
        Objective: minimize total swap rate.

        This places weight 1 on every swap decision variable sigma(k,i,j) in the
        swap block and 0 elsewhere. Degenerate swap variables are already forced
        to 0 by `_zero_out_degenerate_swaps()`.
        """
        a = np.zeros(self._offsets.total_num_vars)
        a[self._offsets.swap : self._offsets.arrive] = 1
        return a

    def _add_flow_and_symmetry_constraints(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                # r+ = r- for every edge
                lhs = self._var_arrive((i, j)) - self._var_depart((i, j))
                self._add_eq_constraint(lhs, 0)

                if i < j:
                    # r+(i,j) = r+(j,i) and r-(i,j) = r-(j,i)
                    lhs = self._var_arrive((i, j)) - self._var_arrive((j, i))
                    self._add_eq_constraint(lhs, 0)

                if i < j:
                    # g(i,j) = g(j,i)
                    lhs = self._var_generate((i, j)) - self._var_generate((j, i))
                    self._add_eq_constraint(lhs, 0)

                    # Couple swap variables at intermediate k: sigma(k,i,j) = sigma(k,j,i)
                    for k in range(self.num_nodes):
                        if k != i and k != j:
                            lhs = self._var_swap((k, i, j)) - self._var_swap((k, j, i))
                            self._add_eq_constraint(lhs, 0)

                # r+(i,j) = g(i,j) + sum_k (sigma(k,i,j)) with k distinct from i,j
                sum_term = sp.coo_matrix((1, self._offsets.total_num_vars))
                for k in range(self.num_nodes):
                    if k != i and k != j:
                        sum_term = sum_term + self._var_swap((k, i, j))
                lhs = self._var_generate((i, j)) + sum_term - self._var_arrive((i, j))
                self._add_eq_constraint(lhs, 0)

                # r-(i,j) = c(i,j) + sum_k (sigma(i,j,k) + sigma(j,i,k)) with k distinct from i,j
                sum_term = sp.coo_matrix((1, self._offsets.total_num_vars))
                for k in range(self.num_nodes):
                    if k != i and k != j:
                        sum_term = (
                            sum_term
                            + self._var_swap((i, j, k))
                            + self._var_swap((j, i, k))
                        )
                lhs = self._var_consume((i, j)) + sum_term - self._var_depart((i, j))
                self._add_eq_constraint(lhs, 0)

    def _add_eq_constraint(self, lhs, rhs: float):
        # Accept sparse or dense lhs; convert to sparse triplets without storing dense rows
        if sp.issparse(lhs):
            coo = lhs.tocoo()
            cols = coo.col
            data = coo.data
        else:
            idxs = np.nonzero(lhs)[0]
            cols = idxs
            data = lhs[idxs]
        row = len(self.b_eq)
        if len(cols) > 0:
            self.eq_rows.extend([row] * len(cols))
            self.eq_cols.extend(cols.tolist())
            self.eq_data.extend(data.tolist())
        self.b_eq.append(rhs)

    def _add_ub_constraint(self, lhs: NDArray[np.float64], rhs: float):
        self.a_ub.append(lhs)
        self.b_ub.append(rhs)

    def _add_univariate_ub_constraint(self, idx: int, ub: float):
        self.ub[idx] = ub

    def _zero_out_degenerate_swaps(self):
        """
        Disallow degenerate swap variables by setting ub=0 for any sigma(i,j,k)
        where i == j or k in {i, j}. These do not correspond to physical swaps.
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                for k in range(self.num_nodes):
                    if i == j or k == i or k == j:
                        idx = self._offsets.swap + self._triple_to_idx((i, j, k))
                        self.ub[idx] = 0.0

    def _edge_to_idx(self, edge: tuple[int, int]) -> int:
        (i, j) = edge
        return i * self.num_nodes + j

    def _triple_to_idx(self, triple: tuple[int, int, int]) -> int:
        (i, j, k) = triple
        return i * self.num_nodes**2 + j * self.num_nodes + k

    def _mk_var_array(self, offset: int, idx: int):
        col = offset + idx
        # Return a 1 x total_vars sparse row with a single 1 at 'col'
        return sp.coo_matrix(
            (np.array([1.0]), (np.array([0]), np.array([col]))),
            shape=(1, self._offsets.total_num_vars),
        )

    def _var_generate(self, edge: tuple[int, int]) -> NDArray[np.float64]:
        idx = self._edge_to_idx(edge)
        return self._mk_var_array(self._offsets.generate, idx)

    def _var_consume(self, edge: tuple[int, int]) -> NDArray[np.float64]:
        idx = self._edge_to_idx(edge)
        return self._mk_var_array(self._offsets.consume, idx)

    def _var_swap(self, triple: tuple[int, int, int]) -> NDArray[np.float64]:
        idx = self._triple_to_idx(triple)
        return self._mk_var_array(self._offsets.swap, idx)

    def _var_arrive(self, edge: tuple[int, int]) -> NDArray[np.float64]:
        idx = self._edge_to_idx(edge)
        return self._mk_var_array(self._offsets.arrive, idx)

    def _var_depart(self, edge: tuple[int, int]) -> NDArray[np.float64]:
        idx = self._edge_to_idx(edge)
        return self._mk_var_array(self._offsets.depart, idx)

    def consume_matrix(self, lp_result) -> np.ndarray:
        """
        Return the symmetric matrix C with entries C[i, j] = c(i,j).

        The consume variables occupy the range
        ``[self._offsets.consume, self._offsets.desired_consume)``,
        which is reshaped into an (N × N) array. The matrix is expected
        to be symmetric by construction; this method asserts that
        property and returns the raw matrix.

        Parameters
        ----------
        lp_result : scipy.optimize.OptimizeResult
            Result object returned by :pyfunc:`scipy.optimize.linprog`.

        Returns
        -------
        numpy.ndarray
            An (N × N) array of optimal consume rates (symmetric).
        """
        if lp_result.status != 0:
            raise ValueError(
                f"LP not solved to optimality (status={lp_result.status})."
            )

        start, end = self._offsets.consume, self._offsets.swap
        consume_vec = lp_result.x[start:end]
        mat = consume_vec.reshape(self.num_nodes, self.num_nodes)

        # Assert symmetry to catch potential bugs
        if not np.allclose(mat, mat.T, atol=1e-9):
            raise AssertionError("Consume matrix should be symmetric but is not.")

        return mat

    def generate_matrix(self, lp_result) -> np.ndarray:
        """
        Return the symmetric matrix G with entries G[i, j] = g(i,j).

        The generate variables occupy the range
        ``[self._offsets.generate, self._offsets.consume)``,
        which is reshaped into an (N × N) array. The matrix is expected
        to be symmetric by construction; this method asserts that
        property and returns the raw matrix.

        Parameters
        ----------
        lp_result : scipy.optimize.OptimizeResult
            Result object returned by :pyfunc:`scipy.optimize.linprog`.

        Returns
        -------
        numpy.ndarray
            An (N × N) array of optimal generation rates (symmetric).
        """
        if lp_result.status != 0:
            raise ValueError(
                f"LP not solved to optimality (status={lp_result.status})."
            )

        start, end = self._offsets.generate, self._offsets.consume
        generate_vec = lp_result.x[start:end]
        mat = generate_vec.reshape(self.num_nodes, self.num_nodes)

        # Assert symmetry to catch potential bugs
        if not np.allclose(mat, mat.T, atol=1e-9):
            raise AssertionError("Generate matrix should be symmetric but is not.")

        return mat

    def total_swap_undirected(self, lp_result) -> float:
        """
        Return total swap rate (undirected, physical swaps):
        sum over k and unordered pairs i<j with k distinct of sigma(k,i,j).
        """
        if lp_result.status != 0:
            raise ValueError(
                f"LP not solved to optimality (status={lp_result.status})."
            )
        total = 0.0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                for k in range(self.num_nodes):
                    if k != i and k != j:
                        idx = self._offsets.swap + self._triple_to_idx((k, i, j))
                        total += lp_result.x[idx]
        return total

    def swap_matrix(self, lp_result) -> np.ndarray:
        """
        Return the symmetric matrix S with entries S[i, j] equal to the aggregate
        swap rate over all intermediate nodes k for the undirected pair (i, j).
        """
        if lp_result.status != 0:
            raise ValueError(
                f"LP not solved to optimality (status={lp_result.status})."
            )

        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                total = 0.0
                for k in range(self.num_nodes):
                    if k != i and k != j:
                        idx = self._offsets.swap + self._triple_to_idx((k, i, j))
                        total += lp_result.x[idx]
                mat[i, j] = total
                mat[j, i] = total
        return mat

    def total_generation_undirected(self, lp_result) -> float:
        """
        Return total generation in undirected units: sum over upper triangle of G.
        """
        G = self.generate_matrix(lp_result)
        return np.triu(G, k=0).sum()

    def total_consumption_undirected(self, lp_result) -> float:
        """
        Return total consumption in undirected units: sum over upper triangle of C.
        """
        C = self.consume_matrix(lp_result)
        return np.triu(C, k=0).sum()


class Offsets:
    def __init__(self, num_nodes: int):
        num_vars_generate = num_nodes**2
        num_vars_consume = num_nodes**2
        num_vars_desired_consume = num_nodes**2
        num_vars_swap = num_nodes**3
        num_vars_arrive = num_nodes**2
        num_vars_depart = num_nodes**2

        self.generate = 0
        self.consume = self.generate + num_vars_generate
        self.swap = self.consume + num_vars_consume
        self.arrive = self.swap + num_vars_swap
        self.depart = self.arrive + num_vars_arrive

        self.total_num_vars = self.depart + num_vars_depart


def create_cycle_adjacency_matrix(
    num_nodes: int, edge_weight: float = 10.0
) -> NDArray[np.float64]:
    """
    Create an undirected cycle graph adjacency matrix on `num_nodes` nodes.

    Nodes are labeled 0..num_nodes-1 and edges are:
      (i, (i+1) mod N) for all i

    The returned matrix is symmetric with 0 diagonal and `edge_weight` on
    the two cycle neighbors of each node.

    Args:
        num_nodes: Number of nodes in the cycle (N). Must be >= 3 for a non-degenerate cycle.
        edge_weight: Weight/capacity assigned to each undirected edge.

    Returns:
        numpy.ndarray: (N x N) adjacency/capacity matrix.
    """
    if num_nodes < 3:
        raise ValueError(f"cycle graph requires num_nodes >= 3 (got {num_nodes})")

    g = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        g[i, j] = edge_weight
        g[j, i] = edge_weight
    return g


def create_chain_adjacency_matrix(
    num_nodes: int, edge_weight: float = 10.0
) -> NDArray[np.float64]:
    """
    Create an undirected straight chain (path) graph adjacency matrix on `num_nodes` nodes.

    Nodes are labeled 0..num_nodes-1 and edges are:
      (i, i+1) for i = 0..num_nodes-2
    """
    if num_nodes < 2:
        raise ValueError(f"chain graph requires num_nodes >= 2 (got {num_nodes})")

    g = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes - 1):
        j = i + 1
        g[i, j] = edge_weight
        g[j, i] = edge_weight
    return g


def create_grid_adjacency_matrix(rows, cols) -> NDArray[np.float64]:
    """
    Create adjacency matrix for a grid graph with wraparound connections.
    Top connects to bottom, left connects to right (torus topology).

    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        numpy array: Adjacency matrix g[i,j] where g[i,j] = 1 if nodes i and j are connected
    """
    n_nodes = rows * cols
    g = np.zeros((n_nodes, n_nodes), dtype=float)

    def get_node_index(row, col):
        """Convert 2D grid coordinates to 1D node index"""
        return row * cols + col

    def get_neighbors(row, col):
        """Get the 4 neighbors of a node with wraparound"""
        neighbors = [
            ((row - 1) % rows, col),  # Up (with wraparound)
            ((row + 1) % rows, col),  # Down (with wraparound)
            (row, (col - 1) % cols),  # Left (with wraparound)
            (row, (col + 1) % cols),  # Right (with wraparound)
        ]
        return neighbors

    # Build adjacency matrix
    for row in range(rows):
        for col in range(cols):
            node_i = get_node_index(row, col)
            neighbors = get_neighbors(row, col)

            for neighbor_row, neighbor_col in neighbors:
                node_j = get_node_index(neighbor_row, neighbor_col)
                g[node_i, node_j] = 10

    return g


def create_random_adjacency_matrix(
    n_nodes, edge_probability=0.3, max_edge_weight=5, seed: int | None = None
) -> NDArray[np.float64]:
    """
    Create adjacency matrix for a random connected graph.

    Args:
        n_nodes: Number of nodes in the graph
        edge_probability: Probability that any given edge exists (default 0.3)
        max_edge_weight: Weight to assign to edges (default 5)
        seed: Optional seed for reproducible randomness. If None, uses NumPy global RNG.

    Returns:
        numpy array: Adjacency matrix g[i,j] where g[i,j] = rand(0..max_edge_weight) if nodes i and j are connected
    """
    g = np.zeros((n_nodes, n_nodes), dtype=float)

    rng = np.random.default_rng(seed) if seed is not None else np.random

    # Generate random edges with given probability
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_probability:
                weight = rng.uniform(0, max_edge_weight)
                g[i, j] = weight
                g[j, i] = weight

    # Find connected components using DFS
    def find_connected_components():
        visited = [False] * n_nodes
        components = []

        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(n_nodes):
                if g[node, neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor, component)

        for node in range(n_nodes):
            if not visited[node]:
                component = []
                dfs(node, component)
                components.append(component)

        return components

    # Connect all components to ensure the graph is connected
    components = find_connected_components()

    # Connect components pairwise
    while len(components) > 1:
        # Take first two components
        comp1 = components.pop(0)
        comp2 = components.pop(0)

        # Connect with a random edge between the components
        node1 = rng.choice(comp1)
        node2 = rng.choice(comp2)

        weight = rng.uniform(0, max_edge_weight)
        g[node1, node2] = weight
        g[node2, node1] = weight

        # Merge the components and add back to list
        merged_component = comp1 + comp2
        components.append(merged_component)

    return g


def create_sparse_symmetric_adjacency_matrix(
    n_nodes: int,
    edge_fraction: float = 0.2,
    max_edge_weight: float = 5.0,
    seed: int | None = None,
    min_positive_edges: int = 1,
) -> NDArray[np.float64]:
    """
    Create a symmetric demand matrix with an exact number of active undirected pairs.

    Unlike `create_random_adjacency_matrix`, this does not force connectivity. It is
    intended for sparse consumption demands where only a controlled fraction of node
    pairs should carry demand.
    """
    if n_nodes < 2:
        raise ValueError(f"sparse adjacency requires n_nodes >= 2 (got {n_nodes})")
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError(
            f"edge_fraction must lie in [0, 1] (got {edge_fraction})"
        )
    if min_positive_edges < 0:
        raise ValueError(
            f"min_positive_edges must be nonnegative (got {min_positive_edges})"
        )

    rng = np.random.default_rng(seed)
    g = np.zeros((n_nodes, n_nodes), dtype=float)
    undirected_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    total_pairs = len(undirected_pairs)
    target_edges = int(round(edge_fraction * total_pairs))
    num_active_edges = min(
        total_pairs, max(min_positive_edges if total_pairs > 0 else 0, target_edges)
    )

    if num_active_edges == 0:
        return g

    chosen_indices = rng.choice(total_pairs, size=num_active_edges, replace=False)
    for idx in np.atleast_1d(chosen_indices):
        i, j = undirected_pairs[int(idx)]
        weight = float(rng.uniform(0.0, max_edge_weight))
        g[i, j] = weight
        g[j, i] = weight

    return g


def undirected_capacity_sum(matrix: np.ndarray) -> float:
    """
    Sum over unique undirected pairs (upper triangle including diagonal).
    """
    return np.triu(matrix, k=0).sum()


def sparse_edge_list_to_matrix(entries: list[dict], num_nodes: int) -> np.ndarray:
    """
    Convert sparse undirected edge entries like {"src": i, "dst": j, "rate": r}
    into a symmetric dense matrix.
    """
    mat = np.zeros((num_nodes, num_nodes), dtype=float)
    for entry in entries:
        i = int(entry["src"])
        j = int(entry["dst"])
        rate = float(entry["rate"])
        mat[i, j] += rate
        mat[j, i] += rate
    return mat


def sparse_swap_list_to_matrix(entries: list[dict], num_nodes: int) -> np.ndarray:
    """
    Convert sparse swap entries like {"k": k, "src": i, "dst": j, "rate": r}
    into a symmetric dense matrix aggregated over the intermediate node k.
    """
    mat = np.zeros((num_nodes, num_nodes), dtype=float)
    for entry in entries:
        i = int(entry["src"])
        j = int(entry["dst"])
        rate = float(entry["rate"])
        mat[i, j] += rate
        mat[j, i] += rate
    return mat


def infer_topology_from_generation_capacity(
    generation_capacity_matrix: np.ndarray,
) -> str | None:
    """
    Infer simple topologies from the nonzero pattern of the generation-capacity matrix.
    Returns "cycle", "chain", or None if no supported inference is available.
    """
    if (
        generation_capacity_matrix.ndim != 2
        or generation_capacity_matrix.shape[0] != generation_capacity_matrix.shape[1]
    ):
        return None

    n = generation_capacity_matrix.shape[0]
    if n == 0:
        return None

    adj_bool = generation_capacity_matrix > 0
    degrees = adj_bool.sum(axis=1)
    G = build_generation_graph_nx(generation_capacity_matrix)

    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        return None

    if G.number_of_edges() == n and np.all(degrees == 2):
        return "cycle"

    if G.number_of_edges() == max(n - 1, 0):
        degree_counts = np.bincount(degrees.astype(int), minlength=3)
        if n == 1:
            return "chain"
        if degree_counts[1] == 2 and degree_counts[2] == max(n - 2, 0):
            return "chain"

    return None


def infer_num_nodes_from_solution_payload(payload: dict) -> int:
    problem = payload.get("problem", {})
    if "num_nodes" in problem:
        return int(problem["num_nodes"])

    inputs = payload.get("inputs", {})
    generation_capacity_matrix = inputs.get("generation_capacity_matrix")
    if generation_capacity_matrix is not None:
        return len(generation_capacity_matrix)

    max_node = -1
    decision_variables = payload.get("decision_variables", {})
    for key in ("generation_rate_g", "consumption_rate_c", "swap_rate_sigma"):
        for entry in decision_variables.get(key, []):
            max_node = max(max_node, int(entry["src"]), int(entry["dst"]))
    if max_node >= 0:
        return max_node + 1

    raise ValueError("Unable to infer num_nodes from solution payload.")


def build_generation_graph_nx(generation_graph: np.ndarray) -> nx.Graph:
    """
    Build an undirected NetworkX graph from the generation capacity matrix.
    Edge exists if capacity > 0 (weights are not used for hop count).
    """
    n = generation_graph.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if generation_graph[i, j] > 0:
                G.add_edge(i, j)
    return G


def get_apsp_for_generation_graph(generation_graph: np.ndarray):
    """
    Return cached all-pairs shortest-path lengths and paths for the given generation graph
    (undirected, hop weight = 1 per edge). Caches by the boolean edge pattern.
    """
    from functools import lru_cache

    n = generation_graph.shape[0]
    # Use boolean adjacency (undirected) as cache key bytes
    adj_bool = (generation_graph > 0).astype(np.uint8)
    key_bytes = adj_bool.tobytes()

    @lru_cache(maxsize=None)
    def _inner(n_key: int, adj_bytes: bytes):
        mat = np.frombuffer(adj_bytes, dtype=np.uint8).reshape(n_key, n_key)
        G = nx.Graph()
        G.add_nodes_from(range(n_key))
        for i in range(n_key):
            for j in range(i + 1, n_key):
                if mat[i, j]:
                    G.add_edge(i, j)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        paths = dict(nx.all_pairs_shortest_path(G))
        return lengths, paths

    return _inner(n, key_bytes)


def compute_path_based_metrics(
    consumption: np.ndarray,
    generation_graph: np.ndarray,
):
    """
    Compute path-based totals and per-edge loads using one shortest path per pair.

    Returns a dict with:
      - total_gen_path: sum c(i,j) * L(i,j)
      - total_swaps_path: sum c(i,j) * max(L(i,j)-1, 0)
      - unreachable_pairs: count of pairs with c(i,j)>0 and no path
      - path_edge_loads: NxN symmetric matrix with per-edge undirected loads
      - max_path_edge_load: maximum undirected per-edge load
    """
    N = consumption.shape[0]

    # Use cached APSP keyed by the generation graph topology
    lengths_dict, paths_dict = get_apsp_for_generation_graph(generation_graph)

    total_gen_path = 0.0
    total_swaps_path = 0.0
    unreachable_pairs = 0
    path_edge_loads = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            cij = consumption[i, j]
            if cij <= 0:
                continue
            if i not in lengths_dict or j not in lengths_dict[i]:
                unreachable_pairs += 1
                continue
            hops = lengths_dict[i][j]
            total_gen_path += cij * hops
            total_swaps_path += cij * max(hops - 1, 0)

            # Accumulate per-edge loads along one cached shortest path
            path = paths_dict[i][j]
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                path_edge_loads[u, v] += cij
                path_edge_loads[v, u] += cij

    # Compute maximum per-edge load (undirected) for the path-based scheme
    upper_loads = np.triu(path_edge_loads, k=1)
    max_path_edge_load = upper_loads.max() if upper_loads.size else 0.0

    return {
        "total_gen_path": total_gen_path,
        "total_swaps_path": total_swaps_path,
        "unreachable_pairs": unreachable_pairs,
        "path_edge_loads": path_edge_loads,
        "max_path_edge_load": max_path_edge_load,
    }


def plot_swaps_and_maxgen_vs_nodes(
    grid_sizes: list[int],
    trials: int = 3,
    edge_probability: float = 0.2,
    max_edge_weight: float = 7.0,
    gen_scale: float = 1.0,
    cons_scale: float = 1.0,
    save_path: str = "./output/plots/swaps_and_maxgen_vs_nodes.png",
    seed: int | None = None,
    retry_scales: list[float] | None = None,
):
    """
    For each grid size s in grid_sizes, run 'trials' random instances on an s x s torus,
    compute total swap counts and maximum per-edge generation for LP and path-based routing,
    and plot mean ± std vs N in two subplots.
    If the LP is infeasible for a trial, retry with increased generation scaling factors.
    """
    if seed is not None:
        np.random.seed(seed)
    if retry_scales is None:
        retry_scales = [1.25, 1.5, 2.0]
    Ns: list[int] = []
    swaps_lp_means: list[float] = []
    swaps_lp_stds: list[float] = []
    swaps_path_means: list[float] = []
    swaps_path_stds: list[float] = []

    maxgen_lp_means: list[float] = []
    maxgen_lp_stds: list[float] = []
    maxgen_path_means: list[float] = []
    maxgen_path_stds: list[float] = []

    # Ratios (Path/LP)
    ratio_swaps_means: list[float] = []
    ratio_swaps_stds: list[float] = []
    ratio_maxgen_means: list[float] = []
    ratio_maxgen_stds: list[float] = []
    # Ratios (Path/LP without max generation constraint)
    ratio_swaps_means_nocap: list[float] = []
    ratio_swaps_stds_nocap: list[float] = []
    ratio_maxgen_means_nocap: list[float] = []
    ratio_maxgen_stds_nocap: list[float] = []
    use_progress = should_use_progress()
    for s in tqdm(grid_sizes, desc="Grid sizes", disable=not use_progress):
        nrows, ncols = s, s
        N = nrows * ncols
        lp_swaps_vals = []
        path_swaps_vals = []
        lp_max_vals = []
        path_max_vals = []
        # Unconstrained LP (no max generation caps)
        lp_swaps_vals_nc = []
        lp_max_vals_nc = []
        for _ in tqdm(
            range(trials),
            desc=f"{s}x{s} trials",
            leave=False,
            disable=not use_progress,
        ):
            # Build graphs
            base_generation_graph = create_grid_adjacency_matrix(nrows, ncols).astype(
                float
            )
            consumption_graph = (
                create_random_adjacency_matrix(
                    N,
                    edge_probability=edge_probability,
                    max_edge_weight=max_edge_weight,
                ).astype(float)
                * cons_scale
            )
            attempt_scales = [gen_scale] + [gen_scale * r for r in retry_scales]
            solved = False
            for gs in attempt_scales:
                generation_graph = base_generation_graph * gs
                spec = LinearSpec(N)
                spec.add_generate_constraints(generation_graph)
                spec.add_consume_constraints(consumption_graph)
                result = spec.solve()
                if result.status != 0:
                    continue
                solved = True
                # LP metrics
                consume_matrix = spec.consume_matrix(result)
                generate_matrix = spec.generate_matrix(result)
                swaps_lp = spec.total_swap_undirected(result)
                upper_gen = np.triu(generate_matrix, k=1)
                max_gen_lp = upper_gen.max() if upper_gen.size else 0.0

                # LP without max generation constraints (only zero-out non-edges)
                spec_nc = LinearSpec(N)
                spec_nc.add_generate_zero_constraints(generation_graph)
                spec_nc.add_consume_constraints(consumption_graph)
                result_nc = spec_nc.solve()
                if result_nc.status == 0:
                    swaps_lp_nc = spec_nc.total_swap_undirected(result_nc)
                    generate_matrix_nc = spec_nc.generate_matrix(result_nc)
                    upper_gen_nc = np.triu(generate_matrix_nc, k=1)
                    max_gen_lp_nc = upper_gen_nc.max() if upper_gen_nc.size else 0.0
                else:
                    swaps_lp_nc = np.nan
                    max_gen_lp_nc = np.nan

                # Path-based metrics
                metrics = compute_path_based_metrics(consume_matrix, generation_graph)
                swaps_path = metrics["total_swaps_path"]
                path_edge_loads = metrics["path_edge_loads"]
                upper_path = np.triu(path_edge_loads, k=1)
                max_gen_path = upper_path.max() if upper_path.size else 0.0

                lp_swaps_vals.append(swaps_lp)
                path_swaps_vals.append(swaps_path)
                lp_max_vals.append(max_gen_lp)
                path_max_vals.append(max_gen_path)
                lp_swaps_vals_nc.append(swaps_lp_nc)
                lp_max_vals_nc.append(max_gen_lp_nc)
                break
            if not solved:
                # Infeasible across all retries
                lp_swaps_vals.append(np.nan)
                path_swaps_vals.append(np.nan)
                lp_max_vals.append(np.nan)
                path_max_vals.append(np.nan)
                lp_swaps_vals_nc.append(np.nan)
                lp_max_vals_nc.append(np.nan)
        Ns.append(N)
        # Aggregate with NaN guards
        lp_swaps_arr = np.array(lp_swaps_vals, dtype=float)
        path_swaps_arr = np.array(path_swaps_vals, dtype=float)
        lp_max_arr = np.array(lp_max_vals, dtype=float)
        path_max_arr = np.array(path_max_vals, dtype=float)
        lp_swaps_arr_nc = np.array(lp_swaps_vals_nc, dtype=float)
        lp_max_arr_nc = np.array(lp_max_vals_nc, dtype=float)
        # Compute ratios Path/LP with guards for division by zero/NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_swaps_arr = path_swaps_arr / lp_swaps_arr
            ratio_maxgen_arr = path_max_arr / lp_max_arr
            ratio_swaps_arr_nocap = path_swaps_arr / lp_swaps_arr_nc
            ratio_maxgen_arr_nocap = path_max_arr / lp_max_arr_nc

        finite = np.isfinite(ratio_swaps_arr)
        ratio_swaps_means.append(
            ratio_swaps_arr[finite].mean() if finite.any() else np.nan
        )
        ratio_swaps_stds.append(
            ratio_swaps_arr[finite].std(ddof=0) if finite.any() else np.nan
        )

        finite = np.isfinite(ratio_maxgen_arr)
        ratio_maxgen_means.append(
            ratio_maxgen_arr[finite].mean() if finite.any() else np.nan
        )
        ratio_maxgen_stds.append(
            ratio_maxgen_arr[finite].std(ddof=0) if finite.any() else np.nan
        )

        # Unconstrained ratios
        finite = np.isfinite(ratio_swaps_arr_nocap)
        ratio_swaps_means_nocap.append(
            ratio_swaps_arr_nocap[finite].mean() if finite.any() else np.nan
        )
        ratio_swaps_stds_nocap.append(
            ratio_swaps_arr_nocap[finite].std(ddof=0) if finite.any() else np.nan
        )

        finite = np.isfinite(ratio_maxgen_arr_nocap)
        ratio_maxgen_means_nocap.append(
            ratio_maxgen_arr_nocap[finite].mean() if finite.any() else np.nan
        )
        ratio_maxgen_stds_nocap.append(
            ratio_maxgen_arr_nocap[finite].std(ddof=0) if finite.any() else np.nan
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    # Top subplot: ratio Path/LP total swaps
    axes[0].errorbar(
        Ns,
        ratio_swaps_means,
        yerr=ratio_swaps_stds,
        fmt="-o",
        capsize=4,
        label="Path/LP swaps",
    )
    axes[0].errorbar(
        Ns,
        ratio_swaps_means_nocap,
        yerr=ratio_swaps_stds_nocap,
        fmt="-s",
        capsize=4,
        label="Path/LP (no max cap) swaps",
    )
    axes[0].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Path/LP total swaps")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    # Bottom subplot: ratio Path/LP max per-edge generation
    axes[1].errorbar(
        Ns,
        ratio_maxgen_means,
        yerr=ratio_maxgen_stds,
        fmt="-o",
        capsize=4,
        label="Path/LP max gen/edge",
    )
    axes[1].errorbar(
        Ns,
        ratio_maxgen_means_nocap,
        yerr=ratio_maxgen_stds_nocap,
        fmt="-s",
        capsize=4,
        label="Path/LP (no max cap) max gen/edge",
    )
    axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Number of nodes (N)")
    axes[1].set_ylabel("Path/LP max per-edge gen")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()
    fig.suptitle(
        "Ratios (Path/LP): Swaps and Max Per-Edge Generation vs Number of Nodes"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")


def run_trial_swaps(
    nrows: int,
    ncols: int,
    edge_probability: float = 0.2,
    max_edge_weight: float = 7.0,
    gen_scale: float = 1.0,
    cons_scale: float = 1.0,
    retry_scales: list[float] | None = None,
):
    """
    Run a single LP vs path-based trial and return (swaps_lp_undirected, swaps_path_undirected).
    If infeasible at the initial gen_scale, retry with increasing generation scales
    provided by retry_scales. Returns (nan, nan) if all attempts fail.
    """
    if retry_scales is None:
        retry_scales = []

    N = nrows * ncols
    # Build demand once so LP and path-based use the same instance
    base_generation_graph = create_grid_adjacency_matrix(nrows, ncols).astype(float)
    consumption_graph = (
        create_random_adjacency_matrix(
            N, edge_probability=edge_probability, max_edge_weight=max_edge_weight
        ).astype(float)
        * cons_scale
    )

    attempt_scales = [gen_scale] + [gen_scale * s for s in retry_scales]
    for gs in attempt_scales:
        generation_graph = base_generation_graph * gs

        spec = LinearSpec(N)
        spec.add_generate_constraints(generation_graph)
        spec.add_consume_constraints(consumption_graph)
        result = spec.solve()
        if result.status != 0:
            continue

        consume_matrix = spec.consume_matrix(result)
        swaps_lp = spec.total_swap_undirected(result)

        metrics = compute_path_based_metrics(consume_matrix, generation_graph)
        swaps_path = metrics["total_swaps_path"]

        return swaps_lp, swaps_path

    return np.nan, np.nan


def single_run():
    nrows, ncols = 5, 5
    generation_graph = create_grid_adjacency_matrix(nrows, ncols)
    consumption_graph = create_random_adjacency_matrix(
        nrows * ncols,
        edge_probability=0.2,
        max_edge_weight=7,
    )

    # Apply scaling to adjust feasibility
    generation_graph = generation_graph.astype(float) * gen_scale
    consumption_graph = consumption_graph.astype(float) * cons_scale

    # Diagnostic: Check supply vs demand balance
    total_generation_capacity = np.triu(generation_graph, k=0).sum()
    total_consumption_demand = np.triu(consumption_graph, k=0).sum()

    print("\n=== FEASIBILITY DIAGNOSTICS ===")
    print(f"Total generation capacity (undirected): {total_generation_capacity}")
    print(f"Total consumption demand (undirected): {total_consumption_demand}")
    print(
        f"Supply/Demand ratio: {total_generation_capacity / total_consumption_demand if total_consumption_demand > 0 else 'inf'}"
    )

    # Check if demand exceeds supply
    if total_consumption_demand > total_generation_capacity:
        print("\nWARNING: Total consumption demand exceeds generation capacity!")
        print("This will make the linear program infeasible.")
        print("Consider reducing edge_probability or increasing grid connections.")

    spec = LinearSpec(nrows * ncols)
    spec.add_generate_constraints(generation_graph)
    spec.add_consume_constraints(consumption_graph)
    result = spec.solve()
    print(f"\nGrid optimization status: {result.message}")
    print(f"Optimization status code: {result.status}")

    if result.status == 0:
        # Extract and display the optimal matrices
        consume_matrix = spec.consume_matrix(result)
        generate_matrix = spec.generate_matrix(result)

        print("\n=== OPTIMIZATION RESULTS ===")
        # Undirected totals using helpers
        total_gen_undirected = spec.total_generation_undirected(result)
        total_consume_undirected = spec.total_consumption_undirected(result)
        swap_total_undirected = spec.total_swap_undirected(result)

        print(f"Total optimal generation (undirected): {total_gen_undirected}")
        print(f"Total optimal consumption (undirected): {total_consume_undirected}")
        print(f"Total swap rate (undirected): {swap_total_undirected}")

        # Compute utilization against undirected capacity
        gen_capacity_undirected = undirected_capacity_sum(generation_graph)
        print(
            f"Generation utilization: {total_gen_undirected / gen_capacity_undirected:.2%}"
        )

        # === PATH-BASED COMPARISON ===
        metrics = compute_path_based_metrics(consume_matrix, generation_graph)
        total_gen_path = metrics["total_gen_path"]
        total_swaps_path = metrics["total_swaps_path"]
        unreachable_pairs = metrics["unreachable_pairs"]
        path_edge_loads = metrics["path_edge_loads"]

        # LP per-edge generation (undirected)
        lp_edge_gen = generate_matrix  # symmetric by construction
        # Compare only on upper triangle (exclude diagonal)
        upper = np.triu(np.ones_like(lp_edge_gen, dtype=bool), k=1)
        per_edge_abs_err = np.abs(lp_edge_gen - path_edge_loads)[upper]
        max_per_edge_abs_err = per_edge_abs_err.max() if per_edge_abs_err.size else 0.0
        l1_per_edge_err = per_edge_abs_err.sum()

        # Capacity violations under path routing (undirected check)
        capacities_upper = generation_graph[upper]
        path_upper = path_edge_loads[upper]
        cap_violations = int(np.sum(path_upper > capacities_upper + 1e-9))

        print("\n=== PATH-BASED COMPARISON ===")
        print(f"Total path-based generation (undirected): {total_gen_path}")
        print(f"Total path-based swaps (undirected): {total_swaps_path}")
        print(
            f"LP vs Path total generation ratio: {total_gen_undirected / total_gen_path if total_gen_path > 0 else float('inf'):.4f}"
        )
        print(
            f"LP vs Path total swaps ratio: {swap_total_undirected / total_swaps_path if total_swaps_path > 0 else float('inf'):.4f}"
        )
        print(f"Unreachable demand pairs (with c(i,j)>0): {unreachable_pairs}")
        print(f"Per-edge absolute error (max): {max_per_edge_abs_err}")
        print(f"Per-edge absolute error (L1 sum over edges): {l1_per_edge_err}")

        # Capacity feasibility of path-based routing
        if cap_violations > 0:
            print(
                f"WARNING: Path-based routing exceeds capacity on {cap_violations} undirected edges."
            )
        else:
            print("Path-based routing respects per-edge capacities (undirected check).")

    else:
        print("\n=== OPTIMIZATION FAILED ===")
        print(f"Status: {result.status} | Message: {result.message}")
        print(
            "Try adjusting the scaling constants in main() to improve feasibility or loosen demand:"
        )
        print("  - Increase gen_scale to raise per-edge generation capacities.")
        print("  - Decrease cons_scale to reduce per-pair consumption demands.")
        gen_capacity_undirected = undirected_capacity_sum(generation_graph)
        demand_undirected = undirected_capacity_sum(consumption_graph)
        print(f"Current undirected capacity: {gen_capacity_undirected}")
        print(f"Current undirected demand: {demand_undirected}")


def single_run_topology(
    *,
    topology: str,
    num_nodes: int,
    edge_weight: float = 10.0,
    gen_scale: float = 1.0,
    cons_scale: float = 1.0,
    cons_edge_fraction: float = 0.2,
    cons_max_edge_weight: float = 7.0,
    seed: int | None = None,
    objective: str = "min_sum_generate",
    swap_rate: float = 1.0,
    json_output_path: str | None = None,
    simulation_config_output_path: str | None = None,
    json_pretty: bool = True,
    json_emit_full_matrices: bool = True,
    output_mode: str = "json",
):
    valid_output_modes = {"json", "simulation-config", "json+simulation-config"}
    if output_mode not in valid_output_modes:
        raise ValueError(
            f"Unknown output_mode '{output_mode}'. Expected one of {sorted(valid_output_modes)}."
        )

    if topology == "cycle":
        generation_graph = create_cycle_adjacency_matrix(
            num_nodes, edge_weight=edge_weight
        )
        default_output_path = f"./output/lp/single_run_cycle_n{num_nodes}_solution.json"
    elif topology == "chain":
        generation_graph = create_chain_adjacency_matrix(
            num_nodes, edge_weight=edge_weight
        )
        default_output_path = f"./output/lp/single_run_chain_n{num_nodes}_solution.json"
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Expected 'cycle' or 'chain'."
        )

    if json_output_path is None:
        json_output_path = default_output_path
    if simulation_config_output_path is None:
        simulation_config_output_path = default_output_path.replace(".json", "_sim_config.json")

    consumption_graph = create_sparse_symmetric_adjacency_matrix(
        num_nodes,
        edge_fraction=cons_edge_fraction,
        max_edge_weight=cons_max_edge_weight,
        seed=seed,
        min_positive_edges=1,
    )

    # Apply scaling to adjust feasibility
    generation_graph = generation_graph.astype(float) * gen_scale
    consumption_graph = consumption_graph.astype(float) * cons_scale
    requested_simulation_input = build_simulation_input_config(
        generation_graph=generation_graph,
        consumption_graph=consumption_graph,
        swap_rates=np.full(num_nodes, float(swap_rate), dtype=float),
    )

    spec = LinearSpec(num_nodes)
    spec.add_generate_constraints(generation_graph)
    spec.add_consume_constraints(consumption_graph)
    result = spec.solve(objective=objective)
    print(f"\n{topology.capitalize()} optimization status: {result.message}")
    print(f"Optimization status code: {result.status}")

    # Always emit a JSON record, even on infeasible/failed solves, so runs are trackable
    payload: dict = {
        "problem": {
            "kind": f"{topology}_generation_sparse_consumption",
            "topology": str(topology),
            "num_nodes": int(num_nodes),
            "parameters": {
                "edge_weight": float(edge_weight),
                "gen_scale": float(gen_scale),
                "cons_scale": float(cons_scale),
                "cons_edge_fraction": float(cons_edge_fraction),
                "cons_max_edge_weight": float(cons_max_edge_weight),
                "seed": None if seed is None else int(seed),
                "objective": str(objective),
            },
        },
        "solver": {
            "method": "scipy.optimize.linprog(method='highs')",
            "status": int(result.status),
            "message": str(result.message),
            "success": bool(result.status == 0),
        },
        "totals_undirected": {},
        "decision_variables": {
            # Always present; may remain empty if solve failed
            # All decision variables are emitted as sparse undirected lists (i<j), nonzero only
            "generation_rate_g": [],
            "consumption_rate_c": [],
            "swap_rate_sigma": [],
            "arrive_rate_r_plus": [],
            "depart_rate_r_minus": [],
        },
    }

    # Include scaled input matrices for reproducibility/debugging
    if json_emit_full_matrices:
        payload["inputs"] = {
            "generation_capacity_matrix": generation_graph.tolist(),
            "consumption_demand_matrix": consumption_graph.tolist(),
        }

    if result.status != 0:
        print("\n=== OPTIMIZATION FAILED ===")
        print(f"Status: {result.status} | Message: {result.message}")
        gen_capacity_undirected = undirected_capacity_sum(generation_graph)
        demand_undirected = undirected_capacity_sum(consumption_graph)
        print(f"Current undirected capacity: {gen_capacity_undirected}")
        print(f"Current undirected demand: {demand_undirected}")

        payload["totals_undirected"] = {
            "generation_capacity": float(gen_capacity_undirected),
            "consumption_demand": float(demand_undirected),
        }

        if output_mode in {"json", "json+simulation-config"}:
            emit_single_run_lp_json(
                payload=payload,
                json_output_path=json_output_path,
                json_pretty=json_pretty,
            )
        if output_mode in {"simulation-config", "json+simulation-config"}:
            emit_simulation_input_json(
                payload=requested_simulation_input,
                json_output_path=simulation_config_output_path,
                json_pretty=json_pretty,
            )
        return

    print(
        f"\n=== OPTIMIZATION RESULTS ({topology.upper()} GEN + SPARSE CONSUMPTION) ==="
    )
    total_gen_undirected = spec.total_generation_undirected(result)
    total_consume_undirected = spec.total_consumption_undirected(result)
    swap_total_undirected = spec.total_swap_undirected(result)

    print(f"Total optimal generation (undirected): {total_gen_undirected}")
    print(f"Total optimal consumption (undirected): {total_consume_undirected}")
    print(f"Total swap rate (undirected): {swap_total_undirected}")

    # Populate totals section
    payload["totals_undirected"] = {
        "total_generation": float(total_gen_undirected),
        "total_consumption": float(total_consume_undirected),
        "total_swaps": float(swap_total_undirected),
        "generation_capacity": float(undirected_capacity_sum(generation_graph)),
        "consumption_demand": float(undirected_capacity_sum(consumption_graph)),
    }
    solution_simulation_input = build_lp_solution_simulation_input_config(
        spec=spec,
        lp_result=result,
    )

    # Helper to build nice, descriptive keys (kept for dense variables)
    # Extract full decision vector and map into descriptive structures
    x = result.x
    offsets = (
        spec._offsets
    )  # intentionally using internal offsets for full variable dump

    # Treat tiny values (including -0.0) as zero for sparse emission
    zero_tol = 1e-12

    # g(i,j): generation on undirected unique pairs (i<j), sparse list of {src,dst,rate}
    g_list: list[dict] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            idx = offsets.generate + spec._edge_to_idx((i, j))
            rate = float(x[idx])
            if abs(rate) > zero_tol:
                g_list.append({"src": i, "dst": j, "rate": rate})
    payload["decision_variables"]["generation_rate_g"] = g_list

    # c(i,j): consumption rate on undirected unique pairs (i<j), sparse list of {src,dst,rate}
    c_list: list[dict] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            idx = offsets.consume + spec._edge_to_idx((i, j))
            rate = float(x[idx])
            if abs(rate) > zero_tol:
                c_list.append({"src": i, "dst": j, "rate": rate})
    payload["decision_variables"]["consumption_rate_c"] = c_list

    # sigma(k,i,j): swaps, emitted for all k and undirected unique pairs (i<j), sparse list of {k,src,dst,rate}
    sigma_list: list[dict] = []
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                idx = offsets.swap + spec._triple_to_idx((k, i, j))
                rate = float(x[idx])
                if abs(rate) > zero_tol:
                    sigma_list.append({"k": k, "src": i, "dst": j, "rate": rate})
    payload["decision_variables"]["swap_rate_sigma"] = sigma_list

    # r+(i,j): arrive rate on undirected unique pairs (i<j), sparse list of {src,dst,rate}
    r_plus_list: list[dict] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            idx = offsets.arrive + spec._edge_to_idx((i, j))
            rate = float(x[idx])
            if abs(rate) > zero_tol:
                r_plus_list.append({"src": i, "dst": j, "rate": rate})
    payload["decision_variables"]["arrive_rate_r_plus"] = r_plus_list

    # r-(i,j): depart rate on undirected unique pairs (i<j), sparse list of {src,dst,rate}
    r_minus_list: list[dict] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            idx = offsets.depart + spec._edge_to_idx((i, j))
            rate = float(x[idx])
            if abs(rate) > zero_tol:
                r_minus_list.append({"src": i, "dst": j, "rate": rate})
    payload["decision_variables"]["depart_rate_r_minus"] = r_minus_list

    # Emit JSON
    if output_mode in {"json", "json+simulation-config"}:
        emit_single_run_lp_json(
            payload=payload,
            json_output_path=json_output_path,
            json_pretty=json_pretty,
        )
    if output_mode in {"simulation-config", "json+simulation-config"}:
        emit_simulation_input_json(
            payload=solution_simulation_input,
            json_output_path=simulation_config_output_path,
            json_pretty=json_pretty,
        )


def single_run_cycle(
    num_nodes: int = 25,
    edge_weight: float = 10.0,
    gen_scale: float = 1.0,
    cons_scale: float = 1.0,
    cons_edge_probability: float = 0.1,
    cons_max_edge_weight: float = 7.0,
    seed: int | None = None,
    objective: str = "min_sum_generate",
    swap_rate: float = 1.0,
    json_output_path: str | None = "./output/lp/single_run_cycle_solution.json",
    simulation_config_output_path: str | None = None,
    json_pretty: bool = True,
    json_emit_full_matrices: bool = True,
    output_mode: str = "json",
):
    return single_run_topology(
        topology="cycle",
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        gen_scale=gen_scale,
        cons_scale=cons_scale,
        cons_edge_fraction=cons_edge_probability,
        cons_max_edge_weight=cons_max_edge_weight,
        seed=seed,
        objective=objective,
        swap_rate=swap_rate,
        json_output_path=json_output_path,
        simulation_config_output_path=simulation_config_output_path,
        json_pretty=json_pretty,
        json_emit_full_matrices=json_emit_full_matrices,
        output_mode=output_mode,
    )


def write_requested_lp_results(
    *,
    output_dir: str = "./output/lp_results",
    edge_weight: float = 10.0,
    gen_scale: float = 10.0,
    cons_scale: float = 1.0,
    cons_edge_fraction: float = 0.2,
    cons_max_edge_weight: float = 7.0,
    objective: str = "min_sum_generate",
    swap_rate: float = 1.0,
    output_mode: str = "json",
) -> None:
    runs = [
        {"topology": "cycle", "num_nodes": 10, "seed": 10},
        {"topology": "cycle", "num_nodes": 15, "seed": 15},
        {"topology": "chain", "num_nodes": 5, "seed": 5},
        {"topology": "chain", "num_nodes": 10, "seed": 20},
    ]

    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        topology = run["topology"]
        num_nodes = run["num_nodes"]
        seed = run["seed"]
        output_path = os.path.join(output_dir, f"{topology}_n{num_nodes}_lp_solution.json")
        sim_config_output_path = os.path.join(
            output_dir, f"{topology}_n{num_nodes}_sim_config.json"
        )
        print(
            f"\n--- Writing LP result for {topology} topology with n={num_nodes} "
            f"(consumption density={cons_edge_fraction:.0%}, seed={seed}) ---"
        )
        single_run_topology(
            topology=topology,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
            gen_scale=gen_scale,
            cons_scale=cons_scale,
            cons_edge_fraction=cons_edge_fraction,
            cons_max_edge_weight=cons_max_edge_weight,
            seed=seed,
            objective=objective,
            swap_rate=swap_rate,
            json_output_path=output_path,
            simulation_config_output_path=sim_config_output_path,
            json_pretty=True,
            json_emit_full_matrices=True,
            output_mode=output_mode,
        )


if __name__ == "__main__":
    # # Scaling constants to adjust feasibility (edit these to tune)
    # gen_scale = 1.0  # increase to raise per-edge generation capacities
    # cons_scale = 1.0  # decrease to reduce per-pair consumption demands

    # # Run experiment: swaps and max per-edge generation vs number of nodes (square grids s x s)
    # grid_sizes = [3, 4, 5, 6, 7]
    # trials = 10
    # default_edge_probability = 0.2
    # default_max_edge_weight = 7.0
    # plot_swaps_and_maxgen_vs_nodes(
    #     grid_sizes,
    #     trials=trials,
    #     edge_probability=default_edge_probability,
    #     max_edge_weight=default_max_edge_weight,
    #     gen_scale=gen_scale,
    #     cons_scale=cons_scale,
    #     save_path="./output/plots/swaps_and_maxgen_vs_nodes.png",
    #     seed=None,
    #     retry_scales=[1.5, 3.0, 6.0],
    # )

    # Write LP result files for the requested larger cycle and chain topologies.
    write_requested_lp_results(
        output_dir="./output/lp_results",
        edge_weight=10.0,
        gen_scale=10.0,
        cons_scale=1.0,
        cons_edge_fraction=0.2,
        cons_max_edge_weight=7.0,
        objective="min_sum_generate",
        swap_rate=1.0,
        output_mode="json+simulation-config",
    )
