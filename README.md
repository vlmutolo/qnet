# qbp-sim

Continuous-time Gillespie simulation of quantum backpressure routing.

Always use `uv` here.

- Use `uv sync` to create or refresh the environment.
- Use `uv run ...` to run commands inside the project environment.
- Use `uv add ...` to add runtime dependencies.
- Use `uv add --group dev ...` to add development-only dependencies.
- Do not use `pip install`, `python -m venv`, or ad hoc global package installs for this repo.

## Quick start

```bash
uv sync --all-groups
uv run qbp-sim run --config configs/example.json --until 50
uv run qbp-sim example --until 50 --seed 0
uv run qbp-sim example --until 50 --trace output/traces/example.vortex
uv run qbp-sim example --until 50 --snapshots output/snapshots/example.jsonl.zst
uv run qbp-sim replay --trace output/traces/example.vortex
uv run qbp-sim analyze --snapshots output/snapshots/example.jsonl.zst
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo --dry-run
uv run pytest
```

The user manual lives in `docs/manual.typ` and is rendered to `docs/qbp-sim.pdf`.
Build it with:

```bash
typst compile docs/manual.typ docs/qbp-sim.pdf
```

## Current model

The simulator is now a continuous-time analogue of the paper's virtual/physical backpressure architecture, not an exact reproduction of the paper's slot-based theorem.

There is also a separate LP benchmark module at `qbp_sim.lp`. That module represents a different,
global average-rate optimization model and should be treated as an optimal comparison target for
aggregate outputs, not as the same model or control law as the Gillespie backpressure simulator.
Its simulation-config output now emits the LP's optimal generation, consumption, and per-node
swap rates using the same Pydantic schema as the simulator input layer. When generating LP
instances from `qbp_sim.lp`, the `swap_rate` argument is also used as the uniform per-node
swap cap `K_i` from the paper's LP; the default LP cap is `10.0` so the stock examples are not
trivially infeasible.

State is stored as dense symmetric matrices:

- `Q[x, y]`: Bell-pair inventory
- `D[x, y]`: virtual demand backlog (`gamma`)
- `alpha[x, y]`: scarcity pressure
- `H^R[x, y]`: queued virtual service requests awaiting physical realization
- `H^mu[i, y, z]`: queued virtual swap requests awaiting physical realization

JSON simulation input is validated with Pydantic and currently only needs:

- `generation_rates`
- `consumption_rates`
- `swap_rates`

It may also include `capacity_headroom`, an optional multiplier for controllable capacity and
opportunity rates. The default is `1.01`. At runtime, `capacity_headroom` scales
`generation_rates`, `swap_rates`, and service hazards, while leaving demand arrivals from
`consumption_rates` unchanged. `instant_service_fulfillment` and `instant_swap_fulfillment` are
opt-in experimental modes, defaulting to `false`, that use a local deterministic frontier after
sampled events. The frontier realizes at most one pending physical service on the active edge
before considering pending swaps that use that edge; deterministic realizations are logged as
separate zero-time `physical_service` or `physical_swap` events. Input may also include an optional
virtual swap scheduler policy:

```json
{
  "virtual_swap_policy": {
    "mode": "limited_info_bp",
    "k": 4,
    "memory": 8
  }
}
```

The default policy is `bp`, which preserves the original backpressure behavior: each node
scans every swap candidate and picks the largest positive virtual pressure. The
`limited_info_bp` policy is a limited-information variant. On each actor refresh, a node
samples `k` fresh swap candidates, re-scores its remembered candidates, keeps the best `memory`
candidates it has seen, and exposes only the best positive remembered candidate to the virtual
swap clock. The centralized simulator still stores dense matrices, but this policy constrains
which entries the simulated actor is allowed to inspect while choosing swaps.
Set `memory` to `0` for a query-only variant that picks from the fresh `k` candidates without
carrying remembered candidates across refreshes.

The `max_min` policy implements the HotNets path-oblivious balancing baseline. It is not a
backpressure policy: it chooses physical swaps from current inventory `Q`, not scarcity pressure
`alpha`. At a swap opportunity, node `i` marks `(y,z)` preferred when both input inventories
`Q[i,y]` and `Q[i,z]` exceed the output inventory `Q[y,z]` by more than one, then chooses the
preferred output edge with minimum `Q[y,z]`. Demand admission in this mode moves requests directly
to pending physical service without updating `alpha`, so comparisons against backpressure separate
the inventory-balancing baseline from the demand-aware control law.
The `limited_info_max_min` policy applies the same `k`-query, `memory`-candidate restriction to
max-min swap selection.

The runtime topology is inferred from `generation_rates > 0`, and service hazards default to the
requested `consumption_rates` on each demanded pair times `capacity_headroom`.

The Gillespie engine currently supports these event families:

- demand arrivals
- direct Bell-pair generation
- virtual service requests
- direct service-request admission for max-min mode
- virtual swap requests
- max-min inventory-balancing physical swaps
- physical service realizations
- physical swap realizations

Backpressure virtual service and swap decisions evolve `gamma`/`D` and `alpha` without checking
current Bell-pair availability. Physical service and swap events then realize those queued requests
from `H^R` and `H^mu` whenever inventory is available.

The service ratio tracked in snapshots is currently:

- `service_ratio = services_completed / demand_arrivals`

This lets analysis treat it as a sampled time series independent of the event trace.

## Architecture

The simulator is split into two systems:

- an event producer that inspects current state and samples the next Gillespie event
- an event applier that mutates state from a concrete event record

That separation is intentional. It keeps event generation and state mutation decoupled, and it makes replay straightforward because a saved event log can be fed directly into the applier without resampling randomness.

Experiment commands write durable per-run artifacts as `events.vortex`, `simulation_config.json`,
and `run_metadata.json`. Plots produced by those commands use sampled snapshots in memory, but do
not save snapshot checkpoint files. Snapshots are still available as an explicit lower-level CLI
artifact type:

- traces are per-event logs for replay
- snapshots are sampled aggregates for analysis and plotting

Simulation, replay, and analysis are therefore mostly orthogonal modes.

Experiment sweeps can be specified with `ExperimentMatrixConfig`, exported from the root
`qbp_sim` facade and from `qbp_sim.experiments`. The matrix expands Cartesian-product axes into
concrete cases and can be inspected or run through the generic matrix command:

```json
{
  "topologies": ["chain", "grid"],
  "graph_sizes": [9, 16],
  "consumption_edge_fractions": [null, 0.25],
  "headrooms": [1.0, 1.01],
  "policies": [
    {"mode": "bp", "label": "bp"},
    {"mode": "max_min"},
    {"mode": "limited_info_bp", "k": 2, "memory": 2},
    {"mode": "limited_info_max_min", "k": 2, "memory": 2}
  ],
  "seed_offsets": [0, 100],
  "until_time": 100000.0,
  "sample_every": 1000,
  "trace_time_mode": "none"
}
```

The core axes are topology, graph size, consumption-demand sparsity, capacity headroom, swap
policy, generation/consumption/swap scale factors, stochastic seed replicate, trace precision,
trace time mode, and instant physical-fulfillment modes.

```bash
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo
```

Each matrix case writes `lp_solution.json`, `simulation_config.json`, `events.vortex`, and
`run_metadata.json`; the matrix output directory also gets `summary.csv`.

## Layout

The package is organized into presentation-facing layers:

- `src/qbp_sim/facade.py`: public Python API for consumer scripts
- `src/qbp_sim/core/`: simulator types, indexing helpers, numba kernels, event applier, event producer, instant-fulfillment frontier, run loop, and replay helper
- `src/qbp_sim/io/`: concrete event records, event traces, and sampled snapshots
- `src/qbp_sim/config/`: Pydantic JSON input config and runtime conversion
- `src/qbp_sim/analysis/`: snapshot summaries and Altair chart helpers
- `src/qbp_sim/experiments/`: LP-derived experiment setup, run metadata, and plots
- `src/qbp_sim/cli/`: command-line parser and command handlers
- `src/qbp_sim/lp/`: separate LP benchmark model used for comparison against backpressure summaries
- `src/qbp_sim/examples.py`: built-in example networks
- `tests/`: subsystem-organized unit, replay, trace, experiment, LP, import-contract, and gated stochastic simulator tests

The root `qbp_sim` package intentionally exports only the consumer-facing facade: JSON config
types, experiment matrix types, `RunOptions`, `RunOutput`, `run_simulation`, `replay_trace`, and
`build_four_node_example_config`. Developer-level internals remain importable from explicit
subpackages such as `qbp_sim.core`, `qbp_sim.io`, `qbp_sim.simulator`, and `qbp_sim.lp`.

The compatibility modules `qbp_sim.simulator`, `qbp_sim.events`, `qbp_sim.trace`, and
`qbp_sim.snapshots` remain available for existing imports. The root-level `linear.py` entry point
has intentionally moved to `qbp_sim.lp`.

## Typical commands

Run the built-in four-node example:

```bash
uv run qbp-sim example --until 100 --seed 1
```

Run from a JSON config:

```bash
uv run qbp-sim run --config configs/run-001.json --until 100
```

Run the built-in example with the limited-information virtual swap policy:

```bash
uv run qbp-sim example --until 100 --virtual-swap-policy limited-info-bp --swap-k 4 --swap-memory 8
```

Run the built-in example with local instantaneous physical fulfillment enabled:

```bash
uv run qbp-sim example --until 100 --instant-service-fulfillment --instant-swap-fulfillment
```

Compare full-information and limited-information policies on an LP-derived cycle and plot
`service_ratio = services_completed / demand_arrivals` from `t=0`:

```bash
uv run qbp-sim limited-info-service-ratio --n 5 --until 1000 --limited-policies 1:1 2:2 4:4 --plot-start-time 100
```

Compare service-gap decay under different capacity-headroom multipliers:

```bash
uv run qbp-sim headroom-service-ratio --n 16 --until 10000 --headrooms 1.0 1.01 1.05
```

Run a generic experiment matrix:

```bash
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo
```

Write a compact Vortex event trace:

```bash
uv run qbp-sim example --until 100 --trace output/traces/run-001.vortex
```

Columnar event traces (`.vortex` and `.parquet`) store `time`, `total_rate`, and
`event_rate` as `float32` by default. Use `--trace-float-precision float64` when
you need full precision, or `float16` only for short runs whose time/rate values
fit in fp16 range.
Use `--trace-time-mode none` to omit `time`, `total_rate`, and `event_rate` from JSONL,
Parquet, and Vortex traces. That mode preserves event order and state-transition replay, but
does not preserve the real simulation clock unless analysis supplies an external final time.

Write sampled snapshots:

```bash
uv run qbp-sim example --until 100 --sample-every 100 --snapshots output/snapshots/run-001.jsonl.zst
```

Replay a saved event trace:

```bash
uv run qbp-sim replay --trace output/traces/run-001.vortex
```

Analyze snapshots and print a summary:

```bash
uv run qbp-sim analyze --snapshots output/snapshots/run-001.jsonl.zst
```

Plot service ratio from snapshots:

```bash
uv run qbp-sim analyze \
  --snapshots output/snapshots/run-001.jsonl.zst \
  --plot-metric service_ratio \
  --plot-out output/plots/service-ratio.png
```

Sync the environment, including dev dependencies:

```bash
uv sync --all-groups
```

Add a new dependency:

```bash
uv add scipy
```

Run the test suite:

```bash
uv run pytest
```

Run the gated stochastic checks:

```bash
QBP_SIM_RUN_GATED_TESTS=1 uv run pytest
```

Build distributable Python artifacts:

```bash
uv build
```

GitHub Actions also includes a `build` workflow that runs tests, renders `docs/qbp-sim.pdf`,
builds the wheel/source distribution, and uploads those files as workflow artifacts.

Long-running simulations and analysis jobs should be managed with `pueue` so they can continue
outside the active shell session. Generated experiment outputs belong under `output/`, which is
ignored by Git.
