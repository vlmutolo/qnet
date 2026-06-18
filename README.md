# qbp-sim

`qbp-sim` runs continuous-time Gillespie simulations for quantum-network routing experiments based on backpressure-style control.

Use `uv` for all project commands.

```bash
uv sync --all-groups
uv run qbp-sim --help
uv run pytest
```

## Quick Start

Run the built-in four-node example:

```bash
uv run qbp-sim example --until 50 --seed 0
```

Run a JSON config:

```bash
uv run qbp-sim run --config docs/examples/basic_config.json --until 50
```

Write and replay an event trace:

```bash
uv run qbp-sim example --until 50 --trace output/traces/example.vortex
uv run qbp-sim replay --trace output/traces/example.vortex
```

Run a matrix dry run:

```bash
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo --dry-run
```

Build the manual:

```bash
typst compile docs/manual.typ docs/qbp-sim.pdf
```

## Python Examples

The `examples/` directory contains runnable scripts with hard-coded typed configs. Each script writes traces and PNG plots under `output/examples/`.

```bash
uv run examples/01_basic_run_service_ratio.py
uv run examples/02_compare_policies.py
uv run examples/03_headroom_sweep.py
uv run examples/04_instant_fulfillment_modes.py
uv run examples/05_trace_summary_and_event_mix.py
uv run examples/06_lp_derived_config.py
```

The examples use Polars for analysis. Vortex traces open through `vx.open(path).to_polars()`. Parquet traces open through `pl.scan_parquet(path)`.

## Model

The simulator uses continuous time and samples one stochastic event at a time with a Gillespie direct method. It records concrete event traces that can be replayed without resampling randomness.

Simulation state is stored in dense symmetric matrices:

- `Q[x, y]`: Bell-pair inventory.
- `D[x, y]`: virtual demand backlog, also called `gamma`.
- `alpha[x, y]`: backpressure scarcity signal.
- `H^R[x, y]`: pending physical service requests.
- `H^mu[i, y, z]`: pending physical swap requests.

Simulation configs use symmetric matrices and per-node swap rates:

- `generation_rates[x][y]`: Bell-pair generation rate on edge `(x, y)`.
- `consumption_rates[x][y]`: demand arrival rate for pair `(x, y)`.
- `swap_rates[i]`: swap opportunity rate at node `i`.
- `capacity_headroom`: multiplier on generation, swap, and service opportunity rates. Demand arrivals are unchanged. The default is `1.01`.
- `virtual_swap_policy`: swap-selection policy.

Supported policies:

- `bp`: full-information backpressure. Each node scans all swap candidates and chooses the largest positive virtual pressure.
- `limited_info_bp`: backpressure with `k` queried candidates and `memory` remembered candidates per node.
- `max_min`: path-oblivious inventory balancing. It uses physical inventory `Q` rather than scarcity `alpha`.
- `limited_info_max_min`: max-min with `k` queried candidates and `memory` remembered candidates per node.

The simulator samples these event families:

- demand arrivals
- Bell-pair generation
- virtual service requests
- direct service admission for max-min mode
- virtual swap requests
- max-min physical swaps
- physical service realizations
- physical swap realizations

Backpressure virtual events update `D`, `alpha`, `H^R`, and `H^mu`. Physical realization events consume inventory from `Q` when inventory is available.

The primary service metric is:

```text
service_ratio = services_completed / demand_arrivals
```

## LP Benchmark

The `qbp_sim.lp` module solves a global average-rate linear program. It produces LP outputs and simulator configs for benchmark instances. The LP is an aggregate-rate optimizer; the Gillespie simulator is a stochastic event process.

LP-derived configs use the same `SimulationInputConfig` schema as hand-written configs. The LP `swap_rate` argument sets the uniform per-node swap cap used while solving the LP and the corresponding simulator swap opportunity rate.

## Traces, Snapshots, And Metadata

Traces are per-event logs for replay and detailed analysis. Supported event trace formats are:

- `vortex`: compact Vortex columnar trace.
- `parquet`: Parquet columnar trace.
- `jsonl_zst`: Zstandard-compressed JSONL trace.

Use an explicit trace format when the filename does not imply one:

```bash
uv run qbp-sim example \
  --until 100 \
  --trace output/traces/run-001.parquet \
  --trace-format parquet
```

Columnar traces store `time`, `total_rate`, and `event_rate` as `float32` by default. Use `--trace-float-precision float64` for higher precision. Use `--trace-time-mode none` to omit time and rate columns from Vortex, Parquet, and JSONL traces.

Snapshots are sampled aggregate summaries. They are useful for quick plots and summaries:

```bash
uv run qbp-sim example \
  --until 100 \
  --sample-every 100 \
  --snapshots output/snapshots/run-001.jsonl.zst
```

Experiment commands write:

- `lp_solution.json`
- `simulation_config.json`
- `events.<format>`
- `run_metadata.json`

Matrix runs also write `summary.csv`.

## Experiment Matrices

`ExperimentMatrixConfig` expands Cartesian products of experiment settings:

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
  "trace_format": "vortex",
  "trace_time_mode": "none"
}
```

Run the matrix:

```bash
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo
```

Inspect expanded cases without running them:

```bash
uv run qbp-sim matrix --config docs/examples/matrix_config.json --output-dir output/matrix_demo --dry-run
```

## Public Python API

Import public types and helpers from the package root:

```python
from qbp_sim import (
    RunOptions,
    TraceFormat,
    VirtualSwapPolicyConfig,
    build_four_node_example_config,
    run_simulation,
)

config = build_four_node_example_config()
output = run_simulation(
    config,
    RunOptions(
        until_time=50.0,
        seed=0,
        trace_path="output/traces/python_example.parquet",
        trace_format=TraceFormat.PARQUET,
    ),
)
print(output.service_ratio)
```

Public root exports:

- `SimulationInputConfig`
- `VirtualSwapPolicyConfig`
- `ExperimentMatrixConfig`
- `ExperimentPolicyConfig`
- `ExperimentMatrixCase`
- `RunOptions`
- `RunOutput`
- `VirtualSwapPolicyMode`
- `TopologyName`
- `TraceFloatPrecision`
- `TraceFormat`
- `TraceTimeMode`
- `load_simulation_config`
- `load_experiment_matrix_config`
- `build_four_node_example_config`
- `run_simulation`
- `replay_trace`

Developer modules remain available under `qbp_sim.core`, `qbp_sim.io`, `qbp_sim.analysis`, `qbp_sim.experiments`, and `qbp_sim.lp`.

## Common Commands

Run the built-in example with limited-information backpressure:

```bash
uv run qbp-sim example --until 100 --virtual-swap-policy limited-info-bp --swap-k 4 --swap-memory 8
```

Run with deterministic instant fulfillment:

```bash
uv run qbp-sim example --until 100 --instant-service-fulfillment --instant-swap-fulfillment
```

Compare limited-information policies:

```bash
uv run qbp-sim limited-info-service-ratio \
  --n 5 \
  --until 1000 \
  --limited-policies 1:1 2:2 4:4 \
  --plot-start-time 100
```

Compare capacity headroom values:

```bash
uv run qbp-sim headroom-service-ratio --n 16 --until 10000 --headrooms 1.0 1.01 1.05
```

Analyze snapshots:

```bash
uv run qbp-sim analyze \
  --snapshots output/snapshots/run-001.jsonl.zst \
  --plot-metric service_ratio \
  --plot-out output/plots/service-ratio.png
```

Build package artifacts:

```bash
uv build
```

Run the full test suite:

```bash
uv run pytest
QBP_SIM_RUN_GATED_TESTS=1 uv run pytest
```

## Repository Layout

- `src/qbp_sim/facade.py`: public Python API.
- `src/qbp_sim/core/`: simulator state, kernels, event producer, event applier, frontier realization, run loop, and replay.
- `src/qbp_sim/io/`: event records, event traces, and snapshots.
- `src/qbp_sim/config/`: Pydantic config models and runtime conversion.
- `src/qbp_sim/analysis/`: snapshot summaries and Altair chart helpers.
- `src/qbp_sim/experiments/`: LP-derived experiment setup, metadata, matrix execution, and plots.
- `src/qbp_sim/cli/`: command-line parser and command handlers.
- `src/qbp_sim/lp/`: LP benchmark model.
- `src/qbp_sim/examples.py`: built-in example networks.
- `examples/`: runnable Python examples.
- `docs/`: Typst manual and JSON examples.
- `tests/`: unit, replay, trace, experiment, LP, import-contract, and gated stochastic tests.

Generated runs belong under `output/`, which is ignored by Git. Long simulations and analysis jobs should run under `pueue`.
