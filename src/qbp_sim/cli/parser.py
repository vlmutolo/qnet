
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from qbp_sim.core.types import (
    VIRTUAL_SWAP_POLICY_GLOBAL,
    VIRTUAL_SWAP_POLICY_MAX_MIN,
    VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY,
    GillespieQBPConfig,
    VirtualSwapPolicy,
)



def _add_virtual_swap_policy_args(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--virtual-swap-policy",
        choices=[
            VIRTUAL_SWAP_POLICY_GLOBAL,
            VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY,
            VIRTUAL_SWAP_POLICY_MAX_MIN,
            "power-of-k-memory",
            "max-min",
        ],
        default=None,
        help="Override the virtual swap scheduler policy.",
    )
    command_parser.add_argument(
        "--swap-k",
        type=int,
        default=None,
        help="Fresh candidate swaps queried per actor refresh for power_of_k_memory.",
    )
    command_parser.add_argument(
        "--swap-memory",
        type=int,
        default=None,
        help="Best candidate swaps remembered per actor for power_of_k_memory.",
    )


def _normalize_virtual_swap_policy_mode(mode: str) -> str:
    return mode.replace("-", "_")


def _apply_virtual_swap_policy_args(config: GillespieQBPConfig, args: argparse.Namespace) -> GillespieQBPConfig:
    requested_mode = args.virtual_swap_policy
    requested_k = args.swap_k
    requested_memory = args.swap_memory
    if requested_mode is None and requested_k is None and requested_memory is None:
        return config

    mode = (
        _normalize_virtual_swap_policy_mode(requested_mode)
        if requested_mode is not None
        else VIRTUAL_SWAP_POLICY_POWER_OF_K_MEMORY
    )
    current = config.virtual_swap_policy
    return replace(
        config,
        virtual_swap_policy=VirtualSwapPolicy(
            mode=mode,
            k=current.k if requested_k is None else requested_k,
            memory=current.memory if requested_memory is None else requested_memory,
        ),
    )


def _add_instant_service_fulfillment_arg(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--instant-service-fulfillment",
        action="store_true",
        help=(
            "Use the local deterministic frontier to immediately realize one pending physical "
            "service when inventory and H^R meet on the same edge."
        ),
    )


def _add_instant_swap_fulfillment_arg(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--instant-swap-fulfillment",
        action="store_true",
        help=(
            "Use the local deterministic frontier to immediately realize pending physical swaps "
            "instead of sampling physical swap hazards."
        ),
    )


def _apply_instant_service_fulfillment_arg(
    config: GillespieQBPConfig,
    args: argparse.Namespace,
) -> GillespieQBPConfig:
    instant_service = getattr(args, "instant_service_fulfillment", False)
    instant_swap = getattr(args, "instant_swap_fulfillment", False)
    if not instant_service and not instant_swap:
        return config
    return replace(
        config,
        instant_service_fulfillment=config.instant_service_fulfillment or instant_service,
        instant_swap_fulfillment=config.instant_swap_fulfillment or instant_swap,
    )


def _parse_limited_policy(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("x", ":").replace(",", ":")
    parts = normalized.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("limited policy must be formatted as K:M, for example 4:8.")
    try:
        k = int(parts[0])
        memory = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limited policy K and M must be integers.") from exc
    if k <= 0 or memory < 0:
        raise argparse.ArgumentTypeError("limited policy K must be positive and M must be non-negative.")
    return k, memory


def _max_events_limit(value: int) -> int | None:
    return None if value <= 0 else value


def _add_trace_float_precision_arg(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--trace-float-precision",
        choices=("float16", "float32", "float64"),
        default="float32",
        help=(
            "Floating-point precision for columnar event traces. "
            "float32 is the default; float16 is only suitable when time/rate values stay within fp16 range."
        ),
    )


def _add_trace_time_mode_arg(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--trace-time-mode",
        choices=("full", "none"),
        default="full",
        help=(
            "Controls persisted timing/rate trace fields. "
            "Use 'none' to omit time, total_rate, and event_rate for smaller traces that cannot reconstruct simulation time."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qbp-sim",
        description="Continuous-time Gillespie simulation for quantum backpressure networks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a Gillespie simulation from a JSON config file.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="INFILE",
        help="Path to a JSON simulation config.",
    )
    run_parser.add_argument("--until", type=float, default=50.0, help="Stop at simulation time T.")
    run_parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many events even if the time limit has not been reached; use 0 for no event cap.",
    )
    run_parser.add_argument("--seed", type=int, default=0, help="Seed for NumPy RNG.")
    run_parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Record an aggregate snapshot every N events.",
    )
    run_parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write every event to a trace file. Use .vortex for compact Vortex, .parquet for buffered Parquet, or .jsonl.zst for compressed JSONL.",
    )
    _add_trace_float_precision_arg(run_parser)
    _add_trace_time_mode_arg(run_parser)
    run_parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write sampled aggregate snapshots into a Zstandard-compressed JSONL file.",
    )
    _add_virtual_swap_policy_args(run_parser)
    _add_instant_service_fulfillment_arg(run_parser)
    _add_instant_swap_fulfillment_arg(run_parser)

    example_parser = subparsers.add_parser(
        "example",
        help="Run the four-node counterexample network as a Gillespie simulation.",
    )
    example_parser.add_argument("--until", type=float, default=50.0, help="Stop at simulation time T.")
    example_parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many events even if the time limit has not been reached; use 0 for no event cap.",
    )
    example_parser.add_argument("--seed", type=int, default=0, help="Seed for NumPy RNG.")
    example_parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Record an aggregate snapshot every N events.",
    )
    example_parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write every event to a trace file. Use .vortex for compact Vortex, .parquet for buffered Parquet, or .jsonl.zst for compressed JSONL.",
    )
    _add_trace_float_precision_arg(example_parser)
    _add_trace_time_mode_arg(example_parser)
    example_parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write sampled aggregate snapshots into a Zstandard-compressed JSONL file.",
    )
    _add_virtual_swap_policy_args(example_parser)
    _add_instant_service_fulfillment_arg(example_parser)
    _add_instant_swap_fulfillment_arg(example_parser)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a trace file against the built-in four-node example config.",
    )
    replay_parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        metavar="INFILE",
        help="Read a .vortex, .parquet, or .jsonl.zst event trace to replay.",
    )
    replay_parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Record an aggregate snapshot every N replayed events.",
    )
    replay_parser.add_argument(
        "--until",
        type=float,
        default=None,
        help="Optional terminal time to restore after replaying the final logged event.",
    )
    replay_parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write sampled aggregate snapshots during replay into a compressed JSONL file.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a snapshot file and optionally plot a metric.",
    )
    analyze_parser.add_argument(
        "--snapshots",
        type=Path,
        required=True,
        metavar="INFILE",
        help="Read a Zstandard-compressed JSONL snapshot file.",
    )
    analyze_parser.add_argument(
        "--plot-metric",
        type=str,
        default=None,
        help="Snapshot metric to plot, for example service_ratio or total_backlog.",
    )
    analyze_parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write the requested plot to an image file.",
    )

    cycle_parser = subparsers.add_parser(
        "cycle-service-ratio",
        help="Solve LP-derived cycle instances, run BP on them, and plot service-gap decay.",
    )
    cycle_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Cycle sizes to generate and simulate.",
    )
    cycle_parser.add_argument(
        "--burn-in",
        type=float,
        default=0.0,
        help="Warm the BP simulator to this time, then reset counters/time before the measured run.",
    )
    cycle_parser.add_argument(
        "--until",
        type=float,
        default=100.0,
        help="Stop each BP simulation at this simulation time.",
    )
    cycle_parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop each BP simulation after this many events if it has not reached --until; use 0 for no event cap.",
    )
    cycle_parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Record a snapshot every N events for plotting.",
    )
    cycle_parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed; each cycle run uses seed_base + n.",
    )
    cycle_parser.add_argument(
        "--gen-scale",
        type=float,
        default=10.0,
        help="Scale factor applied to cycle-edge generation capacities before solving the LP.",
    )
    cycle_parser.add_argument(
        "--cons-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sampled consumption demands before solving the LP.",
    )
    cycle_parser.add_argument(
        "--cons-edge-fraction",
        type=float,
        default=None,
        help=(
            "Override the fraction of end-to-end consumption pairs given nonzero demand. "
            "By default the sweep uses a size-aware rule with about n/2 active demand pairs."
        ),
    )
    cycle_parser.add_argument(
        "--swap-rate",
        type=float,
        default=100.0,
        help="Uniform per-node swap cap used in the LP instances.",
    )
    cycle_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cycle_service_ratio"),
        metavar="OUTDIR",
        help="Directory to write LP outputs, simulation configs, compact Vortex event traces, and run metadata.",
    )
    _add_trace_float_precision_arg(cycle_parser)
    _add_trace_time_mode_arg(cycle_parser)
    _add_instant_service_fulfillment_arg(cycle_parser)
    _add_instant_swap_fulfillment_arg(cycle_parser)
    cycle_parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("output/plots/cycle_service_ratio_gap.png"),
        metavar="OUTFILE",
        help="Path for the combined Altair-rendered plot.",
    )

    headroom_parser = subparsers.add_parser(
        "headroom-service-ratio",
        help=(
            "Solve one LP-derived cycle instance, apply capacity headroom to generation, "
            "swap, and service rates, run BP, and plot service-gap decay."
        ),
    )
    headroom_parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="Cycle size to generate, solve, and simulate.",
    )
    headroom_parser.add_argument(
        "--headrooms",
        type=float,
        nargs="+",
        default=[1.0, 1.01, 1.05],
        help=(
            "Capacity multipliers applied at runtime to LP-derived generation rates, "
            "swap rates, and service opportunity rates. Demand rates are not scaled."
        ),
    )
    headroom_parser.add_argument(
        "--burn-in",
        type=float,
        default=0.0,
        help="Warm the BP simulator to this time, then reset counters/time before the measured run.",
    )
    headroom_parser.add_argument(
        "--until",
        type=float,
        default=100.0,
        help="Stop each BP simulation at this simulation time.",
    )
    headroom_parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop each BP simulation after this many events if it has not reached --until; use 0 for no event cap.",
    )
    headroom_parser.add_argument(
        "--sample-every",
        type=int,
        default=500,
        help="Record a snapshot every N events for plotting.",
    )
    headroom_parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed for the LP and each BP run.",
    )
    headroom_parser.add_argument(
        "--gen-scale",
        type=float,
        default=10.0,
        help="Scale factor applied to cycle-edge generation capacities before solving the LP.",
    )
    headroom_parser.add_argument(
        "--cons-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sampled consumption demands before solving the LP.",
    )
    headroom_parser.add_argument(
        "--cons-edge-fraction",
        type=float,
        default=None,
        help=(
            "Override the fraction of end-to-end consumption pairs given nonzero demand. "
            "By default the experiment uses a size-aware rule with about n/2 active demand pairs."
        ),
    )
    headroom_parser.add_argument(
        "--swap-rate",
        type=float,
        default=100.0,
        help="Uniform per-node swap cap used in the LP instance.",
    )
    headroom_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/headroom_service_ratio"),
        metavar="OUTDIR",
        help="Directory to write LP outputs, headroom configs, compact Vortex event traces, and run metadata.",
    )
    _add_trace_float_precision_arg(headroom_parser)
    _add_trace_time_mode_arg(headroom_parser)
    _add_instant_service_fulfillment_arg(headroom_parser)
    _add_instant_swap_fulfillment_arg(headroom_parser)
    headroom_parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("output/plots/headroom_service_ratio_gap.png"),
        metavar="OUTFILE",
        help="Path for the combined Altair-rendered plot.",
    )

    limited_parser = subparsers.add_parser(
        "limited-info-service-ratio",
        help="Compare full-info BP against limited-info power-of-k-memory policies using service_ratio from t=0.",
    )
    limited_parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Cycle size to generate, solve, and simulate.",
    )
    limited_parser.add_argument(
        "--limited-policies",
        type=_parse_limited_policy,
        nargs="+",
        default=[(1, 1), (2, 2), (4, 4)],
        metavar="K:M",
        help="Limited-info policies to compare, formatted as K:M. Example: --limited-policies 1:1 2:2 4:8",
    )
    limited_parser.add_argument(
        "--until",
        type=float,
        default=1_000.0,
        help="Stop each simulation at this simulation time. Measurements start at t=0.",
    )
    limited_parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop each simulation after this many events if it has not reached --until; use 0 for no event cap.",
    )
    limited_parser.add_argument(
        "--sample-every",
        type=int,
        default=1_000,
        help="Record a snapshot every N events for plotting.",
    )
    limited_parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed for the LP and each BP run.",
    )
    limited_parser.add_argument(
        "--gen-scale",
        type=float,
        default=10.0,
        help="Scale factor applied to cycle-edge generation capacities before solving the LP.",
    )
    limited_parser.add_argument(
        "--cons-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sampled consumption demands before solving the LP.",
    )
    limited_parser.add_argument(
        "--cons-edge-fraction",
        type=float,
        default=None,
        help=(
            "Override the fraction of end-to-end consumption pairs given nonzero demand. "
            "By default the experiment uses a size-aware rule with about n/2 active demand pairs."
        ),
    )
    limited_parser.add_argument(
        "--swap-rate",
        type=float,
        default=100.0,
        help="Uniform per-node swap cap used in the LP instance.",
    )
    limited_parser.add_argument(
        "--headroom",
        type=float,
        default=1.01,
        help=(
            "Capacity multiplier applied at runtime to LP-derived generation rates, "
            "swap rates, and service opportunity rates. Demand rates are not scaled."
        ),
    )
    limited_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/limited_info_service_ratio"),
        metavar="OUTDIR",
        help="Directory to write LP outputs, policy configs, compact Vortex event traces, and run metadata.",
    )
    _add_trace_float_precision_arg(limited_parser)
    _add_trace_time_mode_arg(limited_parser)
    _add_instant_service_fulfillment_arg(limited_parser)
    _add_instant_swap_fulfillment_arg(limited_parser)
    limited_parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("output/plots/limited_info_service_ratio.png"),
        metavar="OUTFILE",
        help="Path for the full-info vs limited-info service_ratio plot.",
    )
    limited_parser.add_argument(
        "--plot-start-time",
        type=float,
        default=100.0,
        help="Positive lower bound for the log-scaled simulation-time axis.",
    )

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Run or inspect a JSON experiment matrix.",
    )
    matrix_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="INFILE",
        help="Path to a JSON experiment matrix config.",
    )
    matrix_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/matrix"),
        metavar="OUTDIR",
        help="Directory for one subdirectory per resolved matrix case plus summary.csv.",
    )
    matrix_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded cases as JSON without solving LPs or running simulations.",
    )
    return parser
