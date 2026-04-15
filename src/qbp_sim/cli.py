from __future__ import annotations

import argparse
from pathlib import Path

from qbp_sim.analysis import plot_snapshot_metric, summarize_snapshots
from qbp_sim.config import load_simulation_config
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.experiments import plot_cycle_service_ratio_runs, run_cycle_service_ratio_experiment
from qbp_sim.simulator import GillespieQBPSimulator, replay_event_stream
from qbp_sim.snapshots import SnapshotReader, SnapshotWriter
from qbp_sim.trace import EventTraceReader, EventTraceWriter


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
        default=200_000,
        help="Stop after this many events even if the time limit has not been reached.",
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
        help="Write every event as JSONL into a Zstandard-compressed trace file.",
    )
    run_parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write sampled aggregate snapshots into a Zstandard-compressed JSONL file.",
    )

    example_parser = subparsers.add_parser(
        "example",
        help="Run the four-node counterexample network as a Gillespie simulation.",
    )
    example_parser.add_argument("--until", type=float, default=50.0, help="Stop at simulation time T.")
    example_parser.add_argument(
        "--max-events",
        type=int,
        default=200_000,
        help="Stop after this many events even if time limit has not been reached.",
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
        help="Write every event as JSONL into a Zstandard-compressed trace file.",
    )
    example_parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        metavar="OUTFILE",
        help="Write sampled aggregate snapshots into a Zstandard-compressed JSONL file.",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a trace file against the built-in four-node example config.",
    )
    replay_parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        metavar="INFILE",
        help="Read a Zstandard-compressed JSONL event trace to replay.",
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
        "--until",
        type=float,
        default=100.0,
        help="Stop each BP simulation at this simulation time.",
    )
    cycle_parser.add_argument(
        "--max-events",
        type=int,
        default=200_000,
        help="Stop each BP simulation after this many events if it has not reached --until.",
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
        help="Directory to write LP outputs and BP snapshots.",
    )
    cycle_parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("output/plots/cycle_service_ratio_gap.png"),
        metavar="OUTFILE",
        help="Path for the combined Altair-rendered plot.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        input_config = load_simulation_config(args.config)
        sim = GillespieQBPSimulator(config=input_config.to_runtime_config(), seed=args.seed)
        if args.trace is None and args.snapshots is None:
            result = sim.run(until_time=args.until, max_events=args.max_events, sample_every=args.sample_every)
        elif args.trace is not None and args.snapshots is None:
            with EventTraceWriter(args.trace) as trace_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                )
        elif args.trace is None and args.snapshots is not None:
            with SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    snapshot_writer=snapshot_writer,
                )
        else:
            with EventTraceWriter(args.trace) as trace_writer, SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                    snapshot_writer=snapshot_writer,
                )
        print(result.format_summary())
        if result.sample_times:
            print("\nSnapshots")
            for idx, time_value in enumerate(result.sample_times[-5:]):
                backlog = result.total_backlog_samples[-5:][idx]
                inventory = result.total_inventory_samples[-5:][idx]
                scarcity = result.total_alpha_samples[-5:][idx]
                print(
                    f"t={time_value:.3f} backlog={backlog} inventory={inventory} scarcity={scarcity}"
                )

    if args.command == "example":
        config = build_four_node_counterexample()
        sim = GillespieQBPSimulator(config=config, seed=args.seed)
        if args.trace is None and args.snapshots is None:
            result = sim.run(until_time=args.until, max_events=args.max_events, sample_every=args.sample_every)
        elif args.trace is not None and args.snapshots is None:
            with EventTraceWriter(args.trace) as trace_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                )
        elif args.trace is None and args.snapshots is not None:
            with SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    snapshot_writer=snapshot_writer,
                )
        else:
            with EventTraceWriter(args.trace) as trace_writer, SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=args.max_events,
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                    snapshot_writer=snapshot_writer,
                )
        print(result.format_summary())
        if result.sample_times:
            print("\nSnapshots")
            for idx, time_value in enumerate(result.sample_times[-5:]):
                backlog = result.total_backlog_samples[-5:][idx]
                inventory = result.total_inventory_samples[-5:][idx]
                scarcity = result.total_alpha_samples[-5:][idx]
                print(
                    f"t={time_value:.3f} backlog={backlog} inventory={inventory} scarcity={scarcity}"
                )
    if args.command == "replay":
        config = build_four_node_counterexample()
        if args.snapshots is None:
            with EventTraceReader(args.trace) as trace_reader:
                result = replay_event_stream(
                    config=config,
                    events=trace_reader,
                    sample_every=args.sample_every,
                    final_time=args.until,
                )
        else:
            with EventTraceReader(args.trace) as trace_reader, SnapshotWriter(args.snapshots) as snapshot_writer:
                result = replay_event_stream(
                    config=config,
                    events=trace_reader,
                    sample_every=args.sample_every,
                    final_time=args.until,
                    snapshot_writer=snapshot_writer,
                )
        print(result.format_summary())
        if result.sample_times:
            print("\nSnapshots")
            for idx, time_value in enumerate(result.sample_times[-5:]):
                backlog = result.total_backlog_samples[-5:][idx]
                inventory = result.total_inventory_samples[-5:][idx]
                scarcity = result.total_alpha_samples[-5:][idx]
                print(
                    f"t={time_value:.3f} backlog={backlog} inventory={inventory} scarcity={scarcity}"
                )
    if args.command == "analyze":
        with SnapshotReader(args.snapshots) as snapshot_reader:
            snapshots = list(snapshot_reader)
        summary = summarize_snapshots(snapshots)
        print(summary.format_summary())
        if args.plot_metric is not None:
            if args.plot_out is None:
                parser.error("--plot-out is required when --plot-metric is provided.")
            plot_snapshot_metric(snapshots=snapshots, metric=args.plot_metric, output_path=args.plot_out)
            print(f"\nWrote plot to {args.plot_out}")
    if args.command == "cycle-service-ratio":
        runs = run_cycle_service_ratio_experiment(
            cycle_sizes=args.sizes,
            output_dir=args.output_dir,
            until_time=args.until,
            max_events=args.max_events,
            sample_every=args.sample_every,
            seed_base=args.seed_base,
            gen_scale=args.gen_scale,
            cons_scale=args.cons_scale,
            cons_edge_fraction=args.cons_edge_fraction,
            swap_rate=args.swap_rate,
        )
        plot_cycle_service_ratio_runs(runs, args.plot_out)
        print("Cycle service-gap experiment")
        for run in runs:
            summary = run.summary
            print(
                f"n={run.n_nodes} final_time={summary.final_time:.3f} "
                f"final_service_ratio={summary.final_service_ratio:.6f} "
                f"snapshots={summary.num_snapshots} snapshots_path={run.snapshots_path}"
            )
        print(f"\nWrote plot to {args.plot_out}")
