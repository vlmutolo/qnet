
from __future__ import annotations

from qbp_sim.analysis import plot_snapshot_metric, summarize_snapshots
from qbp_sim.cli.parser import (
    _apply_instant_service_fulfillment_arg,
    _apply_virtual_swap_policy_args,
    _build_parser,
    _max_events_limit,
)
from qbp_sim.config import load_simulation_config
from qbp_sim.core.replay import replay_event_stream
from qbp_sim.core.simulator import GillespieQBPSimulator
from qbp_sim.examples import build_four_node_counterexample
from qbp_sim.experiments import (
    plot_cycle_service_ratio_runs,
    plot_headroom_runs,
    plot_limited_info_service_ratio_runs,
    run_cycle_service_ratio_experiment,
    run_headroom_experiment,
    run_limited_info_service_ratio_experiment,
)
from qbp_sim.io.snapshots import SnapshotReader, SnapshotWriter
from qbp_sim.io.trace import open_event_trace_reader, open_event_trace_writer

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        input_config = load_simulation_config(args.config)
        config = _apply_instant_service_fulfillment_arg(
            _apply_virtual_swap_policy_args(input_config.to_runtime_config(), args),
            args,
        )
        sim = GillespieQBPSimulator(config=config, seed=args.seed)
        if args.trace is None and args.snapshots is None:
            result = sim.run(until_time=args.until, max_events=_max_events_limit(args.max_events), sample_every=args.sample_every)
        elif args.trace is not None and args.snapshots is None:
            with open_event_trace_writer(
                args.trace,
                float_precision=args.trace_float_precision,
                time_mode=args.trace_time_mode,
            ) as trace_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                )
        elif args.trace is None and args.snapshots is not None:
            with SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
                    sample_every=args.sample_every,
                    snapshot_writer=snapshot_writer,
                )
        else:
            with open_event_trace_writer(
                args.trace,
                float_precision=args.trace_float_precision,
                time_mode=args.trace_time_mode,
            ) as trace_writer, SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
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
        config = _apply_instant_service_fulfillment_arg(
            _apply_virtual_swap_policy_args(build_four_node_counterexample(), args),
            args,
        )
        sim = GillespieQBPSimulator(config=config, seed=args.seed)
        if args.trace is None and args.snapshots is None:
            result = sim.run(until_time=args.until, max_events=_max_events_limit(args.max_events), sample_every=args.sample_every)
        elif args.trace is not None and args.snapshots is None:
            with open_event_trace_writer(
                args.trace,
                float_precision=args.trace_float_precision,
                time_mode=args.trace_time_mode,
            ) as trace_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
                    sample_every=args.sample_every,
                    trace_writer=trace_writer,
                )
        elif args.trace is None and args.snapshots is not None:
            with SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
                    sample_every=args.sample_every,
                    snapshot_writer=snapshot_writer,
                )
        else:
            with open_event_trace_writer(
                args.trace,
                float_precision=args.trace_float_precision,
                time_mode=args.trace_time_mode,
            ) as trace_writer, SnapshotWriter(args.snapshots) as snapshot_writer:
                result = sim.run(
                    until_time=args.until,
                    max_events=_max_events_limit(args.max_events),
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
            with open_event_trace_reader(args.trace) as trace_reader:
                result = replay_event_stream(
                    config=config,
                    events=trace_reader,
                    sample_every=args.sample_every,
                    final_time=args.until,
                )
        else:
            with open_event_trace_reader(args.trace) as trace_reader, SnapshotWriter(args.snapshots) as snapshot_writer:
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
            burn_in_time=args.burn_in,
            until_time=args.until,
            max_events=_max_events_limit(args.max_events),
            sample_every=args.sample_every,
            seed_base=args.seed_base,
            gen_scale=args.gen_scale,
            cons_scale=args.cons_scale,
            cons_edge_fraction=args.cons_edge_fraction,
            swap_rate=args.swap_rate,
            trace_float_precision=args.trace_float_precision,
            trace_time_mode=args.trace_time_mode,
            instant_service_fulfillment=args.instant_service_fulfillment,
            instant_swap_fulfillment=args.instant_swap_fulfillment,
        )
        plot_cycle_service_ratio_runs(runs, args.plot_out)
        print("Cycle service-gap experiment")
        for run in runs:
            summary = run.summary
            print(
                f"n={run.n_nodes} final_time={summary.final_time:.3f} "
                f"final_service_ratio={summary.final_service_ratio:.6f} "
                f"samples={summary.num_snapshots} trace_path={run.trace_path} metadata_path={run.metadata_path}"
            )
        print(f"\nWrote plot to {args.plot_out}")
    if args.command == "headroom-service-ratio":
        runs = run_headroom_experiment(
            n_nodes=args.n,
            capacity_headrooms=args.headrooms,
            output_dir=args.output_dir,
            burn_in_time=args.burn_in,
            until_time=args.until,
            max_events=_max_events_limit(args.max_events),
            sample_every=args.sample_every,
            seed_base=args.seed_base,
            gen_scale=args.gen_scale,
            cons_scale=args.cons_scale,
            cons_edge_fraction=args.cons_edge_fraction,
            swap_rate=args.swap_rate,
            trace_float_precision=args.trace_float_precision,
            trace_time_mode=args.trace_time_mode,
            instant_service_fulfillment=args.instant_service_fulfillment,
            instant_swap_fulfillment=args.instant_swap_fulfillment,
        )
        plot_headroom_runs(runs, args.plot_out)
        print("Headroom service-gap experiment")
        for run in runs:
            summary = run.summary
            print(
                f"n={run.n_nodes} capacity_headroom={run.capacity_headroom:g} "
                f"final_time={summary.final_time:.3f} "
                f"final_service_ratio={summary.final_service_ratio:.6f} "
                f"samples={summary.num_snapshots} trace_path={run.trace_path} metadata_path={run.metadata_path}"
            )
        print(f"\nWrote plot to {args.plot_out}")
    if args.command == "limited-info-service-ratio":
        runs = run_limited_info_service_ratio_experiment(
            n_nodes=args.n,
            limited_policies=args.limited_policies,
            output_dir=args.output_dir,
            until_time=args.until,
            max_events=_max_events_limit(args.max_events),
            sample_every=args.sample_every,
            seed_base=args.seed_base,
            gen_scale=args.gen_scale,
            cons_scale=args.cons_scale,
            cons_edge_fraction=args.cons_edge_fraction,
            swap_rate=args.swap_rate,
            capacity_headroom=args.headroom,
            trace_float_precision=args.trace_float_precision,
            trace_time_mode=args.trace_time_mode,
            instant_service_fulfillment=args.instant_service_fulfillment,
            instant_swap_fulfillment=args.instant_swap_fulfillment,
        )
        plot_limited_info_service_ratio_runs(runs, args.plot_out, plot_start_time=args.plot_start_time)
        print("Limited-info service-ratio experiment")
        for run in runs:
            summary = run.summary
            print(
                f"n={run.n_nodes} policy={run.policy_label} "
                f"final_time={summary.final_time:.3f} "
                f"final_service_ratio={summary.final_service_ratio:.6f} "
                f"samples={summary.num_snapshots} trace_path={run.trace_path} metadata_path={run.metadata_path}"
            )
        print(f"\nWrote plot to {args.plot_out}")
