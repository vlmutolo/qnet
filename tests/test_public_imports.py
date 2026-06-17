from __future__ import annotations


def test_public_package_exports_consumer_facade_only() -> None:
    import qbp_sim
    from qbp_sim import (
        ExperimentMatrixConfig,
        RunOptions,
        SimulationInputConfig,
        TraceTimeMode,
        VirtualSwapPolicyMode,
        build_four_node_example_config,
        replay_trace,
        run_simulation,
    )

    assert ExperimentMatrixConfig is not None
    assert RunOptions is not None
    assert SimulationInputConfig is not None
    assert VirtualSwapPolicyMode.MAX_MIN == "max_min"
    assert TraceTimeMode.NONE == "none"
    assert build_four_node_example_config is not None
    assert replay_trace is not None
    assert run_simulation is not None
    assert "GillespieQBPSimulator" not in qbp_sim.__all__
    assert "QBPEventApplier" not in qbp_sim.__all__
    assert "open_event_trace_writer" not in qbp_sim.__all__


def test_developer_subpackage_imports_remain_available() -> None:
    from qbp_sim.lp import LinearSpec, build_simulation_input_config
    from qbp_sim.lp import linear
    from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator
    from qbp_sim.trace import open_event_trace_writer

    assert LinearSpec is linear.LinearSpec
    assert build_simulation_input_config is linear.build_simulation_input_config
    assert GillespieQBPConfig is not None
    assert GillespieQBPSimulator is not None
    assert open_event_trace_writer is not None
