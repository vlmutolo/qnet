
from __future__ import annotations


def test_public_package_imports_remain_available() -> None:
    from qbp_sim import GillespieQBPSimulator as PackageSimulator
    from qbp_sim.config import SimulationInputConfig
    from qbp_sim.simulator import GillespieQBPConfig, GillespieQBPSimulator
    from qbp_sim.trace import open_event_trace_writer

    assert PackageSimulator is GillespieQBPSimulator
    assert GillespieQBPConfig is not None
    assert SimulationInputConfig is not None
    assert open_event_trace_writer is not None


def test_lp_package_import_is_available() -> None:
    from qbp_sim.lp import LinearSpec, build_simulation_input_config
    from qbp_sim.lp import linear

    assert LinearSpec is linear.LinearSpec
    assert build_simulation_input_config is linear.build_simulation_input_config
