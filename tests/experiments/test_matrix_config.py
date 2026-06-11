from __future__ import annotations

import json

import pytest

from qbp_sim.experiments import (
    ExperimentMatrixConfig,
    ExperimentPolicyConfig,
    load_experiment_matrix_config,
)


def test_experiment_matrix_defaults_to_full_and_limited_policy_cases() -> None:
    config = ExperimentMatrixConfig()
    cases = config.cases()

    assert config.case_count == 2
    assert [case.policy_label for case in cases] == ["full info", "limited k=1, m=1"]
    assert {case.topology for case in cases} == {"cycle"}
    assert {case.n_nodes for case in cases} == {4}
    assert {case.capacity_headroom for case in cases} == {1.01}
    assert cases[0].virtual_swap_policy.mode == "global"
    assert cases[1].virtual_swap_policy.mode == "power_of_k_memory"
    assert cases[1].virtual_swap_policy.k == 1
    assert cases[1].virtual_swap_policy.memory == 1
    assert {case.trace_time_mode for case in cases} == {"full"}


def test_experiment_matrix_expands_cartesian_product_and_slugs_cases() -> None:
    config = ExperimentMatrixConfig(
        topologies=["cycle", "grid"],
        graph_sizes=[9, 16],
        consumption_edge_fractions=[None, 0.25],
        headrooms=[1.0, 1.01],
        policies=[
            ExperimentPolicyConfig(mode="global"),
            ExperimentPolicyConfig(mode="power-of-k-memory", k=2, memory=3),
        ],
        seed_base=10,
        seed_offsets=[0, 100],
        instant_service_fulfillment=[False, True],
        instant_swap_fulfillment=[False],
        until_time=500.0,
        sample_every=50,
    )

    cases = config.cases()

    assert len(cases) == 2 * 2 * 2 * 2 * 2 * 2 * 2
    assert config.case_count == len(cases)
    assert {case.topology for case in cases} == {"cycle", "grid"}
    assert {case.n_nodes for case in cases} == {9, 16}
    assert {case.consumption_edge_fraction for case in cases} == {None, 0.25}
    assert {case.capacity_headroom for case in cases} == {1.0, 1.01}
    assert {case.seed for case in cases} == {19, 26, 119, 126}
    assert {case.instant_service_fulfillment for case in cases} == {False, True}
    assert all(case.until_time == 500.0 for case in cases)
    assert all(case.sample_every == 50 for case in cases)
    assert any(
        case.slug == "grid_n16_headroom_1p01_cons_frac_0p25_limited_k2_m3_seed126"
        for case in cases
    )


def test_experiment_matrix_loads_json_config(tmp_path) -> None:
    config_path = tmp_path / "matrix.json"
    config_path.write_text(
        json.dumps(
            {
                "topologies": ["chain"],
                "graph_sizes": [10],
                "consumption_edge_fractions": [0.2],
                "headrooms": [1.01, 1.05],
                "policies": [
                    {"mode": "global", "label": "full information"},
                    {"mode": "power_of_k_memory", "k": 5, "memory": 0},
                ],
                "seed_base": 7,
                "seed_offsets": [0],
                "until_time": 100000.0,
                "max_events": None,
                "sample_every": 1000,
                "trace_float_precision": "float32",
                "trace_time_mode": "none",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_experiment_matrix_config(config_path)
    cases = loaded.cases()

    assert len(cases) == 4
    assert {case.policy_label for case in cases} == {"full information", "limited k=5, m=0"}
    assert {case.capacity_headroom for case in cases} == {1.01, 1.05}
    assert all(case.topology == "chain" for case in cases)
    assert all(case.seed == 17 for case in cases)
    assert all(case.max_events is None for case in cases)
    assert all(case.trace_time_mode == "none" for case in cases)


def test_experiment_matrix_rejects_invalid_policy_and_sparsity() -> None:
    with pytest.raises(Exception, match="global policy must not set k"):
        ExperimentPolicyConfig(mode="global", k=1)

    with pytest.raises(Exception, match="positive k"):
        ExperimentPolicyConfig(mode="power_of_k_memory", k=0, memory=1)

    with pytest.raises(Exception, match="consumption_edge_fractions"):
        ExperimentMatrixConfig(consumption_edge_fractions=[0.0])
