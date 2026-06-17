from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from qbp_sim.config import SimulationInputConfig
from qbp_sim.experiments import load_experiment_matrix_config


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_docs_example_json_configs_validate() -> None:
    basic = SimulationInputConfig.from_json_file(DOCS / "examples" / "basic_config.json")
    matrix = load_experiment_matrix_config(DOCS / "examples" / "matrix_config.json")

    assert basic.num_nodes == 4
    assert matrix.case_count == 2


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_typst_manual_compiles(tmp_path) -> None:
    output_path = tmp_path / "qbp-sim.pdf"
    subprocess.run(
        ["typst", "compile", str(DOCS / "manual.typ"), str(output_path)],
        cwd=ROOT,
        check=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
