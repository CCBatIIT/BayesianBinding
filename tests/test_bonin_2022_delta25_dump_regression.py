import json
from pathlib import Path

import pytest

from bayesian_binding.data import load_dat
from bayesian_binding.optimization import fit_map_global
from bayesian_binding.regression import summarize_cooperative_posterior


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "bonin_2022_delta25_dump"


def load_delta25_dump_experiments():
    """Load the three public Δ25-hTS/dUMP isotherms (eLife 2022 source data)."""
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    return [
        load_dat(
            DATA_DIR / spec["file"],
            cell_concentration_mM=spec["cell_concentration_mM"],
            syringe_concentration_mM=spec["syringe_concentration_mM"],
            cell_volume_mL=metadata["cell_volume_mL"],
            temperature_k=metadata["temperature_K"],
            name=name,
        )
        for name, spec in metadata["datasets"].items()
    ]


@pytest.mark.regression
def test_bonin_2022_delta25_dump_noncooperative_map_regression():
    """Global cooperative MAP fit of the public Δ25-hTS/dUMP isotherms returns ~no cooperativity.

    The N-terminal Δ25 truncation abolishes the positive cooperativity of full-length hTS/dUMP
    (Bonin et al. 2019, ~9-fold): the sequential two-site fit returns delta_delta_g ≈ 0
    (cooperativity ratio ≈ 1, DG1 ≈ DG2), i.e. effectively independent two-site binding. Data are
    the public eLife 2022 (Bonin, Sapienza, Lee) Appendix 1 Figure 3 source-data isotherms (also on
    Dryad doi:10.5061/dryad.j9kd51cfx), integrated first-pass from the raw MicroCal traces, so the
    regression asserts the robust qualitative result rather than tight published values.
    """
    experiments = load_delta25_dump_experiments()
    assert [e.n_injections for e in experiments] == [39, 39, 20]

    map_result = fit_map_global(experiments, model_name="cooperative")
    map_params = {key: float(value) for key, value in map_result.params.items()}

    # The shared cell-concentration scaling lands near the ~0.77 of the full-length analysis.
    assert 0.6 < map_params["cell_concentration_scale"] < 1.0

    summary = summarize_cooperative_posterior(map_result.params)
    delta_g1 = summary["delta_g1_kcal_per_mol"]["mean"]
    delta_g2 = summary["delta_g2_kcal_per_mol"]["mean"]
    ratio = summary["cooperativity_ratio"]["mean"]

    # Δ25 removes cooperativity: the two microscopic steps are ~equal (contrast full-length ~9-fold).
    assert abs(map_params["delta_delta_g"]) < 0.6
    assert 0.5 < ratio < 2.0
    assert abs(delta_g1 - delta_g2) < 0.6
    assert -9.0 < delta_g1 < -6.0  # sensible micromolar affinity
