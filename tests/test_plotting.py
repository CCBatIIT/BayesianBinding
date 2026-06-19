"""Smoke test for the notebook/report plotting helper (headless via the Agg backend in conftest)."""

from pathlib import Path

from bayesian_binding.data import load_dat
from bayesian_binding.models import TwoComponentBindingModel
from bayesian_binding.plotting import plot_fit

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_plot_fit_returns_axes_with_observed_and_model():
    import matplotlib.pyplot as plt

    experiment = load_dat(FIXTURES / "simple.DAT")
    q_model = TwoComponentBindingModel.expected_heats(
        experiment.injection_volumes_liter,
        cell_volume_liter=experiment.cell_volume_liter,
        cell_concentration_molar=experiment.cell_concentration_molar,
        syringe_concentration_molar=experiment.syringe_concentration_molar,
        temperature_k=experiment.temperature_k,
        delta_g=-5.0,
        delta_h=-4.0,
        heat_offset=0.0,
    )
    ax = plot_fit(experiment, q_model, title="unit test")
    assert ax.get_title() == "unit test"
    assert len(ax.lines) >= 1  # the model curve (+ baseline)
    assert len(ax.collections) >= 1  # the observed-heats scatter
    plt.close("all")
