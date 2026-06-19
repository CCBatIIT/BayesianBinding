"""Unit tests for the single-experiment optimization helpers across all ITC binding models.

``sample_site_names``, ``initial_params`` and ``parameter_bounds`` each branch per model. The cooperative
and dimerization branches are otherwise exercised only by the *global* fit path, so these direct calls
cover the single-experiment branches and verify the three helpers agree on the active site set and that
each initial value lies inside its optimizer bound.
"""

from pathlib import Path

import pytest

from bayesian_binding.data import load_dat
from bayesian_binding.optimization import initial_params, parameter_bounds, sample_site_names

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_ITC_MODELS = [
    "two_component",
    "cooperative_equivalent_sites",
    "cooperative",
    "cooperative_equal_affinity",
    "dimerization_cooperative",
    "dimerization_monomer_cooperative",
    "racemic_mixture",
    "enantiomeric_mixture",
]


@pytest.fixture(scope="module")
def experiment():
    return load_dat(FIXTURES / "simple.DAT")


@pytest.mark.parametrize("model_name", _ITC_MODELS)
def test_site_names_initial_params_and_bounds_agree(experiment, model_name):
    """The three helpers expose the same active sites, and each initial value is inside its bound."""
    sites = sample_site_names(model_name)
    init = initial_params(experiment, model_name=model_name)
    bounds = parameter_bounds(experiment, model_name=model_name)
    assert set(sites) == set(init) == set(bounds)
    for name in sites:
        low, high = bounds[name]
        assert low <= init[name] <= high


@pytest.mark.parametrize("model_name", ["cooperative", "dimerization_monomer_cooperative"])
def test_fixed_sites_are_dropped_everywhere(experiment, model_name):
    pinned = sample_site_names(model_name)[-1]  # a thermodynamic site
    fixed = {pinned: 0.0}
    assert pinned not in sample_site_names(model_name, fixed)
    assert pinned not in initial_params(experiment, model_name=model_name, fixed=fixed)
    assert pinned not in parameter_bounds(experiment, model_name=model_name, fixed=fixed)


def test_unknown_model_raises_keyerror(experiment):
    with pytest.raises(KeyError):
        sample_site_names("not_a_model")
    with pytest.raises(KeyError):
        initial_params(experiment, model_name="not_a_model")
    with pytest.raises(KeyError):
        parameter_bounds(experiment, model_name="not_a_model")


def test_uniform_concentration_priors_swap_sites(experiment):
    """The uniform-prior path exposes molar concentration sites instead of log-concentration ones."""
    kwargs = dict(uniform_cell_concentration=True, uniform_syringe_concentration=True)
    sites = sample_site_names("two_component", **kwargs)
    init = initial_params(experiment, model_name="two_component", **kwargs)
    bounds = parameter_bounds(experiment, model_name="two_component", **kwargs)
    for molar in ("cell_concentration_molar", "syringe_concentration_molar"):
        assert molar in sites and molar in init and molar in bounds
    assert "log_cell_concentration" not in sites and "log_syringe_concentration" not in sites
