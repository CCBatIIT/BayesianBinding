"""Fast smoke tests: a quick MAP fit and a short MCMC for every validated model.

These run in the default ``pytest`` invocation (they are unmarked, unlike the slow
``regression`` tests) so a single ``pytest`` quickly proves that MAP optimization and NUTS
still work end to end for each model -- the single-experiment two-component and
enantiomeric-mixture models and the shared-thermodynamics global fit (cooperative plus the
mixtures). They use small synthetic isotherms generated from each model so they are
self-contained and fast; they assert that the pipeline runs and returns finite values, not
that it recovers the truth (that is the job of the ``regression`` tests).

NUTS is initialized from the MAP estimate (fast now that the MAP optimizer uses exact
jitted gradients), as the example notebooks do, which keeps the chains in the best-fit mode
and inside the prior support. The synthetic titrations are designed to *saturate* (ligand
well past the receptor), so the heat offset is well determined and lands comfortably inside
its prior; an incomplete titration would push the offset prior away from the true value.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_binding.data import ITCExperiment
from bayesian_binding.inference import run_mcmc, run_mcmc_global
from bayesian_binding.models import MODEL_REGISTRY
from bayesian_binding.optimization import fit_map, fit_map_global

# Ground-truth thermodynamics per model (kcal/mol; rho dimensionless). Chosen well inside
# the priors and well-determined by the synthetic design below.
SMOKE_TRUTH = {
    "two_component": dict(delta_g=-9.0, delta_h=-5.0),
    "cooperative": dict(delta_g=-7.0, delta_delta_g=-1.0, delta_h_first=-9.0, delta_h_second=-5.0),
    "racemic_mixture": dict(rho=0.5, delta_g1=-10.0, delta_delta_g=3.0, delta_h1=-7.0, delta_h2=-3.0),
    "enantiomeric_mixture": dict(rho=0.45, delta_g1=-10.0, delta_delta_g=3.0, delta_h1=-7.0, delta_h2=-3.0),
}

# Per-model titration design (cell/syringe concentrations in micromolar). The high syringe
# concentrations (with 20 injections) drive the titration to saturation so the heat offset
# is identifiable.
SMOKE_DESIGN = {
    "two_component": dict(cell_uM=40.0, syringe_uM=2000.0),
    "cooperative": dict(cell_uM=25.0, syringe_uM=2000.0),
    "racemic_mixture": dict(cell_uM=40.0, syringe_uM=2000.0),
    "enantiomeric_mixture": dict(cell_uM=40.0, syringe_uM=2000.0),
}

_CELL_VOLUME_LITER = 1.3513e-3
_TEMPERATURE_K = 298.15


def synthetic_experiment(
    model_name: str,
    *,
    cell_uM: float,
    syringe_uM: float,
    n_injections: int = 20,
    heat_offset: float = 1.0,
    noise_microcalorie: float = 0.1,
    seed: int = 0,
    name: str | None = None,
) -> ITCExperiment:
    """Build a small synthetic ITC experiment from a model's own ``expected_heats``."""
    rng = np.random.default_rng(seed)
    volumes = np.full(n_injections, 8.0e-6)
    q = np.asarray(
        MODEL_REGISTRY[model_name].expected_heats(
            volumes,
            cell_volume_liter=_CELL_VOLUME_LITER,
            cell_concentration_molar=cell_uM * 1.0e-6,
            syringe_concentration_molar=syringe_uM * 1.0e-6,
            temperature_k=_TEMPERATURE_K,
            heat_offset=heat_offset,
            **SMOKE_TRUTH[model_name],
        ),
        dtype=float,
    )
    q = q + rng.normal(0.0, noise_microcalorie, size=q.shape)
    return ITCExperiment(
        name=name or f"smoke_{model_name}",
        injection_volumes_liter=volumes,
        heats_microcalorie=q,
        cell_concentration_molar=cell_uM * 1.0e-6,
        syringe_concentration_molar=syringe_uM * 1.0e-6,
        cell_volume_liter=_CELL_VOLUME_LITER,
        temperature_k=_TEMPERATURE_K,
    )


def _assert_all_finite(samples: dict) -> None:
    for name, value in samples.items():
        array = np.asarray(value)
        assert np.all(np.isfinite(array)), f"non-finite samples for site {name!r}"


# Short NUTS settings; large enough to exercise warmup + sampling, small enough to be fast.
_WARMUP = 25
_SAMPLES = 25


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", ["two_component", "racemic_mixture", "enantiomeric_mixture"])
def test_single_experiment_map_and_mcmc_smoke(model_name):
    """MAP fit and a short MCMC run for each single-experiment model on synthetic data."""
    experiment = synthetic_experiment(model_name, **SMOKE_DESIGN[model_name], seed=11)

    # L-BFGS-B can report success=False (ABNORMAL_TERMINATION_IN_LNSRCH) at a flat optimum
    # even when the recovered point is good, so check the result is finite rather than the
    # optimizer's success flag; fit_map already keeps the best multistart result.
    map_result = fit_map(experiment, model_name=model_name)
    assert np.isfinite(map_result.log_posterior)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in map_result.params.values())

    mcmc = run_mcmc(
        experiment,
        model_name=model_name,
        init_params=map_result.params,
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
    )
    samples = mcmc.get_samples()
    _assert_all_finite(samples)
    # The shared thermodynamic sites for this model must be present and sampled.
    for site in SMOKE_TRUTH[model_name]:
        if site == "rho" and model_name == "racemic_mixture":
            continue  # rho is fixed at 0.5 for the racemic model, so it is not a site
        assert site in samples and np.asarray(samples[site]).shape[0] == _SAMPLES


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", ["cooperative", "racemic_mixture", "enantiomeric_mixture"])
def test_global_map_and_mcmc_smoke(model_name):
    """Global shared-thermodynamics MAP fit and short MCMC across two synthetic isotherms.

    Covers the cooperative global fit (as in Bonin et al. 2019) and the generalized global
    fit for the mixture models, where the binding thermodynamics and a single shared
    composition ``rho`` are shared while each isotherm keeps its own offset and noise scale.
    """
    design = SMOKE_DESIGN[model_name]
    experiments = [
        synthetic_experiment(model_name, cell_uM=design["cell_uM"], syringe_uM=design["syringe_uM"], seed=1, name="iso0"),
        synthetic_experiment(model_name, cell_uM=1.4 * design["cell_uM"], syringe_uM=design["syringe_uM"], seed=2, name="iso1"),
    ]

    # See the single-experiment test: L-BFGS-B's success flag is unreliable at a flat
    # optimum, so assert the recovered point is finite instead.
    map_result = fit_map_global(experiments, model_name=model_name, fit_concentration_scale=False)
    assert np.isfinite(map_result.log_posterior)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in map_result.params.values())

    mcmc = run_mcmc_global(
        experiments,
        model_name=model_name,
        fit_concentration_scale=False,
        init_params=map_result.params,
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
    )
    samples = mcmc.get_samples()
    _assert_all_finite(samples)
    # Each experiment keeps its own nuisance sites; the thermodynamics are shared (one site).
    for index in range(len(experiments)):
        assert f"heat_offset_{index}" in samples
        assert f"log_sigma_{index}" in samples
    if model_name == "enantiomeric_mixture":
        assert "rho" in samples  # free, shared composition
