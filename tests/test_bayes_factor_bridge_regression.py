"""Regression test for the recommended per-model bridge Bayes factor.

Reproduces the cooperative-vs-equivalent-sites Bayes factor for the public Δ25-hTS/dUMP isotherms
(``data/bonin_2022_delta25_dump``) by per-model bridge sampling
(:func:`bayesian_binding.evidence.bayes_factor_bridge`) -- the recommended estimator. The Δ25
N-terminal truncation abolishes cooperativity, so the cooperative model is **not** favored over the
no-cooperativity (equivalent-sites) null: the log10 Bayes factor is negative and well-determined, with
a small closed-form asymptotic standard error and a passing Bennett-overlap reliability flag. (The
methods note reports about -2.7 for the original TS-project reduction; the first-pass packaged data give
a similar negative factor, ~-1.)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_binding.data import load_dat
from bayesian_binding.evidence import bayes_factor_bridge
from bayesian_binding.inference import build_global_numpyro_model, run_mcmc_global
from bayesian_binding.optimization import fit_map_global

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "bonin_2022_delta25_dump"
_NUM_CHAINS = 4
_NUM_SAMPLES = 4000


def _load_delta25_dump():
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


def _fit(experiments, model_name):
    init = fit_map_global(experiments, model_name=model_name).params
    mcmc = run_mcmc_global(
        experiments,
        model_name=model_name,
        init_params=init,
        rng_seed=20260613,
        num_warmup=2000,
        num_samples=_NUM_SAMPLES,
        num_chains=_NUM_CHAINS,
        target_accept_prob=0.9,
        progress_bar=False,
    )
    return mcmc.get_samples()


@pytest.mark.regression
def test_delta25_dump_cooperative_vs_equivalent_sites_bridge_regression():
    """Cooperative is not favored over the equivalent-sites null for Δ25/dUMP (per-model bridge).

    Δ25 truncates the N-terminus and abolishes cooperativity, so the cooperative model is disfavored
    relative to the no-cooperativity (equivalent-sites) null: the log10 Bayes factor is negative and
    well-determined. The two posteriors are near-Gaussian here, so the per-model bridge sits in its
    reliable regime (overlap ~ 1, tiny asymptotic SE). (Cross-estimator agreement of the per-model
    bridge with the nested between-model bridge is checked separately, on a controlled overlapping pair,
    in ``test_bayes_factor_consistency.py`` -- the cooperative/equivalent posteriors are too separated
    along ΔΔH for the nested estimator to be reliable here.)
    """
    experiments = _load_delta25_dump()
    assert [e.n_injections for e in experiments] == [39, 39, 20]

    cooperative = _fit(experiments, "cooperative")
    equivalent_sites = _fit(experiments, "cooperative_equivalent_sites")
    model_cooperative = build_global_numpyro_model(experiments, model_name="cooperative")
    model_equivalent = build_global_numpyro_model(experiments, model_name="cooperative_equivalent_sites")

    # Recommended estimator: per-model bridge (each posterior bridged to its own MVN reference).
    out = bayes_factor_bridge(cooperative, model_cooperative, equivalent_sites, model_equivalent, rng_seed=20260613)
    assert -5.0 < out["log10_bf"] < 0.0  # cooperative disfavored
    assert out["reliable"]
    assert np.isfinite(out["se"]) and out["se"] < 1.0
