"""Fast tests for the nested-model Bayes-factor estimator (Nguyen et al. 2022).

Runs in the default ``pytest`` suite: the BAR solver is checked against an analytic two-Gaussian
case, the reparameterizations are checked for exact round-trips, and the full nested estimator and
its convergence helper are run end to end on a small two-component fit (kept fast and only checked
for finite, well-formed output -- precise Bayes-factor values are the job of the example, not a
fast test).
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_binding.bayes_factor import (
    AffineWarp,
    DirectNesting,
    GaussianProposal,
    bayes_factor_convergence,
    bennett_acceptance_ratio,
    nested_bayes_factor,
)
from bayesian_binding.data import ITCExperiment
from bayesian_binding.evidence import bridge_diagnostics
from bayesian_binding.inference import build_numpyro_model, run_mcmc
from bayesian_binding.models import MODEL_REGISTRY
from bayesian_binding.optimization import initial_params

_CELL_VOLUME_LITER = 1.3513e-3
_TEMPERATURE_K = 298.15


def test_bennett_acceptance_ratio_matches_analytic_gaussians():
    """BAR recovers ``ln(sigma_0 / sigma_1)`` for two zero-mean Gaussians with known constants."""
    rng = np.random.default_rng(0)
    sigma_0, sigma_1 = 1.0, 1.6
    x0 = rng.normal(0.0, sigma_0, 40000)
    x1 = rng.normal(0.0, sigma_1, 40000)
    w_F = x0**2 / (2 * sigma_1**2) - x0**2 / (2 * sigma_0**2)
    w_R = x1**2 / (2 * sigma_0**2) - x1**2 / (2 * sigma_1**2)
    assert abs(bennett_acceptance_ratio(w_F, w_R) - np.log(sigma_0 / sigma_1)) < 0.05


def test_bennett_acceptance_ratio_drops_nonfinite():
    value = bennett_acceptance_ratio([0.1, np.inf, -0.2, 0.0], [0.0, np.nan, 0.3, -0.1])
    assert np.isfinite(value)


def test_bridge_diagnostics_overlap_and_asymptotic_se():
    """Bennett overlap -> 1 (identical) / -> 0 (disjoint); the asymptotic SE agrees with a bootstrap."""
    rng = np.random.default_rng(5)

    def works(d, n=4000):
        x0 = rng.normal(0.0, 1.0, n)
        x1 = rng.normal(d, 1.0, n)
        return d * d / 2 - d * x0, d * x1 - d * d / 2  # two unit Gaussians, true delta_f = 0

    overlap_same, _ = bridge_diagnostics(*works(0.0), 0.0)
    overlap_far, se_far = bridge_diagnostics(*works(6.0), 0.0)
    assert overlap_same > 0.95 and overlap_far < 0.05
    assert se_far > 0.1  # poor overlap -> large asymptotic SE

    w_f, w_r = works(1.5)
    delta_f = bennett_acceptance_ratio(w_f, w_r)
    _, se = bridge_diagnostics(w_f, w_r, delta_f)
    boots = [
        bennett_acceptance_ratio(w_f[rng.integers(0, w_f.size, w_f.size)], w_r[rng.integers(0, w_r.size, w_r.size)])
        for _ in range(200)
    ]
    assert abs(se - np.std(boots)) < 0.4 * np.std(boots)  # closed-form asymptotic SE ~ bootstrap SE


def test_direct_nesting_round_trip():
    rng = np.random.default_rng(2)
    complex_samples = {"delta_g1": rng.normal(-10, 0.3, 32), "rho": rng.uniform(0.1, 0.9, 32)}
    nesting = DirectNesting(gamma_sites=("rho",))
    theta1, gamma = nesting.split(complex_samples)
    assert "rho" not in theta1 and gamma.shape == (32, 1)
    recovered = nesting.augment(theta1, gamma)
    np.testing.assert_allclose(recovered["rho"], complex_samples["rho"], rtol=1e-10)


def test_affine_warp_fit_frames_and_jacobian():
    """``fit`` recovers the mean offset and SD ratio; the two frame maps are exact inverses."""
    rng = np.random.default_rng(11)
    x_simple = rng.normal(-6.87, 0.20, 5000)
    x_complex = rng.normal(-6.11, 0.10, 5000)
    warp = AffineWarp.fit({"delta_g": x_simple}, {"delta_g": x_complex}, ("delta_g",), scale=True)
    assert abs(warp.mu_simple["delta_g"] - x_simple.mean()) < 1e-9
    assert abs(warp.scale["delta_g"] - x_complex.std() / x_simple.std()) < 1e-9
    assert abs(warp.log_scale_total - np.log(warp.scale["delta_g"])) < 1e-12
    # T maps the simple mean onto the complex mean; T_inv undoes T exactly.
    mapped = warp.to_complex_frame({"delta_g": np.array([warp.mu_simple["delta_g"]])})["delta_g"][0]
    assert abs(mapped - warp.mu_complex["delta_g"]) < 1e-9
    back = warp.to_simple_frame(warp.to_complex_frame({"delta_g": x_simple}))["delta_g"]
    np.testing.assert_allclose(back, x_simple, rtol=1e-10)
    # A pure shift (scale=False) has unit scale and zero Jacobian.
    shift_only = AffineWarp.fit({"delta_g": x_simple}, {"delta_g": x_complex}, ("delta_g",), scale=False)
    assert shift_only.scale["delta_g"] == 1.0 and shift_only.log_scale_total == 0.0


def test_affine_warp_recovers_ratio_for_shifted_scaled_gaussians():
    """Mapped BAR with the affine warp recovers ``ln(sigma_c / sigma_s)`` for two Gaussians that differ
    in BOTH mean and scale -- exercising the warp transform and the constant ``ln s`` Jacobian (a wrong
    Jacobian sign would bias the estimate by ``2 ln s`` and fail)."""
    rng = np.random.default_rng(12)
    mu_s, sd_s = -6.87, 0.20
    mu_c, sd_c = -6.11, 0.10
    x0 = rng.normal(mu_s, sd_s, 60000)  # simple (state 0) samples
    x1 = rng.normal(mu_c, sd_c, 60000)  # complex (state 1) samples
    warp = AffineWarp.fit({"delta_g": x0}, {"delta_g": x1}, ("delta_g",), scale=True)
    log_s = warp.log_scale_total
    def u_s(x):  # -log of a unit-peak Gaussian (Z = sd * sqrt(2 pi))
        return (x - mu_s) ** 2 / (2 * sd_s**2)

    def u_c(x):
        return (x - mu_c) ** 2 / (2 * sd_c**2)
    tx0 = warp.to_complex_frame({"delta_g": x0})["delta_g"]       # forward: simple -> complex frame
    tinv_x1 = warp.to_simple_frame({"delta_g": x1})["delta_g"]    # reverse: complex -> simple frame
    w_F = u_c(tx0) - u_s(x0) - log_s
    w_R = u_s(tinv_x1) - u_c(x1) + log_s
    ln_bf = -bennett_acceptance_ratio(w_F, w_R)  # complex over simple = ln(Z_c / Z_s) = ln(sd_c / sd_s)
    assert abs(ln_bf - np.log(sd_c / sd_s)) < 0.05


def test_affine_warp_identity_matches_unwarped(mixture_fits):
    """An identity warp (zero shift, unit scale) reproduces the un-warped Bayes factor exactly."""
    base = nested_bayes_factor(
        mixture_fits["racemic_mixture"], mixture_fits["enantiomeric_mixture"],
        mixture_fits["model_rm"], mixture_fits["model_em"], _NESTING_RM_EM, warp=None,
    )
    identity = AffineWarp(mu_simple={"delta_g1": 0.0}, mu_complex={"delta_g1": 0.0}, scale={"delta_g1": 1.0})
    warped = nested_bayes_factor(
        mixture_fits["racemic_mixture"], mixture_fits["enantiomeric_mixture"],
        mixture_fits["model_rm"], mixture_fits["model_em"], _NESTING_RM_EM, warp=identity,
    )
    assert abs(base.ln_bayes_factor - warped.ln_bayes_factor) < 1e-6


def test_gaussian_proposal_fit_and_logpdf():
    rng = np.random.default_rng(3)
    gamma = rng.multivariate_normal([4.0, -2.0], [[0.5, 0.1], [0.1, 0.3]], size=2000)
    proposal = GaussianProposal.fit(gamma)
    assert proposal.mean.shape == (2,)
    draws = proposal.sample(50, np.random.default_rng(4))
    assert draws.shape == (50, 2) and np.all(np.isfinite(proposal.logpdf(draws)))


@pytest.fixture(scope="module")
def mixture_fits():
    """Fit racemic- and enantiomeric-mixture models to small synthetic data. RM is EM with the
    composition ``rho`` fixed at 0.5, so RM nests in EM via ``DirectNesting`` on ``rho``."""
    rng = np.random.default_rng(7)
    volumes = np.full(18, 8.0e-6)
    common = dict(
        cell_volume_liter=_CELL_VOLUME_LITER,
        cell_concentration_molar=40e-6,
        syringe_concentration_molar=2000e-6,
        temperature_k=_TEMPERATURE_K,
    )
    heats = np.asarray(
        MODEL_REGISTRY["two_component"].expected_heats(
            volumes, delta_g=-10.0, delta_h=-6.0, heat_offset=1.0, **common
        ),
        dtype=float,
    )
    heats = heats + rng.normal(0.0, 0.3, size=heats.shape)
    experiment = ITCExperiment(
        name="bf_smoke",
        injection_volumes_liter=volumes,
        heats_microcalorie=heats,
        cell_concentration_molar=40e-6,
        syringe_concentration_molar=2000e-6,
        cell_volume_liter=_CELL_VOLUME_LITER,
        temperature_k=_TEMPERATURE_K,
    )

    def fit(model_name):
        # Mixtures have a spurious 'no binding' basin; initialize from the physics-based point.
        init = initial_params(experiment, model_name=model_name)
        mcmc = run_mcmc(
            experiment,
            model_name=model_name,
            init_params=init,
            num_warmup=400,
            num_samples=400,
            num_chains=1,
            target_accept_prob=0.9,
            progress_bar=False,
        )
        return mcmc.get_samples()

    return {
        "experiment": experiment,
        "racemic_mixture": fit("racemic_mixture"),
        "enantiomeric_mixture": fit("enantiomeric_mixture"),
        "model_rm": build_numpyro_model(experiment, model_name="racemic_mixture"),
        "model_em": build_numpyro_model(experiment, model_name="enantiomeric_mixture"),
    }


_NESTING_RM_EM = DirectNesting(gamma_sites=("rho",))


def test_nested_bayes_factor_runs(mixture_fits):
    # Warp-BAR is applied by default (warped=True); the result carries the asymptotic SE + overlap.
    result = nested_bayes_factor(
        mixture_fits["racemic_mixture"],
        mixture_fits["enantiomeric_mixture"],
        mixture_fits["model_rm"],
        mixture_fits["model_em"],
        _NESTING_RM_EM,
    )
    assert np.isfinite(result.ln_bayes_factor)
    assert result.n_simple_samples > 0 and result.n_complex_samples > 0
    assert 0.0 <= result.overlap <= 1.0
    assert isinstance(result.reliable, bool)
    assert result.warped is True
    # warp=None gives the un-warped diagnostic.
    unwarped = nested_bayes_factor(
        mixture_fits["racemic_mixture"],
        mixture_fits["enantiomeric_mixture"],
        mixture_fits["model_rm"],
        mixture_fits["model_em"],
        _NESTING_RM_EM,
        warp=None,
    )
    assert unwarped.warped is False and np.isfinite(unwarped.ln_bayes_factor)


def test_bayes_factor_convergence_runs(mixture_fits):
    rows = bayes_factor_convergence(
        mixture_fits["racemic_mixture"],
        mixture_fits["enantiomeric_mixture"],
        mixture_fits["model_rm"],
        mixture_fits["model_em"],
        _NESTING_RM_EM,
        fractions=(0.5, 1.0),
    )
    assert len(rows) == 2
    assert all(np.isfinite(r["ln_bayes_factor"]) for r in rows)
    assert all(0.0 <= r["overlap"] <= 1.0 for r in rows)
    assert rows[1]["n_complex_samples"] > rows[0]["n_complex_samples"]
