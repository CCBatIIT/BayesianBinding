"""The two Bayes-factor estimators compute the *same quantity*, checked against an analytic value.

Real ITC model pairs (e.g. cooperative vs equivalent-sites, racemic vs enantiomeric) are deliberately
well *separated*, so the nested between-model bridge sits in its low-overlap/biased regime there and
cannot be expected to agree with the per-model bridge (this is exactly why per-model bridge is the
recommended estimator). To verify that the two estimators nevertheless target the *same* Bayes factor,
we use a controlled conjugate linear-Gaussian pair whose posteriors overlap by construction and whose
Bayes factor is known in closed form:

    simple model:   y = a * x         + N(0, sigma),   a    ~ N(0, tau)
    complex model:  y = a * x + b * z + N(0, sigma),   a, b ~ N(0, tau)   (DirectNesting on "b")

With orthogonal columns ``x _|_ z`` the shared coefficient ``a`` has the same posterior in both models,
so the per-model bridge (each posterior -> its own Gaussian) and the nested bridge (between the two
posteriors) must both reproduce ``ln Z = ln N(y; 0, sigma^2 I + tau^2 X X^T)``.

Exact Gaussian posterior samples are drawn directly (no MCMC), so the test is fast and deterministic.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from scipy.stats import multivariate_normal

from bayesian_binding.bayes_factor import DirectNesting, nested_bayes_factor
from bayesian_binding.evidence import bayes_factor_bridge

_LN10 = np.log(10.0)
_N_DATA = 24
_SIGMA = 0.6
_TAU = 3.0
_N_SAMPLES = 6000
_TOL = 0.05  # log10 units; the realized agreement is ~1e-4, so this is a generous but meaningful band


def _build_problem(corr=0.0):
    rng = np.random.default_rng(0)
    # Orthonormal columns scaled by sqrt(n); `corr` mixes z toward x so that corr(x, z) = corr, which
    # induces a posterior correlation between the shared `a` and the added `b` (corr=0 -> x _|_ z).
    basis, _ = np.linalg.qr(rng.standard_normal((_N_DATA, 2)))
    x = basis[:, 0]
    z = corr * basis[:, 0] + np.sqrt(1.0 - corr**2) * basis[:, 1]
    design = np.column_stack([x, z]) * np.sqrt(_N_DATA)
    y = design @ np.array([1.0, 0.35]) + _SIGMA * rng.standard_normal(_N_DATA)
    design_simple = design[:, :1]

    def analytic_logz(matrix):
        cov = _SIGMA**2 * np.eye(_N_DATA) + _TAU**2 * (matrix @ matrix.T)
        return float(multivariate_normal.logpdf(y, mean=np.zeros(_N_DATA), cov=cov))

    def gaussian_posterior(matrix):
        precision = matrix.T @ matrix / _SIGMA**2 + np.eye(matrix.shape[1]) / _TAU**2
        cov = np.linalg.inv(precision)
        return cov @ matrix.T @ y / _SIGMA**2, cov

    analytic_log10_bf = (analytic_logz(design) - analytic_logz(design_simple)) / _LN10

    mean_c, cov_c = gaussian_posterior(design)
    mean_s, cov_s = gaussian_posterior(design_simple)
    draws_c = rng.multivariate_normal(mean_c, cov_c, size=_N_SAMPLES)
    draws_s = rng.multivariate_normal(mean_s, cov_s, size=_N_SAMPLES)
    complex_samples = {"a": draws_c[:, 0], "b": draws_c[:, 1]}
    simple_samples = {"a": draws_s[:, 0]}

    def make_model(matrix, names):
        matrix_j, y_j = jnp.asarray(matrix), jnp.asarray(y)

        def model():
            coeffs = jnp.stack([numpyro.sample(name, dist.Normal(0.0, _TAU)) for name in names])
            numpyro.sample("y", dist.Normal(matrix_j @ coeffs, _SIGMA), obs=y_j)

        return model

    return {
        "analytic_log10_bf": analytic_log10_bf,
        "simple_samples": simple_samples,
        "complex_samples": complex_samples,
        "simple_model": make_model(design_simple, ["a"]),
        "complex_model": make_model(design, ["a", "b"]),
    }


def test_bridge_and_nested_recover_the_same_analytic_bayes_factor():
    """Per-model bridge and nested warp-BAR both reproduce the analytic BF (same quantity, both ways)."""
    problem = _build_problem()
    analytic = problem["analytic_log10_bf"]
    assert analytic > 0.3  # the complex model is meaningfully favored, not a trivial BF ~ 1

    bridge = bayes_factor_bridge(
        problem["complex_samples"], problem["complex_model"],
        problem["simple_samples"], problem["simple_model"],
        rng_seed=1,
    )
    assert bridge["reliable"]
    assert abs(bridge["log10_bf"] - analytic) < _TOL

    nested = nested_bayes_factor(
        problem["simple_samples"], problem["complex_samples"],
        problem["simple_model"], problem["complex_model"],
        DirectNesting(gamma_sites=("b",)), rng_seed=1,
    )
    assert nested.reliable
    nested_log10_bf = nested.ln_bayes_factor / _LN10
    assert abs(nested_log10_bf - analytic) < _TOL

    # The whole point: the two estimators agree with each other on this overlapping pair.
    assert abs(bridge["log10_bf"] - nested_log10_bf) < _TOL


def test_unwarped_nested_also_recovers_the_analytic_bayes_factor_when_overlap_is_good():
    """With well-overlapping posteriors the un-warped nested estimator is unbiased too (warp ~ identity)."""
    problem = _build_problem()
    nested = nested_bayes_factor(
        problem["simple_samples"], problem["complex_samples"],
        problem["simple_model"], problem["complex_model"],
        DirectNesting(gamma_sites=("b",)), warp=None, rng_seed=1,
    )
    assert nested.reliable and not nested.warped
    assert abs(nested.ln_bayes_factor / _LN10 - problem["analytic_log10_bf"]) < _TOL


# --- Conditional proposal f(gamma | theta_1): captures theta_1–gamma correlation the warp cannot ---


def test_conditional_proposal_restores_overlap_under_correlation():
    """When the added dimension correlates with the shared parameters, the marginal proposal f(gamma)
    loses bridge overlap (here enough to be flagged unreliable), while the conditional f(gamma|theta_1)
    restores the overlap, flips it back to reliable, and still recovers the analytic Bayes factor."""
    problem = _build_problem(corr=0.97)  # strong x–z correlation -> strong a–b posterior correlation
    args = (
        problem["simple_samples"], problem["complex_samples"],
        problem["simple_model"], problem["complex_model"], DirectNesting(gamma_sites=("b",)),
    )
    marginal = nested_bayes_factor(*args, rng_seed=1, conditional_proposal=False)
    conditional = nested_bayes_factor(*args, rng_seed=1, conditional_proposal=True)

    assert not marginal.conditional_proposal and conditional.conditional_proposal
    assert conditional.overlap > marginal.overlap + 0.2  # substantially better overlap
    assert conditional.reliable
    assert abs(conditional.ln_bayes_factor / _LN10 - problem["analytic_log10_bf"]) < _TOL


def test_conditional_proposal_matches_marginal_without_correlation():
    """With no theta_1–gamma correlation the conditional proposal reduces to the marginal one (no harm):
    it still recovers the analytic Bayes factor on the orthogonal design."""
    problem = _build_problem()  # orthogonal design, corr = 0
    conditional = nested_bayes_factor(
        problem["simple_samples"], problem["complex_samples"],
        problem["simple_model"], problem["complex_model"],
        DirectNesting(gamma_sites=("b",)), rng_seed=1, conditional_proposal=True,
    )
    assert conditional.reliable and conditional.conditional_proposal
    assert abs(conditional.ln_bayes_factor / _LN10 - problem["analytic_log10_bf"]) < _TOL
