"""Bayes factors for nested models by bridge sampling / Bennett Acceptance Ratio.

This implements the nested-model bridge-sampling Bayes factor of Nguyen et al. 2022 (PLOS ONE
17(9):e0273656, Methods + S1 Appendix; reference code at
``github.com/nguyentrunghai/bayesian-itc``). It is distinct from the single-model
marginal-likelihood bridge sampling in :mod:`bayesian_binding.evidence`.

For two nested models with parameters ``theta_2 = (theta_1, gamma)`` (``theta_1`` shared,
``gamma`` extra in the more complex model), the Bayes factor ``R = Z_complex / Z_simple`` is the
ratio of normalizing constants of the unnormalized posteriors. It is estimated by bridging two
distributions:

- state **F** (complex): the complex posterior ``p2(theta_1, gamma)``;
- state **I** (simple, augmented): ``p1(theta_1) * f(gamma)``, where ``f(gamma)`` is a Gaussian
  proposal fit to the complex model's posterior of the extra parameters ``gamma``.

With potentials ``u_I = -ln p1(theta_1) - ln f(gamma)`` and ``u_F = -ln p2(theta_1, gamma)``, the
forward/reverse work and BAR give ``ln R``::

    w_F = u_F(I-samples) - u_I(I-samples)      # I-samples: simple theta_1 + f(gamma) draws
    w_R = u_I(F-samples) - u_F(F-samples)      # F-samples: complex posterior, split into (theta_1, gamma)
    delta_F = BAR(w_F, w_R)
    ln R (complex over simple) = -delta_F
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bayesian_binding import _jax_config as _jax_config

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import multivariate_normal

from bayesian_binding.evidence import (
    _regularized_covariance,
    bridge_diagnostics,
)

_WARP_AUTO = "auto"

# The nested between-model bridge is more bias-prone at moderate overlap than the per-model bridge: the
# Gaussian proposal over the added dimensions gamma, and any theta1-gamma correlation the affine warp
# cannot remove, undersample the bridge region. It therefore uses a STRICTER overlap gate than the
# per-model bridge's evidence.DEFAULT_OVERLAP_THRESHOLD (0.10). Empirically, on real cooperative data the
# nested estimate was biased by ~2 log10 at a Bennett overlap of 0.23 while the per-model bridge stayed
# reliable, so we require substantially higher overlap before trusting the nested estimate (otherwise
# the flag recommends deferring to the per-model bridge).
DEFAULT_NESTED_OVERLAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Bennett Acceptance Ratio
# ---------------------------------------------------------------------------
def bennett_acceptance_ratio(w_F: np.ndarray, w_R: np.ndarray, *, bracket: float = 1.0e3) -> float:
    """Return the BAR free-energy difference ``delta_F = f_1 - f_0 = -ln(Z_1 / Z_0)``.

    ``w_F`` are forward work values ``u_1 - u_0`` for samples drawn from state 0; ``w_R`` are
    reverse work values ``u_0 - u_1`` for samples drawn from state 1 (``u_i`` = negative log of the
    unnormalized density of state ``i``). Non-finite work values are dropped. The estimate is the
    root of the self-consistent BAR equation (Shirts et al. 2003), solved with Brent's method.
    """
    w_F = np.asarray(w_F, dtype=float)
    w_R = np.asarray(w_R, dtype=float)
    w_F = w_F[np.isfinite(w_F)]
    w_R = w_R[np.isfinite(w_R)]
    if w_F.size == 0 or w_R.size == 0:
        raise ValueError("BAR requires at least one finite forward and reverse work value.")
    log_ratio = np.log(w_F.size / w_R.size)

    def equation(delta_f: float) -> float:
        # sum_i 1/(1+exp(M + w_F_i - dF)) - sum_j 1/(1+exp(-M + w_R_j + dF)), M = ln(n_F/n_R).
        return float(
            np.sum(expit(delta_f - log_ratio - w_F)) - np.sum(expit(log_ratio - w_R - delta_f))
        )

    lo, hi = -bracket, bracket
    while np.sign(equation(lo)) == np.sign(equation(hi)):
        lo *= 2.0
        hi *= 2.0
        if hi > 1.0e8:
            raise RuntimeError("BAR failed to bracket a root; check the work arrays for overlap.")
    return float(brentq(equation, lo, hi, xtol=1.0e-12, maxiter=200))


# ---------------------------------------------------------------------------
# Gaussian proposal over the extra parameters gamma
# ---------------------------------------------------------------------------
@dataclass
class GaussianProposal:
    """Multivariate-normal proposal ``f(gamma)`` fit to the complex model's posterior gamma."""

    mean: np.ndarray
    covariance: np.ndarray
    _dist: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dist = multivariate_normal(mean=self.mean, cov=self.covariance, allow_singular=False)

    @classmethod
    def fit(cls, gamma: np.ndarray) -> "GaussianProposal":
        gamma = np.atleast_2d(np.asarray(gamma, dtype=float))
        return cls(mean=gamma.mean(axis=0), covariance=_regularized_covariance(gamma))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.atleast_2d(rng.multivariate_normal(self.mean, self.covariance, size=n))

    def logpdf(self, gamma: np.ndarray) -> np.ndarray:
        return np.asarray(self._dist.logpdf(np.atleast_2d(gamma)), dtype=float)


@dataclass
class ConditionalGaussianProposal:
    """Proposal for ``gamma`` **conditioned on the shared parameters** ``theta_1``, fit from the joint
    complex posterior over ``(theta_1, gamma)``.

    The marginal :class:`GaussianProposal` draws ``gamma`` independently of ``theta_1``; but in the
    complex posterior the two are usually *correlated*, so an independent draw lands the augmented simple
    state off the complex ridge and depresses the bridge overlap (the dominant residual bias the affine
    ``theta_1`` warp cannot remove). This proposal instead uses the standard Gaussian conditional
    ``gamma | theta_1 ~ N(mu_g + (theta_1 - mu_s) B^T, C)`` with ``B = Sigma_gs Sigma_ss^-1`` and
    ``C = Sigma_gg - B Sigma_sg``, restoring that correlation. It is a normalized density in ``gamma``
    for *every* ``theta_1`` (it integrates to 1 over ``gamma``), so the augmented simple state still has
    normalizing constant ``Z_simple`` and the BAR identity ``ln BF = -Delta f`` is unchanged.
    """

    mean_shared: np.ndarray  # mu_s, shape (d,)
    mean_gamma: np.ndarray  # mu_g, shape (k,)
    coef: np.ndarray  # B = Sigma_gs Sigma_ss^-1, shape (k, d)
    cond_cov: np.ndarray  # C, shape (k, k)
    _dist: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dist = multivariate_normal(
            mean=np.zeros(self.cond_cov.shape[0]), cov=self.cond_cov, allow_singular=False
        )

    @classmethod
    def fit(cls, shared: np.ndarray, gamma: np.ndarray) -> "ConditionalGaussianProposal":
        shared = np.atleast_2d(np.asarray(shared, dtype=float))
        gamma = np.atleast_2d(np.asarray(gamma, dtype=float))
        dim = shared.shape[1]
        cov = _regularized_covariance(np.hstack([shared, gamma]))
        cov_ss, cov_sg, cov_gg = cov[:dim, :dim], cov[:dim, dim:], cov[dim:, dim:]
        coef = np.linalg.solve(cov_ss, cov_sg).T  # (k, d)
        cond_cov = cov_gg - coef @ cov_sg
        cond_cov = 0.5 * (cond_cov + cond_cov.T)  # symmetrize against round-off
        return cls(mean_shared=shared.mean(axis=0), mean_gamma=gamma.mean(axis=0), coef=coef, cond_cov=cond_cov)

    def _conditional_mean(self, shared: np.ndarray) -> np.ndarray:
        shared = np.atleast_2d(np.asarray(shared, dtype=float))
        return self.mean_gamma + (shared - self.mean_shared) @ self.coef.T  # (n, k)

    def sample(self, shared: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self._conditional_mean(shared)
        noise = rng.multivariate_normal(np.zeros(self.cond_cov.shape[0]), self.cond_cov, size=mean.shape[0])
        return mean + noise

    def logpdf(self, gamma: np.ndarray, shared: np.ndarray) -> np.ndarray:
        gamma = np.atleast_2d(np.asarray(gamma, dtype=float))
        return np.asarray(self._dist.logpdf(gamma - self._conditional_mean(shared)), dtype=float)


# ---------------------------------------------------------------------------
# Affine warp of shared sites (mapped / warp bridge sampling)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AffineWarp:
    """Affine map of selected *shared* sites that aligns the simple and complex posteriors before the
    cross-density evaluations in BAR -- "mapped"/warp bridge sampling (Meng & Schilling 2002).

    A nested bridge can have poor overlap not because of the *added* dimension ``gamma`` (which is
    drawn from a proposal matched to its complex-posterior marginal) but because a *shared* parameter
    shifts between the two posteriors -- e.g. ``delta_g`` (ΔG₁), strongly correlated with ``delta_delta_g``
    (ΔΔG), sits at the first-site value under the complex (cooperative) model and at the common/average
    value under the equal-affinity null. This warp removes that mismatch.

    For each site ``x`` it uses ``T(x) = mu_complex + (x - mu_simple) * s`` (mapping the simple-posterior
    marginal of ``x`` onto the complex one, ``s = sd_complex / sd_simple``) and its inverse
    ``T_inv(y) = mu_simple + (y - mu_complex) / s``. ``T`` is applied to the simple sample before it is
    evaluated under the complex model (forward work); ``T_inv`` to the complex sample before it is
    evaluated under the simple model (reverse work). The only Jacobian is the constant
    ``sum_site ln s`` (a pure shift, ``scale=False``, has ``s = 1`` and no correction); no per-sample
    Jacobian is needed because the warped sites have flat priors and the warped values stay in support.
    """

    mu_simple: dict[str, float]
    mu_complex: dict[str, float]
    scale: dict[str, float]

    @classmethod
    def fit(
        cls,
        simple_samples: Mapping[str, np.ndarray],
        complex_samples: Mapping[str, np.ndarray],
        sites: Sequence[str],
        *,
        scale: bool = True,
    ) -> "AffineWarp":
        """Fit per-site means (and, if ``scale``, the SD ratio) from the two posterior sample sets."""
        mu_s: dict[str, float] = {}
        mu_c: dict[str, float] = {}
        sc: dict[str, float] = {}
        for site in sites:
            s_arr = np.asarray(simple_samples[site], dtype=float).ravel()
            c_arr = np.asarray(complex_samples[site], dtype=float).ravel()
            mu_s[site] = float(s_arr.mean())
            mu_c[site] = float(c_arr.mean())
            if scale:
                sd_s = float(s_arr.std())
                sd_c = float(c_arr.std())
                sc[site] = (sd_c / sd_s) if (sd_s > 0.0 and np.isfinite(sd_c) and sd_c > 0.0) else 1.0
            else:
                sc[site] = 1.0
        return cls(mu_simple=mu_s, mu_complex=mu_c, scale=sc)

    @property
    def log_scale_total(self) -> float:
        """The constant Jacobian term ``sum_site ln s`` (0 for a pure shift)."""
        return float(sum(np.log(s) for s in self.scale.values()))

    def to_complex_frame(self, simple_dict: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map simple-frame samples into the complex frame: ``x -> mu_complex + (x - mu_simple) * s``."""
        out = {k: np.asarray(v, dtype=float) for k, v in simple_dict.items()}
        for site, s in self.scale.items():
            if site in out:
                out[site] = self.mu_complex[site] + (out[site] - self.mu_simple[site]) * s
        return out

    def to_simple_frame(self, complex_dict: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map complex-frame samples into the simple frame: ``y -> mu_simple + (y - mu_complex) / s``."""
        out = {k: np.asarray(v, dtype=float) for k, v in complex_dict.items()}
        for site, s in self.scale.items():
            if site in out:
                out[site] = self.mu_simple[site] + (out[site] - self.mu_complex[site]) / s
        return out


# ---------------------------------------------------------------------------
# Nesting specification: the (theta_1, gamma) reparameterization for a model pair
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DirectNesting:
    """Degenerate nesting where the complex model simply adds ``gamma_sites`` to the simple model
    (all other parameters shared verbatim), e.g. racemic vs enantiomeric mixture with ``gamma = rho``.
    """

    gamma_sites: tuple[str, ...]

    @property
    def n_gamma(self) -> int:
        return len(self.gamma_sites)

    def split(self, complex_samples: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        gamma = np.column_stack([np.asarray(complex_samples[s], dtype=float) for s in self.gamma_sites])
        theta1 = {k: np.asarray(v, dtype=float) for k, v in complex_samples.items() if k not in self.gamma_sites}
        return theta1, gamma

    def augment(self, simple_samples: Mapping[str, np.ndarray], gamma: np.ndarray) -> dict[str, np.ndarray]:
        gamma = np.atleast_2d(np.asarray(gamma, dtype=float))
        out = {k: np.asarray(v, dtype=float) for k, v in simple_samples.items()}
        for index, site in enumerate(self.gamma_sites):
            out[site] = gamma[:, index]
        return out


# ---------------------------------------------------------------------------
# Result + estimator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BayesFactorResult:
    ln_bayes_factor: float  # ln BF in favor of the complex model over the simple model
    ln_bayes_factor_se: float  # closed-form asymptotic SE (Shirts et al. 2003)
    overlap: float  # Bennett harmonic-mean overlap of the two posteriors, in [0, 1]
    reliable: bool  # overlap >= threshold and the asymptotic SE is finite
    n_simple_samples: int
    n_complex_samples: int
    n_gamma: int
    warped: bool  # whether the shared sites were affine-warped (warp-BAR)
    warning: str | None = None
    conditional_proposal: bool = False  # whether gamma was drawn from f(gamma | theta_1) vs f(gamma)


def _latent_samples(samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep only sampled latent sites (drop deterministic ``q_model*`` and scalars)."""
    return {
        name: np.asarray(value, dtype=float)
        for name, value in samples.items()
        if not str(name).startswith("q_model") and np.asarray(value).ndim >= 1
    }


def _auto_warp(simple_samples, complex_samples, nesting) -> AffineWarp:
    """Fit an :class:`AffineWarp` over the shared (``theta_1``) sampled sites of the two posteriors.

    The shared sites are the simple-frame latent sites common to the simple posterior and to the
    ``theta_1`` obtained by ``nesting.split`` of the complex posterior (i.e. everything but the extra
    ``gamma``). This is what ``nested_bayes_factor`` warps by default.
    """
    simple_latent = _latent_samples(simple_samples)
    theta1_complex, _gamma = nesting.split(_latent_samples(complex_samples))
    shared = [name for name in simple_latent if name in theta1_complex]
    return AffineWarp.fit(simple_latent, theta1_complex, shared)


def _resolve_warp(warp, simple_samples, complex_samples, nesting) -> "AffineWarp | None":
    """``"auto"`` -> fit the default warp; ``None``/``False`` -> un-warped; else use as given."""
    if isinstance(warp, str) and warp == _WARP_AUTO:
        return _auto_warp(simple_samples, complex_samples, nesting)
    if warp is None or warp is False:
        return None
    return warp


def _batched_log_density(model, site_names: Sequence[str]):
    """Return a jitted+vmapped evaluator: dict-of-arrays -> array of unnormalized log densities."""
    site_names = tuple(site_names)

    def single(theta_row):
        params = {name: theta_row[index] for index, name in enumerate(site_names)}
        return log_density(model, (), {}, params)[0]

    batched = jax.jit(jax.vmap(single))

    def run(sample_dict: Mapping[str, np.ndarray]) -> np.ndarray:
        matrix = jnp.stack(
            [jnp.asarray(np.asarray(sample_dict[name], dtype=float)) for name in site_names], axis=1
        )
        return np.asarray(batched(matrix), dtype=float)

    return run


def _work_arrays(
    simple_samples: Mapping[str, np.ndarray],
    complex_samples: Mapping[str, np.ndarray],
    simple_model,
    complex_model,
    nesting,
    *,
    rng_seed: int,
    warp: "AffineWarp | None" = None,
    conditional_proposal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the forward (I-sample) and reverse (F-sample) work arrays for BAR.

    If ``warp`` is given, the named shared sites are mapped between the two posteriors before each
    cross-density evaluation (mapped/warp BAR): ``T_inv`` is applied to the complex sample's shared
    coordinate before the simple-model evaluation (reverse work), ``T`` to the simple sample before
    the complex-model evaluation (forward work), and the constant Jacobian ``ln s`` is added/subtracted.
    ``warp=None`` reproduces the un-warped estimator exactly.

    If ``conditional_proposal`` is set, the proposal for the added dimensions is the Gaussian conditional
    ``f(gamma | theta_1)`` (:class:`ConditionalGaussianProposal`) -- it conditions on the shared sites in
    the complex frame -- instead of the marginal ``f(gamma)``, capturing the ``theta_1``-``gamma``
    correlation the affine warp cannot.
    """
    simple_samples = _latent_samples(simple_samples)
    complex_samples = _latent_samples(complex_samples)
    log_density_simple = _batched_log_density(simple_model, list(simple_samples))
    log_density_complex = _batched_log_density(complex_model, list(complex_samples))
    log_s = warp.log_scale_total if warp is not None else 0.0

    theta1_from_complex, gamma_complex = nesting.split(complex_samples)
    theta1_eval = warp.to_simple_frame(theta1_from_complex) if warp is not None else theta1_from_complex
    simple_for_aug = warp.to_complex_frame(simple_samples) if warp is not None else simple_samples
    n_simple = next(iter(simple_samples.values())).shape[0]
    rng = np.random.default_rng(rng_seed)

    if conditional_proposal:
        # Condition gamma on the shared theta_1 (in the complex frame, the frame the proposal is fit in).
        shared_sites = list(theta1_from_complex)

        def _stack(sample_dict):
            return np.column_stack([np.asarray(sample_dict[name], dtype=float) for name in shared_sites])

        proposal = ConditionalGaussianProposal.fit(_stack(theta1_from_complex), gamma_complex)
        shared_aug = _stack(simple_for_aug)
        gamma_draws = proposal.sample(shared_aug, rng)
        logq_reverse = proposal.logpdf(gamma_complex, _stack(theta1_from_complex))
        logq_forward = proposal.logpdf(gamma_draws, shared_aug)
    else:
        proposal = GaussianProposal.fit(gamma_complex)
        gamma_draws = proposal.sample(n_simple, rng)
        logq_reverse = proposal.logpdf(gamma_complex)
        logq_forward = proposal.logpdf(gamma_draws)

    # State F (complex posterior): reverse work.
    u_f_f = -log_density_complex(complex_samples)
    u_f_i = -log_density_simple(theta1_eval) - logq_reverse
    w_R = u_f_i - u_f_f + log_s

    # State I (simple posterior augmented with proposal draws): forward work.
    complex_from_simple = nesting.augment(simple_for_aug, gamma_draws)
    u_i_i = -log_density_simple(simple_samples) - logq_forward
    u_i_f = -log_density_complex(complex_from_simple)
    w_F = u_i_f - u_i_i - log_s
    return np.asarray(w_F, dtype=float), np.asarray(w_R, dtype=float)


def nested_bayes_factor(
    simple_samples: Mapping[str, np.ndarray],
    complex_samples: Mapping[str, np.ndarray],
    simple_model,
    complex_model,
    nesting,
    *,
    rng_seed: int = 20260613,
    warp: "AffineWarp | str | None" = _WARP_AUTO,
    overlap_threshold: float = DEFAULT_NESTED_OVERLAP_THRESHOLD,
    conditional_proposal: bool = False,
) -> BayesFactorResult:
    """Estimate ``ln BF`` (complex over simple) for a nested model pair from posterior samples.

    This is the **alternative** estimator; the recommended one is per-model bridge sampling
    (:func:`bayesian_binding.evidence.bayes_factor_bridge`). By default (``warp="auto"``) the shared
    parameters are affine-warped to align the two posteriors before the cross-density evaluations
    (warp-BAR), which removes the shared-parameter shift that biases the un-warped nested bridge; pass
    ``warp=None`` for the un-warped diagnostic, or an explicit :class:`AffineWarp`.

    Set ``conditional_proposal=True`` to draw the added dimensions from ``f(gamma | theta_1)`` (a Gaussian
    conditional fit to the complex posterior; :class:`ConditionalGaussianProposal`) instead of the
    marginal ``f(gamma)``. This restores the ``theta_1``-``gamma`` correlation that the affine warp
    cannot, and can lift the overlap (and reduce the finite-sample bias) when ``gamma`` co-varies with the
    shared parameters; with no such correlation it reduces to the marginal proposal.

    ``simple_model`` / ``complex_model`` are no-argument NumPyro model closures built for the dataset
    that produced the samples; ``nesting`` is a :class:`DirectNesting`.
    The result carries a closed-form **asymptotic** standard error (Shirts et al. 2003) and the
    **Bennett overlap integral**; when the overlap is below ``overlap_threshold`` (default
    :data:`DEFAULT_NESTED_OVERLAP_THRESHOLD`, a deliberately stricter gate than the per-model bridge's,
    since the nested bridge is bias-prone at moderate overlap even after warping) the estimate is
    flagged unreliable (``reliable=False``) and the per-model bridge is recommended instead.
    """
    resolved_warp = _resolve_warp(warp, simple_samples, complex_samples, nesting)
    w_F, w_R = _work_arrays(
        simple_samples, complex_samples, simple_model, complex_model, nesting,
        rng_seed=rng_seed, warp=resolved_warp, conditional_proposal=conditional_proposal,
    )
    delta_f = bennett_acceptance_ratio(w_F, w_R)
    overlap, se = bridge_diagnostics(w_F, w_R, delta_f)
    reliable = bool(overlap >= overlap_threshold and np.isfinite(se))
    warning = None
    if not reliable:
        warning = (
            f"nested-bridge overlap {overlap:.3f} < {overlap_threshold:.2f}: the two posteriors overlap "
            "too little for the nested bridge to be trusted (it is bias-prone in this regime), so this "
            "Bayes factor is likely biased. Use per-model bridge sampling "
            "(bayesian_binding.evidence.bayes_factor_bridge)."
        )
    return BayesFactorResult(
        ln_bayes_factor=-delta_f,
        ln_bayes_factor_se=float(se),
        overlap=float(overlap),
        reliable=reliable,
        n_simple_samples=int(w_F.size),
        n_complex_samples=int(w_R.size),
        n_gamma=int(nesting.n_gamma),
        warped=resolved_warp is not None,
        warning=warning,
        conditional_proposal=bool(conditional_proposal),
    )


def bayes_factor_convergence(
    simple_samples: Mapping[str, np.ndarray],
    complex_samples: Mapping[str, np.ndarray],
    simple_model,
    complex_model,
    nesting,
    *,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    rng_seed: int = 20260613,
    warp: "AffineWarp | str | None" = _WARP_AUTO,
    conditional_proposal: bool = False,
) -> list[dict[str, float]]:
    """Return ``ln BF`` (complex over simple) vs. simulation length with closed-form asymptotic SEs.

    Warp-BAR is applied by default, and ``conditional_proposal`` selects the ``f(gamma | theta_1)``
    proposal (see :func:`nested_bayes_factor`). The work arrays are computed once; for each fraction of
    the samples the BAR estimate uses the leading prefix of each work array, and the standard error is the
    closed-form **asymptotic** BAR SE (Shirts et al. 2003) -- not a bootstrap. Returns rows with
    ``fraction``, ``n_complex_samples``, ``ln_bayes_factor``, ``ln_bayes_factor_se``, and the Bennett
    ``overlap``.
    """
    resolved_warp = _resolve_warp(warp, simple_samples, complex_samples, nesting)
    w_F, w_R = _work_arrays(
        simple_samples, complex_samples, simple_model, complex_model, nesting,
        rng_seed=rng_seed, warp=resolved_warp, conditional_proposal=conditional_proposal,
    )
    w_F = w_F[np.isfinite(w_F)]
    w_R = w_R[np.isfinite(w_R)]
    rows: list[dict[str, float]] = []
    for fraction in fractions:
        n_f = max(2, int(round(fraction * w_F.size)))
        n_r = max(2, int(round(fraction * w_R.size)))
        prefix_f = w_F[:n_f]
        prefix_r = w_R[:n_r]
        ln_bf = -bennett_acceptance_ratio(prefix_f, prefix_r)
        overlap, se = bridge_diagnostics(prefix_f, prefix_r, -ln_bf)
        rows.append(
            {
                "fraction": float(fraction),
                "n_complex_samples": int(n_r),
                "ln_bayes_factor": float(ln_bf),
                "ln_bayes_factor_se": float(se),
                "overlap": float(overlap),
            }
        )
    return rows
