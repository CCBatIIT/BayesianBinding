"""NumPyro model builders and sampling helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesian_binding import _jax_config as _jax_config

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from bayesian_binding.constants import MICROCALORIES_PER_KCAL
from bayesian_binding.data import CuratedITCCurve, ITCExperiment
from bayesian_binding.models import MODEL_REGISTRY
from bayesian_binding.scattering import WAXSReduced, add_waxs_observation, binding_1to1_concentrations


@dataclass(frozen=True)
class PriorBounds:
    low: float
    high: float


DEFAULT_BOUNDS = {
    "delta_g": PriorBounds(-40.0, 4.0),
    "delta_delta_g": PriorBounds(-20.0, 20.0),
    "delta_g_dimer": PriorBounds(-40.0, 10.0),
    "delta_g_binding": PriorBounds(-40.0, 4.0),
    "delta_delta_g_binding": PriorBounds(-20.0, 20.0),
    # Monomer ligand site is weaker than (or equal to) the first dimer site, so the free-energy offset
    # delta_g_monomer - delta_g_binding is non-negative; 0 means equal affinity, 15 kcal/mol ~ 11 decades
    # weaker. A soft non-negative prior here replaces the deterministic fit's hard gap bound, which railed.
    "delta_delta_g_monomer": PriorBounds(0.0, 15.0),
    "delta_h": PriorBounds(-100.0, 100.0),
    "delta_h_first": PriorBounds(-150.0, 150.0),
    "delta_h_second": PriorBounds(-150.0, 150.0),
    "delta_h_dimer": PriorBounds(-150.0, 150.0),
    "delta_h_monomer": PriorBounds(-150.0, 150.0),
}


# Priors for the enantiomeric-mixture models, following Nguyen et al. 2022 (Eqs 6-7, 11).
# ``delta_delta_g = delta_g2 - delta_g1`` is constrained non-negative so ligand 1 is the
# higher-affinity binder; this differs from the cooperative model, where a signed
# ``delta_delta_g`` encodes positive/negative cooperativity. The bound is therefore
# model-scoped and overlaid on ``DEFAULT_BOUNDS`` only for the mixture models, rather than
# living in ``DEFAULT_BOUNDS`` where it would clash with the cooperative parameterization.
ENANTIOMER_MIXTURE_BOUNDS = {
    "delta_g1": PriorBounds(-40.0, 40.0),
    "delta_delta_g": PriorBounds(0.0, 40.0),
    "delta_h1": PriorBounds(-100.0, 100.0),
    "delta_h2": PriorBounds(-100.0, 100.0),
}

# Parameter names the mixture models pass to ``expected_heats`` (besides ``rho`` and the
# per-experiment ``heat_offset``). ``rho`` is handled separately because it is the syringe
# composition, fixed at 0.5 for the racemic model and free on (0, 1) for the enantiomeric.
_ENANTIOMER_MIXTURE_THERMODYNAMIC_NAMES = ("delta_g1", "delta_delta_g", "delta_h1", "delta_h2")
_MIXTURE_MODEL_NAMES = ("racemic_mixture", "enantiomeric_mixture")


# Thermodynamic parameters shared across experiments in a global fit. These are the
# kwargs passed to each model's ``expected_heats`` other than the per-experiment
# metadata (volumes, concentrations, temperature) and ``heat_offset``.
GLOBAL_THERMODYNAMIC_PARAMETERS = {
    "two_component": ("delta_g", "delta_h"),
    "cooperative": ("delta_g", "delta_delta_g", "delta_h_first", "delta_h_second"),
    # No-cooperativity null: the two sites are forced equal, so only delta_g and delta_h are fit.
    "cooperative_equivalent_sites": ("delta_g", "delta_h"),
    # Free-energy-only null: delta_delta_g fixed to 0 but both step enthalpies free.
    "cooperative_equal_affinity": ("delta_g", "delta_h_first", "delta_h_second"),
    "dimerization_cooperative": (
        "delta_g_dimer",
        "delta_g_binding",
        "delta_delta_g_binding",
        "delta_h_dimer",
        "delta_h_first",
        "delta_h_second",
    ),
    "dimerization_monomer_cooperative": (
        "delta_g_dimer",
        "delta_g_binding",
        "delta_delta_g_binding",
        "delta_delta_g_monomer",
        "delta_h_dimer",
        "delta_h_first",
        "delta_h_second",
        "delta_h_monomer",
    ),
    # The mixture models share their binding free energies and enthalpies across
    # experiments; the syringe composition ``rho`` is handled separately (shared, fixed
    # for the racemic model and free for the enantiomeric one).
    "racemic_mixture": _ENANTIOMER_MIXTURE_THERMODYNAMIC_NAMES,
    "enantiomeric_mixture": _ENANTIOMER_MIXTURE_THERMODYNAMIC_NAMES,
}


def _thermo_to_heat_kwargs(model_name: str, thermodynamics: Mapping[str, Any]) -> dict[str, Any]:
    """Map sampled thermodynamic parameters to ``expected_heats`` kwargs.

    Identity except for the two cooperative nulls, whose reduced parameters are expanded to the full
    cooperative kwargs: ``cooperative_equivalent_sites`` (two fully equal sites) and
    ``cooperative_equal_affinity`` (``delta_delta_g = 0`` only, enthalpies free).
    """
    if model_name in ("cooperative_equivalent_sites", "cooperative_equal_affinity"):
        return MODEL_REGISTRY[model_name].heat_kwargs(thermodynamics)
    return dict(thermodynamics)


def _resolve_bounds(
    model_name: str, bounds: Mapping[str, PriorBounds] | None
) -> dict[str, PriorBounds]:
    """Return prior bounds for ``model_name`` with user overrides applied last.

    Mixture models overlay :data:`ENANTIOMER_MIXTURE_BOUNDS` on top of
    :data:`DEFAULT_BOUNDS` (chiefly to flip ``delta_delta_g`` to a non-negative prior),
    so the same ``delta_delta_g`` key carries the correct, model-appropriate range.
    """
    resolved = dict(DEFAULT_BOUNDS)
    if model_name in _MIXTURE_MODEL_NAMES:
        resolved.update(ENANTIOMER_MIXTURE_BOUNDS)
    resolved.update(bounds or {})
    return resolved


def _normal_log_concentration(name: str, stated_value: float, relative_uncertainty: float):
    sigma = float(np.sqrt(np.log1p(relative_uncertainty**2)))
    return numpyro.sample(name, dist.Normal(jnp.log(stated_value), sigma))


def _sample_concentration(
    kind: str,
    stated_molar: float,
    fixed: Mapping[str, float],
    *,
    uniform: bool,
    concentration_range_factor: float,
    relative_uncertainty: float,
):
    """Return a molar concentration, registering the appropriate latent site.

    With an informative (lognormal) prior the latent site is the log
    concentration, ``log_{kind}_concentration``, matching the rest of the
    codebase. With an uninformative uniform prior on the linear concentration
    (used when the stated value is unavailable, as in Nguyen et al. 2022) the
    latent site is the linear concentration, ``{kind}_concentration_molar``,
    drawn on ``[stated / factor, stated * factor]``.
    """
    log_name = f"log_{kind}_concentration"
    linear_name = f"{kind}_concentration_molar"
    if log_name in fixed:
        return jnp.exp(jnp.asarray(fixed[log_name]))
    if linear_name in fixed:
        return jnp.asarray(fixed[linear_name])
    if uniform:
        low = stated_molar / concentration_range_factor
        high = stated_molar * concentration_range_factor
        return numpyro.sample(linear_name, dist.Uniform(low, high))
    return jnp.exp(_normal_log_concentration(log_name, stated_molar, relative_uncertainty))


def _uniform(name: str, low: float, high: float):
    return numpyro.sample(name, dist.Uniform(low, high))


def _get_fixed_or_sample(fixed: Mapping[str, float], name: str, sampler):
    if name in fixed:
        return jnp.asarray(fixed[name])
    return sampler()


def _sample_mixture_composition(racemic: bool, fixed: Mapping[str, float]):
    """Return the syringe composition ``rho`` for an enantiomeric-mixture model.

    Fixed at 0.5 for the racemic-mixture model (a 1:1 racemate) and sampled on ``(0, 1)``
    for the enantiomeric-mixture model. Shared by the single-experiment and global model
    builders so ``rho`` registers as one latent site in both.
    """
    if racemic:
        return _get_fixed_or_sample(fixed, "rho", lambda: jnp.asarray(0.5))
    return _get_fixed_or_sample(fixed, "rho", lambda: _uniform("rho", 0.0, 1.0))


def _heat_offset_bounds(heats: np.ndarray) -> tuple[float, float]:
    interval = float(np.nanmax(heats) - np.nanmin(heats))
    if not np.isfinite(interval) or interval == 0.0:
        interval = max(1.0, float(np.nanstd(heats)))
    return float(np.nanmin(heats) - interval), float(np.nanmax(heats) + interval)


def _log_sigma_bounds(heats: np.ndarray) -> tuple[float, float]:
    tail = heats[-min(4, heats.size) :]
    guess = float(np.log(max(np.nanstd(tail), 1.0e-6)))
    return guess - 10.0, guess + 5.0


def build_numpyro_model(
    experiment: ITCExperiment,
    *,
    model_name: str = "two_component",
    fixed: Mapping[str, float] | None = None,
    concentration_relative_uncertainty: float = 0.10,
    bounds: Mapping[str, PriorBounds] | None = None,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
    concentration_range_factor: float = 10.0,
):
    """Return a NumPyro model closure for an ITC experiment.

    Fixed parameters should be supplied in the same names used by the model,
    for example `fixed={"delta_delta_g": 0.0}` or
    `fixed={"delta_g_dimer": -7.0}`.

    Set `uniform_cell_concentration` / `uniform_syringe_concentration` to use an
    uninformative uniform prior on the linear concentration (drawn on
    `[stated / factor, stated * factor]`) instead of the default informative
    lognormal prior. This reproduces the Nguyen et al. 2022 treatment of datasets
    whose stated concentrations are unavailable.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name!r}; choices are {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name]
    fixed = dict(fixed or {})
    if "cell_concentration_molar" in fixed:
        fixed["log_cell_concentration"] = float(np.log(fixed.pop("cell_concentration_molar")))
    if "syringe_concentration_molar" in fixed:
        fixed["log_syringe_concentration"] = float(np.log(fixed.pop("syringe_concentration_molar")))
    if experiment.cell_concentration_molar <= 0.0 and "log_cell_concentration" not in fixed:
        raise ValueError("Cell concentration must be positive; pass metadata or fix log_cell_concentration.")
    if experiment.syringe_concentration_molar <= 0.0 and "log_syringe_concentration" not in fixed:
        raise ValueError("Syringe concentration must be positive; pass metadata or fix log_syringe_concentration.")
    all_bounds = _resolve_bounds(model_name, bounds)
    heats = np.asarray(experiment.heats_microcalorie, dtype=float)
    offset_low, offset_high = _heat_offset_bounds(heats)
    log_sigma_low, log_sigma_high = _log_sigma_bounds(heats)

    def numpyro_model():
        cell_concentration_molar = _sample_concentration(
            "cell",
            experiment.cell_concentration_molar,
            fixed,
            uniform=uniform_cell_concentration,
            concentration_range_factor=concentration_range_factor,
            relative_uncertainty=concentration_relative_uncertainty,
        )
        syringe_concentration_molar = _sample_concentration(
            "syringe",
            experiment.syringe_concentration_molar,
            fixed,
            uniform=uniform_syringe_concentration,
            concentration_range_factor=concentration_range_factor,
            relative_uncertainty=concentration_relative_uncertainty,
        )
        common: dict[str, Any] = {
            "injection_volumes_liter": jnp.asarray(experiment.injection_volumes_liter),
            "cell_volume_liter": float(experiment.cell_volume_liter),
            "cell_concentration_molar": cell_concentration_molar,
            "syringe_concentration_molar": syringe_concentration_molar,
            "temperature_k": float(experiment.temperature_k),
        }
        heat_offset = _get_fixed_or_sample(
            fixed,
            "heat_offset",
            lambda: _uniform("heat_offset", offset_low, offset_high),
        )
        log_sigma = _get_fixed_or_sample(
            fixed,
            "log_sigma",
            lambda: _uniform("log_sigma", log_sigma_low, log_sigma_high),
        )

        if model_name == "two_component":
            b = all_bounds["delta_g"]
            delta_g = _get_fixed_or_sample(fixed, "delta_g", lambda: _uniform("delta_g", b.low, b.high))
            b = all_bounds["delta_h"]
            delta_h = _get_fixed_or_sample(fixed, "delta_h", lambda: _uniform("delta_h", b.low, b.high))
            q_model = model.expected_heats(
                **common,
                delta_g=delta_g,
                delta_h=delta_h,
                heat_offset=heat_offset,
            )
        elif model_name == "cooperative":
            b = all_bounds["delta_g"]
            delta_g = _get_fixed_or_sample(fixed, "delta_g", lambda: _uniform("delta_g", b.low, b.high))
            b = all_bounds["delta_delta_g"]
            delta_delta_g = _get_fixed_or_sample(
                fixed, "delta_delta_g", lambda: _uniform("delta_delta_g", b.low, b.high)
            )
            b = all_bounds["delta_h_first"]
            delta_h_first = _get_fixed_or_sample(
                fixed, "delta_h_first", lambda: _uniform("delta_h_first", b.low, b.high)
            )
            b = all_bounds["delta_h_second"]
            delta_h_second = _get_fixed_or_sample(
                fixed, "delta_h_second", lambda: _uniform("delta_h_second", b.low, b.high)
            )
            q_model = model.expected_heats(
                **common,
                delta_g=delta_g,
                delta_delta_g=delta_delta_g,
                delta_h_first=delta_h_first,
                delta_h_second=delta_h_second,
                heat_offset=heat_offset,
            )
        elif model_name == "cooperative_equivalent_sites":
            # No-cooperativity null: the two sites are equal, so only delta_g and delta_h are fit.
            b = all_bounds["delta_g"]
            delta_g = _get_fixed_or_sample(fixed, "delta_g", lambda: _uniform("delta_g", b.low, b.high))
            b = all_bounds["delta_h"]
            delta_h = _get_fixed_or_sample(fixed, "delta_h", lambda: _uniform("delta_h", b.low, b.high))
            q_model = model.expected_heats(
                **common,
                **model.heat_kwargs({"delta_g": delta_g, "delta_h": delta_h}),
                heat_offset=heat_offset,
            )
        elif model_name == "cooperative_equal_affinity":
            # Free-energy-only null: delta_delta_g fixed to 0, but the two step enthalpies are free.
            b = all_bounds["delta_g"]
            delta_g = _get_fixed_or_sample(fixed, "delta_g", lambda: _uniform("delta_g", b.low, b.high))
            b = all_bounds["delta_h_first"]
            delta_h_first = _get_fixed_or_sample(
                fixed, "delta_h_first", lambda: _uniform("delta_h_first", b.low, b.high)
            )
            b = all_bounds["delta_h_second"]
            delta_h_second = _get_fixed_or_sample(
                fixed, "delta_h_second", lambda: _uniform("delta_h_second", b.low, b.high)
            )
            q_model = model.expected_heats(
                **common,
                **model.heat_kwargs(
                    {"delta_g": delta_g, "delta_h_first": delta_h_first, "delta_h_second": delta_h_second}
                ),
                heat_offset=heat_offset,
            )
        elif model_name in ("dimerization_cooperative", "dimerization_monomer_cooperative"):
            parameter_names = list(GLOBAL_THERMODYNAMIC_PARAMETERS[model_name])
            sampled = {}
            for parameter_name in parameter_names:
                b = all_bounds[parameter_name]
                sampled[parameter_name] = _get_fixed_or_sample(
                    fixed,
                    parameter_name,
                    lambda parameter_name=parameter_name, b=b: _uniform(parameter_name, b.low, b.high),
                )
            q_model = model.expected_heats(
                **common,
                heat_offset=heat_offset,
                **sampled,
            )
        elif model_name in _MIXTURE_MODEL_NAMES:
            # Priors follow Nguyen et al. 2022 (Eqs 6-7, 11); the non-negative
            # delta_delta_g prior (ligand 1 is the higher-affinity binder) is carried by
            # ENANTIOMER_MIXTURE_BOUNDS, overlaid in _resolve_bounds.
            rho = _sample_mixture_composition(model.racemic, fixed)
            sampled = {}
            for parameter_name in _ENANTIOMER_MIXTURE_THERMODYNAMIC_NAMES:
                b = all_bounds[parameter_name]
                sampled[parameter_name] = _get_fixed_or_sample(
                    fixed,
                    parameter_name,
                    lambda parameter_name=parameter_name, b=b: _uniform(parameter_name, b.low, b.high),
                )
            q_model = model.expected_heats(
                **common,
                rho=rho,
                heat_offset=heat_offset,
                **sampled,
            )
        else:
            raise AssertionError("registry and model builder are out of sync")

        numpyro.deterministic("q_model", q_model)
        numpyro.sample(
            "q_obs",
            dist.Normal(q_model, jnp.exp(log_sigma)),
            obs=jnp.asarray(experiment.heats_microcalorie),
        )

    return numpyro_model


def run_mcmc(
    experiment: ITCExperiment,
    *,
    model_name: str = "two_component",
    rng_seed: int = 20260610,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    fixed: Mapping[str, float] | None = None,
    concentration_relative_uncertainty: float = 0.10,
    uniform_cell_concentration: bool = False,
    uniform_syringe_concentration: bool = False,
    concentration_range_factor: float = 10.0,
    target_accept_prob: float = 0.8,
    chain_method: str = "parallel",
    init_params: Mapping[str, Any] | None = None,
    initialize_with_map: bool = False,
    progress_bar: bool = True,
) -> MCMC:
    """Run NUTS for an ITC model and return the NumPyro `MCMC` object."""
    concentration_kwargs = dict(
        uniform_cell_concentration=uniform_cell_concentration,
        uniform_syringe_concentration=uniform_syringe_concentration,
        concentration_range_factor=concentration_range_factor,
    )
    if initialize_with_map:
        from bayesian_binding.optimization import fit_map

        init_params = fit_map(
            experiment,
            model_name=model_name,
            fixed=fixed,
            concentration_relative_uncertainty=concentration_relative_uncertainty,
            **concentration_kwargs,
        ).params
    model = build_numpyro_model(
        experiment,
        model_name=model_name,
        fixed=fixed,
        concentration_relative_uncertainty=concentration_relative_uncertainty,
        **concentration_kwargs,
    )
    if init_params is None:
        kernel = NUTS(model, target_accept_prob=target_accept_prob)
    else:
        kernel = NUTS(
            model,
            init_strategy=init_to_value(values=dict(init_params)),
            target_accept_prob=target_accept_prob,
        )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(rng_seed))
    return mcmc


def build_global_numpyro_model(
    experiments: Sequence[ITCExperiment],
    *,
    model_name: str = "cooperative",
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, PriorBounds] | None = None,
    fit_concentration_scale: bool = True,
    concentration_scale_prior_mean: float = 1.0,
    concentration_scale_relative_sd: float = 0.25,
    fit_syringe_concentration_scale: bool = False,
    syringe_concentration_scale_relative_sd: float = 0.10,
):
    """Return a NumPyro model that fits several ITC experiments globally.

    The thermodynamic parameters listed in ``GLOBAL_THERMODYNAMIC_PARAMETERS`` for
    ``model_name`` are shared across every experiment, while each experiment keeps
    its own ``heat_offset_{i}`` and ``log_sigma_{i}`` (indexed by position in
    ``experiments``). This reproduces the global-fit protocol of Bonin et al. 2019,
    in which a single set of macroscopic affinities and enthalpies is applied to
    every isotherm.

    A single multiplicative ``cell_concentration_scale`` is shared across the
    experiments and applied to each stated cell concentration. It captures the
    protein-concentration scaling factor used by Bonin et al. to absorb extinction
    coefficient error and any binding-incompetent fraction. When
    ``fit_concentration_scale`` is True it carries a lognormal prior centered on
    ``concentration_scale_prior_mean`` (1.0 by default, i.e. the stated
    concentrations) with log standard deviation ``concentration_scale_relative_sd``;
    otherwise it is fixed at ``concentration_scale_prior_mean``.

    Syringe (titrant) concentrations are held at their stated values by default. When
    ``fit_syringe_concentration_scale`` is True a single multiplicative
    ``syringe_concentration_scale`` is shared across the system's isotherms (one common
    titrant-stock error) and applied to every stated syringe concentration; it carries a
    lognormal prior with unit median and log standard deviation
    ``syringe_concentration_scale_relative_sd`` (~10% by default), mirroring the shared
    ``cell_concentration_scale`` and following the lognormal concentration priors of
    Nguyen et al. 2018/2022.

    For the enantiomeric-mixture models the shared thermodynamics are
    ``delta_g1``/``delta_delta_g``/``delta_h1``/``delta_h2`` and the syringe composition
    ``rho`` is a single shared site (fixed at 0.5 for ``racemic_mixture``, free on
    ``(0, 1)`` for ``enantiomeric_mixture``), so several titrations of one enantiomeric
    stock are described by one composition.

    ``fixed`` may pin any latent site, e.g. ``{"delta_delta_g": 0.0}`` for an
    independent-binding null model or ``{"cell_concentration_scale": 0.78}``.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_name!r}; choices are {sorted(MODEL_REGISTRY)}")
    if model_name not in GLOBAL_THERMODYNAMIC_PARAMETERS:
        raise KeyError(f"Global fitting is not supported for model {model_name!r}")
    if len(experiments) == 0:
        raise ValueError("Global fitting requires at least one experiment.")
    model = MODEL_REGISTRY[model_name]
    is_mixture = model_name in _MIXTURE_MODEL_NAMES
    fixed = dict(fixed or {})
    all_bounds = _resolve_bounds(model_name, bounds)
    thermodynamic_names = GLOBAL_THERMODYNAMIC_PARAMETERS[model_name]
    offset_bounds = [
        _heat_offset_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
        for experiment in experiments
    ]
    log_sigma_bounds = [
        _log_sigma_bounds(np.asarray(experiment.heats_microcalorie, dtype=float))
        for experiment in experiments
    ]

    def numpyro_model():
        thermodynamics: dict[str, Any] = {}
        for name in thermodynamic_names:
            b = all_bounds[name]
            thermodynamics[name] = _get_fixed_or_sample(
                fixed, name, lambda name=name, b=b: _uniform(name, b.low, b.high)
            )
        if fit_concentration_scale:
            cell_concentration_scale = _get_fixed_or_sample(
                fixed,
                "cell_concentration_scale",
                lambda: numpyro.sample(
                    "cell_concentration_scale",
                    dist.LogNormal(
                        float(np.log(concentration_scale_prior_mean)),
                        float(concentration_scale_relative_sd),
                    ),
                ),
            )
        else:
            cell_concentration_scale = jnp.asarray(
                fixed.get("cell_concentration_scale", concentration_scale_prior_mean)
            )
        if fit_syringe_concentration_scale:
            syringe_concentration_scale = _get_fixed_or_sample(
                fixed,
                "syringe_concentration_scale",
                lambda: numpyro.sample(
                    "syringe_concentration_scale",
                    dist.LogNormal(0.0, float(syringe_concentration_scale_relative_sd)),
                ),
            )
        else:
            syringe_concentration_scale = jnp.asarray(fixed.get("syringe_concentration_scale", 1.0))
        # The syringe composition rho is a single shared site across experiments.
        extra: dict[str, Any] = {}
        if is_mixture:
            extra["rho"] = _sample_mixture_composition(model.racemic, fixed)

        for index, experiment in enumerate(experiments):
            heat_offset = _get_fixed_or_sample(
                fixed,
                f"heat_offset_{index}",
                lambda index=index: _uniform(f"heat_offset_{index}", *offset_bounds[index]),
            )
            log_sigma = _get_fixed_or_sample(
                fixed,
                f"log_sigma_{index}",
                lambda index=index: _uniform(f"log_sigma_{index}", *log_sigma_bounds[index]),
            )
            syringe_concentration_molar = (
                float(experiment.syringe_concentration_molar) * syringe_concentration_scale
            )
            q_model = model.expected_heats(
                jnp.asarray(experiment.injection_volumes_liter),
                cell_volume_liter=float(experiment.cell_volume_liter),
                cell_concentration_molar=float(experiment.cell_concentration_molar)
                * cell_concentration_scale,
                syringe_concentration_molar=syringe_concentration_molar,
                temperature_k=float(experiment.temperature_k),
                heat_offset=heat_offset,
                **_thermo_to_heat_kwargs(model_name, thermodynamics),
                **extra,
            )
            numpyro.deterministic(f"q_model_{index}", q_model)
            numpyro.sample(
                f"q_obs_{index}",
                dist.Normal(q_model, jnp.exp(log_sigma)),
                obs=jnp.asarray(experiment.heats_microcalorie),
            )

    return numpyro_model


def run_mcmc_global(
    experiments: Sequence[ITCExperiment],
    *,
    model_name: str = "cooperative",
    rng_seed: int = 20260610,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, PriorBounds] | None = None,
    fit_concentration_scale: bool = True,
    concentration_scale_prior_mean: float = 1.0,
    concentration_scale_relative_sd: float = 0.25,
    target_accept_prob: float = 0.9,
    init_params: Mapping[str, Any] | None = None,
    initialize_with_map: bool = False,
    progress_bar: bool = True,
) -> MCMC:
    """Run NUTS for a global multi-experiment ITC fit and return the ``MCMC`` object."""
    scale_kwargs = dict(
        fit_concentration_scale=fit_concentration_scale,
        concentration_scale_prior_mean=concentration_scale_prior_mean,
        concentration_scale_relative_sd=concentration_scale_relative_sd,
    )
    if initialize_with_map:
        from bayesian_binding.optimization import fit_map_global

        init_params = fit_map_global(
            experiments,
            model_name=model_name,
            fixed=fixed,
            bounds=bounds,
            **scale_kwargs,
        ).params
    model = build_global_numpyro_model(
        experiments,
        model_name=model_name,
        fixed=fixed,
        bounds=bounds,
        **scale_kwargs,
    )
    if init_params is None:
        kernel = NUTS(model, target_accept_prob=target_accept_prob)
    else:
        kernel = NUTS(
            model,
            init_strategy=init_to_value(values=dict(init_params)),
            target_accept_prob=target_accept_prob,
        )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(rng_seed))
    return mcmc


# ---------------------------------------------------------------------------
# Curated sequential-titration ITC models (pre-propagated concentrations)
# ---------------------------------------------------------------------------
# Models whose ``enthalpy_density`` expects monomer-equivalent total protein (2 * dimer total). The
# preformed-dimer model (``cooperative``) instead takes the dimer total directly.
_MONOMER_EQUIVALENT_MODELS = ("dimerization_cooperative", "dimerization_monomer_cooperative")
_CURATED_MODELS = ("cooperative", "dimerization_cooperative", "dimerization_monomer_cooperative")


def _curated_protein_factor(model_name: str) -> float:
    return 2.0 if model_name in _MONOMER_EQUIVALENT_MODELS else 1.0


def _sample_curated_thermodynamics(
    model_name: str,
    all_bounds: Mapping[str, PriorBounds],
    fixed: Mapping[str, Any],
    *,
    delta_g_dimer_prior: tuple[float, float] | None,
    delta_h_prior_sd: float,
) -> dict[str, Any]:
    """Sample the shared thermodynamics for a curated fit.

    ``delta_g_dimer`` carries a weakly-informative Normal prior (the dimerization constant is only
    weakly identified by a narrow protein-concentration range); every ``delta_h*`` term carries a
    weakly-informative ``Normal(0, delta_h_prior_sd)`` prior to stop the dilution/offset nuisances and
    the binding enthalpies from compensating without bound (the deterministic fit's failure mode). All
    other free energies keep the package's uniform priors, including the non-negative
    ``delta_delta_g_monomer`` (monomer site no tighter than the first dimer site).
    """
    thermo: dict[str, Any] = {}
    for name in GLOBAL_THERMODYNAMIC_PARAMETERS[model_name]:
        if name in fixed:
            thermo[name] = jnp.asarray(fixed[name])
        elif name == "delta_g_dimer" and delta_g_dimer_prior is not None:
            mean, sd = delta_g_dimer_prior
            thermo[name] = numpyro.sample(name, dist.Normal(float(mean), float(sd)))
        elif name.startswith("delta_h"):
            thermo[name] = numpyro.sample(name, dist.Normal(0.0, float(delta_h_prior_sd)))
        else:
            b = all_bounds[name]
            thermo[name] = _uniform(name, b.low, b.high)
    return thermo


def build_global_curated_itc_model(
    curves: Sequence[CuratedITCCurve],
    *,
    model_name: str = "dimerization_monomer_cooperative",
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, PriorBounds] | None = None,
    delta_g_dimer_prior: tuple[float, float] | None = (-8.0, 3.0),
    delta_h_prior_sd: float = 20.0,
    fit_protein_concentration_scale: bool = False,
    protein_concentration_scale_relative_sd: float = 0.10,
):
    """Return a NumPyro model for a global fit to several curated sequential-titration curves.

    For each :class:`CuratedITCCurve` the integrated heat of injection ``i`` is predicted directly from
    the model's ``enthalpy_density`` using the pre-propagated before/after totals::

        q_i = V_cell * (H(after_i) - f_i * H(before_i)) * 1e9 + heat_offset_curve   [microcalories]

    The shared thermodynamics (``model_name``) are common to every curve; each curve keeps its own
    ``heat_offset_{c}`` (baseline / constant injection heat, which for these constant-volume,
    constant-syringe segments also absorbs the otherwise-degenerate ligand-dilution term) and Gaussian
    noise scale ``log_sigma_{c}``. Only ``include_mask`` injections enter the likelihood; the rest are
    available for posterior predictive checks. ``model_name`` must be one of ``cooperative`` (preformed
    dimer), ``dimerization_cooperative``, or ``dimerization_monomer_cooperative``.

    An optional shared ``protein_concentration_scale`` (default off) multiplies the curated dimer totals
    to absorb extinction-coefficient / active-fraction error; it is disabled by default because it
    aliases the weakly-identified dimerization constant.
    """
    if model_name not in _CURATED_MODELS:
        raise KeyError(
            f"Curated fitting supports {_CURATED_MODELS}; got {model_name!r}."
        )
    if len(curves) == 0:
        raise ValueError("Curated fitting requires at least one curve.")
    model = MODEL_REGISTRY[model_name]
    fixed = dict(fixed or {})
    all_bounds = _resolve_bounds(model_name, bounds)
    factor = _curated_protein_factor(model_name)

    prepared = []
    for curve in curves:
        mask = np.asarray(curve.include_mask, dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Curve {curve.name!r} has no included injections.")
        observed = np.asarray(curve.observed_heats_microcalorie, dtype=float)
        included_index = jnp.asarray(np.flatnonzero(mask))
        prepared.append(
            {
                "name": curve.name,
                "cell_volume_liter": float(curve.cell_volume_liter),
                "temperature_k": float(curve.temperature_k),
                "dilution": jnp.asarray(curve.dilution_factors),
                "protein_before": jnp.asarray(curve.dimer_before_molar) * factor,
                "protein_after": jnp.asarray(curve.dimer_after_molar) * factor,
                "ligand_before": jnp.asarray(curve.ligand_before_molar),
                "ligand_after": jnp.asarray(curve.ligand_after_molar),
                "included_index": included_index,
                "observed_included": jnp.asarray(observed[mask]),
                "offset_bounds": _heat_offset_bounds(observed[mask]),
                "log_sigma_bounds": _log_sigma_bounds(observed[mask]),
            }
        )

    def numpyro_model():
        thermo = _sample_curated_thermodynamics(
            model_name,
            all_bounds,
            fixed,
            delta_g_dimer_prior=delta_g_dimer_prior,
            delta_h_prior_sd=delta_h_prior_sd,
        )
        if fit_protein_concentration_scale and "protein_concentration_scale" not in fixed:
            scale = numpyro.sample(
                "protein_concentration_scale",
                dist.LogNormal(0.0, float(protein_concentration_scale_relative_sd)),
            )
        else:
            scale = jnp.asarray(fixed.get("protein_concentration_scale", 1.0))

        for index, curve in enumerate(prepared):
            heat_offset = _get_fixed_or_sample(
                fixed,
                f"heat_offset_{index}",
                lambda index=index, b=curve["offset_bounds"]: _uniform(f"heat_offset_{index}", *b),
            )
            log_sigma = _get_fixed_or_sample(
                fixed,
                f"log_sigma_{index}",
                lambda index=index, b=curve["log_sigma_bounds"]: _uniform(f"log_sigma_{index}", *b),
            )
            density_before = model.enthalpy_density(
                curve["protein_before"] * scale, curve["ligand_before"], curve["temperature_k"], **thermo
            )
            density_after = model.enthalpy_density(
                curve["protein_after"] * scale, curve["ligand_after"], curve["temperature_k"], **thermo
            )
            q_model = (
                curve["cell_volume_liter"]
                * (density_after - curve["dilution"] * density_before)
                * MICROCALORIES_PER_KCAL
                + heat_offset
            )
            numpyro.deterministic(f"q_model_{index}", q_model)
            numpyro.sample(
                f"q_obs_{index}",
                dist.Normal(q_model[curve["included_index"]], jnp.exp(log_sigma)),
                obs=curve["observed_included"],
            )

    return numpyro_model


def curated_sample_site_names(
    curves: Sequence[CuratedITCCurve],
    model_name: str,
    fixed: Mapping[str, float] | None = None,
    *,
    fit_protein_concentration_scale: bool = False,
) -> tuple[str, ...]:
    """Latent sample-site names for a curated global fit (for MAP packing / bridge sampling)."""
    fixed = dict(fixed or {})
    names = list(GLOBAL_THERMODYNAMIC_PARAMETERS[model_name])
    if fit_protein_concentration_scale:
        names.append("protein_concentration_scale")
    for index in range(len(curves)):
        names.append(f"heat_offset_{index}")
        names.append(f"log_sigma_{index}")
    return tuple(name for name in names if name not in fixed)


def fit_map_curated(
    curves: Sequence[CuratedITCCurve],
    *,
    model_name: str = "dimerization_monomer_cooperative",
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, PriorBounds] | None = None,
    delta_g_dimer_prior: tuple[float, float] | None = (-8.0, 3.0),
    delta_h_prior_sd: float = 20.0,
    fit_protein_concentration_scale: bool = False,
    protein_concentration_scale_relative_sd: float = 0.10,
    maxiter: int = 3000,
) -> dict[str, Any]:
    """MAP estimate for a curated global fit (L-BFGS-B on the NumPyro log posterior).

    Returns a dict of latent values suitable for ``init_params`` in :func:`run_nuts`. Mirrors
    :func:`bayesian_binding.optimization.fit_map` but for the curated model's site set.
    """
    from numpyro.infer.util import log_density as _log_density
    from scipy.optimize import minimize

    fixed = dict(fixed or {})
    builder_kwargs = dict(
        model_name=model_name,
        fixed=fixed,
        bounds=bounds,
        delta_g_dimer_prior=delta_g_dimer_prior,
        delta_h_prior_sd=delta_h_prior_sd,
        fit_protein_concentration_scale=fit_protein_concentration_scale,
        protein_concentration_scale_relative_sd=protein_concentration_scale_relative_sd,
    )
    model = build_global_curated_itc_model(curves, **builder_kwargs)
    names = curated_sample_site_names(
        curves, model_name, fixed, fit_protein_concentration_scale=fit_protein_concentration_scale
    )
    all_bounds = _resolve_bounds(model_name, bounds)

    init: dict[str, float] = {}
    opt_bounds: list[tuple[float, float]] = []
    defaults = {
        "delta_g_dimer": -8.0,
        "delta_g_binding": -7.0,
        "delta_delta_g_binding": 0.0,
        "delta_delta_g_monomer": 2.0,
        "delta_g": -8.0,
        "delta_delta_g": 0.0,
        "delta_h_dimer": -3.0,
        "delta_h_first": -8.0,
        "delta_h_second": -5.0,
        "delta_h_monomer": -6.0,
        "delta_h": -6.0,
    }
    for name in names:
        if name.startswith("heat_offset_") or name.startswith("log_sigma_"):
            continue
        if name == "protein_concentration_scale":
            init[name] = 1.0
            opt_bounds.append(
                (
                    float(np.exp(-8.0 * protein_concentration_scale_relative_sd)),
                    float(np.exp(8.0 * protein_concentration_scale_relative_sd)),
                )
            )
            continue
        if name.startswith("delta_h"):
            opt_bounds.append((-150.0, 150.0))
        elif name == "delta_g_dimer" and delta_g_dimer_prior is not None:
            opt_bounds.append((-40.0, 10.0))
        else:
            b = all_bounds[name]
            opt_bounds.append((b.low, b.high))
        init[name] = float(defaults.get(name, 0.0))
    for index, curve in enumerate(curves):
        observed = np.asarray(curve.observed_heats_microcalorie, dtype=float)[np.asarray(curve.include_mask, dtype=bool)]
        offset_bounds = _heat_offset_bounds(observed)
        log_sigma_bounds = _log_sigma_bounds(observed)
        if f"heat_offset_{index}" in names:
            init[f"heat_offset_{index}"] = float(np.clip(np.median(observed), *offset_bounds))
            opt_bounds.append(offset_bounds)
        if f"log_sigma_{index}" in names:
            init[f"log_sigma_{index}"] = float(
                np.clip(np.log(max(np.std(observed), 1.0e-3)), *log_sigma_bounds)
            )
            opt_bounds.append(log_sigma_bounds)

    ordered = [name for name in names]

    def negative_log_density(theta):
        params = {name: theta[i] for i, name in enumerate(ordered)}
        value, _ = _log_density(model, (), {}, params)
        return -value

    value_and_grad = jax.jit(jax.value_and_grad(negative_log_density))

    def objective(theta):
        value, grad = value_and_grad(jnp.asarray(theta))
        value = float(value)
        if not np.isfinite(value):
            return 1.0e100, np.zeros_like(theta)
        return value, np.where(np.isfinite(np.asarray(grad, dtype=float)), grad, 0.0)

    theta0 = np.asarray([init[name] for name in ordered], dtype=float)
    result = minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=opt_bounds,
        options={"maxiter": maxiter, "ftol": 1.0e-10, "gtol": 1.0e-8},
    )
    return {name: jnp.asarray(value) for name, value in zip(ordered, np.asarray(result.x, dtype=float))}


# ---------------------------------------------------------------------------
# Joint ITC + WAXS models (shared delta_g) and a generic NUTS runner
# ---------------------------------------------------------------------------
def run_nuts(
    model,
    *,
    rng_seed: int = 20260613,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    target_accept_prob: float = 0.9,
    init_params: Mapping[str, Any] | None = None,
    progress_bar: bool = True,
) -> MCMC:
    """Run NUTS for an arbitrary no-argument NumPyro ``model`` closure (e.g. the joint model)."""
    if init_params is None:
        kernel = NUTS(model, target_accept_prob=target_accept_prob)
    else:
        kernel = NUTS(model, init_strategy=init_to_value(values=dict(init_params)), target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(rng_seed))
    return mcmc


def _two_component_itc_observation(experiment, delta_g, index, *, concentration_relative_uncertainty):
    """Add the two-component ITC heat likelihood for one experiment, sharing ``delta_g``.

    Concentrations carry an informative lognormal prior; ``delta_h``, heat offset, and noise scale
    are ITC-only nuisances, indexed by ``index`` so several titrations do not collide.
    """
    heats = np.asarray(experiment.heats_microcalorie, dtype=float)
    offset_low, offset_high = _heat_offset_bounds(heats)
    log_sigma_low, log_sigma_high = _log_sigma_bounds(heats)
    # Lognormal scale corresponding to a relative (coefficient-of-variation) concentration uncertainty.
    sigma_conc = float(np.sqrt(np.log1p(concentration_relative_uncertainty**2)))
    log_cell = numpyro.sample(
        f"log_cell_concentration_{index}", dist.Normal(float(np.log(experiment.cell_concentration_molar)), sigma_conc)
    )
    log_syringe = numpyro.sample(
        f"log_syringe_concentration_{index}", dist.Normal(float(np.log(experiment.syringe_concentration_molar)), sigma_conc)
    )
    heat_offset = _uniform(f"heat_offset_{index}", offset_low, offset_high)
    log_sigma = _uniform(f"log_sigma_itc_{index}", log_sigma_low, log_sigma_high)
    db = DEFAULT_BOUNDS["delta_h"]
    delta_h = _uniform(f"delta_h_{index}", db.low, db.high)
    q_model = MODEL_REGISTRY["two_component"].expected_heats(
        jnp.asarray(experiment.injection_volumes_liter),
        cell_volume_liter=float(experiment.cell_volume_liter),
        cell_concentration_molar=jnp.exp(log_cell),
        syringe_concentration_molar=jnp.exp(log_syringe),
        temperature_k=float(experiment.temperature_k),
        delta_g=delta_g,
        delta_h=delta_h,
        heat_offset=heat_offset,
    )
    numpyro.deterministic(f"q_model_{index}", q_model)
    numpyro.sample(f"q_obs_{index}", dist.Normal(q_model, jnp.exp(log_sigma)), obs=jnp.asarray(experiment.heats_microcalorie))


def _as_list(value, single_type):
    return [value] if isinstance(value, single_type) else list(value)


def build_waxs_numpyro_model(
    waxs_reduced,
    *,
    delta_g_bounds: PriorBounds | None = None,
    binding_priors: Mapping[str, PriorBounds] | None = None,
    c_model=binding_1to1_concentrations,
    waxs_log_sigma_bounds: tuple[float, float] = (-6.0, 6.0),
):
    """A WAXS-only model: sample the binding parameter(s) and add the MCR observation(s).

    By default it samples a single ``delta_g`` (the paper's single-parameter analysis, ``c_model``
    being :func:`scattering.binding_1to1_concentrations`). For a **multi-parameter** concentration
    model, pass ``binding_priors`` (a mapping of parameter name -> :class:`PriorBounds`) with a matching
    ``c_model`` from :func:`scattering.binding_model_concentrations`; the sampled parameters are passed
    to the C-model as a dict so NUTS infers them jointly. For example, a cooperative two-site fit::

        from bayesian_binding import (binding_model_concentrations, build_waxs_numpyro_model,
                                      run_nuts, PriorBounds)
        model = build_waxs_numpyro_model(
            reduced, c_model=binding_model_concentrations("cooperative"),
            binding_priors={"delta_g": PriorBounds(-12, -2), "delta_delta_g": PriorBounds(-4, 4)})
        mcmc = run_nuts(model, num_warmup=1000, num_samples=1000)

    ``c_model`` may also be a sequence matching ``waxs_reduced``. This supports shared-parameter
    fits across different concentration models, such as AdK+AMP and AdK+ATP two-component branches
    plus an AdK+ADP steady-state branch using the same ``delta_g_amp``/``delta_g_atp`` parameters.
    """
    reduced_list = _as_list(waxs_reduced, WAXSReduced)
    if callable(c_model):
        c_model_list = [c_model] * len(reduced_list)
    else:
        c_model_list = list(c_model)
        if len(c_model_list) != len(reduced_list):
            raise ValueError("When c_model is a sequence, it must match the number of WAXS datasets.")

    def numpyro_model():
        if binding_priors is None:
            bounds = delta_g_bounds or DEFAULT_BOUNDS["delta_g"]
            binding = _uniform("delta_g", bounds.low, bounds.high)  # single scalar parameter
        else:
            binding = {name: _uniform(name, b.low, b.high) for name, b in binding_priors.items()}
        for j, (reduced, c_model_j) in enumerate(zip(reduced_list, c_model_list)):
            add_waxs_observation(
                binding,
                reduced,
                c_model=c_model_j,
                site_prefix=f"waxs_{j}",
                log_sigma_bounds=waxs_log_sigma_bounds,
            )

    return numpyro_model


def build_joint_numpyro_model(
    itc_experiments,
    waxs_reduced,
    *,
    delta_g_bounds: PriorBounds | None = None,
    c_model=binding_1to1_concentrations,
    concentration_relative_uncertainty: float = 0.10,
    waxs_log_sigma_bounds: tuple[float, float] = (-6.0, 6.0),
):
    """Joint ITC + WAXS model with a single shared ``delta_g``.

    Each ITC experiment contributes a two-component heat likelihood (its own ``delta_h`` and
    nuisances); each prepared WAXS dataset contributes the MCR profile likelihood (its reference
    patterns profiled out, its own noise scale). The binding free energy ``delta_g`` is shared
    across all of them -- see ``docs/WAXS.md``.
    """
    itc_list = _as_list(itc_experiments, ITCExperiment)
    reduced_list = _as_list(waxs_reduced, WAXSReduced)
    bounds = delta_g_bounds or DEFAULT_BOUNDS["delta_g"]

    def numpyro_model():
        delta_g = _uniform("delta_g", bounds.low, bounds.high)
        for index, experiment in enumerate(itc_list):
            _two_component_itc_observation(
                experiment, delta_g, index, concentration_relative_uncertainty=concentration_relative_uncertainty
            )
        for j, reduced in enumerate(reduced_list):
            add_waxs_observation(delta_g, reduced, c_model=c_model, site_prefix=f"waxs_{j}", log_sigma_bounds=waxs_log_sigma_bounds)

    return numpyro_model


def to_inference_data(mcmc: MCMC) -> az.InferenceData:
    """Convert NumPyro samples to ArviZ InferenceData."""
    return az.from_numpyro(mcmc)


def save_summary(mcmc: MCMC, path: str | Path) -> None:
    """Write an ArviZ posterior summary table to CSV."""
    summary = az.summary(to_inference_data(mcmc))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path)
