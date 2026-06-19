"""Wide-angle X-ray solution scattering (WAXS) as a binding modality.

Implements the multivariate curve resolution (MCR) of Minh & Makowski 2013 (Biophys J 104:873):
the scattering data matrix ``D`` (one row per curve, columns = scattering angles) is modeled as
``D = C @ R + eps``, where the binding model supplies the species concentrations ``C(delta_g)`` and
the reference patterns ``R`` are profiled out by (weighted) least squares. Because the resulting
profile log-likelihood is differentiable in ``delta_g``, it plugs into NUTS and ``delta_g`` can be
shared with the ITC heat likelihood in a joint fit (see ``inference.build_joint_numpyro_model``).

This module holds the WAXS data container + loader (Part 1), the binding -> concentrations model
(Part 2), the MCR profile likelihood for the joint NUTS fit (Part 3), and the paper's
single-parameter ``delta_g`` grid analysis with credible intervals (Part 4,
:func:`waxs_delta_g_posterior`, which reproduces the Table 1 binding free energies).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bayesian_binding import _jax_config as _jax_config

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from bayesian_binding.constants import DEFAULT_TEMPERATURE_K, MOLAR_PER_MILLIMOLAR
from bayesian_binding.models import AdKADPSteadyStateModel, MODEL_REGISTRY, dissociation_constant_from_delta_g


# ---------------------------------------------------------------------------
# Part 1: WAXS data container + loader
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WAXSDataset:
    """A set of WAXS curves at different protein/ligand concentrations.

    Replicate exposures are kept as separate rows (not collapsed), so a future per-curve scaling
    factor for the beam-intensity (``i0``) normalization error can be added. ``condition_id`` groups
    the rows that share a ``(protein, ligand)`` condition, which is used to estimate the
    measurement noise from replicate variance.
    """

    name: str
    two_theta_deg: np.ndarray  # (n_angles,)
    intensities: np.ndarray  # (n_curves, n_angles)
    protein_molar: np.ndarray  # (n_curves,)
    ligand_molar: np.ndarray  # (n_curves,)
    i0: np.ndarray  # (n_curves,) beam-normalization scalar per curve
    condition_id: np.ndarray  # (n_curves,) int, groups replicate exposures
    temperature_k: float = DEFAULT_TEMPERATURE_K

    @property
    def n_curves(self) -> int:
        return int(self.intensities.shape[0])

    @property
    def n_angles(self) -> int:
        return int(self.intensities.shape[1])

    def angle_variance(self, *, floor: float = 1.0e-12) -> np.ndarray:
        """Per-angle measurement variance pooled over replicates within each condition.

        Returns a length-``n_angles`` vector: for each angle, the mean over conditions of the
        within-condition sample variance across replicate curves. Conditions with a single replicate
        contribute nothing. Used as the angle weights in the profile likelihood.
        """
        variances = []
        for condition in np.unique(self.condition_id):
            rows = self.intensities[self.condition_id == condition]
            if rows.shape[0] >= 2:
                variances.append(rows.var(axis=0, ddof=1))
        if not variances:
            return np.full(self.n_angles, 1.0)
        return np.maximum(np.mean(variances, axis=0), floor)

    def save(self, path: str | Path) -> Path:
        """Save to a compressed ``.npz`` (used for the committed example subset)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            name=self.name,
            two_theta_deg=self.two_theta_deg,
            intensities=self.intensities,
            protein_molar=self.protein_molar,
            ligand_molar=self.ligand_molar,
            i0=self.i0,
            condition_id=self.condition_id,
            temperature_k=self.temperature_k,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WAXSDataset":
        with np.load(path, allow_pickle=False) as handle:
            return cls(
                name=str(handle["name"]),
                two_theta_deg=handle["two_theta_deg"],
                intensities=handle["intensities"],
                protein_molar=handle["protein_molar"],
                ligand_molar=handle["ligand_molar"],
                i0=handle["i0"],
                condition_id=handle["condition_id"],
                temperature_k=float(handle["temperature_k"]),
            )


def _encode_conditions(protein_molar: np.ndarray, ligand_molar: np.ndarray) -> np.ndarray:
    """Integer id per distinct (protein, ligand) pair, ordered by appearance."""
    pairs = list(zip(protein_molar.tolist(), ligand_molar.tolist()))
    mapping: dict[tuple[float, float], int] = {}
    ids = np.empty(len(pairs), dtype=int)
    for index, pair in enumerate(pairs):
        ids[index] = mapping.setdefault(pair, len(mapping))
    return ids


def load_waxs(
    curves_csv_gz: str | Path,
    analysis_table_csv: str | Path,
    *,
    run: str,
    ligand_column: str,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    concentration_units: str = "mM",
    min_two_theta_deg: float | None = None,
    name: str | None = None,
) -> WAXSDataset:
    """Load a WAXS series from the archive format (long-form curves + an analysis table).

    The analysis table has one row per curve with the curve label, an ``I0`` beam-normalization
    scalar, a protein column (``[AdK]``/``[Adk]``), and ``ligand_column`` (e.g. ``[ATP]``); the curve
    label joins to ``curves_csv_gz`` as ``{run}:{label}``. Concentrations are converted to molar
    (``concentration_units`` of ``"mM"`` or ``"M"``); points below ``min_two_theta_deg`` (the
    beam-stop region) are dropped.
    """
    table = pd.read_csv(analysis_table_csv)
    label_column = table.columns[0]  # "Label" or "Name"
    protein_column = next(c for c in table.columns if str(c).lower() in ("[adk]", "[protein]"))
    table = table[[label_column, "I0", protein_column, ligand_column]].copy()
    table.columns = ["label", "i0", "protein", "ligand"]
    table = table.dropna(subset=["label", "protein", "ligand"])
    table["protein"] = pd.to_numeric(table["protein"], errors="coerce")
    table["ligand"] = pd.to_numeric(table["ligand"], errors="coerce")
    table["i0"] = pd.to_numeric(table["i0"], errors="coerce")
    table = table.dropna(subset=["protein", "ligand"]).reset_index(drop=True)
    table["curve_id"] = run + ":" + table["label"].astype(str)

    wanted = set(table["curve_id"])
    curves = pd.read_csv(curves_csv_gz)
    curves = curves[curves["curve_id"].isin(wanted)]
    if min_two_theta_deg is not None:
        curves = curves[curves["two_theta_deg"] >= min_two_theta_deg]

    wide = curves.pivot(index="curve_id", columns="point_index", values="intensity").sort_index(axis=1)
    two_theta = (
        curves.drop_duplicates("point_index").sort_values("point_index")["two_theta_deg"].to_numpy(dtype=float)
    )
    table = table[table["curve_id"].isin(wide.index)].set_index("curve_id").loc[wide.index].reset_index()

    scale = MOLAR_PER_MILLIMOLAR if concentration_units == "mM" else 1.0
    protein_molar = table["protein"].to_numpy(dtype=float) * scale
    ligand_molar = table["ligand"].to_numpy(dtype=float) * scale
    return WAXSDataset(
        name=name or Path(analysis_table_csv).stem,
        two_theta_deg=two_theta,
        intensities=wide.to_numpy(dtype=float),
        protein_molar=protein_molar,
        ligand_molar=ligand_molar,
        i0=table["i0"].to_numpy(dtype=float),
        condition_id=_encode_conditions(protein_molar, ligand_molar),
        temperature_k=float(temperature_k),
    )


def load_waxs_calibrated(
    curves_csv_gz: str | Path,
    analysis_table_csv: str | Path,
    counts: np.ndarray,
    *,
    run: str,
    protein_column: str,
    ligand_column: str,
    notes_column: str | None = None,
    ion_chamber_column: str = "I2",
    wavelength_angstrom: float = 0.979492,
    min_angle_index: int = 40,
    max_angle_index: int = 885,
    one_over_d_max: float | None = 0.2,
    temperature_k: float = 277.15,
    concentration_units: str = "mM",
    name: str | None = None,
) -> WAXSDataset:
    """Load + photon-count normalize a WAXS series from raw ``.chi`` curves (port of ``loadTable.m``).

    Reproduces the Minh & Makowski 2013 intensity normalization end-to-end, taking a WAXS series
    from raw ``.chi`` curves to an analysis-ready :class:`WAXSDataset`. (The repository ships only
    the preprocessed ``.npz`` datasets used by the regression tests; the raw ``.chi`` curves, the
    analysis table, and the per-run integration weights are not included -- see ``docs/WAXS.md``.)
    The steps are:

    1. drop exposures flagged in the notes column (``notes_column``; defaults to the table's second,
       blank-header column) -- outliers and water;
    2. per condition, fit the summed scattering ``Isum = intensities @ counts`` against the
       ion-chamber reading (``ion_chamber_column``) and divide each curve by the fit, removing
       beam-intensity drift between exposures;
    3. truncate to the beam-stop angular range ``[min_angle_index, max_angle_index]`` (1-based, as in
       ``loadTable.m``) and then to the small-angle window ``1/d <= one_over_d_max`` (``analyze1site.m``;
       ``1/d = 2 sin(theta)/lambda``). Pass ``one_over_d_max=None`` to keep the full beam-stop range.

    ``counts`` is the per-angle integration weighting for the run (one weight per ``.chi`` angular
    bin), supplied by the caller. Concentrations are converted to molar.
    """
    counts = np.asarray(counts, dtype=float).ravel()
    table = pd.read_csv(analysis_table_csv)
    label_column = table.columns[0]
    # Outlier removal: drop exposures with a non-empty note (flagged outliers, water); the notes
    # column defaults to the table's second, blank-header column.
    notes = notes_column if notes_column is not None else table.columns[1]
    keep = table[notes].astype("string").fillna("").str.strip() == ""
    table = table[keep].copy()
    for column in (ion_chamber_column, protein_column, ligand_column):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=[ion_chamber_column, protein_column, ligand_column])
    table["curve_id"] = run + ":" + table[label_column].astype(str)

    # Pivot the long-form curves to a (curves x angles) matrix and align the table rows to it.
    curves = pd.read_csv(curves_csv_gz)
    curves = curves[curves["curve_id"].isin(set(table["curve_id"]))]
    wide = curves.pivot(index="curve_id", columns="point_index", values="intensity").sort_index(axis=1)
    two_theta = (
        curves.drop_duplicates("point_index").sort_values("point_index")["two_theta_deg"].to_numpy(dtype=float)
    )
    table = table[table["curve_id"].isin(wide.index)].set_index("curve_id").loc[wide.index].reset_index()

    raw = wide.loc[table["curve_id"]].to_numpy(dtype=float)
    if counts.size != raw.shape[1]:
        raise ValueError(f"counts length {counts.size} does not match the {raw.shape[1]} chi angles")
    scale = MOLAR_PER_MILLIMOLAR if concentration_units == "mM" else 1.0
    protein_molar = table[protein_column].to_numpy(dtype=float) * scale
    ligand_molar = table[ligand_column].to_numpy(dtype=float) * scale
    ion_chamber = table[ion_chamber_column].to_numpy(dtype=float)
    condition_id = _encode_conditions(protein_molar, ligand_molar)

    # Per-condition ion-chamber calibration: fit Isum -> ion chamber, divide each curve by the fit.
    summed = raw @ counts
    normalized = np.empty_like(raw)
    for condition in np.unique(condition_id):
        sel = condition_id == condition
        design = np.vstack([summed[sel], np.ones(int(sel.sum()))]).T
        slope, intercept = np.linalg.lstsq(design, ion_chamber[sel], rcond=None)[0]
        normalized[sel] = raw[sel] / (summed[sel] * slope + intercept)[:, None]

    # Keep the beam-stop angular range (loadTable.m indices), then the small-angle window.
    low, high = min_angle_index - 1, max_angle_index  # 1-based inclusive -> python slice
    normalized = normalized[:, low:high]
    two_theta = two_theta[low:high]
    if one_over_d_max is not None:
        # Reciprocal spacing 1/d = 2 sin(theta)/lambda; drop the wide angles (solvent/ligand nonlinearity).
        one_over_d = 2.0 * np.sin(np.radians(two_theta) / 2.0) / wavelength_angstrom
        exceed = np.where(one_over_d > one_over_d_max)[0]
        last = int(exceed[0]) + 1 if exceed.size else len(two_theta)  # MATLAB 1:lastInd is inclusive
        normalized = normalized[:, :last]
        two_theta = two_theta[:last]

    return WAXSDataset(
        name=name or Path(analysis_table_csv).stem,
        two_theta_deg=two_theta,
        intensities=normalized,
        protein_molar=protein_molar,
        ligand_molar=ligand_molar,
        i0=ion_chamber,  # beam-intensity proxy used for calibration (intensities are already normalized)
        condition_id=condition_id,
        temperature_k=float(temperature_k),
    )


# ---------------------------------------------------------------------------
# Part 2: binding model -> species concentrations C (pluggable)
# ---------------------------------------------------------------------------
# A C-model maps (binding_params, protein_molar, ligand_molar, temperature_k) to a (n_curves x K)
# matrix of species concentrations -- the columns the MCR resolves into reference patterns. The first
# argument is the binding parameter(s) being inferred: a scalar for a single-parameter model (e.g.
# ``binding_1to1_concentrations`` takes ``delta_g``), or a mapping of parameters for a multi-parameter
# model (e.g. ``binding_model_concentrations(\"cooperative\")`` takes ``{\"delta_g\", \"delta_delta_g\"}``).
# The observation code (Part 3) only passes this argument through, so any K and any parameterization
# works; ``binding_model_concentrations`` adapts any ``models.py`` model into a C-model.
CModel = Callable[[Any, Any, Any, float], Any]


def binding_1to1_concentrations(delta_g, protein_molar, ligand_molar, temperature_k):
    """Concentrations of {buffer, free ligand, free protein, complex} for 1:1 binding (K=4).

    Reuses the closed-form 1:1 equilibrium of ``TwoComponentBindingModel`` (``models.py``). The
    buffer column is a constant 1 (the always-present solvent background).
    """
    protein = jnp.asarray(protein_molar)
    ligand = jnp.asarray(ligand_molar)
    kd = dissociation_constant_from_delta_g(delta_g, temperature_k)
    total = protein + ligand + kd
    # Physically valid root of the 1:1 quadratic (P + L <-> PL): the minus sign keeps PL <= min(P, L).
    # Clamp the discriminant at 0 against round-off when binding is near-stoichiometric (Kd << P, L).
    complex_conc = 0.5 * (total - jnp.sqrt(jnp.maximum(total**2 - 4.0 * protein * ligand, 0.0)))
    free_protein = protein - complex_conc
    free_ligand = ligand - complex_conc
    # Columns: buffer/solvent background (constant), free ligand, free (apo) protein, complex.
    return jnp.stack([jnp.ones_like(protein), free_ligand, free_protein, complex_conc], axis=1)


def binding_model_concentrations(model, *, species_order=None, include_buffer: bool = True) -> CModel:
    """Adapt a ``models.py`` binding model into a WAXS concentration model (a :data:`CModel`).

    Returns a callable ``(params, protein_molar, ligand_molar, temperature_k) -> C``, where ``C`` is the
    ``(n_curves x K)`` matrix whose columns are a constant buffer/background term followed by the model's
    equilibrium species (from its ``equilibrium_species`` method). ``params`` is a mapping of the model's
    binding free-energy parameters -- e.g. ``{"delta_g": ...}`` for ``"two_component"`` or
    ``{"delta_g": ..., "delta_delta_g": ...}`` for ``"cooperative"`` -- so a multi-parameter model can be
    sampled by NUTS (see :func:`inference.build_waxs_numpyro_model`) and resolves the *same* equilibrium
    species the ITC fit uses.

    ``model`` is a model instance or a ``MODEL_REGISTRY`` name. ``species_order`` optionally selects and
    orders the species columns (default: the model's natural order).
    """
    binding_model = MODEL_REGISTRY[model] if isinstance(model, str) else model

    def c_model(params, protein_molar, ligand_molar, temperature_k):
        species = binding_model.equilibrium_species(
            protein_molar, ligand_molar, temperature_k, **dict(params)
        )
        names = list(species) if species_order is None else list(species_order)
        values = [jnp.asarray(species[name]) for name in names]
        columns = ([jnp.ones_like(values[0])] if include_buffer else []) + values
        return jnp.stack(columns, axis=1)

    return c_model


_ADK_NUCLEOTIDE_COLUMNS = (
    "free_amp",
    "free_adp",
    "free_atp",
    "free_protein",
    "protein_amp",
    "protein_atp",
    "protein_amp_atp",
    "protein_adp",
    "protein_adp2",
)


def adk_nucleotide_concentrations(nucleotide: str, *, include_buffer: bool = True) -> CModel:
    """Concentration model for AdK AMP/ATP/ADP WAXS series with shared AMP/ATP affinities.

    The returned callable has the standard WAXS C-model signature
    ``(params, protein_molar, ligand_molar, temperature_k) -> C`` and uses one shared parameter
    dictionary across all three nucleotide datasets:

    - ``delta_g_amp`` for the AdK+AMP two-component branch;
    - ``delta_g_atp`` for the AdK+ATP two-component branch;
    - ``delta_delta_g_amp_atp`` for the mixed ``AdK:AMP:ATP`` coupling;
    - ``delta_g_adp`` and ``delta_delta_g_adp`` for the cooperative AdK+ADP branch;
    - ``log_adp_dismutation_keq`` for ``2 ADP <-> AMP + ATP``.

    All branches return the same species columns, optionally preceded by a constant buffer column:
    ``free_amp, free_adp, free_atp, free_protein, protein_amp, protein_atp, protein_amp_atp,
    protein_adp, protein_adp2``. This lets AMP-only and ATP-only titrations remain simple
    two-component models while the ADP titration adds mixed AMP/ATP and cooperative ADP branches in
    its steady-state concentration calculation.
    """
    key = nucleotide.strip().upper()
    if key not in {"AMP", "ATP", "ADP"}:
        raise ValueError("nucleotide must be one of 'AMP', 'ATP', or 'ADP'")

    def c_model(params, protein_molar, ligand_molar, temperature_k):
        protein = jnp.atleast_1d(jnp.asarray(protein_molar, dtype=float))
        ligand = jnp.atleast_1d(jnp.asarray(ligand_molar, dtype=float))
        zeros = jnp.zeros_like(protein)
        if key == "ADP":
            species = AdKADPSteadyStateModel.equilibrium_species(
                protein,
                ligand,
                temperature_k,
                delta_g_amp=params["delta_g_amp"],
                delta_g_atp=params["delta_g_atp"],
                delta_delta_g_amp_atp=params["delta_delta_g_amp_atp"],
                delta_g_adp=params["delta_g_adp"],
                delta_delta_g_adp=params["delta_delta_g_adp"],
                log_adp_dismutation_keq=params["log_adp_dismutation_keq"],
            )
        else:
            delta_g_name = "delta_g_amp" if key == "AMP" else "delta_g_atp"
            species_1to1 = MODEL_REGISTRY["two_component"].equilibrium_species(
                protein,
                ligand,
                temperature_k,
                delta_g=params[delta_g_name],
            )
            species = {
                "free_amp": species_1to1["free_ligand"] if key == "AMP" else zeros,
                "free_adp": zeros,
                "free_atp": species_1to1["free_ligand"] if key == "ATP" else zeros,
                "free_protein": species_1to1["free_protein"],
                "protein_amp": species_1to1["complex"] if key == "AMP" else zeros,
                "protein_atp": species_1to1["complex"] if key == "ATP" else zeros,
                "protein_amp_atp": zeros,
                "protein_adp": zeros,
                "protein_adp2": zeros,
            }
        columns = [jnp.asarray(species[name]) for name in _ADK_NUCLEOTIDE_COLUMNS]
        if include_buffer:
            columns = [jnp.ones_like(protein)] + columns
        return jnp.stack(columns, axis=1)

    return c_model


# ---------------------------------------------------------------------------
# Part 3: MCR profile likelihood
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WAXSReduced:
    """Pre-processed WAXS data for the profile likelihood.

    The per-angle measurement noise whitens the data and an SVD truncation denoises it (Minh &
    Makowski 2013, Eqs 15-18). Everything here is fixed (it does not depend on ``delta_g``); the
    likelihood only differentiates through ``C(delta_g)`` and the profiled reference patterns.
    """

    data_reduced: np.ndarray  # (n_curves x n_svd) whitened, SVD-projected data
    protein_molar: np.ndarray  # (n_curves,)
    ligand_molar: np.ndarray  # (n_curves,)
    temperature_k: float
    n_svd: int
    name: str


def prepare_waxs(dataset: WAXSDataset, *, n_svd: int = 6) -> WAXSReduced:
    """Whiten by the per-angle noise, SVD, and project onto the top ``n_svd`` components."""
    whitening = np.sqrt(dataset.angle_variance())  # (n_angles,)
    data_white = dataset.intensities / whitening
    # Right singular vectors of the whitened data span angle space; keep the most significant.
    _u, _s, vt = np.linalg.svd(data_white, full_matrices=False)
    n_svd = int(min(n_svd, vt.shape[0]))
    basis = vt[:n_svd].T  # (n_angles x n_svd)
    return WAXSReduced(
        data_reduced=data_white @ basis,
        protein_molar=dataset.protein_molar,
        ligand_molar=dataset.ligand_molar,
        temperature_k=dataset.temperature_k,
        n_svd=n_svd,
        name=dataset.name,
    )


def waxs_profile_loglik(
    binding_params,
    reduced: WAXSReduced,
    log_sigma,
    *,
    c_model: CModel = binding_1to1_concentrations,
    scale=None,
):
    """Gaussian log-likelihood of ``D = C R + eps`` with ``R`` profiled by least squares.

    Works in the whitened, SVD-reduced space of ``reduced``; ``R`` (here its reduced-space
    coefficients) is solved by least squares for the given binding parameters, leaving the residual.
    ``binding_params`` is the C-model's first argument -- a scalar ``delta_g`` for the 1:1 model or a
    mapping for a multi-parameter model (see :func:`binding_model_concentrations`). ``scale`` is an
    optional per-curve multiplier for the future beam-intensity correction (``None`` -> 1).
    """
    data = jnp.asarray(reduced.data_reduced)
    if scale is not None:
        data = data / jnp.asarray(scale)[:, None]
    concentrations = c_model(binding_params, reduced.protein_molar, reduced.ligand_molar, reduced.temperature_k)
    residual = _profile_residual(concentrations, data)
    sigma = jnp.exp(log_sigma)
    n_elements = data.shape[0] * data.shape[1]
    return -0.5 * jnp.sum(residual**2) / sigma**2 - 0.5 * n_elements * jnp.log(2.0 * jnp.pi * sigma**2)


def _profile_residual(concentrations, data):
    """Residual ``data - C @ lstsq(C, data)``, with columns normalized for conditioning.

    Column scaling leaves the residual (the column-space projection) unchanged but keeps the least
    squares well conditioned when the concentration columns (molar, ~1e-4) and the buffer column
    (~1) differ by orders of magnitude.
    """
    column_norm = jnp.sqrt(jnp.sum(concentrations**2, axis=0)) + 1.0e-12
    normalized = concentrations / column_norm
    coefficients, _residuals, _rank, _sv = jnp.linalg.lstsq(normalized, data, rcond=None)
    return data - normalized @ coefficients


def resolved_reference_patterns(
    delta_g, dataset: WAXSDataset, *, c_model: CModel = binding_1to1_concentrations
):
    """Return the least-squares reference patterns ``R`` (K x n_angles) at ``delta_g`` (for plotting).

    Solves ``D = C(delta_g) R`` in the full angle space (not the reduced space used by the
    likelihood), so the rows are the scattering patterns of the K species vs. scattering angle.
    """
    concentrations = np.asarray(
        c_model(delta_g, dataset.protein_molar, dataset.ligand_molar, dataset.temperature_k)
    )
    column_norm = np.sqrt(np.sum(concentrations**2, axis=0)) + 1.0e-12
    coefficients = np.linalg.lstsq(concentrations / column_norm, dataset.intensities, rcond=None)[0]
    return coefficients / column_norm[:, None]


def add_waxs_observation(
    binding_params,
    reduced: WAXSReduced,
    *,
    c_model: CModel = binding_1to1_concentrations,
    site_prefix: str = "waxs",
    log_sigma_bounds: tuple[float, float] = (-6.0, 6.0),
):
    """Register the WAXS MCR log-likelihood for the given binding parameters as a NumPyro factor.

    Samples a WAXS noise scale ``log_sigma_{site_prefix}`` and adds ``numpyro.factor`` with
    :func:`waxs_profile_loglik`. The reference patterns ``R`` are profiled out (not sampled).
    ``binding_params`` is the C-model's first argument -- the sampled ``delta_g`` (or the parameter
    mapping for a multi-parameter model), shared with any other modality in the enclosing model. A
    per-curve scaling factor can be added here later.
    """
    log_sigma = numpyro.sample(f"log_sigma_{site_prefix}", dist.Uniform(*log_sigma_bounds))
    loglik = waxs_profile_loglik(binding_params, reduced, log_sigma, c_model=c_model)
    numpyro.factor(f"{site_prefix}_log_likelihood", loglik)


# ---------------------------------------------------------------------------
# Part 4: single-parameter delta_g grid analysis (Minh & Makowski 2013, Table 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WAXSDeltaGPosterior:
    """Result of the single-parameter WAXS binding analysis: the posterior over ``delta_g``.

    ``delta_g`` is the posterior median (the paper's point estimate, F=0.5) and ``ci68`` is the 68%
    credible interval (16th-84th percentiles). ``delta_g_mle`` is the profile-likelihood maximum;
    ``grid`` / ``log_posterior`` are the refined grid and its (unnormalized) log posterior.
    """

    delta_g: float
    ci68: tuple[float, float]
    delta_g_mle: float
    grid: np.ndarray
    log_posterior: np.ndarray


def _condition_means(values: np.ndarray, cond_index: np.ndarray, n_conditions: int) -> np.ndarray:
    """Mean of ``values`` rows within each condition (``n_conditions x ...``)."""
    return np.stack([values[cond_index == n].mean(axis=0) for n in range(n_conditions)])


def waxs_delta_g_posterior(
    dataset: WAXSDataset,
    *,
    n_coordinates: int = 4,
    c_model: CModel = binding_1to1_concentrations,
    grid_low: float = -15.0,
    grid_high: float = 0.0,
    coarse_points: int = 301,
    fine_step: float = 0.001,
    prior_variance: float = 1000.0,
) -> WAXSDeltaGPosterior:
    """Single-parameter (``delta_g``) WAXS binding analysis on a grid (Minh & Makowski 2013, Table 1).

    A faithful port of the paper's ``analyze1site.m``. The scattering series is reduced to its
    **condition means**, the SVD of that matrix gives an angle-space basis, and each exposure is
    projected onto it and normalized by the singular value to give the SVD *coordinates* (Eqs 15-18).
    The four-species 1:1 model is fit to the top ``n_coordinates`` coordinates by weighted least
    squares; the weights are the inverse **shrinkage variance** (Eq 11): the scatter of each
    coordinate about its condition mean, pooled over the protein / protein-free groups and divided by
    the replicate count. Fitting the coordinates (not the raw angles) is what surfaces the subtle
    binding signal that the dominant free-ligand scattering otherwise swamps.

    The (log) posterior -- a diffuse Gaussian prior on ``delta_g`` plus the weighted-least-squares
    log-likelihood -- is evaluated on a coarse grid over ``[grid_low, grid_high]``, refined to
    ``fine_step`` over the high-density region. The point estimate is the posterior median (F=0.5) and
    the 68% CI runs from the 16th to the 84th percentile (the paper's recipe). With four species and
    more conditions than species the fit is over-determined, so ``n_coordinates = 4`` is identifiable.
    """
    intensities = np.asarray(dataset.intensities, dtype=float)
    protein = np.asarray(dataset.protein_molar, dtype=float)
    ligand = np.asarray(dataset.ligand_molar, dtype=float)
    temperature_k = float(dataset.temperature_k)

    conditions = np.unique(dataset.condition_id)
    n_conditions = len(conditions)
    cond_index = np.searchsorted(conditions, np.asarray(dataset.condition_id))
    replicate_counts = np.array([int(np.sum(cond_index == n)) for n in range(n_conditions)], dtype=float)
    protein_cond = np.array([protein[cond_index == n][0] for n in range(n_conditions)])
    ligand_cond = np.array([ligand[cond_index == n][0] for n in range(n_conditions)])
    has_protein = protein_cond > 0

    # SVD of the condition-mean data: cond_means.T = U S V', so cond_means = V S U'. ``coordinates``
    # are the normalized condition coordinates V; ``exposure_scores`` projects every exposure on U/S.
    cond_means = _condition_means(intensities, cond_index, n_conditions)
    u_full, s_full, vt_full = np.linalg.svd(cond_means.T, full_matrices=False)
    k = int(min(n_coordinates, n_conditions, len(s_full)))
    coordinates = vt_full.T[:, :k]  # (n_conditions x k)
    exposure_scores = (intensities @ u_full[:, :k]) / s_full[:k][None, :]  # (n_curves x k)

    # Shrinkage variance (Eq 11): scatter about the condition mean, pooled over protein / protein-free
    # exposure groups, divided by the replicate count -> variance of the condition-mean coordinate.
    deviations = exposure_scores - _condition_means(exposure_scores, cond_index, n_conditions)[cond_index]
    variance = np.zeros((n_conditions, k))
    for mask in (has_protein, ~has_protein):
        exposures = mask[cond_index]
        variance[mask] = (deviations[exposures] ** 2).sum(axis=0) / max(int(exposures.sum()) - 1, 1)
    weights = replicate_counts[:, None] / variance  # = 1 / (variance / replicate_counts)

    def log_posterior(delta_g: float) -> float:
        # Species concentrations at this delta_g; column-normalized so the absolute column scale
        # doesn't matter (equivalent to the paper's fixed 1e-3 background, but scale-invariant).
        c = np.asarray(c_model(delta_g, protein_cond, ligand_cond, temperature_k), dtype=float)
        c = c / (np.sqrt((c**2).sum(axis=0)) + 1e-12)
        # Profile each SVD coordinate by weighted least squares of C -> coordinate, summing the
        # inverse-variance-weighted residual into chi-square (R = the per-coordinate coefficients).
        chi_square = 0.0
        for j in range(k):
            sqrt_w = np.sqrt(weights[:, j])
            coeff, _res, _rank, _sv = np.linalg.lstsq(c * sqrt_w[:, None], coordinates[:, j] * sqrt_w, rcond=None)
            chi_square += float((((coordinates[:, j] - c @ coeff) ** 2) * weights[:, j]).sum())
        return -(delta_g**2) / (2.0 * prior_variance) - 0.5 * chi_square  # diffuse prior + WLS log-likelihood

    # Evaluate coarsely, keep the region with non-negligible density, then refine to fine_step there.
    coarse = np.linspace(grid_low, grid_high, coarse_points)
    lp_coarse = np.array([log_posterior(float(g)) for g in coarse])
    keep = np.where(lp_coarse - lp_coarse.max() > np.log(1e-5))[0]
    if keep.size < 2:  # widen by a grid point on each side (matches analyze1site.m)
        keep = np.arange(max(keep[0] - 1, 0), min(keep[-1] + 1, len(coarse) - 1) + 1)
    grid = np.arange(float(coarse[keep[0]]), float(coarse[keep[-1]]) + 1e-9, fine_step)
    log_post = np.array([log_posterior(float(g)) for g in grid])
    # Normalized CDF over the fine grid -> point estimate (median) and 68% CI (16th/84th percentiles).
    cdf = np.cumsum(np.exp(log_post - log_post.max()))
    cdf = cdf / cdf[-1]

    def quantile(q: float) -> float:
        return float(grid[int(np.argmin(np.abs(cdf - q)))])

    return WAXSDeltaGPosterior(
        delta_g=quantile(0.5),
        ci68=(quantile(0.16), quantile(0.84)),
        delta_g_mle=float(grid[int(np.argmax(log_post))]),
        grid=grid,
        log_posterior=log_post,
    )
