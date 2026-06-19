"""Fast tests for the WAXS modality (multivariate curve resolution, Minh & Makowski 2013).

Runs in the default ``pytest`` suite on small synthetic data: the 1:1 concentration model matches
the analytic binding fractions, the MCR profile likelihood peaks at the true ``delta_g`` and recovers
the reference patterns ``R`` by least squares, and a short joint ITC+WAXS NUTS run samples a finite
shared ``delta_g``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_binding.data import ITCExperiment
from bayesian_binding.inference import (
    PriorBounds,
    build_joint_numpyro_model,
    build_waxs_numpyro_model,
    run_nuts,
)
from bayesian_binding.models import MODEL_REGISTRY, dissociation_constant_from_delta_g
from bayesian_binding.scattering import (
    WAXSDataset,
    adk_nucleotide_concentrations,
    binding_1to1_concentrations,
    binding_model_concentrations,
    load_waxs_calibrated,
    prepare_waxs,
    resolved_reference_patterns,
    waxs_delta_g_posterior,
    waxs_profile_loglik,
)

_T = 298.15
_DG_TRUE = -6.0


def test_binding_1to1_concentrations_matches_analytic_fraction():
    """The complex column matches the closed-form 1:1 bound fraction; columns are well formed."""
    protein = np.array([0.0, 400e-6, 400e-6])
    ligand = np.array([100e-6, 100e-6, 0.0])
    c = np.asarray(binding_1to1_concentrations(_DG_TRUE, protein, ligand, _T))
    assert c.shape == (3, 4)
    np.testing.assert_allclose(c[:, 0], 1.0)  # buffer
    # mass balance: free + complex = total, for each row.
    np.testing.assert_allclose(c[:, 1] + c[:, 3], ligand, atol=1e-12)  # free ligand + complex = L_tot
    np.testing.assert_allclose(c[:, 2] + c[:, 3], protein, atol=1e-12)  # free protein + complex = P_tot
    # analytic complex for the binding row (P=400uM, L=100uM) from the 1:1 quadratic root.
    kd = float(dissociation_constant_from_delta_g(_DG_TRUE, _T))
    p, lig = 400e-6, 100e-6
    expected = 0.5 * ((p + lig + kd) - np.sqrt((p + lig + kd) ** 2 - 4 * p * lig))
    assert abs(c[1, 3] - expected) < 1e-12


@pytest.fixture(scope="module")
def synthetic_waxs():
    """Synthetic WAXS series generated from C(delta_g_true) @ R + noise, with replicates."""
    rng = np.random.default_rng(0)
    ligands = np.array([0.0, 50, 100, 200, 400, 800, 1600, 3200]) * 1e-6
    conditions = [(p, lig) for p in (0.0, 400e-6) for lig in ligands]
    n_angles = 80
    two_theta = np.linspace(0.1, 28, n_angles)
    q = np.linspace(0.5, 8, n_angles)
    # Reference patterns scaled so each species contributes comparably to the molar-concentration data.
    reference = np.stack([
        (np.exp(-q / 3) + 0.2) * 1e3,
        (0.3 * np.exp(-q / 2)) * 1e7,
        (np.exp(-((q - 2) ** 2) / 4) + 0.5) * 1e7,
        (np.exp(-((q - 2.5) ** 2) / 3) + 0.55) * 1e7,
    ])
    proteins, ligand_list, rows, condition_ids = [], [], [], []
    for index, (protein, ligand) in enumerate(conditions):
        c = np.asarray(binding_1to1_concentrations(_DG_TRUE, np.array([protein]), np.array([ligand]), _T))[0]
        for _ in range(4):
            rows.append(c @ reference + rng.normal(0.0, 5.0, n_angles))
            proteins.append(protein)
            ligand_list.append(ligand)
            condition_ids.append(index)
    dataset = WAXSDataset(
        "synthetic", two_theta, np.array(rows), np.array(proteins), np.array(ligand_list),
        np.ones(len(rows)), np.array(condition_ids), _T,
    )
    return {"dataset": dataset, "reference": reference}


def test_waxs_profile_likelihood_peaks_at_true_delta_g(synthetic_waxs):
    reduced = prepare_waxs(synthetic_waxs["dataset"], n_svd=6)
    grid = np.linspace(-12.0, -1.0, 111)
    loglik = np.array([float(waxs_profile_loglik(g, reduced, 0.0)) for g in grid])
    assert np.all(np.isfinite(loglik))
    assert abs(grid[int(np.argmax(loglik))] - _DG_TRUE) <= 0.2


def test_waxs_delta_g_posterior_recovers_true_delta_g(synthetic_waxs):
    """The single-parameter grid analysis (Minh & Makowski Table 1 style) recovers the true delta_g.

    Fast synthetic counterpart of ``test_lysozyme_nag2_waxs_regression`` (which reproduces the paper's
    real (NAG)2 result and runs only under ``-m regression``).
    """
    result = waxs_delta_g_posterior(synthetic_waxs["dataset"], n_coordinates=6)
    assert abs(result.delta_g - _DG_TRUE) <= 0.1
    assert abs(result.delta_g_mle - _DG_TRUE) <= 0.1
    lower, upper = result.ci68
    # Clean synthetic data gives a razor-sharp posterior; allow a grid-snapping tolerance.
    assert lower - 0.1 <= _DG_TRUE <= upper + 0.1
    assert upper - lower <= 1.0


def test_resolved_reference_patterns_recovers_R(synthetic_waxs):
    recovered = resolved_reference_patterns(_DG_TRUE, synthetic_waxs["dataset"])
    reference = synthetic_waxs["reference"]
    assert recovered.shape == reference.shape
    assert np.linalg.norm(recovered - reference) / np.linalg.norm(reference) < 0.05


def test_load_waxs_calibrated_drops_outliers_and_normalizes(tmp_path):
    """The loadTable.m port drops flagged exposures and divides by the per-condition ion-chamber fit."""
    import pandas as pd

    angles = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    counts = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    # 3 replicates of two conditions plus one flagged outlier; ion chamber I2 = 3 * (intensity @ counts).
    specs = [("a1", "", 0.0, 1.0), ("a2", "", 0.0, 1.0), ("a3", "", 0.0, 1.0),
             ("b1", "", 1.0, 1.0), ("b2", "", 1.0, 1.0), ("b3", "", 1.0, 1.0),
             ("x1", "outlier", 1.0, 1.0)]
    table_rows, curve_rows, intensity = [], [], {}
    for label, notes, protein, ligand in specs:
        inten = rng.uniform(1.0, 5.0, size=len(angles))
        intensity[label] = inten
        i_sum = float(inten @ counts)
        table_rows.append({"Label": label, "notes": notes, "I0": i_sum, "I2": 3.0 * i_sum,
                           "I3": i_sum, "[P]": protein, "[L]": ligand})
        for j, value in enumerate(inten):
            curve_rows.append({"curve_id": f"RUN:{label}", "point_index": j,
                               "two_theta_deg": angles[j], "intensity": value})
    pd.DataFrame(table_rows).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(tmp_path / "c.csv.gz", index=False)

    dataset = load_waxs_calibrated(
        tmp_path / "c.csv.gz", tmp_path / "t.csv", counts, run="RUN",
        protein_column="[P]", ligand_column="[L]", ion_chamber_column="I2",
        min_angle_index=1, max_angle_index=len(angles), one_over_d_max=None, temperature_k=290.0,
    )
    assert dataset.n_curves == 6 and dataset.n_angles == 5  # the flagged "x1" is dropped
    assert set(dataset.condition_id.tolist()) == {0, 1}
    np.testing.assert_allclose(dataset.protein_molar, np.array([0, 0, 0, 1, 1, 1]) * 1e-3)
    # I2 = 3 * Isum -> calibration slope 3, intercept 0 -> intensities divided by 3 * Isum.
    expected = intensity["a1"] / (3.0 * (intensity["a1"] @ counts))
    np.testing.assert_allclose(dataset.intensities[0], expected, rtol=1e-6)


def test_waxs_dataset_save_load_round_trip(tmp_path, synthetic_waxs):
    dataset = synthetic_waxs["dataset"]
    loaded = WAXSDataset.load(dataset.save(tmp_path / "waxs.npz"))
    np.testing.assert_allclose(loaded.intensities, dataset.intensities)
    np.testing.assert_allclose(loaded.ligand_molar, dataset.ligand_molar)
    assert loaded.n_curves == dataset.n_curves


def test_joint_itc_waxs_shares_delta_g(synthetic_waxs):
    """A short joint NUTS run samples a finite shared delta_g and the per-modality nuisances."""
    rng = np.random.default_rng(1)
    volumes = np.full(16, 8.0e-6)
    common = dict(cell_volume_liter=1.3513e-3, cell_concentration_molar=40e-6, syringe_concentration_molar=2000e-6, temperature_k=_T)
    heats = np.asarray(MODEL_REGISTRY["two_component"].expected_heats(volumes, delta_g=_DG_TRUE, delta_h=-5.0, heat_offset=1.0, **common), dtype=float)
    heats = heats + rng.normal(0.0, 0.3, heats.shape)
    itc = ITCExperiment("itc", volumes, heats, 40e-6, 2000e-6, 1.3513e-3, _T)
    reduced = prepare_waxs(synthetic_waxs["dataset"], n_svd=6)

    model = build_joint_numpyro_model(itc, reduced)
    init = {"delta_g": -6.0, "log_cell_concentration_0": float(np.log(40e-6)),
            "log_syringe_concentration_0": float(np.log(2000e-6)), "heat_offset_0": 1.0,
            "log_sigma_itc_0": float(np.log(0.3)), "delta_h_0": -5.0, "log_sigma_waxs_0": 0.0}
    mcmc = run_nuts(model, init_params=init, num_warmup=80, num_samples=80, num_chains=1, progress_bar=False)
    samples = mcmc.get_samples()
    assert "delta_g" in samples and "delta_h_0" in samples and "log_sigma_waxs_0" in samples
    assert np.all(np.isfinite(np.asarray(samples["delta_g"])))


def test_binding_model_concentrations_adapter_matches_1to1():
    """The two-component adapter reproduces binding_1to1_concentrations exactly."""
    protein = np.array([0.0, 400e-6, 400e-6])
    ligand = np.array([100e-6, 1e-3, 1e-2])
    c_model = binding_model_concentrations("two_component")
    adapted = np.asarray(c_model({"delta_g": _DG_TRUE}, protein, ligand, _T))
    direct = np.asarray(binding_1to1_concentrations(_DG_TRUE, protein, ligand, _T))
    np.testing.assert_allclose(adapted, direct)


def test_adk_nucleotide_concentration_branches_share_parameter_names():
    params = {
        "delta_g_amp": -5.0,
        "delta_g_atp": -6.0,
        "delta_delta_g_amp_atp": -1.0,
        "delta_g_adp": -5.5,
        "delta_delta_g_adp": -0.5,
        "log_adp_dismutation_keq": np.log(0.8),
    }
    protein = np.array([0.0, 400e-6, 400e-6])
    ligand = np.array([100e-6, 1e-3, 4e-3])
    amp = np.asarray(adk_nucleotide_concentrations("AMP")(params, protein, ligand, _T))
    atp = np.asarray(adk_nucleotide_concentrations("ATP")(params, protein, ligand, _T))
    adp = np.asarray(adk_nucleotide_concentrations("ADP")(params, protein, ligand, _T))
    assert amp.shape == atp.shape == adp.shape == (3, 10)
    np.testing.assert_allclose(amp[:, 0], 1.0)
    # Columns are buffer, free_amp, free_adp, free_atp, free_protein, protein_amp, protein_atp,
    # protein_amp_atp, protein_adp, protein_adp2.
    np.testing.assert_allclose(amp[:, [2, 3, 6, 7, 8, 9]], 0.0, atol=1e-14)
    np.testing.assert_allclose(atp[:, [1, 2, 5, 7, 8, 9]], 0.0, atol=1e-14)
    assert np.all(adp[:, 1] >= 0.0) and np.all(adp[:, 2] >= 0.0) and np.all(adp[:, 3] >= 0.0)
    assert np.any(adp[:, 7] > 0.0) and np.any(adp[:, 8] > 0.0) and np.any(adp[:, 9] > 0.0)
    np.testing.assert_allclose(
        adp[:, 1]
        + adp[:, 2]
        + adp[:, 3]
        + adp[:, 5]
        + adp[:, 6]
        + 2.0 * adp[:, 7]
        + adp[:, 8]
        + 2.0 * adp[:, 9],
        ligand,
        rtol=1e-6,
        atol=1e-12,
    )


def test_waxs_model_accepts_per_dataset_adk_nucleotide_c_models():
    import jax
    import numpyro.handlers as handlers

    rng = np.random.default_rng(7)
    c_models = [
        adk_nucleotide_concentrations("AMP"),
        adk_nucleotide_concentrations("ATP"),
        adk_nucleotide_concentrations("ADP"),
    ]
    params = {
        "delta_g_amp": -5.0,
        "delta_g_atp": -6.0,
        "delta_delta_g_amp_atp": -1.0,
        "delta_g_adp": -5.5,
        "delta_delta_g_adp": -0.5,
        "log_adp_dismutation_keq": np.log(0.8),
    }
    datasets = []
    n_angles = 18
    q = np.linspace(0.5, 5.0, n_angles)
    reference = np.vstack([
        (np.exp(-q / 3) + 0.2) * 1e3,
        (0.2 * np.exp(-q / 2)) * 1e7,
        (0.3 * np.exp(-q / 2)) * 1e7,
        (0.4 * np.exp(-q / 2)) * 1e7,
        (np.exp(-((q - 2.0) ** 2) / 4) + 0.5) * 1e7,
        (np.exp(-((q - 2.3) ** 2) / 3) + 0.55) * 1e7,
        (np.exp(-((q - 2.6) ** 2) / 3) + 0.6) * 1e7,
        (np.exp(-((q - 2.9) ** 2) / 3) + 0.65) * 1e7,
        (np.exp(-((q - 3.2) ** 2) / 3) + 0.7) * 1e7,
        (np.exp(-((q - 3.5) ** 2) / 3) + 0.75) * 1e7,
    ])
    for c_model in c_models:
        rows, proteins, ligands, condition_ids = [], [], [], []
        conditions = [(0.0, 0.0), (0.0, 1e-3), (400e-6, 0.0), (400e-6, 1e-3), (400e-6, 4e-3)]
        for index, (protein, ligand) in enumerate(conditions):
            c = np.asarray(c_model(params, np.array([protein]), np.array([ligand]), _T))[0]
            for _ in range(2):
                rows.append(c @ reference + rng.normal(0.0, 2.0, n_angles))
                proteins.append(protein)
                ligands.append(ligand)
                condition_ids.append(index)
        datasets.append(
            WAXSDataset(
                "adk_branch",
                np.linspace(1.0, 20.0, n_angles),
                np.asarray(rows),
                np.asarray(proteins),
                np.asarray(ligands),
                np.ones(len(rows)),
                np.asarray(condition_ids),
                _T,
            )
        )
    reduced = [prepare_waxs(dataset, n_svd=5) for dataset in datasets]
    model = build_waxs_numpyro_model(
        reduced,
        c_model=c_models,
        binding_priors={
            "delta_g_amp": PriorBounds(-10.0, -1.0),
            "delta_g_atp": PriorBounds(-10.0, -1.0),
            "delta_delta_g_amp_atp": PriorBounds(-5.0, 5.0),
            "delta_g_adp": PriorBounds(-10.0, -1.0),
            "delta_delta_g_adp": PriorBounds(-5.0, 5.0),
            "log_adp_dismutation_keq": PriorBounds(-5.0, 5.0),
        },
    )
    trace = handlers.trace(
        handlers.seed(
            handlers.substitute(model, data={**params, "log_sigma_waxs_0": 0.0, "log_sigma_waxs_1": 0.0, "log_sigma_waxs_2": 0.0}),
            jax.random.PRNGKey(0),
        )
    ).get_trace()
    assert {
        "delta_g_amp",
        "delta_g_atp",
        "delta_delta_g_amp_atp",
        "delta_g_adp",
        "delta_delta_g_adp",
        "log_adp_dismutation_keq",
    } <= set(trace)
    assert {"log_sigma_waxs_0", "log_sigma_waxs_1", "log_sigma_waxs_2"} <= set(trace)


def test_waxs_multiparameter_cooperative_nuts_samples_both_parameters():
    """A multi-parameter (cooperative) WAXS model samples both binding parameters via NUTS.

    Exercises the pluggable concentration-model path: data generated from the two-site model's
    species (K=5), fit with binding_model_concentrations('cooperative') + binding_priors.
    """
    rng = np.random.default_rng(3)
    c_model = binding_model_concentrations("cooperative")
    true = {"delta_g": -6.0, "delta_delta_g": -1.0}
    conditions = [(p, lig) for p in (0.0, 4e-4) for lig in np.array([0.0, 100, 400, 1600]) * 1e-6]
    n_angles = 60
    q = np.linspace(0.5, 8.0, n_angles)
    reference = np.stack([  # K=5: buffer, free ligand, apo, singly-bound, doubly-bound
        (np.exp(-q / 3) + 0.2) * 1e3,
        (0.3 * np.exp(-q / 2)) * 1e7,
        (np.exp(-((q - 2.0) ** 2) / 4) + 0.5) * 1e7,
        (np.exp(-((q - 2.3) ** 2) / 3) + 0.55) * 1e7,
        (np.exp(-((q - 2.6) ** 2) / 3) + 0.6) * 1e7,
    ])
    rows, proteins, ligands, condition_ids = [], [], [], []
    for index, (protein, ligand) in enumerate(conditions):
        c = np.asarray(c_model(true, np.array([protein]), np.array([ligand]), _T))[0]  # (5,)
        for _ in range(3):
            rows.append(c @ reference + rng.normal(0.0, 5.0, n_angles))
            proteins.append(protein)
            ligands.append(ligand)
            condition_ids.append(index)
    dataset = WAXSDataset("synthetic_coop", np.linspace(1.0, 28.0, n_angles), np.array(rows),
                          np.array(proteins), np.array(ligands), np.ones(len(rows)), np.array(condition_ids), _T)
    reduced = prepare_waxs(dataset, n_svd=6)

    model = build_waxs_numpyro_model(
        reduced, c_model=c_model,
        binding_priors={"delta_g": PriorBounds(-10.0, -2.0), "delta_delta_g": PriorBounds(-4.0, 4.0)},
    )
    init = {"delta_g": -6.0, "delta_delta_g": -1.0, "log_sigma_waxs_0": 0.0}
    mcmc = run_nuts(model, init_params=init, num_warmup=60, num_samples=60, num_chains=1, progress_bar=False)
    samples = mcmc.get_samples()
    assert {"delta_g", "delta_delta_g", "log_sigma_waxs_0"} <= set(samples)
    assert np.all(np.isfinite(np.asarray(samples["delta_g"])))
    assert np.all(np.isfinite(np.asarray(samples["delta_delta_g"])))
