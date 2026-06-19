"""Tests for the curated sequential-titration ITC path (TrpR L39E phosphate-style data)."""

from pathlib import Path

import numpy as np
from numpyro.infer.util import log_density

from bayesian_binding.constants import MICROCALORIES_PER_KCAL
from bayesian_binding.data import CuratedITCCurve, load_curated_itc_curve
from bayesian_binding.inference import (
    build_global_curated_itc_model,
    curated_sample_site_names,
    fit_map_curated,
)
from bayesian_binding.models import MODEL_REGISTRY

_HEADER = (
    "curve_id,curve_injection,source_file,source_injection,include_in_fit,exclusion_reason,"
    "cell_volume_ml,injection_volume_ul,syringe_ligand_um,dimer_um_before,ligand_um_before,"
    "dimer_um_after,ligand_um_after,observed_heat,baseline_power,raw_area,temperature_mean_c"
)


def _synthetic_curve(name, *, dimer0_um, syringe_um, n, true, model_name, seed):
    """Build a CuratedITCCurve by propagating the dilution recursion and predicting heats from the model."""
    model = MODEL_REGISTRY[model_name]
    factor = 2.0 if model_name.startswith("dimerization") else 1.0
    v_l = 1.0e-5  # 10 uL
    V_l = 1.4229e-3
    f = 1.0 - v_l / V_l
    dimer_before, ligand_before, dimer_after, ligand_after = [], [], [], []
    d, lig = dimer0_um * 1e-6, 0.0
    for _ in range(n):
        dimer_before.append(d)
        ligand_before.append(lig)
        d_new = f * d
        lig_new = f * lig + (1.0 - f) * syringe_um * 1e-6
        dimer_after.append(d_new)
        ligand_after.append(lig_new)
        d, lig = d_new, lig_new
    dimer_before = np.asarray(dimer_before)
    ligand_before = np.asarray(ligand_before)
    dimer_after = np.asarray(dimer_after)
    ligand_after = np.asarray(ligand_after)
    h_before = np.asarray(model.enthalpy_density(dimer_before * factor, ligand_before, 298.15, **true))
    h_after = np.asarray(model.enthalpy_density(dimer_after * factor, ligand_after, 298.15, **true))
    q = V_l * (h_after - f * h_before) * MICROCALORIES_PER_KCAL + 0.5
    rng = np.random.default_rng(seed)
    q = q + rng.normal(0, 0.02 * max(np.std(q), 1e-3), size=q.shape)
    mask = np.ones(n, dtype=bool)
    return CuratedITCCurve(
        name=name,
        injection_volumes_liter=np.full(n, v_l),
        cell_volume_liter=V_l,
        temperature_k=298.15,
        syringe_ligand_molar=np.full(n, syringe_um * 1e-6),
        dimer_before_molar=dimer_before,
        ligand_before_molar=ligand_before,
        dimer_after_molar=dimer_after,
        ligand_after_molar=ligand_after,
        observed_heats_microcalorie=q,
        include_mask=mask,
    )


def test_load_curated_itc_curve_units_and_mask(tmp_path: Path):
    rows = [
        "syn,1,src1.itc,2,1,,1.4229,10.0,20.0,3.0,0.0,2.98,0.14,-13.2,8.8,1735.0,25.01",
        "syn,2,src1.itc,3,0,baseline,1.4229,10.0,20.0,2.98,0.14,2.96,0.28,-9.9,8.8,1734.0,25.02",
    ]
    path = tmp_path / "curve_syn.itc"
    path.write_text("# curated_itc_injection_summary_v1\n" + _HEADER + "\n" + "\n".join(rows) + "\n")
    curve = load_curated_itc_curve(path)
    assert curve.n_injections == 2
    assert curve.n_included == 1
    assert list(curve.include_mask) == [True, False]
    # uM -> molar, mL -> liter, uL -> liter
    np.testing.assert_allclose(curve.dimer_before_molar, [3.0e-6, 2.98e-6])
    np.testing.assert_allclose(curve.syringe_ligand_molar, [20e-6, 20e-6])
    assert abs(curve.cell_volume_liter - 1.4229e-3) < 1e-12
    np.testing.assert_allclose(curve.injection_volumes_liter, [1.0e-5, 1.0e-5])
    assert abs(curve.temperature_k - (np.mean([25.01, 25.02]) + 273.15)) < 1e-9
    # f_i = 1 - v/V
    np.testing.assert_allclose(curve.dilution_factors, 1.0 - 1.0e-5 / 1.4229e-3)


def test_build_global_curated_model_is_callable_and_finite():
    true = dict(
        delta_g_dimer=-8.5, delta_g_binding=-7.0, delta_delta_g_binding=-0.5, delta_delta_g_monomer=2.0,
        delta_h_dimer=-3.0, delta_h_first=-8.0, delta_h_second=-5.0, delta_h_monomer=-6.0,
    )
    curves = [
        _synthetic_curve("c1", dimer0_um=4.0, syringe_um=20.0, n=20, true=true,
                         model_name="dimerization_monomer_cooperative", seed=0),
        _synthetic_curve("c2", dimer0_um=2.5, syringe_um=10.0, n=20, true=true,
                         model_name="dimerization_monomer_cooperative", seed=1),
    ]
    model = build_global_curated_itc_model(curves, model_name="dimerization_monomer_cooperative")
    assert callable(model)
    names = curated_sample_site_names(curves, "dimerization_monomer_cooperative")
    # eight shared thermodynamics + per-curve offset/sigma
    assert "delta_delta_g_monomer" in names
    assert {"heat_offset_0", "heat_offset_1", "log_sigma_0", "log_sigma_1"} <= set(names)
    params = {n: 0.0 for n in names}
    params.update(true)
    params.update({f"heat_offset_{i}": 0.5 for i in range(2)})
    params.update({f"log_sigma_{i}": -1.0 for i in range(2)})
    value, _ = log_density(model, (), {}, params)
    assert np.isfinite(float(value))


def test_fit_map_curated_runs_for_all_curated_models():
    true = dict(
        delta_g_dimer=-8.5, delta_g_binding=-7.0, delta_delta_g_binding=-0.5, delta_delta_g_monomer=2.0,
        delta_h_dimer=-3.0, delta_h_first=-8.0, delta_h_second=-5.0, delta_h_monomer=-6.0,
    )
    curves = [
        _synthetic_curve("c1", dimer0_um=4.0, syringe_um=20.0, n=24, true=true,
                         model_name="dimerization_monomer_cooperative", seed=2),
        _synthetic_curve("c2", dimer0_um=2.5, syringe_um=10.0, n=24, true=true,
                         model_name="dimerization_monomer_cooperative", seed=3),
    ]
    for model_name in ("cooperative", "dimerization_cooperative", "dimerization_monomer_cooperative"):
        params = fit_map_curated(curves, model_name=model_name)
        assert all(np.isfinite(float(v)) for v in params.values())
        assert "heat_offset_0" in params and "log_sigma_1" in params
