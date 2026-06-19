"""Regression test: WAXS binding analysis of lysozyme:(NAG)2 vs Minh & Makowski 2013 Table 1.

Reproduces the single-parameter (delta_g) WAXS result for lysozyme binding chitobiose, (NAG)2. The
estimator (:func:`waxs_delta_g_posterior`) is a faithful port of the paper's ``analyze1site.m``:
condition-mean SVD, the four-species 1:1 model fit to the truncated SVD coordinates by weighted
least squares (inverse shrinkage variance), with the posterior median as the point estimate and the
16th/84th percentiles as the 68% CI. The committed data subset (``data/lysozyme_waxs_nag2/``) is the
paper's own preprocessed analysis input (``NAG2.mat``).

The paper's Table 1 "With SVD" entry is delta_g = -5.73 kcal/mol, 68% CI [-5.80, -5.67]; the value
saved by the original MATLAB run is -5.733, [-5.802, -5.669]. This test reproduces it to within 0.05
kcal/mol (in practice ~0.004; the tiny residual is the gas constant used here vs in the paper).
"""

from pathlib import Path

import pytest

from bayesian_binding.scattering import WAXSDataset, waxs_delta_g_posterior

REPO = Path(__file__).resolve().parents[1]

# Minh & Makowski 2013, Table 1, lysozyme-(NAG)2, "With SVD" (saved values from analyze1site.m).
REFERENCE_DELTA_G = -5.733
REFERENCE_CI68 = (-5.802, -5.669)
TOLERANCE = 0.05  # the paper's CI precision; achieved deviation is ~0.004 kcal/mol


@pytest.mark.regression
def test_lysozyme_nag2_waxs_delta_g_matches_table1():
    dataset = WAXSDataset.load(REPO / "data" / "lysozyme_waxs_nag2" / "waxs_nag2.npz")
    assert dataset.n_curves == 192 and dataset.n_angles == 364  # 20 conditions, small-angle window

    result = waxs_delta_g_posterior(dataset, n_coordinates=4)  # K=4 species, the paper's truncation
    lower, upper = result.ci68

    # Point estimate (posterior median) reproduces Table 1 to within the paper's CI precision.
    assert result.delta_g == pytest.approx(REFERENCE_DELTA_G, abs=TOLERANCE)
    # The 68% credible interval matches the published bounds ...
    assert lower == pytest.approx(REFERENCE_CI68[0], abs=TOLERANCE)
    assert upper == pytest.approx(REFERENCE_CI68[1], abs=TOLERANCE)
    # ... and brackets the point estimate.
    assert lower < result.delta_g < upper
