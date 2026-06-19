# Computing Bayes factors

A Bayes factor is the ratio of two models' marginal likelihoods, `BF = Z₁ / Z₀`. BayesianBinding
provides two estimators from posterior samples. They agree where both are trustworthy; they differ
when one model fits the data far better than the other.

## Recommended: per-model bridge sampling

Estimate each model's log marginal likelihood `log Z` on its own by bridging its posterior to a
multivariate-normal reference fit to that same posterior, then take the ratio
(`bayesian_binding.evidence`):

```python
from bayesian_binding import bayes_factor_bridge, build_global_numpyro_model, run_mcmc_global

samples_full  = run_mcmc_global(experiments, model_name="cooperative").get_samples()
samples_null  = run_mcmc_global(experiments, model_name="cooperative_equivalent_sites").get_samples()
out = bayes_factor_bridge(
    samples_full, build_global_numpyro_model(experiments, model_name="cooperative"),
    samples_null, build_global_numpyro_model(experiments, model_name="cooperative_equivalent_sites"),
)
print(out["log10_bf"], "±", out["se"], "reliable:", out["reliable"])
```

`bayes_factor_bridge` wraps two `bridge_sampling` calls and `log_bayes_factor`. **Why it is reliable:**
the only overlap required is between a posterior and its own Gaussian fit, which is always good, so the
estimate does not degrade as the Bayes factor grows, and the model ladder is chain-consistent by
construction (`log BF(A/C) = log BF(A/B) + log BF(B/C)` exactly). This is the standard recipe
(Meng & Wong 1996; Gronau et al. 2017) and is what `examples/nguyen_2022_model_selection/` and
`examples/model_comparison.py` use.

## Alternative: nested warp-BAR

For a nested pair where the complex model simply **adds one parameter** that the null fixes
(`theta = (theta_shared, gamma)`; in this package, racemic → enantiomeric mixture, which adds the
composition `gamma = rho` via `DirectNesting`), the Bayes factor can be estimated by bridging directly
between the two model posteriors with the Bennett acceptance ratio
(`bayesian_binding.bayes_factor.nested_bayes_factor`). This is **biased when the two posteriors overlap
poorly** — which happens whenever one model fits much better, because the *shared* parameters (and the
Gaussian proposal for the added `gamma`) shift between the two best fits; the bias can go in either
direction. By default `nested_bayes_factor` applies an **affine warp** (`AffineWarp`, "warp-BAR") that
realigns the shared parameters before the cross-density evaluations, which removes the marginal
location/scale part of that shift; pass `warp=None` for the un-warped diagnostic. The affine warp acts
only on the *marginals* of the shared parameters; for overlap lost specifically to a **`theta_1`–`gamma`
correlation**, pass `conditional_proposal=True`, which draws the added dimensions from the Gaussian
conditional `f(gamma | theta_1)` instead of the marginal `f(gamma)`. That restores the correlation and
can lift a flagged-unreliable factor back to reliable (e.g. on a controlled pair with posterior
correlation ≈ 0.97 it raised the Bennett overlap from ≈ 0.45 to ≈ 0.92); it reduces to the marginal
proposal when there is no correlation. What *no* proposal can fix is a genuine separation in `gamma`
itself (e.g. `rho` pinned far from 0.5) — there the per-model bridge remains the answer, and it stays the
recommended default. (The nested estimator itself is correct: on a controlled conjugate pair with
overlapping posteriors it reproduces the analytic Bayes factor to ~1e-4 — see
`tests/test_bayes_factor_consistency.py` — so a real-data disagreement is low-overlap bias, not a bug.)

## Standard errors and the overlap flag

Both estimators report a **closed-form asymptotic standard error** (no bootstrap): the optimal-bridge /
BAR asymptotic variance of Bennett (1976) and Shirts et al. (2003) for the nested estimator, and the
Meng–Wong / Frühwirth-Schnatter (2004) relative MSE for per-model bridge (these coincide, since the
optimal bridge *is* BAR). The shared helper `bridge_diagnostics` also returns **Bennett's overlap
integral** `O = ∫ p₀p₁ / (½p₀ + ½p₁) dx ∈ [0, 1]` (1 = identical, → 0 = disjoint): the variance scales
like `1/O`, and a low overlap means the work arrays are one-sided and the estimate is biased. Each
result therefore carries `overlap` and a `reliable` flag (overlap ≥ a threshold). The two estimators use
**different thresholds**: per-model bridge defaults to `0.10` (a posterior and its own Gaussian fit
overlap well unless the fit is poor), while the **nested estimator defaults to `0.5`**
(`DEFAULT_NESTED_OVERLAP_THRESHOLD`) because it is bias-prone at moderate overlap even after warping (in
testing, a Bennett overlap around 0.2 already produced a ~2 log10 bias) while the per-model bridge stayed
reliable. When an estimate is unreliable it warns and recommends a remedy — switch to per-model bridge
for a flagged nested factor, or bridge in the unconstrained space / a heavier-tailed reference for a
flagged per-model fit. **Note:** a Monte-Carlo/bootstrap error does *not* warn you about the nested
estimator's bias (it measures variance, not accuracy) — the overlap flag does.

## Gold-standard check

For the largest factors, confirm with thermodynamic integration along the constrained parameter
(Savage–Dickey on the marginal-posterior force); it assumes neither Gaussianity nor overlap. A driver
lives under `analysis/` (not packaged). A fuller methodological treatment, with worked thymidylate-
synthase and TrpR examples, is in `reports/Bayesian Binding Methods/bayes_factor_estimators_note.md`.

## References

- Bennett, C. H. (1976). Efficient estimation of free energy differences from Monte Carlo data.
  *J. Comput. Phys.* 22, 245–268.
- Shirts, M. R.; Bair, E.; Hooker, G.; Pande, V. S. (2003). Equilibrium free energies from
  nonequilibrium measurements using maximum-likelihood methods. *Phys. Rev. Lett.* 91, 140601.
- Meng, X.-L.; Wong, W. H. (1996). Simulating ratios of normalizing constants via a simple identity.
  *Statistica Sinica* 6, 831–860.
- Frühwirth-Schnatter, S. (2004). Estimating marginal likelihoods … via thermodynamic integration.
  *Econometrics Journal* 7, 143–167.
- Gronau, Q. F. et al. (2017). A tutorial on bridge sampling. *J. Math. Psychol.* 81, 80–97.
- Nguyen, T. H.; La, V. N. T.; Burke, K.; Minh, D. D. L. (2022). Bayesian regression and model
  selection for isothermal titration calorimetry with enantiomeric mixtures. *PLOS ONE* 17(9),
  e0273656.
