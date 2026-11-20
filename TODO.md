Data
====

* GALAH: Remove [O/Fe] from [alpha/Fe] calculation? It correlates terribly with the other alpha elements.

Error floor in the merged catalog
=================================

Revisit `params_err = sqrt(C_ii + 0.03^2)` (`astrotune/apply.py`, top-of-module
column list; `astrotune/merge.py`, `EPS_ERR` and `err = np.sqrt(np.diagonal(cov) +
eps_err**2)`). The 0.03 is hand-set from the reference notebook and now does two
different jobs, both of which predate the learned calibration:

* it is added to every entry's covariance *before* inversion
  (`_Entries.__init__`: `C_i + e^2 I`), so it sets the weights and ends up inside
  `params_cov`, not only in the reported error;
* it is added a second time to the diagonal for `params_err`, so `params_err` is
  deliberately not `sqrt(diag(params_cov))`.

Things to decide:

* Is an entry-level regularizer still needed once the delta model gives a
  per-catalog, per-parameter error, including a learned additive term that can
  absorb exactly this kind of floor? If it is only there to keep the inverses
  conditioned, it should be small and described as conditioning, not as 0.03.
* Is a uniform 0.03 defensible when it is not uniform in effect? It exceeds the
  real [alpha/Fe] scatter (~0.015 dex from the pair chi2) and is far below a
  typical logg error (0.05-0.1), so `params_err` is floor-dominated on
  [alpha/Fe] and floor-irrelevant on logg.
* Choose between: drop it; take per-parameter floors from the learned additive
  term; or keep it as the merge regularizer only, and report `sqrt(C_ii)`.

Cheap experiment: rerun `astrotune-apply --reuse-transform --eps-err <value>` and
compare `params_cov`, `params_err` and the rejection fractions against the 0.03
baseline.

Correlated entries in the merge
===============================

`astrotune.merge.combine` weights every entry by `(C_i + eps^2 I)^-1`, i.e. it
assumes the entries of a star are independent. That is false for a catalog
group: `apogee_dr17` and `sdssv_aspcap` run the same pipeline on the same
spectra, and the delta model now fits a shared systematic `G_g` for them
(`registry.GROUPS`, `evaluation/shared.txt`: sigma = 0.016/0.010/0.005/0.014 in
[5 ln Teff, [Fe/H], logg, [alpha/Fe]]). Both are large catalogs, so the merge
currently treats them as two independent measurements and over-weights their
(strongly correlated) information; the latent `y`-solve in `train_latent.py` has
the same flaw.

The mathematics is the same hierarchical model the delta likelihood already
implements: entries of a group share a latent offset, so the group contributes
`V_g = A_g^-1 + G_g` (Woodbury) instead of a sum of independent precisions, and
the star's value follows from the group-level observations. To do:

* in `merge`: replace the per-entry precision sum by a two-level solve, using the
  group index (available from `registry.group_index`) and the fitted `G_g`
  (currently in `delta/evaluation/shared.txt` / `const_delta.npz`, not yet read by
  `apply`);
* in the latent trainer's `y`-solve: same structure, or (cheaper, approximate)
  inflate the σ of a group's entries so the pair is down-weighted;
* keep it optional and off by default until the pair chi2 of grouped pairs is
  checked to improve (it is currently 0.36 overall and 0.04 in [alpha/Fe], so
  there is a long way to go).

Student-t tails in the bijector training
========================================

The delta model fits a per-catalog `nu` (`learn_nu`, `evaluation/nu.txt`, values
7-10), which absorbs the heavy tails of the star-to-star scatter. The bijector's
latent `y`-solve in `train_latent.py` is still Gaussian: it weights every
observation by `1/sigma_x^2` and has no tail parameter, so one wild entry can pull
a star's `y`. Consider

* a Student-t version of the `y`-solve (`w_i = 1 / (sigma_i^2 (1 + r_i^2/(nu
  sigma_i^2)))`), reusing the fitted per-catalog `nu` as a fixed input rather than
  fitting it inside the bijector (that would need the model to be re-entered, and
  is likely to be unstable);
* or an outlier-robust reweighting step inside the existing iterated solve.

Not implemented; do not start before the error-calibration iteration settles, as
the two would then be fitted against conflicting noise models.

Iterate the bijector and the delta model
========================================

The intended workflow, now that the plumbing exists (`sigma_x_raw` in the prepared
tables, `J` and the catalog-space covariance in the delta star table, and
`astrotune.delta.feed.calibrated_sigma`):

a. train a bijector on `sigma_x` (the hand-set floors and rescalings);
b. train a delta model on `sigma_x_raw` with that bijector's `J`;
c. rebuild the observations with `astrotune-observations --sigma-source delta
   --delta-dir models/X` and train a new bijector (it then sees the calibrated
   errors, `sqrt(exp(ln_scale) sig_raw^2 + exp(ln_add))`);
d. retrain the delta model on the new bijector's `J` (the star table must be
   rebuilt: `--rebuild`, or delete `delta/star_table.npz`, because `J` changes);
e. repeat until the per-catalog `s` and `a` stop moving (say 2 %), watching the
   per-catalog scale ratios (`evaluation/per_catalog.txt`) converge to 1.

Still to build: a driver that runs a.-e. with run names `..._it2`, `_it3`, and a
per-iteration summary. Things to watch: a global scale degeneracy (fix the median
`ln s` over catalogs, or a reference catalog, if `s` drifts as a whole), and the
bijector's validation chi2, which should stay flat or improve slightly -- if it
degrades, the calibrated sigma is overfitting.

Note `desi_ddp` is the one prepared table without a `sigma_x_raw` column (its
preparation needs a Gaia TAP login); `stars.raw_sigma` reconstructs it exactly
from `sigma_x` with `prep.error_scale.quoted_sigma`, verified to 1e-6 on the other
19 catalogs. Re-preparing with credentials is preferable.
