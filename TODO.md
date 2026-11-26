Data
====

* SAGA — **investigated 2026-11-26, not worth it as a model catalog** (branch
  `saga` deleted; nothing was built).
  - Low-metallicity stars. URL:
    http://sagadatabase.jp/Data/txt_table_recommended.tsv (10 MB, HTTP only,
    latin-1, tab separated, 636 columns, 13,414 stars). References:
    https://ui.adsabs.harvard.edu/abs/2008PASJ...60.1159S/abstract,
    https://ui.adsabs.harvard.edu/abs/2022ApJ...931..147L/abstract,
    https://ui.adsabs.harvard.edu/abs/2022ApJ...931..146A/abstract
  - The difficulties:
    * The host has an A record (153.126.128.103, a sakura VPS) but *our* resolver
      does not see it: `curl` needs `--doh-url https://dns.google/dns-query` (or
      the same IP in /etc/hosts). Everything after that is ordinary HTTP.
    * Coverage is thin: RA/Dec in only 47.8% of the rows (6,403), Teff/log g/
      [M/H] in ~50%, [Fe/H] in 33%, [Mg/H] in 25%. There are **no uncertainties
      at all** for Teff, log g or [Fe/H]; the `d(X)` columns only cover the
      individual abundances. Coordinates + Teff + log g + [Fe/H] leaves 3,693
      stars, and only 2,467 of those have an alpha element (1,854 with an error),
      so a four-parameter catalog would be ~2,500 stars with invented error bars.
    * **It is a meta-catalog, and its "recommended" values are copied verbatim
      from the source papers.** Li et al. (2022, ApJ 931, 147) *is* our
      `vmp_stars_li2022`: 280 of its 298 prepared stars (94%) are in SAGA within
      2" (median 0.25"), SAGA's Teff and log g are *identical* to VMP's (median
      difference 0.0 K / 0.00 dex, scatter 0.0), and its [M/H] equals VMP's
      [Fe/H] (median 0.000, 89% within 0.05 dex). The individual abundances of
      that paper are *not* in the recommended table for those stars ([Mg/H] 6.8%,
      [Si/H] 2.1%), so SAGA does not even hand us VMP's alpha.
  - What it would add: the other 6,123 coordinate-bearing rows, a 2008-onward
    literature compilation (2MASS names) spanning [Fe/H] -7.1..+0.55, median
    -0.90 — the metal-poor end the model is thinnest in, but again as copies of
    other surveys' values, with no errors, and overlapping whatever we already
    have.
  - If it is ever revisited the only defensible form is a **deduplicated
    residual** sample: crossmatch to Gaia (GAVO), drop every star whose id is
    already in a prepared table, and keep what is left. That is cheap to test
    *after* the crossmatch, but the expected yield (~2,500 stars, mostly with no
    alpha) does not justify the work now.

* MaStar:
  - Description: https://www.sdss4.org/dr17/mastar/
  - Data model: https://data.sdss.org/datamodel/files/MANGA_SPECTRO_MASTAR/DRPVER/MPROCVER/vac/parameters/VER/mastar-goodstars-DRPVER-MPROCVER-params-VER.html
  - Download URL: https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/vac/parameters/v2/mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits
  - Info about stellar parameters: https://www.sdss4.org/dr17/mastar/mastar-stellar-parameters/
  - References: https://scixplorer.org/abs/2022MNRAS.509.4308H/abstract?fromADS=true&ads_compat=1,
                https://scixplorer.org/abs/2022AJ....163...56I/abstract?fromADS=true&ads_compat=1


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
