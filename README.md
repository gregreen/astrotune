# Astrotune

Bringing disparate stellar parameter catalogs into harmony.

Twenty catalogs based on several spectroscopic surveys — APOGEE, GALAH, LAMOST, DESI, RAVE,
SDSS-V, and several other — measure the same stars, and they disagree about their parameters.
Part of that disagreement is astrophysics (resolution, wavelength coverage,
atmosphere grids), part is bookkeeping (units, zero points, a systematic in one
pipeline). Astrotune separates the two by learning, for every catalog, an
invertible map into one common ("harmonized") parameter space, using the stars
that more than one catalog observes as the anchors. **No catalog is the
reference**: what is fitted is a set of maps that make the catalogs mutually
consistent, plus a calibrated error budget for each of them.

Four parameters, in this order everywhere:

| index | parameter | note |
|---|---|---|
| 0 | `5 ln(Teff / 5000 K)` | log temperature: Gaussian in `ln Teff` ⇒ Gaussian here |
| 1 | `[Fe/H]` | |
| 2 | `logg` | |
| 3 | `[alpha/Fe]` | |

The order is frozen — the index is part of every on-disk format
(`config.PARAM_NAMES`).

What the pipeline produces:

* **prepared catalog tables** — `data/training/<key>/<key>.h5`, one schema for
  all 20 catalogs;
* **a bijector** — `models/<run>/bijector`, a SavedModel that maps each catalog
  into the harmonized space and back;
* **a delta model** — `models/<run>/delta/`, which says how wrong each catalog's
  quoted uncertainties are, per star and per parameter;
* **a merged catalog** — `models/<run>/harmonized/merged_catalog.h5`, one row
  per Gaia DR3 source with a covariance;
* diagnostics: catalog agreement, Kiel maps, videos of the transformation, the
  catalog connectivity graph.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # also (re)creates the astrotune-* commands

astrotune-prepare --all                    # catalogs -> prepared tables
astrotune-observations                     # tables -> observations + control points
astrotune-train-latent --run-name my_run   # ~1 day on an A100
astrotune-apply --run-dir models/my_run    # harmonize and merge
```

Long jobs are safe to detach (`setsid nohup ... &`); training checkpoints every
~10 minutes and resumes with `--resume`.

## Layout

```
astrotune/
  config.py            paths, parameter order, error floors, default x_err
  registry.py          the 20 catalogs in fixed order (the category index `c`),
                       candidate catalogs, and GROUPS (catalogs sharing a pipeline)
  prep/                machinery shared by all catalogs
    download.py        cached HTTP downloads
    gaia.py            Gaia DR3 crossmatch: wide search, cached, archive-aware
    transforms.py      parameter transformations, error floors, extract_table
    cuts.py            quality-cut helpers (finite masks, SNR-chi2 relation)
    abundances.py      inverse-variance [alpha/Fe] from individual abundances
    io.py              save/load/validate, parity report against the legacy tables
    pipeline.py        orchestration: download -> load -> cuts -> build -> save
    overlaps.py        shared stars by Teff (`astrotune-overlaps`)
    error_scale.py     is the quoted sigma right? scale + plateau fit
    plots.py           diagnostic figures
  catalogs/            one module per catalog (the contract is at the end)
  cli.py               `astrotune-prepare`
  training_data.py     observations + control points (`astrotune-observations`)
  train_latent.py      latent-y trainer (`astrotune-train-latent`)
  model/               VectorNetwork (velocity field), ODEBijector, SavedModel IO
  delta/               post-hoc error calibration
    stars.py           star table: every entry with its Jacobian, grouped by star
    covariance.py      S = J diag(s^2 sig_raw^2 + a^2) J^T, cropped to measured axes
    model.py           the star likelihood (groups, Student's t)
    train.py           baselines, fit, evaluation (`astrotune-delta-train`)
    kiel.py, uncertainties.py, diagnostics.py, feed.py
  apply.py             harmonize every catalog (`astrotune-apply`)
  merge.py             combine a star's entries (product of Gaussians + chi2 cut)
  evaluate.py          catalog agreement, pair-loss era (`astrotune-evaluate`)
  videos.py            Kiel/pair videos (`astrotune-videos`)
  catalog_graph.py     connectivity figure (`astrotune-catalog-graph`)
data/
  raw/<key>/           downloaded source catalogs (never tracked)
  interim/<key>/       caches: crossmatched tables + gaia_crossmatch.json
  training/<key>/      <key>.h5 + <key>.meta.json (never tracked)
  observations_and_ctrl_points.h5      stage 2 output (never tracked)
models/<run>/
  bijector/            SavedModel: to_harmonized / from_harmonized
  latent_y.npz         the fitted per-star values (state of the run)
  config.json          every hyperparameter + the observations it was trained on
  metrics_history.json, agreement_history.json, progress/, checkpoints/
  videos/, evaluation/, harmonized/
  delta/               star_table.npz, delta_model.*, evaluation/, kiel/, ...
diagnostic_plots/<key>/    preparation figures (gitignored)
notes/                     derivations (error_calibration_model.tex)
scripts/                   one-off analyses, e.g. compare_gaia_taps.py
TODO.md, porting_project.md  open issues and the full porting record
```

## Environment

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

TensorFlow is installed with the `and-cuda` extra: the required CUDA 12.5 and
cuDNN 9.3 libraries come from `nvidia-*` wheels, so no system CUDA toolkit or
module load is needed. Verified on an A100 80GB (driver 615.71.10). The GPU is
shared, so `astrotune.model.io.enable_memory_growth()` is called before any
allocation; without it the process takes ~75 GB and starves whatever else runs.

Two things that look like broken code but are the environment:

* the `astrotune-*` commands are generated by `pip install -e .`; after a new
  entry point is added to `pyproject.toml`, rerun it (`--no-deps` is enough) or
  you get `command not found` for a tool that exists in the source tree;
* `python -m astrotune-prepare` does **not** work — the module is
  `astrotune.cli`. Use the console script, or `python -m astrotune.cli`.

## The model in one page

**The maps.** For every catalog `c` a velocity field `v_θ(·, c)` is integrated
from `t = 0` (the catalog's own parameters) to `t = 1` (harmonized),

    dx/dt = v_θ(x, c),

and that integral is `f_c`. Inverting `f_c` means integrating backwards, so the
map is invertible by construction and its Jacobian comes out of the same
integration (`model/`: `VectorNetwork`, `ODEBijector`). One network,
FiLM-conditioned on the category index `c`, gives every catalog its own map while
sharing the shape of the space.

**One free value per star.** A star seen by `K` catalogs has one true parameter
vector `y_s`; each observation is `x_a = f_{c(a)}^{-1}(y_s) + noise`. With 20
catalogs and 2.3 million stars that is ~9 million free numbers, fitted *per star*
and not by gradient descent (see the Gauss–Newton step below). This is the
"latent-y" model: no catalog is regressed onto another, so inverse-variance
weighting cannot shrink one map towards one survey.

**The data term.** With `σ_a` the per-observation uncertainty (see "two sigmas"
below) and `ρ` the softened χ²,

    χ²_a = Σ_p ( x_{a,p} − [f_{c(a)}^{-1}(y_{s(a)})]_p )² / σ_{a,p}² ,
    ρ(χ²) = S · asinh(χ² / S),      S = chi2_turnover = 50 ,
    L_data = (1/N) Σ_a ρ(χ²_a) .

`ρ` is quadratic in χ² for good points and logarithmic for bad ones, so one
mismatched star cannot dominate a step; its derivative is the weight
`w = [1 + (χ²/S)²]^{-1/2}` used by the solver. A parameter that a catalog does
not measure carries `σ = 1e9`, so its term is ~1e-18 and it drops out of the
sum without a mask.

**The penalties.** A likelihood alone is degenerate where the data are thin —
nothing constrains a map in a corner no star occupies. Each catalog is therefore
held in place by *control points* `ξ` (drawn from that catalog plus a wide
Gaussian covering the whole space), which the map must not move:

    Σ_i |f_c(ξ_i) − ξ_i| / e_i          (l1, l1_total)
    Σ_i ((f_c(ξ_i) − ξ_i) / e_i)²       (l2, l2_total)
    Σ |s|, Σ s²                          (path length of the ODE state, l1_path, l2_path)

plus a penalty against stretching or compressing volume (`weight_div`, measured
on pairs of nearby control points) and weight decay:

    L = L_data + penalty_ctrl + penalty_div + penalty_l2  [+ optional obs path]

## Stage 1 — catalogs to prepared tables

```bash
astrotune-prepare --list                    # keys, labels, parameter counts
astrotune-prepare --all                     # all 20, in registry order
astrotune-prepare apogee_dr17 --no-plots
```

Per catalog: download (cached in `data/raw/`) → read → Gaia crossmatch when the
catalog ships no Gaia ids → derived columns → quality cuts → transform and
assemble → validate → save → diagnostics → parity check.

`data/training/<key>/<key>.h5` has exactly these columns:

| column | meaning |
|---|---|
| `id` | Gaia DR3 source id (int64) |
| `x` | `(n, d)` parameters, in the frozen order |
| `sigma_x` | `(n, d)` quoted σ, rescaled by the catalog's `UNC_SCALES`, then floored |
| `sigma_x_raw` | `(n, d)` quoted σ in our units, no rescale, no floor |
| `x_err` | `(n, d)` regularization error: a prior on how far the map may move a star |
| extras | e.g. `ra_gdr3`, `dec_gdr3`, `ra`, `dec` |

plus `<key>.meta.json` (input rows, cut flow, sources, provenance) and
`diagnostic_plots/<key>/`.

Decisions that matter:

* **The cuts are the notebook's, one set per catalog**, and `cut_flow.png`
  records how many stars each removes. They deliberately exclude corners where
  a grid or pipeline is unreliable (DESI's cool-dwarf locus, LAMOST CSN's `logg`
  range, SLAM's Teff range, ...).
* **Error floors** (0.01, 0.01, 0.04, 0.02) are *regularization*, not physics:
  they keep a catalog quoting a nearly zero uncertainty from claiming infinite
  weight. Applied in quadrature, `σ = hypot(s·q, floor)`.
* **`UNC_SCALES`** lets a catalog admit its quoted errors are off by a constant
  factor (SLAM's Teff errors by 5) before the floors. `astrotune-error-scale`
  measures that factor from the data.
* **Two sigmas.** The bijector trains on `sigma_x` (scaled and floored); the
  delta model learns from `sigma_x_raw` (quoted only), because its job is to
  discover the scale and the additive term itself. Mixing them up gives a
  silently biased answer and no error message.
* **Unmeasured parameters** are `x = 0`, `sigma_x = x_err = 1e9` — never 0 — so
  they vanish from quadratic forms without masks; `1e6` is the "missing"
  threshold on the evidence side.
* **Crossmatches** (VMP, LAMOST CSN, RAVE, DESI DD-Payne, LAMOST DD-Payne) go
  through `prep/gaia.py`. The archive is searched once at 1 arcsec (nothing
  wider is a counterpart) and the matched rows are cached *with their
  separations*; each catalog's own limit (0.3–1.0 arcsec, the notebook's final
  cut) is applied on every read, so it can be changed without another query.
  Per-batch counts, attempts and the archive are recorded in
  `data/interim/<key>/gaia_crossmatch.json`, and a rebuild that matches markedly
  fewer stars than the previous one is refused.
* **Parity.** Every catalog is compared against the legacy notebook output
  (`training_data/`, read-only) on row count, id set, row order and the
  parameter columns; `tests/test_parity.py` enforces it. A catalog that cannot
  reproduce the legacy table must say so (`PARITY_SKIP`).

## Stage 2 — observations and control points

```bash
astrotune-observations [-o FILE] [--dedupe none|first]
                       [--dispersed-xerr per-catalog|uniform]
                       [--sigma-source floored|delta --delta-dir RUN]
```

Writes `data/observations_and_ctrl_points.h5`:

* `obs_dict`: one row per **catalog measurement** of every star that at least
  two catalogs observe — `id`, `x`, `xe`, `xs`, `c`. Two catalogs ⇒ two rows
  sharing an `id`, and that id is what ties them to one `y`.
* `ctrl_dict`: the control points — `x_control`, `xe_control`, `c_control`,
  `from_catalog` (True for points taken from the catalog, False for the
  dispersed ones).

Decisions:

* **Only multi-catalog stars.** One measurement leaves `y` free (its best fit is
  exact) and teaches the maps nothing.
* **No dedupe by default** (`--dedupe none`): a star measured twice by one
  catalog is two observations of one `y`, which is right for repeat
  observations. `--dedupe first` keeps one row per id per catalog.
* **Control-point counts** are chosen so that the penalty weighs about as much
  as the data term: per catalog `min(n_obs/(n_cat−1), max(2·n_obs, 1024), n_rows)`
  points from the catalog itself, plus `n_max/16` dispersed points drawn from a
  wide Gaussian built from the union of all catalogs' 1–99 % boxes.
* **`sigma_source`** chooses which σ the bijector sees: `floored` (the prepared
  `sigma_x`) or `delta` (the calibrated σ of a trained delta model, i.e. the
  iteration of stage 5).
* The file records what made it (`catalogs`, `seed`, `dedupe`,
  `dispersed_xerr`, `sigma_source`, `error_floors`, `n_stars`), and the trainer
  copies those attributes into its `config.json`.

## Stage 3 — train the latent model

```bash
astrotune-train-latent --run-name my_run [--observations-path FILE] [any TrainConfig field]
```

Writes `models/my_run/`: `bijector/` (SavedModel with `to_harmonized` /
`from_harmonized`), `checkpoints/`, `latent_y.npz` (the fitted per-star values),
`config.json` (every hyperparameter *and* the observations file it used),
`metrics_history.json`, `agreement_history.json`, `progress/`.

Steps: a progress bar and a figure (loss history plus the catalogs before and
after the transformation, with the agreement in the title) every 5 minutes
(`--plot-every-seconds`), a checkpoint every ~10 minutes and on Ctrl-C/SIGTERM;
`--resume` continues from the latest one and may only change the plotting and
checkpoint intervals.

### The Gauss–Newton step for y

`y_s` is not updated with the rest of the model. Each step first solves the
star's own normal equations exactly, then takes the map's Adam step at the new
`y` — two blocks of a coordinate descent, which is what keeps the map from
training against values that are about to move, and what keeps the Jacobian
solve outside the tape that differentiates the map.

With `J_a = ∂f_{c(a)}^{-1}/∂y` at `y_s` (transported through the ODE) and
`W_a = diag(w_a / σ_a²)` the curvature weights of `ρ`,

    A_s = Σ_{a∈s} J_aᵀ W_a J_a ,
    b_s = Σ_{a∈s} J_aᵀ W_a (x_a − f_{c(a)}^{-1}(y_s)) ,
    δ   = (A_s + λ diag(mean diag A_s))^{-1} b_s ,
    y_s ← y_s + τ · δ     with |δ|_∞ ≤ y_trust_abs .

Defaults: `λ = y_ridge = 1e-3`, `τ = y_learning_rate = 1`, `y_trust_abs = 0.5`.

* The ridge is diagonal on purpose. A parameter that none of the star's
  catalogs measures has `σ = 1e9`, so that row of `A_s` is ~1e-18 and the system
  is singular; the ridge drives the step in that direction to ~0 instead of
  leaving it undefined, and it damps the nearly flat directions without leaking
  one component's step into another.
* The trust region matters. A full Newton step on a nearly singular `A_s` can be
  hundreds of sigma, and the map then trains on a value that is itself wrong;
  the two can push each other away (`latent_gn_5h` went from χ² 4.1 to 105 in
  fifteen steps before this was added).
* `y` starts at the precision-weighted mean of the star's harmonized
  observations — also the exact minimiser for a linearised map. Held-out stars
  keep their own `y_val`, refitted on every validation batch: profiling a cold
  `y` would make the validation loss differ from the training loss by ~0.3 in χ²
  for that reason alone. `pretrain_steps = 256` holds `y` at that guess first, so
  the maps are fitted against the means before `y` is allowed to move;
  `y_solver="gradient"` keeps the older one-tape variant.

### Training decisions

* **Blocks of stars, not shuffled observations.** All observations of a star sit
  in the same batch, so one solve sees all of its data and lands on the optimum.
  An epoch is `block_shuffles` (default 1) permutations of the star blocks;
  `drop_remainder=False` means the short final batch is used rather than dropped.
* **Validation by star**: ~1/8 of the stars, chosen by a hash of the id, so the
  split is identical across runs and cannot leak a star's other measurements.
* **Hyperparameters** (`LatentTrainConfig`; every field is a CLI flag): 64 units,
  3 hidden layers, batch 65536 observations, 250 epochs, Adam with `beta_2=0.99`
  and AMSGrad, 8 cosine phases with 5 % warmup, global-norm clip 100,
  `chi2_turnover = 50`, ODE `rtol = atol = 1e-5`. The defaults are the reference
  notebook's.
* **Frozen axes**: `--freeze-vars embedding film` (the reference config),
  or `--freeze-after-frac F` to freeze them after a fraction `F` of the run.
* **Validation losses are only comparable within one loss definition.** The
  reference arm and the `y` handling have changed more than once; compare runs
  by per-catalog scale ratios and pair χ² instead, which are definition-free.
* **`y` is state, not model.** It is saved with the run (`latent_y.npz`) but is
  not in the SavedModel: inference for a star goes through `to_harmonized`.

The old pair-loss trainer (`astrotune-train`, `data/pairs_and_ctrl_points.h5`,
`astrotune-evaluate`) still exists and is untouched; it is not the model used
for new runs.

## Stage 4 — calibrate the uncertainties (the delta model)

The bijector makes the catalogs agree; the delta model asks whether the
*uncertainties* they quote were ever right. It is fitted afterwards, on the same
stars, and it corrects each catalog entry's covariance.

```bash
astrotune-apply --run-dir models/my_run            # writes the transform cache it needs
astrotune-delta-train --run-dir models/my_run      # fit; writes models/my_run/delta/
astrotune-delta-diagnose --run-dir models/my_run   # how well calibrated, before and after
astrotune-delta-kiel --run-dir models/my_run       # where in the Kiel diagram it corrected
astrotune-delta-uncertainty --run-dir models/my_run  # quoted vs calibrated σ per catalog
```

`star_table.npz` holds every entry with its parameters, its **catalog-space**
sigma (`sigma_x_raw`), the bijector Jacobian at that entry, its catalog and
group, and the index of its star; the fit is the latent-y likelihood with the
star's value integrated out, so a star counts once however many catalogs saw it.

### The corrected covariance

The model does not touch the diagonal and leave the rest alone. With `C_a` the
entry's covariance propagated from its quoted errors through the Jacobian, and
`ℓ_a`, `α_a` per-parameter *log variance scale* and *additive variance*:

    S_a = M_a C_a M_a + diag(α_a),      M_a = diag(exp(ℓ_a / 2)),

a congruence plus a diagonal term — positive definite for any scale (it is a
change of units), and `M = I, α → 0` reproduces the quoted covariance exactly.
Because `C_a = J_a diag(σ_raw²) J_aᵀ`, this is the same as

    S_a = J_a diag( exp(ℓ_a) · σ_raw² + exp(α'_a) ) J_aᵀ ,

i.e. **the correction acts on the catalog's own quoted sigma and is then pushed
through the Jacobian**. That is the point: `ℓ` and `α` come out in the catalog's
own units, so a scale of 8 means "this catalog's errors are 8× too small" in a
sense comparable across catalogs, which is what stage 5 feeds back.

`ℓ` and `α` are outputs of a small network, `ℓ_a = f_ℓ(x_a, ln σ_a, group)`
(soft-clamped to `σ` within a factor 8 and `α ≥ (3e-3)²`, with a softplus tail so
gradients do not die at the boundary).

### Groups: a shared systematic

APOGEE DR17 and SDSS-V ASPCAP run the *same* pipeline on the *same* spectra, so
their mistakes are common. No rescaling of two covariances can express that: an
additive term inflates both, and therefore inflates their *difference*, while
what is observed is a difference much smaller than the quoted errors predict.

So catalogs are partitioned into groups (`registry.GROUPS`; today one group,
`apogee_dr17 + sdssv_aspcap`), and each group gets a latent value between the
star and its entries:

    t_g = z + u_g,  u_g ~ N(0, G_g);     y_a = t_{g(a)} + ε_a,  ε_a ~ N(0, S_a).

Stacking a group's entries and using Woodbury, the shared term collapses into a
single effective observation of the star:

    ẑ_g = A_g^{-1} Σ_{a∈g} S_a^{-1} y_a ,     V_g = A_g^{-1} + G_g ,
    A_g = Σ_{a∈g} S_a^{-1} .

The group's *central value is unchanged* — it is still the precision-weighted
mean — only its covariance grows by the shared part. The star then has one
effective observation per group, so the Gaussian marginal applies again:

    −2 ln p = [ Σ_g q_g + ln det A + Σ_g ln det V_g + n_res ln 2π ]
              + Σ_g [ ln det Σ_g − ln det V_g ],
    A = Σ_g V_g^{-1},   Σ_g = blockdiag(S_a) + 1⊗G_g 1ᵀ⊗I .

`n_res` counts *groups* instead of entries at the second level, so the model has
exactly as many degrees of freedom as the data: `G_g = 0` collapses this to the
flat model, `G_g → ∞` makes the group one measurement with a systematic error.
The total error of an entry, and the one the merge must use, is

    S_a^tot = S_a + G_{g(a)} .

Two covariances, and conflating them understates a group member's error by
exactly the shared part. For a group of two the data provide two observables
(`S₁+S₂` from within-group differences, `h+G_g` from comparisons with other
groups) for three unknowns, which is why the correction is *shared* within a
group — one `ℓ_g`, one `α_g` applied to each member's own `C_a`. That fixes the
split by construction and is testable: it predicts the composition of the
within-group difference, not just its size. Full derivation:
`notes/error_calibration_model.tex`.

### Student's t tails

Even with the right variances a Gaussian cannot fit the tails: too many pairs
disagree by far more than any Gaussian allows, while the core is already slightly
too wide. Give the star's noise a Gamma-distributed precision `u`
(`u ~ Gamma(ν/2, ν/2)`, shared by all its entries, because "this star is a
binary / a bad match" is a property of the star), and the integral over `u` is a
Gamma integral:

    −2 ln p = ln det A + Σ ln det S_a + (ν + n_res) ln(1 + q/ν)
              + 2 ln Γ(ν/2) − 2 ln Γ((ν+n_res)/2) + n_res ln(νπ) .

So a `t` costs no iteration and needs no augmentation; `ν → ∞` recovers the
Gaussian term for term. `ν_g = 2 + softplus(ϑ_g)` is one number per group, so
`ν > 2` is automatic and the tail weight is a property of the pipeline. Under
this law `q/n_res ~ F(n_res, ν)`, which is the reference distribution for the
coverage and pair-consistency diagnostics.

### What the fit writes

`models/my_run/delta/`: `star_table.npz`, `delta_model.*`, `delta_model_config.json`,
`diagnostics/`, `evaluation/` (nested baselines: quoted σ, the flat 0.03 floor, a
constant per-group scale and additive term, then the network), `kiel/` (one 4×4
page per catalog: a column per parameter, a row per [Fe/H] bin, the median learned
variance in Teff–logg bins, plus `kiel_cube.npz`), `uncertainty_distribution/`
and `evaluation/shared.txt` (the fitted `G_g` per group and parameter). Weights
are fitted with Adam against the star likelihood plus weak priors that say what
the correction should be where the data are silent (nothing added, no shared term),
imposed both on the data and on control points.

Three degeneracies to know when reading the numbers: scale and additive term are
separated only by the *spread* of quoted uncertainties within a group; inside a
group only the sum `S₁+S₂` is measured (hence the shared correction); and a group
that is weakly connected to the rest pins only a combination, so its numbers are
prior-dominated and best read as differences.

## Stage 5 — iterate the two models

The delta model's output does not depend on the bijector, so its calibrated σ can
be fed back and the bijector retrained:

```bash
astrotune-observations --sigma-source delta --delta-dir models/lrXX -o data/obs_delta.h5
astrotune-train-latent --run-name next --observations-path data/obs_delta.h5
astrotune-apply --run-dir models/next
astrotune-delta-train --run-dir models/next --rebuild
```

`--rebuild` is essential after retraining the bijector: the star table stores the
Jacobian at every entry, and that Jacobian has changed. Rebuilding it is the
expensive step (every catalog re-mapped), so do not reuse a star table across
runs. The five phases of the loop are described in `porting_project.md` §9.10.

## Stage 6 — apply and merge

```bash
astrotune-apply --run-dir models/my_run [--reuse-transform] [--eps-err 0.03]
```

`apply` maps every catalog to the harmonized space, propagating each entry's
covariance through the Jacobian, and combines the entries of each Gaia DR3 source
by the product of Gaussians — a precision sum, iterated with χ² outlier
rejection (`chi2_max = 30`, the entry or entries with the lowest χ² are never
dropped; a star left with one entry takes it). With `C_i` the propagated
covariance plus a regularization `ε² I` (`eps_err`, 0.03),

    C⁻¹ = Σ_i (C_i + ε²I)^{-1} ,     ŷ = C Σ_i (C_i + ε²I)^{-1} y_i ,
    χ²_i = (y_i − ŷ)ᵀ (C_i + ε²I)^{-1} (y_i − ŷ) .

Writes `models/my_run/harmonized/`: `merged_catalog.h5` (one row per star,
`params_source` a bit mask of the catalogs that contributed), `individual_estimates.h5`,
`metadata.json` (bit → catalog, merge settings, counts, training provenance) and
`summary.png`; read it back with `astrotune.apply.read_merged`. See
`porting_project.md` §9.9 for the format and the fixes to the notebook's merge.

**Note:** the merge still treats entries as independent, so a group contributes
twice (TODO.md, "Correlated entries in the merge"); the right weight is
`V_g = A_g⁻¹ + G_g`, and `params_err` still adds a hand-set 0.03 floor.

## Other tools

`astrotune-evaluate --run-dir models/my_run` (also run at the end of training)
compares `x_j − x_k` with `f_j(x_j) − f_k(x_k)` for held-out stars, every pair
and parameter, and writes `evaluation/agreement.json`, `agreement_summary.png`
(each point a catalog pair; below the diagonal is an improvement),
`pairs/pair_JJ_KK.png` and `param_shifts.png`. It belongs to the pair-loss era
and reads the legacy pairs file.

`astrotune-videos --run-dir models/my_run` writes the transformation as a movie
to `models/my_run/videos/`, in planes `kiel` ([Teff, logg]) and `feh_alpha`
([Fe/H], [α/Fe]) — unhighlighted, one catalog highlighted in colour, or the two
sides of a catalog pair (same stars, as measured by each; they should converge).
Pair-only planes `param1v1` and `param1v1_diff` compare the two catalogs
parameter by parameter (2×2 grid; onto the diagonal and onto zero).

```bash
astrotune-videos --run-dir models/my_run --standard-overview   # the set you usually want
```

`--standard-overview` is kiel + feh_alpha + param1v1, the plain videos, one
highlight per catalog, and every pair with more than 10 000 shared stars plus
each catalog's three strongest, so no catalog is left out. The data (control
points, pairs, counts) come from the observations the run was trained on —
`--pairs-path` forces the legacy pair file instead. The encoder is `ffmpeg`.

`astrotune-error-scale <key> --run-dir models/my_run [--partners K1 K2]` asks
whether a catalog's quoted uncertainty is right: binning partners' differences by
the quoted σ, the excess variance is fitted as `v = s²q² + σ_sys²` (scale and
plateau), in the harmonized space so that systematic offsets do not inflate the
scatter. Figures in `diagnostic_plots/<key>/error_scale*`; see
`porting_project.md` §5.7.

`astrotune-catalog-graph [--layout spring|circle] [--recount]` draws the
connectivity: circles (radius ~ log rows, colour = family of spectra, stroke =
missing parameter) joined by edges (~ log shared stars), plus the all-pairs
matrix, to `diagnostic_plots/catalog_graph/`, with counts in
`catalog_overlaps.json` and PageRank in `catalog_pagerank.ecsv`. No model
needed; `--recount` after the catalogs change.

## Gotchas

1. **Two sigmas.** `sigma_x` (scaled, floored) is what the bijector and the
   apply-step use; `sigma_x_raw` (quoted only) is what the delta model learns
   from. Using the wrong column is silent and biased.
2. **Error floors are regularization.** They exist to stop a catalog claiming
   infinite weight; they are not measurement errors and must not be reported as
   such.
3. **Unmeasured parameters** are `x = 0`, `sigma_x = x_err = 1e9`. Never 0 in
   the sigma, or the entry claims to know a value it does not measure.
4. **Parameter order is frozen** (`config.PARAM_NAMES`). The index is part of
   every on-disk format and of the model's output layer.
5. **`y` is run state, not model.** `latent_y.npz` is not in the SavedModel;
   inference goes through `to_harmonized`.
6. **The star table stores Jacobians** — rebuild it (`--rebuild`) whenever the
   bijector it was built with changes.
7. **Groups are not independent.** `apogee_dr17` and `sdssv_aspcap` share a
   pipeline; the delta model knows, the merge does not yet.
8. **A rebuild that loses stars is refused** (`data/interim/<key>/gaia_crossmatch.json`).
   If a crossmatch legitimately changes, say so with
   `--allow-crossmatch-regression`; do not disable the check by habit.
9. **The Gaia archives cap results.** ESA and GAVO both silently truncate at
   `MAXREC` (GAVO: 20 001 rows), and GAVO/ARI leave a job in `PENDING` until it
   is set to `RUN`. `prep/gaia.py` handles both, retries, splits a capped batch
   and refuses a cache that loses stars. `--gaia-archive {esa,gavo,ari,ari-lite}`
   chooses the service; GAVO (`gaia.dr3lite`) reproduced the legacy tables
   exactly and is what to use when ESA is unavailable.
10. **`astrotune-prepare` is a script, not a module** — `python -m astrotune-prepare`
    fails with "No module named". Use the console script or `python -m astrotune.cli`.
11. **Entry points come from `pip install -e .`** — after `pyproject.toml` gains a
    command, rerun it or the command is missing while the code is fine.
12. **`data/` and `models/` are not tracked.** `--force` re-downloads and
    re-queries everything; without it, caches are reused, which is what you want.
13. **Validation numbers are not comparable across loss revisions.** Compare
    per-catalog scale ratios and pair χ².
14. **The GPU is shared**; memory growth is enabled before allocation. Two
    training jobs at once will slow both down.

## Open issues

`TODO.md` is the live list. The main ones: the hand-set 0.03 floor inside the
merge covariance and in `params_err`; correlated entries in the merge and in the
latent `y`-solve (the group structure above); removing [O/Fe] from GALAH's
[α/Fe]; and the remaining phases of the latent↔delta iteration.
`porting_project.md` records every decision made while porting the notebook
(§5.7 error scales, §9.9 merge format, §9.10 the delta model).

## Catalog module contract

Each module in `astrotune/catalogs/` provides:

```python
KEY, LABEL, N_PARAMS          # identity
REFERENCE, SOURCES            # provenance, recorded in the sidecar metadata
ID_COL, PARAM_COLS, UNC_COLS, EXTRA_COLS     # column names in the raw table
X_ERR                         # regularization error per parameter
ERROR_FLOORS                  # optional: per-catalog override of config.ERROR_FLOORS
UNC_SCALES                    # optional: constant rescale of the quoted sigma
HAS_LOGG, PARITY_SKIP         # optional: figures and the parity exemption text

def download(ctx) -> None                 # populate ctx.raw_dir (cached)
def load(ctx) -> Table                    # read raw files; apply the Gaia crossmatch
def prepare_columns(t, ctx) -> Table      # optional: derived columns

def cuts(t, ctx) -> dict[str, mask]       # named quality cuts (the cut-flow figure)
def param_errs(t, ctx) -> (n, d) array    # regularization errors
def unc_transf(d) -> callables            # optional: sigma rescale before the floors
def finalize(table, ctx) -> Table         # optional: final touch-ups
def diagnostics(t, mask, table, ctx)      # optional: catalog-specific figures
```

The shared pipeline (`prep/pipeline.py`) handles the transformation, the error
floors, `sigma_x`/`sigma_x_raw`, saving, validation and the generic diagnostic
figures. Crossmatched catalogs do not implement the query themselves: they call
`prep.gaia.crossmatch_catalog` with their `MAX_SEPARATION` (the notebook's final
cut) and let it cache the wide search, guard the counts and write the report.
