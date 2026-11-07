# Astrotune porting project — handoff notes

Status as of the latest commit: **the catalog layer is complete (16/16 catalogs built
and verified); the training code is the next thing to port.**

This document is meant to let a new agent (or person) pick up the work without
re-deriving context. Read it top to bottom, then start with
[§9 What remains](#9-what-remains-the-training-code).

---

## 1. What the project is

**Astrotune** learns a set of minimal, invertible transformations that map each
input stellar-parameter catalog into a common ("harmonized") space, using stars
observed in more than one survey as anchors.

Each catalog `c` gets a bijection `f_c: x -> y` defined by integrating an ODE

```
dx/dt = v_c(t, x),   x(0) = x,   y = x(1)
```

where the velocity field `v_c` is a neural network conditioned on a categorical
embedding of the catalog. Training minimises a pairwise consistency loss
(does a star get mapped to the same place from either catalog?) plus
regularisation terms (path length, distance from identity, divergence, weight
decay).

Available knowledge:

| Source | Contents |
|---|---|
| `harmonized_parameters.tex` | The paper. §2 "Astrotune Method" (line ~297) has the formal method, losses and training recipe; Appendix A "Input stellar catalogs" (line ~482) documents every cut and parameter mapping per catalog. |
| `data/_reference/notebook_flat.py` | **The reference implementation**, flattened to plain Python with `# ===== cell N (kind) =====` markers (7157 lines). This is what you port from. |
| `merge_coordinates.ipynb` | Original notebook, 119 MB (mostly embedded videos/images). Do not open it; keep it only as an archive. |
| `training_data/` | Outputs of the original notebook, kept **read-only as the parity reference**. |
| `data/training/<key>/` | The new per-catalog tables. |

`data/_reference/notebook_flat.py` is inside `data/`, so it is **gitignored**. If it
is missing, regenerate it from the media-stripped notebook:

```python
import json, pathlib
nb = json.loads(pathlib.Path("merge_coordinates_no_media.ipynb").read_text())
out = []
for i, cell in enumerate(nb["cells"], 1):
    out.append(f"# ===== cell {i} ({cell['cell_type']}) =====\n"
               + "".join(cell["source"]) + "\n")
pathlib.Path("data/_reference/notebook_flat.py").write_text("\n".join(out))
```

Cell numbers quoted below always refer to that file.

---

## 2. Environment

```bash
.venv/bin/python                 # Python 3.9.25
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

| Package | Version | Note |
|---|---|---|
| numpy / scipy / astropy | 1.26.4 / 1.13.1 / 6.0.1 | numpy pinned < 2.1 |
| h5py / pandas / tables | 3.14.0 / 2.3.3 / 3.9.2 | `tables` is needed for the RAVE `pandas.read_hdf` files |
| matplotlib / seaborn / distinctipy | 3.9.4 / 0.13.2 / 1.3.4 | plotting |
| tensorflow `[and-cuda]` | 2.18.1 | bundles CUDA 12.5 + cuDNN 9.3 via `nvidia-*` wheels |
| tf-keras | 2.18.0 | **required** by TFP ≥ 0.25 on TF ≥ 2.16; import fails without it |
| tensorflow-probability | 0.25.0 | bijectors, `DormandPrince` ODE solver |
| dm-sonnet | 2.0.2 | `snt.Module`, `snt.Linear`, `snt.Embed`, `snt.LayerNorm` |

GPU: NVIDIA A100 80GB (PCIe), driver 615.71.10. Verified that
`tf.config.list_physical_devices('GPU')` returns a device and that a matmul
actually runs on it. There are **no CUDA/cuDNN modules** in the Lmod tree
(only `mpi/*`) and `/usr/local/cuda-12.9` has no cuDNN, hence the pip wheels.
Do not `module load` anything for GPU work.

Useful commands:

```bash
.venv/bin/python -m astrotune.cli --list          # catalogue registry
.venv/bin/python -m astrotune.cli apogee_dr17     # build one catalog
.venv/bin/python -m astrotune.cli --all           # build all sixteen
.venv/bin/python -m astrotune.cli --all --no-plots
.venv/bin/python -m pytest -q                     # 54 pass, 1 skip
```

---

## 3. Repository layout

```
astrotune/
  config.py          paths (env-overridable), parameter order, error floors
  registry.py        THE ordered catalog list — index == category index c
  cli.py             `astrotune-prepare` entry point
  prep/
    download.py      cached HTTP downloads, SDSS-V auth via env/prompt
    transforms.py    Teff transform, error floors, unmask(), extract_table()
    cuts.py          finite_mask(), snr_chisq_relation()
    abundances.py    calc_alpha_Fe_from_abundances()
    gaia.py          batched Gaia DR3 crossmatch + apply_crossmatch()
    io.py            save/load/validate, sidecar metadata, parity_report()
    plots.py         diagnostic figures (headless, always written to disk)
    pipeline.py      orchestration + Context/dataclass + module hooks
  catalogs/          one module per catalog (16 files)
tests/
  test_transforms.py unit tests
  test_parity.py     schema + row-for-row comparison with training_data/
data/
  raw/<key>/         downloaded source files            (gitignored)
  interim/<key>/     derived caches (Gaia crossmatches)  (gitignored)
  training/<key>/    <key>.h5 + <key>.meta.json          (gitignored)
  _reference/        notebook_flat.py                    (gitignored)
diagnostic_plots/<key>/                                 (gitignored)
training_data/       legacy notebook outputs, read-only  (gitignored)
models/              checkpoints, saved bijectors        (gitignored)
harmonized_parameters.tex   the paper (tracked)
requirements.txt, pyproject.toml, README.md, TODO.md
```

Two path overrides exist for cluster use: `ASTROTUNE_DATA_DIR`,
`ASTROTUNE_DIAGNOSTIC_DIR`.

---

## 4. Data contract

Every prepared catalog is an `astropy.table.Table` written with
`serialize_metadata=True` to `data/training/<key>/<key>.h5`:

| column | shape / dtype | meaning |
|---|---|---|
| `id` | `(n,) int64` | Gaia DR3 `source_id`; rows with `id <= 0` dropped |
| `x` | `(n, d) float32` | transformed parameters, `d` = 3 or 4 |
| `sigma_x` | `(n, d) float32` | statistical uncertainties (incl. error floors); `1e9` = "no measurement" |
| `x_err` | `(n, d) float32` | **regularisation** errors (paper App. A); `1e9` for dummy axes |
| extras | `(n,)` | `ra`/`dec` (names vary per catalog, see each module's `EXTRA_COLS`) |

Parameter order is fixed: `(5*ln(Teff/5000 K), [Fe/H], logg, [alpha/Fe])`.
Catalogs without `[alpha/Fe]` store only 3 columns; padding to 4 happens in the
training-data stage (`x = 0`, `sigma_x = x_err = 1e9`).

A `.meta.json` sidecar records row counts per cut, source URLs, git commit,
package versions — use it to regenerate the numbers quoted in Appendix A.

**`sigma_x` vs `x_err`**: `sigma_x` is statistical and drives the pair chi²;
`x_err` is the regularisation scale that says how much the transformation is
allowed to move along each axis. They are not interchangeable.

---

## 5. The catalog layer (done)

### 5.1 Module contract

A module in `astrotune/catalogs/` supplies metadata plus a few functions; the
shared pipeline does everything else:

```python
KEY, LABEL, N_PARAMS, REFERENCE, SOURCES
ID_COL, PARAM_COLS, UNC_COLS, EXTRA_COLS
ERROR_FLOORS            # optional; default config.ERROR_FLOORS[:d]
HAS_LOGG                # optional, False -> Teff/[Fe/H] figure instead of Kiel
PARITY_EXEMPT           # optional str: intentional deviation from legacy table

def download(ctx)                     # populate ctx.raw_dir (cached)
def load(ctx) -> Table                # read raw files, apply cached crossmatch
def prepare_columns(t, ctx) -> Table  # optional derived columns (e.g. alpha/Fe)
def cuts(t, ctx) -> dict[str, mask]   # named, applied cumulatively
def param_errs(t, ctx)                # x_err, full-length (n,d) or per-axis constants
def unc_transf(d)                     # optional rescale of uncertainties before floors
def finalize(table, ctx) -> Table     # optional post-assembly fix (MDC dummy axis)
def diagnostics(t, mask, table, ctx)  # optional extra figures
```

### 5.2 Registry order is frozen

`astrotune/registry.py` is the single source of truth. **The position in
`CATALOGS` is the category index `c`** used by the model and baked into every
checkpoint. Never reorder; append only.

| c | key | label | n_par | rows built |
|---:|---|---|---:|---:|
| 0 | `apogee_dr17` | APOGEE (DR17) | 4 | 572,775 |
| 1 | `lamost_mrs` | LAMOST MRS | 4 | 854,512 |
| 2 | `lamost_hotpayne` | LAMOST HotPayne | 4 | 361,825 |
| 3 | `galah_dr4` | GALAH (DR4) | 4 | 530,521 |
| 4 | `lamost_ddp_dr9` | LAMOST DD-Payne (DR9) | 4 | 4,115,061 |
| 5 | `lamost_mstar_ding2022` | LAMOST M-Stars (Ding) | 3 | 308,548 |
| 6 | `vmp_stars_li2022` | VMP (Li) | 3 | 298 |
| 7 | `lamost_csn_dr10` | LAMOST M-Dwarfs (Zhang) | 3 | 272,776 |
| 8 | `desi_dr1` | DESI (DR1) | 4 | 3,214,877 |
| 9 | `desi_ddp` | DESI (Zhang) | 4 | 141,766 |
| 10 | `sdssv_aspcap` | SDSS-V (ASPCAP) | 4 | 467,719 |
| 11 | `sdssv_bossnet` | SDSS-V (BOSSNet) | 3 | 268,589 |
| 12 | `rave_dr6` | RAVE (DR6) | 4 | 174,871 |
| 13 | `pristine_pigs` | PIGS | 3 | 11,425 |
| 14 | `sdssv_slam` | SDSS-V (SLAM) | 4 | 21,162 |
| 15 | `sdssv_mdc` | SDSS-V (MDw Cannon) | 4 | 15,994 |

This order matches `table_pairs` in cell 349 of `notebook_flat.py`, so legacy
checkpoints and pair files remain interpretable.

### 5.3 Verification policy

`tests/test_parity.py` compares each built table against
`training_data/<key>.h5`: identical row count, identical id set, identical row
order, and `max|x_new - x_legacy| < 1e-5` (float32 round-off). All 16 pass.

One deliberate exception: **`lamost_hotpayne`**. The original notebook used the
`[alpha/Fe]` *value* column as that axis' uncertainty, so `sigma_x[:, 3]` was
the abundance itself. The port uses `e_[alpha/Fe]`. The module declares
`PARITY_EXEMPT`; the pipeline logs it and the test still requires rows, ids,
order, `x`, `x_err` and `sigma_x[:, :3]` to match, and asserts that
`sigma_x[:, 3]` *does* differ. If you retrain, this changes the alpha-axis
weighting for that catalog relative to the published run.

### 5.4 Gotchas discovered during the port

- **Duplicate Gaia ids are real and large**: 53,403 in APOGEE DR17, 134,120 in
  DESI DR1, 36,583 in ASPCAP. They reproduce the legacy tables exactly, so the
  current per-catalog behaviour was preserved. This matters for the *join*:
  `astropy.table.join(..., keys='id', join_type='inner')` is many-to-many and
  will explode those rows into all pairwise combinations. **Decide a
  deduplication policy before building pairs.**
- Catalogs with 3 parameters are padded to 4 in the pairs/control-point stage,
  not in the catalog layer.
- `sdssv_mdc` has no `logg` at all: it carries a dummy 4th axis with
  `x = 0`, `sigma_x = x_err = 1e9`, and `HAS_LOGG = False`.
- `lamost_mstar_ding2022` applies no error floors (`ERROR_FLOORS = (0, 0, 0)`)
  and must promote its integer columns to float before the paper's fractional
  offsets.
- `galah_dr4` uses `e_fe_h` (not `e_feh`), and `logg == 5` is a placeholder.
- DESI DR1 stacks `backup, dark, bright` in that order; changing it changes row
  order (not content) and breaks the row-order parity assertion.
- `lamost_mrs`: the `.fits` and `.h5` copies on disk are identical; the module
  reads the FITS.
- Several catalogs cache a **matched catalog** under `data/interim/<key>/`;
  `lamost_ddp_dr9` does too now (`crossmatched.h5`). The stale
  `crossmatch.h5.stale` there is the old crossmatch-table cache and can be
  deleted.

---

## 6. The method as implemented (what training must do)

Read §2 of the paper first; this is the mapping to code.

**Model** (cells 4–7):
- `VectorNetwork` — MLP with FiLM conditioning on a learned catalog embedding;
  takes `(t, x)` concatenated and the integer category `c`; returns velocity.
  Defaults: `output_scale=1e-5`, residual connections, layer norm.
- `CatMLP` — simpler concatenation-based variant, unused in the final run.
- `ODEBijector(tfb.Bijector)` — wraps the velocity network and a
  `tfp.math.ode.DormandPrince` solver. Key methods:
  `_forward`/`_inverse` (plain map), `forward_with_s`/`inverse_with_s` (also
  return the integrated path length `s`, used as a penalty),
  `augmented_forward` (adds `ln det J`), `return_path` (trajectory for movies).
  Note `_forward_log_det_jacobian` raises `NotImplementedError` on purpose.

**Training data** (cells 346–361):
- `extract_pairs(t0, t1, c=(j, k), rng)` inner-joins two catalogs on `id`,
  randomly orients each pair, and returns `x0, x1, xe0, xe1, xs0, xs1, c0, c1`
  with missing parameters padded to `(x=0, xe=xs=1e9)`.
- Control points: per catalog `n_sel = min(n_pairs/(n_cat-1), max(2*n_paired, 1024), len(t))`
  sampled from the catalog, plus `n_draw = n_max//16` dispersed points per
  catalog drawn from a wide Gaussian built from the union of the 1–99 percentile
  boxes. Stored as `x_control, xe_control, c_control, from_catalog`.
- RNG: `np.random.default_rng(3)`, shared and stateful across both stages, so
  the order of operations matters for exact reproduction.
- Written with `h5py` to `training_data/pairs_and_ctrl_points.h5` under groups
  `pair_dict/` and `ctrl_dict/` (gzip level 3, chunked).
- Post-load fix-up (cell 360): `xe0/xe1 > 1e6 → 0.05`, `xe_control > 1e6 → 0.05`.

**Loss** (cell 412, `grad_step`):
1. Noise injection: `x0 += clip(xs0, 1e-5, 1) * N(0,1)`; control points get
   `sigma_ctrl * N(0,1)` jitter and a second copy offset by `eps_div` for the
   divergence term.
2. Learned per-catalog error model `ln_err_model = VectorNetwork(...)` gives a
   multiplicative correction to `xs1`; the pair term is a log-space chi² with an
   importance-sampling correction (see the paper's pair loss), then softened
   with `chi2_turnover * arcsinh(chi²/chi2_turnover)` computed stably via
   `arcsinh_of_exp()` (uses `tf.keras.ops.logaddexp` + `softplus`).
3. Penalties: path length (`l1_path`, `l2_path`) on pairs and control points;
   distance of control points from identity normalised by `xe_ctrl`;
   divergence `weight_div * log((s/s')² + eps)²`; weight l2 (`l2_weights`).
4. Adam with `PiecewiseConstantDecay` halving `1e-3` eight times; global-norm
   clipping at 100; 32 epochs, batch 1024, `val_frac = 1/8` (validation = the
   **first** `n_val` rows).
5. Late in training the embedding and FiLM variables are frozen
   (`var_freeze_list`). Checkpoints: `tf.train.Checkpoint(bij=bij)` into
   `models/`, plus `tf.saved_model.save` with explicit `to_harmonized` /
   `from_harmonized` signatures.

**Applying the model** (cells 461–488):
- `transform_estimates` uses `tf.GradientTape` + `batch_jacobian` to push the
  per-star covariance through the map (`cov_y = J diag(sigma²) Jᵀ`).
- Per-catalog results go to `transformed_data.h5` groups `catalog_{i}` with
  `params_orig/params_err_orig/params_harm/params_cov_harm/gdr3_source_id`.
- `combine_parameters` merges multiple estimates of the same Gaia source by
  inverse-covariance summation (with a 0.03 dex diagonal regulariser), tracking
  a `params_source` bitmask; a few iterations drop entries with chi² > 25 (or
  > 9 in the later variant).
- Final product `merged_params_catalog_clean.h5` with
  `params_est/params_err/params_cov/lndet_cov/params_source`.

---

## 7. Conventions

- **No notebooks** in the new code. Everything is a module or a CLI command.
- **Diagnostic plots are on by default** and written to
  `diagnostic_plots/<catalog>/`; `--no-plots` disables them. Training and
  inference should follow the same pattern (figures to disk, not `plt.show()`).
- Plots are headless (`matplotlib` falls back to `Agg` when `DISPLAY` is unset).
- Data, models, plots and the legacy notebook are **never tracked**; see
  `.gitignore`. The old git history lives in `.git.old/` (also ignored).
- Logging via the `logging` module; the CLI configures the format.
- Every stage must be able to prove it reproduces the legacy artefact, or must
  declare its deviation in code near the thing that deviates.

---

## 8. Reproduction of the catalog layer

```bash
.venv/bin/pip install -e .
.venv/bin/python -m astrotune.cli --all
.venv/bin/python -m pytest -q          # 54 passed, 1 skipped
```

Catalogs that need the network are already cached under `data/raw/`; SDSS-V
downloads need `SDSS_USER` / `SDSS_PASSWORD` (or an interactive prompt). Gaia
crossmatches are cached under `data/interim/` and are only recomputed with
`--force`.

---

## 9. What remains: the training code

Suggested module split, mirroring the catalog layer:

```
astrotune/model/vector_network.py   cells 4–5   VectorNetwork
astrotune/model/ode_bijector.py     cells 6–7   CatMLP, ODEBijector
astrotune/training_data.py          cells 346–361  pairs + control points
astrotune/train.py                  cells 374–417  datasets, grad_step, loop, save
astrotune/apply.py                  cells 461–488  transform + merge
astrotune/diagnostics.py            cells 362–373, 419–453, 456–460, 526–544
```

Recommended order and the specific traps in each step:

1. **`model/`** — mechanical port; keep `output_scale=1e-5`, the residual/FiLM
   defaults and `DormandPrince(rtol=atol=1e-5)`. Sanity-check with the 2D toy
   problem (cells 8–16) as a test: it converges in a few thousand steps without
   any data files.
2. **`training_data.py`** — port `extract_pairs` and the control-point sampler
   verbatim first, then verify against `training_data/pairs_and_ctrl_points.h5`
   (counts per pair, distributions). Expect small differences for
   `lamost_hotpayne` because of the `sigma_x` fix. **Resolve the duplicate-id
   question first** (§5.4) — it changes pair counts materially.
3. **`train.py`** — port `grad_step` *last*, after the data is byte-comparable.
4. **`apply.py`**, then **`diagnostics.py`**. The legacy artefacts to compare
   against are `transformed_data.h5` and `merged_params_catalog_clean.h5` in the
   repo root.

### Bugs in the notebook that must be fixed while porting (do not copy)

- Cell 391/393/394: `variables`, `metrics_history`, `n_steps` are used before
  being defined — artefacts of out-of-order execution.
- Cell 397: `err_model` should be `ln_err_model`.
- Cell 412: `f'{n_bij_variables}'` and `f'{n_err_variables}'` raise
  `NameError` (use `n_bij_vars`, `n_err_vars`); `chi2_tunover` is a typo for
  `chi2_turnover`; `s_ctrl1` is unused.
- Cell 428: `np.concat` only exists in NumPy ≥ 2 — use `np.concatenate`.
- Cell 456 (`get_transformed_pairs`): reads global `x_ctrl` / `c_ctrl` left over
  from the training loop. Pass them explicitly.
- Cell 460: uses `d`, the loop variable left over from cell 387/412.
- Cell 487/511: `params_comb, params_cov_comb = combine_parameters(...)` — the
  function returns a dict.
- Cell 519: `lndet_cov` is undefined there; it lives in the results dict.
- Cell 536: `v[..., 0].ndim` with `v` undefined.
- Cell 464: relies on the global `n_params` (= 4) and on `tables`.

### Open questions to settle before training

1. **Duplicate Gaia ids** (§5.4): dedupe per catalog (e.g. keep highest SNR), or
   allow the many-to-many join? This is the biggest open decision.
2. **`lamost_hotpayne` sigma fix**: retrain from scratch (clean but not
   comparable to the published numbers), or reproduce the old behaviour behind
   a flag? Current code uses the corrected values.
3. Whether the learned error model (`ln_err_model`, importance-sampling term) is
   worth keeping; it was still experimental in the notebook.
4. Where checkpoints live: `models/` is fine for now but it is a single-run
   directory. Consider `models/<run-name>/` with a config snapshot.
5. The notebook's hyperparameters (cell 412) are hand-tuned; keep them as the
   default so a fresh run is comparable to the paper.
