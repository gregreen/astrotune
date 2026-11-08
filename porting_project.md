# Astrotune porting project — handoff notes

Status: **the catalog layer is complete (16 legacy catalogs built and verified,
plus 3 new ones promoted into the model on 2026-11-08: 19 in total);
the training layer is ported and working (model, pairs/control points, training
loop with resume, evaluation, videos) and is being tuned; `apply.py`/`merge.py`
(applying the model and merging by Gaia id) are written and tested but have not
yet been run on the full catalogs; most of the paper-figure diagnostics are not
ported yet.**

This document is the main hand-off document. It is meant to let a new agent (or
person) pick up the work without re-deriving context. Read it top to bottom,
then go to [§9 The training layer](#9-the-training-layer) for the current state
of the work and [§10 What remains](#10-what-remains) for the to-do list.

Two kinds of knowledge are recorded here: things that are **verified** (with the
evidence) and things that are **hypotheses**; they are labelled as such.

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
| imageio-ffmpeg | 0.6.0 | static `ffmpeg` 7.0.2 (with libx264) for the videos; no system ffmpeg is installed. `astrotune.videos` uses an `ffmpeg` on `PATH` if there is one |

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
.venv/bin/python -m pytest -q                     # catalog layer: 54 pass, 1 skip
.venv/bin/python -m pytest -q tests/test_training.py   # 6 tests, ~3 min (one trains on GPU)
```

Training commands are in [§9.2](#92-how-to-run).

---

## 3. Repository layout

```
astrotune/
  config.py          paths (env-overridable), parameter order, error floors
  registry.py        THE ordered catalog list — index == category index c
  cli.py             `astrotune-prepare` entry point
  model/             VectorNetwork, ODEBijector, load_bijector(run_dir)     (TensorFlow)
  training_data.py   pairs + control points            (`astrotune-pairs`)
  train.py           TrainConfig, Trainer, resume      (`astrotune-train`)
  evaluate.py        agreement before/after harmonization (`astrotune-evaluate`)
  training_plots.py  figures written during training
  videos.py          Kiel / [Fe/H]-[alpha/Fe] / param1v1 videos (`astrotune-videos`)
  apply.py           harmonize each catalog, merge by Gaia id (`astrotune-apply`)
  merge.py           combination of Gaussian estimates + outlier rejection (NumPy only)
  prep/
    download.py      cached HTTP downloads, SDSS-V auth via env/prompt
    transforms.py    Teff transform, error floors, unmask(), extract_table()
    cuts.py          finite_mask(), snr_chisq_relation()
    abundances.py    calc_alpha_Fe_from_abundances()
    gaia.py          batched Gaia DR3 crossmatch + apply_crossmatch()
    io.py            save/load/validate, sidecar metadata, parity_report()
    plots.py         diagnostic figures (headless, always written to disk)
    overlaps.py      stars shared with the other catalogs, by Teff (`astrotune-overlaps`)
    pipeline.py      orchestration + Context/dataclass + module hooks
  catalogs/          one module per catalog (19 in the model, §5.5; `CANDIDATES` is empty)
tests/
  test_transforms.py unit tests
  test_parity.py     schema + row-for-row comparison with training_data/
  test_training.py   pairs/control points, a short real training step, video helpers
data/
  raw/<key>/         downloaded source files            (gitignored)
  interim/<key>/     derived caches (Gaia crossmatches)  (gitignored)
  training/<key>/    <key>.h5 + <key>.meta.json          (gitignored)
  pairs_and_ctrl_points.h5   training pairs + control points (gitignored)
  _reference/        notebook_flat.py                    (gitignored)
diagnostic_plots/<key>/                                 (gitignored)
training_data/       legacy notebook outputs, read-only  (gitignored)
models/<run-name>/   one directory per training run      (gitignored)
models/bij_checkpoint-1.*, models/bijector/, models/metrics_history.json
                     the *legacy* notebook's trained model, read-only
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
PARITY_DIFFERS          # required with PARITY_EXEMPT: {"x_err": (3,)} = which parameters
                        # of which columns differ from the legacy table
X_ERR_DISPERSED         # optional: x_err of the catalog's dispersed control points
                        # (default: its median measured x_err, per parameter; §9.3)

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

Two deliberate exceptions. Each declares `PARITY_EXEMPT` (the reason) and
`PARITY_DIFFERS` (exactly which parameters of which columns differ); the test
requires every other column to match and the declared ones to really differ.
The second is **`sdssv_slam`**: the owner raised its `[alpha/Fe]` regularization
error `x_err[:, 3]` from the legacy 0.2 to 0.4, because SLAM's [alpha/Fe]
differs from nearly all other surveys.

The first is **`lamost_hotpayne`**. The original notebook used the
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

### 5.5 The three newest catalogs (`desi_nagaraj2026`, `sdssv_clam`, `lamost_mstar_li2025`)

**Status (2026-11-08): promoted.** They were first prepared as *candidates*
(`registry.CANDIDATES`: no category index, ignored by training), and the owner
then had them **appended to `registry.CATALOGS` (indices 16, 17, 18)**, so
`models/turnover50_8ep_19cat` is the first 19-catalog run (§9.4). `CANDIDATES` is
now empty but the mechanism remains (§5.6). Older runs were trained with the first
16 only; `registry.run_keys(run_dir)` reads a run's catalog list from its
`config.json` (first 16 if absent), and `load_bijector`, `evaluate` and `videos`
use it, so old runs still load and evaluate (give them their own `--pairs-path`).
The sections below describe how each catalog was made and why.

**`desi_nagaraj2026` — DESI (Nagaraj)**, Nagaraj et al. 2026 (arXiv 2610.04294),
a neural-network (Payne-style) fit to the un-normalised DESI SEDs. Raw file:
`data/raw/desi_nagaraj2026/desi_dr1_nagaraj2026.fits` (Zenodo record 21816591;
the downloader fetches it and was **verified byte-identical** to a manual
`wget`). Prepared table: `data/training/desi_nagaraj2026/` (**80,455 stars**).
Everything below was determined by looking at Kiel and [Fe/H]–[α/Fe] diagrams
and at the stars shared with APOGEE/GALAH/ASPCAP, and is documented in the
module docstring:

- **It is a partial release**: 170,111 stars, ~5% of the 3.2 million in the
  paper. The full catalog is not yet available (Zenodo, VizieR "before final
  publication"); overlaps with other surveys should grow roughly in proportion
  if the partial sample is representative (not verified).
- Has Gaia DR3 ids (no crossmatch); 15,042 rows have id −1 and are dropped.
  There is **no [α/Fe] column**: it is the inverse-variance mean of [Mg/Fe],
  [Si/Fe], [Ca/Fe], [Ti/Fe] (`calc_alpha_Fe_from_abundances`). Checked against
  APOGEE/GALAH/ASPCAP (609 stars): offset +0.065, scatter 0.058; dropping Si
  (which has a long negative tail) or Ti changes the scatter by < 0.01, so all
  four are kept.
- `FEH` is the calibrated [Fe/H] (used); `FEH_RAW` has a ~0.3 dex offset and is
  used only for the grid-edge cut.
- **Cuts** (`cuts()` in the module, in order; stars left): Gaia id (155,069),
  `SNR_MED ≥ 20` (93,619; by far the largest cut), `CHI2_FLAG ≤ 3` (89,936; the
  paper's recommendation; flag 3–6 stars are clearly worse against APOGEE etc.),
  finite uncertainties (89,251), fit not stuck at its starting point (89,230),
  grid-edge cuts on Teff (82,345), logg (81,136) and raw [Fe/H] (80,476),
  [α/Fe] range (80,462), loose uncertainty limits (80,455).
- **Grid edges** (the pile-ups are large): the neural network's training grid
  spans Teff 4000–8000 K, logg 0–5, raw [Fe/H] −2.5…1.0; 14% of all stars sit at
  Teff ≤ 4010 K (23,376; 17% in the first 100 K), 3,718 are at logg = 5.0 exactly,
  557 at raw [Fe/H] = −2.5. The cuts are slightly inside: Teff in
  (4050, 7950), logg in (0.1, 4.97), raw [Fe/H] in (−2.45, 0.95). **Consequence:
  the catalog contains no stars cooler than 4050 K**, so it cannot overlap with
  M-dwarf catalogs (SLAM, MDw Cannon, ...).
- **Failed "global" fits (a finding)**: 47% of the stars have
  `HAS_GAIA_GUESS = False` (no Gaia XP spectrum) and were fit with a coarse grid
  search (8×5×7 nodes in Teff × logg × [Fe/H]) followed by a local polish. The
  3,140 stars with NaN uncertainties are fits where the polish never left the
  starting node: micro-turbulence is exactly 2.595 (its fixed starting value) and
  Teff/logg/[Fe/H] are grid-node values (logg = 2.505, 3.752, 5.0, ...),
  visible as narrow spikes in the histograms (`grid_edges.png`). The
  finite-uncertainty cut removes them; an explicit cut removes the remaining
  few. Other no-guess stars are kept (the paper reports comparable quality) but
  are a different population (fainter, more dwarfs, higher [α/Fe], lower
  [Fe/H]); `REQUIRE_GAIA_GUESS = True` drops them.
- Micro-turbulence has a floor (0.75 km/s) on which ~6% of the selected stars
  sit, but
  they follow the main sequence and the [Fe/H]–[α/Fe] relation normally, so
  **no cut is applied** (a judgement call: with only 27 matched stars the
  comparison surveys cannot say).
- Reported uncertainties are small relative to the real scatter against other
  surveys (median TEFF_ERR 33 K, LOGG_ERR 0.09, FEH_ERR 0.04, [α/Fe] err 0.03,
  vs a DESI−APOGEE/GALAH/ASPCAP scatter of ~0.11 in 5 ln Teff, 0.115 in [Fe/H],
  0.28 in logg, 0.059 in [α/Fe] — which includes the partner's errors), and
  there are offsets (Nagaraj − others: +0.05 in 5 ln Teff, +0.06 in logg, +0.065
  in [α/Fe], ~0 in [Fe/H]).
- Regularization errors are a first guess, `X_ERR = (0.1, 0.1, 0.1, 0.2)`.
- 21 duplicate Gaia ids remain (the project policy is not to dedupe).

**Overlaps** (`python -m astrotune.prep.overlaps desi_nagaraj2026`, figure
`diagnostic_plots/desi_nagaraj2026/overlaps.png`; unique Gaia ids shared; "cool"
= Teff < 4600 K): DESI DR1 34,462 (2,790 cool), LAMOST DD-Payne 4,946 (257),
DESI (Zhang) 3,006 (175), LAMOST MRS 640 (18), SDSS-V BOSSNet 459 (54),
APOGEE 400 (40), SDSS-V ASPCAP 171 (9), LAMOST HotPayne 38, GALAH 38, M-Dwarfs
(Zhang) 6, M-Stars (Ding) 4, RAVE 2, and **0 with VMP, PIGS, SLAM and MDw
Cannon**. Only 43% of Nagaraj's stars are in the DESI DR1 files used by
`desi_dr1` (`rvpix-main-*`; the cuts of `desi_dr1` hardly matter: 34,513 before,
34,462 after); the paper used the merged `mwsall` catalog covering all DESI
programs, which is the likely reason (not verified). The 34,462 stars shared
with DESI DR1 are the *same spectra* analysed by a different pipeline, so those
pairs are not independent measurements. **Implication for the SLAM/MDC/DESI
[α/Fe] problem (§9.7): this catalog cannot help the M-dwarf regime at all** (it
has no stars < 4050 K and zero overlap with SLAM/MDC); at most it adds K-star
anchors, a few tens of cool stars in common with APOGEE and ASPCAP in the
partial release.

**`sdssv_clam` — SDSS-V (CLAM)**, Medan et al. 2026 (arXiv 2607.22822), the
BOSS-CLAM value-added catalog of DR20 (forward-model pipeline on BOSS spectra,
labels from ASPCAP, BOSS-MINESweeper, wide binaries and a hot-star sample).
Raw file: `data/raw/sdssv_clam/boss_clam_lite-1.0.0.h5` (506 MB, **public: no
SDSS login**, so the module uses the plain downloader). Prepared table:
`data/training/sdssv_clam/` (**894,216 stars**, 652 duplicate-id rows kept).
Choices, all in the module docstring:

- One row per BOSS spectrum (1,708,214), M dwarfs to OB stars, Gaia DR3 ids
  (1,370 missing), `snr` in `meta/`. The file already requires SNR ≥ 10 (the
  minimum is exactly 10). Parameters: Teff, logg, [Fe/H], **[α/M]**; as for
  APOGEE and SDSS-V ASPCAP, [α/M] is used as the 4th parameter ([α/Fe]); there is
  no [M/H] column to tell them apart.
- **Uncertainties** are the square roots of the diagonal of
  `param_covariance` (order teff, logg, fe_h, alpha_m; ~0.1% have a non-positive
  variance, i.e. NaN, and are cut). **They are too small**: the scatter against
  APOGEE/GALAH/ASPCAP at SNR 10–30 is 5–8× the quoted error (3–5× once the
  systematic floor measured at SNR > 50 is subtracted), and the paper's own
  wide-binary test gives σ[Fe/H] ≈ 0.15 and σ[α/M] ≈ 0.06 at SNR = 10 against
  0.04 and 0.019 quoted (3–4×). They are therefore **multiplied by 4** (`UNC_SCALES`, in the style of
  `sdssv_slam`'s `unc_transf`) before the default floors are added; this keeps
  the SNR dependence. A judgement call; the Nagaraj catalog is *not* scaled (no
  such external evidence). Raw scatter against other catalogs (~0.1 in
  5 ln Teff, 0.09 in [Fe/H], 0.2 in logg, 0.057 in [α/M] even at high SNR) is
  dominated by systematics between catalogs, which is what the bijector is for,
  so `sigma_x` is deliberately not inflated to that scatter.
- **Flags** (owner's rules): `flag_bad` (IN_WD, IN_YSO, BAD_SPEC_FIT,
  LOW_MASS_SUSPECT and the correlation flags; **43% of all rows**) removes the
  star. `flag_warn` (IN_OB or FE_H_ALPHA_SUSPECT; 6.5% of the final table, of
  which 26,747 have Teff ≤ 6500 K and the rest are hotter; *every* star above
  6500 K has it) keeps the star but marks its [α/M] as **not measured**
  (`x = 0`, `sigma_x = x_err = 1e9`; `finalize()` in the module), the same
  per-element convention as the dummy axis of `sdssv_mdc`. The loss handles this
  element-wise, and the new `_measured` helper in `prep/plots.py` keeps such
  placeholders out of the figures. A warned star with a bad [α/M] is *not*
  rejected for it. **Caveat**: the paper says that for these stars [Fe/H] is not
  trustworthy either. At the owner's request [Fe/H] is kept (`WARN_IGNORES_FEH =
  False`; set it to True to drop it too). Evidence is mixed: warn-only stars
  (137 matched) agree fine with APOGEE/GALAH/ASPCAP ([Fe/H] +0.015 ± 0.12), but at
  6500–8000 K the [Fe/H] offset is +0.6 against DESI DR1 and BOSSNet (and −0.01
  to −0.14 against APOGEE and MRS, on tens of stars), and the Teff offset
  +7–15%; the other catalogs are themselves uncertain there.
- **Cuts** (in order; stars left): Gaia id (1,706,844), `flag_bad` clear
  (976,712), SNR ≥ 15 (898,013), Teff > 3400 K (897,811), logg in (0, 5.3)
  (897,771), [Fe/H] in (−2.5, 0.8) (896,867), [α/M] in (−0.35, 0.75) for
  non-warned stars (896,754), finite uncertainties below limits (894,216).
  - **SNR**: the scatter against the references is flat above SNR ≈ 30 and grows
    to ≈ 1.7× at SNR 10–20 ([Fe/H]: 0.22, 0.18, 0.15, 0.09 in the bins 10–15,
    15–20, 20–30, 30–50). SNR 15 was chosen over 20 because 20 removes a third
    of the cool DESI anchors (4,958 → 3,240 below 4000 K) and 14% of the SLAM
    ones; with the scaled errors the noisy stars get little weight anyway.
  - **Cool edge** (a judgement call): the sample falls off steeply below
    ≈ 3500 K (peak at 3620 K), and agreement degrades towards the edge (the
    [α/M] offset: ≈ 0 at 3600–3700 K, +0.03, +0.07, +0.11 below 3400 K). That
    trend is measured in bins of CLAM's *own* noisy Teff at a steeply falling
    edge, so part of it is a selection effect. The first version cut at 3500 K;
    **3400 K** keeps ≈ 530 SLAM anchors in 3400–3500 K (SLAM lies at
    Teff ≈ 3200–3700 K, median 3465 K).
  - **No hot cut** (owner's choice): `flag_bad`/`flag_warn` and the error
    limits handle hot stars (max Teff in the table: 19,752 K; 6,990 stars above
    8000 K). **Known issue:** above ≈ 8000 K logg is pinned near 4.1 (stripes in
    the Kiel diagram, `sample_edges.png`), so logg of the hottest stars is
    unreliable. An upper cut at 8000 K would remove 1,881 BOSSNet, 476 LAMOST
    HotPayne (more than half of the CLAM–HotPayne overlap) and 287 DESI DR1
    anchors; restore one with `TEFF_MAX`.
  - Not cut: the two telescopes differ by small offsets (APO vs LCO: Teff
    −0.04 vs 0.00 in 5 ln Teff, [Fe/H] −0.002 vs +0.013, [α/M] +0.001 vs −0.013
    against the references); and a vertical step near Teff = 4200 K in the cool
    dwarf sequence is visible in the Kiel diagram (probably a change of
    training label source; not verified).
- Regularization errors: first guess `X_ERR = (0.2,)*4` (as the other BOSS
  catalogs, SLAM 0.2 and BOSSNet 0.3), increased for Teff and [Fe/H] by 0.2 per
  1000 K above 6000 K.

**Overlaps** (`astrotune-overlaps sdssv_clam`, figure
`diagnostic_plots/sdssv_clam/overlaps.png`: shared stars per catalog, and
heatmaps by Teff for all shared stars and for those with [α/Fe] in *both*
catalogs; "cool" = Teff < 4600 K; shared unique stars, of which with [α/Fe] in
both): **SLAM 8,298 (8,298)**, **MDw Cannon 1,475 (1,475)**, DESI DR1 41,183
(37,288; 6,609 cool), APOGEE 25,827 (25,765; 7,461 cool), ASPCAP 12,012
(11,988; 3,922 cool), LAMOST DD-Payne 47,855 (46,207), BOSSNet 166,935 (0:
BOSSNet has no [α/Fe]), LAMOST M-Stars (Ding) 5,072 and M-Dwarfs (Zhang) 3,805
(3 parameters), MRS 6,524, GALAH 5,435, DESI (Zhang) 3,384, HotPayne 835, DESI
(Nagaraj) 729, RAVE 616, PIGS 26, VMP 6. Two independence caveats: SLAM and
BOSSNet are also analyses of BOSS spectra, so those pairs are not independent
measurements; MDC, APOGEE and ASPCAP (APOGEE spectra), DESI and LAMOST are.
**Unlike Nagaraj, CLAM reaches the cool stars that matter for §9.7**: 8,296 cool
shared stars with SLAM and 1,473 with MDC with [α/Fe] in both. The `flag_bad`
cut matters for them: it removes 59% of the SLAM and half of the MDC overlap
(20,505 → 8,360 and 3,055 → 1,506 before the SNR cut). **Verified:** of the
12,145 SLAM and 1,549 MDC overlap stars it removes, `LOW_MASS_TEFF_LOGG_CORR`
(weak Teff–logg correlation for Teff < 4200 K, set on 172,591 rows) is set on 89%
and 82% (72% and 64% have *only* that flag), then `TEFF_LOGG_CORR` (12%, 5%),
`BAD_SPEC_FIT` (9%, 28%) and `IN_YSO` (7%, 7%). Relaxing `LOW_MASS_TEFF_LOGG_CORR`
alone would roughly triple the SLAM anchors if those stars turn out to be good;
that is a decision for the owner (the rule given was to exclude `flag_bad`).

**Preliminary evidence on the cool-star [α/Fe] problem** (median of CLAM [α/M]
minus the partner's [α/Fe], CLAM Teff 3500–4000 K; one number per partner, over
*different stars* for each partner; not a validated result): SLAM +0.032
(7,759 stars, scatter 0.029), MDw Cannon −0.015 (1,438), APOGEE +0.034 (1,666),
ASPCAP −0.002 (351), **DESI DR1 −0.109 (3,999, scatter 0.071)**. SLAM, MDC and
APOGEE agree within ≈ 0.05; DESI DR1 sits ≈ 0.1–0.14 above all of them (implied
DESI − SLAM = +0.14, DESI − MDC = +0.09, APOGEE − DESI = −0.14); this is
consistent with the DESI downshift of ≈ −0.14 that the first models learned
(§9.7), and suggests that SLAM and MDC are not the outliers. The same pattern
holds at 4600–6000 K (DESI DR1 −0.126, the others within ±0.03). Teff, logg and
[Fe/H] offsets at 3500–4000 K (CLAM minus partner, 5 ln Teff / [Fe/H] / logg):
SLAM +0.07 / +0.01 / +0.16, MDC −0.07 / −0.09 / n/a, APOGEE −0.22 / +0.01 /
+0.13, DESI −0.10 / +0.10 / +0.08.

**`lamost_mstar_li2025` — LAMOST M-Stars (Li)**, Li et al. 2025, ApJS 281, 58
(doi 10.3847/1538-4365/ae1586): a CNN refinement of the LAMOST **DR10** M-type
catalog giving Teff, logg, [M/H], [α/M] and RVs for dwarfs and giants. Raw file:
`data/raw/lamost_mstar_li2025/Recommended_Catalog_Upload_v1.fits` (the paper's
Table 9, 870,518 spectra, 170,648,640 bytes; NADC doi 10.12149/101668;
**public**, but slow, ≈ 30 MB/min, so the downloader gets a 180 s timeout).
Prepared table: `data/training/lamost_mstar_li2025/` (**575,864 rows, 458,365
unique stars; 117,499 duplicate-id rows kept**; 538,232 dwarfs, 37,632 giants).

- **The TODO.md URL was the wrong file** (found by comparing the downloaded
  columns with the paper's Table 9). `dr12_v2.0_LRS_mstellar.fits.gz` on
  lamost.org is LAMOST's *official* DR12 M-star catalog: 985,962 spectra, LASP
  parameters (Teff, logg, [M/H] only, `-9999` sentinels), no `parameter_flag`,
  no [α/M]. Li's catalog is DR10-based, so the key has **no `_dr12`** (agreed
  with the owner). The official DR12 catalog was *not* built; if wanted it is a
  separate candidate (three parameters, SNR columns `snru..snrz`).
- Parameters: [M/H] → the [Fe/H] slot (as the other M-star catalogs), [α/M] →
  the [α/Fe] slot (as APOGEE/ASPCAP). The CNN was **trained on APOGEE labels**:
  agreement with APOGEE is not an independent test (scatter 0.06 / 0.04 / 0.13 /
  0.025 in 5 ln Teff, [M/H], logg, [α/M]; 2–5× larger against the others), so
  **APOGEE pairs are not independent**; the LAMOST DD-Payne, Ding and Zhang
  M-star catalogs use the same LAMOST spectra. CLAM (BOSS), MDC (APOGEE
  spectra), DESI and SLAM are independent of it.
- **Cuts** (in order; rows left): Gaia id (861,454); **`parameter_flag == 1`**
  (749,638; the paper's recommended sample, which already contains its
  error-reasonableness cut and its Table 10 ranges: dM Teff 3225–4125 K, logg
  4.35–5.45, [M/H] −0.85…0.35, [α/M] −0.13…0.25; gM 3425–4075 K, −0.05…1.55,
  −1.35…0.45, −0.03…0.31); total SNR above 30 (578,859); class consistent with the
  Gaia CMD (575,864); finite uncertainties below limits (no further star).
  - **Edges:** inside the flagged sample the distributions are *truncated* by
    the flag at the Table 10 limits, not piled up against them (at most 3% of
    stars within 5% of a limit), so no edge cuts were added. The dwarf sequence
    ends sharply at logg 4.35 and Teff 4125 K for that reason.
  - **SNR:** the quadrature sum over u, g, r, i, z, with the catalog's `-9999`
    sentinel counted as 0 (a naive sum of squares would be dominated by it); for
    M stars it is ≈ 1.43 × `snr_i` (u and g are empty), so 30 ≈ the paper's
    own `snr_i > 20`. Scatter against the other catalogs, by SNR_tot < 30 /
    30–45 / 45–70 / 70–150: Teff 0.19 / 0.14 / 0.14 / 0.14, [M/H] 0.20 / 0.15 /
    0.14 / 0.13, [α/M] 0.096 / 0.089 / 0.084 / 0.078. Cost of the threshold in
    cool anchors (flagged rows with Gaia id): SNR_tot > 0 / 15 / 30 / 45:
    SLAM 4,778 / 4,493 / 3,970 / 3,415, MDC 8,812 / 8,438 / 7,779 / 7,179, DESI
    DR1 below 3800 K 22,316 / 20,132 / 16,683 / 13,564.
  - **CMD class check** (the paper's own criterion for reliable members):
    `CMD_pos == 'other'` (outside both M-star regions; Teff scatter 0.39 against
    0.14 for the rest) and dM↔gM contradictions are removed (2,995 rows); stars
    without Gaia astrometry (`CMD_pos` empty, 18%) are kept. The `orp_*` bad
    pixel fractions were examined and not used (only ≈ 8% effect).
- **Uncertainties** are the spread of ten cross-validated CNNs (internal
  precision) and too small: the scatter in excess of the high-SNR plateau (an
  estimate of the random error) is 1.3–3.4× the quoted value (Teff 1.9, [M/H]
  1.6–1.7, logg 1.3–1.6, [α/M] 2.3–3.4), so they are **multiplied by 2**
  (`UNC_SCALES`; a judgement call, to be reviewed with the other catalogs; the
  default [α/Fe] floor of 0.05 dominates the final [α/M] error anyway). Median
  quoted errors: 23 K, 0.055 in logg and [M/H], 0.016 in [α/M].
- Regularization errors `X_ERR = (0.2,)*4`: a first guess, untuned.

**Overlaps** (`astrotune-overlaps lamost_mstar_li2025`; unique shared stars,
all of them cool because Li's Teff is ≤ 4125 K; with [α/Fe] in both in
parentheses): **DESI DR1 28,006 (28,006), CLAM\* 7,500 (7,497)**, DD-Payne
6,850, APOGEE 5,576, **MDw Cannon 4,493**, ASPCAP 4,133, **SLAM 2,762**, MRS
1,628, GALAH 230, BOSSNet 590 (0: no [α/Fe]), and the same-spectra
LAMOST M-star catalogs Ding 226,079 and Zhang 210,954 (3 parameters, so no
[α/Fe]). **CLAM and Li share no spectra**, so that pair is the cleanest bridge
(7,500 stars). Li is the largest source of cool [α/Fe] anchors against DESI.

**Preliminary evidence on the cool-star [α/Fe] problem, second opinion** (median
of Li [α/M] minus the partner's [α/Fe]; Li Teff 3500–4000 K; one number per
partner over different stars): SLAM +0.031 (3,590 rows, scatter 0.035), MDC
−0.039 (6,623), **DESI DR1 −0.078 (27,299, scatter 0.070)**, APOGEE +0.006
(trained on it), ASPCAP +0.034, DD-Payne −0.026, **CLAM −0.007 (8,661, scatter
0.034)**. The independent CLAM-based numbers (above: SLAM +0.032, MDC −0.015,
DESI −0.109, APOGEE +0.034) agree within ≈ 0.03–0.06. Both put DESI DR1 high
(implied DESI − SLAM +0.11 via Li, +0.14 via CLAM; APOGEE − DESI −0.08 /
−0.14) and SLAM ≈ APOGEE (±0.03); MDC sits ≈ 0.05 above APOGEE and SLAM
(SLAM − MDC −0.07 / −0.05). Li − DESI at 4000–4130 K is −0.146, i.e. the DESI
excess grows with Teff. Also: against SLAM/MDC/ASPCAP Li's Teff is high by
0.19–0.24 in 5 ln Teff ([M/H] low by 0.15–0.21 vs SLAM/MDC) while it agrees with
APOGEE DR17 to 0.004: APOGEE DR17 and SDSS-V ASPCAP differ by ≈ 4% in Teff for
these cool stars, which is not a Li problem.

---

### 5.6 How to add a new catalog

Adding a catalog is a two-stage process. **Stage A** (prepare) is cheap, safe
and reversible; do it first and look at the results. **Stage B** (promote into
the model) changes the shape of the model and invalidates old checkpoints, so
do it only when the project owner has agreed that the catalog is worth it
(§5.5 shows an example where it was not obviously worth it: the Nagaraj catalog
cannot overlap the cool stars it was hoped to constrain).

#### Stage A — prepare the catalog (does not touch training)

1. **Read the paper / README / FITS header first.** Find: the Gaia DR3 source id
   (or RA/Dec for a crossmatch), Teff, [Fe/H] (calibrated or raw? use the
   calibrated one), logg, [α/Fe] (or the individual abundances to average),
   each with its uncertainty, and every quality indicator (SNR, χ², flags,
   "recommended cuts"). Note the edges of the grid the pipeline was trained
   on, units and dtypes (Teff can be an integer column), and sentinel values
   (`-1` ids, `-9999` errors, `1e9`, `NaN`). Masked columns must go through
   `prep.transforms.unmask`. **Check that the file you downloaded is the
   product you think it is**: compare its columns with the paper's table of
   fields before building anything (the `lamost_mstar_li2025` TODO URL pointed
   at a different catalog, §5.5), and watch for sentinels in *every* column you
   combine (a `-9999` SNR squared into a quadrature sum dominates it).
2. **Write `astrotune/catalogs/<key>.py`** following the module contract (§5.1).
   Copy the closest existing module:

   | Situation | Template |
   |---|---|
   | FITS with Gaia ids | `apogee_dr17`, `galah_dr4`, `desi_nagaraj2026` |
   | [α/Fe] must be built from abundances | `galah_dr4`, `desi_nagaraj2026` (`calc_alpha_Fe_from_abundances`) |
   | no Gaia id: crossmatch on RA/Dec (batched, cached in `data/interim/<key>/`) | `lamost_ddp_dr9`, `rave_dr6` (`prep.gaia.cached_crossmatch`, `attach_gdr3`) |
   | several files to stack | `desi_dr1` |
   | no [α/Fe] (3 parameters) | `lamost_mstar_ding2022`, `pristine_pigs` |
   | a parameter in the *middle* is missing | `sdssv_mdc` (dummy axis in `finalize()`) |
   | a parameter is missing only for *some stars* (flagged unreliable) | `sdssv_clam` (`finalize()` with a `flag_warn` extra column) |
   | HDF5 file; uncertainties from a covariance matrix; uncertainties that need scaling | `sdssv_clam` |
   | SNR in several bands with sentinels; a flag that already embeds range cuts; dwarfs and giants in one file | `lamost_mstar_li2025` |
   | downloads needing a login | `sdssv_aspcap` (`download_sdss`, `SDSS_USER`/`SDSS_PASSWORD`) |
   | uncertainties known to be over-/under-estimated | `sdssv_slam` (`unc_transf`) |

   Rules that are easy to get wrong:
   - The parameter order is fixed: `(5 ln(Teff/5000 K), [Fe/H], logg, [α/Fe])`.
     `N_PARAMS = 3` is only for a catalog **lacking the last parameter**,
     [α/Fe]; it is padded to 4 later, at the pair stage. A missing parameter in
     any other position needs an explicit dummy axis (`x = 0`,
     `sigma_x = x_err = 1e9`), as in `sdssv_mdc`.
   - `ID_COL` must hold Gaia DR3 source ids; rows with `id <= 0` are dropped
     automatically. Duplicate ids are **kept** (§5.4); the validator only warns.
   - `sigma_x` (statistical errors, drives the pair χ²) gets the default error
     floors `(0.01, 0.01, 0.04, 0.05)` in transformed units added in quadrature,
     unless the module sets `ERROR_FLOORS`. `x_err` (from `param_errs`) is the
     *regularization* scale: how far the transformation may move each parameter
     for this catalog. Start from 0.1 and make it larger where the catalog is
     less trustworthy (hot stars, cool edge, ...). They are not
     interchangeable (§4).
   - Document in the module docstring: the reference, any caveat (partial
     release, calibration, fit types), and **why each cut was chosen**.
3. **Register it as a candidate** in `registry.CANDIDATES` (key, label, module
   path, `n_params`, `legacy_file=""`). Do *not* add it to `CATALOGS` yet.
   `astrotune-prepare --list` then shows it as "candidate".
4. **Choose the cuts the way the other catalogs did.** Make a Kiel diagram
   (Teff–logg) and an [Fe/H]–[α/Fe] diagram, add cuts cumulatively
   (`cuts()` returns an ordered dict of named masks; the per-cut counts end up
   in the sidecar and in `cut_flow.png`) and watch what each cut does. The
   usual set:
   - SNR ≳ 20, and the catalog's own recommended quality flags/χ² cuts;
   - finite uncertainties, and loose upper limits on them;
   - extreme parameter values;
   - **pile-ups at the edges of the pipeline's parameter grid**: histogram
     every parameter in fine bins; if there is a spike at a boundary, cut a
     little inside it;
   - **spikes at repeated exact values**: `np.unique(column, return_counts=True)`.
     Failed fits often stay at their starting point (this is how the Nagaraj
     fits that never left the grid-search node were found: micro-turbulence
     exactly 2.595 on 3,387 stars);
   - check sub-populations (e.g. different fit types) in the same plots.

   To judge a cut quantitatively, join the candidate with APOGEE, GALAH and
   ASPCAP on `id` (their prepared tables are in `data/training/`) and compare
   the `x` columns in bins of SNR / flag / fit type: robust scatter
   (1.4826·MAD) and median offset. This only has power where there are enough
   overlap stars; with a small catalog it cannot tune the SNR cut, say so.
5. **Build it:** `astrotune-prepare <key>` (downloads into `data/raw/<key>/`
   with the cached downloader; writes `data/training/<key>/<key>.h5` + sidecar
   and diagnostics to `diagnostic_plots/<key>/`). Add catalog-specific figures
   to `diagnostics()` that justify the cuts (e.g. `grid_edges.png`).
6. **Measure the overlaps:** `astrotune-overlaps <key>` (§5.5). A catalog is
   only constrained where it shares stars with others, **and the parameter
   regime matters**: check the overlap *by Teff* with the catalogs you hoped to
   tie it to. A catalog that is the same spectra as an existing one (e.g. two
   DESI pipelines) gives non-independent pairs.
7. **Add tests** (`tests/test_<key>.py`, see `tests/test_nagaraj.py`): the
   candidate/registry status, and `cuts()` on a few synthetic rows where each
   row violates exactly one cut. The generic contract tests in
   `tests/test_parity.py` run automatically once the table exists (the legacy
   comparison is skipped when there is no legacy table). Run
   `pytest -q tests --ignore=tests/test_training.py`.
8. **Write down the findings** in §5.5 of this file (what you found, the cut
   counts, the overlaps) and a line in the "what remains" list if relevant.

#### Stage B — promote it into the model (retraining required)

Only after the owner agrees. Mechanically: move the spec from
`registry.CANDIDATES` to the **end** of `registry.CATALOGS` (never insert or
reorder: the position is the category index, §5.2); the prepared table from
Stage A is reused as is. The consequences, all verified in the code:

- **The model changes shape; old checkpoints cannot be warm-started.** The number
  of categories is `len(registry.CATALOGS)`, and the embedding dimension is
  `ceil(log2(n_cat))`: going from 16 to 17 catalogs changes it from 4 to 5, so
  the embedding table and every FiLM layer have different shapes. Retrain from
  scratch.
- **Old runs keep loading** (fixed 2026-11-08): `load_bijector` uses
  `registry.run_keys(run_dir)`, i.e. the catalog list saved in that run's
  `config.json`; `evaluate`/`videos`/`train` take the number of categories from
  the pairs file they are given. Always pass the run's own `--pairs-path`.
- **Rebuild the pairs file:** `astrotune-pairs` (all catalogs, shared stateful
  RNG: the pairs of the *old* catalogs are re-randomised too, so the file no
  longer matches the legacy `training_data/pairs_and_ctrl_points.h5`; the
  parity statement in §9.1 holds only for the 16-catalog build). Back up the old
  `data/pairs_and_ctrl_points.h5` first: `--resume` of an old run requires the
  same file. Padding of a missing [α/Fe] and the control-point sampling are
  automatic. Mind the duplicate-id policy (§5.4): it multiplies pairs.
- **Pairs and training time change** (new pairs, control points; the epoch length
  is set by the smaller of the two datasets).
- **Figures change:** `training_plots.catalog_colors(n)` depends on `n`, so every
  catalog gets a different color; labels and videos pick the new catalog up
  automatically (its index is the next free integer).
- **Update the documentation:** the catalog table in §5.2 (add the new row),
  every "16" in this file and in `README.md`, the example results in §9.4.
- **Evaluate before believing it:** train, then run `astrotune-evaluate` *and*
  the per-pair checks of §9.7. A new catalog can make previously good catalogs
  worse (that is how the cool-star problem was found), and a catalog with no
  overlap in some regime is only regularised there.

---

## 6. The method as implemented in the notebook

Read §2 of the paper first; this is the mapping to the **notebook** code. §9.3
lists exactly where `astrotune/train.py` deviates from it (the learned error
model is not ported, among other things).

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
   **(Not ported: the error model and the importance-sampling term are
   skipped, §9.3.)**
3. Penalties: path length (`l1_path`, `l2_path`) on pairs and control points;
   distance of control points from identity normalised by `xe_ctrl`;
   divergence `weight_div * log((s/s')² + eps)²`; weight l2 (`l2_weights`).
4. Adam with `PiecewiseConstantDecay` halving `1e-3` eight times; global-norm
   clipping at 100; 32 epochs, batch 1024, `val_frac = 1/8` (validation = the
   **first** `n_val` rows).
5. Late in training the embedding and FiLM variables are frozen
   (`var_freeze_list`) — **intended, but it never took effect** (the loop assigns
   to `freeze_list`, a different name, so the published run trained everything
   throughout). Checkpoints: `tf.train.Checkpoint(bij=bij)` into
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

## 9. The training layer

The training layer is ported and works end to end: it trains a bijector on the
real data, the catalogs measurably agree better afterwards, and runs can be
resumed. It is being tuned (§9.4, §9.7). It was ported in this order, mirroring
the catalog layer:

| Code | Notebook cells | What |
|---|---|---|
| `astrotune/model/vector_network.py` | 4–5 | `VectorNetwork` |
| `astrotune/model/ode_bijector.py` | 6–7 | `ODEBijector` (`CatMLP` dropped: unused) |
| `astrotune/model/io.py` | — | `load_bijector(run_dir)`: rebuild the full bijector from a run's checkpoint (the SavedModel only exposes `to_harmonized`/`from_harmonized`, not `return_path`) |
| `astrotune/training_data.py` | 346–361 | pairs + control points, save/load |
| `astrotune/train.py` | 374–417 | `TrainConfig`, `Trainer` (loss, loop, resume, save) |
| `astrotune/evaluate.py` | (new) | agreement before/after harmonization |
| `astrotune/training_plots.py` | 378, 382–383 | figures written during training |
| `astrotune/videos.py` | 424–448 | Kiel and [Fe/H]–[α/Fe] videos |
| `astrotune/apply.py`, `astrotune/merge.py` | 461–488 | transform catalogs, merge by Gaia id (§9.9) |
| `astrotune/diagnostics.py` | 419–453, 456–460, 526–544 | **mostly not ported** |

### 9.1 Data for training

`data/pairs_and_ctrl_points.h5` (`config.PAIRS_PATH`), built by
`astrotune-pairs` with `dedupe="none"` and seed 3. Built with
`--dispersed-xerr uniform` and the legacy SLAM `x_err` it reproduces the legacy
`training_data/pairs_and_ctrl_points.h5`: **verified** identical shapes
(2,151,614 pairs; 1,777,022 control points), identical categories, `x` equal to
float32 round-off (max 1.9e-6), and `xs` differing only in `[alpha/Fe]` for
`lamost_hotpayne` (the intended sigma fix, §5.3). The file is stamped with the
options it was made with (`seed`, `dedupe`, `dispersed_xerr`, catalog list; read
with `training_data.read_attrs`), and each training run's `config.json` records
them (`pairs_attrs`). Default is still the legacy
many-to-many join on duplicate Gaia ids (§5.4); `--dedupe first` keeps one row
per id per catalog. This question is still open (§10).

**Two pairs files exist (2026-11-08).** `data/pairs_and_ctrl_points.h5` is the
one the first runs (`test30`, `turnover50`, `turnover50_8ep`) used: dispersed
control points at the legacy uniform `x_err = 0.2` and SLAM's [α/Fe] at 0.2.
`data/pairs_per_catalog_xerr.h5` has per-catalog dispersed errors (§9.3) and SLAM's
0.4; it is **bit-identical in every position, noise scale and category** (verified),
and differs only in those `x_err`. Do not overwrite the first while a run may be
resumed: `--resume` only checks that the sizes match, so it would silently change
the run. Pass `--pairs-path` to train on the second.

`load_training_data(fix_unknown_errors=True)` applies the notebook's post-load
fix-up (`xe > 1e6 → 0.05`). Pass `False` when you need to tell which parameters
a catalog measures (`xe_control >= 1e6`); `evaluate.py`/`videos.py` rely on it.

### 9.2 How to run

```bash
.venv/bin/pip install -e .          # registers the astrotune-* commands
.venv/bin/python -m astrotune.training_data                   # build the pairs file
.venv/bin/python -m astrotune.train --run-name my_run         # train
.venv/bin/python -m astrotune.train --run-name my_run --resume   # continue it
.venv/bin/python -m astrotune.evaluate --run-dir models/my_run
.venv/bin/python -m astrotune.videos   --run-dir models/my_run
```

Every `TrainConfig` field is a command-line option (`--chi2-turnover 50`,
`--n-epochs 8`, `--val-every 4`, `--plot-every-seconds 300`, ...); the defaults
are the notebook's hyperparameters except where §9.3 says otherwise. The
important ones: `--n-epochs` (32 in the paper; one epoch = 1,518 steps because
the control-point set limits it), `--val-every N` (one validation batch every N
steps: 1 as in the notebook, 4 is much cheaper), `--max-steps` (testing).

A run writes `models/<run-name>/`:

| Path | Content |
|---|---|
| `config.json` | the `TrainConfig`, data sizes, catalog keys, versions, start/resume times |
| `progress/progress_0001.png, ...` | a figure every 5 min (loss history; catalogs before/after in Teff–logg and [Fe/H]–[α/Fe]; the title shows the held-out agreement summary) |
| `checkpoints/` | `ckpt-<step>` (model) + `state-<step>.npz` (optimizer state, metrics, figure counter); latest 3 kept |
| `bijector/` | SavedModel with `to_harmonized(x, c)` / `from_harmonized(x, c)` |
| `metrics_history.json`, `agreement_history.json` | per-step loss/penalties; agreement at each figure |
| `evaluation/` | final before/after agreement (see §9.5) |
| `videos/` | made with `astrotune.videos` (not automatic) |

**Resume.** State is saved at every progress figure, every 10 min
(`--checkpoint-every-seconds`), and on Ctrl-C/SIGTERM (the handler saves, logs
the resume command, re-raises). `--resume` reuses the saved config; only the
plotting/checkpoint intervals can be changed (`RESUME_MUTABLE` in `train.py`),
and the pairs file must be the same size. Starting a new run in a directory that
already has checkpoints raises `FileExistsError`. **Verified** by killing a real
run with SIGTERM at step 75 and resuming to 120. *Not* verified: that the
data order after resume is bit-identical to an uninterrupted run (the datasets
are seeded and fast-forwarded with `.skip()`, which should do it).

**Speed.** About 1.0–1.3 s/step alone on the A100 (batch 1024, ODE solver at
rtol = atol = 1e-5), i.e. ~25 min/epoch. The GPU is shared with other jobs
and with analysis scripts; observed 1.7 s/step under contention. Use
`--val-every 4` for long runs.

### 9.3 What `train.py` does, and where it deviates from the notebook

Loss as implemented (per step): add Gaussian noise to `x0` (scale
`clip(xs0, 1e-5, 1)`); map `x0` forward with catalog `c0`, then backward with
`c1`; pair term = log-space chi² of the result against `x1` with the reported
`sigma_x`, softened to `T*arcsinh(chi²/T)`; plus control-point penalties (path
length, distance from identity normalised by `xe_ctrl`, divergence) and weight
decay. Adam, learning rate 1e-3 halved at seven evenly spaced steps, global-norm
clip 100, batch 1024, validation = the first 1/8 of rows.

Deviations (each is declared in code next to the thing that deviates):

0. **The dispersed control points use a per-catalog `x_err`, not the notebook's
   uniform 0.2** (`astrotune-pairs --dispersed-xerr per-catalog`, the default;
   `uniform` gives the legacy file). The dispersed points (143,440, 8.1% of the
   control points, one wide Gaussian shared by all catalogs, one copy per catalog)
   are the prior where a catalog has *no data*. The uniform 0.2 disagreed with the
   same catalog's real control points in both directions: [α/Fe] 0.05 for APOGEE,
   GALAH and ASPCAP (4× too loose), 0.4–0.5 for DESI DR1, RAVE and HotPayne
   (too tight), and 0.2 instead of the strong pin on parameters a catalog does
   not measure. Now each catalog gets the **median measured `x_err` per
   parameter** (or its module's `X_ERR_DISPERSED`); a parameter nobody measures
   is stored as 1e9 and pinned at 0.05 by the loader, as for real points. It is
   deliberately a *constant*, not the catalog's functional form (an alternative
   the owner raised): those forms (hot-star penalties, edge distances) are only
   meaningful where the catalog has data, where the real control points already
   carry them, and extrapolating them would loosen the prior exactly where
   nothing constrains the map. Positions are unchanged (the RNG sequence is
   untouched). **Not yet compared empirically** (§10).

1. **The learned error model (`ln_err_model`) and the importance-sampling term
   are not ported** (requested by the project owner: it did not seem to work
   well). The chi² uses the reported `sigma_x` directly.
2. **`chi2_turnover` default is 50, not 15.** With reported uncertainties the
   mean chi² is ~16, so a turnover of 15 flattens a large part of the
   distribution. (The owner said the earlier *uniform-uncertainty* approach
   "worked better" by eye and does not want to give up on reported
   uncertainties yet; this is being explored, §9.4.)
3. **Pair path-length penalty is reported but not in the loss**, because the
   notebook's active loss leaves it out (the paper describes it as part of the
   loss). `--pair-path-in-loss` adds it. Default: off.
4. **Embedding/FiLM freezing late in training is off by default**, because it
   never took effect in the notebook (§6). `--freeze-after-frac 0.75` turns it
   on (`--freeze-vars` chooses which variable names). Default: off.
5. `1e-30` is added inside the log of the squared residual (avoids NaN
   gradients when a residual is exactly 0).
6. The notebook's code bugs (§ "Bugs in the notebook", below) are not copied.

**Paper vs. notebook code** (the code is what was ported, and what the default
reproduces): the paper (§2) says batches of 2048 pairs + 2048 control points,
a network of six hidden layers of 256 units with concatenated one-hot
conditioning, and a learning rate stepping 1e-3 → 1e-4 → 1e-5. The notebook
code (cell 412) uses batch 1024, three hidden layers of 64 units with FiLM
conditioning on a learned embedding, and halves the rate seven times. The paper
text should be updated, or the code changed, so they agree.

### 9.4 Training runs so far

All with the defaults except as noted; `val-every 4`; 1 epoch = 1,518 steps.
Agreement is on held-out pairs (200k sample), averaged over catalog pairs
(47/47/44/27 pairs for Teff/[Fe/H]/logg/[α/Fe]). "|offset|" is the median of
catalog_j − catalog_k, "scatter" is 1.4826·MAD; before → after harmonization.

| Run | Settings | Teff | [Fe/H] | logg | [α/Fe] |
|---|---|---|---|---|---|
| `models/test30` | turnover 15, 1 epoch (24 min) | \|off\| 0.108→0.038 · σ 0.094→0.089 | 0.072→0.039 · 0.132→0.124 | 0.115→0.054 · 0.162→0.148 | 0.115→0.015 · 0.112→0.102 |
| `models/turnover50` | turnover 50, 1 epoch | 0.108→0.036 · 0.094→0.089 | 0.072→0.038 · 0.132→0.123 | 0.115→0.049 · 0.162→0.147 | 0.115→0.017 · 0.112→0.097 |
| `models/turnover50_8ep_19cat` | turnover 50, 8 epochs (19,766 steps), **19 catalogs**, `data/pairs_19cat.h5` (3,585,330 pairs, 2,891,485 control points; per-catalog dispersed `x_err`, SLAM [α/Fe] x_err 0.4) | *in progress* (started 2026-11-08 07:05; ~1.5 s/step with the other run, ETA ~8 h). Not directly comparable with the rows above: different catalogs and pairs, and the dispersed-`x_err` change is confounded with it. | | | |
| `models/turnover50_8ep` | turnover 50, 8 epochs (12,147 steps) | *in progress* (started 2026-11-08 ~02:00; ~1.7 s/step under GPU contention, ETA several hours) | | | |

Reading this: **the surveys do come into alignment** (the main acceptance test
the owner asked for): systematic offsets shrink 2–8×; scatter barely changes
(it is mostly measurement noise, which a smooth map cannot remove). Turnover 50
is marginally better than 15 on aggregate and reduces the SLAM regression below
by a third to a half without removing it. The new model does not need to match
the legacy `models/bijector`; it has not been compared with it.

Videos for these runs are in `models/<run>/videos/` (§9.5).

### 9.5 Evaluating a run (`astrotune-evaluate`, `astrotune-videos`)

`evaluate.py` runs automatically at the end of training and writes
`models/<run>/evaluation/`: `agreement.json` (offset and scatter per catalog
pair and parameter), `agreement_summary.png` (one point per catalog pair; below
the diagonal = improved), `pairs/pair_JJ_KK.png` (catalog-vs-catalog, original
and harmonized, for the 8 best-populated pairs), `param_shifts.png` (`f(x) − x`
per catalog).

`videos.py` writes, per plane (`kiel` = Teff–logg, `feh_alpha` = [Fe/H]–[α/Fe]):
`<plane>.mp4`, `<plane>_control_10.mp4` (SDSS-V ASPCAP highlighted),
`<plane>_pair_3_10.mp4` (the stars shared by GALAH DR4 and ASPCAP, as measured
by each; the two sets of points should converge). Options `--plane`,
`--highlight KEY`, `--pair KEY KEY`, `--no-plain`. 90 frames, 30 fps, 1600×1600,
same ffmpeg settings as the notebook. Stars whose catalog does not measure a
plotted parameter are not drawn (this removes a spurious line at logg = 0 from
the MDw Cannon dummy axis that the first version of the Kiel videos showed).

The notebook's `param1v1` / `param1v1_diff` videos (`gen_param1v1_video`: a 2×2
grid, one panel per parameter, catalog j on the horizontal axis, catalog k (or
k − j) on the vertical axis, for one catalog pair) are `--plane param1v1` and
`--plane param1v1_diff`; they only exist for pairs (`--pair KEY KEY` or
`--all-pairs-min N`; the notebook made the diff version for every pair with
≥ 50 pairs). Differences from the notebook: panels use only stars for which
*both* catalogs measure that parameter (and say so if there are none); axis
limits are 0.5–99.5 percentiles over all frames plus a margin, rather than the
first frame's full range, so outliers do not set the scale; the Teff axis is
labelled in 1000 K with a tick spacing chosen from the range (the notebook's
fixed 5/10/20/30 ticks left cool-star panels unlabelled).
Trajectories come from `load_bijector`, i.e. from the latest checkpoint.

**Pitfall: `evaluate.py` hides small catalog pairs.** It only reports catalog
pairs with ≥ 100 *validation* pairs (`min_pairs`), so pairs of a few hundred
stars in total (the M-dwarf pairs, §9.7) never appear in `agreement.json`, and
the aggregate can look uniformly good while a catalog gets worse. Also, in the
videos the gray background does not say *which* catalog is which; use
`--highlight` to find a culprit.

### 9.6 Missing parameters: [α/Fe] (and the MDC dummy logg)

Audit requested by the owner. **Verdict: no outright bug found; one
train/evaluation inconsistency, measured to be small.** Details:

**Which stars lack [α/Fe] (verified).** Only the five 3-parameter catalogs
(LAMOST M-Stars (Ding), VMP (Li), LAMOST M-Dwarfs (Zhang), SDSS-V BOSSNet,
PIGS). They are padded to 4 parameters in the pairs/control-point stage with
`x = 0`, `sigma_x = xe = 1e9`. **No 4-parameter catalog has any star with a
missing or zero-filled [α/Fe]** (checked on all 11: `sigma_x[:,3] ≥ 1e6`
fraction 0, `x[:,3] == 0` fraction 0, `x_err[:,3] ≥ 1e6` fraction 0).
SDSS-V MDw Cannon is a 4-parameter catalog with a *dummy logg* (index 2): the
same padding values on that axis, `HAS_LOGG = False`. The same code paths
handle both.

**How the loss treats unknown parameters (checked in `train.py`):**
- Pair chi²: if `x0`'s parameter is unknown then `xs1` for it is set to 1e9
  (`tf.where(xs0 > 1e6, xs0, xs1)`); if `x1`'s is unknown it is already 1e9.
  Either way that axis enters the log-space chi² with weight ~e⁻⁴¹, i.e. it is
  excluded, as intended. So [α/Fe] is only constrained by pairs between two
  4-parameter catalogs (and only if both measure it).
- Control points: the unknown axis has `xe = 1e9`, replaced by 0.05 on load,
  which holds that axis at identity (strongly). **Verified:** for the 3-parameter
  catalogs the output [α/Fe] ≈ input (median difference −0.015…+0.055).

**The inconsistency (inherited from the notebook).** The noise injection uses
`clip(xs0, 1e-5, 1.0)`, so an *unknown* axis (`xs0 = 1e9`) gets noise scale
1.0: during training, the network is fed `[α/Fe]_in ~ N(0, 1)` for every star
from a 3-parameter catalog, while evaluation, videos and inference feed exactly
0. The same random value then flows (as the harmonized [α/Fe], since that map is
~identity) into the *inverse* map of the partner catalog. **Measured effect is
small** (`test30` and `turnover50` agree): for the 3-parameter catalogs, the
harmonized Teff/[Fe/H]/logg change by only 0.005–0.035 (std over alpha_in ~
N(0,1)) with mean shifts ≤ 0.02; moving [α/Fe]_in by +1 changes the other
outputs of 4-parameter catalogs by 0.002–0.024 (SLAM: < 0.001). This is below the
catalog-to-catalog scatter (~0.1) and below the error floors scale. If you want
training to match evaluation, make the noise zero for unknown axes (e.g.
`noise = where(xs0 > 1e6, 0, clip(xs0, 1e-5, 1))`); this is **not implemented**
and not expected to change the results materially.

**It does not explain the SLAM problem below**: SLAM and its partners DESI, MDC,
ASPCAP and APOGEE all measure [α/Fe]; the 3-parameter catalogs only touch
SLAM's other coordinates.

### 9.7 Known problem: cool-star [α/Fe] (SLAM, MDw Cannon, DESI)

Found by looking at the `test30` videos: in the [Fe/H]–[α/Fe] plane one survey
moves *away* from the others. It is **SDSS-V SLAM**: its [α/Fe] starts at
about −0.04 and ends at about −0.12 (median shift −0.078), 0.08–0.2 dex from
APOGEE/GALAH/ASPCAP in the harmonized space. The `turnover50` run has the same
behaviour with smaller amplitude.

Evidence (verified; all pairs, not only validation; "cool" = 5 ln(Teff/5000 K)
< −1.4, i.e. Teff ≲ 3800 K; median [α/Fe] offset catalog_j − catalog_k, original
→ harmonized):

| Pair | n | `test30` | `turnover50` |
|---|---|---|---|
| SLAM vs MDw Cannon | 679 | −0.074 → **−0.145** | −0.074 → −0.126 |
| SLAM vs ASPCAP | 102 | −0.007 → **+0.083** | −0.007 → +0.049 |
| SLAM vs APOGEE | 39 | −0.014 → **+0.054** | −0.014 → +0.028 |
| DESI DR1 vs MDw Cannon | 777 | −0.031 → **−0.158** | −0.031 → −0.167 |
| DESI DR1 vs SLAM | 1,410 | +0.062 → +0.004 | +0.062 → −0.024 |
| ASPCAP vs MDw Cannon | 946 | −0.094 → −0.072 | −0.094 → −0.087 |
| APOGEE vs MDw Cannon | 350 | −0.064 → −0.065 | −0.064 → −0.074 |

Interpretation (partly **hypothesis**): DESI DR1's [α/Fe] is mapped down by
~0.14 (verified: median shift −0.11 … −0.16 in every Teff bin), which is what
the many warm-star pairs with APOGEE/ASPCAP/LAMOST require. SLAM's largest
[α/Fe] anchor is DESI (1,410 stars), so SLAM is dragged down with it, whereas
the small cool-star pairs (SLAM–ASPCAP 102, SLAM–APOGEE 39, MDC–APOGEE 350)
carry almost no weight among ~2 million pairs in a batch-averaged loss, and the
network (3 × 64 units, 1 epoch) is not flexible enough, or not trained long
enough, to satisfy them all. Note the *original* offsets among DESI, SLAM and
MDC are mutually consistent to ~0.02, so this is not simply inconsistent data.
The owner's comment: the maps are **general invertible transformations of the
full 4D space**, not uniform shifts — the "−0.14 in every bin" above is a
measured median per bin, not a statement about the form of the map.

Milder regressions seen in the aggregate table (`test30`): LAMOST HotPayne
[Fe/H] against APOGEE (+0.019 → −0.148, n = 102), LAMOST MRS (−0.020 → −0.139,
n = 2,594) and DESI (−0.085 → −0.166, n = 281); and MDC [α/Fe] vs APOGEE
unchanged (−0.060 → −0.065). Not investigated further.

The owner asked to test whether this improves with (a) turnover 50 (done:
reduced, not removed) and (b) longer training (the 8-epoch run, in progress).
Later, the DESI/other-survey overlap at low temperature may improve with more
data (§10).

### 9.9 Applying the model and merging by star (`astrotune-apply`)

```
astrotune-apply --run-dir models/my_run          # -> models/my_run/harmonized/
astrotune-apply --run-dir models/my_run --only sdssv_slam sdssv_mdc   # quick test
```

`apply.py` (TensorFlow: transform, I/O, summary figure) and `merge.py` (NumPy
only: the combination, tested in `tests/test_merge.py`). Steps: (1) each
catalog's `x`, `sigma_x` go through the bijector of the catalog's category and
the covariance through its Jacobian, `C_y = J diag(σ²) Jᵀ` (first order);
(2) entries of the same Gaia DR3 id are combined as a product of Gaussians,
`C = (Σ (C_i + e²I)⁻¹)⁻¹`, `y = C Σ (C_i + e²I)⁻¹ y_i`, `e = 0.03`, then entries
with chi² > 30 against the combined estimate (notebook: 25; raised by the owner
because typical chi² values are ~15) are dropped and the combination is
repeated, 3 times (chi² is always computed for *all* entries, so a dropped entry
can come back); (3) stars with non-finite values or outside `PARAM_LIMS` are
removed. Options: `--dedupe first` (one row per id per catalog; the default keeps
repeated spectra as independent entries, as the notebook did), `--eps-err`,
`--chi2-max`, `--n-iter`. Every run caches the transformed estimates in
`<out-dir>/transform_cache_<dedupe>/` (the slow GPU step); `--reuse-transform`
reloads them (rows are checked against the tables), so merge settings can be
changed in minutes. Only valid for the same model: delete the cache or use a
different `--out-dir` if the model changes.

Output in `<run>/harmonized/`:

- `merged_catalog.h5` (table at path `data`; `apply.read_merged(path)` returns
  `(table, metadata)`): `gdr3_source_id` (sorted), `params_est` (4),
  `params_err` (4), `params_cov` (4,4), `lndet_cov`, `params_source` (uint32 bit
  mask: bit *i* = catalog *i* contributed; `merge.catalogs_in(mask, i)`),
  `n_estimates`, `chi2_max_used` (largest chi² among the entries used: large =
  the entries disagree; consider cutting), `params_measured` (4, bool).
- `individual_estimates.h5`: one table `catalog_<i>` per catalog (`params_orig`,
  `params_err_orig`, `params_harm`, `params_cov_harm`, `measured`,
  `chi2_merged`, `used`).
- `metadata.json` (also stored in the tables' metadata): schema version, run and
  full training config, parameter names and the Teff transform, the merge
  settings and counts, the rejected-fraction history, and `surveys`: for each
  catalog its **index and bit, key, label, reference**, parameters it measures,
  numbers of entries/unique stars/entries used/stars contributing (before and
  after the final cuts), rejection fractions and the provenance of its prepared
  table. `summary.png` is an overview.

**Unmeasured parameters** (catalogs without [α/Fe]; MDw Cannon logg; CLAM
`flag_warn` stars' [α/M]; anything with `sigma_x > 1e6`) are fed to the map as 0,
contribute nothing to `C_y`, and are stored as `value 0, C_pp = 1e9`, zero
off-diagonal. A merged star for which no contributing entry measured a
parameter has exactly that encoding and `params_measured = False`: **do not use
the value** (the Kiel plots must mask on `params_measured`).

**Math errors and bugs found in the notebook's merge (fixed here; the legacy
`merged_params_catalog_clean.h5` contains them):**

1. `params_err = sqrt(C_ii² + e²)`: the variance was squared. In the legacy file
   this reproduces to 1e-7 and the median error is 0.030 for all three
   parameters, i.e. the floor, although the median `sqrt(C_ii)` is 0.034 /
   0.041 / 0.078. Errors above 1 were overstated, those below 1 understated.
   Now `sqrt(C_ii + e²)`.
2. **Missing parameters inflated the whole covariance.** The notebook zeroed
   the uncertainty of a missing [α/Fe] for the transformation, but not of the
   MDw Cannon logg (`σ = 1e9`), so every MDC entry had covariance ~1e14 on all
   four parameters (verified in `transformed_data.h5`: median diag
   4e14, 1e14, 1e18, 1e14) and the catalog carried no weight in the merge.
   Handled per element now (`sigma_x > 1e6`), so MDC (and CLAM warn stars)
   contribute their measured parameters.
3. **Zero estimates.** If outlier rejection left one entry of a star that had
   several, `combine_parameters` took neither the single-entry nor the
   multi-entry branch: estimate 0, covariance 0, `lndet_cov = -inf`. The legacy
   catalog has 15 such rows. Now a single remaining entry is copied.
4. **Rejection could remove every entry** of a star (one wild entry drags the first
   mean so that all fail chi²); the star then flipped between NaN and all entries
   across iterations. The best-matching entry of each star is now never
   rejected. This changes which stars survive relative to the notebook.
5. `lndet_cov` included `ln(1e9)` for stars with an unmeasured parameter; it is
   now computed over the measured parameters.
6. `params_source` was `uint16`, which overflows at 16 catalogs (19 now);
   it is `uint32`.

**Caveats, not changed:** the combination treats the entries as independent
(repeated spectra of a star and catalogs sharing training labels, e.g. DD-Payne
and Li+2025 trained on APOGEE, are not); the covariance is the product of
Gaussians, so entries that disagree beyond their errors still give a small
covariance (use `chi2_max_used`); the `e = 0.03` floor enters the weights and is
added again, in quadrature, in `params_err` (as in the notebook), so `params_err`
is not simply `sqrt(C_ii)`. Matmuls run in TF32 on the A100, so propagated
covariances are accurate to ~1e-3 relative. Entries with a single catalog never
go through a chi² test (chi² = 0).

### 9.8 Gotchas for whoever continues

- **Number of categories is per run, not global.** `load_bijector` reads the
  catalog list from the run's `config.json` (`registry.run_keys`); `evaluate`,
  `videos` and `train` use `max(c_control)+1` of the pairs file they are given.
  The old 16-catalog runs therefore still work with
  `data/pairs_and_ctrl_points.h5`; the 19-catalog run needs
  `data/pairs_19cat.h5`.
- **Progress figures hide parameters a catalog does not measure** (they used to
  show placeholder zeros as a horizontal stripe at [α/Fe] = 0): the loader sets
  `ctrl_dict["missing_control"]` before the `x_err` fix-up, and
  `training_plots.catalog_sample_valid` turns it into a mask. Figures from before
  2026-11-08 07:00 still show the stripe.
- Catalog order is the category index and is baked into checkpoints (§5.2):
  append only.
- `config.json` is rewritten on resume with the (possibly updated) config and a
  `resumed` list of timestamps.
- Training tests: `tests/test_training.py` trains 40 real steps on synthetic data
  (≈ 2.5 min); use `-k video` or `-k "pairs or dedupe or control or save"` for
  the fast ones.
- Do not run heavy analysis on the GPU while a long run is going: it slows the
  run (1.0 → 1.7 s/step observed).

### Notes from the original port order (all done; kept for context)

1. `model/` — kept `output_scale=1e-5`, the residual/FiLM defaults and
   `DormandPrince(rtol=atol=1e-5)`. The 2D toy problem (notebook cells 8–16) was
   not ported as a test.
2. `training_data.py` — ported `extract_pairs` and the control-point sampler
   verbatim, then verified against the legacy pairs file (§9.1).
3. `train.py` — ported `grad_step` last, after the data matched.
4. `apply.py` and `merge.py` are written (§9.9); `diagnostics.py` is next. The
   legacy artefacts to compare against are `transformed_data.h5` and
   `merged_params_catalog_clean.h5` in the repo root.

## 10. What remains

In rough priority order:

0. **Compare the two pairs files** (§9.1) with identical settings: the evaluation
   summary, and the cool-star [α/Fe] pairs of §9.7, for uniform vs per-catalog
   dispersed `x_err` (and SLAM 0.2 vs 0.4). Needs the GPU, so after the 8-epoch
   run. This is the test of whether the change helps; so far it is only
   verified to be consistent.
1. **Finish the 8-epoch run** (`models/turnover50_8ep`, §9.4) and compare it with
   `test30`/`turnover50`: the aggregate agreement table, and the cool-star pairs
   of §9.7 (does the SLAM/MDC/DESI [α/Fe] regression shrink with more
   training?). Make its videos with `astrotune-videos`, including
   `--plane feh_alpha --highlight sdssv_slam --highlight sdssv_mdc`.
2. **Per-catalog regression report in `evaluate.py`** (not done; proposed and
   not yet approved by the owner). Use *all* pairs with a low threshold (≈ 30),
   also stratify by Teff, flag every catalog pair that gets worse, and print
   per-catalog summaries. This is what exposed the SLAM problem (§9.7) and the
   present report cannot show it (§9.5). Related: label the background
   catalogs in the pair videos so a reader can tell which survey is which.
3. **Modelling questions** raised by §9.7, for the owner to decide: whether the
   small cool-star pairs need more weight (e.g. per-pair-type balancing), a
   bigger network, or more data; whether to revisit the uniform-uncertainty
   chi² the owner thought worked better; whether to turn on
   `--pair-path-in-loss` / `--freeze-after-frac`.
4. **Optional:** zero noise on unknown axes in the loss so training matches
   evaluation (§9.6; small effect).
5. **`apply.py`** (cells 461–488): **done** (§9.9), but run it on the full
   catalogs with the final model (≈ 100 min for 12.9 M entries on a shared GPU).
   **`diagnostics.py`** (cells 419–453, 456–460, 526–544; the paper's figures) is
   still to do. The notebook bugs listed below must be fixed, not copied. Legacy
   artefacts to compare against: `transformed_data.h5`,
   `merged_params_catalog_clean.h5` in the repo root.
6. **New data** (from `TODO.md`; each is a new catalog module: follow the
   two-stage procedure in §5.6, then append to the registry and retrain — see the
   `load_bijector` gotcha, §9.8):
   - SDSS-V DR20 (new BOSS data, no new APOGEE data; the Astra 0.8.1 summary
     files are listed in `TODO.md`): **not done**.
   - The new `The CLAM` catalog (BOSS-CLAM, arXiv 2607.22822): **prepared as a
     candidate** (`sdssv_clam`, §5.5; 894,216 stars). Unlike Nagaraj it overlaps
     the cool-star catalogs heavily (SLAM 8,298, MDw Cannon 1,475, DESI DR1 6,609
     cool stars, all with [α/Fe] in both) and its first comparison supports DESI
     DR1 being the outlier in [α/Fe] at low temperature (§5.5). Open decisions for
     the owner before promoting it: the ×4 uncertainty scaling, whether to keep
     `LOW_MASS_TEFF_LOGG_CORR` stars (see §5.5), `WARN_IGNORES_FEH`, and the
     pinned logg of stars above 8000 K.
   - LAMOST DR12 for the LRS data: **not done**.
   - The LAMOST M-star catalog of Li et al. (2025): **prepared as a candidate**
     (`lamost_mstar_li2025`, §5.5; 575,864 rows; DR10 spectra, so the TODO's
     `_dr12` key and URL were corrected; the URL there is the *official* DR12
     M-star catalog, not built). It overlaps DESI DR1 (28,006 stars), CLAM
     (7,500, independent spectra), MDw Cannon (4,493) and SLAM (2,762), all with
     [α/Fe] in both, and its comparison supports the CLAM-based picture of
     §5.5 (DESI DR1 high in [α/Fe] at low temperature). Open decisions for the
     owner before promoting it: the ×2 uncertainty scaling, the SNR threshold
     (30 in the quadrature sum), the APOGEE non-independence, and the many
     repeat spectra (24% of rows; no dedupe, per policy).
   - The Nagaraj et al. (2026) DESI catalog (arXiv 2610.04294): **prepared as a
     candidate** (`desi_nagaraj2026`, §5.5; partial release of ~170,000 stars, not
     yet in the model). The owner hoped it would improve the overlap between
     DESI and the other surveys at low temperature, i.e. where §9.7 shows the
     problem; §5.5 shows it cannot reach the M-dwarf regime (no stars < 4050 K, no
     overlap with SLAM/MDC), only add K-star anchors. Promote it by appending to
     `registry.CATALOGS` and retraining; wait for the full release for much
     larger overlaps.
7. **Open decisions** (below).

### Bugs in the notebook that must be fixed while porting (do not copy)

Items for cells 391–412 and 428 have been fixed in `train.py`/`videos.py`. The
rest are still open: they belong to the `apply.py`/`diagnostics.py` stage.

- ~~Cells 391/393/394, 397, 412, 428~~ (fixed: `variables`/`metrics_history`/
  `n_steps` are defined before use; `ln_err_model` is not ported; `n_bij_vars`
  used; `chi2_turnover` spelt correctly; `np.concatenate`).
- Cell 456 (`get_transformed_pairs`): reads global `x_ctrl` / `c_ctrl` left over
  from the training loop. Pass them explicitly.
- Cell 460: uses `d`, the loop variable left over from cell 387/412.
- Cell 487/511: `params_comb, params_cov_comb = combine_parameters(...)` — the
  function returns a dict.
- Cell 519: `lndet_cov` is undefined there; it lives in the results dict.
- Cell 536: `v[..., 0].ndim` with `v` undefined.
- Cell 464: relies on the global `n_params` (= 4) and on `tables`. (Fixed in
  `apply.py`; see §9.9 for the *math* errors in cells 463–488.)

### Open decisions

1. **Duplicate Gaia ids** (§5.4): dedupe per catalog (e.g. keep highest SNR), or
   allow the many-to-many join? The default is still the legacy join; the
   owner has asked not to change it for now. Biggest open decision.
2. **`lamost_hotpayne` sigma fix**: current code uses the corrected values, so
   results are not comparable with the published numbers.
3. **Learned error model**: skipped on the owner's instruction (it did not seem
   to work well); the question is whether to revisit it later.
4. **Hyperparameters**: the notebook's (cell 412) are hand-tuned and are the
   defaults, apart from `chi2_turnover` (§9.3). The paper's description differs
   from them (§9.3).
5. Reproducing the paper's results exactly is **not** a goal: the new bijector
   does not have to match the old one, but the catalogs must come into
   alignment (§9.4, §9.5).
6. Whether `models/<run>` directories should be archived or pruned: the legacy
   notebook model in `models/` is read-only; `test30` and `turnover50` are
   quick 1-epoch experiments.
