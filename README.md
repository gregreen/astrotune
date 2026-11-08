# Astrotune

Bringing disparate stellar parameter catalogs into harmony.

Astrotune learns a set of minimal, invertible transformations that map each
input catalog into a common ("harmonized") parameter space, using stars
observed in multiple catalogs as anchors.

## Layout

```
astrotune/
  config.py            paths, parameter order, error floors
  registry.py          ordered list of catalogs (the category index `c`)
  prep/                shared machinery for building catalog tables
    download.py        cached HTTP downloads (+ SDSS-V authentication)
    transforms.py      parameter transformations, error floors, extract_table
    cuts.py            quality-cut helpers (finite masks, SNR-chi2 relation)
    abundances.py      inverse-variance [alpha/Fe] from individual abundances
    gaia.py            Gaia DR3 crossmatching (batched + cached)
    io.py              save/load/validate, parity checks against legacy output
    plots.py           diagnostic figures
    overlaps.py        stars shared with the other catalogs, by Teff (`astrotune-overlaps`)
    pipeline.py        orchestration: download -> load -> select -> build -> save
  catalogs/            one module per input catalog (registry.CATALOGS) and candidates
  cli.py               `astrotune-prepare` command line entry point
  model/               VectorNetwork (velocity field) and ODEBijector
  training_data.py     pairs + control points (`astrotune-pairs`)
  train.py             training loop, checkpoints, SavedModel (`astrotune-train`)
  training_plots.py    progress figures written during training
  evaluate.py          catalog agreement before/after harmonization
  videos.py            Kiel-diagram videos of the transformation (`astrotune-videos`)
data/
  raw/<catalog>/       downloaded source catalogs (never tracked)
  interim/<catalog>/   derived caches, e.g. Gaia crossmatch tables
  training/<catalog>/  <catalog>.h5 + <catalog>.meta.json
diagnostic_plots/
  <catalog>/           figures produced during preparation
```

## Environment

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

TensorFlow is installed with the `and-cuda` extra: the required CUDA 12.5 and
cuDNN 9.3 libraries come from `nvidia-*` wheels, so no system CUDA toolkit or
module load is needed. Verified on an A100 80GB (driver 615.71.10).

## Usage

```
astrotune-prepare --list
astrotune-prepare apogee_dr17
astrotune-prepare --all
astrotune-prepare apogee_dr17 --no-plots
```

Diagnostic plots are written to `diagnostic_plots/<catalog>/` by default and
can be disabled with `--no-plots`.

### Candidate catalogs

`registry.CANDIDATES` holds catalogs that can be prepared but are not yet part of
the model (they have no category index; training ignores them). It is currently
empty: `desi_nagaraj2026`, `sdssv_clam` and `lamost_mstar_li2025` were promoted
into `registry.CATALOGS` (19 catalogs) on 2026-11-08. `--all` builds the catalogs
in the model.

```
astrotune-prepare sdssv_clam             # download + cuts + diagnostics
astrotune-overlaps sdssv_clam            # stars shared with every other catalog, by Teff,
                                         # and those with [alpha/Fe] in both
```

A catalog may mark a parameter as not measured for individual stars
(`sigma_x = x_err = 1e9`, `x = 0`), as `sdssv_clam` does for its `FLAGS_WARN`
stars; the loss and the figures treat this like a catalog that lacks the
parameter altogether.

### Training

```
astrotune-pairs                       # build data/pairs_and_ctrl_points.h5
astrotune-train --run-name my_run     # train; writes models/my_run/
```

`astrotune-pairs --dedupe first` keeps one row per Gaia id in every catalog;
the default (`none`) reproduces the legacy many-to-many join. The dispersed
control points get a per-catalog `x_err` (the catalog's median, or its
`X_ERR_DISPERSED`); `--dispersed-xerr uniform` gives the legacy uniform 0.2. Use
`-o` to write a different file and `astrotune-train --pairs-path` to train on it.

During training a progress bar is shown and a figure (loss history plus the
catalogs before/after the transformation) is saved every 5 minutes to
`models/<run>/progress/progress_0001.png, progress_0002.png, ...`
(`--plot-every-seconds`, `--plot-every-steps`). Checkpoints, a SavedModel with
`to_harmonized` / `from_harmonized`, `config.json` and `metrics_history.json`
are written alongside. Every field of `TrainConfig` is a command-line option;
the defaults are the reference notebook's hyperparameters. The notebook's
learned error model is not ported. Two options are off by default:
`--pair-path-in-loss` (add the pair path-length penalty to the loss) and
`--freeze-after-frac F` (freeze the embedding and FiLM variables after a
fraction `F` of training).

Resumable state (model, optimizer, metrics, figure numbering) is saved at each
progress figure and every 10 minutes (`--checkpoint-every-seconds`), and on
Ctrl-C / SIGTERM. Continue a run with

```
astrotune-train --run-name my_run --resume
```

Only plotting/checkpoint intervals may be changed when resuming.

### Evaluating a run

The test of a run is whether catalogs agree better after the transformation.
For held-out pairs, `astrotune-evaluate --run-dir models/my_run` (also run at
the end of training) compares `x_j - x_k` with `f_j(x_j) - f_k(x_k)` for every
catalog pair and parameter, and writes to `models/my_run/evaluation/`:
`agreement.json` (offset and scatter per pair/parameter),
`agreement_summary.png` (each point a catalog pair; below the diagonal is an
improvement), `pairs/pair_JJ_KK.png` (catalog-vs-catalog, original and
harmonized, per parameter) and `param_shifts.png` (`f(x) - x` per catalog).
The same summary is shown in the title of every progress figure.

### Videos

```
astrotune-videos --run-dir models/my_run
```

writes to `models/my_run/videos/`, in two planes (`kiel`: Teff-logg;
`feh_alpha`: [Fe/H]-[alpha/Fe]): `kiel.mp4` (no highlighting),
`kiel_control_10.mp4` (SDSS-V ASPCAP highlighted) and `kiel_pair_3_10.mp4` (the
stars shared by GALAH DR4 and ASPCAP, as measured by each; they should
converge), and the same three as `feh_alpha*.mp4`. Stars whose catalog does
not measure a plotted parameter are not drawn. Use `--plane`, `--highlight KEY`
and `--pair KEY KEY` for others.

Two more, pair-only video types compare the two catalogs parameter by parameter
(2x2 grid, one panel per parameter; catalog j horizontal, catalog k vertical):
`--plane param1v1` (k vs j; points should converge onto the diagonal) and
`--plane param1v1_diff` (k − j vs j; onto zero). They write
`param1v1[_diff]_pair_<j>_<k>.mp4` and need `--pair`, or `--all-pairs-min N` for
every catalog pair with at least N stars in common (the notebook used 50):

```
astrotune-videos --run-dir models/my_run --no-plain --plane param1v1_diff --all-pairs-min 50
```

The encoder is `ffmpeg` from `PATH`, else the static binary of `imageio-ffmpeg`.

## Catalog module contract

Each module in `astrotune/catalogs/` provides:

```python
KEY, LABEL, N_PARAMS          # identity
REFERENCE, SOURCES            # provenance, recorded in the sidecar metadata
ID_COL, PARAM_COLS, UNC_COLS, EXTRA_COLS
ERROR_FLOORS                  # regularization error floors, per parameter

def download(ctx) -> None                 # populate ctx.raw_dir (cached)
def load(ctx) -> Table                    # read raw files into memory
def prepare_columns(t, ctx) -> Table      # optional: derived columns
def select(t, ctx) -> mask                # quality cuts
def param_errs(t, ctx) -> (n, d) array    # regularization errors, full length
def diagnostics(t, mask, table, ctx)      # optional: catalog-specific plots
```

The shared pipeline handles transformation, error floors, saving, validation
and the generic diagnostic figures.
