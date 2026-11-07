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
    pipeline.py        orchestration: download -> load -> select -> build -> save
  catalogs/            one module per input catalog
  cli.py               `astrotune-prepare` command line entry point
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
