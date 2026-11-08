"""The ordered registry of input catalogs.

The position of a catalog in :data:`CATALOGS` *is* its category index ``c``,
which is baked into every training pair and every model checkpoint. Reordering
or inserting entries changes the meaning of previously trained models, so the
order is frozen and mirrors the order used by the original analysis.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CatalogSpec:
    key: str
    label: str
    module: str
    n_params: int
    legacy_file: str
    """Basename of the corresponding table produced by the legacy notebook."""


#: DO NOT REORDER. Index == category index ``c``.
CATALOGS: Tuple[CatalogSpec, ...] = (
    CatalogSpec("apogee_dr17", "APOGEE (DR17)", "astrotune.catalogs.apogee_dr17", 4, "apogee_dr17.h5"),
    CatalogSpec("lamost_mrs", "LAMOST MRS", "astrotune.catalogs.lamost_mrs", 4, "lamost_mrs.h5"),
    CatalogSpec("lamost_hotpayne", "LAMOST HotPayne", "astrotune.catalogs.lamost_hotpayne", 4, "lamost_hotpayne.h5"),
    CatalogSpec("galah_dr4", "GALAH (DR4)", "astrotune.catalogs.galah_dr4", 4, "galah_dr4.h5"),
    CatalogSpec("lamost_ddp_dr9", "LAMOST DD-Payne (DR9)", "astrotune.catalogs.lamost_ddp_dr9", 4, "lamost_ddp_dr9.h5"),
    CatalogSpec("lamost_mstar_ding2022", "LAMOST M-Stars (Ding)", "astrotune.catalogs.lamost_mstar_ding2022", 3, "lamost_mstar_ding2022.h5"),
    CatalogSpec("vmp_stars_li2022", "VMP (Li)", "astrotune.catalogs.vmp_stars_li2022", 3, "vmp_stars_li2022.h5"),
    CatalogSpec("lamost_csn_dr10", "LAMOST M-Dwarfs (Zhang)", "astrotune.catalogs.lamost_csn_dr10", 3, "lamost_csn_dr10.h5"),
    CatalogSpec("desi_dr1", "DESI (DR1)", "astrotune.catalogs.desi_dr1", 4, "desi_dr1.h5"),
    CatalogSpec("desi_ddp", "DESI (Zhang)", "astrotune.catalogs.desi_ddp", 4, "desi_ddp.h5"),
    CatalogSpec("sdssv_aspcap", "SDSS-V (ASPCAP)", "astrotune.catalogs.sdssv_aspcap", 4, "sdssv_aspcap.h5"),
    CatalogSpec("sdssv_bossnet", "SDSS-V (BOSSNet)", "astrotune.catalogs.sdssv_bossnet", 3, "sdssv_bossnet.h5"),
    CatalogSpec("rave_dr6", "RAVE (DR6)", "astrotune.catalogs.rave_dr6", 4, "rave_dr6.h5"),
    CatalogSpec("pristine_pigs", "PIGS", "astrotune.catalogs.pristine_pigs", 3, "pristine_pigs.h5"),
    CatalogSpec("sdssv_slam", "SDSS-V (SLAM)", "astrotune.catalogs.sdssv_slam", 4, "sdssv_slam.h5"),
    CatalogSpec("sdssv_mdc", "SDSS-V (MDw Cannon)", "astrotune.catalogs.sdssv_mdc", 4, "sdssv_mdc.h5"),
    # Added after the notebook (no legacy table). Models trained before these
    # were added use the first 16 entries only (see run_keys).
    CatalogSpec("desi_nagaraj2026", "DESI (Nagaraj)", "astrotune.catalogs.desi_nagaraj2026", 4, ""),
    CatalogSpec("sdssv_clam", "SDSS-V (CLAM)", "astrotune.catalogs.sdssv_clam", 4, ""),
    CatalogSpec("lamost_mstar_li2025", "LAMOST M-Stars (Li)", "astrotune.catalogs.lamost_mstar_li2025", 4, ""),
)

INDEX = {spec.key: i for i, spec in enumerate(CATALOGS)}

#: Catalogs that can be prepared (``astrotune-prepare KEY``) but are **not yet
#: part of the model**: they have no category index, and training, evaluation
#: and the videos ignore them. Adding a catalog to :data:`CATALOGS` changes the
#: number of categories, so a model trained with N catalogs cannot be loaded
#: once N+1 are registered. To promote a candidate, append it to
#: :data:`CATALOGS` (never insert) and retrain.
#: Currently empty: the first three candidates were promoted (19 catalogs).
CANDIDATES: Tuple[CatalogSpec, ...] = ()


def run_keys(run_dir) -> Tuple[str, ...]:
    """The catalogs a trained run was built with (from its ``config.json``).

    Runs made before the registry grew were trained on a prefix of
    :data:`CATALOGS`; for those the first 16 are returned.
    """
    import json
    from pathlib import Path

    try:
        with open(Path(run_dir) / "config.json") as f:
            return tuple(json.load(f)["catalogs"])
    except (OSError, KeyError):
        return keys()[:16]


def keys() -> Tuple[str, ...]:
    return tuple(spec.key for spec in CATALOGS)


def labels() -> Tuple[str, ...]:
    return tuple(spec.label for spec in CATALOGS)


def category_index(key: str) -> int:
    """Return the fixed category index ``c`` for a catalog key."""
    return INDEX[key]


def get_spec(key: str) -> CatalogSpec:
    """Look up a catalog (including candidates that are not yet in the model)."""
    for spec in CATALOGS + CANDIDATES:
        if spec.key == key:
            return spec
    known = ", ".join(keys() + candidate_keys())
    raise KeyError(f"Unknown catalog {key!r}. Known: {known}")


def candidate_keys() -> Tuple[str, ...]:
    return tuple(spec.key for spec in CANDIDATES)


def get_module(key: str):
    """Import and return the catalog module."""
    return importlib.import_module(get_spec(key).module)
