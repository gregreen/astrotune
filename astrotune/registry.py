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
)

INDEX = {spec.key: i for i, spec in enumerate(CATALOGS)}


def keys() -> Tuple[str, ...]:
    return tuple(spec.key for spec in CATALOGS)


def labels() -> Tuple[str, ...]:
    return tuple(spec.label for spec in CATALOGS)


def category_index(key: str) -> int:
    """Return the fixed category index ``c`` for a catalog key."""
    return INDEX[key]


def get_spec(key: str) -> CatalogSpec:
    for spec in CATALOGS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown catalog {key!r}. Known: {', '.join(keys())}")


def get_module(key: str):
    """Import and return the catalog module."""
    return importlib.import_module(get_spec(key).module)
