"""Gaia DR3 crossmatching for catalogs that do not ship source identifiers.

The crossmatch is expensive (data are uploaded to the Gaia archive and joined
server-side), so results are cached per catalog under ``data/interim/<key>/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import astropy.table
import astropy.units as units
import numpy as np
from astropy.table import Table
from tqdm.auto import tqdm

from .transforms import unmask

log = logging.getLogger(__name__)

MATCH_COLUMNS = ("match_idx", "gdr3_source_id", "ra", "dec", "angsep_arcsec")


def gaia_crossmatch_batched(
    ra,
    dec,
    *,
    max_separation=0.5 * units.arcsec,
    max_batch_size=1024 * 256,
    verbose=False,
) -> Table:
    """Nearest-neighbor match to ``gaiadr3.gaia_source`` for arbitrary coordinates."""
    from astroquery.gaia import Gaia

    ra = np.asarray(ra, dtype="f8")
    dec = np.asarray(dec, dtype="f8")
    max_sep_deg = max_separation.to("deg").value

    batches = [
        (ra[i : i + max_batch_size], dec[i : i + max_batch_size], i)
        for i in range(0, len(ra), max_batch_size)
    ]

    jobs, table_names = [], []
    if verbose:
        log.info("Launching %d crossmatch job(s)", len(batches))

    for batch_ra, batch_dec, offset in tqdm(batches, disable=not verbose):
        upload = Table(
            {"ra": batch_ra, "dec": batch_dec, "index": np.arange(len(batch_ra)) + offset}
        )
        table_name = f"crossmatch_table_{offset}"
        table_names.append(table_name)

        query = f"""
          SELECT
            t.index,
            g.source_id, g.ra, g.dec,
            DISTANCE(POINT(t.ra, t.dec), POINT(g.ra, g.dec)) * 3600. AS angsep_arcsec
          FROM tap_upload.{table_name} AS t
          JOIN gaiadr3.gaia_source AS g
          ON DISTANCE(POINT(t.ra, t.dec), POINT(g.ra, g.dec)) < {max_sep_deg}
        """.strip()

        jobs.append(
            Gaia.launch_job_async(
                query, verbose=verbose, background=True,
                upload_resource=upload, upload_table_name=table_name,
            )
        )

    results = [job.get_results() for job in jobs]

    for table_name in table_names:
        try:
            Gaia.delete_user_table(table_name=table_name)
        except Exception as exc:
            log.warning("Failed to delete %s: %s", table_name, exc)

    all_results = astropy.table.vstack(results)
    all_results.sort(["angsep_arcsec"])
    nearest = astropy.table.unique(all_results, keys=["index"], keep="first")

    max_sep_arcsec = max_separation.to("arcsec").value
    keep = nearest["angsep_arcsec"] < max_sep_arcsec
    log.info(
        "%s/%s (%.1f%%) sources matched within %.2f arcsec",
        int(np.count_nonzero(keep)),
        f"{len(nearest):,}",
        100 * np.mean(keep),
        max_sep_arcsec,
    )
    nearest = nearest[keep]

    return Table(
        {
            "match_idx": nearest["index"],
            "gdr3_source_id": nearest["source_id"],
            "ra": nearest["ra"],
            "dec": nearest["dec"],
            "angsep_arcsec": nearest["angsep_arcsec"],
        }
    )


def apply_crossmatch(catalog: Table, crossmatch: Table, *, max_separation=None) -> Table:
    """Subselect ``catalog`` to matched rows and attach the Gaia identifiers."""
    if max_separation is None:
        keep = slice(None)
    else:
        keep = crossmatch["angsep_arcsec"] < max_separation.to("arcsec").value

    n_orig = len(catalog)
    out = catalog[np.asarray(crossmatch["match_idx"])[keep]]
    out["gdr3_source_id"] = np.asarray(crossmatch["gdr3_source_id"])[keep]
    out["ra_gdr3"] = np.asarray(crossmatch["ra"])[keep]
    out["dec_gdr3"] = np.asarray(crossmatch["dec"])[keep]
    log.info("%.2f%% of sources have a crossmatch", 100 * len(out) / n_orig)
    return out


def cached_crossmatch(
    key: str,
    interim_dir: Path,
    ra,
    dec,
    *,
    force: bool = False,
    max_separation=0.5 * units.arcsec,
    verbose=False,
) -> Table:
    """Crossmatch with an on-disk cache in ``data/interim/<key>/``."""
    path = Path(interim_dir) / "gaia_crossmatch.h5"
    if path.exists() and not force:
        log.info("Using cached crossmatch %s", path)
        return Table.read(path)

    crossmatch = gaia_crossmatch_batched(
        ra, dec, max_separation=max_separation, verbose=verbose
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    crossmatch.write(path, serialize_metadata=True, overwrite=True)
    return crossmatch


def attach_gdr3(catalog: Table, crossmatch: Table) -> Table:
    """Add ``gdr3_source_id``, ``ra`` and ``dec`` columns to a catalog in place.

    The catalog must already be aligned row-for-row with ``crossmatch``
    (i.e. ``crossmatch['match_idx']`` is a permutation of ``arange(len)``).
    """
    out = catalog.copy()
    out["gdr3_source_id"] = unmask(crossmatch["gdr3_source_id"]).astype("i8")
    out["ra"] = unmask(crossmatch["ra"])
    out["dec"] = unmask(crossmatch["dec"])
    return out
