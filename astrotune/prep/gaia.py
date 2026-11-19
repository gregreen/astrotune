"""Gaia DR3 crossmatching for catalogs that do not ship source identifiers.

The crossmatch is expensive (the coordinates are uploaded to the Gaia archive
and joined server-side), so results are cached per catalog under
``data/interim/<key>/`` -- together with a JSON report of what the archive
actually returned.

Why all the bookkeeping
-----------------------
A TAP job can fail in ways that look like success. ``astroquery`` raises for the
``ERROR`` phase, but a job the server *aborted* (queue eviction, quota,
maintenance) comes back as a perfectly parseable empty or truncated table, and
the failure then shows up much later as a catalog that quietly has fewer stars.
Since the catalog sizes feed every downstream number, each batch is checked:

* the job phase is polled here, with a deadline, instead of waiting forever;
* every failure (phase, HTTP, parse, validation) is retried with a fresh upload
  table name, so a retry cannot trip over the leftovers of the failed attempt;
* the returned pair indices must belong to the batch that was uploaded;
* after all batches, ones whose match rate is an outlier are re-run and, if
  their rate does not recover, flagged in the result metadata;
* :func:`crossmatch_catalog` compares the fresh crossmatch with the previous
  run's report and refuses to overwrite a cache with one that matches markedly
  fewer stars.

Which archive
-------------
``archive=`` picks the service (see :data:`ARCHIVES`): ``esa`` is the official
archive, ``gavo`` is GAVO's independent mirror of ``gaia.dr3lite`` and ``ari`` /
``ari-lite`` are the Heidelberg mirror. They differ in ways that matter:

* GAVO and ARI queue a new job in ``PENDING`` and only run it once it is set to
  ``RUN``;
* GAVO and ARI silently insert ``TOP 3000`` into a query that has no TOP, so
  they are queried with an explicit one;
* ARI's asynchronous queue never advances (jobs sit in QUEUED), while its
  synchronous endpoint answers cone searches in a fraction of a second, so it
  is queried synchronously;
* only the three-argument ``POINT('ICRS', ra, dec)`` is parsed by all of them.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import astropy.table
import astropy.units as units
import numpy as np
from astropy.table import Table
from tqdm.auto import tqdm

from .transforms import unmask

log = logging.getLogger(__name__)

MATCH_COLUMNS = ("match_idx", "gdr3_source_id", "ra", "dec", "angsep_arcsec")

GAIA_TABLE = "gaiadr3.gaia_source"

#: Rows uploaded per TAP job. Large uploads are what the archive chokes on: the
#: notebook's 2**20-row batches are how a 4.1-million-row catalog came back with
#: 197 thousand matches.
DEFAULT_MAX_BATCH_SIZE = 1024 * 64

#: Radius the archive is actually searched with. The TAP query is the expensive
#: part -- minutes per batch, and it needs a working archive -- while narrowing
#: the match afterwards costs a numpy comparison. So the search is run once,
#: wide, and the result is cached together with ``angsep_arcsec``; every catalog
#: then applies its own, narrower limit when it reads the cache. Raising a
#: catalog's limit up to this radius therefore needs no new query at all.
#:
#: One arcsecond is the ceiling: beyond it a positional match is not a match any
#: more (the nearest Gaia source that far from a bright star is as likely to be
#: an unrelated neighbour), so nothing here searches wider.
DEFAULT_SEARCH_SEPARATION = 1.0 * units.arcsec

#: A batch may stay queued or executing for this long before the job is aborted.
JOB_TIMEOUT = 2700.0
POLL_INTERVAL = 2.0
MAX_ATTEMPTS = 4
RETRY_BACKOFF = 5.0

#: A batch matching less than this fraction of the median batch rate is re-run.
SUSPICIOUS_FRACTION = 0.5
#: How often a batch that hits the service's row limit is halved before giving up.
MAX_SPLIT_DEPTH = 8
#: Relative drop in the number of matches that aborts a cached-crossmatch rebuild.
REGRESSION_TOLERANCE = 0.01
#: Below this match fraction the result is suspicious, whatever the batches say.
LOW_MATCH_FRACTION = 0.25

REPORT_NAME = "gaia_crossmatch.json"

_TERMINAL_PHASES = (
    "COMPLETED",
    "ERROR",
    "ABORTED",
    "UNKNOWN",
    "HELD",
    "SUSPENDED",
    "ARCHIVED",
)


class CrossmatchError(RuntimeError):
    """The archive did not return a usable crossmatch."""


def _gaia():
    """The TAP client, imported lazily so the package works without astroquery."""
    from astroquery.gaia import Gaia

    return Gaia


@dataclass(frozen=True)
class Archive:
    """A TAP service that serves a Gaia DR3 source table."""

    name: str
    table: str
    #: ``None`` for the official archive, which has a dedicated astroquery class.
    url: Optional[str] = None
    output_format: str = "votable"
    #: Some services cap every result set: GAVO (and the ARI mirror, which runs
    #: the same software) silently inserts ``TOP 3000`` when the query carries
    #: no TOP. Set this to emit an explicit TOP and to treat a result that fills
    #: it as suspicious instead of trusting it.
    top: Optional[int] = None
    #: TAP ``MAXREC`` to request. GAVO answers at most 20,000 rows by default
    #: whatever the ADQL TOP says -- an upload join over a full batch comes back
    #: with exactly 20,001 pairs, which looks like a result and is not -- so the
    #: parameter has to be raised explicitly.
    maxrec: Optional[int] = None
    #: Run the query synchronously. The ARI mirror leaves asynchronous jobs
    #: queued for many minutes (or forever), while its synchronous endpoint
    #: answers the same join in a fraction of a second.
    sync: bool = False
    #: What the table is missing relative to ``gaiadr3.gaia_source``.
    note: str = ""


#: The official ESA archive (``astroquery.gaia``).
ESA = Archive(name="esa", table=GAIA_TABLE, note="full gaia_source")

#: GAVO, the German VO: a mirror that only carries the *lite* Gaia DR3 table
#: (source_id, ra, dec, parallax, pm, photometry). That is everything a
#: positional crossmatch needs, and it is a completely independent service, so
#: it works when the official archive is down for a data release.
GAVO = Archive(
    name="gavo",
    table="gaia.dr3lite",
    url="https://dc.g-vo.org/tap",
    top=10_000_000,
    maxrec=10_000_000,
    note="gaia.dr3lite: no astrometry beyond ra/dec, no astrophysical parameters",
)

#: Heidelberg ARI, which mirrors Gaia DR3 in full (and also serves the lite
#: table). Handy as a second opinion: same table name as the official archive.
ARI = Archive(
    name="ari",
    table=GAIA_TABLE,
    url="https://gaia.ari.uni-heidelberg.de/tap",
    top=10_000_000,
    maxrec=10_000_000,
    sync=True,
    note="Heidelberg ARI mirror, full gaia_source",
)

#: The same service's lite table, for comparison with GAVO's ``dr3lite``.
ARI_LITE = Archive(
    name="ari-lite",
    table="gaiadr3.gaia_source_lite",
    url="https://gaia.ari.uni-heidelberg.de/tap",
    top=10_000_000,
    maxrec=10_000_000,
    sync=True,
    note="Heidelberg ARI mirror, gaia_source_lite",
)

ARCHIVES = {archive.name: archive for archive in (ESA, GAVO, ARI, ARI_LITE)}
#: Accepted spellings, so ``--gaia-archive gaia`` works as people expect.
ALIASES = {"gaia": "esa"}
_clients: dict = {}


def archive_by_name(name) -> Archive:
    """Look up an :class:`Archive` by name (``esa`` or ``gavo``) or pass one through."""
    if isinstance(name, Archive):
        return name
    try:
        return ARCHIVES[ALIASES.get(name, name)]
    except KeyError:
        raise ValueError(
            f"unknown Gaia archive {name!r}; known: {', '.join(sorted(ARCHIVES))}"
        ) from None


def _client(archive: Archive):
    """The TAP client for an archive (GAVO clients are reused across batches)."""
    if archive.url is None:
        return _gaia()
    client = _clients.get(archive.url)
    if client is None:
        from astroquery.utils.tap.core import TapPlus

        client = TapPlus(url=archive.url)
        _clients[archive.url] = client
    return client


@dataclass
class Batch:
    """What one TAP job was asked for and what came back."""

    number: int
    offset: int
    n_uploaded: int
    n_pairs: int = 0
    n_matched: int = 0
    n_multiple: int = 0
    attempts: int = 0
    seconds: float = 0.0
    rerun: bool = False
    suspect: bool = False

    @property
    def fraction(self) -> float:
        """Fraction of the uploaded rows that got a match."""
        return self.n_matched / self.n_uploaded if self.n_uploaded else float("nan")

    def summary(self) -> dict:
        return {
            "number": self.number,
            "offset": self.offset,
            "n_uploaded": self.n_uploaded,
            "n_pairs": self.n_pairs,
            "n_matched": self.n_matched,
            "n_multiple": self.n_multiple,
            "match_fraction": self.fraction,
            "attempts": self.attempts,
            "seconds": round(self.seconds, 1),
            "rerun": self.rerun,
            "suspect": self.suspect,
        }


def _point(alias: str) -> str:
    """An ADQL POINT for a table alias, in the portable three-argument form."""
    return f"POINT('ICRS', {alias}.ra, {alias}.dec)"


def _pairs_query(
    table_name: str, separation_deg: float, method: str, table: str, top: Optional[int] = None
) -> str:
    """ADQL for all (uploaded row, Gaia source) pairs within ``separation_deg``."""
    if method == "distance":
        # The standard three-argument POINT is the only form all three services
        # parse: the two-argument POINT (an extension ESA and GAVO accept) and
        # the four-argument DISTANCE that GAVO documents are both rejected by
        # the ARI mirror. It is still served from the spatial index.
        condition = (
            f"DISTANCE({_point('g')}, {_point('t')}) < {separation_deg}"
        )
    elif method == "contains":
        # The form the archive can answer from its spatial index.
        condition = (
            "1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
            f"CIRCLE('ICRS', t.ra, t.dec, {separation_deg}))"
        )
    else:
        raise ValueError(f"unknown crossmatch method {method!r}; use 'distance' or 'contains'")

    top_clause = f"TOP {top} " if top else ""
    return f"""
      SELECT {top_clause}
        t.index,
        g.source_id, g.ra, g.dec,
        DISTANCE({_point('g')}, {_point('t')}) * 3600. AS angsep_arcsec
      FROM tap_upload.{table_name} AS t
      JOIN {table} AS g
      ON {condition}
    """.strip()


def _wait_for_job(job, *, timeout: float = JOB_TIMEOUT, verbose: bool = False) -> str:
    """Poll a TAP job until a terminal phase; never wait forever.

    GAVO and the ARI mirror queue a new job in ``PENDING`` and only execute it
    once it is set to ``RUN``, which is what astroquery's own
    ``wait_for_job_end()`` does. Driving the phases by hand means doing it here
    too -- without this the job simply never starts.
    """
    deadline = time.monotonic() + timeout
    started = False
    while True:
        phase = str(job.get_phase(update=True)).upper().strip()
        if verbose:
            log.info("job %s: %s", job.jobid, phase)
        if phase in _TERMINAL_PHASES:
            return phase
        if phase == "PENDING" and not started:
            try:
                job.start(verbose=verbose)
                started = True
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("could not start job %s: %s", job.jobid, exc)
        if time.monotonic() >= deadline:
            try:
                job.abort()
            except Exception as exc:  # pragma: no cover - best effort
                log.debug("could not abort job %s: %s", job.jobid, exc)
            raise CrossmatchError(
                f"job {job.jobid} was still {phase} after {timeout:.0f} s"
            )
        time.sleep(POLL_INTERVAL)


def _job_error(job) -> str:
    """The server's explanation for a failed job, if it gave one."""
    try:
        message = " ".join(str(job.get_error()).split())
    except Exception as exc:  # pragma: no cover - depends on the service
        message = f"<no error message: {exc}>"
    return (message or "<empty error message>")[:400]


def _drop_upload(table_name: str, archive: Archive = ESA) -> None:
    """Remove an uploaded table; cleanup is best effort."""
    try:
        _client(archive).delete_user_table(table_name=table_name)
    except Exception as exc:  # pragma: no cover - cleanup is best effort
        log.debug("could not drop upload table %s: %s", table_name, exc)


def _run_query(
    build_query: Callable[[str], str],
    upload: Table,
    *,
    archive: Archive = ESA,
    timeout: float = JOB_TIMEOUT,
    verbose: bool = False,
) -> tuple[Table, int]:
    """Run one TAP query, retrying transient archive failures.

    Every attempt uploads under a fresh table name, so a retry can never trip
    over the leftovers of the attempt that just failed.
    """
    client = _client(archive)
    last_error: Optional[BaseException] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        table_name = f"crossmatch_{uuid.uuid4().hex[:12]}"
        try:
            limit = {}
            if archive.url is not None and archive.maxrec is not None:
                # astroquery's Gaia class does not pass MAXREC through, the
                # plain TAP client does.
                limit["maxrec"] = archive.maxrec
            if archive.sync:
                job = client.launch_job(
                    build_query(table_name),
                    verbose=verbose,
                    upload_resource=upload,
                    upload_table_name=table_name,
                    output_format=archive.output_format,
                    **limit,
                )
            else:
                job = client.launch_job_async(
                    build_query(table_name),
                    verbose=verbose,
                    background=True,
                    upload_resource=upload,
                    upload_table_name=table_name,
                    output_format=archive.output_format,
                    **limit,
                )
                phase = _wait_for_job(job, timeout=timeout, verbose=verbose)
                if phase != "COMPLETED":
                    raise CrossmatchError(
                        f"job {job.jobid} ended in {phase}: {_job_error(job)}"
                    )
            result = job.get_results()
            if result is None:
                raise CrossmatchError(f"job {job.jobid} completed without a result table")
            return result, attempt
        except Exception as exc:  # noqa: BLE001 - every failure here is retryable
            last_error = exc
            log.warning(
                "%s crossmatch attempt %d/%d failed (%s): %s",
                archive.name,
                attempt,
                MAX_ATTEMPTS,
                type(exc).__name__,
                " ".join(str(exc).split())[:300],
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)
        finally:
            _drop_upload(table_name, archive)

    raise CrossmatchError(
        f"no usable result from {archive.name} after {MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _validate_pairs(pairs: Table, index: np.ndarray, separation_deg: float) -> Table:
    """Check that a batch result actually describes the batch that was uploaded."""
    missing = [
        name
        for name in ("index", "source_id", "ra", "dec", "angsep_arcsec")
        if name not in pairs.colnames
    ]
    if missing:
        raise CrossmatchError(f"the archive result is missing the column(s) {missing}")
    if len(pairs) == 0:
        return pairs  # an empty batch is legitimate; the audit decides if it is plausible

    idx = np.asarray(pairs["index"])
    low, high = int(index.min()), int(index.max())
    outside = (idx < low) | (idx > high)
    if np.any(outside):
        raise CrossmatchError(
            f"{int(np.count_nonzero(outside))} of {len(pairs)} returned pairs carry an "
            f"index outside the uploaded range {low}..{high}"
        )

    separation = np.asarray(pairs["angsep_arcsec"], dtype="f8")
    if not np.all(np.isfinite(separation)):
        raise CrossmatchError("the archive result contains non-finite separations")
    limit = separation_deg * 3600.0 * 1.01
    if np.any(separation > limit):
        raise CrossmatchError(
            f"{int(np.count_nonzero(separation > limit))} returned pairs are farther "
            f"than the {limit / 3600:.5f} deg that was searched"
        )
    return pairs


def _nearest(pairs: Table, max_sep_arcsec: float) -> tuple[Table, int, int]:
    """Keep the closest Gaia source per uploaded row, within ``max_sep_arcsec``."""
    dtypes = (
        ("index", "i8"),
        ("source_id", "i8"),
        ("ra", "f8"),
        ("dec", "f8"),
        ("angsep_arcsec", "f8"),
    )
    if len(pairs) == 0:
        return Table({name: np.asarray([], dtype=dtype) for name, dtype in dtypes}), 0, 0

    ordered = pairs.copy()
    ordered.sort("angsep_arcsec")
    nearest = astropy.table.unique(ordered, keys=["index"], keep="first", silent=True)
    _, counts = np.unique(np.asarray(ordered["index"]), return_counts=True)
    n_multiple = int(np.count_nonzero(counts > 1))

    keep = np.asarray(nearest["angsep_arcsec"], dtype="f8") < max_sep_arcsec
    selected = Table({name: np.asarray(nearest[name])[keep] for name, _ in dtypes})
    return selected, n_multiple, len(selected)


def _run_batch(
    ra: np.ndarray,
    dec: np.ndarray,
    index: np.ndarray,
    *,
    number: int,
    offset: int,
    separation_deg: float,
    max_sep_arcsec: float,
    method: str,
    archive: Archive,
    timeout: float,
    verbose: bool,
    depth: int = 0,
) -> tuple[Table, Batch]:
    """Crossmatch one batch and return its selected pairs plus the accounting.

    A result that fills the service's row limit is not a result: GAVO answers an
    upload join with exactly 20,001 pairs when its default MAXREC of 20,000 is
    reached, which looks like a successful batch and silently drops two thirds
    of the matches. Such a batch is split in half and each half run separately,
    down to :data:`MAX_SPLIT_DEPTH` levels.
    """
    report = Batch(number=number, offset=offset, n_uploaded=len(index))
    upload = Table({"ra": ra, "dec": dec, "index": index})

    started = time.monotonic()
    pairs, report.attempts = _run_query(
        lambda table_name: _pairs_query(
            table_name, separation_deg, method, archive.table, archive.top
        ),
        upload,
        archive=archive,
        timeout=timeout,
        verbose=verbose,
    )
    report.seconds = time.monotonic() - started
    pairs = _validate_pairs(pairs, index, separation_deg)

    limit = archive.maxrec if archive.maxrec is not None else archive.top
    if limit is not None and len(pairs) >= limit:
        if depth >= MAX_SPLIT_DEPTH or len(index) < 2:
            raise CrossmatchError(
                f"{archive.name} returned {len(pairs):,} pairs, which fills its row "
                f"limit of {limit:,}; the result is cut off even after "
                f"{depth} split(s)"
            )
        middle = len(index) // 2
        log.warning(
            "%s batch %d returned %s pairs (the %s row limit): splitting it in two",
            archive.name,
            number,
            f"{len(pairs):,}",
            f"{limit:,}",
        )
        left, left_report = _run_batch(
            ra[:middle],
            dec[:middle],
            index[:middle],
            number=number,
            offset=offset,
            separation_deg=separation_deg,
            max_sep_arcsec=max_sep_arcsec,
            method=method,
            archive=archive,
            timeout=timeout,
            verbose=verbose,
            depth=depth + 1,
        )
        right, right_report = _run_batch(
            ra[middle:],
            dec[middle:],
            index[middle:],
            number=number,
            offset=offset,
            separation_deg=separation_deg,
            max_sep_arcsec=max_sep_arcsec,
            method=method,
            archive=archive,
            timeout=timeout,
            verbose=verbose,
            depth=depth + 1,
        )
        merged = Batch(
            number=number,
            offset=offset,
            n_uploaded=len(index),
            n_pairs=left_report.n_pairs + right_report.n_pairs,
            n_matched=left_report.n_matched + right_report.n_matched,
            n_multiple=left_report.n_multiple + right_report.n_multiple,
            attempts=max(left_report.attempts, right_report.attempts),
            seconds=left_report.seconds + right_report.seconds,
            rerun=left_report.rerun or right_report.rerun,
            suspect=left_report.suspect or right_report.suspect,
        )
        return astropy.table.vstack([left, right]), merged

    report.n_pairs = len(pairs)
    selected, report.n_multiple, report.n_matched = _nearest(pairs, max_sep_arcsec)
    return selected, report


def _audit_batches(
    chunks: list[np.ndarray],
    pieces: list[Table],
    reports: list[Batch],
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    separation_deg: float,
    max_sep_arcsec: float,
    method: str,
    archive: Archive,
    timeout: float,
    verbose: bool,
) -> tuple[list[Table], list[Batch]]:
    """Re-run batches whose match rate is an outlier -- the sign of a lost result."""
    fractions = np.array([r.fraction for r in reports], dtype="f8")
    if len(reports) < 2 or not np.all(np.isfinite(fractions)):
        return pieces, reports
    median = float(np.median(fractions))
    if median <= 0:
        return pieces, reports

    for i, report in enumerate(reports):
        if report.fraction >= SUSPICIOUS_FRACTION * median:
            continue

        log.warning(
            "batch %d matched %.1f%% of its %s sources against a %.1f%% median: re-running",
            report.number,
            100 * report.fraction,
            f"{report.n_uploaded:,}",
            100 * median,
        )
        chunk = chunks[i]
        piece, retry = _run_batch(
            ra[chunk],
            dec[chunk],
            chunk,
            number=report.number,
            offset=report.offset,
            separation_deg=separation_deg,
            max_sep_arcsec=max_sep_arcsec,
            method=method,
            archive=archive,
            timeout=timeout,
            verbose=verbose,
        )
        retry.rerun = True
        if retry.n_matched > report.n_matched:
            pieces[i], reports[i] = piece, retry
            log.warning(
                "batch %d recovered %s matches (was %s)",
                report.number,
                f"{retry.n_matched:,}",
                f"{report.n_matched:,}",
            )

        if reports[i].fraction < SUSPICIOUS_FRACTION * median:
            reports[i].suspect = True
            log.error(
                "batch %d still matches only %.1f%% of its sources; its result may be "
                "incomplete and the catalog built from it will be too",
                report.number,
                100 * reports[i].fraction,
            )

    return pieces, reports


def gaia_crossmatch_batched(
    ra,
    dec,
    *,
    max_separation=0.5 * units.arcsec,
    max_batch_size=DEFAULT_MAX_BATCH_SIZE,
    method="distance",
    archive=ESA,
    timeout=JOB_TIMEOUT,
    audit=True,
    verbose=False,
) -> Table:
    """Nearest-neighbor match to ``gaiadr3.gaia_source`` for arbitrary coordinates.

    Batches run one at a time, each with its own deadline, retry budget and
    validation, so a failure costs one batch of work rather than a silently
    smaller catalog. The returned table carries the accounting in ``meta``
    (``n_uploaded``, ``n_matched``, ``match_fraction`` and one entry per batch),
    which :func:`write_report` persists and :func:`check_regression` compares
    against.
    """
    archive = archive_by_name(archive)
    ra = np.asarray(ra, dtype="f8")
    dec = np.asarray(dec, dtype="f8")
    if ra.shape != dec.shape:
        raise ValueError(f"ra {ra.shape} and dec {dec.shape} have different shapes")

    finite = np.isfinite(ra) & np.isfinite(dec)
    if not finite.all():
        # Uploading NaN coordinates makes the whole batch unreliable, so leave
        # those rows out and let them stay unmatched.
        log.warning(
            "%s of %s rows have non-finite coordinates and are not matched",
            f"{int(np.count_nonzero(~finite)):,}",
            f"{len(ra):,}",
        )
    rows = np.flatnonzero(finite)

    max_sep_arcsec = max_separation.to("arcsec").value
    separation_deg = max_separation.to("deg").value
    chunks = [rows[i : i + max_batch_size] for i in range(0, len(rows), max_batch_size)]
    log.info(
        "crossmatching %s sources in %d batch(es) of at most %s rows",
        f"{len(rows):,}",
        len(chunks),
        f"{max_batch_size:,}",
    )

    pieces: list[Table] = []
    reports: list[Batch] = []
    for number, chunk in enumerate(tqdm(chunks, disable=not verbose)):
        piece, report = _run_batch(
            ra[chunk],
            dec[chunk],
            chunk,
            number=number,
            offset=int(chunk[0]) if len(chunk) else 0,
            separation_deg=separation_deg,
            max_sep_arcsec=max_sep_arcsec,
            method=method,
            archive=archive,
            timeout=timeout,
            verbose=verbose,
        )
        pieces.append(piece)
        reports.append(report)

    if audit:
        pieces, reports = _audit_batches(
            chunks,
            pieces,
            reports,
            ra,
            dec,
            separation_deg=separation_deg,
            max_sep_arcsec=max_sep_arcsec,
            method=method,
            archive=archive,
            timeout=timeout,
            verbose=verbose,
        )

    non_empty = [piece for piece in pieces if len(piece)]
    if non_empty:
        combined = astropy.table.vstack(non_empty)
        crossmatch = Table(
            {
                "match_idx": np.asarray(combined["index"], dtype="i8"),
                "gdr3_source_id": np.asarray(combined["source_id"], dtype="i8"),
                "ra": np.asarray(combined["ra"], dtype="f8"),
                "dec": np.asarray(combined["dec"], dtype="f8"),
                "angsep_arcsec": np.asarray(combined["angsep_arcsec"], dtype="f8"),
            }
        )
    else:
        crossmatch = Table(
            {
                "match_idx": np.asarray([], dtype="i8"),
                "gdr3_source_id": np.asarray([], dtype="i8"),
                "ra": np.asarray([], dtype="f8"),
                "dec": np.asarray([], dtype="f8"),
                "angsep_arcsec": np.asarray([], dtype="f8"),
            }
        )

    n_uploaded = int(len(rows))
    n_matched = int(len(crossmatch))
    crossmatch.meta.update(
        {
            "n_rows": int(len(ra)),
            "n_uploaded": n_uploaded,
            "n_matched": n_matched,
            "match_fraction": n_matched / n_uploaded if n_uploaded else 0.0,
            "max_separation_arcsec": float(max_sep_arcsec),
            "method": method,
            "archive": archive.name,
            "gaia_table": archive.table,
            "batch_size": int(max_batch_size),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "batches": [report.summary() for report in reports],
            "suspect_batches": [r.number for r in reports if r.suspect],
        }
    )

    log.info(
        "%s/%s (%.1f%%) sources matched within %.2f arcsec",
        f"{n_matched:,}",
        f"{n_uploaded:,}",
        100 * crossmatch.meta["match_fraction"],
        max_sep_arcsec,
    )
    if crossmatch.meta["suspect_batches"]:
        log.error(
            "batch(es) %s never reached the median match rate; treat this crossmatch "
            "as incomplete",
            crossmatch.meta["suspect_batches"],
        )
    return crossmatch


def apply_crossmatch(catalog: Table, crossmatch: Table, *, max_separation=None) -> Table:
    """Subselect ``catalog`` to matched rows and attach the Gaia identifiers.

    ``angsep_arcsec`` is carried over as well, so that the cached table can be
    narrowed again later without going back to the archive.
    """
    if max_separation is None:
        keep = np.ones(len(crossmatch), dtype=bool)
    else:
        keep = (
            np.asarray(crossmatch["angsep_arcsec"], dtype="f8")
            < max_separation.to("arcsec").value
        )

    n_orig = len(catalog)
    match_idx = np.asarray(crossmatch["match_idx"])[keep]
    if len(match_idx) == 0:
        raise CrossmatchError(
            f"not one of the {n_orig:,} sources has a Gaia DR3 counterpart within the "
            "separation that was searched, so the catalog would be empty"
        )
    if np.any(match_idx < 0) or np.any(match_idx >= n_orig):
        raise CrossmatchError(
            f"crossmatch indices do not fit a catalog of {n_orig:,} rows; the cached "
            "crossmatch and the raw catalog are out of step"
        )

    out = catalog[match_idx]
    out["gdr3_source_id"] = np.asarray(crossmatch["gdr3_source_id"])[keep]
    out["ra_gdr3"] = np.asarray(crossmatch["ra"])[keep]
    out["dec_gdr3"] = np.asarray(crossmatch["dec"])[keep]
    out["angsep_arcsec"] = np.asarray(crossmatch["angsep_arcsec"], dtype="f8")[keep]

    fraction = len(out) / n_orig if n_orig else 0.0
    if fraction < LOW_MATCH_FRACTION:
        log.warning(
            "only %.1f%% of the %s sources have a crossmatch, which is low enough to be "
            "worth checking",
            100 * fraction,
            f"{n_orig:,}",
        )
    log.info("%.2f%% of sources have a crossmatch", 100 * fraction)
    return out


def _report_path(interim_dir) -> Path:
    return Path(interim_dir) / REPORT_NAME


def write_report(
    interim_dir,
    crossmatch: Table,
    *,
    key: Optional[str] = None,
    n_rows_after_cut: Optional[int] = None,
) -> Path:
    """Persist the accounting of a crossmatch next to its cache."""
    payload = dict(crossmatch.meta)
    payload["key"] = key
    if n_rows_after_cut is not None:
        payload["n_rows_after_cut"] = int(n_rows_after_cut)

    path = _report_path(interim_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    log.info("wrote crossmatch report %s", path)
    return path


def read_report(interim_dir) -> Optional[dict]:
    """The accounting of the previous crossmatch, if there is one."""
    path = _report_path(interim_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read the crossmatch report %s: %s", path, exc)
        return None


def check_regression(
    previous: Optional[dict],
    crossmatch: Table,
    *,
    key: Optional[str] = None,
    tolerance: float = REGRESSION_TOLERANCE,
) -> None:
    """Refuse a crossmatch that matches markedly fewer stars than the last one.

    A positional crossmatch does not shrink by itself: fewer matches mean the
    archive returned an incomplete result, and overwriting the cache would hide
    that behind a smaller catalog. The comparison only means something between
    runs that searched the same radius, so a deliberate change of
    ``search_separation`` re-baselines instead of tripping the guard.
    """
    if not previous:
        return
    before = previous.get("n_matched")
    after = crossmatch.meta.get("n_matched")
    if not isinstance(before, int) or not isinstance(after, int) or before <= 0:
        return

    old_radius = previous.get("max_separation_arcsec")
    new_radius = crossmatch.meta.get("max_separation_arcsec")
    if (
        isinstance(old_radius, (int, float))
        and isinstance(new_radius, (int, float))
        and abs(old_radius - new_radius) > 1e-6
    ):
        log.info(
            "search radius changed from %.2f to %.2f arcsec: re-baselining the "
            "crossmatch guard (%s -> %s stars matched)",
            old_radius,
            new_radius,
            f"{before:,}",
            f"{after:,}",
        )
        return

    # Fewer matches are expected when the search radius itself was narrowed, so
    # the guard only applies to runs that searched the same distance.
    before_radius = previous.get("max_separation_arcsec")
    after_radius = crossmatch.meta.get("max_separation_arcsec")
    if before_radius != after_radius:
        log.info(
            "%s: search radius changed from %s to %s arcsec, so the new baseline "
            "replaces the old one",
            key or "crossmatch",
            before_radius,
            after_radius,
        )
        return

    if after >= before * (1.0 - tolerance):
        return

    raise CrossmatchError(
        f"{key or 'crossmatch'}: the archive matched {after:,} of "
        f"{crossmatch.meta.get('n_uploaded', 0):,} sources, but the previous run "
        f"({previous.get('created_utc', 'undated')}) matched {before:,}, i.e. "
        f"{1.0 - after / before:.1%} fewer. That is the signature of an incomplete "
        "result, not of a smaller catalog, so the cached crossmatch was left "
        "untouched. Re-run when the archive is healthy; pass "
        "allow_regression=True only if the drop is really what you want."
    )


def narrow(table: Table, max_separation=None, *, key: Optional[str] = None) -> Table:
    """Cut a cached crossmatch down to ``max_separation``.

    This is the cheap half of the two-level radius: the archive is asked for
    everything within :data:`DEFAULT_SEARCH_SEPARATION` and cached, and each
    catalog narrows that to its own limit here -- on a cache hit as well as
    after a fresh query -- so changing the limit never costs a TAP query.
    """
    if max_separation is None or "angsep_arcsec" not in table.colnames:
        return table
    keep = np.asarray(table["angsep_arcsec"], dtype="f8") < max_separation.to(
        "arcsec"
    ).value
    if keep.all():
        return table
    narrowed = table[keep]
    log.info(
        "[%s] %s of %s crossmatched rows are within %.2f arcsec",
        key or "crossmatch",
        f"{len(narrowed):,}",
        f"{len(table):,}",
        max_separation.to("arcsec").value,
    )
    return narrowed


def crossmatch_catalog(
    make_catalog: Callable[[], Table],
    *,
    ra_col: str,
    dec_col: str,
    cache_path=None,
    search_separation=None,
    max_separation=None,
    force: bool = False,
    allow_regression: bool = False,
    key: Optional[str] = None,
    report_dir=None,
    archive=ESA,
    postprocess: Optional[Callable[[Table], Table]] = None,
    verbose: bool = False,
    **write_kwargs,
) -> Table:
    """Crossmatch a catalog, guard the result, then cache it and report on it.

    Two radii, deliberately: ``max_separation`` is what the catalog wants, while
    the archive is searched with ``search_separation`` (by default
    :data:`DEFAULT_SEARCH_SEPARATION`) and the cache keeps everything found that
    way, along with the separations. The catalog's own limit is then applied on
    every load, cached or not, so lowering a match radius -- or raising it again
    towards the search radius -- never costs a TAP query.

    ``make_catalog`` is only called when the cache is missing, so a cache hit
    stays cheap. The cache is written *after* the comparison with the previous
    run, so a bad crossmatch leaves the good one in place.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists() and not force:
        log.info("using cached crossmatch %s", cache_path)
        cached = Table.read(cache_path)
        if "angsep_arcsec" not in cached.colnames and max_separation is not None:
            log.warning(
                "[%s] %s predates the wide search radius: it was cut when it was "
                "written, so its %.2f arcsec limit cannot be changed without "
                "--force",
                key or "crossmatch",
                cache_path.name,
                max_separation.to("arcsec").value,
            )
        return narrow(cached, max_separation, key=key)

    search = search_separation if search_separation is not None else DEFAULT_SEARCH_SEPARATION
    if max_separation is not None and max_separation > search:
        raise ValueError(
            f"the catalog wants {max_separation.to('arcsec').value:.2f} arcsec but the "
            f"archive is only searched to {search.to('arcsec').value:.2f}; raise "
            "search_separation or lower max_separation"
        )

    catalog = make_catalog()
    crossmatch = gaia_crossmatch_batched(
        catalog[ra_col],
        catalog[dec_col],
        max_separation=search,
        archive=archive,
        verbose=verbose,
    )
    # Cache the wide selection, not the narrow one.
    matched = apply_crossmatch(catalog, crossmatch, max_separation=search)
    if postprocess is not None:
        matched = postprocess(matched)

    directory = report_dir if report_dir is not None else (
        cache_path.parent if cache_path is not None else None
    )
    if directory is not None:
        previous = read_report(directory)
        if not allow_regression:
            check_regression(previous, crossmatch, key=key)
        write_report(directory, crossmatch, key=key, n_rows_after_cut=len(matched))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        matched.write(cache_path, overwrite=True, **write_kwargs)
        log.info("wrote %s (%s rows)", cache_path, f"{len(matched):,}")
    return narrow(matched, max_separation, key=key)


def cached_crossmatch(
    key: str,
    interim_dir: Path,
    ra,
    dec,
    *,
    force: bool = False,
    max_separation=0.5 * units.arcsec,
    allow_regression: bool = False,
    verbose=False,
) -> Table:
    """Crossmatch with an on-disk cache in ``data/interim/<key>/``."""
    path = Path(interim_dir) / "gaia_crossmatch.h5"
    if path.exists() and not force:
        log.info("using cached crossmatch %s", path)
        return Table.read(path)

    crossmatch = gaia_crossmatch_batched(
        ra, dec, max_separation=max_separation, verbose=verbose
    )
    if not allow_regression:
        check_regression(read_report(interim_dir), crossmatch, key=key)
    write_report(interim_dir, crossmatch, key=key)

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
