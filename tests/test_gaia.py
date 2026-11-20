"""Tests for the Gaia crossmatch bookkeeping (astrotune.prep.gaia).

The archive is replaced by a scripted fake: it can abort a job, time out, raise
from the upload, or return a table that is empty or about a different batch.
"""

import numpy as np
import pytest
from astropy.table import Table, vstack as astropy_table_vstack
import astropy.units as units

from astrotune.prep import gaia as g


# --- fake archive ------------------------------------------------------------
class FakeJob:
    def __init__(self, response):
        self.jobid = f"job-{id(self)}"
        self._response = response
        self._phases = ["EXECUTING", "COMPLETED"]
        self.starts = 0

    def get_phase(self, update=False):
        if self._response.get("requires_start") and not self.starts:
            return "PENDING"
        phase = self._phases.pop(0) if len(self._phases) > 1 else self._phases[0]
        return self._response.get("final_phase", "COMPLETED") if phase == "COMPLETED" else phase

    def start(self, verbose=False):
        self.starts += 1

    def get_results(self):
        if "error" in self._response:
            raise self._response["error"]
        return self._response["table"]

    def get_error(self):
        return self._response.get("message", "")

    def abort(self):
        self._response["aborted"] = True


class FakeArchive:
    """Plays back a script, one entry per ``launch_job_async`` call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.dropped = []

    def launch_job_async(self, query, **kwargs):
        table_name = kwargs["upload_table_name"]
        assert f"tap_upload.{table_name}" in query
        upload = kwargs["upload_resource"]
        response = self.responses.pop(0)
        if callable(response):
            response = response(upload)
        self.calls.append({"table_name": table_name, "n_uploaded": len(upload)})
        if "raise" in response:
            raise response["raise"]
        return FakeJob(response)

    def delete_user_table(self, table_name):
        self.dropped.append(table_name)


def matched_pairs(upload, *, separation=0.1, skip=()):
    """A plausible result: every uploaded row gets one Gaia source."""
    idx = np.asarray(upload["index"])
    if len(skip):
        idx = idx[~np.isin(idx, skip)]
    return {
        "table": Table(
            {
                "index": idx,
                "source_id": idx + 10_000,
                "ra": np.asarray(upload["ra"])[: len(idx)],
                "dec": np.asarray(upload["dec"])[: len(idx)],
                "angsep_arcsec": np.full(len(idx), separation),
            }
        )
    }


def coordinates(n):
    return np.linspace(10.0, 20.0, n), np.linspace(-5.0, 5.0, n)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(g, "POLL_INTERVAL", 0.0)
    monkeypatch.setattr(g, "RETRY_BACKOFF", 0.0)


def install(monkeypatch, archive):
    monkeypatch.setattr(g, "_gaia", lambda: archive)
    return archive


# --- batching and accounting -------------------------------------------------
def test_all_batches_are_matched_and_reported(monkeypatch):
    ra, dec = coordinates(250)
    archive = install(
        monkeypatch,
        FakeArchive([matched_pairs, matched_pairs, matched_pairs]),
    )

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert len(crossmatch) == 250
    assert crossmatch.meta["n_uploaded"] == 250
    assert crossmatch.meta["n_matched"] == 250
    assert crossmatch.meta["match_fraction"] == 1.0
    assert len(crossmatch.meta["batches"]) == 3
    assert [b["n_uploaded"] for b in crossmatch.meta["batches"]] == [100, 100, 50]
    assert crossmatch.meta["suspect_batches"] == []
    # every uploaded table is removed again, even on success
    assert archive.dropped == [c["table_name"] for c in archive.calls]


def test_non_finite_coordinates_are_skipped_not_uploaded(monkeypatch):
    ra, dec = coordinates(10)
    ra[3] = np.nan

    archive = install(monkeypatch, FakeArchive([matched_pairs]))

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert archive.calls[0]["n_uploaded"] == 9
    assert crossmatch.meta["n_rows"] == 10
    assert crossmatch.meta["n_uploaded"] == 9
    assert 3 not in set(np.asarray(crossmatch["match_idx"]))


def test_duplicate_sources_keep_the_nearest(monkeypatch):
    ra, dec = coordinates(2)

    def two_hits(upload):
        idx = np.asarray(upload["index"])
        return {
            "table": Table(
                {
                    "index": np.repeat(idx, 2),
                    "source_id": np.array([1, 2, 3, 4])[: 2 * len(idx)],
                    "ra": np.zeros(2 * len(idx)),
                    "dec": np.zeros(2 * len(idx)),
                    "angsep_arcsec": np.array([0.4, 0.2, 0.3, 0.1])[: 2 * len(idx)],
                }
            )
        }

    install(monkeypatch, FakeArchive([two_hits]))
    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=10)

    assert len(crossmatch) == 2
    assert list(np.asarray(crossmatch["gdr3_source_id"])) == [2, 4]
    assert crossmatch.meta["batches"][0]["n_multiple"] == 2


# --- failures ---------------------------------------------------------------
def test_aborted_job_is_retried_with_a_fresh_upload_name(monkeypatch):
    ra, dec = coordinates(20)
    archive = install(
        monkeypatch,
        FakeArchive([{"final_phase": "ABORTED"}, matched_pairs]),
    )

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert len(crossmatch) == 20
    assert crossmatch.meta["batches"][0]["attempts"] == 2
    names = [c["table_name"] for c in archive.calls]
    assert len(set(names)) == len(names)  # the retry must not reuse the upload
    assert archive.dropped == names


def test_transport_error_is_retried(monkeypatch):
    ra, dec = coordinates(20)
    archive = install(
        monkeypatch,
        FakeArchive([{"raise": RuntimeError("Error 500: null")}, matched_pairs]),
    )

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert len(crossmatch) == 20
    assert crossmatch.meta["batches"][0]["attempts"] == 2


def test_a_batch_that_never_succeeds_raises(monkeypatch):
    ra, dec = coordinates(20)
    install(
        monkeypatch,
        FakeArchive([{"final_phase": "ERROR", "message": "quota exceeded"}] * g.MAX_ATTEMPTS),
    )

    with pytest.raises(g.CrossmatchError, match="no usable result"):
        g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)


def test_indices_from_another_batch_are_rejected(monkeypatch):
    ra, dec = coordinates(20)

    def wrong_batch(upload):
        pairs = matched_pairs(upload)
        pairs["table"]["index"] = np.asarray(pairs["table"]["index"]) + 1000
        return pairs

    install(monkeypatch, FakeArchive([wrong_batch] * g.MAX_ATTEMPTS))

    with pytest.raises(g.CrossmatchError, match="outside the uploaded range"):
        g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)


def test_a_silently_empty_batch_is_rerun_by_the_audit(monkeypatch):
    ra, dec = coordinates(300)
    empty = {"table": Table({name: np.asarray([], dtype=dtype) for name, dtype in (
        ("index", "i8"), ("source_id", "i8"), ("ra", "f8"), ("dec", "f8"), ("angsep_arcsec", "f8"))})}

    install(monkeypatch, FakeArchive([matched_pairs, empty, matched_pairs, matched_pairs]))

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert crossmatch.meta["n_matched"] == 300
    assert crossmatch.meta["batches"][1]["rerun"] is True
    assert crossmatch.meta["suspect_batches"] == []


def test_a_batch_that_stays_short_is_flagged(monkeypatch):
    ra, dec = coordinates(300)
    empty = {"table": Table({name: np.asarray([], dtype=dtype) for name, dtype in (
        ("index", "i8"), ("source_id", "i8"), ("ra", "f8"), ("dec", "f8"), ("angsep_arcsec", "f8"))})}

    install(monkeypatch, FakeArchive([matched_pairs, empty, matched_pairs, empty]))

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert crossmatch.meta["n_matched"] == 200
    assert crossmatch.meta["suspect_batches"] == [1]


def test_pending_jobs_are_started(monkeypatch):
    """GAVO and ARI leave a job in PENDING until it is explicitly set to RUN."""
    ra, dec = coordinates(10)

    def needs_start(upload):
        response = matched_pairs(upload)
        response["requires_start"] = True
        return response

    archive = install(monkeypatch, FakeArchive([needs_start]))

    crossmatch = g.gaia_crossmatch_batched(ra, dec, max_batch_size=100)

    assert len(crossmatch) == 10
    assert crossmatch.meta["batches"][0]["attempts"] == 1


def test_a_capped_batch_is_split_instead_of_truncated(monkeypatch):
    """A result that fills the row limit is not a result: GAVO answers with 20,001."""
    ra, dec = coordinates(10)
    capped = g.Archive(name="capped", table="x.y", url="http://example.invalid/tap", top=5, maxrec=5)
    fake = FakeArchive([matched_pairs] * 30)  # one pair per uploaded row
    monkeypatch.setattr(g, "_client", lambda archive: fake)

    crossmatch = g.gaia_crossmatch_batched(ra, dec, archive=capped, max_batch_size=100)

    assert crossmatch.meta["n_matched"] == 10  # nothing lost
    assert len(fake.calls) == 7  # 10 -> 5+5 -> (2,3)+(2,3)
    assert crossmatch.meta["batches"][0]["n_matched"] == 10


def test_a_capped_single_row_raises(monkeypatch):
    ra, dec = coordinates(1)

    def too_many(upload):
        pairs = matched_pairs(upload)
        pairs["table"] = astropy_table_vstack([pairs["table"]] * 5)
        return pairs

    capped = g.Archive(name="capped", table="x.y", url="http://example.invalid/tap", top=5, maxrec=5)
    fake = FakeArchive([too_many] * 30)
    monkeypatch.setattr(g, "_client", lambda archive: fake)

    with pytest.raises(g.CrossmatchError, match="cut off"):
        g.gaia_crossmatch_batched(ra, dec, archive=capped, max_batch_size=100)


def test_the_archive_table_and_top_reach_the_query():
    query = g._pairs_query("u1", 0.000139, "distance", "gaia.dr3lite", 10_000_000)
    assert "TOP 10000000" in query
    assert "gaia.dr3lite" in query
    assert "DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', t.ra, t.dec))" in query


def test_unknown_archive_name_is_reported():
    with pytest.raises(ValueError, match="unknown Gaia archive"):
        g.archive_by_name("nope")


# --- cache guard ------------------------------------------------------------
def test_check_regression_rejects_a_smaller_match():
    previous = {"n_matched": 1000, "created_utc": "2026-11-01T00:00:00+00:00"}
    crossmatch = Table({"match_idx": np.arange(900)})
    crossmatch.meta.update({"n_matched": 900, "n_uploaded": 1200})

    with pytest.raises(g.CrossmatchError, match="fewer"):
        g.check_regression(previous, crossmatch, key="demo")


def test_check_regression_tolerates_rounding():
    previous = {"n_matched": 1000}
    crossmatch = Table({"match_idx": np.arange(999)})
    crossmatch.meta.update({"n_matched": 999, "n_uploaded": 1200})

    g.check_regression(previous, crossmatch, key="demo")  # 0.1 % < 1 %


def test_check_regression_rebaselines_when_the_search_radius_changes():
    previous = {"n_matched": 334, "max_separation_arcsec": 1.5}
    crossmatch = Table({"match_idx": np.arange(298)})
    crossmatch.meta.update({"n_matched": 298, "n_uploaded": 385, "max_separation_arcsec": 1.0})

    g.check_regression(previous, crossmatch, key="demo")  # deliberate: 1.5 -> 1.0 arcsec


def test_check_regression_still_fires_at_the_same_radius():
    previous = {"n_matched": 334, "max_separation_arcsec": 1.0}
    crossmatch = Table({"match_idx": np.arange(298)})
    crossmatch.meta.update({"n_matched": 298, "n_uploaded": 385, "max_separation_arcsec": 1.0})

    with pytest.raises(g.CrossmatchError, match="incomplete"):
        g.check_regression(previous, crossmatch, key="demo")


def test_crossmatch_catalog_keeps_the_old_cache_when_the_result_shrinks(monkeypatch, tmp_path):
    catalog = Table({"RAdeg": np.linspace(10, 20, 10), "DEdeg": np.linspace(-5, 5, 10)})
    full = install(monkeypatch, FakeArchive([matched_pairs]))
    cache = tmp_path / "crossmatched.fits"

    matched = g.crossmatch_catalog(
        lambda: catalog,
        ra_col="RAdeg",
        dec_col="DEdeg",
        cache_path=cache,
        max_separation=0.5 * units.arcsec,
        key="demo",
    )
    assert len(matched) == 10
    assert cache.exists()
    report = g.read_report(tmp_path)
    assert report["n_matched"] == 10
    assert full.dropped  # upload cleaned up

    # the archive now returns only part of the catalog: the guard must refuse
    install(monkeypatch, FakeArchive([lambda upload: matched_pairs(upload, skip=(0, 1, 2, 3, 4))]))
    with pytest.raises(g.CrossmatchError, match="incomplete"):
        g.crossmatch_catalog(
            lambda: catalog,
            ra_col="RAdeg",
            dec_col="DEdeg",
            cache_path=cache,
            max_separation=0.5 * units.arcsec,
            key="demo",
            force=True,
        )
    assert len(Table.read(cache)) == 10  # untouched
    assert g.read_report(tmp_path)["n_matched"] == 10

    # ... unless the operator says so
    install(monkeypatch, FakeArchive([lambda upload: matched_pairs(upload, skip=(0, 1, 2, 3, 4))]))
    accepted = g.crossmatch_catalog(
        lambda: catalog,
        ra_col="RAdeg",
        dec_col="DEdeg",
        cache_path=cache,
        max_separation=0.5 * units.arcsec,
        key="demo",
        force=True,
        allow_regression=True,
    )
    assert len(accepted) == 5
    assert g.read_report(tmp_path)["n_matched"] == 5


def test_a_wider_match_radius_needs_no_new_query(monkeypatch, tmp_path):
    """The archive is searched once, wide; the catalog limit is applied afterwards."""
    catalog = Table({"RAdeg": np.linspace(10, 20, 4), "DEdeg": np.linspace(-5, 5, 4)})
    cache = tmp_path / "crossmatched.fits"

    def pairs_at(upload, separations):
        idx = np.asarray(upload["index"])
        return {
            "table": Table(
                {
                    "index": idx,
                    "source_id": idx + 100,
                    "ra": np.zeros(len(idx)),
                    "dec": np.zeros(len(idx)),
                    "angsep_arcsec": np.asarray(separations, dtype="f8")[: len(idx)],
                }
            )
        }

    archive = install(
        monkeypatch, FakeArchive([lambda upload: pairs_at(upload, [0.3, 0.4, 0.9, 1.2])])
    )
    narrow = g.crossmatch_catalog(
        lambda: catalog,
        ra_col="RAdeg",
        dec_col="DEdeg",
        cache_path=cache,
        search_separation=1.5 * units.arcsec,
        max_separation=0.5 * units.arcsec,
        key="demo",
    )
    assert len(narrow) == 2  # only the 0.3 and 0.4 arcsec matches
    assert "angsep_arcsec" in Table.read(cache).colnames
    assert len(Table.read(cache)) == 4  # every pair inside the 1.5 arcsec search
    assert len(archive.calls) == 1

    # widen the catalog's limit: the cache answers, the archive is not touched
    wider = g.crossmatch_catalog(
        lambda: (_ for _ in ()).throw(AssertionError("no new query")),
        ra_col="RAdeg",
        dec_col="DEdeg",
        cache_path=cache,
        search_separation=1.5 * units.arcsec,
        max_separation=1.0 * units.arcsec,
        key="demo",
    )
    assert len(wider) == 3
    assert len(archive.calls) == 1


def test_a_limit_wider_than_the_search_is_refused(monkeypatch, tmp_path):
    catalog = Table({"RAdeg": [10.0], "DEdeg": [0.0]})
    install(monkeypatch, FakeArchive([]))

    with pytest.raises(ValueError, match="only searched to"):
        g.crossmatch_catalog(
            lambda: catalog,
            ra_col="RAdeg",
            dec_col="DEdeg",
            cache_path=tmp_path / "c.fits",
            search_separation=0.5 * units.arcsec,
            max_separation=1.0 * units.arcsec,
        )


def test_crossmatch_catalog_does_not_touch_the_archive_on_a_cache_hit(monkeypatch, tmp_path):
    cache = tmp_path / "crossmatched.fits"
    Table({"gdr3_source_id": [1, 2], "x": [0.0, 1.0]}).write(cache)

    def explode():
        raise AssertionError("the raw catalog should not be read on a cache hit")

    archive = install(monkeypatch, FakeArchive([]))
    out = g.crossmatch_catalog(
        explode, ra_col="RAdeg", dec_col="DEdeg", cache_path=cache
    )

    assert len(out) == 2
    assert archive.calls == []
