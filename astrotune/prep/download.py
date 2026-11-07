"""Cached downloads for the input catalogs."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from getpass import getpass
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    auth: tuple[str, str] | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` unless it already exists.

    The download is written to a temporary file first, so an interrupted run
    never leaves a truncated catalog behind.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        log.info("Using cached %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, auth=auth, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            disable=None,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))

    shutil.move(tmp, dest)
    return dest


def cosmodb_download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Download a VizieR/CDS catalog (``nph-Cat/fits.gz`` style URLs)."""
    return download(url, dest, overwrite=overwrite)


def sdss_credentials() -> tuple[str, str]:
    """SDSS-V credentials, taken from the environment when available."""
    user = os.environ.get("SDSS_USER")
    password = os.environ.get("SDSS_PASSWORD")
    if not user:
        user = input("SDSS username: ")
    if not password:
        password = getpass("SDSS password: ")
    return user, password


def download_sdss(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Download a file that requires SDSS-V authentication."""
    if Path(dest).exists() and not overwrite:
        log.info("Using cached %s", dest)
        return Path(dest)
    return download(url, dest, auth=sdss_credentials(), overwrite=overwrite)


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name
