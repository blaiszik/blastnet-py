"""Anonymous Kaggle dataset listing, downloads, and catalog search."""

from __future__ import annotations

import http.client
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

MOMENTUM128 = "waitongchung/blastnet-momentum-3d-sr-dataset"
API = "https://www.kaggle.com/api/v1"
_urlopen = urlopen
_sleep = time.sleep


def _json(url: str, timeout: float) -> dict | list:
    with _urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def list_files(slug: str) -> list[dict]:
    """List every file in a public Kaggle dataset, following pagination tokens."""
    url = f"{API}/datasets/list/{slug}"
    token = None
    files = []
    while True:
        page_url = f"{url}?{urlencode({'pageToken': token})}" if token else url
        try:
            payload = _json(page_url, 60.0)
        except HTTPError as err:
            if err.code != 404:
                raise
            # Observed 2026-10-10: this endpoint answered anonymously, then returned 404 for both
            # the first page and pageToken pages, while download and search kept working.
            raise RuntimeError(
                f"Kaggle's anonymous file listing for {slug!r} returned 404 (it intermittently "
                "refuses unauthenticated clients); download() and catalog() are unaffected, and "
                "`unzip -l` on the downloaded archive lists its contents") from err
        files.extend({"name": entry["name"], "size": entry["totalBytes"]}
                     for entry in payload.get("datasetFiles", []))
        token = payload.get("nextPageToken")
        if not token:
            return files


def _header(headers, key: str) -> str | None:
    value = headers.get(key)
    if value is None:
        value = headers.get(key.lower())
    return str(value) if value is not None else None


def _expected_size(headers, status: int, existing: int) -> int | None:
    content_range = _header(headers, "Content-Range")
    if content_range:
        match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+|\*)", content_range.strip())
        if match and match.group(1) != "*":
            return int(match.group(1))
    length = _header(headers, "Content-Length")
    if length is not None:
        return int(length) + (existing if status == 206 else 0)
    return None


def download(slug: str, out: str | Path, file: str | None = None, resume: bool = True,
             chunk: int = 8 << 20, timeout: float = 60.0, *, progress: bool = False) -> Path:
    """Download a public Kaggle dataset or one file, atomically replacing its target."""
    target_name = Path(file).name if file else f"{slug.rsplit('/', 1)[-1]}.zip"
    target_dir = Path(out)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / target_name
    partial = target.with_name(f"{target.name}.part")
    api_url = f"{API}/datasets/download/{slug}"
    if file:
        api_url += f"/{file}"

    for attempt in range(5):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = Request(api_url, headers=headers)
            with _urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                response_headers = response.headers
                # A server may ignore Range. A 200 response starts a fresh file.
                append = bool(offset and status == 206)
                expected = _expected_size(response_headers, status, offset if append else 0)
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while True:
                        try:
                            block = response.read(chunk)
                        except http.client.IncompleteRead as exc:
                            if exc.partial:
                                stream.write(exc.partial)
                            raise
                        if not block:
                            break
                        stream.write(block)
                        if progress:
                            print(f"Downloaded {partial.stat().st_size} bytes", end="\r", flush=True)
            actual = partial.stat().st_size
            if expected is not None and actual != expected:
                raise OSError(f"download size mismatch: expected {expected} bytes, received {actual}")
            partial.replace(target)
            if progress:
                print(f"Downloaded {target.name} ({actual} bytes)")
            return target
        except (URLError, http.client.IncompleteRead):
            if attempt == 4:
                raise
            _sleep(2 ** attempt)
    raise AssertionError("unreachable")


def catalog(search: str = "blastnet", max_pages: int = 20) -> pd.DataFrame:
    """Search Kaggle's public dataset catalog and return unique dataset rows."""
    rows = []
    seen = set()
    for page in range(1, max_pages + 1):
        url = f"{API}/datasets/list?{urlencode({'search': search, 'page': page})}"
        payload = _json(url, 60.0)
        datasets = payload if isinstance(payload, list) else payload.get("datasets", [])
        if not datasets:
            break
        for dataset in datasets:
            ref = dataset.get("ref")
            if ref in seen:
                continue
            seen.add(ref)
            rows.append({
                "ref": ref,
                "title": dataset.get("title"),
                "totalBytes": dataset.get("totalBytes"),
                "lastUpdated": dataset.get("lastUpdated"),
                "url": f"https://www.kaggle.com/datasets/{ref}",
            })
    return pd.DataFrame(rows, columns=["ref", "title", "totalBytes", "lastUpdated", "url"])
