"""Offline tests for anonymous Kaggle access and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from blastnet.cli import main
from blastnet.data import kaggle


class Response:
    def __init__(self, body: bytes, headers: dict | None = None, status: int = 200):
        self.body = body
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self, size: int = -1):
        if size < 0:
            size = len(self.body)
        body, self.body = self.body[:size], self.body[size:]
        return body

    def getcode(self):
        return self.status


def json_response(body):
    return Response(json.dumps(body).encode())


def test_list_files_follows_tokens(monkeypatch):
    urls = []
    pages = [
        {"datasetFiles": [{"name": "a.dat", "totalBytes": 4}], "nextPageToken": "next token"},
        {"datasetFiles": [{"name": "b.dat", "totalBytes": 7}], "nextPageToken": ""},
    ]

    def open_url(url, **kwargs):
        urls.append(str(url))
        return json_response(pages.pop(0))

    monkeypatch.setattr(kaggle, "_urlopen", open_url)
    assert kaggle.list_files("owner/dataset") == [
        {"name": "a.dat", "size": 4}, {"name": "b.dat", "size": 7}
    ]
    assert urls == [
        f"{kaggle.API}/datasets/list/owner/dataset",
        f"{kaggle.API}/datasets/list/owner/dataset?pageToken=next+token",
    ]


def test_download_direct_response(tmp_path, monkeypatch):
    payload = b"complete archive"
    monkeypatch.setattr(kaggle, "_urlopen", lambda *a, **k: Response(
        payload, {"Content-Length": str(len(payload))}))
    target = kaggle.download("owner/public-set", tmp_path)
    assert target.name == "public-set.zip"
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + ".part").exists()


def test_download_resumes_with_content_range(tmp_path, monkeypatch):
    target = tmp_path / "field.dat.part"
    target.write_bytes(b"first-")
    calls = []

    def open_url(request, **kwargs):
        calls.append((request.full_url, request.get_header("Range")))
        return Response(b"second", {"Content-Range": "bytes 6-11/12", "Content-Length": "6"}, 206)

    monkeypatch.setattr(kaggle, "_urlopen", open_url)
    result = kaggle.download("owner/dataset", tmp_path, file="dir/field.dat")
    assert result.read_bytes() == b"first-second"
    assert calls == [(f"{kaggle.API}/datasets/download/owner/dataset/dir/field.dat", "bytes=6-")]


def test_download_restarts_when_range_is_ignored(tmp_path, monkeypatch):
    target = tmp_path / "field.dat.part"
    target.write_bytes(b"stale")
    monkeypatch.setattr(kaggle, "_urlopen", lambda *a, **k: Response(
        b"fresh data", {"Content-Length": "10"}, 200))
    result = kaggle.download("owner/dataset", tmp_path, file="field.dat")
    assert result.read_bytes() == b"fresh data"


def test_download_retries_and_reissues_api_request(tmp_path, monkeypatch):
    calls = []

    def open_url(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError("temporary")
        return Response(b"ok", {"Content-Length": "2"})

    delays = []
    monkeypatch.setattr(kaggle, "_urlopen", open_url)
    monkeypatch.setattr(kaggle, "_sleep", delays.append)
    assert kaggle.download("owner/dataset", tmp_path).read_bytes() == b"ok"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert delays == [1]


def test_download_size_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "_urlopen", lambda *a, **k: Response(
        b"short", {"Content-Length": "10"}))
    with pytest.raises(IOError, match="size mismatch"):
        kaggle.download("owner/dataset", tmp_path)
    assert (tmp_path / "dataset.zip.part").exists()
    assert not (tmp_path / "dataset.zip").exists()


def test_catalog_deduplicates_refs(monkeypatch):
    pages = [
        {"datasets": [{"ref": "a/one", "title": "One", "totalBytes": 4,
                       "lastUpdated": "2026-01-01"}]},
        {"datasets": [{"ref": "a/one", "title": "duplicate"},
                       {"ref": "b/two", "title": "Two", "totalBytes": 8,
                        "lastUpdated": "2026-02-01"}]},
        {"datasets": []},
    ]
    monkeypatch.setattr(kaggle, "_urlopen", lambda *a, **k: json_response(pages.pop(0)))
    result = kaggle.catalog("fluid flow")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["ref", "title", "totalBytes", "lastUpdated", "url"]
    assert result["ref"].tolist() == ["a/one", "b/two"]
    assert result.loc[0, "url"] == "https://www.kaggle.com/datasets/a/one"


def test_download_cli(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(kaggle, "download", lambda *a, **k: calls.append((a, k)) or Path("out.zip"))
    assert main(["download", "owner/set", "--out", str(tmp_path), "--file", "x.dat"]) == 0
    assert calls == [(("owner/set", str(tmp_path)), {"file": "x.dat"})]
    assert capsys.readouterr().out.strip() == "out.zip"


def test_catalog_cli_csv(monkeypatch, tmp_path, capsys):
    frame = pd.DataFrame([{"ref": "a/one", "title": "One"}])
    calls = []
    monkeypatch.setattr(kaggle, "catalog", lambda search: calls.append(search) or frame)
    path = tmp_path / "catalog.csv"
    assert main(["catalog", "--search", "fluid", "--csv", str(path)]) == 0
    assert calls == ["fluid"]
    assert "a/one" in path.read_text()
    assert capsys.readouterr().out == ""


def test_catalog_cli_prints_table(monkeypatch, capsys):
    frame = pd.DataFrame([{"ref": "a/one", "title": "One"}])
    monkeypatch.setattr(kaggle, "catalog", lambda search: frame)
    assert main(["catalog"]) == 0
    assert "a/one" in capsys.readouterr().out
