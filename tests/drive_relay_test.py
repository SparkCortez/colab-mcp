# Added to this local copy of the fork, 2026.
# Licensed under the Apache License, Version 2.0.

import json
import os
from types import SimpleNamespace

import pytest
import requests

from colab_mcp import drive_relay


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.text = body.decode(errors="replace") if isinstance(body, bytes) else body

    def json(self):
        return json.loads(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDrive:
    """In-memory Drive answering the few calls the relay makes."""

    def __init__(self):
        self.files = {}
        self.sessions = {}
        self.credentials = SimpleNamespace(valid=True, token="tok")

    def _check(self, headers):
        auth = (headers or {}).get("Authorization")
        assert auth in (None, "Bearer tok")

    def post(self, url, params=None, headers=None, data=None, timeout=None):
        self._check(headers)
        assert url == drive_relay.UPLOAD_URL and params["uploadType"] == "resumable"
        sid = f"s{len(self.sessions)}"
        self.sessions[sid] = json.loads(data)["name"]
        return FakeResponse(headers={"Location": f"https://upload/{sid}"})

    def put(self, url, data=None, headers=None, timeout=None):
        self._check(headers)
        body = data.read() if hasattr(data, "read") else data
        assert int(headers["Content-Length"]) == len(body)
        fid = f"id{len(self.files)}"
        self.files[fid] = (self.sessions[url.rsplit("/", 1)[1]], body)
        return FakeResponse(body=json.dumps({"id": fid}).encode())

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self._check(headers)
        fid = url.rsplit("/", 1)[1]
        if fid not in self.files:
            return FakeResponse(404, b"not found")
        return FakeResponse(body=self.files[fid][1])

    def delete(self, url, timeout=None):
        self.files.pop(url.rsplit("/", 1)[1], None)
        return FakeResponse(204)


@pytest.fixture
def drive(monkeypatch):
    fake = FakeDrive()
    # Kernel code imports requests inside the "runtime"; route it to the fake.
    for name in ("get", "post", "put"):
        monkeypatch.setattr(requests, name, getattr(fake, name))
    return fake


class FakeRuntime:
    async def execute(self, code, _callback):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, {})
        return [{"output_type": "stream", "name": "stdout", "text": buffer.getvalue()}]


@pytest.mark.asyncio
async def test_push_and_pull_through_drive_cleans_up(tmp_path, drive):
    data = os.urandom(300_000)
    src = tmp_path / "video.mp4"
    src.write_bytes(data)
    remote = tmp_path / "remote" / "video.mp4"

    assert await drive_relay.push_file(FakeRuntime(), drive, str(src), str(remote)) == len(data)
    assert remote.read_bytes() == data
    assert drive.files == {}

    back = tmp_path / "back" / "out.mp4"
    assert await drive_relay.pull_file(FakeRuntime(), drive, str(remote), str(back)) == len(data)
    assert back.read_bytes() == data
    assert drive.files == {}
    assert all(name.startswith("colab-mcp-relay-") for name in drive.sessions.values())


@pytest.mark.asyncio
async def test_failed_runtime_fetch_still_deletes_relay_file(tmp_path, drive):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 10)

    class Broken(FakeRuntime):
        async def execute(self, code, _callback):
            return [{"output_type": "error", "ename": "ConnectionError", "evalue": "boom"}]

    with pytest.raises(Exception, match="ConnectionError"):
        await drive_relay.push_file(Broken(), drive, str(src), "/content/a.bin")
    assert drive.files == {}


@pytest.fixture
def tools(monkeypatch, drive):
    import colab_mcp

    runtime = FakeRuntime()

    async def connected():
        return runtime

    monkeypatch.setattr(colab_mcp, "_connected_direct_runtime", connected)
    monkeypatch.setattr(drive_relay, "DRIVE_THRESHOLD_BYTES", 1000)
    return colab_mcp


@pytest.mark.asyncio
async def test_auto_uses_drive_for_large_files_when_granted(tmp_path, tools, drive, monkeypatch):
    monkeypatch.setattr(tools, "_drive_session_for_transfer", lambda force: drive)
    big, small = tmp_path / "big.bin", tmp_path / "small.bin"
    big.write_bytes(os.urandom(5000))
    small.write_bytes(b"tiny")
    result = await tools.upload_to_runtime.fn(str(big), str(tmp_path / "r" / "big.bin"))
    assert "via Google Drive" in result
    result = await tools.upload_to_runtime.fn(str(small), str(tmp_path / "r" / "small.bin"))
    assert "via kernel" in result
    result = await tools.download_from_runtime.fn(str(tmp_path / "r" / "big.bin"), str(tmp_path / "d.bin"))
    assert "via Google Drive" in result
    assert (tmp_path / "d.bin").read_bytes() == big.read_bytes()


@pytest.mark.asyncio
async def test_auto_falls_back_to_kernel_without_drive_access(tmp_path, tools, monkeypatch):
    monkeypatch.setattr(tools, "_drive_session_for_transfer", lambda force: None)
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(5000))
    result = await tools.upload_to_runtime.fn(str(big), str(tmp_path / "r.bin"))
    assert "via kernel" in result and "via='drive'" in result
    assert (tmp_path / "r.bin").read_bytes() == big.read_bytes()


@pytest.mark.asyncio
async def test_auto_falls_back_when_drive_fails_but_forced_drive_reports(tmp_path, tools, monkeypatch):
    def broken(force):
        raise RuntimeError("drive down")

    monkeypatch.setattr(tools, "_drive_session_for_transfer", broken)
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(5000))
    result = await tools.upload_to_runtime.fn(str(big), str(tmp_path / "r.bin"))
    assert "fell back to the kernel" in result
    result = await tools.upload_to_runtime.fn(str(big), str(tmp_path / "r2.bin"), via="drive")
    assert result.startswith("Failed to upload") and "drive down" in result
    assert "via must be" in await tools.upload_to_runtime.fn(str(big), via="fast")


@pytest.mark.asyncio
async def test_batch_upload_reports_route(tmp_path, tools, drive, monkeypatch):
    monkeypatch.setattr(tools, "_drive_session_for_transfer", lambda force: drive)
    folder = tmp_path / "clips"
    folder.mkdir()
    (folder / "a.bin").write_bytes(os.urandom(20000))
    result = json.loads(
        await tools.upload_files_to_runtime.fn([str(folder)], str(tmp_path / "remote"))
    )
    assert "Google Drive" in result["route"]
    assert (tmp_path / "remote" / "clips" / "a.bin").exists()
