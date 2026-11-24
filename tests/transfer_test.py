import json
import os

import pytest
from unittest.mock import Mock

from colab_mcp import transfer


def _run(code):
    """Execute generated kernel code locally and capture stdout like a kernel."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, {})
    return [{"output_type": "stream", "name": "stdout", "text": buffer.getvalue()}]


class FakeRuntime:
    async def execute(self, code, _callback):
        return _run(code)


def test_stdout_text_joins_streams_and_raises_on_error():
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "ab"},
        {"output_type": "stream", "name": "stderr", "text": "ignored"},
        {"output_type": "stream", "name": "stdout", "text": ["c", "d"]},
    ]
    assert transfer.stdout_text(outputs) == "abcd"
    with pytest.raises(transfer.TransferError, match="FileNotFoundError"):
        transfer.stdout_text([{"output_type": "error", "ename": "FileNotFoundError", "evalue": "x"}])


def test_build_notebook_cells_and_gpu():
    nb = transfer.build_notebook(
        [{"type": "markdown", "source": "# Hi"}, {"source": "print(1)\nprint(2)"}], "t4"
    )
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code"]
    assert nb["cells"][1]["source"] == ["print(1)\n", "print(2)"]
    assert nb["cells"][1]["outputs"] == []
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    assert "accelerator" not in transfer.build_notebook([], "NONE")["metadata"]
    with pytest.raises(ValueError):
        transfer.build_notebook([{"type": "raw", "source": ""}])


def test_multipart_body_contains_metadata_and_content():
    body = transfer.multipart_body({"name": "a.ipynb"}, b'{"cells": []}', "B")
    assert body.startswith(b"--B\r\n") and body.endswith(b"--B--\r\n")
    assert b'"name": "a.ipynb"' in body and b'{"cells": []}' in body


@pytest.mark.asyncio
async def test_upload_and_download_round_trip(tmp_path, monkeypatch):
    import colab_mcp

    monkeypatch.setattr(transfer, "CHUNK_BYTES", 1000)
    manager = Mock(connected=True, runtime=FakeRuntime())
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)

    data = os.urandom(2500)
    source = tmp_path / "in.bin"
    source.write_bytes(data)
    remote = tmp_path / "remote" / "copy.bin"
    result = await colab_mcp.upload_to_runtime.fn(str(source), str(remote))
    assert "2500 bytes" in result and remote.read_bytes() == data

    back = tmp_path / "back.bin"
    result = await colab_mcp.download_from_runtime.fn(str(remote), str(back))
    assert "2500 bytes" in result and back.read_bytes() == data

    missing = await colab_mcp.download_from_runtime.fn(str(tmp_path / "nope"), str(back))
    assert "Remote file not found" in missing


@pytest.mark.asyncio
async def test_upload_empty_file(tmp_path, monkeypatch):
    import colab_mcp

    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", Mock(connected=True, runtime=FakeRuntime()))
    source = tmp_path / "empty"
    source.write_bytes(b"")
    remote = tmp_path / "r" / "empty"
    assert "0 bytes" in await colab_mcp.upload_to_runtime.fn(str(source), str(remote))
    assert remote.read_bytes() == b""


@pytest.mark.asyncio
async def test_transfer_requires_oauth(monkeypatch):
    import colab_mcp

    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", None)
    assert "client-oauth-config" in await colab_mcp.upload_to_runtime.fn("/etc/hostname")


@pytest.mark.asyncio
async def test_list_runtimes_reports_connection_and_units(monkeypatch):
    import colab_mcp

    a = Mock(endpoint="gpu-a", accelerator=Mock(value="T4"))
    a.variant.name, a.machine_shape.name = "GPU", "STANDARD"
    client = Mock()
    client.list_assignments.return_value = [a]
    client.get_ccu_info.return_value = Mock(current_balance=12.5, consumption_rate_hourly=1.8)
    monkeypatch.setattr(colab_mcp, "_colab_client", client)
    from colab_mcp.runtime import DirectRuntimeManager

    manager = DirectRuntimeManager(client)
    manager.runtime, manager.endpoint = Mock(connected=True), "gpu-a"
    manager.note_assigned("gpu-a", "T4", 1800)
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)
    data = json.loads(await colab_mcp.list_runtimes.fn())
    assert data["runtimes"][0]["direct_connected"] is True
    assert data["runtimes"][0]["idle_timeout_sec"] == 1800
    assert "every 5 min" in data["keep_alive"]
    assert data["compute_units"] == {"balance": 12.5, "consumption_per_hour": 1.8}


@pytest.mark.asyncio
async def test_create_notebook_uploads_to_drive(monkeypatch):
    import colab_mcp
    from colab_mcp import auth

    session = Mock()
    session.post.return_value = Mock(status_code=200, json=lambda: {"id": "abc"})
    monkeypatch.setattr(colab_mcp, "_client_oauth_config", "/x.json")
    monkeypatch.setattr(auth, "get_drive_session", lambda _config: session)
    data = json.loads(
        await colab_mcp.create_notebook.fn("Exp", [{"type": "code", "source": "1"}], "T4", "fold")
    )
    assert data == {"id": "abc", "name": "Exp.ipynb", "url": "https://colab.research.google.com/drive/abc"}
    body = session.post.call_args.kwargs["data"]
    assert b'"parents": ["fold"]' in body and transfer.COLAB_MIME_TYPE.encode() in body


@pytest.fixture
def fake_runtime(monkeypatch):
    import colab_mcp

    monkeypatch.setattr(transfer, "CHUNK_BYTES", 700)
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", Mock(connected=True, runtime=FakeRuntime()))
    return colab_mcp


def _tree(tmp_path):
    src = tmp_path / "src"
    (src / "data" / "sub").mkdir(parents=True)
    (src / "data" / "a.txt").write_text("alpha")
    (src / "data" / "sub" / "b.bin").write_bytes(os.urandom(3000))
    (src / "x.png").write_bytes(b"png1")
    (src / "y.png").write_bytes(b"png2")
    return src


@pytest.mark.asyncio
async def test_batch_upload_dirs_and_globs(tmp_path, fake_runtime):
    src = _tree(tmp_path)
    remote = tmp_path / "remote"
    data = json.loads(
        await fake_runtime.upload_files_to_runtime.fn([str(src / "data"), str(src / "*.png")], str(remote))
    )
    assert data["uploaded"] == ["data", "x.png", "y.png"]
    assert (remote / "data" / "sub" / "b.bin").read_bytes() == (src / "data" / "sub" / "b.bin").read_bytes()
    assert (remote / "y.png").read_bytes() == b"png2"
    assert not any(p.name.startswith("colab-mcp-upload") for p in __import__("pathlib").Path("/tmp").iterdir())


@pytest.mark.asyncio
async def test_batch_upload_reports_missing_local(tmp_path, fake_runtime):
    result = await fake_runtime.upload_files_to_runtime.fn([str(tmp_path / "nope*.txt")], str(tmp_path))
    assert "Local path not found" in result


@pytest.mark.asyncio
async def test_batch_download_with_missing_pattern(tmp_path, fake_runtime):
    src = _tree(tmp_path)
    out = tmp_path / "out"
    data = json.loads(
        await fake_runtime.download_files_from_runtime.fn(
            [str(src / "*.png"), str(src / "data"), str(src / "nothing*")], str(out)
        )
    )
    assert data["downloaded"] == ["data", "x.png", "y.png"]
    assert data["missing"] == [str(src / "nothing*")]
    assert (out / "data" / "a.txt").read_text() == "alpha"
    none = await fake_runtime.download_files_from_runtime.fn([str(src / "zzz*")], str(out))
    assert "No remote files matched" in none


@pytest.mark.asyncio
async def test_list_and_delete_runtime_files(tmp_path, fake_runtime):
    src = _tree(tmp_path)
    flat = json.loads(await fake_runtime.list_runtime_files.fn(str(src)))
    assert [os.path.basename(e["path"]) for e in flat["entries"]] == ["data", "x.png", "y.png"]
    assert flat["entries"][0]["type"] == "dir" and flat["entries"][1]["size"] == 4
    deep = json.loads(await fake_runtime.list_runtime_files.fn(str(src), recursive=True, limit=3))
    assert len(deep["entries"]) == 3 and deep["truncated"] is True
    assert "not found" in await fake_runtime.list_runtime_files.fn(str(tmp_path / "missing"))

    report = json.loads(
        await fake_runtime.delete_runtime_files.fn([str(src / "*.png"), str(src / "data"), str(src / "q")])
    )
    assert len(report["deleted"]) == 3 and report["missing"] == [str(src / "q")]
    assert os.listdir(src) == []
