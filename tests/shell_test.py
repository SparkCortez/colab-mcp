"""run_shell / shell_job, with the generated kernel code executed locally."""

import asyncio
import io
import json
import time
from contextlib import redirect_stdout

import pytest

from colab_mcp import shell


class LocalRuntime:
    """Runs kernel code in this process, like a kernel would."""

    def __init__(self):
        self.timeouts = []

    async def execute(self, code, _callback, timeout=None):
        self.timeouts.append(timeout)

        def run():
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code, {})
            return [{"output_type": "stream", "name": "stdout", "text": buf.getvalue()}]

        return await asyncio.to_thread(run)


@pytest.fixture
def tools(monkeypatch, tmp_path):
    import colab_mcp

    runtime = LocalRuntime()

    async def connected(runtime_endpoint=""):
        return runtime

    monkeypatch.setattr(colab_mcp, "_connected_direct_runtime", connected)
    monkeypatch.setattr(shell, "JOBS_DIR", str(tmp_path / "jobs"))
    return colab_mcp


@pytest.mark.asyncio
async def test_foreground_returns_streams_and_exit_code(tools, tmp_path):
    out = json.loads(await tools.run_shell.fn(
        "echo hi; echo oops >&2; pwd; exit 3", cwd=str(tmp_path)
    ))
    assert out["stdout"] == f"hi\n{tmp_path}\n"
    assert out["stderr"] == "oops\n"
    assert out["exit_code"] == 3
    assert out["timed_out"] is False


@pytest.mark.asyncio
async def test_foreground_timeout_kills_process_group(tools, tmp_path):
    marker = tmp_path / "late"
    started = time.time()
    out = json.loads(await tools.run_shell.fn(
        f"echo start; (sleep 3; touch {marker}) & sleep 30", timeout_seconds=1, cwd=str(tmp_path)
    ))
    assert time.time() - started < 10
    assert out["timed_out"] is True
    assert out["stdout"] == "start\n"
    await asyncio.sleep(3.5)
    assert not marker.exists(), "the backgrounded child of the command must be killed too"


@pytest.mark.asyncio
async def test_output_is_tail_truncated(tools, tmp_path):
    out = json.loads(await tools.run_shell.fn(
        "python3 -c \"print('x' * 30000)\"", cwd=str(tmp_path)
    ))
    assert len(out["stdout"]) == shell.MAX_OUTPUT_CHARS
    assert out["stdout_truncated_chars"] == 30001 - shell.MAX_OUTPUT_CHARS


@pytest.mark.asyncio
async def test_background_job_lifecycle(tools, tmp_path):
    started = json.loads(await tools.run_shell.fn(
        "echo one; sleep 0.5; echo two; exit 7", cwd=str(tmp_path), background=True
    ))
    job_id = started["job_id"]
    status = json.loads(await tools.shell_job.fn(job_id))
    assert status["state"] == "running"
    for _ in range(50):
        await asyncio.sleep(0.1)
        status = json.loads(await tools.shell_job.fn(job_id, tail_lines=1))
        if status["state"] != "running":
            break
    assert status["state"] == "exited"
    assert status["exit_code"] == 7
    assert status["tail"] == "two"
    listed = json.loads(await tools.shell_job.fn(action="list"))
    assert [j["job_id"] for j in listed] == [job_id]


@pytest.mark.asyncio
async def test_kill_stops_job(tools, tmp_path):
    job_id = json.loads(
        await tools.run_shell.fn("sleep 60", cwd=str(tmp_path), background=True)
    )["job_id"]
    await asyncio.sleep(0.3)
    killed = json.loads(await tools.shell_job.fn(job_id, action="kill"))
    assert killed["state"] == "killed"
    assert killed["exit_code"] is None


@pytest.mark.asyncio
async def test_bad_arguments(tools):
    assert await tools.run_shell.fn("  ") == "command is empty."
    assert "job_id is required" in await tools.shell_job.fn()
    assert "action must be" in await tools.shell_job.fn("x", action="nope")
    assert "unknown job" in await tools.shell_job.fn("nope")
