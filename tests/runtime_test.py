# Copyright 2026 Atsushi Onozawa.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import pytest

from colab_mcp.runtime import (
    DirectColabRuntime,
    DirectRuntimeError,
    DirectRuntimeManager,
    RuntimeTarget,
)


class FakeKernelClient:
    def execute_interactive(self, code, *, output_hook, allow_stdin, timeout):
        assert code == "print('hello')"
        assert allow_stdin is False
        assert timeout > 0
        output_hook(
            {"output_type": "stream", "name": "stdout", "text": "hello\n"}
        )
        return {"content": {"status": "ok"}}


@pytest.mark.asyncio
async def test_direct_runtime_forwards_incremental_kernel_outputs(monkeypatch):
    import jupyter_kernel_client.client

    monkeypatch.setattr(
        jupyter_kernel_client.client,
        "output_hook",
        lambda outputs, message: (outputs.append(message) or [len(outputs) - 1]),
    )
    runtime = DirectColabRuntime(
        RuntimeTarget(endpoint="ep", url="https://runtime", token="token")
    )
    runtime._kernel_client = FakeKernelClient()
    received = []

    outputs = await runtime.execute("print('hello')", received.append)

    assert outputs == [
        {"output_type": "stream", "name": "stdout", "text": "hello\n"}
    ]
    assert received == outputs


@pytest.mark.asyncio
async def test_direct_runtime_manager_selects_single_active_assignment(monkeypatch):
    assignment = SimpleNamespace(
        endpoint="endpoint-1",
        runtime_proxy_info=SimpleNamespace(url="https://runtime", token="token"),
    )

    class FakeColabClient:
        def list_assignments(self):
            return [assignment]

    connected = []

    async def fake_connect(self):
        connected.append(self.target)

    monkeypatch.setattr(DirectColabRuntime, "connect", fake_connect)
    manager = DirectRuntimeManager(FakeColabClient())

    target = await manager.connect()

    assert target == RuntimeTarget("endpoint-1", "https://runtime", "token")
    assert manager.endpoint == "endpoint-1"
    assert connected == [target]


@pytest.mark.asyncio
async def test_direct_runtime_manager_requires_endpoint_when_multiple_are_active():
    assignments = [
        SimpleNamespace(
            endpoint="endpoint-1",
            runtime_proxy_info=SimpleNamespace(url="https://one", token="one"),
        ),
        SimpleNamespace(
            endpoint="endpoint-2",
            runtime_proxy_info=SimpleNamespace(url="https://two", token="two"),
        ),
    ]

    class FakeColabClient:
        def list_assignments(self):
            return assignments

    manager = DirectRuntimeManager(FakeColabClient())

    with pytest.raises(DirectRuntimeError, match="Multiple active Colab runtimes"):
        await manager.connect()


class HangingKernelClient:
    """Mimics jupyter-kernel-client: blocks until woken, then checks the link."""

    def __init__(self):
        import threading

        ws = SimpleNamespace(
            connection_ready=threading.Event(),
            _message_received=threading.Event(),
            stop_channels=lambda: None,
            kernel_socket=None,
        )
        ws.connection_ready.set()
        self._manager = SimpleNamespace(client=ws)
        self.ws = ws

    def execute_interactive(self, code, *, output_hook, allow_stdin, timeout):
        while True:
            if not self.ws.connection_ready.is_set():
                raise RuntimeError("Connection was lost.")
            self.ws._message_received.wait(timeout=timeout)
            self.ws._message_received.clear()


@pytest.fixture
def fast_watchdog(monkeypatch):
    import colab_mcp.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "LIVENESS_POLL_SECONDS", 0.01)
    monkeypatch.setattr(runtime_module, "CONNECTION_LOST_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "ASSIGNMENT_CHECK_SECONDS", 0.02)


@pytest.mark.asyncio
async def test_execution_fails_when_connection_is_lost(fast_watchdog):
    import asyncio

    runtime = DirectColabRuntime(RuntimeTarget(endpoint="ep", url="u", token="t"))
    client = HangingKernelClient()
    runtime._kernel_client = client

    async def drop_connection():
        await asyncio.sleep(0.05)
        client.ws.connection_ready.clear()  # what _on_close does, without waking

    asyncio.create_task(drop_connection())
    with pytest.raises(DirectRuntimeError, match="Lost the connection to Colab runtime ep"):
        await asyncio.wait_for(runtime.execute("x", lambda _e: None), timeout=5)
    assert runtime.connected is False


@pytest.mark.asyncio
async def test_execution_fails_when_runtime_is_unassigned(fast_watchdog):
    import asyncio

    calls = []

    async def assigned():
        calls.append(1)
        return len(calls) < 2

    runtime = DirectColabRuntime(RuntimeTarget(endpoint="ep", url="u", token="t"), assigned)
    runtime._kernel_client = HangingKernelClient()
    with pytest.raises(DirectRuntimeError, match="no longer assigned"):
        await asyncio.wait_for(runtime.execute("x", lambda _e: None), timeout=5)


@pytest.mark.asyncio
async def test_assignment_check_errors_do_not_abort(fast_watchdog):
    import asyncio

    async def flaky():
        raise OSError("api down")

    runtime = DirectColabRuntime(RuntimeTarget(endpoint="ep", url="u", token="t"), flaky)
    client = HangingKernelClient()
    runtime._kernel_client = client

    async def finish():
        await asyncio.sleep(0.2)
        client.ws.connection_ready.clear()

    asyncio.create_task(finish())
    with pytest.raises(DirectRuntimeError, match="Lost the connection"):
        await asyncio.wait_for(runtime.execute("x", lambda _e: None), timeout=5)


class KeepAliveClient:
    def __init__(self, endpoints=("ep",)):
        self.endpoints = list(endpoints)
        self.pings = []
        self.fail = None

    def list_assignments(self):
        return [
            SimpleNamespace(
                endpoint=e,
                accelerator=SimpleNamespace(value="T4"),
                runtime_proxy_info=SimpleNamespace(url="https://runtime", token="t"),
            )
            for e in self.endpoints
        ]

    def keep_alive(self, endpoint):
        if self.fail:
            raise RuntimeError(self.fail)
        self.pings.append(endpoint)

    def get_ccu_info(self):
        return SimpleNamespace(current_balance=0.0, consumption_rate_hourly=1.15)


def _attached(client, **kwargs):
    import time as _time

    manager = DirectRuntimeManager(client, **kwargs)
    manager.runtime = SimpleNamespace(
        busy=0, last_activity=_time.time(), connected=True, close=None
    )
    manager.endpoint = manager.last_endpoint = "ep"
    manager.note_assigned("ep", "T4", 1800)
    return manager


@pytest.mark.asyncio
async def test_keep_alive_pings_recently_used_runtime():
    client = KeepAliveClient()
    manager = _attached(client)
    assert await manager.keep_alive_tick() == "active"
    assert client.pings == ["ep"]
    assert manager.records["ep"].last_keep_alive is not None


@pytest.mark.asyncio
async def test_keep_alive_pauses_after_idle_limit_but_not_while_busy():
    client = KeepAliveClient()
    manager = _attached(client, keep_alive_idle_limit=60)
    manager.runtime.last_activity -= 120
    assert (await manager.keep_alive_tick()).startswith("paused")
    assert client.pings == []
    manager.runtime.busy = 1
    assert await manager.keep_alive_tick() == "active"


@pytest.mark.asyncio
async def test_keep_alive_disabled_sends_nothing():
    client = KeepAliveClient()
    manager = _attached(client, keep_alive_interval=0)
    assert await manager.keep_alive_tick() == "off"
    assert client.pings == []


@pytest.mark.asyncio
async def test_keep_alive_records_failures():
    client = KeepAliveClient()
    client.fail = "403 Forbidden"
    manager = _attached(client)
    assert await manager.keep_alive_tick() == "failing"
    assert "403" in manager.records["ep"].keep_alive_error


@pytest.mark.asyncio
async def test_lost_runtime_is_explained_and_not_replaced_silently():
    client = KeepAliveClient()
    manager = _attached(client)
    await manager.keep_alive_tick()
    client.endpoints = ["someone-elses"]
    assert await manager.keep_alive_tick() == "lost"
    manager.runtime, manager.endpoint = None, None
    with pytest.raises(DirectRuntimeError) as err:
        await manager.connect()
    message = str(err.value)
    assert "ep (T4) is no longer assigned" in message
    assert "Keep-alive pings were succeeding" in message
    assert "Last seen alive" in message and "balance 0.0" in message
    assert "someone-elses" in message


@pytest.mark.asyncio
async def test_released_runtime_names_the_tool():
    client = KeepAliveClient(endpoints=[])
    manager = _attached(client)
    manager.note_released("ep", "stop_runtime")
    message = await manager.explain_loss("ep")
    assert "released by this MCP (stop_runtime)" in message
