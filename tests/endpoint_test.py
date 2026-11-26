"""runtime_endpoint: tools reach any account's runtime without moving the session."""

import json
from types import SimpleNamespace

import pytest

from colab_mcp import race as race_module
from colab_mcp.client import Accelerator
from colab_mcp.runtime import DirectColabRuntime


class Colab:
    def __init__(self, endpoints):
        self.endpoints = list(endpoints)
        self.unassigned = []

    def list_assignments(self):
        return [
            SimpleNamespace(
                endpoint=e,
                accelerator=Accelerator.T4,
                runtime_proxy_info=SimpleNamespace(url=f"https://{e}", token="tok"),
            )
            for e in self.endpoints
            if e not in self.unassigned
        ]

    def unassign(self, endpoint):
        self.unassigned.append(endpoint)

    def keep_alive(self, endpoint):
        pass


@pytest.fixture
def world(monkeypatch):
    """Two accounts; the session uses a@x.com (runtime a-1), b@x.com owns b-1."""
    import colab_mcp
    from colab_mcp import auth
    from colab_mcp.runtime import DirectRuntimeManager

    ran = []  # (endpoint, side kernel?, code)

    async def fake_connect(self):
        self._kernel_client = SimpleNamespace()

    async def fake_close(self, shutdown_kernel=False):
        self._kernel_client = None

    async def fake_execute(self, code, callback, *, timeout=None):
        ran.append((self.target.endpoint, self.is_side, code))
        text = json.dumps({"timed_out": False, "exit_code": 0, "stdout": self.target.endpoint,
                           "stderr": "", "seconds": 0})
        return [{"output_type": "stream", "name": "stdout", "text": text}]

    monkeypatch.setattr(DirectColabRuntime, "connect", fake_connect)
    monkeypatch.setattr(DirectColabRuntime, "close", fake_close)
    monkeypatch.setattr(DirectColabRuntime, "execute", fake_execute)
    monkeypatch.setattr(DirectColabRuntime, "is_side", False, raising=False)
    original_side = DirectRuntimeManager.side

    async def side(self):
        runtime = await original_side(self)
        runtime.is_side = True
        return runtime

    monkeypatch.setattr(DirectRuntimeManager, "side", side)

    a, b = Colab(["a-1"]), Colab(["b-1"])
    pairs = {
        "a@x.com": (a, DirectRuntimeManager(a, keep_alive_interval=0)),
        "b@x.com": (b, DirectRuntimeManager(b, keep_alive_interval=0)),
    }
    monkeypatch.setattr(colab_mcp, "_client_oauth_config", "cfg.json")
    monkeypatch.setattr(colab_mcp, "_accounts", pairs)
    monkeypatch.setattr(colab_mcp, "_endpoint_managers", {})
    monkeypatch.setattr(colab_mcp, "_colab_client", a)
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", pairs["a@x.com"][1])
    monkeypatch.setattr(colab_mcp, "_active_account", "a@x.com")
    monkeypatch.setattr(colab_mcp, "_keep_alive_interval", 0)
    monkeypatch.setattr(auth, "list_saved_accounts", lambda: sorted(pairs))
    return SimpleNamespace(mcp=colab_mcp, pairs=pairs, ran=ran, a=a, b=b)


@pytest.mark.asyncio
async def test_run_shell_on_another_accounts_runtime_keeps_session(world):
    out = json.loads(await world.mcp.run_shell.fn("hostname", runtime_endpoint="b-1"))
    assert out["stdout"] == "b-1"
    assert world.ran[-1][:2] == ("b-1", True)
    assert world.mcp._active_account == "a@x.com"
    assert world.pairs["a@x.com"][1].endpoint is None
    assert "b-1" in world.mcp._endpoint_managers
    assert world.mcp._runtime_account("b-1") == "b@x.com"

    out = json.loads(await world.mcp.run_shell.fn("hostname"))
    assert out["stdout"] == "a-1"


@pytest.mark.asyncio
async def test_second_call_reuses_the_connection(world):
    await world.mcp.run_shell.fn("x", runtime_endpoint="b-1")
    manager = world.mcp._endpoint_managers["b-1"][1]
    await world.mcp.list_runtime_files.fn(runtime_endpoint="b-1")
    assert world.mcp._endpoint_managers["b-1"][1] is manager
    assert [r[0] for r in world.ran] == ["b-1", "b-1"]


@pytest.mark.asyncio
async def test_unknown_endpoint_is_reported(world):
    out = await world.mcp.run_shell.fn("x", runtime_endpoint="nope")
    assert "No runtime 'nope' was found on any signed-in account" in out
    assert world.ran == []


@pytest.mark.asyncio
async def test_code_cell_runs_on_main_kernel_of_the_endpoint(world):
    out = await world.mcp.run_code_cell_blocking.fn(code="print(1)", runtime_endpoint="b-1")
    assert "b-1" in out
    assert world.ran[-1] == ("b-1", False, "print(1)")
    started = await world.mcp.run_code_cell.fn(code="print(2)", runtime_endpoint="b-1")
    assert started["status"] == "running"


@pytest.mark.asyncio
async def test_stop_runtime_finds_the_owning_account(world):
    await world.mcp.run_shell.fn("x", runtime_endpoint="b-1")
    out = await world.mcp.stop_runtime.fn(runtime_endpoint="b-1")
    assert out == "Stopped runtime(s): b-1 (T4) on b@x.com"
    assert world.b.unassigned == ["b-1"] and world.a.unassigned == []
    assert "b-1" not in world.mcp._endpoint_managers


@pytest.mark.asyncio
async def test_transfers_use_the_runtime_accounts_drive_token(world, tmp_path, monkeypatch):
    seen = []

    def drive_for(force, account=None):
        seen.append(account)
        return None

    monkeypatch.setattr(world.mcp, "_drive_session_for_transfer", drive_for)
    from colab_mcp import drive_relay, transfer

    async def push_file(runtime, source, remote_path):
        return 30

    monkeypatch.setattr(transfer, "push_file", push_file)
    monkeypatch.setattr(drive_relay, "DRIVE_THRESHOLD_BYTES", 10)
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 30)
    out = await world.mcp.upload_to_runtime.fn(str(f), runtime_endpoint="b-1")
    assert "via kernel" in out
    assert seen == ["b@x.com"]


@pytest.mark.asyncio
async def test_race_count_keeps_n_runtimes_from_different_accounts(world, monkeypatch):
    class Racer(Colab):
        def __init__(self, name, delay):
            super().__init__([])
            self.name, self.delay = name, delay

        def assign(self, nbh, variant, acc):
            import time

            time.sleep(self.delay)
            ep = f"{self.name}-new"
            self.endpoints.append(ep)
            return SimpleNamespace(endpoint=ep, idle_timeout_sec=1800)

        def _get_assignment(self, nbh, variant, acc):
            from colab_mcp.client import ExistingAssignment

            live = [e for e in self.endpoints if e not in self.unassigned]
            return ExistingAssignment(endpoint=live[0], fit=1800) if live else None

    from colab_mcp import auth
    from colab_mcp.runtime import DirectRuntimeManager

    racers = {f"{n}@x.com": Racer(n, d) for n, d in [("p", 0.05), ("q", 0.1), ("r", 0.4)]}
    pairs = {e: (c, DirectRuntimeManager(c, keep_alive_interval=0)) for e, c in racers.items()}
    monkeypatch.setattr(world.mcp, "_accounts", pairs)
    monkeypatch.setattr(world.mcp, "_colab_client", racers["r@x.com"])
    monkeypatch.setattr(world.mcp, "_direct_runtime_manager", pairs["r@x.com"][1])
    monkeypatch.setattr(world.mcp, "_active_account", "r@x.com")
    monkeypatch.setattr(auth, "list_saved_accounts", lambda: sorted(pairs))
    monkeypatch.setattr(race_module, "LATE_CHECK_SECONDS", 0.05)

    out = json.loads(await world.mcp.race_runtime.fn("T4", count=2))
    assert out["winner"]["account"] == "p@x.com"
    assert [r["endpoint"] for r in out["runtimes"]] == ["p-new", "q-new"]
    assert all(r["direct_connected"] for r in out["runtimes"])
    assert world.mcp._active_account == "p@x.com"
    assert world.mcp._endpoint_managers["q-new"][0] == "q@x.com"
    for race in list(world.mcp._background_races):
        await race.finish()
    assert racers["r@x.com"].unassigned == ["r-new"]
    assert racers["p@x.com"].unassigned == racers["q@x.com"].unassigned == []

    out = json.loads(await world.mcp.run_shell.fn("hostname", runtime_endpoint="q-new"))
    assert out["stdout"] == "q-new"
