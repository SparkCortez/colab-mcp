"""race_runtime: the first account to get a runtime wins, every other one is wound down."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from colab_mcp import race as race_module
from colab_mcp.client import (
    Accelerator,
    ColabRequestError,
    ExistingAssignment,
    GetAssignmentResponse,
    TooManyAssignmentsError,
    Variant,
)
from colab_mcp.race import Entrant, Race


def http_error(status):
    return ColabRequestError("boom", None, SimpleNamespace(status_code=status, reason="Err"))


class FakeColab:
    """Plays a script of assign outcomes; numbers are seconds to block first."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.calls = 0
        self.unassigned = []
        self.created = {}  # notebook hash -> endpoint, what GET /assign would report
        self.lock = threading.Lock()

    def assign(self, nbh, variant, acc):
        # Like Colab, asking again for a hash that has a runtime returns it.
        if nbh in self.created and self.created[nbh] not in self.unassigned:
            return {"assignment": ExistingAssignment(endpoint=self.created[nbh], fit=1800), "is_new": False}
        with self.lock:
            self.calls += 1
            step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, tuple):
            time.sleep(step[0])
            step = step[1]
        if isinstance(step, Exception):
            raise step
        if step == "late":
            # The request timed out on our side but Colab finishes it later.
            self.created[nbh] = f"{self.name}-late"
            raise requests.exceptions.ReadTimeout("slow")
        endpoint = f"{self.name}-{self.calls}"
        self.created[nbh] = endpoint
        return SimpleNamespace(endpoint=endpoint, idle_timeout_sec=1800)

    def _get_assignment(self, nbh, variant, acc):
        if nbh in self.created and self.created[nbh] not in self.unassigned:
            return ExistingAssignment(endpoint=self.created[nbh], fit=1800)
        return GetAssignmentResponse(acc="T4", nbh="x", token="t", variant="GPU")

    def unassign(self, endpoint):
        self.unassigned.append(endpoint)


def make_race(clients, deadline=5.0, late=0.05):
    entrants = [Entrant(c.name, c) for c in clients]
    return Race(
        entrants, Variant.GPU, Accelerator.T4, deadline,
        retry_delays=(0.05,), late_check_seconds=late,
    ), entrants


@pytest.mark.asyncio
async def test_fastest_account_wins_and_slow_one_is_released_when_it_arrives():
    fast, slow = FakeColab("fast", [(0.05, "ok")]), FakeColab("slow", [(0.4, "ok")])
    race, entrants = make_race([slow, fast])
    winner = await race.run()
    assert winner.email == "fast"
    assert winner.endpoint == "fast-1"
    assert slow.unassigned == []  # still waiting on Colab
    await race.finish()
    assert slow.unassigned == ["slow-1"]
    assert fast.unassigned == []
    assert entrants[0].outcome.startswith("got a runtime after another account won")


@pytest.mark.asyncio
async def test_temporary_errors_are_retried_until_success():
    flaky = FakeColab("flaky", [http_error(503), http_error(502), "ok"])
    full = FakeColab("full", [TooManyAssignmentsError("412")])
    race, entrants = make_race([flaky, full])
    winner = await race.run()
    assert winner.email == "flaky" and flaky.calls == 3
    assert "too many runtimes" in entrants[1].outcome
    await race.finish()
    assert full.unassigned == [] and flaky.unassigned == []


@pytest.mark.asyncio
async def test_permanent_error_drops_only_that_account():
    denied = FakeColab("denied", [http_error(403)])
    ok = FakeColab("ok", [(0.1, "ok")])
    race, entrants = make_race([denied, ok])
    winner = await race.run()
    assert winner.email == "ok"
    assert entrants[0].outcome == "HTTP 403 Err" and denied.calls == 1


@pytest.mark.asyncio
async def test_no_winner_before_deadline():
    a = FakeColab("a", [http_error(503)] * 100)
    b = FakeColab("b", [http_error(503)] * 100)
    race, entrants = make_race([a, b], deadline=0.3)
    started = time.monotonic()
    assert await race.run() is None
    assert time.monotonic() - started < 2
    assert all(e.outcome == "stopped: deadline reached" for e in entrants)


@pytest.mark.asyncio
async def test_runtime_created_after_a_client_timeout_is_released():
    late = FakeColab("late", [(0.1, "late")])
    winner_client = FakeColab("win", [(0.02, "ok")])
    race, entrants = make_race([late, winner_client], late=0.05)
    assert (await race.run()).email == "win"
    await race.finish()
    assert late.unassigned == ["late-late"]
    assert entrants[0].outcome == "stopped: another account won"


@pytest.mark.asyncio
async def test_retry_sleep_ends_as_soon_as_someone_wins():
    waiting = FakeColab("waiting", [http_error(503)] * 5)
    win = FakeColab("win", [(0.1, "ok")])
    entrants = [Entrant("waiting", waiting), Entrant("win", win)]
    race = Race(entrants, Variant.GPU, Accelerator.T4, 60, retry_delays=(30,), late_check_seconds=0)
    await race.run()
    started = time.monotonic()
    await asyncio.wait_for(race.finish(), timeout=5)
    assert time.monotonic() - started < 2
    assert waiting.calls == 1
    assert entrants[0].outcome == "stopped: another account won"


# -- the MCP tool -----------------------------------------------------------


class FakeManager:
    def __init__(self):
        self.endpoint = self.last_endpoint = None
        self.connected = False
        self.attached = []
        self.keep_alive_enabled = True
        self.keep_alive_interval = 300
        self.keep_alive_idle_limit = 3600

    async def attach_target(self, target):
        self.attached.append(target)
        self.endpoint = self.last_endpoint = target.endpoint
        self.connected = True

    def note_assigned(self, *args):
        pass

    def note_released(self, *args):
        pass

    async def close(self):
        self.connected = False
        self.endpoint = None


class ToolColab(FakeColab):
    def list_assignments(self):
        return [
            SimpleNamespace(
                endpoint=ep,
                accelerator=Accelerator.T4,
                runtime_proxy_info=SimpleNamespace(url=f"https://{ep}", token="tok"),
            )
            for ep in self.created.values()
            if ep not in self.unassigned
        ]


@pytest.fixture
def accounts(monkeypatch):
    import colab_mcp
    from colab_mcp import auth

    pairs = {
        "a@x.com": (ToolColab("a", [(0.3, "ok")]), FakeManager()),
        "b@x.com": (ToolColab("b", [http_error(503), (0.05, "ok")]), FakeManager()),
    }
    monkeypatch.setattr(colab_mcp, "_client_oauth_config", "cfg.json")
    monkeypatch.setattr(colab_mcp, "_accounts", pairs)
    monkeypatch.setattr(colab_mcp, "_colab_client", pairs["a@x.com"][0])
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", pairs["a@x.com"][1])
    monkeypatch.setattr(colab_mcp, "_active_account", "a@x.com")
    monkeypatch.setattr(auth, "list_saved_accounts", lambda: sorted(pairs))
    monkeypatch.setattr(race_module, "RETRY_DELAYS_SECONDS", (0.05,))
    monkeypatch.setattr(race_module, "LATE_CHECK_SECONDS", 0.05)
    return colab_mcp, pairs


@pytest.mark.asyncio
async def test_tool_switches_to_winner_and_stops_the_rest(accounts):
    colab_mcp, pairs = accounts
    out = json.loads(await colab_mcp.race_runtime.fn("T4"))
    assert out["winner"]["account"] == "b@x.com"
    assert out["winner"]["direct_connected"] is True
    assert colab_mcp._active_account == "b@x.com"
    assert pairs["b@x.com"][1].endpoint == out["winner"]["endpoint"]
    for race in list(colab_mcp._background_races):
        await race.finish()
    assert pairs["a@x.com"][0].unassigned == ["a-1"]
    assert pairs["b@x.com"][0].unassigned == []


@pytest.mark.asyncio
async def test_tool_reuses_matching_runtime(accounts):
    colab_mcp, pairs = accounts
    client, manager = pairs["a@x.com"]
    client.created["own"] = "a-own"
    manager.endpoint = manager.last_endpoint = "a-own"
    manager.connected = True
    out = await colab_mcp.race_runtime.fn("T4")
    assert "reused" in out and client.calls == 0


@pytest.mark.asyncio
async def test_tool_releases_previous_runtime_of_this_session(accounts):
    colab_mcp, pairs = accounts
    client, manager = pairs["a@x.com"]
    client.created["own"] = "a-own"
    manager.endpoint = manager.last_endpoint = "a-own"
    out = json.loads(await colab_mcp.race_runtime.fn("T4", force_new=True))
    assert out["winner"]["account"] == "b@x.com"
    assert "a-own" in client.unassigned
    assert "Released previous runtime a-own" in out["previous_runtime"]


@pytest.mark.asyncio
async def test_tool_reports_unknown_accounts_and_no_winner(accounts, monkeypatch):
    colab_mcp, pairs = accounts
    pairs["a@x.com"][0].script = [http_error(503)] * 100
    out = json.loads(await colab_mcp.race_runtime.fn(
        "T4", accounts=["a@x.com", "zzz@x.com"], timeout_minutes=0.005
    ))
    assert out["winner"] is None
    assert {"account": "zzz@x.com", "outcome": "not signed in"} in out["accounts"]
    assert colab_mcp._active_account == "a@x.com"


@pytest.mark.asyncio
async def test_deadline_is_kept_even_while_a_request_hangs():
    hanging = FakeColab("hang", [(0.8, "ok")])
    race, entrants = make_race([hanging], deadline=0.2)
    started = time.monotonic()
    assert await race.run() is None
    assert time.monotonic() - started < 0.6
    await race.finish()
    assert hanging.unassigned == ["hang-1"]
    assert entrants[0].outcome == "got a runtime after the deadline; released it"
