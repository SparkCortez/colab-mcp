# Added to this local copy of the fork, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ask several accounts for a runtime at once and keep the first one.

Free GPUs are scarce: one account may get 503s or a POST that Colab holds
open for minutes while another gets a runtime at once.  ``race`` starts an
assignment on every account, retries temporary errors until a deadline, and
picks whichever succeeds first.  Every other account is then wound down:

* attempts still waiting on Colab cannot be interrupted (they run in worker
  threads), so they keep running in the background and a runtime they get
  later is released right away;
* after an attempt ends (even with a timeout) the account's notebook hash is
  checked again, because Colab may still finish a request it timed out on.

Only runtimes created under this race's own notebook hashes are released, so
runtimes that other sessions use on the same accounts are never touched.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Waits between retries of one account after a temporary error.
RETRY_DELAYS_SECONDS = (10, 20, 30, 45, 60)
# A runtime Colab finishes after the attempt gave up is caught by re-checking
# the notebook hash once more this long after the attempt ended.
LATE_CHECK_SECONDS = 120


@dataclass
class Entrant:
    """One account in the race."""

    email: str
    client: Any
    notebook_hash: str = field(default_factory=lambda: str(uuid.uuid4()))
    outcome: str = "waiting"
    attempts: int = 0
    endpoint: str | None = None
    fit: int | None = None
    seconds: float | None = None
    released: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        outcome = self.outcome
        if outcome == "waiting" or outcome.startswith("retrying"):
            outcome = "still waiting on Colab; a runtime it gets later is released automatically"
        row: dict[str, Any] = {"account": self.email, "outcome": outcome, "attempts": self.attempts}
        if self.seconds is not None:
            row["seconds"] = round(self.seconds, 1)
        if self.endpoint:
            row["endpoint"] = self.endpoint
        if self.released:
            row["released"] = self.released
        return row


def _endpoint_and_fit(result: Any) -> tuple[str, int | None]:
    assignment = result["assignment"] if isinstance(result, dict) else result
    return assignment.endpoint, getattr(assignment, "idle_timeout_sec", None)


def _is_temporary(exc: Exception) -> bool:
    import requests

    from colab_mcp.client import ColabRequestError

    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, ColabRequestError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status is not None and status >= 500
    return False


def _describe(exc: Exception) -> str:
    from colab_mcp.client import ColabRequestError, TooManyAssignmentsError

    if isinstance(exc, TooManyAssignmentsError):
        return "too many runtimes on this account (one may belong to another session)"
    if isinstance(exc, ColabRequestError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = getattr(getattr(exc, "response", None), "reason", "")
        return f"HTTP {status} {reason}".strip()
    return f"{type(exc).__name__}: {str(exc)[:150]}"


class Race:
    def __init__(
        self,
        entrants: list[Entrant],
        variant: Any,
        accelerator: Any,
        deadline_seconds: float,
        retry_delays: tuple[float, ...] | None = None,
        late_check_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.entrants = entrants
        self.variant = variant
        self.accelerator = accelerator
        self.deadline_seconds = deadline_seconds
        self.retry_delays = RETRY_DELAYS_SECONDS if retry_delays is None else retry_delays
        self.late_check_seconds = (
            LATE_CHECK_SECONDS if late_check_seconds is None else late_check_seconds
        )
        self.clock = clock
        self.winner: Entrant | None = None
        self.closed = False
        self.decided = asyncio.Event()
        self.started = clock()
        self.tasks: list[asyncio.Task] = []
        self.cleanups: list[asyncio.Task] = []

    @property
    def over(self) -> bool:
        return self.winner is not None or self.clock() - self.started >= self.deadline_seconds

    async def run(self) -> Entrant | None:
        """Wait until a winner exists or every entrant has given up."""
        self.tasks = [asyncio.create_task(self._enter(e)) for e in self.entrants]
        waiter = asyncio.create_task(self.decided.wait())
        pending = set(self.tasks) | {waiter}
        while self.winner is None and any(not t.done() for t in self.tasks):
            remaining = self.deadline_seconds - (self.clock() - self.started)
            if remaining <= 0:
                break
            _done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
        waiter.cancel()
        # From here on nobody can win: a request Colab still answers is released.
        self.closed = True
        return self.winner

    async def finish(self) -> None:
        """Wait for every losing attempt and its late check (background use)."""
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(*self.cleanups, return_exceptions=True)

    async def _release(self, entrant: Entrant, endpoint: str) -> None:
        try:
            await asyncio.to_thread(entrant.client.unassign, endpoint)
            entrant.released.append(endpoint)
            logger.info("Race: released %s on %s (lost the race)", endpoint, entrant.email)
        except Exception:
            logger.warning("Race: could not release %s on %s", endpoint, entrant.email, exc_info=True)

    async def _enter(self, entrant: Entrant) -> None:
        delays = iter(self.retry_delays)
        delay = self.retry_delays[-1] if self.retry_delays else 0
        try:
            while not self.over:
                entrant.attempts += 1
                try:
                    result = await asyncio.to_thread(
                        entrant.client.assign, entrant.notebook_hash, self.variant, self.accelerator
                    )
                except Exception as exc:
                    if _is_temporary(exc) and not self.over:
                        entrant.outcome = f"retrying after {_describe(exc)}"
                        delay = next(delays, delay)
                        await self._sleep_unless_decided(delay)
                        continue
                    entrant.outcome = _describe(exc) if not self.over else (
                        "stopped: " + ("another account won" if self.winner else "deadline reached")
                    )
                    return
                endpoint, fit = _endpoint_and_fit(result)
                entrant.endpoint, entrant.fit = endpoint, fit
                entrant.seconds = self.clock() - self.started
                if self.winner is None and not self.closed:
                    self.winner = entrant
                    entrant.outcome = "won"
                    self.decided.set()
                    return
                entrant.outcome = (
                    "got a runtime after another account won; released it"
                    if self.winner is not None
                    else "got a runtime after the deadline; released it"
                )
                await self._release(entrant, endpoint)
                return
            if entrant.outcome == "waiting" or entrant.outcome.startswith("retrying"):
                entrant.outcome = "stopped: " + (
                    "another account won" if self.winner else "deadline reached"
                )
        finally:
            if self.winner is not entrant:
                self.cleanups.append(asyncio.create_task(self._late_check(entrant)))

    async def _sleep_unless_decided(self, delay: float) -> None:
        remaining = self.deadline_seconds - (self.clock() - self.started)
        try:
            await asyncio.wait_for(self.decided.wait(), timeout=max(0.0, min(delay, remaining)))
        except asyncio.TimeoutError:
            pass

    async def _late_check(self, entrant: Entrant) -> None:
        """Release a runtime Colab created for this hash after we stopped waiting."""
        from colab_mcp.client import Assignment, ExistingAssignment

        for wait in (0.0, self.late_check_seconds):
            if wait:
                await asyncio.sleep(wait)
            try:
                found = await asyncio.to_thread(
                    entrant.client._get_assignment,
                    entrant.notebook_hash,
                    self.variant,
                    self.accelerator,
                )
            except Exception:
                logger.debug("Race: late check failed for %s", entrant.email, exc_info=True)
                continue
            if isinstance(found, (Assignment, ExistingAssignment)) and found.endpoint not in entrant.released:
                await self._release(entrant, found.endpoint)
                return
