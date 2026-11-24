# Copyright 2026 Atsushi Onozawa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Direct, browser-independent access to a Colab Jupyter kernel.

The hosted Colab runtime proxy exposes a Jupyter-compatible endpoint.  This
module deliberately keeps that transport separate from the browser MCP bridge:
the browser is useful for notebook UI operations, but it is not required for
executing Python or receiving incremental kernel output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)

OutputCallback = Callable[[dict[str, Any]], None]
AssignmentCheck = Callable[[], Awaitable[bool]]
LossExplainer = Callable[[], Awaitable[str]]

# jupyter-kernel-client clears ``connection_ready`` when the websocket closes
# but never wakes a thread blocked in ``execute_interactive``, so a lost
# runtime would leave an execution "running" forever.  A watchdog polls the
# connection (and, less often, whether Colab still assigns the runtime) and
# aborts the execution with an explicit error instead.
LIVENESS_POLL_SECONDS = 5.0
CONNECTION_LOST_GRACE_SECONDS = 30.0
ASSIGNMENT_CHECK_SECONDS = 60.0

# Colab reclaims a runtime it considers idle (free T4s after roughly 20-30
# minutes without a frontend).  Like colab-vscode, ping the tunnel keep-alive
# endpoint every 5 minutes while the runtime was used recently; stop after the
# idle limit so a forgotten runtime is not kept alive forever.  0 disables it.
KEEP_ALIVE_INTERVAL_SECONDS = 300.0
KEEP_ALIVE_IDLE_LIMIT_SECONDS = 60 * 60.0


def _clock(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    ago = max(0, int(time.time() - ts))
    stamp = time.strftime("%H:%M:%S UTC", time.gmtime(ts))
    return f"{stamp} ({ago // 60} min {ago % 60} s ago)"


@dataclass
class RuntimeRecord:
    """What this MCP process knows about a runtime it attached to."""

    endpoint: str
    accelerator: str | None = None
    idle_timeout_sec: int | None = None
    attached_at: float = field(default_factory=time.time)
    last_seen_alive: float | None = None
    last_keep_alive: float | None = None
    keep_alive_error: str | None = None
    keep_alive_state: str = "pending"
    released_by: str | None = None
    released_at: float | None = None
    lost_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def iso(ts: float | None) -> str | None:
            return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

        return {
            "idle_timeout_sec": self.idle_timeout_sec,
            "keep_alive": self.keep_alive_state,
            "last_keep_alive": iso(self.last_keep_alive),
            "keep_alive_error": self.keep_alive_error,
            "last_seen_alive": iso(self.last_seen_alive),
        }


class DirectRuntimeError(RuntimeError):
    """Raised when a direct Colab runtime cannot be selected or connected."""


@dataclass(frozen=True)
class RuntimeTarget:
    """The short-lived proxy credentials needed by a Jupyter kernel client."""

    endpoint: str
    url: str
    token: str


class DirectColabRuntime:
    """A small async wrapper around ``jupyter-kernel-client``.

    ``jupyter-kernel-client`` is synchronous and invokes its output hook from
    the reader thread.  All blocking work therefore runs in a worker thread;
    callers receive each normalized output record through ``output_callback``.
    """

    def __init__(
        self,
        target: RuntimeTarget,
        assignment_check: AssignmentCheck | None = None,
        explain_loss: LossExplainer | None = None,
    ):
        self.target = target
        self._assignment_check = assignment_check
        self._explain_loss = explain_loss
        # Wall-clock time of the last execution start/end, and how many are
        # running; the keep-alive pauses once the runtime has been idle long.
        self.last_activity = time.time()
        self.busy = 0
        self._kernel_client: Any | None = None
        self._connect_lock = asyncio.Lock()
        self._execute_lock = asyncio.Lock()
        self._lost_reason: str | None = None

    @property
    def connected(self) -> bool:
        client = self._kernel_client
        if client is None:
            return False
        try:
            return bool(getattr(client, "is_alive", lambda: True)())
        except Exception:
            return False

    def _build_kernel_client(self) -> Any:
        try:
            import jupyter_kernel_client
        except ImportError as exc:  # pragma: no cover - dependency is packaged
            raise DirectRuntimeError(
                "Direct Colab execution requires jupyter-kernel-client."
            ) from exc

        client_kwargs: dict[str, Any] = {
            "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
            "extra_params": {"colab-runtime-proxy-token": self.target.token},
        }
        headers = {
            "X-Colab-Client-Agent": "colab-mcp",
            "X-Colab-Runtime-Proxy-Token": self.target.token,
        }
        client_type = getattr(jupyter_kernel_client, "ColabKernelClient", None)
        if client_type is not None:
            client = client_type(
                server_url=self.target.url,
                proxy_token=self.target.token,
                client_kwargs=client_kwargs,
                headers=headers,
            )
        else:
            client = jupyter_kernel_client.KernelClient(
                server_url=self.target.url,
                token=self.target.token,
                client_kwargs=client_kwargs,
                headers=headers,
            )

        # Closing the MCP runtime client must not shut down a user-owned Colab
        # kernel.  The official Colab CLI uses the same ownership boundary.
        client._own_kernel = False
        client.start()
        return client

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._kernel_client is not None:
                return
            self._kernel_client = await asyncio.to_thread(self._build_kernel_client)

    def _execute_sync(
        self,
        code: str,
        output_callback: OutputCallback,
        timeout: float,
    ) -> list[dict[str, Any]]:
        client = self._kernel_client
        if client is None:
            raise DirectRuntimeError("The direct Colab runtime is not connected.")

        # The library's default output hook coalesces display updates and
        # returns a final list.  Wrap it so callers see each newly completed
        # output record as soon as it arrives on IOPub.
        from jupyter_kernel_client.client import output_hook as default_output_hook

        outputs: list[dict[str, Any]] = []

        def on_output(message: dict[str, Any]) -> None:
            new_indexes = default_output_hook(outputs, message)
            for index in sorted(new_indexes):
                if index < len(outputs):
                    output_callback(dict(outputs[index]))

        reply = client.execute_interactive(
            code,
            output_hook=on_output,
            allow_stdin=False,
            timeout=timeout,
        )
        reply_content = reply.get("content", {}) if reply else {"status": "error"}
        if reply_content.get("status") == "error" and not any(
            item.get("output_type") == "error" for item in outputs
        ):
            error = {
                "output_type": "error",
                "ename": reply_content.get("ename", "Error"),
                "evalue": reply_content.get("evalue", "Unknown error"),
                "traceback": reply_content.get("traceback", []),
            }
            outputs.append(error)
            output_callback(error)
        return outputs

    async def execute(
        self,
        code: str,
        output_callback: OutputCallback,
        *,
        timeout: float = 24 * 60 * 60,
    ) -> list[dict[str, Any]]:
        await self.connect()

        def callback_from_thread(output: dict[str, Any]) -> None:
            # ``CodeExecutionRegistry.publish`` is thread-safe, but callers
            # may use a coroutine-backed callback in tests or future clients.
            try:
                output_callback(output)
            except Exception:
                logger.exception("Direct runtime output callback failed")

        self.busy += 1
        self.last_activity = time.time()
        try:
            return await self._execute_locked(code, callback_from_thread, timeout)
        finally:
            self.busy -= 1
            self.last_activity = time.time()

    async def _execute_locked(
        self, code: str, callback_from_thread: OutputCallback, timeout: float
    ) -> list[dict[str, Any]]:
        async with self._execute_lock:
            client = self._kernel_client
            self._lost_reason = None
            work = asyncio.ensure_future(
                asyncio.to_thread(
                    self._execute_sync,
                    code,
                    callback_from_thread,
                    timeout,
                )
            )
            watchdog = asyncio.create_task(self._watch(client))
            try:
                return await work
            except Exception as exc:
                if self._lost_reason:
                    raise DirectRuntimeError(self._lost_reason) from exc
                raise
            finally:
                watchdog.cancel()

    @staticmethod
    def _websocket_client(client: Any) -> Any | None:
        """Return the inner websocket client that owns the connection events."""
        manager = getattr(client, "_manager", None)
        return getattr(manager, "client", None)

    async def _watch(self, client: Any) -> None:
        """Abort the running execution once the runtime is unreachable."""
        lost_since: float | None = None
        last_assignment_check = time.monotonic()
        while True:
            await asyncio.sleep(LIVENESS_POLL_SECONDS)
            ws = self._websocket_client(client)
            ready = getattr(ws, "connection_ready", None)
            if ready is None or ready.is_set():
                lost_since = None
            else:
                lost_since = lost_since or time.monotonic()
                if time.monotonic() - lost_since >= CONNECTION_LOST_GRACE_SECONDS:
                    await self._abort(
                        client,
                        f"Lost the connection to Colab runtime {self.target.endpoint} "
                        f"for {CONNECTION_LOST_GRACE_SECONDS:.0f}s; the runtime was "
                        "probably stopped or reclaimed. Check list_runtimes.",
                    )
                    return
            if (
                self._assignment_check is not None
                and time.monotonic() - last_assignment_check >= ASSIGNMENT_CHECK_SECONDS
            ):
                last_assignment_check = time.monotonic()
                try:
                    assigned = await self._assignment_check()
                except Exception:
                    # A transient API failure is not evidence the runtime died.
                    logger.debug("Runtime assignment check failed", exc_info=True)
                    assigned = True
                if not assigned:
                    reason = (
                        f"Colab runtime {self.target.endpoint} is no longer assigned "
                        "(stopped by stop_runtime/change_runtime, idle timeout, or "
                        "reclaimed by Colab)."
                    )
                    if self._explain_loss is not None:
                        try:
                            reason = await self._explain_loss()
                        except Exception:
                            logger.debug("Could not explain runtime loss", exc_info=True)
                    await self._abort(
                        client, f"{reason} Output after this point is lost."
                    )
                    return

    async def _abort(self, client: Any, reason: str) -> None:
        logger.warning(reason)
        self._lost_reason = reason
        if self._kernel_client is client:
            await self.close()
        ws = self._websocket_client(client)
        received = getattr(ws, "_message_received", None)
        ready = getattr(ws, "connection_ready", None)
        if ready is not None:
            ready.clear()
        if received is not None:
            # Wake execute_interactive so it re-checks connection_ready and
            # raises instead of waiting for the full execution timeout.
            received.set()

    async def close(self) -> None:
        client = self._kernel_client
        self._kernel_client = None
        if client is None:
            return

        def stop() -> None:
            try:
                manager_client = client._manager.client
                manager_client.stop_channels()
                if manager_client.kernel_socket:
                    manager_client.kernel_socket.close()
            except Exception:  # pragma: no cover - cleanup must be best effort
                logger.debug("Error closing direct Colab runtime", exc_info=True)

        await asyncio.to_thread(stop)


class DirectRuntimeManager:
    """Select and own one direct runtime from the OAuth-backed Colab client."""

    def __init__(
        self,
        colab_client: Any,
        keep_alive_interval: float = KEEP_ALIVE_INTERVAL_SECONDS,
        keep_alive_idle_limit: float = KEEP_ALIVE_IDLE_LIMIT_SECONDS,
    ):
        self.colab_client = colab_client
        self.runtime: DirectColabRuntime | None = None
        self.endpoint: str | None = None
        self.keep_alive_interval = keep_alive_interval
        self.keep_alive_idle_limit = keep_alive_idle_limit
        # Survives close() so a vanished runtime can still be explained.
        self.last_endpoint: str | None = None
        self.records: dict[str, RuntimeRecord] = {}
        self._keep_alive_task: asyncio.Task | None = None

    # -- bookkeeping -------------------------------------------------------

    def record(self, endpoint: str) -> RuntimeRecord:
        if endpoint not in self.records:
            self.records[endpoint] = RuntimeRecord(endpoint)
        return self.records[endpoint]

    def note_assigned(
        self, endpoint: str, accelerator: str | None, idle_timeout_sec: int | None
    ) -> None:
        rec = self.record(endpoint)
        rec.accelerator = accelerator or rec.accelerator
        if idle_timeout_sec:
            rec.idle_timeout_sec = idle_timeout_sec
        rec.last_seen_alive = time.time()

    def note_released(self, endpoint: str, by: str) -> None:
        rec = self.record(endpoint)
        rec.released_by = by
        rec.released_at = time.time()

    @property
    def keep_alive_enabled(self) -> bool:
        return self.keep_alive_interval > 0 and self.keep_alive_idle_limit > 0

    async def explain_loss(self, endpoint: str | None = None) -> str:
        """Describe why a runtime this process used is gone, as far as known."""
        endpoint = endpoint or self.last_endpoint
        rec = self.records.get(endpoint or "")
        if rec is None:
            return f"Colab runtime {endpoint} is no longer assigned."
        if rec.lost_at is None:
            rec.lost_at = time.time()
        label = f"{endpoint} ({rec.accelerator})" if rec.accelerator else f"{endpoint}"
        if rec.released_by:
            cause = f"It was released by this MCP ({rec.released_by}) at {_clock(rec.released_at)}."
        elif rec.keep_alive_state == "off":
            cause = (
                "Keep-alive is off, so Colab most likely reclaimed it as idle"
                + (f" (idle timeout {rec.idle_timeout_sec} s)." if rec.idle_timeout_sec else ".")
            )
        elif rec.keep_alive_state.startswith("paused"):
            cause = (
                f"Keep-alive had {rec.keep_alive_state}, so it most likely hit Colab's "
                "idle timeout."
            )
        elif rec.last_keep_alive is not None and rec.keep_alive_error is None:
            cause = (
                f"Keep-alive pings were succeeding (last {_clock(rec.last_keep_alive)}), so "
                "Colab most likely reclaimed it for a usage limit (free-tier GPU time or "
                "compute units), or it was stopped from another session or the Colab UI."
            )
        elif rec.keep_alive_error:
            cause = (
                f"Keep-alive pings were failing ({rec.keep_alive_error}), so it most likely "
                "hit Colab's idle timeout."
            )
        else:
            cause = (
                "No keep-alive ping had been sent yet; it may have been stopped from "
                "another session or the Colab UI, or reclaimed by Colab."
            )
        parts = [f"Colab runtime {label} is no longer assigned. {cause}"]
        parts.append(f"Last seen alive: {_clock(rec.last_seen_alive)}.")
        runtime = self.runtime if self.endpoint == endpoint else None
        if runtime is not None:
            parts.append(f"Last MCP activity: {_clock(runtime.last_activity)}.")
        try:
            ccu = await asyncio.to_thread(self.colab_client.get_ccu_info)
            parts.append(
                f"Compute units: balance {ccu.current_balance}, "
                f"burning {ccu.consumption_rate_hourly}/h."
            )
        except Exception:
            pass
        parts.append("Files in /content are gone; start a runtime with change_runtime and re-upload.")
        return " ".join(parts)

    # -- keep-alive --------------------------------------------------------

    def _start_keep_alive(self) -> None:
        if self._keep_alive_task is None or self._keep_alive_task.done():
            self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def _stop_keep_alive(self) -> None:
        task, self._keep_alive_task = self._keep_alive_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _keep_alive_loop(self) -> None:
        while True:
            await asyncio.sleep(max(self.keep_alive_interval, 1.0))
            try:
                if await self.keep_alive_tick() == "lost":
                    return
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never kill the loop
                logger.exception("Keep-alive tick failed")

    async def keep_alive_tick(self) -> str:
        """One keep-alive round; returns the runtime's keep-alive state."""
        endpoint = self.endpoint
        if not endpoint:
            return "detached"
        rec = self.record(endpoint)
        try:
            assignments = await asyncio.to_thread(self.colab_client.list_assignments)
        except Exception as exc:
            logger.debug("Keep-alive assignment check failed: %s", exc)
            assignments = None
        if assignments is not None:
            if not any(a.endpoint == endpoint for a in assignments):
                logger.warning(await self.explain_loss(endpoint))
                rec.keep_alive_state = "lost"
                return "lost"
            rec.last_seen_alive = time.time()
        if not self.keep_alive_enabled:
            rec.keep_alive_state = "off"
            return rec.keep_alive_state
        runtime = self.runtime
        idle = 0.0 if runtime is None or runtime.busy else time.time() - runtime.last_activity
        if idle > self.keep_alive_idle_limit:
            rec.keep_alive_state = (
                f"paused after {int(self.keep_alive_idle_limit // 60)} min without MCP activity"
            )
            return rec.keep_alive_state
        try:
            await asyncio.to_thread(self.colab_client.keep_alive, endpoint)
            rec.last_keep_alive = time.time()
            rec.keep_alive_error = None
            rec.keep_alive_state = "active"
        except Exception as exc:
            rec.keep_alive_error = str(exc)[:200]
            rec.keep_alive_state = "failing"
            logger.warning("Keep-alive for %s failed: %s", endpoint, exc)
        return rec.keep_alive_state

    # -- connection --------------------------------------------------------

    async def _select_target(self, endpoint: str = "") -> RuntimeTarget:
        assignments = await asyncio.to_thread(self.colab_client.list_assignments)
        assignments = assignments or []
        own = self.last_endpoint
        own_gone = own is not None and not any(a.endpoint == own for a in assignments)
        if endpoint:
            matches = [item for item in assignments if item.endpoint == endpoint]
            if not matches:
                if endpoint in self.records:
                    raise DirectRuntimeError(await self.explain_loss(endpoint))
                raise DirectRuntimeError(
                    f"No active Colab runtime was found for endpoint '{endpoint}'."
                )
        elif own is not None and not own_gone:
            matches = [item for item in assignments if item.endpoint == own]
        elif own_gone:
            # Never fall back silently to a runtime another session may own.
            message = await self.explain_loss(own)
            if assignments:
                choices = ", ".join(item.endpoint for item in assignments)
                message += (
                    f" Other runtimes on this account (possibly another session's): {choices}."
                )
            raise DirectRuntimeError(message)
        elif len(assignments) == 1:
            matches = assignments
        elif not assignments:
            raise DirectRuntimeError(
                "No active Colab runtime is available. Start one in Colab or "
                "call change_runtime first."
            )
        else:
            choices = ", ".join(item.endpoint for item in assignments)
            raise DirectRuntimeError(
                "Multiple active Colab runtimes were found. Retry with "
                f"runtime_endpoint set to one of: {choices}."
            )

        accelerator = getattr(matches[0], "accelerator", None)
        self.note_assigned(
            matches[0].endpoint, getattr(accelerator, "value", accelerator), None
        )
        info = matches[0].runtime_proxy_info
        return RuntimeTarget(
            endpoint=matches[0].endpoint,
            url=info.url,
            token=info.token,
        )

    async def connect(self, endpoint: str = "") -> RuntimeTarget:
        target = await self._select_target(endpoint)
        if (
            self.runtime is not None
            and self.endpoint == target.endpoint
            and self.runtime.target.url == target.url
            and self.runtime.target.token == target.token
        ):
            return target
        await self.close()
        await self._attach(target)
        return target

    def _assignment_check_for(self, endpoint: str) -> AssignmentCheck:
        async def check() -> bool:
            assignments = await asyncio.to_thread(self.colab_client.list_assignments)
            return any(item.endpoint == endpoint for item in assignments or [])

        return check

    async def _attach(self, target: RuntimeTarget) -> None:
        runtime = DirectColabRuntime(
            target,
            self._assignment_check_for(target.endpoint),
            lambda: self.explain_loss(target.endpoint),
        )
        await runtime.connect()
        self.runtime = runtime
        self.endpoint = target.endpoint
        self.last_endpoint = target.endpoint
        rec = self.record(target.endpoint)
        rec.released_by = rec.released_at = rec.lost_at = None
        rec.keep_alive_state = "on" if self.keep_alive_enabled else "off"
        self._start_keep_alive()

    async def attach_target(self, target: RuntimeTarget) -> None:
        await self.close()
        await self._attach(target)

    async def close(self) -> None:
        await self._stop_keep_alive()
        if self.runtime is not None:
            await self.runtime.close()
        self.runtime = None
        self.endpoint = None

    @property
    def connected(self) -> bool:
        return self.runtime is not None and self.runtime.connected
