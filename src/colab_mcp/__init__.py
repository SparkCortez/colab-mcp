# Copyright 2026 Google Inc.
# Modified by Sebastian Gil Pinzon and Atsushi Onozawa, 2026.
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

import argparse
import asyncio
import datetime
import ipaddress
import json
import logging
import os
import tempfile
import sys
import time
import uuid
import webbrowser
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from colab_mcp.connection import (
    InvalidNotebookUrl,
    build_connection_info,
    normalize_notebook_url,
)
from colab_mcp.execution import CodeExecutionRegistry
from colab_mcp.runtime import (
    KEEP_ALIVE_IDLE_LIMIT_SECONDS,
    KEEP_ALIVE_INTERVAL_SECONDS,
    DirectRuntimeError,
    DirectRuntimeManager,
    RuntimeTarget,
)
from colab_mcp.session import ColabSessionProxy, NOT_CONNECTED_MSG
from colab_mcp import process_registry


class _NoDocketFastMCP(FastMCP):
    """FastMCP without the Docket background-task worker.

    No tool here uses background tasks, and fastmcp 2.14's default in-memory
    Docket (fakeredis) wakes every 10 ms while idle, costing ~3% CPU per
    server instance.
    """

    @asynccontextmanager
    async def _docket_lifespan(self) -> AsyncIterator[None]:
        from fastmcp.server.dependencies import _current_server

        token = _current_server.set(weakref.ref(self))
        try:
            yield
        finally:
            _current_server.reset(token)


mcp = _NoDocketFastMCP(name="ColabMCP")

# These will be set during main_async() startup
_proxy_client = None
_session_mcp = None
_colab_client = None  # OAuth-backed runtime API
_direct_runtime_manager: DirectRuntimeManager | None = None
_client_oauth_config: str | None = None
# Signed-in Google accounts loaded so far: email -> (ColabClient, DirectRuntimeManager).
# _colab_client/_direct_runtime_manager always point at the active account's pair.
_accounts: dict[str, tuple[object, DirectRuntimeManager]] = {}
_active_account: str | None = None
# One lock per account, so a slow assignment on one account never blocks another.
_change_runtime_locks: dict[str | None, asyncio.Lock] = {}
# (account, accelerator) -> notebook hash of an assignment still in progress.
_pending_assignments: dict[tuple[str | None, str], str] = {}
_keep_alive_interval = KEEP_ALIVE_INTERVAL_SECONDS
_keep_alive_idle_limit = KEEP_ALIVE_IDLE_LIMIT_SECONDS
_process_profile = "default"
_last_notebook_url: str | None = None
_connection_nonce: str | None = None
_connection_attempt_url: str | None = None
_connection_attempt_started_at: float | None = None
_connection_open_lock = asyncio.Lock()
_process_entry = None
_execution_registry = CodeExecutionRegistry()
CONNECTION_FAILURE_GRACE_SECONDS = 60.0


def _mark_process_state(state: str) -> None:
    """Best-effort lifecycle update used to clean failed client restarts."""
    try:
        entry = _process_entry
        process_registry.mark_state(
            state,
            pid=os.getpid(),
            started_at=entry.started_at if entry is not None else None,
            instance_id=entry.instance_id if entry is not None else None,
        )
    except Exception as exc:  # pragma: no cover - diagnostics must not break tools
        logging.debug("Could not update process state to %s: %s", state, exc)


async def _forward_or_stub(tool_name: str, arguments: dict) -> str:
    """Forward a tool call to the browser if connected, otherwise return stub message."""
    if _proxy_client is not None and _proxy_client.is_connected():
        _mark_process_state("connected")
        try:
            result = await _proxy_client.proxy_mcp_client.call_tool(tool_name, arguments)
            # Extract text from result
            if hasattr(result, 'content'):
                return "\n".join(c.text for c in result.content if hasattr(c, 'text'))
            return str(result)
        except Exception as e:
            return f"Error calling {tool_name}: {e}. Try calling open_colab_browser_connection to reconnect."
    if (
        _connection_attempt_url is not None
        and _connection_attempt_started_at is not None
        and time.monotonic() - _connection_attempt_started_at
        >= CONNECTION_FAILURE_GRACE_SECONDS
    ):
        # A browser URL was issued and the full connection wait window has
        # elapsed without a browser connection. This avoids marking a process
        # failed during the normal manual-tab paste race.
        _mark_process_state("connection_failed")
    return NOT_CONNECTED_MSG


def _connection_info() -> dict[str, object] | None:
    """Return current local connection coordinates without logging secrets."""
    global _connection_nonce
    if _proxy_client is None or _proxy_client.wss is None:
        return None
    info = build_connection_info(
        _last_notebook_url,
        token=_proxy_client.wss.token,
        port=_proxy_client.wss.port,
        nonce=_connection_nonce,
    )
    _connection_nonce = info.nonce
    return info.as_dict()


@mcp.tool()
async def get_colab_connection_info() -> dict[str, object]:
    """Return current browser and direct-runtime connection information.

    This is a read-only diagnostic: it does not open a browser, wait for a
    connection, or reserve a new manual-handoff target. Use
    ``prepare_colab_browser_connection`` when an existing tab should receive
    fresh credentials.

    The token is returned only because the caller explicitly requested this
    tool. It is never written to logs, the process registry, or diagnostics.
    The ``token`` and ``port`` fields are diagnostic coordinates. For Colab's
    single manual connection field, use ``prepare_colab_browser_connection``
    so the returned ``TOKEN&PORT`` value can be pasted as-is.
    """
    info = _connection_info()
    if info is None and _direct_runtime_manager is None:
        return {
            "connected": False,
            "direct_runtime_connected": False,
            "error": "COLAB_MCP_NOT_INITIALIZED",
            "message": (
                "The local Colab MCP proxy is not initialized yet. "
                "Start the server with proxy support and retry."
            ),
        }
    if info is None:
        info = {}
    info["connected"] = bool(_proxy_client and _proxy_client.is_connected())
    info["direct_runtime_connected"] = bool(
        _direct_runtime_manager and _direct_runtime_manager.connected
    )
    if _direct_runtime_manager and _direct_runtime_manager.endpoint:
        info["runtime_endpoint"] = _direct_runtime_manager.endpoint
    if not info["connected"] and not info["direct_runtime_connected"]:
        info["message"] = (
            "No direct runtime or browser connection is active. With OAuth "
            "configured, call open_colab_browser_connection to attach to an "
            "active runtime. For an already-open tab, call "
            "prepare_colab_browser_connection; otherwise call "
            "open_colab_browser_connection to start the browser handoff."
        )
    return info


@mcp.tool()
async def prepare_colab_browser_connection() -> str:
    """Prepare a manual connection for an already-open Colab tab.

    This tool has no browser side effect and does not wait for a WebSocket
    connection. It returns only the single ``TOKEN&PORT`` value expected by
    Colab's MCP proxy-token field. It does not open or target a notebook URL.
    Once prepared, use the existing tab to connect; do not call the open tool
    again from a second client for the same daemon.
    """
    global _connection_attempt_url, _connection_attempt_started_at

    if _proxy_client is None or _proxy_client.wss is None:
        return (
            "COLAB_MCP_NOT_INITIALIZED: The local Colab MCP proxy is not "
            "initialized yet. Start the server with proxy support and retry."
        )

    async with _connection_open_lock:
        connection_token = f"{_proxy_client.wss.token}&{_proxy_client.wss.port}"
        if _proxy_client.is_connected():
            return connection_token

        # Never replace a handoff that another client has already prepared.
        # Return only the current credentials so the caller can use the same
        # tab without exposing a notebook URL.
        if _connection_attempt_url is not None:
            return connection_token

        # An empty marker reserves this process for the existing-tab handoff
        # without inventing or returning a notebook URL.
        _connection_attempt_url = ""
        _connection_attempt_started_at = time.monotonic()
        return connection_token


@mcp.tool()
async def open_colab_browser_connection(
    notebook_url: str = "",
    open_new_tab: bool = True,
    runtime_endpoint: str = "",
) -> str:
    """Connect to Colab, preferring a direct runtime when OAuth is available.

    ``notebook_url`` may be any existing HTTPS Google Colab notebook URL. If
    omitted, the historical ``empty.ipynb`` scratch notebook is opened. The
    current token and port are attached to a URL fragment. A non-secret
    port/nonce query identifies the initial URL; one MCP process opens at most
    one browser tab, so a shared daemon can serve multiple MCP clients through
    the same Colab tab. Set ``open_new_tab=False`` to prepare the URL without
    launching a browser, then paste it into an already-open target tab. When
    OAuth is configured, ``runtime_endpoint`` selects a direct runtime and no
    browser action is required; if omitted, a single active runtime is chosen.
    """
    if _proxy_client is not None and _proxy_client.is_connected():
        _mark_process_state("connected")
        if notebook_url:
            try:
                normalized = normalize_notebook_url(notebook_url)
            except InvalidNotebookUrl as exc:
                return f"Invalid notebook URL: {exc}"
            return (
                "Already connected to Colab. The requested notebook was not "
                f"opened because this server already has a live browser session "
                f"({normalized}). Use the existing tab and call get_cells to "
                "verify its state; use open_new_tab=false for an explicit "
                "same-tab endpoint change."
            )
        return "Already connected to Colab."

    if runtime_endpoint and _direct_runtime_manager is None:
        return (
            "Direct runtime_endpoint connection requires OAuth setup. Start the "
            "server with --client-oauth-config and retry."
        )

    if _direct_runtime_manager is not None:
        try:
            target = await _direct_runtime_manager.connect(runtime_endpoint)
            _mark_process_state("connected")
            return (
                "Connected directly to Colab runtime "
                f"{target.endpoint}; browser interaction is not required."
            )
        except DirectRuntimeError as exc:
            # An explicit endpoint means the caller requested the direct path;
            # do not silently open a browser for a typo or an ambiguous target.
            if runtime_endpoint:
                return f"Direct Colab runtime connection failed: {exc}"
            logging.info("Direct runtime unavailable; using browser bridge: %s", exc)

    if _proxy_client is None:
        return (
            "COLAB_MCP_NOT_INITIALIZED: The local Colab MCP proxy is not "
            "initialized. Please wait for the server to start and retry."
        )

    try:
        connection_info = build_connection_info(
            notebook_url,
            token=_proxy_client.wss.token,
            port=_proxy_client.wss.port,
        )
    except InvalidNotebookUrl as exc:
        return f"Invalid notebook URL: {exc}"

    global _connection_attempt_url, _connection_attempt_started_at
    global _connection_nonce, _last_notebook_url
    async with _connection_open_lock:
        # Recheck after another concurrent caller has opened the browser. This
        # lets multiple HTTP MCP clients share one daemon without opening
        # duplicate Colab tabs.
        if _proxy_client.is_connected():
            return "Already connected to Colab. Use the existing tab and call get_cells to verify its state."
        if _connection_attempt_url is not None:
            if (
                _connection_attempt_started_at is not None
                and time.monotonic() - _connection_attempt_started_at
                >= CONNECTION_FAILURE_GRACE_SECONDS
            ):
                _mark_process_state("connection_failed")
            if _connection_attempt_url == "":
                return (
                    "A manual connection has already been prepared for this "
                    "MCP process. Enter the returned TOKEN&PORT value in the "
                    "existing Colab tab; no second browser tab was opened."
                )
            return (
                "A browser connection has already been opened by this MCP "
                f"process for {_connection_attempt_url}. No second tab was "
                "opened. Reload that tab or call get_colab_connection_info "
                "to inspect the current connection coordinates."
            )

        if _connection_nonce is not None:
            connection_info = build_connection_info(
                notebook_url,
                token=_proxy_client.wss.token,
                port=_proxy_client.wss.port,
                nonce=_connection_nonce,
            )

        # Keep the token in the fragment. Do not log this URL: it contains a
        # bearer credential. The query string contains only a port and nonce.
        _connection_attempt_url = connection_info.notebook_url
        _connection_attempt_started_at = time.monotonic()
        _connection_nonce = connection_info.nonce
        _last_notebook_url = connection_info.notebook_url
        if not open_new_tab:
            return (
                "Connection URL prepared for the existing Colab tab; the "
                "browser was not opened. Paste this complete URL into that "
                "tab's address bar, then wait for it to connect:\n"
                f"{connection_info.url}\n"
                f"port={connection_info.port}\n"
                "WARNING: this response contains a bearer token. Do not log, "
                "share, or commit it."
            )
        webbrowser.open_new(connection_info.url)

    # Wait for browser to connect
    await _proxy_client.await_proxy_connection()

    if _proxy_client.is_connected():
        _mark_process_state("connected")
        tool_names = await _proxy_client.await_tools_ready()
        tools_text = ", ".join(tool_names) if tool_names else "none discovered"
        return (
            f"Connection successful for {connection_info.notebook_url}. "
            f"Available notebook tools: {tools_text}. You can now create, "
            "edit, and execute cells in the Colab notebook."
        )

    # Mark this instance so a client-launched retry can remove this failed
    # peer without terminating an active Claude/Codex session.
    _mark_process_state("connection_failed")

    # Timed out — surface diagnostic info about other running servers so the
    # user can recognize the "old browser tab pointed at a dead port" case.
    try:
        others = [
            e
            for e in process_registry.list_running(profile=_process_profile)
            if e.pid != os.getpid()
        ]
    except Exception:
        others = []
    my_port = _proxy_client.wss.port
    if others:
        peer_ports = ", ".join(f"{e.port} (pid {e.pid})" for e in others)
        return (
            f"Connection timed out. This server is on port {my_port}, but "
            f"{len(others)} other colab-mcp server(s) are also running: "
            f"{peer_ports}. If you have an old Colab tab open, it may be "
            "pointing at one of those instead of this server. Keep this "
            "daemon running and call prepare_colab_browser_connection, then "
            "paste its TOKEN&PORT value into the existing tab; use `--kill-stale` "
            "only as an explicit maintenance action."
        )
    return (
        f"Connection timed out. This server is on port {my_port}. Common causes:\n"
        "  1. Stale Colab tab(s) - reload the tab opened for this server; "
        "the current token/port are in get_colab_connection_info. Do not "
        "open a second tab from a concurrent MCP client.\n"
        "  2. Local Network Access permission denied - Chrome shows a prompt "
        "the first time Colab tries to reach localhost. Click 'Allow'. If you "
        "previously clicked 'Block', open colab.research.google.com -> site "
        "settings -> reset the 'Insecure content' / 'Other' permission and retry.\n"
        "  3. Browser tab was never opened - make sure your default browser "
        "is set and not blocking pop-ups for python.exe.\n"
        "  4. Existing-tab handoff - call prepare_colab_browser_connection. "
        "It returns one TOKEN&PORT value for Colab's single manual "
        "connection field without opening another tab."
    )


@mcp.tool()
async def add_code_cell(cellIndex: int, code: str = "", language: str = "python") -> str:
    """Add a code cell at ``cellIndex``; use ``-1`` to append.

    ``cellIndex`` is intentionally required so an omitted placement can never
    silently insert a cell at the beginning of an existing notebook.
    Requires an active browser connection via open_colab_browser_connection.
    """
    return await _forward_or_stub("add_code_cell", {"code": code, "cellIndex": cellIndex, "language": language})


@mcp.tool()
async def add_text_cell(cellIndex: int, content: str = "") -> str:
    """Add a Markdown cell at ``cellIndex``; use ``-1`` to append.

    ``cellIndex`` is intentionally required so an omitted placement can never
    silently change the order of an existing notebook. Requires an active
    browser connection via open_colab_browser_connection.
    """
    return await _forward_or_stub("add_text_cell", {"content": content, "cellIndex": cellIndex})


@mcp.tool()
async def get_cells() -> str:
    """Read the current notebook state: list of cells with their IDs, contents, and outputs. Essential for iterative work (write -> run -> read -> adjust). Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("get_cells", {})


@mcp.tool()
async def run_code_cell(cellId: str = "", code: str = "") -> dict[str, object]:
    """Start code in the background and return an execution ID.

    This is the default execution path, so the MCP client remains available
    while any cell runs. If a direct OAuth-backed runtime is connected, pass
    ``code`` and execution/output streaming happen without a browser. The
    ``cellId`` form remains available through the browser notebook bridge.
    """
    if not cellId and not code:
        return {
            "status": "failed",
            "error": "cellId or code is required to start a background code execution.",
        }
    if code and _direct_runtime_manager is not None and not _direct_runtime_manager.connected:
        try:
            await _direct_runtime_manager.connect()
        except DirectRuntimeError as exc:
            return {"status": "failed", "error": str(exc)}
    direct_runtime = (
        _direct_runtime_manager.runtime
        if _direct_runtime_manager and _direct_runtime_manager.connected
        else None
    )
    if direct_runtime is None and (_proxy_client is None or not _proxy_client.is_connected()):
        if code and _direct_runtime_manager is None:
            return {
                "status": "failed",
                "error": (
                    "Direct execution requires OAuth runtime setup. Start with "
                    "--client-oauth-config, then call open_colab_browser_connection "
                    "to attach to a runtime."
                ),
            }
        return {"status": "failed", "error": NOT_CONNECTED_MSG}
    if direct_runtime is not None and not code and (
        _proxy_client is None or not _proxy_client.is_connected()
    ):
        return {
            "status": "failed",
            "error": (
                "Direct runtime execution requires the code argument. The "
                "cellId form is provided by the optional browser notebook bridge."
            ),
        }

    async def runner(publish) -> str:
        if direct_runtime is not None and code:
            outputs = await direct_runtime.execute(code, publish)
            return json.dumps(outputs, ensure_ascii=False)

        result = await _forward_or_stub("run_code_cell", {"cellId": cellId})
        # Treat transport loss as a failed local execution so callers do not
        # mistake an error message for a completed cell result.
        if (
            result == NOT_CONNECTED_MSG
            or result.startswith("COLAB_NOT_CONNECTED:")
            or result.startswith("Error calling ")
        ):
            raise RuntimeError(result)
        publish({"output_type": "result", "text": result})
        return result

    try:
        return await _execution_registry.start(cellId or "<direct>", runner)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return {"status": "failed", "error": str(exc)}


@mcp.tool()
async def run_code_cell_blocking(cellId: str = "", code: str = "") -> str:
    """Execute a code cell and wait for its final browser-side result.

    Prefer run_code_cell for long-running work so the MCP client remains
    available while the cell executes. This blocking variant preserves the
    previous behavior for short cells and callers that need the result in the
    same tool response.
    """
    if code and _direct_runtime_manager is not None and not _direct_runtime_manager.connected:
        try:
            await _direct_runtime_manager.connect()
        except DirectRuntimeError as exc:
            return f"Direct Colab runtime connection failed: {exc}"
    if _direct_runtime_manager and _direct_runtime_manager.runtime and code:
        outputs = await _direct_runtime_manager.runtime.execute(code, lambda _event: None)
        return json.dumps(outputs, ensure_ascii=False)
    if _direct_runtime_manager and _direct_runtime_manager.connected and not code:
        return (
            "Direct runtime execution requires the code argument. The cellId "
            "form is provided by the optional browser notebook bridge."
        )
    return await _forward_or_stub("run_code_cell", {"cellId": cellId})


@mcp.tool()
async def get_code_execution(
    execution_id: str,
    cursor: int = 0,
    wait_seconds: float = 0.0,
) -> dict[str, object]:
    """Get status and incremental output for a background execution.

    ``cursor`` is the last received event ID. Set ``wait_seconds`` to a small
    positive value for long-polling when no new output has arrived yet.
    """
    return await _execution_registry.get_events(
        execution_id,
        cursor=cursor,
        wait_seconds=wait_seconds,
    )


@mcp.tool()
async def list_code_executions() -> list[dict[str, object]]:
    """List retained background code executions for this MCP process."""
    return _execution_registry.list()


@mcp.tool()
async def update_cell(cellId: str = "", content: str = "") -> str:
    """Update the contents of an existing cell in the Colab notebook. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("update_cell", {"cellId": cellId, "content": content})


@mcp.tool()
async def delete_cell(cellId: str = "") -> str:
    """Delete a cell from the Colab notebook by cellId. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("delete_cell", {"cellId": cellId})


@mcp.tool()
async def move_cell(cellId: str = "", cellIndex: int = 0) -> str:
    """Move a cell to a new position in the Colab notebook by cellId and target index. Requires an active browser connection via open_colab_browser_connection."""
    return await _forward_or_stub("move_cell", {"cellId": cellId, "cellIndex": cellIndex})


def _keep_alive_summary(manager) -> str:
    if manager is None or not manager.keep_alive_enabled:
        return "Keep-alive is off; Colab reclaims the runtime once it is idle."
    return (
        f"Keep-alive pings Colab every {int(manager.keep_alive_interval // 60)} min while "
        f"the runtime was used in the last {int(manager.keep_alive_idle_limit // 60)} min "
        "(set_keep_alive to change)."
    )


async def _release_runtime(client, manager, endpoint: str, accelerator: str, by: str) -> str:
    if manager is not None:
        manager.note_released(endpoint, by)
        if manager.endpoint == endpoint:
            await manager.close()
    await asyncio.to_thread(client.unassign, endpoint)
    return f"Released previous runtime {endpoint} ({accelerator}). "


@mcp.tool()
async def change_runtime(accelerator: str = "T4", force_new: bool = False) -> str:
    """Get a Colab runtime with the given accelerator (NONE, T4, L4, A100) and connect to it directly. Idempotent: if this session's runtime already has that accelerator it is reused (force_new=true replaces it). The old runtime is released only after the new one exists. If Colab is slow and the call times out, calling it again resumes the same request instead of creating a second runtime. Only this session's runtime is ever replaced. Requires OAuth setup."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    from colab_mcp.client import (
        ASSIGN_TIMEOUT_SECONDS,
        Accelerator,
        ColabRequestError,
        TooManyAssignmentsError,
        Variant,
    )
    import requests
    import uuid

    try:
        acc = Accelerator(accelerator.upper())
    except ValueError:
        return f"Unknown accelerator '{accelerator}'. Use NONE, T4, L4 or A100."
    variant = Variant.GPU if acc != Accelerator.NONE else Variant.DEFAULT
    # Pin the account for the whole call: switch_account may run while Colab
    # is still answering, and every later step must stay on this account.
    client, manager, account = _colab_client, _direct_runtime_manager, _active_account

    async with _change_runtime_locks.setdefault(account, asyncio.Lock()):
        try:
            assignments = await asyncio.to_thread(client.list_assignments) or []
            own_endpoint = None
            if manager is not None:
                own_endpoint = manager.endpoint or manager.last_endpoint
            own = next((a for a in assignments if a.endpoint == own_endpoint), None)

            if own is not None and own.accelerator == acc and not force_new:
                if manager is not None and not (
                    manager.connected and manager.endpoint == own.endpoint
                ):
                    await manager.connect(own.endpoint)
                rec = manager.record(own.endpoint) if manager is not None else None
                fit = rec.idle_timeout_sec if rec is not None else None
                return (
                    f"This session already has a {acc.value} runtime ({own.endpoint}); "
                    "reused it, nothing was released. Pass force_new=true to replace it. "
                    + (f"Idle timeout: {fit} s. " if fit else "")
                    + _keep_alive_summary(manager)
                )

            # One notebook hash per (account, accelerator) until it succeeds, so a
            # retry after a timeout picks up the runtime Colab created meanwhile.
            key = (account, acc.value)
            notebook_hash = _pending_assignments.setdefault(key, str(uuid.uuid4()))
            released = ""
            try:
                try:
                    result = await asyncio.to_thread(
                        client.assign, notebook_hash, variant, acc
                    )
                except TooManyAssignmentsError:
                    if own is None:
                        listed = ", ".join(
                            f"{a.endpoint} ({a.accelerator.value})" for a in assignments
                        )
                        return (
                            "Colab refused another runtime for this account (too many "
                            f"assignments). Active runtimes: {listed or 'none'}. They may "
                            "belong to another session; stop one explicitly with "
                            "stop_runtime(runtime_endpoint=...) if it is safe."
                        )
                    # Make room by releasing this session's own runtime, then retry.
                    released = await _release_runtime(
                        client, manager,
                        own.endpoint, own.accelerator.value, f"change_runtime to {acc.value}"
                    )
                    own = None
                    result = await asyncio.to_thread(
                        client.assign, notebook_hash, variant, acc
                    )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                ColabRequestError,
            ) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, ColabRequestError) and (status is None or status < 500):
                    raise
                kept = (
                    f"Your current runtime {own.endpoint} ({own.accelerator.value}) was not touched. "
                    if own is not None
                    else ""
                )
                problem = (
                    f"Colab answered HTTP {status} ({exc.response.reason}) while assigning a "
                    f"{acc.value} runtime. This is a temporary error on Google's side, often "
                    "while GPUs are scarce."
                    if isinstance(exc, ColabRequestError)
                    else f"Colab did not answer within {ASSIGN_TIMEOUT_SECONDS} s while "
                    f"assigning a {acc.value} runtime ({type(exc).__name__}). It may still be "
                    "provisioning."
                )
                return (
                    f"{released}{problem} {kept}Call change_runtime('{acc.value}') again in a "
                    "minute: it resumes the same request and will not create a second runtime."
                )
            _pending_assignments.pop(key, None)

            if isinstance(result, dict):
                assignment = result["assignment"]
                endpoint = assignment.endpoint
                fit = getattr(assignment, "idle_timeout_sec", None)
            else:
                endpoint = result.endpoint
                fit = result.idle_timeout_sec

            if own is not None and own.endpoint != endpoint:
                released = await _release_runtime(
                    client, manager, own.endpoint, own.accelerator.value, f"change_runtime to {acc.value}"
                )
            fit_text = f"Idle timeout: {fit} s. " if fit else ""

            if manager is not None:
                # The assignment response in older Colab API variants does not
                # always include runtimeProxyInfo. Refresh the assignment list to
                # obtain the current short-lived URL/token pair.
                listed = await asyncio.to_thread(client.list_assignments)
                selected = next(
                    (item for item in listed or [] if item.endpoint == endpoint), None
                )
                if selected is not None:
                    await manager.attach_target(
                        RuntimeTarget(
                            endpoint=selected.endpoint,
                            url=selected.runtime_proxy_info.url,
                            token=selected.runtime_proxy_info.token,
                        )
                    )
                    manager.note_assigned(endpoint, acc.value, fit)
                    return (
                        f"{released}New {acc.value} runtime {endpoint}, directly connected "
                        f"(browser not required). {fit_text}{_keep_alive_summary(manager)}"
                    )
            return (
                f"{released}New {acc.value} runtime {endpoint}. {fit_text}"
                "Use open_colab_browser_connection to connect to it."
            )
        except Exception as e:
            return f"Failed to change runtime: {e}"


@mcp.tool()
async def stop_runtime(runtime_endpoint: str = "", all_runtimes: bool = False) -> str:
    """Stop (unassign) a Colab runtime so it stops consuming compute quota. By default stops only the runtime this MCP session is connected to; pass runtime_endpoint for a specific one, or all_runtimes=true to stop every runtime on the account (including ones other sessions are using). Requires OAuth setup."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        assignments = await asyncio.to_thread(_colab_client.list_assignments)
        assignments = assignments or []
        own_endpoint = (
            _direct_runtime_manager.endpoint if _direct_runtime_manager is not None else None
        )
        if runtime_endpoint:
            targets = [a for a in assignments if a.endpoint == runtime_endpoint]
            if not targets:
                return f"No active Colab runtime was found for endpoint '{runtime_endpoint}'."
        elif all_runtimes:
            targets = assignments
        else:
            targets = [a for a in assignments if a.endpoint == own_endpoint]
            if not targets:
                if not assignments:
                    return "No active Colab runtime to stop."
                listed = ", ".join(f"{a.endpoint} ({a.accelerator.value})" for a in assignments)
                return (
                    "This session is not connected to any runtime, so nothing was stopped. "
                    f"Active runtimes on the account: {listed}. They may belong to another "
                    "session; pass runtime_endpoint or all_runtimes=true to stop them."
                )
        if not targets:
            return "No active Colab runtime to stop."

        stopped = []
        for a in targets:
            if _direct_runtime_manager is not None:
                _direct_runtime_manager.note_released(a.endpoint, "stop_runtime")
                if _direct_runtime_manager.endpoint == a.endpoint:
                    await _direct_runtime_manager.close()
            await asyncio.to_thread(_colab_client.unassign, a.endpoint)
            stopped.append(f"{a.endpoint} ({a.accelerator.value})")
        return "Stopped runtime(s): " + ", ".join(stopped)
    except Exception as e:
        return f"Failed to stop runtime: {e}"


@mcp.tool()
async def list_runtimes() -> str:
    """List active Colab runtimes (endpoint, accelerator, whether this MCP is directly connected) for this session's active account (see list_accounts/switch_account), plus its compute-unit balance and hourly burn. Requires OAuth setup."""
    if _colab_client is None:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        assignments = await asyncio.to_thread(_colab_client.list_assignments)
        connected = (
            _direct_runtime_manager.endpoint
            if _direct_runtime_manager is not None and _direct_runtime_manager.connected
            else None
        )
        result: dict[str, object] = {
            "account": _active_account,
            "runtimes": [
                {
                    "endpoint": a.endpoint,
                    "accelerator": a.accelerator.value,
                    "variant": a.variant.name,
                    "machine_shape": a.machine_shape.name,
                    "direct_connected": a.endpoint == connected,
                    **(
                        _direct_runtime_manager.records[a.endpoint].as_dict()
                        if _direct_runtime_manager is not None
                        and a.endpoint in _direct_runtime_manager.records
                        else {"idle_timeout_sec": None}
                    ),
                }
                for a in assignments or []
            ],
            "keep_alive": _keep_alive_summary(_direct_runtime_manager),
        }
        try:
            ccu = await asyncio.to_thread(_colab_client.get_ccu_info)
            result["compute_units"] = {
                "balance": ccu.current_balance,
                "consumption_per_hour": ccu.consumption_rate_hourly,
            }
        except Exception as exc:
            result["compute_units"] = f"unavailable: {exc}"
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Failed to list runtimes: {e}"


@mcp.tool()
async def set_keep_alive(enabled: bool = True, idle_limit_minutes: float = 60, interval_minutes: float = 5) -> str:
    """Control the keep-alive that stops Colab reclaiming this session's runtime as idle (it otherwise disappears ~20-30 min after the last call, losing /content). While enabled, Colab is pinged every interval_minutes as long as the runtime was used in the last idle_limit_minutes. Disable it to let an unused runtime be reclaimed sooner. Applies to all accounts in this MCP process."""
    global _keep_alive_interval, _keep_alive_idle_limit
    if enabled and (idle_limit_minutes <= 0 or interval_minutes <= 0):
        return "idle_limit_minutes and interval_minutes must be positive; use enabled=false to turn keep-alive off."
    _keep_alive_interval = interval_minutes * 60 if enabled else 0
    _keep_alive_idle_limit = idle_limit_minutes * 60 if enabled else 0
    for _client, manager in _accounts.values():
        manager.keep_alive_interval = _keep_alive_interval
        manager.keep_alive_idle_limit = _keep_alive_idle_limit
        if manager.endpoint:
            manager.record(manager.endpoint).keep_alive_state = "on" if enabled else "off"
    return _keep_alive_summary(_direct_runtime_manager if _direct_runtime_manager is not None else None) if enabled else "Keep-alive is off; Colab reclaims runtimes once they are idle."


def _account_pair(email: str, session=None):
    """Return (and cache) an account's (ColabClient, DirectRuntimeManager)."""
    from colab_mcp import auth
    from colab_mcp.client import ColabClient, Prod

    if email not in _accounts:
        if session is None:
            session = auth.get_credentials(
                _client_oauth_config,
                token_path=auth.account_token_path(email),
                login_hint=email,
            )
        client = ColabClient(Prod(), session)
        _accounts[email] = (
            client,
            DirectRuntimeManager(client, _keep_alive_interval, _keep_alive_idle_limit),
        )
    return _accounts[email]


def _activate_account(email: str, session=None) -> None:
    """Make ``email`` this session's account, loading its client on first use."""
    global _colab_client, _direct_runtime_manager, _active_account
    _colab_client, _direct_runtime_manager = _account_pair(email, session)
    _active_account = email


@mcp.tool()
async def list_accounts(include_compute_units: bool = True) -> str:
    """List the Google accounts signed in to this MCP, which one this session uses (active), the default for new sessions, and each account's runtimes and compute-unit balance. Use it to pick the account with GPU quota left, then switch_account."""
    from colab_mcp import auth

    if not _client_oauth_config:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    default = auth.get_default_account()
    rows = []
    for email in auth.list_saved_accounts():
        row: dict[str, object] = {
            "email": email,
            "active": email == _active_account,
            "default": email == default,
        }
        if include_compute_units:
            try:
                client, manager = await asyncio.to_thread(_account_pair, email)
                assignments = await asyncio.to_thread(client.list_assignments)
                row["runtimes"] = [
                    {
                        "endpoint": a.endpoint,
                        "accelerator": a.accelerator.value,
                        "direct_connected": manager.connected and manager.endpoint == a.endpoint,
                    }
                    for a in assignments or []
                ]
                ccu = await asyncio.to_thread(client.get_ccu_info)
                row["compute_units"] = {
                    "balance": ccu.current_balance,
                    "consumption_per_hour": ccu.consumption_rate_hourly,
                }
            except Exception as e:
                row["error"] = str(e)
        rows.append(row)
    return json.dumps({"accounts": rows}, indent=2)


@mcp.tool()
async def add_account(timeout_seconds: int = 300, make_default: bool = False) -> str:
    """Sign in another Google account. Opens the Google account chooser in the laptop's browser; the user picks (or adds) the account and approves Colab and Drive access on one consent screen within timeout_seconds. The new account becomes this session's active account. Google may show an "app not verified" warning; choose Advanced → continue."""
    from colab_mcp import auth

    if not _client_oauth_config:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        email, session, drive_granted = await asyncio.to_thread(
            auth.authorize_new_account, _client_oauth_config, timeout_seconds
        )
    except Exception as e:
        return f"Sign-in failed or was not completed: {e}"
    # An account that is already loaded keeps its client (and any direct
    # runtime connection); its token file was just refreshed on disk.
    _activate_account(email, session)
    if make_default or auth.get_default_account() is None:
        auth.set_default_account(email)
    return (
        f"Signed in {email}; this session now uses it."
        + (" It is also the default for new sessions." if make_default else "")
        + (
            " Drive access granted (fast transfers and create_notebook ready)."
            if drive_granted
            else " Drive access was not granted, so large transfers use the slower kernel route."
        )
    )


@mcp.tool()
async def switch_account(email: str, make_default: bool = False) -> str:
    """Switch this session to another signed-in account (see list_accounts). Runtime, file and notebook tools then act on that account. Runtimes and running executions of the previous account keep running; switch back to manage them. make_default also makes it the account new sessions start with."""
    from colab_mcp import auth

    if not _client_oauth_config:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    email = email.strip().lower()
    saved = auth.list_saved_accounts()
    if email not in saved:
        return f"Account {email} is not signed in. Signed-in accounts: {', '.join(saved) or 'none'}. Use add_account to sign in a new one."
    try:
        await asyncio.to_thread(_account_pair, email)
    except Exception as e:
        return f"Failed to load account {email}: {e}"
    _activate_account(email)
    if make_default:
        auth.set_default_account(email)
    return f"This session now uses {email}." + (
        " It is also the default for new sessions." if make_default else ""
    )


@mcp.tool()
async def remove_account(email: str) -> str:
    """Sign out an account: delete its saved tokens from this laptop. Its Colab runtimes are not stopped (use stop_runtime first). The active account cannot be removed; switch_account away first."""
    from colab_mcp import auth

    email = email.strip().lower()
    if email not in auth.list_saved_accounts():
        return f"Account {email} is not signed in."
    if email == _active_account:
        return f"{email} is this session's active account; switch_account to another account first."
    if email in _accounts:
        await _accounts.pop(email)[1].close()
    auth.remove_account(email)
    return f"Removed {email}. Other running MCP sessions that already loaded it keep working until they restart."


async def _connected_direct_runtime():
    """Return the runtime's side kernel for file and shell tools, or an error string.

    The side kernel is a second kernel on the same runtime, so these tools keep
    working while a cell occupies the main kernel.  If it cannot be started the
    main kernel is used instead.
    """
    manager = _direct_runtime_manager
    if manager is None:
        return "Direct runtime access requires --client-oauth-config."
    if not manager.connected:
        try:
            await manager.connect()
        except DirectRuntimeError as exc:
            return f"Direct Colab runtime connection failed: {exc}"
    try:
        return await manager.side()
    except Exception:
        logging.getLogger(__name__).warning(
            "Side kernel unavailable; using the main kernel", exc_info=True
        )
        return manager.runtime


def _shell_result(outputs) -> dict:
    from colab_mcp import transfer

    return json.loads(transfer.stdout_text(outputs).strip())


@mcp.tool()
async def run_shell(
    command: str,
    cwd: str = "/content",
    timeout_seconds: float = 300,
    background: bool = False,
) -> str:
    """Run a bash command on the directly connected runtime and return stdout, stderr and exit_code as JSON.

    It runs in a separate side kernel, so it works while a cell is still running in the notebook kernel (check GPU use, tail logs, pip install, etc.). There is no TTY: interactive programs (vim, htop, prompts) do not work; use -y / batch flags.

    A foreground command is killed after timeout_seconds (timed_out: true, partial output kept); only the last 20000 characters of each stream are returned. Shell commands share one side kernel, so a long foreground command delays the next one.

    background=True starts the command detached and returns a job_id at once; its output goes to a log file. Use shell_job to read its state and log tail, list jobs or kill one. Background jobs survive the MCP session but not the runtime.
    """
    from colab_mcp import shell

    if not command.strip():
        return "command is empty."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    try:
        if background:
            job_id = uuid.uuid4().hex[:8]
            outputs = await runtime.execute(
                shell.start_job_code(job_id, command, cwd), lambda _event: None, timeout=60
            )
            info = _shell_result(outputs)
            info["note"] = f"Started in the background. Check it with shell_job(job_id='{job_id}')."
            return json.dumps(info, ensure_ascii=False)
        timeout = max(1.0, float(timeout_seconds))
        outputs = await runtime.execute(
            shell.run_code(command, cwd, timeout), lambda _event: None, timeout=timeout + 60
        )
        return json.dumps(_shell_result(outputs), ensure_ascii=False)
    except Exception as e:
        return f"run_shell failed: {e}"


@mcp.tool()
async def shell_job(job_id: str = "", action: str = "status", tail_lines: int = 50) -> str:
    """Inspect background jobs started by run_shell(background=True).

    action="status" (default) returns state (running / exited / killed), exit_code, elapsed seconds and the last tail_lines of the log. action="list" lists every job on the runtime (job_id may be empty). action="kill" stops the job's whole process group (SIGTERM, then SIGKILL).
    """
    from colab_mcp import shell

    if action not in ("status", "list", "kill"):
        return "action must be 'status', 'list' or 'kill'."
    if action != "list" and not job_id:
        return f"job_id is required for action '{action}'."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    if action == "list":
        code = shell.job_list_code()
    elif action == "kill":
        code = shell.job_kill_code(job_id)
    else:
        code = shell.job_status_code(job_id, max(0, tail_lines))
    try:
        outputs = await runtime.execute(code, lambda _event: None, timeout=60)
        return json.dumps(_shell_result(outputs), ensure_ascii=False)
    except Exception as e:
        return f"shell_job failed: {e}"


def _drive_session_for_transfer(force: bool):
    """Drive session of the active account for relaying files, or None.

    Without ``force`` only an account that already granted Drive access is
    used, so a transfer never stops to wait for a consent screen.
    """
    from colab_mcp import auth

    if not _client_oauth_config:
        return None
    if _active_account:
        token_path = auth.account_drive_token_path(_active_account)
        kwargs = {"token_path": token_path, "login_hint": _active_account}
    else:
        token_path = auth.DRIVE_TOKEN_CONFIG_PATH
        kwargs = {}
    if not force and not os.path.exists(token_path):
        return None
    return auth.get_drive_session(_client_oauth_config, **kwargs)


def _route_note(route: str, size: int, seconds: float, note: str = "") -> str:
    rate = size / 1e6 / seconds if seconds > 0 else 0
    return f"via {route}, {rate:.2f} MB/s" + (f"; {note}" if note else "")


async def _push(runtime, source: str, remote_path: str, via: str) -> tuple[int, str]:
    """Send a local file to the runtime by Drive or kernel; return (size, route note)."""
    from colab_mcp import drive_relay, transfer

    size = os.path.getsize(source)
    started = time.monotonic()
    note = ""
    if via == "drive" or (via == "auto" and size >= drive_relay.DRIVE_THRESHOLD_BYTES):
        try:
            session = await asyncio.to_thread(_drive_session_for_transfer, via == "drive")
            if session is not None:
                written = await drive_relay.push_file(runtime, session, source, remote_path)
                return written, _route_note("Google Drive", written, time.monotonic() - started)
            note = "no Drive access for this account yet (call with via='drive' once to grant it)"
        except Exception as exc:
            if via == "drive":
                raise
            note = f"Drive relay failed ({exc}), fell back to the kernel"
        started = time.monotonic()
    written = await transfer.push_file(runtime, source, remote_path)
    return written, _route_note("kernel", written, time.monotonic() - started, note)


async def _pull(runtime, remote_path: str, target: str, via: str) -> tuple[int | None, str]:
    """Fetch a runtime file by Drive or kernel; (None, "") if it does not exist."""
    from colab_mcp import drive_relay, transfer

    outputs = await runtime.execute(transfer.stat_code(remote_path), lambda _event: None)
    size = json.loads(transfer.stdout_text(outputs).strip())["size"]
    if size is None:
        return None, ""
    started = time.monotonic()
    note = ""
    if via == "drive" or (via == "auto" and size >= drive_relay.DRIVE_THRESHOLD_BYTES):
        try:
            session = await asyncio.to_thread(_drive_session_for_transfer, via == "drive")
            if session is not None:
                received = await drive_relay.pull_file(runtime, session, remote_path, target)
                return received, _route_note("Google Drive", received, time.monotonic() - started)
            note = "no Drive access for this account yet (call with via='drive' once to grant it)"
        except Exception as exc:
            if via == "drive":
                raise
            note = f"Drive relay failed ({exc}), fell back to the kernel"
        started = time.monotonic()
    received = await transfer.pull_file(runtime, remote_path, target)
    return received, _route_note("kernel", received or 0, time.monotonic() - started, note)


_VIA_CHOICES = ("auto", "drive", "kernel")


@mcp.tool()
async def upload_to_runtime(local_path: str, remote_path: str = "", via: str = "auto") -> str:
    """Copy a local file to the directly connected Colab runtime (default destination /content/<file name>); no browser is needed. For several files or a directory use upload_files_to_runtime. via: 'auto' (default) relays files of 20 MB or more through Google Drive (several times faster than the kernel) when this account has granted Drive access, else uses the kernel; 'drive' forces Drive (asks for Drive consent once); 'kernel' forces the kernel route (1 GiB max)."""
    if via not in _VIA_CHOICES:
        return f"via must be one of {', '.join(_VIA_CHOICES)}."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    source = os.path.expanduser(local_path)
    if not os.path.isfile(source):
        return f"Local file not found: {local_path}"
    remote_path = remote_path or f"/content/{os.path.basename(source)}"
    try:
        written, route = await _push(runtime, source, remote_path, via)
        return f"Uploaded {local_path} to {remote_path} ({written} bytes, {route})."
    except Exception as e:
        return f"Failed to upload {local_path}: {e}"


@mcp.tool()
async def download_from_runtime(remote_path: str, local_path: str = "", via: str = "auto") -> str:
    """Copy a file from the directly connected Colab runtime to this machine (default destination ~/Downloads/colab/<file name>); no browser is needed. For several files, globs or a directory use download_files_from_runtime. via: 'auto' (default) relays files of 20 MB or more through Google Drive (several times faster than the kernel) when this account has granted Drive access, else uses the kernel; 'drive' forces Drive (asks for Drive consent once); 'kernel' forces the kernel route (1 GiB max)."""
    if via not in _VIA_CHOICES:
        return f"via must be one of {', '.join(_VIA_CHOICES)}."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    target = os.path.expanduser(
        local_path or os.path.join("~/Downloads/colab", os.path.basename(remote_path))
    )
    try:
        size, route = await _pull(runtime, remote_path, target, via)
        if size is None:
            return f"Remote file not found: {remote_path}"
        return f"Downloaded {remote_path} to {target} ({size} bytes, {route})."
    except Exception as e:
        return f"Failed to download {remote_path}: {e}"


@mcp.tool()
async def upload_files_to_runtime(local_paths: list[str], remote_dir: str = "/content", via: str = "auto") -> str:
    """Upload several local files and/or directories (glob patterns and ~ allowed) to remote_dir on the directly connected runtime in one compressed transfer. Each entry keeps its base name; directories are copied recursively. via: 'auto' (default) relays files of 20 MB or more through Google Drive (several times faster than the kernel) when this account has granted Drive access, else uses the kernel; 'drive' forces Drive (asks for Drive consent once); 'kernel' forces the kernel route (1 GiB max)."""
    import uuid
    from colab_mcp import transfer

    if via not in _VIA_CHOICES:
        return f"via must be one of {', '.join(_VIA_CHOICES)}."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    if not local_paths:
        return "local_paths is empty."
    remote_archive = f"/tmp/colab-mcp-upload-{uuid.uuid4().hex}.tar.gz"
    try:
        sources = transfer.expand_local(local_paths)
        with tempfile.TemporaryDirectory(prefix="colab-mcp-") as workdir:
            archive = os.path.join(workdir, "batch.tar.gz")
            await asyncio.to_thread(transfer.pack_local, sources, archive)
            size, route = await _push(runtime, archive, remote_archive, via)
        outputs = await runtime.execute(
            transfer.remote_unpack_code(remote_archive, remote_dir), lambda _event: None
        )
        names = json.loads(transfer.stdout_text(outputs).strip())
        return json.dumps(
            {"remote_dir": remote_dir, "uploaded": names, "compressed_bytes": size, "route": route},
            ensure_ascii=False,
        )
    except Exception as e:
        try:
            await runtime.execute(transfer.remove_remote_code(remote_archive), lambda _event: None)
        except Exception:
            pass
        return f"Failed to upload files: {e}"


@mcp.tool()
async def download_files_from_runtime(remote_paths: list[str], local_dir: str = "~/Downloads/colab", via: str = "auto") -> str:
    """Download several runtime files and/or directories (glob patterns allowed, e.g. /content/outputs/*.png) into local_dir in one compressed transfer. Each match keeps its base name; directories are copied recursively. via: 'auto' (default) relays files of 20 MB or more through Google Drive (several times faster than the kernel) when this account has granted Drive access, else uses the kernel; 'drive' forces Drive (asks for Drive consent once); 'kernel' forces the kernel route (1 GiB max)."""
    import uuid
    from colab_mcp import transfer

    if via not in _VIA_CHOICES:
        return f"via must be one of {', '.join(_VIA_CHOICES)}."
    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    if not remote_paths:
        return "remote_paths is empty."
    remote_archive = f"/tmp/colab-mcp-download-{uuid.uuid4().hex}.tar.gz"
    dest = os.path.expanduser(local_dir)
    try:
        outputs = await runtime.execute(
            transfer.remote_pack_code(remote_paths, remote_archive), lambda _event: None
        )
        report = json.loads(transfer.stdout_text(outputs).strip())
        if not report["found"]:
            return f"No remote files matched: {', '.join(report['missing'])}"
        with tempfile.TemporaryDirectory(prefix="colab-mcp-") as workdir:
            archive = os.path.join(workdir, "batch.tar.gz")
            size, route = await _pull(runtime, remote_archive, archive, via)
            names = await asyncio.to_thread(transfer.unpack_local, archive, dest)
        return json.dumps(
            {
                "local_dir": dest,
                "downloaded": names,
                "missing": report["missing"],
                "compressed_bytes": size,
                "route": route,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Failed to download files: {e}"
    finally:
        try:
            await runtime.execute(transfer.remove_remote_code(remote_archive), lambda _event: None)
        except Exception:
            pass


@mcp.tool()
async def list_runtime_files(path: str = "/content", recursive: bool = False, limit: int = 500) -> str:
    """List files on the directly connected runtime with type, size in bytes and UTC modification time. recursive walks subdirectories; at most limit entries are returned."""
    from colab_mcp import transfer

    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    try:
        outputs = await runtime.execute(
            transfer.list_remote_code(path, recursive, max(1, limit)), lambda _event: None
        )
        result = json.loads(transfer.stdout_text(outputs).strip())
        if "error" in result:
            return f"Remote path not found: {path}"
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Failed to list {path}: {e}"


@mcp.tool()
async def delete_runtime_files(remote_paths: list[str]) -> str:
    """Delete files and/or directories (recursively) on the directly connected runtime. Glob patterns are allowed; the result lists what was deleted and which patterns matched nothing."""
    from colab_mcp import transfer

    runtime = await _connected_direct_runtime()
    if isinstance(runtime, str):
        return runtime
    if not remote_paths:
        return "remote_paths is empty."
    try:
        outputs = await runtime.execute(
            transfer.delete_remote_code(remote_paths), lambda _event: None
        )
        return transfer.stdout_text(outputs).strip()
    except Exception as e:
        return f"Failed to delete files: {e}"


@mcp.tool()
async def create_notebook(
    title: str,
    cells: list[dict[str, str]] | None = None,
    accelerator: str = "",
    folder_id: str = "",
) -> str:
    """Create a new Colab notebook in Google Drive and return its id and URL (open it with open_colab_browser_connection(notebook_url=...)). cells is an optional list of {"type": "code"|"markdown", "source": "..."}; accelerator (T4/L4/A100) is saved as the notebook's default GPU. The first call asks for Drive (drive.file) consent in the browser."""
    import uuid
    from colab_mcp import transfer
    from colab_mcp.auth import get_drive_session

    if not _client_oauth_config:
        return "Runtime API not initialized. Start with --client-oauth-config flag pointing to your OAuth client secrets JSON."
    try:
        notebook = transfer.build_notebook(cells, accelerator)
    except ValueError as exc:
        return str(exc)
    name = title if title.endswith(".ipynb") else f"{title}.ipynb"
    metadata: dict[str, object] = {"name": name, "mimeType": transfer.COLAB_MIME_TYPE}
    if folder_id:
        metadata["parents"] = [folder_id]
    boundary = uuid.uuid4().hex
    try:
        from colab_mcp.auth import account_drive_token_path

        drive_kwargs = (
            {"token_path": account_drive_token_path(_active_account), "login_hint": _active_account}
            if _active_account
            else {}
        )
        session = await asyncio.to_thread(
            get_drive_session, _client_oauth_config, **drive_kwargs
        )
        response = await asyncio.to_thread(
            session.post,
            "https://www.googleapis.com/upload/drive/v3/files",
            params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            data=transfer.multipart_body(
                metadata, json.dumps(notebook).encode(), boundary
            ),
        )
        if response.status_code >= 400:
            return f"Failed to create notebook: HTTP {response.status_code} {response.text[:500]}"
        file_id = response.json()["id"]
        return json.dumps(
            {
                "id": file_id,
                "name": name,
                "url": f"https://colab.research.google.com/drive/{file_id}",
            }
        )
    except Exception as e:
        return f"Failed to create notebook: {e}"


def init_logger(logdir):
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v):
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "-p",
        "--enable-proxy",
        help="if set, enable the runtime proxy (enabled by default).",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--client-oauth-config",
        help=(
            "Path to OAuth client secrets JSON for direct runtime execution "
            "and the change_runtime tool."
        ),
        action="store",
        default=None,
    )
    parser.add_argument(
        "--keep-alive-interval",
        type=float,
        default=KEEP_ALIVE_INTERVAL_SECONDS,
        help="Seconds between Colab keep-alive pings for the connected runtime (0 disables).",
    )
    parser.add_argument(
        "--keep-alive-idle-limit",
        type=float,
        default=KEEP_ALIVE_IDLE_LIMIT_SECONDS / 60,
        help=(
            "Stop keep-alive pings after this many minutes without MCP activity on the "
            "runtime, so Colab can reclaim it (0 disables keep-alive)."
        ),
    )
    parser.add_argument(
        "--list-running",
        help="List all currently-running colab-mcp servers and exit.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--kill-stale",
        help="Explicitly terminate verified colab-mcp peers in this profile and exit. Does not run during normal startup.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stop-pid",
        type=int,
        default=None,
        metavar="PID",
        help="Explicitly stop one verified colab-mcp process in this profile and exit.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Explicitly stop verified peers in this profile before starting.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("COLAB_MCP_PROFILE", "default"),
        help="Process group used by --replace/--kill-stale/--stop-pid (default: %(default)s).",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        default=False,
        help="With an explicit stop/kill action, include every profile. Never implied by normal startup.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
        help=(
            "MCP transport. stdio is the default for client-launched servers; "
            "use streamable-http for one shared local daemon."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP transport bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--allow-insecure-non-loopback",
        action="store_true",
        default=False,
        help=(
            "Explicitly allow an unauthenticated HTTP transport to bind outside "
            "loopback. This can expose notebook tools and bearer credentials."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="HTTP transport port (FastMCP default when omitted); ignored for stdio.",
    )
    parser.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help="HTTP transport endpoint path (FastMCP default /mcp when omitted).",
    )
    args = parser.parse_args(v)
    if (
        args.transport != "stdio"
        and not _is_loopback_host(args.host)
        and not args.allow_insecure_non_loopback
    ):
        parser.error(
            "non-loopback HTTP binding is unauthenticated and disabled by default; "
            "use --host 127.0.0.1 or explicitly acknowledge the risk with "
            "--allow-insecure-non-loopback"
        )
    return args


def _is_loopback_host(host: str) -> bool:
    """Return whether an HTTP bind host is unambiguously loopback-only."""
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _print_running_servers() -> None:
    entries = process_registry.list_running(profile=_process_profile)
    if not entries:
        print("No colab-mcp servers currently registered as running.")
        return
    print(f"Found {len(entries)} running colab-mcp server(s):")
    import datetime as _dt
    for e in entries:
        started = _dt.datetime.fromtimestamp(e.started_at).strftime("%Y-%m-%d %H:%M:%S")
        port = str(e.port) if e.port else "?"
        host = e.host or "unknown"
        print(
            f"  pid={e.pid:<6}  port={port:<6}  host={host}  "
            f"profile={e.profile}  started={started}"
        )


async def _run_mcp_transport(args) -> None:
    """Run the configured MCP transport without changing stdio defaults."""
    if args.transport == "stdio":
        await mcp.run_async()
        return

    kwargs = {
        "transport": args.transport,
        "host": args.host,
    }
    if args.port is not None:
        kwargs["port"] = args.port
    if args.path is not None:
        kwargs["path"] = args.path
    await mcp.run_async(**kwargs)


async def main_async():
    global _proxy_client, _session_mcp, _colab_client, _direct_runtime_manager
    global _process_profile, _process_entry, _client_oauth_config
    global _keep_alive_interval, _keep_alive_idle_limit
    args = parse_args(sys.argv[1:])
    _process_profile = args.profile
    init_logger(args.log)

    # Diagnostic / cleanup flags exit early.
    if args.list_running:
        _print_running_servers()
        return
    if args.stop_pid is not None:
        stopped = process_registry.stop(
            args.stop_pid,
            profile=None if args.all_profiles else args.profile,
        )
        if stopped is None:
            print(
                f"Refused to stop pid={args.stop_pid}: it is not a verified "
                f"colab-mcp server in profile '{args.profile}'."
            )
        else:
            print(f"Stopped colab-mcp pid={stopped.pid} port={stopped.port or '?'}.")
        return
    if args.kill_stale:
        removed = process_registry.cleanup_stale(
            kill=True,
            profile=None if args.all_profiles else args.profile,
        )
        if not removed:
            print("No stale colab-mcp servers found.")
        else:
            print(f"Terminated {len(removed)} stale colab-mcp server(s):")
            for e in removed:
                print(f"  pid={e.pid} port={e.port or '?'} profile={e.profile}")
        return

    # Normal startup never kills a live peer. Replacement is explicit and
    # profile-scoped, so Claude and Codex can coexist without killing one
    # another accidentally.
    if args.replace:
        replaced = process_registry.replace_existing(profile=args.profile)
        logging.info(
            "Explicitly replaced %d verified colab-mcp peer(s) in profile=%s",
            len(replaced),
            args.profile,
        )

    # Prune dead/PID-reused entries before binding a port. This does not signal
    # any process and is safe for normal startup.
    dead = process_registry.prune_dead()
    if dead:
        logging.info(f"Pruned {dead} stale entries from process registry")

    failed = process_registry.cleanup_unhealthy(
        profile=args.profile,
        transport=args.transport,
    )
    if failed:
        logging.info(
            "Removed %d failed %s peer(s) in profile=%s",
            len(failed),
            args.transport,
            args.profile,
        )

    if args.enable_proxy:
        logging.info("enabling session proxy tools")
        _session_mcp = ColabSessionProxy()
        await _session_mcp.start_proxy_server()
        _proxy_client = _session_mcp.proxy_client
        # Register ourselves now that we know the port.
        try:
            _process_entry = process_registry.register(
                port=_session_mcp.wss.port,
                host=_session_mcp.wss.host,
                profile=args.profile,
            )
            logging.info(
                "Registered colab-mcp pid=%s port=%s profile=%s",
                _process_entry.pid,
                _process_entry.port,
                _process_entry.profile,
            )
        except Exception as exc:
            logging.warning(f"Could not register process: {exc}")

    _client_oauth_config = args.client_oauth_config
    _keep_alive_interval = args.keep_alive_interval
    _keep_alive_idle_limit = args.keep_alive_idle_limit * 60
    if args.client_oauth_config:
        try:
            from colab_mcp import auth
            logging.info("initializing Colab API client with OAuth")
            auth.migrate_legacy_token(args.client_oauth_config)
            account = auth.get_default_account()
            if account is None:
                account, session, _drive = auth.authorize_new_account(args.client_oauth_config)
                auth.set_default_account(account)
                _activate_account(account, session)
            else:
                _activate_account(account)
            logging.info("Colab API client ready for %s", account)
        except Exception as e:
            logging.warning(f"Failed to initialize Colab API client: {e}")

    try:
        await _run_mcp_transport(args)

    finally:
        await _execution_registry.close()
        for _client, manager in _accounts.values():
            await manager.close()
        if _direct_runtime_manager is not None:
            await _direct_runtime_manager.close()
        if args.enable_proxy and _session_mcp:
            await _session_mcp.cleanup()
        # Always unregister so a clean shutdown doesn't leave a stale entry.
        try:
            if _process_entry is None:
                process_registry.unregister()
            else:
                process_registry.unregister(
                    started_at=_process_entry.started_at,
                    instance_id=_process_entry.instance_id,
                )
        except Exception as exc:
            logging.warning(f"Could not unregister process: {exc}")


def main() -> None:
    asyncio.run(main_async())
