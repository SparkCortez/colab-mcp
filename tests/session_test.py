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

import asyncio
from colab_mcp import session
from fastmcp import Client
from fastmcp.server.middleware import MiddlewareContext
import pytest
from unittest.mock import patch, AsyncMock, Mock


@pytest.fixture(autouse=True)
def mock_webbrowser(monkeypatch):
    import webbrowser

    mock_open = Mock()
    monkeypatch.setattr(webbrowser, "open_new", mock_open)
    return mock_open


@pytest.fixture
def mock_wss():
    """Provides a mock ColabWebSocketServer instance."""
    return MockColabWebSocketServer()


class MockColabWebSocketServer:
    def __init__(self):
        self.connection_live = asyncio.Event()
        self.read_stream = AsyncMock()
        self.write_stream = AsyncMock()
        self.token = "test-token"
        self.port = 1234

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_proxy_client(mock_wss):
    client = Mock(spec=session.ColabProxyClient)
    client.wss = mock_wss
    client.is_connected.return_value = False
    return client


class TestDirectTools:
    """Tests for the direct tool registration on the mcp server."""

    @pytest.mark.asyncio
    async def test_mcp_has_expected_tools(self):
        from colab_mcp import mcp
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            assert tool_names == {
                "list_accounts",
                "set_keep_alive",
                "add_account",
                "switch_account",
                "remove_account",
                "open_colab_browser_connection",
                "prepare_colab_browser_connection",
                "add_code_cell",
                "add_text_cell",
                "get_cells",
                "run_code_cell",
                "update_cell",
                "delete_cell",
                "move_cell",
                "change_runtime",
                "stop_runtime",
                "list_runtimes",
                "upload_to_runtime",
                "download_from_runtime",
                "create_notebook",
                "upload_files_to_runtime",
                "download_files_from_runtime",
                "list_runtime_files",
                "delete_runtime_files",
                "get_colab_connection_info",
                "run_code_cell_blocking",
                "get_code_execution",
                "list_code_executions",
            }

    @pytest.mark.asyncio
    async def test_stub_returns_not_connected_when_no_proxy(self):
        from colab_mcp import mcp
        async with Client(mcp) as client:
            result = await client.call_tool(
                "add_code_cell", {"code": "print('hi')", "cellIndex": -1}
            )
            assert any(
                session.NOT_CONNECTED_MSG in c.text for c in result.content
            )

    @pytest.mark.asyncio
    async def test_cell_add_tools_require_explicit_index(self):
        from colab_mcp import mcp

        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}

        assert "cellIndex" in tools["add_code_cell"].inputSchema["required"]
        assert "cellIndex" in tools["add_text_cell"].inputSchema["required"]


class TestAwaitToolsReady:
    """Tests for await_tools_ready polling."""

    @pytest.mark.asyncio
    async def test_returns_tool_names(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        mock_wss.connection_live.set()
        client.proxy_mcp_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "add_code_cell"
        client.proxy_mcp_client.list_tools = AsyncMock(return_value=[mock_tool])

        result = await client.await_tools_ready()
        assert result == ["add_code_cell"]

    @pytest.mark.asyncio
    async def test_polls_until_available(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        mock_wss.connection_live.set()
        client.proxy_mcp_client = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "run_code_cell"
        client.proxy_mcp_client.list_tools = AsyncMock(
            side_effect=[[], [mock_tool]]
        )

        with patch("colab_mcp.session.TOOLS_READY_POLL_INTERVAL", 0.01):
            result = await client.await_tools_ready()
        assert result == ["run_code_cell"]

    @pytest.mark.asyncio
    async def test_not_connected(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        result = await client.await_tools_ready()
        assert result == []


class TestColabProxyMiddleware:
    @pytest.mark.asyncio
    async def test_connection_live(self, mock_proxy_client):
        """Tests connection state change from disconnected to connected."""
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = True
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_not_live(self, mock_proxy_client):
        """Tests connection state change from connected to disconnected."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = False
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        assert middleware.last_message_connected is False
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_connection_change(self, mock_proxy_client):
        """Tests no connection state change."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_call_tool_await_connection(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.side_effect = [False, True]
        mock_proxy_client.await_proxy_connection = AsyncMock()
        mock_proxy_client.await_tools_ready = AsyncMock(
            return_value=["add_code_cell", "run_code_cell"]
        )
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        mock_proxy_client.await_tools_ready.assert_called_once()
        context.fastmcp_context.report_progress.assert_called()
        context.fastmcp_context.send_tool_list_changed.assert_called_once()
        assert result.structured_content["result"] is True
        assert "add_code_cell" in result.structured_content["available_tools"]

    @pytest.mark.asyncio
    async def test_on_call_tool_connected_but_no_tools(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.side_effect = [False, True]
        mock_proxy_client.await_proxy_connection = AsyncMock()
        mock_proxy_client.await_tools_ready = AsyncMock(return_value=[])
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        assert result.structured_content["result"] is True
        assert result.structured_content["available_tools"] == []

    @pytest.mark.asyncio
    async def test_on_call_tool_timeout(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.return_value = False
        mock_proxy_client.await_proxy_connection = AsyncMock()
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        assert result.structured_content == {"result": False}


class TestInjectedTools:
    @pytest.mark.asyncio
    async def test_connected(self, mock_wss):
        mock_wss.connection_live.set()
        proxy_client = session.ColabProxyClient(mock_wss)
        proxy_client.proxy_mcp_client = Mock()
        tools = session._make_injected_tools(proxy_client)
        connection_tool = [t for t in tools if t.name == session.INJECTED_TOOL_NAME][0]
        result = await connection_tool.fn()
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnected(self, mock_wss, mock_webbrowser):
        proxy_client = session.ColabProxyClient(mock_wss)
        tools = session._make_injected_tools(proxy_client)
        connection_tool = [t for t in tools if t.name == session.INJECTED_TOOL_NAME][0]
        result = await connection_tool.fn()
        assert result is False
        mock_webbrowser.assert_called_once()
        args, _ = mock_webbrowser.call_args
        assert "mcpProxyToken=test-token" in args[0]
        assert "mcpProxyPort=1234" in args[0]
        # ?p=<port> in the query forces a unique URL per server instance so
        # Chrome opens a fresh tab instead of dedup'ing onto a stale one.
        assert "?p=1234" in args[0]

    def test_has_all_expected_tools(self, mock_wss):
        proxy_client = session.ColabProxyClient(mock_wss)
        tools = session._make_injected_tools(proxy_client)
        tool_names = {t.name for t in tools}
        assert tool_names == {
            session.INJECTED_TOOL_NAME,
            "add_code_cell",
            "add_text_cell",
            "get_cells",
            "run_code_cell",
            "update_cell",
            "delete_cell",
            "move_cell",
        }


class TestColabProxyClient:
    def test_is_connected(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.is_connected() is False
        mock_wss.connection_live.set()
        assert client.is_connected() is False
        client.proxy_mcp_client = Mock()
        assert client.is_connected() is True

    def test_client_factory_connection_live(self, mock_wss):
        mock_wss.connection_live.set()
        client = session.ColabProxyClient(mock_wss)
        client.proxy_mcp_client = Mock()

        assert client.client_factory() is client.proxy_mcp_client

    def test_client_factory_connection_not_live(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.client_factory() is client.stubbed_mcp_client

    @pytest.mark.asyncio
    async def test_await_proxy_connection(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        client._start_task = asyncio.create_task(asyncio.sleep(0.01))
        mock_wss.connection_live.set()
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.1):
            await client.await_proxy_connection()
        await client._start_task

    @pytest.mark.asyncio
    @patch("colab_mcp.session.Client")
    @patch("colab_mcp.session.ColabTransport", spec=session.ColabTransport)
    async def test_start_proxy_client(
        self, mock_colab_transport, mock_client, mock_wss
    ):
        mock_client.return_value.__aenter__ = AsyncMock()
        client = session.ColabProxyClient(mock_wss)
        mock_wss.connection_live.set()
        async with client:
            await client._start_task

        mock_colab_transport.assert_called_once_with(mock_wss)
        mock_client.assert_called_with(mock_colab_transport.return_value)


class TestColabTransport:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ClientSession")
    async def test_connect_session(self, mock_client_session, mock_wss):
        transport = session.ColabTransport(mock_wss)
        mock_client_session.return_value.__aenter__ = AsyncMock()
        async with transport.connect_session(foo="bar") as client_session:
            assert (
                client_session
                == mock_client_session.return_value.__aenter__.return_value
            )

        mock_client_session.assert_called_once_with(
            mock_wss.read_stream, mock_wss.write_stream, foo="bar"
        )


class TestColabSessionProxy:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ColabWebSocketServer")
    @patch("colab_mcp.session.ColabProxyClient")
    async def test_start_proxy_server(
        self,
        mock_colab_proxy_client,
        mock_colab_web_socket_server,
    ):
        mock_colab_web_socket_server.return_value.__aenter__ = AsyncMock()
        mock_colab_proxy_client.return_value.__aenter__ = AsyncMock()
        proxy = session.ColabSessionProxy()
        await proxy.start_proxy_server()
        mock_colab_proxy_client.assert_called_once()
        assert proxy.proxy_client is not None

    @pytest.mark.asyncio
    async def test_cleanup(self):
        proxy = session.ColabSessionProxy()
        proxy._exit_stack = AsyncMock()
        await proxy.cleanup()
        proxy._exit_stack.aclose.assert_called_once()


class TestStopRuntime:
    """Tests for the OAuth-backed stop_runtime tool."""

    @staticmethod
    def _assignment(endpoint, accelerator="T4"):
        return Mock(endpoint=endpoint, accelerator=Mock(value=accelerator))

    @pytest.mark.asyncio
    async def test_requires_oauth(self, monkeypatch):
        import colab_mcp
        monkeypatch.setattr(colab_mcp, "_colab_client", None)
        result = await colab_mcp.stop_runtime.fn()
        assert "not initialized" in result

    @pytest.mark.asyncio
    async def test_stops_all_and_closes_direct_runtime(self, monkeypatch):
        import colab_mcp
        client = Mock()
        client.list_assignments.return_value = [
            self._assignment("gpu-a"),
            self._assignment("cpu-b", "NONE"),
        ]
        manager = Mock(endpoint="gpu-a", close=AsyncMock())
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)
        result = await colab_mcp.stop_runtime.fn(all_runtimes=True)
        assert [c.args[0] for c in client.unassign.call_args_list] == ["gpu-a", "cpu-b"]
        manager.close.assert_awaited_once()
        assert "gpu-a (T4)" in result and "cpu-b (NONE)" in result

    @pytest.mark.asyncio
    async def test_stops_only_given_endpoint(self, monkeypatch):
        import colab_mcp
        client = Mock()
        client.list_assignments.return_value = [
            self._assignment("gpu-a"),
            self._assignment("cpu-b", "NONE"),
        ]
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", None)
        await colab_mcp.stop_runtime.fn(runtime_endpoint="cpu-b")
        client.unassign.assert_called_once_with("cpu-b")

    @pytest.mark.asyncio
    async def test_unknown_endpoint_and_nothing_active(self, monkeypatch):
        import colab_mcp
        client = Mock()
        client.list_assignments.return_value = []
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        assert "No active Colab runtime to stop" in await colab_mcp.stop_runtime.fn()
        assert "No active Colab runtime was found" in await colab_mcp.stop_runtime.fn("x")
        client.unassign.assert_not_called()


    @pytest.mark.asyncio
    async def test_default_stops_only_own_runtime(self, monkeypatch):
        import colab_mcp
        client = Mock()
        client.list_assignments.return_value = [
            self._assignment("other-session"),
            self._assignment("mine", "NONE"),
        ]
        manager = Mock(endpoint="mine", close=AsyncMock())
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)
        result = await colab_mcp.stop_runtime.fn()
        client.unassign.assert_called_once_with("mine")
        assert "other-session" not in result

    @pytest.mark.asyncio
    async def test_default_without_own_runtime_stops_nothing(self, monkeypatch):
        import colab_mcp
        client = Mock()
        client.list_assignments.return_value = [self._assignment("other-session")]
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", Mock(endpoint=None))
        result = await colab_mcp.stop_runtime.fn()
        client.unassign.assert_not_called()
        assert "nothing was stopped" in result and "other-session" in result


class TestChangeRuntimeScope:
    @pytest.mark.asyncio
    async def test_change_runtime_leaves_other_sessions_runtimes(self, monkeypatch):
        import colab_mcp
        from colab_mcp.client import Accelerator
        client = Mock()
        client.list_assignments.return_value = [
            Mock(endpoint="mine", accelerator=Accelerator.T4),
            Mock(endpoint="other", accelerator=Accelerator.T4),
        ]
        client.assign.return_value = Mock(endpoint="new", idle_timeout_sec=1800)
        manager = Mock(endpoint="mine", close=AsyncMock())
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)
        monkeypatch.setattr(colab_mcp, "_pending_assignments", {})
        result = await colab_mcp.change_runtime.fn("NONE")
        client.unassign.assert_called_once_with("mine")
        manager.note_released.assert_called_once()
        assert "Released previous runtime mine (T4)" in result

    @pytest.mark.asyncio
    async def test_change_runtime_reports_too_many_assignments(self, monkeypatch):
        import colab_mcp
        from colab_mcp.client import TooManyAssignmentsError
        client = Mock()
        client.assign.side_effect = TooManyAssignmentsError("412")
        client.list_assignments.return_value = [
            Mock(endpoint="other-gpu", accelerator=Mock(value="T4"))
        ]
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", Mock(endpoint=None))
        result = await colab_mcp.change_runtime.fn("T4")
        client.unassign.assert_not_called()
        assert "other-gpu (T4)" in result


class TestChangeRuntimeIdempotent:
    @pytest.fixture
    def env(self, monkeypatch):
        import colab_mcp
        from colab_mcp.client import Accelerator
        client = Mock()
        client.list_assignments.return_value = [
            Mock(endpoint="mine", accelerator=Accelerator.T4)
        ]
        manager = Mock(
            endpoint="mine",
            connected=True,
            close=AsyncMock(),
            connect=AsyncMock(),
            attach_target=AsyncMock(),
            keep_alive_enabled=True,
            keep_alive_interval=300,
            keep_alive_idle_limit=3600,
        )
        manager.record.return_value = Mock(idle_timeout_sec=1800)
        monkeypatch.setattr(colab_mcp, "_colab_client", client)
        monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", manager)
        monkeypatch.setattr(colab_mcp, "_pending_assignments", {})
        return colab_mcp, client, manager

    @pytest.mark.asyncio
    async def test_reuses_runtime_with_same_accelerator(self, env):
        colab_mcp, client, manager = env
        result = await colab_mcp.change_runtime.fn("t4")
        client.assign.assert_not_called()
        client.unassign.assert_not_called()
        assert "reused" in result and "1800 s" in result

    @pytest.mark.asyncio
    async def test_force_new_assigns_before_releasing(self, env):
        colab_mcp, client, manager = env
        calls = []
        client.assign.side_effect = lambda *a: calls.append("assign") or Mock(
            endpoint="new", idle_timeout_sec=1800
        )
        client.unassign.side_effect = lambda e: calls.append(f"unassign {e}")
        await colab_mcp.change_runtime.fn("T4", force_new=True)
        assert calls == ["assign", "unassign mine"]

    @pytest.mark.asyncio
    async def test_timeout_keeps_runtime_and_retry_reuses_request(self, env):
        import requests
        colab_mcp, client, manager = env
        client.assign.side_effect = requests.exceptions.ReadTimeout("read timed out")
        result = await colab_mcp.change_runtime.fn("L4")
        client.unassign.assert_not_called()
        assert "mine (T4) was not touched" in result and "again" in result
        first_hash = client.assign.call_args.args[0]
        client.assign.side_effect = None
        client.assign.return_value = {"assignment": Mock(endpoint="new", idle_timeout_sec=None), "is_new": False}
        await colab_mcp.change_runtime.fn("L4")
        assert client.assign.call_args.args[0] == first_hash
        assert colab_mcp._pending_assignments == {}

    @pytest.mark.asyncio
    async def test_too_many_releases_own_then_retries(self, env):
        from colab_mcp.client import TooManyAssignmentsError
        colab_mcp, client, manager = env
        client.assign.side_effect = [
            TooManyAssignmentsError("412"),
            Mock(endpoint="new", idle_timeout_sec=600),
        ]
        result = await colab_mcp.change_runtime.fn("L4")
        client.unassign.assert_called_once_with("mine")
        assert "Released previous runtime mine" in result and "new" in result

    @pytest.mark.asyncio
    async def test_bad_gateway_is_reported_as_temporary(self, env):
        from colab_mcp.client import ColabRequestError
        colab_mcp, client, manager = env
        response = Mock(status_code=502, reason="Bad Gateway")
        client.assign.side_effect = ColabRequestError("502", None, response)
        result = await colab_mcp.change_runtime.fn("L4")
        client.unassign.assert_not_called()
        assert "HTTP 502" in result and "mine (T4) was not touched" in result
        assert colab_mcp._pending_assignments

    @pytest.mark.asyncio
    async def test_client_errors_are_not_retried_silently(self, env):
        from colab_mcp.client import ColabRequestError
        colab_mcp, client, manager = env
        response = Mock(status_code=403, reason="Forbidden")
        client.assign.side_effect = ColabRequestError("403 Forbidden", None, response)
        result = await colab_mcp.change_runtime.fn("L4")
        assert result.startswith("Failed to change runtime") and "403" in result

    @pytest.mark.asyncio
    async def test_switch_account_mid_call_stays_on_original_account(self, env, monkeypatch):
        from colab_mcp.client import Accelerator
        colab_mcp, client, manager = env
        other_client = Mock()
        new = Mock(endpoint="new", accelerator=Accelerator.L4)
        client.list_assignments.side_effect = [
            [Mock(endpoint="mine", accelerator=Accelerator.T4)],
            [new],
        ]

        def assign(*args):
            # Another tool call switches the session while Colab is answering.
            colab_mcp._colab_client = other_client
            colab_mcp._direct_runtime_manager = Mock()
            return Mock(endpoint="new", idle_timeout_sec=1800)

        client.assign.side_effect = assign
        await colab_mcp.change_runtime.fn("L4")
        client.unassign.assert_called_once_with("mine")
        manager.attach_target.assert_awaited_once()
        other_client.unassign.assert_not_called()
        other_client.list_assignments.assert_not_called()
