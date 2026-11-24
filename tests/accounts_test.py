# Added to this local copy of the fork, 2026.
# Licensed under the Apache License, Version 2.0.

import json
from unittest.mock import AsyncMock, Mock

import pytest

import colab_mcp
from colab_mcp import auth


@pytest.fixture
def accounts_dir(tmp_path, monkeypatch):
    folder = tmp_path / "accounts"
    monkeypatch.setattr(auth, "ACCOUNTS_DIR", str(folder))
    monkeypatch.setattr(auth, "DEFAULT_ACCOUNT_PATH", str(tmp_path / "default_account"))
    monkeypatch.setattr(auth, "TOKEN_CONFIG_PATH", str(tmp_path / "legacy.json"))
    monkeypatch.setattr(auth, "DRIVE_TOKEN_CONFIG_PATH", str(tmp_path / "legacy-drive.json"))
    return folder


def _save(email):
    auth._write_token_atomically(auth.account_token_path(email), "{}")


@pytest.fixture
def server(monkeypatch, accounts_dir):
    """A server state with fake per-account clients."""
    clients = {}

    def fake_pair(email, session=None):
        if email not in colab_mcp._accounts:
            client = Mock(name=email)
            client.list_assignments.return_value = []
            client.get_ccu_info.return_value = Mock(
                current_balance=10.0, consumption_rate_hourly=0.0
            )
            clients[email] = client
            colab_mcp._accounts[email] = (client, Mock(connected=False, close=AsyncMock()))
        return colab_mcp._accounts[email]

    monkeypatch.setattr(colab_mcp, "_accounts", {})
    monkeypatch.setattr(colab_mcp, "_account_pair", fake_pair)
    monkeypatch.setattr(colab_mcp, "_client_oauth_config", "client.json")
    monkeypatch.setattr(colab_mcp, "_colab_client", None)
    monkeypatch.setattr(colab_mcp, "_direct_runtime_manager", None)
    monkeypatch.setattr(colab_mcp, "_active_account", None)
    return clients


class TestAuthHelpers:
    def test_list_ignores_drive_tokens_and_default_falls_back(self, accounts_dir):
        _save("b@x.com")
        _save("a@x.com")
        auth._write_token_atomically(auth.account_drive_token_path("a@x.com"), "{}")
        assert auth.list_saved_accounts() == ["a@x.com", "b@x.com"]
        assert auth.get_default_account() == "a@x.com"
        auth.set_default_account("b@x.com")
        assert auth.get_default_account() == "b@x.com"

    def test_remove_deletes_tokens_and_default(self, accounts_dir):
        _save("a@x.com")
        auth._write_token_atomically(auth.account_drive_token_path("a@x.com"), "{}")
        auth.set_default_account("a@x.com")
        auth.remove_account("a@x.com")
        assert auth.list_saved_accounts() == []
        assert auth.get_default_account() is None
        assert not accounts_dir.joinpath("a@x.com.drive.json").exists()

    def test_migrate_legacy_token(self, accounts_dir, tmp_path, monkeypatch):
        (tmp_path / "legacy.json").write_text('{"token": "t"}')
        (tmp_path / "legacy-drive.json").write_text('{"token": "d"}')
        session = Mock()
        session.get.return_value = Mock(json=Mock(return_value={"email": "Me@X.com"}))
        monkeypatch.setattr(auth, "get_credentials", Mock(return_value=session))
        assert auth.migrate_legacy_token("client.json") == "me@x.com"
        assert auth.account_token_path("me@x.com").read_text() == '{"token": "t"}'
        assert auth.account_drive_token_path("me@x.com").read_text() == '{"token": "d"}'
        assert auth.get_default_account() == "me@x.com"
        # Second run is a no-op.
        assert auth.migrate_legacy_token("client.json") is None


class TestAccountTools:
    @pytest.mark.asyncio
    async def test_switch_changes_active_client(self, server):
        _save("a@x.com")
        _save("b@x.com")
        colab_mcp._activate_account("a@x.com")
        result = await colab_mcp.switch_account.fn("B@x.com")
        assert "b@x.com" in result
        assert colab_mcp._active_account == "b@x.com"
        assert colab_mcp._colab_client is server["b@x.com"]
        assert colab_mcp._direct_runtime_manager is colab_mcp._accounts["b@x.com"][1]

    @pytest.mark.asyncio
    async def test_switch_unknown_account(self, server):
        _save("a@x.com")
        result = await colab_mcp.switch_account.fn("z@x.com")
        assert "not signed in" in result and "a@x.com" in result

    @pytest.mark.asyncio
    async def test_switch_make_default(self, server):
        _save("a@x.com")
        _save("b@x.com")
        await colab_mcp.switch_account.fn("b@x.com", make_default=True)
        assert auth.get_default_account() == "b@x.com"

    @pytest.mark.asyncio
    async def test_list_accounts(self, server):
        _save("a@x.com")
        _save("b@x.com")
        colab_mcp._activate_account("b@x.com")
        rows = json.loads(await colab_mcp.list_accounts.fn())["accounts"]
        assert [(r["email"], r["active"], r["default"]) for r in rows] == [
            ("a@x.com", False, True),
            ("b@x.com", True, False),
        ]
        assert rows[0]["compute_units"]["balance"] == 10.0

    @pytest.mark.asyncio
    async def test_add_account_activates(self, server, monkeypatch):
        monkeypatch.setattr(
            auth, "authorize_new_account", Mock(return_value=("new@x.com", object(), True))
        )
        result = await colab_mcp.add_account.fn()
        assert "new@x.com" in result
        assert colab_mcp._active_account == "new@x.com"

    @pytest.mark.asyncio
    async def test_add_account_failure(self, server, monkeypatch):
        monkeypatch.setattr(
            auth, "authorize_new_account", Mock(side_effect=TimeoutError("timed out"))
        )
        result = await colab_mcp.add_account.fn()
        assert "not completed" in result and colab_mcp._active_account is None

    @pytest.mark.asyncio
    async def test_remove_refuses_active(self, server):
        _save("a@x.com")
        _save("b@x.com")
        colab_mcp._activate_account("a@x.com")
        assert "switch_account" in await colab_mcp.remove_account.fn("a@x.com")
        colab_mcp._accounts["b@x.com"] = (Mock(), Mock(close=AsyncMock()))
        manager = colab_mcp._accounts["b@x.com"][1]
        assert "Removed" in await colab_mcp.remove_account.fn("b@x.com")
        manager.close.assert_awaited_once()
        assert auth.list_saved_accounts() == ["a@x.com"]


class TestCombinedConsent:
    def _run(self, accounts_dir, monkeypatch, granted):
        creds = Mock(token="t", granted_scopes=granted, scopes=None)
        creds.to_json.return_value = '{"token": "t"}'
        flow = Mock()
        flow.run_local_server.return_value = creds
        from_file = Mock(return_value=flow)
        monkeypatch.setattr(auth.InstalledAppFlow, "from_client_secrets_file", from_file)
        monkeypatch.setattr(auth, "fetch_email", Mock(return_value="new@x.com"))
        monkeypatch.setattr(auth.requests, "AuthorizedSession", Mock())
        result = auth.authorize_new_account("client.json")
        assert from_file.call_args.args[1] == auth.SCOPES + auth.DRIVE_SCOPES
        return result

    def test_one_consent_saves_colab_and_drive(self, accounts_dir, monkeypatch):
        email, _session, drive = self._run(
            accounts_dir, monkeypatch, auth.SCOPES + auth.DRIVE_SCOPES
        )
        assert (email, drive) == ("new@x.com", True)
        assert auth.account_token_path(email).exists()
        assert auth.account_drive_token_path(email).exists()

    def test_unticked_drive_keeps_colab_only(self, accounts_dir, monkeypatch):
        email, _session, drive = self._run(accounts_dir, monkeypatch, list(auth.SCOPES))
        assert drive is False
        assert auth.account_token_path(email).exists()
        assert not auth.account_drive_token_path(email).exists()

    def test_unticked_colab_is_an_error(self, accounts_dir, monkeypatch):
        with pytest.raises(PermissionError, match="colaboratory"):
            self._run(accounts_dir, monkeypatch, auth.DRIVE_SCOPES + ["openid"])
