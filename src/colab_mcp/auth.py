# Copyright 2026 Google Inc.
# Added to this fork by Sebastian Gil Pinzon, 2026.
# Modified by Atsushi Onozawa, 2026.
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


import os
from pathlib import Path
import shutil
import tempfile
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.transport import requests

# 0 = any free loopback port (Desktop OAuth clients accept any), so two
# sign-ins at once (or a slow-to-release port) never collide.
OAUTH_SERVER_PORT = 0

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
    "openid",
]

TOKEN_CONFIG_PATH = os.path.expanduser("~/.colab-mcp-auth-token.json")


def _ensure_private_permissions(token_path: Path) -> None:
    """Restrict an existing token file to its owner on POSIX systems."""
    if os.name != "nt" and token_path.exists():
        token_path.chmod(0o600)


def _write_token_atomically(token_path: Path, contents: str) -> None:
    """Atomically replace the OAuth token cache with owner-only permissions."""
    token_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{token_path.name}.", dir=token_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            descriptor = -1
            token_file.write(contents)
            token_file.flush()
            os.fsync(token_file.fileno())
        os.replace(temporary_path, token_path)
        _ensure_private_permissions(token_path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def _load_or_authorize(config, token_path: Path, scopes, login_hint=None):
    creds = None
    if token_path.exists():
        _ensure_private_permissions(token_path)
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config, scopes)
            extra = {"login_hint": login_hint} if login_hint else {}
            creds = flow.run_local_server(
                port=OAUTH_SERVER_PORT, timeout_seconds=300, **extra
            )

        _write_token_atomically(token_path, creds.to_json())

    return requests.AuthorizedSession(creds)


def get_credentials(config, token_path=None, login_hint=None):
    path = Path(token_path) if token_path else Path(TOKEN_CONFIG_PATH)
    return _load_or_authorize(config, path, SCOPES, login_hint)


# Drive access is requested lazily (on the first create_notebook call) and
# cached separately, so adding it never forces a new consent at startup.
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_TOKEN_CONFIG_PATH = os.path.expanduser("~/.colab-mcp-drive-token.json")


def get_drive_session(config, token_path=None, login_hint=None):
    path = Path(token_path) if token_path else Path(DRIVE_TOKEN_CONFIG_PATH)
    return _load_or_authorize(config, path, DRIVE_SCOPES, login_hint)


# Multiple Google accounts: one Colab token (and an optional Drive token) per
# account, named after the account's email address.
ACCOUNTS_DIR = os.path.expanduser("~/.config/colab-mcp/accounts")
DEFAULT_ACCOUNT_PATH = os.path.expanduser("~/.config/colab-mcp/default_account")
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def account_token_path(email: str) -> Path:
    return Path(ACCOUNTS_DIR) / f"{email}.json"


def account_drive_token_path(email: str) -> Path:
    return Path(ACCOUNTS_DIR) / f"{email}.drive.json"


def list_saved_accounts() -> list[str]:
    folder = Path(ACCOUNTS_DIR)
    if not folder.is_dir():
        return []
    return sorted(
        p.name[: -len(".json")]
        for p in folder.glob("*.json")
        if not p.name.endswith(".drive.json")
    )


def get_default_account() -> str | None:
    path = Path(DEFAULT_ACCOUNT_PATH)
    if path.exists():
        email = path.read_text(encoding="utf-8").strip()
        if email in list_saved_accounts():
            return email
    accounts = list_saved_accounts()
    return accounts[0] if accounts else None


def set_default_account(email: str) -> None:
    path = Path(DEFAULT_ACCOUNT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(email + "\n", encoding="utf-8")


def fetch_email(session) -> str:
    response = session.get(USERINFO_URL, timeout=30)
    response.raise_for_status()
    email = response.json().get("email")
    if not email:
        raise ValueError("Google did not return an email address for this account.")
    return email.lower()


def authorize_new_account(config, timeout_seconds=None) -> tuple[str, object, bool]:
    """Run one consent flow for Colab and Drive, with an account chooser.

    Colab and Drive (``drive.file``, used by create_notebook and fast file
    transfers) are requested on the same screen.  Google lets the user untick
    Drive, so the Drive token is saved only when it was actually granted.
    Returns ``(email, session, drive_granted)``; existing tokens for the same
    account are replaced.
    """
    # Granular consent may return fewer scopes than requested; accept that
    # instead of oauthlib's "Scope has changed" error.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    flow = InstalledAppFlow.from_client_secrets_file(config, SCOPES + DRIVE_SCOPES)
    creds = flow.run_local_server(
        port=OAUTH_SERVER_PORT,
        prompt="select_account consent",
        timeout_seconds=timeout_seconds,
    )
    if not creds or not creds.token:
        raise TimeoutError("Google sign-in was not completed.")
    granted = set(getattr(creds, "granted_scopes", None) or creds.scopes or [])
    missing = [scope for scope in SCOPES if scope.startswith("https://") and scope not in granted]
    if missing:
        raise PermissionError(
            "Colab access was not granted (unticked on the consent screen): " + ", ".join(missing)
        )
    session = requests.AuthorizedSession(creds)
    email = fetch_email(session)
    _write_token_atomically(account_token_path(email), creds.to_json())
    drive_granted = DRIVE_SCOPES[0] in granted
    if drive_granted:
        # The same refresh token serves both scopes; keep the Drive copy where
        # get_drive_session looks for it.
        _write_token_atomically(account_drive_token_path(email), creds.to_json())
    return email, session, drive_granted


def remove_account(email: str) -> None:
    account_token_path(email).unlink(missing_ok=True)
    account_drive_token_path(email).unlink(missing_ok=True)
    default = Path(DEFAULT_ACCOUNT_PATH)
    if default.exists() and default.read_text(encoding="utf-8").strip() == email:
        default.unlink()


def migrate_legacy_token(config) -> str | None:
    """Copy the single-account token into the accounts folder once.

    The legacy files are left in place so older builds keep working.
    """
    legacy = Path(TOKEN_CONFIG_PATH)
    if list_saved_accounts() or not legacy.exists():
        return None
    session = get_credentials(config)
    email = fetch_email(session)
    _write_token_atomically(
        account_token_path(email), legacy.read_text(encoding="utf-8")
    )
    legacy_drive = Path(DRIVE_TOKEN_CONFIG_PATH)
    if legacy_drive.exists():
        shutil.copyfile(legacy_drive, account_drive_token_path(email))
        _ensure_private_permissions(account_drive_token_path(email))
    set_default_account(email)
    return email
