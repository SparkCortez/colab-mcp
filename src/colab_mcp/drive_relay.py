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

"""Move large files between this machine and a runtime through Google Drive.

The kernel route sends base64 through one websocket to a runtime that may be
on another continent, which tops out well under the local line speed.  Here
the local side talks to Drive (a nearby Google edge) at line speed, and the
runtime talks to Drive over Google's network.  Each relay file is created with
the ``drive.file`` scope and deleted as soon as the transfer finishes.
"""

import json
import os
import uuid
from typing import Any

from colab_mcp.transfer import TransferError, stdout_text

UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
# Files at least this large go through Drive when a Drive token is available.
DRIVE_THRESHOLD_BYTES = 20 * 1024 * 1024
MAX_DRIVE_TRANSFER_BYTES = 20 * 1024 * 1024 * 1024
STREAM_CHUNK = 8 * 1024 * 1024
HTTP_TIMEOUT = (30, 300)


def _noop(_event):
    pass


def access_token(session) -> str:
    """A currently valid OAuth access token for the runtime to use."""
    from google.auth.transport.requests import Request

    creds = session.credentials
    if not creds.valid:
        creds.refresh(Request())
    return creds.token


def relay_name(name: str) -> str:
    return f"colab-mcp-relay-{uuid.uuid4().hex[:12]}-{os.path.basename(name)}"


def upload_local(session, source: str, name: str) -> str:
    """Upload a local file to Drive with a resumable session; return its id."""
    size = os.path.getsize(source)
    start = session.post(
        UPLOAD_URL,
        params={"uploadType": "resumable", "fields": "id"},
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(size),
        },
        data=json.dumps({"name": name}),
        timeout=HTTP_TIMEOUT,
    )
    if start.status_code >= 400:
        raise TransferError(f"Drive upload start failed: HTTP {start.status_code} {start.text[:300]}")
    with open(source, "rb") as handle:
        done = session.put(
            start.headers["Location"],
            data=handle,
            headers={"Content-Length": str(size)},
            timeout=HTTP_TIMEOUT,
        )
    if done.status_code >= 400:
        raise TransferError(f"Drive upload failed: HTTP {done.status_code} {done.text[:300]}")
    return done.json()["id"]


def download_local(session, file_id: str, target: str) -> int:
    """Stream a Drive file to ``target`` (written via a .part file)."""
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    partial = target + ".part"
    try:
        with session.get(
            f"{FILES_URL}/{file_id}",
            params={"alt": "media"},
            stream=True,
            timeout=HTTP_TIMEOUT,
        ) as response:
            if response.status_code >= 400:
                raise TransferError(
                    f"Drive download failed: HTTP {response.status_code} {response.text[:300]}"
                )
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(STREAM_CHUNK):
                    handle.write(chunk)
        os.replace(partial, target)
        return os.path.getsize(target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def delete(session, file_id: str) -> None:
    try:
        session.delete(f"{FILES_URL}/{file_id}", timeout=HTTP_TIMEOUT)
    except Exception:
        pass


def runtime_fetch_code(file_id: str, token: str, remote_path: str) -> str:
    """Kernel code that downloads a Drive file to ``remote_path``."""
    return (
        "import os, requests\n"
        f"_p = {remote_path!r}\n"
        "os.makedirs(os.path.dirname(_p) or '.', exist_ok=True)\n"
        f"with requests.get({FILES_URL + '/' + file_id!r}, params={{'alt': 'media'}},\n"
        f"                  headers={{'Authorization': 'Bearer ' + {token!r}}}, stream=True, timeout=(30, 300)) as _r:\n"
        "    _r.raise_for_status()\n"
        "    with open(_p + '.part', 'wb') as _f:\n"
        f"        for _c in _r.iter_content({STREAM_CHUNK}):\n"
        "            _f.write(_c)\n"
        "os.replace(_p + '.part', _p)\n"
        "print(os.path.getsize(_p))\n"
    )


def runtime_push_code(remote_path: str, token: str, name: str) -> str:
    """Kernel code that uploads a runtime file to Drive and prints its id."""
    return (
        "import json, os, requests\n"
        f"_p, _h = {remote_path!r}, {{'Authorization': 'Bearer ' + {token!r}}}\n"
        "_n = os.path.getsize(_p)\n"
        f"_s = requests.post({UPLOAD_URL!r}, params={{'uploadType': 'resumable', 'fields': 'id'}},\n"
        "                   headers={**_h, 'Content-Type': 'application/json; charset=UTF-8',\n"
        "                            'X-Upload-Content-Length': str(_n)},\n"
        f"                   data=json.dumps({{'name': {name!r}}}), timeout=(30, 300))\n"
        "_s.raise_for_status()\n"
        "with open(_p, 'rb') as _f:\n"
        "    _d = requests.put(_s.headers['Location'], data=_f,\n"
        "                      headers={**_h, 'Content-Length': str(_n)}, timeout=(30, 300))\n"
        "_d.raise_for_status()\n"
        "print(json.dumps({'id': _d.json()['id'], 'size': _n}))\n"
    )


async def push_file(runtime: Any, session: Any, source: str, remote_path: str) -> int:
    """Local file -> Drive -> runtime; returns the size written on the runtime."""
    import asyncio

    size = os.path.getsize(source)
    if size > MAX_DRIVE_TRANSFER_BYTES:
        raise TransferError(f"{source} is {size} bytes; the Drive limit here is {MAX_DRIVE_TRANSFER_BYTES}.")
    file_id = await asyncio.to_thread(upload_local, session, source, relay_name(source))
    try:
        token = await asyncio.to_thread(access_token, session)
        outputs = await runtime.execute(runtime_fetch_code(file_id, token, remote_path), _noop)
        written = int(stdout_text(outputs).strip())
    finally:
        await asyncio.to_thread(delete, session, file_id)
    if written != size:
        raise TransferError(f"Upload incomplete: remote has {written} of {size} bytes.")
    return written


async def pull_file(runtime: Any, session: Any, remote_path: str, target: str) -> int:
    """Runtime file -> Drive -> local ``target``; returns the size."""
    import asyncio

    token = await asyncio.to_thread(access_token, session)
    outputs = await runtime.execute(
        runtime_push_code(remote_path, token, relay_name(remote_path)), _noop
    )
    report = json.loads(stdout_text(outputs).strip())
    try:
        received = await asyncio.to_thread(download_local, session, report["id"], target)
    finally:
        await asyncio.to_thread(delete, session, report["id"])
    if received != report["size"]:
        raise TransferError(f"Download incomplete: got {received} of {report['size']} bytes.")
    return received
