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

"""Helpers for moving files through a direct kernel and building notebooks."""

import base64
import glob
import json
import os
import tarfile
from typing import Any

# Raw bytes per kernel round trip. Base64 grows this by a third, which keeps
# each execute request and stdout message comfortably small.
CHUNK_BYTES = 4 * 1024 * 1024
MAX_TRANSFER_BYTES = 1024 * 1024 * 1024
COLAB_MIME_TYPE = "application/vnd.google.colaboratory"


class TransferError(Exception):
    """A file transfer through the kernel failed."""


def stdout_text(outputs: list[dict[str, Any]]) -> str:
    """Concatenate stdout from kernel outputs, raising on a kernel error."""
    parts = []
    for item in outputs:
        if item.get("output_type") == "error":
            raise TransferError(f"{item.get('ename')}: {item.get('evalue')}")
        if item.get("output_type") == "stream" and item.get("name") == "stdout":
            text = item.get("text", "")
            parts.append("".join(text) if isinstance(text, list) else text)
    return "".join(parts)


def upload_chunk_code(remote_path: str, chunk: bytes, first: bool) -> str:
    """Kernel code that writes (first chunk) or appends one chunk."""
    encoded = base64.b64encode(chunk).decode("ascii")
    mode = "wb" if first else "ab"
    return (
        "import base64, os\n"
        f"_p = {remote_path!r}\n"
        "os.makedirs(os.path.dirname(_p) or '.', exist_ok=True)\n"
        f"with open(_p, {mode!r}) as _f:\n"
        f"    _f.write(base64.b64decode({encoded!r}))\n"
        "print(os.path.getsize(_p))\n"
    )


def stat_code(remote_path: str) -> str:
    """Kernel code that prints the size of a regular file as JSON."""
    return (
        "import json, os\n"
        f"_p = {remote_path!r}\n"
        "print(json.dumps({'size': os.path.getsize(_p) if os.path.isfile(_p) else None}))\n"
    )


def download_chunk_code(remote_path: str, offset: int, length: int) -> str:
    """Kernel code that prints one base64-encoded chunk of a file."""
    return (
        "import base64\n"
        f"with open({remote_path!r}, 'rb') as _f:\n"
        f"    _f.seek({offset})\n"
        f"    print(base64.b64encode(_f.read({length})).decode('ascii'))\n"
    )


def build_notebook(cells: list[dict[str, Any]] | None, accelerator: str = "") -> dict:
    """Build an nbformat 4 notebook from ``[{type, source}]`` cell specs."""
    nb_cells = []
    for spec in cells or []:
        kind = spec.get("type", "code")
        if kind not in ("code", "markdown"):
            raise ValueError(f"Unsupported cell type '{kind}'; use 'code' or 'markdown'.")
        cell: dict[str, Any] = {
            "cell_type": kind,
            "metadata": {},
            "source": spec.get("source", "").splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    metadata: dict[str, Any] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    if accelerator and accelerator.upper() != "NONE":
        metadata["accelerator"] = "GPU"
        metadata["colab"]["gpuType"] = accelerator.upper()
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": metadata, "cells": nb_cells}


def multipart_body(metadata: dict, content: bytes, boundary: str) -> bytes:
    """Encode a Drive multipart/related upload body."""
    return b"".join(
        [
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(),
            json.dumps(metadata).encode(),
            f"\r\n--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )


def _noop(_event):
    pass


async def push_file(runtime, source: str, remote_path: str) -> int:
    """Copy a local file to the runtime in chunks; return the remote size."""
    size = os.path.getsize(source)
    if size > MAX_TRANSFER_BYTES:
        raise TransferError(f"{source} is {size} bytes; the limit is {MAX_TRANSFER_BYTES} bytes.")
    written = 0
    first = True
    with open(source, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk and not first:
                break
            outputs = await runtime.execute(upload_chunk_code(remote_path, chunk, first), _noop)
            written = int(stdout_text(outputs).strip())
            first = False
    if written != size:
        raise TransferError(f"Upload incomplete: remote has {written} of {size} bytes.")
    return written


async def pull_file(runtime, remote_path: str, target: str) -> int | None:
    """Copy a runtime file to ``target`` in chunks; None if it does not exist."""
    outputs = await runtime.execute(stat_code(remote_path), _noop)
    size = json.loads(stdout_text(outputs).strip())["size"]
    if size is None:
        return None
    if size > MAX_TRANSFER_BYTES:
        raise TransferError(f"{remote_path} is {size} bytes; the limit is {MAX_TRANSFER_BYTES} bytes.")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    partial = target + ".part"
    try:
        with open(partial, "wb") as handle:
            for offset in range(0, size, CHUNK_BYTES):
                outputs = await runtime.execute(
                    download_chunk_code(remote_path, offset, CHUNK_BYTES), _noop
                )
                handle.write(base64.b64decode(stdout_text(outputs).strip()))
        received = os.path.getsize(partial)
        if received != size:
            raise TransferError(f"Download incomplete: got {received} of {size} bytes.")
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return size


def expand_local(paths: list[str]) -> list[str]:
    """Expand ~ and glob patterns; raise if any entry matches nothing."""
    expanded = []
    for raw in paths:
        pattern = os.path.expanduser(raw)
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        matches = [m for m in matches if os.path.exists(m)]
        if not matches:
            raise TransferError(f"Local path not found: {raw}")
        expanded.extend(matches)
    return expanded


def pack_local(paths: list[str], archive: str) -> list[str]:
    """Write files/directories into a tar.gz, each under its base name."""
    names = []
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            name = os.path.basename(os.path.normpath(path))
            tar.add(path, arcname=name)
            names.append(name)
    return names


def unpack_local(archive: str, dest: str) -> list[str]:
    """Extract a tar.gz safely into ``dest`` and return the top-level names."""
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
        return sorted({m.name.split("/")[0] for m in tar.getmembers()})


def remote_unpack_code(archive: str, dest: str) -> str:
    """Kernel code that extracts an uploaded archive and deletes it."""
    return (
        "import json, os, tarfile\n"
        f"os.makedirs({dest!r}, exist_ok=True)\n"
        f"with tarfile.open({archive!r}, 'r:gz') as _t:\n"
        f"    _t.extractall({dest!r}, filter='data')\n"
        "    _n = sorted({m.name.split('/')[0] for m in _t.getmembers()})\n"
        f"os.remove({archive!r})\n"
        "print(json.dumps(_n))\n"
    )


def remote_pack_code(patterns: list[str], archive: str) -> str:
    """Kernel code that globs paths, tars them, and reports what matched."""
    return (
        "import glob, json, os, tarfile\n"
        "_found, _missing = [], []\n"
        f"for _pat in {patterns!r}:\n"
        "    _m = sorted(glob.glob(os.path.expanduser(_pat)))\n"
        "    (_found.extend(_m) if _m else _missing.append(_pat))\n"
        "if _found:\n"
        f"    with tarfile.open({archive!r}, 'w:gz') as _t:\n"
        "        for _p in _found:\n"
        "            _t.add(_p, arcname=os.path.basename(os.path.normpath(_p)))\n"
        "print(json.dumps({'found': _found, 'missing': _missing}))\n"
    )


def remove_remote_code(path: str) -> str:
    return f"import os\nos.path.exists({path!r}) and os.remove({path!r})\n"


def list_remote_code(path: str, recursive: bool, limit: int) -> str:
    """Kernel code that lists files with sizes and modification times as JSON."""
    return (
        "import json, os, time\n"
        f"_root, _rec, _limit = os.path.expanduser({path!r}), {recursive!r}, {limit!r}\n"
        "_out, _more = [], False\n"
        "def _add(_p):\n"
        "    global _more\n"
        "    if len(_out) >= _limit:\n"
        "        _more = True\n"
        "        return\n"
        "    _s = os.stat(_p)\n"
        "    _out.append({'path': _p, 'type': 'dir' if os.path.isdir(_p) else 'file',\n"
        "                 'size': _s.st_size, 'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(_s.st_mtime))})\n"
        "if not os.path.exists(_root):\n"
        "    print(json.dumps({'error': 'not found'}))\n"
        "elif os.path.isfile(_root):\n"
        "    _add(_root); print(json.dumps({'entries': _out, 'truncated': False}))\n"
        "else:\n"
        "    if _rec:\n"
        "        for _d, _dirs, _files in os.walk(_root):\n"
        "            _dirs.sort()\n"
        "            for _n in sorted(_dirs) + sorted(_files):\n"
        "                _add(os.path.join(_d, _n))\n"
        "    else:\n"
        "        for _n in sorted(os.listdir(_root)):\n"
        "            _add(os.path.join(_root, _n))\n"
        "    print(json.dumps({'entries': _out, 'truncated': _more}))\n"
    )


def delete_remote_code(patterns: list[str]) -> str:
    """Kernel code that deletes files/directories matching glob patterns."""
    return (
        "import glob, json, os, shutil\n"
        "_deleted, _missing = [], []\n"
        f"for _pat in {patterns!r}:\n"
        "    _m = sorted(glob.glob(os.path.expanduser(_pat)))\n"
        "    if not _m:\n"
        "        _missing.append(_pat)\n"
        "    for _p in _m:\n"
        "        shutil.rmtree(_p) if os.path.isdir(_p) and not os.path.islink(_p) else os.remove(_p)\n"
        "        _deleted.append(_p)\n"
        "print(json.dumps({'deleted': _deleted, 'missing': _missing}))\n"
    )
