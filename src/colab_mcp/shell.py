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

"""Kernel code for running shell commands on a runtime.

The code runs in the side kernel (see ``DirectRuntimeManager.side``), so a
command works while a long cell keeps the main kernel busy.  Foreground
commands run in their own process group and are killed as a group on timeout;
background jobs are detached into a new session, log to a file and record
their exit code, so they outlive the kernel that started them.
"""

JOBS_DIR = "/tmp/colab-mcp-jobs"
# Characters of stdout/stderr returned for a foreground command (the tail).
MAX_OUTPUT_CHARS = 20000


def run_code(command: str, cwd: str, timeout: float, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Kernel code that runs ``command`` with bash and prints a JSON result."""
    return (
        "import json, os, signal, subprocess, time\n"
        f"_cmd, _cwd, _limit, _max = {command!r}, {cwd!r}, {float(timeout)!r}, {int(max_chars)!r}\n"
        "_t0 = time.time()\n"
        "_res = {'timed_out': False}\n"
        "try:\n"
        "    _p = subprocess.Popen(['bash', '-c', _cmd], cwd=_cwd or None, stdin=subprocess.DEVNULL,\n"
        "                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)\n"
        "    try:\n"
        "        _out, _err = _p.communicate(timeout=_limit)\n"
        "    except subprocess.TimeoutExpired:\n"
        "        _res['timed_out'] = True\n"
        "        try:\n"
        "            os.killpg(_p.pid, signal.SIGKILL)\n"
        "        except ProcessLookupError:\n"
        "            pass\n"
        "        _out, _err = _p.communicate()\n"
        "    _res['exit_code'] = _p.returncode\n"
        "except Exception as _e:\n"
        "    _out, _err = b'', repr(_e).encode()\n"
        "    _res['exit_code'] = None\n"
        "for _k, _v in (('stdout', _out), ('stderr', _err)):\n"
        "    _s = _v.decode('utf-8', 'replace')\n"
        "    _res[_k] = _s[-_max:]\n"
        "    if len(_s) > _max:\n"
        "        _res[_k + '_truncated_chars'] = len(_s) - _max\n"
        "_res['seconds'] = round(time.time() - _t0, 2)\n"
        "print(json.dumps(_res))\n"
    )


def start_job_code(job_id: str, command: str, cwd: str) -> str:
    """Kernel code that starts ``command`` as a detached job and prints its info."""
    return (
        "import json, os, subprocess, time\n"
        f"_d, _id, _cmd, _cwd = {JOBS_DIR!r}, {job_id!r}, {command!r}, {cwd!r}\n"
        "os.makedirs(_d, exist_ok=True)\n"
        "_base = os.path.join(_d, _id)\n"
        "_log = open(_base + '.log', 'wb')\n"
        "_p = subprocess.Popen(['bash', '-c', 'bash -c \"$0\"; echo $? > \"$1\"', _cmd, _base + '.exit'],\n"
        "                      cwd=_cwd or None, stdin=subprocess.DEVNULL, stdout=_log,\n"
        "                      stderr=subprocess.STDOUT, start_new_session=True)\n"
        "_log.close()\n"
        "_meta = {'job_id': _id, 'pid': _p.pid, 'command': _cmd, 'cwd': _cwd,\n"
        "         'started': time.time(), 'log': _base + '.log'}\n"
        "with open(_base + '.json', 'w') as _f:\n"
        "    json.dump(_meta, _f)\n"
        "print(json.dumps(_meta))\n"
    )


def _job_helpers() -> str:
    return (
        "import glob, json, os, signal, time\n"
        f"_d = {JOBS_DIR!r}\n"
        "def _alive(pid):\n"
        "    try:\n"
        "        with open(f'/proc/{pid}/stat') as f:\n"
        "            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'\n"
        "    except OSError:\n"
        "        return False\n"
        "def _info(meta, tail):\n"
        "    base = os.path.join(_d, meta['job_id'])\n"
        "    code = None\n"
        "    if os.path.exists(base + '.exit'):\n"
        "        txt = open(base + '.exit').read().strip()\n"
        "        code = int(txt) if txt.lstrip('-').isdigit() else None\n"
        "    running = code is None and _alive(meta['pid'])\n"
        "    if code is None and not running:\n"
        "        state = 'killed'\n"
        "    else:\n"
        "        state = 'running' if running else 'exited'\n"
        "    if code is not None:\n"
        "        end = os.path.getmtime(base + '.exit')\n"
        "    elif running or not os.path.exists(base + '.log'):\n"
        "        end = time.time()\n"
        "    else:\n"
        "        end = max(os.path.getmtime(base + '.log'), meta['started'])\n"
        "    out = {'job_id': meta['job_id'], 'state': state, 'exit_code': code, 'pid': meta['pid'],\n"
        "           'command': meta['command'], 'seconds': round(end - meta['started'], 1),\n"
        "           'log': base + '.log'}\n"
        "    try:\n"
        "        out['log_bytes'] = os.path.getsize(base + '.log')\n"
        "    except OSError:\n"
        "        out['log_bytes'] = 0\n"
        "    if tail:\n"
        "        try:\n"
        "            with open(base + '.log', 'rb') as f:\n"
        "                f.seek(max(0, out['log_bytes'] - 200000))\n"
        "                lines = f.read().decode('utf-8', 'replace').splitlines()\n"
        "            out['tail'] = '\\n'.join(lines[-tail:])\n"
        "        except OSError:\n"
        "            out['tail'] = ''\n"
        "    return out\n"
        "def _load(job_id):\n"
        "    with open(os.path.join(_d, job_id + '.json')) as f:\n"
        "        return json.load(f)\n"
    )


def job_status_code(job_id: str, tail_lines: int) -> str:
    """Kernel code that prints a job's state, exit code and log tail."""
    return _job_helpers() + (
        f"_id, _tail = {job_id!r}, {int(tail_lines)!r}\n"
        "try:\n"
        "    print(json.dumps(_info(_load(_id), _tail)))\n"
        "except FileNotFoundError:\n"
        "    print(json.dumps({'error': 'unknown job ' + _id}))\n"
    )


def job_list_code() -> str:
    """Kernel code that prints every job's state, newest first."""
    return _job_helpers() + (
        "_jobs = []\n"
        "for _p in sorted(glob.glob(os.path.join(_d, '*.json')), key=os.path.getmtime, reverse=True):\n"
        "    with open(_p) as _f:\n"
        "        _jobs.append(_info(json.load(_f), 0))\n"
        "print(json.dumps(_jobs))\n"
    )


def job_kill_code(job_id: str) -> str:
    """Kernel code that stops a job's whole process group (TERM, then KILL)."""
    return _job_helpers() + (
        f"_id = {job_id!r}\n"
        "try:\n"
        "    _m = _load(_id)\n"
        "except FileNotFoundError:\n"
        "    print(json.dumps({'error': 'unknown job ' + _id}))\n"
        "else:\n"
        "    for _sig in (signal.SIGTERM, signal.SIGKILL):\n"
        "        if not _alive(_m['pid']):\n"
        "            break\n"
        "        try:\n"
        "            os.killpg(_m['pid'], _sig)\n"
        "        except ProcessLookupError:\n"
        "            break\n"
        "        for _ in range(50):\n"
        "            if not _alive(_m['pid']):\n"
        "                break\n"
        "            time.sleep(0.1)\n"
        "    print(json.dumps(_info(_m, 20)))\n"
    )
