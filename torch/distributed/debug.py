import os
import socket
import multiprocessing
import requests
from concurrent.futures import ThreadPoolExecutor
import json
import time
import tempfile

from flask import Flask

import torch.distributed as dist
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
    _ExperimentalConfig,
)
from torch._C._distributed_c10d import _WorkerServer, _register_handler


def _torch_profile(req, resp):
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(2)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(open(f.name, "rb").read(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)

MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = int(os.environ["MASTER_PORT"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def _tcpstore_client() -> dist.Store:
    store = dist.TCPStore(
        host_name=MASTER_ADDR,
        port=MASTER_PORT,
        is_master=False,
    )
    store = dist.PrefixStore("debug_server", store)
    return store


def fetch_all(endpoint: str) -> list[bytes]:
    store = _tcpstore_client()
    keys = [f"rank{r}" for r in range(WORLD_SIZE)]
    addrs = store.multi_get(keys)
    addrs = [f"{addr.decode()}/handler/{endpoint}" for addr in addrs]

    with ThreadPoolExecutor(max_workers=10) as executor:
        resps = executor.map(requests.post, addrs)

    return addrs, resps


app = Flask(__name__)


def nav():
    return """
    <style>
        body {
            font-family: sans-serif;
        }
        pre {
            white-space: pre-wrap;
            max-width: 100%;
        }
    </style>
    <h1>Torch Distributed Debug Server</h1>

    <ul>
        <li><a href="/">Home</a></li>
        <li><a href="/stacks">Python Stack Traces</a></li>
        <li><a href="/fr_trace">FlightRecorder</a></li>
        <li><a href="/fr_trace_nccl">FlightRecorder NCCL</a></li>
        <li><a href="/profile">torch profiler</a></li>
    </ul>
    """


@app.route("/")
def index():
    return nav()


@app.route("/stacks")
def stacks():
    addrs, resps = fetch_all("dump_traceback")

    def generate():
        yield nav()

        yield "<h2>Stacks</h2>"

        for i, addr, resp in zip(range(len(addrs)), addrs, resps):
            yield f"<h3>Rank {i}: {addr}</h3>"
            if resp.status_code != 200:
                yield f"<p>Failed to fetch: status={resp.status_code}</p>"

            stack = resp.text
            yield f"<pre>{stack}</pre>"

    return generate()


def format_json(blob: str):
    parsed = json.loads(blob)
    return json.dumps(parsed, indent=2)


@app.route("/fr_trace")
def fr_trace():
    addrs, resps = fetch_all("fr_trace_json")

    def generate():
        yield nav()

        yield "<h2>FlightRecorder</h2>"

        for i, addr, resp in zip(range(len(addrs)), addrs, resps):
            yield f"<h3>Rank {i}: {addr}</h3>"
            if resp.status_code != 200:
                yield f"<p>Failed to fetch: status={resp.status_code}</p>"

            stack = format_json(resp.text)
            yield f"<pre>{stack}</pre>"

    return generate()


@app.route("/fr_trace_nccl")
def fr_trace_nccl():
    addrs, resps = fetch_all("dump_nccl_trace_json?onlyactive=true")

    def generate():
        yield nav()

        yield "<h2>FlightRecorder NCCL</h2>"

        for i, addr, resp in zip(range(len(addrs)), addrs, resps):
            yield f"<h3>Rank {i}: {addr}</h3>"
            if resp.status_code != 200:
                yield f"<p>Failed to fetch: status={resp.status_code}</p>"

            stack = format_json(resp.text)
            yield f"<pre>{stack}</pre>"

    return generate()


@app.route("/profile")
def profiler():
    addrs, resps = fetch_all("torch_profile")

    def generate():
        yield nav()

        yield """
        <h2>torch profile</h2>
        <script>
        function stringToArrayBuffer(str) {
            const encoder = new TextEncoder();
            return encoder.encode(str).buffer;
        }
        async function openPerfetto(data) {
            const ui = window.open('https://ui.perfetto.dev/#!/');
            if (!ui) { alert('Popup blocked. Allow popups for this page and click again.'); return; }

            // Perfetto readiness handshake: PING until we receive PONG
            await new Promise((resolve, reject) => {
            const onMsg = (e) => {
                if (e.source === ui && e.data === 'PONG') {
                window.removeEventListener('message', onMsg);
                clearInterval(pinger);
                resolve();
                }
            };
            window.addEventListener('message', onMsg);
            const pinger = setInterval(() => { try { ui.postMessage('PING', '*'); } catch (_e) {} }, 250);
            setTimeout(() => { clearInterval(pinger); window.removeEventListener('message', onMsg); reject(); }, 20000);
            }).catch(() => { alert('Perfetto UI did not respond. Try again.'); return; });

            ui.postMessage({
            perfetto: {
                buffer: stringToArrayBuffer(JSON.stringify(data)),
                title: "torch profiler",
                fileName: "trace.json",
            }
            }, '*');
        }
        </script>
        """

        for i, addr, resp in zip(range(len(addrs)), addrs, resps):
            yield f"<h3>Rank {i}: {addr}</h3>"
            if resp.status_code != 200:
                yield f"<p>Failed to fetch: status={resp.status_code}</p>"

            stack = resp.text
            yield f"""
            <script>
            function run{i}() {{
                var data = {stack};
                openPerfetto(data);
            }}
            </script>
            <button onclick="run{i}()">View {i}</button>
            """

    return generate()


def _interactive_server() -> None:
    app.run(host="::", port=25999)


def enable_debug_server() -> None:
    global _worker_server, _p

    store = _tcpstore_client()

    _worker_server = _WorkerServer("::", 0)
    store.set(f"rank{RANK}", f"http://{socket.gethostname()}:{_worker_server.port}")

    if RANK == 0:
        _p = multiprocessing.Process(
            target=_interactive_server,
        )
        _p.start()
