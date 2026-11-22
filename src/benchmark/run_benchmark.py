"""Phase 4: load-test the API with Locust at several concurrency levels and batch sizes.

    python src/benchmark/run_benchmark.py                          # 3 concurrency levels x 3 batch sizes
    python src/benchmark/run_benchmark.py --compare-buckets 1000,8 --users 1,4   # A/B test of length bucketing

It starts the API itself (one process per setting, prediction log OFF so only inference is measured), warms it
up, runs Locust for every (batch, users) pair, samples the server's memory, and writes a report under reports/.

A/B mode starts one server per bucket value and, for every (batch, users) pair, runs them back to back in
alternating order. That keeps machine state (power mode, heat, background load) equal for both settings.
Bucket 1000 means "no bucketing" (the whole batch is one padded group), which is the old behaviour.

Needs: pip install locust psutil, and the model bundle from src/serve/prepare_bundle.py.
Limits: the load generator runs on the same laptop and takes some CPU, so absolute numbers are a floor.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60).read()


def wait_healthy(base, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit("the API process died during start-up (is the bundle built? python src/serve/prepare_bundle.py)")
        try:
            if urllib.request.urlopen(base + "/health", timeout=2).status == 200:
                return
        except Exception:
            time.sleep(1)
    sys.exit("the API did not become healthy in time")


def tree_rss(pid):
    """Memory of the server process AND its children. On Windows, a venv's python.exe is a small launcher that
    starts the real interpreter as a child, so measuring only the first pid reports a few MB."""
    try:
        procs = [psutil.Process(pid)]
        procs += procs[0].children(recursive=True)
    except psutil.Error:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def kill_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=10)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def port_in_use(port):
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def machine_state():
    """Power and clock state. A laptop on battery or in a power-saving mode can be several times slower."""
    bat = psutil.sensors_battery()
    plugged = "no battery" if bat is None else ("plugged in" if bat.power_plugged else "ON BATTERY")
    f = psutil.cpu_freq()
    return f"{plugged}, CPU clock {f.current:.0f} of {f.max:.0f} MHz" if f else plugged


class MemorySampler(threading.Thread):
    """Records the peak resident memory (RSS) of the whole server process tree."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = pid, 0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, tree_rss(self.pid))
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30, help="seconds per run with batch size 1")
    ap.add_argument("--batch-duration", type=int, default=90, help="seconds per run with batch size > 1")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads inside the API")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--users", default="1,4,16", help="concurrency levels, comma separated")
    ap.add_argument("--batches", default="1,8,32", help="batch sizes, comma separated")
    ap.add_argument("--bucket", type=int, default=8, help="length-bucket size inside the API (BUCKET_SIZE)")
    ap.add_argument("--compare-buckets", help="A/B mode: bucket sizes to compare, e.g. 1000,8")
    ap.add_argument("--report", default="benchmark.md", help="file name under reports/")
    a = ap.parse_args()

    users_list = [int(x) for x in a.users.split(",")]
    batch_list = [int(x) for x in a.batches.split(",")]
    buckets = [int(x) for x in a.compare_buckets.split(",")] if a.compare_buckets else [a.bucket]
    ports = {b: a.port + i for i, b in enumerate(buckets)}
    for port in ports.values():
        if port_in_use(port):
            sys.exit(f"port {port} is already in use: stop the old API server (or leftover python process), or pass --port")
    out_dir = ROOT / "reports" / "benchmark_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    servers, rows = {}, []
    state_start = machine_state()
    print("machine state:", state_start)
    try:
        for b in buckets:
            env = {**os.environ, "LOG_DB": "off", "ORT_THREADS": str(a.threads), "BUCKET_SIZE": str(b)}
            servers[b] = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve.app:app", "--app-dir", "src",
                                           "--port", str(ports[b]), "--log-level", "warning"], cwd=ROOT, env=env)
        for b in buckets:
            base = f"http://127.0.0.1:{ports[b]}"
            wait_healthy(base, servers[b])
            for _ in range(30):                                   # warm-up: first requests are slower
                post(base + "/predict", {"text": "yeh ek warm up comment hai"})
        meta = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{ports[buckets[0]]}/version").read())
        idle_mb = tree_rss(servers[buckets[0]].pid) / 1e6
        print(f"server(s) up, model {meta['model_version']}, idle memory {idle_mb:.0f} MB each")

        combo = 0
        for batch in batch_list:
            for users in users_list:
                order = buckets if combo % 2 == 0 else buckets[::-1]      # alternate order to balance drift
                combo += 1
                secs = a.duration if batch == 1 else a.batch_duration
                for b in order:
                    sampler = MemorySampler(servers[b].pid)
                    sampler.start()
                    prefix = out_dir / f"u{users}_b{batch}_k{b}"
                    print(f"running: bucket {b}, {users} users, batch {batch}, {secs}s ...", flush=True)
                    subprocess.run([sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--headless",
                                    "-u", str(users), "-r", str(users), "-t", f"{secs}s",
                                    "--host", f"http://127.0.0.1:{ports[b]}", "--csv", str(prefix), "--only-summary"],
                                   cwd=ROOT, env={**os.environ, "BATCH_SIZE": str(batch)},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    sampler.stop = True
                    sampler.join()
                    with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as f:
                        stats = list(csv.DictReader(f))
                    http = next(r for r in stats if r["Type"] == "POST")        # the real round trips
                    model = next(r for r in stats if r["Type"] == "MODEL")      # model time reported by the server
                    rps = float(http["Requests/s"])
                    rows.append({"bucket": b, "batch": batch, "users": users, "secs": secs, "rps": rps,
                                 "comments_s": rps * batch, "p50": float(http["50%"]), "p95": float(http["95%"]),
                                 "p99": float(http["99%"]), "m50": float(model["50%"]), "m95": float(model["95%"]),
                                 "requests": int(http["Request Count"]), "failures": int(http["Failure Count"]),
                                 "mem_mb": sampler.peak / 1e6})
        state_end = machine_state()
    finally:
        for p in servers.values():
            kill_tree(p.pid)

    bundle = ROOT / "serving" / "model"
    rows.sort(key=lambda r: (r["batch"], r["users"], buckets.index(r["bucket"])))
    lines = ["# API load test (Locust)", "",
             f"Model `{meta['model_version']}` (ONNX fp32), one uvicorn process per setting, {a.threads} ONNX Runtime threads, "
             f"prediction log off. Run length is in the Seconds column, no think time, so users = concurrency. "
             "Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; "
             "for batch runs one request carries several comments, so comments/s = requests/s x batch size.", "",
             f"Hardware: {platform.processor() or platform.machine()}, {psutil.cpu_count()} logical cores, "
             f"{psutil.virtual_memory().total / 1e9:.0f} GB RAM, {platform.system()} {platform.release()}. "
             f"Model file {(bundle / 'model.onnx').stat().st_size / 1e6:.0f} MB; server memory after warm-up {idle_mb:.0f} MB. "
             "The load generator ran on the same machine and used part of the CPU.",
             f"Machine state at start: {state_start}. At end: {state_end}.", ""]
    if len(buckets) > 1:
        lines += [f"A/B test of the length bucket ({' vs '.join(map(str, buckets))}; 1000 = no bucketing). For each "
                  "(batch, users) pair the settings ran back to back, in alternating order.", ""]
    lines += ["Percentiles from fewer than 100 requests are shown as n/a: with so few samples p95 and p99 are just the slowest "
              "request. 'Model-only' is the tokenise + inference time the API reports itself; the rest of the round trip is "
              "HTTP, JSON, queueing and the load generator.", "",
              "| Bucket | Batch size | Users | Seconds | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Model-only p50 ms | Model-only p95 ms | Failures | Peak server memory (MB) |",
              "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        few = r["requests"] < 100
        p95, p99, m95 = ("n/a", "n/a", "n/a") if few else (f"{r['p95']:.0f}", f"{r['p99']:.0f}", f"{r['m95']:.0f}")
        lines.append(f"| {r['bucket']} | {r['batch']} | {r['users']} | {r['secs']} | {r['rps']:.1f} | {r['comments_s']:.1f} | "
                     f"{r['p50']:.0f} | {p95} | {p99} | {r['m50']:.0f} | {m95} | {r['failures']} of {r['requests']} | {r['mem_mb']:.0f} |")
    if len(buckets) > 1:
        base_b, new_b = buckets[0], buckets[-1]
        lines += ["", f"**Comments/s with bucket {new_b} relative to bucket {base_b}:**", "",
                  "| Batch size | Users | Bucket " + str(base_b) + " | Bucket " + str(new_b) + " | Ratio |", "|--:|--:|--:|--:|--:|"]
        for batch in batch_list:
            for users in users_list:
                x = {r["bucket"]: r["comments_s"] for r in rows if r["batch"] == batch and r["users"] == users}
                if base_b in x and new_b in x and x[base_b] > 0:
                    lines.append(f"| {batch} | {users} | {x[base_b]:.1f} | {x[new_b]:.1f} | {x[new_b] / x[base_b]:.2f}x |")
    out = ROOT / "reports" / a.report
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
