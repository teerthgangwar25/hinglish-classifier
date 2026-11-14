"""Phase 4: load-test the API with Locust at several concurrency levels and batch sizes.

    python src/benchmark/run_benchmark.py                  # 3 concurrency levels x 3 batch sizes, 30 s each
    python src/benchmark/run_benchmark.py --duration 10 --threads 4

It starts the API itself (one process, prediction log OFF so only inference is measured), warms it up,
runs Locust for every (users, batch) pair, samples the server's memory, and writes reports/benchmark.md.
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
USERS = [1, 4, 16]
BATCHES = [1, 8, 32]


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30).read()


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


class MemorySampler(threading.Thread):
    """Records the peak resident memory (RSS) of the server process."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.p, self.peak, self.stop = psutil.Process(pid), 0, False

    def run(self):
        while not self.stop:
            try:
                self.peak = max(self.peak, self.p.memory_info().rss)
            except psutil.Error:
                return
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30, help="seconds per run")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads inside the API")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    out_dir = ROOT / "reports" / "benchmark_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "LOG_DB": "off", "ORT_THREADS": str(a.threads)}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve.app:app", "--app-dir", "src",
                               "--port", str(a.port), "--log-level", "warning"], cwd=ROOT, env=env)
    rows = []
    try:
        wait_healthy(base, server)
        meta = json.loads(urllib.request.urlopen(base + "/version").read())
        for _ in range(30):                                   # warm-up: first requests are slower
            post(base + "/predict", {"text": "yeh ek warm up comment hai"})
        idle_mb = psutil.Process(server.pid).memory_info().rss / 1e6
        print(f"server up, model {meta['model_version']}, idle memory {idle_mb:.0f} MB")

        for batch in BATCHES:
            for users in USERS:
                sampler = MemorySampler(server.pid)
                sampler.start()
                prefix = out_dir / f"u{users}_b{batch}"
                print(f"running: {users} users, batch {batch}, {a.duration}s ...", flush=True)
                subprocess.run([sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--headless",
                                "-u", str(users), "-r", str(users), "-t", f"{a.duration}s", "--host", base,
                                "--csv", str(prefix), "--only-summary"],
                               cwd=ROOT, env={**env, "BATCH_SIZE": str(batch)}, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                sampler.stop = True
                sampler.join()
                with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as f:
                    agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
                rps = float(agg["Requests/s"])
                rows.append({"batch": batch, "users": users, "rps": rps, "comments_s": rps * batch,
                             "p50": float(agg["50%"]), "p95": float(agg["95%"]), "p99": float(agg["99%"]),
                             "requests": int(agg["Request Count"]), "failures": int(agg["Failure Count"]),
                             "mem_mb": sampler.peak / 1e6})
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    bundle = ROOT / "serving" / "model"
    lines = ["# API load test (Locust)", "",
             f"Model `{meta['model_version']}` (ONNX fp32), served by one uvicorn process, {a.threads} ONNX Runtime threads, "
             f"prediction log off. {a.duration} s per run, no think time, so users = concurrency. "
             "Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; "
             "for batch runs one request carries several comments, so comments/s = requests/s x batch size.", "",
             f"Hardware: {platform.processor() or platform.machine()}, {psutil.cpu_count()} logical cores, "
             f"{psutil.virtual_memory().total / 1e9:.0f} GB RAM, {platform.system()} {platform.release()}. "
             f"Model file {(bundle / 'model.onnx').stat().st_size / 1e6:.0f} MB; server memory after warm-up {idle_mb:.0f} MB. "
             "The load generator ran on the same machine and used part of the CPU.", "",
             "| Batch size | Users | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Failures | Peak server memory (MB) |",
             "|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['batch']} | {r['users']} | {r['rps']:.1f} | {r['comments_s']:.1f} | {r['p50']:.0f} | "
                     f"{r['p95']:.0f} | {r['p99']:.0f} | {r['failures']} of {r['requests']} | {r['mem_mb']:.0f} |")
    out = ROOT / "reports" / "benchmark.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
