# API load test (Locust)

Model `hingbert_lr1e-05_seed1-onnx-fp32` (ONNX fp32), served by one uvicorn process, 4 ONNX Runtime threads, prediction log off. 30 s per run, no think time, so users = concurrency. Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; for batch runs one request carries several comments, so comments/s = requests/s x batch size.

Hardware: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 16 logical cores, 17 GB RAM, Windows 11. Model file 712 MB; server memory after warm-up 842 MB. The load generator ran on the same machine and used part of the CPU.

Percentiles from fewer than 100 requests are shown as n/a: with so few samples p95 and p99 are just the slowest request. 'Model-only' is the tokenise + inference time the API reports itself; the rest of the round trip is HTTP, JSON, queueing and the load generator.

| Batch size | Users | Seconds | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Model-only p50 ms | Model-only p95 ms | Failures | Peak server memory (MB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1 | 30 | 40.3 | 40.3 | 20 | 55 | 60 | 18 | 53 | 0 of 1178 | 851 |
| 1 | 4 | 30 | 49.5 | 49.5 | 70 | 170 | 200 | 66 | 170 | 0 of 1445 | 863 |
| 1 | 16 | 30 | 53.5 | 53.5 | 250 | 680 | 830 | 240 | 680 | 0 of 1561 | 897 |
| 8 | 1 | 90 | 2.5 | 19.8 | 430 | 540 | 570 | 430 | 540 | 0 of 219 | 909 |
| 8 | 4 | 90 | 2.9 | 23.5 | 1500 | 1800 | 1900 | 1500 | 1800 | 0 of 259 | 1022 |
| 8 | 16 | 90 | 2.9 | 23.4 | 5300 | 7800 | 8100 | 5300 | 7700 | 0 of 262 | 1270 |
| 32 | 1 | 90 | 0.5 | 15.2 | 2100 | n/a | n/a | 2100 | n/a | 0 of 42 | 1270 |
| 32 | 4 | 90 | 0.6 | 18.8 | 6800 | n/a | n/a | 6800 | n/a | 0 of 52 | 1627 |
| 32 | 16 | 90 | 0.6 | 17.6 | 26000 | n/a | n/a | 26000 | n/a | 0 of 47 | 2805 |