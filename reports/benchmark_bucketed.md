# API load test (Locust)

Model `hingbert_lr1e-05_seed1-onnx-fp32` (ONNX fp32), served by one uvicorn process, 4 ONNX Runtime threads, prediction log off, length bucket 8. 30 s per run, no think time, so users = concurrency. Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; for batch runs one request carries several comments, so comments/s = requests/s x batch size.

Hardware: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 16 logical cores, 17 GB RAM, Windows 11. Model file 712 MB; server memory after warm-up 842 MB. The load generator ran on the same machine and used part of the CPU.

Percentiles from fewer than 100 requests are shown as n/a: with so few samples p95 and p99 are just the slowest request. 'Model-only' is the tokenise + inference time the API reports itself; the rest of the round trip is HTTP, JSON, queueing and the load generator.

| Batch size | Users | Seconds | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Model-only p50 ms | Model-only p95 ms | Failures | Peak server memory (MB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 1 | 90 | 0.6 | 4.4 | 2100 | n/a | n/a | 2100 | n/a | 0 of 49 | 906 |
| 8 | 4 | 90 | 0.7 | 5.9 | 5700 | n/a | n/a | 5700 | n/a | 0 of 66 | 1015 |
| 8 | 16 | 90 | 0.6 | 4.8 | 22000 | n/a | n/a | 22000 | n/a | 0 of 53 | 1290 |
| 32 | 1 | 90 | 0.2 | 8.0 | 4000 | n/a | n/a | 4000 | n/a | 0 of 22 | 1290 |
| 32 | 4 | 90 | 0.7 | 23.0 | 2500 | n/a | n/a | 2500 | n/a | 0 of 64 | 1293 |
| 32 | 16 | 90 | 0.2 | 5.6 | 79000 | n/a | n/a | 78000 | n/a | 0 of 16 | 1311 |