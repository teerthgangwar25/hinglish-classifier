# API load test (Locust)

Model `hingbert_lr1e-05_seed1-onnx-fp32` (ONNX fp32), one uvicorn process per setting, 4 ONNX Runtime threads, prediction log off. Run length is in the Seconds column, no think time, so users = concurrency. Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; for batch runs one request carries several comments, so comments/s = requests/s x batch size.

Hardware: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 16 logical cores, 17 GB RAM, Windows 11. Model file 712 MB; server memory after warm-up 842 MB. The load generator ran on the same machine and used part of the CPU.
Machine state at start: plugged in, CPU clock 2400 of 2400 MHz. At end: plugged in, CPU clock 1520 of 2400 MHz.

A/B test of the length bucket (1000 vs 8; 1000 = no bucketing). For each (batch, users) pair the settings ran back to back, in alternating order.

Percentiles from fewer than 100 requests are shown as n/a: with so few samples p95 and p99 are just the slowest request. 'Model-only' is the tokenise + inference time the API reports itself; the rest of the round trip is HTTP, JSON, queueing and the load generator.

| Bucket | Batch size | Users | Seconds | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Model-only p50 ms | Model-only p95 ms | Failures | Peak server memory (MB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1000 | 1 | 1 | 30 | 20.1 | 20.1 | 25 | 160 | 300 | 23 | 150 | 0 of 590 | 851 |
| 8 | 1 | 1 | 30 | 7.7 | 7.7 | 110 | 280 | 320 | 100 | 270 | 0 of 228 | 849 |
| 1000 | 1 | 4 | 30 | 8.1 | 8.1 | 440 | 970 | 1100 | 430 | 960 | 0 of 240 | 863 |
| 8 | 1 | 4 | 30 | 16.3 | 16.3 | 94 | 830 | 1100 | 91 | 770 | 0 of 486 | 862 |
| 1000 | 8 | 1 | 90 | 1.2 | 9.4 | 420 | 2200 | 2800 | 420 | 2200 | 0 of 104 | 905 |
| 8 | 8 | 1 | 90 | 0.6 | 5.0 | 1700 | n/a | n/a | 1700 | n/a | 0 of 55 | 903 |
| 1000 | 8 | 4 | 90 | 0.7 | 5.5 | 6100 | n/a | n/a | 6100 | n/a | 0 of 60 | 1014 |
| 8 | 8 | 4 | 90 | 0.6 | 4.5 | 7600 | n/a | n/a | 7600 | n/a | 0 of 50 | 1013 |
| 1000 | 32 | 1 | 90 | 0.1 | 4.0 | 8800 | n/a | n/a | 8800 | n/a | 0 of 10 | 1091 |
| 8 | 32 | 1 | 90 | 0.3 | 9.4 | 4000 | n/a | n/a | 4000 | n/a | 0 of 26 | 1014 |
| 1000 | 32 | 4 | 90 | 0.1 | 3.3 | 42000 | n/a | n/a | 42000 | n/a | 0 of 8 | 1509 |
| 8 | 32 | 4 | 90 | 0.2 | 7.8 | 17000 | n/a | n/a | 17000 | n/a | 0 of 22 | 1031 |

**Comments/s with bucket 8 relative to bucket 1000:**

| Batch size | Users | Bucket 1000 | Bucket 8 | Ratio |
|--:|--:|--:|--:|--:|
| 1 | 1 | 20.1 | 7.7 | 0.38x |
| 1 | 4 | 8.1 | 16.3 | 2.01x |
| 8 | 1 | 9.4 | 5.0 | 0.53x |
| 8 | 4 | 5.5 | 4.5 | 0.82x |
| 32 | 1 | 4.0 | 9.4 | 2.38x |
| 32 | 4 | 3.3 | 7.8 | 2.40x |