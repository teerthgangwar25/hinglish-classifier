# API load test (Locust)

Model `hingbert_lr1e-05_seed1-onnx-fp32` (ONNX fp32), served by one uvicorn process, 4 ONNX Runtime threads, prediction log off. 30 s per run, no think time, so users = concurrency. Latency is the full HTTP round trip (JSON, cleaning, tokenising, inference) per request; for batch runs one request carries several comments, so comments/s = requests/s x batch size.

Hardware: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 16 logical cores, 17 GB RAM, Windows 11. Model file 712 MB; server memory after warm-up 4 MB. The load generator ran on the same machine and used part of the CPU.

| Batch size | Users | Requests/s | Comments/s | p50 ms | p95 ms | p99 ms | Failures | Peak server memory (MB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1 | 7.9 | 7.9 | 100 | 290 | 300 | 0 of 217 | 4 |
| 1 | 4 | 8.4 | 8.4 | 420 | 950 | 1000 | 0 of 244 | 4 |
| 1 | 16 | 8.9 | 8.9 | 1500 | 4100 | 4200 | 0 of 270 | 4 |
| 8 | 1 | 0.5 | 4.0 | 2200 | 2200 | 2200 | 0 of 15 | 4 |
| 8 | 4 | 0.6 | 4.7 | 6400 | 8700 | 8700 | 0 of 17 | 4 |
| 8 | 16 | 0.4 | 3.0 | 24000 | 31000 | 31000 | 0 of 12 | 4 |
| 32 | 1 | 0.3 | 10.5 | 1600 | 15000 | 15000 | 0 of 9 | 4 |
| 32 | 4 | 0.7 | 21.0 | 5100 | 7400 | 7400 | 0 of 19 | 4 |
| 32 | 16 | 0.2 | 5.0 | 22000 | 25000 | 25000 | 0 of 4 | 4 |