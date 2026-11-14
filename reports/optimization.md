# Optimisation comparison (D8): hingbert_lr1e-05_seed1

Dev split, 2812 comments, CPU, 4 threads, frozen threshold 0.57. CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (16 logical cores). Latency is end-to-end (tokenise + model) over 200 single requests; throughput at batch 32.

| Format | Macro-F1 | Drop vs PyTorch | Recall aggr. | Recall not-aggr. | Same label as PyTorch | Mean abs prob diff | Size (MB) | p50 ms | p95 ms | Comments/s |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| PyTorch fp32 | 0.7894 | +0.0000 | 0.940 | 0.619 | 1.0000 | 0.0000 | 711 | 26.4 | 63.9 | 15.0 |
| ONNX fp32 | 0.7894 | +0.0000 | 0.940 | 0.619 | 1.0000 | 0.0000 | 712 | 22.1 | 56.1 | 16.5 |
| ONNX int8 | 0.7757 | +0.0137 | 0.923 | 0.632 | 0.9733 | 0.0625 | 179 | 9.2 | 22.3 | 43.6 |

**Rule (D8):** adopt a faster format only if the macro-F1 drop is at most 0.010.
**Decision:** ONNX fp32.