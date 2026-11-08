# Dev results (mean ± std over seeds; std shown only when n > 1)

Dev numbers at the tuned threshold are slightly optimistic: dev picked the threshold.

| model    |    lr | seeds   |   n | dev_macro_f1_at_0.5   | dev_macro_f1_tuned   | dev_recall_offensive_tuned   | dev_recall_not_offensive_tuned   |
|:---------|------:|:--------|----:|:----------------------|:---------------------|:-----------------------------|:---------------------------------|
| hingbert | 1e-05 | 1,2,3   |   3 | 0.782 ± 0.004         | 0.788 ± 0.002        | 0.943 ± 0.005                | 0.607 ± 0.019                    |
| hingbert | 2e-05 | 1       |   1 | 0.779                 | 0.782                | 0.954                        | 0.564                            |
| hingbert | 3e-05 | 1       |   1 | 0.773                 | 0.778                | 0.956                        | 0.549                            |
| muril    | 1e-05 | 1       |   1 | 0.760                 | 0.766                | 0.942                        | 0.556                            |
| muril    | 2e-05 | 1,2,3   |   3 | 0.764 ± 0.008         | 0.770 ± 0.002        | 0.939 ± 0.011                | 0.575 ± 0.031                    |
| muril    | 3e-05 | 1       |   1 | 0.628                 | 0.629                | 0.991                        | 0.211                            |
| xlmr     | 1e-05 | 1,2,3   |   3 | 0.734 ± 0.006         | 0.754 ± 0.006        | 0.933 ± 0.010                | 0.552 ± 0.011                    |
| xlmr     | 2e-05 | 1       |   1 | 0.744                 | 0.746                | 0.935                        | 0.528                            |
| xlmr     | 3e-05 | 1       |   1 | 0.745                 | 0.750                | 0.952                        | 0.497                            |

TF-IDF baseline: macro-F1 @0.5 = 0.740, tuned = 0.751