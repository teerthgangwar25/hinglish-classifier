# Test results (single sealed evaluation)

Scored 2026-11-11 12:32:28 UTC at git commit `c7244a24`. Thresholds were frozen on dev; nothing was tuned on test.

Bootstrap: 2000 resamples of the test set, percentile 95% intervals. McNemar: exact, two-sided; 4 planned tests, so the Bonferroni level is 0.0125.

## FB (970 rows, aggressive share 0.799)

| Model | Macro-F1 (mean ± sd over seeds) | 95% CI | Macro-F1 @0.5 | Recall aggr. | Recall not-aggr. | Recall floor 0.90 held? |
|:--|:--|:--|:--|:--|:--|:--|
| hingbert | 0.843 ± 0.014 | [0.816, 0.869] | 0.847 | 0.957 | 0.694 | yes |
| muril | 0.813 ± 0.005 | [0.782, 0.841] | 0.815 | 0.963 | 0.607 | yes |
| xlmr | 0.801 ± 0.002 | [0.768, 0.831] | 0.803 | 0.963 | 0.579 | yes |
| tfidf | 0.769 ± n/a | [0.734, 0.801] | 0.771 | 0.945 | 0.549 | yes |

**Paired comparisons** (difference in seed-averaged macro-F1; McNemar per seed-matched pair)

| Comparison | Δ macro-F1 | 95% CI of Δ | CI excludes 0? | McNemar per seed: (b, c, p) |
|:--|:--|:--|:--|:--|
| hingbert − muril | +0.030 | [+0.007, +0.054] | yes | (40, 18, 0.005); (42, 36, 0.572); (45, 34, 0.260) |
| hingbert − tfidf | +0.074 | [+0.045, +0.105] | yes | (71, 23, 0.000); (61, 28, 0.001); (68, 34, 0.001) |

b = first model right and second wrong, c = the reverse. The three seed pairs are not independent tests.

## TW (1194 rows, aggressive share 0.704)

| Model | Macro-F1 (mean ± sd over seeds) | 95% CI | Macro-F1 @0.5 | Recall aggr. | Recall not-aggr. | Recall floor 0.90 held? |
|:--|:--|:--|:--|:--|:--|:--|
| hingbert | 0.602 ± 0.053 | [0.575, 0.627] | 0.588 | 0.504 | 0.854 | NO |
| muril | 0.625 ± 0.027 | [0.600, 0.649] | 0.614 | 0.634 | 0.681 | NO |
| xlmr | 0.608 ± 0.052 | [0.583, 0.631] | 0.566 | 0.581 | 0.732 | NO |
| tfidf | 0.597 ± n/a | [0.569, 0.625] | 0.563 | 0.543 | 0.760 | NO |

**Paired comparisons** (difference in seed-averaged macro-F1; McNemar per seed-matched pair)

| Comparison | Δ macro-F1 | 95% CI of Δ | CI excludes 0? | McNemar per seed: (b, c, p) |
|:--|:--|:--|:--|:--|
| hingbert − muril | -0.023 | [-0.048, +0.000] | no | (128, 203, 0.000); (157, 159, 0.955); (149, 215, 0.001) |
| hingbert − tfidf | +0.004 | [-0.022, +0.031] | no | (143, 219, 0.000); (204, 140, 0.001); (180, 166, 0.485) |

b = first model right and second wrong, c = the reverse. The three seed pairs are not independent tests.
