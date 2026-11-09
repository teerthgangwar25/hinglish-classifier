# Error analysis on dev: hingbert_lr1e-05_seed1

Overall: {'threshold': 0.57, 'macro_f1': 0.79, 'recall_offensive': 0.941, 'recall_not_offensive': 0.619}

## By original 3-class label (share flagged offensive; high is good for OAG/CAG, bad for NAG)

| label3   |    n |   flagged_offensive |
|:---------|-----:|--------------------:|
| CAG      | 1168 |               0.912 |
| NAG      |  475 |               0.381 |
| OAG      | 1169 |               0.969 |

## By script

| script     |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:-----------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| devanagari |  559 |             0.794 |      0.774 |              0.926 |                   0.6  |        0.141 |
| mixed      |  119 |             0.496 |      0.924 |              0.898 |                   0.95 |        0.076 |
| roman      | 2134 |             0.859 |      0.764 |              0.945 |                   0.56 |        0.109 |

## By length

| length             |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:-------------------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| long (>40)         |  371 |             0.876 |      0.785 |              0.951 |                  0.609 |        0.092 |
| medium (11-40)     | 1798 |             0.869 |      0.773 |              0.966 |                  0.523 |        0.092 |
| short (<=10 words) |  643 |             0.698 |      0.783 |              0.844 |                  0.737 |        0.188 |

## By language mix (HingBERT-LID)

| mix_bucket     |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:---------------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| balanced       |  390 |             0.795 |      0.777 |              0.906 |                  0.65  |        0.146 |
| mostly-English |   12 |             0.667 |    nan     |              0.625 |                  1     |        0.25  |
| mostly-Hindi   | 2339 |             0.85  |      0.775 |              0.95  |                  0.568 |        0.108 |
| too-few-words  |   71 |             0.451 |      0.883 |              0.781 |                  0.974 |        0.113 |