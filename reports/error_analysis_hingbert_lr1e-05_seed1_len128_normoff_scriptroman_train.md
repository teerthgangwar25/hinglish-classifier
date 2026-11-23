# Error analysis on dev: hingbert_lr1e-05_seed1_len128_normoff_scriptroman_train

Overall: {'threshold': 0.34, 'macro_f1': 0.756, 'recall_offensive': 0.921, 'recall_not_offensive': 0.587}

## By original 3-class label (share flagged offensive; high is good for OAG/CAG, bad for NAG)

| label3   |    n |   flagged_offensive |
|:---------|-----:|--------------------:|
| CAG      | 1168 |               0.878 |
| NAG      |  475 |               0.413 |
| OAG      | 1169 |               0.964 |

## By script

| script     |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:-----------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| devanagari |  559 |             0.794 |      0.729 |              0.82  |                  0.713 |        0.202 |
| mixed      |  119 |             0.496 |      0.848 |              0.898 |                  0.8   |        0.151 |
| roman      | 2134 |             0.859 |      0.738 |              0.946 |                  0.497 |        0.117 |

## By length

| length             |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:-------------------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| long (>40)         |  371 |             0.876 |      0.777 |              0.932 |                  0.652 |        0.102 |
| medium (11-40)     | 1798 |             0.869 |      0.731 |              0.945 |                  0.489 |        0.115 |
| short (<=10 words) |  643 |             0.698 |      0.753 |              0.829 |                  0.691 |        0.213 |

## By language mix (HingBERT-LID)

| mix_bucket     |    n |   offensive_share |   macro_f1 |   recall_offensive |   recall_not_offensive |   error_rate |
|:---------------|-----:|------------------:|-----------:|-------------------:|-----------------------:|-------------:|
| balanced       |  390 |             0.795 |      0.741 |              0.906 |                  0.562 |        0.164 |
| mostly-English |   12 |             0.667 |    nan     |              0.75  |                  1     |        0.167 |
| mostly-Hindi   | 2339 |             0.85  |      0.741 |              0.926 |                  0.551 |        0.131 |
| too-few-words  |   71 |             0.451 |      0.871 |              0.812 |                  0.923 |        0.127 |