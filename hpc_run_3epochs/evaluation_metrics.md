|                             |   base (no adapter) |   adapted (best) |   delta (adapted-base) |
|:----------------------------|--------------------:|-----------------:|-----------------------:|
| test_loss                   |              2.8138 |           1.8351 |                -0.9786 |
| test_perplexity             |             16.6724 |           6.266  |               -10.4064 |
| n_generated                 |             96      |          96      |                 0      |
| findings_rougeL             |              0.0952 |           0.2271 |                 0.1319 |
| findings_f1                 |              0.1162 |           0.2539 |                 0.1377 |
| assess_rougeL               |              0.1927 |           0.2481 |                 0.0554 |
| assess_term_recall          |              0.216  |           0.2389 |                 0.0229 |
| format_ok                   |              1      |           1      |                 0      |
| words_findings              |             87.1771 |          14.4583 |               -72.7188 |
| words_assess                |            115.885  |         101.219  |               -14.6667 |
| halluc_case_rate            |              0.0208 |           0.3646 |                 0.3438 |
| halluc_per_case             |              0.0208 |           0.5729 |                 0.5521 |
| halluc_case_rate_image_only |              0      |           0.25   |                 0.25   |
| ground_parse_ok             |              0.4333 |           1      |                 0.5667 |
| ground_open_mIoU            |              0      |           0.0742 |                 0.0742 |
| ground_open_recall50        |              0      |           0.05   |                 0.05   |
| ground_open_precision50     |              0      |           0.1    |                 0.1    |
| ground_ref_mIoU             |              0.0117 |           0.1952 |                 0.1835 |
| ground_ref_recall50         |              0      |           0.119  |                 0.119  |
| ground_ref_precision50      |              0      |           0.119  |                 0.119  |
| ground_negative_specificity |              1      |           0.875  |                -0.125  |
| n_ground_eval               |             60      |          60      |                 0      |