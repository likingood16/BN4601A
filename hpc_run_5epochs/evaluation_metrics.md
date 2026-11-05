|                             |   base (no adapter) |   adapted (best) |   delta (adapted-base) |
|:----------------------------|--------------------:|-----------------:|-----------------------:|
| test_loss                   |              2.8138 |           1.8358 |                -0.978  |
| test_perplexity             |             16.6724 |           6.2698 |               -10.4026 |
| n_generated                 |             96      |          96      |                 0      |
| findings_rougeL             |              0.0952 |           0.2325 |                 0.1373 |
| findings_f1                 |              0.1162 |           0.2622 |                 0.1461 |
| assess_rougeL               |              0.1927 |           0.2521 |                 0.0594 |
| assess_term_recall          |              0.216  |           0.2251 |                 0.0091 |
| format_ok                   |              1      |           1      |                 0      |
| words_findings              |             87.1771 |          12.6771 |               -74.5    |
| words_assess                |            115.885  |          92.7083 |               -23.1771 |
| halluc_case_rate            |              0.0208 |           0.375  |                 0.3542 |
| halluc_per_case             |              0.0208 |           0.5104 |                 0.4896 |
| halluc_case_rate_image_only |              0      |           0.3438 |                 0.3438 |
| ground_parse_ok             |              0.4333 |           1      |                 0.5667 |
| ground_open_mIoU            |              0      |           0      |                 0      |
| ground_open_recall50        |              0      |           0      |                 0      |
| ground_open_precision50     |              0      |           0      |                 0      |
| ground_ref_mIoU             |              0.0117 |           0.163  |                 0.1513 |
| ground_ref_recall50         |              0      |           0.0714 |                 0.0714 |
| ground_ref_precision50      |              0      |           0.0714 |                 0.0714 |
| ground_negative_specificity |              1      |           1      |                 0      |
| n_ground_eval               |             60      |          60      |                 0      |