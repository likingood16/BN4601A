|                             |   base (no adapter) |   adapted (best) |   delta (adapted-base) |
|:----------------------------|--------------------:|-----------------:|-----------------------:|
| test_loss                   |              2.8759 |           2.0338 |                -0.8421 |
| test_perplexity             |             17.7412 |           7.6427 |               -10.0985 |
| n_generated                 |             16      |          16      |                 0      |
| findings_rougeL             |              0.0889 |           0.2447 |                 0.1558 |
| findings_f1                 |              0.12   |           0.2664 |                 0.1464 |
| assess_rougeL               |              0.1902 |           0.24   |                 0.0498 |
| assess_term_recall          |              0.2267 |           0.2148 |                -0.0119 |
| format_ok                   |              1      |           1      |                 0      |
| words_findings              |             86.75   |          13.125  |               -73.625  |
| words_assess                |            124.875  |         113.312  |               -11.5625 |
| halluc_case_rate            |              0      |           0.625  |                 0.625  |
| halluc_per_case             |              0      |           1.0625 |                 1.0625 |
| halluc_case_rate_image_only |              0      |           1      |                 1      |
| ground_parse_ok             |              0.4545 |           1      |                 0.5455 |
| ground_open_mIoU            |              0      |           0.089  |                 0.089  |
| ground_open_recall50        |              0      |           0      |                 0      |
| ground_open_precision50     |              0      |           0      |                 0      |
| ground_ref_mIoU             |              0      |           0.1435 |                 0.1435 |
| ground_ref_recall50         |              0      |           0.1333 |                 0.1333 |
| ground_ref_precision50      |              0      |           0.1333 |                 0.1333 |
| ground_negative_specificity |              1      |           1      |                 0      |
| n_ground_eval               |             22      |          22      |                 0      |