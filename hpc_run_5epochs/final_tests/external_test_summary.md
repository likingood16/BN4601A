# External test sets (never trained on)

| source    |   n |   label_recall |   text_mentions_expected |   normal_correct |   best_iou |   recall50 |   negative_ok |   n_normal |   findings_rougeL |   findings_f1 |   impression_term_recall |   format_ok |   abnormal_detected |   normal_recognised |
|:----------|----:|---------------:|-------------------------:|-----------------:|-----------:|-----------:|--------------:|-----------:|------------------:|--------------:|-------------------------:|------------:|--------------------:|--------------------:|
| fracatlas |  25 |          0.8   |                     0.8  |            0.8   |      0.005 |      0     |             1 |        nan |            nan    |       nan     |                  nan     |         nan |                 nan |                 nan |
| nih_cxr14 |  25 |          0.3   |                     0.35 |            0.8   |      0     |      0     |             1 |        nan |            nan    |       nan     |                  nan     |         nan |                 nan |                 nan |
| slake     |  25 |          0.196 |                     0.28 |            1     |      0.046 |      0.014 |             1 |        nan |            nan    |       nan     |                  nan     |         nan |                 nan |                 nan |
| openi     |  60 |          0.105 |                   nan    |            0.667 |    nan     |    nan     |           nan |         20 |              0.07 |         0.074 |                    0.082 |           1 |                   1 |                   0 |

label_recall = share of expected finding labels named in findings + impression; normal_correct = normal/abnormal call matches the label; best_iou / recall50 = box overlap with the expected box. All are rule-based proxies - clinician review of the *_review.md / *.md files is the real test.
