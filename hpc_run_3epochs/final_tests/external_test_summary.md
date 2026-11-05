# External test sets (never trained on)

| source    |   n |   label_recall |   text_mentions_expected |   normal_correct |   best_iou |   recall50 |   negative_ok |   n_normal |   findings_rougeL |   findings_f1 |   impression_term_recall |   format_ok |   abnormal_detected |   normal_recognised |
|:----------|----:|---------------:|-------------------------:|-----------------:|-----------:|-----------:|--------------:|-----------:|------------------:|--------------:|-------------------------:|------------:|--------------------:|--------------------:|
| fracatlas |  25 |          0.85  |                     0.85 |            0.8   |      0.008 |      0     |           1   |        nan |           nan     |       nan     |                  nan     |         nan |              nan    |                 nan |
| nih_cxr14 |  25 |          0.3   |                     0.35 |            0.8   |      0.121 |      0.1   |           0.8 |        nan |           nan     |       nan     |                  nan     |         nan |              nan    |                 nan |
| slake     |  25 |          0.174 |                     0.2  |            1     |      0.102 |      0.058 |           1   |        nan |           nan     |       nan     |                  nan     |         nan |              nan    |                 nan |
| openi     |  60 |          0.068 |                   nan    |            0.633 |    nan     |    nan     |         nan   |         20 |             0.067 |         0.075 |                    0.037 |           1 |                0.95 |                   0 |

label_recall = share of expected finding labels named in findings + impression; normal_correct = normal/abnormal call matches the label; best_iou / recall50 = box overlap with the expected box. All are rule-based proxies - clinician review of the *_review.md / *.md files is the real test.
