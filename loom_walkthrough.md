# Loom walkthrough (target: 2–3 minutes)

Record your own screen and voice. This is a recording script, not a Loom link.

## 0:00–0:25 — Task and data
Show README and artifacts/audit.json. I used 48,000 labeled loads from January
through October to predict the 12,000 November–December loads. I checked unique
IDs, missing fields, invalid weights, possible target anomalies and new cities.

## 0:25–0:55 — Data quality
Show monthly_quote_rpm_correlation in audit.json. The quote signal changes its
relationship with rates across months, so I tested including and excluding it.
Nonpositive weights are treated as missing and flagged. I retained unusual
targets rather than deleting inconvenient rows. There are eight new cities;
the main model also uses supplied coordinates.

## 0:55–1:25 — Split and validation
Show artifacts/metrics.csv. January–July is training, August is for selection
and early stopping, and September–October is the locked test. The choice was
made on August MAE, then the selected tree count was fixed for the test.
The main model's locked-test MAE is $117.41, RMSE is
$636.91, and R-squared is 0.8258.
These are internal results, not the hidden final score.

## 1:25–1:55 — Code and model
Show features(), fit_model(), and the candidate-selection block in train.py.
The selected model is core_rpm_mae. CatBoost supports
mixed numerical and categorical freight data. Features include lane,
equipment, distance, weight, calendar variables and market/geographic inputs.
IDs and target values never enter features. Final fitting uses all labeled data.

## 1:55–2:30 — Outputs and December
Show the CSV header, the scorer terminal output and candidate_december.png.
Predictions are matched to template load IDs. The December scenario has fewer
features, so I trained and separately validated a reduced-schema model rather
than fabricating market or quote values. Only date changes in this chart.
The unmodified supplied scorer validates both files and generates this image.

## 2:30–2:45 — Limits
Only one incomplete year is available. Winter behavior and unseen-city accuracy
remain uncertain. More historical years and rolling/geographic holdouts would
be the next improvements. Share the actual Loom URL with the submission.
