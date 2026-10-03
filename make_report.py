"""Generate a factual DOCX report and timed Loom script from run artifacts."""
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent


def main():
    result = json.loads((ROOT / "artifacts/results.json").read_text())
    audit = json.loads((ROOT / "artifacts/audit.json").read_text())
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Freight Rate Prediction — Validation Report", 0)
    doc.add_paragraph("Reproducible submission based only on supplied development labels.")
    doc.add_heading("1. Data and quality checks", 1)
    doc.add_paragraph(
        "The labeled development data contains 48,000 loads dated January 1–October 31, 2025. "
        "The unlabeled prediction data contains 12,000 loads dated November 1–December 31, 2025. "
        "All development targets are present and positive. No duplicate IDs or complete "
        "development feature-and-target rows were found."
    )
    doc.add_paragraph(
        f"Development data has {audit['missing_development']['weight']} missing weights, "
        f"{audit['nonpositive_weights_development']} nonpositive weights and "
        f"{audit['missing_development']['market_index']} missing market indices. "
        f"The prediction data has {audit['missing_validation']['weight']} missing weights, "
        f"{audit['nonpositive_weights_validation']} nonpositive weights and "
        f"{audit['missing_validation']['market_index']} missing market indices. "
        "Nonpositive weights are marked missing, not converted to their absolute values. "
        "A weight-invalid flag distinguishes affected rows. Numeric missingness is encoded "
        "as -999, a sentinel outside the observed positive weight/market-index ranges; "
        "categorical missingness is encoded as Unknown."
    )
    doc.add_paragraph(
        f"There are {len(audit['new_validation_pickups'])} new cities at prediction time: "
        + ", ".join(audit["new_validation_pickups"]) + ". "
        "CatBoost handles unseen category values; supplied latitude/longitude features "
        "give the main model a geographic representation without needing those city names "
        "to have appeared during training. Coordinates were used as supplied, with no "
        "unsupported geocoding corrections."
    )
    doc.add_paragraph(
        f"{audit['rate_per_mile_above_4']} development rates exceed $4/mile. "
        "These are possible anomalies, not proven labeling errors. They are retained in "
        "training and every validation metric. MAE-loss candidates reduce their leverage. "
        "No target trimming or outlier exclusion was used to improve reported scores."
    )
    doc.add_paragraph(
        "The quote signal is not temporally stable: its monthly correlation with observed "
        "rate/mile changes sign and is approximately zero in August. This may indicate "
        "regime-dependent behavior, contamination or unreliable availability; its provenance "
        "is unknown. Both inclusion and exclusion were tested on future-month tuning data. "
        "It is not assumed to be a trustworthy direct quote of the target."
    )
    doc.add_heading("2. Temporal split and leakage prevention", 1)
    rows = result["split_rows"]
    doc.add_paragraph(
        f"Training: January–July ({rows['Jan_Jul_training']:,} loads). "
        f"Tuning/early stopping: August ({rows['August_tuning']:,} loads). "
        f"Locked test: September–October ({rows['September_October_locked_test']:,} loads). "
        "This forward split mirrors predicting later months. A shuffled random split would "
        "mix time regimes and overstate reliability."
    )
    doc.add_paragraph(
        "Selection is based only on August dollar MAE. Each candidate is trained with up "
        "to 1,400 trees and early stopping after 130 unimproved iterations on its own loss. "
        "After selection, the selected tree count is fixed, the model is refit on January–August, "
        "and September–October is evaluated once without selecting another candidate. "
        "The final model is then refit on all January–October labeled loads. "
        "The 12,000 final-validation labels are not available and are never used. "
        "load_id is used only for output matching, not as a model feature. "
        "Feature transforms are row-local and do not use target means or test statistics."
    )
    doc.add_heading("3. Models and results", 1)
    doc.add_paragraph(
        "Baseline: median development dollars/mile multiplied by distance. "
        "Candidates: CatBoost with core features or core plus quote signal, MAE or RMSE "
        "loss, and either dollar rate or rate/mile targets. Rate/mile predictions are "
        "converted to dollars using the row's distance. The chart model is selected "
        "independently among reduced-schema candidates. All candidates use depth 7, "
        "learning rate 0.055, L2 regularization 5, seed 42 and four CPU threads."
    )
    doc.add_paragraph(
        "Features include origin, destination, directed lane, equipment, distance, log distance, "
        "cleaned weight, weight/distance, missing-weight flag, elapsed days, weekday, weekend, "
        "day-of-month, month and annual sine/cosine terms. The main schema adds coordinates, "
        "latitude/longitude differences and market index. Categories are learned by CatBoost "
        "within the fitted training data."
    )
    doc.add_paragraph(
        f"Selected main model: {result['selected']['main']['name']} "
        f"({result['selected']['main']['trees']} trees). "
        f"Selected chart model: {result['selected']['chart']['name']} "
        f"({result['selected']['chart']['trees']} trees)."
    )
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Shading Accent 1"
    for cell, text in zip(table.rows[0].cells, ["Model / split", "MAE ($)", "RMSE ($)", "R²", "MAPE"]):
        cell.text = text
    for row in result["metrics"]:
        if row["split"] == "locked test seen cities":
            continue
        cells = table.add_row().cells
        values = [row["name"] + "\n" + row["split"], f"{row['MAE']:.2f}",
                  f"{row['RMSE']:.2f}", f"{row['R2']:.4f}", f"{row['MAPE_percent']:.2f}%"]
        for cell, text in zip(cells, values):
            cell.text = text
    doc.add_paragraph(
        "MAE is the primary selection metric because no official accuracy metric was provided. "
        "RMSE highlights sensitivity to rare high-rate loads. MAPE is diagnostic and can "
        "overweight small actual rates. These are internal temporal validation results, "
        "not the organizer's hidden final-validation score."
    )
    importance = result["feature_importance"]["main"]
    top = sorted(importance, key=importance.get, reverse=True)[:6]
    doc.add_paragraph(
        "Top final-main-model feature importances: "
        + ", ".join(f"{name} ({importance[name]:.1f})" for name in top)
        + ". CatBoost importances are model-specific and are not evidence of causal effects."
    )
    doc.add_heading("4. Fixed December chart", 1)
    doc.add_paragraph(
        "The chart inputs contain only pickup, delivery, distance, equipment, weight and date. "
        "They do not contain coordinates, market index or quote signal. Rather than inventing "
        "those values or silently applying a different feature schema, a dedicated model is "
        "trained using only features available in that input. Its locked temporal test score "
        "is shown above. This is a separate reduced-information forecast, not the full model's "
        "prediction at unspecified values of the omitted features."
    )
    doc.add_paragraph(
        "For every day in December 2025 the fixed scenario is Lexington → Fort Wayne, "
        "360 miles, Dry Van, 32,000 lb. Only date changes. The following image was generated "
        "by the supplied scorer, copied byte-for-byte to score.py without modifying its "
        "plotting or validation logic."
    )
    doc.add_picture(str(ROOT / "scorer_results/candidate_december.png"), width=Inches(6.4))
    doc.add_paragraph(
        "Scorer execution: python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv. "
        "The scorer confirmed 12,000 valid final predictions and 31 valid fixed-scenario "
        "December predictions. It does not calculate prediction-accuracy metrics; those "
        "are calculated by the organizer after submission."
    )
    doc.add_heading("5. Reproducibility and limitations", 1)
    doc.add_paragraph(
        "Run train.py, score.py, test_solution.py and make_report.py in that order. "
        "The final CSV is filled by joining on load_id in template order, not by assuming "
        "the validation rows and template rows are aligned. Positive finite outputs are "
        "required by the scorer; predictions are floored at $1 and written to cents. "
        "artifacts/audit.json, results.json, metrics.csv and locked_test_predictions.csv "
        "record the data checks, configuration and internal holdout results."
    )
    doc.add_paragraph(
        "Only ten months from one year are available. Annual seasonality and holiday effects "
        "are therefore weakly identified, and CatBoost does not extrapolate numerical trends "
        "beyond training support reliably. The locked test covers September–October, not "
        "November–December; it cannot prove performance during winter or on the eight unseen "
        "cities. No spatial holdout experiment was used. Future work should use multiple years, "
        "rolling-origin validation and a geographic holdout, and verify quote-signal provenance "
        "before relying on it. No holiday-specific coefficients were invented."
    )
    doc.save(ROOT / "freight_rate_report.docx")
    main_test = next(r for r in result["metrics"] if r.get("role") == "main"
                     and r["split"] == "September-October locked test")
    script = f"""# Loom walkthrough (target: 2–3 minutes)

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
The main model's locked-test MAE is ${main_test['MAE']:.2f}, RMSE is
${main_test['RMSE']:.2f}, and R-squared is {main_test['R2']:.4f}.
These are internal results, not the hidden final score.

## 1:25–1:55 — Code and model
Show features(), fit_model(), and the candidate-selection block in train.py.
The selected model is {result['selected']['main']['name']}. CatBoost supports
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
"""
    (ROOT / "loom_walkthrough.md").write_text(script)
    print("Created freight_rate_report.docx and loom_walkthrough.md")


if __name__ == "__main__":
    main()