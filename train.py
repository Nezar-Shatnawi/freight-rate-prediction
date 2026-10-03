"""Reproducible, temporally validated freight-rate prediction pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
CATEGORIES = ["pickup", "delivery", "equipment", "lane"]
CORE = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def features(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Row-local transforms only: no target encoding or fitted preprocessing."""
    required = set(CORE) | ({"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                            "market_index"} if mode != "chart" else set())
    if mode == "quote":
        required.add("quote_signal")
    if not required.issubset(frame):
        raise ValueError(f"Missing features: {sorted(required - set(frame))}")
    x = frame[CORE].copy()
    date = pd.to_datetime(x.pop("date"), errors="raise")
    if date.isna().any():
        raise ValueError("Missing dates")
    for col in ["pickup", "delivery", "equipment"]:
        x[col] = x[col].fillna("Unknown").astype(str).str.strip()
    x["lane"] = x.pickup + " -> " + x.delivery
    x["distance"] = pd.to_numeric(x.distance, errors="coerce")
    if x.distance.isna().any() or (x.distance <= 0).any():
        raise ValueError("Distances must be finite and positive")
    weight = pd.to_numeric(x.weight, errors="coerce")
    x["weight_invalid"] = ((weight <= 0) | weight.isna()).astype(int)
    x["weight"] = weight.mask(weight <= 0)
    x["day_index"] = (date - pd.Timestamp("2025-01-01")).dt.days
    x["weekday"] = date.dt.dayofweek
    x["weekend"] = (date.dt.dayofweek >= 5).astype(int)
    x["day_of_month"] = date.dt.day
    x["month"] = date.dt.month
    x["year_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    x["year_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    x["log_distance"] = np.log1p(x.distance)
    x["weight_per_mile"] = x.weight / x.distance
    if mode != "chart":
        for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"]:
            x[col] = pd.to_numeric(frame[col], errors="coerce")
        x["lat_delta"] = x.delivery_lat - x.pickup_lat
        x["lon_delta"] = x.delivery_lon - x.pickup_lon
        # Coordinates are supplied features, not silently replaced by real-world geocoding.
    if mode == "quote":
        x["quote_signal"] = pd.to_numeric(frame.quote_signal, errors="coerce")
    numeric = x.select_dtypes(include="number").columns
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    return x


def metrics(actual, predicted) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE_percent": float(np.mean(np.abs((np.asarray(actual) - predicted) / actual)) * 100),
    }


def make_model(loss: str, iterations: int = 1400) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations, depth=7, learning_rate=0.055, loss_function=loss,
        random_seed=SEED, thread_count=4, l2_leaf_reg=5,
        allow_writing_files=False, verbose=False,
    )


def fit_model(frame, mode, loss, target_kind, valid=None):
    model = make_model(loss)
    x = features(frame, mode)
    y = frame.posted_rate / frame.distance if target_kind == "rpm" else frame.posted_rate
    kwargs = {}
    if valid is not None:
        yv = valid.posted_rate / valid.distance if target_kind == "rpm" else valid.posted_rate
        kwargs = dict(eval_set=(features(valid, mode), yv),
                      early_stopping_rounds=130, use_best_model=True)
    model.fit(x, y, cat_features=CATEGORIES, **kwargs)
    return model


def predict(model, frame, mode, target_kind):
    p = model.predict(features(frame, mode))
    if target_kind == "rpm":
        p *= frame.distance.to_numpy()
    return np.maximum(p, 1.0)


def audit(train, validation) -> dict:
    return {
        "development_rows": len(train), "validation_rows": len(validation),
        "development_dates": [train.date.min(), train.date.max()],
        "validation_dates": [validation.date.min(), validation.date.max()],
        "missing_development": train.isna().sum().to_dict(),
        "missing_validation": validation.isna().sum().to_dict(),
        "nonpositive_weights_development": int((train.weight <= 0).sum()),
        "nonpositive_weights_validation": int((validation.weight <= 0).sum()),
        "duplicate_development_ids": int(train.load_id.duplicated().sum()),
        "duplicate_validation_ids": int(validation.load_id.duplicated().sum()),
        "duplicate_development_features_and_target":
            int(train.drop(columns="load_id").duplicated().sum()),
        "new_validation_pickups": sorted(set(validation.pickup) - set(train.pickup)),
        "new_validation_deliveries": sorted(set(validation.delivery) - set(train.delivery)),
        "rate_per_mile_above_4": int((train.posted_rate / train.distance > 4).sum()),
        "monthly_quote_rpm_correlation": {
            month: float(group.quote_signal.corr(group.posted_rate / group.distance))
            for month, group in train.groupby(train.date.str[:7])
        },
    }


def run(root: Path):
    data = root / "data"
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    train = pd.read_csv(data / "train_test.csv")
    validation = pd.read_csv(data / "validation.csv")
    template = pd.read_csv(data / "validation_predictions_template.csv")
    december = pd.read_csv(data / "december_chart_inputs.csv")
    if train.posted_rate.isna().any() or (train.posted_rate <= 0).any():
        raise ValueError("Target rates must be positive and observed")
    if validation.load_id.duplicated().any() or template.load_id.duplicated().any():
        raise ValueError("Duplicate output IDs")
    if set(template.load_id) != set(validation.load_id):
        raise ValueError("Template and validation IDs differ")
    audit_info = audit(train, validation)
    (out / "audit.json").write_text(json.dumps(audit_info, indent=2))
    # Choose hyperparameters only on August; September-October is a locked test.
    development = train[train.date < "2025-08-01"].copy()
    tuning = train[(train.date >= "2025-08-01") & (train.date < "2025-09-01")].copy()
    test = train[train.date >= "2025-09-01"].copy()
    records, candidates = [], {}
    specs = [
        ("core_rpm_mae", "core", "MAE", "rpm"),
        ("core_rate_rmse", "core", "RMSE", "rate"),
        ("core_rpm_rmse", "core", "RMSE", "rpm"),
        ("quote_rpm_mae", "quote", "MAE", "rpm"),
        ("quote_rate_rmse", "quote", "RMSE", "rate"),
        ("chart_rpm_mae", "chart", "MAE", "rpm"),
        ("chart_rate_rmse", "chart", "RMSE", "rate"),
    ]
    baseline_rpm = float(np.median(development.posted_rate / development.distance))
    records.append(dict(name="median_rpm_baseline", split="August tuning",
                        **metrics(tuning.posted_rate, baseline_rpm * tuning.distance.to_numpy())))
    for name, mode, loss, target_kind in specs:
        model = fit_model(development, mode, loss, target_kind, valid=tuning)
        p = predict(model, tuning, mode, target_kind)
        record = dict(name=name, mode=mode, loss=loss, target_kind=target_kind,
                      split="August tuning", trees=model.tree_count_,
                      **metrics(tuning.posted_rate, p))
        records.append(record)
        candidates[name] = record
        print(json.dumps(record), flush=True)
    # Primary metric is MAE, declared before looking at the locked test.
    main = min((v for v in candidates.values() if v["mode"] != "chart"), key=lambda v: v["MAE"])
    chart = min((v for v in candidates.values() if v["mode"] == "chart"), key=lambda v: v["MAE"])
    chosen = {"main": main, "chart": chart}
    pretest = train[train.date < "2025-09-01"].copy()
    test_predictions = pd.DataFrame({"load_id": test.load_id, "date": test.date,
                                     "posted_rate": test.posted_rate})
    records.append(dict(name="median_rpm_baseline", split="September-October locked test",
                        **metrics(test.posted_rate,
                                  np.median(pretest.posted_rate / pretest.distance)
                                  * test.distance.to_numpy())))
    importance = {}
    for label, spec in chosen.items():
        model = make_model(spec["loss"], iterations=spec["trees"])
        y = pretest.posted_rate / pretest.distance if spec["target_kind"] == "rpm" else pretest.posted_rate
        model.fit(features(pretest, spec["mode"]), y, cat_features=CATEGORIES)
        p = predict(model, test, spec["mode"], spec["target_kind"])
        test_predictions[f"{label}_predicted_rate"] = p
        records.append(dict(name=spec["name"], role=label, split="September-October locked test",
                            **metrics(test.posted_rate, p)))
        for month in ["2025-09", "2025-10"]:
            mask = test.date.str.startswith(month).to_numpy()
            records.append(dict(name=spec["name"], role=label, split=f"{month} locked test",
                                **metrics(test.loc[mask, "posted_rate"], p[mask])))
        seen = test.pickup.isin(pretest.pickup) & test.delivery.isin(pretest.delivery)
        records.append(dict(name=spec["name"], role=label,
                            split="locked test seen cities", rows=int(seen.sum()),
                            **metrics(test.loc[seen, "posted_rate"], p[seen])))
        # Final refit uses all supplied labeled rows, and no final-validation labels.
        model = make_model(spec["loss"], iterations=spec["trees"])
        y = train.posted_rate / train.distance if spec["target_kind"] == "rpm" else train.posted_rate
        model.fit(features(train, spec["mode"]), y, cat_features=CATEGORIES)
        model.save_model(str(root / "models" / f"{label}.cbm"))
        importance[label] = dict(zip(model.feature_names_, model.feature_importances_.tolist()))
        if label == "main":
            rates = predict(model, validation, spec["mode"], spec["target_kind"])
            by_id = pd.Series(rates, index=validation.load_id)
            template["predicted_rate"] = template.load_id.map(by_id)
            if not np.isfinite(template.predicted_rate).all():
                raise ValueError("Unmatched template IDs")
            template[["load_id", "predicted_rate"]].to_csv(root / "validation_predictions.csv",
                                                        index=False, float_format="%.2f")
        else:
            december["predicted_rate"] = predict(model, december, spec["mode"], spec["target_kind"])
            december.to_csv(data / "december_chart_inputs.csv", index=False, float_format="%.2f")
            december.to_csv(root / "december_predictions.csv", index=False, float_format="%.2f")
    test_predictions.to_csv(out / "locked_test_predictions.csv", index=False)
    pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
    payload = {
        "seed": SEED, "selection_metric": "August MAE",
        "split_rows": {"Jan_Jul_training": len(development), "August_tuning": len(tuning),
                       "September_October_locked_test": len(test)},
        "selected": chosen, "metrics": records, "feature_importance": importance,
        "main_final_prediction_summary": template.predicted_rate.describe().to_dict(),
        "december_summary": december.predicted_rate.describe().to_dict(),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().root)