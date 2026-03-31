"""
Home Credit Scoring Pipeline — Training Script

Usage:
    python src/train_credit.py

Reads from:  data/credit.duckdb (credit_features table built by dbt)
Outputs:     models/lgbm_credit_v1.pkl
             MLflow experiment: credit_scoring
"""

import logging
import pickle
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "data" / "credit.duckdb"
MODEL_DIR  = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_credit_v1.pkl"

MODEL_DIR.mkdir(exist_ok=True)

# --- Feature config ----------------------------------------------------------
# NOTE: Unlike fraud pipeline (TimeSeriesSplit), credit uses StratifiedKFold
# because applications are snapshots, not time-series events.
# The application date is not reliable enough to enforce temporal ordering.
# This is documented intentionally — showing domain-appropriate CV choice.

NUMERIC_FEATURES = [
    "amt_credit", "amt_annuity", "amt_income_total", "amt_goods_price",
    "credit_income_ratio", "annuity_income_ratio", "goods_price_credit_ratio",
    "gender_enc", "owns_car", "owns_realty",
    "cnt_children", "cnt_fam_members", "age_years",
    "is_employed", "days_employed_clean", "employment_age_ratio",
    "days_registration", "days_id_publish",
    "region_rating_client", "reg_city_not_work_city",
    "ext_source_1", "ext_source_2", "ext_source_3", "ext_source_avg",
    # Bureau features
    "bureau_record_count", "active_bureau_count", "closed_bureau_count",
    "days_since_last_bureau", "avg_days_credit",
    "total_days_overdue", "overdue_bureau_count", "max_days_overdue",
    "total_bureau_credit", "total_bureau_debt", "bureau_debt_ratio",
    "total_credit_prolongations", "distinct_credit_types",
    "bureau_debt_to_new_credit",
    # Payment features
    "total_instalments", "avg_payment_lag", "worst_payment_lag",
    "payment_lag_volatility", "late_payment_rate",
    "total_underpayments", "underpayment_rate",
    "avg_payment_ratio", "min_payment_ratio", "overall_payment_completeness",
    "prev_loan_count", "prev_approved_count", "prev_approval_rate",
    "avg_prev_credit_ratio",
]

CATEGORICAL_FEATURES = [
    "name_contract_type",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "organization_type",
    "worst_ever_bureau_status",
]

TARGET = "target"


def load_features(db_path: Path) -> pd.DataFrame:
    log.info(f"Loading features from {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM credit_features").df()
    con.close()
    log.info(f"Loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> tuple:
    encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("MISSING").astype(str))
        encoders[col] = le
    return df, encoders


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    avail_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df, encoders = encode_categoricals(df, avail_cat)
    feature_cols = avail_num + avail_cat
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    log.info(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"Default rate: {y.mean():.2%}")
    return X, y, encoders


def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    default_rate = y.mean()
    scale_pos_weight = (1 - default_rate) / default_rate

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # StratifiedKFold — appropriate for credit (snapshot data, not time-series)
    # See TECH_STACK.md for rationale vs TimeSeriesSplit used in fraud pipeline
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log.info("Starting 5-fold stratified cross-validation...")
    auc_scores, ap_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        ap  = average_precision_score(y_val, proba)
        auc_scores.append(auc)
        ap_scores.append(ap)
        log.info(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}")

    mean_auc = np.mean(auc_scores)
    mean_ap  = np.mean(ap_scores)
    log.info(f"CV Mean AUC: {mean_auc:.4f} (+/- {np.std(auc_scores):.4f})")
    log.info(f"CV Mean AP:  {mean_ap:.4f}")

    # Final fit on all data
    log.info("Fitting final model on full dataset...")
    model.set_params(n_estimators=model.best_iteration_ or 500)
    model.fit(X, y)

    return model, mean_auc, mean_ap, auc_scores


def evaluate_holdout(model, X, y):
    """Random 20% holdout — stratified."""
    from sklearn.model_selection import train_test_split
    _, X_hold, _, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    proba = model.predict_proba(X_hold)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_hold, proba)
    ap  = average_precision_score(y_hold, proba)
    log.info(f"Holdout AUC: {auc:.4f}  AP: {ap:.4f}")
    log.info(
        "\nClassification Report:\n"
        + classification_report(y_hold, preds, target_names=["Repaid", "Default"])
    )
    return auc, ap, proba, y_hold


def main():
    mlflow.set_experiment("credit_scoring")

    with mlflow.start_run(run_name="lgbm_baseline"):
        df = load_features(DB_PATH)
        X, y, encoders = build_feature_matrix(df)

        model, cv_auc, cv_ap, fold_aucs = train(X, y)
        holdout_auc, holdout_ap, proba, y_hold = evaluate_holdout(model, X, y)

        # Feature importance
        fi = pd.Series(model.feature_importances_, index=X.columns)
        fi = fi.sort_values(ascending=False)
        log.info("\nTop 15 feature importances:")
        log.info(fi.head(15).to_string())

        # MLflow logging
        mlflow.log_params({
            "n_estimators":      model.n_estimators,
            "learning_rate":     model.learning_rate,
            "num_leaves":        model.num_leaves,
            "scale_pos_weight":  round(model.scale_pos_weight, 2),
            "n_features":        X.shape[1],
            "cv_strategy":       "StratifiedKFold-5",
        })
        mlflow.log_metrics({
            "cv_mean_auc":   round(cv_auc, 4),
            "cv_mean_ap":    round(cv_ap, 4),
            "holdout_auc":   round(holdout_auc, 4),
            "holdout_ap":    round(holdout_ap, 4),
        })
        for i, auc in enumerate(fold_aucs, 1):
            mlflow.log_metric(f"fold_{i}_auc", round(auc, 4))

        mlflow.lightgbm.log_model(model, name="lgbm_model")

        artifact = {"model": model, "feature_cols": X.columns.tolist(), "encoders": encoders}
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(str(MODEL_PATH))

        log.info(f"Model saved to {MODEL_PATH}")
        log.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
