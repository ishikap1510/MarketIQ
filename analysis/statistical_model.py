"""
statistical_model.py — Statistical and ML modelling functions.

All functions return structured result dicts so the notebook can display
summaries without re-running models.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import (
    silhouette_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42


# ── OLS regression ────────────────────────────────────────────────────────────

def run_ols(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a simple OLS regression and return the fitted results object.

    Args:
        df: DataFrame containing both columns.
        x_col: Name of the predictor column.
        y_col: Name of the outcome column.

    Returns:
        Fitted statsmodels OLS RegressionResultsWrapper.
    """
    X = sm.add_constant(df[x_col].astype(float))
    y = df[y_col].astype(float)
    model = sm.OLS(y, X).fit()
    return model


# ── Pearson correlation ───────────────────────────────────────────────────────

def run_pearson(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns.

    Args:
        df: DataFrame containing both columns.
        col1: First variable name.
        col2: Second variable name (the outcome to correlate against).

    Returns:
        dict with keys: col1, col2, r, p_value, significance, n.
    """
    r, p = stats.pearsonr(df[col1].astype(float), df[col2].astype(float))

    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    return {
        "col1": col1,
        "col2": col2,
        "r": round(r, 3),
        "p_value": round(p, 4),
        "significance": sig,
        "n": len(df),
    }


# ── K-Means clustering ────────────────────────────────────────────────────────

def run_kmeans(
    X_scaled: np.ndarray,
    k_range: range,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Fit K-Means for each K in k_range and collect inertia and silhouette scores.

    Args:
        X_scaled: Pre-scaled feature matrix (n_samples × n_features).
        k_range: Range of K values to evaluate (e.g. range(2, 8)).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (inertias list, silhouette_scores list, list of fitted KMeans objects).
    """
    inertias: list[float] = []
    silhouettes: list[float] = []
    models: list[KMeans] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
        models.append(km)

    return inertias, silhouettes, models


# ── sklearn Linear Regression ────────────────────────────────────────────────

def run_linear_regression(
    df: pd.DataFrame,
    features: list,
    target: str,
    cv: int = 5,
) -> dict:
    """Fit a multiple linear regression with cross-validated performance metrics.

    Features are standardised internally before fitting.

    Args:
        df: DataFrame containing all feature and target columns.
        features: List of predictor column names.
        target: Name of the outcome column.
        cv: Number of cross-validation folds (default 5).

    Returns:
        dict with keys: r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std,
                        coefficients (Series), y_pred (ndarray), scaler, model.
    """
    X = df[features].astype(float).values
    y = df[target].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()

    cv_r2   = cross_val_score(lr, X_scaled, y, cv=cv, scoring="r2")
    cv_rmse = cross_val_score(lr, X_scaled, y, cv=cv, scoring="neg_root_mean_squared_error")
    cv_mae  = cross_val_score(lr, X_scaled, y, cv=cv, scoring="neg_mean_absolute_error")

    lr.fit(X_scaled, y)
    y_pred = lr.predict(X_scaled)

    return {
        "r2_mean":      cv_r2.mean(),
        "r2_std":       cv_r2.std(),
        "rmse_mean":    abs(cv_rmse.mean()),
        "rmse_std":     cv_rmse.std(),
        "mae_mean":     abs(cv_mae.mean()),
        "mae_std":      cv_mae.std(),
        "coefficients": pd.Series(lr.coef_, index=features),
        "y_pred":       y_pred,
        "scaler":       scaler,
        "model":        lr,
    }


# ── Decision Tree classifier ──────────────────────────────────────────────────

def run_decision_tree(
    df: pd.DataFrame,
    features: list,
    target: str,
    max_depth: int = 3,
) -> dict:
    """Fit a Decision Tree classifier with Leave-One-Out cross-validation.

    LOO-CV is used because the dataset is small (n=10), ensuring every sample
    is used as a test point exactly once.

    Args:
        df: DataFrame containing all feature and target columns.
        features: List of predictor column names.
        target: Name of the binary outcome column (0/1).
        max_depth: Maximum tree depth to prevent overfitting (default 3).

    Returns:
        dict with keys: accuracy, precision, recall, y_pred_cv, importances,
                        model (fitted on full data), feature_names.
    """
    X = df[features].astype(float)
    y = df[target].astype(int)

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(dt, X, y, cv=loo)

    dt.fit(X, y)

    return {
        "accuracy":      accuracy_score(y, y_pred_cv),
        "precision":     precision_score(y, y_pred_cv, zero_division=0),
        "recall":        recall_score(y, y_pred_cv, zero_division=0),
        "y_pred_cv":     y_pred_cv,
        "y_true":        y.values,
        "importances":   pd.Series(dt.feature_importances_, index=features),
        "model":         dt,
        "feature_names": features,
    }


# ── Summary printer ───────────────────────────────────────────────────────────

def print_model_summary(results: dict) -> None:
    """Print a clean formatted summary of any model result dict.

    Handles Linear Regression, Decision Tree, and KMeans result dicts
    by inspecting available keys.

    Args:
        results: Result dict returned by run_linear_regression(),
                 run_decision_tree(), or run_kmeans().

    Returns:
        None — output is printed to stdout.
    """
    width = 50
    print(f"\n{'─' * width}")

    if "r2_mean" in results:
        print("  sklearn Linear Regression Summary")
        print(f"{'─' * width}")
        print(f"  R² (CV mean)   : {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
        print(f"  RMSE (CV mean) : ${results['rmse_mean']:,.0f} ± ${results['rmse_std']:,.0f}")
        print(f"  MAE  (CV mean) : ${results['mae_mean']:,.0f} ± ${results['mae_std']:,.0f}")
        print()
        print("  Feature Coefficients:")
        for feat, coef in results["coefficients"].items():
            print(f"    {feat:<28} {coef:+.2f}")

    elif "accuracy" in results:
        top_feat = results["importances"].idxmax()
        print("  Decision Tree Classifier Summary")
        print(f"{'─' * width}")
        print(f"  Accuracy   : {results['accuracy']:.2%}")
        print(f"  Precision  : {results['precision']:.2f}")
        print(f"  Recall     : {results['recall']:.2f}")
        print(f"  CV method  : Leave-One-Out (n={len(results['y_true'])})")
        print(f"  Top feature: {top_feat}")

    else:
        print("  (Unknown result dict — no known keys to summarise)")

    print(f"{'─' * width}\n")
