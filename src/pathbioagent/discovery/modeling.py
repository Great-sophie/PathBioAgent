from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

TARGETS = [
    "overall_median_pearson",
    "best_scale_median_pearson",
    "median_scale_range",
]


FEATURES = [
    "rna_mean_expression",
    "rna_median_expression",
    "rna_std_expression",
    "rna_variance_expression",
    "rna_iqr_expression",
    "rna_expression_range_10_90",
    "rna_nonzero_fraction",
    "rna_above1_fraction",
    "rna_between_cancer_variance",
    "rna_between_cancer_range",
    "rna_mean_within_cancer_variance",
    "rna_pooled_within_cancer_variance",
    "rna_cancer_eta_squared",
    "rna_cancer_tau_specificity",
    "rna_cancer_entropy_specificity",
    "rna_dominant_cancer_mean",
    "rna_dominant_cancer_margin",
]


def _results_root(
    results_root: Optional[str | Path] = None,
) -> Path:

    if results_root is not None:
        return Path(results_root)

    env = os.getenv(
        "PATHBIO_RESULTS_ROOT"
    )

    if not env:
        raise RuntimeError(
            "PATHBIO_RESULTS_ROOT is not set."
        )

    return Path(env)


# ============================================================
# Metrics
# ============================================================

def _spearman(
    y_true,
    y_pred,
):
    result = spearmanr(
        y_true,
        y_pred,
        nan_policy="omit",
    )

    return float(
        result.statistic
    )


def _metrics(
    y_true,
    y_pred,
):

    return {
        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),

        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "spearman":
            _spearman(
                y_true,
                y_pred,
            ),
    }


# ============================================================
# Models
# ============================================================

def _build_models():

    dummy = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                DummyRegressor(
                    strategy="mean"
                ),
            ),
        ]
    )

    elastic_net = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[
                        0.05,
                        0.1,
                        0.2,
                        0.5,
                        0.8,
                        0.95,
                        1.0,
                    ],
                    alphas=np.logspace(
                        -4,
                        1,
                        60,
                    ),
                    cv=5,
                    max_iter=50000,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "dummy_mean":
            dummy,

        "elastic_net":
            elastic_net,

        "random_forest":
            random_forest,
    }


# ============================================================
# Cross-validation
# ============================================================

def run_regression_models(
    results_root: Optional[str | Path] = None,
):

    root = _results_root(
        results_root
    )

    table_path = (
        root
        / "pathbioagent_discovery"
        / "gene_discovery_model_table.csv"
    )

    df = pd.read_csv(
        table_path
    )

    missing_features = [
        f
        for f in FEATURES
        if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing modeling features: "
            f"{missing_features}"
        )

    missing_targets = [
        t
        for t in TARGETS
        if t not in df.columns
    ]

    if missing_targets:
        raise ValueError(
            "Missing targets: "
            f"{missing_targets}"
        )

    X = df[
        FEATURES
    ].copy()

    splitter = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )

    models = _build_models()

    fold_rows = []
    prediction_rows = []

    # ========================================================
    # Outer CV
    # ========================================================

    for target in TARGETS:

        y = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        valid = (
            y.notna()
        )

        X_target = (
            X.loc[valid]
            .reset_index(
                drop=True
            )
        )

        y_target = (
            y.loc[valid]
            .reset_index(
                drop=True
            )
        )

        genes = (
            df.loc[
                valid,
                "gene",
            ]
            .reset_index(
                drop=True
            )
        )

        for model_name, estimator in models.items():

            for split_index, (
                train_idx,
                test_idx,
            ) in enumerate(
                splitter.split(
                    X_target
                )
            ):

                repeat_index = (
                    split_index // 5
                )

                fold_index = (
                    split_index % 5
                )

                model = clone(
                    estimator
                )

                model.fit(
                    X_target.iloc[
                        train_idx
                    ],
                    y_target.iloc[
                        train_idx
                    ],
                )

                pred = model.predict(
                    X_target.iloc[
                        test_idx
                    ]
                )

                fold_metrics = _metrics(
                    y_target.iloc[
                        test_idx
                    ],
                    pred,
                )

                fold_rows.append(
                    {
                        "target":
                            target,

                        "model":
                            model_name,

                        "repeat":
                            repeat_index,

                        "fold":
                            fold_index,

                        "n_train":
                            int(
                                len(
                                    train_idx
                                )
                            ),

                        "n_test":
                            int(
                                len(
                                    test_idx
                                )
                            ),

                        **fold_metrics,
                    }
                )

                for local_idx, y_hat in zip(
                    test_idx,
                    pred,
                ):

                    prediction_rows.append(
                        {
                            "target":
                                target,

                            "model":
                                model_name,

                            "repeat":
                                repeat_index,

                            "fold":
                                fold_index,

                            "gene":
                                genes.iloc[
                                    local_idx
                                ],

                            "observed":
                                float(
                                    y_target.iloc[
                                        local_idx
                                    ]
                                ),

                            "predicted":
                                float(
                                    y_hat
                                ),
                        }
                    )

    metrics = pd.DataFrame(
        fold_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    return (
        df,
        metrics,
        predictions,
    )


# ============================================================
# Full-data exploratory feature importance
# ============================================================

def fit_feature_importance(
    df: pd.DataFrame,
):

    rows = []

    X = df[
        FEATURES
    ].copy()

    models = _build_models()

    for target in TARGETS:

        y = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        valid = y.notna()

        X_target = X.loc[
            valid
        ]

        y_target = y.loc[
            valid
        ]

        # ----------------------------------------------------
        # Elastic Net
        # ----------------------------------------------------

        enet = clone(
            models["elastic_net"]
        )

        enet.fit(
            X_target,
            y_target,
        )

        coefficients = (
            enet.named_steps[
                "model"
            ].coef_
        )

        for feature, value in zip(
            FEATURES,
            coefficients,
        ):

            rows.append(
                {
                    "target":
                        target,

                    "model":
                        "elastic_net",

                    "feature":
                        feature,

                    "importance":
                        float(
                            abs(value)
                        ),

                    "signed_effect":
                        float(
                            value
                        ),
                }
            )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        rf = clone(
            models[
                "random_forest"
            ]
        )

        rf.fit(
            X_target,
            y_target,
        )

        importances = (
            rf.named_steps[
                "model"
            ].feature_importances_
        )

        for feature, value in zip(
            FEATURES,
            importances,
        ):

            rows.append(
                {
                    "target":
                        target,

                    "model":
                        "random_forest",

                    "feature":
                        feature,

                    "importance":
                        float(
                            value
                        ),

                    "signed_effect":
                        np.nan,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Univariate associations
# ============================================================

def feature_target_correlations(
    df: pd.DataFrame,
):

    rows = []

    for target in TARGETS:

        y = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        for feature in FEATURES:

            x = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

            valid = (
                x.notna()
                &
                y.notna()
            )

            rho = spearmanr(
                x.loc[valid],
                y.loc[valid],
            )

            rows.append(
                {
                    "target":
                        target,

                    "feature":
                        feature,

                    "spearman_rho":
                        float(
                            rho.statistic
                        ),

                    "p_value":
                        float(
                            rho.pvalue
                        ),

                    "n_genes":
                        int(
                            valid.sum()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Save outputs
# ============================================================

def save_modeling_outputs(
    results_root: Optional[str | Path] = None,
):

    root = _results_root(
        results_root
    )

    output_dir = (
        root
        / "pathbioagent_discovery"
        / "modeling"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        metrics,
        predictions,
    ) = run_regression_models(
        root
    )

    importance = (
        fit_feature_importance(
            df
        )
    )

    correlations = (
        feature_target_correlations(
            df
        )
    )

    summary = (
        metrics.groupby(
            [
                "target",
                "model",
            ],
            as_index=False,
        )
        .agg(
            mean_r2=(
                "r2",
                "mean",
            ),
            std_r2=(
                "r2",
                "std",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_spearman=(
                "spearman",
                "mean",
            ),
            std_spearman=(
                "spearman",
                "std",
            ),
        )
    )

    metrics.to_csv(
        output_dir
        / "cv_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "cv_summary.csv",
        index=False,
    )

    predictions.to_csv(
        output_dir
        / "oof_predictions.csv",
        index=False,
    )

    importance.to_csv(
        output_dir
        / "feature_importance.csv",
        index=False,
    )

    correlations.to_csv(
        output_dir
        / "feature_target_correlations.csv",
        index=False,
    )

    return {
        "summary":
            summary,

        "metrics":
            metrics,

        "predictions":
            predictions,

        "importance":
            importance,

        "correlations":
            correlations,

        "output_dir":
            output_dir,
    }


if __name__ == "__main__":

    results = (
        save_modeling_outputs()
    )

    summary = results[
        "summary"
    ]

    importance = results[
        "importance"
    ]

    correlations = results[
        "correlations"
    ]

    print("=" * 78)
    print(
        "PathBioAgent Molecular Visibility Modeling"
    )
    print("=" * 78)

    print()
    print(
        "Cross-validation summary:"
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    for target in TARGETS:

        print()
        print("-" * 78)

        print(
            f"Target: {target}"
        )

        print("-" * 78)

        print(
            "Top Elastic Net features:"
        )

        top_enet = (
            importance[
                (
                    importance[
                        "target"
                    ]
                    == target
                )
                &
                (
                    importance[
                        "model"
                    ]
                    == "elastic_net"
                )
            ]
            .sort_values(
                "importance",
                ascending=False,
            )
            .head(8)
        )

        print(
            top_enet[
                [
                    "feature",
                    "importance",
                    "signed_effect",
                ]
            ]
            .to_string(
                index=False
            )
        )

        print()

        print(
            "Strongest univariate associations:"
        )

        corr = (
            correlations[
                correlations[
                    "target"
                ]
                == target
            ]
            .assign(
                abs_rho=lambda x:
                    x[
                        "spearman_rho"
                    ].abs()
            )
            .sort_values(
                "abs_rho",
                ascending=False,
            )
            .head(8)
        )

        print(
            corr[
                [
                    "feature",
                    "spearman_rho",
                    "p_value",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()

    print(
        "Saved modeling outputs to:"
    )

    print(
        results[
            "output_dir"
        ]
    )
