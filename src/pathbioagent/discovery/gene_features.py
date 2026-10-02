from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _results_root(
    results_root: Optional[str | Path] = None,
) -> Path:
    if results_root is not None:
        return Path(results_root)

    env = os.getenv("PATHBIO_RESULTS_ROOT")

    if not env:
        raise RuntimeError(
            "PATHBIO_RESULTS_ROOT is not set."
        )

    return Path(env)


def _tau_specificity(values: np.ndarray) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return np.nan

    maximum = values.max()

    if maximum <= 0:
        return 0.0

    return float(
        np.sum(
            1.0 - values / maximum
        )
        /
        (len(values) - 1)
    )


def _entropy_specificity(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    total = values.sum()

    if total <= 0:
        return 0.0

    p = values / total

    p = p[
        p > 0
    ]

    if len(p) <= 1:
        return 1.0

    entropy = -np.sum(
        p * np.log(p)
    )

    max_entropy = np.log(
        len(values)
    )

    if max_entropy <= 0:
        return 0.0

    return float(
        1.0
        -
        entropy / max_entropy
    )


def build_rna_gene_features(
    results_root: Optional[str | Path] = None,
) -> pd.DataFrame:

    root = _results_root(
        results_root
    )

    rna = pd.read_csv(
        root
        / "rna"
        / "primary63_log2_tpm.csv"
    )

    meta = pd.read_csv(
        root
        / "rna"
        / "primary63_rna_metadata.csv"
    )

    master = pd.read_csv(
        root
        / "pathbioagent_discovery"
        / "gene_discovery_master.csv"
    )

    # ========================================================
    # Validation
    # ========================================================

    if "case_id" not in rna.columns:
        raise ValueError(
            "RNA matrix does not contain case_id."
        )

    if not {
        "case_id",
        "cancer",
    }.issubset(meta.columns):
        raise ValueError(
            "RNA metadata must contain "
            "case_id and cancer."
        )

    if rna["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case_id in RNA matrix."
        )

    if meta["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case_id in RNA metadata."
        )

    data = rna.merge(
        meta[
            [
                "case_id",
                "cancer",
            ]
        ],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(rna):
        raise ValueError(
            "Not all RNA cases matched metadata."
        )

    target_genes = (
        master["gene"]
        .astype(str)
        .tolist()
    )

    available_genes = [
        gene
        for gene in target_genes
        if gene in data.columns
    ]

    missing_genes = [
        gene
        for gene in target_genes
        if gene not in data.columns
    ]

    # ========================================================
    # Feature extraction
    # ========================================================

    rows = []

    cancer_groups = {
        cancer: group
        for cancer, group
        in data.groupby(
            "cancer",
            sort=True,
        )
    }

    for gene in available_genes:

        x = pd.to_numeric(
            data[gene],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite = x[
            np.isfinite(x)
        ]

        if len(finite) == 0:
            continue

        q10 = np.nanquantile(
            finite,
            0.10,
        )

        q25 = np.nanquantile(
            finite,
            0.25,
        )

        q75 = np.nanquantile(
            finite,
            0.75,
        )

        q90 = np.nanquantile(
            finite,
            0.90,
        )

        cancer_means = []
        cancer_variances = []
        cancer_ns = []
        cancer_labels = []

        for cancer, group in cancer_groups.items():

            values = pd.to_numeric(
                group[gene],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                continue

            cancer_labels.append(
                cancer
            )

            cancer_ns.append(
                len(values)
            )

            cancer_means.append(
                float(
                    np.mean(values)
                )
            )

            if len(values) > 1:
                cancer_variances.append(
                    float(
                        np.var(
                            values,
                            ddof=1,
                        )
                    )
                )
            else:
                cancer_variances.append(
                    np.nan
                )

        cancer_means_arr = np.asarray(
            cancer_means,
            dtype=float,
        )

        cancer_variances_arr = np.asarray(
            cancer_variances,
            dtype=float,
        )

        cancer_ns_arr = np.asarray(
            cancer_ns,
            dtype=float,
        )

        # ----------------------------------------------------
        # Pooled within-cancer variance
        # ----------------------------------------------------

        valid_var = (
            np.isfinite(
                cancer_variances_arr
            )
            &
            (cancer_ns_arr > 1)
        )

        denominator = np.sum(
            cancer_ns_arr[
                valid_var
            ]
            - 1.0
        )

        if denominator > 0:

            pooled_within_variance = (
                np.sum(
                    (
                        cancer_ns_arr[
                            valid_var
                        ]
                        - 1.0
                    )
                    *
                    cancer_variances_arr[
                        valid_var
                    ]
                )
                /
                denominator
            )

        else:

            pooled_within_variance = (
                np.nan
            )

        # ----------------------------------------------------
        # Cancer-associated variance fraction
        # ----------------------------------------------------

        grand_mean = np.nanmean(
            finite
        )

        ss_total = np.nansum(
            (
                finite
                -
                grand_mean
            )
            ** 2
        )

        ss_between = 0.0

        for n, mean_value in zip(
            cancer_ns_arr,
            cancer_means_arr,
        ):
            ss_between += (
                n
                *
                (
                    mean_value
                    -
                    grand_mean
                )
                ** 2
            )

        cancer_eta_squared = (
            float(
                ss_between
                /
                ss_total
            )
            if ss_total > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Dominant cancer
        # ----------------------------------------------------

        if len(cancer_means_arr):

            order = np.argsort(
                cancer_means_arr
            )[::-1]

            best_idx = order[0]

            dominant_cancer = (
                cancer_labels[
                    best_idx
                ]
            )

            dominant_cancer_mean = (
                cancer_means_arr[
                    best_idx
                ]
            )

            if len(order) > 1:
                second_cancer_mean = (
                    cancer_means_arr[
                        order[1]
                    ]
                )
            else:
                second_cancer_mean = (
                    np.nan
                )

        else:

            dominant_cancer = None
            dominant_cancer_mean = np.nan
            second_cancer_mean = np.nan

        rows.append(
            {
                "gene":
                    gene,

                "rna_mean_expression":
                    float(
                        np.nanmean(finite)
                    ),

                "rna_median_expression":
                    float(
                        np.nanmedian(finite)
                    ),

                "rna_std_expression":
                    float(
                        np.nanstd(
                            finite,
                            ddof=1,
                        )
                    ),

                "rna_variance_expression":
                    float(
                        np.nanvar(
                            finite,
                            ddof=1,
                        )
                    ),

                "rna_iqr_expression":
                    float(
                        q75 - q25
                    ),

                "rna_q10_expression":
                    float(q10),

                "rna_q90_expression":
                    float(q90),

                "rna_expression_range_10_90":
                    float(
                        q90 - q10
                    ),

                "rna_nonzero_fraction":
                    float(
                        np.mean(
                            finite > 0
                        )
                    ),

                "rna_above1_fraction":
                    float(
                        np.mean(
                            finite > 1
                        )
                    ),

                "rna_between_cancer_variance":
                    float(
                        np.nanvar(
                            cancer_means_arr,
                            ddof=1,
                        )
                    ),

                "rna_between_cancer_range":
                    float(
                        np.nanmax(
                            cancer_means_arr
                        )
                        -
                        np.nanmin(
                            cancer_means_arr
                        )
                    ),

                "rna_mean_within_cancer_variance":
                    float(
                        np.nanmean(
                            cancer_variances_arr
                        )
                    ),

                "rna_pooled_within_cancer_variance":
                    float(
                        pooled_within_variance
                    ),

                "rna_cancer_eta_squared":
                    cancer_eta_squared,

                "rna_cancer_tau_specificity":
                    _tau_specificity(
                        cancer_means_arr
                    ),

                "rna_cancer_entropy_specificity":
                    _entropy_specificity(
                        cancer_means_arr
                    ),

                "rna_dominant_cancer":
                    dominant_cancer,

                "rna_dominant_cancer_mean":
                    float(
                        dominant_cancer_mean
                    ),

                "rna_dominant_cancer_margin":
                    float(
                        dominant_cancer_mean
                        -
                        second_cancer_mean
                    )
                    if np.isfinite(
                        second_cancer_mean
                    )
                    else np.nan,

                "rna_n_cases":
                    int(
                        len(finite)
                    ),

                "rna_n_cancers":
                    int(
                        len(
                            cancer_means_arr
                        )
                    ),
            }
        )

    features = pd.DataFrame(
        rows
    )

    features.attrs[
        "missing_genes"
    ] = missing_genes

    return features


def save_rna_gene_features(
    results_root: Optional[str | Path] = None,
):

    root = _results_root(
        results_root
    )

    output_dir = (
        root
        / "pathbioagent_discovery"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = build_rna_gene_features(
        root
    )

    feature_path = (
        output_dir
        / "gene_rna_features.csv"
    )

    features.to_csv(
        feature_path,
        index=False,
    )

    master = pd.read_csv(
        output_dir
        / "gene_discovery_master.csv"
    )

    model_table = master.merge(
        features,
        on="gene",
        how="inner",
        validate="one_to_one",
    )

    model_path = (
        output_dir
        / "gene_discovery_model_table.csv"
    )

    model_table.to_csv(
        model_path,
        index=False,
    )

    return (
        feature_path,
        model_path,
        features,
        model_table,
    )


if __name__ == "__main__":

    (
        feature_path,
        model_path,
        features,
        model_table,
    ) = save_rna_gene_features()

    print("=" * 72)
    print("PathBioAgent RNA Gene Features")
    print("=" * 72)

    print(
        f"Saved features: {feature_path}"
    )

    print(
        f"Saved model table: {model_path}"
    )

    print(
        f"RNA feature table shape: "
        f"{features.shape}"
    )

    print(
        f"Model table shape: "
        f"{model_table.shape}"
    )

    print(
        "Cases per gene:",
        int(
            features[
                "rna_n_cases"
            ].median()
        ),
    )

    print(
        "Cancers per gene:",
        int(
            features[
                "rna_n_cancers"
            ].median()
        ),
    )

    print()

    print(
        "Example RNA-derived features:"
    )

    print(
        model_table[
            [
                "gene",
                "best_scale_median_pearson",
                "rna_mean_expression",
                "rna_variance_expression",
                "rna_pooled_within_cancer_variance",
                "rna_between_cancer_variance",
                "rna_cancer_eta_squared",
                "rna_cancer_tau_specificity",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )
