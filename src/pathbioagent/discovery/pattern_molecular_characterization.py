
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
)


# ============================================================
# Configuration
# ============================================================

CONTINUOUS_RNA_FEATURES = [
    "rna_mean_expression",
    "rna_nonzero_fraction",
    "rna_above1_fraction",
    "rna_between_cancer_variance",
    "rna_cancer_eta_squared",
    "rna_cancer_tau_specificity",
    "rna_cancer_entropy_specificity",
]

CATEGORICAL_FEATURE = (
    "rna_dominant_cancer"
)

DEFAULT_ALPHA = 0.05


# ============================================================
# Paths
# ============================================================

def _results_root() -> Path:

    return Path(
        os.environ.get(
            "PATHBIO_RESULTS_ROOT",
            "/media/sophie/Data2T/"
            "Paper3_8model_gene_analysis",
        )
    )


def _default_gene_table() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "gene_discovery_model_table.csv"
    )


def _default_assignments() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_hierarchy"
        / "gene_hierarchy_assignments.csv"
    )


def _default_output_dir() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "molecular_characterization"
    )


# ============================================================
# BH correction
# ============================================================

def benjamini_hochberg(
    p_values: List[float],
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    if n == 0:
        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    adjusted = (
        ranked
        * n
        /
        np.arange(
            1,
            n + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    result = np.empty(
        n,
        dtype=float,
    )

    result[
        order
    ] = adjusted

    return result


# ============================================================
# Load and merge
# ============================================================

def load_analysis_table(
    gene_table_path: Optional[
        str | Path
    ] = None,
    assignment_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    gene_path = (
        Path(
            gene_table_path
        )
        if gene_table_path is not None
        else _default_gene_table()
    )

    assignment_path = (
        Path(
            assignment_path
        )
        if assignment_path is not None
        else _default_assignments()
    )

    if not gene_path.exists():

        raise FileNotFoundError(
            f"Gene table not found: "
            f"{gene_path}"
        )

    if not assignment_path.exists():

        raise FileNotFoundError(
            f"Hierarchy assignments not found: "
            f"{assignment_path}"
        )

    genes = pd.read_csv(
        gene_path
    )

    assignments = pd.read_csv(
        assignment_path
    )

    required_assignment = [
        "gene",
        "fine_cluster",
    ]

    missing = [
        column
        for column
        in required_assignment
        if column
        not in assignments.columns
    ]

    if missing:

        raise ValueError(
            f"Missing assignment columns: "
            f"{missing}"
        )

    keep_assignments = (
        assignments[
            required_assignment
        ]
        .drop_duplicates(
            subset=[
                "gene"
            ]
        )
    )

    merged = genes.merge(
        keep_assignments,
        on="gene",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(
        genes
    ):

        print(
            "Warning: merged table contains "
            f"{len(merged)} of "
            f"{len(genes)} genes."
        )

    return merged


# ============================================================
# Cluster-level descriptive summary
# ============================================================

def build_cluster_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    for cluster, group in (
        df.groupby(
            "fine_cluster"
        )
    ):

        row = {
            "fine_cluster":
                int(
                    cluster
                ),

            "n_genes":
                int(
                    len(group)
                ),
        }

        for feature in (
            CONTINUOUS_RNA_FEATURES
        ):

            if feature not in (
                group.columns
            ):
                continue

            values = pd.to_numeric(
                group[
                    feature
                ],
                errors="coerce",
            )

            row[
                f"{feature}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{feature}_median"
            ] = float(
                values.median()
            )

            row[
                f"{feature}_q25"
            ] = float(
                values.quantile(
                    0.25
                )
            )

            row[
                f"{feature}_q75"
            ] = float(
                values.quantile(
                    0.75
                )
            )

        if (
            CATEGORICAL_FEATURE
            in group.columns
        ):

            counts = (
                group[
                    CATEGORICAL_FEATURE
                ]
                .dropna()
                .value_counts()
            )

            if len(
                counts
            ):

                row[
                    "dominant_cancer"
                ] = str(
                    counts.index[
                        0
                    ]
                )

                row[
                    "dominant_cancer_fraction"
                ] = float(
                    counts.iloc[
                        0
                    ]
                    /
                    counts.sum()
                )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "fine_cluster"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Omnibus continuous tests
# ============================================================

def _epsilon_squared(
    h_statistic: float,
    n_total: int,
    n_groups: int,
) -> float:

    denominator = (
        n_total
        - n_groups
    )

    if denominator <= 0:

        return np.nan

    value = (
        h_statistic
        - n_groups
        + 1
    ) / denominator

    return float(
        max(
            0.0,
            value,
        )
    )


def test_continuous_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    clusters = sorted(
        df[
            "fine_cluster"
        ]
        .dropna()
        .unique()
    )

    for feature in (
        CONTINUOUS_RNA_FEATURES
    ):

        if feature not in df.columns:
            continue

        groups = []

        n_total = 0

        for cluster in clusters:

            values = pd.to_numeric(
                df.loc[
                    df[
                        "fine_cluster"
                    ]
                    == cluster,
                    feature,
                ],
                errors="coerce",
            ).dropna()

            groups.append(
                values.to_numpy()
            )

            n_total += len(
                values
            )

        valid_groups = [
            group
            for group
            in groups
            if len(group) > 0
        ]

        if len(
            valid_groups
        ) < 2:

            continue

        h_statistic, p_value = (
            kruskal(
                *valid_groups
            )
        )

        epsilon_squared = (
            _epsilon_squared(
                h_statistic=
                    float(
                        h_statistic
                    ),
                n_total=
                    n_total,
                n_groups=
                    len(
                        valid_groups
                    ),
            )
        )

        rows.append(
            {
                "feature":
                    feature,

                "kruskal_h":
                    float(
                        h_statistic
                    ),

                "p_value":
                    float(
                        p_value
                    ),

                "epsilon_squared":
                    epsilon_squared,

                "n_genes":
                    int(
                        n_total
                    ),

                "n_clusters":
                    int(
                        len(
                            valid_groups
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    result[
        "q_value"
    ] = benjamini_hochberg(
        result[
            "p_value"
        ].tolist()
    )

    result[
        "significant_fdr_0p05"
    ] = (
        result[
            "q_value"
        ]
        <
        DEFAULT_ALPHA
    )

    return result.sort_values(
        [
            "q_value",
            "epsilon_squared",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


# ============================================================
# Pairwise continuous tests
# ============================================================

def _rank_biserial(
    u_statistic: float,
    n_a: int,
    n_b: int,
) -> float:

    denominator = (
        n_a
        * n_b
    )

    if denominator <= 0:

        return np.nan

    return float(
        (
            2.0
            * u_statistic
            /
            denominator
        )
        - 1.0
    )


def pairwise_continuous_tests(
    df: pd.DataFrame,
    omnibus: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:

    significant_features = (
        omnibus.loc[
            omnibus[
                "q_value"
            ]
            < alpha,
            "feature",
        ]
        .tolist()
    )

    rows: List[Dict] = []

    clusters = sorted(
        df[
            "fine_cluster"
        ]
        .dropna()
        .unique()
    )

    for feature in (
        significant_features
    ):

        feature_rows = []

        for cluster_a, cluster_b in (
            combinations(
                clusters,
                2,
            )
        ):

            a = pd.to_numeric(
                df.loc[
                    df[
                        "fine_cluster"
                    ]
                    == cluster_a,
                    feature,
                ],
                errors="coerce",
            ).dropna()

            b = pd.to_numeric(
                df.loc[
                    df[
                        "fine_cluster"
                    ]
                    == cluster_b,
                    feature,
                ],
                errors="coerce",
            ).dropna()

            if (
                len(a) == 0
                or len(b) == 0
            ):
                continue

            result = mannwhitneyu(
                a,
                b,
                alternative="two-sided",
            )

            u_statistic = float(
                result.statistic
            )

            feature_rows.append(
                {
                    "feature":
                        feature,

                    "cluster_a":
                        int(
                            cluster_a
                        ),

                    "cluster_b":
                        int(
                            cluster_b
                        ),

                    "n_a":
                        int(
                            len(a)
                        ),

                    "n_b":
                        int(
                            len(b)
                        ),

                    "median_a":
                        float(
                            a.median()
                        ),

                    "median_b":
                        float(
                            b.median()
                        ),

                    "median_difference_a_minus_b":
                        float(
                            a.median()
                            -
                            b.median()
                        ),

                    "u_statistic":
                        u_statistic,

                    "p_value":
                        float(
                            result.pvalue
                        ),

                    "rank_biserial":
                        _rank_biserial(
                            u_statistic,
                            len(a),
                            len(b),
                        ),
                }
            )

        if feature_rows:

            feature_frame = (
                pd.DataFrame(
                    feature_rows
                )
            )

            feature_frame[
                "q_value_within_feature"
            ] = benjamini_hochberg(
                feature_frame[
                    "p_value"
                ].tolist()
            )

            rows.extend(
                feature_frame.to_dict(
                    orient="records"
                )
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    result[
        "significant_fdr_0p05"
    ] = (
        result[
            "q_value_within_feature"
        ]
        <
        alpha
    )

    return result.sort_values(
        [
            "feature",
            "q_value_within_feature",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# Dominant-cancer association
# ============================================================

def _cramers_v(
    chi2: float,
    n: int,
    n_rows: int,
    n_cols: int,
) -> float:

    denominator = (
        n
        * min(
            n_rows - 1,
            n_cols - 1,
        )
    )

    if denominator <= 0:

        return np.nan

    return float(
        np.sqrt(
            chi2
            / denominator
        )
    )


def analyze_dominant_cancer(
    df: pd.DataFrame,
) -> Dict:

    if (
        CATEGORICAL_FEATURE
        not in df.columns
    ):

        return {
            "test":
                None,

            "counts":
                pd.DataFrame(),

            "fractions":
                pd.DataFrame(),

            "standardized_residuals":
                pd.DataFrame(),
        }

    work = df[
        [
            "fine_cluster",
            CATEGORICAL_FEATURE,
        ]
    ].dropna()

    counts = pd.crosstab(
        work[
            "fine_cluster"
        ],
        work[
            CATEGORICAL_FEATURE
        ],
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(
            counts.to_numpy()
        )
    )

    expected_df = pd.DataFrame(
        expected,
        index=counts.index,
        columns=counts.columns,
    )

    residuals = (
        counts
        - expected_df
    ) / np.sqrt(
        expected_df
    )

    fractions = counts.div(
        counts.sum(
            axis=1
        ),
        axis=0,
    )

    n = int(
        counts.to_numpy().sum()
    )

    cramers_v = _cramers_v(
        chi2=float(
            chi2
        ),
        n=n,
        n_rows=counts.shape[
            0
        ],
        n_cols=counts.shape[
            1
        ],
    )

    test = {
        "chi2":
            float(
                chi2
            ),

        "p_value":
            float(
                p_value
            ),

        "degrees_of_freedom":
            int(
                dof
            ),

        "cramers_v":
            cramers_v,

        "n_genes":
            n,
    }

    return {
        "test":
            test,

        "counts":
            counts,

        "fractions":
            fractions,

        "standardized_residuals":
            residuals,
    }


# ============================================================
# Main analysis
# ============================================================

def run_molecular_characterization(
    gene_table_path: Optional[
        str | Path
    ] = None,
    assignment_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    alpha: float = DEFAULT_ALPHA,
) -> Dict:

    df = load_analysis_table(
        gene_table_path=
            gene_table_path,
        assignment_path=
            assignment_path,
    )

    cluster_summary = (
        build_cluster_summary(
            df
        )
    )

    omnibus = (
        test_continuous_features(
            df
        )
    )

    pairwise = (
        pairwise_continuous_tests(
            df=df,
            omnibus=omnibus,
            alpha=alpha,
        )
    )

    cancer = (
        analyze_dominant_cancer(
            df
        )
    )

    out_dir = (
        Path(
            output_dir
        )
        if output_dir is not None
        else _default_output_dir()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir
        / "molecular_characterization_table.csv",
        index=False,
    )

    cluster_summary.to_csv(
        out_dir
        / "cluster_rna_summary.csv",
        index=False,
    )

    omnibus.to_csv(
        out_dir
        / "continuous_feature_tests.csv",
        index=False,
    )

    pairwise.to_csv(
        out_dir
        / "pairwise_feature_tests.csv",
        index=False,
    )

    cancer[
        "counts"
    ].to_csv(
        out_dir
        / "dominant_cancer_counts.csv"
    )

    cancer[
        "fractions"
    ].to_csv(
        out_dir
        / "dominant_cancer_fractions.csv"
    )

    cancer[
        "standardized_residuals"
    ].to_csv(
        out_dir
        / "dominant_cancer_standardized_residuals.csv"
    )

    metadata = {
        "n_genes":
            int(
                len(df)
            ),

        "n_fine_clusters":
            int(
                df[
                    "fine_cluster"
                ].nunique()
            ),

        "continuous_rna_features":
            CONTINUOUS_RNA_FEATURES,

        "categorical_feature":
            CATEGORICAL_FEATURE,

        "alpha":
            float(
                alpha
            ),

        "dominant_cancer_test":
            cancer[
                "test"
            ],

        "scientific_scope": (
            "This analysis characterizes statistical "
            "associations between descriptive "
            "morphology-derived molecular visibility "
            "clusters and RNA-derived gene properties "
            "within the same primary 63-case cohort. "
            "It does not establish biological mechanism, "
            "causality, clinical biomarker validity, "
            "or external generalization."
        ),
    }

    (
        out_dir
        / "molecular_characterization_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    return {
        "analysis_table":
            df,

        "cluster_summary":
            cluster_summary,

        "continuous_tests":
            omnibus,

        "pairwise_tests":
            pairwise,

        "dominant_cancer":
            cancer,

        "output_dir":
            str(
                out_dir
            ),
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Characterize RNA-derived properties "
            "of cross-scale morphology-derived "
            "molecular visibility patterns."
        )
    )

    parser.add_argument(
        "--gene-table",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--assignments",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    args = parser.parse_args()

    result = run_molecular_characterization(
        gene_table_path=
            args.gene_table,
        assignment_path=
            args.assignments,
        output_dir=
            args.output_dir,
        alpha=
            args.alpha,
    )

    print(
        "\nContinuous RNA feature tests:"
    )

    print(
        result[
            "continuous_tests"
        ].to_string(
            index=False
        )
    )

    print(
        "\nCluster RNA summary:"
    )

    print(
        result[
            "cluster_summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nDominant cancer association:"
    )

    print(
        result[
            "dominant_cancer"
        ][
            "test"
        ]
    )

    print(
        "\nOutput directory:"
    )

    print(
        result[
            "output_dir"
        ]
    )


if __name__ == "__main__":
    main()
