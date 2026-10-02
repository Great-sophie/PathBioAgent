
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

CONTINUOUS_FEATURES = [
    "visibility_level",
    "shape_10_minus_2p5",
    "shape_40_minus_10",
    "scale_range",
    "vote_consensus_fraction",
]

PREFERRED_SCALE_LEVELS = [
    "2p5x",
    "10x",
    "40x",
]

# Preferred-scale category should contribute,
# but should not dominate continuous morphology behavior.
PREFERRED_SCALE_WEIGHT = 0.35


# ============================================================
# Paths
# ============================================================

def _results_root() -> Path:

    return Path(
        os.environ.get(
            "PATHBIO_RESULTS_ROOT",
            "/media/sophie/Data2T/Paper3_8model_gene_analysis",
        )
    )


def _default_input_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "gene_discovery_model_table.csv"
    )


def _default_output_dir() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "clustering"
    )


# ============================================================
# Data loading
# ============================================================

def load_gene_table(
    input_path: Optional[str | Path] = None,
) -> pd.DataFrame:

    path = (
        Path(input_path)
        if input_path is not None
        else _default_input_path()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Gene discovery table not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "gene",
        "median_pearson_2p5x",
        "median_pearson_10x",
        "median_pearson_40x",
        "derived_median_optimum_scale",
        "vote_consensus_fraction",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# Morphology feature construction
# ============================================================

def build_pattern_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    v25 = pd.to_numeric(
        out["median_pearson_2p5x"],
        errors="coerce",
    )

    v10 = pd.to_numeric(
        out["median_pearson_10x"],
        errors="coerce",
    )

    v40 = pd.to_numeric(
        out["median_pearson_40x"],
        errors="coerce",
    )

    out["visibility_level"] = (
        v25
        + v10
        + v40
    ) / 3.0

    out["shape_10_minus_2p5"] = (
        v10
        - v25
    )

    out["shape_40_minus_10"] = (
        v40
        - v10
    )

    out["scale_range"] = (
        pd.concat(
            [
                v25,
                v10,
                v40,
            ],
            axis=1,
        ).max(axis=1)
        -
        pd.concat(
            [
                v25,
                v10,
                v40,
            ],
            axis=1,
        ).min(axis=1)
    )

    out[
        "vote_consensus_fraction"
    ] = pd.to_numeric(
        out["vote_consensus_fraction"],
        errors="coerce",
    )

    preferred = (
        out[
            "derived_median_optimum_scale"
        ]
        .astype(str)
        .replace(
            {
                "2.5x": "2p5x",
                "2.5X": "2p5x",
            }
        )
    )

    out[
        "preferred_scale_clean"
    ] = preferred

    return out


# ============================================================
# Clustering matrix
# ============================================================

def build_clustering_matrix(
    feature_df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    List[str],
    RobustScaler,
]:

    continuous = (
        feature_df[
            CONTINUOUS_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .copy()
    )

    # Defensive median imputation.
    continuous = continuous.fillna(
        continuous.median(
            numeric_only=True
        )
    )

    scaler = RobustScaler()

    continuous_scaled = (
        scaler.fit_transform(
            continuous
        )
    )

    encoded_scale = []

    encoded_names = []

    preferred = feature_df[
        "preferred_scale_clean"
    ].astype(str)

    for scale in PREFERRED_SCALE_LEVELS:

        values = (
            preferred
            == scale
        ).astype(float)

        encoded_scale.append(
            values.to_numpy()
            * PREFERRED_SCALE_WEIGHT
        )

        encoded_names.append(
            f"preferred_scale_{scale}"
        )

    if encoded_scale:

        encoded_matrix = np.column_stack(
            encoded_scale
        )

        matrix = np.column_stack(
            [
                continuous_scaled,
                encoded_matrix,
            ]
        )

    else:

        matrix = continuous_scaled

    feature_names = (
        CONTINUOUS_FEATURES
        + encoded_names
    )

    return (
        matrix,
        feature_names,
        scaler,
    )


# ============================================================
# K diagnostics
# ============================================================

def evaluate_cluster_numbers(
    matrix: np.ndarray,
    k_min: int = 3,
    k_max: int = 8,
) -> pd.DataFrame:

    if k_min < 2:
        raise ValueError(
            "k_min must be >= 2"
        )

    if k_max < k_min:
        raise ValueError(
            "k_max must be >= k_min"
        )

    rows: List[Dict] = []

    n_samples = matrix.shape[0]

    for k in range(
        k_min,
        k_max + 1,
    ):

        if k >= n_samples:
            break

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        )

        labels = model.fit_predict(
            matrix
        )

        silhouette = silhouette_score(
            matrix,
            labels,
        )

        calinski = (
            calinski_harabasz_score(
                matrix,
                labels,
            )
        )

        davies = (
            davies_bouldin_score(
                matrix,
                labels,
            )
        )

        cluster_sizes = np.bincount(
            labels,
            minlength=k,
        )

        rows.append(
            {
                "k": k,
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "minimum_cluster_size":
                    int(
                        cluster_sizes.min()
                    ),
                "maximum_cluster_size":
                    int(
                        cluster_sizes.max()
                    ),
                "smallest_cluster_fraction":
                    float(
                        cluster_sizes.min()
                        / n_samples
                    ),
                "largest_cluster_fraction":
                    float(
                        cluster_sizes.max()
                        / n_samples
                    ),
            }
        )

    diagnostics = pd.DataFrame(
        rows
    )

    if diagnostics.empty:
        raise RuntimeError(
            "No valid clustering diagnostics "
            "could be calculated."
        )

    # --------------------------------------------------------
    # Rank-based diagnostic summary.
    #
    # Lower mean rank is better.
    # This is only a reproducible selection heuristic,
    # not proof that one k is biologically correct.
    # --------------------------------------------------------

    diagnostics[
        "rank_silhouette"
    ] = diagnostics[
        "silhouette"
    ].rank(
        ascending=False,
        method="min",
    )

    diagnostics[
        "rank_calinski_harabasz"
    ] = diagnostics[
        "calinski_harabasz"
    ].rank(
        ascending=False,
        method="min",
    )

    diagnostics[
        "rank_davies_bouldin"
    ] = diagnostics[
        "davies_bouldin"
    ].rank(
        ascending=True,
        method="min",
    )

    diagnostics[
        "mean_metric_rank"
    ] = diagnostics[
        [
            "rank_silhouette",
            "rank_calinski_harabasz",
            "rank_davies_bouldin",
        ]
    ].mean(
        axis=1
    )

    diagnostics[
        "recommended_by_rank"
    ] = False

    best_idx = (
        diagnostics[
            "mean_metric_rank"
        ]
        .idxmin()
    )

    diagnostics.loc[
        best_idx,
        "recommended_by_rank"
    ] = True

    return diagnostics


def choose_k(
    diagnostics: pd.DataFrame,
) -> int:

    selected = diagnostics.loc[
        diagnostics[
            "recommended_by_rank"
        ]
    ]

    if selected.empty:
        raise RuntimeError(
            "No recommended cluster number found."
        )

    return int(
        selected.iloc[0]["k"]
    )


# ============================================================
# Final clustering
# ============================================================

def fit_final_clustering(
    matrix: np.ndarray,
    k: int,
) -> Tuple[
    KMeans,
    np.ndarray,
]:

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=100,
    )

    labels = model.fit_predict(
        matrix
    )

    return (
        model,
        labels,
    )


# ============================================================
# Assignment diagnostics
# ============================================================

def calculate_assignment_distances(
    matrix: np.ndarray,
    model: KMeans,
    labels: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    distances = model.transform(
        matrix
    )

    own_distance = distances[
        np.arange(
            len(labels)
        ),
        labels,
    ]

    sorted_distances = np.sort(
        distances,
        axis=1,
    )

    nearest = (
        sorted_distances[:, 0]
    )

    second = (
        sorted_distances[:, 1]
    )

    separation_margin = (
        second
        - nearest
    )

    return (
        own_distance,
        second,
        separation_margin,
    )


# ============================================================
# Cluster profile summaries
# ============================================================

def build_cluster_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:

    work = feature_df.copy()

    work[
        "cluster"
    ] = labels

    rows: List[Dict] = []

    n_total = len(
        work
    )

    for cluster_id, group in work.groupby(
        "cluster"
    ):

        preferred_counts = (
            group[
                "preferred_scale_clean"
            ]
            .value_counts(
                dropna=False
            )
        )

        dominant_scale = (
            str(
                preferred_counts.index[0]
            )
            if len(
                preferred_counts
            )
            else "unknown"
        )

        dominant_scale_fraction = (
            float(
                preferred_counts.iloc[0]
                / len(group)
            )
            if len(group)
            else np.nan
        )

        row = {
            "cluster":
                int(cluster_id),

            "n_genes":
                int(len(group)),

            "fraction_of_genes":
                float(
                    len(group)
                    / n_total
                ),

            "dominant_preferred_scale":
                dominant_scale,

            "dominant_preferred_scale_fraction":
                dominant_scale_fraction,

            "fraction_prefer_2p5x":
                float(
                    (
                        group[
                            "preferred_scale_clean"
                        ]
                        == "2p5x"
                    ).mean()
                ),

            "fraction_prefer_10x":
                float(
                    (
                        group[
                            "preferred_scale_clean"
                        ]
                        == "10x"
                    ).mean()
                ),

            "fraction_prefer_40x":
                float(
                    (
                        group[
                            "preferred_scale_clean"
                        ]
                        == "40x"
                    ).mean()
                ),
        }

        for feature in CONTINUOUS_FEATURES:

            values = pd.to_numeric(
                group[feature],
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

        for scale_column in [
            "median_pearson_2p5x",
            "median_pearson_10x",
            "median_pearson_40x",
        ]:

            values = pd.to_numeric(
                group[
                    scale_column
                ],
                errors="coerce",
            )

            row[
                f"{scale_column}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{scale_column}_median"
            ] = float(
                values.median()
            )

        rows.append(
            row
        )

    profiles = pd.DataFrame(
        rows
    )

    return profiles.sort_values(
        "cluster"
    ).reset_index(
        drop=True
    )


# ============================================================
# Feature-space cluster summaries
# ============================================================

def build_cluster_feature_summary(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:

    rows: List[Dict] = []

    for cluster_id in sorted(
        np.unique(
            labels
        )
    ):

        mask = (
            labels
            == cluster_id
        )

        subset = matrix[
            mask
        ]

        row = {
            "cluster":
                int(cluster_id),

            "n_genes":
                int(
                    mask.sum()
                ),
        }

        for i, feature in enumerate(
            feature_names
        ):

            row[
                f"{feature}_mean"
            ] = float(
                subset[:, i].mean()
            )

            row[
                f"{feature}_median"
            ] = float(
                np.median(
                    subset[:, i]
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main pipeline
# ============================================================

def run_pattern_clustering(
    input_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    k: Optional[int] = None,
    k_min: int = 3,
    k_max: int = 8,
) -> Dict:

    df = load_gene_table(
        input_path=input_path
    )

    feature_df = (
        build_pattern_features(
            df
        )
    )

    (
        matrix,
        feature_names,
        _,
    ) = build_clustering_matrix(
        feature_df
    )

    diagnostics = (
        evaluate_cluster_numbers(
            matrix,
            k_min=k_min,
            k_max=k_max,
        )
    )

    if k is None:

        selected_k = choose_k(
            diagnostics
        )

        selection_source = (
            "rank_based_diagnostic_heuristic"
        )

    else:

        selected_k = int(
            k
        )

        selection_source = (
            "user_specified"
        )

        if (
            selected_k < 2
            or selected_k >= len(df)
        ):
            raise ValueError(
                f"Invalid k={selected_k}"
            )

    (
        model,
        labels,
    ) = fit_final_clustering(
        matrix,
        selected_k,
    )

    (
        own_distance,
        second_distance,
        separation_margin,
    ) = calculate_assignment_distances(
        matrix,
        model,
        labels,
    )

    assignments = (
        feature_df.copy()
    )

    assignments[
        "cluster"
    ] = labels

    assignments[
        "cluster_distance"
    ] = own_distance

    assignments[
        "second_cluster_distance"
    ] = second_distance

    assignments[
        "cluster_separation_margin"
    ] = separation_margin

    assignment_columns = [
        "gene",
        "cluster",
        "cluster_distance",
        "second_cluster_distance",
        "cluster_separation_margin",

        "visibility_level",
        "shape_10_minus_2p5",
        "shape_40_minus_10",
        "scale_range",
        "vote_consensus_fraction",

        "median_pearson_2p5x",
        "median_pearson_10x",
        "median_pearson_40x",

        "derived_median_optimum_scale",
        "preferred_scale_clean",

        "overall_median_pearson",
        "high_visibility_gene",
        "strong_scale_dependence",
        "strong_model_consensus",

        "rna_cancer_tau_specificity",
        "rna_cancer_eta_squared",
        "rna_dominant_cancer",
    ]

    assignment_columns = [
        c
        for c in assignment_columns
        if c in assignments.columns
    ]

    assignments = assignments[
        assignment_columns
    ].sort_values(
        [
            "cluster",
            "cluster_distance",
        ],
        ascending=[
            True,
            True,
        ],
    )

    profiles = (
        build_cluster_profiles(
            feature_df,
            labels,
        )
    )

    feature_summary = (
        build_cluster_feature_summary(
            matrix,
            labels,
            feature_names,
        )
    )

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path = (
        out_dir
        / "cluster_diagnostics.csv"
    )

    assignments_path = (
        out_dir
        / "gene_cluster_assignments.csv"
    )

    profiles_path = (
        out_dir
        / "cluster_profiles.csv"
    )

    feature_summary_path = (
        out_dir
        / "cluster_feature_summary.csv"
    )

    metadata_path = (
        out_dir
        / "clustering_metadata.json"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    assignments.to_csv(
        assignments_path,
        index=False,
    )

    profiles.to_csv(
        profiles_path,
        index=False,
    )

    feature_summary.to_csv(
        feature_summary_path,
        index=False,
    )

    metadata = {
        "selected_k":
            selected_k,

        "selection_source":
            selection_source,

        "k_min":
            k_min,

        "k_max":
            k_max,

        "n_genes":
            int(
                len(df)
            ),

        "continuous_features":
            CONTINUOUS_FEATURES,

        "preferred_scale_levels":
            PREFERRED_SCALE_LEVELS,

        "preferred_scale_weight":
            PREFERRED_SCALE_WEIGHT,

        "random_state":
            RANDOM_STATE,

        "scientific_scope": (
            "Clusters summarize recurring cross-scale "
            "morphology-derived molecular visibility "
            "patterns. They are descriptive structure "
            "in this cohort and do not establish "
            "biological mechanism, pathway membership, "
            "cell type, biomarker validity, or external "
            "generalizability."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    return {
        "selected_k":
            selected_k,

        "selection_source":
            selection_source,

        "diagnostics":
            diagnostics,

        "assignments":
            assignments,

        "profiles":
            profiles,

        "feature_summary":
            feature_summary,

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
            "Cluster recurring cross-scale "
            "morphology-derived molecular "
            "visibility patterns."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Optional fixed number of clusters. "
            "If omitted, k is selected from diagnostics."
        ),
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    result = run_pattern_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
    )

    print(
        "\nCluster diagnostics:"
    )

    print(
        result[
            "diagnostics"
        ].to_string(
            index=False
        )
    )

    print(
        "\nSelected k:",
        result[
            "selected_k"
        ],
    )

    print(
        "Selection:",
        result[
            "selection_source"
        ],
    )

    print(
        "\nCluster profiles:"
    )

    print(
        result[
            "profiles"
        ].to_string(
            index=False
        )
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
