
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pathbioagent.discovery.pattern_clustering import (
    RANDOM_STATE,
    build_clustering_matrix,
    build_pattern_features,
    load_gene_table,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_COARSE_K = 3
DEFAULT_FINE_K = 6
DEFAULT_N_INIT = 100

MIN_MEANINGFUL_CHILD_FRACTION = 0.05


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


def _default_output_dir() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_hierarchy"
    )


# ============================================================
# Clustering
# ============================================================

def _fit_kmeans(
    matrix: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> Tuple[
    KMeans,
    np.ndarray,
]:

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(
        matrix
    )

    return (
        model,
        labels,
    )


# ============================================================
# Entropy
# ============================================================

def _normalized_entropy(
    counts: np.ndarray,
) -> float:

    counts = np.asarray(
        counts,
        dtype=float,
    )

    total = counts.sum()

    if total <= 0:
        return np.nan

    probabilities = (
        counts[
            counts > 0
        ]
        / total
    )

    if len(probabilities) <= 1:
        return 0.0

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    max_entropy = np.log(
        len(probabilities)
    )

    if max_entropy <= 0:
        return 0.0

    return float(
        entropy
        / max_entropy
    )


# ============================================================
# Contingency table
# ============================================================

def build_contingency(
    assignments: pd.DataFrame,
    coarse_k: int,
    fine_k: int,
) -> pd.DataFrame:

    table = pd.crosstab(
        assignments[
            "coarse_cluster"
        ],
        assignments[
            "fine_cluster"
        ],
        dropna=False,
    )

    table = table.reindex(
        index=range(
            coarse_k
        ),
        columns=range(
            fine_k
        ),
        fill_value=0,
    )

    table.index.name = (
        "coarse_cluster"
    )

    table.columns = [
        f"fine_cluster_{c}"
        for c in table.columns
    ]

    table[
        "total_genes"
    ] = table.sum(
        axis=1
    )

    return (
        table.reset_index()
    )


# ============================================================
# Transition fractions
# ============================================================

def build_transition_fraction_table(
    assignments: pd.DataFrame,
    coarse_k: int,
    fine_k: int,
) -> pd.DataFrame:

    rows: List[Dict] = []

    coarse_sizes = (
        assignments[
            "coarse_cluster"
        ]
        .value_counts()
        .to_dict()
    )

    fine_sizes = (
        assignments[
            "fine_cluster"
        ]
        .value_counts()
        .to_dict()
    )

    grouped = (
        assignments
        .groupby(
            [
                "coarse_cluster",
                "fine_cluster",
            ]
        )
        .size()
    )

    for coarse_cluster in range(
        coarse_k
    ):

        for fine_cluster in range(
            fine_k
        ):

            n_genes = int(
                grouped.get(
                    (
                        coarse_cluster,
                        fine_cluster,
                    ),
                    0,
                )
            )

            coarse_n = int(
                coarse_sizes.get(
                    coarse_cluster,
                    0,
                )
            )

            fine_n = int(
                fine_sizes.get(
                    fine_cluster,
                    0,
                )
            )

            fraction_within_coarse = (
                n_genes / coarse_n
                if coarse_n > 0
                else np.nan
            )

            fraction_within_fine = (
                n_genes / fine_n
                if fine_n > 0
                else np.nan
            )

            rows.append(
                {
                    "coarse_cluster":
                        coarse_cluster,

                    "fine_cluster":
                        fine_cluster,

                    "n_genes":
                        n_genes,

                    "fraction_within_coarse":
                        float(
                            fraction_within_coarse
                        ),

                    "fraction_within_fine":
                        float(
                            fraction_within_fine
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Coarse hierarchy summary
# ============================================================

def build_hierarchy_summary(
    assignments: pd.DataFrame,
    coarse_k: int,
    fine_k: int,
) -> pd.DataFrame:

    rows: List[Dict] = []

    for coarse_cluster in range(
        coarse_k
    ):

        subset = assignments.loc[
            assignments[
                "coarse_cluster"
            ]
            == coarse_cluster
        ]

        counts = (
            subset[
                "fine_cluster"
            ]
            .value_counts()
            .reindex(
                range(
                    fine_k
                ),
                fill_value=0,
            )
        )

        total = int(
            counts.sum()
        )

        if total == 0:
            continue

        fractions = (
            counts
            / total
        )

        dominant_child = int(
            fractions.idxmax()
        )

        dominant_child_fraction = (
            float(
                fractions.max()
            )
        )

        meaningful_children = (
            fractions.loc[
                fractions
                >=
                MIN_MEANINGFUL_CHILD_FRACTION
            ]
        )

        row = {
            "coarse_cluster":
                coarse_cluster,

            "n_genes":
                total,

            "dominant_fine_cluster":
                dominant_child,

            "dominant_child_fraction":
                dominant_child_fraction,

            "normalized_split_entropy":
                _normalized_entropy(
                    counts.to_numpy()
                ),

            "n_meaningful_children":
                int(
                    len(
                        meaningful_children
                    )
                ),

            "meaningful_children":
                ";".join(
                    str(
                        int(child)
                    )
                    for child
                    in meaningful_children.index
                ),
        }

        for fine_cluster in range(
            fine_k
        ):

            row[
                f"fraction_to_fine_"
                f"{fine_cluster}"
            ] = float(
                fractions.loc[
                    fine_cluster
                ]
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Fine-to-coarse summary
# ============================================================

def build_fine_parent_summary(
    assignments: pd.DataFrame,
    coarse_k: int,
    fine_k: int,
) -> pd.DataFrame:

    rows: List[Dict] = []

    for fine_cluster in range(
        fine_k
    ):

        subset = assignments.loc[
            assignments[
                "fine_cluster"
            ]
            == fine_cluster
        ]

        counts = (
            subset[
                "coarse_cluster"
            ]
            .value_counts()
            .reindex(
                range(
                    coarse_k
                ),
                fill_value=0,
            )
        )

        total = int(
            counts.sum()
        )

        if total == 0:
            continue

        fractions = (
            counts
            / total
        )

        dominant_parent = int(
            fractions.idxmax()
        )

        dominant_parent_fraction = (
            float(
                fractions.max()
            )
        )

        rows.append(
            {
                "fine_cluster":
                    fine_cluster,

                "n_genes":
                    total,

                "dominant_coarse_parent":
                    dominant_parent,

                "dominant_parent_fraction":
                    dominant_parent_fraction,

                "normalized_parent_entropy":
                    _normalized_entropy(
                        counts.to_numpy()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Pattern profile summaries
# ============================================================

def build_level_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    level_name: str,
) -> pd.DataFrame:

    work = feature_df.copy()

    work[
        level_name
    ] = labels

    rows: List[Dict] = []

    for cluster_id, group in (
        work.groupby(
            level_name
        )
    ):

        preferred_counts = (
            group[
                "preferred_scale_clean"
            ]
            .value_counts()
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
            level_name:
                int(
                    cluster_id
                ),

            "n_genes":
                int(
                    len(group)
                ),

            "dominant_preferred_scale":
                dominant_scale,

            "dominant_preferred_scale_fraction":
                dominant_scale_fraction,
        }

        for feature in [
            "visibility_level",
            "shape_10_minus_2p5",
            "shape_40_minus_10",
            "scale_range",
            "vote_consensus_fraction",
            "median_pearson_2p5x",
            "median_pearson_10x",
            "median_pearson_40x",
        ]:

            if feature not in group.columns:
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

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            level_name
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main pipeline
# ============================================================

def run_cluster_hierarchy(
    input_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    coarse_k: int = DEFAULT_COARSE_K,
    fine_k: int = DEFAULT_FINE_K,
    random_state: int = RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> Dict:

    if coarse_k >= fine_k:
        raise ValueError(
            "coarse_k must be smaller "
            "than fine_k."
        )

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

    (
        coarse_model,
        coarse_labels,
    ) = _fit_kmeans(
        matrix=matrix,
        k=coarse_k,
        random_state=random_state,
        n_init=n_init,
    )

    (
        fine_model,
        fine_labels,
    ) = _fit_kmeans(
        matrix=matrix,
        k=fine_k,
        random_state=random_state,
        n_init=n_init,
    )

    assignments = pd.DataFrame(
        {
            "gene":
                df[
                    "gene"
                ].astype(str),

            "coarse_cluster":
                coarse_labels,

            "fine_cluster":
                fine_labels,
        }
    )

    # --------------------------------------------------------
    # Distance to assigned centroid at each resolution
    # --------------------------------------------------------

    coarse_distances = (
        coarse_model.transform(
            matrix
        )
    )

    fine_distances = (
        fine_model.transform(
            matrix
        )
    )

    assignments[
        "coarse_cluster_distance"
    ] = coarse_distances[
        np.arange(
            len(
                coarse_labels
            )
        ),
        coarse_labels,
    ]

    assignments[
        "fine_cluster_distance"
    ] = fine_distances[
        np.arange(
            len(
                fine_labels
            )
        ),
        fine_labels,
    ]

    # --------------------------------------------------------
    # Build hierarchy outputs
    # --------------------------------------------------------

    contingency = (
        build_contingency(
            assignments=
                assignments,
            coarse_k=
                coarse_k,
            fine_k=
                fine_k,
        )
    )

    transition = (
        build_transition_fraction_table(
            assignments=
                assignments,
            coarse_k=
                coarse_k,
            fine_k=
                fine_k,
        )
    )

    hierarchy_summary = (
        build_hierarchy_summary(
            assignments=
                assignments,
            coarse_k=
                coarse_k,
            fine_k=
                fine_k,
        )
    )

    fine_parent_summary = (
        build_fine_parent_summary(
            assignments=
                assignments,
            coarse_k=
                coarse_k,
            fine_k=
                fine_k,
        )
    )

    coarse_profiles = (
        build_level_profiles(
            feature_df=
                feature_df,
            labels=
                coarse_labels,
            level_name=
                "coarse_cluster",
        )
    )

    fine_profiles = (
        build_level_profiles(
            feature_df=
                feature_df,
            labels=
                fine_labels,
            level_name=
                "fine_cluster",
        )
    )

    # --------------------------------------------------------
    # Add useful original features to assignments
    # --------------------------------------------------------

    extra_columns = [
        "median_pearson_2p5x",
        "median_pearson_10x",
        "median_pearson_40x",
        "derived_median_optimum_scale",
        "vote_consensus_fraction",
        "overall_median_pearson",
        "rna_cancer_tau_specificity",
        "rna_cancer_eta_squared",
        "rna_dominant_cancer",
    ]

    for column in extra_columns:

        if column in df.columns:

            assignments[
                column
            ] = df[
                column
            ].to_numpy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    contingency.to_csv(
        out_dir
        / "k3_k6_contingency.csv",
        index=False,
    )

    transition.to_csv(
        out_dir
        / "k3_k6_transition_fraction.csv",
        index=False,
    )

    hierarchy_summary.to_csv(
        out_dir
        / "hierarchy_summary.csv",
        index=False,
    )

    fine_parent_summary.to_csv(
        out_dir
        / "fine_to_coarse_summary.csv",
        index=False,
    )

    assignments.to_csv(
        out_dir
        / "gene_hierarchy_assignments.csv",
        index=False,
    )

    coarse_profiles.to_csv(
        out_dir
        / "coarse_cluster_profiles.csv",
        index=False,
    )

    fine_profiles.to_csv(
        out_dir
        / "fine_cluster_profiles.csv",
        index=False,
    )

    metadata = {
        "coarse_k":
            int(
                coarse_k
            ),

        "fine_k":
            int(
                fine_k
            ),

        "random_state":
            int(
                random_state
            ),

        "n_init":
            int(
                n_init
            ),

        "n_genes":
            int(
                len(
                    df
                )
            ),

        "feature_names":
            feature_names,

        "minimum_meaningful_child_fraction":
            MIN_MEANINGFUL_CHILD_FRACTION,

        "scientific_scope": (
            "The hierarchy summarizes how a stable "
            "coarse clustering resolution maps onto "
            "a finer clustering resolution in the same "
            "cross-scale morphology-derived molecular "
            "visibility feature space. The mapping is "
            "descriptive and does not establish a true "
            "biological hierarchy, lineage, pathway, "
            "cell type, or causal mechanism."
        ),
    }

    (
        out_dir
        / "hierarchy_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    return {
        "assignments":
            assignments,

        "contingency":
            contingency,

        "transition":
            transition,

        "hierarchy_summary":
            hierarchy_summary,

        "fine_parent_summary":
            fine_parent_summary,

        "coarse_profiles":
            coarse_profiles,

        "fine_profiles":
            fine_profiles,

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
            "Map stable coarse and fine resolutions "
            "of cross-scale morphology-derived "
            "molecular visibility patterns."
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
        "--coarse-k",
        type=int,
        default=DEFAULT_COARSE_K,
    )

    parser.add_argument(
        "--fine-k",
        type=int,
        default=DEFAULT_FINE_K,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
    )

    args = parser.parse_args()

    result = run_cluster_hierarchy(
        input_path=
            args.input,
        output_dir=
            args.output_dir,
        coarse_k=
            args.coarse_k,
        fine_k=
            args.fine_k,
        random_state=
            args.random_state,
        n_init=
            args.n_init,
    )

    print(
        "\nK3 -> K6 contingency:"
    )

    print(
        result[
            "contingency"
        ].to_string(
            index=False
        )
    )

    print(
        "\nCoarse-to-fine hierarchy summary:"
    )

    print(
        result[
            "hierarchy_summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nFine-to-coarse parent summary:"
    )

    print(
        result[
            "fine_parent_summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nCoarse cluster profiles:"
    )

    print(
        result[
            "coarse_profiles"
        ].to_string(
            index=False
        )
    )

    print(
        "\nFine cluster profiles:"
    )

    print(
        result[
            "fine_profiles"
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
