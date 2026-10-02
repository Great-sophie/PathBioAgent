
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from pathbioagent.discovery.pattern_clustering import (
    RANDOM_STATE,
    build_clustering_matrix,
    build_pattern_features,
    load_gene_table,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_K_VALUES = [
    3,
    4,
    5,
    6,
]

DEFAULT_N_REPEATS = 30

DEFAULT_SUBSAMPLE_FRACTION = 0.80

DEFAULT_N_INIT = 20


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


def _default_output_dir() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_stability"
    )


# ============================================================
# KMeans helper
# ============================================================

def _fit_kmeans(
    matrix: np.ndarray,
    k: int,
    seed: int,
    n_init: int = DEFAULT_N_INIT,
) -> Tuple[
    KMeans,
    np.ndarray,
]:

    model = KMeans(
        n_clusters=k,
        random_state=seed,
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
# Cluster matching
# ============================================================

def _match_centroids(
    reference_centroids: np.ndarray,
    candidate_centroids: np.ndarray,
) -> Tuple[
    Dict[int, int],
    np.ndarray,
]:

    """
    Match candidate cluster labels to reference
    cluster labels using minimum centroid distance.

    Returns:
        mapping:
            candidate_label -> reference_label

        matched_distances:
            Euclidean centroid distance after matching
            in reference cluster order.
    """

    if (
        reference_centroids.shape
        != candidate_centroids.shape
    ):
        raise ValueError(
            "Reference and candidate centroid "
            "matrices must have the same shape."
        )

    diff = (
        reference_centroids[:, None, :]
        -
        candidate_centroids[None, :, :]
    )

    cost = np.sqrt(
        np.sum(
            diff ** 2,
            axis=2,
        )
    )

    reference_idx, candidate_idx = (
        linear_sum_assignment(
            cost
        )
    )

    mapping = {
        int(candidate):
            int(reference)
        for reference, candidate
        in zip(
            reference_idx,
            candidate_idx,
        )
    }

    matched_distances = np.full(
        reference_centroids.shape[0],
        np.nan,
        dtype=float,
    )

    for reference, candidate in zip(
        reference_idx,
        candidate_idx,
    ):

        matched_distances[
            reference
        ] = cost[
            reference,
            candidate,
        ]

    return (
        mapping,
        matched_distances,
    )


def _map_labels(
    labels: np.ndarray,
    mapping: Dict[int, int],
) -> np.ndarray:

    return np.asarray(
        [
            mapping[
                int(label)
            ]
            for label in labels
        ],
        dtype=int,
    )


# ============================================================
# Cluster proportion diagnostics
# ============================================================

def _cluster_proportions(
    labels: np.ndarray,
    k: int,
) -> np.ndarray:

    counts = np.bincount(
        labels,
        minlength=k,
    ).astype(float)

    total = counts.sum()

    if total <= 0:
        return np.zeros(
            k,
            dtype=float,
        )

    return (
        counts
        / total
    )


def _cluster_proportion_difference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    k: int,
) -> Tuple[
    float,
    float,
]:

    reference = _cluster_proportions(
        reference_labels,
        k,
    )

    candidate = _cluster_proportions(
        candidate_labels,
        k,
    )

    difference = np.abs(
        reference
        - candidate
    )

    return (
        float(
            difference.mean()
        ),
        float(
            difference.max()
        ),
    )


# ============================================================
# Full-data seed stability
# ============================================================

def evaluate_seed_stability(
    matrix: np.ndarray,
    k: int,
    reference_model: KMeans,
    reference_labels: np.ndarray,
    n_repeats: int,
    base_seed: int,
    n_init: int,
) -> List[Dict]:

    rows: List[Dict] = []

    reference_centroids = (
        reference_model.cluster_centers_
    )

    for repeat in range(
        n_repeats
    ):

        seed = (
            base_seed
            + 1000
            + repeat
        )

        model, labels = _fit_kmeans(
            matrix=matrix,
            k=k,
            seed=seed,
            n_init=n_init,
        )

        mapping, centroid_distances = (
            _match_centroids(
                reference_centroids,
                model.cluster_centers_,
            )
        )

        mapped_labels = _map_labels(
            labels,
            mapping,
        )

        ari = adjusted_rand_score(
            reference_labels,
            labels,
        )

        nmi = (
            normalized_mutual_info_score(
                reference_labels,
                labels,
            )
        )

        agreement = float(
            np.mean(
                reference_labels
                == mapped_labels
            )
        )

        (
            mean_size_diff,
            max_size_diff,
        ) = _cluster_proportion_difference(
            reference_labels,
            mapped_labels,
            k,
        )

        rows.append(
            {
                "k":
                    k,

                "stability_type":
                    "seed",

                "repeat":
                    repeat,

                "seed":
                    seed,

                "sample_fraction":
                    1.0,

                "n_genes":
                    int(
                        matrix.shape[0]
                    ),

                "ari":
                    float(ari),

                "nmi":
                    float(nmi),

                "matched_label_agreement":
                    agreement,

                "mean_centroid_distance":
                    float(
                        np.mean(
                            centroid_distances
                        )
                    ),

                "max_centroid_distance":
                    float(
                        np.max(
                            centroid_distances
                        )
                    ),

                "mean_cluster_fraction_difference":
                    mean_size_diff,

                "max_cluster_fraction_difference":
                    max_size_diff,
            }
        )

    return rows


# ============================================================
# Subsampling stability
# ============================================================

def evaluate_subsample_stability(
    matrix: np.ndarray,
    k: int,
    reference_model: KMeans,
    reference_labels: np.ndarray,
    n_repeats: int,
    subsample_fraction: float,
    base_seed: int,
    n_init: int,
) -> List[Dict]:

    if not (
        0.0
        < subsample_fraction
        < 1.0
    ):
        raise ValueError(
            "subsample_fraction must be "
            "between 0 and 1."
        )

    rows: List[Dict] = []

    n_genes = matrix.shape[0]

    sample_size = int(
        round(
            n_genes
            * subsample_fraction
        )
    )

    if sample_size <= k:
        raise ValueError(
            "Subsample is too small for "
            f"k={k}."
        )

    reference_centroids = (
        reference_model.cluster_centers_
    )

    rng = np.random.default_rng(
        base_seed
        + 10000
        + k
    )

    for repeat in range(
        n_repeats
    ):

        indices = rng.choice(
            n_genes,
            size=sample_size,
            replace=False,
        )

        indices = np.sort(
            indices
        )

        sample_matrix = matrix[
            indices
        ]

        reference_sample_labels = (
            reference_labels[
                indices
            ]
        )

        seed = (
            base_seed
            + 20000
            + repeat
        )

        model, labels = _fit_kmeans(
            matrix=sample_matrix,
            k=k,
            seed=seed,
            n_init=n_init,
        )

        mapping, centroid_distances = (
            _match_centroids(
                reference_centroids,
                model.cluster_centers_,
            )
        )

        mapped_labels = _map_labels(
            labels,
            mapping,
        )

        # ARI and NMI are label-permutation invariant,
        # so raw candidate labels are valid here.
        ari = adjusted_rand_score(
            reference_sample_labels,
            labels,
        )

        nmi = (
            normalized_mutual_info_score(
                reference_sample_labels,
                labels,
            )
        )

        agreement = float(
            np.mean(
                reference_sample_labels
                == mapped_labels
            )
        )

        (
            mean_size_diff,
            max_size_diff,
        ) = _cluster_proportion_difference(
            reference_sample_labels,
            mapped_labels,
            k,
        )

        rows.append(
            {
                "k":
                    k,

                "stability_type":
                    "subsample",

                "repeat":
                    repeat,

                "seed":
                    seed,

                "sample_fraction":
                    subsample_fraction,

                "n_genes":
                    int(
                        sample_size
                    ),

                "ari":
                    float(ari),

                "nmi":
                    float(nmi),

                "matched_label_agreement":
                    agreement,

                "mean_centroid_distance":
                    float(
                        np.mean(
                            centroid_distances
                        )
                    ),

                "max_centroid_distance":
                    float(
                        np.max(
                            centroid_distances
                        )
                    ),

                "mean_cluster_fraction_difference":
                    mean_size_diff,

                "max_cluster_fraction_difference":
                    max_size_diff,
            }
        )

    return rows


# ============================================================
# Summary statistics
# ============================================================

def _summarize_metric(
    values: pd.Series,
    prefix: str,
) -> Dict:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:

        return {
            f"{prefix}_mean":
                np.nan,

            f"{prefix}_std":
                np.nan,

            f"{prefix}_median":
                np.nan,

            f"{prefix}_q10":
                np.nan,

            f"{prefix}_q90":
                np.nan,
        }

    return {
        f"{prefix}_mean":
            float(
                values.mean()
            ),

        f"{prefix}_std":
            float(
                values.std(
                    ddof=1
                )
            ),

        f"{prefix}_median":
            float(
                values.median()
            ),

        f"{prefix}_q10":
            float(
                values.quantile(
                    0.10
                )
            ),

        f"{prefix}_q90":
            float(
                values.quantile(
                    0.90
                )
            ),
    }


def build_stability_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:

    rows: List[Dict] = []

    metrics = [
        "ari",
        "nmi",
        "matched_label_agreement",
        "mean_centroid_distance",
        "max_centroid_distance",
        "mean_cluster_fraction_difference",
        "max_cluster_fraction_difference",
    ]

    for (
        k,
        stability_type,
    ), group in runs.groupby(
        [
            "k",
            "stability_type",
        ]
    ):

        row = {
            "k":
                int(k),

            "stability_type":
                stability_type,

            "n_repeats":
                int(
                    len(group)
                ),

            "sample_fraction":
                float(
                    group[
                        "sample_fraction"
                    ].iloc[0]
                ),
        }

        for metric in metrics:

            row.update(
                _summarize_metric(
                    group[
                        metric
                    ],
                    metric,
                )
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "k",
                "stability_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Per-cluster stability
# ============================================================

def build_cluster_stability_profiles(
    matrix: np.ndarray,
    k_values: Sequence[int],
    n_repeats: int,
    subsample_fraction: float,
    base_seed: int,
    n_init: int,
) -> pd.DataFrame:

    """
    Summarize how reproducible each reference centroid
    is under subsampling.
    """

    rows: List[Dict] = []

    n_genes = matrix.shape[0]

    for k in k_values:

        (
            reference_model,
            reference_labels,
        ) = _fit_kmeans(
            matrix=matrix,
            k=k,
            seed=base_seed,
            n_init=max(
                n_init,
                50,
            ),
        )

        reference_centroids = (
            reference_model.cluster_centers_
        )

        reference_fractions = (
            _cluster_proportions(
                reference_labels,
                k,
            )
        )

        cluster_distances = {
            cluster_id: []
            for cluster_id in range(k)
        }

        cluster_fraction_diffs = {
            cluster_id: []
            for cluster_id in range(k)
        }

        rng = np.random.default_rng(
            base_seed
            + 30000
            + k
        )

        sample_size = int(
            round(
                n_genes
                * subsample_fraction
            )
        )

        for repeat in range(
            n_repeats
        ):

            indices = rng.choice(
                n_genes,
                size=sample_size,
                replace=False,
            )

            sample_matrix = matrix[
                indices
            ]

            seed = (
                base_seed
                + 40000
                + repeat
            )

            model, labels = _fit_kmeans(
                matrix=sample_matrix,
                k=k,
                seed=seed,
                n_init=n_init,
            )

            (
                mapping,
                matched_distances,
            ) = _match_centroids(
                reference_centroids,
                model.cluster_centers_,
            )

            mapped_labels = (
                _map_labels(
                    labels,
                    mapping,
                )
            )

            sample_fractions = (
                _cluster_proportions(
                    mapped_labels,
                    k,
                )
            )

            for cluster_id in range(
                k
            ):

                cluster_distances[
                    cluster_id
                ].append(
                    matched_distances[
                        cluster_id
                    ]
                )

                cluster_fraction_diffs[
                    cluster_id
                ].append(
                    abs(
                        sample_fractions[
                            cluster_id
                        ]
                        -
                        reference_fractions[
                            cluster_id
                        ]
                    )
                )

        for cluster_id in range(
            k
        ):

            distances = np.asarray(
                cluster_distances[
                    cluster_id
                ],
                dtype=float,
            )

            fraction_diffs = np.asarray(
                cluster_fraction_diffs[
                    cluster_id
                ],
                dtype=float,
            )

            rows.append(
                {
                    "k":
                        int(k),

                    "cluster":
                        int(
                            cluster_id
                        ),

                    "reference_cluster_fraction":
                        float(
                            reference_fractions[
                                cluster_id
                            ]
                        ),

                    "centroid_distance_mean":
                        float(
                            distances.mean()
                        ),

                    "centroid_distance_std":
                        float(
                            distances.std(
                                ddof=1
                            )
                        ),

                    "centroid_distance_median":
                        float(
                            np.median(
                                distances
                            )
                        ),

                    "centroid_distance_q90":
                        float(
                            np.quantile(
                                distances,
                                0.90,
                            )
                        ),

                    "cluster_fraction_difference_mean":
                        float(
                            fraction_diffs.mean()
                        ),

                    "cluster_fraction_difference_q90":
                        float(
                            np.quantile(
                                fraction_diffs,
                                0.90,
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Stability ranking
# ============================================================

def build_k_stability_ranking(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    """
    Create a descriptive ranking based only on
    subsampling stability.

    Higher ARI/NMI/agreement is better.
    Lower centroid distance and cluster-size
    difference are better.

    This is a heuristic, not evidence that one
    cluster number is biologically correct.
    """

    subsample = summary.loc[
        summary[
            "stability_type"
        ]
        == "subsample"
    ].copy()

    if subsample.empty:
        return pd.DataFrame()

    subsample[
        "rank_ari"
    ] = subsample[
        "ari_median"
    ].rank(
        ascending=False,
        method="min",
    )

    subsample[
        "rank_nmi"
    ] = subsample[
        "nmi_median"
    ].rank(
        ascending=False,
        method="min",
    )

    subsample[
        "rank_agreement"
    ] = subsample[
        "matched_label_agreement_median"
    ].rank(
        ascending=False,
        method="min",
    )

    subsample[
        "rank_centroid"
    ] = subsample[
        "mean_centroid_distance_median"
    ].rank(
        ascending=True,
        method="min",
    )

    subsample[
        "rank_cluster_size"
    ] = subsample[
        "mean_cluster_fraction_difference_median"
    ].rank(
        ascending=True,
        method="min",
    )

    rank_columns = [
        "rank_ari",
        "rank_nmi",
        "rank_agreement",
        "rank_centroid",
        "rank_cluster_size",
    ]

    subsample[
        "mean_stability_rank"
    ] = subsample[
        rank_columns
    ].mean(
        axis=1
    )

    subsample[
        "recommended_by_stability"
    ] = False

    best_idx = subsample[
        "mean_stability_rank"
    ].idxmin()

    subsample.loc[
        best_idx,
        "recommended_by_stability"
    ] = True

    columns = [
        "k",
        "ari_median",
        "nmi_median",
        "matched_label_agreement_median",
        "mean_centroid_distance_median",
        "mean_cluster_fraction_difference_median",
        *rank_columns,
        "mean_stability_rank",
        "recommended_by_stability",
    ]

    return (
        subsample[
            columns
        ]
        .sort_values(
            "mean_stability_rank"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main pipeline
# ============================================================

def run_cluster_stability(
    input_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    n_repeats: int = DEFAULT_N_REPEATS,
    subsample_fraction: float = DEFAULT_SUBSAMPLE_FRACTION,
    base_seed: int = RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
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

    run_rows: List[Dict] = []

    for k in k_values:

        if (
            k < 2
            or k >= matrix.shape[0]
        ):
            raise ValueError(
                f"Invalid k={k}"
            )

        (
            reference_model,
            reference_labels,
        ) = _fit_kmeans(
            matrix=matrix,
            k=k,
            seed=base_seed,
            n_init=max(
                n_init,
                50,
            ),
        )

        seed_rows = (
            evaluate_seed_stability(
                matrix=matrix,
                k=k,
                reference_model=
                    reference_model,
                reference_labels=
                    reference_labels,
                n_repeats=
                    n_repeats,
                base_seed=
                    base_seed,
                n_init=
                    n_init,
            )
        )

        subsample_rows = (
            evaluate_subsample_stability(
                matrix=matrix,
                k=k,
                reference_model=
                    reference_model,
                reference_labels=
                    reference_labels,
                n_repeats=
                    n_repeats,
                subsample_fraction=
                    subsample_fraction,
                base_seed=
                    base_seed,
                n_init=
                    n_init,
            )
        )

        run_rows.extend(
            seed_rows
        )

        run_rows.extend(
            subsample_rows
        )

    runs = pd.DataFrame(
        run_rows
    )

    summary = (
        build_stability_summary(
            runs
        )
    )

    cluster_profiles = (
        build_cluster_stability_profiles(
            matrix=matrix,
            k_values=k_values,
            n_repeats=n_repeats,
            subsample_fraction=
                subsample_fraction,
            base_seed=base_seed,
            n_init=n_init,
        )
    )

    ranking = (
        build_k_stability_ranking(
            summary
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

    runs_path = (
        out_dir
        / "stability_runs.csv"
    )

    summary_path = (
        out_dir
        / "stability_summary.csv"
    )

    cluster_profiles_path = (
        out_dir
        / "cluster_stability_profiles.csv"
    )

    ranking_path = (
        out_dir
        / "k_stability_ranking.csv"
    )

    metadata_path = (
        out_dir
        / "stability_metadata.json"
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    cluster_profiles.to_csv(
        cluster_profiles_path,
        index=False,
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    metadata = {
        "k_values":
            [
                int(k)
                for k in k_values
            ],

        "n_repeats":
            int(
                n_repeats
            ),

        "subsample_fraction":
            float(
                subsample_fraction
            ),

        "base_seed":
            int(
                base_seed
            ),

        "n_init":
            int(
                n_init
            ),

        "n_genes":
            int(
                matrix.shape[0]
            ),

        "feature_names":
            feature_names,

        "scientific_scope": (
            "Stability analysis evaluates reproducibility "
            "of descriptive cross-scale morphology-derived "
            "molecular visibility clusters under KMeans "
            "initialization and gene subsampling. High "
            "stability does not establish that clusters "
            "represent discrete biological classes or "
            "mechanisms."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    return {
        "runs":
            runs,

        "summary":
            summary,

        "cluster_profiles":
            cluster_profiles,

        "ranking":
            ranking,

        "output_dir":
            str(
                out_dir
            ),
    }


# ============================================================
# CLI helpers
# ============================================================

def _parse_k_values(
    value: str,
) -> List[int]:

    parts = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    if not parts:
        raise ValueError(
            "No k values supplied."
        )

    return [
        int(item)
        for item in parts
    ]


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stability of cross-scale "
            "morphology-derived molecular "
            "visibility clusters."
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
        "--k-values",
        type=str,
        default="3,4,5,6",
        help=(
            "Comma-separated cluster numbers."
        ),
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
    )

    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=DEFAULT_SUBSAMPLE_FRACTION,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=RANDOM_STATE,
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT,
    )

    args = parser.parse_args()

    k_values = _parse_k_values(
        args.k_values
    )

    result = run_cluster_stability(
        input_path=args.input,
        output_dir=args.output_dir,
        k_values=k_values,
        n_repeats=args.n_repeats,
        subsample_fraction=
            args.subsample_fraction,
        base_seed=args.base_seed,
        n_init=args.n_init,
    )

    print(
        "\nStability summary:"
    )

    print(
        result[
            "summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nK stability ranking:"
    )

    print(
        result[
            "ranking"
        ].to_string(
            index=False
        )
    )

    print(
        "\nPer-cluster stability:"
    )

    print(
        result[
            "cluster_profiles"
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
