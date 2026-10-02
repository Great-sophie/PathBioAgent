from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SCALES = ("2p5x", "10x", "40x")


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


def _load_inputs(
    results_root: Optional[str | Path] = None,
):
    root = _results_root(results_root)

    base = (
        root
        / "gene_visibility"
        / "top5000"
    )

    raw = pd.read_csv(
        base / "top5000_gene_visibility.csv"
    )

    cross_model = pd.read_csv(
        base
        / "top5000_cross_model_gene_summary.csv"
    )

    by_model = pd.read_csv(
        base
        / "top5000_preferred_scale_by_model_gene.csv"
    )

    return root, raw, cross_model, by_model


def _validate_inputs(
    raw: pd.DataFrame,
    cross_model: pd.DataFrame,
    by_model: pd.DataFrame,
):
    required_raw = {
        "model",
        "scale",
        "gene",
        "within_cancer_variance",
        "pearson",
        "spearman",
    }

    required_cross = {
        "gene",
        "10x",
        "2p5x",
        "40x",
        "consensus_preferred_scale",
        "consensus_best_median_pearson",
        "n_models_prefer_40x",
        "n_models_prefer_10x",
        "n_models_prefer_2p5x",
    }

    required_model = {
        "model",
        "gene",
        "10x",
        "2p5x",
        "40x",
        "preferred_scale",
        "best_pearson",
        "second_best_pearson",
        "scale_margin",
    }

    for name, df, required in [
        ("raw", raw, required_raw),
        ("cross_model", cross_model, required_cross),
        ("by_model", by_model, required_model),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing)}"
            )

    genes_raw = set(raw["gene"])
    genes_cross = set(cross_model["gene"])
    genes_model = set(by_model["gene"])

    if not (
        genes_raw
        ==
        genes_cross
        ==
        genes_model
    ):
        raise ValueError(
            "Gene sets are inconsistent across inputs."
        )


def _median_optimum_scale(
    row: pd.Series,
) -> str:
    values = {
        scale: row[scale]
        for scale in SCALES
    }

    return max(
        values,
        key=values.get,
    )


def _second_best_value(
    values,
):
    clean = [
        float(x)
        for x in values
        if pd.notna(x)
    ]

    if len(clean) < 2:
        return np.nan

    return sorted(
        clean,
        reverse=True,
    )[1]


def build_gene_discovery_table(
    results_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build one auditable discovery row per gene.

    Important:
    - median-optimum scale and model-vote preference are
      preserved as separate concepts.
    - no biological interpretation is inferred here.
    """

    root, raw, cross_model, by_model = (
        _load_inputs(results_root)
    )

    _validate_inputs(
        raw,
        cross_model,
        by_model,
    )

    # ========================================================
    # Start with existing cross-model scale medians
    # ========================================================

    master = cross_model.copy()

    master = master.rename(
        columns={
            "2p5x": "median_pearson_2p5x",
            "10x": "median_pearson_10x",
            "40x": "median_pearson_40x",
            "consensus_preferred_scale":
                "reported_median_optimum_scale",
            "consensus_best_median_pearson":
                "best_scale_median_pearson",
        }
    )

    # ========================================================
    # Deterministically reproduce median-optimum scale
    # ========================================================

    scale_cols = {
        "2p5x": "median_pearson_2p5x",
        "10x": "median_pearson_10x",
        "40x": "median_pearson_40x",
    }

    def derive_optimum(row):
        values = {
            scale: row[column]
            for scale, column
            in scale_cols.items()
        }

        return max(
            values,
            key=values.get,
        )

    master[
        "derived_median_optimum_scale"
    ] = master.apply(
        derive_optimum,
        axis=1,
    )

    master[
        "median_optimum_reproduced"
    ] = (
        master[
            "reported_median_optimum_scale"
        ]
        ==
        master[
            "derived_median_optimum_scale"
        ]
    )

    # ========================================================
    # Scale effect descriptors
    # ========================================================

    master[
        "delta_10x_minus_40x"
    ] = (
        master["median_pearson_10x"]
        -
        master["median_pearson_40x"]
    )

    master[
        "delta_10x_minus_2p5x"
    ] = (
        master["median_pearson_10x"]
        -
        master["median_pearson_2p5x"]
    )

    master[
        "delta_40x_minus_2p5x"
    ] = (
        master["median_pearson_40x"]
        -
        master["median_pearson_2p5x"]
    )

    master[
        "median_scale_range"
    ] = master[
        [
            "median_pearson_2p5x",
            "median_pearson_10x",
            "median_pearson_40x",
        ]
    ].max(axis=1) - master[
        [
            "median_pearson_2p5x",
            "median_pearson_10x",
            "median_pearson_40x",
        ]
    ].min(axis=1)

    master[
        "second_best_scale_median_pearson"
    ] = master.apply(
        lambda row: _second_best_value(
            [
                row["median_pearson_2p5x"],
                row["median_pearson_10x"],
                row["median_pearson_40x"],
            ]
        ),
        axis=1,
    )

    master[
        "median_scale_margin"
    ] = (
        master[
            "best_scale_median_pearson"
        ]
        -
        master[
            "second_best_scale_median_pearson"
        ]
    )

    # ========================================================
    # Model-vote preference
    # ========================================================

    vote_columns = {
        "2p5x": "n_models_prefer_2p5x",
        "10x": "n_models_prefer_10x",
        "40x": "n_models_prefer_40x",
    }

    def vote_mode(row):
        votes = {
            scale: int(row[col])
            for scale, col
            in vote_columns.items()
        }

        maximum = max(votes.values())

        winners = [
            scale
            for scale, count
            in votes.items()
            if count == maximum
        ]

        if len(winners) != 1:
            return "tie"

        return winners[0]

    master[
        "vote_preferred_scale"
    ] = master.apply(
        vote_mode,
        axis=1,
    )

    master[
        "max_model_votes"
    ] = master[
        list(vote_columns.values())
    ].max(axis=1)

    master[
        "vote_consensus_fraction"
    ] = (
        master["max_model_votes"]
        / 8.0
    )

    master[
        "vote_tie"
    ] = (
        master[
            list(vote_columns.values())
        ]
        .eq(
            master[
                list(vote_columns.values())
            ].max(axis=1),
            axis=0,
        )
        .sum(axis=1)
        > 1
    )

    master[
        "median_vote_concordant"
    ] = (
        master["vote_preferred_scale"]
        ==
        master[
            "derived_median_optimum_scale"
        ]
    )

    # ========================================================
    # Raw cross-model visibility descriptors
    # ========================================================

    raw_scale = (
        raw.groupby(
            [
                "gene",
                "scale",
            ],
            as_index=False,
        )
        .agg(
            model_mean_pearson=(
                "pearson",
                "mean",
            ),
            model_std_pearson=(
                "pearson",
                "std",
            ),
            model_median_pearson=(
                "pearson",
                "median",
            ),
            positive_model_fraction=(
                "pearson",
                lambda x: (
                    np.asarray(x) > 0
                ).mean(),
            ),
            median_spearman=(
                "spearman",
                "median",
            ),
        )
    )

    for scale in SCALES:
        part = raw_scale[
            raw_scale["scale"] == scale
        ].copy()

        part = part.drop(
            columns=["scale"]
        )

        part = part.rename(
            columns={
                col: f"{col}_{scale}"
                for col in part.columns
                if col != "gene"
            }
        )

        master = master.merge(
            part,
            on="gene",
            how="left",
            validate="one_to_one",
        )

    # ========================================================
    # Overall gene-level stability descriptors
    # ========================================================

    overall = (
        raw.groupby(
            "gene",
            as_index=False,
        )
        .agg(
            overall_mean_pearson=(
                "pearson",
                "mean",
            ),
            overall_median_pearson=(
                "pearson",
                "median",
            ),
            overall_std_pearson=(
                "pearson",
                "std",
            ),
            overall_mean_spearman=(
                "spearman",
                "mean",
            ),
            overall_median_spearman=(
                "spearman",
                "median",
            ),
            within_cancer_variance_median=(
                "within_cancer_variance",
                "median",
            ),
            within_cancer_variance_min=(
                "within_cancer_variance",
                "min",
            ),
            within_cancer_variance_max=(
                "within_cancer_variance",
                "max",
            ),
        )
    )

    master = master.merge(
        overall,
        on="gene",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # Per-model optimum stability
    # ========================================================

    model_summary = (
        by_model.groupby(
            "gene",
            as_index=False,
        )
        .agg(
            median_model_best_pearson=(
                "best_pearson",
                "median",
            ),
            median_model_scale_margin=(
                "scale_margin",
                "median",
            ),
            minimum_model_scale_margin=(
                "scale_margin",
                "min",
            ),
            maximum_model_scale_margin=(
                "scale_margin",
                "max",
            ),
        )
    )

    master = master.merge(
        model_summary,
        on="gene",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # Basic discovery categories
    # ========================================================

    master[
        "high_visibility_gene"
    ] = (
        master[
            "best_scale_median_pearson"
        ]
        >=
        master[
            "best_scale_median_pearson"
        ].quantile(0.90)
    )

    master[
        "strong_scale_dependence"
    ] = (
        master[
            "median_scale_range"
        ]
        >=
        master[
            "median_scale_range"
        ].quantile(0.90)
    )

    master[
        "strong_model_consensus"
    ] = (
        master[
            "vote_consensus_fraction"
        ]
        >= 0.625
    )

    # ========================================================
    # Stable sort
    # ========================================================

    master = master.sort_values(
        [
            "best_scale_median_pearson",
            "gene",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return master


def save_gene_discovery_table(
    results_root: Optional[str | Path] = None,
) -> Path:
    root = _results_root(results_root)

    output_dir = (
        root
        / "pathbioagent_discovery"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "gene_discovery_master.csv"
    )

    df = build_gene_discovery_table(
        root
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":
    path = save_gene_discovery_table()

    df = pd.read_csv(path)

    print(
        f"Saved: {path}"
    )

    print(
        f"Shape: {df.shape}"
    )

    print(
        "Median optimum reproduced:",
        df[
            "median_optimum_reproduced"
        ].mean(),
    )

    print(
        "Median/vote concordance:",
        df[
            "median_vote_concordant"
        ].mean(),
    )

    print()

    print(
        df[
            [
                "gene",
                "best_scale_median_pearson",
                "derived_median_optimum_scale",
                "vote_preferred_scale",
                "vote_consensus_fraction",
                "median_vote_concordant",
                "median_scale_range",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )
