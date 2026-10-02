
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# Data
# ============================================================

def _default_table_path() -> Path:

    root = os.environ.get(
        "PATHBIO_RESULTS_ROOT",
        "/media/sophie/Data2T/Paper3_8model_gene_analysis",
    )

    return (
        Path(root)
        / "pathbioagent_discovery"
        / "gene_discovery_model_table.csv"
    )


def load_gene_table(
    table_path: Optional[str | Path] = None,
) -> pd.DataFrame:

    path = (
        Path(table_path)
        if table_path is not None
        else _default_table_path()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Gene table not found: {path}"
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
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# Robust scaling
# ============================================================

def _robust_z(
    values: pd.Series,
) -> np.ndarray:

    x = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    median = np.nanmedian(x)

    q1, q3 = np.nanpercentile(
        x,
        [25, 75],
    )

    scale = q3 - q1

    if (
        not np.isfinite(scale)
        or scale < 1e-12
    ):
        scale = np.nanstd(x)

    if (
        not np.isfinite(scale)
        or scale < 1e-12
    ):
        scale = 1.0

    x = np.where(
        np.isfinite(x),
        x,
        median,
    )

    return (
        x - median
    ) / scale


# ============================================================
# Morphology feature construction
# ============================================================

def _build_morphology_space(
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

    # --------------------------------------------------------
    # Absolute visibility level
    # --------------------------------------------------------

    out[
        "_visibility_level"
    ] = (
        v25
        + v10
        + v40
    ) / 3.0


    # --------------------------------------------------------
    # Scale-profile shape
    #
    # Two independent scale contrasts.
    # This separates profile shape from absolute level.
    # --------------------------------------------------------

    out[
        "_shape_10_minus_2p5"
    ] = (
        v10
        - v25
    )

    out[
        "_shape_40_minus_10"
    ] = (
        v40
        - v10
    )


    # --------------------------------------------------------
    # Scale sensitivity
    # --------------------------------------------------------

    out[
        "_scale_range"
    ] = pd.concat(
        [
            v25,
            v10,
            v40,
        ],
        axis=1,
    ).max(
        axis=1
    ) - pd.concat(
        [
            v25,
            v10,
            v40,
        ],
        axis=1,
    ).min(
        axis=1
    )


    # --------------------------------------------------------
    # Robust standardized versions
    # --------------------------------------------------------

    out[
        "_z_visibility"
    ] = _robust_z(
        out["_visibility_level"]
    )

    out[
        "_z_shape1"
    ] = _robust_z(
        out["_shape_10_minus_2p5"]
    )

    out[
        "_z_shape2"
    ] = _robust_z(
        out["_shape_40_minus_10"]
    )

    out[
        "_z_range"
    ] = _robust_z(
        out["_scale_range"]
    )

    out[
        "_z_consensus"
    ] = _robust_z(
        out[
            "vote_consensus_fraction"
        ]
    )

    return out


# ============================================================
# Morphology distance
# ============================================================

def find_similar_genes(
    gene: str,
    top_k: int = 10,
    table_path: Optional[str | Path] = None,
) -> pd.DataFrame:

    df = load_gene_table(
        table_path
    )

    df = _build_morphology_space(
        df
    )

    gene_upper = (
        str(gene)
        .strip()
        .upper()
    )

    gene_lookup = {
        str(g).upper(): i
        for i, g in enumerate(
            df["gene"]
        )
    }

    if gene_upper not in gene_lookup:
        raise ValueError(
            f"Gene not found: {gene}"
        )

    q_idx = gene_lookup[
        gene_upper
    ]

    q = df.iloc[
        q_idx
    ]


    # ========================================================
    # 1. Scale-profile shape distance
    # ========================================================

    shape_distance = np.sqrt(
        (
            (
                df["_z_shape1"]
                - q["_z_shape1"]
            )
            ** 2
            +
            (
                df["_z_shape2"]
                - q["_z_shape2"]
            )
            ** 2
        )
        / 2.0
    )


    # ========================================================
    # 2. Absolute visibility level
    # ========================================================

    level_distance = np.abs(
        df["_z_visibility"]
        - q["_z_visibility"]
    )


    # ========================================================
    # 3. Scale sensitivity
    # ========================================================

    range_distance = np.abs(
        df["_z_range"]
        - q["_z_range"]
    )


    # ========================================================
    # 4. Cross-model preference consensus
    # ========================================================

    consensus_distance = np.abs(
        df["_z_consensus"]
        - q["_z_consensus"]
    )


    # ========================================================
    # 5. Preferred-scale mismatch
    # ========================================================

    preferred_scale_match = (
        df[
            "derived_median_optimum_scale"
        ].astype(str)
        ==
        str(
            q[
                "derived_median_optimum_scale"
            ]
        )
    )

    scale_penalty = (
        ~preferred_scale_match
    ).astype(float)


    # ========================================================
    # Composite pattern distance
    # ========================================================

    pattern_distance = (
        0.40
        * shape_distance
        +
        0.25
        * level_distance
        +
        0.15
        * range_distance
        +
        0.10
        * consensus_distance
        +
        0.10
        * scale_penalty
    )


    # Similarity is secondary.
    # Distance is the primary auditable metric.
    similarity_score = (
        1.0
        /
        (
            1.0
            + pattern_distance
        )
    )


    work = df.copy()

    work[
        "pattern_distance"
    ] = pattern_distance

    work[
        "similarity_score"
    ] = similarity_score

    work[
        "shape_distance"
    ] = shape_distance

    work[
        "visibility_level_difference"
    ] = np.abs(
        work[
            "_visibility_level"
        ]
        - q[
            "_visibility_level"
        ]
    )

    work[
        "scale_range_difference"
    ] = np.abs(
        work[
            "_scale_range"
        ]
        - q[
            "_scale_range"
        ]
    )

    work[
        "consensus_difference"
    ] = np.abs(
        pd.to_numeric(
            work[
                "vote_consensus_fraction"
            ],
            errors="coerce",
        )
        - float(
            q[
                "vote_consensus_fraction"
            ]
        )
    )

    work[
        "preferred_scale_match"
    ] = preferred_scale_match


    # Exclude query itself
    work = work[
        work.index
        != q_idx
    ]


    work = (
        work.sort_values(
            "pattern_distance",
            ascending=True,
        )
        .head(
            top_k
        )
        .copy()
    )


    # ========================================================
    # Human-readable reasoning
    # ========================================================

    reasons = []

    for _, row in work.iterrows():

        r = []

        if row[
            "preferred_scale_match"
        ]:
            r.append(
                "same preferred scale"
            )

        if (
            row[
                "visibility_level_difference"
            ]
            <= 0.05
        ):
            r.append(
                "similar absolute visibility"
            )

        if (
            row[
                "scale_range_difference"
            ]
            <= 0.03
        ):
            r.append(
                "similar scale sensitivity"
            )

        if (
            row[
                "shape_distance"
            ]
            <= 0.50
        ):
            r.append(
                "similar cross-scale profile shape"
            )

        if (
            row[
                "consensus_difference"
            ]
            <= 0.125
        ):
            r.append(
                "similar model consensus"
            )

        if not r:
            r.append(
                "closest composite profile"
            )

        reasons.append(
            "; ".join(
                r
            )
        )


    work[
        "similarity_reason"
    ] = reasons


    columns = [
        "gene",

        "pattern_distance",
        "similarity_score",

        "shape_distance",
        "visibility_level_difference",
        "scale_range_difference",
        "consensus_difference",
        "preferred_scale_match",

        "median_pearson_2p5x",
        "median_pearson_10x",
        "median_pearson_40x",

        "derived_median_optimum_scale",
        "vote_consensus_fraction",

        "rna_cancer_tau_specificity",
        "rna_cancer_eta_squared",
        "rna_dominant_cancer",

        "similarity_reason",
    ]

    columns = [
        c
        for c in columns
        if c in work.columns
    ]

    return (
        work[
            columns
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Agent evidence
# ============================================================

def build_gene_similarity_evidence(
    gene: str,
    top_k: int = 10,
) -> Dict:

    result = find_similar_genes(
        gene=gene,
        top_k=top_k,
    )

    return {

        "evidence_type":
            "gene_visibility_pattern_similarity",

        "query_gene":
            gene,

        "distance_definition": {
            "scale_profile_shape":
                0.40,

            "visibility_level":
                0.25,

            "scale_sensitivity":
                0.15,

            "model_consensus":
                0.10,

            "preferred_scale_mismatch":
                0.10,
        },

        "similar_genes":
            result.to_dict(
                orient="records"
            ),

        "scientific_scope": (
            "Similarity describes cross-scale "
            "morphology-derived molecular visibility "
            "patterns. It does not establish shared "
            "biological mechanism, cell type, pathway, "
            "or functional equivalence."
        ),
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Find genes with similar cross-scale "
            "morphology-derived molecular visibility "
            "patterns."
        )
    )

    parser.add_argument(
        "gene",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    result = find_similar_genes(
        gene=args.gene,
        top_k=args.top_k,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
