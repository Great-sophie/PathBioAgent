
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

GLOBAL_FDR_THRESHOLD = 0.05

DEFAULT_TOP_N_TERMS = 6

HALLMARK_LIBRARY = (
    "MSigDB_Hallmark_2020"
)

GO_REACTOME_LIBRARIES = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
]


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


def _default_cluster_profile_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_hierarchy"
        / "fine_cluster_profiles.csv"
    )


def _default_rna_summary_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "molecular_characterization"
        / "cluster_rna_summary.csv"
    )


def _default_cancer_residual_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "molecular_characterization"
        / "dominant_cancer_standardized_residuals.csv"
    )


def _default_hallmark_enrichment_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "biological_enrichment_hallmark"
        / "enrichment_significant_global_fdr.csv"
    )


def _default_combined_enrichment_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "biological_enrichment_combined"
        / "enrichment_significant_global_fdr.csv"
    )


def _default_stability_summary_path() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_stability"
        / "stability_summary.csv"
    )


def _default_output_dir() -> Path:

    return (
        _results_root()
        / "pathbioagent_discovery"
        / "cluster_biological_evidence"
    )


# ============================================================
# Generic helpers
# ============================================================

def _safe_float(
    value,
) -> Optional[float]:

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not np.isfinite(
        result
    ):

        return None

    return result


def _format_float(
    value,
    digits: int = 3,
) -> str:

    value = _safe_float(
        value
    )

    if value is None:
        return "NA"

    return (
        f"{value:.{digits}f}"
    )


def _format_q(
    value,
) -> str:

    value = _safe_float(
        value
    )

    if value is None:
        return "NA"

    if value < 0.001:

        return (
            f"{value:.2e}"
        )

    return (
        f"{value:.3f}"
    )


def _read_csv_required(
    path: Path,
    description: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found: "
            f"{path}"
        )

    return pd.read_csv(
        path
    )


def _read_csv_optional(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(
        path
    )


# ============================================================
# Visibility phenotype
# ============================================================

def derive_visibility_pattern(
    row: pd.Series,
) -> str:

    dominant_scale = str(
        row.get(
            "dominant_preferred_scale",
            "unknown",
        )
    )

    dominant_fraction = (
        _safe_float(
            row.get(
                "dominant_preferred_scale_fraction"
            )
        )
        or 0.0
    )

    visibility = (
        _safe_float(
            row.get(
                "visibility_level_median"
            )
        )
        or 0.0
    )

    shape_10_minus_2p5 = (
        _safe_float(
            row.get(
                "shape_10_minus_2p5_median"
            )
        )
        or 0.0
    )

    shape_40_minus_10 = (
        _safe_float(
            row.get(
                "shape_40_minus_10_median"
            )
        )
        or 0.0
    )

    scale_range = (
        _safe_float(
            row.get(
                "scale_range_median"
            )
        )
        or 0.0
    )

    # --------------------------------------------------------
    # Near-zero and scale-flat pattern
    # --------------------------------------------------------

    if (
        visibility <= 0.03
        and scale_range <= 0.09
    ):

        return (
            "low-visibility / scale-flat"
        )

    # --------------------------------------------------------
    # Clear low-magnification preference
    # --------------------------------------------------------

    if (
        dominant_scale == "2p5x"
        and dominant_fraction >= 0.90
    ):

        return (
            "2.5x-enhanced"
        )

    # --------------------------------------------------------
    # Clear 40x preference
    # --------------------------------------------------------

    if (
        dominant_scale == "40x"
        and dominant_fraction >= 0.90
        and shape_40_minus_10 > 0.04
    ):

        return (
            "40x-enhanced"
        )

    # --------------------------------------------------------
    # Large 2.5x -> high-magnification gain
    # --------------------------------------------------------

    if (
        shape_10_minus_2p5 >= 0.18
        and scale_range >= 0.20
    ):

        return (
            "high-magnification enhanced"
        )

    # --------------------------------------------------------
    # Strong visibility with weak scale dependence
    # --------------------------------------------------------

    if (
        visibility >= 0.18
        and scale_range <= 0.10
    ):

        return (
            "high-visibility / "
            "relatively scale-stable"
        )

    # --------------------------------------------------------
    # 10x peak
    # --------------------------------------------------------

    if (
        dominant_scale == "10x"
        and dominant_fraction >= 0.75
        and shape_10_minus_2p5 > 0.07
        and shape_40_minus_10 < -0.04
    ):

        return (
            "10x-peaked / lower visibility"
        )

    return (
        f"{dominant_scale}-leaning "
        f"visibility pattern"
    )


def build_visibility_summary(
    row: pd.Series,
) -> str:

    pattern = derive_visibility_pattern(
        row
    )

    p2 = _format_float(
        row.get(
            "median_pearson_2p5x_median"
        )
    )

    p10 = _format_float(
        row.get(
            "median_pearson_10x_median"
        )
    )

    p40 = _format_float(
        row.get(
            "median_pearson_40x_median"
        )
    )

    scale_range = _format_float(
        row.get(
            "scale_range_median"
        )
    )

    consensus = _format_float(
        row.get(
            "vote_consensus_fraction_median"
        )
    )

    return (
        f"{pattern}; median visibility "
        f"2.5x={p2}, 10x={p10}, 40x={p40}; "
        f"scale range={scale_range}; "
        f"model-vote consensus={consensus}"
    )


# ============================================================
# Relative RNA characterization
# ============================================================

def _relative_category(
    series: pd.Series,
    value,
) -> str:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    value = _safe_float(
        value
    )

    if (
        value is None
        or numeric.empty
    ):

        return "unknown"

    minimum = float(
        numeric.min()
    )

    maximum = float(
        numeric.max()
    )

    if np.isclose(
        value,
        maximum,
    ):

        return "highest"

    if np.isclose(
        value,
        minimum,
    ):

        return "lowest"

    percentile = float(
        (
            numeric
            <= value
        ).mean()
    )

    if percentile >= 0.67:
        return "higher"

    if percentile <= 0.33:
        return "lower"

    return "intermediate"


def add_relative_rna_categories(
    rna: pd.DataFrame,
) -> pd.DataFrame:

    rna = rna.copy()

    mapping = {
        "rna_mean_expression_median":
            "rna_expression_relative",

        "rna_above1_fraction_median":
            "rna_expression_prevalence_relative",

        "rna_cancer_tau_specificity_median":
            "rna_tau_specificity_relative",

        "rna_cancer_eta_squared_median":
            "rna_eta_squared_relative",

        "rna_between_cancer_variance_median":
            "rna_between_cancer_variance_relative",
    }

    for source, target in (
        mapping.items()
    ):

        if source not in rna.columns:
            continue

        rna[
            target
        ] = [
            _relative_category(
                rna[
                    source
                ],
                value,
            )
            for value in rna[
                source
            ]
        ]

    return rna


def build_rna_summary(
    row: pd.Series,
) -> str:

    expression = (
        row.get(
            "rna_expression_relative",
            "unknown",
        )
    )

    prevalence = (
        row.get(
            "rna_expression_prevalence_relative",
            "unknown",
        )
    )

    tau = (
        row.get(
            "rna_tau_specificity_relative",
            "unknown",
        )
    )

    eta = (
        row.get(
            "rna_eta_squared_relative",
            "unknown",
        )
    )

    mean_expression = _format_float(
        row.get(
            "rna_mean_expression_median"
        )
    )

    above1 = _format_float(
        row.get(
            "rna_above1_fraction_median"
        )
    )

    tau_value = _format_float(
        row.get(
            "rna_cancer_tau_specificity_median"
        )
    )

    eta_value = _format_float(
        row.get(
            "rna_cancer_eta_squared_median"
        )
    )

    return (
        f"{expression} RNA abundance "
        f"(median={mean_expression}); "
        f"{prevalence} expression prevalence "
        f"(fraction >1={above1}); "
        f"{tau} cancer specificity by tau "
        f"(median={tau_value}); "
        f"{eta} cancer-associated structure "
        f"by eta-squared "
        f"(median={eta_value})"
    )


# ============================================================
# Cancer composition
# ============================================================

def load_cancer_residuals(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    residuals = pd.read_csv(
        path,
        index_col=0,
    )

    residuals.index = pd.to_numeric(
        residuals.index,
        errors="coerce",
    )

    residuals = residuals.loc[
        residuals.index.notna()
    ].copy()

    residuals.index = (
        residuals.index.astype(
            int
        )
    )

    return residuals


def summarize_cancer_residuals(
    residuals: pd.DataFrame,
    cluster: int,
) -> Dict:

    if (
        residuals.empty
        or cluster
        not in residuals.index
    ):

        return {
            "overrepresented_cancers":
                "",

            "underrepresented_cancers":
                "",

            "largest_positive_cancer_residual":
                np.nan,

            "largest_negative_cancer_residual":
                np.nan,
        }

    row = pd.to_numeric(
        residuals.loc[
            cluster
        ],
        errors="coerce",
    )

    positive = row.loc[
        row > 2.0
    ].sort_values(
        ascending=False
    )

    negative = row.loc[
        row < -2.0
    ].sort_values(
        ascending=True
    )

    overrepresented = "; ".join(
        f"{name} ({value:+.2f})"
        for name, value
        in positive.items()
    )

    underrepresented = "; ".join(
        f"{name} ({value:+.2f})"
        for name, value
        in negative.items()
    )

    return {
        "overrepresented_cancers":
            overrepresented,

        "underrepresented_cancers":
            underrepresented,

        "largest_positive_cancer_residual":
            (
                float(
                    row.max()
                )
                if row.notna().any()
                else np.nan
            ),

        "largest_negative_cancer_residual":
            (
                float(
                    row.min()
                )
                if row.notna().any()
                else np.nan
            ),
    }


# ============================================================
# Enrichment
# ============================================================

def _ensure_significant_enrichment(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:

        return df

    df = df.copy()

    if (
        "q_value_global"
        not in df.columns
    ):

        return pd.DataFrame()

    df[
        "q_value_global"
    ] = pd.to_numeric(
        df[
            "q_value_global"
        ],
        errors="coerce",
    )

    df = df.loc[
        df[
            "q_value_global"
        ]
        <
        GLOBAL_FDR_THRESHOLD
    ].copy()

    return df


def summarize_enrichment(
    enrichment: pd.DataFrame,
    cluster: int,
    libraries: Optional[
        Sequence[str]
    ] = None,
    top_n: int = DEFAULT_TOP_N_TERMS,
) -> Dict:

    if enrichment.empty:

        return {
            "n_significant":
                0,

            "top_term":
                "",

            "top_q_value":
                np.nan,

            "top_fold_enrichment":
                np.nan,

            "terms":
                "",

            "term_details":
                [],
        }

    subset = enrichment.loc[
        enrichment[
            "fine_cluster"
        ]
        == cluster
    ].copy()

    if libraries is not None:

        subset = subset.loc[
            subset[
                "library"
            ].isin(
                libraries
            )
        ].copy()

    if subset.empty:

        return {
            "n_significant":
                0,

            "top_term":
                "",

            "top_q_value":
                np.nan,

            "top_fold_enrichment":
                np.nan,

            "terms":
                "",

            "term_details":
                [],
        }

    subset[
        "q_value_global"
    ] = pd.to_numeric(
        subset[
            "q_value_global"
        ],
        errors="coerce",
    )

    subset[
        "fold_enrichment"
    ] = pd.to_numeric(
        subset[
            "fold_enrichment"
        ],
        errors="coerce",
    )

    subset = subset.sort_values(
        [
            "q_value_global",
            "fold_enrichment",
        ],
        ascending=[
            True,
            False,
        ],
    )

    top = subset.head(
        top_n
    )

    details: List[Dict] = []

    for _, row in top.iterrows():

        details.append(
            {
                "library":
                    str(
                        row.get(
                            "library",
                            "",
                        )
                    ),

                "term":
                    str(
                        row.get(
                            "term",
                            "",
                        )
                    ),

                "q_value_global":
                    _safe_float(
                        row.get(
                            "q_value_global"
                        )
                    ),

                "fold_enrichment":
                    _safe_float(
                        row.get(
                            "fold_enrichment"
                        )
                    ),

                "odds_ratio":
                    _safe_float(
                        row.get(
                            "odds_ratio"
                        )
                    ),

                "overlap":
                    (
                        int(
                            row.get(
                                "overlap"
                            )
                        )
                        if pd.notna(
                            row.get(
                                "overlap"
                            )
                        )
                        else None
                    ),
            }
        )

    strongest = top.iloc[
        0
    ]

    terms_string = "; ".join(
        (
            f"{row['term']} "
            f"[{row['library']}; "
            f"q={_format_q(row['q_value_global'])}; "
            f"FE={_format_float(row['fold_enrichment'], 2)}]"
        )
        for _, row
        in top.iterrows()
    )

    return {
        "n_significant":
            int(
                len(
                    subset
                )
            ),

        "top_term":
            str(
                strongest.get(
                    "term",
                    "",
                )
            ),

        "top_q_value":
            _safe_float(
                strongest.get(
                    "q_value_global"
                )
            ),

        "top_fold_enrichment":
            _safe_float(
                strongest.get(
                    "fold_enrichment"
                )
            ),

        "terms":
            terms_string,

        "term_details":
            details,
    }


# ============================================================
# K=6 global stability
# ============================================================

def load_k6_stability(
    path: Path,
) -> Dict:

    if not path.exists():

        return {}

    df = pd.read_csv(
        path
    )

    required = {
        "k",
        "stability_type",
    }

    if not required.issubset(
        df.columns
    ):

        return {}

    subset = df.loc[
        (
            pd.to_numeric(
                df[
                    "k"
                ],
                errors="coerce",
            )
            == 6
        )
        &
        (
            df[
                "stability_type"
            ]
            == "subsample"
        )
    ]

    if subset.empty:

        return {}

    row = subset.iloc[
        0
    ]

    return {
        "k6_subsample_ari_median":
            _safe_float(
                row.get(
                    "ari_median"
                )
            ),

        "k6_subsample_nmi_median":
            _safe_float(
                row.get(
                    "nmi_median"
                )
            ),

        "k6_subsample_label_agreement_median":
            _safe_float(
                row.get(
                    "matched_label_agreement_median"
                )
            ),

        "k6_subsample_centroid_distance_median":
            _safe_float(
                row.get(
                    "mean_centroid_distance_median"
                )
            ),
    }


# ============================================================
# Evidence status
# ============================================================

def determine_evidence_status(
    hallmark_n: int,
    go_reactome_n: int,
    has_rna: bool,
) -> str:

    if (
        hallmark_n > 0
        or go_reactome_n > 0
    ):

        return (
            "supported_exploratory_"
            "biological_pattern"
        )

    if has_rna:

        return (
            "rna_characterized_without_"
            "global_fdr_pathway_enrichment"
        )

    return (
        "descriptive_visibility_pattern_only"
    )


# ============================================================
# Agent-facing deterministic summary
# ============================================================

def build_agent_summary(
    cluster: int,
    pattern: str,
    n_genes: int,
    visibility_summary: str,
    rna_summary: str,
    cancer_summary: Dict,
    hallmark: Dict,
    go_reactome: Dict,
    evidence_status: str,
) -> str:

    parts = [
        (
            f"Fine cluster {cluster} "
            f"contains {n_genes} genes and "
            f"shows a {pattern} "
            f"cross-scale morphology-derived "
            f"molecular visibility pattern."
        ),

        (
            f"Visibility evidence: "
            f"{visibility_summary}."
        ),

        (
            f"RNA characterization: "
            f"{rna_summary}."
        ),
    ]

    if cancer_summary[
        "overrepresented_cancers"
    ]:

        parts.append(
            "Cancer-composition analysis shows "
            "over-representation relative to "
            "expectation for: "
            f"{cancer_summary['overrepresented_cancers']}."
        )

    if cancer_summary[
        "underrepresented_cancers"
    ]:

        parts.append(
            "Under-representation relative to "
            "expectation is observed for: "
            f"{cancer_summary['underrepresented_cancers']}."
        )

    if hallmark[
        "n_significant"
    ] > 0:

        parts.append(
            "Hallmark programs surviving global "
            "FDR include: "
            f"{hallmark['terms']}."
        )

    else:

        parts.append(
            "No Hallmark program survived the "
            "global FDR threshold of 0.05."
        )

    if go_reactome[
        "n_significant"
    ] > 0:

        parts.append(
            "GO/Reactome programs surviving "
            "global FDR include: "
            f"{go_reactome['terms']}."
        )

    else:

        parts.append(
            "No GO/Reactome program survived "
            "the global FDR threshold of 0.05."
        )

    parts.append(
        (
            f"Evidence status: "
            f"{evidence_status}. "
            f"These associations are exploratory "
            f"within the same 63-case cohort and "
            f"do not establish mechanism, "
            f"causality, cell type, biomarker "
            f"validity, or external generalization."
        )
    )

    return " ".join(
        parts
    )


# ============================================================
# Main assembly
# ============================================================

def build_cluster_biological_evidence(
    cluster_profile_path: Optional[
        str | Path
    ] = None,
    rna_summary_path: Optional[
        str | Path
    ] = None,
    cancer_residual_path: Optional[
        str | Path
    ] = None,
    hallmark_enrichment_path: Optional[
        str | Path
    ] = None,
    combined_enrichment_path: Optional[
        str | Path
    ] = None,
    stability_summary_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    top_n_terms: int = DEFAULT_TOP_N_TERMS,
) -> Dict:

    cluster_profile_path = Path(
        cluster_profile_path
        if cluster_profile_path is not None
        else _default_cluster_profile_path()
    )

    rna_summary_path = Path(
        rna_summary_path
        if rna_summary_path is not None
        else _default_rna_summary_path()
    )

    cancer_residual_path = Path(
        cancer_residual_path
        if cancer_residual_path is not None
        else _default_cancer_residual_path()
    )

    hallmark_enrichment_path = Path(
        hallmark_enrichment_path
        if hallmark_enrichment_path is not None
        else _default_hallmark_enrichment_path()
    )

    combined_enrichment_path = Path(
        combined_enrichment_path
        if combined_enrichment_path is not None
        else _default_combined_enrichment_path()
    )

    stability_summary_path = Path(
        stability_summary_path
        if stability_summary_path is not None
        else _default_stability_summary_path()
    )

    # --------------------------------------------------------
    # Load visibility patterns
    # --------------------------------------------------------

    profiles = _read_csv_required(
        cluster_profile_path,
        "Fine cluster profile table",
    )

    if (
        "fine_cluster"
        not in profiles.columns
    ):

        raise ValueError(
            "fine_cluster column missing from "
            f"{cluster_profile_path}"
        )

    profiles[
        "fine_cluster"
    ] = profiles[
        "fine_cluster"
    ].astype(
        int
    )

    # --------------------------------------------------------
    # Load RNA characterization
    # --------------------------------------------------------

    rna = _read_csv_required(
        rna_summary_path,
        "Cluster RNA summary",
    )

    if (
        "fine_cluster"
        not in rna.columns
    ):

        raise ValueError(
            "fine_cluster column missing from "
            f"{rna_summary_path}"
        )

    rna[
        "fine_cluster"
    ] = rna[
        "fine_cluster"
    ].astype(
        int
    )

    rna = add_relative_rna_categories(
        rna
    )

    # --------------------------------------------------------
    # Load cancer-composition residuals
    # --------------------------------------------------------

    cancer_residuals = (
        load_cancer_residuals(
            cancer_residual_path
        )
    )

    # --------------------------------------------------------
    # Load enrichment
    # --------------------------------------------------------

    hallmark = _read_csv_optional(
        hallmark_enrichment_path
    )

    hallmark = (
        _ensure_significant_enrichment(
            hallmark
        )
    )

    combined = _read_csv_optional(
        combined_enrichment_path
    )

    combined = (
        _ensure_significant_enrichment(
            combined
        )
    )

    # Combined analysis also contains Hallmark.
    # For this evidence layer, the combined section is
    # intentionally restricted to GO + Reactome so the
    # Hallmark evidence is not counted twice.
    if not combined.empty:

        combined_go_reactome = (
            combined.loc[
                combined[
                    "library"
                ].isin(
                    GO_REACTOME_LIBRARIES
                )
            ].copy()
        )

    else:

        combined_go_reactome = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Global k=6 stability
    # --------------------------------------------------------

    stability = load_k6_stability(
        stability_summary_path
    )

    # --------------------------------------------------------
    # Merge visibility + RNA
    # --------------------------------------------------------

    merged = profiles.merge(
        rna,
        on="fine_cluster",
        how="left",
        suffixes=(
            "_visibility",
            "_rna",
        ),
        validate="one_to_one",
    )

    rows: List[Dict] = []

    json_clusters: List[Dict] = []

    for _, row in merged.iterrows():

        cluster = int(
            row[
                "fine_cluster"
            ]
        )

        n_genes = int(
            row.get(
                "n_genes_visibility",
                row.get(
                    "n_genes",
                    0,
                ),
            )
        )

        pattern = (
            derive_visibility_pattern(
                row
            )
        )

        visibility_summary = (
            build_visibility_summary(
                row
            )
        )

        rna_summary = (
            build_rna_summary(
                row
            )
        )

        cancer_summary = (
            summarize_cancer_residuals(
                residuals=
                    cancer_residuals,
                cluster=
                    cluster,
            )
        )

        hallmark_summary = (
            summarize_enrichment(
                enrichment=
                    hallmark,
                cluster=
                    cluster,
                libraries=[
                    HALLMARK_LIBRARY
                ],
                top_n=
                    top_n_terms,
            )
        )

        go_reactome_summary = (
            summarize_enrichment(
                enrichment=
                    combined_go_reactome,
                cluster=
                    cluster,
                libraries=
                    GO_REACTOME_LIBRARIES,
                top_n=
                    top_n_terms,
            )
        )

        has_rna = (
            pd.notna(
                row.get(
                    "rna_mean_expression_median"
                )
            )
        )

        evidence_status = (
            determine_evidence_status(
                hallmark_n=
                    hallmark_summary[
                        "n_significant"
                    ],

                go_reactome_n=
                    go_reactome_summary[
                        "n_significant"
                    ],

                has_rna=
                    bool(
                        has_rna
                    ),
            )
        )

        agent_summary = (
            build_agent_summary(
                cluster=
                    cluster,

                pattern=
                    pattern,

                n_genes=
                    n_genes,

                visibility_summary=
                    visibility_summary,

                rna_summary=
                    rna_summary,

                cancer_summary=
                    cancer_summary,

                hallmark=
                    hallmark_summary,

                go_reactome=
                    go_reactome_summary,

                evidence_status=
                    evidence_status,
            )
        )

        output_row = {
            "fine_cluster":
                cluster,

            "n_genes":
                n_genes,

            # --------------------------------------------
            # Visibility phenotype
            # --------------------------------------------

            "visibility_pattern":
                pattern,

            "dominant_preferred_scale":
                row.get(
                    "dominant_preferred_scale"
                ),

            "dominant_preferred_scale_fraction":
                row.get(
                    "dominant_preferred_scale_fraction"
                ),

            "visibility_level_median":
                row.get(
                    "visibility_level_median"
                ),

            "median_pearson_2p5x":
                row.get(
                    "median_pearson_2p5x_median"
                ),

            "median_pearson_10x":
                row.get(
                    "median_pearson_10x_median"
                ),

            "median_pearson_40x":
                row.get(
                    "median_pearson_40x_median"
                ),

            "median_scale_range":
                row.get(
                    "scale_range_median"
                ),

            "vote_consensus_fraction_median":
                row.get(
                    "vote_consensus_fraction_median"
                ),

            "visibility_summary":
                visibility_summary,

            # --------------------------------------------
            # RNA characterization
            # --------------------------------------------

            "rna_mean_expression_median":
                row.get(
                    "rna_mean_expression_median"
                ),

            "rna_above1_fraction_median":
                row.get(
                    "rna_above1_fraction_median"
                ),

            "rna_nonzero_fraction_median":
                row.get(
                    "rna_nonzero_fraction_median"
                ),

            "rna_between_cancer_variance_median":
                row.get(
                    "rna_between_cancer_variance_median"
                ),

            "rna_cancer_eta_squared_median":
                row.get(
                    "rna_cancer_eta_squared_median"
                ),

            "rna_cancer_tau_specificity_median":
                row.get(
                    "rna_cancer_tau_specificity_median"
                ),

            "rna_cancer_entropy_specificity_median":
                row.get(
                    "rna_cancer_entropy_specificity_median"
                ),

            "rna_expression_relative":
                row.get(
                    "rna_expression_relative"
                ),

            "rna_expression_prevalence_relative":
                row.get(
                    "rna_expression_prevalence_relative"
                ),

            "rna_tau_specificity_relative":
                row.get(
                    "rna_tau_specificity_relative"
                ),

            "rna_eta_squared_relative":
                row.get(
                    "rna_eta_squared_relative"
                ),

            "rna_between_cancer_variance_relative":
                row.get(
                    "rna_between_cancer_variance_relative"
                ),

            "rna_summary":
                rna_summary,

            # --------------------------------------------
            # Cancer composition
            # --------------------------------------------

            "dominant_cancer":
                row.get(
                    "dominant_cancer"
                ),

            "dominant_cancer_fraction":
                row.get(
                    "dominant_cancer_fraction"
                ),

            "overrepresented_cancers":
                cancer_summary[
                    "overrepresented_cancers"
                ],

            "underrepresented_cancers":
                cancer_summary[
                    "underrepresented_cancers"
                ],

            # --------------------------------------------
            # Hallmark enrichment
            # --------------------------------------------

            "n_significant_hallmark_terms":
                hallmark_summary[
                    "n_significant"
                ],

            "top_hallmark_term":
                hallmark_summary[
                    "top_term"
                ],

            "top_hallmark_q_global":
                hallmark_summary[
                    "top_q_value"
                ],

            "top_hallmark_fold_enrichment":
                hallmark_summary[
                    "top_fold_enrichment"
                ],

            "significant_hallmark_terms":
                hallmark_summary[
                    "terms"
                ],

            # --------------------------------------------
            # GO / Reactome enrichment
            # --------------------------------------------

            "n_significant_go_reactome_terms":
                go_reactome_summary[
                    "n_significant"
                ],

            "top_go_reactome_term":
                go_reactome_summary[
                    "top_term"
                ],

            "top_go_reactome_q_global":
                go_reactome_summary[
                    "top_q_value"
                ],

            "top_go_reactome_fold_enrichment":
                go_reactome_summary[
                    "top_fold_enrichment"
                ],

            "significant_go_reactome_terms":
                go_reactome_summary[
                    "terms"
                ],

            # --------------------------------------------
            # Evidence status
            # --------------------------------------------

            "evidence_status":
                evidence_status,

            "agent_summary":
                agent_summary,
        }

        # Add same global stability metrics to each row.
        output_row.update(
            stability
        )

        rows.append(
            output_row
        )

        json_clusters.append(
            {
                "fine_cluster":
                    cluster,

                "n_genes":
                    n_genes,

                "visibility": {
                    "pattern":
                        pattern,

                    "summary":
                        visibility_summary,

                    "dominant_preferred_scale":
                        row.get(
                            "dominant_preferred_scale"
                        ),

                    "median_pearson": {
                        "2p5x":
                            _safe_float(
                                row.get(
                                    "median_pearson_2p5x_median"
                                )
                            ),

                        "10x":
                            _safe_float(
                                row.get(
                                    "median_pearson_10x_median"
                                )
                            ),

                        "40x":
                            _safe_float(
                                row.get(
                                    "median_pearson_40x_median"
                                )
                            ),
                    },

                    "median_scale_range":
                        _safe_float(
                            row.get(
                                "scale_range_median"
                            )
                        ),

                    "vote_consensus_fraction_median":
                        _safe_float(
                            row.get(
                                "vote_consensus_fraction_median"
                            )
                        ),
                },

                "rna": {
                    "summary":
                        rna_summary,

                    "mean_expression_median":
                        _safe_float(
                            row.get(
                                "rna_mean_expression_median"
                            )
                        ),

                    "above1_fraction_median":
                        _safe_float(
                            row.get(
                                "rna_above1_fraction_median"
                            )
                        ),

                    "tau_specificity_median":
                        _safe_float(
                            row.get(
                                "rna_cancer_tau_specificity_median"
                            )
                        ),

                    "eta_squared_median":
                        _safe_float(
                            row.get(
                                "rna_cancer_eta_squared_median"
                            )
                        ),

                    "between_cancer_variance_median":
                        _safe_float(
                            row.get(
                                "rna_between_cancer_variance_median"
                            )
                        ),
                },

                "cancer_composition": {
                    "dominant_cancer":
                        (
                            None
                            if pd.isna(
                                row.get(
                                    "dominant_cancer"
                                )
                            )
                            else str(
                                row.get(
                                    "dominant_cancer"
                                )
                            )
                        ),

                    "overrepresented":
                        cancer_summary[
                            "overrepresented_cancers"
                        ],

                    "underrepresented":
                        cancer_summary[
                            "underrepresented_cancers"
                        ],
                },

                "hallmark_enrichment":
                    hallmark_summary,

                "go_reactome_enrichment":
                    go_reactome_summary,

                "evidence_status":
                    evidence_status,

                "agent_summary":
                    agent_summary,
            }
        )

    evidence = (
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

    # --------------------------------------------------------
    # Long-format enrichment evidence
    # --------------------------------------------------------

    long_frames = []

    if not hallmark.empty:

        hall_long = hallmark.copy()

        hall_long[
            "evidence_source"
        ] = "hallmark"

        long_frames.append(
            hall_long
        )

    if not combined_go_reactome.empty:

        combined_long = (
            combined_go_reactome.copy()
        )

        combined_long[
            "evidence_source"
        ] = "go_reactome"

        long_frames.append(
            combined_long
        )

    if long_frames:

        enrichment_long = pd.concat(
            long_frames,
            ignore_index=True,
        )

        enrichment_long = (
            enrichment_long.sort_values(
                [
                    "fine_cluster",
                    "q_value_global",
                    "fold_enrichment",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    else:

        enrichment_long = (
            pd.DataFrame()
        )

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

    evidence_path = (
        out_dir
        / "cluster_biological_evidence.csv"
    )

    enrichment_long_path = (
        out_dir
        / "cluster_enrichment_evidence_long.csv"
    )

    json_path = (
        out_dir
        / "cluster_biological_evidence.json"
    )

    metadata_path = (
        out_dir
        / "cluster_biological_evidence_metadata.json"
    )

    evidence.to_csv(
        evidence_path,
        index=False,
    )

    enrichment_long.to_csv(
        enrichment_long_path,
        index=False,
    )

    json_payload = {
        "evidence_type":
            "cluster_biological_evidence",

        "n_clusters":
            int(
                len(
                    evidence
                )
            ),

        "global_fdr_threshold":
            GLOBAL_FDR_THRESHOLD,

        "stability":
            stability,

        "clusters":
            json_clusters,

        "scientific_scope": (
            "This evidence layer integrates "
            "cross-scale morphology-derived molecular "
            "visibility phenotypes, RNA-derived gene "
            "properties, cancer-composition associations, "
            "and pathway over-representation results. "
            "All molecular characterization is derived "
            "from the same primary 63-case cohort. "
            "The evidence is exploratory and "
            "hypothesis-generating and does not establish "
            "mechanism, causality, cell identity, "
            "clinical biomarker validity, or external "
            "generalizability."
        ),
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            indent=2,
            ensure_ascii=False,
        )
    )

    metadata = {
        "input_files": {
            "cluster_profiles":
                str(
                    cluster_profile_path
                ),

            "rna_summary":
                str(
                    rna_summary_path
                ),

            "cancer_residuals":
                str(
                    cancer_residual_path
                ),

            "hallmark_enrichment":
                str(
                    hallmark_enrichment_path
                ),

            "combined_enrichment":
                str(
                    combined_enrichment_path
                ),

            "stability_summary":
                str(
                    stability_summary_path
                ),
        },

        "output_files": {
            "cluster_evidence":
                str(
                    evidence_path
                ),

            "enrichment_long":
                str(
                    enrichment_long_path
                ),

            "json_evidence":
                str(
                    json_path
                ),
        },

        "global_fdr_threshold":
            GLOBAL_FDR_THRESHOLD,

        "top_n_terms_per_evidence_section":
            int(
                top_n_terms
            ),

        "important_methodological_note": (
            "Hallmark evidence is read from the "
            "Hallmark-only global-FDR analysis. "
            "The combined enrichment section is "
            "restricted to GO Biological Process "
            "and Reactome to avoid double-counting "
            "Hallmark terms."
        ),

        "pairwise_rna_note": (
            "Pairwise RNA tests are intentionally not "
            "included in this evidence layer because the "
            "current pairwise table uses within-feature "
            "BH correction rather than a single global "
            "correction across all pairwise tests."
        ),

        "scientific_scope": (
            "Cluster labels summarize descriptive "
            "cross-scale morphology-derived molecular "
            "visibility patterns. Biological enrichment "
            "and RNA characterization indicate "
            "associations within the same discovery "
            "cohort and should not be interpreted as "
            "causal mechanisms."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )

    return {
        "evidence":
            evidence,

        "enrichment_long":
            enrichment_long,

        "json_payload":
            json_payload,

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
            "Integrate cross-scale visibility patterns, "
            "RNA characterization, cancer composition, "
            "and pathway enrichment into a deterministic "
            "cluster biological evidence layer."
        )
    )

    parser.add_argument(
        "--cluster-profiles",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rna-summary",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cancer-residuals",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--hallmark-enrichment",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--combined-enrichment",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--stability-summary",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--top-n-terms",
        type=int,
        default=DEFAULT_TOP_N_TERMS,
    )

    args = parser.parse_args()

    result = (
        build_cluster_biological_evidence(
            cluster_profile_path=
                args.cluster_profiles,

            rna_summary_path=
                args.rna_summary,

            cancer_residual_path=
                args.cancer_residuals,

            hallmark_enrichment_path=
                args.hallmark_enrichment,

            combined_enrichment_path=
                args.combined_enrichment,

            stability_summary_path=
                args.stability_summary,

            output_dir=
                args.output_dir,

            top_n_terms=
                args.top_n_terms,
        )
    )

    evidence = result[
        "evidence"
    ]

    print(
        "\nCluster biological evidence:"
    )

    display_columns = [
        "fine_cluster",
        "n_genes",
        "visibility_pattern",
        "rna_expression_relative",
        "rna_tau_specificity_relative",
        "n_significant_hallmark_terms",
        "top_hallmark_term",
        "n_significant_go_reactome_terms",
        "top_go_reactome_term",
        "evidence_status",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in evidence.columns
    ]

    print(
        evidence[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nAgent summaries:"
    )

    for _, row in (
        evidence.iterrows()
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Cluster "
            f"{int(row['fine_cluster'])}"
        )

        print(
            row[
                "agent_summary"
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
