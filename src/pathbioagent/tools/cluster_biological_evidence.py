
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ============================================================
# Paths
# ============================================================

def _default_evidence_path() -> Path:

    root = Path(
        os.environ.get(
            "PATHBIO_RESULTS_ROOT",
            "/media/sophie/Data2T/"
            "Paper3_8model_gene_analysis",
        )
    )

    return (
        root
        / "pathbioagent_discovery"
        / "cluster_biological_evidence"
        / "cluster_biological_evidence.csv"
    )


# ============================================================
# Normalization helpers
# ============================================================

def _normalize_text(
    value,
) -> str:

    if value is None:
        return ""

    text = str(
        value
    ).lower()

    # Normalize magnification notation.
    text = (
        text
        .replace(
            "2.5×",
            "2p5x",
        )
        .replace(
            "2.5x",
            "2p5x",
        )
        .replace(
            "2p5×",
            "2p5x",
        )
        .replace(
            "10×",
            "10x",
        )
        .replace(
            "40×",
            "40x",
        )
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _normalize_scale(
    scale,
) -> str:

    text = _normalize_text(
        scale
    )

    compact = text.replace(
        " ",
        "",
    )

    aliases = {
        "2p5":
            "2p5x",

        "2p5x":
            "2p5x",

        "2.5":
            "2p5x",

        "10":
            "10x",

        "10x":
            "10x",

        "40":
            "40x",

        "40x":
            "40x",
    }

    return aliases.get(
        compact,
        compact,
    )


def _query_tokens(
    query,
) -> list[str]:

    normalized = _normalize_text(
        query
    )

    if not normalized:
        return []

    return normalized.split()


def _token_match(
    query,
    target,
) -> bool:

    query_tokens = _query_tokens(
        query
    )

    target_normalized = _normalize_text(
        target
    )

    if not query_tokens:
        return False

    return all(
        token in target_normalized.split()
        for token in query_tokens
    )


# ============================================================
# Evidence-strength classification
# ============================================================

def classify_evidence_strength(
    n_hallmark_terms: int,
    n_go_reactome_terms: int,
) -> str:

    n_hallmark_terms = int(
        n_hallmark_terms or 0
    )

    n_go_reactome_terms = int(
        n_go_reactome_terms or 0
    )

    if (
        n_hallmark_terms > 0
        and n_go_reactome_terms > 0
    ):

        return (
            "multi_source_enrichment_supported"
        )

    if (
        n_hallmark_terms > 0
        or n_go_reactome_terms > 0
    ):

        return (
            "single_source_enrichment_supported"
        )

    return (
        "rna_characterized_without_"
        "global_fdr_enrichment"
    )


def _attach_evidence_strength(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    hallmark = pd.to_numeric(
        result.get(
            "n_significant_hallmark_terms",
            0,
        ),
        errors="coerce",
    ).fillna(
        0
    )

    go_reactome = pd.to_numeric(
        result.get(
            "n_significant_go_reactome_terms",
            0,
        ),
        errors="coerce",
    ).fillna(
        0
    )

    result[
        "evidence_strength"
    ] = [
        classify_evidence_strength(
            n_hallmark_terms=int(
                h
            ),
            n_go_reactome_terms=int(
                g
            ),
        )
        for h, g in zip(
            hallmark,
            go_reactome,
        )
    ]

    return result


# ============================================================
# Load evidence
# ============================================================

def load_cluster_biological_evidence(
    evidence_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    path = (
        Path(
            evidence_path
        )
        if evidence_path is not None
        else _default_evidence_path()
    )

    if not path.exists():

        raise FileNotFoundError(
            "Cluster biological evidence "
            f"not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "fine_cluster",
        "visibility_pattern",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns "
            f"in biological evidence: "
            f"{missing}"
        )

    df = df.copy()

    df[
        "fine_cluster"
    ] = pd.to_numeric(
        df[
            "fine_cluster"
        ],
        errors="raise",
    ).astype(
        int
    )

    df = (
        df.sort_values(
            "fine_cluster"
        )
        .reset_index(
            drop=True
        )
    )

    df = _attach_evidence_strength(
        df
    )

    return df


# ============================================================
# Direct retrieval
# ============================================================

def get_cluster_evidence(
    cluster_id: int,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> Dict:

    df = load_cluster_biological_evidence(
        evidence_path
    )

    cluster_id = int(
        cluster_id
    )

    subset = df.loc[
        df[
            "fine_cluster"
        ]
        == cluster_id
    ]

    if subset.empty:

        raise ValueError(
            "Cluster not found in biological "
            f"evidence: {cluster_id}"
        )

    row = subset.iloc[
        0
    ]

    return row.to_dict()


# ============================================================
# Search by scale
# ============================================================

def find_clusters_by_scale(
    scale: str,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    df = load_cluster_biological_evidence(
        evidence_path
    )

    requested_scale = (
        _normalize_scale(
            scale
        )
    )

    if not requested_scale:

        return df.iloc[
            0:0
        ].copy()

    rows = []

    for _, row in df.iterrows():

        pattern = _normalize_text(
            row.get(
                "visibility_pattern",
                "",
            )
        )

        dominant_scale = (
            _normalize_scale(
                row.get(
                    "dominant_preferred_scale",
                    "",
                )
            )
        )

        score = 0

        # Strongest evidence:
        # scale explicitly defines the visibility pattern.
        if requested_scale in pattern.split():

            score += 4

        # Preferred scale also contributes,
        # but should rank below explicit pattern.
        if (
            dominant_scale
            == requested_scale
        ):

            score += 2

        if score <= 0:
            continue

        record = row.to_dict()

        record[
            "_scale_match_score"
        ] = score

        rows.append(
            record
        )

    if not rows:

        return df.iloc[
            0:0
        ].copy()

    result = pd.DataFrame(
        rows
    )

    result = (
        result.sort_values(
            [
                "_scale_match_score",
                "fine_cluster",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop(
            columns=[
                "_scale_match_score"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Search by visibility phenotype
# ============================================================

def find_clusters_by_visibility_pattern(
    pattern: str,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    df = load_cluster_biological_evidence(
        evidence_path
    )

    query_tokens = _query_tokens(
        pattern
    )

    if not query_tokens:

        return df.iloc[
            0:0
        ].copy()

    rows = []

    for _, row in df.iterrows():

        target = _normalize_text(
            row.get(
                "visibility_pattern",
                "",
            )
        )

        target_tokens = target.split()

        overlap = sum(
            token in target_tokens
            for token in query_tokens
        )

        if (
            overlap
            != len(
                query_tokens
            )
        ):
            continue

        record = row.to_dict()

        record[
            "_pattern_match_score"
        ] = (
            overlap
            / len(
                query_tokens
            )
        )

        rows.append(
            record
        )

    if not rows:

        return df.iloc[
            0:0
        ].copy()

    result = pd.DataFrame(
        rows
    )

    result = (
        result.sort_values(
            [
                "_pattern_match_score",
                "fine_cluster",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop(
            columns=[
                "_pattern_match_score"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Search by biological program
# ============================================================

PROGRAM_COLUMNS = [
    "top_hallmark_term",
    "significant_hallmark_terms",
    "top_go_reactome_term",
    "significant_go_reactome_terms",
]


def find_clusters_by_program(
    program: str,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    df = load_cluster_biological_evidence(
        evidence_path
    )

    query_tokens = _query_tokens(
        program
    )

    if not query_tokens:

        return df.iloc[
            0:0
        ].copy()

    rows = []

    for _, row in df.iterrows():

        score = 0

        # Top terms get a stronger match score.
        for column in [
            "top_hallmark_term",
            "top_go_reactome_term",
        ]:

            value = row.get(
                column,
                "",
            )

            if _token_match(
                program,
                value,
            ):

                score += 4

        # Full significant-term lists.
        for column in [
            "significant_hallmark_terms",
            "significant_go_reactome_terms",
        ]:

            value = row.get(
                column,
                "",
            )

            if _token_match(
                program,
                value,
            ):

                score += 2

        if score <= 0:
            continue

        record = row.to_dict()

        record[
            "_program_match_score"
        ] = score

        rows.append(
            record
        )

    if not rows:

        return df.iloc[
            0:0
        ].copy()

    result = pd.DataFrame(
        rows
    )

    strength_rank = {
        "multi_source_enrichment_supported":
            3,

        "single_source_enrichment_supported":
            2,

        "rna_characterized_without_global_fdr_enrichment":
            1,
    }

    result[
        "_strength_score"
    ] = (
        result[
            "evidence_strength"
        ]
        .map(
            strength_rank
        )
        .fillna(
            0
        )
    )

    result = (
        result.sort_values(
            [
                "_program_match_score",
                "_strength_score",
                "fine_cluster",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .drop(
            columns=[
                "_program_match_score",
                "_strength_score",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Cluster comparison
# ============================================================

def compare_clusters(
    cluster_a: int,
    cluster_b: int,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> Dict:

    a = get_cluster_evidence(
        cluster_a,
        evidence_path=
            evidence_path,
    )

    b = get_cluster_evidence(
        cluster_b,
        evidence_path=
            evidence_path,
    )

    comparison = {
        "cluster_a":
            a,

        "cluster_b":
            b,

        "differences": {
            "visibility_pattern":
                {
                    "cluster_a":
                        a.get(
                            "visibility_pattern"
                        ),

                    "cluster_b":
                        b.get(
                            "visibility_pattern"
                        ),
                },

            "dominant_preferred_scale":
                {
                    "cluster_a":
                        a.get(
                            "dominant_preferred_scale"
                        ),

                    "cluster_b":
                        b.get(
                            "dominant_preferred_scale"
                        ),
                },

            "rna_expression_relative":
                {
                    "cluster_a":
                        a.get(
                            "rna_expression_relative"
                        ),

                    "cluster_b":
                        b.get(
                            "rna_expression_relative"
                        ),
                },

            "rna_tau_specificity_relative":
                {
                    "cluster_a":
                        a.get(
                            "rna_tau_specificity_relative"
                        ),

                    "cluster_b":
                        b.get(
                            "rna_tau_specificity_relative"
                        ),
                },

            "hallmark_terms":
                {
                    "cluster_a":
                        a.get(
                            "significant_hallmark_terms"
                        ),

                    "cluster_b":
                        b.get(
                            "significant_hallmark_terms"
                        ),
                },

            "go_reactome_terms":
                {
                    "cluster_a":
                        a.get(
                            "significant_go_reactome_terms"
                        ),

                    "cluster_b":
                        b.get(
                            "significant_go_reactome_terms"
                        ),
                },

            "evidence_strength":
                {
                    "cluster_a":
                        a.get(
                            "evidence_strength"
                        ),

                    "cluster_b":
                        b.get(
                            "evidence_strength"
                        ),
                },
        },
    }

    return comparison


def build_evidence_confidence(
    evidence: Dict,
) -> Dict:
    """
    Summarize confidence of biological evidence.
    """

    supporting = []

    limitations = [
        "Exploratory association",
        "Same primary cohort analysis",
        "Does not establish biological causality",
    ]


    if evidence.get(
        "visibility_pattern"
    ):

        supporting.append(
            "Cross-scale visibility phenotype"
        )


    # --------------------------------------------------------
    # Pathway enrichment evidence
    # --------------------------------------------------------

    if (
        evidence.get(
            "significant_hallmark_terms"
        )
        or
        evidence.get(
            "significant_go_reactome_terms"
        )
        or
        evidence.get(
            "top_hallmark_term"
        )
        or
        evidence.get(
            "top_go_reactome_term"
        )
    ):

        supporting.append(
            "Pathway enrichment evidence"
        )


    # --------------------------------------------------------
    # RNA characterization evidence
    # --------------------------------------------------------

    if (
        evidence.get(
            "rna_mean_expression_median"
        )
        is not None
        or
        evidence.get(
            "rna_tau_specificity_relative"
        )
        is not None
        or
        evidence.get(
            "rna_expression_relative"
        )
    ):

        supporting.append(
            "RNA characterization"
        )


    if len(supporting) >= 3:

        level = "moderate"

    elif len(supporting) == 2:

        level = "limited"

    else:

        level = "weak"


    return {

        "level":
            level,

        "supporting_evidence":
            supporting,

        "limitations":
            limitations,
    }




# ============================================================
# Compact evidence payload for Agent
# ============================================================

def build_cluster_biological_evidence_response(
    cluster_id: int,
    evidence_path: Optional[
        str | Path
    ] = None,
) -> Dict:

    evidence = get_cluster_evidence(
        cluster_id,
        evidence_path=
            evidence_path,
    )

    return {
        "evidence_type":
            "cluster_biological_evidence",

        "confidence":
            build_evidence_confidence(
                evidence
            ),

        "fine_cluster":
            evidence.get(
                "fine_cluster"
            ),

        "n_genes":
            evidence.get(
                "n_genes"
            ),

        "visibility_pattern":
            evidence.get(
                "visibility_pattern"
            ),

        "dominant_preferred_scale":
            evidence.get(
                "dominant_preferred_scale"
            ),

        "visibility": {
            "2p5x":
                evidence.get(
                    "median_pearson_2p5x"
                ),

            "10x":
                evidence.get(
                    "median_pearson_10x"
                ),

            "40x":
                evidence.get(
                    "median_pearson_40x"
                ),
        },

        "rna": {
            "expression_relative":
                evidence.get(
                    "rna_expression_relative"
                ),

            "tau_specificity_relative":
                evidence.get(
                    "rna_tau_specificity_relative"
                ),

            "summary":
                evidence.get(
                    "rna_summary"
                ),
        },

        "biological_programs": {
            "hallmark":
                evidence.get(
                    "significant_hallmark_terms"
                ),

            "go_reactome":
                evidence.get(
                    "significant_go_reactome_terms"
                ),
        },

        "evidence_strength":
            evidence.get(
                "evidence_strength"
            ),

        "scientific_scope": (
            "This evidence summarizes exploratory "
            "associations between cross-scale "
            "morphology-derived molecular visibility "
            "patterns and molecular characteristics "
            "within the same primary 63-case cohort. "
            "It does not establish biological "
            "mechanism, causality, cell identity, "
            "clinical biomarker validity, or "
            "external generalization."
        ),
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Query deterministic biological "
            "evidence for molecular visibility "
            "pattern clusters."
        )
    )

    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--scale",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--program",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Two cluster IDs, e.g. 2,5",
    )

    parser.add_argument(
        "--evidence-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.cluster is not None:

        result = get_cluster_evidence(
            args.cluster,
            evidence_path=
                args.evidence_path,
        )

        print(
            pd.Series(
                result
            ).to_string()
        )

        return

    if args.scale is not None:

        result = find_clusters_by_scale(
            args.scale,
            evidence_path=
                args.evidence_path,
        )

        print(
            result.to_string(
                index=False
            )
        )

        return

    if args.pattern is not None:

        result = (
            find_clusters_by_visibility_pattern(
                args.pattern,
                evidence_path=
                    args.evidence_path,
            )
        )

        print(
            result.to_string(
                index=False
            )
        )

        return

    if args.program is not None:

        result = find_clusters_by_program(
            args.program,
            evidence_path=
                args.evidence_path,
        )

        print(
            result.to_string(
                index=False
            )
        )

        return

    if args.compare is not None:

        parts = [
            item.strip()
            for item
            in args.compare.split(",")
            if item.strip()
        ]

        if len(parts) != 2:

            raise ValueError(
                "--compare requires exactly "
                "two cluster IDs, e.g. 2,5"
            )

        result = compare_clusters(
            int(
                parts[0]
            ),
            int(
                parts[1]
            ),
            evidence_path=
                args.evidence_path,
        )

        print(
            result
        )

        return

    df = load_cluster_biological_evidence(
        args.evidence_path
    )

    display_columns = [
        "fine_cluster",
        "visibility_pattern",
        "top_hallmark_term",
        "top_go_reactome_term",
        "evidence_strength",
    ]

    display_columns = [
        column
        for column
        in display_columns
        if column in df.columns
    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
