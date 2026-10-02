
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact


# ============================================================
# Configuration
# ============================================================

DEFAULT_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
    "Reactome_2022",
]

DEFAULT_ORGANISM = "Human"

DEFAULT_ALPHA = 0.05

DEFAULT_MIN_GENE_SET_SIZE = 10

DEFAULT_MAX_GENE_SET_SIZE = 500

DEFAULT_MIN_OVERLAP = 3


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


def _default_assignment_path() -> Path:

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
        / "biological_enrichment"
    )


# ============================================================
# Gene normalization
# ============================================================

def normalize_gene(
    gene: str,
) -> str:

    return (
        str(gene)
        .strip()
        .upper()
    )


def normalize_gene_set(
    genes: Iterable[str],
) -> set[str]:

    return {
        normalize_gene(gene)
        for gene in genes
        if str(gene).strip()
    }


# ============================================================
# BH correction
# ============================================================

def benjamini_hochberg(
    p_values: Sequence[float],
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    if n == 0:
        return np.asarray(
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
# Load cluster assignments
# ============================================================

def load_assignments(
    assignment_path: Optional[
        str | Path
    ] = None,
) -> pd.DataFrame:

    path = (
        Path(
            assignment_path
        )
        if assignment_path is not None
        else _default_assignment_path()
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cluster assignment file not found: "
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "gene",
        "fine_cluster",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = (
        df[
            required
        ]
        .dropna(
            subset=[
                "gene",
                "fine_cluster",
            ]
        )
        .copy()
    )

    df[
        "gene"
    ] = (
        df[
            "gene"
        ]
        .map(
            normalize_gene
        )
    )

    df[
        "fine_cluster"
    ] = (
        df[
            "fine_cluster"
        ]
        .astype(int)
    )

    df = (
        df.drop_duplicates(
            subset=[
                "gene"
            ]
        )
    )

    return df


# ============================================================
# Local GMT support
# ============================================================

def parse_gmt(
    path: str | Path,
) -> Dict[str, List[str]]:

    path = Path(
        path
    )

    gene_sets: Dict[
        str,
        List[str],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            parts = (
                line.rstrip(
                    "\n"
                )
                .split(
                    "\t"
                )
            )

            if len(parts) < 3:
                continue

            term = parts[
                0
            ]

            genes = parts[
                2:
            ]

            gene_sets[
                term
            ] = genes

    return gene_sets


# ============================================================
# GSEApy / Enrichr library retrieval
# ============================================================

def fetch_gene_set_library(
    library: str,
    organism: str = DEFAULT_ORGANISM,
    gmt_dir: Optional[
        str | Path
    ] = None,
) -> Dict[
    str,
    set[str],
]:

    # --------------------------------------------------------
    # Prefer local GMT for reproducibility
    # --------------------------------------------------------

    if gmt_dir is not None:

        gmt_dir = Path(
            gmt_dir
        )

        candidates = [
            gmt_dir
            / f"{library}.gmt",

            gmt_dir
            / library,
        ]

        for candidate in candidates:

            if candidate.exists():

                raw = parse_gmt(
                    candidate
                )

                return {
                    term:
                        normalize_gene_set(
                            genes
                        )
                    for term, genes
                    in raw.items()
                }

    # --------------------------------------------------------
    # Otherwise retrieve through GSEApy
    # --------------------------------------------------------

    try:

        import gseapy as gp

    except ImportError as exc:

        raise RuntimeError(
            "gseapy is required to download "
            "Enrichr gene-set libraries. "
            "Install it with:\n"
            "python -m pip install gseapy\n"
            "Alternatively provide local GMT files "
            "with --gmt-dir."
        ) from exc

    print(
        f"Loading gene-set library: "
        f"{library}"
    )

    raw = gp.get_library(
        name=library,
        organism=organism,
    )

    if not raw:

        raise RuntimeError(
            f"Gene-set library returned no terms: "
            f"{library}"
        )

    gene_sets = {
        term:
            normalize_gene_set(
                genes
            )
        for term, genes
        in raw.items()
    }

    return gene_sets


# ============================================================
# Restrict library to study universe
# ============================================================

def restrict_gene_sets_to_universe(
    gene_sets: Dict[
        str,
        set[str],
    ],
    universe: set[str],
    min_size: int,
    max_size: int,
) -> Dict[
    str,
    set[str],
]:

    restricted: Dict[
        str,
        set[str],
    ] = {}

    for term, genes in (
        gene_sets.items()
    ):

        overlap = (
            genes
            & universe
        )

        size = len(
            overlap
        )

        if (
            size < min_size
            or size > max_size
        ):
            continue

        restricted[
            term
        ] = overlap

    return restricted


# ============================================================
# ORA
# ============================================================

def run_cluster_ora(
    cluster_genes: set[str],
    universe: set[str],
    gene_sets: Dict[
        str,
        set[str],
    ],
    cluster_id: int,
    library: str,
    min_overlap: int,
) -> pd.DataFrame:

    cluster_genes = (
        cluster_genes
        & universe
    )

    cluster_size = len(
        cluster_genes
    )

    universe_size = len(
        universe
    )

    rows: List[Dict] = []

    for term, term_genes in (
        gene_sets.items()
    ):

        term_genes = (
            term_genes
            & universe
        )

        overlap_genes = (
            cluster_genes
            & term_genes
        )

        overlap = len(
            overlap_genes
        )

        if overlap < min_overlap:
            continue

        gene_set_size = len(
            term_genes
        )

        a = overlap

        b = (
            cluster_size
            - overlap
        )

        c = (
            gene_set_size
            - overlap
        )

        d = (
            universe_size
            - a
            - b
            - c
        )

        if d < 0:
            continue

        odds_ratio, p_value = (
            fisher_exact(
                [
                    [
                        a,
                        b,
                    ],
                    [
                        c,
                        d,
                    ],
                ],
                alternative="greater",
            )
        )

        expected_overlap = (
            cluster_size
            * gene_set_size
            / universe_size
        )

        fold_enrichment = (
            overlap
            / expected_overlap
            if expected_overlap > 0
            else np.nan
        )

        cluster_gene_fraction = (
            overlap
            / cluster_size
            if cluster_size > 0
            else np.nan
        )

        gene_set_fraction = (
            overlap
            / gene_set_size
            if gene_set_size > 0
            else np.nan
        )

        rows.append(
            {
                "fine_cluster":
                    int(
                        cluster_id
                    ),

                "library":
                    library,

                "term":
                    term,

                "overlap":
                    int(
                        overlap
                    ),

                "cluster_size":
                    int(
                        cluster_size
                    ),

                "gene_set_size_in_universe":
                    int(
                        gene_set_size
                    ),

                "universe_size":
                    int(
                        universe_size
                    ),

                "expected_overlap":
                    float(
                        expected_overlap
                    ),

                "fold_enrichment":
                    float(
                        fold_enrichment
                    ),

                "odds_ratio":
                    float(
                        odds_ratio
                    ),

                "p_value":
                    float(
                        p_value
                    ),

                "cluster_gene_fraction":
                    float(
                        cluster_gene_fraction
                    ),

                "gene_set_fraction":
                    float(
                        gene_set_fraction
                    ),

                "overlap_genes":
                    ";".join(
                        sorted(
                            overlap_genes
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# Multiple-testing correction
# ============================================================

def add_fdr_columns(
    enrichment: pd.DataFrame,
) -> pd.DataFrame:

    if enrichment.empty:

        return enrichment

    enrichment = (
        enrichment.copy()
    )

    enrichment[
        "q_value_cluster_library"
    ] = np.nan

    # --------------------------------------------------------
    # FDR within each cluster × library
    # --------------------------------------------------------

    for (
        cluster,
        library,
    ), indices in (
        enrichment.groupby(
            [
                "fine_cluster",
                "library",
            ]
        )
        .groups
        .items()
    ):

        idx = list(
            indices
        )

        enrichment.loc[
            idx,
            "q_value_cluster_library",
        ] = benjamini_hochberg(
            enrichment.loc[
                idx,
                "p_value",
            ].tolist()
        )

    # --------------------------------------------------------
    # FDR across all enrichment tests
    # --------------------------------------------------------

    enrichment[
        "q_value_global"
    ] = benjamini_hochberg(
        enrichment[
            "p_value"
        ].tolist()
    )

    enrichment[
        "significant_cluster_library_fdr_0p05"
    ] = (
        enrichment[
            "q_value_cluster_library"
        ]
        <
        DEFAULT_ALPHA
    )

    enrichment[
        "significant_global_fdr_0p05"
    ] = (
        enrichment[
            "q_value_global"
        ]
        <
        DEFAULT_ALPHA
    )

    return enrichment


# ============================================================
# Cluster enrichment summary
# ============================================================

def build_cluster_enrichment_summary(
    enrichment: pd.DataFrame,
) -> pd.DataFrame:

    if enrichment.empty:

        return pd.DataFrame()

    rows: List[Dict] = []

    for cluster, group in (
        enrichment.groupby(
            "fine_cluster"
        )
    ):

        significant = group.loc[
            group[
                "significant_global_fdr_0p05"
            ]
        ]

        row = {
            "fine_cluster":
                int(
                    cluster
                ),

            "n_terms_tested":
                int(
                    len(
                        group
                    )
                ),

            "n_significant_global_fdr":
                int(
                    len(
                        significant
                    )
                ),

            "minimum_q_value_global":
                float(
                    group[
                        "q_value_global"
                    ].min()
                ),
        }

        if not significant.empty:

            strongest = (
                significant
                .sort_values(
                    [
                        "q_value_global",
                        "fold_enrichment",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .iloc[
                    0
                ]
            )

            row[
                "top_term"
            ] = strongest[
                "term"
            ]

            row[
                "top_library"
            ] = strongest[
                "library"
            ]

            row[
                "top_fold_enrichment"
            ] = float(
                strongest[
                    "fold_enrichment"
                ]
            )

        else:

            row[
                "top_term"
            ] = None

            row[
                "top_library"
            ] = None

            row[
                "top_fold_enrichment"
            ] = np.nan

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
# Top terms per cluster / library
# ============================================================

def build_top_terms(
    enrichment: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:

    if enrichment.empty:

        return pd.DataFrame()

    rows = []

    for (
        cluster,
        library,
    ), group in enrichment.groupby(
        [
            "fine_cluster",
            "library",
        ]
    ):

        selected = (
            group
            .sort_values(
                [
                    "q_value_global",
                    "fold_enrichment",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(
                top_n
            )
            .copy()
        )

        selected[
            "rank_within_cluster_library"
        ] = np.arange(
            1,
            len(
                selected
            )
            + 1,
        )

        rows.append(
            selected
        )

    if not rows:

        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Main pipeline
# ============================================================

def run_biological_enrichment(
    assignment_path: Optional[
        str | Path
    ] = None,
    output_dir: Optional[
        str | Path
    ] = None,
    libraries: Sequence[
        str
    ] = DEFAULT_LIBRARIES,
    organism: str = DEFAULT_ORGANISM,
    gmt_dir: Optional[
        str | Path
    ] = None,
    min_gene_set_size: int = DEFAULT_MIN_GENE_SET_SIZE,
    max_gene_set_size: int = DEFAULT_MAX_GENE_SET_SIZE,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    top_n: int = 15,
) -> Dict:

    assignments = (
        load_assignments(
            assignment_path
        )
    )

    universe = set(
        assignments[
            "gene"
        ]
    )

    clusters = sorted(
        assignments[
            "fine_cluster"
        ]
        .unique()
    )

    all_results: List[
        pd.DataFrame
    ] = []

    library_summary_rows = []

    for library in libraries:

        raw_gene_sets = (
            fetch_gene_set_library(
                library=library,
                organism=organism,
                gmt_dir=gmt_dir,
            )
        )

        restricted_gene_sets = (
            restrict_gene_sets_to_universe(
                gene_sets=
                    raw_gene_sets,
                universe=
                    universe,
                min_size=
                    min_gene_set_size,
                max_size=
                    max_gene_set_size,
            )
        )

        print(
            f"{library}: "
            f"{len(raw_gene_sets)} raw terms, "
            f"{len(restricted_gene_sets)} "
            f"terms retained in study universe"
        )

        library_summary_rows.append(
            {
                "library":
                    library,

                "n_raw_terms":
                    int(
                        len(
                            raw_gene_sets
                        )
                    ),

                "n_terms_after_universe_filter":
                    int(
                        len(
                            restricted_gene_sets
                        )
                    ),
            }
        )

        for cluster in clusters:

            cluster_genes = set(
                assignments.loc[
                    assignments[
                        "fine_cluster"
                    ]
                    == cluster,
                    "gene",
                ]
            )

            result = run_cluster_ora(
                cluster_genes=
                    cluster_genes,
                universe=
                    universe,
                gene_sets=
                    restricted_gene_sets,
                cluster_id=
                    int(
                        cluster
                    ),
                library=
                    library,
                min_overlap=
                    min_overlap,
            )

            if not result.empty:

                all_results.append(
                    result
                )

    if all_results:

        enrichment = pd.concat(
            all_results,
            ignore_index=True,
        )

        enrichment = (
            add_fdr_columns(
                enrichment
            )
        )

        enrichment = (
            enrichment.sort_values(
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

        enrichment = (
            pd.DataFrame()
        )

    significant = (
        enrichment.loc[
            enrichment[
                "significant_global_fdr_0p05"
            ]
        ].copy()
        if not enrichment.empty
        else pd.DataFrame()
    )

    summary = (
        build_cluster_enrichment_summary(
            enrichment
        )
    )

    top_terms = (
        build_top_terms(
            enrichment=
                enrichment,
            top_n=
                top_n,
        )
    )

    library_summary = (
        pd.DataFrame(
            library_summary_rows
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

    enrichment.to_csv(
        out_dir
        / "enrichment_all.csv",
        index=False,
    )

    significant.to_csv(
        out_dir
        / "enrichment_significant_global_fdr.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "cluster_enrichment_summary.csv",
        index=False,
    )

    top_terms.to_csv(
        out_dir
        / "top_terms_by_cluster.csv",
        index=False,
    )

    library_summary.to_csv(
        out_dir
        / "library_summary.csv",
        index=False,
    )

    metadata = {
        "n_genes_in_background":
            int(
                len(
                    universe
                )
            ),

        "n_clusters":
            int(
                len(
                    clusters
                )
            ),

        "clusters":
            [
                int(
                    cluster
                )
                for cluster
                in clusters
            ],

        "libraries":
            list(
                libraries
            ),

        "organism":
            organism,

        "min_gene_set_size":
            int(
                min_gene_set_size
            ),

        "max_gene_set_size":
            int(
                max_gene_set_size
            ),

        "min_overlap":
            int(
                min_overlap
            ),

        "multiple_testing": {
            "cluster_library_fdr":
                (
                    "Benjamini-Hochberg within "
                    "each fine cluster × library"
                ),

            "global_fdr":
                (
                    "Benjamini-Hochberg across "
                    "all tested cluster-library-term "
                    "combinations"
                ),
        },

        "scientific_scope": (
            "Over-representation analysis compares "
            "each morphology-derived visibility "
            "cluster against the complete set of "
            "genes included in the 5000-gene "
            "discovery universe. Enrichment is "
            "descriptive and hypothesis-generating; "
            "it does not establish biological "
            "mechanism, causality, cell type, "
            "clinical biomarker validity, or "
            "external generalization."
        ),
    }

    (
        out_dir
        / "biological_enrichment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    return {
        "enrichment":
            enrichment,

        "significant":
            significant,

        "summary":
            summary,

        "top_terms":
            top_terms,

        "library_summary":
            library_summary,

        "output_dir":
            str(
                out_dir
            ),
    }


# ============================================================
# CLI helpers
# ============================================================

def _parse_libraries(
    value: str,
) -> List[str]:

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Perform biological enrichment of "
            "cross-scale morphology-derived "
            "molecular visibility clusters."
        )
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
        "--libraries",
        type=str,
        default=",".join(
            DEFAULT_LIBRARIES
        ),
    )

    parser.add_argument(
        "--organism",
        type=str,
        default=DEFAULT_ORGANISM,
    )

    parser.add_argument(
        "--gmt-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing "
            "<library>.gmt files. Local GMT files "
            "are preferred over online retrieval."
        ),
    )

    parser.add_argument(
        "--min-gene-set-size",
        type=int,
        default=
            DEFAULT_MIN_GENE_SET_SIZE,
    )

    parser.add_argument(
        "--max-gene-set-size",
        type=int,
        default=
            DEFAULT_MAX_GENE_SET_SIZE,
    )

    parser.add_argument(
        "--min-overlap",
        type=int,
        default=
            DEFAULT_MIN_OVERLAP,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    libraries = _parse_libraries(
        args.libraries
    )

    result = run_biological_enrichment(
        assignment_path=
            args.assignments,
        output_dir=
            args.output_dir,
        libraries=
            libraries,
        organism=
            args.organism,
        gmt_dir=
            args.gmt_dir,
        min_gene_set_size=
            args.min_gene_set_size,
        max_gene_set_size=
            args.max_gene_set_size,
        min_overlap=
            args.min_overlap,
        top_n=
            args.top_n,
    )

    print(
        "\nLibrary summary:"
    )

    print(
        result[
            "library_summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nCluster enrichment summary:"
    )

    print(
        result[
            "summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nTop enriched terms:"
    )

    if result[
        "top_terms"
    ].empty:

        print(
            "No enrichment results."
        )

    else:

        columns = [
            "fine_cluster",
            "library",
            "term",
            "overlap",
            "fold_enrichment",
            "odds_ratio",
            "q_value_cluster_library",
            "q_value_global",
        ]

        columns = [
            column
            for column
            in columns
            if column
            in result[
                "top_terms"
            ].columns
        ]

        print(
            result[
                "top_terms"
            ][
                columns
            ]
            .head(
                100
            )
            .to_string(
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
