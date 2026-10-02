
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathbioagent.tools.cluster_biological_evidence import (
    classify_evidence_strength,
    compare_clusters,
    find_clusters_by_program,
    find_clusters_by_scale,
    find_clusters_by_visibility_pattern,
    get_cluster_evidence,
    load_cluster_biological_evidence,
)


# ============================================================
# Test fixture
# ============================================================

@pytest.fixture
def evidence_table_path(
    tmp_path: Path,
) -> Path:

    df = pd.DataFrame(
        [
            {
                "fine_cluster": 0,
                "n_genes": 957,
                "visibility_pattern":
                    "10x-peaked / lower visibility",
                "dominant_preferred_scale":
                    "10x",
                "median_pearson_2p5x":
                    0.042,
                "median_pearson_10x":
                    0.145,
                "median_pearson_40x":
                    0.064,
                "rna_expression_relative":
                    "higher",
                "rna_tau_specificity_relative":
                    "intermediate",
                "n_significant_hallmark_terms":
                    0,
                "top_hallmark_term":
                    "",
                "significant_hallmark_terms":
                    "",
                "n_significant_go_reactome_terms":
                    1,
                "top_go_reactome_term":
                    (
                        "Regulation Of Blood Vessel "
                        "Endothelial Cell Migration"
                    ),
                "significant_go_reactome_terms":
                    (
                        "Regulation Of Blood Vessel "
                        "Endothelial Cell Migration"
                    ),
                "agent_summary":
                    (
                        "Cluster 0 is a 10x-peaked "
                        "visibility pattern."
                    ),
            },
            {
                "fine_cluster": 1,
                "n_genes": 1348,
                "visibility_pattern":
                    "low-visibility / scale-flat",
                "dominant_preferred_scale":
                    "40x",
                "median_pearson_2p5x":
                    0.008,
                "median_pearson_10x":
                    0.009,
                "median_pearson_40x":
                    0.014,
                "rna_expression_relative":
                    "intermediate",
                "rna_tau_specificity_relative":
                    "intermediate",
                "n_significant_hallmark_terms":
                    0,
                "top_hallmark_term":
                    "",
                "significant_hallmark_terms":
                    "",
                "n_significant_go_reactome_terms":
                    0,
                "top_go_reactome_term":
                    "",
                "significant_go_reactome_terms":
                    "",
                "agent_summary":
                    (
                        "Cluster 1 is low visibility "
                        "and scale-flat."
                    ),
            },
            {
                "fine_cluster": 2,
                "n_genes": 1098,
                "visibility_pattern":
                    (
                        "high-visibility / "
                        "relatively scale-stable"
                    ),
                "dominant_preferred_scale":
                    "10x",
                "median_pearson_2p5x":
                    0.211,
                "median_pearson_10x":
                    0.235,
                "median_pearson_40x":
                    0.219,
                "rna_expression_relative":
                    "highest",
                "rna_tau_specificity_relative":
                    "lowest",
                "n_significant_hallmark_terms":
                    1,
                "top_hallmark_term":
                    (
                        "Epithelial Mesenchymal "
                        "Transition"
                    ),
                "significant_hallmark_terms":
                    (
                        "Epithelial Mesenchymal "
                        "Transition"
                    ),
                "n_significant_go_reactome_terms":
                    8,
                "top_go_reactome_term":
                    (
                        "Extracellular Matrix "
                        "Organization"
                    ),
                "significant_go_reactome_terms":
                    (
                        "Extracellular Matrix "
                        "Organization; "
                        "Adaptive Immune System; "
                        "Endocytosis"
                    ),
                "agent_summary":
                    (
                        "Cluster 2 shows high "
                        "visibility and relatively "
                        "scale-stable behavior."
                    ),
            },
            {
                "fine_cluster": 3,
                "n_genes": 685,
                "visibility_pattern":
                    "40x-enhanced",
                "dominant_preferred_scale":
                    "40x",
                "median_pearson_2p5x":
                    0.014,
                "median_pearson_10x":
                    0.097,
                "median_pearson_40x":
                    0.163,
                "rna_expression_relative":
                    "intermediate",
                "rna_tau_specificity_relative":
                    "intermediate",
                "n_significant_hallmark_terms":
                    4,
                "top_hallmark_term":
                    "G2-M Checkpoint",
                "significant_hallmark_terms":
                    (
                        "G2-M Checkpoint; "
                        "E2F Targets; "
                        "Glycolysis; "
                        "Mitotic Spindle"
                    ),
                "n_significant_go_reactome_terms":
                    2,
                "top_go_reactome_term":
                    "Cell Cycle, Mitotic",
                "significant_go_reactome_terms":
                    (
                        "Cell Cycle, Mitotic; "
                        "Cell Cycle"
                    ),
                "agent_summary":
                    (
                        "Cluster 3 is 40x-enhanced "
                        "and enriched for mitotic "
                        "programs."
                    ),
            },
            {
                "fine_cluster": 4,
                "n_genes": 460,
                "visibility_pattern":
                    "high-magnification enhanced",
                "dominant_preferred_scale":
                    "10x",
                "median_pearson_2p5x":
                    0.007,
                "median_pearson_10x":
                    0.254,
                "median_pearson_40x":
                    0.236,
                "rna_expression_relative":
                    "intermediate",
                "rna_tau_specificity_relative":
                    "higher",
                "n_significant_hallmark_terms":
                    1,
                "top_hallmark_term":
                    "Interferon Alpha Response",
                "significant_hallmark_terms":
                    "Interferon Alpha Response",
                "n_significant_go_reactome_terms":
                    0,
                "top_go_reactome_term":
                    "",
                "significant_go_reactome_terms":
                    "",
                "agent_summary":
                    (
                        "Cluster 4 is a "
                        "high-magnification enhanced "
                        "pattern."
                    ),
            },
            {
                "fine_cluster": 5,
                "n_genes": 452,
                "visibility_pattern":
                    "2.5x-enhanced",
                "dominant_preferred_scale":
                    "2p5x",
                "median_pearson_2p5x":
                    0.138,
                "median_pearson_10x":
                    -0.012,
                "median_pearson_40x":
                    -0.018,
                "rna_expression_relative":
                    "lowest",
                "rna_tau_specificity_relative":
                    "highest",
                "n_significant_hallmark_terms":
                    0,
                "top_hallmark_term":
                    "",
                "significant_hallmark_terms":
                    "",
                "n_significant_go_reactome_terms":
                    0,
                "top_go_reactome_term":
                    "",
                "significant_go_reactome_terms":
                    "",
                "agent_summary":
                    (
                        "Cluster 5 is "
                        "2.5x-enhanced."
                    ),
            },
        ]
    )

    path = (
        tmp_path
        / "cluster_biological_evidence.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# Loading
# ============================================================

def test_load_cluster_biological_evidence(
    evidence_table_path,
):

    df = load_cluster_biological_evidence(
        evidence_table_path
    )

    assert len(df) == 6

    assert set(
        df["fine_cluster"]
    ) == {
        0,
        1,
        2,
        3,
        4,
        5,
    }

    assert (
        df.loc[
            df["fine_cluster"] == 3,
            "visibility_pattern",
        ].iloc[0]
        == "40x-enhanced"
    )


# ============================================================
# Direct cluster retrieval
# ============================================================

def test_get_cluster_evidence(
    evidence_table_path,
):

    result = get_cluster_evidence(
        3,
        evidence_path=
            evidence_table_path,
    )

    assert (
        result["fine_cluster"]
        == 3
    )

    assert (
        result["visibility_pattern"]
        == "40x-enhanced"
    )

    assert (
        result["top_hallmark_term"]
        == "G2-M Checkpoint"
    )

    assert (
        result["top_go_reactome_term"]
        == "Cell Cycle, Mitotic"
    )


def test_get_cluster_evidence_missing_cluster(
    evidence_table_path,
):

    with pytest.raises(
        ValueError
    ):

        get_cluster_evidence(
            99,
            evidence_path=
                evidence_table_path,
        )


# ============================================================
# Evidence-strength classification
# ============================================================

def test_classify_evidence_strength():

    assert (
        classify_evidence_strength(
            n_hallmark_terms=1,
            n_go_reactome_terms=8,
        )
        ==
        "multi_source_enrichment_supported"
    )

    assert (
        classify_evidence_strength(
            n_hallmark_terms=4,
            n_go_reactome_terms=2,
        )
        ==
        "multi_source_enrichment_supported"
    )

    assert (
        classify_evidence_strength(
            n_hallmark_terms=1,
            n_go_reactome_terms=0,
        )
        ==
        "single_source_enrichment_supported"
    )

    assert (
        classify_evidence_strength(
            n_hallmark_terms=0,
            n_go_reactome_terms=1,
        )
        ==
        "single_source_enrichment_supported"
    )

    assert (
        classify_evidence_strength(
            n_hallmark_terms=0,
            n_go_reactome_terms=0,
        )
        ==
        (
            "rna_characterized_without_"
            "global_fdr_enrichment"
        )
    )


# ============================================================
# Search by scale
# ============================================================

def test_find_clusters_by_scale(
    evidence_table_path,
):

    result = find_clusters_by_scale(
        "40x",
        evidence_path=
            evidence_table_path,
    )

    cluster_ids = set(
        result[
            "fine_cluster"
        ]
    )

    assert 3 in cluster_ids

    assert (
        result.iloc[
            0
        ][
            "visibility_pattern"
        ]
        == "40x-enhanced"
    )


# ============================================================
# Search by visibility pattern
# ============================================================

def test_find_clusters_by_visibility_pattern(
    evidence_table_path,
):

    result = (
        find_clusters_by_visibility_pattern(
            "scale stable",
            evidence_path=
                evidence_table_path,
        )
    )

    assert len(
        result
    ) == 1

    assert (
        result.iloc[
            0
        ][
            "fine_cluster"
        ]
        == 2
    )

    result_40x = (
        find_clusters_by_visibility_pattern(
            "40X enhanced",
            evidence_path=
                evidence_table_path,
        )
    )

    assert (
        result_40x.iloc[
            0
        ][
            "fine_cluster"
        ]
        == 3
    )


# ============================================================
# Search by biological program
# ============================================================

def test_find_clusters_by_program(
    evidence_table_path,
):

    cell_cycle = (
        find_clusters_by_program(
            "cell cycle",
            evidence_path=
                evidence_table_path,
        )
    )

    assert len(
        cell_cycle
    ) >= 1

    assert (
        3
        in set(
            cell_cycle[
                "fine_cluster"
            ]
        )
    )

    interferon = (
        find_clusters_by_program(
            "interferon alpha",
            evidence_path=
                evidence_table_path,
        )
    )

    assert set(
        interferon[
            "fine_cluster"
        ]
    ) == {
        4
    }

    extracellular = (
        find_clusters_by_program(
            "extracellular matrix",
            evidence_path=
                evidence_table_path,
        )
    )

    assert set(
        extracellular[
            "fine_cluster"
        ]
    ) == {
        2
    }


# ============================================================
# Cluster comparison
# ============================================================

def test_compare_clusters(
    evidence_table_path,
):

    result = compare_clusters(
        2,
        5,
        evidence_path=
            evidence_table_path,
    )

    assert (
        result[
            "cluster_a"
        ][
            "fine_cluster"
        ]
        == 2
    )

    assert (
        result[
            "cluster_b"
        ][
            "fine_cluster"
        ]
        == 5
    )

    assert (
        result[
            "cluster_a"
        ][
            "visibility_pattern"
        ]
        ==
        (
            "high-visibility / "
            "relatively scale-stable"
        )
    )

    assert (
        result[
            "cluster_b"
        ][
            "visibility_pattern"
        ]
        ==
        "2.5x-enhanced"
    )

    assert (
        result[
            "cluster_a"
        ][
            "evidence_strength"
        ]
        ==
        "multi_source_enrichment_supported"
    )

    assert (
        result[
            "cluster_b"
        ][
            "evidence_strength"
        ]
        ==
        (
            "rna_characterized_without_"
            "global_fdr_enrichment"
        )
    )


# ============================================================
# Empty search behavior
# ============================================================

def test_search_returns_empty_dataframe_for_no_match(
    evidence_table_path,
):

    result = find_clusters_by_program(
        "definitely_not_a_real_program",
        evidence_path=
            evidence_table_path,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert result.empty
