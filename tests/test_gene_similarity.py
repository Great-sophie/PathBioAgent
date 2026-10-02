
from pathbioagent.tools.gene_similarity import (
    find_similar_genes,
    build_gene_similarity_evidence,
)


def test_gene_similarity_excludes_query_gene():
    result = find_similar_genes(
        "SFTPA1",
        top_k=10,
    )

    assert "SFTPA1" not in set(
        result["gene"]
    )


def test_gene_similarity_sorted_by_distance():
    result = find_similar_genes(
        "SFTPA1",
        top_k=10,
    )

    distances = result[
        "pattern_distance"
    ].tolist()

    assert distances == sorted(
        distances
    )


def test_gene_similarity_expected_columns():
    result = find_similar_genes(
        "SFTPA1",
        top_k=5,
    )

    required = {
        "gene",
        "pattern_distance",
        "shape_distance",
        "visibility_level_difference",
        "preferred_scale_match",
    }

    assert required.issubset(
        result.columns
    )


def test_gene_similarity_evidence_scope():
    evidence = (
        build_gene_similarity_evidence(
            "SFTPA1",
            top_k=5,
        )
    )

    assert (
        evidence["evidence_type"]
        ==
        "gene_visibility_pattern_similarity"
    )

    assert len(
        evidence["similar_genes"]
    ) == 5

    assert (
        "does not establish"
        in evidence["scientific_scope"]
    )
