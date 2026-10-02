from pathbioagent.tools.cluster_biological_comparison import (
    compare_cluster_biology,
)


def test_compare_clusters():

    result = compare_cluster_biology(
        3,
        5,
    )


    assert (
        result["evidence_type"]
        ==
        "cluster_biological_comparison"
    )


    assert (
        result["cluster_a"]["fine_cluster"]
        ==
        3
    )


    assert (
        result["cluster_b"]["fine_cluster"]
        ==
        5
    )


    assert (
        len(
            result["key_differences"]
        )
        > 0
    )
