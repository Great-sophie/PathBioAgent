from pathbioagent.graph import build_graph


def test_cluster_comparison_agent():

    graph = build_graph()


    result = graph.invoke(
        {
            "query":
                "Compare cluster 3 and cluster 5 biologically.",

            "trace":
                [],

            "iteration":
                0,

            "max_iterations":
                1,
        }
    )


    assert (
        result.get(
            "cluster_comparison_evidence"
        )
        is not None
    )


    assert (
        result[
            "cluster_comparison_evidence"
        ][
            "evidence_type"
        ]
        ==
        "cluster_biological_comparison"
    )
