from pathbioagent.graph import build_graph


def test_biological_pattern_cluster_query_end_to_end():

    graph = build_graph()


    state = {

        "query":
            "What is the biological evidence for cluster 3?",

        "trace": [],

        "iteration": 0,

        "max_iterations": 3,

        "tested_candidates": [],

        "candidate_queue": [],

        "supported": [],

        "insufficient": [],

        "not_supported": [],

        "unknown_status": [],

    }


    result = graph.invoke(
        state
    )


    assert (
        result["structured_query"]["intent"]
        ==
        "biological_pattern"
    )


    assert (
        result.get(
            "biological_pattern_evidence"
        )
        is not None
    )


    evidence = result[
        "biological_pattern_evidence"
    ]


    assert (
        evidence["evidence_type"]
        ==
        "cluster_biological_evidence"
    )


    assert (
        evidence["fine_cluster"]
        ==
        3
    )


    assert (
        evidence["visibility_pattern"]
        ==
        "40x-enhanced"
    )


    assert (
        evidence["dominant_preferred_scale"]
        ==
        "40x"
    )


def test_biological_pattern_program_query_end_to_end():

    graph = build_graph()


    state = {

        "query":
            "Which visibility pattern is associated with cell cycle?",

        "trace": [],

        "iteration": 0,

        "max_iterations": 3,

        "tested_candidates": [],

        "candidate_queue": [],

        "supported": [],

        "insufficient": [],

        "not_supported": [],

        "unknown_status": [],

    }


    result = graph.invoke(
        state
    )


    assert (
        result["structured_query"]["intent"]
        ==
        "biological_pattern"
    )


    assert (
        result.get(
            "biological_pattern_evidence"
        )
        is not None
    )
