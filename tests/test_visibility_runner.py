"""
Tests for target-scale visibility query behavior.

Verify that:

1. Visibility queries are treated as ranking tasks.
2. Target-scale visibility is separated from scale preference.
3. Highest visibility program is correctly surfaced.
"""


from pathbioagent.graph import build_graph



def run_agent_query(
    query: str,
):
    """
    Helper function to execute PathBioAgent graph.
    """

    graph = build_graph()

    result = graph.invoke(
        {
            "query": query,

            "max_iterations": 20,

            "trace": [],
        }
    )

    return result.get(
        "final_report",
        result,
    )



# ============================================================
# Test 1
# ============================================================

def test_visibility_query_primary_answer_type():

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert (
        report[
            "primary_answer_type"
        ]
        ==
        "target_scale_visibility_ranking"
    )



# ============================================================
# Test 2
# ============================================================

def test_visibility_query_target_scale():

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert (
        report[
            "target_scale"
        ]
        ==
        "10x"
    )



# ============================================================
# Test 3
# ============================================================

def test_visibility_query_contains_ranking():

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    ranking = report.get(
        "visibility_ranking",
        []
    )


    assert len(
        ranking
    ) > 0



# ============================================================
# Test 4
# ============================================================

def test_visibility_query_top_program():

    """
    Current evidence:

    Antigen presentation:
        highest median 10x visibility

    This verifies ranking behavior,
    not biological causality.
    """

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    ranking = report[
        "visibility_ranking"
    ]


    assert (
        ranking[0][
            "program"
        ]
        ==
        "Antigen presentation"
    )



# ============================================================
# Test 5
# ============================================================

def test_visibility_query_separates_preference_from_visibility():

    """
    A program may have high visibility
    at a target scale without having
    a unique preferred scale.

    Therefore:

        target-scale visibility
                !=
        preferred scale
    """

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    cards = report[
        "evidence_cards"
    ]


    antigen = None


    for card in cards:

        if (
            card[
                "program"
            ]
            ==
            "Antigen presentation"
        ):

            antigen = card

            break


    assert antigen is not None



    # Target magnification comes from query
    assert (
        report[
            "target_scale"
        ]
        ==
        "10x"
    )


    # Visibility value exists
    assert (
        antigen[
            "target_scale_visibility"
        ][
            "median_across_models"
        ]
        >
        0
    )


    # Preferred scale can still be different
    assert (
        antigen[
            "vote_preferred_scale"
        ]
        ==
        "tie"
    )



# ============================================================
# Test 6
# ============================================================

def test_visibility_query_has_metric_definition():

    report = run_agent_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert (
        report[
            "visibility_ranking_metric"
        ]
        ==
        "median_across_models"
    )
