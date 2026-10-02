"""
Tests for interactive PathBioAgent chat runner.

The tests validate that the chat layer correctly connects
user queries with the evidence-grounded agent workflow.
"""


from pathlib import Path
import sys


# ------------------------------------------------------------
# Add project root
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from pathbioagent.graph import build_graph



# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------


def run_chat_query(query):

    graph = build_graph()

    result = graph.invoke(
        {
            "query":
                query,

            "max_iterations":
                20,

            "trace":
                [],
        }
    )


    return result.get(
        "final_report",
        result,
    )



# ============================================================
# Tests
# ============================================================


def test_chat_agent_returns_report():

    report = run_chat_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert report is not None


    assert (
        "query"
        in report
    )



def test_chat_visibility_query_type():

    report = run_chat_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert (
        report[
            "ranking_intent"
        ]
        ==
        "visibility"
    )


    assert (
        report[
            "target_scale"
        ]
        ==
        "10x"
    )



def test_chat_visibility_contains_evidence_cards():

    report = run_chat_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    cards = report[
        "evidence_cards"
    ]


    assert isinstance(
        cards,
        list,
    )


    assert len(cards) > 0



def test_chat_preference_query():

    report = run_chat_query(
        "Which programs prefer 10x?"
    )


    assert (
        report[
            "ranking_intent"
        ]
        ==
        "preference"
    )


    assert (
        "supported_exploratory_hypotheses"
        in report
    )



def test_chat_agent_separates_visibility_and_preference():

    visibility_report = run_chat_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    preference_report = run_chat_query(
        "Which programs prefer 10x?"
    )


    assert (
        visibility_report[
            "ranking_intent"
        ]
        !=
        preference_report[
            "ranking_intent"
        ]
    )



def test_chat_agent_trace_available():

    report = run_chat_query(
        "Which programs show the highest molecular visibility at 10x?"
    )


    assert (
        "trace"
        in report
    )
