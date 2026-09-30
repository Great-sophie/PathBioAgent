from typing import Literal

from langgraph.graph import StateGraph, START, END

from pathbioagent.state import BiomarkerAgentState
from pathbioagent.nodes import (
    parse_query,
    search_candidates,
    select_candidate,
    test_candidate,
    evidence_gate,
    build_report,
)


def route_after_evidence_gate(
    state: BiomarkerAgentState,
) -> Literal["select_candidate", "build_report"]:
    """
    Decide whether the agent should test another candidate
    or finish and build the final report.
    """

    tested = state.get("tested_candidates", [])
    candidates = state.get("candidate_queue", [])

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Safety stop
    if iteration >= max_iterations:
        return "build_report"

    # More candidates remain
    if len(tested) < len(candidates):
        return "select_candidate"

    # All candidates tested
    return "build_report"


def build_graph():
    """
    Build and compile the PathBioAgent LangGraph.
    """

    builder = StateGraph(BiomarkerAgentState)

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------

    builder.add_node(
        "parse_query",
        parse_query,
    )

    builder.add_node(
        "search_candidates",
        search_candidates,
    )

    builder.add_node(
        "select_candidate",
        select_candidate,
    )

    builder.add_node(
        "test_candidate",
        test_candidate,
    )

    builder.add_node(
        "evidence_gate",
        evidence_gate,
    )

    builder.add_node(
        "build_report",
        build_report,
    )

    # --------------------------------------------------
    # Graph edges
    # --------------------------------------------------

    builder.add_edge(
        START,
        "parse_query",
    )

    builder.add_edge(
        "parse_query",
        "search_candidates",
    )

    builder.add_edge(
        "search_candidates",
        "select_candidate",
    )

    builder.add_edge(
        "select_candidate",
        "test_candidate",
    )

    builder.add_edge(
        "test_candidate",
        "evidence_gate",
    )

    # --------------------------------------------------
    # Conditional loop
    # --------------------------------------------------

    builder.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "select_candidate": "select_candidate",
            "build_report": "build_report",
        },
    )

    builder.add_edge(
        "build_report",
        END,
    )

    # --------------------------------------------------
    # Compile
    # --------------------------------------------------

    graph = builder.compile()

    return graph
