from typing import Literal

from langgraph.graph import StateGraph, START, END

from pathbioagent.state import BiomarkerAgentState

from pathbioagent.nodes import (
    interpret_query_node,
    parse_query,
    search_candidates,
    select_candidate,
    test_candidate,
    evidence_gate,
    build_report,
    discovery_evidence_node,
    gene_explanation_node,
    cluster_comparison_node,
    biological_pattern_evidence_node,
)





def route_after_interpret_query(
    state: BiomarkerAgentState,
) -> Literal[
    "discovery_evidence",
    "gene_explanation",
    "biological_pattern_evidence",
    "cluster_comparison",
    "parse_query",
]:
    """
    Route query interpretation results
    to scientific evidence modules.
    """

    structured = state.get(
        "structured_query",
        {},
    )

    intent = structured.get(
        "intent"
    )


    if intent == "cluster_comparison":
        return "cluster_comparison"


    if intent == "gene_explanation":
        return "gene_explanation"


    if intent == "biological_pattern":
        return "biological_pattern_evidence"


    if intent == "discovery":
        return "discovery_evidence"


    return "parse_query"


def route_after_evidence_gate(
    state: BiomarkerAgentState,
) -> Literal[
    "select_candidate",
    "build_report",
]:
    """
    Decide whether another candidate should be tested
    or the final report should be generated.
    """

    tested = state.get(
        "tested_candidates",
        [],
    )

    candidates = state.get(
        "candidate_queue",
        [],
    )

    iteration = state.get(
        "iteration",
        0,
    )

    max_iterations = state.get(
        "max_iterations",
        10,
    )

    # Safety stop
    if iteration >= max_iterations:
        return "build_report"

    # More candidates remain
    if len(tested) < len(candidates):
        return "select_candidate"

    return "build_report"


def build_graph():
    """
    Build and compile PathBioAgent.

    Workflow:

        user query
            ↓
        structured query interpretation
            ↓
        deterministic candidate search
            ↓
        deterministic evidence collection
            ↓
        deterministic evidence gate
            ↓
        final evidence report
    """

    builder = StateGraph(
        BiomarkerAgentState
    )

    # Nodes
    builder.add_node(
        "interpret_query",
        interpret_query_node,
    )

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

    builder.add_node(
        "discovery_evidence",
        discovery_evidence_node,
    )

    builder.add_node(
        "gene_explanation",
        gene_explanation_node,
    )

    builder.add_node(
        "cluster_comparison",
        cluster_comparison_node,
    )


    builder.add_node(
        "biological_pattern_evidence",
        biological_pattern_evidence_node,
    )

    # Main path
    builder.add_edge(
        START,
        "interpret_query",
    )

    builder.add_conditional_edges(
        "interpret_query",
        route_after_interpret_query,
        {
            "discovery_evidence":
                "discovery_evidence",

            "gene_explanation":
                "gene_explanation",

            "biological_pattern_evidence":
                "biological_pattern_evidence",

            "cluster_comparison":
                "cluster_comparison",

            "parse_query":
                "parse_query",
        },
    )

    builder.add_edge(
        "discovery_evidence",
        "build_report",
    )

    
    builder.add_edge(
        "gene_explanation",
        "build_report",
    )


    builder.add_edge(
        "biological_pattern_evidence",
        "build_report",
    )


    builder.add_edge(
        "cluster_comparison",
        "build_report",
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

    # Candidate loop
    builder.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {
            "select_candidate":
                "select_candidate",

            "build_report":
                "build_report",
        },
    )

    builder.add_edge(
        "build_report",
        END,
    )

    return builder.compile()
