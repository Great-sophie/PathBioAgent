
from pathbioagent.llm.query_interpreter import interpret_query
from pathbioagent.graph import build_graph


def test_discovery_query_intent():
    q = (
        "Which factors explain morphology-derived "
        "molecular visibility?"
    )

    result = interpret_query(q)

    assert result.intent == "discovery"


def test_gene_explanation_query_intent():
    q = "Why is SFTPA1 highly visible?"

    result = interpret_query(q)

    assert result.intent == "gene_explanation"


def test_discovery_graph_route():
    graph = build_graph()

    result = graph.invoke(
        {
            "query": (
                "Which factors explain morphology-derived "
                "molecular visibility?"
            )
        }
    )

    assert result["ranking_intent"] == "discovery"
    assert result.get("discovery_evidence") is not None
    assert result.get("gene_explanation") is None
    assert result.get("gene_reasoning") is None


def test_gene_explanation_graph_route():
    graph = build_graph()

    result = graph.invoke(
        {
            "query": "Why is SFTPA1 highly visible?"
        }
    )

    assert result["ranking_intent"] == "gene_explanation"
    assert result.get("discovery_evidence") is None
    assert result.get("gene_explanation") is not None
    assert result.get("gene_reasoning") is not None

    report = result["final_report"]

    assert report.get("gene_explanation") is not None
    assert report.get("gene_reasoning") is not None
