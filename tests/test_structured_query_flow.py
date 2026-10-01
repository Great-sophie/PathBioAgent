from pathbioagent.llm.schemas import (
    QueryInterpretation,
)

import pathbioagent.nodes as nodes


# ============================================================
# Helpers
# ============================================================

def make_structured_query(
    *,
    intent="robustness",
    target_scale="10x",
):
    """
    Build a synthetic structured query for tests.
    """

    return {
        "intent":
            intent,

        "target_scale":
            target_scale,

        "entity":
            "molecular_program",

        "confidence":
            0.95,

        "parser_source":
            "llm",

        "model_name":
            "test-model",

        "explicit_scale_guard_applied":
            False,
    }


# ============================================================
# 1. Query interpreter node
# ============================================================

def test_interpret_query_node_stores_structured_query(
    monkeypatch,
):
    """
    interpret_query_node() should store the validated
    structured interpretation in LangGraph state.
    """

    fake_result = QueryInterpretation(
        intent="robustness",
        target_scale="10x",
        entity="molecular_program",
        confidence=0.95,
        parser_source="llm",
        model_name="test-model",
        explicit_scale_guard_applied=False,
    )

    monkeypatch.setattr(
        nodes,
        "interpret_query",
        lambda query: fake_result,
    )

    result = nodes.interpret_query_node(
        {
            "query":
                "Which programs remain stable at 10x?",

            "trace":
                [],
        }
    )

    structured = result[
        "structured_query"
    ]

    assert (
        structured["intent"]
        ==
        "robustness"
    )

    assert (
        structured["target_scale"]
        ==
        "10x"
    )

    assert (
        structured["entity"]
        ==
        "molecular_program"
    )

    assert (
        structured["parser_source"]
        ==
        "llm"
    )

    assert (
        structured["confidence"]
        ==
        0.95
    )

    assert any(
        "Query interpretation"
        in message
        for message in result["trace"]
    )


# ============================================================
# 2. Structured interpretation overrides raw text
# ============================================================

def test_parse_query_prefers_structured_interpretation():
    """
    Raw query deliberately says:
        preference at 40x

    Structured interpretation deliberately says:
        robustness at 10x

    The structured interpretation must win.
    """

    state = {
        "query":
            "Which programs prefer 40x?",

        "structured_query":
            make_structured_query(
                intent="robustness",
                target_scale="10x",
            ),

        "trace":
            [],

        "max_iterations":
            10,
    }

    result = nodes.parse_query(
        state
    )

    assert (
        result["task_type"]
        ==
        "robustness_scale_specific_program_search"
    )

    assert (
        result["max_iterations"]
        ==
        10
    )

    assert any(
        "ranking_intent=robustness"
        in message
        for message in result["trace"]
    )

    assert any(
        "target_scale=10x"
        in message
        for message in result["trace"]
    )


# ============================================================
# 3. Candidate search receives structured intent and scale
# ============================================================

def test_candidate_search_uses_structured_query(
    monkeypatch,
):
    """
    search_candidates() must forward the structured
    interpretation into deterministic candidate ranking.
    """

    captured = {}

    def fake_search(
        query,
        *,
        ranking_intent,
        target_scale,
    ):
        captured[
            "query"
        ] = query

        captured[
            "ranking_intent"
        ] = ranking_intent

        captured[
            "target_scale"
        ] = target_scale

        return [
            "Program_A",
            "Program_B",
        ]

    monkeypatch.setattr(
        nodes,
        "search_program_candidates",
        fake_search,
    )

    state = {
        # Deliberately conflicts with structured query
        "query":
            "Which programs prefer 40x?",

        "structured_query":
            make_structured_query(
                intent="robustness",
                target_scale="10x",
            ),

        "trace":
            [],
    }

    result = nodes.search_candidates(
        state
    )

    assert (
        captured["ranking_intent"]
        ==
        "robustness"
    )

    assert (
        captured["target_scale"]
        ==
        "10x"
    )

    assert (
        result["candidate_queue"]
        ==
        [
            "Program_A",
            "Program_B",
        ]
    )


# ============================================================
# 4. Evidence gate receives structured target scale
# ============================================================

def test_evidence_gate_uses_structured_target_scale(
    monkeypatch,
):
    """
    evidence_gate() must use target_scale from structured_query
    rather than re-parsing the raw user text.
    """

    captured = {}

    def fake_evaluate(
        evidence,
        target_scale=None,
    ):
        captured[
            "target_scale"
        ] = target_scale

        return {
            "status":
                "supported_exploratory_hypothesis",

            "consensus_ok":
                True,

            "target_ok":
                True,

            "robust_ok":
                True,

            "evidence_conflict":
                False,

            "reasons":
                [
                    "Synthetic support."
                ],
        }

    monkeypatch.setattr(
        nodes,
        "evaluate_program_evidence",
        fake_evaluate,
    )

    state = {
        # Raw query says 40x
        "query":
            "Which programs prefer 40x?",

        # Structured interpretation says 10x
        "structured_query":
            make_structured_query(
                intent="robustness",
                target_scale="10x",
            ),

        "current_candidate":
            "Program_A",

        "evidence": {
            "Program_A": {
                "program":
                    "Program_A",
            }
        },

        "trace":
            [],
    }

    result = nodes.evidence_gate(
        state
    )

    assert (
        captured["target_scale"]
        ==
        "10x"
    )

    assert (
        result["current_decision"]
        ==
        "supported_exploratory_hypothesis"
    )

    decision = (
        result["evidence"]
        ["Program_A"]
        ["decision"]
    )

    assert (
        decision["status"]
        ==
        "supported_exploratory_hypothesis"
    )


# ============================================================
# 5. Explicit structured None must be preserved
# ============================================================

def test_structured_none_scale_is_preserved():
    """
    target_scale=None is a valid structured interpretation.

    It means:
        no target magnification was requested.

    The raw query must NOT override this value.
    """

    state = {
        # Raw query deliberately contains 40x
        "query":
            "Which programs prefer 40x?",

        "structured_query":
            make_structured_query(
                intent="general",
                target_scale=None,
            ),
    }

    (
        target_scale,
        ranking_intent,
        entity,
    ) = nodes._resolve_query_context(
        state
    )

    assert target_scale is None

    assert (
        ranking_intent
        ==
        "general"
    )

    assert (
        entity
        ==
        "molecular_program"
    )