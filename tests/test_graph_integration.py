import pathbioagent.graph as graph_module


# ============================================================
# Synthetic LangGraph nodes
# ============================================================

def fake_parse_query(state):
    trace = list(state.get("trace", []))

    trace.append("parse_query")

    return {
        "query": state["query"],
        "task_type": "synthetic_test",
        "candidate_queue": [],
        "current_candidate": None,
        "tested_candidates": [],
        "evidence": {},
        "current_decision": None,
        "iteration": 0,
        "max_iterations": state.get(
            "max_iterations",
            10,
        ),
        "trace": trace,
        "final_report": None,
    }


def fake_search_candidates(state):
    trace = list(state.get("trace", []))

    trace.append("search_candidates")

    return {
        "candidate_queue": [
            "Program_A",
            "Program_B",
        ],
        "trace": trace,
    }


def fake_select_candidate(state):
    trace = list(state.get("trace", []))

    tested = state.get(
        "tested_candidates",
        [],
    )

    queue = state.get(
        "candidate_queue",
        [],
    )

    candidate = next(
        (
            item
            for item in queue
            if item not in tested
        ),
        None,
    )

    trace.append(
        f"select_candidate:{candidate}"
    )

    return {
        "current_candidate": candidate,
        "trace": trace,
    }


def fake_test_candidate(state):
    trace = list(state.get("trace", []))

    candidate = state[
        "current_candidate"
    ]

    evidence = dict(
        state.get(
            "evidence",
            {},
        )
    )

    evidence[candidate] = {
        "program": candidate,
        "synthetic_score": 1.0,
    }

    tested = list(
        state.get(
            "tested_candidates",
            [],
        )
    )

    tested.append(candidate)

    iteration = (
        state.get(
            "iteration",
            0,
        )
        + 1
    )

    trace.append(
        f"test_candidate:{candidate}"
    )

    return {
        "evidence": evidence,
        "tested_candidates": tested,
        "iteration": iteration,
        "trace": trace,
    }


def fake_evidence_gate(state):
    trace = list(state.get("trace", []))

    candidate = state[
        "current_candidate"
    ]

    evidence = dict(
        state.get(
            "evidence",
            {},
        )
    )

    item = dict(
        evidence[candidate]
    )

    item["decision"] = {
        "status":
            "supported_exploratory_hypothesis",
        "raw_gate_status":
            "supported_exploratory_hypothesis",
        "consensus_ok": True,
        "target_ok": True,
        "robust_ok": True,
        "evidence_conflict": False,
        "reasons": [
            "Synthetic integration evidence."
        ],
    }

    evidence[candidate] = item

    trace.append(
        f"evidence_gate:{candidate}"
    )

    return {
        "evidence": evidence,
        "current_decision":
            "supported_exploratory_hypothesis",
        "trace": trace,
    }


def fake_build_report(state):
    trace = list(state.get("trace", []))

    trace.append("build_report")

    evidence = state.get(
        "evidence",
        {},
    )

    supported = []

    evidence_cards = []

    for program, item in evidence.items():

        decision = item.get(
            "decision",
            {},
        )

        status = decision.get(
            "status",
            "unknown",
        )

        if (
            status
            ==
            "supported_exploratory_hypothesis"
        ):
            supported.append(program)

        evidence_cards.append(
            {
                "program": program,
                "status": status,
                "evidence_flags": {},
                "reasons": decision.get(
                    "reasons",
                    [],
                ),
            }
        )

    report = {
        "query": state.get(
            "query"
        ),
        "candidates_tested": len(
            state.get(
                "tested_candidates",
                [],
            )
        ),
        "total_candidates": len(
            state.get(
                "candidate_queue",
                [],
            )
        ),
        "supported": supported,
        "evidence_cards":
            evidence_cards,
    }

    return {
        "final_report": report,
        "trace": trace,
    }


# ============================================================
# Routing tests
# ============================================================

def test_route_continues_when_candidates_remain():
    state = {
        "tested_candidates": [
            "Program_A",
        ],
        "candidate_queue": [
            "Program_A",
            "Program_B",
        ],
        "iteration": 1,
        "max_iterations": 10,
    }

    route = (
        graph_module
        .route_after_evidence_gate(
            state
        )
    )

    assert route == "select_candidate"


def test_route_builds_report_when_all_candidates_tested():
    state = {
        "tested_candidates": [
            "Program_A",
            "Program_B",
        ],
        "candidate_queue": [
            "Program_A",
            "Program_B",
        ],
        "iteration": 2,
        "max_iterations": 10,
    }

    route = (
        graph_module
        .route_after_evidence_gate(
            state
        )
    )

    assert route == "build_report"


def test_route_respects_iteration_safety_stop():
    state = {
        "tested_candidates": [
            "Program_A",
        ],
        "candidate_queue": [
            "Program_A",
            "Program_B",
            "Program_C",
        ],
        "iteration": 1,
        "max_iterations": 1,
    }

    route = (
        graph_module
        .route_after_evidence_gate(
            state
        )
    )

    assert route == "build_report"


# ============================================================
# Full LangGraph integration
# ============================================================

def test_full_graph_runs_query_to_final_report(
    monkeypatch,
):
    """
    Test the full LangGraph topology using synthetic
    deterministic nodes.

    Scientific evidence calculations are covered
    separately by unit tests. This test verifies
    graph wiring, looping, state propagation,
    termination, and final-report creation.
    """

    # Important:
    # patch graph_module BEFORE build_graph()
    # so LangGraph compiles the synthetic nodes.
    monkeypatch.setattr(
        graph_module,
        "parse_query",
        fake_parse_query,
    )

    monkeypatch.setattr(
        graph_module,
        "search_candidates",
        fake_search_candidates,
    )

    monkeypatch.setattr(
        graph_module,
        "select_candidate",
        fake_select_candidate,
    )

    monkeypatch.setattr(
        graph_module,
        "test_candidate",
        fake_test_candidate,
    )

    monkeypatch.setattr(
        graph_module,
        "evidence_gate",
        fake_evidence_gate,
    )

    monkeypatch.setattr(
        graph_module,
        "build_report",
        fake_build_report,
    )

    graph = graph_module.build_graph()

    result = graph.invoke(
        {
            "query":
                "Which programs prefer 10x?",
            "max_iterations": 10,
            "trace": [],
        }
    )

    # --------------------------------------------------------
    # Graph reached final report
    # --------------------------------------------------------

    assert (
        result["final_report"]
        is not None
    )

    report = result[
        "final_report"
    ]

    # --------------------------------------------------------
    # Both candidates were tested
    # --------------------------------------------------------

    assert (
        result["tested_candidates"]
        ==
        [
            "Program_A",
            "Program_B",
        ]
    )

    assert (
        report["candidates_tested"]
        == 2
    )

    assert (
        report["total_candidates"]
        == 2
    )

    # --------------------------------------------------------
    # Both passed synthetic evidence gate
    # --------------------------------------------------------

    assert (
        report["supported"]
        ==
        [
            "Program_A",
            "Program_B",
        ]
    )

    # --------------------------------------------------------
    # Evidence cards exist
    # --------------------------------------------------------

    assert (
        len(
            report["evidence_cards"]
        )
        == 2
    )

    for card in report[
        "evidence_cards"
    ]:

        assert "program" in card

        assert "status" in card

        assert (
            "evidence_flags"
            in card
        )

        assert "reasons" in card

    # --------------------------------------------------------
    # Graph loop executed exactly twice
    # --------------------------------------------------------

    assert (
        result["iteration"]
        == 2
    )

    # --------------------------------------------------------
    # Trace proves topology was traversed
    # --------------------------------------------------------

    trace = result[
        "trace"
    ]

    assert "parse_query" in trace

    assert (
        "search_candidates"
        in trace
    )

    assert (
        "select_candidate:Program_A"
        in trace
    )

    assert (
        "test_candidate:Program_A"
        in trace
    )

    assert (
        "evidence_gate:Program_A"
        in trace
    )

    assert (
        "select_candidate:Program_B"
        in trace
    )

    assert (
        "test_candidate:Program_B"
        in trace
    )

    assert (
        "evidence_gate:Program_B"
        in trace
    )

    assert "build_report" in trace

    # --------------------------------------------------------
    # Build report must be the final trace event
    # --------------------------------------------------------

    assert (
        trace[-1]
        == "build_report"
    )


def test_graph_stops_at_max_iterations(
    monkeypatch,
):
    """
    Verify the graph cannot loop indefinitely.
    """

    monkeypatch.setattr(
        graph_module,
        "parse_query",
        fake_parse_query,
    )

    monkeypatch.setattr(
        graph_module,
        "search_candidates",
        fake_search_candidates,
    )

    monkeypatch.setattr(
        graph_module,
        "select_candidate",
        fake_select_candidate,
    )

    monkeypatch.setattr(
        graph_module,
        "test_candidate",
        fake_test_candidate,
    )

    monkeypatch.setattr(
        graph_module,
        "evidence_gate",
        fake_evidence_gate,
    )

    monkeypatch.setattr(
        graph_module,
        "build_report",
        fake_build_report,
    )

    graph = graph_module.build_graph()

    result = graph.invoke(
        {
            "query":
                "Which programs prefer 10x?",
            "max_iterations": 1,
            "trace": [],
        }
    )

    assert (
        result["iteration"]
        == 1
    )

    assert (
        result["tested_candidates"]
        ==
        [
            "Program_A",
        ]
    )

    assert (
        result["final_report"]
        is not None
    )

    assert (
        result[
            "final_report"
        ][
            "candidates_tested"
        ]
        == 1
    )