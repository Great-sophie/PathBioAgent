from typing import Any, Dict

from pathbioagent.state import BiomarkerAgentState

from pathbioagent.tools.program_evidence import (
    infer_target_scale,
    search_program_candidates,
    collect_program_evidence,
    evaluate_program_evidence,
)


# ============================================================
# 1. Parse user query
# ============================================================

def parse_query(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    query = state.get(
        "query",
        "",
    )

    target_scale = infer_target_scale(
        query
    )

    if target_scale is not None:
        task_type = "scale_specific_program_search"
    else:
        task_type = "program_search"

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        f"Parsed query -> "
        f"task_type={task_type}, "
        f"target_scale={target_scale}"
    )

    return {
        "task_type":
            task_type,

        "candidate_queue":
            [],

        "current_candidate":
            None,

        "tested_candidates":
            [],

        "evidence":
            {},

        "current_decision":
            None,

        "iteration":
            0,

        "max_iterations":
            state.get(
                "max_iterations",
                10,
            ),

        "trace":
            trace,
    }


# ============================================================
# 2. Search real molecular-program candidates
# ============================================================

def search_candidates(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    query = state.get(
        "query",
        "",
    )

    candidates = (
        search_program_candidates(
            query
        )
    )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        f"Real-data candidate search "
        f"returned {len(candidates)} programs"
    )

    return {
        "candidate_queue":
            candidates,

        "trace":
            trace,
    }


# ============================================================
# 3. Select next untested candidate
# ============================================================

def select_candidate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    queue = state.get(
        "candidate_queue",
        [],
    )

    tested = state.get(
        "tested_candidates",
        [],
    )

    current_candidate = None

    for candidate in queue:

        if candidate not in tested:

            current_candidate = (
                candidate
            )

            break

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    if current_candidate is None:

        trace.append(
            "No untested candidate remains"
        )

    else:

        trace.append(
            "Selected candidate="
            f"{current_candidate}"
        )

    return {
        "current_candidate":
            current_candidate,

        "trace":
            trace,
    }


# ============================================================
# 4. Collect real evidence
# ============================================================

def test_candidate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    candidate = state.get(
        "current_candidate"
    )

    if candidate is None:
        raise ValueError(
            "No current candidate "
            "available for evidence testing."
        )

    candidate_evidence = (
        collect_program_evidence(
            candidate
        )
    )

    all_evidence = dict(
        state.get(
            "evidence",
            {},
        )
    )

    all_evidence[
        candidate
    ] = candidate_evidence

    tested = list(
        state.get(
            "tested_candidates",
            [],
        )
    )

    if candidate not in tested:
        tested.append(
            candidate
        )

    iteration = (
        state.get(
            "iteration",
            0,
        )
        + 1
    )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    preferred_scale = (
        candidate_evidence[
            "consensus"
        ][
            "preferred_scale"
        ]
    )

    agreement = (
        candidate_evidence[
            "consensus"
        ][
            "max_models_agree"
        ]
    )

    trace.append(
        f"Collected real evidence for "
        f"{candidate}: "
        f"preferred_scale={preferred_scale}, "
        f"agreement={agreement}/8"
    )

    return {
        "evidence":
            all_evidence,

        "tested_candidates":
            tested,

        "iteration":
            iteration,

        "trace":
            trace,
    }


# ============================================================
# 5. Deterministic evidence gate
# ============================================================

def evidence_gate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    candidate = state.get(
        "current_candidate"
    )

    if candidate is None:
        raise ValueError(
            "No current candidate "
            "available for evidence gate."
        )

    query = state.get(
        "query",
        "",
    )

    target_scale = (
        infer_target_scale(
            query
        )
    )

    all_evidence = dict(
        state.get(
            "evidence",
            {},
        )
    )

    candidate_evidence = (
        all_evidence[
            candidate
        ]
    )

    decision = (
        evaluate_program_evidence(
            candidate_evidence,
            target_scale=target_scale,
        )
    )

    updated_candidate = dict(
        candidate_evidence
    )

    updated_candidate[
        "decision"
    ] = decision

    all_evidence[
        candidate
    ] = updated_candidate

    status = decision[
        "status"
    ]

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        f"Evidence gate: "
        f"{candidate} -> {status}"
    )

    return {
        "evidence":
            all_evidence,

        "current_decision":
            status,

        "trace":
            trace,
    }


# ============================================================
# 6. Build final real-data evidence report
# ============================================================

def build_report(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:

    query = state.get(
        "query",
        "",
    )

    target_scale = (
        infer_target_scale(
            query
        )
    )

    evidence = state.get(
        "evidence",
        {},
    )

    supported = []
    not_supported = []
    insufficient = []

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

        consensus = item.get(
            "consensus",
            {},
        )

        optimum = item.get(
            "cross_model_optimum",
            {},
        )

        robustness = item.get(
            "robustness",
            {},
        )

        lomo = item.get(
            "lomo",
            {},
        )

        card = {
            "program":
                program,

            "status":
                status,

            "vote_preferred_scale":
                consensus.get(
                    "preferred_scale"
                ),

            "max_models_agree":
                consensus.get(
                    "max_models_agree"
                ),

            "scale_votes":
                consensus.get(
                    "votes"
                ),

            "cross_model_optimum_scale":
                optimum.get(
                    "scale"
                ),

            "cross_model_median_visibility":
                optimum.get(
                    "median_visibility"
                ),

            "robustness_available":
                robustness.get(
                    "available"
                ),

            "lomo_available":
                lomo.get(
                    "available"
                ),

            "lomo_compatibility_fraction":
                lomo.get(
                    "compatibility_fraction"
                ),

            "median_optimum_stability_fraction":
                robustness.get(
                    "median_optimum_stability_fraction"
                ),
        }

        evidence_cards.append(
            card
        )

        if (
            status
            ==
            "supported_exploratory_hypothesis"
        ):

            supported.append(
                program
            )

        elif (
            status
            ==
            "insufficient_robustness_evidence"
        ):

            insufficient.append(
                program
            )

        else:

            not_supported.append(
                program
            )

    report = {
        "query":
            query,

        "target_scale":
            target_scale,

        "n_tested":
            len(
                state.get(
                    "tested_candidates",
                    [],
                )
            ),

        "supported_exploratory_hypotheses":
            supported,

        "not_supported":
            not_supported,

        "insufficient_evidence":
            insufficient,

        "evidence_cards":
            evidence_cards,

        "interpretation_scope": (
            "Statuses describe support for "
            "exploratory morphology-molecular "
            "scale hypotheses in the available "
            "dataset. They do not establish "
            "clinical biomarker validity."
        ),
    }

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        "Built final real-data evidence report"
    )

    return {
        "final_report":
            report,

        "trace":
            trace,
    }