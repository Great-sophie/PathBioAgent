from typing import Any, Dict, List

from pathbioagent.state import BiomarkerAgentState

from pathbioagent.tools.program_evidence import (
    infer_target_scale,
    infer_ranking_intent,
    search_program_candidates,
    collect_program_evidence,
    evaluate_program_evidence,
)


# ============================================================
# 1. Parse query
# ============================================================

def parse_query(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Parse the user query into:
        - target magnification
        - ranking intent
        - coarse task type

    Scientific ranking itself remains deterministic
    inside program_evidence.py.
    """

    query = state.get(
        "query",
        "",
    )

    target_scale = infer_target_scale(
        query
    )

    ranking_intent = infer_ranking_intent(
        query
    )

    if target_scale is not None:
        task_type = (
            f"{ranking_intent}_"
            f"scale_specific_program_search"
        )
    else:
        task_type = (
            f"{ranking_intent}_"
            f"program_search"
        )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        "Parsed query -> "
        f"task_type={task_type}, "
        f"target_scale={target_scale}, "
        f"ranking_intent={ranking_intent}"
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
# 2. Search candidates
# ============================================================

def search_candidates(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Search and rank real molecular-program candidates
    according to the query intent.
    """

    query = state.get(
        "query",
        "",
    )

    candidates = search_program_candidates(
        query
    )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        "Real-data candidate search "
        f"returned {len(candidates)} programs"
    )

    if len(candidates) == 0:
        trace.append(
            "No program satisfied the "
            "query-specific candidate criteria"
        )

    return {
        "candidate_queue":
            candidates,

        "trace":
            trace,
    }


# ============================================================
# 3. Select next candidate
# ============================================================

def select_candidate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Select the first candidate that has not yet
    been evaluated.
    """

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

            current_candidate = candidate
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
            f"Selected candidate="
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
    """
    Collect deterministic real-data evidence
    for the current molecular program.
    """

    candidate = state.get(
        "current_candidate"
    )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    if candidate is None:

        trace.append(
            "Evidence collection skipped: "
            "no current candidate"
        )

        return {
            "trace":
                trace,
        }

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

    consensus = candidate_evidence[
        "consensus"
    ]

    alignment = candidate_evidence[
        "evidence_alignment"
    ]

    robustness = candidate_evidence[
        "robustness"
    ]

    trace.append(
        "Collected real evidence for "
        f"{candidate}: "
        f"vote_preferred_scale="
        f"{consensus['preferred_scale']}, "
        f"agreement="
        f"{consensus['max_models_agree']}/8, "
        f"median_optimum="
        f"{alignment['cross_model_median_optimum_scale']}, "
        f"conflict="
        f"{alignment['conflict']}, "
        f"robustness_available="
        f"{robustness.get('available')}"
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
# 5. Normalize support status
# ============================================================

def _normalize_support_status(
    decision: Dict[str, Any],
) -> str:
    """
    Keep hypothesis support status separate from
    evidence-quality flags.

    Backward compatibility:
    if evaluate_program_evidence() still returns
    'conflicting_evidence' as a raw status, map it to
    supported_exploratory_hypothesis when all underlying
    support gates passed. Conflict remains an independent flag.
    """

    raw_status = decision.get(
        "status",
        "unknown",
    )

    if raw_status != "conflicting_evidence":
        return raw_status

    basic_support = (
        decision.get(
            "consensus_ok",
            False,
        )
        and
        decision.get(
            "target_ok",
            False,
        )
        and
        decision.get(
            "robust_ok",
            False,
        )
    )

    if basic_support:

        return (
            "supported_exploratory_hypothesis"
        )

    return "not_supported"


# ============================================================
# 6. Evidence gate
# ============================================================

def evidence_gate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Apply deterministic scientific evidence rules.

    Primary decision status and evidence-quality flags
    are stored separately.
    """

    candidate = state.get(
        "current_candidate"
    )

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    if candidate is None:

        trace.append(
            "Evidence gate skipped: "
            "no current candidate"
        )

        return {
            "current_decision":
                "no_candidate",

            "trace":
                trace,
        }

    query = state.get(
        "query",
        "",
    )

    target_scale = infer_target_scale(
        query
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

    raw_decision = (
        evaluate_program_evidence(
            candidate_evidence,
            target_scale=target_scale,
        )
    )

    # --------------------------------------------------------
    # Separate support status from evidence flags
    # --------------------------------------------------------

    raw_status = raw_decision.get(
        "status",
        "unknown",
    )

    support_status = (
        _normalize_support_status(
            raw_decision
        )
    )

    decision = dict(
        raw_decision
    )

    decision[
        "raw_gate_status"
    ] = raw_status

    decision[
        "status"
    ] = support_status

    # --------------------------------------------------------
    # Store decision
    # --------------------------------------------------------

    updated_candidate = dict(
        candidate_evidence
    )

    updated_candidate[
        "decision"
    ] = decision

    all_evidence[
        candidate
    ] = updated_candidate

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace.append(
        "Evidence gate: "
        f"{candidate} -> "
        f"{support_status}"
    )

    if decision.get(
        "evidence_conflict",
        False,
    ):

        trace.append(
            "Evidence conflict flag: "
            f"{candidate}; "
            f"vote="
            f"{decision.get('vote_preferred_scale')}, "
            f"median_optimum="
            f"{decision.get('cross_model_median_optimum_scale')}"
        )

    if not decision.get(
        "robust_ok",
        False,
    ):

        trace.append(
            "Robustness warning: "
            f"{candidate} did not pass "
            "the predefined robustness gate"
        )

    return {
        "evidence":
            all_evidence,

        "current_decision":
            support_status,

        "trace":
            trace,
    }


# ============================================================
# 7. Build evidence flags
# ============================================================

def _build_evidence_flags(
    item: Dict[str, Any],
) -> Dict[str, bool]:
    """
    Build orthogonal evidence-quality flags.

    These are diagnostics, not mutually exclusive
    hypothesis statuses.
    """

    consensus = item.get(
        "consensus",
        {},
    )

    alignment = item.get(
        "evidence_alignment",
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

    preferred_scale = (
        consensus.get(
            "preferred_scale"
        )
    )

    max_models_agree = (
        consensus.get(
            "max_models_agree"
        )
    )

    robustness_available = (
        robustness.get(
            "available",
            False,
        )
    )

    lomo_available = (
        lomo.get(
            "available",
            False,
        )
    )

    stability = robustness.get(
        "median_optimum_stability_fraction"
    )

    lomo_fraction = lomo.get(
        "compatibility_fraction"
    )

    return {
        "aggregation_conflict":
            bool(
                alignment.get(
                    "conflict",
                    False,
                )
            ),

        "tie_consensus":
            preferred_scale
            ==
            "tie",

        "weak_model_consensus":
            (
                max_models_agree is not None
                and
                max_models_agree < 5
            ),

        "missing_robustness":
            not robustness_available,

        "missing_lomo":
            not lomo_available,

        "low_optimum_stability":
            (
                robustness_available
                and
                stability is not None
                and
                stability < 0.75
            ),

        "low_lomo_compatibility":
            (
                lomo_available
                and
                lomo_fraction is not None
                and
                lomo_fraction < 0.875
            ),
    }


# ============================================================
# 8. Build evidence card
# ============================================================

def _build_evidence_card(
    program: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create an auditable Evidence Card for one
    molecular program.
    """

    decision = item.get(
        "decision",
        {},
    )

    consensus = item.get(
        "consensus",
        {},
    )

    optimum = item.get(
        "cross_model_optimum",
        {},
    )

    visibility = item.get(
        "cross_model_visibility",
        {},
    )

    alignment = item.get(
        "evidence_alignment",
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

    flags = _build_evidence_flags(
        item
    )

    return {
        # ====================================================
        # Identity and decision
        # ====================================================

        "program":
            program,

        "status":
            decision.get(
                "status",
                "unknown",
            ),

        "raw_gate_status":
            decision.get(
                "raw_gate_status",
                decision.get(
                    "status",
                    "unknown",
                ),
            ),

        # ====================================================
        # Vote-based scale evidence
        # ====================================================

        "vote_preferred_scale":
            alignment.get(
                "vote_preferred_scale",
                consensus.get(
                    "preferred_scale"
                ),
            ),

        "max_models_agree":
            consensus.get(
                "max_models_agree"
            ),

        "scale_votes":
            consensus.get(
                "votes"
            ),

        "coverage_fraction":
            consensus.get(
                "coverage_fraction"
            ),

        # ====================================================
        # Cross-model visibility
        # ====================================================

        "cross_model_visibility":
            visibility,

        "cross_model_median_optimum_scale":
            alignment.get(
                "cross_model_median_optimum_scale",
                optimum.get(
                    "scale"
                ),
            ),

        "cross_model_median_visibility":
            optimum.get(
                "median_visibility"
            ),

        "cross_model_second_best_visibility":
            optimum.get(
                "second_best_median_visibility"
            ),

        "cross_model_scale_margin":
            optimum.get(
                "margin"
            ),

        # ====================================================
        # Alignment diagnostics
        # ====================================================

        "evidence_concordant":
            alignment.get(
                "concordant"
            ),

        "evidence_conflict":
            alignment.get(
                "conflict",
                False,
            ),

        # ====================================================
        # Robustness
        # ====================================================

        "robustness_available":
            robustness.get(
                "available"
            ),

        "minimum_vote_margin":
            robustness.get(
                "minimum_vote_margin"
            ),

        "median_vote_margin":
            robustness.get(
                "median_vote_margin"
            ),

        "median_optimum_stability_fraction":
            robustness.get(
                "median_optimum_stability_fraction"
            ),

        # ====================================================
        # Leave-one-model-out
        # ====================================================

        "lomo_available":
            lomo.get(
                "available"
            ),

        "lomo_compatible_exclusions":
            lomo.get(
                "compatible_exclusions"
            ),

        "lomo_n_exclusions":
            lomo.get(
                "n_exclusions"
            ),

        "lomo_compatibility_fraction":
            lomo.get(
                "compatibility_fraction"
            ),

        # ====================================================
        # Gate diagnostics
        # ====================================================

        "consensus_ok":
            decision.get(
                "consensus_ok"
            ),

        "target_ok":
            decision.get(
                "target_ok"
            ),

        "robust_ok":
            decision.get(
                "robust_ok"
            ),

        # ====================================================
        # Independent evidence flags
        # ====================================================

        "evidence_flags":
            flags,

        # ====================================================
        # Human-readable reasons
        # ====================================================

        "reasons":
            decision.get(
                "reasons",
                [],
            ),
    }


# ============================================================
# 9. Build final report
# ============================================================

def build_report(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Build the final scientific evidence report.

    Important design rule:

        hypothesis status
            !=
        evidence-quality flags

    Therefore a program may simultaneously be:

        status = not_supported
        aggregation_conflict = True

    This is intentional.
    """

    query = state.get(
        "query",
        "",
    )

    target_scale = infer_target_scale(
        query
    )

    ranking_intent = infer_ranking_intent(
        query
    )

    evidence = state.get(
        "evidence",
        {},
    )

    tested_candidates = state.get(
        "tested_candidates",
        [],
    )

    # ========================================================
    # Primary decision groups
    # ========================================================

    supported: List[str] = []

    not_supported: List[str] = []

    insufficient: List[str] = []

    not_eligible: List[str] = []

    unknown: List[str] = []

    # ========================================================
    # Independent diagnostic groups
    # ========================================================

    conflict_programs: List[str] = []

    tie_programs: List[str] = []

    low_stability_programs: List[str] = []

    low_lomo_programs: List[str] = []

    missing_robustness_programs: List[str] = []

    # ========================================================
    # Evidence cards
    # ========================================================

    evidence_cards: List[
        Dict[str, Any]
    ] = []

    # Preserve Agent evaluation order
    for program in tested_candidates:

        if program not in evidence:
            continue

        item = evidence[
            program
        ]

        card = _build_evidence_card(
            program,
            item,
        )

        evidence_cards.append(
            card
        )

        status = card[
            "status"
        ]

        flags = card[
            "evidence_flags"
        ]

        # ====================================================
        # Primary status classification
        # ====================================================

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
            "not_supported"
        ):

            not_supported.append(
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

        elif (
            status
            ==
            "not_eligible"
        ):

            not_eligible.append(
                program
            )

        else:

            unknown.append(
                program
            )

        # ====================================================
        # Independent evidence flags
        # ====================================================

        if flags[
            "aggregation_conflict"
        ]:

            conflict_programs.append(
                program
            )

        if flags[
            "tie_consensus"
        ]:

            tie_programs.append(
                program
            )

        if flags[
            "low_optimum_stability"
        ]:

            low_stability_programs.append(
                program
            )

        if flags[
            "low_lomo_compatibility"
        ]:

            low_lomo_programs.append(
                program
            )

        if flags[
            "missing_robustness"
        ]:

            missing_robustness_programs.append(
                program
            )

    # ========================================================
    # Final structured report
    # ========================================================

    report = {
        # ----------------------------------------------------
        # Query interpretation
        # ----------------------------------------------------

        "query":
            query,

        "target_scale":
            target_scale,

        "ranking_intent":
            ranking_intent,

        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------

        "n_candidates":
            len(
                state.get(
                    "candidate_queue",
                    [],
                )
            ),

        "n_tested":
            len(
                tested_candidates
            ),

        # ----------------------------------------------------
        # Primary scientific decisions
        # ----------------------------------------------------

        "supported_exploratory_hypotheses":
            supported,

        "not_supported":
            not_supported,

        "insufficient_robustness_evidence":
            insufficient,

        "not_eligible":
            not_eligible,

        "unknown_status":
            unknown,

        # ----------------------------------------------------
        # Independent evidence diagnostics
        # ----------------------------------------------------

        "evidence_conflict_programs":
            conflict_programs,

        "tie_consensus_programs":
            tie_programs,

        "low_optimum_stability_programs":
            low_stability_programs,

        "low_lomo_compatibility_programs":
            low_lomo_programs,

        "missing_robustness_programs":
            missing_robustness_programs,

        # ----------------------------------------------------
        # Auditable evidence cards
        # ----------------------------------------------------

        "evidence_cards":
            evidence_cards,

        # ----------------------------------------------------
        # Interpretation boundary
        # ----------------------------------------------------

        "interpretation_scope": (
            "Primary statuses describe support for "
            "exploratory morphology-molecular scale "
            "hypotheses in the available dataset. "
            "Evidence-quality flags are independent "
            "diagnostics and may coexist with any "
            "primary status. "
            "These results do not establish clinical "
            "biomarker validity, biological causality, "
            "or patient-level molecular diagnosis."
        ),
    }

    # ========================================================
    # Trace
    # ========================================================

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        "Built final real-data evidence report: "
        f"supported={len(supported)}, "
        f"not_supported={len(not_supported)}, "
        f"insufficient={len(insufficient)}, "
        f"conflicts={len(conflict_programs)}"
    )

    return {
        "final_report":
            report,

        "trace":
            trace,
    }