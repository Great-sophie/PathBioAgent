from typing import Any, Dict, List



from pathbioagent.state import BiomarkerAgentState




from pathbioagent.tools.gene_reasoning import (
    build_gene_reasoning,
)


from pathbioagent.tools.cluster_biological_query import (
    resolve_biological_pattern_query,
)


from pathbioagent.tools.cluster_biological_comparison import (
    compare_cluster_biology,
)


from pathbioagent.tools.gene_explanation import (
    explain_gene,
)

from pathbioagent.tools.discovery_evidence import (
    build_discovery_evidence,
)



from pathbioagent.llm import (

    interpret_query,

)



from pathbioagent.tools.program_evidence import (

    infer_target_scale,

    infer_ranking_intent,

    search_program_candidates,

    collect_program_evidence,

    evaluate_program_evidence,

)





# ============================================================

# 0. Resolve query context

# ============================================================



def _resolve_query_context(

    state: BiomarkerAgentState,

):

    """

    Resolve target scale, ranking intent, and entity.



    Priority

    --------

    1. structured_query produced by the LLM interpreter

       or deterministic fallback.

    2. legacy deterministic parsing of the raw query.



    Important

    ---------

    target_scale=None inside structured_query is meaningful.

    It means that the interpreter intentionally found no

    requested magnification.



    Therefore None must not trigger raw-query re-parsing.

    """



    query = state.get(

        "query",

        "",

    )



    structured_query = state.get(

        "structured_query"

    )



    # --------------------------------------------------------

    # Structured interpretation exists

    # --------------------------------------------------------



    if structured_query:



        ranking_intent = (

            structured_query.get(

                "intent"

            )

        )



        # Explicit None is valid.

        if (

            "target_scale"

            in structured_query

        ):



            target_scale = (

                structured_query[

                    "target_scale"

                ]

            )



        else:



            target_scale = (

                infer_target_scale(

                    query

                )

            )



        entity = (

            structured_query.get(

                "entity",

                "molecular_program",

            )

        )



        # Defensive backward compatibility

        if ranking_intent is None:



            ranking_intent = (

                infer_ranking_intent(

                    query

                )

            )



        return (

            target_scale,

            ranking_intent,

            entity,

        )



    # --------------------------------------------------------

    # Legacy deterministic path

    # --------------------------------------------------------



    return (

        infer_target_scale(

            query

        ),

        infer_ranking_intent(

            query

        ),

        "molecular_program",

    )





# ============================================================

# 1. Interpret natural-language query

# ============================================================




def discovery_evidence_node(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Build discovery evidence for
    morphology-derived molecular visibility.
    """

    (
        target_scale,
        ranking_intent,
        entity,
    ) = _resolve_query_context(
        state
    )


    if ranking_intent != "discovery":

        return {}


    evidence = build_discovery_evidence()


    return {
        "discovery_evidence": evidence,
    }





def gene_explanation_node(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Explain a single gene's morphology-derived
    molecular visibility.
    """

    import re

    query = state.get(
        "query",
        "",
    )

    genes = re.findall(
        r"\b[A-Z][A-Z0-9-]{2,9}\b",
        query,
    )


    stopwords = {
        "WHY",
        "THE",
        "IS",
        "ARE",
        "HIGHLY",
        "VISIBLE",
        "EXPLAIN",
        "GENE",
    }


    genes = [
        g
        for g in genes
        if g not in stopwords
    ]


    if not genes:
        return {}


    evidence = explain_gene(
        genes[0]
    )


    reasoning = build_gene_reasoning(
        evidence
    )


    return {
        "gene_explanation":
            evidence,

        "gene_reasoning":
            reasoning,
    }



def biological_pattern_evidence_node(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Retrieve biological evidence for
    morphology-derived visibility patterns.

    Handles:
        - cluster biology
        - scale-associated patterns
        - molecular program associations
    """

    (
        target_scale,
        ranking_intent,
        entity,
    ) = _resolve_query_context(
        state
    )


    if ranking_intent != "biological_pattern":

        return {}


    query = state.get(
        "query",
        "",
    )


    evidence = (
        resolve_biological_pattern_query(
            query
        )
    )


    return {
        "biological_pattern_evidence":
            evidence,
    }



def cluster_comparison_node(
    state: BiomarkerAgentState,
):

    import re


    query = state.get(
        "query",
        "",
    )


    clusters = re.findall(
        r"cluster\s+(\d+)",
        query.lower(),
    )


    if len(clusters) < 2:
        return {}


    result = compare_cluster_biology(
        int(clusters[0]),
        int(clusters[1]),
    )


    return {
        "cluster_comparison_evidence":
            result,
    }




def interpret_query_node(

    state: BiomarkerAgentState,

) -> Dict[str, Any]:

    """

    Convert the user's natural-language question into a

    validated structured scientific query.



    The interpreter may use an LLM.



    If an LLM cannot be used, interpret_query() falls back

    to the deterministic parser.



    This node performs language interpretation only.



    It does NOT:

        - calculate scientific evidence;

        - modify correlations;

        - alter robustness metrics;

        - determine statistical significance;

        - decide whether a hypothesis is supported.

    """



    query = state.get(

        "query",

        "",

    )



    interpretation = (

        interpret_query(

            query

        )

    )



    structured_query = (

        interpretation.model_dump()

    )



    trace = list(

        state.get(

            "trace",

            [],

        )

    )



    trace.append(

        "Query interpretation -> "

        f"intent="

        f"{structured_query.get('intent')}, "

        f"target_scale="

        f"{structured_query.get('target_scale')}, "

        f"entity="

        f"{structured_query.get('entity')}, "

        f"source="

        f"{structured_query.get('parser_source')}, "

        f"confidence="

        f"{structured_query.get('confidence')}"

    )



    if (

        structured_query.get(

            "explicit_scale_guard_applied",

            False,

        )

    ):



        trace.append(

            "Explicit scale guard applied: "

            "literal user magnification was preserved"

        )



    return {

        "structured_query":

            structured_query,



        "ranking_intent":

            structured_query.get(
                "intent"
            ),



        "trace":

            trace,

    }





# ============================================================

# 2. Parse structured query into workflow task

# ============================================================



def parse_query(

    state: BiomarkerAgentState,

) -> Dict[str, Any]:

    """

    Convert the validated query interpretation into

    workflow-level task metadata.



    Scientific candidate ranking remains deterministic.

    """



    (

        target_scale,

        ranking_intent,

        entity,

    ) = _resolve_query_context(

        state

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

        "Parsed structured query -> "

        f"task_type={task_type}, "

        f"target_scale={target_scale}, "

        f"ranking_intent={ranking_intent}, "

        f"entity={entity}"

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

# 3. Search candidates

# ============================================================



def search_candidates(

    state: BiomarkerAgentState,

) -> Dict[str, Any]:

    """

    Search and rank real molecular-program candidates.



    The language interpretation comes from structured_query.



    Candidate ranking itself remains deterministic in

    program_evidence.py.

    """



    query = state.get(

        "query",

        "",

    )



    (

        target_scale,

        ranking_intent,

        entity,

    ) = _resolve_query_context(

        state

    )



    candidates = (

        search_program_candidates(

            query,

            ranking_intent=ranking_intent,

            target_scale=target_scale,

        )

    )



    trace = list(

        state.get(

            "trace",

            [],

        )

    )



    trace.append(

        "Real-data candidate search -> "

        f"intent={ranking_intent}, "

        f"target_scale={target_scale}, "

        f"entity={entity}, "

        f"returned={len(candidates)}"

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

# 4. Select next candidate

# ============================================================



def select_candidate(

    state: BiomarkerAgentState,

) -> Dict[str, Any]:

    """

    Select the first candidate that has not yet been

    evaluated.

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

# 5. Collect real evidence

# ============================================================



def test_candidate(

    state: BiomarkerAgentState,

) -> Dict[str, Any]:

    """

    Collect deterministic real-data evidence for the current

    molecular program.

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



    consensus = candidate_evidence.get(

        "consensus",

        {},

    )



    alignment = candidate_evidence.get(

        "evidence_alignment",

        {},

    )



    robustness = candidate_evidence.get(

        "robustness",

        {},

    )



    trace.append(

        "Collected real evidence for "

        f"{candidate}: "

        f"vote_preferred_scale="

        f"{consensus.get('preferred_scale')}, "

        f"agreement="

        f"{consensus.get('max_models_agree')}/8, "

        f"median_optimum="

        f"{alignment.get('cross_model_median_optimum_scale')}, "

        f"conflict="

        f"{alignment.get('conflict')}, "

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

# 6. Normalize support status

# ============================================================



def _normalize_support_status(

    decision: Dict[str, Any],

) -> str:

    """

    Keep hypothesis support status separate from

    evidence-quality flags.



    Backward compatibility

    ----------------------

    If evaluate_program_evidence() returns the historical raw

    status:



        conflicting_evidence



    map it into a primary support status.



    Conflict remains an independent evidence-quality

    diagnostic.

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

# 7. Evidence gate

# ============================================================



def _normalize_scale_key(scale: Any) -> Any:
    """Normalize scale labels used by visibility tables."""

    if scale is None:
        return None

    value = str(scale).strip().lower()

    aliases = {
        "2.5x": "2p5x",
        "2p5x": "2p5x",
        "10x": "10x",
        "40x": "40x",
    }

    return aliases.get(value, value)


def _get_visibility_for_scale(
    visibility_by_scale: Any,
    target_scale: Any,
) -> Any:
    """
    Return scale-specific visibility while tolerating scale aliases.

    The scientific values are not modified; this helper only resolves
    labels such as ``2.5x`` versus ``2p5x``.
    """

    if not isinstance(visibility_by_scale, dict):
        return None

    normalized_target = _normalize_scale_key(target_scale)

    if normalized_target is None:
        return None

    for scale_name, value in visibility_by_scale.items():
        if _normalize_scale_key(scale_name) == normalized_target:
            return value

    return None


def evidence_gate(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Apply deterministic scientific evidence rules.

    The historical scale-hypothesis gate is preserved for backward
    compatibility. Query-specific semantics are stored separately so
    that three different scientific questions are not conflated:

    visibility
        How strong is molecular visibility at the requested scale?

    preference
        Is the requested scale the preferred scale?

    robustness
        Is the preferred-scale conclusion stable?

    In particular, a visibility query must not treat a preferred-scale
    mismatch as evidence that visibility at the target scale is absent.
    """

    candidate = state.get("current_candidate")

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    if candidate is None:
        trace.append(
            "Evidence gate skipped: no current candidate"
        )

        return {
            "current_decision": "no_candidate",
            "trace": trace,
        }

    (
        target_scale,
        ranking_intent,
        entity,
    ) = _resolve_query_context(state)

    all_evidence = dict(
        state.get(
            "evidence",
            {},
        )
    )

    candidate_evidence = all_evidence.get(candidate)

    if not isinstance(candidate_evidence, dict):
        trace.append(
            "Evidence gate failed: "
            f"missing evidence for {candidate}"
        )

        return {
            "evidence": all_evidence,
            "current_decision": "missing_evidence",
            "trace": trace,
        }

    # --------------------------------------------------------
    # Historical deterministic scale-hypothesis gate
    # --------------------------------------------------------

    raw_decision = evaluate_program_evidence(
        candidate_evidence,
        target_scale=target_scale,
    )

    raw_status = raw_decision.get(
        "status",
        "unknown",
    )

    support_status = _normalize_support_status(
        raw_decision
    )

    decision = dict(raw_decision)

    # Keep the historical fields unchanged.
    decision["raw_gate_status"] = raw_status
    decision["status"] = support_status

    # Explicitly name what that status means.
    decision["scale_hypothesis_status"] = support_status

    # --------------------------------------------------------
    # Query context and query-specific semantics
    # --------------------------------------------------------

    decision["ranking_intent"] = ranking_intent
    decision["query_entity"] = entity
    decision["query_target_scale"] = target_scale

    # Preference and robustness queries retain the historical gate
    # interpretation unless a more specific branch below overrides it.
    decision["query_status"] = support_status
    decision["target_scale_visibility"] = None

    if ranking_intent == "visibility":
        visibility_by_scale = candidate_evidence.get(
            "cross_model_visibility",
            {},
        )

        target_visibility = _get_visibility_for_scale(
            visibility_by_scale,
            target_scale,
        )

        if isinstance(target_visibility, dict):
            decision["query_status"] = (
                "ranked_target_scale_visibility"
            )
            decision["target_scale_visibility"] = dict(
                target_visibility
            )
        else:
            decision["query_status"] = (
                "insufficient_visibility_evidence"
            )

    # --------------------------------------------------------
    # Store decision
    # --------------------------------------------------------

    updated_candidate = dict(candidate_evidence)
    updated_candidate["decision"] = decision
    all_evidence[candidate] = updated_candidate

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace.append(
        "Evidence gate: "
        f"{candidate} -> {support_status}"
    )

    if ranking_intent == "visibility":
        target_visibility = decision.get(
            "target_scale_visibility"
        )

        if isinstance(target_visibility, dict):
            trace.append(
                "Visibility query status: "
                f"{candidate} -> "
                f"{decision.get('query_status')}; "
                f"target_scale={target_scale}; "
                "median_visibility="
                f"{target_visibility.get('median_across_models')}"
            )
        else:
            trace.append(
                "Visibility query status: "
                f"{candidate} -> "
                "insufficient_visibility_evidence; "
                f"target_scale={target_scale}"
            )

    if decision.get(
        "evidence_conflict",
        False,
    ):
        trace.append(
            "Evidence conflict flag: "
            f"{candidate}; "
            f"vote={decision.get('vote_preferred_scale')}, "
            "median_optimum="
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
        "evidence": all_evidence,

        # Backward-compatible workflow control. The graph routing and
        # existing tests may depend on the historical support status.
        "current_decision": support_status,

        "trace": trace,
    }
# ============================================================

# 8. Build evidence-quality flags

# ============================================================



def _build_evidence_flags(

    item: Dict[str, Any],

) -> Dict[str, bool]:

    """

    Build orthogonal evidence-quality flags.



    These diagnostics are not mutually exclusive with the

    primary support status.

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

# 9. Build Evidence Card

# ============================================================



def _build_evidence_card(
    program: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    """Create an auditable Evidence Card for one molecular program."""

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

    flags = _build_evidence_flags(item)

    return {
        # ----------------------------------------------------
        # Identity and historical scale-hypothesis decision
        # ----------------------------------------------------
        "program": program,

        "status": decision.get(
            "status",
            "unknown",
        ),

        "raw_gate_status": decision.get(
            "raw_gate_status",
            decision.get(
                "status",
                "unknown",
            ),
        ),

        "scale_hypothesis_status": decision.get(
            "scale_hypothesis_status",
            decision.get(
                "status",
                "unknown",
            ),
        ),

        # ----------------------------------------------------
        # Query-specific interpretation
        # ----------------------------------------------------
        "ranking_intent": decision.get(
            "ranking_intent"
        ),

        "query_entity": decision.get(
            "query_entity"
        ),

        "query_target_scale": decision.get(
            "query_target_scale"
        ),

        "query_status": decision.get(
            "query_status",
            decision.get(
                "status",
                "unknown",
            ),
        ),

        "target_scale_visibility": decision.get(
            "target_scale_visibility"
        ),

        # ----------------------------------------------------
        # Vote-based scale evidence
        # ----------------------------------------------------
        "vote_preferred_scale": alignment.get(
            "vote_preferred_scale",
            consensus.get(
                "preferred_scale"
            ),
        ),

        "max_models_agree": consensus.get(
            "max_models_agree"
        ),

        "scale_votes": consensus.get(
            "votes"
        ),

        "coverage_fraction": consensus.get(
            "coverage_fraction"
        ),

        "eligible_main": consensus.get(
            "eligible_main"
        ),

        # ----------------------------------------------------
        # Cross-model visibility
        # ----------------------------------------------------
        "cross_model_visibility": visibility,

        "cross_model_median_optimum_scale": alignment.get(
            "cross_model_median_optimum_scale",
            optimum.get(
                "scale"
            ),
        ),

        "cross_model_median_visibility": optimum.get(
            "median_visibility"
        ),

        "cross_model_second_best_visibility": optimum.get(
            "second_best_median_visibility"
        ),

        "cross_model_scale_margin": optimum.get(
            "margin"
        ),

        # ----------------------------------------------------
        # Alignment diagnostics
        # ----------------------------------------------------
        "evidence_concordant": alignment.get(
            "concordant"
        ),

        "evidence_conflict": alignment.get(
            "conflict",
            False,
        ),

        # ----------------------------------------------------
        # Robustness summary
        # ----------------------------------------------------
        "robustness_available": robustness.get(
            "available"
        ),

        "minimum_vote_margin": robustness.get(
            "minimum_vote_margin"
        ),

        "median_vote_margin": robustness.get(
            "median_vote_margin"
        ),

        "compatible_exclusions": robustness.get(
            "compatible_exclusions"
        ),

        "n_exclusions": robustness.get(
            "n_exclusions"
        ),

        "median_optimum_stability_fraction": robustness.get(
            "median_optimum_stability_fraction"
        ),

        # ----------------------------------------------------
        # Leave-one-model-out evidence
        # ----------------------------------------------------
        "lomo_available": lomo.get(
            "available"
        ),

        "lomo_compatible_exclusions": lomo.get(
            "compatible_exclusions"
        ),

        "lomo_n_exclusions": lomo.get(
            "n_exclusions"
        ),

        "lomo_compatibility_fraction": lomo.get(
            "compatibility_fraction"
        ),

        "lomo_minimum_vote_margin": lomo.get(
            "minimum_vote_margin"
        ),

        "lomo_median_vote_margin": lomo.get(
            "median_vote_margin"
        ),

        # ----------------------------------------------------
        # Historical gate components
        # ----------------------------------------------------
        "consensus_ok": decision.get(
            "consensus_ok"
        ),

        "target_ok": decision.get(
            "target_ok"
        ),

        "robust_ok": decision.get(
            "robust_ok"
        ),

        # ----------------------------------------------------
        # Independent diagnostics
        # ----------------------------------------------------
        "evidence_flags": flags,

        # ----------------------------------------------------
        # Human-readable reasons
        # ----------------------------------------------------
        "reasons": decision.get(
            "reasons",
            [],
        ),
    }


def _build_visibility_ranking(
    evidence_cards: List[Dict[str, Any]],
    target_scale: Any,
) -> List[Dict[str, Any]]:
    """
    Build a deterministic ranking by target-scale median visibility.

    This ranking answers the absolute visibility question only. Scale
    preference and robustness remain secondary diagnostics and are not
    used to exclude a program from this ranking.
    """

    rows: List[Dict[str, Any]] = []

    for card in evidence_cards:
        target_visibility = card.get(
            "target_scale_visibility"
        )

        if not isinstance(target_visibility, dict):
            target_visibility = _get_visibility_for_scale(
                card.get(
                    "cross_model_visibility",
                    {},
                ),
                target_scale,
            )

        if not isinstance(target_visibility, dict):
            continue

        median_visibility = target_visibility.get(
            "median_across_models"
        )

        if median_visibility is None:
            continue

        rows.append(
            {
                "program": card.get("program"),
                "target_scale": target_scale,
                "median_across_models": median_visibility,
                "mean_across_models": target_visibility.get(
                    "mean_across_models"
                ),
                "min_across_models": target_visibility.get(
                    "min_across_models"
                ),
                "max_across_models": target_visibility.get(
                    "max_across_models"
                ),
                "n_models": target_visibility.get(
                    "n_models"
                ),
                "query_status": card.get(
                    "query_status"
                ),
                "scale_hypothesis_status": card.get(
                    "scale_hypothesis_status"
                ),
                "vote_preferred_scale": card.get(
                    "vote_preferred_scale"
                ),
                "max_models_agree": card.get(
                    "max_models_agree"
                ),
                "evidence_concordant": card.get(
                    "evidence_concordant"
                ),
                "evidence_conflict": card.get(
                    "evidence_conflict"
                ),
                "median_optimum_stability_fraction": card.get(
                    "median_optimum_stability_fraction"
                ),
                "lomo_compatibility_fraction": card.get(
                    "lomo_compatibility_fraction"
                ),
                "evidence_flags": card.get(
                    "evidence_flags",
                    {},
                ),
            }
        )

    rows.sort(
        key=lambda row: row.get(
            "median_across_models",
            float("-inf"),
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows
# ============================================================

# 10. Build final report

# ============================================================



def build_report(
    state: BiomarkerAgentState,
) -> Dict[str, Any]:
    """
    Build the final scientific evidence report.

    Two orthogonal outputs are intentionally preserved:

    1. scale-hypothesis support status
       Derived from consensus, target-scale preference, and robustness.

    2. query-specific answer
       For ``visibility`` queries, this is the absolute target-scale
       visibility ranking and must not be replaced by preferred-scale
       support status.

    This separation prevents the invalid inference:

        "target scale is not uniquely preferred"
            -> "visibility at target scale is low/absent"
    """

    query = state.get(
        "query",
        "",
    )

    (
        target_scale,
        ranking_intent,
        entity,
    ) = _resolve_query_context(state)

    structured_query = state.get(
        "structured_query",
        {},
    )

    evidence = state.get(
        "evidence",
        {},
    )

    discovery_evidence = state.get(
        "discovery_evidence",
        None,
    )

    tested_candidates = state.get(
        "tested_candidates",
        [],
    )

    candidate_queue = state.get(
        "candidate_queue",
        [],
    )

    # ========================================================
    # Historical primary decision groups
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
    missing_lomo_programs: List[str] = []
    weak_consensus_programs: List[str] = []

    # ========================================================
    # Evidence Cards
    # ========================================================

    evidence_cards: List[Dict[str, Any]] = []

    # Preserve Agent evaluation order in the evidence-card audit trail.
    for program in tested_candidates:
        if program not in evidence:
            continue

        item = evidence[program]

        card = _build_evidence_card(
            program,
            item,
        )

        evidence_cards.append(card)

        status = card.get(
            "status",
            "unknown",
        )

        flags = card.get(
            "evidence_flags",
            {},
        )

        # ----------------------------------------------------
        # Historical scale-hypothesis classification
        # ----------------------------------------------------

        if status == "supported_exploratory_hypothesis":
            supported.append(program)
        elif status == "not_supported":
            not_supported.append(program)
        elif status == "insufficient_robustness_evidence":
            insufficient.append(program)
        elif status == "not_eligible":
            not_eligible.append(program)
        else:
            unknown.append(program)

        # ----------------------------------------------------
        # Orthogonal evidence-quality diagnostics
        # ----------------------------------------------------

        if flags.get("aggregation_conflict", False):
            conflict_programs.append(program)

        if flags.get("tie_consensus", False):
            tie_programs.append(program)

        if flags.get("low_optimum_stability", False):
            low_stability_programs.append(program)

        if flags.get("low_lomo_compatibility", False):
            low_lomo_programs.append(program)

        if flags.get("missing_robustness", False):
            missing_robustness_programs.append(program)

        if flags.get("missing_lomo", False):
            missing_lomo_programs.append(program)

        if flags.get("weak_model_consensus", False):
            weak_consensus_programs.append(program)

    # ========================================================
    # Query-specific answer layer
    # ========================================================

    visibility_ranking: List[Dict[str, Any]] = []

    if (
        ranking_intent == "visibility"
        and target_scale is not None
    ):
        visibility_ranking = _build_visibility_ranking(
            evidence_cards,
            target_scale,
        )

        primary_answer_type = (
            "target_scale_visibility_ranking"
        )

        query_semantics = (
            "Programs are ranked by absolute cross-model molecular "
            f"visibility at {target_scale}, using median_across_models. "
            "Preferred-scale consensus and robustness are secondary "
            "diagnostics and do not exclude a program from this ranking."
        )
    elif ranking_intent == "preference":
        primary_answer_type = "scale_preference_support"
        query_semantics = (
            "The primary question concerns preferred magnification. "
            "Scale-hypothesis support status is the primary output."
        )
    elif ranking_intent == "robustness":
        primary_answer_type = "scale_preference_robustness"
        query_semantics = (
            "The primary question concerns stability of preferred-scale "
            "evidence across model exclusions and aggregation choices."
        )

    elif ranking_intent == "biological_pattern":

        primary_answer_type = (
            "biological_pattern_evidence"
        )

        query_semantics = (
            "The query concerns biological interpretation "
            "of morphology-derived visibility patterns, "
            "including cluster-level molecular programs "
            "and pathway-associated evidence."
        )
    elif ranking_intent == "biological_pattern":

        primary_answer_type = (
            "biological_pattern_evidence"
        )

        query_semantics = (
            "The query concerns biological interpretation "
            "of morphology-derived visibility patterns, "
            "including cluster-level molecular programs "
            "and pathway-associated evidence."
        )

    else:
        primary_answer_type = "scale_hypothesis_support"
        query_semantics = (
            "The primary output follows the deterministic evidence gate."
        )

    # ========================================================
    # Interpretation scope
    # ========================================================

    interpretation_scope = (
        "Scale-hypothesis statuses and query-specific ranking outputs "
        "answer different questions. For visibility queries, absolute "
        "target-scale visibility is the primary answer, while preferred-"
        "scale and robustness fields are secondary diagnostics. For "
        "preference or robustness queries, the deterministic scale-"
        "hypothesis gate remains primary. All findings are exploratory "
        "and do not establish clinical biomarker validity, biological "
        "causality, or patient-level molecular diagnosis."
    )

    # ========================================================
    # Final report object
    # ========================================================

    final_report = {
        # ----------------------------------------------------
        # Query metadata
        # ----------------------------------------------------
        "query": query,

        "discovery_evidence":
            discovery_evidence,

        "gene_explanation":
            state.get(
                "gene_explanation"
            ),

        "biological_pattern_evidence":
            state.get(
                "biological_pattern_evidence"
            ),

        "gene_reasoning":
            state.get(
                "gene_reasoning"
            ),

        "task_type": state.get("task_type"),
        "ranking_intent": ranking_intent,
        "target_scale": target_scale,
        "entity": entity,
        "structured_query": structured_query,
        "parser_source": structured_query.get("parser_source"),
        "query_confidence": structured_query.get("confidence"),
        "model_name": structured_query.get("model_name"),

        # ----------------------------------------------------
        # Query-specific primary answer metadata
        # ----------------------------------------------------
        "primary_answer_type": primary_answer_type,
        "query_semantics": query_semantics,

        "visibility_ranking_metric": (
            "median_across_models"
            if visibility_ranking
            else None
        ),

        "visibility_ranking": visibility_ranking,

        # ----------------------------------------------------
        # Candidate accounting
        # ----------------------------------------------------
        "candidates_tested": len(tested_candidates),
        "total_candidates": len(candidate_queue),
        "tested_candidates": list(tested_candidates),
        "candidate_queue": list(candidate_queue),

        # ----------------------------------------------------
        # Historical scale-hypothesis decisions
        # ----------------------------------------------------
        "supported_exploratory_hypotheses": supported,
        "not_supported": not_supported,
        "insufficient_robustness_evidence": insufficient,
        "not_eligible": not_eligible,
        "unknown_status": unknown,

        # Compatibility aliases
        "supported": supported,
        "insufficient": insufficient,

        # ----------------------------------------------------
        # Evidence diagnostics
        # ----------------------------------------------------
        "evidence_conflict_programs": conflict_programs,
        "tie_consensus_programs": tie_programs,
        "low_optimum_stability_programs": low_stability_programs,
        "low_lomo_compatibility_programs": low_lomo_programs,
        "missing_robustness_programs": missing_robustness_programs,
        "missing_lomo_programs": missing_lomo_programs,
        "weak_model_consensus_programs": weak_consensus_programs,

        # Compatibility aliases
        "aggregation_conflict": conflict_programs,
        "tie_consensus": tie_programs,

        # ----------------------------------------------------
        # Evidence Cards
        # ----------------------------------------------------
        "evidence_cards": evidence_cards,

        # ----------------------------------------------------
        # Scientific scope
        # ----------------------------------------------------
        "interpretation_scope": interpretation_scope,
    }

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    # Expose agent execution provenance for chat interface/tests.
    # Scientific evidence fields remain unchanged.
    final_report["trace"] = list(trace)

    trace.append(
        "Built final report -> "
        f"tested={len(tested_candidates)}/"
        f"{len(candidate_queue)}, "
        f"supported={len(supported)}, "
        f"not_supported={len(not_supported)}, "
        f"insufficient={len(insufficient)}, "
        f"conflicts={len(conflict_programs)}"
    )

    if visibility_ranking:
        top_row = visibility_ranking[0]
        trace.append(
            "Built target-scale visibility ranking -> "
            f"scale={target_scale}, "
            f"n={len(visibility_ranking)}, "
            f"top={top_row.get('program')}, "
            "median_visibility="
            f"{top_row.get('median_across_models')}"
        )

    return {
        "final_report": final_report,
        "trace": trace,
    }

