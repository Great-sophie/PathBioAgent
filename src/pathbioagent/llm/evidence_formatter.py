"""
Evidence formatter for PathBioAgent.

This module converts the deterministic PathBioAgent final report
into a compact, LLM-friendly evidence package.

Important design principle
--------------------------
The formatter does NOT:
    - rank candidates
    - change evidence-gate decisions
    - infer biological mechanisms
    - generate scientific conclusions

It only reorganizes evidence already produced by the deterministic
workflow.

A key distinction is preserved between:

    1. target-scale visibility
    2. preferred scale
    3. robustness of scale preference

For example:

    "high molecular visibility at 10x"

is NOT equivalent to:

    "10x is the preferred scale".
"""


from typing import Any, Dict, List, Optional



# ============================================================
# Generic helpers
# ============================================================


def _safe_dict(
    value: Any,
) -> Dict[str, Any]:
    """
    Return value if it is a dictionary,
    otherwise return an empty dictionary.
    """

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}



def _safe_list(
    value: Any,
) -> List[Any]:
    """
    Convert optional values into a list.
    """

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    return [value]



def _normalize_scale(
    scale: Any,
) -> Optional[str]:
    """
    Normalize magnification labels.

    Examples
    --------
    "10X"      -> "10x"
    "10x"      -> "10x"
    "2.5x"     -> "2.5x"
    "2p5x"     -> "2p5x"

    We intentionally do not force 2.5x and 2p5x to be identical
    globally because existing tables may use either representation.
    """

    if scale is None:
        return None

    return str(
        scale
    ).strip().lower()



def _scales_match(
    left: Any,
    right: Any,
) -> bool:
    """
    Compare scale labels while tolerating
    2.5x / 2p5x naming differences.
    """

    left_norm = _normalize_scale(
        left
    )

    right_norm = _normalize_scale(
        right
    )


    if (
        left_norm is None
        or right_norm is None
    ):
        return False


    aliases = {
        "2.5x": "2p5x",
        "2p5x": "2p5x",
        "10x": "10x",
        "40x": "40x",
    }


    left_norm = aliases.get(
        left_norm,
        left_norm,
    )

    right_norm = aliases.get(
        right_norm,
        right_norm,
    )


    return (
        left_norm
        == right_norm
    )



def _active_flags(
    flags: Any,
) -> List[str]:
    """
    Convert evidence flag dictionary into a compact
    list of active flag names.

    Example
    -------
    {
        "aggregation_conflict": True,
        "tie_consensus": False
    }

    becomes:

    [
        "aggregation_conflict"
    ]
    """

    if not isinstance(
        flags,
        dict,
    ):
        return []


    return [
        str(name)
        for name, active in flags.items()
        if bool(active)
    ]



# ============================================================
# Visibility extraction
# ============================================================


def _extract_visibility_from_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract standard cross-model visibility metrics
    from one scale-specific row.
    """

    if not isinstance(
        row,
        dict,
    ):
        return {}


    output = {}


    supported_keys = [
        "scale",
        "median_across_models",
        "mean_across_models",
        "min_across_models",
        "max_across_models",
        "n_models",
    ]


    for key in supported_keys:

        if key in row:

            output[
                key
            ] = row.get(
                key
            )


    return output



def _find_scale_in_mapping(
    mapping: Dict[str, Any],
    target_scale: str,
) -> Dict[str, Any]:
    """
    Search a dictionary keyed by scale.

    Example
    -------
    {
        "10x": {
            "median_across_models": ...
        }
    }
    """

    for scale_name, value in mapping.items():

        if not _scales_match(
            scale_name,
            target_scale,
        ):
            continue


        if isinstance(
            value,
            dict,
        ):

            result = (
                _extract_visibility_from_row(
                    value
                )
            )


            if (
                "scale"
                not in result
            ):

                result[
                    "scale"
                ] = scale_name


            return result


        # A compact mapping may directly store a scalar:
        #
        # {
        #     "10x": 0.40
        # }

        return {
            "scale":
                scale_name,

            "median_across_models":
                value,
        }


    return {}



def _find_scale_in_rows(
    rows: List[Any],
    target_scale: str,
) -> Dict[str, Any]:
    """
    Search a list of scale-specific dictionaries.
    """

    for row in rows:

        if not isinstance(
            row,
            dict,
        ):
            continue


        if not _scales_match(
            row.get(
                "scale"
            ),
            target_scale,
        ):
            continue


        return (
            _extract_visibility_from_row(
                row
            )
        )


    return {}



def _extract_target_scale_visibility(
    card: Dict[str, Any],
    target_scale: Optional[str],
) -> Dict[str, Any]:
    """
    Extract visibility at the requested target scale.

    The function supports several possible evidence-card
    representations to remain compatible with evolving
    PathBioAgent report schemas.

    Supported examples
    ------------------

    A:
        card["cross_model_visibility"] = {
            "10x": {
                "median_across_models": 0.40,
                ...
            }
        }

    B:
        card["cross_model_visibility"] = [
            {
                "scale": "10x",
                ...
            }
        ]

    C:
        card["visibility_by_scale"] = ...

    D:
        scale-specific values may already exist directly
        inside the evidence card.
    """

    if target_scale is None:

        return {}


    # --------------------------------------------------------
    # Common nested containers
    # --------------------------------------------------------

    possible_containers = [
        "cross_model_visibility",
        "visibility_by_scale",
        "scale_visibility",
        "visibility",
    ]


    for key in possible_containers:

        value = card.get(
            key
        )


        if isinstance(
            value,
            dict,
        ):

            result = (
                _find_scale_in_mapping(
                    value,
                    target_scale,
                )
            )


            if result:

                return result


        if isinstance(
            value,
            list,
        ):

            result = (
                _find_scale_in_rows(
                    value,
                    target_scale,
                )
            )


            if result:

                return result


    # --------------------------------------------------------
    # Some reports may store current scale-specific
    # visibility directly on the evidence card.
    # --------------------------------------------------------

    card_scale = card.get(
        "scale"
    )


    if (
        card_scale is not None
        and _scales_match(
            card_scale,
            target_scale,
        )
    ):

        result = (
            _extract_visibility_from_row(
                card
            )
        )


        if result:

            return result


    # --------------------------------------------------------
    # Direct target-scale-specific fields
    #
    # Example:
    # target_scale_visibility = {
    #     ...
    # }
    # --------------------------------------------------------

    direct = card.get(
        "target_scale_visibility"
    )


    if isinstance(
        direct,
        dict,
    ):

        result = (
            _extract_visibility_from_row(
                direct
            )
        )


        if result:

            if (
                "scale"
                not in result
            ):

                result[
                    "scale"
                ] = target_scale


            return result


    return {}



# ============================================================
# Scale vote extraction
# ============================================================


def _extract_scale_votes(
    card: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collect available model-vote counts.
    """

    votes = {}


    direct_votes = card.get(
        "scale_votes"
    )


    if isinstance(
        direct_votes,
        dict,
    ):

        votes.update(
            direct_votes
        )


    vote_keys = {
        "2p5x":
            "n_models_prefer_2p5x",

        "10x":
            "n_models_prefer_10x",

        "40x":
            "n_models_prefer_40x",
    }


    for scale, key in vote_keys.items():

        if key in card:

            votes[
                scale
            ] = card.get(
                key
            )


    return votes



# ============================================================
# Evidence card formatting
# ============================================================


def format_evidence_card(
    card: Dict[str, Any],
    *,
    target_scale: Optional[str],
    ranking_position: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert one full evidence card into a compact
    LLM-friendly representation.
    """

    flags = _safe_dict(
        card.get(
            "evidence_flags"
        )
    )


    target_visibility = (
        _extract_target_scale_visibility(
            card,
            target_scale,
        )
    )


    scale_votes = (
        _extract_scale_votes(
            card
        )
    )


    formatted = {
        "program":
            card.get(
                "program"
            ),

        "ranking_position":
            ranking_position,

        "status":
            card.get(
                "status"
            ),

        # ----------------------------------------------------
        # Query-specific absolute visibility
        # ----------------------------------------------------

        "target_scale":
            target_scale,

        "target_scale_visibility":
            target_visibility
            if target_visibility
            else None,

        # ----------------------------------------------------
        # Scale preference
        # ----------------------------------------------------

        "vote_preferred_scale":
            card.get(
                "vote_preferred_scale"
            ),

        "max_models_agree":
            card.get(
                "max_models_agree"
            ),

        "scale_votes":
            scale_votes
            if scale_votes
            else None,

        "scale_margin":
            card.get(
                "scale_margin"
            ),

        # ----------------------------------------------------
        # Cross-model aggregation
        # ----------------------------------------------------

        "cross_model_median_optimum_scale":
            card.get(
                "cross_model_median_optimum_scale"
            ),

        "cross_model_median_visibility":
            card.get(
                "cross_model_median_visibility"
            ),

        "cross_model_scale_margin":
            card.get(
                "cross_model_scale_margin"
            ),

        # ----------------------------------------------------
        # Robustness
        # ----------------------------------------------------

        "median_optimum_stability_fraction":
            card.get(
                "median_optimum_stability_fraction"
            ),

        "lomo_compatibility_fraction":
            card.get(
                "lomo_compatibility_fraction"
            ),

        # ----------------------------------------------------
        # Evidence alignment
        # ----------------------------------------------------

        "evidence_concordant":
            card.get(
                "evidence_concordant"
            ),

        "evidence_conflict":
            card.get(
                "evidence_conflict"
            ),

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        "active_evidence_flags":
            _active_flags(
                flags
            ),

        "evidence_flags":
            flags,

        "reasons":
            _safe_list(
                card.get(
                    "reasons"
                )
            ),
    }


    return formatted



def build_evidence_cards(
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build compact evidence cards while preserving
    deterministic candidate order.

    The original order is important because it may already
    represent the deterministic ranking produced upstream.
    """

    target_scale = report.get(
        "target_scale"
    )


    cards = _safe_list(
        report.get(
            "evidence_cards"
        )
    )


    formatted_cards = []


    for index, item in enumerate(
        cards,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue


        formatted_cards.append(
            format_evidence_card(
                item,
                target_scale=target_scale,
                ranking_position=index,
            )
        )


    return formatted_cards



# ============================================================
# Report-level diagnostics
# ============================================================


def _build_query_semantics(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Explicitly tell the explanation layer what the
    deterministic query intent means.

    This is particularly important for visibility queries.

    A visibility query asks:
        "How strong is visibility at this scale?"

    A preference query asks:
        "Which scale is preferred?"

    A robustness query asks:
        "How stable is that preference?"
    """

    ranking_intent = report.get(
        "ranking_intent"
    )


    target_scale = report.get(
        "target_scale"
    )


    if ranking_intent == "visibility":

        distinction = (
            "Target-scale visibility and preferred-scale evidence "
            "must be interpreted separately. A program may have high "
            "visibility at the requested scale even when that scale is "
            "not its uniquely preferred scale."
        )


    elif ranking_intent == "preference":

        distinction = (
            "The primary question concerns scale preference rather "
            "than absolute visibility at a single scale."
        )


    elif ranking_intent == "robustness":

        distinction = (
            "The primary question concerns robustness and stability "
            "of scale preference across models or perturbations."
        )


    else:

        distinction = (
            "Interpret the evidence according to the reported ranking "
            "intent without changing deterministic rankings or gate "
            "decisions."
        )


    return {
        "ranking_intent":
            ranking_intent,

        "target_scale":
            target_scale,

        "interpretation_rule":
            distinction,
    }



def _build_status_summary(
    report: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """
    Preserve deterministic evidence-gate categories.
    """

    return {
        "supported_exploratory_hypotheses":
            _safe_list(
                report.get(
                    "supported_exploratory_hypotheses"
                )
            ),

        "not_supported":
            _safe_list(
                report.get(
                    "not_supported"
                )
            ),

        "insufficient_robustness_evidence":
            _safe_list(
                report.get(
                    "insufficient_robustness_evidence"
                )
            ),

        "not_eligible":
            _safe_list(
                report.get(
                    "not_eligible"
                )
            ),
    }



# ============================================================
# Main public API
# ============================================================


def prepare_report_for_llm(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a deterministic PathBioAgent final report into
    a compact evidence package for the explanation LLM.

    This function does not perform inference.

    Parameters
    ----------
    report:
        Final deterministic PathBioAgent report.

    Returns
    -------
    dict:
        Compact evidence package suitable for scientific
        explanation.
    """

    if not isinstance(
        report,
        dict,
    ):

        raise TypeError(
            "report must be a dictionary"
        )


    evidence_cards = (
        build_evidence_cards(
            report
        )
    )


    output = {
        # ----------------------------------------------------
        # Original question
        # ----------------------------------------------------

        "query":
            report.get(
                "query"
            ),

        # ----------------------------------------------------
        # Query semantics
        # ----------------------------------------------------

        "query_semantics":
            _build_query_semantics(
                report
            ),

        # ----------------------------------------------------
        # Deterministic candidate counts
        # ----------------------------------------------------

        "candidate_summary": {
            "candidates_tested":
                report.get(
                    "candidates_tested",
                    report.get(
                        "n_tested"
                    ),
                ),

            "total_candidates":
                report.get(
                    "total_candidates",
                    report.get(
                        "n_candidates"
                    ),
                ),
        },

        # ----------------------------------------------------
        # Evidence-gate result
        # ----------------------------------------------------

        "status_summary":
            _build_status_summary(
                report
            ),

        # ----------------------------------------------------
        # Candidate evidence
        #
        # Order is preserved from deterministic ranking.
        # ----------------------------------------------------

        "evidence_cards":
            evidence_cards,

        # ----------------------------------------------------
        # Interpretation boundary
        # ----------------------------------------------------

        "interpretation_constraints": [
            (
                "Candidate order originates from the deterministic "
                "PathBioAgent workflow and must not be changed."
            ),
            (
                "Evidence-gate status and absolute target-scale "
                "visibility are distinct concepts."
            ),
            (
                "A not-supported target-scale preference does not "
                "necessarily imply low visibility at that scale."
            ),
            (
                "Do not infer biological mechanism, causality, "
                "clinical validity, or biomarker validation."
            ),
            (
                "Treat all supported findings as exploratory unless "
                "external validation evidence is explicitly provided."
            ),
        ],
    }


    return output