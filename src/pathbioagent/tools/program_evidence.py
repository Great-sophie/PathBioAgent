import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()


ROOT = Path(
    os.getenv(
        "PATHBIO_RESULTS_ROOT",
        "/media/sophie/Data2T/Paper3_8model_gene_analysis",
    )
)


FILES = {
    "consensus":
        ROOT
        / "program_visibility"
        / "program_cross_model_consensus.csv",

    "cross_model":
        ROOT
        / "program_visibility"
        / "program_cross_model_visibility.csv",

    "preferred_scale":
        ROOT
        / "program_visibility"
        / "program_preferred_scale_by_model.csv",

    "robustness":
        ROOT
        / "final_figures"
        / "Figure6"
        / "source_data"
        / "Figure6_program_robustness_summary.csv",

    "lomo":
        ROOT
        / "final_figures"
        / "Figure6"
        / "source_data"
        / "Figure6B_leave_one_model_out_vote_consensus.csv",
}


# ============================================================
# 1. Data loading
# ============================================================

def load_tables() -> Dict[str, pd.DataFrame]:
    """
    Load all real molecular-program evidence tables.
    """

    tables: Dict[str, pd.DataFrame] = {}

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file [{name}]: {path}"
            )

        tables[name] = pd.read_csv(path)

    return tables


# ============================================================
# 2. Query interpretation
# ============================================================

def infer_target_scale(
    query: str,
) -> Optional[str]:
    """
    Detect an explicitly requested magnification.

    Returns:
        "2p5x"
        "10x"
        "40x"
        None
    """

    q = query.lower()

    if "2.5x" in q or "2p5x" in q:
        return "2p5x"

    if "10x" in q:
        return "10x"

    if "40x" in q:
        return "40x"

    return None


def infer_ranking_intent(
    query: str,
) -> str:
    """
    Infer how candidate molecular programs
    should be ranked.

    Returns:
        preference
        visibility
        robustness
        general
    """

    q = query.lower()

    robustness_terms = [
        "robust",
        "robustness",
        "stable",
        "stability",
        "consistent",
        "reliable",
    ]

    visibility_terms = [
        "strong",
        "strongest",
        "highest",
        "visibility",
        "visible",
        "signal",
    ]

    preference_terms = [
        "prefer",
        "preference",
        "preferred",
    ]

    # Robustness is checked first because
    # "most robust 10x preference"
    # contains both robust and preference.
    if any(
        term in q
        for term in robustness_terms
    ):
        return "robustness"

    if any(
        term in q
        for term in visibility_terms
    ):
        return "visibility"

    if any(
        term in q
        for term in preference_terms
    ):
        return "preference"

    return "general"


# ============================================================
# 3. Candidate search
# ============================================================

def search_program_candidates(
    query: str,
) -> List[str]:
    """
    Search and rank eligible molecular programs.

    preference:
        target-scale votes
        -> target-scale visibility

    visibility:
        target-scale visibility
        -> target-scale votes

    robustness + target scale:
        require actual consensus preference
        for requested scale
        -> LOMO compatibility
        -> optimum stability
        -> vote margin
        -> votes
        -> visibility

    robustness without target scale:
        LOMO compatibility
        -> optimum stability
        -> vote margin
        -> model agreement

    general:
        model agreement
        -> coverage
    """

    tables = load_tables()

    consensus = tables[
        "consensus"
    ].copy()

    cross_model = tables[
        "cross_model"
    ].copy()

    robustness = tables[
        "robustness"
    ].copy()

    # --------------------------------------------------------
    # Eligible programs only
    # --------------------------------------------------------

    if "eligible_main" in consensus.columns:

        consensus = consensus[
            consensus["eligible_main"] == True
        ].copy()

    target_scale = infer_target_scale(
        query
    )

    ranking_intent = infer_ranking_intent(
        query
    )

    ranked = consensus.copy()

    # ========================================================
    # Add target-scale evidence
    # ========================================================

    if target_scale is not None:

        vote_col = (
            f"n_models_prefer_{target_scale}"
        )

        if vote_col not in ranked.columns:

            raise ValueError(
                f"Missing vote column: {vote_col}"
            )

        ranked[
            "target_scale_votes"
        ] = ranked[
            vote_col
        ]

        ranked[
            "target_scale_match"
        ] = (
            ranked[
                "consensus_preferred_scale"
            ]
            ==
            target_scale
        )

        target_visibility = (
            cross_model[
                cross_model[
                    "scale"
                ]
                ==
                target_scale
            ][
                [
                    "program",
                    "median_across_models",
                ]
            ]
            .rename(
                columns={
                    "median_across_models":
                        "target_scale_visibility"
                }
            )
        )

        ranked = ranked.merge(
            target_visibility,
            on="program",
            how="left",
        )

    else:

        ranked[
            "target_scale_votes"
        ] = 0

        ranked[
            "target_scale_match"
        ] = False

        ranked[
            "target_scale_visibility"
        ] = float("nan")

    # ========================================================
    # Add robustness evidence
    # ========================================================

    robustness_subset = robustness[
        [
            "program",
            "minimum_vote_margin",
            "median_vote_margin",
            "compatible_exclusions",
            "n_exclusions",
            "median_optimum_stability_fraction",
        ]
    ].copy()

    robustness_subset[
        "lomo_summary_fraction"
    ] = (
        robustness_subset[
            "compatible_exclusions"
        ]
        /
        robustness_subset[
            "n_exclusions"
        ]
    )

    ranked = ranked.merge(
        robustness_subset,
        on="program",
        how="left",
    )

    # Programs with unavailable robustness evidence
    # should rank after programs with actual evidence.
    robustness_columns = [
        "minimum_vote_margin",
        "median_vote_margin",
        "lomo_summary_fraction",
        "median_optimum_stability_fraction",
    ]

    for column in robustness_columns:

        ranked[column] = (
            ranked[column]
            .fillna(-1)
        )

    # ========================================================
    # Query-aware ranking
    # ========================================================

    # --------------------------------------------------------
    # Preference
    #
    # Which programs prefer 10x?
    # --------------------------------------------------------

    if (
        ranking_intent == "preference"
        and
        target_scale is not None
    ):

        ranked = ranked.sort_values(
            [
                "target_scale_votes",
                "target_scale_visibility",
            ],
            ascending=[
                False,
                False,
            ],
        )

    # --------------------------------------------------------
    # Visibility
    #
    # Which programs are strongest at 10x?
    # --------------------------------------------------------

    elif (
        ranking_intent == "visibility"
        and
        target_scale is not None
    ):

        ranked = ranked.sort_values(
            [
                "target_scale_visibility",
                "target_scale_votes",
            ],
            ascending=[
                False,
                False,
            ],
        )

    # --------------------------------------------------------
    # Robustness + target scale
    #
    # Which programs have the most robust
    # 10x preference?
    # --------------------------------------------------------

    elif (
        ranking_intent == "robustness"
        and
        target_scale is not None
    ):

        # Important:
        # Only programs whose consensus preferred
        # scale actually matches the requested scale
        # are eligible for this ranking.
        ranked = ranked[
            ranked[
                "target_scale_match"
            ]
            ==
            True
        ].copy()

        ranked = ranked.sort_values(
            [
                "lomo_summary_fraction",
                "median_optimum_stability_fraction",
                "minimum_vote_margin",
                "target_scale_votes",
                "target_scale_visibility",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )

    # --------------------------------------------------------
    # Robustness without target scale
    # --------------------------------------------------------

    elif ranking_intent == "robustness":

        ranked = ranked.sort_values(
            [
                "lomo_summary_fraction",
                "median_optimum_stability_fraction",
                "minimum_vote_margin",
                "max_models_agree",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    else:

        ranked = ranked.sort_values(
            [
                "max_models_agree",
                "coverage_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

    return ranked[
        "program"
    ].tolist()


# ============================================================
# 4. Evidence collection
# ============================================================

def collect_program_evidence(
    program: str,
) -> Dict[str, Any]:
    """
    Collect all available real evidence
    for one molecular program.
    """

    tables = load_tables()

    consensus_df = tables[
        "consensus"
    ]

    cross_df = tables[
        "cross_model"
    ]

    preferred_df = tables[
        "preferred_scale"
    ]

    robustness_df = tables[
        "robustness"
    ]

    lomo_df = tables[
        "lomo"
    ]

    # ========================================================
    # Consensus
    # ========================================================

    rows = consensus_df[
        consensus_df[
            "program"
        ]
        ==
        program
    ]

    if rows.empty:

        raise ValueError(
            f"Unknown program: {program}"
        )

    row = rows.iloc[
        0
    ]

    consensus = {
        "preferred_scale":
            str(
                row[
                    "consensus_preferred_scale"
                ]
            ),

        "max_models_agree":
            int(
                row[
                    "max_models_agree"
                ]
            ),

        "votes": {
            "2p5x":
                int(
                    row[
                        "n_models_prefer_2p5x"
                    ]
                ),

            "10x":
                int(
                    row[
                        "n_models_prefer_10x"
                    ]
                ),

            "40x":
                int(
                    row[
                        "n_models_prefer_40x"
                    ]
                ),
        },

        "coverage_fraction":
            float(
                row[
                    "coverage_fraction"
                ]
            ),

        "eligible_main":
            bool(
                row[
                    "eligible_main"
                ]
            ),
    }

    # ========================================================
    # Cross-model visibility
    # ========================================================

    cross_rows = cross_df[
        cross_df[
            "program"
        ]
        ==
        program
    ].copy()

    if cross_rows.empty:

        raise ValueError(
            "No cross-model visibility "
            f"available for: {program}"
        )

    cross_visibility: Dict[
        str,
        Dict[str, Any]
    ] = {}

    for _, r in cross_rows.iterrows():

        scale = str(
            r[
                "scale"
            ]
        )

        cross_visibility[
            scale
        ] = {
            "median":
                float(
                    r[
                        "median_across_models"
                    ]
                ),

            "mean":
                float(
                    r[
                        "mean_across_models"
                    ]
                ),

            "min":
                float(
                    r[
                        "min_across_models"
                    ]
                ),

            "max":
                float(
                    r[
                        "max_across_models"
                    ]
                ),

            "n_models":
                int(
                    r[
                        "n_models"
                    ]
                ),
        }

    ranked_cross = (
        cross_rows
        .sort_values(
            "median_across_models",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    best_row = ranked_cross.iloc[
        0
    ]

    best_scale = str(
        best_row[
            "scale"
        ]
    )

    best_visibility = float(
        best_row[
            "median_across_models"
        ]
    )

    second_best_visibility = None
    cross_model_margin = None

    if len(
        ranked_cross
    ) >= 2:

        second_best_visibility = float(
            ranked_cross.iloc[
                1
            ][
                "median_across_models"
            ]
        )

        cross_model_margin = (
            best_visibility
            -
            second_best_visibility
        )

    # ========================================================
    # Evidence alignment
    # ========================================================

    vote_preferred_scale = (
        consensus[
            "preferred_scale"
        ]
    )

    # A tie is treated as unresolved consensus,
    # not as a direct conflict.
    evidence_concordance = (
        vote_preferred_scale
        !=
        "tie"
        and
        vote_preferred_scale
        ==
        best_scale
    )

    evidence_conflict = (
        vote_preferred_scale
        !=
        "tie"
        and
        vote_preferred_scale
        !=
        best_scale
    )

    evidence_alignment = {
        "vote_preferred_scale":
            vote_preferred_scale,

        "cross_model_median_optimum_scale":
            best_scale,

        "concordant":
            evidence_concordance,

        "conflict":
            evidence_conflict,
    }

    # ========================================================
    # Model-level preferences
    # ========================================================

    preferred_rows = preferred_df[
        preferred_df[
            "program"
        ]
        ==
        program
    ].copy()

    model_preferences = []

    for _, r in preferred_rows.iterrows():

        model_preferences.append(
            {
                "model":
                    str(
                        r[
                            "model"
                        ]
                    ),

                "preferred_scale":
                    str(
                        r[
                            "preferred_scale"
                        ]
                    ),

                "best_median_pearson":
                    float(
                        r[
                            "best_median_pearson"
                        ]
                    ),

                "second_best_median_pearson":
                    float(
                        r[
                            "second_best_median_pearson"
                        ]
                    ),

                "scale_margin":
                    float(
                        r[
                            "scale_margin"
                        ]
                    ),
            }
        )

    # ========================================================
    # Robustness summary
    # ========================================================

    robust_rows = robustness_df[
        robustness_df[
            "program"
        ]
        ==
        program
    ]

    if robust_rows.empty:

        robustness = {
            "available":
                False,
        }

    else:

        r = robust_rows.iloc[
            0
        ]

        robustness = {
            "available":
                True,

            "minimum_vote_margin":
                int(
                    r[
                        "minimum_vote_margin"
                    ]
                ),

            "median_vote_margin":
                float(
                    r[
                        "median_vote_margin"
                    ]
                ),

            "compatible_exclusions":
                int(
                    r[
                        "compatible_exclusions"
                    ]
                ),

            "n_exclusions":
                int(
                    r[
                        "n_exclusions"
                    ]
                ),

            "median_optimum_stability_fraction":
                float(
                    r[
                        "median_optimum_stability_fraction"
                    ]
                ),
        }

    # ========================================================
    # Leave-one-model-out
    # ========================================================

    lomo_rows = lomo_df[
        lomo_df[
            "program"
        ]
        ==
        program
    ].copy()

    if lomo_rows.empty:

        lomo = {
            "available":
                False,
        }

    else:

        n_total = len(
            lomo_rows
        )

        n_compatible = int(
            lomo_rows[
                "compatible_with_full"
            ].sum()
        )

        lomo = {
            "available":
                True,

            "n_exclusions":
                int(
                    n_total
                ),

            "compatible_exclusions":
                n_compatible,

            "compatibility_fraction":
                float(
                    n_compatible
                    /
                    n_total
                ),

            "minimum_vote_margin":
                int(
                    lomo_rows[
                        "vote_margin"
                    ].min()
                ),

            "median_vote_margin":
                float(
                    lomo_rows[
                        "vote_margin"
                    ].median()
                ),
        }

    # ========================================================
    # Structured evidence object
    # ========================================================

    return {
        "program":
            program,

        "consensus":
            consensus,

        "cross_model_visibility":
            cross_visibility,

        "cross_model_optimum": {
            "scale":
                best_scale,

            "median_visibility":
                best_visibility,

            "second_best_median_visibility":
                second_best_visibility,

            "margin":
                cross_model_margin,
        },

        "evidence_alignment":
            evidence_alignment,

        "model_preferences":
            model_preferences,

        "robustness":
            robustness,

        "lomo":
            lomo,
    }


# ============================================================
# 5. Deterministic evidence gate
# ============================================================

def evaluate_program_evidence(
    evidence: Dict[str, Any],
    target_scale: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministically evaluate support for an exploratory
    morphology-molecular scale hypothesis.

    Decision states:

        supported_exploratory_hypothesis
        conflicting_evidence
        not_supported
        insufficient_robustness_evidence
        not_eligible

    Important:
        This evaluates consistency of evidence in the
        available dataset.

        It does NOT establish:
            clinical biomarker validity,
            biological causality,
            patient-level molecular diagnosis.
    """

    consensus = evidence[
        "consensus"
    ]

    robustness = evidence[
        "robustness"
    ]

    lomo = evidence[
        "lomo"
    ]

    alignment = evidence[
        "evidence_alignment"
    ]

    reasons: List[str] = []

    # ========================================================
    # A. Eligibility
    # ========================================================

    if not consensus[
        "eligible_main"
    ]:

        return {
            "status":
                "not_eligible",

            "consensus_ok":
                False,

            "target_ok":
                False,

            "robust_ok":
                False,

            "evidence_conflict":
                alignment[
                    "conflict"
                ],

            "reasons": [
                "Program is not eligible "
                "for the main analysis."
            ],
        }

    # ========================================================
    # B. Robustness evidence availability
    # ========================================================

    if not robustness[
        "available"
    ]:

        return {
            "status":
                "insufficient_robustness_evidence",

            "consensus_ok":
                False,

            "target_ok":
                False,

            "robust_ok":
                False,

            "evidence_conflict":
                alignment[
                    "conflict"
                ],

            "reasons": [
                "No program-level robustness "
                "summary is available."
            ],
        }

    if not lomo[
        "available"
    ]:

        return {
            "status":
                "insufficient_robustness_evidence",

            "consensus_ok":
                False,

            "target_ok":
                False,

            "robust_ok":
                False,

            "evidence_conflict":
                alignment[
                    "conflict"
                ],

            "reasons": [
                "No leave-one-model-out "
                "evidence is available."
            ],
        }

    # ========================================================
    # C. Cross-model vote consensus
    # ========================================================

    consensus_ok = (
        consensus[
            "preferred_scale"
        ]
        !=
        "tie"
        and
        consensus[
            "max_models_agree"
        ]
        >=
        5
    )

    if (
        consensus[
            "preferred_scale"
        ]
        ==
        "tie"
    ):

        reasons.append(
            "Cross-model preferred scale "
            "is tied."
        )

    elif (
        consensus[
            "max_models_agree"
        ]
        <
        5
    ):

        reasons.append(
            "Fewer than 5 of 8 models "
            "agree on the preferred scale."
        )

    # ========================================================
    # D. Requested-scale agreement
    # ========================================================

    target_ok = (
        target_scale is None
        or
        consensus[
            "preferred_scale"
        ]
        ==
        target_scale
    )

    if (
        target_scale is not None
        and
        not target_ok
    ):

        reasons.append(
            "Consensus preferred scale "
            f"is {consensus['preferred_scale']}, "
            f"not {target_scale}."
        )

    # ========================================================
    # E. Robustness gate
    # ========================================================

    robust_ok = (
        robustness[
            "compatible_exclusions"
        ]
        ==
        robustness[
            "n_exclusions"
        ]
        and
        robustness[
            "median_optimum_stability_fraction"
        ]
        >=
        0.75
        and
        lomo[
            "compatibility_fraction"
        ]
        >=
        0.875
    )

    if not robust_ok:

        reasons.append(
            "Program does not pass "
            "the predefined robustness gate."
        )

    # ========================================================
    # F. Evidence alignment
    # ========================================================

    evidence_conflict = bool(
        alignment[
            "conflict"
        ]
    )

    if evidence_conflict:

        reasons.append(
            "Vote-based preferred scale "
            "and cross-model median optimum disagree."
        )

    # ========================================================
    # G. Final decision
    # ========================================================

    basic_support = (
        consensus_ok
        and
        target_ok
        and
        robust_ok
    )

    # Strong underlying evidence exists,
    # but two aggregation strategies disagree.
    if (
        basic_support
        and
        evidence_conflict
    ):

        status = (
            "conflicting_evidence"
        )

        reasons.append(
            "The hypothesis passes consensus "
            "and robustness gates, but the "
            "preferred scale is sensitive to "
            "the aggregation definition."
        )

    # Fully aligned support
    elif (
        basic_support
        and
        not evidence_conflict
    ):

        status = (
            "supported_exploratory_hypothesis"
        )

        reasons.append(
            "Program passes cross-model consensus, "
            "robustness, and evidence-alignment checks."
        )

    # One or more required conditions fail
    else:

        status = (
            "not_supported"
        )

    # ========================================================
    # H. Structured decision
    # ========================================================

    return {
        "status":
            status,

        "consensus_ok":
            consensus_ok,

        "target_ok":
            target_ok,

        "robust_ok":
            robust_ok,

        "evidence_conflict":
            evidence_conflict,

        "vote_preferred_scale":
            alignment[
                "vote_preferred_scale"
            ],

        "cross_model_median_optimum_scale":
            alignment[
                "cross_model_median_optimum_scale"
            ],

        "reasons":
            reasons,
    }