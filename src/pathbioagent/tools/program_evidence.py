import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_RESULTS_ROOT = Path(
    "/media/sophie/Data2T/Paper3_8model_gene_analysis"
)

_QUERY_UNSET = object()

VALID_SCALES = (
    "2p5x",
    "10x",
    "40x",
)

VALID_INTENTS = (
    "preference",
    "visibility",
    "robustness",
    "general",
)


# ============================================================
# Small utilities
# ============================================================

def _results_root() -> Path:
    """
    Resolve the root directory containing the real
    morphology-molecular analysis outputs.

    Environment variable takes priority:

        PATHBIO_RESULTS_ROOT

    Otherwise the historical local default is used.
    """

    raw = os.getenv(
        "PATHBIO_RESULTS_ROOT"
    )

    if raw:
        return Path(raw).expanduser()

    return DEFAULT_RESULTS_ROOT


def _native(
    value: Any,
) -> Any:
    """
    Convert pandas / NumPy scalar-like values into
    JSON-friendly Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def _row_to_dict(
    row: pd.Series,
) -> Dict[str, Any]:
    """
    Convert one pandas Series into a clean native dict.
    """

    return {
        str(key): _native(value)
        for key, value in row.to_dict().items()
    }


def _records(
    frame: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into JSON-friendly records.
    """

    output: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        output.append(
            _row_to_dict(row)
        )

    return output


def _normalize_scale(
    scale: Optional[str],
) -> Optional[str]:
    """
    Normalize a scale string into the internal representation.

    Examples:
        2.5x -> 2p5x
        2p5x -> 2p5x
        10x  -> 10x
        40x  -> 40x
    """

    if scale is None:
        return None

    normalized = (
        str(scale)
        .strip()
        .lower()
        .replace("×", "x")
    )

    aliases = {
        "2.5x": "2p5x",
        "2.5": "2p5x",
        "2p5x": "2p5x",
        "2p5": "2p5x",
        "10x": "10x",
        "10": "10x",
        "40x": "40x",
        "40": "40x",
    }

    if normalized in aliases:
        return aliases[
            normalized
        ]

    raise ValueError(
        "Unsupported target scale: "
        f"{scale!r}. "
        f"Expected one of {VALID_SCALES}."
    )


def _validate_intent(
    intent: str,
) -> str:
    """
    Validate one structured ranking intent.
    """

    normalized = (
        str(intent)
        .strip()
        .lower()
    )

    if normalized not in VALID_INTENTS:
        raise ValueError(
            "Unsupported ranking intent: "
            f"{intent!r}. "
            f"Expected one of {VALID_INTENTS}."
        )

    return normalized


# ============================================================
# Load real evidence tables
# ============================================================

def load_tables() -> Dict[str, pd.DataFrame]:
    """
    Load the real program-level evidence tables used by
    PathBioAgent.

    Required files
    --------------
    program_visibility/
        program_visibility_8model_3scale.csv
        program_cross_model_consensus.csv
        program_cross_model_visibility.csv
        program_preferred_scale_by_model.csv

    final_figures/Figure6/source_data/
        Figure6_program_robustness_summary.csv
        Figure6B_leave_one_model_out_vote_consensus.csv
    """

    root = _results_root()

    paths = {
        "visibility":
            root
            / "program_visibility"
            / "program_visibility_8model_3scale.csv",

        "consensus":
            root
            / "program_visibility"
            / "program_cross_model_consensus.csv",

        "cross_model":
            root
            / "program_visibility"
            / "program_cross_model_visibility.csv",

        "preferred_scale":
            root
            / "program_visibility"
            / "program_preferred_scale_by_model.csv",

        "robustness":
            root
            / "final_figures"
            / "Figure6"
            / "source_data"
            / "Figure6_program_robustness_summary.csv",

        "lomo":
            root
            / "final_figures"
            / "Figure6"
            / "source_data"
            / "Figure6B_leave_one_model_out_vote_consensus.csv",
    }

    missing = [
        (name, path)
        for name, path in paths.items()
        if not path.exists()
    ]

    if missing:

        label, path = missing[0]

        raise FileNotFoundError(
            f"Missing file [{label}]: {path}"
        )

    return {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }


# ============================================================
# Deterministic query parsing
# ============================================================

def infer_target_scale(
    query: str,
) -> Optional[str]:
    """
    Infer an explicitly requested magnification from text.

    Returns:
        "2p5x"
        "10x"
        "40x"
        None
    """

    text = (
        query
        .lower()
        .replace("×", "x")
    )

    # 2.5x variants
    if re.search(
        r"\b2(?:\.|p)5\s*x\b",
        text,
    ):
        return "2p5x"

    # 10x
    if re.search(
        r"\b10\s*x\b",
        text,
    ):
        return "10x"

    # 40x
    if re.search(
        r"\b40\s*x\b",
        text,
    ):
        return "40x"

    return None


def infer_ranking_intent(
    query: str,
) -> str:
    """
    Infer the scientific ranking intent from a query.

    Priority matters:

        robustness
            >
        visibility
            >
        preference
            >
        general

    This prevents a query such as

        "most robust 10x preference"

    from being incorrectly classified as simple preference.
    """

    text = query.lower()


    # --------------------------------------------------------
    # Discovery
    # --------------------------------------------------------
    # Questions asking what factors explain
    # morphology-derived molecular visibility.
    # --------------------------------------------------------

    discovery_terms = (
        "what determines",
        "what explains",
        "which factors",
        "associated with",
        "gene properties",
        "predict molecular visibility",
        "why are genes visible",
        "patterns across genes",
        "molecular visibility factors",
    )


    if any(
        term in text
        for term in discovery_terms
    ):
        return "discovery"


    # --------------------------------------------------------
    # Robustness
    # --------------------------------------------------------

    robustness_terms = (
        "robust",
        "robustness",
        "stable",
        "stability",
        "reproducible",
        "reproducibility",
        "consistent",
        "consistency",
        "leave-one-model",
        "leave one model",
        "lomo",
        "resistant",
    )

    if any(
        term in text
        for term in robustness_terms
    ):
        return "robustness"

    # --------------------------------------------------------
    # Gene-level explanation
    # --------------------------------------------------------
    #
    # Example:
    #   Why is SFTPA1 highly visible?
    #
    # This must be checked before visibility ranking.
    #

    explanation_terms = (
        "why is",
        "why are",
        "why does",
        "explain",
        "explanation",
        "reason for",
    )

    if any(
        term in text
        for term in explanation_terms
    ):
        return "gene_explanation"



    # --------------------------------------------------------
    # Visibility / absolute strength
    # --------------------------------------------------------

    visibility_terms = (
        "strongest",
        "stronger",
        "strength",
        "visibility",
        "visible",
        "highest visibility",
        "most visible",
        "largest",
        "highest",
    )

    if any(
        term in text
        for term in visibility_terms
    ):
        return "visibility"

    # --------------------------------------------------------
    # Preferred scale
    # --------------------------------------------------------

    preference_terms = (
        "prefer",
        "preferred",
        "preference",
        "optimal scale",
        "optimum scale",
        "best scale",
    )

    if any(
        term in text
        for term in preference_terms
    ):
        return "preference"

    return "general"


# ============================================================
# Candidate search and query-aware ranking
# ============================================================

def search_program_candidates(
    query: str,
    *,
    ranking_intent: Any = _QUERY_UNSET,
    target_scale: Any = _QUERY_UNSET,
) -> List[str]:
    """
    Search and rank eligible molecular programs.

    Parameters
    ----------
    query:
        Original natural-language user query.

    ranking_intent:
        Optional structured override supplied by the
        LLM query interpreter.

        Supported:
            preference
            visibility
            robustness
            general

        If omitted, deterministic parsing of `query`
        is used.

    target_scale:
        Optional structured scale override.

        Supported:
            2p5x
            10x
            40x
            None

        IMPORTANT:
        Explicit None means:
            "the structured interpreter found no requested scale"

        It therefore must NOT trigger re-parsing of the
        original query.

    Ranking rules
    -------------

    preference:
        target-scale model votes
        -> target-scale visibility

    visibility:
        target-scale visibility
        -> target-scale model votes

    robustness + target scale:
        require actual consensus preference for requested scale
        -> LOMO summary compatibility
        -> optimum stability
        -> vote margin
        -> target-scale votes
        -> target-scale visibility

    robustness without target scale:
        LOMO summary compatibility
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

    if (
        "eligible_main"
        in consensus.columns
    ):

        consensus = consensus[
            consensus[
                "eligible_main"
            ]
            ==
            True
        ].copy()

    # --------------------------------------------------------
    # Resolve structured or legacy query interpretation
    # --------------------------------------------------------

    if target_scale is _QUERY_UNSET:

        resolved_scale = (
            infer_target_scale(
                query
            )
        )

    else:

        resolved_scale = (
            _normalize_scale(
                target_scale
            )
            if target_scale is not None
            else None
        )

    if ranking_intent is _QUERY_UNSET:

        resolved_intent = (
            infer_ranking_intent(
                query
            )
        )

    else:

        resolved_intent = (
            _validate_intent(
                ranking_intent
            )
        )

    target_scale = resolved_scale
    ranking_intent = resolved_intent

    ranked = consensus.copy()

    # ========================================================
    # Add target-scale evidence
    # ========================================================

    if target_scale is not None:

        vote_col = (
            "n_models_prefer_"
            f"{target_scale}"
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

    robustness_columns_needed = [
        "program",
        "minimum_vote_margin",
        "median_vote_margin",
        "compatible_exclusions",
        "n_exclusions",
        "median_optimum_stability_fraction",
    ]

    existing_robustness_columns = [
        column
        for column in robustness_columns_needed
        if column in robustness.columns
    ]

    robustness_subset = robustness[
        existing_robustness_columns
    ].copy()

    if (
        "compatible_exclusions"
        in robustness_subset.columns
        and
        "n_exclusions"
        in robustness_subset.columns
    ):

        denominator = (
            robustness_subset[
                "n_exclusions"
            ]
            .replace(
                0,
                pd.NA,
            )
        )

        robustness_subset[
            "lomo_summary_fraction"
        ] = (
            robustness_subset[
                "compatible_exclusions"
            ]
            /
            denominator
        )

    else:

        robustness_subset[
            "lomo_summary_fraction"
        ] = float("nan")

    ranked = ranked.merge(
        robustness_subset,
        on="program",
        how="left",
    )

    # Missing robustness should rank after real evidence.
    robustness_sort_columns = [
        "minimum_vote_margin",
        "median_vote_margin",
        "lomo_summary_fraction",
        "median_optimum_stability_fraction",
    ]

    for column in robustness_sort_columns:

        if column not in ranked.columns:
            ranked[column] = -1.0

        else:
            ranked[column] = (
                ranked[column]
                .fillna(-1.0)
            )

    # ========================================================
    # Query-aware ranking
    # ========================================================

    # --------------------------------------------------------
    # Preference
    #
    # Example:
    # Which programs prefer 10x?
    # --------------------------------------------------------

    if (
        ranking_intent
        ==
        "preference"
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
            na_position="last",
        )

    # --------------------------------------------------------
    # Visibility
    #
    # Example:
    # Which programs are strongest at 10x?
    # --------------------------------------------------------

    elif (
        ranking_intent
        ==
        "visibility"
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
            na_position="last",
        )

    # --------------------------------------------------------
    # Robustness + requested target scale
    #
    # Example:
    # Which programs have the most robust 10x preference?
    # --------------------------------------------------------

    elif (
        ranking_intent
        ==
        "robustness"
        and
        target_scale is not None
    ):

        # Scientific requirement:
        # a "robust 10x preference" must first actually
        # have 10x as the vote-consensus preferred scale.
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
            na_position="last",
        )

    # --------------------------------------------------------
    # Robustness without a requested target scale
    # --------------------------------------------------------

    elif (
        ranking_intent
        ==
        "robustness"
    ):

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
            na_position="last",
        )

    # --------------------------------------------------------
    # General search
    # --------------------------------------------------------

    else:

        sort_columns = []

        ascending = []

        if (
            "max_models_agree"
            in ranked.columns
        ):
            sort_columns.append(
                "max_models_agree"
            )
            ascending.append(
                False
            )

        if (
            "coverage_fraction"
            in ranked.columns
        ):
            sort_columns.append(
                "coverage_fraction"
            )
            ascending.append(
                False
            )

        if sort_columns:

            ranked = ranked.sort_values(
                sort_columns,
                ascending=ascending,
                na_position="last",
            )

    return (
        ranked[
            "program"
        ]
        .astype(str)
        .tolist()
    )


# ============================================================
# Evidence collection
# ============================================================

def collect_program_evidence(
    program: str,
) -> Dict[str, Any]:
    """
    Collect all available real evidence for one molecular
    program.

    Returned evidence is deliberately structured so the
    deterministic evidence gate and the LLM explanation layer
    remain separate.
    """

    tables = load_tables()

    consensus_table = tables[
        "consensus"
    ]

    cross_model_table = tables[
        "cross_model"
    ]

    preferred_table = tables[
        "preferred_scale"
    ]

    visibility_table = tables[
        "visibility"
    ]

    robustness_table = tables[
        "robustness"
    ]

    lomo_table = tables[
        "lomo"
    ]

    # ========================================================
    # Consensus
    # ========================================================

    consensus_rows = consensus_table[
        consensus_table[
            "program"
        ]
        ==
        program
    ]

    if consensus_rows.empty:

        raise ValueError(
            "Program not found in consensus table: "
            f"{program}"
        )

    consensus_row = (
        consensus_rows
        .iloc[0]
    )

    preferred_scale = _native(
        consensus_row.get(
            "consensus_preferred_scale"
        )
    )

    max_models_agree = _native(
        consensus_row.get(
            "max_models_agree"
        )
    )

    consensus = {
        "preferred_scale":
            preferred_scale,

        "max_models_agree":
            max_models_agree,

        "votes": {
            "2p5x":
                _native(
                    consensus_row.get(
                        "n_models_prefer_2p5x"
                    )
                ),

            "10x":
                _native(
                    consensus_row.get(
                        "n_models_prefer_10x"
                    )
                ),

            "40x":
                _native(
                    consensus_row.get(
                        "n_models_prefer_40x"
                    )
                ),
        },

        "n_genes_total":
            _native(
                consensus_row.get(
                    "n_genes_total"
                )
            ),

        "n_genes_top5000":
            _native(
                consensus_row.get(
                    "n_genes_top5000"
                )
            ),

        "coverage_fraction":
            _native(
                consensus_row.get(
                    "coverage_fraction"
                )
            ),

        "eligible_main":
            bool(
                consensus_row.get(
                    "eligible_main",
                    True,
                )
            ),
    }

    # ========================================================
    # Cross-model visibility
    # ========================================================

    cross_rows = (
        cross_model_table[
            cross_model_table[
                "program"
            ]
            ==
            program
        ]
        .copy()
    )

    cross_model_visibility: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for _, row in cross_rows.iterrows():

        scale = str(
            row[
                "scale"
            ]
        )

        cross_model_visibility[
            scale
        ] = {
            "median_across_models":
                _native(
                    row.get(
                        "median_across_models"
                    )
                ),

            "mean_across_models":
                _native(
                    row.get(
                        "mean_across_models"
                    )
                ),

            "min_across_models":
                _native(
                    row.get(
                        "min_across_models"
                    )
                ),

            "max_across_models":
                _native(
                    row.get(
                        "max_across_models"
                    )
                ),

            "n_models":
                _native(
                    row.get(
                        "n_models"
                    )
                ),
        }

    # ========================================================
    # Cross-model median optimum
    # ========================================================

    valid_optimum = (
        cross_rows[
            [
                "scale",
                "median_across_models",
            ]
        ]
        .dropna(
            subset=[
                "median_across_models"
            ]
        )
        .copy()
    )

    if valid_optimum.empty:

        cross_model_optimum = {
            "scale":
                None,

            "median_visibility":
                None,

            "second_best_median_visibility":
                None,

            "margin":
                None,
        }

    else:

        valid_optimum = (
            valid_optimum.sort_values(
                "median_across_models",
                ascending=False,
            )
        )

        best = (
            valid_optimum
            .iloc[0]
        )

        second = (
            valid_optimum
            .iloc[1]
            if len(
                valid_optimum
            ) > 1
            else None
        )

        best_value = float(
            best[
                "median_across_models"
            ]
        )

        second_value = (
            float(
                second[
                    "median_across_models"
                ]
            )
            if second is not None
            else None
        )

        margin = (
            best_value
            -
            second_value
            if second_value is not None
            else None
        )

        cross_model_optimum = {
            "scale":
                str(
                    best[
                        "scale"
                    ]
                ),

            "median_visibility":
                best_value,

            "second_best_median_visibility":
                second_value,

            "margin":
                margin,
        }

    # ========================================================
    # Model-level preferred scales
    # ========================================================

    preferred_rows = (
        preferred_table[
            preferred_table[
                "program"
            ]
            ==
            program
        ]
        .copy()
    )

    model_preferences = _records(
        preferred_rows
    )

    # ========================================================
    # Model × scale visibility
    # ========================================================

    visibility_rows = (
        visibility_table[
            visibility_table[
                "program"
            ]
            ==
            program
        ]
        .copy()
    )

    model_scale_visibility = _records(
        visibility_rows
    )

    # ========================================================
    # Robustness summary
    # ========================================================

    robustness_rows = (
        robustness_table[
            robustness_table[
                "program"
            ]
            ==
            program
        ]
    )

    if robustness_rows.empty:

        robustness = {
            "available":
                False,

            "minimum_vote_margin":
                None,

            "median_vote_margin":
                None,

            "compatible_exclusions":
                None,

            "n_exclusions":
                None,

            "median_optimum_stability_fraction":
                None,
        }

    else:

        row = (
            robustness_rows
            .iloc[0]
        )

        robustness = {
            "available":
                True,

            "minimum_vote_margin":
                _native(
                    row.get(
                        "minimum_vote_margin"
                    )
                ),

            "median_vote_margin":
                _native(
                    row.get(
                        "median_vote_margin"
                    )
                ),

            "compatible_exclusions":
                _native(
                    row.get(
                        "compatible_exclusions"
                    )
                ),

            "n_exclusions":
                _native(
                    row.get(
                        "n_exclusions"
                    )
                ),

            "median_optimum_stability_fraction":
                _native(
                    row.get(
                        "median_optimum_stability_fraction"
                    )
                ),
        }

    # ========================================================
    # Leave-one-model-out evidence
    # ========================================================

    lomo_rows = (
        lomo_table[
            lomo_table[
                "program"
            ]
            ==
            program
        ]
        .copy()
    )

    if lomo_rows.empty:

        lomo = {
            "available":
                False,

            "compatible_exclusions":
                None,

            "n_exclusions":
                None,

            "compatibility_fraction":
                None,

            "minimum_vote_margin":
                None,

            "median_vote_margin":
                None,

            "records":
                [],
        }

    else:

        n_exclusions = int(
            len(
                lomo_rows
            )
        )

        compatible_series = (
            lomo_rows[
                "compatible_with_full"
            ]
            if (
                "compatible_with_full"
                in lomo_rows.columns
            )
            else pd.Series(
                [False]
                * n_exclusions
            )
        )

        compatible_exclusions = int(
            compatible_series
            .fillna(False)
            .astype(bool)
            .sum()
        )

        compatibility_fraction = (
            compatible_exclusions
            /
            n_exclusions
            if n_exclusions > 0
            else None
        )

        if (
            "vote_margin"
            in lomo_rows.columns
        ):

            margins = (
                lomo_rows[
                    "vote_margin"
                ]
                .dropna()
            )

            minimum_vote_margin = (
                float(
                    margins.min()
                )
                if not margins.empty
                else None
            )

            median_vote_margin = (
                float(
                    margins.median()
                )
                if not margins.empty
                else None
            )

        else:

            minimum_vote_margin = None
            median_vote_margin = None

        lomo = {
            "available":
                True,

            "compatible_exclusions":
                compatible_exclusions,

            "n_exclusions":
                n_exclusions,

            "compatibility_fraction":
                compatibility_fraction,

            "minimum_vote_margin":
                minimum_vote_margin,

            "median_vote_margin":
                median_vote_margin,

            "records":
                _records(
                    lomo_rows
                ),
        }

    # ========================================================
    # Agreement between two aggregation views
    # ========================================================

    median_optimum_scale = (
        cross_model_optimum[
            "scale"
        ]
    )

    concordant = (
        preferred_scale
        not in (
            None,
            "tie",
        )
        and
        median_optimum_scale
        is not None
        and
        preferred_scale
        ==
        median_optimum_scale
    )

    conflict = (
        preferred_scale
        not in (
            None,
            "tie",
        )
        and
        median_optimum_scale
        is not None
        and
        preferred_scale
        !=
        median_optimum_scale
    )

    evidence_alignment = {
        "vote_preferred_scale":
            preferred_scale,

        "cross_model_median_optimum_scale":
            median_optimum_scale,

        "concordant":
            bool(
                concordant
            ),

        # A vote tie is unresolved, not a direct conflict.
        "conflict":
            bool(
                conflict
            ),
    }

    # ========================================================
    # Final evidence object
    # ========================================================

    return {
        "program":
            program,

        "consensus":
            consensus,

        "cross_model_visibility":
            cross_model_visibility,

        "cross_model_optimum":
            cross_model_optimum,

        "model_preferences":
            model_preferences,

        "model_scale_visibility":
            model_scale_visibility,

        "robustness":
            robustness,

        "lomo":
            lomo,

        "evidence_alignment":
            evidence_alignment,
    }


# ============================================================
# Deterministic evidence gate
# ============================================================

def evaluate_program_evidence(
    evidence: Dict[str, Any],
    target_scale: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply deterministic scientific evidence rules.

    This function does not use an LLM.

    Main gate
    ---------

    Consensus:
        preferred scale must not be a tie
        and >= 5 / 8 models must agree.

    Target:
        if the user requested a scale, the vote-based
        consensus preferred scale must equal that scale.

    Robustness:
        robustness summary must be available;
        all reported exclusions must be compatible;
        median optimum stability >= 0.75;
        LOMO must be available;
        LOMO compatibility >= 0.875.

    Important
    ---------
    Evidence conflict is a diagnostic dimension.

    For backward compatibility, a fully supported program
    with aggregation disagreement may still return the raw
    status "conflicting_evidence". nodes.py normalizes this
    into the primary support status while retaining the
    conflict flag.
    """

    target_scale = _normalize_scale(
        target_scale
    )

    consensus = evidence.get(
        "consensus",
        {},
    )

    robustness = evidence.get(
        "robustness",
        {},
    )

    lomo = evidence.get(
        "lomo",
        {},
    )

    alignment = evidence.get(
        "evidence_alignment",
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

    eligible_main = bool(
        consensus.get(
            "eligible_main",
            True,
        )
    )

    # --------------------------------------------------------
    # Consensus gate
    # --------------------------------------------------------

    consensus_ok = (
        preferred_scale
        not in (
            None,
            "tie",
        )
        and
        max_models_agree
        is not None
        and
        max_models_agree >= 5
    )

    # --------------------------------------------------------
    # Target-scale gate
    # --------------------------------------------------------

    target_ok = (
        target_scale is None
        or
        preferred_scale
        ==
        target_scale
    )

    # --------------------------------------------------------
    # Robustness components
    # --------------------------------------------------------

    robustness_available = bool(
        robustness.get(
            "available",
            False,
        )
    )

    compatible_exclusions = (
        robustness.get(
            "compatible_exclusions"
        )
    )

    n_exclusions = robustness.get(
        "n_exclusions"
    )

    stability = robustness.get(
        "median_optimum_stability_fraction"
    )

    lomo_available = bool(
        lomo.get(
            "available",
            False,
        )
    )

    lomo_fraction = lomo.get(
        "compatibility_fraction"
    )

    robustness_summary_ok = (
        robustness_available
        and
        compatible_exclusions
        is not None
        and
        n_exclusions
        is not None
        and
        n_exclusions > 0
        and
        compatible_exclusions
        ==
        n_exclusions
        and
        stability
        is not None
        and
        stability >= 0.75
    )

    lomo_ok = (
        lomo_available
        and
        lomo_fraction
        is not None
        and
        lomo_fraction >= 0.875
    )

    robust_ok = (
        robustness_summary_ok
        and
        lomo_ok
    )

    evidence_conflict = bool(
        alignment.get(
            "conflict",
            False,
        )
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reasons: List[str] = []

    if not eligible_main:

        reasons.append(
            "Program is not eligible for the main analysis."
        )

    if not consensus_ok:

        reasons.append(
            "Cross-model consensus gate failed."
        )

    if not target_ok:

        reasons.append(
            "Vote-based preferred scale does not match "
            "the requested target scale."
        )

    if not robustness_available:

        reasons.append(
            "Robustness summary evidence is unavailable."
        )

    if robustness_available and not robustness_summary_ok:

        reasons.append(
            "Robustness gate failed."
        )

    if not lomo_available:

        reasons.append(
            "Leave-one-model-out evidence is unavailable."
        )

    elif not lomo_ok:

        reasons.append(
            "Leave-one-model-out compatibility gate failed."
        )

    if evidence_conflict:

        reasons.append(
            "Vote-based preferred scale and cross-model "
            "median optimum disagree."
        )

    # --------------------------------------------------------
    # Primary / raw status
    # --------------------------------------------------------

    if not eligible_main:

        status = (
            "not_eligible"
        )

    elif (
        not robustness_available
        or
        not lomo_available
    ):

        # Preserve explicit missing-evidence status where the
        # support question cannot be fully adjudicated.
        status = (
            "insufficient_robustness_evidence"
        )

    elif (
        consensus_ok
        and
        target_ok
        and
        robust_ok
    ):

        if evidence_conflict:

            # Backward-compatible raw status.
            # nodes.py separates this diagnostic from the
            # primary support status.
            status = (
                "conflicting_evidence"
            )

        else:

            status = (
                "supported_exploratory_hypothesis"
            )

            reasons.append(
                "Program passes cross-model consensus, "
                "robustness, and evidence-alignment checks."
            )

    else:

        status = (
            "not_supported"
        )

    return {
        "status":
            status,

        "consensus_ok":
            bool(
                consensus_ok
            ),

        "target_ok":
            bool(
                target_ok
            ),

        "robust_ok":
            bool(
                robust_ok
            ),

        "vote_preferred_scale":
            preferred_scale,

        "cross_model_median_optimum_scale":
            alignment.get(
                "cross_model_median_optimum_scale"
            ),

        "evidence_concordant":
            bool(
                alignment.get(
                    "concordant",
                    False,
                )
            ),

        "evidence_conflict":
            evidence_conflict,

        "reasons":
            reasons,
    }