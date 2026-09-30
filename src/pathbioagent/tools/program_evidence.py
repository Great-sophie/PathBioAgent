import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv


load_dotenv()


ROOT = Path(
    os.getenv(
        "PATHBIO_RESULTS_ROOT",
        "/media/sophie/Data2T/Paper3_8model_gene_analysis",
    )
)


FILES = {
    "consensus": ROOT
    / "program_visibility"
    / "program_cross_model_consensus.csv",

    "cross_model": ROOT
    / "program_visibility"
    / "program_cross_model_visibility.csv",

    "preferred_scale": ROOT
    / "program_visibility"
    / "program_preferred_scale_by_model.csv",

    "robustness": ROOT
    / "final_figures"
    / "Figure6"
    / "source_data"
    / "Figure6_program_robustness_summary.csv",

    "lomo": ROOT
    / "final_figures"
    / "Figure6"
    / "source_data"
    / "Figure6B_leave_one_model_out_vote_consensus.csv",
}


def load_tables() -> Dict[str, pd.DataFrame]:
    tables = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file [{name}]: {path}"
            )

        tables[name] = pd.read_csv(path)

    return tables


def infer_target_scale(
    query: str,
) -> Optional[str]:

    q = query.lower()

    if "2.5x" in q or "2p5x" in q:
        return "2p5x"

    if "10x" in q:
        return "10x"

    if "40x" in q:
        return "40x"

    return None


def search_program_candidates(
    query: str,
) -> List[str]:

    tables = load_tables()

    consensus = tables["consensus"].copy()
    cross_model = tables["cross_model"].copy()

    consensus = consensus[
        consensus["eligible_main"] == True
    ].copy()

    target_scale = infer_target_scale(query)

    if target_scale is not None:

        vote_col = f"n_models_prefer_{target_scale}"

        scale_visibility = cross_model[
            cross_model["scale"] == target_scale
        ][
            [
                "program",
                "median_across_models",
            ]
        ]

        ranked = consensus.merge(
            scale_visibility,
            on="program",
            how="left",
        )

        ranked = ranked.sort_values(
            [
                vote_col,
                "median_across_models",
            ],
            ascending=[
                False,
                False,
            ],
        )

    else:

        ranked = consensus.sort_values(
            [
                "max_models_agree",
                "coverage_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

    return ranked["program"].tolist()


def collect_program_evidence(
    program: str,
) -> Dict[str, Any]:

    tables = load_tables()

    consensus_df = tables["consensus"]
    cross_df = tables["cross_model"]
    robustness_df = tables["robustness"]
    lomo_df = tables["lomo"]

    # --------------------------------------------------
    # Consensus
    # --------------------------------------------------

    rows = consensus_df[
        consensus_df["program"] == program
    ]

    if rows.empty:
        raise ValueError(
            f"Unknown program: {program}"
        )

    row = rows.iloc[0]

    consensus = {
        "preferred_scale":
            str(row["consensus_preferred_scale"]),

        "max_models_agree":
            int(row["max_models_agree"]),

        "votes": {
            "2p5x":
                int(row["n_models_prefer_2p5x"]),

            "10x":
                int(row["n_models_prefer_10x"]),

            "40x":
                int(row["n_models_prefer_40x"]),
        },

        "coverage_fraction":
            float(row["coverage_fraction"]),
    }

    # --------------------------------------------------
    # Cross-model visibility
    # --------------------------------------------------

    cross_rows = cross_df[
        cross_df["program"] == program
    ].copy()

    cross_visibility = {}

    for _, r in cross_rows.iterrows():

        cross_visibility[
            str(r["scale"])
        ] = {
            "median":
                float(r["median_across_models"]),

            "mean":
                float(r["mean_across_models"]),

            "min":
                float(r["min_across_models"]),

            "max":
                float(r["max_across_models"]),
        }

    best_row = cross_rows.sort_values(
        "median_across_models",
        ascending=False,
    ).iloc[0]

    # --------------------------------------------------
    # Robustness
    # --------------------------------------------------

    robust_rows = robustness_df[
        robustness_df["program"] == program
    ]

    if robust_rows.empty:

        robustness = {
            "available": False
        }

    else:

        r = robust_rows.iloc[0]

        robustness = {
            "available":
                True,

            "compatible_exclusions":
                int(r["compatible_exclusions"]),

            "n_exclusions":
                int(r["n_exclusions"]),

            "median_optimum_stability_fraction":
                float(
                    r[
                        "median_optimum_stability_fraction"
                    ]
                ),
        }

    # --------------------------------------------------
    # Leave-one-model-out
    # --------------------------------------------------

    lomo_rows = lomo_df[
        lomo_df["program"] == program
    ]

    if lomo_rows.empty:

        lomo = {
            "available": False
        }

    else:

        n_total = len(lomo_rows)

        n_compatible = int(
            lomo_rows[
                "compatible_with_full"
            ].sum()
        )

        lomo = {
            "available":
                True,

            "n_exclusions":
                n_total,

            "compatible_exclusions":
                n_compatible,

            "compatibility_fraction":
                n_compatible / n_total,
        }

    return {
        "program":
            program,

        "consensus":
            consensus,

        "cross_model_visibility":
            cross_visibility,

        "cross_model_optimum": {
            "scale":
                str(best_row["scale"]),

            "median_visibility":
                float(
                    best_row[
                        "median_across_models"
                    ]
                ),
        },

        "robustness":
            robustness,

        "lomo":
            lomo,
    }


def evaluate_program_evidence(
    evidence: Dict[str, Any],
    target_scale: Optional[str] = None,
) -> Dict[str, Any]:

    consensus = evidence["consensus"]
    robustness = evidence["robustness"]
    lomo = evidence["lomo"]

    if not robustness["available"]:

        return {
            "status":
                "insufficient_robustness_evidence"
        }

    if not lomo["available"]:

        return {
            "status":
                "insufficient_robustness_evidence"
        }

    consensus_ok = (
        consensus["preferred_scale"] != "tie"
        and
        consensus["max_models_agree"] >= 5
    )

    target_ok = (
        target_scale is None
        or
        consensus["preferred_scale"]
        == target_scale
    )

    robust_ok = (
        robustness["compatible_exclusions"]
        ==
        robustness["n_exclusions"]
        and
        robustness[
            "median_optimum_stability_fraction"
        ]
        >= 0.75
        and
        lomo["compatibility_fraction"]
        >= 0.875
    )

    if (
        consensus_ok
        and
        target_ok
        and
        robust_ok
    ):

        status = (
            "supported_exploratory_hypothesis"
        )

    else:

        status = "not_supported"

    return {
        "status":
            status,

        "consensus_ok":
            consensus_ok,

        "target_ok":
            target_ok,

        "robust_ok":
            robust_ok,
    }