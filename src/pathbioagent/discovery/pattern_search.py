from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


def _results_root(
    results_root: Optional[str | Path] = None,
) -> Path:
    if results_root is not None:
        return Path(results_root)

    env = os.getenv("PATHBIO_RESULTS_ROOT")

    if not env:
        raise RuntimeError(
            "PATHBIO_RESULTS_ROOT is not set."
        )

    return Path(env)


def load_gene_discovery_table(
    results_root: Optional[str | Path] = None,
) -> pd.DataFrame:

    root = _results_root(
        results_root
    )

    path = (
        root
        / "pathbioagent_discovery"
        / "gene_discovery_master.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Discovery table not found: {path}"
        )

    return pd.read_csv(path)


def _top(
    df: pd.DataFrame,
    column: str,
    *,
    n: int = 20,
    ascending: bool = False,
) -> pd.DataFrame:

    return (
        df.sort_values(
            column,
            ascending=ascending,
        )
        .head(n)
        .copy()
    )


def discover_patterns(
    results_root: Optional[str | Path] = None,
    *,
    top_n: int = 20,
):
    """
    Deterministically identify major gene-level
    molecular-visibility patterns.

    No LLM inference occurs here.
    """

    df = load_gene_discovery_table(
        results_root
    )

    # ========================================================
    # 1. Highest absolute molecular visibility
    # ========================================================

    high_visibility = _top(
        df,
        "best_scale_median_pearson",
        n=top_n,
    )

    # ========================================================
    # 2. Strongest 10x advantage
    # ========================================================

    ten_x_advantage = (
        df.assign(
            ten_x_advantage=lambda x:
                x[
                    [
                        "delta_10x_minus_40x",
                        "delta_10x_minus_2p5x",
                    ]
                ].min(axis=1)
        )
        .sort_values(
            "ten_x_advantage",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # 3. Strongest 40x advantage
    # ========================================================

    forty_x_advantage = (
        df.assign(
            forty_x_advantage=lambda x:
                pd.concat(
                    [
                        -x[
                            "delta_10x_minus_40x"
                        ],
                        x[
                            "delta_40x_minus_2p5x"
                        ],
                    ],
                    axis=1,
                ).min(axis=1)
        )
        .sort_values(
            "forty_x_advantage",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # 4. Strongest 2.5x advantage
    # ========================================================

    two_p5_x_advantage = (
        df.assign(
            two_p5_x_advantage=lambda x:
                pd.concat(
                    [
                        -x[
                            "delta_10x_minus_2p5x"
                        ],
                        -x[
                            "delta_40x_minus_2p5x"
                        ],
                    ],
                    axis=1,
                ).min(axis=1)
        )
        .sort_values(
            "two_p5_x_advantage",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # 5. Stable cross-model preference
    # ========================================================

    stable_preference = (
        df[
            df[
                "vote_preferred_scale"
            ]
            != "tie"
        ]
        .sort_values(
            [
                "vote_consensus_fraction",
                "median_model_scale_margin",
                "best_scale_median_pearson",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # 6. Median-vote disagreement
    # ========================================================

    disagreement = (
        df[
            ~df[
                "median_vote_concordant"
            ]
        ]
        .sort_values(
            [
                "best_scale_median_pearson",
                "vote_consensus_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # 7. Strong scale dependence
    # ========================================================

    scale_sensitive = _top(
        df,
        "median_scale_range",
        n=top_n,
    )

    # ========================================================
    # 8. Summary statistics
    # ========================================================

    summary = {
        "n_genes": int(
            len(df)
        ),

        "median_vote_concordance_fraction":
            float(
                df[
                    "median_vote_concordant"
                ].mean()
            ),

        "median_vote_disagreement_fraction":
            float(
                1.0
                -
                df[
                    "median_vote_concordant"
                ].mean()
            ),

        "vote_tie_fraction":
            float(
                df[
                    "vote_tie"
                ].mean()
            ),

        "strong_model_consensus_fraction":
            float(
                df[
                    "strong_model_consensus"
                ].mean()
            ),

        "median_best_visibility":
            float(
                df[
                    "best_scale_median_pearson"
                ].median()
            ),

        "median_scale_range":
            float(
                df[
                    "median_scale_range"
                ].median()
            ),
    }

    return {
        "summary": summary,
        "high_visibility": high_visibility,
        "ten_x_advantage": ten_x_advantage,
        "forty_x_advantage": forty_x_advantage,
        "two_p5_x_advantage": two_p5_x_advantage,
        "stable_preference": stable_preference,
        "disagreement": disagreement,
        "scale_sensitive": scale_sensitive,
    }


def save_pattern_outputs(
    results_root: Optional[str | Path] = None,
    *,
    top_n: int = 20,
):

    root = _results_root(
        results_root
    )

    output_dir = (
        root
        / "pathbioagent_discovery"
        / "patterns"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = discover_patterns(
        root,
        top_n=top_n,
    )

    for name, value in results.items():

        if name == "summary":
            continue

        value.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    summary = pd.DataFrame(
        [
            results["summary"]
        ]
    )

    summary.to_csv(
        output_dir / "discovery_summary.csv",
        index=False,
    )

    return results


if __name__ == "__main__":

    results = save_pattern_outputs()

    summary = results[
        "summary"
    ]

    print("=" * 72)
    print("PathBioAgent Gene Discovery")
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Top molecularly visible genes:"
    )

    print(
        results[
            "high_visibility"
        ][
            [
                "gene",
                "best_scale_median_pearson",
                "derived_median_optimum_scale",
                "vote_preferred_scale",
                "vote_consensus_fraction",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Highest median-vote disagreements:"
    )

    print(
        results[
            "disagreement"
        ][
            [
                "gene",
                "best_scale_median_pearson",
                "derived_median_optimum_scale",
                "vote_preferred_scale",
                "vote_consensus_fraction",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )
