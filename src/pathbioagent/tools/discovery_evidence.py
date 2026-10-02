from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


DEFAULT_TARGET = "overall_median_pearson"


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


def _modeling_dir(
    results_root: Optional[str | Path] = None,
) -> Path:

    root = _results_root(
        results_root
    )

    path = (
        root
        / "pathbioagent_discovery"
        / "modeling"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Modeling directory not found: {path}"
        )

    return path


def load_discovery_evidence(
    results_root: Optional[str | Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load deterministic modeling evidence produced by
    PathBioAgent discovery modules.
    """

    directory = _modeling_dir(
        results_root
    )

    paths = {
        "cv_summary":
            directory
            / "cv_summary.csv",

        "correlations":
            directory
            / "feature_target_correlations.csv",

        "importance":
            directory
            / "feature_importance.csv",
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing discovery evidence files:\n"
            + "\n".join(missing)
        )

    return {
        name: pd.read_csv(path)
        for name, path in paths.items()
    }


def _model_summary(
    cv_summary: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:

    subset = (
        cv_summary[
            cv_summary["target"]
            == target
        ]
        .copy()
    )

    if subset.empty:
        return []

    subset = subset.sort_values(
        [
            "mean_spearman",
            "mean_r2",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )

    columns = [
        "model",
        "mean_r2",
        "std_r2",
        "mean_mae",
        "mean_spearman",
        "std_spearman",
    ]

    return (
        subset[columns]
        .to_dict(
            orient="records"
        )
    )


def _top_correlations(
    correlations: pd.DataFrame,
    target: str,
    *,
    top_n: int,
) -> list[dict[str, Any]]:

    subset = (
        correlations[
            correlations["target"]
            == target
        ]
        .copy()
    )

    if subset.empty:
        return []

    subset[
        "abs_rho"
    ] = (
        subset[
            "spearman_rho"
        ]
        .abs()
    )

    subset = (
        subset
        .sort_values(
            "abs_rho",
            ascending=False,
        )
        .head(top_n)
    )

    columns = [
        "feature",
        "spearman_rho",
        "p_value",
        "n_genes",
    ]

    return (
        subset[columns]
        .to_dict(
            orient="records"
        )
    )


def _top_importance(
    importance: pd.DataFrame,
    target: str,
    *,
    model: str,
    top_n: int,
) -> list[dict[str, Any]]:

    subset = (
        importance[
            (
                importance["target"]
                == target
            )
            &
            (
                importance["model"]
                == model
            )
        ]
        .copy()
    )

    if subset.empty:
        return []

    subset = (
        subset
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(top_n)
    )

    columns = [
        "feature",
        "importance",
        "signed_effect",
    ]

    return (
        subset[columns]
        .to_dict(
            orient="records"
        )
    )


def build_discovery_evidence(
    *,
    target: str = DEFAULT_TARGET,
    top_n: int = 8,
    results_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Build an auditable discovery evidence object.

    No LLM reasoning occurs here.
    """

    evidence = load_discovery_evidence(
        results_root
    )

    cv_summary = evidence[
        "cv_summary"
    ]

    correlations = evidence[
        "correlations"
    ]

    importance = evidence[
        "importance"
    ]

    models = _model_summary(
        cv_summary,
        target,
    )

    return {
        "evidence_type":
            "molecular_visibility_discovery",

        "target":
            target,

        "model_performance":
            models,

        "top_univariate_associations":
            _top_correlations(
                correlations,
                target,
                top_n=top_n,
            ),

        "elastic_net_features":
            _top_importance(
                importance,
                target,
                model="elastic_net",
                top_n=top_n,
            ),

        "random_forest_features":
            _top_importance(
                importance,
                target,
                model="random_forest",
                top_n=top_n,
            ),

        "scientific_scope": (
            "These results are internal gene-level discovery evidence. "
            "RNA-derived expression features and morphology-derived "
            "molecular visibility were derived from the same primary "
            "63-case cohort. Cross-validation evaluates internal "
            "predictive structure, not external generalization."
        ),
    }


def format_discovery_evidence(
    evidence: Dict[str, Any],
) -> str:
    """
    Deterministic text formatter for CLI/debugging.
    """

    target = evidence.get(
        "target"
    )

    lines = [
        f"Discovery target: {target}",
        "",
        "Cross-validation:",
    ]

    for row in evidence.get(
        "model_performance",
        [],
    ):

        spearman = row.get(
            "mean_spearman"
        )

        if pd.isna(spearman):
            spearman_text = "NA"
        else:
            spearman_text = (
                f"{spearman:.3f}"
            )

        lines.append(
            "- "
            f"{row.get('model')}: "
            f"R2={row.get('mean_r2'):.3f}, "
            f"MAE={row.get('mean_mae'):.3f}, "
            f"Spearman={spearman_text}"
        )

    lines.extend(
        [
            "",
            "Top univariate associations:",
        ]
    )

    for row in evidence.get(
        "top_univariate_associations",
        [],
    ):

        lines.append(
            "- "
            f"{row.get('feature')}: "
            f"rho={row.get('spearman_rho'):.3f}, "
            f"p={row.get('p_value'):.3g}"
        )

    lines.extend(
        [
            "",
            "Top Elastic Net features:",
        ]
    )

    for row in evidence.get(
        "elastic_net_features",
        [],
    ):

        lines.append(
            "- "
            f"{row.get('feature')}: "
            f"importance={row.get('importance'):.4f}, "
            f"signed_effect="
            f"{row.get('signed_effect'):.4f}"
        )

    lines.extend(
        [
            "",
            "Scope:",
            evidence.get(
                "scientific_scope",
                "",
            ),
        ]
    )

    return "\n".join(
        lines
    )


if __name__ == "__main__":

    result = build_discovery_evidence()

    print(
        format_discovery_evidence(
            result
        )
    )
