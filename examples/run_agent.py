import argparse
import json
from pathlib import Path
from typing import Any, Dict

from pathbioagent.graph import build_graph


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PathBioAgent: evidence-driven LangGraph agent "
            "for multiscale morphology-molecular hypothesis analysis."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            "Scientific question, e.g. "
            "'Which programs prefer 40x?'"
        ),
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help=(
            "Maximum number of candidate programs "
            "to evaluate. Default: 10."
        ),
    )

    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the full LangGraph execution trace.",
    )

    parser.add_argument(
        "--full-cards",
        action="store_true",
        help="Print complete Evidence Cards.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help=(
            "Optional output JSON path, e.g. "
            "outputs/report.json"
        ),
    )

    return parser.parse_args()


# ============================================================
# Formatting helpers
# ============================================================

def print_header(
    title: str,
) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_program_group(
    title: str,
    programs: list,
) -> None:
    print()
    print(title)

    if not programs:
        print("  None")
        return

    for i, program in enumerate(
        programs,
        start=1,
    ):
        print(
            f"  {i}. {program}"
        )


# ============================================================
# Compact Evidence Card
# ============================================================

def print_compact_card(
    card: Dict[str, Any],
) -> None:

    program = card.get(
        "program",
        "unknown",
    )

    status = card.get(
        "status",
        "unknown",
    )

    vote_scale = card.get(
        "vote_preferred_scale",
    )

    median_scale = card.get(
        "cross_model_median_optimum_scale",
    )

    agreement = card.get(
        "max_models_agree",
    )

    votes = card.get(
        "scale_votes",
        {},
    )

    conflict = card.get(
        "evidence_conflict",
        False,
    )

    stability = card.get(
        "median_optimum_stability_fraction",
    )

    lomo = card.get(
        "lomo_compatibility_fraction",
    )

    flags = card.get(
        "evidence_flags",
        {},
    )

    print()
    print("-" * 78)

    print(
        f"Program: {program}"
    )

    print(
        f"Status: {status}"
    )

    print(
        "Vote preferred scale: "
        f"{vote_scale}"
    )

    print(
        "Cross-model median optimum: "
        f"{median_scale}"
    )

    if agreement is not None:

        print(
            "Model agreement: "
            f"{agreement}/8"
        )

    if votes:

        print(
            "Scale votes: "
            f"2.5x={votes.get('2p5x')}, "
            f"10x={votes.get('10x')}, "
            f"40x={votes.get('40x')}"
        )

    print(
        "Evidence conflict: "
        f"{conflict}"
    )

    if stability is not None:

        print(
            "Optimum stability: "
            f"{stability:.3f}"
        )

    if lomo is not None:

        print(
            "LOMO compatibility: "
            f"{lomo:.3f}"
        )

    active_flags = [
        name
        for name, value in flags.items()
        if value
    ]

    if active_flags:

        print(
            "Evidence flags: "
            + ", ".join(
                active_flags
            )
        )

    else:

        print(
            "Evidence flags: none"
        )

    reasons = card.get(
        "reasons",
        [],
    )

    if reasons:

        print("Reasons:")

        for reason in reasons:
            print(
                f"  - {reason}"
            )


# ============================================================
# Report output
# ============================================================

def print_report(
    report: Dict[str, Any],
    full_cards: bool = False,
) -> None:

    print_header(
        "PATHBIOAGENT REPORT"
    )

    print(
        "Query:",
        report.get(
            "query"
        ),
    )

    print(
        "Ranking intent:",
        report.get(
            "ranking_intent"
        ),
    )

    print(
        "Target scale:",
        report.get(
            "target_scale"
        ),
    )

    print(
        "Candidates tested:",
        f"{report.get('n_tested', 0)}"
        f"/{report.get('n_candidates', 0)}",
    )

    # ========================================================
    # Primary decisions
    # ========================================================

    print_header(
        "PRIMARY DECISIONS"
    )

    print_program_group(
        "SUPPORTED EXPLORATORY HYPOTHESES",
        report.get(
            "supported_exploratory_hypotheses",
            [],
        ),
    )

    print_program_group(
        "NOT SUPPORTED",
        report.get(
            "not_supported",
            [],
        ),
    )

    print_program_group(
        "INSUFFICIENT ROBUSTNESS EVIDENCE",
        report.get(
            "insufficient_robustness_evidence",
            [],
        ),
    )

    print_program_group(
        "NOT ELIGIBLE",
        report.get(
            "not_eligible",
            [],
        ),
    )

    # ========================================================
    # Independent evidence diagnostics
    # ========================================================

    print_header(
        "EVIDENCE DIAGNOSTICS"
    )

    print_program_group(
        "AGGREGATION CONFLICT",
        report.get(
            "evidence_conflict_programs",
            [],
        ),
    )

    print_program_group(
        "TIE CONSENSUS",
        report.get(
            "tie_consensus_programs",
            [],
        ),
    )

    print_program_group(
        "LOW OPTIMUM STABILITY",
        report.get(
            "low_optimum_stability_programs",
            [],
        ),
    )

    print_program_group(
        "LOW LOMO COMPATIBILITY",
        report.get(
            "low_lomo_compatibility_programs",
            [],
        ),
    )

    print_program_group(
        "MISSING ROBUSTNESS",
        report.get(
            "missing_robustness_programs",
            [],
        ),
    )

    # ========================================================
    # Evidence cards
    # ========================================================

    print_header(
        "EVIDENCE CARDS"
    )

    cards = report.get(
        "evidence_cards",
        [],
    )

    if not cards:

        print(
            "No Evidence Cards available."
        )

    else:

        for card in cards:

            if full_cards:

                print()
                print("-" * 78)

                print(
                    json.dumps(
                        card,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            else:

                print_compact_card(
                    card
                )

    # ========================================================
    # Interpretation scope
    # ========================================================

    print_header(
        "INTERPRETATION SCOPE"
    )

    print(
        report.get(
            "interpretation_scope",
            "Not available.",
        )
    )


# ============================================================
# Save JSON
# ============================================================

def save_report(
    report: Dict[str, Any],
    output_path: str,
) -> None:

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved report to: {path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.max_iterations < 1:

        raise ValueError(
            "--max-iterations must be >= 1"
        )

    print_header(
        "PathBioAgent"
    )

    print(
        "Query:",
        args.query,
    )

    print(
        "Max iterations:",
        args.max_iterations,
    )

    # --------------------------------------------------------
    # Build LangGraph
    # --------------------------------------------------------

    graph = build_graph()

    initial_state = {
        "query":
            args.query,

        "max_iterations":
            args.max_iterations,

        "trace":
            [],
    }

    # --------------------------------------------------------
    # Run Agent
    # --------------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    final_report = result.get(
        "final_report"
    )

    if final_report is None:

        raise RuntimeError(
            "Agent completed without producing "
            "a final_report."
        )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(
        final_report,
        full_cards=args.full_cards,
    )

    # --------------------------------------------------------
    # Optional trace
    # --------------------------------------------------------

    if args.show_trace:

        print_header(
            "AGENT TRACE"
        )

        trace = result.get(
            "trace",
            [],
        )

        if not trace:

            print(
                "No trace available."
            )

        else:

            for i, step in enumerate(
                trace,
                start=1,
            ):

                print(
                    f"{i:02d}. {step}"
                )

    # --------------------------------------------------------
    # Optional JSON output
    # --------------------------------------------------------

    if args.save_json:

        save_report(
            final_report,
            args.save_json,
        )


if __name__ == "__main__":
    main()