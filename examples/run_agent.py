import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pathbioagent.graph import build_graph


# ============================================================
# CLI arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PathBioAgent: evidence-driven LangGraph agent "
            "for multiscale morphology-molecular hypothesis "
            "analysis."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            "Scientific question, e.g. "
            "'Which programs have the most robust "
            "10x preference?'"
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
        help=(
            "Print the complete LangGraph execution trace."
        ),
    )

    parser.add_argument(
        "--full-cards",
        action="store_true",
        help=(
            "Print complete Evidence Cards as JSON."
        ),
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
    programs: List[str],
) -> None:
    print()
    print(title)

    if not programs:
        print("  None")
        return

    for index, program in enumerate(
        programs,
        start=1,
    ):
        print(
            f"  {index}. {program}"
        )


def format_optional_float(
    value: Any,
) -> str:
    """
    Format numerical values safely.

    Returns 'None' when value is unavailable.
    """

    if value is None:
        return "None"

    try:
        return f"{float(value):.3f}"

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


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
        "vote_preferred_scale"
    )

    median_scale = card.get(
        "cross_model_median_optimum_scale"
    )

    agreement = card.get(
        "max_models_agree"
    )

    votes = card.get(
        "scale_votes",
        {},
    ) or {}

    conflict = card.get(
        "evidence_conflict",
        False,
    )

    stability = card.get(
        "median_optimum_stability_fraction"
    )

    lomo = card.get(
        "lomo_compatibility_fraction"
    )

    flags = card.get(
        "evidence_flags",
        {},
    ) or {}

    reasons = card.get(
        "reasons",
        [],
    ) or []

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
            f"{format_optional_float(stability)}"
        )

    if lomo is not None:
        print(
            "LOMO compatibility: "
            f"{format_optional_float(lomo)}"
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

    if reasons:
        print("Reasons:")

        for reason in reasons:
            print(
                f"  - {reason}"
            )


# ============================================================
# Full Evidence Card
# ============================================================

def print_full_card(
    card: Dict[str, Any],
) -> None:
    print()
    print("-" * 78)

    print(
        json.dumps(
            card,
            indent=2,
            ensure_ascii=False,
        )
    )


# ============================================================
# Report access helpers
# ============================================================

def get_supported(
    report: Dict[str, Any],
) -> List[str]:
    return report.get(
        "supported_exploratory_hypotheses",
        report.get(
            "supported",
            [],
        ),
    )


def get_not_supported(
    report: Dict[str, Any],
) -> List[str]:
    return report.get(
        "not_supported",
        [],
    )


def get_insufficient(
    report: Dict[str, Any],
) -> List[str]:
    return report.get(
        "insufficient_robustness_evidence",
        report.get(
            "insufficient",
            [],
        ),
    )


def get_not_eligible(
    report: Dict[str, Any],
) -> List[str]:
    return report.get(
        "not_eligible",
        [],
    )


# ============================================================
# Main report printer
# ============================================================

def print_report(
    report: Dict[str, Any],
    *,
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

    parser_source = report.get(
        "parser_source"
    )

    if parser_source is not None:
        print(
            "Query parser:",
            parser_source,
        )

    confidence = report.get(
        "query_confidence"
    )

    if confidence is not None:
        print(
            "Query confidence:",
            format_optional_float(
                confidence
            ),
        )

    model_name = report.get(
        "model_name"
    )

    if model_name:
        print(
            "LLM model:",
            model_name,
        )

    # --------------------------------------------------------
    # Candidate accounting
    # --------------------------------------------------------

    n_tested = report.get(
        "candidates_tested",
        report.get(
            "n_tested",
            0,
        ),
    )

    n_candidates = report.get(
        "total_candidates",
        report.get(
            "n_candidates",
            0,
        ),
    )

    print(
        "Candidates tested:",
        f"{n_tested}/{n_candidates}",
    )

    # ========================================================
    # Primary decisions
    # ========================================================

    print_header(
        "PRIMARY DECISIONS"
    )

    print_program_group(
        "SUPPORTED EXPLORATORY HYPOTHESES",
        get_supported(
            report
        ),
    )

    print_program_group(
        "NOT SUPPORTED",
        get_not_supported(
            report
        ),
    )

    print_program_group(
        "INSUFFICIENT ROBUSTNESS EVIDENCE",
        get_insufficient(
            report
        ),
    )

    print_program_group(
        "NOT ELIGIBLE",
        get_not_eligible(
            report
        ),
    )

    unknown = report.get(
        "unknown_status",
        [],
    )

    if unknown:
        print_program_group(
            "UNKNOWN STATUS",
            unknown,
        )

    # ========================================================
    # Evidence diagnostics
    # ========================================================

    print_header(
        "EVIDENCE DIAGNOSTICS"
    )

    print_program_group(
        "AGGREGATION CONFLICT",
        report.get(
            "evidence_conflict_programs",
            report.get(
                "aggregation_conflict",
                [],
            ),
        ),
    )

    print_program_group(
        "TIE CONSENSUS",
        report.get(
            "tie_consensus_programs",
            report.get(
                "tie_consensus",
                [],
            ),
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

    missing_lomo = report.get(
        "missing_lomo_programs",
        [],
    )

    if missing_lomo:
        print_program_group(
            "MISSING LOMO",
            missing_lomo,
        )

    weak_consensus = report.get(
        "weak_model_consensus_programs",
        [],
    )

    if weak_consensus:
        print_program_group(
            "WEAK MODEL CONSENSUS",
            weak_consensus,
        )

    # ========================================================
    # Evidence Cards
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
                print_full_card(
                    card
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

    interpretation_scope = (
        report.get(
            "interpretation_scope"
        )
    )

    if interpretation_scope:
        print(
            interpretation_scope
        )

    else:
        print(
            "Results are exploratory and should not be "
            "interpreted as clinical biomarker validation, "
            "biological causality, or patient-level "
            "molecular diagnosis."
        )


# ============================================================
# Trace
# ============================================================

def print_trace(
    trace: List[str],
) -> None:
    print_header(
        "AGENT TRACE"
    )

    if not trace:
        print(
            "No trace available."
        )
        return

    for index, event in enumerate(
        trace,
        start=1,
    ):
        print(
            f"{index:02d}. {event}"
        )


# ============================================================
# Save JSON
# ============================================================

def save_json_report(
    output_path: str,
    *,
    report: Dict[str, Any],
    trace: List[str],
) -> None:
    path = Path(
        output_path
    )

    if path.parent != Path("."):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    payload = {
        "report":
            report,

        "trace":
            trace,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved JSON report:",
        path,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.query.strip():
        raise ValueError(
            "--query cannot be empty."
        )

    if args.max_iterations < 1:
        raise ValueError(
            "--max-iterations must be >= 1."
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
    # Build graph
    # --------------------------------------------------------

    graph = build_graph()

    # --------------------------------------------------------
    # Invoke Agent
    # --------------------------------------------------------

    result = graph.invoke(
        {
            "query":
                args.query,

            "max_iterations":
                args.max_iterations,

            "trace":
                [],
        }
    )

    # --------------------------------------------------------
    # Extract final report
    # --------------------------------------------------------

    report = result.get(
        "final_report"
    )

    if report is None:
        raise RuntimeError(
            "PathBioAgent finished without "
            "producing final_report."
        )

    trace = result.get(
        "trace",
        [],
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(
        report,
        full_cards=args.full_cards,
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    if args.show_trace:
        print_trace(
            trace
        )

    # --------------------------------------------------------
    # Optional JSON output
    # --------------------------------------------------------

    if args.save_json:
        save_json_report(
            args.save_json,
            report=report,
            trace=trace,
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()