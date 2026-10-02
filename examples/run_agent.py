#!/usr/bin/env python3

"""
Command-line runner for PathBioAgent.

Core design
-----------
The LangGraph workflow remains deterministic:

    query
      ↓
    structured query interpretation
      ↓
    candidate search
      ↓
    evidence collection
      ↓
    evidence gate
      ↓
    final_report

Optional:

    final_report
      ↓
    --explain
      ↓
    LLM scientific interpretation

The LLM explanation layer does NOT modify:
    - ranking
    - evidence
    - evidence gate
    - candidate status
"""


import argparse
import json
import sys
from typing import Any, Dict, List, Optional


from pathbioagent.graph import build_graph



# ============================================================
# Argument parser
# ============================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run PathBioAgent on a morphology-molecular "
            "evidence query."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Scientific query for PathBioAgent.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help=(
            "Maximum number of candidate evaluation "
            "iterations. Default: 10."
        ),
    )

    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the auditable LangGraph execution trace.",
    )

    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print compact evidence cards.",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Generate an optional LLM scientific explanation "
            "after deterministic evidence analysis."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete final report as JSON.",
    )

    return parser



# ============================================================
# Formatting helpers
# ============================================================


def _print_header(
    title: str,
) -> None:
    """
    Print section header.
    """

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)



def _safe_list(
    value: Any,
) -> List[Any]:
    """
    Convert optional report value into a list.
    """

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    return [value]



def _format_value(
    value: Any,
) -> str:
    """
    Human-readable formatting for CLI values.
    """

    if value is None:
        return "None"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.3f}"

    return str(value)



# ============================================================
# Query metadata
# ============================================================


def print_query_summary(
    result: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """
    Print structured query interpretation.
    """

    structured = result.get(
        "structured_query",
        {},
    )

    if not isinstance(
        structured,
        dict,
    ):
        structured = {}


    ranking_intent = report.get(
        "ranking_intent",
        structured.get(
            "intent"
        ),
    )


    target_scale = report.get(
        "target_scale",
        structured.get(
            "target_scale"
        ),
    )


    entity = report.get(
        "entity",
        structured.get(
            "entity"
        ),
    )


    parser_source = report.get(
        "parser_source",
        structured.get(
            "parser_source"
        ),
    )


    confidence = report.get(
        "query_confidence",
        structured.get(
            "confidence"
        ),
    )


    model_name = report.get(
        "model_name",
        structured.get(
            "model_name"
        ),
    )


    _print_header(
        "QUERY INTERPRETATION"
    )


    print(
        "Query:",
        report.get(
            "query",
            result.get(
                "query",
                ""
            ),
        ),
    )


    print(
        "Ranking intent:",
        _format_value(
            ranking_intent
        ),
    )


    print(
        "Target scale:",
        _format_value(
            target_scale
        ),
    )


    print(
        "Entity:",
        _format_value(
            entity
        ),
    )


    print(
        "Query parser:",
        _format_value(
            parser_source
        ),
    )


    print(
        "Query confidence:",
        _format_value(
            confidence
        ),
    )


    print(
        "LLM model:",
        _format_value(
            model_name
        ),
    )



# ============================================================
# Candidate summary
# ============================================================


def print_candidate_summary(
    report: Dict[str, Any],
) -> None:
    """
    Print candidate testing summary.
    """

    tested = report.get(
        "candidates_tested",
        report.get(
            "n_tested",
            None,
        ),
    )


    total = report.get(
        "total_candidates",
        report.get(
            "n_candidates",
            None,
        ),
    )


    supported = _safe_list(
        report.get(
            "supported_exploratory_hypotheses"
        )
    )


    not_supported = _safe_list(
        report.get(
            "not_supported"
        )
    )


    insufficient = _safe_list(
        report.get(
            "insufficient_robustness_evidence"
        )
    )


    not_eligible = _safe_list(
        report.get(
            "not_eligible"
        )
    )


    _print_header(
        "DETERMINISTIC EVIDENCE RESULT"
    )


    if (
        tested is not None
        or total is not None
    ):
        print(
            "Candidates tested:",
            f"{tested}/{total}",
        )


    print()


    print(
        "Supported exploratory hypotheses:"
    )


    if supported:

        for program in supported:

            print(
                f"  - {program}"
            )

    else:

        print(
            "  None"
        )


    if not_supported:

        print()
        print(
            "Not supported:"
        )

        for program in not_supported:

            print(
                f"  - {program}"
            )


    if insufficient:

        print()
        print(
            "Insufficient robustness evidence:"
        )

        for program in insufficient:

            print(
                f"  - {program}"
            )


    if not_eligible:

        print()
        print(
            "Not eligible:"
        )

        for program in not_eligible:

            print(
                f"  - {program}"
            )



# ============================================================
# Evidence cards
# ============================================================


def print_evidence_cards(
    report: Dict[str, Any],
) -> None:
    """
    Print evidence cards in compact readable form.
    """

    cards = _safe_list(
        report.get(
            "evidence_cards"
        )
    )


    _print_header(
        "EVIDENCE CARDS"
    )


    if not cards:

        print(
            "No evidence cards available."
        )

        return


    for index, card in enumerate(
        cards,
        start=1,
    ):

        if not isinstance(
            card,
            dict,
        ):

            print(
                f"{index}. {card}"
            )

            continue


        print(
            f"{index}. "
            f"{card.get('program', 'Unknown')}"
        )


        print(
            "   status:",
            _format_value(
                card.get(
                    "status"
                )
            ),
        )


        print(
            "   vote preferred scale:",
            _format_value(
                card.get(
                    "vote_preferred_scale"
                )
            ),
        )


        print(
            "   model agreement:",
            _format_value(
                card.get(
                    "max_models_agree"
                )
            ),
        )


        print(
            "   median optimum scale:",
            _format_value(
                card.get(
                    "cross_model_median_optimum_scale"
                )
            ),
        )


        print(
            "   optimum stability:",
            _format_value(
                card.get(
                    "median_optimum_stability_fraction"
                )
            ),
        )


        print(
            "   LOMO compatibility:",
            _format_value(
                card.get(
                    "lomo_compatibility_fraction"
                )
            ),
        )


        flags = card.get(
            "evidence_flags"
        )


        if flags:

            print(
                "   evidence flags:",
                flags,
            )


        reasons = card.get(
            "reasons"
        )


        if reasons:

            print(
                "   reasons:",
                reasons,
            )


        print()



# ============================================================
# Trace
# ============================================================


def print_trace(
    result: Dict[str, Any],
) -> None:
    """
    Print execution trace.
    """

    trace = _safe_list(
        result.get(
            "trace"
        )
    )


    _print_header(
        "EXECUTION TRACE"
    )


    if not trace:

        print(
            "No trace entries available."
        )

        return


    for index, entry in enumerate(
        trace,
        start=1,
    ):

        print(
            f"{index}. {entry}"
        )



# ============================================================
# Optional LLM explanation
# ============================================================


def generate_optional_explanation(
    report: Dict[str, Any],
) -> Optional[str]:
    """
    Generate optional scientific explanation.

    Import is intentionally lazy so ordinary deterministic
    execution does not require the report-generation LLM.
    """

    try:

        from pathbioagent.llm.report_generator import (
            generate_scientific_summary,
        )


        explanation = (
            generate_scientific_summary(
                report
            )
        )


        if explanation is None:

            return None


        return str(
            explanation
        ).strip()


    except Exception as exc:

        print(
            "\nWarning: scientific explanation "
            f"generation failed: {exc}",
            file=sys.stderr,
        )

        return None



# ============================================================
# JSON output
# ============================================================


def print_json_report(
    report: Dict[str, Any],
) -> None:
    """
    Print complete report as JSON.
    """

    _print_header(
        "FINAL REPORT JSON"
    )


    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )



# ============================================================
# Main
# ============================================================


def main() -> int:
    """
    Run PathBioAgent.
    """

    parser = (
        build_argument_parser()
    )


    args = parser.parse_args()


    if args.max_iterations < 1:

        parser.error(
            "--max-iterations must be >= 1"
        )


    # --------------------------------------------------------
    # Build deterministic LangGraph
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
    # Run agent
    # --------------------------------------------------------

    try:

        result = graph.invoke(
            initial_state
        )

    except Exception as exc:

        print(
            "PathBioAgent execution failed:",
            exc,
            file=sys.stderr,
        )

        return 1


    if not isinstance(
        result,
        dict,
    ):

        print(
            "PathBioAgent returned an invalid graph state.",
            file=sys.stderr,
        )

        return 1


    report = result.get(
        "final_report",
        {},
    )


    if not isinstance(
        report,
        dict,
    ):

        print(
            "PathBioAgent did not return a valid final_report.",
            file=sys.stderr,
        )

        return 1


    # --------------------------------------------------------
    # Core deterministic result
    # --------------------------------------------------------

    print_query_summary(
        result,
        report,
    )


    print_candidate_summary(
        report
    )


    # --------------------------------------------------------
    # Optional evidence display
    # --------------------------------------------------------

    if args.show_evidence:

        print_evidence_cards(
            report
        )


    # --------------------------------------------------------
    # Optional trace
    # --------------------------------------------------------

    if args.show_trace:

        print_trace(
            result
        )


    # --------------------------------------------------------
    # Optional LLM explanation
    # --------------------------------------------------------

    if args.explain:

        explanation = (
            generate_optional_explanation(
                report
            )
        )


        _print_header(
            "SCIENTIFIC INTERPRETATION"
        )


        if explanation:

            print(
                explanation
            )

        else:

            print(
                "Scientific interpretation "
                "was not generated."
            )


    # --------------------------------------------------------
    # Optional JSON
    # --------------------------------------------------------

    if args.json:

        print_json_report(
            report
        )


    return 0



if __name__ == "__main__":
    raise SystemExit(
        main()
    )