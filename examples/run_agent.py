#!/usr/bin/env python3

"""
PathBioAgent command line runner.

Supports:
    - natural language query
    - structured query interpretation
    - evidence-grounded report
    - scientific explanation
    - JSON export
"""

import argparse
import json
from typing import Any, Dict


from pathbioagent.graph import build_graph


# ============================================================
# Formatting helpers
# ============================================================

def print_header(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_query_interpretation(report: Dict[str, Any]):

    print_header(
        "QUERY INTERPRETATION"
    )

    print(
        f"Query: {report.get('query')}"
    )

    print(
        f"Ranking intent: "
        f"{report.get('ranking_intent')}"
    )

    print(
        f"Target scale: "
        f"{report.get('target_scale')}"
    )

    print(
        f"Entity: "
        f"{report.get('entity')}"
    )

    print(
        f"Query parser: "
        f"{report.get('parser_source')}"
    )

    print(
        f"Query confidence: "
        f"{report.get('query_confidence')}"
    )

    if report.get(
        "model_name"
    ):
        print(
            f"LLM model: "
            f"{report.get('model_name')}"
        )


# ============================================================
# Scientific result display
# ============================================================

def print_visibility_ranking(
    report: Dict[str, Any]
):

    print_header(
        "TARGET-SCALE VISIBILITY RANKING"
    )

    print(
        f"Scale: {report.get('target_scale')}"
    )

    print(
        f"Metric: "
        f"{report.get('visibility_ranking_metric')}"
    )

    print()

    ranking = report.get(
        "visibility_ranking",
        []
    )

    print(
        f"{'Rank':<6}"
        f"{'Program':<35}"
        f"{'Visibility':<12}"
        f"Status"
    )

    print("-" * 72)

    for item in ranking:

        print(
            f"{item.get('rank', ''):<6}"
            f"{item.get('program',''):<35}"
            f"{item.get('median_across_models',0):<12.4f}"
            f"{item.get('scale_hypothesis_status','')}"
        )


    print()

    print(
        "Interpretation:"
    )

    print(
        "Ranking is based on absolute "
        "target-scale molecular visibility."
    )

    print(
        "Scale preference and robustness "
        "are reported as secondary diagnostics."
    )


def print_hypothesis_result(
    report: Dict[str, Any]
):

    print_header(
        "SUPPORTED EXPLORATORY HYPOTHESES"
    )

    supported = report.get(
        "supported_exploratory_hypotheses",
        []
    )

    if supported:

        for item in supported:
            print(
                f"  - {item}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Not supported:"
    )

    for item in report.get(
        "not_supported",
        []
    ):

        print(
            f"  - {item}"
        )


def print_evidence_cards(
    report: Dict[str, Any]
):

    print_header(
        "EVIDENCE CARDS"
    )

    for idx, card in enumerate(
        report.get(
            "evidence_cards",
            []
        ),
        start=1,
    ):

        print(
            f"{idx}. "
            f"{card.get('program')}"
        )

        print(
            f"   status: "
            f"{card.get('status')}"
        )

        if (
            "target_scale_visibility"
            in card
        ):

            visibility = (
                card[
                    "target_scale_visibility"
                ]
            )

            print(
                "   target-scale visibility: "
                f"{visibility.get('median_across_models'):.4f}"
            )


        print(
            "   preferred scale: "
            f"{card.get('vote_preferred_scale')}"
        )

        print(
            "   model agreement: "
            f"{card.get('max_models_agree')}"
        )

        print(
            "   stability: "
            f"{card.get('median_optimum_stability_fraction')}"
        )

        print()


# ============================================================
# Explanation
# ============================================================

def print_explanation(
    report: Dict[str, Any]
):

    print_header(
        "SCIENTIFIC INTERPRETATION"
    )

    if (
        report.get(
            "primary_answer_type"
        )
        ==
        "target_scale_visibility_ranking"
    ):

        print(
            "This is a target-scale visibility query."
        )

        print(
            "Programs are ranked by "
            "absolute molecular visibility at "
            f"{report.get('target_scale')}."
        )

        print()

        for item in report.get(
            "visibility_ranking",
            []
        )[:3]:

            print(
                f"- {item['program']}: "
                f"{item['median_across_models']:.4f}"
            )

        print()

        print(
            "Scale preference should not be "
            "interpreted as the ranking criterion."
        )

    else:

        print(
            "This query evaluates "
            "scale-specific biological hypotheses."
        )

        print()

        for item in report.get(
            "supported_exploratory_hypotheses",
            []
        ):

            print(
                f"- {item}"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="Natural language query",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final JSON",
    )

    parser.add_argument(
        "--show-evidence",
        action="store_true",
    )

    parser.add_argument(
        "--show-trace",
        action="store_true",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
    )


    args = parser.parse_args()


    graph = build_graph()


    result = graph.invoke(
        {
            "query":
                args.query,

            "max_iterations":
                20,

            "trace":
                [],
        }
    )


    report = result.get(
        "final_report",
        result,
    )


    print_query_interpretation(
        report
    )


    print_header(
        "DETERMINISTIC EVIDENCE RESULT"
    )


    print(
        f"Candidates tested: "
        f"{report.get('candidates_tested')}/"
        f"{report.get('total_candidates')}"
    )


    if (
        report.get(
            "primary_answer_type"
        )
        ==
        "target_scale_visibility_ranking"
    ):

        print_visibility_ranking(
            report
        )

    else:

        print_hypothesis_result(
            report
        )


    if args.show_trace:

        print_header(
            "EXECUTION TRACE"
        )

        for i, item in enumerate(
            report.get(
                "trace",
                []
            ),
            start=1,
        ):

            print(
                f"{i}. {item}"
            )


    if args.show_evidence:

        print_evidence_cards(
            report
        )


    if args.explain:

        print_explanation(
            report
        )


    if args.json:

        print_header(
            "FINAL REPORT JSON"
        )

        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":

    main()
