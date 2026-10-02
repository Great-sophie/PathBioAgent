#!/usr/bin/env python3

"""
Interactive PathBioAgent chat interface.

Usage:

    python examples/chat_agent.py

Features:
    - multi-turn query
    - evidence-grounded answers
    - follow-up questions
"""


import json

from pathbioagent.graph import build_graph



# ============================================================
# Display helpers
# ============================================================


def print_banner():

    print()
    print("=" * 72)
    print(
        "PathBioAgent Interactive Scientific Assistant"
    )
    print("=" * 72)

    print(
        """
Ask questions about morphology-molecular relationships.

Examples:

  Which programs show the highest molecular visibility at 10x?

  Which programs prefer 40x?

  Why is ECM / stroma not supported?

Type 'exit' to quit.
"""
    )




def _normalize_entity_name(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = (
        value
        .replace(" / ", "/")
        .replace("/ ", "/")
        .replace(" /", "/")
    )

    return " ".join(
        value.split()
    )


def _find_entity_card(report):
    """
    Find the molecular-program evidence card explicitly named
    in the user's query.

    ``report["entity"]`` stores the entity TYPE
    ("molecular_program"), not the concrete program name.

    Concrete program resolution is therefore performed
    deterministically against the evidence-card program names.
    """

    query = _normalize_entity_name(
        report.get(
            "query",
            ""
        )
    )

    if not query:
        return None


    cards = report.get(
        "evidence_cards",
        []
    )


    # --------------------------------------------------------
    # Exact program-name occurrence in the user query
    # --------------------------------------------------------

    for card in cards:

        program = card.get(
            "program"
        )

        normalized_program = (
            _normalize_entity_name(
                program
            )
        )

        if (
            normalized_program
            and
            normalized_program in query
        ):

            return card


    return None

def summarize_report(report):

    print()

    print("-" * 72)

    print(
        "Answer"
    )

    print("-" * 72)


    answer_type = report.get(
        "primary_answer_type"
    )


    # --------------------------------------------------------
    # Entity-specific explanation
    # --------------------------------------------------------

    entity_card = _find_entity_card(
        report
    )

    if (
        entity_card is not None
        and answer_type
        !=
        "target_scale_visibility_ranking"
    ):

        program = entity_card.get(
            "program"
        )

        status = entity_card.get(
            "status",
            "unknown",
        )

        print(
            program
        )

        print(
            "status:",
            status,
        )


        preferred_scale = entity_card.get(
            "vote_preferred_scale"
        )

        if preferred_scale:

            print(
                "preferred scale:",
                preferred_scale,
            )


        reasons = entity_card.get(
            "reasons",
            [],
        ) or []


        if reasons:

            print()

            print(
                "Why:"
            )

            for reason in reasons:

                print(
                    f"- {reason}"
                )


        return


    # --------------------------------------------------------
    # Visibility ranking
    # --------------------------------------------------------

    if (
        answer_type
        ==
        "target_scale_visibility_ranking"
    ):

        print(
            "Target-scale visibility ranking:"
        )

        print()

        ranking = report.get(
            "visibility_ranking",
            []
        )


        for item in ranking[:5]:

            print(
                f"{item.get('rank')}. "
                f"{item.get('program')} "
                f""
                f"({item.get('median_across_models'):.4f})"
            )


        print()

        print(
            "Note:"
        )

        print(
            "Ranking reflects target-scale visibility. "
            "Preferred-scale evidence and robustness "
            "are interpreted separately."
        )


    # --------------------------------------------------------
    # Preference / hypothesis
    # --------------------------------------------------------

    else:

        print(
            "Supported exploratory hypotheses:"
        )

        for item in report.get(
            "supported_exploratory_hypotheses",
            []
        ):

            print(
                f"- {item}"
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
                f"- {item}"
            )



def show_evidence(report):

    print()

    print(
        "Evidence summary:"
    )


    entity_card = _find_entity_card(
        report
    )

    if entity_card is not None:

        cards = [
            entity_card
        ]

    else:

        cards = report.get(
            "evidence_cards",
            []
        )


    for card in cards:

        print()

        print(
            card.get(
                "program"
            )
        )

        print(
            "status:",
            card.get(
                "status"
            )
        )


        if card.get(
            "vote_preferred_scale"
        ):

            print(
                "preferred scale:",
                card.get(
                    "vote_preferred_scale"
                )
            )


        if card.get(
            "target_scale_visibility"
        ):

            print(
                "target visibility:",
                card[
                    "target_scale_visibility"
                ].get(
                    "median_across_models"
                )
            )



# ============================================================
# Agent execution
# ============================================================


def run_query(
    graph,
    query,
):

    result = graph.invoke(
        {
            "query":
                query,

            "max_iterations":
                20,

            "trace":
                [],
        }
    )


    return result.get(
        "final_report",
        result,
    )



# ============================================================
# Main loop
# ============================================================


def main():


    graph = build_graph()


    print_banner()


    while True:


        try:

            query = input(
                "\nYou: "
            )


        except KeyboardInterrupt:

            print()

            break


        if not query.strip():

            continue


        if query.lower() in [
            "exit",
            "quit",
            "q",
        ]:

            print(
                "Bye."
            )

            break



        try:

            report = run_query(
                graph,
                query,
            )


            summarize_report(
                report
            )


            show_evidence(
                report
            )


        except Exception as e:

            print()

            print(
                "Agent error:"
            )

            print(
                e
            )



if __name__ == "__main__":

    main()
