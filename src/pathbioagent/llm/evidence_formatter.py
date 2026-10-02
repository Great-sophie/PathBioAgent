"""
Evidence formatter for PathBioAgent.
"""


from typing import Dict, Any, List



def build_evidence_cards(
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:

    cards = []

    for item in report.get(
        "evidence_cards",
        [],
    ):

        cards.append(
            {
                "program":
                    item.get("program"),

                "status":
                    item.get("status"),

                "preferred_scale":
                    item.get(
                        "vote_preferred_scale"
                    ),

                "model_agreement":
                    item.get(
                        "max_models_agree"
                    ),

                "median_optimum_scale":
                    item.get(
                        "cross_model_median_optimum_scale"
                    ),

                "optimum_stability":
                    item.get(
                        "median_optimum_stability_fraction"
                    ),

                "lomo_compatibility":
                    item.get(
                        "lomo_compatibility_fraction"
                    ),
            }
        )

    return cards



def prepare_report_for_llm(
    report: Dict[str, Any],
) -> Dict[str, Any]:

    return {

        "query":
            report.get("query"),

        "ranking_intent":
            report.get(
                "ranking_intent"
            ),

        "target_scale":
            report.get(
                "target_scale"
            ),

        "supported_programs":
            report.get(
                "supported_exploratory_hypotheses",
                [],
            ),

        "evidence_cards":
            build_evidence_cards(
                report
            ),
    }