from typing import Dict, Any

from pathbioagent.tools.cluster_biological_evidence import (
    build_cluster_biological_evidence_response,
)


def compare_cluster_biology(
    cluster_a: int,
    cluster_b: int,
) -> Dict[str, Any]:
    """
    Compare biological evidence between two visibility clusters.
    """

    a = build_cluster_biological_evidence_response(
        cluster_a
    )

    b = build_cluster_biological_evidence_response(
        cluster_b
    )


    differences = []


    if (
        a.get("visibility_pattern")
        !=
        b.get("visibility_pattern")
    ):

        differences.append(
            "Different visibility patterns"
        )


    if (
        a.get("dominant_preferred_scale")
        !=
        b.get("dominant_preferred_scale")
    ):

        differences.append(
            "Different preferred magnification"
        )


    if (
        a.get("biological_programs")
        !=
        b.get("biological_programs")
    ):

        differences.append(
            "Different biological pathway enrichment"
        )


    return {

        "evidence_type":
            "cluster_biological_comparison",


        "cluster_a":
            a,


        "cluster_b":
            b,


        "key_differences":
            differences,


        "confidence":

            {
                "level":
                    "moderate",

                "basis":
                    [
                        "Cluster-level visibility evidence",
                        "RNA characterization",
                        "Pathway enrichment",
                    ],

                "limitations":
                    [
                        "Exploratory comparison",
                        "No causal inference",
                    ],
            },
    }
