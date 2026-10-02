"""
Query adapter for biological pattern evidence.

This module converts natural language biological-pattern
queries into deterministic evidence retrieval requests.

It does NOT perform biological inference.
It only resolves query intent into evidence lookup.
"""


from typing import Dict, Any

import re


from pathbioagent.tools.cluster_biological_evidence import (
    build_cluster_biological_evidence_response,
    find_clusters_by_scale,
    find_clusters_by_visibility_pattern,
    find_clusters_by_program,
)



def resolve_biological_pattern_query(
    query: str,
) -> Dict[str, Any]:
    """
    Resolve biological pattern questions.

    Supported examples:

        What is the biological evidence for cluster 3?

        Which clusters prefer 40x?

        Which visibility pattern is associated
        with G2-M checkpoint?

    """

    if not query.strip():

        return {
            "evidence_type":
                "biological_pattern_empty_query"
        }


    q = query.lower()



    # ========================================================
    # Explicit cluster query
    # ========================================================

    cluster_match = re.search(
        r"cluster\s+(\d+)",
        q,
    )


    if cluster_match:

        cluster_id = int(
            cluster_match.group(1)
        )


        return (
            build_cluster_biological_evidence_response(
                cluster_id
            )
        )



    # ========================================================
    # Scale preference query
    # ========================================================

    if (
        "40x" in q
        or "40×" in q
    ):

        clusters = find_clusters_by_scale(
            "40x"
        )

        return {

            "evidence_type":
                "biological_pattern_scale",

            "requested_scale":
                "40x",

            "clusters":
                clusters,

        }



    if (
        "2.5x" in q
        or "2.5×" in q
        or "2p5x" in q
    ):

        clusters = find_clusters_by_scale(
            "2p5x"
        )

        return {

            "evidence_type":
                "biological_pattern_scale",

            "requested_scale":
                "2p5x",

            "clusters":
                clusters,

        }



    if "10x" in q:

        clusters = find_clusters_by_scale(
            "10x"
        )

        return {

            "evidence_type":
                "biological_pattern_scale",

            "requested_scale":
                "10x",

            "clusters":
                clusters,

        }



    # ========================================================
    # Visibility pattern query
    # ========================================================

    pattern_terms = [

        "scale-stable",
        "scale stable",
        "stable visibility",

        "40x-enhanced",
        "40x enhanced",

        "2.5x-enhanced",
        "2.5x enhanced",

        "low magnification",

        "high magnification",

    ]


    for term in pattern_terms:

        if term in q:

            return {

                "evidence_type":
                    "biological_visibility_pattern",

                "pattern":
                    term,

                "clusters":
                    find_clusters_by_visibility_pattern(
                        term
                    ),

            }



    # ========================================================
    # Molecular program query
    # ========================================================

    program_terms = [

        "cell cycle",
        "g2-m",
        "g2m",

        "e2f",

        "extracellular matrix",
        "interferon",

    ]


    for term in program_terms:

        if term in q:

            return {

                "evidence_type":
                    "biological_program_pattern",

                "program":
                    term,

                "clusters":
                    find_clusters_by_program(
                        term
                    ),

            }



    # ========================================================
    # Fallback
    # ========================================================

    return {

        "evidence_type":
            "biological_pattern_unresolved",

        "query":
            query,

        "message":
            "No deterministic biological pattern matched."

    }
