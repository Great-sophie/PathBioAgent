
from __future__ import annotations

import pytest

from pathbioagent.llm.query_interpreter import (
    deterministic_interpret_query,
)


# ============================================================
# Helpers
# ============================================================

def _get_field(
    result,
    field: str,
):

    if isinstance(
        result,
        dict,
    ):
        return result.get(
            field
        )

    if hasattr(
        result,
        field,
    ):
        return getattr(
            result,
            field,
        )

    if hasattr(
        result,
        "model_dump",
    ):

        data = result.model_dump()

        return data.get(
            field
        )

    return None


def _normalize_intent(
    value,
) -> str | None:

    if value is None:
        return None

    if hasattr(
        value,
        "value",
    ):
        value = value.value

    return str(
        value
    ).strip()


def _ranking_intent(
    query: str,
) -> str | None:

    result = (
        deterministic_interpret_query(
            query
        )
    )

    value = _get_field(
        result,
        "intent",
    )

    return _normalize_intent(
        value
    )


# ============================================================
# Biological-pattern routing
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Which biological programs "
            "characterize 40x-enhanced genes?"
        ),
        (
            "What biological programs are "
            "associated with scale-stable visibility?"
        ),
        (
            "Which visibility pattern is "
            "associated with cell cycle?"
        ),
        (
            "Which cluster is associated "
            "with cell cycle programs?"
        ),
        (
            "What pathways characterize "
            "cluster 3?"
        ),
        (
            "Compare the biology of the "
            "scale-stable and 2.5x-enhanced "
            "clusters."
        ),        (
            "What molecular programs are "
            "enriched in the 40x-enhanced "
            "visibility pattern?"
        ),
        (
            "Which biological processes are "
            "associated with high-visibility "
            "scale-stable genes?"
        ),
        (
            "What distinguishes the biology "
            "of 40x-enhanced genes?"
        ),
    ],
)
def test_biological_pattern_queries_route_correctly(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "biological_pattern"
    )


# ============================================================
# Magnification notation variants
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Which biological programs "
            "characterize 40x-enhanced genes?"
        ),
        (
            "Which biological programs "
            "characterize 40×-enhanced genes?"
        ),
        (
            "Which biological programs "
            "characterize 2.5x-enhanced genes?"
        ),
        (
            "Which biological programs "
            "characterize 2.5×-enhanced genes?"
        ),
    ],
)
def test_magnification_notation_routes_to_biological_pattern(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "biological_pattern"
    )


# ============================================================
# Preferred scale vs enhanced visibility
#
# Both belong to biological-pattern reasoning, but later
# deterministic retrieval must distinguish:
#
# "40x-enhanced"
#     -> visibility-pattern retrieval
#
# "prefer 40x"
#     -> preferred-scale retrieval
#
# This file only fixes the high-level intent.
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Which clusters prefer 40x?"
        ),
        (
            "Which visibility pattern is "
            "40x-enhanced?"
        ),
        (
            "What biology characterizes "
            "genes preferring 40x?"
        ),
        (
            "What biology characterizes "
            "the 40x-enhanced pattern?"
        ),
    ],
)
def test_scale_related_biological_queries_route_correctly(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "biological_pattern"
    )


# ============================================================
# Program-centric queries
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Which visibility pattern is "
            "associated with G2-M checkpoint?"
        ),
        (
            "Which cluster is enriched for "
            "E2F targets?"
        ),
        (
            "Which morphology-derived visibility "
            "pattern is associated with "
            "extracellular matrix organization?"
        ),
        (
            "Is interferon alpha response "
            "associated with a particular "
            "visibility pattern?"
        ),
    ],
)
def test_program_to_pattern_queries_route_correctly(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "biological_pattern"
    )


# ============================================================
# Cluster-centric queries
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "What is the biological evidence "
            "for cluster 3?"
        ),
        (
            "Summarize the biology of "
            "fine cluster 2."
        ),
        (
            "What molecular characteristics "
            "define cluster 5?"
        ),
        (
            "Which pathways support the "
            "biological interpretation of "
            "cluster 3?"
        ),
    ],
)
def test_cluster_biology_queries_route_correctly(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "biological_pattern"
    )


# ============================================================
# Regression protection:
# existing discovery route must remain intact
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Which factors explain "
            "morphology-derived molecular visibility?"
        ),
        (
            "What RNA features are associated "
            "with molecular visibility?"
        ),
    ],
)
def test_existing_discovery_queries_still_route_to_discovery(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "discovery"
    )


# ============================================================
# Regression protection:
# gene explanation must keep higher specificity
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        (
            "Why is SFTPA1 highly visible?"
        ),
        (
            "Why does SFTPA1 show high "
            "morphology-derived molecular visibility?"
        ),
    ],
)
def test_existing_gene_explanation_queries_still_route_correctly(
    query,
):

    assert (
        _ranking_intent(
            query
        )
        ==
        "gene_explanation"
    )


# ============================================================
# Avoid accidental gene-query capture
# ============================================================

def test_gene_specific_query_is_not_biological_pattern():

    intent = _ranking_intent(
        "Why is SFTPA1 highly visible?"
    )

    assert (
        intent
        !=
        "biological_pattern"
    )


# ============================================================
# Avoid accidental generic-discovery capture
# ============================================================

def test_generic_visibility_discovery_is_not_biological_pattern():

    intent = _ranking_intent(
        "Which factors explain "
        "morphology-derived molecular visibility?"
    )

    assert (
        intent
        !=
        "biological_pattern"
    )
