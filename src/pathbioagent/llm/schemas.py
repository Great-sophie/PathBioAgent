from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Allowed structured query values
# ============================================================
QueryIntent = Literal[
    "preference",
    "visibility",
    "robustness",
    "discovery",
    "gene_explanation",
    "general",
]


TargetScale = Literal[
    "2p5x",
    "10x",
    "40x",
]

QueryEntity = Literal[
    "molecular_program",
]

ParserSource = Literal[
    "llm",
    "deterministic_fallback",
]


# ============================================================
# LLM output schema
# ============================================================

class LLMStructuredQuery(BaseModel):
    """
    Structured scientific query produced by the LLM.

    The LLM is only allowed to interpret language.
    It does NOT make scientific evidence decisions.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    intent: QueryIntent = Field(
        description=(
            "Scientific query intent: "
            "preference, visibility, robustness, "
            "discovery, or general."
        )
    )

    target_scale: Optional[
        TargetScale
    ] = Field(
        default=None,
        description=(
            "Requested histological scale. "
            "Use 2p5x, 10x, or 40x. "
            "Use null if no scale is specified."
        ),
    )

    entity: QueryEntity = Field(
        default="molecular_program",
        description=(
            "Entity being queried. "
            "Current v0.2 supports molecular_program."
        ),
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the language interpretation."
        ),
    )


# ============================================================
# Final interpreter result
# ============================================================

class QueryInterpretation(BaseModel):
    """
    Auditable output returned by the query interpreter.

    parser_source tells downstream components whether
    the interpretation came from the LLM or deterministic
    fallback logic.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    intent: QueryIntent

    target_scale: Optional[
        TargetScale
    ] = None

    entity: QueryEntity = (
        "molecular_program"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    parser_source: ParserSource

    model_name: Optional[str] = None

    explicit_scale_guard_applied: bool = False
