import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from pathbioagent.llm.schemas import (
    LLMStructuredQuery,
    QueryInterpretation,
)
from pathbioagent.tools.program_evidence import (
    infer_ranking_intent,
    infer_target_scale,
)


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are the query interpretation layer of PathBioAgent.

Your only task is to convert a user's scientific question into
a structured query.

You MUST NOT:
- judge whether a biological hypothesis is supported;
- change statistical evidence;
- infer biomarker validity;
- infer causality;
- perform the evidence gate.

Supported intent definitions:

1. preference
   The user asks which molecular programs prefer a particular
   histological scale.

   Example:
   "Which programs prefer 10x?"

2. visibility
   The user asks which molecular programs have the strongest
   morphology-derived molecular visibility at a scale.

   Example:
   "Which programs are strongest at 10x?"

3. robustness
   The user asks whether a scale preference is stable, robust,
   reproducible, or resistant to model exclusion.

   Example:
   "Which programs have the most robust 10x preference?"

4. general
   The question does not clearly match the above categories.

Scale normalization:
- 2.5x -> 2p5x
- 10x -> 10x
- 40x -> 40x
- no explicit scale -> null

Current supported entity:
- molecular_program

Return only the structured fields required by the schema.
""".strip()


# ============================================================
# Deterministic fallback
# ============================================================

def deterministic_interpret_query(
    query: str,
) -> QueryInterpretation:
    """
    Existing deterministic parser.

    This remains available as a safe fallback when an LLM
    cannot be used.
    """

    return QueryInterpretation(
        intent=infer_ranking_intent(
            query
        ),
        target_scale=infer_target_scale(
            query
        ),
        entity="molecular_program",
        confidence=1.0,
        parser_source=(
            "deterministic_fallback"
        ),
        model_name=None,
        explicit_scale_guard_applied=False,
    )


# ============================================================
# LLM construction
# ============================================================

def _build_structured_llm(
    model_name: str,
):
    """
    Build the language model with structured output.

    Kept as a separate function so tests can monkeypatch it
    without making external API calls.
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    return llm.with_structured_output(
        LLMStructuredQuery
    )


# ============================================================
# LLM interpreter
# ============================================================

def interpret_query(
    query: str,
    *,
    allow_fallback: bool = True,
    model_name: Optional[str] = None,
) -> QueryInterpretation:
    """
    Interpret a natural-language scientific query.

    Architecture:

        query
          ↓
        LLM language interpretation
          ↓
        Pydantic structured output
          ↓
        deterministic explicit-scale guard
          ↓
        QueryInterpretation

    The LLM cannot change evidence or scientific support status.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if model_name is None:
        model_name = os.getenv(
            "PATHBIO_LLM_MODEL",
            "gpt-5.6-luna",
        )

    # --------------------------------------------------------
    # No API key
    # --------------------------------------------------------

    if not api_key:

        if allow_fallback:
            return (
                deterministic_interpret_query(
                    query
                )
            )

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Explicit deterministic scale
    #
    # If the user literally says 10x, 40x, or 2.5x,
    # the LLM is not allowed to reinterpret that value.
    # --------------------------------------------------------

    explicit_scale = (
        infer_target_scale(
            query
        )
    )

    try:

        structured_llm = (
            _build_structured_llm(
                model_name
            )
        )

        parsed = structured_llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=query
                ),
            ]
        )

        target_scale = (
            parsed.target_scale
        )

        scale_guard_applied = False

        if (
            explicit_scale is not None
            and
            target_scale
            != explicit_scale
        ):
            target_scale = (
                explicit_scale
            )

            scale_guard_applied = True

        return QueryInterpretation(
            intent=parsed.intent,
            target_scale=target_scale,
            entity=parsed.entity,
            confidence=parsed.confidence,
            parser_source="llm",
            model_name=model_name,
            explicit_scale_guard_applied=(
                scale_guard_applied
            ),
        )

    except Exception:

        if allow_fallback:
            return (
                deterministic_interpret_query(
                    query
                )
            )

        raise