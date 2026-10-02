"""
PathBioAgent LLM query interpreter.

Natural language query
        ↓
structured intent

Responsible for:
    - intent detection
    - target scale extraction
    - entity extraction

Not responsible for:
    - ranking
    - evidence evaluation
    - biological interpretation
"""


from __future__ import annotations


import json
import os
import re


from dotenv import load_dotenv


from langchain_openai import ChatOpenAI


from pathbioagent.llm.schemas import (
    LLMStructuredQuery,
    QueryInterpretation,
)


from pathbioagent.tools.program_evidence import (
    infer_target_scale,
    infer_ranking_intent,
)



# ============================================================
# Constants
# ============================================================


DEFAULT_OPENAI_MODEL = (
    "gpt-5.6-luna"
)


DEFAULT_DEEPSEEK_MODEL = (
    "deepseek-flash"
)


DEFAULT_DEEPSEEK_URL = (
    "https://api.deepseek.com"
)



SYSTEM_PROMPT = """

You are a query interpretation component.

Convert the user question into structured fields.

Return:

{
 "intent":
    "preference" |
"visibility" |
"robustness" |
"discovery" |
"general"O


 "target_scale":
    "2p5x" |
    "10x" |
    "40x" |
    null,

 "entity":
    "molecular_program",

 "confidence":
    float
}

Do not answer the question.
Do not rank candidates.

"""



# ============================================================
# Environment
# ============================================================


def _load_environment():

    load_dotenv()



def _provider():

    return os.getenv(
        "PATHBIO_LLM_PROVIDER",
        "openai",
    ).lower()



def _model_name():

    configured = os.getenv(
        "PATHBIO_LLM_MODEL"
    )

    if configured:

        return configured


    if _provider() == "deepseek":

        return DEFAULT_DEEPSEEK_MODEL


    return DEFAULT_OPENAI_MODEL



def _api_key():

    if _provider() == "deepseek":

        return os.getenv(
            "DEEPSEEK_API_KEY"
        )


    return os.getenv(
        "OPENAI_API_KEY"
    )



def _base_url():

    if _provider() == "deepseek":

        return os.getenv(
            "PATHBIO_LLM_BASE_URL",
            DEFAULT_DEEPSEEK_URL,
        )


    return None



# ============================================================
# deterministic fallback
# ============================================================


def _deterministic_fallback(
    query: str,
):

    return QueryInterpretation(

        intent=
            (
                "general"
                if _is_support_explanation_query(
                    query
                )
                else infer_ranking_intent(
                    query
                )
            ),

        target_scale=
            infer_target_scale(
                query
            ),

        entity=
            "molecular_program",

        confidence=
            1.0,

        parser_source=
            "deterministic_fallback",

        model_name=None,

        explicit_scale_guard_applied=False,

    )



def deterministic_interpret_query(
    query: str,
):

    if not query.strip():

        raise ValueError(
            "query cannot be empty"
        )


    # --------------------------------------------------------
    # Biological-pattern evidence layer
    #
    # This must be resolved before legacy ranking intents
    # such as visibility, robustness, and preference.
    # --------------------------------------------------------

    if _is_biological_pattern_query(
        query
    ):

        return QueryInterpretation(

            intent=
                "biological_pattern",

            target_scale=
                infer_target_scale(
                    query
                ),

            entity=
                "molecular_program",

            confidence=
                1.0,

            parser_source=
                "deterministic_fallback",

            model_name=
                None,

            explicit_scale_guard_applied=
                False,

        )


    # Preserve all existing routing behavior.
    return _deterministic_fallback(
        query
    )



# ============================================================
# Explicit scale extraction
# ============================================================


def _extract_explicit_scale(
    query: str,
):

    q = query.lower()


    if re.search(
        r"2\.5\s*[x×]",
        q
    ):

        return "2p5x"



    if re.search(
        r"10\s*[x×]",
        q
    ):

        return "10x"



    if re.search(
        r"40\s*[x×]",
        q
    ):

        return "40x"



    return None



# ============================================================


def _is_gene_explanation_query(
    query: str,
) -> bool:
    """
    Detect single-gene molecular visibility
    explanation questions.

    Examples:
        Why is SFTPA1 highly visible?
        Explain ROS1 visibility.
    """

    q = query.lower()


    explanation_terms = (
        "why is",
        "why are",
        "why does",
        "explain",
        "reason for",
    )


    visibility_terms = (
        "visible",
        "visibility",
        "molecular visibility",
    )


    return (
        any(
            term in q
            for term in explanation_terms
        )
        and
        any(
            term in q
            for term in visibility_terms
        )
    )


# Explanation-intent guard
# ============================================================


def _is_support_explanation_query(
    query: str,
) -> bool:
    """
    Detect questions asking why a scientific hypothesis is
    supported or not supported.

    These are explanation requests, not visibility,
    preference, or robustness ranking requests.
    """

    q = query.lower().strip()

    explanation_cue = (
        "why" in q
        or "explain" in q
        or "reason" in q
    )

    support_cue = re.search(
        r"\b("
        r"support|supported|unsupported"
        r")\b",
        q,
    )

    return bool(
        explanation_cue
        and
        support_cue
    )





def _is_discovery_query(
    query: str,
) -> bool:
    """
    Detect discovery queries asking for
    factors explaining morphology-derived
    molecular visibility.
    """

    q = query.lower()

    terms = (
        "what determines",
        "what explains",
        "which factors",
        "gene properties",
        "molecular visibility factors",
        "patterns across genes",
        "predict molecular visibility",
        "morphology-derived molecular visibility",
    )

    return any(
        term in q
        for term in terms
    )



# ============================================================
# Biological-pattern intent guard
# ============================================================


def _is_biological_pattern_query(
    query: str,
) -> bool:
    """
    Detect questions about biological characteristics,
    pathway enrichment, or molecular interpretation of
    cross-scale visibility-pattern clusters.

    This intent is distinct from:
        - discovery
        - gene_explanation
        - preference
        - visibility
        - robustness

    The detector identifies the evidence layer only.
    It does not infer biological conclusions.
    """

    q = (
        query
        .lower()
        .replace("×", "x")
    )

    q = re.sub(
        r"\s+",
        " ",
        q,
    ).strip()


    # --------------------------------------------------------
    # Explicit cluster language
    # --------------------------------------------------------

    cluster_reference = bool(
        re.search(
            r"\b(?:fine\s+)?clusters?\b",
            q,
        )
    )


    # --------------------------------------------------------
    # Explicit visibility-pattern language
    # --------------------------------------------------------

    visibility_pattern_reference = (
        "visibility pattern"
        in q
    )


    # --------------------------------------------------------
    # Known descriptive visibility phenotypes
    # --------------------------------------------------------

    pattern_descriptor = any(
        term in q
        for term in (
            "scale-stable",
            "scale stable",
            "scale-flat",
            "scale flat",
            "high-visibility",
            "high visibility",
            "low-visibility",
            "low visibility",
            "high-magnification",
            "high magnification",
        )
    )


    if re.search(
        r"\b(?:2(?:\.|p)5|10|40)\s*x"
        r"\s*[- ]?(?:enhanced|peaked)\b",
        q,
    ):

        pattern_descriptor = True


    # --------------------------------------------------------
    # Biological / molecular interpretation language
    # --------------------------------------------------------

    biological_language = any(
        term in q
        for term in (
            "biology",
            "biological",
            "biologically",
            "pathway",
            "pathways",
            "molecular program",
            "molecular programs",
            "biological program",
            "biological programs",
            "biological process",
            "biological processes",
            "molecular characteristic",
            "molecular characteristics",
            "biological evidence",
            "biological interpretation",
        )
    )


    # --------------------------------------------------------
    # Relationship / enrichment language
    # --------------------------------------------------------

    relation_language = any(
        term in q
        for term in (
            "associated with",
            "association with",
            "enriched",
            "enrichment",
            "characterize",
            "characterizes",
            "characterise",
            "characterises",
            "define",
            "defines",
            "distinguish",
            "distinguishes",
            "support",
        )
    )


    comparison_language = any(
        term in q
        for term in (
            "compare",
            "comparison",
            " versus ",
            " vs ",
        )
    )


    preference_language = any(
        term in q
        for term in (
            "prefer",
            "prefers",
            "preferred",
            "preferring",
            "preference",
        )
    )


    explicit_scale = (
        infer_target_scale(
            query
        )
        is not None
    )


    # --------------------------------------------------------
    # Rule 1
    #
    # Explicit visibility-pattern questions.
    # --------------------------------------------------------

    if visibility_pattern_reference:

        return True


    # --------------------------------------------------------
    # Rule 2
    #
    # Cluster-level biological / enrichment / comparison
    # questions.
    # --------------------------------------------------------

    if cluster_reference:

        if (
            biological_language
            or relation_language
            or comparison_language
        ):

            return True


        # Example:
        # Which clusters prefer 40x?
        if (
            explicit_scale
            and preference_language
        ):

            return True


    # --------------------------------------------------------
    # Rule 3
    #
    # Named visibility phenotype + biological question.
    # --------------------------------------------------------

    if (
        pattern_descriptor
        and (
            biological_language
            or relation_language
            or comparison_language
        )
    ):

        return True


    # --------------------------------------------------------
    # Rule 4
    #
    # Biology of genes preferring a specified scale.
    # --------------------------------------------------------

    if (
        biological_language
        and explicit_scale
        and preference_language
    ):

        return True


    return False


# ============================================================
# LLM builder
# ============================================================


def _build_structured_llm(
    model_name
):

    key = _api_key()


    if not key:

        raise RuntimeError(
            "Missing API key"
        )


    kwargs = {

        "model":
            model_name,

        "api_key":
            key,

        "temperature":
            0,

    }


    if _base_url():

        kwargs[
            "base_url"
        ] = _base_url()



    llm = ChatOpenAI(
        **kwargs
    )


    if _provider() == "deepseek":

        return llm.bind(
            response_format={
                "type":
                    "json_object"
            }
        )


    return llm.with_structured_output(
        LLMStructuredQuery
    )



# ============================================================
# Parser
# ============================================================


def _parse_deepseek_response(
    response
):

    content = response.content


    if isinstance(
        content,
        str,
    ):

        content = json.loads(
            content
        )


    return LLMStructuredQuery(
        **content
    )



def _parse_openai_response(
    response
):
    """
    Parse OpenAI structured response.

    Supports:
        - LLMStructuredQuery
        - pydantic object
        - dict
        - fake response used in tests
    """


    if isinstance(
        response,
        LLMStructuredQuery,
    ):

        return response



    if hasattr(
        response,
        "model_dump",
    ):

        return LLMStructuredQuery(
            **response.model_dump()
        )



    if isinstance(
        response,
        dict,
    ):

        return LLMStructuredQuery(
            **response
        )



    if hasattr(
        response,
        "content",
    ):

        content = response.content


        if isinstance(
            content,
            str,
        ):

            content = json.loads(
                content
            )


        return LLMStructuredQuery(
            **content
        )


    raise TypeError(
        f"Unsupported response type: {type(response)}"
    )



# ============================================================
# Main
# ============================================================


def interpret_query(
    query: str,
    *,
    allow_fallback=True,
):


    if not query.strip():

        raise ValueError(
            "query cannot be empty"
        )


    # --------------------------------------------------------
    # Deterministic biological-pattern routing
    # --------------------------------------------------------
    # Biological-pattern queries map to a deterministic
    # evidence layer. Do not allow the LLM to reinterpret
    # them as discovery, visibility, robustness, preference,
    # or general.
    # --------------------------------------------------------

    if _is_biological_pattern_query(
        query
    ):

        return deterministic_interpret_query(
            query
        )


    _load_environment()



    explicit_scale = (
        _extract_explicit_scale(
            query
        )
    )


    try:


        llm = _build_structured_llm(
            _model_name()
        )


        messages = [

            {
                "role":
                    "system",

                "content":
                    (
                        SYSTEM_PROMPT
                        + "\n\n"
                        + (
                            "Return the result as a valid JSON object. "
                            "Do not use Markdown. "
                            "Do not include any text outside the JSON object."
                        )
                    ),
            },


            {
                "role":
                    "user",

                "content":
                    query,
            },

        ]


        response = llm.invoke(
            messages
        )


        # --------------------------------------------------------
        # Structured output compatibility
        # --------------------------------------------------------
        # LangChain structured output and unit tests may directly
        # return LLMStructuredQuery objects.
        # These should bypass provider-specific parsers.
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Structured output compatibility
        # --------------------------------------------------------
        # Accept structured query objects directly.
        # This supports LangChain structured output,
        # DeepSeek structured models, and unit-test mocks.
        # --------------------------------------------------------

        if (
            hasattr(response, "intent")
            and
            hasattr(response, "target_scale")
            and
            hasattr(response, "entity")
        ):

            parsed = response


        else:

            provider = _provider()


            if provider == "deepseek":

                parsed = (
                    _parse_deepseek_response(
                        response
                    )
                )

            else:

                parsed = (
                    _parse_openai_response(
                        response
                    )
                )


        final_scale = (

            explicit_scale
            if explicit_scale
            else
            parsed.target_scale

        )



        guard = (

            explicit_scale is not None
            and
            explicit_scale != parsed.target_scale

        )



        if _is_discovery_query(
            query
        ):
            final_intent = "discovery"

        elif _is_gene_explanation_query(
            query
        ):
            final_intent = "gene_explanation"


        elif _is_support_explanation_query(
            query
        ):

            final_intent = "general"

        else:

            final_intent = parsed.intent


        return QueryInterpretation(

            intent=
                final_intent,


            target_scale=
                final_scale,


            entity=
                parsed.entity,


            confidence=
                parsed.confidence,


            parser_source=
                "llm",


            model_name=
                _model_name(),


            explicit_scale_guard_applied=
                guard,

        )



    except Exception as e:


        if not allow_fallback:

            raise



        return deterministic_interpret_query(
            query
        )
