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
    "general",

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
            infer_ranking_intent(
                query
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
                    SYSTEM_PROMPT,
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



        return QueryInterpretation(

            intent=
                parsed.intent,


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


        print(
            "\n========== QUERY INTERPRETER DEBUG =========="
        )


        print(
            "Exception type:",
            type(e)
        )


        print(
            "Exception:",
            repr(e)
        )


        print(
            "Provider:",
            _provider()
        )


        print(
            "Model:",
            _model_name()
        )


        print(
            "Query:",
            query
        )


        print(
            "==============================================\n"
        )


        if not allow_fallback:

            raise



        return deterministic_interpret_query(
            query
        )
