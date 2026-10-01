"""
PathBioAgent LLM query interpreter.

Supports:
    - OpenAI structured output
    - DeepSeek JSON output
    - deterministic fallback

LLM responsibility:
    Natural language -> structured intent

NOT responsible:
    ranking
    evidence evaluation
    biological conclusions
"""

from __future__ import annotations


import json
import os
import re

from typing import Any, Optional


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
# Defaults
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



# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """

You are the query interpretation component of PathBioAgent.

Only convert user questions into structured fields.

Do NOT answer questions.
Do NOT rank programs.
Do NOT interpret biological meaning.

Return JSON:

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


Examples:

Question:
Which programs prefer 10x?

Output:
{
"intent":"preference",
"target_scale":"10x",
"entity":"molecular_program"
}


Question:
Which programs have the most robust 10x preference?

Output:
{
"intent":"robustness",
"target_scale":"10x",
"entity":"molecular_program"
}

"""



# ============================================================
# Environment
# ============================================================


def _load_environment():

    # keep zero argument for tests
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
# deterministic parser
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
    query:str,
):

    if not query.strip():

        raise ValueError(
            "query cannot be empty"
        )

    return _deterministic_fallback(
        query
    )



# ============================================================
# scale guard
# ============================================================


def _extract_explicit_scale(
    query:str
):

    q=query.lower()


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
# LLM builders
# ============================================================


def _build_structured_llm(
    model_name
):

    key=_api_key()

    if not key:

        raise RuntimeError(
            "Missing API key"
        )


    kwargs={

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
        ]=_base_url()



    llm=ChatOpenAI(
        **kwargs
    )


    #
    # DeepSeek does not support
    # OpenAI json_schema response format
    #

    if _provider()=="deepseek":

        return llm.bind(
            response_format={
                "type":
                    "json_object"
            }
        )


    return llm.with_structured_output(
        LLMStructuredQuery
    )




def _build_llm(
    model_name=None
):

    return _build_structured_llm(
        model_name or _model_name()
    )



# ============================================================
# Parse LLM response
# ============================================================


def _parse_deepseek_response(
    response
):

    if hasattr(
        response,
        "content"
    ):

        content=response.content

    else:

        content=response


    if isinstance(
        content,
        str
    ):

        data=json.loads(
            content
        )

    else:

        data=content



    return LLMStructuredQuery(
        **data
    )



def _parse_openai_response(
    response
):

    if isinstance(
        response,
        LLMStructuredQuery
    ):

        return response


    if hasattr(
        response,
        "model_dump"
    ):

        return LLMStructuredQuery(
            **response.model_dump()
        )


    return LLMStructuredQuery(
        **response
    )



# ============================================================
# Main
# ============================================================


def interpret_query(
    query:str,
    *,
    allow_fallback=True,
):


    if not query.strip():

        raise ValueError(
            "query cannot be empty"
        )


    _load_environment()



    try:

        llm=_build_structured_llm(
            _model_name()
        )


        messages=[

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
            }

        ]


        response=llm.invoke(
            messages
        )



        if _provider()=="deepseek":

            parsed=(
                _parse_deepseek_response(
                    response
                )
            )


        else:

            parsed=(
                _parse_openai_response(
                    response
                )
            )



        explicit_scale=(
            _extract_explicit_scale(
                query
            )
        )


        final_scale=(

            explicit_scale
            if explicit_scale
            else
            parsed.target_scale
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
                (
                    explicit_scale is not None
                    and
                    explicit_scale !=
                    parsed.target_scale
                ),
        )



    except Exception:


        if not allow_fallback:

            raise


        return deterministic_interpret_query(
            query
        )