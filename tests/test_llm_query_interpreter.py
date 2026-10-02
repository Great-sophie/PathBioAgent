from pathbioagent.llm import LLMStructuredQuery
import pathbioagent.llm.query_interpreter as qi


# ============================================================
# Deterministic fallback
# ============================================================

def test_fallback_robust_10x(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )

    # Prevent .env from re-injecting an API key during tests
    monkeypatch.setattr(
        qi,
        "load_dotenv",
        lambda: None,
    )

    result = qi.interpret_query(
        "Which programs have the most robust 10x preference?"
    )

    assert result.intent == "robustness"
    assert result.target_scale == "10x"

    assert (
        result.parser_source
        == "deterministic_fallback"
    )


def test_fallback_visibility_40x(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        qi,
        "load_dotenv",
        lambda: None,
    )

    result = qi.interpret_query(
        "Which programs are strongest at 40x?"
    )

    assert result.intent == "visibility"
    assert result.target_scale == "40x"


def test_fallback_2p5x_normalization(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        qi,
        "load_dotenv",
        lambda: None,
    )

    result = qi.interpret_query(
        "Which programs prefer 2.5x?"
    )

    assert result.intent == "preference"
    assert result.target_scale == "2p5x"


# ============================================================
# Fake structured LLM
# ============================================================

class FakeStructuredLLM:
    def __init__(self, parsed_result):
        self.parsed_result = parsed_result

    def invoke(self, messages):
        return self.parsed_result


def test_llm_structured_output(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        qi,
        "load_dotenv",
        lambda: None,
    )

    fake_result = LLMStructuredQuery(
        intent="robustness",
        target_scale="10x",
        entity="molecular_program",
        confidence=0.95,
    )

    monkeypatch.setattr(
        qi,
        "_build_structured_llm",
        lambda model_name: FakeStructuredLLM(
            fake_result
        ),
    )

    result = qi.interpret_query(
        "Which programs remain stable around 10x?"
    )

    assert result.intent == "robustness"
    assert result.target_scale == "10x"
    assert result.parser_source == "llm"
    assert result.confidence == 0.95


# ============================================================
# Explicit scale guard
# ============================================================

def test_explicit_scale_cannot_be_changed_by_llm(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    monkeypatch.setattr(
        qi,
        "load_dotenv",
        lambda: None,
    )

    fake_result = LLMStructuredQuery(
        intent="preference",
        target_scale="40x",
        entity="molecular_program",
        confidence=0.90,
    )

    monkeypatch.setattr(
        qi,
        "_build_structured_llm",
        lambda model_name: FakeStructuredLLM(
            fake_result
        ),
    )

    result = qi.interpret_query(
        "Which programs prefer 10x?"
    )

    # User explicitly wrote 10x.
    # LLM is not allowed to reinterpret it as 40x.
    assert result.target_scale == "10x"

    assert (
        result.explicit_scale_guard_applied
        is True
    )


def test_empty_query_rejected():
    try:
        qi.interpret_query("   ")

    except ValueError:
        return

    raise AssertionError(
        "Empty query should raise ValueError."
    )
