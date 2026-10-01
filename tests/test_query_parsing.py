from pathbioagent.tools.program_evidence import (
    infer_target_scale,
    infer_ranking_intent,
)


def test_infer_target_scale_10x():
    query = "Which programs prefer 10x?"
    assert infer_target_scale(query) == "10x"


def test_infer_target_scale_40x():
    query = "Which programs prefer 40x?"
    assert infer_target_scale(query) == "40x"


def test_infer_target_scale_2p5x():
    query = "Which programs prefer 2.5x?"
    assert infer_target_scale(query) == "2p5x"


def test_infer_target_scale_none():
    query = "Which programs are most robust?"
    assert infer_target_scale(query) is None


def test_infer_preference_intent():
    query = "Which programs prefer 10x?"
    assert infer_ranking_intent(query) == "preference"


def test_infer_visibility_intent():
    query = "Which programs are strongest at 10x?"
    assert infer_ranking_intent(query) == "visibility"


def test_infer_robustness_intent():
    query = "Which programs have the most robust 10x preference?"
    assert infer_ranking_intent(query) == "robustness"


def test_robustness_has_priority_over_preference():
    query = "Which programs have the most robust 10x preference?"
    assert infer_ranking_intent(query) != "preference"
    assert infer_ranking_intent(query) == "robustness"


def test_general_intent():
    query = "Which molecular programs should be examined?"
    assert infer_ranking_intent(query) == "general"