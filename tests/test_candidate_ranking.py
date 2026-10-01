import pandas as pd

import pathbioagent.tools.program_evidence as pe


def make_fake_tables():
    """
    Minimal synthetic evidence tables for testing
    query-aware candidate ranking.

    Program_A:
        strong and robust 10x preference

    Program_B:
        10x preference, higher absolute 10x visibility
        than Program_A

    Program_C:
        consensus preference = 40x,
        but highest absolute visibility at 10x

    Program_D:
        tied consensus and missing robustness
    """

    consensus = pd.DataFrame(
        [
            {
                "program": "Program_A",
                "n_models_prefer_40x": 0,
                "n_models_prefer_10x": 6,
                "n_models_prefer_2p5x": 2,
                "max_models_agree": 6,
                "consensus_preferred_scale": "10x",
                "coverage_fraction": 1.0,
                "eligible_main": True,
            },
            {
                "program": "Program_B",
                "n_models_prefer_40x": 2,
                "n_models_prefer_10x": 5,
                "n_models_prefer_2p5x": 1,
                "max_models_agree": 5,
                "consensus_preferred_scale": "10x",
                "coverage_fraction": 1.0,
                "eligible_main": True,
            },
            {
                "program": "Program_C",
                "n_models_prefer_40x": 6,
                "n_models_prefer_10x": 2,
                "n_models_prefer_2p5x": 0,
                "max_models_agree": 6,
                "consensus_preferred_scale": "40x",
                "coverage_fraction": 1.0,
                "eligible_main": True,
            },
            {
                "program": "Program_D",
                "n_models_prefer_40x": 4,
                "n_models_prefer_10x": 4,
                "n_models_prefer_2p5x": 0,
                "max_models_agree": 4,
                "consensus_preferred_scale": "tie",
                "coverage_fraction": 1.0,
                "eligible_main": True,
            },
        ]
    )

    cross_model = pd.DataFrame(
        [
            {
                "program": "Program_A",
                "scale": "10x",
                "median_across_models": 0.20,
            },
            {
                "program": "Program_B",
                "scale": "10x",
                "median_across_models": 0.30,
            },
            {
                "program": "Program_C",
                "scale": "10x",
                "median_across_models": 0.50,
            },
            {
                "program": "Program_D",
                "scale": "10x",
                "median_across_models": 0.25,
            },
            {
                "program": "Program_A",
                "scale": "40x",
                "median_across_models": 0.10,
            },
            {
                "program": "Program_B",
                "scale": "40x",
                "median_across_models": 0.20,
            },
            {
                "program": "Program_C",
                "scale": "40x",
                "median_across_models": 0.60,
            },
            {
                "program": "Program_D",
                "scale": "40x",
                "median_across_models": 0.30,
            },
        ]
    )

    robustness = pd.DataFrame(
        [
            {
                "program": "Program_A",
                "minimum_vote_margin": 3,
                "median_vote_margin": 3.0,
                "compatible_exclusions": 8,
                "n_exclusions": 8,
                "median_optimum_stability_fraction": 1.0,
            },
            {
                "program": "Program_B",
                "minimum_vote_margin": 2,
                "median_vote_margin": 2.0,
                "compatible_exclusions": 8,
                "n_exclusions": 8,
                "median_optimum_stability_fraction": 1.0,
            },
            {
                "program": "Program_C",
                "minimum_vote_margin": 4,
                "median_vote_margin": 4.0,
                "compatible_exclusions": 8,
                "n_exclusions": 8,
                "median_optimum_stability_fraction": 1.0,
            },
            # Program_D intentionally omitted
        ]
    )

    return {
        "consensus": consensus,
        "cross_model": cross_model,
        "robustness": robustness,
    }


def patch_tables(monkeypatch):
    monkeypatch.setattr(
        pe,
        "load_tables",
        make_fake_tables,
    )


def test_preference_ranking_prioritizes_votes(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs prefer 10x?"
    )

    assert candidates[0] == "Program_A"
    assert candidates[1] == "Program_B"

    # Program_C has the strongest absolute
    # 10x visibility, but only 2/8 models
    # prefer 10x.
    assert candidates.index(
        "Program_C"
    ) > candidates.index(
        "Program_B"
    )


def test_visibility_ranking_prioritizes_absolute_visibility(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs are strongest at 10x?"
    )

    # Program_C is 40x-preferred overall,
    # but has the highest absolute 10x visibility.
    assert candidates[0] == "Program_C"

    assert candidates[1] == "Program_B"


def test_robust_10x_query_filters_non_10x_programs(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs have the most robust "
        "10x preference?"
    )

    assert candidates == [
        "Program_A",
        "Program_B",
    ]


def test_robustness_ranking_uses_stability_and_margin(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs have the most robust "
        "10x preference?"
    )

    # Both pass the scale filter and have
    # perfect stability. Program_A has the
    # larger minimum vote margin.
    assert candidates[0] == "Program_A"
    assert candidates[1] == "Program_B"


def test_40x_preference_ranking(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs prefer 40x?"
    )

    assert candidates[0] == "Program_C"


def test_missing_robustness_does_not_enter_robust_10x_candidates(
    monkeypatch,
):
    patch_tables(monkeypatch)

    candidates = pe.search_program_candidates(
        "Which programs have the most robust "
        "10x preference?"
    )

    assert "Program_D" not in candidates