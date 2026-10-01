import pathbioagent.nodes as nodes


# ============================================================
# Helpers
# ============================================================

def make_state(candidate="Program_A"):
    return {
        "query": "Which programs prefer 10x?",
        "current_candidate": candidate,
        "evidence": {
            candidate: {
                "program": candidate,
            }
        },
        "trace": [],
    }


def make_clean_item():
    """
    Synthetic evidence item with clean support
    and no evidence-quality warnings.
    """
    return {
        "consensus": {
            "preferred_scale": "10x",
            "max_models_agree": 6,
        },
        "evidence_alignment": {
            "conflict": False,
        },
        "robustness": {
            "available": True,
            "median_optimum_stability_fraction": 1.0,
        },
        "lomo": {
            "available": True,
            "compatibility_fraction": 1.0,
        },
    }


# ============================================================
# 1. Status normalization
# ============================================================

def test_normalize_conflicting_status_to_supported_when_all_gates_pass():
    decision = {
        "status": "conflicting_evidence",
        "consensus_ok": True,
        "target_ok": True,
        "robust_ok": True,
    }

    status = nodes._normalize_support_status(
        decision
    )

    assert (
        status
        == "supported_exploratory_hypothesis"
    )


def test_normalize_conflicting_status_to_not_supported_when_gate_fails():
    decision = {
        "status": "conflicting_evidence",
        "consensus_ok": True,
        "target_ok": True,
        "robust_ok": False,
    }

    status = nodes._normalize_support_status(
        decision
    )

    assert status == "not_supported"


def test_normalize_preserves_regular_status():
    decision = {
        "status": "insufficient_robustness_evidence",
        "consensus_ok": True,
        "target_ok": True,
        "robust_ok": False,
    }

    status = nodes._normalize_support_status(
        decision
    )

    assert (
        status
        == "insufficient_robustness_evidence"
    )


# ============================================================
# 2. Evidence gate
# ============================================================

def test_evidence_gate_clean_support(
    monkeypatch,
):
    state = make_state()

    def fake_evaluate(
        evidence,
        target_scale=None,
    ):
        assert target_scale == "10x"

        return {
            "status":
                "supported_exploratory_hypothesis",
            "consensus_ok": True,
            "target_ok": True,
            "robust_ok": True,
            "evidence_conflict": False,
            "reasons": [
                "Synthetic clean support."
            ],
        }

    monkeypatch.setattr(
        nodes,
        "evaluate_program_evidence",
        fake_evaluate,
    )

    result = nodes.evidence_gate(
        state
    )

    decision = (
        result["evidence"]
        ["Program_A"]
        ["decision"]
    )

    assert (
        decision["status"]
        == "supported_exploratory_hypothesis"
    )

    assert (
        decision["raw_gate_status"]
        == "supported_exploratory_hypothesis"
    )

    assert decision["consensus_ok"] is True
    assert decision["target_ok"] is True
    assert decision["robust_ok"] is True

    assert (
        result["current_decision"]
        == "supported_exploratory_hypothesis"
    )


def test_evidence_gate_keeps_conflict_as_diagnostic(
    monkeypatch,
):
    """
    Scientific design principle:

    primary support status and evidence conflict
    are separate dimensions.
    """

    state = make_state(
        "Program_Conflict"
    )

    def fake_evaluate(
        evidence,
        target_scale=None,
    ):
        return {
            "status":
                "conflicting_evidence",
            "consensus_ok": True,
            "target_ok": True,
            "robust_ok": False,
            "evidence_conflict": True,
            "reasons": [
                "Aggregation methods disagree."
            ],
        }

    monkeypatch.setattr(
        nodes,
        "evaluate_program_evidence",
        fake_evaluate,
    )

    result = nodes.evidence_gate(
        state
    )

    decision = (
        result["evidence"]
        ["Program_Conflict"]
        ["decision"]
    )

    assert (
        decision["raw_gate_status"]
        == "conflicting_evidence"
    )

    assert (
        decision["status"]
        == "not_supported"
    )

    assert (
        decision["evidence_conflict"]
        is True
    )


def test_evidence_gate_insufficient_robustness_is_preserved(
    monkeypatch,
):
    state = make_state(
        "Program_Missing"
    )

    def fake_evaluate(
        evidence,
        target_scale=None,
    ):
        return {
            "status":
                "insufficient_robustness_evidence",
            "consensus_ok": True,
            "target_ok": True,
            "robust_ok": False,
            "evidence_conflict": False,
            "reasons": [
                "Robustness evidence unavailable."
            ],
        }

    monkeypatch.setattr(
        nodes,
        "evaluate_program_evidence",
        fake_evaluate,
    )

    result = nodes.evidence_gate(
        state
    )

    decision = (
        result["evidence"]
        ["Program_Missing"]
        ["decision"]
    )

    assert (
        decision["status"]
        == "insufficient_robustness_evidence"
    )

    assert (
        decision["raw_gate_status"]
        == "insufficient_robustness_evidence"
    )


def test_evidence_gate_without_candidate():
    state = {
        "query":
            "Which programs prefer 10x?",
        "current_candidate": None,
        "evidence": {},
        "trace": [],
    }

    result = nodes.evidence_gate(
        state
    )

    assert (
        result["current_decision"]
        == "no_candidate"
    )

    assert any(
        "no current candidate"
        in message.lower()
        for message in result["trace"]
    )


# ============================================================
# 3. Evidence-quality flags
# ============================================================

def test_clean_evidence_has_no_flags():
    flags = nodes._build_evidence_flags(
        make_clean_item()
    )

    assert all(
        value is False
        for value in flags.values()
    )


def test_aggregation_conflict_flag():
    item = make_clean_item()

    item["evidence_alignment"][
        "conflict"
    ] = True

    flags = nodes._build_evidence_flags(
        item
    )

    assert (
        flags["aggregation_conflict"]
        is True
    )


def test_tie_consensus_flag():
    item = make_clean_item()

    item["consensus"][
        "preferred_scale"
    ] = "tie"

    item["consensus"][
        "max_models_agree"
    ] = 4

    flags = nodes._build_evidence_flags(
        item
    )

    assert flags["tie_consensus"] is True

    assert (
        flags["weak_model_consensus"]
        is True
    )


def test_low_optimum_stability_flag():
    item = make_clean_item()

    item["robustness"][
        "median_optimum_stability_fraction"
    ] = 0.50

    flags = nodes._build_evidence_flags(
        item
    )

    assert (
        flags["low_optimum_stability"]
        is True
    )


def test_low_lomo_compatibility_flag():
    item = make_clean_item()

    item["lomo"][
        "compatibility_fraction"
    ] = 0.75

    flags = nodes._build_evidence_flags(
        item
    )

    assert (
        flags["low_lomo_compatibility"]
        is True
    )


def test_missing_robustness_and_lomo_flags():
    item = make_clean_item()

    item["robustness"] = {
        "available": False,
    }

    item["lomo"] = {
        "available": False,
    }

    flags = nodes._build_evidence_flags(
        item
    )

    assert (
        flags["missing_robustness"]
        is True
    )

    assert (
        flags["missing_lomo"]
        is True
    )


# ============================================================
# 4. ECM-like combined diagnostic case
# ============================================================

def test_ecm_like_case_has_conflict_and_low_stability():
    """
    Mimics the diagnostic structure observed
    for ECM / stroma without using real data.

    Primary decision and evidence diagnostics
    must remain conceptually separate.
    """

    item = make_clean_item()

    item["consensus"][
        "preferred_scale"
    ] = "40x"

    item["consensus"][
        "max_models_agree"
    ] = 5

    item["evidence_alignment"][
        "conflict"
    ] = True

    item["robustness"][
        "median_optimum_stability_fraction"
    ] = 0.50

    flags = nodes._build_evidence_flags(
        item
    )

    assert (
        flags["aggregation_conflict"]
        is True
    )

    assert (
        flags["low_optimum_stability"]
        is True
    )

    assert (
        flags["tie_consensus"]
        is False
    )

    assert (
        flags["missing_robustness"]
        is False
    )