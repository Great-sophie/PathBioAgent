from typing import Any, Dict, List, Optional, TypedDict


class BiomarkerAgentState(TypedDict, total=False):
    """
    Shared LangGraph state for PathBioAgent.
    """

    # ========================================================
    # User input
    # ========================================================

    query: str


    # ========================================================
    # Structured query interpretation
    # ========================================================

    structured_query: Dict[str, Any]

    ranking_intent: Optional[str]

    task_type: str


    # ========================================================
    # Candidate management
    # ========================================================

    candidate_queue: List[str]

    current_candidate: Optional[str]

    tested_candidates: List[str]


    # ========================================================
    # Scientific evidence
    # ========================================================

    evidence: Dict[str, Any]

    # ========================================================
    # Molecular visibility discovery evidence
    # ========================================================

    discovery_evidence: Dict[str, Any]


    # ========================================================
    # Gene-level explanation evidence
    # ========================================================

    gene_explanation: Dict[str, Any]

    # ========================================================
    # Biological pattern evidence
    # ========================================================

    biological_pattern_evidence: Dict[str, Any]


    # ========================================================
    # Cluster biological comparison evidence
    # ========================================================

    cluster_comparison_evidence: Dict[str, Any]


    # ========================================================
    # Gene-level scientific reasoning
    # ========================================================

    gene_reasoning: str


    # ========================================================
    # Current evidence decision
    # ========================================================

    current_decision: Optional[str]


    # ========================================================
    # Workflow control
    # ========================================================

    iteration: int

    max_iterations: int


    # ========================================================
    # Audit trace
    # ========================================================

    trace: List[str]


    # ========================================================
    # LLM-generated scientific interpretation
    # ========================================================

    scientific_interpretation: Optional[str]


    # ========================================================
    # Final structured report
    # ========================================================

    final_report: Optional[Dict[str, Any]]