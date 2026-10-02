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