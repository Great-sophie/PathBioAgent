from typing import Any, Dict, List, Optional, TypedDict


class BiomarkerAgentState(TypedDict, total=False):
    """Shared LangGraph state for PathBioAgent."""

    # User input
    query: str

    # Structured query interpretation produced by
    # the LLM interpreter or deterministic fallback.
    structured_query: Dict[str, Any]

    # Workflow task type
    task_type: str

    # Candidate management
    candidate_queue: List[str]
    current_candidate: Optional[str]
    tested_candidates: List[str]

    # Scientific evidence
    evidence: Dict[str, Any]

    # Current deterministic decision
    current_decision: Optional[str]

    # Workflow control
    iteration: int
    max_iterations: int

    # Auditable execution history
    trace: List[str]

    # Final structured output
    final_report: Optional[Dict[str, Any]]