from typing import Any, Dict, List, Optional, TypedDict


class BiomarkerAgentState(TypedDict, total=False):
    """Shared LangGraph state for PathBioAgent."""

    # User query
    query: str

    # Parsed task type
    task_type: str

    # Candidate management
    candidate_queue: List[str]
    current_candidate: Optional[str]
    tested_candidates: List[str]

    # Scientific evidence collected by tools
    evidence: Dict[str, Any]

    # Agent decision
    current_decision: Optional[str]

    # Workflow control
    iteration: int
    max_iterations: int

    # Auditable execution history
    trace: List[str]

    # Final structured output
    final_report: Optional[Dict[str, Any]]
