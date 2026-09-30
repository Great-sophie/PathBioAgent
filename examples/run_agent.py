import json

from pathbioagent.graph import build_graph


def main():

    print("=" * 70)
    print("PathBioAgent v0.1")
    print("Evidence-driven morphology-molecular hypothesis agent")
    print("=" * 70)

    graph = build_graph()

    initial_state = {
        "query": (
            "Which molecular programs show the strongest "
            "and most stable morphology association at 10x?"
        ),
        "max_iterations": 10,
        "trace": [],
    }

    print()
    print("USER QUERY")
    print(initial_state["query"])

    # --------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------

    result = graph.invoke(
        initial_state
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(
        json.dumps(
            result["final_report"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 70)
    print("AGENT TRACE")
    print("=" * 70)

    for i, step in enumerate(
        result.get("trace", []),
        start=1,
    ):
        print(f"{i:02d}. {step}")


if __name__ == "__main__":
    main()
