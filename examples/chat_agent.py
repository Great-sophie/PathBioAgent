from pathbioagent.graph import build_graph


def main():

    print("=" * 70)
    print("PathBioAgent Interactive Mode")
    print("Type 'exit' to quit")
    print("=" * 70)


    graph = build_graph()


    while True:

        query = input(
            "\nUser> "
        ).strip()


        if query.lower() in [
            "exit",
            "quit",
        ]:
            break


        result = graph.invoke(
            {
                "query": query,
                "trace": [],
                "max_iterations": 10,
            }
        )


        print("\nAgent Report")
        print("-" * 70)


        report = result.get(
            "final_report",
            {}
        )


        print(
            report
        )


if __name__ == "__main__":

    main()