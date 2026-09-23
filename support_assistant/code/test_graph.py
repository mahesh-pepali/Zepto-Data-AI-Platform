from support_assistant.code.graph import run_support_graph


def run_test(query: str):
    print("=" * 60)
    print(f"Query: {query}")

    result = run_support_graph(query)

    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    run_test("How does order tracking work?")
    run_test("What is the capital of India?")