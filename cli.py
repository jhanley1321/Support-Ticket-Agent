import json


def run_cli(graph):
    state = {
        "messages": [],
        "intent": None,
        "schema": None,
        "router_confidence": None,
        "needs_review": False,
        "result": None,
        "correction": None,
        "retries": 0,
        "next": None,
    }

    while True:
        user_input = input("Message: ").strip()
        if user_input.lower() == "exit":
            print("Bye")
            break

        # Per-turn control fields.
        state["retries"] = 0
        state["next"] = None
        state["correction"] = None

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)

        print("\n--- ROUTE ---")
        print(
            f"intent={state.get('intent')} schema={state.get('schema')} "
            f"router_conf={state.get('router_confidence')} needs_review={state.get('needs_review')}"
        )
        print("--- RESULT ---")
        print(json.dumps(state.get("result"), indent=2))
        print("")