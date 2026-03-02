import json
from typing import Callable, Dict, Any


def run_cli(
    graph,
    make_state: Callable[[], Dict[str, Any]],
    reset_turn_fields: Callable[[Dict[str, Any]], None],
) -> None:
    state = make_state()

    while True:
        user_input = input("Message: ").strip()
        if user_input.lower() == "exit":
            print("Bye")
            return

        reset_turn_fields(state)

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)

        print("\n--- ROUTE ---")
        print(
            f"intent={state.get('intent')} schema={state.get('schema')} "
            f"router_conf={state.get('router_confidence')} needs_review={state.get('needs_review')}"
        )
        print("--- RESULT ---")
        print(json.dumps(state.get("result"), indent=2))