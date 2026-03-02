def make_initial_state() -> dict:
    return {
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


def reset_turn_fields(state: dict) -> None:
    state["retries"] = 0
    state["next"] = None
    state["correction"] = None