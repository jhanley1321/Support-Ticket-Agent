from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

from cli import run_cli
from state_store import make_initial_state, reset_turn_fields
from prompts import ROUTER_SYSTEM_PROMPT

from agents.verifier_agent import VerifierAgent
from agents.worker_agent import WorkerAgent

from ticket_models import (
    IntentDecision,
    TicketResult,
    SummaryResult,
    ExtractedFields,
    UnknownResult,
    State,
    content_of,
)


# ----------------------------
# Debug logging
# ----------------------------

class DebugLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def log(self, msg: str) -> None:
        if self.enabled:
            print(f"[debug] {msg}")


# ----------------------------
# LLM factory
# ----------------------------

def build_llm(model: str, provider: Optional[str] = None):
    if provider is None and ":" in model:
        prefix, rest = model.split(":", 1)
        if prefix == "ollama":
            return ChatOllama(model=rest)
        return init_chat_model(model)

    if provider == "ollama":
        if ":" in model:
            prefix, rest = model.split(":", 1)
            model = rest if prefix == "ollama" else model
        return ChatOllama(model=model)

    if provider is not None:
        return init_chat_model(model, model_provider=provider)

    raise ValueError(
        f"Unable to infer provider for model='{model}'. "
        "Pass provider='ollama' (or other) OR use a prefix like 'anthropic:...'."
    )


# ----------------------------
# Router node (still in main for now)
# ----------------------------

class IntentRouterAgent:
    def __init__(self, llm, logger: DebugLogger):
        self.llm = llm
        self.log = logger
        self.structured = self.llm.with_structured_output(IntentDecision)

    def __call__(self, state: State) -> dict:
        self.log.log("router: deciding intent/schema")
        user_text = content_of(state["messages"][-1])

        decision: IntentDecision = self.structured.invoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=user_text)]
        )

        return {
            "intent": decision.intent,
            "schema": decision.schema_name,
            "router_confidence": decision.confidence,
            "needs_review": decision.needs_review,
        }


# ----------------------------
# Graph assembly
# ----------------------------

class SupportAgentGraph:
    def __init__(self, llm, debug: bool = False):
        logger = DebugLogger(enabled=debug)

        self.router = IntentRouterAgent(llm, logger)

        schema_models = {
            "TicketResult": TicketResult,
            "SummaryResult": SummaryResult,
            "ExtractedFields": ExtractedFields,
            "UnknownResult": UnknownResult,
        }

        self.worker = WorkerAgent(
            llm,
            log=logger.log,
            schema_models=schema_models,
            content_of=content_of,
        )

        self.verifier = VerifierAgent(log=logger.log)

        self.graph = self._build()

    def _build(self):
        gb = StateGraph(State)

        gb.add_node("router", self.router)
        gb.add_node("worker", self.worker)
        gb.add_node("verifier", self.verifier)

        gb.add_edge(START, "router")
        gb.add_edge("router", "worker")
        gb.add_edge("worker", "verifier")

        gb.add_conditional_edges(
            "verifier",
            lambda s: s.get("next"),
            {"worker": "worker", "end": END},
        )

        return gb.compile()


if __name__ == "__main__":
    load_dotenv()

    MODEL = "llama3.2"
    PROVIDER = "ollama"
    DEBUG = True

    llm = build_llm(MODEL, PROVIDER)
    app = SupportAgentGraph(llm, debug=DEBUG)

    run_cli(
        app.graph,
        make_state=make_initial_state,
        reset_turn_fields=reset_turn_fields,
    )