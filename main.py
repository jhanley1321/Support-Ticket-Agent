from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Type
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

#-------------
from cli import run_cli
from state_store import make_initial_state, reset_turn_fields 


# ----------------------------
# Types / Schemas
# ----------------------------

Intent = Literal["classify_ticket", "summarize", "extract_fields", "unknown"]
SchemaName = Literal["TicketResult", "SummaryResult", "ExtractedFields", "UnknownResult"]


class IntentDecision(BaseModel):
    intent: Intent
    # Don't name this field "schema" (it clashes with BaseModel.schema()).
    # We alias it to "schema" so the structured output still uses that key.
    schema_name: SchemaName = Field(..., alias="schema")
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = False

    # Allows either "schema" (alias) or "schema_name" to populate this field.
    model_config = {"populate_by_name": True}


class TicketResult(BaseModel):
    category: str
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str
    action_items: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class SummaryResult(BaseModel):
    summary: str
    action_items: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractedFields(BaseModel):
    requester: Optional[str] = None
    product: Optional[str] = None
    issue: Optional[str] = None
    urgency: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class UnknownResult(BaseModel):
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ----------------------------
# Graph state
# ----------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

    intent: Optional[Intent]
    schema: Optional[SchemaName]
    router_confidence: Optional[float]
    needs_review: Optional[bool]

    result: Optional[dict]

    correction: Optional[str]
    retries: int
    next: Optional[str]


def _content(msg) -> str:
    # LangGraph message lists can contain dicts or LangChain message objects.
    return msg.get("content") if isinstance(msg, dict) else msg.content


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
    """
    Supports:
      - model="anthropic:..." (provider inferred from prefix)
      - model="llama3.2", provider="ollama"
      - model="gpt-4.1", provider="openai" (etc.)
    """
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
# Nodes
# ----------------------------

class IntentRouterAgent:
    def __init__(self, llm, logger: DebugLogger):
        self.llm = llm
        self.log = logger
        self.structured = self.llm.with_structured_output(IntentDecision)

    def __call__(self, state: State) -> dict:
        self.log.log("router: deciding intent/schema")
        user_text = _content(state["messages"][-1])

        system = SystemMessage(
            content=(
                "You are an intent router for a support-ticket assistant.\n"
                "Choose ONE intent:\n"
                "- classify_ticket: user text is a ticket; categorize/priority/summary/action items\n"
                "- summarize: user wants a summary + action items\n"
                "- extract_fields: user wants key fields pulled out\n"
                "- unknown: not sure\n\n"
                "Also choose which schema to fill next (field name is 'schema'):\n"
                "TicketResult / SummaryResult / ExtractedFields / UnknownResult.\n"
                "Return confidence 0..1 and set needs_review if uncertain."
            )
        )

        decision: IntentDecision = self.structured.invoke([system, HumanMessage(content=user_text)])

        return {
            "intent": decision.intent,
            "schema": decision.schema_name,  # IMPORTANT: do not use decision.schema (method)
            "router_confidence": decision.confidence,
            "needs_review": decision.needs_review,
        }


class WorkerAgent:
    def __init__(self, llm, logger: DebugLogger):
        self.llm = llm
        self.log = logger

    def _schema_model(self, schema: SchemaName | str) -> Type[BaseModel]:
        mapping: dict[str, Type[BaseModel]] = {
            "TicketResult": TicketResult,
            "SummaryResult": SummaryResult,
            "ExtractedFields": ExtractedFields,
            "UnknownResult": UnknownResult,
        }
        return mapping.get(str(schema), UnknownResult)

    def __call__(self, state: State) -> dict:
        self.log.log("worker: producing structured output")
        user_text = _content(state["messages"][-1])

        intent = state.get("intent") or "unknown"
        schema = state.get("schema") or "UnknownResult"
        correction = state.get("correction")

        schema_model = self._schema_model(schema)
        structured = self.llm.with_structured_output(schema_model)

        system_text = (
            f"You are the Worker/Executor.\n"
            f"Intent: {intent}\n"
            f"Fill the {schema} schema.\n"
            f"Confidence must be 0..1.\n"
        )
        if correction:
            system_text += f"\nFix this and retry: {correction}\n"

        result_obj: BaseModel = structured.invoke(
            [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        )

        return {
            "result": result_obj.model_dump(),
            "correction": None,
        }


class VerifierAgent:
    def __init__(self, logger: DebugLogger):
        self.log = logger

    def __call__(self, state: State) -> dict:
        self.log.log("verifier: checking output")
        result = state.get("result")
        retries = state.get("retries", 0)

        if not isinstance(result, dict):
            return self._retry_or_end(retries, "Result missing or not a JSON object.")

        conf = result.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            return self._retry_or_end(
                retries,
                "Set 'confidence' to a number between 0 and 1 and keep required fields present."
            )

        return {"next": "end"}

    def _retry_or_end(self, retries: int, feedback: str) -> dict:
        if retries < 1:
            self.log.log("verifier: requesting retry")
            return {"retries": retries + 1, "correction": feedback, "next": "worker"}

        self.log.log("verifier: retries exhausted; marking needs_review")
        return {"next": "end", "needs_review": True}


# ----------------------------
# Graph assembly
# ----------------------------

class SupportAgentGraph:
    def __init__(self, llm, debug: bool = False):
        logger = DebugLogger(enabled=debug)

        self.router = IntentRouterAgent(llm, logger)
        self.worker = WorkerAgent(llm, logger)
        self.verifier = VerifierAgent(logger)

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


# # ----------------------------
# # CLI
# # ----------------------------

# def run_cli(graph):
#     state: State = {
#         "messages": [],
#         "intent": None,
#         "schema": None,
#         "router_confidence": None,
#         "needs_review": False,
#         "result": None,
#         "correction": None,
#         "retries": 0,
#         "next": None,
#     }

#     while True:
#         user_input = input("Message: ").strip()
#         if user_input.lower() == "exit":
#             print("Bye")
#             break

#         # Per-turn control fields.
#         state["retries"] = 0
#         state["next"] = None
#         state["correction"] = None

#         state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
#         state = graph.invoke(state)

#         print("\n--- ROUTE ---")
#         print(
#             f"intent={state.get('intent')} schema={state.get('schema')} "
#             f"router_conf={state.get('router_confidence')} needs_review={state.get('needs_review')}"
#         )
#         print("--- RESULT ---")
#         print(json.dumps(state.get("result"), indent=2))
#         print("")


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