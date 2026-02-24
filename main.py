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


# ----------------------------
# Schemas (minimal)
# ----------------------------

Intent = Literal["classify_ticket", "summarize", "extract_fields", "unknown"]
SchemaName = Literal["TicketResult", "SummaryResult", "ExtractedFields", "UnknownResult"]


class IntentDecision(BaseModel):
    intent: Intent
    schema: SchemaName
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = False


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


class VerifierDecision(BaseModel):
    is_valid: bool
    needs_review: bool = False
    feedback: str = ""


# ----------------------------
# State
# ----------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

    # Router outputs
    intent: Optional[Intent]
    schema: Optional[SchemaName]
    router_confidence: Optional[float]
    needs_review: Optional[bool]

    # Worker output
    result: Optional[dict]

    # Verifier loop control
    correction: Optional[str]
    retries: int
    next: Optional[str]


def _content(msg) -> str:
    return msg.get("content") if isinstance(msg, dict) else msg.content


# ----------------------------
# Minimal LLM factory (provider-ready)
# ----------------------------

def build_llm(model: str, provider: Optional[str] = None):
    # Original-style: "provider:model"
    if provider is None and ":" in model:
        prefix, rest = model.split(":", 1)
        if prefix == "ollama":
            return ChatOllama(model=rest)
        return init_chat_model(model)

    # Preferred-style: provider + model separately
    if provider == "ollama":
        if ":" in model:
            prefix, rest = model.split(":", 1)
            model = rest if prefix == "ollama" else model
        return ChatOllama(model=model)

    if provider is not None:
        return init_chat_model(model, model_provider=provider)

    raise ValueError(
        f"Unable to infer provider for model='{model}'. "
        "Use provider='ollama' (or other) OR prefix like 'anthropic:...'."
    )


# ----------------------------
# Nodes
# ----------------------------

class IntentRouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.structured = self.llm.with_structured_output(IntentDecision)

    def __call__(self, state: State) -> dict:
        last = state["messages"][-1]
        user_text = _content(last)

        system = SystemMessage(
            content=(
                "You are an intent router for a support-ticket assistant.\n"
                "Choose ONE intent:\n"
                "- classify_ticket: user text is a ticket that needs categorization/priority/summary/action items\n"
                "- summarize: user wants a summary + action items\n"
                "- extract_fields: user wants key fields pulled out\n"
                "- unknown: not sure\n\n"
                "Also choose which schema to fill:\n"
                "TicketResult / SummaryResult / ExtractedFields / UnknownResult.\n"
                "Return confidence 0..1 and set needs_review if uncertain."
            )
        )

        decision: IntentDecision = self.structured.invoke([system, HumanMessage(content=user_text)])

        return {
            "intent": decision.intent,
            "schema": decision.schema,
            "router_confidence": decision.confidence,
            "needs_review": decision.needs_review,
        }


class WorkerAgent:
    def __init__(self, llm):
        self.llm = llm

    def _schema_model(self, schema: SchemaName) -> Type[BaseModel]:
        return {
            "TicketResult": TicketResult,
            "SummaryResult": SummaryResult,
            "ExtractedFields": ExtractedFields,
            "UnknownResult": UnknownResult,
        }[schema]

    def __call__(self, state: State) -> dict:
        last = state["messages"][-1]
        user_text = _content(last)

        intent = state.get("intent") or "unknown"
        schema = state.get("schema") or "UnknownResult"
        correction = state.get("correction")

        schema_model = self._schema_model(schema)
        structured = self.llm.with_structured_output(schema_model)

        system_text = (
            f"You are the Worker/Executor.\n"
            f"Intent: {intent}\n"
            f"Fill the {schema} schema correctly.\n"
            f"Be concise. Confidence must be 0..1.\n"
        )
        if correction:
            system_text += f"\nCorrection from verifier (apply this): {correction}\n"

        result_obj: BaseModel = structured.invoke(
            [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        )

        return {
            "result": result_obj.model_dump(),
            "correction": None,  # clear correction after use
        }


class VerifierAgent:
    """
    Minimal verifier:
    - Checks that result exists
    - Checks confidence field exists and is 0..1
    - If invalid: one retry with a correction message
    """
    def __call__(self, state: State) -> dict:
        result = state.get("result")
        retries = state.get("retries", 0)

        if not isinstance(result, dict):
            return self._retry_or_end(retries, "Result missing or not a JSON object.")

        conf = result.get("confidence")
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            return self._retry_or_end(
                retries,
                "Set 'confidence' to a number between 0 and 1 and keep all required fields present."
            )

        # Minimal: pass through, maybe keep needs_review if router was unsure
        return {"next": "end"}

    @staticmethod
    def _retry_or_end(retries: int, feedback: str) -> dict:
        if retries < 1:
            return {
                "retries": retries + 1,
                "correction": feedback,
                "next": "worker",
            }
        return {
            "next": "end",
            "needs_review": True,
        }


# ----------------------------
# Build graph
# ----------------------------

class SupportAgentGraph:
    def __init__(self, llm):
        self.llm = llm
        self.router = IntentRouterAgent(llm)
        self.worker = WorkerAgent(llm)
        self.verifier = VerifierAgent()
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


# ----------------------------
# CLI
# ----------------------------

def run_cli(graph):
    state: State = {
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
        user_input = input("Message: ")
        if user_input.strip().lower() == "exit":
            print("Bye")
            break

        # reset per turn loop control
        state["retries"] = 0
        state["next"] = None
        state["correction"] = None

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)

        print("\n--- ROUTE ---")
        print(f"intent={state.get('intent')} schema={state.get('schema')} "
              f"router_conf={state.get('router_confidence')} needs_review={state.get('needs_review')}")
        print("--- RESULT ---")
        print(json.dumps(state.get("result"), indent=2))
        print("")


if __name__ == "__main__":
    load_dotenv()

    # swap these later without touching graph logic
    MODEL = "llama3.2"
    PROVIDER = "ollama"

    llm = build_llm(MODEL, PROVIDER)
    app = SupportAgentGraph(llm)
    run_cli(app.graph)