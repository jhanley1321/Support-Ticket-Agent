# worker_agent.py

from typing import Any, Callable, Mapping, Type

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class WorkerAgent:
    def __init__(
        self,
        llm,
        *,
        log: Callable[[str], None],
        schema_models: Mapping[str, Type[BaseModel]],
        content_of: Callable[[Any], str],
    ):
        self.llm = llm
        self._log = log
        self._schema_models = schema_models
        self._content_of = content_of

    def __call__(self, state: dict) -> dict:
        self._log("worker: producing structured output")

        user_text = self._content_of(state["messages"][-1])
        intent = state.get("intent") or "unknown"
        schema_name = state.get("schema") or "UnknownResult"
        correction = state.get("correction")

        schema_model = self._schema_models.get(schema_name) or self._schema_models["UnknownResult"]
        structured = self.llm.with_structured_output(schema_model)

        system_text = (
            f"You are the Worker/Executor.\n"
            f"Intent: {intent}\n"
            f"Fill the {schema_name} schema.\n"
            f"Confidence must be 0..1.\n"
        )
        if correction:
            system_text += f"\nFix this and retry: {correction}\n"

        result_obj: BaseModel = structured.invoke(
            [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        )

        return {"result": result_obj.model_dump(), "correction": None}