from __future__ import annotations

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


Intent = Literal["classify_ticket", "summarize", "extract_fields", "unknown"]
SchemaName = Literal["TicketResult", "SummaryResult", "ExtractedFields", "UnknownResult"]


class IntentDecision(BaseModel):
    intent: Intent
    schema_name: SchemaName = Field(..., alias="schema")
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = False
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


def content_of(msg) -> str:
    return msg.get("content") if isinstance(msg, dict) else msg.content