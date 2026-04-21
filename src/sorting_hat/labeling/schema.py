"""Pydantic schemas for the intent-labeling module."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IntentLabel(str, Enum):
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    PROJECT_QA = "project_qa"
    CHAT = "chat"
    NO_ANSWER_NEEDED = "no_answer_needed"


class TurnInput(BaseModel):
    """One ASR turn to classify."""
    turn_id: str
    text: str
    role: str = "interviewer"
    source: str | None = None
    timestamp: str | None = None


class SessionContext(BaseModel):
    """Session-level metadata available at labeling time."""
    session_id: str | None = None
    interview_mode: str | None = None
    programming_language: str | None = None
    goal_position: str | None = None
    goal_company: str | None = None


class LabelResult(BaseModel):
    """LLM-produced label for a single turn."""
    label: IntentLabel = Field(description="Primary intent label")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0-1")
    reason: str = Field(description="Short explanation (1-2 sentences) for the label choice")
    secondary_label: IntentLabel | None = Field(
        default=None,
        description="Optional secondary label for mixed turns. null if pure single-intent.",
    )


class LabeledTurn(BaseModel):
    """Full record combining input + label, suitable for writing to JSONL."""
    turn_id: str
    text: str
    role: str
    source: str | None = None
    timestamp: str | None = None
    session_id: str | None = None
    interview_mode: str | None = None
    programming_language: str | None = None
    goal_position: str | None = None
    label: IntentLabel
    confidence: float
    reason: str
    secondary_label: IntentLabel | None = None
    model: str
    labeled_at: str | None = None
