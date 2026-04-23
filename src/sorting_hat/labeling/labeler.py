"""LLM-backed intent labeler using LiteLLM.

LiteLLM provides a unified interface; this class routes through:
  • A LiteLLM proxy/gateway (set LITELLM_BASE_URL + LITELLM_API_KEY), OR
  • Direct provider access (set ANTHROPIC_API_KEY / OPENAI_API_KEY, etc.).

Output is forced to JSON via `response_format={"type": "json_object"}` and
validated with Pydantic. The system prompt also spells out the JSON schema,
giving an extra belt-and-suspenders guarantee for providers whose json_object
mode is only a soft hint.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Iterable, Sequence

import litellm
from pydantic import ValidationError

from sorting_hat.labeling.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from sorting_hat.labeling.schema import (
    IntentLabel,
    LabeledTurn,
    LabelResult,
    SessionContext,
    TurnInput,
)


DEFAULT_MODEL = "azure/gpt-5.4"


def _resolve_model(model: str, api_base: str | None) -> str:
    """When going through a LiteLLM gateway, prefix with `litellm_proxy/` so the
    SDK passes the model name through verbatim (instead of trying to route via
    the inferred provider)."""
    if api_base and not model.startswith(("litellm_proxy/", "openai/")):
        return f"litellm_proxy/{model}"
    return model


def _merge_consecutive(
    turns: Sequence[TurnInput],
    gap_seconds: float = 30.0,
) -> list[TurnInput]:
    """Merge consecutive turns from the same role within gap_seconds into one.

    Reduces ASR fragment noise so the context window covers more distinct
    exchanges. Falls back to role-only merging when timestamps are absent.
    """
    if not turns:
        return []

    def ts(t: TurnInput) -> float | None:
        if not t.timestamp:
            return None
        try:
            from datetime import timezone
            dt = datetime.fromisoformat(t.timestamp.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None

    merged: list[TurnInput] = []
    cur = turns[0]
    cur_ts = ts(cur)

    for nxt in turns[1:]:
        nxt_ts = ts(nxt)
        same_role = (nxt.role or "") == (cur.role or "")
        within_gap = (
            cur_ts is None or nxt_ts is None or (nxt_ts - cur_ts) <= gap_seconds
        )
        if same_role and within_gap:
            cur = TurnInput(
                turn_id=cur.turn_id,
                text=(cur.text or "").rstrip() + " " + (nxt.text or "").lstrip(),
                role=cur.role,
                source=cur.source,
                timestamp=cur.timestamp,
            )
            cur_ts = cur_ts  # keep start timestamp of the merged block
        else:
            merged.append(cur)
            cur = nxt
            cur_ts = nxt_ts

    merged.append(cur)
    return merged


def _truncate_middle(txt: str, max_chars: int) -> str:
    """Keep head and tail, elide the middle with ' … '."""
    if len(txt) <= max_chars:
        return txt
    half = (max_chars - 3) // 2  # 3 chars for " … "
    return txt[:half] + " … " + txt[-half:]


def _format_prior_turns(
    prior: Sequence[TurnInput],
    max_turns: int = 6,
    max_chars: int = 300,
    merge_gap_seconds: float = 30.0,
) -> str:
    if not prior:
        return "(no prior turns)"
    merged = _merge_consecutive(prior, gap_seconds=merge_gap_seconds)
    subset = merged[-max_turns:]
    lines = []
    for t in subset:
        tag = t.role or "?"
        txt = (t.text or "").strip().replace("\n", " ")
        txt = _truncate_middle(txt, max_chars)
        lines.append(f"[{tag}] {txt}")
    return "\n".join(lines)


def _render_user(
    turn: TurnInput,
    prior: Sequence[TurnInput],
    session: SessionContext,
    max_prior_turns: int = 6,
) -> str:
    return USER_TEMPLATE.format(
        interview_mode=session.interview_mode or "-",
        programming_language=session.programming_language or "-",
        goal_position=session.goal_position or "-",
        goal_company=session.goal_company or "-",
        prior_turns=_format_prior_turns(prior, max_turns=max_prior_turns),
        role=turn.role or "-",
        source=turn.source or "-",
        text=(turn.text or "").strip(),
    )


class IntentLabeler:
    """Label interview turns with 5-class intent via LiteLLM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_API_BASE")
        self.max_tokens = max_tokens

    def _completion_kwargs(self, messages: list[dict]) -> dict:
        # Always stream + accumulate: the LiteLLM proxy's non-streaming path
        # returns empty content for Azure Responses-API-format models (content: null),
        # while streaming deltas come through correctly.
        kwargs: dict = {
            "model": _resolve_model(self.model, self.api_base),
            "max_tokens": self.max_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": True,
        }
        # gpt-5 and o-series only accept temperature=1 (their default); omit the knob entirely.
        m = self.model.lower()
        if not (m.startswith("azure/gpt-5") or m.startswith("gpt-5")
                or "/o1" in m or "/o3" in m or "/o4" in m):
            kwargs["temperature"] = 0.0
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    def _stream_content(self, messages: list[dict], retries: int = 3) -> str:
        delay = 2.0
        for attempt in range(retries):
            try:
                chunks = []
                stream = litellm.completion(**self._completion_kwargs(messages))
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        chunks.append(delta)
                return "".join(chunks)
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def label_turn(
        self,
        turn: TurnInput,
        prior_turns: Sequence[TurnInput] = (),
        session: SessionContext | None = None,
        max_prior_turns: int = 6,
    ) -> LabelResult:
        session = session or SessionContext()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _render_user(turn, prior_turns, session, max_prior_turns)},
        ]
        content = self._stream_content(messages)
        if not content:
            raise RuntimeError(f"empty LLM response for turn {turn.turn_id}")
        try:
            return LabelResult.model_validate_json(content)
        except ValidationError as e:
            cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                return LabelResult.model_validate_json(cleaned)
            except ValidationError:
                raise RuntimeError(
                    f"invalid LLM JSON for turn {turn.turn_id}: {e}\nraw: {content[:500]}"
                ) from e

    def label_many(
        self,
        items: Iterable[tuple[TurnInput, Sequence[TurnInput], SessionContext]],
        on_error: str = "skip",
    ) -> Iterable[LabeledTurn]:
        """Yield labeled turns. on_error='skip' logs and continues; 'raise' propagates."""
        now = datetime.now(timezone.utc).isoformat()
        for turn, prior, session in items:
            try:
                result = self.label_turn(turn, prior, session)
            except Exception as exc:
                if on_error == "raise":
                    raise
                import sys
                print(f"[skip] turn {turn.turn_id}: {exc}", file=sys.stderr)
                continue
            yield LabeledTurn(
                turn_id=turn.turn_id,
                text=turn.text,
                role=turn.role,
                source=turn.source,
                timestamp=turn.timestamp,
                session_id=session.session_id,
                interview_mode=session.interview_mode,
                programming_language=session.programming_language,
                goal_position=session.goal_position,
                label=result.label,
                confidence=result.confidence,
                reason=result.reason,
                secondary_label=result.secondary_label,
                model=self.model,
                labeled_at=now,
            )


__all__ = [
    "IntentLabeler",
    "LabelResult",
    "LabeledTurn",
    "TurnInput",
    "SessionContext",
    "IntentLabel",
    "DEFAULT_MODEL",
]
