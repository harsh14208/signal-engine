"""AI-driven trade evaluation before execution.

Provides a pluggable evaluator. By default a no-op is used so the engine keeps
running offline. When enabled, it calls the Kimi (Moonshot) OpenAI-compatible
chat API once per rebalance to evaluate the whole proposed target book.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TradeEvaluation:
    """Result of an AI trade evaluation.

    Advisory only: nothing in execute_alpaca.py blocks or skips a rebalance
    based on this result. `decision` is a caution flag recorded for human
    review; `scale` is the only field with any effect on execution (it
    resizes, never cancels, the target book).
    """

    decision: str = "approve"  # "approve" | "reject" — advisory flag, not enforced
    scale: float = 1.0  # 0.0 - 1.0, applied to the whole target book
    confidence: float = 1.0  # 0.0 - 1.0
    reasoning: str = ""
    raw_response: dict[str, Any] | None = None
    error: bool = False  # True if the evaluation itself failed and this is a fallback


class TradeEvaluator(Protocol):
    """Protocol for a trade evaluator."""

    def evaluate(self, context: dict[str, Any]) -> TradeEvaluation: ...


class NoOpEvaluator:
    """Default evaluator that always approves and leaves positions unchanged."""

    def evaluate(self, context: dict[str, Any]) -> TradeEvaluation:
        return TradeEvaluation(decision="approve", scale=1.0, confidence=1.0, reasoning="")


class KimiEvaluator:
    """Kimi (Moonshot AI) evaluator using the OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://api.moonshot.cn/v1",
        model: str = "moonshot-v1-8k",
        timeout: float = 30.0,
        max_tokens: int = 512,
        temperature: float = 0.2,
        required: bool = False,
    ):
        self.api_key = api_key or _find_api_key()
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.required = required

        # Kimi Code keys (sk-kimi-...) use the separate coding endpoint and require
        # temperature=1.0. The system prompt works best when inlined into the user
        # message. Auto-detect so the same KIMI_API_KEY env var works for both
        # Moonshot consumer keys and Kimi Code developer keys.
        self.use_coding_endpoint = (
            self.api_key is not None and self.api_key.startswith("sk-kimi-")
        )
        if self.use_coding_endpoint:
            if self.api_base == "https://api.moonshot.cn/v1":
                self.api_base = "https://api.kimi.com/coding/v1"
            if self.temperature == 0.2:  # untouched default
                self.temperature = 1.0
            if self.model == "moonshot-v1-8k":  # untouched default
                self.model = "kimi-k2-0711preview"
            if self.max_tokens == 512:  # untouched default; reasoning models need more
                self.max_tokens = 2048

    def evaluate(self, context: dict[str, Any]) -> TradeEvaluation:
        if not self.api_key:
            if self.required:
                raise RuntimeError(
                    "Kimi API key not found. Set KIMI_API_KEY or OPENAI_API_KEY "
                    "in the environment or .env file."
                )
            return TradeEvaluation(
                decision="approve",
                scale=1.0,
                confidence=1.0,
                reasoning="API key not configured; no-op fallback.",
                error=True,
            )

        prompt = _build_prompt(context)
        try:
            raw = self._call(prompt)
        except Exception as exc:
            if self.required:
                raise
            return TradeEvaluation(
                decision="approve",
                scale=1.0,
                confidence=1.0,
                reasoning=f"Kimi API call failed: {exc}",
                raw_response={"error": str(exc)},
                error=True,
            )
        return _parse_response(raw)

    def _call(self, prompt: str) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.use_coding_endpoint:
            # Kimi Code endpoint: inline the system prompt in the user message and
            # do not send a separate system message or response_format.
            body["messages"] = [
                {
                    "role": "user",
                    "content": f"{_SYSTEM_PROMPT}\n\n{prompt}",
                }
            ]
        else:
            body["messages"] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            # Moonshot consumer endpoint supports json_object on some models.
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            text = resp.read().decode()
            return json.loads(text) if text else {}
def _find_api_key() -> str | None:
    """Search environment, then the repo `.env` file for an API key."""
    env_key = os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    for key in ("kimi_api_key", "openai_api_key"):
        val = _env_file_key(key)
        if val:
            return val
    return None


def _env_file_key(key: str, path: Path | None = None) -> str | None:
    """Parse a simple KEY=VALUE `.env` file for a specific key."""
    path = path or (Path.cwd() / ".env")
    if not path.exists():
        return None
    key_lower = key.lower()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().lower() == key_lower:
                return v.strip().strip('"').strip("'")
    return None


def _parse_response(raw: dict[str, Any]) -> TradeEvaluation:
    """Extract the JSON evaluation from the chat response."""
    try:
        choices = raw.get("choices") or []
        if not choices:
            raise ValueError("no choices in response")
        message = choices[0].get("message", {})
        # Reasoning models (Kimi Code) may return the answer in reasoning_content
        # and leave content empty, or include the JSON in content.
        content = message.get("content") or message.get("reasoning_content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty content in response")

        # Some reasoning models wrap the JSON in a fenced block or include extra
        # text. Try to extract the first JSON object.
        parsed = _extract_json_from_text(content)
        if parsed is None:
            raise ValueError("no JSON object found in response")

        decision = str(parsed.get("decision", "approve")).lower()
        scale = float(parsed.get("scale", 1.0))
        confidence = float(parsed.get("confidence", 1.0))
        reasoning = str(parsed.get("reasoning", ""))

        return TradeEvaluation(
            decision="approve" if decision not in ("approve", "reject") else decision,
            scale=max(0.0, min(1.0, scale)),
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            raw_response=raw,
        )
    except Exception as exc:
        return TradeEvaluation(
            decision="approve",
            scale=1.0,
            confidence=1.0,
            reasoning=f"Could not parse Kimi response: {exc}",
            raw_response=raw,
            error=True,
        )


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()

    # 1. Fenced code block.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. First top-level JSON object.
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _build_prompt(context: dict[str, Any]) -> str:
    """Turn a structured context dict into a concise evaluation prompt."""
    target_date = context.get("target_date", "unknown")
    book = context.get("book", "champion")
    mode = context.get("mode", "scale")

    lines = [
        "You are reviewing a proposed systematic trend/carry portfolio rebalance.",
        "",
        f"Target date: {target_date}",
        f"Book: {book}",
        f"Evaluator mode: {mode}",
        "",
    ]

    units = context.get("target_units") or {}
    notional = context.get("target_notional") or {}
    if units:
        lines.append("Proposed target positions (units / notional $):")
        for sym in sorted(units):
            u = units.get(sym, 0.0)
            n = notional.get(sym, 0.0) if isinstance(notional, dict) else 0.0
            lines.append(f"  {sym}: {u:,.2f} units / ${n:,.0f}")
    else:
        lines.append("No target positions.")

    current = context.get("current_positions") or {}
    if current:
        lines.append("")
        lines.append("Current held positions (units):")
        for sym in sorted(current):
            lines.append(f"  {sym}: {current[sym]:,.2f}")

    deltas = context.get("order_deltas") or {}
    if deltas:
        lines.append("")
        lines.append("Planned order deltas (units):")
        for sym in sorted(deltas):
            lines.append(f"  {sym}: {deltas[sym]:+,.2f}")

    metrics = context.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append("Recent engine metrics:")
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")

    state = context.get("engine_state") or {}
    if state:
        lines.append("")
        lines.append("Engine state:")
        for k, v in state.items():
            lines.append(f"  {k}: {v}")

    regime = context.get("regime") or {}
    if regime:
        lines.append("")
        lines.append("Macro regime / overlays:")
        for k, v in regime.items():
            lines.append(f"  {k}: {v}")

    risk = context.get("risk") or {}
    if risk:
        lines.append("")
        lines.append("Risk summary:")
        for k, v in risk.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append(_EVAL_INSTRUCTIONS)
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are a disciplined quantitative risk assistant evaluating a systematic "
    "trend/carry portfolio rebalance. You are skeptical, data-first, and avoid "
    "narrative reasoning. Respond only in valid JSON."
)

_EVAL_INSTRUCTIONS = """\
This evaluation is advisory guidance only — it informs position sizing, it never
blocks or skips a rebalance outright. There is no execution path that halts
trading on your assessment; the only lever with any effect is `scale`.

Return a JSON object with exactly these keys:
  decision: "approve" or "reject" (a caution flag for human review, not an execution veto — recorded but does not stop the rebalance)
  scale: a number between 0.0 and 1.0 (recommended position scaling; 1.0 = full size, 0.0 = flat)
  confidence: a number between 0.0 and 1.0 (how sure you are of this assessment)
  reasoning: a concise one-sentence explanation

Guidelines:
- The strategy is a diversified vol-targeted trend/carry engine. Avoid micromanaging individual names; judge the portfolio-level risk and recent edge decay.
- Scale down when: recent Sharpe is weak, the book is unusually concentrated, correlation-spike or drawdown overlays are already active, or macro regime is hostile to trend.
- For severe, unambiguous risks (e.g. kill switch engaged, data/revision alarm, or the target implies a major unintended concentration): set decision="reject" AND scale near 0.0 — scale is what actually reduces exposure, so express the severity there, not just in decision.
- Do not invent price targets. Use only the data provided.
"""


def build_evaluation_context(
    target: dict[str, Any],
    current_positions: dict[str, float] | None = None,
    order_deltas: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    engine_state: dict[str, Any] | None = None,
    regime: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    mode: str = "scale",
) -> dict[str, Any]:
    """Build a structured context dict for the AI evaluator."""
    return {
        "target_date": target.get("date"),
        "book": target.get("book", "champion"),
        "target_units": target.get("units") or {},
        "target_notional": target.get("notional") or {},
        "current_positions": current_positions or {},
        "order_deltas": order_deltas or {},
        "metrics": metrics or {},
        "engine_state": engine_state or {},
        "regime": regime or {},
        "risk": risk or {},
        "mode": mode,
    }


def make_evaluator(
    use: bool = False,
    provider: str = "kimi",
    api_key: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    max_tokens: int = 512,
    temperature: float = 0.2,
    required: bool = False,
) -> TradeEvaluator:
    """Factory for the configured evaluator."""
    if not use:
        return NoOpEvaluator()
    if provider.lower() == "kimi":
        return KimiEvaluator(
            api_key=api_key,
            api_base=api_base or "https://api.moonshot.cn/v1",
            model=model or "moonshot-v1-8k",
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            required=required,
        )
    # Generic OpenAI-compatible endpoint.
    if provider.lower() in ("openai", "openai-compatible"):
        return KimiEvaluator(
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            model=model or "gpt-4o-mini",
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            required=required,
        )
    raise ValueError(f"Unknown AI evaluator provider: {provider}")


def format_evaluation_for_logging(evaluation: TradeEvaluation) -> dict[str, Any]:
    """Return a JSON-serializable dict of the evaluation result."""
    return {
        "decision": evaluation.decision,
        "scale": evaluation.scale,
        "confidence": evaluation.confidence,
        "reasoning": evaluation.reasoning,
        "error": evaluation.error,
    }
