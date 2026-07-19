"""Tests for the AI trade evaluator."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from signal_engine.evaluator import (
    KimiEvaluator,
    NoOpEvaluator,
    TradeEvaluation,
    _build_prompt,
    _env_file_key,
    _find_api_key,
    _parse_response,
    build_evaluation_context,
    format_evaluation_for_logging,
    make_evaluator,
)


class TestNoOpEvaluator:
    def test_approves_and_full_scale(self):
        ev = NoOpEvaluator()
        result = ev.evaluate({})
        assert result.decision == "approve"
        assert result.scale == 1.0
        assert result.confidence == 1.0
        assert result.reasoning == ""
        assert result.error is False


class TestParseResponse:
    def test_parses_valid_json(self):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": "reject",
                                "scale": 0.5,
                                "confidence": 0.8,
                                "reasoning": "recent Sharpe is weak",
                            }
                        )
                    }
                }
            ]
        }
        result = _parse_response(raw)
        assert result.decision == "reject"
        assert result.scale == pytest.approx(0.5)
        assert result.confidence == pytest.approx(0.8)
        assert result.reasoning == "recent Sharpe is weak"
        assert result.error is False

    def test_clamps_out_of_range_values(self):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"decision": "approve", "scale": 1.5, "confidence": -0.2}
                        )
                    }
                }
            ]
        }
        result = _parse_response(raw)
        assert result.scale == pytest.approx(1.0)
        assert result.confidence == pytest.approx(0.0)

    def test_defaults_to_approve_on_malformed_json(self):
        raw = {"choices": [{"message": {"content": "not json"}}]}
        result = _parse_response(raw)
        assert result.decision == "approve"
        assert result.scale == 1.0
        assert result.error is True
        assert "Could not parse" in result.reasoning

    def test_defaults_to_approve_on_missing_choices(self):
        result = _parse_response({})
        assert result.decision == "approve"
        assert result.error is True


class TestKimiEvaluator:
    def test_no_api_key_returns_no_op_when_not_required(self):
        with patch("signal_engine.evaluator._find_api_key", return_value=None):
            ev = KimiEvaluator(api_key=None, required=False)
            result = ev.evaluate({})
        assert result.decision == "approve"
        assert result.scale == 1.0
        assert result.error is True
        assert "API key not configured" in result.reasoning

    def test_no_api_key_raises_when_required(self):
        with patch("signal_engine.evaluator._find_api_key", return_value=None):
            ev = KimiEvaluator(api_key=None, required=True)
            with pytest.raises(RuntimeError, match="API key not found"):
                ev.evaluate({})

    def test_calls_api_and_parses_response(self):
        ev = KimiEvaluator(api_key="test-key", required=False)
        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": "approve",
                                "scale": 0.9,
                                "confidence": 0.7,
                                "reasoning": "ok",
                            }
                        )
                    }
                }
            ]
        }
        with patch.object(ev, "_call", return_value=fake_response):
            result = ev.evaluate({"target_date": "2026-07-19"})
        assert result.decision == "approve"
        assert result.scale == pytest.approx(0.9)
        assert result.confidence == pytest.approx(0.7)
        assert result.reasoning == "ok"

    def test_api_error_falls_back_when_not_required(self):
        ev = KimiEvaluator(api_key="test-key", required=False)
        with patch.object(ev, "_call", side_effect=urllib.error.URLError("timed out")):
            result = ev.evaluate({})
        assert result.decision == "approve"
        assert result.error is True
        assert "failed" in result.reasoning

    def test_api_error_raises_when_required(self):
        ev = KimiEvaluator(api_key="test-key", required=True)
        with patch.object(ev, "_call", side_effect=urllib.error.URLError("timed out")):
            with pytest.raises(urllib.error.URLError):
                ev.evaluate({})


class TestKimiCodingEndpoint:
    """Kimi Code keys (sk-kimi-...) route to the coding endpoint."""

    def test_coding_key_uses_kimi_com_endpoint(self):
        ev = KimiEvaluator(api_key="sk-kimi-test-key")
        assert ev.use_coding_endpoint is True
        assert ev.api_base == "https://api.kimi.com/coding/v1"
        assert ev.temperature == 1.0
        assert ev.model == "kimi-k2-0711preview"

    def test_moonshot_key_uses_moonshot_cn_endpoint(self):
        ev = KimiEvaluator(api_key="sk-moonshot-test-key")
        assert ev.use_coding_endpoint is False
        assert ev.api_base == "https://api.moonshot.cn/v1"
        assert ev.temperature == 0.2
        assert ev.model == "moonshot-v1-8k"

    def test_coding_request_inlines_system_prompt(self):
        ev = KimiEvaluator(api_key="sk-kimi-test-key")
        captured: list = []

        class FakeResponse:
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": '{"decision":"approve","scale":1.0,"confidence":0.9,"reasoning":"ok"}'}}]
                }).encode()

        def fake_urlopen(req, **kwargs):
            captured.append(req)
            return FakeResponse()

        with patch("signal_engine.evaluator.urllib.request.urlopen", side_effect=fake_urlopen):
            ev.evaluate({})

        assert len(captured) == 1
        body = json.loads(captured[0].data)
        assert body["temperature"] == 1.0
        assert "response_format" not in body
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert "You are a disciplined" in body["messages"][0]["content"]

    def test_moonshot_request_uses_system_message(self):
        ev = KimiEvaluator(api_key="sk-moonshot-test-key")
        captured: list = []

        class FakeResponse:
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": '{"decision":"approve","scale":1.0,"confidence":0.9,"reasoning":"ok"}'}}]
                }).encode()

        def fake_urlopen(req, **kwargs):
            captured.append(req)
            return FakeResponse()

        with patch("signal_engine.evaluator.urllib.request.urlopen", side_effect=fake_urlopen):
            ev.evaluate({})

        assert len(captured) == 1
        body = json.loads(captured[0].data)
        assert body["temperature"] == 0.2
        assert body["response_format"] == {"type": "json_object"}
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
class TestBuildEvaluationContext:
    def test_includes_all_provided_fields(self):
        target = {"date": "2026-07-19", "book": "champion", "units": {"SPY": 10.0}}
        ctx = build_evaluation_context(
            target=target,
            current_positions={"SPY": 5.0},
            order_deltas={"SPY": 5.0},
            metrics={"sharpe": 0.6},
            engine_state={"governor": 1.1},
            regime={"vix": 18.0},
            risk={"gross_notional": 1_000_000.0},
            mode="scale",
        )
        assert ctx["target_date"] == "2026-07-19"
        assert ctx["book"] == "champion"
        assert ctx["target_units"]["SPY"] == 10.0
        assert ctx["current_positions"]["SPY"] == 5.0
        assert ctx["order_deltas"]["SPY"] == 5.0
        assert ctx["metrics"]["sharpe"] == 0.6
        assert ctx["engine_state"]["governor"] == 1.1
        assert ctx["regime"]["vix"] == 18.0
        assert ctx["risk"]["gross_notional"] == 1_000_000.0
        assert ctx["mode"] == "scale"


class TestBuildPrompt:
    def test_contains_target_and_deltas(self):
        ctx = build_evaluation_context(
            target={"date": "2026-07-19", "units": {"SPY": 10.0}, "notional": {"SPY": 1000.0}},
            current_positions={"SPY": 5.0},
            order_deltas={"SPY": 5.0},
            mode="scale",
        )
        prompt = _build_prompt(ctx)
        assert "2026-07-19" in prompt
        assert "SPY" in prompt
        assert "Proposed target positions" in prompt
        assert "Planned order deltas" in prompt


class TestMakeEvaluator:
    def test_returns_no_op_when_disabled(self):
        ev = make_evaluator(use=False)
        assert isinstance(ev, NoOpEvaluator)

    def test_returns_kimi_evaluator(self):
        ev = make_evaluator(use=True, provider="kimi", api_key="key")
        assert isinstance(ev, KimiEvaluator)
        assert ev.api_key == "key"
        assert ev.api_base == "https://api.moonshot.cn/v1"

    def test_returns_openai_evaluator(self):
        ev = make_evaluator(use=True, provider="openai", api_key="key")
        assert isinstance(ev, KimiEvaluator)
        assert ev.api_base == "https://api.openai.com/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown AI evaluator provider"):
            make_evaluator(use=True, provider="unknown")


class TestFormatEvaluationForLogging:
    def test_serializes_all_fields(self):
        ev = TradeEvaluation(
            decision="approve", scale=0.8, confidence=0.7, reasoning="ok", error=False
        )
        d = format_evaluation_for_logging(ev)
        assert d == {
            "decision": "approve",
            "scale": 0.8,
            "confidence": 0.7,
            "reasoning": "ok",
            "error": False,
        }


class TestEnvFileKey:
    def test_reads_key_from_env_file(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("kimi_api_key=secret123\n")
        assert _env_file_key("kimi_api_key", path=env) == "secret123"

    def test_returns_none_when_missing(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("other_key=value\n")
        assert _env_file_key("kimi_api_key", path=env) is None

    def test_returns_none_when_file_missing(self, tmp_path):
        assert _env_file_key("kimi_api_key", path=tmp_path / "missing") is None


class TestFindApiKey:
    def test_prefers_env_over_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIMI_API_KEY", "env-key")
        env = tmp_path / ".env"
        env.write_text("kimi_api_key=file-key\n")
        # Patch CWD to tmp_path so _find_api_key looks at the right .env.
        with patch("signal_engine.evaluator.Path.cwd", return_value=tmp_path):
            assert _find_api_key() == "env-key"

    def test_falls_back_to_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("kimi_api_key=file-key\n")
        with patch("signal_engine.evaluator.Path.cwd", return_value=tmp_path):
            assert _find_api_key() == "file-key"
