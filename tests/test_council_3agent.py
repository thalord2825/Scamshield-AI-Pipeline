"""
Unit & Algorithmic Tests for Tier-2 3-Agent Forensic Council (Maker-Checker Consensus)
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from pipeline.forensic_council import ForensicCouncil, CircuitBreaker


def test_council_3agent_configuration_loading():
    """Verify that the 3-Agent consolidated roles are loaded with proper weights and personas."""
    council = ForensicCouncil()
    specialists = council.specialists

    # Must contain the 3 consolidated roles
    expected_roles = {
        "cyber_financial_analyst",
        "psycholinguistic_slang_profiler",
        "public_defender_advocate",
    }
    assert expected_roles.issubset(set(specialists.keys())), f"Missing roles in {specialists.keys()}"

    # Verify weights
    assert specialists["cyber_financial_analyst"]["weight"] == 1.3
    assert specialists["psycholinguistic_slang_profiler"]["weight"] == 1.2
    assert specialists["public_defender_advocate"]["weight"] == 1.3

    # Verify Maker-Checker assignments
    assert "gpt-6.1-sol" in specialists["cyber_financial_analyst"]["primary_maker_model"]
    assert "gemini-3.5-flash" in specialists["cyber_financial_analyst"]["secondary_checker_model"]


def test_strict_gemini_model_floor_enforcement():
    """Verify Rule 5: Reject any Gemini model lower than 3.5 and promote automatically."""
    council = ForensicCouncil()
    council.gemini_api_key = "dummy_key"

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"vote": "LOW-RISK", "scam_prob": 0.1, "confidence": 0.9, "risk_level": "LOW", "key_cues": [], "analysis": "Safe"}'}]}}]
        }
        mock_post.return_value = mock_resp

        # Test with forbidden gemini-1.5-flash
        asyncio.run(council._call_gemini_api("gemini-1.5-flash", "test prompt", "test sys"))
        called_url = mock_post.call_args[0][0]
        # Must have been promoted to at least gemini-3.5-flash-lite
        assert "gemini-3.5" in called_url
        assert "gemini-1.5" not in called_url

        # Test with forbidden gemini-2.0-flash
        asyncio.run(council._call_gemini_api("gemini-2.0-flash", "test prompt", "test sys"))
        called_url = mock_post.call_args[0][0]
        assert "gemini-3.5" in called_url
        assert "gemini-2.0" not in called_url


def test_borda_weighted_consensus_calculation():
    """Verify the weighted Bayesian Borda consensus formula across 3 agents."""
    council = ForensicCouncil()

    # Mock completed results for 3 seats:
    # Seat 1 (w=1.3): scam_prob = 0.90
    # Seat 2 (w=1.2): scam_prob = 0.80
    # Seat 3 (w=1.3): scam_prob = 0.70 (Defender also votes SCAM-RISK, so no veto)
    mock_results = [
        {"role_key": "cyber_financial_analyst", "weight": 1.3, "scam_prob": 0.90, "vote": "SCAM-RISK", "confidence": 0.90, "display_name": "Cyber", "model_used": "gpt-6.1-sol", "analysis": "Phishing URL"},
        {"role_key": "psycholinguistic_slang_profiler", "weight": 1.2, "scam_prob": 0.80, "vote": "SCAM-RISK", "confidence": 0.85, "display_name": "Psycho", "model_used": "gpt-6.1-sol", "analysis": "Urgency cues"},
        {"role_key": "public_defender_advocate", "weight": 1.3, "scam_prob": 0.70, "vote": "SCAM-RISK", "confidence": 0.70, "display_name": "Defender", "model_used": "gpt-6.1-sol", "analysis": "Suspicious"},
    ]

    # Weighted calculation:
    # total_w = 1.3 + 1.2 + 1.3 = 3.8
    # sum(w * p) = 1.3*0.90 + 1.2*0.80 + 1.3*0.70 = 1.17 + 0.96 + 0.91 = 3.04
    # weighted_p = 3.04 / 3.8 = 0.80
    expected_prob = (1.3 * 0.90 + 1.2 * 0.80 + 1.3 * 0.70) / (1.3 + 1.2 + 1.3)
    assert abs(expected_prob - 0.80) < 1e-4

    # Run internal consensus resolution
    decision = council._compute_consensus(mock_results, start_overall=0.0, early_exit_triggered=False)
    assert decision["final_label"] == "SCAM-RISK"
    assert decision["risk_level"] in ["HIGH", "CRITICAL"]
    assert abs(decision["consensus_scam_prob"] - 0.80) < 0.01
    assert decision["defender_intervened"] is False


def test_public_defender_veto_boost_prevents_false_positive():
    """Verify that Public Defender asserting LOW-RISK applies -0.25 penalty and prevents false positives."""
    council = ForensicCouncil()

    # Ambiguous case:
    # Seat 1: scam_prob = 0.65, vote = SCAM-RISK (w=1.3)
    # Seat 2: scam_prob = 0.60, vote = SCAM-RISK (w=1.2)
    # Seat 3 (Defender): scam_prob = 0.10, vote = LOW-RISK, confidence = 0.85 (w=1.3)
    # Raw weighted prob = (1.3*0.65 + 1.2*0.60 + 1.3*0.10) / 3.8 = (0.845 + 0.72 + 0.13) / 3.8 = 1.695 / 3.8 = 0.446
    # With Public Defender Veto Boost (-0.25):
    # final_prob = max(0, 0.446 - 0.25) = 0.196 -> LOW-RISK!
    mock_results = [
        {"role_key": "cyber_financial_analyst", "weight": 1.3, "scam_prob": 0.65, "vote": "SCAM-RISK", "confidence": 0.65, "display_name": "Cyber", "model_used": "gpt-6.1-sol", "analysis": "Link looks slightly off"},
        {"role_key": "psycholinguistic_slang_profiler", "weight": 1.2, "scam_prob": 0.60, "vote": "SCAM-RISK", "confidence": 0.60, "display_name": "Psycho", "model_used": "gpt-6.1-sol", "analysis": "Mild urgency"},
        {"role_key": "public_defender_advocate", "weight": 1.3, "scam_prob": 0.10, "vote": "LOW-RISK", "confidence": 0.85, "display_name": "Defender", "model_used": "gpt-6.1-sol", "analysis": "Standard bank OTP notification"},
    ]

    decision = council._compute_consensus(mock_results, start_overall=0.0, early_exit_triggered=False)
    assert decision["defender_intervened"] is True
    assert decision["final_label"] == "LOW-RISK"
    assert decision["consensus_scam_prob"] < 0.40


def test_vote_contradiction_triggers_unknown_label():
    """Verify that severe vote contradiction in boundary zone triggers UNKNOWN (Decree 13/2023 compliance)."""
    council = ForensicCouncil()

    # Conflicting votes:
    # Seat 1: vote = SCAM-RISK, scam_prob = 0.65 (w=1.3)
    # Seat 2: vote = LOW-RISK, scam_prob = 0.45 (w=1.2)
    # Seat 3: vote = LOW-RISK, scam_prob = 0.50 (w=1.3)
    # The votes are split, boundary zone -> must trigger UNKNOWN
    mock_results = [
        {"role_key": "cyber_financial_analyst", "weight": 1.3, "scam_prob": 0.65, "vote": "SCAM-RISK", "confidence": 0.65, "display_name": "Cyber", "model_used": "gpt-6.1-sol", "analysis": "Odd domain"},
        {"role_key": "psycholinguistic_slang_profiler", "weight": 1.2, "scam_prob": 0.45, "vote": "LOW-RISK", "confidence": 0.65, "display_name": "Psycho", "model_used": "gpt-6.1-sol", "analysis": "Neutral language"},
        {"role_key": "public_defender_advocate", "weight": 1.3, "scam_prob": 0.50, "vote": "LOW-RISK", "confidence": 0.50, "display_name": "Defender", "model_used": "gpt-6.1-sol", "analysis": "Ambiguous message"},
    ]

    decision = council._compute_consensus(mock_results, start_overall=0.0, early_exit_triggered=False)
    assert decision["final_label"] == "UNKNOWN"
    assert decision["risk_level"] == "MEDIUM"


def test_circuit_breaker_resilience():
    """Verify CircuitBreaker opens after consecutive failures and enters cooldown."""
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    assert cb.is_available("gpt-6.1-sol") is True

    cb.record_failure("gpt-6.1-sol")
    cb.record_failure("gpt-6.1-sol")
    assert cb.is_available("gpt-6.1-sol") is True

    cb.record_failure("gpt-6.1-sol")
    # 3 failures -> Circuit opens
    assert cb.is_available("gpt-6.1-sol") is False

    # Success resets consecutive failures
    cb.record_success("gpt-6.1-sol")
    assert cb.is_available("gpt-6.1-sol") is True
