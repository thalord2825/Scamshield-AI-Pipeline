"""
Unit & Integration Tests for ScamShield AI Pipeline Microservice
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app, startup_event

# Run startup hook
startup_event()
client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ScamShield AI Pipeline Engine"
    assert data["tier1_ready"] is True


def test_scan_tier1_fastpath_scam():
    payload = {
        "content": "Vietcombank: Tai khoan cua ban tam khoa do bat thuong. Dang nhap https://vcb-xacthuc.xyz de mo khoa.",
        "check_type": "TEXT"
    }
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert data["verdict"]["risk_level"] in ["DANGEROUS", "SUSPICIOUS"]
    assert data["tier_executed"] == 1
    assert len(data["actionable_recommendations"]) > 0


def test_scan_tier1_fastpath_ham():
    payload = {
        "content": "Ma OTP xac thuc giao dich ShopeePay cua ban la 839201. Ma co hieu luc trong 3 phut. Khong chia se cho bat ky ai.",
        "check_type": "TEXT"
    }
    response = client.post("/api/v1/scan", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert data["verdict"]["risk_level"] == "SAFE"
    assert data["tier_executed"] == 1
