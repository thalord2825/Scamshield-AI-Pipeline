"""
Unit Tests for PII Masking & Privacy Shield (Decree 13/2023/ND-CP Compliance)
"""

import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from pipeline.pii_masking import mask_pii, redact_sensitive_entities


def test_mask_vietnamese_phone_numbers():
    text = "Lien he Zalo so 0912345678 hoac +84987654321 de nhan thuong."
    masked, entities = mask_pii(text)
    assert "[PHONE_REDACTED]" in masked
    assert "0912345678" not in masked
    assert "+84987654321" not in masked
    assert len(entities.get("phones", [])) >= 2


def test_mask_bank_accounts():
    text = "Vui long chuyen tien vao STK 1903456789012 ngan hang Techcombank."
    masked, entities = mask_pii(text)
    assert "[BANK_ACC_REDACTED]" in masked
    assert "1903456789012" not in masked
    assert len(entities.get("bank_accounts", [])) >= 1


def test_mask_cccd_citizen_id():
    text = "Yeu cau cong dan co so CCCD 001201001234 den co quan cong an xac minh."
    masked, entities = mask_pii(text)
    assert "[ID_REDACTED]" in masked
    assert "001201001234" not in masked
    assert len(entities.get("citizen_ids", [])) >= 1


def test_preserve_urls_and_suspicious_domains():
    text = "Truy cap ngay https://vietcombank-login.xyz/vcb de cap nhat tai khoan 0912345678."
    masked, entities = mask_pii(text)
    assert "https://vietcombank-login.xyz/vcb" in masked
    assert "[PHONE_REDACTED]" in masked
    assert "0912345678" not in masked


def test_empty_or_clean_message():
    text = "Vietcombank tran trong thong bao chuong trinh tri an khach hang nam 2026."
    masked, entities = mask_pii(text)
    assert masked == text
    assert len(entities.get("phones", [])) == 0
    assert len(entities.get("bank_accounts", [])) == 0
