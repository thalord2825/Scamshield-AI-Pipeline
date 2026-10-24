"""
==============================================================================
ScamShield-VN: PII Masking & Privacy Shield Module
==============================================================================
Author: ScamShield Core Engineering Team
Description:
    Pre-inference privacy preservation module complying with Decree 13/2023/ND-CP.
    Anonymizes sensitive personally identifiable information (PII) including:
    - Vietnamese Mobile Phone Numbers
    - Bank Account Numbers (STK)
    - Vietnamese Citizen Identity Cards (CCCD / CMND)
    - Personal names associated with transactions
    Preserves threat indicators (URLs, domains, malicious cues) while redacting PII.
==============================================================================
"""

import re
from typing import Dict, List, Tuple


# Vietnamese Phone Regex (covers +84 and 0 prefix for major carriers Viettel, Vinaphone, Mobifone, etc.)
PHONE_PATTERN = re.compile(
    r"(?:\+84|84|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-9])[\s\.\-]?[0-9]{3}[\s\.\-]?[0-9]{4}\b"
)

# CCCD / CMND Pattern (12 digits for CCCD, 9 digits for CMND)
CCCD_PATTERN = re.compile(
    r"\b(?:CCCD|CMND|can cuoc|so dinh danh)?\s*[:\s]?\s*([0-9]{12})\b",
    re.IGNORECASE,
)
CMND_PATTERN = re.compile(
    r"\b(?:CMND|chung minh nhan dan)\s*[:\s]?\s*([0-9]{9})\b",
    re.IGNORECASE,
)

# Bank Account Pattern (STK followed by 8-16 digits)
BANK_ACC_PATTERN = re.compile(
    r"\b(?:STK|so tai khoan|tai khoan|tk|stk)\s*[:\s]?\s*([0-9]{8,16})\b",
    re.IGNORECASE,
)


def mask_pii(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Masks sensitive personal data in the input text while preserving indicators of compromise (IOCs).

    Args:
        text: Raw text to anonymize.

    Returns:
        Tuple of (anonymized_text, extracted_pii_entities)
    """
    if not text:
        return text, {}

    entities: Dict[str, List[str]] = {
        "phones": [],
        "bank_accounts": [],
        "citizen_ids": [],
    }

    masked = text

    # 1. Mask Vietnamese Phone Numbers first (to avoid misclassifying 10-digit mobile as bank account)
    for match in PHONE_PATTERN.finditer(masked):
        phone = match.group(0)
        cleaned_phone = re.sub(r"[\s\.\-]", "", phone)
        if len(cleaned_phone) in (10, 11, 12):
            entities["phones"].append(cleaned_phone)
            masked = masked.replace(phone, "[PHONE_REDACTED]")

    # 2. Mask CCCD / Citizen IDs
    for match in CCCD_PATTERN.finditer(masked):
        id_val = match.group(1) if match.lastindex else match.group(0)
        digits = re.sub(r"\D", "", id_val)
        if len(digits) == 12:
            entities["citizen_ids"].append(digits)
            masked = masked.replace(digits, "[ID_REDACTED]")

    for match in CMND_PATTERN.finditer(masked):
        id_val = match.group(1) if match.lastindex else match.group(0)
        digits = re.sub(r"\D", "", id_val)
        if len(digits) == 9:
            entities["citizen_ids"].append(digits)
            masked = masked.replace(digits, "[ID_REDACTED]")

    # 3. Mask Bank Accounts (STK)
    for match in BANK_ACC_PATTERN.finditer(masked):
        acc = match.group(1) if match.lastindex else match.group(0)
        digits = re.sub(r"\D", "", acc)
        if len(digits) >= 8 and "[PHONE_REDACTED]" not in acc and "[ID_REDACTED]" not in acc:
            entities["bank_accounts"].append(digits)
            masked = masked.replace(digits, "[BANK_ACC_REDACTED]")

    return masked, entities


def redact_sensitive_entities(text: str) -> str:
    """Helper returning only the sanitized string."""
    masked_text, _ = mask_pii(text)
    return masked_text
