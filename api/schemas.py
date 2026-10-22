"""
==============================================================================
ScamShield-VN: Pydantic Data Contracts for AI Pipeline Microservice
==============================================================================
Complies with docs/04_API_CONTRACTS_AND_SCHEMAS.md for Frontend React & Spring Boot.
==============================================================================
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator


class ScanRequest(BaseModel):
    """Payload sent by Client / Spring Boot Backend to scan an incoming message."""
    content: Optional[str] = Field(None, description="Message text or content to scan")
    message: Optional[str] = Field(None, description="Alternative field for message content")
    sender: Optional[str] = Field(None, description="Phone number, email or brand header")
    check_type: Literal["TEXT", "URL", "PHONE", "BANK_ACCOUNT"] = Field("TEXT", description="Type of verification")
    target_bank: Optional[str] = Field(None, description="Bank code (e.g. VCB, MB, BIDV)")
    target_account_number: Optional[str] = Field(None, description="Bank account number if check_type is BANK_ACCOUNT")
    force_tier2: bool = Field(False, description="Force Tier-2 Council execution regardless of Tier 1 confidence")

    @model_validator(mode="after")
    def validate_content_or_message(self):
        if not self.content and not self.message:
            raise ValueError("Either 'content' or 'message' must be provided.")
        if not self.content and self.message:
            self.content = self.message
        return self


class AgentFinding(BaseModel):
    role: str
    verdict: Literal["SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"]
    score: float
    finding: str


class CouncilBreakdown(BaseModel):
    consensus_ratio: str
    lead_model: str
    early_exit: bool = False
    defender_intervened: bool = False
    agents: List[AgentFinding] = []


class RadarMetrics(BaseModel):
    urgency: int = Field(0, ge=0, le=100)
    authority_impersonation: int = Field(0, ge=0, le=100)
    financial_lure: int = Field(0, ge=0, le=100)
    suspicious_channel: int = Field(0, ge=0, le=100)
    data_harvesting: int = Field(0, ge=0, le=100)


class Verdict(BaseModel):
    risk_level: Literal["SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"]
    risk_score: float = Field(..., ge=0.0, le=100.0)
    scam_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str


class ScanResultData(BaseModel):
    scan_id: str
    timestamp: str
    tier_executed: int = Field(..., description="1: ViSoBERT On-Device/CPU | 2: Cloud 3-Agent Council")
    verdict: Verdict
    radar_metrics: RadarMetrics
    council_breakdown: Optional[CouncilBreakdown] = None
    actionable_recommendations: List[str] = []
    telemetry: Optional[Dict[str, Any]] = None


class StandardScanResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[ScanResultData] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    tier1_ready: bool
    tier2_ready: bool
