"""Skill 3: SemanticMatcher（含已合并的 EvidenceExtractor）— Pydantic 输出模型"""

from pydantic import BaseModel, Field
from typing import Literal


class SingleMatch(BaseModel):
    requirement_id: str = ""
    requirement_name: str = ""
    status: Literal["satisfied", "not_satisfied", "cannot_judge"] = "cannot_judge"
    reasoning: str = ""
    synonyms_used: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    evidence: str = Field(default="", description="简历原文摘录（>=15字符）")
    evidence_location: str = Field(default="", description="证据在原文中的位置描述")
    evidence_support: Literal["确凿", "部分", "未找到", "信息不足"] = "未找到"
    evidence_sources: list[dict] = Field(default_factory=list, description="证据来源溯源，含 source_type: resume_evidence/system_inference/hr_verified_evidence")
    needs_human_review: bool = False
    status_changed: bool = False


class SemanticMatcherOutput(BaseModel):
    matches: list[SingleMatch] = Field(default_factory=list)


class HROverrideRecord(BaseModel):
    """HR改判审计记录"""
    requirement_id: str
    operator: str = Field(description="HR姓名或ID")
    timestamp: str = Field(description="ISO 8601 时间戳")
    before_status: Literal["satisfied", "not_satisfied", "cannot_judge"]
    after_status: Literal["satisfied", "not_satisfied", "cannot_judge"]
    reason: str = Field(description="修改原因")
    supplementary_evidence: str = Field(default="", description="HR补充的证据信息")
