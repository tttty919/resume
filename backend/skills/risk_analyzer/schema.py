"""Skill 4: RiskAnalyzer — Pydantic 输出模型"""

from pydantic import BaseModel, Field


class RiskAnalysis(BaseModel):
    core_advantages: list[str] = Field(default_factory=list, description="核心优势")
    key_risks: list[str] = Field(default_factory=list, description="关键风险")
    info_gaps: list[str] = Field(default_factory=list, description="信息缺口")
    interview_suggestions: list[str] = Field(default_factory=list, description="建议面试追问的问题")
    source_type: str = Field(default="system_inference", description="整体内容性质：LLM 综合推理产出，非简历原文证据，供报告/HR复核区分证据来源")


class RiskAnalyzerOutput(BaseModel):
    analysis: RiskAnalysis = Field(default_factory=RiskAnalysis)
