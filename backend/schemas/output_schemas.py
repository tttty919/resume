"""Pydantic 输出模型 —— 所有 Skill 的结构化输出合约

设计原则：
- 每个 Skill 的输出都有独立的 Pydantic 模型
- 模型之间通过引用保持一致性
- 所有字段都有明确的类型和默认值
"""

from pydantic import BaseModel, Field
from typing import Literal


# ── Skill 1: JDParser ────────────────────────────────────────────

class RequirementItem(BaseModel):
    """单条岗位要求"""
    id: str = Field(description="唯一标识，如 req-001")
    name: str = Field(description="要求简短名称，10字以内")
    description: str = Field(description="具体可验证的详细说明")
    type: Literal["must", "bonus"] = Field(description="必须项 / 加分项")
    importance: Literal["high", "medium", "low"] = Field(description="重要程度")
    keywords: list[str] = Field(default_factory=list, description="3-5个关键词语义扩展用")


class JDParserOutput(BaseModel):
    """Skill 1 输出"""
    requirements: list[RequirementItem] = Field(description="结构化要求列表")
    parsing_notes: str = Field(default="", description="解析说明与 HR 需补充的信息")


# ── Skill 2: ResumeExtractor ──────────────────────────────────────

class BasicInfo(BaseModel):
    name: str | None = None
    school: str | None = Field(default=None, description="毕业院校")
    major: str | None = Field(default=None, description="专业")
    education: str | None = None
    degree: Literal["博士", "硕士", "本科", "大专", None] = None
    work_years: str | None = None
    current_role: str | None = None


class WorkExperience(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    responsibilities: list[str] = Field(default_factory=list, description="原文摘录的职责描述")


class ProjectExperience(BaseModel):
    name: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    description: str = ""
    responsibility: str = ""


class ResumeExtractorOutput(BaseModel):
    basic_info: BasicInfo = Field(default_factory=BasicInfo)
    skills: list[str] = Field(default_factory=list)
    work_experiences: list[WorkExperience] = Field(default_factory=list)
    project_experiences: list[ProjectExperience] = Field(default_factory=list)
    extraction_notes: str = Field(default="", description="信息缺失说明")


# ── Skill 3: SemanticMatcher ─────────────────────────────────────

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


class SemanticMatcherOutput(BaseModel):
    matches: list[SingleMatch] = Field(default_factory=list)


# ── Skill 4: EvidenceExtractor ────────────────────────────────────

class ValidationResult(BaseModel):
    verified: bool = False
    match_type: Literal["exact", "fuzzy", "not_found"] = "not_found"
    similarity: float = 0.0


class EvidenceSource(BaseModel):
    """单条证据及其来源标记"""
    content: str = Field(default="", description="证据文本")
    source_type: Literal["resume_evidence", "hr_verified_evidence", "system_inference"] = "resume_evidence"
    source_location: str = Field(default="", description="简历位置 / HR确认方式")
    verified: bool = False


class HROverrideRecord(BaseModel):
    """HR改判审计记录"""
    requirement_id: str
    operator: str = Field(description="HR姓名或ID")
    timestamp: str = Field(description="ISO 8601 时间戳")
    before_status: Literal["satisfied", "not_satisfied", "cannot_judge"]
    after_status: Literal["satisfied", "not_satisfied", "cannot_judge"]
    reason: str = Field(description="修改原因")
    supplementary_evidence: str = Field(default="", description="HR补充的证据信息")


class ValidatedItem(BaseModel):
    requirement_id: str = ""
    requirement_name: str = ""
    status: Literal["satisfied", "not_satisfied", "cannot_judge"] = "cannot_judge"
    evidence: str = Field(default="", description="简历原文摘录（>=15字符）")
    evidence_support: Literal["确凿", "部分", "未找到", "信息不足"] = "未找到"
    evidence_sources: list[EvidenceSource] = Field(default_factory=list, description="带来源标记的证据列表")
    evidence_location: str = Field(default="", description="证据在原文中的位置描述")
    validation: ValidationResult = Field(default_factory=ValidationResult)
    needs_human_review: bool = False
    status_changed: bool = False
    hr_override: HROverrideRecord | None = None


class EvidenceExtractorOutput(BaseModel):
    validated_items: list[ValidatedItem] = Field(default_factory=list)


# ── Skill 5: RiskAnalyzer ────────────────────────────────────────

class RiskAnalysis(BaseModel):
    core_advantages: list[str] = Field(default_factory=list, description="核心优势")
    key_risks: list[str] = Field(default_factory=list, description="关键风险")
    info_gaps: list[str] = Field(default_factory=list, description="信息缺口")
    interview_suggestions: list[str] = Field(default_factory=list, description="建议面试追问的问题")


class RiskAnalyzerOutput(BaseModel):
    analysis: RiskAnalysis = Field(default_factory=RiskAnalysis)


# ── Skill 6: RecommendationGen ────────────────────────────────────

class MatchSummary(BaseModel):
    satisfied_count: int = 0
    not_satisfied_count: int = 0
    cannot_judge_count: int = 0
    core_advantages: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    recommendation: str = ""
    needs_human_review: bool = True
    human_review_questions: list[str] = Field(default_factory=list)


class RecommendationGenOutput(BaseModel):
    summary: MatchSummary = Field(default_factory=MatchSummary)
    recommendation_reason: str = Field(default="", description="推荐理由（LLM润色文本）")
