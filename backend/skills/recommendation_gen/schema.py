"""Skill 5: RecommendationGen — Pydantic 输出模型"""

from pydantic import BaseModel, Field


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
