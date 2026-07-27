"""Skill 1: JDParser — Pydantic 输出模型"""

from pydantic import BaseModel, Field
from typing import Literal


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
