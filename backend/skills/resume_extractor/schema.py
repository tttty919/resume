"""Skill 2: ResumeExtractor — Pydantic 输出模型"""

from pydantic import BaseModel, Field
from typing import Literal


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
