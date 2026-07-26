"""ScreeningState — LangGraph 全局状态定义

所有 6 个 Skill 节点共享同一个 TypedDict 的状态空间。
HR 复核后可以从断点重新注入，局部重跑下游节点。
"""

from typing import TypedDict, Any


class ScreeningState(TypedDict, total=False):
    # === 输入 ===
    job_id: str
    job_title: str
    raw_jd: str                              # 原始 JD 文本
    requirements: list[dict[str, Any]]        # 结构化要求（Skill 1 输出）
    resume_file_path: str                     # 简历文件路径
    raw_resume_text: str                      # 简历纯文本（文档解析输出）

    # === 中间状态 ===
    structured_resume: dict[str, Any]         # Skill 2 输出
    match_results: list[dict[str, Any]]       # Skill 3 输出
    validated_items: list[dict[str, Any]]     # Skill 4 输出（含证据校验结果）
    risk_analysis: dict[str, Any]             # Skill 5 输出
    scored_items: list[dict[str, Any]]        # 评分引擎输出的结果

    # === HR 复核状态 ===
    hr_decisions: dict[str, str]              # {req_id: "confirm"|"deny"|"follow"}
    hr_overrides: list[dict[str, Any]]        # HROverrideRecord 审计记录列表
    needs_rerun: bool                         # 是否需要局部重跑

    # === 最终输出 ===
    final_result: dict[str, Any]              # Skill 6 输出

    # === 元数据 ===
    parsing_notes: str                        # JD 解析备注
    trace_log: list[dict[str, Any]]           # 全链路追踪
    current_step: str                         # 当前所在节点
    error: str | None                         # 错误信息
