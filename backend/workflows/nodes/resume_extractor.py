"""Skill 2: ResumeExtractor — 将简历纯文本解析为结构化数据

调用链: DocumentParser (文档解析) → Skill 2 (ResumeExtractor)
条件路由: 如果 structured_resume 已存在 → 跳过
"""

from backend.core.exceptions import ParseException
from backend.schemas.output_schemas import ResumeExtractorOutput
from backend.utils.llm_utils import build_chain, parse_llm_json, safe_pydantic_validate


async def run(state: dict) -> dict:
    """Skill 2 节点入口

    State 输入:
        state["raw_resume_text"]: str     — 简历纯文本（文档解析输出）
        state["structured_resume"]: dict  — 可选，已存在则跳过

    State 输出:
        state["structured_resume"]: dict  — 结构化简历
        state["trace_log"]: list          — 追加 trace
    """
    if state.get("structured_resume") and state["structured_resume"].get("basic_info"):
        _append_trace(state, "ResumeExtractor", "跳过（structured_resume 已存在）")
        return state

    raw_text = state.get("raw_resume_text", "")
    if not raw_text.strip():
        return {**state, "error": "Skill2 ResumeExtractor: raw_resume_text 为空"}

    chain = build_chain("resume_extractor.txt")
    response = await chain.ainvoke({"raw_resume": raw_text})

    try:
        parsed = parse_llm_json(response.content)
    except ParseException:
        _append_trace(state, "ResumeExtractor", "JSON 解析失败", error=True)
        return {**state, "error": "Skill2 ResumeExtractor: LLM 返回无法解析为 JSON"}

    result = safe_pydantic_validate(parsed, ResumeExtractorOutput, "ResumeExtractor")

    skills_count = len(result.get("skills", []))
    name = result.get("basic_info", {}).get("name", "未知")
    _append_trace(state, "ResumeExtractor",
                  f"提取 {name} 的简历: {skills_count} 项技能, "
                  f"{len(result.get('work_experiences', []))} 段工作经历, "
                  f"{len(result.get('project_experiences', []))} 个项目")

    return {**state, "structured_resume": result}


def _append_trace(state: dict, step: str, summary: str, error: bool = False):
    traces = state.get("trace_log", [])
    traces.append({"step": step, "summary": summary, "status": "error" if error else "ok"})
    state["trace_log"] = traces
