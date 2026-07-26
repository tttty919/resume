"""Skill 1: JDParser — 将原始 JD 文本解析为结构化岗位要求

调用链：无 Tool 依赖，纯 LLM 推理
条件路由：如果 requirements 已存在且非空 → 跳过本节点
"""

from backend.core.exceptions import ParseException
from backend.schemas.output_schemas import JDParserOutput
from backend.utils.llm_utils import build_chain, parse_llm_json, safe_pydantic_validate


async def run(state: dict) -> dict:
    """Skill 1 节点入口

    State 输入:
        state["raw_jd"]: str            — 原始 JD 文本
        state["requirements"]: list     — 可选，已存在则跳过

    State 输出:
        state["requirements"]: list[dict]  — 结构化要求列表
        state["parsing_notes"]: str        — 解析说明
        state["trace_log"]: list           — 追加一条 trace 记录
    """
    # ── 条件路由：已有 requirements 则跳过 ──
    if state.get("requirements") and len(state["requirements"]) > 0:
        _append_trace(state, "JDParser", "跳过（requirements 已存在）")
        return state

    raw_jd = state.get("raw_jd", "")
    if not raw_jd.strip():
        return {**state, "error": "Skill1 JDParser: raw_jd 为空，无法解析"}

    # ── LLM 调用 ──
    chain = build_chain("jd_parser.txt")
    response = await chain.ainvoke({"raw_jd": raw_jd})

    # ── JSON 解析 ──
    try:
        parsed = parse_llm_json(response.content)
    except ParseException:
        _append_trace(state, "JDParser", "JSON 解析失败", error=True)
        return {**state, "error": "Skill1 JDParser: LLM 返回无法解析为 JSON"}

    # ── Pydantic 校验 ──
    result = safe_pydantic_validate(parsed, JDParserOutput, "JDParser")
    requirements = result.get("requirements", [])

    _append_trace(state, "JDParser",
                  f"提取 {len(requirements)} 条要求: "
                  + ", ".join(r.get("name", "?") for r in requirements))

    return {
        **state,
        "requirements": requirements,
        "parsing_notes": result.get("parsing_notes", ""),
    }


def _append_trace(state: dict, step: str, summary: str, error: bool = False):
    """向 trace_log 追加一条记录"""
    traces = state.get("trace_log", [])
    traces.append({
        "step": step,
        "summary": summary,
        "status": "error" if error else "ok",
    })
    state["trace_log"] = traces
