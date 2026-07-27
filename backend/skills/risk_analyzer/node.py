"""Skill 5: RiskAnalyzer — 综合风险评估

调用链: requirements + validated_items → LLM 分析 → RiskAnalyzerOutput
无外部 Tool 依赖，纯 LLM 推理节点。
"""

import json

from langchain_core.prompts import ChatPromptTemplate

from backend.core.exceptions import ParseException
from backend.skills.risk_analyzer.schema import RiskAnalyzerOutput
from backend.utils.llm_utils import create_llm, load_prompt, parse_llm_json, safe_pydantic_validate


async def analyze(
    requirements: list[dict],
    validated_items: list[dict],
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> dict:
    """独立的风险分析函数，可从 dev_server 直接调用"""
    satisfied = sum(1 for i in validated_items if i.get("status") == "satisfied")
    not_satisfied = sum(1 for i in validated_items if i.get("status") == "not_satisfied")
    cannot_judge = sum(1 for i in validated_items if i.get("status") == "cannot_judge")

    summary_text = (
        f"匹配结果汇总：{len(validated_items)} 条要求中，"
        f"{satisfied} 项满足、{not_satisfied} 项不满足、{cannot_judge} 项无法判断。"
    )

    llm = create_llm(api_key=api_key, base_url=base_url, model=model)
    prompt_text = load_prompt("skills/risk_analyzer/prompt.txt")
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm

    response = await chain.ainvoke({
        "requirements_json": json.dumps(requirements, ensure_ascii=False, indent=2),
        "validated_items_json": json.dumps(validated_items, ensure_ascii=False, indent=2),
        "summary_text": summary_text,
    })

    try:
        parsed = parse_llm_json(response.content)
    except ParseException:
        return {"analysis": {}}

    return safe_pydantic_validate(parsed, RiskAnalyzerOutput, "RiskAnalyzer")


async def run(state: dict) -> dict:
    """Skill 5 节点入口（LangGraph 模式）"""
    requirements = state.get("requirements", [])
    validated_items = state.get("validated_items", [])

    result = await analyze(requirements, validated_items)

    traces = state.get("trace_log", [])
    analysis = result.get("analysis", {})
    risks_count = len(analysis.get("key_risks", []))
    advantages_count = len(analysis.get("core_advantages", []))
    traces.append({
        "step": "RiskAnalyzer",
        "summary": f"风险评估: {advantages_count} 优势, {risks_count} 风险",
        "status": "ok",
    })

    return {**state, "risk_analysis": analysis, "trace_log": traces}
