"""Skill 5: RecommendationGen — 最终决策建议生成

调用链: requirements + validated_items + risk_analysis → 评分引擎 → LLM 润色
核心原则: 评分引擎（纯规则）算分定级 → LLM 只做文本润色，不修改推荐等级。

analyze_and_recommend(): 合并风险分析+推荐生成为一个 LLM 调用，省一次 API 往返。
"""

import json

from langchain_core.prompts import ChatPromptTemplate

from backend.core.exceptions import ParseException
from backend.skills.recommendation_gen.schema import RecommendationGenOutput
from backend.utils.llm_utils import create_llm, load_prompt, parse_llm_json, safe_pydantic_validate
from backend.skills.recommendation_gen.scoring_engine import score_matches


async def generate(
    requirements: list[dict],
    matches: list[dict],
    validated_items: list[dict] | None = None,
    risk_analysis: dict | None = None,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> dict:
    """生成最终推荐，可从 dev_server 直接调用。

    流程: 规则算分 → LLM 生成推荐文本。
    validated_items 可选（未提供时使用 matches）；后续可接入独立的证据校验步骤。
    """
    if validated_items is None:
        validated_items = matches
    if risk_analysis is None:
        risk_analysis = {}

    # Step 1: 评分引擎 — 纯规则算分定级
    scoring = score_matches(matches, validated_items, requirements)

    # Step 2: LLM — 文本润色
    llm = create_llm(api_key=api_key, base_url=base_url, model=model)
    prompt_text = load_prompt("skills/recommendation_gen/prompt.txt")
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm

    response = await chain.ainvoke({
        "requirements_json": json.dumps(requirements, ensure_ascii=False, indent=2),
        "validated_items_json": json.dumps(validated_items, ensure_ascii=False, indent=2),
        "risk_analysis_json": json.dumps(risk_analysis, ensure_ascii=False, indent=2),
        "scoring_json": json.dumps(scoring, ensure_ascii=False, indent=2),
    })

    try:
        parsed = parse_llm_json(response.content)
    except ParseException:
        result = {"summary": {"recommendation": scoring["tier_label"]}, "recommendation_reason": ""}
        result["scoring"] = scoring
        return result

    result = safe_pydantic_validate(parsed, RecommendationGenOutput, "RecommendationGen")
    # 推荐等级由评分引擎（硬规则）定级，LLM 只负责文本润色 —— 强制覆写，
    # 防止 LLM 输出的 summary.recommendation 与 scoring.tier_label 不一致。
    if not isinstance(result.get("summary"), dict):
        result["summary"] = {}
    result["summary"]["recommendation"] = scoring["tier_label"]
    result["scoring"] = scoring
    return result


async def analyze_and_recommend(
    requirements: list[dict],
    matches: list[dict],
    validated_items: list[dict] | None = None,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> dict:
    """合并风险分析 + 推荐生成为一个 LLM 调用，省一次 API 往返。

    流程: 规则算分（无 LLM）→ 单次 LLM 调用同时输出 analysis + summary + recommendation_reason。

    Returns:
        {"analysis": {...}, "scoring": {...}, "summary": {...}, "recommendation_reason": "..."}
    """
    if validated_items is None:
        validated_items = matches

    # Step 1: 评分引擎 — 纯规则（<1ms）
    scoring = score_matches(matches, validated_items, requirements)

    # Step 2: 单次 LLM 调用 — 风险分析 + 推荐总结
    llm = create_llm(api_key=api_key, base_url=base_url, model=model)
    prompt_text = load_prompt("skills/recommendation_gen/merged_prompt.txt")
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm

    response = await chain.ainvoke({
        "requirements_json": json.dumps(requirements, ensure_ascii=False, indent=2),
        "validated_items_json": json.dumps(validated_items, ensure_ascii=False, indent=2),
        "scoring_json": json.dumps(scoring, ensure_ascii=False, indent=2),
    })

    try:
        parsed = parse_llm_json(response.content)
    except ParseException:
        # 降级：用规则评分兜底
        return {
            "analysis": {},
            "scoring": scoring,
            "summary": {"recommendation": scoring["tier_label"]},
            "recommendation_reason": "",
        }

    # 强制覆写推荐等级（规则引擎定级，LLM 不可修改）
    if not isinstance(parsed.get("summary"), dict):
        parsed["summary"] = {}
    parsed["summary"]["recommendation"] = scoring["tier_label"]
    parsed["scoring"] = scoring
    return parsed


async def run(state: dict) -> dict:
    """Skill 6 节点入口（LangGraph 模式）"""
    requirements = state.get("requirements", [])
    matches = state.get("match_results", [])
    validated_items = state.get("validated_items", [])
    risk_analysis = state.get("risk_analysis", {})

    result = await generate(requirements, matches, validated_items, risk_analysis)

    traces = state.get("trace_log", [])
    s = result.get("summary", {})
    traces.append({
        "step": "RecommendationGen",
        "summary": f"推荐: {s.get('recommendation', '?')} (得分: {result.get('scoring', {}).get('overall_score', '?')})",
        "status": "ok",
    })

    return {**state, "recommendation": result, "trace_log": traces}
