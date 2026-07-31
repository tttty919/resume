"""Skill 3: SemanticMatcher — 语义匹配 + 证据提取

调用链: requirements + structured_resume + raw_resume_text
       → 硬条件代码判断（秒出）
       → QueryExpander 同义词扩展
       → LLM 判断 + 逐字摘录证据（仅软性要求）
"""

import asyncio as _asyncio
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import SecretStr

from backend.core.exceptions import ParseException
from backend.skills.semantic_matcher.schema import SemanticMatcherOutput
from backend.skills.semantic_matcher.tools import expand
from backend.utils.llm_utils import create_llm, load_prompt, parse_llm_json, safe_pydantic_validate


# ── 硬条件代码判断 ─────────────────────────────────────────

_HARD_CONDITION_NAMES = {"学历", "学位", "工作年限", "经验年限", "毕业院校", "学校层次", "年龄"}

_DEGREE_RANK = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}

_985_KEYWORDS = {"985", "双一流", "C9", "清华", "北大", "浙大", "复旦", "上海交通", "南京大学",
                 "中国科学技术", "哈尔滨工业", "西安交通"}


def _parse_years(text: str) -> float | None:
    """从文本中提取年限数字，如 '5年' → 5.0, '3-5年' → 3.0"""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _is_hard_condition(req: dict) -> bool:
    name = req.get("name", "")
    return any(h in name for h in _HARD_CONDITION_NAMES)


def _match_hard_condition(req: dict, resume: dict, raw_resume_text: str) -> dict | None:
    """尝试用代码判断硬条件，返回 SingleMatch dict 或 None（无法机械判断时）"""
    name = req.get("name", "")
    desc = req.get("description", "")
    basic = resume.get("basic_info", {}) if isinstance(resume, dict) else {}

    result_base = {
        "requirement_id": req.get("id", ""),
        "requirement_name": name,
        "synonyms_used": [],
        "needs_human_review": False,
        "status_changed": False,
    }

    # ── 学历/学位 ──
    if "学历" in name or "学位" in name:
        resume_degree = basic.get("degree")
        if not resume_degree or resume_degree not in _DEGREE_RANK:
            return {**result_base, "status": "cannot_judge", "confidence": 0.9,
                    "reasoning": "简历未明确标注学历/学位",
                    "evidence": "", "evidence_location": "", "evidence_support": "未找到"}

        required_rank = 0
        for deg, rank in _DEGREE_RANK.items():
            if deg in desc or deg in name:
                required_rank = max(required_rank, rank)

        if required_rank == 0:
            return None  # 无法确定要求的学历等级，交给 LLM

        resume_rank = _DEGREE_RANK[resume_degree]
        if resume_rank >= required_rank:
            return {**result_base, "status": "satisfied", "confidence": 0.95,
                    "reasoning": f"简历学历 {resume_degree} 满足要求",
                    "evidence": resume_degree, "evidence_location": "basic_info.degree",
                    "evidence_support": "确凿"}
        else:
            return {**result_base, "status": "not_satisfied", "confidence": 0.95,
                    "reasoning": f"简历学历 {resume_degree} 低于要求",
                    "evidence": resume_degree, "evidence_location": "basic_info.degree",
                    "evidence_support": "确凿"}

    # ── 工作年限/经验年限 ──
    if "年限" in name or "经验" in name:
        resume_years_str = basic.get("work_years", "")
        resume_years = _parse_years(resume_years_str)

        if resume_years is None:
            return {**result_base, "status": "cannot_judge", "confidence": 0.9,
                    "reasoning": "简历未明确标注工作年限",
                    "evidence": "", "evidence_location": "", "evidence_support": "未找到"}

        required_years = _parse_years(desc) or _parse_years(name)
        if required_years is None:
            return None  # 无法提取要求年限，交给 LLM

        if resume_years >= required_years:
            return {**result_base, "status": "satisfied", "confidence": 0.95,
                    "reasoning": f"简历工作年限 {resume_years_str} 满足 {required_years} 年要求",
                    "evidence": resume_years_str, "evidence_location": "basic_info.work_years",
                    "evidence_support": "确凿"}
        else:
            return {**result_base, "status": "not_satisfied", "confidence": 0.95,
                    "reasoning": f"简历工作年限 {resume_years_str} 不满足 {required_years} 年要求",
                    "evidence": resume_years_str, "evidence_location": "basic_info.work_years",
                    "evidence_support": "确凿"}

    # ── 毕业院校/学校层次 ──
    if "院校" in name or "学校" in name:
        school = basic.get("school", "")
        if not school:
            return {**result_base, "status": "cannot_judge", "confidence": 0.9,
                    "reasoning": "简历未标注毕业院校",
                    "evidence": "", "evidence_location": "", "evidence_support": "未找到"}

        need_985 = any(k in desc or k in name for k in ("985", "双一流", "一流大学", "重点大学", "C9"))
        if need_985:
            is_985 = any(k in school for k in _985_KEYWORDS)
            if is_985:
                return {**result_base, "status": "satisfied", "confidence": 0.9,
                        "reasoning": f"毕业院校 {school} 符合要求",
                        "evidence": school, "evidence_location": "basic_info.school",
                        "evidence_support": "确凿"}
            else:
                return {**result_base, "status": "not_satisfied", "confidence": 0.8,
                        "reasoning": f"毕业院校 {school} 未在 985/双一流名单中",
                        "evidence": school, "evidence_location": "basic_info.school",
                        "evidence_support": "部分"}
        return None  # 非 985 类要求，交给 LLM

    return None  # 未匹配到任何硬条件模式

_synonym_cache: dict[str, list[str]] = {}


def _should_query_rag(req: dict) -> bool:
    name = req.get("name", "")
    if any(h in name for h in _HARD_CONDITION_NAMES):
        return False
    keywords = req.get("keywords", [])
    if not keywords:
        return False
    return True


def _build_rag_query(req: dict) -> str:
    parts = [req.get("name", "")]
    parts.extend(req.get("keywords", [])[:3])
    parts.append(req.get("description", "")[:100])
    return " ".join(p for p in parts if p)


def _query_with_cache(req: dict) -> list[str]:
    query = _build_rag_query(req)
    if query in _synonym_cache:
        return _synonym_cache[query]
    result = expand(query)
    _synonym_cache[query] = result
    return result


# ── 主函数 ────────────────────────────────────────────────

async def match(
    requirements: list[dict],
    resume: dict,
    raw_resume_text: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    extra_synonyms: dict[str, list[str]] | None = None,
) -> dict:
    """独立的匹配函数，可从 dev_server 直接调用

    extra_synonyms: 可选。{requirement_id: [同义词...]}，由控制器（ScreeningAgent）
        在动态决策时注入 —— 例如某条要求首轮判不了、控制器主动查了同义词后，
        把结果传进来重新匹配。默认 None 时行为与原来完全一致（零影响）。
    """
    if not isinstance(requirements, list):
        logger.error(f"requirements 类型错误: {type(requirements)}, value={str(requirements)[:100]}")
        return {"matches": []}
    if not isinstance(resume, dict):
        logger.error(f"resume 类型错误: {type(resume)}, value={str(resume)[:100]}")
        return {"matches": []}

    # Step 1: 硬条件代码判断（秒出）
    hard_matches = []
    soft_requirements = []

    for req in requirements:
        if _is_hard_condition(req):
            result = _match_hard_condition(req, resume, raw_resume_text)
            if result is not None:
                hard_matches.append(result)
                logger.debug(f"硬条件 [{req.get('name')}] → {result['status']} (代码判断)")
            else:
                soft_requirements.append(req)
        else:
            soft_requirements.append(req)

    logger.info(f"硬条件代码判断: {len(hard_matches)} 条秒判, {len(soft_requirements)} 条交给 LLM")

    # Step 2: 选择性 RAG 扩展（仅软性要求，并行查询 ChromaDB）
    synonyms_map: dict[str, list[str]] = {}
    rag_hits = 0
    rag_skips = 0

    async def _run_rag(req: dict) -> tuple[str, list[str]] | None:
        if _should_query_rag(req):
            return req.get("id", ""), await _asyncio.to_thread(_query_with_cache, req)
        return None

    rag_tasks = [_run_rag(r) for r in soft_requirements]
    rag_results = await _asyncio.gather(*rag_tasks)

    for result in rag_results:
        if result is not None:
            rid, synonyms = result
            synonyms_map[rid] = synonyms
            rag_hits += 1
        else:
            rag_skips += 1

    logger.info(f"RAG 查询: {rag_hits} 次检索, {rag_skips} 次跳过")

    # Step 2.5: 合并控制器注入的同义词（ScreeningAgent 动态决策时传入，默认无）
    if extra_synonyms:
        for rid, syns in extra_synonyms.items():
            if not syns:
                continue
            merged = list(dict.fromkeys(synonyms_map.get(rid, []) + list(syns)))
            synonyms_map[rid] = merged
            logger.info(f"控制器注入同义词 [{rid}]: +{len(syns)} 条 → 合并后 {len(merged)} 条")

    # Step 3: LLM 匹配 + 证据提取（分批并行，仅软性要求）
    llm_matches = []
    if soft_requirements:
        BATCH_SIZE = 3
        llm = create_llm(api_key=api_key, base_url=base_url, model=model)
        prompt_text = load_prompt("skills/semantic_matcher/prompt.txt")
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm

        resume_json_str = json.dumps(resume, ensure_ascii=False, indent=2)
        raw_text = raw_resume_text or resume_json_str

        batches = [soft_requirements[i:i + BATCH_SIZE] for i in range(0, len(soft_requirements), BATCH_SIZE)]
        logger.info(f"分批并行: {len(soft_requirements)} 条软性要求 → {len(batches)} 批 (每批 {BATCH_SIZE} 条)")

        async def _run_batch(batch_reqs: list[dict]) -> list[dict]:
            batch_synonyms = {r.get("id", ""): synonyms_map.get(r.get("id", ""), []) for r in batch_reqs}
            invoke_args = {
                "requirements_json": json.dumps(batch_reqs, ensure_ascii=False, indent=2),
                "resume_json": resume_json_str,
                "synonyms_json": json.dumps(batch_synonyms, ensure_ascii=False, indent=2),
                "raw_resume_text": raw_text,
            }

            for attempt in (1, 2):
                response = await _asyncio.to_thread(chain.invoke, invoke_args)
                try:
                    parsed = parse_llm_json(response.content)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict):
                        return parsed.get("matches", [])
                    raise ParseException(f"JSON 顶层类型异常: {type(parsed)}", file_type="llm_response")
                except ParseException:
                    if attempt == 1:
                        logger.warning(f"批次解析失败（第1次），{len(batch_reqs)} 条要求，重试一次")
                    else:
                        logger.warning(f"批次解析失败（重试后仍失败），{len(batch_reqs)} 条要求返回空，LLM 完整返回: {response.content}")
            return []

        batch_results = await _asyncio.gather(*[_run_batch(b) for b in batches])

        for batch_matches in batch_results:
            for m in batch_matches:
                if not isinstance(m, dict):
                    continue
                llm_matches.append(m)

    # Step 4: 合并硬条件 + LLM 结果 + 填充默认值 + 强制 confidence 规则
    matches = hard_matches + llm_matches
    matched_ids = set()

    for m in matches:
        m.setdefault("evidence", "")
        m.setdefault("evidence_location", "")
        m.setdefault("evidence_support", "未找到")
        m.setdefault("needs_human_review", False)
        m.setdefault("status_changed", False)
        matched_ids.add(m.get("requirement_id", ""))

        # 强制规则：satisfied + confidence < 0.6 → 改判 cannot_judge
        if m.get("status") == "satisfied" and m.get("confidence", 0) < 0.6:
            m["status"] = "cannot_judge"
            m["evidence_support"] = "信息不足"
            m["needs_human_review"] = True
            logger.info(f"[{m.get('requirement_name', '')}] satisfied 但 confidence={m.get('confidence')} < 0.6 → 强制改判 cannot_judge")

    # 补全缺失的 requirement（LLM 可能漏掉某些条目）
    for req in requirements:
        rid = req.get("id", "")
        if rid and rid not in matched_ids:
            logger.warning(f"[{req.get('name', rid)}] LLM 未返回匹配结果，自动补为 cannot_judge")
            matches.append({
                "requirement_id": rid,
                "requirement_name": req.get("name", rid),
                "status": "cannot_judge",
                "confidence": 0.3,
                "reasoning": "LLM 未返回该要求的匹配结果，自动标记为无法判断",
                "synonyms_used": [],
                "evidence": "",
                "evidence_location": "",
                "evidence_support": "信息不足",
                "needs_human_review": True,
                "status_changed": False,
            })

    # Step 5: 证据来源溯源 —— 区分"简历原文摘录"与"LLM 推理/无实据"，
    # 供评分引擎（scoring_engine.py）判断证据是否可信，避免推理结论被当成已验证证据。
    for m in matches:
        evidence_text = m.get("evidence", "")
        support = m.get("evidence_support", "未找到")
        if evidence_text and support in ("确凿", "部分"):
            source_type = "resume_evidence"
            verified = support == "确凿"
        else:
            source_type = "system_inference"
            verified = False
        m["evidence_sources"] = [{
            "content": evidence_text or m.get("reasoning", ""),
            "source_type": source_type,
            "source_location": m.get("evidence_location", "") or "LLM 推理（未摘录原文）",
            "verified": verified,
        }]

    return safe_pydantic_validate({"matches": matches}, SemanticMatcherOutput, "SemanticMatcher")


async def run(state: dict) -> dict:
    """Skill 3 节点入口（LangGraph 模式）"""
    requirements = state.get("requirements", [])
    resume = state.get("structured_resume", {})
    raw_text = state.get("raw_resume_text", "")

    result = await match(requirements, resume, raw_text)

    traces = state.get("trace_log", [])
    matches = result.get("matches", [])
    satisfied = sum(1 for m in matches if isinstance(m, dict) and m.get("status") == "satisfied")

    traces.append({
        "step": "SemanticMatcher",
        "summary": f"匹配+证据: {satisfied}/{len(matches)} 满足",
        "status": "ok",
    })

    return {**state, "match_results": matches, "validated_items": matches, "trace_log": traces}
