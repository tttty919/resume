"""工具箱 (第1步) —— ScreeningAgent 可调用的能力清单

把系统里已有的能力（匹配 / 查同义词 / 风险分析 / 推荐生成）和三个新的小工具
（深挖证据 / 校验证据 / 生成HR问题）统一登记成一份"工具清单"。
第2步的控制器 (agent_loop) 会按当前状态从中挑选工具。

设计要点
--------
1. 每个工具 = 名称 + 说明 + 输入 + 可调用实现，粒度小、职责单一。
2. 招聘域红线：analyze_risk / generate_recommendation 【不】进入控制器的可选清单
   （见 CONTROLLER_ACTIONS）—— 它们由控制器在收尾阶段固定调用，防止 LLM 自主
   跳过风控。此处仍登记，供收尾直接使用。
3. ToolBox 用依赖注入：真实运行用本类；离线演示/测试可传入 FakeToolBox 子类，
   无需 API Key 即可验证控制器逻辑。
"""

from __future__ import annotations

import asyncio
import json

from backend.core.logger import get_logger
from backend.skills.semantic_matcher.node import match as _match
from backend.skills.semantic_matcher.tools import expand as _expand
from backend.skills.risk_analyzer.node import analyze as _analyze_risk
from backend.skills.recommendation_gen.node import generate as _generate_rec
from backend.utils.llm_utils import create_llm

log = get_logger()


# ── 控制器可自主选择的动作清单（喂给决策 LLM 的"菜单"）──────────────
# 注意：这里【没有】风险分析和推荐生成 —— 它们不是可选项，收尾时固定跑。
CONTROLLER_ACTIONS = [
    {"tool": "query_synonyms",
     "desc": "为某条要求查同义词。遇到简历里出现、但和要求用词不同的术语时用。"},
    {"tool": "rematch",
     "desc": "带上已查到的同义词，重新匹配某条要求。通常紧跟在 query_synonyms 后。"},
    {"tool": "deep_extract",
     "desc": "深挖整份简历，为某条尚无证据的要求寻找隐含证据。"},
    {"tool": "verify_evidence",
     "desc": "校验某条要求现有证据是否可靠。当某条命中了但置信度偏低时用。"},
    {"tool": "ask_hr_question",
     "desc": "为某条判不了的必须项生成一条精准的HR复核问题，并挂起该条等待HR。"},
    {"tool": "stop",
     "desc": "认为所有要求都已有稳定结论、或继续也无意义时，收尾停止。"},
]


class ToolBox:
    """真实工具箱：包一层现有 Skill + 三个新小工具。"""

    def __init__(self, resume: dict, raw_text: str, llm_cfg: dict):
        self.resume = resume or {}
        self.raw_text = raw_text or ""
        self.cfg = llm_cfg or {}

    # —— LLM 小工具的通用调用器 ——
    async def _ask_json(self, prompt_text: str) -> dict:
        llm = create_llm(
            api_key=self.cfg.get("api_key", ""),
            base_url=self.cfg.get("base_url", ""),
            model=self.cfg.get("model", ""),
        )
        resp = await asyncio.to_thread(llm.invoke, prompt_text)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            log.warning(f"小工具 JSON 解析失败: {e} | 原文前120字: {text[:120]}")
            return {}

    # —— 现有能力：匹配 ——
    async def match_requirements(
        self, reqs: list[dict], extra_synonyms: dict[str, list[str]] | None = None
    ) -> list[dict]:
        result = await _match(
            reqs, self.resume, self.raw_text,
            api_key=self.cfg.get("api_key", ""),
            base_url=self.cfg.get("base_url", ""),
            model=self.cfg.get("model", ""),
            extra_synonyms=extra_synonyms,
        )
        return result.get("matches", [])

    # —— 现有能力：查同义词（SkillMapper / ChromaDB）——
    async def query_synonyms(self, term: str) -> list[str]:
        return await asyncio.to_thread(_expand, term)

    # —— 现有能力：风险分析（收尾固定调用，非控制器可选）——
    async def analyze_risk(self, reqs: list[dict], items: list[dict]) -> dict:
        result = await _analyze_risk(
            reqs, items,
            api_key=self.cfg.get("api_key", ""),
            base_url=self.cfg.get("base_url", ""),
            model=self.cfg.get("model", ""),
        )
        return result.get("analysis", {})

    # —— 现有能力：推荐生成（收尾固定调用，非控制器可选）——
    async def generate_recommendation(
        self, reqs: list[dict], items: list[dict], analysis: dict
    ) -> dict:
        return await _generate_rec(
            reqs, items, None, analysis,
            api_key=self.cfg.get("api_key", ""),
            base_url=self.cfg.get("base_url", ""),
            model=self.cfg.get("model", ""),
        )

    # —— 新工具：深挖证据 ——
    async def deep_extract(self, req: dict) -> dict:
        prompt = (
            "你是招聘筛选助手。请在下面这份简历原文里，仔细寻找与【某条岗位要求】相关的"
            "任何隐含证据（哪怕没用要求里的原词）。只依据原文，不要编造。\n\n"
            f"岗位要求：{req.get('name','')} —— {req.get('description','')}\n\n"
            f"简历原文：\n{self.raw_text[:6000]}\n\n"
            '严格返回 JSON：{"found": true/false, "evidence": "摘录的原文片段(没有则空)", '
            '"location": "大概在简历哪部分", "confidence": 0~1的小数}'
        )
        return await self._ask_json(prompt)

    # —— 新工具：校验证据 ——
    async def verify_evidence(self, req: dict, item: dict) -> dict:
        prompt = (
            "你是招聘筛选的证据审核员。判断下面这条'命中'结论的证据是否可靠、是否"
            "真的能支撑该要求。宁可谨慎。\n\n"
            f"岗位要求：{req.get('name','')} —— {req.get('description','')}\n"
            f"当前结论：{item.get('status','')}，置信度 {item.get('confidence','')}\n"
            f"所依据的证据：{item.get('evidence','(无)')}\n\n"
            '严格返回 JSON：{"reliable": true/false, "confidence": 0~1的小数, '
            '"reason": "一句话说明为什么可靠或不可靠"}'
        )
        return await self._ask_json(prompt)

    # —— 新工具：生成HR复核问题 ——
    async def ask_hr_question(self, req: dict, item: dict) -> dict:
        prompt = (
            "你是招聘筛选助手。系统对下面这条【必须项】无法给出可靠结论，需要HR人工确认。"
            "请生成一条简短、精准、可直接问候选人或HR的复核问题。\n\n"
            f"岗位要求：{req.get('name','')} —— {req.get('description','')}\n"
            f"当前状态：{item.get('status','')}，证据：{item.get('evidence','(无)')}\n\n"
            '严格返回 JSON：{"question": "复核问题"}'
        )
        return await self._ask_json(prompt)
