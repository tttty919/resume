"""ScreeningAgent 最小控制器 — 停止条件 + 局部回退
导师评审意见（七）：
- 读取当前任务状态，检查哪些要求还没有可靠结论
- 根据问题类型选择需要调用的 Skill 或 Tool
- HR 补充信息后，只重算受影响的要求、风险和推荐
- 停止条件：全部要求稳定且无待校验证据 / 自动重试达上限 / 必须由HR确认
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

MAX_AUTO_RETRIES = 2


@dataclass
class ControllerState:
    """ScreeningAgent 运行时状态"""
    total_requirements: int = 0
    retry_count: int = 0
    hr_override_count: int = 0
    stop_reason: str = ""                          # 空 = 未停止
    pending_hr_items: list[str] = field(default_factory=list)  # 待HR确认的 requirement_id

class ScreeningAgent:
    """Minimal controller around the fixed screening workflow.

    Skills still perform the domain work. The controller reads current state and
    chooses a safe next action: retry unresolved requirements, pause for HR, or
    continue to risk analysis and recommendation generation.
    """

    def __init__(self, requirements: list[dict], retry_limit: int = MAX_AUTO_RETRIES):
        self.requirements = requirements or []
        self.retry_limit = retry_limit
        self.state = ControllerState(total_requirements=len(self.requirements))

    def decide_after_match(self, validated_items: list[dict]) -> dict:
        should_stop, stop_reason, pending_hr = check_stop_condition(validated_items, self.state.retry_count)
        unresolved = find_unresolved(self.requirements, validated_items)

        if unresolved and self.state.retry_count < self.retry_limit:
            action = "retry_unresolved"
            reason = f"{len(unresolved)} unresolved requirement(s); retry targeted matching"
        elif pending_hr:
            action = "pause_for_hr"
            reason = stop_reason
        else:
            action = "continue"
            reason = stop_reason or "all requirements stable"

        self.state.stop_reason = reason
        self.state.pending_hr_items = pending_hr
        self.state.last_action = action
        if not hasattr(self.state, "trace"):
            self.state.trace = []
        decision = {
            "action": action,
            "reason": reason,
            "pending_hr": pending_hr,
            "unresolved": unresolved,
            "retry_count": self.state.retry_count,
            "should_stop": should_stop,
        }
        self.state.trace.append({
            "at": now_iso(),
            "action": action,
            "reason": reason,
            "pending_hr": pending_hr,
            "unresolved_ids": [r.get("id", "") for r in unresolved],
            "retry_count": self.state.retry_count,
        })
        return decision

    def mark_retry(self) -> int:
        self.state.retry_count += 1
        return self.state.retry_count

    def decide_after_retry(self, validated_items: list[dict]) -> dict:
        return self.decide_after_match(validated_items)


def check_stop_condition(validated_items: list[dict], retry_count: int = 0) -> tuple[bool, str, list[str]]:
    """判断是否应停止自动处理

    Args:
        validated_items: Skill 4 输出（含 hr_override 字段）
        retry_count: 当前自动重试次数

    Returns:
        (should_stop, stop_reason, pending_hr_items)
    """
    pending_hr: list[str] = []

    for item in validated_items:
        rid = item.get("requirement_id", "")
        status = item.get("status", "cannot_judge")
        confidence = item.get("confidence", 1.0)
        needs_review = item.get("needs_human_review", False)
        evidence = item.get("evidence", "")

        # 必须项 + cannot_judge OR 显式标记需人工 → 发给HR
        if needs_review or (status == "cannot_judge"):
            pending_hr.append(rid)
            continue

        # 证据为空且状态不是 not_satisfied（not_satisfied 可能是无证据的合理结论）
        if status == "satisfied" and not evidence:
            pending_hr.append(rid)

    # 规则1: 存在必须由HR确认的事项 → 停止，等待HR
    if pending_hr:
        return True, f"{len(pending_hr)} 项要求需要HR确认", pending_hr

    # 规则2: 自动重试达上限 → 停止
    if retry_count >= MAX_AUTO_RETRIES:
        return True, f"自动重试已达上限（{MAX_AUTO_RETRIES}次）", []

    # 规则3: 全部要求稳定且无待校验证据 → 正常停止
    return True, "全部要求已有稳定结论", []


def should_continue(state: ControllerState, validated_items: list[dict]) -> bool:
    """HR 操作后是否应继续自动处理"""
    stop, reason, pending = check_stop_condition(validated_items, state.retry_count)
    state.stop_reason = reason
    state.pending_hr_items = pending
    return not stop or reason == "全部要求已有稳定结论"


async def rerun_affected(
    requirements: list[dict],
    matches: list[dict],
    validated_items: list[dict],
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> dict:
    """HR 修改后，只重跑 Skill 5 (RiskAnalyzer) + Skill 6 (RecommendationGen)

    导师评审意见：不需要重跑 Skill 3/4 —— HR 已直接给出结论。
    只重算受影响的要求、风险和推荐。
    """
    from backend.skills.risk_analyzer.node import analyze as analyze_risk
    from backend.skills.recommendation_gen.node import generate as generate_recommendation

    risk_result = await analyze_risk(requirements, validated_items, api_key, base_url, model)
    analysis = risk_result.get("analysis", {})

    rec_result = await generate_recommendation(
        requirements, matches, validated_items, analysis, api_key, base_url, model
    )

    return {
        "risk_analysis": analysis,
        "recommendation": rec_result,
    }


def find_unresolved(requirements: list[dict], validated_items: list[dict]) -> list[dict]:
    """找出尚未获得可靠结论的 must 要求和需要重算的要求

    Returns:
        需要处理的 requirement 列表
    """
    # 构建已验证项索引
    validated_map: dict[str, dict] = {}
    for item in validated_items:
        rid = item.get("requirement_id", "")
        if rid:
            validated_map[rid] = item

    unresolved = []
    for req in requirements:
        rid = req.get("id", "")
        existing = validated_map.get(rid)

        # 从未处理过
        if not existing:
            unresolved.append(req)
            continue

        status = existing.get("status", "cannot_judge")

        # 已经满足 → 不需要重算
        if status == "satisfied":
            continue

        # 明确不满足 → 不需要重算（这是确定性的结论）
        if status == "not_satisfied":
            continue

        # cannot_judge + must 类型 → 需要重新审视
        if status == "cannot_judge" and req.get("type") == "must":
            unresolved.append(req)
            continue

    return unresolved


def now_iso() -> str:
    """返回 ISO 8601 时间戳（UTC）"""
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════
#  第2步：LLM 驱动的自主控制循环 agent_loop
#
#  和上面固定 3 动作的 ScreeningAgent 不同，这里每一轮都让 LLM 看当前状态、
#  自主选择下一个工具（查同义词 / 重匹配 / 深挖 / 校验 / 提问 / 收尾）。
#
#  两道招聘域红线（写死在代码里，不交给 LLM）：
#   1) 硬兜底：循环次数 MAX_ITERATIONS、每条要求动作次数 MAX_ATTEMPTS_PER_REQ 到顶
#      就强制收尾，防止空转烧钱。
#   2) 风险分析 + 推荐生成【不】在 LLM 可选清单里，收尾阶段固定执行，不许跳过。
#
#  依赖注入：toolbox / decide 可替换为假实现，便于离线演示与测试（见 demo_agent_loop.py）。
# ══════════════════════════════════════════════════════════════════════

import json as _json
import inspect as _inspect

MAX_ITERATIONS = 3
MAX_ATTEMPTS_PER_REQ = 1
MIN_CONFIDENCE = 0.6

# 消耗"再努力"配额的动作（受单条上限限制，防止死缠一条）。
# ask_hr_question 不在此列 —— 它是"交给HR"的收尾出口，任何时候都应允许。
_RETRY_ACTIONS = {"query_synonyms", "rematch", "deep_extract", "verify_evidence"}


def _init_agent_state(requirements: list[dict]) -> dict:
    reqs = {r.get("id", ""): r for r in requirements if r.get("id")}
    return {
        "requirements": reqs,
        "items": {},                 # requirement_id -> 最新 match item
        "extra_synonyms": {},        # requirement_id -> [同义词]
        "attempts": {rid: 0 for rid in reqs},
        "pending_hr": [],
        "hr_questions": {},
        "iteration": 0,
        "stop_reason": "",
        "trace": [],
    }


def _apply_matches(state: dict, items: list[dict]) -> None:
    for m in items:
        rid = m.get("requirement_id", "")
        if rid:
            state["items"][rid] = m


def _trace(state: dict, tool: str, rid: str, reason: str, detail: str = "") -> None:
    entry = {
        "at": now_iso(),
        "iteration": state["iteration"],
        "tool": tool,
        "requirement_id": rid,
        "reason": reason,
        "detail": detail,
    }
    state["trace"].append(entry)
    cb = state.get("_on_step")
    if cb:
        try:
            cb(entry)
        except Exception:
            pass  # 推送失败不影响主流程


def compact_state(state: dict) -> list[dict]:
    """把当前状态压缩成决策 LLM 能一眼看懂的清单。"""
    rows = []
    for rid, req in state["requirements"].items():
        item = state["items"].get(rid, {})
        rows.append({
            "id": rid,
            "name": req.get("name", ""),
            "type": req.get("type", "nice"),
            "status": item.get("status", "未处理"),
            "confidence": item.get("confidence", 0),
            "has_evidence": bool(item.get("evidence")),
            "attempts": state["attempts"].get(rid, 0),
            "synonyms_found": len(state["extra_synonyms"].get(rid, [])),
        })
    return rows


def _tools_done_for(state: dict, rid: str) -> set:
    return {t["tool"] for t in state["trace"] if t.get("requirement_id") == rid}


def rule_decide(state: dict, llm_cfg: dict | None = None) -> dict:
    """确定性决策器（不调 LLM）。

    用途：离线演示 / 单元测试，也可作为 LLM 决策失败时的安全兜底。
    逻辑就是把"遇到判不了先查同义词→重匹配→深挖→仍不行才问HR"写成规则。
    """
    for rid, req in state["requirements"].items():
        item = state["items"].get(rid, {})
        status = item.get("status", "未处理")
        conf = item.get("confidence", 0)
        done = _tools_done_for(state, rid)

        # 已稳定 / 确定性不满足 → 跳过
        if status == "satisfied" and conf >= MIN_CONFIDENCE:
            continue
        if status == "not_satisfied":
            continue

        # 命中但置信度低 → 先校验，再试深挖
        if status == "satisfied" and conf < MIN_CONFIDENCE:
            if "verify_evidence" not in done:
                return {"tool": "verify_evidence", "requirement_id": rid, "reason": "命中但置信度偏低，先核实证据"}
            if "deep_extract" not in done:
                return {"tool": "deep_extract", "requirement_id": rid, "reason": "证据偏弱，深挖简历找佐证"}
            continue

        # 判不了 → 同义词 → 重匹配 → 深挖 → 问HR
        if "query_synonyms" not in done:
            return {"tool": "query_synonyms", "requirement_id": rid, "reason": "用词可能不同，先查同义词"}
        if state["extra_synonyms"].get(rid) and "rematch" not in done:
            return {"tool": "rematch", "requirement_id": rid, "reason": "带同义词重新匹配"}
        if "deep_extract" not in done:
            return {"tool": "deep_extract", "requirement_id": rid, "reason": "同义词无果，深挖简历"}
        if req.get("type") == "must" and "ask_hr_question" not in done:
            return {"tool": "ask_hr_question", "requirement_id": rid, "reason": "补不到证据，转HR复核"}

    return {"tool": "stop", "requirement_id": "", "reason": "所有要求已有稳定结论或无法再自动处理"}


async def llm_decide(state: dict, llm_cfg: dict) -> dict:
    """默认决策器：让 LLM 看状态 + 动作菜单，返回下一步 {tool, requirement_id, reason}。"""
    from backend.workflows.tools_registry import CONTROLLER_ACTIONS
    from backend.utils.llm_utils import create_llm

    prompt = (
        "你是招聘筛选的控制器。下面是当前每条岗位要求的处理状态，请选择下一个要执行的动作。\n"
        "目标：让每条【必须(must)】项都得到可靠结论（satisfied/not_satisfied，置信度≥0.6），"
        "判不了的必须项要生成HR复核问题。已经稳定的要求不要再折腾。\n\n"
        f"当前状态：\n{_json.dumps(compact_state(state), ensure_ascii=False, indent=2)}\n\n"
        f"可选动作：\n{_json.dumps(CONTROLLER_ACTIONS, ensure_ascii=False, indent=2)}\n\n"
        "规则：\n"
        "- 遇到 status=cannot_judge 或 confidence 偏低的必须项，才需要动作；先 query_synonyms/deep_extract 找补，仍不行再 ask_hr_question。\n"
        "- 命中但置信度低(<0.6)时用 verify_evidence 复核。\n"
        "- 所有必须项都稳定，或剩下的判不了也没法补时，选 stop。\n"
        '严格返回 JSON：{"tool": "动作名", "requirement_id": "针对哪条(stop可空)", "reason": "一句话理由"}'
    )
    try:
        llm = create_llm(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", ""),
            model=llm_cfg.get("model", ""),
        )
        import asyncio as _a
        resp = await _a.to_thread(llm.invoke, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = _json.loads(text)
        if isinstance(parsed, dict) and parsed.get("tool"):
            return parsed
    except Exception:
        pass
    # LLM 调用或解析失败 → 退回规则决策兜底（而非提前 stop），
    # 避免一次网络抖动/格式错误就让某个候选人没查够即收尾、必须项漏查。
    return rule_decide(state, llm_cfg)


async def _run_action(toolbox, state: dict, tool: str, rid: str, reason: str) -> None:
    """执行控制器选中的一个动作，并把结果写回 state。"""
    req = state["requirements"].get(rid, {})

    if tool == "query_synonyms":
        term = req.get("name", "") + " " + " ".join(req.get("keywords", [])[:3])
        syns = await toolbox.query_synonyms(term.strip())
        merged = list(dict.fromkeys(state["extra_synonyms"].get(rid, []) + list(syns)))
        state["extra_synonyms"][rid] = merged
        _trace(state, tool, rid, reason, f"查到 {len(syns)} 个同义词")

    elif tool == "rematch":
        items = await toolbox.match_requirements(
            [req], extra_synonyms={rid: state["extra_synonyms"].get(rid, [])}
        )
        _apply_matches(state, items)
        new = state["items"].get(rid, {})
        _trace(state, tool, rid, reason, f"重匹配 → {new.get('status')} ({new.get('confidence')})")

    elif tool == "deep_extract":
        res = await toolbox.deep_extract(req)
        if res.get("found") and res.get("evidence"):
            item = state["items"].setdefault(rid, {"requirement_id": rid,
                                                   "requirement_name": req.get("name", "")})
            item["evidence"] = res["evidence"]
            item["evidence_location"] = res.get("location", "")
            if item.get("status") in ("cannot_judge", "未处理", None) and res.get("confidence", 0) >= MIN_CONFIDENCE:
                item["status"] = "satisfied"
                item["confidence"] = res.get("confidence", MIN_CONFIDENCE)
        _trace(state, tool, rid, reason, f"深挖 found={res.get('found')}")

    elif tool == "verify_evidence":
        item = state["items"].get(rid, {})
        res = await toolbox.verify_evidence(req, item)
        if not res.get("reliable", True):
            item["confidence"] = min(item.get("confidence", 0), res.get("confidence", 0.4))
            item["needs_human_review"] = True
        _trace(state, tool, rid, reason, f"校验 reliable={res.get('reliable')}")

    elif tool == "ask_hr_question":
        item = state["items"].get(rid, {})
        res = await toolbox.ask_hr_question(req, item)
        q = res.get("question", "")
        if q:
            state["hr_questions"][rid] = q
        if rid not in state["pending_hr"]:
            state["pending_hr"].append(rid)
        _trace(state, tool, rid, reason, "已生成HR复核问题并挂起该条")

    else:
        _trace(state, tool, rid, reason, "未知动作，忽略")


def _mark_unresolved_must(state: dict) -> None:
    """收尾前：仍无可靠结论的必须项，统一标记为待HR确认。"""
    for rid, req in state["requirements"].items():
        if req.get("type") != "must":
            continue
        item = state["items"].get(rid, {})
        status = item.get("status", "cannot_judge")
        conf = item.get("confidence", 0)
        unstable = status in ("cannot_judge", "未处理", None) or (status == "satisfied" and conf < MIN_CONFIDENCE)
        if unstable and rid not in state["pending_hr"]:
            state["pending_hr"].append(rid)


async def agent_loop(
    requirements: list[dict],
    resume: dict,
    raw_text: str,
    llm_cfg: dict | None = None,
    *,
    toolbox=None,
    decide=None,
    on_step=None,
) -> dict:
    """LLM 驱动的自主筛选控制循环（第2步骨架）。

    on_step: 可选回调 (entry: dict) -> None，每产生一条决策/动作就调用一次，
             供 router 实时通过 SSE 推给前端。异常会被吞掉，不影响主流程。

    返回结构与原固定流水线兼容：{matches, analysis, scoring, ...}，
    这样第3步接进 router 时改动最小。
    """
    llm_cfg = llm_cfg or {}
    if toolbox is None:
        from backend.workflows.tools_registry import ToolBox
        toolbox = ToolBox(resume, raw_text, llm_cfg)
    decide = decide or llm_decide

    state = _init_agent_state(requirements)
    if on_step:
        state["_on_step"] = on_step

    # 引导：先整体匹配一次（"匹配"是每次都要做的基线，不由 LLM 决定要不要做）
    boot_items = await toolbox.match_requirements(requirements)
    _apply_matches(state, boot_items)
    _trace(state, "match_requirements", "*", "引导：首轮整体匹配", f"{len(boot_items)} 条")

    # 自主决策循环（带硬兜底）
    while state["iteration"] < MAX_ITERATIONS:
        state["iteration"] += 1
        decision = decide(state, llm_cfg)
        if _inspect.isawaitable(decision):
            decision = await decision
        tool = decision.get("tool", "stop")
        rid = decision.get("requirement_id", "") or ""
        reason = decision.get("reason", "")

        if tool == "stop":
            state["stop_reason"] = "控制器判断处理充分，收尾"
            break

        # 硬兜底：单条"再努力"动作次数到顶 → 拒绝该动作，避免死缠一条
        if tool in _RETRY_ACTIONS and rid:
            if state["attempts"].get(rid, 0) >= MAX_ATTEMPTS_PER_REQ:
                _trace(state, tool, rid, reason, f"已达单条动作上限({MAX_ATTEMPTS_PER_REQ})，跳过")
                # 达上限仍判不了的必须项，转人工
                req = state["requirements"].get(rid, {})
                if req.get("type") == "must" and rid not in state["pending_hr"]:
                    state["pending_hr"].append(rid)
                continue
            state["attempts"][rid] = state["attempts"].get(rid, 0) + 1

        await _run_action(toolbox, state, tool, rid, reason)
    else:
        state["stop_reason"] = f"达到最大循环次数({MAX_ITERATIONS})，强制收尾"

    # ── 收尾（固定执行，红线：风控不许跳过）──
    _mark_unresolved_must(state)
    items_list = list(state["items"].values())
    analysis = await toolbox.analyze_risk(list(state["requirements"].values()), items_list)
    rec = await toolbox.generate_recommendation(list(state["requirements"].values()), items_list, analysis)

    return {
        "matches": items_list,
        "analysis": analysis,
        "scoring": rec.get("scoring", {}),
        "summary": rec.get("summary", {}),
        "recommendation_reason": rec.get("recommendation_reason", ""),
        "pending_hr": state["pending_hr"],
        "hr_questions": state["hr_questions"],
        "stop_reason": state["stop_reason"],
        "trace": state["trace"],
        "iterations_used": state["iteration"],
    }
