"""ScreeningAgent 最小控制器 — 停止条件 + 局部回退

导师评审意见（七）：
- 读取当前任务状态，检查哪些要求还没有可靠结论
- 根据问题类型选择需要调用的 Skill 或 Tool
- HR 补充信息后，只重算受影响的要求、风险和推荐
- 停止条件：全部要求稳定且无待校验证据 / 自动重试达上限 / 必须由HR确认
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.workflows.nodes.risk_analyzer import analyze as analyze_risk
from backend.workflows.nodes.recommendation_gen import generate as generate_recommendation

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
