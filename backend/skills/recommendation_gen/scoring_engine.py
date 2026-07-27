"""ScoringEngine — 综合评分引擎（纯规则，不调 LLM）

被 RecommendationGen (Skill 5) 调用。

## 设计原则

三个独立维度，各管各的：
- **匹配结论** = 要求状态 × 要求权重 → 决定推荐等级（硬规则，不可被抵消）
- **置信度评估** = 判断确定性 → 只决定是否人工复核，不参与分数计算
- **证据质量** = 证据是否来自原文 → 标记幻觉，不参与分数计算

核心规则：
1. 必须项 not_satisfied → 暂不推荐（不管置信度多高多低）
2. 必须项 cannot_judge → 需人工复核
3. 证据校验失败 → 该条结论不可用，降级为 cannot_judge
4. AI 只生成建议，最终决策由 HR 确认
"""


def score_matches(
    matches: list[dict],
    validated_items: list[dict],
    requirements: list[dict],
) -> dict:
    """计算综合评分，三组独立输出

    Args:
        matches: Skill 3 匹配结果 [{requirement_id, status, confidence, ...}]
        validated_items: Skill 4 证据校验结果 [{requirement_id, evidence_support, validation, status_changed, ...}]
        requirements: Skill 1 原始要求 [{id, type: must/bonus, importance: high/medium/low, ...}]

    Returns:
        dict 结构见下方注释
    """
    if not matches:
        return _empty_result()

    # 用 id 建主索引；同时用 name 建兜底索引：语义匹配偶尔会返回与原始要求
    # 对不上的 requirement_id（LLM 复述偏差 / id 丢失），只按 id 查会让全部要求
    # 落回默认的 "bonus"，导致「必须 0/0 · 加分 N」这类错误汇总。按名字兜底找回类型。
    req_map = {}
    req_by_name = {}
    for r in requirements:
        rid = r.get("id", "")
        if rid:
            req_map[rid] = r
        name = (r.get("name") or "").strip()
        if name and name not in req_by_name:
            req_by_name[name] = r

    def _resolve_req(m: dict) -> dict:
        rid = m.get("requirement_id", "")
        if rid and rid in req_map:
            return req_map[rid]
        name = (m.get("requirement_name") or "").strip()
        if name and name in req_by_name:
            return req_by_name[name]
        return {}

    importance_weight = {"high": 3, "medium": 2, "low": 1}

    # ═══════════════════════════════════════════════════════════════
    # 一、匹配结论 —— 影响推荐等级（与置信度无关）
    # ═══════════════════════════════════════════════════════════════

    must_items = []   # {id, name, status, importance}
    bonus_items = []  # {id, name, status, importance}

    total_weight = 0
    satisfied_weight = 0

    for m in matches:
        rid = m.get("requirement_id", "")
        req = _resolve_req(m)
        req_type = req.get("type", "bonus")
        importance = req.get("importance", "medium")
        status = m.get("status", "cannot_judge")

        w = importance_weight.get(importance, 2)
        if req_type == "must":
            w *= 2

        total_weight += w
        if status == "satisfied":
            satisfied_weight += w

        item_info = {
            "id": rid,
            "name": req.get("name", rid),
            "status": status,
            "importance": importance,
        }
        if req_type == "must":
            must_items.append(item_info)
        else:
            bonus_items.append(item_info)

    weighted_satisfaction = satisfied_weight / max(total_weight, 1)

    must_satisfied = [i for i in must_items if i["status"] == "satisfied"]
    must_not_satisfied = [i for i in must_items if i["status"] == "not_satisfied"]
    must_cannot_judge = [i for i in must_items if i["status"] == "cannot_judge"]

    bonus_satisfied = [i for i in bonus_items if i["status"] == "satisfied"]

    # ── 硬规则定级（置信度不参与）──
    if must_not_satisfied:
        tier = "not_recommend"
        tier_label = "暂不推荐"
    elif must_cannot_judge:
        tier = "needs_review"
        tier_label = "需人工复核"
    elif weighted_satisfaction >= 0.70:
        tier = "recommend"
        tier_label = "推荐面试"
    elif weighted_satisfaction >= 0.45:
        tier = "consider"
        tier_label = "可考虑"
    else:
        tier = "not_recommend"
        tier_label = "暂不推荐"

    # ═══════════════════════════════════════════════════════════════
    # 二、置信度评估 —— 只决定是否人工复核
    # ═══════════════════════════════════════════════════════════════

    confidences = [m.get("confidence", 0.5) for m in matches]
    avg_confidence = sum(confidences) / max(len(confidences), 1)

    low_confidence_items = [
        {
            "requirement_id": m.get("requirement_id", ""),
            "requirement_name": m.get("requirement_name", ""),
            "status": m.get("status", "cannot_judge"),
            "confidence": m.get("confidence", 0.5),
        }
        for m in matches
        if m.get("confidence", 0.5) < 0.5
    ]

    human_review_reasons = []

    must_low_conf = [
        i for i in must_items
        if any(l["requirement_id"] == i["id"] for l in low_confidence_items)
    ]
    if must_low_conf:
        human_review_reasons.append(
            f"{len(must_low_conf)} 项必须项置信度不足: "
            + ", ".join(i["name"] for i in must_low_conf)
        )

    if must_cannot_judge:
        human_review_reasons.append(
            f"{len(must_cannot_judge)} 项必须项无法判断: "
            + ", ".join(i["name"] for i in must_cannot_judge)
        )

    evidence_uncertain = [
        item for item in validated_items
        if item.get("evidence_support") in ("部分", "未找到")
        and item.get("status") == "satisfied"
    ]
    if evidence_uncertain:
        human_review_reasons.append(
            f"{len(evidence_uncertain)} 项证据不完整需HR确认"
        )

    needs_human_review = bool(human_review_reasons)

    # ═══════════════════════════════════════════════════════════════
    # 三、证据质量 —— 独立于匹配结论和置信度
    # ═══════════════════════════════════════════════════════════════

    total_evidence_items = 0
    verified_items = 0
    hallucinated_items = []

    for item in validated_items:
        if item.get("status") == "satisfied":
            total_evidence_items += 1
            sources = item.get("evidence_sources", [])
            if sources:
                is_verified = any(
                    s.get("verified") and s.get("source_type") != "system_inference"
                    for s in sources
                )
            else:
                is_verified = item.get("validation", {}).get("verified", False)

            if is_verified:
                verified_items += 1

        if item.get("status_changed"):
            hallucinated_items.append({
                "requirement_id": item.get("requirement_id", ""),
                "requirement_name": item.get("requirement_name", ""),
                "evidence_support": item.get("evidence_support", "未找到"),
            })

    evidence_rate = (
        verified_items / max(total_evidence_items, 1)
        if total_evidence_items > 0
        else 0.0
    )

    # ═══════════════════════════════════════════════════════════════
    # 汇总输出
    # ═══════════════════════════════════════════════════════════════

    return {
        "match_conclusion": {
            "tier": tier,
            "tier_label": tier_label,
            "weighted_satisfaction": round(weighted_satisfaction, 4),
            "must_summary": {
                "total": len(must_items),
                "satisfied": len(must_satisfied),
                "not_satisfied": len(must_not_satisfied),
                "cannot_judge": len(must_cannot_judge),
            },
            "bonus_summary": {
                "total": len(bonus_items),
                "satisfied": len(bonus_satisfied),
            },
            "hard_rule_triggered": bool(must_not_satisfied or must_cannot_judge),
        },
        "confidence_assessment": {
            "avg_confidence": round(avg_confidence, 4),
            "low_confidence_items": low_confidence_items,
            "needs_human_review": needs_human_review,
            "human_review_reasons": human_review_reasons,
        },
        "evidence_quality": {
            "verified_rate": round(evidence_rate, 4),
            "verified_count": verified_items,
            "total_evidence_count": total_evidence_items,
            "hallucinated_count": len(hallucinated_items),
            "hallucinated_items": hallucinated_items,
        },
        "counts": {
            "satisfied": sum(1 for m in matches if m.get("status") == "satisfied"),
            "not_satisfied": sum(1 for m in matches if m.get("status") == "not_satisfied"),
            "cannot_judge": sum(1 for m in matches if m.get("status") == "cannot_judge"),
            "must_total": len(must_items),
            "must_satisfied": len(must_satisfied),
            "bonus_total": len(bonus_items),
        },
        # 兼容旧字段
        "overall_score": round(weighted_satisfaction, 4),
        "tier": tier,
        "tier_label": tier_label,
        "breakdown": (
            f"加权满足率={weighted_satisfaction:.2%} "
            f"(必须项: {len(must_satisfied)}/{len(must_items)}满足"
            + (f", {len(must_not_satisfied)}不满足" if must_not_satisfied else "")
            + (f", {len(must_cannot_judge)}无法判断" if must_cannot_judge else "")
            + ") | "
            f"证据验证率={evidence_rate:.2%} | "
            f"幻觉降级={len(hallucinated_items)}项 | "
            f"置信度={avg_confidence:.2%}"
            + (" | 需人工复核" if needs_human_review else "")
        ),
    }


def _empty_result() -> dict:
    """无匹配数据时的空结果"""
    return {
        "match_conclusion": {
            "tier": "insufficient_data",
            "tier_label": "数据不足",
            "weighted_satisfaction": 0.0,
            "must_summary": {"total": 0, "satisfied": 0, "not_satisfied": 0, "cannot_judge": 0},
            "bonus_summary": {"total": 0, "satisfied": 0},
            "hard_rule_triggered": False,
        },
        "confidence_assessment": {
            "avg_confidence": 0.0,
            "low_confidence_items": [],
            "needs_human_review": True,
            "human_review_reasons": ["无匹配数据"],
        },
        "evidence_quality": {
            "verified_rate": 0.0,
            "verified_count": 0,
            "total_evidence_count": 0,
            "hallucinated_count": 0,
            "hallucinated_items": [],
        },
        "counts": {
            "satisfied": 0, "not_satisfied": 0, "cannot_judge": 0,
            "must_total": 0, "must_satisfied": 0, "bonus_total": 0,
        },
        "overall_score": 0.0,
        "tier": "insufficient_data",
        "tier_label": "数据不足",
        "breakdown": "无匹配数据",
    }
