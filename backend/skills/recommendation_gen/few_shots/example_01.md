## 示例 1："推荐面试" — 必须项基本满足，仅经验年限需HR确认

**输入：**

scoring: counts{satisfied=4, not_satisfied=0, cannot_judge=1, must_satisfied=3, must_not_satisfied=0, must_cannot_judge=1}, tier_label="推荐面试", hard_rule_triggered=false, needs_human_review=true（原因: req-005产品经验年限不确定）, hallucinated_count=0, verified_rate=0.80, overall_score=78

risk_analysis 核心: 优势4条（AI产品设计确凿/数据分析突出/AI工具实操强/实习覆盖完整链路），风险2条（产品经验年限不确定/原型工具覆盖面窄），info_gaps1条（全职经验未明确）

**正确输出：**

summary:
- satisfied_count=4, not_satisfied_count=0, cannot_judge_count=1
- core_advantages: 4条精炼版（25字以内/条），从risk_analysis去重合并
- key_risks: 2条，每条关联requirement_id
- recommendation: "推荐面试"（=tier_label，不可修改）
- needs_human_review: true
- human_review_questions: 3条，每条关联具体requirement

recommendation_reason:
"4项满足，0项不满足，加权匹配分78分。3项必须项（AI产品PRD、数据分析、原型输出）均有确凿的项目证据和量化成果，核心能力匹配度高。仅学历与经验1项因候选人无全职经验而标记为无法判断，confidence=55%，建议HR电话确认产品经验年限的定义后直接安排面试。"

**常见错误（禁止）：**
- ❌ 把tier_label从"推荐面试"改成"需人工复核" → 定级权在评分引擎
- ❌ recommendation_reason写成"该候选人综合素质较高，建议推进" → 必须引用具体满足/不满足数量和关键优势名称
- ❌ 面试问题写"请确认候选人能力" → 必须指明哪个requirement、确认什么
- ❌ 把risk_analysis的每条风险原文照搬 → 应提炼为4-6条精炼列表（每条不超过25字）
- ❌ 逐条列举requirement → recommendation_reason是一段连续文字，不超过120字
