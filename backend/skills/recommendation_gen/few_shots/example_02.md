## 示例 2："需人工复核" — 硬规则触发 + 存在低置信度项

**输入：**

scoring: counts{satisfied=3, not_satisfied=0, cannot_judge=2, must_satisfied=3, must_not_satisfied=0, must_cannot_judge=2}, tier_label="需人工复核", hard_rule_triggered=true（原因: 2项must项cannot_judge）, needs_human_review=true（原因: req-005经验不足+req-008原型证据缺失）, low_confidence_items=[req-008(0.45), req-005(0.50)], hallucinated_count=0, verified_rate=0.60, overall_score=62

risk_analysis 核心: 优势4条（AI大模型顶尖/驱动AI落地极强/LLM评测体系化思维/学术+竞赛），风险3条（经验年限严重不足/原型能力薄弱/技术vs产品思维平衡），info_gaps2条

**正确输出：**

summary:
- satisfied_count=3, not_satisfied_count=0, cannot_judge_count=2
- recommendation: "需人工复核"（=tier_label，不可修改）
- needs_human_review: true

recommendation_reason:
"3项满足，0项不满足，但2项必须项（学历与经验、原型与PRD输出）因证据不足标记为无法判断，触发了硬规则需人工复核。候选人AI大模型技术能力顶尖（SFT+RAG+Prompt全链路+ACL论文），在所有候选人中技术深度最强，但产品经验（仅4个月实习）和原型设计能力（PRD基础+Figma学习中）存在明显短板。建议HR优先电话确认产品经验年限和原型能力，如果候选人能证明具备独立产品输出能力，可从'需复核'升级为'推荐面试'。"

**常见错误（禁止）：**
- ❌ 因为"技术深度最强"而把推荐等级改成"推荐面试" → hard_rule_triggered=true，定级不可修改
- ❌ recommendation_reason写"建议面试" → tier_label是"需人工复核"，理由必须一致
- ❌ 面试问题只问技术（"SFT怎么调参"）→ 应聚焦信息缺口：产品经验+原型能力
- ❌ 忽略low_confidence_items → 有2项低置信度（0.45/0.50），理由中必须提示"部分判断置信度较低"
- ❌ recommendation_reason写"候选人综合能力较强" → 3满足+2无法判断，不能只说优势不说短板
