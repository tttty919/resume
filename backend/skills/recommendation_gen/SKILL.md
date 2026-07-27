---
name: recommendation_gen
description: 生成最终录用建议（规则评分引擎定级 + LLM 文本润色）
entry_point: backend.skills.recommendation_gen.node.generate
output_schema: backend.skills.recommendation_gen.schema.RecommendationGenOutput
---

# RecommendationGen

## 输入

- `requirements: list[dict]` — JDParser 输出的结构化要求
- `matches: list[dict]` — SemanticMatcher 输出的匹配结果
- `validated_items: list[dict]`（可选，默认等于 matches）
- `risk_analysis: dict`（可选）— RiskAnalyzer 输出

## 输出

`RecommendationGenOutput`：`summary`（满足/不满足/无法判断计数、优势、风险、推荐结论）+ `recommendation_reason`（LLM 润色文本）；附带 `scoring`（评分引擎的原始打分结果）。

## 说明

核心原则：`scoring_engine.py` 中的纯规则评分引擎负责算分定级，LLM 只做文本润色，不修改推荐等级。
