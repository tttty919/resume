---
name: risk_analyzer
description: 基于匹配结果做综合风险评估，提炼核心优势、关键风险、信息缺口与面试建议
entry_point: backend.skills.risk_analyzer.node.analyze
output_schema: backend.skills.risk_analyzer.schema.RiskAnalyzerOutput
---

# RiskAnalyzer

## 输入

- `requirements: list[dict]` — JDParser 输出的结构化要求
- `validated_items: list[dict]` — SemanticMatcher 输出的匹配结果

## 输出

`RiskAnalyzerOutput`：`analysis` 含 `core_advantages`/`key_risks`/`info_gaps`/`interview_suggestions`。

## 说明

无外部 Tool 依赖，纯 LLM 推理节点。`run(state)` 提供 LangGraph 风格的节点入口。
