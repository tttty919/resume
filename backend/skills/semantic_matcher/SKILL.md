---
name: semantic_matcher
description: 逐条要求判断简历是否满足（硬条件代码判断 + RAG 同义词扩展 + LLM 语义匹配与证据提取）
entry_point: backend.skills.semantic_matcher.node.match
output_schema: backend.skills.semantic_matcher.schema.SemanticMatcherOutput
---

# SemanticMatcher

## 输入

- `requirements: list[dict]` — JDParser 输出的结构化要求
- `resume: dict` — ResumeExtractor 输出的结构化简历
- `raw_resume_text: str`（可选）— 简历原文，用于逐字摘录证据

## 输出

`SemanticMatcherOutput`：`matches` 列表，每项含 requirement_id/status(satisfied|not_satisfied|cannot_judge)/confidence/reasoning/evidence 等。

## 说明

包含已合并的 EvidenceExtractor 能力（不再是独立 Skill）。流程：
1. 学历/年限/院校等硬条件先走代码规则判断（`_match_hard_condition`），无法机械判断的交给 LLM。
2. 软性要求通过 `tools.py`（`expand`/`learn`）查询 ChromaDB 做同义词扩展。
3. 剩余要求分批并行调用 LLM 做语义匹配 + 证据摘录。

`run(state)` 提供 LangGraph 风格的节点入口。
