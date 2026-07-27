---
name: jd_parser
description: 将原始 JD 文本解析为结构化岗位要求列表
entry_point: backend.skills.jd_parser.node.parse
output_schema: backend.skills.jd_parser.schema.JDParserOutput
---

# JDParser

## 输入

- `raw_jd: str` — 原始 JD 文本

## 输出

`JDParserOutput`：结构化的 `requirements` 列表（每项含 id/name/description/type/importance/keywords）+ `parsing_notes` 说明。

## 说明

无 Tool 依赖，纯 LLM 推理。`few_shots/` 目录下的示例会自动拼接进 prompt。`run(state)` 提供 LangGraph 风格的节点入口，若 `state["requirements"]` 已存在且非空则跳过。
