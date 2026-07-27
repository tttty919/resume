---
name: resume_extractor
description: 将简历纯文本解析为结构化数据（基本信息、工作经历、项目经历、技能）
entry_point: backend.skills.resume_extractor.node.extract
output_schema: backend.skills.resume_extractor.schema.ResumeExtractorOutput
---

# ResumeExtractor

## 输入

- `raw_resume_text: str` — 简历纯文本（由 DocumentParser 从原始文件解析得到）

## 输出

`ResumeExtractorOutput`：`basic_info`（姓名/学历/工作年限等）+ `work_experiences` + `project_experiences` + `skills`。

## 说明

无 Tool 依赖，纯 LLM 推理。`run(state)` 提供 LangGraph 风格的节点入口，若 `state["structured_resume"]` 已存在则跳过。
