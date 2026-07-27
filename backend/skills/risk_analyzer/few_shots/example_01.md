## 示例 1：均衡型AI PM — 必须项大部分满足 + 仅有经验年限需确认

**输入：**

requirements: 5条（req-001 AI产品PRD must/high, req-005 学历经验 must/high, req-007 数据分析 must/high, req-008 原型输出 must/medium, req-010 CS/AI编程 bonus/medium）

validated_items 摘要: 4 satisfied + 1 cannot_judge（req-005，confidence=0.55，实习生身份需确认）, 0 not_satisfied, 无低置信度(<0.6)的satisfied项, needs_human_review=1条

**正确输出：**

core_advantages:
- "AI产品设计与PRD能力确凿（req-001）：淘宝AI购物助手独立负责'多轮对话澄清'模块，对话完成率58%→73%，有可验证的量化成果"
- "数据分析能力突出（req-007）：SQL精通+Python熟练，字节飞书实习中使用SQL/Python做用户留存预测模型（Random Forest），技术栈扎实"
- "AI工具实操经验强：Claude Code/Cursor日常使用，淘宝LLM Prompt优化使推荐点击率提升11.2%，Prompt Engineering有实际业务验证"
- "三段实习覆盖AI PM完整链路：淘宝（AI产品迭代）+小红书（搜索策略）+字节（数据分析），经验广度好"

key_risks:
- "【严重】产品经验年限不确定（req-005，must/high）：JD要求'2年以上互联网/AI产品经验'，候选人3段实习均为实习生身份，累计约2年但非全职。若JD严格指全职经验则不满足，需HR确认公司对产品经验的定义"
- "【中等】原型设计输出偏弱：Figma熟练但Axure仅基础，若团队以Axure为核心工具可能存在上手成本"

info_gaps:
- "全职产品经验年限未明确（req-005）：简历无全职工作经历，无法判断实习经验是否计入2年要求"
- "多团队协作经验未详细描述：简历提到'推动算法侧优化'但未展开说明跨团队协作的规模和复杂度"

interview_suggestions:
- "[req-005] 你在淘宝的AI产品经理实习是日常实习还是校招留用岗？是否有其他全职产品经验未在简历中体现？"
- "[req-001] 淘宝AI购物助手的'多轮对话澄清'模块是你独立设计的，能具体说说你是如何确定追问的维度（预算/风格/年龄）的？有没有做过用户测试来验证这些维度的有效性？"
- "[req-007] 你在飞书做的用户留存预测模型，这个模型的准确率如何？模型产出的洞察是否被业务方采纳并推动了实际的产品改动？"

**常见错误（禁止）：**
- ❌ 把evidence_support="确凿"的项目写成风险 → "AI产品能力确凿"在优势里，不能又在风险里说"产品能力待验证"
- ❌ 风险不标注严重程度和关联requirement → 必须标注【严重/中等/轻微】+ req_id + type/importance
- ❌ 面试问题过于宽泛（如"请介绍一下你的项目经验"）→ 必须指向具体模块和可验证指标
- ❌ 风险列表写"候选人可能缺乏XX" → 不能用"可能"，要基于validated_items的具体判断
