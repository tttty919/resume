## 示例 1：均衡型AI PM匹配 — 有项目=满足 vs 仅标签=无法判断

**输入：**

requirements: 5条（req-001 AI产品需求与PRD must/high, req-005 学历与经验 must/high, req-007 数据分析能力 must/high, req-008 原型与PRD输出 must/medium, req-010 CS/AI专业或编程 bonus/medium）

resume（陈思远）: 华中科技大学硕士（信息管理），3段实习（淘宝AI产品+小红书搜索策略+字节数据分析），技能含PRD编写/Figma/SQL/Python/HTML等，有具体项目量化成果（对话完成率58%→73%，推荐点击率+11.2%，搜索跳出率-18%）

**正确输出（关键判断）：**

req-001（AI产品需求与PRD）→ **satisfied**，confidence=0.92，evidence_support="确凿"
- 理由：候选人在淘宝独立负责"多轮对话澄清"模块，有从需求分析到功能上线的完整项目证据
- 证据：逐字摘录"独立负责'多轮对话澄清'模块...上线后用户对话完成率从58%提升至73%"
- ❌ 不应判cannot_judge：有完整项目经历+量化成果，远超"仅技能标签"

req-005（学历与经验）→ **cannot_judge**，confidence=0.55，evidence_support="信息不足"
- 理由：学历满足（硕士），但JD要求"2年以上产品经验"，3段实习累计约2年但均为实习生身份，需HR确认定义
- ❌ 不应判not_satisfied：简历有产品实习经验，不是明确不满足
- ❌ 不应判satisfied：实习生不等同于全职产品经验

req-007（数据分析能力，含"等"→满足其一即可）→ **satisfied**，confidence=0.93，evidence_support="确凿"
- 理由：SQL精通+Python熟练，飞书实习中使用SQL/Python构建Random Forest预测模型，有项目证据
- 要求写"SQL、Python等"，「等」=举例非穷举，候选人两个都满足

req-008（原型与PRD输出，含"等"→满足其一即可）→ **satisfied**，confidence=0.88，evidence_support="确凿"
- 理由：Figma（熟练）+PRD编写（熟练），有淘宝产品设计和AI质检系统项目中的PRD输出证据
- 要求"Figma/Axure等"，候选人Figma熟练满足其一

req-010（CS/AI专业或编程，bonus）→ **satisfied**，confidence=0.78，evidence_support="确凿"
- 理由：专业非严格CS/AI，但具备HTML/CSS/JS编程能力，满足OR条件"具备基础编程能力"
- bonus项：技能列表有编程标签即可满足

**常见错误（禁止）：**
- ❌ 看到简历有"2年"实习时长就判req-005 satisfied → 实习≠全职，需区分
- ❌ 看到req-007写"SQL、Python等"要求两个都满足 → "等"表示举例非穷举
- ❌ req-010因为专业不是CS就判not_satisfied → 有OR条件的必须检查两侧
- ❌ 仅凭技能列表"SQL（精通）"判satisfied → 要求有项目证据（此处刚好有项目，所以是对的）
