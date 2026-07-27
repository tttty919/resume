## 示例 2：AI研究员转型PM — not_satisfied vs cannot_judge 区分 + OR条件

**输入：**

requirements: 5条（req-001 AI产品需求与PRD must/high, req-002 AI大模型理解 must/high, req-005 学历与经验 must/high, req-007 数据分析能力 must/high, req-008 原型与PRD输出 must/medium）

resume（王子轩）: 上交大AI硕士，商汤AI产品实习4个月+上海AI Lab大模型训练实习，Python/PyTorch/LangChain/RAG等50+AI技能，ACL论文，但产品经验仅4个月且Figma"学习中"

**正确输出（关键判断）：**

req-001（AI产品需求与PRD）→ **satisfied**，confidence=0.85，evidence_support="确凿"
- 理由：商汤实习独立完成智能投研助手V1.0产品方案（含需求梳理、竞品调研、PRD编写）
- ❌ 不应因"只有4个月实习"降级 → 实习期间确实输出了产品方案和PRD，匹配要求本身

req-002（AI大模型理解）→ **satisfied**，confidence=0.95，evidence_support="确凿"
- 理由：参与书生大模型SFT+商汤RAG架构设计+Prompt模板+ACL论文，全链路覆盖远超要求

req-005（学历与经验）→ **cannot_judge**，confidence=0.50，evidence_support="信息不足"
- 理由：学历远超要求（上交大AI硕士+浙大CS本科）。但仅4个月AI产品实习，远低于"2年"要求
- ❌ 不应判not_satisfied：有产品实习，不是零经验
- ❌ 不应判satisfied：4个月≠2年，差距太大

req-007（数据分析能力，含"等"→满足其一即可）→ **satisfied**，confidence=0.90，evidence_support="确凿"
- 理由：Python精通有项目证据（SFT数据配比优化使推理提升4.2pp），SQL熟练有技能标签+AI Lab数据分析实战
- 要求写"SQL、Python等"，候选人两个都满足

req-008（原型与PRD输出）→ **cannot_judge**，confidence=0.45，evidence_support="信息不足"
- 理由：PRD标注"基础"、Figma标注"学习中使用"——仅是技能标签/学习阶段，无实际项目的交互原型输出证据
- ❌ 不应判satisfied：仅技能标签且标注"学习中"，无项目证据
- ❌ 不应判not_satisfied：候选人确实有基础了解，不是完全不会

**常见错误（禁止）：**
- ❌ 把req-008判为satisfied → "Figma学习中"=无项目，只有标签没有项目=cannot_judge
- ❌ 把req-005判为not_satisfied → 简历有4个月产品经验，不是0经验，信息不足应判cannot_judge
- ❌ 把req-008判为not_satisfied → "Figma学习中"≠不会Figma，只是无法判断实际水平
- ❌ 看到req-002满分就忽略→ AI研究员背景在技术项上是强项，confidence可以给高
