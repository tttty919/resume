## 示例 2：AI研究员转型PM — 技术型候选人 + 论文识别

**简历原文（片段）：**
王子轩
Base 地：上海 | 电话：(+86) 185-2173-9801 | 邮箱: wangzx.ai@sjtu.edu.cn

教育背景
上海交通大学（985/211，C9）人工智能专业 | 硕士 2024.09-2027.07
浙江大学（985/211）计算机科学与技术专业 | 本科 2020.09-2024.07

实习经历
商汤科技 AI产品实习生（大模型应用方向）2026.02–2026.06
- 参与商汤日日新大模型在金融行业的应用落地，负责智能投研助手的产品化工作。
- 将公司自研的大模型能力适配到金融研报自动生成场景，设计Prompt模板与RAG知识库架构，解决研报数据时效性和引用准确性问题。
- 独立完成智能投研助手V1.0的产品方案，包括需求梳理、竞品调研（BloombergGPT、Kensho等）、功能设计和PRD编写。
- 设计50维度的LLM输出质量评测体系，覆盖事实准确性、逻辑连贯性、金融专业性等维度。

上海人工智能实验室 大模型训练实习生 2025.06–2025.09
- 参与书生通用大模型的SFT数据构建工作，负责数据清洗脚本编写与质量过滤规则的制定。
- 基于模型在特定任务上的表现分析，提出训练数据配比优化方案，使模型在逻辑推理基准测试中提升4.2个百分点。

项目经历
论文：LLM-as-Judge可靠性研究 2025.09-2026.01
- 作为第一作者，对LLM-as-Judge评估范式的可靠性进行系统性研究，比较GPT-4、Claude、DeepSeek等模型与人工评分的一致性。
- 论文被ACL 2026接收为Workshop论文。

个人技能
AI/ML：Python（精通）、PyTorch（熟练）、Transformers（熟练）、LangChain、RAG、Prompt Engineering、模型微调（SFT/LoRA）
产品工具：Figma（学习中使用）、PRD编写（基础）、竞品分析
数据分析：SQL（熟练）、Pandas（精通）、NumPy（精通）、Matplotlib、WandB

**正确输出（关键字段）：**

basic_info: name="王子轩", school="上海交通大学", major="人工智能", degree="硕士", work_years=null, current_role="AI产品实习生（大模型应用方向）"

skills（部分关键项）:
- AI/ML: Python, PyTorch, Transformers, LangChain, RAG, Prompt Engineering, 模型微调, SFT, LoRA
- 数据: SQL, Pandas, NumPy, Matplotlib, WandB
- 产品: Figma, PRD编写, 竞品分析
- 学术/领域: 深度学习, 强化学习, 多模态大模型, 计算机视觉, 自然语言处理, Scaling Law, LLM评测, LLM-as-Judge
- 模型: GPT-4, Claude, DeepSeek, Qwen, FastAPI
- 竞赛: Kaggle

extraction_notes: "1. 工作年限未明确提及，设为null；2. 硕士为在读状态；3. 有ACL论文发表（Workshop），作为项目经历提取；4. 技能列表已覆盖所有AI/ML技术名词和产品工具；5. Kaggle银牌在获奖荣誉中提到，未单独设字段但技能中已保留。"

**常见错误（禁止）：**
- ❌ 只提取"产品技能"部分的技能，漏掉AI/ML部分 → 必须全文扫描所有技术名词
- ❌ 论文发表不提取 → 项目经历中应包含论文研究项目
- ❌ work_years根据"2024.09入学"推算为"应届" → 不能推算，填null
- ❌ 把"Figma（学习中使用）"提取为"Figma（熟练）" → 必须保留原文中的熟练度标注
- ❌ 漏掉"WandB""Scaling Law"等非主流但简历明确提及的技能 → 所有技术名词都要提取
