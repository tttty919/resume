## 示例 1：均衡型AI PM — 多段实习 + 完整技能提取

**简历原文（片段）：**
陈思远
Base 地：杭州 | 电话：(+86) 158-6912-3478 | 邮箱: chensiyuan.pm@gmail.com

教育背景
华中科技大学（985/211）信息管理与信息系统专业 | 硕士 2023.09-2026.07
武汉理工大学（211）电子商务专业 | 本科 2019.09-2023.07

实习经历
阿里巴巴（淘宝）AI产品经理实习生 2026.01–至今
- 参与淘宝"AI购物助手"的产品迭代，帮助用户通过自然语言对话完成商品搜索、比价和购买决策。
- 独立负责"多轮对话澄清"模块，当用户需求模糊时，通过追问收集用户偏好缩小候选商品范围。上线后用户对话完成率从58%提升至73%。
- 针对商品推荐场景，迭代优化LLM的System Prompt，增加商品属性权重、用户画像注入和品牌偏好考虑，使推荐点击率提升11.2%。

小红书 产品实习生（搜索方向）2025.06–2025.12
- 参与小红书搜索功能的产品优化，负责搜索无结果页面的策略设计。
- 设计"相关推荐+社区问答引流+AI生成内容兜底"的三层降级策略，降低搜索跳出率18%。

字节跳动（飞书）数据分析实习生 2024.06–2024.09
- 使用SQL和Python分析飞书文档模块的用户行为数据，识别活跃用户的核心使用路径。
- 构建用户留存预测模型（Random Forest），识别高流失风险用户的关键行为特征。

个人技能
产品技能：PRD编写（熟练）、Figma（熟练）、Axure（基础）、用户访谈、竞品分析、数据埋点设计
AI工具：Claude Code（日常使用）、Cursor（日常使用）、ChatGPT、Prompt Engineering
数据分析：SQL（精通）、Python（熟练，Pandas/Scikit-learn）、Quick BI、DataV
开发能力：基础的HTML/CSS/JavaScript，能使用Vibe Coding完成简单的工具页面开发。

**正确输出（关键字段）：**

basic_info: name="陈思远", school="华中科技大学", major="信息管理与信息系统", degree="硕士", work_years=null, current_role="AI产品经理实习生"

skills（部分关键项）:
- 产品工具: PRD编写, Figma, Axure, 用户访谈, 竞品分析, 数据埋点设计
- AI工具: Claude Code, Cursor, ChatGPT, Prompt Engineering
- 数据分析: SQL, Python, Pandas, Scikit-learn, Quick BI, DataV, Random Forest
- 开发: HTML, CSS, JavaScript, Vibe Coding
- 领域: 需求分析, 功能设计, 策略设计, 效果监控, 指标体系, 意图识别, LLM质检

work_experiences: 3段，每段包含 company + role + duration + responsibilities（原文摘录，不改写）

extraction_notes: "1. 工作年限未明确提及，设为null；2. 硕士为在读状态；3. 技能列表已覆盖所有技术名词，包括课程、实习和项目中出现的工具与方法。"

**常见错误（禁止）：**
- ❌ work_years填"约2年" → 简历未明确写"X年经验"时必须填null，不能估算
- ❌ 技能列表漏掉"Random Forest""Vibe Coding"等项目中出现的具体技术 → 必须全文扫描
- ❌ responsibilities改写原文（如"负责多轮对话模块"）→ 必须逐字摘录，保留"独立负责"等措辞
- ❌ degree写"硕士 在读" → degree字段只取值"博士|硕士|本科|大专|null"
- ❌ 把"985/211"脑补成具体的GPA或排名 → 不能编造
