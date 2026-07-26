"""RAG 技能同义词种子数据

按设计报告 2.7 节，覆盖常见技术岗位的技能同义词映射。
每条记录的 canonical 是标准名称，同义词部分是从各种表述中映射过来。
"""

from backend.storage.chroma_client import chroma_client

SEED_SKILLS: list[dict] = [
    # ==================== AI / LLM ====================
    {"id": "ai_01", "document": "RAG项目经验 检索增强生成 知识库问答 向量检索+LLM 文档智能问答 文档检索 知识问答系统", "metadata": {"canonical": "RAG项目经验", "category": "AI技术"}},
    {"id": "ai_02", "document": "大模型API调用 LLM API OpenAI GPT Claude 大模型应用 调用GPT接口", "metadata": {"canonical": "大模型API调用", "category": "AI技术"}},
    {"id": "ai_03", "document": "Prompt Engineering 提示词工程 提示词优化 调Prompt 提示词设计 Prompt模板 提示词管理", "metadata": {"canonical": "Prompt Engineering", "category": "AI技术"}},
    {"id": "ai_04", "document": "LangChain LangChain框架 LlamaIndex Agent框架 LLM编排 LLM应用框架", "metadata": {"canonical": "LangChain", "category": "AI技术"}},
    {"id": "ai_05", "document": "向量数据库 Milvus Chroma Pinecone Qdrant Weaviate 向量检索 语义检索 ANN检索", "metadata": {"canonical": "向量数据库", "category": "AI技术"}},
    {"id": "ai_06", "document": "Embedding 文本嵌入 向量化 语义表示 嵌入模型 文本表征 BGE Embedding", "metadata": {"canonical": "Embedding", "category": "AI技术"}},
    {"id": "ai_07", "document": "Agent开发 AI Agent 智能体 多Agent Function Calling Tool Call 工具调用 Agent协作", "metadata": {"canonical": "Agent开发", "category": "AI技术"}},
    {"id": "ai_08", "document": "模型微调 Fine-tuning LoRA QLoRA 参数高效微调 指令微调 SFT RLHF", "metadata": {"canonical": "模型微调", "category": "AI技术"}},
    {"id": "ai_09", "document": "NLP 自然语言处理 文本分类 命名实体识别 情感分析 意图识别 语义理解", "metadata": {"canonical": "NLP", "category": "AI技术"}},
    {"id": "ai_10", "document": "计算机视觉 CV 图像识别 目标检测 OCR 图像分类 视觉AI", "metadata": {"canonical": "计算机视觉", "category": "AI技术"}},
    # ==================== 编程语言 ====================
    {"id": "py_01", "document": "Python Python开发 Python编程 Python后端 Python脚本 用Python", "metadata": {"canonical": "Python开发", "category": "编程语言"}},
    {"id": "py_02", "document": "FastAPI FastAPI框架 Python Web框架 ASGI 高性能API FastAPI后端", "metadata": {"canonical": "FastAPI", "category": "编程语言"}},
    {"id": "py_03", "document": "Flask Flask框架 Python Web WSGI Django 轻量级Web框架", "metadata": {"canonical": "Flask/Django", "category": "编程语言"}},
    {"id": "py_04", "document": "Java Spring SpringBoot Java后端 JVM 企业级开发", "metadata": {"canonical": "Java开发", "category": "编程语言"}},
    {"id": "py_05", "document": "Go Golang Go开发 云原生 高并发 微服务开发", "metadata": {"canonical": "Go开发", "category": "编程语言"}},
    {"id": "py_06", "document": "JavaScript JS TypeScript TS 前端脚本 Node.js 全栈开发", "metadata": {"canonical": "JavaScript/TypeScript", "category": "编程语言"}},
    # ==================== DevOps / 基础设施 ====================
    {"id": "dev_01", "document": "Docker 容器化 Docker部署 镜像管理 Dockerfile 容器编排 Docker Compose", "metadata": {"canonical": "Docker", "category": "DevOps"}},
    {"id": "dev_02", "document": "CI/CD 持续集成 持续部署 GitHub Actions Jenkins GitLab CI 自动化构建 自动化部署 流水线", "metadata": {"canonical": "CI/CD", "category": "DevOps"}},
    {"id": "dev_03", "document": "Kubernetes K8s 容器编排 集群管理 服务编排 Helm 云原生部署", "metadata": {"canonical": "Kubernetes", "category": "DevOps"}},
    {"id": "dev_04", "document": "Git Git版本控制 代码管理 GitHub GitLab 分支管理 Code Review 版本管理", "metadata": {"canonical": "Git", "category": "DevOps"}},
    {"id": "dev_05", "document": "Linux 服务器运维 Shell 命令行 系统管理 服务部署 环境配置", "metadata": {"canonical": "Linux运维", "category": "DevOps"}},
    {"id": "dev_06", "document": "云计算 AWS 阿里云 腾讯云 华为云 云服务 云部署 GCP Azure", "metadata": {"canonical": "云计算", "category": "DevOps"}},
    # ==================== 数据 ====================
    {"id": "dat_01", "document": "SQL MySQL PostgreSQL 数据库查询 关系型数据库 SQL查询 数据检索", "metadata": {"canonical": "SQL", "category": "数据处理"}},
    {"id": "dat_02", "document": "pandas 数据处理 数据分析 数据清洗 DataFrame 数据加工 数据统计", "metadata": {"canonical": "pandas", "category": "数据处理"}},
    {"id": "dat_03", "document": "ETL 数据管道 数据抽取 数据转换 数据加载 数据流水线 数据集成", "metadata": {"canonical": "ETL", "category": "数据处理"}},
    {"id": "dat_04", "document": "Redis 缓存 内存数据库 分布式缓存 消息队列 会话管理", "metadata": {"canonical": "Redis", "category": "数据处理"}},
    {"id": "dat_05", "document": "Elasticsearch ES 全文检索 搜索引擎 日志检索 倒排索引", "metadata": {"canonical": "Elasticsearch", "category": "数据处理"}},
    {"id": "dat_06", "document": "数据可视化 图表 ECharts Grafana Dashboard 报表 BI 数据看板", "metadata": {"canonical": "数据可视化", "category": "数据处理"}},
    # ==================== 前端 ====================
    {"id": "fe_01", "document": "React React.js React框架 前端开发 JSX Hooks 组件化 SPA", "metadata": {"canonical": "React", "category": "前端"}},
    {"id": "fe_02", "document": "Vue Vue.js Vue3 前端开发 渐进式框架 响应式 组件化", "metadata": {"canonical": "Vue", "category": "前端"}},
    {"id": "fe_03", "document": "前端工程化 Webpack Vite 构建工具 打包 模块化 前端工具链", "metadata": {"canonical": "前端工程化", "category": "前端"}},
    # ==================== 架构设计 ====================
    {"id": "arch_01", "document": "微服务 微服务架构 服务拆分 分布式系统 服务治理 服务注册 配置中心", "metadata": {"canonical": "微服务", "category": "架构设计"}},
    {"id": "arch_02", "document": "API设计 RESTful API接口设计 OpenAPI Swagger 接口文档 前后端分离", "metadata": {"canonical": "API设计", "category": "架构设计"}},
    {"id": "arch_03", "document": "系统设计 架构设计 高可用 高并发 性能优化 系统架构 技术选型", "metadata": {"canonical": "系统设计", "category": "架构设计"}},
    {"id": "arch_04", "document": "数据库设计 MySQL设计 MongoDB 数据库建模 索引优化 分库分表 读写分离", "metadata": {"canonical": "数据库设计", "category": "架构设计"}},

    # ==================== 学历/学位表述 ====================
    {"id": "edu_01", "document": "本科 学士 本科毕业 大学本科 本科学位 全日制本科", "metadata": {"canonical": "本科", "category": "学历"}},
    {"id": "edu_02", "document": "硕士 研究生 硕士毕业 硕士研究生 硕士学位 在读硕士", "metadata": {"canonical": "硕士", "category": "学历"}},
    {"id": "edu_03", "document": "博士 博士研究生 博士毕业 博士学位 PhD 博士在读", "metadata": {"canonical": "博士", "category": "学历"}},
    {"id": "edu_04", "document": "大专 专科 大专毕业 专科学位 高职", "metadata": {"canonical": "大专", "category": "学历"}},
    {"id": "edu_05", "document": "计算机相关专业 计算机科学 CS 软件工程 计算机科学与技术 信息工程 信息技术", "metadata": {"canonical": "计算机相关专业", "category": "学历"}},

    # ==================== 经验深度表述 ====================
    {"id": "exp_01", "document": "主导 负责 从0到1 从零搭建 架构设计 独立完成 技术负责人 核心开发", "metadata": {"canonical": "主导/深度参与", "category": "经验深度"}},
    {"id": "exp_02", "document": "参与 协助 配合 了解 使用过 接触过 辅助开发 联调", "metadata": {"canonical": "参与/了解", "category": "经验深度"}},
    {"id": "exp_03", "document": "生产环境 上线 投产 线上 生产级 正式环境 商用 企业级", "metadata": {"canonical": "生产环境经验", "category": "经验深度"}},
    {"id": "exp_04", "document": "Demo 原型 验证 概念验证 POC 实验性 调研 尝试", "metadata": {"canonical": "原型/实验", "category": "经验深度"}},

    # ==================== 职位名称变体 ====================
    {"id": "role_01", "document": "Python后端开发 Python开发工程师 Python工程师 后端开发 Python程序员 服务端开发", "metadata": {"canonical": "Python后端开发", "category": "职位"}},
    {"id": "role_02", "document": "AI应用开发工程师 AI开发 AI工程师 LLM应用开发 大模型应用开发 智能应用开发", "metadata": {"canonical": "AI应用开发工程师", "category": "职位"}},
    {"id": "role_03", "document": "全栈工程师 全栈开发 前后端开发 FullStack 全栈", "metadata": {"canonical": "全栈工程师", "category": "职位"}},
    {"id": "role_04", "document": "数据工程师 数据开发 大数据开发 数据平台 数据仓库", "metadata": {"canonical": "数据工程师", "category": "职位"}},
    {"id": "role_05", "document": "算法工程师 算法岗 机器学习工程师 ML Engineer 深度学习 推荐算法", "metadata": {"canonical": "算法工程师", "category": "职位"}},

    # ==================== 项目类型表述 ====================
    {"id": "proj_01", "document": "企业内部系统 后台管理系统 内部工具 运营后台 Admin系统 管理后台", "metadata": {"canonical": "企业内部系统", "category": "项目类型"}},
    {"id": "proj_02", "document": "ToC产品 面向用户 C端产品 用户端 消费者应用 公众产品", "metadata": {"canonical": "ToC产品", "category": "项目类型"}},
    {"id": "proj_03", "document": "高并发系统 大流量 高QPS 秒杀 高负载 性能敏感", "metadata": {"canonical": "高并发系统", "category": "项目类型"}},
    {"id": "proj_04", "document": "数据平台 数据中台 数据产品 数据系统 分析平台 数据服务", "metadata": {"canonical": "数据平台", "category": "项目类型"}},
    {"id": "proj_05", "document": "智能客服 ChatBot 对话系统 问答机器人 客服系统 NLP应用", "metadata": {"canonical": "智能客服/ChatBot", "category": "项目类型"}},

    # ==================== AI 产品经理领域 ====================
    {"id": "pm_01", "document": "AI编程工具 AI Coding Vibe Coding AI辅助编程 智能编程 AI代码生成 Claude Code Cursor Copilot Codex 用AI写代码 AI编程", "metadata": {"canonical": "AI编程工具", "category": "AI产品"}},
    {"id": "pm_02", "document": "市场调研 用户研究 竞品分析 行业研究 市场分析 用户洞察 需求调研 用户访谈 用户反馈回收 用户画像 场景调研", "metadata": {"canonical": "市场调研与用户洞察", "category": "AI产品"}},
    {"id": "pm_03", "document": "PRD 产品需求文档 产品规格书 需求规格说明 功能说明文档 BRD MRD 需求文档 写PRD", "metadata": {"canonical": "PRD产品需求文档", "category": "AI产品"}},
    {"id": "pm_04", "document": "指标体系 KPI OKR 数据指标 核心指标 评测指标 效果指标 评价体系 数据埋点 指标监控 效果量化", "metadata": {"canonical": "指标体系建设", "category": "AI产品"}},
    {"id": "pm_05", "document": "工作流沉淀 SOP 标准流程 知识库 最佳实践 流程优化 规范化 文档沉淀 经验总结 可复用 效率提升 模板化", "metadata": {"canonical": "工作流沉淀", "category": "AI产品"}},
    {"id": "pm_06", "document": "API调用 接口调用 REST API HTTP请求 调用接口 API集成 调接口 后端接口 接口对接", "metadata": {"canonical": "API调用能力", "category": "AI产品"}},
    {"id": "pm_07", "document": "产品Sense 产品思维 用户同理心 产品判断力 商业敏感度 用户视角 产品洞察 需求判断", "metadata": {"canonical": "产品Sense", "category": "AI产品"}},
    {"id": "pm_08", "document": "自驱力 Owner意识 主动性 独立推动 自驱 Ownership 主人翁意识 主动承担 独立负责", "metadata": {"canonical": "自驱力", "category": "AI产品"}},
    {"id": "pm_09", "document": "数据分析 AB测试 数据驱动 实验设计 效果评估 数据验证 量化分析 统计显著", "metadata": {"canonical": "数据分析能力", "category": "AI产品"}},
    {"id": "pm_10", "document": "AI产品设计 AI功能设计 AI产品经理 AI产品 AI交互 智能产品设计 AI场景 AI落地", "metadata": {"canonical": "AI产品设计", "category": "AI产品"}},
    {"id": "pm_11", "document": "从0到1 项目孵化 独立孵化 全流程 从零搭建 产品孵化 0-1 完整闭环 从想法到上线", "metadata": {"canonical": "从0到1孵化", "category": "AI产品"}},

    # ==================== 认证/证书 ====================
    {"id": "cert_01", "document": "AWS认证 AWS Certified 云计算认证 Solutions Architect", "metadata": {"canonical": "AWS认证", "category": "认证"}},
    {"id": "cert_02", "document": "CKA认证 Kubernetes管理员 Certified Kubernetes Administrator 云原生认证", "metadata": {"canonical": "CKA认证", "category": "认证"}},
    {"id": "cert_03", "document": "PMP认证 项目管理专业人士 项目管理认证 Project Management", "metadata": {"canonical": "PMP认证", "category": "认证"}},
]


def seed(clear_first: bool = True, force_rebuild: bool = False) -> int:
    """向 ChromaDB 写入同义词种子数据

    Args:
        clear_first: 是否先清空已有数据
        force_rebuild: 强制重建集合（切换 embedding 模型时使用）

    Returns:
        int: 写入的记录数
    """
    if force_rebuild:
        chroma_client.rebuild()
    elif clear_first and chroma_client.count() > 0:
        chroma_client.clear()

    if chroma_client.count() > 0:
        print(f"ChromaDB 已有 {chroma_client.count()} 条记录，跳过种子化")
        return chroma_client.count()

    ids = [s["id"] for s in SEED_SKILLS]
    documents = [s["document"] for s in SEED_SKILLS]
    metadatas = [s["metadata"] for s in SEED_SKILLS]

    chroma_client.add(ids=ids, documents=documents, metadatas=metadatas)

    count = chroma_client.count()
    print(f"ChromaDB 种子数据写入完成: {count} 条记录")
    return count


if __name__ == "__main__":
    seed(force_rebuild=True)
