"""QueryExpander — RAG 同义词检索

被 Skill 3 (SemanticMatcher) 调用，在匹配前为每项要求扩展语义等价的同义词。
支持反馈回路: HR 确认的同义词写入 ChromaDB，持续丰富知识库。

由于 ChromaDB 默认 embedding 模型对中文支持差，增加关键词回退机制：
先尝试向量检索 → 如果结果质量差 → 回退到关键词匹配。
"""

from backend.core.logger import get_logger
from backend.storage.chroma_client import chroma_client

log = get_logger()

# 向量检索最低相似度：低于此值说明 embedding 不适用，回退到关键词匹配
MIN_VECTOR_SIMILARITY = 0.3


def expand(query: str, n_results: int = 5) -> list[str]:
    """扩展查询词的同义表述（向量检索 + 关键词回退）"""
    try:
        synonyms = _vector_search(query, n_results)
        if synonyms:
            log.debug(f"QueryExpander (vector): '{query[:50]}...' -> {len(synonyms)} synonyms")
            return synonyms
    except Exception as e:
        log.warning(f"向量检索失败（HuggingFace 不可用），跳过 RAG: {e}")

    # 向量检索失败或为空 → 回退到关键词匹配
    try:
        synonyms = _keyword_fallback(query, n_results)
        log.debug(f"QueryExpander (keyword): '{query[:50]}...' -> {len(synonyms)} synonyms")
        return synonyms
    except Exception as e:
        log.warning(f"关键词回退也失败，返回空: {e}")
        return []


def _vector_search(query: str, n_results: int) -> list[str]:
    """向量检索 + 相似度过滤"""
    results = chroma_client.collection.query(query_texts=[query], n_results=n_results)
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    synonyms: list[str] = []
    for distance, doc in zip(distances, documents):
        similarity = 1.0 - distance
        if similarity < MIN_VECTOR_SIMILARITY:
            continue  # embedding 不相关，跳过
        parts = doc.split()
        synonyms.extend(parts[1:])  # 跳过 canonical
    return list(set(synonyms))


def _keyword_fallback(query: str, n_results: int) -> list[str]:
    """关键词回退：直接匹配种子文档中的词汇

    当 embedding 模型对中文效果差时使用。
    对查询词进行分词，然后匹配包含这些词的种子文档。
    """
    # 获取所有种子文档
    all_docs = chroma_client.collection.get()
    documents = all_docs.get("documents", [])

    if not documents:
        return []

    # 从查询中提取关键词（中文按单字拆太碎，按常见词长匹配）
    query_terms = _extract_terms(query)

    scored: list[tuple[str, int]] = []
    for doc in documents:
        score = 0
        # 计算查询词在文档中的命中数
        for term in query_terms:
            if term in doc:
                score += 1
        if score > 0:
            parts = doc.split()
            for synonym in parts[1:]:  # 跳过 canonical
                scored.append((synonym, score))

    # 去重 + 按得分排序 + 取 top
    seen: set[str] = set()
    result: list[str] = []
    for synonym, score in sorted(scored, key=lambda x: -x[1]):
        if synonym not in seen and len(synonym) >= 2:
            seen.add(synonym)
            result.append(synonym)
        if len(result) >= n_results * 3:
            break

    return result


def _extract_terms(query: str) -> list[str]:
    """从查询中提取有意义的搜索词（简单分词）"""
    import re
    # 按空格、标点分割
    raw_terms = re.split(r'[\s,，、。；;]+', query)
    terms = []
    for t in raw_terms:
        t = t.strip()
        if len(t) >= 2:
            terms.append(t)
            # 对于中文长词，也试 2-3 字片段
            if len(t) >= 4:
                for i in range(len(t) - 1):
                    chunk = t[i:i+2]
                    if len(chunk) >= 2:
                        terms.append(chunk)
    return list(set(terms))


def learn(requirement_name: str, new_synonym: str) -> dict:
    """HR 反馈回路: 将确认的同义词写入知识库

    Args:
        requirement_name: JD 要求的标准名称 (canonical)，如 "LLM应用开发经验"
        new_synonym: 简历中实际出现且被 HR 确认的同义表述，如 "智能对话开发"

    Returns:
        dict: {added: bool, count: int, message: str}
    """
    if not requirement_name or not new_synonym:
        return {"added": False, "count": 0, "message": "参数不能为空"}

    # 检查是否已存在（去重：关键词匹配）
    all_docs = chroma_client.collection.get()
    for doc in all_docs.get("documents", []):
        if new_synonym in doc:
            return {"added": False, "count": chroma_client.count(), "message": f"'{new_synonym}' 已在知识库中"}

    import uuid
    doc_id = f"feedback_{uuid.uuid4().hex[:8]}"
    doc_text = f"{requirement_name} {new_synonym}"

    chroma_client.add(
        ids=[doc_id],
        documents=[doc_text],
        metadatas=[{"source": "hr_feedback", "requirement": requirement_name, "synonym": new_synonym}],
    )

    total = chroma_client.count()
    log.info(f"反馈回路: '{requirement_name}' + 同义词 '{new_synonym}' -> ChromaDB (总记录: {total})")
    return {"added": True, "count": total, "message": f"已添加: {new_synonym} (总记录: {total})"}
