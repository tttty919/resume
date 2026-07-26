"""文本清洗工具 — JD 与简历预处理

处理 PDF/DOCX 解析和粘贴文本中的噪音：
- Unicode 控制字符
- 多余空白行
- PDF 页眉页脚特征
- 重复标点/乱码
"""

import re
from typing import Literal


def clean_text(raw: str, source_type: Literal["jd", "resume"] = "resume") -> str:
    """清洗原始文本，去噪但保留语义结构

    Args:
        raw: 原始文本
        source_type: "jd"（JD 文本）或 "resume"（简历文本）
    """
    if not raw:
        return ""

    text = raw

    # 1. 移除 Unicode 控制字符（保留常用空白）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 2. 统一换行为 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 移除 PDF 常见乱码字符
    text = re.sub(r"[�]+", "", text)

    # 4. 合并连续空行（3+ → 2）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. 每行去首尾空白，去除纯空白行
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    text = "\n".join(lines)

    # 6. 合并行内连续空格
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[\t]{2,}", "\t", text)

    # 7. 移除页眉页脚常见模式（PDF 解析残留）
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)  # 纯数字行
    text = re.sub(r"^Page \d+ of \d+$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^第\s*\d+\s*页", "", text, flags=re.MULTILINE)
    # 连续出现的相同短行（水印特征）— 保留第一次
    seen_short = set()
    deduped = []
    for line in text.split("\n"):
        s = line.strip()
        if len(s) <= 10:
            if s in seen_short:
                continue
            seen_short.add(s)
        deduped.append(line)
    text = "\n".join(deduped)

    # 8. 简历特有处理
    if source_type == "resume":
        # 移除常见分隔线
        text = re.sub(r"^[-=_]{5,}$", "", text, flags=re.MULTILINE)
        # 移除邮箱/电话格式异常（保留正常格式）
        # 不做过度清洗，保留原文完整性

    # 9. 移除开头结尾空行
    text = text.strip()

    return text
