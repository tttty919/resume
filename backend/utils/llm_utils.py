"""共享 LLM 工具函数 —— 所有 Skill 节点和 dev_server 复用"""

import json
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

from backend.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from backend.core.exceptions import ParseException

_BACKEND_DIR = Path(__file__).parents[1]


def create_llm(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    temperature: float = 0,
    max_tokens: int = 8192,
    request_timeout: float = 300,
    enable_thinking: bool = False,
) -> ChatOpenAI:
    """统一创建 LLM 实例

    enable_thinking: 开启 DeepSeek 思考模式（thinking.type=enabled）。
    思考 token 会增加延迟但提升复杂判断的准确率，适合匹配/风险评估等关键步骤。
    """
    model_kwargs: dict = {}
    if enable_thinking:
        model_kwargs["thinking"] = {"type": "enabled"}
        # 思考 tokens 不展示给用户但会计入输出，需扩大上限
        if max_tokens < 16384:
            max_tokens = 16384

    return ChatOpenAI(
        model=model or DEEPSEEK_MODEL,
        api_key=SecretStr(api_key) if api_key else SecretStr(DEEPSEEK_API_KEY),
        base_url=base_url or DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        model_kwargs=model_kwargs,
    )


def load_prompt(relative_path: str) -> str:
    """加载 prompt 模板文件，路径相对于 backend/ 根目录，如 'skills/jd_parser/prompt.txt'"""
    prompt_path = _BACKEND_DIR / relative_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_few_shots(relative_dir: str) -> str:
    """加载 few_shots/ 目录下所有示例文件，按文件名排序拼接成一段文本

    目录不存在或为空时返回空字符串。
    """
    dir_path = _BACKEND_DIR / relative_dir
    if not dir_path.exists():
        return ""
    parts = [f.read_text(encoding="utf-8") for f in sorted(dir_path.glob("*.md"))]
    return "\n\n".join(parts)


def build_chain(prompt_filename: str, few_shots_dir: str | None = None, **llm_kwargs) -> Runnable:
    """加载 prompt 模板（可选拼接 few-shot 示例）并构建 LLM Chain"""
    prompt_text = load_prompt(prompt_filename)
    if few_shots_dir:
        few_shot_text = load_few_shots(few_shots_dir)
        if few_shot_text:
            prompt_text = f"{prompt_text}\n\n{few_shot_text}"
    llm = create_llm(**llm_kwargs)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return prompt | llm


def parse_llm_json(text: str) -> dict:
    """解析 LLM 返回的 JSON 字符串

    按顺序尝试：
    1. 去掉 ```json / ``` 围栏
    2. json.loads 直接解析
    3. 匹配第一个完整 JSON 对象 { ... }
    4. 仍失败则抛出 ParseException
    """
    if not text or not text.strip():
        raise ParseException("LLM 返回为空", file_type="llm_response")

    text = text.strip()

    # 去掉 markdown 代码围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 匹配第一个完整 JSON 对象
    depth = 0
    start = text.find("{")
    if start >= 0:
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # 截断补全：JSON 不完整时尝试补上缺少的 } ] "
    stripped = text.rstrip()
    open_braces = stripped.count('{') - stripped.count('}')
    open_brackets = stripped.count('[') - stripped.count(']')
    in_string = stripped.count('"') % 2 != 0
    if in_string:
        stripped += '"'
    stripped += ']' * open_brackets
    stripped += '}' * open_braces
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    raise ParseException(
        f"LLM 返回无法解析为 JSON (已尝试截断补全): {text[:200]}",
        file_type="llm_response",
    )


def safe_pydantic_validate(parsed: dict, model_class, skill_name: str) -> dict:
    """Pydantic 校验，失败时降级为原始 dict 并记录 warning

    Args:
        parsed: LLM 返回并 json.parse 后的 dict
        model_class: Pydantic BaseModel 类
        skill_name: Skill 名称，用于日志

    Returns:
        model_dump() 后的 dict，或降级的原始 dict
    """
    try:
        validated = model_class(**parsed)
        return validated.model_dump()
    except Exception as e:
        logger.warning(f"[{skill_name}] Schema 校验失败，降级使用原始解析: {e}")
        return parsed if isinstance(parsed, dict) else {}
