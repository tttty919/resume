"""F1 Skill 开发调试服务器

启动: python -m backend.dev_server
然后浏览器打开 http://127.0.0.1:8765

支持:
- POST /api/parse-jd           Skill 1 — 粘贴 JD 文本
- POST /api/extract-resume     Skill 2 — 上传 PDF/DOCX 文件（文档解析 → LLM 提取）
- POST /api/extract-resume-text Skill 2 — 粘贴简历文本（调试用）
"""

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

import os as _os
if not _os.environ.get("HF_ENDPOINT"):
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not _os.environ.get("HF_HUB_OFFLINE"):
    _os.environ["HF_HUB_OFFLINE"] = "1"

# Windows asyncio + httpx 兼容：避免 ProactorEventLoop 导致的 [Errno 22] Invalid argument
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from starlette.responses import Response
from langchain_core.prompts import ChatPromptTemplate

from backend.core.config import get_settings
from backend.core.logger import setup_logger, get_logger
from backend.schemas.common import APIResponse
from backend.tools.document_parser import document_parser
from backend.tools.text_cleaner import clean_text
from backend.storage.resume_store import (
    get_cached as cache_get, save as cache_save,
    check as cache_check, get as cache_get_by_md5,
    list_by_job as cache_list_by_job, delete as cache_delete,
)
from backend.utils.llm_utils import create_llm, load_prompt, parse_llm_json

setup_logger("DEBUG")
log = get_logger()

app = FastAPI(title="F1 Skill Dev Server")
settings = get_settings()

# ── 多租户工作区隔离 ──────────────────────────────────────────
# 每个请求带 X-Space 头（前端注入）或 ?space= 查询参数，写进 ContextVar；
# jobs.json / 上传文件 / 简历缓存库都按 space 分目录，用户之间数据互不可见。
from urllib.parse import unquote as _unquote
from backend.utils.space import set_space, data_dir, upload_dir as _space_upload_dir


@app.middleware("http")
async def _space_middleware(request: Request, call_next):
    raw = request.headers.get("X-Space")
    if raw is not None:
        raw = _unquote(raw)               # 前端 encodeURIComponent 过，解回原文
    else:
        raw = request.query_params.get("space")  # EventSource 等无法带头时的兜底
    set_space(raw or "default")
    return await call_next(request)

# ── Agent router ─────────────────────────────────────────────
from backend.agent.router import router as agent_router
from backend.agent.session import get_session, get_queue, emit, AgentSession, CandidateSlot
app.include_router(agent_router)

# ── Screening router ─────────────────────────────────────────
from backend.screening.router import router as screening_router
app.include_router(screening_router)

# ── Chat (conversational agent) router ────────────────────────
from backend.chat.router import router as chat_router
app.include_router(chat_router)


# ── 通用 LLM 调用 ─────────────────────────────────────────

import asyncio as _asyncio


def _call_llm(prompt_file: str, template_vars: dict, api_key: str, base_url: str, model: str, retries: int = 2) -> dict:
    """通用 LLM 调用：加载 prompt → 填充模板 → 调用 DeepSeek → 解析 JSON（带重试）"""
    prompt_text = load_prompt(prompt_file)
    llm = create_llm(api_key=api_key, base_url=base_url, model=model)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm

    last_err = None
    for attempt in range(retries + 1):
        try:
            response = chain.invoke(template_vars)
            return parse_llm_json(response.content)
        except Exception as e:
            last_err = e
            if attempt < retries:
                log.warning(f"LLM 调用失败 (attempt {attempt+1}/{retries+1}): {e}, 重试中...")
                import time; time.sleep(2)
    raise last_err


async def _call_llm_async(prompt_file: str, template_vars: dict, api_key: str, base_url: str, model: str, retries: int = 2) -> dict:
    """异步版 LLM 调用，不阻塞事件循环"""
    return await _asyncio.to_thread(_call_llm, prompt_file, template_vars, api_key, base_url, model, retries)


# ── JDParser 规则兜底 ────────────────────────────────────────

def _compact_name(text: str, max_len: int = 10) -> str:
    text = re.sub(r"[：:，,。.；;（）()、]", " ", text).strip()
    text = re.sub(r"\s+", "", text)
    return text[:max_len] or "岗位要求"


def _keywords(text: str) -> list[str]:
    patterns = [
        "AI Coding", "AI coding", "Cursor", "Copilot", "AI PRD", "AI/Agent",
        "Agent", "API", "AI工具", "市场调研", "竞品分析", "用户洞察", "信息归纳",
        "产品方案", "原型", "上线", "指标体系", "效果评测", "数据解读", "归因分析",
        "用户反馈", "工作流", "人工智能", "软件工程", "计算机科学", "编程工具",
        "代码逻辑", "需求文档", "数据分析脚本", "问题排查", "大模型", "算法复现",
        "功能落地", "产品经理", "Owner",
    ]
    found = []
    lower = text.lower()
    for pattern in patterns:
        if pattern.lower() in lower and pattern not in found:
            found.append(pattern)
    if not found:
        found = re.findall(r"[A-Za-z][A-Za-z0-9+/.-]{1,}|[一-鿿]{2,6}", text)[:5]
    return found[:5]


def _extract_numbered(section: str) -> list[str]:
    items = []
    matches = list(re.finditer(r"(?:^|\n)\s*\d+[.、]\s*", section))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section)
        item = section[start:end].strip()
        item = re.sub(r"\n+", " ", item)
        item = re.sub(r"\s+", " ", item)
        if item:
            items.append(item)
    return items


def _section_between(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end_positions = [text.find(marker, start) for marker in end_markers if text.find(marker, start) >= 0]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end].strip()


def _fallback_parse_jd(raw_jd: str) -> dict:
    """Rule-based fallback when the LLM incorrectly returns an empty JD parse."""
    if "岗位职责" not in raw_jd and "任职要求" not in raw_jd:
        return {"requirements": [], "parsing_notes": "未识别到岗位职责或任职要求"}

    responsibilities = _extract_numbered(_section_between(raw_jd, "岗位职责", ["任职要求", "AI能力要求"]))
    qualifications = _extract_numbered(_section_between(raw_jd, "任职要求", ["AI能力要求"]))
    ai_section = _section_between(raw_jd, "AI能力要求", [])

    requirements = []

    def add_req(description: str, req_type: str = "must", importance: str = "medium"):
        rid = f"req-{len(requirements) + 1:03d}"
        requirements.append({
            "id": rid,
            "name": _compact_name(description),
            "description": description,
            "type": req_type,
            "importance": importance,
            "keywords": _keywords(description),
        })

    for item in responsibilities:
        importance = "high" if any(key in item for key in ["AI", "AI Coding", "AI PRD", "上线", "指标"]) else "medium"
        add_req(item, "must", importance)

    for item in qualifications:
        importance = "high" if any(key in item for key in ["Cursor", "Copilot", "AI Coding", "大模型", "具体的AI应用案例"]) else "medium"
        add_req(item, "must", importance)

    if ai_section:
        ai_desc = re.sub(r"\s+", " ", ai_section).strip()
        if ai_desc:
            add_req(ai_desc, "must", "high")

    return {
        "requirements": requirements,
        "parsing_notes": "LLM 未返回有效要求，已按岗位职责、任职要求和 AI 能力要求进行规则兜底解析",
    }


def _ensure_jd_parse(raw_jd: str, result: dict) -> dict:
    requirements = result.get("requirements", []) if isinstance(result, dict) else []
    if requirements:
        return result
    fallback = _fallback_parse_jd(raw_jd)
    return fallback if fallback.get("requirements") else result


# ── Skill 1: JDParser ──────────────────────────────────────

@app.post("/api/parse-jd")
async def parse_jd(request: Request):
    """Skill 1: 粘贴 JD 文本 → LLM 结构化解析"""
    body = await request.json()
    raw_jd = body.get("jd", "").strip()
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not raw_jd:
        return APIResponse(success=False, message="JD 文本不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        raw_jd = clean_text(raw_jd, "jd")
        result = await _call_llm_async("skills/jd_parser/prompt.txt", {"raw_jd": raw_jd}, api_key, base_url, model)
        result = _ensure_jd_parse(raw_jd, result)
        log.info(f"JDParser 成功: {len(result.get('requirements', []))} 条要求")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"JDParser 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


@app.post("/api/judge-jd")
async def judge_jd(request: Request):
    """Skill 1 质量评审: JD 原文 + Parser 输出 → LLM 打分"""
    body = await request.json()
    raw_jd = body.get("jd", "").strip()
    parser_output = body.get("parser_output", {})
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not raw_jd:
        return APIResponse(success=False, message="JD 文本不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        result = await _call_llm_async(
            "skills/jd_parser/judge_prompt.txt",
            {"raw_jd": raw_jd, "parser_output": json.dumps(parser_output, ensure_ascii=False, indent=2)},
            api_key, base_url, model
        )
        log.info(f"JD Judge: {result.get('total_score', '?')} 分 - {result.get('verdict', '?')}")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"JD Judge 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── Skill 2: ResumeExtractor (文件上传) ─────────────────────

@app.post("/api/extract-resume")
async def extract_resume(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    """Skill 2: 上传 PDF/DOCX → 文档解析 → LLM 结构化提取"""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model

    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()
    if not file.filename:
        return APIResponse(success=False, message="未选择文件").model_dump()

    # 保存上传文件到临时目录
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc"):
        return APIResponse(success=False, message=f"不支持的文件格式: {suffix}，支持 PDF/DOCX").model_dump()

    upload_dir = _space_upload_dir(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"tmp_{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Check SQLite cache first
        cached = cache_get(str(tmp_path))
        if cached and cached.get("raw_text") and cached.get("parsed_resume") and isinstance(cached["parsed_resume"], dict) and cached["parsed_resume"].get("basic_info"):
            log.info(f"[Cache HIT] {file.filename} — 跳过文档解析+LLM提取")
            text = cached["raw_text"]
            result = cached["parsed_resume"]
            images_count = 0
        else:
            # Step 1: 文档解析
            log.info(f"文档解析开始: {file.filename}")
            text, images = document_parser.parse(str(tmp_path))
            text = clean_text(text, "resume")
            images_count = len(images)

            if not text.strip():
                return APIResponse(success=False, message="文档解析结果为空，请检查文件内容").model_dump()

            # Step 2: Skill 2 — LLM 提取
            log.info(f"Skill 2 开始提取: {file.filename} ({len(text)} chars)")
            result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": text}, api_key, base_url, model)

            # Save to cache
            try:
                cache_save(str(tmp_path), text, result)
            except Exception as e:
                log.warning(f"[Cache] 保存失败 ({file.filename}): {e}")

        skills_count = len(result.get("skills", []))
        name = result.get("basic_info", {}).get("name", "未知")
        log.info(f"ResumeExtractor 成功: {name}, {skills_count} 技能")

        return APIResponse(
            success=True,
            message=f"解析完成: {len(text)} 字, {images_count} 张图片",
            data={
                "file_name": file.filename,
                "text_length": len(text),
                "images_count": images_count,
                "extracted": result,
                "raw_text": text,
            },
        ).model_dump()

    except Exception as e:
        log.error(f"ResumeExtractor 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()
    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink()


# ── Skill 2: ResumeExtractor (文本粘贴 — 调试用) ────────────

@app.post("/api/extract-resume-text")
async def extract_resume_text(request: Request):
    """Skill 2 (调试): 粘贴简历文本 → 跳过文档解析，直接 LLM 提取"""
    body = await request.json()
    raw_resume = body.get("resume", "").strip()
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not raw_resume:
        return APIResponse(success=False, message="简历文本不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        raw_resume = clean_text(raw_resume, "resume")
        result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": raw_resume}, api_key, base_url, model)
        name = result.get("basic_info", {}).get("name", "未知")
        log.info(f"ResumeExtractor(文本) 成功: {name}")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"ResumeExtractor(文本) 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── Skill 3: SemanticMatcher（合并原 Skill 3+4）────────────

@app.post("/api/match")
async def semantic_match(request: Request):
    """Skill 3: 传入 requirements + resume JSON + raw_resume_text → 逐项语义匹配 + 证据摘录 + 机械校验"""
    body = await request.json()
    requirements = body.get("requirements", [])
    resume = body.get("resume", {})
    raw_resume_text = body.get("raw_resume_text", "")
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    # 防御：前端可能发来 JSON 字符串而非解析后的对象
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except (json.JSONDecodeError, TypeError):
            requirements = []
    if isinstance(resume, str):
        try:
            resume = json.loads(resume)
        except (json.JSONDecodeError, TypeError):
            resume = {}

    if not requirements:
        return APIResponse(success=False, message="requirements 不能为空").model_dump()
    if not resume:
        return APIResponse(success=False, message="resume 不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        from backend.skills.semantic_matcher.node import match
        log.info(f"SemanticMatcher 开始: {len(requirements)} 条要求, 候选人 {resume.get('basic_info', {}).get('name', '?') if isinstance(resume, dict) else '?'}")
        result = await match(requirements, resume, raw_resume_text, api_key, base_url, model)
        matches = result.get("matches", [])
        satisfied = sum(1 for m in matches if isinstance(m, dict) and m.get("status") == "satisfied")
        verified = sum(1 for m in matches if isinstance(m, dict) and m.get("validation", {}).get("verified"))
        log.info(f"SemanticMatcher 完成: {satisfied}/{len(matches)} 满足, {verified} 证据验证通过")
        return APIResponse(
            success=True,
            message=f"{satisfied}/{len(matches)} 项满足, {verified} 证据验证通过",
            data=result,
        ).model_dump()
    except Exception as e:
        import traceback
        log.error(f"SemanticMatcher 失败: {e}\n{traceback.format_exc()}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── 证据校验调试端点（纯代码，非 LLM）──────────────────────

@app.post("/api/judge-evidence")
async def judge_evidence(request: Request):
    """证据校验质量评审: 简历原文 + matches + 提取结果 → LLM 打分"""
    body = await request.json()
    raw_resume = body.get("raw_resume", "")
    matches = body.get("matches", [])
    extractor_output = body.get("extractor_output", {})
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        result = await _call_llm_async(
            "skills/semantic_matcher/judge_prompt.txt",
            {
                "raw_resume": raw_resume,
                "matches_json": json.dumps(matches, ensure_ascii=False, indent=2),
                "extractor_output": json.dumps(extractor_output, ensure_ascii=False, indent=2),
            },
            api_key, base_url, model,
        )
        log.info(f"Evidence Judge: {result.get('total_score', '?')} 分 - {result.get('verdict', '?')}")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"Evidence Judge 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── Skill 4: RiskAnalyzer ────────────────────────────────────

@app.post("/api/analyze-risk")
async def analyze_risk(request: Request):
    """Skill 4: 传入 requirements + validated_items → 综合风险评估"""
    body = await request.json()
    requirements = body.get("requirements", [])
    validated_items = body.get("validated_items", [])
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not requirements:
        return APIResponse(success=False, message="requirements 不能为空").model_dump()
    if not validated_items:
        return APIResponse(success=False, message="validated_items 不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    # Support JSON strings from frontend
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="requirements JSON 解析失败").model_dump()
    if isinstance(validated_items, str):
        try:
            validated_items = json.loads(validated_items)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="validated_items JSON 解析失败").model_dump()

    try:
        from backend.skills.risk_analyzer.node import analyze
        log.info(f"RiskAnalyzer 开始: {len(requirements)} 条要求, {len(validated_items)} 项证据")
        result = await analyze(requirements, validated_items, api_key, base_url, model)
        analysis = result.get("analysis", {})
        log.info(f"RiskAnalyzer 完成: {len(analysis.get('key_risks',[]))} 风险, {len(analysis.get('core_advantages',[]))} 优势")
        return APIResponse(
            success=True,
            message=f"{len(analysis.get('core_advantages',[]))} 优势, {len(analysis.get('key_risks',[]))} 风险",
            data=result,
        ).model_dump()
    except Exception as e:
        log.error(f"RiskAnalyzer 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


@app.post("/api/judge-risk")
async def judge_risk(request: Request):
    """Skill 4 质量评审: validated_items + 分析结果 → LLM 打分"""
    body = await request.json()
    validated_items = body.get("validated_items", [])
    analyzer_output = body.get("analyzer_output", {})
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    if isinstance(validated_items, str):
        try:
            validated_items = json.loads(validated_items)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="validated_items JSON 解析失败").model_dump()

    try:
        result = await _call_llm_async(
            "skills/risk_analyzer/judge_prompt.txt",
            {
                "validated_items_json": json.dumps(validated_items, ensure_ascii=False, indent=2),
                "analyzer_output": json.dumps(analyzer_output, ensure_ascii=False, indent=2),
            },
            api_key, base_url, model,
        )
        log.info(f"Risk Judge: {result.get('total_score', '?')} 分 - {result.get('verdict', '?')}")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"Risk Judge 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


@app.post("/api/judge-recommendation")
async def judge_recommendation(request: Request):
    """Skill 5 质量评审: requirements + validated_items + risk_analysis + scoring + generator_output → LLM 打分"""
    body = await request.json()
    requirements = body.get("requirements", [])
    validated_items = body.get("validated_items", [])
    risk_analysis = body.get("risk_analysis", {})
    scoring = body.get("scoring", {})
    generator_output = body.get("generator_output", {})
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    for field, name in [(requirements, "requirements"), (validated_items, "validated_items")]:
        if isinstance(field, str):
            try:
                field = json.loads(field)
            except json.JSONDecodeError:
                return APIResponse(success=False, message=f"{name} JSON 解析失败").model_dump()

    if isinstance(risk_analysis, str):
        try:
            risk_analysis = json.loads(risk_analysis)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="risk_analysis JSON 解析失败").model_dump()
    if isinstance(scoring, str):
        try:
            scoring = json.loads(scoring)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="scoring JSON 解析失败").model_dump()
    if isinstance(generator_output, str):
        try:
            generator_output = json.loads(generator_output)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="generator_output JSON 解析失败").model_dump()

    try:
        result = await _call_llm_async(
            "skills/recommendation_gen/judge_prompt.txt",
            {
                "requirements_json": json.dumps(requirements, ensure_ascii=False, indent=2),
                "validated_items_json": json.dumps(validated_items, ensure_ascii=False, indent=2),
                "risk_analysis_json": json.dumps(risk_analysis, ensure_ascii=False, indent=2),
                "scoring_json": json.dumps(scoring, ensure_ascii=False, indent=2),
                "generator_output": json.dumps(generator_output, ensure_ascii=False, indent=2),
            },
            api_key, base_url, model,
        )
        log.info(f"Recommendation Judge: {result.get('total_score', '?')} 分 - {result.get('verdict', '?')}")
        return APIResponse(success=True, data=result).model_dump()
    except Exception as e:
        log.error(f"Recommendation Judge 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── Skill 5: RecommendationGen ──────────────────────────────

@app.post("/api/generate-recommendation")
async def generate_recommendation(request: Request):
    """Skill 5: 传入 requirements + matches + validated_items + risk_analysis → 规则评分 + LLM 推荐"""
    body = await request.json()
    # UI 双输入模式: context = {requirements, matches, validated_items} 合并 JSON
    context = body.get("context", {})
    if context:
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except json.JSONDecodeError:
                return APIResponse(success=False, message="context JSON 解析失败").model_dump()
        requirements = context.get("requirements", [])
        matches = context.get("matches", [])
        validated_items = context.get("validated_items", [])
    else:
        requirements = body.get("requirements", [])
        matches = body.get("matches", [])
        validated_items = body.get("validated_items", [])
    risk_analysis = body.get("risk_analysis", {})
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not requirements:
        return APIResponse(success=False, message="requirements 不能为空").model_dump()
    if not matches:
        return APIResponse(success=False, message="matches 不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    # Support JSON strings from frontend
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="requirements JSON 解析失败").model_dump()
    if isinstance(matches, str):
        try:
            matches = json.loads(matches)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="matches JSON 解析失败").model_dump()
    if isinstance(validated_items, str):
        try:
            validated_items = json.loads(validated_items)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="validated_items JSON 解析失败").model_dump()
    if isinstance(risk_analysis, str):
        try:
            risk_analysis = json.loads(risk_analysis)
        except json.JSONDecodeError:
            return APIResponse(success=False, message="risk_analysis JSON 解析失败").model_dump()

    try:
        from backend.skills.recommendation_gen.node import generate
        log.info(f"RecommendationGen 开始: {len(matches)} 匹配项 → 规则评分 → LLM 推荐")
        result = await generate(requirements, matches, validated_items, risk_analysis, api_key, base_url, model)
        s = result.get("scoring", {})
        summary = result.get("summary", {})
        log.info(f"RecommendationGen 完成: {summary.get('recommendation', '?')} (得分: {s.get('overall_score', '?')})")
        return APIResponse(
            success=True,
            message=f"{summary.get('recommendation', '?')} (综合得分: {s.get('overall_score', 0)})",
            data=result,
        ).model_dump()
    except Exception as e:
        log.error(f"RecommendationGen 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── 反馈回路: HR 确认同义词 → 写入 ChromaDB ─────────────

@app.post("/api/feedback")
async def feedback(request: Request):
    """HR 确认匹配正确后，将简历中的同义表述写入 ChromaDB 知识库"""
    body = await request.json()
    requirement_name = body.get("requirement_name", "").strip()
    new_synonym = body.get("synonym", "").strip()

    if not requirement_name:
        return APIResponse(success=False, message="requirement_name 不能为空").model_dump()
    if not new_synonym:
        return APIResponse(success=False, message="synonym 不能为空").model_dump()

    from backend.skills.semantic_matcher.tools import learn
    result = learn(requirement_name, new_synonym)
    return APIResponse(success=result["added"], message=result["message"], data=result).model_dump()


# ── 岗位 CRUD (JSON 文件持久化) ───────────────────────

_job_resume_counts: dict[str, int] = {}
_job_screened_counts: dict[str, int] = {}


def _jobs_file() -> Path:
    """当前 space 专属的 jobs.json（多租户隔离）。"""
    return data_dir() / "jobs.json"


def _load_jobs() -> list[dict]:
    jf = _jobs_file()
    if jf.exists():
        try:
            return json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_jobs(jobs: list[dict]) -> None:
    jf = _jobs_file()
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/jobs")
async def list_jobs():
    jobs = _load_jobs()
    for job in jobs:
        jid = job.get("id", "")
        cached = cache_list_by_job(jid)
        job["resumeCount"] = len(cached)
        job["screenedCount"] = _job_screened_counts.get(jid, 0)
    return APIResponse(success=True, data=jobs).model_dump()


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    jobs = _load_jobs()
    jobs.insert(0, body)
    _save_jobs(jobs)
    return APIResponse(success=True, data=body).model_dump()


@app.put("/api/jobs/{job_id}")
async def update_job(job_id: str, request: Request):
    body = await request.json()
    jobs = _load_jobs()
    for i, j in enumerate(jobs):
        if j.get("id") == job_id:
            jobs[i] = body
            _save_jobs(jobs)
            return APIResponse(success=True, data=body).model_dump()
    return APIResponse(success=False, message="岗位不存在").model_dump()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    jobs = _load_jobs()
    jobs = [j for j in jobs if j.get("id") != job_id]
    _save_jobs(jobs)
    return APIResponse(success=True, message="已删除").model_dump()


# ── Resume Cache API ─────────────────────────────────

@app.post("/api/resumes/check")
async def resume_check(request: Request):
    body = await request.json()
    md5 = (body.get("md5") or "").strip().lower()
    if not md5 or len(md5) != 32:
        return APIResponse(success=False, message="无效的 MD5 哈希").model_dump()
    entry = cache_check(md5)
    if entry:
        return APIResponse(success=True, data={"cached": True, **entry}).model_dump()
    return APIResponse(success=True, data={"cached": False}).model_dump()


@app.get("/api/resumes/{md5}")
async def resume_get(md5: str):
    md5 = md5.strip().lower()
    if len(md5) != 32:
        return APIResponse(success=False, message="无效的 MD5 哈希").model_dump()
    entry = cache_get_by_md5(md5)
    if entry:
        return APIResponse(success=True, data=entry).model_dump()
    return APIResponse(success=False, message="未找到缓存简历").model_dump()


@app.get("/api/resumes/job/{job_id}")
async def resume_list_by_job(job_id: str):
    entries = cache_list_by_job(job_id)
    return APIResponse(success=True, data={"resumes": entries, "count": len(entries)}).model_dump()


@app.delete("/api/resumes/{md5}")
async def resume_delete(md5: str):
    md5 = md5.strip().lower()
    if len(md5) != 32:
        return APIResponse(success=False, message="无效的 MD5 哈希").model_dump()
    deleted = cache_delete(md5)
    if deleted:
        return APIResponse(success=True, message="已删除").model_dump()
    return APIResponse(success=False, message="未找到缓存简历").model_dump()


# ── HR 人工复核: 修改单项结论 + 审计记录 ─────────────

def _apply_hr_override(
    target_item: dict,
    operator: str,
    new_status: str,
    reason: str,
    supplementary_evidence: str,
) -> dict:
    """把一条 HR 改判应用到 validated_items 里对应的 item 上（原地修改），返回审计记录。"""
    from datetime import datetime, timezone

    before_status = target_item.get("status", "cannot_judge")

    override_record = {
        "requirement_id": target_item.get("requirement_id", ""),
        "operator": operator,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "before_status": before_status,
        "after_status": new_status,
        "reason": reason,
        "supplementary_evidence": supplementary_evidence,
    }
    target_item["hr_override"] = override_record
    target_item["status"] = new_status
    target_item["needs_human_review"] = False

    # 如果 HR 提供了补充证据，追加为 hr_verified_evidence
    if supplementary_evidence:
        sources = target_item.get("evidence_sources", [])
        sources.append({
            "content": supplementary_evidence,
            "source_type": "hr_verified_evidence",
            "source_location": f"HR确认({operator})",
            "verified": True,
        })
        target_item["evidence_sources"] = sources

    return override_record


@app.post("/api/hr-override")
async def hr_override(request: Request):
    """HR 确认/驳回某项要求的判断结果，记录审计信息，触发局部重算"""
    body = await request.json()
    requirement_id = body.get("requirement_id", "").strip()
    operator = body.get("operator", "").strip()
    new_status = body.get("new_status", "").strip()
    reason = body.get("reason", "").strip()
    supplementary_evidence = body.get("supplementary_evidence", "").strip()

    # 需要传入当前筛选结果用于局部更新
    validated_items = body.get("validated_items", [])
    requirements = body.get("requirements", [])
    matches = body.get("matches", [])

    if not requirement_id:
        return APIResponse(success=False, message="requirement_id 不能为空").model_dump()
    if not operator:
        return APIResponse(success=False, message="operator（操作人）不能为空").model_dump()
    if new_status not in ("satisfied", "not_satisfied", "cannot_judge"):
        return APIResponse(success=False, message="new_status 必须是 satisfied/not_satisfied/cannot_judge").model_dump()
    if not reason:
        return APIResponse(success=False, message="reason（修改原因）不能为空").model_dump()

    target_item = None
    for item in validated_items:
        if item.get("requirement_id") == requirement_id:
            target_item = item
            break

    if not target_item:
        return APIResponse(success=False, message=f"未找到 requirement_id={requirement_id}").model_dump()

    before_status = target_item.get("status", "cannot_judge")
    override_record = _apply_hr_override(target_item, operator, new_status, reason, supplementary_evidence)

    # 局部重算: 调用 controller.rerun_affected()
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    rerun_result = {}
    if requirements and matches and validated_items:
        try:
            from backend.workflows.controller import rerun_affected
            rerun_result = await rerun_affected(requirements, matches, validated_items, api_key, base_url, model)
            log.info(f"HR Override 局部重算完成: {requirement_id} → {new_status}")
        except Exception as e:
            log.error(f"HR Override 局部重算失败: {e}")
            rerun_result = {"rerun_error": str(e)}
    else:
        log.info(f"HR Override 仅记录审计（无完整上下文，跳过重算）: {requirement_id} → {new_status}")

    return APIResponse(
        success=True,
        message=f"已更新 {requirement_id}: {before_status} → {new_status}",
        data={
            "override_record": override_record,
            "updated_item": target_item,
            "validated_items": validated_items,
            **rerun_result,
        },
    ).model_dump()


@app.post("/api/hr-override-batch")
async def hr_override_batch(request: Request):
    """HR 一次性提交多条改判，只触发一次局部重算（Skill4风险分析 + Skill5推荐生成）"""
    body = await request.json()
    operator = body.get("operator", "").strip()
    overrides = body.get("overrides", [])

    validated_items = body.get("validated_items", [])
    requirements = body.get("requirements", [])
    matches = body.get("matches", [])

    if not operator:
        return APIResponse(success=False, message="operator（操作人）不能为空").model_dump()
    if not isinstance(overrides, list) or not overrides:
        return APIResponse(success=False, message="overrides 不能为空").model_dump()

    override_records = []
    for ov in overrides:
        requirement_id = (ov.get("requirement_id") or "").strip()
        new_status = (ov.get("new_status") or "").strip()
        reason = (ov.get("reason") or "").strip()
        supplementary_evidence = (ov.get("supplementary_evidence") or "").strip()

        if not requirement_id:
            return APIResponse(success=False, message="overrides 中存在空的 requirement_id").model_dump()
        if new_status not in ("satisfied", "not_satisfied", "cannot_judge"):
            return APIResponse(success=False, message=f"{requirement_id}: new_status 必须是 satisfied/not_satisfied/cannot_judge").model_dump()
        if not reason:
            return APIResponse(success=False, message=f"{requirement_id}: reason（修改原因）不能为空").model_dump()

        target_item = next((item for item in validated_items if item.get("requirement_id") == requirement_id), None)
        if not target_item:
            return APIResponse(success=False, message=f"未找到 requirement_id={requirement_id}").model_dump()

        override_records.append(_apply_hr_override(target_item, operator, new_status, reason, supplementary_evidence))

    # 局部重算: 无论这一批改了几条，只调用一次 rerun_affected()
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    rerun_result = {}
    if requirements and matches and validated_items:
        try:
            from backend.workflows.controller import rerun_affected
            rerun_result = await rerun_affected(requirements, matches, validated_items, api_key, base_url, model)
            log.info(f"HR 批量 Override 局部重算完成: {len(override_records)} 条")
        except Exception as e:
            log.error(f"HR 批量 Override 局部重算失败: {e}")
            rerun_result = {"rerun_error": str(e)}
    else:
        log.info(f"HR 批量 Override 仅记录审计（无完整上下文，跳过重算）: {len(override_records)} 条")

    return APIResponse(
        success=True,
        message=f"已批量更新 {len(override_records)} 项",
        data={
            "override_records": override_records,
            "validated_items": validated_items,
            **rerun_result,
        },
    ).model_dump()


# ── Pipeline: Skill 1→5 全流程串联（测试用）─────────────

@app.post("/api/pipeline")
async def pipeline(request: Request):
    """JD 文本 + 简历文本 → Skill 1→5 全流程

    支持两种模式:
    1. 传入 jd + resume → 先解析 JD，再匹配简历
    2. 传入 requirements + resume → 跳过 JD 解析，直接用已有 requirements 匹配
       （推荐：JD 解析一次后复用，保证多份简历用同一把"尺子"衡量）
    """
    body = await request.json()
    jd_text = body.get("jd", "").strip()
    resume_text = body.get("resume", "").strip()
    requirements = body.get("requirements", None)  # 可选：复用已解析的 requirements
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not requirements and not jd_text:
        return APIResponse(success=False, message="JD 文本或 requirements 不能同时为空").model_dump()
    if not resume_text:
        return APIResponse(success=False, message="简历文本不能为空").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    try:
        from backend.skills.semantic_matcher.node import match
        from backend.skills.risk_analyzer.node import analyze as analyze_risk
        from backend.skills.recommendation_gen.node import generate as generate_recommendation
        from backend.workflows.controller import ScreeningAgent

        pipeline_log = []

        # Step 1: JDParser（如果已有 requirements 则跳过）
        if requirements:
            if isinstance(requirements, str):
                try:
                    requirements = json.loads(requirements)
                except json.JSONDecodeError:
                    return APIResponse(success=False, message="requirements JSON 解析失败").model_dump()
            pipeline_log.append(f"JDParser: 复用已有 {len(requirements)} 条要求")
        else:
            log.info("Pipeline Step 1/5: JDParser")
            jd_text = clean_text(jd_text, "jd")
            jd_result = await _call_llm_async("skills/jd_parser/prompt.txt", {"raw_jd": jd_text}, api_key, base_url, model)
            jd_result = _ensure_jd_parse(jd_text, jd_result)
            requirements = jd_result.get("requirements", [])
            pipeline_log.append(f"JDParser: {len(requirements)} 条要求")

        # Step 2: ResumeExtractor
        log.info("Pipeline Step 2/5: ResumeExtractor")
        resume_text = clean_text(resume_text, "resume")
        resume_result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": resume_text}, api_key, base_url, model)
        name = resume_result.get("basic_info", {}).get("name", "未知")
        pipeline_log.append(f"ResumeExtractor: {name}, {len(resume_result.get('skills', []))} 技能")

        # Step 3: SemanticMatcher + Evidence（合并：判断 + 证据摘录 + 机械校验）
        log.info("Pipeline Step 3/5: SemanticMatcher + Evidence")
        match_result = await match(requirements, resume_result, resume_text, api_key, base_url, model)
        matches = match_result.get("matches", [])
        satisfied = sum(1 for m in matches if m.get("status") == "satisfied")
        verified = sum(1 for m in matches if m.get("validation", {}).get("verified"))
        downgraded = sum(1 for m in matches if m.get("status_changed"))
        pipeline_log.append(f"SemanticMatcher: {satisfied}/{len(matches)} 满足, {verified} 验证通过" + (f", {downgraded} 降级" if downgraded else ""))
        items = matches  # 合并后 validated_items 就是 matches

        # Step 4: RiskAnalyzer
        log.info("Pipeline Step 4/5: RiskAnalyzer")
        risk_result = await analyze_risk(requirements, items, api_key, base_url, model)
        analysis = risk_result.get("analysis", {})
        risks_count = len(analysis.get("key_risks", []))
        advantages_count = len(analysis.get("core_advantages", []))
        pipeline_log.append(f"RiskAnalyzer: {advantages_count} 优势, {risks_count} 风险")

        # Step 5: RecommendationGen
        log.info("Pipeline Step 5/5: RecommendationGen")
        rec_result = await generate_recommendation(requirements, matches, items, analysis, api_key, base_url, model)
        scoring = rec_result.get("scoring", {})
        summary = rec_result.get("summary", {})
        pipeline_log.append(f"RecommendationGen: {summary.get('recommendation', '?')} (得分: {scoring.get('overall_score', '?')})")

        log.success(f"Pipeline 完成: {' | '.join(pipeline_log)}")

        return APIResponse(
            success=True,
            message=" | ".join(pipeline_log),
            data={
                "requirements": requirements,
                "resume": resume_result,
                "matches": matches,
                "validated_items": items,
                "risk_analysis": analysis,
                "recommendation": rec_result,
                "pipeline_log": pipeline_log,
            },
        ).model_dump()
    except Exception as e:
        log.error(f"Pipeline 失败: {e}")
        return APIResponse(success=False, message=str(e)).model_dump()


# ── 全流程筛选: JD + 多份简历 → 对比排序 ──────────────────

@app.post("/api/screen")
async def screen(
    jd_text: str = Form(""),
    files: list[UploadFile] = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    """全流程筛选: 上传 JD 文本 + 多份 PDF/DOCX 简历 → 完整 5-Skill 流程 → 排名对比"""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model

    if not jd_text.strip():
        return APIResponse(success=False, message="JD 文本不能为空").model_dump()
    if not files:
        return APIResponse(success=False, message="请至少上传一份简历").model_dump()
    if not api_key:
        return APIResponse(success=False, message="请填写 API Key").model_dump()

    from backend.skills.semantic_matcher.node import match
    from backend.skills.risk_analyzer.node import analyze as analyze_risk
    from backend.skills.recommendation_gen.node import generate as generate_recommendation

    upload_dir = _space_upload_dir(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # JD 解析（只跑一次）
    jd_text = clean_text(jd_text, "jd")
    log.info(f"全流程筛选: 解析 JD ({len(jd_text)} chars)")
    try:
        jd_result = await _call_llm_async("skills/jd_parser/prompt.txt", {"raw_jd": jd_text}, api_key, base_url, model)
        jd_result = _ensure_jd_parse(jd_text, jd_result)
        requirements = jd_result.get("requirements", [])
        log.info(f"JD 解析完成: {len(requirements)} 条要求")
    except Exception as e:
        return APIResponse(success=False, message=f"JD 解析失败: {e}").model_dump()

    # 先保存所有文件到磁盘（UploadFile 只能读一次）
    total = len(files)
    file_records = []
    for idx, file in enumerate(files):
        filename = file.filename or f"resume_{idx}"
        tmp_path = upload_dir / f"screen_{idx}_{filename}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_records.append({"idx": idx, "filename": filename, "path": tmp_path})

    # 并发限制：最多同时 3 个 LLM 调用链
    semaphore = asyncio.Semaphore(3)

    async def process_one(record: dict) -> dict:
        """处理单份简历的完整 5-Skill 流程"""
        async with semaphore:
            filename = record["filename"]
            tmp_path = record["path"]
            idx = record["idx"]
            log.info(f"[{idx+1}/{total}] 开始处理: {filename}")

            try:
                # Check cache
                cached = cache_get(str(tmp_path))
                if cached and cached.get("raw_text") and cached.get("parsed_resume") and isinstance(cached["parsed_resume"], dict) and cached["parsed_resume"].get("basic_info"):
                    text = cached["raw_text"]
                    resume_result = cached["parsed_resume"]
                    images = []
                else:
                    # 文档解析
                    text, images = document_parser.parse(str(tmp_path))
                    text = clean_text(text, "resume")
                    if not text.strip():
                        return {"file_name": filename, "error": "文档解析结果为空"}

                    # Skill 2: 简历提取
                    resume_result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": text}, api_key, base_url, model)
                    try:
                        cache_save(str(tmp_path), text, resume_result)
                    except Exception as e:
                        log.warning(f"[Cache] 保存失败 ({filename}): {e}")

                name = resume_result.get("basic_info", {}).get("name", filename)

                # Skill 3: 语义匹配 + 证据摘录 + 机械校验
                match_result = await match(requirements, resume_result, text, api_key, base_url, model)
                matches = match_result.get("matches", [])
                items = matches  # 合并后 validated_items 就是 matches

                # Skill 4: 风险分析
                risk_result = await analyze_risk(requirements, items, api_key, base_url, model)
                analysis = risk_result.get("analysis", {})

                # Skill 5: 推荐生成
                rec_result = await generate_recommendation(requirements, matches, items, analysis, api_key, base_url, model)
                scoring = rec_result.get("scoring", {})
                summary = rec_result.get("summary", {})

                log.info(f"[{idx+1}/{total}] {name}: {summary.get('recommendation', '?')} "
                         f"(得分: {scoring.get('overall_score', '?')})")

                return {
                    "file_name": filename,
                    "name": name,
                    "text_length": len(text),
                    "requirements": requirements,
                    "resume": resume_result,
                    "matches": matches,
                    "validated_items": items,
                    "risk_analysis": analysis,
                    "recommendation": rec_result,
                }
            except Exception as e:
                log.error(f"[{idx+1}/{total}] {filename} 处理失败: {e}")
                return {"file_name": filename, "name": filename, "error": str(e)}
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    # 并行处理所有简历
    log.info(f"全流程筛选: {total} 份简历, 并发数=3")
    candidates = list(await asyncio.gather(*[process_one(r) for r in file_records]))

    # 按得分排序
    candidates.sort(
        key=lambda c: c.get("recommendation", {}).get("scoring", {}).get("overall_score", 0),
        reverse=True,
    )

    # 对比摘要
    comparison = []
    for c in candidates:
        scoring = c.get("recommendation", {}).get("scoring", {})
        summary = c.get("recommendation", {}).get("summary", {})
        comparison.append({
            "name": c.get("name", c.get("file_name", "")),
            "file_name": c.get("file_name", ""),
            "score": scoring.get("overall_score"),
            "tier": scoring.get("tier_label"),
            "satisfied": scoring.get("counts", {}).get("satisfied"),
            "not_satisfied": scoring.get("counts", {}).get("not_satisfied"),
            "cannot_judge": scoring.get("counts", {}).get("cannot_judge"),
            "must_satisfied": scoring.get("counts", {}).get("must_satisfied"),
            "must_total": scoring.get("counts", {}).get("must_total"),
            "recommendation": summary.get("recommendation"),
            "error": c.get("error"),
        })

    log.success(f"全流程筛选完成: {total} 候选人, {sum(1 for c in candidates if 'error' not in c)} 成功")

    return APIResponse(
        success=True,
        message=f"{total} 份简历处理完成",
        data={
            "requirements": requirements,
            "candidates": candidates,
            "comparison": comparison,
        },
    ).model_dump()


# ── SSE 工具 ──────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    # 确保 JSON 不包含换行，防止 SSE 事件被截断
    safe_payload = payload.replace("\n", " ").replace("\r", " ")
    return f"event: {event}\ndata: {safe_payload}\n\n"


# ── SSE 流式筛选（带进度）───────────────────────────────────

@app.post("/api/screen-stream")
async def screen_stream(
    jd_text: str = Form(""),
    requirements: str = Form(""),  # JSON, 可选：如果传入则跳过 JD 解析
    files: list[UploadFile] = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    concurrency: int = Form(3),
    job_id: str = Form(""),
):
    """Skill 2 简历提取 SSE 流式版"""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model
    concurrency = max(1, min(concurrency, 5))

    def _err(msg):
        return Response(content="event: error\ndata: {\"error\":\"" + msg + "\"}\n\n", media_type="text/event-stream")

    if not jd_text.strip() and not requirements.strip():
        return _err("JD 文本或 requirements 不能同时为空")
    if not files:
        return _err("请至少上传一份简历")
    if not api_key:
        return _err("请填写 API Key")

    upload_dir = _space_upload_dir(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 先保存所有文件（UploadFile 在 StreamingResponse 里可能已关闭）
    total = len(files)
    file_records = []
    for idx, file in enumerate(files):
        filename = file.filename or f"resume_{idx}"
        tmp_path = upload_dir / f"screen_sse_{idx}_{filename}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_records.append({"idx": idx, "filename": filename, "path": tmp_path})

    # Track per-job resume upload count
    if job_id:
        _job_resume_counts[job_id] = _job_resume_counts.get(job_id, 0) + len(files)

    async def event_stream():
        event_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(concurrency)

        # ---- Step 0: JD / Requirements ----
        if requirements:
            try:
                parsed_reqs = json.loads(requirements)
            except json.JSONDecodeError:
                parsed_reqs = []
            yield _sse("progress", {"step": "jd_done", "message": f"使用已有 {len(parsed_reqs)} 条要求", "requirements": parsed_reqs})
        else:
            yield _sse("progress", {"step": "jd_parse", "message": "正在解析 JD...", "index": 0, "total": total})
            try:
                jd_text_clean = clean_text(jd_text, "jd")
                jd_result = await _call_llm_async("skills/jd_parser/prompt.txt", {"raw_jd": jd_text_clean}, api_key, base_url, model)
                jd_result = _ensure_jd_parse(jd_text_clean, jd_result)
                parsed_reqs = jd_result.get("requirements", [])
                yield _sse("progress", {"step": "jd_done", "message": f"JD 解析完成：{len(parsed_reqs)} 条要求", "requirements": parsed_reqs})
            except Exception as e:
                yield _sse("error", {"message": f"JD 解析失败: {e}"})
                return

        yield _sse("progress", {"step": "parallel_start", "message": f"开始并行处理 {total} 份简历（并发数={concurrency}）...", "total": total})

        # ---- 并行 worker（仅 Skill 2: 简历提取） ----
        async def process_one(record: dict):
            async with semaphore:
                idx = record["idx"]
                filename = record["filename"]
                tmp_path = record["path"]

                try:
                    # Check cache
                    cached = cache_get(str(tmp_path))
                    if cached and cached.get("raw_text") and cached.get("parsed_resume") and isinstance(cached["parsed_resume"], dict) and cached["parsed_resume"].get("basic_info"):
                        text = cached["raw_text"]
                        resume_result = cached["parsed_resume"]
                        name = resume_result.get("basic_info", {}).get("name", filename)
                        await event_queue.put(_sse("progress", {"step": "resume_step", "index": idx, "total": total,
                            "sub_step": "文档解析", "message": f"[{idx+1}/{total}] {filename} — 文档解析...（缓存命中，跳过）"}))
                    else:
                        await event_queue.put(_sse("progress", {"step": "resume_step", "index": idx, "total": total,
                            "sub_step": "文档解析", "message": f"[{idx+1}/{total}] {filename} — 文档解析..."}))

                        text, _ = document_parser.parse(str(tmp_path))
                        text = clean_text(text, "resume")
                        if not text.strip():
                            await event_queue.put(_sse("candidate_error", {"index": idx, "name": filename, "error": "解析结果为空"}))
                            return

                        await event_queue.put(_sse("progress", {"step": "resume_step", "index": idx, "total": total,
                            "sub_step": "简历提取", "message": f"[{idx+1}/{total}] {filename} — LLM 简历提取..."}))

                        resume_result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": text}, api_key, base_url, model)
                        try:
                            cache_save(str(tmp_path), text, resume_result)
                        except Exception:
                            pass

                    name = resume_result.get("basic_info", {}).get("name", filename)

                    candidate_event = {
                        "index": idx, "name": name, "file_name": filename,
                        "resume": resume_result,
                        "raw_text": text,  # 保留原文，供 Skills 3-5 证据提取使用
                    }
                    await event_queue.put(_sse("candidate_done", candidate_event))
                    log.info(f"[SSE {idx+1}/{total}] {name}: 简历提取完成")

                except Exception as e:
                    log.error(f"[SSE {idx+1}/{total}] {filename} 失败: {e}")
                    await event_queue.put(_sse("candidate_error", {"index": idx, "name": filename, "error": str(e)}))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

        # 启动所有 worker（受 semaphore 控制并发）
        tasks = [asyncio.create_task(process_one(r)) for r in file_records]

        # 从 queue 消费事件，直到所有 task 完成
        done_count = 0
        while done_count < total:
            # 检查是否有事件可消费，或者等 task 完成
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                yield event
                # 统计完成数
                if '"candidate_done"' in event or '"candidate_error"' in event:
                    done_count += 1
            except asyncio.TimeoutError:
                # 检查是否所有 task 都完成了但 queue 空了
                if all(t.done() for t in tasks):
                    # 消费 queue 中剩余事件
                    while not event_queue.empty():
                        event = await event_queue.get()
                        yield event
                        if '"candidate_done"' in event or '"candidate_error"' in event:
                            done_count += 1
                    break

        # 确保所有 task 异常被收集
        await asyncio.gather(*tasks, return_exceptions=True)

        # ---- Complete ----
        yield _sse("complete", {
            "message": f"全部完成：{done_count}/{total} 位候选人",
            "total": total,
            "completed": done_count,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── SSE 分析（Skill 3→4→5，HR 复核时触发）─────────────────

@app.post("/api/analyze-stream")
async def analyze_stream(request: Request):
    """Skill 3→4→5 SSE 流式版：接收已提取的简历数据 + requirements，运行匹配+风险+推荐"""
    body = await request.json()
    requirements = body.get("requirements", [])
    candidates = body.get("candidates", [])
    job_id = body.get("job_id", "")
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model
    concurrency = body.get("concurrency", 3)

    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except json.JSONDecodeError:
            requirements = []
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except json.JSONDecodeError:
            candidates = []

    if not requirements:
        return Response(content="data: {\"error\":\"requirements 不能为空\"}\n\n", media_type="text/event-stream")
    if not candidates:
        return Response(content="data: {\"error\":\"candidates 不能为空\"}\n\n", media_type="text/event-stream")
    if not api_key:
        return Response(content="data: {\"error\":\"请填写 API Key\"}\n\n", media_type="text/event-stream")

    from backend.skills.semantic_matcher.node import match
    from backend.skills.risk_analyzer.node import analyze as analyze_risk
    from backend.skills.recommendation_gen.node import generate as generate_recommendation

    total = len(candidates)
    concurrency = max(1, min(concurrency, 5))

    async def event_stream():
        event_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(concurrency)

        yield _sse("progress", {"step": "analyze_start", "message": f"开始分析 {total} 位候选人 (Skill 3→4→5)...", "total": total})

        async def process_one(c: dict):
            async with semaphore:
                idx = c.get("index", 0)
                name = c.get("name", c.get("file_name", "未知"))
                resume = c.get("resume", {})
                raw_text = c.get("raw_text", "")
                requirements_local = requirements

                if isinstance(resume, str):
                    try:
                        resume = json.loads(resume)
                    except json.JSONDecodeError:
                        resume = {}

                try:
                    await event_queue.put(_sse("progress", {"step": "analyze_step", "index": idx, "total": total,
                        "sub_step": "Skill 3: 语义匹配+证据", "message": f"[{idx+1}/{total}] {name} — 语义匹配+证据校验..."}))

                    match_result = await match(requirements_local, resume, raw_text, api_key, base_url, model)
                    matches = match_result.get("matches", [])

                    await event_queue.put(_sse("progress", {"step": "analyze_step", "index": idx, "total": total,
                        "sub_step": "Skill 4: 风险分析", "message": f"[{idx+1}/{total}] {name} — 风险分析..."}))

                    risk_result = await analyze_risk(requirements_local, matches, api_key, base_url, model)
                    analysis = risk_result.get("analysis", {})

                    await event_queue.put(_sse("progress", {"step": "analyze_step", "index": idx, "total": total,
                        "sub_step": "Skill 5: 生成推荐", "message": f"[{idx+1}/{total}] {name} — 生成推荐..."}))

                    rec_result = await generate_recommendation(requirements_local, matches, matches, analysis, api_key, base_url, model)
                    scoring = rec_result.get("scoring", {})
                    summary = rec_result.get("summary", {})
                    recommendation_reason = rec_result.get("recommendation_reason", "")

                    candidate_event = {
                        "index": idx, "name": name,
                        "score": scoring.get("overall_score"),
                        "tier": scoring.get("tier_label"),
                        "satisfied": scoring.get("counts", {}).get("satisfied"),
                        "not_satisfied": scoring.get("counts", {}).get("not_satisfied"),
                        "cannot_judge": scoring.get("counts", {}).get("cannot_judge"),
                        "must_satisfied": scoring.get("counts", {}).get("must_satisfied"),
                        "must_total": scoring.get("counts", {}).get("must_total"),
                        "matches": matches,
                        "analysis": analysis,
                        "scoring": scoring,
                        "recommendation_reason": recommendation_reason,
                        "core_advantages": summary.get("core_advantages", []),
                        "key_risks": summary.get("key_risks", []),
                        "human_review_questions": summary.get("human_review_questions", []),
                        "interview_suggestions": analysis.get("interview_suggestions", []),
                    }
                    await event_queue.put(_sse("candidate_done", candidate_event))
                    if job_id:
                        _job_screened_counts[job_id] = _job_screened_counts.get(job_id, 0) + 1
                    log.info(f"[Analyze {idx+1}/{total}] {name}: {scoring.get('tier_label')} (得分: {scoring.get('overall_score')})")

                except Exception as e:
                    log.error(f"[Analyze {idx+1}/{total}] {name} 失败: {e}")
                    await event_queue.put(_sse("candidate_error", {"index": idx, "name": name, "error": str(e)}))

        tasks = [asyncio.create_task(process_one(c)) for c in candidates]

        done_count = 0
        while done_count < total:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                yield event
                if '"candidate_done"' in event or '"candidate_error"' in event:
                    done_count += 1
            except asyncio.TimeoutError:
                if all(t.done() for t in tasks):
                    while not event_queue.empty():
                        event = await event_queue.get()
                        yield event
                        if '"candidate_done"' in event or '"candidate_error"' in event:
                            done_count += 1
                    break

        await asyncio.gather(*tasks, return_exceptions=True)

        yield _sse("complete", {
            "message": f"全部分析完成：{done_count}/{total} 位候选人",
            "total": total,
            "completed": done_count,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Agent 驱动筛选（含停止条件 + 局部重算）─────────────────

@app.post("/api/screen-agent")
async def screen_agent(
    jd_text: str = Form(""),
    files: list[UploadFile] = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    """全流程筛选 Agent 版：controller 驱动的 SSE 流式 + 停止条件判断"""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model

    if not jd_text.strip():
        return Response(content="data: {\"error\":\"JD 文本不能为空\"}\n\n", media_type="text/event-stream")
    if not files:
        return Response(content="data: {\"error\":\"请至少上传一份简历\"}\n\n", media_type="text/event-stream")
    if not api_key:
        return Response(content="data: {\"error\":\"请填写 API Key\"}\n\n", media_type="text/event-stream")

    from backend.skills.semantic_matcher.node import match
    from backend.skills.risk_analyzer.node import analyze as analyze_risk
    from backend.skills.recommendation_gen.node import generate as generate_recommendation
    from backend.workflows.controller import check_stop_condition, find_unresolved, ControllerState

    upload_dir = _space_upload_dir(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    async def event_stream():
        total = len(files)

        # ---- Step 0: JD 解析 ----
        yield _sse("progress", {"step": "jd_parse", "message": "正在解析 JD...", "index": 0, "total": total})
        try:
            jd_text_clean = clean_text(jd_text, "jd")
            jd_result = _call_llm("skills/jd_parser/prompt.txt", {"raw_jd": jd_text_clean}, api_key, base_url, model)
            requirements = jd_result.get("requirements", [])
            yield _sse("progress", {"step": "jd_done", "message": f"JD 解析完成：{len(requirements)} 条要求", "requirements": requirements})
        except Exception as e:
            yield _sse("error", {"message": f"JD 解析失败: {e}"})
            return

        # ---- Step 1-N: 逐份 Agent 驱动处理 ----
        processed = []

        for idx, file in enumerate(files):
            filename = file.filename or f"resume_{idx}"
            tmp_path = upload_dir / f"screen_agent_{idx}_{filename}"
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 每份简历独立的 controller 状态
            ctrl_state = ControllerState(total_requirements=len(requirements))

            try:
                # Check cache
                cached = cache_get(str(tmp_path))
                if cached and cached.get("raw_text") and cached.get("parsed_resume") and isinstance(cached["parsed_resume"], dict) and cached["parsed_resume"].get("basic_info"):
                    text = cached["raw_text"]
                    resume_result = cached["parsed_resume"]
                    name = resume_result.get("basic_info", {}).get("name", filename)
                    yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                        "sub_step": "文档解析", "message": f"正在处理简历 {idx+1}/{total}: {filename} — 文档解析...（缓存命中，跳过）"})
                else:
                    yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                        "sub_step": "文档解析", "message": f"正在处理简历 {idx+1}/{total}: {filename} — 文档解析..."})

                    text, _ = document_parser.parse(str(tmp_path))
                    text = clean_text(text, "resume")
                    if not text.strip():
                        yield _sse("candidate_error", {"index": idx, "name": filename, "error": "解析结果为空"})
                        continue

                    yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                        "sub_step": "简历提取", "message": f"正在处理简历 {idx+1}/{total}: {filename} — 简历提取..."})

                    resume_result = await _call_llm_async("skills/resume_extractor/prompt.txt", {"raw_resume": text}, api_key, base_url, model)
                    try:
                        cache_save(str(tmp_path), text, resume_result)
                    except Exception as e:
                        log.warning(f"[Cache] 保存失败 ({filename}): {e}")

                name = resume_result.get("basic_info", {}).get("name", filename)

                # ── Agent 循环：匹配 → 检查 → 重算/暂停 ──
                yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "语义匹配+证据", "message": f"正在处理简历 {idx+1}/{total}: {name} — 语义匹配+证据..."})

                match_result = await match(requirements, resume_result, text, api_key, base_url, model)
                items = match_result.get("matches", [])

                # Agent: 检查停止条件
                should_stop, stop_reason, pending_hr = check_stop_condition(items, ctrl_state.retry_count)
                ctrl_state.stop_reason = stop_reason
                ctrl_state.pending_hr_items = pending_hr

                yield _sse("agent_check", {
                    "index": idx, "name": name,
                    "should_stop": should_stop,
                    "stop_reason": stop_reason,
                    "pending_hr": pending_hr,
                    "retry_count": ctrl_state.retry_count,
                })

                # Agent: 找出未解决的 must 要求，尝试重试
                unresolved = find_unresolved(requirements, items)
                if unresolved and ctrl_state.retry_count < 2:
                    yield _sse("agent_retry", {
                        "index": idx, "name": name,
                        "message": f"Agent: {len(unresolved)} 项要求未解决，重试第 {ctrl_state.retry_count + 1} 次...",
                        "unresolved_ids": [r["id"] for r in unresolved],
                    })
                    # 仅对未解决项重跑匹配
                    retry_result = await match(unresolved, resume_result, text, api_key, base_url, model)
                    retry_matches = retry_result.get("matches", [])
                    # 合并：用重跑结果覆盖对应项
                    retry_map = {m.get("requirement_id", ""): m for m in retry_matches}
                    for item in items:
                        if item.get("requirement_id", "") in retry_map:
                            item.update(retry_map[item["requirement_id"]])
                    ctrl_state.retry_count += 1

                    # 重新检查
                    should_stop, stop_reason, pending_hr = check_stop_condition(items, ctrl_state.retry_count)
                    ctrl_state.stop_reason = stop_reason
                    ctrl_state.pending_hr_items = pending_hr

                # 风险分析
                yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "风险分析", "message": f"正在处理简历 {idx+1}/{total}: {name} — 风险分析..."})
                risk_result = await analyze_risk(requirements, items, api_key, base_url, model)
                analysis = risk_result.get("analysis", {})

                # 推荐生成
                yield _sse("progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "生成推荐", "message": f"正在处理简历 {idx+1}/{total}: {name} — 生成推荐..."})
                rec_result = await generate_recommendation(requirements, items, items, analysis, api_key, base_url, model)
                scoring = rec_result.get("scoring", {})
                summary = rec_result.get("summary", {})
                recommendation_reason = rec_result.get("recommendation_reason", "")

                yield _sse("candidate_done", {
                    "index": idx, "name": name, "file_name": filename,
                    "score": scoring.get("overall_score"),
                    "tier": scoring.get("tier_label"),
                    "tier_reason": scoring.get("tier_reason", ""),
                    "satisfied": scoring.get("counts", {}).get("satisfied"),
                    "not_satisfied": scoring.get("counts", {}).get("not_satisfied"),
                    "cannot_judge": scoring.get("counts", {}).get("cannot_judge"),
                    "must_satisfied": scoring.get("counts", {}).get("must_satisfied"),
                    "must_total": scoring.get("counts", {}).get("must_total"),
                    "recommendation": summary.get("recommendation"),
                    "recommendation_reason": recommendation_reason,
                    "core_advantages": summary.get("core_advantages", []),
                    "key_risks": summary.get("key_risks", []),
                    "human_review_questions": summary.get("human_review_questions", []),
                    "interview_suggestions": analysis.get("interview_suggestions", []),
                    "stop_reason": ctrl_state.stop_reason,
                    "pending_hr": ctrl_state.pending_hr_items,
                    "retry_count": ctrl_state.retry_count,
                    "matches": items,
                    "analysis": analysis,
                    "scoring": scoring,
                    "resume": resume_result,
                    "raw_text": text,
                })

                processed.append(name)
                log.info(f"[Agent {idx+1}/{total}] {name}: {scoring.get('tier_label')} (得分: {scoring.get('overall_score')}, 重试: {ctrl_state.retry_count}, 停因: {ctrl_state.stop_reason})")

            except Exception as e:
                log.error(f"[Agent {idx+1}/{total}] {filename} 失败: {e}")
                yield _sse("candidate_error", {"index": idx, "name": filename, "error": str(e)})
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        # ---- Complete ----
        yield _sse("complete", {
            "message": f"Agent 筛选完成：{len(processed)} 位候选人",
            "requirements": requirements,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Session-aware Screening (background + reconnection) ─────

@app.post("/api/screen-session")
async def screen_session(
    jd_text: str = Form(""),
    requirements: str = Form(""),
    files: list[UploadFile] | None = File(default=None),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    concurrency: int = Form(3),
    job_id: str = Form(""),
    candidates_json: str = Form(""),  # JSON: pre-parsed candidates [{index, name, file_name, resume, raw_text}, ...]
):
    """Full pipeline with session. Accepts files OR pre-parsed candidates_json.
    Returns session_id immediately; connect to GET /api/session/{sid}/stream for events."""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model
    concurrency = max(1, min(concurrency, 5))

    # Parse pre-parsed candidates if provided
    pre_parsed = []
    if candidates_json:
        try:
            pre_parsed = json.loads(candidates_json)
        except json.JSONDecodeError:
            pre_parsed = []

    has_files = bool(files and any(f.filename for f in files))
    if not pre_parsed and not has_files:
        return JSONResponse({"success": False, "message": "请至少上传一份简历或提供预解析数据"}, status_code=400)

    if not jd_text.strip() and not requirements.strip():
        return JSONResponse({"success": False, "message": "JD 文本或 requirements 不能同时为空"}, status_code=400)
    if not api_key:
        return JSONResponse({"success": False, "message": "请填写 API Key"}, status_code=400)

    # Build candidate records from files OR pre-parsed data
    file_records = []
    if pre_parsed:
        for pp in pre_parsed:
            file_records.append({
                "idx": pp.get("index", len(file_records)),
                "filename": pp.get("file_name", pp.get("name", "")),
                "path": None,
                "resume": pp.get("resume"),
                "raw_text": pp.get("raw_text", ""),
            })
    else:
        upload_dir = _space_upload_dir(settings.app.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        for idx, file in enumerate(files):
            filename = file.filename or f"resume_{idx}"
            tmp_path = upload_dir / f"session_{idx}_{filename}"
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_records.append({"idx": idx, "filename": filename, "path": tmp_path})

    # Track job resume counts
    if job_id:
        _job_resume_counts[job_id] = _job_resume_counts.get(job_id, 0) + len(file_records)

    # Create session manually (requirements not yet known)
    import uuid
    from backend.agent.session import _sessions, _event_queues
    sid = uuid.uuid4().hex[:12]
    session = AgentSession(id=sid)
    session.job_id = job_id
    _sessions[sid] = session
    _event_queues[sid] = asyncio.Queue()
    emit(sid, "session_created", {"session_id": sid, "total": len(file_records)})

    async def background_pipeline():
        from backend.skills.semantic_matcher.node import match
        from backend.skills.risk_analyzer.node import analyze as analyze_risk
        from backend.skills.recommendation_gen.node import generate as generate_recommendation
        from backend.workflows.controller import ScreeningAgent

        session = get_session(sid)
        total = len(file_records)

        # Step 0: JD / Requirements
        requirements_local = []
        if requirements:
            try:
                requirements_local = json.loads(requirements)
            except json.JSONDecodeError:
                requirements_local = []
        if not requirements_local and jd_text.strip():
            try:
                from backend.skills.jd_parser.node import parse
                jd_result = await parse(jd_text.strip(), api_key, base_url, model)
                requirements_local = jd_result.get("requirements", [])
            except Exception as e:
                log.error(f"[Session {sid}] JD 解析失败: {e}")
                emit(sid, "error", {"message": f"JD 解析失败: {e}"})
                return

        emit(sid, "jd_done", {"requirements": requirements_local})
        emit(sid, "session_created", {"session_id": sid, "total": total})
        session.requirements = requirements_local

        # Initialize candidate slots
        for rec in file_records:
            slot = CandidateSlot(
                index=rec["idx"],
                name=rec["filename"],
                file_name=rec["filename"],
                status="waiting",
            )
            slot.raw_text = ""
            session.candidates.append(slot)

        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(rec):
            idx = rec["idx"]
            filename = rec["filename"]
            tmp_path = rec["path"]
            slot = session.candidates[idx]
            try:
                pre_resume = rec.get("resume")
                pre_raw_text = rec.get("raw_text", "")

                if pre_resume and pre_raw_text:
                    # Use pre-parsed data — skip document parsing and resume extraction
                    text = pre_raw_text
                    resume_result = pre_resume
                    slot.raw_text = text
                    slot.resume = resume_result
                    emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                        "sub_step": "数据就绪", "message": f"[{idx+1}/{total}] {filename} — 使用预解析数据"})
                else:
                    # Normal flow: parse document then extract resume
                    emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                        "sub_step": "文档解析", "message": f"[{idx+1}/{total}] {filename} — 文档解析..."})
                    slot.status = "document_parsing"

                    text, _ = await asyncio.to_thread(document_parser.parse, str(tmp_path))
                    text = clean_text(text)
                    slot.raw_text = text

                    # Check cache
                    cached = cache_get(str(tmp_path))
                    if cached:
                        emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                            "sub_step": "简历提取", "message": f"[{idx+1}/{total}] {filename} — 缓存命中，跳过提取"})
                        resume_result = cached.get("parsed_resume", {})
                    else:
                        from backend.skills.resume_extractor.node import extract as extract_resume
                        emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                            "sub_step": "简历提取", "message": f"[{idx+1}/{total}] {filename} — LLM 简历提取..."})
                        slot.status = "resume_extraction"
                        resume_result = await extract_resume(text, api_key, base_url, model)
                        try:
                            cache_save(str(tmp_path), text, resume_result, job_id)
                        except Exception as e:
                            log.warning(f"[Session {sid}] 缓存保存失败 ({filename}): {e}")

                    slot.resume = resume_result

                name = (resume_result.get("basic_info") or {}).get("name") or filename
                slot.name = name

                # 解析（文档解析 + Skill 2 简历提取）到此真正完成——把解析好的简历数据发给前端，
                # 让"简历已经解析"通知与"查看解析结果"落到实处（此前解析数据要等 candidate_done 才回传）。
                emit(sid, "resume_parsed", {
                    "index": idx, "name": name, "file_name": filename,
                    "resume": resume_result, "raw_text": slot.raw_text,
                })

                # Matching
                emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "语义匹配", "message": f"[{idx+1}/{total}] {slot.name} — 语义匹配..."})
                slot.status = "matching"
                match_result = await match(requirements_local, resume_result, slot.raw_text, api_key, base_url, model)
                slot.matches = match_result.get("matches", [])
                agent = ScreeningAgent(requirements_local)
                decision = agent.decide_after_match(slot.matches)
                emit(sid, "agent_check", {
                    "index": idx,
                    "name": slot.name,
                    "action": decision["action"],
                    "should_stop": decision["should_stop"],
                    "stop_reason": decision["reason"],
                    "pending_hr": decision["pending_hr"],
                    "retry_count": decision["retry_count"],
                })
                if decision["action"] == "retry_unresolved":
                    unresolved = decision["unresolved"]
                    retry_count = agent.mark_retry()
                    emit(sid, "agent_retry", {
                        "index": idx,
                        "name": slot.name,
                        "message": f"Agent: {len(unresolved)} 项要求未解决，重试第 {retry_count} 次...",
                        "unresolved_ids": [r.get("id", "") for r in unresolved],
                    })
                    retry_result = await match(unresolved, resume_result, slot.raw_text, api_key, base_url, model)
                    retry_map = {m.get("requirement_id", ""): m for m in retry_result.get("matches", [])}
                    for item in slot.matches:
                        rid = item.get("requirement_id", "")
                        if rid in retry_map:
                            item.update(retry_map[rid])
                    decision = agent.decide_after_retry(slot.matches)
                    emit(sid, "agent_check", {
                        "index": idx,
                        "name": slot.name,
                        "action": decision["action"],
                        "should_stop": decision["should_stop"],
                        "stop_reason": decision["reason"],
                        "pending_hr": decision["pending_hr"],
                        "retry_count": decision["retry_count"],
                    })

                # Risk analysis
                emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "风险分析", "message": f"[{idx+1}/{total}] {slot.name} — 风险分析..."})
                slot.status = "risk_analysis"
                risk_result = await analyze_risk(requirements_local, slot.matches, api_key, base_url, model)
                slot.analysis = risk_result.get("analysis", {})

                # Recommendation
                emit(sid, "progress", {"step": "resume_step", "index": idx, "total": total,
                    "sub_step": "生成推荐", "message": f"[{idx+1}/{total}] {slot.name} — 生成推荐..."})
                rec_result = await generate_recommendation(requirements_local, slot.matches, slot.matches, slot.analysis, api_key, base_url, model)
                slot.scoring = rec_result.get("scoring", {})
                summary = rec_result.get("summary", {})
                slot.status = "done"

                emit(sid, "candidate_done", {
                    "index": idx, "name": slot.name, "file_name": filename,
                    "score": slot.scoring.get("overall_score"),
                    "tier": slot.scoring.get("tier_label"),
                    "satisfied": slot.scoring.get("counts", {}).get("satisfied"),
                    "not_satisfied": slot.scoring.get("counts", {}).get("not_satisfied"),
                    "cannot_judge": slot.scoring.get("counts", {}).get("cannot_judge"),
                    "must_satisfied": slot.scoring.get("counts", {}).get("must_satisfied"),
                    "must_total": slot.scoring.get("counts", {}).get("must_total"),
                    "recommendation": summary.get("recommendation"),
                    "recommendation_reason": rec_result.get("recommendation_reason", ""),
                    "core_advantages": summary.get("core_advantages", []),
                    "key_risks": summary.get("key_risks", []),
                    "human_review_questions": summary.get("human_review_questions", []),
                    "interview_suggestions": slot.analysis.get("interview_suggestions", []),
                    "matches": slot.matches,
                    "analysis": slot.analysis,
                    "scoring": slot.scoring,
                    "resume": slot.resume,
                    "raw_text": slot.raw_text,
                })
                if job_id:
                    _job_screened_counts[job_id] = _job_screened_counts.get(job_id, 0) + 1
                log.info(f"[Session {sid}] {slot.name}: {slot.scoring.get('tier_label')} (得分: {slot.scoring.get('overall_score')})")

            except Exception as e:
                log.error(f"[Session {sid}] {filename} 失败: {e}")
                slot.status = "error"
                emit(sid, "candidate_error", {"index": idx, "name": filename, "error": str(e)})
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()

        tasks = [asyncio.create_task(process_one(rec)) for rec in file_records]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            session.phase = "cancelled"
            emit(sid, "cancelled", {"message": "任务已取消"})
            log.info(f"[Session {sid}] 已取消")
            return

        if session.cancelled:
            return

        emit(sid, "complete", {
            "message": f"全部分析完成：{total} 位候选人",
            "total": total,
            "completed": total,
        })
        log.info(f"[Session {sid}] 全部完成")

    session.task = asyncio.create_task(background_pipeline())
    return {"success": True, "session_id": sid}


@app.get("/api/session/{sid}/stream")
async def session_stream(sid: str):
    """SSE stream with event replay for reconnection."""
    session = get_session(sid)
    if not session:
        return JSONResponse({"success": False, "message": "会话不存在或已过期"}, status_code=404)

    q = get_queue(sid)
    if not q:
        return JSONResponse({"success": False, "message": "会话队列不存在"}, status_code=404)

    async def event_stream():
        sent_count = 0
        # Replay stored events
        for evt_type, payload in session.events:
            yield _sse(evt_type, payload)
            sent_count += 1

        # Stream new events
        while True:
            try:
                evt_type, payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield _sse(evt_type, payload)
                sent_count += 1
                if evt_type in ("complete", "cancelled", "error"):
                    break
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {"message": "等待中..."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/session/{sid}/status")
async def session_status(sid: str):
    """Current session status snapshot (polling fallback)."""
    session = get_session(sid)
    if not session:
        return JSONResponse({"success": False, "message": "会话不存在或已过期"}, status_code=404)

    candidates_status = []
    for c in session.candidates:
        sc = getattr(c, 'scoring', {}) or {}
        candidates_status.append({
            "index": c.index,
            "name": c.name,
            "file_name": c.file_name,
            "status": c.status,
            "resume": c.resume,
            "raw_text": c.raw_text,
            "matches": c.matches,
            "analysis": c.analysis,
            "scoring": c.scoring,
            "stop_reason": getattr(c, "stop_reason", ""),
            "pending_hr": getattr(c, "pending_hr", []),
            "score": sc.get("overall_score") if isinstance(sc, dict) else None,
            "tier": sc.get("tier_label", "") if isinstance(sc, dict) else "",
        })

    return {
        "success": True,
        "session_id": sid,
        "job_id": getattr(session, "job_id", ""),
        "phase": getattr(session, "phase", ""),
        "cancelled": getattr(session, "cancelled", False),
        "requirements": session.requirements,
        "total": len(session.candidates),
        "completed": sum(1 for c in session.candidates if c.status in ("done", "error")),
        "candidates": candidates_status,
    }


@app.post("/api/session/{sid}/cancel")
async def session_cancel(sid: str):
    """Cancel a running screening session's background task."""
    session = get_session(sid)
    if not session:
        return JSONResponse({"success": False, "message": "会话不存在或已过期"}, status_code=404)

    task = getattr(session, "task", None)
    if task and not task.done():
        task.cancel()
    session.cancelled = True
    return {"success": True, "session_id": sid, "cancelled": True}


# ── 静态文件 ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    # 前端是单文件，改动频繁；禁用缓存，避免浏览器一直显示旧版本。
    return FileResponse(
        Path(__file__).parent.parent / "skill-tester" / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    import uvicorn
    # 云平台（Railway 等）会注入 PORT；本地开发回落到配置里的 dev_port。
    # HOST 默认沿用配置（本地 127.0.0.1）；容器里由 Dockerfile 设 HOST=0.0.0.0 对外可访问。
    port = int(_os.environ.get("PORT", settings.app.dev_port))
    host = _os.environ.get("HOST", settings.app.host)
    print(f"\n  F1 Skill 服务器启动: http://{host}:{port}")
    print(f"  API Key: {'已配置' if settings.deepseek.api_key else '未配置（需在前端填写）'}")
    uvicorn.run(app, host=host, port=port, workers=1, timeout_keep_alive=600)
