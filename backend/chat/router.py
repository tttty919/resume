"""Chat API router — natural-language dashboard assistant.

Endpoint:
  POST /api/chat — HR sends free-text + optional file attachments; the LLM
  decides (via tool-calling) whether to enter a JD, start a screening run,
  check task status, or cancel a task. Tool-calling IS the intent
  recognition step — there is no separate NL classifier.

The tool set intentionally excludes anything that edits/deletes existing
structured data (JD fields, resumes, screening results) — HR is redirected
to the relevant management page for those, consistent with /api/hr-override
requiring a structured form.
"""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.utils.llm_utils import create_llm

log = get_logger()
router = APIRouter(prefix="/api")


def _sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    safe_payload = payload.replace("\n", " ").replace("\r", " ")
    return f"event: {event_type}\ndata: {safe_payload}\n\n"


CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enter_jd",
            "description": "录入一条新的JD：创建岗位并解析出结构化岗位要求。当HR明确要求录入/新建一个岗位，并且提供了JD内容（粘贴的文本，或本轮拖入的文件）时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "岗位名称，从JD内容或HR的话里提炼一个简短名称"},
                    "jd_text": {"type": "string", "description": "JD原文文本。如果HR是拖入文件而不是粘贴文本，这里留空字符串，系统会自动读取附件内容"},
                },
                "required": ["job_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_screening",
            "description": "对本轮HR上传的简历文件，与指定岗位的要求做批量匹配，生成排名报告。仅当本轮确实有简历文件附件、且目标岗位已经明确（不是靠猜的）时才调用；如果岗位不明确，不要调用，先反问HR。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "目标岗位的 id，必须是现有岗位列表里给出的 id，不能编造"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_status",
            "description": "查询某个正在进行的筛选任务目前的进度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "任务的 session_id，必须是当前进行中任务列表里给出的 id"},
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "取消一个正在进行的筛选任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "任务的 session_id，必须是当前进行中任务列表里给出的 id"},
                },
                "required": ["session_id"],
            },
        },
    },
]

NAV_MAP = {
    "enter_jd": {"page": "jobs", "label": "去JD管理页面看看"},
}


def build_chat_system_prompt(jobs: list, attached_filenames: list, active_sessions: list) -> str:
    jobs_lines = "\n".join(
        f"- {j.get('id')}: {j.get('name')}（简历{j.get('resumeCount', 0)}份，已筛{j.get('screenedCount', 0)}份）"
        for j in jobs
    ) or "（暂无岗位）"
    files_lines = "、".join(attached_filenames) if attached_filenames else "（本轮无附件）"
    sessions_lines = "\n".join(
        f"- {s.get('session_id')}: {s.get('job_name', '')}"
        for s in active_sessions
    ) or "（当前没有进行中的任务）"

    return f"""你是HR简历筛选平台首页的助理，只负责三件事：
① 录入JD（新建岗位并解析岗位要求）
② 对已上传的简历发起批量筛选，生成排名报告
③ 查询或取消正在进行的筛选任务

现有岗位列表（判断HR说的是哪个岗位时参考，调用工具必须使用其中的 id，不能编造）：
{jobs_lines}

当前进行中的筛选任务（判断HR问的"进度"或要取消的是哪个任务时参考，调用工具必须使用其中的 session_id）：
{sessions_lines}

本轮HR随消息附带的文件：{files_lines}

规则：
- 如果HR要求修改已有JD/岗位要求的字段、删除岗位或简历、修改筛选结果里的判断，都不要尝试处理——你没有对应的工具，也不应该有。直接用自然语言回复，引导HR去对应的管理页面（职位管理 / 简历管理 / 筛选结果）操作，说明具体去哪个页面。
- 如果HR要求批量筛选，但没说清楚是哪个岗位，或者本轮没有简历文件附件，不要调用 start_screening，先向HR确认清楚。
- 除了上述三件事，其余情况直接用自然语言回复，不要调用工具。
- 回复措辞简洁，像同事之间对话，不要写成正式公文。
- 重要：当你调用 start_screening 成功后，只需简短确认任务已启动（一句话），不要写"完成后告诉你结果"或"可以问我进度"之类的话——筛选进度和结果会自动显示在下方对话中，无需你后续跟进。不要提"查看筛选结果"链接，结果会自动推送。
"""


async def _tool_enter_jd(args: dict, files: list[UploadFile], llm_cfg: dict) -> dict:
    from backend import dev_server
    from backend.tools.document_parser import document_parser
    from backend.tools.text_cleaner import clean_text

    settings = get_settings()
    jd_text = (args.get("jd_text") or "").strip()
    job_name = (args.get("job_name") or "").strip() or "未命名岗位"

    if not jd_text and files:
        from backend.utils.space import upload_dir as _space_upload
        upload_dir = _space_upload()
        f = files[0]
        tmp_path = upload_dir / f"chat_jd_{uuid.uuid4().hex[:8]}_{f.filename}"
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            jd_text, _ = document_parser.parse(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    if not jd_text:
        return {"success": False, "message": "没有可解析的JD内容，请粘贴JD文本或拖入JD文件"}

    jd_clean = clean_text(jd_text, "jd")
    try:
        parsed = await dev_server._call_llm_async(
            "skills/jd_parser/prompt.txt", {"raw_jd": jd_clean},
            llm_cfg["api_key"], llm_cfg["base_url"], llm_cfg["model"],
        )
        parsed = dev_server._ensure_jd_parse(jd_clean, parsed)
    except Exception as e:
        return {"success": False, "message": f"JD 解析失败: {e}"}

    requirements = parsed.get("requirements", [])
    if not requirements:
        return {"success": False, "message": "JD 解析未提取到岗位要求：" + parsed.get("parsing_notes", "请确认这是一份完整的JD")}

    job = {
        "id": "job-" + uuid.uuid4().hex[:10],
        "name": job_name,
        "jd": jd_text,
        "status": "招聘中",
        "department": "",
        "roleCategory": "",
        "jobType": "全职",
        "location": "",
        "salary": "",
        "experience": "",
        "education": "",
        "context": "",
        "mustSkills": [r.get("name") for r in requirements if r.get("type") == "must"],
        "bonusSkills": [r.get("name") for r in requirements if r.get("type") == "bonus"],
        "requirements": requirements,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    jobs = dev_server._load_jobs()
    jobs.insert(0, job)
    dev_server._save_jobs(jobs)

    return {
        "success": True, "job_id": job["id"], "job_name": job_name,
        "requirement_count": len(requirements),
    }


async def _tool_start_screening(args: dict, files: list[UploadFile], llm_cfg: dict) -> dict:
    from backend import dev_server

    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return {"success": False, "message": "缺少 job_id"}
    if not files:
        return {"success": False, "message": "没有检测到简历附件，请先把简历文件拖进来"}

    jobs = dev_server._load_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if not job:
        return {"success": False, "message": "未找到该岗位"}
    requirements = job.get("requirements") or []
    if not requirements:
        return {"success": False, "message": "该岗位尚未解析JD要求，无法开始筛选"}

    result = await dev_server.screen_session(
        jd_text="",
        requirements=json.dumps(requirements, ensure_ascii=False),
        files=files,
        api_key=llm_cfg["api_key"], base_url=llm_cfg["base_url"], model=llm_cfg["model"],
        concurrency=3, job_id=job_id, candidates_json="",
    )
    if isinstance(result, JSONResponse):
        result = json.loads(bytes(result.body))
    if not result.get("success"):
        return result
    return {
        "success": True, "session_id": result["session_id"],
        "job_id": job_id, "job_name": job.get("name", ""),
        "candidate_count": len(files),
    }


async def _tool_get_task_status(args: dict, files: list[UploadFile], llm_cfg: dict) -> dict:
    from backend import dev_server

    sid = (args.get("session_id") or "").strip()
    if not sid:
        return {"success": False, "message": "缺少 session_id"}
    result = await dev_server.session_status(sid)
    if isinstance(result, JSONResponse):
        result = json.loads(bytes(result.body))
    if not result.get("success"):
        return result
    return {
        "success": True, "session_id": sid, "phase": result.get("phase"),
        "total": result.get("total"), "completed": result.get("completed"),
        "cancelled": result.get("cancelled", False),
    }


async def _tool_cancel_task(args: dict, files: list[UploadFile], llm_cfg: dict) -> dict:
    from backend import dev_server

    sid = (args.get("session_id") or "").strip()
    if not sid:
        return {"success": False, "message": "缺少 session_id"}
    result = await dev_server.session_cancel(sid)
    if isinstance(result, JSONResponse):
        result = json.loads(bytes(result.body))
    return result


TOOL_DISPATCH = {
    "enter_jd": _tool_enter_jd,
    "start_screening": _tool_start_screening,
    "get_task_status": _tool_get_task_status,
    "cancel_task": _tool_cancel_task,
}


async def chat_turn(history: list, active_sessions: list, files: list[UploadFile], api_key: str, base_url: str, model: str):
    from backend import dev_server

    jobs = dev_server._load_jobs()
    for job in jobs:
        jid = job.get("id", "")
        job["resumeCount"] = len(dev_server.cache_list_by_job(jid))
        job["screenedCount"] = dev_server._job_screened_counts.get(jid, 0)
    system_prompt = build_chat_system_prompt(jobs, [f.filename for f in files], active_sessions)

    lc_messages = [SystemMessage(content=system_prompt)]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    llm = create_llm(api_key=api_key, base_url=base_url, model=model).bind_tools(CHAT_TOOLS)

    try:
        response = await llm.ainvoke(lc_messages)
    except Exception as e:
        log.error(f"[Chat] 第一次 LLM 调用失败: {e}")
        yield _sse("assistant_text", {"text": f"抱歉，调用大模型失败：{e}"})
        yield _sse("done", {})
        return

    tool_calls = response.tool_calls or []
    if not tool_calls:
        yield _sse("assistant_text", {"text": response.content or ""})
        yield _sse("done", {})
        return

    llm_cfg = {"api_key": api_key, "base_url": base_url, "model": model}
    lc_messages.append(response)

    for call in tool_calls:
        name = call.get("name")
        call_args = call.get("args") or {}
        yield _sse("tool_call", {"name": name, "args": call_args})

        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            result = {"success": False, "message": f"未知工具: {name}"}
        else:
            try:
                result = await handler(call_args, files, llm_cfg)
            except Exception as e:
                log.error(f"[Chat] 工具 {name} 执行失败: {e}")
                result = {"success": False, "message": str(e)}

        yield _sse("tool_result", {"name": name, "result": result})
        if result.get("success") and name in NAV_MAP:
            yield _sse("navigation", NAV_MAP[name])

        lc_messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call.get("id", "")))

    try:
        final = await llm.ainvoke(lc_messages)
        final_text = final.content or ""
    except Exception as e:
        log.error(f"[Chat] 第二次 LLM 调用失败: {e}")
        final_text = f"任务已执行，但生成回复时出错：{e}"

    yield _sse("assistant_text", {"text": final_text})
    yield _sse("done", {})


@router.post("/chat")
async def chat(
    history: str = Form("[]"),
    active_sessions: str = Form("[]"),
    files: list[UploadFile] | None = File(default=None),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    settings = get_settings()
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model
    files = files or []

    try:
        history_list = json.loads(history) if history else []
    except json.JSONDecodeError:
        history_list = []
    try:
        sessions_list = json.loads(active_sessions) if active_sessions else []
    except json.JSONDecodeError:
        sessions_list = []

    async def event_stream():
        async for chunk in chat_turn(history_list, sessions_list, files, api_key, base_url, model):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
