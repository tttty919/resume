"""Chat API router — natural-language dashboard assistant.

Endpoint:
  POST /api/chat — HR sends free-text + optional file attachments; the LLM
  decides (via tool-calling) whether to enter a JD, start a screening run,
  check task status, cancel a task, or answer a read-only query about
  existing jobs/candidates. Tool-calling IS the intent recognition step —
  there is no separate NL classifier.

The tool set intentionally excludes anything that edits/deletes existing
structured data (JD fields, resumes, screening results) — HR is redirected
to the relevant management page for those, consistent with /api/hr-override
requiring a structured form. Read-only lookups (get_job_detail,
get_job_candidates, find_candidate) are answered directly instead.
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
from backend.utils.space import upload_dir as _space_upload_dir
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
            "name": "enter_resume",
            "description": "录入（解析并保存）本轮HR上传的简历文件，归档到指定岗位下，但不会立刻发起匹配/生成报告。当HR只是想先把简历存进系统、还没决定要不要马上匹配，或者明确说“先录入/先存着”时调用；如果HR是想要匹配出报告，应该调用 start_screening 而不是这个。仅当本轮确实有简历文件附件、且目标岗位已经明确时才调用；如果岗位不明确，不要调用，先反问HR。",
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
    {
        "type": "function",
        "function": {
            "name": "get_job_detail",
            "description": "查询某个已有岗位的JD原文和结构化岗位要求。当HR问关于某个岗位JD内容/要求/技能项本身的问题时调用（例如“这个JD大概讲的是什么”“这个岗位要求哪些技能”），返回原始数据后你自己组织语言回答，不要因为问法生僻就拒答。",
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
            "name": "get_job_candidates",
            "description": "查询某个岗位下所有候选人的完整记录（简历摘要、解析/筛选状态、匹配分数、档位、HR是否已确认推进）。当HR问关于某岗位候选人的任何数量/阈值/名单/进度类问题时调用（例如“筛了多少份”“多少80分以上”“谁确定要面试了”“张三在这个岗位下吗”），返回的是完整原始列表，你自己在列表里数/找/比较来回答，不要指望参数里能直接给你算好的统计结果。注意：候选人的分数/状态只在后端进程运行期间保留，重启后清空，不要凭空编造历史数据。",
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
            "name": "find_candidate",
            "description": "按姓名跨岗位查找候选人是否已经在简历库里（不知道候选人挂在哪个岗位下时用这个；如果已经知道岗位，优先用 get_job_candidates）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_name": {"type": "string", "description": "候选人姓名，支持部分匹配"},
                },
                "required": ["candidate_name"],
            },
        },
    },
]

NAV_MAP = {
    # enter_resume/start_screening 不放在这里：前端在拿到 tool_result 后自己会推送带明确
    # taskId 的通知消息（真正解析/匹配完成时才带按钮），这里再加一条会导致「一句话两个按钮」
    # 且会在任务刚发起（尚未完成）时就提前挂出「查看结果」按钮。
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

    return f"""你是HR简历筛选平台首页的助理，负责五件事：
① 录入JD（新建岗位并解析岗位要求）
② 录入简历（解析并归档到某个岗位下，先不匹配）
③ 对已录入的简历发起批量筛选，生成排名报告
④ 查询或取消正在进行的筛选任务
⑤ 回答关于已有岗位/候选人的信息类问题（不涉及修改任何字段），例如JD写的什么/要求什么、某岗位筛了多少份简历、多少分以上、谁确定要推进面试了、某候选人在不在库里等——这类问题要主动调用 get_job_detail / get_job_candidates / find_candidate 查出原始数据后直接回答，你可以自己对返回的列表做计数/筛选/比较；不要因为具体问法没有预先列在这里就回复"请去管理页面查看"。这些例子只是举例，不是穷举，只要是"只读查询已有信息"类的问题都应该主动查。

现有岗位列表（判断HR说的是哪个岗位时参考，调用工具必须使用其中的 id，不能编造）：
{jobs_lines}

当前进行中的筛选任务（判断HR问的"进度"或要取消的是哪个任务时参考，调用工具必须使用其中的 session_id）：
{sessions_lines}

本轮HR随消息附带的文件：{files_lines}

规则：
- 如果HR要求修改已有JD/岗位要求的字段、删除岗位或简历、修改筛选结果里的判断，都不要尝试处理——你没有对应的工具，也不应该有。直接用自然语言回复，引导HR去对应的管理页面（职位管理 / 简历管理 / 筛选结果）操作，说明具体去哪个页面。查询/了解已有信息不算"修改"，不要引导去管理页面，直接查数据回答。
- HR上传简历时，如果只是要求"录入/保存/先存着"，还没提出要立刻出匹配报告，调用 enter_resume；如果明确要求匹配某岗位、出排名报告，调用 start_screening。如果HR的意图不清楚（比如只说"这是简历"），先反问HR是要先存起来还是马上匹配。
- 如果HR要求批量筛选或录入简历，但没说清楚是哪个岗位，或者本轮没有简历文件附件，不要调用 start_screening / enter_resume，先向HR确认清楚。
- 除了上述五件事，其余情况直接用自然语言回复，不要调用工具。
- 意图识别与边界：只处理简历筛选平台范围内的事（录入JD、录入/筛选简历、查询或取消任务、查询已有信息，以及引导去哪个管理页面）。如果HR问的是与招聘/简历筛选完全无关的问题（比如闲聊、常识问答、写代码、算数、天气等），不要展开回答，用一句话礼貌说明你只负责这个平台上的简历筛选事务，并把话题引导回来。不要编造平台没有的能力。
- 回复措辞简洁，像同事之间对话，不要写成正式公文。
- 输出纯文本，不要使用任何 Markdown 记号（例如 ** 加粗、# 标题、- 列表符号、`代码` 反引号）。需要强调岗位名或候选人名时，直接写文字或用中文书名号《》，不要加星号。
"""


async def _tool_enter_jd(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server
    from backend.tools.document_parser import document_parser
    from backend.tools.text_cleaner import clean_text

    settings = get_settings()
    jd_text = (args.get("jd_text") or "").strip()
    job_name = (args.get("job_name") or "").strip() or "未命名岗位"

    if not jd_text and files:
        upload_dir = _space_upload_dir(settings.app.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
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


async def _tool_enter_resume(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server
    from backend.tools.document_parser import document_parser
    from backend.tools.text_cleaner import clean_text
    from backend.storage.resume_store import get_cached as cache_get, save as cache_save

    settings = get_settings()
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return {"success": False, "message": "缺少 job_id"}
    if not files:
        return {"success": False, "message": "没有检测到简历附件，请先把简历文件拖进来"}

    jobs = dev_server._load_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if not job:
        return {"success": False, "message": "未找到该岗位"}

    upload_dir = _space_upload_dir(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        tmp_path = upload_dir / f"chat_resume_{uuid.uuid4().hex[:8]}_{f.filename}"
        try:
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(f.file, out)

            cached = cache_get(str(tmp_path))
            if cached and cached.get("raw_text") and isinstance(cached.get("parsed_resume"), dict) and cached["parsed_resume"].get("basic_info"):
                resume = cached["parsed_resume"]
            else:
                text, _ = document_parser.parse(str(tmp_path))
                text = clean_text(text, "resume")
                if not text.strip():
                    results.append({"file_name": f.filename, "success": False, "error": "文档解析结果为空"})
                    continue
                resume = await dev_server._call_llm_async(
                    "skills/resume_extractor/prompt.txt", {"raw_resume": text},
                    llm_cfg["api_key"], llm_cfg["base_url"], llm_cfg["model"],
                )
                try:
                    cache_save(str(tmp_path), text, resume, job_id)
                except Exception as e:
                    log.warning(f"[Chat] 简历缓存保存失败 ({f.filename}): {e}")

            name = (resume.get("basic_info") or {}).get("name") or f.filename
            results.append({"file_name": f.filename, "success": True, "name": name, "resume": resume})
        except Exception as e:
            log.error(f"[Chat] 简历录入失败 ({f.filename}): {e}")
            results.append({"file_name": f.filename, "success": False, "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    ok_count = sum(1 for r in results if r["success"])
    return {
        "success": ok_count > 0, "job_id": job_id, "job_name": job.get("name", ""),
        "results": results, "parsed_count": ok_count, "failed_count": len(results) - ok_count,
        "message": "" if ok_count else "全部简历解析失败",
    }


async def _tool_start_screening(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
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


async def _tool_get_task_status(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
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


async def _tool_cancel_task(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server

    sid = (args.get("session_id") or "").strip()
    if not sid:
        return {"success": False, "message": "缺少 session_id"}
    result = await dev_server.session_cancel(sid)
    if isinstance(result, JSONResponse):
        result = json.loads(bytes(result.body))
    return result


async def _tool_get_job_detail(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server

    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return {"success": False, "message": "缺少 job_id"}
    jobs = dev_server._load_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if not job:
        return {"success": False, "message": "未找到该岗位"}

    jd_text = job.get("jd", "") or ""
    return {
        "success": True,
        "job_id": job_id,
        "job_name": job.get("name", ""),
        "jd": jd_text[:800],
        "jd_truncated": len(jd_text) > 800,
        "requirements": job.get("requirements", []),
        "resumeCount": len(dev_server.cache_list_by_job(job_id)),
        "screenedCount": dev_server._job_screened_counts.get(job_id, 0),
    }


def _collect_job_candidates(job_id: str) -> list[dict]:
    """Scan every in-memory AgentSession whose job_id matches, flatten their candidate slots."""
    from backend.agent.session import _sessions

    out = []
    for session in _sessions.values():
        if getattr(session, "job_id", "") != job_id:
            continue
        for c in session.candidates:
            resume = c.resume or {}
            basic = resume.get("basic_info") or {}
            sc = c.scoring or {}
            out.append({
                "name": c.name,
                "file_name": c.file_name,
                "status": c.status,
                "school": basic.get("school", ""),
                "degree": basic.get("degree", ""),
                "major": basic.get("major", ""),
                "current_role": basic.get("current_role", ""),
                "skills": resume.get("skills", []),
                "score": sc.get("overall_score") if isinstance(sc, dict) else None,
                "tier": sc.get("tier_label", "") if isinstance(sc, dict) else "",
            })
    return out


async def _tool_get_job_candidates(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server

    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return {"success": False, "message": "缺少 job_id"}
    jobs = dev_server._load_jobs()
    job = next((j for j in jobs if j.get("id") == job_id), None)
    if not job:
        return {"success": False, "message": "未找到该岗位"}

    candidates = _collect_job_candidates(job_id)
    reviewed_map = {}
    for r in (review_state or []):
        if r.get("job_id") == job_id:
            key = r.get("name") or r.get("file_name")
            if key:
                reviewed_map[key] = bool(r.get("reviewed"))
    for c in candidates:
        c["reviewed"] = reviewed_map.get(c["name"], reviewed_map.get(c["file_name"], False))

    return {
        "success": True,
        "job_id": job_id,
        "job_name": job.get("name", ""),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "note": "候选人分数/状态仅保留在后端进程内存中，服务重启后清空，请勿凭空推测重启前的历史数据",
    }


async def _tool_find_candidate(args: dict, files: list[UploadFile], llm_cfg: dict, review_state: list) -> dict:
    from backend import dev_server
    from backend.storage.resume_store import search_by_name

    name = (args.get("candidate_name") or "").strip()
    if not name:
        return {"success": False, "message": "缺少 candidate_name"}

    jobs = dev_server._load_jobs()
    job_name_by_id = {j.get("id"): j.get("name", "") for j in jobs}
    hits = search_by_name(name)
    for h in hits:
        h["job_name"] = job_name_by_id.get(h.get("job_id", ""), "")

    return {"success": True, "candidate_name": name, "match_count": len(hits), "matches": hits}


TOOL_DISPATCH = {
    "enter_jd": _tool_enter_jd,
    "enter_resume": _tool_enter_resume,
    "start_screening": _tool_start_screening,
    "get_task_status": _tool_get_task_status,
    "cancel_task": _tool_cancel_task,
    "get_job_detail": _tool_get_job_detail,
    "get_job_candidates": _tool_get_job_candidates,
    "find_candidate": _tool_find_candidate,
}


async def chat_turn(history: list, active_sessions: list, files: list[UploadFile], api_key: str, base_url: str, model: str, review_state: list | None = None):
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
                result = await handler(call_args, files, llm_cfg, review_state or [])
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
    review_state: str = Form("[]"),
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
    try:
        review_state_list = json.loads(review_state) if review_state else []
    except json.JSONDecodeError:
        review_state_list = []

    async def event_stream():
        async for chunk in chat_turn(history_list, sessions_list, files, api_key, base_url, model, review_state_list):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
