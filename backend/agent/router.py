"""Agent API router — session-based screening with HR feedback loop.

Endpoints:
  POST /api/agent/start        — create session, start background processing
  GET  /api/agent/stream/{sid} — SSE event stream (reconnectable)
  POST /api/agent/feedback/{sid} — HR submits feedback, triggers resume
  GET  /api/agent/status/{sid} — current session state snapshot
"""

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File, Request
from starlette.responses import Response
from fastapi.responses import StreamingResponse

from backend.agent.session import (
    create_session, get_session, get_queue, emit,
    pause_candidate, resume_candidate, cleanup_session,
)
from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.tools.document_parser import document_parser
from backend.tools.text_cleaner import clean_text
from backend.storage.resume_store import get_cached as cache_get, save as cache_save
from backend.utils.llm_utils import create_llm, load_prompt, ChatPromptTemplate

settings = get_settings()
log = get_logger()
router = APIRouter(prefix="/api/agent")


def _sse(event_type: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    safe_payload = payload.replace("\n", " ").replace("\r", " ")
    return f"event: {event_type}\ndata: {safe_payload}\n\n"


async def _call_llm(prompt_file: str, template_vars: dict, api_key: str, base_url: str, model: str) -> dict:
    prompt_text = load_prompt(prompt_file)
    llm = create_llm(api_key=api_key, base_url=base_url, model=model)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    import random
    for attempt in range(3):
        try:
            result = await chain.ainvoke(template_vars)
            text = result.content if hasattr(result, "content") else str(result)
            parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            if attempt == 2:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
    return {}


async def _process_candidates(sid: str, api_key: str, base_url: str, model: str):
    """Background task: process each candidate through the agent pipeline."""
    from backend.workflows.nodes.semantic_matcher import match
    from backend.workflows.nodes.risk_analyzer import analyze as analyze_risk
    from backend.workflows.nodes.recommendation_gen import generate as generate_recommendation
    from backend.workflows.controller import ScreeningAgent

    session = get_session(sid)
    if not session:
        return

    requirements = session.requirements
    session.phase = "processing"
    emit(sid, "phase", {"phase": "processing"})

    for slot in session.candidates:
        if slot.status == "done":
            continue

        session.active_idx = slot.index
        agent = ScreeningAgent(requirements)
        tmp_path = None

        try:
            # ── Document parsing (skip if pre-parsed, check SQLite cache) ──
            if slot.raw_text:
                text = slot.raw_text
                emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                    "sub_step": "文档解析", "message": f"正在处理: {slot.file_name} — 文档解析...（已有缓存，跳过）"})
            else:
                emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                    "sub_step": "文档解析", "message": f"正在处理: {slot.file_name} — 文档解析..."})
                upload_dir = Path(settings.app.upload_dir)
                files_found = list(upload_dir.glob(f"agent_{slot.index}_*"))
                if files_found:
                    tmp_path = files_found[0]
                    # Check SQLite cache by file MD5
                    cached = cache_get(str(tmp_path))
                    if cached and cached.get("raw_text"):
                        text = cached["raw_text"]
                        emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                            "sub_step": "文档解析", "message": f"正在处理: {slot.file_name} — 文档解析...（SQLite 缓存命中，跳过）"})
                    else:
                        text, _ = document_parser.parse(str(tmp_path))
                        text = clean_text(text, "resume")
                else:
                    emit(sid, "candidate_error", {"index": slot.index, "name": slot.file_name, "error": "未找到简历文件"})
                    continue

            if not text.strip():
                emit(sid, "candidate_error", {"index": slot.index, "name": slot.name or slot.file_name, "error": "解析结果为空"})
                continue

            # ── Resume extraction (skip if pre-parsed or cached) ──
            if isinstance(slot.resume, dict) and slot.resume.get("basic_info"):
                resume_result = slot.resume
                slot.name = resume_result.get("basic_info", {}).get("name", slot.file_name)
                emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                    "sub_step": "简历提取", "message": f"正在处理: {slot.name or slot.file_name} — 简历提取...（已有缓存，跳过）"})
            else:
                if not isinstance(slot.resume, dict):
                    log.warning(f"[Agent] slot.resume 类型异常: {type(slot.resume)}, 将重新提取")
                    slot.resume = {}
                # Check SQLite cache for LLM extraction result
                cached_resume = None
                if tmp_path:
                    cached = cache_get(str(tmp_path))
                    if cached and cached.get("parsed_resume") and isinstance(cached["parsed_resume"], dict) and cached["parsed_resume"].get("basic_info"):
                        cached_resume = cached["parsed_resume"]
                        emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                            "sub_step": "简历提取", "message": f"正在处理: {slot.name or slot.file_name} — 简历提取...（SQLite 缓存命中，跳过）"})

                if cached_resume:
                    resume_result = cached_resume
                else:
                    emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                        "sub_step": "简历提取", "message": f"正在处理: {slot.name or slot.file_name} — 简历提取..."})
                    resume_result = await _call_llm("resume_extractor.txt", {"raw_resume": text}, api_key, base_url, model)
                    if not isinstance(resume_result, dict):
                        raise RuntimeError(f"LLM 返回类型异常: {type(resume_result)}")
                    # Save to cache
                    if tmp_path:
                        try:
                            cache_save(str(tmp_path), text, resume_result)
                        except Exception as e:
                            log.warning(f"[Agent] 缓存写入失败 ({slot.file_name}): {e}")

                slot.name = resume_result.get("basic_info", {}).get("name", slot.file_name)
                slot.resume = resume_result

            slot.raw_text = text
            slot.status = "matching"

            # ── Matching ──
            emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                "sub_step": "语义匹配+证据", "message": f"正在处理: {slot.name} — 语义匹配+证据..."})

            match_result = await match(requirements, resume_result, text, api_key, base_url, model)
            items = match_result.get("matches", [])
            slot.matches = items

            decision = agent.decide_after_match(items)
            should_stop = decision["should_stop"]
            stop_reason = decision["reason"]
            pending_hr = decision["pending_hr"]
            emit(sid, "agent_check", {
                "index": slot.index, "name": slot.name,
                "action": decision["action"],
                "should_stop": should_stop, "stop_reason": stop_reason,
                "pending_hr": pending_hr, "retry_count": decision["retry_count"],
            })

            # Attempt auto-retry for unresolved must items
            unresolved = decision["unresolved"]
            if decision["action"] == "retry_unresolved":
                retry_count = agent.mark_retry()
                emit(sid, "agent_retry", {
                    "index": slot.index, "name": slot.name,
                    "message": f"Agent: {len(unresolved)} 项要求未解决，重试第 {retry_count} 次...",
                    "unresolved_ids": [r["id"] for r in unresolved],
                })
                retry_result = await match(unresolved, resume_result, text, api_key, base_url, model)
                retry_map = {m.get("requirement_id", ""): m for m in retry_result.get("matches", [])}
                for item in items:
                    if item.get("requirement_id", "") in retry_map:
                        item.update(retry_map[item["requirement_id"]])
                decision = agent.decide_after_retry(items)
                should_stop = decision["should_stop"]
                stop_reason = decision["reason"]
                pending_hr = decision["pending_hr"]
                emit(sid, "agent_check", {
                    "index": slot.index, "name": slot.name,
                    "action": decision["action"],
                    "should_stop": should_stop, "stop_reason": stop_reason,
                    "pending_hr": pending_hr, "retry_count": decision["retry_count"],
                })

            # If agent says stop and HR needed, pause this candidate
            if should_stop and pending_hr:
                pause_candidate(sid, slot.index, stop_reason, pending_hr)
                continue  # skip risk + recommend for now

            # ── Risk Analysis ──
            emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                "sub_step": "风险分析", "message": f"正在处理: {slot.name} — 风险分析..."})
            risk_result = await analyze_risk(requirements, items, api_key, base_url, model)
            slot.analysis = risk_result.get("analysis", {})
            slot.status = "risk_analysis"

            # ── Recommendation ──
            emit(sid, "progress", {"step": "resume_step", "index": slot.index, "total": len(session.candidates),
                "sub_step": "生成推荐", "message": f"正在处理: {slot.name} — 生成推荐..."})
            rec_result = await generate_recommendation(requirements, items, items, slot.analysis, api_key, base_url, model)
            slot.scoring = rec_result.get("scoring", {})
            slot.status = "done"
            summary = rec_result.get("summary", {})
            recommendation_reason = rec_result.get("recommendation_reason", "")

            emit(sid, "candidate_done", {
                "index": slot.index, "name": slot.name, "file_name": slot.file_name,
                "score": slot.scoring.get("overall_score"),
                "tier": slot.scoring.get("tier_label"),
                "tier_reason": slot.scoring.get("tier_reason", ""),
                "satisfied": slot.scoring.get("counts", {}).get("satisfied"),
                "not_satisfied": slot.scoring.get("counts", {}).get("not_satisfied"),
                "cannot_judge": slot.scoring.get("counts", {}).get("cannot_judge"),
                "recommendation": summary.get("recommendation"),
                "recommendation_reason": recommendation_reason,
                "core_advantages": summary.get("core_advantages", []),
                "key_risks": summary.get("key_risks", []),
                "human_review_questions": summary.get("human_review_questions", []),
                "interview_suggestions": slot.analysis.get("interview_suggestions", []),
                "resume": slot.resume,
                "matches": slot.matches,
                "analysis": slot.analysis,
                "scoring": slot.scoring,
            })

            log.info(f"[Agent] {slot.name}: {slot.scoring.get('tier_label')} (得分: {slot.scoring.get('overall_score')})")

        except Exception as e:
            log.error(f"[Agent] {slot.name or slot.file_name} 失败: {e}")
            slot.status = "error"
            emit(sid, "candidate_error", {"index": slot.index, "name": slot.name or slot.file_name, "error": str(e)})

    # ── Final state check ──
    paused = [s for s in session.candidates if s.status == "paused"]
    if paused:
        session.phase = "waiting_hr"
        emit(sid, "agent_waiting_hr", {
            "message": f"Agent 已暂停，{len(paused)} 位候选人需要 HR 复核",
            "paused_candidates": [{"index": s.index, "name": s.name, "pending_hr": s.pending_hr} for s in paused],
        })
    else:
        session.phase = "done"
    emit(sid, "complete", {"message": "Agent 处理完毕", "phase": session.phase})

    # Schedule cleanup after 5-minute TTL to allow reconnection
    async def _delayed_cleanup():
        await asyncio.sleep(300)
        cleanup_session(sid)
    asyncio.create_task(_delayed_cleanup())


@router.post("/start")
async def agent_start(
    jd_text: str = Form(""),
    files: list[UploadFile] = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    pre_parsed: str = Form(""),
):
    """Create an agent session and start background processing."""
    api_key = api_key or settings.deepseek.api_key
    base_url = base_url or settings.deepseek.base_url
    model = model or settings.deepseek.model

    if not jd_text.strip():
        return {"success": False, "message": "JD 文本不能为空"}
    if not files:
        return {"success": False, "message": "请至少上传一份简历"}
    if not api_key:
        return {"success": False, "message": "请填写 API Key"}

    # ---- JD parse ----
    try:
        jd_clean = clean_text(jd_text, "jd")
        jd_result = await _call_llm("jd_parser.txt", {"raw_jd": jd_clean}, api_key, base_url, model)
        requirements = jd_result.get("requirements", [])
    except Exception as e:
        return {"success": False, "message": f"JD 解析失败: {e}"}

    # ---- Parse pre_parsed data ----
    pre_parsed_list = None
    if pre_parsed:
        try:
            pre_parsed_list = json.loads(pre_parsed)
        except json.JSONDecodeError:
            pass

    # ---- Save files ----
    upload_dir = Path(settings.app.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    candidate_names = []
    for idx, file in enumerate(files):
        fname = file.filename or f"resume_{idx}"
        candidate_names.append(fname)
        tmp_path = upload_dir / f"agent_{idx}_{fname}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # ---- Create session ----
    session = create_session(requirements, candidate_names, pre_parsed_list)
    sid = session.id

    # ---- Start background processing ----
    asyncio.create_task(_process_candidates(sid, api_key, base_url, model))

    return {"success": True, "session_id": sid, "requirements": requirements, "total": len(candidate_names)}


@router.get("/stream/{sid}")
async def agent_stream(sid: str):
    """SSE event stream for an agent session. Reconnectable — replays missed events."""
    session = get_session(sid)
    if not session:
        return Response(content=_sse("error", {"message": "会话不存在"}), media_type="text/event-stream")

    q = get_queue(sid)
    if not q:
        return Response(content=_sse("error", {"message": "会话队列丢失"}), media_type="text/event-stream")

    sent_count = len(session.events)

    async def event_stream():
        nonlocal sent_count
        # Replay events already in the log
        for evt_type, payload in session.events:
            yield _sse(evt_type, payload)

        # Stream new events as they arrive
        if session.phase == "done":
            return
        while True:
            try:
                evt_type, payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield _sse(evt_type, payload)
                if evt_type == "complete":
                    break
            except asyncio.TimeoutError:
                yield _sse("heartbeat", {"message": "等待中..."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/feedback/{sid}")
async def agent_feedback(sid: str):
    """HR submits feedback for paused candidates. Triggers resume processing."""
    from fastapi import Request

    # We need the request body; access via dependency injection doesn't work here
    # because FastAPI can't mix Form with JSON easily. Let's read the raw body.
    return {"success": False, "message": "Use POST with JSON body containing overrides"}


@router.post("/resume/{sid}")
async def agent_resume(sid: str, request: Request):
    """HR resumes: expects JSON body with overrides dict keyed by 'candidate_index-requirement_id'.

    Triggers rerun_affected() for paused candidates, then continues processing.
    Reconnect to GET /stream/{sid} to get new events.
    """
    from backend.workflows.controller import rerun_affected
    from backend.workflows.nodes.semantic_matcher import match

    session = get_session(sid)
    if not session:
        return {"success": False, "message": "会话不存在"}

    try:
        body = await request.json()
    except Exception:
        return {"success": False, "message": "请求体必须是 JSON"}

    overrides = body.get("overrides", {})  # { "cand_idx-req_id": { new_status, reason } }
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model

    if not overrides:
        return {"success": False, "message": "未提供任何复核修改"}

    # Apply overrides to paused candidates
    for slot in session.candidates:
        if slot.status != "paused":
            continue
        slot_key = str(slot.index)
        slot_overrides = {k: v for k, v in overrides.items() if k.startswith(slot_key + "-")}
        if not slot_overrides:
            continue

        # Apply to matches
        for item in slot.matches:
            rid = item.get("requirement_id", "")
            key = f"{slot.index}-{rid}"
            if key in slot_overrides:
                ov = slot_overrides[key]
                item["status"] = ov.get("new_status", item["status"])
                item["_hr_overridden"] = True
                item["confidence"] = max(item.get("confidence", 0.5), 0.85)  # HR confirms boosts confidence

        # Mark for resume
        emit(sid, "agent_resume", {"index": slot.index, "name": slot.name, "message": f"HR 已复核 {slot.name}，Agent 继续处理"})

        # Rerun affected (risk + recommendation)
        try:
            result = await rerun_affected(session.requirements, slot.matches, slot.matches, api_key, base_url, model)
            slot.analysis = result.get("risk_analysis", {})
            rec_full = result.get("recommendation", {})
            slot.scoring = rec_full.get("scoring", {})
            slot.status = "done"
            summary = rec_full.get("summary", {})
            recommendation_reason = rec_full.get("recommendation_reason", "")

            emit(sid, "candidate_done", {
                "index": slot.index, "name": slot.name, "file_name": slot.file_name,
                "score": slot.scoring.get("overall_score"),
                "tier": slot.scoring.get("tier_label"),
                "tier_reason": slot.scoring.get("tier_reason", ""),
                "satisfied": slot.scoring.get("counts", {}).get("satisfied"),
                "not_satisfied": slot.scoring.get("counts", {}).get("not_satisfied"),
                "cannot_judge": slot.scoring.get("counts", {}).get("cannot_judge"),
                "recommendation": summary.get("recommendation"),
                "recommendation_reason": recommendation_reason,
                "core_advantages": summary.get("core_advantages", []),
                "key_risks": summary.get("key_risks", []),
                "human_review_questions": summary.get("human_review_questions", []),
                "interview_suggestions": (slot.analysis or {}).get("interview_suggestions", []),
                "resume": slot.resume,
                "matches": slot.matches,
                "analysis": slot.analysis,
                "scoring": slot.scoring,
            })
            log.info(f"[Agent.Resume] {slot.name}: done (得分: {slot.scoring.get('overall_score')})")
        except Exception as e:
            log.error(f"[Agent.Resume] {slot.name} 失败: {e}")
            slot.status = "error"
            emit(sid, "candidate_error", {"index": slot.index, "name": slot.name, "error": str(e)})

    # Check if all done
    remaining = [s for s in session.candidates if s.status not in ("done", "error")]
    if not remaining:
        session.phase = "done"
    emit(sid, "complete", {"message": "Agent 继续处理完毕", "phase": session.phase})

    return {"success": True, "session_id": sid, "phase": session.phase}


@router.get("/status/{sid}")
async def agent_status(sid: str):
    """Return current session state snapshot (for polling)."""
    session = get_session(sid)
    if not session:
        return {"success": False, "message": "会话不存在"}

    return {
        "success": True,
        "session_id": sid,
        "phase": session.phase,
        "total": len(session.candidates),
        "active_idx": session.active_idx,
        "candidates": [
            {
                "index": s.index, "name": s.name, "file_name": s.file_name,
                "status": s.status, "stop_reason": s.stop_reason,
                "pending_hr": s.pending_hr, "retry_count": s.retry_count,
                "score": s.scoring.get("overall_score") if s.scoring else None,
                "tier": s.scoring.get("tier_label") if s.scoring else None,
            }
            for s in session.candidates
        ],
        "requirements": session.requirements,
    }
