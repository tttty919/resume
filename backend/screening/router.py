"""Screening API router — background screening with reconnectable SSE."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from starlette.responses import Response

from backend.screening.session import (
    create_session, get_session, get_queue, emit, cleanup_session,
)
from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.tools.document_parser import document_parser
from backend.tools.text_cleaner import clean_text
from backend.storage.resume_store import save as cache_save
from backend.utils.llm_utils import create_llm, load_prompt, ChatPromptTemplate

settings = get_settings()
log = get_logger()
router = APIRouter(prefix="/api/screening")


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


async def _process_screening(
    sid: str, job_id: str, files: list[tuple[str, bytes]],
    api_key: str, base_url: str, model: str,
):
    """Background task: process each resume through the screening pipeline."""
    from backend.workflows.nodes.semantic_matcher import match
    from backend.workflows.nodes.risk_analyzer import analyze as analyze_risk
    from backend.workflows.nodes.recommendation_gen import generate as generate_recommendation

    session = get_session(sid)
    if not session:
        return

    requirements = session.requirements
    session.phase = "processing"
    emit(sid, "phase", {"phase": "processing"})
    upload_dir = Path(settings.app.upload_dir)

    for slot in session.candidates:
        idx = slot["index"]
        fname, file_bytes = files[idx]
        slot["status"] = "parsing"
        session.active_idx = idx

        try:
            # ── Document parsing ──
            emit(sid, "progress", {
                "step": "parsing", "index": idx, "total": len(session.candidates),
                "message": f"正在解析: {fname}...",
            })
            tmp_path = upload_dir / f"screening_{idx}_{fname}"
            tmp_path.write_bytes(file_bytes)

            text, _ = document_parser.parse(str(tmp_path))
            text = clean_text(text, "resume")
            slot["raw_text"] = text

            if not text.strip():
                emit(sid, "candidate_error", {"index": idx, "name": fname, "error": "解析结果为空"})
                slot["status"] = "error"
                continue

            # ── Resume extraction (LLM) ──
            emit(sid, "progress", {
                "step": "extracting", "index": idx, "total": len(session.candidates),
                "message": f"正在提取: {fname}...",
            })
            resume_result = await _call_llm("resume_extractor.txt", {"raw_resume": text}, api_key, base_url, model)
            slot["name"] = resume_result.get("basic_info", {}).get("name", fname)
            slot["resume"] = resume_result

            # ── Matching ──
            emit(sid, "progress", {
                "step": "matching", "index": idx, "total": len(session.candidates),
                "message": f"正在匹配: {slot['name']}...",
            })
            match_result = await match(requirements, resume_result, text, api_key, base_url, model)
            items = match_result.get("matches", [])
            slot["matches"] = items

            # ── Risk Analysis ──
            emit(sid, "progress", {
                "step": "risk_analysis", "index": idx, "total": len(session.candidates),
                "message": f"风险评估: {slot['name']}...",
            })
            risk_result = await analyze_risk(requirements, items, api_key, base_url, model)
            slot["analysis"] = risk_result.get("analysis", {})

            # ── Recommendation ──
            emit(sid, "progress", {
                "step": "recommendation", "index": idx, "total": len(session.candidates),
                "message": f"生成推荐: {slot['name']}...",
            })
            rec_result = await generate_recommendation(requirements, items, None, slot["analysis"], api_key, base_url, model)
            slot["scoring"] = rec_result.get("scoring", {})
            slot["status"] = "done"

            emit(sid, "candidate_done", {
                "index": idx, "name": slot["name"], "file_name": fname,
                "score": slot["scoring"].get("overall_score"),
                "tier": slot["scoring"].get("tier_label"),
                "satisfied": slot["scoring"].get("counts", {}).get("satisfied"),
                "not_satisfied": slot["scoring"].get("counts", {}).get("not_satisfied"),
                "cannot_judge": slot["scoring"].get("counts", {}).get("cannot_judge"),
                "resume": slot["resume"],
                "matches": slot["matches"],
                "analysis": slot["analysis"],
                "scoring": slot["scoring"],
            })

            # ── Save to cache ──
            try:
                cache_save(str(tmp_path), text, resume_result, job_id)
            except Exception:
                pass

            log.info(f"[Screening] {slot['name']}: done (score: {slot['scoring'].get('overall_score')})")

        except Exception as e:
            log.error(f"[Screening] {slot.get('name', fname)} failed: {e}")
            slot["status"] = "error"
            emit(sid, "candidate_error", {"index": idx, "name": slot.get("name", fname), "error": str(e)})

    session.phase = "done"
    emit(sid, "complete", {"message": "筛选完成", "phase": "done"})

    async def _delayed_cleanup():
        await asyncio.sleep(300)
        cleanup_session(sid)
    asyncio.create_task(_delayed_cleanup())


@router.post("/start")
async def screening_start(request: Request):
    """Create a screening session and start background processing.

    Expects JSON: { job_id, requirements, files: [{name, size}], api_key?, base_url?, model? }
    The files should already be uploaded to the upload directory.
    """
    body = await request.json()
    job_id = body.get("job_id", "")
    files_meta = body.get("files", [])
    api_key = body.get("api_key") or settings.deepseek.api_key
    base_url = body.get("base_url") or settings.deepseek.base_url
    model = body.get("model") or settings.deepseek.model
    requirements = body.get("requirements", [])

    if not requirements:
        return {"success": False, "message": "岗位要求为空"}
    if not files_meta:
        return {"success": False, "message": "请至少上传一份简历"}
    if not api_key:
        return {"success": False, "message": "请填写 API Key"}

    # Load file content from upload dir
    upload_dir = Path(settings.app.upload_dir)
    files_data: list[tuple[str, bytes]] = []
    candidate_names: list[str] = []
    for fm in files_meta:
        fname = fm.get("name", "")
        candidate_names.append(fname)
        # Find the file in uploads (matching by name pattern)
        found = list(upload_dir.glob(f"screening_*_{fname}")) or list(upload_dir.glob(f"*_{fname}"))
        if found:
            files_data.append((fname, found[0].read_bytes()))
        else:
            files_data.append((fname, b""))

    session = create_session(job_id, requirements, candidate_names)
    sid = session.id

    asyncio.create_task(_process_screening(sid, job_id, files_data, api_key, base_url, model))

    return {"success": True, "session_id": sid, "total": len(candidate_names)}


@router.get("/stream/{sid}")
async def screening_stream(sid: str):
    """SSE event stream for a screening session. Reconnectable."""
    session = get_session(sid)
    if not session:
        return Response(content=_sse("error", {"message": "会话不存在"}), media_type="text/event-stream")

    q = get_queue(sid)
    if not q:
        return Response(content=_sse("error", {"message": "会话队列丢失"}), media_type="text/event-stream")

    async def event_stream():
        # Replay events already in the log
        for evt_type, payload in session.events:
            yield _sse(evt_type, payload)

        # If session is already done, stop after replay
        if session.phase == "done":
            return

        # Stream new events as they arrive
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


@router.get("/status/{sid}")
async def screening_status(sid: str):
    """Return current session state snapshot."""
    session = get_session(sid)
    if not session:
        return {"success": False, "message": "会话不存在"}

    return {
        "success": True,
        "session_id": sid,
        "phase": session.phase,
        "total": len(session.candidates),
        "active_idx": session.active_idx,
    }
