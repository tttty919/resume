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
from backend.utils.space import upload_dir as _space_upload_dir
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
    session = get_session(sid)
    if not session:
        return

    requirements = session.requirements
    session.phase = "processing"
    emit(sid, "phase", {"phase": "processing"})
    upload_dir = _space_upload_dir(settings.app.upload_dir)

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
            resume_result = await _call_llm("skills/resume_extractor/prompt.txt", {"raw_resume": text}, api_key, base_url, model)
            slot["name"] = resume_result.get("basic_info", {}).get("name", fname)
            slot["resume"] = resume_result

            # ── 自主筛选控制循环（Skill 3 起：匹配 + 自主找补 + 风险 + 推荐）──
            # 原本是写死的 match → risk → recommendation 三步直线；
            # 现在交给 ScreeningAgent.agent_loop 按候选人情况自主编排。
            # 默认用 rule_decide（确定性、可审计、无额外 LLM 调用）；
            # 若想让 LLM 参与决策，把 decide=rule_decide 换成 decide=None 即可。
            from backend.workflows.controller import agent_loop, rule_decide

            emit(sid, "progress", {
                "step": "matching", "index": idx, "total": len(session.candidates),
                "message": f"正在筛选: {slot['name']}...",
            })

            def _on_step(entry: dict, _idx=idx, _name=slot["name"]):
                # 把控制器每一步决策实时推给前端（可审计）
                emit(sid, "agent_step", {"index": _idx, "name": _name, **entry})

            llm_cfg = {"api_key": api_key, "base_url": base_url, "model": model}
            agent_result = await agent_loop(
                requirements, resume_result, text, llm_cfg,
                decide=rule_decide,
                on_step=_on_step,
            )

            items = agent_result["matches"]
            slot["matches"] = items
            slot["analysis"] = agent_result["analysis"]
            slot["scoring"] = agent_result["scoring"]
            slot["pending_hr"] = agent_result.get("pending_hr", [])
            slot["hr_questions"] = agent_result.get("hr_questions", {})
            slot["agent_trace"] = agent_result.get("trace", [])
            slot["status"] = "pending_hr" if slot["pending_hr"] else "done"

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
                "status": slot["status"],
                "pending_hr": slot["pending_hr"],
                "hr_questions": slot["hr_questions"],
                "agent_trace": slot["agent_trace"],
            })

            # 若有必须项需 HR 复核，单独发一条事件，方便前端弹出复核面板
            if slot["pending_hr"]:
                emit(sid, "candidate_pending_hr", {
                    "index": idx, "name": slot["name"],
                    "pending_hr": slot["pending_hr"],
                    "hr_questions": slot["hr_questions"],
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

    # 若有候选人待 HR 复核，延长会话存活时间，避免 HR 还没处理就被清理
    has_pending = any(c.get("status") == "pending_hr" for c in session.candidates)
    cleanup_delay = 3600 if has_pending else 300

    async def _delayed_cleanup():
        await asyncio.sleep(cleanup_delay)
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
    upload_dir = _space_upload_dir(settings.app.upload_dir)
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
    # 存下凭据，供 HR 复核后局部重算
    session.api_key, session.base_url, session.model = api_key, base_url, model

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


@router.post("/hr_review/{sid}")
async def screening_hr_review(sid: str, request: Request):
    """HR 复核回填 (第4步)。

    HR 对某候选人 pending_hr 的要求给出结论后，只重算受影响的要求 + 风险 + 推荐，
    不重跑整个流水线。HR 确认的同义表达会写回知识库 (learn)，越用越准。

    Body: {
      index: int,                       # 候选人序号
      decisions: {                      # 逐条 requirement 的 HR 结论
        "<req_id>": {
          "status": "satisfied|not_satisfied|cannot_judge",
          "confidence": 0~1,            # 可选，默认 0.9
          "evidence": "HR 补充的证据",   # 可选
          "confirmed_synonym": "简历里被HR认可的同义表达"  # 可选 → 写回知识库
        }
      }
    }
    """
    session = get_session(sid)
    if not session:
        return {"success": False, "message": "会话不存在"}

    body = await request.json()
    idx = body.get("index")
    decisions = body.get("decisions", {}) or {}
    if idx is None or not decisions:
        return {"success": False, "message": "缺少 index 或 decisions"}

    slot = next((c for c in session.candidates if c.get("index") == idx), None)
    if slot is None:
        return {"success": False, "message": "候选人不存在"}

    requirements = session.requirements
    req_name = {r.get("id", ""): r.get("name", "") for r in requirements}
    matches = slot.get("matches", [])
    by_id = {m.get("requirement_id", ""): m for m in matches}

    # 1) 应用 HR 结论（覆盖对应要求），并记录审计字段
    for rid, dec in decisions.items():
        m = by_id.get(rid)
        if m is None:
            m = {"requirement_id": rid, "requirement_name": req_name.get(rid, rid)}
            matches.append(m)
            by_id[rid] = m
        m["status"] = dec.get("status", m.get("status", "cannot_judge"))
        m["confidence"] = float(dec.get("confidence", 0.9))
        if dec.get("evidence"):
            m["evidence"] = dec["evidence"]
        m["needs_human_review"] = False
        m["hr_override"] = True

        # 2) 反馈回路：HR 认可的同义词写回 ChromaDB
        syn = dec.get("confirmed_synonym")
        if syn:
            try:
                from backend.skills.semantic_matcher.tools import learn
                learn(req_name.get(rid, rid), syn)
            except Exception as e:
                log.warning(f"[HR复核] 同义词写回失败: {e}")

    # 从待复核列表移除已处理项
    slot["matches"] = matches
    slot["pending_hr"] = [r for r in slot.get("pending_hr", []) if r not in decisions]

    # 3) 只重算受影响：风险 + 推荐（复用 controller.rerun_affected）
    from backend.workflows.controller import rerun_affected
    try:
        rerun = await rerun_affected(
            requirements, matches, matches,
            session.api_key, session.base_url, session.model,
        )
        slot["analysis"] = rerun.get("risk_analysis", slot.get("analysis", {}))
        slot["scoring"] = rerun.get("recommendation", {}).get("scoring", slot.get("scoring", {}))
    except Exception as e:
        log.error(f"[HR复核] 局部重算失败: {e}")
        return {"success": False, "message": f"重算失败: {e}"}

    slot["status"] = "pending_hr" if slot["pending_hr"] else "done"

    emit(sid, "candidate_reviewed", {
        "index": idx, "name": slot.get("name", ""),
        "status": slot["status"],
        "pending_hr": slot["pending_hr"],
        "matches": slot["matches"],
        "analysis": slot["analysis"],
        "scoring": slot["scoring"],
    })

    return {
        "success": True,
        "index": idx,
        "status": slot["status"],
        "pending_hr": slot["pending_hr"],
        "scoring": slot["scoring"],
    }
