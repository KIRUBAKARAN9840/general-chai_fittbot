# routes/ai_chatbot.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
import pytz, os, hashlib, orjson
from datetime import datetime
import io
from sqlalchemy.orm import Session
from app.models.database import get_db


# ⚠️ Use only dependency callables; don't annotate them with Redis/OpenAI types here.
from app.models.deps import get_http, get_oai, get_mem
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.kb_store import KB
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
    PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
    build_messages, heuristic_confidence, OPENAI_MODEL,
    sse_json, sse_escape, gpt_small_route, is_yes, is_no, is_fitness_related,
    has_action_verb, is_fittbot_meta_query, is_plan_request, STYLE_PLAN, STYLE_CHAT_FORMAT, pretty_plan
)
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.report_analysis import (
    is_analysis_intent,
    is_followup_question,
    set_mode, get_mode,
    set_analysis_artifacts, get_analysis_artifacts,
    build_analysis_dataset_dict,
    build_summary_hints,         # if you need it elsewhere
    run_analysis_generator,      # <-- main async generator
    STYLE_INSIGHT_REPORT,        # optional if referenced in follow-ups
)


from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.asr import transcribe_audio

router=APIRouter(prefix="/chatbot",tags=["chatbot"])


APP_ENV = os.getenv("APP_ENV", "prod")
TZNAME  = os.getenv("TZ", "Asia/Kolkata")
IST     = pytz.timezone(TZNAME)

class KBUpsertIn(BaseModel):
    source: str
    text: str

class KBSearchIn(BaseModel):
    query: str
    k: int = 4

@router.get("/healthz")
async def healthz():
    return {"ok": True, "env": APP_ENV, "tz": TZNAME, "kb_chunks": len(KB.texts)}



class RichTextStreamFilter:
    def __init__(self): self.buf = ""
    def feed(self, ch: str) -> str:
        if not ch: return ""
        # Just normalize newlines; keep markdown intact
        ch = ch.replace("\r\n", "\n").replace("\r", "\n")
        self.buf += ch
        # For streaming, emit as we get it (no markdown cleaning)
        out, self.buf = self.buf, ""
        return out
    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


import re

def pretty_plan(markdown: str) -> str:
    if not markdown:
        return ""

    txt = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize headings → blank line + Title
    txt = re.sub(r'^\s*#{1,6}\s*', '', txt, flags=re.M)  # drop leading #'s
    # Convert **bold** to plain text
    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
    # Convert *italic* to plain text
    txt = re.sub(r'\*(.*?)\*', r'\1', txt)

    # Numbered lists: "1. Something" → "1) Something"
    txt = re.sub(r'^\s*(\d+)\.\s*', r'\1) ', txt, flags=re.M)

    # Bullets: "- something" or "• something" → "• something"
    txt = re.sub(r'^\s*[-•]\s*', '• ', txt, flags=re.M)

    # Ensure a space after colons/comma if missing
    txt = re.sub(r':(?!\s)', ': ', txt)
    txt = re.sub(r',(?!\s)', ', ', txt)

    # Fix "words-smashed" by accidental no-space around hyphens
    # "withwhole-grain" → "with whole-grain"
    txt = re.sub(r'([A-Za-z])([-–—])([A-Za-z])', r'\1 \2 \3', txt)

    # Collapse triple+ newlines; ensure max 2 in a row
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = "\n".join(line.rstrip() for line in txt.split("\n"))

    return txt.strip()


@router.get("/chat/stream_test", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def chat_stream(
    user_id: int,
    text: str = Query(...),
    mem = Depends(get_mem),
    oai  = Depends(get_oai),
    db: Session = Depends(get_db),   
):
    if not user_id or not text.strip():
        raise HTTPException(400, "user_id and text required")

    now_iso = datetime.now(IST).isoformat()
    tlower  = text.lower().strip()

    # pend is always a dict
    pend = (await mem.get_pending(user_id)) or {}
    mode = await get_mode(mem, user_id)  # "analysis" or None

    # If user explicitly goes to plan/chat, exit analysis mode (so routes stay clean)
    if is_plan_request(tlower):
        await set_mode(mem, user_id, None)

    if pend.get("state") == "awaiting_analysis_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            await set_mode(mem, user_id, "analysis")
            return StreamingResponse(
                run_analysis_generator(db, mem, oai, user_id),
                media_type="text/event-stream",
                headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
            )
        if is_no(text):
            await mem.set_pending(user_id, None)
            async def _cancel_an():
                yield sse_json({"type":"analysis","status":"cancelled"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_cancel_an(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        async def _clar_an():
            yield sse_json({"type":"analysis","status":"confirm","prompt":"Shall I start the analysis?"})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_clar_an(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # ---- NAV confirm ----
    if pend.get("state") == "awaiting_nav_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            async def _nav_yes():
                yield sse_json({"type":"nav","is_navigation": True,
                                "prompt":"Thanks for your confirmation. Redirecting to today's diet logs"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_nav_yes(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        if is_no(text):
            await mem.set_pending(user_id, None)
            async def _nav_no():
                yield sse_json({"type":"nav","is_navigation": False,
                                "prompt":"Thanks for your response. You can continue chatting here."})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_nav_no(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        async def _nav_clar():
            yield sse_json({"type":"nav","status":"confirm","prompt":""})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_nav_clar(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # =========================
    # 2) FRESH TURN: TRIGGERED ACTIONS
    # =========================

    # --- analysis start or follow-ups
    # follow-ups while already in analysis mode
    if mode == "analysis" and not is_plan_request(tlower):
        if is_followup_question(text):
            dataset, summary = await get_analysis_artifacts(mem, user_id)
            if dataset:
                await mem.add(user_id, "user", text.strip())
                msgs = [
                    {"role":"system","content": GENERAL_SYSTEM},
                    {"role":"system","content": STYLE_CHAT_FORMAT},
                    {"role":"system","content":
                        "ANALYSIS_MODE=ON. Use this context without re-querying DB.\n"
                        f"DATASET:\n{orjson.dumps(dataset).decode()}\n\n"
                        f"PRIOR_SUMMARY:\n{summary or ''}\n"
                        "Answer follow-up concisely with numbers where helpful."
                    },
                    {"role":"user","content": text.strip()},
                ]
                resp = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=0)
                import re as _re
                content = (resp.choices[0].message.content or "").strip()
                content = _re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=_re.I)
                pretty = pretty_plan(content)

                async def _one_shot_followup():
                    yield sse_escape(pretty)
                    await mem.add(user_id, "assistant", pretty)
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_one_shot_followup(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # start analysis flow if user asks for it and we aren't already mid-flow
    if is_analysis_intent(tlower) and not pend:
        await mem.set_pending(user_id, {"state":"awaiting_analysis_confirm"})
        async def _ask_confirm():
            yield sse_json({"type":"analysis","status":"confirm",
                            "prompt":"Sure—let me analyse your diet and workout data. Shall we start?"})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_ask_confirm(), media_type="text/event-stream",
                                headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # =========================
    # 3) NORMAL CHAT (fallback)
    # =========================
    
    # Check if the query is fitness-related first
    if not is_fitness_related(text) and not is_fittbot_meta_query(text):
        async def _not_fitness():
            redirect_msg = "I'm a specialized fitness assistant and can only help with exercise, health, and wellness topics. How can I help you with your fitness journey today?"
            yield sse_escape(redirect_msg)
            await mem.add(user_id, "user", text.strip())
            await mem.add(user_id, "assistant", redirect_msg)
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_not_fitness(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    
    is_meta = is_fittbot_meta_query(text) and not (is_plan_request(text) or is_fitness_related(text))
    is_plan = is_plan_request(text) or is_fitness_related(text)

    await mem.add(user_id, "user", text.strip())

    if is_plan:
        msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
                                       context_only=False, k=TOP_K)
        msgs.insert(1, {"role": "system", "content": STYLE_PLAN})
        temperature = 0
    else:
        msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
                                       context_only=is_meta, k=8 if is_meta else TOP_K)
        msgs.insert(1, {"role": "system", "content": STYLE_CHAT_FORMAT})
        temperature = 0

    resp    = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=temperature)
    content = (resp.choices[0].message.content or "").strip()

    import re as _re
    content = _re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=_re.I)
    
    # Remove food logging prompts
    content = _re.sub(r'Would you like to log more foods.*?\?.*?🍏?', '', content, flags=_re.I | _re.DOTALL)
    content = _re.sub(r'Let me know.*?log.*?for you.*?🍏?', '', content, flags=_re.I | _re.DOTALL)
    content = _re.sub(r'Do you want.*?log.*?\?', '', content, flags=_re.I)

    pretty  = pretty_plan(content)
    async def _one_shot():
        yield sse_escape(pretty)
        await mem.add(user_id, "assistant", pretty)
        yield "event: done\ndata: [DONE]\n\n"
    return StreamingResponse(_one_shot(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ---- KB endpoints (same names) ----
@router.post("/kb/upsert")
async def kb_upsert(inp: KBUpsertIn):
    return {"added_chunks": KB.upsert(inp.source, inp.text)}

@router.post("/kb/search")
async def kb_search(inp: KBSearchIn):
    return {"hits": KB.search(inp.query, k=inp.k)}

@router.post("/kb/upsert_file")
async def kb_upsert_file(
    src: str = Depends(lambda: "upload"),
    file: UploadFile = File(...),
):
    data = await file.read()
    if file.filename.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    elif file.filename.endswith((".docx", ".doc")):
        from docx import Document
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    else:
        text = data.decode("utf-8", "ignore")
    return {"added_chunks": KB.upsert(src or file.filename, text)}

@router.post("/voice/transcribe", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def voice_transcribe(
    audio: UploadFile = File(...),
    http = Depends(get_http),
    oai  = Depends(get_oai),
):
    # 1) Transcribe
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")


    

    # 2) Detect language + translate to English
    def _translate_to_english(text: str) -> dict:
        try:
            sys = (
                "You are a translator. Output ONLY JSON like "
                "{\"lang\":\"xx\",\"english\":\"...\"}. "
                "Detect source language code (ISO-639-1 if possible). "
                "Translate to natural English. Do not add extra words. "
                "Keep food names recognizable; use common transliterations if needed."
            )
            resp = oai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            data = orjson.loads(resp.choices[0].message.content)
            lang = (data.get("lang") or "unknown").strip()
            eng  = (data.get("english") or text).strip()
            return {"lang": lang, "english": eng}
        except Exception:
            # fail-safe: return original text as English
            return {"lang":"unknown","english":text}

    tinfo = _translate_to_english(transcript)
    transcript_en = tinfo["english"]
    lang_code     = tinfo["lang"]

    print("vanakam")
    
    print(f"[voice] lang={lang_code} raw={transcript!r} | en={transcript_en!r}")

    # Return both raw & translated for the client
    return {
        "transcript": transcript_en,
        "lang": lang_code,
        "english": transcript_en,
    }




@router.post("/voice/stream_test", dependencies=[Depends(RateLimiter(times=20, seconds=60))])
async def voice_stream_sse(
    user_id: int,
    audio: UploadFile = File(...),
    mem   = Depends(get_mem),
    oai   = Depends(get_oai),
    http  = Depends(get_http),
):
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")

    await mem.add(user_id, "user", transcript)
    msgs, _ = await build_messages(user_id, transcript, use_context=True, oai=oai, mem=mem)
    stream = oai_chat_stream(msgs, oai)
    filt   = PlainTextStreamFilter()

    async def token_iter():
        parts = []
        try:
            for chunk in stream:
                choice = chunk.choices[0]
                if choice.delta and getattr(choice.delta, "content", None):
                    token = choice.delta.content
                    emit = filt.feed(token)
                    if emit:
                        parts.append(emit)
                        yield sse_escape(emit)
        finally:
            tail = filt.flush()
            if tail:
                parts.append(tail)
                yield sse_escape(tail)
            final = "".join(parts)
            if final.strip():
                await mem.add(user_id, "assistant", final.strip())
            yield "event: done\ndata: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(token_iter(), media_type="text/event-stream; charset=utf-8", headers=headers)


class Userid(BaseModel):
    user_id: int

@router.post("/delete_chat")
async def chat_close(
    req: Userid,
    mem = Depends(get_mem),
):

    print(f"Deleting chat history for user {req.user_id}")

    history_key = f"chat:{req.user_id}:history"
    pending_key = f"chat:{req.user_id}:pending"
    deleted = await mem.r.delete(history_key, pending_key)

    return {
        "status":200
    }


@router.delete("/kb/clear")
async def kb_clear():
    """Clear all KB content completely"""
    initial_count = len(KB.texts)
    KB.texts.clear()  # Clear all stored text chunks
    
    return {
        "status": "cleared", 
        "cleared_chunks": initial_count,
        "remaining_chunks": len(KB.texts)
    }

@router.get("/kb/status")
async def kb_status():
    """Check current KB status"""
    return {
        "total_chunks": len(KB.texts),
        "kb_empty": len(KB.texts) == 0
    }
