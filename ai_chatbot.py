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
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.nutrition import FOOD_DB
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
    PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
    build_messages, heuristic_confidence, gpt_extract_items, first_missing_quantity,OPENAI_MODEL,
    sse_json, sse_escape, gpt_small_route, _scale_macros, is_yes, is_no,is_fit_chat,
    has_action_verb, food_hits,ensure_per_unit_macros, is_fittbot_meta_query,normalize_food, explicit_log_command, STYLE_PLAN, is_plan_request,STYLE_CHAT_FORMAT,pretty_plan
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

    # Fix “words-smashed” by accidental no-space around hyphens
    # "withwhole-grain" → "with whole-grain"
    txt = re.sub(r'([A-Za-z])([-–—])([A-Za-z])', r'\1 \2 \3', txt)

    # Collapse triple+ newlines; ensure max 2 in a row
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = "\n".join(line.rstrip() for line in txt.split("\n"))

    return txt.strip()


_MACRO_KEYS = ("calories","protein","carbs","fat","fiber","sugar")


# @router.get("/chat/stream_test", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
# async def chat_stream(
#     user_id: int,
#     text: str = Query(...),
#     mem = Depends(get_mem),
#     oai  = Depends(get_oai),
#     db: Session = Depends(get_db),           # === ANALYSIS
# ):
#     if not user_id or not text.strip():
#         raise HTTPException(400, "user_id and text required")

#     now_iso = datetime.now(IST).isoformat()
#     pend    = await mem.get_pending(user_id)
#     tlower  = text.lower().strip()
#     mode    = await get_mode(mem, user_id)   # === ANALYSIS

#     # if explicit switch to other intents arrives, clear analysis mode so routing stays clean
#     if explicit_log_command(tlower) or is_plan_request(tlower):
#         await set_mode(mem, user_id, None)   # === ANALYSIS

#     if pend.get("state") == "awaiting_analysis_confirm":
#         if is_yes(text):
#             await mem.set_pending(user_id, None)
#             return StreamingResponse(
#                 run_analysis_generator(db, mem, oai, user_id),
#                 media_type="text/event-stream",
#                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#             )

#     # start analysis flow on intent
#     if is_analysis_intent(tlower) and not pend:
#         await mem.set_pending(user_id, {"state":"awaiting_analysis_confirm"})
#         async def _ask_confirm():
#             yield sse_json({"type":"analysis","status":"confirm",
#                             "prompt":"Sure—let me analyse your diet and workout data. Shall we start?"})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_ask_confirm(), media_type="text/event-stream",
#                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ===== follow-ups while in analysis mode =====
#     if (await get_mode(mem, user_id)) == "analysis" and not pend and not explicit_log_command(tlower) and not is_plan_request(tlower):
#         if is_followup_question(text):
#             dataset, summary = await get_analysis_artifacts(mem, user_id)
#             if dataset:
#                 await mem.add(user_id, "user", text.strip())
#                 msgs = [
#                     {"role":"system","content": GENERAL_SYSTEM},
#                     {"role":"system","content": STYLE_CHAT_FORMAT},
#                     {"role":"system","content":
#                         "ANALYSIS_MODE=ON. Use this context without re-querying DB.\n"
#                         f"DATASET:\n{orjson.dumps(dataset).decode()}\n\n"
#                         f"PRIOR_SUMMARY:\n{summary or ''}\n"
#                         "Answer follow-up concisely with numbers where helpful."
#                     },
#                     {"role":"user","content": text.strip()},
#                 ]
#                 resp = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=0)
#                 content = (resp.choices[0].message.content or "").strip()
#                 content = re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=re.I)
#                 pretty = pretty_plan(content)

#                 async def _one_shot_followup():
#                     yield sse_escape(pretty)
#                     await mem.add(user_id, "assistant", pretty)
#                     yield "event: done\ndata: [DONE]\n\n"
#                 return StreamingResponse(_one_shot_followup(), media_type="text/event-stream",
#                                         headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})



#     if pend.get("state") == "awaiting_nav_confirm":
#         if is_yes(text):
#             await mem.set_pending(user_id, None)
#             async def _nav_yes():
#                 yield sse_json({"type":"nav","is_navigation": True,
#                                 "prompt":"Thanks for your confirmation. Redirecting to today's diet logs"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_nav_yes(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _nav_no():
#                 yield sse_json({"type":"nav","is_navigation": False,
#                                 "prompt":"Thanks for your response. You can continue chatting here."})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_nav_no(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         async def _nav_clarify():
#             yield sse_json({"type":"nav","status":"confirm","prompt":""})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_nav_clarify(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ awaiting log-intent confirm ------------
#     if pend.get("state") == "awaiting_log_intent_confirm":
#         proposal = pend.get("proposal")
#         if is_yes(text):
#             idx, fname = first_missing_quantity(proposal.get("items", []))
#             if idx >= 0:
#                 ask = proposal["items"][idx].get("ask") or f"How many {proposal['items'][idx].get('unit_hint','pieces')} of {fname or 'items'} did you have?"
#                 await mem.set_pending(user_id, {"state":"awaiting_quantity", "holder": {"proposal": proposal, "next_q_idx": idx}})
#                 async def _needs_qty():
#                     yield sse_json({"type":"food_log","status":"needs_quantity","prompt":ask,"proposal":proposal})
#                     yield "event: ping\ndata: {}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                 return StreamingResponse(_needs_qty(), media_type="text/event-stream",
#                                          headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#             else:
#                 proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
#                 proposal["proposal_id"] = proposal_id
#                 await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
#                 async def _confirm():
#                     yield sse_json({"type":"food_log","status":"confirm","prompt":"Shall I log this entry?","proposal":proposal})
#                     yield "event: ping\ndata: {}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                 return StreamingResponse(_confirm(), media_type="text/event-stream",
#                                          headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _no_log():
#                 yield sse_json({"type":"food_log","status":"skip_log","prompt":"Okay, I won’t log it. How can I help you?"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_no_log(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         async def _clarify_log_intent():
#             yield sse_json({"type":"food_log","status":"ask_log_intent",
#                             "prompt":"Do you want me to log this? (yes/no)",
#                             "proposal": proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_clarify_log_intent(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ awaiting quantity ------------
#     if pend.get("state") == "awaiting_quantity":
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _cancel_q():
#                 yield sse_json({"type":"food_log","status":"cancelled"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_cancel_q(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         holder   = pend["holder"]
#         proposal = holder["proposal"]
#         idx, _   = first_missing_quantity(proposal["items"])
#         holder["next_q_idx"] = idx
#         pend["holder"] = holder
#         await mem.set_pending(user_id, pend)

#         if idx == -1:
#             await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})

#         from chatbot_services.llm_helpers import QuantityValidator, extract_numbers
#         nums = extract_numbers(text)

#         async def stream_needs_qty(fname: str, ask: str):
#             yield sse_json({"type":"food_log","status":"needs_quantity",
#                             "prompt": ask or f"How many {fname or 'items'} did you have?",
#                             "proposal": proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"

#         if idx >= 0 and nums:
#             qty = float(nums[0])
#             try:
#                 QuantityValidator(quantity=qty)
#             except Exception:
#                 item = proposal["items"][idx]
#                 return StreamingResponse(
#                     stream_needs_qty(item.get("food") or "items", item.get("ask")),
#                     media_type="text/event-stream",
#                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#                 )

#             item = proposal["items"][idx]
#             item["quantity"] = qty
#             ensure_per_unit_macros(item)
#             _scale_macros([item])

#             new_idx, _ = first_missing_quantity(proposal["items"])
#             if new_idx >= 0:
#                 holder["next_q_idx"] = new_idx
#                 pend["holder"] = holder
#                 await mem.set_pending(user_id, pend)
#                 nxt = proposal["items"][new_idx]
#                 return StreamingResponse(
#                     stream_needs_qty(nxt.get("food") or "items", nxt.get("ask")),
#                     media_type="text/event-stream",
#                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#                 )

#             proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
#             proposal["proposal_id"] = proposal_id
#             await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
#             async def _confirm():
#                 yield sse_json({"type":"food_log","status":"confirm",
#                                 "prompt":"Shall I log this entry?","proposal":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_confirm(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         else:
#             item = proposal["items"][idx] if idx >= 0 else {}
#             return StreamingResponse(
#                 stream_needs_qty(item.get("food") or "items", item.get("ask")),
#                 media_type="text/event-stream",
#                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#             )

#     # ------------ awaiting confirm ------------
#     if pend.get("state") == "awaiting_log_confirm":
#         proposal = pend.get("proposal")
#         if is_yes(text):
#             async def _logged_then_nav():
#                 yield sse_json({"type":"food_log","status":"logged","is_log":True,"entry":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})
#             return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _cancel():
#                 yield sse_json({"type":"food_log","status":"cancelled","prompt":"Okay, I didn’t log anything. If you have any fitness related queries we can continue chatting"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_cancel(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         async def _clarify():
#             yield sse_json({"type":"food_log","status":"confirm",
#                             "prompt":"I didn't catch that – shall I log this entry?",
#                             "proposal":proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_clarify(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ============= ANALYSIS: follow-ups while in mode ============
#        # ------------ fresh turn: foods mentioned but NO explicit 'log/add' → ask one follow-up ------------
#     if not explicit_log_command(tlower) and food_hits(text) > 0:
#         items = gpt_extract_items(text, oai)
#         for it in items:
#             it["food"] = normalize_food(it.get("food"))
#             ensure_per_unit_macros(it)
#         known_items = []
#         for it in items:
#             if it.get("food"):
#                 try:
#                     if all(isinstance(it.get(k), (int, float)) for k in ("calories","protein","carbs","fat","fiber","sugar")):
#                         known_items.append(it)
#                 except Exception:
#                     pass
#         if known_items:
#             for it in known_items:
#                 if it.get("quantity") not in (None, 0, ""):
#                     _scale_macros([it])
#             proposal = {"meal_type": None, "log_time_iso": now_iso, "items": known_items}
#             await mem.set_pending(user_id, {"state":"awaiting_log_intent_confirm", "proposal": proposal})
#             async def _ask_log_intent():
#                 yield sse_json({
#                     "type": "food_log",
#                     "status": "ask_log_intent",
#                     "prompt": "I spotted food items. Do you want me to log this? (yes/no)",
#                     "proposal": proposal
#                 })
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_ask_log_intent(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ---------- normal chat ----------
#     is_meta = is_fittbot_meta_query(text) and not (is_plan_request(text) or is_fit_chat(text))
#     is_plan = is_plan_request(text) or is_fit_chat(text)

#     await mem.add(user_id, "user", text.strip())

#     if is_plan:
#         msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
#                                        context_only=False, k=TOP_K)
#         msgs.insert(1, {"role": "system", "content": STYLE_PLAN})
#         temperature = 0
#     else:
#         msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
#                                        context_only=is_meta, k=8 if is_meta else TOP_K)
#         msgs.insert(1, {"role": "system", "content": STYLE_CHAT_FORMAT})
#         temperature = 0

#     resp    = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=temperature)
#     content = (resp.choices[0].message.content or "").strip()

#     content = re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=re.I)

#     pretty  = pretty_plan(content)
#     async def _one_shot():
#         yield sse_escape(pretty)
#         await mem.add(user_id, "assistant", pretty)
#         yield "event: done\ndata: [DONE]\n\n"
#     return StreamingResponse(_one_shot(), media_type="text/event-stream",
#                              headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# @router.get("/chat/stream_test", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
# async def chat_stream(
#     user_id: int,
#     text: str = Query(...),
#     mem = Depends(get_mem),
#     oai  = Depends(get_oai),
# ):
#     if not user_id or not text.strip():
#         raise HTTPException(400, "user_id and text required")

#     now_iso = datetime.now(IST).isoformat()
#     pend    = await mem.get_pending(user_id)
#     tlower  = text.lower().strip()

#     # ------------ awaiting navigation confirm ------------
#     if pend.get("state") == "awaiting_nav_confirm":
#         if is_yes(text):
#             await mem.set_pending(user_id, None)
#             async def _nav_yes():
#                 yield sse_json({"type":"nav","is_navigation": True,
#                                 "prompt":"Thanks for your confirmation. Redirecting to today's diet logs"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_nav_yes(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _nav_no():
#                 yield sse_json({"type":"nav","is_navigation": False,
#                                 "prompt":"Thanks for your response. You can continue chatting here."})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_nav_no(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         async def _nav_clarify():
#             yield sse_json({"type":"nav","status":"confirm","prompt":""})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_nav_clarify(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ awaiting log-intent confirm (user mentioned foods but no 'log/add') ------------
#     if pend.get("state") == "awaiting_log_intent_confirm":
#         proposal = pend.get("proposal")  # {"meal_type": None, "log_time_iso": now_iso, "items": [...]}

#         if is_yes(text):
#             # Continue like explicit log flow
#             idx, fname = first_missing_quantity(proposal.get("items", []))
#             if idx >= 0:
#                 ask = proposal["items"][idx].get("ask") or f"How many {proposal['items'][idx].get('unit_hint','pieces')} of {fname or 'items'} did you have?"
#                 await mem.set_pending(user_id, {"state":"awaiting_quantity", "holder": {"proposal": proposal, "next_q_idx": idx}})
#                 async def _needs_qty():
#                     yield sse_json({"type":"food_log","status":"needs_quantity","prompt":ask,"proposal":proposal})
#                     yield "event: ping\ndata: {}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                 return StreamingResponse(_needs_qty(), media_type="text/event-stream",
#                                          headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#             else:
#                 proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
#                 proposal["proposal_id"] = proposal_id
#                 await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
#                 async def _confirm():
#                     yield sse_json({"type":"food_log","status":"confirm","prompt":"Shall I log this entry?","proposal":proposal})
#                     yield "event: ping\ndata: {}\n\n"
#                     yield "event: done\ndata: [DONE]\n\n"
#                 return StreamingResponse(_confirm(), media_type="text/event-stream",
#                                          headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _no_log():
#                 yield sse_json({"type":"food_log","status":"skip_log","prompt":"Okay, I won’t log it. How can I help you?"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_no_log(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         async def _clarify_log_intent():
#             yield sse_json({"type":"food_log","status":"ask_log_intent",
#                             "prompt":"Do you want me to log this? (yes/no)",
#                             "proposal": proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_clarify_log_intent(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ awaiting quantity ------------
#     if pend.get("state") == "awaiting_quantity":
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _cancel_q():
#                 yield sse_json({"type":"food_log","status":"cancelled"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_cancel_q(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         holder   = pend["holder"]
#         proposal = holder["proposal"]
#         idx, _   = first_missing_quantity(proposal["items"])
#         holder["next_q_idx"] = idx
#         pend["holder"] = holder
#         await mem.set_pending(user_id, pend)

#         if idx == -1:
#             await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})

#         from client_api.chatbot.chatbot_services.llm_helpers import QuantityValidator, extract_numbers
#         nums = extract_numbers(text)

#         async def stream_needs_qty(fname: str, ask: str):
#             yield sse_json({"type":"food_log","status":"needs_quantity",
#                             "prompt": ask or f"How many {fname or 'items'} did you have?",
#                             "proposal": proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"

#         if idx >= 0 and nums:
#             qty = float(nums[0])
#             try:
#                 QuantityValidator(quantity=qty)
#             except Exception:
#                 item = proposal["items"][idx]
#                 return StreamingResponse(
#                     stream_needs_qty(item.get("food") or "items", item.get("ask")),
#                     media_type="text/event-stream",
#                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#                 )

#             item = proposal["items"][idx]
#             item["quantity"] = qty
#             ensure_per_unit_macros(item)
#             _scale_macros([item])

#             new_idx, _ = first_missing_quantity(proposal["items"])
#             if new_idx >= 0:
#                 holder["next_q_idx"] = new_idx
#                 pend["holder"] = holder
#                 await mem.set_pending(user_id, pend)
#                 nxt = proposal["items"][new_idx]
#                 return StreamingResponse(
#                     stream_needs_qty(nxt.get("food") or "items", nxt.get("ask")),
#                     media_type="text/event-stream",
#                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#                 )

#             proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
#             proposal["proposal_id"] = proposal_id
#             await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
#             async def _confirm():
#                 yield sse_json({"type":"food_log","status":"confirm",
#                                 "prompt":"Shall I log this entry?","proposal":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_confirm(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         else:
#             item = proposal["items"][idx] if idx >= 0 else {}
#             return StreamingResponse(
#                 stream_needs_qty(item.get("food") or "items", item.get("ask")),
#                 media_type="text/event-stream",
#                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
#             )

#     # ------------ awaiting confirm ------------
#     if pend.get("state") == "awaiting_log_confirm":
#         proposal = pend.get("proposal")
#         if is_yes(text):
#             async def _logged_then_nav():
#                 yield sse_json({"type":"food_log","status":"logged","is_log":True,"entry":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})
#             return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         if is_no(text):
#             await mem.set_pending(user_id, None)
#             async def _cancel():
#                 yield sse_json({"type":"food_log","status":"cancelled","prompt":"Okay, I didn’t log anything. If you have any fitness related queries we can continue chatting"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_cancel(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         async def _clarify():
#             yield sse_json({"type":"food_log","status":"confirm",
#                             "prompt":"I didn't catch that – shall I log this entry?",
#                             "proposal":proposal})
#             yield "event: ping\ndata: {}\n\n"
#             yield "event: done\ndata: [DONE]\n\n"
#         return StreamingResponse(_clarify(), media_type="text/event-stream",
#                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ fresh turn: explicit 'log/add' command → go straight to logging flow ------------
#     if explicit_log_command(tlower):
#         import re as _re

#         # strip leading verbs like "log", "add", etc., repeatedly
#         cleaned_text = text
#         while True:
#             new_text = _re.sub(r'^\s*(log|add|record|track)\b[:,\-\s]*', '', cleaned_text, flags=_re.I)
#             if new_text == cleaned_text: break
#             cleaned_text = new_text

#         # Tokenize → candidate phrases (1–3 grams), remove obvious non-food verbs/stopwords
#         raw = _re.sub(r"[^A-Za-z\u0900-\u0fff\s]", " ", cleaned_text, flags=_re.UNICODE)  # keep Indic scripts too
#         words = [w.lower() for w in raw.split() if w.strip()]

#         STOP = {
#             "log","add","record","track","please","plz","me","i","my","to","for","of","and",
#             "a","an","the","yes","no","ok","okay","hai","haa","haan","haanji","bro","dear"
#         }

#         words = [w for w in words if w not in STOP]

#         # build 1-gram, 2-gram, 3-gram contiguous phrases
#         cands = set()
#         n = len(words)
#         for i in range(n):
#             cands.add(words[i])
#             if i+1 < n: cands.add(words[i]+" "+words[i+1])
#             if i+2 < n: cands.add(words[i]+" "+words[i+1]+" "+words[i+2])

#         # if user typed only verbs like "log log" -> no candidates → ask for food name and exit
#         if not cands:
#             await mem.set_pending(user_id, None)
#             async def _need_food_name():
#                 yield sse_json({"type":"food_log","status":"needs_food",
#                                 "prompt":"What food would you like to log?"})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_need_food_name(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         # call LLM with constrained candidates
#         items = gpt_extract_items(cleaned_text, oai, candidates=sorted(cands))

#         for it in items:
#             it["food"] = normalize_food(it.get("food"))
#             ensure_per_unit_macros(it)

#         # keep only items that actually have macros (from LLM or DB)
#         known_items = [it for it in items if it.get("food") and all(isinstance(it.get(k),(int,float)) for k in ("calories","protein","carbs","fat","fiber","sugar"))]

#         if not known_items:
#             await mem.set_pending(user_id, None)
#             async def _no_food():
#                 yield sse_json({"type":"food_log","status":"not_food",
#                                 "prompt":"Oops, sorry—I couldn’t recognize any food items. I didn’t log anything."})
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_no_food(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#         # scale if quantity already present
#         for it in known_items:
#             if it.get("quantity") not in (None, 0, ""):
#                 _scale_macros([it])

#         proposal = {"meal_type": None, "log_time_iso": now_iso, "items": known_items}
#         idx, fname = first_missing_quantity(known_items)
#         if idx >= 0:
#             ask = known_items[idx].get("ask") or f"How many {known_items[idx].get('unit_hint','pieces')} of {fname or 'items'} did you have?"
#             await mem.set_pending(user_id, {"state":"awaiting_quantity", "holder": {"proposal": proposal, "next_q_idx": idx}})
#             async def _needs_qty():
#                 yield sse_json({"type":"food_log","status":"needs_quantity","prompt":ask,"proposal":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_needs_qty(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
#         else:
#             proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
#             proposal["proposal_id"] = proposal_id
#             await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
#             async def _confirm():
#                 yield sse_json({"type":"food_log","status":"confirm","prompt":"Shall I log this entry?","proposal":proposal})
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_confirm(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ------------ fresh turn: foods mentioned but NO explicit 'log/add' → ask one follow-up ------------
#     if not explicit_log_command(tlower) and food_hits(text) > 0:
#         items = gpt_extract_items(text, oai)
#         for it in items:
#             it["food"] = normalize_food(it.get("food"))
#             ensure_per_unit_macros(it)

#         known_items = []
#         for it in items:
#             if it.get("food"):
#                 try:
#                     if all(isinstance(it.get(k), (int, float)) for k in ("calories","protein","carbs","fat","fiber","sugar")):
#                         known_items.append(it)
#                 except Exception:
#                     pass

#         if known_items:
#             for it in known_items:
#                 if it.get("quantity") not in (None, 0, ""):
#                     _scale_macros([it])

#             proposal = {"meal_type": None, "log_time_iso": now_iso, "items": known_items}
#             await mem.set_pending(user_id, {"state":"awaiting_log_intent_confirm", "proposal": proposal})

#             async def _ask_log_intent():
#                 yield sse_json({
#                     "type": "food_log",
#                     "status": "ask_log_intent",
#                     "prompt": "I spotted food items. Do you want me to log this? (yes/no)",
#                     "proposal": proposal
#                 })
#                 yield "event: ping\ndata: {}\n\n"
#                 yield "event: done\ndata: [DONE]\n\n"
#             return StreamingResponse(_ask_log_intent(), media_type="text/event-stream",
#                                      headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

#     # ---------- normal chat ----------
#     # Decide if this is a Fittbot meta (plans/pricing/FAQ) question or a diet/workout plan request
#     is_meta = is_fittbot_meta_query(text) and not (is_plan_request(text) or is_fit_chat(text))
#     is_plan = is_plan_request(text) or is_fit_chat(text)

#     await mem.add(user_id, "user", text.strip())

#     if is_plan:
#         msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
#                                        context_only=False, k=TOP_K)
#         msgs.insert(1, {"role": "system", "content": STYLE_PLAN})
#         temperature = 0
#     else:
#         msgs, _ = await build_messages(user_id, text.strip(), use_context=True, oai=oai, mem=mem,
#                                        context_only=is_meta, k=8 if is_meta else TOP_K)
#         msgs.insert(1, {"role": "system", "content": STYLE_CHAT_FORMAT})
#         temperature = 0

#     resp    = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=temperature)
#     content = (resp.choices[0].message.content or "").strip()

#     # Normalize brand spelling → always "Fittbot"
#     import re as _re
#     content = _re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=_re.I)

#     pretty  = pretty_plan(content)
#     async def _one_shot():
#         yield sse_escape(pretty)
#         await mem.add(user_id, "assistant", pretty)
#         yield "event: done\ndata: [DONE]\n\n"
#     return StreamingResponse(_one_shot(), media_type="text/event-stream",
#                              headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})



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

    # If user explicitly goes to log or plan/chat, exit analysis mode (so routes stay clean)
    if explicit_log_command(tlower) or is_plan_request(tlower):
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

    # ---- FOOD LOG: awaiting log-intent confirm ----
    if pend.get("state") == "awaiting_log_intent_confirm":
        proposal = pend.get("proposal") or {}
        if is_yes(text):
            idx, fname = first_missing_quantity(proposal.get("items", []))
            if idx >= 0:
                ask = proposal["items"][idx].get("ask") or f"How many {proposal['items'][idx].get('unit_hint','pieces')} of {fname or 'items'} did you have?"
                await mem.set_pending(user_id, {"state":"awaiting_quantity",
                                                "holder": {"proposal": proposal, "next_q_idx": idx}})
                async def _needs_qty():
                    yield sse_json({"type":"food_log","status":"needs_quantity","prompt":ask,"proposal":proposal})
                    yield "event: ping\ndata: {}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_needs_qty(), media_type="text/event-stream",
                                         headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            else:
                import hashlib, orjson
                proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
                proposal["proposal_id"] = proposal_id
                await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
                async def _confirm():
                    yield sse_json({"type":"food_log","status":"confirm","prompt":"Shall I log this entry?","proposal":proposal})
                    yield "event: ping\ndata: {}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_confirm(), media_type="text/event-stream",
                                         headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        if is_no(text):
            await mem.set_pending(user_id, None)
            async def _no_log():
                yield sse_json({"type":"food_log","status":"skip_log","prompt":"Okay, I won’t log it. How can I help you?"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_no_log(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        async def _clar_log():
            yield sse_json({"type":"food_log","status":"ask_log_intent",
                            "prompt":"Do you want me to log this? (yes/no)","proposal": proposal})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_clar_log(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # ---- FOOD LOG: awaiting quantity ----
    if pend.get("state") == "awaiting_quantity":
        if is_no(text):
            await mem.set_pending(user_id, None)
            async def _cancel_q():
                yield sse_json({"type":"food_log","status":"cancelled"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_cancel_q(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        holder   = pend["holder"]
        proposal = holder["proposal"]
        idx, _   = first_missing_quantity(proposal["items"])
        holder["next_q_idx"] = idx
        pend["holder"] = holder
        await mem.set_pending(user_id, pend)

        if idx == -1:
            await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})

        from chatbot_services.llm_helpers import QuantityValidator, extract_numbers  # <- keep this path
        nums = extract_numbers(text)

        async def stream_needs_qty(fname: str, ask: str):
            yield sse_json({"type":"food_log","status":"needs_quantity",
                            "prompt": ask or f"How many {fname or 'items'} did you have?",
                            "proposal": proposal})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        if idx >= 0 and nums:
            qty = float(nums[0])
            try:
                QuantityValidator(quantity=qty)
            except Exception:
                item = proposal["items"][idx]
                return StreamingResponse(
                    stream_needs_qty(item.get("food") or "items", item.get("ask")),
                    media_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
                )

            item = proposal["items"][idx]
            item["quantity"] = qty
            ensure_per_unit_macros(item)
            _scale_macros([item])

            new_idx, _ = first_missing_quantity(proposal["items"])
            if new_idx >= 0:
                holder["next_q_idx"] = new_idx
                pend["holder"] = holder
                await mem.set_pending(user_id, pend)
                nxt = proposal["items"][new_idx]
                return StreamingResponse(
                    stream_needs_qty(nxt.get("food") or "items", nxt.get("ask")),
                    media_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
                )

            import hashlib, orjson
            proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
            proposal["proposal_id"] = proposal_id
            await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
            async def _confirm():
                yield sse_json({"type":"food_log","status":"confirm",
                                "prompt":"Shall I log this entry?","proposal":proposal})
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_confirm(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            item = proposal["items"][idx] if idx >= 0 else {}
            return StreamingResponse(
                stream_needs_qty(item.get("food") or "items", item.get("ask")),
                media_type="text/event-stream",
                headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
            )

    # ---- FOOD LOG: awaiting final confirm ----
    if pend.get("state") == "awaiting_log_confirm":
        proposal ={}or pend.get("proposal")
        if is_yes(text):
            async def _logged_then_nav():
                yield sse_json({"type":"food_log","status":"logged","is_log":True,"entry":proposal})
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            await mem.set_pending(user_id, {"state":"awaiting_nav_confirm"})
            return StreamingResponse(_logged_then_nav(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        if is_no(text):
            await mem.set_pending(user_id, None)
            async def _cancel():
                yield sse_json({"type":"food_log","status":"cancelled","prompt":"Okay, I didn’t log anything. If you have any fitness related queries we can continue chatting"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_cancel(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        async def _clar():
            yield sse_json({"type":"food_log","status":"confirm",
                            "prompt":"I didn't catch that – shall I log this entry?",
                            "proposal":proposal})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_clar(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # =========================
    # 2) FRESH TURN: TRIGGERED ACTIONS
    # =========================

    # --- A) explicit "log/add/record/track ..." → go straight to logging flow
    if explicit_log_command(tlower):
        import re as _re, hashlib, orjson
        cleaned_text = text
        while True:
            new_text = _re.sub(r'^\s*(log|add|record|track)\b[:,\-\s]*', '', cleaned_text, flags=_re.I)
            if new_text == cleaned_text: break
            cleaned_text = new_text
        raw = _re.sub(r"[^A-Za-z\u0900-\u0fff\s]", " ", cleaned_text, flags=_re.UNICODE)
        words = [w.lower() for w in raw.split() if w.strip()]
        STOP = {"log","add","record","track","please","plz","me","i","my","to","for","of","and","a","an","the","yes","no","ok","okay","hai","haa","haan","haanji","bro","dear"}
        words = [w for w in words if w not in STOP]
        cands = set()
        n = len(words)
        for i in range(n):
            cands.add(words[i])
            if i+1 < n: cands.add(words[i]+" "+words[i+1])
            if i+2 < n: cands.add(words[i]+" "+words[i+1]+" "+words[i+2])
        if not cands:
            await mem.set_pending(user_id, None)
            async def _need_food_name():
                yield sse_json({"type":"food_log","status":"needs_food","prompt":"What food would you like to log?"})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_need_food_name(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        items = gpt_extract_items(cleaned_text, oai, candidates=sorted(cands))
        for it in items:
            it["food"] = normalize_food(it.get("food"))
            ensure_per_unit_macros(it)
        known_items = [it for it in items if it.get("food") and all(isinstance(it.get(k),(int,float)) for k in ("calories","protein","carbs","fat","fiber","sugar"))]
        if not known_items:
            await mem.set_pending(user_id, None)
            async def _no_food():
                yield sse_json({"type":"food_log","status":"not_food","prompt":"Oops, sorry—I couldn’t recognize any food items. I didn’t log anything."})
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_no_food(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        for it in known_items:
            if it.get("quantity") not in (None, 0, ""):
                _scale_macros([it])

        proposal = {"meal_type": None, "log_time_iso": now_iso, "items": known_items}
        idx, fname = first_missing_quantity(known_items)
        if idx >= 0:
            ask = known_items[idx].get("ask") or f"How many {known_items[idx].get('unit_hint','pieces')} of {fname or 'items'} did you have?"
            await mem.set_pending(user_id, {"state":"awaiting_quantity", "holder": {"proposal": proposal, "next_q_idx": idx}})
            async def _needs_qty():
                yield sse_json({"type":"food_log","status":"needs_quantity","prompt":ask,"proposal":proposal})
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_needs_qty(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            proposal_id = hashlib.sha1(orjson.dumps(proposal)).hexdigest()
            proposal["proposal_id"] = proposal_id
            await mem.set_pending(user_id, {"state":"awaiting_log_confirm", "proposal": proposal})
            async def _confirm():
                yield sse_json({"type":"food_log","status":"confirm","prompt":"Shall I log this entry?","proposal":proposal})
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_confirm(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # --- B) foods mentioned (no explicit "log") → ask log intent once
    if food_hits(text) > 0:
        items = gpt_extract_items(text, oai)
        for it in items:
            it["food"] = normalize_food(it.get("food"))
            ensure_per_unit_macros(it)
        known_items = []
        for it in items:
            if it.get("food"):
                try:
                    if all(isinstance(it.get(k), (int, float)) for k in ("calories","protein","carbs","fat","fiber","sugar")):
                        known_items.append(it)
                except Exception:
                    pass
        if known_items:
            for it in known_items:
                if it.get("quantity") not in (None, 0, ""):
                    _scale_macros([it])
            proposal = {"meal_type": None, "log_time_iso": now_iso, "items": known_items}
            await mem.set_pending(user_id, {"state":"awaiting_log_intent_confirm", "proposal": proposal})
            async def _ask_log_intent():
                yield sse_json({"type":"food_log","status":"ask_log_intent",
                                "prompt":"I spotted food items. Do you want me to log this? (yes/no)",
                                "proposal": proposal})
                yield "event: ping\ndata: {}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_ask_log_intent(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # --- C) analysis start or follow-ups (only when nothing above already handled this turn)
    # follow-ups while already in analysis mode
    if mode == "analysis" and not explicit_log_command(tlower) and not is_plan_request(tlower):
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
    is_meta = is_fittbot_meta_query(text) and not (is_plan_request(text) or is_fit_chat(text))
    is_plan = is_plan_request(text) or is_fit_chat(text)

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

    # 3) Quick multilingual trigger check
    LOG_TRIGGERS = {
        # English
        "log","add","record","track",
        # Hindi / Marathi
        "लॉग","जोड़","जोड","रिकॉर्ड","दर्ज","ऐड","नोंद","नोंदवा",
        # Tamil
        "லாக்","சேர்க்க","பதிவு","ரெக்கார்ட்",
        # Telugu
        "లాగ్","చేర్చు","నమోదు","రికార్డ్",
        # Kannada
        "ಲಾಗ್","ಸೇರಿಸಿ","ದಾಖಲೆ",
        # Malayalam
        "ലോഗ്","ചേർക്കുക","രേഖപ്പെടുത്തുക",
        # Gujarati
        "લોગ","ઉમેરો","નોંધ",
        # Bengali
        "লগ","যোগ","রেকর্ড","নথিভুক্ত",
        # Punjabi (Gurmukhi)
        "ਲੋਗ","ਜੋੜੋ","ਦਰਜ","ਰਿਕਾਰਡ",
        # Urdu
        "لاگ","شامل","ریکارڈ",
        # Odia
        "ଲଗ୍","ଯୋଡନ୍ତୁ","ରେକର୍ଡ",
    }
    def _has_multilingual_trigger(s: str) -> bool:
        t = s.lower()
        return any(k in t for k in LOG_TRIGGERS)

    tinfo = _translate_to_english(transcript)
    transcript_en = tinfo["english"]
    lang_code     = tinfo["lang"]
    has_trigger   = _has_multilingual_trigger(transcript) or _has_multilingual_trigger(transcript_en)

    print("vanakam")
    
    print(f"[voice] lang={lang_code} raw={transcript!r} | en={transcript_en!r} | trigger={has_trigger}")

    # Return both raw & translated for the client (optional), plus trigger hint
    return {
        "transcript": transcript_en,
        "lang": lang_code,
        "english": transcript_en,
        "has_log_or_add": bool(has_trigger),
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