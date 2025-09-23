# routes/ai_chatbot.py - Complete Fixed Version with Improved AI Intent Detection

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
import pytz, os, hashlib, orjson, json, re
from datetime import datetime
import io
from sqlalchemy.orm import Session
from app.models.database import get_db

from app.models.deps import get_http, get_oai, get_mem
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.kb_store import KB
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.llm_helpers import (
    PlainTextStreamFilter, oai_chat_stream, GENERAL_SYSTEM, TOP_K,
    build_messages, heuristic_confidence, OPENAI_MODEL,
    sse_json, sse_escape, gpt_small_route, is_yes, is_no, is_fitness_related,
    has_action_verb, is_fittbot_meta_query, is_plan_request, STYLE_PLAN, STYLE_CHAT_FORMAT, pretty_plan
)
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.report_analysis import (
    is_analysis_intent, is_followup_question, set_mode, get_mode,
    set_analysis_artifacts, get_analysis_artifacts, build_analysis_dataset_dict,
    build_summary_hints, run_analysis_generator, STYLE_INSIGHT_REPORT,
)
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.asr import transcribe_audio

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

APP_ENV = os.getenv("APP_ENV", "prod")
TZNAME = os.getenv("TZ", "Asia/Kolkata")
IST = pytz.timezone(TZNAME)

class KBUpsertIn(BaseModel):
    source: str
    text: str

class KBSearchIn(BaseModel):
    query: str
    k: int = 4

# ALL HELPER FUNCTIONS DEFINED FIRST

def has_action_indicators(text: str) -> bool:
    """Check for explicit action words that indicate intent to create/log something"""
    action_words = [
        # Creation words
        'create', 'make', 'build', 'design', 'plan', 'generate', 'develop',
        'give', 'tell', 'show', 'provide', 'suggest', 'recommend',
        # Logging words  
        'log', 'track', 'record', 'save', 'add', 'input', 'enter',
        # Past tense logging
        'ate', 'had', 'consumed', 'did', 'completed', 'finished', 'done'
    ]
    
    text_lower = text.lower()
    return any(word in text_lower for word in action_words)

def is_simple_food_mention(text: str) -> bool:
    """Detect simple food mentions - now just marks candidates for AI check"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    # Only for very short inputs (1-2 words)
    if len(words) <= 2:
        # Exclude obvious non-food contexts first
        non_food_contexts = [
            'recipe', 'cook', 'cooking', 'how to', 'make', 'prepare', 'bake',
            'nutrition', 'calories', 'healthy', 'benefits', 'vitamin', 'value',
            'information', 'facts', 'content', 'good for', 'bad for'
        ]
        
        if any(context in text_lower for context in non_food_contexts):
            return False
        
        # Mark as candidate for AI check - we'll do the AI check in the main function
        return True
    
    return False

def is_simple_food_mention_with_ai(text: str, oai) -> bool:
    """Use AI to universally detect if a text is a food name"""
    try:
        prompt = f"""
        Is "{text}" a food item, dish, or beverage that someone might want to log in a food diary?
        
        Consider:
        - Foods from any cuisine (Indian, Western, Asian, etc.)
        - Fruits, vegetables, grains, proteins, dairy
        - Prepared dishes, snacks, beverages
        - Regional/local food names
        - Common misspellings and typos (like "bananana" for "banana", "pickel" for "pickle")
        
        Examples:
        - "apple" → YES
        - "idli" → YES  
        - "bananana" → YES (typo of banana)
        - "pickel" → YES (typo of pickle)
        - "computer" → NO
        - "exercise" → NO
        
        Return only "YES" or "NO".
        """
        
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a food classifier. Be generous with typos and variations. Return only YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(f"DEBUG: AI Food Check: '{text}' → {result}")
        return result == "YES"
        
    except Exception as e:
        print(f"DEBUG: Food AI check error: {e}")
        # Fallback: if AI fails, be more generous with potential food words
        # Check if it's similar to common food words
        potential_foods = ['apple', 'banana', 'orange', 'mango', 'rice', 'chicken', 'fish', 'bread', 'milk', 'egg']
        text_lower = text.lower()
        
        # Check for similar words (simple edit distance)
        for food in potential_foods:
            if abs(len(text_lower) - len(food)) <= 2:  # Similar length
                # Count matching characters
                matches = sum(1 for a, b in zip(text_lower, food) if a == b)
                if matches >= len(food) * 0.7:  # 70% character match
                    print(f"DEBUG: Fallback matched '{text}' to '{food}'")
                    return True
        
        # If no matches and it's a reasonable word, assume it might be food
        return len(text) > 2 and text.isalpha() and text.lower() not in ['the', 'and', 'but', 'for', 'you', 'are', 'can', 'how', 'what', 'why', 'when', 'where', 'this', 'that', 'with', 'from', 'they', 'have', 'been', 'there', 'will', 'some', 'time', 'very', 'more', 'come', 'could', 'like', 'into', 'know', 'exercise', 'workout', 'fitness']

def correct_food_spelling_with_ai(text: str, oai) -> str:
    """Use AI to correct spelling of food names"""
    try:
        prompt = f"""
        The user typed "{text}" which appears to be a misspelled food name.
        
        What is the correct spelling of this food item?
        
        Examples:
        - "bananana" → "banana"
        - "pickel" → "pickle"
        - "chiken" → "chicken"
        - "aple" → "apple"
        
        Return only the corrected word, nothing else.
        """
        
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a spelling corrector for food names. Return only the corrected word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        corrected = response.choices[0].message.content.strip().lower()
        print(f"DEBUG: Spell correction: '{text}' → '{corrected}'")
        return corrected
        
    except Exception as e:
        print(f"DEBUG: Spell correction error: {e}")
        return text  # Return original if correction fails

def detect_user_intent_with_ai(text: str, oai) -> dict:
    """Use AI to detect user intent with improved accuracy and specificity"""
    
    prompt = f"""
    Analyze this user message and determine their SPECIFIC intent: "{text}"
    
    ONLY detect these intents if the user is clearly requesting ACTION:
    
    1. WORKOUT_PLAN: User wants to CREATE/MAKE/GET a workout plan, exercise template, training routine
       - Examples: "create workout plan", "make exercise routine", "plan my workouts", "tell workout plan", "give workout plan", "show me workout plan"
       - Handle typos: "worutplan", "workoutplan", "excersize plan"
       - NOT: "workout benefits", "exercise tips", "how to do pushups"
    
    2. DIET_PLAN: User wants to CREATE/MAKE/GET a diet plan, meal plan, nutrition template
       - Examples: "create diet plan", "make meal plan", "plan my meals", "tell me diet plan", "give diet plan"
       - Handle typos: "deit plan", "meal templete", "mealplan"
       - NOT: "nutrition facts", "food benefits", "calories in apple", "what to eat"
    
    3. FOOD_LOGGING: User wants to LOG/TRACK/RECORD food they ate/consumed
       - Examples: "log apple", "I ate rice", "track my breakfast", "add food to diary"
       - NOT: "apple nutrition", "benefits of rice", "how many calories"
    
    4. WORKOUT_LOGGING: User wants to LOG/SAVE/RECORD exercises they did
       - Examples: "log my workout", "I did pushups", "save exercise", "record training"
       - NOT: "how to do exercise", "workout tips", "exercise benefits"
    
    5. NONE: General questions, information requests, nutrition facts, exercise tips, etc.
    
    IMPORTANT RULES:
    - Information/question requests → NONE
    - Nutrition facts/benefits → NONE  
    - Exercise instructions/tips → NONE
    - Only detect intent if user wants to CREATE, GET, or LOG something
    - Be very liberal with detecting plan creation intents (workout/diet plans)
    - Handle spelling mistakes and typos generously
    - Words like "tell", "give", "show", "provide" + "plan" = plan creation intent
    
    Return JSON:
    {{
        "intent": "WORKOUT_PLAN|DIET_PLAN|FOOD_LOGGING|WORKOUT_LOGGING|NONE",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation of why this intent was chosen",
        "action_detected": true/false
    }}
    """
    
    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise intent classifier. Only detect action intents (create/log/plan/track). Information requests should be NONE. Be conservative - when unsure, choose NONE."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        intent_data = json.loads(result)
        
        print(f"DEBUG AI Intent Detection: '{text}' → {intent_data}")
        return intent_data
        
    except Exception as e:
        print(f"DEBUG AI Intent Detection Error: {e}")
        return {"intent": "NONE", "confidence": 0.0, "reasoning": "error", "action_detected": False}

@router.get("/healthz")
async def healthz():
    return {"ok": True, "env": APP_ENV, "tz": TZNAME, "kb_chunks": len(KB.texts)}

class RichTextStreamFilter:
    def __init__(self): self.buf = ""
    def feed(self, ch: str) -> str:
        if not ch: return ""
        ch = ch.replace("\r\n", "\n").replace("\r", "\n")
        self.buf += ch
        out, self.buf = self.buf, ""
        return out
    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out

def pretty_plan(markdown: str) -> str:
    if not markdown:
        return ""

    txt = markdown.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r'^\s*#{1,6}\s*', '', txt, flags=re.M)
    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
    txt = re.sub(r'\*(.*?)\*', r'\1', txt)
    txt = re.sub(r'^\s*(\d+)\.\s*', r'\1) ', txt, flags=re.M)
    txt = re.sub(r'^\s*[-•]\s*', '• ', txt, flags=re.M)
    txt = re.sub(r':(?!\s)', ': ', txt)
    txt = re.sub(r',(?!\s)', ', ', txt)
    txt = re.sub(r'([A-Za-z])([-–—])([A-Za-z])', r'\1 \2 \3', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = "\n".join(line.rstrip() for line in txt.split("\n"))
    return txt.strip()

@router.get("/chat/stream_test", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
async def chat_stream(
    user_id: int,
    text: str = Query(...),
    mem = Depends(get_mem),
    oai = Depends(get_oai),
    db: Session = Depends(get_db),   
):
    if not user_id or not text.strip():
        raise HTTPException(400, "user_id and text required")

    text = text.strip()
    tlower = text.lower().strip()
    
    pend = (await mem.get_pending(user_id)) or {}
    mode = await get_mode(mem, user_id)

    print(f"DEBUG: User {user_id} input: '{text}'")
    print(f"DEBUG: Current pending state: {pend.get('state')}")

    if is_plan_request(tlower):
        await set_mode(mem, user_id, None)

    # Handle analysis confirmation
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

    # Handle navigation confirmation  
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

    # Handle workout plan confirmation
    if pend.get("state") == "awaiting_workout_redirect_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            async def _redirect_to_workout_plan():
                redirect_message = "Excellent! To create your personalized workout plan, go to: Home → Workout → Here you can find the 'Kyra default template' for workouts, or choose 'Make your own template' for more customizing options!"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_message)
                yield sse_json({
                    "type": "workout_plan_redirect",
                    "redirect": True,
                    "message": redirect_message,
                    "navigation_path": "Home → Workout → Templates"
                })
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_redirect_to_workout_plan(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        elif is_no(text):
            await mem.set_pending(user_id, None)
            # Continue with normal chat - fall through
        elif any(word in text.lower() for word in ['yes', 'no', 'yeah', 'nope', 'sure', 'nah']):
            async def _workout_redirect_clarify():
                clarify_msg = "Would you like me to guide you to the workout template creation section? Please say yes or no."
                yield f"data: {json.dumps({'message': clarify_msg, 'type': 'confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_workout_redirect_clarify(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            print(f"DEBUG: Clearing workout redirect state due to topic change to: {text}")
            await mem.set_pending(user_id, None)

    # Handle diet plan confirmation
    if pend.get("state") == "awaiting_diet_redirect_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            async def _redirect_to_diet_plan():
                redirect_message = "Great! To create your personalized diet plan, go to: Home → Diet → Here you can create your own food template or use the 'Kyra default template' as a starting point!"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_message)
                yield sse_json({
                    "type": "diet_plan_redirect",
                    "redirect": True,
                    "message": redirect_message,
                    "navigation_path": "Home → Diet → Create Template"
                })
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_redirect_to_diet_plan(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        elif is_no(text):
            await mem.set_pending(user_id, None)
            # Continue with normal chat - fall through
        elif any(word in text.lower() for word in ['yes', 'no', 'yeah', 'nope', 'sure', 'nah']):
            async def _diet_redirect_clarify():
                clarify_msg = "Would you like me to guide you to the diet template creation section? Please say yes or no."
                yield f"data: {json.dumps({'message': clarify_msg, 'type': 'confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_diet_redirect_clarify(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            print(f"DEBUG: Clearing diet redirect state due to topic change to: {text}")
            await mem.set_pending(user_id, None)

    # Handle food logging confirmation
    if pend.get("state") == "awaiting_food_redirect_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            async def _redirect_to_food_log():
                redirect_message = "Perfect! To log your food, go to: Home → Diet → Log food with Kyra AI. There you can log food by name or even scan items!"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_message)
                yield sse_json({
                    "type": "food_log_redirect",
                    "redirect": True,
                    "message": redirect_message,
                    "navigation_path": "Home → Diet → Log food with Kyra AI"
                })
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_redirect_to_food_log(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        elif is_no(text):
            await mem.set_pending(user_id, None)
            # Continue with normal chat - fall through
        elif any(word in text.lower() for word in ['yes', 'no', 'yeah', 'nope', 'sure', 'nah']):
            async def _food_redirect_clarify():
                clarify_msg = "Would you like me to guide you to the food logging section? Please say yes or no."
                yield f"data: {json.dumps({'message': clarify_msg, 'type': 'confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_food_redirect_clarify(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            print(f"DEBUG: Clearing food redirect state due to topic change to: {text}")
            await mem.set_pending(user_id, None)

    # Handle workout logging confirmation
    if pend.get("state") == "awaiting_workout_log_redirect_confirm":
        if is_yes(text):
            await mem.set_pending(user_id, None)
            async def _redirect_to_workout_log():
                redirect_message = "Great! To log your workout, go to: Home → Workout → Here you can find your templates and log exercises in your existing template, or create a new workout template!"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_message)
                yield sse_json({
                    "type": "workout_log_redirect",
                    "redirect": True,
                    "message": redirect_message,
                    "navigation_path": "Home → Workout → Templates & Logging"
                })
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_redirect_to_workout_log(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        elif is_no(text):
            await mem.set_pending(user_id, None)
            # Continue with normal chat - fall through
        elif any(word in text.lower() for word in ['yes', 'no', 'yeah', 'nope', 'sure', 'nah']):
            async def _workout_log_redirect_clarify():
                clarify_msg = "Would you like me to guide you to the workout logging section? Please say yes or no."
                yield f"data: {json.dumps({'message': clarify_msg, 'type': 'confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_workout_log_redirect_clarify(), media_type="text/event-stream",
                                     headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            print(f"DEBUG: Clearing workout log redirect state due to topic change to: {text}")
            await mem.set_pending(user_id, None)

    # IMPROVED AI-powered intent detection - only run if no pending states
    # First check if there are clear action indicators OR if it's a fitness-related question
    has_action = has_action_indicators(text)
    is_fitness_question = is_fitness_related(text)
    
    print(f"DEBUG Intent Check: has_action={has_action}, is_fitness={is_fitness_question}, text='{text}'")
    
    # PRIORITY CHECK: Simple food mentions should be checked FIRST before other intent detection
    # This prevents single food words from being processed as fitness questions
    if not has_action and is_simple_food_mention(text):
        # Use AI to verify it's actually a food
        if is_simple_food_mention_with_ai(text.strip(), oai):
            print(f"DEBUG: AI-verified food mention detected for: {text}")
            
            # Correct spelling if needed
            corrected_food = correct_food_spelling_with_ai(text.strip(), oai)
            display_food = corrected_food if corrected_food != text.strip().lower() else text.strip()
            
            await mem.set_pending(user_id, {
                "state": "awaiting_food_redirect_confirm", 
                "original_text": text.strip(),
                "corrected_food": display_food
            })
            
            async def _ask_simple_food_redirect():
                redirect_msg = f"Are you looking to log {display_food} in your food diary? I can guide you to the food logging section where you can easily track your meals!"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_msg)
                yield f"data: {json.dumps({'message': redirect_msg, 'type': 'food_redirect_confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_ask_simple_food_redirect(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        else:
            print(f"DEBUG: AI determined '{text}' is not a food item")
    
    # Run AI intent detection for action requests OR fitness questions (to catch edge cases)
    if has_action or is_fitness_question:
        intent_result = detect_user_intent_with_ai(text, oai)
        intent_type = intent_result.get("intent", "NONE")
        confidence = intent_result.get("confidence", 0.0)
        action_detected = intent_result.get("action_detected", False)
        reasoning = intent_result.get("reasoning", "")
        
        print(f"DEBUG AI Intent: {intent_type} (confidence: {confidence}, action: {action_detected})")
        print(f"DEBUG AI Reasoning: {reasoning}")
        print(f"DEBUG has_action: {has_action}, is_fitness: {is_fitness_question}")
        
        # Use different confidence thresholds for different intent types
        # More liberal for plan creation, stricter for logging
        if intent_type in ["WORKOUT_PLAN", "DIET_PLAN"]:
            min_confidence = 0.7  # Lower threshold for plan creation
        else:
            min_confidence = 0.8  # Higher threshold for logging
        
        # PRIORITY 1: Check for workout plan intent
        if intent_type == "WORKOUT_PLAN" and confidence > min_confidence and action_detected:
            print(f"DEBUG: Workout plan intent detected for: {text}")
            await mem.set_pending(user_id, {
                "state": "awaiting_workout_redirect_confirm",
                "original_text": text.strip()
            })
            
            async def _ask_workout_redirect():
                redirect_msg = "I can help you create a personalized workout plan! Would you like me to guide you to the workout template section where you can use our Kyra default template or make your own template with more customizing options?"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_msg)
                yield f"data: {json.dumps({'message': redirect_msg, 'type': 'workout_redirect_confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_ask_workout_redirect(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 2: Check for diet plan intent
        elif intent_type == "DIET_PLAN" and confidence > min_confidence and action_detected:
            print(f"DEBUG: Diet plan intent detected for: {text}")
            await mem.set_pending(user_id, {
                "state": "awaiting_diet_redirect_confirm",
                "original_text": text.strip()
            })
            
            async def _ask_diet_redirect():
                redirect_msg = "I can help you create a personalized diet plan! Would you like me to guide you to the diet template section where you can build your own custom plan or use our Kyra default template?"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_msg)
                yield f"data: {json.dumps({'message': redirect_msg, 'type': 'diet_redirect_confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_ask_diet_redirect(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 3: Check for food logging intent
        elif intent_type == "FOOD_LOGGING" and confidence > min_confidence and action_detected:
            print(f"DEBUG: Food logging intent detected for: {text}")
            await mem.set_pending(user_id, {
                "state": "awaiting_food_redirect_confirm",
                "original_text": text.strip()
            })
            
            async def _ask_food_redirect():
                redirect_msg = "I noticed you want to log food! For the best experience with food logging, scanning, and nutrition tracking, would you like me to guide you to the dedicated food logging section?"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_msg)
                yield f"data: {json.dumps({'message': redirect_msg, 'type': 'food_redirect_confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_ask_food_redirect(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 4: Check for workout logging intent
        elif intent_type == "WORKOUT_LOGGING" and confidence > min_confidence and action_detected:
            print(f"DEBUG: Workout logging intent detected for: {text}")
            await mem.set_pending(user_id, {
                "state": "awaiting_workout_log_redirect_confirm",
                "original_text": text.strip()
            })
            
            async def _ask_workout_log_redirect():
                redirect_msg = "I see you want to log your workout! Would you like me to guide you to the workout logging section where you can track your exercises in your templates?"
                await mem.add(user_id, "user", text.strip())
                await mem.add(user_id, "assistant", redirect_msg)
                yield f"data: {json.dumps({'message': redirect_msg, 'type': 'workout_log_redirect_confirm'})}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            
            return StreamingResponse(_ask_workout_log_redirect(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
        
        # If AI detected an intent but it's informational (not action), continue to normal chat
        print(f"DEBUG: AI detected {intent_type} but treating as informational (action_detected: {action_detected})")

    # Continue with existing analysis and normal chat logic
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
                content = (resp.choices[0].message.content or "").strip()
                content = re.sub(r'\bfit\s*bot\b|\bfit+bot\b|\bfitbot\b', 'Fittbot', content, flags=re.I)
                pretty = pretty_plan(content)

                async def _one_shot_followup():
                    yield sse_escape(pretty)
                    await mem.add(user_id, "assistant", pretty)
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_one_shot_followup(), media_type="text/event-stream",
                                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    if is_analysis_intent(tlower) and not pend:
        await mem.set_pending(user_id, {"state":"awaiting_analysis_confirm"})
        async def _ask_confirm():
            yield sse_json({"type":"analysis","status":"confirm",
                            "prompt":"Sure—let me analyse your diet and workout data. Shall we start?"})
            yield "event: ping\ndata: {}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_ask_confirm(), media_type="text/event-stream",
                                headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # Regular chat processing with friendly responses
    print(f"DEBUG: Processing as regular fitness chat")
    
    # Remove the problematic is_fitness_related check that's causing issues
    # Instead, let's handle greetings specifically and treat everything else as potential fitness content
    
    # More precise greeting detection - only trigger for actual greetings, not fitness questions
    simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    personal_questions = ['how are you', 'how do you do', 'whats up', 'what\'s up', 'your name', 'who are you', 'what are you', 'introduce yourself', 'tell me about yourself']
    
    # Check if the ENTIRE message is just a greeting (not part of a longer question)
    text_clean = text.lower().strip()
    is_simple_greeting = text_clean in simple_greetings
    is_personal_question = any(personal in text_clean for personal in personal_questions) and len(text.split()) <= 5
    
    is_greeting_or_personal = is_simple_greeting or is_personal_question
    
    # Only give the generic redirect for clear non-fitness topics AND not greetings
    non_fitness_keywords = [
        'weather', 'politics', 'election', 'movies', 'films', 'music', 'songs', 'concert',
        'sports team', 'football team', 'basketball', 'news today', 'stock market', 'crypto',
        'programming', 'coding', 'javascript', 'python', 'technology', 'computer',
        'travel destination', 'vacation', 'restaurant review', 'recipe for'
    ]
    
    # Only redirect if it's clearly about non-fitness topics AND not a greeting
    is_clearly_non_fitness = any(keyword in text.lower() for keyword in non_fitness_keywords) and not is_greeting_or_personal and 'fitness' not in text.lower() and 'health' not in text.lower()
    
    if is_clearly_non_fitness:
        async def _friendly_redirect():
            friendly_msg = "I focus on fitness, health, and wellness topics to give you the best guidance possible! Whether you need workout routines, nutrition advice, meal planning, or health tips, I'm here to help. What aspect of your fitness journey can I assist with today?"
            yield sse_escape(friendly_msg)
            await mem.add(user_id, "user", text.strip())
            await mem.add(user_id, "assistant", friendly_msg)
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_friendly_redirect(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    
    elif is_greeting_or_personal:
        async def _greeting_response():
            if text_clean in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']:
                friendly_msg = "Hello! I'm Kyra, your friendly fitness assistant. I'm here to help you with workouts, nutrition, diet planning, and all things fitness. What would you like to work on today?"
            elif any(phrase in text_clean for phrase in ['how are you', 'how do you do']):
                friendly_msg = "I'm doing great, thanks for asking! I'm excited to help you with your fitness journey. Whether you need workout plans, diet advice, or nutrition guidance, I'm here for you. What can I help you with?"
            elif any(phrase in text_clean for phrase in ['your name', 'who are you', 'what are you', 'introduce yourself']):
                friendly_msg = "I'm Kyra, your dedicated fitness companion! I specialize in helping people achieve their health and fitness goals through personalized workout plans, nutrition guidance, and wellness tips. How can I support your fitness journey?"
            else:
                friendly_msg = "Hi there! I'm Kyra, and I'm passionate about helping you with fitness, nutrition, and wellness. Whether you want to build muscle, lose weight, plan meals, or just get healthier, I'm here to guide you. What fitness goal are you working towards?"
            
            yield sse_escape(friendly_msg)
            await mem.add(user_id, "user", text.strip())
            await mem.add(user_id, "assistant", friendly_msg)
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_greeting_response(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # For all other queries (fitness, nutrition, supplements, etc.), proceed with normal AI chat
    is_meta = is_fittbot_meta_query(text)
    is_plan = is_plan_request(text)

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

    resp = oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=False, temperature=temperature)
    content = (resp.choices[0].message.content or "").strip()

    # Fix brand name confusion - Kyra is AI, Fittbot is app
    content = re.sub(r'\bfit\s*bot\b|\bfit+bot\b', 'Fittbot', content, flags=re.I)
    content = re.sub(r'\bfitbot\b', 'Fittbot', content, flags=re.I)
    
    # Don't replace Kyra with Fittbot when talking about the AI
    # But do replace wrong usage like "About Kyra" when user asks about Fittbot
    if 'fittbot' in text.lower() and 'about kyra' in content.lower():
        content = re.sub(r'\bAbout Kyra\b', 'About Fittbot', content, flags=re.I)
        content = re.sub(r'\bKyra is a comprehensive fitness app\b', 'Fittbot is a comprehensive fitness app', content, flags=re.I)
        content = re.sub(r'\bKyra is perfect for\b', 'Fittbot is perfect for', content, flags=re.I)
    
    content = re.sub(r'Would you like to log more foods.*?\?.*?🍏?', '', content, flags=re.I | re.DOTALL)
    content = re.sub(r'Let me know.*?log.*?for you.*?🍏?', '', content, flags=re.I | re.DOTALL)
    content = re.sub(r'Do you want.*?log.*?\?', '', content, flags=re.I)

    pretty = pretty_plan(content)
    async def _one_shot():
        yield sse_escape(pretty)
        await mem.add(user_id, "assistant", pretty)
        yield "event: done\ndata: [DONE]\n\n"
    return StreamingResponse(_one_shot(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# Keep all existing endpoints
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
        text = text.decode("utf-8", "ignore")
    return {"added_chunks": KB.upsert(src or file.filename, text)}

@router.post("/voice/transcribe", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def voice_transcribe(
    audio: UploadFile = File(...),
    http = Depends(get_http),
    oai = Depends(get_oai),
):
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")

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
            eng = (data.get("english") or text).strip()
            return {"lang": lang, "english": eng}
        except Exception:
            return {"lang":"unknown","english":text}

    tinfo = _translate_to_english(transcript)
    transcript_en = tinfo["english"]
    lang_code = tinfo["lang"]

    print(f"[voice] lang={lang_code} raw={transcript!r} | en={transcript_en!r}")

    return {
        "transcript": transcript_en,
        "lang": lang_code,
        "english": transcript_en,
    }

@router.post("/voice/stream_test", dependencies=[Depends(RateLimiter(times=20, seconds=60))])
async def voice_stream_sse(
    user_id: int,
    audio: UploadFile = File(...),
    mem = Depends(get_mem),
    oai = Depends(get_oai),
    http = Depends(get_http),
):
    transcript = await transcribe_audio(audio, http=http)
    if not transcript:
        raise HTTPException(400, "empty transcript")

    await mem.add(user_id, "user", transcript)
    msgs, _ = await build_messages(user_id, transcript, use_context=True, oai=oai, mem=mem)
    stream = oai_chat_stream(msgs, oai)
    filt = PlainTextStreamFilter()

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
    return {"status": 200}

@router.delete("/kb/clear")
async def kb_clear():
    """Clear all KB content completely"""
    initial_count = len(KB.texts)
    KB.texts.clear()
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
