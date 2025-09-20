#chatbot_services/llm_helpers.py
import os, re, json
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

# ===== env / model =====
APP_ENV        = os.getenv("APP_ENV", "prod")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_CHUNK_TOKENS = 350
TOP_K            = 4

# ---------- tiny tokenizer ----------
rough_tokens = lambda s: max(1, len(s) // 4)

def chunk_text(text: str, max_tok: int = MAX_CHUNK_TOKENS) -> List[str]:
    parts, buf, c = [], [], 0
    for line in text.split("\n"):
        t = rough_tokens(line)
        if c + t > max_tok and buf:
            parts.append("\n".join(buf).strip()); buf, c = [], 0
        if line.strip():
            buf.append(line.strip()); c += t
    if buf:
        parts.append("\n".join(buf).strip())
    return parts

# ====== nutrition db & macros ======
from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.nutrition import FOOD_DB, MACROS

# richer per-unit fallback (approx Indian servings)
PER_UNIT_DB: Dict[str, Dict[str, float]] = {
    "idli":              {"unit":"piece","calories":58,"protein":2,"carbs":12,"fat":0.3,"fiber":0.8,"sugar":0.2},
    "dosa":              {"unit":"piece","calories":133,"protein":3,"carbs":26,"fat":3,"fiber":1,"sugar":1},
    "chapati":           {"unit":"piece","calories":120,"protein":3,"carbs":22,"fat":2,"fiber":2,"sugar":1},
    "roti":              {"unit":"piece","calories":120,"protein":3,"carbs":22,"fat":2,"fiber":2,"sugar":1},
    "poori":             {"unit":"piece","calories":101,"protein":2.5,"carbs":11,"fat":5,"fiber":1,"sugar":0.5},
    "puri":              {"unit":"piece","calories":101,"protein":2.5,"carbs":11,"fat":5,"fiber":1,"sugar":0.5},
    "vada":              {"unit":"piece","calories":110,"protein":3,"carbs":10,"fat":7,"fiber":1,"sugar":0.5},
    "pongal":            {"unit":"100g","calories":130,"protein":4,"carbs":19,"fat":3.5,"fiber":1.5,"sugar":0.8},
    "rice":              {"unit":"100g","calories":130,"protein":2.7,"carbs":28,"fat":0.3,"fiber":0.4,"sugar":0},
    "biryani":           {"unit":"100g","calories":170,"protein":5.5,"carbs":23,"fat":5.5,"fiber":0.8,"sugar":1},
    "fried rice":        {"unit":"100g","calories":160,"protein":3.5,"carbs":25,"fat":4.5,"fiber":0.8,"sugar":1},
    "chicken rice":      {"unit":"100g","calories":180,"protein":8,"carbs":23,"fat":6,"fiber":0.7,"sugar":0.7},
    "cauliflower rice":  {"unit":"100g","calories":25,"protein":2,"carbs":5,"fat":0.2,"fiber":2,"sugar":2},
    "momos":             {"unit":"piece","calories":45,"protein":2,"carbs":6,"fat":1.5,"fiber":0.3,"sugar":0.3},
    "laddu":             {"unit":"piece","calories":175,"protein":3,"carbs":20,"fat":9,"fiber":1,"sugar":14},
    "eggs":              {"unit":"piece","calories":78,"protein":6,"carbs":0.6,"fat":5,"fiber":0,"sugar":0.6},
    "egg":               {"unit":"piece","calories":78,"protein":6,"carbs":0.6,"fat":5,"fiber":0,"sugar":0.6},
}

SYN: Dict[str, str] = {
    "puri": "poori",
    "rotti": "roti",
    "chapathi": "chapati",
    "friedrice": "fried rice",
    "chickenrice": "chicken rice",
}

UNIT_WORDS = ("grams","ml","pieces","cups")

def normalize_food(name: Optional[str]) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"\blog\s*(it|this)?\b", "", n).strip()
    if n in SYN: n = SYN[n]
    return n

# ====== intent gates ======
ACTION_VERBS = {"log","add","track","record","ate","eat","had","having"}
MEAL_WORDS   = {"breakfast","lunch","dinner","snack","snacks","brunch"}
HEALTH_CHAT  = {"bmi","body mass index","weight category","underweight","overweight",
                "height","cm","meter","mtr","kg","kgs","kilogram","ideal weight"}

def explicit_log_command(t: str) -> bool:
    lt = t.lower()
    return bool(re.search(r"\b(log|add|record|track)\b", lt))

def has_action_verb(t: str) -> bool:
    lt = t.lower()
    return any(v in lt for v in ACTION_VERBS)

def food_hits(text: str) -> int:
    lt = text.lower()
    keys = set(FOOD_DB.keys()) | set(PER_UNIT_DB.keys())
    return sum(1 for f in keys if f in lt)

def heuristic_confidence(text: str) -> float:
    t = text.lower()
    if any(k in t for k in HEALTH_CHAT):
        return 0.0
    verbs = has_action_verb(t)
    foods = food_hits(t)
    if verbs and foods:
        return min(1.0, 0.75 + 0.05 * max(0, foods - 1))
    if (MEAL_WORDS & set(t.split())) and (foods > 0) and re.search(r"\b\d+\b", t):
        return 0.6
    return 0.0

YES_RE = re.compile(
    r"\b(yes|yeah|yea|yup|y|absolutely|sure|sounds\s*good|great|perfect|"
    r"ok(?:ay)?|fine|alright|go\s*(ahead|for\s*it)|do\s*it|please\s*(do|log)|"
    r"log\s*(it)?|haa?n|हाँ|ठीक|चलो|कर\s*दो)\b",
    re.I,
)
NO_RE  = re.compile(r"\b(no|nope|nah|cancel|skip|नहीं|मत)\b", re.I)

is_yes = lambda t: bool(YES_RE.search(t))
is_no  = lambda t: bool(NO_RE.search(t))

# ===== SSE helpers =====
def sse_escape(txt: str) -> str:
    return "".join([f"data: {l}\n" for l in txt.split("\n")]) + "\n"

def sse_json(obj: Any) -> str:
    return sse_escape(json.dumps(obj, ensure_ascii=False))

# ===== qty validation & extraction =====
class QuantityValidator(BaseModel):
    quantity: float = Field(gt=0)
    @field_validator("quantity")
    @classmethod
    def _reasonable(cls, v: float):
        if v > 2000:  # guardrail
            raise ValueError("quantity too large")
        return v

_WORD_NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"half":0.5,"quarter":0.25
}

def extract_numbers(t: str):
    nums = [float(x) if "." in x else int(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    lt = t.lower()
    for w,n in _WORD_NUM.items():
        if re.search(rf"\b{re.escape(w)}\b", lt): nums.append(n)
    return nums

# ===== per-unit completion + scaling =====
def fallback_unit_hint(food: str) -> str:
    f = (food or "").lower()
    if any(w in f for w in ["rice","biryani","pulao","poha","upma","pongal","khichdi"]): return "grams"
    if any(w in f for w in ["juice","milk","coffee","tea","soup","water"]):   return "ml"
    if any(w in f for w in ["idli","dosa","chapati","roti","poori","puri","egg","vada","momo","momos","laddu"]):
        return "pieces"
    return "grams"

def _per_basis_for_name(name: str, unit_hint: Optional[str]) -> str:
    if name in PER_UNIT_DB:
        u = (PER_UNIT_DB[name].get("unit") or "").lower()
        if u == "piece": return "piece"
        if u == "100g":  return "100g"
        if u == "100ml": return "100ml"
    if (unit_hint or "") == "grams": return "100g"
    if (unit_hint or "") == "ml":    return "100ml"
    if (unit_hint or "") == "pieces":return "piece"
    return "unit"

def _get_per_unit_from_sources(name: str) -> Optional[Dict[str, float]]:
    db1 = FOOD_DB.get(name)
    if db1 and all(k in db1 for k in MACROS):
        return {k: float(db1[k]) for k in MACROS}
    db2 = PER_UNIT_DB.get(name)
    if db2 and all(k in db2 for k in MACROS):
        return {k: float(db2[k]) for k in MACROS}
    return None

def ensure_per_unit_macros(item: Dict[str, Any]) -> None:
    name = normalize_food(item.get("food"))
    unit_hint = item.get("unit_hint") or fallback_unit_hint(name)
    src = _get_per_unit_from_sources(name)
    if src:
        for k in MACROS:
            item[k] = float(item.get(k) or src[k])
    item["_per_basis"] = _per_basis_for_name(name, unit_hint)
    item["unit_hint"]  = unit_hint

def scale_item_inplace(item: Dict[str, Any]) -> None:
    q = item.get("quantity")
    try:
        qv = float(q) if q is not None else None
    except Exception:
        qv = None
    if qv is None:
        return

    unit = (item.get("unit_hint") or "pieces").lower()
    basis = (item.get("_per_basis") or "unit").lower()

    if unit in ("grams","g"):
        factor = (qv/100.0) if basis in ("100g","100ml") else qv if basis in ("gram","g") else (qv/100.0)
    elif unit in ("ml","milliliters","millilitre","milliliter"):
        factor = (qv/100.0) if basis in ("100ml","100g") else qv if basis in ("ml",) else (qv/100.0)
    else:
        factor = qv

    for k in MACROS:
        v = item.get(k)
        if isinstance(v, (int, float)):
            item[k] = round(float(v) * factor, 2)

def is_food_query(text: str) -> bool:
    """Specifically detect if the query is about food/nutrition"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Direct food terms - comprehensive list
    food_indicators = {
        # Generic food terms
        'food', 'foods', 'eat', 'eating', 'meal', 'meals', 'snack', 'snacks',
        'breakfast', 'lunch', 'dinner', 'brunch', 'supper', 'nutrition', 'nutritional',
        'calories', 'protein', 'vitamin', 'vitamins', 'mineral', 'minerals', 
        'healthy', 'diet', 'recipe', 'recipes', 'cook', 'cooking', 'ingredient', 'ingredients',
        'macros', 'macronutrients', 'micronutrients', 'fiber', 'carbs', 'carbohydrates',
        'fat', 'fats', 'sugar', 'sodium', 'antioxidants', 'nutrients', 'nutrient',
        
        # Common fruits
        'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'grape', 'grapes',
        'berry', 'berries', 'strawberry', 'strawberries', 'blueberry', 'blueberries',
        'mango', 'mangoes', 'pineapple', 'watermelon', 'papaya', 'kiwi', 'peach', 'pear',
        'cherry', 'cherries', 'lemon', 'lime', 'avocado', 'coconut', 'pomegranate',
        
        # Common vegetables
        'vegetable', 'vegetables', 'veggie', 'veggies', 'spinach', 'broccoli', 'carrot', 'carrots',
        'potato', 'potatoes', 'tomato', 'tomatoes', 'onion', 'onions', 'garlic', 'ginger',
        'cucumber', 'lettuce', 'cabbage', 'cauliflower', 'bell pepper', 'pepper', 'peppers',
        'corn', 'peas', 'beans', 'green beans', 'sweet potato', 'beetroot', 'radish',
        
        # Proteins
        'chicken', 'fish', 'beef', 'pork', 'turkey', 'lamb', 'eggs', 'egg', 'milk', 
        'cheese', 'yogurt', 'tofu', 'paneer', 'nuts', 'almonds', 'walnuts', 'cashews',
        'peanuts', 'seeds', 'chia seeds', 'flax seeds', 'quinoa', 'lentils', 'chickpeas',
        
        # Grains and starches
        'rice', 'bread', 'pasta', 'noodles', 'oats', 'oatmeal', 'cereal', 'wheat', 'barley',
        'millet', 'buckwheat', 'rye', 'flour', 'chapati', 'roti', 'naan', 'paratha',
        
        # Indian foods
        'idli', 'dosa', 'poori', 'puri', 'vada', 'pongal', 'biryani', 'dal', 'curry',
        'sabzi', 'samosa', 'pakora', 'upma', 'poha', 'khichdi', 'pulao', 'curd',
        
        # Beverages
        'juice', 'smoothie', 'tea', 'coffee', 'water', 'lassi', 'buttermilk',
        
        # Preparation methods
        'boiled', 'steamed', 'grilled', 'baked', 'fried', 'roasted', 'raw', 'fresh',
        'organic', 'whole grain', 'lean', 'low fat', 'sugar free'
    }
    
    # Food question patterns
    food_patterns = [
        r'\b(tell|about|what|is).*(apple|banana|food|nutrition|fruit|vegetable)',
        r'\b(calories|protein|vitamin|nutrition).*(in|of|content)',
        r'\b(healthy|good|bad|benefits).*(food|eat|diet|fruit|vegetable)',
        r'\b(how to|recipe|cook|prepare).*(food|meal|dish)',
        r'\b(nutritional|health).*(value|benefit|information)',
        r'\bneed.*(information|details).*(food|nutrition)',
        r'\bnutrition.*(facts|information|data)'
    ]
    
    # Check direct terms
    words = text_lower.split()
    if any(word in food_indicators for word in words):
        return True
    
    # Check patterns
    for pattern in food_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def is_fitness_related(text: str) -> bool:
    """Check if text is fitness/health/wellness related"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # First check if it's a food query
    if is_food_query(text):
        return True
    
    # Fitness and health keywords
    fitness_keywords = {
        # Basic fitness terms
        'workout', 'exercise', 'fitness', 'gym', 'muscle', 'strength', 'cardio',
        'running', 'weight', 'training', 'bodybuilding', 'yoga', 'pilates',
        'stretching', 'flexibility', 'endurance', 'sports', 'athletics',
        'health', 'wellness', 'diet', 'supplements', 'recovery', 'sleep', 'rest', 
        
        # Injury and pain related terms
        'injury', 'injuries', 'pain', 'ache', 'aches', 'hurt', 'hurts', 'hurting',
        'sore', 'soreness', 'cramp', 'cramps', 'cramping', 'spasm', 'spasms',
        'strain', 'strains', 'sprain', 'sprains', 'pulled muscle', 'torn',
        'inflammation', 'swelling', 'bruise', 'bruises', 'stiff', 'stiffness',
        'tight', 'tightness', 'knot', 'knots', 'trigger point', 'tender',
        
        # Body parts (fitness context)
        'muscle', 'muscles', 'joint', 'joints', 'bone', 'bones', 'tendon', 'tendons',
        'ligament', 'ligaments', 'spine', 'back', 'neck', 'shoulder', 'shoulders',
        'arm', 'arms', 'elbow', 'wrist', 'hand', 'hands', 'chest', 'abs', 'core',
        'hip', 'hips', 'leg', 'legs', 'thigh', 'thighs', 'knee', 'knees',
        'calf', 'calves', 'ankle', 'ankles', 'foot', 'feet', 'hamstring', 'quad',
        'glutes', 'glute', 'bicep', 'biceps', 'tricep', 'triceps',
        
        # Recovery and therapy
        'rehabilitation', 'rehab', 'physical therapy', 'physiotherapy', 'massage',
        'foam roll', 'foam rolling', 'ice', 'heat', 'compression', 'elevation',
        'rest day', 'recovery day', 'active recovery', 'stretching',
        
        # Health metrics and conditions
        'metabolism', 'heart rate', 'blood pressure', 'stress', 'mental health',
        'meditation', 'mindfulness', 'hydration', 'water', 'energy', 'fatigue',
        'body', 'physique', 'lean', 'bulk', 'cut', 'lose weight', 'gain weight',
        'muscle building', 'fat loss', 'toning', 'calories burned', 'heart health',
        'bone density', 'posture', 'balance', 'form', 'technique', 'routine',
        'program', 'schedule', 'plan', 'goal', 'motivation', 'progress',
        'tracking', 'measurement', 'bmi', 'body fat',
        
        # Exercise types and equipment
        'squats', 'deadlifts', 'bench press', 'pushups', 'pullups', 'planks',
        'lunges', 'burpees', 'hiit', 'crossfit', 'zumba', 'swimming', 'cycling',
        'walking', 'jogging', 'marathon', 'triathlon', 'weights', 'dumbbell',
        'barbell', 'kettlebell', 'resistance band', 'treadmill',
        
        # Common fitness issues and symptoms
        'dehydration', 'overtraining', 'plateau', 'doms', 'delayed onset muscle soreness',
        'shin splints', 'runner knee', 'tennis elbow', 'carpal tunnel',
        'lower back pain', 'sciatica', 'headache', 'migraine', 'dizziness',
        'nausea', 'exhaustion', 'burnout', 'insomnia'
    }
    
    # Non-fitness keywords that should be rejected (be specific)
    non_fitness_keywords = {
        'python programming', 'javascript code', 'html css', 'database query',
        'algorithm implementation', 'software development', 'computer science',
        'mathematics problem', 'physics equation', 'chemistry formula',
        'history facts', 'geography quiz', 'literature analysis', 'art history',
        'movie review', 'game strategy', 'business plan', 'marketing campaign',
        'political news', 'weather forecast', 'travel booking', 'entertainment news',
        'coding', 'programming', 'software', 'technology', 'computer', 'laptop',
        'phone', 'mobile', 'app development', 'web development'
    }
    
    # Check for non-fitness keywords first (but be more specific to avoid false positives)
    for keyword in non_fitness_keywords:
        if keyword in text_lower:
            return False
    
    # Check for fitness keywords
    if any(keyword in text_lower for keyword in fitness_keywords):
        return True
    
    # Check for fitness-related questions and symptoms
    fitness_question_patterns = [
        r'\b(how|what|when|where|why).*(exercise|workout|fitness|health|muscle|weight|diet|nutrition)',
        r'\b(lose|gain|build|tone).*(weight|muscle|strength|endurance)',
        r'\b(best|good|effective).*(exercise|workout|diet|nutrition)',
        r'\b(fitness|health|wellness|exercise).*(plan|routine|program|schedule)',
        r'\b(i|my).*(hurt|pain|ache|cramp|sore|tight|stiff)',
        r'\b(got|have|having).*(cramp|pain|ache|injury|soreness)',
        r'\b(muscle|leg|back|neck|shoulder|knee|ankle).*(cramp|pain|ache|hurt|sore)',
        r'\b(pulled|strained|twisted|injured).*(muscle|back|leg|arm|shoulder)',
        r'\bwhy.*(cramp|pain|ache|hurt|sore)',
        r'\bhow to.*(treat|heal|recover|fix).*(cramp|pain|injury|soreness)',
        r'\b(prevent|avoiding|stop).*(cramp|injury|pain)',
        r'\b(during|after|before).*(workout|exercise).*(pain|cramp|ache)'
    ]
    
    for pattern in fitness_question_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def _scale_macros(items: List[Dict[str, Any]]) -> None:
    for it in items:
        scale_item_inplace(it)

# ===== LLM wrappers / system styles =====
GENERAL_SYSTEM = (
    "You are KyraAI, a precise yet friendly fitness & nutrition assistant trained by Fittbot.\n"
    "Always answer in English, even if the user speaks another language (you may understand any language).\n"
    "Markdown is allowed; keep it clean and concise, you are allowed to use emojis.\n"
    "\n"
    "SCOPE: You help with fitness, exercise, nutrition, and health-related topics INCLUDING:\n"
    "✅ Food nutrition information (calories, protein, vitamins in fruits/vegetables/grains/proteins/etc.)\n"
    "✅ Healthy eating advice, meal planning, diet guidance, recipe suggestions\n"
    "✅ Exercise routines, workout plans, fitness techniques, and training advice\n"
    "✅ Health and wellness guidance (lifestyle, not medical diagnosis/treatment)\n"
    "✅ Common fitness-related issues: muscle cramps, soreness, basic injury prevention\n"
    "✅ Recovery advice: stretching, hydration, rest, basic pain management\n"
    "✅ Sports nutrition, pre/post workout meals, hydration\n"
    "✅ Cooking methods and food preparation for healthy living\n"
    "\n"
    "❌ You do NOT help with: programming/coding, technology, general trivia unrelated to health/fitness, "
    "academic subjects, entertainment, business advice (unless fitness-related)\n"
    "\n"
    "For health issues like cramps, pain, or soreness: Provide general wellness advice, possible causes, "
    "basic remedies (hydration, stretching, rest), and when to see a healthcare professional. "
    "Do not diagnose or provide medical treatment advice.\n"
    "\n"
    "Food logging: You can log food given in any language, but reply in English.\n"
    "Training: when someone asks when you were last trained, tell them you are getting trained everyday in a funny way.\n"
    "\n"
    "Fittbot knowledge policy:\n"
    "- If the user's question is about Fittbot plans, pricing, FAQs, app features, onboarding, or support, "
    "  first consult the provided CONTEXT and ground your answer strictly in it.\n"
    "- If the required info is missing from CONTEXT, say you don't have that information yet and invite the user "
    "  to provide or upsert it—do not guess or hallucinate.\n"
    "\n"
    "Brand policy:\n"
    "- The correct spelling is 'Fittbot' (exact casing). Normalize any variants in your output.\n"
    "- Do not compare Fittbot to other brands, do not make better/worse claims, and do not recommend one brand "
    "  over another. If asked to compare, follow the comparison policy.\n"
    "- Do not provide medical advice or diagnose conditions.\n"
    "\n"
    "Key Members of fittbot:\n"
    "- Founder and CEO - Mr. Nishad Shaik.\n"
    "- CoFounder and head of hr and operations - Mrs.Shama Ara.\n"
    "- CoFounder and CTO - Mr. Naveen Kumar.\n"
    "- CoFounder and IT Strategist - Mr. Martin Raju.\n"
)

COMPARE_POLICY = (
    "COMPARISON POLICY:\n"
    "- If the user asks to compare Fittbot with another fitness app/brand (Google Fit, MyFitnessPal, etc.) "
    "or asks 'which is better/best', do NOT engage in the comparison.\n"
    "- Respond with: 'I specialize in Fittbot features and services. I can tell you about Fittbot's "
    "capabilities and how they can help with your fitness goals. What specific Fittbot feature would you like to know about?'\n"
    "- Do NOT provide comparison tables, pros/cons lists, or decision criteria that involve competitors.\n"
    "- Do NOT name or describe other fitness apps beyond acknowledging the user mentioned them.\n"
    "- Focus exclusively on Fittbot's features and benefits from the provided CONTEXT.\n"
    "- If pressed for comparisons, say: 'I focus exclusively on Fittbot's features and benefits. "
    "Let me tell you about what Fittbot offers for your fitness journey.'\n"
)

STYLE_PLAN = (
    "When the user asks for a DIET PLAN or WORKOUT PLAN, present the answer in clean Markdown:\n"
    "1) Start with a short bold title.\n"
    "2) Use numbered sections for days or workouts.\n"
    "3) Use short bold labels (e.g., **Breakfast**, **Lunch**, **Workout A**, **Sets x Reps**).\n"
    "4) Keep lines concise (1–2 sentences each). No long paragraphs.\n"
    "5) If giving options, put them as sub-bullets.\n"
    "6) End with a short **Notes** section (3–5 bullets)."
)

PLAN_TRIGGERS = {
    "diet plan","meal plan","weight gain plan","weight loss plan","bulking plan",
    "cutting plan","nutrition plan","calorie plan","workout plan","training plan",
    "exercise plan","push pull legs","full body plan","split routine","program",
    "routine","2 weeks plan","weekly plan","daily plan"
}

STYLE_CHAT_FORMAT = (
    "Format answers in clean, concise Markdown:\n"
    "• Start with a short bold title.\n"
    "• Prefer numbered steps and short bullet points (1–2 lines each).\n"
    "• Use bold labels like **Ingredients**, **Steps**, **Macros**, **Tips**.\n"
    "• For nutrition lists, show a compact table: Item | Calories | Protein | Carbs | Fat.\n"
    "• Avoid long paragraphs; break into bullets."
)

_FIT_NUTRI_KEYWORDS = {
    "meal","recipe","cook","make","ingredients","macros","calories","protein",
    "carb","fat","diet","snack","breakfast","lunch","dinner","post workout",
    "pre workout","workout","exercise","sets","reps","routine","program","split",
    "strength","hypertrophy","cardio","warm up","cool down","mobility"
}

def is_fit_chat(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _FIT_NUTRI_KEYWORDS)

def is_plan_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in PLAN_TRIGGERS)

def _tighten_blank_lines(s: str) -> str:
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s

def _ensure_spaces(s: str) -> str:
    s = re.sub(r"([,:;])([^\s])", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    return s

def pretty_plan(raw: str) -> str:
    if not raw:
        return ""
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = _ensure_spaces(s)
    s = _tighten_blank_lines(s)
    s = s.replace("•", "- ")
    s = re.sub(r"(^|\n)(#+ .+)\n(?!\n|- |\d+\.)", r"\1\2\n\n", s)
    return s

# streaming filter that strips markdown emphasis outside code blocks
class PlainTextStreamFilter:
    _re_h = re.compile(r'^\s{0,3}#{1,6}\s*')
    _re_b = re.compile(r'^\s{0,3}(?:[-*•]|\d+[.)])\s+')
    _re_i = re.compile(r'(`+|\*\*|\*|__|_)')

    def __init__(self):
        self.buf = ""
        self.code = False

    def _toggle(self, l: str) -> str:
        out = []
        i = 0
        while i < len(l):
            if l.startswith("```", i):
                self.code = not self.code
                i += 3
                continue
            out.append(l[i])
            i += 1
        return "".join(out)

    def _clean(self, l: str) -> str:
        return l if self.code else self._re_i.sub("", self._re_b.sub("", self._re_h.sub("", l)))

    def feed(self, ch: str) -> str:
        if not ch:
            return ""
        ch = ch.replace("\r\n", "\n").replace("\r", "\n")
        self.buf += ch
        if "\n" not in self.buf:
            return ""
        lines = self.buf.split("\n")
        self.buf = lines[-1]
        ready = lines[:-1]
        return "".join(self._clean(self._toggle(ln)) + "\n" for ln in ready)

    def flush(self) -> str:
        if not self.buf:
            return ""
        ln = self._clean(self._toggle(self.buf))
        self.buf = ""
        return ln

# ===== LLM call helpers =====
CAL_GUIDE=("chapati 120kcal … idli 58kcal … (macros: calories, protein, carbs, fat, fiber, sugar)")

def oai_chat_stream(msgs: List[Dict[str,str]], oai: OpenAI):
    return oai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, stream=True)

def _fallback_split_items(user_text: str) -> List[Dict[str, Any]]:
    t = re.sub(r"\blog\s*(it|this)?\b", "", user_text, flags=re.I)
    parts = re.split(r",| and ", t)
    out=[]
    for p in parts:
        name = normalize_food(re.sub(r"[^a-zA-Z\s]", " ", p))
        if not name or len(name) < 2:
            continue
        unit_hint = fallback_unit_hint(name)
        fake = {"food": name, "unit": None, "quantity": None, "unit_hint": unit_hint}
        ensure_per_unit_macros(fake)
        out.append({
            **fake,
            "ask": f"How many {unit_hint} of {name} did you have?",
            "qty_from": "none",
        })
    return out or [{
        "food": normalize_food(user_text),
        "unit": None, "quantity": None, "qty_from":"none",
        "unit_hint": fallback_unit_hint(user_text),
        "ask": f"How many {fallback_unit_hint(user_text)} of {normalize_food(user_text)} did you have?",
    }]

def gpt_extract_items(user_text: str, oai: OpenAI, candidates: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    base_rules = (
        "You are a strict nutrition item extractor. Return ONLY JSON:\n"
        "{items:[{food, unit, quantity, calories, protein, carbs, fat, fiber, sugar, unit_hint, ask, qty_from}]}.\n"
        "- Detect foods from free text for an Indian audience.\n"
        "- If quantity present ('2 idlis', '200g rice'), set quantity and qty_from:'user'; else quantity:null, qty_from:'none'.\n"
        "- unit_hint MUST be one of ['grams','ml','pieces','cups'] (use grams for rice/biryani/poha/etc.; ml for drinks; pieces for idli/dosa/roti/egg/momos/laddu etc.).\n"
        "- Provide PER-UNIT macros (unscaled): per 100g for grams, per 100ml for ml, per piece for pieces.\n"
        "- 'ask' is a short question using unit_hint (e.g., 'How many grams of rice did you have?').\n"
        "- Do NOT include trigger verbs like 'log','add','record','track' in the food name.\n"
        "- If you cannot identify any foods, return {items:[]} (empty). No guesses. No prose."
    )
    if candidates:
        constraint = (
            "\nHARD CONSTRAINTS:\n"
            f"- Consider ONLY these candidate strings as possible foods (case-insensitive): {candidates}.\n"
            "- You MUST NOT add items that are not present in the candidates list.\n"
            "- If none are foods, return {items:[]}. Do not invent."
        )
    else:
        constraint = ""
    system = base_rules + constraint
    msgs = [{"role":"system","content":system},{"role":"user","content":user_text}]
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            response_format={"type":"json_object"},
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        items = data.get("items", [])
        cleaned: List[Dict[str, Any]] = []

        if isinstance(items, list):
            for it in items:
                name = normalize_food(it.get("food"))
                if not name: continue
                it["food"] = name
                if not it.get("unit_hint") or it.get("unit_hint") not in UNIT_WORDS:
                    it["unit_hint"] = fallback_unit_hint(name)
                if not it.get("ask"):
                    it["ask"] = f"How many {it['unit_hint']} of {name} did you have?"
                ensure_per_unit_macros(it)
                for k in ("calories","protein","carbs","fat","fiber","sugar"):
                    v = it.get(k)
                    if isinstance(v, str):
                        try: it[k] = float(v.strip())
                        except Exception: pass
                cleaned.append(it)
        return cleaned if cleaned else _fallback_split_items(user_text)
    except Exception:
        return _fallback_split_items(user_text)

# ===== Router (small) – kept for compatibility; returns chat by default =====
ROUTER_SYSTEM = (
 "You are KyraAI Router. Output ONLY JSON.\n"
 "Pick intent 'log_food' or 'chat'. Schema {intent, food|null}."
)

def gpt_small_route(now_iso: str, user_text: str, oai: OpenAI) -> Dict[str, Any]:
    msgs=[
        {"role":"system","content":ROUTER_SYSTEM},
        {"role":"system","content":CAL_GUIDE},
        {"role":"system","content":f"NOW={now_iso}"},
        {"role":"user","content":user_text},
    ]
    try:
        resp = oai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=msgs,
            response_format={"type":"json_object"},
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"intent":"chat","food":None, "_reason":"json_parse_failed"}

def first_missing_quantity(items: List[Dict[str,Any]]):
    for i,it in enumerate(items):
        if it.get("quantity") in (None,0,""):
            return i,(it.get("food") or it.get("name"))
    return -1,None

# ===== KB-aware message builder with meta & comparison detection =====
PLAN_KEYWORDS = {
    "plan","plans","pricing","price","subscription","premium","basic","pro",
    "monthly","yearly","annual","offer","trial","faq","features"
}
FIT_PLAN_HINTS = {
    "diet","meal","food","calorie","macros","protein","carbs","fat",
    "workout","exercise","sets","reps","split","routine","push","pull","legs",
    "full body","upper","lower","strength","hypertrophy","cardio","warm up","cool down"
}

def is_fittbot_meta_query(t: str) -> bool:
    tt = (t or "").lower()
    if is_plan_request(tt) or is_fit_chat(tt) or any(h in tt for h in FIT_PLAN_HINTS):
        return False
    has_meta = any(k in tt for k in PLAN_KEYWORDS)
    mentions_brand = ("fittbot" in tt) or ("kyra" in tt)
    return has_meta or mentions_brand

def extract_plan_tokens(t: str) -> List[str]:
    tt = (t or "").lower()
    want = []
    for k in ["premium","basic","pro","starter","plus","gold","silver","platinum","monthly","yearly","annual"]:
        if k in tt: want.append(k)
    return want

# --- competitor comparison detection ---
_COMPETITOR_TOKENS = {
    "healthify", "healthifyme", "cult", "cultfit", "fittr", "curefit",
    "myfitnesspal", "loseit", "hevy", "fitbit", "google fit", "apple fitness",
    "garmin", "strava", "zorba", "ultrahuman"
}
_COMPARE_TRIGGERS = {"vs", "versus", "compare", "comparison", "better", "best", "cheaper", "cost", "pricing", "price"}

def is_competitor_compare_query(t: str) -> bool:
    tt = (t or "").lower()
    has_fittbot = "fittbot" in tt or "fitbot" in tt or "fitt bot" in tt
    has_trigger = any(k in tt for k in _COMPARE_TRIGGERS)
    has_other   = any(k in tt for k in _COMPETITOR_TOKENS)
    vs_style = " vs " in tt or "versus" in tt
    return (has_fittbot and (has_trigger or has_other or vs_style)) or (vs_style and has_other)

async def build_messages(
    user_id: str,
    user_text_en: str,
    *, use_context: bool, oai: OpenAI, mem=None,
    context_only: bool=False,
    k: int = TOP_K
) -> Tuple[List[Dict[str,str]], List[Dict[str,Any]]]:
    from app.fittbot_api.v1.client.client_api.chatbot.chatbot_services.kb_store import KB

    # Check if it's a food/nutrition query or fitness related
    is_food = is_food_query(user_text_en)
    is_fitness = is_fitness_related(user_text_en)
    
    # If it's neither food nor fitness related, we'll let the system prompt handle the rejection
    # but we still need to build the message properly
    
    # detect meta & comparison
    cmp_intent = is_competitor_compare_query(user_text_en)
    context_only = context_only or cmp_intent

    ctx_hits: List[Dict[str,Any]] = []
    if use_context:
        ctx_hits = KB.search(user_text_en, k=k) or []
        if is_fittbot_meta_query(user_text_en):
            toks = extract_plan_tokens(user_text_en)
            if toks:
                q2 = " ".join({"fittbot", "plans", "pricing", *toks})
            else:
                q2 = "fittbot plans pricing features premium basic pro monthly yearly"
            extra = KB.search(q2, k=max(3, k)) or []
            seen = set(id(h["text"]) for h in ctx_hits)
            for h in extra:
                if id(h["text"]) not in seen:
                    ctx_hits.append(h); seen.add(id(h["text"]))

    ctx = "\n\n---\n\n".join(h["text"] for h in ctx_hits) if ctx_hits else "(none)"

    history: List[Dict[str, str]] = []
    if mem is not None:
        try:
            history = await mem.recent(user_id)
        except Exception:
            history = []

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": GENERAL_SYSTEM},
    ]

    if cmp_intent:
        msgs.append({"role": "system", "content": COMPARE_POLICY})

    if context_only:
        msgs.append({"role":"system","content":
            "CONTEXT_ONLY=TRUE. You MUST answer strictly from CONTEXT. "
            "If the answer is not present, say you don't have that information and ask the user to add it. "
            "Never guess or fabricate."
        })

    # Add a system message to handle non-fitness queries
    if not (is_food or is_fitness):
        msgs.append({"role": "system", "content": 
            "The user's query appears to be outside fitness, health, wellness, or food topics. "
            "Politely redirect them by saying: 'I'm a specialized fitness assistant and can only help with exercise, health, and wellness topics. How can I help you with your fitness journey today?'"
        })

    msgs += [
        {"role": "system", "content": f"CONTEXT:\n{ctx}"},
        *history,
        {"role": "user", "content": user_text_en.strip()},
    ]
    return msgs, ctx_hits

__all__ = [
    "PlainTextStreamFilter","oai_chat_stream","GENERAL_SYSTEM","TOP_K","build_messages",
    "heuristic_confidence","gpt_extract_items","first_missing_quantity","OPENAI_MODEL",
    "sse_json","sse_escape","gpt_small_route","_scale_macros","is_yes","is_no",
    "QuantityValidator","extract_numbers","has_action_verb","food_hits","explicit_log_command",
    "fallback_unit_hint","normalize_food","ensure_per_unit_macros","STYLE_PLAN",
    "is_plan_request","is_fit_chat","STYLE_CHAT_FORMAT","pretty_plan",
    "is_fittbot_meta_query","extract_plan_tokens","is_competitor_compare_query","COMPARE_POLICY",
    "is_food_query","is_fitness_related",
]
