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

# Import specialized chatbot functionalities
from app.models.fittbot_models import Client, WeightJourney, WorkoutTemplate, ClientDietTemplate, MealTemplate, ActualDiet, ClientTarget
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
        'need', 'want', 'help', 'assist',
        # Logging words
        'log', 'track', 'record', 'save', 'add', 'input', 'enter',
        # Past tense logging
        'ate', 'had', 'consumed', 'did', 'completed', 'finished', 'done',
        # Plan-related words
        'workout', 'exercise', 'diet', 'meal', 'nutrition', 'template', 'routine'
    ]

    text_lower = text.lower()

    # Also check for fitness-related phrases that imply action
    fitness_action_phrases = [
        'workout plan', 'exercise plan', 'diet plan', 'meal plan',
        'fitness routine', 'training program', 'nutrition plan',
        'weight loss plan', 'muscle gain plan'
    ]

    return (any(word in text_lower for word in action_words) or
            any(phrase in text_lower for phrase in fitness_action_phrases))

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

# =============================================================================
# SPECIALIZED CHATBOT FUNCTIONALITIES INTEGRATED
# =============================================================================

# Food Logging Functions
def extract_food_info_using_ai(text: str, oai):
    """AI-driven food extraction with comprehensive prompt"""

    prompt = f"""
    Analyze this text and extract food information: "{text}"

    CRITICAL FOOD IDENTIFICATION RULES:
    1. COMPOUND FOODS: Treat compound words as SINGLE dishes:
       - "curdrice" = "curd rice" (one dish, not separate curd and rice)
       - "lemonrice" = "lemon rice" (one dish)
       - "masalatea" = "masala tea" (one dish)

    2. FOOD DETECTION: Extract ALL foods/drinks, handle misspellings liberally
    3. CONTEXT AWARENESS: Consider Indian cuisine context for units and dishes

    INTELLIGENT UNIT ASSIGNMENT:
    When quantity is provided, choose the MOST LOGICAL unit based on:

    INDIAN RICE DISHES (use plates/bowls):
    - Any rice dish: biryani, pulao, fried rice, lemon rice, curd rice → plates
    - Curries, dal, sambar → bowls

    MEASUREMENT CONTEXT:
    - "spoon", "spoons" → tablespoons (NOT grams)
    - Small countable items → pieces
    - Liquids → ml, cups, glasses
    - Large servings → plates, bowls
    - Precise measurements → grams, kg

    QUANTITY INTERPRETATION:
    - If user provides quantity, extract it exactly
    - If no quantity, set to null
    - Choose unit that matches natural serving size

    Return ONLY valid JSON array:
    [
        {{
            "name": "properly_formatted_food_name",
            "quantity": number_or_null,
            "unit": "contextually_appropriate_unit",
            "calories": number_or_null,
            "protein": number_or_null,
            "carbs": number_or_null,
            "fat": number_or_null,
            "fiber": number_or_null,
            "sugar": number_or_null
        }}
    ]
    """

    try:
        response = oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a nutrition expert. Extract food information accurately and return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        import re
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

        foods = json.loads(result)
        if not isinstance(foods, list):
            foods = [foods] if isinstance(foods, dict) else []

        return {"foods": foods}

    except Exception as e:
        print(f"Food extraction error: {e}")
        return {"foods": []}

def calculate_nutrition_using_ai(food_name, quantity, unit, oai):
    """Enhanced nutrition calculation with better unit handling"""
    try:
        prompt = f"""
        Calculate nutrition for: {quantity} {unit} of {food_name}

        Use these REALISTIC conversions:
        - 1 plate (rice dishes) = 300 grams
        - 1 tablespoon = 15 grams (solids) or 15 ml (liquids)
        - 1 teaspoon = 5 grams (solids) or 5 ml (liquids)
        - 1 cup = 200 grams (solids) or 200 ml (liquids)
        - 1 bowl = 200 grams
        - 1 glass = 200 ml
        - 1 piece varies by food type (estimate appropriately)

        Return ONLY valid JSON with realistic values:
        {{
            "calories": number,
            "protein": number,
            "carbs": number,
            "fat": number,
            "fiber": number,
            "sugar": number
        }}
        """

        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a nutrition expert. Always provide realistic nutrition values."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

        nutrition = json.loads(result)
        return nutrition

    except Exception as e:
        print(f"AI nutrition calculation failed: {e}")
        return {"calories": 100, "protein": 3, "carbs": 15, "fat": 2, "fiber": 1, "sugar": 2}

def store_diet_data_to_db(db: Session, client_id: int, date: str, logged_foods: list, meal: str):
    """Store logged food data in actual_diet table"""
    try:
        existing_entry = db.query(ActualDiet).filter(
            ActualDiet.client_id == client_id,
            ActualDiet.date == date
        ).first()

        food_items = []
        for food in logged_foods:
            food_item = {
                "id": str(int(datetime.now().timestamp() * 1000)),
                "name": food.get('name', ''),
                "quantity": f"{food.get('quantity', 0)} {food.get('unit', 'serving')}",
                "calories": food.get('calories', 0),
                "protein": food.get('protein', 0),
                "carbs": food.get('carbs', 0),
                "fat": food.get('fat', 0),
                "fiber": food.get('fiber', 0),
                "sugar": food.get('sugar', 0),
                "image_url": ""
            }
            food_items.append(food_item)

        if existing_entry:
            diet_data = existing_entry.diet_data if existing_entry.diet_data else []
            meal_found = False
            for meal_category in diet_data:
                if meal_category.get("title", "").lower() == meal.lower():
                    meal_category["foodList"].extend(food_items)
                    meal_category["itemsCount"] = len(meal_category["foodList"])
                    meal_found = True
                    break

            if not meal_found:
                default_structure = get_default_diet_structure()
                for default_meal in default_structure:
                    if default_meal.get("title", "").lower() == meal.lower():
                        default_meal["foodList"] = food_items
                        default_meal["itemsCount"] = len(food_items)
                        diet_data.append(default_meal)
                        break

            from sqlalchemy.orm import attributes
            attributes.flag_modified(existing_entry, "diet_data")
            existing_entry.diet_data = diet_data
            db.commit()

        else:
            diet_data = get_default_diet_structure()
            for meal_category in diet_data:
                if meal_category.get("title", "").lower() == meal.lower():
                    meal_category["foodList"] = food_items
                    meal_category["itemsCount"] = len(food_items)
                    break

            new_entry = ActualDiet(
                client_id=client_id,
                date=date,
                diet_data=diet_data
            )
            db.add(new_entry)
            db.commit()

        return True

    except Exception as e:
        print(f"Error storing diet data: {e}")
        db.rollback()
        return False

def get_default_diet_structure():
    """Return the default diet structure"""
    return [
        {"id": "1", "title": "Pre workout", "tagline": "Energy boost", "foodList": [], "timeRange": "6:30-7:00 AM", "itemsCount": 0},
        {"id": "2", "title": "Post workout", "tagline": "Recovery fuel", "foodList": [], "timeRange": "7:30-8:00 AM", "itemsCount": 0},
        {"id": "3", "title": "Early morning Detox", "tagline": "Early morning nutrition", "foodList": [], "timeRange": "5:30-6:00 AM", "itemsCount": 0},
        {"id": "4", "title": "Pre-Breakfast / Pre-Meal Starter", "tagline": "Pre-breakfast fuel", "foodList": [], "timeRange": "7:00-7:30 AM", "itemsCount": 0},
        {"id": "5", "title": "Breakfast", "tagline": "Start your day right", "foodList": [], "timeRange": "8:30-9:30 AM", "itemsCount": 0},
        {"id": "6", "title": "Mid-Morning snack", "tagline": "Healthy meal", "foodList": [], "timeRange": "10:00-11:00 AM", "itemsCount": 0},
        {"id": "7", "title": "Lunch", "tagline": "Nutritious midday meal", "foodList": [], "timeRange": "1:00-2:00 PM", "itemsCount": 0},
        {"id": "8", "title": "Evening snack", "tagline": "Healthy meal", "foodList": [], "timeRange": "4:00-5:00 PM", "itemsCount": 0},
        {"id": "9", "title": "Dinner", "tagline": "End your day well", "foodList": [], "timeRange": "7:30-8:30 PM", "itemsCount": 0},
        {"id": "10", "title": "Bed time", "tagline": "Rest well", "foodList": [], "timeRange": "9:30-10:00 PM", "itemsCount": 0}
    ]

# Workout Template Functions
def fetch_client_profile(db: Session, client_id: int):
    """Fetch complete client profile for workout generation"""
    try:
        w = db.query(WeightJourney).where(WeightJourney.client_id == client_id).order_by(WeightJourney.id.desc()).first()
        current_weight = float(w.actual_weight) if w and w.actual_weight is not None else 70.0
        target_weight = float(w.target_weight) if w and w.target_weight is not None else 65.0

        c = db.query(Client).where(Client.client_id == client_id).first()
        client_goal = (getattr(c, "goals", None) or getattr(c, "goal", None) or "muscle gain") if c else "muscle gain"
        lifestyle = c.lifestyle if c else "moderate"

        ct = db.query(ClientTarget).where(ClientTarget.client_id == client_id).first()
        target_calories = float(ct.calories) if ct and ct.calories else 2000.0

        return {
            "client_id": client_id,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "client_goal": client_goal,
            "target_calories": target_calories,
            "lifestyle": lifestyle,
        }
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return {
            "client_id": client_id,
            "current_weight": 70.0,
            "target_weight": 65.0,
            "client_goal": "muscle gain",
            "target_calories": 2000.0,
            "lifestyle": "moderate",
        }

def generate_workout_template_with_ai(user_request: str, profile: dict, oai):
    """Generate workout template using AI"""
    try:
        print(f"DEBUG: Starting workout generation for request: '{user_request}'")
        prompt = f"""
        Create a workout template based on this request: "{user_request}"

        User Profile:
        - Current Weight: {profile['current_weight']}kg
        - Target Weight: {profile['target_weight']}kg
        - Goal: {profile['client_goal']}
        - Lifestyle: {profile['lifestyle']}

        Generate a 6-day workout plan (Monday-Saturday) with proper exercise selection.

        Return ONLY valid JSON:
        {{
            "template_name": "descriptive_name",
            "days": {{
                "monday": {{
                    "title": "Day 1 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}},
                        {{"name": "Exercise Name", "sets": 3, "reps": 10}}
                    ]
                }},
                "tuesday": {{
                    "title": "Day 2 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}}
                    ]
                }},
                "wednesday": {{
                    "title": "Day 3 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}}
                    ]
                }},
                "thursday": {{
                    "title": "Day 4 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}}
                    ]
                }},
                "friday": {{
                    "title": "Day 5 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}}
                    ]
                }},
                "saturday": {{
                    "title": "Day 6 - Focus Area",
                    "exercises": [
                        {{"name": "Exercise Name", "sets": 3, "reps": 12}}
                    ]
                }}
            }}
        }}
        """

        print(f"DEBUG: Sending workout request to OpenAI...")
        response = oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fitness expert. Generate comprehensive workout templates in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()
        print(f"DEBUG: OpenAI workout response length: {len(result)}")
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

        workout_data = json.loads(result)
        print(f"DEBUG: Successfully parsed workout JSON with keys: {list(workout_data.keys())}")
        return workout_data

    except Exception as e:
        print(f"Workout generation error: {e}")
        return None

def generate_diet_template_with_ai(user_request: str, profile: dict, oai):
    """Generate diet template using AI"""
    try:
        print(f"DEBUG: Starting diet generation for request: '{user_request}'")
        prompt = f"""
        Create a diet template based on this request: "{user_request}"

        User Profile:
        - Current Weight: {profile['current_weight']}kg
        - Target Weight: {profile['target_weight']}kg
        - Goal: {profile['client_goal']}
        - Target Calories: {profile['target_calories']}
        - Lifestyle: {profile['lifestyle']}

        Generate a comprehensive daily meal plan with 10 meal slots as shown in the structure.

        Return ONLY valid JSON with this exact structure - include all 10 meal slots:
        {{
            "template_name": "descriptive_name",
            "meals": [
                {{
                    "id": "1",
                    "title": "Pre workout",
                    "tagline": "Energy boost",
                    "foodList": [
                        {{"name": "Food Name", "quantity": "amount unit", "calories": 100, "protein": 5, "carbs": 15, "fat": 3}}
                    ],
                    "timeRange": "6:30-7:00 AM",
                    "itemsCount": 1
                }},
                {{
                    "id": "2",
                    "title": "Post workout",
                    "tagline": "Recovery fuel",
                    "foodList": [
                        {{"name": "Food Name", "quantity": "amount unit", "calories": 100, "protein": 5, "carbs": 15, "fat": 3}}
                    ],
                    "timeRange": "7:30-8:00 AM",
                    "itemsCount": 1
                }},
                {{
                    "id": "3",
                    "title": "Breakfast",
                    "tagline": "Start your day right",
                    "foodList": [
                        {{"name": "Food Name", "quantity": "amount unit", "calories": 100, "protein": 5, "carbs": 15, "fat": 3}}
                    ],
                    "timeRange": "8:30-9:30 AM",
                    "itemsCount": 1
                }}
            ]
        }}
        """

        print(f"DEBUG: Sending diet request to OpenAI...")
        response = oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Generate comprehensive diet templates in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()
        print(f"DEBUG: OpenAI diet response length: {len(result)}")
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

        diet_data = json.loads(result)
        print(f"DEBUG: Successfully parsed diet JSON with keys: {list(diet_data.keys())}")
        return diet_data

    except Exception as e:
        print(f"Diet generation error: {e}")
        return None

def format_workout_template_display(template: dict) -> str:
    """Format workout template for display"""
    if not template or not template.get('days'):
        return "❌ No workout data available"

    formatted_lines = []
    formatted_lines.append(f"💪 {template.get('template_name', 'Your Workout Template').upper()} 💪")
    formatted_lines.append("═" * 50)
    formatted_lines.append("")

    day_count = 1
    day_emojis = {1: "🔥", 2: "💥", 3: "⚡", 4: "🚀", 5: "💪", 6: "🎯", 7: "🌟"}

    for day_key, day_data in template['days'].items():
        if not isinstance(day_data, dict):
            continue

        day_emoji = day_emojis.get(day_count, "💫")
        title = day_data.get('title', f'Day {day_count}')

        formatted_lines.append(f"{day_emoji} {title.upper()} {day_emoji}")
        formatted_lines.append("─" * 30)
        formatted_lines.append("")

        exercises = day_data.get('exercises', [])
        if exercises:
            for i, exercise in enumerate(exercises, 1):
                name = exercise.get('name', 'Unknown Exercise')
                sets = exercise.get('sets', 0)
                reps = exercise.get('reps', 0)
                formatted_lines.append(f"   🏋️ {i}. {name}")
                formatted_lines.append(f"      📊 {sets} sets × {reps} reps")
                formatted_lines.append("")
        else:
            formatted_lines.append("   ⚠️ No exercises added yet")
            formatted_lines.append("")

        formatted_lines.append("")
        day_count += 1

    return "\n".join(formatted_lines)

def extract_workout_info_using_ai(text: str, oai):
    """Extract workout information using AI"""
    try:
        prompt = f"""
        Analyze this text and extract workout/exercise information: "{text}"

        Extract any exercises mentioned along with sets, reps, and duration if provided.

        Return ONLY valid JSON array:
        [
            {{
                "name": "exercise_name",
                "sets": number_or_null,
                "reps": number_or_null,
                "duration": "time_string_or_null",
                "weight": "weight_string_or_null"
            }}
        ]

        Examples:
        - "I did 3 sets of 10 pushups" → [{{"name": "pushups", "sets": 3, "reps": 10, "duration": null, "weight": null}}]
        - "ran for 30 minutes" → [{{"name": "running", "sets": null, "reps": null, "duration": "30 minutes", "weight": null}}]
        - "bench press with 60kg, 3 sets of 8" → [{{"name": "bench press", "sets": 3, "reps": 8, "duration": null, "weight": "60kg"}}]
        """

        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fitness expert. Extract workout information accurately and return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )

        result = response.choices[0].message.content.strip()
        result = re.sub(r"^```json\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

        exercises = json.loads(result)
        if not isinstance(exercises, list):
            exercises = [exercises] if isinstance(exercises, dict) else []

        return {"exercises": exercises}

    except Exception as e:
        print(f"Workout extraction error: {e}")
        return {"exercises": []}

def parse_quantity_input(text: str):
    """Parse quantity input from user"""
    text_lower = text.lower().strip()

    # Pattern matching for quantity with optional unit
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(spoons?|tablespoons?|tbsp|teaspoons?|tsp|plates?|bowls?|pieces?|pcs?|pc|g|grams?|kg|ml|cups?|glasses?)',
        r'(\d+(?:\.\d+)?)',  # Just numbers
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            quantity = float(match.group(1))
            unit = match.group(2) if len(match.groups()) > 1 else None

            # Normalize unit
            unit_map = {
                'spoon': 'tablespoons', 'spoons': 'tablespoons',
                'tablespoon': 'tablespoons', 'tbsp': 'tablespoons',
                'teaspoon': 'teaspoons', 'tsp': 'teaspoons',
                'plate': 'plates', 'bowl': 'bowls',
                'piece': 'pieces', 'pcs': 'pieces', 'pc': 'pieces',
                'g': 'grams', 'gram': 'grams',
                'kg': 'kg', 'kilogram': 'kg',
                'cup': 'cups', 'glass': 'glasses',
                'ml': 'ml'
            }

            if unit:
                unit = unit_map.get(unit, unit)

            return quantity, unit

    return None, None

def format_diet_template_display(template: dict) -> str:
    """Format diet template for display"""
    if not template or not template.get('meals'):
        return "❌ No diet data available"

    formatted_lines = []
    formatted_lines.append(f"🍽️ {template.get('template_name', 'Your Diet Template').upper()} 🍽️")
    formatted_lines.append("═" * 50)
    formatted_lines.append("")

    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0

    for meal in template['meals']:
        title = meal.get('title', 'Unknown Meal')
        time_range = meal.get('timeRange', '')
        tagline = meal.get('tagline', '')

        formatted_lines.append(f"🕐 {title.upper()} - {time_range}")
        if tagline:
            formatted_lines.append(f"   {tagline}")
        formatted_lines.append("─" * 30)
        formatted_lines.append("")

        food_list = meal.get('foodList', [])
        if food_list:
            for food in food_list:
                name = food.get('name', 'Unknown Food')
                quantity = food.get('quantity', '')
                calories = food.get('calories', 0)
                protein = food.get('protein', 0)
                carbs = food.get('carbs', 0)
                fat = food.get('fat', 0)

                formatted_lines.append(f"   🥗 {name} - {quantity}")
                formatted_lines.append(f"      📊 {calories}cal | P:{protein}g | C:{carbs}g | F:{fat}g")
                formatted_lines.append("")

                total_calories += calories
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
        else:
            formatted_lines.append("   ⚠️ No foods added yet")
            formatted_lines.append("")

        formatted_lines.append("")

    formatted_lines.append("═" * 50)
    formatted_lines.append(f"📊 DAILY TOTALS: {total_calories}cal | Protein: {total_protein}g | Carbs: {total_carbs}g | Fat: {total_fat}g")

    return "\n".join(formatted_lines)

def detect_user_intent_with_ai(text: str, oai) -> dict:
    """Use AI to detect user intent with improved accuracy and specificity"""
    
    prompt = f"""
    Analyze this user message and determine their SPECIFIC intent: "{text}"
    
    BE VERY LIBERAL with intent detection. Detect these intents:

    1. WORKOUT_PLAN: User wants a workout plan, exercise routine, or training program
       - Examples: "workout plan", "exercise routine", "training program", "fitness plan", "workout for weight loss"
       - Keywords: workout, exercise, training, fitness + plan/routine/program/template
       - BE LIBERAL: Even questions like "workout plan for beginners" should be WORKOUT_PLAN
       - Handle typos: "worutplan", "workoutplan", "excersize plan"

    2. DIET_PLAN: User wants a diet plan, meal plan, or nutrition guidance
       - Examples: "diet plan", "meal plan", "nutrition plan", "eating plan", "food plan"
       - Keywords: diet, meal, nutrition, eating, food + plan/template/guide
       - BE LIBERAL: Even questions like "what should I eat" can be DIET_PLAN if they want guidance
       - Handle typos: "deit plan", "meal templete", "mealplan"

    3. FOOD_LOGGING: User mentions eating/consuming food - BE EXTREMELY LIBERAL
       - Examples: "I ate rice", "had breakfast", "consumed apple", "ate pizza", just mentions food names
       - Key phrases: "I ate", "I had", "I consumed", "I finished", "just ate", "just had"
       - BE GENEROUS: ANY mention of eating/consuming food is likely logging intent
       - Even simple food names like "apple", "rice" should be FOOD_LOGGING

    4. WORKOUT_LOGGING: User mentions doing exercises or working out
       - Examples: "I did pushups", "completed workout", "finished training", "did exercises"
       - Key phrases: "I did", "I completed", "I finished", "just did", "done with"
       - BE LIBERAL: Any mention of doing exercises = logging intent

    5. NONE: Only for completely unrelated topics or pure information requests

    CRITICAL RULES:
    - BE EXTREMELY LIBERAL - when in doubt, choose an action intent over NONE
    - Simple mentions of food = FOOD_LOGGING
    - Any request for plans/routines = corresponding PLAN intent
    - Past tense actions = logging intents
    - Confidence should be HIGH for obvious matches
    
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
                    "content": "You are an intent classifier. BE LIBERAL - when in doubt, choose an action intent. Fitness-related requests usually want plans or logging. Be generous with confidence scores."
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

@router.get("/chat/stream_test")
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

    # Handle food quantity input (NEW INTEGRATED FUNCTIONALITY)
    if pend.get("state") == "awaiting_food_quantity":
        print("DEBUG: In awaiting_food_quantity state")
        try:
            foods = pend.get("foods", [])
            current_index = pend.get("current_food_index", 0)
            current_food = foods[current_index] if current_index < len(foods) else None
            logged_foods = pend.get("logged_foods", [])

            if not current_food:
                print("DEBUG: No current food found, clearing state")
                await mem.set_pending(user_id, None)
                async def _error_no_food():
                    yield sse_escape("Something went wrong. Please tell me what you ate again.")
                    yield "event: done\ndata: [DONE]\n\n"
                return StreamingResponse(_error_no_food(), media_type="text/event-stream",
                                       headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

            # Parse quantity from input
            quantity, unit = parse_quantity_input(text)

            if quantity is not None:
                print(f"DEBUG: Parsed quantity: {quantity} {unit} for {current_food['name']}")

                # Use default unit if none provided
                if not unit:
                    unit = current_food.get('unit', 'pieces')

                # Update food with quantity and calculate nutrition
                foods[current_index]["quantity"] = quantity
                foods[current_index]["unit"] = unit

                nutrition = calculate_nutrition_using_ai(
                    current_food['name'], quantity, unit, oai
                )
                foods[current_index].update(nutrition)

                # Move to logged foods
                logged_foods.append(foods[current_index])

                # Check for next food needing quantity
                next_food_index = -1
                for i in range(current_index + 1, len(foods)):
                    if foods[i].get("quantity") is None:
                        next_food_index = i
                        break

                if next_food_index != -1:
                    # Ask for next food quantity
                    next_food = foods[next_food_index]
                    ask_msg = f"Great! Now, how much {next_food['name']} did you have? (e.g., '2 pieces', '1 plate', '100g')"

                    await mem.set_pending(user_id, {
                        "state": "awaiting_food_quantity",
                        "foods": foods,
                        "current_food_index": next_food_index,
                        "logged_foods": logged_foods
                    })

                    async def _ask_next_quantity():
                        yield sse_escape(ask_msg)
                        yield "event: done\ndata: [DONE]\n\n"

                    return StreamingResponse(_ask_next_quantity(), media_type="text/event-stream",
                                           headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
                else:
                    # All foods processed, log everything
                    await mem.set_pending(user_id, None)

                    # Store to database
                    today_date = datetime.now(IST).strftime("%Y-%m-%d")
                    store_diet_data_to_db(db, user_id, today_date, logged_foods, "breakfast")

                    # Create summary
                    food_summaries = []
                    total_calories = 0
                    for food in logged_foods:
                        food_summaries.append(f"{food['quantity']} {food['unit']} of {food['name']}")
                        total_calories += food.get('calories', 0)

                    if len(food_summaries) == 1:
                        message = f"✅ Logged {food_summaries[0]}!"
                    else:
                        message = f"✅ Logged {', '.join(food_summaries[:-1])} and {food_summaries[-1]}!"

                    message += f"\n📊 Total: {total_calories} calories added to your food diary"

                    await mem.add(user_id, "user", text.strip())
                    await mem.add(user_id, "assistant", message)

                    async def _final_food_log():
                        yield sse_json({
                            "type": "food_logged",
                            "logged_foods": logged_foods,
                            "message": message
                        })
                        yield "event: done\ndata: [DONE]\n\n"

                    return StreamingResponse(_final_food_log(), media_type="text/event-stream",
                                           headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
            else:
                # Ask again for valid quantity
                ask_msg = f"Please enter a valid quantity for {current_food['name']}. For example: '2', '1.5 plates', or '100g'"

                async def _ask_quantity_again():
                    yield sse_escape(ask_msg)
                    yield "event: done\ndata: [DONE]\n\n"

                return StreamingResponse(_ask_quantity_again(), media_type="text/event-stream",
                                       headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        except Exception as e:
            print(f"Error processing food quantity: {e}")
            await mem.set_pending(user_id, None)
            async def _quantity_error():
                yield sse_escape("Something went wrong. Please tell me what you ate again.")
                yield "event: done\ndata: [DONE]\n\n"
            return StreamingResponse(_quantity_error(), media_type="text/event-stream",
                                   headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

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

    # First check if there are clear action indicators OR if it's a fitness-related question
    has_action = has_action_indicators(text)
    is_fitness_question = is_fitness_related(text)
    
    print(f"DEBUG Intent Check: has_action={has_action}, is_fitness={is_fitness_question}, text='{text}'")
    
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
    
    # Run AI intent detection more aggressively - check for any fitness-related content
    should_check_intent = (
        has_action or
        is_fitness_question or
        any(word in text.lower() for word in ['workout', 'exercise', 'diet', 'meal', 'plan', 'routine', 'training', 'fitness', 'nutrition']) or
        len(text.split()) <= 5  # Also check short phrases that might be plans
    )

    print(f"DEBUG: should_check_intent={should_check_intent}")

    # Run AI intent detection for action requests OR fitness questions (to catch edge cases)
    if should_check_intent:
        intent_result = detect_user_intent_with_ai(text, oai)
        intent_type = intent_result.get("intent", "NONE")
        confidence = intent_result.get("confidence", 0.0)
        action_detected = intent_result.get("action_detected", False)
        reasoning = intent_result.get("reasoning", "")
        
        # More liberal for plan creation and logging
        if intent_type in ["WORKOUT_PLAN", "DIET_PLAN"]:
            min_confidence = 0.3  # Much lower threshold for plan creation
        elif intent_type in ["FOOD_LOGGING", "WORKOUT_LOGGING"]:
            min_confidence = 0.3  # Even lower threshold for logging (since these are important)
        else:
            min_confidence = 0.5  # Lower threshold for other intents

        print(f"DEBUG AI Intent: {intent_type} (confidence: {confidence}, action: {action_detected})")
        print(f"DEBUG AI Reasoning: {reasoning}")
        print(f"DEBUG has_action: {has_action}, is_fitness: {is_fitness_question}")
        print(f"DEBUG min_confidence for {intent_type}: {min_confidence}")
        print(f"DEBUG Will trigger: {intent_type in ['WORKOUT_PLAN', 'DIET_PLAN', 'FOOD_LOGGING', 'WORKOUT_LOGGING'] and confidence > min_confidence}")

        
        # PRIORITY 1: Check for workout plan intent - HANDLE DIRECTLY
        if intent_type == "WORKOUT_PLAN" and confidence > min_confidence:
            print(f"DEBUG: Workout plan intent detected for: {text}")

            async def _generate_workout_plan():
                try:
                    await mem.add(user_id, "user", text.strip())

                    # Get user profile
                    profile = fetch_client_profile(db, user_id)
                    print(f"DEBUG: User profile fetched: {profile}")

                    # Generate workout template
                    print(f"DEBUG: Generating workout template with AI...")
                    workout_template = generate_workout_template_with_ai(text, profile, oai)
                    print(f"DEBUG: Workout template generated: {workout_template is not None}")

                    if workout_template:
                        # Format the template for display
                        formatted_template = format_workout_template_display(workout_template)

                        response_message = f"🏋️ I've created a personalized workout template for you!\n\n{formatted_template}\n\n💡 This template is based on your profile and goals. You can modify it anytime by asking me to adjust specific exercises, sets, or reps!"

                        await mem.add(user_id, "assistant", response_message)

                        # Send response with template data
                        yield sse_json({
                            "type": "workout_template",
                            "template": workout_template,
                            "message": response_message
                        })
                    else:
                        error_msg = "I encountered an issue generating your workout plan. Let me help you with general workout advice instead."
                        await mem.add(user_id, "assistant", error_msg)
                        yield sse_escape(error_msg)

                    yield "event: done\ndata: [DONE]\n\n"

                except Exception as e:
                    print(f"Error generating workout plan: {e}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
                    error_msg = "I encountered an issue generating your workout plan. Let me help you with general workout advice instead."
                    await mem.add(user_id, "assistant", error_msg)
                    yield sse_escape(error_msg)
                    yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(_generate_workout_plan(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 2: Check for diet plan intent - HANDLE DIRECTLY
        elif intent_type == "DIET_PLAN" and confidence > min_confidence:
            print(f"DEBUG: Diet plan intent detected for: {text}")

            async def _generate_diet_plan():
                try:
                    await mem.add(user_id, "user", text.strip())

                    # Get user profile
                    profile = fetch_client_profile(db, user_id)
                    print(f"DEBUG: User profile fetched for diet: {profile}")

                    # Generate diet template
                    print(f"DEBUG: Generating diet template with AI...")
                    diet_template = generate_diet_template_with_ai(text, profile, oai)
                    print(f"DEBUG: Diet template generated: {diet_template is not None}")

                    if diet_template:
                        # Format the template for display
                        formatted_template = format_diet_template_display(diet_template)

                        response_message = f"🍽️ I've created a personalized diet template for you!\n\n{formatted_template}\n\n💡 This diet plan is tailored to your goals and calorie needs. You can modify it anytime by asking me to adjust meals or portions!"

                        await mem.add(user_id, "assistant", response_message)

                        # Send response with template data
                        yield sse_json({
                            "type": "diet_template",
                            "template": diet_template,
                            "message": response_message
                        })
                    else:
                        error_msg = "I encountered an issue generating your diet plan. Let me help you with general nutrition advice instead."
                        await mem.add(user_id, "assistant", error_msg)
                        yield sse_escape(error_msg)

                    yield "event: done\ndata: [DONE]\n\n"

                except Exception as e:
                    print(f"Error generating diet plan: {e}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
                    error_msg = "I encountered an issue generating your diet plan. Let me help you with general nutrition advice instead."
                    await mem.add(user_id, "assistant", error_msg)
                    yield sse_escape(error_msg)
                    yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(_generate_diet_plan(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 3: Check for food logging intent - HANDLE DIRECTLY
        elif intent_type == "FOOD_LOGGING" and confidence > min_confidence:
            print(f"DEBUG: Food logging intent detected for: {text}")

            async def _handle_food_logging():
                try:
                    await mem.add(user_id, "user", text.strip())

                    # Extract food information
                    food_info = extract_food_info_using_ai(text, oai)
                    foods = food_info.get("foods", [])

                    if not foods:
                        error_msg = "I couldn't identify any food items from your message. Could you tell me what you ate? For example: 'I had 2 apples and a bowl of rice'"
                        await mem.add(user_id, "assistant", error_msg)
                        yield sse_escape(error_msg)
                        yield "event: done\ndata: [DONE]\n\n"
                        return

                    # Process foods - calculate nutrition for those with quantities
                    logged_foods = []
                    foods_needing_quantities = []

                    for food in foods:
                        if food.get('quantity') is not None:
                            # Calculate nutrition
                            nutrition = calculate_nutrition_using_ai(
                                food['name'], food['quantity'], food['unit'], oai
                            )
                            food.update(nutrition)
                            logged_foods.append(food)
                        else:
                            foods_needing_quantities.append(food)

                    if logged_foods:
                        # Store logged foods in database (default to "breakfast" meal)
                        today_date = datetime.now(IST).strftime("%Y-%m-%d")
                        store_diet_data_to_db(db, user_id, today_date, logged_foods, "breakfast")

                        # Create summary
                        food_summaries = []
                        total_calories = 0
                        for food in logged_foods:
                            quantity = food.get('quantity', 0)
                            unit = food.get('unit', 'pieces')
                            name = food.get('name', '')
                            calories = food.get('calories', 0)

                            food_summaries.append(f"{quantity} {unit} of {name}")
                            total_calories += calories

                        if len(food_summaries) == 1:
                            message = f"✅ Logged {food_summaries[0]}!"
                        else:
                            message = f"✅ Logged {', '.join(food_summaries[:-1])} and {food_summaries[-1]}!"

                        message += f"\n📊 Total: {total_calories} calories added to your food diary"

                        if foods_needing_quantities:
                            pending_names = [f['name'] for f in foods_needing_quantities]
                            message += f"\n\n❓ I also noticed: {', '.join(pending_names)}. Would you like to log these too? If yes, please tell me the quantities."

                        await mem.add(user_id, "assistant", message)

                        yield sse_json({
                            "type": "food_logged",
                            "logged_foods": logged_foods,
                            "pending_foods": foods_needing_quantities,
                            "message": message
                        })
                    else:
                        # All foods need quantities
                        first_food = foods_needing_quantities[0]
                        ask_msg = f"Great! I found {first_food['name']}. How much did you have? (e.g., '2 pieces', '1 plate', '100g')"

                        await mem.set_pending(user_id, {
                            "state": "awaiting_food_quantity",
                            "foods": foods_needing_quantities,
                            "current_food_index": 0,
                            "logged_foods": []
                        })

                        await mem.add(user_id, "assistant", ask_msg)
                        yield sse_escape(ask_msg)

                    yield "event: done\ndata: [DONE]\n\n"

                except Exception as e:
                    print(f"Error handling food logging: {e}")
                    error_msg = "I encountered an issue logging your food. Please try again or tell me what you ate in a different way."
                    await mem.add(user_id, "assistant", error_msg)
                    yield sse_escape(error_msg)
                    yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(_handle_food_logging(), media_type="text/event-stream",
                                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

        # PRIORITY 4: Check for workout logging intent - HANDLE DIRECTLY
        elif intent_type == "WORKOUT_LOGGING" and confidence > min_confidence:
            print(f"DEBUG: Workout logging intent detected for: {text}")

            async def _handle_workout_logging():
                try:
                    await mem.add(user_id, "user", text.strip())

                    # Extract workout information using AI
                    workout_info = extract_workout_info_using_ai(text, oai)
                    exercises = workout_info.get("exercises", [])

                    if not exercises:
                        error_msg = "I couldn't identify any exercises from your message. Could you tell me what exercises you did? For example: 'I did 3 sets of 10 pushups and ran for 30 minutes'"
                        await mem.add(user_id, "assistant", error_msg)
                        yield sse_escape(error_msg)
                        yield "event: done\ndata: [DONE]\n\n"
                        return

                    # Create workout summary
                    exercise_summaries = []
                    for exercise in exercises:
                        name = exercise.get('name', 'Unknown Exercise')
                        sets = exercise.get('sets', '')
                        reps = exercise.get('reps', '')
                        duration = exercise.get('duration', '')

                        if sets and reps:
                            exercise_summaries.append(f"{name}: {sets} sets × {reps} reps")
                        elif duration:
                            exercise_summaries.append(f"{name}: {duration}")
                        else:
                            exercise_summaries.append(name)

                    message = f"🏋️ Great workout! I logged:\n\n"
                    for i, summary in enumerate(exercise_summaries, 1):
                        message += f"{i}. {summary}\n"

                    message += f"\n💪 Keep up the excellent work! Remember to stay hydrated and get proper rest for recovery."

                    await mem.add(user_id, "assistant", message)

                    yield sse_json({
                        "type": "workout_logged",
                        "exercises": exercises,
                        "message": message
                    })

                    yield "event: done\ndata: [DONE]\n\n"

                except Exception as e:
                    print(f"Error handling workout logging: {e}")
                    error_msg = "I encountered an issue logging your workout. Please try again or describe your exercises differently."
                    await mem.add(user_id, "assistant", error_msg)
                    yield sse_escape(error_msg)
                    yield "event: done\ndata: [DONE]\n\n"

            return StreamingResponse(_handle_workout_logging(), media_type="text/event-stream",
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
