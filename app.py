# app.py
# Full app: Manual Entry, Image Upload, Diet Plan, Ask AI
# Uses dotenv for GOOGLE_API_KEY and falls back to Streamlit secrets if available.

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline
from rapidfuzz import process, fuzz
import google.generativeai as genai

# ----------------------------
# Streamlit page config (must be first Streamlit call)
# ----------------------------
st.set_page_config(page_title="Indian Food Calorie Tracker", layout="centered")

# ----------------------------
# GEMINI / API key setup (dotenv first, then secrets)
# ----------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # Try Streamlit secrets if .env didn't provide it (safe access)
    try:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        API_KEY = None

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    # Inform user but do not stop app; AI features will warn if used.
    st.info("⚠️ GOOGLE_API_KEY not found. AI features will not work locally until you set GOOGLE_API_KEY in .env or Streamlit secrets.")

# ----------------------------
# Load Nutrition DB
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("indian_food_with_nutrition_v4.csv")  # replace path if different
    # normalize column names to lowercase/strip spaces
    df.columns = df.columns.str.strip().str.lower()
    # support datasets that use 'name' instead of 'food'
    if "food" not in df.columns and "name" in df.columns:
        df.rename(columns={"name": "food"}, inplace=True)
    return df

df = load_data()

# ----------------------------
# Helper: fuzzy lookup for nutrition
# ----------------------------
def get_food_nutrition(food_name, df, qty=1):
    """Return nutrition dict and score if match found, else (None, score)."""
    choices = df["food"].tolist()
    match, score, idx = process.extractOne(food_name, choices, scorer=fuzz.WRatio)
    if score and score > 60:
        row = df.iloc[idx]
        return {
            "food": match,
            "quantity": qty,
            "calories": float(row.get("calories", 0)) * qty,
            "protein": float(row.get("protein_g", 0)) * qty,
            "fat": float(row.get("fat_g", 0)) * qty,
            "carbs": float(row.get("carbs_g", 0)) * qty
        }, score
    else:
        return None, score

# ----------------------------
# Helper: BMI / BMR / activity multiplier
# ----------------------------
def calculate_bmi(weight, height):
    return round(weight / ((height/100)**2), 2)

def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

def activity_multiplier(level):
    return {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "Active": 1.725,
        "Very Active": 1.9
    }[level]

# ----------------------------
# Load HF classifier (cached)
# ----------------------------
@st.cache_resource
def load_model():
    try:
        return pipeline("image-classification", model="NOTGOD6000/finetuned-indian-food")
    except Exception as e:
        # if transformers not installed or model cannot be loaded, return None
        st.warning(f"Model load warning: {e}")
        return None

classifier = load_model()

# ----------------------------
# Helper: Gemini wrappers
# ----------------------------
def get_gemini_response_text(prompt, image_parts=None):
    """Return Gemini text response; safe handling if API key missing."""
    if not API_KEY:
        return "⚠️ AI not configured. Set GOOGLE_API_KEY to enable AI responses."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        if image_parts:
            resp = model.generate_content([prompt, image_parts[0]])
        else:
            resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

def prepare_image_parts(uploaded_file):
    """Return Gemini-compatible image parts or None."""
    if uploaded_file is None:
        return None
    return [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]

# ----------------------------
# Session state: food_log
# ----------------------------
if "food_log" not in st.session_state:
    st.session_state.food_log = []

# ----------------------------
# Sidebar: user profile (BMR/BMI)
# ----------------------------
st.sidebar.header("User Profile")
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)
height = st.sidebar.number_input("Height (cm)", 140, 210, 170)
age = st.sidebar.number_input("Age", 10, 80, 25)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
activity = st.sidebar.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])

bmi = calculate_bmi(weight, height)
bmr = calculate_bmr(weight, height, age, gender)
daily_needs = round(bmr * activity_multiplier(activity), 2)

st.sidebar.markdown(f"**BMI:** {bmi}")
st.sidebar.markdown(f"**Daily Calorie Needs:** {daily_needs} kcal")

# ----------------------------
# Main tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Manual Entry", "Upload Image", "Diet Plan", "Ask AI"])

# --------- TAB 1: Manual Entry ---------
with tab1:
    st.header("🍲 Manual Food Entry")
    manual_input = st.text_input("E.g. 2 Chapati and 1 Dal", key="manual_input")
    # Always show "Not satisfied? Ask AI" single button
    if st.button("🤖 Not satisfied? Ask AI (open Ask AI tab)"):
        # set flag to show Ask AI tab content; we don't auto-redirect tabs (Streamlit doesn't provide direct tab switching)
        st.session_state.get("ask_ai_focus", True)
        st.info("Open the Ask AI tab and paste the food name or upload the image.")

    if manual_input:
        # Split by " and " and parse quantities
        added_something = False
        for item in manual_input.split(" and "):
            item = item.strip()
            if not item:
                continue
            qty = 1
            parts = item.split()
            if parts and parts[0].isdigit():
                qty = int(parts[0])
                food_text = " ".join(parts[1:])
            else:
                food_text = item

            nutrition, score = get_food_nutrition(food_text, df, qty)
            if nutrition:
                st.write(f"{qty} x {nutrition['food']}  — match {score:.0f}%")
                if st.button(f"Add {qty} x {nutrition['food']} to Log", key=f"manual_add_{food_text}"):
                    # append a copy to avoid accidental mutation
                    st.session_state.food_log.append(dict(nutrition))
                    st.success(f"Added {nutrition['food']} x {qty} to log")
                    added_something = True
            else:
                st.warning(f"⚠️ '{food_text}' not found in database. Use Ask AI tab for help.")

    # Optional food summary in this tab
    if st.checkbox("📊 Show Food Summary & Log", key="show_summary_manual"):
        if st.session_state.food_log:
            st.subheader("Today's Food Log")
            st.dataframe(pd.DataFrame(st.session_state.food_log), use_container_width=True)
        else:
            st.info("Food log is empty.")

# --------- TAB 2: Upload Image ---------
with tab2:
    st.header("📸 Upload Food Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="upload_image")

    top_predicted = None
    preds = None

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        except Exception:
            st.error("Unable to open image.")

        # Run classifier if available
        if classifier is not None:
            try:
                preds = classifier(img)
                st.subheader("Predicted Foods (top 3)")
                for i, p in enumerate(preds[:3]):
                    st.write(f"{i+1}. {p['label']} — {p['score']*100:.1f}%")
                top_predicted = preds[0]["label"]
            except Exception as e:
                st.warning(f"Classifier error: {e}")
        else:
            st.info("Image classifier not available in this environment.")

        # Allow user to add a quantity and add selected prediction to log
        if top_predicted:
            nutrition, score = get_food_nutrition(top_predicted, df, qty=1)
            if nutrition:
                qty_for_add = st.number_input(f"Quantity for '{nutrition['food']}'", min_value=1, value=1, key="img_qty")
                if st.button("➕ Add predicted item to Log"):
                    new_item = dict(nutrition)
                    new_item["quantity"] = qty_for_add
                    # scale values based on qty
                    new_item["calories"] = new_item.get("calories", 0) * qty_for_add
                    new_item["protein"] = new_item.get("protein", 0) * qty_for_add
                    new_item["fat"] = new_item.get("fat", 0) * qty_for_add
                    new_item["carbs"] = new_item.get("carbs", 0) * qty_for_add
                    st.session_state.food_log.append(new_item)
                    st.success(f"Added {new_item['food']} x {qty_for_add} to log.")
            else:
                st.warning("Predicted food not found in nutrition DB.")

        # ONE Ask AI fallback button (outside loop)
        if st.button("🤖 Not satisfied? Ask AI (Image)"):
            st.session_state.get("ask_ai_focus", True)
            st.info("Open the Ask AI tab and upload the same image or paste the food name for AI analysis.")

    # Optional food summary in this tab
    if st.checkbox("📊 Show Food Summary & Log", key="show_summary_image"):
        if st.session_state.food_log:
            st.subheader("Today's Food Log")
            st.dataframe(pd.DataFrame(st.session_state.food_log), use_container_width=True)
        else:
            st.info("Food log is empty.")

# --------- TAB 3: Diet Plan (ordering changed per your request) ---------
with tab3:
    st.header("🥗 Diet Plan")
    # ORDER — Meals per day → Preferences → Health issues → Suggestions → Duration
    meals = st.number_input("Meals per day", min_value=2, max_value=8, value=3)
    preferences = st.selectbox("Preferences", ["No preference", "Vegetarian", "Non-Vegetarian", "Vegan"])
    health_issues = st.text_input("Health issues / goals (comma separated, optional)", "")
    user_suggestions = st.text_area("Add your own suggestions / constraints (optional)", "")
    duration = st.selectbox("Plan duration", ["7 Days", "14 Days", "30 Days"])

    st.write("Daily calorie target used from sidebar (BMR × activity). You can override below if needed.")
    cal_override = st.number_input("Daily Calorie Need (kcal) — override", min_value=800, max_value=5000, value=int(daily_needs))

    if st.button("Generate Diet Plan"):
        if not API_KEY:
            st.warning("⚠️ AI key not configured — cannot generate diet plan. Add GOOGLE_API_KEY to enable.")
        else:
            prompt = f"""
You are a registered nutritionist. Create a {duration} diet plan with {meals} meals per day.
Preferences: {preferences}
Health issues/goals: {health_issues or 'None'}
User suggestions: {user_suggestions or 'None'}
Daily calorie target: {cal_override} kcal (±10%).
Requirements:
1) Produce Day 1 .. Day N (N = {duration.split()[0]}) with exactly {meals} meals each day.
2) For each meal list foods (Indian-friendly options), approximate calories per meal, and a simple portion size.
3) Include a macronutrient split per day (approx % for protein/carbs/fats).
4) Avoid repeating the same main dish more than twice in the whole plan.
5) Add a brief health summary and tips at the end.
"""
            plan_text = get_gemini_response_text(prompt)
            st.subheader("Diet Plan (AI)")
            st.write(plan_text)

# --------- TAB 4: Ask AI ---------
with tab4:
    st.header("🤖 Ask AI")
    st.write("Ask about a food, nutrition, macros/micros, benefits or upload an image for identification.")

    # Image upload option (for Ask AI)
    ask_image = st.file_uploader("Upload an image for AI analysis (optional)", type=["jpg", "jpeg", "png"], key="askai_img")
    if ask_image:
        try:
            img_ask = Image.open(ask_image)
            st.image(img_ask, caption="Uploaded Image", use_column_width=True)
        except Exception:
            st.error("Could not open image.")

    # Analyze uploaded image (Ask AI)
    if st.button("🔎 Analyze Uploaded Food"):
        if not ask_image:
            st.warning("⚠️ Please upload an image first.")
        else:
            image_parts = prepare_image_parts(ask_image)
            prompt_img = """
Identify the food(s) in this image. For each food item, provide:
1) Nutrition per serving (Calories, Protein, Fat, Carbs).
2) Key micronutrients if available (iron, vitamin C, etc).
3) Whether it's healthy / moderately healthy / unhealthy and why.
4) Short suggestions for healthier alternatives.
"""
            resp_text = get_gemini_response_text(prompt_img, image_parts=image_parts)
            st.write(resp_text)

            # Offer to map predicted text to CSV (optional) — user asked to hold log features so we don't auto-add.

    st.divider()

    # Text query input
    query = st.text_input("Ask about food or other queries", key="askai_text")
    if st.button("🔎 Analyze"):
        if not query or query.strip() == "":
            st.warning("⚠️ Please enter a question or food name.")
        else:
            prompt_text = f"""
For the query: '{query}', provide:
- Nutrition (Calories, Protein, Fat, Carbs) per serving if applicable.
- Key micronutrients (if known).
- Health benefits and risks.
- Whether it's healthy, moderately healthy, or unhealthy and why.
- Short suggestions / alternatives.
"""
            answer = get_gemini_response_text(prompt_text)
            st.write(answer)

    # Optional food summary & log in Ask AI tab
    if st.checkbox("📊 Show Food Summary & Log", key="show_summary_askai"):
        if st.session_state.food_log:
            st.subheader("Today's Food Log")
            st.dataframe(pd.DataFrame(st.session_state.food_log), use_container_width=True)
        else:
            st.info("Food log is empty.")

# ----------------------------
# Bottom: Global Food Summary area (optional across app — always hidden unless checkboxes used above)
# ----------------------------
st.caption("⚠️ Nutrition values are approximate and may vary depending on serving size (e.g., 100g, 1 piece, 1 cup) and recipe.")

