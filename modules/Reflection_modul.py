import streamlit as st
import json
import os
import random
from datetime import datetime

# Kérdések betöltése JSON-ből
def load_questions(filepath="data/questions.json"):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Véletlenszerű kérdés választása
def get_random_question(questions):
    if not questions:
        return None
    return random.choice(questions)

# Fő futtató függvény (Streamlit modulhoz)
def run():
    st.markdown("## 🧠 Napi önreflexió")

    questions = load_questions()
    question = get_random_question(questions)

    if not question:
        st.warning("⚠️ Nem található kérdés a kérdésbankban.")
        return

    st.markdown("### 🤔 Ma elgondolkodhatsz ezen:")
    st.markdown(f"**{question['text']}**")

    response = st.text_area("✏️ Válaszod:", height=150)

    if st.button("✅ Válasz rögzítése"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            "id": question.get("id"),
            "theme": question.get("theme"),
            "level": question.get("level"),
            "question": question.get("text"),
            "response": response,
            "timestamp": timestamp
        }

        st.success("A válaszod rögzítve lett.")
        st.json(result)

        # Opcionális: válasz naplózása fájlba
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "responses.json")

        try:
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(result)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            st.error(f"Hiba a mentés közben: {e}")
