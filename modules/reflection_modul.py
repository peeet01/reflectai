import streamlit as st
from datetime import datetime

# Védetten próbáljuk betölteni a kérdéseket
def load_questions(filepath="data/questions.json"):
    import json, os
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Hiba a kérdések betöltésekor: {e}")
        return []

def get_random_question(questions):
    import random
    if not questions:
        return None
    return random.choice(questions)

def run():
    st.markdown("### Napi önreflexiós kérdés")

    try:
        questions = load_questions("data/questions.json")
        question = get_random_question(questions)

        if question:
            st.markdown(f"**{question['text']}**")
            response = st.text_area("Válaszod:", height=150)

            if st.button("Válasz rögzítése"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("A válaszod rögzítve lett.")
                st.json({
                    "id": question.get("id"),
                    "theme": question.get("theme"),
                    "level": question.get("level"),
                    "question": question.get("text"),
                    "response": response,
                    "timestamp": timestamp
                })
        else:
            st.warning("Nem található kérdés a kérdésbankban.")
    except Exception as e:
        st.error(f"Hiba az önreflexiós modul futása során: {e}")
