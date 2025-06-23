import streamlit as st
import os
import sys
from datetime import datetime

# Biztosítjuk, hogy a 'modules' mappa elérhető legyen importhoz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from modules.questions import load_questions, get_random_question

def run():
    questions = load_questions("data/questions.json")
    question = get_random_question(questions)

    if question:
        st.markdown("### 🤔 Napi önreflexiós kérdés")
        st.markdown(f"**{question['text']}**")
        response = st.text_area("✏️ Válaszod:", height=150)
        if st.button("✅ Válasz rögzítése"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ A válaszod rögzítve lett.")
            st.json({
                "id": question.get("id"),
                "theme": question.get("theme"),
                "level": question.get("level"),
                "question": question.get("text"),
                "response": response,
                "timestamp": timestamp
            })
    else:
        st.warning("⚠️ Nem található kérdés a kérdésbankban.")
