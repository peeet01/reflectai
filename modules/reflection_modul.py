import streamlit as st
from modules.questions import load_questions, get_random_question
from datetime import datetime

def run():
    questions = load_questions("data/questions.json")
    question = get_random_question(questions)

    if question:
        st.markdown("### 🤔 Napi önreflexiós kérdés")
        st.markdown(f"**{question['text']}**")
        response = st.text_area("✏️ Válaszod:", height=150)
        if st.button("✅ Válasz rögzítése"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ A válaszod ideiglenesen rögzítve lett.")
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
