import streamlit as st
from datetime import datetime
from modules.questions import load_questions, get_random_question

def run():
    st.header("🧠 Napi önreflexió")

    questions = load_questions()
    question = get_random_question(questions)

    if not question:
        st.warning("⚠️ Nem található kérdés a kérdésbankban.")
        return

    st.subheader("🤔 Mai kérdés")
    st.markdown(f"**{question['text']}**")

    response = st.text_area("✏️ Válaszod:", height=150)

    if st.button("✅ Válasz rögzítése"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("A válaszod rögzítve lett!")

        # Későbbi mentéshez megjelenítjük JSON-ben is:
        st.json({
            "id": question.get("id"),
            "theme": question.get("theme"),
            "level": question.get("level"),
            "question": question.get("text"),
            "response": response,
            "timestamp": timestamp
        })
