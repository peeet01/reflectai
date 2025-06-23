
import streamlit as st
import json
import os
import random

def load_questions(filepath="data/questions.json"):
    if not os.path.exists(filepath):
        st.warning("A kérdésfájl nem található.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_random_reflection_question(questions):
    reflection_qs = [q for q in questions if q.get("theme") == "önreflexió"]
    if not reflection_qs:
        return None
    return random.choice(reflection_qs)

def run():
    st.header("🧭 Napi önreflexió")

    questions = load_questions()
    if not questions:
        st.error("Nincs betölthető kérdés.")
        return

    question = get_random_reflection_question(questions)
    if not question:
        st.warning("Nincs 'önreflexió' témájú kérdés a listában.")
        return

    st.subheader("Kérdés:")
    st.write(f"**{question['text']}**")

    answer = st.text_area("Válaszod:", height=150)

    if st.button("Válasz mentése"):
        st.success("✅ Válasz elmentve (vagy legalábbis elképzeltük).")
