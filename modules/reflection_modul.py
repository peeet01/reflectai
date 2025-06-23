import streamlit as st
import json
import random
import os

def run():
    st.title("🪞 Napi önreflexió")

    json_path = os.path.join("data", "questions.json")
    if not os.path.exists(json_path):
        st.error("A kérdésfájl (questions.json) nem található.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Csak az "önreflexió" témájú kérdések
    filtered = [q for q in questions if q["theme"].lower() == "önreflexió"]
    if not filtered:
        st.warning("Nincs elérhető önreflexiós kérdés.")
        return

    q = random.choice(filtered)
    st.subheader("Mai kérdésed:")
    st.markdown(f"**{q['text']}**")

    response = st.text_area("Válaszod:")
    if st.button("Mentés"):
        if response.strip():
            st.success("Válaszod mentve. Köszönjük!")
        else:
            st.warning("Kérlek, írj be valamit előbb.")
