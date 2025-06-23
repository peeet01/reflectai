import streamlit as st
from datetime import datetime

from modules.reflection_modul import run as run_reflection
from modules.questions import load_questions, get_random_question

st.set_page_config(
    page_title="Neurolab AI – Scientific Playground Sandbox",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neurolab AI – Scientific Playground Sandbox")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
st.text_input("📝 Megfigyelés vagy jegyzet (opcionális):")

st.sidebar.title("📂 Modulválasztó")
module_name = st.sidebar.radio("Kérlek válassz:", (
    "🧠 Napi önreflexió",
))

if module_name == "🧠 Napi önreflexió":
    run_reflection()
