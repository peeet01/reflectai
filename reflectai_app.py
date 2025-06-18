import streamlit as st
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian

st.set_page_config(page_title="ReflectAI Pro – MI szimulációk", layout="wide")
st.title("🧠 ReflectAI Pro – Tudományos MI szimulációk")

# Menüpont választó
page = st.sidebar.radio("📂 Válassz modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás"
])

# Modulhívások
if page == "Kuramoto szinkronizáció":
    run_kuramoto()

elif page == "Hebbian tanulás":
    run_hebbian()
