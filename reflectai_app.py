import streamlit as st

# 🔹 Modulok importálása
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.graph_sync_analysis import run as run_graph
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.lorenz_sim import run as run_lorenz
from modules.predict_lorenz import run as run_lorenz_pred
from modules.berry_curvature import run as run_berry
from modules.noise_resilience import run as run_noise
from modules.plasticity_dynamics import run as run_plasticity  # 🔺 ÚJ MODUL

# 🔸 App konfiguráció
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI – Tudományos MI szimulátor")

# 🔸 Kérdésfeltevés
user_input = st.text_input("💬 Kérdésed, megjegyzésed vagy kutatási parancsod:")
if user_input:
    st.info(f"🔍 Ezt írtad be: **{user_input}**")
    st.markdown("> A rendszer jelenleg nem generál választ, de a bemenet rögzítésre került.")

# 🔸 Modulválasztó
page = st.sidebar.radio("📂 Modulválasztó", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto–Hebbian háló",
    "Topológiai szinkronizáció",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "Berry-görbület analízis",
    "Zajrezisztencia vizsgálat",
    "Plaszticitás-dinamika
