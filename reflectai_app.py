import streamlit as st

# Modulok importálása
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.graph_sync_analysis import run as run_graph
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.lorenz_sim import run as run_lorenz
from modules.predict_lorenz import run as run_lorenz_pred
from modules.topo_protect import run as run_topo_protect
from modules.berry_curvature import run as run_berry
from modules.noise_robustness import run as run_noise
from modules.esn_prediction import run as run_esn
from modules.plasticity_dynamics import run as run_plasticity  # ⬅️ ÚJ

# App beállítás
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI – Tudományos MI szimulátor")

# Kérdésdoboz
user_input = st.text_input("💬 Kérdésed, megjegyzésed vagy kutatási parancsod:")
if user_input:
    st.info(f"🔍 Ezt írtad be: **{user_input}**")
    st.markdown("> A rendszer jelenleg nem generál választ, de a bemenet rögzítésre került.")

# Modulválasztó menü
page = st.sidebar.radio("📂 Modulválasztó", [
    "Kuramoto
