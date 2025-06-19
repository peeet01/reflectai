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
from modules.berry_curvature import run as run_berry_curvature
from modules.esn_prediction import run as run_esn
from modules.noise_robustness import run as run_noise_robustness  # 🆕

# App beállítás
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI – Tudományos MI szimulátor")

# Felhasználói kérdés
user_input = st.text_input("💬 Kérdésed, megjegyzésed vagy kutatási parancsod:")
if user_input:
    st.info(f"🔍 Ezt írtad be: **{user_input}**")
    st.markdown("> A rendszer jelenleg nem generál választ, de a bemenet rögzítésre került.")

# Modulválasztó
page = st.sidebar.radio("📂 Modulválasztó", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto–Hebbian háló",
    "Topológiai szinkronizáció",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "Topológiai védettség (Chern-szám)",
    "Topológiai Chern–szám analízis",
    "Zajtűrés és szinkronizációs robusztusság",  # 🆕
    "Echo State Network (ESN) predikció"
])

# Modulok meghívása
if page == "Kuramoto szinkronizáció":
    run_kuramoto()
elif page == "Hebbian tanulás":
    run_hebbian()
elif page == "XOR predikció":
    run_xor()
elif page == "Kuramoto–Hebbian háló":
    run_kuramoto_hebbian()
elif page == "Topológiai szinkronizáció":
    run_graph()
elif page == "Lorenz szimuláció":
    run_lorenz()
elif page == "Lorenz predikció":
    run_lorenz_pred()
elif page == "Topológiai védettség (Chern-szám)":
    run_topo_protect()
elif page == "Topológiai Chern–szám analízis":
    run_berry_curvature()
elif page == "Zajtűrés és szinkronizációs robusztusság":
    run_noise_robustness()
elif page == "Echo State Network (ESN) predikció":
    run_esn()
