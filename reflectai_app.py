import streamlit as st

# Modulok importálása
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.kuramoto_hebbian import run as run_kuramoto_hebbian
from modules.topo_protect import run as run_topo_protect
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_lorenz_pred
from modules.berry_curvature import run as run_berry_curvature
from modules.noise_robustness import run as run_noise_robust
from modules.esn_prediction import run as run_esn
from modules.plasticity_dynamics import run as run_plasticity
from modules.fractal_dimension import run as run_fractal_dimension

# Streamlit oldal beállítás
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI – Kvázi-tudati MI szimulációk")

# Oldalsáv – modulválasztó
st.sidebar.title("📁 Modulválasztó")
module_name = st.sidebar.radio("Válassz modult:", (
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto–Hebbian háló",
    "Topológiai szinkronizáció",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "Topológiai védettség (Chern-szám)",
    "Topológiai Chern–szám analízis",
    "Zajtűrés és szinkronizációs robusztusság",
    "Echo State Network (ESN) predikció",
    "Plaszticitás dinamikája",
    "Fraktáldimenzió analízis"
))

# Feltételes modulfuttatás
if module_name == "Kuramoto szinkronizáció":
    run_kuramoto()

elif module_name == "Hebbian tanulás":
    run_hebbian()

elif module_name == "XOR predikció":
    run_xor()

elif module_name == "Kuramoto–Hebbian háló":
    run_kuramoto_hebbian()

elif module_name == "Topológiai szinkronizáció":
    run_topo_protect()

elif module_name == "Lorenz szimuláció":
    run_lorenz_sim()

elif module_name == "Lorenz predikció":
    run_lorenz_pred()

elif module_name == "Topológiai védettség (Chern-szám)":
    run_berry_curvature()

elif module_name == "Topológiai Chern–szám analízis":
    run_berry_curvature()

elif module_name == "Zajtűrés és szinkronizációs robusztusság":
    run_noise_robust()

elif module_name == "Echo State Network
