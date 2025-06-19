import streamlit as st
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.kuramoto_hebbian_net import run as run_kuramoto_hebbian
from modules.topo_protect import run as run_topo
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_lorenz_pred
from modules.berry_curvature import run as run_berry
from modules.esn_prediction import run as run_esn
from modules.noise_robustness import run as run_noise
from modules.plasticity_dynamics import run as run_plasticity
from modules.fractal_dimension import run as run_fractal

# Oldalbeállítások
st.set_page_config(page_title="ReflectAI", layout="wide")

# Oldalsáv – modulválasztó
st.sidebar.title("📁 Modulválasztó")

module_name = st.sidebar.radio(
    "Válassz egy modult:",
    [
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
        "Hebbian plas
