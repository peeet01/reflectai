import streamlit as st

# Modulok importálása
from modules.kuramoto_sim import run as run_kuramoto_lstm
from modules.hebbian_learning import run as run_hebbian_learning
from modules.xor_prediction import run as run_xor
from modules.kuramoto_hebbian_net import run as run_kuramoto_hebbian
from modules.graph_sync_analysis import run as run_topo_sync
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_lorenz_pred
from modules.topo_protect import run as run_topo_chern
from modules.berry_curvature import run as run_berry
from modules.noise_robustness import run as run_noise
from modules.esn_prediction import run as run_esn
from modules.plasticity_dynamics import run as run_plasticity
from modules.fractal_dimension import run as run_fractal_dimension  # ÚJ MODUL ✅

# Oldalsáv – Modulválasztó
st.sidebar.title("📁 Modulválasztó")

modulok = {
    "Kuramoto szinkronizáció": run_kuramoto_lstm,
    "Hebbian tanulás": run_hebbian_learning,
    "XOR predikció": run_xor,
    "Kuramoto–Hebbian háló": run_kuramoto_hebbian,
    "Topológiai szinkronizáció": run_topo_sync,
    "Lorenz szimuláció": run_lorenz_sim,
    "Lorenz predikció": run_lorenz_pred,
    "Topológiai védettség (Chern-szám)": run_topo_chern,
    "Topológiai Chern–szám analízis": run_berry,
    "Zajűrés és szinkronizációs robusztusság": run_noise,
    "Echo State Network (ESN) predikció": run_esn,
    "Plaszticitás szimuláció": run_plasticity,
    "Fraktáldimenzió analízis": run_fractal_dimension  # ÚJ MODUL ✅
}

modul_valasztas = st.sidebar.radio("Válassz modult:", list(modulok.keys()))

# Modul futtatása
if modul_valasztas in modulok:
    modulok[modul_valasztas]()
else:
    st.warning("Nincs érvényes modul kiválasztva.")
