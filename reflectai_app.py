reflectai_app.py – 22 modul indító app, kutatási napló és authentikáció nélkül

import streamlit as st from datetime import datetime

from modules.kuramoto_sim import run as run_kuramoto from modules.hebbian_learning import run as run_hebbian from modules.xor_prediction import run as run_xor from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian from modules.topo_protect import run as run_topo_protect from modules.lorenz_sim import run as run_lorenz_sim from modules.mlp_predict_lorenz import run as run_lorenz_pred from modules.berry_curvature import run as run_berry from modules.noise_robustness import run as run_noise from modules.esn_prediction import run as run_esn from modules.plasticity_dynamics import run as run_plasticity from modules.fractal_dimension import run as run_fractal from modules.memory_landscape import run as run_memory_landscape from modules.graph_sync_analysis import run as run_graph_sync_analysis from modules.persistent_homology import run as run_homology from modules.lyapunov_spectrum import run as run_lyapunov_spectrum from modules.insight_learning import run as run_insight_learning from modules.generative_kuramoto import run as run_generative_kuramoto from modules.reflection_modul import run as run_reflection from modules.data_upload import run as run_data_upload from modules.help_module import run as run_help

Streamlit oldalbeállítás

st.set_page_config(page_title="ReflectAI – Tudományos Sandbox", layout="wide") st.title("ReflectAI – Interaktív kutatási modulgyűjtemény") st.markdown("Válassz egy modult a bal oldali menüből a futtatáshoz.")

Modulválasztó

modules = { "Kuramoto szinkronizáció": run_kuramoto, "Hebbian tanulás": run_hebbian, "XOR predikció": run_xor, "Kuramoto–Hebbian háló": run_kuramoto_hebbian, "Topológiai szinkronizáció": run_topo_protect, "Lorenz szimuláció": run_lorenz_sim, "Lorenz predikció (MLP)": run_lorenz_pred, "Chern-szám vizsgálat": run_berry, "Zajtűrés": run_noise, "ESN predikció": run_esn, "Plaszticitás dinamikája": run_plasticity, "Fraktál dimenzió": run_fractal, "Memória tájkép": run_memory_landscape, "Gráf szinkron analízis": run_graph_sync_analysis, "Perzisztens homológia": run_homology, "Lyapunov spektrum": run_lyapunov_spectrum, "Belátás alapú tanulás": run_insight_learning, "Generatív Kuramoto": run_generative_kuramoto, "Reflexió modul": run_reflection, "Adatfeltöltés": run_data_upload, "Súgó / Help": run_help }

Felhasználói választás

choice = st.sidebar.selectbox("Választható modulok:", list(modules.keys()))

Modul futtatása

moduleschoice

