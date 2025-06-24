# modules/modules_registry.py
import importlib
import streamlit as st

# Kulcs: modul neve a UI-ban, Érték: modulpath és futtató függvény
MODULES = {
    "Kuramoto szinkronizáció": ("kuramoto_sim", "run"),
    "Hebbian tanulás": ("hebbian_learning", "run"),
    "XOR predikció": ("xor_prediction", "run"),
    "Kuramoto–Hebbian háló": ("kuramoto_hebbian_sim", "run"),
    "Topológiai szinkronizáció": ("topo_protect", "run"),
    "Lorenz szimuláció": ("lorenz_sim", "run"),
    "Lorenz predikció": ("predict_lorenz", "run"),
    "Topológiai védettség (Chern-szám)": ("berry_curvature", "run"),
    "Topológiai Chern–szám analízis": ("berry_curvature", "run"),
    "Zajtűrés és szinkronizációs robusztusság": ("noise_robustness", "run"),
    "Echo State Network (ESN) predikció": ("esn_prediction", "run"),
    "Hebbian plaszticitás dinamikája": ("plasticity_dynamics", "run"),
    "Szinkronfraktál dimenzióanalízis": ("fractal_dimension", "run"),
    "Belátás alapú tanulás (Insight Learning)": ("insight_learning", "run"),
    "Generatív Kuramoto hálózat": ("generative_kuramoto", "run"),
    "Memória tájkép (Pro)": ("memory_landscape", "run"),
    "Gráfalapú szinkronanalízis": ("graph_sync_analysis", "run"),
    "Perzisztens homológia": ("persistent_homology", "run"),
    "Lyapunov spektrum": ("lyapunov_spectrum", "run"),
    "Adatfeltöltés modul": ("data_upload", "run"),
    "Napi önreflexió": ("reflection_modul", "run"),
    "Súgó / Help": ("help_module", "run"),
}

def safe_run(module_key):
    try:
        module_name, function_name = MODULES[module_key]
        module = importlib.import_module(f"modules.{module_name}")
        getattr(module, function_name)()
    except Exception as e:
        st.error(f"❌ [Hiba a(z) {module_key} modulban] {e}")
