# reflectai_app.py - ReflectAI: 22 modulos kutatóalkalmazás

import streamlit as st
from datetime import datetime
import importlib

# Modul-regiszter: kulcs = modulnév (oldalsáv), érték = modulfájl név (py fájl név .py nélkül)
modules = {
    "Berry Curvature": "berry_curvature",
    "Data Upload": "data_upload",
    "ESN Prediction": "esn_prediction",
    "Fractal Dimension": "fractal_dimension",
    "Generative Kuramoto": "generative_kuramoto",
    "Graph Sync Analysis": "graph_sync_analysis",
    "Hebbian Learning": "hebbian_learning",
    "Hebbian Learning Viz": "hebbian_learning_viz",
    "Help": "help_module",
    "Insight Learning": "insight_learning",
    "Kuramoto Hebbian Sim": "kuramoto_hebbian_sim",
    "Kuramoto Sim": "kuramoto_sim",
    "Lorenz Sim": "lorenz_sim",
    "Lyapunov Spectrum": "lyapunov_spectrum",
    "Memory Landscape": "memory_landscape",
    "MLP Predict Lorenz": "mlp_predict_lorenz",
    "Noise Robustness": "noise_robustness",
    "Persistent Homology": "persistent_homology",
    "Plasticity Dynamics": "plasticity_dynamics",
    "Questions": "questions",
    "Reflection Modul": "reflection_modul",
    "XOR Prediction": "xor_prediction"
}

st.set_page_config(page_title="ReflectAI", layout="wide")
st.sidebar.title("ReflectAI modulválasztó")

selected_module = st.sidebar.selectbox("Válassz modult:", list(modules.keys()))

# Modul betöltése
module_name = modules[selected_module]
try:
    module = importlib.import_module(module_name)
    if hasattr(module, "app"):
        module.app()
    else:
        st.error(f"A(z) `{module_name}` modul nem tartalmaz `app()` függvényt.")
except ModuleNotFoundError:
    st.error(f"Nem található a(z) `{module_name}.py` fájl.")
except Exception as e:
    st.error(f"Hiba történt a modul betöltésekor: {e}")
