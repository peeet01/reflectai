# reflectai_app.py - ReflectAI főfájl (22 modul dinamikus betöltéssel)

import streamlit as st
from datetime import datetime
import importlib

# Modul-regiszter: kulcs = megjelenő név, érték = modulnév (fájlnév)
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
    "Kuramoto Hebbian Sim": "kuramoto_hebbian",
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
    "XOR Prediction": "xor_prediction",
}

st.set_page_config(page_title="ReflectAI", layout="wide")
st.sidebar.title("ReflectAI Modulválasztó")
selected = st.sidebar.selectbox("Válassz modult", list(modules.values()))

try:
    mod = importlib.import_module(f"modules.{selected}")
    app_fn = getattr(mod, "app", None)
    if app_fn:
        app_fn()
    else:
        st.error(f"A(z) `{selected}` modul nem tartalmaz `app` függvényt.")
except ModuleNotFoundError:
    st.error(f"A(z) `{selected}` modul nem található.")
except Exception as e:
    st.exception(e)
