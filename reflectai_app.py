import streamlit as st
from datetime import datetime
import importlib

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

# --- Streamlit oldal ---
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI Modulválasztó")

# Modulválasztó a sidebar-ban
selected_label = st.sidebar.selectbox("Válassz modult", list(modules.keys()))
selected_module = modules[selected_label]

try:
    mod = importlib.import_module(f"modules.{selected_module}")
    mod.app()  # 🔧 Fontos: minden modul végén legyen: app = run
except Exception as e:
    st.error(f"Hiba történt a(z) {selected_label} betöltésekor:\n\n{e}")
