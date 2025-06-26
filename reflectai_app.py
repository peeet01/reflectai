import streamlit as st
from datetime import datetime
import importlib

# 🌐 Alkalmazás metaadatai
st.set_page_config(page_title="Neurolab AI", layout="wide")

# 🧪 Cím és leírás

st.image("static/nyitokep.png", use_container_width=1500)

# 📦 Modul-regiszter (modulnév: fájlnév)
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

# 🧭 Modulválasztó az oldalsávban
selected_title = st.sidebar.radio("🔬 Modulválasztó", list(modules.keys()))
selected_module_name = modules[selected_title]

# 🔄 Modul betöltése
try:
    module = importlib.import_module(selected_module_name)
    if hasattr(module, "app"):
        module.app()
    else:
        st.error(f"A(z) `{selected_module_name}` modul nem tartalmaz `app` függvényt.")
except Exception as e:
    st.error(f"❌ Hiba történt a(z) `{selected_title}` modul betöltésekor:")
    st.exception(e)
