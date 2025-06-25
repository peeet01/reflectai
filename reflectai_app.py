import streamlit as st

# 🔽 Automatikus modulimportok
import plasticity_dynamics
import noise_robustness
import reflection_modul
import kuramoto_sim
import mlp_predict_lorenz
import kuramoto_hebbian_sim
import topo_protect
import questions
import xor_prediction
import lyapunov_spectrum
import help_module
import hebbian_learning
import graph_sync_analysis
import insight_learning
import berry_curvature
import hebbian_learning_viz
import memory_landscape
import persistent_homology
import data_upload
import lorenz_sim
import generative_kuramoto
import fractal_dimension
import esn_prediction

# 📚 Modulválasztó regiszter
modulok = {
    "plasticity_dynamics": plasticity_dynamics.app,
    "noise_robustness": noise_robustness.app,
    "reflection_modul": reflection_modul.app,
    "kuramoto_sim": kuramoto_sim.app,
    "mlp_predict_lorenz": mlp_predict_lorenz.app,
    "kuramoto_hebbian_sim": kuramoto_hebbian_sim.app,
    "topo_protect": topo_protect.app,
    "questions": questions.app,
    "xor_prediction": xor_prediction.app,
    "lyapunov_spectrum": lyapunov_spectrum.app,
    "help_module": help_module.app,
    "hebbian_learning": hebbian_learning.app,
    "graph_sync_analysis": graph_sync_analysis.app,
    "insight_learning": insight_learning.app,
    "berry_curvature": berry_curvature.app,
    "hebbian_learning_viz": hebbian_learning_viz.app,
    "memory_landscape": memory_landscape.app,
    "persistent_homology": persistent_homology.app,
    "data_upload": data_upload.app,
    "lorenz_sim": lorenz_sim.app,
    "generative_kuramoto": generative_kuramoto.app,
    "fractal_dimension": fractal_dimension.app,
    "esn_prediction": esn_prediction.app,
}

# 🎛 Modulválasztó felület
st.sidebar.title("ReflectAI Modulválasztó")
valasztott = st.sidebar.selectbox("Válassz modult", list(modulok.keys()))

# ▶ Modul indítása
modulok[valasztott]()
