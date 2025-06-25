import streamlit as st from datetime import datetime import importlib

Modul-regiszter: kulcs = modulnév (oldalsávban),

érték = Python modul neve (importhoz)

modules = { "Berry Curvature": "berry_curvature", "Data Upload": "data_upload", "ESN Prediction": "esn_prediction", "Fractal Dimension": "fractal_dimension", "Generative Kuramoto": "generative_kuramoto", "Graph Sync Analysis": "graph_sync_analysis", "Hebbian Learning": "hebbian_learning", "Hebbian Learning Viz": "hebbian_learning_viz", "Help": "help_module", "Insight Learning": "insight_learning", "Kuramoto Hebbian Sim": "kuramoto_hebbian", "Kuramoto Sim": "kuramoto_sim", "Lorenz Sim": "lorenz_sim", "Lyapunov Spectrum": "lyapunov_spectrum", "Memory Landscape": "memory_landscape", "MLP Predict Lorenz": "mlp_predict_lorenz", "Noise Robustness": "noise_robustness", "Persistent Homology": "persistent_homology", "Plasticity Dynamics": "plasticity_dynamics", "Questions": "questions", "Reflection Modul": "reflection_modul", "XOR Prediction": "xor_prediction", }

st.set_page_config(page_title="ReflectAI", layout="wide") st.sidebar.title("ReflectAI Modulválasztó") modul_nev = st.sidebar.selectbox("Válassz modult", list(modules.values()))

if modul_nev: modul = importlib.import_module(f"modules.{modul_nev}") modul.app()

