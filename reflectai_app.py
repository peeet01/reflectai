import streamlit as st import importlib import os

🌐 Metaadat

st.set_page_config(page_title="Neurolab AI", layout="wide")

🌌 Kezdőkép beállítás

st.markdown(""" <style> .clickable-image { position: relative; display: inline-block; } .clickable-button { position: absolute; top: 24.5%;  /* kb. agy pozíciója a képen */ left: 35.5%; width: 20%; height: 20%; background-color: rgba(0, 0, 0, 0); border: none; cursor: pointer; z-index: 2; } .background-image { width: 100%; } </style> """, unsafe_allow_html=True)

🔧 Modulregiszter

modules = { "Berry Curvature": "berry_curvature", "Data Upload": "data_upload", "ESN Prediction": "esn_prediction", "Fractal Dimension": "fractal_dimension", "Generative Kuramoto": "generative_kuramoto", "Graph Sync Analysis": "graph_sync_analysis", "Hebbian Learning": "hebbian_learning", "Hebbian Learning Viz": "hebbian_learning_viz", "Help": "help_module", "Insight Learning": "insight_learning", "Kuramoto Hebbian Sim": "kuramoto_hebbian_sim", "Kuramoto Sim": "kuramoto_sim", "Lorenz Sim": "lorenz_sim", "Lyapunov Spectrum": "lyapunov_spectrum", "Memory Landscape": "memory_landscape", "MLP Predict Lorenz": "mlp_predict_lorenz", "Noise Robustness": "noise_robustness", "Persistent Homology": "persistent_homology", "Plasticity Dynamics": "plasticity_dynamics", "Questions": "questions", "Reflection Modul": "reflection_modul", "XOR Prediction": "xor_prediction" }

🤔 Modulválasztás

if "show_menu" not in st.session_state: st.session_state.show_menu = False

if not st.session_state.show_menu: st.markdown('<div class="clickable-image">', unsafe_allow_html=True) st.image("static/nyitokep.png", use_column_width=True) st.markdown( '<form action="" method="post">' '<button class="clickable-button" name="activate" type="submit"></button>' '</form>' '</div>', unsafe_allow_html=True ) if st.experimental_get_query_params().get("activate") or st.form_submit_button("activate"): st.session_state.show_menu = True st.stop()

→ Ha show_menu True, akkor menü jelenik meg

st.sidebar.subheader("🔪 Modulválasztó") selected_title = st.sidebar.radio("Válassz modult:", list(modules.keys()))

selected_module_name = modules[selected_title] try: module = importlib.import_module(f"modules.{selected_module_name}") if hasattr(module, "app"): module.app() else: st.error(f"❌ A(z) {selected_module_name} modul nem tartalmaz app() nevű függvényt.") except ModuleNotFoundError: st.error(f"📦 A(z) {selected_module_name} modul nem található a modules/ mappában.") except Exception as e: st.error(f"🚨 Hiba történt a(z) {selected_title} modul betöltésekor:") st.exception(e)

