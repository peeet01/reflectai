# reflectai_app.py

import streamlit as st
from modules import (
    kuramoto_sim,
    hebbian_learning,
    hebbian_learning_viz,
    kuramoto_hebbiansim,
    lorenz_sim,
    predict_lorenz,
    xor_prediction,
    graph_sync_analysis,
    modules_utils
)

st.set_page_config(page_title="ReflectAI App", page_icon="🧠", layout="wide")
st.title("🧠 ReflectAI App")

# Modulválasztó menü
menu = st.sidebar.selectbox(
    "Válassz modult:",
    (
        "Kuramoto szinkronizáció",
        "Hebbian tanulás",
        "Hebbian vizualizáció",
        "Kuramoto + Hebbian",
        "Lorenz szimuláció",
        "Lorenz előrejelzés",
        "XOR predikció",
        "Gráf szinkron elemzés"
    )
)

# Modulok futtatása
if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()
elif menu == "Hebbian tanulás":
    hebbian_learning.run()
elif menu == "Hebbian vizualizáció":
    hebbian_learning_viz.run()
elif menu == "Kuramoto + Hebbian":
    kuramoto_hebbiansim.run()
elif menu == "Lorenz szimuláció":
    lorenz_sim.run()
elif menu == "Lorenz előrejelzés":
    predict_lorenz.run()
elif menu == "XOR predikció":
    xor_prediction.run()
elif menu == "Gráf szinkron elemzés":
    graph_sync_analysis.run()
