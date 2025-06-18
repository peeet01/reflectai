import streamlit as st

from modules import (
    kuramoto_sim,
    hebbian_learning,
    hebbian_learning_viz,
    kuramoto_hebbiansim,
    lorenz_sim,
    predict_lorenz,
    xor_prediction,
    graph_sync_analysis
)

st.title("🧠 ReflectAI App")

menu = st.sidebar.selectbox("Válassz modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "Hebbian vizualizáció",
    "Kuramoto-Hebbian szinkronizáció",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "XOR predikció",
    "Gráf szinkron analízis"
])

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()
elif menu == "Hebbian tanulás":
    hebbian_learning.run()
elif menu == "Hebbian vizualizáció":
    hebbian_learning_viz.run()
elif menu == "Kuramoto-Hebbian szinkronizáció":
    kuramoto_hebbiansim.run()
elif menu == "Lorenz szimuláció":
    lorenz_sim.run()
elif menu == "Lorenz predikció":
    predict_lorenz.run()
elif menu == "XOR predikció":
    xor_prediction.run()
elif menu == "Gráf szinkron analízis":
    graph_sync_analysis.run()
