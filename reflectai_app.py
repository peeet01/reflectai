import streamlit as st

from modules import (
    graph_sync_analysis,
    hebbian_learning,
    hebbian_learning_viz,
    kuramoto_hebbiansim,
    kuramoto_sim,
    lorenz_sim,
    mlp_predict_lorenz,
    modules_utils,
    predict_lorenz,
    utils,
    xor_prediction,
)

st.set_page_config(page_title="ReflectAI App", layout="centered")
st.title("🧠 ReflectAI App")

menu = st.sidebar.selectbox("Válassz modult", (
    "Kuramoto szinkronizáció",
    "Kuramoto-Hebbian szimuláció",
    "Hebbian tanulás",
    "Hebbian vizualizáció",
    "Lorenz szimuláció",
    "Lorenz jóslás (MLP)",
    "Lorenz jóslás (egyszerű)",
    "Gyakorlati XOR predikció",
    "Görbe szinkron analízis",
    "Utils teszt",
))

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Kuramoto-Hebbian szimuláció":
    kuramoto_hebbiansim.run()

elif menu == "Hebbian tanulás":
    hebbian_learning.run()

elif menu == "Hebbian vizualizáció":
    hebbian_learning_viz.run()

elif menu == "Lorenz szimuláció":
    lorenz_sim.run()

elif menu == "Lorenz jóslás (MLP)":
    mlp_predict_lorenz.run()

elif menu == "Lorenz jóslás (egyszerű)":
    predict_lorenz.run()

elif menu == "Gyakorlati XOR predikció":
    xor_prediction.run()

elif menu == "Görbe szinkron analízis":
    graph_sync_analysis.run()

elif menu == "Utils teszt":
    utils.run()
