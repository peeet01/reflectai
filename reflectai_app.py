# -*- coding: utf-8 -*-
import streamlit as st

# Modulok importálása (neveknek meg kell egyezniük a modules könyvtár fájljaival, de .py nélkül)
from modules import (
    kuramoto_sim,
    kuramoto_hebbian_sim,
    hebbian_learning,
    hebbian_learning_visual,
    xor_prediction,
    mlp_predict_lorenz,
    predict_lorenz,
    lorenz_sim,
    graph_sync_analysis,
)

st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI App")

menu = st.sidebar.selectbox("Válassz modult", (
    "Kuramoto szinkronizáció",
    "Kuramoto–Hebbian háló",
    "Hebbian tanulás",
    "Hebbian tanulás vizualizáció",
    "XOR predikció",
    "MLP predikció Lorenz adatokon",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "Topológiai gráf szinkron analízis"
))

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Kuramoto–Hebbian háló":
    kuramoto_hebbian_sim.run()

elif menu == "Hebbian tanulás":
    hebbian_learning.run()

elif menu == "Hebbian tanulás vizualizáció":
    hebbian_learning_visual.run()

elif menu == "XOR predikció":
    xor_prediction.run()

elif menu == "MLP predikció Lorenz adatokon":
    mlp_predict_lorenz.run()

elif menu == "Lorenz szimuláció":
    lorenz_sim.run()

elif menu == "Lorenz predikció":
    predict_lorenz.run()

elif menu == "Topológiai gráf szinkron analízis":
    graph_sync_analysis.run()
