# -*- coding: utf-8 -*-
import streamlit as st

# Modulok importálása
from modules import (
    kuramoto_sim,
    kuramoto_hebbiansim,
    hebbian_learning,
    hebbian_learning_visual,
    graph_sync_analysis,
    xor_prediction,
    lorenz_sim,
    predict_lorenz,
    mlp_predict_lorenz,
)

st.set_page_config(page_title="ReflectAI App", layout="centered")
st.title("🧠 ReflectAI App")

# Oldalsáv menü
menu = st.sidebar.selectbox(
    "Válassz egy modult:",
    (
        "Kuramoto szinkronizáció",
        "Adaptív Kuramoto–Hebbian háló",
        "Hebbian tanulás zajjal",
        "Topológia és zaj hatása",
        "XOR predikció",
        "Lorenz szimuláció",
        "Lorenz predikció",
        "MLP Lorenz predikció"
    )
)

# Menü működés
if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Adaptív Kuramoto–Hebbian háló":
    kuramoto_hebbiansim.run()

elif menu == "Hebbian tanulás zajjal":
    hebbian_learning_visual.run()

elif menu == "Topológia és zaj hatása":
    graph_sync_analysis.run()

elif menu == "XOR predikció":
    xor_prediction.run()

elif menu == "Lorenz szimuláció":
    lorenz_sim.run()

elif menu == "Lorenz predikció":
    predict_lorenz.run()

elif menu == "MLP Lorenz predikció":
    mlp_predict_lorenz.run()
