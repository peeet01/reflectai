
# -*- coding: utf-8 -*-
import streamlit as st

# Modulok importálása
from modules import (
    kuramoto_sim,
    kuramoto_hebbiansim,
    hebbian_learning,
    hebbian_learning_vizu,
    xor_prediction,
    graph_sync_analysis,
    predict_lorenz,
    mlp_predict_lorenz
)

st.set_page_config(page_title="ReflectAI App", layout="wide")
st.title("🧠 ReflectAI App")

# Menü kiválasztás
menu = st.sidebar.selectbox("Válassz modult", [
    "Kuramoto szinkronizáció",
    "Adaptív Kuramoto-Hebbian háló",
    "Hebbian tanulás",
    "Hebbian tanulás vizualizációval",
    "XOR predikció",
    "Topológiai szinkron analízis",
    "Lorenz előrejelzés",
    "MLP Lorenz előrejelzés"
])

# Menü alapján modul futtatás
if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()
elif menu == "Adaptív Kuramoto-Hebbian háló":
    kuramoto_hebbiansim.run()
elif menu == "Hebbian tanulás":
    hebbian_learning.run()
elif menu == "Hebbian tanulás vizualizációval":
    hebbian_learning_vizu.run()
elif menu == "XOR predikció":
    xor_prediction.run()
elif menu == "Topológiai szinkron analízis":
    graph_sync_analysis.run()
elif menu == "Lorenz előrejelzés":
    predict_lorenz.run()
elif menu == "MLP Lorenz előrejelzés":
    mlp_predict_lorenz.run()
