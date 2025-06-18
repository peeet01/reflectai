
# -*- coding: utf-8 -*-
import streamlit as st
from modules import kuramoto_sim
from modules.hebbian_learning import hebbian_learning_with_noise
from modules.xor_prediction import run_xor_prediction
from modules.graph_sync_analysis import adaptive_kuramoto_hebbian_network, topology_noise_effects

st.set_page_config(page_title="ReflectAI App", layout="centered")
st.title("🧠 ReflectAI App")

selected = st.sidebar.radio("Válassz modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás zajjal",
    "XOR predikciós tanulási feladat",
    "Adaptív Kuramoto–Hebbian háló",
    "Tudományos kérdés: Topológia és zaj hatása"
])

if selected == "Kuramoto szinkronizáció":
    kuramoto_sim.run()
elif selected == "Hebbian tanulás zajjal":
    hebbian_learning_with_noise()
elif selected == "XOR predikciós tanulási feladat":
    run_xor_prediction()
elif selected == "Adaptív Kuramoto–Hebbian háló":
    adaptive_kuramoto_hebbian_network()
elif selected == "Tudományos kérdés: Topológia és zaj hatása":
    topology_noise_effects()
