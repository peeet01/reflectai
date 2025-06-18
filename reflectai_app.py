# -*- coding: utf-8 -*-
import streamlit as st
from modules import kuramoto_sim, hebbian_learning, xor_prediction, graph_sync_analysis

st.set_page_config(page_title='ReflectAI', layout='wide')
st.title('🧠 ReflectAI App')

menu = st.sidebar.radio("Navigáció", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás zajjal",
    "XOR predikciós tanulási feladat",
    "Adaptív Kuramoto–Hebbian háló",
    "🔬 Tudományos kérdés: Topológia és zaj hatása"
])

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Hebbian tanulás zajjal":
    hebbian_learning.run()

elif menu == "XOR predikciós tanulási feladat":
    xor_prediction.run()

elif menu == "Adaptív Kuramoto–Hebbian háló":
    graph_sync_analysis.run_adaptive_graph_simulation()

elif menu == "🔬 Tudományos kérdés: Topológia és zaj hatása":
    graph_sync_analysis.run_topology_noise_simulation()
