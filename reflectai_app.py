# -*- coding: utf-8 -*-
import streamlit as st

from modules import (
    kuramoto_sim,
    hebbian_learning,
    xor_prediction,
    kuramoto_hebbiansim,
    graph_sync_analysis
)

st.set_page_config(page_title="ReflectAI App", layout="wide")
st.title("🧠 ReflectAI App")

menu = st.sidebar.selectbox("Válassz modult:", (
    "Kuramoto szinkronizáció",
    "Hebbian tanulás zajjal",
    "XOR predikciós tanulási feladat",
    "Adaptív Kuramoto–Hebbian háló",
    "🧪 Tudományos kérdés: Topológia és zaj hatása"
))

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Hebbian tanulás zajjal":
    hebbian_learning.run()

elif menu == "XOR predikciós tanulási feladat":
    xor_prediction.run()

elif menu == "Adaptív Kuramoto–Hebbian háló":
    kuramoto_hebbiansim.run()

elif menu == "🧪 Tudományos kérdés: Topológia és zaj hatása":
    graph_sync_analysis.run()
