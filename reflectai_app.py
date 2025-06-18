# -*- coding: utf-8 -*-
import streamlit as st

from modules import (
    kuramoto_sim,
    hebbian_learning,
    xor_prediction,
    kuramoto_hebbiansim,
    graph_sync_analysis
)

# Alkalmazás címe
st.set_page_config(page_title="ReflectAI App", layout="centered")
st.title("🧠 ReflectAI App")

# Oldalsáv menü
menu = st.sidebar.selectbox(
    "Modul kiválasztása",
    (
        "Kuramoto szinkronizáció",
        "Hebbian tanulás zajjal",
        "XOR predikciós tanulási feladat",
        "Adaptív Kuramoto–Hebbian háló",
        "Tudományos kérdés: Topológia és zaj hatása"
    )
)

# Menü vezérlés
if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Hebbian tanulás zajjal":
    hebbian_learning.run()

elif menu == "XOR predikciós tanulási feladat":
    xor_prediction.run()

elif menu == "Adaptív Kuramoto–Hebbian háló":
    kuramoto_hebbiansim.run()

elif menu == "Tudományos kérdés: Topológia és zaj hatása":
    graph_sync_analysis.run()
