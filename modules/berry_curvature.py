# modules/berry_curvature.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def berry_curvature_example(kx, ky):
    # Egyszerű Berry-görbület példafüggvény
    denom = (1 + kx**2 + ky**2)**2
    return 2 * ky / denom

def run():
    st.subheader("🌀 Topológiai védettség – Berry-görbület")
    st.write("Ez a modul egy egyszerű Berry-görbület térképet vizualizál.")

    # Rács definiálása kx, ky térben
    kx = np.linspace(-3, 3, 200)
    ky = np.linspace(-3, 3, 200)
    KX, KY = np.meshgrid(kx, ky)

    # Berry-görbület kiszámítása
    Berry = berry_curvature_example(KX, KY)

    # Ábra
    fig, ax = plt.subplots()
    contour = ax.contourf(KX, KY, Berry, levels=50, cmap='coolwarm')
    fig.colorbar(contour)
    ax.set_title("Berry-görbület hőtérkép")
    ax.set_xlabel("kₓ")
    ax.set_ylabel("kᵧ")

    st.pyplot(fig)
    st.success("Berry-görbület vizualizáció sikeres.")
