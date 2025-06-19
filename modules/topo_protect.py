import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def compute_chern_number(H):
    eigvals, eigvecs = eigh(H)
    proj = np.outer(eigvecs[:, 0], eigvecs[:, 0].conj())
    berry_curvature = np.imag(np.trace(proj @ proj @ proj))
    return round(berry_curvature, 3)

def run():
    st.subheader("🧭 Topológiai védettség – Chern-szám modul")
    st.write("Egy egyszerű oszcillátorháló alapján számoljuk a hálózat topológiai jellemzőjét (Chern-szám).")

    N = st.slider("🧩 Csomópontok száma", 5, 30, 10)
    rewiring_p = st.slider("🔁 Rewiring valószínűség", 0.0, 1.0, 0.2)

    G = nx.watts_strogatz_graph(N, k=4, p=rewiring_p)
    A = nx.to_numpy_array(G)
    L = np.diag(A.sum(axis=1)) - A

    chern = compute_chern_number(L)

    fig, ax = plt.subplots()
    nx.draw_circular(G, ax=ax, with_labels=True, node_color="skyblue")
    st.pyplot(fig)
    st.info(f"📌 Számított Chern-szám (approx.): **{chern}**")
