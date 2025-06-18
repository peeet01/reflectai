import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def run_graph():
    st.subheader("🌐 Topológiai szinkronizáció")
    st.write("Vizsgálat különböző gráfokkal.")

    G = nx.erdos_renyi_graph(n=10, p=0.3)
    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, ax=ax)
    st.pyplot(fig)
