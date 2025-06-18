import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def run():
    st.subheader("Gráf Szinkronizáció analízis")
    st.write("Gráf szinkronizáció analízis modul fut.")

    G = nx.erdos_renyi_graph(n=10, p=0.3)
    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, ax=ax)
    st.pyplot(fig)
