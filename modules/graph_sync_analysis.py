
import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Topológiai gráfvizsgálat szinkronizációval")
    G = nx.erdos_renyi_graph(20, 0.1)
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue')
    st.pyplot(fig)
    st.write("Csúcsszám:", G.number_of_nodes())
