import streamlit as st import numpy as np import networkx as nx import plotly.graph_objects as go import time

def kuramoto_step(theta, K, A, omega, dt): N = len(theta) dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1) return theta + dt * dtheta

def compute_order_parameter(theta): return np.abs(np.mean(np.exp(1j * theta)))

def generate_graph(N, graph_type): if graph_type == "Teljes": G = nx.complete_graph(N) elif graph_type == "Véletlen (Erdős-Rényi)": G = nx.erdos_renyi_graph(N, p=0.3) elif graph_type == "Kis világ (Watts-Strogatz)": G = nx.watts_strogatz_graph(N, k=4, p=0.3) elif graph_type == "Skálafüggetlen (Barabási-Albert)": G = nx.barabasi_albert_graph(N, m=2) else: G = nx.complete_graph(N) return G

def run(): st.title("🧠 Kuramoto Szinkronizáció – Interaktív Vizualizáció")

N = st.slider("Oszcillátorok száma", 5, 100, 30)
K = st.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, 0.1)
steps = st.slider("Iterációk", 100, 2000, 500, 100)
dt = 0.05

graph_type = st.selectbox("Hálózat típusa", ["Teljes", "Véletlen (Erdős-Rényi)", "Kis világ (Watts-Strogatz)", "Skálafüggetlen (Barabási-Albert)"])
palette = st.selectbox("Szíséma", ["Turbo", "Viridis", "Electric", "Hot", "Rainbow"])

st.subheader("Szimuláció futtatása")
progress = st.progress(0)

np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, N)
omega = np.random.normal(0, 1, N)

