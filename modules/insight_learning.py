import streamlit as st import numpy as np import matplotlib.pyplot as plt import time import os from datetime import datetime import plotly.graph_objects as go from scipy.ndimage import gaussian_filter

Bevezető

st.set_page_config(layout="wide") st.title("🧠 Belátás alapú tanulás – Insight Learning szimuláció") st.markdown(""" Ez a modul egy egyszerű környezetben modellezi a belátás alapú tanulást,

ahol az ügynök egy ponton hirtelen megérti, hogyan érheti el a célt. A szimuláció lépésről lépésre építi fel a tapasztalatokat, majd amikor az aktiváció meghalad egy kritikus szintet, megtörténik az „aha” pillanat. """)

def generate_environment(grid_size, agent_pos, goal_pos, obstacle_pos): env = np.zeros((grid_size, grid_size)) env[tuple(goal_pos)] = 2 env[tuple(obstacle_pos)] = -1 env[tuple(agent_pos)] = 1 return env

def simulate_learning(grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight): steps = [] found = False steps_to_goal = [] activations = []

for episode in range(episodes):
    pos = agent_pos.copy()
    path = [tuple(pos)]
    activation_map = np.zeros((grid_size, grid_size))

    for _ in range(max_steps):
        activation_map[tuple(pos)] += 1
        if pos == goal_pos:
            found = True
            break
        if use_insight and pos[1] < obstacle_pos[1] and pos[0] == obstacle_pos[0]:
            pos[0] -= 1
        else:
            if pos[1] < grid_size - 1:
                pos[1] += 1
            elif pos[0] > 0:
                pos[0] -= 1
        path.append(tuple(pos))

    steps.append(path)
    steps_to_goal.append(len(path))
    activations.append(activation_map)

return steps, found, steps_to_goal, activations

def plot_environment(grid_size, steps, goal_pos, obstacle_pos): fig, ax = plt.subplots(figsize=(5, 5)) ax.set_xlim(-0.5, grid_size - 0.5) ax.set_ylim(-0.5, grid_size - 0.5) ax.invert_yaxis() ax.grid(True)

for x in range(grid_size):
    for y in range(grid_size):
        if [x, y] == goal_pos:
            ax.text(y, x, '🏁', ha='center', va='center')
        elif [x, y] == obstacle_pos:
            ax.text(y, x, '🧱', ha='center', va='center')

for path in steps[-5:]:
    xs, ys = zip(*path)
    ax.plot(ys, xs, alpha=0.6)
return fig

def mandelbrot_activation(grid_size): x = np.linspace(-2, 1, grid_size) y = np.linspace(-1.5, 1.5, grid_size) X, Y = np.meshgrid(x, y) C = X + 1j * Y Z = np.zeros_like(C) output = np.zeros(C.shape, dtype=int) for i in range(30): Z = Z**2 + C output += (np.abs(Z) < 2).astype(int) return gaussian_filter(output, sigma=1.0)

def plot_brain_activity_3d(activation_map): z = activation_map x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0])) aha_level = np.max(z) * 0.85 eruption_mask = z > aha_level erupt_x = x[eruption_mask] erupt_y = y[eruption_mask] erupt_z = z[eruption_mask]

fig = go.Figure()
fig.add_trace(go.Surface(
    z=z, x=x, y=y, colorscale='Viridis', opacity=0.95, showscale=False,
    lighting=dict(ambient=0.6, diffuse=0.8),
    lightposition=dict(x=50, y=100, z=200)
))
fig.add_trace(go.Surface(
    z=np.full_like(z, aha_level), x=x, y=y,
    opacity=0.2, colorscale=[[0, 'white'], [1, 'white']], showscale=False,
    name='Aha-szint'
))
if len(erupt_z) > 0:
    fig.add_trace(go.Scatter3d(
        x=erupt_x, y=erupt_y, z=erupt_z + 0.3,
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.9, line=dict(width=2, color='orange')),
        name='Kitörés'
    ))
fig.update_layout(
    title="🔥 3D agyi domborzat – Mandelbrot alapú belátás",
    scene=dict(
        xaxis_title="Neuron X", yaxis_title="Neuron Y", zaxis_title="Aktiváció",
        zaxis=dict(nticks=6, range=[0, np.max(z)+2])
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    template="plotly_dark"
)
return fig

Paraméterek

st.sidebar.header("Paraméterek") grid_size = st.sidebar.slider("🔲 Rács méret", 5, 15, 7) episodes = st.sidebar.slider("🔁 Epizódok száma", 10, 200, 50, step=10) max_steps = st.sidebar.slider("🚶‍♂️ Lépések epizódonként", 5, 50, 20) use_insight = st.sidebar.checkbox("💡 Belátás aktiválása", value=True)

agent_pos = [grid_size - 1, 0] goal_pos = [0, grid_size - 1] obstacle_pos = [grid_size // 2, grid_size // 2]

steps, found, steps_to_goal, activations = simulate_learning( grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight )

st.markdown("### 🌍 Környezet vizualizáció") st.pyplot(plot_environment(grid_size, steps, goal_pos, obstacle_pos))

st.markdown("### 📉 Lépések száma epizódonként") fig_steps, ax = plt.subplots() ax.plot(steps_to_goal, marker='o') ax.set_xlabel("Epizód") ax.set_ylabel("Lépésszám") ax.set_title("Tanulási görbe") st.pyplot(fig_steps)

st.markdown("### 🧠 Aktivációs agymodell") selected_ep = st.slider("🧪 Epizód kiválasztása", 0, episodes - 1, episodes - 1)

tabs = st.tabs(["2D", "3D"]) with tabs[0]: st.pyplot(plt.imshow(activations[selected_ep], cmap="plasma")) with tabs[1]: mandel_map = mandelbrot_activation(grid_size) st.plotly_chart(plot_brain_activity_3d(mandel_map), use_container_width=True)

st.markdown("### 💾 Eredmények exportálása") df = np.array(activations[selected_ep]) csv = '\n'.join([','.join(map(str, row)) for row in df]) st.download_button("⬇️ CSV letöltése", csv.encode('utf-8'), file_name="activation_map.csv")

st.markdown("### 📘 Tudományos háttér") st.markdown(r""" A belátásos tanulás (insight learning)

olyan tanulási forma, ahol a megoldás nem fokozatosan, hanem hirtelen jelenik meg a gondolkodás eredményeként.

🧪 Kulcselem:

A tapasztalatok egy "aktivációs térképet" alkotnak, ahol ha az ingerelés meghalad egy szintet, megtörténik az "aha" pillanat:

A(x,y) > \theta \Rightarrow \text{Bel\u00e1t\u00e1s} \text{ (kit\u00f6r\u00e9s)}

ahol:

: aktiváció a r\u00e9gi\u00f3ban

: kritikus szint ("aha" szint)


Ez a szimuláció a felismerést egy kitöréssel ábrázolja a 3D felszínen.

🧠 Előnyei:

Modellálja a kreatív problémamegoldást

Időalapú és viselkedési tanulási minták szimulálhatók vele """)


