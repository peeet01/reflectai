import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

# Bevezető
st.set_page_config(layout="wide")

st.title("\U0001f9e0 Belátás alapú tanulás – Insight Learning szimuláció")

st.markdown("""
Ez a modul egy egyszerű környezetben modellezi a **belátás alapú tanulást**, 
ahol az ügynök egy ponton _hirtelen_ megérti, hogyan érheti el a célt.
""")

# Szimulációs függvények
def generate_environment(grid_size, agent_pos, goal_pos, obstacle_pos):
    env = np.zeros((grid_size, grid_size))
    env[tuple(goal_pos)] = 2
    env[tuple(obstacle_pos)] = -1
    env[tuple(agent_pos)] = 1
    return env

def simulate_learning(grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight):
    steps = []
    found = False
    steps_to_goal = []
    activations = []
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

def plot_environment(grid_size, steps, goal_pos, obstacle_pos):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()
    ax.grid(True)
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

def plot_brain_activity_2d(activation_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(activation_map, cmap="plasma", interpolation='nearest')
    ax.set_title("\U0001f9e0 Aktivációs térkép (neurális mintázat)")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    fig.colorbar(im, ax=ax, label="Aktiváció gyakoriság")
    return fig

def plot_brain_activity_3d(activation_map):
    z = gaussian_filter(activation_map, sigma=1.2)
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    aha_level = np.max(z) * 0.7
    eruption_mask = z > aha_level
    erupt_x = x[eruption_mask]
    erupt_y = y[eruption_mask]
    erupt_z = z[eruption_mask]
    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        opacity=0.95,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=1.0, roughness=0.2),
        lightposition=dict(x=30, y=50, z=100)
    ))
    fig.add_trace(go.Surface(
        z=np.full_like(z, aha_level),
        x=x,
        y=y,
        opacity=0.15,
        showscale=False,
        colorscale=[[0, 'white'], [1, 'white']],
        name='Aha-szint'
    ))
    if len(erupt_z) > 0:
        fig.add_trace(go.Scatter3d(
            x=erupt_x,
            y=erupt_y,
            z=erupt_z + 0.2,
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8,
                symbol='circle',
                line=dict(width=2, color='orangered')
            ),
            name='Tűzkitörés'
        ))
    fig.update_layout(
        title="\U0001f525 3D agyi aktiváció – Belátás mint kitörés",
        scene=dict(
            xaxis_title="Neuron X",
            yaxis_title="Neuron Y",
            zaxis_title="Aktiváció",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(nticks=6, range=[0, np.max(z) + 2])
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    return fig

# Paraméterek beállítása
st.sidebar.header("⚖️ Paraméterek")
grid_size = st.sidebar.slider("Rács méret", 5, 15, 7)
episodes = st.sidebar.slider("Epizódok száma", 10, 200, 50, step=10)
max_steps = st.sidebar.slider("Lépések epizódonként", 5, 50, 20)
use_insight = st.sidebar.checkbox("Belátás aktiválása", value=True)

# Szimuláció futtatása
agent_pos = [grid_size - 1, 0]
goal_pos = [0, grid_size - 1]
obstacle_pos = [grid_size // 2, grid_size // 2]

steps, found, steps_to_goal, activations = simulate_learning(
    grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight
)

st.markdown("### 🌍 Környezet vizualizáció")
st.pyplot(plot_environment(grid_size, steps, goal_pos, obstacle_pos))

st.markdown("### 📉 Tanulási görbe")
fig_steps, ax_steps = plt.subplots()
ax_steps.plot(steps_to_goal, marker='o')
ax_steps.set_xlabel("Epizód")
ax_steps.set_ylabel("Lépésszám")
ax_steps.set_title("Epizódonkénti lépésszám")
st.pyplot(fig_steps)

st.markdown("### 🧠 Aktivációs mintázatok")
selected_ep = st.slider("Megfigyelni kívánt epizód", 0, episodes - 1, episodes - 1)
tabs = st.tabs(["2D", "3D"])
with tabs[0]:
    st.pyplot(plot_brain_activity_2d(activations[selected_ep]))
with tabs[1]:
    st.plotly_chart(plot_brain_activity_3d(activations[selected_ep]), use_container_width=True)

if found:
    st.success("🎉 Az ügynök elérte a célt!")
else:
    st.warning("🤔 Az ügynök nem érte el a célt.")

# Tudományos leírás (LaTeX formázva)
st.markdown("""
### 📘 Tudományos háttér

A **belátásos tanulás** (
$\textit{insight learning}$
) egy kognitív folyamat, ahol a megoldás nem fokozatos, hanem **hirtelen felismeréssel** jön létre.

#### Matematikai modell

A neuronok aktivációs térképe $A(x, y)$

$$
A(x, y) = \sum_{t=0}^{T} \delta(x - x_t, y - y_t)
$$

Az "aha" szintet pedig küszöbszinttel jelöljük:

$$
A_{\text{kritikus}} = \alpha \cdot \max A(x, y)
$$

ahol $\alpha \in [0, 1]$.

Ahol $A(x,y) > A_{\text{kritikus}}$ ott "kitörést" észlelünk.
""")

# Riport mentése
with st.expander("📄 Riport mentése"):
    if st.button("📅 TXT generálás"):
        report = f"""
**Insight Learning riport**

- Időpont: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Rács méret: {grid_size}x{grid_size}
- Epizódok: {episodes}
- Lépések/epizód: {max_steps}
- Belátás aktiválva: {'Igen' if use_insight else 'Nem'}
- Cél elérve: {'Igen' if found else 'Nem'}
- Átlagos lépésszám: {np.mean(steps_to_goal):.2f}
"""
        fname = "insight_report.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report)
        with open(fname, "rb") as f:
            st.download_button("Letöltés", f, file_name=fname)
        os.remove(fname)
