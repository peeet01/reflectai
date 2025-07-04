import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from scipy.ndimage import gaussian_filter

# 🎯 Definálás
def generate_environment(grid_size, agent_pos, goal_pos, obstacle_pos):
    env = np.zeros((grid_size, grid_size))
    env[tuple(goal_pos)] = 2  # Goal
    env[tuple(obstacle_pos)] = -1  # Obstacle
    env[tuple(agent_pos)] = 1  # Agent
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
                pos[0] -= 1  # Insight: jump over obstacle
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
    ax.set_title("🧠 Aktivációs térkép (neurális mintázat)")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    fig.colorbar(im, ax=ax, label="Aktiváció gyakoriság")
    return fig

def plot_brain_activity_3d(activation_map):
    z = gaussian_filter(activation_map, sigma=1.2)
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    aha_level = np.max(z) * 0.7  # Aha level for breakthrough
    eruption_mask = z > aha_level
    erupt_x = x[eruption_mask]
    erupt_y = y[eruption_mask]
    erupt_z = z[eruption_mask]

    fig = go.Figure()

    # Terrain (activation)
    fig.add_trace(go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale='Inferno',
        opacity=0.95,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=1.0, roughness=0.2),
        lightposition=dict(x=30, y=50, z=100)
    ))

    # Aha level (glass layer)
    fig.add_trace(go.Surface(
        z=np.full_like(z, aha_level),
        x=x,
        y=y,
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'white'], [1, 'white']],
        name='Aha-szint'
    ))

    # Eruption points
    if len(erupt_z) > 0:
        fig.add_trace(go.Scatter3d(
            x=erupt_x,
            y=erupt_y,
            z=erupt_z + 0.2,  # slightly raise eruption above surface
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                opacity=0.9,
                symbol='circle',
                line=dict(width=2, color='orangered')
            ),
            name='Lava eruption'
        ))

    fig.update_layout(
        title="🔥 3D Brain Activation – 'Aha' Insight Eruption",
        scene=dict(
            xaxis_title="Neuron X",
            yaxis_title="Neuron Y",
            zaxis_title="Activation",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(nticks=6, range=[0, np.max(z) + 2])
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_dark"
    )
    return fig

def run():
    st.title("🧠 Insight Learning – Belátásos tanulás szimuláció")

    st.markdown("""
    **Insight Learning**: A **belátásos tanulás** egy kognitív folyamat, ahol a probléma megoldása nem véletlenszerű próbálkozással,  
    hanem egy **strukturális átlátás** révén történik. Az ügynök egy **hirtelen** megértéssel találja meg a megoldást.

    - Az ügynök kezdetben véletlenszerűen próbálkozik, majd hirtelen felismeri a helyes megoldást, amit "aha" pillanatként tapasztal meg.
    """)

    grid_size = st.slider("🔲 Rács méret", 5, 15, 7)
    episodes = st.slider("🔁 Epizódok száma", 10, 200, 50, step=10)
    max_steps = st.slider("🚶‍♂️ Lépések epizódonként", 5, 50, 20)
    use_insight = st.checkbox("💡 Belátás aktiválása", value=True)

    agent_pos = [grid_size - 1, 0]
    goal_pos = [0, grid_size - 1]
    obstacle_pos = [grid_size // 2, grid_size // 2]

    steps, found, steps_to_goal, activations = simulate_learning(
        grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight
    )

    st.markdown("### 🌍 Környezet vizualizáció")
    fig_env = plot_environment(grid_size, steps, goal_pos, obstacle_pos)
    st.pyplot(fig_env)

    st.markdown("### 📉 Lépések száma epizódonként")
    fig_steps, ax_steps = plt.subplots()
    ax_steps.plot(steps_to_goal, marker='o')
    ax_steps.set_xlabel("Epizód")
    ax_steps.set_ylabel("Lépésszám")
    ax_steps.set_title("Tanulási görbe")
    st.pyplot(fig_steps)

    st.markdown("### 🧠 Aktivációs agymodell")
    selected_ep = st.slider("🧪 Megfigyelni kívánt epizód", 0, episodes - 1, episodes - 1)

    tabs = st.tabs(["2D Térkép", "3D Modell"])
    with tabs[0]:
        fig_brain_2d = plot_brain_activity_2d(activations[selected_ep])
        st.pyplot(fig_brain_2d)
    with tabs[1]:
        fig_brain_3d = plot_brain_activity_3d(activations[selected_ep])
        st.plotly_chart(fig_brain_3d, use_container_width=True)

    if found:
        st.success("🎉 Az ügynök elérte a célt – belátás vagy stratégia révén!")
    else:
        st.warning("🤔 Az ügynök még nem találta meg a célt.")

    with st.expander("📝 Riport generálása és letöltés"):
        if st.button("📥 Riport letöltése (.txt)"):
            report_text = f"""Belátás alapú tanulási riport
------------------------------
Dátum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Rács méret: {grid_size}x{grid_size}
Epizódok: {episodes}
Lépések epizódonként: {max_steps}
Belátás aktiválva: {use_insight}
Cél elérve: {"Igen" if found else "Nem"}
Átlagos lépésszám: {np.mean(steps_to_goal):.2f}
"""
            filename = "insight_learning_report.txt"
            with open(filename, "w") as f:
                f.write(report_text)
            with open(filename, "rb") as f:
                st.download_button("⬇️ Letöltés", f, file_name=filename)
            os.remove(filename)

    with st.expander("📘 Tudományos háttér – Mi az a belátás?"):
        st.markdown("""
        A **belátásos tanulás** (insight learning) egy kognitív folyamat, ahol a probléma megoldása nem véletlenszerű próbálkozással,  
        hanem egy *strukturális átlátás* révén történik.

        ### 🐒 Köhler-féle csimpánz kísérlet:
        - Egy banán elérhetetlen, de eszköz segítségével mégis megszerezhető.
        - A megoldás **nem fokozatos**, hanem **hirtelen jelentkezik**.

        A szimulált aktivációs térkép azt reprezentálja, hogy az „agy” mely régiói (pozíciói) milyen gyakran voltak aktívak a sikeres vagy sikertelen keresés során. 
        Amikor az aktiváció meghalad egy *kritikus küszöbszintet*, az ügynök felismeri a megoldást – ezt vizualizáljuk egy "kitörésként" a domborzati agymodellben.
        """)

# ReflectAI kompatibilitás
app = run
