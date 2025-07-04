import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

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
                pos[0] -= 1  # Insight: ugrás
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
    from scipy.ndimage import gaussian_filter

    z = gaussian_filter(activation_map, sigma=1.2)
    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))

    # Aha-szint definiálása
    aha_level = np.max(z) * 0.7

    # Kitörési pontok detektálása
    eruption_mask = z > aha_level
    erupt_x = x[eruption_mask]
    erupt_y = y[eruption_mask]
    erupt_z = z[eruption_mask]

    fig = go.Figure()

    # Domborzat (aktiváció)
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

    # Aha-szint sík (üveglapként)
    fig.add_trace(go.Surface(
        z=np.full_like(z, aha_level),
        x=x,
        y=y,
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'white'], [1, 'white']],
        name='Aha-szint'
    ))

    # Kitörési lávapontok
    if len(erupt_z) > 0:
        fig.add_trace(go.Scatter3d(
            x=erupt_x,
            y=erupt_y,
            z=erupt_z + 0.2,  # picit emeld ki a felszínről
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                opacity=0.9,
                symbol='circle',
                line=dict(width=2, color='orangered')
            ),
            name='Lávakitörés'
        ))

    fig.update_layout(
        title="🔥 3D agyi aktiváció – 'Aha' felismerés robbanással",
        scene=dict(
            xaxis_title="Neuron X",
            yaxis_title="Neuron Y",
            zaxis_title="Aktiváció",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(nticks=6, range=[0, np.max(z) + 2])
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_dark"
    )
    return fig
        )
    ])

    fig.update_layout(
        title="🧠 3D agyi aktiváció – domborzati modell",
        scene=dict(
            xaxis_title="Neuron X",
            yaxis_title="Neuron Y",
            zaxis_title="Aktiváció",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(nticks=6, range=[0, np.max(z) + 1])
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_dark"
    )
    return fig

def run():
    st.title("🧠 Belátás alapú tanulás – Insight Learning szimuláció")

    st.markdown("""
    Ez a modul egy egyszerű környezetben modellezi a **belátás alapú tanulást**, ahol az ügynök egy ponton _hirtelen_ megérti, hogyan érheti el a célt.
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

    # === TUDOMÁNYOS RÉSZ – FRISSÍTVE LATEX-KÉPLETEKKEL ===
    st.markdown("### 📚 Tudományos háttér – Mi az a belátásos tanulás?")
    st.markdown(r"""
A **belátásos tanulás** (insight learning) egy kognitív folyamat, ahol a megoldás **nem fokozatos tanulással**,  
hanem hirtelen, **strukturált megértéssel** jelenik meg.

#### 🧠 Modell képletesen:
A probléma belátásos megoldása:

$$
\text{Megértés} = \lim_{t \to t_c} \left[ \sum_{i=0}^{t} f(p_i, e_i) \right]
$$

Ahol:
- \( p_i \): az aktuális pozíció vagy mentális állapot
- \( e_i \): környezeti inger vagy esemény
- \( t_c \): a belátás időpillanata

#### 🧪 Klasszikus példák:
- Köhler csimpánzkísérletei: eszközhasználat hirtelen fellépő megértéssel
- Emberi probléma-megoldás: “aha-élmény”

#### 📊 A modellben:
- Az aktivációs térkép a viselkedési stratégiák fejlődését mutatja
- A belátás aktiválása drámaian csökkentheti a célhoz vezető lépések számát
    """)

# 🔁 ReflectAI kompatibilitás
app = run
