import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

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

def plot_brain_activity(activation_map, grid_size):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(activation_map, cmap="plasma", interpolation='nearest')
    ax.set_title("🧠 Aktivációs térkép (neurális mintázat)")
    ax.set_xlabel("Neuron X")
    ax.set_ylabel("Neuron Y")
    fig.colorbar(im, ax=ax, label="Aktiváció gyakoriság")
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

    # 💡 Új agymodell vizualizáció
    st.markdown("### 🧠 Vizualizált agymodell – Aktivációs mintázat")
    selected_ep = st.slider("🧪 Válassz epizódot megfigyeléshez", 0, episodes - 1, episodes - 1)
    fig_brain = plot_brain_activity(activations[selected_ep], grid_size)
    st.pyplot(fig_brain)

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
        hanem egy _strukturális átlátás_ révén történik.

        ### 🐒 Köhler-féle csimpánz kísérlet:
        - Egy banán elérhetetlen, de eszköz segítségével mégis megszerezhető.
        - A megoldás **nem fokozatos**, hanem **hirtelen jelentkezik**.

        A szimulált aktivációs térkép azt reprezentálja, hogy az „agy” mely régiói (pozíciói) milyen gyakran voltak aktívak a sikeres vagy sikertelen keresés során.
        """)

# Kötelező ReflectAI-kompatibilitás
app = run
