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

    for episode in range(episodes):
        pos = agent_pos.copy()
        path = [tuple(pos)]
        for _ in range(max_steps):
            if pos == goal_pos:
                found = True
                break
            if use_insight and pos[1] < obstacle_pos[1] and pos[0] == obstacle_pos[0]:
                pos[0] -= 1  # Insight: ugrás az akadály fölé
            else:
                if pos[1] < grid_size - 1:
                    pos[1] += 1
                elif pos[0] > 0:
                    pos[0] -= 1
            path.append(tuple(pos))
        steps.append(path)
        steps_to_goal.append(len(path))
    return steps, found, steps_to_goal

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

def run():
    st.title("🧠 Belátás alapú tanulás – Insight Learning szimuláció")

    st.markdown("""
    Ez a modul egy egyszerű környezetben modellezi a **belátás alapú tanulást**, ahol az ügynök egy ponton _hirtelen_ megérti, hogyan érheti el a célt.
    """)

    # Paraméterek
    grid_size = st.slider("🔲 Rács méret", 5, 15, 7)
    episodes = st.slider("🔁 Epizódok száma", 10, 200, 50, step=10)
    max_steps = st.slider("🚶‍♂️ Lépések epizódonként", 5, 50, 20)
    use_insight = st.checkbox("💡 Belátás aktiválása", value=True)

    # Állandó pozíciók
    agent_pos = [grid_size - 1, 0]
    goal_pos = [0, grid_size - 1]
    obstacle_pos = [grid_size // 2, grid_size // 2]

    # Szimuláció
    steps, found, steps_to_goal = simulate_learning(
        grid_size, agent_pos, goal_pos, obstacle_pos, episodes, max_steps, use_insight
    )

    # Környezet kirajzolása
    st.markdown("### 🌍 Környezet vizualizáció")
    fig_env = plot_environment(grid_size, steps, goal_pos, obstacle_pos)
    st.pyplot(fig_env)

    # Tanulási görbe
    st.markdown("### 📉 Lépések száma epizódonként")
    fig_steps, ax_steps = plt.subplots()
    ax_steps.plot(steps_to_goal, marker='o')
    ax_steps.set_xlabel("Epizód")
    ax_steps.set_ylabel("Lépések száma")
    ax_steps.set_title("Tanulási görbe")
    st.pyplot(fig_steps)

    if found:
        st.success("🎉 Az ügynök elérte a célt – belátás vagy stratégia révén!")
    else:
        st.warning("🤔 Az ügynök még nem találta meg a célt.")

    # Riport letöltés
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
        A **belátásos tanulás** (insight learning) olyan folyamat, amikor egy élőlény – gyakran váratlanul – _"megérti"_ a probléma megoldását.  
        Ez szemben áll a klasszikus **próba-szerencse** tanulással, és gyakran jellemző az emberekre és magasabb rendű állatokra is.

        ### 🐒 Köhler kísérlete:
        - A csimpánz egy bottal szerezte meg a távoli banánt
        - Nem véletlenszerű próbálgatással, hanem egy _hirtelen felismeréssel_

        Ez a szimuláció ennek egyszerű absztrakciója, és lehetővé teszi, hogy megfigyeljük:
        - Hogyan változik a tanulási teljesítmény
        - Miben más a belátás, mint a kondicionálás vagy megerősítéses tanulás
        """)

# Kötelező ReflectAI-kompatibilitás
app = run
