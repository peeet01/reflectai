import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.markdown("## 🧠 Belátás alapú tanulás (Insight Learning)")
    st.markdown(
        "Ez a modul egy egyszerű környezetben mutatja be a belátás alapú tanulást, "
        "ahol egy ügynök hirtelen megtalálja a megoldást a próbálkozások után."
    )

    st.markdown("### 🧩 Feladat leírása")
    st.write("Az ügynöknek át kell ugrania egy akadályt, hogy elérje a célt.")

    grid_size = 7
    env = np.zeros((grid_size, grid_size))
    agent_pos = [grid_size - 1, 0]
    goal_pos = [0, grid_size - 1]
    obstacle_pos = [3, 3]

    env[tuple(goal_pos)] = 2
    env[tuple(obstacle_pos)] = -1
    env[tuple(agent_pos)] = 1

    st.markdown("### 🔁 Tanulás futtatása")

    steps = []
    found = False
    for episode in range(50):
        pos = agent_pos.copy()
        path = [tuple(pos)]
        for _ in range(20):
            if pos == goal_pos:
                found = True
                break
            if pos[1] < obstacle_pos[1] and pos[0] == obstacle_pos[0]:
                pos[0] -= 1  # Ugrás
            else:
                if pos[1] < grid_size - 1:
                    pos[1] += 1
                elif pos[0] > 0:
                    pos[0] -= 1
            path.append(tuple(pos))
        steps.append(path)

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

    st.pyplot(fig)

    if found:
        st.success("🎉 Az ügynök megtalálta a célt – belátás révén ugrással kerülte ki az akadályt.")
    else:
        st.warning("🤔 Az ügynök még nem talált megoldást.")
