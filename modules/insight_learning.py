
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def simulate_environment(step, insight_step):
    if step < insight_step:
        return 0
    else:
        return 1


def run(hidden_size=5, learning_rate=0.1, epochs=100, note=""):
    st.header("🧠 Belátás-alapú tanulás szimuláció")

    total_steps = 100
    insight_step = np.random.randint(20, 80)

    performance = []
    memory_strength = 0.0

    st.markdown(f"🔍 **Belátás pillanata várhatóan a(z) {insight_step}. lépés körül.**")

    for t in range(total_steps):
        outcome = simulate_environment(t, insight_step)

        if outcome == 1:
            memory_strength += learning_rate * (1 - memory_strength)
        else:
            memory_strength -= learning_rate * memory_strength * 0.3

        performance.append(memory_strength)

    fig, ax = plt.subplots()
    ax.plot(performance, label="Megoldás sikeressége", color="purple")
    ax.axvline(insight_step, color="red", linestyle="--", label="Belátás pontja")
    ax.set_xlabel("Idő (lépések)")
    ax.set_ylabel("Megtanult sikeresség")
    ax.set_title("🧠 Insight Learning szimuláció")
    ax.legend()
    st.pyplot(fig)

    if note:
        st.subheader("🗒️ Megjegyzés:")
        st.info(note)
