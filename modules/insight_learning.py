import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


def simulate_environment(step, insight_step):
    if step < insight_step:
        return 0
    else:
        return 1


def run(trials=5, pause_time=1.0, complexity="közepes"):
    st.header("🧠 Belátás-alapú tanulás szimuláció")

    total_steps = 100
    insight_step = np.random.randint(20, 80)

    performance = []
    memory_strength = 0.0

    st.markdown(f"🔍 **Belátás pillanata várhatóan a(z) {insight_step}. lépés körül.**")
    st.markdown(f"🧪 **Próbálkozások száma:** {trials}")
    st.markdown(f"⚙️ **Komplexitás:** {complexity}")
    st.markdown(f"⏱️ **Pihenőidő próbálkozások között:** {pause_time} másodperc")

    for trial in range(trials):
        performance = []
        memory_strength = 0.0
        for t in range(total_steps):
            outcome = simulate_environment(t, insight_step)
            if outcome == 1:
                memory_strength += 0.1 * (1 - memory_strength)
            else:
                memory_strength -= 0.1 * memory_strength * 0.3
            performance.append(memory_strength)

        st.markdown(f"### 🔁 Próbálkozás {trial + 1}")
        fig, ax = plt.subplots()
        ax.plot(performance, label="Megoldás sikeressége", color="purple")
        ax.axvline(insight_step, color="red", linestyle="--", label="Belátás pontja")
        ax.set_xlabel("Idő (lépések)")
        ax.set_ylabel("Megtanult sikeresség")
        ax.set_title(f"🧠 Insight Learning szimuláció – {trial + 1}. kör")
        ax.legend()
        st.pyplot(fig)

        time.sleep(pause_time)
