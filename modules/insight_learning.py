import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def generate_problem(complexity):
    if complexity == "alacsony":
        return 2
    elif complexity == "közepes":
        return 3
    else:  # "magas"
        return 4


def simulate_trial(num_elements, insight_step, t):
    if t < insight_step:
        return np.random.rand() < 0.1
    else:
        return np.random.rand() < (0.5 + 0.1 * num_elements)


def run(trials=5, pause_time=1.0, complexity="közepes"):
    st.header("💡 Belátás-alapú tanulási szimuláció")

    num_elements = generate_problem(complexity)
    insight_step = np.random.randint(2, trials)
    st.markdown(f"🔍 **A belátás várhatóan a(z) {insight_step}. próbálkozás körül történik.**")

    success_history = []

    for t in range(1, trials + 1):
        success = simulate_trial(num_elements, insight_step, t)
        success_history.append(success)
        st.write(f"🧪 Próbálkozás {t}: {'✅ Sikeres' if success else '❌ Sikertelen'}")

    success_rate = np.mean(success_history)
    st.markdown(f"📈 **Sikerességi arány:** {success_rate:.2f}")

    fig, ax = plt.subplots()
    ax.plot(range(1, trials + 1), success_history, marker="o")
    ax.set_xlabel("Próbálkozás")
    ax.set_ylabel("Siker (1) / Sikertelenség (0)")
    ax.set_title("Belátás-alapú tanulás szimuláció")
    st.pyplot(fig)
