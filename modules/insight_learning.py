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
    # Ha még nincs belátás
    if t < insight_step:
        return np.random.rand() < 0.1  # 10% esély
    else:
        return np.random.rand() < (0.5 + 0.1 * num_elements)  # nagyobb esély a sikerre


def run(trials=5, pause_time=1.0, complexity="közepes"):
    st.header("💡 Belátás-alapú tanulási szimuláció")

    num_elements = generate_problem(complexity)
    insight_step = np.random.randint(2, trials)
    st.markdown(f"🔍 **A belátás várhatóan a(z) {insight_step}. próbálkozás körül történik.**")

    success_history = []
    log_messages = []

    for t in range(1
