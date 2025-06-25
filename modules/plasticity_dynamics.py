import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.markdown("## Hebbian Plaszticitás Dinamikája")

    st.markdown(
        "A Hebbian-tanulás egy olyan szinaptikus módosítási szabály, ahol a szinapszis erőssége nő, "
        "ha a pre- és posztszinaptikus neuronok egyidejűleg aktívak. Itt egy egyszerű modellt használunk "
        "a tanulási dinamika szimulációjához."
    )

    # Paraméterek
    time_steps = st.slider("Időlépések száma", min_value=100, max_value=1000, value=500, step=50)
    learning_rate = st.slider("Tanulási ráta (η)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    decay = st.slider("Szinaptikus hanyatlás (λ)", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

    st.divider()

    # Inicializálás
    np.random.seed(42)
    w = 0.5  # Kezdeti szinaptikus súly
    weights = [w]

    # Pre- és posztszinaptikus aktivitás
    pre_activity = np.random.rand(time_steps)
    post_activity = np.random.rand(time_steps)

    # Dinamika
    for t in range(1, time_steps):
        dw = learning_rate * pre_activity[t] * post_activity[t] - decay * w
        w += dw
        weights.append(w)

    # Eredmények megjelenítése
    fig, ax = plt.subplots()
    ax.plot(weights, label="Szinaptikus súly")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Súly érték")
    ax.set_title("Hebbian szinaptikus súlyváltozás")
    ax.legend()
    st.pyplot(fig)
app = run
