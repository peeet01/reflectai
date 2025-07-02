import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BCM tanulási szabály
def bcm_learning(x, eta=0.01, tau=100, w0=0.5, theta0=0.1, steps=500):
    w = w0
    theta = theta0
    w_hist, theta_hist, y_hist = [], [], []

    for t in range(steps):
        y = w * x[t]
        dw = eta * x[t] * y * (y - theta)
        dtheta = (y**2 - theta) / tau
        w += dw
        theta += dtheta
        w_hist.append(w)
        theta_hist.append(theta)
        y_hist.append(y)

    return np.array(w_hist), np.array(theta_hist), np.array(y_hist)

# Jelgenerátor
def generate_input_signal(kind, length, amplitude=1.0, noise_level=0.0):
    t = np.linspace(0, 10, length)
    if kind == "Szinusz":
        signal = amplitude * np.sin(2 * np.pi * t)
    elif kind == "Fehér zaj":
        signal = np.random.randn(length)
    elif kind == "Lépcsős":
        signal = amplitude * np.where(t % 2 < 1, 1, 0)
    else:
        signal = np.zeros(length)
    noise = noise_level * np.random.randn(length)
    return signal + noise

# ✅ A run() függvény
def run():
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
Ez a modul a **BCM (Bienenstock–Cooper–Munro)** tanulási szabály működését szemlélteti, amely a szinaptikus módosulásokat egy dinamikusan változó küszöbön keresztül modellezi.
    """)

    # ⚙️ Paraméterek
    signal_type = st.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    tau = st.slider("Küszöb időállandó (τ)", 10, 500, 100, step=10)
    w0 = st.slider("Kezdeti súly (w₀)", -2.0, 2.0, 0.5, step=0.1)
    theta0 = st.slider("Kezdeti küszöb (θ₀)", 0.0, 1.0, 0.1, step=0.05)
    amplitude = st.slider("Jel amplitúdó", 0.1, 2.0, 1.0, step=0.1)
    noise_level = st.slider("Zaj szint", 0.0, 1.0, 0.0, step=0.05)

    # 📊 Szimuláció
    x = generate_input_signal(signal_type, steps, amplitude, noise_level)
    w, theta, y = bcm_learning(x, eta, tau, w0, theta0, steps)

    # 📈 2D grafikon
    st.subheader("📈 Tanulási dinamika")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)")
    ax.set_xlabel("Idő")
    ax.set_title("BCM tanulás időfüggvényei")
    ax.legend()
    st.pyplot(fig)

    # 📥 Export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({"x": x, "y": y, "w": w, "θ": theta})
    csv = df.to_csv(index_label="idő").encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="bcm_learning.csv")

    # 📘 Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **BCM (Bienenstock–Cooper–Munro) szabály** egy biológiai inspirációjú tanulási mechanizmus, amelyet a vizuális kéreg fejlődésének modellezésére hoztak létre. Az elmélet kulcsa, hogy a szinaptikus súlyok változása nemcsak a pre- és posztszinaptikus aktivitástól függ, hanem egy **dinamikusan változó küszöbtől** (θ) is.

#### 🧮 Formális leírás:

- **Súlyváltozás:**  
  \( \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta) \)  
  A tanulás akkor történik, ha a posztszinaptikus aktivitás (y) nagyobb, mint a küszöb (θ). Ez megerősíti a kapcsolatot. Ha kisebb, a súly gyengül.

- **Küszöbszint változása:**  
  \( \frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta) \)  
  Ez az adaptív küszöb biztosítja a **homeosztatikus stabilitást**, vagyis nem engedi a rendszer instabil tanulásba futni.

---

### 🧩 Csúszkák magyarázata – Mit állítasz be?

- **Bemeneti jel típusa:**  
  Kiválasztható szinusz, zaj vagy lépcsős jel. Ezek különböző típusú ingerléseket modelleznek:
  - *Szinusz*: periodikus bemenet (pl. hang vagy fényhullám)
  - *Zaj*: kaotikus környezeti input
  - *Lépcsős*: inger-válasz típusú bemenetek

- **Szimuláció lépései:**  
  A tanulási folyamat időtartama. Minél több lépés, annál több változás látszik az eredményekben.

- **Tanulási ráta (η):**  
  Ez szabályozza, hogy milyen gyorsan változik a szinaptikus súly.  
  Túl magas érték instabilitást, túl alacsony lassú tanulást okozhat.

- **Küszöb időállandó (τ):**  
  Ez határozza meg, hogy a tanulási küszöb (θ) milyen gyorsan alkalmazkodik.  
  Nagyobb τ → lassabb alkalmazkodás, stabilabb tanulás.

- **Kezdeti súly (w₀):**  
  A szinaptikus kapcsolat induló erőssége.  
  Befolyásolhatja, hogy az első néhány lépésben milyen gyorsan indul meg a tanulás.

- **Kezdeti küszöb (θ₀):**  
  A tanulási küszöb induló értéke. Ha túl magas, sokáig nem történik tanulás.

- **Jel amplitúdó:**  
  A bemeneti jel intenzitása.  
  Nagyobb amplitúdó erőteljesebb válaszokat vált ki → gyorsabb és élesebb tanulási görbék.

- **Zaj szint:**  
  Véletlenszerű komponens a jelhez adva.  
  Segíti a robusztusság tesztelését → ellenőrizheted, mennyire érzékeny a modell a környezeti zavarokra.

---

### 🎯 Alkalmazási területek:

- Szenzoros kéreg modellezése
- Látás és hallás fejlődési szimulációja
- Homeosztatikus tanulás és stabilizáció vizsgálata
- Érzékelő rendszerek adaptív vezérlése

A BCM tanulás révén a modell képes megtanulni *mikor érdemes tanulni* – azaz nemcsak a bemenetet veszi figyelembe, hanem a tanulás feltételeit is folyamatosan szabályozza.

    """)

# ✅ Modul regisztráláshoz:
app = run
