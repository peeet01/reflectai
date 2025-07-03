"""
Fractal Dimension Module – Box Counting módszerrel.

Ez a modul lehetővé teszi különféle pontfelhők fraktál dimenziójának becslését
a box counting algoritmus segítségével, valamint a zajra való érzékenység vizsgálatát.

Felhasználási területek:
- Kaotikus rendszerek analízise
- Topológiai komplexitás jellemzése
- Mintázatok kvantitatív mérése nemlineáris dinamikában
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix

def generate_cloud(kind, n_points=300):
    if kind == "Szimmetrikus spirál":
        theta = np.linspace(0, 4 * np.pi, n_points)
        r = np.linspace(0.1, 1, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.vstack([x, y]).T
    elif kind == "Lorenz-projekció":
        # Lorenz attractor 2D-s vetítése
        x, y, z = [0], [1], [1.05]
        dt, a, b, c = 0.01, 10, 28, 8/3
        for _ in range(n_points):
            dx = a * (y[-1] - x[-1])
            dy = x[-1] * (b - z[-1]) - y[-1]
            dz = x[-1] * y[-1] - c * z[-1]
            x.append(x[-1] + dx * dt)
            y.append(y[-1] + dy * dt)
            z.append(z[-1] + dz * dt)
        return np.vstack([x, y]).T
    elif kind == "Random felhő":
        return np.random.rand(n_points, 2)
    else:
        return np.zeros((n_points, 2))

def box_counting(data, epsilons):
    N = []
    for eps in epsilons:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        bins = np.ceil((max_vals - min_vals) / eps).astype(int)
        grid = np.floor((data - min_vals) / eps).astype(int)
        unique_boxes = np.unique(grid, axis=0)
        N.append(len(unique_boxes))
    return N

def run():
    st.title("🧮 Fraktál Dimenzió – Box Counting módszerrel")

    st.markdown("""
A fraktál dimenzió egy kvantitatív mérőszám, amely azt mutatja meg, hogy egy geometriai objektum
milyen komplexitással tölti ki a teret. A **box counting** módszer segítségével egy mintázat fraktál dimenzióját
becsülhetjük meg.

A dimenzió kiszámítása a következő:

$$
D = \\lim_{\\varepsilon \\to 0} \\frac{\\log N(\\varepsilon)}{\\log (1/\\varepsilon)}
$$

Ahol:
- \( N(\\varepsilon) \): az ε méretű dobozok száma, amik lefedik a mintázatot
""")

    kind = st.selectbox("🔘 Pontfelhő típusa", ["Szimmetrikus spirál", "Lorenz-projekció", "Random felhő"])
    n_points = st.slider("📊 Pontok száma", 100, 1000, 300, step=50)
    eps_start = st.slider("📏 Kezdő ε (log10 skálán)", -2.0, 0.0, -1.0)
    eps_end = st.slider("📏 Végső ε (log10 skálán)", -1.5, -0.1, -0.5)
    steps = st.slider("📈 Lépések száma", 5, 20, 10)
    noise_level = st.slider("📉 Zajszint (%)", 0, 50, 0)

    # Adatok generálása
    data = generate_cloud(kind, n_points)
    if noise_level > 0:
        data += np.random.randn(*data.shape) * (noise_level / 100)

    # Box counting
    epsilons = np.logspace(eps_start, eps_end, steps)
    counts = box_counting(data, epsilons)
    logs = np.log(1 / epsilons)
    logN = np.log(counts)
    slope, intercept = np.polyfit(logs, logN, 1)

    # Vizualizáció
    st.subheader("🌀 Mintázat")
    fig1, ax1 = plt.subplots()
    ax1.scatter(data[:, 0], data[:, 1], s=5)
    ax1.set_title("Pontfelhő")
    ax1.axis("equal")
    st.pyplot(fig1)

    st.subheader("📐 Box Counting eredmény")
    fig2, ax2 = plt.subplots()
    ax2.plot(logs, logN, "o-", label="Mért értékek")
    ax2.plot(logs, slope * logs + intercept, "--", label=f"Illesztett egyenes (D ≈ {slope:.2f})")
    ax2.set_xlabel("log(1/ε)")
    ax2.set_ylabel("log(N(ε))")
    ax2.set_title("Box Counting")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📥 CSV export")
    df = pd.DataFrame({
        "epsilon": epsilons,
        "N(epsilon)": counts,
        "log(1/epsilon)": logs,
        "log(N(epsilon))": logN
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Eredmény letöltése CSV-ben", data=csv, file_name="box_counting_results.csv")

    st.markdown("### 📚 Tudományos háttér")
    st.markdown("""
A **fraktál dimenzió** nem feltétlenül egész szám – gyakran nemlineáris dinamikus rendszerekből származó adatok jellemzésére használják.  
A **box counting módszer** egy egyszerű, de hatékony eljárás a fraktál dimenzió közelítésére.

**Tipikus fraktál dimenziók:**
- Vonal: 1.0
- Felület: 2.0
- Kaotikus attraktor: 1.2 – 1.9 között

**Jelentősége:**
- A mintázat „komplexitásának” vagy „sűrűségének” kvantifikálása
- Topológiai és geometriai tulajdonságok leírása nemlineáris rendszerekben
- Zajérzékenység tesztelése és vizsgálata

Az ε (dobozméret) csökkentésével a lefedés finomabb lesz, és a log-log ábrán az egyenes meredeksége a fraktál dimenzió közelítését adja.
""")

app = run
