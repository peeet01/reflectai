
import streamlit as st import numpy as np import matplotlib.pyplot as plt import seaborn as sns

def generate_lattice(N): return np.zeros((N, N), dtype=int)

def drop_grain(grid, threshold=4): N = grid.shape[0] avalanche_sizes = []

i, j = np.random.randint(0, N), np.random.randint(0, N)
grid[i, j] += 1
avalanche = 0

unstable = True
while unstable:
    unstable = False
    to_topple = np.argwhere(grid >= threshold)
    for x, y in to_topple:
        grid[x, y] -= 4
        avalanche += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                grid[nx, ny] += 1
        unstable = True

avalanche_sizes.append(avalanche)
return avalanche_sizes

def simulate_avalanche(N, steps): grid = generate_lattice(N) all_sizes = [] for _ in range(steps): sizes = drop_grain(grid) all_sizes.extend(sizes) return all_sizes

def run(): st.title("🌋 Criticality Explorer – Önszerveződő Kritikalitás") st.markdown(""" Ez a modul az önszerveződő kritikalitás (SOC) jelenségét modellezi és szemlélteti egy egyszerű homokdombmodell (Bak-Tang-Wiesenfeld) segítségével.

**A cél:** megfigyelni, hogyan vezet egy egyszerű szabály a rendezetlenség és a rendezettség határán lévő kritikus viselkedéshez.
""")

N = st.slider("Rács mérete (N x N)", 10, 100, 25)
steps = st.slider("Szimuláció lépések száma", 100, 5000, 1000, step=100)

if st.button("Szimuláció futtatása"):
    with st.spinner("Szimuláció folyamatban..."):
        sizes = simulate_avalanche(N, steps)

    st.subheader("📉 Lavinaméret-eloszlás (log-log)")
    fig, ax = plt.subplots()
    counts, bins = np.histogram(sizes, bins=50)
    bins_center = (bins[:-1] + bins[1:]) / 2
    ax.loglog(bins_center, counts, marker='o', linestyle='none')
    ax.set_xlabel("Lavinaméret")
    ax.set_ylabel("Gyakoriság")
    ax.set_title("Skálafüggetlen eloszlás – Lavinaméretek")
    st.pyplot(fig)

    st.subheader("🧠 Tudományos háttér")
    st.markdown("""
    Az önszerveződő kritikalitás (SOC) olyan rendszerek jellemzője, amelyek spontán kritikus állapotba kerülnek anélkül, hogy külső paraméterhangolás szükséges lenne.

    A modell alapképlete:

P(s) \propto s^{-\tau}

ahol $s$ a lavinaméret és $\tau$ egy jellemző kitevő (tipikusan 1.5–2.0).

    **Következtetések:**
    - A hálózat folyamatosan a rendezettség és káosz határán működik.
    - A tanulási és feldolgozási képességek maximálisak lehetnek ebben az állapotban.
    - Hasznos neuromorf számítástechnikában és agykutatásban.
    """)

    st.success("Szimuláció befejezve!")

app = run

