modules/lyapunov_spectrum.py

import streamlit as st import numpy as np import matplotlib.pyplot as plt

def lyapunov_exponent(f, x0, d0, n, r): x, d = x0, d0 le = 0.0 for i in range(n): dx = d dfx = (f(x + dx, r) - f(x, r)) / dx le += np.log(abs(dfx)) x = f(x, r) return le / n

def logistic_map(x, r): return r * x * (1 - x)

def run(): st.title("Lyapunov-spektrum kalkulátor") st.markdown("Ez a modul egy logisztikus leképezés Lyapunov-exponensét számítja ki.")

r_min = st.slider("r alsó határ", 2.5, 3.5, 2.9, step=0.01)
r_max = st.slider("r felső határ", 3.5, 4.0, 4.0, step=0.01)
num_r = st.slider("r felbontás", 100, 1000, 500, step=100)
n_iter = st.slider("Iterációk száma", 100, 2000, 1000, step=100)

r_vals = np.linspace(r_min, r_max, num_r)
les = [lyapunov_exponent(logistic_map, 0.5, 1e-8, n_iter, r) for r in r_vals]

fig, ax = plt.subplots()
ax.plot(r_vals, les, lw=1)
ax.axhline(0, color='red', linestyle='--')
ax.set_title("Lyapunov-exponens logisztikus leképezéshez")
ax.set_xlabel("r paraméter")
ax.set_ylabel("Lyapunov-exponens")
st.pyplot(fig)

st.success("Lyapunov-exponens számítás sikeres.")

