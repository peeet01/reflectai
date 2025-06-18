import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("Lorenz szimuláció")
    st.write("Lorenz szimuláció modul fut.")

    sigma = 10
    rho = 28
    beta = 8 / 3
    dt = 0.01
    steps = 10000

    xs = np.empty(steps)
    ys = np.empty(steps)
    zs = np.empty(steps)

    xs[0], ys[0], zs[0] = 0., 1., 1.05

    for i in range(1, steps):
        x, y, z = xs[i-1], ys[i-1], zs[i-1]
        xs[i] = x + dt * sigma * (y - x)
        ys[i] = y + dt * (x * (rho - z) - y)
        zs[i] = z + dt * (x * y - beta * z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(xs, ys, zs)
    st.pyplot(fig)
