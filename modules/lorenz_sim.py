
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def run():
    st.subheader("Lorenz attraktor")
    dt = 0.01
    steps = 10000
    xyz = np.empty((steps, 3))
    xyz[0] = (0., 1., 1.05)
    sigma, rho, beta = 10., 28., 8./3.
    for i in range(1, steps):
        x, y, z = xyz[i - 1]
        xyz[i] = (
            x + sigma * (y - x) * dt,
            y + (x * (rho - z) - y) * dt,
            z + (x * y - beta * z) * dt
        )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*xyz.T, lw=0.5)
    st.pyplot(fig)
