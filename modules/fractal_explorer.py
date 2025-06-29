import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import io
import base64

# 🔬 Mandelbrot fraktál számítás
def mandelbrot_set(width, height, zoom, x_center, y_center, max_iter):
    x_range = 3.5 / zoom
    y_range = 2.0 / zoom
    x = np.linspace(x_center - x_range / 2, x_center + x_range / 2, width)
    y = np.linspace(y_center - y_range / 2, y_center + y_range / 2, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    div_time = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        Z = Z**2 + C
        diverge = np.abs(Z) > 2
        div_now = diverge & (div_time == 0)
        div_time[div_now] = i
        Z[diverge] = 2

    return X, Y, div_time

# 💾 Kép mentése gombként
def get_image_download_link(fig, filename='mandelbrot.png'):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Kép letöltése</a>'
    return href

# 🎨 Modul fő belépési pontja
def app():
    st.title("🌀 Fractal Explorer – Mandelbrot")
    st.markdown("Fedezd fel a Mandelbrot-halmazt különböző nézőpontokból!")

    # 📘 Matematikai háttér
    with st.expander("📚 Matematikai háttér"):
        st.latex(r"Z_{n+1} = Z_n^2 + C")
        st.markdown("""
        A Mandelbrot-halmaz azon komplex számok halmaza, melyekre a fenti iterációs képlet nem divergens.
        Egy pont akkor része a halmaznak, ha |Z| nem haladja meg a 2-t **véges számú iteráció után sem**.
        Ez gyönyörű, önhasonló, végtelen komplexitású alakzatokat eredményez.
        """)

    # ⚙️ Paraméterek
    st.sidebar.header("🛠️ Beállítások")
    zoom = st.sidebar.slider("Zoom", 1.0, 100.0, 1.0, step=0.5)
    x_center = st.sidebar.slider("X középpont", -2.0, 2.0, -0.5, step=0.01)
    y_center = st.sidebar.slider("Y középpont", -2.0, 2.0, 0.0, step=0.01)
    max_iter = st.sidebar.slider("Iterációk száma", 50, 1000, 200, step=50)
    width = st.sidebar.slider("Szélesség (px)", 300, 1000, 600, step=100)
    height = st.sidebar.slider("Magasság (px)", 300, 1000, 400, step=100)
    show_3d = st.sidebar.checkbox("🌐 3D nézet")

    # 📂 Paraméterbetöltés (JSON vagy sablon)
    if st.sidebar.button("🔁 Alapértelmezett nézet"):
        zoom = 1.0
        x_center = -0.5
        y_center = 0.0

    X, Y, Z = mandelbrot_set(width, height, zoom, x_center, y_center, max_iter)

    if show_3d:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=cm.inferno, linewidth=0, antialiased=False)
        ax.set_title("Mandelbrot 3D magasságtérkép")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.set_zlabel("Iterációk (magasság)")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        ax.imshow(Z, cmap="inferno", extent=[X.min(), X.max(), Y.min(), Y.max()])
        ax.set_title("Mandelbrot-halmaz (2D)")
        ax.axis("off")
        st.pyplot(fig)

    # 💾 Letöltési link
    st.markdown(get_image_download_link(fig), unsafe_allow_html=True)

    # 👁️‍🗨️ Extra magyarázat
    with st.expander("ℹ️ Tudtad?"):
        st.markdown("""
        A Mandelbrot-halmaz pereme **végtelen bonyolultságú** – ha belenagyítasz, újabb és újabb mintázatok bukkanak fel.
        A halmaz minden pontja kapcsolatban áll más részekkel, ez a **kaotikus viselkedés** egyik gyönyörű példája.
        """)
