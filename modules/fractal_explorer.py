import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
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

# 💾 Kép mentése linkké
def get_image_download_link(fig, filename='mandelbrot.png'):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Kép letöltése</a>'
    return href

# 🚀 Fő modul futtatása
def run():
    st.title("🌀 Mandelbrot Fraktálvizualizáció")

    st.markdown("""
A Mandelbrot-halmaz az egyik legismertebb fraktál, amely a komplex számok síkján jön létre,  
önhasonló mintázatokkal és végtelen komplexitással.

Ez a modul lehetővé teszi a Mandelbrot-halmaz felfedezését 2D és interaktív 3D formában,  
valamint különböző paraméterek vizsgálatát.
""")

    st.subheader("🔧 Paraméterek")

    col1, col2 = st.columns(2)
    with col1:
        zoom = st.slider("Zoom", 1.0, 100.0, 1.0, step=0.5)
        x_center = st.slider("X középpont", -2.0, 2.0, -0.5, step=0.01)
        y_center = st.slider("Y középpont", -2.0, 2.0, 0.0, step=0.01)
    with col2:
        max_iter = st.slider("Iterációk száma", 50, 1000, 200, step=50)
        width = st.slider("Szélesség (px)", 300, 800, 600, step=100)
        height = st.slider("Magasság (px)", 300, 800, 400, step=100)
        show_3d = st.checkbox("🌐 Interaktív 3D nézet (Plotly)")

    if st.button("🔁 Alapértelmezett nézet"):
        zoom = 1.0
        x_center = -0.5
        y_center = 0.0

    st.info("🔄 Fraktál generálása folyamatban...")
    X, Y, Z = mandelbrot_set(width, height, zoom, x_center, y_center, max_iter)

    if show_3d:
        st.subheader("🌐 Interaktív 3D Mandelbrot-halmaz")
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Inferno')])
        fig.update_layout(
            scene=dict(
                xaxis_title="Re(z)",
                yaxis_title="Im(z)",
                zaxis_title="Iterációk",
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("🖼️ Kép (2D)")
        fig, ax = plt.subplots()
        ax.imshow(Z, cmap="inferno", extent=[X.min(), X.max(), Y.min(), Y.max()])
        ax.set_title("Mandelbrot-halmaz (2D)")
        ax.axis("off")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig), unsafe_allow_html=True)

    st.subheader("📚 Tudományos háttér")

    st.latex(r"Z_{n+1} = Z_n^2 + C")

    st.markdown("""
A **Mandelbrot-halmaz** azon komplex számok gyűjteménye, amelyekre a következő iteratív képlet:

$$
Z_{n+1} = Z_n^2 + C
$$

nem divergens, azaz a sorozat tagjai nem tartanak a végtelenhez.  
Egy adott \( C \in \mathbb{C} \) szám akkor része a Mandelbrot-halmaznak, ha a sorozat \( |Z_n| \le 2 \) marad végtelen sok iteráció után is.

#### Jellemzők:
- Végtelen részletesség
- Önmagába ágyazott mintázatok (önhasonlóság)
- Fraktál dimenziója meghaladja a 2-t

A Mandelbrot-halmaz minden pontja egyfajta **stabilitási teszt** eredménye, amely a komplex számokat a kaotikusság szempontjából osztályozza.
""")

# Kötelező ReflectAI-kompatibilitás
app = run
