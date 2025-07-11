import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# === Boltzmann-eloszlás definíciója ===
def boltzmann_distribution(energy, T, k=1.0):
    return np.exp(-energy / (k * T))

# === Energia tér generálása ===
def generate_energy_surface(x_range, y_range, scale=1.0):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = scale * (X**2 + Y**2)
    return X, Y, Z

# === Main app ===
def run():
    st.set_page_config(layout="wide")
    st.title("🌡️ Boltzmann-eloszlás – Energia és valószínűségi eloszlás")

    st.markdown("""
    A **Boltzmann-eloszlás** leírja a részecskék energia szerinti eloszlását egy hőmérsékleten.  
    Az energia növekedésével a valószínűség exponenciálisan csökken.
    
    A következő vizualizáció egy 2D energiateret generál, amelyhez hozzárendeljük a Boltzmann-súlyokat, majd 3D-ben ábrázoljuk.
    """)

    # === Beállítások ===
    st.sidebar.header("⚙️ Paraméterek")
    T = st.sidebar.slider("Hőmérséklet (T)", 0.1, 5.0, 1.0, 0.1)
    energy_scale = st.sidebar.slider("Energiafelület skálázása", 0.1, 5.0, 1.0, 0.1)
    x_range = (-3, 3)
    y_range = (-3, 3)

    # === Energiafelület ===
    X, Y, E = generate_energy_surface(x_range, y_range, scale=energy_scale)
    P = boltzmann_distribution(E, T)
    
    # === 3D Plotly grafikon ===
    st.subheader("🌐 3D Boltzmann-eloszlás felület")
    fig3d = go.Figure(data=[go.Surface(
        x=X, y=Y, z=P,
        surfacecolor=E,
        colorscale='Viridis',
        colorbar=dict(title='Energia'),
        showscale=True
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Valószínűség',
        ),
        title="Boltzmann-eloszlás energiafüggvény mentén",
        margin=dict(l=0, r=0, t=60, b=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # === 2D metszet ===
    st.subheader("📈 2D metszet az energia mentén")
    E_1d = np.linspace(0, 10, 200)
    P_1d = boltzmann_distribution(E_1d, T)

    fig2d, ax = plt.subplots()
    ax.plot(E_1d, P_1d, color="crimson")
    ax.set_xlabel("Energia")
    ax.set_ylabel("Valószínűség")
    ax.set_title(f"Boltzmann-eloszlás (T = {T})")
    st.pyplot(fig2d)

    # === CSV letöltés ===
    st.subheader("📥 CSV export")
    df = pd.DataFrame({
        "X": X.flatten(),
        "Y": Y.flatten(),
        "Energia": E.flatten(),
        "Valószínűség": P.flatten()
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Letöltés (CSV)", data=csv, file_name="boltzmann_distribution.csv")

    # === Tudományos háttér ===
    st.markdown("### 📚 Tudományos háttér")
    st.latex(r"""
    P(E) = \frac{1}{Z} \exp\left(-\frac{E}{kT}\right)
    """)
    st.markdown("""
    - **\( P(E) \)**: valószínűség, hogy a rendszer \( E \) energiájú állapotban van  
    - **\( k \)**: Boltzmann-állandó (itt 1-nek tekintjük)  
    - **\( T \)**: hőmérséklet  
    - **\( Z \)**: partíciós függvény (összegzés minden lehetséges állapoton)

    Az eloszlás alapja a **termodinamika** és **statikus fizika** törvényein alapul, valamint széleskörű alkalmazása van:
    - Molekuláris dinamika
    - Anyagszerkezetek modellezése
    - Valószínűségi gépi tanulás (pl. Boltzmann-gépek)
    """)

# App futtatása ReflectAI-hez
app = run
