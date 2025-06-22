import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """CSV fájl beolvasása és cache-elése."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Hiba a fájl beolvasásakor: {e}")
    return None


def get_uploaded_data(required_columns=None, allow_default=False, default=None):
    """
    Általános adatbetöltő komponens:
    - Oldalsávban fájl feltöltése
    - Ellenőrzi a kötelező oszlopokat
    - Automatikus fallback default adatra (pl. lorenz, xor, fractal)
    - Session state-be ment
    """
    st.sidebar.subheader("📁 Adatfeltöltés")
    uploaded_file = st.sidebar.file_uploader("Tölts fel egy CSV fájlt", type=["csv"])

    df = load_data(uploaded_file)

    # Fallback, ha nincs fájl és engedélyezett
    if df is None and allow_default and default:
        st.sidebar.warning(f"⚠️ Nincs fájl, fallback: `{default}`")
        df = get_default_data(default)

    # Ellenőrzés
    if df is not None:
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                st.error(f"❌ Hiányzó oszlop(ok): {', '.join(missing)}")
                return None
        st.session_state["uploaded_df"] = df
        st.sidebar.success("✅ Adat betöltve")
    else:
        st.sidebar.info("📂 Várakozás fájl feltöltésére...")

    return df


def get_default_data(name):
    """Előre definiált példák: XOR, Lorenz, Fractal."""
    if name == "xor":
        return pd.DataFrame({
            "Input1": [0, 0, 1, 1],
            "Input2": [0, 1, 0, 1],
            "Target": [0, 1, 1, 0]
        })

    elif name == "lorenz":
        steps = 1000
        t = np.linspace(0, 40, steps)
        x = np.sin(t)
        y = np.cos(t)
        z = np.sin(0.5 * t)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    elif name == "fractal":
        size = 64
        np.random.seed(42)
        matrix = np.random.rand(size, size)
        return pd.DataFrame(matrix)

    else:
        st.warning(f"⚠️ Nincs ilyen nevű alapértelmezett adat: `{name}`")
        return None


def show_data_overview(df, title="📊 Feltöltött adat előnézete"):
    """Alapadatok megjelenítése: méret, hiányzó értékek, előnézet."""
    if df is not None:
        st.subheader(title)
        st.write("ℹ️ Méret:", df.shape)
        st.dataframe(df.head())

        if df.isnull().values.any():
            st.warning("⚠️ Hiányzó értékek találhatók az adathalmazban!")
    else:
        st.info("📂 Nincs elérhető adat az előnézethez.")


def run():
    """Opcionális menüpont – közvetlen adatfeltöltéshez."""
    st.title("📁 Adatfeltöltés")
    st.markdown("""
    Tölts fel CSV fájlt, vagy használj alapértelmezett mintákat (XOR, Lorenz, Fractal).
    """)

    df = get_uploaded_data(allow_default=True, default="xor")
    if df is not None:
        show_data_overview(df)
    else:
        st.info("Nem történt betöltés.")
