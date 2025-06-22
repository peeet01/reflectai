import streamlit as st
import pandas as pd

# 🔄 Cache-elt CSV beolvasás
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Hiba a fájl beolvasásakor: {e}")
    return None

# 📥 Adatfeltöltés és validálás
def get_uploaded_data(required_columns=None, allow_default=False, default=None):
    """
    Fájlbetöltés, oszlop-ellenőrzés és opcionális fallback támogatás.
    """
    st.sidebar.subheader("📁 Adatfeltöltés")
    uploaded_file = st.sidebar.file_uploader("Tölts fel egy CSV fájlt", type=["csv"])

    df = load_data(uploaded_file)

    if df is not None:
        st.sidebar.success("✅ Fájl betöltve")
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                st.error(f"❌ Hiányzó oszlop(ok): {', '.join(missing)}")
                return None
    else:
        st.sidebar.info("📂 Várakozás fájl feltöltésére...")
        if allow_default and default:
            st.sidebar.warning(f"⚠️ Alapértelmezett adat használata: `{default}`")
            df = get_default_data(default)

    if df is not None:
        st.session_state["uploaded_df"] = df
    return df

# 🧰 Alapértelmezett adat generálás
def get_default_data(name):
    import numpy as np
    if name == "xor":
        return pd.DataFrame({
            "Input1": [0, 0, 1, 1],
            "Input2": [0, 1, 0, 1],
            "Target": [0, 1, 1, 0]
        })
    elif name == "lorenz":
        steps = 1000
        xs = np.sin(np.linspace(0, 50, steps))
        ys = np.cos(np.linspace(0, 50, steps))
        zs = np.sin(np.linspace(0, 50, steps * 0.5))
        return pd.DataFrame({"x": xs, "y": ys, "z": zs})
    else:
        st.warning("⚠️ Nincs ilyen nevű alapértelmezett adathalmaz.")
        return None

# 👁️ Adatok előnézete
def show_data_overview(df, title="📊 Feltöltött adat előnézete"):
    if df is not None:
        st.subheader(title)
        st.write("ℹ️ Adatok mérete:", df.shape)
        st.dataframe(df.head())

        if df.isnull().values.any():
            st.warning("⚠️ Hiányzó értékek találhatók az adathalmazban!")
    else:
        st.info("📂 Nincs elérhető adat az előnézethez.")

# 🚀 Streamlit oldal futtatásához (ha menüből hívod)
def run():
    st.title("📁 Adatfeltöltő modul")
    st.markdown("""
    Tölts fel CSV fájlt, vagy használj előre definiált (alapértelmezett) adatkészletet pl. XOR vagy Lorenz.
    """)

    df = get_uploaded_data(allow_default=True, default="xor")

    if df is not None:
        show_data_overview(df)
    else:
        st.info("Nincs betöltött vagy alapértelmezett adat.")
