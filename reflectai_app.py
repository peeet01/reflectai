import streamlit as st
import yaml
import streamlit_authenticator as stauth
from modules.modules_registry import MODULES, safe_run
from utils.metadata_loader import load_metadata

# Alkalmazás beállítása
st.set_page_config(page_title="Neurolab AI – Scientific Playground Sandbox", page_icon="🧠", layout="wide")

# Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.safe_load(file)

# Hitelesítés inicializálása
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Bejelentkezés megjelenítése
name, authentication_status, username = authenticator.login("main", "Bejelentkezés")

# Hitelesítési állapot alapján megjelenítés
if authentication_status is False:
    st.error("Hibás felhasználónév vagy jelszó")
elif authentication_status is None:
    st.warning("Kérlek jelentkezz be a folytatáshoz")
elif authentication_status:
    # Fő tartalom csak hitelesített felhasználónak
    authenticator.logout("Kijelentkezés", "sidebar")

    st.title("🧠 Neurolab AI – Scientific Playground Sandbox")
    st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
    st.text_input("📝 Megfigyelés vagy jegyzet (opcionális):")

    # Modulválasztó
    st.sidebar.title("🗂️ Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz egy modult:", list(MODULES.keys()))

    # Metaadat betöltés
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)

    # Metaadat megjelenítés
    st.subheader(f"📘 {metadata.get('title', module_key)}")
    st.write(metadata.get("description", ""))

    if metadata.get("equations"):
        st.markdown("#### 🧮 Egyenletek:")
        for eq in metadata["equations"]:
            st.latex(eq)

    if metadata.get("parameters"):
        st.markdown("#### 🎛️ Paraméterek:")
        for param, desc in metadata["parameters"].items():
            st.markdown(f"- **{param}**: {desc}")

    if metadata.get("applications"):
        st.markdown("#### 🔬 Alkalmazási területek:")
        for app in metadata["applications"]:
            st.markdown(f"- {app}")

    st.divider()

    # Modul futtatása
    safe_run(module_key)
