import streamlit as st from modules_registry import MODULES from utils.metadata_loader import load_metadata

st.set_page_config(page_title="Neurolab AI – Scientific Playground Sandbox", page_icon="🧠", layout="wide")

st.title("Neurolab AI – Scientific Playground Sandbox") st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")

Megjegyzés / napló lehetőség

st.text_input("Megfigyelés vagy jegyzet (opcionális):")

Modulválasztó

st.sidebar.title("Modulválasztó") module_name = st.sidebar.radio("Kérlek válassz:", list(MODULES.keys()))

Modul kulcs visszafejtése a modulnévből (kulcs -> cím mapping alapján)

selected_module_key = None for key, data in MODULES.items(): if data['title'] == module_name: selected_module_key = key break

if selected_module_key is None: st.error("Nem található a kiválasztott modul.") else: module_data = MODULES[selected_module_key] try: metadata = load_metadata(selected_module_key)

# Metaadatok megjelenítése opcionálisan
    with st.expander("ℹ️ Tudományos háttér és metaadatok", expanded=False):
        st.subheader(metadata.get("title", module_name))
        st.markdown(metadata.get("description", "Nincs leírás."))
        if metadata.get("equations"):
            st.markdown("**Képletek:**")
            for eq in metadata["equations"]:
                st.latex(eq)
        if metadata.get("parameters"):
            st.markdown("**Paraméterek:**")
            for name, desc in metadata["parameters"].items():
                st.markdown(f"- **{name}**: {desc}")
        if metadata.get("applications"):
            st.markdown("**Alkalmazási területek:**")
            for app in metadata["applications"]:
                st.markdown(f"- {app}")

    # Modul futtatása
    module_data['run']()

except Exception as e:
    st.error(f"❌ Hiba

