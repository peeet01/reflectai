import streamlit as st import importlib import os

--- Alkalmazás fejléc ---

st.set_page_config(page_title="ReflectAI - Modul Választó", layout="wide") st.title("🧠 ReflectAI - Tudományos Modul Választó") st.markdown("Válassz egy modult a bal oldali menüből.")

--- Modulok listázása ---

MODULE_PATH = "." module_files = [f for f in os.listdir(MODULE_PATH) if f.endswith(".py") and f not in ("reflectai_app.py", "init.py")]

Modul nevek szépen formázva

module_names = {f: f.replace(".py", "").replace("_", " ").title() for f in module_files} selected_label = st.sidebar.selectbox("📂 Modul kiválasztása:", list(module_names.values())) selected_file = [k for k, v in module_names.items() if v == selected_label][0]

--- Modul betöltése dinamikusan ---

module_name = selected_file.replace(".py", "") try: module = importlib.import_module(module_name) if hasattr(module, "main"): module.main() else: st.error(f"A(z) {module_name} modul nem tartalmaz main() függvényt.") except Exception as e: st.error(f"Hiba történt a modul betöltése során: {e}")

