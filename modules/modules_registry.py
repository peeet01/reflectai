import importlib
import streamlit as st

MODULES = {
    "Kutatási napló": ("journal", "journal_module"),
    "Reflexió sablon": ("reflection_template", "reflection_template_module"),
    # Add hozzá a többi modult is, ahogy már megadtad korábban
}

def safe_run(module_key):
    try:
        module_name, function_name = MODULES[module_key]
        module = importlib.import_module(f"modules.{module_name}")
        getattr(module, function_name)()
    except Exception as e:
        st.error(f"❌ Hiba a(z) {module_key} modulban: {e}")
