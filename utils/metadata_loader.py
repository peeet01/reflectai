import streamlit as st
from modules.metadata_loader import get_file_metadata, get_csv_metadata

uploaded_file = st.file_uploader("Adj meg egy CSV fájlt")

if uploaded_file:
    # Mentés ideiglenesen
    temp_path = f\"temp_{uploaded_file.name}\"
    with open(temp_path, \"wb\") as f:
        f.write(uploaded_file.getbuffer())

    meta = get_file_metadata(temp_path)
    csv_meta = get_csv_metadata(temp_path)

    st.subheader(\"📁 Fájl metaadatai:\")
    st.json(meta)
    st.subheader(\"🧬 CSV szerkezeti metaadatok:\")
    st.json(csv_meta)
