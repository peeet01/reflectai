import json
import os

def load_metadata(module_key: str):
    # Összeállítjuk a relatív fájlutat
    base_path = os.path.join("data", "meta")
    filename = f"{module_key.lower().replace(' ', '_')}.json"
    file_path = os.path.join(base_path, filename)

    # Fájl betöltése
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata
    except FileNotFoundError:
        return {"error": f"Metadata file not found: {file_path}"}
    except json.JSONDecodeError:
        return {"error": f"Hibás JSON formátum a fájlban: {file_path}"}
