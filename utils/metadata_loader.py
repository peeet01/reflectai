import json
import os

def load_metadata(module_key):
    filename = module_key.lower().replace(" ", "_") + ".json"
    filepath = os.path.join("data", "meta", filename)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return {"version": "0.0", "author": "Ismeretlen"}
