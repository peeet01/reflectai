import json
import os

def load_metadata(module_key):
    base_path = os.path.join("data", "meta")
    filepath = os.path.join(base_path, f"{module_key}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "title": module_key,
            "description": "Nincs elérhető leírás ehhez a modulhoz.",
            "equations": [],
            "parameters": {},
            "applications": []
        }
