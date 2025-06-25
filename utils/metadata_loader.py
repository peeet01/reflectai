import json
import os

def load_metadata(file_name):
    file_path = os.path.join("data", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
