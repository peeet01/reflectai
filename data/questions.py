
import json
import random

def load_questions(path='data/questions.json'):
    """Betölti a kérdésbankot JSON fájlból."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Hiba: nem található a kérdésbank a következő útvonalon: {path}")
        return []
    except json.JSONDecodeError:
        print(f"Hiba: érvénytelen JSON formátum a {path} fájlban.")
        return []

def get_random_question(questions, level=None, theme=None):
    """
    Visszaad egy véletlenszerű kérdést a megadott szűrési feltételek szerint.
    - level: reflexió szintje (pl. 0–3)
    - theme: témakör (pl. 'önismeret')
    """
    filtered = questions
    if level is not None:
        filtered = [q for q in filtered if q.get('level') == level]
    if theme is not None:
        filtered = [q for q in filtered if q.get('theme') == theme]
    return random.choice(filtered) if filtered else None
