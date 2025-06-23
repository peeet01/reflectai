import json
import os
import random

def load_questions(filepath="data/questions.json"):
    """
    Betölti a kérdéseket egy JSON fájlból.
    
    Parameters:
        filepath (str): Az elérési út a kérdésfájlhoz (alapértelmezett: data/questions.json)
    
    Returns:
        list: A kérdések listája szótárként.
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def get_random_question(questions):
    """
    Véletlenszerű kérdés kiválasztása a listából.

    Parameters:
        questions (list): A betöltött kérdések listája.

    Returns:
        dict or None: Egy kérdés szótárként, vagy None, ha üres.
    """
    if not questions:
        return None
    return random.choice(questions)
