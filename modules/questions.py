import json
import os
import random

def load_questions(filepath="data/questions.json"):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_random_question(questions):
    if not questions:
        return None
    return random.choice(questions)

