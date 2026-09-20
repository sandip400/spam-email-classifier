"""
Load the trained model and classify new messages.
Usage: python predict.py "Free entry! Win a prize now!!!"
"""
import sys
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["model"]


def predict(text: str):
    vectorizer, model = load_model()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]  # probability of being spam
    label = "SPAM" if pred == 1 else "HAM"
    return label, float(proba)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your message here"')
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    label, proba = predict(message)
    print(f"Message : {message}")
    print(f"Result  : {label}  (spam probability: {proba:.2%})")
