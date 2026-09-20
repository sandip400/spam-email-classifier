"""
Train a spam/ham classifier on SMS/email text data.
Saves the fitted vectorizer + model to ../model.pkl
"""
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sms_spam.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": model}, f)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
