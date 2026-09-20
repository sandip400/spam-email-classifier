"""
Streamlit web app for the Spam/Ham classifier.
Run locally:   streamlit run app.py
Deploy free:   push repo to GitHub -> streamlit.io/cloud -> point at this file
"""
import pickle
from pathlib import Path
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam / Ham Message Classifier")
st.write("Paste an email or SMS message below to check if it's spam.")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["model"]


if not MODEL_PATH.exists():
    st.error("model.pkl not found. Run `python src/train.py` first to train and save the model.")
    st.stop()

vectorizer, model = load_model()

message = st.text_area("Message text", height=150, placeholder="Type or paste a message...")

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][1]

        if pred == 1:
            st.error(f"🚨 This looks like SPAM ({proba:.1%} confidence)")
        else:
            st.success(f"✅ This looks like a legitimate message ({(1 - proba):.1%} confidence)")

        st.progress(float(proba))
        st.caption(f"Spam probability: {proba:.2%}")

st.divider()
st.caption("Model: TF-IDF + Multinomial Naive Bayes, trained on the SMS Spam Collection dataset.")
