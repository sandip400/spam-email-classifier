<<<<<<< HEAD
# spam-email-classifier
A Python web app that uses TF-IDF + Naive Bayes to classify messages as Spam or Ham. Built with scikit-learn and Streamlit, it provides instant spam detection through a simple browser interface.
=======
# Spam Email/SMS Classifier

A TF-IDF + Naive Bayes classifier that flags messages as spam or ham (legitimate),
with a Streamlit web app for interactive use.

## Project structure
```
spam-email-classifier/
├── data/
│   └── sms_spam.csv        # 5,574 labeled SMS messages (label, message)
├── src/
│   ├── train.py             # trains model, prints metrics, saves model.pkl
│   └── predict.py           # CLI: classify a single message
├── app.py                   # Streamlit web app
├── requirements.txt
├── model.pkl                # created after you run train.py (not included yet)
└── README.md
```

## Setup
```bash
cd spam-email-classifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model
```bash
python src/train.py
```
This reads `data/sms_spam.csv`, trains a TF-IDF + MultinomialNB pipeline,
prints accuracy/precision/recall/F1, and saves `model.pkl` in the project root.

Expect ~97-98% accuracy on this dataset out of the box.

## Try it from the command line
```bash
python src/predict.py "Congratulations! You've won a free iPhone, click here now!"
```

## Run the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Deploy for free
1. Push this folder to a public GitHub repo (include `model.pkl` after training,
   or add a build step that runs `train.py` on startup).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud), sign in with GitHub.
3. "New app" → select the repo → set main file to `app.py` → Deploy.
4. You'll get a public URL in a couple of minutes.

## Dataset
`data/sms_spam.csv` is the well-known **SMS Spam Collection** dataset
(5,574 messages, ~13% spam). Swap in your own CSV with `label,message`
columns (`label` = "spam" or "ham") to retrain on different data — e.g.
the Enron email spam corpus for full email text instead of SMS.

## Notes
- Naive Bayes was chosen for speed and strong baseline performance on text
  classification; swapping in `LinearSVC` or `LogisticRegression` from
  scikit-learn is a one-line change in `train.py` if you want to compare.
- For a production system, retrain periodically as spam patterns evolve.
>>>>>>> e83d988 (Initial commit: Spam email classifier app and model)
