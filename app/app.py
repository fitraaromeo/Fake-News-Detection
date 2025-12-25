import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from src.preprocessing import clean_text
from src.infer_lstm import LSTMInfer
from src.infer_transformer import TransformerInfer

st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection (LSTM vs BERT vs DistilBERT)")

# paths (lokal)
LSTM_MODEL_PATH = "models/lstm_best.h5"
LSTM_TOKENIZER_PATH = "models/lstm_tokenizer.joblib"
BERT_DIR = "models/bert/final"
DISTIL_DIR = "models/distilbert/final"

@st.cache_resource
def load_models():
    lstm = LSTMInfer(LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH)
    bert = TransformerInfer(BERT_DIR)
    distil = TransformerInfer(DISTIL_DIR)
    return lstm, bert, distil

lstm_model, bert_model, distil_model = load_models()

model_choice = st.selectbox("Choose Model", ["LSTM (Non-Pretrained)", "BERT (Transfer Learning)", "DistilBERT (Transfer Learning)"])
user_text = st.text_area("Input news text (title + content recommended):", height=220)

col1, col2 = st.columns([1, 1])

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    text = clean_text(user_text)

    if model_choice.startswith("LSTM"):
        pred, prob_fake = lstm_model.predict(text)
        label = "FAKE" if pred == 1 else "REAL"
        col1.metric("Prediction", label)
        col2.metric("Probability (Fake)", f"{prob_fake:.4f}")

    elif model_choice.startswith("BERT"):
        pred, probs, prob_pred = bert_model.predict(text)
        label = "FAKE" if pred == 1 else "REAL"
        col1.metric("Prediction", label)
        col2.write({"prob_real": float(probs[0]), "prob_fake": float(probs[1])})

    else:
        pred, probs, prob_pred = distil_model.predict(text)
        label = "FAKE" if pred == 1 else "REAL"
        col1.metric("Prediction", label)
        col2.write({"prob_real": float(probs[0]), "prob_fake": float(probs[1])})

st.caption("Labels: 0 = Real, 1 = Fake • Preprocessing: lowercase, remove URL, remove non-letters.")