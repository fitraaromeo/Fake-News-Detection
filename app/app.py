import sys
import os
import json

# Pastikan root project masuk sys.path supaya `from src...` tidak error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.preprocessing import clean_text
from src.infer_lstm import LSTMInfer
from src.infer_transformer import TransformerInfer


# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Fake News Detection", layout="wide")

# Paths (lokal)
LSTM_MODEL_PATH = "models/lstm_best.h5"
LSTM_TOKENIZER_PATH = "models/lstm_tokenizer.joblib"
BERT_DIR = "models/bert/final"
DISTIL_DIR = "models/distilbert/final"

ASSETS_DIR = "assets"
CM_LSTM_PATH = os.path.join(ASSETS_DIR, "cm_lstm.png")
CM_BERT_PATH = os.path.join(ASSETS_DIR, "cm_bert.png")
CM_DISTIL_PATH = os.path.join(ASSETS_DIR, "cm_distilbert.png")
METRICS_PATH = os.path.join(ASSETS_DIR, "metrics.json")


# =========================================================
# Default metrics (fallback) - sesuai hasil kamu (1.00 semua)
# =========================================================
DEFAULT_METRICS = {
    "LSTM": {"accuracy": 1.00, "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 4490},
    "BERT": {"accuracy": 1.00, "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 4490},
    "DistilBERT": {"accuracy": 1.00, "precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 4490},
}

MODEL_DESC = {
    "LSTM": "Baseline model trained from scratch (non-pretrained, TensorFlow).",
    "BERT": "Pretrained Transformer (bert-base-uncased) fine-tuned for classification.",
    "DistilBERT": "Lightweight pretrained Transformer fine-tuned for faster inference.",
}


# =========================================================
# Helpers
# =========================================================
def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    return os.path.isdir(path)


@st.cache_data
def load_metrics() -> dict:
    """Load metrics from assets/metrics.json if exists, else fallback."""
    if file_exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_METRICS
    return DEFAULT_METRICS


@st.cache_resource
def load_models():
    """Load all models once. Raises FileNotFoundError with clear messages."""
    missing = []
    if not file_exists(LSTM_MODEL_PATH):
        missing.append(LSTM_MODEL_PATH)
    if not file_exists(LSTM_TOKENIZER_PATH):
        missing.append(LSTM_TOKENIZER_PATH)
    if not dir_exists(BERT_DIR):
        missing.append(BERT_DIR)
    if not dir_exists(DISTIL_DIR):
        missing.append(DISTIL_DIR)

    if missing:
        raise FileNotFoundError("Missing required model paths:\n- " + "\n- ".join(missing))

    lstm = LSTMInfer(LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH)
    bert = TransformerInfer(BERT_DIR)
    distil = TransformerInfer(DISTIL_DIR)
    return lstm, bert, distil


def render_probs_chart(probs: dict):
    """
    probs: {"REAL": float, "FAKE": float}
    """
    st.bar_chart(probs)


# =========================================================
# Sidebar Navigation
# =========================================================
st.sidebar.title("📌 Menu")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Predict", "How to Use"])
st.sidebar.divider()

metrics = load_metrics()

# =========================================================
# PAGE: Dashboard
# =========================================================
if page == "Dashboard":
    st.title("📊 Dashboard — Model Comparison & Evaluation")

    st.markdown(
        "Halaman ini menampilkan ringkasan performa model dan visualisasi evaluasi. "
        "Nilai metrik diambil dari `assets/metrics.json` (jika ada), jika tidak ada akan menggunakan nilai default."
    )

    # Summary metrics table
    st.subheader("📈 Summary Metrics (Test Set)")
    rows = []
    for k in ["LSTM", "BERT", "DistilBERT"]:
        m = metrics.get(k, {})
        rows.append(
            {
                "Model": k,
                "Accuracy": m.get("accuracy", "-"),
                "Precision": m.get("precision", "-"),
                "Recall": m.get("recall", "-"),
                "F1": m.get("f1", "-"),
                "Support": m.get("support", "-"),
            }
        )
    st.dataframe(rows, use_container_width=True)

    # Optional: show confusion matrix images if available
    st.subheader("🧩 Confusion Matrix (Optional Images)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**LSTM**")
        if file_exists(CM_LSTM_PATH):
            st.image(CM_LSTM_PATH, use_container_width=True)
        else:
            st.info("Jika ada, simpan gambar confusion matrix LSTM ke `assets/cm_lstm.png`")

    with c2:
        st.markdown("**BERT**")
        if file_exists(CM_BERT_PATH):
            st.image(CM_BERT_PATH, use_container_width=True)
        else:
            st.info("Jika ada, simpan gambar confusion matrix BERT ke `assets/cm_bert.png`")

    with c3:
        st.markdown("**DistilBERT**")
        if file_exists(CM_DISTIL_PATH):
            st.image(CM_DISTIL_PATH, use_container_width=True)
        else:
            st.info("Jika ada, simpan gambar confusion matrix DistilBERT ke `assets/cm_distilbert.png`")

    st.divider()
    st.caption("Dataset test support: Real=2142, Fake=2348, Total=4490 (sesuai hasil classification report kamu).")

# =========================================================
# PAGE: Predict
# =========================================================
elif page == "Predict":
    st.title("🧠 Predict — Fake News Detection")

    # Sidebar model settings
    st.sidebar.subheader("⚙️ Model Settings")
    model_choice = st.sidebar.radio("Choose Model", ["LSTM", "BERT", "DistilBERT"], index=0)
    st.sidebar.caption(MODEL_DESC[model_choice])

    st.sidebar.divider()
    do_clean = st.sidebar.checkbox("Apply text cleaning (recommended)", value=True)
    threshold = st.sidebar.slider("LSTM threshold (Fake if prob ≥ threshold)", 0.1, 0.9, 0.5, 0.05)

    # Load models (cached)
    try:
        lstm_model, bert_model, distil_model = load_models()
    except Exception as e:
        st.error("Model files are not ready.")
        st.code(str(e))
        st.info(
            "Pastikan kamu sudah menaruh file model di folder `models/`:\n"
            "- models/lstm_best.h5\n"
            "- models/lstm_tokenizer.joblib\n"
            "- models/bert/final/\n"
            "- models/distilbert/final/"
        )
        st.stop()

    # Show device for transformer models
    if model_choice == "BERT":
        st.sidebar.write("Device:", getattr(bert_model, "device", "unknown"))
    elif model_choice == "DistilBERT":
        st.sidebar.write("Device:", getattr(distil_model, "device", "unknown"))
    else:
        st.sidebar.write("Device:", "CPU (TensorFlow)")

    user_text = st.text_area(
        "Input news text (title + content recommended):",
        height=240,
        placeholder="Paste headline + article content here..."
    )

    colA, colB = st.columns([1, 1])

    if st.button("🔎 Predict", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text.")
            st.stop()

        text = clean_text(user_text) if do_clean else user_text.strip()

        with st.spinner("Running inference..."):
            if model_choice == "LSTM":
                pred, prob_fake = lstm_model.predict(text)
                pred = 1 if float(prob_fake) >= float(threshold) else 0

                label = "FAKE" if pred == 1 else "REAL"
                probs = {"REAL": float(1 - prob_fake), "FAKE": float(prob_fake)}
                confidence = probs[label]

            elif model_choice == "BERT":
                pred, probs_arr, prob_pred = bert_model.predict(text)
                label = "FAKE" if pred == 1 else "REAL"
                probs = {"REAL": float(probs_arr[0]), "FAKE": float(probs_arr[1])}
                confidence = float(prob_pred)

            else:  # DistilBERT
                pred, probs_arr, prob_pred = distil_model.predict(text)
                label = "FAKE" if pred == 1 else "REAL"
                probs = {"REAL": float(probs_arr[0]), "FAKE": float(probs_arr[1])}
                confidence = float(prob_pred)

        with colA:
            st.subheader("✅ Result")
            st.metric("Model", model_choice)
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.4f}")
            st.progress(min(max(float(confidence), 0.0), 1.0))

        with colB:
            st.subheader("📉 Probability Visualization")
            render_probs_chart(probs)
            st.write(probs)

    st.divider()
    st.caption("Labels: 0 = Real, 1 = Fake • Preprocessing: lowercase, remove URL, remove non-letters.")

# =========================================================
# PAGE: How to Use
# =========================================================
else:
    st.title("📘 How to Use — Tata Cara Penggunaan")

    st.markdown(
        """
### 1) Persiapan File
Pastikan folder project memiliki struktur berikut:

**models/**
- models/lstm_best.h5
- models/lstm_tokenizer.joblib
- models/bert/final/
- models/distilbert/final/

**assets/** *(opsional untuk Dashboard)*
- assets/metrics.json *(jika tidak ada, app memakai default metrics)*
- assets/cm_lstm.png
- assets/cm_bert.png
- assets/cm_distilbert.png

---

### 2) Menjalankan Aplikasi
Jalankan dari **root project**:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
"""
    )